// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3.8-Flash-Next on the direct-CUDA engine (<see cref="Q4eCudaEngine"/>): reads the GGUF's
    /// hyper-parameters, hands the engine its weights as shard offsets its loader streams into
    /// VRAM, and serves the PLE rows from the mapped n-gram table (hash and dequantization on the
    /// host, as the native token span does).
    /// </summary>
    internal sealed unsafe class Qwen4ExpCudaExecutor : IQwen4ExpPleSource, IDisposable
    {
        private const string Tag = "q4e-cuda";
        private const string A = Qwen4ExpModel.ArchitectureId;
        private GgufShardSet _gguf;
        private Q4eCudaEngine _engine;

        // ---- the PLE n-gram hash and its table ----
        private int _pleNgram, _pleHeadsPerNgram, _pleHeads, _pleHeadDim, _pleEos;
        private ulong[] _pleMultipliers, _pleOffsets, _pleVocab;
        private IntPtr _pleTable;
        // The table copied into process memory (Strata's --ple-io ram), or zero while rows come from the mapping.
        private IntPtr _pleRam;
        private readonly System.Collections.Generic.List<IntPtr> _requant = new();

        /// <summary>A hyper-connection matrix: as stored, or under TS_Q4E_HC_Q8=1 a BF16 one as Q8_0 (Strata's read),
        /// converted from the file into memory the engine uploads from while it loads.</summary>
        private CudaWeightDesc HcWeight(string name)
        {
            CudaWeightDesc w = _gguf.Weight(name);
            if (!Qwen4ExpModel.HcQ8 || w.GgmlType != WeightRequant.GgmlTypeBf16 || w.Ne0 % 32 != 0 || w.Ne2 > 1)
                return w;
            long rows = w.Ne1, cols = w.Ne0, srcBytes = rows * cols * 2, bytes = WeightRequant.Q8_0Bytes(rows, cols);
            IntPtr tmp = IntPtr.Zero, src = w.HostPtr;
            if (src == IntPtr.Zero)
            {
                tmp = (IntPtr)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)srcBytes);
                w.Source.Read(w.SourceOffset, tmp, srcBytes);
                src = tmp;
            }
            IntPtr dst = (IntPtr)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)bytes, 64);
            _requant.Add(dst);
            try
            {
                WeightRequant.Bf16ToQ8_0((ushort*)src, (byte*)dst, rows, cols);
            }
            finally
            {
                if (tmp != IntPtr.Zero)
                    System.Runtime.InteropServices.NativeMemory.Free((void*)tmp);
            }
            return new CudaWeightDesc
            {
                HostPtr = dst, GgmlType = WeightRequant.GgmlTypeQ8_0, Ne0 = w.Ne0, Ne1 = w.Ne1, Ne2 = 1,
                RowBytes = cols / 32 * 34, Name = name,
            };
        }

        /// <summary>
        /// The host experts' worker team, unless a count was given (--cpu-moe-threads, TS_CPU_MOE_THREADS): during a
        /// captured step the process's other threads are the stream wait and the doorbell server, both spinning, so the
        /// team takes the CPUs left after those and one more (on the 5090 VM's 13.6-CPU quota, 6 workers: 65.5/45.9/58.8
        /// tok/s code/prose/list; 10: 69.0/48.0/63.5; 16 oversubscribed: 63.4/45.1/58.0). The general default (half the
        /// CPUs) is for offloaded layers that share the machine with a server's threads.
        /// </summary>
        private static void SizeHostExpertTeam()
        {
            if (MoeCpuOffloadConfig.CpuThreads > 0 || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TS_CPU_MOE_THREADS")))
                return;
            int cpus = Environment.ProcessorCount;   // the cgroup quota, rounded up
            int threads = Math.Clamp(cpus - 3, Math.Max(1, cpus / 2), 32);
            TensorSharp.GGML.GgmlBasicOps.SetHostMoeThreads(threads);
            Console.Error.WriteLine($"[{Tag}] host experts: {threads} CPU threads (of {cpus}); --cpu-moe-threads overrides.");
        }

        private void FreeRequantized()
        {
            foreach (IntPtr p in _requant)
                System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)p);
            _requant.Clear();
        }
        private int _pleType;
        private long _pleRowBytes;
        // The table's bytes in its shard, warmed into the page cache after load.
        private (string Path, long Offset, long Bytes) _pleRange;
        private Task _pleWarm;
        private readonly CancellationTokenSource _pleWarmCancel = new CancellationTokenSource();

        /// <summary>The engine behind this executor, for tests that probe its stages.</summary>
        internal Q4eCudaEngine Engine => _engine;

        public int NPast => _engine.NPast;
        public int VocabSize => _engine.VocabSize;
        public int ContextSize => _engine.ContextSize;
        public int UBatch => _engine.UBatch;
        public int ActiveSlot => _engine.ActiveSlot;
        public int RowWidth { get; private set; }

        /// <param name="mtpPath">A shared-target MTP (NextN) GGUF to draft with, or null.</param>
        /// <param name="kvQ8">Store the attention KV cache as Q8_0 (--kv-cache-dtype q8_0) instead of F16.</param>
        public Qwen4ExpCudaExecutor(string ggufPath, int maxContext, int nUbatch, int nGpu, string mtpPath = null, bool kvQ8 = false)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            _gguf = new GgufShardSet(ggufPath, Tag, string.IsNullOrWhiteSpace(mtpPath) ? null : System.IO.Path.GetFullPath(mtpPath));
            try
            {
                string arch = _gguf.First.GetString("general.architecture") ?? string.Empty;
                if (arch != A)
                    throw new NotSupportedException($"[{Tag}] the direct-CUDA qwen4exp engine runs {A}, not '{arch}'.");
                _gguf.BeginStreaming();
                var desc = BuildModelDesc(maxContext, nUbatch);
                desc.KvQ8 = kvQ8;
                try
                {
                    _engine = new Q4eCudaEngine(desc, nGpu);
                }
                finally
                {
                    FreeRequantized();
                }
                if (_engine.HostExpertsActive)
                    SizeHostExpertTeam();
                _gguf.EndLoad();
                // Every token reads one scattered row per PLE head, and a row whose page is not cached
                // is a storage round trip (about a millisecond on a network filesystem: a cold table
                // ran decode at a quarter of its speed). Read it once in the background.
                if (_pleRange.Bytes > 0)
                    _pleWarm = PleInRam(_pleRange.Bytes)
                        ? Task.Run(() => CopyPleToRam(_pleWarmCancel.Token))
                        : MappedTableWarm.Start(new[] { _pleRange }, Tag, "the PLE n-gram table", _pleWarmCancel.Token);
                // TS_HOST_WARM_WAIT=1: serve only once the table is warm (or in RAM), as the GGML executor does.
                if (_pleWarm != null && Environment.GetEnvironmentVariable("TS_HOST_WARM_WAIT") == "1")
                    _pleWarm.Wait();
            }
            catch
            {
                Dispose();
                throw;
            }
            Console.Error.WriteLine($"[{Tag}] model ready in {sw.Elapsed.TotalSeconds:F1}s");
        }

        public void Forward(int[] tokens, float[] logitsOut) => _engine.Forward(tokens, logitsOut);
        public void Forward(int[] tokens, float[] logitsOut, float[] hOut) => _engine.Forward(tokens, logitsOut, hOut);
        public int HiddenWidth => _engine.HiddenWidth;
        public bool HasMtp => _engine.HasMtp;
        public void MtpForward(int[] tokens, float[] hPrev, int p0, float[] logitsOut, float[] hOut)
            => _engine.MtpForward(tokens, hPrev, p0, logitsOut, hOut);
        public void ForwardVerify(int[] tokens, float[] logitsAll, float[] hAll) => _engine.ForwardVerify(tokens, logitsAll, hAll);
        public bool CommitVerify(int keep) => _engine.CommitVerify(keep);
        public bool RestoreVerify() => _engine.RestoreVerify();
        public void Reset() => _engine.Reset();
        public bool ResetChecked() => _engine.ResetChecked();
        public bool Rewind(int nPast) => _engine.Rewind(nPast);

        // ---- sequence slots ----
        public int SlotAlloc() => _engine.SlotAlloc();
        public bool SetActiveSlot(int slot) => _engine.SetActiveSlot(slot);
        public bool SlotFree(int slot) => _engine.SlotFree(slot);
        public bool SlotHead(int slot, out int head) => _engine.SlotHead(slot, out head);
        public bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits)
            => _engine.ForwardBatchedDecode(slots, tokens, positions, logits);

        private Q4eCudaEngine.ModelDesc BuildModelDesc(int nCtx, int nUbatch)
        {
            GgufFile g = _gguf.First;
            // A file that counts NextN blocks in block_count keeps them out of the trunk.
            int nLayer = (int)g.GetUint32($"{A}.block_count") - (int)g.GetUint32($"{A}.nextn_predict_layers", 0);
            int ctxTrain = (int)g.GetUint32($"{A}.context_length", 0);
            int headDim = (int)g.GetUint32($"{A}.attention.key_length");
            if ((int)g.GetUint32($"{A}.attention.value_length", (uint)headDim) != headDim)
                throw new NotSupportedException($"[{Tag}] attention value width differs from the key width");
            if ((int)g.GetUint32($"{A}.ssm.state_size") != Q4eCudaEngine.GdnStateSize)
                throw new NotSupportedException($"[{Tag}] Gated DeltaNet state size {g.GetUint32($"{A}.ssm.state_size")} (the kernels take {Q4eCudaEngine.GdnStateSize})");
            float attnScale = g.GetFloat32($"{A}.attention.scale", 0f);
            if (attnScale == 0f)
                attnScale = 1.0f / MathF.Sqrt(headDim);

            var m = new Q4eCudaEngine.ModelDesc
            {
                NLayer = nLayer,
                NEmbd = (int)g.GetUint32($"{A}.embedding_length"),
                Hc = (int)g.GetUint32($"{A}.hyper_connection.count"),
                HcLowRank = (int)g.GetUint32($"{A}.hyper_connection.low_rank"),
                NHead = (int)g.GetUint32($"{A}.attention.head_count"),
                NKvHead = (int)g.GetUint32($"{A}.attention.head_count_kv"),
                HeadDim = headDim,
                NRot = (int)g.GetUint32($"{A}.rope.dimension_count", (uint)headDim),
                RopeBase = g.GetFloat32($"{A}.rope.freq_base", 10000f),
                RopeFreqScale = 1.0f / g.GetFloat32($"{A}.rope.scaling.factor", 1f),
                AttnScale = attnScale,
                GdnKHeads = (int)g.GetUint32($"{A}.ssm.group_count"),
                GdnVHeads = (int)g.GetUint32($"{A}.ssm.time_step_rank"),
                DConv = (int)g.GetUint32($"{A}.ssm.conv_kernel"),
                IdxHeads = (int)g.GetUint32($"{A}.attention.indexer.head_count", 0),
                IdxDim = (int)g.GetUint32($"{A}.attention.indexer.key_length", 0),
                IdxTopK = (int)g.GetUint32($"{A}.attention.indexer.top_k", 0),
                NExpert = (int)g.GetUint32($"{A}.expert_count"),
                NExpertUsed = (int)g.GetUint32($"{A}.expert_used_count"),
                NFfExp = (int)g.GetUint32($"{A}.expert_feed_forward_length"),
                NFfShexp = (int)g.GetUint32($"{A}.expert_shared_feed_forward_length",
                    g.GetUint32($"{A}.expert_feed_forward_length")),
                RmsEps = g.GetFloat32($"{A}.attention.layer_norm_rms_epsilon", 1e-6f),
                NCtx = ctxTrain > 0 ? Math.Min(nCtx, ctxTrain) : nCtx,
                NUbatch = nUbatch,
                TokEmbd = _gguf.Weight("token_embd.weight"),
                OutputHcNorm = _gguf.Floats("output_hc_norm.weight"),
                OutputHcDown = HcWeight("output_hc_down.weight"),
                OutputHcUp = HcWeight("output_hc_up.weight"),
                Layers = new Q4eCudaEngine.LayerDesc[nLayer],
            };
            // An untied head, or the embedding read back.
            m.Output = _gguf.Has("output.weight") ? _gguf.Weight("output.weight") : m.TokEmbd;
            m.NVocab = m.Output.Ne1;

            // Layer typing: GDN everywhere except every full_attention_interval-th layer, unless
            // the file lists the recurrent layers.
            var recurrent = new bool[nLayer];
            uint[] listed = g.GetUint32Array($"{A}.attention.recurrent_layers");
            int interval = (int)g.GetUint32($"{A}.full_attention_interval", 4);
            for (int il = 0; il < nLayer; il++)
                recurrent[il] = listed != null && listed.Length >= nLayer ? listed[il] != 0 : (il + 1) % Math.Max(1, interval) != 0;
            int[] ratios = g.GetInt32Array($"{A}.attention.compress_ratios") ?? Array.Empty<int>();
            var pleLayers = g.GetInt32Array($"{A}.ple.layers") ?? Array.Empty<int>();
            if (pleLayers.Length > 0)
            {
                ReadPle(g, m.NEmbd);
                m.PleConvKernel = (int)g.GetUint32($"{A}.ple.conv_kernel");
                m.PleDilation = _pleNgram;
                m.Ple = this;
            }

            for (int il = 0; il < nLayer; il++)
            {
                bool ple = Array.IndexOf(pleLayers, il) >= 0;
                m.Layers[il] = BuildLayer($"blk.{il}.", recurrent[il], ple,
                    !recurrent[il] && il < ratios.Length && m.IdxHeads > 0 ? ratios[il] : 0, m.PleConvKernel);
            }
            if (_gguf.Extra != null)
                m.Mtp = BuildMtp(m.NEmbd * m.Hc);
            m.ExpertRouteCounts = LoadRouteCounts(nLayer, m.NExpert);
            // TS_Q4E_EXPERT_VRAM_GB: VRAM for expert slots, forcing residency on one GPU (0 keeps every
            // expert in pinned host memory); unset plans it from free memory when the experts do not fit.
            string vram = Environment.GetEnvironmentVariable("TS_Q4E_EXPERT_VRAM_GB");
            if (!string.IsNullOrWhiteSpace(vram))
            {
                if (!double.TryParse(vram, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture,
                        out double gb) || gb < 0)
                    throw new ArgumentException($"TS_Q4E_EXPERT_VRAM_GB must be a non-negative number of GB, not '{vram}'.");
                m.ExpertVramBytes = (long)(gb * 1e9);
            }
            // Under residency a decode step computes its missed experts with the host's expert kernel
            // (ggml's CPU dot products) while the GPU computes the rest; TS_Q4E_HOST_EXPERTS=0 stages them.
            m.HostExperts = TensorSharp.GGML.GgmlBasicOps.HostExpertRows;
            return m;
        }

        private Q4eCudaEngine.LayerDesc BuildLayer(string p, bool recurrent, bool ple, int ratio, int pleKernel)
        {
            var L = new Q4eCudaEngine.LayerDesc
            {
                Recurrent = recurrent,
                Ple = ple,
                CompressRatio = ratio,
                HcAttnNorm = _gguf.Floats(p + "hc_attn_norm.weight"),
                HcAttnDown = HcWeight(p + "hc_attn_down.weight"),
                HcAttnUp = HcWeight(p + "hc_attn_up.weight"),
                HcAttnInject = _gguf.Floats(p + "hc_attn_inject.weight"),
                HcFfnNorm = _gguf.Floats(p + "hc_ffn_norm.weight"),
                HcFfnDown = HcWeight(p + "hc_ffn_down.weight"),
                HcFfnUp = HcWeight(p + "hc_ffn_up.weight"),
                HcFfnInject = _gguf.Floats(p + "hc_ffn_inject.weight"),
                Router = _gguf.Floats(p + "ffn_gate_inp.weight"),
                ShexpGate = _gguf.Floats(p + "ffn_gate_inp_shexp.weight"),
                GateShexp = _gguf.Weight(p + "ffn_gate_shexp.weight"),
                UpShexp = _gguf.Weight(p + "ffn_up_shexp.weight"),
                DownShexp = _gguf.Weight(p + "ffn_down_shexp.weight"),
                DownExps = _gguf.Weight(p + "ffn_down_exps.weight"),
            };
            // A BF16 router (ISTA's file) is also kept as stored: decode reads it at half the bytes, the same sums.
            var (routerPtr, routerInfo) = _gguf.Raw(p + "ffn_gate_inp.weight");
            if (routerInfo.Type == GgmlTensorType.BF16 && routerPtr != IntPtr.Zero)
            {
                var bf = new ushort[routerInfo.NumElements];
                fixed (ushort* d = bf)
                    Buffer.MemoryCopy((void*)routerPtr, d, bf.LongLength * 2, bf.LongLength * 2);
                L.RouterBf16 = bf;
            }
            if (!_gguf.Has(p + "ffn_gate_exps.weight") && _gguf.Has(p + "ffn_gate_up_exps.weight"))
                throw new NotSupportedException(
                    $"[{Tag}] this GGUF stacks gate and up into {p}ffn_gate_up_exps; the engine takes the separate pair.");
            L.GateExps = _gguf.Weight(p + "ffn_gate_exps.weight");
            L.UpExps = _gguf.Weight(p + "ffn_up_exps.weight");

            if (L.Recurrent)
            {
                L.Qkv = _gguf.Weight(p + "attn_qkv.weight");
                L.Gate = _gguf.Weight(p + "attn_gate.weight");
                L.SsmOut = _gguf.Weight(p + "ssm_out.weight");
                L.SsmAlpha = _gguf.Floats(p + "ssm_alpha.weight");
                L.SsmBeta = _gguf.Floats(p + "ssm_beta.weight");
                L.ConvW = _gguf.Floats(p + "ssm_conv1d.weight");
                L.DtBias = _gguf.Floats(p + "ssm_dt.bias");
                L.SsmA = _gguf.Floats(p + "ssm_a");
                L.SsmNorm = _gguf.Floats(p + "ssm_norm.weight");
            }
            else
            {
                L.Wq = _gguf.Weight(p + "attn_q.weight");
                L.Wk = _gguf.Weight(p + "attn_k.weight");
                L.Wv = _gguf.Weight(p + "attn_v.weight");
                L.Wo = _gguf.Weight(p + "attn_output.weight");
                L.QNorm = _gguf.Floats(p + "attn_q_norm.weight");
                L.KNorm = _gguf.Floats(p + "attn_k_norm.weight");
                if (L.CompressRatio > 0)
                {
                    L.IdxK = _gguf.Floats(p + "indexer.k_proj.weight");
                    L.IdxQ = _gguf.Floats(p + "indexer.q_proj.weight");
                    L.IdxKNorm = _gguf.Floats(p + "indexer.k_norm.weight");
                    L.IdxQNorm = _gguf.Floats(p + "indexer.q_norm.weight");
                }
            }

            if (ple)
            {
                L.PleKey = _gguf.Weight(p + "ple_key.weight");
                L.PleValue = _gguf.Weight(p + "ple_value.weight");
                L.PleNormKey = _gguf.Floats(p + "ple_norm_key.weight");
                L.PleNormQuery = _gguf.Floats(p + "ple_norm_query.weight");
                L.PleNormConv = _gguf.Floats(p + "ple_norm_conv.weight");
                // [hc * E, kern] with the taps fastest; the kernel wants them tap-major.
                float[] conv = _gguf.Floats(p + "ple_conv1d.weight");
                int kern = pleKernel, ch = conv.Length / kern;
                var t = new float[conv.Length];
                for (int c = 0; c < ch; c++)
                    for (int k = 0; k < kern; k++)
                        t[(long)k * ch + c] = conv[(long)c * kern + k];
                L.PleConvT = t;
            }
            return L;
        }

        /// <summary>A stacked expert weight requantized on the host (WeightRequant.Requantize), freed with the other
        /// load-time copies once the engine has uploaded it; the weight as it was when the type is already this one or
        /// cannot be produced.</summary>
        private unsafe CudaWeightDesc RequantExperts(CudaWeightDesc w, int dstType)
        {
            if (w.GgmlType == dstType)
                return w;
            long rows = (long)w.Ne1 * Math.Max(1, w.Ne2), srcRow = w.RowBytes;
            IntPtr host = w.HostPtr;
            var source = w.Source;
            long offset = w.SourceOffset;
            IntPtr q = WeightRequant.Requantize((r0, n, buf) =>
            {
                if (host != IntPtr.Zero)
                    Buffer.MemoryCopy((void*)(host + (nint)(r0 * srcRow)), (void*)buf, n * srcRow, n * srcRow);
                else
                    source.Read(offset + r0 * srcRow, buf, n * srcRow);
            }, w.GgmlType, w.Ne0, rows, dstType, out long bytes);
            if (q == IntPtr.Zero)
            {
                Console.Error.WriteLine($"[{Tag}] {w.Name}: ggml cannot quantize to {((GgmlTensorType)dstType).ToString()} here; kept as stored");
                return w;
            }
            _requant.Add(q);
            return new CudaWeightDesc
            {
                HostPtr = q, GgmlType = dstType, Ne0 = w.Ne0, Ne1 = w.Ne1, Ne2 = w.Ne2,
                RowBytes = bytes / rows, Name = w.Name,
            };
        }

        /// <summary>The shared-target NextN block the GGUF set carries beside the checkpoint (validated against it).</summary>
        private Q4eCudaEngine.MtpDesc BuildMtp(int hcDim)
        {
            int layer = Qwen4ExpModel.ValidateMtpHead(_gguf.First, _gguf.Extra);
            string p = $"blk.{layer}.";
            var mtp = new Q4eCudaEngine.MtpDesc
            {
                // Dense attention: the published head declares no compression for its block (validated above), and
                // its indexer tensors are inactive.
                Layer = BuildLayer(p, recurrent: false, ple: false, ratio: 0, pleKernel: 0),
                Enorm = _gguf.Floats(p + "nextn.enorm.weight"),
                Hnorm = _gguf.Floats(p + "nextn.hnorm.weight"),
                EhProj = _gguf.Weight(p + "nextn.eh_proj.weight"),
                HeadHcNorm = _gguf.Floats(p + "nextn.hc_head_norm.weight"),
                HeadHcDown = HcWeight(p + "nextn.hc_head_down.weight"),
                HeadHcUp = HcWeight(p + "nextn.hc_head_up.weight"),
            };
            if (mtp.Hnorm.Length != hcDim)
                throw new NotSupportedException($"[{Tag}] the MTP block's hnorm is {mtp.Hnorm.Length} wide, the trunk's streams {hcDim}");
            int expertType = WeightRequant.MtpExpertType();
            if (expertType == WeightRequant.GgmlTypeQ4_0)
                throw new NotSupportedException(
                    $"[{Tag}] TS_Q4E_MTP_EXPERTS=q4_0: the direct-CUDA expert kernels do not decode Q4_0 experts; use q2_0 (Strata's) or q8_0 on --backend cuda.");
            if (expertType >= 0)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                long before = mtp.Layer.GateExps.TotalBytes + mtp.Layer.UpExps.TotalBytes + mtp.Layer.DownExps.TotalBytes;
                mtp.Layer.GateExps = RequantExperts(mtp.Layer.GateExps, expertType);
                mtp.Layer.UpExps = RequantExperts(mtp.Layer.UpExps, expertType);
                mtp.Layer.DownExps = RequantExperts(mtp.Layer.DownExps, expertType);
                long after = mtp.Layer.GateExps.TotalBytes + mtp.Layer.UpExps.TotalBytes + mtp.Layer.DownExps.TotalBytes;
                Console.Error.WriteLine($"[{Tag}] MTP experts as {((GgmlTensorType)expertType).ToString()}: "
                    + $"{before / 1048576.0:F0} -> {after / 1048576.0:F0} MiB in {sw.Elapsed.TotalSeconds:F1}s (TS_Q4E_MTP_EXPERTS)");
            }
            Console.Error.WriteLine($"[{Tag}] MTP draft block {layer} from {System.IO.Path.GetFileName(_gguf.Paths[^1])}");
            return mtp;
        }

        /// <summary>The routing profile the ggml executor records for these files, as per-layer counts: it
        /// ranks the experts that get VRAM slots when the GPU cannot hold them all. Null without one.</summary>
        private ulong[][] LoadRouteCounts(int layers, int experts)
        {
            string identity = MoeRoutingProfile.IdentityFor(_gguf.Paths, A, layers, experts);
            var profile = MoeRoutingProfile.TryLoad(MoeRoutingProfile.PathFor(identity), identity, layers, experts);
            if (profile == null)
                return null;
            var counts = new ulong[layers][];
            for (int l = 0; l < layers; l++)
                counts[l] = profile.Counts(l).ToArray();
            return counts;
        }

        private void ReadPle(GgufFile g, int hidden)
        {
            _pleNgram = (int)g.GetUint32($"{A}.ple.ngram_size");
            _pleHeadsPerNgram = (int)g.GetUint32($"{A}.ple.heads_per_ngram");
            _pleEos = (int)g.GetUint32($"{A}.ple.eos_token_id");
            _pleHeadDim = (int)g.GetUint32($"{A}.embedding_length_per_layer_input");
            _pleHeads = (_pleNgram - 1) * _pleHeadsPerNgram;
            _pleMultipliers = g.GetUint64Array($"{A}.ple.layer_multipliers");
            _pleOffsets = g.GetUint64Array($"{A}.ple.head_offsets");
            _pleVocab = g.GetUint64Array($"{A}.ple.head_vocab_sizes");
            if (_pleNgram < 2 || _pleHeads * _pleHeadDim != hidden
                || _pleMultipliers == null || _pleMultipliers.Length < _pleNgram
                || _pleOffsets == null || _pleOffsets.Length < _pleHeads
                || _pleVocab == null || _pleVocab.Length < _pleHeads)
                throw new NotSupportedException($"[{Tag}] the PLE n-gram hash constants are missing or inconsistent.");
            RowWidth = hidden;
            // The table stays a host mapping: every use reads one scattered row per head.
            var (path, offset, bytes, mapped) = _gguf.MapTensor("per_layer_token_embd.weight");
            _pleRange = (path, offset, bytes);
            var info = _gguf.InfoOf("per_layer_token_embd.weight");
            if ((int)info.Shape[0] != _pleHeadDim)
                throw new NotSupportedException($"[{Tag}] PLE table rows are {info.Shape[0]} wide, not {_pleHeadDim}.");
            _pleTable = mapped;
            _pleType = (int)info.Type;
            _pleRowBytes = ManagedQuantizedOps.RowSize(_pleType, _pleHeadDim);
        }

        /// <summary>
        /// Whether to keep the PLE n-gram table in process memory (Strata's <c>--ple-io ram</c>): every token
        /// reads one scattered row per head, and from the mapping a row whose page is not cached is a storage
        /// round trip - on a network filesystem the decode thread waited on those reads with the GPU idle
        /// (Flash Next UD-Q2_K_XL, 26.8 GB table on geesefs). TS_Q4E_PLE_RAM=1 forces it, 0 keeps the mapping;
        /// unset, it is used when the system has the table plus 16 GB available (Strata's rule).
        /// </summary>
        private static bool PleInRam(long bytes)
        {
            string forced = Environment.GetEnvironmentVariable("TS_Q4E_PLE_RAM");
            if (forced == "1") return true;
            if (forced == "0" || !OperatingSystem.IsLinux()) return false;
            // What the process can still take: MemAvailable capped by its memory cgroup (a container's
            // limit counts its tmpfs files and the engine's pinned host experts, which MemAvailable does
            // not see), with room left for the rest of the process.
            return MappedTableWarm.HostMemoryAvailable() >= bytes + (6L << 30);
        }

        /// <summary>Reads the table into process memory in parallel, then serves rows from there.</summary>
        private void CopyPleToRam(CancellationToken cancel)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            long bytes = _pleRange.Bytes;
            IntPtr ram = (IntPtr)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)bytes, 4096);
            try
            {
                using var handle = System.IO.File.OpenHandle(_pleRange.Path, System.IO.FileMode.Open, System.IO.FileAccess.Read,
                    System.IO.FileShare.Read, System.IO.FileOptions.RandomAccess);
                const long Chunk = 64L << 20;
                long chunks = (bytes + Chunk - 1) / Chunk;
                Parallel.For(0L, chunks, new ParallelOptions { MaxDegreeOfParallelism = 16, CancellationToken = cancel }, c =>
                {
                    long off = c * Chunk, len = Math.Min(Chunk, bytes - off);
                    while (len > 0)
                    {
                        int n = System.IO.RandomAccess.Read(handle, new Span<byte>((byte*)ram + off, (int)Math.Min(len, int.MaxValue)),
                            _pleRange.Offset + off);
                        if (n <= 0) throw new System.IO.EndOfStreamException("PLE table shorter than its tensor.");
                        off += n;
                        len -= n;
                    }
                });
            }
            catch
            {
                System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)ram);
                if (cancel.IsCancellationRequested) return;
                throw;
            }
            _pleRam = ram;
            Volatile.Write(ref _pleTable, ram);
            Console.Error.WriteLine($"[{Tag}] the PLE n-gram table ({bytes / 1e9:F1} GB) is in RAM, read in " +
                $"{sw.Elapsed.TotalSeconds:F0} s; its rows no longer touch the model files (TS_Q4E_PLE_RAM=0 keeps the mapping).");
        }

        /// <summary>
        /// The n-gram hash of the native token span (Qwen4ExpModel.ComputePleRows), over the
        /// sequence's token history by absolute position: for each n in 2..ngram, the XOR of the
        /// context tokens times their multipliers, reduced per head, where an EOS in the window
        /// hides everything before it and positions before 0 read as EOS. Then the rows, dequantized.
        /// </summary>
        public void GatherPleRows(Qwen4ExpPleHistory[] histories, ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, float* dst)
        {
            var ctx = new long[_pleNgram];
            for (int t = 0; t < tokens.Length; t++)
            {
                var hist = histories[t];
                int pos = positions[t];
                ctx[0] = tokens[t];
                bool cut = false;
                for (int s = 1; s < _pleNgram; s++)
                {
                    int j = pos - s;
                    long tok = cut || j < 0 || j >= hist.Length ? _pleEos : hist.Tokens[j];
                    ctx[s] = tok;
                    if (tok == _pleEos)
                        cut = true;
                }
                float* rowDst = dst + (long)t * RowWidth;
                for (int n = 2; n <= _pleNgram; n++)
                {
                    ulong mixed = (ulong)ctx[0] * _pleMultipliers[0];
                    for (int j = 1; j < n; j++)
                        mixed ^= (ulong)ctx[j] * _pleMultipliers[j];
                    int baseHead = (n - 2) * _pleHeadsPerNgram;
                    for (int hh = 0; hh < _pleHeadsPerNgram; hh++)
                    {
                        int h = baseHead + hh;
                        long row = (long)(mixed % _pleVocab[h] + _pleOffsets[h]);
                        ManagedQuantizedOps.DequantizeRowToFloat32(_pleType, (IntPtr)((byte*)Volatile.Read(ref _pleTable) + row * _pleRowBytes),
                            rowDst + (long)h * _pleHeadDim, _pleHeadDim);
                    }
                }
            }
        }

        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _pleWarmCancel.Cancel();
            try { _pleWarm?.Wait(); }
            catch (AggregateException) { }
            _pleWarmCancel.Dispose();
            _engine?.Dispose();
            if (_pleRam != IntPtr.Zero)
            {
                System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)_pleRam);
                _pleRam = IntPtr.Zero;
            }
            _engine = null;
            _gguf?.Dispose();
            _gguf = null;
        }
    }
}
