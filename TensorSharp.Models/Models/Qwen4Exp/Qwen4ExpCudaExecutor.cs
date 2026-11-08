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

        public Qwen4ExpCudaExecutor(string ggufPath, int maxContext, int nUbatch, int nGpu)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            _gguf = new GgufShardSet(ggufPath, Tag);
            try
            {
                string arch = _gguf.First.GetString("general.architecture") ?? string.Empty;
                if (arch != A)
                    throw new NotSupportedException($"[{Tag}] the direct-CUDA qwen4exp engine runs {A}, not '{arch}'.");
                _gguf.BeginStreaming();
                var desc = BuildModelDesc(maxContext, nUbatch);
                _engine = new Q4eCudaEngine(desc, nGpu);
                _gguf.EndLoad();
                // Every token reads one scattered row per PLE head, and a row whose page is not cached
                // is a storage round trip (about a millisecond on a network filesystem: a cold table
                // ran decode at a quarter of its speed). Read it once in the background.
                if (_pleRange.Bytes > 0)
                    _pleWarm = MappedTableWarm.Start(new[] { _pleRange }, Tag, "the PLE n-gram table", _pleWarmCancel.Token);
            }
            catch
            {
                Dispose();
                throw;
            }
            Console.Error.WriteLine($"[{Tag}] model ready in {sw.Elapsed.TotalSeconds:F1}s");
        }

        public void Forward(int[] tokens, float[] logitsOut) => _engine.Forward(tokens, logitsOut);
        public void Reset() => _engine.Reset();
        public bool ResetChecked() => _engine.ResetChecked();
        public bool Rewind(int nPast) => _engine.Rewind(nPast);
        /// <summary>Whether slot <paramref name="slot"/>, holding <paramref name="head"/> tokens, can
        /// rewind to <paramref name="target"/> (its head, 0, or from its checkpoint up). Side-effect free.</summary>
        public bool CanRewind(int slot, int head, int target) => _engine.CanRewind(slot, head, target);
        /// <summary>What a slot that cannot grow calls: free one retained slot, true when it did.</summary>
        public Func<bool> ReclaimRetainedSlot { set => _engine.ReclaimRetainedSlot = value; }
        /// <summary>The engine's serving counters, from any thread.</summary>
        public TensorSharp.Cuda.Q4eCudaEngine.TelemetryCounters ReadTelemetry() => _engine.ReadTelemetry();
        /// <summary>Receives sampled forwards' stage times (TS_Q4E_PERF_SAMPLE).</summary>
        public Action<string, double, int, bool> StageObserver { set => _engine.StageObserver = value; }

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
            int nLayer = (int)g.GetUint32($"{A}.block_count");
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
                OutputHcDown = _gguf.Weight("output_hc_down.weight"),
                OutputHcUp = _gguf.Weight("output_hc_up.weight"),
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
                string p = $"blk.{il}.";
                bool ple = Array.IndexOf(pleLayers, il) >= 0;
                var L = new Q4eCudaEngine.LayerDesc
                {
                    Recurrent = recurrent[il],
                    Ple = ple,
                    CompressRatio = !recurrent[il] && il < ratios.Length && m.IdxHeads > 0 ? ratios[il] : 0,
                    HcAttnNorm = _gguf.Floats(p + "hc_attn_norm.weight"),
                    HcAttnDown = _gguf.Weight(p + "hc_attn_down.weight"),
                    HcAttnUp = _gguf.Weight(p + "hc_attn_up.weight"),
                    HcAttnInject = _gguf.Floats(p + "hc_attn_inject.weight"),
                    HcFfnNorm = _gguf.Floats(p + "hc_ffn_norm.weight"),
                    HcFfnDown = _gguf.Weight(p + "hc_ffn_down.weight"),
                    HcFfnUp = _gguf.Weight(p + "hc_ffn_up.weight"),
                    HcFfnInject = _gguf.Floats(p + "hc_ffn_inject.weight"),
                    Router = _gguf.Floats(p + "ffn_gate_inp.weight"),
                    ShexpGate = _gguf.Floats(p + "ffn_gate_inp_shexp.weight"),
                    GateShexp = _gguf.Weight(p + "ffn_gate_shexp.weight"),
                    UpShexp = _gguf.Weight(p + "ffn_up_shexp.weight"),
                    DownShexp = _gguf.Weight(p + "ffn_down_shexp.weight"),
                    DownExps = _gguf.Weight(p + "ffn_down_exps.weight"),
                };
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
                    int kern = m.PleConvKernel, ch = conv.Length / kern;
                    var t = new float[conv.Length];
                    for (int c = 0; c < ch; c++)
                        for (int k = 0; k < kern; k++)
                            t[(long)k * ch + c] = conv[(long)c * kern + k];
                    L.PleConvT = t;
                }
                m.Layers[il] = L;
            }
            return m;
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
                        ManagedQuantizedOps.DequantizeRowToFloat32(_pleType, (IntPtr)((byte*)_pleTable + row * _pleRowBytes),
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
            _engine = null;
            _gguf?.Dispose();
            _gguf = null;
        }
    }
}
