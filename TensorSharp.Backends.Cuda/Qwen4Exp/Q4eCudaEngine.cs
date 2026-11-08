// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Qwen3.8-Flash-Next (qwen4exp) on the direct-CUDA backend: the whole model resident across one
// or more GPUs (contiguous runs of layers per device), driven without ggml.
//
// The residual stream is 4 hyper-connection streams wide. Per layer, both halves read it through
// a low-rank gated mixer and write back through a per-stream 2 * sigmoid scatter:
//   mixing:     Gated DeltaNet (36 layers: a causal convolution, the delta rule with a per-head
//               decay, a sigmoid-gated RMS norm) or gated GQA attention (every 4th layer) whose
//               Qwen Sparse Attention indexer keeps the best blocks of 4 cells past its width
//   FFN:        a 512-expert softmax-routed MoE (top 10) with a sigmoid-gated shared expert
// plus, ahead of one layer, the PLE block: n-gram hashed embedding rows (gathered on the host),
// a per-stream gate and a dilated depthwise convolution. The head is the output mixer and the
// LM head.
//
// The reference is the native executor ggml_cuda runs (ggml_ops_qwen4exp.cpp). The embedding,
// router and expert kernels are the DeepSeek V4 engine's (Dsv4Kernels, CudaMoe), the delta
// recurrence is GLM's (GlmKernels), and the rest is this model's (Q4eKernels).
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine : IDisposable
    {
        private const int TF32 = 0, TF16 = 1, TQ8_0 = 8;

        /// <summary>The Gated DeltaNet state size (head width) the kernels are written for.</summary>
        public const int GdnStateSize = Q4eKernels.GdnHeadDim;

        public sealed class LayerDesc
        {
            /// <summary>Gated DeltaNet; otherwise gated attention.</summary>
            public bool Recurrent;
            /// <summary>The PLE block runs ahead of this layer.</summary>
            public bool Ple;
            /// <summary>QSA block size of an attention layer; 0 is dense attention.</summary>
            public int CompressRatio;

            public float[] HcAttnNorm, HcFfnNorm;
            public CudaWeightDesc HcAttnDown, HcAttnUp, HcFfnDown, HcFfnUp;
            /// <summary>The scatter logits' weights [hc, hc * E].</summary>
            public float[] HcAttnInject, HcFfnInject;

            // ---- Gated DeltaNet ----
            public CudaWeightDesc Qkv, Gate, SsmOut;
            /// <summary>[HV, E] each.</summary>
            public float[] SsmAlpha, SsmBeta;
            /// <summary>Per channel [C, dConv]: the file's [dConv, C] as stored.</summary>
            public float[] ConvW;
            public float[] DtBias, SsmA, SsmNorm;

            // ---- attention ----
            public CudaWeightDesc Wq, Wk, Wv, Wo;
            public float[] QNorm, KNorm;
            /// <summary>The indexer projections as F32 ([D, E] and [IH * D, E]) and their norms.</summary>
            public float[] IdxK, IdxQ, IdxKNorm, IdxQNorm;

            // ---- PLE ----
            public CudaWeightDesc PleKey, PleValue;
            public float[] PleNormKey, PleNormQuery, PleNormConv;
            /// <summary>The convolution taps tap-major [kern, hc * E] (the file stores [hc * E, kern]).</summary>
            public float[] PleConvT;

            // ---- FFN ----
            /// <summary>[nExpert, E].</summary>
            public float[] Router;
            /// <summary>The router as the file's BF16 [nExpert, E], when it ships so (read at half the bytes, the same sums).</summary>
            public ushort[] RouterBf16;
            /// <summary>The shared expert's gate vector [E].</summary>
            public float[] ShexpGate;
            public CudaWeightDesc GateExps, UpExps, DownExps, GateShexp, UpShexp, DownShexp;
        }

        /// <summary>
        /// The NextN (MTP) draft block of a separate shared-target GGUF: each hyper-connection stream joins the
        /// normed token embedding and its own normed stream ([enorm(e), hnorm(h_s)], 2E wide) and eh_proj maps it
        /// back to E; one dense-attention layer and its MoE follow, with their own KV; the output mixer is the
        /// block's own, the LM head the trunk's. Reference: ggml_ops_qwen4exp_mtp.cpp.
        /// </summary>
        public sealed class MtpDesc
        {
            /// <summary>The block's layer: gated attention without QSA or PLE, and the routed MoE.</summary>
            public LayerDesc Layer;
            /// <summary>[E] for the token embedding, [hc * E] for the trunk streams.</summary>
            public float[] Enorm, Hnorm;
            /// <summary>The block's output mixer.</summary>
            public float[] HeadHcNorm;
            public CudaWeightDesc EhProj, HeadHcDown, HeadHcUp;
        }

        public sealed class ModelDesc
        {
            public int NLayer, NEmbd, NVocab, Hc, HcLowRank;
            public int NHead, NKvHead, HeadDim, NRot;
            public float RopeBase, RopeFreqScale, AttnScale;
            public int GdnKHeads, GdnVHeads, DConv;
            public int IdxHeads, IdxDim, IdxTopK;
            public int NExpert, NExpertUsed, NFfExp, NFfShexp;
            public int PleConvKernel, PleDilation;
            public float RmsEps;
            public int NCtx, NUbatch;
            /// <summary>The attention KV cache as Q8_0 (<c>--kv-cache-dtype q8_0</c>): int8 codes with an F16 scale per
            /// 32 values, about half the F16 cache's bytes, which a GPU then spends on expert slots.</summary>
            public bool KvQ8;
            public CudaWeightDesc TokEmbd, Output;
            public float[] OutputHcNorm;
            public CudaWeightDesc OutputHcDown, OutputHcUp;
            public LayerDesc[] Layers;
            /// <summary>The PLE rows' host gather; required when a layer carries the PLE block.</summary>
            public IQwen4ExpPleSource Ple;
            /// <summary>Per layer, how often each expert was routed (the routing profile), or null: ranks
            /// the experts that get VRAM slots when one GPU cannot hold them all (Residency.cs).</summary>
            public ulong[][] ExpertRouteCounts;
            /// <summary>VRAM for expert slots, forcing residency on one GPU; negative plans it from free memory.</summary>
            public long ExpertVramBytes = -1;
            /// <summary>The host's expert kernel; with it a decode step under residency computes its missed experts
            /// on the CPU (Residency.cs). Null keeps staging them.</summary>
            public CudaHostExpertRows HostExperts;
            /// <summary>The MTP draft block, or null.</summary>
            public MtpDesc Mtp;
        }

        private sealed class DevLayer
        {
            public int Device;
            public bool Recurrent, Ple;
            public int Ratio;
            public Tensor HcAttnNorm, HcFfnNorm, HcAttnInject, HcFfnInject;
            public DeviceWeight HcAttnDown, HcAttnUp, HcFfnDown, HcFfnUp;
            public DeviceWeight Qkv, Gate, SsmOut;
            public Tensor SsmAlpha, SsmBeta, ConvW, DtBias, SsmA, SsmNorm;
            public DeviceWeight Wq, Wk, Wv, Wo;
            public Tensor QNorm, KNorm, IdxK, IdxQ, IdxKNorm, IdxQNorm;
            public DeviceWeight PleKey, PleValue;
            public Tensor PleNormKey, PleNormQuery, PleNormConv, PleConvT;
            public Tensor Router, ShexpGate, RouterBf16;
            public DeviceWeight GateExps, UpExps, DownExps, GateShexp, UpShexp, DownShexp;
            /// <summary>Expert residency's staging view of this layer (invalid when every expert is resident).</summary>
            public CudaExpertStageLayer Stage;
        }

        private sealed class Dev
        {
            public int Ordinal;
            public CudaAllocator Alloc;
            public Dsv4Kernels DK;
            public GlmKernels GK;
            public Q4eKernels QK;
            public CudaWeightArena Weights;
            public List<Tensor> OwnedTensors = new List<Tensor>();
            // Layer-boundary handoff of the streams through pinned host memory: DtoH on this
            // device's stream, HtoD on the next.
            public IntPtr BoundaryPinned, XsReadyEv, CopyDoneEv;
            /// <summary>The shared expert's stream in a decode-size step, forked from and joined back into the device's
            /// stream (a captured step holds both branches).</summary>
            public IntPtr ShStream, ShForkEv, ShJoinEv;
            public Tensor Tokens;
            /// <summary>A captured decode step's position, copied in from pinned memory at launch.</summary>
            public Tensor PosDev;
            public CudaMoeScratch Moe;

            // hyper-connections
            public Tensor Xs, Xn, Lo, G, Inject, Cur, BlockOut;
            /// <summary>The decode-size mixer's per-slice partials [rows, slices, 1 + hc].</summary>
            public Tensor HcPartials;
            /// <summary>A decode-size input quantized once for its projections (QuantizeShared).</summary>
            public Tensor SharedQ81;
            public Tensor GemvScratch;
            // Gated DeltaNet
            public Tensor Qkv, Z, Alpha, Beta, Scr, Core, GOut;
            // attention
            public Tensor Qg, K, V, Q, Gate, AttnO;
            // QSA
            public Tensor IdxRaw, IdxQ, Scores, Cells, CellCnt;
            // PLE
            public Tensor PleEmb, PleKeyOut, PleValueOut, PleGated, PleNorm;
            public IntPtr PlePinned;
            // FFN
            public Tensor RouterLogits, ShGate, ShUp, ShDown;
            /// <summary>A prompt's router widened from BF16 (allocated on first use).</summary>
            public Tensor RouterF32;
            // head
            public Tensor Logits, BatchLogits;
            // A batched decode step's per-row state pointers: [layer][conv, ssm][row], then the
            // PLE histories [row], staged through pinned memory.
            public Tensor BatchStatePtrs;
            public IntPtr BatchStatePinned;

            public IntPtr Stream => Alloc.Stream.Handle;
            public void MakeCurrent() => Alloc.Context.MakeCurrent();
        }

        /// <summary>One sequence's caches.</summary>
        private sealed class Slot
        {
            public int Id;
            public int NPast;
            /// <summary>Cache bytes per device.</summary>
            public long[] DeviceBytes;
            public bool Failed;
            public SlotLayer[] Layers;
            /// <summary>Per device: [layer][conv, ssm] device pointers of this slot's GDN states, then
            /// the PLE history pointer (the kernels take arrays of per-sequence state pointers).</summary>
            public Tensor[] StatePtrs;
            /// <summary>The PLE convolution history [hist, hc * E], on the PLE layer's device.</summary>
            public Tensor PleHist;
            public readonly Qwen4ExpPleHistory PleTokens = new Qwen4ExpPleHistory();
            public readonly List<Tensor> Owned = new List<Tensor>();
            /// <summary>The MTP block's KV (allocated on the slot's first draft), and the positions it holds.</summary>
            public SlotLayer Mtp;
            public int MtpPast;
        }

        private sealed class SlotLayer
        {
            public Tensor ConvState, Ssm;                    // Gated DeltaNet
            public Tensor KCache, VCache, IdxRaw, IdxPooled; // attention, QSA
            public Tensor KScale, VScale;                    // Q8_0 KV: one F16 scale per 32 cached values
        }

        private readonly ModelDesc _m;
        private readonly Dev[] _devs;
        private readonly DevLayer[] _layers;
        private readonly int _lastDev;
        private readonly int _pleLayer = -1;
        private DeviceWeight _tokEmbd, _output, _outHcDown, _outHcUp;
        private Tensor _outHcNorm;
        private IntPtr _pinnedTokens, _pinnedLogits, _pinnedHidden;
        private Slot _active;
        private readonly int _perf;
        private readonly CudaStageTimer _stages;

        public int NPast => _active.NPast;
        public int ContextSize => _m.NCtx;
        public int UBatch => _m.NUbatch;
        public int VocabSize => _m.NVocab;
        public int HiddenSize => _m.NEmbd;

        private int HcDim => _m.Hc * _m.NEmbd;
        private int ConvDim => (2 * _m.GdnKHeads + _m.GdnVHeads) * Q4eKernels.GdnHeadDim;
        private int ValueDim => _m.GdnVHeads * Q4eKernels.GdnHeadDim;
        private int PleHistRows => (_m.PleConvKernel - 1) * _m.PleDilation;

        public Q4eCudaEngine(ModelDesc m, int nGpu)
        {
            _m = m ?? throw new ArgumentNullException(nameof(m));
            if (m.Hc != 4)
                throw new NotSupportedException($"[q4e-cuda] {m.Hc} hyper-connection streams (the kernels take 4)");
            if (m.HeadDim != Q4eKernels.AttnHeadDim)
                throw new NotSupportedException($"[q4e-cuda] attention head_dim {m.HeadDim} (the kernels take {Q4eKernels.AttnHeadDim})");
            if (m.NKvHead <= 0 || m.NHead % m.NKvHead != 0)
                throw new NotSupportedException($"[q4e-cuda] {m.NHead} query heads over {m.NKvHead} KV heads");
            if (m.NRot <= 0 || m.NRot > m.HeadDim || m.NRot % 2 != 0)
                throw new NotSupportedException($"[q4e-cuda] rotary width {m.NRot}");
            if (m.NExpertUsed > 16 || m.NExpert > 512)
                throw new NotSupportedException($"[q4e-cuda] {m.NExpert} experts, {m.NExpertUsed} per token (the router takes 512 and 16)");
            if (m.NEmbd % 256 != 0 || m.HcLowRank % 32 != 0)
                throw new NotSupportedException($"[q4e-cuda] hidden width {m.NEmbd} / hyper-connection rank {m.HcLowRank}");
            bool anyQsa = false;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = m.Layers[il];
                // Before any device is touched: refused inside the per-device upload, two devices
                // refusing at once reached the host as an AggregateException and aborted the CLI.
                CudaMoe.RequireExpertWeights("q4e-cuda", il, L.GateExps, L.UpExps, L.DownExps);
                if (L.Ple)
                {
                    if (_pleLayer >= 0)
                        throw new NotSupportedException("[q4e-cuda] PLE on more than one layer");
                    _pleLayer = il;
                }
                anyQsa |= !L.Recurrent && L.CompressRatio > 0;
            }
            if (m.Mtp != null)
            {
                var ml = m.Mtp.Layer;
                if (ml == null || ml.Recurrent || ml.Ple || ml.CompressRatio != 0 || m.Mtp.Enorm?.Length != m.NEmbd
                    || m.Mtp.Hnorm?.Length != m.Hc * m.NEmbd || m.Mtp.HeadHcNorm?.Length != m.Hc * m.NEmbd
                    || !m.Mtp.EhProj.IsValid || m.Mtp.EhProj.Ne0 != 2L * m.NEmbd || m.Mtp.EhProj.Ne1 != m.NEmbd)
                    throw new NotSupportedException("[q4e-cuda] the MTP block is not a dense-attention NextN layer of this model");
                CudaMoe.RequireExpertWeights("q4e-cuda", m.NLayer, ml.GateExps, ml.UpExps, ml.DownExps);
            }
            if (anyQsa && (m.IdxDim % 32 != 0 || m.IdxDim > Q4eKernels.MaxIndexerDim || m.IdxDim < m.NRot || m.IdxTopK <= 0))
                throw new NotSupportedException($"[q4e-cuda] indexer head width {m.IdxDim}, top_k {m.IdxTopK}");
            if (_pleLayer >= 0 && (m.Ple == null || m.Ple.RowWidth != m.NEmbd))
                throw new NotSupportedException("[q4e-cuda] the PLE block needs a row source as wide as the hidden state");
            _perf = EnvInt("TS_Q4E_PERF", 0);
            _stages = new CudaStageTimer("q4e-cuda", _perf, StageNames);

            CudaBackend.Register();
            CudaDriverApi.cuInit(0);
            CudaDriverApi.cuDeviceGetCount(out int devCount).ThrowOnError();
            int useDevs = nGpu > 0 ? Math.Min(nGpu, devCount) : devCount;
            if (useDevs < 1)
                throw new InvalidOperationException("No CUDA devices available for the qwen4exp engine.");

            _devs = new Dev[useDevs];
            for (int d = 0; d < useDevs; d++)
            {
                var dev = new Dev { Ordinal = d, Alloc = new CudaAllocator(d) };
                dev.MakeCurrent();
                dev.DK = Dsv4Kernels.Create();
                dev.GK = GlmKernels.Create();
                dev.QK = Q4eKernels.Create();
                _devs[d] = dev;
                if (m.KvQ8 && !dev.QK.HasKvQ8)
                    throw new NotSupportedException("[q4e-cuda] --kv-cache-dtype q8_0 needs the Q8_0 KV kernels, which the loaded "
                        + "tensorsharp_q4e_kernels module predates (a committed PTX baseline not yet regenerated). Rebuild "
                        + "TensorSharp.Backends.Cuda with a CUDA 13 nvcc on PATH, or run with the F16 cache.");
            }
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = m.Layers[il];
                CudaMoe.RequireExpertDecoders("q4e-cuda", _devs[0].DK.DecodesQ5_1Experts, il, L.GateExps, L.UpExps, L.DownExps,
                    decodesLowBitIq: _devs[0].DK.DecodesLowBitIqExperts, decodesLegacy: _devs[0].DK.DecodesLegacyQuants);
                // Dense Q2_0/Q4_0/Q5_0 rows and BF16 matrices (ISTA's pack) need the module's legacy decoders too.
                if (!_devs[0].DK.DecodesLegacyQuants)
                    foreach (var w in LayerWeights(L, experts: false))
                        if (w.GgmlType is 2 or 6 or 42 or 30)
                            throw new NotSupportedException(
                                $"[q4e-cuda] {w.Name ?? $"a layer {il} weight"} is {CudaMoe.GgmlTypeName(w.GgmlType)}, and the loaded tensorsharp_dsv4_kernels " +
                                "module predates its decoder (a committed PTX baseline not yet regenerated). Rebuild TensorSharp.Backends.Cuda " +
                                "with a CUDA 13 nvcc on PATH, or run this model with --backend ggml_cuda.");
            }

            if (m.Mtp != null)
            {
                var ml = m.Mtp.Layer;
                CudaMoe.RequireExpertDecoders("q4e-cuda", _devs[0].DK.DecodesQ5_1Experts, m.NLayer, ml.GateExps, ml.UpExps, ml.DownExps,
                    decodesLowBitIq: _devs[0].DK.DecodesLowBitIqExperts, decodesLegacy: _devs[0].DK.DecodesLegacyQuants);
                if (!_devs[0].DK.DecodesLegacyQuants)
                    foreach (var w in MtpWeights(m.Mtp, experts: false))
                        if (w.GgmlType is 2 or 6 or 42 or 30)
                            throw new NotSupportedException(
                                $"[q4e-cuda] the MTP block's {w.Name ?? "weight"} is {CudaMoe.GgmlTypeName(w.GgmlType)}, which the loaded " +
                                "tensorsharp_dsv4_kernels module cannot decode; rebuild TensorSharp.Backends.Cuda with a CUDA 13 nvcc on PATH.");
            }

            // ---- expert residency: one GPU that cannot hold every routed expert keeps the hottest in
            // VRAM slots and reads the rest from pinned host memory (Residency.cs) ----
            long nonExpertBytes = 0;
            for (int il = 0; il < m.NLayer; il++)
            {
                foreach (var w in LayerWeights(m.Layers[il], experts: false))
                    nonExpertBytes += CudaWeightArena.Align(w.TotalBytes);
                nonExpertBytes += LayerFloatBytes(m.Layers[il]) + LayerCacheBytes(il);
            }
            // The MTP block stays whole in VRAM (its experts included), with one sequence's draft KV and its scratch.
            nonExpertBytes += MtpDeviceBytes();
            _residency = PlanResidency(useDevs, nonExpertBytes);

            // ---- placement: contiguous runs of layers, each device filled to the same share of
            // what it has free after its fixed scratch ----
            var layerBytes = new long[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
            {
                foreach (var w in LayerWeights(m.Layers[il], experts: _residency == null))
                    layerBytes[il] += CudaWeightArena.Align(w.TotalBytes);
                layerBytes[il] += LayerFloatBytes(m.Layers[il]) + LayerCacheBytes(il);
            }
            int[] assignment = PlaceLayers(layerBytes);
            _lastDev = assignment[m.NLayer - 1];

            // ---- arenas ----
            var arenaNeed = new long[useDevs];
            for (int il = 0; il < m.NLayer; il++)
                foreach (var w in LayerWeights(m.Layers[il], experts: _residency == null))
                    arenaNeed[assignment[il]] += CudaWeightArena.Align(w.TotalBytes);
            if (_residency != null)
                arenaNeed[0] += ResidencyArenaBytes();
            arenaNeed[0] += CudaWeightArena.Align(m.TokEmbd.TotalBytes);
            arenaNeed[_lastDev] += CudaWeightArena.Align(m.Output.TotalBytes)
                + CudaWeightArena.Align(m.OutputHcDown.TotalBytes) + CudaWeightArena.Align(m.OutputHcUp.TotalBytes);
            arenaNeed[_lastDev] += MtpArenaBytes();
            for (int d = 0; d < useDevs; d++)
            {
                _devs[d].MakeCurrent();
                _devs[d].Weights = new CudaWeightArena(_devs[d].Alloc, arenaNeed[d]);
            }

            _layers = new DevLayer[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = m.Layers[il];
                _layers[il] = new DevLayer { Device = assignment[il], Recurrent = L.Recurrent, Ple = L.Ple, Ratio = L.Recurrent ? 0 : L.CompressRatio };
            }
            if (m.Mtp != null)
                _mtpLayer = new DevLayer { Device = _lastDev };

            var sw = Stopwatch.StartNew();
            if (_residency != null)
            {
                LoadHostExperts();
                CreateExpertStage(_devs[0]);
            }
            Parallel.For(0, useDevs, d =>
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                for (int il = 0; il < m.NLayer; il++)
                    if (assignment[il] == d)
                        UploadLayer(dev, il);
                if (d == 0)
                    _tokEmbd = dev.Weights.Place(m.TokEmbd);
                if (d == _lastDev)
                {
                    _output = dev.Weights.Place(m.Output);
                    _outHcDown = dev.Weights.Place(m.OutputHcDown);
                    _outHcUp = dev.Weights.Place(m.OutputHcUp);
                    _outHcNorm = UploadF32(dev, m.OutputHcNorm);
                    if (_mtpLayer != null)
                        UploadMtp(dev);
                }
            });
            CudaWeightArena.StreamAll(Array.ConvertAll(_devs, d => d.Weights), "q4e-cuda");

            foreach (var dev in _devs)
                AllocateScratch(dev);
            ReportResidency();
            StartHostExperts();
            _active = CreateSlot();
            _slots[_active.Id] = _active;
            CachePrefillWeights();

            double gib = 0;
            foreach (long b in layerBytes)
                gib += b;
            gib /= 1024.0 * 1024 * 1024;
            Console.Error.WriteLine(
                $"[q4e-cuda] {gib:F1} GiB of weights and caches across {useDevs} GPU(s) " +
                $"(layer split {string.Join("/", CountPerDev(assignment, useDevs))}), loaded in {sw.Elapsed.TotalSeconds:F1}s " +
                $"(n_ctx={m.NCtx}, ubatch={m.NUbatch})");
        }

        // -------------------------------------------------------------------
        // Placement and upload
        // -------------------------------------------------------------------

        private static IEnumerable<CudaWeightDesc> LayerWeights(LayerDesc l, bool experts = true)
        {
            foreach (var w in new[]
                {
                    l.HcAttnDown, l.HcAttnUp, l.HcFfnDown, l.HcFfnUp,
                    l.Qkv, l.Gate, l.SsmOut, l.Wq, l.Wk, l.Wv, l.Wo, l.PleKey, l.PleValue,
                    l.GateShexp, l.UpShexp, l.DownShexp,
                })
            {
                if (w.IsValid)
                    yield return w;
            }
            if (!experts)
                yield break;
            foreach (var w in new[] { l.GateExps, l.UpExps, l.DownExps })
                if (w.IsValid)
                    yield return w;
        }

        /// <summary>The F32 tensors a layer uploads beside its arena weights.</summary>
        private static long LayerFloatBytes(LayerDesc l)
        {
            // A BF16 router is uploaded as stored (UploadLayer), not widened.
            long n = l.RouterBf16 != null && l.Router.Length / Math.Max(1, l.ShexpGate?.Length ?? 1) >= 256
                ? CudaWeightArena.Align((long)l.RouterBf16.Length * 2) - CudaWeightArena.Align((long)l.Router.Length * 4) : 0;
            foreach (var a in new[]
                {
                    l.HcAttnNorm, l.HcFfnNorm, l.HcAttnInject, l.HcFfnInject, l.SsmAlpha, l.SsmBeta, l.ConvW, l.DtBias,
                    l.SsmA, l.SsmNorm, l.QNorm, l.KNorm, l.IdxK, l.IdxQ, l.IdxKNorm, l.IdxQNorm, l.PleNormKey,
                    l.PleNormQuery, l.PleNormConv, l.PleConvT, l.Router, l.ShexpGate,
                })
            {
                if (a != null)
                    n += CudaWeightArena.Align((long)a.Length * 4);
            }
            return n;
        }

        /// <summary>One attention layer's K (or V) cache of one sequence at the full context, scales included.</summary>
        private long KvCacheBytes()
        {
            long values = (long)_m.NCtx * _m.NKvHead * _m.HeadDim;
            return _m.KvQ8 ? values + values / Q4eKernels.KvQ8Block * 2 : values * 2;
        }

        /// <summary>One sequence's cache bytes for a layer at the full context.</summary>
        private long LayerCacheBytes(int il)
        {
            var m = _m;
            var L = m.Layers[il];
            long b = 0;
            if (L.Recurrent)
                b += (long)(m.DConv - 1) * ConvDim * 4 + (long)m.GdnVHeads * Q4eKernels.GdnHeadDim * Q4eKernels.GdnHeadDim * 4;
            else
            {
                b += 2 * KvCacheBytes();
                if (L.CompressRatio > 0)
                    b += (long)m.NCtx * m.IdxDim * 2 + (long)(m.NCtx / L.CompressRatio) * m.IdxDim * 4;
            }
            if (L.Ple)
                b += (long)(m.PleConvKernel - 1) * m.PleDilation * m.Hc * m.NEmbd * 4;
            return b;
        }

        /// <summary>Scratch a device holds whatever layers it carries (ubatch-sized buffers).</summary>
        private long DeviceScratchBytes()
        {
            var m = _m;
            long nt = m.NUbatch, e = m.NEmbd, hcE = (long)m.Hc * e;
            long b = 4 * nt * hcE * 4 + 4 * nt * e * 4;
            b += nt * ((long)ConvDim + 3L * ValueDim) * 4 + nt * m.GdnVHeads * (4L * Q4eKernels.GdnHeadDim + 1) * 4;
            b += nt * (long)m.NHead * m.HeadDim * 4 * 5;
            b += nt * (long)(m.NCtx / 4 + m.IdxTopK + 4) * 4;
            b += CudaMoeScratch.Bytes(m.NUbatch, m.NExpertUsed, m.NEmbd, Math.Max(m.NFfExp, 1), m.NExpert);
            b += 3 * nt * (long)m.NFfShexp * 4;
            return b + (256L << 20);
        }

        private int[] PlaceLayers(long[] layerBytes)
        {
            int nDev = _devs.Length, n = layerBytes.Length;
            var budget = new long[nDev];
            long scratch = DeviceScratchBytes();
            for (int d = 0; d < nDev; d++)
            {
                _devs[d].MakeCurrent();
                var (free, _) = _devs[d].Alloc.GetMemoryInfo();
                long b = (long)free - scratch - (1L << 30);
                if (d == 0) b -= _m.TokEmbd.TotalBytes;
                if (d == nDev - 1) b -= _m.Output.TotalBytes + (long)_m.NVocab * 8 + MtpDeviceBytes();
                budget[d] = Math.Max(0, b);
            }
            long total = 0, totalBudget = 0;
            foreach (long x in layerBytes) total += x;
            foreach (long x in budget) totalBudget += x;
            if (total > totalBudget)
                throw new InvalidOperationException(
                    $"[q4e-cuda] the model needs {total / 1073741824.0:F1} GiB of weights and caches but the " +
                    $"{nDev} GPU(s) have {totalBudget / 1073741824.0:F1} GiB free; lower the context or add devices.");

            // Fill each device to the same fraction of its budget, leaving at least one layer for
            // every device after it.
            double share = (double)total / totalBudget;
            var assignment = new int[n];
            int dev = 0;
            long used = 0;
            for (int il = 0; il < n; il++)
            {
                int layersLeft = n - il, devsAfter = nDev - 1 - dev;
                bool full = used + layerBytes[il] > share * budget[dev] * 1.0001;
                if (dev < nDev - 1 && used > 0 && (full || layersLeft <= devsAfter))
                {
                    dev++;
                    used = 0;
                }
                assignment[il] = dev;
                used += layerBytes[il];
                if (used > budget[dev])
                    throw new InvalidOperationException($"[q4e-cuda] layer {il} does not fit on device {dev}");
            }
            return assignment;
        }

        private static int[] CountPerDev(int[] assignment, int nDev)
        {
            var c = new int[nDev];
            foreach (int d in assignment)
                c[d]++;
            return c;
        }

        private void UploadLayer(Dev dev, int il) => UploadLayer(dev, _m.Layers[il], _layers[il], il);

        /// <param name="il">The trunk layer, or -1 for the MTP block (whose experts always go to the arena).</param>
        private void UploadLayer(Dev dev, LayerDesc src, DevLayer dst, int il)
        {
            dst.HcAttnNorm = UploadF32(dev, src.HcAttnNorm);
            dst.HcFfnNorm = UploadF32(dev, src.HcFfnNorm);
            dst.HcAttnInject = UploadF32(dev, src.HcAttnInject);
            dst.HcFfnInject = UploadF32(dev, src.HcFfnInject);
            dst.HcAttnDown = dev.Weights.Place(src.HcAttnDown);
            dst.HcAttnUp = dev.Weights.Place(src.HcAttnUp);
            dst.HcFfnDown = dev.Weights.Place(src.HcFfnDown);
            dst.HcFfnUp = dev.Weights.Place(src.HcFfnUp);
            if (src.Recurrent)
            {
                dst.Qkv = dev.Weights.Place(src.Qkv);
                dst.Gate = dev.Weights.Place(src.Gate);
                dst.SsmOut = dev.Weights.Place(src.SsmOut);
                dst.SsmAlpha = UploadF32(dev, src.SsmAlpha);
                dst.SsmBeta = UploadF32(dev, src.SsmBeta);
                dst.ConvW = UploadF32(dev, src.ConvW);
                dst.DtBias = UploadF32(dev, src.DtBias);
                dst.SsmA = UploadF32(dev, src.SsmA);
                dst.SsmNorm = UploadF32(dev, src.SsmNorm);
            }
            else
            {
                dst.Wq = dev.Weights.Place(src.Wq);
                dst.Wk = dev.Weights.Place(src.Wk);
                dst.Wv = dev.Weights.Place(src.Wv);
                dst.Wo = dev.Weights.Place(src.Wo);
                dst.QNorm = UploadF32(dev, src.QNorm);
                dst.KNorm = UploadF32(dev, src.KNorm);
                if (dst.Ratio > 0)
                {
                    dst.IdxK = UploadF32(dev, src.IdxK);
                    dst.IdxQ = UploadF32(dev, src.IdxQ);
                    dst.IdxKNorm = UploadF32(dev, src.IdxKNorm);
                    dst.IdxQNorm = UploadF32(dev, src.IdxQNorm);
                }
            }
            if (src.Ple)
            {
                dst.PleKey = dev.Weights.Place(src.PleKey);
                dst.PleValue = dev.Weights.Place(src.PleValue);
                dst.PleNormKey = UploadF32(dev, src.PleNormKey);
                dst.PleNormQuery = UploadF32(dev, src.PleNormQuery);
                dst.PleNormConv = UploadF32(dev, src.PleNormConv);
                dst.PleConvT = UploadF32(dev, src.PleConvT);
            }
            // The BF16 GEMV is the warp-per-output form, for 256 or more experts (Flash Next's 512).
            if (src.RouterBf16 != null && _m.NExpert >= 256 && dev.DK.HasGemvBf16 && dev.QK.HasBf16ToF32 && src.RouterBf16.Length == src.Router?.Length
                && Environment.GetEnvironmentVariable("TS_Q4E_BF16_ROUTER") != "0")
                dst.RouterBf16 = UploadU16(dev, src.RouterBf16);
            else
                dst.Router = UploadF32(dev, src.Router);
            dst.ShexpGate = UploadF32(dev, src.ShexpGate);
            if (_residency != null && il >= 0)
                PlaceResidentExperts(dev, il, dst);
            else
            {
                dst.GateExps = dev.Weights.Place(src.GateExps);
                dst.UpExps = dev.Weights.Place(src.UpExps);
                dst.DownExps = dev.Weights.Place(src.DownExps);
            }
            dst.GateShexp = dev.Weights.Place(src.GateShexp);
            dst.UpShexp = dev.Weights.Place(src.UpShexp);
            dst.DownShexp = dev.Weights.Place(src.DownShexp);
        }

        // -------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------

        internal static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

        /// <summary>Address of element <paramref name="offset"/> of an F32 tensor.</summary>
        private static IntPtr At(Tensor t, long offset) => (IntPtr)((long)Ptr(t) + offset * 4);

        private static Tensor AllocT(Dev dev, DType type, params long[] sizes)
        {
            dev.MakeCurrent();
            var t = new Tensor(dev.Alloc, type, sizes);
            dev.OwnedTensors.Add(t);
            return t;
        }

        private static Tensor AllocF32(Dev dev, params long[] sizes) => AllocT(dev, DType.Float32, sizes);

        private static Tensor UploadF32(Dev dev, float[] data)
        {
            if (data == null || data.Length == 0)
                return null;
            Tensor t = AllocF32(dev, data.Length);
            fixed (float* src = data)
                CudaDriverApi.cuMemcpyHtoD(Ptr(t), (IntPtr)src, new UIntPtr((ulong)data.Length * 4)).ThrowOnError();
            return t;
        }

        private static Tensor UploadU16(Dev dev, ushort[] data)
        {
            Tensor t = AllocT(dev, DType.UInt8, (long)data.Length * 2);
            fixed (ushort* src = data)
                CudaDriverApi.cuMemcpyHtoD(Ptr(t), (IntPtr)src, new UIntPtr((ulong)data.Length * 2)).ThrowOnError();
            return t;
        }

        private static Tensor Block(Tensor t, long first, long rows, long cols)
        {
            using Tensor flat = t.View(t.ElementCount());
            using Tensor slice = flat.Narrow(0, first * cols, rows * cols);
            return slice.View(rows, cols);
        }

        private static int EnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        /// <summary>result[rows, w.Ne1] = input[rows, w.Ne0] x w^T through the shared matmul routing,
        /// asking for row-invariant kernels so a batched step computes each sequence's row as its
        /// own forward does.</summary>
        private static void MatMul(in DeviceWeight w, Tensor input, Tensor output, int rows)
        {
            CudaDecodeCapture.Refuse($"the shared matmul route (type {w.Type}, {w.Ne0} -> {w.Ne1})");
            using Tensor a = Block(input, 0, rows, w.Ne0);
            using Tensor r = Block(output, 0, rows, w.Ne1);
            if (rows >= PrefillF16MinRows && w.PrefillF16 != IntPtr.Zero)
                CudaQuantizedOps.AddmmResidentToFloat32(r, a, w.PrefillF16, TF16, w.Ne0, w.Ne1);
            else
                CudaQuantizedOps.AddmmResidentToFloat32(r, a, w.Ptr, w.Type, w.Ne0, w.Ne1, rowInvariant: true);
        }

        /// <summary>A decode-size input quantized to q8_1 once, for its several projections; zero past
        /// <see cref="PerSlotMaxRows"/> rows, where each projection quantizes for its own kernel. The
        /// device holds one such input at a time.</summary>
        private static IntPtr QuantizeShared(Dev dev, Tensor input, int inDim, int rows) => QuantizeShared(dev, input, inDim, rows, dev.Stream);

        private static IntPtr QuantizeShared(Dev dev, Tensor input, int inDim, int rows, IntPtr stream)
        {
            if (rows > PerSlotMaxRows)
                return IntPtr.Zero;
            dev.Alloc.Kernels.LaunchQuantizeQ81Rows(Ptr(input), Ptr(dev.SharedQ81), inDim, rows, stream, CudaQuantizedOps.Q81WarpQuantizeEnabled);
            return Ptr(dev.SharedQ81);
        }

        /// <summary><see cref="MatMul(in DeviceWeight, Tensor, Tensor, int)"/> of an input
        /// <see cref="QuantizeShared"/> already quantized: the same kernel, without quantizing again.</summary>
        private static void MatMul(Dev dev, in DeviceWeight w, Tensor input, IntPtr xq, Tensor output, int rows)
            => MatMul(dev, w, input, xq, output, rows, dev.Stream);

        /// <summary>Whether <see cref="MatMul(Dev, in DeviceWeight, Tensor, IntPtr, Tensor, int, IntPtr)"/> of a quantized
        /// decode-size input takes a kernel that launches on the stream given (not the shared route, which is the
        /// allocator's and not capturable).</summary>
        private static bool DecodeKernelFor(Dev dev, in DeviceWeight w, int rows)
            => rows <= PerSlotMaxRows
               && ((w.Type == 30 && w.Ne0 % 2 == 0 && dev.DK.DecodesLegacyQuants)
                   || CudaQuantizedOps.ResidentMatmulTakesQ81(w.Type, (int)w.Ne0, (int)w.Ne1, rows)
                   || Dsv4Kernels.DenseQ81RowsSupports(w.Type, (int)w.Ne0));

        private static void MatMul(Dev dev, in DeviceWeight w, Tensor input, IntPtr xq, Tensor output, int rows, IntPtr stream)
        {
            // BF16 matrices (ISTA's hyper-connection and PLE value weights) at decode size: F32 activations, as
            // ggml computes them, in a kernel a captured step can hold.
            if (w.Type == 30 && rows <= PerSlotMaxRows && w.Ne0 % 2 == 0 && dev.DK.DecodesLegacyQuants)
            {
                dev.DK.DenseBf16Rows(w.Ptr, Ptr(input), Ptr(output), w.Ne1, w.Ne0, rows, stream);
                return;
            }
            // Q2_0 rows past decode size: the shared route has no Q2_0 decoder, so quantize the input into the
            // routed experts' q8_1 scratch (free outside their call) and run the warp-per-row kernel.
            if (xq == IntPtr.Zero && w.Type == 42 && dev.DK.DecodesLegacyQuants
                && (long)rows * (w.Ne0 / 32) * 36 <= dev.Moe.ActQ8A.ElementCount())
            {
                CudaQuantizedOps.QuantizeRowsQ81(dev.Alloc, Ptr(input), Ptr(dev.Moe.ActQ8A), w.Ne0, rows);
                dev.DK.DenseQ81Rows(w.Ptr, Ptr(dev.Moe.ActQ8A), Ptr(output), w.Type, w.Ne1, w.Ne0, w.RowBytes, rows, stream);
                return;
            }
            if (xq != IntPtr.Zero
                && CudaQuantizedOps.TryResidentMatmulQ81(dev.Alloc, w.Ptr, w.Type, xq, Ptr(output), w.Ne0, w.Ne1, rows, stream))
                return;
            // The q8_1 decode kernels above cover Q2_K/Q3_K/Q6_K/Q8_0. Any other layout the warp-per-row
            // dot reads (UD-Q2_K_XL keeps much of its dense half in Q5_K) runs here rather than on the
            // shared route, which quantizes into shared scratch and would cost the step its capture.
            if (xq != IntPtr.Zero && Dsv4Kernels.DenseQ81RowsSupports(w.Type, w.Ne0))
            {
                dev.DK.DenseQ81Rows(w.Ptr, xq, Ptr(output), w.Type, w.Ne1, w.Ne0, w.RowBytes, rows, stream);
                return;
            }
            MatMul(w, input, output, rows);
        }

        /// <summary>output[rows, outDim] = input[rows, inDim] x w^T for an F32 weight [outDim, inDim]:
        /// the split GEMV (row invariant) up to its row bound, cuBLAS past it.</summary>
        /// <summary>A projection of a decode-size input through the device's own q8_1 buffer (so a
        /// captured decode step never reads a shared scratch another call may reallocate); wider inputs
        /// take the shared routing.</summary>
        private static void MatMulQ(Dev dev, in DeviceWeight w, Tensor input, Tensor output, int rows)
            => MatMul(dev, w, input, QuantizeShared(dev, input, w.Ne0, rows), output, rows);

        private static void MatMulF32(Dev dev, Tensor w, Tensor input, Tensor output, int inDim, int outDim, int rows)
        {
            if (rows <= Dsv4Kernels.GemvMaxRows)
            {
                dev.DK.Gemv(Ptr(input), Ptr(w), Ptr(output), inDim, outDim, rows, dev.Stream, Ptr(dev.GemvScratch));
                return;
            }
            CudaDecodeCapture.Refuse("cuBLAS");
            using Tensor a = Block(input, 0, rows, inDim);
            using Tensor r = Block(output, 0, rows, outDim);
            CudaQuantizedOps.AddmmResidentToFloat32(r, a, Ptr(w), TF32, inDim, outDim);
        }

        // Prefill-size GEMMs run on F16 copies of the large dense Q8_0 projections: cuBLAS on them
        // beat the int8 MMQ GEMM 1.7x at 512 rows on the A40 (GLM-5.3-Flash's measurement, the same
        // kernels), at one extra F16 copy of each. Copies are taken only while a device keeps this
        // much free for slots, scratch and cuBLAS; a weight left without one takes the quantized route.
        private const long PrefillCacheReserveBytes = 4L << 30;
        private const long PrefillCacheMinElements = 1L << 21;
        /// <summary>Rows from which a projection with an F16 copy uses it.</summary>
        private const int PrefillF16MinRows = 64;

        private void CachePrefillWeights()
        {
            long bytes = 0;
            int cached = 0, skipped = 0;
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                void Cache(ref DeviceWeight w)
                {
                    long elements = (long)w.Ne0 * w.Ne1;
                    if (w.Ptr == IntPtr.Zero || w.Type != TQ8_0 || elements < PrefillCacheMinElements)
                        return;
                    (long free, _) = dev.Alloc.GetMemoryInfo();
                    if (free - elements * 2 < PrefillCacheReserveBytes)
                    {
                        skipped++;
                        return;
                    }
                    Tensor t = AllocT(dev, DType.Float16, elements);
                    dev.Alloc.Kernels.LaunchDequantWeightF16(w.Ptr, Ptr(t), w.Type, w.Ne0, elements, dev.Stream);
                    w.PrefillF16 = Ptr(t);
                    bytes += elements * 2;
                    cached++;
                }
                foreach (var L in _layers)
                {
                    if (L.Device != dev.Ordinal)
                        continue;
                    Cache(ref L.HcAttnDown); Cache(ref L.HcAttnUp); Cache(ref L.HcFfnDown); Cache(ref L.HcFfnUp);
                    Cache(ref L.Qkv); Cache(ref L.Gate); Cache(ref L.SsmOut);
                    Cache(ref L.Wq); Cache(ref L.Wk); Cache(ref L.Wv); Cache(ref L.Wo);
                    Cache(ref L.PleKey); Cache(ref L.PleValue);
                    Cache(ref L.GateShexp); Cache(ref L.UpShexp); Cache(ref L.DownShexp);
                }
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            Console.Error.WriteLine($"[q4e-cuda] prefill F16 copies of {cached} dense projections ({bytes / (1024.0 * 1024 * 1024):F1} GiB)"
                + (skipped > 0 ? $"; {skipped} stay quantized (device memory)" : ""));
        }

        public void Dispose()
        {
            foreach (var slot in _slots.Values)
                FreeSlot(slot);
            _graphs.Clear();
            _slots.Clear();
            foreach (var dev in _devs)
            {
                if (dev == null)
                    continue;
                try
                {
                    dev.MakeCurrent();
                    CudaDriverApi.cuStreamSynchronize(dev.Stream);
                    foreach (var t in dev.OwnedTensors)
                        t.Dispose();
                    dev.OwnedTensors.Clear();
                    dev.Weights?.Dispose();
                    if (dev.XsReadyEv != IntPtr.Zero) CudaDriverApi.cuEventDestroy(dev.XsReadyEv);
                    if (dev.CopyDoneEv != IntPtr.Zero) CudaDriverApi.cuEventDestroy(dev.CopyDoneEv);
                    if (dev.ShForkEv != IntPtr.Zero) CudaDriverApi.cuEventDestroy(dev.ShForkEv);
                    if (dev.ShJoinEv != IntPtr.Zero) CudaDriverApi.cuEventDestroy(dev.ShJoinEv);
                    if (dev.ShStream != IntPtr.Zero) CudaDriverApi.cuStreamDestroy(dev.ShStream);
                    if (dev.BoundaryPinned != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(dev.BoundaryPinned);
                    if (dev.BatchStatePinned != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(dev.BatchStatePinned);
                    if (dev.PlePinned != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(dev.PlePinned);
                    dev.DK?.Dispose();
                    dev.GK?.Dispose();
                    dev.QK?.Dispose();
                    if (dev.Ordinal == 0)
                    {
                        StopHostExperts();
                        FreeHostExperts();
                    }
                    dev.Alloc?.Dispose();
                }
                catch
                {
                    // teardown must not throw
                }
            }
            _stages.Dispose();
            if (_pinnedTokens != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedTokens);
            if (_posPinned != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_posPinned);
            if (_pinnedLogits != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedLogits);
            if (_pinnedHidden != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedHidden);
            FreeMtpPinned();
        }
    }
}
