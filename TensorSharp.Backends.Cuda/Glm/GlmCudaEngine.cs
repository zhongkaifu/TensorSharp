// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// GLM-5.3-Flash (glm5next) on the direct-CUDA backend: the whole model resident across one or
// more GPUs (contiguous runs of layers per device), driven without ggml.
//
// Per trunk layer, both halves wrapped in Sinkhorn hyper-connections over 4 streams:
//   attention:  KDA linear attention (34 layers: gated delta rule with a per-channel decay)
//               or NoPE MLA over one 512-wide latent per token (11 layers; its pooled DSA
//               indexer caches a key and a pooling gate per token)
//   FFN:        dense SwiGLU (3 leading layers) or a 288-expert sigmoid-routed MoE with a
//               shared expert
// and the head reads the unweighted mean of the streams.
//
// The reference is the native executor ggml_cuda runs (ggml_ops_glm_dsa.cpp: build_g5n,
// build_kda, build_attention, build_indexer_g5n, build_moe). The hyper-connection, attention
// core and expert kernels are the DeepSeek V4 engine's (Dsv4Kernels, CudaMoe); the KDA and MLA
// cache kernels are this model's (GlmKernels).
//
// Past indexer_top_k cached tokens an MLA layer attends only over the pools its indexer selects
// (build_indexer_g5n / build_topk_mask_g5n); below it every cell is visible anyway.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class GlmCudaEngine : IDisposable
    {
        private const int HC = 4;
        private const int HcMixDim = (2 + HC) * HC;
        private const int TF32 = 0, TF16 = 1, TQ8_0 = 8;

        public sealed class LayerDesc
        {
            /// <summary>KDA linear attention; otherwise NoPE MLA with the pooled indexer.</summary>
            public bool Recurrent;
            /// <summary>Routed MoE FFN; otherwise dense SwiGLU.</summary>
            public bool Moe;
            public float[] AttnNorm, FfnNorm;
            public CudaWeightDesc HcAttnFn, HcFfnFn;
            public float[] HcAttnScale, HcAttnBase, HcFfnScale, HcFfnBase;

            // ---- KDA ----
            public CudaWeightDesc KdaQ, KdaK, KdaV, KdaFA, KdaFB, KdaGA, KdaGB, KdaBeta, KdaOut;
            /// <summary>Per channel [dInner, dConv]: the file's [dConv, 1, dInner] as stored.</summary>
            public float[] ConvQ, ConvK, ConvV;
            public float[] DtBias, SsmA, SsmNorm;

            // ---- MLA ----
            public CudaWeightDesc WqA, WqB, WkvA, WkB, WvB, Wo;
            public float[] QANorm, KvANorm;
            public CudaWeightDesc IdxQB, IdxK, IdxGate;
            /// <summary>Head weights [dModel, heads] F32; the pool position embedding [kpool, D] slot-major.</summary>
            public float[] IdxKNormW, IdxKNormB, IdxProj, IdxApe;

            // ---- FFN ----
            public CudaWeightDesc FfnGate, FfnUp, FfnDown;
            public float[] GateInp, ExpProbsBias;
            public CudaWeightDesc GateExps, UpExps, DownExps, GateShexp, UpShexp, DownShexp;
        }

        public sealed class ModelDesc
        {
            public int NLayer, NEmbd, NHead, NVocab;
            public int KdaHeadDim, DConv;
            public float KdaGateLowerBound;
            public int QLoraRank, KvLoraRank, HeadDimK, HeadDimV;
            public int IdxNHead, IdxHeadDim, IdxTopK, IdxKpool;
            public int NExpert, NExpertUsed, NFfExp, NFf, NFfShexp;
            public float ExpertWeightsScale;
            public bool ExpertWeightsNorm;
            public float SwigluClamp;
            public int HcSinkhornIters;
            public float HcEps, RmsEps, NormEps;
            public int NCtx, NUbatch;
            public CudaWeightDesc TokEmbd, Output;
            public float[] OutputNorm;
            public LayerDesc[] Layers;
        }

        private sealed class DevLayer
        {
            public int Device;
            public bool Recurrent, Moe;
            public Tensor AttnNorm, FfnNorm, HcAttnScale, HcAttnBase, HcFfnScale, HcFfnBase;
            public DeviceWeight HcAttnFn, HcFfnFn;
            public DeviceWeight KdaQ, KdaK, KdaV, KdaFA, KdaFB, KdaGA, KdaGB, KdaBeta, KdaOut;
            public Tensor ConvQ, ConvK, ConvV, DtBias, SsmA, SsmNorm;
            public DeviceWeight WqA, WqB, WkvA, WkB, WvB, Wo, IdxK, IdxGate, IdxQB;
            public Tensor QANorm, KvANorm, IdxKNormW, IdxKNormB, IdxApe;
            /// <summary>The indexer's head weights with 1/sqrt(D * H) folded in.</summary>
            public Tensor IdxProj;
            /// <summary>wk_b and wv_b as F16, the operands of the per-head batched GEMMs.</summary>
            public Tensor WkBF16, WvBF16;
            public DeviceWeight FfnGate, FfnUp, FfnDown, GateExps, UpExps, DownExps, GateShexp, UpShexp, DownShexp;
            public Tensor GateInp, ExpProbsBias;
        }

        private sealed class Dev
        {
            public int Ordinal;
            public CudaAllocator Alloc;
            public Dsv4Kernels DK;
            public GlmKernels GK;
            public CudaWeightArena Weights;
            public List<Tensor> OwnedTensors = new List<Tensor>();
            // Layer-boundary handoff of the hidden streams through pinned host memory (peer DMA
            // is not trusted on these topologies): DtoH on this device's stream, HtoD on the next.
            public IntPtr BoundaryPinned, XsReadyEv, CopyDoneEv;
            public Tensor Tokens;
            /// <summary>A captured decode step's position, copied in from pinned memory at launch.</summary>
            public Tensor PosDev;
            public Tensor NegInf;          // attention sinks that take no weight
            public CudaMoeScratch Moe;

            public Tensor Xs, XsOut, Cur, Inv, Mixes, Pre, Post, Comb, AttnOut, FfnOut;
            /// <summary>A decode-size input quantized once for its projections (QuantizeShared).</summary>
            public Tensor SharedQ81;
            /// <summary>A decode-size input projected once (MatMulQ): kept apart from SharedQ81, which
            /// may still hold the layer input's quantization.</summary>
            public Tensor InnerQ81;
            /// <summary>The hyper-connection pre-block's per-slice partials.</summary>
            public Tensor HcPartials;
            // KDA
            public Tensor KQ, KK, KV, KF, KG, KLow, KBeta, KScr, KCore, KOutIn;
            // MLA
            public Tensor Qr, Q, QF16, KvRaw, IdxKey, IdxGateOut, QAbs, AttnO, AttnOF16, VOut;
            // the pooled indexer's selection
            public Tensor IdxQ, IdxW, PoolScores, TopkIdx, TopkCnt, Cells, CellCnt;
            // FFN
            public Tensor RouterLogits, ShGate, ShUp, ShDown, Logits;
            public Tensor GemvScratch;
            // A batched decode step: every row's logits, and the per-row KDA state pointer tables
            // ([layer][conv, ssm][row]) staged through pinned memory.
            public Tensor BatchLogits, BatchStatePtrs;
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
            /// <summary>Per device: [layer][conv, ssm] device pointers of this slot's KDA states
            /// (the kernels take arrays of per-sequence state pointers).</summary>
            public Tensor[] StatePtrs;
            public readonly List<Tensor> Owned = new List<Tensor>();
        }

        private sealed class SlotLayer
        {
            public Tensor ConvState, Ssm;      // KDA
            public Tensor KvCache, IdxCache;   // MLA
            public Tensor PoolKeys;            // [nCtx / kpool, D] F32, filled as pools complete
        }

        private readonly ModelDesc _m;
        private readonly Dev[] _devs;
        private readonly DevLayer[] _layers;
        private readonly int _lastDev;
        private DeviceWeight _tokEmbd, _output;
        private Tensor _outputNorm;
        private IntPtr _pinnedTokens, _pinnedLogits, _pinnedHidden;
        private Slot _active;
        private readonly int _perf;
        private readonly CudaStageTimer _stages;

        public int NPast => _active.NPast;
        public int ContextSize => _m.NCtx;
        public int UBatch => _m.NUbatch;
        public int VocabSize => _m.NVocab;

        public GlmCudaEngine(ModelDesc m, int nGpu)
        {
            _m = m ?? throw new ArgumentNullException(nameof(m));
            if (m.KdaHeadDim != GlmKernels.KdaHeadDim)
                throw new NotSupportedException($"[glm-cuda] KDA head_dim {m.KdaHeadDim} (the kernels take {GlmKernels.KdaHeadDim})");
            if (m.KvLoraRank != 512)
                throw new NotSupportedException($"[glm-cuda] kv_lora_rank {m.KvLoraRank} (the attention kernel takes a 512-wide latent)");
            if (m.NExpertUsed > 16)
                throw new NotSupportedException($"[glm-cuda] at most 16 experts per token, got {m.NExpertUsed}");
            if (m.NEmbd % 4 != 0 || HC * m.NEmbd % 256 != 0)
                throw new NotSupportedException($"[glm-cuda] hidden width {m.NEmbd}");
            _perf = EnvInt("TS_GLM_PERF", 0);
            _stages = new CudaStageTimer("glm-cuda", _perf, StageNames);

            CudaBackend.Register();
            CudaDriverApi.cuInit(0);
            CudaDriverApi.cuDeviceGetCount(out int devCount).ThrowOnError();
            int useDevs = nGpu > 0 ? Math.Min(nGpu, devCount) : devCount;
            if (useDevs < 1)
                throw new InvalidOperationException("No CUDA devices available for the GLM engine.");

            _devs = new Dev[useDevs];
            for (int d = 0; d < useDevs; d++)
            {
                var dev = new Dev { Ordinal = d, Alloc = new CudaAllocator(d) };
                dev.MakeCurrent();
                dev.DK = Dsv4Kernels.Create();
                dev.GK = GlmKernels.Create();
                _devs[d] = dev;
            }
            // ---- placement: contiguous runs of layers, each device filled to the same share of
            // what it has free after its fixed scratch ----
            var layerBytes = new long[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
            {
                foreach (var w in LayerWeights(m.Layers[il]))
                    layerBytes[il] += CudaWeightArena.Align(w.TotalBytes);
                layerBytes[il] += LayerCacheBytes(il);
            }
            int[] assignment = PlaceLayers(layerBytes);
            _lastDev = assignment[m.NLayer - 1];

            // ---- arenas ----
            var arenaNeed = new long[useDevs];
            for (int il = 0; il < m.NLayer; il++)
                foreach (var w in LayerWeights(m.Layers[il]))
                    arenaNeed[assignment[il]] += CudaWeightArena.Align(w.TotalBytes);
            arenaNeed[0] += CudaWeightArena.Align(m.TokEmbd.TotalBytes);
            arenaNeed[_lastDev] += CudaWeightArena.Align(m.Output.TotalBytes);
            for (int d = 0; d < useDevs; d++)
            {
                _devs[d].MakeCurrent();
                _devs[d].Weights = new CudaWeightArena(_devs[d].Alloc, arenaNeed[d]);
            }

            _layers = new DevLayer[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
                _layers[il] = new DevLayer { Device = assignment[il], Recurrent = m.Layers[il].Recurrent, Moe = m.Layers[il].Moe };

            var sw = Stopwatch.StartNew();
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
                    _outputNorm = UploadF32(dev, m.OutputNorm);
                }
            });
            CudaWeightArena.StreamAll(Array.ConvertAll(_devs, d => d.Weights), "glm-cuda");

            // The MLA's per-head projections run as batched GEMMs on F16 copies.
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                for (int il = 0; il < m.NLayer; il++)
                {
                    var L = _layers[il];
                    if (L.Device != dev.Ordinal || L.Recurrent)
                        continue;
                    L.WkBF16 = DequantF16(dev, L.WkB, Elements(m.Layers[il].WkB));
                    L.WvBF16 = DequantF16(dev, L.WvB, Elements(m.Layers[il].WvB));
                }
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }

            foreach (var dev in _devs)
                AllocateScratch(dev);
            _active = CreateSlot();
            _slots[_active.Id] = _active;
            CachePrefillWeights();

            double gib = 0;
            foreach (long b in layerBytes)
                gib += b;
            gib /= 1024.0 * 1024 * 1024;
            Console.Error.WriteLine(
                $"[glm-cuda] {gib:F1} GiB of weights and caches across {useDevs} GPU(s) " +
                $"(layer split {string.Join("/", CountPerDev(assignment, useDevs))}), loaded in {sw.Elapsed.TotalSeconds:F1}s " +
                $"(n_ctx={m.NCtx}, ubatch={m.NUbatch})");
        }

        // -------------------------------------------------------------------
        // Placement and upload
        // -------------------------------------------------------------------

        private static IEnumerable<CudaWeightDesc> LayerWeights(LayerDesc l)
        {
            foreach (var w in new[]
                {
                    l.HcAttnFn, l.HcFfnFn,
                    l.KdaQ, l.KdaK, l.KdaV, l.KdaFA, l.KdaFB, l.KdaGA, l.KdaGB, l.KdaBeta, l.KdaOut,
                    l.WqA, l.WqB, l.WkvA, l.WkB, l.WvB, l.Wo, l.IdxK, l.IdxGate, l.IdxQB,
                    l.FfnGate, l.FfnUp, l.FfnDown, l.GateExps, l.UpExps, l.DownExps, l.GateShexp, l.UpShexp, l.DownShexp,
                })
            {
                if (w.IsValid)
                    yield return w;
            }
        }

        /// <summary>One sequence's cache bytes for a layer at the full context.</summary>
        private long LayerCacheBytes(int il)
        {
            var m = _m;
            if (m.Layers[il].Recurrent)
            {
                long dInner = (long)m.NHead * m.KdaHeadDim;
                return (m.DConv - 1) * 3 * dInner * 4 + (long)m.NHead * m.KdaHeadDim * m.KdaHeadDim * 4;
            }
            return (long)m.NCtx * (m.KvLoraRank + 2L * m.IdxHeadDim) * 2 + (long)m.NCtx / m.IdxKpool * m.IdxHeadDim * 4;
        }

        /// <summary>Scratch a device holds whatever layers it carries (ubatch-sized buffers).</summary>
        private long DeviceScratchBytes()
        {
            var m = _m;
            long nt = m.NUbatch, e = m.NEmbd, dInner = (long)m.NHead * m.KdaHeadDim;
            long b = 2 * nt * HC * e * 4 + 4 * nt * e * 4;
            b += 7 * nt * dInner * 4 + nt * m.NHead * GlmKernels.KdaScratch * 4L;
            b += nt * (long)m.NHead * (m.HeadDimK * 6 + m.KvLoraRank * 6);
            b += CudaMoeScratch.Bytes(m.NUbatch, m.NExpertUsed, m.NEmbd, Math.Max(m.NFfExp, 1), m.NExpert);
            b += 3 * nt * (long)Math.Max(m.NFf, m.NFfShexp) * 4;
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
                if (d == nDev - 1) b -= _m.Output.TotalBytes + (long)_m.NVocab * 8;
                budget[d] = Math.Max(0, b);
            }
            long total = 0, totalBudget = 0;
            foreach (long x in layerBytes) total += x;
            foreach (long x in budget) totalBudget += x;
            if (total > totalBudget)
                throw new InvalidOperationException(
                    $"[glm-cuda] the model needs {total / 1073741824.0:F1} GiB of weights and caches but the " +
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
                    throw new InvalidOperationException($"[glm-cuda] layer {il} does not fit on device {dev}");
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

        private void UploadLayer(Dev dev, int il)
        {
            var src = _m.Layers[il];
            var dst = _layers[il];
            dst.AttnNorm = UploadF32(dev, src.AttnNorm);
            dst.FfnNorm = UploadF32(dev, src.FfnNorm);
            dst.HcAttnFn = dev.Weights.Place(src.HcAttnFn);
            dst.HcFfnFn = dev.Weights.Place(src.HcFfnFn);
            dst.HcAttnScale = UploadF32(dev, src.HcAttnScale);
            dst.HcAttnBase = UploadF32(dev, src.HcAttnBase);
            dst.HcFfnScale = UploadF32(dev, src.HcFfnScale);
            dst.HcFfnBase = UploadF32(dev, src.HcFfnBase);
            if (src.Recurrent)
            {
                dst.KdaQ = dev.Weights.Place(src.KdaQ);
                dst.KdaK = dev.Weights.Place(src.KdaK);
                dst.KdaV = dev.Weights.Place(src.KdaV);
                dst.KdaFA = dev.Weights.Place(src.KdaFA);
                dst.KdaFB = dev.Weights.Place(src.KdaFB);
                dst.KdaGA = dev.Weights.Place(src.KdaGA);
                dst.KdaGB = dev.Weights.Place(src.KdaGB);
                dst.KdaBeta = dev.Weights.Place(src.KdaBeta);
                dst.KdaOut = dev.Weights.Place(src.KdaOut);
                dst.ConvQ = UploadF32(dev, src.ConvQ);
                dst.ConvK = UploadF32(dev, src.ConvK);
                dst.ConvV = UploadF32(dev, src.ConvV);
                dst.DtBias = UploadF32(dev, src.DtBias);
                dst.SsmA = UploadF32(dev, src.SsmA);
                dst.SsmNorm = UploadF32(dev, src.SsmNorm);
            }
            else
            {
                dst.WqA = dev.Weights.Place(src.WqA);
                dst.WqB = dev.Weights.Place(src.WqB);
                dst.WkvA = dev.Weights.Place(src.WkvA);
                dst.WkB = dev.Weights.Place(src.WkB);
                dst.WvB = dev.Weights.Place(src.WvB);
                dst.Wo = dev.Weights.Place(src.Wo);
                dst.IdxK = dev.Weights.Place(src.IdxK);
                dst.IdxGate = dev.Weights.Place(src.IdxGate);
                dst.IdxQB = dev.Weights.Place(src.IdxQB);
                dst.IdxApe = UploadF32(dev, src.IdxApe);
                // Both of the head weights' scale constants, folded in on the small tensor.
                float headScale = 1.0f / MathF.Sqrt((float)_m.IdxHeadDim * _m.IdxNHead);
                dst.IdxProj = UploadF32(dev, Array.ConvertAll(src.IdxProj, v => v * headScale));
                dst.QANorm = UploadF32(dev, src.QANorm);
                dst.KvANorm = UploadF32(dev, src.KvANorm);
                dst.IdxKNormW = UploadF32(dev, src.IdxKNormW);
                dst.IdxKNormB = UploadF32(dev, src.IdxKNormB);
            }
            if (src.Moe)
            {
                dst.GateExps = dev.Weights.Place(src.GateExps);
                dst.UpExps = dev.Weights.Place(src.UpExps);
                dst.DownExps = dev.Weights.Place(src.DownExps);
                dst.GateShexp = dev.Weights.Place(src.GateShexp);
                dst.UpShexp = dev.Weights.Place(src.UpShexp);
                dst.DownShexp = dev.Weights.Place(src.DownShexp);
                dst.GateInp = UploadF32(dev, src.GateInp);
                dst.ExpProbsBias = UploadF32(dev, src.ExpProbsBias);
                if (dst.GateExps.Type != dst.UpExps.Type)
                    throw new NotSupportedException($"[glm-cuda] layer {il}: expert gate/up quant types differ");
                CudaMoe.RequireExpertType("glm-cuda", dst.GateExps.Type);
                CudaMoe.RequireExpertType("glm-cuda", dst.DownExps.Type);
            }
            else
            {
                dst.FfnGate = dev.Weights.Place(src.FfnGate);
                dst.FfnUp = dev.Weights.Place(src.FfnUp);
                dst.FfnDown = dev.Weights.Place(src.FfnDown);
            }
        }

        // Prefill-size GEMMs run on F16 copies of the large dense Q8_0 projections: cuBLAS on them
        // beat the int8 MMQ GEMM 1.7x at 512 rows on the A40 (and dequantizing per call 1.4x), at
        // one extra F16 copy of each. Copies are taken only while a device keeps this much free
        // for slots, scratch and cuBLAS; a weight left without one takes the quantized route.
        private const long PrefillCacheReserveBytes = 4L << 30;
        private const long PrefillCacheMinElements = 1L << 22;
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
                    w.PrefillF16 = Ptr(DequantF16(dev, w, elements));
                    bytes += elements * 2;
                    cached++;
                }
                foreach (var L in _layers)
                {
                    if (L.Device != dev.Ordinal)
                        continue;
                    Cache(ref L.KdaQ); Cache(ref L.KdaK); Cache(ref L.KdaV); Cache(ref L.KdaOut);
                    Cache(ref L.WqA); Cache(ref L.WqB); Cache(ref L.WkvA); Cache(ref L.Wo);
                    Cache(ref L.FfnGate); Cache(ref L.FfnUp); Cache(ref L.FfnDown);
                    Cache(ref L.GateShexp); Cache(ref L.UpShexp); Cache(ref L.DownShexp);
                }
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            Console.Error.WriteLine($"[glm-cuda] prefill F16 copies of {cached} dense projections ({bytes / (1024.0 * 1024 * 1024):F1} GiB)"
                + (skipped > 0 ? $"; {skipped} stay quantized (device memory)" : ""));
        }

        /// <summary>An arena weight of <paramref name="elements"/> values dequantized into an F16
        /// tensor of the same row-major layout.</summary>
        private static Tensor DequantF16(Dev dev, in DeviceWeight w, long elements)
        {
            Tensor t = AllocT(dev, DType.Float16, elements);
            dev.Alloc.Kernels.LaunchDequantWeightF16(w.Ptr, Ptr(t), w.Type, w.Ne0, elements, dev.Stream);
            return t;
        }

        private static long Elements(in CudaWeightDesc w) => (long)w.Ne0 * w.Ne1 * Math.Max(1, w.Ne2);

        // -------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------

        internal static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

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

        private static Tensor Rows(Tensor t, int rows)
            => t.Sizes[0] == rows ? t.CopyRef() : t.Narrow(0, 0, rows);

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

        private void RmsNorm(Tensor data, Tensor weight, int rows)
        {
            using Tensor view = Rows(data, rows);
            Ops.RMSNorm(view, view, weight, null, _m.RmsEps);
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
        private static IntPtr QuantizeShared(Dev dev, Tensor input, int inDim, int rows)
        {
            if (rows > PerSlotMaxRows)
                return IntPtr.Zero;
            CudaQuantizedOps.QuantizeRowsQ81(dev.Alloc, Ptr(input), Ptr(dev.SharedQ81), inDim, rows);
            return Ptr(dev.SharedQ81);
        }

        /// <summary><see cref="MatMul(in DeviceWeight, Tensor, Tensor, int)"/> of an input
        /// <see cref="QuantizeShared"/> already quantized: the same kernel, without quantizing again.</summary>
        private static void MatMul(Dev dev, in DeviceWeight w, Tensor input, IntPtr xq, Tensor output, int rows)
        {
            if (xq != IntPtr.Zero
                && CudaQuantizedOps.TryResidentMatmulQ81(dev.Alloc, w.Ptr, w.Type, xq, Ptr(output), w.Ne0, w.Ne1, rows))
                return;
            MatMul(w, input, output, rows);
        }

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

        /// <summary>A decode-size projection of an input nothing else projects (the low-rank ups, the
        /// output and down projections, the head) through the device's own second q8_1 buffer: the
        /// kernel the shared route picks, without its scratch (a captured step must own every buffer
        /// it reads) and without touching the layer input's quantization in SharedQ81. Wider inputs
        /// and other weight types take the shared route.</summary>
        private static void MatMulQ(Dev dev, in DeviceWeight w, Tensor input, Tensor output, int rows)
        {
            if (rows <= PerSlotMaxRows && CudaQuantizedOps.ResidentMatmulTakesQ81(w.Type, w.Ne0, w.Ne1, rows))
            {
                CudaQuantizedOps.QuantizeRowsQ81(dev.Alloc, Ptr(input), Ptr(dev.InnerQ81), w.Ne0, rows);
                CudaQuantizedOps.TryResidentMatmulQ81(dev.Alloc, w.Ptr, w.Type, Ptr(dev.InnerQ81), Ptr(output), w.Ne0, w.Ne1, rows);
                return;
            }
            MatMul(w, input, output, rows);
        }

        public void Dispose()
        {
            foreach (var slot in _slots.Values)
                FreeSlot(slot);
            _slots.Clear();
            _graphs.Clear();
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
                    if (dev.BoundaryPinned != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(dev.BoundaryPinned);
                    dev.DK?.Dispose();
                    dev.GK?.Dispose();
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
            if (_visionPinned != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_visionPinned);
            if (_pinnedLogits != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedLogits);
            if (_pinnedHidden != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedHidden);
        }
    }
}
