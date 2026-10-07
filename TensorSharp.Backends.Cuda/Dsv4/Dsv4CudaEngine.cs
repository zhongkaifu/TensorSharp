// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) whole-model engine for the direct-CUDA backend.
//
// This is the GPU counterpart of DeepSeek4CpuExecutor: an imperative,
// single-sequence executor that keeps the quantized weights resident in
// device memory (layer-split across all visible GPUs by cumulative weight
// bytes, so a model larger than one GPU's VRAM is hosted across several) and
// drives the DSV4-specific kernels in tensorsharp_dsv4_kernels.cu plus the
// generic quantized matmul kernels in tensorsharp_kernels.cu, entirely
// through the CUDA driver API + cuBLAS. It has no dependency on ggml.
//
// The model-side executor (DeepSeek4CudaExecutor in TensorSharp.Models)
// parses the split-GGUF shards and hands this engine host pointers into the
// mapped tensor data plus pre-dequantized F32 copies of the small tensors
// (norms, gates, sinks, APE tables, the BF16 router) and precomputed RoPE
// cos/sin tables; this engine owns everything device-side.
//
// Hidden state flows device to device at layer-group boundaries via
// cuMemcpyPeerAsync ordered with events, so prefill chunks pipeline across
// GPUs naturally. Caches are F16 (parity with the native executor and
// llama.cpp); compressor state rings are F32.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Dsv4CudaEngine : IDisposable
    {
        private const int HC = 4;
        private const int HcMixDim = (2 + HC) * HC; // 24
        private const int CsaRatio = 4;
        private const int HcaRatio = 128;
        private const int Q81BlockBytes = 36;

        // ggml type ids this engine dispatches on
        private const int TF32 = 0, TF16 = 1, TQ8_0 = 8, TQ2_K = 10, TQ3_K = 11, TQ4_K = 12, TQ5_K = 13, TQ6_K = 14,
            TIQ2_XXS = 16, TIQ3_S = 21, TBF16 = 30, TMXFP4 = 39;

        public sealed class LayerDesc
        {
            public int Ratio;
            public float ClampExp, ClampShexp;
            public CudaWeightDesc WqA, WqB, Wkv, WoA, WoB;
            public CudaWeightDesc CompWkv, CompWgate, IdxProj, IdxQB, IdxCompWkv, IdxCompWgate;
            public CudaWeightDesc GateExps, UpExps, DownExps, GateShexp, UpShexp, DownShexp;
            public float[] AttnNorm, QANorm, KvNorm, Sinks;
            public float[] HcAttnFn, HcAttnScale, HcAttnBase, HcFfnFn, HcFfnScale, HcFfnBase;
            public float[] CompApe, CompNorm, IdxCompApe, IdxCompNorm;
            public float[] GateInp, ExpProbsBias, FfnNorm;
            public int[] Tid2Eid;

            // ---- V4.1 ----
            /// <summary>Layer that builds the compressed and indexer caches this
            /// layer reads, and the layer that publishes the sparse selection.
            /// Both -1 on uncompressed layers.</summary>
            public int KvSource = -1, IndexSource = -1;
            /// <summary>V4.1 projects the compressed latent into the indexer's key
            /// space instead of running a second compressor for it.</summary>
            public CudaWeightDesc IndexerK;
            public float[] IndexerKNorm;
            /// <summary>Index into the GGUF's table list, or -1.</summary>
            public int EngramIndex = -1;
            public CudaWeightDesc EngramWkv;
            /// <summary>The layer's Engram table, uploaded when the split can hold it
            /// (<see cref="EngramResident"/>); the host gathers from its mapping otherwise.</summary>
            public CudaWeightDesc EngramTable;
            public float[] EngramQ, EngramK;
        }

        public sealed class ModelDesc
        {
            public int NLayer, NEmbd, NHead, HeadDim, NRot, QLoraRank, OGroups, OLoraRank, NSwa, NVocab;
            public int NExpert, NExpertUsed, NFfExp, HashLayerCount;
            public int IdxNHead, IdxHeadSize, IdxTopK;
            public int HcSinkhornIters;
            public float RmsEps, HcEps, ExpertWeightsScale;
            public bool ExpertWeightsNorm;
            public int NCtx, NUbatch;
            public CudaWeightDesc TokEmbd, Output;
            /// <summary>Host quantized matmul for <c>--n-cpu-moe</c> layers; see
            /// <see cref="IDsv4HostMatMul"/>. Required only when the engine
            /// decides to offload (it will say so and throw if it is null).</summary>
            public IDsv4HostMatMul HostMatMul;
            public float[] OutputNorm, HcHeadFn, HcHeadScale, HcHeadBase;
            public float[] RopeRawTable, RopeCompTable; // [nCtx * nRot] interleaved cos/sin
            public LayerDesc[] Layers;

            // ---- V4.1 ----
            /// <summary>deepseek41 rather than deepseek4: compression ratios 1 and
            /// 2, shared caches, candidate pruning, Engram tables, delayed
            /// hyper-connection gates and the trained cache quantization.</summary>
            public bool V41;
            /// <summary>Indexer layer that prunes the key space for the layers
            /// after it, and the geometry of that pruning. -1 when absent.</summary>
            public int CandidateSource = -1, CandidateTopk, CandidateBlock;
            /// <summary>Host-side Engram gather; required when any layer names an
            /// Engram table.</summary>
            public IDsv41EngramSource Engram;
            /// <summary>Optional DSpark speculative-decoding module (see
            /// Dsv4CudaEngine.Dspark.cs); null when no drafter was loaded.</summary>
            public DsparkDesc Dspark;
        }

        private sealed class DevLayer
        {
            public int Device;
            public int Ratio;
            public float ClampExp, ClampShexp;
            public DeviceWeight WqA, WqB, Wkv, WoA, WoB;
            public DeviceWeight CompWkv, CompWgate, IdxProj, IdxQB, IdxCompWkv, IdxCompWgate;
            public DeviceWeight GateExps, UpExps, DownExps, GateShexp, UpShexp, DownShexp;
            // Small dense tensors are allocator-owned Tensors (the arena holds only
            // the packed quantized weights), so the shared Ops can consume them
            // directly. The caches belong to each sequence's slot.
            public Tensor AttnNorm, QANorm, KvNorm, Sinks;
            public Tensor HcAttnFn, HcAttnScale, HcAttnBase, HcFfnFn, HcFfnScale, HcFfnBase;
            public Tensor CompApe, CompNorm, IdxCompApe, IdxCompNorm;
            public Tensor GateInp, ExpProbsBias, FfnNorm, Tid2Eid;
            public int ShFf;

            // ---- V4.1 ----
            public int KvSource = -1, IndexSource = -1, EngramIndex = -1;
            public DeviceWeight IndexerK, EngramWkv, EngramTable;
            public Tensor IndexerKNorm, EngramQ, EngramK;
            // This Engram layer's own pinned staging, and the event recorded after the copy out
            // of it: the host fills the staging while the device may still be copying the last
            // fill, so each layer needs its own and must wait for that copy before refilling.
            public IntPtr EngramPinned, EngramCopied;
        }

        private sealed class Dev
        {
            public int Ordinal;
            public CudaAllocator Alloc;
            public Dsv4Kernels DK;
            public IntPtr Event;
            // Every packed quantized weight assigned to this device (see CudaWeightArena).
            public CudaWeightArena Weights;
            public Tensor RopeRaw, RopeComp;
            public bool NeedsTokens;
            public Tensor TokensDev0, TokensDev1;
            // Each row's position in a batched decode step (one row per sequence).
            public Tensor Positions;
            public IntPtr TokEv0, TokEv1; // guards pinned-buffer reuse per parity
            // Layer-boundary handoff staging: this device's outgoing hidden
            // streams go DtoH into BoundaryPinned on this device's stream, then
            // HtoD on the next device's stream. Direct cuMemcpyPeerAsync is NOT
            // used: on several cloud PCIe topologies peer DMA reports available
            // but silently transfers corrupt data (see CudaP2PCommunicator's
            // round-trip self-test for the same failure mode).
            public IntPtr BoundaryPinned;
            public IntPtr XsReadyEv;   // recorded on this stream after the DtoH
            public IntPtr CopyDoneEv;  // recorded on the DST stream after the HtoD
            // V4.1 state the next device's layers read (Dsv4CudaEngine.LayerSplit.cs), staged
            // through the same event chain as the streams.
            public IntPtr StatePinned;
            public long StatePinnedBytes;

            // Scratch. Every buffer is an allocator-owned Tensor so it comes out
            // of the shared CudaAllocator pool and is visible to its VRAM
            // accounting; the DSV4-specific kernels take the raw device pointer
            // via Ptr(), the shared Ops take the Tensor (or a per-ubatch row view).
            public Tensor Xs, XsOut, Cur, Inv, Mixes, Pre, Post, Comb;
            /// <summary>The decode-size hyper-connection pre-block's per-slice partials.</summary>
            public Tensor HcPartials;
            public Tensor Qr, Q, KvRaw, StKv, StScore, LidStKv, LidStScore;
            public Tensor Iq, Iw, IdxScores, TopkIdx, TopkCnt;
            // V4.1: the delayed hyper-connection gates (a block collapses the
            // streams with the PREVIOUS block's gates), the compressed latent
            // being built, the candidate mask, and the Engram staging.
            public Tensor PreAttn, PreFfn, Latent, LatentK, CandMask, EngramLookup, EngramKv;
            // The selected Engram rows of a device-resident table, [nt, columns].
            public Tensor EngramRows;
            public Tensor AttnO, OGrouped, OGroupedOut, OG, AttnOut, FfnOut;
            public Tensor RouterLogits, Sel, SelW, Counts, Offsets, Cursors, RowOfSlot, SlotToken;
            public Tensor ActQ8A, ActQ8B, ExpGate, ExpUp, ExpDown, ShGate, ShUp, ShDown;
            // dense split-q8_1 activation scratch for the register-staged expert
            // kernels (A = per-token rows into gate/up, B = packed rows into down)
            public Tensor SplitQsA, SplitDA, SplitQsB, SplitDB;
            public Tensor Logits;
            // Partial sums of the split small-row GEMV (the hyper-connection mixes).
            public Tensor GemvScratch;
            public List<Tensor> OwnedTensors = new List<Tensor>();

            public IntPtr Stream => Alloc.Stream.Handle;
            public void MakeCurrent() => Alloc.Context.MakeCurrent();
        }

        private readonly ModelDesc _m;
        private readonly Dev[] _devs;
        private readonly DevLayer[] _layers;
        private readonly int _ringRaw;
        private readonly int _compRowsCsa;
        private readonly int _compRowsHca;
        private readonly int _lastDev;
        private DeviceWeight _outputQW;
        private DeviceWeight _tokEmbdQW;
        private Tensor _outputNorm, _hcHeadFn, _hcHeadScale, _hcHeadBase;
        private IntPtr _pinnedTokens0, _pinnedTokens1;
        // Logits leave the device through PINNED staging: a device-to-host copy
        // into a pageable managed array is not just un-overlappable, the driver
        // stages it in small synchronous chunks (~0.1 GB/s here), which costs
        // more than the decode step that produced the logits.
        private IntPtr _pinnedLogits;
        private int _chunkParity;
        private readonly int _perf;
        private readonly CudaStageTimer _stages;
        private readonly bool _syncDebug;

        /// <summary>Head of the active slot.</summary>
        public int NPast => _active.NPast;

        /// <summary>Whether V4.1's Engram tables are in VRAM (gathered on the device) rather than
        /// host mappings the executor gathers from.</summary>
        public bool EngramResident { get; private set; }
        public int ContextSize => _m.NCtx;

        /// <summary>Prefill micro-batch the engine chunks by. Speculative prefill
        /// should hand it whole ubatches: a half-sized chunk re-reads every
        /// touched expert's weights for half as many tokens.</summary>
        public int UBatch => _m.NUbatch;

        /// <param name="nCpuMoe">Routed-expert CPU offload: 0 none (the default —
        /// offload is opt-in, and a model that does not fit is refused with the
        /// number of layers that would make it fit), N the first N layers,
        /// int.MaxValue every layer, -1 auto (the fewest leading layers that make
        /// the model fit; opt-in only). See Dsv4CudaEngine.HostMoe.cs.</param>
        public Dsv4CudaEngine(ModelDesc m, int nGpu, int nCpuMoe = 0)
        {
            _m = m ?? throw new ArgumentNullException(nameof(m));
            if (m.HeadDim != 512)
                throw new NotSupportedException($"DSV4 CUDA engine requires head_dim 512, got {m.HeadDim}.");
            if (m.IdxNHead > 0 && m.IdxHeadSize != 128)
                throw new NotSupportedException($"DSV4 CUDA engine requires indexer key_length 128, got {m.IdxHeadSize}.");
            if (m.NExpertUsed > 16)
                throw new NotSupportedException($"DSV4 CUDA engine supports at most 16 experts per token, got {m.NExpertUsed}.");

            _perf = EnvInt("TS_DSV4_PERF", 0);
            _stages = new CudaStageTimer("dsv4-cuda", _perf, StageNames);
            _syncDebug = EnvInt("TS_DSV4_CUDA_SYNCDBG", 0) != 0;

            // The model layer coerces DSV4's backend to Cpu (it only needs a cheap
            // host allocator), so the CUDA op handlers are not registered for us:
            // do it here, since this engine drives Ops.RMSNorm / Ops.SiLUMulClamp
            // on CUDA storage. Idempotent.
            CudaBackend.Register();

            CudaDriverApi.cuInit(0);
            CudaDriverApi.cuDeviceGetCount(out int devCount).ThrowOnError();
            int useDevs = nGpu > 0 ? Math.Min(nGpu, devCount) : devCount;
            if (useDevs < 1)
                throw new InvalidOperationException("No CUDA devices available for the DSV4 engine.");

            // A speculative verify writes KV for tokens that may be rejected. The
            // rejected tail is never restored -- instead every ring the next pass
            // still reads from is widened by the draft window, so a stale write
            // can no longer alias a live position (the raw ring already has
            // NUbatch of headroom; the compressor state rings do not).
            _maxDraft = m.Dspark != null ? m.Dspark.BlockSize : 0;
            _ringRaw = Pad(m.NSwa + m.NUbatch, 256);
            // V4.1 compresses disjoint blocks, so a head aligned to every ratio reads nothing a
            // rewind dropped; a query at the new head reads the raw window (head - n_swa, head],
            // which the ring still holds while the head moved back at most _rewindSpan.
            TruncateAlign = ComputeTruncateAlign(m);
            _rewindCheckpoint = m.V41;
            _rewindSpan = m.V41 ? Math.Max(0, _ringRaw - m.NSwa + 1) : 0;
            // V4.1 compresses at ratios 1 and 2 rather than 4 and 128, so the
            // same two row counts stand for a different pair of groups.
            _compRowsCsa = m.NCtx / (m.V41 ? 2 : CsaRatio) + 1;
            _compRowsHca = m.NCtx / (m.V41 ? 1 : HcaRatio) + 1;

            // ---- devices ----
            // Created before the layer split, which needs each device's free
            // VRAM (cuMemGetInfo wants a current context) to size its share.
            _devs = new Dev[useDevs];
            for (int d = 0; d < useDevs; d++)
            {
                var dev = new Dev { Ordinal = d };
                dev.Alloc = new CudaAllocator(d);
                dev.MakeCurrent();
                dev.DK = Dsv4Kernels.Create();
                CudaDriverApi.cuEventCreate(out IntPtr ev, 0x02 /*CU_EVENT_DISABLE_TIMING*/).ThrowOnError();
                dev.Event = ev;
                _devs[d] = dev;
            }

            // ---- layer placement: contiguous ranges balanced by quantized bytes ----
            // V4.1's Engram tables (tens of GiB) go to VRAM with their layers when the split holds
            // them without offloading more experts, as ggml_cuda places them; the host gathers the
            // rows a token selects from its mapping otherwise.
            var layerExpBytes = new long[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = m.Layers[il];
                layerExpBytes[il] = L.GateExps.TotalBytes + L.UpExps.TotalBytes + L.DownExps.TotalBytes;
            }
            long[] LayerBytes()
            {
                var bytes = new long[m.NLayer];
                for (int il = 0; il < m.NLayer; il++)
                    foreach (var qw in EnumerateQuantWeights(m.Layers[il]))
                        bytes[il] += qw.TotalBytes;
                return bytes;
            }

            long dsparkBytes = DsparkBytes(m.Dspark);
            var assignment = new int[m.NLayer];
            EngramResident = m.V41 && Array.Exists(m.Layers, L => L.EngramTable.IsValid);
            long[] layerBytes;
            if (EngramResident)
            {
                var hostAssignment = new int[m.NLayer];
                int hostOffload;
                EngramResident = false;
                try { hostOffload = PlaceLayers(m, useDevs, LayerBytes(), layerExpBytes, dsparkBytes, nCpuMoe, hostAssignment); }
                catch (InvalidOperationException) { hostOffload = int.MaxValue; }
                EngramResident = true;
                try
                {
                    _nCpuMoe = PlaceLayers(m, useDevs, LayerBytes(), layerExpBytes, dsparkBytes, nCpuMoe, assignment);
                    EngramResident = _nCpuMoe <= hostOffload;
                }
                catch (InvalidOperationException)
                {
                    EngramResident = false;
                }
                if (!EngramResident)
                {
                    if (hostOffload == int.MaxValue)
                        _nCpuMoe = PlaceLayers(m, useDevs, LayerBytes(), layerExpBytes, dsparkBytes, nCpuMoe, assignment);
                    else
                    {
                        Array.Copy(hostAssignment, assignment, assignment.Length);
                        _nCpuMoe = hostOffload;
                    }
                    Console.Error.WriteLine("[dsv4-cuda] Engram tables stay host mappings: the split cannot hold them " +
                        "beside the model without offloading more experts");
                }
                layerBytes = LayerBytes();
            }
            else
            {
                layerBytes = LayerBytes();
                _nCpuMoe = PlaceLayers(m, useDevs, layerBytes, layerExpBytes, dsparkBytes, nCpuMoe, assignment);
            }
            long totalBytes = 0;
            foreach (long b in layerBytes)
                totalBytes += b;
            _hostMatMul = m.HostMatMul;
            if (_nCpuMoe > 0 && _hostMatMul == null)
                throw new InvalidOperationException(
                    "[dsv4-cuda] routed-expert CPU offload is required to fit this model but no host matmul was " +
                    "supplied (ModelDesc.HostMatMul).");
            _lastDev = useDevs - 1;
            PlanBoundaries(m, assignment);

            // per-device boundary staging (pinned + events)
            long xsBytes = (long)m.NUbatch * HC * m.NEmbd * 4;
            for (int d = 0; d < useDevs; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                CudaDriverApi.cuMemHostAlloc(out IntPtr pinnedXs, new UIntPtr((ulong)xsBytes), 0x1 /*PORTABLE*/).ThrowOnError();
                dev.BoundaryPinned = pinnedXs;
                CudaDriverApi.cuEventCreate(out IntPtr xr, 0x02).ThrowOnError();
                CudaDriverApi.cuEventCreate(out IntPtr cd, 0x02).ThrowOnError();
                dev.XsReadyEv = xr;
                dev.CopyDoneEv = cd;
            }

            // ---- arena sizing per device (packed quantized weights only) ----
            // Norms, gates, APE/RoPE tables and the caches are separate allocator
            // tensors, so they are not counted here.
            var arenaNeed = new long[useDevs];
            for (int il = 0; il < m.NLayer; il++)
            {
                int d = assignment[il];
                foreach (var qw in EnumerateQuantWeights(m.Layers[il], skipRoutedExperts: il < _nCpuMoe))
                    arenaNeed[d] += Align(qw.TotalBytes);
            }
            arenaNeed[0] += Align(m.TokEmbd.TotalBytes);
            arenaNeed[_lastDev] += Align(m.Output.TotalBytes) + dsparkBytes;

            for (int d = 0; d < useDevs; d++)
            {
                _devs[d].MakeCurrent();
                _devs[d].Weights = new CudaWeightArena(_devs[d].Alloc, arenaNeed[d]);
            }

            // ---- upload weights (parallel across devices) ----
            _layers = new DevLayer[m.NLayer];
            _hostMoe = new HostMoeLayer[m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
                _layers[il] = new DevLayer { Device = assignment[il], Ratio = m.Layers[il].Ratio };

            var perDevLayers = new List<int>[useDevs];
            for (int d = 0; d < useDevs; d++)
                perDevLayers[d] = new List<int>();
            for (int il = 0; il < m.NLayer; il++)
                perDevLayers[assignment[il]].Add(il);

            var uploadSw = Stopwatch.StartNew();
            Parallel.For(0, useDevs, d =>
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                foreach (int il in perDevLayers[d])
                    UploadLayer(dev, il);
                if (d == 0)
                    _tokEmbdQW = dev.Weights.Place(m.TokEmbd);
                if (d == _lastDev)
                {
                    _outputQW = dev.Weights.Place(m.Output);
                    _outputNorm = UploadF32(dev, m.OutputNorm);
                    _hcHeadFn = UploadF32(dev, m.HcHeadFn);
                    _hcHeadScale = UploadF32(dev, m.HcHeadScale);
                    _hcHeadBase = UploadF32(dev, m.HcHeadBase);
                }
                dev.RopeRaw = UploadF32(dev, m.RopeRawTable);
                dev.RopeComp = UploadF32(dev, m.RopeCompTable);
            });

            // The drafter shares the loader pass: it plans its uploads here so
            // its (few GiB of) weights stream in with everything else.
            if (m.Dspark != null)
                SetupDspark(m.Dspark);

            // Weights sourced from disk were only *placed* above; move the bytes
            // now, with the reader concurrency the filesystem actually likes.
            CudaWeightArena.StreamAll(Array.ConvertAll(_devs, d => d.Weights), "dsv4-cuda");

            // ---- scratch + tokens + logits ----
            for (int d = 0; d < useDevs; d++)
                AllocateScratch(_devs[d]);

            _devs[0].NeedsTokens = true;
            for (int il = 0; il < Math.Min(m.HashLayerCount, m.NLayer); il++)
                _devs[assignment[il]].NeedsTokens = true;
            foreach (var dev in _devs)
            {
                if (!dev.NeedsTokens)
                    continue;
                dev.MakeCurrent();
                dev.TokensDev0 = AllocI32(dev, m.NUbatch);
                dev.TokensDev1 = AllocI32(dev, m.NUbatch);
                CudaDriverApi.cuEventCreate(out IntPtr te0, 0x02).ThrowOnError();
                CudaDriverApi.cuEventCreate(out IntPtr te1, 0x02).ThrowOnError();
                dev.TokEv0 = te0;
                dev.TokEv1 = te1;
            }

            var last = _devs[_lastDev];
            last.Logits = AllocF32(last, 1, m.NVocab);
            if (m.Dspark != null)
            {
                int specRows = m.Dspark.BlockSize + 1;
                _specLogits = AllocF32(last, specRows, m.NVocab);
            }

            _devs[0].MakeCurrent();
            // CU_MEMHOSTALLOC_PORTABLE (0x1): the pinned token buffers are copied
            // from by every device that needs token ids, not just device 0.
            CudaDriverApi.cuMemHostAlloc(out _pinnedTokens0, new UIntPtr((ulong)m.NUbatch * 4), 0x1).ThrowOnError();
            CudaDriverApi.cuMemHostAlloc(out _pinnedTokens1, new UIntPtr((ulong)m.NUbatch * 4), 0x1).ThrowOnError();
            long logitRows = m.Dspark != null ? m.Dspark.BlockSize + 1 : 1;
            CudaDriverApi.cuMemHostAlloc(out _pinnedLogits, new UIntPtr((ulong)(logitRows * m.NVocab * 4L)), 0x1).ThrowOnError();

            // Slot 0 serves the single-sequence path; the server allocates more per request.
            _active = CreateSlot();

            double gib = totalBytes / (1024.0 * 1024 * 1024);
            Console.Error.WriteLine(
                $"[dsv4-cuda] {gib:F1} GiB of weights resident across {useDevs} GPU(s) " +
                $"(layer split {string.Join("/", CountPerDev(assignment, useDevs))}), uploaded in {uploadSw.Elapsed.TotalSeconds:F1}s " +
                $"(n_ctx={m.NCtx}, ubatch={m.NUbatch})");
        }

        /// <summary>
        /// Places contiguous layer ranges so no device exceeds
        /// <paramref name="limit"/> bytes (the last device additionally carries
        /// <paramref name="extraLast"/>, the drafter). Every device gets at
        /// least one layer and the last device owns the tail, which is what the
        /// output head and the boundary hand-offs assume.
        /// </summary>
        /// <summary>
        /// Choose the routed-expert CPU offload count and the layer→device
        /// split, both sized against the VRAM each device actually has free.
        ///
        /// <para>Two things the old byte-balanced split could not do. It divided
        /// the weights evenly no matter what each card had free, so one device
        /// hosting a display (or another process) OOM'd on a split its siblings
        /// could have absorbed. And it had no answer at all when the model
        /// simply outweighed the cards — a 151 GiB V4 Flash checkpoint against
        /// 139 GiB of VRAM could only end in `CUDA error 2: out of memory`.
        /// Here every device is filled to the same FRACTION of its own budget,
        /// and when even a perfect split does not fit, the leading layers' routed
        /// experts (91% of the bytes) move to system RAM — the fewest that make
        /// it fit, since each one costs a host matmul per token.</para>
        /// </summary>
        /// <returns>Number of leading layers whose routed experts stay on the host.</returns>
        private int PlaceLayers(ModelDesc m, int nDev, long[] layerBytes, long[] layerExpBytes,
            long dsparkBytes, int nCpuMoeReq, int[] assignment)
        {
            int nLayer = m.NLayer;

            // Per-device budget: free VRAM now, minus the run-time residents the
            // split cannot attribute to a layer (per-device scratch, rope tables,
            // the logits staging and the allocator's own slack).
            long reserveMb = EnvInt("TS_DSV4_VRAM_RESERVE_MB", 2048);
            var budget = new long[nDev];
            var freeBytes = new long[nDev];
            for (int d = 0; d < nDev; d++)
            {
                _devs[d].MakeCurrent();
                CudaDriverApi.cuMemGetInfo(out UIntPtr free, out UIntPtr _).ThrowOnError();
                freeBytes[d] = (long)free.ToUInt64();
                long reserve = reserveMb * 1024 * 1024 + PerDeviceFixedBytes(m);
                budget[d] = Math.Max(freeBytes[d] - reserve, 0);
            }

            // Fixed residents the split places explicitly: embedding table on the
            // first device, output head + drafter on the last.
            var fixedBytes = new long[nDev];
            fixedBytes[0] += Align(m.TokEmbd.TotalBytes);
            fixedBytes[nDev - 1] += Align(m.Output.TotalBytes) + dsparkBytes;

            long Cost(int il, int nCpu)
            {
                long b = layerBytes[il];
                if (il < nCpu) b -= layerExpBytes[il];
                return Align(b) + LayerCacheBytes(m, il);
            }

            // Layers stay in pipeline order, so every device takes one contiguous
            // run: fill each up to `frac` of its budget and report whether all of
            // them fit.
            bool Pack(double frac, int nCpu, int[] outAssign)
            {
                int dev = 0;
                long used = fixedBytes[0];
                for (int il = 0; il < nLayer; il++)
                {
                    long cost = Cost(il, nCpu);
                    while (used + cost > (long)(budget[dev] * frac))
                    {
                        if (dev + 1 >= nDev)
                            return false;
                        used = fixedBytes[++dev];
                    }
                    used += cost;
                    if (outAssign != null)
                        outAssign[il] = dev;
                }
                return true;
            }

            int need = 0;
            while (need <= nLayer && !Pack(1.0, need, null))
                need++;

            if (need > nLayer)
            {
                long freeTotal = 0, expTotal = 0;
                for (int d = 0; d < nDev; d++) freeTotal += freeBytes[d];
                foreach (long b in layerExpBytes) expTotal += b;
                long weights = 0;
                foreach (long b in layerBytes) weights += b;
                throw new InvalidOperationException(
                    $"[dsv4-cuda] model does not fit: {(weights + dsparkBytes) / (double)(1L << 30):F1} GiB of weights " +
                    $"({expTotal / (double)(1L << 30):F1} GiB of them routed experts) against " +
                    $"{freeTotal / (double)(1L << 30):F1} GiB free across {nDev} device(s), even with every expert on " +
                    "the host. Free VRAM, add devices, or use a smaller quantization.");
            }

            int nCpuMoe;
            if (nCpuMoeReq < 0)
            {
                nCpuMoe = need;   // opt-in auto
            }
            else
            {
                nCpuMoe = Math.Min(nCpuMoeReq, nLayer);   // operator's choice
                if (nCpuMoe < need)
                {
                    // Decline instead of loading into a certain out-of-memory
                    // abort. Naming WHICH number would work is the whole value
                    // here: the operator cannot derive it from the model size,
                    // because what has to fit is the weights PLUS this context's
                    // KV caches.
                    long freeTotal = 0;
                    for (int d = 0; d < nDev; d++) freeTotal += freeBytes[d];
                    long weightBytes = dsparkBytes, wouldFree = 0;
                    foreach (long b in layerBytes) weightBytes += b;
                    for (int il = 0; il < need && il < nLayer; il++) wouldFree += layerExpBytes[il];
                    throw new InvalidOperationException(
                        $"[dsv4-cuda] not enough VRAM: {weightBytes / (double)(1L << 30):F1} GiB of weights plus " +
                        $"this context's KV caches against {freeTotal / (double)(1L << 30):F1} GiB free across " +
                        $"{nDev} device(s)" + (nCpuMoe > 0 ? " at the requested offload" : string.Empty) +
                        $". Re-run with --n-cpu-moe {need} (moves the routed experts of the first {need} layer(s), " +
                        $"{wouldFree / (double)(1L << 30):F1} GiB, to system RAM) or --cpu-moe to offload every layer.");
                }
            }

            // Balance: smallest peak budget fraction that still fits.
            double lo = 0.0, hi = 1.0;
            for (int i = 0; i < 40; i++)
            {
                double mid = 0.5 * (lo + hi);
                if (Pack(mid, nCpuMoe, null)) hi = mid; else lo = mid;
            }
            if (!Pack(hi, nCpuMoe, assignment))
                throw new InvalidOperationException("[dsv4-cuda] cannot split the model across the visible GPUs");

            if (nCpuMoe > 0)
            {
                long host = 0;
                for (int il = 0; il < nCpuMoe; il++) host += layerExpBytes[il];
                Console.Error.WriteLine($"[dsv4-cuda] MoE CPU offload: routed experts of layers 0..{nCpuMoe - 1} " +
                    $"({host / (double)(1L << 30):F1} GiB) stay in system RAM and run on the host" +
                    (nCpuMoeReq < 0 ? " (auto: the model does not fit the visible VRAM otherwise)" : string.Empty));
            }
            return nCpuMoe;
        }

        /// <summary>
        /// Per-layer device memory beyond the packed weights: the raw SWA ring,
        /// the CSA/HCA compressed-K caches, the indexer cache and the compressor
        /// state rings, all allocated on the layer's own device. A split that
        /// ignores them fits the weights and then dies allocating the caches.
        /// </summary>
        private long LayerCacheBytes(ModelDesc m, int il)
        {
            int ratio = m.Layers[il].Ratio;
            int hd = m.HeadDim;
            long b = (long)_ringRaw * hd * 2;                 // RingK (F16)
            if (m.V41)
            {
                // Every V4.1 slot also keeps the rewind checkpoint's shadows of the modular
                // rings, and only a ratio group's source layer owns the group's caches.
                b *= 2;
                if (ratio != 0 && m.Layers[il].KvSource == il)
                {
                    long rows = V41Rows(ratio);
                    b += rows * hd * 2 + rows * m.IdxHeadSize * 2;       // CompK + LidK (F16)
                    if (ratio > 1)
                        b += 2L * 2 * V41StateRows(ratio) * hd * 4;      // Hist kv+score and shadows (F32)
                }
                return b;
            }
            if (ratio == CsaRatio)
            {
                b += (long)_compRowsCsa * hd * 2;             // CompK  (F16)
                b += (long)_compRowsCsa * m.IdxHeadSize * 2;  // LidK   (F16)
                b += 2L * (2 * CsaRatio + _maxDraft) * 2 * hd * 4;                // Hist kv+score (F32)
                b += 2L * (2 * CsaRatio + _maxDraft) * 2 * m.IdxHeadSize * 4;     // LidHist kv+score (F32)
            }
            else if (ratio == HcaRatio)
            {
                b += (long)_compRowsHca * hd * 2;             // CompK  (F16)
                b += 2L * (HcaRatio + _maxDraft) * hd * 4;                        // Hist kv+score (F32)
            }
            return b;
        }

        /// <summary>Device-resident scratch that every device carries regardless
        /// of how many layers it hosts (rope tables + the per-ubatch working
        /// set sized in <see cref="AllocateScratch"/>).</summary>
        private static long PerDeviceFixedBytes(ModelDesc m)
        {
            long nt = m.NUbatch;
            long e = m.NEmbd;
            long s = nt * m.NExpertUsed;
            long b = 2L * m.NCtx * m.NRot * 4;            // rope raw + comp tables
            b += 2 * nt * HC * e * 4;                     // Xs / XsOut
            b += 2 * s * m.NFfExp * 4;                    // ExpGate / ExpUp
            b += s * e * 4;                               // ExpDown
            b += 4 * nt * (long)m.NHead * m.HeadDim * 4;  // Q / AttnO / OGrouped (+slack)
            return b;
        }

        private static int[] CountPerDev(int[] assignment, int nDev)
        {
            var counts = new int[nDev];
            foreach (int d in assignment)
                counts[d]++;
            return counts;
        }

        private static int EnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        private static int Pad(int v, int p) => (v + p - 1) / p * p;
        private static long Align(long v) => (v + 255) & ~255L;

        /// <param name="skipRoutedExperts"><c>--n-cpu-moe</c>: this layer's
        /// stacked expert tensors never reach VRAM, so they must not be counted
        /// into the arena or planned for upload either.</param>
        private IEnumerable<CudaWeightDesc> EnumerateQuantWeights(LayerDesc l, bool skipRoutedExperts = false)
        {
            yield return l.WqA;
            yield return l.WqB;
            yield return l.Wkv;
            yield return l.WoA;
            yield return l.WoB;
            if (l.CompWkv.IsValid) yield return l.CompWkv;
            if (l.CompWgate.IsValid) yield return l.CompWgate;
            if (l.IdxProj.IsValid) yield return l.IdxProj;
            if (l.IdxQB.IsValid) yield return l.IdxQB;
            if (l.IdxCompWkv.IsValid) yield return l.IdxCompWkv;
            if (l.IdxCompWgate.IsValid) yield return l.IdxCompWgate;
            // V4.1 only, and only on the layers that carry them. This enumeration
            // sizes the arena as well as filling it, so a weight missing here
            // overflows the arena rather than being quietly skipped.
            if (l.IndexerK.IsValid) yield return l.IndexerK;
            if (l.EngramWkv.IsValid) yield return l.EngramWkv;
            if (EngramResident && l.EngramTable.IsValid) yield return l.EngramTable;
            if (!skipRoutedExperts)
            {
                yield return l.GateExps;
                yield return l.UpExps;
                yield return l.DownExps;
            }
            yield return l.GateShexp;
            yield return l.UpShexp;
            yield return l.DownShexp;
        }

        /// <summary>Small dense tensor (norm/gate/table) as an allocator-owned
        /// F32 tensor. Null/empty input yields null, which every consumer treats
        /// as "absent".</summary>
        private static Tensor UploadF32(Dev dev, float[] data)
        {
            if (data == null || data.Length == 0)
                return null;
            Tensor t = AllocF32(dev, data.Length);
            fixed (float* src = data)
                CudaDriverApi.cuMemcpyHtoD(Ptr(t), (IntPtr)src, new UIntPtr((ulong)data.Length * 4)).ThrowOnError();
            return t;
        }

        private static Tensor UploadI32(Dev dev, int[] data)
        {
            if (data == null || data.Length == 0)
                return null;
            Tensor t = AllocI32(dev, data.Length);
            fixed (int* src = data)
                CudaDriverApi.cuMemcpyHtoD(Ptr(t), (IntPtr)src, new UIntPtr((ulong)data.Length * 4)).ThrowOnError();
            return t;
        }

        private void UploadLayer(Dev dev, int il)
        {
            var src = _m.Layers[il];
            var dst = _layers[il];
            dst.ClampExp = src.ClampExp;
            dst.ClampShexp = src.ClampShexp;

            dst.WqA = dev.Weights.Place(src.WqA);
            dst.WqB = dev.Weights.Place(src.WqB);
            dst.Wkv = dev.Weights.Place(src.Wkv);
            dst.WoA = dev.Weights.Place(src.WoA);
            dst.WoB = dev.Weights.Place(src.WoB);
            dst.CompWkv = dev.Weights.Place(src.CompWkv);
            dst.CompWgate = dev.Weights.Place(src.CompWgate);
            dst.IdxProj = dev.Weights.Place(src.IdxProj);
            dst.IdxQB = dev.Weights.Place(src.IdxQB);
            dst.IdxCompWkv = dev.Weights.Place(src.IdxCompWkv);
            dst.IdxCompWgate = dev.Weights.Place(src.IdxCompWgate);
            // V4.1: the indexer's K projection off the compressed latent, and the
            // Engram projection. Both are default-valued on layers that do not
            // carry them, which UploadQuant passes through unchanged.
            dst.IndexerK = dev.Weights.Place(src.IndexerK);
            dst.EngramWkv = dev.Weights.Place(src.EngramWkv);
            if (EngramResident)
                dst.EngramTable = dev.Weights.Place(src.EngramTable);
            if (il < _nCpuMoe)
            {
                // --n-cpu-moe: the stacked experts stay in system RAM and their
                // FFN runs on the host (Dsv4CudaEngine.HostMoe.cs). Leaving the
                // DeviceWeight entries invalid is deliberate — MoeFfn dispatches on
                // _hostMoe[il], and a stray device read would be a null deref
                // rather than silently wrong output.
                _hostMoe[il] = LoadHostExperts(src);
            }
            else
            {
                dst.GateExps = dev.Weights.Place(src.GateExps);
                dst.UpExps = dev.Weights.Place(src.UpExps);
                dst.DownExps = dev.Weights.Place(src.DownExps);
            }
            dst.GateShexp = dev.Weights.Place(src.GateShexp);
            dst.UpShexp = dev.Weights.Place(src.UpShexp);
            dst.DownShexp = dev.Weights.Place(src.DownShexp);
            dst.ShFf = src.UpShexp.Ne1;

            dst.AttnNorm = UploadF32(dev, src.AttnNorm);
            dst.QANorm = UploadF32(dev, src.QANorm);
            dst.KvNorm = UploadF32(dev, src.KvNorm);
            dst.Sinks = UploadF32(dev, src.Sinks);
            dst.HcAttnFn = UploadF32(dev, src.HcAttnFn);
            dst.HcAttnScale = UploadF32(dev, src.HcAttnScale);
            dst.HcAttnBase = UploadF32(dev, src.HcAttnBase);
            dst.HcFfnFn = UploadF32(dev, src.HcFfnFn);
            dst.HcFfnScale = UploadF32(dev, src.HcFfnScale);
            dst.HcFfnBase = UploadF32(dev, src.HcFfnBase);
            dst.CompApe = UploadF32(dev, src.CompApe);
            dst.CompNorm = UploadF32(dev, src.CompNorm);
            dst.IdxCompApe = UploadF32(dev, src.IdxCompApe);
            dst.IdxCompNorm = UploadF32(dev, src.IdxCompNorm);
            dst.GateInp = UploadF32(dev, src.GateInp);
            dst.ExpProbsBias = UploadF32(dev, src.ExpProbsBias);
            dst.FfnNorm = UploadF32(dev, src.FfnNorm);
            dst.Tid2Eid = UploadI32(dev, src.Tid2Eid);

            // The caches are each sequence's own (AllocSlotLayer).
            if (_m.V41)
            {
                dst.KvSource = src.KvSource;
                dst.IndexSource = src.IndexSource;
                dst.EngramIndex = src.EngramIndex;
                dst.IndexerKNorm = UploadF32(dev, src.IndexerKNorm);
                dst.EngramQ = UploadF32(dev, src.EngramQ);
                dst.EngramK = UploadF32(dev, src.EngramK);
            }
        }

        /// <summary>Device pointer of a contiguous allocator-owned tensor (or of
        /// its first element when the tensor is a row view of a larger buffer).</summary>
        internal static IntPtr Ptr(Tensor t)
            => t == null ? IntPtr.Zero : ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

        /// <summary>
        /// Scratch tensor from the device's <see cref="CudaAllocator"/>. Replaces
        /// the engine's former private cuMemAlloc list: pooled, counted by the
        /// allocator's VRAM stats, and usable directly by the shared Ops.
        /// </summary>
        private static Tensor AllocT(Dev dev, DType type, params long[] sizes)
        {
            dev.MakeCurrent();
            var t = new Tensor(dev.Alloc, type, sizes);
            dev.OwnedTensors.Add(t);
            return t;
        }

        /// <summary>Compressed rows a V4.1 ratio group needs for the whole
        /// context. Ratio 1 keeps one row per token, ratio 2 one per pair.</summary>
        private int V41Rows(int ratio) => ratio == 2 ? _compRowsCsa : _compRowsHca;

        private static Tensor AllocF32(Dev dev, params long[] sizes) => AllocT(dev, DType.Float32, sizes);

        private static Tensor AllocI32(Dev dev, params long[] sizes) => AllocT(dev, DType.Int32, sizes);

        /// <summary>Raw byte scratch (quantized activation blocks). Rounded up to
        /// whole rows so the tensor stays 2-D and contiguous.</summary>
        private static Tensor AllocU8(Dev dev, long rows, long rowBytes) => AllocT(dev, DType.UInt8, Math.Max(rows, 1), Math.Max(rowBytes, 1));

        /// <summary>
        /// Row view [rows, cols] over a scratch tensor allocated for the largest
        /// ubatch. Cheap (no device work) and disposed by the caller; used to hand
        /// the shared Ops the exact shape of the current chunk.
        /// </summary>
        private static Tensor Rows(Tensor t, int rows)
            => t.Sizes[0] == rows ? t.CopyRef() : t.Narrow(0, 0, rows);

        /// <summary>Row view of <paramref name="rows"/> rows starting at
        /// <paramref name="first"/>, reshaped to [rows, cols].</summary>
        private static Tensor Block(Tensor t, long first, long rows, long cols)
        {
            using Tensor flat = t.View(t.ElementCount());
            using Tensor slice = flat.Narrow(0, first * cols, rows * cols);
            return slice.View(rows, cols);
        }

        private void AllocateScratch(Dev dev)
        {
            dev.MakeCurrent();
            var m = _m;
            int nt = m.NUbatch;
            int e = m.NEmbd;
            int hd = m.HeadDim;
            int s = nt * m.NExpertUsed;

            dev.Xs = AllocF32(dev, nt, HC * e);
            dev.XsOut = AllocF32(dev, nt, HC * e);
            dev.Cur = AllocF32(dev, nt, e);
            dev.Inv = AllocF32(dev, nt);
            dev.Mixes = AllocF32(dev, nt, HcMixDim);
            dev.HcPartials = AllocF32(dev, MaxBatchedDecodeRows, HC * m.NEmbd / Dsv4Kernels.HcSlice * Dsv4Kernels.HcPartialFloats);
            dev.Pre = AllocF32(dev, nt, HC);
            dev.Post = AllocF32(dev, nt, HC);
            dev.Comb = AllocF32(dev, nt, HC * HC);
            dev.Qr = AllocF32(dev, nt, m.QLoraRank);
            dev.Q = AllocF32(dev, nt, (long)m.NHead * hd);
            dev.KvRaw = AllocF32(dev, nt, hd);
            dev.StKv = AllocF32(dev, nt, 2 * hd);
            dev.StScore = AllocF32(dev, nt, 2 * hd);
            dev.LidStKv = AllocF32(dev, nt, 2 * m.IdxHeadSize);
            dev.LidStScore = AllocF32(dev, nt, 2 * m.IdxHeadSize);
            dev.Iq = AllocF32(dev, nt, (long)m.IdxNHead * m.IdxHeadSize);
            dev.Iw = AllocF32(dev, nt, Math.Max(m.IdxNHead, 1));
            dev.IdxScores = AllocF32(dev, nt, Math.Max(_compRowsCsa, m.V41 ? _compRowsHca : 0));
            dev.TopkIdx = AllocI32(dev, nt, Math.Max(m.IdxTopK, 1));
            dev.TopkCnt = AllocI32(dev, nt);
            dev.AttnO = AllocF32(dev, nt, (long)m.NHead * hd);
            dev.OGrouped = AllocF32(dev, nt, (long)m.NHead * hd);
            dev.OGroupedOut = AllocF32(dev, (long)m.OGroups * nt, m.OLoraRank);
            dev.OG = AllocF32(dev, nt, (long)m.OGroups * m.OLoraRank);
            dev.AttnOut = AllocF32(dev, nt, e);
            if (m.V41)
            {
                // The delayed gates: each block computes its own but collapses
                // with the previous block's, so two live buffers, not one.
                dev.PreAttn = AllocF32(dev, nt, HC);
                dev.PreFfn = AllocF32(dev, nt, HC);
                // One extra row so a ratio-1 group can hold every token's block.
                dev.Latent = AllocF32(dev, nt + 1, hd);
                dev.LatentK = AllocF32(dev, nt + 1, m.IdxHeadSize);
                if (m.CandidateSource >= 0)
                    dev.CandMask = AllocT(dev, DType.UInt8, nt, Math.Max(_compRowsCsa, _compRowsHca));
                if (m.Engram != null)
                {
                    long cols = (long)m.Engram.HashColumns * m.Engram.HeadDim;
                    dev.EngramLookup = AllocF32(dev, nt, cols);
                    dev.EngramKv = AllocF32(dev, nt, (long)(HC + 1) * e);
                    if (EngramResident)
                        dev.EngramRows = AllocI32(dev, nt, m.Engram.HashColumns);
                    foreach (DevLayer layer in _layers)
                    {
                        if (layer.Device != dev.Ordinal || layer.EngramIndex < 0)
                            continue;
                        CudaDriverApi.cuMemHostAlloc(out layer.EngramPinned, new UIntPtr((ulong)(nt * cols * 4)), 0x1 /*PORTABLE*/).ThrowOnError();
                        CudaDriverApi.cuEventCreate(out layer.EngramCopied, 0x02 /*DISABLE_TIMING*/).ThrowOnError();
                    }
                }
            }
            dev.FfnOut = AllocF32(dev, nt, e);
            dev.GemvScratch = AllocF32(dev, Dsv4Kernels.GemvScratchFloats);
            dev.RouterLogits = AllocF32(dev, nt, m.NExpert);
            dev.Sel = AllocI32(dev, nt, m.NExpertUsed);
            dev.SelW = AllocF32(dev, nt, m.NExpertUsed);
            // Counts/Offsets/Cursors are int32 histograms; Counts is cleared with
            // the F32 fill kernel because 0.0f and 0 share the same bit pattern
            // and that keeps the clear stream-ordered with the grouping kernels.
            dev.Counts = AllocI32(dev, m.NExpert);
            dev.Offsets = AllocI32(dev, m.NExpert);
            dev.Cursors = AllocI32(dev, m.NExpert);
            dev.RowOfSlot = AllocI32(dev, s);
            dev.SlotToken = AllocI32(dev, s);

            // q8_1 activation scratch: A covers [max(nt, S)] rows of the widest
            // quantized input (oGroups*oLoraRank for wo_b); B covers the packed
            // expert hidden rows for the down projection.
            int maxIn = Math.Max(Math.Max(e, m.OGroups * m.OLoraRank), Math.Max(m.QLoraRank, m.NFfExp));
            long q8RowBytesA = (long)(maxIn / 32) * Q81BlockBytes;
            long rowsA = Math.Max(Math.Max(nt, s), (long)m.OGroups * nt);
            dev.ActQ8A = AllocU8(dev, rowsA, q8RowBytesA);
            dev.ActQ8B = AllocU8(dev, s, (long)(m.NFfExp / 32) * Q81BlockBytes);

            // split-layout twins for the staged expert kernels
            dev.SplitQsA = AllocU8(dev, nt, e);
            dev.SplitDA = AllocF32(dev, nt, e / 32);
            dev.SplitQsB = AllocU8(dev, s, m.NFfExp);
            dev.SplitDB = AllocF32(dev, s, m.NFfExp / 32);

            dev.ExpGate = AllocF32(dev, s, m.NFfExp);
            dev.ExpUp = AllocF32(dev, s, m.NFfExp);
            dev.ExpDown = AllocF32(dev, s, e);
            int shFf = 0;
            foreach (var dl in _layers)
                if (dl.Device == dev.Ordinal && dl.ShFf > shFf)
                    shFf = dl.ShFf;
            if (_ds != null && _ds.Dev.Ordinal == dev.Ordinal)
                foreach (var st in _ds.Stages)
                    if (st.ShFf > shFf)
                        shFf = st.ShFf;
            shFf = Math.Max(shFf, 1);
            dev.ShGate = AllocF32(dev, nt, shFf);
            dev.ShUp = AllocF32(dev, nt, shFf);
            dev.ShDown = AllocF32(dev, nt, e);

            // The F16 weight-dequant and F16/BF16 activation panels the prefill
            // GEMMs need are NOT allocated here any more: CudaQuantizedOps owns
            // one grow-on-demand scratch per allocator and every matmul on this
            // device shares it (see RunResidentMatmul / RunF16Gemm).
        }

        // -------------------------------------------------------------------
        // Reset / caches
        // -------------------------------------------------------------------

        /// <summary>Clears the active slot back to position 0. Weights and other slots are untouched.</summary>
        public void Reset()
        {
            ClearSlot(_active);
            _v41CandActive = false;
        }

        private static void Memset0(Tensor t)
        {
            if (t == null)
                return;
            long bytes = t.ElementCount() * t.ElementType.Size();
            if (bytes > 0)
                CudaDriverApi.cuMemsetD8(Ptr(t), 0, new UIntPtr((ulong)bytes)).ThrowOnError();
        }

        // -------------------------------------------------------------------
        // Forward
        // -------------------------------------------------------------------

        public void Forward(int[] tokens, float[] logitsOut)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            Slot slot = _active;
            if (slot.Failed)
                throw new InvalidOperationException("[dsv4-cuda] the sequence's last forward failed; reset or free its slot before reuse");
            if (slot.NPast + tokens.Length > _m.NCtx)
                throw new InvalidOperationException($"[dsv4-cuda] context overflow: n_past={slot.NPast} + {tokens.Length} > n_ctx={_m.NCtx}");

            var sw = Stopwatch.StartNew();
            long start = Stopwatch.GetTimestamp();
            slot.Failed = true;
            int done = 0;
            while (done < tokens.Length)
            {
                int nt = Math.Min(_m.NUbatch, tokens.Length - done);
                bool last = done + nt == tokens.Length;
                ForwardUbatch(tokens, done, nt, slot.NPast, last ? logitsOut : null, false, null, 0);
                slot.NPast += nt;
                done += nt;
            }
            slot.Failed = false;
            // Only a multi-token forward ends at a prompt boundary.
            if (tokens.Length > 1)
                CheckpointActiveSlot();
            if (_perf > 0)
            {
                double secs = sw.Elapsed.TotalSeconds;
                Console.Error.WriteLine($"[dsv4-cuda] forward {tokens.Length} tokens in {secs:F3}s ({tokens.Length / secs:F1} tok/s), "
                    + $"host issue {IssueMs(start):F1}ms");
            }
        }

        private void CheckSync(Dev dev, string stage)
        {
            if (!_syncDebug)
                return;
            int rc = CudaDriverApi.cuStreamSynchronize(dev.Stream);
            if (rc != 0)
                throw new InvalidOperationException($"[dsv4-cuda] CUDA error {rc} after {stage} on device {dev.Ordinal}");
        }

        // TS_DSV4_PERF=2 / 3: per-stage time, synchronized host time or GPU time (see CudaStageTimer).
        private static readonly string[] StageNames =
        {
            "embed", "hc", "attnproj", "comp", "idx", "attncore", "outproj",
            "router", "experts", "shexp", "lmhead", "boundary",
        };
        // TS_DSV4_PERF>=1: when the host had queued the whole step (the last launch before the
        // logits sync). Close to the step's wall time = the step is bound by launch overhead.
        private long _issueEnd;

        private double IssueMs(long start)
            => _issueEnd > start ? (_issueEnd - start) * 1000.0 / Stopwatch.Frequency : 0;

        private void StageBegin(Dev dev) => _stages.SpanStart(dev.Alloc.Context, dev.Stream);

        private void StageEnd(Dev dev, int stage) => _stages.End(dev.Alloc.Context, dev.Stream, stage);

        // TS_DSV4_CUDA_DEBUG=1: print the first values of layer-0 stage outputs
        // (paired with TS_DSV4_CPU_DEBUG=1 prints in DeepSeek4CpuExecutor for
        // stage-by-stage A/B against the exact managed reference).
        private static readonly bool StageDebug = EnvInt("TS_DSV4_CUDA_DEBUG", 0) != 0;

        /// <summary>
        /// TS_DSV4_CUDA_TRACE_DIR writes whole tensors under the same names the
        /// managed executor writes with TS_DSV4_CPU_TRACE_DIR, so the two
        /// directories can be diffed tensor by tensor. That is the only tractable
        /// way to find which layer of a forty-layer graph first disagrees.
        /// </summary>
        private static readonly string TraceDir = Environment.GetEnvironmentVariable("TS_DSV4_CUDA_TRACE_DIR");
        private int _traceP0;

        private void Trace(Dev dev, string name, Tensor t, long count)
        {
            if (TraceDir == null || t == null)
                return;
            System.IO.Directory.CreateDirectory(TraceDir);
            dev.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(dev.Stream);
            var host = new float[count];
            fixed (float* h = host)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)h, Ptr(t), new UIntPtr((ulong)count * 4)).ThrowOnError();
            var bytes = new byte[count * 4];
            Buffer.BlockCopy(host, 0, bytes, 0, bytes.Length);
            System.IO.File.WriteAllBytes(System.IO.Path.Combine(TraceDir, $"p{_traceP0:D6}_{name}.f32"), bytes);
        }

        private static string TraceLayer(int il, string what) => $"blk{il:D2}_{what}";

        private void Dump(Dev dev, string label, Tensor t, int n = 6)
        {
            if (!StageDebug || t == null)
                return;
            IntPtr ptr = Ptr(t);
            dev.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(dev.Stream);
            var tmp = new float[n];
            fixed (float* p = tmp)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, ptr, new UIntPtr((ulong)n * 4));
            Console.Error.WriteLine($"[dbg-cuda] {label}: {string.Join(" ", Array.ConvertAll(tmp, v => v.ToString("G6")))}");
        }

        private void DumpF16(Dev dev, string label, Tensor t, int n = 6)
        {
            if (!StageDebug || t == null)
                return;
            IntPtr ptr = Ptr(t);
            dev.MakeCurrent();
            CudaDriverApi.cuStreamSynchronize(dev.Stream);
            var tmp = new ushort[n];
            fixed (ushort* p = tmp)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, ptr, new UIntPtr((ulong)n * 2));
            var vals = new float[n];
            for (int i = 0; i < n; i++)
                vals[i] = (float)BitConverter.UInt16BitsToHalf(tmp[i]);
            Console.Error.WriteLine($"[dbg-cuda] {label}: {string.Join(" ", Array.ConvertAll(vals, v => v.ToString("G6")))}");
        }

        /// <param name="allLogitsRows">Emit LM-head logits for every row (the
        /// speculative verify) instead of only the last.</param>
        /// <param name="hAllOut">When set, receives the DSpark target features of
        /// every row, starting at row <paramref name="hRowOff"/>.</param>
        /// <param name="batch">A batched decode step: row i is its own sequence at its
        /// own position, and <paramref name="p0"/> is unused. Null for a ubatch of the
        /// active slot at positions p0, p0 + 1, ...</param>
        private void ForwardUbatch(int[] tokens, int tokOff, int nt, int p0, float[] logitsOut,
            bool allLogitsRows, float[] hAllOut, int hRowOff, DecodeBatch batch = null)
        {
            var m = _m;
            int e = m.NEmbd;

            // tokens -> pinned buffer (parity-double-buffered) -> devices that need
            // them. Per-device parity events guard reuse of the pinned slot two
            // chunks later without forcing a stream sync.
            int parity = _chunkParity;
            _chunkParity ^= 1;
            IntPtr pinned = parity == 0 ? _pinnedTokens0 : _pinnedTokens1;
            foreach (var dev in _devs)
            {
                if (!dev.NeedsTokens)
                    continue;
                CudaDriverApi.cuEventSynchronize(parity == 0 ? dev.TokEv0 : dev.TokEv1);
            }
            Marshal.Copy(tokens, tokOff, pinned, nt);
            foreach (var dev in _devs)
            {
                if (!dev.NeedsTokens)
                    continue;
                dev.MakeCurrent();
                Tensor dst = parity == 0 ? dev.TokensDev0 : dev.TokensDev1;
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dst), pinned, new UIntPtr((ulong)nt * 4), dev.Stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(parity == 0 ? dev.TokEv0 : dev.TokEv1, dev.Stream).ThrowOnError();
            }
            Tensor TokensOf(Dev dev) => parity == 0 ? dev.TokensDev0 : dev.TokensDev1;

            // A candidate mask belongs to one ubatch; a stale one would prune the
            // next ubatch's queries against the previous ubatch's scores.
            _v41CandActive = false;
            _committed.Clear();
            _traceP0 = p0;
            if (batch != null)
                StagePositions(batch);
            if (m.V41 && m.Engram != null)
            {
                if (batch == null)
                {
                    m.Engram.BeginEngramUbatch(_active.Engram, new ReadOnlySpan<int>(tokens, tokOff, nt), p0);
                }
                else
                {
                    var histories = new Dsv41EngramHistory[nt];
                    for (int i = 0; i < nt; i++)
                        histories[i] = batch.Rows[i].Engram;
                    m.Engram.BeginEngramRows(histories, new ReadOnlySpan<int>(tokens, tokOff, nt), batch.Positions);
                }
            }

            // embedding on device 0
            var dev0 = _devs[0];
            dev0.MakeCurrent();
            StageBegin(dev0);
            dev0.DK.Embed(_tokEmbdQW.Ptr, TokensOf(dev0), dev0.Xs, _tokEmbdQW.Type, _tokEmbdQW.RowBytes, nt, e, dev0.Stream);
            StageEnd(dev0, 0);
            CheckSync(dev0, "embed");
            Dump(dev0, "embed.xs", dev0.Xs);
            Trace(dev0, "embedding", dev0.Xs, (long)nt * HC * e);

            int curDev = 0;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                if (L.Device != curDev)
                {
                    // Hand the hidden streams to the next device, staged through
                    // pinned host memory (peer DMA lies on this hardware class —
                    // see the BoundaryPinned comment). Event chain: DtoH on the
                    // src stream -> dst waits XsReady -> HtoD on the dst stream
                    // -> src waits CopyDone before anything may overwrite its Xs
                    // or the pinned slice (next chunk's work).
                    var src = _devs[curDev];
                    var dst = _devs[L.Device];
                    var copyBytes = new UIntPtr((ulong)((long)nt * HC * e * 4));
                    // Each event is RECORDED on a stream in its own context (the
                    // driver rejects a cross-context record with "invalid
                    // resource handle"); only the WAITS cross contexts, which
                    // is the supported multi-GPU pattern.
                    src.MakeCurrent();
                    CudaDriverApi.cuMemcpyDtoHAsync(src.BoundaryPinned, Ptr(src.Xs), copyBytes, src.Stream).ThrowOnError();
                    var state = StageBoundaryState(src, dst, nt);
                    StageEnd(src, 11);
                    CudaDriverApi.cuEventRecord(src.XsReadyEv, src.Stream).ThrowOnError();
                    dst.MakeCurrent();
                    CudaDriverApi.cuStreamWaitEvent(dst.Stream, src.XsReadyEv, 0).ThrowOnError();
                    StageBegin(dst);
                    CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dst.Xs), src.BoundaryPinned, copyBytes, dst.Stream).ThrowOnError();
                    UnstageBoundaryState(src, dst, state);
                    CudaDriverApi.cuEventRecord(dst.CopyDoneEv, dst.Stream).ThrowOnError();
                    src.MakeCurrent();
                    CudaDriverApi.cuStreamWaitEvent(src.Stream, dst.CopyDoneEv, 0).ThrowOnError();
                    curDev = L.Device;
                    StageEnd(dst, 11);
                }

                var dev = _devs[curDev];
                dev.MakeCurrent();

                // ---- Engram (V4.1, on the layers the GGUF metadata names) ----
                // Runs before the attention block and rewrites the residual in
                // place, which is where the reference puts it.
                if (L.EngramIndex >= 0)
                    EngramLayer(dev, L, nt);

                // ---- attention super-block ----
                // V4.1 delays the hyper-connection collapse by one block: each
                // block computes its own gates but collapses the streams with the
                // previous block's, and the first block uses the stream mean.
                bool dbg = StageDebug && il == 0;
                HcPre(dev, L, nt, attn: true,
                    delayed: m.V41 ? (il == 0 ? null : dev.PreFfn) : null,
                    publish: m.V41 ? dev.PreAttn : null,
                    meanCollapse: m.V41 && il == 0,
                    normW: dbg ? null : L.AttnNorm);
                if (dbg)
                {
                    Dump(dev, "L0.attn.mixes", dev.Mixes);
                    Dump(dev, "L0.attn.pre", dev.Pre, 4);
                    Dump(dev, "L0.attn.comb", dev.Comb, 8);
                    Dump(dev, "L0.attn.cur", dev.Cur);
                    RmsNorm(dev, dev.Cur, L.AttnNorm, nt);
                }
                StageEnd(dev, 1);
                if (dbg)
                    Dump(dev, "L0.attn.cur_norm", dev.Cur);
                Trace(dev, TraceLayer(il, "attn_input"), dev.Cur, (long)nt * e);
                CheckSync(dev, $"hc_pre_attn L{il}");
                if (batch != null)
                    AttentionRows(dev, L, il, nt, batch);
                else if (m.V41)
                    AttentionV41(dev, L, il, nt, p0);
                else
                    Attention(dev, L, il, nt, p0, TokensOf(dev));
                if (dbg)
                    Dump(dev, "L0.attn.out", dev.AttnOut);
                Trace(dev, TraceLayer(il, "attn_out"), dev.AttnOut, (long)nt * e);
                dev.DK.HcPost(dev.Xs, dev.AttnOut, dev.Post, dev.Comb, dev.XsOut, nt, e, dev.Stream);
                SwapXs(dev);
                StageEnd(dev, 1);
                if (dbg)
                    Dump(dev, "L0.attn.xs_post", dev.Xs);
                CheckSync(dev, $"attn L{il}");

                // ---- FFN super-block ----
                HcPre(dev, L, nt, attn: false,
                    delayed: m.V41 ? dev.PreAttn : null,
                    publish: m.V41 ? dev.PreFfn : null,
                    normW: L.FfnNorm);
                Trace(dev, TraceLayer(il, "ffn_input"), dev.Cur, (long)nt * e);
                StageEnd(dev, 1);
                MoeFfn(dev, L, il, nt, TokensOf(dev));
                if (dbg)
                    Dump(dev, "L0.ffn.out", dev.FfnOut);
                Trace(dev, TraceLayer(il, "ffn_out"), dev.FfnOut, (long)nt * e);
                dev.DK.HcPost(dev.Xs, dev.FfnOut, dev.Post, dev.Comb, dev.XsOut, nt, e, dev.Stream);
                SwapXs(dev);
                Trace(dev, TraceLayer(il, "hidden"), dev.Xs, (long)nt * HC * e);
                StageEnd(dev, 1);
                if (StageDebug)
                    Dump(dev, $"L{il}.ffn.xs_post", dev.Xs);
                CheckSync(dev, $"ffn L{il}");

                // DSpark reads the mean over this block's output streams for a
                // few late layers; capture them into the drafter's feature row.
                if (_ds != null && (hAllOut != null || _dsSelfCatchUp) && _ds.CaptureSlot[il] >= 0)
                {
                    dev.DK.HcMean(dev.Xs, _ds.CapH, nt, e, DsparkFeatureSize, _ds.CaptureSlot[il] * e, dev.Stream);
                    CheckSync(dev, $"dspark capture L{il}");
                }
            }

            if (_dsSelfCatchUp)
                DsparkWriteRingFromCapture(nt, p0);
            else if (hAllOut != null)
                DsparkCaptureOut(nt, hAllOut, hRowOff);

            if (logitsOut != null)
            {
                var dev = _devs[curDev];
                if (curDev != _lastDev)
                    throw new InvalidOperationException("[dsv4-cuda] output head is not on the final layer device");
                dev.MakeCurrent();
                int headRows = allLogitsRows ? nt : 1;
                Tensor logitsDst = batch != null ? _batchLogits : allLogitsRows ? _specLogits : dev.Logits;
                IntPtr pinnedLogits = batch != null ? _pinnedBatchLogits : _pinnedLogits;
                if (m.V41)
                {
                    // V4.1 has no head mixer: the last layer's FFN block already
                    // published the gates the head collapses with.
                    using Tensor headXs = allLogitsRows ? dev.Xs.CopyRef() : dev.Xs.Narrow(0, nt - 1, 1);
                    using Tensor headPre = allLogitsRows ? dev.PreFfn.CopyRef() : dev.PreFfn.Narrow(0, nt - 1, 1);
                    dev.DK.HcCollapse(headXs, headPre, dev.Cur, headRows, e, dev.Stream);
                }
                else
                using (Tensor headX = allLogitsRows ? dev.Xs.CopyRef() : dev.Xs.Narrow(0, nt - 1, 1))
                {
                    dev.DK.HcHead(headX, Ptr(_hcHeadFn), Ptr(_hcHeadScale), Ptr(_hcHeadBase), dev.Cur, e,
                        m.HcHeadScale.Length, m.HcHeadBase.Length, m.RmsEps, dev.Stream, headRows);
                }
                Dump(dev, "head.cur", dev.Cur);
                RmsNorm(dev, dev.Cur, _outputNorm, headRows);
                MatMul(dev, _outputQW, dev.Cur, logitsDst, headRows);
                StageEnd(dev, 10);
                Dump(dev, "head.logits", logitsDst, 8);
                long logitCount = (long)headRows * m.NVocab;
                CudaDriverApi.cuMemcpyDtoHAsync(pinnedLogits, Ptr(logitsDst), new UIntPtr((ulong)logitCount * 4UL), dev.Stream).ThrowOnError();
                if (_perf > 0)
                    _issueEnd = Stopwatch.GetTimestamp();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy((void*)pinnedLogits, dst, logitsOut.LongLength * 4L, logitCount * 4L);
                _stages.Report();
            }
        }

        private static void SwapXs(Dev dev)
        {
            (dev.Xs, dev.XsOut) = (dev.XsOut, dev.Xs);
        }

        /// <summary>In-place RMS norm of the first <paramref name="rows"/> rows,
        /// through the shared Ops (the engine no longer carries its own kernel).</summary>
        private void RmsNorm(Dev dev, Tensor data, Tensor weight, int rows)
        {
            using Tensor view = Rows(data, rows);
            Ops.RMSNorm(view, view, weight, null, _m.RmsEps);
        }

        /// <param name="delayed">V4.1 only. The block still derives its own gates
        /// from the current streams, but collapses them with the gates the
        /// PREVIOUS block published; the first block of the model collapses with
        /// a plain stream mean instead. Publishing happens into
        /// <paramref name="publish"/> after the collapse, so the two buffers
        /// never alias.</param>
        /// <param name="normW">When set, Cur leaves RMS normed with it, and a decode-size batch takes the
        /// two-kernel pre-block. (Longer ubatches keep cuBLAS for the F32 mixes: at 512 rows the
        /// two-kernel form reads hc_fn from L2 once per row slice, and prefill measured 1.5% slower.)</param>
        private void HcPre(Dev dev, DevLayer l, int nt, bool attn,
            Tensor delayed = null, Tensor publish = null, bool meanCollapse = false, Tensor normW = null)
        {
            var m = _m;
            int flatDim = HC * m.NEmbd;
            Tensor fn = attn ? l.HcAttnFn : l.HcFfnFn;
            Tensor scale = attn ? l.HcAttnScale : l.HcFfnScale;
            Tensor baseW = attn ? l.HcAttnBase : l.HcFfnBase;

            if (normW != null && nt <= MaxBatchedDecodeRows && flatDim % Dsv4Kernels.HcSlice == 0)
            {
                dev.DK.HcMixPartials(dev.Xs, Ptr(fn), 0, dev.HcPartials, flatDim, nt, dev.Stream);
                dev.DK.HcPreFinish(dev.HcPartials, flatDim / Dsv4Kernels.HcSlice, dev.Xs, Ptr(scale), Ptr(baseW), Ptr(normW),
                    dev.Pre, dev.Post, dev.Comb, delayed, meanCollapse, publish, dev.Cur, m.NEmbd, nt,
                    m.HcSinkhornIters, m.HcEps, m.RmsEps, dev.Stream);
                return;
            }

            dev.DK.HcRms(dev.Xs, dev.Inv, nt, flatDim, m.RmsEps, dev.Stream);
            // mixes = hc_fn (F32 [24, flatDim]) x flat streams
            MatMulF32(dev, Ptr(fn), dev.Xs, dev.Mixes, flatDim, HcMixDim, nt);
            dev.DK.HcGatesComb(dev.Mixes, dev.Inv, Ptr(scale), Ptr(baseW), dev.Pre, dev.Post, dev.Comb,
                nt, m.HcSinkhornIters, m.HcEps, dev.Stream);

            if (meanCollapse)
                dev.DK.HcMean(dev.Xs, dev.Cur, nt, m.NEmbd, m.NEmbd, 0, dev.Stream);
            else
                dev.DK.HcCollapse(dev.Xs, delayed ?? dev.Pre, dev.Cur, nt, m.NEmbd, dev.Stream);

            if (publish != null)
                CudaDriverApi.cuMemcpyDtoDAsync(Ptr(publish), Ptr(dev.Pre),
                    new UIntPtr((ulong)((long)nt * HC * 4)), dev.Stream).ThrowOnError();
            if (normW != null)
                RmsNorm(dev, dev.Cur, normW, nt);
        }

        // -------------------------------------------------------------------
        // Attention
        // -------------------------------------------------------------------

        private void Attention(Dev dev, DevLayer l, int il, int nt, int p0, Tensor tokensDev)
        {
            var m = _m;
            int e = m.NEmbd, nh = m.NHead, hd = m.HeadDim, rot = m.NRot;
            bool comp = l.Ratio != 0;
            SlotLayer c = _active.Layers[il];
            IntPtr ropeTab = Ptr(comp ? dev.RopeComp : dev.RopeRaw);

            // q = wq_b(rms(wq_a(cur))), kv = wkv(cur)
            bool dbg = StageDebug && il == 0;
            MatMul(dev, l.WqA, dev.Cur, dev.Qr, nt);
            RmsNorm(dev, dev.Qr, l.QANorm, nt);
            if (dbg)
                Dump(dev, "L0.qr", dev.Qr);
            MatMul(dev, l.WqB, dev.Qr, dev.Q, nt);
            MatMul(dev, l.Wkv, dev.Cur, dev.KvRaw, nt);
            if (dbg)
                Dump(dev, "L0.kv_raw", dev.KvRaw);
            dev.DK.AttnPrep(dev.Q, dev.KvRaw, Ptr(l.KvNorm), ropeTab, Ptr(c.RingK), p0, _ringRaw, nh, hd, rot, m.RmsEps, nt, dev.Stream);
            StageEnd(dev, 2);
            if (dbg)
            {
                Dump(dev, "L0.q_prep", dev.Q);
                DumpF16(dev, "L0.ring0", c.RingK);
            }
            CheckSync(dev, $"attn_prep L{il}");

            int mode = 0;
            if (l.Ratio == CsaRatio)
            {
                int cw = 2 * hd;
                MatMul(dev, l.CompWkv, dev.Cur, dev.StKv, nt);
                MatMul(dev, l.CompWgate, dev.Cur, dev.StScore, nt);
                dev.DK.ApeAdd(dev.StScore, Ptr(l.CompApe), p0, CsaRatio, nt, cw, dev.Stream);
                RunCompressor(dev, nt, p0, CsaRatio, 2, hd, 2 * CsaRatio + _maxDraft, cw,
                    dev.StKv, dev.StScore, c.HistKv, c.HistScore, l.CompNorm, c.CompK, dev.RopeComp);

                int lcw = 2 * m.IdxHeadSize;
                MatMul(dev, l.IdxCompWkv, dev.Cur, dev.LidStKv, nt);
                MatMul(dev, l.IdxCompWgate, dev.Cur, dev.LidStScore, nt);
                dev.DK.ApeAdd(dev.LidStScore, Ptr(l.IdxCompApe), p0, CsaRatio, nt, lcw, dev.Stream);
                RunCompressor(dev, nt, p0, CsaRatio, 2, m.IdxHeadSize, 2 * CsaRatio + _maxDraft, lcw,
                    dev.LidStKv, dev.LidStScore, c.LidHistKv, c.LidHistScore, l.IdxCompNorm, c.LidK, dev.RopeComp);
                StageEnd(dev, 3);
                CheckSync(dev, $"compress L{il}");

                int maxVis = (int)(((long)p0 + nt) / CsaRatio);
                if (maxVis > m.IdxTopK)
                {
                    // lightning indexer + top-k
                    MatMul(dev, l.IdxQB, dev.Qr, dev.Iq, nt);
                    MatMul(dev, l.IdxProj, dev.Cur, dev.Iw, nt);
                    float iwScale = 1.0f / MathF.Sqrt((float)m.IdxHeadSize * m.IdxNHead);
                    dev.DK.IdxPrep(dev.Iq, dev.Iw, Ptr(dev.RopeComp), p0, m.IdxNHead, m.IdxHeadSize, rot, iwScale, nt, dev.Stream);
                    dev.DK.IdxScores(dev.Iq, dev.Iw, Ptr(c.LidK), dev.IdxScores, p0, CsaRatio, m.IdxNHead, m.IdxHeadSize,
                        nt, _compRowsCsa, maxVis, dev.Stream);
                    dev.DK.TopK(dev.IdxScores, dev.TopkIdx, dev.TopkCnt, p0, CsaRatio, m.IdxTopK, _compRowsCsa, nt, dev.Stream);
                    StageEnd(dev, 4);
                    CheckSync(dev, $"indexer L{il}");
                    mode = 1;
                }
                else
                {
                    mode = 2; // every visible row is selected: skip the indexer entirely
                }
            }
            else if (l.Ratio == HcaRatio)
            {
                MatMul(dev, l.CompWkv, dev.Cur, dev.StKv, nt);
                MatMul(dev, l.CompWgate, dev.Cur, dev.StScore, nt);
                dev.DK.ApeAdd(dev.StScore, Ptr(l.CompApe), p0, HcaRatio, nt, hd, dev.Stream);
                RunCompressor(dev, nt, p0, HcaRatio, 1, hd, HcaRatio + _maxDraft, hd,
                    dev.StKv, dev.StScore, c.HistKv, c.HistScore, l.CompNorm, c.CompK, dev.RopeComp);
                StageEnd(dev, 3);
                CheckSync(dev, $"compress L{il}");
                mode = 2;
            }

            float kqScale = 1.0f / MathF.Sqrt(hd);
            dev.DK.Attention(dev.Q, Ptr(c.RingK), Ptr(c.CompK), dev.TopkIdx, dev.TopkCnt, Ptr(l.Sinks), dev.AttnO,
                p0, m.NSwa, _ringRaw, nh, hd, mode, l.Ratio == 0 ? 1 : l.Ratio, m.IdxTopK, kqScale, nt, dev.Stream);
            StageEnd(dev, 5);
            if (dbg)
                Dump(dev, "L0.attn_core", dev.AttnO);
            CheckSync(dev, $"attn_core L{il}");

            // inverse rope + grouped LoRA out-projection
            int hpg = nh / m.OGroups;
            dev.DK.AttnFinish(dev.AttnO, ropeTab, dev.OGrouped, p0, nh, hd, rot, hpg, nt, dev.Stream);

            OutProjectionGroups(dev, l, nt);
            CheckSync(dev, $"out_proj L{il}");
        }

        /// <summary>The grouped LoRA output projection over OGrouped ([G, nt, groupDim]):
        /// one matmul per group against its WoA row block, then WoB over the regrouped rows.</summary>
        private void OutProjectionGroups(Dev dev, DevLayer l, int nt)
        {
            var m = _m;
            int groupDim = m.NHead / m.OGroups * m.HeadDim;
            var woA = l.WoA;
            if (nt == 1 && woA.Type == TBF16 && (groupDim & 7) == 0)
            {
                // Decode: the 8 group matvecs are ~11us each, so folding them
                // into one grid saves more in launch gaps than it costs.
                dev.Alloc.Kernels.LaunchMatvecBf16(woA.Ptr, Ptr(dev.OGrouped), Ptr(dev.OGroupedOut),
                    groupDim, m.OLoraRank, dev.Stream, m.OGroups);
            }
            else
            {
                for (int g = 0; g < m.OGroups; g++)
                {
                    var slice = woA;
                    slice.Ptr = (IntPtr)((long)woA.Ptr + (long)g * m.OLoraRank * woA.RowBytes);
                    slice.Ne1 = m.OLoraRank;
                    using Tensor input = Block(dev.OGrouped, (long)g * nt, nt, groupDim);
                    using Tensor output = Block(dev.OGroupedOut, (long)g * nt, nt, m.OLoraRank);
                    MatMul(dev, slice, input, output, nt);
                }
            }
            dev.DK.Regroup(dev.OGroupedOut, dev.OG, m.OGroups, nt, m.OLoraRank, dev.Stream);
            MatMul(dev, l.WoB, dev.OG, dev.AttnOut, nt);
            StageEnd(dev, 6);
        }

        // -------------------------------------------------------------------
        // V4.1 attention
        //
        // Mirrors DeepSeek4CpuExecutor.AttentionV41, which is held to the
        // PyTorch reference by InferenceWeb.Tests.Dsv41CpuExecutorTests. The
        // differences from V4 are: no per-head query norm, a non-overlapping
        // compressor window with no absolute positional embedding, the indexer's
        // K projected from the compressed latent, one layer per ratio group
        // owning the caches and the sparse selection, optional candidate
        // pruning, and the trained cache quantization on every commit.
        // -------------------------------------------------------------------

        private void AttentionV41(Dev dev, DevLayer l, int il, int nt, int p0)
        {
            var m = _m;
            int e = m.NEmbd, nh = m.NHead, hd = m.HeadDim, rot = m.NRot;
            int ratio = l.Ratio;
            IntPtr ropeTab = Ptr(ratio != 0 ? dev.RopeComp : dev.RopeRaw);
            SlotLayer c = _active.Layers[il];

            MatMul(dev, l.WqA, dev.Cur, dev.Qr, nt);
            RmsNorm(dev, dev.Qr, l.QANorm, nt);
            MatMul(dev, l.WqB, dev.Qr, dev.Q, nt);
            MatMul(dev, l.Wkv, dev.Cur, dev.KvRaw, nt);
            dev.DK.V41AttnPrep(dev.Q, dev.KvRaw, Ptr(l.KvNorm), ropeTab, Ptr(c.RingK),
                p0, _ringRaw, nh, hd, rot, m.RmsEps, nt, dev.Stream);
            Trace(dev, TraceLayer(il, "q"), dev.Q, (long)nt * nh * hd);
            Trace(dev, TraceLayer(il, "raw_k"), dev.KvRaw, (long)nt * hd);
            StageEnd(dev, 2);
            CheckSync(dev, $"v41_attn_prep L{il}");

            int mode = 0;
            if (ratio != 0)
            {
                if (l.KvSource == il)
                    CompressV41(dev, l, nt, p0, ratio);
                StageEnd(dev, 3);
                CheckSync(dev, $"v41_compress L{il}");
                if (l.IndexSource == il)
                {
                    BuildIndexerV41(dev, l, il, nt, p0, ratio);
                    StageEnd(dev, 4);
                    CheckSync(dev, $"v41_indexer L{il}");
                }
                mode = 1;   // the selection published by this group's index source
            }

            Tensor comp = ratio != 0 ? CompKOn(_active, l.KvSource, dev) : null;
            float kqScale = 1.0f / MathF.Sqrt(hd);
            dev.DK.Attention(dev.Q, Ptr(c.RingK), Ptr(comp), dev.TopkIdx, dev.TopkCnt, Ptr(l.Sinks), dev.AttnO,
                p0, m.NSwa, _ringRaw, nh, hd, mode, ratio == 0 ? 1 : ratio, m.IdxTopK, kqScale, nt, dev.Stream);
            StageEnd(dev, 5);
            CheckSync(dev, $"v41_attn_core L{il}");

            OutProjection(dev, l, nt, p0, ropeTab);
        }

        /// <summary>One normalized latent per complete block, then the indexer K
        /// projection off the UNROTATED latent, then both caches committed.</summary>
        private void CompressV41(Dev dev, DevLayer l, int nt, int p0, int ratio)
        {
            var m = _m;
            int hd = m.HeadDim, id = m.IdxHeadSize;
            SlotLayer c = _active.Layers[l.KvSource];

            MatMul(dev, l.CompWkv, dev.Cur, dev.StKv, nt);
            if (ratio > 1)
                MatMul(dev, l.CompWgate, dev.Cur, dev.StScore, nt);

            long firstBoundary = -1;
            for (long p = p0; p < (long)p0 + nt; p++)
            {
                if ((p + 1) % ratio == 0) { firstBoundary = p; break; }
            }
            int nBlocks = firstBoundary < 0 ? 0 : (int)(((long)p0 + nt - 1 - firstBoundary) / ratio) + 1;

            if (nBlocks > 0)
            {
                dev.DK.V41Compress(dev.StKv, dev.StScore, Ptr(c.HistKv), Ptr(c.HistScore), Ptr(l.CompNorm),
                    dev.Latent, firstBoundary, nBlocks, p0, ratio, V41StateRows(ratio), hd, m.RmsEps, dev.Stream);

                using (Tensor latentRows = Rows(dev.Latent, nBlocks))
                using (Tensor keyRows = Rows(dev.LatentK, nBlocks))
                {
                    MatMul(dev, l.IndexerK, latentRows, keyRows, nBlocks);
                    RmsNorm(dev, keyRows, l.IndexerKNorm, nBlocks);
                }

                dev.DK.V41Commit(dev.LatentK, Ptr(dev.RopeComp), Ptr(c.LidK),
                    firstBoundary, nBlocks, ratio, id, m.NRot, 1, dev.Stream);
                dev.DK.V41Commit(dev.Latent, Ptr(dev.RopeComp), Ptr(c.CompK),
                    firstBoundary, nBlocks, ratio, hd, m.NRot, 2, dev.Stream);
                NoteCommitted(l.KvSource, _active, firstBoundary, ratio, nBlocks);
            }

            if (ratio > 1)
                dev.DK.V41Persist(dev.StKv, dev.StScore, Ptr(c.HistKv), Ptr(c.HistScore), p0, nt, V41StateRows(ratio), hd, dev.Stream);
        }

        private void BuildIndexerV41(Dev dev, DevLayer l, int il, int nt, int p0, int ratio)
        {
            var m = _m;
            Tensor lidK = LidKOn(_active, l.KvSource, dev);
            int rows = dev.IdxScores.Sizes[1] is long c ? (int)c : 0;

            MatMul(dev, l.IdxQB, dev.Qr, dev.Iq, nt);
            MatMul(dev, l.IdxProj, dev.Cur, dev.Iw, nt);
            float iwScale = 1.0f / MathF.Sqrt((float)m.IdxHeadSize * m.IdxNHead);
            dev.DK.V41IdxPrep(dev.Iq, dev.Iw, Ptr(dev.RopeComp), p0, m.IdxNHead, m.IdxHeadSize, m.NRot,
                iwScale, nt, dev.Stream);

            // Rows this layer may see: visibility, then whatever the candidate
            // layer left standing for the layers after it.
            bool prune = _v41CandActive && il > m.CandidateSource;
            int maxVis = (int)(((long)p0 + nt) / ratio);
            dev.DK.V41IdxScores(dev.Iq, dev.Iw, Ptr(lidK), prune ? Ptr(dev.CandMask) : IntPtr.Zero,
                dev.IdxScores, p0, ratio, m.IdxNHead, m.IdxHeadSize, rows, maxVis, nt, dev.Stream);

            if (il == m.CandidateSource)
            {
                dev.DK.V41Candidate(dev.IdxScores, Ptr(dev.CandMask), p0, ratio, rows,
                    m.CandidateBlock, m.CandidateTopk, nt, dev.Stream);
                _v41CandActive = true;
            }

            dev.DK.TopK(dev.IdxScores, dev.TopkIdx, dev.TopkCnt, p0, ratio, m.IdxTopK, rows, nt, dev.Stream);
        }

        /// <summary>Inverse RoPE on the rope slice, then the grouped LoRA output
        /// projection. Shared by both architectures.</summary>
        private void OutProjection(Dev dev, DevLayer l, int nt, int p0, IntPtr ropeTab)
        {
            var m = _m;
            int nh = m.NHead, hd = m.HeadDim;
            int hpg = nh / m.OGroups;
            // p0, not 0: the inverse rotation has to undo the rotation each token
            // was given at its OWN absolute position.
            dev.DK.AttnFinish(dev.AttnO, ropeTab, dev.OGrouped, p0, nh, hd, m.NRot, hpg, nt, dev.Stream);
            OutProjectionGroups(dev, l, nt);
        }

        /// <summary>
        /// Gathers this ubatch's Engram rows for one layer and folds them into the residual. The
        /// executor hashes the token history into row indices; a device-resident table is gathered
        /// on the device from those, a host-mapped one is dequantized by the executor and its rows
        /// cross the bus.
        /// </summary>
        /// <remarks>The host runs ahead of the device, so the copy out of the staging queued for
        /// the previous fill (the last ubatch, or another Engram layer on this device) may not have
        /// run yet: refilling a shared staging handed that copy this layer's rows. Each layer
        /// fills its own staging, after the copy out of its last fill has run.</remarks>
        private void EngramLayer(Dev dev, DevLayer l, int nt)
        {
            var m = _m;
            int columns = m.Engram.HashColumns;
            long cols = (long)columns * m.Engram.HeadDim;
            CudaDriverApi.cuEventSynchronize(l.EngramCopied).ThrowOnError();
            if (l.EngramTable.Ptr != IntPtr.Zero)
            {
                // Device-resident table: only the selected row indices cross the bus.
                m.Engram.CopyEngramRowIndices(l.EngramIndex, nt, (int*)l.EngramPinned);
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.EngramRows), l.EngramPinned,
                    new UIntPtr((ulong)((long)nt * columns * 4)), dev.Stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(l.EngramCopied, dev.Stream).ThrowOnError();
                dev.DK.V41EngramGather(l.EngramTable.Ptr, l.EngramTable.Type, l.EngramTable.RowBytes,
                    dev.EngramRows, dev.EngramLookup, columns, m.Engram.HeadDim, nt, dev.Stream);
            }
            else
            {
                m.Engram.GatherEngramRows(l.EngramIndex, nt, (float*)l.EngramPinned);
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.EngramLookup), l.EngramPinned,
                    new UIntPtr((ulong)(nt * cols * 4)), dev.Stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(l.EngramCopied, dev.Stream).ThrowOnError();
            }
            MatMul(dev, l.EngramWkv, dev.EngramLookup, dev.EngramKv, nt);
            dev.DK.V41EngramGate(dev.Xs, dev.EngramKv, Ptr(l.EngramQ), Ptr(l.EngramK),
                HC, m.NEmbd, m.RmsEps, nt, dev.Stream);
        }

        /// <summary>A candidate mask belongs to one ubatch: the rows it names are
        /// scored against this ubatch's queries.</summary>
        private bool _v41CandActive;

        private void RunCompressor(Dev dev, int nt, int p0, int ratio, int coff, int head, int stateSize, int cw,
            Tensor stKv, Tensor stScore, Tensor histKv, Tensor histScore, Tensor normW, Tensor cache, Tensor ropeTab)
        {
            long firstBoundary = -1;
            for (long p = p0; p < (long)p0 + nt; p++)
            {
                if ((p + 1) % ratio == 0)
                {
                    firstBoundary = p;
                    break;
                }
            }
            if (firstBoundary >= 0)
            {
                int nBlocks = (int)(((long)p0 + nt - 1 - firstBoundary) / ratio) + 1;
                dev.DK.Compress(stKv, stScore, Ptr(histKv), Ptr(histScore), Ptr(normW), Ptr(ropeTab), Ptr(cache),
                    firstBoundary, nBlocks, p0, ratio, coff, head, stateSize, _m.NRot, _m.RmsEps, dev.Stream);
            }
            dev.DK.Persist(stKv, stScore, Ptr(histKv), Ptr(histScore), p0, nt, stateSize, cw, dev.Stream);
        }

        // -------------------------------------------------------------------
        // MoE FFN
        // -------------------------------------------------------------------

        private void MoeFfn(Dev dev, DevLayer l, int il, int nt, Tensor tokensDev)
        {
            var m = _m;
            int e = m.NEmbd, ff = m.NFfExp, nUsed = m.NExpertUsed, nEx = m.NExpert;
            int s = nt * nUsed;

            // router logits (gate_inp is F32) + selection/weights
            bool dbg = StageDebug && il == 0;
            MatMulF32(dev, Ptr(l.GateInp), dev.Cur, dev.RouterLogits, e, nEx, nt);
            dev.DK.MoeSelect(dev.RouterLogits, Ptr(l.ExpProbsBias), Ptr(l.Tid2Eid), tokensDev, dev.Sel, dev.SelW,
                nEx, nUsed, m.ExpertWeightsNorm ? 1 : 0, m.ExpertWeightsScale, nt, dev.Stream);
            StageEnd(dev, 7);
            if (dbg)
            {
                Dump(dev, "L0.router", dev.RouterLogits);
                Dump(dev, "L0.selw", dev.SelW, m.NExpertUsed);
            }
            CheckSync(dev, $"moe_select L{il}");

            // shared expert (dense) -> ShDown
            int shFf = l.ShFf;
            MatMul(dev, l.UpShexp, dev.Cur, dev.ShUp, nt);
            MatMul(dev, l.GateShexp, dev.Cur, dev.ShGate, nt);
            SwigluClamp(dev.ShGate, dev.ShUp, (long)nt * shFf, l.ClampShexp);
            MatMul(dev, l.DownShexp, dev.ShGate, dev.ShDown, nt);
            StageEnd(dev, 9);
            CheckSync(dev, $"shexp L{il}");

            // routed experts
            //
            // The DSpark drafter's stages are appended past the trunk and run
            // through this same MoE builder with il >= NLayer. They are never
            // offloaded (they live on the output-head device and _hostMoe is
            // sized to the trunk), so anything outside the trunk range is
            // resident by definition — indexing _hostMoe unguarded threw
            // IndexOutOfRange on the first drafted block.
            HostMoeLayer hostExperts = (uint)il < (uint)_hostMoe.Length ? _hostMoe[il] : null;
            if (hostExperts != null)
            {
                // Offloaded layer: the host fills ExpDown in slot order, so the
                // weighted-sum + shared-expert epilogue below is shared with the
                // resident path unchanged.
                MoeFfnHost(dev, hostExperts, il, nt);
                dev.DK.MoeScatterAdd(dev.ExpDown, null, dev.SelW, dev.ShDown, dev.FfnOut, nt, nUsed, e, dev.Stream);
                StageEnd(dev, 8);
                CheckSync(dev, $"host experts L{il}");
                return;
            }

            int guType = l.GateExps.Type;
            int downType = l.DownExps.Type;
            if (l.UpExps.Type != guType)
                throw new NotSupportedException("[dsv4-cuda] expert gate/up quant types differ");
            RequireExpertType(guType);
            RequireExpertType(downType);

            // Up to MaxBatchedDecodeRows tokens (a decode step, a batched decode step, a verify
            // window) run the decode kernels once per (token, selected expert) slot, so every token
            // computes exactly what it computes alone. Longer ubatches group the slots by expert and
            // run on tensor cores when the widths allow, else through the register-staged kernels.
            bool perSlot = nt <= MaxBatchedDecodeRows;
            bool mma = !perSlot
                && StagedSupportsType(guType) && StagedSupportsType(downType)
                && e % Dsv4Kernels.MoeMmaK == 0 && ff % Dsv4Kernels.MoeMmaK == 0;
            bool staged = !perSlot && !mma
                && StagedSupportsType(guType) && StagedSupportsType(downType)
                && Dsv4Kernels.StagedSupports(e) && Dsv4Kernels.StagedSupports(ff);
            if (staged)
                QuantizeQ81Split(dev, dev.Cur, dev.SplitQsA, dev.SplitDA, e, nt);
            else if (!mma)
                QuantizeQ81(dev, dev.Cur, dev.ActQ8A, e, nt);
            if (perSlot)
            {
                dev.DK.MoeGateUpDecode(l.GateExps.Ptr, l.UpExps.Ptr, dev.ActQ8A, dev.Sel,
                    dev.ExpGate, dev.ExpUp, guType, ff, e, l.GateExps.RowBytes, nt, nUsed, dev.Stream);
                CudaMoe.SwigluQuantize(dev.Alloc.Kernels, dev.ExpGate, dev.ExpUp, dev.ActQ8B, null,
                    ff, s, l.ClampExp, dev.Stream);
                dev.DK.MoeDownDecode(l.DownExps.Ptr, dev.ActQ8B, dev.Sel, dev.ExpDown,
                    downType, e, ff, l.DownExps.RowBytes, s, dev.Stream);
                dev.DK.MoeScatterAdd(dev.ExpDown, null, dev.SelW, dev.ShDown, dev.FfnOut, nt, nUsed, e, dev.Stream);
            }
            else
            {
                // grouping plan (all on device; Counts zeroed via the fill kernel
                // so everything stays stream-ordered)
                dev.Alloc.Kernels.LaunchFillF32(Ptr(dev.Counts), nEx, 0f, dev.Stream);
                dev.DK.MoeCount(dev.Sel, dev.Counts, s, dev.Stream);
                dev.DK.MoeScan(dev.Counts, dev.Offsets, dev.Cursors, nEx, dev.Stream);
                dev.DK.MoeScatter(dev.Sel, dev.Cursors, dev.RowOfSlot, dev.SlotToken, s, nUsed, dev.Stream);

                if (mma)
                {
                    dev.DK.MoeMma(l.GateExps.Ptr, l.UpExps.Ptr, dev.Cur, dev.Counts, dev.Offsets, dev.SlotToken,
                        dev.ExpGate, dev.ExpUp, guType, ff, e, l.GateExps.RowBytes, nEx, dev.Stream);
                    SwigluClamp(dev.ExpGate, dev.ExpUp, (long)s * ff, l.ClampExp);
                    dev.DK.MoeMma(l.DownExps.Ptr, IntPtr.Zero, dev.ExpGate, dev.Counts, dev.Offsets, null,
                        dev.ExpDown, null, downType, e, ff, l.DownExps.RowBytes, nEx, dev.Stream);
                }
                // Staged-weight kernels decode each expert weight row once into
                // registers and reuse it across the expert's member tokens; the
                // per-token kernels re-decode per (row, token).
                else if (staged)
                {
                    dev.DK.MoeGateUpStaged(l.GateExps.Ptr, l.UpExps.Ptr, dev.SplitQsA, dev.SplitDA,
                        dev.Counts, dev.Offsets, dev.SlotToken,
                        dev.ExpGate, dev.ExpUp, guType, ff, e, l.GateExps.RowBytes, nEx, dev.Stream);
                    CudaMoe.SwigluQuantize(dev.Alloc.Kernels, dev.ExpGate, dev.ExpUp, dev.SplitQsB, dev.SplitDB,
                        ff, s, l.ClampExp, dev.Stream);
                    dev.DK.MoeDownStaged(l.DownExps.Ptr, dev.SplitQsB, dev.SplitDB, dev.Counts, dev.Offsets, dev.ExpDown,
                        downType, e, ff, l.DownExps.RowBytes, nEx, dev.Stream);
                }
                else
                {
                    dev.DK.MoeGateUp(l.GateExps.Ptr, l.UpExps.Ptr, dev.ActQ8A, dev.Counts, dev.Offsets, dev.SlotToken,
                        dev.ExpGate, dev.ExpUp, guType, ff, e, l.GateExps.RowBytes, nEx, dev.Stream);
                    CudaMoe.SwigluQuantize(dev.Alloc.Kernels, dev.ExpGate, dev.ExpUp, dev.ActQ8B, null,
                        ff, s, l.ClampExp, dev.Stream);
                    dev.DK.MoeDown(l.DownExps.Ptr, dev.ActQ8B, dev.Counts, dev.Offsets, dev.ExpDown,
                        downType, e, ff, l.DownExps.RowBytes, nEx, dev.Stream);
                }
                dev.DK.MoeScatterAdd(dev.ExpDown, dev.RowOfSlot, dev.SelW, dev.ShDown, dev.FfnOut, nt, nUsed, e, dev.Stream);
            }
            StageEnd(dev, 8);
            CheckSync(dev, $"experts L{il}");
        }

        /// <summary>
        /// Clamped SwiGLU over the leading <paramref name="n"/> elements, through
        /// the shared Ops.SiLUMulClamp (gate is overwritten in place).
        /// </summary>
        private static void SwigluClamp(Tensor gate, Tensor up, long n, float limit)
        {
            using Tensor g = Flat(gate, n);
            using Tensor u = Flat(up, n);
            Ops.SiLUMulClamp(g, g, u, limit);
        }

        private static Tensor Flat(Tensor t, long n)
        {
            using Tensor flat = t.View(t.ElementCount());
            return flat.Narrow(0, 0, n);
        }

        private static void RequireExpertType(int type)
        {
            if (!StagedSupportsType(type) && type != TIQ2_XXS)
            {
                throw new NotSupportedException(
                    $"[dsv4-cuda] unsupported expert quant type {type} (supported: Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ3_S, IQ2_XXS, MXFP4)");
            }
        }

        /// <summary>The layouts the grouped expert kernels (tensor-core and register-staged)
        /// decode; IQ2_XXS, used by the DSpark drafter, only has the per-token path.</summary>
        private static bool StagedSupportsType(int type)
            => type == TIQ3_S || type == TMXFP4 || type == TQ8_0 || type == TQ6_K || type == TQ5_K || type == TQ4_K
                || type == TQ3_K || type == TQ2_K;

        // -------------------------------------------------------------------
        // dense matmul dispatch
        // -------------------------------------------------------------------

        private void QuantizeQ81(Dev dev, Tensor input, Tensor scratch, int inDim, int rows)
        {
            dev.Alloc.Kernels.LaunchQuantizeQ81Rows(Ptr(input), Ptr(scratch), inDim, rows, dev.Stream, warpCooperative: true);
        }

        /// <summary>Dense split q8_1 (contiguous int8 row + separate per-block
        /// scale). Values are bit-identical to the interleaved layout; the dense
        /// form is what makes the staged expert kernels' activation reads
        /// vectorizable.</summary>
        private void QuantizeQ81Split(Dev dev, Tensor input, Tensor qs, Tensor d, int inDim, int rows)
        {
            dev.Alloc.Kernels.LaunchQuantizeQ81SplitRows(Ptr(input), Ptr(qs), Ptr(d), inDim, rows, dev.Stream);
        }

        /// <summary>
        /// result[rows, w.Ne1] = input[rows, w.Ne0] x w^T for a weight this engine
        /// already holds resident in its per-device arena.
        ///
        /// The kernel choice (dp4a matvec, MMQ int8 GEMM, dequant-to-F16 + cuBLAS,
        /// BF16 tensor cores, ...) is NOT decided here: it is the shared routing in
        /// CudaQuantizedOps, so DSV4 gets exactly the same tuning as every other
        /// direct-CUDA model and there is one implementation to maintain. The engine
        /// asks for row-invariant kernels, so a batched decode step computes each
        /// sequence's row exactly as that sequence's own one-token forward does.
        /// </summary>
        private void MatMul(Dev dev, in DeviceWeight w, Tensor input, Tensor output, int rows)
        {
            // Scratch buffers are sized for the WIDEST user on this device (the
            // shared expert's ff differs per layer, compressor widths differ
            // between CSA and HCA), so the operands are compact [rows, dim]
            // blocks at the front of the buffer, not full-width row views.
            using Tensor a = Block(input, 0, rows, w.Ne0);
            using Tensor r = Block(output, 0, rows, w.Ne1);
            CudaQuantizedOps.AddmmResidentToFloat32(r, a, w.Ptr, w.Type, w.Ne0, w.Ne1, rowInvariant: true);
        }

        /// <summary>Dense F32 weight (MoE router, hyper-connection mixer) held in
        /// the arena rather than as a tensor: same shared routing, type F32.</summary>
        private void MatMulF32(Dev dev, IntPtr wF32, Tensor input, Tensor output, int inDim, int outDim, int rows)
        {
            if (rows <= Dsv4Kernels.GemvMaxRows)
            {
                dev.DK.Gemv(Ptr(input), wF32, Ptr(output), inDim, outDim, rows, dev.Stream, Ptr(dev.GemvScratch));
                return;
            }
            using Tensor a = Block(input, 0, rows, inDim);
            using Tensor r = Block(output, 0, rows, outDim);
            CudaQuantizedOps.AddmmResidentToFloat32(r, a, wF32, TF32, inDim, outDim);
        }

        public void Dispose()
        {
            _stages.Dispose();
            FreeHostMoeBuffers();
            // Slot caches come out of the device allocators, so they go before the allocators do.
            FreeSlots();
            foreach (var dev in _devs)
            {
                if (dev == null)
                    continue;
                try
                {
                    dev.MakeCurrent();
                    CudaDriverApi.cuStreamSynchronize(dev.Stream);
                    // Scratch, caches, small tensors and the weight arena are all
                    // allocator-owned: returning them to the pool is the only
                    // teardown needed (the allocator frees the pool on Dispose).
                    foreach (var t in dev.OwnedTensors)
                        t.Dispose();
                    dev.OwnedTensors.Clear();
                    dev.Weights?.Dispose();
                    if (dev.Event != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.Event);
                    if (dev.TokEv0 != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.TokEv0);
                    if (dev.TokEv1 != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.TokEv1);
                    if (dev.XsReadyEv != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.XsReadyEv);
                    if (dev.CopyDoneEv != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(dev.CopyDoneEv);
                    if (dev.BoundaryPinned != IntPtr.Zero)
                        CudaDriverApi.cuMemFreeHost(dev.BoundaryPinned);
                    dev.DK?.Dispose();
                    dev.Alloc?.Dispose();
                }
                catch
                {
                    // teardown must not throw
                }
            }
            foreach (var dev in _devs)
            {
                if (dev != null && dev.StatePinned != IntPtr.Zero)
                    CudaDriverApi.cuMemFreeHost(dev.StatePinned);
            }
            if (_layers != null)
            {
                foreach (var layer in _layers)
                {
                    if (layer == null)
                        continue;
                    if (layer.EngramPinned != IntPtr.Zero)
                        CudaDriverApi.cuMemFreeHost(layer.EngramPinned);
                    if (layer.EngramCopied != IntPtr.Zero)
                        CudaDriverApi.cuEventDestroy(layer.EngramCopied);
                }
            }
            if (_pinnedTokens0 != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedTokens0);
            if (_pinnedTokens1 != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(_pinnedTokens1);
        }
    }
}
