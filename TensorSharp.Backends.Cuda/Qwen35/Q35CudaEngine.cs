// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Qwen3.5 / Qwen3.8 dense (qwen35) on the direct-CUDA backend: the whole model resident on one GPU
// and driven without ggml.
//
// Layer il is gated GQA attention when (il + 1) % fullAttentionInterval == 0 and a Gated DeltaNet
// layer otherwise; every layer then runs a SwiGLU FFN, both halves pre-normed and residual:
//   attention: one projection to [Q|gate per head | K | V], per-head RMS norms on Q and K,
//              interleaved M-RoPE on the first 64 dims, q8_0 caches, causal attention whose output
//              is multiplied by sigmoid(gate) before the output projection;
//   GDN:       one projection to [qkv | z | beta | alpha], a causal depthwise convolution with SiLU,
//              L2-normed q and k, the gated delta rule (value head h reads key head h % hk) and a
//              gated RMS norm with silu(z) before the output projection.
// Projections are NVFP4 (repacked at load into code/scale planes, see tensorsharp_q35_kernels.cu):
// up to 8 activation rows take the q8_1 GEMV, more the FP4 tensor-core GEMM. The embedding is
// Q4_0 and the LM head Q6_K, read as stored.
//
// The reference is the native executor ggml_cuda runs (ggml_ops_qwen35_verify.cpp); the kernels
// follow its numerics (q8_1 / NVFP4 activation rounding, q8_0 caches, F16 attention operands).
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine : IDisposable
    {
        private const int TQ4_0 = 2, TQ6_K = 14, TNVFP4 = 40;

        /// <summary>Rotated dims per attention head and Gated DeltaNet head width the kernels take.</summary>
        public const int RopeDims = Q35Kernels.RopeDims, GdnHeadDim = Q35Kernels.GdnHeadDim;

        public sealed class LayerDesc
        {
            public bool Recurrent;
            /// <summary>The projections the block's input feeds, fused row-wise in this order:
            /// attention Q|gate, K, V; GDN qkv, z (attn_gate), beta, alpha.</summary>
            public CudaWeightDesc[] InProj;
            public CudaWeightDesc OutProj;
            public CudaWeightDesc FfnGate, FfnUp, FfnDown;
            public float[] AttnNorm, PostNorm;
            public float[] QNorm, KNorm;                         // attention
            public float[] ConvW, DtBias, SsmA, SsmNorm;         // GDN: ConvW per channel [C, dConv]
        }

        public sealed class ModelDesc
        {
            public int NLayer, NEmbd, NFf, NVocab;
            public int NHead, NKvHead, HeadDim;
            public int GdnKHeads, GdnVHeads, DConv;
            /// <summary>M-RoPE (t, h, w) pair counts.</summary>
            public int SectT, SectH, SectW;
            public float RopeBase, Eps;
            public int NCtx, NUbatch;
            public CudaWeightDesc TokEmbd, Output;
            public CudaNinferModelDesc Ninfer;
            /// <summary>Explicit experimental companion output head; null retains the original Q6 path.</summary>
            public float[] OutputNorm;
            public LayerDesc[] Layers;
        }

        /// <summary>An NVFP4 matrix as the kernels read it: codes [rows][k/2], UE4M3 scales [rows][k/16].</summary>
        private struct Planes
        {
            public IntPtr Codes, Scales;
            public int Rows, K;
        }

        private sealed class Layer
        {
            public bool Recurrent;
            /// <summary>Index among the layers of its kind (attention or Gated DeltaNet).</summary>
            public int Ordinal;
            public Planes InProj, OutProj, GateUp, Down;
            public IntPtr AttnNorm, PostNorm, QNorm, KNorm, ConvW, DtBias, SsmA, SsmNorm;
        }

        /// <summary>One sequence's caches: attention K/V in shared blocks of
        /// <see cref="Q35Kernels.KvBlockTokens"/> tokens (see Q35CudaEngine.Slots.cs), per GDN layer
        /// the convolution state [dConv - 1][convDim] and delta state [hv][128][128] F32 (indexed by
        /// the layer's ordinal among its kind).</summary>
        public sealed class Slot
        {
            public int NPast;
            /// <summary>M-RoPE delta: a row past the last position table sits at rotary position
            /// KV index + RopeDelta (an image's grid takes fewer positions than rows).</summary>
            public int RopeDelta;
            /// <summary>The attention cache blocks, in token order (shared ones are copied before a write).</summary>
            internal readonly List<IntPtr> Blocks = new List<IntPtr>();
            /// <summary>Bumped whenever <see cref="Blocks"/> changes, so a bound block table is refreshed.</summary>
            internal int Version;
            /// <summary>Tokens the slot's blocks hold.</summary>
            public int Capacity => Blocks.Count * Q35Kernels.KvBlockTokens;
            internal IntPtr[] Conv, State;
            /// <summary>Drafter KV ring per draft layer and the position each ring row holds.</summary>
            internal IntPtr[] RingK, RingV;
            internal IntPtr RingPos;
            /// <summary>One allocation holding the recurrent states (and the ring when the drafter
            /// was loaded first); the ring's contiguous base.</summary>
            internal IntPtr Arena, RingBase;
            internal readonly List<IntPtr> Owned = new List<IntPtr>();
        }

        /// <summary>Sequences one batched step takes.</summary>
        public const int MaxSeqs = 16;
        /// <summary>Rows (sequences x window) one verify takes.</summary>
        public const int MaxVerifyRows = 64;

        private readonly ModelDesc _m;
        private readonly CudaAllocator _alloc;
        private readonly Q35Kernels _k;
        private readonly Layer[] _layers;
        private readonly List<IntPtr> _owned = new List<IntPtr>();
        private IntPtr _tokEmbd, _output, _outputNorm;
        private long _tokEmbdRowBytes, _outputRowBytes;
        private readonly int _sms;
        private readonly int _nAttn, _nGdn;
        private Slot _active;

        // Device pointer tables the kernels index by sequence: the attention block table
        // [MaxSeqs][_maxBlocks], per GDN ordinal convolution and delta states [MaxSeqs]; _bound is
        // the slot list they currently describe, at the block versions in _boundVersions.
        private IntPtr _tabBlocks, _tabConv, _tabState;
        private int _maxBlocks;
        private Slot[] _bound = Array.Empty<Slot>();
        private int[] _boundVersions = Array.Empty<int>();

        // Verify buffers, per GDN ordinal [MaxVerifyRows] rows: the unconvolved inputs, the
        // convolved and normalized q|k|v and the decays / betas the commit replays; the feature
        // taps [MaxVerifyRows][FeatureWidth] of every verified row.
        private IntPtr _vRaw, _vQkv, _vG, _vBeta, _features, _keptDev;
        private int[] _featureSlot;
        private int _featureCount;
        private int _featureBufferWidth;

        // ---- scratch, sized for NUbatch rows ----
        private IntPtr _x, _xn, _proj, _q, _gate, _attn, _block, _gdnQkv, _gdnG, _gdnBeta, _gdnO, _gdnY, _act;
        private IntPtr _q8, _q8d, _fp4c, _fp4s, _fp4r, _partial, _counters, _attnPartial, _logits, _argmax;
        private IntPtr _rowsDev, _tokensDev, _ropePosDev, _cacheRowDev, _keyEndDev, _countDev, _lastX, _lastBlock;
        private IntPtr _gdnPreparedScratch;
        private long _gdnPreparedScratchBytes;
        private IntPtr _pinned;
        private const int PinnedBytes = 1 << 20;

        public int VocabSize => _m.NVocab;
        public int ContextSize => _m.NCtx;
        public int UBatch => _m.NUbatch;
        public int HiddenSize => _m.NEmbd;
        /// <summary>Actually submitted prepared GDN prefill passes (diagnostic coverage).</summary>
        public long PreparedGdnPrefillLaunchCount => _k.PreparedGdnLaunches;
        public bool PreparedGdnScratchReusesProjection { get; private set; }
        /// <summary>The sequence <see cref="Forward"/> runs.</summary>
        public Slot Active
        {
            get => _active;
            set => _active = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>Floats per verified row in <see cref="FeatureRows"/>: one hidden-size block per
        /// tapped layer (<see cref="SetFeatureTaps"/>), 0 without taps.</summary>
        public int FeatureWidth => _featureCount * _m.NEmbd;

        /// <summary>The last verify's feature taps, [rows][FeatureWidth] F32 on the device.</summary>
        public IntPtr FeatureRows => _features;

        /// <summary>Copies the last Verify's feature rows for numerical diagnostics. The
        /// caller supplies at least rows * FeatureWidth floats; normal inference leaves
        /// these rows on the device and does not incur this synchronization/readback.</summary>
        public void ReadFeatureRows(float[] destination, int rows)
        {
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            long count = (long)rows * FeatureWidth;
            if (rows < 0 || rows > MaxVerifyRows || destination.LongLength < count)
                throw new ArgumentException("[q35-cuda] feature output shape is invalid");
            if (count == 0)
                return;
            if (_features == IntPtr.Zero)
                throw new InvalidOperationException("[q35-cuda] no Verify feature rows are available");
            MakeCurrent();
            Sync();
            fixed (float* dst = destination)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)dst, _features, new UIntPtr((ulong)count * 4)).ThrowOnError();
        }

        private int ConvDim => (2 * _m.GdnKHeads + _m.GdnVHeads) * Q35Kernels.GdnHeadDim;
        private int ValueDim => _m.GdnVHeads * Q35Kernels.GdnHeadDim;
        private int GdnPackedWidth => ConvDim + ValueDim + 2 * _m.GdnVHeads;
        private int AttnProjWidth => _m.NHead * 2 * _m.HeadDim + 2 * _m.NKvHead * _m.HeadDim;
        private int QDim => _m.NHead * _m.HeadDim;

        /// <summary>
        /// Whether this machine can run the engine: its kernels need the block-scaled FP4 tensor cores
        /// of compute capability 12.x (sm_120a), and their PTX/fatbin exist only where TensorSharp was
        /// built with nvcc. <paramref name="reason"/> says what is missing otherwise.
        /// </summary>
        public static bool IsSupported(out string reason)
        {
            reason = null;
            try
            {
                string path = CudaKernels.LocatePtxPath("tensorsharp_q35_kernels.ptx");
                if (path == null)
                {
                    reason = "tensorsharp_q35_kernels.ptx is missing (TensorSharp.Backends.Cuda was built without nvcc)";
                    return false;
                }
                var device = CudaDevice.GetDevice(0);
                if (device.ComputeCapabilityMajor != 12)
                {
                    reason = $"GPU 0 is sm_{device.ComputeCapabilityMajor}{device.ComputeCapabilityMinor}; the kernels need sm_12x FP4 tensor cores";
                    return false;
                }
                // A CC 12 GPU is not enough: the image the driver loads must be sm_120a / sm_121a
                // SASS (a PTX-only or other-architecture build traps in every GEMM), and the
                // managed path should serve the model rather than the engine refusing it later.
                return Q35Kernels.ProbeTensorCoreKernels(path, out reason);
            }
            catch (Exception ex)
            {
                reason = ex.Message;
                return false;
            }
        }

        public Q35CudaEngine(ModelDesc m)
        {
            _m = m ?? throw new ArgumentNullException(nameof(m));
            ValidateNinferModel(m);
            PreflightNinferFlashInfer();
            if (m.HeadDim != Q35Kernels.AttnHeadDim)
                throw new NotSupportedException($"[q35-cuda] attention head_dim {m.HeadDim} (the kernels take {Q35Kernels.AttnHeadDim})");
            if (m.NKvHead <= 0 || m.NHead % m.NKvHead != 0 || m.NHead / m.NKvHead > Q35Kernels.AttnBlockRows)
                throw new NotSupportedException($"[q35-cuda] {m.NHead} query heads over {m.NKvHead} KV heads");
            if (m.DConv < 2 || m.DConv > 8)
                throw new NotSupportedException($"[q35-cuda] convolution kernel {m.DConv}");
            if (m.SectT + m.SectH + m.SectW > Q35Kernels.RopeDims / 2)
                throw new NotSupportedException("[q35-cuda] M-RoPE sections exceed the rotated width");

            _servingTrace = ServingTraceLog.FromEnvironment();
            try
            {
                ValidateVerifyDecodeQ81Graphs(_verifyDecodeQ81 || _projectionTrace, _verifyGraphsEnabled, _draftGraphsEnabled);
                CudaBackend.Register();
                CudaDriverApi.cuInit(0);
                _alloc = new CudaAllocator(0);
                MakeCurrent();
                _k = Q35Kernels.Create(identifyImage: m.Ninfer != null);
                CudaDriverApi.cuDeviceGetAttribute(out _sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, 0).ThrowOnError();

                InitializeNinferModules();
                var sw = Stopwatch.StartNew();
                _layers = new Layer[m.NLayer];
                using (var loader = new Loader(this))
                {
                    for (int il = 0; il < m.NLayer; il++)
                    {
                        _layers[il] = m.Ninfer == null ? LoadLayer(loader, m.Layers[il]) : LoadNinferLayer(loader, il);
                        _layers[il].Ordinal = _layers[il].Recurrent ? _nGdn++ : _nAttn++;
                    }
                    if (m.Ninfer == null)
                        _tokEmbd = loader.Raw(m.TokEmbd, TQ4_0, out _tokEmbdRowBytes);
                    else
                        _tokEmbd = LoadNinferEmbedding(loader, m.Ninfer.Embedding);
                    if (m.Ninfer != null)
                        _ninferHead = LoadNinferMatrix(loader, m.Ninfer.Head);
                    else
                        _output = loader.Raw(m.Output, TQ6_K, out _outputRowBytes);
                }
                CompleteNinferLoadRepack();
                CompleteNinferFlashInferLoad();
                _outputNorm = UploadF32(m.OutputNorm);
                AllocateScratch();
                SetFeatureTaps(null);
                _active = CreateSlot(InitialSlotTokens);
                Console.Error.WriteLine($"[q35-cuda] weights loaded in {sw.Elapsed.TotalSeconds:F1}s, {_sms} SMs, n_ctx={m.NCtx}, ubatch={m.NUbatch}");
            }
            catch
            {
                if (_alloc != null)
                {
                    try { Dispose(); }
                    catch (Exception cleanup) { Console.Error.WriteLine("[q35-cuda] companion construction cleanup failed: " + cleanup.Message); }
                }
                _servingTrace?.Close("constructor_failed");
                throw;
            }
        }

        private IntPtr Stream => _alloc.Stream.Handle;
        private void MakeCurrent() => _alloc.Context.MakeCurrent();

        private IntPtr Alloc(long bytes, List<IntPtr> owner = null)
        {
            CudaDriverApi.cuMemAlloc(out IntPtr p, new UIntPtr((ulong)Math.Max(256, bytes))).ThrowOnError();
            (owner ?? _owned).Add(p);
            return p;
        }

        private IntPtr UploadF32(float[] data)
        {
            if (data == null)
                return IntPtr.Zero;
            IntPtr p = Alloc((long)data.Length * 4);
            fixed (float* src = data)
                CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)src, new UIntPtr((ulong)data.Length * 4)).ThrowOnError();
            return p;
        }

        private void Sync() => CudaDriverApi.cuStreamSynchronize(Stream).ThrowOnError();

        /// <summary>Waits for everything issued on the engine's stream.</summary>
        public void Synchronize()
        {
            MakeCurrent();
            Sync();
        }

        // ---------------------------------------------------------------------------------------
        // Loading
        // ---------------------------------------------------------------------------------------

        /// <summary>Streams weight bytes from their source through pinned host and device staging:
        /// NVFP4 matrices are repacked into planes on the device, other weights copied as stored.</summary>
        private sealed partial class Loader : IDisposable
        {
            private const long StagingBytes = 128L << 20;
            private readonly Q35CudaEngine _e;
            private readonly IntPtr _host, _dev;

            public Loader(Q35CudaEngine e)
            {
                _e = e;
                CudaDriverApi.cuMemHostAlloc(out _host, new UIntPtr((ulong)StagingBytes), 0).ThrowOnError();
                CudaDriverApi.cuMemAlloc(out _dev, new UIntPtr((ulong)StagingBytes)).ThrowOnError();
            }

            private static void Read(in CudaWeightDesc w, long offset, IntPtr dst, long bytes)
            {
                if (w.HostPtr != IntPtr.Zero)
                    Buffer.MemoryCopy((void*)(w.HostPtr + (nint)offset), (void*)dst, bytes, bytes);
                else
                    w.Source.Read(w.SourceOffset + offset, dst, bytes);
            }

            /// <summary>Repacks NVFP4 matrices that share K into consecutive rows of one plane pair.</summary>
            public Planes Fused(params CudaWeightDesc[] parts) => Fused(false, parts);

            /// <param name="interleave">Equal parts stored in groups of 8 rows of each in turn (the
            /// gate|up projection, so a GEMM tile holds both halves of its FFN columns).</param>
            public Planes Fused(bool interleave, params CudaWeightDesc[] parts)
            {
                int k = parts[0].Ne0, rows = 0;
                foreach (var w in parts)
                {
                    if (w.GgmlType != TNVFP4)
                        throw new NotSupportedException($"[q35-cuda] {w.Name}: type {w.GgmlType}; the engine takes NVFP4 projections");
                    if (w.Ne0 != k || k % Q35Kernels.GemmStageK != 0)
                        throw new NotSupportedException($"[q35-cuda] {w.Name}: input width {w.Ne0}");
                    rows += w.Ne1;
                }
                if (interleave)
                    foreach (var w in parts)
                        if (w.Ne1 != parts[0].Ne1 || w.Ne1 % 8 != 0)
                            throw new NotSupportedException($"[q35-cuda] {w.Name}: interleaved parts need equal row counts that are multiples of 8");
                var p = new Planes { Rows = rows, K = k, Codes = _e.Alloc((long)rows * k / 2), Scales = _e.Alloc((long)rows * k / 16) };
                int row = 0;
                for (int pi = 0; pi < parts.Length; pi++)
                {
                    var w = parts[pi];
                    long rowBytes = w.RowBytes;
                    int chunkRows = (int)Math.Max(1, StagingBytes / rowBytes);
                    for (int r0 = 0; r0 < w.Ne1; r0 += chunkRows)
                    {
                        int n = Math.Min(chunkRows, w.Ne1 - r0);
                        long bytes = n * rowBytes;
                        Read(w, r0 * rowBytes, _host, bytes);
                        CudaDriverApi.cuMemcpyHtoD(_dev, _host, new UIntPtr((ulong)bytes)).ThrowOnError();
                        long dstRow = row + r0;
                        if (interleave)
                            _e._k.RepackNvfp4(_dev, rowBytes, p.Codes, p.Scales, n, k, _e.Stream, r0, parts.Length, pi);
                        else
                            _e._k.RepackNvfp4(_dev, rowBytes, p.Codes + (nint)(dstRow * k / 2), p.Scales + (nint)(dstRow * k / 16), n, k, _e.Stream);
                        _e.Sync();
                    }
                    row += w.Ne1;
                }
                return p;
            }

            public IntPtr Raw(in CudaWeightDesc w, int type, out long rowBytes)
            {
                if (w.GgmlType != type)
                    throw new NotSupportedException($"[q35-cuda] {w.Name}: type {w.GgmlType}, expected {type}");
                rowBytes = w.RowBytes;
                IntPtr dst = _e.Alloc(w.TotalBytes);
                RawInto(w, dst);
                return dst;
            }

            /// <summary>The weight's bytes as stored, to <paramref name="dst"/>.</summary>
            public void RawInto(in CudaWeightDesc w, IntPtr dst)
            {
                long total = w.TotalBytes;
                for (long off = 0; off < total; off += StagingBytes)
                {
                    long bytes = Math.Min(StagingBytes, total - off);
                    Read(w, off, _host, bytes);
                    CudaDriverApi.cuMemcpyHtoD(dst + (nint)off, _host, new UIntPtr((ulong)bytes)).ThrowOnError();
                }
            }

            public void Dispose()
            {
                CudaDriverApi.cuMemFreeHost(_host);
                CudaDriverApi.cuMemFree(_dev);
            }
        }

        private Layer LoadLayer(Loader loader, LayerDesc d)
        {
            var L = new Layer { Recurrent = d.Recurrent };
            L.InProj = loader.Fused(d.InProj);
            L.OutProj = loader.Fused(d.OutProj);
            L.GateUp = loader.Fused(interleave: true, d.FfnGate, d.FfnUp);
            L.Down = loader.Fused(d.FfnDown);
            L.AttnNorm = UploadF32(d.AttnNorm);
            L.PostNorm = UploadF32(d.PostNorm);
            if (d.Recurrent)
            {
                L.ConvW = UploadF32(d.ConvW);
                L.DtBias = UploadF32(d.DtBias);
                L.SsmA = UploadF32(d.SsmA);
                L.SsmNorm = UploadF32(d.SsmNorm);
                if (L.InProj.Rows != GdnPackedWidth)
                    throw new NotSupportedException($"[q35-cuda] GDN in-projection rows {L.InProj.Rows}, expected {GdnPackedWidth}");
            }
            else
            {
                L.QNorm = UploadF32(d.QNorm);
                L.KNorm = UploadF32(d.KNorm);
                if (L.InProj.Rows != AttnProjWidth)
                    throw new NotSupportedException($"[q35-cuda] attention projection rows {L.InProj.Rows}, expected {AttnProjWidth}");
            }
            return L;
        }

        /// <summary>Rows one forward may carry: a full prompt chunk and a verify window beside it
        /// (<see cref="MixedVerify"/>).</summary>
        private int RowCap => _m.NUbatch + MaxVerifyRows;

        private void AllocateScratch()
        {
            int nt = RowCap, e = _m.NEmbd;
            int widest = Math.Max(Math.Max(AttnProjWidth, GdnPackedWidth), 2 * _m.NFf);
            int widestK = Math.Max(Math.Max(e, _m.NFf), Math.Max(QDim, ValueDim));
            SharedScratchEnabled = SharedScratchRequested(Environment.GetEnvironmentVariable("TS_Q35_SHARED_SCRATCH"));
            _x = Alloc((long)nt * e * 4);
            if (SharedScratchEnabled)
                AllocateSharedScratch(nt);
            else
                _xn = Alloc((long)nt * e * 4);
            _block = Alloc((long)nt * e * 4);
            _proj = Alloc((long)nt * widest * 4);
            if (!SharedScratchEnabled)
            {
                _q = Alloc((long)nt * QDim * 4);
                _gate = Alloc((long)nt * QDim * 4);
                _attn = Alloc((long)nt * QDim * 4);
                _gdnQkv = Alloc((long)nt * ConvDim * 4);
                _gdnG = Alloc((long)nt * _m.GdnVHeads * 4);
                _gdnBeta = Alloc((long)nt * _m.GdnVHeads * 4);
                _gdnO = Alloc((long)nt * ValueDim * 4);
                _gdnY = Alloc((long)nt * ValueDim * 4);
            }
            if (Q35Kernels.GdnPreparedPrefillEnabled && _nGdn > 0)
            {
                _gdnPreparedScratchBytes = Q35Kernels.GdnPreparedScratchCapacityBytes(nt, MaxSeqs, _m.GdnKHeads, _m.GdnVHeads);
                // Every GDN group's packed projection lives below this bound, including a
                // mixed prefill/Verify step. Its unused tail can hold the packets until
                // GdnOut finishes. The following FFN may overwrite that tail on the same
                // stream; no packet is used by Commit or by the next layer. Only _proj's
                // allocation is owned, so this alias must never be separately freed.
                long packedBytes = checked((long)nt * GdnPackedWidth * sizeof(float));
                long tailOffset = checked(packedBytes + 255) & ~255L;
                long tailBytes = checked((long)nt * widest * sizeof(float)) - tailOffset;
                if (tailBytes >= _gdnPreparedScratchBytes)
                {
                    _gdnPreparedScratch = _proj + (nint)tailOffset;
                    PreparedGdnScratchReusesProjection = true;
                }
                else
                    _gdnPreparedScratch = Alloc(_gdnPreparedScratchBytes);
                Console.Error.WriteLine($"[q35-cuda] prepared GDN prefill enabled, scratch={_gdnPreparedScratchBytes} bytes, projectionTail={PreparedGdnScratchReusesProjection}");
            }
            if (!SharedScratchEnabled)
                _act = Alloc((long)nt * _m.NFf * 4);
            PublishScratchSpans(nt);
            if (nt < MaxVerifyRows)
                throw new NotSupportedException($"[q35-cuda] ubatch {nt} below the {MaxVerifyRows}-row verify window");
            // q8_1 rows: a GEMV's activations, or the LM head's over a whole verify.
            long q8Rows = Math.Max(Q35Kernels.GemvMaxRows, MaxVerifyRows);
            _q8 = Alloc(Math.Max(Q35Kernels.GemvMaxRows * (long)widestK, q8Rows * e));
            _q8d = Alloc(Math.Max(Q35Kernels.GemvMaxRows * (long)(widestK / 32), q8Rows * (e / 32)) * 4);
            _fp4c = Alloc((long)nt * widestK / 2);
            _fp4s = Alloc((long)nt * widestK / 16);
            _fp4r = Alloc((long)nt * 4);
            // Split-K partials for verify-size launches (<= 64 rows, <= 8 splits).
            _partial = Alloc(8L * 64 * widest * 4);
            _counters = Alloc(1L << 20);
            CudaDriverApi.cuMemsetD8(_counters, 0, new UIntPtr(1u << 20)).ThrowOnError();
            // The q8_0 split kernel (GGUF route) takes up to 160 splits, the NInfer pipelined kernel 64.
            _attnPartial = Alloc(Q35Kernels.AttnSplitFloats(MaxVerifyRows, _m.NHead,
                _m.Ninfer == null ? Q35Kernels.AttnMaxSplits : Q35Kernels.AttnMaxSplitCount) * 4);
            _logits = Alloc((long)MaxVerifyRows * _m.NVocab * 4);
            _argmax = Alloc(MaxVerifyRows * 4);
            _lastX = Alloc((long)MaxSeqs * e * 4);
            _lastBlock = Alloc((long)MaxSeqs * e * 4);
            // Per row: token, four M-RoPE components, cache row, key end; then each sequence's real rows.
            _rowsDev = Alloc((7L * nt + MaxSeqs) * 4);
            _countDev = _rowsDev + (nint)(7L * nt * 4);
            _tokensDev = _rowsDev;
            _ropePosDev = _rowsDev + (nint)(1L * nt * 4);
            _cacheRowDev = _rowsDev + (nint)(5L * nt * 4);
            _keyEndDev = _rowsDev + (nint)(6L * nt * 4);
            _maxBlocks = (_m.NCtx + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens + 1;
            _tabBlocks = Alloc(8L * _maxBlocks * MaxSeqs);
            _tabConv = Alloc(8L * Math.Max(1, _nGdn) * MaxSeqs);
            _tabState = Alloc(8L * Math.Max(1, _nGdn) * MaxSeqs);
            _keptDev = Alloc(4L * MaxSeqs);
            CudaDriverApi.cuMemHostAlloc(out _pinned, new UIntPtr(PinnedBytes), 0).ThrowOnError();
        }

        private void EnsureVerifyBuffers()
        {
            if (_vQkv == IntPtr.Zero)
            {
                long rows = (long)Math.Max(1, _nGdn) * MaxVerifyRows;
                _vRaw = Alloc(rows * ConvDim * 4);
                _vQkv = Alloc(rows * ConvDim * 4);
                _vG = Alloc(rows * _m.GdnVHeads * 4);
                _vBeta = Alloc(rows * _m.GdnVHeads * 4);
            }
            // A feature layout may be selected after the first Verify. Grow before capture;
            // never leave an executable graph pointing at replaced or undersized storage.
            if (FeatureWidth > _featureBufferWidth)
            {
                InvalidateVerifyGraphs();
                // Preserve the old buffer/capacity if allocation fails, so a later request
                // selecting the old layout can still use it safely.
                IntPtr grown = Alloc((long)MaxVerifyRows * FeatureWidth * 4);
                if (_features != IntPtr.Zero)
                {
                    Sync();
                    CudaDriverApi.cuMemFree(_features).ThrowOnError();
                    _owned.Remove(_features);
                }
                _features = grown;
                _featureBufferWidth = FeatureWidth;
            }
        }

        // ---------------------------------------------------------------------------------------
        // Forward
        // ---------------------------------------------------------------------------------------

        /// <summary>Rows that replace the token embeddings of a forward: Count rows of the model
        /// width from Rows, or from device memory at Device (in this context, complete before the
        /// forward), at token Insert of the forward.</summary>
        public readonly record struct EmbeddingSpan(float[] Rows, int Count, int Insert, IntPtr Device = default);

        // The current forward's embedding overrides, consumed chunk by chunk by RunRows.
        private IReadOnlyList<EmbeddingSpan> _embedSpans;

        /// <summary>
        /// Runs <paramref name="tokens"/> through the active sequence from its current position and
        /// returns the logits of the last row in <paramref name="logitsOut"/> (vocab floats) when it
        /// is not null. <paramref name="ropePos"/> holds the four M-RoPE components [4][n] (null:
        /// text positions, every component the token's position plus the sequence's delta; with a
        /// table the delta afterwards is the last row's largest component + 1 - the new position).
        /// <paramref name="embeds"/> replace the embeddings of some tokens (images).
        /// </summary>
        public void Forward(int[] tokens, float[] logitsOut, int[] ropePos = null, IReadOnlyList<EmbeddingSpan> embeds = null)
        {
            if (tokens == null || tokens.Length == 0)
                return;
            MakeCurrent();
            var slots = new[] { _active };
            EnsureRoom(slots, tokens.Length);
            // The drafter keeps the last ringRows positions: tap only the rows that reach them.
            int tapFrom = !HasDrafter ? int.MaxValue : _active.NPast + tokens.Length - RingRows;
            _embedSpans = embeds;
            try
            {
                for (int start = 0; start < tokens.Length; start += _m.NUbatch)
                {
                    int n = Math.Min(_m.NUbatch, tokens.Length - start);
                    int first = !HasDrafter ? n : Math.Clamp(tapFrom - (_active.NPast), 0, n);
                    RunRows(slots, n, tokens, start, ropePos, tokens.Length, verify: false, tapFirst: first);
                }
            }
            finally
            {
                _embedSpans = null;
            }
            if (ropePos != null)
            {
                int last = tokens.Length - 1, total = tokens.Length;
                int max = Math.Max(ropePos[last], Math.Max(ropePos[total + last], ropePos[2 * total + last]));
                _active.RopeDelta = max + 1 - _active.NPast;
            }
            // Chunks share nothing but the caches, so only the last chunk's last row reaches the head.
            if (logitsOut != null)
                HeadLastRows(1, Math.Min(_m.NUbatch, tokens.Length - (tokens.Length - 1) / _m.NUbatch * _m.NUbatch), logitsOut, null);
        }

        /// <summary>
        /// One decode step for several sequences: token i advances sequence i. Fills
        /// <paramref name="logitsOut"/> ([n][vocab], optional) and <paramref name="argmaxOut"/> ([n],
        /// optional) from each sequence's row.
        /// </summary>
        public void DecodeBatch(Slot[] slots, int[] tokens, float[] logitsOut, int[] argmaxOut)
        {
            MakeCurrent();
            EnsureRoom(slots, 1);
            RunRows(slots, 1, tokens, 0, null, tokens.Length, verify: false, tapFirst: 0, decode: true);
            HeadLastRows(slots.Length, 1, logitsOut, argmaxOut);
        }

        /// <summary>
        /// Verifies a window of <paramref name="window"/> rows per sequence (tokens sequence-major,
        /// the anchor then its drafts) at each sequence's position: the attention caches take every
        /// row, the recurrent states stay as they were, <paramref name="drawnOut"/> receives every
        /// row's argmax and <see cref="FeatureRows"/> its taps (<paramref name="logitsOut"/>, when set,
        /// every row's logits). <see cref="Commit"/> then keeps a prefix of each window.
        /// </summary>
        public void Verify(Slot[] slots, int window, int[] tokens, int[] drawnOut, float[] logitsOut = null)
        {
            RequireNinferVerification();
            ValidateNinferVerifyRequest(slots, window, tokens, drawnOut, logitsOut);
            if (_m.Ninfer != null) ForgetNinferVerification();
            int n = slots.Length * window;
            if (window < 1 || n > MaxVerifyRows || tokens.Length < n || drawnOut.Length < n)
                throw new ArgumentException($"[q35-cuda] verify of {slots.Length} x {window} rows (at most {MaxVerifyRows})");
            MakeCurrent();
            EnsureRoom(slots, window);
            EnsureVerifyBuffers();
            var pass = SelectNinferResidualPass(NinferResidualNormEnabled, 1, window, slots.Length,
                true, false, _embedSpans != null, _featureCount != 0, _d != null);
            RunRows(slots, window, tokens, 0, null, n, verify: true, tapFirst: 0, passMode: pass);
            _verifyRow0 = 0;
            if (pass == NinferLayerPass.FusedResidualNorm && _ninferFinalResidualState == NinferResidualState.FusedCanonical)
                _k.PlainNinferNorm(_x, _outputNorm, _xn, n, _m.NEmbd, _m.Eps, Stream);
            else
                FinalNorm(_x, _block, _outputNorm, _xn, n, _m.NEmbd, _m.Eps, Stream);
            Head(_xn, n, logitsOut, drawnOut);
            RecordNinferVerification(slots, window);
            if (pass == NinferLayerPass.FusedResidualNorm) NinferResidualNormCompletedVerifies++;
        }

        // The row of the last verify's first window row in the step's activations (its drafter taps).
        private int _verifyRow0;

        /// <summary>
        /// A prompt chunk of <paramref name="prefill"/> and a <see cref="Verify"/> of
        /// <paramref name="slots"/> in one forward: every projection reads its weights once for both
        /// (a separate verify reads all of them again for a few rows). The chunk advances its sequence
        /// as <see cref="Forward"/> does and its last row's logits go to <paramref name="prefillLogits"/>;
        /// the window is left for <see cref="Commit"/> as a verify leaves it. Verify rows share the
        /// prompt's FP4 projections (as a batched verify past 8 rows does). <paramref name="ropePos"/>
        /// ([4][chunk]) and <paramref name="embeds"/> are the chunk's, as for <see cref="Forward"/>.
        /// </summary>
        public void MixedVerify(Slot prefill, int[] prefillTokens, float[] prefillLogits, Slot[] slots, int window, int[] tokens, int[] drawnOut,
            int[] ropePos = null, IReadOnlyList<EmbeddingSpan> embeds = null)
        {
            if (_m.Ninfer != null) throw new NotSupportedException("NInfer target mixed draft verification is not yet supported.");
            int p = prefillTokens.Length, nv = slots.Length * window, e = _m.NEmbd;
            if (p < 1 || window < 1 || nv > MaxVerifyRows || p + nv > RowCap || tokens.Length < nv || drawnOut.Length < nv
                || Array.IndexOf(slots, prefill) >= 0 || (ropePos != null && ropePos.Length < 4 * p))
                throw new ArgumentException($"[q35-cuda] mixed step of {p} prompt rows and {slots.Length} x {window} verify rows");
            MakeCurrent();
            EnsureRoom(new[] { prefill }, p);
            EnsureRoom(slots, window);
            EnsureVerifyBuffers();
            // The drafter keeps the last ringRows positions: tap only the prompt rows that reach them.
            int first = _d == null ? p : Math.Clamp(p - _ringRows, 0, p);
            _embedSpans = embeds;
            try
            {
                RunGroups(first,
                    new RowGroup(new[] { prefill }, p, false, null, prefillTokens, 0, ropePos, p),
                    new RowGroup(slots, window, true, null, tokens));
            }
            finally
            {
                _embedSpans = null;
            }
            if (ropePos != null)
            {
                int max = Math.Max(ropePos[p - 1], Math.Max(ropePos[2 * p - 1], ropePos[3 * p - 1]));
                prefill.RopeDelta = max + 1 - prefill.NPast;
            }
            _verifyRow0 = p;
            _k.CopyRows(_x + (nint)((long)(p - 1) * e * 4), e, _lastX, e, e, 1, Stream);
            _k.CopyRows(_block + (nint)((long)(p - 1) * e * 4), e, _lastBlock, e, e, 1, Stream);
            FinalNorm(_lastX, _lastBlock, _outputNorm, _xn, 1, e, _m.Eps, Stream);
            Head(_xn, 1, prefillLogits, null);
            FinalNorm(_x + (nint)((long)p * e * 4), _block + (nint)((long)p * e * 4), _outputNorm, _xn, nv, e, _m.Eps, Stream);
            Head(_xn, nv, null, drawnOut);
        }

        /// <summary>The rows the engine can carry next to a prompt chunk in <see cref="MixedVerify"/>.</summary>
        public int MixedRowCapacity => RowCap;

        /// <summary>Whether <see cref="MixedVerify"/> computes a chunk of <paramref name="chunkRows"/> and
        /// <paramref name="verifyRows"/> verify rows in the numerics each gets on its own: up to 8 rows a
        /// projection takes the q8_1 GEMV (8-bit activations), above it the FP4 GEMM that the mixed
        /// forward uses for every row.</summary>
        public static bool MixedKeepsNumerics(int chunkRows, int verifyRows) =>
            chunkRows > Q35Kernels.GemvMaxRows && verifyRows > Q35Kernels.GemvMaxRows;

        /// <summary>Keeps the first kept[i] rows (1 .. window) of sequence i's last verify: the
        /// recurrent states advance over them and the position moves past them.</summary>
        public void Commit(Slot[] slots, int window, int[] kept)
        {
            if (_m.Ninfer != null) { CommitNinfer(slots, window, kept); return; }
            MakeCurrent();
            BindSlots(slots);
            for (int i = 0; i < slots.Length; i++)
                if (kept[i] < 0 || kept[i] > window)
                    throw new ArgumentOutOfRangeException(nameof(kept));
            fixed (int* k = kept)
                Upload(_keptDev, k, slots.Length);
            if (_nGdn > 0)
            {
                _k.GdnConvState(_vRaw, ConvDim, (long)MaxVerifyRows * ConvDim, window, _keptDev, slots.Length, ConvDim, _m.DConv,
                    _tabConv, MaxSeqs, _nGdn, Stream);
                _k.GdnReplay(_vQkv, (long)MaxVerifyRows * ConvDim, ConvDim, _m.GdnKHeads, _m.GdnVHeads, window, _keptDev, _vG, _vBeta,
                    (long)MaxVerifyRows * _m.GdnVHeads, _tabState, MaxSeqs, slots.Length, _nGdn, Stream);
            }
            var before = new int[slots.Length];
            for (int i = 0; i < slots.Length; i++)
            {
                before[i] = slots[i].NPast;
                slots[i].NPast += kept[i];
            }
            if (_d != null)
                DraftInjectKept(slots, window, kept, before);
            Sync();
            _servingTrace?.CommitCompleted(slots, window, kept, before);
        }

        private void Upload(IntPtr dst, int* src, int count)
        {
            long bytes = (long)count * 4;
            if (bytes > PinnedBytes)
                throw new InvalidOperationException("[q35-cuda] staging overflow");
            Buffer.MemoryCopy(src, (void*)_pinned, bytes, bytes);
            CudaDriverApi.cuMemcpyHtoDAsync(dst, _pinned, new UIntPtr((ulong)bytes), Stream).ThrowOnError();
            // The pinned buffer is reused by the next upload: let this copy land first.
            Sync();
        }

        /// <summary>
        /// rowsPerSeq rows of every sequence in <paramref name="slots"/> (tokens[start ..], sequence-
        /// major) through the layers, from each sequence's position. verify: the recurrent states
        /// stay and the window's inputs go to the verify buffers; otherwise the states advance and
        /// each position moves past its rows. Leaves the last layer's FFN output pending in _block.
        /// </summary>
        private void RunRows(Slot[] slots, int rowsPerSeq, int[] tokens, int start, int[] ropePos, int ropeTotal, bool verify, int tapFirst,
            int[] counts = null, bool decode = false, NinferLayerPass passMode = NinferLayerPass.Legacy)
            => RunGroups(tapFirst, new RowGroup(slots, rowsPerSeq, verify, counts, tokens, start, ropePos, ropeTotal, decode, passMode));

        /// <summary>rowsPerSeq rows of each of Slots (Tokens[Start ..], sequence-major); verify as in
        /// <see cref="RunRows"/>. Counts: the real rows of each sequence (padding beyond).</summary>
        internal readonly struct RowGroup
        {
            public readonly Slot[] Slots;
            public readonly int RowsPerSeq, Start, RopeTotal;
            public readonly bool Verify, Decode;
            public readonly int[] Counts, Tokens, RopePos;
            internal readonly NinferLayerPass PassMode;

            public RowGroup(Slot[] slots, int rowsPerSeq, bool verify, int[] counts, int[] tokens, int start = 0, int[] ropePos = null,
                int ropeTotal = 0, bool decode = false, NinferLayerPass passMode = NinferLayerPass.Legacy)
            {
                if (passMode != NinferLayerPass.Legacy && (passMode != NinferLayerPass.FusedResidualNorm
                    || !verify || decode || counts != null || rowsPerSeq != 8 || slots?.Length != 1))
                    throw new ArgumentException("The fused residual/norm pass requires one complete T8 verification lane.");
                Slots = slots;
                RowsPerSeq = rowsPerSeq;
                Verify = verify;
                Decode = decode;
                Counts = counts;
                Tokens = tokens;
                Start = start;
                RopePos = ropePos;
                RopeTotal = ropeTotal;
                PassMode = passMode;
            }

            public int Rows => Slots.Length * RowsPerSeq;
        }

        /// <summary>Groups of rows through the layers in one forward, each group's rows after the
        /// previous group's: the projections and the FFN run over every row, attention and the
        /// recurrence per group over its own sequences. Rows from tapFirst on feed the drafter.</summary>
        private void RunGroupsCore(int tapFirst, RowGroup[] groups)
        {
            RequireNoPendingNinferVerification();
            int e = _m.NEmbd, nt = RowCap, n = 0, nSeq = 0;
            foreach (var g in groups)
            {
                if (g.PassMode != NinferLayerPass.Legacy && SelectNinferResidualPass(NinferResidualNormEnabled,
                    groups.Length, g.RowsPerSeq, g.Slots.Length, g.Verify, g.Counts != null,
                    _embedSpans != null, _featureCount != 0, _d != null, tapFirst) != g.PassMode)
                    throw new InvalidOperationException("The requested residual/norm pass is outside its admitted profile.");
                n += g.Rows;
                nSeq += g.Slots.Length;
            }
            if (n > nt)
                throw new ArgumentException($"[q35-cuda] {n} rows in one step (at most {nt})");
            ResetProjectionDispatchStats();
            var all = new Slot[nSeq];
            for (int gi = 0, q0 = 0; gi < groups.Length; q0 += groups[gi].Slots.Length, gi++)
                Array.Copy(groups[gi].Slots, 0, all, q0, groups[gi].Slots.Length);
            BindSlots(all);
            var h = (int*)_pinned;
            if ((7L * nt + MaxSeqs) * 4 > PinnedBytes)
                throw new InvalidOperationException("[q35-cuda] staging overflow");
            for (int gi = 0, r0 = 0, q0 = 0; gi < groups.Length; r0 += groups[gi].Rows, q0 += groups[gi].Slots.Length, gi++)
            {
                var g = groups[gi];
                for (int q = 0; q < g.Slots.Length; q++)
                {
                    h[7 * nt + q0 + q] = g.Counts != null ? g.Counts[q] : g.RowsPerSeq;
                    for (int t = 0; t < g.RowsPerSeq; t++)
                    {
                        int li = q * g.RowsPerSeq + t, i = r0 + li, pos = g.Slots[q].NPast + t, rope = pos + g.Slots[q].RopeDelta;
                        h[i] = g.Tokens[g.Start + li];
                        // The M-RoPE kernel reads component c of row i at ropePos[c * RowCap + i].
                        for (int c = 0; c < 4; c++)
                            h[nt + c * nt + i] = g.RopePos != null ? g.RopePos[c * g.RopeTotal + g.Start + li] : (c == 3 ? 0 : rope);
                        h[5 * nt + i] = pos;
                        h[6 * nt + i] = pos + 1;
                    }
                }
            }
            CudaDriverApi.cuMemcpyHtoDAsync(_rowsDev, _pinned, new UIntPtr((7UL * (ulong)nt + MaxSeqs) * 4), Stream).ThrowOnError();
            Sync();

            if (_m.Ninfer == null)
                _k.EmbedQ4_0(_tokEmbd, _tokEmbdRowBytes, _tokensDev, _x, n, e, Stream);
            else
                _k.EmbedFp8(_tokEmbd, _tokEmbd + (nint)((long)_m.NVocab * e), _tokensDev, _x, n, e, _m.NVocab, Stream);
            if (_embedSpans != null)
            {
                // Embedding overrides belong to the first group (a forward's prompt chunk).
                int start = groups[0].Start, n0 = groups[0].Rows;
                foreach (var span in _embedSpans)
                {
                    // The span's rows that fall in this chunk [start, start + n0).
                    int lo = Math.Max(span.Insert, start), hi = Math.Min(span.Insert + span.Count, start + n0);
                    if (lo >= hi)
                        continue;
                    long bytes = (long)(hi - lo) * e * 4;
                    IntPtr dst = _x + (nint)((long)(lo - start) * e * 4);
                    // On the stream, after the embedding lookup it overwrites (pageable: the call
                    // returns once the source is staged).
                    if (span.Device != IntPtr.Zero)
                        CudaDriverApi.cuMemcpyDtoDAsync(dst, span.Device + (nint)((long)(lo - span.Insert) * e * 4), new UIntPtr((ulong)bytes), Stream).ThrowOnError();
                    else
                        fixed (float* src = &span.Rows[(long)(lo - span.Insert) * e])
                            CudaDriverApi.cuMemcpyHtoDAsync(dst, (IntPtr)src, new UIntPtr((ulong)bytes), Stream).ThrowOnError();
                }
            }
            var maxKeys = new int[groups.Length];
            for (int gi = 0; gi < groups.Length; gi++)
            {
                foreach (var s in groups[gi].Slots)
                    maxKeys[gi] = Math.Max(maxKeys[gi], s.NPast);
                maxKeys[gi] += groups[gi].RowsPerSeq;
            }
            if (!TryRunNinferDecodeGraph(n, tapFirst, groups, maxKeys))
                RunVerifyLayerPass(n, tapFirst, groups, maxKeys);
            for (int gi = 0, r0 = 0, q0 = 0; gi < groups.Length; r0 += groups[gi].Rows, q0 += groups[gi].Slots.Length, gi++)
            {
                var g = groups[gi];
                if (g.Verify)
                    continue;
                var before = new int[g.Slots.Length];
                for (int q = 0; q < g.Slots.Length; q++)
                {
                    before[q] = g.Slots[q].NPast;
                    g.Slots[q].NPast += g.Counts != null ? g.Counts[q] : g.RowsPerSeq;
                }
                if (_d != null)
                    DraftInjectRun(g.Slots, g.RowsPerSeq, Math.Clamp(tapFirst - r0, 0, g.Rows), before, g.Counts, r0, q0);
                if (HasNinferDrafter)
                    InjectNinferDraftRun(g, r0, q0, tapFirst, before);
            }
        }

        // Device submissions only. BindSlots, metadata uploads, embedding overrides, position
        // updates, drafter injection and all host reads stay in RunGroups or its caller.
        private void RunLayerKernels(int n, int tapFirst, RowGroup[] groups, int[] maxKeys)
        {
            if (_m.Ninfer != null)
            {
                RunNinferLayers(n, tapFirst, groups, maxKeys);
                return;
            }
            if (TryRunVerifyDecodeQ81(n, tapFirst, groups, maxKeys))
                return;
            _servingTrace?.LayerBranch(n <= Q35Kernels.GemvMaxRows ? "native_default_W4A8_Q81" : "native_default_W4A4");
            int e = _m.NEmbd;
            IntPtr pending = IntPtr.Zero;   // the previous block's output, added before the next norm
            for (int il = 0; il < _m.NLayer; il++)
            {
                var L = _layers[il];
                // Above the GEMV's rows the pre-norm feeds the FP4 GEMM's quantizer directly.
                bool fused = n > Q35Kernels.GemvMaxRows;
                if (fused)
                    _k.AddRmsNormNvfp4(_x, pending, L.AttnNorm, _m.Eps, _fp4c, _fp4s, _fp4r, e, n, Stream);
                else
                    _k.AddRmsNorm(_x, pending, L.AttnNorm, _xn, n, e, _m.Eps, Stream);
                if (_featureSlot != null && _featureSlot[il] >= 0)
                    for (int gi = 0, r0 = 0; gi < groups.Length; r0 += groups[gi].Rows, gi++)
                        if (groups[gi].Verify)
                            _k.CopyRows(_x + (nint)((long)r0 * e * 4), e, _features + (nint)((long)_featureSlot[il] * e * 4), FeatureWidth, e,
                                groups[gi].Rows, Stream);
                if (_d != null)
                    DraftTap(il, tapFirst, n);
                if (fused)
                    GemmObserved("InProj", L.InProj, _proj, n);
                else
                    ProjectObserved("InProj", L.InProj, _xn, _proj, n);
                for (int gi = 0, r0 = 0, q0 = 0; gi < groups.Length; r0 += groups[gi].Rows, q0 += groups[gi].Slots.Length, gi++)
                {
                    if (L.Recurrent)
                        GdnCore(L, groups[gi], r0, q0);
                    else
                        AttnCore(L, groups[gi], r0, q0, maxKeys[gi]);
                }
                if (L.Recurrent)
                {
                    _k.GdnOut(_gdnO, _proj, GdnPackedWidth, ConvDim, n, _m.GdnVHeads, L.SsmNorm, _m.Eps, _gdnY, Stream);
                    ProjectObserved("OutProj", L.OutProj, _gdnY, _block, n);
                }
                else
                    ProjectObserved("OutProj", L.OutProj, _attn, _block, n);
                if (fused)
                {
                    _k.AddRmsNormNvfp4(_x, _block, L.PostNorm, _m.Eps, _fp4c, _fp4s, _fp4r, e, n, Stream);
                    _k.GemmW4A4(L.GateUp.Codes, L.GateUp.Scales, L.GateUp.Rows, L.GateUp.K, _fp4c, _fp4s, _fp4r, n, 1.0f, _act,
                        _partial, _counters, _sms, 1, Stream, swiglu: true);
                    RecordProjectionDispatch("GateUp", "NVFP4_W4A4", 0, n);
                }
                else
                {
                    _k.AddRmsNorm(_x, _block, L.PostNorm, _xn, n, e, _m.Eps, Stream);
                    ProjectObserved("GateUp", L.GateUp, _xn, _proj, n);
                }
                if (!fused)
                {
                    _k.SwiGlu(_proj, _act, _m.NFf, n, Stream, interleaved: true);
                    ProjectObserved("Down", L.Down, _act, _block, n);
                }
                else
                {
                    // SwiGLU happened in the gate|up GEMM's epilogue (half the bytes written and read back).
                    _k.QuantizeNvfp4(_act, _fp4c, _fp4s, _fp4r, _m.NFf, n, Stream);
                    GemmObserved("Down", L.Down, _block, n);
                }
                pending = _block;
            }
        }

        /// <summary>
        /// Prompt chunks of several sequences in shared forwards: chunk i continues sequence i from its
        /// position. Groups of sequences run padded to their longest chunk (padding rows leave every
        /// state as it was: zero beta and decay for the recurrence, the convolution advanced over the
        /// real rows only, attention rows past each sequence never read). Fills logitsOut[i] with the
        /// logits of chunk i's last token.
        /// </summary>
        public void PrefillBatch(Slot[] slots, int[][] tokens, float[][] logitsOut)
        {
            MakeCurrent();
            int e = _m.NEmbd;
            for (int i = 0; i < slots.Length;)
            {
                int maxLen = tokens[i].Length, j = i + 1;
                while (j < slots.Length && j - i < MaxSeqs)
                {
                    int m = Math.Max(maxLen, tokens[j].Length);
                    if ((long)(j - i + 1) * m > _m.NUbatch)
                        break;
                    maxLen = m;
                    j++;
                }
                if (maxLen > _m.NUbatch)
                    throw new ArgumentException($"[q35-cuda] a {maxLen}-token prefill chunk exceeds the {_m.NUbatch}-row ubatch");
                var group = slots[i..j];
                var counts = new int[group.Length];
                var flat = new int[group.Length * maxLen];
                for (int q = 0; q < group.Length; q++)
                {
                    int[] t = tokens[i + q];
                    counts[q] = t.Length;
                    for (int r = 0; r < maxLen; r++)
                        flat[q * maxLen + r] = t[Math.Min(r, t.Length - 1)];
                }
                EnsureRoom(group, maxLen);
                RunRows(group, maxLen, flat, 0, null, flat.Length, verify: false, tapFirst: 0, counts);
                for (int q = 0; q < group.Length; q++)
                {
                    long row = (long)q * maxLen + counts[q] - 1;
                    _k.CopyRows(_x + (nint)(row * e * 4), e, _lastX + (nint)((long)q * e * 4), e, e, 1, Stream);
                    _k.CopyRows(_block + (nint)(row * e * 4), e, _lastBlock + (nint)((long)q * e * 4), e, e, 1, Stream);
                }
                FinalNorm(_lastX, _lastBlock, _outputNorm, _xn, group.Length, e, _m.Eps, Stream);
                var flatLogits = new float[(long)group.Length * _m.NVocab];
                Head(_xn, group.Length, flatLogits, null);
                for (int q = 0; q < group.Length; q++)
                {
                    logitsOut[i + q] = new float[_m.NVocab];
                    Array.Copy(flatLogits, (long)q * _m.NVocab, logitsOut[i + q], 0, _m.NVocab);
                }
                i = j;
            }
        }

        /// <summary>The final norm and LM head over the last row of each of nSeq runs of rowsPerSeq
        /// rows (the residual in _x, its pending FFN output in _block).</summary>
        private void HeadLastRows(int nSeq, int rowsPerSeq, float[] logitsOut, int[] argmaxOut)
        {
            int e = _m.NEmbd;
            long last = (long)(rowsPerSeq - 1) * e * 4;
            _k.CopyRows(_x + (nint)last, (long)rowsPerSeq * e, _lastX, e, e, nSeq, Stream);
            _k.CopyRows(_block + (nint)last, (long)rowsPerSeq * e, _lastBlock, e, e, nSeq, Stream);
            FinalNorm(_lastX, _lastBlock, _outputNorm, _xn, nSeq, e, _m.Eps, Stream);
            Head(_xn, nSeq, logitsOut, argmaxOut);
        }

        /// <summary>LM head over <paramref name="rows"/> normed rows: logits to the host when
        /// <paramref name="logitsOut"/> is set, row argmaxes when <paramref name="argmaxOut"/> is.</summary>
        private void Head(IntPtr normed, int rows, float[] logitsOut, int[] argmaxOut)
        {
            int e = _m.NEmbd;
            if (_ninferHead != null)
                ProjectNinferFp8(_ninferHead, normed, _logits, rows, Q35Kernels.Fp8Epilogue.Round, isVerify: true);
            else
            {
                _k.QuantizeQ81(normed, _q8, _q8d, e, rows, Stream);
                _k.GemvQ6k(_output, _outputRowBytes, _m.NVocab, e, _q8, _q8d, _logits, rows, Stream);
            }
            if (argmaxOut != null)
                _k.Argmax(_logits, rows, _m.NVocab, _argmax, Stream);
            Sync();
            if (logitsOut != null)
                fixed (float* dst = logitsOut)
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)dst, _logits, new UIntPtr((ulong)rows * (ulong)_m.NVocab * 4)).ThrowOnError();
            if (argmaxOut != null)
                fixed (int* dst = argmaxOut)
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)dst, _argmax, new UIntPtr((ulong)rows * 4)).ThrowOnError();
            _servingTrace?.HeadCompleted(rows, _m.NVocab, logitsOut != null, argmaxOut != null);
        }

        /// <summary>output[n, w.Rows] = input[n, w.K] W^T: the q8_1 GEMV for up to 8 rows, the FP4
        /// tensor-core GEMM above.</summary>
        private void Project(in Planes w, IntPtr input, IntPtr output, int n)
        {
            if (n <= Q35Kernels.GemvMaxRows)
            {
                _k.QuantizeQ81(input, _q8, _q8d, w.K, n, Stream);
                _k.GemvNvfp4(w.Codes, w.Scales, w.Rows, w.K, _q8, _q8d, 1.0f, output, n, Stream);
                return;
            }
            _k.QuantizeNvfp4(input, _fp4c, _fp4s, _fp4r, w.K, n, Stream);
            GemmQuantized(w, output, n);
        }

        /// <summary>The FP4 GEMM over activations already in the NVFP4 planes.</summary>
        private void GemmQuantized(in Planes w, IntPtr output, int n)
        {
            int maxSplits = n <= 64 ? 8 : 1;
            _k.GemmW4A4(w.Codes, w.Scales, w.Rows, w.K, _fp4c, _fp4s, _fp4r, n, 1.0f, output, _partial, _counters, _sms, maxSplits, Stream);
        }

        private IntPtr _a8K, _a8Ks, _a8Vt, _a8Vs;
        private int _a8Cap;
        private int Attn8Capacity => (_m.NCtx + Q35Kernels.Attn8TileKeys - 1) / Q35Kernels.Attn8TileKeys * Q35Kernels.Attn8TileKeys;

        /// <summary>One sequence's prompt chunk (not a verify window, not the split decode path) with at
        /// least <see cref="Q35Kernels.Attn8MinKeys"/> visible keys, on Qwen3.5's 24/4 x 256 geometry.</summary>
        private bool UsesAttn8(in RowGroup g, int maxKeys)
            => Q35Kernels.Attn8Enabled && !g.Verify && g.Slots.Length == 1 && g.Counts == null
               && g.RowsPerSeq * (_m.NHead / _m.NKvHead) > Q35Kernels.AttnBlockRows
               && maxKeys >= Q35Kernels.Attn8MinKeys && maxKeys <= Attn8Capacity
               && _m.NHead == 24 && _m.NKvHead == 4 && _m.HeadDim == Q35Kernels.AttnHeadDim;

        private static readonly bool Attn8Check = Environment.GetEnvironmentVariable("TS_Q35_ATTN8_CHECK") == "1";

        /// <summary>Diagnostic (TS_Q35_ATTN8_CHECK=1): the q8_0 F16 prompt attention over the same rows,
        /// and the 8-bit result's error against it (relative RMS and largest absolute difference).</summary>
        private void CheckAttn8(Layer L, in RowGroup g, int r0, int q0, int maxKeys, long kOff, long vOff, IntPtr tab, nint qOff)
        {
            long count = (long)g.Rows * QDim;
            IntPtr reference = Alloc(count * 4);
            try
            {
                _k.Attention(_q + qOff, _gate + qOff, _keyEndDev + (nint)(4L * r0), tab, _maxBlocks, kOff, vOff, g.RowsPerSeq, g.Slots.Length,
                    _m.NHead, _m.NKvHead, 1.0f / MathF.Sqrt(_m.HeadDim), maxKeys, IntPtr.Zero, reference, _sms, Stream, _act, (long)RowCap * _m.NFf * 4);
                Sync();
                var a = new float[count]; var b = new float[count];
                fixed (float* pa = a) CudaDriverApi.cuMemcpyDtoH((IntPtr)pa, _attn + qOff, new UIntPtr((ulong)count * 4)).ThrowOnError();
                fixed (float* pb = b) CudaDriverApi.cuMemcpyDtoH((IntPtr)pb, reference, new UIntPtr((ulong)count * 4)).ThrowOnError();
                double err = 0, norm = 0, worst = 0;
                for (long i = 0; i < count; i++)
                {
                    double d = a[i] - b[i];
                    err += d * d; norm += (double)b[i] * b[i]; worst = Math.Max(worst, Math.Abs(d));
                }
                Console.Error.WriteLine($"[q35-cuda] attn8 check layer {L.Ordinal} rows {g.Rows} keys {maxKeys}: rel-rms {Math.Sqrt(err / Math.Max(norm, 1e-30)):E2} max-abs {worst:E2}");
            }
            finally { CudaDriverApi.cuMemFree(reference); _owned.Remove(reference); }
        }

        /// <summary>The band for at least <paramref name="keys"/> visible keys, grown in steps of
        /// <see cref="Attn8BandStepKeys"/> up to the context; <paramref name="cap"/> is its key capacity
        /// (the band's layout stride). It is scratch: it claims no growth reserve, a short slot or block
        /// allocation releases it (<see cref="ReleaseAttn8Band"/>), and when even the band does not fit the
        /// chunk runs the 16-bit kernels (false). Sized to the context up front it held 540 MB through a
        /// 256K context's whole life, and the shared-prefix checkpoint of a 258K-token document found no
        /// room for its slot.</summary>
        private bool TryEnsureAttn8Band(int keys, out int cap)
        {
            cap = Math.Min(Attn8Capacity, (keys + Attn8BandStepKeys - 1) / Attn8BandStepKeys * Attn8BandStepKeys);
            if (_a8Cap >= cap)
            {
                cap = _a8Cap;
                return true;
            }
            ReleaseAttn8Band();
            var (k, ks, vt, vs) = Q35Kernels.Attn8BandBytes(cap, _m.NKvHead);
            try
            {
                EnsureDeviceMemory(k + ks + vt + vs, reserveBytes: 0);
                _a8K = Alloc(k, _a8Owned); _a8Ks = Alloc(ks, _a8Owned); _a8Vt = Alloc(vt, _a8Owned); _a8Vs = Alloc(Math.Max(vs, 16), _a8Owned);
            }
            catch (Exception ex) when (ex is OutOfMemoryException or CudaException)
            {
                ReleaseAttn8Band();
                if (!_attn8FallbackReported)
                {
                    _attn8FallbackReported = true;
                    Console.Error.WriteLine($"[q35-cuda] no device room for the {cap}-key 8-bit attention band; such chunks run the 16-bit kernels");
                }
                cap = 0;
                return false;
            }
            _a8Cap = cap;
            return true;
        }

        private bool _attn8FallbackReported;
        private const int Attn8BandStepKeys = 16384;
        private readonly List<IntPtr> _a8Owned = new List<IntPtr>(4);

        /// <summary>Frees scratch the next forward rebuilds (the 8-bit prompt-attention band). True when it held memory.</summary>
        public bool ReleaseTransientScratch()
        {
            MakeCurrent();
            return ReleaseAttn8Band();
        }

        /// <summary>Frees the 8-bit attention band (the stream is synchronized first). True when it held memory.</summary>
        internal bool ReleaseAttn8Band()
        {
            if (_a8Owned.Count == 0) return false;
            Sync();
            foreach (IntPtr b in _a8Owned) CudaDriverApi.cuMemFree(b);
            _a8Owned.Clear();
            _a8K = _a8Ks = _a8Vt = _a8Vs = IntPtr.Zero;
            _a8Cap = 0;
            return true;
        }

        /// <summary>Attention of one group's rows (rows r0.., sequences q0.. of the bound tables) over
        /// their sequences' caches, into _attn rows r0..; the out-projection follows for all rows.</summary>
        private void AttnCore(Layer L, in RowGroup g, int r0, int q0, int maxKeys)
        {
            int n = g.Rows;
            long kOff = KvLayerOffset(L), vOff = kOff + KvLayerBytes;
            IntPtr tab = _tabBlocks + (nint)((long)q0 * _maxBlocks * 8);
            nint qOff = (nint)((long)r0 * QDim * 4);
            _k.AttnPrep(_proj + (nint)((long)r0 * AttnProjWidth * 4), n, g.RowsPerSeq, _m.NHead, _m.NKvHead, L.QNorm, L.KNorm, _m.Eps,
                _ropePosDev + (nint)(4L * r0), RowCap, _m.SectT, _m.SectH, _m.SectW, _m.RopeBase, _cacheRowDev + (nint)(4L * r0), tab,
                _maxBlocks, kOff, vOff, _q + qOff, _gate + qOff, Stream);
            if (UsesAttn8(g, maxKeys) && TryEnsureAttn8Band(maxKeys, out int cap))
            {
                // 8-bit prompt attention: the visible keys once into the band, then INT8 QK / E4M3 PV.
                _k.Attn8Band(Q35Kernels.Attn8Cache.Q8, tab, kOff, vOff, maxKeys, cap, _m.NKvHead, _a8K, _a8Ks, _a8Vt, _a8Vs, Stream);
                _k.Attn8Prefill(_q + qOff, _gate + qOff, _keyEndDev + (nint)(4L * r0), _a8K, _a8Ks, _a8Vt, _a8Vs, cap,
                    g.RowsPerSeq, _m.NHead, _m.NKvHead, 1.0f / MathF.Sqrt(_m.HeadDim), _attn + qOff, Stream);
                if (Attn8Check)
                    CheckAttn8(L, g, r0, q0, maxKeys, kOff, vOff, tab, qOff);
                return;
            }
            // The FFN activations are idle during attention: key-split partials of prompt chunks go there.
            _k.Attention(_q + qOff, _gate + qOff, _keyEndDev + (nint)(4L * r0), tab, _maxBlocks, kOff, vOff, g.RowsPerSeq, g.Slots.Length,
                _m.NHead, _m.NKvHead, 1.0f / MathF.Sqrt(_m.HeadDim), maxKeys, n <= MaxVerifyRows ? _attnPartial : IntPtr.Zero, _attn + qOff,
                _sms, Stream, _act, (long)RowCap * _m.NFf * 4);
        }

        /// <summary>The Gated DeltaNet recurrence of one group's rows (rows r0.., sequences q0..) into
        /// _gdnO rows r0..; the output norm and projection follow for all rows. A verify group's
        /// inputs go to the verify buffers (one verify group per step).</summary>
        private void GdnCore(Layer L, in RowGroup g, int r0, int q0)
        {
            int n = g.Rows, hv = _m.GdnVHeads;
            long rowsOff = (long)L.Ordinal * MaxVerifyRows;
            bool verify = g.Verify;
            IntPtr proj = _proj + (nint)((long)r0 * GdnPackedWidth * 4);
            IntPtr qkv = verify ? _vQkv + (nint)(rowsOff * ConvDim * 4) : _gdnQkv + (nint)((long)r0 * ConvDim * 4);
            IntPtr gd = verify ? _vG + (nint)(rowsOff * hv * 4) : _gdnG + (nint)((long)r0 * hv * 4);
            IntPtr beta = verify ? _vBeta + (nint)(rowsOff * hv * 4) : _gdnBeta + (nint)((long)r0 * hv * 4);
            IntPtr raw = verify ? _vRaw + (nint)(rowsOff * ConvDim * 4) : IntPtr.Zero;
            IntPtr count = g.Counts != null ? _countDev + (nint)(4L * q0) : IntPtr.Zero;
            _k.GdnConv(proj, GdnPackedWidth, g.RowsPerSeq, g.Slots.Length, ConvDim, _m.DConv, L.ConvW, TabConv(L) + (nint)(8L * q0), qkv, raw,
                !verify, Stream, count);
            _k.GdnPrep(qkv, proj, GdnPackedWidth, n, _m.GdnKHeads, hv, ConvDim, ValueDim, L.DtBias, L.SsmA, _m.Eps, gd, beta, Stream,
                count, g.RowsPerSeq);
            _k.GdnScan(qkv, ConvDim, _m.GdnKHeads, hv, g.RowsPerSeq, g.Slots.Length, gd, beta, TabState(L) + (nint)(8L * q0),
                _gdnO + (nint)((long)r0 * ValueDim * 4), verify, Stream, _gdnPreparedScratch, _gdnPreparedScratchBytes);
        }

        public void Dispose()
        {
            try
            {
                MakeCurrent();
                Sync();
                FreeDraftGraphs();
                FreeVerifyGraphs();
                LogNinferResidualNormSummary();
                if (NinferGateUpSwiGluEnabled)
                    Console.Error.WriteLine($"[q35-cuda] NInfer GateUp SwiGLU host submissions={NinferGateUpSwiGluHostSubmissions}, graph captures={NinferGateUpSwiGluCaptureCount}, graph launches={NinferGateUpSwiGluReplayCount} (capture included; not individual GPU kernel replay counts), module={NinferGateUpSwiGluImageSha256}");
                InvalidateNinferDecodeGraphs();
                FreeSlot(_active);
                ReleaseFreeBlocks();
                ReleaseFreeArenas();
                FreeHostPool();
                foreach (IntPtr p in _owned)
                    CudaDriverApi.cuMemFree(p);
                _owned.Clear();
                foreach (IntPtr p in _a8Owned)
                    CudaDriverApi.cuMemFree(p);
                _a8Owned.Clear();
                if (_pinned != IntPtr.Zero)
                    CudaDriverApi.cuMemFreeHost(_pinned);
                _pinned = IntPtr.Zero;
                DisposeNinferLoadRepack();
                DisposeNinferA16Candidate();
                DisposeNinferExactVerify();
                DisposeNinferFlashInfer();
                DisposeNinferGdnChunk();
                DisposeNinferDraft();
                _ninfer?.Dispose();
                _ninferAttention?.Dispose();
                FreeNinferHostEmbedding();
                _k?.Dispose();
                _alloc?.Dispose();
            }
            finally
            {
                _servingTrace?.Close("engine_dispose_finally");
            }
        }
    }
}
