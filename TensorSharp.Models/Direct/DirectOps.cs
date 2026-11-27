// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend (BackendType.Cuda / BackendType.Cpu) execution primitives for
// the Wan video networks (WanDirectT5 / WanDirectDiT / WanDirectVae). The same
// model code runs on both allocators:
//   - Linear layers keep their GGUF quantization on CUDA (resident device
//     weights through CudaQuantizedOps' MMQ/dp4a/cuBLAS routing) and run as
//     dequantized F32 BLAS matmuls on CPU.
//   - Attention uses the streaming online-softmax CUDA kernel when available
//     (ts_wan_attn_*, O(seq) memory, optional additive bias for T5's relative
//     position bias) and a chunked per-head GEMM+softmax fallback otherwise.
//   - Small rowwise ops (bias add, AdaLN modulate/gate, interleaved RoPE) have
//     dedicated CUDA kernels and unsafe managed CPU loops — the generic
//     elementwise ops don't broadcast a [dim] vector over rows on every
//     backend, so these never rely on implicit broadcasting.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Direct
{
    /// <summary>Shared allocator context for the direct model implementations.</summary>
    internal sealed class DirectContext : IDisposable
    {
        public IAllocator Allocator { get; }
        public bool IsCuda { get; }
        public bool IsMlx { get; }
        public bool IsAccelerated => IsCuda || IsMlx;
        public CudaAllocator CudaAllocator { get; }
        public MlxAllocator MlxAllocator { get; }

        private readonly Dictionary<long, Tensor> _ones = new();
        private readonly List<Tensor> _owned = new();
        private readonly List<DirectLinear> _linears = new();

        public DirectContext(IAllocator allocator)
        {
            Allocator = allocator;
            CudaAllocator = allocator as CudaAllocator;
            IsCuda = CudaAllocator != null;
            MlxAllocator = allocator as MlxAllocator;
            IsMlx = MlxAllocator != null;
        }

        public Tensor NewF32(params long[] sizes) => new Tensor(Allocator, DType.Float32, sizes);
        public Tensor NewF32(ReadOnlySpan<long> sizes) => new Tensor(Allocator, DType.Float32, sizes);

        /// <summary>Cached all-ones gain vector (LayerNorm without affine).</summary>
        public Tensor Ones(long dim)
        {
            lock (_ones)
            {
                if (_ones.TryGetValue(dim, out var t)) return t;
                t = NewF32(dim);
                Ops.Fill(t, 1f);
                _ones[dim] = t;
                return t;
            }
        }

        /// <summary>Track a long-lived tensor (weights) for disposal with the context.</summary>
        public Tensor Own(Tensor t) { _owned.Add(t); return t; }
        internal DirectLinear Track(DirectLinear linear)
        {
            linear.RetainDeviceWeight();
            _linears.Add(linear);
            return linear;
        }

        /// <summary>Release unshared cached device weights while keeping host
        /// descriptors valid for a later forward pass.</summary>
        internal void ReleaseDeviceResidency()
        {
            foreach (var linear in _linears) linear.ReleaseDeviceResidency();
        }

        public Tensor FromFloats(float[] data, params long[] sizes)
        {
            var t = NewF32(sizes);
            t.SetElementsAsFloat(data);
            return t;
        }

        /// <summary>Hand back everything this context holds on the device.</summary>
        public void Dispose()
        {
            // Contexts share the model allocator. Release only their own
            // descriptors, preserving other live networks' resident weights.
            foreach (var linear in _linears) linear.Dispose();
            _linears.Clear();
            foreach (var t in _owned) t.Dispose();
            _owned.Clear();
            lock (_ones)
            {
                foreach (var t in _ones.Values) t.Dispose();
                _ones.Clear();
            }
        }
    }

    /// <summary>
    /// One linear layer y = x W^T (+ bias). On CUDA the weight stays in its GGUF
    /// storage type behind CudaQuantizedOps' per-allocator resident cache (keyed
    /// by the stable host pointer); on CPU it is dequantized once into an F32
    /// tensor whose transposed view feeds the BLAS matmul.
    /// </summary>
    internal sealed class DirectLinear : IDisposable
    {
        private readonly DirectContext _ctx;
        private readonly IntPtr _host;        // stable host bytes (GGUF mmap or owned dequant/copy)
        private readonly IntPtr _ownedHost;   // non-null when we allocated _host
        private readonly int _type;           // ggml type of _host
        private readonly long _bytes;
        private Tensor _wCpu;                 // CPU path: [ne1, ne0] F32
        private Tensor _wCpuT;                // transposed view [ne0, ne1]
        private Tensor _bias;                 // [ne1] F32 or null
        private readonly bool _prescale;      // q8_1-activation overflow guard (quantized types)
        private readonly bool _h3MappedDense; // H3 keeps dense mapped weights and F32 activations
        private readonly bool _h3PreserveFloat32; // H3 quantized projections avoid q8_1/Half activation narrowing
        private readonly CudaLtx25ProjectionSession _ltxProjection;
        private ResidentWeightLease _residentWeight;
        private bool _disposed;

        // Two model contexts can bind the same mapped tensor. Cache eviction
        // belongs to the final live descriptor, rather than the first context
        // disposed. CUDA caches by allocator; MLX caches by device ID.
        private sealed class ResidentWeightLease
        {
            private readonly record struct Key(CudaAllocator Cuda, int MlxDevice, IntPtr Host);
            private static readonly object Sync = new();
            private static readonly Dictionary<Key, int> Owners = new();
            private readonly DirectContext _context;
            private readonly Key _key;

            private ResidentWeightLease(DirectContext context, Key key)
            {
                _context = context;
                _key = key;
                lock (Sync)
                    Owners[key] = Owners.TryGetValue(key, out int count) ? count + 1 : 1;
            }

            public static ResidentWeightLease Acquire(DirectContext context, IntPtr host)
                => host == IntPtr.Zero || !context.IsAccelerated ? null
                    : new ResidentWeightLease(context, new Key(context.CudaAllocator,
                        context.IsMlx ? context.MlxAllocator.DeviceId : -1, host));

            private void Evict()
            {
                if (_context.IsCuda)
                    CudaQuantizedOps.ReleaseQuantizedWeight(_context.CudaAllocator, _key.Host);
                else
                    MlxQuantizedOps.ReleaseQuantizedWeight(_context.MlxAllocator, _key.Host);
            }

            public void Release()
            {
                lock (Sync)
                {
                    if (!Owners.TryGetValue(_key, out int count)) return;
                    if (count > 1) Owners[_key] = count - 1;
                    else
                    {
                        Owners.Remove(_key);
                        Evict();
                    }
                }
            }

            public void EvictIfUnshared()
            {
                lock (Sync)
                    if (Owners.TryGetValue(_key, out int count) && count == 1) Evict();
            }
        }

        public long InDim { get; }
        public long OutDim { get; }
        internal IntPtr CacheKey => _host;

        // The ggml activation-quantization (q8_1 blocks with FP16 block sums)
        // overflows for very large activations (UMT5-XXL hidden states reach the
        // thousands); scaling the activation down and the product back up is
        // exact for the scale-invariant quantized formats. Same constant as the
        // native wan_mm / qi_mm.
        private const float MmScale = 1024.0f;

        /// <summary>
        /// On CPU, keep a QUANTIZED weight in its GGUF storage type and let the
        /// managed quantized matmul read it directly, instead of expanding it to
        /// F32 once at load. The expansion costs 4x the memory and reads 4x the
        /// bytes on every forward; MiniMax-H3 alone would need ~130 GB for its 32B
        /// Q4_K text encoder. F16/BF16/F32 weights keep the plain GEMM, which is
        /// what every existing direct model (the Wan VAE especially) already uses.
        /// </summary>
        internal bool UseManagedQuantOnCpu =>
            _host != IntPtr.Zero &&
            ManagedQuantizedOps.SupportsCpuQuantizedStorage((GgmlTensorType)_type);

        private DirectLinear(DirectContext ctx, IntPtr host, IntPtr ownedHost, int type,
                                long ne0, long ne1, long bytes, Tensor bias, bool prescale, bool h3MappedDense = false,
                                bool h3PreserveFloat32 = false, CudaLtx25ProjectionSession ltxProjection = null)
        {
            _ctx = ctx; _host = host; _ownedHost = ownedHost; _type = type;
            InDim = ne0; OutDim = ne1; _bytes = bytes; _bias = bias; _prescale = prescale;
            _h3MappedDense = h3MappedDense;
            _h3PreserveFloat32 = h3PreserveFloat32;
            _ltxProjection = ltxProjection;
        }

        /// <summary>Load from a GGUF tensor (weights [ne1, ne0] in GGUF row-major).</summary>
        public static DirectLinear FromGguf(DirectContext ctx, GgufFile gguf, string weightName,
                                               string biasName = null, bool prescale = false, bool h3PreserveFloat32 = false,
                                               CudaLtx25ProjectionSession ltxProjection = null)
        {
            var info = gguf.Tensors[weightName];
            long ne0 = (long)info.Shape[0];
            long ne1 = 1;
            for (int i = 1; i < info.Shape.Length; i++) ne1 *= (long)info.Shape[i];
            long bytes = gguf.GetTensorByteCount(info);
            if (!gguf.TryGetTensorDataPointer(info, out IntPtr p))
                throw new System.IO.InvalidDataException($"GGUF tensor '{weightName}' has no data pointer.");

            Tensor bias = null;
            if (biasName != null && gguf.Tensors.ContainsKey(biasName))
                bias = ctx.Own(ctx.FromFloats(DirectOps.DequantTensor(gguf, biasName), (long)gguf.Tensors[biasName].NumElements));

            var lin = new DirectLinear(ctx, p, IntPtr.Zero, (int)info.Type, ne0, ne1, bytes, bias,
                                          prescale && !h3PreserveFloat32 && GgufFile.GetBlockSize(info.Type) > 1,
                                          h3PreserveFloat32: h3PreserveFloat32, ltxProjection: ltxProjection);
            if (ctx.IsMlx && MlxQuantizedOps.CanPreloadDirectFloatType((int)info.Type))
                MlxQuantizedOps.PreloadDirectFloatWeight(ctx.MlxAllocator, p, p, (int)info.Type, ne0, ne1, bytes);
            else if (!ctx.IsCuda && (ctx.IsMlx || !lin.UseManagedQuantOnCpu))
                lin.SetCpuWeight(DirectOps.DequantTensor(gguf, weightName), ne0, ne1);
            return ctx.Track(lin);
        }

        /// <summary>A contiguous block of rows [firstRow, firstRow + rowCount) of a 2-D GGUF weight as its own
        /// linear layer (a slice of a tied embedding used as an output head, for instance).</summary>
        public static DirectLinear FromGgufRows(DirectContext ctx, GgufFile gguf, string weightName, long firstRow, long rowCount)
        {
            var info = gguf.Tensors[weightName];
            if (info.Shape.Length != 2) throw new ArgumentException($"'{weightName}' is not a matrix.");
            long ne0 = (long)info.Shape[0], rows = (long)info.Shape[1];
            if (firstRow < 0 || rowCount <= 0 || firstRow + rowCount > rows)
                throw new ArgumentOutOfRangeException(nameof(rowCount), $"rows [{firstRow}, {firstRow + rowCount}) of {rows}.");
            long rowBytes = gguf.GetTensorByteCount(info) / rows;
            if (rowBytes * rows != gguf.GetTensorByteCount(info) || GgufFile.GetBlockSize(info.Type) > 1 && ne0 % GgufFile.GetBlockSize(info.Type) != 0)
                throw new NotSupportedException($"'{weightName}' rows are not byte-addressable.");
            if (!gguf.TryGetTensorDataPointer(info, out IntPtr p))
                throw new System.IO.InvalidDataException($"GGUF tensor '{weightName}' has no data pointer.");
            IntPtr start = p + (nint)(firstRow * rowBytes);
            var lin = new DirectLinear(ctx, start, IntPtr.Zero, (int)info.Type, ne0, rowCount, rowCount * rowBytes, null, false);
            if (ctx.IsMlx && MlxQuantizedOps.CanPreloadDirectFloatType((int)info.Type))
                MlxQuantizedOps.PreloadDirectFloatWeight(ctx.MlxAllocator, start, start, (int)info.Type, ne0, rowCount, rowCount * rowBytes);
            else if (!ctx.IsCuda && (ctx.IsMlx || !lin.UseManagedQuantOnCpu))
            {
                var ids = new int[rowCount];
                for (int i = 0; i < rowCount; i++) ids[i] = checked((int)(firstRow + i));
                var w = new float[ne0 * rowCount];
                DirectOps.DequantRows(gguf, weightName, ids, w);
                lin.SetCpuWeight(w, ne0, rowCount);
            }
            return ctx.Track(lin);
        }

        /// <summary>Load the 5D patch-embedding conv as the flattened
        /// [ic*kh*kw, oc] matmul weight (same bytes; see WanDiT.PatchW5D).</summary>
        public static DirectLinear FromGgufPatch(DirectContext ctx, GgufFile gguf, string weightName, string biasName)
        {
            var info = gguf.Tensors[weightName];
            long ne0 = 1;
            for (int i = 0; i + 1 < info.Shape.Length; i++) ne0 *= (long)info.Shape[i];
            long ne1 = (long)info.Shape[info.Shape.Length - 1];
            long blockSize = GgufFile.GetBlockSize(info.Type);
            if (blockSize > 1 && ne0 % blockSize != 0)
                throw new NotSupportedException(
                    $"patch_embedding.weight is {info.Type} with block size {blockSize}, which does not divide " +
                    $"the {ne0}-wide patch rows; requantize the DiT with an F16/F32 patch embedding.");
            long bytes = gguf.GetTensorByteCount(info);
            if (!gguf.TryGetTensorDataPointer(info, out IntPtr p))
                throw new System.IO.InvalidDataException($"GGUF tensor '{weightName}' has no data pointer.");
            Tensor bias = null;
            if (biasName != null && gguf.Tensors.ContainsKey(biasName))
                bias = ctx.Own(ctx.FromFloats(DirectOps.DequantTensor(gguf, biasName), (long)gguf.Tensors[biasName].NumElements));
            var lin = new DirectLinear(ctx, p, IntPtr.Zero, (int)info.Type, ne0, ne1, bytes, bias, false);
            if (ctx.IsMlx && MlxQuantizedOps.CanPreloadDirectFloatType((int)info.Type))
                MlxQuantizedOps.PreloadDirectFloatWeight(ctx.MlxAllocator, p, p, (int)info.Type, ne0, ne1, bytes);
            else if (!ctx.IsCuda && (ctx.IsMlx || !lin.UseManagedQuantOnCpu))
                lin.SetCpuWeight(DirectOps.DequantTensor(gguf, weightName), ne0, ne1);
            return ctx.Track(lin);
        }

        /// <summary>Load from a managed F32 array (weight [ne1, ne0] row-major, e.g. VAE safetensors).</summary>
        public static DirectLinear FromFloats(DirectContext ctx, float[] weight, long ne0, long ne1, float[] bias,
            bool mlxDeviceOnly = false)
        {
            Tensor biasT = bias != null ? ctx.Own(ctx.FromFloats(bias, bias.LongLength)) : null;
            if (ctx.IsCuda)
            {
                // Stable pinned host copy: the resident cache keys on this address.
                long bytes = weight.LongLength * sizeof(float);
                IntPtr host = Marshal.AllocHGlobal((IntPtr)bytes);
                Marshal.Copy(weight, 0, host, weight.Length);
                return ctx.Track(new DirectLinear(ctx, host, host, 0 /*F32*/, ne0, ne1, bytes, biasT, false));
            }
            var lin = new DirectLinear(ctx, IntPtr.Zero, IntPtr.Zero, 0, ne0, ne1, 0, biasT, false);
            // H3's dense video decoder otherwise retains ~9.7 GB of duplicate
            // host weights beside the identical F32 MLX arrays. Opt in only for
            // these immutable weights; all existing callers retain their path.
            lin._wCpu = ctx.Own(ctx.IsMlx && mlxDeviceOnly
                ? MlxMiniMaxH3Ops.CreateFloat32Weight(ctx.MlxAllocator, weight, ne1, ne0)
                : ctx.FromFloats(weight, ne1, ne0));
            lin._wCpuT = lin._wCpu.Transpose();
            return ctx.Track(lin);
        }

        /// <summary>H3-specific CUDA dense projection: the owner keeps the mapped
        /// weight alive, and CUDA converts bounded F16/BF16 slices for F32 GEMM.
        /// Activations and results retain F32 storage and arithmetic.</summary>
        internal static DirectLinear FromMappedH3Dense(DirectContext ctx, IntPtr host, int type,
            long ne0, long ne1, long bytes, float[] bias)
        {
            if (!ctx.IsCuda || host == IntPtr.Zero || type is not (0 or 1 or 30) || ne0 <= 0 || ne1 <= 0 ||
                bytes != checked(ne0*ne1*(type == 0 ? sizeof(float) : sizeof(ushort))))
                throw new ArgumentException("H3 mapped CUDA dense weights require a valid F16/BF16/F32 matrix.");
            Tensor biasT=bias != null ? ctx.Own(ctx.FromFloats(bias,bias.LongLength)) : null;
            return ctx.Track(new DirectLinear(ctx,host,IntPtr.Zero,type,ne0,ne1,bytes,biasT,false,h3MappedDense:true));
        }

        internal void SetCpuWeight(float[] w, long ne0, long ne1)
        {
            _wCpu = _ctx.Own(_ctx.FromFloats(w, ne1, ne0));
            _wCpuT = _wCpu.Transpose();
        }

        /// <summary>x [rows, InDim] contiguous F32 -> new tensor [rows, OutDim].</summary>
        public Tensor Forward(Tensor x)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var r = _ctx.NewF32(x.Sizes[0], OutDim);
            try
            {
                if (_ctx.IsCuda)
                {
                    Tensor xin = x;
                    Tensor scaled = null;
                    if (_prescale)
                    {
                        scaled = _ctx.NewF32(x.Sizes[0], InDim);
                        Ops.Mul(scaled, x, 1.0f / MmScale);
                        xin = scaled;
                    }
                    bool completed = _ltxProjection?.TryPrefillLinear(r, xin, _host, _type, InDim, OutDim, _bytes) == true ||
                        _ltxProjection?.TryLinear(r, xin, _host, _type, InDim, OutDim, _bytes) == true || (_h3PreserveFloat32
                        ? CudaMiniMaxH3Ops.TryLinear(r, xin, _host, _type, InDim, OutDim, _bytes)
                        : _h3MappedDense
                            ? CudaMiniMaxH3Ops.TryMappedDense(r, xin, _host, _type, InDim, OutDim, _bytes)
                            : CudaQuantizedOps.TryAddmmQuantizedToFloat32(r, xin, _host, _host, _type, InDim, OutDim, _bytes));
                    if (!completed)
                        throw new InvalidOperationException(
                            $"CUDA quantized linear failed (type {_type}, [{x.Sizes[0]},{InDim}]x[{InDim},{OutDim}]).");
                    if (scaled != null)
                    {
                        Ops.Mul(r, r, MmScale);
                        scaled.Dispose();
                    }
                }
                else if (_ctx.IsMlx)
                {
                    if (_wCpu != null)
                        Ops.Addmm(r, 0f, r, 1f, x, _wCpuT);
                    else if (!MlxQuantizedOps.TryAddmmDirectFloat32(r, x, _host, _host, _type,
                        InDim, OutDim, _bytes))
                        throw new InvalidOperationException(
                            $"MLX compressed F32 linear failed (type {_type}, [{x.Sizes[0]},{InDim}]x[{InDim},{OutDim}]).");
                }
                else if (_wCpu == null)
                {
                    unsafe
                    {
                        float* xp = (float*)CpuNativeHelpers.GetBufferStart(x);
                        float* rp = (float*)CpuNativeHelpers.GetBufferStart(r);
                        ManagedQuantizedOps.AddmmQuantizedToFloat32(
                            _type, _host, InDim, OutDim,
                            xp, checked((int)InDim), checked((int)x.Sizes[0]),
                            rp, checked((int)OutDim));
                    }
                }
                else
                {
                    DirectOps.CpuGemmABt(x, _wCpu, r, 1f, 0f);
                }
                if (_bias != null) DirectOps.AddBiasRows(_ctx, r, _bias);
                return r;
            }
            catch { r.Dispose(); throw; }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _residentWeight?.Release();
            _wCpuT?.Dispose();
            if (_ownedHost != IntPtr.Zero)
                Marshal.FreeHGlobal(_ownedHost);
            // _wCpu/_bias are context-owned tensors.
        }

        internal void ReleaseDeviceResidency()
        {
            if (!_disposed) _residentWeight?.EvictIfUnshared();
        }

        internal void RetainDeviceWeight()
        {
            _residentWeight ??= ResidentWeightLease.Acquire(_ctx, _host);
        }
    }

    /// <summary>Backend-dispatched primitives shared by the direct networks.</summary>
    internal static class DirectOps
    {
        /// <summary>
        /// Row-parallel loop on the persistent CPU pool rather than the ThreadPool.
        /// These loops are the whole managed cost of the direct path and a single
        /// call is often only tens of microseconds of work per core, which is the
        /// regime where a Parallel.For fork/join costs more than the work it
        /// distributes - see CpuWorkerPool for the measurements. Rows are handed
        /// out in contiguous chunks, a few per worker, so the tail still balances.
        /// </summary>
        internal static void RowsParallel(long rows, Action<long> body)
        {
            if (rows <= 0) return;
            int threads = CpuWorkerPool.Shared.ThreadCount;
            if (rows == 1 || threads <= 1)
            {
                for (long i = 0; i < rows; i++) body(i);
                return;
            }
            long chunk = Math.Max(1, (rows + threads * 4L - 1) / (threads * 4L));
            int blocks = checked((int)((rows + chunk - 1) / chunk));
            CpuWorkerPool.Shared.For(blocks, b =>
            {
                long start = b * chunk, end = Math.Min(rows, start + chunk);
                for (long i = start; i < end; i++) body(i);
            });
        }

        // ---- weight loading helpers ------------------------------------------------

        /// <summary>Dequantize a whole GGUF tensor to a managed F32 array
        /// (parallel over row chunks — a UMT5-XXL is ~22 GB of F32 and the
        /// single-call path decodes it on one core).</summary>
        public static unsafe float[] DequantTensor(GgufFile gguf, string name)
        {
            var info = gguf.Tensors[name];
            var dst = new float[info.NumElements];
            long ne0 = (long)info.Shape[0];
            long rows = info.NumElements / Math.Max(1, ne0);
            long rowBytes = NativeDequant.RowSize((int)info.Type, ne0);
            if (rows > 1 && rowBytes > 0 && gguf.TryGetTensorDataPointer(info, out IntPtr basePtr))
            {
                int chunks = Math.Min((int)rows, Environment.ProcessorCount * 4);
                long rowsPerChunk = (rows + chunks - 1) / chunks;
                Parallel.For(0, chunks, c =>
                {
                    long r0 = c * rowsPerChunk;
                    long rn = Math.Min(rowsPerChunk, rows - r0);
                    if (rn <= 0) return;
                    NativeDequant.DequantizeToFloat32((int)info.Type,
                        (IntPtr)((byte*)basePtr + r0 * rowBytes), dst, checked((int)(r0 * ne0)), rn * ne0);
                });
                return dst;
            }
            long srcBytes = gguf.GetTensorByteCount(info);
            IntPtr tmp = Marshal.AllocHGlobal((IntPtr)srcBytes);
            try
            {
                gguf.ReadTensorDataToNative(info, tmp, srcBytes);
                NativeDequant.DequantizeToFloat32((int)info.Type, tmp, dst, 0, info.NumElements);
            }
            finally { Marshal.FreeHGlobal(tmp); }
            return dst;
        }

        /// <summary>Dequantize selected rows of a row-major GGUF matrix [rows, ne0]
        /// (token embedding lookup without materializing the full table).</summary>
        public static unsafe void DequantRows(GgufFile gguf, string name, int[] rowIds, float[] dst)
        {
            var info = gguf.Tensors[name];
            long ne0 = (long)info.Shape[0];
            long rowBytes = NativeDequant.RowSize((int)info.Type, ne0);
            long srcBytes = gguf.GetTensorByteCount(info);
            if (!gguf.TryGetTensorDataPointer(info, out IntPtr basePtr))
                throw new System.IO.InvalidDataException($"GGUF tensor '{name}' has no data pointer.");
            if (rowBytes * (long)info.Shape[1] > srcBytes)
                throw new System.IO.InvalidDataException($"GGUF tensor '{name}' row size mismatch.");
            Parallel.For(0, rowIds.Length, i =>
            {
                IntPtr row = (IntPtr)((byte*)basePtr + rowBytes * rowIds[i]);
                NativeDequant.DequantizeToFloat32((int)info.Type, row, dst, (int)(i * ne0), ne0);
            });
        }

        // ---- CPU GEMM --------------------------------------------------------------
        // Both orientations the direct paths need (the VAE im2col convs, attention)
        // run on the packed register-tiled CpuSgemm, which takes the operands'
        // strides as they are (narrowed/head views need no copy). C must have
        // contiguous rows.

        /// <summary>C[m,n] = beta*C + alpha * A[m,k] x B[n,k]^T (dot form; both row-major;
        /// C may be a contiguous-row view — row stride taken from its Strides).</summary>
        public static unsafe void CpuGemmABt(Tensor a, Tensor b, Tensor c, float alpha, float beta)
        {
            if (a.Storage is MlxStorage)
            {
                using var transposed = b.Transpose();
                Ops.Addmm(c, beta, c, alpha, a, transposed);
                return;
            }
            RequireContiguousRows(c);
            CpuSgemm.Gemm(checked((int)a.Sizes[0]), checked((int)b.Sizes[0]), checked((int)a.Sizes[1]), alpha,
                (float*)CpuNativeHelpers.GetBufferStart(a), a.Strides[0], a.Strides[1],
                (float*)CpuNativeHelpers.GetBufferStart(b), b.Strides[1], b.Strides[0],
                beta, (float*)CpuNativeHelpers.GetBufferStart(c), c.Strides[0]);
        }

        /// <summary>C[m,n] = beta*C + alpha * A[m,k] x B[k,n] (saxpy form; both row-major).</summary>
        public static unsafe void CpuGemmAB(Tensor a, Tensor b, Tensor c, float alpha, float beta)
        {
            if (a.Storage is MlxStorage)
            {
                Ops.Addmm(c, beta, c, alpha, a, b);
                return;
            }
            RequireContiguousRows(c);
            CpuSgemm.Gemm(checked((int)a.Sizes[0]), checked((int)b.Sizes[1]), checked((int)a.Sizes[1]), alpha,
                (float*)CpuNativeHelpers.GetBufferStart(a), a.Strides[0], a.Strides[1],
                (float*)CpuNativeHelpers.GetBufferStart(b), b.Strides[0], b.Strides[1],
                beta, (float*)CpuNativeHelpers.GetBufferStart(c), c.Strides[0]);
        }

        private static void RequireContiguousRows(Tensor c)
        {
            if (c.Strides[1] != 1)
                throw new ArgumentException("The GEMM output must have contiguous rows (unit column stride).", nameof(c));
        }

        // ---- rowwise vector ops ----------------------------------------------------

        /// <summary>t[r, c] += bias[c] for every row.</summary>
        public static void AddBiasRows(DirectContext ctx, Tensor t, Tensor bias)
        {
            if (ctx.IsMlx)
            {
                using var row = bias.View(1, t.Sizes[1]);
                Ops.Add(t, t, row);
                return;
            }
            if (ctx.IsCuda)
            {
                if (!CudaFusedOps.TryAddBiasRows(t, bias))
                    throw new InvalidOperationException("CUDA AddBiasRows failed (non-contiguous tensor?).");
                return;
            }
            CpuAddBiasRows(t, bias);
        }

        private static unsafe void CpuAddBiasRows(Tensor t, Tensor bias)
        {
            long rows = t.Sizes[0], cols = t.Sizes[1];
            float* p = (float*)CpuNativeHelpers.GetBufferStart(t);
            float* b = (float*)CpuNativeHelpers.GetBufferStart(bias);
            RowsParallel(rows, r => RowAddSimd(p + r * cols, p + r * cols, b, cols));
        }

        /// <summary>AdaLN modulation y[r, c] = x[r, c] * (1 + scale[c]) + shift[c] over
        /// rows [rowFrom, rowFrom + rowCount). x and y may alias.</summary>
        public static void ModulateRows(DirectContext ctx, Tensor y, Tensor x, Tensor shift, Tensor scale,
                                        long rowFrom, long rowCount)
        {
            long cols = x.Sizes[1];
            if (ctx.IsMlx)
            {
                using var input = x.Narrow(0, rowFrom, rowCount);
                using var target = y.Narrow(0, rowFrom, rowCount);
                using var scalePlusOne = Ops.Add(null, scale, 1f);
                using var scaleRow = scalePlusOne.View(1, cols);
                using var shiftRow = shift.View(1, cols);
                using var scaled = Ops.Mul(null, input, scaleRow);
                Ops.Add(target, scaled, shiftRow);
                return;
            }
            if (ctx.IsCuda)
            {
                if (!CudaWanOps.TryModulateRows(y, x, shift, scale, rowFrom, rowCount))
                    throw new InvalidOperationException("CUDA ModulateRows kernel unavailable (stale PTX?).");
                return;
            }
            ModulateRowsOnHost(y, x, shift, scale, rowFrom, rowCount);
        }

        /// <summary><see cref="ModulateRows"/> on host tensors (contiguous F32 rows; shift and scale [cols]).</summary>
        internal static unsafe void ModulateRowsOnHost(Tensor y, Tensor x, Tensor shift, Tensor scale, long rowFrom, long rowCount)
        {
            long cols = x.Sizes[1];
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            float* py = (float*)CpuNativeHelpers.GetBufferStart(y);
            float* ps = (float*)CpuNativeHelpers.GetBufferStart(shift);
            float* pc = (float*)CpuNativeHelpers.GetBufferStart(scale);
            RowsParallel(rowCount, rr =>
            {
                long r = rowFrom + rr;
                ModulateRowSimd(py + r * cols, px + r * cols, pc, ps, cols);
            });
        }

        /// <summary>Gated residual x[r, c] += v[r, c] * gate[c] over rows [rowFrom, rowFrom+rowCount).</summary>
        public static void GateAddRows(DirectContext ctx, Tensor x, Tensor v, Tensor gate,
                                       long rowFrom, long rowCount)
        {
            long cols = x.Sizes[1];
            if (ctx.IsMlx)
            {
                using var target = x.Narrow(0, rowFrom, rowCount);
                using var update = v.Narrow(0, rowFrom, rowCount);
                using var gateRow = gate.View(1, cols);
                using var gated = Ops.Mul(null, update, gateRow);
                Ops.Add(target, target, gated);
                return;
            }
            if (ctx.IsCuda)
            {
                if (!CudaWanOps.TryGateAddRows(x, v, gate, rowFrom, rowCount))
                    throw new InvalidOperationException("CUDA GateAddRows kernel unavailable (stale PTX?).");
                return;
            }
            GateAddRowsOnHost(x, v, gate, rowFrom, rowCount);
        }

        /// <summary><see cref="GateAddRows"/> on host tensors (contiguous F32 rows; gate [cols]).</summary>
        internal static void GateAddRowsOnHost(Tensor x, Tensor v, Tensor gate, long rowFrom, long rowCount)
            => GateAddRowsOnHost(x, x, v, gate, rowFrom, rowCount);

        /// <summary>y[r, c] = x[r, c] + v[r, c] * gate[c] over rows [rowFrom, rowFrom + rowCount) on host tensors (contiguous F32
        /// rows; gate [cols]): <see cref="GateAddRows"/>' arithmetic into another tensor (y and x may alias).</summary>
        internal static unsafe void GateAddRowsOnHost(Tensor y, Tensor x, Tensor v, Tensor gate, long rowFrom, long rowCount)
        {
            long cols = x.Sizes[1];
            float* py = (float*)CpuNativeHelpers.GetBufferStart(y);
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            float* pv = (float*)CpuNativeHelpers.GetBufferStart(v);
            float* pg = (float*)CpuNativeHelpers.GetBufferStart(gate);
            RowsParallel(rowCount, rr =>
            {
                long r = rowFrom + rr;
                GateAddRowSimd(py + r * cols, px + r * cols, pv + r * cols, pg, cols);
            });
        }

        /// <summary>Scale rows by a per-column vector: x[r, c] *= gain[c] (RMS/LayerNorm gains
        /// are folded into the norm ops; this covers the odd cases).</summary>
        public static void MulColsRows(DirectContext ctx, Tensor x, Tensor gain)
        {
            if (ctx.IsMlx)
            {
                using var row = gain.View(1, x.Sizes[1]);
                Ops.Mul(x, x, row);
                return;
            }
            if (ctx.IsCuda)
            {
                if (!CudaWanOps.TryScaleCols(x, gain))
                    throw new InvalidOperationException("CUDA ScaleCols kernel unavailable (stale PTX?).");
                return;
            }
            unsafe
            {
                long rows = x.Sizes[0], cols = x.Sizes[1];
                float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
                float* pg = (float*)CpuNativeHelpers.GetBufferStart(gain);
                RowsParallel(rows, r => RowMulSimd(px + r * cols, pg, cols));
            }
        }

        // Vector forms of the row ops above: the same operations in the same order as the scalar
        // formula and no FMA contraction, so bit-identical to it.

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        private static unsafe void RowAddSimd(float* dst, float* x, float* b, long cols)
        {
            int w = System.Numerics.Vector<float>.Count;
            long c = 0;
            for (; c + w <= cols; c += w)
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(dst + c,
                    System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(x + c) +
                    System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(b + c));
            for (; c < cols; c++) dst[c] = x[c] + b[c];
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        private static unsafe void RowMulSimd(float* x, float* g, long cols)
        {
            int w = System.Numerics.Vector<float>.Count;
            long c = 0;
            for (; c + w <= cols; c += w)
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(x + c,
                    System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(x + c) *
                    System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(g + c));
            for (; c < cols; c++) x[c] *= g[c];
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ModulateRowSimd(float* y, float* x, float* scale, float* shift, long cols)
        {
            int w = System.Numerics.Vector<float>.Count;
            var one = System.Numerics.Vector<float>.One;
            long c = 0;
            for (; c + w <= cols; c += w)
            {
                var vx = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(x + c);
                var vs = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(scale + c);
                var vh = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(shift + c);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(y + c, vx * (one + vs) + vh);
            }
            for (; c < cols; c++) y[c] = x[c] * (1f + scale[c]) + shift[c];
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        private static unsafe void GateAddRowSimd(float* y, float* x, float* v, float* gate, long cols)
        {
            int w = System.Numerics.Vector<float>.Count;
            long c = 0;
            for (; c + w <= cols; c += w)
            {
                var vx = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(x + c);
                var vv = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(v + c);
                var vg = System.Runtime.CompilerServices.Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(gate + c);
                System.Runtime.CompilerServices.Unsafe.WriteUnaligned(y + c, vx + vv * vg);
            }
            for (; c < cols; c++) y[c] = x[c] + v[c] * gate[c];
        }

        // ---- norms -------------------------------------------------------------------

        /// <summary>LayerNorm over the last dim; gamma/beta optional (non-affine when null).</summary>
        public static Tensor LayerNorm(DirectContext ctx, Tensor x, Tensor gamma, Tensor beta, float eps)
        {
            var r = ctx.NewF32(x.Sizes[0], x.Sizes[1]);
            Ops.LayerNorm(r, x, gamma ?? ctx.Ones(x.Sizes[1]), beta, eps);
            return r;
        }

        /// <summary>RMS norm over the last dim with gain.</summary>
        public static Tensor RmsNorm(DirectContext ctx, Tensor x, Tensor gain, float eps)
        {
            var r = ctx.NewF32(x.Sizes[0], x.Sizes[1]);
            Ops.RMSNorm(r, x, gain, null, eps);
            return r;
        }

        // ---- RoPE ---------------------------------------------------------------------

        /// <summary>
        /// In-place interleaved-pair RoPE over x [seq, heads*headDim] with
        /// pair-duplicated cos/sin tables [seq, headDim] (WanRope layout:
        /// cos[t, 2p] == cos[t, 2p+1]).
        /// </summary>
        public static void RopeInterleaved(DirectContext ctx, Tensor x, Tensor cos, Tensor sin,
                                           int heads, int headDim)
        {
            int seq = checked((int)x.Sizes[0]);
            if (ctx.IsMlx)
            {
                if (!MlxDirectOps.TryRopeInterleaved(x, cos, sin, heads, seq, headDim))
                    throw new InvalidOperationException("MLX F32 interleaved RoPE failed.");
                return;
            }
            if (ctx.IsCuda)
            {
                if (!CudaWanOps.TryRopeInterleaved(x, cos, sin, heads, seq, headDim))
                    throw new InvalidOperationException("CUDA RoPE kernel unavailable (stale PTX?).");
                return;
            }
            unsafe
            {
                float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
                float* pc = (float*)CpuNativeHelpers.GetBufferStart(cos);
                float* ps = (float*)CpuNativeHelpers.GetBufferStart(sin);
                int half = headDim / 2;
                RowsParallel((long)seq * heads, th =>
                {
                    long t = th / heads;
                    float* row = px + th * headDim;
                    float* ct = pc + t * headDim;
                    float* st = ps + t * headDim;
                    for (int p = 0; p < half; p++)
                    {
                        float c = ct[2 * p], s = st[2 * p];
                        float e = row[2 * p], o = row[2 * p + 1];
                        row[2 * p] = e * c - o * s;
                        row[2 * p + 1] = o * c + e * s;
                    }
                });
            }
        }

        // ---- attention ------------------------------------------------------------------

        /// <summary>
        /// Multi-head attention over token-major projections: q [sq, heads*hd],
        /// k/v [sk, heads*hd], optional additive bias [heads, sq, sk] (T5 relative
        /// position bias; scale applied to scores before the bias). Returns
        /// [sq, heads*hd]. CUDA uses the streaming ts_wan_attn kernel; the
        /// fallback runs chunked per-head GEMM+softmax through Ops.
        /// </summary>
        public static Tensor Attention(DirectContext ctx, Tensor q, Tensor k, Tensor v,
                                       int heads, int headDim, float scale, Tensor bias = null)
        {
            int sq = checked((int)q.Sizes[0]);
            int sk = checked((int)k.Sizes[0]);
            var outT = ctx.NewF32(sq, (long)heads * headDim);
            if (ctx.IsMlx)
            {
                try
                {
                    using var q4 = q.View(1, sq, heads, headDim);
                    using var k4 = k.View(1, sk, heads, headDim);
                    using var v4 = v.View(1, sk, heads, headDim);
                    using var result4 = outT.View(1, sq, heads, headDim);
                    using var mask = bias?.View(1, heads, sq, sk);
                    Ops.ScaledDotProductAttention(result4, q4, k4, v4, mask, scale);
                    return outT;
                }
                catch { outT.Dispose(); throw; }
            }

            bool trace = Environment.GetEnvironmentVariable("TS_WAN_DIRECT_TIMING") == "1";
            bool fused = Environment.GetEnvironmentVariable("TS_WAN_DIRECT_FUSED_ATTN") != "0";
            // The streaming F32 kernel is launch-lean but scalar (~1.3 TFLOPS); the
            // chunked cuBLAS GEMM+softmax path reaches ~4.7 TFLOPS but pays copies
            // and scratch. Route big unbiased attentions (long-sequence self-attn:
            // 832x480x33f is 14k tokens) to the GEMM path — measured crossover is a
            // total workload of roughly 100 GFLOP.
            long attnFlop = 4L * sq * sk * heads * headDim;
            bool preferGemm = ctx.IsCuda && bias == null && attnFlop > 100_000_000_000L;
            if (!preferGemm && fused &&
                ctx.IsCuda && CudaWanOps.TryAttention(outT, q, k, v, bias, heads, sq, sk, headDim, scale))
                return outT;
            if (!preferGemm && fused && ctx.IsCuda && bias == null && sq == sk &&
                CudaFusedOps.TryVisionAttention(outT, q, k, v, heads, sq, headDim, scale))
            {
                if (trace) Console.WriteLine($"  [wan-attn] vision-kernel path sq={sq} sk={sk}");
                return outT;
            }

            if (trace) Console.WriteLine($"  [wan-attn] GENERIC path sq={sq} sk={sk} cuda={ctx.IsCuda}");
            GenericAttention(ctx, outT, q, k, v, bias, heads, sq, sk, headDim, scale, halfOperands: preferGemm);
            return outT;
        }

        // Chunked per-head GEMM+softmax path (cuBLAS on CUDA, the parallel SIMD
        // kernels on CPU). Bigger chunks on CUDA: fewer launches, and the pool
        // absorbs the score scratch. halfOperands (the long-sequence route, chosen
        // for speed over the exact streaming kernel) runs both GEMMs on IEEE-half
        // operands by name (HalfOperandGemm), as they ran under the old cuBLAS
        // handle; scores, softmax and output stay F32, like ggml's F16 flash
        // attention. Exact F32 costs 1.8-2.0x on these GEMMs (Wan2.1-1.3B at
        // 832x480x33f: +2.2 s per forward) and TF32 still 2x.
        private static void GenericAttention(DirectContext ctx, Tensor outT, Tensor q, Tensor k, Tensor v,
                                             Tensor bias, int heads, int sq, int sk, int headDim, float scale,
                                             bool halfOperands = false)
        {
            if (!ctx.IsCuda && TryCpuBlockedAttention(outT, q, k, v, bias, heads, sq, sk, headDim, scale))
                return;

            long budget = ctx.IsCuda ? 64L << 20 : 16L << 20;   // score elements per chunk
            int qChunk = (int)Math.Max(1, Math.Min(sq, budget / Math.Max(1, sk)));

            for (int h = 0; h < heads; h++)
            {
                using var kNv = k.Narrow(1, (long)h * headDim, headDim);
                using var kh = Ops.NewContiguous(kNv);                 // [sk, hd]
                using var khT = kh.Transpose();                        // [hd, sk] column-major view (BLAS trans)
                using var vNv = v.Narrow(1, (long)h * headDim, headDim);
                using var vh = Ops.NewContiguous(vNv);                 // [sk, hd]
                for (int q0 = 0; q0 < sq; q0 += qChunk)
                {
                    int qn = Math.Min(qChunk, sq - q0);
                    using var qRows = q.Narrow(0, q0, qn);
                    using var qNv = qRows.Narrow(1, (long)h * headDim, headDim);
                    using var qh = Ops.NewContiguous(qNv);             // [qn, hd]
                    using var scores = ctx.NewF32(qn, sk);
                    if (ctx.IsCuda && halfOperands) HalfOperandGemm.Addmm(scores, 0f, scores, scale, qh, khT);
                    else if (ctx.IsCuda) Ops.Addmm(scores, 0f, scores, scale, qh, khT);
                    else CpuGemmABt(qh, kh, scores, scale, 0f);
                    if (bias != null)
                    {
                        using var bSel = bias.Select(0, h);            // [sq, sk] contiguous
                        using var bh = bSel.Narrow(0, q0, qn);
                        Ops.Add(scores, scores, bh);
                    }
                    Ops.Softmax(scores, scores);
                    using var oh = ctx.NewF32(qn, headDim);
                    if (ctx.IsCuda && halfOperands) HalfOperandGemm.Addmm(oh, 0f, oh, 1f, scores, vh);
                    else if (ctx.IsCuda) Ops.Addmm(oh, 0f, oh, 1f, scores, vh);
                    else CpuGemmAB(scores, vh, oh, 1f, 0f);
                    using var oRows = outT.Narrow(0, q0, qn);
                    using var dst = oRows.Narrow(1, (long)h * headDim, headDim);
                    Ops.Copy(dst, oh);
                }
            }
        }

        /// <summary>Fewest query rows per block of <see cref="TryCpuBlockedAttention"/> (tuning;
        /// CpuFloatBench 'attnlong' sweeps it).</summary>
        internal static int CpuAttentionMinQBlock { get; set; } = 128;

        /// <summary>
        /// CPU attention on the packed SGEMM without per-head copies: Q/K/V head slices are
        /// passed as strided views (row stride heads*hd), and the work is cut into independent
        /// (head, query-block) blocks - S = scale*Q_blk K^T into a per-block scratch sized to stay
        /// in L2, + bias, row softmax, then O_blk = P V written straight into the output columns.
        /// With enough blocks for the pool each runs single-threaded (no barriers between the
        /// GEMMs and the softmax); otherwise the blocks run in turn on the parallel GEMM. Same
        /// arithmetic as the chunked path, which still serves the layouts this one declines.
        /// </summary>
        internal static unsafe bool TryCpuBlockedAttention(Tensor outT, Tensor q, Tensor k, Tensor v, Tensor bias,
                                                          int heads, int sq, int sk, int hd, float scale)
        {
            if (q.DimensionCount != 2 || k.DimensionCount != 2 || v.DimensionCount != 2 ||
                q.Strides[1] != 1 || k.Strides[1] != 1 || v.Strides[1] != 1 || outT.Strides[1] != 1 ||
                (bias != null && (bias.DimensionCount != 3 || bias.Strides[2] != 1)))
            {
                return false;
            }

            // Raw pointers below, so extents are checked here: shapes the chunked path's
            // Narrow/Select would reject (or read past, for V rows != sk) fail fast instead of
            // reading out of bounds. A broadcast bias (stride 0 over heads or rows) is fine.
            long width = (long)heads * hd;
            if (q.Sizes[1] < width || k.Sizes[1] < width || v.Sizes[1] < width || v.Sizes[0] != sk)
            {
                throw new ArgumentException(
                    $"Attention shapes q[{q.Sizes[0]},{q.Sizes[1]}] k[{k.Sizes[0]},{k.Sizes[1]}] v[{v.Sizes[0]},{v.Sizes[1]}] " +
                    $"do not hold {heads} heads x {hd} with {sk} keys.");
            }
            if (bias != null && (bias.Sizes[0] < heads || bias.Sizes[1] < sq || bias.Sizes[2] != sk))
            {
                throw new ArgumentException(
                    $"Attention bias [{bias.Sizes[0]},{bias.Sizes[1]},{bias.Sizes[2]}] does not cover [{heads},{sq},{sk}].");
            }
            if (sq == 0 || heads == 0 || hd == 0) return true;   // nothing to write

            float* qp = (float*)CpuNativeHelpers.GetBufferStart(q);
            float* kp = (float*)CpuNativeHelpers.GetBufferStart(k);
            float* vp = (float*)CpuNativeHelpers.GetBufferStart(v);
            float* op = (float*)CpuNativeHelpers.GetBufferStart(outT);
            float* bp = bias != null ? (float*)CpuNativeHelpers.GetBufferStart(bias) : null;
            long qs = q.Strides[0], ks = k.Strides[0], vs = v.Strides[0], os = outT.Strides[0];
            long bh = bias != null ? bias.Strides[0] : 0, br = bias != null ? bias.Strides[1] : 0;

            // Query rows per block: a <= 1 MB score block (L2) when the keys allow it, but at least
            // 128 rows. Every block re-packs its head's K^T and V panels (one packed float per qn
            // FMAs): against a 16-row floor, 128 measured ~6% faster at 4K keys (24 heads) and
            // 11-22% at 8K-16K, although the score block then outgrows L2 (8 MB at 16K keys).
            int qBlock = (int)Math.Clamp((256L * 1024) / Math.Max(1, sk), Math.Clamp(CpuAttentionMinQBlock, 8, 256), 256);
            qBlock = Math.Min(sq, (qBlock + 7) / 8 * 8);
            int qBlocks = (sq + qBlock - 1) / qBlock;
            int items = heads * qBlocks;
            bool blockParallel = items >= CpuParallel.DegreeOfParallelism;

            void Block(int item, bool parallelInside)
            {
                int h = item / qBlocks;
                int q0 = (item - h * qBlocks) * qBlock;
                int qn = Math.Min(qBlock, sq - q0);
                float[] rented = System.Buffers.ArrayPool<float>.Shared.Rent(qn * sk);
                try
                {
                    fixed (float* s = rented)
                    {
                        CpuSgemm.Gemm(qn, sk, hd, scale,
                            qp + q0 * qs + (long)h * hd, qs, 1,
                            kp + (long)h * hd, 1, ks,
                            0f, s, sk, parallelInside);
                        if (bp != null)
                        {
                            for (int i = 0; i < qn; i++)
                                CpuKernels.Binary(CpuKernels.BinaryOp.Add, s + (long)i * sk, s + (long)i * sk, bp + h * bh + (q0 + i) * br, sk);
                        }
                        if (parallelInside) CpuKernels.SoftmaxRows(s, s, qn, sk);
                        else for (int i = 0; i < qn; i++) CpuKernels.SoftmaxRow(s + (long)i * sk, s + (long)i * sk, sk);
                        CpuSgemm.Gemm(qn, hd, sk, 1f,
                            s, sk, 1,
                            vp + (long)h * hd, vs, 1,
                            0f, op + q0 * os + (long)h * hd, os, parallelInside);
                    }
                }
                finally
                {
                    System.Buffers.ArrayPool<float>.Shared.Return(rented);
                }
            }

            if (blockParallel)
            {
                CpuParallel.For(items, item => Block(item, false));
            }
            else
            {
                for (int item = 0; item < items; item++) Block(item, true);
            }
            return true;
        }

        // ---- misc ----------------------------------------------------------------------

        /// <summary>Zero-fill a contiguous F32 tensor on its own device.</summary>
        public static void Zero(DirectContext ctx, Tensor t)
        {
            if (ctx.IsCuda && CudaWanOps.TryZero(t)) return;
            Ops.Fill(t, 0f);
        }

        /// <summary>GELU (tanh approximation, matches ggml_gelu).</summary>
        public static Tensor Gelu(DirectContext ctx, Tensor x)
        {
            var r = ctx.NewF32(x.Sizes);
            Ops.GELU(r, x);
            return r;
        }

        /// <summary>
        /// Exact-form GELU, x * (1 + erf(x / sqrt 2)) / 2 (torch.nn.GELU's default), with erf from
        /// the Abramowitz-Stegun 7.1.26 polynomial (|error| below 1.5e-7) that H3's kernels use.
        /// Models trained on it need it: on Whisper's encoder the tanh form moves single
        /// positions to cosine 0.4 of the reference.
        /// </summary>
        public static unsafe Tensor GeluErf(DirectContext ctx, Tensor x)
        {
            var r = ctx.NewF32(x.Sizes);
            try
            {
                if (ctx.IsCuda && CudaMiniMaxH3Ops.TryTransform(r, x, null, null, 5, 0)) return r;
                if (ctx.IsMlx && MlxMiniMaxH3Ops.TryTransform(r, x, null, null, 5, 0)) return r;
                if (r.Storage is CpuStorage && x.Storage is CpuStorage && x.IsContiguous())
                {
                    long n = r.ElementCount();
                    float* xp = (float*)CpuNativeHelpers.GetBufferStart(x);
                    float* rp = (float*)CpuNativeHelpers.GetBufferStart(r);
                    const long chunk = 1 << 16;
                    Parallel.For(0, (n + chunk - 1) / chunk, c =>
                    {
                        long end = Math.Min(n, (c + 1) * chunk);
                        for (long i = c * chunk; i < end; i++) rp[i] = GeluErfScalar(xp[i]);
                    });
                    return r;
                }
                throw new NotSupportedException("this backend has no exact-GELU path for the direct primitives.");
            }
            catch
            {
                r.Dispose();
                throw;
            }
        }

        internal static float GeluErfScalar(float value)
        {
            float scaled = value * 0.7071067811865475f;
            float absolute = MathF.Abs(scaled);
            float t = 1f / (1f + 0.3275911f * absolute);
            float polynomial = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
            float erf = 1f - polynomial * MathF.Exp(-absolute * absolute);
            if (scaled < 0) erf = -erf;
            return value * 0.5f * (1f + erf);
        }

        public static Tensor Silu(DirectContext ctx, Tensor x)
        {
            var r = ctx.NewF32(x.Sizes);
            Ops.SiLU(r, x);
            return r;
        }

        /// <summary>Download a [rows, cols] F32 tensor into a managed array.</summary>
        public static float[] ToArray(Tensor t)
        {
            long n = 1;
            foreach (long s in t.Sizes) n *= s;
            return t.GetElementsAsFloat(checked((int)n));
        }
    }
}
