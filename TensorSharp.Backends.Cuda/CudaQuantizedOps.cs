using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public static class CudaQuantizedOps
    {
        private sealed class DeviceWeight : IDisposable
        {
            public IntPtr DevicePtr;
            public long RawBytes;
            public int GgmlType;
            public long Ne0;
            public long Ne1;
            public int DeviceId;

            public void Dispose()
            {
                if (DevicePtr != IntPtr.Zero)
                {
                    CudaDriverApi.cuMemFree(DevicePtr);
                    DevicePtr = IntPtr.Zero;
                }
            }
        }

        private static readonly object Sync = new object();
        private static readonly Dictionary<CacheKey, DeviceWeight> Cache = new Dictionary<CacheKey, DeviceWeight>();

        // Per-device reusable scratch for q8_1-quantized activations (the dp4a Q8_0
        // matmul path). Grown on demand; reused across calls (one stream per allocator
        // ⇒ each matmul's read completes before the next quantize overwrites it).
        private static readonly Dictionary<int, (IntPtr ptr, long bytes)> Q81Scratch = new();
        private static IntPtr EnsureQ81Scratch(CudaAllocator allocator, long bytes)
        {
            lock (Sync)
            {
                if (Q81Scratch.TryGetValue(allocator.DeviceId, out var s) && s.bytes >= bytes)
                    return s.ptr;
                allocator.Context.MakeCurrent();
                if (s.ptr != IntPtr.Zero)
                    CudaDriverApi.cuMemFree(s.ptr);
                long alloc = Math.Max(bytes, 64 * 1024);
                CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)alloc)).ThrowOnError();
                Q81Scratch[allocator.DeviceId] = (ptr, alloc);
                return ptr;
            }
        }

        // Row-batched quantized matmul (weight-reuse across small row counts).
        // On by default; set TS_CUDA_QMM_BATCHED=0 to force the legacy per-row
        // kernels (A/B benchmarking).
        internal static readonly bool BatchedMatmulEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QMM_BATCHED"), "0", StringComparison.Ordinal);

        // Q4_0 (the dominant dense quant) uses the int8 dp4a matmul for decode AND
        // verify by default (~memory-bound, matches ggml's mul_mat_vec_q); set
        // TS_CUDA_Q40_DP4A=0 to revert to the FP32 dequant kernels. Settable so the
        // exact-reference tests can pin the FP32 path while a separate test checks
        // the dp4a path against the (looser) int8 tolerance.
        public static bool Q40Dp4aEnabled { get; set; } =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q40_DP4A"), "0", StringComparison.Ordinal);

        public static bool SupportsQuantizedType(int ggmlType)
        {
            return ggmlType == 2 ||  // Q4_0
                ggmlType == 3 ||     // Q4_1
                ggmlType == 6 ||     // Q5_0
                ggmlType == 7 ||     // Q5_1
                ggmlType == 8 ||     // Q8_0
                ggmlType == 9 ||     // Q8_1
                ggmlType == 10 ||    // Q2_K
                ggmlType == 11 ||    // Q3_K
                ggmlType == 12 ||    // Q4_K
                ggmlType == 13 ||    // Q5_K
                ggmlType == 14 ||    // Q6_K
                ggmlType == 16 ||    // IQ2_XXS
                ggmlType == 18 ||    // IQ3_XXS
                ggmlType == 21 ||    // IQ3_S
                ggmlType == 22;      // IQ2_S
        }

        public static void PreloadQuantizedWeight(
            CudaAllocator allocator,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (allocator == null)
                throw new ArgumentNullException(nameof(allocator));
            EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
        }

        public static void ReleaseQuantizedWeight(CudaAllocator allocator, IntPtr cacheKey)
        {
            if (allocator == null || cacheKey == IntPtr.Zero)
                return;

            var key = new CacheKey(allocator.DeviceId, cacheKey);
            lock (Sync)
            {
                if (Cache.TryGetValue(key, out DeviceWeight entry))
                {
                    allocator.Context.MakeCurrent();
                    entry.Dispose();
                    Cache.Remove(key);
                }
            }
        }

        public static void ClearDeviceCache(CudaAllocator allocator)
        {
            if (allocator == null)
                return;

            lock (Sync)
            {
                allocator.Context.MakeCurrent();
                var remove = new List<CacheKey>();
                foreach (var kv in Cache)
                {
                    if (kv.Key.DeviceId == allocator.DeviceId)
                    {
                        kv.Value.Dispose();
                        remove.Add(kv.Key);
                    }
                }

                foreach (CacheKey key in remove)
                    Cache.Remove(key);
            }
        }

        // q8Kernel override for the multi-row Q8_0 path (lets a test drive MMA and dp4a
        // in one process): 0 = auto (env flags), 1 = dp4a, 2 = tensor-core MMA, 3 = scalar.
        public static bool TryAddmmQuantizedToFloat32(
            Tensor result,
            Tensor input,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            int q8Kernel = 0)
        {
            if (!SupportsQuantizedType(ggmlType))
                return false;

            if (!CudaKernelOps.TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !CudaKernelOps.TryGetContiguousFloat(input, out CudaStorage inputStorage, out IntPtr inputPtr, out int inputCount) ||
                input.DimensionCount != 2 ||
                result.DimensionCount != 2 ||
                input.Sizes[1] != ne0 ||
                result.Sizes[0] != input.Sizes[0] ||
                result.Sizes[1] != ne1 ||
                inputCount != input.Sizes[0] * ne0 ||
                resultCount != result.Sizes[0] * ne1)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            CudaKernels kernels = allocator.Kernels;
            if (kernels == null)
                return false;

            DeviceWeight weight = EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            inputStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int rows = checked((int)input.Sizes[0]);
            // Small multi-row batches (speculative MTP verify windows, short
            // prefill chunks) run the per-row quant kernels once per output row --
            // the whole weight matrix is re-read (and re-dequantized) per row, so a
            // B-row matmul costs ~B x a single decode and speculative verification
            // can never amortize. The generic scalar batched kernel tiles
            // TS_QMM_ROW_TILE rows per block, streaming each weight ONCE and reusing
            // it across the tile (measured 2.7-2.9x at B>=4 for the k-quant
            // ffn_down / LM-head weights).
            //   * Q4_0 (the dominant dense quant -- the 12B QAT model is entirely
            //     Q4_0) has its OWN row-tiled kernel below that keeps the efficient
            //     per-row nibble unpacking AND amortizes the weight read + dequant
            //     across the tile; the generic scalar batched path is ~2x slower for
            //     it (warp-per-column under-fills vs the block-per-4-columns design,
            //     and qvalue_at re-branches per element).
            //   * IQ2_XXS / Q8_0 keep their own multi-row paths (the q8_1 dp4a/MMA
            //     GEMM and the IQ2_XXS q8_1 kernel) which already read the weight
            //     once per tile and beat the scalar batched kernel for those types.
            if (BatchedMatmulEnabled && rows >= 2 && rows <= CudaKernels.QuantMatmulBatchMaxRows
                && ggmlType != 2 && ggmlType != 8 && ggmlType != 16)
            {
                kernels.LaunchQuantMatmulBatchedF32(
                    weight.DevicePtr, inputPtr, resultPtr,
                    ggmlType, inDim, outDim, rows, allocator.Stream.Handle);
                resultStorage.MarkDeviceModified();
                return true;
            }
            if (ggmlType == 2)
            {
                // Q4_0 int8 dp4a: quantize the activation rows to q8_1 ONCE, then the
                // block-tile dp4a GEMM. This is the fast path for BOTH single-token
                // decode (rows==1 — the dominant cost, and where the scalar FP32
                // dequant kernel read the LM head's Q4_0 weights at ~26 GB/s) and the
                // verify window. dp4a does 4 int8 MACs/instruction so it is ~memory-
                // bound, matching ggml's mul_mat_vec_q. inDim is always a multiple of
                // 32 for these models; non-conforming shapes fall through to FP32.
                if (Q40Dp4aEnabled && (inDim & 31) == 0)
                {
                    long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                    IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                    kernels.LaunchQuantizeQ81Rows(inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle);
                    kernels.LaunchQuantMatmulQ40Dp4a(
                        weight.DevicePtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                    resultStorage.MarkDeviceModified();
                    return true;
                }
                // FP32 fallback: row-tiled batched kernel for the verify window
                // (decode each weight nibble ONCE and reuse it across the tile's rows)
                // and the per-row kernel for single-row decode.
                if (BatchedMatmulEnabled && rows >= 2 && rows <= CudaKernels.QuantMatmulBatchMaxRows)
                {
                    kernels.LaunchQuantMatmulQ40BatchedF32(
                        weight.DevicePtr, inputPtr, resultPtr,
                        inDim, outDim, rows, allocator.Stream.Handle);
                    resultStorage.MarkDeviceModified();
                    return true;
                }
                kernels.LaunchQuantMatmulQ40F32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else if (ggmlType == 8 && rows >= 2 && (inDim & 31) == 0
                     && q8Kernel != 3
                     && (q8Kernel == 1 || q8Kernel == 2
                         || (q8Kernel == 0 && (CudaKernels.Q8MmaEnabled || CudaKernels.Q8Dp4aEnabled))))
            {
                // Multi-row Q8_0 fast path: quantize the activation rows to q8_1 ONCE
                // into a reused scratch (single-stream per allocator makes reuse safe),
                // then either the tensor-core MMA GEMM or the block-tile dp4a GEMM. The
                // big FFN matmuls are compute-bound for the verify window.
                bool useMma = q8Kernel == 2 || (q8Kernel == 0 && CudaKernels.Q8MmaEnabled);
                long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle);
                if (useMma)
                    kernels.LaunchQuantMatmulQ80Mma(
                        weight.DevicePtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                else
                    kernels.LaunchQuantMatmulQ80Dp4a(
                        weight.DevicePtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
            }
            else if (ggmlType == 8)
            {
                kernels.LaunchQuantMatmulQ80F32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else if (ggmlType == 16 && (inDim & 255) == 0 && ((inDim / 32) * 36) <= 48 * 1024)
            {
                kernels.LaunchQuantMatmulIq2XxsQ81F32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else
            {
                kernels.LaunchQuantMatmulF32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    ggmlType,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }

            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryGetRowsQuantizedToFloat32(
            Tensor result,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            Tensor indices)
        {
            if (!SupportsQuantizedType(ggmlType))
                return false;

            if (!CudaKernelOps.TryGetContiguousRows(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols) ||
                !CudaKernelOps.TryGetContiguous(indices, out CudaStorage indicesStorage, out IntPtr indicesPtr, out long indexCount) ||
                cols != ne0 ||
                indexCount != rows ||
                result.ElementType != DType.Float32 ||
                (indices.ElementType != DType.Int32 && indices.ElementType != DType.Float32))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            CudaKernels kernels = allocator.Kernels;
            if (kernels == null)
                return false;

            DeviceWeight weight = EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            indicesStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchQuantGetRowsF32(
                weight.DevicePtr,
                indicesPtr,
                resultPtr,
                ggmlType,
                checked((int)ne0),
                rows,
                indices.ElementType == DType.Int32 ? 1 : 0,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        private static DeviceWeight EnsureWeight(
            CudaAllocator allocator,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (cacheKey == IntPtr.Zero)
                cacheKey = hostData;
            if (cacheKey == IntPtr.Zero)
                throw new ArgumentException("CUDA quantized weight cache key cannot be zero.", nameof(cacheKey));

            var key = new CacheKey(allocator.DeviceId, cacheKey);
            lock (Sync)
            {
                if (Cache.TryGetValue(key, out DeviceWeight existing))
                    return existing;

                if (hostData == IntPtr.Zero)
                    throw new InvalidOperationException("Quantized weight is not preloaded on this CUDA device and no host data was provided.");

                allocator.Context.MakeCurrent();
                CudaDriverApi.cuMemAlloc(out IntPtr devicePtr, new UIntPtr((ulong)rawBytes)).ThrowOnError();
                try
                {
                    CudaDriverApi.cuMemcpyHtoD(devicePtr, hostData, new UIntPtr((ulong)rawBytes)).ThrowOnError();
                    var entry = new DeviceWeight
                    {
                        DevicePtr = devicePtr,
                        RawBytes = rawBytes,
                        GgmlType = ggmlType,
                        Ne0 = ne0,
                        Ne1 = ne1,
                        DeviceId = allocator.DeviceId,
                    };
                    Cache.Add(key, entry);
                    return entry;
                }
                catch
                {
                    CudaDriverApi.cuMemFree(devicePtr);
                    throw;
                }
            }
        }

        private readonly struct CacheKey : IEquatable<CacheKey>
        {
            public CacheKey(int deviceId, IntPtr key)
            {
                DeviceId = deviceId;
                Key = key;
            }

            public int DeviceId { get; }
            public IntPtr Key { get; }

            public bool Equals(CacheKey other)
            {
                return DeviceId == other.DeviceId && Key == other.Key;
            }

            public override bool Equals(object obj)
            {
                return obj is CacheKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                return HashCode.Combine(DeviceId, Key);
            }
        }
    }
}
