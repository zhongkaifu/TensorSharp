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

        // Row-batched quantized matmul (weight-reuse across small row counts).
        // On by default; set TS_CUDA_QMM_BATCHED=0 to force the legacy per-row
        // kernels (A/B benchmarking).
        internal static readonly bool BatchedMatmulEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QMM_BATCHED"), "0", StringComparison.Ordinal);

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

        public static bool TryAddmmQuantizedToFloat32(
            Tensor result,
            Tensor input,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
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
            // prefill chunks) run the slow SCALAR per-row k-quant/iq-quant kernel
            // (ts_quant_matmul_f32) once per output row -- the whole weight row is
            // re-read AND re-dequantized per row, so a B-row matmul costs ~B x a
            // single decode and speculative verification can never amortize. The
            // generic batched kernel tiles 4 rows per block, decoding each weight
            // ONCE and reusing it across the tile (measured 2.7-2.9x at B>=4 for
            // Q2_K / Q5_K-class weights -- ffn_down and the LM head here).
            //   * IQ2_XXS / Q4_0 / Q8_0 already have fast per-row dp4a/tiled
            //     kernels that amortize well across rows (IQ2_XXS measures a flat
            //     ~0.5 ms/row), so they keep those paths -- the scalar batched
            //     kernel would be SLOWER for them.
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
                kernels.LaunchQuantMatmulQ40F32(
                    weight.DevicePtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
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
