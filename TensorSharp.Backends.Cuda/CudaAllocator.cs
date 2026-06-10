using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed class CudaAllocator : IAllocator, IDisposable
    {
        private int disposed;

        // Striped, size-classed device-memory cache. Replaces the original single
        // global `poolSync` monitor + Dictionary<long, Stack<IntPtr>>, which became a
        // contention hot spot when high-concurrency serving churned many small
        // transient tensors. See CudaDeviceMemoryPool for the design.
        private readonly CudaDeviceMemoryPool pool;

        public CudaAllocator(int deviceId = 0)
        {
            CudaBackend.Register();
            bool poolEnabled = !string.Equals(Environment.GetEnvironmentVariable("TENSORSHARP_CUDA_POOL"), "0", StringComparison.Ordinal);
            long maxCachedBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MAX_MB", 512L) * 1024L * 1024L;
            long maxCachedBlockBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MAX_BLOCK_MB", 256L) * 1024L * 1024L;

            CudaContext context = null;
            CudaStream stream = null;
            CudaCublasHandle blas = null;
            CudaKernels kernels = null;

            try
            {
                context = CudaContext.Create(deviceId);
                context.MakeCurrent();
                stream = CudaStream.Create();
                blas = CudaCublasHandle.Create();
                blas.SetStream(stream.Handle);
                kernels = CudaKernels.TryCreate();
            }
            catch
            {
                kernels?.Dispose();
                blas?.Dispose();
                stream?.Dispose();
                context?.Dispose();
                throw;
            }

            Context = context;
            Stream = stream;
            Blas = blas;
            Kernels = kernels;
            DeviceId = deviceId;

            pool = new CudaDeviceMemoryPool(
                maxCachedBytes,
                maxCachedBlockBytes,
                poolEnabled,
                backingAllocate: AllocateDeviceMemory,
                backingFree: FreeDeviceMemory);
        }

        public BlasEnum BlasEnum => BlasEnum.CUDA;

        public int DeviceId { get; }

        internal CudaContext Context { get; }

        internal CudaStream Stream { get; }

        internal CudaCublasHandle Blas { get; }

        internal CudaKernels Kernels { get; }

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CudaStorage(this, elementType, elementCount);
        }

        internal IntPtr RentDeviceMemory(long requestedBytes, out long allocationBytes)
        {
            ThrowIfDisposed();
            return pool.Rent(requestedBytes, out allocationBytes);
        }

        internal void ReturnDeviceMemory(IntPtr ptr, long allocationBytes)
        {
            pool.Return(ptr, allocationBytes);
        }

        private IntPtr AllocateDeviceMemory(long allocationBytes)
        {
            Context.MakeCurrent();
            CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)allocationBytes)).ThrowOnError();
            return ptr;
        }

        private void FreeDeviceMemory(IntPtr ptr)
        {
            Context.MakeCurrent();
            CudaDriverApi.cuMemFree(ptr);
        }

        public float GetAllocatedMemoryRatio()
        {
            Context.MakeCurrent();
            CudaDriverApi.cuMemGetInfo(out UIntPtr free, out UIntPtr total).ThrowOnError();
            ulong totalBytes = total.ToUInt64();
            if (totalBytes == 0)
                return 0.0f;

            ulong freeBytes = free.ToUInt64();
            return (float)(1.0 - (double)freeBytes / totalBytes);
        }

        /// <summary>
        /// Snapshot of allocator pool counters for diagnostics / load testing.
        /// </summary>
        public CudaAllocatorStats GetStats() => pool.GetStats();

        public void Synchronize()
        {
            Context.MakeCurrent();
            Stream.Synchronize();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposed, 1) != 0)
                return;

            Context.MakeCurrent();
            Stream.Synchronize();
            pool.DrainAndFree();
            Kernels?.Dispose();
            Blas.Dispose();
            Stream.Dispose();
            Context.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref disposed) != 0)
                throw new ObjectDisposedException(nameof(CudaAllocator));
        }

        private static long ReadPoolLimit(string name, long defaultMb)
        {
            string value = Environment.GetEnvironmentVariable(name);
            if (long.TryParse(value, out long parsed) && parsed >= 0)
                return parsed;
            return defaultMb;
        }
    }

    /// <summary>
    /// Immutable snapshot of <see cref="CudaAllocator"/> pool counters.
    /// </summary>
    public readonly struct CudaAllocatorStats
    {
        internal CudaAllocatorStats(
            bool poolEnabled,
            long poolHitCount,
            long poolMissCount,
            long cuMemAllocCount,
            long cuMemFreeCount,
            long returnedToPoolCount,
            long cachedBytes,
            long peakCachedBytes,
            long maxCachedBytes,
            long maxCachedBlockBytes,
            int shardCount)
        {
            PoolEnabled = poolEnabled;
            PoolHitCount = poolHitCount;
            PoolMissCount = poolMissCount;
            CuMemAllocCount = cuMemAllocCount;
            CuMemFreeCount = cuMemFreeCount;
            ReturnedToPoolCount = returnedToPoolCount;
            CachedBytes = cachedBytes;
            PeakCachedBytes = peakCachedBytes;
            MaxCachedBytes = maxCachedBytes;
            MaxCachedBlockBytes = maxCachedBlockBytes;
            ShardCount = shardCount;
        }

        public bool PoolEnabled { get; }

        /// <summary>Rent calls served from the pool (no backing allocation).</summary>
        public long PoolHitCount { get; }

        /// <summary>Rent calls that fell through to a backing allocation.</summary>
        public long PoolMissCount { get; }

        /// <summary>Total cuMemAlloc driver calls issued (== <see cref="PoolMissCount"/>).</summary>
        public long CuMemAllocCount { get; }

        /// <summary>Total cuMemFree driver calls issued.</summary>
        public long CuMemFreeCount { get; }

        /// <summary>Return calls that parked a block back into the pool.</summary>
        public long ReturnedToPoolCount { get; }

        /// <summary>Bytes currently held in the pool (summed across shards).</summary>
        public long CachedBytes { get; }

        /// <summary>Sum of per-shard high-water marks (conservative upper bound, ≤ MaxCachedBytes).</summary>
        public long PeakCachedBytes { get; }

        public long MaxCachedBytes { get; }

        public long MaxCachedBlockBytes { get; }

        /// <summary>Number of independent pool shards.</summary>
        public int ShardCount { get; }

        public long TotalRentCount => PoolHitCount + PoolMissCount;

        public double PoolHitRatio => TotalRentCount == 0 ? 0.0 : (double)PoolHitCount / TotalRentCount;

        public override string ToString()
        {
            return $"CudaAllocatorStats(shards={ShardCount}, hits={PoolHitCount}, misses={PoolMissCount}, hitRatio={PoolHitRatio:P1}, " +
                   $"cuMemAlloc={CuMemAllocCount}, cuMemFree={CuMemFreeCount}, returnedToPool={ReturnedToPoolCount}, " +
                   $"cachedBytes={CachedBytes}, peakCachedBytes={PeakCachedBytes})";
        }
    }
}
