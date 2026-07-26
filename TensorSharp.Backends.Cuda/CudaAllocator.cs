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
            // Budget for the global large-block cache (prefill-sized activations;
            // see CudaDeviceMemoryPool). Big enough that a 2048-token prefill
            // chunk's transients all pool, which also keeps CUDA-graph captures
            // of the prefill loop allocation-free.
            long largeCachedBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_LARGE_MB", 1024L) * 1024L * 1024L;
            // Keep at least this much dedicated VRAM free: once the pool would push
            // free VRAM below it, returned blocks go back to the driver instead of
            // into the cache. Prevents the pool from being the thing that tips a
            // nearly-full device (a big model on a small card) into WDDM shared
            // memory. 0 disables the valve (unbounded caching, the old behaviour).
            long minFreeReserveBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MIN_FREE_MB", 256L) * 1024L * 1024L;

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
                backingFree: FreeDeviceMemory,
                largeCachedBytesCap: largeCachedBytes,
                queryFreeBytes: QueryFreeDeviceBytes,
                minFreeReserveBytes: minFreeReserveBytes);
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
            IntPtr ptr = pool.Rent(requestedBytes, out allocationBytes);
            CudaGraphCapture.OnRent(this, ptr, allocationBytes);
            return ptr;
        }

        internal void ReturnDeviceMemory(IntPtr ptr, long allocationBytes)
        {
            if (CudaGraphCapture.IsCapturing(this))
            {
                // While a graph capture is active the captured kernels reference
                // this block: pooling keeps it reachable for the capture owner
                // (it is tracked and stolen at capture end); a cuMemFree would
                // leave the graph pointing at freed memory, so those returns are
                // quarantined by the capture context instead.
                bool pooled = pool.TryReturnToPool(ptr, allocationBytes);
                if (!CudaGraphCapture.InterceptReturn(this, ptr, allocationBytes, wouldPool: pooled) && !pooled)
                    pool.Return(ptr, allocationBytes);
                return;
            }
            pool.Return(ptr, allocationBytes);
        }

        /// <summary>Remove a specific free block from the pool so a cached CUDA
        /// graph can own it (see <see cref="CudaPrefillGraphCache"/>).</summary>
        internal bool TryStealPooledBlock(IntPtr ptr, long allocationBytes)
        {
            return pool.TrySteal(ptr, allocationBytes);
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

        /// <summary>Free dedicated device memory in bytes, for the pool's
        /// VRAM-pressure valve. Sampled at most every ~10 ms, so the MakeCurrent +
        /// cuMemGetInfo cost stays off the hot return path.</summary>
        private long QueryFreeDeviceBytes()
        {
            Context.MakeCurrent();
            if (CudaDriverApi.cuMemGetInfo(out UIntPtr free, out UIntPtr _) != 0)
                return long.MaxValue; // on query failure, don't throttle caching
            return (long)free.ToUInt64();
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

        /// <summary>Free / total device memory in bytes (driver view, i.e. dedicated
        /// VRAM only — WDDM shared-memory spillover shows up as this staying pinned
        /// near zero while performance collapses).</summary>
        public (long freeBytes, long totalBytes) GetMemoryInfo()
        {
            Context.MakeCurrent();
            CudaDriverApi.cuMemGetInfo(out UIntPtr free, out UIntPtr total).ThrowOnError();
            return ((long)free.ToUInt64(), (long)total.ToUInt64());
        }

        internal static readonly bool VramLogEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_LOG_VRAM"), "1", StringComparison.Ordinal);

        /// <summary>Diagnostic (TS_CUDA_LOG_VRAM=1): prints free/used dedicated VRAM
        /// plus the pool's currently-cached idle bytes at <paramref name="label"/>.</summary>
        public void LogVram(string label)
        {
            if (!VramLogEnabled)
                return;
            (long free, long total) = GetMemoryInfo();
            long cached = pool.GetStats().CachedBytes;
            Console.WriteLine(
                $"[vram] {label}: free={free / (1024 * 1024)} MB used={(total - free) / (1024 * 1024)} MB " +
                $"pool_cached={cached / (1024 * 1024)} MB");
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
            CudaQuantizedOps.ReleaseScratch(this);
            CudaQuantizedOps.ReleaseArena(this);
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
