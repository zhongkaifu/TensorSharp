using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed class CudaAllocator : IAllocator, IDeviceMemoryInfo, IHostRangePinner, IDisposable
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
            long maxCachedBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MAX_MB", 512L) * 1024L * 1024L;
            long maxCachedBlockBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MAX_BLOCK_MB", 256L) * 1024L * 1024L;
            // Budget for the global large-block cache (prefill-sized activations;
            // see CudaDeviceMemoryPool), resolved once the context exists.
            // Keep at least this much dedicated VRAM free: once the pool would push
            // free VRAM below it, returned blocks go back to the driver instead of
            // into the cache. Prevents the pool from being the thing that tips a
            // nearly-full device (a big model on a small card) into WDDM shared
            // memory. 0 disables the valve (unbounded caching).
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

            // Default: a quarter of the device's memory, at least 1 GB. A graph executor's prompt step allocates each
            // op's activations as it runs (a 3.6K-token step of Qwen3.8-27B holds 0.5 GB gate|up products); past the
            // budget every one went back to the driver through a synchronizing cuMemFree (2,863 a prompt, 0.55 s of a
            // 2.3 s prefill on an RTX PRO 4500). An allocation that finds the device full drains the cache first.
            context.MakeCurrent();
            long deviceBytes = CudaDriverApi.cuMemGetInfo(out UIntPtr _, out UIntPtr totalBytes) == 0 ? (long)totalBytes.ToUInt64() : 0;
            long largeCachedBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_LARGE_MB", Math.Max(1024L, deviceBytes / 4 / (1024L * 1024L)))
                * 1024L * 1024L;

            Context = context;
            _stream = stream;
            Blas = blas;
            Kernels = kernels;
            DeviceId = deviceId;

            pool = new CudaDeviceMemoryPool(
                maxCachedBytes,
                maxCachedBlockBytes,
                enabled: true,
                backingAllocate: AllocateDeviceMemory,
                backingFree: FreeDeviceMemory,
                largeCachedBytesCap: largeCachedBytes,
                queryFreeBytes: QueryFreeDeviceBytes,
                minFreeReserveBytes: minFreeReserveBytes);
        }

        public BlasEnum BlasEnum => BlasEnum.CUDA;

        public int DeviceId { get; }

        internal CudaContext Context { get; }

        /// <summary>The stream work is queued on: the allocator's own, or a second stream a graph executor moved this thread's
        /// work to for a concurrent region (<see cref="EnterSideStream"/>).</summary>
        internal CudaStream Stream => _side != null && _sideThread == Environment.CurrentManagedThreadId ? _side : _stream;

        /// <summary>The allocator's own stream, whatever a concurrent region moved.</summary>
        internal CudaStream MainStream => _stream;

        private readonly CudaStream _stream;
        private CudaStream _side;
        private int _sideThread;

        // Blocks returned while work on two streams may still read them (from a concurrent region's fork to its join): the
        // pool hands a returned block out at once, which on one stream is safe (later work runs after), across two is not.
        private readonly object _deferSync = new();
        private readonly List<(IntPtr Ptr, long Bytes)> _deferred = new();
        private int _deferDepth;

        /// <summary>This thread's work goes to <paramref name="side"/> until <see cref="LeaveSideStream"/>, and every block
        /// returned until the matching <see cref="EndDeferredReturns"/> is held back.</summary>
        internal void EnterSideStream(CudaStream side)
        {
            lock (_deferSync) _deferDepth++;
            _sideThread = Environment.CurrentManagedThreadId;
            _side = side;
        }

        internal void LeaveSideStream() => _side = null;

        /// <summary>The streams joined (later work on the allocator's stream waits for the side stream's): the blocks held
        /// back since the matching <see cref="EnterSideStream"/> go back to the pool once no other window is open.</summary>
        internal void EndDeferredReturns()
        {
            List<(IntPtr Ptr, long Bytes)> release = null;
            lock (_deferSync)
            {
                if (_deferDepth > 0 && --_deferDepth == 0 && _deferred.Count > 0)
                {
                    release = new List<(IntPtr, long)>(_deferred);
                    _deferred.Clear();
                }
            }
            if (release != null)
                foreach ((IntPtr ptr, long bytes) in release) ReturnNow(ptr, bytes);
        }

        internal CudaCublasHandle Blas { get; }

        internal CudaKernels Kernels { get; }

        private CudaFp4Kernels fp4Kernels;
        private CudaSsmPrefillKernels ssmPrefillKernels;
        private CudaFlashPrefillKernels flashPrefillKernels;
        private int fp4State, ssmPrefillState, flashPrefillState;   // 0: not loaded yet, 1: loaded, 2: unavailable

        /// <summary>The FP4 tensor-core products for this device (tensorsharp_fp4_kernels), loaded on first use; null
        /// where the image has no tensor-core code for this GPU or is missing.</summary>
        internal CudaFp4Kernels Fp4Kernels => LoadOnce(ref fp4Kernels, ref fp4State, CudaFp4Kernels.TryCreate);

        /// <summary>The recurrent-layer prefill kernels for this device (tensorsharp_ssm_prefill), loaded on first use;
        /// null where the module is missing or stale.</summary>
        internal CudaSsmPrefillKernels SsmPrefillKernels => LoadOnce(ref ssmPrefillKernels, ref ssmPrefillState, CudaSsmPrefillKernels.TryCreate);

        /// <summary>The flash prefill attention kernels for this device (tensorsharp_flash_prefill), loaded on first use;
        /// null where the module is missing or stale.</summary>
        internal CudaFlashPrefillKernels FlashPrefillKernels => LoadOnce(ref flashPrefillKernels, ref flashPrefillState, CudaFlashPrefillKernels.TryCreate);

        /// <summary>Raised before a recorded step is launched (or recorded): host threads its kernels wait on spin up.</summary>
        internal event Action RecordedStepLaunching;

        internal void OnRecordedStepLaunching() => RecordedStepLaunching?.Invoke();

        private readonly System.Collections.Generic.List<IDisposable> ownedResources = new();

        /// <summary>A resource that lives as long as this allocator's context (disposed with it, before the context).</summary>
        internal void Own(IDisposable resource)
        {
            lock (ownedResources) ownedResources.Add(resource);
        }

        private delegate bool ModuleFactory<T>(out T module, out string reason);

        private T LoadOnce<T>(ref T module, ref int state, ModuleFactory<T> create) where T : class
        {
            if (Volatile.Read(ref state) != 0) return module;
            lock (this)
            {
                if (state != 0) return module;
                Context.MakeCurrent();
                bool loaded = create(out module, out _);
                Volatile.Write(ref state, loaded ? 1 : 2);
                return module;
            }
        }

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
            lock (_deferSync)
                if (_deferDepth > 0)
                {
                    _deferred.Add((ptr, allocationBytes));
                    return;
                }
            ReturnNow(ptr, allocationBytes);
        }

        private void ReturnNow(IntPtr ptr, long allocationBytes)
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
            int rc = CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)allocationBytes));
            if (rc == 2 /* CUDA_ERROR_OUT_OF_MEMORY */)
            {
                // Storages whose last reference is an unreachable-but-unfinalized
                // tensor (view chains) only release on collection; reclaim them and
                // the pool's cached blocks before giving up.
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                Stream.Synchronize();
                pool.DrainAndFree();
                rc = CudaDriverApi.cuMemAlloc(out ptr, new UIntPtr((ulong)allocationBytes));
            }
            rc.ThrowOnError();
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

        /// <summary>Free device memory, with the blocks the pool caches for reuse counted as free, and the total.</summary>
        public bool TryGetDeviceMemory(out long freeBytes, out long totalBytes)
        {
            try
            {
                (freeBytes, totalBytes) = GetMemoryInfo();
                freeBytes += pool.GetStats().CachedBytes;
                return totalBytes > 0;
            }
            catch (CudaException)
            {
                freeBytes = totalBytes = 0;
                return false;
            }
        }

        public bool SharesHostMemory => false;

        public IDisposable PinHostRanges(IReadOnlyList<(IntPtr Pointer, long Bytes)> ranges, long allowance, System.Threading.CancellationToken cancel)
        {
            if (allowance > 0) CudaHostPins.Allow(allowance);
            Context.MakeCurrent();
            var pins = new CudaHostPins();
            foreach ((IntPtr pointer, long bytes) in ranges)
            {
                if (cancel.IsCancellationRequested) break;
                pins.Pin(pointer, bytes);
            }
            return pins;
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

        /// <summary>Free every pooled (returned) block back to the driver. For
        /// pipeline stage boundaries where the next stage's working set has a
        /// different shape profile (e.g. a diffusion denoise handing over to a
        /// video VAE): live allocations are untouched, but the caller must ensure
        /// no concurrent Rent/Return on this allocator.</summary>
        public void TrimPool()
        {
            ThrowIfDisposed();
            Context.MakeCurrent();
            Stream.Synchronize();
            pool.DrainAndFree();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposed, 1) != 0)
                return;

            Context.MakeCurrent();
            Stream.Synchronize();
            lock (ownedResources)
            {
                foreach (IDisposable resource in ownedResources) resource.Dispose();
                ownedResources.Clear();
            }
            CudaQuantizedOps.ReleaseScratch(this);
            CudaQuantizedOps.ReleaseArena(this);
            pool.DrainAndFree();
            CudaLtx25PromptOps.ReleaseForAllocator(this);
            Kernels?.Dispose();
            fp4Kernels?.Dispose();
            ssmPrefillKernels?.Dispose();
            flashPrefillKernels?.Dispose();
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
