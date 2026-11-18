using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed class CudaAllocator : IAllocator, IDeviceMemoryInfo, IHostRangePinner, TensorSharp.Graph.IMappedHostAllocator, IDisposable
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
            // 2.3 s prefill on an RTX PRO 4500). An allocation that finds the device full gives back the cached blocks it
            // needs, the least recently used size first (DeviceMemoryReclaimLadder).
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
                minFreeReserveBytes: minFreeReserveBytes,
                mayRelievePressure: () => CudaGraphCapture.MayFreeCachedBlocksUnasked(this));

            ladder = new DeviceMemoryReclaimLadder(
                TryAllocateDevice,
                () => Stream.Synchronize(),
                bytes => pool.TrimLarge(bytes),
                pool.DrainSmall,
                () => RegisteredReclaimers,
                () => CudaGraphCapture.IsCapturing(this),
                CollectAndDrain,
                ReportOnce.FullDrain,
                deviceId,
                VramLogEnabled ? line => Console.WriteLine($"{line}; {GetStats()}") : null,
                minFreeReserveBytes);
        }

        // What an allocation does when the device is full (see DeviceMemoryReclaimLadder).
        private readonly DeviceMemoryReclaimLadder ladder;

        // Copy-on-write: the ladder reads it without a lock.
        private IDeviceMemoryReclaimer[] reclaimers = Array.Empty<IDeviceMemoryReclaimer>();
        private readonly object reclaimerSync = new();

        /// <summary>The reclaimers an allocation that finds the device full asks, in the order they are asked.</summary>
        internal IReadOnlyList<IDeviceMemoryReclaimer> RegisteredReclaimers => Volatile.Read(ref reclaimers);

        /// <summary>Ask <paramref name="reclaimer"/> for device memory when an allocation here finds the device full, after
        /// the cached blocks went back and before the last resort (a full GC and a drain of every cached block). Once per
        /// reclaimer; asked in the order registered.</summary>
        public void RegisterReclaimer(IDeviceMemoryReclaimer reclaimer)
        {
            ArgumentNullException.ThrowIfNull(reclaimer);
            lock (reclaimerSync)
            {
                if (Array.IndexOf(reclaimers, reclaimer) >= 0) return;
                var next = new IDeviceMemoryReclaimer[reclaimers.Length + 1];
                reclaimers.CopyTo(next, 0);
                next[^1] = reclaimer;
                Volatile.Write(ref reclaimers, next);
            }
        }

        /// <summary>Stop asking <paramref name="reclaimer"/> (it is detached or disposed).</summary>
        public void UnregisterReclaimer(IDeviceMemoryReclaimer reclaimer)
        {
            if (reclaimer == null) return;
            lock (reclaimerSync)
            {
                int at = Array.IndexOf(reclaimers, reclaimer);
                if (at < 0) return;
                var next = new IDeviceMemoryReclaimer[reclaimers.Length - 1];
                Array.Copy(reclaimers, 0, next, 0, at);
                Array.Copy(reclaimers, at + 1, next, at, reclaimers.Length - at - 1);
                Volatile.Write(ref reclaimers, next);
            }
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

        /// <summary>Of <paramref name="blocks"/>, the ones still held back (removed: a graph that captured work on them
        /// owns them now, and the region's join no longer returns them).</summary>
        internal List<(IntPtr, long)> TakeDeferredReturns(IReadOnlyList<(IntPtr Ptr, long Bytes)> blocks)
        {
            var taken = new List<(IntPtr, long)>();
            if (blocks.Count == 0) return taken;
            lock (_deferSync)
                foreach ((IntPtr ptr, long bytes) in blocks)
                {
                    int i = _deferred.IndexOf((ptr, bytes));
                    if (i < 0) continue;
                    _deferred.RemoveAt(i);
                    taken.Add((ptr, bytes));
                }
            return taken;
        }

        internal CudaCublasHandle Blas { get; }

        internal CudaKernels Kernels { get; }

        private CudaFp4Kernels fp4Kernels;
        private CudaSsmPrefillKernels ssmPrefillKernels;
        private CudaFlashPrefillKernels flashPrefillKernels;
        private int fp4State, ssmPrefillState, flashPrefillState;   // 0: not loaded yet, 1: loaded, 2: unavailable
        private string fp4Refusal, ssmPrefillRefusal, flashPrefillRefusal;

        /// <summary>The FP4 tensor-core products for this device (tensorsharp_fp4_kernels), loaded on first use; null
        /// where the image has no tensor-core code for this GPU or is missing. Its refusal is reported where an NVFP4
        /// weight needs it (<see cref="WarnNvfp4WithoutFp4Kernels"/>): every prompt step's add-norm asks for it too.</summary>
        internal CudaFp4Kernels Fp4Kernels => LoadOnce(ref fp4Kernels, ref fp4State, ref fp4Refusal, CudaFp4Kernels.TryCreate, null, null);

        /// <summary>An NVFP4 weight is being made resident: warn once, with the reason, when the FP4 kernels are refused
        /// here and its products therefore run without them.</summary>
        internal void WarnNvfp4WithoutFp4Kernels()
        {
            if (Fp4Kernels == null)
                CudaModuleRefusals.Process.Warn(CudaModuleRefusals.ModuleName(CudaFp4Kernels.PtxFileName), DeviceId, fp4Refusal,
                    CudaModuleRefusals.Fp4Consequence);
        }

        /// <summary>Rows an NVFP4 product takes on the q8 matvec (row-invariant) before the FP4 GEMMs: up to
        /// <see cref="CudaFp4Kernels.GemvMaxRows"/>, <see cref="CudaFp4Kernels.MultiSequenceGemvRows"/> while a step of
        /// several sequences runs (CudaGraphExecutor.StepSequences).</summary>
        internal int Nvfp4MatvecRows { get; set; } = CudaFp4Kernels.GemvMaxRows;

        /// <summary>The recurrent-layer prefill kernels for this device (tensorsharp_ssm_prefill), loaded on first use;
        /// null where the module is missing or stale.</summary>
        internal CudaSsmPrefillKernels SsmPrefillKernels => LoadOnce(ref ssmPrefillKernels, ref ssmPrefillState, ref ssmPrefillRefusal,
            CudaSsmPrefillKernels.TryCreate, CudaSsmPrefillKernels.PtxFileName, CudaModuleRefusals.SsmPrefillConsequence);

        /// <summary>The flash prefill attention kernels for this device (tensorsharp_flash_prefill), loaded on first use;
        /// null where the module is missing or stale.</summary>
        internal CudaFlashPrefillKernels FlashPrefillKernels => LoadOnce(ref flashPrefillKernels, ref flashPrefillState, ref flashPrefillRefusal,
            CudaFlashPrefillKernels.TryCreate, CudaFlashPrefillKernels.PtxFileName, CudaModuleRefusals.FlashPrefillConsequence);

        /// <summary>Raised before a recorded step is launched (or recorded): host threads its kernels wait on spin up.</summary>
        internal event Action RecordedStepLaunching;

        internal void OnRecordedStepLaunching() => RecordedStepLaunching?.Invoke();

        /// <summary>Raised once a recorded step is launched: the device runs it while the host has nothing to do until it reads
        /// the step's outputs.</summary>
        internal event Action RecordedStepLaunched;

        internal void OnRecordedStepLaunched() => RecordedStepLaunched?.Invoke();

        private readonly System.Collections.Generic.List<IDisposable> ownedResources = new();

        /// <summary>A resource that lives as long as this allocator's context (disposed with it, before the context).</summary>
        internal void Own(IDisposable resource)
        {
            lock (ownedResources) ownedResources.Add(resource);
        }

        /// <summary>A module loaded on first use (<see cref="CudaModuleRefusals.LoadOnce"/>); a refusal is warned about once,
        /// naming <paramref name="consequence"/>, unless that is null (the caller reports it where it matters).</summary>
        private T LoadOnce<T>(ref T module, ref int state, ref string refusal, CudaModuleFactory<T> create, string ptxFileName,
            string consequence) where T : class
            => Volatile.Read(ref state) != 0 ? module : LoadFirst(ref module, ref state, ref refusal, create, ptxFileName, consequence);

        private T LoadFirst<T>(ref T module, ref int state, ref string refusal, CudaModuleFactory<T> create, string ptxFileName,
            string consequence) where T : class
            => CudaModuleRefusals.LoadOnce(this, ref module, ref state, ref refusal, Context.MakeCurrent, create,
                consequence == null ? null : reason => CudaModuleRefusals.Process.Warn(CudaModuleRefusals.ModuleName(ptxFileName),
                    DeviceId, reason, consequence));

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

        /// <summary>Blocks a recorded graph worked in that were still rented when its capture ended, by the graph's owned
        /// list: the graph keeps writing them at every replay, so when their holder returns one it joins that list instead
        /// of the pool (see <see cref="Claim"/>).</summary>
        private readonly Dictionary<IntPtr, List<(IntPtr, long)>> _claims = new();

        /// <summary>The block at <paramref name="ptr"/> becomes <paramref name="owner"/>'s (a graph's owned blocks) when it
        /// is returned. A capture's work recycles blocks: one returned part way and rented again for something kept past the
        /// capture is not in the pool when it ends, and returned later it went to the pool while the graph still wrote it.</summary>
        internal void Claim(IntPtr ptr, List<(IntPtr, long)> owner)
        {
            lock (_claims) _claims[ptr] = owner;
        }

        /// <summary>Drop a claim (its graph is gone; the block, still rented, returns to the pool as usual).</summary>
        internal void Unclaim(IntPtr ptr)
        {
            lock (_claims) _claims.Remove(ptr);
        }

        internal void ReturnDeviceMemory(IntPtr ptr, long allocationBytes)
        {
            List<(IntPtr, long)> owner = null;
            lock (_claims)
                if (_claims.Count > 0 && _claims.Remove(ptr, out owner))
                    owner.Add((ptr, allocationBytes));
            if (owner != null) return;
            lock (_deferSync)
                if (_deferDepth > 0)
                {
                    _deferred.Add((ptr, allocationBytes));
                    CudaGraphCapture.OnDeferredReturn(this, ptr, allocationBytes);
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

        private const int OutOfMemoryStatus = 2; // CUDA_ERROR_OUT_OF_MEMORY

        /// <summary>The pool's backing allocation: a device block, climbing the reclaim ladder when the device is full.</summary>
        private IntPtr AllocateDeviceMemory(long allocationBytes)
        {
            IntPtr ptr = ladder.Allocate(allocationBytes);
            if (ptr == IntPtr.Zero) OutOfMemoryStatus.ThrowOnError();
            return ptr;
        }

        /// <summary>A device block outside the pool (a kernel's own scratch), made room for as a pool allocation is when the
        /// device is full. Not while a graph is captured here: the ladder's synchronize would abort the capture.</summary>
        internal IntPtr AllocateUnpooled(long bytes)
        {
            ThrowIfDisposed();
            return AllocateDeviceMemory(bytes);
        }

        /// <summary>One cuMemAlloc: the block, or zero when the device is out of memory (other failures throw).</summary>
        private IntPtr TryAllocateDevice(long allocationBytes)
        {
            Context.MakeCurrent();
            int rc = CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)allocationBytes));
            if (rc == OutOfMemoryStatus) return IntPtr.Zero;
            rc.ThrowOnError();
            return ptr;
        }

        /// <summary>The ladder's last resort, the old out-of-memory path: storages whose last reference is an
        /// unreachable-but-unfinalized tensor (view chains) only release on collection; collect them, then drain every
        /// cached block.</summary>
        private void CollectAndDrain()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            Context.MakeCurrent();
            Stream.Synchronize();
            pool.DrainAndFree();
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

        /// <summary>Free device memory, with the large blocks the pool caches counted as free (an allocation that needs the
        /// room gives them back, <see cref="ReclaimableBytes"/>), and the total. The shards' small cached blocks are not
        /// counted: they go back only once the large ones did not make room.</summary>
        public bool TryGetDeviceMemory(out long freeBytes, out long totalBytes)
        {
            try
            {
                (freeBytes, totalBytes) = GetMemoryInfo();
                freeBytes += pool.LargeCachedBytes;
                return totalBytes > 0;
            }
            catch (CudaException)
            {
                freeBytes = totalBytes = 0;
                return false;
            }
        }

        public bool SharesHostMemory => false;

        /// <summary>Page-locked host memory the device reads in place (cuMemHostAlloc PORTABLE | DEVICEMAP), or zero where the
        /// host refuses it: the host experts' packed copies. Any thread.</summary>
        public IntPtr AllocateMappedHost(long bytes, out IntPtr devicePointer)
        {
            devicePointer = IntPtr.Zero;
            if (bytes <= 0) return IntPtr.Zero;
            Context.MakeCurrent();
            if (CudaDriverApi.cuMemHostAlloc(out IntPtr host, new UIntPtr((ulong)bytes),
                    CudaDriverApi.CU_MEMHOSTALLOC_PORTABLE | CudaDriverApi.CU_MEMHOSTALLOC_DEVICEMAP) != 0)
                return IntPtr.Zero;
            if (CudaDriverApi.cuMemHostGetDevicePointer(out devicePointer, host, 0) != 0)
            {
                CudaDriverApi.cuMemFreeHost(host);
                devicePointer = IntPtr.Zero;
                return IntPtr.Zero;
            }
            return host;
        }

        /// <inheritdoc cref="TensorSharp.Graph.IMappedHostAllocator.FreeMappedHost"/>
        public void FreeMappedHost(IntPtr host)
        {
            if (host == IntPtr.Zero) return;
            Context.MakeCurrent();
            CudaDriverApi.cuMemFreeHost(host);
        }

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
            CudaAllocatorStats stats = GetStats();
            Console.WriteLine(
                $"[vram] {label}: free={free / (1024 * 1024)} MB used={(total - free) / (1024 * 1024)} MB " +
                $"pool_cached={stats.CachedBytes / (1024 * 1024)} MB (large={stats.LargeCachedBytes / (1024 * 1024)} MB) " +
                $"cuMemAlloc={stats.CuMemAllocCount} cuMemFree={stats.CuMemFreeCount} largeTrimFrees={stats.LargeTrimFrees} " +
                $"oomTrims={stats.OomTrims} oomReclaims={stats.OomReclaims} fullDrains={stats.FullDrains}");
        }

        /// <summary>
        /// Snapshot of allocator pool counters for diagnostics / load testing.
        /// </summary>
        public CudaAllocatorStats GetStats()
            => pool.GetStats().WithReclaims(ladder.OomTrims, ladder.OomReclaims, ladder.FullDrains);

        /// <summary>Bytes of the large blocks the pool caches: an allocation that finds the device full gives them back
        /// first (<see cref="TrimLargeCache"/>), so they count as free room.</summary>
        public long ReclaimableBytes => pool.LargeCachedBytes;

        /// <summary>The pool's pressure relief as a return under the free-memory reserve runs it, on this thread (tests: the
        /// device need not be full); the bytes freed.</summary>
        internal long RelievePressureForTests(long bytes) => pool.RelievePressure(bytes, keepBlockBytes: 0);

        /// <summary>
        /// Give cached large blocks back to the driver, the least recently used size first, until <paramref name="bytes"/> are
        /// freed or none is left, never one of <paramref name="keepBlockBytes"/> (the size an allocation about to be made
        /// rents); returns the bytes freed. Synchronizes this thread's stream first, so not while a graph is captured here. For callers that
        /// need room outside the pool (a staging arena, a recording): unlike <see cref="TrimPool"/> it keeps every block it
        /// does not need to free, and the small ones.
        /// </summary>
        public long TrimLargeCache(long bytes, long keepBlockBytes = 0)
        {
            ThrowIfDisposed();
            if (bytes <= 0 || pool.LargeCachedBytes == 0) return 0;
            Context.MakeCurrent();
            Stream.Synchronize();
            return pool.TrimLarge(bytes, keepBlockBytes);
        }

        public void Synchronize()
        {
            Context.MakeCurrent();
            Stream.Synchronize();
        }

        /// <summary>Free every pooled (returned) block back to the driver. For
        /// pipeline stage boundaries where the next stage's working set has a
        /// different shape profile (e.g. a diffusion denoise handing over to a
        /// video VAE): live allocations are untouched, but the caller must ensure
        /// no concurrent Rent/Return on this allocator. A caller that only needs
        /// room uses <see cref="TrimLargeCache"/>.</summary>
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
            int shardCount,
            long largeCachedBytes = 0,
            long largeTrimFrees = 0,
            long oomTrims = 0,
            long oomReclaims = 0,
            long fullDrains = 0)
        {
            LargeCachedBytes = largeCachedBytes;
            LargeTrimFrees = largeTrimFrees;
            OomTrims = oomTrims;
            OomReclaims = oomReclaims;
            FullDrains = fullDrains;
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

        /// <summary>Total cuMemAlloc driver calls issued for rents (== <see cref="PoolMissCount"/>, small and large).</summary>
        public long CuMemAllocCount { get; }

        /// <summary>Total cuMemFree driver calls the pool issued: returns past the cache, trims and drains.</summary>
        public long CuMemFreeCount { get; }

        /// <summary>Bytes of the large (2 MiB and up) blocks cached, part of <see cref="CachedBytes"/>.</summary>
        public long LargeCachedBytes { get; }

        /// <summary>Cached large blocks given back to the driver to make room (trims, not drains).</summary>
        public long LargeTrimFrees { get; }

        /// <summary>Allocations that found the device full and succeeded once cached blocks went back.</summary>
        public long OomTrims { get; }

        /// <summary>Allocations that found the device full and succeeded once a registered reclaimer released memory.</summary>
        public long OomReclaims { get; }

        /// <summary>Allocations that found the device full and reached the last resort: a full GC and a drain of every
        /// cached block.</summary>
        public long FullDrains { get; }

        /// <summary>These counters with the allocator's out-of-memory outcomes.</summary>
        internal CudaAllocatorStats WithReclaims(long oomTrims, long oomReclaims, long fullDrains)
            => new(PoolEnabled, PoolHitCount, PoolMissCount, CuMemAllocCount, CuMemFreeCount, ReturnedToPoolCount, CachedBytes,
                PeakCachedBytes, MaxCachedBytes, MaxCachedBlockBytes, ShardCount, LargeCachedBytes, LargeTrimFrees, oomTrims,
                oomReclaims, fullDrains);

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
                   $"cachedBytes={CachedBytes}, largeCachedBytes={LargeCachedBytes}, peakCachedBytes={PeakCachedBytes}, " +
                   $"largeTrimFrees={LargeTrimFrees}, oomTrims={OomTrims}, oomReclaims={OomReclaims}, fullDrains={FullDrains})";
        }
    }
}
