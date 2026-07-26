using System;
using System.Collections.Generic;
using System.Threading;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Striped, size-classed cache of reusable device-memory blocks sitting in
    /// front of a backing allocate/free pair (normally cuMemAlloc / cuMemFree).
    ///
    /// The original allocator guarded a single <c>Dictionary&lt;long, Stack&lt;IntPtr&gt;&gt;</c>
    /// with one global monitor. Under high-concurrency serving that monitor — plus
    /// any single shared <c>cachedBytes</c> counter — becomes a cache-line hot spot:
    /// every rent and return of a small transient tensor serializes on it.
    ///
    /// This pool removes that hot spot by partitioning into <see cref="ShardCount"/>
    /// independent shards (a power of two derived from the CPU count). A thread is
    /// routed to a shard by its current processor id, so in steady state each core
    /// works an almost-private shard: the per-shard monitor is essentially
    /// uncontended, the array-backed <see cref="Stack{IntPtr}"/> allocates nothing on
    /// the hot path, and each shard owns its own byte counter (no shared atomic).
    /// Blocks are fungible, so a block rented on one shard may be returned to another
    /// after a thread migrates — that only rebalances the cache, it is never unsafe.
    ///
    /// The total byte cap is preserved: the sum of the per-shard caps equals the
    /// configured <c>maxCachedBytes</c>.
    /// </summary>
    public sealed class CudaDeviceMemoryPool
    {
        private sealed class Shard
        {
            public readonly object Sync = new object();
            public readonly Dictionary<long, Stack<IntPtr>> Pool = new Dictionary<long, Stack<IntPtr>>();
            public long CachedBytes;
            public long PeakCachedBytes;
            public long HitCount;
            public long MissCount;
            public long ReturnedToPoolCount;
            public long FreedCount;
        }

        private readonly Shard[] shards;
        private readonly int shardMask;
        private readonly long maxCachedBytes;
        private readonly long perShardCap;
        private readonly long maxCachedBlockBytes;
        private readonly bool enabled;
        private readonly Func<long, IntPtr> backingAllocate;
        private readonly Action<IntPtr> backingFree;

        // Large transient blocks (prefill-sized activations) are pooled globally
        // instead of per shard: the per-shard byte cap (maxCachedBytes / shards)
        // is far below a single multi-MB activation, so without this every
        // prefill layer re-issued cuMemAlloc/cuMemFree for its big activations —
        // steady-state launch-path churn, and a capture-breaking allocation when
        // the loop runs under CUDA-graph capture. Rare + low-frequency, so one
        // global lock is fine.
        private const long LargeBlockThreshold = 2L << 20;
        private readonly object largeSync = new object();
        private readonly Dictionary<long, Stack<IntPtr>> largePool = new Dictionary<long, Stack<IntPtr>>();
        private readonly long largeCap;
        private long largeCachedBytes;

        // Each thread is pinned to one shard for the life of the thread. Reading a
        // [ThreadStatic] is a couple of nanoseconds — far cheaper than
        // Thread.GetCurrentProcessorId() — and gives perfect locality: a tensor
        // allocated and freed on the same thread always hits its own shard, so the
        // per-shard monitor stays uncontended. The seed is assigned round-robin the
        // first time a thread touches any pool; masking by each pool's shardMask
        // keeps it valid across pools with different shard counts.
        [ThreadStatic] private static int t_shardSeed;
        private static int s_shardSeedCounter;

        // VRAM-pressure safety valve. On a device that is nearly full (e.g. a 26B
        // model on a 16 GB card, which fits in dedicated VRAM only with a few
        // hundred MB to spare), letting the pool hoard up to ~1.5 GB of idle freed
        // blocks pushes the working set past physical VRAM and the driver silently
        // spills into WDDM shared (system) memory — inference then reads weights
        // over PCIe and collapses to a fraction of its speed. When set, the pool
        // stops caching returned blocks (frees them straight back to the driver)
        // once free VRAM drops below minFreeReserveBytes, keeping headroom so real
        // allocations stay resident. On a roomy card free VRAM never drops that low
        // and the valve never fires, so behaviour there is unchanged. The free
        // reading is sampled at most once per FreeSampleIntervalTicks to keep the
        // hot return path from calling cuMemGetInfo on every block.
        private readonly Func<long> queryFreeBytes;
        private readonly long minFreeReserveBytes;
        private static readonly long FreeSampleIntervalTicks =
            System.Diagnostics.Stopwatch.Frequency / 100; // ~10 ms
        private long lastFreeReading;
        private long lastFreeReadTicks;

        public CudaDeviceMemoryPool(
            long maxCachedBytes,
            long maxCachedBlockBytes,
            bool enabled,
            Func<long, IntPtr> backingAllocate,
            Action<IntPtr> backingFree,
            int shardCount = 0,
            long largeCachedBytesCap = 0,
            Func<long> queryFreeBytes = null,
            long minFreeReserveBytes = 0)
        {
            this.maxCachedBytes = maxCachedBytes;
            this.maxCachedBlockBytes = maxCachedBlockBytes;
            this.enabled = enabled;
            largeCap = largeCachedBytesCap > 0 ? largeCachedBytesCap : maxCachedBytes;
            this.backingAllocate = backingAllocate ?? throw new ArgumentNullException(nameof(backingAllocate));
            this.backingFree = backingFree ?? throw new ArgumentNullException(nameof(backingFree));
            this.queryFreeBytes = queryFreeBytes;
            this.minFreeReserveBytes = minFreeReserveBytes;

            int count = shardCount > 0 ? shardCount : DefaultShardCount();
            count = RoundUpToPowerOfTwo(count);
            shards = new Shard[count];
            for (int i = 0; i < count; i++)
                shards[i] = new Shard();
            shardMask = count - 1;
            // Split the global cap evenly so the sum of per-shard caps == maxCachedBytes.
            perShardCap = Math.Max(maxCachedBytes / count, 0);
        }

        public int ShardCount => shards.Length;

        private Shard CurrentShard()
        {
            int seed = t_shardSeed;
            if (seed == 0)
            {
                // Interlocked.Increment never returns 0 on the first hit for a thread,
                // so 0 stays reserved as the "uninitialized" sentinel.
                seed = Interlocked.Increment(ref s_shardSeedCounter);
                t_shardSeed = seed;
            }

            return shards[seed & shardMask];
        }

        public IntPtr Rent(long requestedBytes, out long allocationBytes)
        {
            allocationBytes = RoundAllocationSize(Math.Max(requestedBytes, 1));

            if (enabled && allocationBytes >= LargeBlockThreshold)
            {
                lock (largeSync)
                {
                    if (largePool.TryGetValue(allocationBytes, out Stack<IntPtr> largeStack) && largeStack.Count > 0)
                    {
                        IntPtr pooled = largeStack.Pop();
                        largeCachedBytes -= allocationBytes;
                        return pooled;
                    }
                }

                return backingAllocate(allocationBytes);
            }

            if (enabled)
            {
                Shard shard = CurrentShard();
                lock (shard.Sync)
                {
                    if (shard.Pool.TryGetValue(allocationBytes, out Stack<IntPtr> stack) && stack.Count > 0)
                    {
                        IntPtr pooled = stack.Pop();
                        shard.CachedBytes -= allocationBytes;
                        shard.HitCount++;
                        return pooled;
                    }

                    shard.MissCount++;
                }
            }

            // Backing allocation runs outside the shard lock so a slow driver call
            // never blocks other threads sharing the shard.
            return backingAllocate(allocationBytes);
        }

        public void Return(IntPtr ptr, long allocationBytes)
        {
            if (TryReturnToPool(ptr, allocationBytes))
                return;

            backingFree(ptr);
        }

        /// <summary>
        /// Like <see cref="Return"/> but never falls back to the backing free:
        /// returns false when the block cannot be pooled (over the block-size or
        /// shard cap, or pooling disabled) so the caller can decide its fate.
        /// Used during CUDA graph capture, where blocks referenced by the graph
        /// must never be cuMemFree'd.
        /// </summary>
        /// <summary>True when the device is close enough to full that the pool
        /// should stop hoarding freed blocks and hand them back to the driver so
        /// live allocations do not spill into shared memory. Samples free VRAM at
        /// most once per <see cref="FreeSampleIntervalTicks"/> to stay off the hot
        /// path; always false when no query / reserve is configured (roomy cards).</summary>
        private bool UnderVramPressure()
        {
            if (queryFreeBytes == null || minFreeReserveBytes <= 0)
                return false;

            long now = System.Diagnostics.Stopwatch.GetTimestamp();
            long last = Volatile.Read(ref lastFreeReadTicks);
            if (last == 0 || now - last >= FreeSampleIntervalTicks)
            {
                long free = queryFreeBytes();
                Volatile.Write(ref lastFreeReading, free);
                Volatile.Write(ref lastFreeReadTicks, now);
                return free < minFreeReserveBytes;
            }

            return Volatile.Read(ref lastFreeReading) < minFreeReserveBytes;
        }

        public bool TryReturnToPool(IntPtr ptr, long allocationBytes)
        {
            if (ptr == IntPtr.Zero)
                return true;

            if (!enabled || allocationBytes <= 0)
                return false;

            // Near-full device: keep headroom instead of caching this block.
            if (UnderVramPressure())
                return false;

            if (allocationBytes >= LargeBlockThreshold)
            {
                lock (largeSync)
                {
                    if (largeCachedBytes + allocationBytes > largeCap)
                        return false;

                    if (!largePool.TryGetValue(allocationBytes, out Stack<IntPtr> largeStack))
                    {
                        largeStack = new Stack<IntPtr>();
                        largePool[allocationBytes] = largeStack;
                    }

                    largeStack.Push(ptr);
                    largeCachedBytes += allocationBytes;
                    return true;
                }
            }

            if (allocationBytes > maxCachedBlockBytes)
                return false;

            Shard shard = CurrentShard();
            lock (shard.Sync)
            {
                if (shard.CachedBytes + allocationBytes > perShardCap)
                    return false;

                if (!shard.Pool.TryGetValue(allocationBytes, out Stack<IntPtr> stack))
                {
                    stack = new Stack<IntPtr>();
                    shard.Pool[allocationBytes] = stack;
                }

                stack.Push(ptr);
                shard.CachedBytes += allocationBytes;
                if (shard.CachedBytes > shard.PeakCachedBytes)
                    shard.PeakCachedBytes = shard.CachedBytes;
                shard.ReturnedToPoolCount++;
                return true;
            }
        }

        /// <summary>
        /// Remove a specific free block from whichever shard holds it. Returns
        /// false when the block is not currently pooled (still live, or already
        /// freed). Used when a cached CUDA graph takes ownership of the blocks
        /// its captured kernels reference.
        /// </summary>
        public bool TrySteal(IntPtr ptr, long allocationBytes)
        {
            if (ptr == IntPtr.Zero)
                return false;

            if (allocationBytes >= LargeBlockThreshold)
            {
                lock (largeSync)
                {
                    if (largePool.TryGetValue(allocationBytes, out Stack<IntPtr> largeStack)
                        && largeStack.Count > 0 && largeStack.Contains(ptr))
                    {
                        var keptLarge = new Stack<IntPtr>();
                        bool removedLarge = false;
                        while (largeStack.Count > 0)
                        {
                            IntPtr candidate = largeStack.Pop();
                            if (!removedLarge && candidate == ptr)
                            {
                                removedLarge = true;
                                continue;
                            }
                            keptLarge.Push(candidate);
                        }
                        while (keptLarge.Count > 0)
                            largeStack.Push(keptLarge.Pop());
                        if (removedLarge)
                        {
                            largeCachedBytes -= allocationBytes;
                            return true;
                        }
                    }
                }
                return false;
            }

            foreach (Shard shard in shards)
            {
                lock (shard.Sync)
                {
                    if (!shard.Pool.TryGetValue(allocationBytes, out Stack<IntPtr> stack) || stack.Count == 0)
                        continue;
                    if (!stack.Contains(ptr))
                        continue;

                    var kept = new Stack<IntPtr>();
                    bool removed = false;
                    while (stack.Count > 0)
                    {
                        IntPtr candidate = stack.Pop();
                        if (!removed && candidate == ptr)
                        {
                            removed = true;
                            continue;
                        }
                        kept.Push(candidate);
                    }
                    // Preserve original ordering (kept is reversed by the pops).
                    while (kept.Count > 0)
                        stack.Push(kept.Pop());

                    if (removed)
                    {
                        shard.CachedBytes -= allocationBytes;
                        return true;
                    }
                }
            }

            return false;
        }

        public CudaAllocatorStats GetStats()
        {
            long hit = 0, miss = 0, returned = 0, freed = 0, cached = 0, peak = 0;
            lock (largeSync)
                cached += largeCachedBytes;
            foreach (Shard shard in shards)
            {
                lock (shard.Sync)
                {
                    hit += shard.HitCount;
                    miss += shard.MissCount;
                    returned += shard.ReturnedToPoolCount;
                    freed += shard.FreedCount;
                    cached += shard.CachedBytes;
                    peak += shard.PeakCachedBytes;
                }
            }

            // Every miss issues exactly one backing allocate; every freed return
            // issues exactly one backing free.
            return new CudaAllocatorStats(
                enabled,
                hit,
                miss,
                cuMemAllocCount: miss,
                cuMemFreeCount: freed,
                returnedToPoolCount: returned,
                cachedBytes: cached,
                peakCachedBytes: peak,
                maxCachedBytes: maxCachedBytes,
                maxCachedBlockBytes: maxCachedBlockBytes,
                shardCount: shards.Length);
        }

        /// <summary>
        /// Frees every cached block via the backing free callback. Callers must
        /// guarantee no concurrent Rent/Return (i.e. call only during disposal).
        /// </summary>
        public void DrainAndFree()
        {
            lock (largeSync)
            {
                foreach (Stack<IntPtr> stack in largePool.Values)
                {
                    while (stack.Count > 0)
                    {
                        IntPtr ptr = stack.Pop();
                        if (ptr != IntPtr.Zero)
                            backingFree(ptr);
                    }
                }
                largePool.Clear();
                largeCachedBytes = 0;
            }

            foreach (Shard shard in shards)
            {
                lock (shard.Sync)
                {
                    foreach (Stack<IntPtr> stack in shard.Pool.Values)
                    {
                        while (stack.Count > 0)
                        {
                            IntPtr ptr = stack.Pop();
                            if (ptr != IntPtr.Zero)
                                backingFree(ptr);
                        }
                    }

                    shard.Pool.Clear();
                    shard.CachedBytes = 0;
                }
            }
        }

        internal static long RoundAllocationSize(long bytes)
        {
            const long smallMax = 1L << 20;
            if (bytes <= 256)
                return 256;

            if (bytes <= smallMax)
            {
                long size = 256;
                while (size < bytes)
                    size <<= 1;
                return size;
            }

            const long largeAlignment = 1L << 20;
            return ((bytes + largeAlignment - 1) / largeAlignment) * largeAlignment;
        }

        private static int DefaultShardCount()
        {
            // One shard per logical core (rounded up to a power of two), capped so
            // the per-shard cap does not become uselessly small.
            int cores = Math.Max(1, Environment.ProcessorCount);
            return Math.Min(RoundUpToPowerOfTwo(cores), 64);
        }

        private static int RoundUpToPowerOfTwo(int value)
        {
            if (value <= 1)
                return 1;
            int result = 1;
            while (result < value)
                result <<= 1;
            return result;
        }
    }
}
