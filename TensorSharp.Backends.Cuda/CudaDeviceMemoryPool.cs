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
    /// The byte cap is the pool's, not a shard's: a shard caches past its even share of
    /// <c>maxCachedBytes</c> while every shard's small blocks together stay within it (the
    /// shards' counters are summed only then, so the hot path keeps no shared counter). A
    /// cap split evenly over the shards was 8 MB a shard on a 48-core host (64 shards of
    /// 512 MB): one thread's step that used more than that freed the rest to the driver
    /// and allocated it again on the next step - every DFlash2 draft pass of Qwen3.8-27B
    /// on an RTX PRO 4000 issued ~70 cuMemAlloc and ~70 synchronizing cuMemFree calls.
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
        private long largeHitCount;      // under largeSync
        private long largeMissCount;     // under largeSync
        private long largeTrimFrees;     // under largeSync: blocks TrimLarge gave back to the driver

        // Under largeSync: when each large size was last rented or returned (a use counter, not a time), so a trim gives
        // back the sizes the work stopped using first. Largest-first freed the hot ones: a 24K-token prompt of
        // Qwen3.8-27B on an RTX PRO 4500 returns 136, 129, 112 and 81 MiB activations at every layer, the largest blocks
        // the cache holds, and trimmed for each other's room 750 times (55% slower prefill) while stale sizes stayed.
        private readonly Dictionary<long, long> largeLastUse = new Dictionary<long, long>();
        private long largeUseClock;

        // Backing frees issued by trims and drains (TrimLarge, DrainSmall, DrainAndFree), beside the shards' FreedCount of
        // returns past the cache: together every cuMemFree the pool issues.
        private long trimAndDrainFrees;

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
        // ~100 ms: cuMemGetInfo took 86 us an A40 call, and at 10 ms a decode step (~19 ms) paid it every token.
        private static readonly long FreeSampleIntervalTicks =
            System.Diagnostics.Stopwatch.Frequency / 10;
        private long lastFreeReading;
        private long lastFreeReadTicks;

        // Whether a return that finds the device under the reserve, or the large-block cache at its cap, may give back cached
        // large blocks the work stopped using instead of its own block (null: never; false while a CUDA graph is captured on
        // the allocator, which must not free, and while a capture that just ended is taking the blocks its work returned to
        // the pool: CudaGraphCapture.MayFreeCachedBlocksUnasked).
        // The valve alone handed every returned block to the driver while the cache kept stale sizes: once a 3.1 GB K/V slot
        // left an RTX PRO 4500 at the reserve, a 47K-token prompt of Qwen3.8-27B freed and allocated its own activations
        // again at every layer (~1,400 synchronizing cuMemFree calls, about 4% of the prefill).
        private readonly Func<bool> mayRelievePressure;

        /// <summary>What a relief frees beyond the reserve or the cap, so the next returns do not relieve again at once.</summary>
        private const long PressureReliefSlackBytes = 64L << 20;

        public CudaDeviceMemoryPool(
            long maxCachedBytes,
            long maxCachedBlockBytes,
            bool enabled,
            Func<long, IntPtr> backingAllocate,
            Action<IntPtr> backingFree,
            int shardCount = 0,
            long largeCachedBytesCap = 0,
            Func<long> queryFreeBytes = null,
            long minFreeReserveBytes = 0,
            Func<bool> mayRelievePressure = null)
        {
            this.mayRelievePressure = mayRelievePressure;
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
            // A shard's even share of the cap; past it a shard caches only while the pool has room (CachedSnapshot).
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
                    largeLastUse[allocationBytes] = ++largeUseClock;
                    if (largePool.TryGetValue(allocationBytes, out Stack<IntPtr> largeStack) && largeStack.Count > 0)
                    {
                        IntPtr pooled = largeStack.Pop();
                        largeCachedBytes -= allocationBytes;
                        largeHitCount++;
                        return pooled;
                    }
                    largeMissCount++;
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

        /// <summary>Whether a rent of <paramref name="requestedBytes"/> would take a cached large block (no driver allocation):
        /// the pool caches large blocks by their exact rounded size. False for small blocks (per-thread shards).</summary>
        public bool HasCachedBlock(long requestedBytes)
        {
            long allocationBytes = RoundAllocationSize(Math.Max(requestedBytes, 1));
            if (!enabled || allocationBytes < LargeBlockThreshold)
                return false;
            lock (largeSync)
                return largePool.TryGetValue(allocationBytes, out Stack<IntPtr> stack) && stack.Count > 0;
        }

        /// <summary>Every shard's cached small blocks, read without their locks (a shard past its share asks).</summary>
        private long CachedSnapshot()
        {
            long total = 0;
            foreach (Shard shard in shards)
                total += Volatile.Read(ref shard.CachedBytes);
            return total;
        }

        public void Return(IntPtr ptr, long allocationBytes)
        {
            if (TryReturnToPool(ptr, allocationBytes))
                return;

            backingFree(ptr);
            Shard shard = CurrentShard();
            lock (shard.Sync)
                shard.FreedCount++;
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
        /// path; always false when no query / reserve is configured (roomy cards).
        /// Under the reserve, cached large blocks the work stopped using (the least
        /// recently used sizes, never <paramref name="returningBytes"/>) go back to
        /// the driver first where <see cref="mayRelievePressure"/> allows it; the
        /// block being returned is cached when they restored the reserve.</summary>
        private bool UnderVramPressure(long returningBytes)
        {
            if (queryFreeBytes == null || minFreeReserveBytes <= 0)
                return false;

            long free;
            long now = System.Diagnostics.Stopwatch.GetTimestamp();
            long last = Volatile.Read(ref lastFreeReadTicks);
            if (last == 0 || now - last >= FreeSampleIntervalTicks)
            {
                free = queryFreeBytes();
                Volatile.Write(ref lastFreeReading, free);
                Volatile.Write(ref lastFreeReadTicks, now);
            }
            else
            {
                free = Volatile.Read(ref lastFreeReading);
            }

            if (free >= minFreeReserveBytes)
                return false;
            if (LargeCachedBytes == 0)
                return true;
            long need = minFreeReserveBytes - free + PressureReliefSlackBytes;
            return RelievePressure(need, keepBlockBytes: returningBytes) < need;
        }

        /// <summary>The pressure valve's relief, and the large-block cap's: cached large blocks the work stopped using go back to
        /// the driver, unasked by their holder and without a synchronize (<see cref="TrimLarge"/>), where
        /// <see cref="mayRelievePressure"/> allows it; the bytes freed (0 where it is not allowed). It runs on whichever thread
        /// returns a block, the GC finalizer's included.</summary>
        internal long RelievePressure(long bytes, long keepBlockBytes, long usedBefore = long.MaxValue)
            => mayRelievePressure == null || !mayRelievePressure() ? 0 : TrimLarge(bytes, keepBlockBytes, usedBefore);

        public bool TryReturnToPool(IntPtr ptr, long allocationBytes)
        {
            if (ptr == IntPtr.Zero)
                return true;

            if (!enabled || allocationBytes <= 0)
                return false;

            // Near-full device: keep headroom instead of caching this block (unless stale cached blocks made the room).
            if (UnderVramPressure(allocationBytes))
                return false;

            if (allocationBytes >= LargeBlockThreshold)
            {
                if (allocationBytes > largeCap)
                    return false;

                // At the cap, the cached sizes the work stopped using go back first (the least recently used, never this
                // size), as under the reserve and where that is allowed; the block is cached once they made its room.
                // Refusing it instead handed every returned activation to the driver while the blocks of K/V slots given up
                // earlier filled the cache: 577 refusals, ~650 synchronizing cuMemFree/cuMemAlloc pairs and 613 ms of
                // cuMemFree in one 829 ms prompt step of Qwen3.8-27B at a 69K context on an RTX PRO 4500 (issue #376).
                // Only sizes not used since this size was last used go back: a size used since then is in the working set
                // of the op this block served. Where an op's blocks together pass the cap, least recently used alone gave up
                // each size of the op for the next one, and every size was allocated and freed again at every op.
                long over, usedBefore;
                lock (largeSync)
                {
                    over = largeCachedBytes + allocationBytes - largeCap;
                    usedBefore = largeLastUse.TryGetValue(allocationBytes, out long used) ? used : largeUseClock;
                }
                if (over > 0)
                    RelievePressure(over + PressureReliefSlackBytes, keepBlockBytes: allocationBytes, usedBefore);

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
                    largeLastUse[allocationBytes] = ++largeUseClock;
                    return true;
                }
            }

            if (allocationBytes > maxCachedBlockBytes)
                return false;

            Shard shard = CurrentShard();
            lock (shard.Sync)
            {
                if (shard.CachedBytes + allocationBytes > perShardCap
                    && CachedSnapshot() + allocationBytes > maxCachedBytes)
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
            long hit = 0, miss = 0, returned = 0, freed = 0, cached = 0, peak = 0, largeCached, trimFrees;
            lock (largeSync)
            {
                largeCached = largeCachedBytes;
                cached += largeCachedBytes;
                hit += largeHitCount;
                miss += largeMissCount;
                trimFrees = largeTrimFrees;
            }
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

            // Every miss (small or large) issues exactly one backing allocate; every freed return, trim and drain one
            // backing free each.
            return new CudaAllocatorStats(
                enabled,
                hit,
                miss,
                cuMemAllocCount: miss,
                cuMemFreeCount: freed + Interlocked.Read(ref trimAndDrainFrees),
                returnedToPoolCount: returned,
                cachedBytes: cached,
                peakCachedBytes: peak,
                maxCachedBytes: maxCachedBytes,
                maxCachedBlockBytes: maxCachedBlockBytes,
                shardCount: shards.Length,
                largeCachedBytes: largeCached,
                largeTrimFrees: trimFrees);
        }

        /// <summary>Bytes of the large blocks (2 MiB and up) cached for reuse: what <see cref="TrimLarge"/> gives back. The
        /// shards' small blocks are left out (they go back only in <see cref="DrainSmall"/>).</summary>
        public long LargeCachedBytes
        {
            get { lock (largeSync) return largeCachedBytes; }
        }

        /// <summary>
        /// Give cached large blocks back to the driver until <paramref name="bytes"/> are freed or none is left: the size
        /// least recently rented or returned first (the work stopped using it; the sizes every layer uses stay cached), the
        /// larger first among sizes as old, never a block of <paramref name="keepBlockBytes"/> (the size an allocation about
        /// to be made rents from the cache). Returns the bytes freed. Queued work may still use a block it returned: the
        /// allocator's ladder and TrimLargeCache synchronize the stream first; the pressure valve relies on cuMemFree waiting
        /// for the device, as its own free of a returned block always has. Before this, an allocation that found the device
        /// full freed every cached block of every size: 1,614 cuMemFree calls and 1,091 ms in one prompt step of
        /// Qwen3.8-27B at a 24K context (issue #376).
        /// </summary>
        public long TrimLarge(long bytes, long keepBlockBytes = 0) => TrimLarge(bytes, keepBlockBytes, long.MaxValue);

        /// <summary><see cref="TrimLarge(long, long)"/> over the sizes last rented or returned before the use-clock value
        /// <paramref name="usedBefore"/> only (the large-block cap's relief: sizes not used since the returning size was).</summary>
        internal long TrimLarge(long bytes, long keepBlockBytes, long usedBefore)
        {
            if (bytes <= 0) return 0;
            List<IntPtr> release = null;
            long freed = 0;
            lock (largeSync)
            {
                if (largeCachedBytes == 0) return 0;
                var sizes = new List<(long LastUse, long Size)>(largePool.Count);
                foreach (KeyValuePair<long, Stack<IntPtr>> bucket in largePool)
                {
                    if (bucket.Value.Count == 0 || bucket.Key == keepBlockBytes)
                        continue;
                    long used = largeLastUse.TryGetValue(bucket.Key, out long u) ? u : 0;
                    if (used < usedBefore)
                        sizes.Add((used, bucket.Key));
                }
                sizes.Sort((a, b) => a.LastUse != b.LastUse ? a.LastUse.CompareTo(b.LastUse) : b.Size.CompareTo(a.Size));
                foreach ((long _, long size) in sizes)
                {
                    if (freed >= bytes) break;
                    Stack<IntPtr> stack = largePool[size];
                    while (stack.Count > 0 && freed < bytes)
                    {
                        (release ??= new List<IntPtr>()).Add(stack.Pop());
                        largeCachedBytes -= size;
                        freed += size;
                    }
                    if (stack.Count == 0)
                    {
                        // A size the work stopped using leaves no entry behind (prompt lengths make many sizes).
                        largePool.Remove(size);
                        largeLastUse.Remove(size);
                    }
                }
                largeTrimFrees += release?.Count ?? 0;
            }

            // Freed outside the lock: each cuMemFree synchronizes, and other threads' large rents and returns go on meanwhile
            // (the blocks taken here are no longer in the cache).
            FreeAll(release);
            return freed;
        }

        /// <summary>Give every shard's cached small blocks back to the driver; returns the bytes freed. The caller makes sure
        /// no queued work still uses a cached block.</summary>
        public long DrainSmall()
        {
            long freed = 0;
            List<IntPtr> release = null;
            foreach (Shard shard in shards)
            {
                lock (shard.Sync)
                {
                    if (shard.CachedBytes == 0)
                        continue;
                    foreach (Stack<IntPtr> stack in shard.Pool.Values)
                        while (stack.Count > 0)
                            (release ??= new List<IntPtr>()).Add(stack.Pop());
                    freed += shard.CachedBytes;
                    shard.Pool.Clear();
                    shard.CachedBytes = 0;
                }
            }

            FreeAll(release);
            return freed;
        }

        /// <summary>
        /// Frees every cached block, large and small, via the backing free callback: for disposal, a pipeline's stage
        /// boundary (<see cref="CudaAllocator.TrimPool"/>) and the allocator's last resort when an allocation finds the
        /// device full. Each block is taken from the cache under its lock, so concurrent rents and returns stay consistent;
        /// the caller makes sure no queued work still uses a cached block.
        /// </summary>
        public void DrainAndFree()
        {
            List<IntPtr> release = null;
            lock (largeSync)
            {
                foreach (Stack<IntPtr> stack in largePool.Values)
                    while (stack.Count > 0)
                        (release ??= new List<IntPtr>()).Add(stack.Pop());
                largePool.Clear();
                largeLastUse.Clear();
                largeCachedBytes = 0;
            }

            FreeAll(release);
            DrainSmall();
        }

        private void FreeAll(List<IntPtr> blocks)
        {
            if (blocks == null)
                return;
            int count = 0;
            foreach (IntPtr ptr in blocks)
            {
                if (ptr == IntPtr.Zero)
                    continue;
                backingFree(ptr);
                count++;
            }
            Interlocked.Add(ref trimAndDrainFrees, count);
            // The device has room again: the next return reads the free memory afresh instead of handing its block to the
            // driver on a reading taken before the trim (for up to FreeSampleIntervalTicks).
            if (count > 0)
                Volatile.Write(ref lastFreeReadTicks, 0);
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
