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

        // Each thread is pinned to one shard for the life of the thread. Reading a
        // [ThreadStatic] is a couple of nanoseconds — far cheaper than
        // Thread.GetCurrentProcessorId() — and gives perfect locality: a tensor
        // allocated and freed on the same thread always hits its own shard, so the
        // per-shard monitor stays uncontended. The seed is assigned round-robin the
        // first time a thread touches any pool; masking by each pool's shardMask
        // keeps it valid across pools with different shard counts.
        [ThreadStatic] private static int t_shardSeed;
        private static int s_shardSeedCounter;

        public CudaDeviceMemoryPool(
            long maxCachedBytes,
            long maxCachedBlockBytes,
            bool enabled,
            Func<long, IntPtr> backingAllocate,
            Action<IntPtr> backingFree,
            int shardCount = 0)
        {
            this.maxCachedBytes = maxCachedBytes;
            this.maxCachedBlockBytes = maxCachedBlockBytes;
            this.enabled = enabled;
            this.backingAllocate = backingAllocate ?? throw new ArgumentNullException(nameof(backingAllocate));
            this.backingFree = backingFree ?? throw new ArgumentNullException(nameof(backingFree));

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
            if (ptr == IntPtr.Zero)
                return;

            if (enabled && allocationBytes > 0 && allocationBytes <= maxCachedBlockBytes)
            {
                Shard shard = CurrentShard();
                lock (shard.Sync)
                {
                    if (shard.CachedBytes + allocationBytes <= perShardCap)
                    {
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
                        return;
                    }

                    shard.FreedCount++;
                }
            }

            backingFree(ptr);
        }

        public CudaAllocatorStats GetStats()
        {
            long hit = 0, miss = 0, returned = 0, freed = 0, cached = 0, peak = 0;
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
