using System;
using System.Collections.Generic;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public class CudaDeviceMemoryPoolStatsTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Return_CountsBackingFreesWithoutCountingCachedBlocks(bool enabled)
    {
        int frees = 0;
        var pool = new CudaDeviceMemoryPool(
            maxCachedBytes: 256,
            maxCachedBlockBytes: 256,
            enabled: enabled,
            backingAllocate: _ => new IntPtr(1),
            backingFree: _ => frees++,
            shardCount: 1);

        pool.Return(new IntPtr(1), 256);
        pool.Return(new IntPtr(2), 256);
        pool.Return(IntPtr.Zero, 256);

        Assert.Equal(enabled ? 1 : 2, frees);
        Assert.Equal(frees, pool.GetStats().CuMemFreeCount);
        Assert.Equal(enabled ? 256 : 0, pool.GetStats().CachedBytes);
    }
    /// <summary>The cap is the pool's: one thread's shard caches past its even share (64 shards of 512 MB were 8 MB each, and
    /// a step using more freed the rest to the driver and allocated it again the next step) while the shards' blocks together
    /// stay within the cap; past it a return goes to the driver.</summary>
    [Fact]
    public void Return_CachesPastAShardsShareWhileThePoolHasRoom()
    {
        int frees = 0;
        var pool = new CudaDeviceMemoryPool(
            maxCachedBytes: 1024,
            maxCachedBlockBytes: 256,
            enabled: true,
            backingAllocate: _ => new IntPtr(1),
            backingFree: _ => frees++,
            shardCount: 64);

        // Pinned to one processor's shard as far as a test can: one thread, back to back (a migration only moves a block
        // to another shard, which the pool-wide total counts the same).
        for (int i = 1; i <= 4; i++)
            pool.Return(new IntPtr(i), 256);
        Assert.Equal(0, frees);
        Assert.Equal(1024, pool.GetStats().CachedBytes);

        pool.Return(new IntPtr(5), 256);
        Assert.Equal(1, frees);
        Assert.Equal(1024, pool.GetStats().CachedBytes);
    }

    private const long MiB = 1L << 20;

    /// <summary>A pool over a fake backing allocator that hands out distinct pointers and records each free's block size.</summary>
    private sealed class FakeBacking
    {
        private long _next = 0x1000;
        private readonly Dictionary<IntPtr, long> _sizes = new();
        public readonly List<long> Frees = new();
        public int Allocations;

        public CudaDeviceMemoryPool Pool(long largeCap = 64L << 30, long maxCached = 64 * MiB)
            => new(maxCached, maxCachedBlockBytes: 256 * MiB, enabled: true, Allocate, Free, shardCount: 1,
                largeCachedBytesCap: largeCap);

        private IntPtr Allocate(long bytes)
        {
            Allocations++;
            var ptr = new IntPtr(_next += 0x1000);
            _sizes[ptr] = bytes;
            return ptr;
        }

        private void Free(IntPtr ptr)
        {
            Frees.Add(_sizes[ptr]);
            _sizes.Remove(ptr);
        }
    }

    /// <summary>Rent blocks of each size, then return them all: the pool caches them.</summary>
    private static void Cache(CudaDeviceMemoryPool pool, params long[] sizes)
    {
        var rented = new List<(IntPtr, long)>();
        foreach (long size in sizes)
            rented.Add((pool.Rent(size, out long bytes), bytes));
        foreach ((IntPtr ptr, long bytes) in rented)
            pool.Return(ptr, bytes);
    }

    /// <summary>A trim gives back the sizes least recently rented or returned first, whatever their size, and stops as soon
    /// as the bytes asked are freed: an allocation that found the device full used to free every cached block of every size
    /// (1,614 synchronizing cuMemFree calls in one step of issue #376), and largest-first freed the activations every layer
    /// uses (they are the largest blocks), which the next layer allocated again.</summary>
    [Fact]
    public void TrimLarge_FreesTheLeastRecentlyUsedSizesFirst_OnlyUntilTheBytesAskedAreFreed()
    {
        var backing = new FakeBacking();
        CudaDeviceMemoryPool pool = backing.Pool();
        Cache(pool, 256 * MiB, 64 * MiB, 64 * MiB, 8 * MiB, 512 * KiB);
        IntPtr hot = pool.Rent(256 * MiB, out long hotBytes);   // the largest size is the one in use
        pool.Return(hot, hotBytes);
        long cachedLarge = pool.LargeCachedBytes;
        Assert.Equal((256 + 64 + 64 + 8) * MiB, cachedLarge);

        long freed = pool.TrimLarge(100 * MiB);

        Assert.Equal(128 * MiB, freed);
        Assert.Equal(new[] { 64 * MiB, 64 * MiB }, backing.Frees);
        Assert.Equal(cachedLarge - freed, pool.LargeCachedBytes);
        CudaAllocatorStats stats = pool.GetStats();
        Assert.Equal(2, stats.LargeTrimFrees);
        Assert.Equal(2, stats.CuMemFreeCount);
        Assert.Equal(cachedLarge - freed, stats.LargeCachedBytes);
        Assert.Equal(cachedLarge - freed + 512 * KiB, stats.CachedBytes);

        Assert.Equal(8 * MiB, pool.TrimLarge(1));
        Assert.Equal(new[] { 64 * MiB, 64 * MiB, 8 * MiB }, backing.Frees);

        // The size in use is served from the cache again, and the small block: nothing else was given back.
        int allocations = backing.Allocations;
        pool.Rent(256 * MiB, out _);
        pool.Rent(512 * KiB, out _);
        Assert.Equal(allocations, backing.Allocations);
    }

    private const long KiB = 1L << 10;

    /// <summary>The size an allocation about to be made rents stays cached, whatever its rank by size.</summary>
    [Fact]
    public void TrimLarge_LeavesTheKeptSizeCached()
    {
        var backing = new FakeBacking();
        CudaDeviceMemoryPool pool = backing.Pool();
        Cache(pool, 4 * MiB, 16 * MiB, 128 * MiB);

        long freed = pool.TrimLarge(130 * MiB, keepBlockBytes: 128 * MiB);

        Assert.Equal(20 * MiB, freed);
        Assert.Equal(new[] { 4 * MiB, 16 * MiB }, backing.Frees);
        Assert.Equal(128 * MiB, pool.LargeCachedBytes);
    }

    /// <summary>Asked for more than the cache holds, a trim frees what there is and says so; nothing asked, nothing freed.</summary>
    [Fact]
    public void TrimLarge_FreesWhatIsCachedWhenAskedForMore_AndNothingWhenAskedForNothing()
    {
        var backing = new FakeBacking();
        CudaDeviceMemoryPool pool = backing.Pool();
        Cache(pool, 4 * MiB, 4 * MiB, 32 * MiB, 256 * KiB);

        Assert.Equal(0, pool.TrimLarge(0));
        Assert.Empty(backing.Frees);

        Assert.Equal(40 * MiB, pool.TrimLarge(1L << 40));
        Assert.Equal(new[] { 4 * MiB, 4 * MiB, 32 * MiB }, backing.Frees);
        Assert.Equal(0, pool.LargeCachedBytes);
        Assert.Equal(0, pool.TrimLarge(MiB));
        Assert.Equal(256 * KiB, pool.GetStats().CachedBytes);   // the shard's small block is not the trim's
    }

    /// <summary>The large cache's bytes leave out the shards' small blocks; the drain of the small blocks leaves out the
    /// large ones; a full drain frees both, and every free is counted as a cuMemFree.</summary>
    [Fact]
    public void LargeCachedBytes_AndDrainSmall_KeepTheTwoCachesApart()
    {
        var backing = new FakeBacking();
        CudaDeviceMemoryPool pool = backing.Pool();
        Cache(pool, 1 * MiB, 1 * MiB, 64 * KiB, 2 * MiB, 32 * MiB);

        Assert.Equal(34 * MiB, pool.LargeCachedBytes);
        Assert.Equal(34 * MiB + 2 * MiB + 64 * KiB, pool.GetStats().CachedBytes);

        Assert.Equal(2 * MiB + 64 * KiB, pool.DrainSmall());
        Assert.Equal(3, backing.Frees.Count);
        Assert.All(backing.Frees, size => Assert.True(size < 2 * MiB));
        Assert.Equal(34 * MiB, pool.LargeCachedBytes);
        Assert.Equal(34 * MiB, pool.GetStats().CachedBytes);
        Assert.Equal(0, pool.DrainSmall());

        pool.DrainAndFree();
        Assert.Equal(5, backing.Frees.Count);
        CudaAllocatorStats stats = pool.GetStats();
        Assert.Equal(0, stats.CachedBytes);
        Assert.Equal(5, stats.CuMemFreeCount);
        Assert.Equal(0, stats.LargeTrimFrees);   // drains are not trims
    }

    /// <summary>After a trim the next return reads the device's free memory afresh: on a reading taken before the trim the
    /// pool's pressure valve kept handing returned blocks to the driver (or caching them) for up to a sampling interval. The
    /// assertion holds however long the steps take (a later return re-reads anyway).</summary>
    [Fact]
    public void TrimLarge_MakesTheNextReturnReadTheFreeMemoryAgain()
    {
        var backing = new FakeBacking();
        long free = 4L << 30;
        int reads = 0;
        var pool = new CudaDeviceMemoryPool(64 * MiB, 256 * MiB, enabled: true,
            bytes => { backing.Allocations++; return new IntPtr(0x100000 + backing.Allocations * 0x1000); }, _ => backing.Frees.Add(0),
            shardCount: 1, largeCachedBytesCap: 1L << 30, queryFreeBytes: () => { reads++; return free; },
            minFreeReserveBytes: 256 * MiB);
        IntPtr a = pool.Rent(32 * MiB, out long aBytes);
        IntPtr b = pool.Rent(16 * MiB, out long bBytes);
        pool.Return(a, aBytes);   // room: cached
        Assert.Equal(1, reads);
        Assert.Equal(32 * MiB, pool.LargeCachedBytes);

        free = 100 * MiB;          // the device filled up since
        Assert.Equal(32 * MiB, pool.TrimLarge(1));
        pool.Return(b, bBytes);

        Assert.Equal(2, reads);
        Assert.Equal(0, pool.LargeCachedBytes);   // read afresh: under the reserve, the block went to the driver
        Assert.Equal(2, backing.Frees.Count);
    }

    /// <summary>Under the free-memory reserve, a return gives back the cached sizes the work stopped using instead of its own
    /// block, and caches its block once they restored the reserve: the valve alone freed every returned block (the activations
    /// each layer allocates again) while stale sizes stayed cached. Not where relief is not allowed (a capture): the block
    /// goes to the driver, as before.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void UnderTheReserve_AReturnGivesBackStaleSizesFirst(bool mayRelieve)
    {
        var frees = new List<IntPtr>();
        long next = 0x100000, free = 4L << 30;
        var pool = new CudaDeviceMemoryPool(64 * MiB, 256 * MiB, enabled: true, _ => new IntPtr(next += 0x1000), frees.Add,
            shardCount: 1, largeCachedBytesCap: 1L << 30, queryFreeBytes: () => free, minFreeReserveBytes: 256 * MiB,
            mayRelievePressure: () => mayRelieve);
        IntPtr decoy = pool.Rent(2 * MiB, out long decoyBytes);
        IntPtr stale1 = pool.Rent(64 * MiB, out long staleBytes), stale2 = pool.Rent(64 * MiB, out _),
            stale3 = pool.Rent(64 * MiB, out _), hot = pool.Rent(136 * MiB, out long hotBytes);
        foreach (IntPtr block in new[] { decoy, stale1, stale2, stale3 })
            pool.Return(block, block == decoy ? decoyBytes : staleBytes);   // room: all cached
        Assert.Equal(194 * MiB, pool.LargeCachedBytes);

        free = 200 * MiB;                     // under the reserve: 56 MiB short, 120 MiB with the slack
        Assert.Equal(2 * MiB, pool.TrimLarge(1));   // the oldest size; a trim makes the next return read afresh
        frees.Clear();
        pool.Return(hot, hotBytes);

        if (mayRelieve)
        {
            Assert.Equal(2, frees.Count);     // two stale 64 MiB blocks, not the block being returned
            Assert.DoesNotContain(hot, frees);
            Assert.All(frees, ptr => Assert.Contains(ptr, new[] { stale1, stale2, stale3 }));
            Assert.Equal(64 * MiB + 136 * MiB, pool.LargeCachedBytes);
        }
        else
        {
            Assert.Equal(new[] { hot }, frees);
            Assert.Equal(192 * MiB, pool.LargeCachedBytes);
        }
    }

    /// <summary>At the large-block cap, a return gives back the cached sizes the work stopped using (blocks of K/V slots given
    /// up earlier) and caches its own block, so a step's activations are allocated once: refused at the cap, every activation
    /// went to the driver and was allocated again by the next op (577 refusals and 613 ms of synchronizing cuMemFree in one
    /// 829 ms prompt step of Qwen3.8-27B at 69K on an RTX PRO 4500, issue #376). Not where relief is not allowed (a capture):
    /// the block goes to the driver, as before. A block larger than the cap is never cached and frees nothing.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AtTheLargeCap_AReturnGivesBackStaleSizesFirst_SoActivationsAreAllocatedOnce(bool mayRelieve)
    {
        var sizes = new Dictionary<IntPtr, long>();
        var frees = new List<long>();
        long next = 0x100000;
        int allocations = 0;
        var pool = new CudaDeviceMemoryPool(64 * MiB, 256 * MiB, enabled: true,
            bytes => { allocations++; var p = new IntPtr(next += 0x1000); sizes[p] = bytes; return p; },
            p => { frees.Add(sizes[p]); sizes.Remove(p); },
            shardCount: 1, largeCachedBytesCap: 512 * MiB, mayRelievePressure: () => mayRelieve);

        // Three blocks of a K/V slot given up: cached, 450 MiB of the 512 MiB cap.
        Cache(pool, 150 * MiB, 150 * MiB, 150 * MiB);
        Assert.Equal(450 * MiB, pool.LargeCachedBytes);
        int staleAllocations = allocations;

        // A step's ops: each rents its activations and returns them; the second return reaches the cap.
        const int Ops = 20;
        for (int op = 0; op < Ops; op++)
        {
            IntPtr a = pool.Rent(40 * MiB, out long aBytes), b = pool.Rent(24 * MiB, out long bBytes);
            pool.Return(a, aBytes);
            pool.Return(b, bBytes);
        }

        if (mayRelieve)
        {
            Assert.Equal(2, allocations - staleAllocations);   // each activation size once
            Assert.Equal(new[] { 150 * MiB }, frees);          // one stale block made the room, no activation went back
            Assert.Equal((2 * 150 + 40 + 24) * MiB, pool.LargeCachedBytes);
        }
        else
        {
            Assert.Equal(1 + Ops, allocations - staleAllocations);   // the refused size is allocated again by every op
            Assert.Equal(Ops, frees.Count);
            Assert.All(frees, bytes => Assert.Equal(24 * MiB, bytes));
            Assert.Equal((450 + 40) * MiB, pool.LargeCachedBytes);
        }

        // Larger than the cap: freed, and the cache keeps what it holds.
        frees.Clear();
        long cached = pool.LargeCachedBytes;
        IntPtr huge = pool.Rent(600 * MiB, out long hugeBytes);
        pool.Return(huge, hugeBytes);
        Assert.Equal(new[] { 600 * MiB }, frees);
        Assert.Equal(cached, pool.LargeCachedBytes);
    }

    /// <summary>A large rent counts as a hit or a miss like a small one, so the cuMemAlloc count covers every allocation.</summary>
    [Fact]
    public void Rent_CountsLargeHitsAndMisses()
    {
        var backing = new FakeBacking();
        CudaDeviceMemoryPool pool = backing.Pool();

        IntPtr large = pool.Rent(10 * MiB, out long bytes);
        IntPtr small = pool.Rent(4 * KiB, out long smallBytes);
        pool.Return(large, bytes);
        pool.Return(small, smallBytes);
        pool.Rent(10 * MiB, out _);
        pool.Rent(4 * KiB, out _);

        CudaAllocatorStats stats = pool.GetStats();
        Assert.Equal(2, backing.Allocations);
        Assert.Equal(2, stats.CuMemAllocCount);
        Assert.Equal(2, stats.PoolMissCount);
        Assert.Equal(2, stats.PoolHitCount);
    }
}
