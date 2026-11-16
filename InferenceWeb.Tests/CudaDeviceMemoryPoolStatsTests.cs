using TensorSharp.Cuda;

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
}
