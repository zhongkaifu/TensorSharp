// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class Qwen35VisionPoolReclaimTests
{
    [Fact]
    public async Task BusyMediaLock_DoesNotReadStatsOrWaitForTheOppositeModelLock()
    {
        object modelLock = new(), mediaLock = new();
        using var mediaHeld = new ManualResetEventSlim();
        using var modelAcquired = new ManualResetEventSlim();
        Task worker;
        lock (modelLock)
        {
            // This is the real inversion: preparation owns media and needs model,
            // while allocation pressure already owns model and must decline media.
            worker = Task.Run(() =>
            {
                lock (mediaLock)
                {
                    mediaHeld.Set();
                    lock (modelLock)
                        modelAcquired.Set();
                }
            });
            Assert.True(mediaHeld.Wait(TimeSpan.FromSeconds(5)));
            Assert.False(Qwen35CudaModel.TryTrimIdleVisionPool(mediaLock,
                () => throw new InvalidOperationException("busy stats must not run"),
                () => throw new InvalidOperationException("busy trim must not run"),
                out long before, out long after));
            Assert.Equal(0, before);
            Assert.Equal(0, after);
            Assert.False(modelAcquired.IsSet);
        }
        await worker.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(modelAcquired.IsSet);
    }

    [Fact]
    public void EmptyPool_DoesNotTrim_AndReleasesMediaLock()
    {
        object mediaLock = new();
        int reads = 0, trims = 0;
        Assert.False(Qwen35CudaModel.TryTrimIdleVisionPool(mediaLock,
            () => { ++reads; return 0; }, () => ++trims, out long before, out long after));
        Assert.Equal(0, before);
        Assert.Equal(0, after);
        Assert.Equal(1, reads);
        Assert.Equal(0, trims);
        AssertOtherThreadCanEnter(mediaLock);
    }

    [Theory]
    [InlineData(4096L, 0L, true)]
    [InlineData(4096L, 1024L, true)]
    [InlineData(4096L, 4096L, false)]
    [InlineData(4096L, 8192L, false)]
    public void OnlyAnObservedCacheReductionReportsProgress(long initial, long final, bool expected)
    {
        object mediaLock = new();
        int reads = 0, trims = 0;
        Assert.Equal(expected, Qwen35CudaModel.TryTrimIdleVisionPool(mediaLock,
            () => { Assert.True(Monitor.IsEntered(mediaLock)); return ++reads == 1 ? initial : final; },
            () => { Assert.True(Monitor.IsEntered(mediaLock)); ++trims; }, out long before, out long after));
        Assert.Equal(initial, before);
        Assert.Equal(final, after);
        Assert.Equal(2, reads);
        Assert.Equal(1, trims);
        AssertOtherThreadCanEnter(mediaLock);
    }

    [Theory]
    [InlineData("before")]
    [InlineData("trim")]
    [InlineData("after")]
    public void StatsOrTrimException_IsNotReportedAsProgress_AndReleasesMediaLock(string failure)
    {
        object mediaLock = new();
        var error = new InvalidOperationException(failure);
        int reads = 0, trims = 0;
        long Stats()
        {
            ++reads;
            if (failure == "before" || failure == "after" && reads == 2)
                throw error;
            return 4096;
        }
        void Trim()
        {
            ++trims;
            if (failure == "trim") throw error;
        }
        Assert.Same(error, Assert.Throws<InvalidOperationException>(() =>
            Qwen35CudaModel.TryTrimIdleVisionPool(mediaLock, Stats, Trim, out _, out _)));
        Assert.Equal(failure == "before" ? 0 : 1, trims);
        AssertOtherThreadCanEnter(mediaLock);
    }

    [Fact]
    public void ACompletedDrainCannotReportEndlessProgress()
    {
        object mediaLock = new();
        long cached = 4096;
        int trims = 0;
        void Trim() { ++trims; cached = 0; }
        Assert.True(Qwen35CudaModel.TryTrimIdleVisionPool(mediaLock, () => cached, Trim,
            out long firstBefore, out long firstAfter));
        Assert.False(Qwen35CudaModel.TryTrimIdleVisionPool(mediaLock, () => cached, Trim,
            out long nextBefore, out long nextAfter));
        Assert.Equal(4096, firstBefore);
        Assert.Equal(0, firstAfter);
        Assert.Equal(0, nextBefore);
        Assert.Equal(0, nextAfter);
        Assert.Equal(1, trims);
    }

    [Fact]
    public void SameThreadMediaOwnership_IsPreservedAfterNestedTryEnter()
    {
        object mediaLock = new();
        long cached = 4096;
        lock (mediaLock)
        {
            Assert.True(Qwen35CudaModel.TryTrimIdleVisionPool(mediaLock, () => cached,
                () => cached = 0, out _, out _));
            Assert.True(Monitor.IsEntered(mediaLock));
        }
        AssertOtherThreadCanEnter(mediaLock);
    }

    [Fact]
    public void RealPoolDrain_FreesOnlyReturnedBlocks_NotLiveOrGraphOwnedBlocks()
    {
        long nextPointer = 100;
        var freed = new List<IntPtr>();
        var pool = new CudaDeviceMemoryPool(1 << 20, 1 << 20, true,
            _ => new IntPtr(++nextPointer), freed.Add, shardCount: 1);
        IntPtr live = pool.Rent(256, out long liveBytes);
        IntPtr graphOwned = pool.Rent(512, out long graphBytes);
        IntPtr idle = pool.Rent(1024, out long idleBytes);
        pool.Return(graphOwned, graphBytes);
        Assert.True(pool.TrySteal(graphOwned, graphBytes));
        pool.Return(idle, idleBytes);

        Assert.True(Qwen35CudaModel.TryTrimIdleVisionPool(new object(),
            () => pool.GetStats().CachedBytes, pool.DrainAndFree, out long before, out long after));

        Assert.Equal(idleBytes, before);
        Assert.Equal(0, after);
        Assert.Equal(new[] { idle }, freed);
        Assert.DoesNotContain(live, freed);
        Assert.DoesNotContain(graphOwned, freed);
        // The allocator can still accept an owned tensor after the pressure drain.
        pool.Return(live, liveBytes);
        Assert.Equal(liveBytes, pool.GetStats().CachedBytes);
        pool.DrainAndFree();
        Assert.Equal(new[] { idle, live }, freed);
        Assert.DoesNotContain(graphOwned, freed);
    }

    private static void AssertOtherThreadCanEnter(object gate)
    {
        Task<bool> acquired = Task.Run(() =>
        {
            if (!Monitor.TryEnter(gate, 0)) return false;
            Monitor.Exit(gate);
            return true;
        });
        Assert.True(acquired.Wait(TimeSpan.FromSeconds(5)));
        Assert.True(acquired.Result);
    }
}
