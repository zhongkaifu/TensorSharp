// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The CUDA allocator's targeted trim and its free-memory reading on a real device (DeviceMemoryReclaimLadderTests run the
// out-of-memory ladder itself over a fake device). Nothing here fills the device.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public class CudaAllocatorReclaimTests
{
    private const long MiB = 1L << 20;

    /// <summary>Allocate tensors of <paramref name="bytes"/> each, then dispose them: their blocks go to the pool's cache.</summary>
    private static void Cache(CudaAllocator allocator, long bytes, int count)
    {
        var tensors = Enumerable.Range(0, count).Select(_ => new Tensor(allocator, DType.Float32, bytes / sizeof(float))).ToList();
        allocator.Synchronize();
        tensors.ForEach(t => t.Dispose());
    }

    /// <summary>A trim of Y bytes over cached blocks of X bytes frees at most ceil(Y/X) of them, the least recently used size
    /// first, and leaves the blocks used since cached; each is one cuMemFree.</summary>
    [CudaFact]
    public void TrimLargeCache_FreesOnlyTheBlocksItNeeds_LeastRecentlyUsedFirst()
    {
        using var allocator = new CudaAllocator();
        Cache(allocator, 64 * MiB, 6);
        Cache(allocator, 16 * MiB, 4);
        CudaAllocatorStats before = allocator.GetStats();
        Assert.Equal((6 * 64 + 4 * 16) * MiB, before.LargeCachedBytes);
        Assert.Equal(before.LargeCachedBytes, allocator.ReclaimableBytes);

        long freed = allocator.TrimLargeCache(100 * MiB);

        CudaAllocatorStats after = allocator.GetStats();
        Assert.Equal(128 * MiB, freed);
        Assert.Equal(2, after.LargeTrimFrees - before.LargeTrimFrees);
        Assert.Equal(2, after.CuMemFreeCount - before.CuMemFreeCount);
        Assert.Equal(before.LargeCachedBytes - 128 * MiB, after.LargeCachedBytes);

        // The 16 MiB blocks are served from the cache again: no driver allocation.
        using var again = new Tensor(allocator, DType.Float32, 16 * MiB / sizeof(float));
        Assert.Equal(after.CuMemAllocCount, allocator.GetStats().CuMemAllocCount);

        // Kept size: the 64 MiB blocks stay, the 16 MiB ones go (the one rented again since is not cached).
        long kept = allocator.TrimLargeCache(1L << 40, keepBlockBytes: 64 * MiB);
        Assert.Equal(3 * 16 * MiB, kept);
        Assert.Equal(4 * 64 * MiB, allocator.ReclaimableBytes);
        Assert.Equal(0, allocator.GetStats().FullDrains);
    }

    /// <summary>The free memory the allocator reports is the driver's plus its cached large blocks, not the small blocks its
    /// shards keep (those go back only after the large ones).</summary>
    [CudaFact]
    public void TryGetDeviceMemory_CountsCachedLargeBlocks_NotTheShardsSmallOnes()
    {
        using var allocator = new CudaAllocator();
        Cache(allocator, 64 * MiB, 4);
        Cache(allocator, MiB, 96);
        CudaAllocatorStats stats = allocator.GetStats();
        long small = stats.CachedBytes - stats.LargeCachedBytes;
        Assert.True(small >= 96 * MiB, $"small cached {small}");
        Assert.Equal(256 * MiB, stats.LargeCachedBytes);

        Assert.True(allocator.TryGetDeviceMemory(out long free, out long total));
        long driverFree = allocator.GetMemoryInfo().freeBytes;

        Assert.True(total > 0);
        // The two driver readings are a few calls apart and nothing else allocates on this context: well within the
        // 96 MiB of small blocks that must not be counted.
        long counted = free - driverFree;
        Assert.InRange(counted, 256 * MiB - 32 * MiB, 256 * MiB + 32 * MiB);
    }

    /// <summary>A registered reclaimer is kept once, in order, and a removed one is no longer asked (the ladder itself runs
    /// over a fake device in DeviceMemoryReclaimLadderTests).</summary>
    [CudaFact]
    public void Reclaimers_RegisterOnceAndUnregister()
    {
        using var allocator = new CudaAllocator();
        var first = new CountingReclaimer();
        var second = new CountingReclaimer();
        allocator.RegisterReclaimer(first);
        allocator.RegisterReclaimer(second);
        allocator.RegisterReclaimer(first);
        Assert.Equal(new IDeviceMemoryReclaimer[] { first, second }, allocator.RegisteredReclaimers);

        allocator.UnregisterReclaimer(first);
        allocator.UnregisterReclaimer(first);
        allocator.UnregisterReclaimer(null);
        Assert.Equal(new IDeviceMemoryReclaimer[] { second }, allocator.RegisteredReclaimers);

        // An allocation that fits never asks.
        using var t = new Tensor(allocator, DType.Float32, 1024);
        Assert.Equal(0, first.Calls + second.Calls);

        allocator.UnregisterReclaimer(second);
        Assert.Empty(allocator.RegisteredReclaimers);
    }

    private sealed class CountingReclaimer : IDeviceMemoryReclaimer
    {
        public int Calls;

        public long Reclaim(long bytes)
        {
            Calls++;
            return 0;
        }
    }
}
