// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #383 on a real device: NVML's reading of the CUDA device the allocator runs on, the budget guard over the real
// allocator (a stand-in reading with a budget a little above what the device holds now: past it the device still has room,
// as WDDM's shared memory had), and blocks a disposed holder returns going back to the driver. DedicatedDeviceMemoryTests run
// the reading and the ladder over fakes. Nothing here fills the device.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public class CudaDedicatedMemoryTests
{
    private const long MiB = 1L << 20;

    /// <summary>NVML finds the allocator's device by its UUID: its total is the device's memory (on Linux cuMemGetInfo's
    /// total within a few percent: the driver keeps some), its use device-wide and at least what this process allocated; the
    /// view takes the smaller figures.</summary>
    [CudaFact]
    public void NvmlReadsTheAllocatorsDevice_AndTheViewTakesTheSmallerFigures()
    {
        using var allocator = new CudaAllocator();
        IDedicatedMemoryProbe nvml = NvmlDedicatedMemoryProbe.Create(allocator.DeviceId);
        Assert.NotNull(nvml);   // the VM's driver ships libnvidia-ml.so.1
        Assert.True(nvml.TryRead(out long used, out long total));
        (long driverFree, long driverTotal) = DriverReading(allocator);
        Assert.InRange(total, driverTotal * 95 / 100, driverTotal * 105 / 100);

        using (var held = new Tensor(allocator, DType.Float32, 512 * MiB / sizeof(float)))
        {
            Assert.True(nvml.TryRead(out long usedWith, out _));
            Assert.True(usedWith - used >= 400 * MiB, $"NVML use moved {(usedWith - used) / MiB} MiB for a 512 MiB tensor");
            Assert.True(allocator.MemoryView.TryRead(out long free, out long viewTotal));
            Assert.True(viewTotal <= Math.Min(total, driverTotal));
            Assert.True(free <= DriverReading(allocator).Free);
        }
        Assert.True(allocator.MemoryView.HasDedicatedProbe);
        Assert.False(allocator.MemoryView.AllocationsMaySpill);   // Linux: a full device fails the allocation itself
    }

    /// <summary>With a budget just above what the device holds, an allocation that would pass it gives back the cached large
    /// blocks it needs first (the budget guard; the device itself still has room, as WDDM's shared memory had), and the
    /// device-wide use stays within the budget.</summary>
    [CudaFact]
    public void ABudgetOverTheRealAllocator_TrimsCachedBlocksBeforeAnAllocationPassesIt()
    {
        using var allocator = new CudaAllocator();
        IDedicatedMemoryProbe nvml = NvmlDedicatedMemoryProbe.Create(allocator.DeviceId);
        Assert.True(nvml.TryRead(out long usedBefore, out _));
        // Cache 4 x 256 MiB, then allow 1.5 GiB past what the device held before: a 1 GiB allocation fits only once some of
        // the cached blocks go (1 GiB cached + 1 GiB + the 256 MiB reserve > 1.5 GiB).
        foreach (var t in Enumerable.Range(0, 4).Select(_ => new Tensor(allocator, DType.Float32, 256 * MiB / sizeof(float))).ToList())
        {
            allocator.Synchronize();
            t.Dispose();
        }
        Assert.Equal(1024 * MiB, allocator.GetStats().LargeCachedBytes);
        long budget = usedBefore + 1536 * MiB;
        allocator.MemoryViewForTests = new DeviceMemoryView(() => DriverReading(allocator), nvml, budget, driverMaySpill: false);
        Assert.True(allocator.MemoryView.AllocationsMaySpill);

        using var big = new Tensor(allocator, DType.Float32, 1024 * MiB / sizeof(float));
        allocator.Synchronize();

        Assert.True(nvml.TryRead(out long usedAfter, out _));
        Assert.True(usedAfter <= budget, $"device-wide use {usedAfter / MiB} MiB past the {budget / MiB} MiB budget");
        Assert.True(allocator.GetStats().LargeCachedBytes < 1024 * MiB);
        Assert.Equal(1, allocator.BudgetGuardOutcomes.Trims);
        Assert.Equal(0, allocator.BudgetGuardOutcomes.Passes);
        Assert.Equal(0, allocator.GetStats().FullDrains);
    }

    /// <summary>Inside a ReturningToDriver scope a disposed tensor's large block goes back to the driver (one cuMemFree, nothing
    /// cached); small blocks and blocks returned outside the scope are cached as before.</summary>
    [CudaFact]
    public void ReturningToDriver_FreesLargeBlocks_AndCachesTheRest()
    {
        using var allocator = new CudaAllocator();
        var large = new Tensor(allocator, DType.Float32, 64 * MiB / sizeof(float));
        var small = new Tensor(allocator, DType.Float32, 1024);
        var outside = new Tensor(allocator, DType.Float32, 32 * MiB / sizeof(float));
        allocator.Synchronize();
        CudaAllocatorStats before = allocator.GetStats();

        using (allocator.ReturningToDriver())
        {
            large.Dispose();
            small.Dispose();
        }
        outside.Dispose();

        CudaAllocatorStats after = allocator.GetStats();
        Assert.Equal(1, after.CuMemFreeCount - before.CuMemFreeCount);
        Assert.Equal(32 * MiB, after.LargeCachedBytes - before.LargeCachedBytes);
        Assert.True(after.CachedBytes - after.LargeCachedBytes > before.CachedBytes - before.LargeCachedBytes);
    }

    private static (long Free, long Total) DriverReading(CudaAllocator allocator)
    {
        allocator.Context.MakeCurrent();
        TensorSharp.Cuda.Interop.CudaDriverApi.cuMemGetInfo(out UIntPtr free, out UIntPtr total);
        return ((long)free.ToUInt64(), (long)total.ToUInt64());
    }
}
