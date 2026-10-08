// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class WeightStreamingSharedPoolTests
{
    [Fact]
    public void HostAndDeviceSharePool_PlannerIncludesBothPhysicalAllocations()
    {
        var budget = Budget();
        var options = Options(budget);
        using var staging = new StreamingHostBuffer(options, 1024);
        // N=2 needs 1024 device bytes plus a distinct 64-byte host output tile.
        // Only 1024 shared bytes remain, so independent per-pool checks would
        // wrongly admit it. N=1 with three output rows requires 768+64 instead.
        var layout = WeightStreamingExecutor.SelectTileLayout(options, 64, 100, 8, Payload);
        Assert.Equal((3, 1), layout);
        using (var output = new StreamingHostBuffer(options, layout.WeightRows * layout.TokenRows * sizeof(float)))
        using (var device = budget.Reserve(options.DevicePools.Select(pool =>
                   new MemoryCharge(pool, Payload(64, layout.WeightRows, layout.TokenRows)))))
        {
            device.Commit();
            Assert.Equal(1856, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
            Assert.Equal(768, budget.Snapshot().Single(pool => pool.Pool == "gpu").Committed);
            Assert.All(budget.Snapshot(), pool => Assert.True(pool.Available >= 0));
        }
        Assert.Equal(1024, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
        Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu").Committed);
    }

    [Fact]
    public void SharedCapacityTakenAfterPlanning_DeviceRefusalReleasesTemporaryHostCredit()
    {
        var budget = Budget();
        var options = Options(budget);
        using var staging = new StreamingHostBuffer(options, 1024);
        var layout = WeightStreamingExecutor.SelectTileLayout(options, 64, 100, 8, Payload);
        using var competitor = budget.Reserve([new MemoryCharge("shared", 256)]);
        competitor.Commit();
        // Reproduce the executor's immediate host reservation followed by the
        // device reservation. A competing owner can invalidate the plan; failure
        // must return pressure and unwind output staging, with no partial charge
        // to the independent GPU constraint or release of the competitor.
        Assert.Throws<MemoryPressureException>(() =>
        {
            using var output = new StreamingHostBuffer(options, layout.WeightRows * layout.TokenRows * sizeof(float));
            using var device = budget.Reserve(options.DevicePools.Select(pool =>
                new MemoryCharge(pool, Payload(64, layout.WeightRows, layout.TokenRows))));
        });
        Assert.Equal(1280, budget.Snapshot().Single(pool => pool.Pool == "shared").Committed);
        Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu").Committed);
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved));
        competitor.Dispose();
        Assert.Equal(layout, WeightStreamingExecutor.SelectTileLayout(options, 64, 100, 8, Payload));
    }

    private static MemoryBudget Budget()
        => new([new MemoryCharge("shared", 2048), new MemoryCharge("gpu", 1024)]);

    private static WeightStreamingOptions Options(MemoryBudget budget)
        => new(budget, "shared", ["shared", "gpu"], tileBytes: 1024, tokenTileRows: 8);

    // Independent fake backend with aligned input, weight and output regions.
    // Native integration tests validate the actual CUDA workspace separately.
    private static long Payload(long width, int rows, int tokens)
        => Align(width * tokens * sizeof(float)) + Align(width / 32 * 34 * rows)
            + Align((long)rows * tokens * sizeof(float));
    private static long Align(long bytes) => (bytes + 255) / 256 * 256;
}
