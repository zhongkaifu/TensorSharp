// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed partial class StreamingDeviceWeightCacheTests
{
    private static WeightStreamingOptions WorkspaceOptions(MemoryBudget budget, long ceiling = 1 << 20) =>
        new(budget, "ram", ["gpu"], 512, 8)
        { DeviceWorkspaceCacheBytes = ceiling, DeviceWorkspaceCacheReserveBytes = 0 };

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8, false)]
    [InlineData(1, false)]
    [InlineData(8, true)]
    [InlineData(1, true)]
    public void IdleWorkspacesReplaceInputAndWeightsAndPreserveResidentDispatch(int type, bool resident)
    {
        using var file = new Fixture(type);
        int[] counts = resident ? [8, 1, 3, 8] : [19, 1, 3, 19];
        int[][] requests = counts.Select((n, i) => Enumerable.Repeat(i % 2, n).ToArray()).ToArray();
        float[][] reference;
        using (var cold = new Model(file.Path, WorkspaceOptions(Budget(), 0), resident))
            reference = requests.Select(cold.Forward).ToArray();
        var budget = Budget();
        using var other = budget.Reserve([new("gpu", 64)]); other.Commit();
        using var model = new Model(file.Path, WorkspaceOptions(budget), resident);
        for (int i = 0; i < requests.Length; i++)
        {
            Assert.Equal(reference[i], model.Forward(requests[i]));
            model.ResetKVCache();
        }
        var usage = model.StreamingWeightUsage!.Value;
        Assert.True(usage.DeviceWorkspaceCacheBytes > 0 && usage.DeviceWorkspaceReuses > 0);
        Assert.Equal(resident ? 3 : 1, usage.DeviceSessionCreations);
        Assert.Equal(0, usage.DeviceCacheBytes);
        Assert.Equal(usage.DeviceWorkspaceCacheBytes + 64, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        model.TrimIdleMemory();
        Assert.Equal(64, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        Assert.Equal(reference[0], model.Forward(requests[0]));
        Assert.Equal(usage.DeviceSessionCreations + 1, model.StreamingWeightUsage.Value.DeviceSessionCreations);
        model.Dispose();
        Assert.Equal(64, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void IncompatibleWorkspaceYieldsToPressureWithoutRevokingAnotherOwner()
    {
        using var file = new Fixture(8);
        var budget = Budget();
        using var model = new Model(file.Path, WorkspaceOptions(budget), false);
        model.Forward([0]);
        long cached = model.StreamingWeightUsage!.Value.DeviceWorkspaceCacheBytes;
        long remaining = budget.Snapshot().Single(p => p.Pool == "gpu").Available;
        using var other = budget.Reserve([new("gpu", remaining - 64)]); other.Commit();
        model.Forward([2]); // A different K needs a new arena; old idle payload must yield first.
        var usage = model.StreamingWeightUsage.Value;
        Assert.Equal(cached, usage.DeviceWorkspaceEvictedBytes);
        Assert.Equal(0, usage.DeviceWorkspaceReuses);
        Assert.Equal(remaining - 64 + usage.DeviceWorkspaceCacheBytes, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        model.Dispose();
        Assert.Equal(remaining - 64, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(3)] // Creation failure followed by failed cleanup.
    [InlineData(6)] // Reused input upload failure followed by failed cleanup.
    [Trait("Requires", "NativeTestHooks")]
    public void FailedWorkspaceNeverReturnsToThePoolAndResetRetriesPhysicalRelease(int faults)
    {
        using var file = new Fixture(8);
        using var hooks = new Hooks();
        var budget = Budget();
        using var model = new Model(file.Path, WorkspaceOptions(budget), false);
        if (faults == 6) model.Forward([0]);
        hooks.Fail(faults);
        Assert.ThrowsAny<Exception>(() => model.Forward([1]));
        Assert.Equal(0, model.StreamingWeightUsage!.Value.DeviceWorkspaceCacheBytes);
        Assert.True(budget.Snapshot().Single(p => p.Pool == "gpu") is var gpu && gpu.Reserved + gpu.Committed > 0);
        Assert.Throws<InvalidOperationException>(() => model.Forward([0]));
        model.ResetKVCache();
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        model.Forward([0]);
        model.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void FailedIdleWorkspaceReleasePreservesCreditAndPreventsReuse()
    {
        using var file = new Fixture(8);
        using var hooks = new Hooks();
        var budget = Budget();
        using var model = new Model(file.Path, WorkspaceOptions(budget), false);
        var expected = model.Forward([0]);
        long cached = model.StreamingWeightUsage!.Value.DeviceWorkspaceCacheBytes;
        hooks.Fail(2);
        Assert.ThrowsAny<Exception>(model.TrimIdleMemory);
        Assert.Equal(cached, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        Assert.Throws<InvalidOperationException>(() => model.Forward([0]));
        model.ResetKVCache();
        Assert.Equal(0, model.StreamingWeightUsage.Value.DeviceWorkspaceCacheBytes);
        Assert.Equal(expected, model.Forward([0]));
    }
}
