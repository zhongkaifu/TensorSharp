// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public class AdaptiveModelMemoryTests
{
    private const long GiB = 1L << 30;
    private static MemoryBudget EmptyBudget() => new([
        new(AdaptiveModelSession.HostPool, 0), new(AdaptiveModelSession.DevicePool, 0)]);
    private static DenseMemoryProfile Profile(string? refusal = null) => new(new()
    {
        DenseWeights = new(4 * GiB, 4 * GiB), Persistent = new(16 << 20, 16 << 20),
        KvCaches = [new(65536, AllocationBlockTokens: 256)]
    }, 1 * GiB, 64 << 20, 4096, 14336, 32, 65536, refusal);

    [Fact]
    public void AvailableHardwareSelectsResidentInsteadOfTinyStreamingDefault()
    {
        var plan = AdaptiveModelSession.PlanLoad(Profile(), new(2048, 512),
            new(32 * GiB, 20 * GiB, 16 * GiB, 14 * GiB), EmptyBudget());
        Assert.True(plan.Accepted);
        Assert.Equal(InferenceWeightPlacement.Resident, plan.SelectedCandidate!.Placement);
        Assert.Equal(512, plan.SelectedChunkTokens);
        Assert.True(plan.SelectedCandidate.PreservesExecutionGraph);
    }

    [Fact]
    public void ExplicitDeviceCeilingSelectsSupportedStreamingAndRetainsContextAccounting()
    {
        var options = new AdaptiveModelMemoryOptions(2048, 512) { MaximumDeviceBytes = GiB };
        var plan = AdaptiveModelSession.PlanLoad(Profile(), options,
            new(32 * GiB, 20 * GiB, 16 * GiB, 14 * GiB), EmptyBudget());
        Assert.True(plan.Accepted);
        Assert.Equal(InferenceWeightPlacement.SsdStreaming, plan.SelectedCandidate!.Placement);
        Assert.Contains(plan.Components, c => c.Name == "kv-0" && c.Bytes.Device == 2048L * 65536);
        Assert.All(plan.PoolPeaks.Where(p => p.Pool == AdaptiveModelSession.DevicePool), p => Assert.True(p.Peak <= GiB));
        Assert.True(AdaptiveModelSession.DeviceCacheLimit(plan, 4 * GiB, options.MaximumStreamingWorkspaceCacheBytes) > 0);
        Assert.Equal(0, AdaptiveModelSession.DeviceCacheLimit(plan, 4 * GiB, 0));
    }

    [Fact]
    public void UnsupportedQuantizationCannotBecomeAnUnboundedFallback()
    {
        var options = new AdaptiveModelMemoryOptions(2048, 512) { MaximumDeviceBytes = GiB };
        var plan = AdaptiveModelSession.PlanLoad(Profile("IQ2 streaming unavailable"), options,
            new(32 * GiB, 20 * GiB, 16 * GiB, 14 * GiB), EmptyBudget());
        Assert.False(plan.Accepted);
        Assert.Contains(plan.Rejections, r => r.Kind == InferenceMemoryRejectionKind.Unsupported);
    }

    [Fact]
    public void LoadFusionPeakCanRejectOtherwiseResidentDeviceWeights()
    {
        var plan = AdaptiveModelSession.PlanLoad(Profile(), new(2048, 512) { AllowWeightStreaming = false },
            new(32 * GiB, 3 * GiB, 16 * GiB, 14 * GiB), EmptyBudget());
        Assert.False(plan.Accepted);
        Assert.Contains(plan.Rejections, r => r.Pool == AdaptiveModelSession.HostPool);
    }

    [Fact]
    public void PolicyRejectsImpossibleGeometry()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelMemoryPolicy(512, 1024));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelMemoryPolicy(0, 1));
        Assert.Equal(512, new ModelMemoryPolicy(4096, 512).PrefillChunkTokens);
    }

    [Fact]
    public void SharedLedgerCeilingsAndExistingOwnersConstrainPlacementAndCaches()
    {
        var budget = new MemoryBudget([new(AdaptiveModelSession.HostPool, 8 * GiB), new(AdaptiveModelSession.DevicePool, GiB)]);
        using var owner = budget.Reserve([new(AdaptiveModelSession.HostPool, 2 * GiB), new(AdaptiveModelSession.DevicePool, 256 << 20)]);
        owner.Commit();
        var before = budget.Snapshot().ToArray();
        var plan = AdaptiveModelSession.PlanLoad(Profile(), new(2048, 512),
            new(32 * GiB, 20 * GiB, 16 * GiB, 14 * GiB), budget, sharedBudget: true);
        Assert.True(plan.Accepted);
        Assert.Equal(InferenceWeightPlacement.SsdStreaming, plan.SelectedCandidate!.Placement);
        Assert.Equal(before, budget.Snapshot());
        var host = plan.Capacities.Single(p => p.Pool == AdaptiveModelSession.HostPool);
        Assert.Equal(6 * GiB, host.AdditionalAvailable);
        var device = plan.Capacities.Single(p => p.Pool == AdaptiveModelSession.DevicePool);
        long peak = plan.PoolPeaks.Where(p => p.Pool == AdaptiveModelSession.DevicePool).Select(p => Math.Max(p.Prefill, p.Decode)).Single();
        Assert.Equal(Math.Max(0, device.AdditionalAvailable - peak), AdaptiveModelSession.DeviceCacheLimit(plan, 4 * GiB, long.MaxValue));
        Assert.True(budget.TrySetCapacity(AdaptiveModelSession.DevicePool, 256 << 20));
        var rejected = AdaptiveModelSession.PlanLoad(Profile(), new(2048, 512),
            new(32 * GiB, 20 * GiB, 16 * GiB, 14 * GiB), budget, sharedBudget: true);
        Assert.False(rejected.Accepted);
    }
}
