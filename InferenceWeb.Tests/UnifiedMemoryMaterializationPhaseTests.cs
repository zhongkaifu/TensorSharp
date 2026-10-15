// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryMaterializationPhaseTests
{
    private sealed class Scope : IDisposable { public void Dispose() { } }

    private static SchedulerConfig Config(RequestMemoryAdmission admission, bool prefix = false) => new()
    {
        BlockSize = 8, NumBlocks = 8192, MaxNumRunningSequences = 3,
        MaxNumBatchedTokens = 64, MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 64,
        EnablePrefixCaching = prefix, StopRepetition = false,
        ExecutionOptionsOverride = new()
        {
            BatchedPathDisabled = true, PerSeqFusedEnabled = true,
            BatchedFusedDecodeEnabled = false, KvHolderPoolMax = 0,
            PrefixCheckpointBudget = prefix ? 4 : 0, RetainedFusedCacheBudget = prefix ? 3 : 0,
        },
        MemoryAdmission = admission
    };

    [Fact]
    public void NoHitMaterializationUsesIndependentQuotaThenExecutionGrowthBeforeScheduledWork()
    {
        var budget = new MemoryBudget([new("device", 100)]);
        var phases = new List<string>();
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 80)])
        {
            EstimateMaterializationPeak = _ => [new("device", 20)],
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            EnterSerialExecution = _ => new Scope(),
            BeforeAdmissionWithRunning = (request, _) =>
            {
                Assert.Equal(0, request.NumComputedTokens);
                phases.Add("physical-materialization"); return [new("device", 100)];
            },
            AfterPrefixMaterialization = request =>
            {
                Assert.Equal(0, request.NumComputedTokens); // No presumed hit.
                Assert.Equal(20, request.MemoryEnvelope!.OwnedCharges.Single().Bytes);
                phases.Add("physical-execution");
                Assert.True(request.MemoryEnvelope.TryResizeTotal([new("device", 80)], [new("device", 100)]));
            }
        };
        var config = Config(admission);
        var pool = new BlockPool(config.NumBlocks, config.BlockSize, 0); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(config, pool);
        var request = new SequenceState("cold", [1, 2, 3, 4], 20000, 8, SamplingConfig.Greedy);
        scheduler.Submit(request);
        var work = Assert.Single(scheduler.Schedule().ScheduledWork);
        Assert.Same(request, work.Sequence); Assert.Equal(4, work.NumScheduledTokens);
        Assert.Equal(new[] { "physical-materialization", "physical-execution" }, phases);
        Assert.Equal(0, request.NumComputedTokens);
        Assert.Equal(80, budget.Snapshot().Single().Reserved);
        Assert.True(scheduler.Abort(request.RequestId)); scheduler.NotifyMemoryReleased(request.RequestId);
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public async Task ActualPrefixCloneUsesTailPeakAndPreservesWarmCacheWithColdOutputParity()
    {
        var budget = new MemoryBudget([new("device", 100)]);
        using var warmCache = budget.Reserve([new("device", 65)]);
        var tails = new List<(string Id, int Reused, long Target)>();
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 80)])
        {
            EstimateMaterializationPeak = _ => [new("device", 20)],
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            EnterSerialExecution = request =>
            {
                Assert.NotNull(request.MemoryEnvelope);
                return new Scope();
            },
            BeforeAdmissionWithRunning = (request, _) =>
            {
                Assert.Equal(0, request.NumComputedTokens); // Still no adopted state.
                return [new("device", 100)];
            },
            AfterPrefixMaterialization = request =>
            {
                long target = request.PrefixCacheReusedTokens > 0 ? 25 : 80;
                tails.Add((request.RequestId, request.PrefixCacheReusedTokens, target));
                if (!request.MemoryEnvelope!.TryResizeTotal([new("device", target)], [new("device", 100)]))
                {
                    Assert.Equal(0, request.PrefixCacheReusedTokens);
                    Assert.True(warmCache.TryResizeTotal([new("device", 100 - target)]));
                    Assert.True(request.MemoryEnvelope.TryResizeTotal([new("device", target)], [new("device", 100)]));
                }
            },
            AfterPrefill = request => Assert.True(request.MemoryEnvelope!.TryResizeTotal([new("device", 20)])),
        };
        using var model = OracleFakes.R(8);
        using var engine = new InferenceEngine(model, Config(admission, prefix: true));
        var common = Enumerable.Range(1, 16).ToArray();
        var first = new SequenceState("seed", common.Concat(Enumerable.Range(17, 16)).ToArray(), 3, 8,
            SamplingConfig.Greedy, cacheScope: "one", sharedPrefixTokens: 16);
        await engine.SubmitRequest(first).Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(warmCache.TryResizeTotal([new("device", 65)]));
        var prompt = common.Concat(Enumerable.Range(81, 16)).ToArray();
        var second = new SequenceState("hit", prompt, 3, 8, SamplingConfig.Greedy,
            cacheScope: "two", sharedPrefixTokens: 16);
        var result = await engine.SubmitRequest(second).Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(16, result.PrefixCacheReusedTokens);
        Assert.Contains(("hit", 16, 25L), tails);
        Assert.Equal(65, warmCache.OwnedCharges.Single().Bytes);
        using var cold = new InferenceEngine(OracleFakes.R(8), new SchedulerConfig
        {
            BlockSize = 8, NumBlocks = 256, EnablePrefixCaching = false, StopRepetition = false,
        });
        var baseline = new SequenceState("cold", prompt, 3, 8, SamplingConfig.Greedy);
        await cold.SubmitRequest(baseline).Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(baseline.OutputTokens, second.OutputTokens);
        Assert.Equal(0, engine.WaitingCount);
    }

    [Fact]
    public void ExecutionTransitionRefusalFailsOnlyDestinationAndNextQueuedRequestCanJoinNeighbour()
    {
        var budget = new MemoryBudget([new("device", 100)]);
        var admission = new RequestMemoryAdmission(budget, request => [new("device", request.RequestId == "refused" ? 80 : 40)])
        {
            EstimateMaterializationPeak = _ => [new("device", 20)],
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            EnterSerialExecution = _ => new Scope(), BeforeAdmissionWithRunning = (_, _) => [new("device", 100)],
            AfterPrefixMaterialization = request =>
            {
                long target = request.RequestId == "refused" ? 80 : 40;
                if (!request.MemoryEnvelope!.TryResizeTotal([new("device", target)], [new("device", 100)]))
                    throw new MemoryPressureException("Next execution phase cannot fit its live neighbour.");
            }
        };
        var config = Config(admission); var pool = new BlockPool(config.NumBlocks, config.BlockSize, 0); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(config, pool);
        var neighbour = new SequenceState("neighbour", [1, 2, 3, 4], 20000, 8, SamplingConfig.Greedy);
        scheduler.Submit(neighbour); scheduler.Schedule(); neighbour.AdvanceComputedTokens(4);
        var refused = new SequenceState("refused", [5, 6, 7, 8], 20000, 8, SamplingConfig.Greedy);
        var successor = new SequenceState("successor", [9, 10, 11, 12], 20000, 8, SamplingConfig.Greedy);
        scheduler.Submit(refused); scheduler.Submit(successor);
        var step = scheduler.Schedule();
        Assert.Equal(SequenceStatus.FinishedError, refused.Status);
        Assert.IsType<MemoryPressureException>(refused.Error);
        Assert.Equal(SequenceStatus.Running, neighbour.Status);
        Assert.Equal(SequenceStatus.Running, successor.Status);
        Assert.Contains(step.ScheduledWork, work => ReferenceEquals(work.Sequence, neighbour));
        Assert.Contains(step.ScheduledWork, work => ReferenceEquals(work.Sequence, successor));
        Assert.DoesNotContain(step.ScheduledWork, work => ReferenceEquals(work.Sequence, refused));
        scheduler.NotifyMemoryReleased(refused.RequestId);
        Assert.Equal(0, scheduler.WaitingCount);
        Assert.True(scheduler.Abort(neighbour.RequestId)); scheduler.NotifyMemoryReleased(neighbour.RequestId);
        Assert.Same(successor, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        Assert.True(scheduler.Abort(successor.RequestId)); scheduler.NotifyMemoryReleased(successor.RequestId);
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void MaterializationPhaseCannotBypassWorstNoHitSubmissionCeiling()
    {
        var budget = new MemoryBudget([new("device", 100)]); int samples = 0;
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 101)])
        {
            EstimateMaterializationPeak = _ => [new("device", 10)],
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            EnterSerialExecution = _ => new Scope(), AfterPrefixMaterialization = _ => { },
            BeforeAdmissionWithRunning = (_, _) => { samples++; return [new("device", 100)]; },
        };
        var config = Config(admission); var pool = new BlockPool(config.NumBlocks, config.BlockSize, 0); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(config, pool);
        Assert.Throws<MemoryPressureException>(() => scheduler.Submit(new("impossible", [1, 2], 20000, 8, SamplingConfig.Greedy)));
        Assert.Equal(0, scheduler.WaitingCount); Assert.Equal(0, samples);
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }
}
