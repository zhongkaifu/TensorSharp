// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryAdmissionCohortTests
{
    private sealed class Scope : IDisposable { public void Dispose() { } }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void LastHolderLeavingRetriesPhysicalAdmissionEvenWhenRetainedChildrenDoNotPulseBudget(bool abort)
    {
        var budget = new MemoryBudget([new("device", 100)]);
        using var optionalJoint = budget.Reserve([new("device", 40)]);
        int samples = 0;
        var cfg = new SchedulerConfig
        {
            BlockSize = 8, NumBlocks = 8192, MaxNumRunningSequences = 3,
            MaxNumBatchedTokens = 8, MaxPrefillChunkSize = 4,
            EnablePrefixCaching = false, StopRepetition = false,
            ExecutionOptionsOverride = new()
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = true,
                BatchedFusedDecodeEnabled = false, KvHolderPoolMax = 0
            },
            MemoryAdmission = new(budget, s => [new("device", s.RequestId == "first" ? 60 : 40)])
            {
                UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
                EnterSerialExecution = _ => new Scope(),
                BeforeAdmissionWithRunning = (s, running) =>
                {
                    samples++;
                    if (s.RequestId != "first")
                    {
                        if (running.Count != 0)
                            throw new MemoryPressureException("A live holder prevents physical reclamation.");
                        // A quiescent adapter can return unused joint credit even
                        // when the previous holder remains a retained checkpoint.
                        optionalJoint.Dispose();
                    }
                    return [new("device", 100)];
                }
            }
        };
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 0);
        using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var first = new SequenceState("first", [1, 2, 3, 4], 20000, 8, SamplingConfig.Greedy);
        var second = new SequenceState("second", [1, 2, 3, 4], 3, 8, SamplingConfig.Greedy);
        var third = new SequenceState("third", [1, 2, 3, 4], 3, 8, SamplingConfig.Greedy);
        scheduler.Submit(first);
        Assert.Same(first, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        first.AdvanceComputedTokens(4);
        using var retained = first.MemoryEnvelope!.TryTake([new("device", 60)])!;
        retained.Commit();
        scheduler.Submit(second); scheduler.Submit(third);
        for (int i = 0; i < 16; i++)
        {
            Assert.Same(first, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
            Assert.True(scheduler.MemoryAdmissionBlocked);
        }
        Assert.Equal(2, samples); // First admission and one refused neighbour.
        var unchanged = budget.ChangeSignal;
        if (abort) Assert.True(scheduler.Abort(first.RequestId));
        else scheduler.NotifyStop(first, SequenceStatus.FinishedStopped, "stop", new SchedulerOutput());
        scheduler.NotifyMemoryReleased(first.RequestId);
        Assert.Same(unchanged, budget.ChangeSignal);
        Assert.False(unchanged.IsCompleted);
        Assert.Equal(60, budget.Snapshot().Single().Committed);
        Assert.Equal(40, budget.Snapshot().Single().Reserved);

        Assert.Same(second, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        Assert.Equal(4, samples); // Idle reclamation, then the third waits once.
        Assert.True(scheduler.Abort(second.RequestId));
        scheduler.NotifyMemoryReleased(second.RequestId);
        Assert.Same(third, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        Assert.Equal(5, samples);
        scheduler.Abort(third.RequestId); scheduler.NotifyMemoryReleased(third.RequestId);
        retained.Dispose();
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }
}
