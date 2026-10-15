// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Memory;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

/// <summary>Actual scheduler/prefix boundary with deterministic allocation failure;
/// no native/model or physical hard-limit qualification.</summary>
public sealed class UnifiedPrefixAdmissionFailureTests
{
    [Fact]
    public void PrefixAdoptionShrinksUnusedCreditBeforeForwardButKeepsItsCloneCharged()
    {
        var budget = new MemoryBudget([new("host", 128)]);
        SequenceState? executing = null;
        BudgetReservation? clone = null;
        int transitions = 0;
        var cfg = new SchedulerConfig
        {
            BlockSize = 4, NumBlocks = 32, MaxNumRunningSequences = 2,
            MaxNumBatchedTokens = 16, MaxPrefillChunkSize = 8,
            SoloPrefillChunkSize = 16, PrefillChunkTokenLimit = 16,
            EnablePrefixCaching = true, StopRepetition = false,
            ExecutionOptionsOverride = new()
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = true,
                BatchedFusedDecodeEnabled = false, PrefixCheckpointBudget = 1,
                RetainedFusedCacheBudget = 1, KvHolderPoolMax = 0
            },
            MemoryAdmission = new(budget, _ => [new("host", 96)])
            {
                UsePerSequenceHolders = true,
                EnterSerialExecution = sequence =>
                {
                    Assert.Null(executing); executing = sequence;
                    return new Scope(() => executing = null);
                },
                AfterPrefixMaterialization = sequence =>
                {
                    Assert.Null(executing);
                    Assert.Equal(12, sequence.NumComputedTokens);
                    Assert.Equal(32, sequence.MemoryEnvelope!.LiveChildCharges.Single().Bytes);
                    Assert.True(sequence.MemoryEnvelope.TryResizeTotal([new("host", 40)]));
                    Assert.Equal(32, budget.Snapshot().Single().Committed);
                    Assert.Equal(8, budget.Snapshot().Single().Reserved);
                    Assert.Equal(88, budget.Snapshot().Single().Available);
                    transitions++;
                }
            }
        };
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 64);
        using var storage = pool.Storage;
        using var model = OracleFakes.R(8);
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        var sequence = new SequenceState("prefix-hit", Enumerable.Range(0, 16).ToArray(), 4, 4, SamplingConfig.Greedy);
        scheduler.AttachRadixCacheContinuation(cache, _ => 12, (seq, length) =>
        {
            Assert.Same(seq, executing);
            clone = seq.MemoryEnvelope!.TryTake([new("host", 32)]);
            Assert.NotNull(clone); clone.Commit();
            foreach (var block in pool.AllocateNew((length + cfg.BlockSize - 1) / cfg.BlockSize)!)
                seq.BlockTable.AppendBlock(block);
            seq.SetComputedTokensForPrefixAdoption(length);
            return true;
        });
        try
        {
            scheduler.Submit(sequence);
            var work = Assert.Single(scheduler.Schedule().ScheduledWork);
            Assert.Same(sequence, work.Sequence);
            Assert.Equal(4, work.NumScheduledTokens);
            Assert.Equal(1, transitions);
            scheduler.Abort(sequence.RequestId);
            scheduler.NotifyMemoryReleased(sequence.RequestId);
            // Closing the request cannot credit the physically live clone.
            Assert.Equal(32, budget.Snapshot().Single().Committed);
            Assert.Equal(96, budget.Snapshot().Single().Available);
        }
        finally
        {
            scheduler.Abort(sequence.RequestId);
            scheduler.NotifyMemoryReleased(sequence.RequestId);
            clone?.Dispose(); cache.Reset(); cache.Detach();
        }
        Assert.Equal(128, budget.Snapshot().Single().Available);
    }

    [Theory]
    [InlineData("scope")]
    [InlineData("lookup")]
    [InlineData("clone")]
    [InlineData("post-adoption")]
    public void FailedPrefixMaterializationDoesNotFailAnAlreadyRunningNeighbour(string failureAt)
    {
        var budget = new MemoryBudget([new("host", 128)]);
        SequenceState? executing = null;
        var injected = new MemoryPressureException("Injected prefix materialization quota refusal");
        var cfg = new SchedulerConfig
        {
            BlockSize = 8, NumBlocks = 32, MaxNumRunningSequences = 2,
            MaxNumBatchedTokens = 8, MaxPrefillChunkSize = 4,
            SoloPrefillChunkSize = 8, PrefillChunkTokenLimit = 8,
            EnablePrefixCaching = true, StopRepetition = false,
            ExecutionOptionsOverride = new()
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = true,
                BatchedFusedDecodeEnabled = false, PrefixCheckpointBudget = 1,
                RetainedFusedCacheBudget = 1, KvHolderPoolMax = 0
            },
            MemoryAdmission = new(budget, _ => [new("host", 48)])
            {
                UsePerSequenceHolders = true,
                AfterPrefixMaterialization = sequence =>
                {
                    Assert.Null(executing);
                    if (sequence.RequestId == "newcomer" && failureAt == "post-adoption")
                    {
                        Assert.Equal(4, sequence.NumComputedTokens);
                        throw injected;
                    }
                },
                EnterSerialExecution = sequence =>
                {
                    Assert.NotNull(sequence.MemoryEnvelope);
                    Assert.Null(executing);
                    if (sequence.RequestId == "newcomer" && failureAt == "scope") throw injected;
                    executing = sequence;
                    return new Scope(() => executing = null);
                }
            }
        };
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 64);
        using var storage = pool.Storage;
        using var model = OracleFakes.R(8);
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        var running = new SequenceState("running", [1, 2, 3, 4], 8, 8, SamplingConfig.Greedy);
        var newcomer = new SequenceState("newcomer", [5, 6, 7, 8], 3, 8, SamplingConfig.Greedy);
        scheduler.AttachRadixCacheContinuation(cache, sequence =>
        {
            Assert.Same(sequence, executing);
            if (sequence.RequestId != "newcomer") return 0;
            if (failureAt == "lookup") throw injected;
            return 4;
        }, (sequence, _) =>
        {
            Assert.Same(sequence, executing);
            Assert.Equal("newcomer", sequence.RequestId);
            if (failureAt == "post-adoption")
            {
                foreach (var block in pool.AllocateNew(1)!) sequence.BlockTable.AppendBlock(block);
                sequence.SetComputedTokensForPrefixAdoption(4);
                return true;
            }
            throw injected;
        });
        try
        {
            scheduler.Submit(running);
            Assert.Same(running, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
            running.SetComputedTokensForPrefixAdoption(4);
            Assert.Null(executing);
            scheduler.Submit(newcomer);

            // Previously this escaped Schedule(). The engine had no output yet
            // and consequently failed every in-flight request, including running.
            SchedulerOutput? output = null;
            Assert.Null(Record.Exception(() => output = scheduler.Schedule()));
            Assert.NotNull(output);
            Assert.Same(running, Assert.Single(output!.ScheduledWork).Sequence);
            Assert.Equal(SequenceStatus.Running, running.Status);
            Assert.Null(running.Error);
            Assert.Equal(SequenceStatus.FinishedError, newcomer.Status);
            Assert.Same(injected, newcomer.Error);
            Assert.Contains(newcomer.RequestId, output.FinishedRequestIds);
            Assert.Null(executing);

            // Engine model-release notification must return only the failed
            // destination's credit; the live neighbour still owns its envelope.
            scheduler.NotifyMemoryReleased(newcomer.RequestId);
            Assert.Null(newcomer.MemoryEnvelope);
            Assert.NotNull(running.MemoryEnvelope);
            Assert.Equal(48, budget.Snapshot().Single().Reserved);
            Assert.Same(running, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        }
        finally
        {
            scheduler.Abort(newcomer.RequestId);
            scheduler.NotifyMemoryReleased(newcomer.RequestId);
            scheduler.Abort(running.RequestId);
            scheduler.NotifyMemoryReleased(running.RequestId);
            cache.Reset(); cache.Detach();
        }
        Assert.Equal(128, budget.Snapshot().Single().Available);
        Assert.Equal(pool.NumBlocks, pool.NumFreeBlocks);
    }

    private sealed class Scope(Action release) : IDisposable
    {
        public void Dispose() => release();
    }
}
