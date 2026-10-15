// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.Memory;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>Actual CPU scheduler/executor checks; no native or model qualification.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class UnifiedMemoryExecutionPolicyTests
{
    private static readonly ExecutionOptions Serial = new()
    {
        BatchedPathDisabled = true, PerSeqFusedEnabled = false,
        BatchedFusedDecodeEnabled = false, RetainedFusedCacheBudget = 0,
        PrefixCheckpointBudget = 0, KvHolderPoolMax = 0
    };

    private static SchedulerConfig Config(RequestMemoryAdmission admission, int lanes = 1) => new()
    {
        BlockSize = 8, NumBlocks = 32, MaxNumRunningSequences = lanes,
        MaxNumBatchedTokens = 8, MaxPrefillChunkSize = 8, SoloPrefillChunkSize = 8,
        EnablePrefixCaching = false, StopRepetition = false,
        MemoryAdmission = admission, ExecutionOptionsOverride = Serial
    };
    private static SequenceState Request(string id) => new(id, [1, 2, 3, 4], 3, 8, SamplingConfig.Greedy);
    private sealed class NoOpScope : IDisposable { public void Dispose() { } }

    private static SchedulerConfig HolderConfig(RequestMemoryAdmission admission, bool joint = false, int lanes = 2,
        int blocks = 64) => new()
    {
        BlockSize = 8, NumBlocks = blocks, MaxNumRunningSequences = lanes,
        MaxNumBatchedTokens = 8, MaxPrefillChunkSize = 4, SoloPrefillChunkSize = 8,
        PrefillChunkTokenLimit = 8, EnablePrefixCaching = false, StopRepetition = false,
        MemoryAdmission = admission,
        ExecutionOptionsOverride = new()
        {
            BatchedPathDisabled = true, PerSeqFusedEnabled = true,
            BatchedFusedDecodeEnabled = joint, RetainedFusedCacheBudget = 0,
            PrefixCheckpointBudget = 0, KvHolderPoolMax = 0
        }
    };

    [Fact]
    public void AdmittedHolderPlannerNeverUsesTheSharedPrimaryEvenWhenSolo()
    {
        var budget = new MemoryBudget([new("device", 128)]);
        var cfg = HolderConfig(new(budget, _ => [new("device", 48)])
        { UsePerSequenceHolders = true, EnterSerialExecution = _ => new NoOpScope() });
        var caps = new ExecutionCapabilities { SupportsPerSequenceFusedForward = true };
        foreach (int count in new[] { 1, 2 })
            Assert.Equal(ExecutionPathKind.PerSequenceFused, Assert.Single(ExecutionPlanner.PlanStep(caps,
                cfg.ResolveExecutionOptions(), cfg, new ExecutionStepFeatures { SequenceCount = count }).Candidates));
        Assert.Throws<NotSupportedException>(() => ExecutionPlanner.PlanStep(new ExecutionCapabilities(),
            cfg.ResolveExecutionOptions(), cfg, new ExecutionStepFeatures { SequenceCount = 1 }));
    }

    [Fact]
    public async Task AdmittedHoldersInterleaveTwoLiveHistoriesAndReleaseTheirOwnCredit()
    {
        var budget = new MemoryBudget([new("device", 128)]);
        using var model = new HolderModel();
        int samples = 0, maxRunning = 0;
        InferenceEngine engine = null!;
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 48)])
        {
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { samples++; return [new("device", 128)]; },
            EnterSerialExecution = seq =>
            {
                Assert.True(Monitor.IsEntered(model.GpuComputeLock));
                maxRunning = Math.Max(maxRunning, engine.RunningCount);
                return model.Enter(seq);
            }
        };
        using (engine = new InferenceEngine(model, HolderConfig(admission)))
        {
            var first = engine.SubmitRequest(Request("a"));
            Assert.True(model.FirstForward.Wait(TimeSpan.FromSeconds(3)));
            var second = engine.SubmitRequest(Request("b"));
            model.AllowFirstForward.Set();
            await Task.WhenAll(first.Completion, second.Completion).WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal(new[] { 5, 5, 5 }, first.Sequence.OutputTokens);
            Assert.Equal(new[] { 6, 6, 6 }, second.Sequence.OutputTokens);
            Assert.Equal(2, maxRunning); Assert.Equal(2, samples);
            Assert.Contains("b", model.ForwardOrder.TakeWhile(x => x != "released:a"));
            Assert.Equal(0, model.UnscopedBinds); Assert.Equal(0, model.JointGraphs);
        }
        Assert.Empty(model.Owners);
        Assert.Equal(128, budget.Snapshot().Single().Available);
    }

    [Fact]
    public async Task CancellingOneAdmittedHolderDoesNotReleaseItsLiveNeighbour()
    {
        var budget = new MemoryBudget([new("device", 128)]);
        using var model = new HolderModel { PauseOnSecondHolder = true };
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 48)])
        { UsePerSequenceHolders = true, EnterSerialExecution = model.Enter };
        using var engine = new InferenceEngine(model, HolderConfig(admission));
        using var cancellation = new CancellationTokenSource();
        var first = engine.SubmitRequest(new SequenceState("a", [1, 2, 3, 4], 100, 8, SamplingConfig.Greedy), cancellation.Token);
        Assert.True(model.FirstForward.Wait(TimeSpan.FromSeconds(3)));
        var second = engine.SubmitRequest(Request("b"));
        model.AllowFirstForward.Set();
        Assert.True(model.SecondForward.Wait(TimeSpan.FromSeconds(3)));
        cancellation.Cancel(); model.AllowSecondForward.Set();
        await Task.WhenAll(first.Completion, second.Completion).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(SequenceStatus.FinishedAborted, first.Sequence.Status);
        Assert.Equal(new[] { 6, 6, 6 }, second.Sequence.OutputTokens);
        Assert.Contains("released:a", model.ForwardOrder);
        Assert.Contains("released:b", model.ForwardOrder);
        engine.Dispose(); Assert.Equal(128, budget.Snapshot().Single().Available);
    }

    [Fact]
    public async Task PhysicalContentionWaitsForLiveHolderReleaseWithoutPerTokenSampling()
    {
        var budget = new MemoryBudget([new("device", 128)]);
        using var model = new HolderModel();
        int samples = 0;
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 48)])
        {
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { samples++; return [new("device", 60)]; },
            EnterSerialExecution = model.Enter
        };
        using var engine = new InferenceEngine(model, HolderConfig(admission));
        var first = engine.SubmitRequest(new SequenceState("a", [1, 2, 3, 4], 12, 8, SamplingConfig.Greedy));
        Assert.True(model.FirstForward.Wait(TimeSpan.FromSeconds(3)));
        var second = engine.SubmitRequest(Request("b")); model.AllowFirstForward.Set();
        await Task.WhenAll(first.Completion, second.Completion).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(12, first.Sequence.OutputTokens.Count);
        Assert.Equal(new[] { 6, 6, 6 }, second.Sequence.OutputTokens);
        Assert.Equal(3, samples); // first, one refused neighbour, then release
        Assert.True(model.ForwardOrder.IndexOf("released:a") < model.ForwardOrder.IndexOf("b"));
        engine.Dispose(); Assert.Equal(128, budget.Snapshot().Single().Available);
    }

    private sealed class HolderModel : IModelArchitecture, IBatchedPagedModel
    {
        public object GpuComputeLock { get; } = new();
        public ModelConfig Config { get; } = new() { VocabSize = 8 };
        public ITokenizer Tokenizer { get; } = new Tokens();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVStateSnapshot => false;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsPerSequenceFusedForward => true;
        public bool BatchedForwardAvailable => false;
        public Dictionary<string, BudgetReservation> Owners { get; } = new();
        public List<string> ForwardOrder { get; } = new();
        public ManualResetEventSlim FirstForward { get; } = new();
        public ManualResetEventSlim AllowFirstForward { get; } = new();
        public ManualResetEventSlim SecondForward { get; } = new();
        public ManualResetEventSlim AllowSecondForward { get; } = new();
        public bool PauseOnSecondHolder;
        public int UnscopedBinds, JointGraphs;
        public bool EnableJoint;
        public bool PauseOnTripleJoint;
        public ManualResetEventSlim TripleJointForward { get; } = new();
        public ManualResetEventSlim AllowTripleJointForward { get; } = new();
        public List<int> JointWidths { get; } = new();
        public string? RefusePrepare;
        private BudgetReservation? _jointEnvelope, _jointAllocation;
        private SequenceState? _scope;
        private string? _bound;
        private int _forwards;
        public IDisposable Enter(SequenceState sequence)
        {
            Assert.Null(_scope); Assert.NotNull(sequence.MemoryEnvelope);
            _scope = sequence; return new Scope(this);
        }
        private sealed class Scope(HolderModel model) : IDisposable
        { public void Dispose() => model._scope = null; }
        public bool BindSequenceCache(string id)
        {
            if (_scope?.RequestId != id) { UnscopedBinds++; throw new InvalidOperationException("Unscoped holder binding"); }
            _bound = id;
            if (Owners.ContainsKey(id)) return false;
            var allocation = _scope.MemoryEnvelope!.TryTake([new("device", 32)])
                ?? throw new MemoryPressureException("Missing holder allocation credit");
            allocation.Commit(); Owners.Add(id, allocation); return true;
        }
        public bool HasFusedSequenceCache(string id) => Owners.ContainsKey(id);
        public bool CanBatchDecode(string id, int position) => Owners.ContainsKey(id);
        public bool PrepareBatchedFusedDecode(string id)
        {
            Assert.Equal(id, _scope?.RequestId);
            if (id == RefusePrepare) throw new MemoryPressureException("Injected holder preparation refusal");
            return Owners.ContainsKey(id);
        }
        public IDisposable EnterJoint(BudgetReservation envelope)
        {
            Assert.Null(_scope); Assert.Null(_jointEnvelope); _jointEnvelope = envelope;
            return new JointScope(this);
        }
        private sealed class JointScope(HolderModel model) : IDisposable
        { public void Dispose() => model._jointEnvelope = null; }
        public bool TryForwardBatchedFusedDecodeSampled(IReadOnlyList<string> ids, int[] tokens, int[] positions, int[] next)
        {
            if (!EnableJoint) return false;
            Assert.Null(_scope); Assert.NotNull(_jointEnvelope); Assert.InRange(ids.Count, 2, 8);
            JointWidths.Add(ids.Count);
            if (ids.Count == 3 && !TripleJointForward.IsSet)
            {
                TripleJointForward.Set();
                if (PauseOnTripleJoint && !AllowTripleJointForward.Wait(TimeSpan.FromSeconds(4)))
                    throw new TimeoutException("Triple joint barrier");
            }
            _jointAllocation ??= _jointEnvelope!.TryTake([new("device", 48)])
                ?? throw new MemoryPressureException("Missing shared graph reservation");
            if (JointGraphs == 0) _jointAllocation.Commit();
            JointGraphs++;
            for (int i = 0; i < ids.Count; ++i)
            { Assert.True(Owners.ContainsKey(ids[i])); next[i] = ExpectedToken(ids[i]); }
            return true;
        }
        public float[] Forward(int[] tokens)
        {
            Assert.Equal(_scope?.RequestId, _bound); Assert.NotNull(_bound);
            if (_forwards++ == 0)
            {
                FirstForward.Set();
                if (!AllowFirstForward.Wait(TimeSpan.FromSeconds(4))) throw new TimeoutException("First holder barrier");
            }
            if (_bound == "b" && !SecondForward.IsSet)
            {
                SecondForward.Set();
                if (PauseOnSecondHolder && !AllowSecondForward.Wait(TimeSpan.FromSeconds(4)))
                    throw new TimeoutException("Second holder barrier");
            }
            ForwardOrder.Add(_bound!);
            var logits = new float[8]; logits[ExpectedToken(_bound!)] = 10; return logits;
        }
        private static int ExpectedToken(string id) => id == "a" ? 5 : id == "b" ? 6 : 7;
        public void OnSequenceReleased(string id)
        {
            if (Owners.Remove(id, out var owner)) { owner.Dispose(); ForwardOrder.Add("released:" + id); }
        }
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context)
        { JointGraphs++; throw new InvalidOperationException("No joint-envelope graph authorized"); }
        public void ResetKVCache()
        {
            // Engine teardown resets the now-idle primary. A reset while any
            // admitted holder is live would erase another request's state.
            Assert.Empty(Owners);
        }
        public void TruncateKVCache(int tokens) => throw new InvalidOperationException("Cannot rewind hybrid state");
        public void Dispose()
        {
            AllowFirstForward.Set(); AllowSecondForward.Set(); AllowTripleJointForward.Set();
            foreach (var owner in Owners.Values) owner.Dispose(); Owners.Clear();
            _jointAllocation?.Dispose(); _jointAllocation = null;
        }
        public long ComputeKVBlockByteSize(int count) => 0;
        public bool TryExtractKVBlock(int start, int count, Span<byte> target) => false;
        public bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => false;
    }

    [Fact]
    public async Task ThreeLargeAllowanceRequestsJoinOneDecodeCallAndCancellationPreservesBothNeighbours()
    {
        var budget = new MemoryBudget([new("device", 256)]);
        using var joint = budget.Reserve([new("device", 48)]);
        using var model = new HolderModel { EnableJoint = true, PauseOnTripleJoint = true };
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 48)])
        {
            UsePerSequenceHolders = true, BeforeAdmissionWithRunning = (_, _) => [new("device", 256)],
            EnterSerialExecution = model.Enter, EnterBatchedExecution = _ => model.EnterJoint(joint)
        };
        using var engine = new InferenceEngine(model, HolderConfig(admission, joint: true, lanes: 3, blocks: 8192));
        using var cancellation = new CancellationTokenSource();
        var first = engine.SubmitRequest(new SequenceState("a", [1, 2, 3, 4], 20000, 8, SamplingConfig.Greedy), cancellation.Token);
        Assert.True(model.FirstForward.Wait(TimeSpan.FromSeconds(3)));
        var second = engine.SubmitRequest(new SequenceState("b", [1, 2, 3, 4], 12, 8, SamplingConfig.Greedy));
        var third = engine.SubmitRequest(new SequenceState("c", [1, 2, 3, 4], 12, 8, SamplingConfig.Greedy));
        model.AllowFirstForward.Set();
        Assert.True(model.TripleJointForward.Wait(TimeSpan.FromSeconds(3)));
        cancellation.Cancel(); model.AllowTripleJointForward.Set();
        await Task.WhenAll(first.Completion, second.Completion, third.Completion).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(SequenceStatus.FinishedAborted, first.Sequence.Status);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, second.Sequence.Status);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, third.Sequence.Status);
        Assert.Contains(3, model.JointWidths);
        Assert.Contains(2, model.JointWidths);
        Assert.Equal(12, second.Sequence.OutputTokens.Count); Assert.Equal(12, third.Sequence.OutputTokens.Count);
        Assert.All(first.Sequence.OutputTokens, token => Assert.Equal(5, token));
        Assert.All(second.Sequence.OutputTokens, token => Assert.Equal(6, token));
        Assert.All(third.Sequence.OutputTokens, token => Assert.Equal(7, token));
        Assert.Empty(model.Owners);
        engine.Dispose(); model.Dispose(); joint.Dispose();
        Assert.Equal(256, budget.Snapshot().Single().Available);
    }

    [Fact]
    public async Task PrefillCreditShrinksBeforeSecondAdmissionAndJointGraphUsesSharedOwner()
    {
        var budget = new MemoryBudget([new("device", 192)]);
        using var joint = budget.Reserve([new("device", 0)]);
        using var model = new HolderModel { EnableJoint = true };
        int completedPrefills = 0, withNeighbour = 0;
        var admission = new RequestMemoryAdmission(budget, seq => [new("device", seq.NumComputedTokens >= seq.PromptTokens.Count ? 40 : 96)])
        {
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmissionWithRunning = (_, running) =>
            {
                if (running.Count > 0)
                {
                    Assert.True(completedPrefills > 0); withNeighbour++;
                    Assert.True(joint.TryResizeTotal([new("device", 48)]));
                }
                return [new("device", 192)];
            },
            AfterPrefill = seq => { Assert.True(seq.MemoryEnvelope!.TryResizeTotal([new("device", 40)])); completedPrefills++; },
            EnterSerialExecution = model.Enter,
            EnterBatchedExecution = _ => model.EnterJoint(joint)
        };
        using var engine = new InferenceEngine(model, HolderConfig(admission, joint: true));
        var first = engine.SubmitRequest(new SequenceState("a", [1, 2, 3, 4], 8, 8, SamplingConfig.Greedy));
        Assert.True(model.FirstForward.Wait(TimeSpan.FromSeconds(3)));
        var second = engine.SubmitRequest(new SequenceState("b", [1, 2, 3, 4], 8, 8, SamplingConfig.Greedy));
        model.AllowFirstForward.Set();
        await Task.WhenAll(first.Completion, second.Completion).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(2, completedPrefills); Assert.Equal(1, withNeighbour);
        Assert.True(model.JointGraphs >= 2);
        Assert.All(first.Sequence.OutputTokens, token => Assert.Equal(5, token));
        Assert.All(second.Sequence.OutputTokens, token => Assert.Equal(6, token));
        Assert.Equal(8, first.Sequence.OutputTokens.Count); Assert.Equal(8, second.Sequence.OutputTokens.Count);
        Assert.Equal(48, budget.Snapshot().Single().Committed);
        Assert.Empty(model.Owners);
        engine.Dispose(); model.Dispose(); joint.Dispose();
        Assert.Equal(192, budget.Snapshot().Single().Available);
    }

    [Fact]
    public async Task HolderPreparationFailureDoesNotFailAnUnsubmittedBatchNeighbour()
    {
        var budget = new MemoryBudget([new("device", 192)]);
        using var joint = budget.Reserve([new("device", 48)]);
        using var model = new HolderModel { EnableJoint = true, RefusePrepare = "b" };
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 48)])
        {
            UsePerSequenceHolders = true, BeforeAdmission = _ => [new("device", 192)],
            EnterSerialExecution = model.Enter, EnterBatchedExecution = _ => model.EnterJoint(joint)
        };
        using var engine = new InferenceEngine(model, HolderConfig(admission, joint: true));
        var first = engine.SubmitRequest(new SequenceState("a", [1, 2, 3, 4], 8, 8, SamplingConfig.Greedy));
        Assert.True(model.FirstForward.Wait(TimeSpan.FromSeconds(3)));
        var second = engine.SubmitRequest(Request("b")); model.AllowFirstForward.Set();
        await first.Completion.WaitAsync(TimeSpan.FromSeconds(5));
        await Assert.ThrowsAsync<MemoryPressureException>(
            () => second.Completion.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, first.Sequence.Status);
        Assert.Equal(SequenceStatus.FinishedError, second.Sequence.Status);
        Assert.IsType<MemoryPressureException>(second.Sequence.Error);
        Assert.Equal(8, first.Sequence.OutputTokens.Count); Assert.Equal(0, model.JointGraphs);
        engine.Dispose(); model.Dispose(); joint.Dispose();
        Assert.Equal(192, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void InitiallyOversizedPeakCanShrinkAtPhysicalBoundaryBeforeAdmission()
    {
        var budget = new MemoryBudget([new("device", 100)]); int samples = 0;
        var cfg = Config(new(budget, _ => [new("device", samples == 0 ? 110 : 80)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { samples++; return [new("device", 100)]; },
            EnterSerialExecution = _ => new NoOpScope()
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("shrink-at-boundary");
        scheduler.Submit(seq);
        Assert.Equal(0, samples); Assert.Null(seq.MemoryEnvelope);
        Assert.Same(seq, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        Assert.Equal(1, samples); Assert.Equal(80, Assert.Single(seq.MemoryEnvelope!.Charges).Bytes);
        scheduler.Abort(seq.RequestId); scheduler.NotifyMemoryReleased(seq.RequestId);
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void InitiallyOversizedPeakStillFailsWhenRefreshCannotMakeItFit()
    {
        var budget = new MemoryBudget([new("device", 100)]); int samples = 0;
        var cfg = Config(new(budget, _ => [new("device", 110)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { samples++; return [new("device", 100)]; },
            EnterSerialExecution = _ => throw new InvalidOperationException("Must not execute")
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("too-large-even-after-boundary");
        scheduler.Submit(seq);
        Assert.Empty(scheduler.Schedule().ScheduledWork);
        Assert.Equal(1, samples); Assert.Equal(SequenceStatus.FinishedError, seq.Status);
        Assert.IsType<MemoryPressureException>(seq.Error); Assert.Null(seq.MemoryEnvelope);
        Assert.Equal(0, scheduler.WaitingCount); Assert.Equal(0, scheduler.RunningCount);
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void OrdinaryAdmissionStillRejectsOversizedPeakAtSubmission()
    {
        var budget = new MemoryBudget([new("device", 100)]); int samples = 0;
        var cfg = Config(new(budget, _ => [new("device", 110)])
        {
            BeforeAdmission = _ => { samples++; return [new("device", 100)]; },
            EnterSerialExecution = _ => new NoOpScope()
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("ordinary-too-large");
        Assert.Throws<MemoryPressureException>(() => scheduler.Submit(seq));
        Assert.Equal(0, samples); Assert.Null(seq.MemoryEnvelope);
        Assert.Equal(0, scheduler.WaitingCount); Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void QueuedFollowerReestimatesAfterPriorRequestMaterializesRetainedAllocation()
    {
        var budget = new MemoryBudget([new("device", 80)]);
        bool retained = false;
        int samples = 0, estimates = 0;
        var cfg = Config(new(budget, _ => { estimates++; return [new("device", retained ? 20 : 80)]; })
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { samples++; return [new("device", 80)]; },
            EnterSerialExecution = _ => new NoOpScope()
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var first = Request("first"); var next = Request("queued-before-materialization");
        scheduler.Submit(first); scheduler.Submit(next);
        Assert.Same(first, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        using var materialized = first.MemoryEnvelope!.TryTake([new("device", 60)])!;
        materialized.Commit(); retained = true;
        Assert.True(scheduler.Abort(first.RequestId)); scheduler.NotifyMemoryReleased(first.RequestId);
        Assert.Equal(60, budget.Snapshot().Single().Committed);
        Assert.Same(next, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        Assert.Equal(20, Assert.Single(next.MemoryEnvelope!.Charges).Bytes);
        Assert.Equal(2, samples); Assert.Equal(4, estimates);
        scheduler.Abort(next.RequestId); scheduler.NotifyMemoryReleased(next.RequestId);
        materialized.Dispose();
        Assert.Equal(80, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void ReestimateRunsAfterPhysicalTrimAndCanIncreaseTheIncrementalReservation()
    {
        var budget = new MemoryBudget([new("device", 80)]);
        using var retained = budget.Reserve([new("device", 60)]); retained.Commit();
        bool trimmed = false;
        var cfg = Config(new(budget, _ => [new("device", trimmed ? 80 : 20)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { retained.Dispose(); trimmed = true; return [new("device", 80)]; },
            EnterSerialExecution = _ => new NoOpScope()
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("trimmed");
        scheduler.Submit(seq);
        Assert.Single(scheduler.Schedule().ScheduledWork);
        Assert.Equal(80, Assert.Single(seq.MemoryEnvelope!.Charges).Bytes);
        scheduler.Abort(seq.RequestId); scheduler.NotifyMemoryReleased(seq.RequestId);
        Assert.Equal(80, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void RefreshedImpossiblePeakFailsBeforeExecutionWithoutLeakingReservation()
    {
        var budget = new MemoryBudget([new("device", 80)]); bool observed = false;
        var cfg = Config(new(budget, _ => [new("device", observed ? 81 : 20)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { observed = true; return [new("device", 80)]; },
            EnterSerialExecution = _ => throw new InvalidOperationException("Must not execute")
        });
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var seq = Request("too-large-after-refresh");
        scheduler.Submit(seq);
        Assert.Empty(scheduler.Schedule().ScheduledWork);
        Assert.Equal(SequenceStatus.FinishedError, seq.Status);
        Assert.IsType<MemoryPressureException>(seq.Error);
        Assert.Null(seq.MemoryEnvelope); Assert.Equal(80, budget.Snapshot().Single().Available);
    }

    [Theory]
    [InlineData(false, true, 1)]
    [InlineData(true, false, 1)]
    [InlineData(true, true, 2)]
    public void IncrementalRefreshRefusesMissingBoundaryExecutionOrMultipleLanes(bool sample, bool execute, int lanes)
    {
        var budget = new MemoryBudget([new("device", 80)]);
        var cfg = Config(new(budget, _ => [new("device", 20)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = sample ? _ => [new("device", 80)] : null,
            EnterSerialExecution = execute ? _ => new NoOpScope() : null
        }, lanes);
        var pool = new BlockPool(32, 8, 32); using var storage = pool.Storage;
        Assert.Throws<ArgumentException>(() => new ContinuousBatchScheduler(cfg, pool));
    }

    [Fact]
    public void SchedulerCopiesPreserveTheExplicitImmutableExecutionPolicy()
    {
        using var env = new EnvScope(); env.Set("TS_SCHED_DISABLE_BATCHED", "0");
        var cfg = Config(new(new MemoryBudget([new("device", 80)]), _ => []));
        Assert.Same(Serial, cfg.ResolveExecutionOptions());
        Assert.Same(Serial, cfg.WithBlockSize(16).ExecutionOptionsOverride);
        Assert.Same(Serial, cfg.WithSpeculation(SpeculationOptions.Disabled).ExecutionOptionsOverride);
        Assert.False(new SchedulerConfig().ResolveExecutionOptions().BatchedPathDisabled);
        Assert.Equal("0", Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED"));
    }

    [Fact]
    public void CapabilityReportDistinguishesAdmittedEnginePolicyFromEnvironmentAndModelFusion()
    {
        var cfg = Config(new(new MemoryBudget([new("device", 80)]), _ => [])
        { EnterSerialExecution = _ => new NoOpScope() });
        var caps = new ExecutionCapabilities
        {
            SupportsBatchedPagedAttention = true, SupportsLinearKvMigration = false,
            SupportsPerSequenceFusedForward = true, SupportsRetainedFusedCache = true
        };
        string report = ExecutionPlanner.BuildCapabilityReport(caps, cfg.ResolveExecutionOptions(), cfg);
        Assert.Contains("active per-engine policy: SchedulerConfig.ExecutionOptionsOverride", report);
        Assert.Contains("does not set environment variables", report);
        Assert.DoesNotContain("active TS_* overrides:", report);
        Assert.Contains("N=1 single-sequence fused fast path: disabled (per-engine policy selects the serial model route)", report);
        Assert.Contains("admitted serial execution: model.Forward", report);
        Assert.Contains("model-owned fused kernels remain available where supported by the model/backend", report);
    }

    [Fact]
    public void OrdinaryCapabilityReportPreservesEnvironmentOverrideWording()
    {
        var caps = new ExecutionCapabilities
        {
            SupportsBatchedPagedAttention = true, SupportsPerSequenceFusedForward = true,
            SupportsRetainedFusedCache = true
        };
        string report = ExecutionPlanner.BuildCapabilityReport(caps, Serial, new SchedulerConfig());
        Assert.Contains("batched paged attention: disabled (TS_SCHED_DISABLE_BATCHED)", report);
        Assert.Contains("per-sequence fused concurrent decode: disabled (TS_PER_SEQ_FUSED=0)", report);
        Assert.Contains("retained fused-cache continuation: off (TS_RETAINED_FUSED_CACHE_MAX=0)", report);
        Assert.Contains("active TS_* overrides: " + Serial.DescribeOverrides(), report);
        Assert.DoesNotContain("per-engine policy", report);
        Assert.DoesNotContain("admitted serial execution", report);
    }

    [Fact]
    public void EvenDefaultValuedEnginePolicyIsNamedWithoutClaimingFusedKernelSupport()
    {
        var cfg = new SchedulerConfig { ExecutionOptionsOverride = ExecutionOptions.Default };
        string report = ExecutionPlanner.BuildCapabilityReport(new ExecutionCapabilities(), cfg.ResolveExecutionOptions(), cfg);
        Assert.Contains("active per-engine policy: SchedulerConfig.ExecutionOptionsOverride; default execution settings", report);
        Assert.DoesNotContain("active TS_* overrides", report);
        Assert.DoesNotContain("model-owned fused kernels", report);
    }

    [Fact]
    public async Task ActualExecutorUsesSerialOverrideAndSamplesOnceForEachRequestNotEachToken()
    {
        using var env = new EnvScope(); env.Set("TS_SCHED_DISABLE_BATCHED", "0");
        var budget = new MemoryBudget([new("device", 80)]); int samples = 0, scopes = 0;
        using var model = new RoutedModel();
        using var engine = new InferenceEngine(model, Config(new(budget, _ => [new("device", 20)])
        {
            ReestimateAfterPhysicalCheck = true,
            BeforeAdmission = _ => { Assert.True(Monitor.IsEntered(model.GpuComputeLock)); samples++; return [new("device", 80)]; },
            EnterSerialExecution = seq => { Assert.NotNull(seq.MemoryEnvelope); scopes++; return new NoOpScope(); }
        }));
        var handles = new[] { engine.SubmitRequest(Request("a")), engine.SubmitRequest(Request("b")) };
        await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.All(handles, h => Assert.Equal(new[] { 5, 5, 5 }, h.Sequence.OutputTokens));
        Assert.Equal(2, samples); Assert.True(scopes >= 6); Assert.True(model.Forwards >= 6);
        Assert.Equal(0, model.BatchedForwards);
        Assert.Throws<NotSupportedException>(() => engine.UpdateSpeculation(new SpeculationOptions { Enabled = true }));
        engine.Dispose(); Assert.Equal(80, budget.Snapshot().Single().Available);
    }

    private sealed class RoutedModel : IModelArchitecture, IBatchedPagedModel
    {
        public object GpuComputeLock { get; } = new();
        public ModelConfig Config { get; } = new() { VocabSize = 8 };
        public ITokenizer Tokenizer { get; } = new Tokens();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public bool SupportsLinearKVMigration => true;
        public int Forwards, BatchedForwards;
        public float[] Forward(int[] tokens) { Forwards++; return [0, 0, 0, 0, 0, 10, 0, 0]; }
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context)
        { BatchedForwards++; throw new InvalidOperationException("The admitted serial route must not call ForwardBatch."); }
        public void ResetKVCache() { }
        public void TruncateKVCache(int tokens) { }
        public void Dispose() { }
        public long ComputeKVBlockByteSize(int count) => count * 4L;
        public bool TryExtractKVBlock(int start, int count, Span<byte> target) { target.Clear(); return true; }
        public bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;
    }
    private sealed class Tokens : ITokenizer
    {
        public string[] Vocab { get; } = Enumerable.Range(0, 8).Select(x => x.ToString()).ToArray();
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => 8;
        public List<int> Encode(string text, bool addSpecial = true) => [];
        public string Decode(List<int> ids) => string.Join(",", ids);
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.Add((byte)('0' + tokenId));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string token) => -1;
    }
}
