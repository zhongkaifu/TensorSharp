// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryPrefillAdmissionTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool, Device = UnifiedMemoryModelSession.DevicePool;
    private static MemoryCharge[] Charges(long host, long device = 0) => [new(Host, host), new(Device, device)];

    [Theory]
    [InlineData(80, 10, 0)] // No growth.
    [InlineData(70, 10, 0)] // Shrink, still above fresh physical limit.
    [InlineData(10, 10, 70)] // A target cannot revoke its live child floor.
    [InlineData(80, 11, 0)] // Growth in Device must also check Host.
    public void StrictAdmissionRefusesEveryUnfitPostLedgerWithoutMutation(long target, long device, long childBytes)
    {
        var budget = new MemoryBudget(Charges(100, 100));
        using var request = budget.Reserve(Charges(80, 10));
        using var joint = budget.Reserve(Charges(10));
        using var child = childBytes > 0 ? request.TryTake(Charges(childBytes)) : null;
        child?.Commit();
        var before = budget.Snapshot().ToArray(); var owners = request.OwnedCharges.ToArray();
        var hwm = budget.HighWatermarks().ToArray(); var signal = budget.ChangeSignal;
        Assert.False(budget.TryResizeTotals([(request, Charges(target, device)), (joint, Charges(10))],
            Charges(50, 100), requirePhysicalFit: true));
        Assert.Equal(before, budget.Snapshot()); Assert.Equal(owners, request.OwnedCharges);
        Assert.Equal(hwm, budget.HighWatermarks()); Assert.False(signal.IsCompleted);
        Assert.Same(signal, budget.ChangeSignal);
    }

    [Fact]
    public void StrictGatePreservesDefaultCleanupAndChecksUntouchedPoolAndExactThreshold()
    {
        var budget = new MemoryBudget(Charges(100, 100));
        using var request = budget.Reserve(Charges(80));
        using var other = budget.Reserve(Charges(0, 60));
        var before = budget.Snapshot().ToArray();
        Assert.False(budget.TryResizeTotals([(request, Charges(80))], Charges(80, 59), true));
        Assert.Equal(before, budget.Snapshot());
        Assert.True(budget.TryResizeTotals([(request, Charges(80))], Charges(80, 60), true));
        Assert.True(budget.TryResizeTotals([(request, Charges(70))], Charges(0, 0)));
        Assert.Equal(70, request.OwnedCharges.Single(c => c.Pool == Host).Bytes);
        Assert.Throws<ArgumentNullException>(() => budget.TryResizeTotals([(request, Charges(70))], null, true));
        Assert.Throws<ArgumentException>(() => budget.TryResizeTotals([(request, Charges(70))], [], true));
    }

    [Theory]
    [InlineData(512)]
    [InlineData(340)]
    [InlineData(257)]
    [InlineData(32)]
    [InlineData(1)]
    public void CandidatePassIsStrictlyBoundedAndIncludesMaximumAndOne(int maximum)
    {
        int[] rows = UnifiedMemoryPrefillAdmission.Candidates(maximum).ToArray();
        Assert.Equal(maximum, rows[0]); Assert.Equal(1, rows[^1]);
        Assert.InRange(rows.Length, 1, 10); Assert.Equal(rows.Length, rows.Distinct().Count());
        Assert.All(rows, n => Assert.InRange(n, 1, maximum));
        Assert.True(rows.Zip(rows.Skip(1)).All(pair => pair.First > pair.Second));
    }

    [Fact]
    public void SmallerCandidateFitsWithoutTouchingWarmExpertsOrSiblingOwners()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var warm = budget.Reserve(Charges(200, 300)); warm.Commit();
        using var sibling = budget.Reserve(Charges(100, 100));
        using var request = budget.Reserve(Charges(50, 50));
        using var joint = budget.Reserve(Charges(100, 100));
        var seen = new List<int>();
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelect(budget, request, joint, new(100, 100), 340,
            n => { seen.Add(n); return Charges(100 + n, 100); }, Charges(650, 1000), out var selected));
        Assert.Equal(128, selected.Rows); Assert.Equal(new[] { 340, 256, 128 }, seen);
        Assert.Equal(200, warm.OwnedCharges.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(300, warm.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(100, sibling.OwnedCharges.Single(c => c.Pool == Host).Bytes);
    }

    [Fact]
    public void MaximumFitUsesOneAttemptAndNonMonotonicShapesAreEvaluatedIndependently()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var request = budget.Reserve(Charges(10)); using var joint = budget.Reserve(Charges(10));
        int attempts = 0;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelect(budget, request, joint, new(10, 0), 340,
            n => { attempts++; return Charges(300); }, Charges(1000, 1000), out var full));
        Assert.Equal(340, full.Rows); Assert.Equal(1, attempts);
        var seen = new List<int>();
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelect(budget, request, joint, new(10, 0), 340,
            n => { seen.Add(n); return Charges(n == 128 ? 200 : 1000 + n); }, Charges(1000, 1000), out var narrow));
        Assert.Equal(128, narrow.Rows); Assert.Equal(new[] { 340, 256, 128 }, seen);
    }

    [Fact]
    public void EveryCandidateRefusalPreservesChildrenOtherSlotsJointAndHighWatermarks()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var request = budget.Reserve(Charges(200, 200));
        using var child = request.TryTake(Charges(180, 180))!; child.Commit();
        using var joint = budget.Reserve(Charges(300, 300));
        using var arena = joint.TryTake(Charges(280, 280))!; arena.Commit();
        using var otherSlot = budget.Reserve(Charges(200, 200)); otherSlot.Commit();
        var before = budget.Snapshot().ToArray(); var hwm = budget.HighWatermarks().ToArray();
        var signal = budget.ChangeSignal; int attempts = 0;
        Assert.False(UnifiedMemoryPrefillAdmission.TrySelect(budget, request, joint, new(300, 300), 512,
            _ => { attempts++; return Charges(1, 1); }, Charges(650, 650), out var selected));
        Assert.Equal(default, selected); Assert.InRange(attempts, 1, 10);
        Assert.Equal(before, budget.Snapshot()); Assert.Equal(hwm, budget.HighWatermarks());
        Assert.Equal(180, child.OwnedCharges.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(280, arena.OwnedCharges.Single(c => c.Pool == Host).Bytes);
        Assert.False(signal.IsCompleted); Assert.Same(signal, budget.ChangeSignal);
    }

    [Fact]
    public void ACompetingOwnerAfterObservationCannotSpendTheSamePhysicalCredit()
    {
        var budget = new MemoryBudget(Charges(100));
        using var request = budget.Reserve(Charges(10)); using var joint = budget.Reserve(Charges(10));
        using var rival = budget.Reserve(Charges(50));
        var before = budget.Snapshot().ToArray();
        Assert.False(UnifiedMemoryPrefillAdmission.TrySelect(budget, request, joint, new(10, 0), 32,
            _ => Charges(50), Charges(100), out _));
        Assert.Equal(before, budget.Snapshot());
    }

    [Fact]
    public void ReenteringMaterializationCannotSilentlyChangeAnAlreadyAdmittedWidth()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var request = budget.Reserve(Charges(200)); using var joint = budget.Reserve(Charges(10));
        var before = budget.Snapshot().ToArray(); var seen = new List<int>();
        Assert.False(UnifiedMemoryPrefillAdmission.TrySelect(budget, request, joint, new(10, 0), 128,
            n => { seen.Add(n); return Charges(n == 128 ? 1100 : 100); }, Charges(1000, 1000), out _, allowDowngrade: false));
        Assert.Equal(new[] { 128 }, seen); Assert.Equal(before, budget.Snapshot());
    }
}

public sealed class RequestPrefillLimitExecutionTests
{
    private sealed class Scope(Action? onDispose = null) : IDisposable { public void Dispose() => onDispose?.Invoke(); }
    private static SchedulerConfig Config(RequestMemoryAdmission admission) => new()
    {
        BlockSize = 4, NumBlocks = 256, MaxNumRunningSequences = 3,
        MaxNumBatchedTokens = 64, MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 64,
        PrefillChunkTokenLimit = 64, EnablePrefixCaching = false, StopRepetition = false,
        ExecutionOptionsOverride = new() { BatchedPathDisabled = true, PerSeqFusedEnabled = true,
            BatchedFusedDecodeEnabled = false, KvHolderPoolMax = 0 }, MemoryAdmission = admission
    };

    [Fact]
    public void PostMaterializationLimitBoundsFreshRunningAndMixedWork()
    {
        var budget = new MemoryBudget([new("host", 100)]);
        var caps = new Dictionary<SequenceState, int>();
        var admission = new RequestMemoryAdmission(budget, _ => [new("host", 10)])
        {
            UsePerSequenceHolders = true, EnterSerialExecution = _ => new Scope(),
            AfterPrefixMaterialization = s => caps[s] = s.RequestId == "first" ? 3 : 2,
            PrefillTokenLimit = s => caps[s]
        };
        var config = Config(admission); var pool = new BlockPool(256, 4, 0); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(config, pool);
        var first = new SequenceState("first", Enumerable.Range(1, 6).ToArray(), 3, 4, SamplingConfig.Greedy);
        scheduler.Submit(first);
        var initial = Assert.Single(scheduler.Schedule().ScheduledWork);
        Assert.Equal(3, initial.NumScheduledTokens); first.AdvanceComputedTokens(3);
        var second = new SequenceState("second", Enumerable.Range(30, 20).ToArray(), 3, 4, SamplingConfig.Greedy);
        scheduler.Submit(second);
        var both = scheduler.Schedule().ScheduledWork;
        Assert.Equal(2, both.Count);
        Assert.Equal(3, both.Single(w => ReferenceEquals(w.Sequence, first)).NumScheduledTokens);
        Assert.Equal(2, both.Single(w => ReferenceEquals(w.Sequence, second)).NumScheduledTokens);
        first.AdvanceComputedTokens(3); second.AdvanceComputedTokens(2);
        var mixed = scheduler.Schedule().ScheduledWork;
        Assert.False(mixed.Single(w => ReferenceEquals(w.Sequence, first)).IsPrefill);
        Assert.Equal(2, mixed.Single(w => ReferenceEquals(w.Sequence, second)).NumScheduledTokens);
        foreach (var s in new[] { first, second }) { scheduler.Abort(s.RequestId); scheduler.NotifyMemoryReleased(s.RequestId); }
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public async Task RealEngineUsesCappedChunksAndPreservesOutputAcrossConcurrentRequests()
    {
        var budget = new MemoryBudget([new("host", 100)]);
        using var model = new RecordingModel(); int activeScopes = 0;
        var admission = new RequestMemoryAdmission(budget, _ => [new("host", 10)])
        {
            UsePerSequenceHolders = true,
            EnterSerialExecution = _ => { Assert.Equal(0, activeScopes++); return new Scope(() => activeScopes--); },
            PrefillTokenLimit = _ => 3,
            RevalidateExecution = _ => Assert.Equal(1, activeScopes)
        };
        using var engine = new InferenceEngine(model, Config(admission));
        var a = new SequenceState("a", [1, 2, 3, 4, 5, 6, 7], 3, 4, SamplingConfig.Greedy);
        var b = new SequenceState("b", [11, 12, 13, 14, 15, 16, 17], 3, 4, SamplingConfig.Greedy);
        await Task.WhenAll(engine.SubmitRequest(a).Completion, engine.SubmitRequest(b).Completion).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.All(model.ForwardRows, n => Assert.InRange(n, 1, 3));
        Assert.Equal(new[] { 8, 9, 10 }, a.OutputTokens); Assert.Equal(new[] { 18, 19, 20 }, b.OutputTokens);
        Assert.Equal(0, activeScopes); Assert.Equal(0, engine.WaitingCount); Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public async Task GlmAdaptiveSelectionDrivesRealChunksAndLatePressureRefusesImmutableWorkBeforeForward()
    {
        const long unit = 1L << 20;
        const string host = UnifiedMemoryModelSession.HostPool, device = UnifiedMemoryModelSession.DevicePool;
        MemoryCharge[] Charges(long h, long d) => [new(host, h * unit), new(device, d * unit)];
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var dense = budget.Reserve(Charges(0, 100));
        using var expert = budget.Reserve([new(device, 600 * unit)]);
        using var joint = budget.Reserve(Charges(0, 0));
        var caps = new Dictionary<SequenceState, int>();
        bool pressure = false;
        InferenceMemoryBytes Bytes(BudgetReservation owner) => new(
            owner.OwnedCharges.Single(c => c.Pool == host).Bytes,
            owner.OwnedCharges.Single(c => c.Pool == device).Bytes);
        var admission = new RequestMemoryAdmission(budget, _ => Charges(10, 10))
        {
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmissionWithRunning = (_, _) => Charges(1000, 1000),
            EnterSerialExecution = _ => new Scope(), PrefillTokenLimit = seq => caps[seq],
            AfterPrefixMaterialization = seq =>
            {
                long cache = expert.OwnedCharges.Single().Bytes;
                Assert.True(GlmUnifiedPrefillAdmission.TrySelect(budget.Snapshot(), Charges(1000, 1000),
                    Bytes(joint), Bytes(seq.MemoryEnvelope!), cache, 200 * unit,
                    Math.Min(64, seq.PromptTokens.Count - seq.NumComputedTokens),
                    rows => (new(20 * unit, rows * 10L * unit), new(10 * unit, 10 * unit)), out var selected));
                // This fixture's cache has only unused credit; fenced physical
                // release ordering is covered separately with native fault seams.
                Assert.True(expert.TryResizeTotal([new(device, selected.ExpertTargetBytes)]));
                Assert.True(budget.TryResizeTotals([(joint, new MemoryCharge[] {new(host, selected.JointTarget.Host), new(device, selected.JointTarget.Device)}),
                    (seq.MemoryEnvelope!, new MemoryCharge[] {new(host, selected.RequestTarget.Host), new(device, selected.RequestTarget.Device)})],
                    Charges(1000, 1000), requirePhysicalFit: true));
                caps[seq] = selected.Rows;
            },
            RevalidateExecution = seq =>
            {
                // The already-scheduled cap is immutable. A lower observation
                // cannot turn its work into a smaller, silently forwarded chunk.
                if (!budget.TryResizeTotals([(joint, joint.OwnedCharges),
                    (seq.MemoryEnvelope!, seq.MemoryEnvelope!.OwnedCharges)],
                    Charges(1000, pressure ? 900 : 1000), requirePhysicalFit: true))
                    throw new MemoryPressureException("late GLM physical pressure");
            },
            AfterSequenceMemoryRelease = running => { if (running.Count == 0) Assert.True(joint.TryResizeTotal(Charges(0, 0))); }
        };
        var config = Config(admission);
        var pool = new BlockPool(256, 4, 0); using var storage = pool.Storage;
        using var model = new RecordingModel();
        var scheduler = new ContinuousBatchScheduler(config, pool);
        var executor = new BatchExecutor(model, pool, scheduler);
        var denied = new SequenceState("glm-pressure", Enumerable.Range(1, 70).ToArray(), 2, 4, SamplingConfig.Greedy);
        scheduler.Submit(denied);
        var first = scheduler.Schedule();
        Assert.Equal(64, Assert.Single(first.ScheduledWork).NumScheduledTokens);
        Assert.Null(Assert.Single(executor.ExecuteStep(first)).Error);
        Assert.Equal(new[] {64}, model.ForwardRows);
        pressure = true;
        var immutable = scheduler.Schedule();
        Assert.Equal(6, Assert.Single(immutable.ScheduledWork).NumScheduledTokens);
        var refusal = Assert.Single(executor.ExecuteStep(immutable));
        Assert.IsType<MemoryPressureException>(refusal.Error);
        Assert.Equal(64, caps[denied]);
        Assert.Equal(64, denied.NumComputedTokens);
        Assert.Equal(new[] {64}, model.ForwardRows);
        scheduler.Abort(denied.RequestId); scheduler.NotifyMemoryReleased(denied.RequestId);
        Assert.True(joint.TryResizeTotal(Charges(0, 0)));
        pressure = false;
        using var healthyModel = new RecordingModel();
        using var engine = new InferenceEngine(healthyModel, config);
        var healthy = new SequenceState("glm-healthy", [101, 102], 2, 4, SamplingConfig.Greedy);
        await engine.SubmitRequest(healthy).Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(new[] {103, 104}, healthy.OutputTokens);
        Assert.Equal(0, engine.WaitingCount);
    }

    [Fact]
    public async Task WiderGlmAdmissionDrivesRealLargeChunksWhileKeepingTwoGenerationsAndFullSlotCredit()
    {
        const long unit = 1L << 20;
        const string host = UnifiedMemoryModelSession.HostPool, device = UnifiedMemoryModelSession.DevicePool;
        MemoryCharge[] Charges(long h, long d) => [new(host, h * unit), new(device, d * unit)];
        var budget = new MemoryBudget(Charges(4096, 2048));
        using var dense = budget.Reserve(Charges(0, 128)); dense.Commit();
        using var expert = budget.Reserve([new(device, 384 * unit)]);
        using var joint = budget.Reserve(Charges(0, 0));
        var caps = new Dictionary<SequenceState, int>();
        var admission = new RequestMemoryAdmission(budget, _ => Charges(32, 128))
        {
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmissionWithRunning = (_, _) => Charges(4096, 2048),
            EnterSerialExecution = _ => new Scope(), PrefillTokenLimit = seq => caps[seq],
            AfterPrefixMaterialization = seq =>
            {
                long cache = expert.OwnedCharges.Single().Bytes;
                Assert.True(GlmUnifiedPrefillAdmission.TrySelect(budget.Snapshot(), Charges(4096, 2048),
                    new(joint.OwnedCharges.Single(c => c.Pool == host).Bytes, joint.OwnedCharges.Single(c => c.Pool == device).Bytes),
                    new(seq.MemoryEnvelope!.OwnedCharges.Single(c => c.Pool == host).Bytes,
                        seq.MemoryEnvelope.OwnedCharges.Single(c => c.Pool == device).Bytes),
                    cache, 128 * unit, 1024,
                    // Both scratch generations and the complete new slot are
                    // explicitly owned. This fixture has unused expert credit;
                    // native physical trim/fence ordering is covered separately.
                    rows => (new(64 * unit, 2L * rows * unit), new(32 * unit, 128 * unit)), out var selected));
                Assert.Equal(832, selected.Rows);
                Assert.True(expert.TryResizeTotal([new(device, selected.ExpertTargetBytes)]));
                Assert.True(budget.TryResizeTotals([(joint, new MemoryCharge[] {new(host, selected.JointTarget.Host),
                        new(device, selected.JointTarget.Device)}),
                    (seq.MemoryEnvelope, new MemoryCharge[] {new(host, selected.RequestTarget.Host),
                        new(device, selected.RequestTarget.Device)})], Charges(4096, 2048), requirePhysicalFit: true));
                caps[seq] = selected.Rows;
            },
            AfterSequenceMemoryRelease = running =>
            { if (running.Count == 0) Assert.True(joint.TryResizeTotal(Charges(0, 0))); }
        };
        var config = new SchedulerConfig
        {
            BlockSize = 4, NumBlocks = 1024, MaxNumRunningSequences = 3,
            MaxNumBatchedTokens = 1024, MaxPrefillChunkSize = 1024, SoloPrefillChunkSize = 1024,
            PrefillChunkTokenLimit = 1024, EnablePrefixCaching = false, StopRepetition = false,
            ExecutionOptionsOverride = new() { BatchedPathDisabled = true, PerSeqFusedEnabled = true,
                BatchedFusedDecodeEnabled = false, KvHolderPoolMax = 0 }, MemoryAdmission = admission
        };
        using var model = new RecordingModel();
        using var engine = new InferenceEngine(model, config);
        var request = new SequenceState("wide-glm", Enumerable.Repeat(1, 1340).ToArray(), 2, 4, SamplingConfig.Greedy);
        await engine.SubmitRequest(request).Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(new[] { 832, 508 }, model.ForwardRows.Where(n => n > 1));
        Assert.Equal(new[] { 2, 3 }, request.OutputTokens);
        Assert.Equal(832, caps[request]);
        Assert.Equal(0, engine.WaitingCount);
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == device).Reserved
            - expert.OwnedCharges.Single().Bytes);
    }

    [Fact]
    public async Task MediaPromptSlicesCrossRequestCapsWithoutGapsOrCrossRequestReuse()
    {
        var budget = new MemoryBudget([new("host", 100)]); var media = new RecordingMedia();
        using var model = new RecordingModel { MultimodalInjector = media };
        var admission = new RequestMemoryAdmission(budget, _ => [new("host", 10)])
        { UsePerSequenceHolders = true, EnterSerialExecution = _ => new Scope(), PrefillTokenLimit = _ => 3 };
        using var engine = new InferenceEngine(model, Config(admission));
        var first = new SequenceState("image-a", [1, 2, 3, 4, 5, 6, 7], 2, 4, SamplingConfig.Greedy);
        var second = new SequenceState("image-b", [11, 12, 13, 14, 15, 16, 17], 2, 4, SamplingConfig.Greedy);
        await Task.WhenAll(engine.SubmitRequest(first).Completion, engine.SubmitRequest(second).Completion)
            .WaitAsync(TimeSpan.FromSeconds(5));
        foreach (string id in new[] { first.RequestId, second.RequestId })
        {
            var slices = media.Slices.Where(s => s.Id == id).ToArray();
            Assert.All(slices, s => Assert.InRange(s.Count, 1, 3));
            Assert.Equal(Enumerable.Range(0, 7), slices.SelectMany(s => Enumerable.Range(s.Start, s.Count)));
            // A prepared embedding crossing the3-row boundary is split once,
            // preserving its absolute rows and request ownership.
            Assert.Equal(new[] { 2, 3, 4, 5 }, slices.SelectMany(s => Enumerable.Range(s.Start, s.Count)).Where(n => n is >= 2 and < 6));
        }
        Assert.Equal(new[] { 8, 9 }, first.OutputTokens); Assert.Equal(new[] { 18, 19 }, second.OutputTokens);
    }

    [Fact]
    public void RevokedLivePrefillReplansMediaFromZeroWhileHealthySiblingContinues()
    {
        var budget = new MemoryBudget([new("host", 100)]); var media = new RecordingMedia();
        using var model = new RecordingModel { MultimodalInjector = media };
        var admission = new RequestMemoryAdmission(budget, _ => [new("host", 10)])
        { UsePerSequenceHolders = true, EnterSerialExecution = _ => new Scope(), PrefillTokenLimit = _ => 3 };
        var cfg = Config(admission); var pool = new BlockPool(256, 4, 0); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(cfg, pool); var executor = new BatchExecutor(model, pool, scheduler);
        var revoked = new SequenceState("revoked-media", Enumerable.Range(1, 15).ToArray(), 3, 4, SamplingConfig.Greedy)
            { UsesLiveCacheContinuation = true, PrefixCacheReusedTokens = 9 };
        foreach (var block in pool.AllocateNew(4)!) revoked.BlockTable.AppendBlock(block);
        revoked.AdvanceComputedTokens(9);
        var sibling = new SequenceState("healthy-media", [21, 22], 3, 4, SamplingConfig.Greedy);
        scheduler.Submit(revoked); scheduler.Submit(sibling);
        var planned = scheduler.Schedule();
        Assert.True(planned.ScheduledWork.Single(w => ReferenceEquals(w.Sequence, revoked)).IsPrefill);
        var first = executor.ExecuteStep(planned);
        Assert.Equal(0, first.Single(r => ReferenceEquals(r.Sequence, revoked)).TokensForwarded);
        Assert.Equal(2, first.Single(r => ReferenceEquals(r.Sequence, sibling)).TokensForwarded);
        Assert.Equal(0, revoked.NumComputedTokens); Assert.Equal(0, revoked.PrefixCacheReusedTokens);
        Assert.False(revoked.UsesLiveCacheContinuation); Assert.False(revoked.PrefillReservationTaken);
        Assert.DoesNotContain(media.Slices, s => s.Id == revoked.RequestId);
        Assert.Equal(new[] { 2 }, model.ForwardRows);
        scheduler.Abort(sibling.RequestId); scheduler.NotifyMemoryReleased(sibling.RequestId);
        int steps = 0;
        while (revoked.NumComputedTokens < revoked.PromptTokens.Count && steps++ < 10)
        {
            var actual = executor.ExecuteStep(scheduler.Schedule());
            Assert.All(actual, r => Assert.Null(r.Error));
        }
        Assert.Equal(15, revoked.NumComputedTokens);
        var slices = media.Slices.Where(s => s.Id == revoked.RequestId).ToArray();
        Assert.Equal(new[] { 0, 3, 6, 9, 12 }, slices.Select(s => s.Start));
        Assert.Equal(Enumerable.Range(0, 15), slices.SelectMany(s => Enumerable.Range(s.Start, s.Count)));
        scheduler.Abort(revoked.RequestId); scheduler.NotifyMemoryReleased(revoked.RequestId);
        Assert.Equal(100, budget.Snapshot().Single().Available);
    }

    [Fact]
    public void OversizedImmutableWorkFailsBeforeKvReservationAndBeforeForward()
    {
        var budget = new MemoryBudget([new("host", 100)]);
        var admission = new RequestMemoryAdmission(budget, _ => [new("host", 10)]) { PrefillTokenLimit = _ => 2 };
        var cfg = Config(admission); var pool = new BlockPool(256, 4, 0); using var storage = pool.Storage;
        using var model = new RecordingModel(); var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var executor = new BatchExecutor(model, pool, scheduler);
        var seq = new SequenceState("oversized", [1, 2, 3, 4], 3, 4, SamplingConfig.Greedy);
        var error = Assert.Throws<TargetInvocationException>(() => Private(executor, "BuildPrefillChunk", seq,
            new ScheduledSequenceWork(seq, 4, true, true)));
        Assert.IsType<MemoryPressureException>(error.InnerException);
        Assert.False(seq.PrefillReservationTaken); Assert.Equal(0, model.PrepareCalls); Assert.Empty(model.ForwardRows);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReplayReadmitsActualPhaseBeforeForwardAndNeverWidensGeneratedRows(bool refuse)
    {
        var budget = new MemoryBudget([new("host", 100)]); using var owner = budget.Reserve([new("host", 10)]);
        int checks = 0; bool executing = false;
        var admission = new RequestMemoryAdmission(budget, _ => [new("host", 10)])
        {
            UsePerSequenceHolders = true, EnterSerialExecution = _ => new Scope(), PrefillTokenLimit = _ => 3,
            RevalidateExecution = s =>
            {
                Assert.True(executing); checks++;
                if (refuse) throw new MemoryPressureException("replay actual peak refused");
                Assert.True(owner.TryResizeTotal([new("host", 30)]));
            }
        };
        var cfg = Config(admission); var pool = new BlockPool(256, 4, 0); using var storage = pool.Storage;
        using var model = new RecordingModel(); var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var executor = new BatchExecutor(model, pool, scheduler);
        var seq = new SequenceState("replay", [1, 2, 3, 4, 5, 6, 7], 4, 4, SamplingConfig.Greedy);
        seq.AppendOutputToken(8); seq.AppendOutputToken(9);
        foreach (var block in pool.AllocateNew(3)!) seq.BlockTable.AppendBlock(block);
        executing = true;
        if (refuse)
        {
            var error = Assert.Throws<TargetInvocationException>(() => Private(executor, "RecomputeLostTail", seq, 9));
            Assert.IsType<MemoryPressureException>(error.InnerException); Assert.Empty(model.ForwardRows);
            Assert.Equal(0, seq.NumComputedTokens); Assert.Equal(10, owner.OwnedCharges.Single().Bytes);
        }
        else
        {
            Private(executor, "RecomputeLostTail", seq, 9);
            Assert.Equal(new[] { 3, 3, 1, 1, 1 }, model.ForwardRows); Assert.Equal(9, seq.NumComputedTokens);
            Assert.Equal(30, owner.OwnedCharges.Single().Bytes);
        }
        Assert.Equal(refuse ? 1 : 5, checks);
        foreach (var block in seq.BlockTable.Blocks) pool.Free(block);
    }

    [Fact]
    public void LateReplayCapacityRefusalDoesNotForwardDeniedChunkOrFailHealthySibling()
    {
        var budget = new MemoryBudget([new("host", 100)]);
        using var replayOwner = budget.Reserve([new("host", 10)]);
        using var siblingOwner = budget.Reserve([new("host", 10)]);
        int activeScopes = 0; var observed = new List<int>();
        var admission = new RequestMemoryAdmission(budget, _ => [new("host", 10)])
        {
            UsePerSequenceHolders = true, PrefillTokenLimit = _ => 3,
            EnterSerialExecution = _ => { Assert.Equal(0, activeScopes++); return new Scope(() => activeScopes--); },
            RevalidateExecution = s =>
            {
                Assert.Equal(1, activeScopes);
                if (s.RequestId != "replay") return;
                observed.Add(s.NumComputedTokens);
                // A restored holder has only its small initial window. The
                // generated replay crosses the next capacity boundary at8.
                long peak = s.NumComputedTokens >= 8 ? 50 : 30;
                if (!budget.TryResizeTotals([(replayOwner, new MemoryCharge[] { new("host", peak) })],
                    [new("host", 40)], requirePhysicalFit: true))
                    throw new MemoryPressureException("restored holder capacity growth refused");
            }
        };
        var cfg = Config(admission); var pool = new BlockPool(256, 4, 0); using var storage = pool.Storage;
        using var model = new RecordingModel(); var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var executor = new BatchExecutor(model, pool, scheduler);
        var replay = new SequenceState("replay", [1, 2, 3, 4, 5, 6, 7], 4, 4, SamplingConfig.Greedy)
            { MemoryEnvelope = replayOwner };
        replay.AppendOutputToken(8); replay.AppendOutputToken(9);
        foreach (var block in pool.AllocateNew(3)!) replay.BlockTable.AppendBlock(block);
        replay.AdvanceComputedTokens(9); // A claimed prefix whose restore returns0.
        var sibling = new SequenceState("healthy", [21, 22], 3, 4, SamplingConfig.Greedy)
            { MemoryEnvelope = siblingOwner };
        foreach (var block in pool.AllocateNew(1)!) sibling.BlockTable.AppendBlock(block);
        var output = new SchedulerOutput();
        output.ScheduledWork.Add(new(replay, 1, true, false));
        output.ScheduledWork.Add(new(sibling, 2, true, true));
        var results = executor.ExecuteStep(output);
        Assert.IsType<MemoryPressureException>(results.Single(r => ReferenceEquals(r.Sequence, replay)).Error);
        Assert.Null(results.Single(r => ReferenceEquals(r.Sequence, sibling)).Error);
        Assert.Equal(new[] { 0, 3, 6, 7, 8 }, observed);
        Assert.Equal(new[] { 3, 3, 1, 1, 2 }, model.ForwardRows); // Denied replay row9 never executes.
        Assert.Equal(8, replay.NumComputedTokens); Assert.Equal(2, sibling.NumComputedTokens);
        Assert.Equal(30, replayOwner.OwnedCharges.Single().Bytes); Assert.Equal(10, siblingOwner.OwnedCharges.Single().Bytes);
        Assert.Equal(0, activeScopes);
        pool.Free(replay.BlockTable.Blocks); pool.Free(sibling.BlockTable.Blocks);
    }

    private static object? Private(BatchExecutor executor, string name, params object[] args)
        => typeof(BatchExecutor).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(executor, args);

    private sealed class RecordingModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly Dictionary<string, int> _holders = new(); private string? _active;
        public List<int> ForwardRows { get; } = new(); public int PrepareCalls { get; private set; }
        public ModelConfig Config { get; } = new() { VocabSize = 256 };
        public ITokenizer Tokenizer { get; } = new Tokens();
        public IMultimodalInjector MultimodalInjector { get; init; } = null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsPerSequenceFusedForward => true;
        public bool SupportsKVCacheTruncation => false;
        public bool BatchedForwardAvailable => false;
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context) => throw new NotSupportedException();
        public bool BindSequenceCache(string requestId) { bool fresh = _holders.TryAdd(requestId, 0); _active = requestId; return fresh; }
        public void ReleaseSequenceCache(string requestId) => _holders.Remove(requestId);
        public void PrepareForPrefill(int count) => PrepareCalls++;
        public float[] Forward(int[] tokens)
        {
            ForwardRows.Add(tokens.Length); var logits = new float[256]; logits[(tokens[^1] + 1) % 256] = 100;
            return logits;
        }
        public void ResetKVCache() { } public void TruncateKVCache(int n) { } public void Dispose() { }
        private sealed class Tokens : ITokenizer
        {
            public string[] Vocab { get; } = Enumerable.Range(0, 256).Select(n => n.ToString()).ToArray();
            public int BosTokenId => -1; public int[] EosTokenIds => []; public int VocabSize => 256;
            public List<int> Encode(string text, bool addSpecial = true) => [];
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int token, List<byte> bytes) { } public bool IsEos(int token) => false;
            public int LookupToken(string token) => -1;
        }
    }

    private sealed class RecordingMedia : IMultimodalInjector
    {
        public List<(string? Id, int Start, int Count)> Slices { get; } = new();
        public void LoadProjectors(string path) { }
        public List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> input, string? requestId = null) => input;
        public bool QueuePromptEmbeddings(int reused, string? requestId = null) => true;
        public bool QueuePromptEmbeddingsForSlice(int start, int count, string? requestId = null)
        { Slices.Add((requestId, start, count)); return true; }
        public int ClampReusablePrefix(int reused, string? requestId = null) => reused;
        public int ClampTrimStart(int start, string? requestId = null) => start;
        public void TrimPreparedPrompt(int start, string? requestId = null) { }
        public bool HasPendingEmbeddings(string requestId) => true;
        public void ClearPreparedPromptState(string requestId) { }
    }
}
