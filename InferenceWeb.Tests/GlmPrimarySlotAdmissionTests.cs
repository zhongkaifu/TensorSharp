// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>Real scheduler/envelope transactions with metadata-only native slot
/// stand-ins. No GPU allocation or model performance acceptance is implied.</summary>
public sealed class GlmPrimarySlotAdmissionTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool, Device = UnifiedMemoryModelSession.DevicePool;
    private const long Slot = 709_230_592;
    private static MemoryCharge[] Charges(long host, long device) => [new(Host, host), new(Device, device)];
    private static SequenceState Sequence(string id) => new(id, Enumerable.Repeat(1, 17).ToArray(), 4, 4, SamplingConfig.Greedy);
    private sealed class Scope : IDisposable { public void Dispose() { } }

    private sealed class Slots : IDisposable
    {
        internal readonly MemoryBudget Budget = new(Charges(1L << 30, 4 * Slot));
        internal readonly GlmUnifiedPrimarySlotClaim Claim = new();
        internal readonly HashSet<SequenceState> Bound = [];
        internal readonly Dictionary<SequenceState, BudgetReservation> Allocated = [];
        internal bool IdlePrimary = true;
        private readonly BudgetReservation _primary;
        internal Slots() { _primary = Budget.Reserve(Charges(0, Slot)); _primary.Commit(); }
        internal bool CanReuse(SequenceState sequence) => Claim.CanReuse(sequence, Bound.Contains(sequence), IdlePrimary, Bound.Contains);
        internal MemoryCharge[] Demand(SequenceState sequence) => Charges(32, CanReuse(sequence) ? 0 : Slot);
        internal bool Materialize(SequenceState sequence, IReadOnlyList<MemoryCharge>? target = null)
        {
            var prepared = GlmUnifiedPrimarySlotClaim.Prepare(sequence);
            if (!Budget.TryResizeTotals([(sequence.MemoryEnvelope!, target ?? Demand(sequence))],
                Charges(1L << 30, 4 * Slot), requirePhysicalFit: true)) return false;
            if (CanReuse(sequence))
                Claim.PublishAfterMaterialization(prepared, Bound.Contains(sequence), IdlePrimary, Bound.Contains);
            return true;
        }
        internal void Bind(SequenceState sequence)
        {
            if (Bound.Contains(sequence)) return;
            if (IdlePrimary && Claim.AllowsBinding(sequence.RequestId, Bound.Contains)) IdlePrimary = false;
            else
            {
                var child = sequence.MemoryEnvelope!.TryTake(Charges(0, Slot));
                Assert.NotNull(child); child.Commit(); Allocated.Add(sequence, child);
            }
            Bound.Add(sequence);
        }
        internal SchedulerConfig Config() => new()
        {
            BlockSize = 4, NumBlocks = 256, MaxNumRunningSequences = 3, MaxNumBatchedTokens = 64,
            MaxPrefillChunkSize = 64, SoloPrefillChunkSize = 64, PrefillChunkTokenLimit = 64,
            EnablePrefixCaching = false, StopRepetition = false,
            ExecutionOptionsOverride = new() { BatchedPathDisabled = true, PerSeqFusedEnabled = true,
                BatchedFusedDecodeEnabled = false, KvHolderPoolMax = 0 },
            MemoryAdmission = new(Budget, _ => Charges(32, Slot))
            {
                UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
                EstimateMaterializationPeak = Demand, EnterSerialExecution = _ => new Scope(),
                BeforeAdmissionWithRunning = (_, _) => Charges(1L << 30, 4 * Slot),
                AfterPrefixMaterialization = sequence => Assert.True(Materialize(sequence))
            }
        };
        public void Dispose()
        {
            foreach (var child in Allocated.Values) child.Dispose();
            _primary.Dispose();
        }
    }

    [Fact]
    public void SameScheduleTwoColdRequestsOwnOnePrimaryPromiseAndOneCompleteNewSlot()
    {
        using var slots = new Slots();
        var a = Sequence("a"); var b = Sequence("b");
        Assert.True(slots.CanReuse(a)); Assert.True(slots.CanReuse(b)); // Forecasts publish no claim.
        var pool = new BlockPool(256, 4, 0); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(slots.Config(), pool);
        scheduler.Submit(a); scheduler.Submit(b);
        var work = scheduler.Schedule();
        Assert.Equal(new[] { a, b }, work.ScheduledWork.Select(w => w.Sequence));
        Assert.Equal(0, a.MemoryEnvelope!.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(Slot, b.MemoryEnvelope!.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.True(slots.Claim.AllowsBinding("a", slots.Bound.Contains));
        Assert.False(slots.Claim.AllowsBinding("b", slots.Bound.Contains));
        // Even an execution-order change cannot let b steal a's zero-slot promise.
        slots.Bind(b); slots.Bind(a);
        Assert.Equal(Slot, b.MemoryEnvelope.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(2 * Slot, slots.Budget.Snapshot().Single(p => p.Pool == Device).Committed);
        Assert.DoesNotContain(a.MemoryEnvelope.LiveChildCharges, c => c.Bytes != 0);
        scheduler.Abort("a"); scheduler.NotifyMemoryReleased("a");
        scheduler.Abort("b"); scheduler.NotifyMemoryReleased("b");
        // Actual native child remains charged even after its request root closes.
        Assert.Equal(2 * Slot, slots.Budget.Snapshot().Single(p => p.Pool == Device).Committed);
    }

    [Fact]
    public void CancelBeforeBindLetsThirdClaimPrimaryWithoutAnOlderFullSlotSiblingStealingIt()
    {
        using var slots = new Slots(); var a = Sequence("a"); var b = Sequence("b"); var c = Sequence("c");
        var pool = new BlockPool(256, 4, 0); using var storage = pool.Storage;
        var scheduler = new ContinuousBatchScheduler(slots.Config(), pool);
        scheduler.Submit(a); scheduler.Submit(b); scheduler.Schedule();
        Assert.True(scheduler.Abort("a")); scheduler.NotifyMemoryReleased("a");
        scheduler.Submit(c); scheduler.Schedule();
        Assert.Equal(0, c.MemoryEnvelope!.OwnedCharges.Single(q => q.Pool == Device).Bytes);
        Assert.Equal(Slot, b.MemoryEnvelope!.OwnedCharges.Single(q => q.Pool == Device).Bytes);
        Assert.False(slots.Claim.AllowsBinding("b", slots.Bound.Contains));
        slots.Bind(b); slots.Bind(c);
        Assert.Equal(2, slots.Bound.Count);
        Assert.Equal(2 * Slot, slots.Budget.Snapshot().Single(q => q.Pool == Device).Committed);
        scheduler.Abort("b"); scheduler.NotifyMemoryReleased("b");
        scheduler.Abort("c"); scheduler.NotifyMemoryReleased("c");
    }

    [Fact]
    public void FailedAtomicMaterializationDoesNotPublishOrReplaceAPrimaryPromise()
    {
        using var slots = new Slots(); var failed = Sequence("failed"); var next = Sequence("next");
        using var envelope = slots.Budget.Reserve(Charges(32, 0)); failed.MemoryEnvelope = envelope;
        var before = slots.Budget.Snapshot().ToArray();
        Assert.False(slots.Materialize(failed, Charges((1L << 30) + 1, 0)));
        Assert.Equal(before, slots.Budget.Snapshot());
        Assert.True(slots.CanReuse(next)); Assert.True(slots.Claim.AllowsBinding("next", slots.Bound.Contains));
        using var nextEnvelope = slots.Budget.Reserve(Charges(32, 0)); next.MemoryEnvelope = nextEnvelope;
        Assert.True(slots.Materialize(next));
        Assert.False(slots.CanReuse(failed)); Assert.False(slots.Claim.AllowsBinding("failed", slots.Bound.Contains));
    }

    [Fact]
    public void PreparedButUnreservedRequestCannotClaimAnIdlePrimary()
    {
        var claim = new GlmUnifiedPrimarySlotClaim(); var sequence = Sequence("unreserved");
        Assert.Throws<InvalidOperationException>(() => claim.PublishAfterMaterialization(
            GlmUnifiedPrimarySlotClaim.Prepare(sequence), false, true, _ => false));
        Assert.True(claim.AllowsBinding("other", _ => false));
    }

    [Theory]
    [InlineData(SequenceStatus.FinishedAborted)]
    [InlineData(SequenceStatus.FinishedError)]
    public void TerminalUnboundOwnerReleasesOnlyThePromiseWhileActualAvailabilityStillControlsReuse(SequenceStatus terminal)
    {
        using var slots = new Slots(); var owner = Sequence("owner"); var next = Sequence("next");
        using var envelope = slots.Budget.Reserve(Charges(32, 0)); owner.MemoryEnvelope = envelope;
        Assert.True(slots.Materialize(owner)); Assert.False(slots.CanReuse(next));
        owner.Status = terminal;
        Assert.True(slots.CanReuse(next)); Assert.True(slots.Claim.AllowsBinding("next", slots.Bound.Contains));
        slots.IdlePrimary = false;
        Assert.False(slots.CanReuse(next)); // A promise retirement is not physical-slot free evidence.
        Assert.Equal(Slot, slots.Budget.Snapshot().Single(p => p.Pool == Device).Committed);
    }

    [Fact]
    public void ReleasedEnvelopeAndAnAlreadyBoundOwnerDoNotLeaveAStalePrimaryPromise()
    {
        using var slots = new Slots(); var owner = Sequence("owner"); var next = Sequence("next");
        using var envelope = slots.Budget.Reserve(Charges(32, 0)); owner.MemoryEnvelope = envelope;
        Assert.True(slots.Materialize(owner));
        owner.MemoryEnvelope = null;
        Assert.True(slots.CanReuse(next));
        owner.MemoryEnvelope = envelope;
        slots.Bound.Add(owner); // It obtained another actual holder; native primary can still be idle.
        Assert.True(slots.CanReuse(next)); Assert.True(slots.Claim.AllowsBinding("next", slots.Bound.Contains));
    }

    [Fact]
    public void ExistingOwnNativeSlotReusesWithoutClaimingOrRefundingAnotherPhysicalHolder()
    {
        using var slots = new Slots(); var claimant = Sequence("claimant"); var hit = Sequence("prefix-hit"); var third = Sequence("third");
        using var coldEnvelope = slots.Budget.Reserve(Charges(32, 0)); claimant.MemoryEnvelope = coldEnvelope;
        Assert.True(slots.Materialize(claimant));
        using var hitEnvelope = slots.Budget.Reserve(Charges(32, Slot)); hit.MemoryEnvelope = hitEnvelope;
        using var hitChild = hitEnvelope.TryTake(Charges(0, Slot))!; hitChild.Commit(); slots.Bound.Add(hit);
        Assert.True(slots.CanReuse(hit));
        var prepared = GlmUnifiedPrimarySlotClaim.Prepare(hit);
        slots.Claim.PublishAfterMaterialization(prepared, true, true, slots.Bound.Contains);
        Assert.False(slots.CanReuse(third));
        Assert.True(slots.Claim.AllowsBinding("claimant", slots.Bound.Contains));
        Assert.Equal(Slot, hitEnvelope.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(2 * Slot, slots.Budget.Snapshot().Single(p => p.Pool == Device).Committed);
    }
}
