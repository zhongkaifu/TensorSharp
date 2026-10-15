// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>Actual sequence admission metadata and quota ownership. Synthetic
/// workspace sizes do not qualify model geometry, hardware or GPU execution.</summary>
public sealed class UnifiedMemoryAdmittedCohortSnapshotTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool, Device = UnifiedMemoryModelSession.DevicePool;
    private static MemoryCharge[] Charges(long host, long device) => [new(Host, host), new(Device, device)];
    private static SequenceState Sequence(string id, BudgetReservation? envelope, int prompt = 20, int computed = 0,
        SequenceStatus status = SequenceStatus.Running)
    {
        var sequence = new SequenceState(id, Enumerable.Repeat(1, prompt).ToArray(), 32, 8, SamplingConfig.Greedy)
            { MemoryEnvelope = envelope, Status = status };
        // Metadata-only block references; no backing KV allocation or device.
        for (int index = 0; index < (computed + 7) / 8; ++index)
            sequence.BlockTable.AppendBlock(new KvBlock(index));
        sequence.SetComputedTokensForPrefixAdoption(computed);
        return sequence;
    }

    [Fact]
    public void FreshAdmittedDecoderRemainsInTheCohortAfterItsPromptCompleted()
    {
        var budget = new MemoryBudget(Charges(128, 128));
        using var currentOwner = budget.Reserve(Charges(16, 16));
        using var decoderOwner = budget.Reserve(Charges(16, 16));
        var current = Sequence("prefill", currentOwner, computed: 15);
        var decoder = Sequence("later-decoder", decoderOwner, prompt: 8, computed: 8);
        var before = budget.Snapshot().ToArray();
        Assert.Same(decoder, Assert.Single(UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted([current, decoder], current)));
        Assert.Equal(before, budget.Snapshot().ToArray());
    }

    [Fact]
    public void FinishedAndUnadmittedCandidatesAreExcludedWhileAdmittedWaitingOwnersRemain()
    {
        var budget = new MemoryBudget(Charges(128, 128));
        using var currentOwner = budget.Reserve(Charges(8, 8));
        using var waitingOwner = budget.Reserve(Charges(8, 8));
        using var stoppedOwner = budget.Reserve(Charges(8, 8));
        using var cappedOwner = budget.Reserve(Charges(8, 8));
        using var abortedOwner = budget.Reserve(Charges(8, 8));
        using var failedOwner = budget.Reserve(Charges(8, 8));
        var current = Sequence("current", currentOwner);
        var waiting = Sequence("admitted-waiting", waitingOwner, status: SequenceStatus.Waiting);
        SequenceState[] candidates = [current, Sequence("pending", null), waiting,
            Sequence("stopped", stoppedOwner, status: SequenceStatus.FinishedStopped),
            Sequence("capped", cappedOwner, status: SequenceStatus.FinishedLengthCapped),
            Sequence("aborted", abortedOwner, status: SequenceStatus.FinishedAborted),
            Sequence("failed", failedOwner, status: SequenceStatus.FinishedError)];
        var before = budget.Snapshot().ToArray();
        Assert.Same(waiting, Assert.Single(UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted(candidates, current)));
        Assert.Throws<ArgumentNullException>(() => UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted(null!, current));
        Assert.Throws<ArgumentNullException>(() => UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted(candidates, null!));
        Assert.Throws<ArgumentException>(() => UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted([null!], current));
        Assert.Equal(before, budget.Snapshot().ToArray());
    }

    [Fact]
    public void ANewSnapshotIncludesLaterAdmissionsWithoutMutatingTheOldSnapshotOrDoubleCountingReferences()
    {
        var budget = new MemoryBudget(Charges(128, 128));
        using var currentOwner = budget.Reserve(Charges(8, 8));
        using var oldOwner = budget.Reserve(Charges(8, 8));
        using var newOwner = budget.Reserve(Charges(8, 8));
        using var decoderOwner = budget.Reserve(Charges(8, 8));
        var current = Sequence("current", currentOwner);
        var old = Sequence("same-id", oldOwner);
        var later = Sequence("later-prefill", newOwner);
        var decoder = Sequence("same-id", decoderOwner, prompt: 8, computed: 8);
        var registry = new List<SequenceState> { current, old };
        var stale = UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted(registry, current);
        registry.Add(later); registry.Add(decoder); registry.Add(old);
        var fresh = UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted(registry, current);
        Assert.Equal(new[] { old }, stale);
        Assert.Equal(new[] { old, later, decoder }, fresh);
        registry.Clear();
        Assert.Equal(new[] { old, later, decoder }, fresh);
        Assert.Equal(32, budget.Snapshot().Single(p => p.Pool == Device).Reserved);
    }

    [Fact]
    public void TailReplanCanAdmitAnOptionalCopyWithoutRefundingOtherPromisesChildrenOrPhysicalCapacity()
    {
        var budget = new MemoryBudget(Charges(1024, 4096));
        using var joint = budget.Reserve(Charges(400, 3000));
        using var jointChild = joint.TryTake(Charges(100, 700))!; jointChild.Commit();
        using var currentOwner = budget.Reserve(Charges(100, 0));
        using var currentChild = currentOwner.TryTake(Charges(8, 0))!; currentChild.Commit();
        using var primary = budget.Reserve(Charges(0, 256)); primary.Commit();
        using var otherPrefillOwner = budget.Reserve(Charges(120, 256));
        using var otherDecoderOwner = budget.Reserve(Charges(40, 256));
        using var unrelated = budget.Reserve(Charges(30, 32)); unrelated.Commit();
        var current = Sequence("tail", currentOwner, computed: 15);
        var prefill = Sequence("other-prefill", otherPrefillOwner, computed: 1);
        var decoder = Sequence("later-decoder", otherDecoderOwner, prompt: 8, computed: 8);
        var otherBefore = new[] { otherPrefillOwner.OwnedCharges.ToArray(), otherDecoderOwner.OwnedCharges.ToArray(), unrelated.OwnedCharges.ToArray() };
        var actualChildren = joint.LiveChildCharges.ToArray();
        Assert.Null(OwnedAllocationScope.TryReserveOptional(budget, Charges(0, 512), Charges(1024, 4096)));

        var cohort = UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted([current, prefill, decoder], current);
        Assert.Equal(new[] { prefill, decoder }, cohort);
        // The other live prefill promises 768 device bytes, the later decoder
        // still needs 128. The current request now has only five staging rows.
        var promises = new Dictionary<SequenceState, InferenceMemoryBytes>
            { [prefill] = new(200, 768), [decoder] = new(64, 128) };
        var graph = new InferenceMemoryBytes(20, 128);
        foreach (var sibling in cohort) graph = GlmUnifiedPrefillAdmission.Floor(graph, promises[sibling]);
        var children = new InferenceMemoryBytes(100, 700);
        var protectedTarget = GlmUnifiedPrefillAdmission.PreserveReplacement(graph, children, 2);
        Assert.Equal(new InferenceMemoryBytes(200, 1084), protectedTarget);
        Assert.True(budget.TryResizeTotals([(joint, Charges(protectedTarget.Host, protectedTarget.Device)),
            (currentOwner, Charges((current.PromptTokens.Count - current.NumComputedTokens) * 4, 0))],
            Charges(1024, 4096), requirePhysicalFit: true));
        Assert.Equal(actualChildren, joint.LiveChildCharges.ToArray());
        Assert.Equal(otherBefore[0], otherPrefillOwner.OwnedCharges.ToArray());
        Assert.Equal(otherBefore[1], otherDecoderOwner.OwnedCharges.ToArray());
        Assert.Equal(otherBefore[2], unrelated.OwnedCharges.ToArray());
        Assert.Equal(8, currentOwner.LiveChildCharges.Single(p => p.Pool == Host).Bytes);
        long occupied = budget.Snapshot().Single(p => p.Pool == Device).Reserved
            + budget.Snapshot().Single(p => p.Pool == Device).Committed;
        var beforeRefusal = budget.Snapshot().ToArray();
        Assert.Null(OwnedAllocationScope.TryReserveOptional(budget, Charges(0, 512), Charges(1024, occupied)));
        Assert.Equal(beforeRefusal, budget.Snapshot().ToArray());
        using var copy = OwnedAllocationScope.TryReserveOptional(budget, Charges(0, 512), Charges(1024, occupied + 512));
        Assert.NotNull(copy);
        using var payload = copy.TryTake(Charges(0, 512))!; payload.Commit(); copy.Dispose();
        Assert.Equal(512, payload.OwnedCharges.Single(p => p.Pool == Device).Bytes);
        Assert.Equal(700, joint.LiveChildCharges.Single(p => p.Pool == Device).Bytes);
        Assert.Equal(256, primary.OwnedCharges.Single(p => p.Pool == Device).Bytes);
    }
}
