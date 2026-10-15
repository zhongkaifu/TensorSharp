// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>Prerequisite planning contract exercised by a complete-state CPU
/// oracle coordinator. This does not wire a live request pager or qualify
/// trained-model, native, speculative or distributed transfers.</summary>
public sealed class WholeStateTransferPlanTests
{
    private const WholeStateComponents State = WholeStateComponents.AttentionKv | WholeStateComponents.Recurrence
        | WholeStateComponents.Convolution | WholeStateComponents.Indexer | WholeStateComponents.PositionMetadata;
    private static readonly WholeStateTransferCapabilities Caps = new(State, State, State, true);
    private static WholeStateTransferMemory Memory() => new([new("ram", 1024), new("gpu", 0)],
        [new("ram", 256)], [new("gpu", 64), new("ram", 0)], [new("ram", 256)]);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteRecurrentStateResumesAtComputedHeadWithoutLosingOrDuplicatingPendingToken(bool deviceToken)
    {
        using var fixture = new Coordinator(deviceToken);
        lock (fixture.ComputeLock)
        {
            var plan = fixture.Plan();
            Assert.Equal(16, plan.ComputedTokens);
            Assert.Equal(deviceToken ? 16 : 17, plan.TotalTokens);
            Assert.Equal(deviceToken ? 0 : 1, plan.UncomputedTokens);
            if (!deviceToken) Assert.Equal(fixture.NextToken, plan.UncomputedTokenAt(0));
            Assert.True(fixture.Capture(plan));
            Assert.Equal(0, fixture.Model.ActiveLength);
            Assert.True(fixture.Budget.Snapshot().Single(p => p.Pool == "ram").Committed > 0);
            Assert.Equal(0, fixture.Budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
            fixture.Sequence.Status = SequenceStatus.Waiting; // Preserve the computed head; do not reset for replay.
            Assert.True(fixture.Restore(plan));
            Assert.Equal(16, fixture.Model.ActiveLength);
            if (deviceToken)
            {
                Assert.Equal(fixture.NextToken, fixture.Sequence.PendingDeviceToken);
                fixture.Sequence.PendingDeviceToken = null;
                fixture.Sequence.AppendOutputToken(fixture.NextToken);
            }
            var actual = fixture.Model.Forward([fixture.NextToken]);
            fixture.Sequence.AdvanceComputedTokens(1);
            Assert.Equal(fixture.Control.Forward([fixture.NextToken]), actual);
            for (int step = 0; step < 8; ++step)
            {
                int token = ArgMax(actual);
                fixture.Sequence.AppendOutputToken(token);
                actual = fixture.Model.Forward([token]);
                fixture.Sequence.AdvanceComputedTokens(1);
                Assert.Equal(fixture.Control.Forward([token]), actual);
            }
            Assert.Equal(9, fixture.Sequence.OutputTokens.Count);
            Assert.Equal(25, fixture.Sequence.NumComputedTokens);
        }
    }

    [Theory]
    [InlineData("capture")]
    [InlineData("export")]
    [InlineData("cancel")]
    [InlineData("stale")]
    public void FailedTransferNeverReleasesAuthoritativeStateOrSiblingCredit(string fault)
    {
        using var fixture = new Coordinator(false);
        using var sibling = fixture.Budget.Reserve([new("gpu", 192), new("ram", 32)]);
        lock (fixture.ComputeLock)
        {
            var plan = fixture.Plan();
            if (fault is "capture" or "export") fixture.Model.FailNext(fault);
            bool captured = false;
            if (fault == "cancel")
                Assert.Throws<OperationCanceledException>(() => fixture.Capture(plan, fault));
            else if (fault == "stale")
                Assert.Throws<InvalidOperationException>(() => fixture.Capture(plan, fault));
            else captured = fixture.Capture(plan, fault);
            Assert.False(captured);
            Assert.Null(fixture.Recovery);
            Assert.Equal(16, fixture.Model.ActiveLength);
            Assert.Equal(64, fixture.SourceAllocation.OwnedCharges.Single().Bytes);
            Assert.Equal(192, sibling.OwnedCharges.Single(c => c.Pool == "gpu").Bytes);
            Assert.Equal(32, sibling.OwnedCharges.Single(c => c.Pool == "ram").Bytes);
            Assert.Equal(64, fixture.Budget.Snapshot().Single(c => c.Pool == "gpu").Committed);
            Assert.Equal(0, fixture.Budget.Snapshot().Single(c => c.Pool == "ram").Committed);
            Assert.Equal(fixture.Control.Forward([fixture.NextToken]), fixture.Model.Forward([fixture.NextToken]));
        }
    }

    [Fact]
    public void RestoreQuotaRefusalKeepsRecoveryAndPendingTailForRetry()
    {
        using var fixture = new Coordinator(false);
        lock (fixture.ComputeLock)
        {
            var plan = fixture.Plan();
            Assert.True(fixture.Capture(plan));
            using var sibling = fixture.Budget.Reserve([new("gpu", 193)]);
            long calls = fixture.Model.StateMemberCalls;
            var before = fixture.Budget.Snapshot().ToArray();
            Assert.False(fixture.Restore(plan));
            Assert.Equal(calls, fixture.Model.StateMemberCalls);
            Assert.Equal(before, fixture.Budget.Snapshot());
            Assert.NotNull(fixture.Recovery);
            Assert.Equal(fixture.NextToken, plan.UncomputedTokenAt(0));
            sibling.Dispose();
            Assert.True(fixture.Restore(plan));
            Assert.Equal(fixture.Control.Forward([fixture.NextToken]), fixture.Model.Forward([fixture.NextToken]));
        }
    }

    [Fact]
    public void ImportFailureRetainsTheLastRecoveryCopyAndRetryResumesExactPendingHistory()
    {
        using var fixture = new Coordinator(false);
        lock (fixture.ComputeLock)
        {
            var plan = fixture.Plan();
            Assert.True(fixture.Capture(plan));
            byte[] recovery = fixture.Recovery!;
            var before = fixture.Budget.Snapshot().ToArray();
            fixture.Model.FailNext("import");
            Assert.False(fixture.Restore(plan));
            Assert.Same(recovery, fixture.Recovery);
            Assert.Equal(before, fixture.Budget.Snapshot());
            Assert.Equal(0, fixture.Model.ActiveLength);
            Assert.Equal(16, fixture.Sequence.NumComputedTokens);
            Assert.Equal(fixture.NextToken, plan.UncomputedTokenAt(0));
            Assert.True(fixture.Restore(plan));
            Assert.Equal(fixture.Control.Forward([fixture.NextToken]), fixture.Model.Forward([fixture.NextToken]));
        }
    }

    [Fact]
    public void PlansRequireComputeFenceCompleteCoverageAndSettledSingleDeviceState()
    {
        using var fixture = new Coordinator(false);
        Assert.Throws<InvalidOperationException>(() => fixture.Plan());
        lock (fixture.ComputeLock)
        {
            foreach (var missing in new[] { WholeStateComponents.Recurrence, WholeStateComponents.Indexer, WholeStateComponents.PositionMetadata })
            {
                Assert.Throws<NotSupportedException>(() => fixture.Plan(caps: Caps with { Exported = State & ~missing }));
                Assert.Throws<NotSupportedException>(() => fixture.Plan(caps: Caps with { Imported = State & ~missing }));
            }
            Assert.Throws<NotSupportedException>(() => fixture.Plan(state: new(16, HasJointAliases: true),
                caps: Caps with { FlushesJointAliases = false }));
            Assert.Throws<NotSupportedException>(() => fixture.Plan(state: new(16, PhysicalDevices: 2)));
            Assert.Throws<NotSupportedException>(() => fixture.Plan(state: new(16, HasActiveDraftState: true)));
            Assert.Throws<NotSupportedException>(() => fixture.Plan(state: new(16, HasUncommittedVerification: true)));
            Assert.Throws<InvalidOperationException>(() => fixture.Plan(state: new(17)));
            var plan = fixture.Plan();
            var foreign = new SequenceState(fixture.Sequence.RequestId, fixture.Sequence.PromptTokens, 16, 8, SamplingConfig.Greedy);
            Assert.Throws<InvalidOperationException>(() => plan.ValidateLogicalState(foreign));
            float previous = fixture.Sequence.LastLogits![0];
            fixture.Sequence.LastLogits[0] += 1;
            Assert.Throws<InvalidOperationException>(() => plan.ValidateLogicalState(fixture.Sequence));
            fixture.Sequence.LastLogits[0] = previous;
            plan.ValidateLogicalState(fixture.Sequence);
            fixture.Sequence.PromptTokens[0]++;
            Assert.Throws<InvalidOperationException>(() => plan.ValidateBeforeDeviceRelease(fixture.Sequence, 16));
        }
    }

    [Fact]
    public void TransferPeaksAreImmutableAndIncludeCoexistingPayloadAndScratchUnderFreshLimits()
    {
        using var fixture = new Coordinator(false);
        lock (fixture.ComputeLock)
        {
            var payload = new[] { new MemoryCharge("ram", 1024), new MemoryCharge("gpu", 0) };
            var plan = WholeStateTransferPlan.Prepare(fixture.Sequence, fixture.ComputeLock, new(16), Caps,
                Memory() with { CapturePayload = payload });
            payload[0] = new("ram", 0);
            Assert.Equal(1280, plan.CapturePeak.Single(p => p.Pool == "ram").Bytes);
            Assert.Throws<NotSupportedException>(() => ((IList<MemoryCharge>)plan.CapturePeak)[0] = new("ram", 0));
            Assert.Equal(256, plan.RestorePeak.Single(p => p.Pool == "ram").Bytes);
            var before = fixture.Budget.Snapshot().ToArray();
            Assert.Throws<MemoryPressureException>(() => plan.TryReserveCapture(fixture.Budget, 16,
                [new("ram", 1279), new("gpu", 256)]));
            Assert.Equal(before, fixture.Budget.Snapshot());
            using var owner = plan.TryReserveCapture(fixture.Budget, 16, [new("ram", 1280), new("gpu", 256)]);
            Assert.NotNull(owner);
            using var payloadChild = owner!.TryTake([new("ram", 1024)]);
            Assert.NotNull(payloadChild); payloadChild!.Commit();
            using var scratchChild = owner.TryTake([new("ram", 256)]);
            Assert.NotNull(scratchChild); scratchChild!.Commit();
            Assert.Null(owner.TryTake([new("ram", 1)])); // No unrelated owner's spare credit.
            Assert.Throws<ArgumentOutOfRangeException>(() => fixture.Plan(memory: Memory() with { RestoreScratch = [new("ram", -1)] }));
            Assert.Throws<OverflowException>(() => fixture.Plan(memory: Memory() with { CapturePayload = [new("ram", long.MaxValue)] }));
        }
    }

    [Fact]
    public void ReacquiringComputeLockCannotPublishStateAfterInPlaceLogitMutation()
    {
        using var fixture = new Coordinator(false);
        WholeStateTransferPlan plan;
        lock (fixture.ComputeLock) plan = fixture.Plan();
        lock (fixture.ComputeLock) fixture.Sequence.LastLogits![0] += 1;
        var before = fixture.Budget.Snapshot().ToArray();
        lock (fixture.ComputeLock)
        {
            Assert.Throws<InvalidOperationException>(() => plan.TryReserveCapture(fixture.Budget, 16));
            Assert.Throws<InvalidOperationException>(() => plan.ValidateBeforeDeviceRelease(fixture.Sequence, 16));
            Assert.Equal(before, fixture.Budget.Snapshot());
            Assert.Equal(16, fixture.Model.ActiveLength);
            Assert.Equal(64, fixture.SourceAllocation.OwnedCharges.Single().Bytes);
        }
    }

    [Fact]
    public void BothPhasePeaksDefensivelyOwnAllInputCollectionsAndCannotBeMutatedByACaller()
    {
        using var fixture = new Coordinator(false);
        lock (fixture.ComputeLock)
        {
            var capture = new List<MemoryCharge> { new("ram", 1024), new("gpu", 0) };
            var restore = new List<MemoryCharge> { new("gpu", 64), new("ram", 0) };
            var plan = fixture.Plan(memory: Memory() with { CapturePayload = capture, RestoreState = restore });
            capture.Clear(); restore[0] = new("gpu", 0);
            Assert.Equal(1280, plan.CapturePeak.Single(p => p.Pool == "ram").Bytes);
            Assert.Equal(64, plan.RestorePeak.Single(p => p.Pool == "gpu").Bytes);
            Assert.Throws<NotSupportedException>(() => ((IList<MemoryCharge>)plan.RestorePeak).Clear());
            Assert.Equal(4L * (fixture.Sequence.NumTotalTokens + fixture.Sequence.LastLogits!.Length),
                WholeStateTransferPlan.ManagedBoundaryCopyBytes(fixture.Sequence));
        }
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    }

    /// <summary>Generic CPU complete-state coordinator, deliberately separate
    /// from block KV and any trained-model adapter. Oracle recurrent history is
    /// independently serialized/restored; ledger ownership governs transfer
    /// payload and scratch, with no source release until successful revalidation.</summary>
    private sealed class Coordinator : IDisposable
    {
        internal readonly OracleModel Model = OracleFakes.R(8), Control = OracleFakes.R(8);
        internal readonly MemoryBudget Budget = new([new("ram", 8192), new("gpu", 256)]);
        internal readonly SequenceState Sequence;
        internal readonly BudgetReservation SourceAllocation;
        private BudgetReservation? _recoveryOwner, _restoredState;
        internal byte[]? Recovery;
        internal readonly int NextToken;
        internal object ComputeLock => ((IModelArchitecture)Model).GpuComputeLock;
        internal Coordinator(bool deviceToken)
        {
            Sequence = new("live", Enumerable.Range(1, 16).ToArray(), 16, 8, SamplingConfig.Greedy);
            for (int i = 0; i < 4; ++i) Sequence.BlockTable.AppendBlock(new(i));
            Model.BindSequenceCache(Sequence.RequestId); Control.BindSequenceCache("control");
            Sequence.LastLogits = Model.Forward(Sequence.PromptTokens.ToArray());
            Assert.Equal(Control.Forward(Sequence.PromptTokens.ToArray()), Sequence.LastLogits);
            Sequence.AdvanceComputedTokens(16); NextToken = ArgMax(Sequence.LastLogits);
            if (deviceToken) { Sequence.PendingDeviceToken = NextToken; Sequence.PendingDevicePosition = 16; }
            else Sequence.AppendOutputToken(NextToken);
            SourceAllocation = Budget.Reserve([new("gpu", 64)]); SourceAllocation.Commit();
        }
        internal WholeStateTransferPlan Plan(WholeStateExecutionState? state = null,
            WholeStateTransferCapabilities? caps = null, WholeStateTransferMemory? memory = null)
            => WholeStateTransferPlan.Prepare(Sequence, ComputeLock, state ?? new(Model.ActiveLength), caps ?? Caps, memory ?? Memory());
        internal bool Capture(WholeStateTransferPlan plan, string? fault = null)
        {
            using var owner = plan.TryReserveCapture(Budget, Model.ActiveLength);
            if (owner == null) return false;
            using var scratch = owner.TryTake([new("ram", 256)]);
            Assert.NotNull(scratch); scratch!.Commit();
            const string key = "export";
            if (!Model.TryCaptureCopy(Sequence.RequestId, key, out _)) return false;
            try
            {
                using var stream = new MemoryStream();
                if (!Model.TryExport(key, stream)) return false;
                if (fault == "cancel") throw new OperationCanceledException();
                if (fault == "stale") Sequence.OutputTokens[0] = (NextToken + 1) % Model.Config.VocabSize;
                plan.ValidateBeforeDeviceRelease(Sequence, Model.ActiveLength);
                byte[] payload = stream.ToArray();
                var child = owner.TryTake([new("ram", payload.Length)]);
                Assert.NotNull(child); child!.Commit();
                _recoveryOwner = child; Recovery = payload;
                Model.OnSequenceReleased(Sequence.RequestId); SourceAllocation.Dispose();
                return true;
            }
            finally { Model.DiscardRetainedCache(key); }
        }
        internal bool Restore(WholeStateTransferPlan plan)
        {
            using var owner = plan.TryReserveRestore(Budget);
            if (owner == null) return false;
            using var scratch = owner.TryTake([new("ram", 256)]);
            Assert.NotNull(scratch); scratch!.Commit();
            using var stream = new MemoryStream(Recovery!, writable: false);
            if (!Model.TryImport("import", plan.ComputedTokens, stream, out _)) return false;
            Assert.True(Model.TryRebindRetainedCache("import", Sequence.RequestId));
            Assert.False(Model.BindSequenceCache(Sequence.RequestId));
            plan.ValidateBeforeDeviceRelease(Sequence, Model.ActiveLength);
            _restoredState = owner.TryTake([new("gpu", 64)]);
            Assert.NotNull(_restoredState); _restoredState!.Commit();
            _recoveryOwner!.Dispose(); _recoveryOwner = null; Recovery = null;
            return true;
        }
        public void Dispose()
        {
            Model.Dispose(); Control.Dispose(); SourceAllocation.Dispose(); _recoveryOwner?.Dispose(); _restoredState?.Dispose();
            Assert.All(Budget.Snapshot(), p => { Assert.Equal(0, p.Reserved); Assert.Equal(0, p.Committed); });
        }
    }
}
