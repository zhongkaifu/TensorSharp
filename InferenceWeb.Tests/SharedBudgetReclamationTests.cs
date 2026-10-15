// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class SharedBudgetReclamationTests
{
    static SequenceState Request(string id) => new(id, [1, 2], 2, 8, SamplingConfig.Greedy);

    [Fact]
    public void AReestimatedLargerPeakCanReclaimWithoutLosingItsFifoPlace()
    {
        var budget = new MemoryBudget([new("ram", 128)]);
        using var retained = budget.Reserve([new("ram", 64)]);
        using var temporary = budget.Reserve([new("ram", 64)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        using var follower = SharedBudgetReclamation.Register(budget, () => { });
        using var donor = SharedBudgetReclamation.Register(budget, () => { });
        var first = Request("growing"); var next = Request("younger-small");
        Assert.Null(head.TryReserve(first, [new("ram", 64)]));
        Assert.Null(follower.TryReserve(next, [new("ram", 32)]));
        temporary.Dispose();
        Assert.Null(head.TryReserve(first, [new("ram", 128)]));
        Assert.False(follower.CanAdmit([new("ram", 32)]));
        using var reclaim = donor.TryBeginReclamation();
        Assert.NotNull(reclaim); Assert.True(reclaim.NeedsReclamation());
        retained.Dispose();
        Assert.False(reclaim.NeedsReclamation());
        Assert.Null(follower.TryReserve(next, [new("ram", 32)]));
        using var admitted = head.TryReserve(first, [new("ram", 128)]);
        Assert.NotNull(admitted);
        admitted.Dispose();
        using var second = follower.TryReserve(next, [new("ram", 32)]);
        Assert.NotNull(second);
    }

    [Fact]
    public void AReestimatedSmallerPeakInvalidatesItsOldLeaseAndStopsAfterEnoughRelease()
    {
        var budget = new MemoryBudget([new("ram", 128)]);
        using var firstCache = budget.Reserve([new("ram", 64)]);
        using var secondCache = budget.Reserve([new("ram", 64)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        using var donor = SharedBudgetReclamation.Register(budget, () => { });
        var request = Request("shrinking");
        Assert.Null(head.TryReserve(request, [new("ram", 128)]));
        using var oldLease = donor.TryBeginReclamation();
        Assert.NotNull(oldLease); Assert.True(oldLease.NeedsReclamation());
        var signal = budget.ChangeSignal;
        Assert.Null(head.TryReserve(request, [new("ram", 64)]));
        Assert.Same(signal, budget.ChangeSignal);
        Assert.False(oldLease.NeedsReclamation());
        oldLease.Dispose();
        using var newLease = donor.TryBeginReclamation();
        Assert.NotNull(newLease); Assert.True(newLease.NeedsReclamation());
        firstCache.Dispose();
        Assert.False(newLease.NeedsReclamation());
        Assert.Equal(64, budget.Snapshot().Single().Reserved); // Second cache was not evicted.
        using var admitted = head.TryReserve(request, [new("ram", 64)]);
        Assert.NotNull(admitted);
    }

    [Fact]
    public void AChangedPoolSetKeepsQueuePriorityAndUsesTheCurrentPoolForReclamation()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        using var hostCache = budget.Reserve([new("ram", 100)]);
        using var deviceCache = budget.Reserve([new("gpu", 100)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        using var follower = SharedBudgetReclamation.Register(budget, () => { });
        using var donor = SharedBudgetReclamation.Register(budget, () => { });
        var first = Request("changed-pool"); var next = Request("younger-device");
        Assert.Null(head.TryReserve(first, [new("ram", 50)]));
        Assert.Null(follower.TryReserve(next, [new("gpu", 50)]));
        Assert.Null(head.TryReserve(first, [new("ram", 0), new("gpu", 50)]));
        using var lease = donor.TryBeginReclamation();
        Assert.NotNull(lease);
        hostCache.Dispose();
        Assert.True(lease.NeedsReclamation()); // Unrelated RAM release cannot satisfy GPU demand.
        deviceCache.Dispose();
        Assert.False(lease.NeedsReclamation());
        Assert.Null(follower.TryReserve(next, [new("gpu", 50)]));
        using var admitted = head.TryReserve(first, [new("gpu", 50)]);
        Assert.NotNull(admitted);
        using var second = follower.TryReserve(next, [new("gpu", 50)]);
        Assert.NotNull(second);
    }

    [Fact]
    public void EquivalentNormalizedPeaksDoNotWakePeersInvalidateLeasesOrResetAttempts()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100), new("unused", 1)]);
        using var permanent = budget.Reserve([new("ram", 100), new("gpu", 100)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        int wakes = 0;
        using var donor = SharedBudgetReclamation.Register(budget, () => wakes++);
        var request = Request("same-total");
        Assert.Null(head.TryReserve(request, [new("ram", 40), new("gpu", 100), new("ram", 60)]));
        using var lease = donor.TryBeginReclamation();
        Assert.NotNull(lease);
        int before = wakes;
        Assert.Null(head.TryReserve(request, [new("gpu", 50), new("ram", 100), new("gpu", 50), new("unused", 0)]));
        Assert.Equal(before, wakes);
        Assert.True(lease.NeedsReclamation());
        lease.Dispose();
        var signal = budget.ChangeSignal;
        before = wakes;
        for (int attempt = 0; attempt < 100; attempt++)
        {
            Assert.Null(head.TryReserve(request, [new("gpu", 100), new("ram", 50), new("ram", 50)]));
            Assert.Null(donor.TryBeginReclamation());
        }
        Assert.Equal(before, wakes);
        Assert.Same(signal, budget.ChangeSignal);
    }

    [Fact]
    public void AChangedPeakAllowsOneNewDonorAttemptWithoutABudgetPulse()
    {
        var budget = new MemoryBudget([new("ram", 200)]);
        using var permanent = budget.Reserve([new("ram", 200)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        int wakes = 0;
        using var donor = SharedBudgetReclamation.Register(budget, () => wakes++);
        var request = Request("fresh-forecast");
        Assert.Null(head.TryReserve(request, [new("ram", 100)]));
        using (var exhausted = donor.TryBeginReclamation()) Assert.NotNull(exhausted);
        Assert.Null(donor.TryBeginReclamation());
        var signal = budget.ChangeSignal;
        int before = wakes;
        Assert.Null(head.TryReserve(request, [new("ram", 150)]));
        Assert.Equal(before + 1, wakes);
        Assert.Same(signal, budget.ChangeSignal);
        using (var retry = donor.TryBeginReclamation())
        {
            Assert.NotNull(retry); Assert.True(retry.NeedsReclamation());
        }
        Assert.Null(donor.TryBeginReclamation());
    }

    [Fact]
    public void InvalidReestimatedPeaksCannotReplaceTheLiveDemand()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var permanent = budget.Reserve([new("ram", 100)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        int wakes = 0;
        using var donor = SharedBudgetReclamation.Register(budget, () => wakes++);
        var request = Request("invalid-forecast");
        Assert.Null(head.TryReserve(request, [new("ram", 100)]));
        using var lease = donor.TryBeginReclamation();
        Assert.NotNull(lease);
        int before = wakes;
        Assert.Throws<OverflowException>(() => head.TryReserve(request, [new("ram", long.MaxValue), new("ram", 1)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => head.TryReserve(request, [new("ram", -1)]));
        Assert.Throws<ArgumentException>(() => head.TryReserve(request, [new("missing", 1)]));
        Assert.Equal(before, wakes);
        Assert.True(lease.NeedsReclamation());
        head.Cancel(request);
        Assert.False(lease.NeedsReclamation());
    }

    [Theory]
    [InlineData(true, "unknown-positive")]
    [InlineData(true, "unknown-zero")]
    [InlineData(true, "negative")]
    [InlineData(true, "overflow")]
    [InlineData(false, "unknown-positive")]
    [InlineData(false, "unknown-zero")]
    [InlineData(false, "negative")]
    [InlineData(false, "overflow")]
    public void InvalidFifoOrWaitOnlyPeaksLeaveTheOriginalDemandLeaseAndWakeStateUnchanged(
        bool waitOnly, string invalidKind)
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        using var permanent = budget.Reserve([new("ram", 100), new("gpu", 100)]);
        using var oldest = SharedBudgetReclamation.Register(budget, () => { });
        using var subject = SharedBudgetReclamation.Register(budget, () => { });
        int wakes = 0;
        using var donor = SharedBudgetReclamation.Register(budget, () => wakes++);
        Assert.Null(oldest.TryReserve(Request("older-host"), [new("ram", 100)]));
        var request = Request("original-device");
        Assert.Null(subject.TryReserve(request, [new("gpu", 100)]));
        using (var hostAttempt = donor.TryBeginReclamation()) Assert.NotNull(hostAttempt);
        using var originalLease = donor.TryBeginReclamation();
        Assert.NotNull(originalLease); Assert.True(originalLease.NeedsReclamation());
        MemoryCharge[] invalid = invalidKind switch
        {
            "unknown-positive" => [new("ram", 1), new("missing", 1)],
            "unknown-zero" => [new("ram", 1), new("missing", 0)],
            "negative" => [new("ram", 1), new("gpu", -1)],
            _ => [new("ram", long.MaxValue), new("ram", 1)]
        };
        // A wait-only replacement must not erase the old request before its
        // input is validated. The other path keeps the request but conflicts
        // with older RAM demand, so neither path reaches budget.TryReserve.
        var attemptedRequest = waitOnly ? Request("invalid-replacement") : request;
        var signal = budget.ChangeSignal;
        int before = wakes;
        if (invalidKind == "overflow")
            Assert.Throws<OverflowException>(() => subject.TryReserve(attemptedRequest, invalid, waitOnly: waitOnly));
        else if (invalidKind == "negative")
            Assert.Throws<ArgumentOutOfRangeException>(() => subject.TryReserve(attemptedRequest, invalid, waitOnly: waitOnly));
        else
            Assert.Throws<ArgumentException>(() => subject.TryReserve(attemptedRequest, invalid, waitOnly: waitOnly));
        Assert.Equal(before, wakes);
        Assert.Same(signal, budget.ChangeSignal);
        Assert.True(originalLease.NeedsReclamation());
        Assert.All(budget.Snapshot(), pool => Assert.Equal(100, pool.Reserved));
        subject.Cancel(request);
        Assert.False(originalLease.NeedsReclamation());
    }

    [Fact]
    public void EqualPhysicalAndConfiguredCapacityRaceQueuesTheLoserWithoutLosingFifoPriority()
    {
        var budget = new MemoryBudget([new("ram", 128)]);
        using var permanentOwner = budget.Reserve([new("ram", 64)]);
        using var first = SharedBudgetReclamation.Register(budget, () => { });
        using var second = SharedBudgetReclamation.Register(budget, () => { });
        var loser = Request("lost-ready-race");
        // Both workers observed readiness before either performed its atomic
        // reservation. The second must queue when the first consumes the credit.
        Assert.True(first.CanAttemptPhysicalAdmission([new("ram", 64)]));
        Assert.True(second.CanAttemptPhysicalAdmission([new("ram", 64)]));
        using var winner = first.TryReserve(Request("winner"), [new("ram", 64)], [new("ram", 128)]);
        Assert.NotNull(winner);
        Assert.Null(second.TryReserve(loser, [new("ram", 64)], [new("ram", 128)]));
        Assert.Equal(128, budget.Snapshot().Single().Reserved);

        winner.Dispose();
        Assert.Null(first.TryReserve(Request("younger-small"), [new("ram", 32)], [new("ram", 128)]));
        using var retry = second.TryReserve(loser, [new("ram", 64)], [new("ram", 128)]);
        Assert.NotNull(retry);
        retry.Dispose();
        Assert.Equal(64, budget.Snapshot().Single().Reserved);
    }

    [Fact]
    public void DeferredPhysicalObservationCannotAdmitWhenCreditArrivesBeforeQueuePublication()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var owner = budget.Reserve([new("ram", 100)]);
        using var first = SharedBudgetReclamation.Register(budget, () => { });
        using var second = SharedBudgetReclamation.Register(budget, () => { });
        var request = Request("physical-observation-required");
        Assert.False(first.CanAdmit([new("ram", 100)]));
        owner.Dispose(); // Arrives between the preliminary eligibility check and publication.
        Assert.Null(first.TryReserve(request, [new("ram", 100)], waitOnly: true));
        Assert.Equal(100, budget.Snapshot().Single().Available);
        Assert.Null(second.TryReserve(Request("follower"), [new("ram", 1)]));
        Assert.Throws<MemoryPressureException>(() => first.TryReserve(request,
            [new("ram", 100)], [new("ram", 50)]));
        first.Cancel(request);
        using var next = second.TryReserve(Request("follower"), [new("ram", 1)]);
        Assert.NotNull(next);
    }

    [Fact]
    public void OlderLargeDemandCannotBeStarvedByRepeatedSmallerRequests()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var external = budget.Reserve([new("ram", 50)]);
        using var older = SharedBudgetReclamation.Register(budget, () => { });
        using var newer = SharedBudgetReclamation.Register(budget, () => { });
        var big = Request("big"); var small = Request("small");
        Assert.Null(older.TryReserve(big, [new("ram", 100)]));
        Assert.True(budget.CanReserve([new("ram", 50)]));
        for (int i = 0; i < 100; i++)
        {
            Assert.Null(newer.TryReserve(small, [new("ram", 50)]));
            Assert.False(newer.CanAdmit([new("ram", 50)]));
        }
        external.Dispose();
        Assert.Null(newer.TryReserve(small, [new("ram", 50)]));
        using var bigOwner = older.TryReserve(big, [new("ram", 100)]);
        Assert.NotNull(bigOwner);
        Assert.Null(newer.TryReserve(small, [new("ram", 50)]));
        bigOwner.Dispose();
        using var smallOwner = newer.TryReserve(small, [new("ram", 50)]);
        Assert.NotNull(smallOwner);
        // A fresh request by the formerly-oldest engine has no inherited priority.
        Assert.Null(older.TryReserve(Request("next-big"), [new("ram", 100)]));
    }

    [Fact]
    public void IndependentPoolsProceedWhileOverlappingMultiPoolDemandWaits()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100), new("disk", 100)]);
        using var external = budget.Reserve([new("gpu", 100)]);
        using var oldest = SharedBudgetReclamation.Register(budget, () => { });
        using var overlapping = SharedBudgetReclamation.Register(budget, () => { });
        using var unrelated = SharedBudgetReclamation.Register(budget, () => { });
        Assert.Null(oldest.TryReserve(Request("uma"), [new("ram", 100), new("gpu", 100)]));
        Assert.Null(overlapping.TryReserve(Request("host"), [new("ram", 1)]));
        using var disk = unrelated.TryReserve(Request("disk"), [new("disk", 100), new("ram", 0)]);
        Assert.NotNull(disk);
        Assert.Equal(100, budget.Snapshot().Single(p => p.Pool == "ram").Available);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CancellationOrShutdownWakesNextOwnerWithoutBudgetChange(bool shutdown)
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var external = budget.Reserve([new("ram", 50)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        int wakes = 0;
        using var next = SharedBudgetReclamation.Register(budget, () => wakes++);
        var request = Request("head"); var follower = Request("follower");
        Assert.Null(head.TryReserve(request, [new("ram", 100)]));
        Assert.Null(next.TryReserve(follower, [new("ram", 50)]));
        var signal = budget.ChangeSignal;
        int before = wakes;
        if (shutdown) head.Dispose(); else head.Cancel(request);
        Assert.True(wakes > before);
        Assert.False(signal.IsCompleted);
        using var admitted = next.TryReserve(follower, [new("ram", 50)]);
        Assert.NotNull(admitted);
    }

    [Fact]
    public void CancellationUsesRequestIdentityAndSuspensionRelinquishesPriority()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var external = budget.Reserve([new("ram", 100)]);
        using var first = SharedBudgetReclamation.Register(budget, () => { });
        using var second = SharedBudgetReclamation.Register(budget, () => { });
        var old = Request("reused"); var replacement = Request("reused"); var peer = Request("peer");
        Assert.Null(first.TryReserve(old, [new("ram", 100)]));
        first.Cancel(old);
        Assert.Null(first.TryReserve(replacement, [new("ram", 100)]));
        first.Cancel(old); // Delayed cancellation cannot erase the replacement's demand.
        Assert.Null(second.TryReserve(peer, [new("ram", 100)]));
        external.Dispose();
        Assert.Null(second.TryReserve(peer, [new("ram", 100)]));
        first.Clear(); // Closed compute gate yields its place.
        using var admitted = second.TryReserve(peer, [new("ram", 100)]);
        Assert.NotNull(admitted);
    }

    [Fact]
    public void ReadyOlderDemandDoesNotEvictExtraBytesForConflictingYoungerDemand()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        using var hostOwner = budget.Reserve([new("ram", 20)]);
        using var deviceOwner = budget.Reserve([new("gpu", 100)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        using var next = SharedBudgetReclamation.Register(budget, () => { });
        using var donor = SharedBudgetReclamation.Register(budget, () => { });
        Assert.Null(head.TryReserve(Request("uma"), [new("ram", 50), new("gpu", 50)]));
        Assert.Null(next.TryReserve(Request("larger-host"), [new("ram", 100)]));
        deviceOwner.Dispose();
        Assert.True(head.CanAdmit([new("ram", 50), new("gpu", 50)]));
        Assert.False(budget.CanReserve([new("ram", 100)]));
        Assert.Null(donor.TryBeginReclamation());
    }

    [Fact]
    public void ReclamationIsExclusiveBoundToDemandAndStopsImmediatelyOnCancellation()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var external = budget.Reserve([new("ram", 100)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        using var donorA = SharedBudgetReclamation.Register(budget, () => { });
        using var donorB = SharedBudgetReclamation.Register(budget, () => { });
        var request = Request("head");
        Assert.Null(head.TryReserve(request, [new("ram", 100)]));
        using var lease = donorA.TryBeginReclamation();
        Assert.NotNull(lease); Assert.True(lease.NeedsReclamation());
        Assert.Null(donorB.TryBeginReclamation());
        head.Cancel(request);
        Assert.False(lease.NeedsReclamation());
        lease.Dispose();
        Assert.Null(donorB.TryBeginReclamation());
        Assert.Equal(0, budget.Snapshot().Single().Available); // No physical owner was refunded.
    }

    [Fact]
    public void UnsuccessfulDonorAttemptsAreBoundedUntilPhysicalAvailabilityChanges()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var external = budget.Reserve([new("ram", 100)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        using var a = SharedBudgetReclamation.Register(budget, () => { });
        using var b = SharedBudgetReclamation.Register(budget, () => { });
        Assert.Null(head.TryReserve(Request("head"), [new("ram", 100)]));
        using (var lease = a.TryBeginReclamation()) Assert.NotNull(lease);
        for (int i = 0; i < 100; i++) Assert.Null(a.TryBeginReclamation());
        using (var lease = b.TryBeginReclamation()) Assert.NotNull(lease);
        Assert.Null(b.TryBeginReclamation());
        Assert.True(budget.TrySetCapacity("ram", 150));
        using var retry = a.TryBeginReclamation();
        Assert.NotNull(retry); Assert.True(retry.NeedsReclamation());
        external.Dispose();
        Assert.False(retry.NeedsReclamation());
    }

    [Fact]
    public void LaterIndependentDemandCanReclaimWithoutAnAvailabilityPulse()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        using var permanentHost = budget.Reserve([new("ram", 100)]);
        using var host = SharedBudgetReclamation.Register(budget, () => { });
        using var device = SharedBudgetReclamation.Register(budget, () => { });
        using var donor = SharedBudgetReclamation.Register(budget, () => { });
        Assert.Null(host.TryReserve(Request("unreclaimable-host"), [new("ram", 100)]));
        using (var first = donor.TryBeginReclamation()) Assert.NotNull(first);
        Assert.Null(donor.TryBeginReclamation());

        // A completed request can retain every allocated byte in its child.
        // Closing its empty envelope then causes no global availability pulse.
        var signal = budget.ChangeSignal;
        using var envelope = budget.Reserve([new("gpu", 100)]);
        using var retainedDeviceCache = envelope.TryTake([new("gpu", 100)]);
        Assert.NotNull(retainedDeviceCache);
        retainedDeviceCache.Commit();
        envelope.Dispose();
        Assert.Same(signal, budget.ChangeSignal);
        Assert.Null(device.TryReserve(Request("reclaimable-device"), [new("gpu", 100)]));

        using var second = donor.TryBeginReclamation();
        Assert.NotNull(second);
        Assert.True(second.NeedsReclamation());
        retainedDeviceCache.Dispose();
        Assert.False(second.NeedsReclamation()); // The lease serves GPU, not older RAM.
        Assert.False(host.CanAdmit([new("ram", 100)]));
        Assert.True(device.CanAdmit([new("gpu", 100)]));
    }

    [Fact]
    public void SwitchingIndependentDemandsDoesNotResetUnsuccessfulDonorAttempts()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        using var permanent = budget.Reserve([new("ram", 100), new("gpu", 100)]);
        using var host = SharedBudgetReclamation.Register(budget, () => { });
        using var device = SharedBudgetReclamation.Register(budget, () => { });
        using var a = SharedBudgetReclamation.Register(budget, () => { });
        using var b = SharedBudgetReclamation.Register(budget, () => { });
        var hostRequest = Request("host");
        var deviceRequest = Request("device");
        Assert.Null(host.TryReserve(hostRequest, [new("ram", 100)]));
        Assert.Null(device.TryReserve(deviceRequest, [new("gpu", 100)]));
        var signal = budget.ChangeSignal;

        // Each donor gets exactly one pass for each independent demand.
        foreach (var donor in new[] { a, b })
        {
            using (var first = donor.TryBeginReclamation()) Assert.NotNull(first);
            using (var second = donor.TryBeginReclamation()) Assert.NotNull(second);
        }
        for (int i = 0; i < 100; i++)
        {
            Assert.Null(a.TryBeginReclamation());
            Assert.Null(b.TryBeginReclamation());
        }
        Assert.Same(signal, budget.ChangeSignal);

        // Removing one demand cannot reset the other demand's attempt history.
        host.Cancel(hostRequest);
        Assert.Null(a.TryBeginReclamation());
        Assert.Null(b.TryBeginReclamation());
        device.Cancel(deviceRequest);
        Assert.Null(a.TryBeginReclamation());
    }

    [Fact]
    public void ExhaustedOlderAttemptDoesNotAllowReclamationForAConflictingFollower()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        using var permanent = budget.Reserve([new("ram", 100), new("gpu", 100)]);
        using var oldest = SharedBudgetReclamation.Register(budget, () => { });
        using var follower = SharedBudgetReclamation.Register(budget, () => { });
        using var donor = SharedBudgetReclamation.Register(budget, () => { });
        var firstRequest = Request("older-host");
        Assert.Null(oldest.TryReserve(firstRequest, [new("ram", 100)]));
        Assert.Null(follower.TryReserve(Request("younger-uma"), [new("ram", 1), new("gpu", 100)]));
        using (var first = donor.TryBeginReclamation()) Assert.NotNull(first);
        for (int i = 0; i < 100; i++) Assert.Null(donor.TryBeginReclamation());
        oldest.Cancel(firstRequest);
        using var next = donor.TryBeginReclamation();
        Assert.NotNull(next);
    }

    [Fact]
    public void RepeatedIndependentDemandCancellationAndReusedIdsKeepBoundedProgress()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        using var permanent = budget.Reserve([new("ram", 100), new("gpu", 100)]);
        using var host = SharedBudgetReclamation.Register(budget, () => { });
        using var device = SharedBudgetReclamation.Register(budget, () => { });
        using var a = SharedBudgetReclamation.Register(budget, () => { });
        using var b = SharedBudgetReclamation.Register(budget, () => { });
        Assert.Null(host.TryReserve(Request("permanent-host"), [new("ram", 100)]));
        using (var first = a.TryBeginReclamation()) Assert.NotNull(first);
        using (var first = b.TryBeginReclamation()) Assert.NotNull(first);
        var signal = budget.ChangeSignal;
        SequenceState? previous = null;

        for (int cycle = 0; cycle < 1000; cycle++)
        {
            var current = Request("reused-device-id");
            Assert.Null(device.TryReserve(current, [new("gpu", 100)]));
            if (previous != null) device.Cancel(previous);
            using (var first = a.TryBeginReclamation())
            {
                Assert.NotNull(first);
                Assert.True(first.NeedsReclamation());
                Assert.Null(b.TryBeginReclamation()); // Still one exclusive lease.
                if ((cycle & 1) == 0)
                {
                    device.Cancel(current);
                    Assert.False(first.NeedsReclamation());
                }
            }
            if ((cycle & 1) != 0)
            {
                using (var second = b.TryBeginReclamation()) Assert.NotNull(second);
                Assert.Null(a.TryBeginReclamation());
                Assert.Null(b.TryBeginReclamation());
                device.Cancel(current);
            }
            Assert.Null(a.TryBeginReclamation());
            Assert.Null(b.TryBeginReclamation());
            Assert.False(host.CanAdmit([new("ram", 100)]));
            Assert.Same(signal, budget.ChangeSignal);
            previous = current;
        }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(100, pool.Reserved));
    }

    [Fact]
    public void OwnPartialReleaseDoesNotCauseAnotherUnsuccessfulTrim()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var ownCache = budget.Reserve([new("ram", 20)]);
        using var external = budget.Reserve([new("ram", 80)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        using var donor = SharedBudgetReclamation.Register(budget, () => { });
        Assert.Null(head.TryReserve(Request("head"), [new("ram", 100)]));
        using (var lease = donor.TryBeginReclamation())
        {
            Assert.NotNull(lease);
            ownCache.Dispose();
            Assert.True(lease.NeedsReclamation());
        }
        for (int i = 0; i < 100; i++) Assert.Null(donor.TryBeginReclamation());
        external.Dispose();
        Assert.True(head.CanAdmit([new("ram", 100)]));
    }

    [Fact]
    public void CompletedBoundedBatchYieldsLeaseButCanContinueItsRemainingVictims()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var owner = budget.Reserve([new("ram", 100)]);
        using var head = SharedBudgetReclamation.Register(budget, () => { });
        using var donor = SharedBudgetReclamation.Register(budget, () => { });
        Assert.Null(head.TryReserve(Request("head"), [new("ram", 100)]));
        using (var lease = donor.TryBeginReclamation())
        {
            Assert.NotNull(lease);
            lease.MoreWork = true;
        }
        using var nextBatch = donor.TryBeginReclamation();
        Assert.NotNull(nextBatch);
    }
}
