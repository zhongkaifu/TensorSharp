// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Collections.Concurrent;
using TensorSharp.Memory;
using TensorSharp.Memory.Hardware;
using TensorSharp.Memory.Planning;

namespace InferenceWeb.Tests;

public sealed class MemoryBudgetAdmissionConstraintTests
{
    [Fact]
    public void PhysicalAdviceRejectsLargeRequestWithoutRewritingBorrowedBudget()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        var before = budget.CapacityTargets().ToArray();
        Task signal = budget.ChangeSignal;
        var now = DateTimeOffset.UtcNow;
        var plan = new PhysicalMemoryCapacityPlanner(new() { MinimumBytes = 0, FutureBurstBytes = 0 }).Plan(
            [new("host", "ram", "physical-ram", PhysicalMemoryMetric.HostAvailable, 100, 10, now)],
            budget.Snapshot(), now);
        Assert.True(plan.Accepted);
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("ram", 50)], plan.DesiredCapacities));
        Assert.Equal(before, budget.CapacityTargets());
        Assert.False(signal.IsCompleted);
        using var small = budget.TryReserve([new("ram", 10)], plan.DesiredCapacities);
        Assert.NotNull(small);
        Assert.Equal(new MemoryPoolSnapshot("ram", 100, 10, 0), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void DuplicateConstraintsTakeMinimumWhileDuplicateChargesAdd()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var previous = budget.Reserve([new("ram", 10)]);
        previous.Commit();
        MemoryCharge[] constraints = [new("ram", 80), new("ram", 60), new("ram", 70)];
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("ram", 20), new("ram", 31)], constraints));
        using var exact = budget.TryReserve([new("ram", 20), new("ram", 30)], constraints);
        Assert.NotNull(exact);
        Assert.Equal(new MemoryPoolSnapshot("ram", 100, 50, 10), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void ExistingOwnersAdmittedAfterSamplingAreCheckedAgainAtomically()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        MemoryCharge[] sampled = [new("ram", 80)];
        using var laterOwner = budget.Reserve([new("ram", 60)]);
        laterOwner.Commit();
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("ram", 30)], sampled));
        Assert.Equal(new MemoryPoolSnapshot("ram", 100, 0, 60), Assert.Single(budget.Snapshot()));
        using var exact = budget.TryReserve([new("ram", 20)], sampled);
        Assert.NotNull(exact);
        // This per-call gate does not silently reduce the caller's unrelated allocations.
        using var unrelated = budget.Reserve([new("ram", 20)]);
        Assert.Equal(100, Assert.Single(budget.CapacityTargets()).Ceiling);
    }

    [Fact]
    public void MultiPoolConstraintRefusalDoesNotPartiallyReserve()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        var before = budget.Snapshot().ToArray();
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve(
            [new("ram", 40), new("gpu", 40)], [new("ram", 50), new("gpu", 39)]));
        Assert.Equal(before, budget.Snapshot());
        // Separate UMA constraints describe the same allocation, not a combined capacity.
        using var owner = budget.TryReserve([new("ram", 40), new("gpu", 40)],
            [new("ram", 50), new("gpu", 40)]);
        Assert.NotNull(owner);
        Assert.All(budget.Snapshot(), p => Assert.Equal(40, p.Reserved));
    }

    [Fact]
    public void MorePermissivePhysicalAdviceCannotBypassLedgerOrPersistentTarget()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        budget.RefreshCapacities([new("ram", 30)]);
        Assert.Null(budget.TryReserve([new("ram", 31)], [new("ram", 1000)]));
        Assert.Null(budget.TryReserve([new("ram", 31)], []));
        Assert.Equal(new MemoryPoolCapacityTarget("ram", 100, 30, 30, 0), Assert.Single(budget.CapacityTargets()));
    }

    [Fact]
    public void SubmissionCeilingAllowsFreshRecoveryCheckWithoutAuthorizingStaleCredit()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var live = budget.Reserve([new("ram", 50)]);
        live.Commit();
        budget.RefreshCapacities([new("ram", 10)]);
        Assert.True(budget.CanEverFitConfiguredCeiling([new("ram", 80)]));
        Assert.False(budget.CanEverFit([new("ram", 80)]));
        Assert.True(budget.CanEverFit([new("ram", 40)])); // Effective owner floor is not free capacity.
        Assert.Null(budget.TryReserve([new("ram", 40)]));
        Assert.False(budget.CanEverFitConfiguredCeiling([new("ram", 60), new("ram", 41)]));
        Assert.Equal(new MemoryPoolCapacityTarget("ram", 100, 10, 50, 40), Assert.Single(budget.CapacityTargets()));
        budget.RefreshCapacities([new("ram", 100)]); // Fresh recovery evidence supplied by the adapter.
        using var recovered = budget.TryReserve([new("ram", 40)], [new("ram", 100)]);
        Assert.NotNull(recovered);
    }

    [Fact]
    public void IsolationCheckRejectsImpossiblePhysicalPeakButKeepsOrdinaryContentionDistinct()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var owner = budget.Reserve([new("ram", 50)]);
        owner.Commit();
        MemoryCharge[] constraints = [new("ram", 90), new("ram", 50)];
        Assert.True(budget.CanEverFit([new("ram", 80)]));
        Assert.False(budget.CanEverFit([new("ram", 80)], constraints));
        Assert.True(budget.CanEverFit([new("ram", 40)], constraints));
        // Fitting alone does not bypass the existing owner during actual reservation.
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("ram", 40)], constraints));
        Assert.Equal(new MemoryPoolSnapshot("ram", 100, 0, 50), Assert.Single(budget.Snapshot()));
        Assert.Throws<ArgumentException>(() => budget.CanEverFit([new("ram", 1)], [new("unknown", 100)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => budget.CanEverFit([new("ram", 1)], [new("ram", -1)]));
    }

    [Fact]
    public void ConfiguredReadinessExposesPhysicalRefusalWithoutBypassingLiveOwners()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var owner = budget.Reserve([new("ram", 50)]);
        owner.Commit();
        budget.RefreshCapacities([new("ram", 50)]);
        Assert.True(budget.CanReserveConfiguredCeiling([new("ram", 40)]));
        Assert.False(budget.CanReserve([new("ram", 40)]));
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("ram", 40)], [new("ram", 50)]));
        Assert.False(budget.CanReserveConfiguredCeiling([new("ram", 30), new("ram", 21)]));
        Assert.Null(budget.TryReserve([new("ram", 40)])); // The hint never grants credit.
        Assert.Equal(new MemoryPoolSnapshot("ram", 50, 0, 50), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void OverTargetOwnerIsNotRevokedEvenWhenNewRequestDoesNotChargeItsPool()
    {
        var budget = new MemoryBudget([new("ram", 100), new("ssd", 100)]);
        using var envelope = budget.Reserve([new("ram", 70)]);
        using var child = envelope.TryTake([new("ram", 60)])!;
        child.Commit();
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("ssd", 1)], [new("ram", 50)]));
        Assert.Equal(new MemoryPoolSnapshot("ram", 100, 10, 60), budget.Snapshot()[0]);
        child.Dispose();
        using var reused = envelope.TryTake([new("ram", 60)]);
        Assert.NotNull(reused);
        Assert.Equal(0, budget.Snapshot()[1].Reserved);
    }

    [Fact]
    public void InvalidConstraintSetCannotReserveOrAlterTargets()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        var before = budget.CapacityTargets().ToArray();
        Assert.Throws<ArgumentException>(() => budget.TryReserve([new("ram", 1)], [new("ram", 50), new("missing", 100)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => budget.TryReserve([new("ram", 1)], [new("ram", -1)]));
        Assert.Throws<OverflowException>(() => budget.TryReserve([new("ram", long.MaxValue), new("ram", 1)], [new("ram", 100)]));
        Assert.Equal(before, budget.CapacityTargets());
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Reserved);
    }

    [Fact]
    public async Task ConcurrentEqualCeilingConstraintsWaitForLedgerCreditThenRetry()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var bothObserved = new Barrier(2);
        var owners = new ConcurrentBag<BudgetReservation>();
        int queued = 0;
        Task[] attempts = Enumerable.Range(0, 2).Select(_ => Task.Run(() =>
        {
            // Both engines observed readiness before either acquired credit.
            Assert.True(budget.CanReserveConfiguredCeiling([new("ram", 60)]));
            Assert.True(bothObserved.SignalAndWait(TimeSpan.FromSeconds(10)));
            // The losing caller must return null, not report terminal physical
            // pressure. A real owner's release is the correct wakeup mechanism.
            var reservation = budget.TryReserve([new("ram", 60)], [new("ram", 100)]);
            if (reservation == null) Interlocked.Increment(ref queued);
            else owners.Add(reservation);
        })).ToArray();
        Task released = Task.CompletedTask;
        try
        {
            await Task.WhenAll(attempts);
            Assert.Single(owners);
            Assert.Equal(1, queued);
            Assert.Equal(new MemoryPoolSnapshot("ram", 100, 60, 0), Assert.Single(budget.Snapshot()));
            released = budget.ChangeSignal;
            Assert.False(released.IsCompleted);
        }
        finally { foreach (var owner in owners) owner.Dispose(); }
        Assert.True(released.IsCompleted);
        using var retry = budget.TryReserve([new("ram", 60)], [new("ram", 100)]);
        Assert.NotNull(retry);
    }

    [Fact]
    public async Task ConcurrentAdmissionsCannotBothUseTheSamePhysicalSlack()
    {
        var budget = new MemoryBudget([new("ram", 1000)]);
        using var start = new ManualResetEventSlim();
        var owners = new ConcurrentBag<BudgetReservation>();
        int refused = 0;
        Task[] tasks = Enumerable.Range(0, 16).Select(_ => Task.Run(() =>
        {
            Assert.True(start.Wait(TimeSpan.FromSeconds(10)));
            try
            {
                var reservation = budget.TryReserve([new("ram", 60)], [new("ram", 100)]);
                Assert.NotNull(reservation);
                owners.Add(reservation);
            }
            catch (MemoryPressureException) { Interlocked.Increment(ref refused); }
        })).ToArray();
        start.Set();
        try
        {
            await Task.WhenAll(tasks);
            Assert.Single(owners);
            Assert.Equal(15, refused);
            Assert.Equal(new MemoryPoolSnapshot("ram", 1000, 60, 0), Assert.Single(budget.Snapshot()));
        }
        finally { foreach (var owner in owners) owner.Dispose(); }
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Reserved);
    }
}
