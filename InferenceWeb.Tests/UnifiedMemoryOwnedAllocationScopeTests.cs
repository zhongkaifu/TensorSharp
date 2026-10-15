// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryOwnedAllocationScopeTests
{
    private sealed class Scope(Action? dispose = null) : IDisposable
    {
        public int Calls;
        public void Dispose() { Calls++; dispose?.Invoke(); }
    }

    private static MemoryBudget Budget() => new([new("device", 128)]);
    private static BudgetReservation Reserve(MemoryBudget budget) => budget.Reserve([new("device", 100)]);

    [Fact]
    public void ReturnedOwnerExistsBeforeTheFactoryPublishesAndNoAllocationFollowsItsReturn()
    {
        // Warm the helper before observing only its successful transaction.
        OwnedAllocationScope.Enter(new Scope(), () => new Scope()).Dispose();
        var root = new Scope(); var scope = new Scope();
        long atEntry = 0;
        Func<IDisposable> factory = () => { atEntry = GC.GetAllocatedBytesForCurrentThread(); return scope; };
        long before = GC.GetAllocatedBytesForCurrentThread();
        var owned = OwnedAllocationScope.Enter(root, factory);
        long after = GC.GetAllocatedBytesForCurrentThread();
        Assert.True(atEntry > before, "The final disposable owner must exist before entry publishes its scope.");
        Assert.Equal(atEntry, after);
        owned.Dispose();
        Assert.Equal(1, scope.Calls); Assert.Equal(1, root.Calls);
    }

    [Fact]
    public void FactoryOomRefundsTheUnpublishedRootAndPreservesTheFailure()
    {
        var budget = Budget(); using var root = Reserve(budget);
        var error = new OutOfMemoryException("injected factory failure before owner publication");
        Assert.Same(error, Assert.Throws<OutOfMemoryException>(() =>
            OwnedAllocationScope.Enter(root, () => throw error)));
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 0), Assert.Single(budget.Snapshot()));
        Assert.Throws<InvalidOperationException>(() => root.TryTake([new("device", 1)]));
    }

    [Fact]
    public void InvalidFactoryClosesTheRootWithoutPublishingAScope()
    {
        var budget = Budget(); using var root = Reserve(budget);
        Assert.Throws<ArgumentNullException>(() => OwnedAllocationScope.Enter(root, null!));
        Assert.Equal(0, Assert.Single(root.OwnedCharges).Bytes);
        Assert.Equal(128, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void MissingReturnedScopeClosesTheRootInsteadOfReturningAnIncompleteOwner()
    {
        var budget = Budget(); using var root = Reserve(budget);
        Assert.Throws<InvalidOperationException>(() => OwnedAllocationScope.Enter(root, () => null!));
        Assert.Equal(0, Assert.Single(root.OwnedCharges).Bytes);
        Assert.Equal(128, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void ScopeRestoresBeforeRootClosureAndLivePayloadChildrenRemainCharged()
    {
        var budget = Budget(); using var root = Reserve(budget);
        using var child = root.TryTake([new("device", 64)])!; child.Commit();
        bool restored = false;
        var scope = new Scope(() =>
        {
            Assert.Equal(100, Assert.Single(root.OwnedCharges).Bytes);
            Assert.Equal(new MemoryPoolSnapshot("device", 128, 36, 64), Assert.Single(budget.Snapshot()));
            restored = true;
        });
        var owned = OwnedAllocationScope.Enter(root, () => scope);
        owned.Dispose();
        Assert.True(restored);
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 64), Assert.Single(budget.Snapshot()));
        Assert.Equal(64, Assert.Single(root.OwnedCharges).Bytes);
        owned.Dispose(); Assert.Equal(1, scope.Calls);
        child.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 0), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void FailedScopeRestorationKeepsRootOpenAndAllowsTheSameOwnerToRetry()
    {
        var budget = Budget(); using var root = Reserve(budget);
        bool reject = true;
        var scope = new Scope(() => { if (reject) throw new InvalidOperationException("inner owner still active"); });
        var owned = OwnedAllocationScope.Enter(root, () => scope);
        Assert.Throws<InvalidOperationException>(() => owned.Dispose());
        Assert.Equal(100, Assert.Single(root.OwnedCharges).Bytes);
        using (var stillOpen = root.TryTake([new("device", 12)]))
            Assert.NotNull(stillOpen);
        Assert.Equal(100, Assert.Single(root.OwnedCharges).Bytes);
        reject = false; owned.Dispose(); owned.Dispose();
        Assert.Equal(2, scope.Calls);
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 0), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void RootDisposalFailureDoesNotRepeatAnAlreadyRestoredScopeOnRetry()
    {
        var order = new List<string>(); bool reject = true;
        var scope = new Scope(() => order.Add("scope"));
        var root = new Scope(() => { order.Add("root"); if (reject) throw new InvalidOperationException("retry root release"); });
        var owned = OwnedAllocationScope.Enter(root, () => scope);
        Assert.Throws<InvalidOperationException>(() => owned.Dispose());
        reject = false; owned.Dispose(); owned.Dispose();
        Assert.Equal(new[] { "scope", "root", "root" }, order);
        Assert.Equal(1, scope.Calls); Assert.Equal(2, root.Calls);
    }

    [Fact]
    public void SuccessfulDisposalAllocatesNothingAndIsIdempotent()
    {
        var root = new Scope(); var scope = new Scope();
        var owned = OwnedAllocationScope.Enter(root, () => scope);
        long before = GC.GetAllocatedBytesForCurrentThread();
        owned.Dispose(); owned.Dispose();
        long after = GC.GetAllocatedBytesForCurrentThread();
        Assert.Equal(0, after - before);
        Assert.Equal(1, scope.Calls); Assert.Equal(1, root.Calls);
    }

    [Fact]
    public void OptionalPhysicalRefusalPreservesExistingRootsAndCommittedChildren()
    {
        var budget = Budget(); using var existing = Reserve(budget);
        using var child = existing.TryTake([new("device", 64)])!; child.Commit();
        var before = budget.Snapshot().ToArray();
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("device", 20)], [new("device", 110)]));
        Assert.Null(OwnedAllocationScope.TryReserveOptional(budget, [new("device", 20)], [new("device", 110)]));
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(100, Assert.Single(existing.OwnedCharges).Bytes);
        Assert.Equal(64, Assert.Single(existing.LiveChildCharges).Bytes);
        using var stillOpen = existing.TryTake([new("device", 12)]);
        Assert.NotNull(stillOpen);
    }

    [Fact]
    public void OptionalLedgerContentionDoesNotRefundOrMutateOtherOwners()
    {
        var budget = Budget(); using var existing = Reserve(budget);
        using var child = existing.TryTake([new("device", 64)])!; child.Commit();
        var before = budget.Snapshot().ToArray();
        Assert.Null(OwnedAllocationScope.TryReserveOptional(budget, [new("device", 29)], [new("device", 128)]));
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(100, Assert.Single(existing.OwnedCharges).Bytes);
        Assert.Equal(64, Assert.Single(existing.LiveChildCharges).Bytes);
    }

    [Fact]
    public void SuccessfulOptionalOwnerClosesOnlyUnusedCreditAndItsPayloadRemainsCharged()
    {
        var budget = Budget();
        using var root = OwnedAllocationScope.TryReserveOptional(budget, [new("device", 100)], [new("device", 128)]);
        Assert.NotNull(root);
        using var child = root.TryTake([new("device", 64)])!; child.Commit();
        using (OwnedAllocationScope.Enter(root, () => new Scope())) { }
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 64), Assert.Single(budget.Snapshot()));
        Assert.Equal(64, Assert.Single(root.OwnedCharges).Bytes);
        child.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 0), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void OptionalReservationDoesNotHideInvalidPoolsOrOtherProgrammingErrors()
    {
        var budget = Budget(); var before = budget.Snapshot().ToArray();
        Assert.Throws<ArgumentException>(() => OwnedAllocationScope.TryReserveOptional(budget,
            [new("unknown", 1)], [new("device", 128)]));
        Assert.Throws<ArgumentException>(() => OwnedAllocationScope.TryReserveOptional(budget,
            [new("device", 1)], [new("unknown", 128)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => OwnedAllocationScope.TryReserveOptional(budget,
            [new("device", -1)], [new("device", 128)]));
        Assert.Throws<ArgumentNullException>(() => OwnedAllocationScope.TryReserveOptional(budget, null!, [new("device", 128)]));
        Assert.Throws<ArgumentNullException>(() => OwnedAllocationScope.TryReserveOptional(budget, [new("device", 1)], null!));
        Assert.Throws<ArgumentNullException>(() => OwnedAllocationScope.TryReserveOptional(null!, [new("device", 1)], [new("device", 128)]));
        Assert.Equal(before, budget.Snapshot().ToArray());
    }

    [Fact]
    public void AlreadyPaidPrimaryAllowsReleasingOnlyTheEmptyRequestSlotPromiseForACopy()
    {
        var budget = Budget();
        using var primary = budget.Reserve([new("device", 64)]); primary.Commit();
        using var request = budget.Reserve([new("device", 64)]);
        Assert.Null(OwnedAllocationScope.TryReserveOptional(budget, [new("device", 64)], [new("device", 128)]));
        Assert.True(OwnedAllocationScope.ReleaseUnusedPoolReservation(request, "device"));
        Assert.Equal(0, Assert.Single(request.OwnedCharges).Bytes);
        Assert.Equal(64, Assert.Single(primary.OwnedCharges).Bytes);
        using var copy = OwnedAllocationScope.TryReserveOptional(budget, [new("device", 64)], [new("device", 128)]);
        Assert.NotNull(copy);
        using var payload = copy.TryTake([new("device", 64)])!; payload.Commit(); copy.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 128), Assert.Single(budget.Snapshot()));
        payload.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 64), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void RequestAllocatedSlotAndPendingChildrenKeepTheirActualOwnership()
    {
        var budget = Budget(); using var request = Reserve(budget);
        using var slot = request.TryTake([new("device", 80)])!; slot.Commit();
        using var pending = request.TryTake([new("device", 10)])!;
        Assert.True(OwnedAllocationScope.ReleaseUnusedPoolReservation(request, "device"));
        Assert.Equal(90, Assert.Single(request.OwnedCharges).Bytes);
        Assert.Equal(90, Assert.Single(request.LiveChildCharges).Bytes);
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 10, 80), Assert.Single(budget.Snapshot()));
        Assert.Null(OwnedAllocationScope.TryReserveOptional(budget, [new("device", 39)], [new("device", 128)]));
        request.Dispose(); pending.Commit();
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 90), Assert.Single(budget.Snapshot()));
        slot.Dispose(); pending.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("device", 128, 0, 0), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void ReleasingDeviceSlackPreservesHostPromisesAndIsIdempotent()
    {
        var budget = new MemoryBudget([new("host", 512), new("device", 128)]);
        using var request = budget.Reserve([new("host", 300), new("device", 100)]);
        using var host = request.TryTake([new("host", 50)])!; host.Commit();
        using var device = request.TryTake([new("device", 40)])!; device.Commit();
        var hostBefore = budget.Snapshot().Single(pool => pool.Pool == "host");
        Assert.True(OwnedAllocationScope.ReleaseUnusedPoolReservation(request, "device"));
        Assert.Equal(hostBefore, budget.Snapshot().Single(pool => pool.Pool == "host"));
        Assert.Equal(300, request.OwnedCharges.Single(pool => pool.Pool == "host").Bytes);
        Assert.Equal(40, request.OwnedCharges.Single(pool => pool.Pool == "device").Bytes);
        var before = budget.Snapshot().ToArray();
        Assert.True(OwnedAllocationScope.ReleaseUnusedPoolReservation(request, "device"));
        Assert.Equal(before, budget.Snapshot().ToArray());
        using var usableHostPromise = request.TryTake([new("host", 12)]);
        Assert.NotNull(usableHostPromise);
    }

    [Fact]
    public void ReleasedLedgerCreditDoesNotOverridePhysicalRefusalOrInvalidOwnerErrors()
    {
        var budget = Budget();
        using var primary = budget.Reserve([new("device", 64)]); primary.Commit();
        using var request = budget.Reserve([new("device", 64)]);
        Assert.True(OwnedAllocationScope.ReleaseUnusedPoolReservation(request, "device"));
        Assert.Equal(64, Assert.Single(budget.Snapshot()).Available);
        var before = budget.Snapshot().ToArray();
        // The ledger has room, but all freshly observed physical capacity is
        // occupied by the existing primary. No imputed free bytes are granted.
        Assert.Throws<MemoryPressureException>(() => budget.TryReserve([new("device", 64)], [new("device", 64)]));
        Assert.Null(OwnedAllocationScope.TryReserveOptional(budget, [new("device", 64)], [new("device", 64)]));
        Assert.Throws<ArgumentException>(() => OwnedAllocationScope.ReleaseUnusedPoolReservation(request, "unknown"));
        Assert.Throws<ArgumentException>(() => OwnedAllocationScope.ReleaseUnusedPoolReservation(request, " "));
        Assert.Throws<ArgumentNullException>(() => OwnedAllocationScope.ReleaseUnusedPoolReservation(null!, "device"));
        Assert.Throws<InvalidOperationException>(() => OwnedAllocationScope.ReleaseUnusedPoolReservation(primary, "device"));
        Assert.Equal(before, budget.Snapshot().ToArray());
        request.Dispose();
        Assert.Throws<InvalidOperationException>(() => OwnedAllocationScope.ReleaseUnusedPoolReservation(request, "device"));
        Assert.Equal(before, budget.Snapshot().ToArray());
    }
}
