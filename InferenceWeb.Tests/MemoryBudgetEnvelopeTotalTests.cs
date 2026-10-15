// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class MemoryBudgetEnvelopeTotalTests
{
    [Fact]
    public void PhaseShrinkCountsPendingAndCommittedChildrenWithoutRevokingEither()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var request = budget.Reserve([new("gpu", 90)]);
        using var pending = request.TryTake([new("gpu", 20)])!;
        using var state = request.TryTake([new("gpu", 30)])!;
        state.Commit();
        Assert.Equal(50, Bytes(request.LiveChildCharges));
        Assert.True(request.TryResizeTotal([new("gpu", 60)]));
        Assert.Equal(10, Bytes(request.Charges));
        Assert.Equal(60, Bytes(request.OwnedCharges));
        Assert.True(request.TryResizeTotal([new("gpu", 10)]));
        Assert.Equal(0, Bytes(request.Charges));
        Assert.Equal(50, Bytes(request.OwnedCharges));
        pending.Commit();
        Assert.Equal(new MemoryPoolSnapshot("gpu", 100, 0, 50), Assert.Single(budget.Snapshot()));
        state.Dispose();
        Assert.Equal(20, Bytes(request.LiveChildCharges));
        Assert.Equal(30, Bytes(request.Charges));
        Assert.True(request.TryResizeTotal([new("gpu", 20)]));
        Assert.Equal(80, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void NestedChildrenRemainOwnedThroughClosedIntermediateAndRootEnvelopes()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var root = budget.Reserve([new("gpu", 90)]);
        using var middle = root.TryTake([new("gpu", 70)])!;
        using var child = middle.TryTake([new("gpu", 40)])!;
        using var leaf = child.TryTake([new("gpu", 20)])!;
        leaf.Commit();
        Assert.Equal(70, Bytes(root.LiveChildCharges));
        Assert.Equal(40, Bytes(middle.LiveChildCharges));
        Assert.Equal(20, Bytes(child.LiveChildCharges));
        middle.Dispose(); // Returns only its unused 30 to the open root.
        Assert.Equal(40, Bytes(root.LiveChildCharges));
        Assert.Equal(50, Bytes(root.Charges));
        Assert.Equal(40, Bytes(middle.OwnedCharges));
        child.Dispose(); // Closed immediate parent: unused 20 goes to the pool.
        Assert.Equal(20, Bytes(root.LiveChildCharges));
        Assert.Equal(20, Bytes(middle.LiveChildCharges));
        Assert.Equal(70, Bytes(root.OwnedCharges));
        Assert.True(root.TryResizeTotal([new("gpu", 30)]));
        Assert.Equal(10, Bytes(root.Charges));
        root.Dispose();
        Assert.Equal(20, Bytes(root.OwnedCharges));
        leaf.Dispose();
        Assert.Equal(0, Bytes(root.OwnedCharges));
        Assert.Equal(0, Bytes(middle.OwnedCharges));
        Assert.Equal(0, Bytes(child.OwnedCharges));
        Assert.Equal(100, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void LiveNestedRefundDoesNotChangeAncestorOwnership()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var root = budget.Reserve([new("gpu", 80)]);
        using var middle = root.TryTake([new("gpu", 50)])!;
        using var child = middle.TryTake([new("gpu", 25)])!;
        child.Commit(); child.Dispose();
        Assert.Equal(50, Bytes(root.LiveChildCharges));
        Assert.Equal(80, Bytes(root.OwnedCharges));
        Assert.Equal(0, Bytes(middle.LiveChildCharges));
        Assert.Equal(50, Bytes(middle.Charges));
        Assert.True(root.TryResizeTotal([new("gpu", 50)]));
        middle.Dispose();
        Assert.Equal(50, Bytes(root.Charges));
        Assert.Equal(0, Bytes(root.LiveChildCharges));
    }

    [Fact]
    public void TakingFromRetainedChildAfterRootCloseStillUpdatesLiveOwnership()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var root = budget.Reserve([new("gpu", 80)]);
        using var retained = root.TryTake([new("gpu", 40)])!;
        root.Dispose();
        using var payload = retained.TryTake([new("gpu", 25)])!;
        payload.Commit();
        Assert.Equal(40, Bytes(root.OwnedCharges));
        retained.Dispose();
        Assert.Equal(25, Bytes(root.OwnedCharges));
        payload.Dispose();
        Assert.Equal(0, Bytes(root.OwnedCharges));
        Assert.Equal(100, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void MultiPoolGrowthUsesTotalNotRemainingAndFailsAtomically()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        using var request = budget.Reserve([new("gpu", 60), new("ram", 60)]);
        using var child = request.TryTake([new("gpu", 20), new("ram", 30)])!;
        child.Commit();
        using var rival = budget.Reserve([new("gpu", 20), new("ram", 30)]);
        var before = budget.Snapshot().ToArray();
        var peaks = budget.HighWatermarks().ToArray();
        Assert.False(request.TryResizeTotal([new("gpu", 70), new("ram", 80)]));
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(peaks, budget.HighWatermarks());
        Assert.False(request.TryResizeTotal([new("gpu", 70), new("ram", 60)],
            [new("gpu", 89), new("ram", 100)]));
        Assert.True(request.TryResizeTotal([new("gpu", 70), new("ram", 60)],
            [new("gpu", 100), new("gpu", 90), new("ram", 90)]));
        Assert.Equal(new[] { new MemoryCharge("gpu", 70), new MemoryCharge("ram", 60) }, request.OwnedCharges);
    }

    [Fact]
    public void InvalidTargetAndFailedEnumerationLeaveEveryOwnershipViewUntouched()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        using var request = budget.Reserve([new("gpu", 60)]);
        using var child = request.TryTake([new("gpu", 20)])!;
        Assert.Throws<InvalidOperationException>(() => child.TryResizeTotal([new("gpu", 10)]));
        Assert.Throws<ArgumentException>(() => request.TryResizeTotal([new("ram", 20)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => request.TryResizeTotal([new("gpu", -1)]));
        Assert.Throws<OutOfMemoryException>(() => request.TryResizeTotal(Failing()));
        Assert.Equal(60, Bytes(request.OwnedCharges));
        Assert.Equal(20, Bytes(request.LiveChildCharges));
        Assert.Equal(40, Bytes(request.Charges));
        request.Dispose();
        Assert.Throws<InvalidOperationException>(() => request.TryResizeTotal([new("gpu", 60)]));
        using var committed = budget.Reserve([new("gpu", 10)]);
        committed.Commit();
        Assert.Throws<InvalidOperationException>(() => committed.TryResizeTotal([new("gpu", 10)]));
        static IEnumerable<MemoryCharge> Failing() { yield return new("gpu", 1); throw new OutOfMemoryException("injected"); }
    }

    [Fact]
    public void PhysicalRefreshAllowsShrinkingBelowChildrenAndBoundsRecovery()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var request = budget.Reserve([new("gpu", 80)]);
        using var state = request.TryTake([new("gpu", 40)])!;
        state.Commit();
        budget.RefreshCapacities([new("gpu", 10)]);
        var signal = budget.ChangeSignal;
        Assert.True(request.TryResizeTotal([new("gpu", 0)], [new("gpu", 0)]));
        Assert.True(signal.IsCompleted);
        Assert.Equal(40, Bytes(request.OwnedCharges));
        Assert.False(request.TryResizeTotal([new("gpu", 41)]));
        state.Dispose();
        Assert.True(request.TryResizeTotal([new("gpu", 10)]));
        Assert.Equal(10, Assert.Single(budget.Snapshot()).Capacity);
        Assert.False(request.TryResizeTotal([new("gpu", 11)]));
    }

    [Fact]
    public async Task TotalResizeAndNestedRefundsStayAtomicUnderCompetingUmaOwners()
    {
        var budget = new MemoryBudget([new("gpu", 1000), new("ram", 1000)]);
        using var root = budget.Reserve([new("gpu", 500), new("ram", 500)]);
        var work = Task.Run(() =>
        {
            for (int i = 0; i < 1000; i++)
            {
                using var child = root.TryTake([new("gpu", 20), new("ram", 20)]);
                if (child == null) continue;
                using var leaf = child.TryTake([new("gpu", 10), new("ram", 10)])!;
                leaf.Commit();
                if (i % 2 == 0) child.Dispose();
                AssertSymmetric(root.OwnedCharges); AssertSymmetric(root.LiveChildCharges);
            }
        });
        var rivalWork = Task.Run(() =>
        {
            for (int i = 0; i < 1000; i++)
            {
                using var rival = budget.TryReserve([new("gpu", 500), new("ram", 500)]);
                rival?.Commit();
            }
        });
        for (int i = 0; i < 1000; i++)
        {
            long target = i % 2 == 0 ? 200 : 500;
            root.TryResizeTotal([new("gpu", target), new("ram", target)]);
            AssertSymmetric(root.OwnedCharges);
            var snapshot = budget.Snapshot();
            Assert.Equal(snapshot[0].Reserved, snapshot[1].Reserved);
            Assert.Equal(snapshot[0].Committed, snapshot[1].Committed);
            Assert.All(snapshot, p => Assert.InRange(p.Available, 0, 1000));
        }
        await Task.WhenAll(work, rivalWork);
        Assert.All(root.LiveChildCharges, c => Assert.Equal(0, c.Bytes));
        root.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(1000, p.Available));
        static void AssertSymmetric(IReadOnlyList<MemoryCharge> c) => Assert.Equal(c[0].Bytes, c[1].Bytes);
    }

    private static long Bytes(IReadOnlyList<MemoryCharge> c) => Assert.Single(c).Bytes;
}
