// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class MemoryBudgetEnvelopeGroupTests
{
    [Fact]
    public void SecondOwnerRefusalLeavesJointAndRequestUnchanged()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        using var joint = budget.Reserve([new("gpu", 20), new("ram", 20)]);
        using var request = budget.Reserve([new("gpu", 20), new("ram", 20)]);
        var before = budget.Snapshot().ToArray();
        // The joint's isolated increase fits; the combined growth does not.
        Assert.False(budget.TryResizeTotals([(joint, [new("gpu", 50), new("ram", 30)]),
            (request, [new("gpu", 80), new("ram", 30)])]));
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(20, joint.OwnedCharges.Single(c => c.Pool == "gpu").Bytes);
        Assert.Equal(20, request.OwnedCharges.Single(c => c.Pool == "gpu").Bytes);
    }

    [Fact]
    public void ReleasedRequestCreditFundsJointGrowthAndRetainedChildrenStayOwned()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        using var joint = budget.Reserve([new("gpu", 30), new("ram", 30)]);
        using var request = budget.Reserve([new("gpu", 70), new("ram", 70)]);
        using var pending = joint.TryTake([new("gpu", 10), new("ram", 10)])!;
        using var state = request.TryTake([new("gpu", 20), new("ram", 20)])!;
        state.Commit();
        var unchanged = budget.ChangeSignal;
        Assert.True(budget.TryResizeTotals([(joint, [new("gpu", 60), new("ram", 60)]),
            (request, [new("gpu", 40), new("ram", 40)])]));
        Assert.Same(unchanged, budget.ChangeSignal);
        Assert.False(unchanged.IsCompleted); // Global availability did not change.
        Assert.Equal(10, pending.OwnedCharges.Single(c => c.Pool == "gpu").Bytes);
        Assert.Equal(20, state.OwnedCharges.Single(c => c.Pool == "gpu").Bytes);
        Assert.Equal(50, joint.Charges.Single(c => c.Pool == "gpu").Bytes);
        Assert.Equal(20, request.Charges.Single(c => c.Pool == "gpu").Bytes);
        Assert.True(budget.TryResizeTotals([(joint, [new("gpu", 1), new("ram", 1)]),
            (request, [new("gpu", 1), new("ram", 1)])], [new("gpu", 0), new("ram", 0)]));
        Assert.True(unchanged.IsCompleted);
        Assert.Equal(10, joint.OwnedCharges.Single(c => c.Pool == "gpu").Bytes);
        Assert.Equal(20, request.OwnedCharges.Single(c => c.Pool == "gpu").Bytes);
        pending.Commit();
        Assert.Equal(30, budget.Snapshot().Single(c => c.Pool == "gpu").Committed);
    }

    [Fact]
    public void FreshConstraintsCoverCombinedGrowthAndUntouchedPools()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        using var joint = budget.Reserve([new("gpu", 20)]);
        using var request = budget.Reserve([new("gpu", 20)]);
        using var unrelated = budget.Reserve([new("ram", 60)]);
        var before = budget.Snapshot().ToArray();
        Assert.False(budget.TryResizeTotals([(joint, [new("gpu", 30)]), (request, [new("gpu", 30)])],
            [new("gpu", 59)]));
        Assert.False(budget.TryResizeTotals([(joint, [new("gpu", 30)]), (request, [new("gpu", 30)])],
            [new("gpu", 60), new("ram", 59)]));
        Assert.Equal(before, budget.Snapshot());
        Assert.True(budget.TryResizeTotals([(joint, [new("gpu", 30)]), (request, [new("gpu", 30)])],
            [new("gpu", 60), new("ram", 60)]));
    }

    [Fact]
    public void InvalidSecondOwnerAndThrowingTargetsCannotPartiallyChangeFirstOwner()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        var foreignBudget = new MemoryBudget([new("gpu", 100)]);
        using var joint = budget.Reserve([new("gpu", 20)]);
        using var request = budget.Reserve([new("gpu", 20)]);
        using var child = request.TryTake([new("gpu", 5)])!;
        using var foreign = foreignBudget.Reserve([new("gpu", 5)]);
        Assert.Throws<ArgumentException>(() => budget.TryResizeTotals([(joint, [new("gpu", 30)]),
            (foreign, [new("gpu", 30)])]));
        Assert.Throws<InvalidOperationException>(() => budget.TryResizeTotals([(joint, [new("gpu", 30)]),
            (child, [new("gpu", 5)])]));
        Assert.Throws<ArgumentException>(() => budget.TryResizeTotals([(joint, [new("gpu", 30)]),
            (request, [new("ram", 30)])]));
        Assert.Throws<ArgumentException>(() => budget.TryResizeTotals([(joint, [new("gpu", 30)]),
            (joint, [new("gpu", 40)])]));
        Assert.Throws<OutOfMemoryException>(() => budget.TryResizeTotals(FailingTargets(joint)));
        Assert.Equal(20, joint.OwnedCharges.Single().Bytes);
        Assert.Equal(20, request.OwnedCharges.Single().Bytes);
        Assert.Equal(40, budget.Snapshot().Single(c => c.Pool == "gpu").Reserved);
    }

    [Fact]
    public void ChangeNotificationRunsAfterBothOwnersCommitOutsideLedgerLock()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var joint = budget.Reserve([new("gpu", 20)]);
        using var request = budget.Reserve([new("gpu", 40)]);
        var signal = budget.ChangeSignal;
        var gate = typeof(MemoryBudget).GetField("_gate", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(budget)!;
        int notifications = 0;
        budget.BeforeChangeNotification = () =>
        {
            Assert.False(Monitor.IsEntered(gate));
            Assert.Equal(30, joint.OwnedCharges.Single().Bytes);
            Assert.Equal(20, request.OwnedCharges.Single().Bytes);
            notifications++;
        };
        Assert.True(budget.TryResizeTotals([(joint, [new("gpu", 30)]), (request, [new("gpu", 20)])]));
        budget.BeforeChangeNotification = null;
        Assert.Equal(1, notifications); Assert.True(signal.IsCompleted);
    }

    private static IEnumerable<(BudgetReservation, IReadOnlyList<MemoryCharge>)> FailingTargets(BudgetReservation first)
    {
        yield return (first, [new("gpu", 30)]);
        throw new OutOfMemoryException("Injected target enumeration failure");
    }
}
