// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class MemoryBudgetAllocationFailureTests
{
    private static MemoryBudget Budget() => new([new("ram", 1000), new("gpu", 1000)]);
    private static readonly MemoryCharge[] Both = [new("ram", 100), new("gpu", 200)];

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReservationObjectFailureLeavesEveryPoolAndWatermarkUnchanged(bool constrained)
    {
        var budget = Budget();
        var before = budget.Snapshot().ToArray();
        var high = budget.HighWatermarks().ToArray();
        Task signal = budget.ChangeSignal;
        budget.BeforeReservationAllocation = _ => throw new OutOfMemoryException("controlled owner allocation failure");
        Assert.Throws<OutOfMemoryException>(() => constrained
            ? budget.TryReserve(Both, [new("ram", 500), new("gpu", 500)]) : budget.TryReserve(Both));
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(high, budget.HighWatermarks());
        Assert.Same(signal, budget.ChangeSignal);
        Assert.False(signal.IsCompleted);
        budget.BeforeReservationAllocation = null;
        using (var retry = budget.Reserve(Both)) retry.Commit();
        Assert.Equal(before, budget.Snapshot());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ChildAllocationFailurePreservesEnvelopeAndAllowsRetry(bool failOwner)
    {
        var budget = Budget();
        using var envelope = budget.Reserve(Both);
        var originalCharges = envelope.Charges.ToArray();
        var before = budget.Snapshot().ToArray();
        var high = budget.HighWatermarks().ToArray();
        Task signal = budget.ChangeSignal;
        var target = failOwner ? MemoryBudget.ReservationAllocationPoint.Owner : MemoryBudget.ReservationAllocationPoint.ChildCharges;
        budget.BeforeReservationAllocation = point =>
        {
            if (point == target) throw new OutOfMemoryException("controlled child allocation failure");
        };
        Assert.Throws<OutOfMemoryException>(() => envelope.TryTake([new("ram", 60), new("gpu", 80)]));
        Assert.Equal(originalCharges, envelope.Charges);
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(high, budget.HighWatermarks());
        Assert.False(envelope.HasChildren);
        Assert.Same(signal, budget.ChangeSignal);
        Assert.False(signal.IsCompleted);
        budget.BeforeReservationAllocation = null;
        using (var retry = envelope.TryTake([new("ram", 60), new("gpu", 80)]))
        {
            Assert.NotNull(retry);
            retry.Commit();
        }
        Assert.Equal(originalCharges, envelope.Charges);
        Assert.Equal(before, budget.Snapshot());
        Assert.False(signal.IsCompleted); // Returned to its live owner, not global credit.
    }

    [Fact]
    public void FailureHookIsPerBudgetAndImpossibleRequestsNeverAllocateOwners()
    {
        var failing = Budget();
        var unaffected = Budget();
        int calls = 0;
        failing.BeforeReservationAllocation = _ => { calls++; throw new OutOfMemoryException("controlled"); };
        Assert.Null(failing.TryReserve([new("ram", 1001)]));
        Assert.Equal(0, calls);
        using var owner = unaffected.Reserve(Both);
        Assert.Throws<OutOfMemoryException>(() => failing.Reserve(Both));
        Assert.Equal(1, calls);
        Assert.All(failing.Snapshot(), pool => Assert.Equal(0, pool.Reserved));
    }

    [Fact]
    public void ReturningChildrenAcrossMultiplePoolsAllocatesNoManagedObjects()
    {
        var budget = Budget();
        using var envelope = budget.Reserve([new("ram", 500), new("gpu", 500)]);
        var before = budget.Snapshot().ToArray();
        var children = new BudgetReservation[65];
        for (int i = 0; i < children.Length; i++)
        {
            children[i] = envelope.TryTake([new("ram", 1), new("gpu", 2)])!;
            children[i].Commit();
        }
        children[0].Dispose(); // Warm the exact multi-pool release path before measuring.
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 1; i < children.Length; i++) children[i].Dispose();
        allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
        Assert.Equal(0, allocated);
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(new MemoryCharge[] { new("gpu", 500), new("ram", 500) }, envelope.Charges);
    }
}
