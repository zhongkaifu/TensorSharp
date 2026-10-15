// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class MemoryBudgetRefreshTests
{
    [Fact]
    public void ReleaseKeepsShrinkingToDesiredTargetInsteadOfReopeningCredit()
    {
        var budget = new MemoryBudget([new("ram", 200)]);
        using var first = budget.Reserve([new("ram", 80)]);
        using var second = budget.Reserve([new("ram", 70)]);
        first.Commit(); second.Commit();
        var target = Assert.Single(budget.RefreshCapacities([new("ram", 40)]));
        Assert.Equal(new MemoryPoolCapacityTarget("ram", 200, 40, 150, 110), target);
        Assert.Null(budget.TryReserve([new("ram", 1)]));
        first.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("ram", 70, 0, 70), Assert.Single(budget.Snapshot()));
        Assert.Null(budget.TryReserve([new("ram", 1)]));
        second.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("ram", 40, 0, 0), Assert.Single(budget.Snapshot()));
        using var next = budget.Reserve([new("ram", 40)]);
        Assert.Null(budget.TryReserve([new("ram", 1)]));
    }

    [Fact]
    public void ChildrenKeepTheirEnvelopeCreditDuringRefreshAndOnlyFreeAfterOwnerRelease()
    {
        var budget = new MemoryBudget([new("ram", 200), new("gpu", 200)]);
        using var envelope = budget.Reserve([new("ram", 100), new("gpu", 120)]);
        using var child = envelope.TryTake([new("ram", 60), new("gpu", 80)])!;
        child.Commit();
        budget.RefreshCapacities([new("ram", 20), new("gpu", 30)]);
        child.Dispose(); // credit returns to the live request, not the pool
        Assert.Equal(new MemoryPoolSnapshot("ram", 100, 100, 0), budget.Snapshot()[0]);
        Assert.Equal(new MemoryPoolSnapshot("gpu", 120, 120, 0), budget.Snapshot()[1]);
        using var reused = envelope.TryTake([new("ram", 60), new("gpu", 80)])!;
        Assert.NotNull(reused);
        reused.Commit();
        envelope.Dispose(); // only unused request credit is now released
        Assert.Equal(new MemoryPoolSnapshot("ram", 60, 0, 60), budget.Snapshot()[0]);
        Assert.Equal(new MemoryPoolSnapshot("gpu", 80, 0, 80), budget.Snapshot()[1]);
        reused.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("ram", 20, 0, 0), budget.Snapshot()[0]);
        Assert.Equal(new MemoryPoolSnapshot("gpu", 30, 0, 0), budget.Snapshot()[1]);
    }

    [Fact]
    public void RecoveryHonorsConfiguredCeilingAndExplicitSetterClearsTarget()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        budget.RefreshCapacities([new("ram", 10)]);
        Assert.Equal(100, Assert.Single(budget.RefreshCapacities([new("ram", long.MaxValue)])).Effective);
        budget.RefreshCapacities([new("ram", 10)]);
        Assert.True(budget.TrySetCapacity("ram", 70));
        Assert.Equal(new MemoryPoolCapacityTarget("ram", 70, 70, 70, 0), Assert.Single(budget.CapacityTargets()));
        Assert.Equal(70, Assert.Single(budget.RefreshCapacities([new("ram", 200)])).Effective);
    }

    [Fact]
    public void InvalidMultiPoolRefreshIsTransactionalAndFailedExplicitSetKeepsTarget()
    {
        var budget = new MemoryBudget([new("ram", 100), new("gpu", 100)]);
        using var owner = budget.Reserve([new("ram", 80)]);
        budget.RefreshCapacities([new("ram", 10), new("gpu", 20)]);
        var before = budget.CapacityTargets().ToArray();
        Assert.Throws<ArgumentException>(() => budget.RefreshCapacities([new("ram", 90), new("unknown", 20)]));
        Assert.Throws<ArgumentException>(() => budget.RefreshCapacities([new("gpu", 30), new("gpu", 40)]));
        Assert.False(budget.TrySetCapacities([new("ram", 70), new("gpu", 90)]));
        Assert.Equal(before, budget.CapacityTargets());
        owner.Dispose();
        Assert.Equal(10, budget.Snapshot().Single(p => p.Pool == "ram").Capacity);
    }

    [Fact]
    public async Task RefreshAccountsForOwnersAdmittedAfterTheObservationWithoutRevokingThem()
    {
        var budget = new MemoryBudget([new("ram", 1000)]);
        using var admitted = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var worker = Task.Run(() =>
        {
            using var owner = budget.Reserve([new("ram", 900)]);
            owner.Commit();
            admitted.Set();
            Assert.True(release.Wait(TimeSpan.FromSeconds(10)));
        });
        try
        {
            Assert.True(admitted.Wait(TimeSpan.FromSeconds(10)));
            budget.RefreshCapacities([new("ram", 50)]);
            Assert.Equal(new MemoryPoolSnapshot("ram", 900, 0, 900), Assert.Single(budget.Snapshot()));
            Assert.Null(budget.TryReserve([new("ram", 1)]));
        }
        finally { release.Set(); }
        await worker;
        Assert.Equal(new MemoryPoolSnapshot("ram", 50, 0, 0), Assert.Single(budget.Snapshot()));
    }
}
