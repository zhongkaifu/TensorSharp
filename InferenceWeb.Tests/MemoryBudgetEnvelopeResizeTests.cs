// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class MemoryBudgetEnvelopeResizeTests
{
    [Fact]
    public void ShrinkPreservesLiveChildrenAndReturnsTheirCreditToTheSameOwner()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var envelope = budget.Reserve([new("gpu", 80)]);
        using var child = envelope.TryTake([new("gpu", 30)])!;
        child.Commit();
        Task signal = budget.ChangeSignal;
        Assert.True(envelope.TryResizeRemaining([new("gpu", 10)]));
        Assert.True(signal.IsCompleted);
        Assert.Equal(new MemoryPoolSnapshot("gpu", 100, 10, 30), Assert.Single(budget.Snapshot()));
        child.Dispose();
        Assert.Equal(new MemoryCharge("gpu", 40), Assert.Single(envelope.Charges));
        Assert.Equal(new MemoryPoolSnapshot("gpu", 100, 40, 0), Assert.Single(budget.Snapshot()));
        envelope.Dispose();
        Assert.Equal(100, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void ClosingResizedEnvelopeKeepsItsChildrenPhysicallyCharged()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var envelope = budget.Reserve([new("gpu", 60)]);
        using var child = envelope.TryTake([new("gpu", 25)])!;
        child.Commit();
        Assert.True(envelope.TryResizeRemaining([new("gpu", 50)]));
        envelope.Dispose();
        Assert.Equal(new MemoryPoolSnapshot("gpu", 100, 0, 25), Assert.Single(budget.Snapshot()));
        child.Dispose();
        Assert.Equal(100, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void FailedMultiPoolGrowthChangesNeitherPoolOwnerWatermarkNorSignal()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        using var envelope = budget.Reserve([new("gpu", 20), new("ram", 20)]);
        using var rival = budget.Reserve([new("ram", 50)]);
        var before = budget.Snapshot().ToArray();
        var charges = envelope.Charges.ToArray();
        var watermarks = budget.HighWatermarks().ToArray();
        Task signal = budget.ChangeSignal;
        Assert.False(envelope.TryResizeRemaining([new("gpu", 80), new("ram", 60)]));
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(charges, envelope.Charges);
        Assert.Equal(watermarks, budget.HighWatermarks());
        Assert.Same(signal, budget.ChangeSignal);
        Assert.False(signal.IsCompleted);
    }

    [Fact]
    public void AbsolutePhysicalLimitsCountLiveChildrenAndUnrelatedPools()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        using var envelope = budget.Reserve([new("gpu", 50)]);
        using var child = envelope.TryTake([new("gpu", 20)])!;
        child.Commit();
        using var host = budget.Reserve([new("ram", 30)]);
        Assert.False(envelope.TryResizeRemaining([new("gpu", 40)], [new("gpu", 59)]));
        Assert.False(envelope.TryResizeRemaining([new("gpu", 40)], [new("gpu", 80), new("ram", 29)]));
        Assert.True(envelope.TryResizeRemaining([new("gpu", 40)], [new("gpu", 80), new("gpu", 60), new("ram", 30)]));
        Assert.Equal(40, Assert.Single(envelope.Charges).Bytes);
    }

    [Fact]
    public void ShrinkRemainsPossibleBelowPhysicalTargetAndCannotRegrowPastIt()
    {
        var budget = new MemoryBudget([new("gpu", 100)]);
        using var envelope = budget.Reserve([new("gpu", 80)]);
        using var child = envelope.TryTake([new("gpu", 30)])!;
        child.Commit();
        budget.RefreshCapacities([new("gpu", 20)]);
        Assert.True(envelope.TryResizeRemaining([new("gpu", 0)], [new("gpu", 10)]));
        Assert.Equal(new MemoryPoolSnapshot("gpu", 30, 0, 30), Assert.Single(budget.Snapshot()));
        Assert.False(envelope.TryResizeRemaining([new("gpu", 1)], [new("gpu", 1000)]));
        child.Dispose();
        Assert.True(envelope.TryResizeRemaining([new("gpu", 0)]));
        Assert.Equal(20, Assert.Single(budget.Snapshot()).Capacity);
        Assert.True(envelope.TryResizeRemaining([new("gpu", 20)]));
        Assert.False(envelope.TryResizeRemaining([new("gpu", 21)]));
    }

    [Fact]
    public void InvalidPoolsAndClosedCommittedOrChildOwnersCannotResize()
    {
        var budget = new MemoryBudget([new("gpu", 100), new("ram", 100)]);
        using var envelope = budget.Reserve([new("gpu", 30)]);
        using var child = envelope.TryTake([new("gpu", 10)])!;
        Assert.Throws<InvalidOperationException>(() => child.TryResizeRemaining([new("gpu", 10)]));
        Assert.Throws<ArgumentException>(() => envelope.TryResizeRemaining([]));
        Assert.Throws<ArgumentException>(() => envelope.TryResizeRemaining([new("ram", 20)]));
        Assert.Throws<ArgumentException>(() => envelope.TryResizeRemaining([new("gpu", 20), new("ram", 0)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => envelope.TryResizeRemaining([new("gpu", -1)]));
        Assert.Equal(20, Assert.Single(envelope.Charges).Bytes);
        child.Dispose();
        envelope.Dispose();
        Assert.Throws<InvalidOperationException>(() => envelope.TryResizeRemaining([new("gpu", 0)]));
        using var committed = budget.Reserve([new("gpu", 10)]);
        committed.Commit();
        Assert.Throws<InvalidOperationException>(() => committed.TryResizeRemaining([new("gpu", 0)]));
    }

    [Fact]
    public void DuplicateChargesNormalizeAndEnumerationFailureCannotPartiallyResize()
    {
        var budget = new MemoryBudget([new("gpu", long.MaxValue)]);
        using var envelope = budget.Reserve([new("gpu", 0)]);
        Assert.True(envelope.TryResizeRemaining([new("gpu", 30), new("gpu", 20)]));
        Assert.Equal(50, Assert.Single(envelope.Charges).Bytes);
        Assert.Throws<OverflowException>(() => envelope.TryResizeRemaining([new("gpu", long.MaxValue), new("gpu", 1)]));
        Assert.Throws<OutOfMemoryException>(() => envelope.TryResizeRemaining(Failing()));
        Assert.Equal(50, Assert.Single(budget.Snapshot()).Reserved);
        static IEnumerable<MemoryCharge> Failing()
        {
            yield return new("gpu", 1);
            throw new OutOfMemoryException("controlled enumeration failure");
        }
    }

    [Fact]
    public async Task ConcurrentRefillsAndOtherOwnersRemainAtomicAcrossUmaConstraints()
    {
        var budget = new MemoryBudget([new("gpu", 1000), new("ram", 1000)]);
        using var envelope = budget.Reserve([new("gpu", 500), new("ram", 500)]);
        using var start = new ManualResetEventSlim();
        int allocations = 0;
        var childWork = Task.Run(() =>
        {
            start.Wait();
            for (int i = 0; i < 1000; i++)
            {
                using var child = envelope.TryTake([new("gpu", 1), new("ram", 1)]);
                if (child != null) { child.Commit(); Interlocked.Increment(ref allocations); }
                AssertSymmetric(budget);
            }
        });
        var rivals = Task.Run(() =>
        {
            start.Wait();
            for (int i = 0; i < 1000; i++)
            {
                using var owner = budget.TryReserve([new("gpu", 400), new("ram", 400)]);
                owner?.Commit(); AssertSymmetric(budget);
            }
        });
        start.Set();
        for (int i = 0; i < 1000; i++)
        {
            long remaining = i % 2 == 0 ? 200 : 500;
            envelope.TryResizeRemaining([new("gpu", remaining), new("ram", remaining)]);
            AssertSymmetric(budget);
        }
        await Task.WhenAll(childWork, rivals);
        Assert.True(allocations > 0);
        envelope.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(1000, p.Available));
        static void AssertSymmetric(MemoryBudget b)
        {
            var s = b.Snapshot();
            Assert.Equal(s[0].Committed, s[1].Committed);
            Assert.Equal(s[0].Reserved, s[1].Reserved);
            Assert.All(s, p => Assert.InRange(p.Available, 0, 1000));
        }
    }
}
