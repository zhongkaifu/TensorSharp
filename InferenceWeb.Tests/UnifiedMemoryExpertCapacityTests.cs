// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Common policy used by the Qwen and GLM adapters. These tests use real
/// shared-ledger ownership and an injected synchronous native lifecycle, without
/// claiming model, GPU, or platform qualification.</summary>
public sealed class UnifiedMemoryExpertCapacityTests
{
    private const string Pool = "test/accelerator/7";
    private static MemoryCharge[] Charges(long bytes) => [new(Pool, bytes)];
    private static long Bytes(IReadOnlyList<MemoryCharge> charges) => Assert.Single(charges).Bytes;

    private sealed class Native : IHostMoeExpertCacheScopeNative
    {
        internal int Updates;
        internal long Budget;
        internal Action<long>? OnUpdate;
        public ulong Begin(long bytes, int rows, int layers, int decodeRows) { Budget = bytes; return 1; }
        public long Update(ulong token, long bytes) { Updates++; OnUpdate?.Invoke(bytes); Budget = bytes; return 0; }
        public long Trim(ulong token, long bytes) => 0;
        public long End(ulong token) => 0;
        public GgmlHostMoeExpertCacheUsage ReadUsage() => new(0, Budget, 0, 0, 0);
    }

    [Fact]
    public void ZeroFloorDisablesCacheOnlyAfterEveryPendingAndCommittedChildIsReleased()
    {
        var budget = new MemoryBudget(Charges(1000));
        using var envelope = budget.Reserve(Charges(600));
        using var pending = envelope.TryTake(Charges(100))!;
        using var committed = envelope.TryTake(Charges(300))!; committed.Commit();
        using var sibling = budget.Reserve(Charges(200));
        var native = new Native { OnUpdate = target =>
        {
            Assert.Equal(0, target);
            Assert.Equal(600, Bytes(envelope.OwnedCharges));
            Assert.Equal(400, Bytes(envelope.LiveChildCharges));
            Assert.False(budget.CanReserve(Charges(201))); // No early refund.
            pending.Dispose(); committed.Dispose();
            Assert.Equal(600, Bytes(envelope.OwnedCharges));
        }};
        using var cache = new GgmlHostMoeExpertCacheScope(600, 32, 3, native, 3);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, envelope, Pool, 0, 0, Charges(1000)));
        Assert.Equal(0, Bytes(envelope.OwnedCharges));
        Assert.Equal(0, Bytes(envelope.LiveChildCharges));
        Assert.Equal(200, Assert.Single(budget.Snapshot()).Reserved);
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Committed);
        Assert.Equal(0, cache.BudgetBytes);
        Assert.True(budget.CanReserve(Charges(800)));
    }

    [Fact]
    public void GrowthReservesTotalWithPendingCommittedAndClosedIntermediateChildrenExactlyOnce()
    {
        var budget = new MemoryBudget(Charges(1000));
        using var envelope = budget.Reserve(Charges(600));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(600, 512, 3, native, 3);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, envelope, Pool, 400, 200, Charges(1000)));
        using var pending = envelope.TryTake(Charges(80))!;
        using var intermediate = envelope.TryTake(Charges(250))!;
        using var committed = intermediate.TryTake(Charges(220))!; committed.Commit();
        intermediate.Dispose(); // The retained 220-byte child still belongs to this root.
        using var sibling = budget.Reserve(Charges(300));
        native.OnUpdate = target =>
        {
            Assert.Equal(600, target);
            Assert.Equal(600, Bytes(envelope.OwnedCharges));
            Assert.Equal(300, Bytes(envelope.LiveChildCharges));
            Assert.Equal(300, Bytes(envelope.Charges));
            Assert.False(budget.CanReserve(Charges(101)));
        };
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, envelope, Pool, 600, 200, Charges(1000)));
        Assert.Equal(600, cache.BudgetBytes);
        Assert.Equal(600, Bytes(envelope.OwnedCharges));
        Assert.Equal(new MemoryPoolSnapshot(Pool, 1000, 680, 220), Assert.Single(budget.Snapshot()));
    }

    [Theory]
    [InlineData(500, 1000)] // A sibling owns configured capacity.
    [InlineData(0, 599)] // Fresh physical allowance is tighter than the ledger.
    public void RefusedGrowthIsAtomicAndNeverCallsNative(long siblingBytes, long physicalLimit)
    {
        var budget = new MemoryBudget(Charges(1000));
        using var envelope = budget.Reserve(Charges(600));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(600, 32, 3, native, 3);
        UnifiedMemoryExpertCapacity.Resize(cache, envelope, Pool, 400, 0, Charges(1000));
        using var child = envelope.TryTake(Charges(100))!; child.Commit();
        using var sibling = budget.Reserve(Charges(siblingBytes));
        native.Updates = 0;
        var before = budget.Snapshot().ToArray();
        var peaks = budget.HighWatermarks().ToArray();
        Assert.False(UnifiedMemoryExpertCapacity.Resize(cache, envelope, Pool, 600, 0, Charges(physicalLimit)));
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(peaks, budget.HighWatermarks().ToArray());
        Assert.Equal(400, Bytes(envelope.OwnedCharges));
        Assert.Equal(100, Bytes(envelope.LiveChildCharges));
        Assert.Equal(400, cache.BudgetBytes);
        Assert.Equal(0, native.Updates);
    }

    [Fact]
    public void FailedGrowthAfterPartialNativeReleaseRestoresOnlyUnusedCredit()
    {
        var budget = new MemoryBudget(Charges(1000));
        using var envelope = budget.Reserve(Charges(600));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(600, 512, 3, native, 3);
        UnifiedMemoryExpertCapacity.Resize(cache, envelope, Pool, 400, 200, Charges(1000));
        using var retired = envelope.TryTake(Charges(120))!; retired.Commit();
        using var retained = envelope.TryTake(Charges(180))!; retained.Commit();
        using var pending = envelope.TryTake(Charges(40))!;
        using var sibling = budget.Reserve(Charges(300));
        native.OnUpdate = _ =>
        {
            Assert.Equal(600, Bytes(envelope.OwnedCharges));
            retired.Dispose();
            throw new IOException("native growth failed after fenced retirement");
        };
        var error = Assert.Throws<IOException>(() => UnifiedMemoryExpertCapacity.Resize(cache, envelope,
            Pool, 600, 200, Charges(1000)));
        Assert.Contains("fenced retirement", error.Message);
        Assert.Equal(400, cache.BudgetBytes);
        Assert.Equal(400, Bytes(envelope.OwnedCharges));
        Assert.Equal(220, Bytes(envelope.LiveChildCharges));
        Assert.Equal(180, Bytes(envelope.Charges));
        Assert.Equal(new MemoryPoolSnapshot(Pool, 1000, 520, 180), Assert.Single(budget.Snapshot()));
    }

    [Fact]
    public void FailedGrowthWithChildrenAbovePreviousCeilingKeepsAllOwnersAndRethrowsNativeFault()
    {
        var budget = new MemoryBudget(Charges(1000));
        using var envelope = budget.Reserve(Charges(600));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(600, 512, 3, native, 3);
        UnifiedMemoryExpertCapacity.Resize(cache, envelope, Pool, 400, 200, Charges(1000));
        using var committed = envelope.TryTake(Charges(100))!; committed.Commit();
        using var pending = envelope.TryTake(Charges(50))!;
        using var sibling = budget.Reserve(Charges(300));
        BudgetReservation? unexpected = null;
        try
        {
            native.OnUpdate = _ =>
            {
                Assert.Equal(600, Bytes(envelope.OwnedCharges));
                // An adversarial native failure leaves new payload alive even
                // though it preserves the old configured cache ceiling.
                unexpected = envelope.TryTake(Charges(300))!;
                unexpected.Commit();
                throw new IOException("native fault with surviving new payload");
            };
            var error = Assert.Throws<IOException>(() => UnifiedMemoryExpertCapacity.Resize(cache,
                envelope, Pool, 600, 200, Charges(1000)));
            Assert.Equal("native fault with surviving new payload", error.Message);
            Assert.Equal(400, cache.BudgetBytes);
            // Rollback refunds slack only. It cannot revoke 450 live bytes to
            // manufacture an exact 400-byte total; this is a failed operation.
            Assert.Equal(450, Bytes(envelope.OwnedCharges));
            Assert.Equal(450, Bytes(envelope.LiveChildCharges));
            Assert.Equal(0, Bytes(envelope.Charges));
            Assert.Equal(300, Bytes(unexpected!.OwnedCharges));
            Assert.Equal(100, Bytes(committed.OwnedCharges));
            Assert.Equal(50, Bytes(pending.OwnedCharges));
            Assert.Equal(new MemoryPoolSnapshot(Pool, 1000, 350, 400), Assert.Single(budget.Snapshot()));
            Assert.Null(envelope.TryTake(Charges(1)));
            Assert.False(budget.CanReserve(Charges(251)));
            Assert.True(budget.CanReserve(Charges(250)));
        }
        finally { unexpected?.Dispose(); }
    }

    [Fact]
    public void ShrinkCannotRefundPendingChildrenWhenNativeReleaseIsIncomplete()
    {
        var budget = new MemoryBudget(Charges(1000));
        using var envelope = budget.Reserve(Charges(600));
        using var pending = envelope.TryTake(Charges(150))!;
        using var committed = envelope.TryTake(Charges(200))!; committed.Commit();
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(600, 32, 3, native, 3);
        var before = budget.Snapshot().ToArray();
        Assert.Throws<InvalidOperationException>(() => UnifiedMemoryExpertCapacity.Resize(cache, envelope,
            Pool, 300, 0, Charges(1000)));
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(600, Bytes(envelope.OwnedCharges));
        Assert.Equal(350, Bytes(envelope.LiveChildCharges));
    }

    [Fact]
    public void WrongPoolAndImmutableFloorAreRejectedBeforeNativeOrLedgerMutation()
    {
        var budget = new MemoryBudget([new(Pool, 1000), new("unrelated", 1000)]);
        using var envelope = budget.Reserve(Charges(600));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(600, 32, 3, native, 3);
        var before = budget.Snapshot().ToArray();
        Assert.Throws<ArgumentException>(() => UnifiedMemoryExpertCapacity.Resize(cache, envelope,
            "unrelated", 400, 200, [new("unrelated", 1000)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => UnifiedMemoryExpertCapacity.Resize(cache, envelope,
            Pool, 199, 200, Charges(1000)));
        Assert.Throws<ArgumentOutOfRangeException>(() => UnifiedMemoryExpertCapacity.Resize(cache, envelope,
            Pool, 601, 200, Charges(1000)));
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(600, cache.BudgetBytes);
        Assert.Equal(0, native.Updates);
    }
}
