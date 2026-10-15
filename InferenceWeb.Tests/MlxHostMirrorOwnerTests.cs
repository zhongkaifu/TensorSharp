// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp.Memory;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

/// <summary>CPU ownership tests only. No MLX/Metal library or Apple GPU is invoked.</summary>
public sealed class MlxHostMirrorOwnerTests
{
    private sealed class Allocator : MlxHostMirrorOwner.IAllocator
    {
        public int Allocations, Clears, Frees;
        public bool ReturnNull, FailClear, FailFree;
        public Action? Allocating, Freeing;
        public nuint Bytes;
        public IntPtr Allocate(nuint bytes, nuint alignment)
        { Allocations++; Bytes = bytes; Assert.Equal((nuint)64, alignment); Allocating?.Invoke(); return ReturnNull ? IntPtr.Zero : new IntPtr(64); }
        public void Clear(IntPtr pointer, nuint bytes)
        { Clears++; Assert.Equal(Bytes, bytes); if (FailClear) throw new IOException("initialize"); }
        public void Free(IntPtr pointer)
        { Frees++; Freeing?.Invoke(); if (FailFree) throw new IOException("free"); }
    }

    [Fact]
    public void SharedHostConstraintsReserveAlignedBytesBeforeAllocationAndRefundAfterFree()
    {
        var budget = new MemoryBudget([new("ram", 128), new("app", 128)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram", "app"]);
        var allocator = new Allocator
        {
            Allocating = () => Assert.All(budget.Snapshot(), p => { Assert.Equal(128, p.Reserved); Assert.Equal(0, p.Committed); }),
            Freeing = () => Assert.All(budget.Snapshot(), p => Assert.Equal(128, p.Committed))
        };
        using var owner = MlxHostMirrorOwner.Allocate(65, allocator);
        Assert.Equal(128, owner.AllocatedBytes);
        Assert.Equal(1, allocator.Clears);
        Assert.Equal(128, scope.Usage.Bytes); // not 256: two constraints, one allocation
        Assert.Throws<InvalidOperationException>(() => scope.Dispose());
        owner.Dispose();
        owner.Dispose();
        Assert.Equal(1, allocator.Frees);
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.All(budget.Snapshot(), p => Assert.Equal(128, p.Available));
    }

    [Fact]
    public void OneByteShortQuotaRefusesBeforeAllocatorAndDoesNotChangeTheLedger()
    {
        var budget = new MemoryBudget([new("ram", 127)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        var allocator = new Allocator();
        Assert.Throws<MemoryPressureException>(() => MlxHostMirrorOwner.Allocate(65, allocator));
        Assert.Equal(0, allocator.Allocations);
        Assert.Equal(0, scope.Usage.Allocations);
        Assert.Equal(127, Assert.Single(budget.Snapshot()).Available);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AllocationOrInitializationFailureReturnsCreditAfterSuccessfulRollback(bool clearFailure)
    {
        var budget = new MemoryBudget([new("ram", 64)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        var allocator = new Allocator { ReturnNull = !clearFailure, FailClear = clearFailure };
        if (clearFailure) Assert.Throws<IOException>(() => MlxHostMirrorOwner.Allocate(1, allocator));
        else Assert.Throws<OutOfMemoryException>(() => MlxHostMirrorOwner.Allocate(1, allocator));
        Assert.Equal(clearFailure ? 1 : 0, allocator.Frees);
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.Equal(64, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void FailedInitializationAndCleanupPublishRetryOwnerWithItsReservation()
    {
        var budget = new MemoryBudget([new("ram", 64)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        var allocator = new Allocator { FailClear = true, FailFree = true };
        var failure = Assert.Throws<MlxHostMirrorAllocationException>(() => MlxHostMirrorOwner.Allocate(1, allocator));
        Assert.Equal(new IntPtr(64), failure.UnreleasedOwner.Pointer);
        Assert.Equal(64, Assert.Single(budget.Snapshot()).Reserved);
        Assert.Throws<InvalidOperationException>(() => scope.Dispose());
        allocator.FailFree = false;
        failure.UnreleasedOwner.Dispose();
        Assert.Equal(64, Assert.Single(budget.Snapshot()).Available);
    }

    [Fact]
    public void FailedFreeRetainsPointerAndCommittedCreditForRetry()
    {
        var budget = new MemoryBudget([new("ram", 64)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        var allocator = new Allocator { FailFree = true };
        using var owner = MlxHostMirrorOwner.Allocate(1, allocator);
        Assert.Throws<IOException>(() => owner.Dispose());
        Assert.Equal(new IntPtr(64), owner.Pointer);
        Assert.Equal(64, Assert.Single(budget.Snapshot()).Committed);
        allocator.FailFree = false;
        owner.Dispose();
        Assert.Equal(64, Assert.Single(budget.Snapshot()).Available);
        Assert.Throws<ObjectDisposedException>(() => owner.Pointer);
    }

    [Theory]
    [InlineData(0, 64)]
    [InlineData(65, 128)]
    public unsafe void NativeAlignedMirrorIsZeroInitializedAndChargesActualRoundedPayload(int bytes, long aligned)
    {
        var budget = new MemoryBudget([new("ram", aligned)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        using var owner = MlxHostMirrorOwner.Allocate(bytes);
        Assert.Equal(0UL, (ulong)owner.Pointer.ToInt64() % 64UL);
        Assert.Equal(aligned, owner.AllocatedBytes);
        for (int i = 0; i < aligned; i++) Assert.Equal((byte)0, ((byte*)owner.Pointer)[i]);
        ((byte*)owner.Pointer)[aligned - 1] = 0x5a;
        Assert.Equal(aligned, Assert.Single(budget.Snapshot()).Committed);
        owner.Dispose();
        Assert.Equal(0, scope.Usage.Bytes);
    }

    [Fact]
    public async Task ConcurrentDisposalFreesOnceWithoutUsingTheMlxWorker()
    {
        var budget = new MemoryBudget([new("ram", 64)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        var allocator = new Allocator();
        var owner = MlxHostMirrorOwner.Allocate(1, allocator);
        await Task.WhenAll(Enumerable.Range(0, 16).Select(_ => Task.Run(owner.Dispose)));
        Assert.Equal(1, allocator.Frees);
        Assert.Equal(0, scope.Usage.Allocations);
    }

    [Fact]
    public void InvalidSizeAndAlignmentOverflowCannotReachTheAllocator()
    {
        var allocator = new Allocator();
        Assert.Throws<ArgumentOutOfRangeException>(() => MlxHostMirrorOwner.Allocate(-1, allocator));
        Assert.Throws<OverflowException>(() => MlxHostMirrorOwner.Allocate(long.MaxValue, allocator));
        Assert.Equal(0, allocator.Allocations);
    }
}
