// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class HostAllocationBudgetTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AllocationOwnerFailureDoesNotTakeCreditOrMutateUsage(bool withinRequest)
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        using var envelope = withinRequest ? budget.Reserve([new("ram", 100)]) : null;
        using var execution = envelope == null ? null : scope.EnterExecution(envelope);
        var before = budget.Snapshot().ToArray();
        var beforeUsage = scope.Usage;
        scope.BeforeAllocationOwnerAllocation = () => throw new OutOfMemoryException("controlled host owner failure");
        Assert.Throws<OutOfMemoryException>(() => HostAllocationBudgetScope.Reserve(20));
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(beforeUsage, scope.Usage);
        if (envelope != null) Assert.False(envelope.HasChildren);
        scope.BeforeAllocationOwnerAllocation = null;
        using (var retry = HostAllocationBudgetScope.Reserve(20))
        {
            Assert.NotNull(retry);
            retry.Commit();
            Assert.Equal(20, scope.Usage.Bytes);
        }
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.Equal(0, scope.Usage.Allocations);
    }

    [Fact]
    public void ExecutionOwnerFailureDoesNotPublishAnUnownedEnvelope()
    {
        var budget = new MemoryBudget([new("ram", 100)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        using var envelope = budget.Reserve([new("ram", 100)]);
        scope.BeforeExecutionOwnerAllocation = () => throw new OutOfMemoryException("controlled execution owner failure");
        Assert.Throws<OutOfMemoryException>(() => scope.EnterExecution(envelope));
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.False(envelope.HasChildren);
        scope.BeforeExecutionOwnerAllocation = null;
        using (scope.EnterExecution(envelope))
        using (var allocation = HostAllocationBudgetScope.Reserve(100)) allocation!.Commit();
        Assert.Equal(new MemoryCharge("ram", 100), Assert.Single(envelope.Charges));
        Assert.Equal(0, scope.Usage.Allocations);
        Assert.Equal(100, budget.Snapshot().Single().Reserved);
        scope.Dispose(); // No hidden execution owner prevented detachment.
    }

    [Fact]
    public void InitialPoolRejectionRollsBackEarlierBlocks()
    {
        var budget = new MemoryBudget([new("ram", 64L << 20)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        var pool = new GgmlMemoryPool(GgmlBackendType.Cpu);
        Assert.Throws<MemoryPressureException>(() => pool.EnsureInitialBlocks());
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.Equal(64L << 20, budget.Snapshot().Single().Available);
        Assert.Equal(0, pool.Trim());
    }
    [Fact]
    public void PooledBlockKeepsActualRoundedChargeAcrossSmallerReuse()
    {
        long page = Environment.SystemPageSize;
        var budget = new MemoryBudget([new("ram", page * 2)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        var pool = new GgmlMemoryPool(GgmlBackendType.Cuda);
        var pointer = pool.Allocate(page + 1);
        try
        {
            Assert.Equal(page * 2, budget.Snapshot().Single().Committed);
            pool.Free(pointer, page + 1);
            pointer = IntPtr.Zero;
            Assert.Throws<InvalidOperationException>(() => scope.Dispose());
            pointer = pool.Allocate(1);
            Assert.Equal(page * 2, scope.Usage.Bytes);
            Assert.Throws<MemoryPressureException>(() => pool.Allocate(1));
            pool.Free(pointer, 1);
            Assert.Throws<InvalidOperationException>(() => pool.Free(pointer, 1));
            pointer = IntPtr.Zero;
            Assert.Equal(page * 2, pool.Trim());
            Assert.Equal(0, budget.Snapshot().Single().Committed);
        }
        finally { if (pointer != IntPtr.Zero) pool.Free(pointer, 1); pool.Trim(); }
    }

    [Fact]
    public void FailedUnmapKeepsOwnershipUntilSuccessfulRetry()
    {
        var budget = new MemoryBudget([new("ram", 16L << 20)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        bool fail = true;
        var release = typeof(GgmlMemoryPool).GetMethod("FreeVirtual", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        var pool = new GgmlMemoryPool(GgmlBackendType.Cuda, (p, n) => !fail && (bool)release.Invoke(null, [p, n])!);
        var ptr = pool.Allocate(16L << 20);
        pool.Free(ptr, 16L << 20); // Above retention threshold: immediate free fails.
        try
        {
            Assert.Equal(16L << 20, scope.Usage.Bytes);
            Assert.Throws<System.IO.IOException>(() => pool.Trim());
            Assert.Throws<MemoryPressureException>(() => pool.Allocate(1));
            Assert.Throws<InvalidOperationException>(() => scope.Dispose());
        }
        finally { fail = false; Assert.Equal(16L << 20, pool.Trim()); }
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.Equal(new MemoryPoolHighWatermark("ram", 16L << 20, 16L << 20), budget.HighWatermarks().Single());
    }

    [Fact]
    public void WeightBuffersConsumeEnvelopeAndOutliveRequestWithoutLosingCredit()
    {
        var budget = new MemoryBudget([new("ram", 256)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        using var envelope = budget.Reserve([new("ram", 256)]);
        IntPtr ptr = IntPtr.Zero;
        try
        {
            using (scope.EnterExecution(envelope))
            {
                ptr = HostBuffers.Allocate(193); // 64-byte alignment rounds to 256.
                Assert.Equal(0, budget.Snapshot().Single().Reserved);
                Assert.Equal(256, budget.Snapshot().Single().Committed);
                Assert.Throws<MemoryPressureException>(() => HostBuffers.Allocate(1));
                Marshal.WriteByte(ptr, 192, 71);
            }
            envelope.Dispose();
            Assert.Equal(256, budget.Snapshot().Single().Committed);
            Assert.Equal(71, Marshal.ReadByte(ptr, 192));
        }
        finally { HostBuffers.Free(ptr); }
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.Equal(new MemoryPoolHighWatermark("ram", 256, 256), budget.HighWatermarks().Single());
    }

    [Fact]
    public void FailedReservationAndForeignEnvelopeDoNotLeakOrBorrowQuota()
    {
        var budget = new MemoryBudget([new("ram", 64)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        using var foreign = new MemoryBudget([new("ram", 64)]).Reserve([new("ram", 64)]);
        Assert.Throws<ArgumentException>(() => scope.EnterExecution(foreign));
        Assert.Throws<MemoryPressureException>(() => HostBuffers.Allocate(65));
        Assert.Throws<ArgumentOutOfRangeException>(() => HostBuffers.Allocate(long.MaxValue, 3));
        Assert.Throws<OverflowException>(() => HostBuffers.Allocate(long.MaxValue));
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.Equal(64, budget.Snapshot().Single().Available);
    }
}
