// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Runtime.Paged;

namespace InferenceWeb.Tests;

public sealed class PagedKvStorageIntegrationTests : IDisposable
{
    private const int PageBytes = 128;
    private const int StagingBytes = 64;
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-page-payload-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    private PagedKvStorage CreateTiered()
        // Exactly one physical page, plus capture scratch and spill staging.
        => new(4, PageBytes, new KvSnapshotOptions(2 * PageBytes + StagingBytes,
            16 * 4096, _directory, StagingBytes));

    [Fact]
    public void ShortPayloadSurvivesSpillReloadWithoutReducingPhysicalPageCharge()
    {
        using var storage = CreateTiered();
        byte[] first = Enumerable.Range(0, 23).Select(i => (byte)(i * 7 + 3)).ToArray();
        storage.Store(0, first, null, exactLength: true);
        Assert.Equal(PageBytes, storage.SlabLength(0));
        Assert.Equal(2 * PageBytes + StagingBytes,
            storage.MemoryUsage!.Single(p => p.Pool == "kv/ram").Committed);

        storage.Store(1, Enumerable.Repeat((byte)211, PageBytes).ToArray(), null);
        Assert.True(storage.ResidencyStats!.Value.Spills > 0);
        Assert.True(storage.MemoryUsage!.Single(p => p.Pool == "kv/ssd").Committed > 0);
        // Prefix accounting remains a full physical page even while it is spilled.
        Assert.Equal(PageBytes, storage.SlabLength(0));
        using (var lease = storage.Acquire(0))
            Assert.Equal(first, lease.ReadOnlySpan.ToArray());
        Assert.Equal(PageBytes, storage.SlabLength(0));
        Assert.Equal(0, storage.AllocatedBytes);
        Assert.True(storage.ResidencyStats!.Value.Loads >= 3);

        storage.Dispose();
        Assert.All(storage.MemoryUsage!, pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExactSnapshotsAndPaddedPartialTailsKeepDistinctReadLengths(bool tiered)
    {
        using var storage = tiered ? CreateTiered() : new PagedKvStorage(4, PageBytes);
        storage.Store(0, Enumerable.Repeat((byte)199, PageBytes).ToArray(), null);
        byte[] shortSnapshot = { 5, 7, 11, 13, 17 };
        storage.Store(0, shortSnapshot, null, exactLength: true);
        using (var lease = storage.Acquire(0))
            Assert.Equal(shortSnapshot, lease.ReadOnlySpan.ToArray());
        Assert.Equal(tiered ? PageBytes : shortSnapshot.Length, storage.SlabLength(0));

        // A partial token block retains a full-size slab; its unwritten suffix must
        // not expose state left by the old complete snapshot.
        byte[] partial = { 29, 31, 37 };
        storage.Store(0, partial, null, exactLength: false);
        using (var lease = storage.Acquire(0))
        {
            Assert.Equal(PageBytes, lease.ReadOnlySpan.Length);
            Assert.Equal(partial, lease.ReadOnlySpan[..partial.Length].ToArray());
            Assert.All(lease.ReadOnlySpan[partial.Length..].ToArray(), b => Assert.Equal((byte)0, b));
        }
        storage.Store(0, shortSnapshot, null, exactLength: true);
        using (var lease = storage.Acquire(0))
            Assert.Equal(shortSnapshot, lease.ReadOnlySpan.ToArray());
    }

    [Fact]
    public void FailedReleasePreservesShortViewAndCreditUntilLeaseEnds()
    {
        using var storage = CreateTiered();
        byte[] bytes = { 2, 3, 5, 7, 11 };
        storage.Store(0, bytes, null, exactLength: true);
        using (var lease = storage.Acquire(0))
        {
            long charged = storage.MemoryUsage!.Sum(p => p.Committed);
            Assert.Throws<InvalidOperationException>(() => storage.ReleaseSlab(0));
            Assert.Throws<InvalidOperationException>(() => storage.Dispose());
            Assert.Equal(charged, storage.MemoryUsage!.Sum(p => p.Committed));
            Assert.Equal(PageBytes, storage.SlabLength(0));
            Assert.Equal(bytes, lease.ReadOnlySpan.ToArray());
            using var second = storage.Acquire(0);
            Assert.Equal(bytes, second.ReadOnlySpan.ToArray());
        }

        storage.ReleaseSlab(0);
        Assert.Equal(0, storage.SlabLength(0));
        Assert.Equal(0, storage.ResidencyStats!.Value.Resources);
        Assert.Equal(PageBytes + StagingBytes, storage.MemoryUsage!.Sum(p => p.Committed));
        // Recycled block ids must not inherit the previous resource's logical size.
        storage.Store(0, new byte[31], null, exactLength: true);
        using (var lease = storage.Acquire(0)) Assert.Equal(31, lease.ReadOnlySpan.Length);
        storage.Dispose();
        Assert.All(storage.MemoryUsage!, pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Fact]
    public void TieredRawSpanAccessCannotAllocateUnchargedManagedSlabs()
    {
        using var storage = CreateTiered();
        Assert.Throws<InvalidOperationException>(() => storage.GetSpan(0));
        Assert.Throws<InvalidOperationException>(() => storage.GetSpan(0, 17));
        Assert.Throws<InvalidOperationException>(() => storage.GetReadOnlySpan(0));
        Assert.Equal(0, storage.AllocatedBytes);
        Assert.Equal(0, storage.ResidencyStats!.Value.Resources);
        Assert.Equal(PageBytes + StagingBytes, storage.MemoryUsage!.Sum(p => p.Committed));
    }

    [Fact]
    public void PinnedPageDeniesAnotherSnapshotAndRetryPublishesOnlyRequestedPayload()
    {
        using var storage = CreateTiered();
        byte[] first = { 19, 23, 29 };
        byte[] second = { 31, 37, 41, 43, 47, 53, 59 };
        storage.Store(0, first, null, exactLength: true);
        using (var held = storage.Acquire(0))
        {
            Assert.Throws<MemoryPressureException>(() => storage.Store(1, second, null, exactLength: true));
            Assert.Equal(first, held.ReadOnlySpan.ToArray());
            Assert.Equal(2 * PageBytes + StagingBytes, storage.MemoryUsage!.Sum(p => p.Committed));
            Assert.All(storage.MemoryUsage!, p => Assert.Equal(0, p.Reserved));
        }
        storage.Store(1, second, null, exactLength: true);
        using (var lease = storage.Acquire(1)) Assert.Equal(second, lease.ReadOnlySpan.ToArray());
        using (var lease = storage.Acquire(0)) Assert.Equal(first, lease.ReadOnlySpan.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OversizedReplacementPreservesExistingPayloadAndAccounting(bool tiered)
    {
        using var storage = tiered ? CreateTiered() : new PagedKvStorage(4, PageBytes);
        byte[] original = { 61, 67, 71, 73, 79 };
        storage.Store(0, original, null, exactLength: true);
        long physical = storage.SlabLength(0);
        long managed = storage.AllocatedBytes;
        Assert.Throws<ArgumentOutOfRangeException>(() => storage.Store(0, new byte[PageBytes + 1], null, exactLength: true));
        Assert.Equal(physical, storage.SlabLength(0));
        Assert.Equal(managed, storage.AllocatedBytes);
        using var lease = storage.Acquire(0);
        Assert.Equal(original, lease.ReadOnlySpan.ToArray());
    }

    [Fact]
    public void ImpossibleManagedResizeKeepsPreviousSlabAndAllocatedBytes()
    {
        // The CLR rejects an array beyond its supported length before allocation;
        // this exercises the failed-new path without consuming a huge host buffer.
        using var storage = new PagedKvStorage(1, long.MaxValue);
        storage.GetSpan(0, 4).Fill(83);
        Exception? error = Record.Exception(() => storage.GetSpan(0));
        Assert.True(error is OverflowException or OutOfMemoryException, $"Unexpected resize outcome: {error}");
        Assert.Equal(4, storage.AllocatedBytes);
        Assert.Equal(4, storage.SlabLength(0));
        Assert.Equal(new byte[] { 83, 83, 83, 83 }, storage.GetReadOnlySpan(0).ToArray());
        storage.ReleaseSlab(0);
        Assert.Equal(0, storage.AllocatedBytes);
    }
}
