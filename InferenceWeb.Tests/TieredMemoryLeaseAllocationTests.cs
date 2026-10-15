// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class TieredMemoryLeaseAllocationTests
{
    private sealed class ManagedBackend : IMemoryBackend
    {
        public MemoryLocation Location { get; } = new("test", "managed", MemoryTier.Host);
        public int Allocations, Live;
        public IReadOnlyList<MemoryCharge> GetAllocationCharges(long bytes) => [new("ram", bytes)];
        public ValueTask<IResourceBuffer> AllocateAsync(long bytes, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var buffer = new Buffer(this, checked((int)bytes));
            Allocations++; Live++;
            return ValueTask.FromResult<IResourceBuffer>(buffer);
        }
        private sealed class Buffer(ManagedBackend owner, int length) : IResourceBuffer
        {
            private readonly byte[] _data = new byte[length];
            private bool _disposed;
            public long ByteLength => _data.Length;
            public nint Pointer { get { ObjectDisposedException.ThrowIf(_disposed, this); return 1; } }
            public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ObjectDisposedException.ThrowIf(_disposed, this);
                _data.AsMemory(checked((int)offset), destination.Length).CopyTo(destination);
                return ValueTask.CompletedTask;
            }
            public ValueTask WriteAsync(long offset, ReadOnlyMemory<byte> source, CancellationToken cancellationToken = default)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ObjectDisposedException.ThrowIf(_disposed, this);
                source.CopyTo(_data.AsMemory(checked((int)offset), source.Length));
                return ValueTask.CompletedTask;
            }
            public void Dispose() { if (!_disposed) { _disposed = true; owner.Live--; } }
        }
    }

    [Theory]
    [InlineData(ResourceAccess.Read, false)]
    [InlineData(ResourceAccess.Read, true)]
    [InlineData(ResourceAccess.Write, false)]
    [InlineData(ResourceAccess.Write, true)]
    public async Task FailedLeaseOwnerAllocationLeavesNoPhantomPinAndPreservesOnlyPublishedReplicas(ResourceAccess access, bool warm)
    {
        var budget = new MemoryBudget([new("ram", 4096), new("ssd", 4096)]);
        var backend = new ManagedBackend();
        string directory = Path.Combine(Path.GetTempPath(), "ts-lease-owner-fault-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var transfers = new BoundedTransfers(budget, "ram", 64, 1);
            using var spill = new SsdSpillStore(budget, "ssd", directory, transfers);
            long stagingCommitted = budget.Snapshot().Single(p => p.Pool == "ram").Committed;
            await using (var scheduler = new TieredMemoryScheduler(budget, [backend], transfers, spill))
            {
                var key = new ResourceKey("fault", 4, "mutable");
                scheduler.Register(new(key, 64, ResourceKind.RecurrentState, Mutable: true));
                if (warm) { using var initial = await scheduler.AcquireAsync(key, backend.Location); }
                int injections = 0;
                scheduler.BeforeLeaseOwnerAllocation = () => { injections++; throw new OutOfMemoryException("lease owner injection"); };
                var error = await Assert.ThrowsAsync<OutOfMemoryException>(() => scheduler.AcquireAsync(key, backend.Location, access).AsTask());
                Assert.Equal("lease owner injection", error.Message);
                Assert.Equal(1, injections);
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                if (warm)
                {
                    var replica = Assert.Single(scheduler.Snapshot());
                    Assert.Equal(0, replica.Readers); Assert.False(replica.Writer);
                }
                else Assert.Empty(scheduler.Snapshot());
                // A cold provisional buffer rolls back with its charge. An
                // already published replica remains live and fully charged.
                Assert.Equal(1, backend.Allocations); Assert.Equal(warm ? 1 : 0, backend.Live);
                Assert.Equal(stagingCommitted + (warm ? 64 : 0), budget.Snapshot().Single(p => p.Pool == "ram").Committed);
                scheduler.BeforeLeaseOwnerAllocation = null;
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                using (var retry = await scheduler.AcquireAsync(key, backend.Location, ResourceAccess.Write, timeout.Token))
                {
                    Assert.Equal(1, scheduler.GetStats().ActiveLeases);
                    byte[] expected = Enumerable.Repeat((byte)73, 64).ToArray();
                    byte[] actual = new byte[64];
                    await retry.WriteAsync(0, expected); await retry.ReadAsync(0, actual);
                    Assert.Equal(expected, actual);
                }
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                // No owner is leaked and the resource unregisters immediately.
                scheduler.Unregister(key);
                Assert.Empty(scheduler.Snapshot()); Assert.Equal(0, backend.Live);
            }
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FailedOverwriteOwnerAllocationPreservesAuthoritativeBytesAndVersion(bool spilled)
    {
        // One transfer chunk and one resident payload fit. Loading the pressure
        // resource therefore forces the mutable source onto the real SSD path.
        var budget = new MemoryBudget([new("ram", 128), new("ssd", 4096)]);
        var backend = new ManagedBackend();
        string directory = Path.Combine(Path.GetTempPath(), "ts-overwrite-owner-fault-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var transfers = new BoundedTransfers(budget, "ram", 64, 1);
            using var spill = new SsdSpillStore(budget, "ssd", directory, transfers);
            await using (var scheduler = new TieredMemoryScheduler(budget, [backend], transfers, spill))
            {
                var key = new ResourceKey("fault", 6, "overwrite");
                scheduler.Register(new(key, 64, ResourceKind.KvPage, Mutable: true));
                byte[] original = Enumerable.Range(0, 64).Select(i => (byte)(i * 3 + 7)).ToArray();
                using (var initial = await scheduler.AcquireAsync(key, backend.Location, ResourceAccess.Write))
                {
                    Assert.Equal(1, initial.Version);
                    await initial.WriteAsync(0, original);
                }
                if (spilled)
                {
                    var pressureKey = new ResourceKey("fault", 7, "pressure");
                    scheduler.Register(new(pressureKey, 64, ResourceKind.KvPage, Mutable: true));
                    using (var pressure = await scheduler.AcquireAsync(pressureKey, backend.Location))
                        Assert.DoesNotContain(scheduler.Snapshot(), r => r.Resource == key);
                    scheduler.Unregister(pressureKey);
                    Assert.Empty(scheduler.Snapshot());
                    Assert.Equal(1, scheduler.GetStats().Spills);
                }
                var budgetBefore = budget.Snapshot().ToArray();
                long copiedBefore = transfers.BytesCopied;
                string[] filesBefore = Directory.GetFiles(directory, "*.bin", SearchOption.AllDirectories);
                Assert.Equal(spilled ? 1 : 0, filesBefore.Length);
                if (spilled) Assert.Equal(original, File.ReadAllBytes(Assert.Single(filesBefore)));
                int injections = 0;
                scheduler.BeforeLeaseOwnerAllocation = () =>
                {
                    injections++;
                    throw new OutOfMemoryException("overwrite owner injection");
                };
                var error = await Assert.ThrowsAsync<OutOfMemoryException>(() => scheduler.AcquireForOverwriteAsync(key, backend.Location).AsTask());
                Assert.Equal("overwrite owner injection", error.Message);
                Assert.Equal(1, injections);
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                Assert.All(scheduler.Snapshot(), r => { Assert.Equal(0, r.Readers); Assert.False(r.Writer); });
                Assert.Equal(budgetBefore, budget.Snapshot().ToArray());
                Assert.Equal(copiedBefore, transfers.BytesCopied); // No old-data read for the failed overwrite.
                Assert.Equal(spilled ? 0 : 1, backend.Live);
                Assert.Equal(filesBefore, Directory.GetFiles(directory, "*.bin", SearchOption.AllDirectories));
                if (spilled) Assert.Equal(original, File.ReadAllBytes(Assert.Single(filesBefore)));

                scheduler.BeforeLeaseOwnerAllocation = null;
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                using (var recovered = await scheduler.AcquireAsync(key, backend.Location, cancellationToken: timeout.Token))
                {
                    Assert.Equal(1, recovered.Version); // Failed acquisition did not publish a new version.
                    byte[] actual = new byte[64]; await recovered.ReadAsync(0, actual);
                    Assert.Equal(original, actual);
                }
                byte[] replacement = Enumerable.Repeat((byte)73, 64).ToArray();
                using (var retry = await scheduler.AcquireForOverwriteAsync(key, backend.Location, timeout.Token))
                {
                    Assert.Equal(2, retry.Version);
                    await retry.WriteAsync(0, replacement);
                }
                using (var verified = await scheduler.AcquireAsync(key, backend.Location, cancellationToken: timeout.Token))
                {
                    byte[] actual = new byte[64]; await verified.ReadAsync(0, actual);
                    Assert.Equal(replacement, actual);
                }
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                scheduler.Unregister(key);
                Assert.Empty(scheduler.Snapshot()); Assert.Equal(0, backend.Live);
                Assert.Empty(Directory.GetFiles(directory, "*.bin", SearchOption.AllDirectories));
                Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "ssd").Committed);
            }
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Theory]
    [InlineData(ResourceAccess.Read)]
    [InlineData(ResourceAccess.Write)]
    public async Task CancelledConflictWaiterDoesNotLoseTheSurvivorOrNextNotification(ResourceAccess access)
    {
        var budget = new MemoryBudget([new("ram", 4096), new("ssd", 4096)]);
        var backend = new ManagedBackend();
        string directory = Path.Combine(Path.GetTempPath(), "ts-lease-waiters-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var transfers = new BoundedTransfers(budget, "ram", 64, 1);
            using var spill = new SsdSpillStore(budget, "ssd", directory, transfers);
            await using (var scheduler = new TieredMemoryScheduler(budget, [backend], transfers, spill))
            {
                var key = new ResourceKey("fault", 8, "waiters");
                scheduler.Register(new(key, 64, ResourceKind.RecurrentState, Mutable: true));
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                using var cancelled = new CancellationTokenSource();
                using var owner = await scheduler.AcquireAsync(key, backend.Location, ResourceAccess.Write, timeout.Token);
                byte[] expected = Enumerable.Repeat((byte)41, 64).ToArray();
                await owner.WriteAsync(0, expected);
                var survivorTask = scheduler.AcquireAsync(key, backend.Location, access, timeout.Token).AsTask();
                var cancelledTask = scheduler.AcquireAsync(key, backend.Location, ResourceAccess.Read, cancelled.Token).AsTask();
                Assert.False(survivorTask.IsCompleted); Assert.False(cancelledTask.IsCompleted);
                cancelled.Cancel();
                await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cancelledTask);
                Assert.False(survivorTask.IsCompleted);
                Assert.Equal(1, scheduler.GetStats().ActiveLeases); // Original owner is still pinned.
                owner.Dispose();
                using var survivor = await survivorTask.WaitAsync(timeout.Token);
                Assert.Equal(access == ResourceAccess.Read ? 1 : 2, survivor.Version);
                byte[] actual = new byte[64]; await survivor.ReadAsync(0, actual);
                Assert.Equal(expected, actual);
                // The old signal has been detached. A newly blocked writer
                // must enroll in another notification and wake on this release.
                var nextTask = scheduler.AcquireAsync(key, backend.Location, ResourceAccess.Write, timeout.Token).AsTask();
                Assert.False(nextTask.IsCompleted);
                Assert.Equal(1, scheduler.GetStats().ActiveLeases);
                survivor.Dispose();
                using (var next = await nextTask.WaitAsync(timeout.Token))
                {
                    Assert.Equal(access == ResourceAccess.Read ? 2 : 3, next.Version);
                    await next.ReadAsync(0, actual); Assert.Equal(expected, actual);
                }
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                scheduler.Unregister(key); Assert.Equal(0, backend.Live);
            }
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Fact]
    public async Task NoWaiterReadAndWriteReleaseAllocateNoManagedNotificationOwner()
    {
        var budget = new MemoryBudget([new("ram", 4096), new("ssd", 4096)]);
        var backend = new ManagedBackend();
        string directory = Path.Combine(Path.GetTempPath(), "ts-lease-release-allocation-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var transfers = new BoundedTransfers(budget, "ram", 64, 1);
            using var spill = new SsdSpillStore(budget, "ssd", directory, transfers);
            await using (var scheduler = new TieredMemoryScheduler(budget, [backend], transfers, spill))
            {
                var key = new ResourceKey("allocation", 0, "release");
                scheduler.Register(new(key, 64, ResourceKind.KvPage, Mutable: true));
                // Warm both public release paths and the directory/LRU locks.
                // Acquires are intentionally outside the allocation interval.
                for (int i = 0; i < 64; i++)
                {
                    using var warm = await scheduler.AcquireAsync(key, backend.Location,
                        i % 2 == 0 ? ResourceAccess.Read : ResourceAccess.Write);
                }
                var reader = await scheduler.AcquireAsync(key, backend.Location);
                long beforeRead = GC.GetAllocatedBytesForCurrentThread();
                reader.Dispose();
                long readerBytes = GC.GetAllocatedBytesForCurrentThread() - beforeRead;
                var writer = await scheduler.AcquireAsync(key, backend.Location, ResourceAccess.Write);
                long beforeWrite = GC.GetAllocatedBytesForCurrentThread();
                writer.Dispose();
                long writerBytes = GC.GetAllocatedBytesForCurrentThread() - beforeWrite;
                Assert.Equal(0, readerBytes); Assert.Equal(0, writerBytes);
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                scheduler.Unregister(key); Assert.Equal(0, backend.Live);
            }
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Fact]
    public async Task FailedSharedReadOwnerDoesNotStealOrLeakTheExistingReadLease()
    {
        var budget = new MemoryBudget([new("ram", 4096), new("ssd", 4096)]);
        var backend = new ManagedBackend();
        string directory = Path.Combine(Path.GetTempPath(), "ts-shared-lease-owner-fault-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var transfers = new BoundedTransfers(budget, "ram", 64, 1);
            using var spill = new SsdSpillStore(budget, "ssd", directory, transfers);
            await using (var scheduler = new TieredMemoryScheduler(budget, [backend], transfers, spill))
            {
                var key = new ResourceKey("fault", 0, "shared");
                scheduler.Register(new(key, 64, ResourceKind.KvPage, Mutable: true));
                using (var retained = await scheduler.AcquireAsync(key, backend.Location))
                {
                    scheduler.BeforeLeaseOwnerAllocation = () => throw new OutOfMemoryException("shared owner injection");
                    await Assert.ThrowsAsync<OutOfMemoryException>(() => scheduler.AcquireAsync(key, backend.Location).AsTask());
                    Assert.Equal(1, scheduler.GetStats().ActiveLeases);
                    Assert.Equal(1, Assert.Single(scheduler.Snapshot()).Readers);
                    Assert.Equal(1, backend.Live);
                    byte[] zeros = new byte[64]; await retained.ReadAsync(0, zeros);
                    Assert.All(zeros, value => Assert.Equal((byte)0, value));
                }
                scheduler.BeforeLeaseOwnerAllocation = null;
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                using (var writer = await scheduler.AcquireAsync(key, backend.Location, ResourceAccess.Write, timeout.Token))
                    Assert.Equal(1, scheduler.GetStats().ActiveLeases);
                scheduler.Unregister(key); Assert.Equal(0, backend.Live);
            }
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }
}
