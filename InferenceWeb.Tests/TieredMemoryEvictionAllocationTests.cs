// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

/// <summary>Real residency/ledger/SSD lifecycle with managed backend buffers.
/// Accelerator placement is simulated; no device or model qualification.</summary>
public sealed class TieredMemoryEvictionAllocationTests
{
    private sealed class Backend(MemoryLocation location, string pool) : IMemoryBackend
    {
        public MemoryLocation Location => location;
        public int Live;
        public IReadOnlyList<MemoryCharge> GetAllocationCharges(long bytes) => [new(pool, bytes)];
        public ValueTask<IResourceBuffer> AllocateAsync(long bytes, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var buffer = new Buffer(this, checked((int)bytes)); Live++;
            return ValueTask.FromResult<IResourceBuffer>(buffer);
        }
        private sealed class Buffer(Backend owner, int length) : IResourceBuffer
        {
            private readonly byte[] _data = new byte[length];
            private bool _disposed;
            public long ByteLength => _data.Length;
            public nint Pointer { get { ObjectDisposedException.ThrowIf(_disposed, this); return 1; } }
            public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
            {
                cancellationToken.ThrowIfCancellationRequested(); ObjectDisposedException.ThrowIf(_disposed, this);
                _data.AsMemory(checked((int)offset), destination.Length).CopyTo(destination);
                return ValueTask.CompletedTask;
            }
            public ValueTask WriteAsync(long offset, ReadOnlyMemory<byte> source, CancellationToken cancellationToken = default)
            {
                cancellationToken.ThrowIfCancellationRequested(); ObjectDisposedException.ThrowIf(_disposed, this);
                source.CopyTo(_data.AsMemory(checked((int)offset), source.Length));
                return ValueTask.CompletedTask;
            }
            public void Dispose() { if (!_disposed) { _disposed = true; owner.Live--; } }
        }
    }

    [Fact]
    public async Task EvictionAttemptTrackingFailureReleasesTransitionAndPreservesTheSource()
    {
        var budget = new MemoryBudget([new("ram", 128), new("ssd", 4096)]);
        var host = new Backend(new("test", "ram", MemoryTier.Host), "ram");
        string directory = Path.Combine(Path.GetTempPath(), "ts-eviction-tracking-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var transfers = new BoundedTransfers(budget, "ram", 64, 1);
            using var spill = new SsdSpillStore(budget, "ssd", directory, transfers);
            await using (var scheduler = new TieredMemoryScheduler(budget, [host], transfers, spill))
            {
                var key = new ResourceKey("allocation", 0, "source");
                var pressureKey = new ResourceKey("allocation", 1, "pressure");
                scheduler.Register(new(key, 64, ResourceKind.RecurrentState, Mutable: true));
                scheduler.Register(new(pressureKey, 64, ResourceKind.Workspace, Mutable: true));
                byte[] original = Enumerable.Range(0, 64).Select(i => (byte)(i + 31)).ToArray();
                using (var source = await scheduler.AcquireAsync(key, host.Location, ResourceAccess.Write))
                    await source.WriteAsync(0, original);
                var before = budget.Snapshot().ToArray();
                int injections = 0;
                scheduler.BeforeEvictionAttemptTracking = () =>
                {
                    injections++; throw new OutOfMemoryException("eviction tracking injection");
                };
                var error = await Assert.ThrowsAsync<OutOfMemoryException>(() => scheduler.AcquireAsync(pressureKey, host.Location).AsTask());
                Assert.Equal("eviction tracking injection", error.Message); Assert.Equal(1, injections);
                Assert.Equal(before, budget.Snapshot().ToArray());
                Assert.Equal(1, host.Live); Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                Assert.Equal(key, Assert.Single(scheduler.Snapshot()).Resource);
                Assert.Empty(Directory.GetFiles(directory, "*.bin", SearchOption.AllDirectories));
                scheduler.BeforeEvictionAttemptTracking = null;
                // A leaked transition would keep this public read blocked even
                // though the original physical source is still resident.
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(1));
                using (var recovered = await scheduler.AcquireAsync(key, host.Location, cancellationToken: timeout.Token))
                {
                    byte[] actual = new byte[64]; await recovered.ReadAsync(0, actual);
                    Assert.Equal(original, actual); Assert.Equal(1, recovered.Version);
                }
                using (var pressure = await scheduler.AcquireAsync(pressureKey, host.Location))
                    Assert.Equal(1, scheduler.GetStats().Spills);
                scheduler.Unregister(pressureKey);
                using (var recovered = await scheduler.AcquireAsync(key, host.Location))
                {
                    byte[] actual = new byte[64]; await recovered.ReadAsync(0, actual);
                    Assert.Equal(original, actual); Assert.Equal(1, recovered.Version);
                }
                scheduler.Unregister(key); Assert.Equal(0, host.Live);
            }
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }

    [Fact]
    public async Task DemotionLruOwnerFailureCannotPublishAReleasedReplica()
    {
        var budget = new MemoryBudget([new("ram", 128), new("accelerator", 64), new("ssd", 4096)]);
        var host = new Backend(new("test", "ram", MemoryTier.Host), "ram");
        var device = new Backend(new("test", "simulated-accelerator", MemoryTier.Accelerator), "accelerator");
        string directory = Path.Combine(Path.GetTempPath(), "ts-demotion-publication-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var transfers = new BoundedTransfers(budget, "ram", 64, 1);
            using var spill = new SsdSpillStore(budget, "ssd", directory, transfers);
            await using (var scheduler = new TieredMemoryScheduler(budget, [host, device], transfers, spill, host.Location))
            {
                var key = new ResourceKey("allocation", 2, "device-source");
                var pressureKey = new ResourceKey("allocation", 3, "device-pressure");
                scheduler.Register(new(key, 64, ResourceKind.KvPage, Mutable: true));
                scheduler.Register(new(pressureKey, 64, ResourceKind.Workspace, Mutable: true));
                byte[] original = Enumerable.Range(0, 64).Select(i => (byte)(i * 2 + 1)).ToArray();
                using (var source = await scheduler.AcquireAsync(key, device.Location, ResourceAccess.Write))
                    await source.WriteAsync(0, original);
                var before = budget.Snapshot().ToArray();
                int injections = 0;
                scheduler.BeforeDemotionLruNodeAllocation = () =>
                {
                    injections++; throw new OutOfMemoryException("demotion LRU owner injection");
                };
                var error = await Assert.ThrowsAsync<OutOfMemoryException>(() => scheduler.AcquireAsync(pressureKey, device.Location).AsTask());
                Assert.Equal("demotion LRU owner injection", error.Message); Assert.Equal(1, injections);
                Assert.Equal(before, budget.Snapshot().ToArray());
                Assert.Equal(0, host.Live); Assert.Equal(1, device.Live);
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                var retained = Assert.Single(scheduler.Snapshot());
                Assert.Equal(key, retained.Resource); Assert.Equal(device.Location, retained.Location);
                Assert.Empty(Directory.GetFiles(directory, "*.bin", SearchOption.AllDirectories));
                scheduler.BeforeDemotionLruNodeAllocation = null;
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                using (var recovered = await scheduler.AcquireAsync(key, device.Location, cancellationToken: timeout.Token))
                {
                    byte[] actual = new byte[64]; await recovered.ReadAsync(0, actual);
                    Assert.Equal(original, actual); Assert.Equal(1, recovered.Version);
                }
                using (var pressure = await scheduler.AcquireAsync(pressureKey, device.Location, cancellationToken: timeout.Token))
                {
                    Assert.Equal(1, host.Live); Assert.Equal(1, device.Live);
                    using var recovered = await scheduler.AcquireAsync(key, host.Location, cancellationToken: timeout.Token);
                    byte[] actual = new byte[64]; await recovered.ReadAsync(0, actual);
                    Assert.Equal(original, actual); Assert.Equal(1, recovered.Version);
                    Assert.Equal(0, scheduler.GetStats().Spills);
                }
                scheduler.Unregister(pressureKey); scheduler.Unregister(key);
                Assert.Equal(0, host.Live); Assert.Equal(0, device.Live);
            }
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
    }
}
