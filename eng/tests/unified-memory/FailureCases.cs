// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using static Fixture;

static partial class Cases
{
    public static async Task FailedRollbackCleanup()
    {
        await using var f = new CleanupFixture(failWrite: false);
        var key = f.Add("rollback", new FailOnceSource());
        using var envelope = f.Budget.Reserve(new[] { new MemoryCharge("ram", Page) });
        await Check.ThrowsAsync<AggregateException>(() => f.Scheduler.AcquireAsync(key, f.Host.Location, allocationEnvelope: envelope).AsTask());
        Check.Equal(0, f.Scheduler.Snapshot().Count); // A partial copy must not be a cache hit.
        Check.Equal(2L * Page, f.HostCommitted);
        Check.Equal(0L, envelope.Charges.Single().Bytes); // A failed free cannot refund request credit.
        await Check.ThrowsAsync<InvalidOperationException>(() => f.Scheduler.AcquireAsync(key, f.Host.Location).AsTask());
        f.Scheduler.Unregister(key); // Retry physical release after the injected failure.
        Check.Equal((long)Page, f.HostCommitted);
        Check.Equal((long)Page, envelope.Charges.Single().Bytes);
        f.Add("rollback", new BytesSource(new byte[Page]));
        using var lease = await f.Scheduler.AcquireAsync(key, f.Host.Location);
        Check.Equal(1, f.Scheduler.GetStats().ActiveLeases);
    }

    public static async Task FailedDemotionCleanup()
    {
        await using var f = new CleanupFixture(failWrite: true);
        var first = f.Add("first", new BytesSource(Enumerable.Repeat((byte)39, Page).ToArray()));
        var second = f.Add("second", new BytesSource(new byte[Page]));
        using (await f.Scheduler.AcquireAsync(first, f.Gpu.Location)) { }
        await Check.ThrowsAsync<AggregateException>(() => f.Scheduler.AcquireAsync(second, f.Gpu.Location).AsTask());
        Check.Equal(2L * Page, f.HostCommitted); // Failed destination plus staging.
        Check.Equal((long)Page, f.Budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        Check.Equal(first, f.Scheduler.Snapshot().Single().Resource); // Authoritative source retained.
        await Check.ThrowsAsync<InvalidOperationException>(() => f.Scheduler.AcquireAsync(first, f.Gpu.Location).AsTask());
        f.Scheduler.Unregister(first);
        Check.Equal((long)Page, f.HostCommitted);
        Check.Equal(0L, f.Budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        using var retry = await f.Scheduler.AcquireAsync(second, f.Gpu.Location);
    }

    public static async Task FailedInitializationCleanup()
    {
        await using var f = new CleanupFixture(failWrite: false, failInitialization: true);
        var key = f.Add("initialization", new BytesSource(new byte[Page]));
        await Check.ThrowsAsync<ResourceAllocationException>(() => f.Scheduler.AcquireAsync(key, f.Host.Location).AsTask());
        Check.Equal(2L * Page, f.HostCommitted);
        Check.Equal(0L, f.Budget.Snapshot().Single(p => p.Pool == "ram").Reserved);
        await Check.ThrowsAsync<InvalidOperationException>(() => f.Scheduler.AcquireAsync(key, f.Host.Location).AsTask());
        // A failed cleanup retry must leave ownership reachable for the next retry.
        Check.Throws<IOException>(() => f.Scheduler.Unregister(key));
        Check.Equal(2L * Page, f.HostCommitted);
        f.Scheduler.Unregister(key);
        Check.Equal((long)Page, f.HostCommitted);
    }

    public static async Task FailedResidentCleanup()
    {
        await using var f = new CleanupFixture(failWrite: false);
        var key = f.Add("resident", new BytesSource(new byte[Page]));
        using (await f.Scheduler.AcquireAsync(key, f.Host.Location)) { }
        Check.Throws<IOException>(() => f.Scheduler.Unregister(key));
        Check.Equal(1, f.Scheduler.Snapshot().Count);
        Check.Equal(2L * Page, f.HostCommitted);
        await Check.ThrowsAsync<InvalidOperationException>(() => f.Scheduler.AcquireAsync(key, f.Host.Location).AsTask());
        f.Scheduler.Unregister(key);
        Check.Equal((long)Page, f.HostCommitted);
    }

    public static Task AdmissionCapacityReduction()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 1000) });
        var queue = new MemoryRequestQueue(budget);
        queue.Enqueue("large", new[] { new MemoryCharge("ram", 800) });
        queue.Enqueue("small", new[] { new MemoryCharge("ram", 400) });
        Check.True(budget.TrySetCapacity("ram", 600));
        Check.Throws<MemoryPressureException>(() => queue.TryAdmit());
        Check.Equal(1, queue.WaitingCount);
        Check.Equal(0, queue.ActiveCount);
        using var admitted = queue.TryAdmit()!;
        Check.Equal("small", admitted.RequestId);
        Check.Equal(0, queue.WaitingCount);
        return Task.CompletedTask;
    }

    private sealed class CleanupBackend(bool failWrite, bool failInitialization) : IMemoryBackend
    {
        private readonly HostMemoryBackend _host = new("ram");
        private int _allocations;
        public MemoryLocation Location => _host.Location;
        public IReadOnlyList<MemoryCharge> GetAllocationCharges(long bytes) => _host.GetAllocationCharges(bytes);
        public async ValueTask<IResourceBuffer> AllocateAsync(long bytes, CancellationToken cancellationToken = default)
        {
            var underlying = await _host.AllocateAsync(bytes, cancellationToken);
            if (Interlocked.Increment(ref _allocations) != 1) return underlying;
            var buffer = new CleanupBuffer(underlying, failWrite);
            if (failInitialization)
                throw new ResourceAllocationException(buffer, new IOException("Injected initialization and cleanup failure"));
            return buffer;
        }
    }

    private sealed class CleanupBuffer(IResourceBuffer underlying, bool failWrite) : IResourceBuffer
    {
        private bool _failRelease = true;
        public long ByteLength => underlying.ByteLength;
        public nint Pointer => underlying.Pointer;
        public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
            => underlying.ReadAsync(offset, destination, cancellationToken);
        public ValueTask WriteAsync(long offset, ReadOnlyMemory<byte> source, CancellationToken cancellationToken = default)
            => failWrite ? throw new IOException("Injected destination transfer failure") : underlying.WriteAsync(offset, source, cancellationToken);
        public void Dispose()
        {
            if (_failRelease) { _failRelease = false; throw new IOException("Injected physical release failure"); }
            underlying.Dispose();
        }
    }

    private sealed class CleanupFixture : IAsyncDisposable
    {
        private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-cleanup-test-" + Guid.NewGuid().ToString("N"));
        public readonly IMemoryBackend Host;
        public readonly SimulatedAccelerator Gpu = new(new HostMemoryBackend("ram"));
        public readonly MemoryBudget Budget = new(new[] { new MemoryCharge("ram", Page * 4), new MemoryCharge("gpu", Page), new MemoryCharge("ssd", Page * 8) });
        private readonly BoundedTransfers _transfers;
        private readonly SsdSpillStore _spill;
        public readonly TieredMemoryScheduler Scheduler;
        public long HostCommitted => Budget.Snapshot().Single(p => p.Pool == "ram").Committed;
        public CleanupFixture(bool failWrite, bool failInitialization = false)
        {
            Host = new CleanupBackend(failWrite, failInitialization);
            _transfers = new(Budget, "ram", Page, 1);
            _spill = new(Budget, "ssd", _root, _transfers);
            Scheduler = new(Budget, new[] { Host, Gpu }, _transfers, _spill, Host.Location);
        }
        public ResourceKey Add(string name, IResourceSource source)
        {
            var key = Key(name);
            Scheduler.Register(new(key, Page, ResourceKind.Weight), source);
            return key;
        }
        public async ValueTask DisposeAsync()
        {
            await Scheduler.DisposeAsync();
            _spill.Dispose();
            _transfers.Dispose();
            Check.True(Budget.Snapshot().All(p => p.Reserved == 0 && p.Committed == 0));
            Directory.Delete(_root);
        }
    }
}
