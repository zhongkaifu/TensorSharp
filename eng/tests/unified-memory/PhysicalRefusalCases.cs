// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using static Fixture;

static partial class Cases
{
    public static async Task PhysicalRefusalDemotion()
    {
        foreach (long disk in new[] { 0L, (long)Page * 8 })
        {
            await using var f = new Fixture(ram: 8 * Page, gpu: Page, physicalPages: 0, disk: disk);
            var a = f.Add("device-authoritative", true); var b = f.Add("device-next");
            var expected = Enumerable.Repeat((byte)93, Page).ToArray();
            using (var lease = await f.Scheduler.AcquireAsync(a, f.Gpu.Location, ResourceAccess.Write))
                await lease.WriteAsync(0, expected);
            if (disk == 0)
            {
                await Check.ThrowsAsync<MemoryPressureException>(() => f.Scheduler.AcquireAsync(b, f.Gpu.Location).AsTask());
                Check.Equal(0L, f.Scheduler.GetStats().Evictions);
            }
            else
            {
                using var lease = await f.Scheduler.AcquireAsync(b, f.Gpu.Location);
                Check.Equal(1L, f.Scheduler.GetStats().Spills);
            }
            var restored = await f.Read(a, f.Gpu.Location);
            Check.Bytes(expected, restored);
            Check.True(f.Scheduler.GetStats().PhysicalAllocationRefusals > 0);
            Check.Equal((long)Page, f.Budget.Snapshot().Single(p => p.Pool == "ram").Committed); // Transfer staging only.
        }
    }

    public static async Task PhysicalRefusalSpill()
    {
        await using var f = new Fixture(ram: 8 * Page, physicalPages: 1);
        using var aCredit = f.Budget.Reserve([new("ram", Page), new("ssd", Page)]);
        using var bCredit = f.Budget.Reserve([new("ram", Page), new("ssd", Page)]);
        var a = f.Add("physical-a", true); var b = f.Add("physical-b", true);
        var expected = Enumerable.Repeat((byte)71, Page).ToArray();
        using (var lease = await f.Scheduler.AcquireAsync(a, f.Host.Location, ResourceAccess.Write, allocationEnvelope: aCredit))
            await lease.WriteAsync(0, expected);
        using (var lease = await f.Scheduler.AcquireAsync(b, f.Host.Location, ResourceAccess.Write, allocationEnvelope: bCredit))
            await lease.WriteAsync(0, Enumerable.Repeat((byte)23, Page).ToArray());
        Check.Equal(1L, f.Scheduler.GetStats().Spills);
        Check.Equal((long)Page, aCredit.Charges.Single(c => c.Pool == "ram").Bytes);
        using (var lease = await f.Scheduler.AcquireAsync(a, f.Host.Location, allocationEnvelope: aCredit))
        {
            var actual = new byte[Page];
            await lease.ReadAsync(0, actual);
            Check.Bytes(expected, actual);
        }
        Check.Equal(2L, f.Scheduler.GetStats().PhysicalAllocationRefusals);
        Check.Equal(2L, f.Scheduler.GetStats().Spills);
        Check.Equal(0, f.Scheduler.GetStats().ActiveLeases);
    }

    public static async Task PhysicalRefusalPinned()
    {
        await using var f = new Fixture(ram: 8 * Page, physicalPages: 1);
        var a = f.Add("pinned"); var b = f.Add("refused");
        using var lease = await f.Scheduler.AcquireAsync(a, f.Host.Location);
        await Check.ThrowsAsync<OutOfMemoryException>(() => f.Scheduler.AcquireAsync(b, f.Host.Location).AsTask());
        Check.Equal(0L, f.Scheduler.GetStats().Evictions);
        Check.Equal(1, f.Scheduler.GetStats().ActiveLeases);
        Check.Equal(1L, f.Scheduler.GetStats().PhysicalAllocationRefusals);
        Check.Equal(a, f.Scheduler.Snapshot().Single().Resource);
    }

    public static async Task PhysicalRefusalPrefetch()
    {
        await using var f = new Fixture(ram: 8 * Page, physicalPages: 1);
        var a = f.Add("demand"); var b = f.Add("prefetch");
        var before = await f.Read(a);
        Check.True(!await f.Scheduler.TryPrefetchAsync(b, f.Host.Location));
        Check.Equal(0L, f.Scheduler.GetStats().Evictions);
        var after = await f.Read(a);
        Check.Bytes(before, after);
        Check.Equal(1L, f.Scheduler.GetStats().PhysicalAllocationRefusals);
    }

    public static async Task PhysicalRefusalFullDisk()
    {
        await using var f = new Fixture(ram: 8 * Page, physicalPages: 1, disk: 0);
        var a = f.Add("authoritative", true); var b = f.Add("cannot-fit");
        using (var lease = await f.Scheduler.AcquireAsync(a, f.Host.Location, ResourceAccess.Write))
            await lease.WriteAsync(0, Enumerable.Repeat((byte)39, Page).ToArray());
        await Check.ThrowsAsync<OutOfMemoryException>(() => f.Scheduler.AcquireAsync(b, f.Host.Location).AsTask());
        Check.Equal(0L, f.Scheduler.GetStats().Evictions);
        Check.True(f.Scheduler.GetStats().PhysicalAllocationRefusals <= 2, "Unbounded physical allocation retries");
        Check.True((await f.Read(a)).All(b => b == 39));
    }
}

// Real host storage behind an intentionally smaller allocator than the ledger.
// This models an OS refusal deterministically; it does not claim a physical cap.
sealed class PhysicallyLimitedBackend(HostMemoryBackend host, long capacity) : IMemoryBackend
{
    private long _live;
    public MemoryLocation Location => host.Location;
    public IReadOnlyList<MemoryCharge> GetAllocationCharges(long bytes) => host.GetAllocationCharges(bytes);
    public async ValueTask<IResourceBuffer> AllocateAsync(long bytes, CancellationToken cancellationToken = default)
    {
        if (Interlocked.Add(ref _live, bytes) > capacity)
        {
            Interlocked.Add(ref _live, -bytes);
            throw new OutOfMemoryException("Injected physical allocator ceiling");
        }
        try { return new Owned(await host.AllocateAsync(bytes, cancellationToken), () => Interlocked.Add(ref _live, -bytes)); }
        catch { Interlocked.Add(ref _live, -bytes); throw; }
    }
    private sealed class Owned(IResourceBuffer inner, Action released) : IResourceBuffer
    {
        private bool _disposed;
        public long ByteLength => inner.ByteLength;
        public nint Pointer => inner.Pointer;
        public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
            => inner.ReadAsync(offset, destination, cancellationToken);
        public ValueTask WriteAsync(long offset, ReadOnlyMemory<byte> source, CancellationToken cancellationToken = default)
            => inner.WriteAsync(offset, source, cancellationToken);
        public void Dispose() { if (_disposed) return; inner.Dispose(); released(); _disposed = true; }
    }
}
