// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Runtime;
using System.Text;
using System.Runtime.InteropServices;
using static Fixture;

static partial class Cases
{
    public static Task Budgets()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 100), new MemoryCharge("gpu-working-set", 70) });
        using var a = budget.Reserve(new[] { new MemoryCharge("ram", 30), new MemoryCharge("ram", 30), new MemoryCharge("gpu-working-set", 60) });
        a.Commit();
        Check.True(budget.TryReserve(new[] { new MemoryCharge("ram", 20), new MemoryCharge("gpu-working-set", 20) }) == null);
        Check.Equal(40L, budget.Snapshot().Single(x => x.Pool == "ram").Available);
        Check.Equal(60L, budget.Snapshot().Single(x => x.Pool == "ram").Committed); // UMA physical once.
        Check.Throws<ArgumentOutOfRangeException>(() => budget.TryReserve(new[] { new MemoryCharge("ram", -1) }));
        Check.Throws<OverflowException>(() => budget.TryReserve(new[] { new MemoryCharge("ram", long.MaxValue), new MemoryCharge("ram", 1) }));
        Check.True(!budget.TrySetCapacity("ram", 59));
        a.Dispose(); a.Dispose();
        Check.True(budget.TrySetCapacity("ram", 59));
        return Task.CompletedTask;
    }

    public static async Task ConcurrentBudgets()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 1024), new MemoryCharge("gpu", 768) });
        await Task.WhenAll(Enumerable.Range(0, 32).Select(async _ =>
        {
            for (int i = 0; i < 100; i++)
            {
                using var ticket = budget.TryReserve(new[] { new MemoryCharge("ram", 128), new MemoryCharge("gpu", 128) });
                ticket?.Commit();
                Check.True(budget.Snapshot().All(p => p.Available >= 0));
                await Task.Yield();
            }
        }));
        Check.True(budget.Snapshot().All(p => p.Committed == 0 && p.Reserved == 0));
    }

    public static Task Envelopes()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 1000) });
        using var parent = budget.Reserve(new[] { new MemoryCharge("ram", 800) });
        using var child = parent.TryTake(new[] { new MemoryCharge("ram", 500) })!;
        child.Commit();
        Check.Equal(300L, budget.Snapshot()[0].Reserved);
        Check.Equal(500L, budget.Snapshot()[0].Committed);
        Check.True(parent.TryTake(new[] { new MemoryCharge("ram", 400) }) == null);
        Check.Throws<InvalidOperationException>(parent.Commit);
        child.Dispose();
        Check.Equal(800L, budget.Snapshot()[0].Reserved);
        var survivor = parent.TryTake(new[] { new MemoryCharge("ram", 700) })!;
        survivor.Commit(); parent.Dispose();
        Check.Equal(700L, budget.Snapshot()[0].Committed);
        Check.Equal(0L, budget.Snapshot()[0].Reserved);
        survivor.Dispose();
        Check.Equal(1000L, budget.Snapshot()[0].Available);
        return Task.CompletedTask;
    }

    public static async Task Admission()
    {
        await using var f = new Fixture(ram: 2 * Page);
        var queue = new MemoryRequestQueue(f.Budget, 8);
        Check.Throws<MemoryPressureException>(() => queue.Enqueue("too-large", new[] { new MemoryCharge("ram", 100 * Page) }));
        for (int i = 0; i < 8; i++) queue.Enqueue($"r{i}", new[] { new MemoryCharge("ram", Page) });
        Check.Throws<MemoryPressureException>(() => queue.Enqueue("overflow", new[] { new MemoryCharge("ram", Page) }));
        Check.True(queue.CancelWaiting("r7"));
        using var first = queue.TryAdmit()!;
        using var second = queue.TryAdmit()!;
        Check.Equal("r0", first.RequestId); Check.Equal("r1", second.RequestId);
        Check.True(queue.TryAdmit() == null);
        var a = f.Add("r0-state", true);
        var b = f.Add("r1-state", true);
        using (var lease = await f.Scheduler.AcquireAsync(a, f.Host.Location, ResourceAccess.Write, allocationEnvelope: first.Envelope))
            await lease.WriteAsync(0, new byte[] { 23 });
        using (var lease = await f.Scheduler.AcquireAsync(b, f.Host.Location, ResourceAccess.Write, allocationEnvelope: second.Envelope))
            await lease.WriteAsync(0, new byte[] { 71 });
        Check.Equal(3L * Page, f.Budget.Snapshot().Single(p => p.Pool == "ram").Committed); // staging + 2 states.
        f.Scheduler.Unregister(a); // returns credit to live first envelope.
        Check.True(queue.TryAdmit() == null);
        first.Dispose();
        using var third = queue.TryAdmit()!;
        Check.Equal("r2", third.RequestId);
        second.Dispose(); // b remains physically charged despite closing its request.
        Check.True(queue.TryAdmit() == null);
        f.Scheduler.Unregister(b);
        using var fourth = queue.TryAdmit()!;
        Check.Equal("r3", fourth.RequestId);
    }

    public static async Task FileWorkingSet()
    {
        const int pages = 256;
        await using var f = new Fixture(ram: 2 * Page);
        string path = Path.Combine(f.Root, "weights.bin");
        using (var stream = File.Create(path))
            for (int p = 0; p < pages; p++) stream.Write(Enumerable.Repeat((byte)(p % 251), Page).ToArray());
        using var file = new FileDataSource(path);
        for (int p = 0; p < pages; p++) f.Add($"p{p}", source: new FileRegionSource(file, p * Page, Page));
        for (int pass = 0; pass < 2; pass++)
            for (int p = 0; p < pages; p++)
            {
                var value = await f.Read(Key($"p{p}"));
                Check.True(value.All(x => x == (byte)(p % 251)));
                Check.True(f.Budget.Snapshot().All(x => x.Available >= 0));
            }
        Check.True(f.Scheduler.GetStats().Evictions >= pages - 2);
        for (int p = 0; p < pages; p++) f.Scheduler.Unregister(Key($"p{p}"));
    }

    public static async Task SingleFlight()
    {
        await using var f = new Fixture();
        var source = new ControlledSource(Page);
        var key = f.Add("coalesce", source: source);
        var tasks = Enumerable.Range(0, 32).Select(_ => f.Scheduler.AcquireAsync(key, f.Host.Location).AsTask()).ToArray();
        await source.Started.Task;
        source.Continue.TrySetResult();
        var leases = await Task.WhenAll(tasks);
        Check.Equal(1, source.Calls);
        Check.Equal(1L, f.Scheduler.GetStats().Loads);
        Check.Equal(32, f.Scheduler.GetStats().ActiveLeases);
        foreach (var lease in leases) lease.Dispose();
    }

    public static async Task Writers()
    {
        await using var f = new Fixture();
        var key = f.Add("mutable", true);
        var reader = await f.Scheduler.AcquireAsync(key, f.Host.Location);
        using (var cts = new CancellationTokenSource())
        {
            var writer = f.Scheduler.AcquireAsync(key, f.Gpu.Location, ResourceAccess.Write, cts.Token).AsTask();
            Check.True(!writer.IsCompleted);
            cts.Cancel();
            await Check.ThrowsAsync<OperationCanceledException>(() => writer);
        }
        reader.Dispose();
        using (var writer = await f.Scheduler.AcquireAsync(key, f.Gpu.Location, ResourceAccess.Write))
        {
            await writer.WriteAsync(0, new byte[] { 99, 18 });
            Check.Equal(1L, writer.Version);
            Check.Equal(1, f.Scheduler.Snapshot().Count);
        }
        Check.Equal((byte)99, (await f.Read(key))[0]);
        using var readOnly = await f.Scheduler.AcquireAsync(key, f.Host.Location);
        Check.Throws<InvalidOperationException>(() => readOnly.WriteAsync(0, new byte[] { 1 }));
    }

    public static async Task SpillRoundTrip()
    {
        await using var f = new Fixture(ram: Page, chunk: 256);
        var a = f.Add("state", true);
        var b = f.Add("weight");
        for (int version = 1; version <= 4; version++)
        {
            using (var writer = await f.Scheduler.AcquireAsync(a, f.Host.Location, ResourceAccess.Write))
            {
                await writer.WriteAsync(0, Enumerable.Repeat((byte)version, Page).ToArray());
                Check.Equal((long)version, writer.Version);
            }
            await f.Read(b); // evicts a, atomically publishing an SSD snapshot.
            Check.Equal((long)Page, f.Budget.Snapshot().Single(x => x.Pool == "ssd").Committed);
            Check.True((await f.Read(a)).All(x => x == (byte)version));
        }
        Check.Equal(4L, f.Scheduler.GetStats().Spills);
    }

    public static async Task CorruptSpill()
    {
        await using var f = new Fixture(ram: Page);
        var a = f.Add("state", true); var b = f.Add("other");
        using (var writer = await f.Scheduler.AcquireAsync(a, f.Host.Location, ResourceAccess.Write)) await writer.WriteAsync(0, new byte[] { 44 });
        await f.Read(b);
        string path = Directory.GetFiles(f.Root, "*.bin", SearchOption.AllDirectories).Single();
        // An external writer/corrupt medium is outside FileShare's advisory protection
        // on Unix; the runtime must still detect changed bytes when restoring.
        if (OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Use a filesystem fault injector for this corruption case on Windows.");
        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite)) stream.WriteByte(45);
        await Check.ThrowsAsync<InvalidDataException>(() => f.Read(a));
        Check.Equal(0, f.Scheduler.GetStats().ActiveLeases);
        Check.True(f.Scheduler.Snapshot().All(r => r.Resource != a));
    }

    public static async Task FullDisk()
    {
        await using var f = new Fixture(ram: Page, disk: 0);
        var a = f.Add("state", true); var b = f.Add("other");
        using (var lease = await f.Scheduler.AcquireAsync(a, f.Host.Location, ResourceAccess.Write)) await lease.WriteAsync(0, new byte[] { 91 });
        await Check.ThrowsAsync<MemoryPressureException>(() => f.Read(b));
        Check.Equal((byte)91, (await f.Read(a))[0]);
        Check.Equal(0L, f.Budget.Snapshot().Single(x => x.Pool == "ssd").Committed);
    }

    public static async Task CancelTransfer()
    {
        await using var f = new Fixture();
        var source = new ControlledSource(Page);
        var key = f.Add("cancel", source: source);
        using var cts = new CancellationTokenSource();
        var pending = f.Scheduler.AcquireAsync(key, f.Host.Location, cancellationToken: cts.Token).AsTask();
        await source.Started.Task;
        cts.Cancel();
        await Check.ThrowsAsync<OperationCanceledException>(() => pending);
        Check.Equal(0, f.Scheduler.Snapshot().Count);
        Check.Equal((long)Page, f.Budget.Snapshot().Single(x => x.Pool == "ram").Committed); // only staging.
        source.Continue.TrySetResult();
        Check.Equal((byte)37, (await f.Read(key))[0]);
    }

    private sealed class FailOnceSource : IResourceSource
    {
        private int _calls;
        public long ByteLength => Page;
        public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            if (Interlocked.Increment(ref _calls) == 1) throw new IOException("Injected read failure");
            destination.Span.Fill(64);
            return ValueTask.CompletedTask;
        }
    }
    public static async Task FailedRead()
    {
        await using var f = new Fixture();
        var key = f.Add("fail", source: new FailOnceSource());
        await Check.ThrowsAsync<IOException>(() => f.Read(key));
        Check.Equal(0, f.Scheduler.Snapshot().Count);
        Check.Equal((byte)64, (await f.Read(key))[0]);
    }
    public static async Task FailedAllocation()
    {
        await using var f = new Fixture(failOnce: true);
        var key = f.Add("oom");
        await Check.ThrowsAsync<OutOfMemoryException>(() => f.Read(key));
        Check.Equal(0L, f.Budget.Snapshot().Single(x => x.Pool == "ram").Reserved);
        await f.Read(key);
    }

    public static async Task Fence()
    {
        await using var f = new Fixture(ram: Page);
        var a = f.Add("a"); var b = f.Add("b");
        var lease = await f.Scheduler.AcquireAsync(a, f.Host.Location);
        var fence = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var retiring = lease.ReleaseAfterAsync(fence.Task).AsTask();
        Check.Throws<InvalidOperationException>(lease.Dispose);
        await Check.ThrowsAsync<MemoryPressureException>(() => f.Read(b));
        fence.SetResult(); await retiring;
        await f.Read(b);
    }
    public static async Task FailedFence()
    {
        await using var f = new Fixture();
        var lease = await f.Scheduler.AcquireAsync(f.Add("a"), f.Host.Location);
        await Check.ThrowsAsync<IOException>(() => lease.ReleaseAfterAsync(Task.FromException(new IOException("Device fence failed"))).AsTask());
        Check.Equal(1, f.Scheduler.GetStats().ActiveLeases);
        lease.Dispose(); // test backend is synchronous; recovery has quiesced it.
    }

    public static async Task WorkingSet()
    {
        await using var f = new Fixture(ram: Page);
        var a = f.Add("a"); var b = f.Add("b");
        await Check.ThrowsAsync<MemoryPressureException>(() => f.Scheduler.AcquireReadSetAsync(new[] { a, b }, f.Host.Location).AsTask());
        Check.Equal(0, f.Scheduler.GetStats().ActiveLeases);
        using var duplicate = await f.Scheduler.AcquireReadSetAsync(new[] { a, a }, f.Host.Location);
        Check.Equal(1, duplicate.Leases.Count);
    }
    public static async Task Prefetch()
    {
        await using var f = new Fixture(ram: Page);
        var a = f.Add("a"); var b = f.Add("b");
        await f.Read(a);
        Check.True(!await f.Scheduler.TryPrefetchAsync(b, f.Host.Location));
        Check.Equal(0L, f.Scheduler.GetStats().Evictions);
        Check.Equal(a, f.Scheduler.Snapshot().Single().Resource);
    }
    public static async Task Demotion()
    {
        await using var f = new Fixture();
        var a = f.Add("a"); var b = f.Add("b");
        await f.Read(a, f.Gpu.Location);
        await f.Read(b, f.Gpu.Location);
        Check.True(f.Scheduler.Snapshot().Any(x => x.Resource == a && x.Location == f.Host.Location));
        long bytes = f.Transfers.BytesCopied;
        await f.Read(a);
        Check.Equal(bytes, f.Transfers.BytesCopied); // hits demoted bytes, no backing-file read.
    }
    public static async Task Lifecycle()
    {
        await using var f = new Fixture();
        var key = f.Add("live");
        var lease = await f.Scheduler.AcquireAsync(key, f.Host.Location);
        Check.Throws<InvalidOperationException>(() => f.Scheduler.Unregister(key));
        await Check.ThrowsAsync<InvalidOperationException>(() => f.Scheduler.DisposeAsync().AsTask());
        lease.Dispose(); lease.Dispose();
        f.Scheduler.Unregister(key);
        Check.Throws<ObjectDisposedException>(() => _ = lease.Pointer);
    }

    public static async Task ConcurrentState()
    {
        await using var f = new Fixture(ram: 4 * Page, disk: 32 * Page, chunk: 512);
        var keys = Enumerable.Range(0, 12).Select(i => f.Add($"state{i}", true)).ToArray();
        await Task.WhenAll(Enumerable.Range(0, 8).Select(async worker =>
        {
            var random = new Random(3900 + worker);
            for (int i = 0; i < 40; i++)
            {
                var key = keys[random.Next(keys.Length)];
                for (int attempt = 0; ; attempt++)
                {
                    try
                    {
                        using var lease = await f.Scheduler.AcquireAsync(key, f.Host.Location, ResourceAccess.Write);
                        var bytes = new byte[8];
                        await lease.ReadAsync(0, bytes);
                        long value = BitConverter.ToInt64(bytes) + 1;
                        await lease.WriteAsync(0, BitConverter.GetBytes(value));
                        Check.True(f.Budget.Snapshot().All(p => p.Available >= 0));
                        break;
                    }
                    catch (MemoryPressureException) when (attempt < 10000) { await Task.Yield(); }
                }
            }
        }));
        long total = 0;
        foreach (var key in keys) total += BitConverter.ToInt64(await f.Read(key));
        Check.Equal(320L, total);
    }

    public static Task Placement()
    {
        const long g = 1L << 30;
        long[] groups = { 4 * g, 3 * g, 2 * g, g };
        int Plan(long free, long cache = 0, long[]? minimum = null, long[]? floor = null)
            => TieredPlacementPlanner.PlanDiscreteResidency(groups, g, free, g, 3 * g, cache, minimum, 4, floor);
        Check.Equal(2, Plan(10 * g)); Check.Equal(0, Plan(4 * g));
        Check.Equal(0, Plan(10 * g, 4 * g, new[] { g, g, g, g }));
        Check.Equal(4, Plan(15 * g, 4 * g, new[] { g, g, g, g }));
        Check.Equal(1, Plan(9 * g, 8 * g, new[] { g, 0L, 0L, 0L }));
        Check.Equal(0, Plan(9 * g, 8 * g, new[] { g, 0L, g, 0L }));
        Check.Equal(1, Plan(9 * g, 8 * g, new[] { 3 * g, 0L, 0L, 0L }, new[] { g, 0L, 0L, 0L }));
        Check.Equal(0, TieredPlacementPlanner.PlanDiscreteResidency(new[] { long.MaxValue }, long.MaxValue, long.MaxValue, 1, 1, 0));
        Check.Equal(2, TieredPlacementPlanner.PlanUnifiedResidency(new[] { 4 * g, 4 * g, 4 * g }, 2 * g, 16 * g, 24 * g, 8 * g, g, 3 * g, 0.5));
        Check.Equal(0, TieredPlacementPlanner.PlanUnifiedResidency(new[] { long.MaxValue }, long.MaxValue, long.MaxValue, long.MaxValue, 1, 1, 1, 1));
        return Task.CompletedTask;
    }
}
