// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.Memory;

var tests = new (string Name, Func<Task> Run)[]
{
    ("atomic multi-pool reservation and UMA constraints", Cases.Budgets),
    ("parallel reservation pressure", Cases.ConcurrentBudgets),
    ("request envelopes avoid double accounting", Cases.Envelopes),
    ("bounded FIFO request admission and cancellation", Cases.Admission),
    ("real file working set larger than managed RAM budget", Cases.FileWorkingSet),
    ("concurrent readers coalesce one load", Cases.SingleFlight),
    ("exclusive writers and version invalidation [simulated accelerator]", Cases.Writers),
    ("lossless mutable RAM-SSD round trip and rewrite", Cases.SpillRoundTrip),
    ("SSD checksum rejects corrupt state", Cases.CorruptSpill),
    ("SSD quota failure preserves authoritative RAM data", Cases.FullDisk),
    ("cancelled transfer rolls back unpublished allocation", Cases.CancelTransfer),
    ("failed read rolls back and can retry", Cases.FailedRead),
    ("failed allocation rolls back reservation", Cases.FailedAllocation),
    ("execution fence prevents eviction", Cases.Fence),
    ("failed fence retains a lease", Cases.FailedFence),
    ("partial working-set acquisition releases all pins", Cases.WorkingSet),
    ("prefetch cannot evict demand data", Cases.Prefetch),
    ("accelerator eviction demotes to RAM [simulated accelerator]", Cases.Demotion),
    ("shutdown and unregister refuse live pointers", Cases.Lifecycle),
    ("concurrent writes under eviction retain every update", Cases.ConcurrentState),
    ("split GGUF catalog resolves actual shard and quantized bytes", Cases.GgufCatalog),
    ("existing placement policy regression vectors", Cases.Placement),
    ("batched streaming F32 matvec equals full mathematical reference", Cases.StreamingMatvec),
};
var results = new List<object>();
int failed = 0;
foreach (var (name, run) in tests)
{
    var clock = Stopwatch.StartNew();
    try
    {
        await run().WaitAsync(TimeSpan.FromSeconds(45));
        Console.WriteLine($"PASS {name} ({clock.ElapsedMilliseconds} ms)");
        results.Add(new { name, passed = true, milliseconds = clock.ElapsedMilliseconds });
    }
    catch (Exception ex)
    {
        failed++;
        Console.WriteLine($"FAIL {name}: {ex}");
        results.Add(new { name, passed = false, milliseconds = clock.ElapsedMilliseconds, error = ex.ToString() });
    }
}
Console.WriteLine($"{tests.Length - failed}/{tests.Length} passed. CUDA/Metal/Vulkan hardware and production model inference: NOT RUN.");
if (args.Length == 2 && args[0] == "--json")
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1]))!);
    await File.WriteAllTextAsync(args[1], JsonSerializer.Serialize(new
    {
        tests = results,
        hardwareInference = "not run",
        note = "Accelerator placement cases use a host-memory test double. File I/O is real; the physical storage medium is not asserted to be NVMe.",
    }, new JsonSerializerOptions { WriteIndented = true }));
}
return failed == 0 ? 0 : 1;

static class Check
{
    public static void True(bool value, string message = "Assertion failed") { if (!value) throw new Exception(message); }
    public static void Equal<T>(T expected, T actual) { if (!EqualityComparer<T>.Default.Equals(expected, actual)) throw new Exception($"Expected {expected}, got {actual}"); }
    public static void Bytes(ReadOnlySpan<byte> expected, ReadOnlySpan<byte> actual) => True(expected.SequenceEqual(actual), "Byte mismatch");
    public static void Throws<T>(Action action) where T : Exception
    { try { action(); } catch (T) { return; } throw new Exception($"Expected {typeof(T).Name}"); }
    public static async Task ThrowsAsync<T>(Func<Task> action) where T : Exception
    { try { await action(); } catch (T) { return; } throw new Exception($"Expected {typeof(T).Name}"); }
}

sealed class BytesSource(byte[] bytes) : IResourceSource
{
    public long ByteLength => bytes.Length;
    public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
    { cancellationToken.ThrowIfCancellationRequested(); bytes.AsMemory(checked((int)offset), destination.Length).CopyTo(destination); return ValueTask.CompletedTask; }
}

sealed class ControlledSource(int length) : IResourceSource
{
    public readonly TaskCompletionSource Started = new(TaskCreationOptions.RunContinuationsAsynchronously);
    public readonly TaskCompletionSource Continue = new(TaskCreationOptions.RunContinuationsAsynchronously);
    public int Calls;
    public long ByteLength => length;
    public async ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref Calls);
        Started.TrySetResult();
        await Continue.Task.WaitAsync(cancellationToken);
        destination.Span.Fill(37);
    }
}

// Only a state-machine test double. It neither allocates VRAM nor proves GPU support.
sealed class SimulatedAccelerator(HostMemoryBackend host) : IMemoryBackend
{
    public MemoryLocation Location => new("local", "SIMULATED-GPU", MemoryTier.Accelerator);
    public IReadOnlyList<MemoryCharge> GetAllocationCharges(long byteLength) => new[] { new MemoryCharge("gpu", (byteLength + 63) / 64 * 64) };
    public ValueTask<IResourceBuffer> AllocateAsync(long byteLength, CancellationToken cancellationToken = default) => host.AllocateAsync(byteLength, cancellationToken);
}

sealed class FailOnceBackend(HostMemoryBackend host) : IMemoryBackend
{
    private int _calls;
    public MemoryLocation Location => host.Location;
    public IReadOnlyList<MemoryCharge> GetAllocationCharges(long bytes) => host.GetAllocationCharges(bytes);
    public ValueTask<IResourceBuffer> AllocateAsync(long bytes, CancellationToken cancellationToken = default)
    {
        if (Interlocked.Increment(ref _calls) == 1) throw new OutOfMemoryException("Injected allocation failure");
        return host.AllocateAsync(bytes, cancellationToken);
    }
}

sealed class Fixture : IAsyncDisposable
{
    public const int Page = 4096;
    public readonly string Root = Path.Combine(Path.GetTempPath(), "ts-unified-test-" + Guid.NewGuid().ToString("N"));
    public readonly HostMemoryBackend Host = new("ram");
    public readonly SimulatedAccelerator Gpu;
    public readonly MemoryBudget Budget;
    public readonly BoundedTransfers Transfers;
    public readonly SsdSpillStore Spill;
    public readonly TieredMemoryScheduler Scheduler;
    public Fixture(long ram = Page * 2, long gpu = Page, long disk = Page * 64, int chunk = Page, bool failOnce = false)
    {
        Directory.CreateDirectory(Root);
        Budget = new(new[] { new MemoryCharge("ram", ram + chunk), new MemoryCharge("gpu", gpu), new MemoryCharge("ssd", disk) });
        Transfers = new(Budget, "ram", chunk, 1);
        Spill = new(Budget, "ssd", Root, Transfers);
        Gpu = new(Host);
        Scheduler = new(Budget, new IMemoryBackend[] { failOnce ? new FailOnceBackend(Host) : Host, Gpu }, Transfers, Spill, Host.Location);
    }
    public static ResourceKey Key(string name) => new("fixture", 0, name);
    public ResourceKey Add(string name, bool mutable = false, int bytes = Page, IResourceSource? source = null)
    {
        var key = Key(name);
        Scheduler.Register(new(key, bytes, mutable ? ResourceKind.KvPage : ResourceKind.Weight, mutable),
            source ?? (mutable ? null : new BytesSource(Enumerable.Range(0, bytes).Select(i => (byte)(i % 251)).ToArray())));
        return key;
    }
    public async ValueTask DisposeAsync()
    {
        await Scheduler.DisposeAsync();
        Spill.Dispose();
        Transfers.Dispose();
        Check.True(Budget.Snapshot().All(x => x.Committed == 0 && x.Reserved == 0), "Leaked resource charge");
        Directory.Delete(Root, true);
    }
    public async Task<byte[]> Read(ResourceKey key, MemoryLocation? location = null)
    {
        using var lease = await Scheduler.AcquireAsync(key, location ?? Host.Location);
        var bytes = new byte[checked((int)lease.ByteLength)];
        await lease.ReadAsync(0, bytes);
        return bytes;
    }
}
