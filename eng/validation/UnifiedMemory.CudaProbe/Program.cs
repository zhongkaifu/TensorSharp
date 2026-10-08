// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Memory;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 >= args.Length || args[i] is not ("--devices" or "--json" or "--peer"))
        throw new ArgumentException("Usage: --devices 0[,1] --json path --peer true|false");
    options.Add(args[i], args[i + 1]);
}
string output = Path.GetFullPath(options.GetValueOrDefault("--json", "artifacts/unified-memory/cuda.json"));
bool peer = bool.Parse(options.GetValueOrDefault("--peer", "false"));
int[] deviceIds = options.GetValueOrDefault("--devices", "0").Split(',')
    .Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
if (deviceIds.Length == 0 || deviceIds.Any(i => i < 0) || deviceIds.Distinct().Count() != deviceIds.Length)
    throw new ArgumentException("Select distinct non-negative CUDA device ordinals.");
if (peer && deviceIds.Length < 2) throw new ArgumentException("Peer validation requires at least two devices.");
var contexts = new List<CudaContext>();
var deviceInfo = new List<object>();
var results = new List<object>();
var routes = new List<object>();
string? error = null;
int exit = 0;
bool initialized = false;
string directory = Path.Combine(Path.GetDirectoryName(output)!, "spill-" + Guid.NewGuid().ToString("N"));
try
{
    // Create() installs TensorSharp's driver resolver. Missing devices/driver are
    // an unavailable validation environment, never a skipped passing test.
    foreach (int id in deviceIds)
    {
        var context = CudaContext.Create(id);
        contexts.Add(context);
        var name = new byte[256];
        CudaDriverApi.cuDeviceGet(out int device, id).ThrowOnError();
        CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError();
        CudaDriverApi.cuMemGetInfo(out var free, out var total).ThrowOnError();
        deviceInfo.Add(new { id, name = Encoding.UTF8.GetString(name).TrimEnd('\0'), free = (ulong)free, total = (ulong)total });
    }
    initialized = true;
    const int page = 1 << 20;
    const int granularity = 2 << 20;
    const int staging = 64 << 10;
    var budget = new MemoryBudget(new[] { new MemoryCharge("ram", page + staging * 2), new MemoryCharge("ssd", page * 32L) }
        .Concat(deviceIds.Select(id => new MemoryCharge($"gpu/{id}", granularity))));
    var host = new HostMemoryBackend("ram");
    var devices = contexts.Select(context => new CudaResidencyBackend(context, $"gpu/{context.DeviceId}",
        allocationGranularity: granularity, enablePeerCopies: peer)).ToArray();
    using (var transfers = new BoundedTransfers(budget, "ram", staging, 2))
    using (var spill = new SsdSpillStore(budget, "ssd", directory, transfers))
    await using (var scheduler = new TieredMemoryScheduler(budget, new IMemoryBackend[] { host }.Concat(devices), transfers, spill, host.Location))
    {
        ResourceKey Key(string name) => new("cuda-probe", 1, name);
        var state = Key("state");
        scheduler.Register(new(state, page, ResourceKind.KvPage, Mutable: true));
        byte[] expected = new byte[page];
        for (int i = 0; i < page / sizeof(uint); i++) BitConverter.TryWriteBytes(expected.AsSpan(i * sizeof(uint)), (uint)(i * 17));

        async Task CheckCase(string name, Func<Task> run)
        {
            var watch = Stopwatch.StartNew();
            try
            {
                await run();
                results.Add(new { name, status = "passed", milliseconds = watch.Elapsed.TotalMilliseconds });
                Console.WriteLine($"PASS {name}");
            }
            catch (Exception ex)
            {
                results.Add(new { name, status = "failed", error = ex.ToString(), milliseconds = watch.Elapsed.TotalMilliseconds });
                throw;
            }
        }
        async Task Verify(ResourceKey key, MemoryLocation location, byte[] reference)
        {
            using var lease = await scheduler.AcquireAsync(key, location);
            byte[] actual = new byte[page];
            await lease.ReadAsync(0, actual);
            Checks.Bytes(location, reference, actual);
        }

        if (devices.Length > 1)
        {
            // Check every directed route independently before pressure tests. A bad
            // pair must not prevent the other direction from producing evidence.
            int failedPairs = 0;
            for (int src = 0; src < devices.Length; src++)
            for (int dst = 0; dst < devices.Length; dst++)
            {
                if (src == dst) continue;
                int from = src, to = dst;
                try
                {
                    await CheckCase($"directed GPU {deviceIds[from]} -> {deviceIds[to]} full-buffer parity", async () =>
                    {
                        using (var write = await scheduler.AcquireAsync(state, devices[from].Location, ResourceAccess.Write))
                            await write.WriteAsync(0, expected);
                        long copies = devices[to].PeerCopyCount;
                        using var read = await scheduler.AcquireAsync(state, devices[to].Location);
                        bool usedPeer = devices[to].PeerCopyCount > copies;
                        routes.Add(new { from = deviceIds[from], to = deviceIds[to], route = usedPeer ? "peer" : "bounded-host-staging" });
                        if (peer && !usedPeer) throw new NotSupportedException("Explicit peer validation requested but this directed pair used host staging.");
                        byte[] actual = new byte[page];
                        await read.ReadAsync(0, actual);
                        Checks.Bytes(devices[to].Location, expected, actual);
                    });
                }
                catch (Exception ex)
                {
                    failedPairs++;
                    Console.Error.WriteLine($"FAIL directed GPU {deviceIds[from]} -> {deviceIds[to]}: {ex.Message}");
                }
            }
            if (failedPairs != 0) throw new InvalidDataException($"{failedPairs} directed CUDA pair(s) failed; do not enable peer copies on this deployment.");
        }

        for (int rank = 0; rank < devices.Length; rank++)
        {
            int current = rank;
            await CheckCase($"GPU {deviceIds[current]} same-device copy preserves unaligned tail", () =>
                Checks.DeviceCopy(devices[current], page + 17, staging));
            await CheckCase($"GPU {deviceIds[current]} kernel mutation and event-backed lease retirement", async () =>
            {
                var lease = await scheduler.AcquireAsync(state, devices[current].Location, ResourceAccess.Write);
                await lease.WriteAsync(0, expected);
                contexts[current].MakeCurrent();
                using (var module = CudaModule.LoadFromBytes(Encoding.UTF8.GetBytes(Kernel.Ptx)))
                {
                    try
                    {
                        Kernel.Launch(module.GetFunction("add_seven"), lease.Pointer, page / sizeof(uint));
                        // On a failed fence leave the lease pinned; scheduler teardown
                        // must refuse to free memory that a failed GPU may still use.
                        await lease.ReleaseAfterAsync(CudaExecutionFence.Record(contexts[current]));
                    }
                    finally { contexts[current].MakeCurrent(); }
                }
                for (int i = 0; i < page / sizeof(uint); i++) BitConverter.TryWriteBytes(expected.AsSpan(i * sizeof(uint)), BitConverter.ToUInt32(expected.AsSpan(i * sizeof(uint))) + 7);
                await Verify(state, devices[current].Location, expected);
            });

        }

        await CheckCase("VRAM and RAM pressure forces lossless SSD restore", async () =>
        {
            for (int i = 0; i < 5; i++)
            {
                var key = Key($"pressure-{i}");
                scheduler.Register(new(key, page, ResourceKind.KvPage, Mutable: true));
                using var lease = await scheduler.AcquireAsync(key, devices[0].Location, ResourceAccess.Write);
                await lease.WriteAsync(0, expected);
            }
            await Verify(state, host.Location, expected);
            await Verify(state, devices[0].Location, expected);
            if (scheduler.GetStats().Spills == 0) throw new Exception("SSD pressure was not exercised.");
        });

        await CheckCase("concurrent GPU readers share completed residency", async () =>
        {
            await Task.WhenAll(Enumerable.Range(0, 16).Select(_ => Task.Run(() => Verify(state, devices[0].Location, expected))));
            if (scheduler.GetStats().ActiveLeases != 0) throw new Exception("Reader lease leaked.");
        });

        if (devices.Length > 1)
        {
            await CheckCase("multi-GPU pressure releases partial working-set pins", async () =>
            {
                using var held = await scheduler.AcquireAsync(state, devices[0].Location);
                var pending = Key("partial-set");
                scheduler.Register(new(pending, page, ResourceKind.KvPage, Mutable: true));
                bool refused = false;
                try
                {
                    using var invalid = await scheduler.AcquireReadSetAsync(new[] {
                        new ResourcePlacement(pending, devices[1].Location),
                        new ResourcePlacement(pending, devices[0].Location) });
                }
                catch (MemoryPressureException) { refused = true; }
                if (!refused || scheduler.GetStats().ActiveLeases != 1)
                    throw new Exception("Pressure did not roll back every partial pin.");
            });
            await CheckCase("multi-GPU working-set pins and all-rank completion fence", async () =>
            {
                using var set = await scheduler.AcquireReadSetAsync(devices.Select(d => new ResourcePlacement(state, d.Location)));
                if (set.Leases.Count != devices.Length) throw new Exception("Missing device lease.");
                await set.ReleaseAfterAsync(Task.WhenAll(contexts.Select(context => CudaExecutionFence.Record(context))));
            });
        }
        await CheckCase("request envelopes retain all GPU charges until physical cleanup", () =>
            Checks.RequestEnvelopes(devices, contexts, directory, page, staging));
        if (budget.Snapshot().Any(p => p.Reserved + p.Committed > p.Capacity)) throw new Exception("Budget exceeded.");
        Console.WriteLine(JsonSerializer.Serialize(new { residency = scheduler.GetStats(), budget = budget.Snapshot() }));
    }
    if (budget.Snapshot().Any(p => p.Reserved != 0 || p.Committed != 0)) throw new Exception("Budget leaked on disposal.");
}
catch (Exception ex)
{
    error = ex.ToString();
    exit = initialized ? 1 : 2;
    Console.Error.WriteLine(error);
}
finally
{
    for (int i = contexts.Count - 1; i >= 0; i--) contexts[i].Dispose();
    Directory.CreateDirectory(Path.GetDirectoryName(output)!);
    await File.WriteAllTextAsync(output, JsonSerializer.Serialize(new
    {
        status = exit == 0 ? "passed" : initialized ? "failed" : "hardware unavailable",
        devices = deviceInfo, tests = results, routes, error,
        multiGpu = initialized && deviceIds.Length > 1 ? "requested" : "not run",
        peerRequested = peer,
        note = "Real CUDA allocations/kernel/events; no production model inference. Payload budgets exclude test reference arrays, driver/context/module overhead and OS page cache. Storage medium is not asserted to be NVMe.",
    }, new JsonSerializerOptions { WriteIndented = true }));
    if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
}
return exit;

static class Checks
{
    public static void Bytes(MemoryLocation location, byte[] expected, byte[] actual)
    {
        if (actual.AsSpan().SequenceEqual(expected)) return;
        int offset = 0;
        while (offset < Math.Min(expected.Length, actual.Length) && expected[offset] == actual[offset]) offset++;
        throw new InvalidDataException($"Byte mismatch at {location}; first offset {offset}; expected SHA256 {Convert.ToHexString(SHA256.HashData(expected))}; actual SHA256 {Convert.ToHexString(SHA256.HashData(actual))}");
    }

    public static async Task DeviceCopy(CudaResidencyBackend device, int length, int staging)
    {
        var charges = device.GetAllocationCharges(length);
        var budget = new MemoryBudget(charges.Select(c => c with { Bytes = checked(c.Bytes * 2) })
            .Append(new MemoryCharge("copy-ram", staging)));
        using (var transfers = new BoundedTransfers(budget, "copy-ram", staging, 1))
        using (var sourceCharge = budget.Reserve(charges))
        using (var source = await device.AllocateAsync(length))
        using (var targetCharge = budget.Reserve(charges))
        using (var target = await device.AllocateAsync(length))
        {
            sourceCharge.Commit(); targetCharge.Commit();
            byte[] expected = Enumerable.Range(0, length).Select(i => (byte)(i * 31 + 7)).ToArray();
            byte[] actual = new byte[length];
            await target.ReadAsync(0, actual);
            if (actual.Any(x => x != 0)) throw new InvalidDataException("New CUDA allocation was not zero-initialized.");
            await source.WriteAsync(0, expected);
            long before = device.DeviceCopyCount;
            await transfers.CopyAsync(source, target);
            if (device.DeviceCopyCount != before + 1) throw new Exception("Same-device direct copy did not execute.");
            await target.ReadAsync(0, actual);
            Bytes(device.Location, expected, actual);
            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            bool rejected = false;
            try { await target.WriteAsync(0, new byte[] { 0 }, cancelled.Token); }
            catch (OperationCanceledException) { rejected = true; }
            if (!rejected) throw new Exception("Cancelled CUDA write was not rejected.");
            await target.ReadAsync(0, actual);
            Bytes(device.Location, expected, actual);
        }
        if (budget.Snapshot().Any(p => p.Reserved != 0 || p.Committed != 0)) throw new Exception("Direct-copy allocation budget leaked.");
    }

    public static async Task RequestEnvelopes(CudaResidencyBackend[] devices, List<CudaContext> contexts,
        string directory, int page, int staging)
    {
        var freeBefore = contexts.Select(context =>
        {
            context.MakeCurrent();
            CudaDriverApi.cuMemGetInfo(out var free, out _).ThrowOnError();
            return free;
        }).ToArray();
        var peaks = devices.SelectMany(d => d.GetAllocationCharges(page)).ToArray();
        var budget = new MemoryBudget(peaks.Concat(new[] { new MemoryCharge("request-ram", staging), new MemoryCharge("request-ssd", 0) }));
        using (var transfers = new BoundedTransfers(budget, "request-ram", staging, 1))
        using (var spill = new SsdSpillStore(budget, "request-ssd", Path.Combine(directory, "requests"), transfers))
        await using (var scheduler = new TieredMemoryScheduler(budget, devices, transfers, spill))
        {
            var queue = new MemoryRequestQueue(budget, 2);
            queue.Enqueue("first", peaks); queue.Enqueue("next", peaks);
            using var first = queue.TryAdmit() ?? throw new Exception("First request was not admitted.");
            if (queue.TryAdmit() != null) throw new Exception("All-rank peak admitted twice.");
            var keys = devices.Select(d => new ResourceKey("cuda-request", 1, d.Location.Device)).ToArray();
            for (int i = 0; i < devices.Length; i++)
            {
                scheduler.Register(new(keys[i], page, ResourceKind.KvPage, Mutable: true));
                var lease = await scheduler.AcquireAsync(keys[i], devices[i].Location,
                    ResourceAccess.Write, allocationEnvelope: first.Envelope);
                await lease.WriteAsync(page - 1, new byte[] { 79 });
                await lease.ReleaseAfterAsync(CudaExecutionFence.Record(contexts[i]));
            }
            first.Dispose();
            if (queue.TryAdmit() != null) throw new Exception("Closed request forgot live GPU allocations.");
            for (int i = 0; i < devices.Length; i++)
            {
                using (var read = await scheduler.AcquireAsync(keys[i], devices[i].Location))
                {
                    var tail = new byte[1];
                    await read.ReadAsync(page - 1, tail);
                    if (tail[0] != 79) throw new InvalidDataException("Closing the request lost GPU state.");
                }
                scheduler.Unregister(keys[i]);
                if (i != devices.Length - 1 && queue.TryAdmit() != null)
                    throw new Exception("Request admitted before all ranks physically released memory.");
            }
            using var next = queue.TryAdmit() ?? throw new Exception("Physical cleanup did not release the next request.");
        }
        if (budget.Snapshot().Any(p => p.Reserved != 0 || p.Committed != 0)) throw new Exception("Request budget leaked.");
        for (int i = 0; i < contexts.Count; i++)
        {
            contexts[i].MakeCurrent();
            CudaDriverApi.cuMemGetInfo(out var free, out _).ThrowOnError();
            if (free < freeBefore[i]) throw new Exception($"GPU {contexts[i].DeviceId} retained physical memory after request cleanup: {freeBefore[i] - free} bytes.");
        }
    }
}

static class Kernel
{
    public const string Ptx = """
        .version 6.0
        .target sm_50
        .address_size 64
        .visible .entry add_seven(.param .u64 data, .param .u32 count) {
            .reg .pred %p;
            .reg .b32 %r<6>;
            .reg .b64 %rd<3>;
            ld.param.u64 %rd0, [data];
            ld.param.u32 %r0, [count];
            mov.u32 %r1, %ctaid.x;
            mov.u32 %r2, %ntid.x;
            mov.u32 %r3, %tid.x;
            mad.lo.u32 %r1, %r1, %r2, %r3;
            setp.ge.u32 %p, %r1, %r0;
            @%p bra DONE;
            mul.wide.u32 %rd1, %r1, 4;
            add.u64 %rd2, %rd0, %rd1;
            ld.global.u32 %r4, [%rd2];
            add.u32 %r5, %r4, 7;
            st.global.u32 [%rd2], %r5;
        DONE: ret;
        }
        """;
    public static unsafe void Launch(nint function, nint pointer, int count)
    {
        void** parameters = stackalloc void*[2];
        parameters[0] = &pointer; parameters[1] = &count;
        CudaDriverApi.cuLaunchKernel(function, (uint)((count + 255) / 256), 1, 1, 256, 1, 1, 0,
            IntPtr.Zero, (nint)parameters, IntPtr.Zero).ThrowOnError();
    }
}
