// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Memory;

int ranks = args.Length >= 1 ? int.Parse(args[0]) : 2;
string report = Path.GetFullPath(args.Length >= 2 ? args[1] : "artifacts/unified-memory/ggml-budget.json");
if (ranks is < 1 or > 2) throw new ArgumentException("Use [1|2] [report.json].");
var observations = new List<object>();
string? error = null;
bool unavailable = false;
GgmlCacheBudgetScope? scope = null;
MemoryBudget? budget = null;
long allocationBytes = 0;
int exactMatvecs = 0;
int rollbackCases = 0;
using var first = new NativeFloats(Enumerable.Range(0, 128 * 128).Select(i => (i / 128 % 4 + 1) / 8f).ToArray());
using var second = new NativeFloats(Enumerable.Range(0, 128 * 128).Select(i => (i / 128 % 4 + 5) / 8f).ToArray());
using var input = new NativeFloats(Enumerable.Repeat(1f, 128).ToArray());
using var output = new NativeFloats(new float[128]);
string[][] mapping = Enumerable.Range(0, ranks).Select(r => new[] { "shared-cache", $"gpu{r}" }).ToArray();
try
{
    if (Native.GetGpuDeviceCount(3) < ranks)
    {
        unavailable = true;
        throw new InvalidOperationException($"{ranks} CUDA devices are required; unavailable hardware is not a pass.");
    }
    Require(Native.MultiDeviceInit(3, Enumerable.Range(0, ranks).ToArray(), ranks) == 1, "CUDA backend initialization failed.");
    Select(0);
    GgmlBasicOps.SetDeviceCopyBudget(0);
    Run(first);
    allocationBytes = Usage(0).DeviceCopyCommittedBytes;
    Require(allocationBytes >= 65536, "Calibration did not exercise the CUDA device-copy cache.");
    budget = new MemoryBudget(new[] { new MemoryCharge("shared-cache", ranks * allocationBytes) }
        .Concat(Enumerable.Range(0, ranks).Select(r => new MemoryCharge($"gpu{r}", 2 * allocationBytes))));
    GgmlCacheBudgetScope? unexpected = null;
    try { unexpected = new GgmlCacheBudgetScope(budget, mapping); }
    catch (InvalidOperationException) { }
    if (unexpected != null)
    {
        GgmlBasicOps.ClearHostBufferCache(); unexpected.Dispose();
        throw new InvalidOperationException("Scope accepted an existing unowned cache allocation.");
    }
    GgmlBasicOps.ClearHostBufferCache();
    var initialWeak = CreateUnreferencedScope(budget, mapping);
    GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
    GC.WaitForPendingFinalizers();
    Require(initialWeak.TryGetTarget(out scope), "Native callback context did not root the installed scope.");
    GgmlCacheBudgetScope? duplicate = null;
    try { duplicate = new GgmlCacheBudgetScope(budget, mapping); }
    catch (InvalidOperationException) { }
    if (duplicate != null)
    {
        duplicate.Dispose();
        throw new InvalidOperationException("A second active scope replaced the first callback owner.");
    }
    Sample("attached-after-old-cache-cleared");

    // Another subsystem's reservation in the SAME budget must block both cache
    // paths before allocation. Lazy caching falls back to ordinary graph scratch.
    using (var external = budget.Reserve(new[] { new MemoryCharge("gpu0", 2 * allocationBytes) }))
    {
        Run(first);
        ExpectFailure(() => Preload(first), "Preload ignored an external reservation.");
        Require(scope.ActiveAllocations == 0, "Rejected allocations retained callback ownership.");
        Sample("external-reservation-refuses-cache-but-streaming-is-exact");
        var changed = budget.ChangeSignal;
        external.Dispose();
        Require(changed.IsCompleted, "Releasing shared credit did not signal admission waiters.");
    }
    Preload(first);
    Require(scope.ActiveAllocations == 1, "Released credit did not permit a real preload.");
    Sample("external-credit-released-preload-committed");
    ExpectFailure(scope.Dispose, "A live native allocation allowed scope disposal.");
    Require(scope.ActiveAllocations == 1, "Failed disposal lost native ownership.");

    // Remove all strong managed references. Native callback context must keep
    // the scope alive until successful detach, even through a compacting GC.
    var weak = new WeakReference<GgmlCacheBudgetScope>(scope);
    scope = null;
    GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
    GC.WaitForPendingFinalizers();
    Require(weak.TryGetTarget(out scope), "Native callback context did not root the live scope.");
    GgmlBasicOps.ClearHostBufferCache();
    Sample("gc-root-survives-and-clear-refunds-preload");

    // The shared pool, rather than either rank's larger local pool, is the
    // limiting constraint after one allocation on each rank.
    for (int rank = 0; rank < ranks; rank++) { Select(rank); Run(first); }
    Sample("all-ranks-share-one-common-capacity");
    Require(budget.Snapshot().Single(p => p.Pool == "shared-cache").Available == 0, "Shared capacity was not exercised.");
    for (int rank = 0; rank < ranks; rank++)
    {
        Select(rank); Run(second);
        ExpectFailure(() => Preload(second), "Preload exceeded the common pool capacity.");
    }
    Sample("common-pool-refuses-extra-copies-on-each-rank");
    GgmlBasicOps.ClearHostBufferCache();

    // Leave abundant common capacity but reduce each GPU constraint to one
    // allocation, proving that ranks do not borrow each other's local credit.
    Require(budget.TrySetCapacity("shared-cache", ranks * 4 * allocationBytes), "Cannot grow shared capacity.");
    for (int rank = 0; rank < ranks; rank++)
        Require(budget.TrySetCapacity($"gpu{rank}", allocationBytes), "Cannot set per-rank capacity.");
    for (int rank = 0; rank < ranks; rank++)
    {
        Select(rank); Preload(first); Run(first); Run(second);
        ExpectFailure(() => Preload(second), "A rank exceeded its own pool capacity.");
    }
    Sample("per-rank-cap-refuses-extra-copies-with-common-space-available");
    GgmlBasicOps.ClearHostBufferCache();
    Sample("all-live-allocations-cleared");

    // Native hooks stop an allocation after reservation, after physical
    // allocation, and after commit but before publication. Both cache families
    // must refund every charge and leave no entry; numerical fallback stays exact.
    Select(0);
    for (int kind = 0; kind < 2; kind++)
    for (int stage = 1; stage <= 3; stage++)
    {
        Native.TestCacheAllocationFailure(kind, stage);
        if (kind == 0) Run(first);
        else ExpectFailure(() => Preload(first), "Injected preload allocation failure was ignored.");
        Require(scope!.ActiveAllocations == 0, "An unpublished allocation retained shared credit.");
        Require(budget.Snapshot().All(p => p.Reserved == 0 && p.Committed == 0), "Rollback did not refund every pool.");
        Sample($"rollback-kind-{kind}-stage-{stage}");
        rollbackCases++;
    }
    Preload(second); Run(second);
    Sample("successful-allocation-after-all-rollback-cases");
    GgmlBasicOps.ClearHostBufferCache();
    Require(scope!.CallbackError == null, "Unexpected managed callback failure.");
    scope.Dispose(); scope = null;
    using (var reattached = new GgmlCacheBudgetScope(budget, mapping))
        Require(reattached.ActiveAllocations == 0, "Reattached scope was not empty.");
    observations.Add(new { Phase = "detached-and-reattached", Budget = budget.Snapshot() });
}
catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(error); }
finally
{
    try { GgmlBasicOps.ClearHostBufferCache(); scope?.Dispose(); Native.Shutdown(); }
    catch (Exception ex) { error = (error == null ? "" : error + "\nCleanup: ") + ex; }
}
Directory.CreateDirectory(Path.GetDirectoryName(report)!);
string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)); }
var native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
    .Select(m => new { m.FileName, Sha256 = Hash(m.FileName) }).ToArray();
await File.WriteAllTextAsync(report, JsonSerializer.Serialize(new
{
    Passed = error == null, Unavailable = unavailable, Error = error, Ranks = ranks,
    Backend = "ggml_cuda", MatrixBytes = 65536, AllocationBytes = allocationBytes,
    ExactMatvecs = exactMatvecs, RollbackCases = rollbackCases, Observations = observations,
    FinalBudget = budget?.Snapshot(), Native = native,
    ProbeSha256 = Hash(typeof(NativeFloats).Assembly.Location),
    GgmlBackendSha256 = Hash(typeof(GgmlCacheBudgetScope).Assembly.Location),
    MemorySha256 = Hash(typeof(MemoryBudget).Assembly.Location),
    GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
    Scope = "Real CUDA lazy-copy and explicit-preload payload allocations bridged to a shared managed MemoryBudget. shared-cache is a deliberate common capacity constraint, not host RAM physically consumed by this CUDA allocation. This probe maps one allocation to common plus per-rank constraints; do not sum those constraints as separate copies. Graph scratch, live KV outside native caches, backend pools and driver overhead are excluded. Two ranks perform independent exact matvecs; no collective or tensor-parallel model inference is claimed. Requires TensorSharp native BUILD_TESTS allocation-failure hooks."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"GGML shared-budget validation passed={error == null}; report={report}");
return unavailable ? 77 : error == null ? 0 : 1;

void Run(NativeFloats weights)
{
    Require(Native.AddmmQuantF32(new(output.Pointer, 1, 128, 128, 1, 128 * sizeof(float)),
        new(input.Pointer, 1, 128, 128, 1, 128 * sizeof(float)), weights.Pointer, 0, 128, 128, 65536) == 1,
        "CUDA cached/streamed matvec failed.");
    float[] actual = output.Read();
    for (int row = 0; row < 128; row++)
        Require(actual[row] == 128 * weights.Values[row * 128], "Cached/streamed/preloaded matvec differs from exact reference.");
    exactMatvecs++;
}
void Preload(NativeFloats weights) => Require(GgmlBasicOps.PreloadQuantizedWeight(
    weights.Pointer, weights.Pointer, 0, 128, 128, 65536), "Preload took an unsupported fallback.");
void Select(int rank) => Require(Native.SetActiveDevice(rank) == 1, $"Cannot select CUDA rank {rank}.");
GgmlCacheMemoryUsage Usage(int rank)
{
    Require(GgmlBasicOps.TryGetCacheMemoryUsage(rank, out var usage), "Cache telemetry is unavailable.");
    Require(usage.DeviceCopyReservedBytes == 0 && usage.PreloadReservedBytes == 0, "Completed operation retained an in-flight charge.");
    return usage;
}
void Sample(string phase)
{
    var nativeUsage = Enumerable.Range(0, ranks).Select(Usage).ToArray();
    var pools = budget!.Snapshot();
    long total = 0;
    for (int rank = 0; rank < ranks; rank++)
    {
        long cached = nativeUsage[rank].DeviceCopyCommittedBytes + nativeUsage[rank].PreloadCommittedBytes;
        Require(pools.Single(p => p.Pool == $"gpu{rank}").Committed == cached, "Managed and native rank charges differ.");
        total += cached;
    }
    Require(pools.Single(p => p.Pool == "shared-cache").Committed == total, "Managed common pool does not cover all native ranks.");
    Require(pools.All(p => p.Available >= 0), "Shared constraints were oversubscribed.");
    Require(scope?.CallbackError == null, "Unexpected managed callback failure.");
    observations.Add(new { Phase = phase, Budget = pools, Native = nativeUsage, ActiveAllocations = scope?.ActiveAllocations });
    Console.WriteLine($"PASS {phase}: cache_bytes={total} allocations={scope?.ActiveAllocations}");
}
static void Require([System.Diagnostics.CodeAnalysis.DoesNotReturnIf(false)] bool condition, string message)
{ if (!condition) throw new InvalidOperationException(message); }
[MethodImpl(MethodImplOptions.NoInlining)]
static WeakReference<GgmlCacheBudgetScope> CreateUnreferencedScope(MemoryBudget budget, string[][] mapping)
    => new(new GgmlCacheBudgetScope(budget, mapping));
static void ExpectFailure(Action action, string message)
{
    try { action(); }
    catch (InvalidOperationException) { return; }
    throw new InvalidOperationException(message);
}
sealed class NativeFloats : IDisposable
{
    public float[] Values { get; }
    public IntPtr Pointer { get; }
    public NativeFloats(float[] values) { Values = values; Pointer = Marshal.AllocHGlobal(values.Length * sizeof(float)); Marshal.Copy(values, 0, Pointer, values.Length); }
    public float[] Read() { var values = new float[Values.Length]; Marshal.Copy(Pointer, values, 0, values.Length); return values; }
    public void Dispose() => Marshal.FreeHGlobal(Pointer);
}
[StructLayout(LayoutKind.Sequential)]
readonly record struct TensorView(IntPtr Data, int Rows, int Columns, int RowStride, int ColumnStride, long Bytes);
static class Native
{
    private const string Library = "GgmlOps";
    [DllImport(Library, EntryPoint = "TSGgml_GetGpuDeviceCount", CallingConvention = CallingConvention.Cdecl)] public static extern int GetGpuDeviceCount(int backend);
    [DllImport(Library, EntryPoint = "TSGgml_MultiDeviceInit", CallingConvention = CallingConvention.Cdecl)] public static extern int MultiDeviceInit(int backend, int[] indices, int count);
    [DllImport(Library, EntryPoint = "TSGgml_SetActiveDevice", CallingConvention = CallingConvention.Cdecl)] public static extern int SetActiveDevice(int rank);
    [DllImport(Library, EntryPoint = "TSGgml_AddmmQuantF32", CallingConvention = CallingConvention.Cdecl)] public static extern int AddmmQuantF32(TensorView output, TensorView input, IntPtr weight, int type, long ne0, long ne1, long bytes);
    [DllImport(Library, EntryPoint = "TSGgml_TestCacheAllocationFailure", CallingConvention = CallingConvention.Cdecl)] public static extern void TestCacheAllocationFailure(int kind, int stage);
    [DllImport(Library, EntryPoint = "TSGgml_Shutdown", CallingConvention = CallingConvention.Cdecl)] public static extern void Shutdown();
}
