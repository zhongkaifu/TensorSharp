// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

var o = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 == args.Length || args[i] is not ("--model" or "--json" or "--arm" or "--reference" or "--context" or "--prompt" or "--steps" or "--width" or "--max-running" or "--chunk" or "--host-bytes" or "--device-bytes" or "--ssd-bytes"))
        throw new ArgumentException("Use --model path --json path --arm reference|candidate --reference reference.json --context 32768 --prompt 30000 --steps 16 --width 2 --chunk 256 --host-bytes N --device-bytes N --ssd-bytes N.");
    o.Add(args[i], args[i + 1]);
}
string path = Path.GetFullPath(o["--model"]), output = Path.GetFullPath(o["--json"]);
string arm = o["--arm"];
int context = int.Parse(o.GetValueOrDefault("--context", "32768")), promptLength = int.Parse(o.GetValueOrDefault("--prompt", "30000")),
    steps = int.Parse(o.GetValueOrDefault("--steps", "16")), width = int.Parse(o.GetValueOrDefault("--width", "2")),
    chunk = int.Parse(o.GetValueOrDefault("--chunk", "256")),
    maxRunning = int.Parse(o.GetValueOrDefault("--max-running", width.ToString()));
if (arm is not ("reference" or "candidate") || context < 128 || context > 131072 || promptLength < 32 || promptLength + steps + 64 > context
    || steps < 2 || steps > 128 || width < 1 || width > 8 || maxRunning < 1 || maxRunning > width || chunk < 1 || chunk > Math.Min(context, 2048))
    throw new ArgumentException("Invalid context/request shape.");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
foreach (var (key, value) in new[] { ("MAX_CONTEXT", context.ToString()), ("KV_CACHE_DTYPE", "f16"),
    ("TS_SCHED_DISABLE_BATCHED", "1"), ("TS_SPEC", "0"), ("TS_MTP", "0") }) Environment.SetEnvironmentVariable(key, value);
KvCacheDtypeConfig.ConfigureFromEnvironment();
const int blockTokens = 256;
var samples = new List<object>();
var requests = new List<object>();
var peaks = new List<object>();
var runningSamples = new List<object>();
MemoryBudget? budget = null;
AdaptiveModelSession? session = null;
ModelBase? model = null;
object? residency = null, afterDispose = null, swapTimings = null;
string? error = null;
bool? parity = null;
double wall = 0, loadSeconds = 0;
bool mpsClient = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_MPS_PIPE_DIRECTORY"));
// Under an MPS allocation limit, CUDA may report board total but client-limited
// free bytes. Their difference is NOT physical board occupancy.
long? DeviceUsed() => !mpsClient && model != null && GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) ? total - free : null;
object? DeviceAvailability() => model != null && GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total)
    ? new { ReportedFreeBytes = free, ReportedTotalBytes = total, MpsClient = mpsClient } : null;
void Sample(string phase)
{
    using var process = Process.GetCurrentProcess();
    samples.Add(new { Phase = phase, UnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
        WorkingSetBytes = process.WorkingSet64, PeakWorkingSetBytes = process.PeakWorkingSet64,
        CudaDeviceUsedBytes = DeviceUsed(), CudaAvailability = DeviceAvailability(),
        ManagedBytes = GC.GetTotalMemory(false), Budget = budget?.Snapshot(),
        Physical = PhysicalMemory.Capture(path), NativeAllocation = session?.NativeAllocationUsage,
        NativeRefusals = session == null ? null : (object)new { session.NativeAllocationRefusals.Count,
            session.NativeAllocationRefusals.First, session.NativeAllocationRefusals.Last },
        HostAllocation = session == null ? null : (object)new { session.HostAllocationUsage.Bytes, session.HostAllocationUsage.PeakBytes, session.HostAllocationUsage.Allocations } });
    Console.WriteLine($"PHASE {phase} {DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}");
}
string Hash(string p) { using var file = File.OpenRead(p); return Convert.ToHexString(SHA256.HashData(file)); }
string modelHash = Hash(path); // Before load/measurement; no baseline in the candidate process.
try
{
    Sample("before-load");
    var load = Stopwatch.StartNew();
    if (arm == "candidate")
    {
        budget = new([new(AdaptiveModelSession.HostPool, long.Parse(o.GetValueOrDefault("--host-bytes", (4L << 30).ToString()))),
            new(AdaptiveModelSession.DevicePool, long.Parse(o.GetValueOrDefault("--device-bytes", (4L << 30).ToString()))),
            new("probe/ssd", long.Parse(o.GetValueOrDefault("--ssd-bytes", (16L << 30).ToString())))]);
        session = AdaptiveModelSession.Create(path, new(context, chunk), budget);
        model = session.Model;
    }
    else model = ModelBase.Create(path, BackendType.GgmlCuda, 1, null!, null!, 1, null!, new ModelMemoryPolicy(context, chunk));
    loadSeconds = load.Elapsed.TotalSeconds;
    Sample("loaded");
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    int[] Prompt(int rank)
    {
        int[] Render(int lines)
        {
            var padding = new StringBuilder();
            for (int line = 0; line < lines; line++)
                padding.Append("The archive contains ordinary notes about rivers, forests, and mountains. Keep the notes separate from the final counting instruction.\n");
            string text = "Read these notes as background only.\n" + padding +
                $"\nNow perform counting exercise {rank + 1}. Starting at {10 + rank * 17}, write the next 100 consecutive integers separated by spaces. Output only the numbers, without an introduction. Continue until all 100 numbers are listed.";
            return renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate,
                new List<ChatMessage> { new() { Role = "user", Content = text } }, model.Config.Architecture,
                addGenerationPrompt: true, enableThinking: false).ToArray();
        }
        int low = 0, high = 1;
        int[] rendered = Render(0);
        if (rendered.Length < promptLength)
        {
            while (Render(high).Length < promptLength) { low = high; high = checked(high * 2); }
            while (high - low > 1)
            {
                int middle = low + (high - low) / 2;
                if (Render(middle).Length < promptLength) low = middle; else high = middle;
            }
            rendered = Render(high);
        }
        if (rendered.Length + steps > context
            || (maxRunning > 1 && rendered.Length + steps > model.MaxReusablePrefixTokens))
            throw new NotSupportedException("Request exceeds the model's admitted or exactly restorable context.");
        return rendered;
    }
    var sequences = Enumerable.Range(0, width).Select(i => new SequenceState($"long-{i}", Prompt(i), steps,
        blockTokens, SamplingConfig.Greedy, cacheScope: $"scope-{i}")).ToArray();
    // Prefill uses the same graph shape in both processes; no unmeasured long warmup.
    KvSnapshotOptions? snapshots = budget == null ? null : KvSnapshotOptions.FromSharedBudget(budget,
        AdaptiveModelSession.HostPool, "probe/ssd", Path.Combine(Path.GetDirectoryName(output)!, "spill-" + Guid.NewGuid().ToString("N")), 64 << 10);
    var admission = snapshots == null ? null : session!.CreateRequestMemoryAdmission(snapshots, blockTokens, maxRunning, chunk);
    foreach (var sequence in sequences)
        peaks.Add(new { sequence.RequestId, PromptTokens = sequence.PromptTokens.Count,
            ExecutionPeak = session?.EstimateRequestPeak(sequence.PromptTokens.Count, steps, chunk),
            TotalPeak = admission?.EstimatePeak(sequence) });
    var config = new SchedulerConfig { BlockSize = blockTokens,
        NumBlocks = checked(width * ((context + blockTokens - 1) / blockTokens) + 16),
        MaxNumRunningSequences = maxRunning, MaxNumBatchedTokens = checked(width * chunk), PrefillChunkTokenLimit = chunk,
        MaxPrefillChunkSize = chunk, SoloPrefillChunkSize = chunk, DecodeQuantumTokens = 256,
        EnablePrefixCaching = false, StopRepetition = false, KvSnapshots = snapshots, MemoryAdmission = admission };
    // Preserve admission evidence even if a native failure prevents final cleanup.
    await File.WriteAllTextAsync(output + ".planning.json", JsonSerializer.Serialize(new {
        Arm = arm, Context = context, Chunk = chunk, MaximumRunningRequests = maxRunning,
        Plan = session?.Plan, RequestPeaks = peaks, Budget = budget?.Snapshot()
    }, new JsonSerializerOptions { WriteIndented = true }));
    using (var engine = new InferenceEngine(model, config))
    {
        Sample("before-requests");
        var watch = Stopwatch.StartNew();
        var handles = sequences.Select(s => engine.SubmitRequest(s)).ToArray();
        var pending = Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromMinutes(20));
        while (!pending.IsCompleted)
        {
            using var process = Process.GetCurrentProcess();
            runningSamples.Add(new { UnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                Running = engine.RunningCount, Waiting = engine.WaitingCount, Budget = budget?.Snapshot(),
                WorkingSetBytes = process.WorkingSet64, CudaDeviceUsedBytes = DeviceUsed(),
                HostStatusBytes = PhysicalMemory.LinuxStatus(), ManagedLiveBytes = GC.GetTotalMemory(false),
                NativeAllocation = session?.NativeAllocationUsage });
            await Task.WhenAny(pending, Task.Delay(100));
        }
        var results = await pending;
        wall = watch.Elapsed.TotalSeconds;
        Sample("requests-completed");
        for (int i = 0; i < results.Length; i++)
        {
            var r = results[i]; var s = sequences[i];
            double pp = r.PrefillElapsedTicks / (double)Stopwatch.Frequency, tg = r.DecodeElapsedTicks / (double)Stopwatch.Frequency;
            requests.Add(new { s.RequestId, PromptTokens = s.PromptTokens.Count,
                PromptHash = Convert.ToHexString(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(s.PromptTokens))),
                Tokens = s.OutputTokens.ToArray(), Text = model.Tokenizer.Decode(s.OutputTokens.ToList()),
                Status = r.Status.ToString(), r.FinishReason,
                PrefillSeconds = pp, DecodeSeconds = tg,
                PrefillTokensPerSecond = pp > 0 ? s.PromptTokens.Count / pp : 0,
                DecodeTokensPerSecond = tg > 0 ? Math.Max(0, s.OutputTokens.Count - 1) / tg : 0,
                TtftSeconds = (r.FirstTokenAt - r.SubmittedAt)?.TotalSeconds });
            if (r.Status != SequenceStatus.FinishedLengthCapped || s.OutputTokens.Count != steps || !s.OutputTokens.Any(t => !model.Tokenizer.IsEos(t)))
                throw new InvalidOperationException($"Incomplete/empty request {i}: {r.Status}/{r.FinishReason}.");
        }
        residency = engine.SnapshotResidencyStats;
        swapTimings = engine.SnapshotSwapTimings;
    }
    Sample("engine-disposed");
    if (o.TryGetValue("--reference", out var referencePath))
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(referencePath));
        var root = reference.RootElement;
        if (!root.GetProperty("Passed").GetBoolean() || root.GetProperty("Arm").GetString() != "reference"
            || root.GetProperty("ModelSha256").GetString() != modelHash || root.GetProperty("ContextTokens").GetInt32() != context
            || root.GetProperty("ChunkTokens").GetInt32() != chunk
            || root.GetProperty("MaximumRunningRequests").GetInt32() != maxRunning
            || !root.GetProperty("PrefillChunkLimitEnforced").GetBoolean())
            throw new InvalidOperationException("Reference identity/shape does not match this candidate.");
        var expected = root.GetProperty("Requests").EnumerateArray().ToArray();
        using var actualDoc = JsonDocument.Parse(JsonSerializer.Serialize(requests));
        var actual = actualDoc.RootElement.EnumerateArray().ToArray();
        parity = expected.Length == actual.Length && expected.Zip(actual).All(pair =>
            pair.First.GetProperty("PromptHash").GetString() == pair.Second.GetProperty("PromptHash").GetString()
            && pair.First.GetProperty("Tokens").EnumerateArray().Select(t => t.GetInt32())
                .SequenceEqual(pair.Second.GetProperty("Tokens").EnumerateArray().Select(t => t.GetInt32())));
        if (parity != true) throw new InvalidOperationException("Candidate tokens differ from the isolated reference process.");
    }
    if (session?.AccountingError != null) throw session.AccountingError;
}
catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(ex); }
finally
{
    try
    {
        if (session != null) session.Dispose(); else model?.Dispose();
        afterDispose = budget?.Snapshot();
        if (budget != null && budget.Snapshot().Any(p => p.Reserved != 0 || p.Committed != 0))
            throw new InvalidOperationException("Model/engine cleanup retained allocation credit.");
        Sample("model-disposed");
    }
    catch (Exception ex) { error = (error == null ? "" : error + "\n") + ex; }
    afterDispose = budget?.Snapshot();
}
var native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
    .Select(m => new { m.FileName, Sha256 = Hash(m.FileName) }).ToArray();
await File.WriteAllTextAsync(output, JsonSerializer.Serialize(new { Passed = error == null, Error = error, Arm = arm,
    Model = path, ModelSha256 = modelHash, ContextTokens = context, MinimumPromptTokens = promptLength, Steps = steps, Width = width, ChunkTokens = chunk,
    MaximumRunningRequests = maxRunning, PrefillChunkLimitEnforced = true,
    BulkRestoreEnabled = Environment.GetEnvironmentVariable("TS_DISABLE_BULK_KV_RESTORE") != "1",
    LoadSeconds = loadSeconds, WallSeconds = wall, TokenParity = parity, Requests = requests, RequestPeaks = peaks, Samples = samples, SwapTimings = swapTimings,
    Residency = residency, RunningSamples = runningSamples, HighWatermarks = budget?.HighWatermarks(), AfterDispose = afterDispose,
    Native = native, GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
    Managed = new[] { typeof(ModelBase).Assembly.Location, typeof(InferenceEngine).Assembly.Location, typeof(MemoryBudget).Assembly.Location,
        typeof(GgmlCacheBudgetScope).Assembly.Location }.Select(p => new { Path = p, Sha256 = Hash(p) }),
    Limitations = "One fresh process per arm; cold graph creation included, OS file cache uncontrolled. Prefill/decode rates use forward compute durations; group wall/TTFT include scheduling and snapshot I/O. Width is submitted concurrency, not proof of simultaneous admission. Quotas cover instrumented allocations, not RSS, driver heaps or file-cache residency. No semantic quality or llama.cpp comparison claim. External cgroup evidence is required for a physical hard-limit claim." }, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"long-context passed={error == null} parity={parity}; {output}");
return error == null ? 0 : 1;
