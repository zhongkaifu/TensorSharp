// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 == args.Length || args[i] is not ("--model" or "--json" or "--widths" or "--steps" or "--prompt-tokens" or "--backend" or "--expect-unsupported"))
        throw new ArgumentException("Use --model path --json path --widths 1,2,4,8,16 --steps 8 --prompt-tokens 64 --backend ggml_cuda|ggml_cpu --expect-unsupported true|false.");
    options.Add(args[i], args[i + 1]);
}
string modelPath = Path.GetFullPath(options["--model"]);
string output = Path.GetFullPath(options.GetValueOrDefault("--json", "artifacts/unified-memory/model.json"));
int[] widths = options.GetValueOrDefault("--widths", "1,2,4,8,16").Split(',').Select(int.Parse).ToArray();
int steps = int.Parse(options.GetValueOrDefault("--steps", "8"));
int promptTokens = int.Parse(options.GetValueOrDefault("--prompt-tokens", "64"));
bool expectUnsupported = bool.Parse(options.GetValueOrDefault("--expect-unsupported", "false"));
BackendType backend = options.GetValueOrDefault("--backend", "ggml_cuda") switch
{
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_cpu" => BackendType.GgmlCpu,
    _ => throw new ArgumentException("Unsupported backend."),
};
if (widths.Length == 0 || widths.Any(w => w < 1 || w > 16) || steps is < 2 or > 64 || promptTokens is < 32 or > 256)
    throw new ArgumentException("Use widths 1..16, steps 2..64 and prompt tokens 32..256.");
const int blockSize = 16;
const int transferBytes = 64 << 10;
string spillDirectory = Path.Combine(Path.GetDirectoryName(output)!, "model-spill-" + Guid.NewGuid().ToString("N"));
Environment.SetEnvironmentVariable("MAX_CONTEXT", "1024");
// Both arms use the same per-sequence model execution, chunk sizes and sample work.
// The candidate additionally proves that the configured storage actually spills.
Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "1");
Environment.SetEnvironmentVariable("KV_CACHE_DTYPE", "f16");
foreach (string name in new[] { "TS_SPEC", "SPECULATIVE_DECODING", "TS_MTP", "TS_SCHED_KV_RAM_BYTES", "TS_SCHED_KV_SSD_BYTES", "TS_SCHED_KV_SPILL_DIRECTORY" })
    Environment.SetEnvironmentVariable(name, null);
KvCacheDtypeConfig.ConfigureFromEnvironment();
var runs = new List<EngineRun>();
var failures = new List<string>();
object? snapshotReplay = null;
object? capabilities = null;
var promptFixtures = new List<PromptFixture>();
var nativeCacheSamples = new List<object>();
void SampleNativeCaches(string phase)
{
    var ranks = new List<GgmlCacheMemoryUsage>();
    for (int rank = 0; rank < 64 && GgmlBasicOps.TryGetCacheMemoryUsage(rank, out var usage); rank++)
        ranks.Add(usage);
    nativeCacheSamples.Add(new { Phase = phase, Available = ranks.Count > 0, Ranks = ranks });
}
string? error = null;
bool rejectedUnsupported = false;
long blockBytes = 0;
long ramBytes = 0;
ModelBase? loadedModel = null;
bool modelCleanupDeferred = false;
try
{
    var model = loadedModel = ModelBase.Create(modelPath, backend);
    SampleNativeCaches("loaded");
    blockBytes = model.ComputeKVBlockByteSize(blockSize);
    capabilities = new { model.SupportsKVStateSnapshot, model.SupportsCrossSequenceKvReuse,
        model.RequiresPerBlockCapture, model.MaxReusablePrefixTokens, model.KVStateFingerprint };
    ramBytes = checked(2 * Align(Math.Max(blockBytes, 1)) + Align(transferBytes));
    KvSnapshotOptions snapshotOptions = new(ramBytes, Math.Max(1L << 30, blockBytes * 1024), spillDirectory, transferBytes);
    if (expectUnsupported)
    {
        try { using var unused = new InferenceEngine(model, Config(2, snapshotOptions)); }
        catch (NotSupportedException) { rejectedUnsupported = true; }
        Require(rejectedUnsupported, "Explicit unsupported snapshot mode was not rejected at engine construction.");
    }
    else
    {
        Require(model.SupportsKVStateSnapshot && model.SupportsCrossSequenceKvReuse && blockBytes > 0,
            "This model does not declare safe cross-sequence host snapshots.");
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        for (int index = 0; index < Math.Max(2, widths.Max()); index++)
        {
            int start = 10 + index * 17;
            string instruction = $"This is counting exercise {index + 1}. Starting at {start}, write the next 100 consecutive integers in ascending order. " +
                "Separate numbers with spaces. Output only the numbers, without an introduction or explanation. Continue until all 100 numbers have been listed.";
            int[] tokens;
            while (true)
            {
                tokens = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate,
                    new List<ChatMessage> { new() { Role = "user", Content = instruction } },
                    model.Config.Architecture, addGenerationPrompt: true, enableThinking: false).ToArray();
                if (tokens.Length >= promptTokens) break;
                instruction += " Remember to include each consecutive number, without gaps, and continue in ascending order.";
            }
            Require(tokens.Length + steps <= Math.Min(1024, model.MaxReusablePrefixTokens),
                $"Rendered prompt {index} plus decode exceeds the model's restorable snapshot window.");
            promptFixtures.Add(new(index, instruction, tokens));
        }
        int[] pool = model.Tokenizer.Encode("100 101 102 103 104 105 106 107 108 109 110 111", addSpecial: false).ToArray();
        Require(pool.Length > 8, "Tokenizer produced too few fixture tokens.");
        int[] Prompt(int index) => promptFixtures[index].Tokens;
        // Warm native graph creation separately from the measured storage comparisons.
        model.ResetKVCache(); model.Forward(Prompt(0)); model.Forward(new[] { pool[0] }); model.ResetKVCache();
        foreach (int width in widths)
        {
            EngineRun baseline, tiered;
            if (runs.Count / 2 % 2 == 0)
            {
                baseline = await RunAndRecord(model, width, null, "managed", Prompt);
                tiered = await RunAndRecord(model, width, snapshotOptions, "tiered", Prompt);
            }
            else
            {
                tiered = await RunAndRecord(model, width, snapshotOptions, "tiered", Prompt);
                baseline = await RunAndRecord(model, width, null, "managed", Prompt);
            }
            if (baseline.Error != null || tiered.Error != null)
            {
                failures.Add($"width {width}: engine arm failed; inspect its Error and partial Requests.");
                continue;
            }
            for (int rank = 0; rank < width; rank++)
                if (!baseline.Requests[rank].Tokens.SequenceEqual(tiered.Requests[rank].Tokens)
                    || baseline.Requests[rank].Status != tiered.Requests[rank].Status
                    || baseline.Requests[rank].FinishReason != tiered.Requests[rank].FinishReason)
                    failures.Add($"width {width}, request {rank}: managed/tiered output mismatch.");
            // A solo request never swaps with prefix capture disabled. It is a
            // timing/output control, and cannot count as spill/restore coverage.
            if (width > 1 && tiered.Residency is not { Spills: > 0, Loads: > 0 })
                failures.Add($"width {width}: did not exercise SSD spill and restoration.");
            SampleNativeCaches($"width-{width}-completed");
        }

        // Teacher-forced replay checks EVERY vocabulary logit, beyond generated argmax.
        // Keep the complete chat framing and capture each prompt at its actual
        // endpoint; a recurrent snapshot must describe that same endpoint.
        int[][] prefixes = Enumerable.Range(0, 2).Select(Prompt).ToArray();
        int[] forced = Enumerable.Range(0, steps).Select(i => pool[i % pool.Length]).ToArray();
        var reference = new float[2][][];
        for (int rank = 0; rank < 2; rank++)
        {
            model.ResetKVCache(); model.Forward(prefixes[rank]);
            reference[rank] = forced.Select(token => (float[])model.Forward(new[] { token }).Clone()).ToArray();
        }
        long[] replaySizes = prefixes.Select(p => model.ComputeKVBlockByteSize(p.Length)).ToArray();
        long replayBytes = replaySizes.Max();
        var replayOptions = new KvSnapshotOptions(2 * Align(replayBytes) + Align(transferBytes),
            Math.Max(1L << 30, replayBytes * 8), spillDirectory, transferBytes);
        using var storage = new PagedKvStorage(2, replayBytes, replayOptions);
        for (int rank = 0; rank < 2; rank++)
        {
            model.ResetKVCache(); model.Forward(prefixes[rank]);
            using var lease = storage.Acquire(rank, ResourceAccess.Write);
            Require(model.TryExtractKVBlock(0, prefixes[rank].Length, lease.Span[..checked((int)replaySizes[rank])]),
                "Model refused snapshot capture.");
        }
        long compared = 0; double maxError = 0; int argmaxDifferences = 0; long outsideTolerance = 0;
        for (int rank = 0; rank < 2; rank++)
        {
            model.ResetKVCache();
            using (var lease = storage.Acquire(rank))
                Require(model.TryInjectKVBlock(0, prefixes[rank].Length, lease.ReadOnlySpan[..checked((int)replaySizes[rank])]),
                    "Model refused snapshot restoration.");
            for (int step = 0; step < forced.Length; step++)
            {
                float[] actual = model.Forward(new[] { forced[step] });
                float[] expected = reference[rank][step];
                Require(actual.Length == expected.Length, "Vocabulary size changed.");
                int maxA = 0, maxB = 0;
                for (int j = 0; j < actual.Length; j++)
                {
                    Require(float.IsFinite(actual[j]) && float.IsFinite(expected[j]), "Non-finite logit.");
                    double delta = Math.Abs((double)actual[j] - expected[j]);
                    maxError = Math.Max(maxError, delta);
                    if (delta > 1e-4 + 1e-4 * Math.Abs(expected[j])) outsideTolerance++;
                    if (actual[j] > actual[maxA]) maxA = j;
                    if (expected[j] > expected[maxB]) maxB = j;
                    compared++;
                }
                if (maxA != maxB) argmaxDifferences++;
            }
        }
        snapshotReplay = new { Prefixes = prefixes, ForcedTokens = forced, ComparedLogits = compared,
            MaxAbsoluteError = maxError, OutsideTolerance = outsideTolerance, ArgmaxDifferences = argmaxDifferences,
            Tolerance = "1e-4 + 1e-4 * abs(reference)", Residency = storage.ResidencyStats };
        Require(storage.ResidencyStats is { Spills: > 0, Loads: >= 4 }, "Teacher-forced replay did not restore both snapshots from SSD.");
        Require(outsideTolerance == 0 && argmaxDifferences == 0, "Snapshot replay logit parity failed.");
        SampleNativeCaches("snapshot-replay-completed");
    }
}
catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(error); }
if (!modelCleanupDeferred)
{
    try { loadedModel?.Dispose(); }
    catch (Exception ex) { error = (error == null ? "" : error + "\n") + ex; }
}
bool passed = error == null && failures.Count == 0;
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)); }
var native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
    .Select(m => new { m.FileName, Sha256 = Hash(m.FileName) }).ToArray();
await File.WriteAllTextAsync(output, JsonSerializer.Serialize(new
{
    Passed = passed, Error = error, Failures = failures, Model = modelPath,
    ModelBytes = new FileInfo(modelPath).Length, ModelSha256 = Hash(modelPath), Backend = backend.ToString(),
    Device = Environment.GetEnvironmentVariable("TS_VALIDATION_DEVICE"),
    GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
    Native = native, RuntimeSha256 = Hash(typeof(InferenceEngine).Assembly.Location),
    ModelsSha256 = Hash(typeof(ModelBase).Assembly.Location), ProbeSha256 = Hash(typeof(EngineRun).Assembly.Location),
    Capabilities = capabilities, BlockBytes = blockBytes, SnapshotRamBytes = ramBytes,
    PromptFixtureKind = "Complete model chat template with non-thinking assistant generation suffix; distinct counting requests; padded within user content to the requested minimum token count.",
    MinimumPromptTokens = promptTokens, PromptFixtures = promptFixtures,
    ExpectedUnsupported = expectUnsupported, RejectedUnsupported = rejectedUnsupported,
    ModelCleanupDeferred = modelCleanupDeferred,
    Runs = runs, SnapshotReplay = snapshotReplay,
    NativeCacheSamples = nativeCacheSamples,
    NativeCacheScope = "GGML-reported lazy-copy and explicit-preload payload per rank; excludes graph arenas, device KV, backend pools and physical driver allocation overhead. Availability is false with an older native binary.",
    Limitations = "Real model host snapshot RAM/SSD validation on the declared Backend. Model weights and live device KV remain resident and outside snapshot budget. Width 1 is an output/timing control only: no ownership swap or host snapshot occurs. Rendered chat fixtures test isolation and exact managed/tiered generated tokens, not semantic quality. Full-logit tolerance replay uses two complete short chat prompts and forced tokens. No whole-model out-of-core, multimodal, MTP, tensor-parallel snapshot, HTTP latency or speedup claim. Timings include engine I/O/compute, exclude load/compilation, and have one sample per width/arm. OS page cache is not bounded; physical filesystem medium is not assumed to be NVMe."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Model snapshot validation passed={passed}; report={output}");
GC.KeepAlive(loadedModel);
return passed ? 0 : 1;

static long Align(long bytes) => checked((bytes + 63) / 64 * 64);
static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
SchedulerConfig Config(int width, KvSnapshotOptions? snapshot) => new()
{
    BlockSize = blockSize, NumBlocks = 512, MaxNumRunningSequences = width,
    MaxNumBatchedTokens = blockSize * width, MaxPrefillChunkSize = blockSize, SoloPrefillChunkSize = blockSize,
    DecodeQuantumTokens = 1, EnablePrefixCaching = false, StopRepetition = false, KvSnapshots = snapshot,
};
async Task<EngineRun> RunEngine(ModelBase model, int width, KvSnapshotOptions? snapshot, string mode, Func<int, int[]> prompt)
{
    model.ResetKVCache();
    var gate = new ComputeGate(); gate.Close();
    using var engine = new InferenceEngine(model, Config(width, snapshot)) { ComputeGate = gate };
    var sequences = Enumerable.Range(0, width).Select(rank => new SequenceState($"{mode}-{width}-{rank}",
        prompt(rank), steps, blockSize, SamplingConfig.Greedy, cacheScope: $"isolation-{rank}")).ToArray();
    DateTime? released = null;
    var clock = new Stopwatch();
    string? runError = null;
    try
    {
        InferenceRequestHandle[] handles;
        lock (model.GpuComputeLock)
            handles = sequences.Select(s => engine.SubmitRequest(s)).ToArray();
        var parked = Stopwatch.StartNew();
        while (engine.StepsHeldByGate == 0 && parked.Elapsed < TimeSpan.FromSeconds(10)) await Task.Delay(1);
        Require(engine.StepsHeldByGate > 0, "Requests did not park before simultaneous release.");
        released = DateTime.UtcNow;
        clock.Start(); gate.Open();
        var completions = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromMinutes(15));
        foreach (var (c, rank) in completions.Select((c, rank) => (c, rank)))
        {
            Require(c.Status == SequenceStatus.FinishedLengthCapped || (c.Status == SequenceStatus.FinishedStopped && c.FinishReason == "eos"), $"Request failed: {c.Status}/{c.FinishReason}");
            Require(sequences[rank].OutputTokens.Any(t => !model.Tokenizer.IsEos(t)), "Empty/EOS-only output.");
            Require(c.Status == SequenceStatus.FinishedLengthCapped && sequences[rank].OutputTokens.Count == steps,
                $"Counting fixture stopped before the requested {steps} decode tokens: {c.Status}/{c.FinishReason}.");
        }
    }
    catch (Exception ex) { runError = ex.ToString(); }
    clock.Stop();
    var residency = engine.SnapshotResidencyStats;
    var budget = engine.SnapshotMemoryUsage;
    // Join before collecting partial output; a failed arm must not leave a
    // worker mutating sequences or GPU state when the next arm starts.
    engine.Dispose();
    var afterDispose = engine.SnapshotMemoryUsage;
    if (budget?.Any(p => p.Reserved + p.Committed > p.Capacity) == true
        || afterDispose?.Any(p => p.Reserved != 0 || p.Committed != 0) == true)
        runError = (runError == null ? "" : runError + "\n") + "Snapshot budget exceeded or charges leaked at disposal.";
    var rows = sequences.Select(s => new RequestRun(s.PromptTokens.ToArray(), s.OutputTokens.ToArray(),
        s.Status.ToString(), s.FinishReason, s.FirstTokenAt is { } first && released is { } start
            ? (first - start).TotalMilliseconds : null, s.Error?.ToString())).ToArray();
    var result = new EngineRun(mode, width, clock.Elapsed.TotalMilliseconds, rows, residency, budget, afterDispose, runError);
    Console.WriteLine($"{mode}: width={width} elapsed_ms={result.Milliseconds:F1} spills={result.Residency?.Spills}");
    return result;
}
async Task<EngineRun> RunAndRecord(ModelBase model, int width, KvSnapshotOptions? snapshot, string mode, Func<int, int[]> prompt)
{
    EngineRun result;
    try { result = await RunEngine(model, width, snapshot, mode, prompt); }
    catch (Exception ex)
    {
        // Work errors are captured inside RunEngine after successful cleanup.
        // An escaping error may be failed disposal: preserve evidence and stop
        // before another arm can reuse a model whose old work/state still lives.
        modelCleanupDeferred = true;
        runs.Add(new EngineRun(mode, width, 0, Array.Empty<RequestRun>(), null, null, null, ex.ToString()));
        throw;
    }
    runs.Add(result);
    return result;
}
record RequestRun(int[] Prompt, int[] Tokens, string Status, string? FinishReason, double? TtftMilliseconds, string? Error);
record EngineRun(string Mode, int Width, double Milliseconds, RequestRun[] Requests,
    MemorySchedulerStats? Residency, IReadOnlyList<MemoryPoolSnapshot>? Budget,
    IReadOnlyList<MemoryPoolSnapshot>? BudgetAfterDispose, string? Error);
record PromptFixture(int Index, string UserText, int[] Tokens);
