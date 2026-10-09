// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 == args.Length || args[i] is not ("--model" or "--json" or "--steps" or "--prompt-tokens" or "--tile-bytes" or "--token-rows" or "--host-bytes" or "--device-bytes" or "--cycles" or "--prefill" or "--prefill-chunk" or "--read-ahead" or "--host-cache-bytes"))
        throw new ArgumentException("Use --model PATH --json PATH --steps 4 --prompt-tokens 32 --tile-bytes 1048576 --token-rows 32 --host-bytes 2097152 --device-bytes 2097152 [--prefill forward|refill] [--prefill-chunk TOKENS] [--read-ahead true|false] [--host-cache-bytes BYTES].");
    options.Add(args[i], args[i + 1]);
}
string modelPath = Path.GetFullPath(options["--model"]);
string reportPath = Path.GetFullPath(options.GetValueOrDefault("--json", "artifacts/unified-memory/weight-model.json"));
int steps = int.Parse(options.GetValueOrDefault("--steps", "4"));
int cycles = int.Parse(options.GetValueOrDefault("--cycles", "2"));
int promptMinimum = int.Parse(options.GetValueOrDefault("--prompt-tokens", "32"));
int tileBytes = int.Parse(options.GetValueOrDefault("--tile-bytes", "1048576"));
int tokenRows = int.Parse(options.GetValueOrDefault("--token-rows", "32"));
long hostBytes = long.Parse(options.GetValueOrDefault("--host-bytes", "2097152"));
long deviceBytes = long.Parse(options.GetValueOrDefault("--device-bytes", "2097152"));
bool readAhead = bool.Parse(options.GetValueOrDefault("--read-ahead", "true"));
string prefillEntry = options.GetValueOrDefault("--prefill", "forward");
if (steps is < 2 or > 32 || promptMinimum is < 1 or > 896 || cycles is < 1 or > 8
    || prefillEntry is not ("forward" or "refill"))
    throw new ArgumentException("Use 2..32 steps, 1..896 minimum prompt tokens, 1..8 cycles, and forward/refill prefill.");
if (options.TryGetValue("--prefill-chunk", out var prefillChunk))
{
    if (!int.TryParse(prefillChunk, out int parsedChunk) || parsedChunk < 1)
        throw new ArgumentException("--prefill-chunk must be positive.");
    Environment.SetEnvironmentVariable("TS_PREFILL_CHUNK", prefillChunk);
}
foreach (string name in new[] { "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TS_SPEC", "SPECULATIVE_DECODING", "TS_MTP" })
    Environment.SetEnvironmentVariable(name, null);
Environment.SetEnvironmentVariable("MAX_CONTEXT", "1024");
Environment.SetEnvironmentVariable("KV_CACHE_DTYPE", "f16");
KvCacheDtypeConfig.ConfigureFromEnvironment();

var budget = new MemoryBudget(new[] { new MemoryCharge("weights/host", hostBytes), new MemoryCharge("weights/gpu0", deviceBytes) });
var streamingOptions = new WeightStreamingOptions(budget, "weights/host", new[] { "weights/gpu0" }, tileBytes, tokenRows, readAhead: readAhead)
{ HostCacheBytes = long.Parse(options.GetValueOrDefault("--host-cache-bytes", "0")) };
var cases = new List<Case>();
var references = new List<float[][]>();
var metrics = new List<object>();
var snapshots = new List<object>();
var nativeSamples = new List<object>();
var lifecycleSamples = new List<object>();
var cycleResults = new List<object>();
var forwardTimings = new List<object>();
var baselineForwardTimings = new List<object>();
var forwardCallTimings = new List<ForwardCallTiming>();
var processMemorySamples = new List<object>();
long totalFileBytesRead = 0, totalLinearTiles = 0;
double totalForwardMilliseconds = 0;
double baselinePrefillMilliseconds = 0, baselineDecodeMilliseconds = 0;
long peakObservedWorkingSetBytes = 0, processLifetimePeakWorkingSetBytes = 0;
string? processMemoryObservationError = null;
WeightStreamingStatistics? usage = null;
long residentParameterBytes = 0;
long expectedFileBytes = 0;
string architecture = "";
bool pressureRefused = false;
int forwardPressureRefusals = 0;
int resetRequiredRefusals = 0;
bool mappedFileCheckAvailable = OperatingSystem.IsLinux();
var mappings = new List<string>();
string? error = null;
ModelBase? model = null;
var timer = Stopwatch.StartNew();
double baselineMilliseconds = 0, streamedMilliseconds = 0;
try
{
    using (var metadata = new GgufFile(modelPath))
    {
        architecture = metadata.GetString("general.architecture") ?? throw new InvalidDataException("Model architecture is missing.");
        Require(architecture is "qwen35" or "gemma4", "This probe requires a supported Qwen35 or Gemma4 checkpoint.");
        expectedFileBytes = metadata.Tensors.Values.Where(t => (t.Type is GgmlTensorType.Q8_0 or GgmlTensorType.F16) && t.Shape.Length == 2)
            .Sum(metadata.GetTensorByteCount);
    }
    Require(expectedFileBytes > hostBytes && expectedFileBytes > deviceBytes,
        "The fixture's quantized weights must exceed both configured staging budgets.");

    model = ModelBase.Create(modelPath, BackendType.GgmlCuda);
    SampleProcessMemory("resident-loaded");
    Require(model.Config.Architecture == architecture && model.StreamingWeightUsage == null, "Resident reference unexpectedly streams weights or changed architecture.");
    SampleNative("resident-loaded", requireNoPreload: false);
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    for (int index = 0; index < 2; index++)
    {
        string text = $"Count upwards from {10 + index * 17} to {50 + index * 17}, separated by spaces. Output only the numbers.";
        int[] prompt;
        do
        {
            prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate,
                new List<ChatMessage> { new() { Role = "user", Content = text } },
                model.Config.Architecture, addGenerationPrompt: true, enableThinking: false).ToArray();
            if (prompt.Length < promptMinimum) text += " Include every consecutive integer without gaps.";
        } while (prompt.Length < promptMinimum);
        Require(prompt.Length + steps < 1024, "Prompt and continuation exceed the bounded test window.");
        model.ResetKVCache();
        long forwardStarted = Stopwatch.GetTimestamp();
        float[] logits = Prefill(model, prompt);
        double prefillMilliseconds = Stopwatch.GetElapsedTime(forwardStarted).TotalMilliseconds;
        forwardCallTimings.Add(new("resident", null, index, 0, "prefill", prefillMilliseconds));
        double decodeMilliseconds = 0;
        SampleProcessMemory($"resident-case-{index}-prefill");
        var rows = new float[steps][];
        int[] generated = new int[steps];
        for (int step = 0; step < steps; step++)
        {
            Require(logits.Length == model.Config.VocabSize && logits.All(float.IsFinite), "Invalid resident reference logits.");
            rows[step] = (float[])logits.Clone();
            generated[step] = Top(logits);
            Require(!model.Tokenizer.IsEos(generated[step]), "Resident counting fixture stopped early; EOS-only output is not validation.");
            if (step + 1 < steps)
            {
                int[] decodeInput = new[] { generated[step] };
                forwardStarted = Stopwatch.GetTimestamp();
                logits = model.Forward(decodeInput);
                double elapsed = Stopwatch.GetElapsedTime(forwardStarted).TotalMilliseconds;
                decodeMilliseconds += elapsed;
                forwardCallTimings.Add(new("resident", null, index, step + 1, "decode", elapsed));
                SampleProcessMemory($"resident-case-{index}-decode-{step + 1}");
            }
        }
        baselinePrefillMilliseconds += prefillMilliseconds;
        baselineDecodeMilliseconds += decodeMilliseconds;
        baselineForwardTimings.Add(new { Case = index, PromptTokens = prompt.Length,
            DecodeCalls = steps - 1, PrefillMilliseconds = prefillMilliseconds,
            DecodeMilliseconds = decodeMilliseconds, ForwardMilliseconds = prefillMilliseconds + decodeMilliseconds });
        cases.Add(new(index, text, prompt, generated));
        references.Add(rows);
    }
    baselineMilliseconds = timer.Elapsed.TotalMilliseconds;
    model.Dispose(); model = null;

    // Another owner consumes the same physical host pool before construction.
    // This must reject the staging allocation, not silently mmap/preload instead.
    using (budget.Reserve(new[] { new MemoryCharge("weights/host", hostBytes) }))
    {
        try { model = ModelBase.Create(modelPath, BackendType.GgmlCuda, weightStreaming: streamingOptions); }
        catch (MemoryPressureException) { pressureRefused = true; }
        Require(pressureRefused && model == null, "An exhausted shared host pool did not reject streaming construction.");
    }
    Require(budget.Snapshot().All(p => p.Reserved == 0 && p.Committed == 0), "Failed construction retained staging charges.");

    timer.Restart();
    for (int cycle = 0; cycle < cycles; cycle++)
    {
        bool forwardPressureRefused = false;
        bool retryRequiredReset = false;
        double cyclePrefillMilliseconds = 0, cycleDecodeMilliseconds = 0;
        usage = null;
        model = ModelBase.Create(modelPath, BackendType.GgmlCuda, weightStreaming: streamingOptions);
        Require(model.Config.Architecture == architecture, "Streaming model adapter changed architecture.");
        residentParameterBytes = model switch
        {
            Qwen35Model qwen => qwen.StreamingResidentParameterBytes,
            Gemma4Model gemma => gemma.StreamingResidentParameterBytes,
            _ => throw new NotSupportedException("Missing resident-parameter accounting for this adapter."),
        };
        Require(model.StreamingWeightUsage?.FileBackedWeightBytes == expectedFileBytes, "Some quantized weights were not represented as file regions.");
        Sample("streamed-loaded");
        SampleNative("streamed-loaded", requireNoPreload: true);
        CheckMappings();
        // A mid-forward allocation refusal can follow writes to activation or state
        // buffers. Releasing quota is insufficient: the model must demand a reset.
        using (budget.Reserve(new[] { new MemoryCharge("weights/gpu0", deviceBytes) }))
        {
            try { Prefill(model, cases[0].PromptTokens); }
            catch (MemoryPressureException) { forwardPressureRefused = true; }
            Require(forwardPressureRefused, "An exhausted shared GPU pool did not refuse streamed forward.");
        }
        forwardPressureRefusals++;
        try { model.Forward(new[] { cases[0].GeneratedTokens[0] }); }
        catch (InvalidOperationException ex) when (ex.Message.Contains("ResetKVCache", StringComparison.Ordinal))
        { retryRequiredReset = true; }
        Require(retryRequiredReset, "A failed streamed forward was retried without an explicit state reset.");
        resetRequiredRefusals++;
        model.ResetKVCache();
        Sample("forward-pressure-released-and-state-reset");
        foreach (var item in cases)
        {
            model.ResetKVCache();
            long forwardStarted = Stopwatch.GetTimestamp();
            float[] logits = Prefill(model, item.PromptTokens);
            double prefillMilliseconds = Stopwatch.GetElapsedTime(forwardStarted).TotalMilliseconds;
            forwardCallTimings.Add(new("streamed", cycle, item.Index, 0, "prefill", prefillMilliseconds));
            double decodeMilliseconds = 0;
            SampleProcessMemory($"streamed-cycle-{cycle}-case-{item.Index}-prefill");
            for (int step = 0; step < steps; step++)
            {
                float[] expected = references[item.Index][step];
                Require(logits.Length == expected.Length && logits.All(float.IsFinite), "Invalid streamed logits.");
                double squaredError = 0, normReference = 0, normActual = 0, dot = 0, maxAbsolute = 0;
                for (int i = 0; i < logits.Length; i++)
                {
                    double actual = logits[i], reference = expected[i], delta = actual - reference;
                    squaredError += delta * delta; normReference += reference * reference;
                    normActual += actual * actual; dot += actual * reference;
                    maxAbsolute = Math.Max(maxAbsolute, Math.Abs(delta));
                }
                double relativeL2 = Math.Sqrt(squaredError / Math.Max(normReference, 1e-300));
                double cosine = dot / Math.Sqrt(Math.Max(normReference * normActual, 1e-300));
                int top = Top(logits);
                bool passed = relativeL2 <= 0.001 && cosine >= 0.999999 && top == item.GeneratedTokens[step];
                metrics.Add(new { Cycle = cycle, Case = item.Index, Step = step, Logits = logits.Length, Passed = passed,
                    RelativeL2 = relativeL2, Cosine = cosine, MaxAbsoluteError = maxAbsolute,
                    ReferenceTop = item.GeneratedTokens[step], ActualTop = top });
                Require(passed, $"Streamed logits differ at case {item.Index}, step {step}: relL2={relativeL2}, cosine={cosine}, top={top}/{item.GeneratedTokens[step]}.");
                Sample($"case-{item.Index}-step-{step}");
                if (step + 1 < steps)
                {
                    int[] decodeInput = new[] { item.GeneratedTokens[step] };
                    forwardStarted = Stopwatch.GetTimestamp();
                    logits = model.Forward(decodeInput);
                    double elapsed = Stopwatch.GetElapsedTime(forwardStarted).TotalMilliseconds;
                    decodeMilliseconds += elapsed;
                    forwardCallTimings.Add(new("streamed", cycle, item.Index, step + 1, "decode", elapsed));
                    SampleProcessMemory($"streamed-cycle-{cycle}-case-{item.Index}-decode-{step + 1}");
                }
            }
            cyclePrefillMilliseconds += prefillMilliseconds;
            cycleDecodeMilliseconds += decodeMilliseconds;
            forwardTimings.Add(new { Cycle = cycle, Case = item.Index, PromptTokens = item.PromptTokens.Length,
                DecodeCalls = steps - 1, PrefillMilliseconds = prefillMilliseconds,
                DecodeMilliseconds = decodeMilliseconds, ForwardMilliseconds = prefillMilliseconds + decodeMilliseconds });
            SampleNative($"streamed-case-{item.Index}-completed", requireNoPreload: true);
            CheckMappings();
            Console.WriteLine($"PASS streamed cycle {cycle}, case {item.Index}: {steps} complete vocabulary rows, {item.PromptTokens.Length} prompt tokens");
        }
        usage = model.StreamingWeightUsage;
        Require(usage is { FileBytesRead: > 0, LinearTiles: > 0, EmbeddingRows: > 0 }, "The streaming operator was not exercised.");
        Require(checked(usage.Value.FileBytesRead + usage.Value.HostCacheHitBytes) > expectedFileBytes && usage.Value.LinearTiles > 100,
            "The test did not repeatedly consume bounded weight tiles across real model layers.");
        Require(usage.Value.PeakHostCacheBytes <= streamingOptions.HostCacheBytes,
            "Optional weight reuse exceeded its cache ceiling.");
        Require(usage.Value.PeakHostStagingBytes <= hostBytes && usage.Value.PeakDeviceWorkspaceBytes <= deviceBytes,
            "Streaming workspace exceeded its shared budget.");
        SampleLifecycle($"cycle-{cycle}-before-dispose", requireLiveGdn: architecture == "qwen35", requireEmpty: false);
        model.Dispose(); model = null;
        var releasedBudget = budget.Snapshot();
        Require(releasedBudget.All(p => p.Reserved == 0 && p.Committed == 0), "A completed cycle retained streaming allocations.");
        SampleLifecycle($"cycle-{cycle}-after-dispose", requireLiveGdn: false, requireEmpty: true);
        cycleResults.Add(new { Cycle = cycle, ForwardPressureRefused = forwardPressureRefused,
            RetryRequiredReset = retryRequiredReset, PrefillMilliseconds = cyclePrefillMilliseconds,
            DecodeMilliseconds = cycleDecodeMilliseconds, ForwardMilliseconds = cyclePrefillMilliseconds + cycleDecodeMilliseconds,
            Usage = usage.Value, BudgetAfterDispose = releasedBudget });
        totalFileBytesRead += usage.Value.FileBytesRead;
        totalLinearTiles += usage.Value.LinearTiles;
        totalForwardMilliseconds += cyclePrefillMilliseconds + cycleDecodeMilliseconds;
    }
    streamedMilliseconds = timer.Elapsed.TotalMilliseconds;
}
catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(error); }
finally
{
    bool released = model == null;
    if (model != null)
    {
        usage ??= model.StreamingWeightUsage;
        try { model.Dispose(); released = true; }
        catch (Exception ex) { error = (error == null ? "" : error + "\n") + "Cleanup: " + ex; }
    }
    // This standalone process owns the backend. Release global native CUDA
    // caches before driver teardown rather than relying on C++ static ordering.
    if (released)
        try { GgmlBasicOps.Shutdown(); }
        catch (Exception ex) { error = (error == null ? "" : error + "\n") + "Backend shutdown: " + ex; }
}
var finalBudget = budget.Snapshot();
if (finalBudget.Any(p => p.Reserved != 0 || p.Committed != 0))
    error = (error == null ? "" : error + "\n") + "Model disposal retained streaming allocations.";
Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(file)); }
var native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
    .Select(m => new { m.FileName, Sha256 = Hash(m.FileName) }).ToArray();
File.WriteAllText(reportPath, JsonSerializer.Serialize(new {
    Passed = error == null, Error = error, Architecture = architecture, Model = modelPath,
    ModelFileBytes = new FileInfo(modelPath).Length, ModelSha256 = Hash(modelPath),
    Device = Environment.GetEnvironmentVariable("TS_VALIDATION_DEVICE"),
    GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"), Native = native,
    ModelsSha256 = Hash(typeof(ModelBase).Assembly.Location), ProbeSha256 = Hash(typeof(Case).Assembly.Location),
    BackendSha256 = Hash(typeof(GgmlBasicOps).Assembly.Location), MemorySha256 = Hash(typeof(MemoryBudget).Assembly.Location),
    Gate = new { MaxRelativeL2 = 0.001, MinCosine = 0.999999, RequireIdenticalTop1 = true },
    Steps = steps, Cycles = cycles, CompletedCycles = cycleResults.Count, MinimumPromptTokens = promptMinimum, TileBytes = tileBytes, TokenRows = tokenRows,
    HostBytes = hostBytes, DeviceBytes = deviceBytes, ReadAhead = readAhead, ExpectedFileBytes = expectedFileBytes,
    HostCacheCeilingBytes = streamingOptions.HostCacheBytes, HostCacheReserveBytes = streamingOptions.HostCacheReserveBytes,
    ResidentParameterBytes = residentParameterBytes, Cases = cases, Metrics = metrics, Usage = usage,
    CycleResults = cycleResults, TotalFileBytesRead = totalFileBytesRead, TotalLinearTiles = totalLinearTiles,
    ForwardTimings = forwardTimings, TotalForwardMilliseconds = totalForwardMilliseconds,
    BaselineForwardTimings = baselineForwardTimings, BaselinePrefillMilliseconds = baselinePrefillMilliseconds,
    BaselineDecodeMilliseconds = baselineDecodeMilliseconds,
    BaselineForwardMilliseconds = baselinePrefillMilliseconds + baselineDecodeMilliseconds,
    ForwardCallTimings = forwardCallTimings,
    ForwardCallTimingScope = "Records each returned validation forward immediately, including calls whose subsequent logits comparison fails; excludes pressure/recovery calls. Completed-case aggregates retain their original success-only meaning.",
    ForwardTimingScope = "Wall time inside successful case prefill (selected Forward/ForwardRefill entry) and decode ModelBase.Forward calls only, with the same scope for resident and streamed arms; excludes model load, pressure tests, reset, logits comparisons, snapshots and disposal. Same-host configuration comparison only, not a standardized throughput benchmark.",
    Execution = new { PrefillEntry = prefillEntry == "refill" ? "ModelBase.ForwardRefill" : "ModelBase.Forward",
        DecodeEntry = "ModelBase.Forward", MaxContext = 1024, KvCacheDtype = "f16",
        PrefillChunkEnvironment = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK"),
        PrefillChunkApplies = prefillEntry == "refill",
        TensorParallelDegree = 1, LayerSplitDegree = 1,
        CudaDisableFusion = Environment.GetEnvironmentVariable("GGML_CUDA_DISABLE_FUSION"),
        CudaMmqPrecision = Environment.GetEnvironmentVariable("GGML_CUDA_MMQ_PREC"),
        CudaCublasComputeType = Environment.GetEnvironmentVariable("GGML_CUDA_CUBLAS_COMPUTE_TYPE"),
        WeightFusionCopies = Environment.GetEnvironmentVariable("TS_WEIGHT_FUSION_COPIES"),
        GemmaDiagnosticDirectory = Environment.GetEnvironmentVariable("TS_GEMMA4_TENSOR_DUMP"),
        GemmaDiagnosticLayers = Environment.GetEnvironmentVariable("TS_GEMMA4_TENSOR_DUMP_LAYERS"),
        Note = "Both arms use the same public prefill entry. A null refill chunk override delegates to each model adapter's default; no fused/per-op graph override is applied by the probe. Weight TokenRows limits each linear's staging, independently of model refill chunks." },
    ProcessMemorySamples = processMemorySamples, PeakObservedWorkingSetBytes = peakObservedWorkingSetBytes,
    ProcessLifetimePeakWorkingSetBytes = processLifetimePeakWorkingSetBytes,
    ProcessMemoryObservationError = processMemoryObservationError,
    ProcessMemoryScope = "Observational process working set sampled after loads/forwards and at budget samples; the lifetime OS peak includes both resident and streamed arms. Neither is a GPU-memory measurement or a budget assertion, and sampled peaks may miss transients.",
    NativeLifecycleSamples = lifecycleSamples,
    NativeLifecycleScope = "GDN graph cache counters exercise populated allocations only for Qwen35. Gemma4 does not use these caches; no populated Gemma graph-cache lifecycle coverage is inferred from their zero values.",
    PressureConstructionRefused = pressureRefused, ForwardPressureRefused = forwardPressureRefusals == cycles,
    RetryRequiredReset = resetRequiredRefusals == cycles, ForwardPressureRefusals = forwardPressureRefusals,
    ResetRequiredRefusals = resetRequiredRefusals, Snapshots = snapshots, FinalBudget = finalBudget,
    NativeSamples = nativeSamples, MappedFileCheckAvailable = mappedFileCheckAvailable, UnexpectedModelMappings = mappings,
    BaselineMilliseconds = baselineMilliseconds, StreamedMilliseconds = streamedMilliseconds,
    Scope = "Real dense single-rank GGML CUDA inference using original Q8_0/F16 file regions and bounded synchronous weight/output-row tiles. Budget covers file-weight host staging, optional aligned host-cache payload and CUDA streamed input/weight/output workspace. Small F32 constants, activations, KV/GDN state, other native caches, backend pools, CUDA runtime, managed cache index and OS file cache are excluded. FileBytesRead counts logical source reads, not physical SSD traffic. Resident fused and streamed per-op execution are compared with the existing ForcedLogitProbe numerical gates. Overall timings include load/validation and are not throughput benchmarks. No multi-GPU streaming or end-to-end process memory cap is claimed."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Weight model validation passed={error == null}; report={reportPath}");
return error == null ? 0 : 1;

void Sample(string phase)
{
    SampleProcessMemory(phase);
    var pools = budget.Snapshot();
    Require(pools.All(p => p.Reserved >= 0 && p.Committed >= 0 && p.Available >= 0), "Budget exceeded its configured limit.");
    snapshots.Add(new { Phase = phase, Pools = pools, Usage = model!.StreamingWeightUsage });
}
float[] Prefill(ModelBase currentModel, int[] tokens)
    => prefillEntry == "refill" ? currentModel.ForwardRefill(tokens) : currentModel.Forward(tokens);
void SampleProcessMemory(string phase)
{
    try
    {
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        long working = process.WorkingSet64, lifetimePeak = process.PeakWorkingSet64;
        peakObservedWorkingSetBytes = Math.Max(peakObservedWorkingSetBytes, working);
        processLifetimePeakWorkingSetBytes = Math.Max(processLifetimePeakWorkingSetBytes, lifetimePeak);
        processMemorySamples.Add(new { Phase = phase, WorkingSetBytes = working, LifetimePeakWorkingSetBytes = lifetimePeak });
    }
    catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or NotSupportedException or InvalidOperationException)
    { processMemoryObservationError ??= ex.Message; }
}
void SampleNative(string phase, bool requireNoPreload)
{
    Require(GgmlBasicOps.TryGetCacheMemoryUsage(0, out var sample), "Native cache telemetry unavailable.");
    nativeSamples.Add(new { Phase = phase, Usage = sample });
    if (requireNoPreload) Require(sample.PreloadCommittedBytes == 0 && sample.PreloadReservedBytes == 0,
        "File-backed inference populated the full-weight preload cache.");
}
void CheckMappings()
{
    if (!mappedFileCheckAvailable) return;
    var found = File.ReadLines("/proc/self/maps").Where(line => line.Contains(modelPath, StringComparison.Ordinal)).ToArray();
    mappings.AddRange(found);
    Require(found.Length == 0, "File-backed inference created a whole-model file mapping.");
}
void SampleLifecycle(string phase, bool requireLiveGdn, bool requireEmpty)
{
    try
    {
        long gdn = NativeLifecycle.GdnChunkedBytes(), recurrentPrefill = NativeLifecycle.RecurrentPrefillBytes();
        lifecycleSamples.Add(new { Phase = phase, Available = true, GdnChunkedBytes = gdn, RecurrentPrefillBytes = recurrentPrefill });
        if (requireLiveGdn) Require(gdn > 0, "The real model did not populate the GDN graph cache targeted by the cleanup regression.");
        if (requireEmpty) Require(gdn == 0 && recurrentPrefill == 0, "Model disposal retained GDN/recurrent-prefill native graph allocations.");
    }
    catch (EntryPointNotFoundException)
    {
        lifecycleSamples.Add(new { Phase = phase, Available = false });
    }
}
static int Top(float[] logits)
{
    int top = 0;
    for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[top]) top = i;
    return top;
}
static void Require([System.Diagnostics.CodeAnalysis.DoesNotReturnIf(false)] bool condition, string message)
{ if (!condition) throw new InvalidOperationException(message); }
sealed record Case(int Index, string Prompt, int[] PromptTokens, int[] GeneratedTokens);
sealed record ForwardCallTiming(string Arm, int? Cycle, int Case, int Step, string Phase, double Milliseconds);
static class NativeLifecycle
{
    [DllImport("GgmlOps", EntryPoint = "TSGgml_TestGdnChunkedCacheBytes", CallingConvention = CallingConvention.Cdecl)]
    public static extern long GdnChunkedBytes();
    [DllImport("GgmlOps", EntryPoint = "TSGgml_TestQwen35RecurrentPrefillCacheBytes", CallingConvention = CallingConvention.Cdecl)]
    public static extern long RecurrentPrefillBytes();
}
