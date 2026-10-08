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
    if (i + 1 == args.Length || args[i] is not ("--model" or "--json" or "--steps" or "--prompt-tokens" or "--tile-bytes" or "--token-rows" or "--host-bytes" or "--device-bytes" or "--cycles"))
        throw new ArgumentException("Use --model PATH --json PATH --steps 4 --prompt-tokens 32 --tile-bytes 1048576 --token-rows 32 --host-bytes 2097152 --device-bytes 2097152.");
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
if (steps is < 2 or > 32 || promptMinimum is < 1 or > 256 || cycles is < 1 or > 8)
    throw new ArgumentException("Use 2..32 steps, 1..256 minimum prompt tokens, and 1..8 cycles.");
foreach (string name in new[] { "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TS_SPEC", "SPECULATIVE_DECODING", "TS_MTP" })
    Environment.SetEnvironmentVariable(name, null);
Environment.SetEnvironmentVariable("MAX_CONTEXT", "1024");
Environment.SetEnvironmentVariable("KV_CACHE_DTYPE", "f16");
KvCacheDtypeConfig.ConfigureFromEnvironment();

var budget = new MemoryBudget(new[] { new MemoryCharge("weights/host", hostBytes), new MemoryCharge("weights/gpu0", deviceBytes) });
var streamingOptions = new WeightStreamingOptions(budget, "weights/host", new[] { "weights/gpu0" }, tileBytes, tokenRows);
var cases = new List<Case>();
var references = new List<float[][]>();
var metrics = new List<object>();
var snapshots = new List<object>();
var nativeSamples = new List<object>();
var lifecycleSamples = new List<object>();
var cycleResults = new List<object>();
var forwardTimings = new List<object>();
long totalFileBytesRead = 0, totalLinearTiles = 0;
double totalForwardMilliseconds = 0;
WeightStreamingStatistics? usage = null;
long residentParameterBytes = 0;
long expectedFileBytes = 0;
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
        expectedFileBytes = metadata.Tensors.Values.Where(t => t.Type == GgmlTensorType.Q8_0 && t.Shape.Length == 2)
            .Sum(metadata.GetTensorByteCount);
    Require(expectedFileBytes > hostBytes && expectedFileBytes > deviceBytes,
        "The fixture's quantized weights must exceed both configured staging budgets.");

    model = ModelBase.Create(modelPath, BackendType.GgmlCuda);
    Require(model is Qwen35Model && model.StreamingWeightUsage == null, "Resident reference unexpectedly streams weights.");
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
        float[] logits = model.Forward(prompt);
        var rows = new float[steps][];
        int[] generated = new int[steps];
        for (int step = 0; step < steps; step++)
        {
            Require(logits.Length == model.Config.VocabSize && logits.All(float.IsFinite), "Invalid resident reference logits.");
            rows[step] = (float[])logits.Clone();
            generated[step] = Top(logits);
            Require(!model.Tokenizer.IsEos(generated[step]), "Resident counting fixture stopped early; EOS-only output is not validation.");
            if (step + 1 < steps) logits = model.Forward(new[] { generated[step] });
        }
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
        Require(model is Qwen35Model, "Streaming model adapter changed architecture.");
        residentParameterBytes = ((Qwen35Model)model).StreamingResidentParameterBytes;
        Require(model.StreamingWeightUsage?.FileBackedWeightBytes == expectedFileBytes, "Some quantized weights were not represented as file regions.");
        Sample("streamed-loaded");
        SampleNative("streamed-loaded", requireNoPreload: true);
        CheckMappings();
        // A mid-forward allocation refusal can follow writes to activation or state
        // buffers. Releasing quota is insufficient: the model must demand a reset.
        using (budget.Reserve(new[] { new MemoryCharge("weights/gpu0", deviceBytes) }))
        {
            try { model.Forward(cases[0].PromptTokens); }
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
            float[] logits = model.Forward(item.PromptTokens);
            double prefillMilliseconds = Stopwatch.GetElapsedTime(forwardStarted).TotalMilliseconds;
            double decodeMilliseconds = 0;
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
                    decodeMilliseconds += Stopwatch.GetElapsedTime(forwardStarted).TotalMilliseconds;
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
        Require(usage.Value.FileBytesRead > expectedFileBytes && usage.Value.LinearTiles > 100,
            "The test did not repeatedly read bounded weight tiles across real model layers.");
        Require(usage.Value.PeakHostStagingBytes <= hostBytes && usage.Value.PeakDeviceWorkspaceBytes <= deviceBytes,
            "Streaming workspace exceeded its shared budget.");
        SampleLifecycle($"cycle-{cycle}-before-dispose", requireLiveGdn: true, requireEmpty: false);
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
    Passed = error == null, Error = error, Model = modelPath, ModelSha256 = Hash(modelPath),
    Device = Environment.GetEnvironmentVariable("TS_VALIDATION_DEVICE"),
    GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"), Native = native,
    ModelsSha256 = Hash(typeof(ModelBase).Assembly.Location), ProbeSha256 = Hash(typeof(Case).Assembly.Location),
    BackendSha256 = Hash(typeof(GgmlBasicOps).Assembly.Location), MemorySha256 = Hash(typeof(MemoryBudget).Assembly.Location),
    Gate = new { MaxRelativeL2 = 0.001, MinCosine = 0.999999, RequireIdenticalTop1 = true },
    Steps = steps, Cycles = cycles, CompletedCycles = cycleResults.Count, MinimumPromptTokens = promptMinimum, TileBytes = tileBytes, TokenRows = tokenRows,
    HostBytes = hostBytes, DeviceBytes = deviceBytes, ExpectedFileBytes = expectedFileBytes,
    ResidentParameterBytes = residentParameterBytes, Cases = cases, Metrics = metrics, Usage = usage,
    CycleResults = cycleResults, TotalFileBytesRead = totalFileBytesRead, TotalLinearTiles = totalLinearTiles,
    ForwardTimings = forwardTimings, TotalForwardMilliseconds = totalForwardMilliseconds,
    ForwardTimingScope = "Wall time inside successful case prefill/decode ModelBase.Forward calls only; excludes model load, pressure tests, reset, logits comparisons, snapshots and disposal. Same-host configuration comparison only, not a standardized throughput benchmark.",
    NativeLifecycleSamples = lifecycleSamples,
    PressureConstructionRefused = pressureRefused, ForwardPressureRefused = forwardPressureRefusals == cycles,
    RetryRequiredReset = resetRequiredRefusals == cycles, ForwardPressureRefusals = forwardPressureRefusals,
    ResetRequiredRefusals = resetRequiredRefusals, Snapshots = snapshots, FinalBudget = finalBudget,
    NativeSamples = nativeSamples, MappedFileCheckAvailable = mappedFileCheckAvailable, UnexpectedModelMappings = mappings,
    BaselineMilliseconds = baselineMilliseconds, StreamedMilliseconds = streamedMilliseconds,
    Scope = "Real dense Qwen35 single-rank GGML CUDA Q8_0 inference using original file regions and bounded synchronous weight/output-row tiles. Budget covers quantized-weight host staging and CUDA streamed input/weight/output workspace. Small F32 constants, activations, KV/GDN state, other native caches, backend pools, CUDA runtime and OS file cache are excluded. Resident fused and streamed per-op execution are compared with the existing ForcedLogitProbe numerical gates. Timings include load/validation and are not throughput benchmarks. No multi-GPU streaming or end-to-end process memory cap is claimed."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Weight model validation passed={error == null}; report={reportPath}");
return error == null ? 0 : 1;

void Sample(string phase)
{
    var pools = budget.Snapshot();
    Require(pools.All(p => p.Reserved >= 0 && p.Committed >= 0 && p.Available >= 0), "Budget exceeded its configured limit.");
    snapshots.Add(new { Phase = phase, Pools = pools, Usage = model!.StreamingWeightUsage });
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
static class NativeLifecycle
{
    [DllImport("GgmlOps", EntryPoint = "TSGgml_TestGdnChunkedCacheBytes", CallingConvention = CallingConvention.Cdecl)]
    public static extern long GdnChunkedBytes();
    [DllImport("GgmlOps", EntryPoint = "TSGgml_TestQwen35RecurrentPrefillCacheBytes", CallingConvention = CallingConvention.Cdecl)]
    public static extern long RecurrentPrefillBytes();
}
