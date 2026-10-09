// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using InferenceWeb.Tests;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
        throw new ArgumentException("Expected --name value pairs.");
    options.Add(args[i][2..], args[++i]);
}
string Value(string name, string fallback) => options.GetValueOrDefault(name, fallback);
int Number(string name, int fallback, int minimum = 1)
{
    int value = int.Parse(Value(name, fallback.ToString()), System.Globalization.CultureInfo.InvariantCulture);
    return value >= minimum ? value : throw new ArgumentOutOfRangeException(name);
}
string output = Path.GetFullPath(Value("output", "artifacts/qwen4exp-expert-cache-probe.json"));
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
bool synthetic = !options.ContainsKey("model");
bool host = Value("placement", "host") switch
{
    "host" => true,
    "device" => false,
    _ => throw new ArgumentException("placement must be host or device"),
};
BackendType backend = Value("backend", "ggml_cuda") switch
{
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_cpu" => BackendType.GgmlCpu,
    _ => throw new ArgumentException("backend must be ggml_cuda or ggml_cpu"),
};
bool q2kxl = Value("quantization", "mixed") switch
{
    "mixed" => false,
    "q2kxl" => true,
    _ => throw new ArgumentException("quantization must be mixed or q2kxl"),
};
int prefill = Number("prefill-tokens", 40), decode = Number("decode-tokens", 64);
int warmups = Number("warmup", 2, 0), iterations = Number("iterations", 5);
string generation = Value("generation", "teacher-forced");
if (generation is not ("teacher-forced" or "greedy"))
    throw new ArgumentException("generation must be greedy or teacher-forced");
foreach (string name in TensorSharp.Runtime.Speculative.SpeculationEnvVars.RemovedNames.Select(pair => pair.Name)
    .Concat(new[] { "TS_SPEC", "TS_SPEC_TYPE", "TS_SPEC_DRAFT", "TS_SPEC_PMIN", "TS_SPEC_DRAFT_MODEL" }))
    Environment.SetEnvironmentVariable(name, null);
Environment.SetEnvironmentVariable("MAX_CONTEXT", Value("max-context", Math.Max(256, prefill + decode + 16).ToString()));
KvCacheDtypeConfig.Set(KvCacheDtype.F16);
string modelPath = synthetic ? Path.Combine(Path.GetDirectoryName(output)!, "fixture.gguf") : Path.GetFullPath(options["model"]);
long? deviceBudgetBytes = options.TryGetValue("device-budget-bytes", out string? budgetText)
    ? long.Parse(budgetText, System.Globalization.CultureInfo.InvariantCulture) : null;
if (deviceBudgetBytes.HasValue && (deviceBudgetBytes.Value <= 0 || backend != BackendType.GgmlCuda))
    throw new ArgumentException("--device-budget-bytes requires a positive capacity and ggml_cuda.");
long? trimTargetBytes = options.TryGetValue("trim-target-bytes", out string? trimText)
    ? long.Parse(trimText, System.Globalization.CultureInfo.InvariantCulture) : null;
if (trimTargetBytes.HasValue && (trimTargetBytes.Value < 0 || !host || backend != BackendType.GgmlCuda))
    throw new ArgumentException("--trim-target-bytes requires a nonnegative target and host expert placement on CUDA.");
MemoryBudget? sharedBudget = null;
GgmlCacheBudgetScope? cacheScope = null;
ModelBase? model = null;
object? completedReport = null;
object? checkpointIdentity = null;
object? modelGeometry = null;
string? failure = null;
string? observedNativePath = null;
string? observedNativeHash = null;
var cleanupErrors = new List<string>();
var budgetObservations = new List<object>();
var trimObservations = new List<object>();
var rowCaptures = new List<object>();
FileStream? logitsStream = null;
string? logitsPath = null;
string? logitsIndexPath = null;
string[] benchmarkEnvironmentNames = ["CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "MAX_CONTEXT", "OMP_NUM_THREADS",
    "TENSORSHARP_TP_DEGREE", "TS_CPU_MOE", "TS_N_CPU_MOE", "TS_CPU_MOE_THREADS", "TS_GGML_CPU_THREADS",
    "TS_HOST_MOE_DEVICE_MIN_BATCH", "TS_HOST_MOE_EXPERT_CACHE_MB", "TS_HOST_MOE_EXPERT_CACHE_LAYERS",
    "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_EXPERT_CACHE_BRIDGE",
    "TS_HOST_MOE_EXPERT_CACHE_OUTPUT_BRIDGE", "TS_HOST_MOE_PIN", "TS_HOST_MOE_PIN_MAX_MB", "TS_HOST_MOE_TIMING",
    "TS_HOST_MOE_DEBUG", "TS_HOST_MOE_VERIFY", "TS_HOST_MOE_EXPERT_FILTER", "GGML_CUDA_ALLREDUCE", "NCCL_P2P_DISABLE",
    "TS_GGML_MEMORY_BUDGET", "TS_GGML_MEMORY_BUDGET_MB"];
bool modelDisposed = false, cacheCleared = false, reuseReleased = false, scopeDetached = false, shutdown = false;
void CaptureLogits(float[] logits, string stage, int iteration, IReadOnlyList<int> history)
{
    if (logitsStream == null) return;
    if (logits.Length == 0 || logits.Any(value => !float.IsFinite(value)))
        throw new InvalidOperationException($"Empty/nonfinite logits at {stage}, iteration {iteration}.");
    var bytes = MemoryMarshal.AsBytes(logits.AsSpan());
    long offset = logitsStream.Position;
    logitsStream.Write(bytes);
    logitsStream.Flush();
    rowCaptures.Add(new { stage, iteration, warmup = iteration < 0, input_tokens = history.ToArray(),
        byte_offset = offset, elements = logits.Length, sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(),
        argmax = Enumerable.Range(0, logits.Length).MaxBy(index => logits[index]) });
    File.WriteAllText(logitsIndexPath!, JsonSerializer.Serialize(new { format = "f32le", data_path = logitsPath,
        rows = rowCaptures, limitations = "Diagnostic capture I/O is outside forward timers; this is not a quiet performance run." },
        new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
}
void ObserveBudget(string stage)
{
    if (sharedBudget == null) return;
    budgetObservations.Add(new { stage, pools = sharedBudget.Snapshot(), active_allocations = cacheScope?.ActiveAllocations,
        callback_error = cacheScope?.CallbackError?.ToString() });
}
void WriteEvidence(bool complete)
{
    var node = completedReport == null ? new JsonObject() : JsonSerializer.SerializeToNode(completedReport)!.AsObject();
    node["schema_version"] = 2;
    node["passed"] = complete && failure == null && cleanupErrors.Count == 0;
    node["run_complete"] = complete;
    node["requested_options"] = JsonSerializer.SerializeToNode(options);
    node["model_path"] = modelPath;
    node["checkpoint_identity"] = JsonSerializer.SerializeToNode(checkpointIdentity);
    node["model_geometry"] = JsonSerializer.SerializeToNode(modelGeometry);
    node["model_sha256"] = synthetic && File.Exists(modelPath) ? Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(modelPath))).ToLowerInvariant() : null;
    node["native_path"] = observedNativePath;
    node["native_sha256"] = observedNativeHash;
    node["managed_assemblies_sha256"] = JsonSerializer.SerializeToNode(AppDomain.CurrentDomain.GetAssemblies()
        .Where(a => a.GetName().Name?.StartsWith("TensorSharp", StringComparison.Ordinal) == true && !a.IsDynamic)
        .Select(a => a.Location).Where(File.Exists).Distinct().OrderBy(path => path)
        .ToDictionary(path => path, path => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant()));
    node["environment"] = JsonSerializer.SerializeToNode(benchmarkEnvironmentNames
        .Where(name => Environment.GetEnvironmentVariable(name) != null)
        .ToDictionary(name => name, Environment.GetEnvironmentVariable));
    node["device_budget_bytes"] = deviceBudgetBytes;
    node["budget_scope"] = deviceBudgetBytes.HasValue ? "rank0 cache, preload, and explicitly routed graph buffers; not all driver/host/KV allocations" : "disabled";
    node["budget_observations"] = JsonSerializer.SerializeToNode(budgetObservations);
    node["trim_observations"] = JsonSerializer.SerializeToNode(trimObservations);
    node["logit_captures"] = JsonSerializer.SerializeToNode(new { format = "f32le", data_path = logitsPath,
        index_path = logitsIndexPath, rows = rowCaptures });
    node["error"] = failure;
    node["cleanup"] = JsonSerializer.SerializeToNode(new { model_disposed = modelDisposed, cache_cleared = cacheCleared,
        reuse_released = reuseReleased, scope_detached = scopeDetached, native_shutdown = shutdown,
        retained_model_owner = model != null, retained_scope_owner = cacheScope != null, errors = cleanupErrors });
    File.WriteAllText(output, node.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
}
try
{
GgmlBasicOps.EnsureBackendAvailable(backend == BackendType.GgmlCuda ? GgmlBackendType.Cuda : GgmlBackendType.Cpu);
observedNativePath = Qwen4ExpExpertCacheScenario.MappedNativePath();
observedNativeHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(observedNativePath))).ToLowerInvariant();
if (options.TryGetValue("logits-dir", out string? captureDirectory))
{
    if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Diagnostic capture requires little-endian float32.");
    captureDirectory = Path.GetFullPath(captureDirectory);
    Directory.CreateDirectory(captureDirectory);
    logitsPath = Path.Combine(captureDirectory, "rows.f32");
    logitsIndexPath = Path.Combine(captureDirectory, "rows.json");
    if (File.Exists(logitsIndexPath)) throw new IOException("Logit index already exists; use a fresh capture directory.");
    logitsStream = new FileStream(logitsPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
}
if (deviceBudgetBytes.HasValue)
{
    sharedBudget = new MemoryBudget([new MemoryCharge("gpu0", deviceBudgetBytes.Value)]);
    cacheScope = new GgmlCacheBudgetScope(sharedBudget, [["gpu0"]], includeGraphBuffers: true);
    ObserveBudget("attached-before-model-load");
}
WriteEvidence(false);
Dictionary<string, float[][]>? captures = null;
if (synthetic)
{
    Qwen4ExpSyntheticModelBuilder.Write(modelPath, q2kxlExperts: q2kxl);
    string other = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(Path.GetDirectoryName(output)!, "other-fixture.gguf"), q2kxlExperts: !q2kxl);
    captures = Qwen4ExpExpertCacheScenario.Exercise(modelPath, other, backend, host);
    foreach (string name in new[] { "a2", "reloaded-a" })
    {
        var difference = Qwen4ExpExpertCacheScenario.Difference(captures[name], captures["a1"]);
        if (difference.RelativeL2 != 0 || difference.ArgmaxMismatches != 0)
            throw new InvalidOperationException("Reset/disposal changed deterministic logits: " + name);
    }
}

MoeCpuOffloadConfig.Reset();
if (host) MoeCpuOffloadConfig.SetAllLayers();
if (options.TryGetValue("model-identity-report", out string? identityReport))
    checkpointIdentity = ProbeCheckpointIdentity.Read(modelPath, identityReport);
var load = Stopwatch.StartNew();
model = ModelBase.Create(modelPath, backend);
load.Stop();
modelGeometry = new { architecture = model.Config.Architecture, hidden_size = model.Config.HiddenSize,
    layers = model.Config.NumLayers, heads = model.Config.NumHeads, kv_heads = model.Config.NumKVHeads,
    key_length = model.Config.KeyLength, value_length = model.Config.ValueLength,
    vocabulary = model.Config.VocabSize, context_limit = model.MaxContextLength, kv_dtype = "f16" };
ObserveBudget("model-loaded");
string nativePath = Qwen4ExpExpertCacheScenario.MappedNativePath();
string nativeHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(nativePath))).ToLowerInvariant();
var managedAssemblies = AppDomain.CurrentDomain.GetAssemblies()
    .Where(assembly => assembly.GetName().Name?.StartsWith("TensorSharp", StringComparison.Ordinal) == true && !assembly.IsDynamic)
    .Select(assembly => assembly.Location).Distinct().OrderBy(path => path)
    .ToDictionary(path => path, path => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant());
int[] tokenPool = synthetic ? Enumerable.Range(0, 251).ToArray() : model.Tokenizer.Encode(
    "The history of computing spans many centuries, beginning with counting tools and algorithms. ", addSpecial: false).ToArray();
if (tokenPool.Length == 0) throw new InvalidOperationException("Empty benchmark token pool.");
int[] prompt = Enumerable.Range(0, prefill).Select(i => tokenPool[(i * 37 + 11) % tokenPool.Length]).ToArray();
int[] forced = Enumerable.Range(0, decode).Select(i => tokenPool[(i * 53 + 17) % tokenPool.Length]).ToArray();
string? renderedPrompt = null;
if (options.TryGetValue("tokens-file", out string? tokenFile))
    prompt = File.ReadAllText(tokenFile).Split(new[] { ',', ' ', '\r', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries)
        .Select(value => int.Parse(value, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
else if (options.TryGetValue("prompt-raw-file", out string? rawFile))
{
    renderedPrompt = File.ReadAllText(rawFile);
    prompt = model.Tokenizer.Encode(renderedPrompt, addSpecial: true).ToArray();
}
else if (options.ContainsKey("prompt") || options.ContainsKey("prompt-file"))
{
    string text = options.TryGetValue("prompt-file", out string? promptFile) ? File.ReadAllText(promptFile) : options["prompt"];
    renderedPrompt = new GgufPromptRenderer().Render(model.Config.ChatTemplate, new List<ChatMessage>
    {
        new() { Role = "system", Content = Value("system", "You are a helpful assistant.") },
        new() { Role = "user", Content = text },
    }, architecture: model.Config.Architecture, enableThinking: Value("thinking", "false") == "true");
    prompt = model.Tokenizer.Encode(renderedPrompt, addSpecial: true).ToArray();
}
if (prompt.Length == 0 || prompt.Any(token => token < 0 || token >= model.Tokenizer.VocabSize))
    throw new ArgumentException("Prompt must contain valid vocabulary token IDs.");
if (prompt.Length + decode > int.Parse(Environment.GetEnvironmentVariable("MAX_CONTEXT")!))
    throw new ArgumentException("Prompt plus generation exceeds --max-context.");
bool requireAllCache = Number("require-all-cache", 0, 0) != 0;
if (requireAllCache && (!host || backend != BackendType.GgmlCuda || prompt.Length > 8))
    throw new ArgumentException("--require-all-cache requires all-host experts on CUDA with a 1..8 token prefill.");
if (options.TryGetValue("prompt-tokens-output", out string? tokenOutput))
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(tokenOutput))!);
    File.WriteAllText(tokenOutput, string.Join(",", prompt) + Environment.NewLine);
}
var runs = new List<object>();
string DecodeGeneration(IEnumerable<int> generated) => model.Tokenizer.Decode(
    generated.TakeWhile(token => !model.Tokenizer.IsEos(token)).ToList());
var statsBeforeTimedWork = backend == BackendType.GgmlCuda ? Qwen4ExpExpertCacheScenario.CacheStats() : null;
string? referenceHash = null;
float[]? finalLogits = null;
int[]? finalGenerated = null;
long expectedCachedRows = 0;
for (int i = -warmups; i < iterations; i++)
{
    if (i >= 0 && trimTargetBytes.HasValue)
    {
        // Quiescent between requests; preserve the model and every KV/graph owner.
        // Capture all logits to compare cold refills and warm decode afterwards.
        var before = Qwen4ExpExpertCacheScenario.CacheStats();
        var trimTimer = Stopwatch.StartNew();
        long released = GgmlBasicOps.TrimHostMoeExpertCache(trimTargetBytes.Value);
        trimTimer.Stop();
        var after = Qwen4ExpExpertCacheScenario.CacheStats();
        if (after.ReservedBytes > trimTargetBytes.Value || before.ReservedBytes - after.ReservedBytes != released
            || before.Calls != after.Calls || before.Hits != after.Hits || before.Misses != after.Misses)
            throw new InvalidOperationException("Expert-cache trim changed counters or failed its release target.");
        trimObservations.Add(new { iteration = i, target_bytes = trimTargetBytes.Value, released_bytes = released,
            milliseconds = trimTimer.Elapsed.TotalMilliseconds, before, after });
        ObserveBudget($"iteration-{i}-after-expert-trim");
        WriteEvidence(false);
    }
    model.ResetKVCache();
    var timer = Stopwatch.StartNew();
    float[] logits = model.ForwardRefill(prompt);
    timer.Stop();
    ObserveBudget($"iteration-{i}-prefill-complete");
    double prefillMs = timer.Elapsed.TotalMilliseconds;
    var inputHistory = new List<int>(prompt);
    CaptureLogits(logits, "prefill", i, inputHistory);
    var generated = new List<int>();
    bool eos = false;
    int forwardSteps = 0;
    timer.Restart();
    if (generation == "teacher-forced")
    {
        foreach (int token in forced)
        {
            logits = model.Forward(new[] { token });
            timer.Stop();
            inputHistory.Add(token);
            CaptureLogits(logits, "decode", i, inputHistory);
            timer.Start();
        }
        forwardSteps = forced.Length;
    }
    else
    {
        for (int position = 0; position < decode; position++)
        {
            int token = 0;
            for (int j = 1; j < logits.Length; j++) if (logits[j] > logits[token]) token = j;
            generated.Add(token);
            if (model.Tokenizer.IsEos(token)) { eos = true; break; }
            if (position + 1 < decode)
            {
                logits = model.Forward(new[] { token }); forwardSteps++;
                timer.Stop();
                inputHistory.Add(token);
                CaptureLogits(logits, "decode", i, inputHistory);
                timer.Start();
            }
        }
    }
    timer.Stop();
    expectedCachedRows = checked(expectedCachedRows + (long)(prompt.Length + forwardSteps) * model.Config.NumLayers);
    ObserveBudget($"iteration-{i}-decode-complete");
    if (logits.Length == 0 || logits.Any(v => !float.IsFinite(v)))
        throw new InvalidOperationException("The benchmark returned empty or nonfinite final logits.");
    byte[] bytes = new byte[logits.Length * sizeof(float)];
    Buffer.BlockCopy(logits, 0, bytes, 0, bytes.Length);
    string hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    referenceHash ??= hash;
    if (referenceHash != hash) throw new InvalidOperationException("Identical repeated inputs changed final logits.");
    finalLogits = (float[])logits.Clone();
    finalGenerated = generated.ToArray();
    var row = new { warmup = i < 0, iteration = i, prefill_tokens = prompt.Length,
        decode_tokens = forwardSteps, generated_tokens = finalGenerated,
        selected_text = generation == "greedy" ? DecodeGeneration(generated) : null,
        finish_reason = generation == "greedy" ? (eos ? "eos" : "length") : "teacher-forced",
        prefill_ms = prefillMs, decode_ms = timer.Elapsed.TotalMilliseconds,
        prefill_tps = prompt.Length / (prefillMs / 1000), decode_tps = forwardSteps / timer.Elapsed.TotalSeconds,
        final_logit_sha256 = hash };
    runs.Add(row);
    Console.WriteLine(JsonSerializer.Serialize(row));
}
Qwen4ExpExpertCacheScenario.Stats? stats = null;
if (backend == BackendType.GgmlCuda)
{
    stats = Qwen4ExpExpertCacheScenario.CacheStats();
    if (stats.ReservedBytes < 0 || stats.ReservedBytes > stats.BudgetBytes)
        throw new InvalidOperationException("Expert-cache device reservation exceeds its budget.");
    bool hasReuse = stats.Hits > statsBeforeTimedWork!.Hits;
    if (host && Number("require-cache", 0, 0) != 0 && !(stats.Calls > statsBeforeTimedWork.Calls
        && hasReuse && stats.Misses > statsBeforeTimedWork.Misses))
        throw new InvalidOperationException("Expert cache did not engage and reuse selected weights.");
    if (requireAllCache && stats.Calls - statsBeforeTimedWork.Calls != expectedCachedRows)
        throw new InvalidOperationException($"Not every expert row used the cache: expected {expectedCachedRows}, " +
            $"observed {stats.Calls - statsBeforeTimedWork.Calls}. Some rows fell back or were unaccounted.");
}
using var process = Process.GetCurrentProcess();
process.Refresh();
completedReport = new
{
    schema_version = 1,
    passed = true,
    synthetic,
    language_quality_validated = false,
    decode_mode = generation,
    backend = backend.ToString(),
    placement = host ? "host" : "device",
    quantization = synthetic ? (q2kxl ? "q2kxl" : "mixed") : "checkpoint",
    model_path = modelPath,
    model_bytes = new FileInfo(modelPath).Length,
    model_sha256 = synthetic ? Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(modelPath))).ToLowerInvariant() : null,
    native_path = nativePath,
    native_sha256 = nativeHash,
    managed_assemblies_sha256 = managedAssemblies,
    model_load_ms = load.Elapsed.TotalMilliseconds,
    process_peak_working_set_bytes = process.PeakWorkingSet64,
    process_working_set_bytes = process.WorkingSet64,
    managed_bytes = GC.GetTotalMemory(false),
    cache_stats = stats,
    cache_stats_before_timed_work = statsBeforeTimedWork,
    expected_cached_rows = expectedCachedRows,
    all_cache_rows_required = requireAllCache,
    prompt_tokens = prompt,
    rendered_prompt = renderedPrompt,
    generated_tokens = finalGenerated,
    selected_text = generation == "greedy" ? DecodeGeneration(finalGenerated!) : null,
    raw_decoded_output = generation == "greedy" ? model.Tokenizer.Decode(finalGenerated!.ToList()) : null,
    final_logits = finalLogits,
    forced_tokens = generation == "teacher-forced" ? forced : null,
    captures,
    runs,
    limitations = new[]
    {
        "Synthetic checkpoints exercise engines and cache ownership; they cannot establish trained language quality or real-model throughput.",
        "Teacher-forced decode excludes sampling and serves identical token inputs; it is not an HTTP end-to-end throughput result.",
        "Working-set and external GPU telemetry are process/driver measurements, not a full allocator peak.",
        "The cache is opt-in; eligible bias-free SiLU expert paths with one through eight rows engage it."
    }
};
}
catch (Exception error)
{
    failure = error.ToString();
    Console.Error.WriteLine(failure);
    ObserveBudget("operation-failed-before-cleanup");
}
finally
{
    try { logitsStream?.Dispose(); logitsStream = null; }
    catch (Exception error) { cleanupErrors.Add("logit capture.Dispose: " + error); }
    // Preserve both physical owners and callback roots if a cleanup step fails.
    // A failed scope detach must not turn into a successful zero-budget result.
    try { model?.Dispose(); model = null; modelDisposed = true; }
    catch (Exception error) { cleanupErrors.Add("model.Dispose: " + error); }
    if (modelDisposed)
    {
        try { GgmlBasicOps.ClearHostBufferCache(); cacheCleared = true; }
        catch (Exception error) { cleanupErrors.Add("ClearHostBufferCache: " + error); }
        try { GgmlBasicOps.ReleaseReuseComputeBuffers(); reuseReleased = true; }
        catch (Exception error) { cleanupErrors.Add("ReleaseReuseComputeBuffers: " + error); }
    }
    ObserveBudget("after-physical-cleanup-before-detach");
    if (cacheScope != null)
    {
        if (cacheScope.CallbackError != null) cleanupErrors.Add("budget callback: " + cacheScope.CallbackError);
        if (cacheScope.ActiveAllocations != 0 || sharedBudget!.Snapshot().Any(pool => pool.Reserved != 0 || pool.Committed != 0))
            cleanupErrors.Add("Covered native allocations retain budget credit after physical cleanup.");
        else if (cleanupErrors.Count == 0)
        {
            try { cacheScope.Dispose(); cacheScope = null; scopeDetached = true; }
            catch (Exception error) { cleanupErrors.Add("budget scope.Dispose: " + error); }
        }
    }
    WriteEvidence(false); // Keep ownership evidence even if a native shutdown aborts.
    if (cleanupErrors.Count == 0 && cacheScope == null)
    {
        try { GgmlBasicOps.Shutdown(); shutdown = true; }
        catch (Exception error) { cleanupErrors.Add("native Shutdown: " + error); }
    }
    WriteEvidence(true);
}
Console.WriteLine("report=" + output);
Environment.ExitCode = failure == null && cleanupErrors.Count == 0 ? 0 : 1;
