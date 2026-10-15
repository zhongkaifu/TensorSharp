// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using InferenceWeb.Tests;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using DirectCudaProbe;

// Fresh-process full-model performance, numerical replay and small semantic probe.
// Keep compilation and A/B process ordering in the invoking runner.
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
bool synthetic = options.TryGetValue("synthetic-q4e", out string? syntheticPath);
if (synthetic && options.ContainsKey("model")) throw new ArgumentException("Choose --model or --synthetic-q4e.");
string modelPath = Path.GetFullPath(synthetic ? syntheticPath! : options["model"]);
string output = Path.GetFullPath(options["output"]);
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
BackendType backend = Value("backend", "cuda") switch
{
    "cuda" => BackendType.Cuda,
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_cpu" => BackendType.GgmlCpu,
    _ => throw new ArgumentException("backend must be cuda, ggml_cuda or ggml_cpu"),
};
int prefill = Number("prefill-tokens", 64), decode = Number("decode-tokens", 64);
int warmups = Number("warmup", 2, 0), iterations = Number("iterations", 5);
int maxNew = Number("max-new", 128), maxContext = Number("max-context", 1024);
bool runQuality = !synthetic && Value("quality", "true") == "true";
bool gptossPerExpertPrefill = bool.Parse(Value("gptoss-per-expert-prefill", "false"));
long? ggmlDeviceCopyBudget = options.TryGetValue("ggml-device-copy-budget-bytes", out string? copyBudgetText)
    ? long.Parse(copyBudgetText, System.Globalization.CultureInfo.InvariantCulture) : null;
if (ggmlDeviceCopyBudget < 0 || (ggmlDeviceCopyBudget.HasValue && backend is not (BackendType.GgmlCuda or BackendType.GgmlCpu)))
    throw new ArgumentException("--ggml-device-copy-budget-bytes requires a nonnegative byte count and a GGML backend; 0 means unlimited.");
long? ggmlHostStagingBudget = HostStagingBudgetOption.Parse(options.GetValueOrDefault("ggml-host-staging-budget-bytes"),
    backend is BackendType.GgmlCuda or BackendType.GgmlCpu);
if (gptossPerExpertPrefill && (synthetic || backend != BackendType.GgmlCuda || prefill < 1025 || runQuality))
    throw new ArgumentException("--gptoss-per-expert-prefill true requires a real GPT-OSS model, --backend ggml_cuda, --prefill-tokens >=1025 and --quality false.");
string qualityParser = Value("quality-parser", "raw");
if (qualityParser is not ("raw" or "production")) throw new ArgumentException("quality-parser must be raw or production.");
string[] qualityCases = QualityOutput.SelectCases(Value("quality-cases", "math,extraction,squares"));
string? qualityReasoningEffort = QualityPrompt.ParseReasoningEffort(options.GetValueOrDefault("quality-reasoning-effort"));
if (synthetic)
{
    Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
    Qwen4ExpSyntheticModelBuilder.Write(modelPath, contextLength: maxContext,
        indexerTopK: Number("synthetic-indexer-top-k", 16),
        q2kxlExperts: Value("synthetic-q2kxl", "false") == "true",
        expertCount: Number("synthetic-experts", Qwen4ExpSyntheticModelBuilder.Experts),
        expertUsedCount: Number("synthetic-experts-used", Qwen4ExpSyntheticModelBuilder.ExpertsUsed));
}
if (prefill + decode > maxContext) throw new ArgumentException("Prompt plus decode exceeds max-context.");
Environment.SetEnvironmentVariable("MAX_CONTEXT", maxContext.ToString());
if (gptossPerExpertPrefill)
{
    // Probe-only dispatch control, installed before GPT-OSS's static limits are
    // initialized. One exact-width prefill retains fused attention but exceeds
    // the per-layer stacked MoE limit of 1024 tokens.
    foreach (string name in new[] { "TS_PREFILL_CHUNK", "TS_GPTOSS_PREFILL_CHUNK", "TS_GPTOSS_FUSED_ATTN_MAX_SEQ" })
        Environment.SetEnvironmentVariable(name, prefill.ToString(System.Globalization.CultureInfo.InvariantCulture));
}
foreach (string name in SpeculativeNames()) Environment.SetEnvironmentVariable(name, null);
if (options.TryGetValue("moe-cpu-layers", out string? offload))
{
    if (offload == "all") MoeCpuOffloadConfig.SetAllLayers();
    else MoeCpuOffloadConfig.SetLayers(int.Parse(offload, System.Globalization.CultureInfo.InvariantCulture));
}
MemoryBudget? hostStagingLedger = null;
GgmlCacheBudgetScope? hostStagingScope = null;
object? hostStagingReport = null;
bool hostStagingCleanupPassed = !ggmlHostStagingBudget.HasValue;
Action<ProbeLifetimeState>? observeHostStaging = null;
Func<IDisposable?>? attachHostStaging = null;
if (ggmlHostStagingBudget is long hostLimit)
{
    // Device cache allocations remain independent of the host quota. Map every
    // available rank so opting into host staging does not disable other ranks.
    int ranks = backend == BackendType.GgmlCuda
        ? Math.Max(1, GgmlBasicOps.GetGpuDeviceCount(GgmlBackendType.Cuda)) : 1;
    hostStagingLedger = new MemoryBudget([new("host-staging", hostLimit), new("device-cache-unlimited", long.MaxValue)]);
    attachHostStaging = () => hostStagingScope = new GgmlCacheBudgetScope(hostStagingLedger,
        Enumerable.Range(0, ranks).Select(_ => new[] { "device-cache-unlimited" }),
        includeGraphBuffers: false, hostPools: ["host-staging"]);
    observeHostStaging = state =>
    {
        var pools = hostStagingLedger.Snapshot();
        var peaks = hostStagingLedger.HighWatermarks();
        var refusals = hostStagingScope?.AllocationRefusals;
        bool ownersZero = hostStagingScope?.ActiveAllocations == 0 && pools.All(pool => pool.Reserved == 0 && pool.Committed == 0);
        hostStagingCleanupPassed = state.ModelCreated && state.ModelDisposed && state.ScopeDisposed
            && state.Error is null && hostStagingScope?.CallbackError is null && ownersZero;
        hostStagingReport = new
        {
            requested_bytes = hostLimit, zero_means = "strict zero quota", device_rank_count = ranks,
            device_cache_pool_capacity_bytes = long.MaxValue, includes_graph_buffers = false,
            high_watermarks = peaks, allocation_refusal_count = refusals?.Count,
            first_allocation_refusal = refusals?.First, last_allocation_refusal = refusals?.Last,
            final_pool_ownership = pools, final_allocation_usage = hostStagingScope?.AllocationUsage,
            final_active_allocations = hostStagingScope?.ActiveAllocations, final_owners_zero = ownersZero,
            model_created = state.ModelCreated, model_disposed = state.ModelDisposed,
            scope_attached = hostStagingScope != null, scope_detached = hostStagingScope != null && state.ScopeDisposed,
            callback_error = hostStagingScope?.CallbackError?.ToString(), lifecycle_error = state.Error?.ToString(),
            host_payload_commit_observed = peaks.Single(pool => pool.Pool == "host-staging").Committed > 0,
            cleanup_passed = hostStagingCleanupPassed,
            scope = "Opt-in probe-only shared quota for instrumented native host staging payload (kind 3), installed before model construction. GLM upload scratch is included by supporting native builds. Device caches use a separate long.MaxValue pool; graph buffers are excluded. Weights, mmap/page-cache residency, tokenizer/GC, FILE/thread/allocator metadata, backend pools and driver memory are not covered. This is not a complete host/RAM/model physical cap. Zero observed host payload cannot prove uploader instrumentation engagement.",
        };
        // Retain ownership/refusal evidence even when model construction or
        // cleanup fails before the ordinary successful probe report exists.
        File.WriteAllText(Path.ChangeExtension(output, ".host-staging-budget.json"),
            JsonSerializer.Serialize(hostStagingReport, new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
    };
}
var timer = Stopwatch.StartNew();
using var lifetime = new ProbeModelLifetime<ModelBase>(() => ModelBase.Create(modelPath, backend),
    attachHostStaging, observeHostStaging);
var model = lifetime.Model;
timer.Stop();
double loadMs = timer.Elapsed.TotalMilliseconds;
GgmlCacheMemoryUsage? ggmlCacheBeforeBudget = null, ggmlCacheAfterBudget = null;
if (ggmlDeviceCopyBudget is long copyBudget)
{
    // Probe-only cache control. Initialization/preload is complete, but no
    // Forward has run. Never clear preloads: their host bytes may be released.
    if (!GgmlBasicOps.TryGetCacheMemoryUsage(0, out var beforeBudget))
        throw new InvalidOperationException("The requested GGML copy budget cannot be verified by this native library.");
    ggmlCacheBeforeBudget = beforeBudget;
    GgmlBasicOps.SetDeviceCopyBudget(copyBudget);
    if (!GgmlBasicOps.TryGetCacheMemoryUsage(0, out var afterBudget) || afterBudget.DeviceCopyBudgetBytes != copyBudget)
        throw new InvalidOperationException("The requested GGML copy budget was not applied to rank 0.");
    ggmlCacheAfterBudget = afterBudget;
    Console.WriteLine($"[probe] GGML lazy device-copy budget={copyBudget} bytes (0=unlimited), existing copies={afterBudget.DeviceCopyCommittedBytes}, preloads={afterBudget.PreloadCommittedBytes}; graph/driver memory excluded");
}
FieldInfo? gptossPrefillUnavailable = null, gptossExpertCounts = null;
int gptossExpectedAssignments = 0;
var gptossObservedAssignmentEnds = new List<int>();
if (gptossPerExpertPrefill)
{
    if (model is not GptOssModel)
        throw new ArgumentException("--gptoss-per-expert-prefill requires GptOssModel; refusing an unrelated architecture.");
    const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
    gptossPrefillUnavailable = typeof(GptOssModel).GetField("_modelPrefillUnavailable", flags)
        ?? throw new MissingFieldException("GPT-OSS prefill dispatch guard is unavailable.");
    gptossExpertCounts = typeof(GptOssModel).GetField("_moeExpertCounts", flags)
        ?? throw new MissingFieldException("GPT-OSS per-expert routing scratch is unavailable.");
    gptossExpectedAssignments = checked(prefill * model.Config.NumExpertsUsed);
    gptossPrefillUnavailable.SetValue(model, true);
    Console.WriteLine("[probe] GPT-OSS per-expert prefill requested: whole-model prefill disabled, exact N=" + prefill);
}
void PrepareGptOssPrefillEvidence()
{
    if (!gptossPerExpertPrefill) return;
    if (gptossPrefillUnavailable!.GetValue(model) is not true ||
        gptossExpertCounts!.GetValue(model) is not int[] counts || counts.Length != model.Config.NumExperts || counts.Length == 0)
        throw new InvalidOperationException("GPT-OSS fallback dispatch or routing scratch changed.");
    // This scratch is cleared and rebuilt by MoEForwardBatchedPerExpert. The
    // sentinel prevents a previous successful replay from supplying false proof.
    Array.Fill(counts, -1);
}
void VerifyGptOssPrefillEvidence()
{
    if (!gptossPerExpertPrefill) return;
    var ends = (int[])gptossExpertCounts!.GetValue(model)!;
    if (ends[^1] != gptossExpectedAssignments || ends.Any(value => value < 0) ||
        ends.Zip(ends.Skip(1), (a, b) => a <= b).Any(ordered => !ordered))
        throw new InvalidOperationException("The requested exact-width GPT-OSS per-expert prefill did not publish its routing assignments.");
    gptossObservedAssignmentEnds.Add(ends[^1]);
}

// Read the existing allocator only for diagnostics. Do not create a second CUDA
// context, which would change memory pressure and invalidate the measurement.
var allocator = typeof(ModelBase).GetField("_allocator", BindingFlags.Instance | BindingFlags.NonPublic)
    ?.GetValue(model) as CudaAllocator;
var engineDevices = new List<(CudaAllocator Allocator, object Device)>();
object? engine = model.GetType().GetProperty("Engine", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(model);
if (engine?.GetType().GetField("_devs", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(engine) is Array devices)
    foreach (object device in devices)
        if (device.GetType().GetField("Alloc")?.GetValue(device) is CudaAllocator engineAllocator)
            engineDevices.Add((engineAllocator, device));
allocator ??= engineDevices.Select(d => d.Allocator).FirstOrDefault();
object? kernels = allocator?.GetType().GetProperty("Kernels", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(allocator);
bool? kernelAvailable = kernels?.GetType().GetProperty("SupportsFusedMoeQuantize", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(kernels) as bool?;
bool? fusionEnabled = kernels?.GetType().GetProperty("MoeFusionEnabled", BindingFlags.Static | BindingFlags.NonPublic)?.GetValue(null) as bool?;
if (backend == BackendType.Cuda && Environment.GetEnvironmentVariable("TENSORSHARP_CUDA_MOE_FUSION") == "1"
    && kernelAvailable != true)
    throw new InvalidOperationException("Fusion was explicitly requested, but the loaded CUDA module lacks its kernel. Rebuild and copy current PTX.");
var fusion = new { requested = Environment.GetEnvironmentVariable("TENSORSHARP_CUDA_MOE_FUSION"),
    kernel_available = kernelAvailable, process_enabled = fusionEnabled,
    eligible_calls_will_use_fusion = kernelAvailable == true && fusionEnabled == true,
    engagement_limitation = "Availability and process policy do not prove that a particular model selected an eligible MoE path." };
var scratch = new List<object>();
foreach ((CudaAllocator engineAllocator, object device) in engineDevices)
{
    object? moe = device.GetType().GetField("Moe")?.GetValue(device);
    if (moe is null) continue;
    var unique = new List<Storage>();
    var fields = new List<object>();
    foreach (FieldInfo field in moe.GetType().GetFields(BindingFlags.Public | BindingFlags.Instance))
    {
        if (field.GetValue(moe) is not Tensor tensor) continue;
        int storage = unique.FindIndex(s => ReferenceEquals(s, tensor.Storage));
        if (storage < 0) { storage = unique.Count; unique.Add(tensor.Storage); }
        fields.Add(new { name = field.Name, unique_storage_index = storage, bytes = tensor.Storage.ByteLength });
    }
    scratch.Add(new { device = engineAllocator.DeviceId, unique_storage_bytes = unique.Sum(s => s.ByteLength),
        unique_storage_count = unique.Count, fields });
}
var memory = new List<object>();
void Memory(string stage)
{
    using var process = Process.GetCurrentProcess();
    process.Refresh();
    object? device = null;
    if (allocator is not null)
    {
        allocator.Synchronize();
        (long free, long total) = allocator.GetMemoryInfo();
        device = new { free_bytes = free, total_bytes = total, used_bytes = total - free, pool = allocator.GetStats() };
    }
    GgmlCacheMemoryUsage? ggmlCache = null;
    if ((backend is BackendType.GgmlCuda or BackendType.GgmlCpu)
        && GgmlBasicOps.TryGetCacheMemoryUsage(0, out var usage)) ggmlCache = usage;
    if (ggmlDeviceCopyBudget.HasValue && ggmlCache?.DeviceCopyBudgetBytes != ggmlDeviceCopyBudget.Value)
        throw new InvalidOperationException("The requested GGML copy budget became unavailable or changed during the probe.");
    memory.Add(new { stage, process_working_set_bytes = process.WorkingSet64,
        process_peak_working_set_bytes = process.PeakWorkingSet64, process_private_bytes = process.PrivateMemorySize64,
        managed_bytes = GC.GetTotalMemory(false), cuda = device, ggml_cache = ggmlCache });
}
Memory("loaded");
int[] pool = synthetic ? Enumerable.Range(0, 251).ToArray()
    : model.Tokenizer.Encode("The history of computing includes counting tools, algorithms, logic and modern electronic computers. ", addSpecial: false).ToArray();
if (pool.Length == 0) throw new InvalidOperationException("Empty tokenization.");
int[] prompt = Enumerable.Range(0, prefill).Select(i => pool[i % pool.Length]).ToArray();
// Step 37 is coprime to the 17-token real prompt pool and 251-token fixture pool,
// so the controlled decode visits varied inputs instead of one repeated token.
int[] forced = Enumerable.Range(0, decode).Select(i => pool[(i * 37 + 5) % pool.Length]).ToArray();
var runs = new List<object>();
string? repeatedHash = null;
string? coldHash = null;
var warmupHashes = new List<string>();
bool repeatsExact = true;
for (int i = -warmups; i < iterations; i++)
{
    model.ResetKVCache();
    PrepareGptOssPrefillEvidence();
    timer.Restart();
    float[] logits = model.ForwardRefill(prompt);
    timer.Stop();
    double prefillMs = timer.Elapsed.TotalMilliseconds;
    VerifyGptOssPrefillEvidence();
    timer.Restart();
    foreach (int token in forced) logits = model.Forward(new[] { token });
    timer.Stop();
    double decodeMs = timer.Elapsed.TotalMilliseconds;
    CheckLogits(logits);
    string hash = HashLogits(logits);
    coldHash ??= hash;
    if (i < 0) warmupHashes.Add(hash);
    else
    {
        repeatedHash ??= hash;
        repeatsExact &= repeatedHash == hash;
    }
    var row = new { iteration = i, warmup = i < 0, prefill_tokens = prompt.Length, decode_tokens = forced.Length,
        prefill_ms = prefillMs, decode_ms = decodeMs, prefill_tps = prompt.Length * 1000.0 / prefillMs,
        decode_tps = forced.Length * 1000.0 / decodeMs, final_logit_sha256 = hash };
    runs.Add(row);
    Console.WriteLine(JsonSerializer.Serialize(row));
    Memory(i < 0 ? $"warmup-{i + warmups}" : $"timed-{i}");
}

// Separate untimed replay captures every full vocabulary row, including prefill.
// Binary rows are little-endian float32, contiguous and vocabulary-sized.
string logitsPath = Path.ChangeExtension(output, ".logits.f32");
string? captureHash = null;
int vocabulary = 0;
using (var writer = new BinaryWriter(File.Create(logitsPath)))
{
    model.ResetKVCache();
    PrepareGptOssPrefillEvidence();
    float[] logits = model.ForwardRefill(prompt);
    VerifyGptOssPrefillEvidence();
    void WriteRow(float[] values)
    {
        CheckLogits(values);
        if (vocabulary == 0) vocabulary = values.Length;
        if (values.Length != vocabulary) throw new InvalidOperationException("Vocabulary size changed.");
        foreach (float value in values) writer.Write(value);
    }
    WriteRow(logits);
    foreach (int token in forced) { logits = model.Forward(new[] { token }); WriteRow(logits); }
    captureHash = HashLogits(logits);
    repeatsExact &= captureHash == repeatedHash;
}
var quality = new List<object>();
bool qualityPassed = true;
if (runQuality)
{
    foreach ((string name, string text) in QualityOutput.Cases.Where(test => qualityCases.Contains(test.Name, StringComparer.Ordinal)))
    {
        QualityPrompt qualityPrompt = QualityPrompt.Create(model.Config.ChatTemplate, model.Config.Architecture, text, qualityReasoningEffort);
        string rendered = qualityPrompt.Text;
        int[] input = model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
        if (input.Length + maxNew > maxContext) throw new InvalidOperationException("Quality case exceeds context.");
        model.ResetKVCache();
        timer.Restart();
        float[] logits = model.ForwardRefill(input);
        double firstMs = timer.Elapsed.TotalMilliseconds;
        var generated = new List<int>();
        bool eos = false;
        for (int position = 0; position < maxNew; position++)
        {
            CheckLogits(logits);
            int token = model.SampleGreedy(logits);
            generated.Add(token);
            if (model.Tokenizer.IsEos(token)) { eos = true; break; }
            if (position + 1 < maxNew) logits = model.Forward(new[] { token });
        }
        timer.Stop();
        string answer = model.Tokenizer.Decode(generated.TakeWhile(token => !model.Tokenizer.IsEos(token)).ToList());
        QualitySelection selection = QualityOutput.Parse(answer, model.Config.Architecture, rendered, qualityParser);
        bool passed = QualityOutput.CompleteCaseAnswer(name, selection, eos, rendered) && QualityOutput.Semantic(name, selection.Content.Trim());
        qualityPassed &= passed;
        var row = new { name, prompt = text, prompt_tokens = input, generated_tokens = generated,
            reasoning_effort_requested = qualityReasoningEffort, rendered_prompt = rendered,
            rendered_prompt_utf8_sha256 = qualityPrompt.Utf8Sha256,
            selected_text = answer, raw_text = selection.RawText, parsed_text = selection.Content,
            reasoning_text = selection.Thinking, parsed_tool_calls = selection.ToolCalls,
            thinking_closed = QualityOutput.ThinkingClosed(rendered, selection.RawText),
            parser = qualityParser, eos, reached_generation_limit = !eos && generated.Count == maxNew,
            passed, first_token_ms = firstMs, total_ms = timer.Elapsed.TotalMilliseconds };
        quality.Add(row);
        Console.WriteLine(JsonSerializer.Serialize(row));
    }
}
Memory("complete");
var assemblies = AppDomain.CurrentDomain.GetAssemblies().Where(a => !a.IsDynamic
        && (a.GetName().Name?.StartsWith("TensorSharp", StringComparison.Ordinal) == true || a == Assembly.GetExecutingAssembly()))
    .Select(a => a.Location).Where(File.Exists).Distinct().Order().ToDictionary(path => path, path => Sha(path));
string ptxDirectory = Path.Combine(AppContext.BaseDirectory, "cuda_kernels");
var ptx = Directory.Exists(ptxDirectory) ? Directory.GetFiles(ptxDirectory, "*.ptx").Order().ToDictionary(path => path, path => Sha(path)) : new();
var native = new Dictionary<string, string>();
using (var process = Process.GetCurrentProcess())
    foreach (ProcessModule module in process.Modules)
        if (module.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase)) native[module.FileName] = Sha(module.FileName);
var modelFile = new FileInfo(modelPath);
var gguf = typeof(ModelBase).GetField("_gguf", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(model) as GgufFile;
// Use paths of the already-open model. Do not open another shard set or scan
// checkpoint payloads while measuring a memory-constrained model.
string[] modelPaths = gguf?.FilePaths.Select(Path.GetFullPath).ToArray() ?? new[] { modelPath };
var modelFiles = modelPaths.Select(path => new FileInfo(path)).Select(file => new
{
    path = file.FullName, bytes = file.Length, last_write_utc = file.LastWriteTimeUtc,
}).ToArray();
int? declaredModelFileCount = gguf is null ? null : checked((int)gguf.GetUint32("split.count", 1));
bool modelFileIdentityIncomplete = gguf is null || declaredModelFileCount < 1
    || modelFiles.Length != declaredModelFileCount || modelPaths.Distinct(StringComparer.Ordinal).Count() != modelPaths.Length;
long? modelTotalFileBytes = modelFileIdentityIncomplete ? null : modelFiles.Sum(file => file.bytes);
var sourceDimensions = gguf?.Metadata.Where(pair => pair.Key.StartsWith(model.Config.Architecture + ".", StringComparison.Ordinal)
    && pair.Value is not string && (pair.Value is not Array array || array.Length <= 128))
    .OrderBy(pair => pair.Key).ToDictionary(pair => pair.Key, pair => pair.Value);
string architecture = model.Config.Architecture;
var dimensions = new { model.Config.NumLayers, model.Config.HiddenSize, model.Config.NumHeads, model.Config.NumKVHeads,
    model.Config.KeyLength, model.Config.ValueLength, model.Config.VocabSize, model.Config.IntermediateSize,
    model.Config.NumExperts, model.Config.NumExpertsUsed };
// Explicit host-quota runs must prove release before writing a passing report.
// The ordinary mode retains end-of-scope model disposal after report writing.
if (ggmlHostStagingBudget.HasValue) lifetime.Dispose();
bool passedAll = repeatsExact && (!runQuality || qualityPassed) && hostStagingCleanupPassed;
var report = new { schema_version = 1, passed = passedAll, synthetic, backend = backend.ToString(), architecture,
    model_path = modelPath, model_bytes = modelFile.Length, model_last_write_utc = modelFile.LastWriteTimeUtc,
    model_bytes_scope = "Selected input file only; use model_total_file_bytes for a complete checkpoint when model_file_identity_incomplete is false.",
    model_files = modelFiles, model_file_count = modelFiles.Length, model_declared_file_count = declaredModelFileCount,
    model_total_file_bytes = modelTotalFileBytes, model_file_identity_incomplete = modelFileIdentityIncomplete,
    model_files_source = gguf is null ? "Selected input file fallback; actual loaded paths unavailable." : "Already-open GgufFile.FilePaths.",
    model_file_identity_limitation = modelFileIdentityIncomplete
        ? "Actual loaded paths are unavailable or do not cover the declared split count; total checkpoint bytes are unknown. Supply independently verified shard provenance."
        : "Every loaded shard path, size and timestamp is recorded; no full weight payload checksums were computed.",
    synthetic_model_sha256 = synthetic ? Sha(modelPath) : null,
    model_identity_limitation = synthetic ? "Deterministic generated fixture has a full SHA-256." : "Path, size and timestamp identify the selected file; no full checkpoint checksum was computed.",
    dimensions, source_dimensions = sourceDimensions,
    max_context = maxContext, moe_cpu_layers_requested = offload, model_load_ms = loadMs, prompt_tokens = prompt, forced_tokens = forced,
    repeated_final_logits_exact = repeatsExact,
    repeat_equality_scope = "All measured iterations and the separate untimed capture; warmup iterations are reported independently.",
    cold_final_logit_sha256 = coldHash,
    warmup_final_logits_match_measured = warmupHashes.Count == 0 ? (bool?)null : warmupHashes.All(hash => hash == repeatedHash),
    warmup_final_logit_sha256 = warmupHashes,
    all_including_warmup_final_logits_exact = repeatsExact && warmupHashes.All(hash => hash == repeatedHash), runs,
    logits = new { path = logitsPath, sha256 = Sha(logitsPath), rows = 1 + forced.Length, columns = vocabulary, format = "little-endian-float32", final_logit_sha256 = captureHash },
    quality_checked = runQuality, quality_passed = runQuality ? (bool?)qualityPassed : null,
    quality_parser = qualityParser, quality_cases_requested = qualityCases, quality_max_new = maxNew,
    quality_semantic_policy = QualityOutput.SemanticPolicy,
    quality_reasoning_effort_requested = qualityReasoningEffort,
    quality, memory, moe_scratch = scratch, fusion,
    ggml_host_staging_budget = hostStagingReport,
    ggml_device_copy_budget = new { requested_bytes = ggmlDeviceCopyBudget,
        before = ggmlCacheBeforeBudget, after = ggmlCacheAfterBudget,
        scope = "Probe-only rank-0 cache policy applied after model construction and before Forward. Zero is unlimited; one byte prevents new lazy copies. Existing copies are not evicted. Explicit preloads, graph/KV buffers, backend pools and driver allocations remain outside this cap." },
    gptoss_per_expert_prefill = new { requested = gptossPerExpertPrefill,
        verified_prefill_calls = gptossObservedAssignmentEnds.Count,
        expected_assignments_per_call = gptossPerExpertPrefill ? (int?)gptossExpectedAssignments : null,
        observed_assignment_ends = gptossObservedAssignmentEnds,
        evidence_scope = "Probe-only whole-model prefill refusal, exact N>1024 and sentinel-cleared per-expert routing scratch checked after every timed/untimed prefill. Decode retains ordinary dispatch; this is not a production feature flag or kernel profiler." },
    managed_assemblies_sha256 = assemblies, ptx_sha256 = ptx, native_sha256 = native,
    environment = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
        .Where(e => e.Key.ToString()!.StartsWith("TS_CUDA_", StringComparison.Ordinal) || e.Key.ToString()!.StartsWith("TS_Q4E_", StringComparison.Ordinal)
            || e.Key.ToString()!.StartsWith("TS_GPTOSS_", StringComparison.Ordinal)
            || e.Key.ToString() is "TENSORSHARP_CUDA_MOE_FUSION" or "TS_WEIGHT_FUSION_COPIES" or "TS_PREFILL_CHUNK")
        .OrderBy(e => e.Key.ToString()).ToDictionary(e => e.Key.ToString()!, e => e.Value?.ToString()),
    limitations = new[] { "Teacher-forced timing exercises the full loaded model, excluding HTTP, tokenization, sampling and semantic validation.",
        "Synthetic checkpoints are seeded engine fixtures; their execution does not establish trained-model quality or throughput.",
        "Semantic checks cover only the selected fixed arithmetic, extraction, square-list, pure-function or raw JSON tasks; they do not establish broad language or agent quality.",
        "CUDA memory snapshots are driver-wide used/free dedicated memory and pool-cache bytes, not per-process peak live allocation accounting; WDDM and other processes affect them.",
        "OS working set and private bytes are snapshots plus OS process peak; model weights may be memory mapped.",
        "Binary hashes identify loaded assemblies, available PTX files and loaded native library, but source/build provenance must be supplied by the invoking runner." } };
File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
Console.WriteLine("report=" + output);
return passedAll ? 0 : 1;

static void CheckLogits(float[] values)
{
    if (values.Length == 0 || values.Any(v => !float.IsFinite(v))) throw new InvalidOperationException("Empty/nonfinite logits.");
}
static string Sha(string path)
{
    using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
        1 << 20, FileOptions.SequentialScan);
    return Convert.ToHexString(SHA256.HashData(input)).ToLowerInvariant();
}
static string HashLogits(float[] values)
{
    byte[] bytes = new byte[values.Length * sizeof(float)];
    Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
    return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
}
static IEnumerable<string> SpeculativeNames() => TensorSharp.Runtime.Speculative.SpeculationEnvVars.RemovedNames.Select(pair => pair.Name)
    .Concat(new[] { "TS_SPEC", "TS_SPEC_TYPE", "TS_SPEC_DRAFT", "TS_SPEC_PMIN", "TS_SPEC_DRAFT_MODEL" });
