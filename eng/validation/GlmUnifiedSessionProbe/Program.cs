using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace GlmUnifiedSessionProbe;

internal static class Program
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true,
        Converters = { new JsonStringEnumConverter() } };
    private static readonly string[] EnvironmentKeys = ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC",
        "GGML_N_THREADS", "TS_CPU_MOE_THREADS", "TS_GLM_LOAD_THREADS", "TS_GLM_NATIVE", "TS_HOST_MOE_FILE_READ",
        "TS_N_CPU_MOE", "TS_CPU_MOE", "TS_SPEC", "TS_SPEC_TYPE", "TS_SPEC_DRAFT", "TS_SPEC_DRAFT_MODEL",
        "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING",
        "MAX_CONTEXT", "KV_CACHE_DTYPE", "TS_GLM_FORCE_FAIL_STEP", "TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS",
        "TS_HOST_MOE_EXPERT_CACHE_TIMING", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS", "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS"];
    private static string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
    private static object Identity(string path) => new { Path = Path.GetFullPath(path), Bytes = new FileInfo(path).Length, Sha256 = Hash(path) };
    private static object Stat(string path) => new { Path = Path.GetFullPath(path), Bytes = new FileInfo(path).Length,
        LastWriteTimeUtcTicks = File.GetLastWriteTimeUtc(path).Ticks };
    private static object Error(Exception e) => new { Type = e.GetType().FullName, e.Message, Detail = e.ToString() };
    private static bool SamePath(string a, string b) => string.Equals(Path.GetFullPath(a), Path.GetFullPath(b),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    private static void Save(string path, object value) => File.WriteAllText(path, JsonSerializer.Serialize(value, Json) + "\n");
    private static object Ledger(MemoryBudget budget) => new { Current = budget.Snapshot(), HighWater = budget.HighWatermarks() };
    private static object Physical(bool cuda = true)
    {
        using var process = Process.GetCurrentProcess(); process.Refresh();
        return new { Utc = DateTime.UtcNow, process.WorkingSet64, process.PrivateMemorySize64,
            process.PeakWorkingSet64, ManagedObservedBytes = GC.GetTotalMemory(false),
            Hardware = cuda ? (object)InferenceHardwareMemory.CaptureCuda() : null,
            Scope = "Observed process/host/whole-device memory, not complete physical ownership or an OS hard cap." };
    }
    private static object[] LoadedNative() => Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
        .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
        .Select(m => Identity(m.FileName)).ToArray();
    private static string[] Shards(string first)
    {
        Match split = Regex.Match(first, @"-00001-of-([0-9]{5})\.gguf\z", RegexOptions.CultureInvariant);
        if (!split.Success) return [first];
        int count = int.Parse(split.Groups[1].Value, CultureInfo.InvariantCulture);
        if (count is < 1 or > 100) throw new ArgumentException("Unsupported split count.");
        return Enumerable.Range(1, count).Select(i => first[..split.Index] + $"-{i:D5}-of-{count:D5}.gguf").ToArray();
    }

    private static async Task<int> Main(string[] args)
    {
        if (args.SequenceEqual(new[] { "--self-test" })) { ProbePolicy.SelfTest(); CacheTelemetryPolicy.SelfTest(); return 0; }
        Dictionary<string, string> arguments = new(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (!args[i].StartsWith("--") || i + 1 == args.Length || !arguments.TryAdd(args[i][2..], args[i + 1]))
                throw new ArgumentException("Require unique --name value pairs, or --self-test.");
        }
        string[] allowed = ["output", "model", "tokens-file", "context", "max-new", "prefill", "expert-prefill-rows",
            "host-bytes", "device-bytes", "expert-bytes", "selected", "deadline-seconds", "cpu-threads", "task",
            "model-identity", "native-sha256", "repetitions", "cache-telemetry", "cache-token-snapshots", "adaptive-prefill", "capture-final-logits", "capture-token-id"];
        if (arguments.Keys.Except(allowed).Any()) throw new ArgumentException("Unknown option.");
        string output = Path.GetFullPath(arguments["output"]);
        if (Directory.Exists(output) || File.Exists(output)) throw new ArgumentException("Fresh output directory required.");
        Directory.CreateDirectory(output);
        Dictionary<string, object?> report = new() { ["SchemaVersion"] = 2, ["Status"] = "starting",
            ["Passed"] = false, ["SemanticPassed"] = null, ["Arguments"] = arguments,
            ["StartedUtc"] = DateTime.UtcNow, ["ProcessId"] = Environment.ProcessId,
            ["ProcessCommand"] = Environment.GetCommandLineArgs(), ["Path"] = "public UnifiedMemoryModelSession.Create -> ConfigureScheduler -> InferenceEngine.SubmitRequest",
            ["Limits"] = new[] { "Serial fresh requests, plain AR, no images, no MTP/draft, no prefix reuse, no output repair or trimming.",
                "Request cancellation deadline is finite. Synchronous native construction/dispatch cannot be forcibly interrupted safely; use an external owned-process watchdog for a hard process deadline.",
                "Engine Dispose can wait60s. If it fails, never dispose session/model beneath its live worker; outstanding ledger is reported as retained.",
                "Attribution measures Forward durations; wall/TTFT also include scheduler and host overhead. No fair Maya throughput ratio is inferred.",
                "Native crashes cannot execute finally. execution-started.json is checkpoint evidence, not a successful completion or cleanup receipt." } };
        IUnifiedMemoryModelSession? session = null; InferenceEngine? engine = null; MemoryBudget? budget = null;
        SequenceState? sequence = null; InferenceRequestHandle? handle = null; InferenceCompletion? completion = null;
        ITokenizer? tokenizer = null;
        ComputeGate? telemetryGate = null;
        CancellationTokenSource? cancellation = null; List<int> stream = []; List<object> errors = [];
        List<Dictionary<string, object?>> rounds = []; List<object> cacheSnapshots = []; List<object> publicationObservations = [];
        bool engineDisposed = false, sessionDisposed = false, completed = false, cudaObserved = false;
        string[] modelFiles = []; Dictionary<string, object>? beforeStats = null;
        string[] managedPaths = [], toolPaths = []; string? intendedNativePath = null, intendedNativeSha = null;
        string? inputPath = null, inputSha = null, manifestPath = null, manifestSha = null;
        Stopwatch total = Stopwatch.StartNew(), requestWall = new(); double? firstStreamMs = null, firstProbeTokenMs = null;
        Exception? failure = null;
        bool cacheTelemetry = false, cacheTokenSnapshots = false; int repetitions = 1;
        try
        {
            int Integer(string name, int fallback, int min = 1, int max = int.MaxValue)
            {
                string raw = arguments.GetValueOrDefault(name, fallback.ToString(CultureInfo.InvariantCulture));
                if (!int.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out int n) || n < min || n > max)
                    throw new ArgumentException("Invalid " + name); return n;
            }
            long Bytes(string name, long fallback, long min)
            {
                string raw = arguments.GetValueOrDefault(name, fallback.ToString(CultureInfo.InvariantCulture));
                if (!long.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out long n) || n < min)
                    throw new ArgumentException("Invalid " + name); return n;
            }
            string selectedText = arguments.GetValueOrDefault("selected", "false");
            if (selectedText is not ("true" or "false")) throw new ArgumentException("selected must be true or false.");
            bool adaptivePrefill = ProbePolicy.AdaptivePrefill(arguments.GetValueOrDefault("adaptive-prefill", "false"), selectedText == "true");
            bool captureFinalLogits = CacheTelemetryPolicy.ParseBoolean(arguments.GetValueOrDefault("capture-final-logits", "false"));
            report["FinalModelLogitsCaptureEnabled"] = captureFinalLogits;
            string task = arguments.GetValueOrDefault("task", "math42");
            if (task is not ("math42" or "unknown")) throw new ArgumentException("task must be math42 or unknown.");
            repetitions = CacheTelemetryPolicy.ParseRepetitions(arguments.GetValueOrDefault("repetitions", "1"));
            cacheTelemetry = CacheTelemetryPolicy.ParseBoolean(arguments.GetValueOrDefault("cache-telemetry", "false"));
            cacheTokenSnapshots = CacheTelemetryPolicy.ParseBoolean(arguments.GetValueOrDefault("cache-token-snapshots", "false"));
            if (cacheTokenSnapshots && !cacheTelemetry) throw new ArgumentException("cache-token-snapshots requires cache-telemetry true.");
            report["RequestedRepetitions"] = repetitions;
            report["RepetitionPolicy"] = "One model/session/engine, serial requests with unique IDs, identical original raw prompt, no prefix or retained conversation. Round1 is the first request in this process; rounds2+ retain weight caches and OS/vendor state. No OS-cache purge or physical cold-state claim.";
            report["CacheTelemetryEnabled"] = cacheTelemetry;
            report["CacheTokenSnapshotsEnabled"] = cacheTokenSnapshots;
            report["TelemetryTimingScope"] = cacheTokenSnapshots
                ? "Instrumented correctness/diagnostic run, unsuitable for quiet throughput comparison. Publication observations close ComputeGate and drain GpuComputeLock; one already-started step may finish. Reported computed/output/step counts, not consumer token ordinal, identify the actual boundary. No exact prefill-only observation is available."
                : cacheTelemetry ? "Cache observations only at quiescent boundaries outside the measured request window. No token-level gate pauses. Completion wall/publication timings are consumer observations, not independently measured device execution; no exact prefill-only cache observation is available."
                : "No token-level gating or cache reads. Completion wall/publication timings are still consumer observations, not independently measured device execution.";
            report["Rounds"] = rounds; report["CacheSnapshots"] = cacheSnapshots;
            int context = Integer("context", 1024), maxNew = Integer("max-new", 128),
                prefill = Integer("prefill", 32, 1, adaptivePrefill ? 1024 : 512);
            int? captureToken = ProbePolicy.CaptureToken(arguments.GetValueOrDefault("capture-token-id"), captureFinalLogits, maxNew, task);
            report["DiagnosticForcedCaptureToken"] = captureToken;
            report["DiagnosticForcedCaptureScope"] = captureToken.HasValue
                ? "Validation only: a single allowed output token completes identical teacher conditioning; no sampled-answer semantic/EOS or throughput acceptance."
                : "No forced output token; ordinary greedy generation.";
            int threads = Integer("cpu-threads", 8, 1, 1024), deadline = Integer("deadline-seconds", 1200, 1, 86400);
            string modelPath = Path.GetFullPath(arguments.GetValueOrDefault("model",
                "E:/models/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf"));
            string tokenPath = Path.GetFullPath(arguments.GetValueOrDefault("tokens-file",
                "artifacts/default-readiness-20261010/external-baseline-fairness-readonly-v2/glm-math-prompt.ids"));
            if (new FileInfo(tokenPath).Length > 4_194_304) throw new ArgumentException("Token file too large.");
            int[] prompt = ProbePolicy.ParseTokens(File.ReadAllText(tokenPath));
            if ((long)prompt.Length + maxNew > context) throw new ArgumentException("Prompt plus max-new exceeds context.");
            modelFiles = Shards(modelPath);
            beforeStats = modelFiles.ToDictionary(p => p, Stat);
            report["ModelStatsBefore"] = beforeStats;
            report["ModelContentIdentityScope"] = "Size/mtime observations only unless linked to --model-identity historical full hashes. No complete model reread during this probe.";
            string manifest = arguments.GetValueOrDefault("model-identity",
                "artifacts/default-readiness-20261010/glm-e-identity-v1.json");
            if (manifest != "none")
            {
                manifestPath = Path.GetFullPath(manifest); manifestSha = Hash(manifestPath);
                using JsonDocument identity = JsonDocument.Parse(File.ReadAllText(manifest));
                var files = identity.RootElement.GetProperty("files").EnumerateArray().ToArray();
                foreach (string path in modelFiles)
                {
                    var matches = files.Where(r => SamePath(r.GetProperty("path").GetString()!, path)).ToArray();
                    if (matches.Length != 1 || matches[0].GetProperty("bytes").GetInt64() != new FileInfo(path).Length
                        || matches[0].GetProperty("mtime_ns").GetInt64() != checked((File.GetLastWriteTimeUtc(path).Ticks - DateTime.UnixEpoch.Ticks) * 100)
                        || !Regex.IsMatch(matches[0].GetProperty("sha256").GetString() ?? "", @"\A[0-9a-fA-F]{64}\z"))
                        throw new InvalidDataException("Historical model identity size/mtime mismatch: " + path);
                }
                report["HistoricalModelIdentity"] = Identity(manifest); report["HistoricalModelManifest"] = identity.RootElement.Clone();
                report["ModelIdentityLinked"] = true;
            }
            else report["ModelIdentityLinked"] = false;
            report["InputFile"] = Identity(tokenPath); report["PromptTokens"] = prompt;
            inputPath = tokenPath; inputSha = Hash(tokenPath);
            string nativePath = Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "GgmlOps.dll" :
                OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so");
            intendedNativePath = nativePath; intendedNativeSha = Hash(nativePath);
            report["IntendedNative"] = Identity(nativePath);
            if (arguments.TryGetValue("native-sha256", out string? expected) && !string.Equals(expected, Hash(nativePath), StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Pinned production native SHA differs.");
            managedPaths = Directory.EnumerateFiles(AppContext.BaseDirectory, "TensorSharp*.dll")
                .Append(typeof(Program).Assembly.Location).Distinct().Order(StringComparer.Ordinal).ToArray();
            report["ManagedBefore"] = managedPaths.Select(Identity).ToArray();
            toolPaths = Directory.Exists(Path.Combine("eng", "validation", "GlmUnifiedSessionProbe"))
                ? Directory.EnumerateFiles(Path.Combine("eng", "validation", "GlmUnifiedSessionProbe"))
                    .Where(p => p.EndsWith(".cs") || p.EndsWith(".csproj")).ToArray() : [];
            report["SourceToolsBefore"] = toolPaths.Select(Identity).ToArray();
            report["EnvironmentBefore"] = EnvironmentKeys.ToDictionary(k => k, Environment.GetEnvironmentVariable);
            if (SpeculationOptions.FromEnvironment().Enabled || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_SPEC_DRAFT_MODEL")))
                throw new NotSupportedException("Inherited speculation/draft is enabled; do not silently change it.");
            MoeCpuOffloadConfig.ConfigureFromEnvironment();
            if (MoeCpuOffloadConfig.IsExplicitlySet) throw new NotSupportedException("Inherited CPU expert placement override is incompatible with unified planning.");
            MoeCpuOffloadConfig.SetCpuThreads(threads);
            report["EffectiveCpuThreads"] = MoeCpuOffloadConfig.CpuThreads;
            report["EnvironmentEffective"] = EnvironmentKeys.ToDictionary(k => k, Environment.GetEnvironmentVariable);
            report["PhysicalBeforeLoad"] = Physical(); cudaObserved = true;
            UnifiedMemoryModelOptions options = new() { ContextTokens = context, PrefillChunkTokens = Math.Min(prefill, 512),
                ExpertCachePrefillRows = Integer("expert-prefill-rows", 32, 1, 128), MaximumRunningRequests = 1,
                PrefixCheckpointCount = 0, RetainedConversationCount = 0,
                MaximumHostBytes = Bytes("host-bytes", long.MaxValue, 1), MaximumDeviceBytes = Bytes("device-bytes", long.MaxValue, 1),
                MaximumExpertCacheBytes = Bytes("expert-bytes", long.MaxValue, 0), EnableGlmSelectedExperts = selectedText == "true",
                EnableAdaptiveGlmPrefill = adaptivePrefill,
                AdaptiveGlmPrefillChunkTokens = adaptivePrefill ? prefill : null,
                EnableAsyncExpertPrefill = false, EnableExpertCacheCapacityGrowth = false };
            report["SessionOptions"] = options;
            Save(Path.Combine(output, "execution-started.json"), report);
            cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(deadline));
            report["PreparationBeforeCreateMs"] = total.Elapsed.TotalMilliseconds;
            Stopwatch load = Stopwatch.StartNew();
            session = UnifiedMemoryModelSession.Create(modelPath, null, BackendType.GgmlCuda, options);
            load.Stop(); report["ModelLoadMs"] = load.Elapsed.TotalMilliseconds;
            budget = session.Budget; report["BudgetAfterLoad"] = Ledger(budget);
            report["EffectiveContextTokens"] = session.ContextTokens; report["EffectivePrefillChunkTokens"] = session.PrefillChunkTokens;
            report["FeatureReport"] = session.FeatureReport; report["PhysicalAfterLoad"] = Physical();
            ModelBase model = session.Model ?? throw new InvalidOperationException("Session returned no model.");
            tokenizer = model.Tokenizer;
            if (model.Config.Architecture is not ("glm5next" or "glm5-next")) throw new InvalidOperationException("Require GLM5Next architecture.");
            if (prompt.Any(t => t >= model.Tokenizer.VocabSize)) throw new InvalidDataException("Prompt exceeds vocabulary.");
            if (captureToken.HasValue && (captureToken.Value >= model.Tokenizer.VocabSize || model.Tokenizer.EosTokenIds.Contains(captureToken.Value)))
                throw new ArgumentException("Diagnostic capture token must be a non-EOS checkpoint token.");
            string promptText = model.Tokenizer.Decode(prompt.ToList()); report["DecodedPrompt"] = promptText;
            report["MathPromptMatches"] = ProbePolicy.MathPromptMatches(promptText);
            report["CheckpointEosIds"] = model.Tokenizer.EosTokenIds;
            report["LoadedNativeAfterLoad"] = LoadedNative();
            if (!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => SamePath(m.FileName, nativePath)))
                throw new InvalidOperationException("Intended deployment native was not observed loaded.");
            SchedulerConfig config = session.ConfigureScheduler(new SchedulerConfig { EnablePrefixCaching = false,
                StopRepetition = false, Speculation = SpeculationOptions.Disabled, BlockSize = 16 });
            if (config.PrefillChunkTokenLimit != session.PrefillChunkTokens || config.MemoryAdmission?.Budget != budget)
                throw new InvalidOperationException("Scheduler lost the session's effective shape/shared admission.");
            report["Scheduler"] = new { config.PrefillChunkTokenLimit, config.MaxPrefillChunkSize, config.SoloPrefillChunkSize,
                config.MaxNumBatchedTokens, config.MaxNumRunningSequences, config.EnablePrefixCaching, config.StopRepetition,
                SharedAdmission = config.MemoryAdmission != null, config.Speculation.Enabled };
            engine = new InferenceEngine(model, config);
            if (cacheTelemetry) { telemetryGate = new ComputeGate(); engine.ComputeGate = telemetryGate; }
            void ObserveCache(string stage, int round, int? publishedToken = null)
            {
                if (!cacheTelemetry) return;
                Stopwatch observation = Stopwatch.StartNew();
                telemetryGate!.Close();
                try
                {
                    lock (model.GpuComputeLock)
                    {
                        cacheSnapshots.Add(new { Stage = stage, Round = round, Utc = DateTime.UtcNow,
                            ConsumerPublishedCount = stream.Count, ConsumerPublishedToken = publishedToken,
                            ComputedTokens = sequence?.NumComputedTokens, OutputCount = sequence?.OutputTokens.Count,
                            SequenceStatus = sequence?.Status, engine.TotalStepsRun, engine.TotalCompleted,
                            engine.RunningCount, engine.WaitingCount, engine.PendingRequestCount,
                            Cache = CacheTelemetryPolicy.Read(), Budget = Ledger(budget) });
                    }
                }
                finally { telemetryGate.Open(); observation.Stop(); }
                publicationObservations.Add(new { Stage = stage, Round = round,
                    ConsumerPublishedCount = stream.Count, ObservationPauseWallMs = observation.Elapsed.TotalMilliseconds,
                    GateClosures = telemetryGate.Closures, GateClosedWallMs = telemetryGate.TotalClosed.TotalMilliseconds });
                File.AppendAllText(Path.Combine(output, "cache-snapshots.jsonl"), JsonSerializer.Serialize(cacheSnapshots[^1]) + "\n");
            }
            ObserveCache("after-load-before-first-request", 0);
            for (int round = 1; round <= repetitions; round++)
            {
            CacheTelemetryPolicy.BeginRound(report);
            completed = false; completion = null; handle = null;
            stream = []; publicationObservations = []; requestWall.Reset();
            firstStreamMs = null; firstProbeTokenMs = null;
            Dictionary<string, object?> roundReport = new() { ["Round"] = round,
                ["CacheStateLabel"] = round == 1 ? "first-request-in-fresh-process; physical cold state unqualified" : "warm-weight/OS/vendor-state; fresh request without prefix reuse",
                ["StartedUtc"] = DateTime.UtcNow, ["PublicationObservations"] = publicationObservations,
                ["Status"] = "starting", ["Passed"] = false };
            rounds.Add(roundReport); report["ActiveRound"] = round;
            var sampling = SamplingConfig.Greedy;
            if (captureToken.HasValue) sampling.FirstTokenAllowList = [captureToken.Value];
            sequence = new SequenceState("glm-unified-" + Guid.NewGuid().ToString("N"), prompt, maxNew, engine.PoolStats.blockSize,
                sampling, cacheBreakpoints: Array.Empty<int>());
            report["RequestId"] = sequence.RequestId;
            roundReport["RequestId"] = sequence.RequestId;
            roundReport["BudgetBeforeSubmit"] = Ledger(budget);
            roundReport["EngineBeforeSubmit"] = new { engine.TotalSubmitted, engine.TotalCompleted,
                engine.TotalStepsRun, TotalForwardMs = engine.TotalForwardTime.TotalMilliseconds };
            ObserveCache("before-submit", round);
            cancellation.Token.ThrowIfCancellationRequested();
            requestWall.Start(); handle = engine.SubmitRequest(sequence, cancellation.Token);
            Save(Path.Combine(output, "execution-started.json"), report);
            await foreach (int token in handle.Tokens.ReadAllAsync(cancellation.Token))
            {
                firstStreamMs ??= requestWall.Elapsed.TotalMilliseconds;
                firstProbeTokenMs ??= total.Elapsed.TotalMilliseconds; stream.Add(token);
                publicationObservations.Add(new { Stage = "publication-before-optional-pause", Round = round,
                    Token = token, PublishedCount = stream.Count, RequestWallMs = requestWall.Elapsed.TotalMilliseconds,
                    Utc = DateTime.UtcNow });
                if (cacheTokenSnapshots) ObserveCache("after-publication", round, token);
                if (stream.Count % 16 == 0) Save(Path.Combine(output, $"round-{round:D2}-partial-stream-tokens.json"), stream);
            }
            completion = await handle.Completion.WaitAsync(cancellation.Token);
            requestWall.Stop(); completed = true;
            if (captureFinalLogits)
            {
                if (captureToken.HasValue && !sequence.OutputTokens.SequenceEqual(new[] { captureToken.Value }))
                    throw new InvalidDataException("Diagnostic capture did not consume the exact forced token.");
                // Validation-only snapshot at a quiescent single-request boundary.
                // This is after forwarding the terminal sampled token (P+O),
                // not the prefill-only prediction or SequenceState.LastLogits.
                float[] captured;
                lock (model.GpuComputeLock)
                {
                    var field = typeof(ModelBase).GetField("_logitsBuffer", BindingFlags.Instance | BindingFlags.NonPublic)
                        ?? throw new MissingFieldException("ModelBase logits diagnostic field is unavailable.");
                    captured = FinalLogitCapture.Copy(field.GetValue(model) as float[], model.Config.VocabSize);
                }
                string capturePath = Path.Combine(output, $"round-{round:D2}-final-logits.f32");
                using (var writer = new BinaryWriter(new FileStream(capturePath, FileMode.CreateNew, FileAccess.Write)))
                    foreach (float value in captured) writer.Write(value);
                report["FinalModelLogitsCapture"] = new { File = Identity(capturePath), Elements = captured.Length,
                    AllFinite = true, ComputedTokens = sequence.NumComputedTokens,
                    OutputTokens = sequence.OutputTokens.ToArray(),
                    Scope = "Validation-only copied GLM model buffer after the terminal sampled token forward, under GpuComputeLock. Compare only identical full prompt and consumed output history. Outside request timing; no teacher or independent arithmetic oracle." };
            }
            report["Completion"] = completion; report["BudgetAfterCompletion"] = Ledger(budget);
            report["PhysicalAfterCompletion"] = Physical();
            ObserveCache("after-completion", round);
            int[] tokens = sequence.OutputTokens.ToArray();
            bool natural = ProbePolicy.NaturalEos(tokens, model.Tokenizer.EosTokenIds, completion.FinishReason);
            string raw = model.Tokenizer.Decode(tokens.ToList());
            var parser = OutputParserFactory.Create(model.Config.Architecture); parser.Init(false, null);
            ParsedOutput parsed = parser.Add(model.Tokenizer.Decode(tokens.Take(tokens.Length - (natural ? 1 : 0)).ToList()), true);
            bool thinkingClosed = !promptText.EndsWith("<think>", StringComparison.Ordinal) || raw.Contains("</think>", StringComparison.Ordinal);
            bool structural = completion.Status == SequenceStatus.FinishedStopped && natural && sequence.Error == null
                && tokens.Length <= maxNew && stream.SequenceEqual(tokens.Take(tokens.Length - 1))
                && completion.OutputTokenCount == tokens.Length && completion.PromptTokenCount == prompt.Length
                && sequence.NumComputedTokens == prompt.Length + tokens.Length && completion.PrefixCacheReusedTokens == 0;
            bool? semantic = task == "math42" ? ProbePolicy.MathPromptMatches(promptText)
                && structural && ProbePolicy.StrictMath(parsed.Content, parsed.ToolCalls?.Count ?? 0, thinkingClosed) : null;
            report["RawText"] = raw; report["Content"] = parsed.Content; report["Thinking"] = parsed.Thinking;
            report["ToolCalls"] = parsed.ToolCalls; report["ThinkingClosed"] = thinkingClosed;
            report["OutputTokens"] = tokens; report["PublishedTokens"] = stream.ToArray();
            report["ComputedTokens"] = sequence.NumComputedTokens; report["StatusBeforeDispose"] = sequence.Status;
            report["RequestAdmittedPrefillChunkTokens"] = config.MemoryAdmission?.PrefillTokenLimit?.Invoke(sequence);
            report["PrefillShapeEvidenceScope"] = "Session ceiling and request-admitted limit are size promises; actual forward widths require execution trace. Engine step/output counts supply separate consumed-work evidence.";
            report["ComputedTokenConvention"] = "Non-speculative prompt-plus-all-sampled-outputs-forwarded (P+O). Includes terminal sampled EOS.";
            report["NaturalEos"] = natural; report["StructuralPassed"] = structural; report["SemanticPassed"] = semantic;
            report["SemanticPolicy"] = "Exact production-parsed content42 for the independently specified17+25 prompt; no trimming/repair. Other tasks unknown.";
            report["FinalLogitsFinite"] = sequence.LastLogits is { Length: > 0 } logits ? logits.All(float.IsFinite) : (bool?)null;
            if (report["FinalLogitsFinite"] is false) throw new InvalidDataException("Final host logits contain nonfinite values.");
            report["Timing"] = new { RequestWallMs = requestWall.Elapsed.TotalMilliseconds, FirstPublishedTokenWallMs = firstStreamMs,
                ProcessToFirstPublishedTokenMs = firstProbeTokenMs,
                EngineFirstTokenMs = completion.FirstTokenAt is DateTime t ? (double?)(t - completion.SubmittedAt).TotalMilliseconds : null,
                PrefillAttributedMs = completion.PrefillElapsedTicks * 1000.0 / Stopwatch.Frequency,
                DecodeAttributedMs = completion.DecodeElapsedTicks * 1000.0 / Stopwatch.Frequency,
                PromptTokens = prompt.Length, EngineOutputTokens = tokens.Length, PublishedTokens = stream.Count,
                OutputForwardCount = sequence.NumComputedTokens - prompt.Length,
                PostFirstPublicationVisibleTokens = Math.Max(0, stream.Count - 1),
                PrefillTokensPerSecond = completion.PrefillElapsedTicks > 0 ? prompt.Length * (double)Stopwatch.Frequency / completion.PrefillElapsedTicks : (double?)null,
                DecodeAttributedTokensPerSecond = completion.DecodeElapsedTicks > 0 ? tokens.Length * (double)Stopwatch.Frequency / completion.DecodeElapsedTicks : (double?)null,
                DecodeWallTokensPerSecond = firstStreamMs.HasValue && requestWall.Elapsed.TotalMilliseconds > firstStreamMs
                    ? Math.Max(0, stream.Count - 1) * 1000.0 / (requestWall.Elapsed.TotalMilliseconds - firstStreamMs.Value) : (double?)null,
                Scope = "Fresh request without prefix reuse; later repetitions retain weight/OS/vendor caches. Attributed decode includes first and terminal output forwards (P+O); Maya leaves its terminal sample unforwarded (P+O-1). Few-token samples do not establish steady decode throughput. No equal-resource/consumed-work throughput ratio qualified." };
            report["Passed"] = structural && semantic == true && report["ModelIdentityLinked"] is true; report["Status"] = semantic is null
                ? "completed-quality-unknown" : semantic == true ? "completed-math-passed" : "completed-math-failed";
            foreach (string key in CacheTelemetryPolicy.RequestResultKeys) roundReport[key] = report.GetValueOrDefault(key);
            roundReport["FinishedUtc"] = DateTime.UtcNow;
            Save(Path.Combine(output, $"round-{round:D2}-report.json"), roundReport);
            Save(Path.Combine(output, "execution-started.json"), report);
            }
            report["CompletedRepetitions"] = rounds.Count(r => r.GetValueOrDefault("Completion") != null);
            report["AllRoundsStructuralPassed"] = CacheTelemetryPolicy.AllRoundsPass(repetitions, rounds.Select(r => r.GetValueOrDefault("StructuralPassed") is true));
            report["AllRoundsSemanticPassed"] = task == "unknown" ? (bool?)null : CacheTelemetryPolicy.AllRoundsPass(repetitions, rounds.Select(r => r.GetValueOrDefault("SemanticPassed") is true));
            report["Passed"] = CacheTelemetryPolicy.AllRoundsPass(repetitions, rounds.Select(r => r.GetValueOrDefault("Passed") is true));
            report["Status"] = report["Passed"] is true ? "completed-all-math-passed"
                : task == "unknown" ? "completed-quality-unknown" : "completed-math-failed";
        }
        catch (Exception e)
        {
            failure = e; errors.Add(Error(e)); report["Status"] = "failed";
            if (e is UnifiedMemoryModelLoadException unreleased) { session = unreleased.UnreleasedSession; budget = session.Budget; }
        }
        finally
        {
            requestWall.Stop();
            report["RequestWallObservedMs"] = sequence == null ? null : requestWall.Elapsed.TotalMilliseconds;
            report["FirstPublishedTokenWallMs"] = firstStreamMs; report["ProcessToFirstPublishedTokenMs"] = firstProbeTokenMs;
            if (engine != null) report["EngineCountersBeforeDispose"] = new { engine.TotalSubmitted, engine.TotalCompleted,
                engine.TotalStepsRun, TotalForwardMs = engine.TotalForwardTime.TotalMilliseconds,
                engine.RunningCount, engine.WaitingCount, engine.PendingRequestCount };
            try { telemetryGate?.Open(); } catch (Exception e) { errors.Add(Error(e)); }
            try { cancellation?.Cancel(); } catch (Exception e) { errors.Add(Error(e)); }
            if (engine != null)
            {
                if (!completed && handle != null) engine.Abort(handle.RequestId);
                try { engine.Dispose(); engineDisposed = true; }
                catch (Exception e) { errors.Add(Error(e)); report["EngineDisposalError"] = Error(e); }
            }
            void Capture(string name, Func<object> read)
            { try { report[name] = read(); } catch (Exception e) { errors.Add(Error(e)); report[name + "Error"] = Error(e); } }
            if (budget != null) Capture("BudgetAfterEngineDisposeAttempt", () => Ledger(budget));
            report["EngineDisposed"] = engineDisposed;
            if (engineDisposed) report["BudgetAfterEngineDisposed"] = report.GetValueOrDefault("BudgetAfterEngineDisposeAttempt");
            if (cacheTelemetry && session != null && (engine == null || engineDisposed)) Capture("CacheAfterEngineDispose", CacheTelemetryPolicy.Read);
            if (session != null && (engine == null || engineDisposed))
            {
                try { session.Dispose(); sessionDisposed = true; }
                catch (Exception e) { errors.Add(Error(e)); report["SessionDisposalError"] = Error(e); }
            }
            else if (session != null) report["SessionDisposalSkipped"] = "Live/quarantined engine ownership: do not free native model underneath worker.";
            report["SessionDisposed"] = sessionDisposed;
            if (cacheTelemetry && sessionDisposed) Capture("CacheAfterSessionDispose", CacheTelemetryPolicy.Read);
            if (budget != null)
            {
                Capture("BudgetAfterSessionDisposeAttempt", () => Ledger(budget));
                if (sessionDisposed) report["BudgetAfterSessionDisposed"] = report.GetValueOrDefault("BudgetAfterSessionDisposeAttempt");
                Capture("LedgerZeroAfterCleanup", () => sessionDisposed && budget.Snapshot().All(p => p.Reserved == 0 && p.Committed == 0));
            }
            else report["LedgerAvailability"] = "Construction returned no public session/budget; do not impute zero ledger from unavailable data.";
            if (sequence != null) { report["OutputTokensAfterCleanup"] = sequence.OutputTokens.ToArray(); report["SequenceStatusAfterCleanup"] = sequence.Status;
                report["FinishReasonAfterCleanup"] = sequence.FinishReason;
                report["SequenceError"] = sequence.Error is Exception se ? Error(se) : null; report["ComputedTokensAfterCleanup"] = sequence.NumComputedTokens; }
            if (sequence != null && tokenizer != null)
            {
                Capture("RawTextAfterCleanup", () => tokenizer.Decode(sequence.OutputTokens.ToList()));
                report["NaturalEosAfterCleanup"] = ProbePolicy.NaturalEos(sequence.OutputTokens.ToArray(), tokenizer.EosTokenIds, sequence.FinishReason);
            }
            report["PublishedTokens"] = stream.ToArray();
            if (rounds.Count > 0 && rounds[^1].GetValueOrDefault("Completion") == null)
            {
                rounds[^1]["Status"] = "incomplete-or-failed";
                rounds[^1]["PublishedTokens"] = stream.ToArray();
                rounds[^1]["OutputTokensAfterCleanup"] = sequence?.OutputTokens.ToArray();
                rounds[^1]["ComputedTokensAfterCleanup"] = sequence?.NumComputedTokens;
                rounds[^1]["RequestWallObservedMs"] = requestWall.Elapsed.TotalMilliseconds;
                rounds[^1]["Error"] = failure == null ? null : Error(failure);
            }
            Capture("LoadedNativeAfter", () => LoadedNative());
            Capture("PhysicalAfterCleanup", () => Physical(cudaObserved));
            if (beforeStats != null)
            {
                Capture("ModelStatsAfter", () => modelFiles.ToDictionary(p => p, Stat));
                report["ModelStatsUnchanged"] = report.TryGetValue("ModelStatsAfter", out var after)
                    && JsonSerializer.Serialize(beforeStats) == JsonSerializer.Serialize(after);
                if (report["ModelStatsUnchanged"] is false) errors.Add(Error(new InvalidDataException("Model size/mtime changed during the run.")));
            }
            if (managedPaths.Length > 0)
            {
                Capture("ManagedAfter", () => managedPaths.Select(Identity).ToArray());
                report["ManagedUnchanged"] = JsonSerializer.Serialize(report.GetValueOrDefault("ManagedBefore"))
                    == JsonSerializer.Serialize(report.GetValueOrDefault("ManagedAfter"));
                if (report["ManagedUnchanged"] is false) errors.Add(Error(new InvalidDataException("Managed binaries changed.")));
            }
            if (toolPaths.Length > 0)
            {
                Capture("SourceToolsAfter", () => toolPaths.Select(Identity).ToArray());
                report["SourceToolsUnchanged"] = JsonSerializer.Serialize(report["SourceToolsBefore"])
                    == JsonSerializer.Serialize(report.GetValueOrDefault("SourceToolsAfter"));
                if (report["SourceToolsUnchanged"] is false) errors.Add(Error(new InvalidDataException("Tool sources changed.")));
            }
            if (intendedNativeSha != null)
            {
                Capture("IntendedNativeAfter", () => Identity(intendedNativePath!));
                try { report["NativeUnchanged"] = Hash(intendedNativePath!) == intendedNativeSha;
                    if (report["NativeUnchanged"] is false) errors.Add(Error(new InvalidDataException("Native binary changed."))); }
                catch (Exception e) { errors.Add(Error(e)); }
            }
            if (inputPath != null && inputSha != null)
            {
                Capture("InputFileAfter", () => Identity(inputPath));
                try { report["InputFileUnchanged"] = Hash(inputPath) == inputSha;
                    if (report["InputFileUnchanged"] is false) errors.Add(Error(new InvalidDataException("Raw input file changed."))); }
                catch (Exception e) { errors.Add(Error(e)); }
            }
            if (manifestPath != null && manifestSha != null)
            {
                Capture("HistoricalModelIdentityAfter", () => Identity(manifestPath));
                try { report["HistoricalModelIdentityUnchanged"] = Hash(manifestPath) == manifestSha;
                    if (report["HistoricalModelIdentityUnchanged"] is false) errors.Add(Error(new InvalidDataException("Historical model identity changed."))); }
                catch (Exception e) { errors.Add(Error(e)); }
            }
            try { cancellation?.Dispose(); } catch (Exception e) { errors.Add(Error(e)); }
            report["EngineCompletionObserved"] = completed;
            report["CompletedRepetitions"] = rounds.Count(r => r.GetValueOrDefault("Completion") != null);
            report["TopLevelRequestFieldsScope"] = "Last started request. Rounds contains every repetition, including an incomplete final request; overall Passed requires all requested rounds to pass and complete cleanup/identity checks.";
            report["EnvironmentAfter"] = EnvironmentKeys.ToDictionary(k => k, Environment.GetEnvironmentVariable);
            report["Errors"] = errors; report["FinishedUtc"] = DateTime.UtcNow; report["TotalProbeWallMs"] = total.Elapsed.TotalMilliseconds;
            if (errors.Count != 0 || (session != null && !sessionDisposed) || (budget != null && report.GetValueOrDefault("LedgerZeroAfterCleanup") is not true))
            { report["Passed"] = false; report["Status"] = "failed"; }
            Save(Path.Combine(output, "report.json"), report);
        }
        Console.WriteLine(JsonSerializer.Serialize(new { Output = output, Status = report["Status"], Passed = report["Passed"] }));
        return failure == null && report["Passed"] is true ? 0 : 1;
    }
}
