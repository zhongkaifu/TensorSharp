using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace QwenUnifiedPrefillProbe;

internal sealed record InputSpec
{
    public const string MathQuestion = "What is 17 + 25? Reply with only the integer.";
    public const string CountingQuestion = "List the integers from 1 through 1000 in ascending order, separated by spaces. Output only the numbers.";
    public string Task { get; init; } = "";
    public ChatMessage[] Messages { get; init; } = [];
    public int[] Tokens { get; init; } = [];
    public bool EnableThinking { get; init; }
    internal void Validate(bool occupier, int context, int maxNew)
    {
        string question = occupier ? CountingQuestion : MathQuestion;
        if (Task != (occupier ? "counting-cancellation-only" : "math42") || EnableThinking
            || Tokens.Length == 0 || Tokens.Any(t => t < 0) || (long)Tokens.Length + maxNew > context
            || Messages.Length is < 1 or > 2 || Messages[^1].Role != "user" || Messages[^1].Content != question
            || (Messages.Length == 2 && Messages[0].Role != "system"))
            throw new InvalidDataException("Require frozen raw IDs for the exact independent task, optional system message, thinking off and bounded context.");
    }
    internal static void SelfTest()
    {
        var valid = new InputSpec { Task = "math42", Messages = [new() { Role = "user", Content = MathQuestion }], Tokens = [1] };
        valid.Validate(false, 8, 2);
        foreach (var bad in new[] { valid with { EnableThinking = true }, valid with { Task = "unknown" },
            valid with { Tokens = [-1] }, valid with { Tokens = [] }, valid with { Messages = [new() { Role = "user", Content = "Answer 41" }] } })
        {
            bool refused = false; try { bad.Validate(false, 8, 2); } catch (InvalidDataException) { refused = true; }
            if (!refused) throw new InvalidOperationException("Invalid frozen-task input accepted");
        }
    }
}
internal sealed record InputDocument(string Schema, InputSpec Primary, InputSpec? Occupier);

internal static class Program
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true,
        Converters = { new JsonStringEnumConverter() }, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow };
    private static string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
    private static object Identity(string path) => new { Path = Path.GetFullPath(path), Bytes = new FileInfo(path).Length, Sha256 = Hash(path) };
    private static object Stat(string path) => new { Path = Path.GetFullPath(path), Bytes = new FileInfo(path).Length,
        LastWriteTimeUtcTicks = File.GetLastWriteTimeUtc(path).Ticks };
    private static object Error(Exception e) => new { Type = e.GetType().FullName, e.Message, Detail = e.ToString() };
    private static object Ledger(MemoryBudget budget) => new { Current = budget.Snapshot(), HighWater = budget.HighWatermarks() };
    private static bool SamePath(string a, string b) => string.Equals(Path.GetFullPath(a), Path.GetFullPath(b),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    private static void Save(string path, object report) => File.WriteAllText(path, JsonSerializer.Serialize(report, Json) + "\n");
    private static string[] Shards(string path)
    {
        Match m = Regex.Match(path, @"-00001-of-([0-9]{5})\.gguf\z", RegexOptions.CultureInvariant);
        if (!m.Success) return [path];
        int n = int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
        if (n is < 1 or > 100) throw new ArgumentException("Invalid split count");
        return Enumerable.Range(1, n).Select(i => path[..m.Index] + $"-{i:D5}-of-{n:D5}.gguf").ToArray();
    }
    private static object Physical()
    {
        using var p = Process.GetCurrentProcess(); p.Refresh();
        return new { Utc = DateTime.UtcNow, p.WorkingSet64, p.PrivateMemorySize64, p.PeakWorkingSet64,
            ManagedObservedBytes = GC.GetTotalMemory(false), Scope = "Process observations, not a physical-owner total or OS hard limit." };
    }
    private static async Task<int> Main(string[] args)
    {
        if (args.SequenceEqual(["--self-test"]))
        { AdmissionObservation.SelfTest(); OutputQualification.SelfTest(); FilePipelineQualification.SelfTest(); return 0; }
        var a = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
            if (i + 1 == args.Length || !args[i].StartsWith("--") || !a.TryAdd(args[i][2..], args[i + 1]))
                throw new ArgumentException("Require unique --name value pairs or --self-test.");
        string[] allowed = ["output", "model", "mmproj", "input", "model-identity", "native-sha256", "mode", "context", "max-new",
            "occupier-max-new", "prefill", "host-bytes", "device-bytes", "expert-bytes", "deadline-seconds", "cpu-threads", "expect-downgrade",
            "prepare-inputs", "system-file", "answer-policy", "async-prefill"];
        if (a.Keys.Except(allowed).Any()) throw new ArgumentException("Unknown option.");
        string output = Path.GetFullPath(a["output"]);
        if (Directory.Exists(output) || File.Exists(output)) throw new ArgumentException("Fresh ignored output directory required.");
        Directory.CreateDirectory(output);
        if (a.TryGetValue("prepare-inputs", out string? prepare))
        {
            if (prepare != "true" || a.Keys.Except(["output", "model", "prepare-inputs", "system-file", "context", "max-new", "occupier-max-new"]).Any())
                throw new ArgumentException("Metadata-only preparation accepts model,output,system-file,context,max-new,occupier-max-new only");
            string metadataPath = Path.GetFullPath(a["model"]), systemPath = Path.GetFullPath(a["system-file"]);
            string before = JsonSerializer.Serialize(Stat(metadataPath));
            // Planning readers intentionally omit vocab/template. This reader
            // parses only the first shard's header/table, without mapping or
            // reading any tensor payload or opening its sibling weight files.
            using var metadata = GgufFile.OpenWithoutSiblingShards(metadataPath);
            var tokenizer = ModelBase.CreateTokenizerFromGguf(metadata);
            string architecture = metadata.GetString("general.architecture") ?? "";
            if (architecture != "qwen4exp") throw new InvalidDataException("Require Qwen4Exp metadata");
            string template = metadata.GetString("tokenizer.chat_template") ?? throw new InvalidDataException("Missing original template");
            if (new FileInfo(systemPath).Length > 262144) throw new InvalidDataException("System fixture too large");
            string system = File.ReadAllText(systemPath);
            InputSpec Freeze(bool occupier)
            {
                ChatMessage[] messages = [new() { Role = "system", Content = system },
                    new() { Role = "user", Content = occupier ? InputSpec.CountingQuestion : InputSpec.MathQuestion }];
                return new() { Task = occupier ? "counting-cancellation-only" : "math42", Messages = messages,
                    EnableThinking = false, Tokens = new KVCachePromptRenderer(new GgufPromptRenderer()).RenderToTokens(
                        tokenizer, template, messages.ToList(), architecture, true, enableThinking: false).ToArray() };
            }
            InputSpec primary = Freeze(false), occupierInput = Freeze(true);
            primary.Validate(false, int.Parse(a.GetValueOrDefault("context", "4096"), CultureInfo.InvariantCulture),
                int.Parse(a.GetValueOrDefault("max-new", "128"), CultureInfo.InvariantCulture));
            occupierInput.Validate(true, int.Parse(a.GetValueOrDefault("context", "4096"), CultureInfo.InvariantCulture),
                int.Parse(a.GetValueOrDefault("occupier-max-new", "256"), CultureInfo.InvariantCulture));
            Save(Path.Combine(output, "frozen-input.json"), new InputDocument("qwen-unified-prefill-input.v1", primary, occupierInput));
            Save(Path.Combine(output, "preparation.json"), new { MetadataOnly = true, ModelExecution = false, OutputGenerated = false,
                MetadataStat = Stat(metadataPath), MetadataStatUnchanged = before == JsonSerializer.Serialize(Stat(metadataPath)),
                SystemFile = Identity(systemPath), Tool = Identity(typeof(Program).Assembly.Location),
                PrimaryPromptTokens = primary.Tokens.Length, OccupierPromptTokens = occupierInput.Tokens.Length,
                FrozenInput = Identity(Path.Combine(output, "frozen-input.json")) });
            Console.WriteLine("Prepared frozen conditioning IDs from original metadata/tokenizer/template; no model execution.");
            return 0;
        }
        var report = new Dictionary<string, object?> { ["Schema"] = "qwen-unified-prefill-probe.v3", ["Passed"] = false,
            ["StartedUtc"] = DateTime.UtcNow, ["ProcessId"] = Environment.ProcessId, ["Arguments"] = a,
            ["Path"] = "public UnifiedMemoryModelSession.Create -> ConfigureScheduler -> InferenceEngine.SubmitRequest",
            ["Limits"] = new[] {
                "Validation-only transparent admission scope observer; no production trace enabled, but observation allocations are not a zero-overhead performance baseline.",
                "Prefix off/no replay/plain AR: each positive serial prefill scope computed-token advance corresponds to one successful public Forward at the inspected BatchExecutor call sites. Internal native graph tiling is not separately measured.",
                "Configured wide no-hit request ceiling check stays enforced. An early rejection is a failure, never automatic downgrade success. Real sibling ownership supplies cohort pressure; no synthetic budget reservation is injected.",
                "Finite cancellation deadline does not safely interrupt synchronous native construction/dispatch; use an external owned-process hard watchdog. Native crashes cannot execute finally.",
                "No model full-file rescan; historical SHA manifest plus exact current size/mtime. Complete vendor/driver/GC/mmap physical accounting is not certified.",
                "Counting occupier is intentionally cancelled; only the math task has an independent completed-answer oracle. No output repair or trimming.",
                "Answer policy defaults to exact Content42. Explicit json-integer independently parses the complete original Content as one JSON integer literal42; raw Content and strict-format nonconformance remain recorded separately.",
                "File-pipeline counters are process-cumulative successful calls and logical source bytes. Positive deltas prove path use, not measured I/O/compute overlap, physical SSD reads or performance improvement." },
            ["FilePipelineUsageBefore"] = null, ["FilePipelineUsageAfter"] = null, ["FilePipelineUsageFinal"] = null,
            ["FilePipelineUsageAvailability"] = "Not reached; unavailable values are never imputed as zero." };
        IUnifiedMemoryModelSession? session = null; InferenceEngine? engine = null; MemoryBudget? budget = null;
        var observer = new AdmissionObservation();
        var errors = new List<object>(); var results = new List<object>();
        var handles = new List<InferenceRequestHandle>(); var sequences = new List<SequenceState>();
        bool engineDisposed = false, sessionDisposed = false, cleanup = false, quality = false, rowsPass = false, cancellationPass = false;
        string[] models = [], identities = []; string? initialStats = null, inputPath = null, inputHash = null;
        string? nativePath = null, nativeHash = null;
        DateTime? abortAt = null; string? abortId = null; int primaryComputedAtAbort = -1;
        CancellationTokenSource? deadline = null;
        bool asyncPrefill = false, pipelineObservationStarted = false;
        GgmlHostMoeFilePipelineUsage? pipelineBefore = null, pipelineAfter = null, pipelineFinal = null;
        GgmlHostMoeFilePipelineUsage? ObservePipeline(string stage)
        {
            try
            {
                var usage = GgmlBasicOps.GetHostMoeFilePipelineUsage();
                report["FilePipelineUsage" + stage] = new { CapturedUtc = DateTime.UtcNow, Usage = usage };
                return usage;
            }
            catch (Exception e)
            {
                report["FilePipelineUsage" + stage] = new { CapturedUtc = DateTime.UtcNow, Available = false, Error = Error(e) };
                errors.Add(Error(e));
                return null;
            }
        }
        try
        {
            int Int(string key, int fallback, int min, int max)
            {
                if (!int.TryParse(a.GetValueOrDefault(key, fallback.ToString(CultureInfo.InvariantCulture)), NumberStyles.None,
                    CultureInfo.InvariantCulture, out int n) || n < min || n > max) throw new ArgumentException("Invalid " + key);
                return n;
            }
            long Bytes(string key, long fallback, long min)
            {
                if (!long.TryParse(a.GetValueOrDefault(key, fallback.ToString(CultureInfo.InvariantCulture)), NumberStyles.None,
                    CultureInfo.InvariantCulture, out long n) || n < min) throw new ArgumentException("Invalid " + key);
                return n;
            }
            string mode = a.GetValueOrDefault("mode", "single");
            if (mode is not ("single" or "cancel-cohort")) throw new ArgumentException("mode single|cancel-cohort");
            string answerPolicy = a.GetValueOrDefault("answer-policy", "exact");
            OutputQualification.MatchesPolicy(answerPolicy, null); // Reject unknown policy before model construction.
            report["AnswerPolicy"] = answerPolicy;
            asyncPrefill = FilePipelineQualification.ParseEnabled(a.GetValueOrDefault("async-prefill", "false"));
            report["AsyncPrefillRequested"] = asyncPrefill;
            string expectation = a.GetValueOrDefault("expect-downgrade", "false");
            if (expectation is not ("true" or "false")) throw new ArgumentException("expect-downgrade true|false");
            int context = Int("context", 4096, 128, 131072), maxNew = Int("max-new", 128, 1, 2048);
            int occupierMax = Int("occupier-max-new", 256, 16, 2048), prefill = Int("prefill", 512, 1, 512);
            inputPath = Path.GetFullPath(a["input"]);
            if (new FileInfo(inputPath).Length > 4_194_304) throw new ArgumentException("Input manifest too large");
            inputHash = Hash(inputPath); report["InputFile"] = Identity(inputPath);
            InputDocument input = JsonSerializer.Deserialize<InputDocument>(File.ReadAllText(inputPath), Json)
                ?? throw new InvalidDataException("Missing input document");
            if (input.Schema != "qwen-unified-prefill-input.v1") throw new InvalidDataException("Unknown input schema");
            input.Primary.Validate(false, context, maxNew);
            if (mode == "cancel-cohort") (input.Occupier ?? throw new InvalidDataException("Frozen occupier input required")).Validate(true, context, occupierMax);
            string modelPath = Path.GetFullPath(a["model"]), projector = Path.GetFullPath(a["mmproj"]);
            models = [.. Shards(modelPath), projector];
            report["ModelStatsBefore"] = models.Select(Stat).ToArray(); initialStats = JsonSerializer.Serialize(models.Select(Stat));
            string manifest = Path.GetFullPath(a["model-identity"]);
            using var history = JsonDocument.Parse(File.ReadAllText(manifest));
            var rows = history.RootElement.GetProperty("files").EnumerateArray().ToArray();
            foreach (string path in models)
            {
                var match = rows.Where(r => SamePath(r.GetProperty("path").GetString()!, path)).ToArray();
                if (match.Length != 1 || match[0].GetProperty("bytes").GetInt64() != new FileInfo(path).Length
                    || match[0].GetProperty("mtime_ns").GetInt64() != checked((File.GetLastWriteTimeUtc(path).Ticks - DateTime.UnixEpoch.Ticks) * 100)
                    || !Regex.IsMatch(match[0].GetProperty("sha256").GetString() ?? "", @"\A[0-9a-fA-F]{64}\z"))
                    throw new InvalidDataException("Historical identity mismatch: " + path);
            }
            report["HistoricalModelIdentity"] = Identity(manifest);
            nativePath = Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "GgmlOps.dll" :
                OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so");
            nativeHash = Hash(nativePath);
            if (!string.Equals(a["native-sha256"], nativeHash, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("Native pin mismatch");
            report["IntendedNative"] = Identity(nativePath);
            identities = Directory.EnumerateFiles(AppContext.BaseDirectory, "*.dll").Append(typeof(Program).Assembly.Location)
                .Append(manifest).Distinct().Order(StringComparer.Ordinal).ToArray();
            report["DeploymentBefore"] = identities.Select(Identity).ToArray();
            var environment = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
                .Where(e => new[] { "TS_", "TENSORSHARP_", "GGML_", "CUDA_", "OMP_" }.Any(prefix => ((string)e.Key).StartsWith(prefix, StringComparison.Ordinal)))
                .ToDictionary(e => (string)e.Key, e => (string?)e.Value);
            report["EnvironmentBefore"] = environment;
            if (SpeculationOptions.FromEnvironment().Enabled || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_SPEC_DRAFT_MODEL"))
                || environment.Any(e => (e.Key.Contains("TRACE", StringComparison.Ordinal) || e.Key.Contains("PROFILE", StringComparison.Ordinal)) && e.Value is not (null or "" or "0")))
                throw new InvalidOperationException("Inherited speculation/trace is not an ordinary probe configuration");
            MoeCpuOffloadConfig.ConfigureFromEnvironment();
            if (MoeCpuOffloadConfig.IsExplicitlySet) throw new InvalidOperationException("Inherited CPU expert placement override");
            MoeCpuOffloadConfig.SetCpuThreads(Int("cpu-threads", 8, 1, 1024));
            UnifiedMemoryModelOptions options = new() { ContextTokens = context, PrefillChunkTokens = prefill,
                MaximumRunningRequests = mode == "single" ? 1 : 2, PrefixCheckpointCount = 0, RetainedConversationCount = 0,
                ExpertCachePrefillRows = Math.Min(prefill, 32), MaximumHostBytes = Bytes("host-bytes", long.MaxValue, 1),
                MaximumDeviceBytes = Bytes("device-bytes", long.MaxValue, 1), MaximumExpertCacheBytes = Bytes("expert-bytes", long.MaxValue, 0),
                EnableAsyncExpertPrefill = asyncPrefill, EnableExpertCacheCapacityGrowth = false };
            report["SessionOptions"] = options; report["PhysicalBeforeLoad"] = Physical();
            pipelineObservationStarted = true;
            pipelineBefore = ObservePipeline("Before");
            if (pipelineBefore == null) throw new InvalidOperationException("Public file-pipeline telemetry unavailable before model construction");
            report["FilePipelineUsageAvailability"] = "Before load, after engine disposal attempt and after session disposal attempt; failures remain explicit.";
            Save(Path.Combine(output, "execution-started.json"), report);
            deadline = new CancellationTokenSource(TimeSpan.FromSeconds(Int("deadline-seconds", 1200, 1, 86400)));
            var load = Stopwatch.StartNew();
            session = UnifiedMemoryModelSession.Create(modelPath, projector, BackendType.GgmlCuda, options);
            load.Stop(); budget = session.Budget;
            report["LoadMs"] = load.Elapsed.TotalMilliseconds; report["BudgetAfterLoad"] = Ledger(budget);
            report["FeatureReport"] = session.FeatureReport; report["SessionPrefillCeiling"] = session.PrefillChunkTokens;
            var model = session.Model ?? throw new InvalidOperationException("No public model");
            if (model.Config.Architecture != "qwen4exp") throw new InvalidDataException("Require Qwen4Exp checkpoint");
            report["LoadedNative"] = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
                .Where(m => SamePath(m.FileName, nativePath)).Select(m => Identity(m.FileName)).ToArray();
            if (((object[])report["LoadedNative"]!).Length != 1) throw new InvalidDataException("Intended native not observed loaded");
            void BindInput(InputSpec spec)
            {
                int[] rendered = new KVCachePromptRenderer(new GgufPromptRenderer()).RenderToTokens(model.Tokenizer,
                    model.Config.ChatTemplate, spec.Messages.ToList(), model.Config.Architecture,
                    addGenerationPrompt: true, enableThinking: false).ToArray();
                if (!rendered.SequenceEqual(spec.Tokens)) throw new InvalidDataException("Frozen IDs do not reproduce the independent exact input messages");
            }
            BindInput(input.Primary); if (mode == "cancel-cohort") BindInput(input.Occupier!);
            SchedulerConfig configured = session.ConfigureScheduler(new() { EnablePrefixCaching = false, StopRepetition = false,
                DecodeQuantumTokens = 1, Speculation = SpeculationOptions.Disabled, BlockSize = 16 });
            RequestMemoryAdmission original = configured.MemoryAdmission ?? throw new InvalidOperationException("Missing production admission");
            if (!ReferenceEquals(original.Budget, budget)) throw new InvalidOperationException("Admission lost real shared budget");
            SchedulerConfig config = AdmissionObservation.WithAdmission(configured, observer.Wrap(original));
            engine = new InferenceEngine(model, config);
            report["Scheduler"] = new { config.MaxNumRunningSequences, config.MaxPrefillChunkSize, config.PrefillChunkTokenLimit,
                config.EnablePrefixCaching, config.StopRepetition, SharedBudget = true, TransparentScopeObserver = true };
            var firstOccupier = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            SequenceState? primary = null, occupier = null;
            observer.AfterScope = seen =>
            {
                if (mode != "cancel-cohort" || abortAt != null || primary == null || occupier == null) return;
                if (seen.Any(r => r.RequestId == primary.RequestId && r.OutputTokens >= 1 && r.AfterComputed > r.BeforeComputed))
                {
                    abortAt = DateTime.UtcNow; abortId = occupier.RequestId; primaryComputedAtAbort = primary.NumComputedTokens;
                    engine.Abort(occupier.RequestId); // Real public cancellation queued between model steps.
                }
            };
            async Task<object> Run(InputSpec spec, bool isOccupier, SequenceState seq)
            {
                var published = new List<int>(); var wall = Stopwatch.StartNew(); double? first = null;
                InferenceRequestHandle handle = engine.SubmitRequest(seq, deadline.Token); handles.Add(handle);
                await foreach (int token in handle.Tokens.ReadAllAsync(deadline.Token))
                { first ??= wall.Elapsed.TotalMilliseconds; published.Add(token); if (isOccupier) firstOccupier.TrySetResult(); }
                InferenceCompletion completion = await handle.Completion.WaitAsync(deadline.Token); wall.Stop();
                int[] full = seq.OutputTokens.ToArray();
                bool eos = OutputQualification.HasTerminalEos(full, model.Tokenizer.EosTokenIds);
                var parser = OutputParserFactory.Create(model.Config.Architecture); parser.Init(false, null);
                ParsedOutput parsed = parser.Add(model.Tokenizer.Decode(full.Take(full.Length - (eos ? 1 : 0)).ToList()), true);
                bool structural = OutputQualification.IsNaturalFusedCompletion(completion, spec.Tokens.Length,
                    full, published, seq.NumComputedTokens, seq.MaxNewTokens, seq.Error != null, model.Tokenizer.EosTokenIds);
                bool strictContent = OutputQualification.IsExact42(parsed.Content);
                bool jsonIntegerContent = OutputQualification.IsJsonInteger42(parsed.Content);
                bool independent = !isOccupier && structural && OutputQualification.MatchesPolicy(answerPolicy, parsed.Content)
                    && (parsed.ToolCalls?.Count ?? 0) == 0;
                return new { seq.RequestId, spec.Task, completion.Status, completion.FinishReason, PromptTokens = spec.Tokens,
                    OutputTokens = full, PublishedTokens = published, seq.NumComputedTokens, Completion = completion,
                    RawText = model.Tokenizer.Decode(full.ToList()), parsed.Content, parsed.Thinking, parsed.ToolCalls,
                    NaturalEos = eos, StructuralPassed = structural, SemanticPassed = isOccupier ? (bool?)null : independent,
                    AnswerPolicy = answerPolicy, StrictExactContent42 = isOccupier ? (bool?)null : strictContent,
                    JsonInteger42Content = isOccupier ? (bool?)null : jsonIntegerContent,
                    StrictFormatNonconformance = isOccupier ? (bool?)null : !strictContent,
                    IntentionallyCancelled = isOccupier, RequestWallMs = wall.Elapsed.TotalMilliseconds, FirstPublishedTokenMs = first,
                    PrefillAttributedMs = completion.PrefillElapsedTicks * 1000.0 / Stopwatch.Frequency,
                    DecodeAttributedMs = completion.DecodeElapsedTicks * 1000.0 / Stopwatch.Frequency };
            }
            Task<object>? occupierTask = null;
            if (mode == "cancel-cohort")
            {
                occupier = new("occupier-" + Guid.NewGuid().ToString("N"), input.Occupier!.Tokens, occupierMax, engine.PoolStats.blockSize, SamplingConfig.Greedy);
                sequences.Add(occupier); occupierTask = Run(input.Occupier, true, occupier);
                Task first = await Task.WhenAny(firstOccupier.Task, occupierTask).WaitAsync(deadline.Token);
                if (first != firstOccupier.Task) { await occupierTask; throw new InvalidOperationException("Occupier ended before its first publication"); }
            }
            primary = new("math-" + Guid.NewGuid().ToString("N"), input.Primary.Tokens, maxNew, engine.PoolStats.blockSize, SamplingConfig.Greedy);
            sequences.Add(primary);
            object primaryResult = await Run(input.Primary, false, primary); results.Add(primaryResult);
            if (occupierTask != null) results.Add(await occupierTask);
            report["Requests"] = results;
            var observations = observer.Snapshot(); report["ExecutionScopes"] = observations;
            var primaryJson = JsonSerializer.SerializeToElement(primaryResult, Json);
            quality = primaryJson.GetProperty("SemanticPassed").GetBoolean();
            rowsPass = sequences.All(seq => observations.Where(r => r.RequestId == seq.RequestId).Sum(r => r.SuccessfulPrefillRows) == seq.PromptTokens.Count)
                && observations.All(r => r.SuccessfulPrefillRows == 0 || r.Kind == "serial"
                    && r.SuccessfulPrefillRows <= Math.Min(r.BeforeLimit, r.AfterLimit));
            bool downgraded = observations.Any(r => r.SuccessfulPrefillRows > 0 && r.BeforeLimit < Math.Min(prefill, r.PromptTokens));
            report["ObservedAutomaticDowngrade"] = downgraded; report["PrefillRowsPassed"] = rowsPass;
            bool expectationPassed = (expectation == "true") == downgraded;
            report["DowngradeExpectationPassed"] = expectationPassed;
            int successfulJointSteps = observations.Where(r => r.Kind == "joint" && r.AfterComputed > r.BeforeComputed)
                .Select(r => r.FinishedUtc).Distinct().Count();
            report["SuccessfulJointSteps"] = successfulJointSteps;
            cancellationPass = mode == "single" || abortAt != null && occupier?.Status == SequenceStatus.FinishedAborted
                && primary.Status == SequenceStatus.FinishedStopped && primary.NumComputedTokens > primaryComputedAtAbort
                && successfulJointSteps > 0
                && observations.Any(r => r.RequestId == primary.RequestId && r.FinishedUtc >= abortAt && r.AfterComputed > primaryComputedAtAbort);
            report["Cancellation"] = new { RequestedUtc = abortAt, RequestId = abortId, primaryComputedAtAbort, Passed = cancellationPass };
            report["BudgetAfterCompletion"] = Ledger(budget); report["PhysicalAfterCompletion"] = Physical();
            report["EngineBeforeDispose"] = new { engine.PendingRequestCount, engine.RunningCount, engine.WaitingCount,
                engine.TotalCompleted, engine.TotalSubmitted, engine.TotalStepsRun };
            if (!expectationPassed) throw new InvalidOperationException("Actual automatic downgrade differs from the explicit expectation");
        }
        catch (UnifiedMemoryModelLoadException e) { session = e.UnreleasedSession; budget = session.Budget; errors.Add(Error(e)); }
        catch (Exception e) { errors.Add(Error(e)); }
        finally
        {
            if (engine != null)
            {
                foreach (var handle in handles) if (!handle.Completion.IsCompleted) engine.Abort(handle.RequestId);
                try { engine.Dispose(); engineDisposed = true; } catch (Exception e) { errors.Add(Error(e)); }
                if (engineDisposed)
                {
                    report["EngineAfterDispose"] = new { engine.PendingRequestCount, engine.PendingSequenceReleaseCount,
                        engine.RunningCount, engine.WaitingCount };
                    if (engine.PendingRequestCount != 0 || engine.PendingSequenceReleaseCount != 0)
                        errors.Add("Engine disposed with outstanding request/release owners");
                }
            }
            if (budget != null) report["BudgetAfterEngineDisposeAttempt"] = Ledger(budget);
            if (pipelineObservationStarted)
            {
                if (engine == null || engineDisposed) pipelineAfter = ObservePipeline("After");
                else report["FilePipelineUsageAfter"] = new { Available = false, Reason = "Engine disposal failed; worker quiescence is unconfirmed." };
            }
            if (session != null && (engine == null || engineDisposed))
                try { session.Dispose(); sessionDisposed = true; } catch (Exception e) { errors.Add(Error(e)); }
            report["EngineDisposed"] = engineDisposed; report["SessionDisposed"] = sessionDisposed;
            if (pipelineObservationStarted)
            {
                if (sessionDisposed) pipelineFinal = ObservePipeline("Final");
                else report["FilePipelineUsageFinal"] = new { Available = false, Reason = "Session disposal did not complete; final quiescence and ring cleanup are unconfirmed." };
            }
            bool pipelinePassed = FilePipelineQualification.Qualifies(asyncPrefill, pipelineBefore, pipelineAfter, pipelineFinal);
            report["FilePipelineQualification"] = new {
                Enabled = asyncPrefill, Passed = pipelinePassed,
                ActualPathCovered = pipelineBefore is { } start && pipelineAfter is { } end
                    && FilePipelineQualification.HasActualUse(start, end),
                CallsDelta = pipelineBefore is { } first && pipelineAfter is { } last
                    && FilePipelineQualification.Monotonic(first, last) ? (long?)(last.Calls - first.Calls) : null,
                SourceBytesDelta = pipelineBefore is { } firstBytes && pipelineAfter is { } lastBytes
                    && FilePipelineQualification.Monotonic(firstBytes, lastBytes) ? (long?)(lastBytes.SourceBytes - firstBytes.SourceBytes) : null,
                Requirement = "Valid monotonic before/after/final counters, no counter drift after engine disposal and final retained ring0; enabled mode requires positive Calls and SourceBytes deltas, disabled mode requires both deltas0." };
            if (budget != null)
            {
                report["BudgetAfterSessionDisposeAttempt"] = Ledger(budget);
                cleanup = sessionDisposed && budget.Snapshot().All(p => p.Reserved == 0 && p.Committed == 0);
                report["LedgerZeroAfterCleanup"] = cleanup;
            }
            else report["LedgerAvailability"] = "Unavailable construction result; never imputed as zero";
            report["ExecutionScopes"] = observer.Snapshot(); report["PartialSequencesAfterCleanup"] = sequences.Select(s => new {
                s.RequestId, s.Status, s.NumComputedTokens, OutputTokens = s.OutputTokens.ToArray(), Error = s.Error?.ToString() }).ToArray();
            bool stable = false;
            try
            {
                report["DeploymentAfter"] = identities.Select(Identity).ToArray(); report["ModelStatsAfter"] = models.Select(Stat).ToArray();
                stable = inputPath != null && Hash(inputPath) == inputHash && nativePath != null && Hash(nativePath) == nativeHash
                    && initialStats == JsonSerializer.Serialize(models.Select(Stat))
                    && JsonSerializer.Serialize(report.GetValueOrDefault("DeploymentBefore"), Json) == JsonSerializer.Serialize(report["DeploymentAfter"], Json);
            }
            catch (Exception e) { errors.Add(Error(e)); }
            report["IdentitiesUnchanged"] = stable; report["Errors"] = errors; report["Requests"] = results;
            report["PhysicalAfterCleanup"] = Physical(); report["FinishedUtc"] = DateTime.UtcNow;
            report["Passed"] = quality && rowsPass && cancellationPass && cleanup && engineDisposed && stable && pipelinePassed && errors.Count == 0;
            Save(Path.Combine(output, "report.json"), report);
        }
        deadline?.Dispose();
        Console.WriteLine(JsonSerializer.Serialize(new { Passed = report["Passed"], Output = output, Errors = errors.Count }));
        return report["Passed"] is true ? 0 : 1;
    }
}
