using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

// Diagnostic teacher forcing only. Never grade these deliberately constrained
// output tokens as generated-answer quality or include capture overhead in a benchmark.
if (args.SequenceEqual(["--self-test"])) return ProbeChecks.Run();
var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Require unique --name value pairs.");
string[] allowed = ["model", "mmproj", "input", "image", "output", "chunk", "merger", "timeout", "native-sha256"];
if (options.Keys.Except(allowed).Any()) throw new ArgumentException("Unknown option.");
int chunk = int.Parse(options["chunk"]);
if (chunk is not (32 or 317)) throw new ArgumentException("This diagnostic compares PP32 and PP317 only.");
if (options["merger"] is not ("legacy" or "reference")) throw new ArgumentException("Use merger legacy or reference.");
string output = Path.GetFullPath(options["output"]);
if (Directory.Exists(output) || File.Exists(output)) throw new ArgumentException("Fresh ignored output required.");
Directory.CreateDirectory(output);
var json = new JsonSerializerOptions { WriteIndented = true, PropertyNameCaseInsensitive = true };
void Save(string name, object value) => File.WriteAllText(Path.Combine(output, name), JsonSerializer.Serialize(value, json));
string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
var report = new Dictionary<string, object?> { ["Schema"] = "qwen-image-logit-probe.v1", ["Completed"] = false,
    ["Scope"] = "Fixed production shared-budget session with cache0 and CPU-only routed expert execution; PP shape diagnostic, not ordinary quality/performance qualification." };
IUnifiedMemoryModelSession? session = null;
InferenceEngine? engine = null;
StateCapture? state = null;
bool engineDisposed = false;
int exit = 1;
try
{
    foreach (System.Collections.DictionaryEntry item in Environment.GetEnvironmentVariables())
        if (((string)item.Key).StartsWith("TS_", StringComparison.Ordinal)
            || ((string)item.Key).StartsWith("TENSORSHARP_", StringComparison.Ordinal))
            if (!string.IsNullOrEmpty((string?)item.Value)) throw new InvalidOperationException("Inherited runtime override: " + item.Key);
    Environment.SetEnvironmentVariable("TS_HOST_MOE_DEVICE_MIN_BATCH", "0");
    Environment.SetEnvironmentVariable("TS_Q4E_PREFILL_COMBINE", "0");
    Environment.SetEnvironmentVariable("TS_Q4E_VISION_MERGER_ERF", options["merger"] == "legacy" ? "0" : "1");
    MoeCpuOffloadConfig.Reset(); MoeCpuOffloadConfig.SetCpuThreads(8);
    Input input = JsonSerializer.Deserialize<Input>(File.ReadAllText(options["input"]), json) ?? throw new InvalidDataException("Missing input.");
    if (input.Schema != "qwen-image-logit-input.v1" || input.ExpectedPromptTokens.Length != 317
        || !input.TeacherTokens.SequenceEqual([271, 101878]) || input.Messages.Count != 1 || input.Messages[0].Role != "user")
        throw new InvalidDataException("Require the original 317-token image prompt and the two common observed teacher IDs.");
    string image = Path.GetFullPath(options["image"]);
    if (Hash(image) != input.ImageSha256) throw new InvalidDataException("Image identity changed.");
    input.Messages[0].ImagePaths = [image];
    string native = Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "GgmlOps.dll" : OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so");
    string nativeHash = Hash(native);
    if (!string.Equals(nativeHash, options["native-sha256"], StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("Native identity mismatch.");
    report["Input"] = input; report["InputFileSha256"] = Hash(options["input"]);
    report["NativeSha256"] = nativeHash; report["Chunk"] = chunk; report["Merger"] = options["merger"];
    report["ManagedCore"] = new[] { typeof(ModelBase).Assembly, typeof(InferenceEngine).Assembly, typeof(TensorSharp.Memory.MemoryBudget).Assembly }
        .Select(a => new { Path = a.Location, Sha256 = Hash(a.Location) }).ToArray();
    var load = Stopwatch.StartNew();
    session = UnifiedMemoryModelSession.Create(options["model"], options["mmproj"], BackendType.GgmlCuda,
        new UnifiedMemoryModelOptions { ContextTokens = 32768, PrefillChunkTokens = chunk, MaximumRunningRequests = 1,
            PrefixCheckpointCount = 0, RetainedConversationCount = 0, MaximumExpertCacheBytes = 0,
            ExpertCachePrefillRows = 32, EnableAsyncExpertPrefill = false, EnableExpertCacheCapacityGrowth = false });
    report["LoadMs"] = load.Elapsed.TotalMilliseconds; report["FeatureReport"] = session.FeatureReport;
    var model = session.Model as Qwen4ExpModel ?? throw new InvalidOperationException("Require Qwen4Exp.");
    state = new StateCapture(model, session.Budget);
    var placement = (bool[]?)typeof(Qwen4ExpModel).GetField("_expertOnHost", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(model);
    if (placement == null || placement.Length == 0 || placement.Any(x => !x))
        throw new InvalidOperationException("Diagnostic requires every routed expert layer on host.");
    report["HostRoutedLayers"] = placement.Length;
    var cacheBefore = GgmlBasicOps.GetHostMoeExpertCacheUsage();
    if (cacheBefore.AccountedBytes != 0 || cacheBefore.BudgetBytes != 0)
        throw new InvalidOperationException("Optional expert cache was not actually disabled.");
    report["ExpertCacheBefore"] = cacheBefore;
    string id = "image-logits-" + Guid.NewGuid().ToString("N");
    var raw = new KVCachePromptRenderer(new GgufPromptRenderer()).RenderToTokens(model.Tokenizer,
        model.Config.ChatTemplate, input.Messages, model.Config.Architecture, addGenerationPrompt: true, enableThinking: false);
    int[] expanded = model.MultimodalInjector.ProcessPromptTokens(input.Messages, raw.ToList(), id).ToArray();
    if (!expanded.SequenceEqual(input.ExpectedPromptTokens)) throw new InvalidDataException("Rendered/expanded prompt differs from original HTTP token IDs.");
    report["PreparedMedia"] = state.PreparedMedia(model.MultimodalInjector, id);
    var captures = new List<object>();
    var scopes = new List<object>();
    report["Captures"] = captures; report["Scopes"] = scopes;
    SchedulerConfig configured = session.ConfigureScheduler(new() { EnablePrefixCaching = false,
        StopRepetition = false, DecodeQuantumTokens = 1, Speculation = SpeculationOptions.Disabled, BlockSize = 16 });
    RequestMemoryAdmission original = configured.MemoryAdmission ?? throw new InvalidOperationException("No admission.");
    var observer = new ObservedAdmission(original, (seq, before, admitted) =>
    {
            int after = seq.NumComputedTokens;
            scopes.Add(new { Before = before, After = after, Admitted = admitted });
            if (after == before || after < expanded.Length) return;
            int step = after - expanded.Length;
            float[] logits = seq.LastLogits ?? throw new InvalidOperationException("Missing actual post-forward logits.");
            ProbeChecks.ValidateLogits(logits, model.Config.VocabSize);
            string name = $"step-{step}.f32";
            using (var file = File.Create(Path.Combine(output, name)))
                file.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(logits.AsSpan()));
            int[] top = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).Take(16).ToArray();
            captures.Add(new { Step = step, Computed = after, LogitsFile = name, Vocabulary = logits.Length,
                Argmax = top[0], Margin = (double)logits[top[0]] - logits[top[1]],
                Top = top.Select(i => new { Id = i, Value = logits[i] }).ToArray(), State = state.Capture(seq.RequestId, after) });
            // Capture the full unmodified distribution and state FIRST. Only the
            // validation sequence's cloned sampling buffer is then constrained
            // to supply identical next model inputs; model arithmetic is unchanged.
            if (seq.OutputTokens.Count < input.TeacherTokens.Length)
            {
                ProbeChecks.ForceTeacher(logits, input.TeacherTokens[seq.OutputTokens.Count]);
            }
    });
    engine = new InferenceEngine(model, observer.Configuration(configured));
    var seq = new SequenceState(id, expanded, input.TeacherTokens.Length, engine.PoolStats.blockSize, SamplingConfig.Greedy,
        mediaSpans: model.MultimodalInjector.GetPreparedMediaSpans(id));
    using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(int.Parse(options.GetValueOrDefault("timeout", "600"))));
    var requestTimer = Stopwatch.StartNew();
    var handle = engine.SubmitRequest(seq, deadline.Token);
    await foreach (int _ in handle.Tokens.ReadAllAsync(deadline.Token)) { }
    InferenceCompletion completion = await handle.Completion.WaitAsync(deadline.Token);
    report["RequestMs"] = requestTimer.Elapsed.TotalMilliseconds;
    report["Completion"] = completion; report["Scopes"] = scopes; report["Captures"] = captures;
    report["TeacherTokensActual"] = seq.OutputTokens.Take(input.TeacherTokens.Length).ToArray();
    report["OutputTokensActual"] = seq.OutputTokens.ToArray();
    var cacheAfter = GgmlBasicOps.GetHostMoeExpertCacheUsage();
    report["ExpertCacheAfter"] = cacheAfter;
    if (cacheAfter.AccountedBytes != 0 || cacheAfter.BudgetBytes != 0)
        throw new InvalidOperationException("Optional expert cache unexpectedly acquired state.");
    int[] deltas = observer.Rows.Where(x => x.Before < expanded.Length && x.After > x.Before).Select(x => Math.Min(x.After, expanded.Length) - x.Before).ToArray();
    bool shape = ProbeChecks.ExactShape(deltas, expanded.Length, chunk);
    report["ActualPrefillRows"] = deltas; report["RequestedShapeEngaged"] = shape;
    if (!shape || captures.Count != 3 || !seq.OutputTokens.Take(input.TeacherTokens.Length).SequenceEqual(input.TeacherTokens) || seq.Error != null)
        throw new InvalidOperationException("Shape, common teacher history or capture coverage was not achieved.");
    report["BudgetBeforeDispose"] = session.Budget.Snapshot();
    report["Completed"] = true; exit = 0;
}
catch (Exception e) { report["Error"] = e.ToString(); }
finally
{
    try { if (engine != null) { engine.Dispose(); engineDisposed = true; } }
    catch (Exception e) { report["EngineCleanupError"] = e.ToString(); exit = 1; }
    if (engine == null || engineDisposed)
    {
        try { state?.Dispose(); state = null; session?.Dispose(); report["SessionDisposed"] = true; }
        catch (Exception e) { report["SessionCleanupError"] = e.ToString(); exit = 1; }
    }
    else report["SessionDisposed"] = false;
    if (session != null)
    {
        var final = session.Budget.Snapshot(); report["FinalBudget"] = final;
        if (final.Any(x => x.Reserved != 0 || x.Committed != 0)) exit = 1;
    }
    report["ExitCode"] = exit; Save("report.json", report);
}
return exit;

sealed record Input(string Schema, List<ChatMessage> Messages, int[] ExpectedPromptTokens, int[] TeacherTokens, string ImageSha256);

sealed class ObservedAdmission(RequestMemoryAdmission original, Action<SequenceState, int, int> finish)
{
    public List<(int Before, int After)> Rows { get; } = [];
    public SchedulerConfig Configuration(SchedulerConfig c) => new()
    {
        BlockSize = c.BlockSize, NumBlocks = c.NumBlocks, MaxNumRunningSequences = c.MaxNumRunningSequences,
        MaxNumBatchedTokens = c.MaxNumBatchedTokens, MaxPrefillChunkSize = c.MaxPrefillChunkSize,
        SoloPrefillChunkSize = c.SoloPrefillChunkSize, PrefillChunkTokenLimit = c.PrefillChunkTokenLimit,
        EnablePrefixCaching = c.EnablePrefixCaching, StopRepetition = c.StopRepetition, DecodeQuantumTokens = c.DecodeQuantumTokens,
        Speculation = c.Speculation, KvSnapshots = c.KvSnapshots, ExecutionOptionsOverride = c.ExecutionOptionsOverride,
        MemoryAdmission = new(original.Budget, original.EstimatePeak, original.MaxQueuedRequests)
        {
            EstimateMaterializationPeak = original.EstimateMaterializationPeak, BeforeAdmission = original.BeforeAdmission,
            BeforeAdmissionWithRunning = original.BeforeAdmissionWithRunning, AfterPrefill = original.AfterPrefill,
            AfterPrefixMaterialization = original.AfterPrefixMaterialization, PrefillTokenLimit = original.PrefillTokenLimit,
            RevalidateExecution = original.RevalidateExecution, AfterSequenceMemoryRelease = original.AfterSequenceMemoryRelease,
            UsePerSequenceHolders = original.UsePerSequenceHolders, ReestimateAfterPhysicalCheck = original.ReestimateAfterPhysicalCheck,
            ExecutionShape = original.ExecutionShape, EnterBatchedExecution = original.EnterBatchedExecution,
            EnterSerialExecution = seq =>
            {
                int before = seq.NumComputedTokens, cap = original.PrefillTokenLimit?.Invoke(seq) ?? int.MaxValue;
                IDisposable actual = original.EnterSerialExecution!(seq);
                return new Scope(() =>
                {
                    try { Rows.Add((before, seq.NumComputedTokens)); finish(seq, before, cap); }
                    finally { actual.Dispose(); }
                });
            }
        }
    };
    sealed class Scope(Action release) : IDisposable
    {
        Action? _release = release;
        public void Dispose() => Interlocked.Exchange(ref _release, null)?.Invoke();
    }
}
