// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 == args.Length) throw new ArgumentException("Options require values.");
    options.Add(args[i], args[i + 1]);
}
string path = Path.GetFullPath(options["--model"]), output = Path.GetFullPath(options["--output"]);
if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
    throw new IOException("Use a fresh or empty output directory; existing evidence will not be overwritten.");
Directory.CreateDirectory(output);
DateTime startedUtc = DateTime.UtcNow;
// Identity is outside every measured interval. Reading the file also means this
// probe cannot be described as a cold-storage benchmark.
string modelHash = Hash(path);
string mode = options.GetValueOrDefault("--mode", "adaptive");
if (mode is not ("resident" or "adaptive")) throw new ArgumentException("--mode resident|adaptive");
int steps = int.Parse(options.GetValueOrDefault("--steps", "64"));
int repeats = int.Parse(options.GetValueOrDefault("--repeats", "3"));
int minimumPrompt = int.Parse(options.GetValueOrDefault("--prompt-tokens", "640"));
int context = int.Parse(options.GetValueOrDefault("--context", "2048"));
if (steps < 2 || repeats < 1 || minimumPrompt < 1 || context <= (long)minimumPrompt + steps)
    throw new ArgumentException("Require positive runs and a context larger than prompt plus decode.");
bool captureLogits = bool.Parse(options.GetValueOrDefault("--capture-logits", "false"));
int[]? teacher = options.TryGetValue("--teacher", out string? teacherFile)
    ? JsonSerializer.Deserialize<int[]>(File.ReadAllText(teacherFile)) : null;
if (options.ContainsKey("--teacher") && (teacher == null || teacher.Length != steps || teacher.Any(token => token < 0)))
    throw new ArgumentException("--teacher must contain a JSON integer array with exactly --steps nonnegative token IDs.");
string? teacherHash = teacherFile == null ? null : Hash(teacherFile);
Environment.SetEnvironmentVariable("MAX_CONTEXT", context.ToString());
Environment.SetEnvironmentVariable("KV_CACHE_DTYPE", "f16");
KvCacheDtypeConfig.ConfigureFromEnvironment();
foreach (string variable in new[] { "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TS_SPEC", "SPECULATIVE_DECODING", "TS_MTP" })
    Environment.SetEnvironmentVariable(variable, null);
string[] benchmarkVariables = ["TS_GGML_Q8_PARALLEL_VECTOR", "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE",
    "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS",
    "OMP_NUM_THREADS", "TS_GGML_CPU_THREADS", "KV_CACHE_DTYPE", "MAX_CONTEXT"];
var benchmarkEnvironment = benchmarkVariables.ToDictionary(name => name, Environment.GetEnvironmentVariable);

var records = new List<object>();
AdaptiveModelSession? session = null;
ModelBase? model = null;
object? plan = null, afterLoad = null, afterDispose = null;
object? geometry = null, captured = null;
LogitCapture? capture = null;
bool shutdown = false;
string? error = null;
int[] prompt = [];
double loadMs = 0;
try
{
    var timer = Stopwatch.StartNew();
    if (mode == "adaptive")
    {
        session = AdaptiveModelSession.Create(path, new(context, (int)Math.Min(context, (long)minimumPrompt + 64))
        {
            MaximumDeviceBytes = long.Parse(options.GetValueOrDefault("--device-bytes", long.MaxValue.ToString())),
            MaximumHostBytes = long.Parse(options.GetValueOrDefault("--host-bytes", long.MaxValue.ToString()))
        });
        model = session.Model;
        plan = session.Plan;
    }
    else model = ModelBase.Create(path, BackendType.GgmlCuda);
    loadMs = timer.Elapsed.TotalMilliseconds;
    afterLoad = MemorySample();
    if (teacher?.Any(token => token >= model.Tokenizer.VocabSize) == true)
        throw new ArgumentException("--teacher contains a token outside the model vocabulary.");
    geometry = new { model.Config.Architecture, model.Config.HiddenSize, model.Config.NumLayers,
        model.Config.NumHeads, model.Config.NumKVHeads, Vocabulary = model.Tokenizer.VocabSize, Context = context };
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    string text = "Count upwards from 10 to 1000, separated by spaces. Output only the numbers.";
    do
    {
        prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate,
            [new ChatMessage { Role = "user", Content = text }], model.Config.Architecture,
            addGenerationPrompt: true, enableThinking: false).ToArray();
        if (prompt.Length < minimumPrompt) text += " Include every consecutive integer without gaps.";
    } while (prompt.Length < minimumPrompt);
    if ((long)prompt.Length + steps > context) throw new ArgumentException("Rendered prompt exceeds admitted context.");
    File.WriteAllText(Path.Combine(output, "prompt.json"), JsonSerializer.Serialize(prompt));
    if (captureLogits) capture = new LogitCapture(output);

    // The warmup is recorded separately. Each measured request starts with empty
    // logical KV but preserves reusable weights, graph workspaces and allocations.
    for (int run = -1; run < repeats; run++)
    {
        if (session != null && !session.RefreshCapacity())
            throw new InvalidOperationException("Hardware pressure requires pausing request admission.");
        model.ResetKVCache();
        var before = MemorySample();
        timer.Restart();
        float[] logits = model.ForwardRefill(prompt);
        double prefillMs = timer.Elapsed.TotalMilliseconds;
        double decodeMs = 0;
        int[] generated = new int[steps];
        int[] consumed = new int[steps - 1];
        var history = capture == null || run < 0 ? null : new List<int>(prompt);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (int step = 0; step < steps; step++)
        {
            if (!logits.All(float.IsFinite)) throw new ArithmeticException("Nonfinite full-vocabulary logits.");
            hash.AppendData(MemoryMarshal.AsBytes(logits.AsSpan()));
            generated[step] = Enumerable.Range(0, logits.Length).MaxBy(i => logits[i]);
            if (run >= 0) capture?.Append(logits, run, step, history!, generated[step]);
            if (step + 1 == steps) break;
            int next = teacher?[step] ?? generated[step];
            consumed[step] = next;
            timer.Restart();
            logits = model.Forward([next]);
            decodeMs += timer.Elapsed.TotalMilliseconds;
            history?.Add(next);
        }
        records.Add(new
        {
            Run = run, Warmup = run < 0, PromptTokens = prompt.Length, DecodeCalls = steps - 1,
            PrefillMilliseconds = prefillMs, DecodeMilliseconds = decodeMs,
            PrefillTokensPerSecond = prompt.Length * 1000 / prefillMs,
            DecodeTokensPerSecond = (steps - 1) * 1000 / decodeMs,
            LogitsSha256 = Convert.ToHexString(hash.GetHashAndReset()), Generated = generated, Consumed = consumed,
            Before = before, After = MemorySample(), Streaming = model.StreamingWeightUsage
        });
        Console.WriteLine($"{mode} run={run}: prefill={prompt.Length * 1000 / prefillMs:F2}, decode={(steps - 1) * 1000 / decodeMs:F2} tok/s");
    }
}
catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(error); }
finally
{
    try
    {
        if (session != null) session.Dispose(); else model?.Dispose();
        afterDispose = session?.Budget.Snapshot();
        if (session?.Budget.Snapshot().Any(p => p.Reserved != 0 || p.Committed != 0) == true)
            throw new InvalidOperationException("Model disposal retained budget owners.");
    }
    catch (Exception ex) { error = (error + Environment.NewLine + ex).Trim(); }
    try
    {
        capture?.Dispose();
        captured = capture?.Evidence();
    }
    catch (Exception ex) { error = (error + Environment.NewLine + ex).Trim(); }
    if (error == null)
    {
        try { GgmlBasicOps.Shutdown(); shutdown = true; }
        catch (Exception ex) { error = ex.ToString(); }
    }
    File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(new
    {
        Executed = error == null, Error = error, Mode = mode, Model = path,
        ProcessId = Environment.ProcessId, StartedUtc = startedUtc, FinishedUtc = DateTime.UtcNow,
        Environment = benchmarkEnvironment,
        ModelBytes = new FileInfo(path).Length, ModelSha256 = modelHash,
        Context = context, LoadMilliseconds = loadMs, Plan = plan, AfterLoad = afterLoad,
        AfterDispose = afterDispose, Records = records, NativeShutdown = shutdown,
        RequestedOptions = options, Steps = steps, Repeats = repeats, ModelGeometry = geometry, Prompt = prompt,
        Generation = teacher == null ? "raw-greedy" : "teacher-forced", Teacher = teacher, TeacherSha256 = teacherHash,
        TeacherConvention = "Exactly steps prediction rows; only teacher[0..steps-2] are consumed. The final teacher ID is retained but not forwarded.",
        CaptureLogits = captureLogits, LogitCapture = captured,
        Scope = "ForwardRefill and Forward wall time only. Warmup separate. Raw predictions, not a language-quality test. Host geometry is a forecast; budget callbacks do not cover process RSS, OS cache, driver/backend internal pools or every native executor.",
        TimingQualification = captureLogits ? "Correctness only: full-logit capture I/O is outside forward timers but perturbs the process; not quiet throughput." : "Capture disabled; no full-logit capture I/O.",
        ModelsAssemblySha256 = Hash(typeof(ModelBase).Assembly.Location),
        ProbeAssemblyPath = typeof(LogitCapture).Assembly.Location,
        ProbeAssemblySha256 = Hash(typeof(LogitCapture).Assembly.Location),
        ManagedAssembliesSha256 = AppDomain.CurrentDomain.GetAssemblies()
            .Where(assembly => !assembly.IsDynamic && assembly.GetName().Name?.StartsWith("TensorSharp", StringComparison.Ordinal) == true)
            .OrderBy(assembly => assembly.GetName().Name)
            .ToDictionary(assembly => assembly.GetName().Name!, assembly => Hash(assembly.Location)),
        Native = NativeIdentity()
    }, new JsonSerializerOptions { WriteIndented = true }));
}
return error == null ? 0 : 1;

object MemorySample()
{
    using var process = Process.GetCurrentProcess();
    bool known = GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total);
    return new { WorkingSet = process.WorkingSet64, PeakWorkingSet = process.PeakWorkingSet64,
        DeviceKnown = known, DeviceTotal = total, DeviceFree = free, Pools = session?.Budget.Snapshot() };
}
static string Hash(string file) { using var s = File.OpenRead(file); return Convert.ToHexString(SHA256.HashData(s)); }
static object NativeIdentity()
{
    using var p = Process.GetCurrentProcess();
    var modules = p.Modules.Cast<ProcessModule>().Where(m => m.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase));
    return modules.Select(m => new { m.FileName, Sha256 = Hash(m.FileName) }).ToArray();
}
