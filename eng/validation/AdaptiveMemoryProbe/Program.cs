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
Directory.CreateDirectory(output);
// Identity is outside every measured interval. Reading the file also means this
// probe cannot be described as a cold-storage benchmark.
string modelHash = Hash(path);
string mode = options.GetValueOrDefault("--mode", "adaptive");
if (mode is not ("resident" or "adaptive")) throw new ArgumentException("--mode resident|adaptive");
int steps = int.Parse(options.GetValueOrDefault("--steps", "64"));
int repeats = int.Parse(options.GetValueOrDefault("--repeats", "3"));
int minimumPrompt = int.Parse(options.GetValueOrDefault("--prompt-tokens", "640"));
int context = int.Parse(options.GetValueOrDefault("--context", "2048"));
if (steps < 2 || repeats < 1 || minimumPrompt < 1 || context <= minimumPrompt + steps)
    throw new ArgumentException("Require positive runs and a context larger than prompt plus decode.");
Environment.SetEnvironmentVariable("MAX_CONTEXT", context.ToString());
Environment.SetEnvironmentVariable("KV_CACHE_DTYPE", "f16");
KvCacheDtypeConfig.ConfigureFromEnvironment();
foreach (string variable in new[] { "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TS_SPEC", "SPECULATIVE_DECODING", "TS_MTP" })
    Environment.SetEnvironmentVariable(variable, null);

var records = new List<object>();
AdaptiveModelSession? session = null;
ModelBase? model = null;
object? plan = null, afterLoad = null, afterDispose = null;
string? error = null;
int[] prompt = [];
double loadMs = 0;
try
{
    var timer = Stopwatch.StartNew();
    if (mode == "adaptive")
    {
        session = AdaptiveModelSession.Create(path, new(context, Math.Min(context, minimumPrompt + 64))
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
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    string text = "Count upwards from 10 to 1000, separated by spaces. Output only the numbers.";
    do
    {
        prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate,
            [new ChatMessage { Role = "user", Content = text }], model.Config.Architecture,
            addGenerationPrompt: true, enableThinking: false).ToArray();
        if (prompt.Length < minimumPrompt) text += " Include every consecutive integer without gaps.";
    } while (prompt.Length < minimumPrompt);
    if (prompt.Length + steps > context) throw new ArgumentException("Rendered prompt exceeds admitted context.");
    File.WriteAllText(Path.Combine(output, "prompt.json"), JsonSerializer.Serialize(prompt));

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
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (int step = 0; step < steps; step++)
        {
            if (!logits.All(float.IsFinite)) throw new ArithmeticException("Nonfinite full-vocabulary logits.");
            hash.AppendData(MemoryMarshal.AsBytes(logits.AsSpan()));
            generated[step] = Enumerable.Range(0, logits.Length).MaxBy(i => logits[i]);
            if (step + 1 == steps) break;
            timer.Restart();
            logits = model.Forward([generated[step]]);
            decodeMs += timer.Elapsed.TotalMilliseconds;
        }
        records.Add(new
        {
            Run = run, Warmup = run < 0, PromptTokens = prompt.Length, DecodeCalls = steps - 1,
            PrefillMilliseconds = prefillMs, DecodeMilliseconds = decodeMs,
            PrefillTokensPerSecond = prompt.Length * 1000 / prefillMs,
            DecodeTokensPerSecond = (steps - 1) * 1000 / decodeMs,
            LogitsSha256 = Convert.ToHexString(hash.GetHashAndReset()), Generated = generated,
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
    File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(new
    {
        Executed = error == null, Error = error, Mode = mode, Model = path,
        ModelBytes = new FileInfo(path).Length, ModelSha256 = modelHash,
        Context = context, LoadMilliseconds = loadMs, Plan = plan, AfterLoad = afterLoad,
        AfterDispose = afterDispose, Records = records,
        Scope = "ForwardRefill and Forward wall time only. Warmup separate. Raw greedy histories for full-logit parity, not a language-quality test. Host geometry is a forecast; budget callbacks do not cover process RSS, OS cache, driver/backend internal pools or every native executor.",
        ModelsAssemblySha256 = Hash(typeof(ModelBase).Assembly.Location), Native = NativeIdentity()
    }, new JsonSerializerOptions { WriteIndented = true }));
    if (error == null) GgmlBasicOps.Shutdown();
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
