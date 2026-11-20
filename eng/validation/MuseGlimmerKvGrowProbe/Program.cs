// Muse-Glimmer KV-cache grow / sliding-window-ring probe.
//
// Drives one Muse-Glimmer model through a prompt and a decode, recording the logits
// of every step, so two processes that differ only in cache configuration can be
// compared position by position (compare.py). See README.md.
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

string Option(string name, string? fallback)
{
    int index = Array.IndexOf(args, name);
    if (index < 0) return fallback!;
    if (index + 1 >= args.Length) throw new ArgumentException($"Missing value for {name}.");
    return args[index + 1];
}

string modelPath = Option("--model", Environment.GetEnvironmentVariable("TS_TEST_MUSE_GLIMMER_MODEL"));
string backendName = Option("--backend", "ggml_metal");
string scenario = Option("--scenario", "decode");
string output = Option("--out", "artifacts/muse-glimmer-kv-grow/report.json");
string? promptFile = Option("--prompt-file", null);
string? systemFile = Option("--system-file", null);
string? teacherFile = Option("--teacher", null);
string? logitsOut = Option("--logits-out", null);
// --nll-text: teacher-force a fixed raw text (no chat template) and report its perplexity. The first
// --nll-prefix tokens are prefilled in one forward; every later token is scored from the logits of the
// one-row forward before it (the decode path), so a run crosses the same cache grows a generation does.
string? nllText = Option("--nll-text", null);
int nllPrefix = int.Parse(Option("--nll-prefix", "64"));
int steps = int.Parse(Option("--steps", nllText != null ? "0" : "64"));
int split = int.Parse(Option("--split", "0"));
int topK = int.Parse(Option("--topk", "8"));
int tp = int.Parse(Option("--tp", "1"));

if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
    throw new ArgumentException("Pass --model <Muse-Glimmer gguf> (or set TS_TEST_MUSE_GLIMMER_MODEL).");
if (!Enum.TryParse(backendName.Replace("_", string.Empty), ignoreCase: true, out BackendType backend))
    throw new ArgumentException($"Unknown backend '{backendName}'.");

var load = Stopwatch.StartNew();
// Released explicitly (and the backend shut down) at the end: ggml-metal asserts at
// process exit if any buffer is still registered, as the server's ProcessExit hook knows.
var model = ModelBase.Create(modelPath, backend, tpDegree: tp);
double loadMs = load.Elapsed.TotalMilliseconds;
if (model.Config.Architecture != "muse-glimmer")
    throw new InvalidOperationException($"Expected a muse-glimmer model, got {model.Config.Architecture}.");

// ---- prompt ------------------------------------------------------------------
int[] prompt;
int[]? teacher = null;
if (nllText != null)
{
    Require(teacherFile == null && promptFile == null && systemFile == null && scenario == "decode",
        "--nll-text scores its own text: it takes no --teacher, --prompt-file or --system-file, and runs the decode scenario.");
    int[] all = model.Tokenizer.Encode(File.ReadAllText(nllText), addSpecial: true).ToArray();
    Require(nllPrefix >= 1 && nllPrefix < all.Length, $"--nll-prefix must be inside the text's {all.Length} tokens.");
    prompt = all[..nllPrefix];
    teacher = all[nllPrefix..];
    steps = steps <= 0 ? teacher.Length : Math.Min(steps, teacher.Length);
}
else
{
    string userText = promptFile != null ? File.ReadAllText(promptFile) : "请详细介绍最终幻想7";
    var messages = new List<ChatMessage>();
    if (systemFile != null)
        messages.Add(new ChatMessage { Role = "system", Content = File.ReadAllText(systemFile) });
    messages.Add(new ChatMessage { Role = "user", Content = userText });
    string rendered = ChatTemplate.RenderFromGgufTemplate(model.Config.ChatTemplate, messages,
        addGenerationPrompt: true, architecture: model.Config.Architecture);
    prompt = model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
}

if (teacherFile != null)
{
    using var doc = JsonDocument.Parse(File.ReadAllText(teacherFile));
    teacher = doc.RootElement.GetProperty("Steps").EnumerateArray()
        .Select(s => s.GetProperty("Chosen").GetInt32()).ToArray();
    int[] teacherPrompt = doc.RootElement.GetProperty("PromptTokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();
    if (!teacherPrompt.SequenceEqual(prompt))
        throw new InvalidOperationException("The teacher report was produced from a different prompt.");
    steps = Math.Min(steps, teacher.Length);
}

Console.WriteLine($"[probe] scenario={scenario} backend={backend} tp={tp} prompt={prompt.Length} tokens steps={steps} split={split} " +
    $"capacity={Field<int>(model, "_kvCacheCapacity")} swaRows={Field<int>(model, "_kvSwaRows")}");

// ---- prefill -----------------------------------------------------------------
var events = new List<string>();
var prefill = Stopwatch.StartNew();
float[] logits;
model.ResetKVCache();
switch (scenario)
{
    case "decode":
        logits = model.Forward(prompt);
        break;
    case "split":
        // Two forwards: the second one starts below the initial capacity and ends past it.
        Require(split > 0 && split < prompt.Length, "--split must be inside the prompt.");
        model.Forward(prompt[..split]);
        events.Add($"after first forward: seq={model.CacheSeqLen} capacity={Field<int>(model, "_kvCacheCapacity")}");
        logits = model.Forward(prompt[split..]);
        break;
    case "snapshot":
    {
        // What a pooled radix-cache hit does: extract whole 256-token blocks, start a
        // fresh cache, inject them, and forward only the remainder.
        Require(split >= 256 && split < prompt.Length, "--split must cover at least one block and stay inside the prompt.");
        int blocks = split / 256;
        int reused = blocks * 256;
        model.Forward(prompt);
        long blockBytes = model.ComputeKVBlockByteSize(256);
        var saved = new List<byte[]>();
        for (int b = 0; b < blocks; b++)
        {
            var buf = new byte[blockBytes];
            Require(model.TryExtractKVBlock(b * 256, 256, buf), $"TryExtractKVBlock refused block {b}.");
            saved.Add(buf);
        }
        // Scramble the same rows first: a reset keeps the device copies on the GPU
        // backends, so a restore that wrote nothing would otherwise read them back.
        model.ResetKVCache();
        model.Forward(prompt.Reverse().ToArray());
        model.ResetKVCache();
        for (int b = 0; b < blocks; b++)
            Require(model.TryInjectKVBlock(b * 256, 256, saved[b]), $"TryInjectKVBlock refused block {b}.");
        events.Add($"injected {reused} tokens in {blocks} blocks of {blockBytes} bytes; seq={model.CacheSeqLen}");
        logits = model.Forward(prompt[reused..]);
        break;
    }
    case "truncate":
    {
        // What a live-cache rewind does: keep the head, drop the tail, re-forward it.
        Require(split > 0 && split < prompt.Length, "--split must be inside the prompt.");
        model.Forward(prompt);
        Require(model.TryTruncateKVCache(split), $"TryTruncateKVCache({split}) refused.");
        events.Add($"truncated to {model.CacheSeqLen}");
        logits = model.Forward(prompt[split..]);
        break;
    }
    default:
        throw new ArgumentException($"Unknown scenario '{scenario}'.");
}
double prefillMs = prefill.Elapsed.TotalMilliseconds;

// ---- decode ------------------------------------------------------------------
var records = new List<StepRecord>(steps);
FileStream? logitsStream = logitsOut != null ? File.Create(logitsOut) : null;
int vocab = model.Config.VocabSize;
double nllSum = 0;
var decode = Stopwatch.StartNew();
for (int i = 0; i < steps; i++)
{
    int position = model.CacheSeqLen;      // the position the next token will occupy
    var top = TopK(logits, Math.Min(vocab, logits.Length), topK);
    int chosen = teacher != null ? teacher[i] : top[0].Id;
    if (nllText != null) nllSum -= LogSoftmaxAt(logits, Math.Min(vocab, logits.Length), chosen);
    records.Add(new StepRecord(i, position, top[0].Id, chosen, top, Field<int>(model, "_kvCacheCapacity")));
    if (logitsStream != null) WriteHalf(logitsStream, logits, vocab);
    if (i + 1 < steps)
        logits = model.Forward(new[] { chosen });
}
double decodeMs = decode.Elapsed.TotalMilliseconds;
logitsStream?.Dispose();

string text = model.Tokenizer.Decode(records.Select(r => r.Chosen).ToList());
object? nll = nllText == null ? null : new
{
    Text = Path.GetFullPath(nllText),
    Prefix = nllPrefix,
    ScoredTokens = steps,
    MeanNll = steps > 0 ? nllSum / steps : double.NaN,
    Perplexity = steps > 0 ? Math.Exp(nllSum / steps) : double.NaN,
    Top1Agreement = steps > 0 ? records.Count(r => r.Argmax == r.Chosen) / (double)steps : double.NaN,
};
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
File.WriteAllText(output, JsonSerializer.Serialize(new
{
    Scenario = scenario,
    Backend = backend.ToString(),
    TensorParallel = tp,
    Model = Path.GetFullPath(modelPath),
    Environment = new
    {
        MAX_CONTEXT = Environment.GetEnvironmentVariable("MAX_CONTEXT"),
        TS_KV_INITIAL_TOKENS = Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS"),
        TS_MUSE_GLIMMER_SWA_RING = Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_SWA_RING"),
        KV_CACHE_DTYPE = Environment.GetEnvironmentVariable("KV_CACHE_DTYPE"),
    },
    KvCacheDtype = model.KvCacheDtype.ToString(),
    SwaRows = Field<int>(model, "_kvSwaRows"),
    FinalCapacity = Field<int>(model, "_kvCacheCapacity"),
    ModelLoadMilliseconds = loadMs,
    PrefillMilliseconds = prefillMs,
    DecodeMilliseconds = decodeMs,
    DecodeTokensPerSecond = steps > 1 ? (steps - 1) / (decodeMs / 1000.0) : 0,
    Split = split,
    Events = events,
    Nll = nll,
    PromptTokens = prompt,
    GeneratedText = text,
    Steps = records,
}, new JsonSerializerOptions { WriteIndented = false }));
Console.WriteLine($"[probe] prefill {prefillMs:F0} ms, decode {steps} steps in {decodeMs:F0} ms; wrote {output}");
if (nllText != null && steps > 0)
    Console.WriteLine($"[probe] nll: {steps} tokens after a {nllPrefix}-token prefix, mean NLL {nllSum / steps:F5}, " +
        $"perplexity {Math.Exp(nllSum / steps):F4}, top-1 agreement {records.Count(r => r.Argmax == r.Chosen) / (double)steps:P2}");
Console.WriteLine($"[probe] text: {Truncate(text.Replace('\n', ' '), 400)}");
model.Dispose();
if (backend is BackendType.GgmlMetal or BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlCpu)
    TensorSharp.GGML.GgmlBasicOps.Shutdown();

static T Field<T>(object target, string name)
{
    for (var type = target.GetType(); type != null; type = type.BaseType)
    {
        var field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        if (field != null) return (T)field.GetValue(target)!;
    }
    return default!;
}

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

static List<TopEntry> TopK(float[] values, int n, int k)
{
    var best = new List<TopEntry>(k + 1);
    for (int i = 0; i < n; i++)
    {
        float v = values[i];
        if (best.Count == k && v <= best[^1].Logit) continue;
        int at = best.Count;
        while (at > 0 && best[at - 1].Logit < v) at--;
        best.Insert(at, new TopEntry(i, v));
        if (best.Count > k) best.RemoveAt(best.Count - 1);
    }
    return best;
}

// log softmax(values[0..n))[id], in double.
static double LogSoftmaxAt(float[] values, int n, int id)
{
    double max = double.NegativeInfinity;
    for (int i = 0; i < n; i++) if (values[i] > max) max = values[i];
    double sum = 0;
    for (int i = 0; i < n; i++) sum += Math.Exp(values[i] - max);
    return values[id] - max - Math.Log(sum);
}

static void WriteHalf(Stream stream, float[] values, int n)
{
    var bytes = new byte[n * 2];
    for (int i = 0; i < n; i++)
        BitConverter.TryWriteBytes(bytes.AsSpan(i * 2, 2), (System.Half)values[i]);
    stream.Write(bytes);
}

static string Truncate(string s, int n) => s.Length <= n ? s : s[..n] + "...";

record TopEntry(int Id, float Logit);
record StepRecord(int Step, int Position, int Argmax, int Chosen, List<TopEntry> Top, int Capacity);
