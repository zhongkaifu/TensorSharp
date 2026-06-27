// Standalone single-sequence decode benchmark for ANY model class. Measures the
// fused N=1 decode path (ModelBase.Forward over a 1-token batch) that the
// server's BatchExecutor N=1 fast path uses for steady-state generation — i.e.
// the decode_tps column in the engine comparison report.
//
// Plain exe (no VSTest host) so nsys/ncu can attach to the CUDA work.
//
// Env knobs:
//   TS_DECODE_MODEL    path to a .gguf (default gemma-4-12B-it-qat-q4_0)
//   TS_DECODE_BACKEND  ggml_cuda (default) | ggml_cpu | cpu | cuda
//   TS_DECODE_PREFILL  prompt token count to prefill before decoding (default 64)
//   TS_DECODE_NEW      decode tokens to time (default 128)
//   TS_DECODE_ITERS    timed runs, median reported (default 3)
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

static int EnvInt(string name, int fallback)
{
    string s = Environment.GetEnvironmentVariable(name);
    return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
}

static int Argmax(float[] v)
{
    int best = 0;
    for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
    return best;
}

static double Median(List<double> xs)
{
    if (xs.Count == 0) return 0;
    var s = new List<double>(xs); s.Sort();
    int m = s.Count / 2;
    return s.Count % 2 == 1 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

string modelPath = Environment.GetEnvironmentVariable("TS_DECODE_MODEL")
    ?? @"C:\Works\models\gemma-4-12B-it-qat-q4_0.gguf";
BackendType backend = (Environment.GetEnvironmentVariable("TS_DECODE_BACKEND") ?? "ggml_cuda").ToLowerInvariant() switch
{
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_metal" => BackendType.GgmlMetal,
    "ggml_cpu" => BackendType.GgmlCpu,
    "cpu" => BackendType.Cpu,
    "cuda" => BackendType.Cuda,
    "mlx" => BackendType.Mlx,
    _ => BackendType.GgmlCuda,
};
int prefillLen = EnvInt("TS_DECODE_PREFILL", 64);
int newTokens = EnvInt("TS_DECODE_NEW", 128);
int iters = EnvInt("TS_DECODE_ITERS", 3);

Console.WriteLine($"[decode-bench] loading {Path.GetFileName(modelPath)} backend={backend} prefill={prefillLen} new={newTokens} iters={iters}");
using var model = ModelBase.Create(modelPath, backend);

string baseText =
    "The history of computing spans many centuries, beginning with simple counting tools " +
    "and culminating in the modern electronic computer. Along the way, mathematicians and " +
    "engineers developed algorithms, logic, and machines that transformed how humanity " +
    "stores, processes, and reasons about information. ";
int[] basePool = model.Tokenizer.Encode(baseText, addSpecial: false).ToArray();
if (basePool.Length == 0) throw new Exception("empty tokenization");

int[] MakePrompt(int n, int salt)
{
    var t = new int[n];
    for (int i = 0; i < n; i++) t[i] = basePool[(i + salt) % basePool.Length];
    return t;
}

long g_lastHash = 0;

// One run: prefill `prefillLen` tokens then greedily decode `newTokens`,
// timing only the decode loop. Returns decode tok/s.
double RunOnce(int salt, out double prefillMs)
{
    model.ResetKVCache();
    var prompt = MakePrompt(prefillLen, salt);
    var swP = Stopwatch.StartNew();
    float[] logits = model.ForwardRefill(prompt);
    swP.Stop();
    prefillMs = swP.Elapsed.TotalMilliseconds;

    int t = Argmax(logits);
    long hash = 1469598103934665603L; // FNV-1a over the greedy token stream (correctness check)
    var sw = Stopwatch.StartNew();
    for (int i = 1; i < newTokens; i++)
    {
        logits = model.Forward(new[] { t });
        t = Argmax(logits);
        hash = (hash ^ (uint)t) * 1099511628211L;
    }
    sw.Stop();
    g_lastHash = hash;
    return (newTokens - 1) / sw.Elapsed.TotalSeconds;
}

Console.WriteLine("[decode-bench] warming up...");
_ = RunOnce(0, out _);
_ = RunOnce(1, out _);
Console.WriteLine($"[decode-bench] greedy token-stream hash (fold-correctness): {g_lastHash:X16}");

var tps = new List<double>();
var pf = new List<double>();
for (int it = 0; it < iters; it++)
{
    double r = RunOnce(100 + it * 7, out double pms);
    tps.Add(r); pf.Add(pms);
    Console.WriteLine($"  run {it}: prefill {pms,8:F1} ms   decode {r,7:F2} tok/s");
}
Console.WriteLine();
Console.WriteLine($"[decode-bench] MEDIAN decode = {Median(tps):F2} tok/s   prefill = {Median(pf):F1} ms ({prefillLen / (Median(pf) / 1000.0):F0} tok/s)");
model.PrintTimingStats();
Console.WriteLine("[decode-bench] done.");
