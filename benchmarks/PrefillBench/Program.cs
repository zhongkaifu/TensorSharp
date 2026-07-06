// Standalone prefill (time-to-first-token) benchmark for the batched paged
// attention path. Measures how long the engine takes to process a prompt of N
// tokens through ForwardBatch before emitting the first decode token — i.e. the
// prefill latency that dominates TTFT. Runs single-sequence and concurrent
// scenarios so the per-sequence-graph vs batched-graph behaviour is visible.
//
// Plain exe (no VSTest host) so nsys/ncu can attach to the CUDA work, mirroring
// benchmarks/GdnDecodeBench.
//
// Env knobs:
//   TS_PREFILL_MODEL    path to a .gguf (default Ministral-3-8B Q8_0)
//   TS_PREFILL_BACKEND  ggml_cuda (default) | ggml_cpu | cpu | cuda
//   TS_PREFILL_LENS     comma list of prompt token counts (default 512,1024,2048,4096)
//   TS_PREFILL_ITERS    timed iterations per length (default 3)
//   TS_PREFILL_CONC     comma list of concurrency levels for a fixed 1024-token prompt (default 1,2,4)
//   TS_SCHED_PREFILL_CHUNK / TS_SCHED_MAX_BATCHED_TOKENS honoured by SchedulerConfig below.
using System.Diagnostics;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

static int EnvInt(string name, int fallback)
{
    string s = Environment.GetEnvironmentVariable(name);
    return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
}

static int[] EnvIntList(string name, int[] fallback)
{
    string s = Environment.GetEnvironmentVariable(name);
    if (string.IsNullOrEmpty(s)) return fallback;
    var parts = s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    var list = new List<int>();
    foreach (var p in parts)
        if (int.TryParse(p, out int v) && v > 0) list.Add(v);
    return list.Count > 0 ? list.ToArray() : fallback;
}

static double Median(List<double> xs)
{
    if (xs.Count == 0) return 0;
    var s = new List<double>(xs); s.Sort();
    int m = s.Count / 2;
    return s.Count % 2 == 1 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

string modelPath = Environment.GetEnvironmentVariable("TS_PREFILL_MODEL")
    ?? @"C:\Works\models\Ministral-3-8B-Instruct-2512-Q8_0.gguf";
BackendType backend = (Environment.GetEnvironmentVariable("TS_PREFILL_BACKEND") ?? "ggml_cuda").ToLowerInvariant() switch
{
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_vulkan" => BackendType.GgmlVulkan,
    "ggml_cpu" => BackendType.GgmlCpu,
    "cpu" => BackendType.Cpu,
    "cuda" => BackendType.Cuda,
    _ => BackendType.GgmlCuda,
};
int[] lens = EnvIntList("TS_PREFILL_LENS", new[] { 512, 1024, 2048, 4096 });
int[] concs = EnvIntList("TS_PREFILL_CONC", new[] { 1, 2, 4 });
int iters = EnvInt("TS_PREFILL_ITERS", 3);
// TS_PREFILL_LEGACY_ONLY=1 skips the (slow) correctness, engine and concurrent
// sections and only times the legacy ForwardRefill path — fast iteration when
// profiling the pure prefill compute with QWEN35_PREFILL_PROFILE=1.
bool legacyOnly = string.Equals(Environment.GetEnvironmentVariable("TS_PREFILL_LEGACY_ONLY"), "1", StringComparison.Ordinal);
// TS_PREFILL_ENGINE_ONLY=1 keeps only the engine/batched single-sequence section
// (the server prefill path) and skips correctness, the legacy ForwardRefill
// section and concurrent — clean isolation for nsys profiling of the fused
// verify prefill graph.
bool engineOnly = string.Equals(Environment.GetEnvironmentVariable("TS_PREFILL_ENGINE_ONLY"), "1", StringComparison.Ordinal);
int blockSize = 256;

Console.WriteLine($"[prefill-bench] loading {Path.GetFileName(modelPath)} backend={backend}");
using var model = ModelBase.Create(modelPath, backend);

// Build a base token pool we can slice prompts of arbitrary length from. Vary
// the leading token per iteration so prefix caching (if on) never shortcuts the
// measured prefill.
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

var cfg = new SchedulerConfig
{
    MaxNumBatchedTokens = EnvInt("TS_SCHED_MAX_BATCHED_TOKENS", 8192),
    MaxNumRunningSequences = 16,
    MaxPrefillChunkSize = EnvInt("TS_SCHED_PREFILL_CHUNK", 2048),
    NumBlocks = EnvInt("TS_SCHED_NUM_BLOCKS", 512),
    BlockSize = blockSize,
    EnablePrefixCaching = false, // force full prefill every run
    DecodeQuantumTokens = blockSize,
};
using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

// Submit one sequence and return the wall time until its FIRST token arrives
// (= prefill latency). maxNewTokens=1 keeps decode out of the measurement.
async Task<double> TtftMsAsync(int[] tokens, string id)
{
    var seq = new SequenceState(id, tokens, maxNewTokens: 1, blockSize, SamplingConfig.Default);
    var sw = Stopwatch.StartNew();
    var handle = engine.SubmitRequest(seq);
    await foreach (var _ in handle.Tokens.ReadAllAsync()) break;
    sw.Stop();
    try { await handle.Completion; } catch { }
    return sw.Elapsed.TotalMilliseconds;
}

// Concurrent: submit `n` sequences at once; return wall time until ALL have
// produced their first token (prefill throughput across the batch).
async Task<double> ConcurrentPrefillMsAsync(int n, int promptLen, int salt)
{
    var sw = Stopwatch.StartNew();
    var tasks = new List<Task>(n);
    for (int i = 0; i < n; i++)
    {
        var seq = new SequenceState($"c{salt}-{i}", MakePrompt(promptLen, salt + i * 13), 1, blockSize, SamplingConfig.Default);
        var handle = engine.SubmitRequest(seq);
        tasks.Add(Task.Run(async () =>
        {
            await foreach (var _ in handle.Tokens.ReadAllAsync()) break;
            try { await handle.Completion; } catch { }
        }));
    }
    await Task.WhenAll(tasks);
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds;
}

async Task<int> FirstTokenAsync(int[] tokens, string id)
{
    var seq = new SequenceState(id, tokens, maxNewTokens: 1, blockSize, SamplingConfig.Greedy);
    var handle = engine.SubmitRequest(seq);
    int tok = -1;
    await foreach (var t in handle.Tokens.ReadAllAsync()) { tok = t; break; }
    try { await handle.Completion; } catch { }
    return tok;
}

// Warm-up: loads CUDA kernels, JITs hot paths, allocates pools.
Console.WriteLine("[prefill-bench] warming up...");
_ = await TtftMsAsync(MakePrompt(256, 0), "warm");
_ = await TtftMsAsync(MakePrompt(256, 1), "warm2");

// Correctness: the batched (fused-FFN) path's first greedy token must match the
// legacy single-sequence ForwardRefill argmax for the same prompt. Run this on
// the real model + backend so the fused dense-FFN kernel is validated end-to-end.
if (!legacyOnly && !engineOnly)
{
    Console.WriteLine();
    Console.WriteLine("==== Correctness (batched fused vs legacy ForwardRefill) ====");
    bool allMatch = true;
    foreach (int len in new[] { 17, 64, 200, 777, 2500 })
    {
        int[] cp = MakePrompt(len, 31 + len);
        model.ResetKVCache();
        float[] lg = model.ForwardRefill(cp);
        int legacyTop1 = Argmax(lg);
        model.ResetKVCache();
        int batchedTop1 = await FirstTokenAsync(cp, $"corr-{len}");
        bool ok = legacyTop1 == batchedTop1;
        allMatch &= ok;
        Console.WriteLine($"  len={len,4}: legacy={legacyTop1,7} batched={batchedTop1,7}  {(ok ? "MATCH" : "*** MISMATCH ***")}");
    }
    model.ResetKVCache();
    Console.WriteLine($"  fusedDenseFFN={(Environment.GetEnvironmentVariable("TS_DISABLE_FUSED_DENSE_FFN") == "1" ? "DISABLED" : "enabled")}  result={(allMatch ? "ALL MATCH" : "MISMATCH!")}");
}

static int Argmax(float[] v)
{
    int best = 0;
    for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
    return best;
}

if (!legacyOnly)
{
Console.WriteLine();
Console.WriteLine("==== Single-sequence prefill (TTFT, engine/batched path = server) ====");
Console.WriteLine($"{"tokens",8} {"ms",10} {"tok/s",10}");
foreach (int len in lens)
{
    var samples = new List<double>();
    for (int it = 0; it < iters; it++)
        samples.Add(await TtftMsAsync(MakePrompt(len, 1000 + it * 7 + len), $"s{len}-{it}"));
    double ms = Median(samples);
    Console.WriteLine($"{len,8} {ms,10:F1} {len / (ms / 1000.0),10:F0}");
}
}

// Legacy ForwardRefill path = what the CLI uses for prompt prefill (and the
// server's per-sequence fallback). Times the model call directly, bypassing the
// engine, so the legacy-path FFN fusion is measured on its own.
if (!engineOnly)
{
Console.WriteLine();
Console.WriteLine("==== Single-sequence prefill (legacy ForwardRefill = CLI path) ====");
Console.WriteLine($"{"tokens",8} {"ms",10} {"tok/s",10}");
foreach (int len in lens)
{
    var samples = new List<double>();
    bool profLegacy = Environment.GetEnvironmentVariable("TS_PREFILL_PROFILE") == "1";
    for (int it = 0; it < iters; it++)
    {
        var p = MakePrompt(len, 2000 + it * 11 + len);
        model.ResetKVCache();
        if (profLegacy) model.ResetForwardTiming();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        _ = model.ForwardRefill(p);
        sw.Stop();
        samples.Add(sw.Elapsed.TotalMilliseconds);
        if (profLegacy) { Console.WriteLine($"-- legacy timing breakdown @ {len} tokens --"); model.PrintTimingStats(); }
    }
    model.ResetKVCache();
    double ms = Median(samples);
    string argmaxNote = "";
    if (Environment.GetEnvironmentVariable("TS_PREFILL_DUMP_ARGMAX") == "1")
    {
        model.ResetKVCache();
        float[] lg = model.ForwardRefill(MakePrompt(len, 99999));
        argmaxNote = $"  argmax={Argmax(lg)}";
        model.ResetKVCache();
    }
    Console.WriteLine($"{len,8} {ms,10:F1} {len / (ms / 1000.0),10:F0}{argmaxNote}");
}
} // !engineOnly (legacy section)

if (!legacyOnly && !engineOnly)
{
Console.WriteLine();
Console.WriteLine("==== Concurrent prefill (1024-token prompts, wall to all-first-token) ====");
Console.WriteLine($"{"n_seqs",8} {"ms",10} {"tok/s",12}");
foreach (int n in concs)
{
    var samples = new List<double>();
    for (int it = 0; it < iters; it++)
        samples.Add(await ConcurrentPrefillMsAsync(n, 1024, 5000 + it * 101 + n));
    double ms = Median(samples);
    Console.WriteLine($"{n,8} {ms,10:F1} {(n * 1024) / (ms / 1000.0),12:F0}");
}
}

Console.WriteLine();
Console.WriteLine("[prefill-bench] done.");
