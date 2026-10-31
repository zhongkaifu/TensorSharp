// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Compare identical token histories, avoiding autoregressive divergence masking
// small or large numerical errors. Artifacts belong in ignored validation paths.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length < 3)
{
    Console.Error.WriteLine("Usage: ForcedLogitProbe MODEL OUTPUT_DIR (BACKEND | --backend BACKEND) [--reference REFERENCE_DIR] [--steps N] [--cases cases.json] [--run-id ID]");
    Console.Error.WriteLine("  BACKEND: ggml_cuda | ggml_cpu | ggml_metal | cuda (direct CUDA) | cpu (managed)");
    Console.Error.WriteLine("  Without --reference the run is greedy (its tokens become the reference's forced tokens) and");
    Console.Error.WriteLine("  metrics.json records each case's generated ids and text. metrics.json carries complete=false");
    Console.Error.WriteLine("  until every case has finished; exit 0 = finished within the tight gate, 1 = finished outside it,");
    Console.Error.WriteLine("  anything else = did not finish.");
    return 2;
}
string modelPath = args[0], directory = Path.GetFullPath(args[1]);
// Teacher forcing makes the comparison independent of where greedy decoding of either side
// would wander, and of TS_DUMP_LOGITS (the first-forward dump): every row is compared.
string backendName = Option("--backend") ?? args[2];
BackendType backend = backendName switch
{
    "ggml_cuda" => BackendType.GgmlCuda, "ggml_cpu" => BackendType.GgmlCpu,
    "ggml_metal" => BackendType.GgmlMetal, "cuda" => BackendType.Cuda, "cpu" => BackendType.Cpu,
    _ => throw new ArgumentException("Unknown backend " + backendName + " (ggml_cuda, ggml_cpu, ggml_metal, cuda, cpu)")
};
string? Option(string name)
{
    int at = Array.IndexOf(args, name);
    return at < 0 ? null : at + 1 < args.Length ? args[at + 1] : throw new ArgumentException("Missing " + name);
}
int steps = int.Parse(Option("--steps") ?? "24");
if (steps < 1 || steps > 1024) throw new ArgumentOutOfRangeException("--steps", "Use 1..1024 steps; at least16 for qualification.");
string? referenceDir = Option("--reference");
// Lets a driver tell this run's output from a previous run's (it also deletes the directory first).
string? runId = Option("--run-id");
var json = new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, PropertyNameCaseInsensitive = true };
Directory.CreateDirectory(directory);
// A metrics.json left by an earlier run must never be read as this run's (a crash before the
// first case would otherwise leave it in place).
File.Delete(Path.Combine(directory, "metrics.json"));
File.Delete(Path.Combine(directory, "cases.json"));
KvCacheDtypeConfig.ConfigureFromEnvironment();
var loadTimer = Stopwatch.StartNew();
using var model = ModelBase.Create(modelPath, backend);
loadTimer.Stop();
ProbeCase[] cases;
if (referenceDir != null)
    cases = JsonSerializer.Deserialize<ProbeCase[]>(File.ReadAllText(Path.Combine(referenceDir, "cases.json")), json)!;
else if (Option("--cases") is string file)
    cases = JsonSerializer.Deserialize<ProbeCase[]>(File.ReadAllText(file), json)!;
else
{
    string Chat(string text) => "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    cases = [
        new() { Name = "arithmetic", Prompt = Chat("What is 7 times 8? State the answer, then explain multiplication in one sentence.") },
        new() { Name = "science", Prompt = Chat("Explain the water cycle in three short sentences for a child.") },
        new() { Name = "code", Prompt = Chat("Write a Python function that returns the first n Fibonacci numbers. Use a loop.") },
        new() { Name = "synthetic_single", PromptTokens = [1] },
        new() { Name = "synthetic_prefill128", PromptTokens = Enumerable.Range(0, 128).Select(i => 100 + i % 17).ToArray() },
    ];
}
var results = new List<object>();
var all = new RowStats();
bool allFinite = true;
bool qualified = true;
int casesDone = 0;
void WriteMetrics(bool complete) =>
    File.WriteAllText(Path.Combine(directory, "metrics.json"), JsonSerializer.Serialize(new { all_finite = allFinite,
        qualified = referenceDir == null ? (bool?)null : qualified,
        // A driver gates only on complete == true: the file is rewritten after every case, so a
        // run that dies part-way leaves the cases it finished, marked incomplete.
        complete, cases_done = casesDone, cases_total = cases.Length, run_id = runId,
        gate = new { max_relative_l2 = 0.001, min_cosine = 0.999999, require_identical_top1 = true },
        model = Path.GetFileName(modelPath), backend = backendName, tp = Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"),
        layer_split = Environment.GetEnvironmentVariable("TENSORSHARP_LAYER_SPLIT_DEGREE"), load_seconds = loadTimer.Elapsed.TotalSeconds,
        reference = referenceDir,
        summary = referenceDir == null ? null : all.Summarize(),
        cases = results }, json));
foreach (var item in cases)
{
    if (Path.GetFileName(item.Name) != item.Name || string.IsNullOrWhiteSpace(item.Name)) throw new ArgumentException("Invalid case name");
    item.PromptTokens ??= model.Tokenizer.Encode(item.Prompt ?? throw new ArgumentException("Missing prompt"), addSpecial: false).ToArray();
    if (item.PromptTokens.Length == 0) throw new ArgumentException("Empty prompt");
    // A case may ask for an exact prompt length (512, 2048, 8192 for the prefill gates): the
    // encoded text is repeated, then cut. cases.json records the resulting tokens, so a
    // --reference run replays exactly them.
    if (item.TargetTokens is int target && target > 0 && item.PromptTokens.Length != target)
        item.PromptTokens = Enumerable.Range(0, target).Select(i => item.PromptTokens[i % item.PromptTokens.Length]).ToArray();
    int count = referenceDir != null ? item.GeneratedTokens!.Length : steps;
    int[] generated = new int[count];
    string binaryPath = Path.Combine(directory, item.Name + ".f32");
    using var actualFile = File.Create(binaryPath);
    using FileStream? referenceFile = referenceDir == null ? null : File.OpenRead(Path.Combine(referenceDir, item.Name + ".f32"));
    model.ResetKVCache();
    float[] logits = model.Forward(item.PromptTokens);
    var metrics = new List<object>();
    var caseStats = new RowStats();
    for (int step = 0; step < count; ++step)
    {
        if (logits.Length != model.Config.VocabSize) throw new InvalidOperationException("Wrong vocabulary width");
        actualFile.Write(MemoryMarshal.AsBytes(logits.AsSpan()));
        bool finite = logits.All(float.IsFinite);
        allFinite &= finite;
        int top = Top(logits);
        int forced = referenceDir == null ? top : item.GeneratedTokens![step];
        generated[step] = forced;
        if (referenceFile != null)
        {
            float[] expected = new float[logits.Length];
            referenceFile.ReadExactly(MemoryMarshal.AsBytes(expected.AsSpan()));
            int referenceTop = Top(expected);
            int second = Enumerable.Range(0, expected.Length).Where(i => i != referenceTop).MaxBy(i => expected[i]);
            double maxAbs = 0, squaredError = 0, dot = 0, normReference = 0, normActual = 0;
            int topRank = 1;
            for (int i = 0; i < logits.Length; ++i)
            {
                double x = logits[i], y = expected[i], delta = x - y;
                maxAbs = Math.Max(maxAbs, Math.Abs(delta)); squaredError += delta * delta;
                dot += x * y; normActual += x * x; normReference += y * y;
                if (logits[i] > logits[referenceTop]) ++topRank;
            }
            double relativeL2 = Math.Sqrt(squaredError / Math.Max(normReference, 1e-300));
            double cosine = dot / Math.Sqrt(Math.Max(normReference * normActual, 1e-300));
            double kl = finite ? KlDivergence(expected, logits) : double.NaN;
            double top5 = Top5Overlap(expected, logits);
            bool rowFinite = finite && double.IsFinite(relativeL2) && double.IsFinite(kl);
            qualified &= rowFinite && relativeL2 <= 0.001 && cosine >= 0.999999 && referenceTop == top;
            caseStats.Add(step, rowFinite, relativeL2, kl, top5, referenceTop == top);
            all.Add(step, rowFinite, relativeL2, kl, top5, referenceTop == top);
            metrics.Add(new { step, finite, reference_top = referenceTop, candidate_top = top,
                top1_equal = referenceTop == top, reference_top_rank_in_candidate = topRank,
                reference_margin = Finite(expected[referenceTop] - expected[second]),
                reference_margin_to_candidate_top = Finite(expected[referenceTop] - expected[top]),
                candidate_margin_to_reference_top = Finite(logits[top] - logits[referenceTop]),
                max_abs = Finite(maxAbs), rms = Finite(Math.Sqrt(squaredError / logits.Length)),
                relative_l2 = Finite(relativeL2), cosine = Finite(cosine), kl = Finite(kl), top5_overlap = top5 });
        }
        else
        {
            int second = Enumerable.Range(0, logits.Length).Where(i => i != top).MaxBy(i => logits[i]);
            metrics.Add(new { step, finite, top, margin = Finite(logits[top] - logits[second]) });
        }
        if (step + 1 < count) logits = model.Forward([forced]);
    }
    if (referenceFile != null && referenceFile.Position != referenceFile.Length) throw new InvalidOperationException("Reference logit width/step mismatch");
    item.GeneratedTokens = generated;
    string? text = null;
    if (referenceDir == null)
    {
        try { text = model.Tokenizer.Decode(generated.ToList()); }
        catch (Exception ex) { text = "<decode failed: " + ex.GetType().Name + ">"; }
    }
    results.Add(new { name = item.Name, prompt_tokens = item.PromptTokens.Length, steps = count,
        summary = referenceFile == null ? null : caseStats.Summarize(),
        generated_tokens = referenceDir == null ? generated : null, generated_text = text, metrics });
    casesDone++;
    WriteMetrics(complete: false);
    Console.WriteLine($"[forced-logits] {item.Name}: {count} rows x {model.Config.VocabSize} logits, finite={allFinite}");
}
File.WriteAllText(Path.Combine(directory, "cases.json"), JsonSerializer.Serialize(cases, json));
WriteMetrics(complete: true);
return allFinite && qualified ? 0 : 1;

// System.Text.Json refuses NaN and infinities: a non-finite value is written as null (and the
// row is counted in non_finite_rows), never silently dropped from a statistic.
static double? Finite(double value) => double.IsFinite(value) ? value : null;

// KL(reference || candidate) of the two softmaxes, in double.
static double KlDivergence(float[] reference, float[] candidate)
{
    double maxR = double.NegativeInfinity, maxC = double.NegativeInfinity;
    for (int i = 0; i < reference.Length; ++i) { maxR = Math.Max(maxR, reference[i]); maxC = Math.Max(maxC, candidate[i]); }
    double sumR = 0, sumC = 0;
    for (int i = 0; i < reference.Length; ++i) { sumR += Math.Exp(reference[i] - maxR); sumC += Math.Exp(candidate[i] - maxC); }
    double logZr = maxR + Math.Log(sumR), logZc = maxC + Math.Log(sumC), kl = 0;
    for (int i = 0; i < reference.Length; ++i)
    {
        double logP = reference[i] - logZr;
        double p = Math.Exp(logP);
        if (p > 0) kl += p * (logP - (candidate[i] - logZc));
    }
    return Math.Max(0, kl);
}

// |top-5(reference) n top-5(candidate)| / 5.
static double Top5Overlap(float[] reference, float[] candidate)
{
    static int[] Top5(float[] v) => Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).ThenBy(i => i).Take(5).ToArray();
    return Top5(reference).Intersect(Top5(candidate)).Count() / 5.0;
}

static int Top(float[] values)
{
    int top = 0;
    for (int i = 1; i < values.Length; ++i) if (values[i] > values[top]) top = i;
    return top;
}

// What the direct-CUDA prefill gates (G3a/G3b) read, per case and overall: the step-0 row (the
// prompt's first-token logits) and the teacher-forced rows' median and worst relative L2 and
// KL(ref || x), top-1 mismatches and mean top-5 overlap. Non-finite rows are counted, never
// sorted into a median.
internal sealed class RowStats
{
    private readonly List<double> _rel = new(), _kl = new(), _top5 = new();
    private readonly List<double> _step0Rel = new(), _step0Kl = new();
    private int _rows, _nonFinite, _top1Mismatches, _step0Top1Mismatches;

    public void Add(int step, bool finite, double relativeL2, double kl, double top5, bool top1Equal)
    {
        _rows++;
        if (!top1Equal) _top1Mismatches++;
        if (step == 0 && !top1Equal) _step0Top1Mismatches++;
        _top5.Add(top5);
        if (!finite) { _nonFinite++; return; }
        _rel.Add(relativeL2);
        _kl.Add(kl);
        if (step == 0) { _step0Rel.Add(relativeL2); _step0Kl.Add(kl); }
    }

    public object Summarize() => new
    {
        rows = _rows,
        non_finite_rows = _nonFinite,
        median_relative_l2 = Median(_rel),
        max_relative_l2 = _rel.Count == 0 ? (double?)null : _rel.Max(),
        median_kl = Median(_kl),
        max_kl = _kl.Count == 0 ? (double?)null : _kl.Max(),
        top1_mismatches = _top1Mismatches,
        mean_top5_overlap = _top5.Count == 0 ? (double?)null : _top5.Average(),
        step0_rows = _step0Rel.Count,
        step0_median_relative_l2 = Median(_step0Rel),
        step0_max_relative_l2 = _step0Rel.Count == 0 ? (double?)null : _step0Rel.Max(),
        step0_median_kl = Median(_step0Kl),
        step0_max_kl = _step0Kl.Count == 0 ? (double?)null : _step0Kl.Max(),
        step0_top1_mismatches = _step0Top1Mismatches,
    };

    private static double? Median(List<double> values)
    {
        if (values.Count == 0) return null;
        double[] s = values.OrderBy(v => v).ToArray();
        return s.Length % 2 == 1 ? s[s.Length / 2] : 0.5 * (s[s.Length / 2 - 1] + s[s.Length / 2]);
    }
}

internal sealed class ProbeCase
{
    public string Name { get; set; } = "";
    public string? Prompt { get; set; }
    public int[]? PromptTokens { get; set; }
    public int? TargetTokens { get; set; }
    public int[]? GeneratedTokens { get; set; }
}
