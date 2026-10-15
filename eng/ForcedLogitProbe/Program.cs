// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Compare identical token histories, avoiding autoregressive divergence masking
// small or large numerical errors. Artifacts belong in ignored validation paths.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length < 3)
{
    Console.Error.WriteLine("Usage: ForcedLogitProbe MODEL OUTPUT_DIR BACKEND [--reference REFERENCE_DIR] [--steps N] [--cases cases.json]");
    return 2;
}
string modelPath = args[0], directory = Path.GetFullPath(args[1]);
BackendType backend = args[2] switch
{
    "ggml_cuda" => BackendType.GgmlCuda, "ggml_cpu" => BackendType.GgmlCpu,
    "ggml_metal" => BackendType.GgmlMetal, _ => throw new ArgumentException("Unknown backend")
};
string? Option(string name)
{
    int at = Array.IndexOf(args, name);
    return at < 0 ? null : at + 1 < args.Length ? args[at + 1] : throw new ArgumentException("Missing " + name);
}
int steps = int.Parse(Option("--steps") ?? "24");
if (steps < 1 || steps > 1024) throw new ArgumentOutOfRangeException("--steps", "Use 1..1024 steps; at least16 for qualification.");
string? referenceDir = Option("--reference");
var json = new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, PropertyNameCaseInsensitive = true };
Directory.CreateDirectory(directory);
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
bool allFinite = true;
bool qualified = true;
foreach (var item in cases)
{
    if (Path.GetFileName(item.Name) != item.Name || string.IsNullOrWhiteSpace(item.Name)) throw new ArgumentException("Invalid case name");
    item.PromptTokens ??= model.Tokenizer.Encode(item.Prompt ?? throw new ArgumentException("Missing prompt"), addSpecial: false).ToArray();
    if (item.PromptTokens.Length == 0) throw new ArgumentException("Empty prompt");
    int count = referenceDir != null ? item.GeneratedTokens!.Length : steps;
    int[] generated = new int[count];
    string binaryPath = Path.Combine(directory, item.Name + ".f32");
    using var actualFile = File.Create(binaryPath);
    using FileStream? referenceFile = referenceDir == null ? null : File.OpenRead(Path.Combine(referenceDir, item.Name + ".f32"));
    model.ResetKVCache();
    float[] logits = model.Forward(item.PromptTokens);
    var metrics = new List<object>();
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
            qualified &= finite && relativeL2 <= 0.001 && cosine >= 0.999999 && referenceTop == top;
            metrics.Add(new { step, finite, reference_top = referenceTop, candidate_top = top,
                top1_equal = referenceTop == top, reference_top_rank_in_candidate = topRank,
                reference_margin = expected[referenceTop] - expected[second],
                reference_margin_to_candidate_top = expected[referenceTop] - expected[top],
                candidate_margin_to_reference_top = logits[top] - logits[referenceTop],
                max_abs = maxAbs, rms = Math.Sqrt(squaredError / logits.Length),
                relative_l2 = relativeL2, cosine });
        }
        else
        {
            int second = Enumerable.Range(0, logits.Length).Where(i => i != top).MaxBy(i => logits[i]);
            metrics.Add(new { step, finite, top, margin = logits[top] - logits[second] });
        }
        if (step + 1 < count) logits = model.Forward([forced]);
    }
    if (referenceFile != null && referenceFile.Position != referenceFile.Length) throw new InvalidOperationException("Reference logit width/step mismatch");
    item.GeneratedTokens = generated;
    int requestedRanks = int.Parse(Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE") ?? "1");
    var cacheMemory = Enumerable.Range(0, requestedRanks).Select(rank =>
    {
        bool available = GgmlBasicOps.TryGetCacheMemoryUsage(rank, out var usage);
        return new { rank, available, usage };
    }).ToArray();
    results.Add(new { name = item.Name, prompt_tokens = item.PromptTokens.Length, steps = count, metrics,
        cache_memory = cacheMemory });
    File.WriteAllText(Path.Combine(directory, "metrics.json"), JsonSerializer.Serialize(new { all_finite = allFinite,
        qualified = referenceDir == null ? (bool?)null : qualified,
        gate = new { max_relative_l2 = 0.001, min_cosine = 0.999999, require_identical_top1 = true },
        model = Path.GetFileName(modelPath), backend = args[2], tp = Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"),
        layer_split = Environment.GetEnvironmentVariable("TENSORSHARP_LAYER_SPLIT_DEGREE"), load_seconds = loadTimer.Elapsed.TotalSeconds,
        reference = referenceDir, cases = results }, json));
    Console.WriteLine($"[forced-logits] {item.Name}: {count} rows x {model.Config.VocabSize} logits, finite={allFinite}");
}
File.WriteAllText(Path.Combine(directory, "cases.json"), JsonSerializer.Serialize(cases, json));
return allFinite && qualified ? 0 : 1;

static int Top(float[] values)
{
    int top = 0;
    for (int i = 1; i < values.Length; ++i) if (values[i] > values[top]) top = i;
    return top;
}

internal sealed class ProbeCase
{
    public string Name { get; set; } = "";
    public string? Prompt { get; set; }
    public int[]? PromptTokens { get; set; }
    public int[]? GeneratedTokens { get; set; }
}
