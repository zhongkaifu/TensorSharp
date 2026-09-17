// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Does the SAME prompt decode to the same greedy tokens when its prefill is cut
// into different chunk sizes? The server prefills a lone request in one large
// chunk and concurrent requests in shares of the step budget, so a model whose
// logits depend on the prefill shape (batched GEMM kernels, MoE top-k over
// near-tied expert scores, GDN chunked recurrence) answers the same prompt
// differently under concurrency without any state being shared.
//
// The first chunking is the reference. Every other run is teacher-forced on the
// reference's tokens, so its logits stay comparable at every position; the
// report names each position where its own argmax differs, the reference's
// top-2 margin there and how far the logit vectors moved.
//
//   ChunkParityProbe <model.gguf> --prompt-file <utf8 text> [--chunks 4096,1024,512]
//                    [--new 128] [--backend ggml_cuda] [--repeat-first] [--out report.json]
//
// --repeat-first reruns the reference chunking once more (a determinism control:
// the same chunking must reproduce itself bit for bit). The prompt text is encoded
// as-is (no chat template), so pass already-templated text for a chat model.
// Layer split / TP comes from TENSORSHARP_TP_DEGREE as for every other bench.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

string modelPath = args.Length > 0 ? args[0] : throw new ArgumentException("usage: ChunkParityProbe <model.gguf> --prompt-file <file> [...]");
string promptFile = null, outPath = null;
int[] chunks = { 4096, 1024, 512 };
int newTokens = 128;
bool repeatFirst = false;
BackendType backend = BackendType.GgmlCuda;
for (int i = 1; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--prompt-file": promptFile = args[++i]; break;
        case "--chunks": chunks = args[++i].Split(',').Select(int.Parse).ToArray(); break;
        case "--new": newTokens = int.Parse(args[++i]); break;
        case "--out": outPath = args[++i]; break;
        case "--repeat-first": repeatFirst = true; break;
        case "--backend":
            backend = args[++i].ToLowerInvariant() switch
            {
                "ggml_cuda" => BackendType.GgmlCuda, "ggml_metal" => BackendType.GgmlMetal,
                "ggml_cpu" => BackendType.GgmlCpu, "cpu" => BackendType.Cpu, "cuda" => BackendType.Cuda,
                var other => throw new ArgumentException($"unknown backend {other}"),
            };
            break;
        default: throw new ArgumentException($"unknown argument {args[i]}");
    }
}
if (promptFile == null) throw new ArgumentException("--prompt-file is required");

using var model = ModelBase.Create(modelPath, backend);
int[] prompt = model.Tokenizer.Encode(File.ReadAllText(promptFile), addSpecial: false).ToArray();
Console.WriteLine($"[chunk-probe] prompt {prompt.Length} tokens, chunks [{string.Join(",", chunks)}], {newTokens} new tokens");

// Returns the logits after every generated position: row p is the distribution
// the token at output index p was drawn from.
List<float[]> Run(int chunk, IReadOnlyList<int> forced, out List<int> tokens, out double prefillMs)
{
    model.ResetKVCache();
    var sw = Stopwatch.StartNew();
    float[] logits = null;
    for (int start = 0; start < prompt.Length; start += chunk)
        logits = model.Forward(prompt.AsSpan(start, Math.Min(chunk, prompt.Length - start)).ToArray());
    prefillMs = sw.Elapsed.TotalMilliseconds;
    var rows = new List<float[]>(newTokens);
    tokens = new List<int>(newTokens);
    for (int p = 0; p < newTokens; p++)
    {
        rows.Add((float[])logits.Clone());
        int next = forced != null ? forced[p] : ArgMax(logits);
        tokens.Add(next);
        if (p + 1 < newTokens) logits = model.Forward(new[] { next });
    }
    return rows;
}

static int ArgMax(float[] v)
{
    int best = 0;
    for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
    return best;
}

static (int Top, float TopLogit, int Second, float SecondLogit) Top2(float[] v)
{
    int a = -1, b = -1;
    for (int i = 0; i < v.Length; i++)
    {
        if (a < 0 || v[i] > v[a]) { b = a; a = i; }
        else if (b < 0 || v[i] > v[b]) b = i;
    }
    return (a, v[a], b, v[b]);
}

var refRows = Run(chunks[0], null, out var refTokens, out double refPrefill);
string refText = model.Tokenizer.Decode(refTokens);
Console.WriteLine($"[chunk-probe] reference chunk={chunks[0]} prefill {refPrefill:F0} ms: {refText.Replace("\n", "\\n")[..Math.Min(160, refText.Length)]}");
var smallestMargins = Enumerable.Range(0, refRows.Count)
    .Select(p => { var t = Top2(refRows[p]); return (p, margin: t.TopLogit - t.SecondLogit); })
    .OrderBy(x => x.margin).Take(5).ToList();
Console.WriteLine($"[chunk-probe] reference's five smallest top-2 margins: {string.Join(", ", smallestMargins.Select(x => $"@{x.p}={x.margin:F4}"))}");

var report = new List<object>();
var variants = chunks.Skip(1).ToList();
if (repeatFirst) variants.Insert(0, chunks[0]);
foreach (int chunk in variants)
{
    var rows = Run(chunk, refTokens, out _, out double prefillMs);
    double maxAbs = 0, sumTopAbs = 0;
    var flips = new List<object>();
    for (int p = 0; p < rows.Count; p++)
    {
        var r = refRows[p]; var v = rows[p];
        double rowMax = 0;
        for (int i = 0; i < r.Length; i++) rowMax = Math.Max(rowMax, Math.Abs(r[i] - v[i]));
        maxAbs = Math.Max(maxAbs, rowMax);
        var rt = Top2(r);
        sumTopAbs += Math.Abs(r[rt.Top] - v[rt.Top]);
        int own = ArgMax(v);
        if (own != refTokens[p])
        {
            flips.Add(new
            {
                position = p,
                referenceToken = model.Tokenizer.Decode(new List<int> { refTokens[p] }),
                variantToken = model.Tokenizer.Decode(new List<int> { own }),
                referenceMargin = rt.TopLogit - rt.SecondLogit,
                referenceLogitOfVariantToken = r[own],
                variantGap = v[own] - v[refTokens[p]],
                rowMaxAbsDiff = rowMax,
            });
        }
    }
    string verdict = flips.Count == 0 ? "same greedy tokens" : $"{flips.Count} greedy flip(s), first at {((dynamic)flips[0]).position}";
    Console.WriteLine($"[chunk-probe] chunk={chunk} prefill {prefillMs:F0} ms: max|dlogit|={maxAbs:G4}, mean |dlogit| of the reference top-1={sumTopAbs / rows.Count:G4}; {verdict}");
    foreach (dynamic f in flips.Take(3))
        Console.WriteLine($"    @{f.position}: reference {JsonSerializer.Serialize((string)f.referenceToken)} (top-2 margin {f.referenceMargin:F4}) vs {JsonSerializer.Serialize((string)f.variantToken)} (wins by {f.variantGap:F4}; row max|d| {f.rowMaxAbsDiff:G4})");
    report.Add(new { chunk, prefillMs, maxAbsLogitDiff = maxAbs, meanTop1AbsDiff = sumTopAbs / rows.Count, flips });
}
if (outPath != null)
    File.WriteAllText(outPath, JsonSerializer.Serialize(new
    {
        model = Path.GetFileName(modelPath), promptTokens = prompt.Length, reference = chunks[0], referenceText = refText,
        smallestReferenceMargins = smallestMargins.Select(x => new { position = x.p, x.margin }), variants = report,
    }, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine("[chunk-probe] done");
