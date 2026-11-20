// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// BatchDigest MODEL BACKEND [--sequences N,...] [--prompt P] [--steps T]
//
// The bits of batched serving steps, for before/after proofs of commits that claim to be bit-identical (the batched-serving
// design's stages): any StandardDecoderModel family (LFM3, Qwen3.5, Hunyuan dense, Mistral3, ...) on any backend, driven
// through IUnifiedBatchModel as the server's scheduler drives it. Per N:
//   - N slots, sequence s with a pseudo-random prompt of P + 3s tokens run alone (prompt steps);
//   - T batched decode steps of the N sequences, each feeding back its host argmax (lowest id on ties);
//   - for N >= 2, halfway through, sequence 1 leaves and a new sequence joins: its prompt runs in the same step as the
//     others' decode rows (a mixed step, as the server batches chunks with decode rows), then it decodes with them;
//   - the same schedule again (the same prompts, after every logits cell) through TryForwardUnifiedGreedy where the model
//     returns device greedy ids: its tokens equal the logits cell's when every device id is the host argmax.
// Prints an FNV-1a digest of every logit bit per step and per cell, the tokens' digest, and the graph counters (runs, builds,
// decode recordings and recorded replays, read by reflection so the tool builds against any commit). Two builds whose steps
// claim the same arithmetic print the same CELL / STEP / ALL lines: `diff <(grep -E '^(CELL|STEP|ALL)' a) <(grep ... b)`.
// Uses only public API (plus those counters), so it builds unchanged at the parent commit of a stage.
using System.Globalization;
using System.Reflection;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

if (args.Length < 2) return Usage();
BackendType backend = args[1] switch
{
    "ggml_metal" => BackendType.GgmlMetal, "ggml_cpu" => BackendType.GgmlCpu, "cpu" => BackendType.Cpu, "mlx" => BackendType.Mlx,
    "ggml_cuda" => BackendType.GgmlCuda, "ggml_vulkan" => BackendType.GgmlVulkan, "cuda" => BackendType.Cuda,
    _ => throw new ArgumentException("unknown backend " + args[1]),
};
string Option(string name, string fallback)
{
    int i = Array.IndexOf(args, name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
}
int[] sizes = Option("--sequences", "1,3,16").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
int promptLength = int.Parse(Option("--prompt", "37"), CultureInfo.InvariantCulture);
int steps = int.Parse(Option("--steps", "24"), CultureInfo.InvariantCulture);
if (sizes.Length == 0 || sizes.Any(n => n < 1) || promptLength < 1 || steps < 2) return Usage();

KvCacheDtypeConfig.ConfigureFromEnvironment();
using ModelBase model = ModelBase.Create(args[0], backend);
if (model is not StandardDecoderModel runtime || model is not IUnifiedBatchModel unified || !unified.SupportsUnifiedBatching)
{
    Console.Error.WriteLine($"{model.Config.Architecture} on {backend} does not batch over slots (IUnifiedBatchModel)");
    return 1;
}
int vocab = model.Config.VocabSize;
int tokenRange = Math.Max(1, Math.Min(vocab, 60000) - 100);
Console.WriteLine(FormattableString.Invariant(
    $"BatchDigest: {model.Config.Architecture} {Path.GetFileName(args[0])} on {backend}; KV {model.KvCacheDtype}; vocab {vocab}; prompts {promptLength} + 3s tokens; {steps} decode steps; device ids {(unified.SupportsUnifiedGreedyIds ? "yes" : "no")}"));

ulong all = Fnv.Start;
int serial = 0;
// Every logits cell first, then the device-id cells: a build whose executor gained device ids runs the same logits cells
// in the same order as its parent (their lines compare), and each id cell repeats its logits cell's prompts and schedule,
// so its tokens equal that cell's exactly when every device id is the host argmax.
foreach (int n in sizes) RunCell(n, deviceIds: false);
if (unified.SupportsUnifiedGreedyIds)
    foreach (int n in sizes) RunCell(n, deviceIds: true);
Console.WriteLine($"ALL {all:X16}");
return 0;

void RunCell(int n, bool deviceIds)
{
    int cell = serial++;
    string kind = deviceIds ? "ids" : "logits";
    var ids = Enumerable.Range(0, n).Select(s => $"d{cell}-{s}").ToList();
    var starts = new List<int>();
    var next = new List<int>();
    int reserve = promptLength + 3 * (n + 1) + steps + 8;
    ulong prompts = Fnv.Start, decode = Fnv.Start, tokens = Fnv.Start;
    long runs0 = Counter("GraphRuns"), builds0 = Counter("GraphBuilds"), rec0 = Counter("DecodeRecordings"), replays0 = Counter("RecordedDecodeReplays");
    string mixed = "-";
    bool joined = false;
    var released = new HashSet<string>();
    try
    {
        int[] Prompt(int s) => Enumerable.Range(0, promptLength + 3 * s)
            .Select(i => 100 + (int)((uint)((i + 11) * 2654435761u + (uint)s * 40503u + (uint)n * 97u) % (uint)tokenRange)).ToArray();
        for (int s = 0; s < n; s++)
        {
            unified.EnsureSequenceSlot(ids[s], reserve);
            int[] p = Prompt(s);
            var one = new float[1][];
            if (!unified.TryForwardUnified(new[] { ids[s] }, new[] { p }, new[] { 0 }, one))
                throw new InvalidOperationException($"the prompt of sequence {s} of {n} declined");
            prompts = Fnv.Add(prompts, one[0]);
            starts.Add(p.Length);
            next.Add(ArgMax(one[0]));
        }
        for (int t = 0; t < steps; t++)
        {
            var rowTokens = new List<int[]>();
            for (int s = 0; s < ids.Count; s++) rowTokens.Add(new[] { next[s] });
            if (n >= 2 && t == steps / 2 && !joined)
            {
                joined = true;
                // Sequence 1 leaves; a new one joins, its prompt in this step beside the others' decode rows.
                runtime.OnSequenceReleased(ids[1]);
                released.Add(ids[1]);
                ids.RemoveAt(1); starts.RemoveAt(1); next.RemoveAt(1); rowTokens.RemoveAt(1);
                string joiner = $"d{cell}-{n}";
                unified.EnsureSequenceSlot(joiner, reserve);
                ids.Add(joiner); starts.Add(0); next.Add(-1);
                rowTokens.Add(Prompt(n));
                mixed = "mixed";
            }
            int count = ids.Count;
            var stepStarts = starts.ToArray();
            ulong step = Fnv.Start;
            int[] chosen = new int[count];
            bool ok;
            if (deviceIds)
            {
                ok = unified.TryForwardUnifiedGreedy(ids, rowTokens, stepStarts, chosen);
                if (ok) foreach (int c in chosen) step = Fnv.Add(step, (uint)c);
            }
            else
            {
                var logits = new float[count][];
                ok = unified.TryForwardUnified(ids, rowTokens, stepStarts, logits);
                if (ok)
                    for (int s = 0; s < count; s++)
                    {
                        step = Fnv.Add(step, logits[s]);
                        chosen[s] = ArgMax(logits[s]);
                    }
            }
            if (!ok && rowTokens[^1].Length > 1)
            {
                // The model declined the mixed step: the joiner's prompt runs alone, then the step as plain decode rows.
                mixed = "declined";
                var one = new float[1][];
                if (!unified.TryForwardUnified(new[] { ids[^1] }, new[] { rowTokens[^1] }, new[] { 0 }, one))
                    throw new InvalidOperationException("the joiner's prompt declined");
                prompts = Fnv.Add(prompts, one[0]);
                starts[^1] = rowTokens[^1].Length;
                next[^1] = ArgMax(one[0]);
                t--;
                continue;
            }
            if (!ok) throw new InvalidOperationException($"batched step {t} of {count} sequences declined");
            for (int s = 0; s < count; s++)
            {
                starts[s] += rowTokens[s].Length;
                next[s] = chosen[s];
                tokens = Fnv.Add(tokens, (uint)chosen[s]);
            }
            decode = Fnv.Add(decode, step);
            Console.WriteLine(FormattableString.Invariant($"STEP {kind} n={n} t={t} rows={count} {step:X16} next={string.Join(',', chosen)}"));
        }
    }
    finally
    {
        foreach (string id in ids) if (!released.Contains(id)) runtime.OnSequenceReleased(id);
    }
    all = Fnv.Add(Fnv.Add(Fnv.Add(all, prompts), decode), tokens);
    Console.WriteLine(FormattableString.Invariant(
        $"CELL {kind} n={n} join={mixed} prompts={prompts:X16} decode={decode:X16} tokens={tokens:X16} runs={Counter("GraphRuns") - runs0} builds={Counter("GraphBuilds") - builds0} recordings={Counter("DecodeRecordings") - rec0} recorded_replays={Counter("RecordedDecodeReplays") - replays0}"));
}

int ArgMax(float[] logits)
{
    int best = 0;
    for (int v = 1; v < vocab; v++) if (logits[v] > logits[best]) best = v;
    return best;
}

long Counter(string name)
{
    PropertyInfo? p = typeof(StandardDecoderModel).GetProperty(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
    return p?.GetValue(runtime) is int v ? v : -1;
}

static int Usage()
{
    Console.Error.WriteLine("usage: BatchDigest MODEL BACKEND [--sequences N,...] [--prompt P] [--steps T]   (BACKEND: ggml_cuda ggml_vulkan ggml_cpu ggml_metal cuda cpu mlx)");
    return 2;
}

static class Fnv
{
    public const ulong Start = 14695981039346656037UL;

    public static ulong Add(ulong h, uint v)
    {
        for (int b = 0; b < 4; b++, v >>= 8) h = (h ^ (v & 0xFF)) * 1099511628211UL;
        return h;
    }

    public static ulong Add(ulong h, ulong v) => Add(Add(h, (uint)v), (uint)(v >> 32));

    public static ulong Add(ulong h, float[] values)
    {
        foreach (float f in values) h = Add(h, BitConverter.SingleToUInt32Bits(f));
        return h;
    }
}
