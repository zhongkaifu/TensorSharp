// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MultiModelUnload KEPT OTHER BACKEND [--prompt P] [--steps T] [--other-steps U]
//
// Two models in one process, one unloaded while the other is mid-generation: does the one kept loaded carry on exactly?
//   1. uninterrupted: KEPT alone, a pseudo-random prompt of P tokens, then T greedy decode steps (each feeding back its host
//      argmax, lowest id on ties); every step's logits digested (FNV-1a over the bits);
//   2. interrupted: KEPT again, the same prompt and the first T / 2 steps; then OTHER is loaded, runs its own prompt and U
//      decode steps, and is disposed; then KEPT carries on with its remaining steps on the same cache (no reset).
// Prints the STEP digests of both runs and SAME or DIFFERS; exits 0 only when every step of run 2 equals run 1's, bit for
// bit. KEPT and OTHER may be the same file (two models of one GGUF). Any family ModelBase.Create loads, any backend.
using System.Globalization;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length < 3) return Usage();
BackendType backend = args[2] switch
{
    "ggml_metal" => BackendType.GgmlMetal, "ggml_cpu" => BackendType.GgmlCpu, "cpu" => BackendType.Cpu, "mlx" => BackendType.Mlx,
    "ggml_cuda" => BackendType.GgmlCuda, "ggml_vulkan" => BackendType.GgmlVulkan, "cuda" => BackendType.Cuda,
    _ => throw new ArgumentException("unknown backend " + args[2]),
};
string Option(string name, string fallback)
{
    int i = Array.IndexOf(args, name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
}
int promptLength = int.Parse(Option("--prompt", "96"), CultureInfo.InvariantCulture);
int steps = int.Parse(Option("--steps", "32"), CultureInfo.InvariantCulture);
int otherSteps = int.Parse(Option("--other-steps", "8"), CultureInfo.InvariantCulture);
if (promptLength < 1 || steps < 2 || otherSteps < 0) return Usage();
KvCacheDtypeConfig.ConfigureFromEnvironment();

static int[] Prompt(int vocab, int length, int seed)
{
    int range = Math.Max(1, Math.Min(vocab, 60000) - 100);
    var tokens = new int[length];
    uint x = (uint)(0x9E3779B9u * (uint)(seed + 1));
    for (int i = 0; i < length; i++)
    {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        tokens[i] = 100 + (int)(x % (uint)range);
    }
    return tokens;
}

static int Argmax(float[] logits)
{
    int best = 0;
    for (int i = 1; i < logits.Length; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

// Run `count` decode steps from `first`, appending each step's digest and token; returns the next token.
static int Decode(ModelBase model, int first, int count, List<ulong> digests, List<int> tokens)
{
    int next = first;
    for (int s = 0; s < count; s++)
    {
        float[] logits = model.Forward(new[] { next });
        digests.Add(Fnv.Add(Fnv.Start, logits));
        tokens.Add(next = Argmax(logits));
    }
    return next;
}

var clock = System.Diagnostics.Stopwatch.StartNew();
List<ulong> want = new(), got = new();
List<int> wantTokens = new(), gotTokens = new();
int vocab;
using (ModelBase alone = ModelBase.Create(args[0], backend))
{
    vocab = alone.Config.VocabSize;
    alone.ResetKVCache();
    float[] logits = alone.ForwardPrompt(Prompt(vocab, promptLength, 1));
    want.Add(Fnv.Add(Fnv.Start, logits));
    Decode(alone, Argmax(logits), steps, want, wantTokens);
}
Console.WriteLine($"uninterrupted: {Path.GetFileName(args[0])} on {backend}, prompt {promptLength}, {steps} steps ({clock.Elapsed.TotalSeconds:F1} s)");

using (ModelBase kept = ModelBase.Create(args[0], backend))
{
    kept.ResetKVCache();
    float[] logits = kept.ForwardPrompt(Prompt(vocab, promptLength, 1));
    got.Add(Fnv.Add(Fnv.Start, logits));
    int next = Decode(kept, Argmax(logits), steps / 2, got, gotTokens);
    var otherClock = System.Diagnostics.Stopwatch.StartNew();
    string otherName;
    using (ModelBase other = ModelBase.Create(args[1], backend))
    {
        otherName = $"{other.Config.Architecture} {Path.GetFileName(args[1])}";
        other.ResetKVCache();
        float[] o = other.ForwardPrompt(Prompt(other.Config.VocabSize, promptLength, 2));
        Decode(other, Argmax(o), otherSteps, new List<ulong>(), new List<int>());
    }
    Console.WriteLine($"other: {otherName} loaded, prompt {promptLength} and {otherSteps} steps, disposed ({otherClock.Elapsed.TotalSeconds:F1} s)");
    Decode(kept, next, steps - steps / 2, got, gotTokens);
}

bool same = want.Count == got.Count;
for (int i = 0; i < Math.Max(want.Count, got.Count); i++)
{
    string w = i < want.Count ? want[i].ToString("X16") : "-", g = i < got.Count ? got[i].ToString("X16") : "-";
    bool equal = w == g;
    same &= equal;
    Console.WriteLine($"STEP {i,3} {(i == 0 ? "prompt" : $"token {(i - 1 < wantTokens.Count ? wantTokens[i - 1] : -1)}")} uninterrupted {w} interrupted {g}{(equal ? "" : "  DIFFERS")}{(i == steps / 2 + 1 ? "  (after the other's dispose)" : "")}");
}
Console.WriteLine(same ? "SAME: every step of the interrupted run is the uninterrupted run's, bit for bit" : "DIFFERS");
return same ? 0 : 1;

static int Usage()
{
    Console.Error.WriteLine("usage: MultiModelUnload KEPT OTHER ggml_cuda|ggml_vulkan|ggml_metal|ggml_cpu|cuda|cpu|mlx [--prompt P] [--steps T] [--other-steps U]");
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

    public static ulong Add(ulong h, float[] values)
    {
        foreach (float f in values) h = Add(h, BitConverter.SingleToUInt32Bits(f));
        return h;
    }
}
