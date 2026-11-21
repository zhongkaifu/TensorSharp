// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The real Inkling 2 Mini checkpoint (<c>INKLING2_MODEL</c>: a GGUF of ggml-org/Inkling-2-Mini-GGUF) against vLLM's outputs
/// on the official checkpoint (<c>INKLING2_VLLM_REFERENCE</c>: the JSON <c>eng/inkling2/capture_vllm_reference.py</c> writes).
/// Each prompt is teacher-forced a token at a time: at every position the model's top-1 against vLLM's, and its
/// log-probabilities of vLLM's top-5 tokens against vLLM's. Then 32 greedy tokens against vLLM's greedy continuation.
/// The backend is <c>INKLING2_BACKEND</c> (default ggml_cuda). vLLM ran the NVFP4/MXFP8 checkpoint with an FP8 KV cache;
/// the GGUF is the same weights at Q8_0, so the comparison is close, not exact.
/// </summary>
public sealed class Inkling2CheckpointTests
{
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public Inkling2CheckpointTests(Xunit.Abstractions.ITestOutputHelper output) => _output = output;

    [ModelFact("INKLING2_MODEL")]
    public void TracksTheVllmReference()
    {
        string model = Environment.GetEnvironmentVariable("INKLING2_MODEL");
        string referencePath = Environment.GetEnvironmentVariable("INKLING2_VLLM_REFERENCE");
        Assert.False(string.IsNullOrEmpty(referencePath), "INKLING2_VLLM_REFERENCE names the vLLM reference JSON");
        BackendType backend = Enum.TryParse(Environment.GetEnvironmentVariable("INKLING2_BACKEND") ?? "GgmlCuda", true, out BackendType b)
            ? b : BackendType.GgmlCuda;
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(referencePath));
        using var m = new Inkling2RuntimeModel(model, backend);

        int positions = 0, top1 = 0, greedyCases = 0, greedyExact = 0;
        var logprobErrors = new List<double>();
        var missMargins = new List<double>();
        double maxChunkRel = 0;
        int caseIndex = -1;
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            caseIndex++;
            int[] ids = c.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            JsonElement[] top5 = c.GetProperty("prompt_top5").EnumerateArray().ToArray();
            m.ResetKVCache();
            int caseTop1 = 0, casePositions = 0;
            float[] logits = null;
            for (int i = 0; i < ids.Length; i++)
            {
                logits = (float[])m.Forward(new[] { ids[i] }).Clone();
                if (i + 1 >= ids.Length || top5[i + 1].GetArrayLength() == 0) continue;
                double[] logp = LogSoftmax(logits);
                var expected = top5[i + 1].EnumerateArray().Select(e => (Id: e[0].GetInt32(), Lp: e[1].GetDouble())).ToArray();
                if (ArgMax(logits) == expected[0].Id) caseTop1++;
                else if (expected.Length > 1)
                {
                    missMargins.Add(expected[0].Lp - expected[1].Lp);
                    if (expected[0].Lp - expected[1].Lp > 1)
                    {
                        int mine = ArgMax(logits);
                        _output.WriteLine($"  miss in case {caseIndex} after token {i} (of {ids.Length}): vLLM " +
                            string.Join(" ", expected.Select(x => $"{x.Id}:{x.Lp:F2}/{logp[x.Id]:F2}")) + $"; TensorSharp's top-1 {mine}:{logp[mine]:F2}");
                    }
                }
                foreach (var (id, lp) in expected) logprobErrors.Add(Math.Abs(logp[id] - lp));
                casePositions++;
            }
            // The prompt as one chunk (the prompt-chunk attention) against the token-at-a-time decode rows.
            float[] last = logits;
            m.ResetKVCache();
            float[] refill = (float[])m.ForwardRefill(ids).Clone();
            double chunkRel = RelL2(refill, last);
            maxChunkRel = Math.Max(maxChunkRel, chunkRel);
            string crossBackend = CompareWithOtherBackends(backend, caseIndex, last, refill);
            logits = refill;
            // Greedy continuation from the prompt's last logits.
            int[] reference = c.GetProperty("completion_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var generated = new List<int>();
            for (int t = 0; t < reference.Length; t++)
            {
                int next = ArgMax(logits);
                generated.Add(next);
                logits = (float[])m.Forward(new[] { next }).Clone();
            }
            int agree = 0;
            while (agree < reference.Length && generated[agree] == reference[agree]) agree++;
            greedyCases++;
            if (agree == reference.Length) greedyExact++;
            positions += casePositions;
            top1 += caseTop1;
            _output.WriteLine($"{c.GetProperty("kind").GetString()} {ids.Length} tokens: top-1 {caseTop1}/{casePositions}, greedy agrees for {agree}/{reference.Length}, chunk vs decode relL2 {chunkRel:E1}{crossBackend}");
        }
        logprobErrors.Sort();
        double median = logprobErrors[logprobErrors.Count / 2], p95 = logprobErrors[(int)(logprobErrors.Count * 0.95)];
        _output.WriteLine($"{backend}: top-1 {top1}/{positions}; |logprob - vLLM| over vLLM's top-5: median {median:F4}, p95 {p95:F4}; " +
            $"greedy 32 tokens exact in {greedyExact}/{greedyCases} cases; chunk vs decode relL2 max {maxChunkRel:E1}");
        missMargins.Sort();
        if (missMargins.Count > 0)
            _output.WriteLine($"vLLM's own top-1 over top-2 margin where the top-1 differs: median {missMargins[missMargins.Count / 2]:F3}, " +
                $"max {missMargins[^1]:F3} nats ({string.Join(", ", missMargins.Select(m => m.ToString("F2")))})");
        Assert.True(top1 >= positions * 0.9, $"top-1 agreement {top1}/{positions}");
    }

    private static double RelL2(float[] x, float[] reference)
    {
        double num = 0, den = 0;
        for (int v = 0; v < reference.Length; v++) { num += (x[v] - reference[v]) * (double)(x[v] - reference[v]); den += (double)reference[v] * reference[v]; }
        return Math.Sqrt(num / den);
    }

    /// <summary>
    /// With <c>INKLING2_DUMP_DIR</c> set, writes this backend's last prompt logits (token-at-a-time and as one chunk) and
    /// compares them with any other backend's from an earlier run (one GGML backend a process).
    /// </summary>
    private static string CompareWithOtherBackends(BackendType backend, int caseIndex, float[] decode, float[] chunk)
    {
        string dir = Environment.GetEnvironmentVariable("INKLING2_DUMP_DIR");
        if (string.IsNullOrEmpty(dir)) return "";
        Directory.CreateDirectory(dir);
        static void Write(string path, float[] x)
        {
            var bytes = new byte[x.Length * sizeof(float)];
            Buffer.BlockCopy(x, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(path, bytes);
        }
        static float[] Read(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var x = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, x, 0, bytes.Length);
            return x;
        }
        // INKLING2_DUMP_TAG names this run beside its backend (a fused and an unfused run of one backend, say).
        string self = backend + (Environment.GetEnvironmentVariable("INKLING2_DUMP_TAG") ?? "");
        Write(Path.Combine(dir, $"{self}-case{caseIndex}-decode.f32"), decode);
        Write(Path.Combine(dir, $"{self}-case{caseIndex}-chunk.f32"), chunk);
        var parts = new List<string>();
        foreach (string path in Directory.GetFiles(dir, $"*-case{caseIndex}-decode.f32"))
        {
            string other = Path.GetFileName(path)[..^$"-case{caseIndex}-decode.f32".Length];
            if (other == self) continue;
            float[] otherDecode = Read(path), otherChunk = Read(Path.Combine(dir, $"{other}-case{caseIndex}-chunk.f32"));
            parts.Add($"vs {other}: decode {RelL2(decode, otherDecode):E1}, chunk {RelL2(chunk, otherChunk):E1}, " +
                $"argmax {(ArgMax(decode) == ArgMax(otherDecode) ? "same" : "differs")}");
        }
        return parts.Count == 0 ? "" : "; " + string.Join("; ", parts);
    }

    private static int ArgMax(float[] x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    private static double[] LogSoftmax(float[] x)
    {
        double max = x.Max(), sum = 0;
        foreach (float v in x) sum += Math.Exp(v - max);
        double log = Math.Log(sum) + max;
        return x.Select(v => v - log).ToArray();
    }
}
