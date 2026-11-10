// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Opt-in, on the real ISTA Q2_0 Qwen3.8-Flash-Next (TS_TEST_QWEN4EXP_MODEL) on <c>--backend cuda</c>: the prompt's
/// routed experts on the int8 kernels (TS_Q2_0_EXPERTS_INT8) against the F16 tensor-core kernel, on one loaded model.
/// Each prompt runs F16, int8, then F16 again (the control: what the same route gives twice). Reported: the
/// last prompt row's logits (relative L2, largest difference, KL, argmax) and the greedy continuation of each arm.
/// Run it with the history-dependent shortcuts off (TS_MOE_ROUTE_TAIL_SKIP=0 TS_MOE_TIER_ADAPT=0
/// TS_Q4E_HYBRID_MISSES=0), or the control itself differs.
/// </summary>
[Collection("Qwen4Exp MTP integration")]
public sealed class Qwen4ExpQ2_0Int8PromptTests(ITestOutputHelper output)
{
    private const int Continuation = 64;

    [ModelFact("TS_TEST_QWEN4EXP_MODEL")]
    public void Int8PromptExperts_AgainstTheF16Route_PromptLogitsAndGreedyText()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MODEL")!;
        using var env = new EnvScope();
        env.Set("MAX_CONTEXT", "32768");
        env.ClearSpeculationVars();
        string doc = File.ReadAllText(Path.Combine(DeepSeek4LoadStageDocumentationTests.RepoRoot(), "docs", "models", "qwen38-flash-next.md"));
        using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
        Assert.IsType<Qwen4ExpCudaModel>(model);
        int[] Chat(string user) => model.Tokenizer.Encode(
            "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: false).ToArray();
        int[] Document(int tokens, string question)
        {
            // The longest prefix of the document (cut at a line) whose prompt stays within the budget.
            int chars = doc.Length;
            int[] ids = Chat("Read this document.\n\n" + doc + "\n\n" + question);
            while (ids.Length > tokens)
            {
                chars = (int)(chars * (double)tokens / ids.Length * 0.98);
                int cut = doc.LastIndexOf('\n', Math.Min(chars, doc.Length - 1));
                ids = Chat("Read this document.\n\n" + doc[..Math.Max(cut, 1)] + "\n\n" + question);
            }
            return ids;
        }
        var prompts = new (string Name, int[] Ids)[]
        {
            ("sky", Chat("Explain why the sky appears blue and why sunsets are red.")),
            ("zh", Chat("请详细介绍《时间简史》，包括宇宙、黑洞和时间的主要观点。")),
            ("code", Chat("Write a Python function that merges two sorted lists into one sorted list, with a short docstring and two tests.")),
            ("math", Chat("A train leaves at 9:40 and arrives at 13:05 after covering 287 km. What was its average speed? Show the steps.")),
            ("doc4k", Document(4000, "In two sentences: what does the document say the expert tier does when a GPU cannot hold every expert?")),
            ("doc16k", Document(15000, "List three environment variables the document describes and what each one does.")),
        };

        bool saved = CudaMoe.Q2_0ExpertsInt8;
        var rows = new List<object>();
        try
        {
            (float[] Logits, int[] Text) Run(int[] prompt, bool int8, bool perSlot = false)
            {
                CudaMoe.Q2_0ExpertsInt8 = int8;
                CudaMoe.PerSlotForTest = perSlot;
                model.ResetKVCache();
                float[] logits = model.Forward(prompt);
                float[] first = (float[])logits.Clone();
                var text = new int[Continuation];
                for (int t = 0; t < Continuation; t++)
                {
                    text[t] = ArgMax(logits);
                    logits = model.Forward([text[t]]);
                }
                return (first, text);
            }

            foreach ((string name, int[] ids) in prompts)
            {
                var f16 = Run(ids, int8: false);
                var i8 = Run(ids, int8: true);
                var control = Run(ids, int8: false);
                // The prompt's experts on the per-slot decode kernels: q8_1 activations, ggml-cuda's MMQ arithmetic.
                var q81 = Run(ids, int8: false, perSlot: true);
                double relI8 = RelativeL2(i8.Logits, f16.Logits), relControl = RelativeL2(control.Logits, f16.Logits);
                double relQ81 = RelativeL2(q81.Logits, f16.Logits);
                double maxI8 = MaxAbs(i8.Logits, f16.Logits), klI8 = Kl(f16.Logits, i8.Logits);
                int sameI8 = SamePrefix(i8.Text, f16.Text), sameControl = SamePrefix(control.Text, f16.Text);
                int sameQ81 = SamePrefix(q81.Text, f16.Text);
                output.WriteLine($"{name} ({ids.Length} tokens): int8 vs F16 prompt logits relative L2 {relI8:E2}, " +
                    $"max |d| {maxI8:G4}, KL {klI8:E2}, argmax {ArgMax(i8.Logits)} vs {ArgMax(f16.Logits)}; " +
                    $"per-slot q8_1 vs F16 {relQ81:E2}; control relative L2 {relControl:E2}; greedy tokens identical " +
                    $"{sameI8}/{Continuation} (q8_1 {sameQ81}, control {sameControl})");
                rows.Add(new
                {
                    prompt = name, tokens = ids.Length,
                    int8_rel_l2 = relI8, int8_max_abs = maxI8, int8_kl = klI8,
                    int8_argmax_same = ArgMax(i8.Logits) == ArgMax(f16.Logits),
                    q81_rel_l2 = relQ81, q81_max_abs = MaxAbs(q81.Logits, f16.Logits),
                    q81_argmax_same = ArgMax(q81.Logits) == ArgMax(f16.Logits),
                    control_rel_l2 = relControl,
                    int8_greedy_same = sameI8, q81_greedy_same = sameQ81, control_greedy_same = sameControl,
                    continuation = Continuation,
                    f16_text = model.Tokenizer.Decode(f16.Text.ToList()), int8_text = model.Tokenizer.Decode(i8.Text.ToList()),
                    q81_text = model.Tokenizer.Decode(q81.Text.ToList()),
                });
            }
        }
        finally
        {
            CudaMoe.Q2_0ExpertsInt8 = saved;
            CudaMoe.PerSlotForTest = false;
            string report = Environment.GetEnvironmentVariable("TS_TEST_Q2I8_REPORT");
            if (!string.IsNullOrEmpty(report))
                File.WriteAllText(report, JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        }
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best])
                best = i;
        return best;
    }

    private static double RelativeL2(float[] got, float[] want)
    {
        double num = 0, den = 0;
        for (int i = 0; i < got.Length; i++)
        {
            double d = (double)got[i] - want[i];
            num += d * d;
            den += (double)want[i] * want[i];
        }
        return Math.Sqrt(num / den);
    }

    private static double MaxAbs(float[] a, float[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++)
            m = Math.Max(m, Math.Abs((double)a[i] - b[i]));
        return m;
    }

    private static int SamePrefix(int[] a, int[] b)
    {
        int n = 0;
        while (n < a.Length && n < b.Length && a[n] == b[n])
            n++;
        return n;
    }

    private static double Kl(float[] p, float[] q)
    {
        double maxP = p.Max(), maxQ = q.Max();
        double zP = p.Sum(x => Math.Exp(x - maxP)), zQ = q.Sum(x => Math.Exp(x - maxQ));
        double logZP = maxP + Math.Log(zP), logZQ = maxQ + Math.Log(zQ), sum = 0;
        for (int i = 0; i < p.Length; ++i)
            sum += Math.Exp(p[i] - logZP) * (p[i] - logZP - q[i] + logZQ);
        return Math.Max(0, sum);
    }
}
