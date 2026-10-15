// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Opt-in complete-logit comparison on actual quantized model weights.
/// This tests prefill microgroup arithmetic; HTTP output quality is separate.</summary>
[Collection("Qwen4Exp MTP integration")]
public sealed class Qwen4ExpPrefillMicrogroupIntegrationTests(ITestOutputHelper output)
{
    private sealed record Phase(string Prompt, int Chunk, float[][] Logits, int[] Teacher,
        double PrefillSeconds, double DecodeSeconds, long CacheCalls);

    [ModelFact("TS_TEST_QWEN4EXP_MODEL")]
    public void ScalarAndPreplannedPrefill_HaveIdenticalCompleteLogitsAndContinuation()
    {
        using var env = new EnvScope();
        env.Set("MAX_CONTEXT", "512"); env.Set("TS_KV_INITIAL_TOKENS", "256");
        env.Set("TS_Q4E_DISABLE_ARENA_DECODE", null); env.ClearSpeculationVars();
        MoeCpuOffloadConfig.Reset(); MoeCpuOffloadConfig.SetAllLayers();
        string modelPath = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MODEL")!;
        var comparisons = new List<object>();
        bool completed = false;
        string? failure = null;
        try
        {
            using var model = Assert.IsType<Qwen4ExpModel>(ModelBase.Create(modelPath, TestGates.PinnedGgmlBackend));
            string[] texts = ["请详细介绍最终幻想7", "请详细介绍时间简史", "请详细介绍最终幻想8"];
            int[] pad = model.Tokenizer.Encode("Read this context carefully. ", addSpecial: false).ToArray();
            Assert.NotEmpty(pad);
            var cases = texts.SelectMany(text => new[] { 17, 32 }.Select(chunk =>
            {
                int[] suffix = model.Tokenizer.Encode("<|im_start|>user\n" + text
                    + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: false).ToArray();
                int[] tokens = Enumerable.Range(0, Math.Max(0, 129 - suffix.Length))
                    .Select(i => pad[i % pad.Length]).Concat(suffix).ToArray();
                return (Text: text, Chunk: chunk, Tokens: tokens);
            })).ToArray();
            var scalar = new List<Phase>();
            // Every model forward has completed and returned its logits before
            // a scope changes. Only expert caches are released; loaded model
            // weights and source registrations stay alive across this boundary.
            using (var scope = new GgmlHostMoeExpertCacheScope(4096L << 20, 32, model.Config.NumLayers, 1))
                foreach (var item in cases) scalar.Add(Run(item.Text, item.Chunk, item.Tokens, null));
            using (var scope = new GgmlHostMoeExpertCacheScope(4096L << 20, 32, model.Config.NumLayers, 3))
                for (int index = 0; index < cases.Length; ++index)
                {
                    var item = cases[index];
                    Phase expected = scalar[index];
                    Phase actual = Run(item.Text, item.Chunk, item.Tokens, expected.Teacher);
                    Assert.Equal(expected.Logits.Length, actual.Logits.Length);
                    for (int phase = 0; phase < actual.Logits.Length; ++phase)
                    {
                        float[] left = expected.Logits[phase], right = actual.Logits[phase];
                        Assert.Equal(left.Length, right.Length);
                        double maximum = 0;
                        for (int token = 0; token < left.Length; ++token)
                        {
                            Assert.True(float.IsFinite(left[token]) && float.IsFinite(right[token]),
                                "Prefill comparison contains nonfinite logits.");
                            maximum = Math.Max(maximum, Math.Abs((double)left[token] - right[token]));
                        }
                        comparisons.Add(new { prompt = item.Text, chunk = item.Chunk, phase,
                            vocabulary = left.Length, max_absolute_logit_difference = maximum });
                        Assert.Equal(0, maximum);
                    }
                    output.WriteLine($"complete-logit parity: prompt={item.Text}, chunk={item.Chunk}, rows={actual.Logits.Length}; "
                        + $"scalar prefill={expected.PrefillSeconds:F6}s, grouped prefill={actual.PrefillSeconds:F6}s");
                }
            completed = true;

            Phase Run(string prompt, int chunk, int[] tokens, int[]? teacher)
            {
                model.ResetKVCache();
                var logits = new List<float[]>();
                var forced = new int[8];
                long callsBefore = GgmlBasicOps.GetHostMoeExpertCacheUsage().Calls;
                float[] current = [];
                var clock = Stopwatch.StartNew();
                for (int offset = 0; offset < tokens.Length; offset += chunk)
                {
                    current = model.Forward(tokens.AsSpan(offset, Math.Min(chunk, tokens.Length - offset)).ToArray());
                    logits.Add((float[])current.Clone());
                }
                double prefillSeconds = clock.Elapsed.TotalSeconds;
                clock.Restart();
                for (int step = 0; step < forced.Length; ++step)
                {
                    int next = teacher?[step] ?? ArgMax(current);
                    forced[step] = next;
                    current = model.Forward([next]); logits.Add((float[])current.Clone());
                }
                double decodeSeconds = clock.Elapsed.TotalSeconds;
                long calls = GgmlBasicOps.GetHostMoeExpertCacheUsage().Calls - callsBefore;
                Assert.True(calls > 0, "The real-model prefill comparison did not engage the expert cache.");
                return new(prompt, chunk, logits.ToArray(), forced, prefillSeconds, decodeSeconds, calls);
            }
        }
        catch (Exception error) { failure = error.ToString(); throw; }
        finally
        {
            MoeCpuOffloadConfig.Reset();
            string? directory = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_REPORT_DIR");
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
                File.WriteAllText(Path.Combine(directory, "prefill-microgroup-parity.json"), JsonSerializer.Serialize(new
                {
                    completed, failure, model = modelPath, comparisons,
                    scope = "Same loaded actual weights, scalar/planned-N3 expert scopes, identical 129-token prompts "
                        + "in 17/32-token chunks, full-vocabulary logits after every prefill chunk and eight teacher-forced decode steps. "
                        + "No HTTP, mmproj, factual-quality, long-context or controlled performance claim."
                }, new JsonSerializerOptions { WriteIndented = true }));
            }
        }
    }

    private static int ArgMax(float[] values)
    {
        Assert.NotEmpty(values);
        int maximum = 0;
        for (int index = 1; index < values.Length; ++index)
            if (values[index] > values[maximum]) maximum = index;
        return maximum;
    }
}
