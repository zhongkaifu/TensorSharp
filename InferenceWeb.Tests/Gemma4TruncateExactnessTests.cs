using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Normal conversation-cache truncation has no speculative verify-slot backup.
/// Once the sliding window wraps, TryTruncate must either restore the required
/// history or refuse without changing it. A refusal permits reset and re-prefill.
/// </summary>
public class Gemma4TruncateExactnessTests
{
    private readonly ITestOutputHelper _output;
    public Gemma4TruncateExactnessTests(ITestOutputHelper output) => _output = output;

    [ModelFact("TS_TEST_MODEL_DIR", "gemma-4-e4b")]
    public void ConversationRewind_IsExactOrRefusesWithoutChangingWrappedState()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), "gemma-4-e4b");
        Assert.False(string.IsNullOrEmpty(path));
        BackendType backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu")
            .Trim().ToLowerInvariant() switch
            {
                "cuda" => BackendType.GgmlCuda,
                "metal" => BackendType.GgmlMetal,
                _ => BackendType.GgmlCpu,
            };
        using var model = ModelBase.Create(path, backend);
        Assert.True(model.SupportsKVCacheTruncation);
        int window = model.Config.SlidingWindow;
        Assert.True(window >= 64, $"Expected a real sliding window, found {window}.");
        var cases = new[]
        {
            (Name: "unwrapped", Prefix: window - 32, Tail: 8, Suffix: 1),
            (Name: "last-unwrapped-head", Prefix: window - 2, Tail: 2, Suffix: 17),
            (Name: "first-wrapped-head", Prefix: window - 1, Tail: 2, Suffix: 1),
            (Name: "wrapped-two-row-rewind", Prefix: 2 * window + 17, Tail: 2, Suffix: 1),
            (Name: "wrapped-two-row-prefill", Prefix: 2 * window + 17, Tail: 2, Suffix: 17),
            (Name: "wrapped-conversation-tail", Prefix: 2 * window + 17, Tail: 64, Suffix: 17),
            (Name: "whole-window-overwritten", Prefix: 2 * window + 17, Tail: window + 17, Suffix: 1),
        };
        int maxLength = cases.Max(c => c.Prefix + c.Tail + c.Suffix + 3);
        int[] source = BuildTokens(model, maxLength + 128);
        // Hold capacity constant across references and trials: growing a global
        // cache must not become an accidental difference in this ring-state test.
        Assert.IsAssignableFrom<ISpeculativeTarget>(model).SpecEnsureCapacity(maxLength);
        var failures = new List<string>();

        foreach (var c in cases)
        {
            int[] prefix = source.Take(c.Prefix).ToArray();
            int[] tail = source.Skip(c.Prefix).Take(c.Tail).ToArray();
            int[] suffix = source.Skip(maxLength + 32).Take(c.Suffix).Reverse().ToArray();
            int[] follow = source.Skip(maxLength + 64).Take(3).ToArray();

            // Both the reference and trial prefill exactly the same prefix in
            // the same call shape. The suffix and teacher-forced decode calls
            // also match. Comparing a single combined prefill to split prefill
            // would confound cache corruption with batch-dependent arithmetic.
            model.ResetKVCache();
            model.ForwardRefill(prefix);
            float[][] prefixReference = Continue(model, suffix, follow);

            model.ResetKVCache();
            model.ForwardRefill(prefix);
            model.ForwardRefill(tail);
            float[][] unchangedHeadReference = Continue(model, suffix, follow);

            model.ResetKVCache();
            model.ForwardRefill(prefix);
            model.ForwardRefill(tail);
            int head = c.Prefix + c.Tail;
            Assert.Equal(head, model.CacheSeqLen);
            bool truncated = model.TryTruncateKVCache(c.Prefix);
            _output.WriteLine($"case={c.Name} window={window} head={head} target={c.Prefix} suffix={c.Suffix} accepted={truncated}");
            if (truncated)
            {
                Assert.Equal(c.Prefix, model.CacheSeqLen);
                Compare(c.Name + ": accepted rewind", prefixReference,
                    Continue(model, suffix, follow), failures);
            }
            else
            {
                Assert.Equal(head, model.CacheSeqLen);
                Compare(c.Name + ": refusal preserves state", unchangedHeadReference,
                    Continue(model, suffix, follow), failures);
                if (head <= window)
                    failures.Add(c.Name + ": unwrapped truncation unexpectedly refused");
                model.ResetKVCache();
                model.ForwardRefill(prefix);
                Compare(c.Name + ": re-prefill recovery", prefixReference,
                    Continue(model, suffix, follow), failures);
            }
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private static float[][] Continue(ModelBase model, int[] suffix, int[] follow)
    {
        var logits = new float[follow.Length + 1][];
        logits[0] = (float[])model.ForwardRefill(suffix).Clone();
        for (int i = 0; i < follow.Length; i++)
            logits[i + 1] = (float[])model.Forward(new[] { follow[i] }).Clone();
        return logits;
    }

    private void Compare(string label, float[][] expected, float[][] actual, List<string> failures)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxAbsolute = 0, maxToleranceRatio = 0;
        int mismatchedArgmax = 0;
        for (int step = 0; step < expected.Length; step++)
        {
            Assert.Equal(expected[step].Length, actual[step].Length);
            for (int i = 0; i < expected[step].Length; i++)
            {
                Assert.True(float.IsFinite(expected[step][i]) && float.IsFinite(actual[step][i]),
                    $"{label}: non-finite logit at step={step}, index={i}");
                double error = Math.Abs((double)actual[step][i] - expected[step][i]);
                maxAbsolute = Math.Max(maxAbsolute, error);
                maxToleranceRatio = Math.Max(maxToleranceRatio,
                    error / (1e-4 + 1e-4 * Math.Abs(expected[step][i])));
            }
            if (Argmax(expected[step]) != Argmax(actual[step])) mismatchedArgmax++;
        }
        _output.WriteLine($"{label}: max_abs={maxAbsolute:G9} max_tolerance_ratio={maxToleranceRatio:G9} argmax_mismatches={mismatchedArgmax}/{expected.Length}");
        if (maxToleranceRatio > 1 || mismatchedArgmax != 0)
            failures.Add($"{label}: max_abs={maxAbsolute:G9}, tolerance_ratio={maxToleranceRatio:G9}, argmax_mismatches={mismatchedArgmax}");
    }

    private static int Argmax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    private static int[] BuildTokens(ModelBase model, int count)
    {
        var text = new System.Text.StringBuilder();
        for (int line = 0; ; line++)
        {
            text.Append($"Record {line}: key cedar-{line * 17 + 31}; value {line * 13 + 7}. Keep these records in order.\n");
            if (line % 64 != 63) continue;
            int[] tokens = model.Tokenizer.Encode(text.ToString(), addSpecial: true).ToArray();
            if (tokens.Length >= count) return tokens.Take(count).ToArray();
        }
    }
}
