using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

// A full prefill and a continuation may use different quantized kernels. This
// test compares identical continuation shapes before/after copying a checkpoint
// and switching holders, isolating cache fidelity from that numerical variation.
public class Gemma4PrefixCloneExactnessTests
{
    private readonly ITestOutputHelper _output;
    public Gemma4PrefixCloneExactnessTests(ITestOutputHelper output) => _output = output;

    [ModelFact("TS_TEST_MODEL_DIR", "gemma-4-e4b")]
    public void WrappedCheckpointClones_PreserveLogitsAcrossInterleavedHolders()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), "gemma-4-e4b");
        Assert.False(string.IsNullOrEmpty(path));
        var backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu").ToLowerInvariant() switch
        {
            "cuda" => BackendType.GgmlCuda,
            "metal" => BackendType.GgmlMetal,
            _ => BackendType.GgmlCpu,
        };
        // Prefixes reach ~5k tokens: pin the context instead of inheriting a lane's MAX_CONTEXT.
        using var env = new EnvScope();
        env.Set("MAX_CONTEXT", "8192");
        using var model = ModelBase.Create(path, backend);
        var fused = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        Assert.True(fused.SupportsPrefixCheckpoints);
        var text = string.Join("\n", Enumerable.Range(0, 1100).Select(i =>
            $"Record {i}: the item code is cedar-{i * 17 + 31}. Keep the records in order."));
        int[] source = model.Tokenizer.Encode(text, addSpecial: true).ToArray();
        Assert.True(source.Length >= 5500);
        var shapes = new[] { (Prefix: 4096, Scheduler: false),
            (Prefix: 4096, Scheduler: true), (Prefix: 4279, Scheduler: true) };

        foreach (var shape in shapes)
        foreach (int suffixLength in new[] { 1, 17, 128, 600 })
        {
            int[] prefix = source.Take(shape.Prefix).ToArray();
            int[][] suffixes = { source.Skip(shape.Prefix).Take(suffixLength).ToArray(),
                source.Skip(shape.Prefix + 64).Take(suffixLength).Reverse().ToArray() };
            var references = new float[2][][];
            for (int branch = 0; branch < 2; ++branch)
            {
                model.ResetKVCache();
                ForwardInput(model, prefix, shape.Scheduler ? 512 : 0);
                references[branch] = Decode(model, suffixes[branch], 4, shape.Scheduler ? 256 : 0);
            }
            model.ResetKVCache();
            ForwardInput(model, prefix, shape.Scheduler ? 512 : 0);
            string checkpoint = $"prefix-{shape.Prefix}-{shape.Scheduler}-{suffixLength}";
            Assert.True(fused.TryCheckpointActiveCache(checkpoint));
            // Alter the original before cloning: the saved checkpoint must own
            // its bytes, including every wrapped sliding-window layer.
            model.ForwardRefill(source.Skip(4200).Take(31).ToArray());
            string[] requests = { checkpoint + "-a", checkpoint + "-b" };
            foreach (string request in requests) Assert.True(fused.TryCloneRetainedCache(checkpoint, request));
            var tokens = new int[2];
            double maxError = 0;
            for (int step = 0; step < 4; ++step)
            for (int branch = 0; branch < 2; ++branch)
            {
                Assert.False(fused.BindSequenceCache(requests[branch]));
                float[] actual = step == 0 ? ForwardInput(model, suffixes[branch], shape.Scheduler ? 256 : 0) : model.Forward(new[] { tokens[branch] });
                float[] expected = references[branch][step];
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < actual.Length; ++i)
                {
                    Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
                    double error = Math.Abs((double)actual[i] - expected[i]);
                    maxError = Math.Max(maxError, error);
                    Assert.True(error <= 1e-4 + 1e-4 * Math.Abs(expected[i]),
                        $"prefix={shape.Prefix} scheduler={shape.Scheduler} suffix={suffixLength} branch={branch} step={step} logit={i} expected={expected[i]} actual={actual[i]} error={error}");
                }
                tokens[branch] = Argmax(actual);
                Assert.Equal(Argmax(expected), tokens[branch]);
            }
            _output.WriteLine($"prefix={shape.Prefix} scheduler={shape.Scheduler} suffix={suffixLength} branches=2 interleaved_steps=4 max_abs={maxError:G9}");
            foreach (string request in requests) fused.OnSequenceReleased(request);
            fused.DiscardRetainedCache(checkpoint);
            fused.RestorePrimaryCache();
        }
    }

    private static float[][] Decode(ModelBase model, int[] suffix, int steps, int chunkSize)
    {
        var output = new float[steps][];
        int token = 0;
        for (int step = 0; step < steps; ++step)
        {
            float[] logits = step == 0 ? ForwardInput(model, suffix, chunkSize) : model.Forward(new[] { token });
            output[step] = (float[])logits.Clone();
            token = Argmax(logits);
        }
        return output;
    }

    private static float[] ForwardInput(ModelBase model, int[] tokens, int chunkSize)
    {
        if (chunkSize == 0) return model.ForwardRefill(tokens);
        float[] logits = null;
        for (int offset = 0; offset < tokens.Length; offset += chunkSize)
            logits = model.Forward(tokens.Skip(offset).Take(chunkSize).ToArray());
        return logits;
    }

    private static int Argmax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; ++i) if (values[i] > values[best]) best = i;
        return best;
    }
}
