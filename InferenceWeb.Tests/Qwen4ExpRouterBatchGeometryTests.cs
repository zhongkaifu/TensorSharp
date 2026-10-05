// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Keeps the trained model's 512-expert router width in a small deterministic
/// checkpoint. Metal selects a different softmax reduction geometry for four
/// rows than for one row unless decode preserves each request's row geometry.
/// Synthetic weights verify numerical isolation, not language quality.
/// </summary>
[Collection("Qwen4Exp MTP integration")]
public sealed class Qwen4ExpRouterBatchGeometryTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-q4e-router-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _environment = new();

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(3)]
    [InlineData(4)]
    public void Router512Experts_MetalBatchedLogitsMatchSoloBitwise(int width)
    {
        const int steps = 8;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "128");
        _environment.Set("TS_KV_INITIAL_TOKENS", "128");
        _environment.Set("TS_PER_SEQ_FUSED", "1");
        _environment.Set("TS_Q4E_DISABLE_ARENA_DECODE", null);
        _environment.ClearSpeculationVars();
        MoeCpuOffloadConfig.Reset();
        // The router stays on Metal. Host expert weights keep the fixture's
        // accelerator footprint small while retaining 512 experts / top ten.
        MoeCpuOffloadConfig.SetLayers(Qwen4ExpSyntheticModelBuilder.Layers);
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "router512.gguf"),
            q2kxlExperts: true, expertCount: 512, expertUsedCount: 10);
        using var model = new Qwen4ExpModel(path, BackendType.GgmlMetal);
        var serialIds = Enumerable.Range(0, width).Select(i => "serial-" + i).ToArray();
        var batchIds = Enumerable.Range(0, width).Select(i => "batch-" + i).ToArray();
        var positions = new int[width];
        var expected = new float[width][][];
        for (int i = 0; i < width; ++i)
        {
            int[] prompt = Prompt(i, 24 + 3 * i);
            Assert.True(model.BindSequenceCache(serialIds[i]));
            float[] initial = (float[])model.Forward(prompt).Clone();
            expected[i] = new float[steps][];
            for (int step = 0; step < steps; ++step)
                expected[i][step] = (float[])model.Forward([Token(i, step)]).Clone();
            Assert.True(model.BindSequenceCache(batchIds[i]));
            AssertBitwiseEqual(initial, model.Forward(prompt), $"width={width}, independent prefill row={i}");
            positions[i] = prompt.Length;
        }
        model.RestorePrimaryCache();
        long before = model.ArenaBatchedDecodeSteps;
        for (int step = 0; step < steps; ++step)
        {
            int[] order = Enumerable.Range(0, width).OrderBy(i => (i + step) % width).ToArray();
            var rows = new float[width][];
            Assert.True(model.TryForwardBatchedFusedDecode(order.Select(i => batchIds[i]).ToArray(),
                order.Select(i => Token(i, step)).ToArray(), order.Select(i => positions[i]).ToArray(), rows),
                model.BatchedFusedDecodeDeclineReason);
            for (int row = 0; row < width; ++row)
            {
                AssertBitwiseEqual(expected[order[row]][step], rows[row],
                    $"width={width}, step={step}, request={order[row]}, caller row={row}");
                if (row > 0) Assert.NotSame(rows[0], rows[row]);
            }
            for (int i = 0; i < width; ++i) ++positions[i];
        }
        Assert.Equal(steps, model.ArenaBatchedDecodeSteps - before);
        // Leaving the arena must publish the same recurrent, PLE and KV state
        // as independent decoding, including after changing caller order.
        for (int i = 0; i < width; ++i)
        {
            model.BindSequenceCache(serialIds[i]);
            float[] serial = (float[])model.Forward([Token(i, steps)]).Clone();
            model.BindSequenceCache(batchIds[i]);
            Assert.Equal(positions[i], model.CacheSeqLen);
            AssertBitwiseEqual(serial, model.Forward([Token(i, steps)]), $"width={width}, solo continuation row={i}");
        }
        output.WriteLine($"Metal: 512 experts/top ten, width={width}, {steps} batched graphs; all prefill, decode and solo-continuation logits bitwise equal to independent requests.");
    }

    private static int[] Prompt(int seed, int count)
        => Enumerable.Range(0, count).Select(i => (11 + 31 * i + 13 * seed) % 250).ToArray();

    private static int Token(int row, int step) => (37 + 23 * row + 17 * step) % 250;

    private static void AssertBitwiseEqual(float[] expected, float[] actual, string phase)
    {
        Assert.Equal(expected.Length, actual.Length);
        int differing = 0;
        double maximumError = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]), $"{phase}: non-finite logit {i}.");
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i])) ++differing;
            maximumError = Math.Max(maximumError, Math.Abs((double)expected[i] - actual[i]));
        }
        Assert.True(differing == 0, $"{phase}: {differing}/{expected.Length} logits differ bitwise; max |dlogit|={maximumError:G9}.");
    }

    public void Dispose()
    {
        _environment.Dispose();
        MoeCpuOffloadConfig.Reset();
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }
}
