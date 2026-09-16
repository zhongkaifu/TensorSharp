// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using TensorSharp;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A two-pair video clip through the real Qwen4Exp forward on the synthetic QSA
/// target fixture (untrained weights, GGML CPU): the video's (T,H,W) coordinates
/// reach the token-span kernel, the QSA position history and the post-clip cache
/// gap, and the second pair's later temporal id changes the logits. The clip's
/// embeddings are synthetic (the fixture tokenizer has no vision pads and no
/// projector); the encoder is covered by <see cref="QwenVLVideoEncoderTests"/>.
/// </summary>
[Collection("Qwen4Exp MTP integration")]
[Trait("Requires", "Models")]
public sealed class Qwen4ExpVideoForwardTests(ITestOutputHelper output)
{
    private const int Pad = 41;   // any vocab id stands in for <|video_pad|> on this fixture
    private static readonly HashSet<int> Pads = new() { Pad };

    [Qwen4ExpQsaTinyFact]
    public void TwoTemporalPairs_ForwardWithIncreasingTimeReachQsaHistoryAndChangeLogits()
    {
        string directory = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MTP_FIXTURE")!;
        // TS_KV_INITIAL_TOKENS=8 also exercises growth immediately after reset
        // when the second video prompt exceeds the first allocation.
        var backend = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MTP_BACKEND") switch
        {
            null or "" or "GgmlCpu" => BackendType.GgmlCpu,
            "GgmlCuda" => BackendType.GgmlCuda,
            "GgmlMetal" => BackendType.GgmlMetal,
            var unsupported => throw new InvalidOperationException($"Unsupported fixture backend {unsupported}"),
        };
        using var model = new Qwen4ExpModel(Path.Combine(directory, "target.gguf"), backend,
            draftGgufPath: Path.Combine(directory, "head.gguf"));
        output.WriteLine($"backend={backend}");
        Assert.Equal(8, model.Config.HiddenSize);

        // One pair (a 2x2 merged grid) versus two pairs separated by a time-label token.
        var pair = new ModelMultimodalInjector.QwenVLVisionSpan(Pad, 4, 2, 2);
        var one = ModelMultimodalInjector.LayoutQwenVLPrompt(new List<int> { 11, 19, Pad, 23, 29 }, new[] { pair }, Pads);
        var two = ModelMultimodalInjector.LayoutQwenVLPrompt(new List<int> { 11, 19, Pad, 23, 19, Pad, 23, 29 }, new[] { pair, pair }, Pads);
        Assert.Equal(new[] { 2 }, one.SpanStarts);
        Assert.Equal(new[] { 2, 8 }, two.SpanStarts);
        int firstT = two.Positions[3 * 2], secondT = two.Positions[3 * 8];
        Assert.True(secondT > firstT);

        float[] oneLogits = Run(model, one.Tokens, one.Positions, one.SpanStarts);
        float[] twoLogits = Run(model, two.Tokens, two.Positions, two.SpanStarts);
        int[] history = Field<int[]>(model, "_qsaPositions");
        Assert.Equal(two.Positions, history[..two.Positions.Length]);
        Assert.Equal(two.Tokens.Count, Field<int>(model, "_qsaPositionCount"));
        int lastT = two.Positions[3 * (two.Tokens.Count - 1)];
        Assert.Equal(two.Tokens.Count - 1 - lastT, Field<int>(model, "_mropeCacheGap"));

        // Text after the clip continues the rotary stream at lastT + 1 on every axis,
        // and the run is deterministic on a fresh cache.
        float[] tail = (float[])model.Forward([31]).Clone();
        AssertFinite(tail);
        Assert.Equal(new[] { lastT + 1, lastT + 1, lastT + 1 }, history[(3 * two.Tokens.Count)..(3 * two.Tokens.Count + 3)]);
        Assert.Equal(twoLogits, Run(model, two.Tokens, two.Positions, two.SpanStarts));
        Assert.NotEqual(oneLogits, twoLogits);

        // Same tokens and embeddings, but the second pair collapsed onto the first
        // pair's temporal id: only the T axis differs, so the logits must too.
        int[] flat = (int[])two.Positions.Clone();
        for (int t = 8; t < 12; t++) flat[3 * t] = firstT;
        Assert.NotEqual(two.Positions, flat);
        float[] flatLogits = Run(model, two.Tokens, flat, two.SpanStarts);
        Assert.NotEqual(twoLogits, flatLogits);
        output.WriteLine($"video pairs at T={firstT},{secondT}; last text T={lastT}; gap={two.Tokens.Count - 1 - lastT}; " +
                         $"max |two-flat| = {MaxAbsDiff(twoLogits, flatLogits)}");
    }

    private static float[] Run(Qwen4ExpModel model, List<int> tokens, int[] positions, int[] starts)
    {
        model.ResetKVCache();
        var allocator = (IAllocator)typeof(ModelBase).GetField("_allocator", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        for (int i = 0; i < starts.Length; i++)
        {
            var embeddings = new Tensor(allocator, DType.Float32, 4, model.Config.HiddenSize);
            var values = new float[4 * model.Config.HiddenSize];
            for (int v = 0; v < values.Length; v++) values[v] = 0.5f * MathF.Sin(0.7f * v + 3f * i + 1f);
            embeddings.SetElementsAsFloat(values);
            model.SetVisionEmbeddings(embeddings, starts[i]);
        }
        model.SetMRoPEPositions((int[])positions.Clone());
        float[] logits = (float[])model.Forward(tokens.ToArray()).Clone();
        AssertFinite(logits);
        return logits;
    }

    private static T Field<T>(Qwen4ExpModel model, string name) => (T)typeof(Qwen4ExpModel)
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    private static void AssertFinite(float[] row) => Assert.All(row, value => Assert.True(float.IsFinite(value)));
}
