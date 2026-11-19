// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Graph;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The GGML greedy head (<see cref="GraphCapabilities.GreedyIdOutputs"/>; ggml_ops_graph_builder.cpp's GreedyHead): each
/// row's id is the host's argmax exactly - the first largest logit, as TokenSampler's ordered scan takes it - in whatever
/// order the device's top-k keeps ties: rows with one, two, seven, eight and nine equal maxima (eight or more: every one of
/// the head's eight candidates ties, so the row is read in full), maxima at the row's ends, signed zeros, infinities, NaNs (a
/// leading NaN is the host scan's pick, a later one never is) and rows of quantized logits, where ties are common. Runs on
/// the pinned GGML backend (TS_TEST_GGML_BACKEND), whatever its executor declares.
/// </summary>
public sealed class GgmlGreedyHeadTests
{
    private readonly ITestOutputHelper _output;

    public GgmlGreedyHeadTests(ITestOutputHelper output) => _output = output;

    /// <summary>TokenSampler.ArgmaxScalar.</summary>
    private static int HostArgmax(float[] v)
    {
        int best = 0;
        float bestValue = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > bestValue)
            {
                bestValue = v[i];
                best = i;
            }
        return best;
    }

    /// <summary>Whether the head reads <paramref name="v"/> in full: a NaN or an infinity, or eight or more logits equal to the
    /// largest (a row of more than eight).</summary>
    private static bool ReadsInFull(float[] v)
    {
        float max = float.NegativeInfinity;
        foreach (float x in v)
        {
            if (!float.IsFinite(x)) return true;
            max = Math.Max(max, x);
        }
        int ties = 0;
        foreach (float x in v) if (x == max) ties++;
        return v.Length > 8 && ties >= 8;
    }

    private static List<(string Name, float[] Row)> Rows(int vocab)
    {
        var rng = new Random(20261119 + vocab);
        float[] Noise()
        {
            var v = new float[vocab];
            for (int i = 0; i < vocab; i++) v[i] = (float)(rng.NextDouble() * 8 - 6);   // in [-6, 2)
            return v;
        }
        float[] Ties(int count, int from)
        {
            float[] v = Noise();
            // `count` maxima spread over the row from `from`, the lowest first in no particular place of the device's order.
            int stride = Math.Max(1, (vocab - from) / count);
            for (int t = 0; t < count; t++) v[Math.Min(vocab - 1, from + t * stride)] = 7.5f;
            return v;
        }
        var rows = new List<(string, float[])> { ("unique", Noise()) };
        if (vocab < 16)
        {
            float[] small = Noise();
            small[vocab - 1] = small[1] = 9f;
            rows.Add(("small-tie", small));
            return rows;
        }
        rows.Add(("two-ties", Ties(2, vocab / 3)));
        rows.Add(("seven-ties", Ties(7, 5)));
        rows.Add(("eight-ties", Ties(8, 11)));
        rows.Add(("nine-ties", Ties(9, 0)));
        float[] ends = Noise();
        ends[0] = ends[vocab - 1] = 4f;
        rows.Add(("ends", ends));
        float[] last = Noise();
        last[vocab - 1] = 4f;
        rows.Add(("last", last));
        float[] zeros = Noise();
        for (int i = 0; i < vocab; i++) zeros[i] = -1f - Math.Abs(zeros[i]);
        zeros[vocab / 2] = -0.0f;
        zeros[vocab / 2 + 7] = 0.0f;
        zeros[vocab - 3] = -0.0f;
        rows.Add(("signed-zeros", zeros));
        float[] posInf = Noise();
        posInf[vocab / 4] = float.PositiveInfinity;
        posInf[vocab / 5] = float.PositiveInfinity;
        rows.Add(("+inf", posInf));
        float[] negInf = Noise();
        for (int i = 0; i < vocab; i += 3) negInf[i] = float.NegativeInfinity;
        rows.Add(("-inf", negInf));
        float[] allNegInf = new float[vocab];
        Array.Fill(allNegInf, float.NegativeInfinity);
        rows.Add(("all -inf", allNegInf));
        float[] leadingNaN = Noise();
        leadingNaN[0] = float.NaN;
        rows.Add(("leading NaN", leadingNaN));
        float[] laterNaN = Noise();
        laterNaN[vocab / 3] = float.NaN;
        laterNaN[vocab - 2] = 5f;
        rows.Add(("later NaN", laterNaN));
        for (int q = 0; q < 24; q++)
        {
            // Logits on a grid: the largest shared by a few ids (fine grids: the candidates decide) or by many (coarse ones:
            // the row is read in full).
            float[] v = Noise();
            float step = MathF.ScaleB(1f, q < 16 ? -12 + q % 4 : -2 + q % 3);
            for (int i = 0; i < vocab; i++) v[i] = MathF.Round(v[i] / step) * step;
            rows.Add(($"grid{q}", v));
        }
        return rows;
    }

    /// <summary>A greedy head with a suppression mask (<see cref="GraphCapabilities.GreedyIdSuppression"/>): the ids are the
    /// host argmax after TokenSampler sets the suppressed logits to -inf - a suppressed largest logit, suppressed ties, a NaN
    /// or +inf at a suppressed id (the row read in full, the mask assigned as the host assigns it), a row whose kept logits
    /// are all -inf.</summary>
    [Theory]
    [InlineData(3000)]
    [InlineData(128000)]
    public void SuppressedGreedyIdsAreTheHostArgmaxOfTheMaskedRow(int vocab)
    {
        GgmlBackendType backend = TestGates.PinnedGgmlBackendType;
        GgmlBasicOps.EnsureBackendAvailable(backend);
        int[] suppressed = { 0, 7, vocab / 3, vocab / 2, vocab / 2 + 1, vocab - 1 };
        List<(string Name, float[] Row)> rows = Rows(vocab);
        var rng = new Random(vocab);
        foreach (int top in new[] { vocab / 3, vocab - 1, 0 })
        {
            float[] v = new float[vocab];
            for (int i = 0; i < vocab; i++) v[i] = (float)(rng.NextDouble() * 4 - 3);
            v[top] = 9f;                       // the largest is suppressed
            v[vocab / 2 + 1] = 8f;             // and so is the next
            v[11] = v[vocab / 4] = 7f;         // the pick: two kept ties
            rows.Add(($"suppressed top {top}", v));
        }
        float[] nanAt = Rows(vocab)[0].Row;
        nanAt[7] = float.NaN;
        rows.Add(("NaN at a suppressed id", nanAt));
        float[] infAt = Rows(vocab)[0].Row;
        infAt[vocab / 2] = float.PositiveInfinity;
        rows.Add(("+inf at a suppressed id", infAt));
        float[] onlySuppressed = new float[vocab];
        Array.Fill(onlySuppressed, float.NegativeInfinity);
        foreach (int id in suppressed) onlySuppressed[id] = 3f;
        rows.Add(("kept logits all -inf", onlySuppressed));
        var mask = new float[vocab];
        Array.Fill(mask, -0.0f);
        foreach (int id in suppressed) mask[id] = float.NegativeInfinity;

        int n = rows.Count;
        var logits = new float[(long)vocab * n];
        for (int r = 0; r < n; r++) Array.Copy(rows[r].Row, 0, logits, (long)r * vocab, vocab);
        GCHandle pin = GCHandle.Alloc(logits, GCHandleType.Pinned), maskPin = GCHandle.Alloc(mask, GCHandleType.Pinned);
        try
        {
            var g = new GraphProgramBuilder(record: true);
            int input = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { vocab, n }, IntPtr.Zero, logits.LongLength * 4);
            int maskLeaf = g.Leaf(GraphLeafRole.State, GraphFormat.F32, new long[] { vocab }, maskPin.AddrOfPinnedObject(), mask.LongLength * 4);
            g.Output(g.Op(GraphOp.Cont, stackalloc[] { g.Op(GraphOp.TopK, stackalloc[] { input, maskLeaf }, stackalloc long[] { 1, 1 }) }), n);
            using IGraphPlan plan = GgmlGraphExecutor.For(backend).Compile(g.ToProgram(replayable: true, reorder: false));
            Assert.True(plan.Run(new[] { input }, new[] { pin.AddrOfPinnedObject() }, new[] { logits.LongLength * 4 }), plan.LastError);
            for (int r = 0; r < n; r++)
            {
                float[] masked = (float[])rows[r].Row.Clone();
                foreach (int id in suppressed) masked[id] = float.NegativeInfinity;
                int expected = HostArgmax(masked);
                Assert.True(expected == (int)plan.Outputs[0][r],
                    $"{backend} vocab {vocab}, row {r} ({rows[r].Name}): id {plan.Outputs[0][r]}, the host's argmax over the masked row {expected}");
            }
        }
        finally
        {
            GgmlBasicOps.InvalidateHostBuffer(maskPin.AddrOfPinnedObject());
            pin.Free();
            maskPin.Free();
        }
    }

    [Theory]
    [InlineData(5)]
    [InlineData(3000)]
    [InlineData(128000)]
    public void GreedyIdsAreTheHostArgmaxOnTiedAndNonFiniteRows(int vocab)
    {
        GgmlBackendType backend = TestGates.PinnedGgmlBackendType;
        GgmlBasicOps.EnsureBackendAvailable(backend);
        List<(string Name, float[] Row)> rows = Rows(vocab);
        int n = rows.Count;
        var logits = new float[(long)vocab * n];
        for (int r = 0; r < n; r++) Array.Copy(rows[r].Row, 0, logits, (long)r * vocab, vocab);

        var g = new GraphProgramBuilder(record: true);
        int input = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { vocab, n }, IntPtr.Zero, logits.LongLength * 4);
        // The shared runtime's greedy head (StandardDecoderModel.EmitHead): a greedy top-1, its ids read as one float a row.
        g.Output(g.Op(GraphOp.Cont, stackalloc[] { g.Op(GraphOp.TopK, stackalloc[] { input }, stackalloc long[] { 1, 1 }) }), n);
        using IGraphPlan plan = GgmlGraphExecutor.For(backend).Compile(g.ToProgram(replayable: true, reorder: false));
        GCHandle pin = GCHandle.Alloc(logits, GCHandleType.Pinned);
        try
        {
            for (int run = 0; run < 2; run++)
            {
                long full = GgmlGraph.GreedyFullRows;
                Assert.True(plan.Run(new[] { input }, new[] { pin.AddrOfPinnedObject() }, new[] { logits.LongLength * 4 }), plan.LastError);
                long readInFull = GgmlGraph.GreedyFullRows - full;
                int expectedFull = 0;
                for (int r = 0; r < n; r++)
                {
                    int expected = HostArgmax(rows[r].Row);
                    if (ReadsInFull(rows[r].Row)) expectedFull++;
                    Assert.True(expected == (int)plan.Outputs[0][r],
                        $"{backend} vocab {vocab} run {run}, row {r} ({rows[r].Name}): id {plan.Outputs[0][r]}, the host's argmax {expected}");
                }
                // Rows whose candidates cannot decide (every one ties, or a NaN or infinity) and only those were read in full.
                Assert.Equal(expectedFull, readInFull);
                _output.WriteLine($"{backend} vocab {vocab} run {run}: {n} rows, {readInFull} read in full");
            }
        }
        finally
        {
            pin.Free();
        }
    }
}

/// <summary>Runs only where the pinned GGML executor picks greedy ids on the device (TS_TEST_GGML_BACKEND=cuda or vulkan).</summary>
public sealed class GgmlDeviceGreedyFactAttribute : FactAttribute
{
    public GgmlDeviceGreedyFactAttribute()
    {
        if (!GgmlGraphExecutor.For(TestGates.PinnedGgmlBackendType).Capabilities.GreedyIdSuppression)
            Skip = $"Requires TS_TEST_GGML_BACKEND=cuda or vulkan: the pinned {TestGates.PinnedGgmlBackendType} executor picks greedy ids on the host.";
    }
}

/// <summary>
/// The shared runtime's greedy steps with suppressed tokens (LFM3-Omni's reply-token policy) on a GGML executor that picks
/// on the device: three sequences decoding together pick what TokenSampler picks from their logits with the same tokens
/// suppressed, step after step (the suppressed set holds the tokens the unsuppressed stream picks most).
/// </summary>
public sealed class GgmlSuppressedGreedyStepTests : IDisposable
{
    private readonly string _dir = System.IO.Directory.CreateTempSubdirectory("greedy-suppressed").FullName;

    public void Dispose()
    {
        try { System.IO.Directory.Delete(_dir, recursive: true); } catch (System.IO.IOException) { }
    }

    private static int[][] Run(string path, IReadOnlyList<int> suppressed, bool device)
    {
        using var model = TensorSharp.Models.ModelBase.Create(path, TestGates.PinnedGgmlBackend);
        var unified = Assert.IsAssignableFrom<TensorSharp.Runtime.Scheduling.IUnifiedBatchModel>(model);
        string[] ids = { "a", "b", "c" };
        int[][] tokens = { new[] { 40, 41, 42, 43, 44 }, new[] { 50, 60, 70 }, new[] { 90, 91, 92, 93, 94, 95, 96 } };
        int[] starts = { 0, 0, 0 };
        foreach (string id in ids) unified.EnsureSequenceSlot(id, 128);
        var picked = new List<int[]>();
        var sampler = new TensorSharp.Runtime.TokenSampler(TensorSharp.Runtime.SamplingConfig.Greedy) { SuppressedTokens = suppressed };
        for (int step = 0; step < 12; step++)
        {
            var next = new int[3];
            if (device && step > 0)
            {
                Assert.True(unified.SupportsUnifiedGreedySuppression);
                Assert.True(unified.TryForwardUnifiedGreedy(ids, tokens, starts, next, suppressed));
            }
            else
            {
                var logits = new float[3][];
                Assert.True(unified.TryForwardUnified(ids, tokens, starts, logits));
                for (int s = 0; s < 3; s++) next[s] = suppressed.Count > 0 ? sampler.Sample((float[])logits[s].Clone(), new[] { 1 }) : HostArgmax(logits[s]);
            }
            picked.Add(next);
            for (int s = 0; s < 3; s++) starts[s] += tokens[s].Length;
            tokens = next.Select(t => new[] { t }).ToArray();
        }
        foreach (string id in ids) ((TensorSharp.Models.StandardDecoderModel)model).OnSequenceReleased(id);
        return picked.ToArray();
    }

    private static int HostArgmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    [GgmlDeviceGreedyFact]
    public void SuppressedGreedyStepsPickWhatTheSamplerPicks()
    {
        string path = new Lfm3SyntheticModelBuilder { UntiedOutput = true }.Write(System.IO.Path.Combine(_dir, "lfm3.gguf"));
        int[][] plain = Run(path, Array.Empty<int>(), device: false);
        int[] frequent = plain.SelectMany(t => t).GroupBy(t => t).OrderByDescending(g => g.Count()).Take(3).Select(g => g.Key).ToArray();
        int[][] host = Run(path, frequent, device: false);
        int[][] onDevice = Run(path, frequent, device: true);
        Assert.Equal(host.Length, onDevice.Length);
        for (int i = 0; i < host.Length; i++) Assert.Equal(host[i], onDevice[i]);
        Assert.DoesNotContain(onDevice.SelectMany(t => t), t => frequent.Contains(t));
    }
}
