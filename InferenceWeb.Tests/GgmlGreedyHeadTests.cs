// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
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
