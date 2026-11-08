// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The TensorSharp Metal grouped expert kernels (tsm/tsm_moe.mm, Strata's moe.metal) on one layer: every held route's
/// silu(gate x) * up x through down, weighted and summed per row in route order, against a double-precision reference
/// over ggml's own dequantization of the same bytes. Rows share experts (the grouping), and a route the host computes
/// (slot -1) contributes nothing.
/// </summary>
public sealed class TsmMoeKernelTests
{
    private readonly ITestOutputHelper _output;
    public TsmMoeKernelTests(ITestOutputHelper output) => _output = output;

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData((int)SyntheticGguf.GgmlType.Q2_0, (int)SyntheticGguf.GgmlType.Q2_0, 1)]
    [InlineData((int)SyntheticGguf.GgmlType.Q2_0, (int)SyntheticGguf.GgmlType.Q2_0, 4)]
    [InlineData((int)SyntheticGguf.GgmlType.IQ3_S, (int)SyntheticGguf.GgmlType.IQ4_NL, 4)]
    [InlineData((int)SyntheticGguf.GgmlType.IQ2_XS, (int)SyntheticGguf.GgmlType.IQ4_NL, 3)]
    [InlineData((int)SyntheticGguf.GgmlType.Q4_K, (int)SyntheticGguf.GgmlType.Q5_1, 2)]
    [InlineData((int)SyntheticGguf.GgmlType.IQ4_XS, (int)SyntheticGguf.GgmlType.Q8_0, 8)]
    public void GroupedExperts_AnswerAsTheDoublePrecisionReference(int gateTypeId, int downTypeId, int rows)
    {
        var gateType = (SyntheticGguf.GgmlType)gateTypeId;
        var downType = (SyntheticGguf.GgmlType)downTypeId;
        const int hidden = 512, nFf = 256, nExpert = 12, nUsed = 4;
        byte[] Stack(string name, SyntheticGguf.GgmlType t, int cols, int rowsPerExpert, float scale)
            => SyntheticGguf.Blocks(name, t, scale, cols, rowsPerExpert, nExpert).RawBlocks;
        byte[] gate = Stack("gate", gateType, hidden, nFf, 1.5f / MathF.Sqrt(hidden));
        byte[] up = Stack("up", gateType, hidden, nFf, 1.5f / MathF.Sqrt(hidden));
        byte[] down = Stack("down", downType, nFf, hidden, 0.5f / MathF.Sqrt(nFf));
        float[] Dequant(byte[] bytes, SyntheticGguf.GgmlType t, long n)
        {
            var f = new float[n];
            var h = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            var o = GCHandle.Alloc(f, GCHandleType.Pinned);
            try { GgmlGgufTensorDequant.DequantizeToFloat32Native((int)t, h.AddrOfPinnedObject(), o.AddrOfPinnedObject(), n); }
            finally { h.Free(); o.Free(); }
            return f;
        }
        float[] gw = Dequant(gate, gateType, (long)nExpert * nFf * hidden), uw = Dequant(up, gateType, (long)nExpert * nFf * hidden);
        float[] dw = Dequant(down, downType, (long)nExpert * hidden * nFf);

        var rng = new Random(rows * 31 + (int)gateType);
        var x = new float[rows * hidden];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        // Routes: rows share experts (slot 3 in every row), one route per row left to the host (-1).
        var slots = new int[rows * nUsed];
        var weights = new float[rows * nUsed];
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < nUsed; k++)
            {
                slots[r * nUsed + k] = k == 0 ? 3 : k == nUsed - 1 && r % 2 == 1 ? -1 : (r * 5 + k * 7) % nExpert;
                weights[r * nUsed + k] = (float)(0.1 + rng.NextDouble());
            }

        var got = new float[rows * hidden];
        Assert.True(GgmlBasicOps.TsmMoeProbe((int)gateType, (int)downType, hidden, nFf, nExpert, gate, up, down, rows, nUsed,
            slots, weights, x, got), GgmlBasicOps.LastNativeError("probe failed"));

        double maxErr = 0, maxRef = 0;
        var h = new double[nFf];
        for (int r = 0; r < rows; r++)
        {
            var y = new double[hidden];
            for (int k = 0; k < nUsed; k++)
            {
                int e = slots[r * nUsed + k];
                if (e < 0) continue;
                for (int j = 0; j < nFf; j++)
                {
                    double g = 0, u = 0;
                    long row = ((long)e * nFf + j) * hidden;
                    for (int c = 0; c < hidden; c++) { g += gw[row + c] * (double)x[r * hidden + c]; u += uw[row + c] * (double)x[r * hidden + c]; }
                    h[j] = g / (1 + Math.Exp(-g)) * u;
                }
                for (int i = 0; i < hidden; i++)
                {
                    double d = 0;
                    long row = ((long)e * hidden + i) * nFf;
                    for (int j = 0; j < nFf; j++) d += dw[row + j] * h[j];
                    y[i] += weights[r * nUsed + k] * d;
                }
            }
            for (int i = 0; i < hidden; i++)
            {
                maxRef = Math.Max(maxRef, Math.Abs(y[i]));
                maxErr = Math.Max(maxErr, Math.Abs(got[r * hidden + i] - y[i]));
            }
        }
        double rel = maxErr / Math.Max(maxRef, 1e-30);
        _output.WriteLine($"{gateType}/{downType}, {rows} rows: max |error| {maxErr:E3} of max |y| {maxRef:E3} ({rel:E3})");
        Assert.True(rel <= 2e-5, $"{gateType}/{downType}: the grouped kernels strayed {rel:E3} from the reference");
    }

    /// <summary>The QSA prompt selection kernels (tsm_qsa_prompt.metal) against the contract of
    /// eng/tests/qwen4exp-qsa.py, on a window shaped like a prompt chunk's: every cell's value is (the heads' relu
    /// scores summed in order + the block's bias) + the mask, in F32 as ggml computes it; the kept cells are the
    /// <c>width</c> best of those the mask allows - every strictly better cell, exact cardinality - and the kernels
    /// take the cells tied at the cutoff lowest index first. Blocks share their cells' score, a forced tail of 1e9
    /// bias makes whole runs of ties, and early rows allow fewer cells than <c>width</c> (all of them kept).</summary>
    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(512, 4, 4, 64, 96)]
    [InlineData(2560, 4, 4, 2048, 64)]
    [InlineData(4100, 4, 2, 2048, 40)]
    public void QsaSelection_KeepsTheBestCells_TiesLowestFirst(int cells, int ratio, int heads, int topK, int rows)
    {
        int blocks = (cells + ratio - 1) / ratio, width = Math.Min(cells, topK + ratio - 1);
        var rng = new Random(cells + rows);
        var scores = new float[rows * heads * blocks];
        for (int i = 0; i < scores.Length; i++) scores[i] = (float)(rng.NextDouble() * 4 - 1.5);
        int tail = 3;   // forced recent blocks
        var bias = new float[rows * blocks];
        var cellBlocks = new int[cells];
        for (int c = 0; c < cells; c++) cellBlocks[c] = c / ratio;
        var maskIn = new ushort[rows * cells];
        const ushort NegInf = 0xFC00, Zero = 0;
        int live = cells - 7;   // the last cells are padding
        for (int t = 0; t < rows; t++)
        {
            // Rows sit at the end of the live window; the first few rows of a short window allow few cells.
            int pos = Math.Min(live - 1, t * (live / rows) + (t % 3));
            for (int c = 0; c < cells; c++) maskIn[t * cells + c] = c <= pos ? Zero : NegInf;
            for (int b = Math.Max(0, pos / ratio - tail + 1); b <= pos / ratio; b++) bias[t * blocks + b] = 1e9f;
        }
        var maskOut = new ushort[rows * cells];
        Assert.True(GgmlBasicOps.TsmQsaSelectProbe(rows, cells, blocks, heads, width, scores, bias, cellBlocks, maskIn, maskOut),
            GgmlBasicOps.LastNativeError("probe failed"));
        int boundaryTies = 0;
        for (int t = 0; t < rows; t++)
        {
            var s = new float[blocks];
            for (int b = 0; b < blocks; b++)
            {
                float v = MathF.Max(scores[(t * heads) * blocks + b], 0f);
                for (int h = 1; h < heads; h++) v += MathF.Max(scores[(t * heads + h) * blocks + b], 0f);
                s[b] = v + bias[t * blocks + b];
            }
            var e = new float[cells];
            var eligible = new System.Collections.Generic.List<int>();
            for (int c = 0; c < cells; c++)
            {
                e[c] = s[cellBlocks[c]] + (maskIn[t * cells + c] == NegInf ? float.NegativeInfinity : 0f);
                if (float.IsFinite(e[c])) eligible.Add(c);
            }
            int take = Math.Min(width, eligible.Count);
            var sorted = eligible.ConvertAll(c => e[c]);
            sorted.Sort();
            float cutoff = take > 0 ? sorted[sorted.Count - take] : float.PositiveInfinity;
            int above = 0;
            foreach (int c in eligible) if (e[c] > cutoff) above++;
            int tiesNeeded = take - above, tiesSeen = 0, chosen = 0;
            for (int c = 0; c < cells; c++)
            {
                bool kept = maskOut[t * cells + c] != NegInf;
                bool want = float.IsFinite(e[c]) && (e[c] > cutoff || (e[c] == cutoff && tiesSeen++ < tiesNeeded));
                Assert.True(kept == want, $"row {t} cell {c}: kept {kept}, contract {want} (value {e[c]}, cutoff {cutoff})");
                if (kept) { Assert.Equal(maskIn[t * cells + c], maskOut[t * cells + c]); chosen++; }
            }
            Assert.Equal(take, chosen);
            if (tiesSeen > tiesNeeded && tiesNeeded > 0) boundaryTies++;
        }
        _output.WriteLine($"{rows} rows over {cells} cells: {boundaryTies} rows split a tie at the cutoff");
        Assert.True(boundaryTies > 0, "the fixture never split a tie at the cutoff");
    }
}
