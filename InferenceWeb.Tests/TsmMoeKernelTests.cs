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
}
