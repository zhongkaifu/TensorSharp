// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The Muse-Glimmer / Muse Spark vision tower's learned position grid as the model was trained to read it, against the
/// reference implementation's own arithmetic: Fixtures/MuseGlimmer/Vision/position-grid-reference.json, made by
/// eng/validation/muse-glimmer-posemb-reference.py with transformers' get_vision_interpolation_indices_and_weights called as
/// main 7394e56's MuseGlimmerVisionPatchEmbedder calls it (huggingface/transformers#50847: grid_thw[:, [0, 2, 1]], bilinear,
/// align_corners=False, zero padding). The table is resampled to the transposed grid (its rows over the grid's width) and read
/// in raster order, as llama.cpp reads it since d3c3e242c (ggml-org/llama.cpp#32433); the order before that fix (the table's
/// rows over the grid's rows) agrees on square grids only. Taps off the table weigh nothing, as in the reference: llama.cpp's
/// clamped bilinear differs at the edges of an axis longer than the table.
/// </summary>
public sealed class MuseGlimmerVisionPositionGridTests(ITestOutputHelper output)
{
    private const string Resource = "InferenceWeb.Tests.Fixtures.MuseGlimmer.Vision.position-grid-reference.json";

    /// <summary>Single-precision rounding of the same bilinear sums (the values are of order 1).</summary>
    private const double Bound = 1e-6;

    internal sealed record Case(int Side, int GridW, int GridH, float[] Values);

    internal static (int Hidden, List<Case> Cases) Load()
    {
        using Stream s = typeof(MuseGlimmerVisionPositionGridTests).Assembly.GetManifestResourceStream(Resource)
            ?? throw new FileNotFoundException(Resource);
        using JsonDocument doc = JsonDocument.Parse(s);
        int hidden = doc.RootElement.GetProperty("hidden").GetInt32();
        var cases = new List<Case>();
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            JsonElement v = c.GetProperty("values");
            var values = new float[v.GetArrayLength()];
            int i = 0;
            foreach (JsonElement e in v.EnumerateArray()) values[i++] = e.GetSingle();
            cases.Add(new Case(c.GetProperty("side").GetInt32(), c.GetProperty("gridW").GetInt32(), c.GetProperty("gridH").GetInt32(), values));
        }
        return (hidden, cases);
    }

    /// <summary>The generator's table: t[i] = sin(0.37 i) + 0.25 cos(1.3 i) over the flat [side * side, hidden] values.</summary>
    internal static float[] Table(int side, int hidden)
    {
        var t = new float[side * side * hidden];
        for (int i = 0; i < t.Length; i++) t[i] = (float)(Math.Sin(0.37 * i) + 0.25 * Math.Cos(1.3 * i));
        return t;
    }

    private static unsafe float[] Resample(float[] table, int side, int hidden, int gridW, int gridH)
    {
        int n = gridW * gridH;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        var dst = new float[n * hidden];
        fixed (float* t = table)
        fixed (float* d = dst)
            MuseGlimmerVisionGrid.ResamplePositionTable(t, side, hidden, gridW, gridH, order, d);
        return dst;
    }

    /// <summary>The order before the fix: the table's rows over the grid's rows (ggml's border-clamped bilinear).</summary>
    private static float[] ResampleUntransposed(float[] table, int side, int hidden, int gridW, int gridH)
    {
        var dst = new float[gridW * gridH * hidden];
        float sfx = (float)gridW / side, sfy = (float)gridH / side;
        for (int t = 0; t < gridW * gridH; t++)
        {
            int gy = t / gridW, gx = t % gridW;
            float xf = (gx + 0.5f) / sfx - 0.5f, yf = (gy + 0.5f) / sfy - 0.5f;
            int x0 = (int)MathF.Floor(xf), y0 = (int)MathF.Floor(yf);
            int x1 = Math.Clamp(x0 + 1, 0, side - 1), y1 = Math.Clamp(y0 + 1, 0, side - 1);
            x0 = Math.Clamp(x0, 0, side - 1);
            y0 = Math.Clamp(y0, 0, side - 1);
            float dx = Math.Clamp(xf - x0, 0f, 1f), dy = Math.Clamp(yf - y0, 0f, 1f);
            for (int c = 0; c < hidden; c++)
                dst[t * hidden + c] = (1 - dx) * (1 - dy) * table[(y0 * side + x0) * hidden + c] + dx * (1 - dy) * table[(y0 * side + x1) * hidden + c]
                    + (1 - dx) * dy * table[(y1 * side + x0) * hidden + c] + dx * dy * table[(y1 * side + x1) * hidden + c];
        }
        return dst;
    }

    private static double MaxDifference(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs((double)a[i] - b[i]));
        return m;
    }

    /// <summary>On every grid, the edges of an axis longer than the table included (their taps off the table weigh nothing,
    /// as the reference's zero padding has it), the resample is the reference's.</summary>
    [Fact]
    public void TheTowersResampleIsTheReferencesOnEveryGrid()
    {
        var (hidden, cases) = Load();
        foreach (Case c in cases)
        {
            double d = MaxDifference(Resample(Table(c.Side, hidden), c.Side, hidden, c.GridW, c.GridH), c.Values);
            output.WriteLine($"side {c.Side} grid {c.GridW}x{c.GridH}: max |d| {d:E2}");
            Assert.True(d <= Bound, $"side {c.Side} grid {c.GridW}x{c.GridH}: {d:E3} from the reference");
        }
        Assert.Contains(cases, c => c.GridW > c.Side);
        Assert.Contains(cases, c => c.GridW != c.GridH && c.GridW <= c.Side && c.GridH <= c.Side);
    }

    /// <summary>llama.cpp d3c3e242c reads the table in the trained order with ggml_interpolate's border-clamped bilinear
    /// (ggml-cpu ops.cpp): the reference's values wherever every tap is in the table, and different only at the edge patches
    /// of an axis with more patches than the table, where the reference weighs the tap off the table as zero.</summary>
    [Fact]
    public void LlamaCppsClampedResampleDiffersOnlyAtTheEdgesOfAnAxisLongerThanTheTable()
    {
        var (hidden, cases) = Load();
        foreach (Case c in cases)
        {
            float[] clamped = ResampleTransposedClamped(Table(c.Side, hidden), c.Side, hidden, c.GridW, c.GridH);
            double inside = 0, edges = 0;
            for (int t = 0; t < c.GridW * c.GridH; t++)
            {
                // The transposed walk: row t / gridH over gridW, column t % gridH over gridH.
                int row = t / c.GridH, col = t % c.GridH;
                bool edge = OffTable(row, c.GridW, c.Side) || OffTable(col, c.GridH, c.Side);
                for (int k = 0; k < hidden; k++)
                {
                    double d = Math.Abs((double)clamped[t * hidden + k] - c.Values[t * hidden + k]);
                    if (edge) edges = Math.Max(edges, d);
                    else inside = Math.Max(inside, d);
                }
            }
            output.WriteLine($"side {c.Side} grid {c.GridW}x{c.GridH}: clamped vs reference inside {inside:E2}, at the edges {edges:E2}");
            Assert.True(inside <= Bound, $"grid {c.GridW}x{c.GridH}: the clamped resample differs off the edges ({inside:E3})");
            if (c.GridW > c.Side || c.GridH > c.Side) Assert.True(edges > 0.01, $"grid {c.GridW}x{c.GridH}: the edges agree ({edges:E3})");
        }
    }

    /// <summary>Whether index <paramref name="i"/> of an axis of <paramref name="size"/> has a weighted bilinear tap off a table
    /// of <paramref name="side"/> (half-pixel centres): the patches where zero padding and clamping differ.</summary>
    private static bool OffTable(int i, int size, int side)
    {
        double src = (i + 0.5) * side / size - 0.5, floor = Math.Floor(src), frac = src - floor;
        return (floor < 0 && frac < 1) || (floor + 1 > side - 1 && frac > 0);
    }

    /// <summary>ggml_interpolate's GGML_SCALE_MODE_BILINEAR over the table in the trained (transposed) order, as llama.cpp's
    /// graph runs it.</summary>
    private static float[] ResampleTransposedClamped(float[] table, int side, int hidden, int gridW, int gridH)
    {
        var dst = new float[gridW * gridH * hidden];
        float sfx = (float)gridH / side, sfy = (float)gridW / side;
        for (int t = 0; t < gridW * gridH; t++)
        {
            int gy = t / gridH, gx = t % gridH;
            float xf = (gx + 0.5f) / sfx - 0.5f, yf = (gy + 0.5f) / sfy - 0.5f;
            int x0 = (int)MathF.Floor(xf), y0 = (int)MathF.Floor(yf);
            int x1 = Math.Clamp(x0 + 1, 0, side - 1), y1 = Math.Clamp(y0 + 1, 0, side - 1);
            x0 = Math.Clamp(x0, 0, side - 1);
            y0 = Math.Clamp(y0, 0, side - 1);
            float dx = Math.Clamp(xf - x0, 0f, 1f), dy = Math.Clamp(yf - y0, 0f, 1f);
            for (int c = 0; c < hidden; c++)
                dst[t * hidden + c] = (1 - dx) * (1 - dy) * table[(y0 * side + x0) * hidden + c] + dx * (1 - dy) * table[(y0 * side + x1) * hidden + c]
                    + (1 - dx) * dy * table[(y1 * side + x0) * hidden + c] + dx * dy * table[(y1 * side + x1) * hidden + c];
        }
        return dst;
    }

    /// <summary>The test sees the defect it guards: the order before the fix is the reference's on a square grid and far from
    /// it on a non-square one.</summary>
    [Fact]
    public void TheOrderBeforeTheFixDiffersFromTheReferenceOnNonSquareGrids()
    {
        var (hidden, cases) = Load();
        foreach (Case c in cases)
        {
            if (c.GridW > c.Side || c.GridH > c.Side) continue;
            double d = MaxDifference(ResampleUntransposed(Table(c.Side, hidden), c.Side, hidden, c.GridW, c.GridH), c.Values);
            output.WriteLine($"side {c.Side} grid {c.GridW}x{c.GridH}: the untransposed read is {d:E2} from the reference");
            if (c.GridW == c.GridH) Assert.True(d <= Bound, $"square grid {c.GridW}: {d:E3}");
            else Assert.True(d > 0.1, $"grid {c.GridW}x{c.GridH}: only {d:E3} from the reference");
        }
    }
}
