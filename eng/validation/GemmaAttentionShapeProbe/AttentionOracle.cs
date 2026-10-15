// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace GemmaAttentionShapeProbe;

internal sealed record Geometry(string Name, int Heads, int KvHeads, int Dim, int CacheStride);

internal static class AttentionOracle
{
    // Q: [query-head, query, dimension]; K/V: [kv-head, cache-stride, dimension].
    // Output: [query, query-head, dimension]. All arithmetic after the stored
    // F32 Q/F16-rounded K/V inputs is scalar FP64, including the softmax.
    internal static double[] Evaluate(Geometry g, float[] q, float[] k, float[] v, int queries, int prefix)
    {
        Validate(g, q, k, v, queries, prefix);
        var result = new double[queries * g.Heads * g.Dim];
        var scores = new double[prefix + queries];
        for (int h = 0; h < g.Heads; h++)
        for (int row = 0; row < queries; row++)
        {
            int kh = h / (g.Heads / g.KvHeads), keys = prefix + row + 1;
            double maximum = double.NegativeInfinity;
            for (int key = 0; key < keys; key++)
            {
                double dot = 0;
                for (int d = 0; d < g.Dim; d++)
                    dot += (double)q[(h * queries + row) * g.Dim + d] * k[(kh * g.CacheStride + key) * g.Dim + d];
                scores[key] = dot;
                maximum = Math.Max(maximum, dot);
            }
            double denominator = 0;
            for (int key = 0; key < keys; key++) denominator += scores[key] = Math.Exp(scores[key] - maximum);
            for (int d = 0; d < g.Dim; d++)
            {
                double value = 0;
                for (int key = 0; key < keys; key++)
                    value += scores[key] / denominator * v[(kh * g.CacheStride + key) * g.Dim + d];
                result[(row * g.Heads + h) * g.Dim + d] = value;
            }
        }
        return result;
    }

    private static void Validate(Geometry g, float[] q, float[] k, float[] v, int queries, int prefix)
    {
        if (g.Heads <= 0 || g.KvHeads <= 0 || g.Heads % g.KvHeads != 0 || g.Dim <= 0 || queries <= 0 || prefix < 0
            || prefix + queries > g.CacheStride || q.Length != g.Heads * queries * g.Dim
            || k.Length != g.KvHeads * g.CacheStride * g.Dim || v.Length != k.Length
            || q.Any(x => !float.IsFinite(x)) || k.Any(x => !float.IsFinite(x)) || v.Any(x => !float.IsFinite(x)))
            throw new ArgumentException("Invalid geometry, lengths, or nonfinite input.");
    }

    internal static float[] QuerySlice(float[] source, int heads, int sourceRows, int dim, int start, int rows)
    {
        if (heads <= 0 || sourceRows <= 0 || dim <= 0 || start < 0 || rows <= 0 || start + rows > sourceRows
            || source.Length != heads * sourceRows * dim) throw new ArgumentException("Invalid head-first Q slice.");
        float[] result = new float[heads * rows * dim];
        for (int h = 0; h < heads; h++)
            Array.Copy(source, (h * sourceRows + start) * dim, result, h * rows * dim, rows * dim);
        return result;
    }

    internal static (float[] Q, float[] K, float[] V) Inputs(Geometry g, string profile, int maxRows, int prefix)
    {
        if (profile is not ("gentle" or "unit-rms")) throw new ArgumentException("Unknown input profile.");
        var q = new float[g.Heads * maxRows * g.Dim];
        var k = new float[g.KvHeads * g.CacheStride * g.Dim];
        var v = new float[k.Length];
        uint state = 0x67249ac5;
        float Next()
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            return (state >> 8) * (2f / 16777216f) - 1f;
        }
        // Generation is independent of the later query count. Every N arm uses
        // the same first N rows and identical cached inputs, including future keys.
        for (int h = 0; h < g.Heads; h++)
        for (int row = 0; row < maxRows; row++)
        {
            int offset = (h * maxRows + row) * g.Dim;
            for (int d = 0; d < g.Dim; d++) q[offset + d] = Next();
            ScaleRow(q, offset, g.Dim, profile == "gentle" ? 0.2 : 1.0);
        }
        for (int h = 0; h < g.KvHeads; h++)
        for (int key = 0; key < g.CacheStride; key++)
        {
            int offset = (h * g.CacheStride + key) * g.Dim;
            if (key >= prefix + maxRows)
            {
                // Finite nonzero sentinels make accidentally visible padding
                // conspicuous without confounding it with 0*NaN propagation.
                for (int d = 0; d < g.Dim; d++) { k[offset + d] = 4 + h * 0.25f; v[offset + d] = 9 + h * 0.5f; }
                continue;
            }
            for (int d = 0; d < g.Dim; d++) { k[offset + d] = Next(); v[offset + d] = Next(); }
            ScaleRow(k, offset, g.Dim, profile == "gentle" ? 0.3 : 1.0);
            ScaleRow(v, offset, g.Dim, profile == "gentle" ? 0.7 : 1.0);
            for (int d = 0; d < g.Dim; d++) { k[offset + d] = (float)(Half)k[offset + d]; v[offset + d] = (float)(Half)v[offset + d]; }
        }
        return (q, k, v);
    }

    private static void ScaleRow(float[] values, int offset, int dim, double rms)
    {
        double squared = 0;
        for (int d = 0; d < dim; d++) squared += (double)values[offset + d] * values[offset + d];
        double scale = rms / Math.Sqrt(squared / dim);
        for (int d = 0; d < dim; d++) values[offset + d] = (float)(values[offset + d] * scale);
    }

    internal static ErrorMetrics Compare(double[] expected, float[] actual)
    {
        if (expected.Length != actual.Length || expected.Length == 0) throw new ArgumentException("Comparison lengths differ or are empty.");
        int actualNonfinite = 0, expectedNonfinite = 0, worst = -1;
        double maximum = 0, squared = 0, norm = 0, signed = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!float.IsFinite(actual[i])) actualNonfinite++;
            if (!double.IsFinite(expected[i])) expectedNonfinite++;
            if (!float.IsFinite(actual[i]) || !double.IsFinite(expected[i])) continue;
            double delta = actual[i] - expected[i];
            if (Math.Abs(delta) > maximum) { maximum = Math.Abs(delta); worst = i; }
            squared += delta * delta; norm += expected[i] * expected[i]; signed += delta;
        }
        bool finite = actualNonfinite == 0 && expectedNonfinite == 0;
        return new(actual.Length, actualNonfinite, expectedNonfinite, finite ? maximum : null,
            finite ? Math.Sqrt(squared / actual.Length) : null,
            finite && norm > 0 ? Math.Sqrt(squared / norm) : null,
            finite ? signed / actual.Length : null, worst,
            finite && maximum <= 2e-3 && (norm == 0 ? squared == 0 : Math.Sqrt(squared / norm) <= 1e-3));
    }
}

internal sealed record ErrorMetrics(int Elements, int ActualNonfinite, int ReferenceNonfinite,
    double? MaxAbs, double? Rms, double? RelativeL2, double? MeanSigned, int WorstIndex,
    bool WithinDiagnosticThreshold);
