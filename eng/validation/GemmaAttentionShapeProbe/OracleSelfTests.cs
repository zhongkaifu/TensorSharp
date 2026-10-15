// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace GemmaAttentionShapeProbe;

internal static class OracleSelfTests
{
    internal static int Run()
    {
        int checks = 0;
        void Equal(double actual, double expected) { checks++; if (!double.IsFinite(actual) || Math.Abs(actual - expected) > 1e-12) throw new Exception($"{actual:R} != {expected:R}"); }
        void Reject(Action action) { checks++; try { action(); } catch (ArgumentException) { return; } throw new Exception("Expected rejection."); }
        var g = new Geometry("tiny", 4, 2, 1, 4);
        // Uniform softmax: prefix1 includes key0 and current key1; each later
        // query extends the range by one. The second KV head has distinct V.
        float[] q = new float[8], k = new float[8], v = [2, 4, 12, 999, 20, 40, 120, 999];
        var oracle = AttentionOracle.Evaluate(g, q, k, v, 2, 1);
        double[] expected = [3, 3, 30, 30, 6, 6, 60, 60];
        for (int i = 0; i < expected.Length; i++) Equal(oracle[i], expected[i]);
        float[] labelled = [0, 1, 2, 10, 11, 12];
        var slice = AttentionOracle.QuerySlice(labelled, 2, 3, 1, 1, 2);
        foreach (var pair in slice.Zip(new float[] { 1, 2, 11, 12 })) Equal(pair.First, pair.Second);
        Reject(() => AttentionOracle.QuerySlice(labelled, 2, 3, 1, 2, 2));
        Reject(() => AttentionOracle.Evaluate(g, q, k, v, 2, 3));
        Reject(() => AttentionOracle.Evaluate(g, [float.NaN, 0, 0, 0, 0, 0, 0, 0], k, v, 2, 1));
        Equal(AttentionOracle.Evaluate(new("stable", 1, 1, 1, 2), [1000], [1000, 999], [2, 9], 1, 1)[0], 2);
        var bad = AttentionOracle.Compare([1, 2], [1, float.NaN]);
        Equal(bad.ActualNonfinite, 1); Equal(bad.WithinDiagnosticThreshold ? 1 : 0, 0);
        Equal(bad.MaxAbs is null ? 1 : 0, 1);
        var delta = AttentionOracle.Compare([1, 2], [2, 0]);
        Equal(delta.MaxAbs!.Value, 2); Equal(delta.Rms!.Value, Math.Sqrt(2.5)); Equal(delta.MeanSigned!.Value, -0.5);
        var data = AttentionOracle.Inputs(new("synthetic", 4, 2, 16, 256), "unit-rms", 8, 127);
        Equal(data.K.All(x => float.IsFinite(x) && x == (float)(Half)x) ? 1 : 0, 1);
        Equal(data.V.All(x => float.IsFinite(x) && x == (float)(Half)x) ? 1 : 0, 1);
        Equal(data.K[135 * 16], 4); Equal(data.V[135 * 16], 9);
        Console.WriteLine($"{checks} CPU oracle checks passed; no native backend initialized.");
        return 0;
    }
}
