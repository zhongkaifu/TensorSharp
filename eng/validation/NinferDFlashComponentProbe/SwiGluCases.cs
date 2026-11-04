// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using R = TensorSharp.Validation.NinferDFlashReference;

internal static partial class Program
{
    private static IEnumerable<Case> SwiGluCases()
    {
        const int n = 34816, k = 5120, half = 17408;
        foreach (bool saturated in new[] { true, false })
        {
            int[] active = { 0, 1, 7, 8, 31, 32, 127, 128, 255, half - 1 };
            byte[] codes = new byte[n * k / 2], scales = Enumerable.Repeat((byte)0x38, n * k / 16).ToArray();
            // Constant scales make original swizzle-independent ones; the raw
            // code rows deliberately stay gate-half then up-half, never TS rows8.
            foreach (int row in active) for (int c = 0; c < k; c += 2)
            {
                codes[row * k / 2 + c / 2] = saturated ? (byte)0x22 : (byte)(EncodedCode(row, c) | EncodedCode(row, c + 1) << 4);
                codes[(row + half) * k / 2 + c / 2] = (byte)(EncodedCode(row + 5, c + 3) | EncodedCode(row + 5, c + 4) << 4);
            }
            foreach (int tokens in new[] { 1, 2, 4, 5, 8, 12, 13, 16 })
            {
                ushort[] input = saturated ? Enumerable.Range(0, k * tokens).Select(i => R.BFloat((1 + i % 4) / 128f)).ToArray() : Values(k * tokens, 47).Select(x => R.BFloat(R.Float(x) / 4096)).ToArray();
                ushort[] expected = new ushort[half * tokens]; double[] ideal = new double[expected.Length], bound = new double[expected.Length];
                for (int t = 0; t < tokens; t++) foreach (int row in active)
                {
                    float gate = R.ProjectRow(codes, scales, input.AsSpan(t * k, k), row, k, tokens);
                    float up = R.ProjectRow(codes, scales, input.AsSpan(t * k, k), row + half, k, tokens);
                    int at = t * half + row;
                    if (saturated)
                    {
                        Need(gate >= 32 && 1f + MathF.Exp(-gate) == 1f, "SwiGLU saturation fixture is not exact.");
                        float value = gate * up; expected[at] = R.BFloat(value); ideal[at] = (double)gate * up;
                        bound[at] = R.BFloatUnitRoundoff * Math.Abs(ideal[at]) + R.FloatUnitRoundoff * Math.Abs(ideal[at]);
                    }
                    else
                    {
                        ideal[at] = gate / (1.0 + Math.Exp(-gate)) * up;
                        bound[at] = R.BFloatUnitRoundoff * Math.Abs(ideal[at]) + 32 * R.FloatUnitRoundoff * Math.Abs((double)gate * up);
                    }
                }
                yield return new($"raw-swiglu-{(saturated ? "exact-saturation" : "bounded-nonlinear")}-t{tokens}", new() { ["codes"] = codes, ["scales"] = scales, ["input"] = Bytes(input) },
                    new() { ["output"] = new(saturated ? Bytes(expected) : null, ideal, bound) },
                    (m, p) => m.SwiGluRaw(p["codes"], p["scales"], p["input"], p["output"], tokens, 1, IntPtr.Zero),
                    new { oracle = "Original raw gate-half/up-half projection F32 accumulators feed SiLU and multiply before sole BF16 output rounding", exactBits = saturated, activeRows = active,
                        bound = saturated ? "(2^-8+2^-24)*abs(FP64 gate*up); gate>=32 makes F32 1+exp(-gate) exactly1" : "2^-8*abs(idealSiLU)+32*2^-24*abs(gate*up); |gate| bounded, no FP32 overflow/underflow", syntheticWeights = true });
            }
        }
    }
}
