// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Validation;
using R = TensorSharp.Validation.NinferDFlashReference;

internal static partial class Program
{
    private static IEnumerable<Case> ProjectionCases()
    {
        foreach (var (n, k) in new[] { (5120, 25600), (6144, 5120), (5120, 4096), (1280, 5120), (256, 5120), (5120, 17408) })
        {
            // Source is written as original adjacent logical K pairs. Scales are
            // materialized in explicit five-axis directory traversal, independently
            // of the reference decoder's flattened offset expression.
            byte[] codes = new byte[checked(n * k / 2)], scales = new byte[checked(n * k / 16)];
            int[] active = new[] { 0, 1, 7, 8, 31, 32, 63, 64, 127, 128, 255, n - 1 }.Distinct().Order().ToArray();
            foreach (int row in active) for (int c = 0; c < k; c += 2)
                codes[row * (k / 2) + c / 2] = (byte)(EncodedCode(row, c) | (EncodedCode(row, c + 1) << 4));
            int at = 0;
            for (int tile = 0; tile < n / 128; tile++) for (int kg = 0; kg < k / 64; kg++)
                for (int row32 = 0; row32 < 32; row32++) for (int quarter = 0; quarter < 4; quarter++) for (int group = 0; group < 4; group++)
                    scales[at++] = EncodedScale(tile * 128 + quarter * 32 + row32, kg * 4 + group);
            foreach (int tokens in new[] { 1, 2, 4, 8, 16, 17, 20, 21, 32 })
            {
                const float inverse = .13f;
                ushort[] input = Values(k * tokens, 17), expected = new ushort[n * tokens];
                var ideal = new double[expected.Length]; var bounds = new double[expected.Length];
                for (int t = 0; t < tokens; t++) foreach (int row in active)
                {
                    float projected = R.ProjectRow(codes, scales, input.AsSpan(t * k, k), row, k, tokens, inverse);
                    int output = t * n + row; expected[output] = R.BFloat(projected);
                    double exact = 0, absolute = 0;
                    for (int c = 0; c < k; c++)
                    {
                        double product = (double)R.Code(EncodedCode(row, c)) * R.Scale(EncodedScale(row, c / 16)) * inverse * R.Float(input[t * k + c]);
                        exact += product; absolute += Math.Abs(product);
                    }
                    ideal[output] = exact; bounds[output] = R.ProjectionBound(k, exact, absolute);
                    Need(Math.Abs(R.Float(expected[output]) - exact) <= bounds[output], "Original CPU A16 schedule misses predeclared FP64 bound.");
                }
                yield return new($"raw-a16-n{n}-k{k}-t{tokens}", new() { ["codes"] = codes, ["scales"] = scales, ["input"] = Bytes(input) },
                    new() { ["output"] = new(Bytes(expected), ideal, bounds) },
                    (m, p) => m.ProjectRaw(p["codes"], p["scales"], p["input"], p["output"], n, k, tokens, inverse, IntPtr.Zero),
                    new { oracle = "Original adjacentK raw codes + independently decoded original m128x4 scales; original production FMA/chain/SHFL_DOWN schedule then BF16", n, k, tokens, activeRows = active,
                        inactiveRows = "exact positive zero from encoded zero weights", inverseDivisor = inverse, valuesPerLane = tokens is >= 17 and <= 20 ? 8 : 16,
                        exactBits = true, bound = "(1+2^-8)*gamma_(K/32+16)*sum(abs(products))+2^-8*abs(FP64ideal); u=2^-24", syntheticWeights = true });
            }
        }
    }
    private static int EncodedCode(int row, int column) => (row * 7 + column * 11 + column / 16) & 15;
    private static byte EncodedScale(int row, int group) => (byte)(48 + (row * 3 + group * 7) % 17);
}
