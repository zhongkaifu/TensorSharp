// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Validation;

/// <summary>Independent decoded-payload FP64 oracle for the specific original
/// raw-weight BF16 MMA experiment. This is not a legacy scalar-schedule oracle.</summary>
internal static class NinferFlashInferA16Reference
{
    internal const double UnitRoundoff = 1.0 / 16777216.0;
    internal static float Fp4(int bits)
    {
        float magnitude = (bits & 7) switch { 0 => 0f, 1 => .5f, 2 => 1f, 3 => 1.5f, 4 => 2f, 5 => 3f, 6 => 4f, _ => 6f };
        return (bits & 8) == 0 ? magnitude : -magnitude;
    }
    internal static float Fp8(byte bits)
    {
        if (bits >= 0x7f) throw new ArgumentOutOfRangeException(nameof(bits), "Only the original positive finite NInfer scale domain is valid.");
        int exponent = bits >> 3, mantissa = bits & 7;
        return exponent == 0 ? MathF.ScaleB(mantissa, -9) : MathF.ScaleB(8 + mantissa, exponent - 10);
    }
    internal static ushort Bf(float value)
    {
        uint bits = (uint)BitConverter.SingleToInt32Bits(value);
        return (ushort)((bits + 0x7fffU + ((bits >> 16) & 1)) >> 16);
    }
    internal static float Float(ushort bits) => BitConverter.Int32BitsToSingle(bits << 16);
    internal static int ScaleOffset(int row, int group, int columns) => checked(
        row / 128 * (columns / 64) * 512 + group / 4 * 512
        + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4);

    internal static int ProveLocalWeightProducts()
    {
        int count = 0;
        for (int s = 0; s < 127; s++) for (int code = 0; code < 16; code++)
        {
            double ideal = (double)Fp4(code) * Fp8((byte)s);
            float actual = Fp4(code) * Fp8((byte)s);
            if (!double.IsFinite(ideal) || actual != ideal || Float(Bf(actual)) != ideal
                || (BitConverter.SingleToInt32Bits(actual) & 0xffff) != 0)
                throw new InvalidOperationException("Original FP4 times E4M3 is not exactly BF16-representable.");
            count++;
        }
        return count;
    }

    internal sealed record Oracle(double[] Ideal, double[] Bounds, float[]? Exact, double MaximumLatticeUnits);

    internal static Oracle Compute(byte[] codes, byte[] scales, ushort[] input, int n, int k,
        int rows, float alpha, int[] activeRows, string family)
    {
        if (codes.Length != n * k / 2 || scales.Length != n * k / 16 || input.Length != rows * k
            || !float.IsFinite(alpha) || alpha <= 0)
            throw new ArgumentException("Incomplete synthetic raw projection fixture.");
        var ideal = new double[rows * n]; var bound = new double[ideal.Length];
        float[]? exact = family == "broad-finite" ? null : new float[ideal.Length];
        // Fixed before GPU execution. All nonzero fixture products/intermediates
        // are normal F32 and finite. BF16*BF16 products are exact F32; arbitrary
        // single-precision accumulation association is covered by depth K+64.
        // One final F32 alpha multiply adds u*abs(ideal). No BF16 output allowance.
        double gamma = (k + 64) * UnitRoundoff / (1 - (k + 64) * UnitRoundoff);
        const double doubleUnitRoundoff = 1.0 / 9007199254740992.0;
        double gammaDouble = k * doubleUnitRoundoff / (1 - k * doubleUnitRoundoff);
        double maxUnits = 0;
        foreach (int row in activeRows) for (int t = 0; t < rows; t++)
        {
            double sum = 0, absolute = 0;
            for (int c = 0; c < k; c++)
            {
                int packed = codes[row * (k / 2) + c / 2];
                int code = c % 2 == 0 ? packed & 15 : packed >> 4;
                double weight = (double)Fp4(code) * Fp8(scales[ScaleOffset(row, c / 16, k)]);
                if (Float(Bf((float)weight)) != weight) throw new InvalidDataException("The local weight product lost BF16 exactness.");
                double product = weight * Float(input[t * k + c]);
                if (family == "exact-lattice" && product * 128 != Math.Truncate(product * 128))
                    throw new InvalidDataException("The exact fixture left its binary product lattice.");
                sum += product; absolute += Math.Abs(product);
            }
            int index = t * n + row;
            ideal[index] = sum * alpha;
            bound[index] = (1 + UnitRoundoff) * Math.Abs(alpha) * gamma * absolute + UnitRoundoff * Math.Abs(ideal[index])
                + (1 + doubleUnitRoundoff) * Math.Abs(alpha) * gammaDouble * absolute + doubleUnitRoundoff * Math.Abs(ideal[index]);
            if (exact != null) exact[index] = (float)ideal[index];
            if (family == "exact-lattice")
            {
                // Products lie on 1/128; alpha=3/4. Every partial sum and
                // scaled result is an integer on a binary grid below 2^24.
                double units = absolute * 128 * 3;
                if (units >= 16777216 || alpha != .75f || (double)exact![index] != ideal[index])
                    throw new InvalidDataException("The exact-MMA lattice proof does not hold.");
                maxUnits = Math.Max(maxUnits, units);
            }
            if (!double.IsFinite(ideal[index]) || !double.IsFinite(bound[index])) throw new InvalidDataException("Nonfinite oracle.");
        }
        return new(ideal, bound, exact, maxUnits);
    }
}
