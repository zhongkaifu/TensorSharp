// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Independent CPU references over original Q4G64_F16S encoded bytes.
using System;

namespace TensorSharp.Validation;

public static class NinferDraftShortlistReference
{
    public const int Rows = 131072, Width = 5120;
    public static float Half(ushort bits) => (float)BitConverter.Int16BitsToHalf(unchecked((short)bits));
    public static float Float(ushort bits) => BitConverter.Int32BitsToSingle(bits << 16);
    public static ushort Bf(float value)
    {
        uint bits = (uint)BitConverter.SingleToInt32Bits(value);
        return (ushort)((bits + 0x7fffU + ((bits >> 16) & 1)) >> 16);
    }
    public static int Code(byte packed, int column) { int nibble = packed >> ((column & 1) * 4) & 15; return nibble < 8 ? nibble : nibble - 16; }
    public static int Q(byte[] codes, int row, int column) => Code(codes[row * (Width / 2) + column / 2], column);
    public static float Scale(ushort[] scales, int row, int column) => Half(scales[row * (Width / 64) + column / 64]);

    /// <summary>Actual original T1 packed-word8 lane schedule, one FMA chain,
    /// then descending warp sum. No scalar weight BF16 rounding.</summary>
    public static ushort GemvRow(byte[] codes, ushort[] scales, ReadOnlySpan<ushort> input, int row)
    {
        var lanes = new float[32]; var previous = new float[32];
        for (int lane = 0; lane < 32; lane++)
        {
            float accumulator = 0;
            for (int tile = 0; tile < 80; tile += 16) for (int g = 0; g < 16; g += 4)
            {
                int group = tile + g + lane / 8;
                float scale = Half(scales[row * 80 + group]);
                int first = group * 64 + lane % 8 * 8;
                for (int j = 0; j < 8; j++)
                    accumulator = MathF.FusedMultiplyAdd(Q(codes, row, first + j) * scale, Float(input[first + j]), accumulator);
            }
            lanes[lane] = accumulator;
        }
        for (int distance = 16; distance > 0; distance >>= 1)
        {
            lanes.CopyTo(previous, 0);
            for (int lane = 0; lane < 32; lane++) lanes[lane] = previous[lane] + previous[lane + distance < 32 ? lane + distance : lane];
        }
        return Bf(lanes[0]);
    }
    /// <summary>Original T2..8 split schedule. Valid as an exact oracle only
    /// when every64-wide integer-Q4/BF16 dot is exactly F32 representable for
    /// every possible partial sum; generated grid-valued fixtures prove this.</summary>
    public static ushort SmallRowExactGrid(byte[] codes, ushort[] scales, ReadOnlySpan<ushort> input, int row)
    {
        var partial = new float[8];
        for (int split = 0; split < 8; split++)
        {
            float accumulator = 0;
            for (int phase = 0; phase < 10; phase++)
            {
                int group = phase * 8 + split; double groupDot = 0, absolute = 0;
                for (int j = 0; j < 64; j++)
                {
                    double product = Q(codes, row, group * 64 + j) * (double)Float(input[group * 64 + j]);
                    groupDot += product; absolute += Math.Abs(product);
                }
                if ((double)(float)groupDot != groupDot || absolute * 512 >= 16777216)
                    throw new InvalidOperationException("Small-T exact-grid fixture does not prove exact partial dots.");
                accumulator = MathF.FusedMultiplyAdd((float)groupDot, Half(scales[row * 80 + group]), accumulator);
            }
            partial[split] = accumulator;
        }
        float sum = partial[0] + partial[1];
        sum += partial[2] + partial[3]; sum += partial[4] + partial[5]; sum += partial[6] + partial[7];
        return Bf(sum);
    }
    public static (double Sum, double Absolute) Ideal(byte[] codes, ushort[] scales, ReadOnlySpan<ushort> input, int row, int tokens)
    {
        double sum = 0, absolute = 0;
        for (int c = 0; c < Width; c++)
        {
            float weight = Q(codes, row, c) * Scale(scales, row, c);
            if (tokens >= 9) weight = Float(Bf(weight)); // original r64_c128 weight boundary
            double product = (double)weight * Float(input[c]); sum += product; absolute += Math.Abs(product);
        }
        return (sum, absolute);
    }
    public static double Bound(double ideal, double sumAbsolute)
    {
        const double u = 1.0 / 16777216, b = 1.0 / 256;
        double gamma = (Width + 128) * u / (1 - (Width + 128) * u);
        return (1 + b) * gamma * sumAbsolute + b * Math.Abs(ideal);
    }
    public static int[] Remap(int[] ids, int[] map)
    {
        if (map.Length != Rows) throw new ArgumentException("Full original shortlist map required.");
        var result = (int[])ids.Clone();
        for (int i = 0; i < result.Length; i++) if ((uint)result[i] < Rows) result[i] = map[result[i]];
        return result;
    }
}
