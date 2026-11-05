// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Independent host references over original artifact encodings. These are not
// a serving implementation and do not qualify device/model execution.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Validation;

public static class NinferDFlashReference
{
    public const double FloatUnitRoundoff = 1.0 / 16777216;
    public const double BFloatUnitRoundoff = 1.0 / 256;
    public static float Float(ushort bits) => BitConverter.Int32BitsToSingle(bits << 16);
    public static ushort BFloat(float value)
    {
        uint bits = (uint)BitConverter.SingleToInt32Bits(value);
        return (ushort)((bits + 0x7fffU + ((bits >> 16) & 1)) >> 16);
    }
    public static float Code(int code)
    {
        float magnitude = (code & 7) switch { 0 => 0, 1 => .5f, 2 => 1, 3 => 1.5f, 4 => 2, 5 => 3, 6 => 4, _ => 6 };
        return (code & 8) == 0 ? magnitude : -magnitude;
    }
    public static float Scale(byte code)
    {
        if (code >= 127) throw new ArgumentOutOfRangeException(nameof(code));
        return code < 8 ? MathF.ScaleB(code, -9) : MathF.ScaleB(8 + code % 8, code / 8 - 10);
    }
    public static int ScaleOffset(int row, int group, int columns) =>
        checked((row / 128 * (columns / 64) + group / 4) * 512 + row % 32 * 16 + row % 128 / 32 * 4 + group % 4);

    /// <summary>One original raw-artifact row, exact production F32 FMA order.
    /// Includes the T17..20 eight-values-per-lane production schedule. The caller
    /// must select the original chunk's active T, not the total number of chunks.</summary>
    public static float ProjectRow(ReadOnlySpan<byte> codes, ReadOnlySpan<byte> scales,
        ReadOnlySpan<ushort> input, int row, int columns, int activeTokens)
    {
        if (columns <= 0 || columns % 512 != 0 || activeTokens < 1 || activeTokens > 32)
            throw new ArgumentOutOfRangeException(nameof(columns));
        return ProjectRow(codes, scales, input, row, columns, activeTokens, 1);
    }
    public static float ProjectRow(ReadOnlySpan<byte> codes, ReadOnlySpan<byte> scales,
        ReadOnlySpan<ushort> input, int row, int columns, int activeTokens, float inverseDivisor)
    {
        if (columns <= 0 || columns % 512 != 0 || activeTokens < 1 || activeTokens > 32
            || row < 0 || input.Length < columns || !float.IsFinite(inverseDivisor) || inverseDivisor <= 0)
            throw new ArgumentOutOfRangeException(nameof(columns));
        int values = activeTokens is >= 17 and <= 20 ? 8 : 16;
        int chains = activeTokens == 1 ? 4 : 1;
        var laneValues = new float[32];
        var previous = new float[32];
        var accumulator = new float[chains];
        for (int lane = 0; lane < 32; lane++)
        {
            Array.Clear(accumulator);
            for (int phase = 0; phase < columns / (32 * values); phase++)
                for (int j = 0; j < values; j++)
                {
                    int column = phase * 32 * values + lane * values + j;
                    byte packed = codes[checked(row * (columns / 2) + column / 2)];
                    float coefficient = Scale(scales[ScaleOffset(row, column / 16, columns)]) * inverseDivisor;
                    float weight = Code((packed >> ((column & 1) * 4)) & 15) * coefficient;
                    int chain = j & (chains - 1);
                    accumulator[chain] = MathF.FusedMultiplyAdd(weight, Float(input[column]), accumulator[chain]);
                }
            float sum = 0;
            for (int chain = 0; chain < chains; chain++) sum += accumulator[chain];
            laneValues[lane] = sum;
        }
        for (int distance = 16; distance > 0; distance >>= 1)
        {
            laneValues.CopyTo(previous, 0);
            for (int lane = 0; lane < 32; lane++)
                laneValues[lane] = previous[lane] + previous[lane + distance < 32 ? lane + distance : lane];
        }
        return laneValues[0];
    }
    public static double ProjectionBound(int columns, double ideal, double absoluteProducts)
    {
        double operations = columns / 32 + 16;
        double gamma = operations * FloatUnitRoundoff / (1 - operations * FloatUnitRoundoff);
        // F32 dot and coefficients, followed by final BF16 rounding. Generated
        // fixtures avoid overflow and underflow; this bound is fixed a priori.
        return (1 + BFloatUnitRoundoff) * gamma * absoluteProducts + BFloatUnitRoundoff * Math.Abs(ideal);
    }
    public static ushort[] DynamicConv(ushort[] hidden, ushort[] dynamic, ushort[] basis,
        int width, int block, int lanes, int side)
    {
        if (width <= 0 || width % 16 != 0 || block < 1 || lanes < 1 || side is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(width));
        int groups = width / 16, columns = block * lanes;
        if (hidden.Length != width * columns || dynamic.Length != groups * 4 * columns || basis.Length != 4 * width)
            throw new ArgumentException("Original dynamic-convolution geometry differs.");
        var result = new ushort[hidden.Length];
        for (int t = 0; t < columns; t++) for (int c = 0; c < width; c++)
        {
            float accumulator = 0;
            for (int tap = 0; tap < 2; tap++)
            {
                float weight = Float(dynamic[t * groups * 4 + groups * 2 * side + groups * tap + c / 16])
                    + Float(basis[(side * 2 + tap) * width + c]);
                float x = t % block >= tap ? Float(hidden[(t - tap) * width + c]) : 0;
                accumulator = MathF.FusedMultiplyAdd(weight, x, accumulator);
            }
            result[t * width + c] = BFloat(accumulator);
        }
        return result;
    }
    public static ushort[] ResidualAdd(ushort[] input, ushort[] residual)
    {
        if (input.Length != residual.Length) throw new ArgumentException("Residual geometry differs.");
        return input.Zip(residual, (a, b) => BFloat(Float(a) + Float(b))).ToArray();
    }
    public static (int[] Ids, float[] Values) Top16(ushort[] logits, int vocabulary, int columns)
    {
        if (vocabulary < 16 || logits.Length != vocabulary * columns) throw new ArgumentException("Top16 geometry differs.");
        var ids = new int[16 * columns]; var values = new float[ids.Length];
        for (int t = 0; t < columns; t++)
        {
            int column = t;
            var selected = Enumerable.Range(0, vocabulary).OrderByDescending(r => Float(logits[column * vocabulary + r])).ThenBy(r => r).Take(16).ToArray();
            for (int k = 0; k < 16; k++) { ids[t * 16 + k] = selected[k]; values[t * 16 + k] = Float(logits[t * vocabulary + selected[k]]); }
        }
        return (ids, values);
    }
    public static int[] Predecessors(int[] candidates, int[] anchors, int positions, int lanes)
    {
        if (candidates.Length != 16 * positions * lanes || anchors.Length != lanes) throw new ArgumentException("Predecessor geometry differs.");
        var result = new int[candidates.Length];
        for (int l = 0; l < lanes; l++) for (int s = 0; s < positions; s++) for (int j = 0; j < 16; j++)
            result[(l * positions + s) * 16 + j] = s == 0 ? anchors[l] : candidates[(l * positions + s - 1) * 16 + j];
        return result;
    }
    public static int[] Walk(float[] scores, int[] candidates, int positions, int lanes)
    {
        if (scores.Length != 256 * positions * lanes || candidates.Length != 16 * positions * lanes)
            throw new ArgumentException("Selector-walk geometry differs.");
        var result = new int[positions * lanes];
        for (int l = 0; l < lanes; l++)
        {
            int predecessor = 0;
            for (int s = 0; s < positions; s++)
            {
                int slot = l * positions + s, best = 0;
                float maximum = float.NegativeInfinity;
                for (int i = 0; i < 16; i++)
                {
                    float value = scores[slot * 256 + predecessor * 16 + i];
                    if (value > maximum) { maximum = value; best = i; }
                }
                result[slot] = candidates[slot * 16 + best]; predecessor = best;
            }
        }
        return result;
    }
    public static double[] PlainNormIdeal(ushort[] input, ushort[] weight, int width, int rows, float epsilon)
    {
        if (input.Length != width * rows || weight.Length != width || epsilon <= 0) throw new ArgumentException("Plain norm geometry differs.");
        var result = new double[input.Length];
        for (int row = 0; row < rows; row++)
        {
            double squares = 0;
            for (int c = 0; c < width; c++) { double x = Float(input[row * width + c]); squares += x * x; }
            double inverse = 1 / Math.Sqrt(squares / width + epsilon);
            for (int c = 0; c < width; c++) result[row * width + c] = Float(input[row * width + c]) * inverse * Float(weight[c]);
        }
        return result;
    }
    public static double NormBound(double ideal, int width) => Math.Abs(ideal) * (BFloatUnitRoundoff + (width + 32) * FloatUnitRoundoff);
    public static double[] RopeIdeal(ushort[] input, int heads, int[] positions, double[] frequencies)
    {
        if (input.Length != positions.Length * heads * 128 || frequencies.Length != 64) throw new ArgumentException("Original DFlash RoPE geometry differs.");
        var result = new double[input.Length];
        const double inverseTwoPi = 1.59154943091895336e-01, twoPi = 6.28318530717958648e+00;
        for (int t = 0; t < positions.Length; t++) for (int h = 0; h < heads; h++) for (int j = 0; j < 64; j++)
        {
            double angle = positions[t] * frequencies[j];
            float reduced = (float)(angle - Math.Round(angle * inverseTwoPi, MidpointRounding.ToEven) * twoPi);
            double sine = Math.Sin(reduced), cosine = Math.Cos(reduced);
            int at = (t * heads + h) * 128 + j;
            double a = Float(input[at]), b = Float(input[at + 64]);
            result[at] = a * cosine - b * sine; result[at + 64] = b * cosine + a * sine;
        }
        return result;
    }
    public static double RopeBound(double ideal, double pairedInputAbsolute) =>
        BFloatUnitRoundoff * Math.Abs(ideal) + 32 * FloatUnitRoundoff * pairedInputAbsolute;
}
