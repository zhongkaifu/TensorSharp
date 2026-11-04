// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Linq;

namespace TensorSharp.Validation;

/// <summary>Independent FP64 causal attention oracle over represented BF16 Q/K/V.
/// The dense fixture's K repeats every31 keys, while V is nonperiodic. Grouping
/// equal scores and prefix-summing V gives the same full softmax algebra without
/// a quadratic CPU oracle at the measured32K context. P is NOT narrowed in this
/// ideal oracle: fixed prior attention bounds cover the native BF16 P/outputs.
/// Exact original-GPU comparison independently checks the complete native math.</summary>
internal static class NinferPrefillAttentionReference
{
    internal const int Hd = 256, Heads = 24, KvHeads = 4, Group = 6;
    internal const double Absolute = 0.015625, Relative = 0.005;
    internal sealed record Case(string Name, int Rows, int Past, string Pattern, int ExpectedSplits)
    {
        internal int Keys => checked(Past + Rows);
        internal int Sentinel => Past + Rows / 2;
    }
    internal static Case[] Cases() => new[] {
        new Case("constant-t64-p0",64,0,"constant",4),
        new Case("sentinel-t65-p63",65,63,"sentinel",3),
        new Case("sentinel-t129-p255",129,255,"sentinel",2),
        new Case("dense-t64-p1",64,1,"dense",4),
        new Case("dense-t129-p127",129,127,"dense",2),
        new Case("dense-t512-p0",512,0,"dense",4),
        new Case("dense-t512-p31744",512,31744,"dense",4),
        new Case("constant-t416-p32256",416,32256,"constant",1)
    };
    internal static ushort Bf(float value) { uint bits = (uint)BitConverter.SingleToInt32Bits(value); return (ushort)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16); }
    internal static float Float(ushort bits) => BitConverter.Int32BitsToSingle(bits << 16);
    internal static float Round(float value) => Float(Bf(value));
    internal static float Constant(int head, int dim) => ((head * 17 + dim * 3) % 31 + 1) / 64f;
    internal static ushort Query(Case c, int token, int head, int dim)
    {
        if (c.Pattern != "dense") return 0;
        if (dim % 251 == 0) return 0x8000;
        return Bf(((dim * 17 + head * 13 + token * 7) % 65 - 32) / 256f);
    }
    internal static ushort Key(Case c, int key, int head, int dim) => c.Pattern == "dense"
        ? Bf((((key % 31) * 7 + dim * 11 + head * 13) % 31 - 15) / 128f) : (ushort)0;
    internal static ushort Value(Case c, int key, int head, int dim)
    {
        if (c.Pattern == "constant" || c.Pattern == "sentinel" && dim % 17 != 0) return Bf(Constant(head, dim));
        if (c.Pattern == "sentinel") return Bf(key == c.Sentinel ? 1 : 0);
        uint bits = unchecked((uint)key * 0x9e3779b9U ^ (uint)dim * 0x85ebca6bU ^ (uint)head * 0xc2b2ae35U);
        bits ^= bits >> 16;
        return Bf(((int)(bits % 127) - 63) / 64f);
    }
    internal static float Gate(Case c, int token, int head, int dim) => c.Pattern == "dense"
        ? Round((head % 5 - 2) * .5f + (token % 3) * .125f + (dim % 7) * .03125f) : 0;

    internal sealed record Fixture(Case Case, ushort[] Query, float[] QueryF32, float[] Gate,
        ushort[] LogicalK, ushort[] LogicalV, double[] IdealCore, float[] IdealGated);

    internal static Fixture Make(Case c)
    {
        int queryElements = checked(c.Rows * Heads * Hd), kvElements = checked(c.Keys * KvHeads * Hd);
        ushort[] q = new ushort[queryElements], k = new ushort[kvElements], v = new ushort[kvElements];
        float[] q32 = new float[queryElements], gate = new float[queryElements];
        for (int key = 0; key < c.Keys; key++) for (int h = 0; h < KvHeads; h++) for (int d = 0; d < Hd; d++)
        { int index = (key * KvHeads + h) * Hd + d; k[index] = Key(c, key, h, d); v[index] = Value(c, key, h, d); }
        for (int token = 0; token < c.Rows; token++) for (int h = 0; h < Heads; h++) for (int d = 0; d < Hd; d++)
        { int index = (token * Heads + h) * Hd + d; q[index] = Query(c, token, h, d); q32[index] = Float(q[index]); gate[index] = Gate(c, token, h, d); }
        var ideal = Ideal(c, q, k, v); var gated = new float[queryElements];
        for (int i = 0; i < gated.Length; i++) gated[i] = Round(Round((float)ideal[i]) * (1f / (1f + MathF.Exp(-gate[i]))));
        var fixture = new Fixture(c, q, q32, gate, k, v, ideal, gated);
        ValidateAlgebra(fixture);
        return fixture;
    }

    private static double[] Ideal(Case c, ushort[] q, ushort[] k, ushort[] v)
    {
        var result = new double[c.Rows * Heads * Hd];
        var prefix = new double[31 * KvHeads * Hd];
        void Add(int key)
        {
            int r = key % 31;
            for (int h = 0; h < KvHeads; h++) for (int d = 0; d < Hd; d++)
                prefix[(r * KvHeads + h) * Hd + d] += Float(v[(key * KvHeads + h) * Hd + d]);
        }
        for (int key = 0; key < c.Past; key++) Add(key);
        double[] scores = new double[31], weights = new double[31]; int[] counts = new int[31];
        for (int t = 0; t < c.Rows; t++)
        {
            Add(c.Past + t); int visible = c.Past + t + 1;
            for (int r = 0; r < 31; r++) counts[r] = visible > r ? (visible - 1 - r) / 31 + 1 : 0;
            for (int h = 0; h < Heads; h++)
            {
                int kh = h / Group; double max = double.NegativeInfinity;
                for (int r = 0; r < 31; r++)
                {
                    double dot = 0;
                    for (int d = 0; d < Hd; d++) dot += (double)Float(q[(t * Heads + h) * Hd + d]) * Float(k[(r * KvHeads + kh) * Hd + d]);
                    scores[r] = dot / 16;
                    if (counts[r] > 0) max = Math.Max(max, scores[r]);
                }
                double denominator = 0;
                for (int r = 0; r < 31; r++) { weights[r] = counts[r] == 0 ? 0 : Math.Exp(scores[r] - max); denominator += counts[r] * weights[r]; }
                for (int d = 0; d < Hd; d++)
                {
                    double numerator = 0;
                    for (int r = 0; r < 31; r++) numerator += weights[r] * prefix[(r * KvHeads + kh) * Hd + d];
                    double value = numerator / denominator;
                    if (!double.IsFinite(value)) throw new InvalidOperationException("Nonfinite independent FP64 attention ideal.");
                    result[(t * Heads + h) * Hd + d] = value;
                }
            }
        }
        return result;
    }

    private static void ValidateAlgebra(Fixture f)
    {
        // One complete naive logical-key dot/softmax/PV oracle per fixture,
        // including32K, validates the score-grouping algebra independently.
        int t = f.Case.Rows - 1, h = 7, d = 17, kh = h / Group, visible = f.Case.Past + t + 1;
        var scores = new double[visible]; double max = double.NegativeInfinity;
        for (int key = 0; key < visible; key++)
        {
            double dot = 0;
            for (int j = 0; j < Hd; j++) dot += (double)Float(f.Query[(t * Heads + h) * Hd + j]) * Float(f.LogicalK[(key * KvHeads + kh) * Hd + j]);
            scores[key] = dot / 16; max = Math.Max(max, scores[key]);
        }
        double numerator = 0, denominator = 0;
        for (int key = 0; key < visible; key++) { double weight = Math.Exp(scores[key] - max); denominator += weight; numerator += weight * Float(f.LogicalV[(key * KvHeads + kh) * Hd + d]); }
        double actual = f.IdealCore[(t * Heads + h) * Hd + d];
        if (Math.Abs(numerator / denominator - actual) > 1e-12) throw new InvalidOperationException("Grouped FP64 oracle differs from the naive logical-key oracle.");
        if (f.Case.Pattern == "constant")
            for (int token = 0; token < f.Case.Rows; token++) for (int head = 0; head < Heads; head++) for (int dim = 0; dim < Hd; dim++)
                if (!double.IsFinite(f.IdealCore[(token * Heads + head) * Hd + dim]) || f.IdealCore[(token * Heads + head) * Hd + dim] != Constant(head / Group, dim))
                    throw new InvalidOperationException("Uniform constant-V attention ideal changed.");
    }
}
