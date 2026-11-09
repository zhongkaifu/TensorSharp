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
/// The end of a DFlash2 draft on TensorSharp's Metal kernels (tsm::df_tail_run, tsm/kernels/tsm_df_tail.metal), stage by
/// stage, on ggml's own bytes:
/// <list type="bullet">
/// <item>The q5_K head over the proposal rows (rows 1..gamma) on the matrix units, against a double-precision product over
/// ggml's dequantization of the same blocks with the activations rounded to fp16 (the operand the kernel uses).</item>
/// <item>The top-16 of each row of the kernel's own logits: the same columns in the same order (value, then index).</item>
/// <item>The selector lattice over the kernel's candidates: 16 anchor scores against the anchor, then a
/// [16 predecessor][16 candidate] matrix per later position, from the q4_K codebooks and the selector projection, with
/// the head logit through the logit scale and tanh softcap.</item>
/// </list>
/// The vocabulary is not a multiple of 8 rows (the last row tile is partial), and the top-k runs over one to three
/// 4096-column blocks.
/// </summary>
public sealed class TsmDfTailTests
{
    private const int Q4K = (int)SyntheticGguf.GgmlType.Q4_K, Q5K = 13;
    private const int Rank = 256, TopK = 16;
    private readonly ITestOutputHelper _output;
    public TsmDfTailTests(ITestOutputHelper output) => _output = output;

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(7, 9187, 512, 30f)]
    [InlineData(3, 4111, 768, 0f)]
    [InlineData(1, 300, 256, 15f)]
    public void DraftTail_AnswersAsTheReference(int gamma, int vocab, int hidden, float softcap)
    {
        var rng = new Random(gamma * 7919 + vocab);
        const float headScale = 0.75f, predScale = 1.25f, succScale = 0.5f, logitScale = 0.9f;

        // K-quant blocks of random codes, scales and mins with small super-block scales, and the values ggml reads from them.
        (byte[] bytes, float[] values) KQuant(int type, int cols, int rows)
        {
            int blockBytes = type == Q5K ? 176 : 144, blocks = cols / 256;
            var b = new byte[(long)rows * blocks * blockBytes];
            rng.NextBytes(b);
            for (long o = 0; o < b.LongLength; o += blockBytes)
            {
                BitConverter.TryWriteBytes(b.AsSpan((int)o), BitConverter.HalfToUInt16Bits((System.Half)(0.0004 + rng.NextDouble() * 0.0008)));
                BitConverter.TryWriteBytes(b.AsSpan((int)o + 2), BitConverter.HalfToUInt16Bits((System.Half)(0.0002 + rng.NextDouble() * 0.0005)));
            }
            var v = new float[(long)rows * cols];
            var hb = GCHandle.Alloc(b, GCHandleType.Pinned);
            var hv = GCHandle.Alloc(v, GCHandleType.Pinned);
            try { GgmlGgufTensorDequant.DequantizeToFloat32Native(type, hb.AddrOfPinnedObject(), hv.AddrOfPinnedObject(), v.LongLength); }
            finally { hb.Free(); hv.Free(); }
            return (b, v);
        }
        var (headB, head) = KQuant(Q5K, hidden, vocab);
        var (predB, pred) = KQuant(Q4K, Rank, vocab);
        var (succB, succ) = KQuant(Q4K, Rank, vocab);
        var h = new float[(gamma + 1) * hidden];
        for (int i = 0; i < h.Length; i++) h[i] = (float)((rng.NextDouble() * 2 - 1) * (1 + (i / hidden) % 2));
        var ph = new float[gamma * Rank];
        for (int i = 0; i < ph.Length; i++) ph[i] = (float)(rng.NextDouble() * 2 - 1);
        int anchor = rng.Next(vocab);

        var cand = new int[gamma * TopK];
        var scores = new float[TopK + TopK * TopK * (gamma - 1)];
        var logits = new float[gamma * vocab];
        long before = GgmlBasicOps.TsmDfTailCount();
        Assert.True(GgmlBasicOps.TsmDfTailProbe(hidden, vocab, Rank, gamma, headB, predB, succB, h, ph, anchor, headScale,
            predScale, succScale, logitScale, softcap, cand, scores, logits), GgmlBasicOps.LastNativeError("probe failed"));
        Assert.Equal(before + 1, GgmlBasicOps.TsmDfTailCount());

        // 1. the head: proposal row e is h's row e + 1, rounded to fp16
        double headWorst = 0;
        for (int e = 0; e < gamma; e++)
        {
            var x = new double[hidden];
            for (int k = 0; k < hidden; k++) x[k] = (double)(System.Half)h[(e + 1) * hidden + k];
            for (int v = 0; v < vocab; v++)
            {
                double s = 0, mag = 0;
                for (int k = 0; k < hidden; k++)
                {
                    double t = head[(long)v * hidden + k] * x[k];
                    s += t;
                    mag += Math.Abs(t);
                }
                s *= headScale;
                mag *= headScale;
                headWorst = Math.Max(headWorst, Math.Abs(logits[e * vocab + v] - s) / (mag + 1e-6));
            }
        }

        // 2. the top-16 of the kernel's own logits: value descending, then the lower column
        for (int e = 0; e < gamma; e++)
        {
            var order = new int[vocab];
            for (int v = 0; v < vocab; v++) order[v] = v;
            int row = e;
            Array.Sort(order, (a, b) =>
            {
                int c = logits[row * vocab + b].CompareTo(logits[row * vocab + a]);
                return c != 0 ? c : a.CompareTo(b);
            });
            for (int j = 0; j < TopK; j++)
                Assert.True(order[j] == cand[e * TopK + j], $"row {e} pick {j}: kernel {cand[e * TopK + j]}, expected {order[j]}");
        }

        // 3. the lattice over the kernel's candidates
        double latticeWorst = 0;
        double Unary(int e, int c)
        {
            double u = logits[e * vocab + c] * (double)logitScale;
            return softcap > 0 ? softcap * Math.Tanh(u / softcap) : u;
        }
        double Score(int e, int prev, int c, out double mag)
        {
            double dot = 0;
            mag = 0;
            for (int i = 0; i < Rank; i++)
            {
                double m = predScale * (double)pred[(long)prev * Rank + i] * ph[e * Rank + i];
                double t = succ[(long)c * Rank + i] * m;
                dot += t;
                mag += Math.Abs(t);
            }
            double u = Unary(e, c);
            mag = Math.Abs(u) + succScale * mag;
            return u + succScale * dot;
        }
        for (int c = 0; c < TopK; c++)
        {
            double want = Score(0, anchor, cand[c], out double mag);
            latticeWorst = Math.Max(latticeWorst, Math.Abs(scores[c] - want) / (mag + 1e-6));
        }
        for (int e = 1; e < gamma; e++)
            for (int p = 0; p < TopK; p++)
                for (int c = 0; c < TopK; c++)
                {
                    double want = Score(e, cand[(e - 1) * TopK + p], cand[e * TopK + c], out double mag);
                    float got = scores[TopK + (e - 1) * TopK * TopK + p * TopK + c];
                    latticeWorst = Math.Max(latticeWorst, Math.Abs(got - want) / (mag + 1e-6));
                }
        _output.WriteLine($"gamma {gamma}, vocab {vocab}, hidden {hidden}: head within {headWorst:E2}, lattice within {latticeWorst:E2} (relative to the sum of magnitudes)");
        Assert.True(headWorst < 1e-5, $"head {headWorst:E3}");
        Assert.True(latticeWorst < 1e-5, $"lattice {latticeWorst:E3}");
    }
}
