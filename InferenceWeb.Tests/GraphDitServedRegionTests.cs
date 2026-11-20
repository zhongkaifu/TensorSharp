// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Graph;
using TensorSharp.GGML;
using Xunit;
using Half = System.Half;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// On ggml_cuda a program marked WideTokenProducts serves the diffusion-transformer regions as the DiT offload's fused nodes
/// (ggml_ops_dit_offload.h: tsg_h3_modulate, tsg_h3_gated_residual, tsg_cudnn_h3_attention, the unmerged LoRA GEMM, the BF16/F16
/// GEMM with the SwiGLU in its narrowing) and runs its wide products on cuBLAS. Each region here, at the widths those routes
/// take (a thousand rows and more), is served (the plan says how many regions were), and its result matches both an
/// independent double-precision oracle and the same program's reference ops (no attribute: nothing served, ggml's kernels).
/// The tolerances are the routes' roundings: BF16 activations for the cuBLAS products, F16 Q/K/V for the attention.
/// </summary>
public sealed class GraphDitServedRegionTests(ITestOutputHelper output)
{
    private const int Q4_K = 12, Q6_K = 14;   // ggml_type

    private sealed class Program : IDisposable
    {
        private readonly List<GCHandle> _pins = new();
        public readonly GraphProgramBuilder G;
        public readonly List<int> Nodes = new();
        public readonly List<IntPtr> Data = new();
        public readonly List<long> Bytes = new();

        public Program(bool wide)
        {
            G = new GraphProgramBuilder(record: true);
            if (wide) G.MarkWideTokenProducts();
        }

        private IntPtr Pin(Array values)
        {
            GCHandle h = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(h);
            return h.AddrOfPinnedObject();
        }

        public int Input(float[] values, params long[] ne) => Input(GraphFormat.F32, values, values.Length * 4L, ne);

        public int Input(int[] values, params long[] ne) => Input(GraphFormat.I32, values, values.Length * 4L, ne);

        private int Input(int format, Array values, long bytes, long[] ne)
        {
            int node = G.Leaf(GraphLeafRole.Input, format, ne, IntPtr.Zero, bytes);
            Nodes.Add(node);
            Data.Add(Pin(values));
            Bytes.Add(bytes);
            return node;
        }

        public int Weight(float[] values, params long[] ne) => G.Leaf(GraphLeafRole.Weight, GraphFormat.F32, ne, Pin(values), values.Length * 4L);

        public int Weight(int format, Array raw, long bytes, params long[] ne) => G.Leaf(GraphLeafRole.Weight, format, ne, Pin(raw), bytes);

        public float[] Run(int outputNode, long count, out int served)
        {
            G.Output(outputNode, count);
            GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
            using IGraphPlan plan = GgmlGraphExecutor.For(GgmlBackendType.Cuda).Compile(G.ToProgram(replayable: true, reorder: false));
            served = GgmlGraphExecutor.ServedRegions(plan);
            Assert.True(plan.Run(Nodes.ToArray(), Data.ToArray(), Bytes.ToArray()), plan.LastError);
            // A replay of the plan computes the same values (the served nodes' host segment rows are refilled each run).
            float[] first = (float[])plan.Outputs[0].Clone();
            Assert.True(plan.Run(Nodes.ToArray(), Data.ToArray(), Bytes.ToArray()), plan.LastError);
            Assert.Equal(first, plan.Outputs[0]);
            return first;
        }

        public void Dispose()
        {
            foreach (GCHandle h in _pins) h.Free();
        }
    }

    private delegate (int Output, double[] Want) Emit(Program p);

    private void Verify(string name, Emit emit, int servedRegions, double tolerance)
    {
        float[] served, reference;
        double[] want;
        int count, none;
        using (var p = new Program(wide: true))
        {
            (int y, want) = emit(p);
            served = p.Run(y, want.Length, out count);
        }
        using (var p = new Program(wide: false))
        {
            (int y, _) = emit(p);
            reference = p.Run(y, want.Length, out none);
        }
        Assert.Equal(servedRegions, count);
        Assert.Equal(0, none);
        double scale = want.Max(Math.Abs);
        double Rel(float[] got, double[] w) => Enumerable.Range(0, w.Length).Max(i => Math.Abs(got[i] - w[i])) / scale;
        Assert.All(served, v => Assert.True(float.IsFinite(v)));
        double servedErr = Rel(served, want), referenceErr = Rel(reference, want);
        double apart = Enumerable.Range(0, want.Length).Max(i => Math.Abs((double)served[i] - reference[i])) / scale;
        output.WriteLine($"{name}: served {count} regions; served vs oracle {servedErr:E3}, reference vs oracle {referenceErr:E3}, served vs reference {apart:E3} (bound {tolerance:E1})");
        Assert.True(servedErr <= tolerance, $"{name}: served vs oracle {servedErr:E3}");
        Assert.True(referenceErr <= tolerance, $"{name}: reference vs oracle {referenceErr:E3}");
        Assert.True(apart <= tolerance, $"{name}: served vs reference {apart:E3}");
    }

    // ================================================================ oracle helpers

    private static float[] Random(Random rng, long n, float scale = 1f)
        => Enumerable.Range(0, checked((int)n)).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();

    private static double[] D(float[] v) => v.Select(x => (double)x).ToArray();

    private static float Bf16(float v)
    {
        uint bits = BitConverter.SingleToUInt32Bits(v);
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(bits & 0xFFFF0000u);
    }

    private static double[] Product(float[] w, double[] x, int inRows, int outRows, int rows)
    {
        var y = new double[(long)outRows * rows];
        System.Threading.Tasks.Parallel.For(0, rows, r =>
        {
            for (int o = 0; o < outRows; o++)
            {
                double s = 0;
                for (int i = 0; i < inRows; i++) s += w[(long)o * inRows + i] * x[(long)r * inRows + i];
                y[(long)r * outRows + o] = s;
            }
        });
        return y;
    }

    /// <summary>Random Q4_K blocks for a [inRows, outRows] weight (ggml's block_q4_K: half d, half dmin, 12 bytes of 6-bit
    /// scales and mins, 128 bytes of nibbles) and their values, as ggml's dequantize_row_q4_K reads them.</summary>
    private static (byte[] Bytes, float[] Values) Q4K(Random rng, int inRows, int outRows)
    {
        const int Qk = 256, Block = 144;
        int blocks = inRows / Qk * outRows;
        var bytes = new byte[blocks * Block];
        var values = new float[inRows * outRows];
        static (int D, int M) ScaleMin(int j, byte[] q, int at)
            => j < 4 ? (q[at + j] & 63, q[at + j + 4] & 63)
                     : ((q[at + j + 4] & 0xF) | ((q[at + j - 4] >> 6) << 4), (q[at + j + 4] >> 4) | ((q[at + j] >> 6) << 4));
        for (int b = 0; b < blocks; b++)
        {
            int at = b * Block;
            Half d = (Half)(0.002f + 0.002f * (float)rng.NextDouble()), dmin = (Half)(0.001f * (float)rng.NextDouble());
            BitConverter.TryWriteBytes(bytes.AsSpan(at), BitConverter.HalfToUInt16Bits(d));
            BitConverter.TryWriteBytes(bytes.AsSpan(at + 2), BitConverter.HalfToUInt16Bits(dmin));
            for (int i = 4; i < Block; i++) bytes[at + i] = (byte)rng.Next(256);
            int y = b * Qk, q = at + 16, scales = at + 4, s = 0;
            for (int j = 0; j < Qk; j += 64, q += 32, s += 2)
            {
                var (sc1, m1) = ScaleMin(s, bytes, scales);
                var (sc2, m2) = ScaleMin(s + 1, bytes, scales);
                float d1 = (float)d * sc1, mm1 = (float)dmin * m1, d2 = (float)d * sc2, mm2 = (float)dmin * m2;
                for (int l = 0; l < 32; l++) values[y++] = d1 * (bytes[q + l] & 0xF) - mm1;
                for (int l = 0; l < 32; l++) values[y++] = d2 * (bytes[q + l] >> 4) - mm2;
            }
        }
        return (bytes, values);
    }

    /// <summary>Random Q6_K blocks for a [inRows, outRows] weight (ggml's block_q6_K: 128 bytes of low nibbles, 64 of high
    /// bit pairs, 16 int8 scales, half d) and their values, as ggml's dequantize_row_q6_K reads them.</summary>
    private static (byte[] Bytes, float[] Values) Q6K(Random rng, int inRows, int outRows)
    {
        const int Qk = 256, Block = 210;
        int blocks = inRows / Qk * outRows;
        var bytes = new byte[blocks * Block];
        var values = new float[inRows * outRows];
        for (int b = 0; b < blocks; b++)
        {
            int at = b * Block;
            for (int i = 0; i < 192; i++) bytes[at + i] = (byte)rng.Next(256);
            for (int i = 0; i < 16; i++) bytes[at + 192 + i] = (byte)(sbyte)rng.Next(-64, 64);
            Half d = (Half)(0.0001f + 0.0001f * (float)rng.NextDouble());
            BitConverter.TryWriteBytes(bytes.AsSpan(at + 208), BitConverter.HalfToUInt16Bits(d));
            int y = b * Qk, ql = at, qh = at + 128, sc = at + 192;
            for (int n = 0; n < Qk; n += 128, y += 128, ql += 64, qh += 32, sc += 8)
                for (int l = 0; l < 32; l++)
                {
                    int i = l / 16;
                    int q1 = ((bytes[ql + l] & 0xF) | ((bytes[qh + l] & 3) << 4)) - 32;
                    int q2 = ((bytes[ql + l + 32] & 0xF) | (((bytes[qh + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((bytes[ql + l] >> 4) | (((bytes[qh + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((bytes[ql + l + 32] >> 4) | (((bytes[qh + l] >> 6) & 3) << 4)) - 32;
                    values[y + l] = (float)d * (sbyte)bytes[sc + i] * q1;
                    values[y + l + 32] = (float)d * (sbyte)bytes[sc + i + 2] * q2;
                    values[y + l + 64] = (float)d * (sbyte)bytes[sc + i + 4] * q3;
                    values[y + l + 96] = (float)d * (sbyte)bytes[sc + i + 6] * q4;
                }
        }
        return (bytes, values);
    }

    /// <summary>A random BF16 [inRows, outRows] weight's bytes and values.</summary>
    private static (byte[] Bytes, float[] Values) Bf16Weight(Random rng, int inRows, int outRows)
    {
        float[] values = Random(rng, (long)inRows * outRows, 0.05f).Select(Bf16).ToArray();
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
            BitConverter.TryWriteBytes(bytes.AsSpan(2 * i), (ushort)(BitConverter.SingleToUInt32Bits(values[i]) >> 16));
        return (bytes, values);
    }

    // ================================================================ regions

    private static readonly GraphSegment[] Segments = { new(0, 120, 0), new(120, 1000, 6), new(1000, 1100, 12) };

    private static int ColumnOf(int row) => Segments.Single(s => row >= s.Start && row < s.End).Column;

    [GgmlFact(BackendType.GgmlCuda)]
    public void AdaLnModulateIsServedAndMatches()
    {
        const int hidden = 256, rows = 1100, columns = 18, shift = 3, scale = 4;
        const float eps = 1e-6f;
        var rng = new Random(3101);
        float[] x = Random(rng, hidden * rows, 3f), w = Random(rng, hidden), mods = Random(rng, hidden * columns, 0.5f);
        Verify("AdaLnModulate", p =>
        {
            int table = p.Input(GraphDitRegions.SegmentTable(Segments), 3, Segments.Length);
            int y = GraphDitRegions.AdaLnModulate(p.G, p.Input(x, hidden, rows), p.Weight(w, hidden), p.Input(mods, hidden, columns), table,
                Segments, hidden, shift, scale, eps);
            var want = new double[hidden * rows];
            for (int r = 0; r < rows; r++)
            {
                double ss = 0;
                for (int c = 0; c < hidden; c++) ss += (double)x[r * hidden + c] * x[r * hidden + c];
                double inv = 1 / Math.Sqrt(ss / hidden + eps);
                int col = ColumnOf(r);
                for (int c = 0; c < hidden; c++)
                {
                    double n = x[r * hidden + c] * inv * w[c];
                    want[r * hidden + c] = n + n * mods[(col + scale) * hidden + c] + mods[(col + shift) * hidden + c];
                }
            }
            return (y, want);
        }, servedRegions: 1, 1e-5);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public void GatedResidualIsServedAndMatches(bool bias, bool delta)
    {
        const int hidden = 256, rows = 1100, columns = 18, gate = 5;
        var rng = new Random(3201);
        float[] b = Random(rng, hidden * rows), mm = Random(rng, hidden * rows, 4f), bv = Random(rng, hidden),
            dv = Random(rng, hidden * rows, 0.25f), mods = Random(rng, hidden * columns);
        Verify($"GatedResidual bias={bias} delta={delta}", p =>
        {
            int table = p.Input(GraphDitRegions.SegmentTable(Segments), 3, Segments.Length);
            int y = GraphDitRegions.GatedResidual(p.G, p.Input(b, hidden, rows), p.Input(mm, hidden, rows), bias ? p.Weight(bv, hidden) : -1,
                delta ? p.Input(dv, hidden, rows) : -1, p.Input(mods, hidden, columns), table, Segments, hidden, gate);
            var want = new double[hidden * rows];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < hidden; c++)
                {
                    int i = r * hidden + c;
                    double u = (double)mm[i] + (bias ? bv[c] : 0) + (delta ? dv[i] : 0);
                    want[i] = b[i] + u * mods[(ColumnOf(r) + gate) * hidden + c];
                }
            return (y, want);
        }, servedRegions: 1, 1e-6);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void TableRopeQkNormAttentionIsServedAndMatches()
    {
        const int d = 64, heads = 2, rows = 300, rot = 48, inner = d * heads, half = rot / 2;
        const float eps = 1e-6f, vScale = 2f;   // MiniMax-H3's V scale at 300 keys (h3_flash_v_scale)
        float scale = 1f / MathF.Sqrt(d);
        var rng = new Random(3301);
        float[] qkv = Random(rng, 3 * inner * rows, 2f), qw = Random(rng, d), kw = Random(rng, d);
        var cos = new float[rot * rows];
        var ssin = new float[rot * rows];
        for (int s = 0; s < rows; s++)
            for (int j = 0; j < half; j++)
            {
                double a = s * 0.013 * (j + 1);
                cos[s * rot + j] = cos[s * rot + j + half] = (float)Math.Cos(a);
                ssin[s * rot + j] = -(float)Math.Sin(a);
                ssin[s * rot + j + half] = (float)Math.Sin(a);
            }
        double[] Rotated(int part, float[] weight)
        {
            var y = new double[inner * rows];
            for (int s = 0; s < rows; s++)
                for (int h = 0; h < heads; h++)
                {
                    int at = s * 3 * inner + part * inner + h * d;
                    double ss = 0;
                    for (int i = 0; i < d; i++) ss += (double)qkv[at + i] * qkv[at + i];
                    double inv = 1 / Math.Sqrt(ss / d + eps);
                    var n = new double[d];
                    for (int i = 0; i < d; i++) n[i] = qkv[at + i] * inv * weight[i];
                    for (int i = 0; i < d; i++)
                        y[s * inner + h * d + i] = i < rot ? n[i] * cos[s * rot + i] + n[i < half ? i + half : i - half] * ssin[s * rot + i] : n[i];
                }
            return y;
        }
        double[] qr = Rotated(0, qw), kr = Rotated(1, kw);
        var want = new double[inner * rows];
        System.Threading.Tasks.Parallel.For(0, rows, i =>
        {
            for (int h = 0; h < heads; h++)
            {
                var p = new double[rows];
                for (int j = 0; j < rows; j++)
                {
                    double s = 0;
                    for (int c = 0; c < d; c++) s += (double)(float)(Half)qr[i * inner + h * d + c] * (double)(float)(Half)kr[j * inner + h * d + c];
                    p[j] = s * scale;
                }
                double max = p.Max(), sum = 0;
                for (int j = 0; j < rows; j++) sum += p[j] = Math.Exp(p[j] - max);
                for (int c = 0; c < d; c++)
                {
                    double o = 0;
                    for (int j = 0; j < rows; j++) o += p[j] / sum * (double)(float)(Half)(qkv[j * 3 * inner + 2 * inner + h * d + c] / vScale);
                    want[i * inner + h * d + c] = o * vScale;
                }
            }
        });
        Verify("TableRopeQkNormAttention", p =>
        {
            int node = p.Input(qkv, 3 * inner, rows);
            int View(int part) => p.G.Op(GraphOp.View, stackalloc[] { node },
                stackalloc long[] { 3, d, heads, rows, 1, d * 4L, 3L * inner * 4, 0, part * inner * 4L });
            int y = GraphDitRegions.TableRopeQkNormAttention(p.G, View(0), View(1), View(2), p.Weight(qw, d), p.Weight(kw, d),
                p.Input(cos, rot, rows), p.Input(ssin, rot, rows), d, heads, rows, rot, eps, scale, vScale);
            return (y, want);
        }, servedRegions: 1, 5e-3);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(Q4_K, false, 256)]
    [InlineData(Q4_K, true, 256)]
    [InlineData(Q6_K, false, 256)]
    [InlineData(Q6_K, true, 256)]
    [InlineData(GraphFormat.BF16, false, 256)]    // K < 2 T: the BF16 weight copied into the K + R rows
    [InlineData(GraphFormat.BF16, true, 2304)]    // K >= 2 T: its product as it is, the expands accumulated into dst
    public void LoraLinearIsServedUnmergedAndMatches(int weightType, bool swigluInput, int inRows)
    {
        // qkv-like: three 128-row slots, a part on each (F16 factors); or fc2-like over fc1's SwiGLU, one slot.
        const int rows = 1100, rank = 16;
        int outRows = swigluInput ? 256 : 384, slot = swigluInput ? 0 : 128;
        int slotRows = slot > 0 ? slot : outRows, slots = outRows / slotRows;
        var rng = new Random((swigluInput ? 3402 : 3401) + 10 * weightType + inRows);
        var (wq, wv) = weightType switch
        {
            Q4_K => Q4K(rng, inRows, outRows),
            Q6_K => Q6K(rng, inRows, outRows),
            _ => Bf16Weight(rng, inRows, outRows),
        };
        float[] gv = Random(rng, 2 * inRows * rows, 2f);   // fc1's [gate | value] when swigluInput, else x itself
        float[] x = swigluInput ? new float[inRows * rows] : gv.Take(inRows * rows).ToArray();
        if (swigluInput)
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < inRows; j++)
                {
                    double gate = gv[r * 2 * inRows + j];
                    x[r * inRows + j] = (float)(gate / (1 + Math.Exp(-gate)) * gv[r * 2 * inRows + inRows + j]);
                }
        var downs = Enumerable.Range(0, slots).Select(_ => Random(rng, rank * inRows, 0.05f).Select(v => (float)(Half)v).ToArray()).ToArray();
        var ups = Enumerable.Range(0, slots).Select(_ => Random(rng, slotRows * rank, 0.05f).Select(v => (float)(Half)v).ToArray()).ToArray();
        // The oracle at the route's activation precision: the cuBLAS GEMM narrows x (or the SwiGLU of fc1) to BF16.
        double[] xb = x.Select(v => (double)Bf16(v)).ToArray();
        double[] want = Product(wv, xb, inRows, outRows, rows);
        for (int s = 0; s < slots; s++)
        {
            double[] delta = Product(ups[s], Product(downs[s], xb, inRows, rank, rows), rank, slotRows, rows);
            for (int r = 0; r < rows; r++)
                for (int o = 0; o < slotRows; o++) want[(long)r * outRows + s * slotRows + o] += delta[(long)r * slotRows + o];
        }
        Verify($"LoraLinear type={weightType} swiglu={swigluInput} K={inRows}", p =>
        {
            int xn = swigluInput ? p.G.Op(GraphOp.Swiglu, p.Input(gv, 2 * inRows, rows)) : p.Input(x, inRows, rows);
            int w = p.Weight(weightType, wq, wq.Length, inRows, outRows);
            var parts = new List<GraphDitRegions.LoraPart>();
            for (int s = 0; s < slots; s++)
            {
                ushort[] dh = downs[s].Select(v => BitConverter.HalfToUInt16Bits((Half)v)).ToArray();
                ushort[] uh = ups[s].Select(v => BitConverter.HalfToUInt16Bits((Half)v)).ToArray();
                parts.Add(new GraphDitRegions.LoraPart(s, p.Weight(GraphFormat.F16, dh, dh.Length * 2L, inRows, rank),
                    p.Weight(GraphFormat.F16, uh, uh.Length * 2L, rank, slotRows)));
            }
            return (GraphDitRegions.LoraLinear(p.G, xn, w, parts, outRows, slot, rows), want);
        }, servedRegions: 1, 1e-2);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(GraphFormat.BF16)]
    [InlineData(GraphFormat.F16)]
    public void SwigluFfnOnAHalfWidthWeightIsServedAndMatches(int format)
    {
        const int n = 256, ff = 512, rows = 1100;
        var rng = new Random(3500 + format);
        float Round(float v) => format == GraphFormat.BF16 ? Bf16(v) : (float)(Half)v;
        float[] x = Random(rng, n * rows), gu = Random(rng, 2 * ff * n, 0.1f).Select(Round).ToArray(),
            down = Random(rng, n * ff, 0.1f).Select(Round).ToArray(), b1 = Random(rng, 2 * ff, 0.5f), b2 = Random(rng, n, 0.5f);
        bool biased = format == GraphFormat.F16;   // the F16 route (the VAE's) carries fc1's bias in fc2's narrowing
        double[] xr = x.Select(v => (double)Round(v)).ToArray();
        double[] gv = Product(gu, xr, n, 2 * ff, rows);
        var act = new double[ff * rows];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < ff; j++)
            {
                double gate = gv[r * 2 * ff + j] + (biased ? b1[j] : 0), up = gv[r * 2 * ff + ff + j] + (biased ? b1[ff + j] : 0);
                act[r * ff + j] = Round((float)(gate / (1 + Math.Exp(-gate)) * up));
            }
        double[] want = Product(down, act, ff, n, rows);
        if (biased)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < n; c++) want[r * n + c] += b2[c];
        Verify($"SwigluFfn {(format == GraphFormat.BF16 ? "bf16" : "f16 biased")}", p =>
        {
            int W(float[] v, long ne0, long ne1)
            {
                ushort[] bits = format == GraphFormat.BF16
                    ? v.Select(f => (ushort)(BitConverter.SingleToUInt32Bits(f) >> 16)).ToArray()
                    : v.Select(f => BitConverter.HalfToUInt16Bits((Half)f)).ToArray();
                return p.Weight(format, bits, bits.Length * 2L, ne0, ne1);
            }
            int y = GraphDitRegions.SwigluFfn(p.G, p.Input(x, n, rows), W(gu, n, 2 * ff), W(down, ff, n),
                biased ? p.Weight(b1, 2 * ff) : -1, biased ? p.Weight(b2, n) : -1);
            return (y, want);
        }, servedRegions: 1, 1e-2);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void AWideProgramDropsAPowerOfTwoGuardAroundAProductCuBlasTakes()
    {
        // MiniMax-H3's q8_1 overflow guard (x / 1024, product, * 1024) around a Q4_K product: in a wide program the run takes the
        // product on cuBLAS BF16 and the guard pair goes, so the guarded program computes the unguarded one's bytes.
        const int inRows = 256, outRows = 128, rows = 1100;
        var rng = new Random(3601);
        var (wq, _) = Q4K(rng, inRows, outRows);
        float[] x = Random(rng, inRows * rows, 4f);
        float[] Run(bool guard)
        {
            using var p = new Program(wide: true);
            int xn = p.Input(x, inRows, rows), w = p.Weight(Q4_K, wq, wq.Length, inRows, outRows);
            int y = guard
                ? p.G.Op(GraphOp.Scale, stackalloc[] { p.G.Op(GraphOp.MulMat, w, p.G.Op(GraphOp.Scale, stackalloc[] { xn }, default, stackalloc[] { 1f / 1024f })) },
                    default, stackalloc[] { 1024f })
                : p.G.Op(GraphOp.MulMat, w, xn);
            return p.Run(y, outRows * rows, out _);
        }
        Assert.Equal(Run(guard: false), Run(guard: true));
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void AServedRegionRefusesASegmentTableItCannotRun()
    {
        // The table is an input fixed for a request: a run whose table no longer covers the rows fails, naming why, instead of
        // modulating with stale segments.
        const int hidden = 256, rows = 1100, columns = 18;
        var rng = new Random(3701);
        float[] x = Random(rng, hidden * rows), w = Random(rng, hidden), mods = Random(rng, hidden * columns);
        int[] table = GraphDitRegions.SegmentTable(Segments);
        using var p = new Program(wide: true);
        int tn = p.Input(table, 3, Segments.Length);
        int y = GraphDitRegions.AdaLnModulate(p.G, p.Input(x, hidden, rows), p.Weight(w, hidden), p.Input(mods, hidden, columns), tn,
            Segments, hidden, 0, 1, 1e-6f);
        p.G.Output(y, hidden * rows);
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        using IGraphPlan plan = GgmlGraphExecutor.For(GgmlBackendType.Cuda).Compile(p.G.ToProgram(replayable: true, reorder: false));
        Assert.Equal(1, GgmlGraphExecutor.ServedRegions(plan));
        Assert.True(plan.Run(p.Nodes.ToArray(), p.Data.ToArray(), p.Bytes.ToArray()), plan.LastError);
        table[4] = 900;   // the second segment now ends before the third begins
        Assert.False(plan.Run(p.Nodes.ToArray(), p.Data.ToArray(), p.Bytes.ToArray()));
        Assert.Contains("segment table", plan.LastError, StringComparison.Ordinal);
    }
}
