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
using Xunit.Abstractions;
using Half = System.Half;

namespace InferenceWeb.Tests;

/// <summary>
/// On ggml_metal a LoraLinear region of a WideTokenProducts program runs its update inside the base product where that moves
/// fewer bytes (ggml_ops_graph_builder.cpp try_lower_lora_linear): y = [W | U] . [x ; D x], W concatenated with the parts'
/// block-diagonal ups each run, never changed. The region is lowered (the plan says so) and matches an independent
/// double-precision oracle and the same program's reference ops (no attribute: ggml's ops beside the product); a region
/// whose update is cheaper beside its product keeps its ops; and with zero ups the lowered product is the weight's own
/// product bit for bit (the extension's terms are exact zeros summed after the weight's).
/// </summary>
public sealed class GraphDitMetalLoraTests(ITestOutputHelper output)
{
    private const int Q8_0 = 8;   // ggml_type

    private sealed class Program : IDisposable
    {
        private readonly List<GCHandle> _pins = new();
        private readonly List<IntPtr> _weights = new();
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

        public int Input(float[] values, params long[] ne)
        {
            int node = G.Leaf(GraphLeafRole.Input, GraphFormat.F32, ne, IntPtr.Zero, values.Length * 4L);
            Nodes.Add(node);
            Data.Add(Pin(values));
            Bytes.Add(values.Length * 4L);
            return node;
        }

        public int Weight(int format, Array raw, long bytes, params long[] ne)
        {
            IntPtr data = Pin(raw);
            _weights.Add(data);
            return G.Leaf(GraphLeafRole.Weight, format, ne, data, bytes);
        }

        public float[] Run(int outputNode, long count, out int lowered)
        {
            G.Output(outputNode, count);
            GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Metal);
            using IGraphPlan plan = GgmlGraphExecutor.For(GgmlBackendType.Metal).Compile(G.ToProgram(replayable: true, reorder: false));
            lowered = GgmlGraphExecutor.LoweredRegions(plan);
            Assert.True(plan.Run(Nodes.ToArray(), Data.ToArray(), Bytes.ToArray()), plan.LastError);
            float[] first = (float[])plan.Outputs[0].Clone();
            Assert.True(plan.Run(Nodes.ToArray(), Data.ToArray(), Bytes.ToArray()), plan.LastError);
            Assert.Equal(first, plan.Outputs[0]);
            return first;
        }

        /// <summary>A weight's device binding is keyed by the address of its host bytes and outlives the plan, so it is dropped
        /// before the pin is freed (<see cref="IGraphExecutor.Invalidate"/>): a later weight of the same size the GC places
        /// at that address would otherwise bind this one's.</summary>
        public void Dispose()
        {
            foreach (IntPtr data in _weights) GgmlGraphExecutor.For(GgmlBackendType.Metal).Invalidate(data);
            foreach (GCHandle h in _pins) h.Free();
        }
    }

    private static float[] Random(Random rng, long n, float scale)
        => Enumerable.Range(0, checked((int)n)).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();

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

    /// <summary>Random Q8_0 blocks (half d, 32 int8) for a [inRows, outRows] weight and their values.</summary>
    private static (byte[] Bytes, float[] Values) Q8(Random rng, int inRows, int outRows)
    {
        const int Qk = 32, Block = 34;
        int blocks = inRows / Qk * outRows;
        var bytes = new byte[blocks * Block];
        var values = new float[inRows * outRows];
        for (int b = 0; b < blocks; b++)
        {
            Half d = (Half)(0.0005f + 0.0005f * (float)rng.NextDouble());
            BitConverter.TryWriteBytes(bytes.AsSpan(b * Block), BitConverter.HalfToUInt16Bits(d));
            for (int i = 0; i < Qk; i++)
            {
                sbyte q = (sbyte)rng.Next(-127, 128);
                bytes[b * Block + 2 + i] = (byte)q;
                values[b * Qk + i] = (float)d * q;
            }
        }
        return (bytes, values);
    }

    private static (byte[] Bytes, float[] Values) F16(Random rng, int inRows, int outRows)
    {
        float[] values = Random(rng, (long)inRows * outRows, 0.05f).Select(v => (float)(Half)v).ToArray();
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(2 * i), BitConverter.HalfToUInt16Bits((Half)values[i]));
        return (bytes, values);
    }

    private static ushort[] Bits(float[] v) => v.Select(f => BitConverter.HalfToUInt16Bits((Half)f)).ToArray();

    private sealed record Case(int Format, int In, int Out, int Slot, int Rank, int Rows, float UpScale);

    private (double[] Want, double[] Base, Func<Program, int> Emit) Build(Case c, int seed)
    {
        var rng = new Random(seed);
        int slotRows = c.Slot > 0 ? c.Slot : c.Out, slots = c.Out / slotRows;
        var (wq, wv) = c.Format == Q8_0 ? Q8(rng, c.In, c.Out) : F16(rng, c.In, c.Out);
        float[] x = Random(rng, (long)c.In * c.Rows, 2f);
        var downs = Enumerable.Range(0, slots).Select(_ => Random(rng, c.Rank * c.In, 0.1f).Select(v => (float)(Half)v).ToArray()).ToArray();
        var ups = Enumerable.Range(0, slots).Select(_ => Random(rng, slotRows * c.Rank, c.UpScale).Select(v => (float)(Half)v).ToArray()).ToArray();
        // The oracle at ggml-metal's activation precision: its matrix kernels read F32 activations as F16.
        double[] xh = x.Select(v => (double)(float)(Half)v).ToArray();
        double[] baseY = Product(wv, xh, c.In, c.Out, c.Rows);
        var want = (double[])baseY.Clone();
        for (int s = 0; s < slots; s++)
        {
            double[] delta = Product(ups[s], Product(downs[s], xh, c.In, c.Rank, c.Rows), c.Rank, slotRows, c.Rows);
            for (int r = 0; r < c.Rows; r++)
                for (int o = 0; o < slotRows; o++) want[(long)r * c.Out + s * slotRows + o] += delta[(long)r * slotRows + o];
        }
        int Emit(Program p)
        {
            int xi = p.Input(x, c.In, c.Rows);
            int w = p.Weight(c.Format, wq, wq.Length, c.In, c.Out);
            var parts = new List<GraphDitRegions.LoraPart>();
            for (int s = 0; s < slots; s++)
            {
                ushort[] dh = Bits(downs[s]), uh = Bits(ups[s]);
                parts.Add(new GraphDitRegions.LoraPart(s, p.Weight(GraphFormat.F16, dh, dh.Length * 2L, c.In, c.Rank),
                    p.Weight(GraphFormat.F16, uh, uh.Length * 2L, c.Rank, slotRows)));
            }
            return GraphDitRegions.LoraLinear(p.G, xi, w, parts, c.Out, c.Slot, c.Rows);
        }
        return (want, baseY, Emit);
    }

    private static readonly Case[] Lowered =
    {
        new(Q8_0, 256, 384, 128, 32, 1100, 0.1f),    // a stacked qkv: three slots, block-diagonal ups
        new(Q8_0, 256, 512, 0, 64, 1100, 0.1f),      // fc1-like: one slot, wider than its input
        new(GraphFormat.F16, 256, 384, 128, 32, 1100, 0.1f),
    };

    public static IEnumerable<object[]> LoweredCases() => Enumerable.Range(0, Lowered.Length).Select(i => new object[] { i });

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(LoweredCases))]
    public void AWideLoraLinearRunsItsUpdateInsideTheProduct(int index)
    {
        Case c = Lowered[index];
        var (want, baseY, emit) = Build(c, 4100 + index);
        float[] lowered, reference;
        int count, none;
        using (var p = new Program(wide: true)) lowered = p.Run(emit(p), want.Length, out count);
        using (var p = new Program(wide: false)) reference = p.Run(emit(p), want.Length, out none);
        Assert.Equal(1, count);
        Assert.Equal(0, none);
        double scale = want.Max(Math.Abs);
        double Rel(float[] got) => Enumerable.Range(0, want.Length).Max(i => Math.Abs(got[i] - want[i])) / scale;
        double moved = Enumerable.Range(0, want.Length).Max(i => Math.Abs(want[i] - baseY[i])) / scale;
        double apart = Enumerable.Range(0, want.Length).Max(i => Math.Abs((double)lowered[i] - reference[i])) / scale;
        double loweredErr = Rel(lowered), referenceErr = Rel(reference);
        output.WriteLine($"{c}: lowered vs oracle {loweredErr:E3}, reference vs oracle {referenceErr:E3}, apart {apart:E3}; the update moves y by {moved:E3}");
        const double Bound = 5e-3;
        Assert.True(moved > 20 * Bound, $"the update moves y by {moved:E3} only");
        Assert.True(loweredErr <= Bound, $"lowered vs oracle {loweredErr:E3}");
        Assert.True(referenceErr <= Bound, $"reference vs oracle {referenceErr:E3}");
        Assert.True(apart <= Bound, $"lowered vs reference {apart:E3}");
    }

    [GgmlFact(BackendType.GgmlMetal)]
    public void AnUpdateCheaperBesideItsProductKeepsItsOps()
    {
        // fc2-like: the input four times the output, so copying the weight and the activations costs more than the delta.
        var c = new Case(Q8_0, 2048, 256, 0, 32, 1100, 0.05f);
        var (want, _, emit) = Build(c, 4200);
        using var p = new Program(wide: true);
        float[] got = p.Run(emit(p), want.Length, out int lowered);
        Assert.Equal(0, lowered);
        double scale = want.Max(Math.Abs);
        Assert.True(Enumerable.Range(0, want.Length).Max(i => Math.Abs(got[i] - want[i])) / scale <= 5e-3);
    }

    [GgmlFact(BackendType.GgmlMetal)]
    public void WithZeroUpsTheLoweredProductIsTheWeightsOwnBitForBit()
    {
        var c = new Case(Q8_0, 256, 384, 128, 32, 1100, 0f);
        var (want, _, emit) = Build(c, 4300);
        float[] lowered, plain;
        int count;
        using (var p = new Program(wide: true)) lowered = p.Run(emit(p), want.Length, out count);
        Assert.Equal(1, count);
        using (var p = new Program(wide: true))
        {
            // The same operands' plain product: the first leaves are x and the weight in the same order.
            var rng = new Random(4300);
            var (wq, _) = Q8(rng, c.In, c.Out);
            float[] x = Random(rng, (long)c.In * c.Rows, 2f);
            int xi = p.Input(x, c.In, c.Rows);
            int w = p.Weight(Q8_0, wq, wq.Length, c.In, c.Out);
            plain = p.Run(p.G.Op(GraphOp.MulMat, w, xi), want.Length, out _);
        }
        int differ = Enumerable.Range(0, plain.Length).Count(i => BitConverter.SingleToInt32Bits(plain[i]) != BitConverter.SingleToInt32Bits(lowered[i]));
        Assert.True(differ == 0, $"{differ} of {plain.Length} values differ from the weight's own product");
    }
}
