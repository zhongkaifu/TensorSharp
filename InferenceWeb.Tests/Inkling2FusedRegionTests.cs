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
using TensorSharp.GGML;
using TensorSharp.Graph;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Inkling 2's fused regions (GraphFusion.Inkling*) on ggml_cuda, each against a double-precision oracle at Inkling 2 Mini's
/// shapes: the region holds a placeholder op of the result's shape, so a region the kernel does not serve fails the oracle
/// as well as the served count. The model's reference ops are checked separately (Inkling2RuntimeTests).
/// </summary>
public sealed class Inkling2FusedRegionTests(ITestOutputHelper output)
{
    private sealed class Program : IDisposable
    {
        private readonly List<GCHandle> _pins = new();
        public readonly GraphProgramBuilder G = new(record: true);
        public readonly List<int> Nodes = new();
        public readonly List<IntPtr> Data = new();
        public readonly List<long> Bytes = new();

        public IntPtr Pin(Array values)
        {
            GCHandle h = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(h);
            return h.AddrOfPinnedObject();
        }

        public int Input(float[] values, params long[] ne) => Input(GraphFormat.F32, values, values.Length * 4L, ne);
        public int Input(int[] values, params long[] ne) => Input(GraphFormat.I32, values, values.Length * 4L, ne);
        public int InputF16(Half[] values, params long[] ne) => Input(GraphFormat.F16, values, values.Length * 2L, ne);

        private int Input(int format, Array values, long bytes, long[] ne)
        {
            int node = G.Leaf(GraphLeafRole.Input, format, ne, IntPtr.Zero, bytes);
            Nodes.Add(node);
            Data.Add(Pin(values));
            Bytes.Add(bytes);
            return node;
        }

        public int Weight(float[] values, params long[] ne) => G.Leaf(GraphLeafRole.Weight, GraphFormat.F32, ne, Pin(values), values.Length * 4L);

        public int State(float[] values, params long[] ne)
            => G.Leaf(GraphLeafRole.State, GraphFormat.F32, ne, Pin(values), values.Length * 4L, handle: values);

        public int View(int x, long ne0, long ne1, long ne2, long nb1, long nb2, long offset)
            => G.Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 3, ne0, ne1, ne2, 1, nb1, nb2, 0, offset });

        /// <summary>Runs the program on ggml_cuda <paramref name="runs"/> times; the outputs of each run.</summary>
        public List<float[][]> Run(int[] outputs, long[] counts, int runs, out int served)
        {
            for (int i = 0; i < outputs.Length; i++) G.Output(outputs[i], counts[i]);
            GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
            using IGraphPlan plan = GgmlGraphExecutor.For(GgmlBackendType.Cuda).Compile(G.ToProgram(replayable: true, reorder: false));
            var all = new List<float[][]>();
            served = 0;
            for (int r = 0; r < runs; r++)
            {
                Assert.True(plan.Run(Nodes.ToArray(), Data.ToArray(), Bytes.ToArray()), plan.LastError);
                served = GgmlGraphExecutor.ServedRegions(plan);
                all.Add(Enumerable.Range(0, outputs.Length).Select(i => (float[])plan.Outputs[i].Clone()).ToArray());
            }
            return all;
        }

        public void Dispose()
        {
            foreach (GCHandle h in _pins) h.Free();
        }
    }

    private static float[] Rand(Random rng, long n, float scale = 1f)
        => Enumerable.Range(0, checked((int)n)).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();

    private void Check(string name, float[] got, double[] want, double tolerance)
    {
        double scale = Math.Max(1e-30, want.Max(Math.Abs));
        double err = Enumerable.Range(0, want.Length).Max(i => Math.Abs(got[i] - want[i])) / scale;
        output.WriteLine($"{name}: max |got - oracle| / max |oracle| = {err:E2} (bound {tolerance:E1})");
        Assert.All(got.Take(want.Length), v => Assert.True(float.IsFinite(v)));
        Assert.True(err <= tolerance, $"{name}: {err:E2} > {tolerance:E1}");
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    [CudaTheory(GgmlBackend = BackendType.GgmlCuda)]
    [InlineData(true)]
    [InlineData(false)]
    public void ClipMatchesItsOracle(bool scaled)
    {
        var rng = new Random(1);
        int d = 128, heads = scaled ? 24 : 8, rows = 3;
        const float threshold = 22.627f;
        float[] x = Rand(rng, (long)d * heads * rows, 4f), w = Rand(rng, (long)d * heads, 1.5f);
        var want = new double[x.Length];
        for (int v = 0; v < heads * rows; v++)
        {
            double ss = 0;
            for (int e = 0; e < d; e++) { double a = x[v * d + e] * (scaled ? w[(v % heads) * d + e] : 1); ss += a * a; }
            double factor = Math.Clamp(1.0 / (Math.Sqrt(ss) + 1e-6) * threshold, 0, 1);
            for (int e = 0; e < d; e++) want[v * d + e] = x[v * d + e] * (scaled ? w[(v % heads) * d + e] : 1) * factor;
        }
        using var p = new Program();
        int xi = p.Input(x, d, heads, rows);
        int wi = p.Weight(w, d, heads);
        if (scaled) p.G.BeginFused(GraphFusion.InklingClip, stackalloc[] { xi, wi }, stackalloc long[] { 1 }, stackalloc[] { threshold });
        else p.G.BeginFused(GraphFusion.InklingClip, stackalloc[] { xi }, stackalloc long[] { 0 }, stackalloc[] { threshold });
        int y = p.G.Op(GraphOp.Scale, stackalloc[] { xi }, default, stackalloc[] { 0f });
        p.G.EndFused(stackalloc[] { y });
        var got = p.Run(new[] { y }, new[] { (long)x.Length }, 1, out int served);
        Assert.Equal(1, served);
        Check($"clip scaled={scaled}", got[0][0], want, 1e-5);
    }

    [CudaTheory(GgmlBackend = BackendType.GgmlCuda)]
    [InlineData(0)]
    [InlineData(8)]
    public void ConvolutionMatchesItsOracleAndAdvancesTheWindow(int heads)
    {
        var rng = new Random(2);
        int channels = 1024, rows = 3, taps = 4, hist = taps - 1;
        float scale = 1f / MathF.Sqrt(taps * (heads > 0 ? 2 : 1) + 1);
        float[] x = Rand(rng, (long)channels * rows), kern = Rand(rng, (long)taps * channels, 0.5f);
        float[] dyn = Rand(rng, (long)taps * Math.Max(1, heads) * rows, 0.5f);
        float[] window = Rand(rng, (long)hist * channels), state = (float[])window.Clone();
        // Five runs, each reading the window the one before left: the third is captured as one CUDA graph and the last two
        // replay it (tsg_inkling_compute).
        const int runs = 5;
        var expected = new List<double[]>();
        var win = window.Select(v => (double)v).ToArray();
        for (int run = 0; run < runs; run++)
        {
            var y = new double[x.Length];
            double In(int c, int tau) => tau < hist ? win[c * hist + tau] : x[(tau - hist) * channels + c];
            for (int c = 0; c < channels; c++)
            {
                for (int t = 0; t < rows; t++)
                {
                    double conv = 0;
                    for (int i = 0; i < taps; i++) conv += In(c, t + i) * kern[c * taps + i];
                    if (heads > 0)
                        for (int lag = 0; lag < taps; lag++)
                            conv += In(c, hist - lag + t) * dyn[t * taps * heads + c / (channels / heads) * taps + lag];
                    y[t * channels + c] = (x[t * channels + c] + conv) * scale;
                }
            }
            var next = new double[win.Length];
            for (int c = 0; c < channels; c++)
                for (int j = 0; j < hist; j++) next[c * hist + j] = In(c, rows + j);
            win = next;
            expected.Add(y);
        }
        using var p = new Program();
        int xi = p.Input(x, channels, rows);
        int wi = p.State(state, hist, channels);
        int ki = p.Weight(kern, taps, channels);
        int di = heads > 0 ? p.Input(dyn, taps * heads, rows) : -1;
        if (heads > 0)
            p.G.BeginFused(GraphFusion.InklingConv, stackalloc[] { xi, wi, ki, di }, stackalloc long[] { heads }, stackalloc[] { scale });
        else
            p.G.BeginFused(GraphFusion.InklingConv, stackalloc[] { xi, wi, ki }, stackalloc long[] { 0 }, stackalloc[] { scale });
        int yo = p.G.Op(GraphOp.Scale, stackalloc[] { xi }, default, stackalloc[] { 0f });
        p.G.EndFused(stackalloc[] { yo });
        var got = p.Run(new[] { yo }, new[] { (long)x.Length }, runs, out int served);
        Assert.Equal(1, served);
        for (int run = 0; run < runs; run++)
            Check($"conv heads={heads} run {run + 1}", got[run][0], expected[run], 1e-5);
    }

    [CudaTheory(GgmlBackend = BackendType.GgmlCuda)]
    [InlineData(1, 4)]
    [InlineData(3, 4)]
    [InlineData(4, 7)]
    public void MixMatchesItsOracle(int queries, int n)
    {
        var rng = new Random(3);
        int h = 2304, rows = 2, slots = 7;
        const float eps = 1e-6f;
        float scale = 1f / MathF.Sqrt(h);
        float[] stream = Rand(rng, (long)slots * h * rows, 2f), query = Rand(rng, (long)h * queries, 1f);
        var want = new double[(long)h * queries * rows];
        for (int r = 0; r < rows; r++)
        {
            var normed = new double[n, h];
            for (int s = 0; s < n; s++)
            {
                double ss = 0;
                for (int e = 0; e < h; e++) { double v = stream[(r * slots + s) * h + e]; ss += v * v; }
                double f = 1.0 / Math.Sqrt(ss / h + eps);
                for (int e = 0; e < h; e++) normed[s, e] = stream[(r * slots + s) * h + e] * f;
            }
            for (int q = 0; q < queries; q++)
            {
                var sc = new double[n];
                for (int s = 0; s < n; s++) { double dot = 0; for (int e = 0; e < h; e++) dot += query[q * h + e] * normed[s, e]; sc[s] = dot * scale; }
                double m = sc.Max(), sum = sc.Sum(v => Math.Exp(v - m));
                for (int e = 0; e < h; e++)
                {
                    double o = 0;
                    for (int s = 0; s < n; s++) o += normed[s, e] * Math.Exp(sc[s] - m) / sum;
                    want[((long)r * queries + q) * h + e] = o;
                }
            }
        }
        using var p = new Program();
        int si = p.Input(stream, (long)slots * h, rows);
        int states = p.View(si, h, n, rows, h * 4L, slots * h * 4L, 0);
        int qi = p.Weight(query, h, queries);
        p.G.BeginFused(GraphFusion.InklingMix, stackalloc[] { states, qi }, default, stackalloc[] { eps, scale });
        int placeholder = p.G.Op(GraphOp.Scale, stackalloc[] { p.G.Op(GraphOp.Cont, p.View(si, h, queries, rows, h * 4L, slots * h * 4L, 0)) },
            default, stackalloc[] { 0f });
        p.G.EndFused(stackalloc[] { placeholder });
        var got = p.Run(new[] { placeholder }, new[] { (long)want.Length }, 1, out int served);
        Assert.Equal(1, served);
        Check($"mix queries={queries} n={n}", got[0][0], want, 1e-5);
    }

    [CudaTheory(GgmlBackend = BackendType.GgmlCuda)]
    [InlineData(2)]
    [InlineData(40)]
    public void ControllerMatchesItsOracle(int rows)
    {
        var rng = new Random(4);
        int h = 2304, rank = 64;
        const float eps = 1e-6f;
        float[] x = Rand(rng, (long)h * rows, 2f), m = Rand(rng, (long)h * rows, 2f);
        float[] down = Rand(rng, 2L * h * rank, 0.05f), gain = Rand(rng, (long)rank * h, 0.2f), forget = Rand(rng, (long)rank * h, 0.2f);
        var want = new double[(long)h * rows];
        for (int r = 0; r < rows; r++)
        {
            double fx = 1.0 / Math.Sqrt(Enumerable.Range(0, h).Sum(e => (double)x[r * h + e] * x[r * h + e]) / h + eps);
            double fm = 1.0 / Math.Sqrt(Enumerable.Range(0, h).Sum(e => (double)m[r * h + e] * m[r * h + e]) / h + eps);
            var g = new double[rank];
            for (int k = 0; k < rank; k++)
            {
                double s = 0;
                for (int i = 0; i < h; i++) s += down[(long)k * 2 * h + i] * x[r * h + i] * fx;
                for (int i = 0; i < h; i++) s += down[(long)k * 2 * h + h + i] * m[r * h + i] * fm;
                g[k] = s;
            }
            for (int c = 0; c < h; c++)
            {
                double ga = 0, fo = 0;
                for (int k = 0; k < rank; k++) { ga += gain[(long)c * rank + k] * g[k]; fo += forget[(long)c * rank + k] * g[k]; }
                want[r * h + c] = x[r * h + c] - Math.Tanh(fo) * x[r * h + c] + Math.Tanh(ga) * m[r * h + c];
            }
        }
        using var p = new Program();
        int xi = p.Input(x, h, rows), mi = p.Input(m, h, rows);
        int di = p.Weight(down, 2 * h, rank), gi = p.Weight(gain, rank, h), fi = p.Weight(forget, rank, h);
        p.G.BeginFused(GraphFusion.InklingControl, stackalloc[] { xi, mi, di, gi, fi }, default, stackalloc[] { eps });
        int y = p.G.Op(GraphOp.Scale, stackalloc[] { xi }, default, stackalloc[] { 0f });
        p.G.EndFused(stackalloc[] { y });
        var got = p.Run(new[] { y }, new[] { (long)want.Length }, 1, out int served);
        Assert.Equal(1, served);
        // A prompt chunk's rows (16 and more) run their products on tensor cores as 3xTF32 (near F32).
        Check($"controller rows={rows}", got[0][0], want, rows >= 16 ? 5e-5 : 1e-5);
    }

    /// <summary>
    /// The attention over a linear window and over a ring, for decode rows (split over the keys) and prompt chunks (tensor
    /// cores; 200 rows take several blocks, tiles every row sees and tiles past the bias extent), with K/V contiguous or as
    /// a window of a larger cache (the strided views a decode step reads). Each score's bias is the row's table T = B r by
    /// distance minus the row's sink logit t (the region's sixth input); a row sees the keys at positions it reaches within
    /// the band (a ring's sliding window) and a zero sink closes its softmax.
    /// </summary>
    [CudaTheory(GgmlBackend = BackendType.GgmlCuda)]
    [InlineData(false, 1, 40, false)]
    [InlineData(false, 1, 300, true)]
    [InlineData(false, 20, 300, false)]
    [InlineData(false, 20, 300, true)]
    [InlineData(false, 80, 300, false)]
    [InlineData(false, 200, 1000, false)]
    [InlineData(false, 80, 9000, false)]
    [InlineData(true, 1, 300, false)]
    [InlineData(true, 3, 300, false)]
    [InlineData(true, 8, 1500, false)]
    [InlineData(false, 8, 2000, false)]
    [InlineData(false, 16, 1000, true)]
    [InlineData(false, 16, 2048, false)]
    [InlineData(true, 12, 1500, false)]
    [InlineData(true, 20, 300, false)]
    public void AttentionMatchesItsOracle(bool ring, int rows, int endPosition, bool strided)
    {
        var rng = new Random(5);
        int d = 128, heads = 24, kvh = 8, rel = 16, extent = 64, window = 100;
        int cells = ring ? 128 : endPosition;   // a ring of 128 cells holds a window of 100 and a step's rows
        int capacity = strided ? cells + 64 : cells;
        int rowStart = endPosition - rows, last = endPosition - 1, band = ring ? window : 0;
        float scale = 1f / MathF.Sqrt(d);
        float[] q = Rand(rng, (long)d * rows * heads), r = Rand(rng, (long)rel * heads * rows, 0.5f), t = Rand(rng, (long)heads * rows, 0.5f);
        float[] b = Rand(rng, (long)rel * (extent + 1), 0.5f);
        for (int jj = 0; jj < rel; jj++) b[extent * rel + jj] = 0;
        // T [extent + 1, heads, rows]: B r by distance (the family's MulMat); the kernel takes t away.
        var table = new float[(long)(extent + 1) * heads * rows];
        for (int i = 0; i < rows; i++)
            for (int h = 0; h < heads; h++)
                for (int dist = 0; dist <= extent; dist++)
                {
                    double acc = 0;
                    for (int jj = 0; jj < rel; jj++) acc += r[((long)i * heads + h) * rel + jj] * (double)b[dist * rel + jj];
                    table[((long)i * heads + h) * (extent + 1) + dist] = (float)acc;
                }
        Half[] k = Rand(rng, (long)d * capacity * kvh).Select(v => (Half)v).ToArray(), v = Rand(rng, (long)d * capacity * kvh).Select(x => (Half)x).ToArray();
        // Cell j's position: a ring's latest write of j; a linear window's j.
        long KeyPos(int j) => ring ? (last >= j ? j + (last - j) / cells * cells : -1) : j;
        var want = new double[(long)d * heads * rows];
        for (int i = 0; i < rows; i++)
            for (int h = 0; h < heads; h++)
            {
                int g = h / (heads / kvh);
                long p = rowStart + i;
                var sc = new List<(double S, int J)>();
                for (int j = 0; j < cells; j++)
                {
                    long kp = KeyPos(j);
                    if (kp < 0 || kp > p || (band > 0 && p - kp >= band)) continue;
                    double dot = 0;
                    for (int e = 0; e < d; e++) dot += q[((long)h * rows + i) * d + e] * (double)k[((long)g * capacity + j) * d + e];
                    long dist = p - kp;
                    int bi = dist < extent ? (int)dist : extent;
                    sc.Add((dot * scale + table[((long)i * heads + h) * (extent + 1) + bi] - t[(long)i * heads + h], j));
                }
                double m = Math.Max(0, sc.Count == 0 ? 0 : sc.Max(x => x.S));
                double sum = Math.Exp(-m) + sc.Sum(x => Math.Exp(x.S - m));
                for (int e = 0; e < d; e++)
                    want[((long)i * heads + h) * d + e] = sc.Sum(x => Math.Exp(x.S - m) * (double)v[((long)g * capacity + x.J) * d + e]) / sum;
            }
        using var prog = new Program();
        // q as the runtime hands it: [d, rows, heads] (here contiguous in that order).
        int qi = prog.Input(q, d, rows, heads);
        int ki = prog.InputF16(k, d, capacity, kvh), vi = prog.InputF16(v, d, capacity, kvh);
        if (strided)
        {
            // The first cells of each KV head of a larger cache, at the cache's strides (no copy).
            ki = prog.View(ki, d, cells, kvh, d * 2L, (long)capacity * d * 2, 0);
            vi = prog.View(vi, d, cells, kvh, d * 2L, (long)capacity * d * 2, 0);
        }
        int ti = prog.Input(table, extent + 1, heads, rows);
        int pi = prog.Input(Enumerable.Range(rowStart, rows).ToArray(), rows);
        int si = prog.Input(t, heads, rows);
        prog.G.BeginFused(GraphFusion.InklingAttention, stackalloc[] { qi, ki, vi, ti, pi, si },
            stackalloc long[] { ring ? cells : 0, extent, band, 0, 0, 0 }, stackalloc[] { scale });
        int y = prog.G.Op(GraphOp.Scale, stackalloc[] { prog.G.Op(GraphOp.Cont, prog.G.Op(GraphOp.Permute, stackalloc[] { qi },
            stackalloc long[] { 0, 2, 1, 3 })) }, default, stackalloc[] { 0f });
        prog.G.EndFused(stackalloc[] { y });
        var got = prog.Run(new[] { y }, new[] { (long)want.Length }, 1, out int served);
        Assert.Equal(1, served);
        // Every window of more than one row (verify windows and prompt chunks) rounds Q and the probabilities to F16 on
        // tensor cores; one row runs in F32.
        Check($"attention ring={ring} rows={rows} end={endPosition} strided={strided}", got[0][0], want, rows > 1 ? 1e-3 : 2e-4);
    }

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void RouterMatchesItsOracle()
    {
        var rng = new Random(6);
        int experts = 128, top = 8, rows = 3;
        const float scale = 8f;
        float[] logits = Rand(rng, (long)experts * rows, 3f), bias = Rand(rng, experts, 0.2f);
        using var p = new Program();
        int li = p.Input(logits, experts, rows), bi = p.Weight(bias, experts);
        p.G.BeginFused(GraphFusion.InklingRoute, stackalloc[] { li, bi }, stackalloc long[] { top }, stackalloc[] { scale });
        int ids = p.G.Op(GraphOp.TopK, stackalloc[] { li }, stackalloc long[] { top });
        int weights = p.G.Op(GraphOp.Reshape, stackalloc[] { p.G.Op(GraphOp.Scale, stackalloc[] { p.G.Op(GraphOp.Cont,
            p.View(li, top, rows, 1, experts * 4L, 0, 0)) }, default, stackalloc[] { 0f }) }, stackalloc long[] { 1, top, rows });
        p.G.EndFused(stackalloc[] { ids, weights });
        var got = p.Run(new[] { ids, weights }, new[] { (long)top * rows, (long)top * rows }, 1, out int served);
        Assert.Equal(1, served);
        // A GGML graph's I32 outputs come back as their F32 values (f5b0ac2: ids read as raw bits had been denormals).
        int[] gotIds = got[0][0].Select(f => (int)f).ToArray();
        var want = new double[(long)top * rows];
        for (int row = 0; row < rows; row++)
        {
            var probs = Enumerable.Range(0, experts).Select(e => Sigmoid(logits[row * experts + e])).ToArray();
            int[] order = Enumerable.Range(0, experts).OrderByDescending(e => probs[e] + bias[e]).ThenBy(e => e).Take(top).ToArray();
            Assert.Equal(order, gotIds.Skip(row * top).Take(top).ToArray());
            double sum = order.Sum(e => probs[e]) + 1e-20;
            for (int j = 0; j < top; j++) want[row * top + j] = probs[order[j]] / sum * scale;
        }
        Check("router weights", got[0][1], want, 1e-6);
    }

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void SituMatchesItsOracle()
    {
        var rng = new Random(7);
        int ff = 1152, top = 8, rows = 2;
        const float beta = 4f, lb = 25f;
        float[] gu = Rand(rng, 2L * ff * top * rows, 30f);
        var want = new double[(long)ff * top * rows];
        for (long vec = 0; vec < top * rows; vec++)
            for (int e = 0; e < ff; e++)
            {
                double g = gu[vec * 2 * ff + e], u = gu[vec * 2 * ff + ff + e];
                want[vec * ff + e] = beta * Math.Tanh(g / beta) * Sigmoid(g) * lb * Math.Tanh(u / lb);
            }
        using var p = new Program();
        int gi = p.Input(gu, 2 * ff, top, rows);
        p.G.BeginFused(GraphFusion.InklingSitu, stackalloc[] { gi }, default, stackalloc[] { beta, lb });
        int y = p.G.Op(GraphOp.Scale, stackalloc[] { p.G.Op(GraphOp.Cont, p.View(gi, ff, top, rows, 2L * ff * 4, 2L * ff * top * 4, 0)) },
            default, stackalloc[] { 0f });
        p.G.EndFused(stackalloc[] { y });
        var got = p.Run(new[] { y }, new[] { (long)want.Length }, 1, out int served);
        Assert.Equal(1, served);
        Check("situ", got[0][0], want, 3e-5);
    }
}
