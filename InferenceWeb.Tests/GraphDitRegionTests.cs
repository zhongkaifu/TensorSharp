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
using TensorSharp.Cpu;
using TensorSharp.Graph;
using TensorSharp.GGML;
using TensorSharp.Models.Graph;
using Xunit;
using Half = System.Half;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The diffusion-transformer regions (GraphFusion.AdaLnModulate, GatedResidual, TableRopeQkNorm, LoraLinear, SwigluFfn with
/// biases) and the ops Conv2d and PadReflect1d, each against an independent double-precision oracle written here: their
/// reference ops (GraphDitRegions) on every executor that runs them as ops - the managed CPU, MLX and direct CUDA tensor
/// executors and the GGML executor on ggml_cpu and ggml_metal, where a program marked WideTokenProducts serves nothing and
/// computes the same bytes - and on ggml_cuda the served form (the DiT offload's fused nodes, in a WideTokenProducts program)
/// against both the oracle and the same program's reference ops.
/// </summary>
public sealed class GraphDitRegionTests(ITestOutputHelper output)
{
    // ================================================================ environments

    /// <summary>An executor and the leaves it binds: tensor executors bind a weight by its tensor, GGML by its host bytes.</summary>
    private sealed class Env : IDisposable
    {
        private readonly List<GCHandle> _pins = new();
        private readonly List<IntPtr> _weights = new();
        private readonly List<Tensor> _tensors = new();

        public Env(IGraphExecutor executor, IAllocator allocator, double productTolerance = 0)
        {
            Executor = executor;
            Allocator = allocator;
            ProductTolerance = productTolerance;
        }

        /// <summary>The executor's own rounding of an F32 product (MLX's GEMMs on this GPU: about 1e-3 of the largest value), the
        /// least a case with a dense product is held to.</summary>
        public double ProductTolerance { get; }

        public IGraphExecutor Executor { get; }
        public IAllocator Allocator { get; }
        public bool Ggml => Allocator == null;

        private static long[] Sizes(long[] ne) => ne.Reverse().ToArray();

        /// <summary>Pins a GGML weight's host bytes, which key its device copy.</summary>
        private IntPtr Pin(Array values)
        {
            GCHandle h = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(h);
            _weights.Add(h.AddrOfPinnedObject());
            return h.AddrOfPinnedObject();
        }

        public int Weight(GraphProgramBuilder g, float[] values, params long[] ne)
        {
            if (Ggml) return g.Leaf(GraphLeafRole.Weight, GraphFormat.F32, ne, Pin(values), values.Length * 4L);
            var t = new Tensor(Allocator, DType.Float32, Sizes(ne));
            t.SetElementsAsFloat(values);
            _tensors.Add(t);
            return g.Leaf(GraphLeafRole.Weight, GraphFormat.F32, ne, GraphIdentity.Of(t.Storage), values.Length * 4L, handle: t);
        }

        /// <summary>A GGML weight in a block format (F16, BF16, Q4_K) from its raw bytes.</summary>
        public int RawWeight(GraphProgramBuilder g, int format, Array bytes, long byteCount, params long[] ne)
        {
            if (!Ggml) throw new NotSupportedException("raw weights bind on the GGML executor only");
            return g.Leaf(GraphLeafRole.Weight, format, ne, Pin(bytes), byteCount);
        }

        /// <summary>An F16 weight (the values must be representable in half precision).</summary>
        public int WeightF16(GraphProgramBuilder g, float[] values, params long[] ne)
        {
            if (Ggml)
            {
                ushort[] half = values.Select(v => BitConverter.HalfToUInt16Bits((Half)v)).ToArray();
                return RawWeight(g, GraphFormat.F16, half, half.Length * 2L, ne);
            }
            // The storages narrow floats differently: CPU's in its copy, MLX's when set from them, CUDA's an element at a time.
            var t = new Tensor(Allocator, DType.Float16, Sizes(ne));
            try
            {
                t.SetElementsAsFloat(values);
            }
            catch (NotSupportedException)
            {
                try
                {
                    using var f = new Tensor(Allocator, DType.Float32, Sizes(ne));
                    f.SetElementsAsFloat(values);
                    Ops.Copy(t, f);
                }
                catch (InvalidOperationException)
                {
                    using Tensor flat = t.View(values.Length);
                    for (long i = 0; i < values.Length; i++) flat.SetElementAsFloat(values[i], i);
                }
            }
            _tensors.Add(t);
            return g.Leaf(GraphLeafRole.Weight, GraphFormat.F16, ne, GraphIdentity.Of(t.Storage), values.Length * 2L, handle: t);
        }

        /// <summary>A GGML weight's device copy is keyed by the address of its host bytes and outlives the plan, so it is
        /// dropped before the pin is freed (<see cref="IGraphExecutor.Invalidate"/>): a later weight of the same size the GC
        /// places at that address would otherwise bind this one's copy.</summary>
        public void Dispose()
        {
            foreach (Tensor t in _tensors) t.Dispose();
            foreach (IntPtr data in _weights) Executor.Invalidate(data);
            foreach (GCHandle h in _pins) h.Free();
        }
    }

    /// <summary>A program's per-run inputs, pinned for the run.</summary>
    private sealed class Inputs : IDisposable
    {
        private readonly List<GCHandle> _pins = new();
        public readonly List<int> Nodes = new();
        public readonly List<IntPtr> Data = new();
        public readonly List<long> Bytes = new();

        public int F32(GraphProgramBuilder g, float[] values, params long[] ne) => Add(g, GraphFormat.F32, values, values.Length * 4L, ne);

        public int I32(GraphProgramBuilder g, int[] values, params long[] ne) => Add(g, GraphFormat.I32, values, values.Length * 4L, ne);

        private int Add(GraphProgramBuilder g, int format, Array values, long bytes, long[] ne)
        {
            int node = g.Leaf(GraphLeafRole.Input, format, ne, IntPtr.Zero, bytes);
            GCHandle h = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(h);
            Nodes.Add(node);
            Data.Add(h.AddrOfPinnedObject());
            Bytes.Add(bytes);
            return node;
        }

        public void Dispose()
        {
            foreach (GCHandle h in _pins) h.Free();
        }
    }

    /// <summary>One case: emits its program (on an environment, into a builder, with its inputs), returning the output node and
    /// the oracle's values; <see cref="Tolerance"/> bounds max |got - want| / max |want|.</summary>
    private sealed record Case(string Name, Func<Env, GraphProgramBuilder, Inputs, (int Output, double[] Want)> Emit, double Tolerance,
        bool GgmlOnly = false)
    {
        /// <summary>The case's result goes through a dense F32 product (a projection, a convolution's im2col product).</summary>
        public bool Products => Name.StartsWith("Lora", StringComparison.Ordinal) || Name.StartsWith("Swiglu", StringComparison.Ordinal)
            || Name.StartsWith("Conv2d", StringComparison.Ordinal);
    }

    private static float[] Run(Env env, Case c, bool wide, out int served, out double[] want)
    {
        var g = new GraphProgramBuilder(record: true);
        if (wide) g.MarkWideTokenProducts();
        using var inputs = new Inputs();
        (int output, double[] expected) = c.Emit(env, g, inputs);
        want = expected;
        g.Output(output, expected.Length);
        using IGraphPlan plan = env.Executor.Compile(g.ToProgram(replayable: true, reorder: false));
        served = env.Ggml ? GgmlGraphExecutor.ServedRegions(plan) : 0;
        Assert.True(plan.Run(inputs.Nodes.ToArray(), inputs.Data.ToArray(), inputs.Bytes.ToArray()), plan.LastError);
        return (float[])plan.Outputs[0].Clone();
    }

    private void Check(string what, float[] got, double[] want, double tolerance)
    {
        Assert.Equal(want.Length, got.Length);
        double scale = want.Max(Math.Abs), worst = 0;
        int at = 0;
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]), $"{what}: value {i} is {got[i]}");
            double e = Math.Abs(got[i] - want[i]);
            if (e > worst) { worst = e; at = i; }
        }
        double rel = worst / Math.Max(scale, 1e-30);
        output.WriteLine($"{what}: max |err| {worst:E3} at {at} (want {want[at]:G6}, got {got[at]:G6}), relative {rel:E3}, bound {tolerance:E1}");
        Assert.True(rel <= tolerance, $"{what}: relative error {rel:E3} exceeds {tolerance:E1} (value {at}: want {want[at]}, got {got[at]})");
    }

    private void VerifyReference(Env env, string name)
    {
        Case c = Cases().Single(k => k.Name == name);
        if (c.GgmlOnly && !env.Ggml) return;
        float[] got = Run(env, c, wide: false, out int served, out double[] want);
        Assert.Equal(0, served);
        Check(name, got, want, c.Products ? Math.Max(c.Tolerance, env.ProductTolerance) : c.Tolerance);
        if (!env.Ggml) return;
        // Off ggml_cuda a wide program serves nothing and computes the same bytes.
        float[] wide = Run(env, c, wide: true, out int wideServed, out _);
        Assert.Equal(0, wideServed);
        Assert.Equal(got, wide);
    }

    public static IEnumerable<object[]> CaseNames() => Cases().Select(c => new object[] { c.Name });

    public static IEnumerable<object[]> TensorCaseNames() => Cases().Where(c => !c.GgmlOnly).Select(c => new object[] { c.Name });

    [Theory]
    [MemberData(nameof(TensorCaseNames))]
    public void ManagedCpuReferenceOpsMatchTheOracle(string name)
    {
        using var env = new Env(new ManagedGraphExecutor(new CpuAllocator(BlasEnum.DotNet)), new CpuAllocator(BlasEnum.DotNet));
        VerifyReference(env, name);
    }

    [MlxTheory]
    [MemberData(nameof(TensorCaseNames))]
    public void MlxReferenceOpsMatchTheOracle(string name)
    {
        var allocator = new TensorSharp.MLX.MlxAllocator();
        using var env = new Env(new TensorSharp.MLX.MlxGraphExecutor(allocator), allocator, productTolerance: 3e-3);
        VerifyReference(env, name);
    }

    [CudaTheory]
    [MemberData(nameof(TensorCaseNames))]
    public void DirectCudaReferenceOpsMatchTheOracle(string name)
    {
        var allocator = new TensorSharp.Cuda.CudaAllocator(0);
        using var env = new Env(new TensorSharp.Cuda.CudaGraphExecutor(allocator), allocator);
        VerifyReference(env, name);
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(CaseNames))]
    public void GgmlCpuReferenceOpsMatchTheOracle(string name) => VerifyGgml(GgmlBackendType.Cpu, name);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(CaseNames))]
    public void GgmlMetalReferenceOpsMatchTheOracle(string name) => VerifyGgml(GgmlBackendType.Metal, name);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(CaseNames))]
    public void GgmlCudaReferenceOpsMatchTheOracle(string name)
    {
        // The reference ops alone (no WideTokenProducts: nothing served) on ggml_cuda.
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        using var env = new Env(GgmlGraphExecutor.For(GgmlBackendType.Cuda), null, productTolerance: GgmlCudaF32ProductTolerance);
        Case c = Cases().Single(k => k.Name == name);
        float[] got = Run(env, c, wide: false, out int served, out double[] want);
        Assert.Equal(0, served);
        Check(name, got, want, c.Products ? Math.Max(c.Tolerance, env.ProductTolerance) : c.Tolerance);
    }

    /// <summary>ggml-cuda's F32 products on GPUs with TF32 tensor cores (compute capability 8.0 and newer) take TF32 operands:
    /// it creates every cuBLAS handle in CUBLAS_TF32_TENSOR_OP_MATH (ggml-cuda/common.cuh) and its mmf kernel multiplies F32
    /// tiles with tf32 mma.sync (ggml-cuda/mma.cuh). A 10-bit mantissa rounds each operand by up to 2^-11 = 4.9e-4 of itself,
    /// so the oracle's F32-class bound cannot hold there: on an RTX PRO 4500 (sm_120) the LoRA and biased SwiGLU products sit
    /// 2.9-4.0e-4 from it, and 0.8-1.4e-7 with NVIDIA_TF32_OVERRIDE=0 (cuBLAS's math kept in F32; these shapes take cuBLAS).
    /// 2e-3 is the bound the VAE encoder's oracle holds ggml-cuda's F32 products to (MiniMaxH3ImageEditTests), four TF32
    /// units.</summary>
    private const double GgmlCudaF32ProductTolerance = 2e-3;

    private void VerifyGgml(GgmlBackendType backend, string name)
    {
        GgmlBasicOps.EnsureBackendAvailable(backend);
        using var env = new Env(GgmlGraphExecutor.For(backend), null);
        VerifyReference(env, name);
    }

    // ================================================================ oracles

    private static float[] Random(Random rng, long n, float scale = 1f)
        => Enumerable.Range(0, checked((int)n)).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();

    private static float[] HalfValues(float[] v) => v.Select(x => (float)(Half)x).ToArray();

    private static float Bf16(float v)
    {
        uint bits = BitConverter.SingleToUInt32Bits(v);
        bits += 0x7FFFu + ((bits >> 16) & 1u);   // round to nearest even
        return BitConverter.UInt32BitsToSingle(bits & 0xFFFF0000u);
    }

    private static ushort Bf16Bits(float v) => (ushort)(BitConverter.SingleToUInt32Bits(Bf16(v)) >> 16);

    /// <summary>y [outRows, rows] = w [outRows, inRows] (row o's inRows values at o * inRows) times x [rows, inRows].</summary>
    private static double[] Product(float[] w, double[] x, int inRows, int outRows, int rows)
    {
        var y = new double[(long)outRows * rows];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < outRows; o++)
            {
                double s = 0;
                for (int i = 0; i < inRows; i++) s += w[(long)o * inRows + i] * x[(long)r * inRows + i];
                y[(long)r * outRows + o] = s;
            }
        return y;
    }

    private static double[] D(float[] v) => v.Select(x => (double)x).ToArray();

    private static readonly GraphSegment[] Segments = { new(0, 3, 0), new(3, 7, 6), new(7, 10, 0) };

    private static int SegmentOf(GraphSegment[] segments, int row) => segments.Single(s => row >= s.Start && row < s.End).Column;

    private static double[] RmsNormed(float[] x, int width, int rows, float eps, float[] weight)
    {
        var y = new double[(long)width * rows];
        for (int r = 0; r < rows; r++)
        {
            double ss = 0;
            for (int c = 0; c < width; c++) ss += (double)x[(long)r * width + c] * x[(long)r * width + c];
            double inv = 1.0 / Math.Sqrt(ss / width + eps);
            for (int c = 0; c < width; c++) y[(long)r * width + c] = x[(long)r * width + c] * inv * (weight != null ? weight[c] : 1f);
        }
        return y;
    }

    // ================================================================ cases

    private static IEnumerable<Case> Cases()
    {
        yield return new Case("AdaLnModulate", (env, g, inputs) =>
        {
            var rng = new Random(1801);
            const int hidden = 64, rows = 10, columns = 12, shift = 0, scale = 1;
            const float eps = 1e-6f;
            float[] x = Random(rng, hidden * rows, 3f), w = Random(rng, hidden), mods = Random(rng, hidden * columns, 0.5f);
            int xn = inputs.F32(g, x, hidden, rows), wn = env.Weight(g, w, hidden), mn = inputs.F32(g, mods, hidden, columns);
            int table = inputs.I32(g, GraphDitRegions.SegmentTable(Segments), 3, Segments.Length);
            GraphDitRegions.CheckSegments(Segments, rows, columns, Math.Max(shift, scale));
            int y = GraphDitRegions.AdaLnModulate(g, xn, wn, mn, table, Segments, hidden, shift, scale, eps);
            double[] n = RmsNormed(x, hidden, rows, eps, w), want = new double[n.Length];
            for (int r = 0; r < rows; r++)
            {
                int col = SegmentOf(Segments, r);
                for (int c = 0; c < hidden; c++)
                {
                    double v = n[r * hidden + c];
                    want[r * hidden + c] = v + v * mods[(col + scale) * hidden + c] + mods[(col + shift) * hidden + c];
                }
            }
            return (y, want);
        }, 2e-5);

        yield return new Case("GatedResidual", (env, g, inputs) =>
        {
            var rng = new Random(1901);
            const int hidden = 64, rows = 10, columns = 12, gate = 2;
            float[] b = Random(rng, hidden * rows), mm = Random(rng, hidden * rows, 4f), bias = Random(rng, hidden),
                delta = Random(rng, hidden * rows, 0.25f), mods = Random(rng, hidden * columns);
            int bn = inputs.F32(g, b, hidden, rows), mmn = inputs.F32(g, mm, hidden, rows), biasn = env.Weight(g, bias, hidden);
            int dn = inputs.F32(g, delta, hidden, rows), mn = inputs.F32(g, mods, hidden, columns);
            int table = inputs.I32(g, GraphDitRegions.SegmentTable(Segments), 3, Segments.Length);
            int y = GraphDitRegions.GatedResidual(g, bn, mmn, biasn, dn, mn, table, Segments, hidden, gate);
            var want = new double[hidden * rows];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < hidden; c++)
                {
                    int i = r * hidden + c;
                    want[i] = b[i] + ((double)delta[i] + ((double)mm[i] + bias[c])) * mods[(SegmentOf(Segments, r) + gate) * hidden + c];
                }
            return (y, want);
        }, 2e-6);

        // q, k and v views of one stacked [q | k | v] product, as a DiT's qkv projection lays them out.
        (int Q, int K, int V, float[] Qkv) Qkv(Inputs inputs, GraphProgramBuilder g, Random rng, int d, int heads, int rows)
        {
            int inner = d * heads;
            float[] qkv = Random(rng, 3 * inner * rows, 2f);
            int node = inputs.F32(g, qkv, 3 * inner, rows);
            int View(int part) => g.Op(GraphOp.View, stackalloc[] { node },
                stackalloc long[] { 3, d, heads, rows, 1, d * 4L, 3L * inner * 4, 0, part * inner * 4L });
            return (View(0), View(1), View(2), qkv);
        }

        // The rotated q (part 0) or k (part 1) of every row, [rows][heads][d].
        double[] Rotated(float[] qkv, int part, int d, int heads, int rows, int rot, float eps, float[] weight, float[] cos, float[] ssin)
        {
            int inner = d * heads, half = rot / 2;
            var y = new double[(long)inner * rows];
            for (int s = 0; s < rows; s++)
                for (int h = 0; h < heads; h++)
                {
                    long at = (long)s * 3 * inner + part * inner + h * d;
                    double ss = 0;
                    for (int i = 0; i < d; i++) ss += (double)qkv[at + i] * qkv[at + i];
                    double inv = 1.0 / Math.Sqrt(ss / d + eps);
                    var n = new double[d];
                    for (int i = 0; i < d; i++) n[i] = qkv[at + i] * inv * weight[i];
                    for (int i = 0; i < d; i++)
                    {
                        double o = n[i];
                        if (i < rot) o = n[i] * cos[s * rot + i] + n[i < half ? i + half : i - half] * ssin[s * rot + i];
                        y[(long)s * inner + h * d + i] = o;
                    }
                }
            return y;
        }

        // The tables of 3-axis fractional positions, as a DiT's host builds them: cos and the signed sine [rot, rows].
        (float[] Cos, float[] SignedSin) Tables(Random rng, int rot, int rows)
        {
            var cos = new float[rot * rows];
            var ssin = new float[rot * rows];
            int half = rot / 2;
            for (int s = 0; s < rows; s++)
                for (int j = 0; j < half; j++)
                {
                    double a = (s * 0.37 + j * 0.11) * (1 + rng.NextDouble());
                    cos[s * rot + j] = cos[s * rot + j + half] = (float)Math.Cos(a);
                    ssin[s * rot + j] = -(float)Math.Sin(a);
                    ssin[s * rot + j + half] = (float)Math.Sin(a);
                }
            return (cos, ssin);
        }

        yield return new Case("TableRopeQkNorm", (env, g, inputs) =>
        {
            var rng = new Random(2001);
            const int d = 32, heads = 2, rows = 6, rot = 24;
            const float eps = 1e-6f;
            var (q, k, _, qkv) = Qkv(inputs, g, rng, d, heads, rows);
            float[] qw = Random(rng, d), kw = Random(rng, d);
            var (cos, ssin) = Tables(rng, rot, rows);
            int qn = env.Weight(g, qw, d), kn = env.Weight(g, kw, d);
            int cn = inputs.F32(g, cos, rot, rows), sn = inputs.F32(g, ssin, rot, rows);
            var (qr, kr) = GraphDitRegions.TableRopeQkNorm(g, q, k, qn, kn, cn, sn, d, heads, rows, rot, eps);
            // Both results in one output: q then k.
            int y = g.Op(GraphOp.Concat, stackalloc[] { qr, kr }, stackalloc long[] { 2 });
            double[] want = Rotated(qkv, 0, d, heads, rows, rot, eps, qw, cos, ssin)
                .Concat(Rotated(qkv, 1, d, heads, rows, rot, eps, kw, cos, ssin)).ToArray();
            return (y, want);
        }, 2e-5);

        yield return new Case("TableRopeQkNormAttention", (env, g, inputs) =>
        {
            var rng = new Random(2002);
            const int d = 32, heads = 2, rows = 6, rot = 24;
            const float eps = 1e-6f, vScale = 2f;
            float scale = 1f / MathF.Sqrt(d);
            var (q, k, v, qkv) = Qkv(inputs, g, rng, d, heads, rows);
            float[] qw = Random(rng, d), kw = Random(rng, d);
            var (cos, ssin) = Tables(rng, rot, rows);
            int qn = env.Weight(g, qw, d), kn = env.Weight(g, kw, d);
            int cn = inputs.F32(g, cos, rot, rows), sn = inputs.F32(g, ssin, rot, rows);
            int y = GraphDitRegions.TableRopeQkNormAttention(g, q, k, v, qn, kn, cn, sn, d, heads, rows, rot, eps, scale, vScale);
            double[] qr = Rotated(qkv, 0, d, heads, rows, rot, eps, qw, cos, ssin), kr = Rotated(qkv, 1, d, heads, rows, rot, eps, kw, cos, ssin);
            int inner = d * heads;
            var want = new double[inner * rows];
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < rows; i++)
                {
                    var p = new double[rows];
                    for (int j = 0; j < rows; j++)
                    {
                        double s = 0;
                        for (int c = 0; c < d; c++) s += qr[i * inner + h * d + c] * (double)(float)(Half)kr[j * inner + h * d + c];
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
            return (y, want);
        }, 3e-3);

        yield return new Case("LoraLinearSlots", (env, g, inputs) =>
        {
            var rng = new Random(2101);
            const int inRows = 32, outRows = 48, slot = 16, rank = 4, rows = 5;
            float[] x = Random(rng, inRows * rows), w = Random(rng, outRows * inRows, 0.3f);
            float[] d0 = Random(rng, rank * inRows, 0.5f), u0 = Random(rng, slot * rank, 0.5f), d2 = Random(rng, rank * inRows, 0.5f),
                u2 = Random(rng, slot * rank, 0.5f);
            int xn = inputs.F32(g, x, inRows, rows), wn = env.Weight(g, w, inRows, outRows);
            var parts = new[]
            {
                new GraphDitRegions.LoraPart(0, env.Weight(g, d0, inRows, rank), env.Weight(g, u0, rank, slot)),
                new GraphDitRegions.LoraPart(2, env.Weight(g, d2, inRows, rank), env.Weight(g, u2, rank, slot)),
            };
            int y = GraphDitRegions.LoraLinear(g, xn, wn, parts, outRows, slot, rows);
            double[] want = Product(w, D(x), inRows, outRows, rows);
            foreach ((int s, float[] dn, float[] un) in new[] { (0, d0, u0), (2, d2, u2) })
            {
                double[] delta = Product(un, Product(dn, D(x), inRows, rank, rows), rank, slot, rows);
                for (int r = 0; r < rows; r++)
                    for (int o = 0; o < slot; o++) want[r * outRows + s * slot + o] += delta[r * slot + o];
            }
            return (y, want);
        }, 2e-5);

        yield return new Case("LoraLinearOneSlot", (env, g, inputs) =>
        {
            var rng = new Random(2102);
            const int inRows = 32, outRows = 24, rank = 8, rows = 5;
            float[] x = Random(rng, inRows * rows), w = Random(rng, outRows * inRows, 0.3f);
            float[] dn = Random(rng, rank * inRows, 0.5f), un = Random(rng, outRows * rank, 0.5f);
            int xn = inputs.F32(g, x, inRows, rows), wn = env.Weight(g, w, inRows, outRows);
            var parts = new[] { new GraphDitRegions.LoraPart(0, env.Weight(g, dn, inRows, rank), env.Weight(g, un, rank, outRows)) };
            int y = GraphDitRegions.LoraLinear(g, xn, wn, parts, outRows, 0, rows);
            double[] want = Product(w, D(x), inRows, outRows, rows);
            double[] delta = Product(un, Product(dn, D(x), inRows, rank, rows), rank, outRows, rows);
            for (int i = 0; i < want.Length; i++) want[i] += delta[i];
            return (y, want);
        }, 2e-5);

        yield return new Case("SwigluFfnBiased", (env, g, inputs) =>
        {
            var rng = new Random(2201);
            const int n = 32, ff = 24, rows = 5;
            float[] x = Random(rng, n * rows), gu = Random(rng, 2 * ff * n, 0.3f), down = Random(rng, n * ff, 0.3f),
                b1 = Random(rng, 2 * ff), b2 = Random(rng, n);
            int xn = inputs.F32(g, x, n, rows);
            int y = GraphDitRegions.SwigluFfn(g, xn, env.Weight(g, gu, n, 2 * ff), env.Weight(g, down, ff, n), env.Weight(g, b1, 2 * ff),
                env.Weight(g, b2, n));
            double[] gv = Product(gu, D(x), n, 2 * ff, rows);
            var act = new double[ff * rows];
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < ff; j++)
                {
                    double gate = gv[r * 2 * ff + j] + b1[j], up = gv[r * 2 * ff + ff + j] + b1[ff + j];
                    act[r * ff + j] = gate / (1 + Math.Exp(-gate)) * up;
                }
            double[] want = Product(down, act, ff, n, rows);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < n; c++) want[r * n + c] += b2[c];
            return (y, want);
        }, 2e-5);

        Case Conv(string name, int kw, int kh, int ic, int oc, int w, int h, int batch, int s0, int s1, int p0, int p1, int d0, int d1,
            bool half, double tolerance) => new Case(name, (env, g, inputs) =>
        {
            var rng = new Random(name.Length * 7919 + kw);
            float[] kernel = Random(rng, kw * kh * ic * oc, 0.5f), x = Random(rng, w * h * ic * batch, 2f);
            if (half) kernel = HalfValues(kernel);
            int kn = half ? env.WeightF16(g, kernel, kw, kh, ic, oc) : env.Weight(g, kernel, kw, kh, ic, oc);
            int xn = inputs.F32(g, x, w, h, ic, batch);
            int y = g.Op(GraphOp.Conv2d, stackalloc[] { kn, xn }, stackalloc long[] { s0, s1, p0, p1, d0, d1 });
            float[] xs = half ? HalfValues(x) : x;
            int ow = (w + 2 * p0 - d0 * (kw - 1) - 1) / s0 + 1, oh = (h + 2 * p1 - d1 * (kh - 1) - 1) / s1 + 1;
            var want = new double[ow * oh * oc * batch];
            for (int b = 0; b < batch; b++)
                for (int o = 0; o < oc; o++)
                    for (int yy = 0; yy < oh; yy++)
                        for (int xx = 0; xx < ow; xx++)
                        {
                            double sum = 0;
                            for (int c = 0; c < ic; c++)
                                for (int i = 0; i < kh; i++)
                                    for (int j = 0; j < kw; j++)
                                    {
                                        int iy = yy * s1 - p1 + i * d1, ix = xx * s0 - p0 + j * d0;
                                        if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                                        sum += (double)kernel[((o * ic + c) * kh + i) * kw + j] * xs[((b * ic + c) * h + iy) * w + ix];
                                    }
                            want[((b * oc + o) * oh + yy) * ow + xx] = sum;
                        }
            return (y, want);
        }, tolerance);

        yield return Conv("Conv2dPadded", 3, 3, 3, 4, 7, 6, 2, 1, 1, 1, 1, 1, 1, half: false, 2e-5);
        yield return Conv("Conv2dStridedDilated", 3, 2, 2, 3, 9, 7, 1, 2, 1, 0, 1, 1, 2, half: false, 2e-5);
        yield return Conv("Conv2dF16Kernel", 3, 3, 4, 5, 6, 5, 1, 1, 1, 1, 1, 1, 1, half: true, 1e-2);

        yield return new Case("PadReflect1d", (env, g, inputs) =>
        {
            var rng = new Random(2301);
            const int width = 5, rows = 3, planes = 2, p0 = 2, p1 = 3;
            float[] x = Random(rng, width * rows * planes);
            int y = g.Op(GraphOp.PadReflect1d, stackalloc[] { inputs.F32(g, x, width, rows, planes) }, stackalloc long[] { p0, p1 });
            const int padded = width + p0 + p1;
            var want = new double[padded * rows * planes];
            for (int r = 0; r < rows * planes; r++)
                for (int i = 0; i < padded; i++)
                {
                    int j = i - p0;
                    if (j < 0) j = -j;
                    else if (j >= width) j = 2 * (width - 1) - j;
                    want[r * padded + i] = x[r * width + j];
                }
            return (y, want);
        }, 0);
    }
}
