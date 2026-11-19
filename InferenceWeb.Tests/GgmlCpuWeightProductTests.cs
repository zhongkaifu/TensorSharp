// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The ggml-cpu side of the generic graph (ggml_ops_cpu_graph.cpp, ggml_ops_cpu_products.cpp): weights only products read
/// run from ggml-cpu's repacked layouts (CPU_REPACK), every other reader keeps the plain bytes, and the weights ggml-cpu has
/// no fast product for on ARM64 (BF16 at every width, sgemm at prompt width on Apple; NVFP4 from two rows) run TensorSharp's
/// products.
/// </summary>
public sealed class GgmlCpuWeightProductTests
{
    private const int K = 512, M = 64;
    private readonly ITestOutputHelper _output;

    public GgmlCpuWeightProductTests(ITestOutputHelper output) => _output = output;

    private sealed class Native : IDisposable
    {
        private readonly List<IntPtr> _blocks = new();

        public IntPtr Copy(byte[] bytes)
        {
            IntPtr p = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, p, bytes.Length);
            _blocks.Add(p);
            return p;
        }

        public IntPtr Copy(float[] values)
        {
            IntPtr p = Marshal.AllocHGlobal(values.Length * sizeof(float));
            Marshal.Copy(values, 0, p, values.Length);
            _blocks.Add(p);
            return p;
        }

        public void Dispose()
        {
            foreach (IntPtr p in _blocks)
            {
                // The CPU caches (repacked copies) are keyed by host pointer: drop them before the bytes go.
                GgmlBasicOps.InvalidateHostBuffer(p);
                Marshal.FreeHGlobal(p);
            }
        }
    }

    private static void WriteHalf(byte[] b, int at, float v) => BitConverter.TryWriteBytes(b.AsSpan(at, 2), (System.Half)v);

    /// <summary>Random but well-formed blocks of <paramref name="type"/> for an [K, M] weight.</summary>
    private static byte[] RandomWeight(int type, Random rng)
    {
        int blocks, size;
        switch (type)
        {
            case 2: size = 18; blocks = K / 32; break;    // Q4_0: d, 16 bytes of nibbles
            case 8: size = 34; blocks = K / 32; break;    // Q8_0: d, 32 int8
            case 12: size = 144; blocks = K / 256; break; // Q4_K: d, dmin, 12 scale bytes, 128 bytes of nibbles
            case 14: size = 210; blocks = K / 256; break; // Q6_K: 128 ql, 64 qh, 16 int8 scales, d
            case 20: size = 18; blocks = K / 32; break;   // IQ4_NL: d, 16 bytes of nibbles
            case 39: size = 17; blocks = K / 32; break;   // MXFP4: an E8M0 exponent, 16 bytes of E2M1 codes
            case 30: size = 2; blocks = K; break;          // BF16
            case 40: size = 36; blocks = K / 64; break;    // NVFP4: four UE4M3 sub-block scales, 32 bytes of E2M1 codes
            default: throw new ArgumentOutOfRangeException(nameof(type));
        }
        var b = new byte[M * blocks * size];
        rng.NextBytes(b);
        for (int i = 0; i < M * blocks; i++)
        {
            int at = i * size;
            switch (type)
            {
                case 2:
                case 8: WriteHalf(b, at, (float)(rng.NextDouble() * 0.02 + 0.005)); break;
                case 12: WriteHalf(b, at, (float)(rng.NextDouble() * 0.01 + 0.002)); WriteHalf(b, at + 2, (float)(rng.NextDouble() * 0.005)); break;
                case 20: WriteHalf(b, at, (float)(rng.NextDouble() * 0.02 + 0.005)); break;
                case 39: b[at] = (byte)(118 + rng.Next(8)); break;   // 2^-9 .. 2^-2
                case 14: WriteHalf(b, at + 208, (float)(rng.NextDouble() * 0.005 + 0.001)); break;
                case 30:
                {
                    float v = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
                    uint bits = BitConverter.SingleToUInt32Bits(v);
                    b[at] = (byte)(bits >> 16);
                    b[at + 1] = (byte)(bits >> 24);
                    break;
                }
                case 40:
                    for (int s = 0; s < 4; s++) b[at + s] = (byte)(0x30 + rng.Next(16));   // finite UE4M3 scales
                    break;
            }
        }
        return b;
    }

    private static float[] Dequantize(int type, byte[] w)
    {
        var f = new float[K * M];
        ManagedQuantizedOps.DequantizeToFloat32(type, w, 0, f, 0, f.Length);
        return f;
    }

    /// <summary>y [M', n] = w rows [r0, r0 + M') · x [K, n] in double precision.</summary>
    private static float[] Reference(float[] w, float[] x, int n, int r0 = 0, int rows = M)
    {
        var y = new float[rows * n];
        for (int j = 0; j < n; j++)
            for (int m = 0; m < rows; m++)
            {
                double s = 0;
                for (int k = 0; k < K; k++) s += (double)w[(r0 + m) * K + k] * x[j * K + k];
                y[j * rows + m] = (float)s;
            }
        return y;
    }

    private static double RelL2(float[] a, float[] b)
    {
        Assert.Equal(b.Length, a.Length);
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    /// <summary>The columns [r0, r0 + rows) of each of n result rows of width M.</summary>
    private static float[] Rows(float[] y, int n, int r0, int rows)
    {
        var o = new float[rows * n];
        for (int j = 0; j < n; j++) Array.Copy(y, j * M + r0, o, j * rows, rows);
        return o;
    }

    /// <summary>One graph over weight bytes at <paramref name="w"/>: each product (a row block when its count is set) and
    /// optionally the weight's rows gathered by id; its outputs after one run.</summary>
    private static float[][] Run(int type, IntPtr w, long wBytes, IntPtr x, int n, (int First, int Count)[] products, int[] gather,
        Native native)
    {
        var b = new GraphProgramBuilder(record: true);
        int wl = b.Leaf(GraphLeafRole.Weight, type, stackalloc long[] { K, M }, w, wBytes);
        int xl = b.Leaf(GraphLeafRole.Weight, GraphFormat.F32, stackalloc long[] { K, n }, x, (long)K * n * sizeof(float));
        long rowBytes = wBytes / M;
        foreach ((int first, int count) in products)
        {
            int src = count == M ? wl
                : b.Op(GraphOp.View, stackalloc[] { wl }, stackalloc long[] { 2, K, count, 1, 1, rowBytes, 0, 0, first * rowBytes });
            b.Output(b.Op(GraphOp.MulMat, src, xl), (long)count * n);
        }
        if (gather != null)
        {
            var ids = new byte[gather.Length * sizeof(int)];
            Buffer.BlockCopy(gather, 0, ids, 0, ids.Length);
            int il = b.Leaf(GraphLeafRole.Weight, GraphFormat.I32, stackalloc long[] { gather.Length }, native.Copy(ids), ids.Length);
            b.Output(b.Op(GraphOp.GetRows, wl, il), (long)K * gather.Length);
        }
        using IGraphPlan plan = GgmlGraphExecutor.For(GgmlBackendType.Cpu).Compile(b.ToProgram(replayable: false, reorder: false));
        Assert.True(plan.Run(ReadOnlySpan<int>.Empty, ReadOnlySpan<IntPtr>.Empty, ReadOnlySpan<long>.Empty), plan.LastError);
        var outputs = new float[plan.Outputs.Length][];
        for (int i = 0; i < outputs.Length; i++) outputs[i] = (float[])plan.Outputs[i].Clone();
        return outputs;
    }

    private static float[] Activations(int n, Random rng)
    {
        var x = new float[K * n];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    private static void WithCpuBackend(Action body)
    {
        _ = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0);
        try { body(); }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    /// <summary>
    /// A weight read only by products (whole, or 8-row-aligned row blocks) runs from the repacked layout; the same bytes
    /// read by a GET_ROWS, or through a row block that splits an interleaved group, stay plain in that graph. All of them
    /// compute the same products: the repacked and plain kernels differ only in their summation order, and both round
    /// the activations to Q8 as ggml-cpu does (so the double-precision reference sits ~1e-2 away). (Which repacked products
    /// give a row the same bits at every width: see <see cref="RepackedNonKQuantProductsGiveEveryRowItsOwnBits"/>.)
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void RepackedAndPlainLayoutsComputeTheSameProducts()
    {
        WithCpuBackend(() =>
        {
            var rng = new Random(7);
            foreach (int type in new[] { 2, 8, 12, 14 })
                foreach (int n in new[] { 1, 4, 5, 37 })
                {
                    using var native = new Native();
                    byte[] bytes = RandomWeight(type, rng);
                    float[] wf = Dequantize(type, bytes);
                    float[] x = Activations(n, rng);
                    IntPtr w = native.Copy(bytes), xp = native.Copy(x);
                    float[] reference = Reference(wf, x, n);

                    // Repackable: the whole weight and two aligned row blocks.
                    float[][] repacked = Run(type, w, bytes.Length, xp, n, new[] { (0, M), (8, 16), (32, 32) }, null, native);
                    // The same bytes gathered by row: plain in that graph (the copy stays cached for the next).
                    int[] ids = { 0, 9, 63 };
                    float[][] plain = Run(type, w, bytes.Length, xp, n, new[] { (0, M) }, ids, native);
                    // A block starting mid-group: plain.
                    float[][] split = Run(type, w, bytes.Length, xp, n, new[] { (4, 16) }, null, native);

                    double toReference = RelL2(repacked[0], reference);
                    double layouts = RelL2(repacked[0], plain[0]);
                    _output.WriteLine($"type {type} rows {n}: repacked vs reference {toReference:E2}, vs plain {layouts:E2}");
                    Assert.True(toReference < 3e-2, $"type {type} rows {n}: {toReference:E2} from the reference");
                    Assert.True(layouts < 1e-5, $"type {type} rows {n}: repacked vs plain layout {layouts:E2}");
                    Assert.True(RelL2(repacked[1], Rows(plain[0], n, 8, 16)) < 1e-5, $"type {type} rows {n}: row block 8..24");
                    Assert.True(RelL2(repacked[2], Rows(plain[0], n, 32, 32)) < 1e-5, $"type {type} rows {n}: row block 32..64");
                    Assert.True(RelL2(split[0], Rows(plain[0], n, 4, 16)) < 1e-5, $"type {type} rows {n}: row block 4..20");
                    // The gathered rows are the plain bytes, dequantized (interleaved bytes would decode to other values).
                    for (int i = 0; i < ids.Length; i++)
                        for (int k = 0; k < K; k++)
                        {
                            float want = wf[ids[i] * K + k], got = plain[1][i * K + k];
                            Assert.True(Math.Abs(got - want) <= 1e-6f * Math.Max(1f, Math.Abs(want)),
                                $"type {type}: gathered row {ids[i]} value {k}: {got} vs {want}");
                        }
                }
        });
    }

    /// <summary>
    /// BF16 weights at every width (NEON rows below 64, Accelerate's sgemm from there) run TensorSharp's products over
    /// unrounded F32 activations: within F32 summation error of the double-precision reference. NVFP4 products keep
    /// ggml-cpu's own arithmetic at every width (activations rounded to Q8_0; see
    /// <see cref="Nvfp4ProductsOfSeveralRowsHaveEachRowsOwnBits"/>).
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void FloatWeightProductsReadUnroundedActivations()
    {
        WithCpuBackend(() =>
        {
            var rng = new Random(11);
            bool arm64 = RuntimeInformation.ProcessArchitecture == Architecture.Arm64;
            foreach ((int type, int[] widths) in new[] { (30, new[] { 1, 3, 4, 7, 32, 63, 64, 65 }), (40, new[] { 1, 31, 255, 256, 300 }) })
                foreach (int n in widths)
                {
                    using var native = new Native();
                    byte[] bytes = RandomWeight(type, rng);
                    float[] wf = Dequantize(type, bytes);
                    float[] x = Activations(n, rng);
                    float[][] y = Run(type, native.Copy(bytes), bytes.Length, native.Copy(x), n, new[] { (0, M), (16, 32) }, null, native);
                    double rel = RelL2(y[0], Reference(wf, x, n));
                    double block = RelL2(y[1], Reference(wf, x, n, 16, 32));
                    _output.WriteLine($"type {type} rows {n}: {rel:E2} (row block {block:E2})");
                    bool ours = arm64 && type == 30;
                    double tolerance = ours ? 1e-6 : 3e-2;
                    Assert.True(rel < tolerance, $"type {type} rows {n}: {rel:E2} from the reference (tolerance {tolerance:E0})");
                    Assert.True(block < tolerance, $"type {type} rows {n}: row block {block:E2} from the reference");
                }
        });
    }

    private static bool SameBits(float[] a, int aAt, float[] b, int bAt, int count)
    {
        for (int i = 0; i < count; i++)
            if (BitConverter.SingleToInt32Bits(a[aAt + i]) != BitConverter.SingleToInt32Bits(b[bAt + i])) return false;
        return true;
    }

    /// <summary>
    /// NVFP4 products of two rows and more run TensorSharp's GEMM over ggml's own Q8_0 activation rows with
    /// ggml_vec_dot_nvfp4_q8_0's arithmetic: every row of a wide product has the bits ggml-cpu's product gives that row
    /// alone (a one-row product is ggml's dot), for the whole weight and for a row block.
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void Nvfp4ProductsOfSeveralRowsHaveEachRowsOwnBits()
    {
        WithCpuBackend(() =>
        {
            var rng = new Random(13);
            foreach (int n in new[] { 2, 3, 4, 5, 8, 37, 255, 256, 300, 512 })
            {
                using var native = new Native();
                byte[] bytes = RandomWeight(40, rng);
                IntPtr w = native.Copy(bytes);
                float[] x = Activations(n, rng);
                float[][] wide = Run(40, w, bytes.Length, native.Copy(x), n, new[] { (0, M), (16, 32) }, null, native);
                foreach (int j in n <= 8 ? System.Linq.Enumerable.Range(0, n) : new[] { 0, 1, n / 2, n - 1 })
                {
                    float[][] one = Run(40, w, bytes.Length, native.Copy(x.AsSpan(j * K, K).ToArray()), 1, new[] { (0, M), (16, 32) }, null, native);
                    Assert.True(SameBits(wide[0], j * M, one[0], 0, M), $"rows {n}: row {j} of the whole weight's product");
                    Assert.True(SameBits(wide[1], j * 32, one[1], 0, 32), $"rows {n}: row {j} of the row block's product");
                }
            }
        });
    }

    /// <summary>
    /// ggml-cpu's repacked product runs groups of four activation rows through its GEMM and the rows past the last group
    /// through its GEMV. For Q4_0, Q8_0, IQ4_NL and MXFP4 the two give a row the same bits, so every row of a product has
    /// the bits it has alone whatever rows share the step (a decode token, a batch of sequences' tokens, a prompt's tail):
    /// concurrent greedy replies equal one-at-a-time ones on those files. (The K-quants' GEMM sums a row in another order
    /// than their GEMV, so theirs depend on the batch, as on llama.cpp; padding their rows to the GEMM halved decode speed.)
    /// A drift guard on the ggml kernels these files rely on.
    /// <para>
    /// Weights are repacked only where <see cref="GgmlBasicOps.CpuRepacksWeights"/> says so: by default on ARM64 alone
    /// (TS_GGML_CPU_REPACK). Elsewhere, x86_64 included, they keep their GGUF layout and this test exercises ggml-cpu's
    /// plain kernels instead: a one-row product runs the type's vec_dot, and a product of two rows or more runs llamafile's
    /// sgemm where it has the type (llamafile_sgemm declines below two rows). Q4_0 and Q8_0 still give every row its own
    /// bits there (their AVX2 vec_dot keeps one accumulator, as tinyBLAS does), and MXFP4 runs its vec_dot at every width
    /// (sgemm has no MXFP4 case). IQ4_NL does not: ggml_vec_dot_iq4_nl_q8_0's AVX2 body sums even and odd blocks in two
    /// accumulators while tinyBLAS_Q0_AVX&lt;block_iq4_nl&gt; keeps one, so a plain IQ4_NL row's bits depend on whether it
    /// is alone. IQ4_NL is therefore checked only where weights are repacked (x86_64 with TS_GGML_CPU_REPACK=1 checks it
    /// and passes); no in-process switch turns repacking on, so on x86_64 by default it is skipped.
    /// </para>
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void RepackedNonKQuantProductsGiveEveryRowItsOwnBits()
    {
        WithCpuBackend(() =>
        {
            var rng = new Random(17);
            var differing = new List<string>();
            bool repacked = GgmlBasicOps.CpuRepacksWeights();
            _output.WriteLine(repacked
                ? "weights repacked: IQ4_NL checked"
                : "weights in their GGUF layout (ggml-cpu's plain kernels): IQ4_NL not checked");
            foreach (int type in new[] { 2, 8, 20, 39 })   // Q4_0, Q8_0, IQ4_NL, MXFP4
            {
                using var native = new Native();
                byte[] bytes = RandomWeight(type, rng);
                IntPtr w = native.Copy(bytes);
                const int most = 13;
                float[] x = Activations(most, rng);
                // Skipped after its data is drawn, so the types after it see the same random data either way.
                if (type == 20 && !repacked) continue;
                var alone = new float[most][];
                for (int j = 0; j < most; j++)
                    alone[j] = Run(type, w, bytes.Length, native.Copy(x.AsSpan(j * K, K).ToArray()), 1, new[] { (0, M) }, null, native)[0];
                foreach (int n in new[] { 2, 3, 4, 5, 6, 7, 8, 9, 13 })
                {
                    float[] batch = Run(type, w, bytes.Length, native.Copy(x.AsSpan(0, n * K).ToArray()), n, new[] { (0, M) }, null, native)[0];
                    for (int j = 0; j < n; j++)
                        if (!SameBits(batch, j * M, alone[j], 0, M)) differing.Add($"type {type} row {j} of {n}");
                }
            }
            Assert.True(differing.Count == 0, "rows whose bits differ from the row alone: " + string.Join(", ", differing));
        });
    }

    /// <summary>
    /// A per-op product (outside the graph runtime) runs on the pool for its width, as graph steps do: a wide one on the
    /// prompt pool after a decode step left the decode pool set, a decode-width one on the decode pool; an op without
    /// products keeps the pool it finds. (On a machine whose two pools have the same width the widths agree trivially.)
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void PerOpProductsRunOnThePoolForTheirWidth()
    {
        WithCpuBackend(() =>
        {
            var rng = new Random(19);
            using var native = new Native();
            byte[] bytes = RandomWeight(8, rng);
            IntPtr w = native.Copy(bytes);
            Run(8, w, bytes.Length, native.Copy(Activations(8, rng)), 8, new[] { (0, M) }, null, native);
            int prompt = GgmlBasicOps.CpuPoolThreads();
            Run(8, w, bytes.Length, native.Copy(Activations(1, rng)), 1, new[] { (0, M) }, null, native);
            int decode = GgmlBasicOps.CpuPoolThreads();
            _output.WriteLine($"prompt pool {prompt} threads, decode pool {decode}");
            Assert.True(prompt > 0 && decode > 0, $"pools {prompt} / {decode}");

            var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0);
            using var weight = new Tensor(allocator, DType.Float32, K, M);
            weight.SetElementsAsFloat(Activations(M, rng));
            int PerOp(int rows)
            {
                using var x = new Tensor(allocator, DType.Float32, rows, K);
                using var y = new Tensor(allocator, DType.Float32, rows, M);
                x.SetElementsAsFloat(Activations(rows, rng));
                Ops.Addmm(y, 0f, y, 1f, x, weight);
                return GgmlBasicOps.CpuPoolThreads();
            }
            Assert.Equal(prompt, PerOp(8));
            Assert.Equal(decode, PerOp(1));
            Assert.Equal(decode, PerOp(4));
            Assert.Equal(prompt, PerOp(5));
            Assert.Equal(decode, PerOp(2));
            using (var a = new Tensor(allocator, DType.Float32, 8, K))
            {
                a.SetElementsAsFloat(Activations(8, rng));
                Ops.Add(a, a, a);
            }
            Assert.Equal(decode, GgmlBasicOps.CpuPoolThreads());
        });
    }
}
