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
/// no fast product for on ARM64 (BF16; BF16 and NVFP4 at prompt width on Apple) run TensorSharp's products.
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
    /// the activations to Q8 as ggml-cpu does (so the double-precision reference sits ~1e-2 away).
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
    /// BF16 weights at every width (NEON rows below 64, Accelerate's sgemm from there), and NVFP4 weights from 256 rows
    /// where sgemm takes them, run TensorSharp's products over unrounded F32 activations: within F32 summation error of
    /// the double-precision reference. Narrower NVFP4 products stay on ggml-cpu's own kernel (activations rounded to Q8_0).
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
                    bool ours = arm64 && (type == 30 || (OperatingSystem.IsMacOS() && n >= 256));
                    double tolerance = ours ? 1e-6 : 3e-2;
                    Assert.True(rel < tolerance, $"type {type} rows {n}: {rel:E2} from the reference (tolerance {tolerance:E0})");
                    Assert.True(block < tolerance, $"type {type} rows {n}: row block {block:E2} from the reference");
                }
        });
    }
}
