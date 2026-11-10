// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// Timing, not a gate: Flash Next's decode-size dense projections as the direct engine runs them, with cold weights
/// (enough copies of each matrix to stream past the L2), at 1-4 rows: microseconds a call and the weight bytes a second,
/// against a device-to-device copy on the same GPU. The decode-once kernels (Dsv4Kernels.DenseQ81Rows) for the layouts
/// they read, the router's BF16 GEMV (Dsv4Kernels.GemvBf16), and the shared q8_1 kernels for the shared expert's Q4_0 and
/// Q5_0 down projections.
/// </summary>
public sealed class Dsv4DenseFastBandwidthTests
{
    private const int Q2_0 = 42, BF16 = 30;
    private readonly ITestOutputHelper _output;

    public Dsv4DenseFastBandwidthTests(ITestOutputHelper output) => _output = output;

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

    [CudaFact]
    [Trait("Category", "Bench")]
    public void FlashNextDenseProjections_ColdWeightBandwidth()
    {
        var shapes = new (string Name, int Type, int In, int Out)[]
        {
            ("qkv IQ4_XS", (int)GgmlTensorType.IQ4_XS, 2560, 10240),
            ("gdn gate Q4_K", (int)GgmlTensorType.Q4_K, 2560, 6144),
            ("ssm_out Q5_K", (int)GgmlTensorType.Q5_K, 6144, 2560),
            ("attn_q Q2_0", Q2_0, 2560, 12288),
            ("attn_o Q4_K", (int)GgmlTensorType.Q4_K, 6144, 2560),
            ("attn_k Q3_K", (int)GgmlTensorType.Q3_K, 2560, 512),
            ("hc down Q8_0", (int)GgmlTensorType.Q8_0, 10240, 320),
            ("hc up Q8_0", (int)GgmlTensorType.Q8_0, 320, 10240),
            ("shexp gate Q2_0", Q2_0, 2560, 640),
            ("shexp gate IQ4_XS", (int)GgmlTensorType.IQ4_XS, 2560, 640),
            ("shexp down Q4_0", (int)GgmlTensorType.Q4_0, 640, 2560),
            ("shexp down Q5_0", (int)GgmlTensorType.Q5_0, 640, 2560),
            ("router BF16", BF16, 2560, 512),
        };
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        IntPtr stream = allocator.Stream.Handle;
        var rnd = new Random(7);

        // The copy-bound reference: a 512 MiB device-to-device copy, counted as bytes read (and as read + write).
        {
            long n = 512L << 20;
            CudaDriverApi.cuMemAlloc(out IntPtr a, new UIntPtr((ulong)n)).ThrowOnError();
            CudaDriverApi.cuMemAlloc(out IntPtr b, new UIntPtr((ulong)n)).ThrowOnError();
            try
            {
                CudaDriverApi.cuMemcpyDtoDAsync(b, a, new UIntPtr((ulong)n), stream).ThrowOnError();
                allocator.Stream.Synchronize();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < 5; i++)
                    CudaDriverApi.cuMemcpyDtoDAsync(b, a, new UIntPtr((ulong)n), stream).ThrowOnError();
                allocator.Stream.Synchronize();
                double s = sw.Elapsed.TotalSeconds / 5;
                _output.WriteLine($"reference: device-to-device copy of 512 MiB {s * 1e3:F2} ms: {n / s / 1e9:F0} GB/s read " +
                    $"({2 * n / s / 1e9:F0} GB/s read + write)");
            }
            finally
            {
                CudaDriverApi.cuMemFree(a);
                CudaDriverApi.cuMemFree(b);
            }
        }

        foreach (var (name, type, inDim, outDim) in shapes)
        {
            bool fast = type != BF16 && dk.DenseFastReads(type, inDim);
            long rowBytes = type == BF16 ? inDim * 2L : type == Q2_0 ? inDim / 64 * 18L : NativeDequant.RowSize(type, inDim);
            long matBytes = rowBytes * outDim;
            int copies = (int)Math.Max(4, Math.Min(256, (768L << 20) / matBytes));
            byte[] w;
            // Block layouts with the F16 scale first: Q2_0 (18 bytes), Q4_0 (18), Q5_0 (22).
            int scaledBlock = type == Q2_0 ? 18 : type == (int)GgmlTensorType.Q4_0 ? 18 : type == (int)GgmlTensorType.Q5_0 ? 22 : 0;
            if (scaledBlock > 0 || type == BF16)
            {
                w = new byte[matBytes];
                rnd.NextBytes(w);
                if (scaledBlock > 0)
                    for (long b = 0; b < w.Length / scaledBlock; b++)
                        BitConverter.TryWriteBytes(w.AsSpan((int)(b * scaledBlock), 2), (System.Half)0.01f);
                else
                    for (long b = 0; b < w.Length / 2; b++)
                        BitConverter.TryWriteBytes(w.AsSpan((int)(b * 2), 2), (ushort)(0x3C00 + rnd.Next(0x100)));
            }
            else
                w = QuantRows(type, outDim, rowBytes, seed: 3);
            var mats = new List<Tensor>();
            try
            {
                for (int c = 0; c < copies; c++)
                    mats.Add(Up(allocator, w));
                var x = new float[4 * inDim];
                for (int i = 0; i < x.Length; i++)
                    x[i] = (float)(rnd.NextDouble() * 2 - 1);
                using Tensor xT = Up(allocator, x);
                using Tensor xq = new Tensor(allocator, DType.UInt8, 4L * (inDim / 32) * CudaKernels.Q81BlockBytes);
                using Tensor y = new Tensor(allocator, DType.Float32, 4, outDim);
                using Tensor gemvScratch = new Tensor(allocator, DType.Float32, 1L << 20);
                allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(xT), Ptr(xq), inDim, 4, stream, warpCooperative: true);
                // The engine's decode-size route (Q4eCudaEngine.MatMul): the decode-once kernels where they read the layout,
                // else the shared q8_1 kernels where they take it, else Dsv4Kernels.DenseQ81Rows's row kernels.
                bool shared = !fast && type != BF16
                    && CudaQuantizedOps.TryResidentMatmulQ81(allocator, Ptr(mats[0]), type, Ptr(xq), Ptr(y), inDim, outDim, 1, stream);
                string path = type == BF16 ? "bf16 gemv" : fast ? "decode-once" : shared ? "shared q8_1" : "row kernels";
                void Call(Tensor m, int rows)
                {
                    if (type == BF16)
                        dk.GemvBf16(Ptr(xT), Ptr(m), Ptr(y), inDim, outDim, rows, stream);
                    else if (shared)
                        CudaQuantizedOps.TryResidentMatmulQ81(allocator, Ptr(m), type, Ptr(xq), Ptr(y), inDim, outDim, rows, stream);
                    else
                        dk.DenseQ81Rows(Ptr(m), Ptr(xq), Ptr(y), type, outDim, inDim, rowBytes, rows, stream);
                }
                string line = $"{name} {inDim} -> {outDim} [{path}] ({matBytes / 1048576.0:F2} MiB, {copies} copies):";
                for (int rows = 1; rows <= 4; rows++)
                {
                    for (int c = 0; c < copies; c++)
                        Call(mats[c], rows);
                    allocator.Stream.Synchronize();
                    const int laps = 3;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int lap = 0; lap < laps; lap++)
                        for (int c = 0; c < copies; c++)
                            Call(mats[c], rows);
                    allocator.Stream.Synchronize();
                    double us = sw.Elapsed.TotalMilliseconds * 1000.0 / (laps * copies);
                    line += $" {rows}r {us:F2} us {matBytes / (us * 1e3):F0} GB/s;";
                }
                _output.WriteLine(line);
            }
            finally
            {
                allocator.Synchronize();
                foreach (Tensor t in mats)
                    t.Dispose();
            }
        }
    }
}
