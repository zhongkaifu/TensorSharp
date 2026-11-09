// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The decode-once dense kernels for every window width 1..8 (Dsv4Kernels.DenseQ81Rows): a window of 3, 5, 6 or 7 rows (a
/// verify of 2, 4, 5 or 6 drafts) now runs the kernel of its own width instead of the next power of two's. A row's arithmetic
/// does not depend on the width, so each width must answer bit for bit as the padded dispatch did, and as the row's own
/// one-row decode step. Flash Next's dense projections: IQ4_XS 2560 -> 10240 (qkv), Q4_K 2560 -> 6144 (gate), Q5_K
/// 6144 -> 2560 (ssm_out), Q8_0 10240 -> 320 (the hyper-connection down copies), Q3_K 2560 -> 512 (k), Q2_0 2560 -> 12288 (q).
/// </summary>
public sealed class Dsv4DenseFastWidthTests
{
    private const int Q2_0 = 42;
    private readonly ITestOutputHelper _output;

    public Dsv4DenseFastWidthTests(ITestOutputHelper output) => _output = output;

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

    private static byte[] Rows(int type, int rows, int inDim, int seed)
    {
        if (type != Q2_0)
            return QuantRows(type, rows, NativeDequant.RowSize(type, inDim), seed);
        var rng = new Random(seed);
        var bytes = new byte[(long)rows * (inDim / 64) * 18];
        rng.NextBytes(bytes);
        for (long b = 0; b < bytes.Length / 18; b++)
            BitConverter.TryWriteBytes(bytes.AsSpan((int)(b * 18), 2), (System.Half)(0.008f + 0.02f * (float)rng.NextDouble()));
        return bytes;
    }

    [CudaTheory]
    [InlineData((int)GgmlTensorType.IQ4_XS, 2560, 10240)]
    [InlineData((int)GgmlTensorType.Q4_K, 2560, 6144)]
    [InlineData((int)GgmlTensorType.Q5_K, 6144, 2560)]
    [InlineData((int)GgmlTensorType.Q8_0, 10240, 320)]
    [InlineData((int)GgmlTensorType.Q3_K, 2560, 512)]
    [InlineData(Q2_0, 2560, 12288)]
    public void EveryWindowWidth_AnswersAsThePaddedDispatchAndTheOneRowStep(int type, int inDim, int outDim)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.HasDenseFastEveryWidth, "the loaded module has no decode-once kernel for every width");
        Assert.True(dk.DenseFastReads(type, inDim), $"the decode-once kernels do not read type {type} at {inDim}");
        long rowBytes = type == Q2_0 ? inDim / 64 * 18L : NativeDequant.RowSize(type, inDim);
        byte[] w = Rows(type, outDim, inDim, seed: type * 7 + inDim);
        const int maxRows = 8;
        var rnd = new Random(inDim + outDim);
        var x = new float[maxRows * inDim];
        for (int i = 0; i < x.Length; i++)
            x[i] = (float)((rnd.NextDouble() * 2 - 1) * (i % 97 == 3 ? 8 : 1));
        using Tensor wT = Up(allocator, w);
        using Tensor xT = Up(allocator, x);
        using Tensor xq = new Tensor(allocator, DType.UInt8, (long)maxRows * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using Tensor own = new Tensor(allocator, DType.Float32, maxRows, outDim);
        using Tensor padded = new Tensor(allocator, DType.Float32, maxRows, outDim);
        using Tensor single = new Tensor(allocator, DType.Float32, maxRows, outDim);
        IntPtr stream = allocator.Stream.Handle;
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(xT), Ptr(xq), inDim, maxRows, stream, warpCooperative: true);
        long q81Row = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
        foreach (int rows in new[] { 3, 5, 6, 7 })
        {
            int pow2 = rows <= 4 ? 4 : 8;
            dk.DenseQ81Rows(Ptr(wT), Ptr(xq), Ptr(own), type, outDim, inDim, rowBytes, rows, stream);
            dk.DenseQ81Rows(Ptr(wT), Ptr(xq), Ptr(padded), type, outDim, inDim, rowBytes, rows, stream, fastWidth: pow2);
            for (int r = 0; r < rows; r++)
                dk.DenseQ81Rows(Ptr(wT), (IntPtr)((long)Ptr(xq) + r * q81Row), (IntPtr)((long)Ptr(single) + (long)r * outDim * 4),
                    type, outDim, inDim, rowBytes, 1, stream);
            allocator.Stream.Synchronize();
            float[] a = Down(own, rows * outDim), b = Down(padded, rows * outDim), c = Down(single, rows * outDim);
            Assert.True(a.AsSpan().SequenceEqual(b), $"{rows} rows: the {rows}-row kernel differs from the {pow2}-row dispatch");
            Assert.True(a.AsSpan().SequenceEqual(c), $"{rows} rows: the {rows}-row kernel differs from each row's one-row step");
            Assert.All(a, v => Assert.True(float.IsFinite(v)));
        }
        _output.WriteLine($"type {type} {inDim} -> {outDim}: widths 3, 5, 6, 7 bit for bit with the padded dispatch and the one-row step");
    }
}
