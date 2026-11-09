// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Security.Cryptography;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The decode-once dense kernels' outputs on fixed inputs, hashed per shape and width (1..8 rows), so two builds can be
/// compared bit for bit: a change that keeps every output's arithmetic must print the same fingerprints. Flash Next's
/// dense shapes in each layout the kernels read.
/// </summary>
public sealed class Dsv4DenseFastFingerprintTests
{
    private const int Q2_0 = 42;
    private readonly ITestOutputHelper _output;

    public Dsv4DenseFastFingerprintTests(ITestOutputHelper output) => _output = output;

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

    [CudaTheory]
    [InlineData((int)GgmlTensorType.IQ4_XS, 2560, 10240)]
    [InlineData((int)GgmlTensorType.Q4_K, 2560, 6144)]
    [InlineData((int)GgmlTensorType.Q5_K, 6144, 2560)]
    [InlineData((int)GgmlTensorType.Q8_0, 10240, 320)]
    [InlineData((int)GgmlTensorType.Q8_0, 320, 10240)]
    [InlineData((int)GgmlTensorType.Q3_K, 2560, 512)]
    [InlineData((int)GgmlTensorType.Q6_K, 2560, 1024)]
    [InlineData(Q2_0, 2560, 12288)]
    public void TheDenseKernelsOutputs_Fingerprint(int type, int inDim, int outDim)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.DenseFastReads(type, inDim));
        long rowBytes = type == Q2_0 ? inDim / 64 * 18L : NativeDequant.RowSize(type, inDim);
        byte[] w;
        if (type == Q2_0)
        {
            var rng = new Random(inDim + outDim);
            w = new byte[rowBytes * outDim];
            rng.NextBytes(w);
            for (long b = 0; b < w.Length / 18; b++)
                BitConverter.TryWriteBytes(w.AsSpan((int)(b * 18), 2), (System.Half)(0.008f + 0.02f * (float)rng.NextDouble()));
        }
        else
            w = QuantRows(type, outDim, rowBytes, seed: type * 13 + outDim);
        var rnd = new Random(inDim * 3 + outDim);
        var x = new float[8 * inDim];
        for (int i = 0; i < x.Length; i++)
            x[i] = (float)((rnd.NextDouble() * 2 - 1) * (i % 97 == 3 ? 8 : 1));
        using Tensor wT = Up(allocator, w);
        using Tensor xT = Up(allocator, x);
        using Tensor xq = new Tensor(allocator, DType.UInt8, 8L * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using Tensor y = new Tensor(allocator, DType.Float32, 8, outDim);
        IntPtr stream = allocator.Stream.Handle;
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(xT), Ptr(xq), inDim, 8, stream, warpCooperative: true);
        string line = $"fingerprint type {type} {inDim} -> {outDim}:";
        for (int rows = 1; rows <= 8; rows++)
        {
            dk.DenseQ81Rows(Ptr(wT), Ptr(xq), Ptr(y), type, outDim, inDim, rowBytes, rows, stream);
            allocator.Stream.Synchronize();
            float[] o = Down(y, rows * outDim);
            Assert.All(o, v => Assert.True(float.IsFinite(v)));
            var bytes = new byte[o.Length * 4];
            Buffer.BlockCopy(o, 0, bytes, 0, bytes.Length);
            line += $" {rows}r {Convert.ToHexString(SHA256.HashData(bytes))[..12]}";
        }
        _output.WriteLine(line);
    }

    /// <summary>The same for the layouts the decode-once kernels do not read (the warp-per-row kernels): Flash Next's shared
    /// expert keeps its 640 -> 2560 down projection in Q4_0 or Q5_0. Run once with TS_DENSE_MULTI_ROWS=0 (the one-row
    /// kernel at every row count) and once without: the two must print the same.</summary>
    [CudaTheory]
    [InlineData((int)GgmlTensorType.Q4_0, 640, 2560)]
    [InlineData((int)GgmlTensorType.Q5_0, 640, 2560)]
    [InlineData((int)GgmlTensorType.Q4_0, 2560, 4096)]
    [InlineData((int)GgmlTensorType.Q2_K, 512, 2560)]
    [InlineData((int)GgmlTensorType.IQ3_S, 768, 3072)]
    public void TheRowKernelsOutputs_Fingerprint(int type, int inDim, int outDim)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.False(dk.DenseFastReads(type, inDim));
        Assert.True(Dsv4Kernels.DenseQ81RowsSupports(type, inDim));
        long rowBytes = NativeDequant.RowSize(type, inDim);
        byte[] w;
        int scaledBlock = type == (int)GgmlTensorType.Q4_0 ? 18 : type == (int)GgmlTensorType.Q5_0 ? 22 : 0;
        if (scaledBlock > 0)
        {
            var rng = new Random(inDim + outDim + type);
            w = new byte[rowBytes * outDim];
            rng.NextBytes(w);
            for (long b = 0; b < w.Length / scaledBlock; b++)
                BitConverter.TryWriteBytes(w.AsSpan((int)(b * scaledBlock), 2), (System.Half)(0.004f + 0.01f * (float)rng.NextDouble()));
        }
        else
            w = QuantRows(type, outDim, rowBytes, seed: type * 13 + outDim);
        var rnd = new Random(inDim * 5 + outDim);
        var x = new float[8 * inDim];
        for (int i = 0; i < x.Length; i++)
            x[i] = (float)((rnd.NextDouble() * 2 - 1) * (i % 89 == 7 ? 8 : 1));
        using Tensor wT = Up(allocator, w);
        using Tensor xT = Up(allocator, x);
        using Tensor xq = new Tensor(allocator, DType.UInt8, 8L * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using Tensor y = new Tensor(allocator, DType.Float32, 8, outDim);
        IntPtr stream = allocator.Stream.Handle;
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(xT), Ptr(xq), inDim, 8, stream, warpCooperative: true);
        string line = $"fingerprint rows type {type} {inDim} -> {outDim}:";
        for (int rows = 1; rows <= 8; rows++)
        {
            dk.DenseQ81Rows(Ptr(wT), Ptr(xq), Ptr(y), type, outDim, inDim, rowBytes, rows, stream);
            allocator.Stream.Synchronize();
            float[] o = Down(y, rows * outDim);
            Assert.All(o, v => Assert.True(float.IsFinite(v)));
            var bytes = new byte[o.Length * 4];
            Buffer.BlockCopy(o, 0, bytes, 0, bytes.Length);
            line += $" {rows}r {Convert.ToHexString(SHA256.HashData(bytes))[..12]}";
        }
        _output.WriteLine(line);
    }
}
