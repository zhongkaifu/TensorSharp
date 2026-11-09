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
/// The direct-CUDA DeepSeek engine's dense projections (<see cref="Dsv4CudaEngine.DenseMatMul"/>) compute
/// every row of a decode-size call - 2 to 16 rows: a batched decode step, a verify window - bit for bit as a
/// one-row call computes it, so a batched step answers each sequence exactly as its own forward does. Held
/// at the dense layouts and shapes of DeepSeek V4.1 1017: Q5_K and Q8_0 shared experts (5120 to 2304 and
/// 2304 to 5120), Q6_K and Q8_0 attention, BF16 compressor and indexer projections (5120 to 512); and at
/// the other quantized layouts the engine's two dense routes cover: Q4_K at the same shapes, Q2_K, Q3_K,
/// IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and MXFP4 at one. Before, a Q5_K or Q4_K projection ran one row on
/// a q8_1 dp4a matvec and several on the scalar batched F32 kernel (IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and MXFP4
/// on an F32 matvec and that kernel), and a BF16 one several on cuBLAS over BF16-rounded activations, so a
/// sequence's logits depended on how many sequences shared its step. F16 and F32 weights (cuBLAS, whose
/// kernel choice follows the row count), the legacy Q4_0 to Q5_1 layouts, IQ2_S and NVFP4 are not held to
/// this; none is a dense layout of the 1017.
/// </summary>
public sealed class Dsv4DenseRowInvarianceTests
{
    private const int Q5_K = (int)GgmlTensorType.Q5_K, Q6_K = (int)GgmlTensorType.Q6_K, Q8_0 = (int)GgmlTensorType.Q8_0;
    private const int Q2_K = (int)GgmlTensorType.Q2_K, Q3_K = (int)GgmlTensorType.Q3_K, Q4_K = (int)GgmlTensorType.Q4_K;
    private const int IQ2_XXS = (int)GgmlTensorType.IQ2_XXS, IQ2_XS = (int)GgmlTensorType.IQ2_XS;
    private const int IQ3_XXS = (int)GgmlTensorType.IQ3_XXS, IQ3_S = (int)GgmlTensorType.IQ3_S;
    private const int IQ4_NL = (int)GgmlTensorType.IQ4_NL, IQ4_XS = (int)GgmlTensorType.IQ4_XS;
    private const int MXFP4 = (int)GgmlTensorType.MXFP4, BF16 = (int)GgmlTensorType.BF16;
    private const int MaxRows = Dsv4CudaEngine.MaxBatchedDecodeRows;

    private readonly ITestOutputHelper _output;

    public Dsv4DenseRowInvarianceTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Each row of every call of 2..16 rows equals the one-row call on that row, to the bit. The rows are
    /// also held to the dequantized weights with the activations rounded as the kernels round them (q8_1
    /// for the quantized layouts, none for BF16), within 1e-4 relative L2 - what remains is accumulation
    /// order - so a projection cannot be row-invariant by being wrong.
    /// </summary>
    [CudaTheory]
    [InlineData(Q5_K, 5120, 2304)]
    [InlineData(Q5_K, 2304, 5120)]
    [InlineData(Q5_K, 5120, 512)]
    [InlineData(BF16, 5120, 2304)]
    [InlineData(BF16, 2304, 5120)]
    [InlineData(BF16, 5120, 512)]
    [InlineData(Q6_K, 5120, 2304)]
    [InlineData(Q6_K, 2304, 5120)]
    [InlineData(Q6_K, 5120, 512)]
    [InlineData(Q8_0, 5120, 2304)]
    [InlineData(Q8_0, 2304, 5120)]
    [InlineData(Q8_0, 5120, 512)]
    [InlineData(Q4_K, 5120, 2304)]
    [InlineData(Q4_K, 2304, 5120)]
    [InlineData(Q4_K, 5120, 512)]
    [InlineData(Q2_K, 5120, 512)]
    [InlineData(Q3_K, 5120, 512)]
    [InlineData(IQ2_XXS, 5120, 512)]
    [InlineData(IQ3_XXS, 5120, 512)]
    [InlineData(IQ3_S, 5120, 512)]
    [InlineData(IQ4_NL, 5120, 512)]
    // ISTA's Flash Next keeps a 640-wide shared-expert down projection in IQ4_NL: whole 32-value blocks, not
    // whole super-blocks.
    [InlineData(IQ4_NL, 640, 2560)]
    [InlineData(Q8_0, 640, 2560)]
    [InlineData(IQ4_XS, 5120, 512)]
    [InlineData(MXFP4, 5120, 512)]
    public void EveryRowOfABatchedCall_IsTheOneRowCallBitForBit(int ggmlType, int inDim, int outDim)
    {
        long rowBytes = ggmlType == BF16 ? 2L * inDim : NativeDequant.RowSize(ggmlType, inDim);
        byte[] weights = ggmlType == BF16 ? Bf16Rows(outDim, inDim, seed: 11) : QuantRows(ggmlType, outDim, rowBytes, seed: 11);
        var act = new float[MaxRows, inDim];
        for (int t = 0; t < MaxRows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        IntPtr stream = allocator.Stream.Handle;
        using var w = Up(allocator, weights);
        using var input = Up(allocator, act);
        using var single = new Tensor(allocator, DType.Float32, MaxRows, outDim);
        using var batched = new Tensor(allocator, DType.Float32, MaxRows, outDim);
        using var q81 = new Tensor(allocator, DType.UInt8, (long)MaxRows * (inDim / 32) * CudaKernels.Q81BlockBytes);
        var dw = new DeviceWeight { Ptr = Dsv4CudaEngine.Ptr(w), Type = ggmlType, Ne0 = inDim, Ne1 = outDim, RowBytes = rowBytes };

        // Each row alone, as a sequence's own decode step runs it.
        for (int r = 0; r < MaxRows; r++)
        {
            using Tensor x = input.Narrow(0, r, 1);
            using Tensor y = single.Narrow(0, r, 1);
            Dsv4CudaEngine.DenseMatMul(allocator.Kernels, dk, stream, dw, x, y, 1, q81);
        }
        allocator.Stream.Synchronize();
        float[] alone = Down(single, MaxRows * outDim);

        float[] got = null;
        for (int rows = 2; rows <= MaxRows; rows++)
        {
            allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(batched), MaxRows * outDim, float.NaN, stream);
            Dsv4CudaEngine.DenseMatMul(allocator.Kernels, dk, stream, dw, input, batched, rows, q81);
            allocator.Stream.Synchronize();
            got = Down(batched, MaxRows * outDim);
            for (int i = 0; i < rows * outDim; i++)
            {
                if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(alone[i]))
                    Assert.Fail($"{Name(ggmlType)} {inDim} -> {outDim}: row {i / outDim} of a {rows}-row call gives " +
                        $"{got[i]:R} at column {i % outDim}, the one-row call {alone[i]:R}");
            }
        }

        // The first and last rows against the dequantized weights.
        float[,] reference = ggmlType == BF16 ? act : QuantizeDequantizeQ8_1(act);
        var wRow = new float[inDim];
        double num = 0, den = 0;
        for (int o = 0; o < outDim; o++)
        {
            if (ggmlType == BF16)
            {
                for (int k = 0; k < inDim; k++)
                    wRow[k] = BitConverter.Int32BitsToSingle((weights[(o * inDim + k) * 2] | weights[(o * inDim + k) * 2 + 1] << 8) << 16);
            }
            else
                NativeDequant.DequantizeToFloat32(ggmlType, weights, checked((int)(o * rowBytes)), wRow, 0, inDim);
            foreach (int t in new[] { 0, MaxRows - 1 })
            {
                double want = 0;
                for (int k = 0; k < inDim; k++)
                    want += (double)wRow[k] * reference[t, k];
                double d = got[t * outDim + o] - want;
                num += d * d;
                den += want * want;
            }
        }
        double rel = Math.Sqrt(num / den);
        _output.WriteLine($"{Name(ggmlType)} {inDim} -> {outDim}: rows of 2..{MaxRows}-row calls bit-equal to one-row calls; " +
            $"relative L2 {rel:E2} against the dequantized weights");
        Assert.True(rel <= 1e-4, $"{Name(ggmlType)} {inDim} -> {outDim}: {rel:E2} from the dequantized weights");
    }

    /// <summary>The layouts that take the q8_1 rows kernel are the ones the shared routing would not keep
    /// row-invariant, and such a decode-size projection needs the device's q8_1 scratch: a device the engine
    /// sized none for (or too little) is refused rather than written past.</summary>
    [Fact]
    public void AQ81RowsProjectionWithoutItsScratch_IsRefused()
    {
        foreach (int type in new[] { Q4_K, Q5_K, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4 })
            Assert.True(Dsv4CudaEngine.TakesDenseQ81Rows(type, 2304), Name(type));
        // The shared routing's own row-invariant kernels; and IQ2_XS, which it cannot run at all.
        foreach (int type in new[] { Q2_K, Q3_K, Q6_K, Q8_0, IQ2_XXS, IQ2_XS, BF16, (int)GgmlTensorType.F16, (int)GgmlTensorType.F32 })
            Assert.False(Dsv4CudaEngine.TakesDenseQ81Rows(type, 2304), Name(type));
        Assert.False(Dsv4CudaEngine.TakesDenseQ81Rows(Q4_K, 2368));
        var w = new DeviceWeight { Type = Q5_K, Ne0 = 2304, Ne1 = 512 };
        var error = Assert.Throws<InvalidOperationException>(() =>
            Dsv4CudaEngine.DenseMatMul(null, null, IntPtr.Zero, w, null, null, 4, null));
        Assert.Contains("no q8_1 scratch for 4 rows of a 2304-wide Q5_K projection", error.Message);
    }

    private static string Name(int ggmlType) => ((GgmlTensorType)ggmlType).ToString();

    /// <summary>[rows, cols] BF16 weights of unit-sized products: random F32 values rounded to nearest even.</summary>
    private static byte[] Bf16Rows(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var bytes = new byte[2L * rows * cols];
        float scale = 2f / MathF.Sqrt(cols);
        for (long i = 0; i < (long)rows * cols; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits((float)(rng.NextDouble() - 0.5) * scale);
            bits += 0x7FFF + ((bits >> 16) & 1);
            bytes[2 * i] = (byte)(bits >> 16);
            bytes[2 * i + 1] = (byte)(bits >> 24);
        }
        return bytes;
    }
}
