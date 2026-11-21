// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
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
/// The CUDA executor's decode-size products of the IQ2 and IQ3 formats (CudaGraphExecutor.QuantizedMatMul's dense row
/// routes), up to 16 rows a call. Those are the formats Unsloth's dynamic quants keep most of a dense model's projections in:
/// Muse-Glimmer-30B's UD-IQ2_XXS has attention Q and gate in IQ2_XXS, FFN gate and up in IQ2_S, FFN down in IQ3_XXS, and
/// attention output in IQ3_S. The IQ2 formats run on <see cref="Dsv4Kernels.DenseQ81Rows"/> over q8_1 activations, the IQ3
/// formats on <see cref="Dsv4Kernels.DenseF32Rows"/> over the F32 activations themselves.
/// <para>Every row of a 2..16-row call equals the one-row call on that row, to the bit, so a verify window answers each
/// position as plain decoding does. The rows are also held to the dequantized weights (with the activations rounded to q8_1
/// where the kernel rounds them), so a projection cannot be row-invariant by being wrong. The shapes are the 30B's: a
/// 6656-wide input, FFN down's 19968, and attention output's 8192.</para>
/// </summary>
public sealed class CudaIqDenseRowsTests(ITestOutputHelper output)
{
    private const int MaxRows = Dsv4Kernels.DenseMultiMaxRows;

    [CudaTheory]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 6656, 512)]
    [InlineData((int)GgmlTensorType.IQ2_S, 6656, 512)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 19968, 256)]
    [InlineData((int)GgmlTensorType.IQ3_S, 6656, 512)]
    // Few output rows of a long input take the split-K form: still row-invariant.
    [InlineData((int)GgmlTensorType.IQ2_S, 6656, 1024)]
    // Past the split-K kernel's outputs: the decode-once IQ2 rows against the one-row (multi) kernel, a partial last block too.
    [InlineData((int)GgmlTensorType.IQ2_XXS, 6656, 4096)]
    [InlineData((int)GgmlTensorType.IQ2_S, 6656, 4100)]
    public void EveryRowOfACallOfUpTo16Rows_IsTheOneRowCallBitForBit(int ggmlType, int inDim, int outDim)
    {
        long rowBytes = NativeDequant.RowSize(ggmlType, inDim);
        byte[] weights = QuantRows(ggmlType, outDim, rowBytes, seed: 23);
        var act = new float[MaxRows, inDim];
        for (int t = 0; t < MaxRows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        if (ggmlType == (int)GgmlTensorType.IQ2_S && !dk.DecodesLowBitIqExperts)
        {
            output.WriteLine("NOT RUN: this module does not decode IQ2_S, and the executor keeps IQ2_S off the dense row route");
            return;
        }
        Assert.True(ggmlType == (int)GgmlTensorType.IQ2_S || Dsv4Kernels.DenseQ81RowsSupports(ggmlType, inDim));
        IntPtr stream = allocator.Stream.Handle;
        using var w = Up(allocator, weights);
        using var input = Up(allocator, act);
        using var q81 = new Tensor(allocator, DType.UInt8, (long)MaxRows * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var single = new Tensor(allocator, DType.Float32, MaxRows, outDim);
        using var batched = new Tensor(allocator, DType.Float32, MaxRows, outDim);
        allocator.Kernels.LaunchQuantizeQ81Rows(Dsv4CudaEngine.Ptr(input), Dsv4CudaEngine.Ptr(q81), inDim, MaxRows, stream, true);
        long qRow = inDim / 32 * CudaKernels.Q81BlockBytes;

        for (int r = 0; r < MaxRows; r++)
            dk.DenseQ81Rows(Dsv4CudaEngine.Ptr(w), Dsv4CudaEngine.Ptr(q81) + (nint)(r * qRow), Dsv4CudaEngine.Ptr(single) + (nint)((long)r * outDim * sizeof(float)),
                ggmlType, outDim, inDim, rowBytes, 1, stream);
        allocator.Stream.Synchronize();
        float[] alone = Down(single, MaxRows * outDim);

        float[] got = null;
        for (int rows = 2; rows <= MaxRows; rows++)
        {
            allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(batched), MaxRows * outDim, float.NaN, stream);
            dk.DenseQ81Rows(Dsv4CudaEngine.Ptr(w), Dsv4CudaEngine.Ptr(q81), Dsv4CudaEngine.Ptr(batched), ggmlType, outDim, inDim, rowBytes, rows, stream);
            allocator.Stream.Synchronize();
            got = Down(batched, MaxRows * outDim);
            for (int i = 0; i < rows * outDim; i++)
                if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(alone[i]))
                    Assert.Fail($"{(GgmlTensorType)ggmlType} {inDim} -> {outDim}: row {i / outDim} of a {rows}-row call gives {got[i]:R} at " +
                        $"column {i % outDim}, the one-row call {alone[i]:R}");
        }

        float[,] reference = QuantizeDequantizeQ8_1(act);
        var wRow = new float[inDim];
        double num = 0, den = 0;
        for (int o = 0; o < outDim; o++)
        {
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
        output.WriteLine($"{(GgmlTensorType)ggmlType} {inDim} -> {outDim}: rows of 2..{MaxRows}-row calls bit-equal to one-row calls; " +
            $"relative L2 {rel:E2} against the dequantized weights");
        Assert.True(rel <= 1e-4, $"{(GgmlTensorType)ggmlType} {inDim} -> {outDim}: {rel:E2} from the dequantized weights");
    }

    [CudaTheory]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 19968, 256)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 6656, 515)]
    [InlineData((int)GgmlTensorType.IQ3_S, 8192, 512)]
    [InlineData((int)GgmlTensorType.IQ3_S, 6656, 33)]
    public void EveryRowOfAnF32CallOfUpTo16Rows_IsTheOneRowCallBitForBit(int ggmlType, int inDim, int outDim)
    {
        long rowBytes = NativeDequant.RowSize(ggmlType, inDim);
        byte[] weights = QuantRows(ggmlType, outDim, rowBytes, seed: 31);
        var act = new float[MaxRows, inDim];
        for (int t = 0; t < MaxRows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.HasDenseF32Rows, "the dsv4 module has no ts_dsv4_dense_f32_rows_f32");
        Assert.True(Dsv4Kernels.DenseF32RowsSupports(ggmlType, inDim));
        IntPtr stream = allocator.Stream.Handle;
        using var w = Up(allocator, weights);
        using var input = Up(allocator, act);
        using var single = new Tensor(allocator, DType.Float32, MaxRows, outDim);
        using var batched = new Tensor(allocator, DType.Float32, MaxRows, outDim);

        for (int r = 0; r < MaxRows; r++)
            dk.DenseF32Rows(Dsv4CudaEngine.Ptr(w), Dsv4CudaEngine.Ptr(input) + (nint)((long)r * inDim * sizeof(float)),
                Dsv4CudaEngine.Ptr(single) + (nint)((long)r * outDim * sizeof(float)), ggmlType, outDim, inDim, rowBytes, 1, stream);
        allocator.Stream.Synchronize();
        float[] alone = Down(single, MaxRows * outDim);

        float[] got = null;
        for (int rows = 2; rows <= MaxRows; rows++)
        {
            allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(batched), MaxRows * outDim, float.NaN, stream);
            dk.DenseF32Rows(Dsv4CudaEngine.Ptr(w), Dsv4CudaEngine.Ptr(input), Dsv4CudaEngine.Ptr(batched), ggmlType, outDim, inDim, rowBytes,
                rows, stream);
            allocator.Stream.Synchronize();
            got = Down(batched, MaxRows * outDim);
            for (int i = 0; i < rows * outDim; i++)
                if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(alone[i]))
                    Assert.Fail($"{(GgmlTensorType)ggmlType} {inDim} -> {outDim}: row {i / outDim} of a {rows}-row call gives {got[i]:R} at " +
                        $"column {i % outDim}, the one-row call {alone[i]:R}");
            for (int i = rows * outDim; i < MaxRows * outDim; i++)
                Assert.True(float.IsNaN(got[i]), $"a {rows}-row call wrote row {i / outDim}");
        }

        // Against the dequantized weights times the F32 activations themselves (no q8_1 rounding), in double.
        var wRow = new float[inDim];
        double num = 0, den = 0;
        for (int o = 0; o < outDim; o++)
        {
            NativeDequant.DequantizeToFloat32(ggmlType, weights, checked((int)(o * rowBytes)), wRow, 0, inDim);
            for (int t = 0; t < MaxRows; t++)
            {
                double want = 0;
                for (int k = 0; k < inDim; k++)
                    want += (double)wRow[k] * act[t, k];
                double d = got[t * outDim + o] - want;
                num += d * d;
                den += want * want;
            }
        }
        double rel = Math.Sqrt(num / den);
        output.WriteLine($"{(GgmlTensorType)ggmlType} {inDim} -> {outDim} over F32 rows: rows of 2..{MaxRows}-row calls bit-equal to one-row " +
            $"calls; relative L2 {rel:E2} against the dequantized weights and the F32 activations");
        Assert.True(rel <= 1e-6, $"{(GgmlTensorType)ggmlType} {inDim} -> {outDim}: {rel:E2} from the dequantized weights");
    }
}
