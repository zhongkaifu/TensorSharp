// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Numerical parity of the MLX kernels for the low-bit GGUF types (Q2_K, Q3_K, IQ2_XS, IQ1_S, IQ1_M;
// MlxLowBitQuantKernels.cs), and of the few-row (2-31 rows, a speculative verify window) and
// dequant-GEMM (32+ rows, a prompt chunk) routes every raw i-quant / K-quant weight now takes,
// against the managed CPU dequant (a port of ggml's dequantize_row_*). Before these kernels the
// low-bit types ran on the CPU managed-dequant path on MLX: Qwen3.8-27B-UD-IQ3_XXS decoded at
// 1.8 tok/s. A wrong grid index or sign bit decodes to fluent garbage, so the arithmetic is pinned.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class MlxLowBitQuantKernelTests
{
    private const int QK_K = 256;
    private readonly ITestOutputHelper _output;

    public MlxLowBitQuantKernelTests(ITestOutputHelper output) => _output = output;

    // rows 1: the decode kernel; 3 and 5: the few-row simdgroup kernel on F32 tiles, 8 on F16 tiles;
    // 40: F16 dequant + GEMM.
    [MlxTheory]
    [InlineData((int)GgmlTensorType.Q2_K, 1)]
    [InlineData((int)GgmlTensorType.Q2_K, 3)]
    [InlineData((int)GgmlTensorType.Q2_K, 8)]
    [InlineData((int)GgmlTensorType.Q2_K, 40)]
    [InlineData((int)GgmlTensorType.Q3_K, 1)]
    [InlineData((int)GgmlTensorType.Q3_K, 8)]
    [InlineData((int)GgmlTensorType.Q3_K, 40)]
    [InlineData((int)GgmlTensorType.IQ2_XS, 1)]
    [InlineData((int)GgmlTensorType.IQ2_XS, 8)]
    [InlineData((int)GgmlTensorType.IQ2_XS, 40)]
    [InlineData((int)GgmlTensorType.IQ1_S, 1)]
    [InlineData((int)GgmlTensorType.IQ1_S, 8)]
    [InlineData((int)GgmlTensorType.IQ1_S, 40)]
    [InlineData((int)GgmlTensorType.IQ1_M, 1)]
    [InlineData((int)GgmlTensorType.IQ1_M, 8)]
    [InlineData((int)GgmlTensorType.IQ1_M, 40)]
    // Types with kernels before: the decode matvec, few-row and dequant-GEMM routes.
    [InlineData((int)GgmlTensorType.IQ2_XXS, 1)]
    [InlineData((int)GgmlTensorType.IQ2_S, 1)]
    [InlineData((int)GgmlTensorType.IQ3_S, 1)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 1)]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 8)]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 40)]
    [InlineData((int)GgmlTensorType.IQ2_S, 8)]
    [InlineData((int)GgmlTensorType.IQ2_S, 40)]
    [InlineData((int)GgmlTensorType.IQ3_S, 8)]
    [InlineData((int)GgmlTensorType.IQ3_S, 40)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 5)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 8)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 40)]
    [InlineData((int)GgmlTensorType.IQ4_XS, 8)]
    [InlineData((int)GgmlTensorType.Q6_K, 8)]
    public void Matmul_MatchesManagedDequantReference(int ggmlType, int rows)
    {
        const int inDim = 512;
        const int outDim = 37;   // not a multiple of the few-row kernel's 32 columns
        byte[] weights = CreateRows(ggmlType, outDim, inDim, seed: 0x4B1D + ggmlType * 7);
        float[,] input = new float[rows, inDim];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < inDim; c++)
                input[r, c] = MathF.Sin((r + 3) * (c + 5) * 0.011f) * 0.6f;
        float[] expected = Reference(ggmlType, weights, outDim, inDim, input);

        using var allocator = new MlxAllocator();
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x6A0000 + ggmlType * 64 + rows);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            Assert.True(MlxQuantizedOps.CanPreloadQuantizedType(ggmlType));
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);
            using var inputTensor = Tensor.FromArray(allocator, input);
            using var outputTensor = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(MlxQuantizedOps.TryAddmmQuantizedToFloat32(outputTensor, inputTensor, cacheKey, IntPtr.Zero,
                ggmlType, inDim, outDim, weights.Length));
            float[] actual = outputTensor.GetElementsAsFloat(rows * outDim);
            // Below 8 rows the routes are F32 (only the summation order differs); from 8 rows the
            // few-row tiles and the dequant GEMM are F16, as the simdgroup kernel those rows took was.
            double scale = 0;
            foreach (float v in expected) scale = Math.Max(scale, Math.Abs(v));
            double tolerance = rows >= 8 ? 4e-3 * scale + 1e-3 : 2e-5 * scale + 1e-5;
            AssertClose($"{(GgmlTensorType)ggmlType} rows={rows}", expected, actual, tolerance);
        }
        finally
        {
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            Marshal.FreeHGlobal(host);
        }
    }

    [MlxTheory]
    [InlineData((int)GgmlTensorType.Q2_K)]
    [InlineData((int)GgmlTensorType.Q3_K)]
    [InlineData((int)GgmlTensorType.IQ2_XS)]
    [InlineData((int)GgmlTensorType.IQ1_S)]
    [InlineData((int)GgmlTensorType.IQ1_M)]
    public void GetRows_MatchesManagedDequantReference(int ggmlType)
    {
        const int inDim = 512;
        const int outDim = 6;
        int[] rowIds = { 4, 0, 2, 2 };
        byte[] weights = CreateRows(ggmlType, outDim, inDim, seed: 0x77E1 + ggmlType);
        float[] expected = new float[rowIds.Length * inDim];
        for (int i = 0; i < rowIds.Length; i++)
            DequantizeRow(ggmlType, weights, rowIds[i], inDim, expected, i * inDim);

        using var allocator = new MlxAllocator();
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        IntPtr cacheKey = new(0x6B0000 + ggmlType);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            MlxQuantizedOps.PreloadQuantizedWeight(allocator, cacheKey, host, ggmlType, inDim, outDim, weights.Length);
            using var indices = Tensor.FromArray(allocator, rowIds);
            using var outputTensor = new Tensor(allocator, DType.Float32, rowIds.Length, inDim);
            Assert.True(MlxQuantizedOps.TryGetRowsQuantizedToFloat32(outputTensor, cacheKey, IntPtr.Zero,
                ggmlType, inDim, outDim, weights.Length, indices));
            AssertClose($"{(GgmlTensorType)ggmlType} get_rows", expected, outputTensor.GetElementsAsFloat(rowIds.Length * inDim), 1e-6);
        }
        finally
        {
            MlxQuantizedOps.ReleaseQuantizedWeight(allocator, cacheKey);
            Marshal.FreeHGlobal(host);
        }
    }

    private void AssertClose(string label, float[] expected, float[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxAbs = 0;
        int worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), $"{label}: non-finite at {i}");
            double err = Math.Abs((double)expected[i] - actual[i]);
            if (err > maxAbs) { maxAbs = err; worst = i; }
        }
        _output.WriteLine($"{label}: n={expected.Length} maxAbsErr={maxAbs:E3} tol={tolerance:E3} (expected {expected[worst]:F7}, actual {actual[worst]:F7})");
        Assert.True(maxAbs <= tolerance, $"{label}: max abs error {maxAbs:E3} > {tolerance:E3} at {worst} (expected {expected[worst]}, actual {actual[worst]})");
    }

    private static float[] Reference(int ggmlType, byte[] weights, int outDim, int inDim, float[,] input)
    {
        int rows = input.GetLength(0);
        var expected = new float[rows * outDim];
        var row = new float[inDim];
        for (int o = 0; o < outDim; o++)
        {
            DequantizeRow(ggmlType, weights, o, inDim, row, 0);
            for (int r = 0; r < rows; r++)
            {
                double sum = 0;
                for (int c = 0; c < inDim; c++)
                    sum += (double)input[r, c] * row[c];
                expected[r * outDim + o] = (float)sum;
            }
        }
        return expected;
    }

    private static void DequantizeRow(int ggmlType, byte[] weights, int row, int inDim, float[] dst, int dstOffset)
    {
        long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, inDim);
        ManagedQuantizedOps.DequantizeToFloat32(ggmlType, weights, checked((int)(row * rowBytes)), dst, dstOffset, inDim);
    }

    /// <summary>Random payload bytes (every pattern is legal: codebook indices, signs, packed
    /// scales) with each type's F16 scale fields set to small, exactly representable values.</summary>
    private static byte[] CreateRows(int ggmlType, int rows, int cols, int seed)
    {
        int blocksPerRow = cols / QK_K;
        int blockBytes = checked((int)ManagedQuantizedOps.RowSize(ggmlType, QK_K));
        byte[] raw = new byte[(long)rows * blocksPerRow * blockBytes];
        uint state = (uint)seed | 1u;
        for (int i = 0; i < raw.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            raw[i] = (byte)(state >> 7);
        }
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int o = (r * blocksPerRow + b) * blockBytes;
                float d = 0.001953125f + r * 0.000244140625f + b * 0.0001220703125f;
                switch ((GgmlTensorType)ggmlType)
                {
                    case GgmlTensorType.Q2_K:
                        WriteHalf(raw, o + 80, d);
                        WriteHalf(raw, o + 82, d * 0.5f);
                        break;
                    case GgmlTensorType.Q3_K:
                        WriteHalf(raw, o + 108, d);
                        break;
                    case GgmlTensorType.Q6_K:
                        WriteHalf(raw, o + 208, d);
                        break;
                    case GgmlTensorType.IQ1_M:
                    {
                        // The F16 super-block scale is the top nibbles of the four scale words.
                        ushort h = BitConverter.HalfToUInt16Bits((System.Half)d);
                        for (int s = 0; s < 4; s++)
                        {
                            int at = o + 48 + 2 * s;
                            raw[at + 1] = (byte)((raw[at + 1] & 0x0F) | (((h >> (4 * s)) & 0xF) << 4));
                        }
                        break;
                    }
                    default:
                        WriteHalf(raw, o, d);   // IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ4_XS: d first
                        break;
                }
            }
        return raw;
    }

    private static void WriteHalf(byte[] raw, int offset, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((System.Half)value);
        raw[offset] = (byte)bits;
        raw[offset + 1] = (byte)(bits >> 8);
    }
}
