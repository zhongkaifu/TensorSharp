// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// NVFP4 products on the direct CUDA backend's resident weights (the executor's QuantizedMatMul): 1..16 rows run the
/// q8_1 dp4a matvec over the ggml blocks. Every row of a multi-row call equals the one-row call on it, to the bit, and
/// the rows are held to the dequantized weights with the activations rounded as the kernel rounds them (q8_1 with an
/// F16 scale), within 1e-4 relative L2 - what remains is accumulation order.
/// </summary>
public sealed class CudaNvfp4MatmulTests
{
    private const int Nvfp4 = 40, BlockBytes = 36, MaxRows = 16;
    private readonly ITestOutputHelper _output;

    public CudaNvfp4MatmulTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(5120, 512)]
    [InlineData(5120, 2304)]
    [InlineData(17408, 320)]
    [InlineData(1024, 4100)]
    public void RowsMatchTheOneRowCallAndTheDequantizedWeights(int inDim, int outDim)
    {
        long rowBytes = inDim / 64 * BlockBytes;
        byte[] weights = Nvfp4Rows(outDim, inDim, seed: 7);
        var act = new float[MaxRows, inDim];
        for (int t = 0; t < MaxRows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);

        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var allocator = new CudaAllocator();
            using var input = Tensor.FromArray(allocator, act);
            var alone = new float[MaxRows * outDim];
            for (int r = 0; r < MaxRows; r++)
            {
                using Tensor x = input.Narrow(0, r, 1);
                using var y = new Tensor(allocator, DType.Float32, 1, outDim);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, x, host, host, Nvfp4, inDim, outDim, weights.Length));
                Array.Copy(y.GetElementsAsFloat(outDim), 0, alone, r * outDim, outDim);
            }

            // The matvec's rows: up to 8 where NVFP4 weights are planar for the FP4 tensor cores (more run on the GEMMs),
            // up to 16 on the ggml blocks elsewhere.
            bool planar = allocator.Fp4Kernels != null;
            float[] got = null;
            foreach (int rows in planar ? new[] { 2, 3, 4, 5, 8 } : new[] { 2, 3, 4, 5, 8, 9, 16 })
            {
                using Tensor x = input.Narrow(0, 0, rows);
                using var y = new Tensor(allocator, DType.Float32, rows, outDim);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, x, host, host, Nvfp4, inDim, outDim, weights.Length));
                got = y.GetElementsAsFloat(rows * outDim);
                for (int i = 0; i < rows * outDim; i++)
                    if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(alone[i]))
                        Assert.Fail($"NVFP4 {inDim} -> {outDim}: row {i / outDim} of a {rows}-row call gives {got[i]:R} at column " +
                            $"{i % outDim}, the one-row call {alone[i]:R}");
            }
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);

            // The planar matvec keeps each 32-value scale in F32, the ggml-block one stores it as F16 (q8_1).
            float[,] reference = QuantizeDequantizeQ8_1(act, storeScaleAsHalf: !planar);
            var wRow = new float[inDim];
            double num = 0, den = 0;
            for (int o = 0; o < outDim; o++)
            {
                NativeDequant.DequantizeToFloat32(Nvfp4, weights, checked((int)(o * rowBytes)), wRow, 0, inDim);
                for (int t = 0; t < MaxRows; t++)
                {
                    double want = 0;
                    for (int k = 0; k < inDim; k++)
                        want += (double)wRow[k] * reference[t, k];
                    double d = alone[t * outDim + o] - want;
                    num += d * d;
                    den += want * want;
                }
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"NVFP4 {inDim} -> {outDim}: rows of multi-row calls bit-equal to one-row calls; relative L2 {rel:E2} " +
                "against the dequantized weights");
            Assert.True(rel <= 1e-4, $"NVFP4 {inDim} -> {outDim}: {rel:E2} from the dequantized weights");
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>
    /// Prompt rows on the FP4 tensor cores (sm_120a / sm_121a): the product equals the dequantized weights times the
    /// activations as the quantizer coded them (its own planes, dequantized on the host), within accumulation order,
    /// for one-wave split-K tiles (17 and 64 rows) and the 128-column tiles (200 rows), and an output edge.
    /// </summary>
    [CudaTheory]
    [InlineData(5120, 512, 17)]
    [InlineData(5120, 2304, 64)]
    [InlineData(17408, 320, 33)]
    [InlineData(1024, 4100, 200)]
    [InlineData(5120, 2304, 300)]
    public void PromptRowsOnTheFp4TensorCoresMatchTheirQuantizedOperands(int inDim, int outDim, int rows)
    {
        using var allocator = new CudaAllocator();
        CudaFp4Kernels fp4 = allocator.Fp4Kernels;
        if (fp4 == null)
        {
            allocator.Context.MakeCurrent();
            CudaFp4Kernels.TryCreate(out CudaFp4Kernels none, out string reason);
            none?.Dispose();
            _output.WriteLine($"no FP4 tensor-core products here ({reason}): the product keeps the dequantize + cuBLAS route");
            return;
        }
        long rowBytes = inDim / 64 * BlockBytes;
        byte[] weights = Nvfp4Rows(outDim, inDim, seed: 5);
        var act = new float[rows, inDim];
        for (int t = 0; t < rows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);
        IntPtr host = Marshal.AllocHGlobal(weights.Length);
        try
        {
            Marshal.Copy(weights, 0, host, weights.Length);
            using var input = Tensor.FromArray(allocator, act);
            using var y = new Tensor(allocator, DType.Float32, rows, outDim);
            Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(y, input, host, host, Nvfp4, inDim, outDim, weights.Length));
            float[] got = y.GetElementsAsFloat(rows * outDim);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);

            // The activations as the quantizer codes them.
            var (codeBytes, scaleBytes, rowScaleBytes) = CudaFp4Kernels.ActivationBytes(rows, inDim);
            using var codes = new Tensor(allocator, DType.UInt8, codeBytes);
            using var scales = new Tensor(allocator, DType.UInt8, scaleBytes);
            using var rowScale = new Tensor(allocator, DType.Float32, rows);
            allocator.Context.MakeCurrent();
            fp4.Quantize(Ptr(input), Ptr(codes), Ptr(scales), Ptr(rowScale), inDim, rows, allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            byte[] c = Bytes(codes, codeBytes), sc = Bytes(scales, scaleBytes);
            float[] rs = rowScale.GetElementsAsFloat(rows);
            var coded = new float[rows, inDim];
            for (int t = 0; t < rows; t++)
                for (int k = 0; k < inDim; k++)
                {
                    int b = c[(long)t * inDim / 2 + k / 16 * 8 + k % 8];
                    int nibble = k % 16 < 8 ? b & 15 : b >> 4;
                    coded[t, k] = E2M1[nibble] * Ue4m3(sc[(long)t * inDim / 16 + k / 16]) * rs[t];
                }

            var wRow = new float[inDim];
            double num = 0, den = 0;
            for (int o = 0; o < outDim; o++)
            {
                NativeDequant.DequantizeToFloat32(Nvfp4, weights, checked((int)(o * rowBytes)), wRow, 0, inDim);
                for (int t = 0; t < rows; t++)
                {
                    double want = 0;
                    for (int k = 0; k < inDim; k++)
                        want += (double)wRow[k] * coded[t, k];
                    double d = got[t * outDim + o] - want;
                    num += d * d;
                    den += want * want;
                }
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"NVFP4 FP4 GEMM {inDim} -> {outDim}, {rows} rows: relative L2 {rel:E2} against the quantized operands");
            Assert.True(rel <= 1e-5, $"NVFP4 FP4 GEMM {inDim} -> {outDim}, {rows} rows: {rel:E2} from the quantized operands");
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    /// <summary>Rows of an NVFP4 table gathered on the device (the planar layout's lookup where weights are planar) equal
    /// the host's dequantization.</summary>
    [CudaFact]
    public void GatheredRowsMatchTheHostDequantization()
    {
        const int k = 512, tableRows = 300;
        byte[] table = Nvfp4Rows(tableRows, k, seed: 9);
        int[] ids = { 0, 299, 17, 17, 128 };
        IntPtr host = Marshal.AllocHGlobal(table.Length);
        try
        {
            Marshal.Copy(table, 0, host, table.Length);
            using var allocator = new CudaAllocator();
            using var indices = new Tensor(allocator, DType.Int32, ids.Length);
            indices.SetElementsAsInt(ids);
            using var rows = new Tensor(allocator, DType.Float32, ids.Length, k);
            Assert.True(CudaQuantizedOps.TryGetRowsQuantizedToFloat32(rows, host, host, Nvfp4, k, tableRows, table.Length, indices));
            float[] got = rows.GetElementsAsFloat(ids.Length * k);
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            var want = new float[k];
            for (int i = 0; i < ids.Length; i++)
            {
                NativeDequant.DequantizeToFloat32(Nvfp4, table, ids[i] * (k / 64) * BlockBytes, want, 0, k);
                for (int c = 0; c < k; c++)
                    Assert.Equal(want[c], got[i * k + c]);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    private static readonly float[] E2M1 = { 0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f };

    /// <summary>A UE4M3 scale byte (ggml's ggml_ue4m3_to_fp32 without its halving: E2M1 above is undoubled).</summary>
    private static float Ue4m3(byte v)
    {
        if (v == 0 || v == 0x7F) return 0f;
        int e = (v >> 3) & 15, m = v & 7;
        return e == 0 ? m / 512f : (1f + m / 8f) * MathF.Pow(2, e - 7);
    }

    private static IntPtr Ptr(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static byte[] Bytes(Tensor t, long count)
    {
        var bytes = new byte[count];
        t.CopyToArray(bytes);
        return bytes;
    }

    /// <summary>NVFP4 rows of random codes under scales a checkpoint holds (UE4M3 exponents 4..10, never the NaN pattern).</summary>
    private static byte[] Nvfp4Rows(int rows, int cols, int seed)
    {
        var bytes = new byte[(long)rows * (cols / 64) * BlockBytes];
        uint x = (uint)(0x9E3779B9 * seed);
        for (int i = 0; i < bytes.Length; i++)
        {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            bytes[i] = (byte)x;
        }
        for (long b = 0; b < bytes.Length; b += BlockBytes)
            for (int s = 0; s < 4; s++)
                bytes[b + s] = (byte)(((4 + bytes[b + s] % 7) << 3) | (bytes[b + s] & 7));
        return bytes;
    }

    private static float[,] QuantizeDequantizeQ8_1(float[,] input, bool storeScaleAsHalf)
    {
        int rows = input.GetLength(0), cols = input.GetLength(1);
        var result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int block = 0; block < cols; block += 32)
            {
                float amax = 0;
                for (int i = 0; i < 32; i++)
                    amax = MathF.Max(amax, MathF.Abs(input[r, block + i]));
                float d = amax > 0 ? amax / 127.0f : 0, id = d > 0 ? 1.0f / d : 0, stored = storeScaleAsHalf ? (float)(System.Half)d : d;
                for (int i = 0; i < 32; i++)
                    result[r, block + i] = stored * Math.Clamp((int)MathF.Round(input[r, block + i] * id, MidpointRounding.ToEven), -127, 127);
            }
        return result;
    }
}
