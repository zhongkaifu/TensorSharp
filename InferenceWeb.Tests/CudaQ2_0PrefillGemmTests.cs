// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Prompt-size Q2_0 products through an F16 copy of the weight and the tensor-core GEMM (CudaQuantizedOps.RunPrefillF16Gemm,
// the direct Flash Next engine's prompt route for ISTA's Q2_0 dense projections): the dequantizer's Q2_0 layout must be
// ggml's (dequantize_row_q2_0), checked against a host decode of the same blocks.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public unsafe class CudaQ2_0PrefillGemmTests
{
    private readonly ITestOutputHelper _output;

    public CudaQ2_0PrefillGemmTests(ITestOutputHelper output) => _output = output;

    /// <summary>One Q2_0 block: an F16 scale and 64 two-bit codes 0..3 meaning -1..2; weight j sits in byte
    /// (j / 32) * 8 + (j % 32) / 4 at bit 2 * (j % 4).</summary>
    private static float DecodeQ2_0(byte[] w, int row, int col, int inDim)
    {
        int blocksPerRow = inDim / 64, block = row * blocksPerRow + col / 64;
        int o = block * 18;
        float d = (float)BitConverter.ToHalf(w, o);
        int j = col % 64;
        int q = (w[o + 2 + (j / 32) * 8 + (j % 32) / 4] >> (2 * (j % 4))) & 3;
        return d * (q - 1);
    }

    [CudaFact]
    public void APromptSizeQ2_0Product_MatchesTheHostDecode()
    {
        const int inDim = 512, outDim = 96, rows = 80;
        var rng = new Random(20261108);
        var w = new byte[outDim * (inDim / 64) * 18];
        for (int b = 0; b < w.Length / 18; b++)
        {
            BitConverter.TryWriteBytes(w.AsSpan(b * 18, 2), (System.Half)(0.02f + 0.03f * (float)rng.NextDouble()));
            for (int k = 0; k < 16; k++)
                w[b * 18 + 2 + k] = (byte)rng.Next(256);
        }
        var x = new float[rows * inDim];
        for (int i = 0; i < x.Length; i++)
            x[i] = (float)(rng.NextDouble() * 2 - 1);

        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(allocator.Kernels.DequantsQ2_0, "the loaded kernel module has no Q2_0 dequantizer");
        CudaDriverApi.cuMemAlloc(out IntPtr dW, new UIntPtr((ulong)w.Length + 16)).ThrowOnError();
        CudaDriverApi.cuMemAlloc(out IntPtr dX, new UIntPtr((ulong)x.Length * 4)).ThrowOnError();
        CudaDriverApi.cuMemAlloc(out IntPtr dY, new UIntPtr((ulong)rows * outDim * 4)).ThrowOnError();
        try
        {
            fixed (byte* p = w)
                CudaDriverApi.cuMemcpyHtoD(dW, (IntPtr)p, new UIntPtr((ulong)w.Length)).ThrowOnError();
            fixed (float* p = x)
                CudaDriverApi.cuMemcpyHtoD(dX, (IntPtr)p, new UIntPtr((ulong)x.Length * 4)).ThrowOnError();
            CudaQuantizedOps.RunPrefillF16Gemm(allocator, dW, 42, dX, dY, inDim, outDim, rows);
            CudaDriverApi.cuStreamSynchronize(allocator.Stream.Handle).ThrowOnError();
            var y = new float[rows * outDim];
            fixed (float* p = y)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, dY, new UIntPtr((ulong)y.Length * 4)).ThrowOnError();

            // The GPU rounds weights and activations to F16 and sums in F32: compare against an F16-rounded host product.
            double maxRel = 0;
            for (int r = 0; r < rows; r++)
                for (int o = 0; o < outDim; o++)
                {
                    double s = 0, mag = 0;
                    for (int i = 0; i < inDim; i++)
                    {
                        double a = (double)(float)(System.Half)x[r * inDim + i], b = (double)(float)(System.Half)DecodeQ2_0(w, o, i, inDim);
                        s += a * b;
                        mag += Math.Abs(a * b);
                    }
                    maxRel = Math.Max(maxRel, Math.Abs(y[r * outDim + o] - s) / Math.Max(mag, 1e-6));
                }
            _output.WriteLine($"Q2_0 prompt product: largest error {maxRel:E2} of the row's magnitude");
            Assert.True(maxRel <= 1e-4, $"the Q2_0 F16 route strays {maxRel:E2} from the host decode");
        }
        finally
        {
            CudaDriverApi.cuMemFree(dW);
            CudaDriverApi.cuMemFree(dX);
            CudaDriverApi.cuMemFree(dY);
        }
    }
}
