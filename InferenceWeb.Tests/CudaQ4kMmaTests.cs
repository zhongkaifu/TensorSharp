// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The drafter's Q4_K product on the int8 tensor cores (CudaGraphKernels.LaunchGemmQ4k, the direct engines'
/// ts_q35_gemm_q4k): the dequantized weights times the q8_1 activations, up to the order of float additions.
/// </summary>
public sealed class CudaQ4kMmaTests
{
    [CudaTheory]
    [InlineData(8, 512, 5120)]    // a draft block's rows; K split across blocks (added atomically)
    [InlineData(3, 80, 1024)]     // ragged rows, a partial row tile
    [InlineData(40, 192, 768)]    // past the 32-row variant
    public void TheTensorCoreProductIsTheDequantizedOne(int rows, int outDim, int k)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            Assert.True(kernels.TakesQ4kMma, $"this image has no sm_80 bodies (arch {kernels.ImageArch})");
            var rng = new Random(41);
            byte[] w = RandomQ4k(rng, outDim, k);
            float[] x = Enumerable.Range(0, rows * k).Select(_ => (float)((rng.NextDouble() * 2 - 1) * 3)).ToArray();
            using var wt = new Tensor(allocator, DType.UInt8, w.Length);
            using var xt = Tensor.FromArray(allocator, x);
            using var xq = new Tensor(allocator, DType.UInt8, (long)rows * k);
            using var xd = new Tensor(allocator, DType.Float32, (long)rows * (k / 32));
            using var y = new Tensor(allocator, DType.Float32, (long)rows * outDim);
            foreach (Tensor t in new[] { wt, xt, xq, xd, y }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(0);
            Upload(P(wt), w);
            allocator.Kernels.LaunchQuantizeQ81SplitRows(P(xt), P(xq), P(xd), k, rows, allocator.Stream.Handle);
            kernels.LaunchGemmQ4k(P(wt), k / 256 * 144, outDim, k, P(xq), P(xd), rows, P(y), allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            ((CudaStorage)y.Storage).MarkDeviceModified();
            ((CudaStorage)xd.Storage).MarkDeviceModified();
            float[] got = y.GetElementsAsFloat(rows * outDim);
            byte[] q = Download(P(xq), rows * k);
            float[] d = xd.GetElementsAsFloat(rows * (k / 32));

            float[] row = new float[k];
            for (int o = 0; o < outDim; o++)
            {
                Dequantize(w, o, k, row);
                for (int r = 0; r < rows; r++)
                {
                    double sum = 0, scale = 0;
                    for (int i = 0; i < k; i++)
                    {
                        double xi = (sbyte)q[(long)r * k + i] * (double)d[r * (k / 32) + i / 32];
                        sum += row[i] * xi;
                        scale += Math.Abs(row[i] * xi);
                    }
                    Assert.True(Math.Abs(got[r * outDim + o] - sum) <= 1e-5 * scale + 1e-6,
                        $"row {r} output {o}: {got[r * outDim + o]} against {sum} (scale {scale})");
                }
            }
        }
    }

    private static byte[] RandomQ4k(Random rng, int rows, int k)
    {
        var raw = new byte[rows * (k / 256) * 144];
        rng.NextBytes(raw);
        for (int b = 0; b < raw.Length; b += 144)
        {
            WriteHalf(raw, b, (float)(0.002 + rng.NextDouble() * 0.01));
            WriteHalf(raw, b + 2, (float)(0.001 + rng.NextDouble() * 0.005));
        }
        return raw;
    }

    private static void WriteHalf(byte[] dst, int offset, float value) => BitConverter.TryWriteBytes(dst.AsSpan(offset, 2), (System.Half)value);

    private static unsafe void Upload(IntPtr device, byte[] bytes)
    {
        fixed (byte* p = bytes)
            Assert.Equal(0, TensorSharp.Cuda.Interop.CudaDriverApi.cuMemcpyHtoD(device, (IntPtr)p, (UIntPtr)(ulong)bytes.Length));
    }

    private static unsafe byte[] Download(IntPtr device, int count)
    {
        var bytes = new byte[count];
        fixed (byte* p = bytes)
            Assert.Equal(0, TensorSharp.Cuda.Interop.CudaDriverApi.cuMemcpyDtoH((IntPtr)p, device, (UIntPtr)(ulong)count));
        return bytes;
    }

    /// <summary>One row of a Q4_K matrix as ggml decodes it.</summary>
    private static void Dequantize(byte[] w, int row, int k, float[] dst)
    {
        int blocks = k / 256;
        for (int b = 0; b < blocks; b++)
        {
            int o = (row * blocks + b) * 144;
            float d = (float)BitConverter.ToHalf(w, o), dmin = (float)BitConverter.ToHalf(w, o + 2);
            for (int j = 0; j < 8; j++)
            {
                int sc, m;
                if (j < 4)
                {
                    sc = w[o + 4 + j] & 63;
                    m = w[o + 4 + j + 4] & 63;
                }
                else
                {
                    sc = (w[o + 4 + j + 4] & 0xF) | ((w[o + 4 + j - 4] >> 6) << 4);
                    m = (w[o + 4 + j + 4] >> 4) | ((w[o + 4 + j] >> 6) << 4);
                }
                for (int l = 0; l < 32; l++)
                {
                    int c = j / 2, nib = j % 2;
                    int byteValue = w[o + 16 + 32 * c + l];
                    int qv = nib == 0 ? byteValue & 0xF : byteValue >> 4;
                    dst[b * 256 + 32 * j + l] = d * sc * qv - dmin * m;
                }
            }
        }
    }
}
