// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// GraphFusion.DeltaNetPrep on the CUDA graph kernels: the tiled kernel a prompt chunk takes (a block per head and 32
/// tokens) gives the per-row kernel's conv window, q, k, v, g and beta to the bit, for whole and partial tiles, heads of
/// 128 and 256, and the in-projection's rows read in place at a pitch.
/// </summary>
public sealed class CudaDeltaNetPrepTests
{
    [CudaTheory]
    [InlineData(128, 2, 4, 32)]
    [InlineData(128, 2, 6, 70)]
    [InlineData(256, 1, 2, 45)]
    public void TiledPrepMatchesThePerRowKernel(int d, int kHeads, int vHeads, int rows)
    {
        const int width = 4, convRows = width - 1;
        const float eps = 1e-6f;
        int channels = (2 * kHeads + vHeads) * d, heads = 2 * kHeads + vHeads;
        long qkvPitch = channels + 3 * d, abPitch = vHeads + 5;
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            Assert.True(CudaGraphKernels.TakesTiledDeltaNetPrep(convRows, width, d, rows));
            var rng = new Random(d + rows);
            using Tensor window = Upload(allocator, rng, channels * convRows, 1f), qkv = Upload(allocator, rng, rows * qkvPitch, 2f);
            using Tensor convW = Upload(allocator, rng, channels * width, 0.5f);
            using Tensor alpha = Upload(allocator, rng, rows * abPitch, 4f), beta = Upload(allocator, rng, rows * abPitch, 4f);
            using Tensor dt = Upload(allocator, rng, vHeads, 1f), a = Upload(allocator, rng, vHeads, 1f);
            long[] sizes = { (long)channels * (convRows + rows), (long)rows * kHeads * d, (long)rows * kHeads * d, (long)rows * vHeads * d,
                (long)rows * vHeads, (long)rows * vHeads };
            var outputs = new Tensor[2][];
            for (int run = 0; run < 2; run++)
            {
                outputs[run] = Array.ConvertAll(sizes, n => Zeros(allocator, n));
                Tensor[] o = outputs[run];
                kernels.LaunchDeltaNetPrep(Ptr(window), Ptr(qkv), Ptr(convW), Ptr(alpha), Ptr(beta), Ptr(dt), Ptr(a), Ptr(o[0]), Ptr(o[1]), Ptr(o[2]),
                    Ptr(o[3]), Ptr(o[4]), Ptr(o[5]), channels, convRows, width, d, kHeads, vHeads, rows, eps, qkvPitch, abPitch,
                    allocator.Stream.Handle, tiled: run == 1);
                foreach (Tensor t in o) ((CudaStorage)t.Storage).MarkDeviceModified();
            }
            allocator.Stream.Synchronize();
            string[] names = { "conv window", "q", "k", "v", "g", "beta" };
            for (int i = 0; i < sizes.Length; i++)
            {
                float[] want = outputs[0][i].GetElementsAsFloat((int)sizes[i]), got = outputs[1][i].GetElementsAsFloat((int)sizes[i]);
                for (int e = 0; e < want.Length; e++)
                    if (BitConverter.SingleToInt32Bits(want[e]) != BitConverter.SingleToInt32Bits(got[e]))
                        Assert.Fail($"d {d}, {heads} heads, {rows} rows: {names[i]} element {e} is {got[e]:R} tiled, {want[e]:R} per row");
            }
            foreach (Tensor[] o in outputs)
                foreach (Tensor t in o) t.Dispose();
        }
    }

    private static Tensor Upload(CudaAllocator allocator, Random rng, long count, float scale)
    {
        var values = new float[count];
        for (long i = 0; i < count; i++) values[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        var t = new Tensor(allocator, DType.Float32, count);
        t.SetElementsAsFloat(values);
        ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        return t;
    }

    private static Tensor Zeros(CudaAllocator allocator, long count)
    {
        var t = new Tensor(allocator, DType.Float32, count);
        t.SetElementsAsFloat(new float[count]);
        ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        return t;
    }

    private static IntPtr Ptr(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
}
