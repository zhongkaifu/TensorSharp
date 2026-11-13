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
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The tensor-core flash prefill attention (ts_flash_prefill_f16_d128 / _d256) against a double-precision host
/// reference over the same F16 cache: causal rows that are the last nq of the window, GQA groups folded into the row
/// tiles, keys below a floor masked, and split-K partials for a few rows over a long window. F16 operands and short
/// F16 accumulation runs keep it within a few 1e-3.
/// </summary>
public sealed class CudaFlashPrefillAttentionTests
{
    private readonly ITestOutputHelper _output;

    public CudaFlashPrefillAttentionTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(256, 24, 4, 130, 130, 0)]   // a prompt chunk: rows past one 128-row tile, no history
    [InlineData(256, 24, 4, 17, 300, 0)]    // a chunk after a history
    [InlineData(256, 8, 1, 3, 2100, 0)]     // a verify window over a long history: split over the keys
    [InlineData(128, 16, 2, 40, 90, 33)]    // 128-wide heads, keys below a floor masked
    public void RowsMatchTheReference(int d, int heads, int kvHeads, int nq, int len, int floor)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (flash == null || !flash.RunsAttention(d))
        {
            _output.WriteLine("no flash prefill attention for this GPU");
            return;
        }
        int cap = len + 37, group = heads / kvHeads;
        var rng = new Random(3);
        float[] qv = new float[heads * nq * d];
        for (int i = 0; i < qv.Length; i++) qv[i] = (float)(rng.NextDouble() * 2 - 1);
        // Values an F16 cache holds exactly (rounded on the host, converted on the device without loss).
        var kh = new float[kvHeads * cap * d];
        var vh = new float[kvHeads * cap * d];
        for (int i = 0; i < kh.Length; i++)
        {
            kh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
            vh[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
        }
        float scale = 1f / MathF.Sqrt(d);

        using var q = Tensor.FromArray(allocator, qv);
        using var k32 = Tensor.FromArray(allocator, kh);
        using var v32 = Tensor.FromArray(allocator, vh);
        using var k = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
        using var v = new Tensor(allocator, DType.Float16, kvHeads * cap * d);
        foreach (Tensor t in new[] { k32, v32, k, v })
            ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        allocator.Kernels.LaunchConvertF32F16(((CudaStorage)k32.Storage).DevicePtrAtElement(0), ((CudaStorage)k.Storage).DevicePtrAtElement(0),
            kh.Length, allocator.Stream.Handle);
        allocator.Kernels.LaunchConvertF32F16(((CudaStorage)v32.Storage).DevicePtrAtElement(0), ((CudaStorage)v.Storage).DevicePtrAtElement(0),
            vh.Length, allocator.Stream.Handle);
        using var output = new Tensor(allocator, DType.Float32, nq * heads * d);
        int splits = flash.Splits(nq * group, kvHeads, len);
        using Tensor partial = splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, d, splits)) : null;
        foreach (Tensor t in new[] { q, output })
            ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
        allocator.Context.MakeCurrent();
        flash.LaunchAttention(P(q), P(k), P(v), cap, nq, len, floor, heads, kvHeads, d, scale, splits,
            partial == null ? IntPtr.Zero : P(partial), P(output), allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        ((CudaStorage)output.Storage).MarkDeviceModified();
        float[] got = output.GetElementsAsFloat(nq * heads * d);

        double num = 0, den = 0;
        var w = new double[len];
        for (int h = 0; h < heads; h++)
        {
            int kvh = h / group;
            for (int t = 0; t < nq; t++)
            {
                int end = len - nq + t;   // last visible key
                double mx = double.NegativeInfinity;
                for (int j = floor; j <= end; j++)
                {
                    double dot = 0;
                    for (int x = 0; x < d; x++) dot += qv[(h * nq + t) * d + x] * (double)kh[(kvh * cap + j) * d + x];
                    w[j] = dot * scale;
                    mx = Math.Max(mx, w[j]);
                }
                double sum = 0;
                for (int j = floor; j <= end; j++) { w[j] = Math.Exp(w[j] - mx); sum += w[j]; }
                for (int x = 0; x < d; x++)
                {
                    double want = 0;
                    for (int j = floor; j <= end; j++) want += w[j] * (double)vh[(kvh * cap + j) * d + x];
                    want /= sum;
                    double diff = got[(t * heads + h) * d + x] - want;
                    num += diff * diff;
                    den += want * want;
                }
            }
        }
        double rel = Math.Sqrt(num / den);
        _output.WriteLine($"d {d}, {heads}/{kvHeads} heads, {nq} rows over {len} keys (floor {floor}), {splits} splits: relative L2 {rel:E2}");
        Assert.True(rel <= 5e-3, $"relative L2 {rel:E2}");
    }
}
