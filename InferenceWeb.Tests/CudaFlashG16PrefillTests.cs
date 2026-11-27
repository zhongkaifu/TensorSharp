// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The group-16 / head_dim 128 flash prefill kernel PR #282 added for Nemotron's
// attention shape (32 query heads over 2 KV heads), against a CPU reference, through
// both direct-CUDA entry points. The route assertion matters: a PTX without the
// kernel serves the shape with the generic kernel and every output still matches,
// so only the recorded route shows which kernel was tested.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class CudaFlashG16PrefillTests
{
    private const int NumQHeads = 32;
    private const int NumKVHeads = 2;
    private const int HeadDim = 128;

    private static (float[,,] q, float[,,] k, float[,,] v) Inputs(int seqLen, int kvLen)
    {
        var q = new float[NumQHeads, seqLen, HeadDim];
        var k = new float[NumKVHeads, kvLen, HeadDim];
        var v = new float[NumKVHeads, kvLen, HeadDim];
        for (int h = 0; h < NumQHeads; h++)
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < HeadDim; d++)
                    q[h, t, d] = MathF.Sin((h + 1) * 0.13f + (t + 2) * 0.017f + (d + 3) * 0.011f);
        for (int h = 0; h < NumKVHeads; h++)
            for (int t = 0; t < kvLen; t++)
                for (int d = 0; d < HeadDim; d++)
                {
                    k[h, t, d] = MathF.Cos((h + 2) * 0.19f + (t + 1) * 0.023f + (d + 1) * 0.007f);
                    v[h, t, d] = MathF.Sin((h + 3) * 0.11f + (t + 2) * 0.029f + (d + 1) * 0.005f) * 0.5f;
                }
        return (q, k, v);
    }

    /// <summary>Online-softmax reference for one (head, query row); rows outside
    /// <paramref name="rows"/> are skipped (null = all rows).</summary>
    private static float[] Reference(float[,,] q, float[,,] k, float[,,] v, int seqLen, int maskStart, int windowSize,
        float scale, int[] rows = null)
    {
        int group = NumQHeads / NumKVHeads;
        var result = new float[seqLen * NumQHeads * HeadDim];
        foreach (int t in rows ?? Enumerable.Range(0, seqLen).ToArray())
        {
            for (int h = 0; h < NumQHeads; h++)
            {
                int kvh = h / group;
                int visible = maskStart + t;
                int lo = windowSize > 0 ? Math.Max(0, visible - windowSize + 1) : 0;
                double m = double.NegativeInfinity, l = 0;
                var acc = new double[HeadDim];
                for (int kp = lo; kp <= visible; kp++)
                {
                    double dot = 0;
                    for (int d = 0; d < HeadDim; d++) dot += q[h, t, d] * k[kvh, kp, d];
                    dot *= scale;
                    double mNew = Math.Max(m, dot);
                    double alpha = Math.Exp(m - mNew), p = Math.Exp(dot - mNew);
                    l = l * alpha + p;
                    for (int d = 0; d < HeadDim; d++) acc[d] = acc[d] * alpha + p * v[kvh, kp, d];
                    m = mNew;
                }
                for (int d = 0; d < HeadDim; d++)
                    result[(t * NumQHeads + h) * HeadDim + d] = (float)(acc[d] / l);
            }
        }
        return result;
    }

    private static void AssertRowsClose(float[] expected, float[] actual, int[] rows, float tolerance)
    {
        foreach (int t in rows)
        {
            for (int i = 0; i < NumQHeads * HeadDim; i++)
            {
                int at = t * NumQHeads * HeadDim + i;
                Assert.True(MathF.Abs(expected[at] - actual[at]) <= tolerance,
                    $"row {t} element {i}: expected {expected[at]:R}, actual {actual[at]:R}, tolerance {tolerance:R}");
            }
        }
    }

    private static Tensor Cache(CudaAllocator allocator, DType type, float[,,] active, int kvLen, int cacheSize)
    {
        using var source = Tensor.FromArray(allocator, active);
        var cache = new Tensor(allocator, type, NumKVHeads, cacheSize, HeadDim);
        Assert.True(CudaFusedOps.TryCopyHeadFirstToCache(cache, source, 0, kvLen, cacheSize, circular: false));
        return cache;
    }

    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Group16D128_MatchesReference_ThroughBothEntryPoints(bool halfCache)
    {
        // Odd seqLen (not a multiple of the kernel's query or key tiles), a continuation
        // (maskStart 8) and a live cache stride wider than the logical prefix.
        const int seqLen = 67, kvLen = 75, cacheSize = 96, maskStart = kvLen - seqLen;
        float scale = 1f / MathF.Sqrt(HeadDim);
        var (q, k, v) = Inputs(seqLen, kvLen);

        using var allocator = new CudaAllocator();
        Assert.True(allocator.Kernels.FlashG16D128Supports(HeadDim, NumQHeads, NumKVHeads),
            "The loaded CUDA kernels have no ts_gqa_prefill_flash_g16d128_* (a stale committed PTX on a no-nvcc " +
            "build); refresh native/ptx/tensorsharp_kernels.ptx or build with nvcc.");

        DType cacheType = halfCache ? DType.Float16 : DType.Float32;
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kCache = Cache(allocator, cacheType, k, kvLen, cacheSize);
        using var vCache = Cache(allocator, cacheType, v, kvLen, cacheSize);
        float[] expected = Reference(q, k, v, seqLen, maskStart, 0, scale);
        float tolerance = halfCache ? 2e-3f : 5e-5f;
        int[] all = Enumerable.Range(0, seqLen).ToArray();

        using (var actual = new Tensor(allocator, DType.Float32, seqLen, NumQHeads * HeadDim))
        {
            Assert.True(CudaFusedOps.TryGqaPrefillAttention(actual, qTensor, kCache, vCache,
                NumQHeads, NumKVHeads, HeadDim, seqLen, kvLen, maskStart, windowSize: 0, scale, kvStride: cacheSize));
            Assert.Equal("flash-g16d128", CudaFusedOps.LastGqaPrefillRoute);
            AssertRowsClose(expected, actual.GetElementsAsFloat(seqLen * NumQHeads * HeadDim), all, tolerance);
        }
    }

    [CudaFact]
    public void Group16D128_LongPrefill_SampledRowsMatchReference()
    {
        // 1100 keys: many key chunks of the online softmax.
        const int len = 1100, cacheSize = 1152;
        float scale = 1f / MathF.Sqrt(HeadDim);
        var (q, k, v) = Inputs(len, len);

        using var allocator = new CudaAllocator();
        Assert.True(allocator.Kernels.FlashG16D128Supports(HeadDim, NumQHeads, NumKVHeads),
            "The loaded CUDA kernels have no ts_gqa_prefill_flash_g16d128_*.");
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kCache = Cache(allocator, DType.Float16, k, len, cacheSize);
        using var vCache = Cache(allocator, DType.Float16, v, len, cacheSize);
        using var actual = new Tensor(allocator, DType.Float32, len, NumQHeads * HeadDim);
        Assert.True(CudaFusedOps.TryGqaPrefillAttention(actual, qTensor, kCache, vCache,
            NumQHeads, NumKVHeads, HeadDim, len, len, maskStart: 0, windowSize: 0, scale, kvStride: cacheSize));
        Assert.Equal("flash-g16d128", CudaFusedOps.LastGqaPrefillRoute);

        int[] rows = { 0, 1, 63, 64, 549, 1099 };
        AssertRowsClose(Reference(q, k, v, len, 0, 0, scale, rows),
            actual.GetElementsAsFloat(len * NumQHeads * HeadDim), rows, 2e-3f);
    }

    [CudaFact]
    public void Group16D128_SlidingWindow_DeclinesToTheGenericKernel()
    {
        // The g16 kernel is gated to full attention; a window takes the generic kernel,
        // which must still be right.
        const int seqLen = 40, kvLen = 40, cacheSize = 48, windowSize = 17;
        float scale = 1f / MathF.Sqrt(HeadDim);
        var (q, k, v) = Inputs(seqLen, kvLen);

        using var allocator = new CudaAllocator();
        using var qTensor = Tensor.FromArray(allocator, q);
        using var kCache = Cache(allocator, DType.Float32, k, kvLen, cacheSize);
        using var vCache = Cache(allocator, DType.Float32, v, kvLen, cacheSize);
        using var actual = new Tensor(allocator, DType.Float32, seqLen, NumQHeads * HeadDim);
        Assert.True(CudaFusedOps.TryGqaPrefillAttention(actual, qTensor, kCache, vCache,
            NumQHeads, NumKVHeads, HeadDim, seqLen, kvLen, maskStart: 0, windowSize, scale, kvStride: cacheSize));
        Assert.Equal("generic", CudaFusedOps.LastGqaPrefillRoute);
        AssertRowsClose(Reference(q, k, v, seqLen, 0, windowSize, scale),
            actual.GetElementsAsFloat(seqLen * NumQHeads * HeadDim), Enumerable.Range(0, seqLen).ToArray(), 5e-5f);
    }
}
