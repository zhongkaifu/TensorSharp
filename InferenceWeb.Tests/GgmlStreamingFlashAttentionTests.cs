// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class GgmlStreamingFlashAttentionTests
{
    [Fact]
    public void InvalidDimensionsAreRejectedBeforeAnyNativeAccess()
    {
        Assert.Throws<ArgumentException>(() => GgmlBasicOps.StreamingFlashAttention(null, null, null, null,
            int.MaxValue, 1, 512, 1, 1, 1, 0, 0, 1));
        Assert.Throws<ArgumentException>(() => GgmlBasicOps.StreamingFlashAttention(null, null, null, null,
            8, 2, 256, 7, 13, 13, 7, 0, 1));
        Assert.Throws<ArgumentException>(() => GgmlBasicOps.StreamingFlashAttention(null, null, null, null,
            8, 2, 256, 1, 13, 12, 12, 0, 1));
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(256, 1, false)]
    [InlineData(256, 1, true)]
    [InlineData(256, 7, false)]
    [InlineData(256, 7, true)]
    [InlineData(512, 1, false)]
    [InlineData(512, 1, true)]
    [InlineData(512, 7, false)]
    [InlineData(512, 7, true)]
    public void FlashMatchesDoubleOracleWithGqaStridedCacheAndCausalWindow(int dim, int queryCount, bool halfKv)
    {
        const int heads = 8, kvHeads = 2;
        int validKeys = queryCount == 1 ? 19 : 13, stride = validKeys + 11;
        int window = queryCount == 1 ? 0 : 3, maskStart = validKeys - queryCount;
        var context = new GgmlContext([0], GgmlBackendType.Cuda);
        var allocator = new GgmlAllocator(context, 0);
        using var q = new Tensor(allocator, DType.Float32, heads, queryCount, dim);
        using var k = new Tensor(allocator, halfKv ? DType.Float16 : DType.Float32, kvHeads, stride, dim);
        using var v = new Tensor(allocator, halfKv ? DType.Float16 : DType.Float32, kvHeads, stride, dim);
        using var output = new Tensor(allocator, DType.Float32, queryCount, heads * dim);
        float[] qValues = new float[heads * queryCount * dim];
        float[] kValues = Enumerable.Repeat(float.NaN, kvHeads * stride * dim).ToArray();
        float[] vValues = (float[])kValues.Clone();
        for (int h = 0; h < heads; h++)
        for (int t = 0; t < queryCount; t++)
        for (int d = 0; d < dim; d++)
            qValues[(h * queryCount + t) * dim + d] = 0.07f * MathF.Sin((h + 1) * 0.3f + t * 0.17f + d * 0.11f);
        for (int h = 0; h < kvHeads; h++)
        for (int t = 0; t < validKeys; t++)
        for (int d = 0; d < dim; d++)
        {
            int i = (h * stride + t) * dim + d;
            kValues[i] = 0.13f * MathF.Cos(h * 0.19f + t * 0.31f - d * 0.07f);
            vValues[i] = 0.23f * MathF.Sin(h * 0.43f - t * 0.29f + d * 0.05f);
            if (halfKv) { kValues[i] = (float)(System.Half)kValues[i]; vValues[i] = (float)(System.Half)vValues[i]; }
        }
        Copy(q, qValues); Copy(k, kValues); Copy(v, vValues);
        GgmlBasicOps.StreamingFlashAttention(q, k, v, output, heads, kvHeads, dim,
            queryCount, validKeys, stride, maskStart, window, 1f);
        float[] actual = output.GetElementsAsFloat(queryCount * heads * dim);
        for (int h = 0; h < heads; h++)
        for (int t = 0; t < queryCount; t++)
        {
            int kh = h / (heads / kvHeads), last = maskStart + t;
            int first = window == 0 ? 0 : Math.Max(0, last - window + 1);
            double[] scores = new double[last - first + 1];
            for (int key = first; key <= last; key++)
                for (int d = 0; d < dim; d++)
                    scores[key - first] += (double)qValues[(h * queryCount + t) * dim + d]
                        * kValues[(kh * stride + key) * dim + d];
            double maximum = scores.Max(), denominator = 0;
            for (int key = 0; key < scores.Length; key++) denominator += scores[key] = Math.Exp(scores[key] - maximum);
            for (int d = 0; d < dim; d++)
            {
                double expected = 0;
                for (int key = first; key <= last; key++)
                    expected += scores[key - first] / denominator * vValues[(kh * stride + key) * dim + d];
                float value = actual[(t * heads + h) * dim + d];
                Assert.True(float.IsFinite(value) && Math.Abs(value - expected) <= 2e-3,
                    $"Flash mismatch h={h}, q={t}, d={d}: {value:R} versus {expected:R}");
            }
        }
        // A missing head-stride check could otherwise read beyond the actual cache.
        Assert.Throws<ArgumentException>(() => GgmlBasicOps.StreamingFlashAttention(q, k, v, output,
            heads, kvHeads, dim, queryCount, validKeys, stride + 1, maskStart, window, 1f));
    }

    private static void Copy(Tensor tensor, float[] values)
    {
        byte[] bytes = tensor.ElementType == DType.Float16
            ? MemoryMarshal.AsBytes(values.Select(x => (System.Half)x).ToArray().AsSpan()).ToArray()
            : MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        Marshal.Copy(bytes, 0, tensor.Storage.PtrAtElement(0), bytes.Length);
        GgmlBasicOps.InvalidateHostBuffer(tensor.Storage.PtrAtElement(0));
    }
}
