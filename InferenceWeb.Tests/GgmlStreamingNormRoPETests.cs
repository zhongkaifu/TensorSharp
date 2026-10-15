// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class GgmlStreamingNormRoPETests
{
    [Fact]
    public void InvalidGeometryIsRejectedBeforeNativeAccess()
    {
        Assert.Throws<ArgumentException>(() => Call(null, null, null, null, 8, 256, 1, -1, 256));
        Assert.Throws<ArgumentException>(() => Call(null, null, null, null, int.MaxValue, 256, 1, 0, 256));
        Assert.Throws<ArgumentException>(() => Call(null, null, null, null, 8, 256, 65536, 0, 256));
        Assert.Throws<ArgumentException>(() => Call(null, null, null, null, 8, 256, 2, int.MaxValue - 1, 256));
        Assert.Throws<ArgumentException>(() => Call(null, null, null, null, 8, 256, 1, 0, 255));
        Assert.Throws<ArgumentException>(() => Call(null, null, null, null, 8, 256, 1, 0, 512));
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(256, 256, 1, false)]
    [InlineData(256, 256, 7, false)]
    [InlineData(512, 128, 1, true)]
    [InlineData(512, 128, 7, true)]
    public void MatchesDoubleOracleAndSupportsInPlaceDecodeAndPrefill(int dim, int ropeDims, int tokens, bool factors)
    {
        const int heads = 8, startPos = 517;
        const float eps = 1e-6f, freqBase = 1000000f;
        var context = new GgmlContext([0], GgmlBackendType.Cuda);
        var allocator = new GgmlAllocator(context, 0);
        using var input = new Tensor(allocator, DType.Float32, tokens, heads * dim);
        using var output = new Tensor(allocator, DType.Float32, tokens, heads * dim);
        using var norm = new Tensor(allocator, DType.Float32, dim);
        using var freq = factors ? new Tensor(allocator, DType.Float32, dim / 2) : null;
        float[] values = Enumerable.Range(0, tokens * heads * dim)
            .Select(i => 0.75f * MathF.Sin(i * 0.013f + 0.37f) + 0.17f * MathF.Cos(i * 0.19f)).ToArray();
        float[] weights = Enumerable.Range(0, dim).Select(i => 0.9f + 0.2f * MathF.Sin(i * 0.03f)).ToArray();
        float[] frequencies = Enumerable.Range(0, dim / 2).Select(i => 1.1f + 0.2f * MathF.Cos(i * 0.07f)).ToArray();
        Copy(input, values); Copy(norm, weights);
        if (freq != null) Copy(freq, frequencies);
        // Seed a previous output device copy; the wrapper must invalidate it
        // before subsequent ordinary operations consume its new host result.
        Ops.Fill(output, -3f);
        Call(input, norm, freq, output, heads, dim, tokens, startPos, ropeDims);
        float[] actual = output.GetElementsAsFloat(values.Length);
        for (int token = 0; token < tokens; token++)
        for (int head = 0; head < heads; head++)
        {
            int offset = (token * heads + head) * dim;
            double squareSum = 0;
            for (int d = 0; d < dim; d++) squareSum += (double)values[offset + d] * values[offset + d];
            double scale = 1 / Math.Sqrt(squareSum / dim + eps);
            for (int d = 0; d < dim; d++)
            {
                double expected;
                if (d >= ropeDims) expected = values[offset + d] * scale * weights[d];
                else
                {
                    int pair = d % (ropeDims / 2), partner = pair + ropeDims / 2;
                    double angle = (startPos + token) * Math.Pow(freqBase, -2.0 * pair / ropeDims)
                        / (factors ? frequencies[pair] : 1);
                    double first = values[offset + pair] * scale * weights[pair];
                    double second = values[offset + partner] * scale * weights[partner];
                    expected = d < ropeDims / 2 ? first * Math.Cos(angle) - second * Math.Sin(angle)
                        : first * Math.Sin(angle) + second * Math.Cos(angle);
                }
                Assert.True(float.IsFinite(actual[offset + d]) && Math.Abs(actual[offset + d] - expected) < 2e-4,
                    $"Norm/RoPE mismatch token={token}, head={head}, d={d}: {actual[offset + d]:R} versus {expected:R}");
            }
        }
        using (var doubled = Ops.Mul(null, output, 2f))
            Assert.Equal(actual.Select(v => v * 2f).ToArray(), doubled.GetElementsAsFloat(values.Length));
        Call(input, norm, freq, input, heads, dim, tokens, startPos, ropeDims);
        Assert.Equal(actual, input.GetElementsAsFloat(values.Length));
        using var tooShort = new Tensor(allocator, DType.Float32, ropeDims / 2 - 1);
        Assert.Throws<ArgumentException>(() => Call(input, norm, tooShort, output, heads, dim, tokens, startPos, ropeDims));
    }

    private static void Call(Tensor input, Tensor norm, Tensor factors, Tensor output,
        int heads, int dim, int tokens, int startPos, int ropeDims)
        => GgmlBasicOps.StreamingNormRoPE(input, norm, factors, output, heads, dim, tokens, startPos, ropeDims, 1e-6f, 1000000f);

    private static void Copy(Tensor tensor, float[] values)
    {
        Marshal.Copy(values, 0, tensor.Storage.PtrAtElement(0), values.Length);
        GgmlBasicOps.InvalidateHostBuffer(tensor.Storage.PtrAtElement(0));
    }
}
