// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class GgmlStreamingNormResidualTests
{
    [Fact]
    public void InvalidEpsilonCannotReachNativeExecution()
    {
        foreach (float eps in new[] { 0f, -1f, float.NaN, float.PositiveInfinity })
            Assert.Throws<ArgumentOutOfRangeException>(() => GgmlBasicOps.StreamingNormResidual(null, null, null, eps));
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(1)]
    [InlineData(7)]
    public void DecodeAndPrefillMatchDoubleOracleWithoutMutatingInput(int rows)
    {
        const int width = 2560;
        const float eps = 1e-6f;
        var context = new GgmlContext([0], GgmlBackendType.Cuda);
        var allocator = new GgmlAllocator(context, 0);
        using var input = new Tensor(allocator, DType.Float32, rows, width);
        using var residual = new Tensor(allocator, DType.Float32, rows, width);
        using var norm = new Tensor(allocator, DType.Float32, width);
        float[] values = Enumerable.Range(0, width * rows).Select(i => 0.3f * MathF.Sin(i * 0.037f) + 0.2f).ToArray();
        float[] previous = Enumerable.Range(0, width * rows).Select(i => 0.7f * MathF.Cos(i * 0.013f)).ToArray();
        float[] weights = Enumerable.Range(0, width).Select(i => 0.9f + 0.2f * MathF.Sin(i * 0.03f)).ToArray();
        Copy(input, values); Copy(norm, weights); Copy(residual, previous);
        // Populate a cached device copy before the native graph writes host data.
        Ops.Mul(residual, residual, 1f);
        GgmlBasicOps.StreamingNormResidual(residual, input, norm, eps);
        float[] actual = residual.GetElementsAsFloat(previous.Length);
        for (int row = 0; row < rows; row++)
        {
            double squareSum = 0;
            for (int d = 0; d < width; d++) squareSum += (double)values[row * width + d] * values[row * width + d];
            double scale = 1 / Math.Sqrt(squareSum / width + eps);
            for (int d = 0; d < width; d++)
            {
                int i = row * width + d;
                double expected = values[i] * scale * weights[d] + previous[i];
                Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected) < 4e-6,
                    $"Norm/residual mismatch row={row}, d={d}: {actual[i]:R} versus {expected:R}");
            }
        }
        Assert.Equal(values, input.GetElementsAsFloat(values.Length));
        using (var doubled = Ops.Mul(null, residual, 2f))
            Assert.Equal(actual.Select(v => v * 2f).ToArray(), doubled.GetElementsAsFloat(actual.Length));
        using var badNorm = new Tensor(allocator, DType.Float32, width - 1);
        Assert.Throws<ArgumentException>(() => GgmlBasicOps.StreamingNormResidual(residual, input, badNorm, eps));
    }

    private static void Copy(Tensor tensor, float[] values)
    {
        Marshal.Copy(values, 0, tensor.Storage.PtrAtElement(0), values.Length);
        GgmlBasicOps.InvalidateHostBuffer(tensor.Storage.PtrAtElement(0));
    }
}
