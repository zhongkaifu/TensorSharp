// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class GemmaLocalFlashPaddingTests
{
    [GgmlFact(BackendType.GgmlCuda)]
    public void NineteenTokenLocalAttentionWithAndWithoutPaddingMatchesDoubleOracle()
    {
        // Gemma12B local geometry. K/V tail values are deliberately finite and
        // nonzero: allowing any padded key into softmax must fail the oracle.
        const int heads = 16, kvHeads = 8, dim = 256, queries = 19, stride = 256;
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cuda), 0);
        using var q = new Tensor(allocator, DType.Float32, heads, queries, dim);
        using var k = new Tensor(allocator, DType.Float16, kvHeads, stride, dim);
        using var v = new Tensor(allocator, DType.Float16, kvHeads, stride, dim);
        using var output = new Tensor(allocator, DType.Float32, queries, heads * dim);
        var qValues = new float[heads * queries * dim];
        var kValues = Enumerable.Repeat(4f, kvHeads * stride * dim).ToArray();
        var vValues = Enumerable.Repeat(9f, kvHeads * stride * dim).ToArray();
        for (int h = 0; h < heads; h++)
        for (int t = 0; t < queries; t++)
        for (int d = 0; d < dim; d++)
            qValues[(h * queries + t) * dim + d] = 0.2f * MathF.Sin(h * 0.37f + t * 0.21f + d * 0.13f);
        for (int h = 0; h < kvHeads; h++)
        for (int t = 0; t < queries; t++)
        for (int d = 0; d < dim; d++)
        {
            int i = (h * stride + t) * dim + d;
            kValues[i] = (float)(System.Half)(0.3f * MathF.Cos(h * 0.29f - t * 0.27f + d * 0.11f));
            vValues[i] = (float)(System.Half)(0.7f * MathF.Sin(h * 0.41f + t * 0.19f - d * 0.07f));
        }
        Copy(q, qValues); Copy(k, kValues); Copy(v, vValues);

        double[] expected = new double[queries * heads * dim];
        for (int h = 0; h < heads; h++)
        for (int t = 0; t < queries; t++)
        {
            int kh = h / (heads / kvHeads);
            var scores = new double[t + 1];
            for (int key = 0; key <= t; key++)
                for (int d = 0; d < dim; d++)
                    scores[key] += (double)qValues[(h * queries + t) * dim + d]
                        * kValues[(kh * stride + key) * dim + d];
            double maximum = scores.Max(), denominator = 0;
            for (int key = 0; key <= t; key++) denominator += scores[key] = Math.Exp(scores[key] - maximum);
            for (int d = 0; d < dim; d++)
                for (int key = 0; key <= t; key++)
                    expected[(t * heads + h) * dim + d] += scores[key] / denominator
                        * vValues[(kh * stride + key) * dim + d];
        }

        foreach (int physicalKeys in new[] { queries, stride })
        {
            // maskStart=0 makes both representations causal over the same
            // positions0..18. Keys19..255 are masked even in the padded call.
            GgmlBasicOps.StreamingFlashAttention(q, k, v, output, heads, kvHeads, dim,
                queries, physicalKeys, stride, 0, 0, 1f);
            float[] actual = output.GetElementsAsFloat(expected.Length);
            double squaredError = 0, squaredReference = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.True(float.IsFinite(actual[i]), $"Nonfinite result at {i}, KV={physicalKeys}.");
                double error = actual[i] - expected[i];
                Assert.True(Math.Abs(error) <= 2e-3, $"Oracle mismatch at {i}, KV={physicalKeys}: {error:R}.");
                squaredError += error * error;
                squaredReference += expected[i] * expected[i];
            }
            // Both implementations must be accurate independently; agreement
            // with each other alone would not catch a shared mask or layout bug.
            double relativeL2 = Math.Sqrt(squaredError / squaredReference);
            Assert.True(relativeL2 <= 1e-3, $"KV={physicalKeys}: relative L2={relativeL2:R}.");
        }
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
