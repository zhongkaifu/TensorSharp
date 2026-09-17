// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ggml-cuda aborts the whole process ("ggml-cuda/fattn.cu:730: fatal error") when a
// flash-attention node it has no kernel for is computed. The CUDA engine tests used to
// die that way after 82 passes: HunyuanDenseServingTests' synthetic mistral3 model has
// a head size of 16, and its batched prefill chunk reached TSGgml_PagedAttentionForward.
// The native side now asks the backend first and runs such a shape as explicit
// attention (ggml_ops_flash_attn_guard.h). This drives the same entry point directly
// with that head size and checks the result against a managed reference.
using System;
using TensorSharp.GGML;
using Xunit;

namespace InferenceWeb.Tests;

public class FlashAttnUnsupportedShapeTests
{
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void PagedAttention_WithAHeadSizeCudaHasNoKernelFor_FallsBackInsteadOfAborting()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);

        const int headDim = 16, numHeads = 4, numKvHeads = 2, blockSize = 4, numBlocks = 8;
        // Sequence 0: a 4-token prefill chunk at positions 6..9 of a 10-token history.
        // Sequence 1: a single decode token at position 6 of a 7-token history.
        int[] queryStartLoc = { 0, 4, 5 };
        int[] seqLens = { 10, 7 };
        int[] positions = { 6, 7, 8, 9, 6 };
        int[] blockTableFlat = { 5, 2, 7, 1, 3 };
        int[] blockTableOffsets = { 0, 3 };
        const int numTokens = 5;

        var rng = new Random(20260917);
        float[] q = Random(rng, numTokens * numHeads * headDim);
        float[] pagedK = Random(rng, numBlocks * blockSize * numKvHeads * headDim);
        float[] pagedV = Random(rng, numBlocks * blockSize * numKvHeads * headDim);
        float scale = 1.0f / MathF.Sqrt(headDim);

        long fallbacksBefore = GgmlBasicOps.FlashAttnFallbackCount();
        float[] actual = new float[numTokens * numHeads * headDim];
        GgmlBasicOps.PagedAttentionForward(q, pagedK, pagedV, actual, queryStartLoc, seqLens, positions,
            blockTableFlat, blockTableOffsets, numSeqs: 2, numTokens, numHeads, numKvHeads, headDim,
            blockSize, scale);

        Assert.True(GgmlBasicOps.FlashAttnFallbackCount() > fallbacksBefore,
            "head size 16 has no ggml-cuda flash-attention kernel, so this call must have taken the explicit path");

        for (int s = 0; s < 2; s++)
        {
            int[] table = blockTableFlat[blockTableOffsets[s]..];
            for (int t = queryStartLoc[s]; t < queryStartLoc[s + 1]; t++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int hk = h / (numHeads / numKvHeads);
                    int visible = positions[t] + 1;
                    var weights = new double[visible];
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < visible; j++)
                    {
                        int kv = (table[j / blockSize] * blockSize + j % blockSize) * numKvHeads * headDim + hk * headDim;
                        double dot = 0;
                        for (int x = 0; x < headDim; x++)
                            dot += q[t * numHeads * headDim + h * headDim + x] * pagedK[kv + x];
                        weights[j] = dot * scale;
                        max = Math.Max(max, weights[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < visible; j++) { weights[j] = Math.Exp(weights[j] - max); sum += weights[j]; }
                    for (int x = 0; x < headDim; x++)
                    {
                        double expected = 0;
                        for (int j = 0; j < visible; j++)
                        {
                            int kv = (table[j / blockSize] * blockSize + j % blockSize) * numKvHeads * headDim + hk * headDim;
                            expected += weights[j] / sum * pagedV[kv + x];
                        }
                        float got = actual[t * numHeads * headDim + h * headDim + x];
                        Assert.True(Math.Abs(got - expected) < 1e-3,
                            $"seq {s} token {t} head {h} dim {x}: expected {expected}, got {got}");
                    }
                }
            }
        }
    }

    private static float[] Random(Random rng, int count)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 1.2 - 0.6);
        return values;
    }
}
