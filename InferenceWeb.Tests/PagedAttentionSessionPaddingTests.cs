// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// TSGgml_PagedAttentionForward caches one graph per (num_q, power-of-two K/V bucket,
/// heads) and uploads only the leading seq_len rows of K/V into the bucket. The rows
/// past seq_len are masked, but the session assumed its backend buffer started zeroed,
/// which a fresh CUDA buffer does not promise, and the CUDA flash-attention kernels
/// compute q.k for masked keys before adding the -inf mask. The session buffer is now
/// cleared when built.
///
/// This test poisons a session's memory, evicts it and rebuilds the same shape, then
/// checks the native result against the managed reference. It is a guard: on the A40
/// test box the allocator did not hand the poisoned memory back, so it also passes
/// against a native library without the clear.
///
/// Runs on whichever backend TS_TEST_GGML_BACKEND selects (default cpu).
/// </summary>
public class PagedAttentionSessionPaddingTests
{
    private static GgmlBackendType ConfiguredBackend() =>
        (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu").Trim().ToLowerInvariant() switch
        {
            "cuda" => GgmlBackendType.Cuda,
            "metal" => GgmlBackendType.Metal,
            "vulkan" => GgmlBackendType.Vulkan,
            _ => GgmlBackendType.Cpu,
        };

    private const int NumHeads = 8;
    private const int NumKvHeads = 1;
    private const int HeadDim = 128;
    private const int BlockSize = 64;

    [Fact]
    public void RebuiltSession_IgnoresLeftoverMemoryPastSequenceLength()
    {
        _ = new GgmlContext(new[] { 0 }, ConfiguredBackend());
        float scale = 1f / MathF.Sqrt(HeadDim);

        for (int round = 0; round < 3; round++)
        {
            // Poison: a decode call whose K/V fill the whole 64-row bucket with values
            // that overflow q.k.
            Run(seqLen: 64, numQ: 1, constantKey: float.MaxValue / 4, scale);

            // Evict that session (16 slots) so its buffer is freed, then rebuild the
            // same shape; the fresh buffer may land on the poisoned memory.
            for (int q = 2; q <= 18; q++)
                Run(seqLen: 64, numQ: q, constantKey: 0.01f, scale);

            float[] got = Run(seqLen: 61, numQ: 1, constantKey: null, scale, out float[] expected);
            Assert.All(got, v => Assert.True(float.IsFinite(v), "paged attention returned a non-finite value"));
            for (int i = 0; i < got.Length; i++)
                Assert.True(Math.Abs(got[i] - expected[i]) < 1e-3f, $"output {i}: native {got[i]} vs managed {expected[i]}");
        }
    }

    private static float[] Run(int seqLen, int numQ, float? constantKey, float scale)
        => Run(seqLen, numQ, constantKey, scale, out _);

    /// <summary>One sequence of <paramref name="seqLen"/> tokens whose last
    /// <paramref name="numQ"/> tokens query it; keys are random unless
    /// <paramref name="constantKey"/> is given.</summary>
    private static float[] Run(int seqLen, int numQ, float? constantKey, float scale, out float[] reference)
    {
        int kvStride = NumKvHeads * HeadDim;
        int qStride = NumHeads * HeadDim;
        int blocks = (seqLen + BlockSize - 1) / BlockSize;
        var rng = new Random(seqLen * 31 + numQ);
        float[] q = Enumerable.Range(0, numQ * qStride).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] k = new float[blocks * BlockSize * kvStride];
        float[] v = new float[k.Length];
        for (int i = 0; i < seqLen * kvStride; i++)
        {
            k[i] = constantKey ?? (float)(rng.NextDouble() - 0.5);
            v[i] = (float)(rng.NextDouble() - 0.5);
        }
        int[] positions = Enumerable.Range(seqLen - numQ, numQ).ToArray();
        int[] queryStartLoc = { 0, numQ };
        int[] seqLens = { seqLen };
        int[] table = Enumerable.Range(0, blocks).ToArray();

        float[] output = new float[numQ * qStride];
        GgmlBasicOps.PagedAttentionForward(q, k, v, output, queryStartLoc, seqLens, positions,
            table, new[] { 0, blocks }, 1, numQ, NumHeads, NumKvHeads, HeadDim, BlockSize, scale);

        reference = new float[output.Length];
        ManagedPagedAttention.Forward(q, k, v, reference, numQ, NumHeads, NumKvHeads, HeadDim, BlockSize,
            queryStartLoc, seqLens, positions, new[] { table }, 1, scale);
        return output;
    }
}
