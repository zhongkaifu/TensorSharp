// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// The host-array paged attention (TSGgml_PagedAttentionForward) caches one
/// graph + backend buffer per shape bucket and pads K/V up to that bucket. It
/// assumed a freshly allocated buffer is zero, so the padding [seq_len, bucket)
/// was never written. Backend buffers are not zeroed: a new session built on
/// memory a freed buffer just used inherits whatever was there, and CUDA flash
/// attention still forms q.k for the -inf-masked padded keys, so NaN garbage
/// turns the whole softmax row into NaN. On Nemotron 3.5 (ggml_cuda) that made
/// the first attention layer of a 4-sequence batched prefill return NaN for
/// every row after a long prompt had come and gone, and the MoE router then
/// threw IndexOutOfRangeException.
///
/// The test poisons every cached session with NaN K/V over the whole bucket,
/// then keeps building new sessions of the same size (a different scale is a
/// different cache key, not a different size) so they are allocated on the
/// memory the evicted poisoned sessions release, and checks each new session's
/// output against a reference. Runs on whichever backend TS_TEST_GGML_BACKEND
/// selects (default cpu). It only fails on a backend whose allocator reuses
/// freed memory AND whose kernel reads masked keys (ggml_cuda does both; the CPU
/// kernel skips -inf keys and macOS hands back zeroed pages).
/// </summary>
public class PagedAttentionSessionBufferTests
{
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 64;
    private const int BlockSize = 16;
    private const int Bucket = 64;       // the cache's smallest padded-KV bucket
    private const int CacheSize = 16;    // kPagedAttnCacheSize

    private static GgmlBackendType ConfiguredBackend() =>
        (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu").Trim().ToLowerInvariant() switch
        {
            "cuda" => GgmlBackendType.Cuda,
            "metal" => GgmlBackendType.Metal,
            "vulkan" => GgmlBackendType.Vulkan,
            _ => GgmlBackendType.Cpu,
        };

    [Fact]
    public void NewSessionOnReusedMemory_IgnoresStalePaddedKeys()
    {
        _ = new GgmlContext(new[] { 0 }, ConfiguredBackend());

        // Fill every cache slot with sessions whose K/V cover the whole bucket
        // with NaN (a full-length call, so all of [0, bucket) is written).
        int scaleIndex = 0;
        for (int i = 0; i < CacheSize; i++)
        {
            RunCase(numQ: 1, seqLen: Bucket, NextScale(ref scaleIndex), poison: true, out _, out _);
        }

        // Each new shape key evicts the least recently used poisoned session and
        // allocates a same-sized buffer, typically on the memory just freed. The
        // padded keys [seqLen, bucket) of the new session were never written.
        var failures = new List<string>();
        for (int i = 0; i < CacheSize; i++)
        {
            int numQ = 26;
            RunCase(numQ, seqLen: numQ, NextScale(ref scaleIndex), poison: false, out float[] actual, out float[] expected);

            int nonFinite = actual.Count(v => !float.IsFinite(v));
            float maxErr = 0;
            for (int j = 0; j < actual.Length; j++)
                if (float.IsFinite(actual[j]))
                    maxErr = Math.Max(maxErr, Math.Abs(actual[j] - expected[j]));
            if (nonFinite > 0 || maxErr > 2e-3f)
                failures.Add($"session {i}: {nonFinite}/{actual.Length} non-finite outputs, max finite error {maxErr}");
        }

        Assert.True(failures.Count == 0,
            "paged attention read stale padded K/V from a reused session buffer:\n" + string.Join("\n", failures));
    }

    private static float NextScale(ref int index) => 1.0f / MathF.Sqrt(HeadDim) * (1.0f + 0.01f * index++);

    private static void RunCase(int numQ, int seqLen, float scale, bool poison, out float[] actual, out float[] expected)
    {
        int kvDim = NumKvHeads * HeadDim;
        int qDim = NumHeads * HeadDim;
        int numBlocks = (seqLen + BlockSize - 1) / BlockSize;

        var rnd = new Random(seqLen * 7919 + numQ);
        float[] q = new float[numQ * qDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rnd.NextDouble() * 2 - 1);
        float[] k = new float[numBlocks * BlockSize * kvDim];
        float[] v = new float[numBlocks * BlockSize * kvDim];
        for (int i = 0; i < k.Length; i++)
        {
            k[i] = poison ? float.NaN : (float)(rnd.NextDouble() * 2 - 1);
            v[i] = poison ? float.NaN : (float)(rnd.NextDouble() * 2 - 1);
        }

        int[] blockTable = Enumerable.Range(0, numBlocks).ToArray();
        int firstPos = seqLen - numQ;
        int[] positions = Enumerable.Range(firstPos, numQ).ToArray();
        actual = new float[numQ * qDim];

        GgmlBasicOps.PagedAttentionForward(
            q, k, v, actual,
            queryStartLoc: new[] { 0, numQ },
            seqLens: new[] { seqLen },
            positions: positions,
            blockTableFlat: blockTable,
            blockTableOffsets: new[] { 0 },
            numSeqs: 1, numTokens: numQ, numHeads: NumHeads, numKvHeads: NumKvHeads,
            headDim: HeadDim, blockSize: BlockSize, scale: scale);

        expected = poison ? actual : Reference(q, k, v, numQ, seqLen, firstPos, scale);
    }

    private static float[] Reference(float[] q, float[] k, float[] v, int numQ, int seqLen, int firstPos, float scale)
    {
        int group = NumHeads / NumKvHeads;
        var result = new float[numQ * NumHeads * HeadDim];
        var scores = new double[seqLen];
        for (int t = 0; t < numQ; t++)
        {
            int visible = Math.Min(firstPos + t + 1, seqLen);
            for (int h = 0; h < NumHeads; h++)
            {
                int kvh = h / group;
                int qOff = (t * NumHeads + h) * HeadDim;
                double max = double.NegativeInfinity;
                for (int p = 0; p < visible; p++)
                {
                    int kOff = (p * NumKvHeads + kvh) * HeadDim;
                    double dot = 0;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    scores[p] = dot * scale;
                    max = Math.Max(max, scores[p]);
                }
                double sum = 0;
                for (int p = 0; p < visible; p++) sum += Math.Exp(scores[p] - max);
                for (int p = 0; p < visible; p++)
                {
                    double w = Math.Exp(scores[p] - max) / sum;
                    int vOff = (p * NumKvHeads + kvh) * HeadDim;
                    for (int d = 0; d < HeadDim; d++) result[qOff + d] += (float)(w * v[vOff + d]);
                }
            }
        }
        return result;
    }
}
