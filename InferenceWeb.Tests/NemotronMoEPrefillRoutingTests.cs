// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

namespace InferenceWeb.Tests;

/// <summary>
/// The multi-token MoE router shared by Nemotron's batched-by-expert and fused
/// prefill paths, driven with a synthetic 4-sequence batch (Nemotron 3.5's 128
/// experts, top-6). A row whose router logits are NaN has no top-k, and the old
/// inline router indexed probs[-1]: that is the IndexOutOfRangeException the
/// 2026-09-16 campaign hit on a 4-sequence batched prefill, failing every
/// request in the step.
/// </summary>
public sealed unsafe class NemotronMoEPrefillRoutingTests
{
    private const int NumExperts = 128;
    private const int NUsed = 6;
    private const int SeqsInBatch = 4;
    private const int TokensPerSeq = 26;
    private const float Scale = 2.5f;

    private static (float[] router, float[] bias) SyntheticBatch(int tokens)
    {
        var rnd = new Random(3);
        var router = new float[tokens * NumExperts];
        for (int i = 0; i < router.Length; i++) router[i] = (float)(rnd.NextDouble() * 8 - 4);
        var bias = new float[NumExperts];
        for (int e = 0; e < NumExperts; e++) bias[e] = (float)(rnd.NextDouble() * 0.2 - 0.1);
        return (router, bias);
    }

    private static void Route(float[] router, float[] bias, int tokens, int[] selected, float[] weights)
    {
        fixed (float* rp = router)
        fixed (float* bp = bias)
        {
            NemotronModel.RouteMoEPrefillTokens(
                rp, bp, tokens, NumExperts, NUsed, normalize: true, Scale, layer: 6,
                new float[NumExperts], new float[NumExperts], new int[NumExperts], selected, weights);
        }
    }

    [Fact]
    public void FourSequenceBatch_RoutesEveryTokenToItsTopExperts_AndGroupsByExpert()
    {
        int tokens = SeqsInBatch * TokensPerSeq;
        var (router, bias) = SyntheticBatch(tokens);
        var selected = new int[tokens * NUsed];
        var weights = new float[tokens * NUsed];

        Route(router, bias, tokens, selected, weights);

        for (int s = 0; s < tokens; s++)
        {
            var probs = Enumerable.Range(0, NumExperts)
                .Select(e => 1.0 / (1.0 + Math.Exp(-router[s * NumExperts + e]))).ToArray();
            var want = Enumerable.Range(0, NumExperts)
                .OrderByDescending(e => probs[e] + bias[e]).Take(NUsed).ToHashSet();
            var got = selected.Skip(s * NUsed).Take(NUsed).ToArray();
            Assert.True(want.SetEquals(got), $"row {s}: experts {string.Join(",", got)} are not the top-{NUsed}");

            double sum = got.Sum(e => probs[e]);
            for (int k = 0; k < NUsed; k++)
                Assert.Equal(probs[got[k]] / sum * Scale, weights[s * NUsed + k], 4);
        }

        var counts = new int[NumExperts];
        var offsets = new int[NumExperts + 1];
        var rows = new int[tokens * NUsed];
        var routedWeights = new float[tokens * NUsed];
        NemotronModel.GroupMoERoutesByExpert(
            selected, weights, tokens, NUsed, NumExperts, counts, offsets, new int[NumExperts], rows, routedWeights);

        Assert.Equal(tokens * NUsed, offsets[NumExperts]);
        var seen = new HashSet<(int row, int expert)>();
        for (int e = 0; e < NumExperts; e++)
        {
            for (int i = offsets[e]; i < offsets[e + 1]; i++)
            {
                int row = rows[i];
                int k = Array.IndexOf(selected, e, row * NUsed, NUsed) - row * NUsed;
                Assert.InRange(k, 0, NUsed - 1);
                Assert.Equal(weights[row * NUsed + k], routedWeights[i]);
                Assert.True(seen.Add((row, e)));
                if (i > offsets[e]) Assert.True(rows[i - 1] < row, "rows inside an expert group stay in token order");
            }
        }
        Assert.Equal(tokens * NUsed, seen.Count);
    }

    [Fact]
    public void NonFiniteRouterRow_FailsWithAClearError_NotAnIndexOutOfRange()
    {
        int tokens = SeqsInBatch * TokensPerSeq;
        var (router, bias) = SyntheticBatch(tokens);
        const int badRow = 2 * TokensPerSeq + 5;  // inside the third sequence
        for (int e = 0; e < NumExperts; e++) router[badRow * NumExperts + e] = float.NaN;

        var ex = Assert.Throws<InvalidOperationException>(
            () => Route(router, bias, tokens, new int[tokens * NUsed], new float[tokens * NUsed]));

        Assert.Contains($"token row {badRow}", ex.Message);
        Assert.Contains("non-finite router logits", ex.Message);
        Assert.Contains("layer 6", ex.Message);
    }

    [Fact]
    public void PartlyNonFiniteRow_StillRoutesToFiniteExperts()
    {
        int tokens = 8;
        var (router, bias) = SyntheticBatch(tokens);
        for (int e = 0; e < NumExperts - NUsed; e++) router[3 * NumExperts + e] = float.NaN;
        var selected = new int[tokens * NUsed];

        Route(router, bias, tokens, selected, new float[tokens * NUsed]);

        Assert.All(selected.Skip(3 * NUsed).Take(NUsed), e => Assert.InRange(e, NumExperts - NUsed, NumExperts - 1));
    }
}
