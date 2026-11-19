// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Reflection;

namespace InferenceWeb.Tests;

/// <summary>
/// A batched verify draws each row where the model left it (<c>TokenSampler.SampleRow</c>), where it had copied every row
/// into a vocab-sized array first, and the top-k scan passes over whole vector blocks no token of which beats the heap's
/// root. Both must draw what <see cref="TokenSampler.Sample"/> over a copy draws, token for token.
/// </summary>
public sealed class TokenSamplerRowDrawTests
{
    private const int EndToken = 7;

    private static IEnumerable<(string Name, Func<int, SamplingConfig> Config)> Configs()
    {
        yield return ("greedy", seed => new SamplingConfig { Temperature = 0f, RepetitionPenalty = 1f, Seed = seed });
        yield return ("greedy, repeat penalty", seed => new SamplingConfig { Temperature = 0f, RepetitionPenalty = 1.1f, PenaltyLastN = 64, Seed = seed });
        yield return ("greedy, presence and frequency", seed => new SamplingConfig
        {
            Temperature = 0f, RepetitionPenalty = 1f, PresencePenalty = 0.5f, FrequencyPenalty = 0.25f, Seed = seed,
        });
        yield return ("greedy, close masked outside its channel", seed => new SamplingConfig
        {
            Temperature = 0f, RepetitionPenalty = 1f, Seed = seed,
            ThinkingBudget = new ThinkingTokenBudget(int.MaxValue, EndToken, openTokenId: 5, openAtStart: false, suppressUnopenedEnd: true),
        });
        yield return ("T 0.3, top-k 20, top-p 0.8, repeat penalty", seed => new SamplingConfig
        {
            Temperature = 0.3f, TopK = 20, TopP = 0.8f, MinP = 0f, RepetitionPenalty = 1.1f, Seed = seed,
        });
        yield return ("T 0.8, top-k 40, top-p 0.9, min-p 0.05", seed => new SamplingConfig
        {
            Temperature = 0.8f, TopK = 40, TopP = 0.9f, MinP = 0.05f, RepetitionPenalty = 1f, Seed = seed,
        });
        yield return ("T 1, no top-k", seed => new SamplingConfig { Temperature = 1f, TopK = 0, TopP = 1f, MinP = 0f, RepetitionPenalty = 1f, Seed = seed });
    }

    /// <summary>Logits with ties, -inf runs and signed zeros, so the scans' tie rules are exercised.</summary>
    private static float[] Row(Random rng, int vocab)
    {
        var row = new float[vocab];
        for (int v = 0; v < vocab; v++)
        {
            int kind = rng.Next(40);
            row[v] = kind switch
            {
                0 => float.NegativeInfinity,
                1 => 0f,
                2 => -0f,
                _ => MathF.Round((float)(rng.NextDouble() * 12 - 6), 1),   // tenths: many exact ties
            };
        }
        // A few clear leaders, sometimes the masked close among them.
        for (int i = 0; i < 3; i++) row[rng.Next(vocab)] = 8f + i;
        if (rng.Next(3) == 0) row[EndToken] = 12f;
        return row;
    }

    [Fact]
    public void SampleRowDrawsWhatSampleDrawsOverACopy()
    {
        const int vocab = 4099, rows = 6;
        foreach (var (name, config) in Configs())
        {
            for (int seed = 0; seed < 8; seed++)
            {
                var copySampler = new TokenSampler(config(seed));
                var rowSampler = new TokenSampler(config(seed));
                var rng = new Random(1000 + seed);
                var history = new List<int> { 1, 2, 3 };
                for (int step = 0; step < 40; step++)
                {
                    // The verify's buffer: rows of logits back to back, the drawn row somewhere inside.
                    var buffer = new float[rows * vocab];
                    for (int r = 0; r < rows; r++) Row(rng, vocab).CopyTo(buffer, r * vocab);
                    int at = rng.Next(rows);
                    var before = (float[])buffer.Clone();
                    float[] copy = buffer.AsSpan(at * vocab, vocab).ToArray();

                    int expected = copySampler.Sample(copy, history);
                    int drawn = rowSampler.SampleRow(buffer.AsSpan(at * vocab, vocab), history);
                    Assert.True(expected == drawn, $"{name}, seed {seed}, step {step}: {drawn} drawn in place, {expected} over a copy");
                    // Only the drawn row may change, and only as Sample changes its own array.
                    for (int r = 0; r < rows; r++)
                        if (r != at)
                            Assert.True(before.AsSpan(r * vocab, vocab).SequenceEqual(buffer.AsSpan(r * vocab, vocab)), $"{name}: row {r} changed");
                    Assert.True(copy.AsSpan().SequenceEqual(buffer.AsSpan(at * vocab, vocab)), $"{name}: the row changed unlike the copy");
                    history.Add(expected);
                    if (step == 10) history.Add(5);   // open the channel partway (the masked close's policy)
                }
            }
        }
    }

    private static readonly MethodInfo ApplyTopK = typeof(TokenSampler).GetMethod("ApplyTopK", BindingFlags.Instance | BindingFlags.NonPublic)!;

    /// <summary>The scan before the block skip, verbatim: every token tested against the root.</summary>
    private static int[] OracleTopK(float[] scores, int k)
    {
        static bool Weaker(float[] s, int a, int b)
        {
            float sa = s[a], sb = s[b];
            if (sa != sb) return sa < sb;
            return a > b;
        }
        static void SiftDown(int[] heap, float[] s, int root, int count)
        {
            while (true)
            {
                int child = 2 * root + 1;
                if (child >= count) return;
                if (child + 1 < count && Weaker(s, heap[child + 1], heap[child])) child++;
                if (!Weaker(s, heap[child], heap[root])) return;
                (heap[root], heap[child]) = (heap[child], heap[root]);
                root = child;
            }
        }
        int n = scores.Length;
        var heap = new int[k];
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if (count < k)
            {
                heap[count++] = i;
                if (count == k)
                    for (int j = (k >> 1) - 1; j >= 0; j--) SiftDown(heap, scores, j, k);
            }
            else if (Weaker(scores, heap[0], i))
            {
                heap[0] = i;
                SiftDown(heap, scores, 0, k);
            }
        }
        // The production sort is unstable on equal scores: compare the heap before it (the same array, the same sort).
        return heap.AsSpan(0, count).ToArray();
    }

    [Fact]
    public void TopKBlockSkipLeavesTheHeapAsTheScalarScanDoes()
    {
        var rng = new Random(77);
        foreach (int n in new[] { 23, 64, 1000, 4099, 248_320 })
            foreach (int k in new[] { 1, 2, 7, 20, 40, 63 })
            {
                if (k >= n) continue;
                for (int trial = 0; trial < (n > 10_000 ? 2 : 12); trial++)
                {
                    var scores = new float[n];
                    for (int v = 0; v < n; v++)
                    {
                        int kind = rng.Next(60);
                        scores[v] = kind switch
                        {
                            0 => float.NaN,
                            1 => float.NegativeInfinity,
                            2 => float.PositiveInfinity,
                            3 => -0f,
                            4 => 0f,
                            _ => MathF.Round((float)(rng.NextDouble() * 4 - 2), 1),
                        };
                    }
                    if (trial % 3 == 0) scores[0] = float.NaN;   // a NaN root: nothing enters
                    int[] expected = OracleTopK(scores, k);
                    var sampler = new TokenSampler(new SamplingConfig { Temperature = 1f, TopK = k, Seed = 1 });
                    var heapField = typeof(TokenSampler).GetField("_heapBuffer", BindingFlags.Instance | BindingFlags.NonPublic)!;
                    int[] sorted = (int[])ApplyTopK.Invoke(sampler, new object[] { scores })!;
                    // ApplyTopK sorted its heap in place; sort the oracle's heap the same way and compare.
                    Array.Sort(expected, 0, expected.Length, Comparer<int>.Create((a, b) => scores[b].CompareTo(scores[a])));
                    Assert.Equal(expected, sorted);
                    Assert.Same(sorted, heapField.GetValue(sampler));
                }
            }
    }
}
