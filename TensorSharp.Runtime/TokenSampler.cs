// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Token sampler supporting temperature, top-k, top-p (nucleus), min-p,
    /// repetition/presence/frequency penalties, and stop sequences.
    /// </summary>
    public sealed class TokenSampler
    {
        private readonly SamplingConfig _config;
        private readonly Random _rng;

        public TokenSampler(SamplingConfig config)
        {
            _config = config ?? SamplingConfig.Default;
            _rng = config?.Seed >= 0 ? new Random(config.Seed) : new Random();
        }

        /// <summary>
        /// Sample a token from the logits array, applying all configured transformations.
        /// </summary>
        /// <param name="logits">Raw logits from the model (vocabSize elements).</param>
        /// <param name="generatedTokenIds">Previously generated token ids for penalty computation.</param>
        /// <returns>Selected token id.</returns>
        public int Sample(float[] logits, IList<int>? generatedTokenIds = null)
        {
            int vocabSize = logits.Length;

            // Greedy (temperature <= 0) reduces to an argmax over the penalized
            // logits — top-k / top-p / min-p are never applied on this branch.
            // Penalties only adjust the handful of distinct already-generated
            // tokens, so compute it allocation-free: a plain argmax when there is
            // nothing to penalize, otherwise penalize just those positions in
            // place, take the argmax, and restore the caller's buffer. This is
            // bit-identical to the full-vocab copy path below but avoids a
            // per-token vocab-sized allocation + memcpy (~250k floats per token
            // at this vocab size — pure waste on the hot decode path).
            if (_config.Temperature <= 0f)
            {
                bool hasHistory = generatedTokenIds != null && generatedTokenIds.Count > 0;
                if (!HasPenalties() || !hasHistory)
                    return Argmax(logits);
                return ArgmaxWithPenaltiesInPlace(logits, generatedTokenIds);
            }

            // Non-greedy: work on a copy to avoid mutating the caller's buffer.
            float[] scores = new float[vocabSize];
            Array.Copy(logits, scores, vocabSize);

            ApplyPenalties(scores, generatedTokenIds);

            ApplyTemperature(scores, _config.Temperature);

            float[] probs = Softmax(scores);

            ApplyMinP(probs);
            int[] candidates = ApplyTopK(probs);
            candidates = ApplyTopP(probs, candidates);

            return SampleFromCandidates(probs, candidates);
        }

        /// <summary>
        /// Greedy argmax over the penalized logits, applied in place to only the
        /// distinct already-generated token positions and then restored, so the
        /// caller's buffer is left unchanged. Produces the exact same index as
        /// copying the whole vocab and running <see cref="ApplyPenalties"/> +
        /// <see cref="Argmax"/>, but without the per-token vocab-sized allocation
        /// and memcpy. Only called when <see cref="HasPenalties"/> and there is
        /// generation history.
        /// </summary>
        private int ArgmaxWithPenaltiesInPlace(float[] logits, IList<int> generatedTokenIds)
        {
            // Distinct counts over the history (frequency penalty needs the count;
            // repetition/presence need presence). Mirrors ApplyPenalties' counting.
            var counts = new Dictionary<int, int>();
            foreach (int id in generatedTokenIds)
            {
                if (id < 0 || id >= logits.Length)
                    continue;
                counts.TryGetValue(id, out int c);
                counts[id] = c + 1;
            }
            if (counts.Count == 0)
                return Argmax(logits);

            float repPenalty = _config.RepetitionPenalty > 0f ? _config.RepetitionPenalty : 1.0f;
            float presPenalty = _config.PresencePenalty;
            float freqPenalty = _config.FrequencyPenalty;

            // Save originals, apply the penalty in place (same arithmetic and order
            // as ApplyPenalties), argmax, then restore the caller's buffer exactly.
            var saved = new KeyValuePair<int, float>[counts.Count];
            int si = 0;
            foreach (var (tokenId, count) in counts)
            {
                float original = logits[tokenId];
                saved[si++] = new KeyValuePair<int, float>(tokenId, original);

                float v = original;
                if (repPenalty != 1.0f)
                    v = v > 0 ? v / repPenalty : v * repPenalty;
                if (presPenalty != 0f)
                    v -= presPenalty;
                if (freqPenalty != 0f)
                    v -= freqPenalty * count;
                logits[tokenId] = v;
            }

            int best = Argmax(logits);

            for (int i = 0; i < saved.Length; i++)
                logits[saved[i].Key] = saved[i].Value;

            return best;
        }

        /// <summary>
        /// Check whether any stop sequence has been produced, and return
        /// the trimmed decoded text and whether generation should stop.
        /// </summary>
        public (string text, bool shouldStop) CheckStopSequences(string decodedSoFar)
        {
            if (_config.StopSequences == null || _config.StopSequences.Count == 0)
                return (decodedSoFar, false);

            foreach (string stop in _config.StopSequences)
            {
                int idx = decodedSoFar.IndexOf(stop, StringComparison.Ordinal);
                if (idx >= 0)
                    return (decodedSoFar.Substring(0, idx), true);
            }
            return (decodedSoFar, false);
        }

        private bool HasPenalties()
        {
            // RepetitionPenalty <= 0 means "disabled" to operators
            // (--repeat-penalty 0); applying it literally would divide
            // positive logits by zero and infinitely boost repeats.
            return (_config.RepetitionPenalty > 0f && _config.RepetitionPenalty != 1.0f) ||
                   _config.PresencePenalty != 0f ||
                   _config.FrequencyPenalty != 0f;
        }

        #region Penalty Application

        /// <summary>
        /// Apply the configured repetition/presence/frequency penalties to
        /// <paramref name="scores"/> in place. Public so speculative decoding
        /// can penalize the MTP draft head's logits with the SAME history the
        /// verification sampler uses — without that alignment, drafting argmaxes
        /// raw logits while verification draws from penalized ones, and
        /// acceptance decays toward zero as the output history grows.
        /// <paramref name="pendingTokens"/> carries tokens drafted in the
        /// current speculative window that are not yet in
        /// <paramref name="generatedTokenIds"/>.
        /// </summary>
        public void ApplyPenalties(
            float[] scores,
            IList<int>? generatedTokenIds,
            IReadOnlyList<int>? pendingTokens = null)
        {
            bool anyHistory = (generatedTokenIds != null && generatedTokenIds.Count > 0)
                || (pendingTokens != null && pendingTokens.Count > 0);
            if (!anyHistory)
                return;
            if (!HasPenalties())
                return;

            // Count occurrences
            var counts = new Dictionary<int, int>();
            if (generatedTokenIds != null)
            {
                foreach (int id in generatedTokenIds)
                {
                    counts.TryGetValue(id, out int c);
                    counts[id] = c + 1;
                }
            }
            if (pendingTokens != null)
            {
                foreach (int id in pendingTokens)
                {
                    counts.TryGetValue(id, out int c);
                    counts[id] = c + 1;
                }
            }

            float repPenalty = _config.RepetitionPenalty > 0f ? _config.RepetitionPenalty : 1.0f;
            float presPenalty = _config.PresencePenalty;
            float freqPenalty = _config.FrequencyPenalty;

            foreach (var (tokenId, count) in counts)
            {
                if (tokenId < 0 || tokenId >= scores.Length)
                    continue;

                // Repetition penalty (multiplicative, as in Ctrl paper)
                if (repPenalty != 1.0f)
                {
                    if (scores[tokenId] > 0)
                        scores[tokenId] /= repPenalty;
                    else
                        scores[tokenId] *= repPenalty;
                }

                // Presence penalty (additive, applied once regardless of count)
                if (presPenalty != 0f)
                    scores[tokenId] -= presPenalty;

                // Frequency penalty (additive, proportional to count)
                if (freqPenalty != 0f)
                    scores[tokenId] -= freqPenalty * count;
            }
        }

        #endregion

        #region Temperature

        private static void ApplyTemperature(float[] scores, float temperature)
        {
            float invT = 1.0f / temperature;
            for (int i = 0; i < scores.Length; i++)
                scores[i] *= invT;
        }

        #endregion

        #region Softmax

        private static float[] Softmax(float[] scores)
        {
            int n = scores.Length;
            float max = float.NegativeInfinity;
            for (int i = 0; i < n; i++)
                if (scores[i] > max) max = scores[i];

            float[] probs = new float[n];
            float sum = 0;
            for (int i = 0; i < n; i++)
            {
                probs[i] = MathF.Exp(scores[i] - max);
                sum += probs[i];
            }
            if (sum > 0)
            {
                float invSum = 1.0f / sum;
                for (int i = 0; i < n; i++)
                    probs[i] *= invSum;
            }
            return probs;
        }

        #endregion

        #region Top-K

        /// <summary>
        /// Returns indices of top-K tokens sorted by probability (descending).
        /// If topK is 0, returns all indices sorted by probability.
        /// </summary>
        private int[] ApplyTopK(float[] probs)
        {
            int n = probs.Length;
            int k = _config.TopK > 0 ? Math.Min(_config.TopK, n) : n;

            // Build index array and partial sort
            int[] indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;

            if (k < n)
            {
                // Partial sort: find top-K elements
                PartialSort(indices, probs, 0, n - 1, k);
                // Sort the top-K by probability descending
                Array.Sort(indices, 0, k, new ProbComparer(probs));
                int[] topK = new int[k];
                Array.Copy(indices, topK, k);
                return topK;
            }

            Array.Sort(indices, new ProbComparer(probs));
            return indices;
        }

        private sealed class ProbComparer : IComparer<int>
        {
            private readonly float[] _p;
            public ProbComparer(float[] p) => _p = p;
            public int Compare(int a, int b) => _p[b].CompareTo(_p[a]);
        }

        private static void PartialSort(int[] indices, float[] probs, int lo, int hi, int k)
        {
            while (lo < hi)
            {
                int pivot = Partition(indices, probs, lo, hi);
                if (pivot == k) return;
                if (pivot < k)
                    lo = pivot + 1;
                else
                    hi = pivot - 1;
            }
        }

        private static int Partition(int[] indices, float[] probs, int lo, int hi)
        {
            float pivotVal = probs[indices[hi]];
            int store = lo;
            for (int i = lo; i < hi; i++)
            {
                if (probs[indices[i]] >= pivotVal)
                {
                    (indices[store], indices[i]) = (indices[i], indices[store]);
                    store++;
                }
            }
            (indices[store], indices[hi]) = (indices[hi], indices[store]);
            return store;
        }

        #endregion

        #region Top-P (Nucleus)

        /// <summary>
        /// Filter candidates to the smallest set whose cumulative probability >= topP.
        /// Input candidates must be sorted by probability descending.
        /// </summary>
        private int[] ApplyTopP(float[] probs, int[] candidates)
        {
            if (_config.TopP >= 1.0f)
                return candidates;

            float cumulative = 0f;
            int cutoff = candidates.Length;
            for (int i = 0; i < candidates.Length; i++)
            {
                cumulative += probs[candidates[i]];
                if (cumulative >= _config.TopP)
                {
                    cutoff = i + 1;
                    break;
                }
            }

            if (cutoff < candidates.Length)
            {
                int[] trimmed = new int[cutoff];
                Array.Copy(candidates, trimmed, cutoff);
                return trimmed;
            }
            return candidates;
        }

        #endregion

        #region Min-P

        /// <summary>
        /// Zero out probabilities below min_p * max_probability.
        /// </summary>
        private void ApplyMinP(float[] probs)
        {
            if (_config.MinP <= 0f)
                return;

            float maxProb = 0f;
            for (int i = 0; i < probs.Length; i++)
                if (probs[i] > maxProb) maxProb = probs[i];

            float threshold = _config.MinP * maxProb;
            for (int i = 0; i < probs.Length; i++)
                if (probs[i] < threshold) probs[i] = 0f;
        }

        #endregion

        #region Final Sampling

        private int SampleFromCandidates(float[] probs, int[] candidates)
        {
            if (candidates.Length == 0)
                return 0;
            if (candidates.Length == 1)
                return candidates[0];

            // Re-normalize probabilities over candidates
            float sum = 0f;
            for (int i = 0; i < candidates.Length; i++)
                sum += probs[candidates[i]];

            if (sum <= 0f)
                return candidates[0];

            float r = (float)_rng.NextDouble() * sum;
            float cumulative = 0f;
            for (int i = 0; i < candidates.Length; i++)
            {
                cumulative += probs[candidates[i]];
                if (r <= cumulative)
                    return candidates[i];
            }
            return candidates[candidates.Length - 1];
        }

        private static int Argmax(float[] values)
        {
            int best = 0;
            float bestVal = values[0];
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > bestVal)
                {
                    bestVal = values[i];
                    best = i;
                }
            }
            return best;
        }

        #endregion
    }
}

