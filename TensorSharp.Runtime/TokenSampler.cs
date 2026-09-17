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
        private float[]? _scoreBuffer;
        private int[]? _indexBuffer;
        // Reused top-k heap. Sized k (typically 20-40), so this is only about
        // dropping one small allocation per sampled token, but the sampler runs
        // once per token for the life of the process.
        private int[]? _heapBuffer;
        private readonly Dictionary<int, int> _penaltyCounts = new();
        private int[]? _penaltyTokenBuffer;
        private float[]? _penaltyOriginalBuffer;
        private int _thinkingScanned;
        private int _thinkingLastScannedToken = -1;
        private int _thinkingClosedAt = -1;
        private int _thinkingCloseRequestedAt = -1;
        // Channel state derived from committed history (see ScanThinkingHistory).
        private bool _thinkingScanStarted;
        private bool _channelOpen;
        private int _channelStart;

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
        /// <summary>True when, for any step that already has at least one
        /// generated token, <see cref="Sample"/> reduces to a plain first-max
        /// argmax over the raw logits after checking TryGetForcedThinkingToken:
        /// greedy temperature, no grammar, no penalties. The engine checks that
        /// forced-token override before consuming a device-side argmax, so
        /// ordinary greedy decoding need not materialize host logits.</summary>
        internal bool IsPlainGreedyArgmax =>
            _config.Temperature <= 0f && _config.Grammar == null && !HasPenalties()
            && _config.ThinkingBudget?.SuppressUnopenedEnd != true;

        public int Sample(float[] logits, IList<int>? generatedTokenIds = null)
        {
            int vocabSize = logits.Length;

            if (TryGetForcedThinkingToken(generatedTokenIds, out int thinkingEnd))
            {
                if (thinkingEnd >= vocabSize)
                    throw new InvalidOperationException("Thinking end token is outside the model vocabulary.");
                return thinkingEnd;
            }

            // A close with no open channel is masked (see ThinkingTokenBudget).
            // TryGetForcedThinkingToken has just brought the channel state up to date.
            var channelPolicy = _config.ThinkingBudget;
            if (channelPolicy is { SuppressUnopenedEnd: true } && !_channelOpen
                && channelPolicy.EndTokenId < vocabSize)
                logits[channelPolicy.EndTokenId] = float.NegativeInfinity;

            // Grammar-constrained decoding. Applied FIRST and by rewriting the
            // logits themselves, so every downstream stage — penalties, top-k,
            // top-p, min-p, temperature, and the greedy shortcut below — can only
            // ever see legal tokens. Filtering afterwards would not be equivalent:
            // top-p would compute its nucleus over a distribution including
            // tokens that cannot be emitted, and could leave nothing behind.
            var grammar = _config.Grammar;
            if (grammar != null && !grammar.IsDead)
            {
                grammar.ApplyMask(logits, allowEos: grammar.IsComplete);
            }

            // First-token constraint (structured output): pick the most probable
            // of the allowed ids. Applies only before anything has been generated;
            // later tokens sample normally.
            var allow = _config.FirstTokenAllowList;
            if (allow != null && allow.Count > 0
                && (generatedTokenIds == null || generatedTokenIds.Count == 0))
            {
                int best = -1;
                foreach (int id in allow)
                {
                    if (id < 0 || id >= vocabSize) continue;
                    if (best < 0 || logits[id] > logits[best]) best = id;
                }
                if (best >= 0)
                    return best;
            }

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
                return ArgmaxWithPenaltiesInPlace(logits, generatedTokenIds!);
            }

            // Non-greedy: work on a copy to avoid mutating the caller's buffer.
            float[] scores = _scoreBuffer != null && _scoreBuffer.Length == vocabSize
                ? _scoreBuffer
                : (_scoreBuffer = new float[vocabSize]);
            Array.Copy(logits, scores, vocabSize);

            ApplyPenalties(scores, generatedTokenIds);

            // Match llama.cpp's default sampler chain exactly: penalties ->
            // top-k -> top-p -> min-p -> temperature -> distribution.  In
            // particular, top-p must normalize the already top-k-truncated set;
            // normalizing the full vocabulary first changes the nucleus and does
            // a needless vocab-sized exp/softmax on every generated token.
            int[] candidates = ApplyTopK(scores);
            candidates = ApplyTopP(scores, candidates);
            candidates = ApplyMinP(scores, candidates);
            ApplyTemperature(scores, candidates, _config.Temperature);

            return SampleFromCandidates(scores, candidates);
        }

        /// <summary>
        /// Apply the same decision to host sampling and pending device argmax
        /// tokens. Peeking never commits a token: only committed output history
        /// advances this scan, and a rollback causes it to be reconstructed.
        /// </summary>
        internal bool TryGetForcedThinkingToken(IList<int>? tokens, out int token)
        {
            token = -1;
            var budget = _config.ThinkingBudget;
            if (budget == null) return false;
            int count = tokens?.Count ?? 0;
            ScanThinkingHistory(tokens, count, budget);

            // An active answer grammar must never be bypassed. A thinking
            // request installs its grammar with delayed activation instead.
            if (!_channelOpen || _config.Grammar?.IsActive == true)
                return false;
            if (_thinkingCloseRequestedAt < 0)
            {
                long inChannel = count - _channelStart;
                if (inChannel < budget.TokenLimit)
                    return false;
                if (budget.CloseAtBoundary != null && inChannel < 2L * budget.TokenLimit
                    && !(inChannel > 0 && budget.CloseAtBoundary(tokens![count - 1])))
                    return false;
            }
            token = budget.EndTokenId;
            return true;
        }

        /// <summary>Request a normal sampled closing token after a detected
        /// reasoning loop. The caller must supply only committed output history;
        /// this neither appends a token nor advances a grammar or model cache.</summary>
        internal bool TryRequestThinkingClosure(IList<int> tokens)
        {
            var budget = _config.ThinkingBudget;
            if (budget?.CloseOnRepetition != true || _config.Grammar?.IsActive == true)
                return false;
            ScanThinkingHistory(tokens, tokens.Count, budget);
            if (!_channelOpen) return false;
            _thinkingCloseRequestedAt = tokens.Count;
            return true;
        }

        private void ScanThinkingHistory(IList<int>? tokens, int count, ThinkingTokenBudget budget)
        {
            if (!_thinkingScanStarted ||
                _thinkingScanned > count ||
                (_thinkingScanned > 0 && tokens![_thinkingScanned - 1] != _thinkingLastScannedToken) ||
                (_thinkingClosedAt >= 0 && (_thinkingClosedAt >= count || tokens![_thinkingClosedAt] != budget.EndTokenId)))
            {
                _thinkingScanStarted = true;
                _thinkingScanned = 0;
                _thinkingClosedAt = -1;
                _thinkingCloseRequestedAt = -1;
                _channelOpen = budget.OpenAtStart;
                _channelStart = 0;
            }
            for (int i = _thinkingScanned; i < count; i++)
            {
                int token = tokens![i];
                if (_channelOpen)
                {
                    if (token == budget.EndTokenId)
                    {
                        _channelOpen = false;
                        _thinkingClosedAt = i;
                        _thinkingCloseRequestedAt = -1;
                    }
                }
                else if (budget.OpenTokenId >= 0 && token == budget.OpenTokenId)
                {
                    // Only a family whose model opens the channel itself can reopen it;
                    // with no open token a closed channel stays closed.
                    _channelOpen = true;
                    _channelStart = i + 1;
                }
            }
            _thinkingScanned = count;
            _thinkingLastScannedToken = count > 0 ? tokens![count - 1] : -1;
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
            // llama.cpp bounds penalty history (64 tokens by default). Reuse a
            // sampler-owned count map so long generations stay O(last_n) and do
            // not allocate a dictionary on every decoded token.
            var counts = CollectPenaltyCounts(logits.Length, generatedTokenIds, null);
            if (counts.Count == 0)
                return Argmax(logits);

            float repPenalty = _config.RepetitionPenalty > 0f ? _config.RepetitionPenalty : 1.0f;
            float presPenalty = _config.PresencePenalty;
            float freqPenalty = _config.FrequencyPenalty;

            // Save originals, apply the penalty in place (same arithmetic and order
            // as ApplyPenalties), argmax, then restore the caller's buffer exactly.
            EnsurePenaltySaveCapacity(counts.Count);
            int[] tokenBuffer = _penaltyTokenBuffer!;
            float[] originalBuffer = _penaltyOriginalBuffer!;
            int si = 0;
            foreach (var (tokenId, count) in counts)
            {
                float original = logits[tokenId];
                tokenBuffer[si] = tokenId;
                originalBuffer[si] = original;
                si++;

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

            for (int i = 0; i < si; i++)
                logits[tokenBuffer[i]] = originalBuffer[i];

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

            var counts = CollectPenaltyCounts(scores.Length, generatedTokenIds, pendingTokens);
            if (counts.Count == 0)
                return;

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

        private Dictionary<int, int> CollectPenaltyCounts(
            int vocabSize,
            IList<int>? generatedTokenIds,
            IReadOnlyList<int>? pendingTokens)
        {
            _penaltyCounts.Clear();
            int lastN = _config.PenaltyLastN;
            if (lastN == 0)
                return _penaltyCounts;

            int remaining = lastN < 0 ? int.MaxValue : lastN;

            // Pending speculative tokens are the newest part of the combined
            // history, so they consume the window before generated tokens.
            if (pendingTokens != null && pendingTokens.Count > 0 && remaining > 0)
            {
                int take = Math.Min(pendingTokens.Count, remaining);
                int start = pendingTokens.Count - take;
                for (int i = start; i < pendingTokens.Count; i++)
                    AddPenaltyToken(pendingTokens[i], vocabSize);
                if (remaining != int.MaxValue)
                    remaining -= take;
            }

            if (generatedTokenIds != null && generatedTokenIds.Count > 0 && remaining > 0)
            {
                int take = Math.Min(generatedTokenIds.Count, remaining);
                int start = generatedTokenIds.Count - take;
                for (int i = start; i < generatedTokenIds.Count; i++)
                    AddPenaltyToken(generatedTokenIds[i], vocabSize);
            }

            return _penaltyCounts;
        }

        private void AddPenaltyToken(int tokenId, int vocabSize)
        {
            if ((uint)tokenId >= (uint)vocabSize)
                return;
            _penaltyCounts.TryGetValue(tokenId, out int count);
            _penaltyCounts[tokenId] = count + 1;
        }

        private void EnsurePenaltySaveCapacity(int count)
        {
            if (_penaltyTokenBuffer != null && _penaltyTokenBuffer.Length >= count)
                return;
            int capacity = Math.Max(16, count);
            _penaltyTokenBuffer = new int[capacity];
            _penaltyOriginalBuffer = new float[capacity];
        }

        #endregion

        #region Temperature

        /// <summary>
        /// Scale the surviving candidates by 1/temperature.
        /// <para>
        /// Only <paramref name="candidates"/> are touched. Nothing reads
        /// <paramref name="scores"/> outside that set afterwards
        /// (<see cref="SampleFromCandidates"/> indexes exclusively through the
        /// candidate list), so this is identical to scaling the whole vocabulary —
        /// it just drops a 248K-element pass per generated token.
        /// </para>
        /// </summary>
        private static void ApplyTemperature(float[] scores, int[] candidates, float temperature)
        {
            float invT = 1.0f / temperature;
            for (int i = 0; i < candidates.Length; i++)
                scores[candidates[i]] *= invT;
        }

        #endregion

        #region Top-K

        /// <summary>
        /// Returns indices of top-K tokens sorted by probability (descending).
        /// If topK is 0, returns all indices sorted by probability.
        /// </summary>
        /// <summary>
        /// Select the <c>top_k</c> highest-scoring token ids, ordered by score
        /// descending.
        /// </summary>
        /// <remarks>
        /// Uses a bounded min-heap (O(n log k), one pass) rather than a
        /// quickselect. The quickselect this replaced picked the last element as
        /// its pivot and partitioned with <c>&gt;=</c>, which degrades to O(n²)
        /// when many scores are EQUAL: every element lands on the same side and
        /// the range shrinks by one per pass. That never showed up on raw logits,
        /// which are all distinct, but grammar-constrained decoding sets every
        /// forbidden token to -inf — ~99.9% of a 248k vocabulary — and the sampler
        /// stalled for minutes on a single token. A heap is indifferent to
        /// duplicates.
        /// <para>
        /// Ties break by ascending token id so the result is deterministic
        /// (quickselect left it dependent on the input permutation).
        /// </para>
        /// </remarks>
        private int[] ApplyTopK(float[] scores)
        {
            int n = scores.Length;
            int k = _config.TopK > 0 ? Math.Min(_config.TopK, n) : n;

            if (k >= n)
            {
                // Only the no-top-k branch needs the identity permutation. Filling it
                // unconditionally wrote a vocab-sized array (hundreds of thousands of ints) on
                // every sampled token and then discarded it whenever top_k was set,
                // which is the common case (the model's own recommendation is 20).
                int[] indices = _indexBuffer != null && _indexBuffer.Length == n
                    ? _indexBuffer
                    : (_indexBuffer = new int[n]);
                for (int i = 0; i < n; i++) indices[i] = i;
                Array.Sort(indices, new ScoreComparer(scores));
                return indices;
            }

            // heap[0..k) is a min-heap on (score, -id): its root is the weakest
            // member of the running top-k, so one comparison rejects most tokens.
            int[] heap = _heapBuffer != null && _heapBuffer.Length == k
                ? _heapBuffer
                : (_heapBuffer = new int[k]);
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

            Array.Sort(heap, 0, count, new ScoreComparer(scores));
            return count == k ? heap : heap.AsSpan(0, count).ToArray();
        }

        /// <summary>True when token <paramref name="a"/> ranks below <paramref name="b"/>.</summary>
        private static bool Weaker(float[] scores, int a, int b)
        {
            float sa = scores[a], sb = scores[b];
            if (sa != sb) return sa < sb;
            return a > b;      // equal scores: lower id wins, for determinism
        }

        private static void SiftDown(int[] heap, float[] scores, int root, int count)
        {
            while (true)
            {
                int child = 2 * root + 1;
                if (child >= count) return;
                if (child + 1 < count && Weaker(scores, heap[child + 1], heap[child])) child++;
                if (!Weaker(scores, heap[child], heap[root])) return;
                (heap[root], heap[child]) = (heap[child], heap[root]);
                root = child;
            }
        }

        private sealed class ScoreComparer : IComparer<int>
        {
            private readonly float[] _scores;
            public ScoreComparer(float[] scores) => _scores = scores;
            public int Compare(int a, int b) => _scores[b].CompareTo(_scores[a]);
        }

        #endregion

        #region Top-P (Nucleus)

        /// <summary>
        /// Filter candidates to the smallest set whose cumulative probability >= topP.
        /// Input candidates must be sorted by probability descending.
        /// </summary>
        private int[] ApplyTopP(float[] scores, int[] candidates)
        {
            if (_config.TopP >= 1.0f || candidates.Length <= 1)
                return candidates;

            // candidates are already sorted by descending logit.  llama.cpp's
            // top-p sampler softmaxes this CURRENT candidate set (after top-k),
            // not the full vocabulary.
            float max = scores[candidates[0]];
            double sum = 0d;
            for (int i = 0; i < candidates.Length; i++)
                sum += Math.Exp(scores[candidates[i]] - max);
            if (!(sum > 0d))
                return new[] { candidates[0] };

            double cumulative = 0d;
            int cutoff = candidates.Length;
            for (int i = 0; i < candidates.Length; i++)
            {
                cumulative += Math.Exp(scores[candidates[i]] - max) / sum;
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
        private int[] ApplyMinP(float[] scores, int[] candidates)
        {
            if (_config.MinP <= 0f || candidates.Length <= 1)
                return candidates;

            // p(token) / p(max) == exp(logit - maxLogit), so this test is
            // independent of the normalization constant and avoids another
            // softmax. Keep at least the highest-logit token, like llama.cpp.
            double minRatio = _config.MinP;
            float max = scores[candidates[0]];
            int keep = 1;
            while (keep < candidates.Length &&
                   Math.Exp(scores[candidates[keep]] - max) >= minRatio)
            {
                keep++;
            }
            if (keep == candidates.Length)
                return candidates;

            var filtered = new int[keep];
            Array.Copy(candidates, filtered, keep);
            return filtered;
        }

        #endregion

        #region Final Sampling

        private int SampleFromCandidates(float[] scores, int[] candidates)
        {
            if (candidates.Length == 0)
                return 0;
            if (candidates.Length == 1)
                return candidates[0];

            // Softmax only the final candidate set. With Qwen3.5's top_k=20
            // this replaces a 248K-element exp loop with at most twenty.
            float max = scores[candidates[0]];
            double sum = 0d;
            for (int i = 0; i < candidates.Length; i++)
                sum += Math.Exp(scores[candidates[i]] - max);

            if (!(sum > 0d))
                return candidates[0];

            double r = _rng.NextDouble() * sum;
            double cumulative = 0d;
            for (int i = 0; i < candidates.Length; i++)
            {
                cumulative += Math.Exp(scores[candidates[i]] - max);
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
