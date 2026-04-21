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
    /// Tracks the canonical sequence of tokens currently held in the model's K/V tensors,
    /// plus the optional next-token logits produced by the most recent forward call.
    ///
    /// This object is the single source of truth for "what is the model in the middle of?".
    /// It mirrors the per-layer K/V tensor state inside the model so that the orchestrator
    /// can decide, for any new input prompt, how many leading tokens are already cached.
    ///
    /// Design invariants:
    ///   1. Every token in <see cref="Tokens"/> has been forwarded through the model exactly
    ///      once (in order). The model's internal `_cacheSeqLen` always equals
    ///      <see cref="Count"/>.
    ///   2. <see cref="NextLogits"/> is non-null iff a forward call recorded its output
    ///      logits via <see cref="RecordAppend"/> with <c>nextLogits</c> set, and no
    ///      subsequent <see cref="TruncateTo"/> / <see cref="Reset"/> has invalidated them.
    ///   3. The cache is owned by the orchestrator (not the model). Resetting / truncating
    ///      the cache must always be paired with the corresponding model call.
    ///
    /// The object itself stores ONLY the token sequence and logits buffer in managed memory.
    /// The actual K/V activations stay in the model's per-layer tensors which live in the
    /// model's allocator (CPU pinned memory for CPU/Metal backends, GPU device memory for
    /// the GGML CUDA backend - the allocator is selected at model construction time and the
    /// cache helper itself never moves K/V data between host and device).
    /// </summary>
    public sealed class KVCache
    {
        private readonly List<int> _tokens = new();
        private float[] _nextLogits;

        /// <summary>The number of tokens currently held in the model's KV state.</summary>
        public int Count => _tokens.Count;

        /// <summary>Read-only view of the cached token sequence.</summary>
        public IReadOnlyList<int> Tokens => _tokens;

        /// <summary>
        /// Logits produced after the most recent token in <see cref="Tokens"/> was forwarded.
        /// Null if no logits were recorded for the current state (e.g. immediately after
        /// truncation, or if the most recent <see cref="RecordAppend"/> didn't supply them).
        /// </summary>
        public float[] NextLogits => _nextLogits;

        /// <summary>True if the cache contains no tokens.</summary>
        public bool IsEmpty => _tokens.Count == 0;

        /// <summary>
        /// Length of the longest common prefix between the cached tokens and
        /// <paramref name="other"/>. Returns 0 for empty / null inputs. Returns
        /// <see cref="Count"/> when <paramref name="other"/> starts with the entire cache.
        /// </summary>
        public int CommonPrefixLength(IReadOnlyList<int> other)
        {
            if (other == null || other.Count == 0 || _tokens.Count == 0)
                return 0;

            int max = Math.Min(_tokens.Count, other.Count);
            int i = 0;
            for (; i < max; i++)
            {
                if (_tokens[i] != other[i])
                    break;
            }
            return i;
        }

        /// <summary>
        /// True if the cached tokens are an exact prefix of <paramref name="input"/>
        /// (or equal to it). False when <paramref name="input"/> is shorter than the cache
        /// or when any token differs.
        /// </summary>
        public bool IsPrefixOf(IReadOnlyList<int> input)
        {
            if (input == null || input.Count < _tokens.Count)
                return false;

            for (int i = 0; i < _tokens.Count; i++)
            {
                if (_tokens[i] != input[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Returns true when the cached token sequence is identical to <paramref name="input"/>
        /// AND logits were recorded for the position right after it.
        /// In that case <paramref name="logits"/> receives a fresh copy of the cached logits.
        /// </summary>
        public bool TryGetExactMatchLogits(IReadOnlyList<int> input, out float[] logits)
        {
            logits = null;
            if (_nextLogits == null || input == null || input.Count != _tokens.Count)
                return false;

            for (int i = 0; i < _tokens.Count; i++)
            {
                if (_tokens[i] != input[i])
                    return false;
            }

            logits = (float[])_nextLogits.Clone();
            return true;
        }

        /// <summary>
        /// Drop everything and return to the empty state. The caller is responsible for
        /// also calling <see cref="IModelArchitecture.ResetKVCache"/> on the model.
        /// </summary>
        public void Reset()
        {
            _tokens.Clear();
            _nextLogits = null;
        }

        /// <summary>
        /// Keep only the first <paramref name="length"/> tokens. Cached logits are
        /// invalidated. The caller is responsible for also calling
        /// <see cref="IModelArchitecture.TruncateKVCache"/> on the model.
        /// </summary>
        public void TruncateTo(int length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), "Truncation length cannot be negative.");
            if (length > _tokens.Count)
                throw new ArgumentOutOfRangeException(nameof(length), "Truncation length exceeds cached token count.");

            if (length < _tokens.Count)
            {
                _tokens.RemoveRange(length, _tokens.Count - length);
                _nextLogits = null;
            }
        }

        /// <summary>
        /// Record that the model has just forwarded <paramref name="newTokens"/> on top of
        /// the current cached prefix. Updates the cache to reflect what is now in the
        /// model's KV state. <paramref name="nextLogits"/> are the logits returned by the
        /// final forward call (logits for the next token to be generated). May be null when
        /// the caller does not need to cache them (e.g. mid-prefill chunks).
        /// </summary>
        public void RecordAppend(IReadOnlyList<int> newTokens, float[] nextLogits)
        {
            if (newTokens == null)
                return;

            for (int i = 0; i < newTokens.Count; i++)
                _tokens.Add(newTokens[i]);

            _nextLogits = nextLogits;
        }

        /// <summary>
        /// Convenience overload for appending a single token (typical decode step).
        /// </summary>
        public void RecordAppend(int token, float[] nextLogits)
        {
            _tokens.Add(token);
            _nextLogits = nextLogits;
        }

        /// <summary>
        /// Plan the operations required to bring the model's KV state from its current
        /// contents to one that contains <paramref name="inputTokens"/> as a prefix
        /// (so that the next-token logits at position <c>inputTokens.Count</c> are available).
        ///
        /// The result describes what the orchestrator must do, but does not modify any
        /// state itself; the orchestrator is responsible for applying the plan to both the
        /// model and to this cache (via <see cref="TruncateTo"/> / <see cref="RecordAppend"/>).
        ///
        /// <paramref name="supportsTruncation"/> models that report <c>false</c> can only
        /// reuse the cache when the entire current cache is a prefix of the new input.
        /// </summary>
        public ReusePlan PlanReuse(IReadOnlyList<int> inputTokens, bool supportsTruncation)
        {
            if (inputTokens == null || inputTokens.Count == 0)
                return ReusePlan.Reset(0);

            // Exact match: nothing to forward, hopefully cached logits are available.
            if (TryGetExactMatchLogits(inputTokens, out float[] cachedLogits))
                return ReusePlan.ExactMatch(cachedLogits);

            int common = CommonPrefixLength(inputTokens);

            // For non-truncatable models (recurrent state): only reuse if the cache is a
            // prefix of the new input.
            if (!supportsTruncation && common < _tokens.Count)
                return ReusePlan.Reset(inputTokens.Count);

            // We always need at least one token in the forward to compute fresh logits for
            // the next step. If the input matches the cache for all but its last position,
            // back the prefix off by one to leave a token to forward.
            if (common == inputTokens.Count)
                common = Math.Max(0, inputTokens.Count - 1);

            int suffixLength = inputTokens.Count - common;

            if (common == 0)
                return ReusePlan.Reset(inputTokens.Count);

            return ReusePlan.Reuse(common, suffixLength);
        }
    }

    /// <summary>
    /// Outcome of <see cref="KVCache.PlanReuse"/>. Describes the work the orchestrator
    /// must do for the next forward call.
    /// </summary>
    public readonly struct ReusePlan
    {
        public ReusePlanKind Kind { get; }

        /// <summary>
        /// Number of tokens to keep from the existing KV cache (also the position at which
        /// the model should append the next forward). 0 when the cache is being reset.
        /// </summary>
        public int ReusedPrefixLength { get; }

        /// <summary>
        /// Number of new tokens to forward through the model on the next call.
        /// 0 when <see cref="Kind"/> is <see cref="ReusePlanKind.ExactMatch"/>.
        /// </summary>
        public int TokensToForward { get; }

        /// <summary>
        /// Pre-computed logits when <see cref="Kind"/> is <see cref="ReusePlanKind.ExactMatch"/>;
        /// otherwise null.
        /// </summary>
        public float[] CachedLogits { get; }

        private ReusePlan(ReusePlanKind kind, int reusedPrefix, int tokensToForward, float[] cachedLogits)
        {
            Kind = kind;
            ReusedPrefixLength = reusedPrefix;
            TokensToForward = tokensToForward;
            CachedLogits = cachedLogits;
        }

        public static ReusePlan ExactMatch(float[] cachedLogits)
            => new(ReusePlanKind.ExactMatch, 0, 0, cachedLogits);

        public static ReusePlan Reuse(int reusedPrefix, int tokensToForward)
            => new(ReusePlanKind.PartialReuse, reusedPrefix, tokensToForward, null);

        public static ReusePlan Reset(int tokensToForward)
            => new(ReusePlanKind.Reset, 0, tokensToForward, null);
    }

    public enum ReusePlanKind
    {
        /// <summary>
        /// The cached tokens already match the input exactly and the cached
        /// next-token logits are valid. No forward call is needed.
        /// </summary>
        ExactMatch,
        /// <summary>
        /// Truncate the model's KV cache to <see cref="ReusePlan.ReusedPrefixLength"/> and
        /// forward the next <see cref="ReusePlan.TokensToForward"/> tokens.
        /// </summary>
        PartialReuse,
        /// <summary>
        /// Reset the model's KV cache and forward all <see cref="ReusePlan.TokensToForward"/>
        /// input tokens from scratch.
        /// </summary>
        Reset,
    }
}
