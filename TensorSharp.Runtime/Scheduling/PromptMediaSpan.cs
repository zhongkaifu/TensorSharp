// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// One image, video frame (pair) or audio clip inside a prompt: the prompt-token
    /// range its expansion occupies (placeholder and marker tokens included) and the
    /// identity of the content that was encoded into it.
    ///
    /// <para>
    /// The placeholder token ids of two different images are identical, so the tokens
    /// alone cannot tell whether cached K/V over a media span is the K/V of THIS image.
    /// <see cref="ContentId"/> can: it is derived from the SHA-256 of the media bytes
    /// plus what the encoder did with them (see <c>ModelMultimodalInjector</c>), so a
    /// client that resends the same picture - under any file name - gets the same id.
    /// </para>
    /// </summary>
    /// <param name="Start">First prompt token of the span (inclusive).</param>
    /// <param name="End">One past the last prompt token of the span.</param>
    /// <param name="ContentId">Content identity of the encoded media.</param>
    public readonly record struct PromptMediaSpan(int Start, int End, string ContentId);

    /// <summary>
    /// The positional media check every prompt-reuse path applies: a cached prefix may
    /// be continued only up to the start of the first media span that differs between
    /// the cached sequence and the new prompt, and never ends inside a span.
    /// </summary>
    public static class PromptMediaSpans
    {
        /// <summary>
        /// Clamp <paramref name="reusableTokens"/> - a prefix whose TOKENS already match
        /// between <paramref name="cached"/> and <paramref name="target"/> - so that
        /// every media span inside the reused prefix is the same content at the same
        /// position in both, and the prefix does not cut a span in half. Media that
        /// starts at or after the result is irrelevant: it is prefilled either way.
        /// </summary>
        /// <param name="allowReuseAcrossSpans">False for a model whose positions after a
        /// media span are not reproduced exactly by continuing a cache (Qwen 3.5's
        /// M-RoPE, see <see cref="IModelArchitecture.SupportsReuseAcrossMediaSpan"/>):
        /// the prefix then stops at the first span either side has.</param>
        public static int ClampReusablePrefix(
            int reusableTokens,
            IReadOnlyList<PromptMediaSpan> target,
            IReadOnlyList<PromptMediaSpan> cached,
            bool allowReuseAcrossSpans = true)
        {
            if (reusableTokens <= 0)
                return Math.Max(0, reusableTokens);
            int result = reusableTokens;
            while (true)
            {
                // Lowering the limit can make it cut a span an earlier pass accepted
                // whole, so repeat until nothing moves (spans need not be sorted).
                int next = ClampOneSide(result, target, cached, allowReuseAcrossSpans);
                next = ClampOneSide(next, cached, target, allowReuseAcrossSpans);
                if (next == result)
                    return result;
                result = next;
            }
        }

        private static int ClampOneSide(
            int result, IReadOnlyList<PromptMediaSpan> spans, IReadOnlyList<PromptMediaSpan> other,
            bool allowReuseAcrossSpans)
        {
            if (spans == null)
                return result;
            for (int i = 0; i < spans.Count; i++)
            {
                PromptMediaSpan span = spans[i];
                if (span.Start >= result)
                    continue;
                if (!allowReuseAcrossSpans || span.End > result || !Contains(other, span))
                    result = Math.Max(0, span.Start);
            }
            return result;
        }

        private static bool Contains(IReadOnlyList<PromptMediaSpan> spans, PromptMediaSpan span)
        {
            if (spans == null)
                return false;
            for (int i = 0; i < spans.Count; i++)
            {
                PromptMediaSpan other = spans[i];
                if (other.Start == span.Start && other.End == span.End
                    && string.Equals(other.ContentId, span.ContentId, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// The salt a pooled prefix block over <c>[blockStart, blockEnd)</c> carries for
        /// its media: the identity and position of every span that overlaps it, or
        /// null when the block holds no media token. Blocks before the first span stay
        /// unsalted, so a media prompt still shares its text-only leading blocks; the
        /// parent-chained block hash carries a span's salt into every later block.
        /// </summary>
        public static string BlockSalt(IReadOnlyList<PromptMediaSpan> spans, int blockStart, int blockEnd)
        {
            if (spans == null || spans.Count == 0)
                return null;
            System.Text.StringBuilder sb = null;
            for (int i = 0; i < spans.Count; i++)
            {
                PromptMediaSpan span = spans[i];
                if (span.End <= blockStart || span.Start >= blockEnd)
                    continue;
                (sb ??= new System.Text.StringBuilder()).Append("mm:").Append(span.ContentId)
                    .Append('@').Append(span.Start).Append('-').Append(span.End).Append(';');
            }
            return sb?.ToString();
        }
    }
}
