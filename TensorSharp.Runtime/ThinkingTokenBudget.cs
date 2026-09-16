// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Immutable sampling policy for a reasoning channel: it closes the channel with
    /// its trained end token once the channel has run for <see cref="TokenLimit"/>
    /// tokens. The closing token counts toward the original output limit.
    /// Tracking belongs to each sampler, so configurations may be cloned safely.
    ///
    /// <para>
    /// By default the prompt already opened the channel, so it is open from the first
    /// generated token and closes once (DeepSeek V4.1's primed <c>&lt;think&gt;</c>).
    /// A family whose model opens the channel itself mid-reply (Gemma 4's
    /// <c>&lt;|channel&gt;</c>) passes <see cref="OpenTokenId"/>: the budget then counts
    /// from the opener, and a closed channel may open again.
    /// </para>
    /// </summary>
    public sealed class ThinkingTokenBudget
    {
        public ThinkingTokenBudget(int tokenLimit, int endTokenId, bool closeOnRepetition = false,
            int openTokenId = -1, bool openAtStart = true, bool suppressUnopenedEnd = false,
            Func<int, bool>? closeAtBoundary = null)
        {
            if (tokenLimit <= 0) throw new ArgumentOutOfRangeException(nameof(tokenLimit));
            if (endTokenId < 0) throw new ArgumentOutOfRangeException(nameof(endTokenId));
            if (openTokenId == endTokenId && openTokenId >= 0) throw new ArgumentException("The open and end tokens must differ.", nameof(openTokenId));
            if (openTokenId < 0 && !openAtStart)
                throw new ArgumentException("A channel with no open token must be open from the start.", nameof(openAtStart));
            TokenLimit = tokenLimit;
            EndTokenId = endTokenId;
            CloseOnRepetition = closeOnRepetition;
            OpenTokenId = openTokenId < 0 ? -1 : openTokenId;
            OpenAtStart = openAtStart;
            SuppressUnopenedEnd = suppressUnopenedEnd;
            CloseAtBoundary = closeAtBoundary;
        }

        /// <summary>Tokens the channel may run before its end token is forced.
        /// <see cref="int.MaxValue"/> means never forced.</summary>
        public int TokenLimit { get; }
        public int EndTokenId { get; }
        /// <summary>Allow the repetition guard to close an open reasoning channel
        /// through ordinary sampling instead of stopping the entire answer.</summary>
        public bool CloseOnRepetition { get; }

        /// <summary>Token with which the model opens the channel itself, or -1 when
        /// only the prompt opens it.</summary>
        public int OpenTokenId { get; }

        /// <summary>The prompt left the channel open, so generation starts inside it.</summary>
        public bool OpenAtStart { get; }

        /// <summary>
        /// Mask the end token while no channel is open. A close with nothing to close
        /// is never well-formed output, and Gemma 4 E4B uses one after a tool result to
        /// restart its answer, which a streaming client then receives twice. Masking
        /// needs host logits, so a sampler with this policy never takes the
        /// device-argmax shortcut.
        /// </summary>
        public bool SuppressUnopenedEnd { get; }

        /// <summary>
        /// When set, a channel past <see cref="TokenLimit"/> is closed only right after a
        /// token this returns true for (a line break), or at twice the limit. Closing a
        /// thought mid-sentence steers what comes next: Gemma 4 E4B, cut off at
        /// "Calculate the total using `", finished the sentence as a tool call instead of
        /// answering. Null closes exactly at the limit.
        /// </summary>
        public Func<int, bool>? CloseAtBoundary { get; }
    }
}
