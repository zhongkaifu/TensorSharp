// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;

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
            Func<int, bool>? closeAtBoundary = null, IReadOnlyList<int>? closingTokenIds = null,
            Func<int, bool>? closeOnStop = null)
        {
            if (tokenLimit <= 0) throw new ArgumentOutOfRangeException(nameof(tokenLimit));
            if (endTokenId < 0) throw new ArgumentOutOfRangeException(nameof(endTokenId));
            if (closingTokenIds != null)
                foreach (int id in closingTokenIds)
                    if (id < 0 || id == endTokenId)
                        throw new ArgumentException("Closing text tokens must be valid and must not contain the end token.", nameof(closingTokenIds));
            ClosingTokenIds = closingTokenIds is { Count: > 0 } ? closingTokenIds.ToArray() : Array.Empty<int>();
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
            CloseOnStop = closeOnStop;
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

        /// <summary>
        /// Tokens forced ahead of the end token when the budget closes the channel, or
        /// empty for a bare close. A bare close leaves the model mid-thought: Qwen3.5-9B,
        /// cut at 450 tokens, went on reasoning in the answer ("Wait, I need to check the
        /// word count again"). Qwen's published thinking-budget recipe inserts a sentence
        /// that hands over to the answer first (see ChatProtocol.ThinkingBudgetClosingText).
        /// </summary>
        public IReadOnlyList<int> ClosingTokenIds { get; }

        /// <summary>
        /// The model's stop tokens (the tokenizer's end-of-sequence test), or null. A stop
        /// drawn while the channel is open would end the turn with reasoning and no answer;
        /// with this set the draw closes the channel instead - the closing text, then the
        /// end token - and the model answers. Qwen3.8-27B NVFP4 under Aider ended 3 of 53
        /// turns that way, each mid-phrase in a counting loop, which an OpenAI client then
        /// received as an empty <c>stop</c> (issue #338).
        /// </summary>
        public Func<int, bool>? CloseOnStop { get; }
    }
}
