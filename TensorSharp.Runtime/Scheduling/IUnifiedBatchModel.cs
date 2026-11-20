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
    /// A model whose every step is one batched forward over per-request device slots (the shared decoder runtime):
    /// prompt chunks and decode tokens of any number of requests run in the same forward, each request attending to
    /// its own slot. There is no shared primary cache, no ownership swap and no paged migration; the executor serves
    /// such a model on <see cref="ExecutionPathKind.Unified"/>.
    /// </summary>
    public interface IUnifiedBatchModel
    {
        /// <summary>False when this load cannot batch over slots (a multi-node pipeline).</summary>
        bool SupportsUnifiedBatching { get; }

        /// <summary>The request's slot, created empty when it has none (a restored prefix already made one), with room
        /// for <paramref name="reserveTokens"/> (its prompt and generation budget: no grow-and-copy mid-stream).
        /// Returns the tokens the slot holds.</summary>
        int EnsureSequenceSlot(string requestId, int reserveTokens);

        /// <summary><see cref="EnsureSequenceSlot(string, int)"/> where the step needs only <paramref name="requiredTokens"/>
        /// (its prompt, its first token and a verify window's slack): a model that sizes slots to the device's memory may hold
        /// less than the reservation and grow the slot when a later step needs it. Others hold the reservation.</summary>
        int EnsureSequenceSlot(string requestId, int reserveTokens, int requiredTokens) => EnsureSequenceSlot(requestId, reserveTokens);

        /// <summary>One forward: request s's <paramref name="tokens"/>[s] at positions <paramref name="starts"/>[s]
        /// onward, every request's rows together; <paramref name="lastRowLogits"/>[s] its last row's logits. Declines
        /// (false, nothing written) when a request's slot does not continue at its start.
        /// <paramref name="features"/>[s], when given for request s (a buffer of c x the drafter's feature width, c no
        /// more than its rows), receives the drafter features of its LAST c rows (a prompt-conditioned drafter's prefill
        /// needs only the rows its context holds); it is set to null when this forward could not capture them.
        /// A request's tokens are its prompt chunk or its one decode token: a model may attend a prompt chunk with
        /// arithmetic decode rows do not share (a plain and a speculating request prefill through it alike). A caller
        /// forwarding generated tokens as a chunk (a re-forward, a verify) must use another entry.</summary>
        bool TryForwardUnified(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, float[][] lastRowLogits,
            float[][]? features = null);

        /// <summary>Whether <see cref="TryForwardUnifiedGreedy"/> picks the tokens on the device (else it is not offered).</summary>
        bool SupportsUnifiedGreedyIds => false;

        /// <summary><see cref="TryForwardUnified"/> for requests that all draw greedily: <paramref name="nextTokens"/>[s] is
        /// request s's last row's largest logit's id (a tie to the lower id), picked where the logits are, so the vocabulary's
        /// logits are not read back. Declines as <see cref="TryForwardUnified"/> does.</summary>
        bool TryForwardUnifiedGreedy(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, int[] nextTokens) => false;

        /// <summary>Whether <see cref="TryForwardUnifiedGreedy(IReadOnlyList{string}, IReadOnlyList{int[]}, int[], int[], IReadOnlyList{int})"/>
        /// takes suppressed tokens (a reply-token policy's) on the device.</summary>
        bool SupportsUnifiedGreedySuppression => false;

        /// <summary><see cref="TryForwardUnifiedGreedy(IReadOnlyList{string}, IReadOnlyList{int[]}, int[], int[])"/> with every
        /// request's <paramref name="suppressed"/> tokens never picked (their logits -inf, as TokenSampler sets them). A model
        /// without <see cref="SupportsUnifiedGreedySuppression"/> declines any.</summary>
        bool TryForwardUnifiedGreedy(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, int[] nextTokens,
            IReadOnlyList<int> suppressed)
            => suppressed is not { Count: > 0 } && TryForwardUnifiedGreedy(requestIds, tokens, starts, nextTokens);

        /// <summary>Whether <see cref="TryForwardUnifiedRows"/> serves (requests driven by an
        /// <see cref="ISequenceStepDriver"/> need it).</summary>
        bool SupportsUnifiedRowIO => false;

        /// <summary>
        /// <see cref="TryForwardUnified"/> with per-request row input and output: <paramref name="inputRows"/>[s], when set,
        /// is decoded in place of the embedding of request s's LAST row (the model's embedding width, unscaled);
        /// <paramref name="lastRowLogits"/>[s] receives the last row's logits only where <paramref name="wantLogits"/>[s]
        /// (else it may stay null: a step no request wants logits from skips the LM head); <paramref name="lastHidden"/>[s],
        /// when given (a buffer of the embedding width), receives the last row's final hidden row (after the final norm,
        /// before the head). Declines as <see cref="TryForwardUnified"/> does.
        /// </summary>
        bool TryForwardUnifiedRows(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts,
            float[]?[] inputRows, bool[] wantLogits, float[][] lastRowLogits, float[]?[] lastHidden) => false;

        /// <summary>A request with pending media embeddings runs its prompt chunk on its own slot through the model's
        /// media path; the last row's logits.</summary>
        float[] ForwardUnifiedMedia(string requestId, int[] tokens, int start);

        /// <summary><see cref="ForwardUnifiedMedia"/> that also writes the last row's final hidden row into
        /// <paramref name="lastHidden"/> (a buffer of the embedding width): a driven request's prompt with media
        /// (<see cref="SupportsUnifiedRowIO"/>).</summary>
        float[] ForwardUnifiedMediaRows(string requestId, int[] tokens, int start, float[] lastHidden)
            => throw new NotSupportedException("This model's media prompts do not return the final hidden row.");

        /// <summary>One short slice of work the model does while the engine has nothing to run (it prepares what the next
        /// requests would otherwise wait for); the engine calls again while it stays idle and this returns true.</summary>
        bool RunIdleSlice() => false;
    }
}
