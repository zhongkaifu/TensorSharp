// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
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

        /// <summary>One forward: request s's <paramref name="tokens"/>[s] at positions <paramref name="starts"/>[s]
        /// onward, every request's rows together; <paramref name="lastRowLogits"/>[s] its last row's logits. Declines
        /// (false, nothing written) when a request's slot does not continue at its start.
        /// <paramref name="features"/>[s], when given for request s (a buffer of c x the drafter's feature width, c no
        /// more than its rows), receives the drafter features of its LAST c rows (a prompt-conditioned drafter's prefill
        /// needs only the rows its context holds); it is set to null when this forward could not capture them.</summary>
        bool TryForwardUnified(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, float[][] lastRowLogits,
            float[][]? features = null);

        /// <summary>Whether <see cref="TryForwardUnifiedGreedy"/> picks the tokens on the device (else it is not offered).</summary>
        bool SupportsUnifiedGreedyIds => false;

        /// <summary><see cref="TryForwardUnified"/> for requests that all draw greedily: <paramref name="nextTokens"/>[s] is
        /// request s's last row's largest logit's id (a tie to the lower id), picked where the logits are, so the vocabulary's
        /// logits are not read back. Declines as <see cref="TryForwardUnified"/> does.</summary>
        bool TryForwardUnifiedGreedy(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, int[] nextTokens) => false;

        /// <summary>A request with pending media embeddings runs its prompt chunk on its own slot through the model's
        /// media path; the last row's logits.</summary>
        float[] ForwardUnifiedMedia(string requestId, int[] tokens, int start);
    }
}
