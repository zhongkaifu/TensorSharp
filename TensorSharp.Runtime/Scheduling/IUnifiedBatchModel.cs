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
        /// <paramref name="features"/>[s], when given for request s (a buffer of its rows x the drafter's feature
        /// width), receives each of its rows' drafter features (a prompt-conditioned drafter's prefill); it is set to
        /// null when this forward could not capture them.</summary>
        bool TryForwardUnified(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, float[][] lastRowLogits,
            float[][]? features = null);

        /// <summary>A request with pending media embeddings runs its prompt chunk on its own slot through the model's
        /// media path; the last row's logits.</summary>
        float[] ForwardUnifiedMedia(string requestId, int[] tokens, int start);
    }
}
