// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    /// <summary>
    /// Unified batching (<see cref="IUnifiedBatchModel"/>): every request has its own slot from its first step, and a
    /// step runs every scheduled request's prompt chunk or decode token in one forward over their slots
    /// (<see cref="TryForwardSlots"/>), whatever the count.
    /// </summary>
    public abstract partial class StandardDecoderModel : IUnifiedBatchModel, ISoloPrefillBudgetModel
    {
        public bool SupportsUnifiedBatching => SupportsPerSequenceFusedForward;

        /// <summary>A lone prompt's step on MLX, in tokens (see <see cref="PreferredSoloPrefillTokens"/>).</summary>
        private const int MlxPromptStepTokens = 512;

        /// <summary>
        /// On MLX a lone prompt prefills in steps of <see cref="MlxPromptStepTokens"/>: MLX holds memory of the order of
        /// each layer's feed-forward activations until a step ends (Qwen3.8-27B on an M5 Pro: 0.9 GB a layer for a
        /// 3.6K-token step, past 40 GB wired), so the scheduler's whole-prompt step ran out of memory where 512-token
        /// steps peak at 22 GB wired (Qwen35Model 19 GB) in the same prompt time (9.09 vs 9.07-9.11 s; 1024-token steps
        /// 8.99 s at 31 GB). Elsewhere the scheduler's own steps (0).
        /// </summary>
        public int PreferredSoloPrefillTokens(int start, int remaining)
            => _backend == BackendType.Mlx ? Math.Min(remaining, MlxPromptStepTokens) : 0;

        public int EnsureSequenceSlot(string requestId, int reserveTokens)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            RestorePrimaryCache();
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            int reserve = Math.Min(_maxContextLength, reserveTokens);
            if (!_slots.TryGetValue(requestId, out Slot slot))
            {
                EnsureDeviceRoomForSlot(requestId, Math.Max(reserve, FreshSlotTokens));
                _slots[requestId] = slot = NewSlot(reserve);
            }
            if (reserve > slot.Capacity)
                EnsureSlotCapacity(slot, reserve);
            return slot.Length;
        }

        public bool TryForwardUnified(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, float[][] lastRowLogits,
            float[][] features = null)
            => TryForwardSlots(requestIds, tokens, starts, lastRowLogits, features);

        public float[] ForwardUnifiedMedia(string requestId, int[] tokens, int start)
        {
            if (!_slots.ContainsKey(requestId))
                throw new InvalidOperationException($"{requestId} has no slot.");
            BindSequenceCache(requestId);
            try
            {
                if (_cacheSeqLen != start)
                    throw new InvalidOperationException($"{requestId}: its slot holds {_cacheSeqLen} tokens, the chunk starts at {start}.");
                return (float[])ForwardPrompt(tokens).Clone();
            }
            finally
            {
                RestorePrimaryCache();
            }
        }

        /// <summary>The request's own slot, whichever cache is active (a unified step leaves the primary active, and a
        /// checkpoint of it would be another request's state).</summary>
        bool IPrefixCacheModel.TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey) || !SupportsPrefixCheckpoints) return false;
            if (!TryGetSlot(requestId, out Slot slot))
                return HolderPrefixCacheAdapter.TryCaptureCopy(this, requestId, payloadKey, out footprint);
            _retainedSlots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(payloadKey) || slot.Length <= 0) return false;
            _retainedSlots.Add(payloadKey, Retained(CopySlot(slot, slot.Length)));
            footprint = MeasureEndState(payloadKey);
            return true;
        }
    }
}
