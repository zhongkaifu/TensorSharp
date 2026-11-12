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
    public abstract partial class StandardDecoderModel : IUnifiedBatchModel
    {
        public bool SupportsUnifiedBatching => SupportsPerSequenceFusedForward;

        public int EnsureSequenceSlot(string requestId, int reserveTokens)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            RestorePrimaryCache();
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            int reserve = Math.Min(_maxContextLength, reserveTokens);
            if (!_slots.TryGetValue(requestId, out Slot slot))
                _slots[requestId] = slot = NewSlot(reserve);
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
