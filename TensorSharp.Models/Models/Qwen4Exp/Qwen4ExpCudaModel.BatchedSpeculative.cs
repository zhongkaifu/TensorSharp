// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.8-Flash-Next on the direct CUDA engine verifying several concurrent sequences' draft windows in one forward
// (Q4eCudaEngine.ForwardBatchedVerify). Without it the scheduler decoded concurrent requests one token a step in a batched
// step, without their drafts: four requests at once on the RTX 5090 decoded 192-203 tok/s in all, below one request
// alone. The MTP drafter keeps its state per slot already, so binding a request's draft state selects its slot.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public sealed partial class Qwen4ExpCudaModel : IBatchedSpeculativeTarget
    {
        /// <summary>The engine's verify and drafter run on the bound slot (<see cref="BindSequenceCache"/>), so a request held
        /// in its own slot speculates there.</summary>
        bool ISpeculativeTarget.SpecTrunkFollowsBoundCache => true;

        public int BatchedSpecWindow(int sequences)
        {
            var engine = _exec?.Engine;
            if (!HasDraftHead || engine == null || !engine.SupportsBatchedVerify || sequences < 1)
                return 0;
            // The solo window (three drafts after the anchor), as far as the decode-size kernels' rows allow.
            int window = Math.Min(((ISpeculativeTarget)this).SpecPreferredDraftWindow + 1, TensorSharp.Cuda.Q4eCudaEngine.VerifyMaxRows);
            return Math.Min(window, engine.BatchedVerifyMaxRows / sequences);
        }

        public int FusedCacheSeqLen(string requestId)
        {
            lock (_sync)
                return requestId != null && _slotByRequest != null && _slotByRequest.TryGetValue(requestId, out int id)
                    && _exec.SlotHead(id, out int head) ? head : -1;
        }

        public int FusedCacheFreeRows(string requestId)
        {
            int head = FusedCacheSeqLen(requestId);
            return head < 0 ? 0 : MaxContextLength - head;
        }

        /// <summary>The drafter's state is its slot's: binding a request's draft state makes its slot the active one.</summary>
        public void BindDraftState(string requestId)
        {
            if (requestId == null)
                return;
            lock (_sync)
            {
                if (_slotByRequest == null || !_slotByRequest.TryGetValue(requestId, out int slot) || !_exec.SetActiveSlot(slot))
                    return;
                _activeSlotKey = requestId;
                _cacheSeqLen = _exec.NPast;
            }
        }

        public void ReleaseDraftState(string requestId) { }

        private sealed class NoBatchedScope : IDisposable
        {
            public static readonly NoBatchedScope Instance = new NoBatchedScope();
            public void Dispose() { }
        }

        public IDisposable BeginBatchedSpeculation() => NoBatchedScope.Instance;

        private int[] SlotsOf(IReadOnlyList<string> requestIds, int[] positions)
        {
            if (requestIds == null || _slotByRequest == null || (positions != null && positions.Length < requestIds.Count))
                return null;
            var slots = new int[requestIds.Count];
            for (int i = 0; i < slots.Length; i++)
            {
                if (requestIds[i] == null || !_slotByRequest.TryGetValue(requestIds[i], out slots[i]))
                    return null;
                if (positions != null && (!_exec.SlotHead(slots[i], out int head) || head != positions[i]))
                    return null;
            }
            return slots;
        }

        public bool TryBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int window, int[] tokens, int[] drawn,
            float[] features)
        {
            lock (_sync)
            {
                var slots = SlotsOf(requestIds, positions);
                if (slots == null || window < 1)
                    return false;
                if (!_exec.Engine.ForwardBatchedVerify(slots, window, tokens, drawn, features))
                    return false;
                _cacheSeqLen = _exec.NPast;
                return true;
            }
        }

        public bool SupportsBatchedSpecLogits => _exec?.Engine?.SupportsBatchedVerify == true;

        public bool TryBatchedSpecVerifyLogits(IReadOnlyList<string> requestIds, int[] positions, int window, int[] tokens,
            float[] logits, float[] features)
        {
            lock (_sync)
            {
                var slots = SlotsOf(requestIds, positions);
                if (slots == null || window < 1 || logits == null)
                    return false;
                if (!_exec.Engine.ForwardBatchedVerify(slots, window, tokens, null, features, logits))
                    return false;
                _cacheSeqLen = _exec.NPast;
                return true;
            }
        }

        public void CommitBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int[] kept)
        {
            lock (_sync)
            {
                var slots = SlotsOf(requestIds, null)
                    ?? throw new InvalidOperationException("qwen4exp batched verify commit: a request lost its slot between verify and commit.");
                if (!_exec.Engine.CommitBatchedVerify(slots, kept))
                    throw new InvalidOperationException("qwen4exp batched verify commit: no matching batched verify to commit.");
                _cacheSeqLen = _exec.NPast;
            }
        }
    }
}
