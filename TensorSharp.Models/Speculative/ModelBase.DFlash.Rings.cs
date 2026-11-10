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
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Per-request DFlash KV rings.
    ///
    /// The drafter's ring is its context: the last ringRows committed positions of
    /// ONE sequence, slot p % ringRows. With a single ring per model, requests that
    /// take turns speculating overwrite each other's rows at colliding slots, and a
    /// batched verify that drafts for several sequences in one step cannot share it
    /// at all. Each request therefore binds a ring of its own. The model's own ring
    /// goes to the first request that asks while no one holds it, and keeps its rows
    /// when that request ends: the next turn of a lone conversation continues on the
    /// context its previous turn left, exactly as before rings were per request.
    /// Further concurrent requests take pooled rings, recycled and never freed while
    /// the model lives: the fused drafter graphs are cached by the ring's host
    /// pointer, so a freed ring whose address came back would replay a graph over a
    /// released device copy. A recycled ring restarts its valid run, so nothing the
    /// previous owner wrote is read as context.
    /// </summary>
    public abstract partial class ModelBase
    {
        private sealed class DFlashRing
        {
            public Tensor[] K, V;
            public int Filled, ValidFrom;
            public bool HostStale;
            public string Owner;
            public long LastUsed;
            /// <summary>A fused inject failed for the request on this ring: its context is incomplete, and the request
            /// decodes without drafts (ModelBase.DFlash.cs, DFlashEncodeAndInject). Cleared when the ring changes hands.</summary>
            public bool DraftingDeclined;
        }

        /// <summary>Rings beyond the model's own (81 MB each on Qwen3.8-27B's DFlash2).
        /// More concurrent speculating requests than this steal the least recently
        /// used ring, which costs the loser its drafting context, never correctness.</summary>
        private const int MaxDFlashRequestRings = 8;

        private DFlashRing _dflashModelRing;
        private DFlashRing _dflashActiveRing;
        private List<DFlashRing> _dflashRingPool;
        private long _dflashRingClock;

        /// <summary>
        /// Point the drafter at <paramref name="requestId"/>'s ring: the model's own
        /// ring when no other request holds it, else a free (or the least recently
        /// used) pooled ring on its first bind. Null selects the model's own ring
        /// without claiming it. Cheap when the ring is already bound.
        /// </summary>
        internal void BindDFlashRing(string requestId)
        {
            if (!_hasDFlash || _dflashRingK == null)
                return;
            _dflashModelRing ??= new DFlashRing { K = _dflashRingK, V = _dflashRingV };
            _dflashActiveRing ??= _dflashModelRing;

            DFlashRing target = requestId == null ? _dflashModelRing : FindOrAssignDFlashRing(requestId);
            target.LastUsed = ++_dflashRingClock;
            if (ReferenceEquals(target, _dflashActiveRing))
                return;

            _dflashActiveRing.Filled = _dflashRingFilled;
            _dflashActiveRing.ValidFrom = _dflashRingValidFrom;
            _dflashActiveRing.HostStale = _dflashRingHostStale;
            _dflashRingK = target.K;
            _dflashRingV = target.V;
            _dflashRingFilled = target.Filled;
            _dflashRingValidFrom = target.ValidFrom;
            _dflashRingHostStale = target.HostStale;
            if (_dflashArrays != null)
            {
                for (int l = 0; l < _dflashArrays.RingK.Length; l++)
                {
                    _dflashArrays.RingK[l] = TensorComputePrimitives.GetStoragePointer(target.K[l]);
                    _dflashArrays.RingV[l] = TensorComputePrimitives.GetStoragePointer(target.V[l]);
                }
            }
            _dflashActiveRing = target;
        }

        /// <summary>Return <paramref name="requestId"/>'s ring to the pool.</summary>
        internal void ReleaseDFlashRing(string requestId)
        {
            // Without ring objects (one request at a time) the decline lives on the model; the request is over.
            _dflashModelDraftingDeclined = false;
            if (requestId == null)
                return;
            if (_dflashModelRing != null && string.Equals(_dflashModelRing.Owner, requestId, StringComparison.Ordinal))
            {
                _dflashModelRing.Owner = null;   // its rows stay for the conversation's next turn
                _dflashModelRing.DraftingDeclined = false;
                return;
            }
            if (_dflashRingPool == null)
                return;
            foreach (var ring in _dflashRingPool)
            {
                if (!string.Equals(ring.Owner, requestId, StringComparison.Ordinal))
                    continue;
                if (ReferenceEquals(ring, _dflashActiveRing))
                    BindDFlashRing(null);
                ring.Owner = null;
                ring.DraftingDeclined = false;
                return;
            }
        }

        private DFlashRing FindOrAssignDFlashRing(string requestId)
        {
            if (_dflashModelRing.Owner == null || string.Equals(_dflashModelRing.Owner, requestId, StringComparison.Ordinal))
            {
                bool pooled = false;
                if (_dflashRingPool != null)
                    foreach (var ring in _dflashRingPool)
                        pooled |= string.Equals(ring.Owner, requestId, StringComparison.Ordinal);
                if (!pooled)
                {
                    if (!string.Equals(_dflashModelRing.Owner, requestId, StringComparison.Ordinal))
                        _dflashModelRing.DraftingDeclined = false;   // a new owner drafts until its own inject fails
                    _dflashModelRing.Owner = requestId;
                    return _dflashModelRing;
                }
            }
            _dflashRingPool ??= new List<DFlashRing>();
            DFlashRing free = null, lru = null;
            foreach (var ring in _dflashRingPool)
            {
                if (string.Equals(ring.Owner, requestId, StringComparison.Ordinal))
                    return ring;
                if (ring.Owner == null)
                    free ??= ring;
                else if (!ReferenceEquals(ring, _dflashActiveRing) && (lru == null || ring.LastUsed < lru.LastUsed))
                    lru = ring;
            }
            DFlashRing chosen = free;
            if (chosen == null && _dflashRingPool.Count < MaxDFlashRequestRings)
            {
                chosen = AllocateDFlashRing();
                _dflashRingPool.Add(chosen);
            }
            chosen ??= lru ?? _dflashModelRing;
            if (ReferenceEquals(chosen, _dflashModelRing))
                return chosen;
            // A new owner starts with an empty valid run and drafting on (a ring taken from its least recently used
            // owner may carry that owner's decline); the bytes are left as they are.
            chosen.Owner = requestId;
            chosen.Filled = 0;
            chosen.ValidFrom = 0;
            chosen.DraftingDeclined = false;
            return chosen;
        }

        private DFlashRing AllocateDFlashRing()
        {
            var cfg = _dflash;
            var ring = new DFlashRing { K = new Tensor[cfg.NumLayers], V = new Tensor[cfg.NumLayers] };
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                ring.K[il] = new Tensor(_allocator, DType.Float32, cfg.NumKVHeads, _dflashRingRows, cfg.HeadDim);
                ring.V[il] = new Tensor(_allocator, DType.Float32, cfg.NumKVHeads, _dflashRingRows, cfg.HeadDim);
                // Finite bytes for the same reason as the model's ring (see LoadDFlashDraftWeights).
                Ops.Fill(ring.K[il], 0f);
                Ops.Fill(ring.V[il], 0f);
            }
            return ring;
        }

        private void DisposeDFlashRingPool()
        {
            if (_dflashModelRing != null)
            {
                // The model's own ring is disposed through the fields by DisposeDFlash.
                _dflashRingK = _dflashModelRing.K;
                _dflashRingV = _dflashModelRing.V;
            }
            if (_dflashRingPool != null)
            {
                foreach (var ring in _dflashRingPool)
                {
                    foreach (var t in ring.K) t?.Dispose();
                    foreach (var t in ring.V) t?.Dispose();
                }
                _dflashRingPool = null;
            }
            _dflashModelRing = null;
            _dflashActiveRing = null;
        }
    }
}
