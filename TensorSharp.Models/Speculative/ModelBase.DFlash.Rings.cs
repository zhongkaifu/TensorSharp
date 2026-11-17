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
            /// <summary>The trunk's feature row of position <see cref="CarryEnd"/> - 1, which a followed prompt's last step
            /// wrote on the device (StandardDecoderModel.BlockDraft.cs), or null before any; <see cref="CarryEnd"/> is -1 once
            /// anything else wrote the ring.</summary>
            public Tensor Carry;
            public int CarryEnd = -1;
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
            EnsureDFlashRingObjects();
            DFlashRing target = requestId == null ? _dflashModelRing : FindOrAssignDFlashRing(requestId);
            target.LastUsed = ++_dflashRingClock;
            SelectDFlashRing(target);
        }

        /// <summary>The model's own ring as an object, bound when none is (the fields hold it until a request binds one).</summary>
        private void EnsureDFlashRingObjects()
        {
            _dflashModelRing ??= new DFlashRing { K = _dflashRingK, V = _dflashRingV };
            _dflashActiveRing ??= _dflashModelRing;
        }

        /// <summary>Make <paramref name="target"/> the bound ring: the bound one's bookkeeping goes back to its object and
        /// the target's into the model's fields (and the fused drafter's ring pointers).</summary>
        private void SelectDFlashRing(DFlashRing target)
        {
            if (ReferenceEquals(target, _dflashActiveRing))
                return;

            _dflashActiveRing.Filled = _dflashRingFilled;
            _dflashActiveRing.ValidFrom = _dflashRingValidFrom;
            _dflashActiveRing.HostStale = _dflashRingHostStale;
            _dflashActiveRing.Carry = _dflashRingCarry;
            _dflashActiveRing.CarryEnd = _dflashRingCarryEnd;
            _dflashRingK = target.K;
            _dflashRingV = target.V;
            _dflashRingFilled = target.Filled;
            _dflashRingValidFrom = target.ValidFrom;
            _dflashRingHostStale = target.HostStale;
            _dflashRingCarry = target.Carry;
            _dflashRingCarryEnd = target.CarryEnd;
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

        /// <summary>
        /// The ring <paramref name="requestId"/> drafts over (assigned on its first use as <see cref="BindDFlashRing"/>
        /// assigns it; null: the bound ring): each draft layer's K and V and its write frontier, with the bound ring left as
        /// it was. The trunk's own steps write a request's prompt rows into it (StandardDecoderModel.BlockDraft.cs, the
        /// followed prompt).
        /// </summary>
        private protected (Tensor[] K, Tensor[] V, int Filled, Tensor Carry) DFlashRingOf(string requestId)
        {
            if (requestId == null || !_hasDFlash || _dflashRingK == null)
                return (_dflashRingK, _dflashRingV, _dflashRingFilled, EnsureDFlashRingCarry());
            EnsureDFlashRingObjects();
            DFlashRing previous = _dflashActiveRing;
            BindDFlashRing(requestId);
            (Tensor[] K, Tensor[] V, int Filled, Tensor Carry) ring = (_dflashRingK, _dflashRingV, _dflashRingFilled, EnsureDFlashRingCarry());
            SelectDFlashRing(previous);
            return ring;
        }

        /// <summary>The bound ring's feature row [<see cref="DFlashConfig.FeatureSize"/>] a followed prompt's last step writes
        /// (its carry), allocated on first use; null without a drafter.</summary>
        private Tensor EnsureDFlashRingCarry()
        {
            if (!_hasDFlash || _dflash == null)
                return null;
            if (_dflashRingCarry == null)
            {
                _dflashRingCarry = new Tensor(_allocator, DType.Float32, _dflash.FeatureSize);
                Ops.Fill(_dflashRingCarry, 0f);
            }
            return _dflashRingCarry;
        }

        /// <summary>The bound ring's carry for a draft at <paramref name="position"/>: the feature row of position - 1 that
        /// a followed prompt's last step wrote, when nothing wrote the ring since (its frontier is still the prompt's end);
        /// else null. The tensor's device copy is ahead of its host one.</summary>
        private protected Tensor DFlashHeldCarry(int position)
            => _hasDFlash && !DFlashDraftingDeclined && _dflashRingCarry != null && position > 0
               && _dflashRingCarryEnd == position && _dflashRingFilled == position ? _dflashRingCarry : null;

        // The bound ring's carry and the position past the row it holds (-1: none); see DFlashRing.Carry.
        private Tensor _dflashRingCarry;
        private int _dflashRingCarryEnd = -1;

        /// <summary>A graph on the device wrote rows <paramref name="startPos"/> .. + <paramref name="n"/> of
        /// <paramref name="requestId"/>'s ring (null: the bound ring): its frontier and valid run, as the injects record
        /// theirs (<see cref="NoteDFlashRingWrite"/>), and its host copy behind; the bound ring is left as it was.</summary>
        private protected void NoteDFlashRingWriteOf(string requestId, int startPos, int n, bool carried = false)
        {
            DFlashRing previous = null;
            if (requestId != null && _hasDFlash && _dflashRingK != null)
            {
                EnsureDFlashRingObjects();
                previous = _dflashActiveRing;
                BindDFlashRing(requestId);
            }
            NoteDFlashRingWrite(startPos, n);
            // The write also left its last row's features in the ring's carry (a followed prompt's step).
            if (carried)
                _dflashRingCarryEnd = startPos + n;
            _dflashRingHostStale = true;
            if (previous != null)
                SelectDFlashRing(previous);
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
            chosen.CarryEnd = -1;
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

        /// <summary>Tests: the rows of the drafter's KV ring.</summary>
        internal int DFlashRingRowsForTests => DFlashRingRows;

        /// <summary>
        /// Tests: the bound ring's rows of draft layer <paramref name="layer"/> for the positions it holds as context
        /// (<paramref name="first"/> up to the write frontier), host-current, in position order; each position's K (and
        /// V) is [kvHeads][headDim].
        /// </summary>
        internal (float[] K, float[] V) DFlashRingContextForTests(int layer, out int first)
        {
            EnsureDFlashRingHostSynchronized();
            int rows = _dflashRingRows, kvHeads = _dflash.NumKVHeads, hd = _dflash.HeadDim;
            first = Math.Max(Math.Max(0, _dflashRingFilled - rows), _dflashRingValidFrom);
            int lo = first, count = Math.Max(0, _dflashRingFilled - lo);
            return (Gather(_dflashRingK[layer]), Gather(_dflashRingV[layer]));

            float[] Gather(Tensor ring)
            {
                float[] all = ring.GetElementsAsFloat(kvHeads * rows * hd);
                var context = new float[(long)count * kvHeads * hd];
                for (int i = 0; i < count; i++)
                    for (int h = 0; h < kvHeads; h++)
                        Array.Copy(all, ((long)h * rows + (lo + i) % rows) * hd, context, ((long)i * kvHeads + h) * hd, hd);
                return context;
            }
        }

        private void DisposeDFlashRingPool()
        {
            // Every ring's carry: the bound ring's is in the fields, the others' in their objects.
            if (_dflashActiveRing != null)
                _dflashActiveRing.Carry = _dflashRingCarry;
            else
                _dflashRingCarry?.Dispose();
            _dflashModelRing?.Carry?.Dispose();
            if (_dflashRingPool != null)
                foreach (var ring in _dflashRingPool)
                    ring.Carry?.Dispose();
            _dflashRingCarry = null;
            _dflashRingCarryEnd = -1;
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
