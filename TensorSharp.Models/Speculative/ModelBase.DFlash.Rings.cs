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
    /// Per-request DFlash KV rings, and the contexts they keep past their requests.
    ///
    /// The drafter's ring is its context: the last ringRows committed positions of
    /// ONE sequence, slot p % ringRows. With a single ring per model, requests that
    /// take turns speculating overwrite each other's rows at colliding slots, and a
    /// batched verify that drafts for several sequences in one step cannot share it
    /// at all. Each request therefore binds a ring of its own: one no request holds and
    /// that keeps no context (the model's own first), else a new pooled one (beside kept
    /// contexts only where the model gives kept rings back when short:
    /// <see cref="RoomForDFlashRing"/>), else the context kept the longest ago, and only for
    /// a request that speculates, the least recently used running request's. A ring a
    /// request takes anew restarts its valid run: nothing another sequence wrote is read as
    /// context.
    ///
    /// The context belongs to its sequence, not to the request that wrote it: a finished
    /// request's ring keeps its rows (<see cref="KeepDFlashRing"/>), and the runtime hands
    /// it to a later request whose tokens begin with the ones it covers - the next turn of
    /// the conversation, over the cache it reuses (<see cref="ContinueDFlashRing"/>; the
    /// runtime matches the tokens). Before, the model's own ring went to whichever request
    /// came next with its rows as they were: a turn that reused a conversation's whole
    /// prompt drafted over the rows of the last request that had speculated, another
    /// conversation's at positions the two shared (issue #376: 51-60% acceptance against the
    /// direct engine's 93-98%, whose ring lived in the sequence's slot). Pooled rings are
    /// recycled, and given back only when the device is short (<see cref="ReleaseIdleDFlashRings"/>),
    /// with the graphs bound to them: the fused drafter graphs are cached by the ring's host
    /// pointer, so a freed ring whose address came back would replay a graph over a
    /// released device copy.
    /// </summary>
    public abstract partial class ModelBase
    {
        private sealed class DFlashRing
        {
            public Tensor[] K, V;
            public int Filled, ValidFrom, High;
            public bool HostStale;
            public string Owner;
            /// <summary>The finished request whose sequence's context the ring keeps for a later request that continues it
            /// (<see cref="KeepDFlashRing"/>); null when it keeps none. Only a ring no request holds keeps one.</summary>
            public string KeptFor;
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

        /// <summary>Rings a later request continued (<see cref="ContinueDFlashRing"/>; tests check the context moved).</summary>
        internal int DFlashRingsContinued { get; private set; }

        /// <summary>
        /// Point the drafter at <paramref name="requestId"/>'s ring: its own, else one assigned on its first bind (see the
        /// class's summary). Null selects the model's own ring without claiming it. Cheap when the ring is already bound.
        /// </summary>
        internal void BindDFlashRing(string requestId) => TryBindDFlashRing(requestId, mayTake: true);

        /// <summary><see cref="BindDFlashRing"/>; with <paramref name="mayTake"/> false a request without a ring of its own
        /// takes one only where no running request loses its own (false, nothing bound, when none is free).</summary>
        private bool TryBindDFlashRing(string requestId, bool mayTake)
        {
            if (!_hasDFlash || _dflashRingK == null)
                return false;
            EnsureDFlashRingObjects();
            DFlashRing target = requestId == null ? _dflashModelRing : FindOrAssignDFlashRing(requestId, mayTake);
            if (target == null)
                return false;
            target.LastUsed = ++_dflashRingClock;
            SelectDFlashRing(target);
            return true;
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
            _dflashActiveRing.High = _dflashRingHigh;
            _dflashActiveRing.HostStale = _dflashRingHostStale;
            _dflashActiveRing.Carry = _dflashRingCarry;
            _dflashActiveRing.CarryEnd = _dflashRingCarryEnd;
            _dflashRingK = target.K;
            _dflashRingV = target.V;
            _dflashRingFilled = target.Filled;
            _dflashRingValidFrom = target.ValidFrom;
            _dflashRingHigh = target.High;
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
        /// assigns it, with <paramref name="mayTake"/> as <see cref="TryBindDFlashRing"/> takes it; null: the bound ring):
        /// each draft layer's K and V and its write frontier, with the bound ring left as it was; null tensors when the
        /// request gets none. The trunk's own steps write a request's prompt rows into it (StandardDecoderModel.BlockDraft.cs,
        /// the followed prompt).
        /// </summary>
        private protected (Tensor[] K, Tensor[] V, int Filled, Tensor Carry) DFlashRingOf(string requestId, bool mayTake = true)
        {
            if (requestId == null || !_hasDFlash || _dflashRingK == null)
                return (_dflashRingK, _dflashRingV, _dflashRingFilled, EnsureDFlashRingCarry());
            EnsureDFlashRingObjects();
            DFlashRing previous = _dflashActiveRing;
            if (!TryBindDFlashRing(requestId, mayTake))
                return (null, null, 0, null);
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

        /// <summary>Return <paramref name="requestId"/>'s ring to the pool, keeping no context (a ring kept for it too).</summary>
        internal void ReleaseDFlashRing(string requestId)
        {
            // Without ring objects (one request at a time) the decline lives on the model; the request is over.
            _dflashModelDraftingDeclined = false;
            if (requestId == null)
                return;
            foreach (DFlashRing ring in AllDFlashRings())
            {
                if (string.Equals(ring.KeptFor, requestId, StringComparison.Ordinal))
                    ring.KeptFor = null;
                if (!string.Equals(ring.Owner, requestId, StringComparison.Ordinal))
                    continue;
                if (ReferenceEquals(ring, _dflashActiveRing) && !ReferenceEquals(ring, _dflashModelRing))
                    BindDFlashRing(null);
                ring.Owner = null;
                ring.DraftingDeclined = false;
            }
        }

        /// <summary>
        /// <paramref name="requestId"/> is over: its ring keeps the context it holds of the request's sequence (the positions
        /// before its frontier) for a later request that continues the sequence (<see cref="ContinueDFlashRing"/>), until
        /// another request needs the ring. Returns that frontier, or 0 when the ring keeps nothing (none, nothing valid, a
        /// declined inject): then the ring goes back as <see cref="ReleaseDFlashRing"/> returns it.
        /// </summary>
        internal int KeepDFlashRing(string requestId)
        {
            DFlashRing ring = requestId == null ? null : OwnedDFlashRing(requestId);
            if (ring == null)
            {
                ReleaseDFlashRing(requestId);
                return 0;
            }
            var (filled, validFrom) = DFlashRunOf(ring);
            if (ring.DraftingDeclined || filled <= validFrom)
            {
                ReleaseDFlashRing(requestId);
                return 0;
            }
            _dflashModelDraftingDeclined = false;
            if (ReferenceEquals(ring, _dflashActiveRing) && !ReferenceEquals(ring, _dflashModelRing))
                BindDFlashRing(null);
            ring.Owner = null;
            ring.KeptFor = requestId;
            ring.LastUsed = ++_dflashRingClock;
            return filled;
        }

        /// <summary>
        /// <paramref name="requestId"/>, which holds no ring yet, takes the one kept for <paramref name="keptFor"/>
        /// (<see cref="KeepDFlashRing"/>) as the context of its first <paramref name="tokens"/> positions: the two sequences
        /// hold the same tokens there (the runtime matched them). Rows from <paramref name="tokens"/> on are not context, and
        /// neither are rows whose slots positions past it took (<see cref="MoveDFlashRunBack"/>); the carry stays only at an
        /// unchanged frontier. False when no ring keeps that context any more.
        /// </summary>
        internal bool ContinueDFlashRing(string requestId, string keptFor, int tokens)
        {
            if (requestId == null || keptFor == null || !_hasDFlash || _dflashRingK == null || tokens <= 0)
                return false;
            EnsureDFlashRingObjects();
            if (OwnedDFlashRing(requestId) != null)
                return false;
            DFlashRing ring = null;
            foreach (DFlashRing candidate in AllDFlashRings())
                if (candidate.Owner == null && string.Equals(candidate.KeptFor, keptFor, StringComparison.Ordinal))
                    ring = candidate;
            if (ring == null)
                return false;
            ring.Owner = requestId;
            ring.KeptFor = null;
            ring.DraftingDeclined = false;
            ring.LastUsed = ++_dflashRingClock;
            DFlashRing previous = _dflashActiveRing;
            SelectDFlashRing(ring);
            if (tokens < _dflashRingFilled)
            {
                MoveDFlashRunBack(tokens, tokens);
                _dflashRingFilled = tokens;
                _dflashRingValidFrom = Math.Min(_dflashRingValidFrom, tokens);
                _dflashRingCarryEnd = -1;
            }
            SelectDFlashRing(previous);
            OnDFlashRingOwnerChanged(ring.K);
            DFlashRingsContinued++;
            return true;
        }

        /// <summary>The ring <paramref name="requestId"/> holds, or null.</summary>
        private DFlashRing OwnedDFlashRing(string requestId)
        {
            foreach (DFlashRing ring in AllDFlashRings())
                if (string.Equals(ring.Owner, requestId, StringComparison.Ordinal))
                    return ring;
            return null;
        }

        /// <summary>The model's own ring, then the pooled ones.</summary>
        private IEnumerable<DFlashRing> AllDFlashRings()
        {
            if (_dflashModelRing != null)
                yield return _dflashModelRing;
            if (_dflashRingPool != null)
                foreach (DFlashRing ring in _dflashRingPool)
                    yield return ring;
        }

        /// <summary>A ring's frontier and valid run's start (the fields for the bound ring).</summary>
        private (int Filled, int ValidFrom) DFlashRunOf(DFlashRing ring)
            => ReferenceEquals(ring, _dflashActiveRing) ? (_dflashRingFilled, _dflashRingValidFrom) : (ring.Filled, ring.ValidFrom);

        /// <summary>The ring <paramref name="requestId"/> holds, or the one it takes (see the class's summary); null when it
        /// may not take one (<paramref name="mayTake"/> false and every ring held by a running request).</summary>
        private DFlashRing FindOrAssignDFlashRing(string requestId, bool mayTake)
        {
            if (OwnedDFlashRing(requestId) is { } own)
                return own;
            // A ring no request holds and that keeps no context: the model's own first.
            DFlashRing chosen = null;
            foreach (DFlashRing ring in AllDFlashRings())
                if (ring.Owner == null && ring.KeptFor == null)
                {
                    chosen = ring;
                    break;
                }
            // A new ring, while the pool may grow: beside the contexts finished requests keep, only where the model gives kept
            // rings back when short and the device has room (RoomForDFlashRing), and while fewer contexts are kept than the
            // model keeps finished sequences a request could continue (KeptContextsWorthARing): a ring kept for a
            // conversation whose cached prefix is gone is never continued. On Qwen3.8-27B DFlash2 the long-turn reproducer of
            // issue #383 had grown the pool to nine 81 MiB rings, most of them for the server's warm-up chats and for
            // conversations the prefix cache had evicted, beside two retained conversations.
            if (chosen == null && (_dflashRingPool?.Count ?? 0) < MaxDFlashRequestRings
                && (!AnyKeptDFlashRing() || (KeptDFlashRingCount() < KeptContextsWorthARing && RoomForDFlashRing(DFlashRingBytes))))
            {
                chosen = AllocateDFlashRing();
                (_dflashRingPool ??= new List<DFlashRing>()).Add(chosen);
            }
            // The context kept the longest ago.
            chosen ??= LeastRecentlyUsedDFlashRing(ring => ring.Owner == null);
            if (chosen == null)
            {
                // A running request's, the least recently used but the bound one: only a request that speculates takes it.
                if (!mayTake)
                    return null;
                chosen = LeastRecentlyUsedDFlashRing(ring => !ReferenceEquals(ring, _dflashActiveRing) && !ReferenceEquals(ring, _dflashModelRing));
                // Every ring is taken (more speculating requests than rings): the model's own, shared, as before.
                if (chosen == null)
                    return _dflashModelRing;
            }
            AssignDFlashRing(chosen, requestId);
            return chosen;
        }

        /// <summary>A ring a request takes anew: whatever it held is another sequence's, so its valid run restarts, and it
        /// drafts until its own inject fails; the bytes are left as they are.</summary>
        private void AssignDFlashRing(DFlashRing ring, string requestId)
        {
            ring.Owner = requestId;
            ring.KeptFor = null;
            ring.DraftingDeclined = false;
            if (ReferenceEquals(ring, _dflashActiveRing))
            {
                _dflashRingFilled = _dflashRingValidFrom = _dflashRingHigh = 0;
                _dflashRingCarryEnd = -1;
            }
            else
            {
                ring.Filled = ring.ValidFrom = ring.High = 0;
                ring.CarryEnd = -1;
            }
            OnDFlashRingOwnerChanged(ring.K);
        }

        /// <summary>Finished sequences the model keeps that a later request could continue (the prefix cache's retained
        /// conversations and checkpoints): kept drafter contexts past this many are not worth a new ring. Unbounded where the
        /// model does not say.</summary>
        private protected virtual int KeptContextsWorthARing => int.MaxValue;

        private int KeptDFlashRingCount()
        {
            int kept = 0;
            foreach (DFlashRing ring in AllDFlashRings())
                if (ring.Owner == null && ring.KeptFor != null)
                    kept++;
            return kept;
        }

        private bool AnyKeptDFlashRing()
        {
            foreach (DFlashRing ring in AllDFlashRings())
                if (ring.Owner == null && ring.KeptFor != null)
                    return true;
            return false;
        }

        private DFlashRing LeastRecentlyUsedDFlashRing(Func<DFlashRing, bool> eligible)
        {
            DFlashRing lru = null;
            foreach (DFlashRing ring in AllDFlashRings())
                if (eligible(ring) && (lru == null || ring.LastUsed < lru.LastUsed))
                    lru = ring;
            return lru;
        }

        /// <summary>A ring changed hands (a new request took it, or continued the context it kept): a model whose steps keep
        /// state over a ring hears of it (StandardDecoderModel: a recording that failed for the last owner is tried again).</summary>
        private protected virtual void OnDFlashRingOwnerChanged(Tensor[] ringK) { }

        /// <summary>Whether another ring of <paramref name="bytes"/> may be made beside the contexts finished requests keep: only
        /// where the model gives kept rings back when its device runs short (<see cref="ReleaseIdleDFlashRings"/>) and the device
        /// has room for it (a model that manages its device says; elsewhere no: a request then takes the context kept the
        /// longest ago, and the pool grows only with the requests that hold rings at once, as before contexts were kept).</summary>
        private protected virtual bool RoomForDFlashRing(long bytes) => false;

        /// <summary>A pooled ring is given up: a model whose steps bind rings forgets those over it.</summary>
        private protected virtual void OnDFlashRingDisposing(Tensor[] ringK) { }

        /// <summary>The device bytes of one ring (each draft layer's K and V) and its carry.</summary>
        private long DFlashRingBytes => _dflash == null ? 0
            : 2L * _dflash.NumLayers * _dflash.NumKVHeads * _dflashRingRows * _dflash.HeadDim * sizeof(float) + (long)_dflash.FeatureSize * sizeof(float);

        /// <summary>The drafter rings on the device (the model's own and the pooled ones) and their bytes (diagnostics).</summary>
        private protected (int Rings, long Bytes) DFlashRingHoldings
        {
            get
            {
                int rings = (_dflashModelRing != null || _hasDFlash ? 1 : 0) + (_dflashRingPool?.Count ?? 0);
                return (rings, rings * DFlashRingBytes);
            }
        }

        /// <summary>The device bytes <see cref="ReleaseIdleDFlashRings"/> gives back at most: the pooled rings no request holds
        /// (the model's own and the bound one stay).</summary>
        private protected long IdleDFlashRingBytes
        {
            get
            {
                long bytes = 0;
                if (_dflashRingPool != null)
                    foreach (DFlashRing ring in _dflashRingPool)
                        if (ring.Owner == null && !ReferenceEquals(ring, _dflashActiveRing)) bytes += DFlashRingBytes;
                return bytes;
            }
        }

        /// <summary>
        /// Gives back pooled rings no request holds, the least recently used first, until <paramref name="bytes"/> are freed
        /// (all of them by default): the contexts they keep go with them (a later turn of those conversations drafts without
        /// its context until its rows come back), and the graphs bound to them. Room a model's device is short of. The model's
        /// own ring and the bound one stay, and so does every ring a request holds - a request that continues a kept context
        /// takes it before its cache asks for room (the runtime continues it as the request adopts its cached prefix).
        /// Returns the bytes given back.
        /// </summary>
        private protected long ReleaseIdleDFlashRings(long bytes = long.MaxValue)
        {
            if (_dflashRingPool == null || _dflashRingPool.Count == 0 || bytes <= 0)
                return 0;
            var idle = new List<DFlashRing>();
            foreach (DFlashRing ring in _dflashRingPool)
                if (ring.Owner == null && !ReferenceEquals(ring, _dflashActiveRing))
                    idle.Add(ring);
            idle.Sort((a, b) => a.LastUsed.CompareTo(b.LastUsed));
            long freed = 0;
            using IDisposable toDriver = ReturningToDriver();
            foreach (DFlashRing ring in idle)
            {
                if (freed >= bytes)
                    break;
                OnDFlashRingDisposing(ring.K);
                foreach (Tensor t in ring.K) { InvalidateTensorDeviceCache(t); t.Dispose(); }
                foreach (Tensor t in ring.V) { InvalidateTensorDeviceCache(t); t.Dispose(); }
                if (ring.Carry != null) { InvalidateTensorDeviceCache(ring.Carry); ring.Carry.Dispose(); }
                _dflashRingPool.Remove(ring);
                freed += DFlashRingBytes;
            }
            // The fused drafters' persistent graphs are cached by ring addresses: none may replay over a freed one.
            if (freed > 0 && (_dflashArrays != null || _dspineState != null))
            {
                if (_dflashArrays != null) GgmlBasicOps.DFlashResetCaches();
                if (_dspineState != null) GgmlBasicOps.DSpineResetCaches();
            }
            return freed;
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

        /// <summary>Tests: <see cref="DFlashRingContextForTests(int, out int)"/> of the ring <paramref name="requestId"/> holds or
        /// that is kept for it, the bound ring left as it was; null K and V when there is none.</summary>
        internal (float[] K, float[] V) DFlashRingContextForTests(string requestId, int layer, out int first)
        {
            first = 0;
            DFlashRing ring = DFlashRingForTests(requestId);
            if (ring == null)
                return (null, null);
            DFlashRing previous = _dflashActiveRing;
            SelectDFlashRing(ring);
            try { return DFlashRingContextForTests(layer, out first); }
            finally { SelectDFlashRing(previous); }
        }

        /// <summary>Tests: the valid run (its first position and the frontier) of the ring <paramref name="requestId"/> holds or
        /// that is kept for it, or null.</summary>
        internal (int ValidFrom, int Filled)? DFlashRingRunForTests(string requestId)
        {
            DFlashRing ring = DFlashRingForTests(requestId);
            if (ring == null)
                return null;
            var (filled, validFrom) = DFlashRunOf(ring);
            return (Math.Max(Math.Max(0, filled - _dflashRingRows), validFrom), filled);
        }

        /// <summary>Tests: how many rings the model holds (its own and the pooled ones).</summary>
        internal int DFlashRingCountForTests => _dflashModelRing == null ? (_hasDFlash ? 1 : 0) : 1 + (_dflashRingPool?.Count ?? 0);

        private DFlashRing DFlashRingForTests(string requestId)
        {
            if (requestId == null || _dflashModelRing == null)
                return null;
            foreach (DFlashRing ring in AllDFlashRings())
                if (string.Equals(ring.Owner, requestId, StringComparison.Ordinal) || string.Equals(ring.KeptFor, requestId, StringComparison.Ordinal))
                    return ring;
            return null;
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
