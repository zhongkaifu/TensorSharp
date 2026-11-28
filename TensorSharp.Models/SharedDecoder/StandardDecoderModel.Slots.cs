// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    /// <summary>
    /// Continuous batching for every shared-runtime family, on every local placement, over
    /// device-resident per-request slots. A slot is one request's whole decoder state: its
    /// attention cache rows on each (layer, rank) device, in the configured KV dtype, and its
    /// recurrent states. The model's active fields hold the bound slot, so a lone request runs
    /// the exact single-sequence kernels; binding another request swaps references, never bytes.
    /// A batched step packs every scheduled row of every slot into one activation: embedding,
    /// norms, projections, the FFN or experts and the head run once over all rows, and only
    /// attention (and a recurrent mixer) runs per sequence, on that sequence's slot, with the
    /// same device kernels a single request uses.
    /// </summary>
    public abstract partial class StandardDecoderModel : IBatchedPagedModel
    {
        /// <summary>One request's decoder state.</summary>
        internal sealed class Slot
        {
            public Tensor[][] K;   // [layer][rank]; null rows for layers another node owns or recurrent layers
            public Tensor[][] V;
            public int Capacity;
            public int Length;
            public int Touched = int.MaxValue;     // positions its writes reached since its last zero (see _cacheTouched)
            public RecurrentState[][] Recurrent;   // [layer][rank]; hybrid stacks only
            public float[] Logits;                 // reused by batched steps
            public bool DeviceDirty;               // a graph wrote its rows/state in device memory
            public int DraftStart;                 // the draft layers attend to rows from here on
            public int RopeDelta;                  // rotary position = KV index + this past a position table
            public int VerifyRows;                 // a batched verify's window, its recurrent states awaiting the commit
            public int DraftFollowed = -1;         // the draft layer holds the trunk's own rows through here (see DraftFollowReady)
            public bool Spare;                     // made ahead of a request (no recordings bind it yet), see ReplenishSpareSlot
            public Tensor DraftCarry;              // the trunk's last followed row's stream (the next followed row's input)
            public long RetainedAt;                // when it entered the retained set (the oldest goes first under pressure)
            /// <summary>A block drafter's KV ring as a slot (StandardDecoderModel.BlockDraft.cs): no request owns it, it holds no
            /// recurrent state, and its steps' inputs are functions of their positions alone, so a step kept over it may run at
            /// any position (<see cref="GraphStep.TryRebase"/>).</summary>
            public bool DraftRing;
            /// <summary>For a <see cref="DraftRing"/>: how often the ring changed hands (a request took it, or continued the
            /// context it kept); a recording of a step over it that failed for one owner is tried again for the next.</summary>
            public int DraftRingOwners;
            /// <summary>A shared-row checkpoint (StandardDecoderModel.RowShares.cs): <see cref="K"/> and <see cref="V"/> are another
            /// slot's arrays, whose rows below <see cref="Length"/> that slot never writes again while this lasts; its
            /// <see cref="Recurrent"/> states are its own. Retained only: it never runs a request, and is never pooled.</summary>
            public bool SharesRows;
            /// <summary>A shared-row checkpoint whose rows past it another request has written since it was captured: one that
            /// took them over with a prompt of its own past the checkpoint (a branch), or continued the end state over them (its
            /// conversation's next turn). Those rows may still be continued, so an adopter of the checkpoint copies it instead
            /// of taking them over (StandardDecoderModel.RowShares.cs).</summary>
            public bool RowsPastTaken;
        }

        private static long s_retainSerial;

        /// <summary>Rows of a batched step: sequence s owns rows [QueryStart[s], QueryStart[s] + Counts[s])
        /// and continues its slot from position Starts[s].</summary>
        private sealed class SlotBatch
        {
            public required int NumSequences { get; init; }
            public required int[] QueryStart { get; init; }
            public required int[] Counts { get; init; }
            public required int[] Starts { get; init; }
            public required Slot[] Slots { get; init; }
        }

        // Parked slots keep their whole allocation; a few cover a burst of short requests.
        private const int SlotPoolMax = 4;

        /// <summary>
        /// The fewest cells a new slot (and a copy into one) holds: 512, and a whole <see cref="ReservationStepTokens"/> where
        /// recorded steps bind their slots (the CUDA executor). There a slot that grows records its steps again, and a chat's
        /// next turn - its history and its answer's reservation - outgrew a 512-cell slot: four Qwen3.8-27B conversations at
        /// once on an RTX PRO 4000 grew every second turn's slot (545-728 cells reserved), 13-30 ms of its first token, where
        /// the direct engine's slots never grew.
        /// </summary>
        private int FreshSlotTokens => GraphExecutor?.Capabilities.SlotBoundRecordings == true ? ReservationStepTokens : 512;

        /// <summary>Slot policies of executors whose recorded steps bind a slot's buffers (the CUDA executor,
        /// <see cref="TensorSharp.Graph.GraphCapabilities.SlotBoundRecordings"/>): a unified model's primary cache starts at a
        /// fresh slot's size whatever the window (<see cref="PrimaryCacheAllocationLength"/>).</summary>
        private bool SlotBoundPolicies => GraphExecutor?.Capabilities.SlotBoundRecordings == true || SlotBoundPoliciesForTests;

        /// <summary>Tests run the slot-bound policies on another executor (the managed CPU one). Initialized from
        /// <see cref="SlotBoundPoliciesOnFlowForTests"/>, so a decision the constructor makes (the primary's size) follows it.</summary>
        internal bool SlotBoundPoliciesForTests { get; set; } = SlotBoundPoliciesOnFlowForTests.Value;

        /// <summary>Tests: the models constructed on this flow take the slot-bound policies from their construction on.</summary>
        internal static readonly System.Threading.AsyncLocal<bool> SlotBoundPoliciesOnFlowForTests = new();

        /// <summary>Cells a grown slot keeps past what it needs (see <see cref="EnsureSlotCapacity"/>).</summary>
        private const int SlotGrowthHeadroomTokens = 4096;

        /// <summary>The step a request's reservation is rounded up to when a slot grows to it, so a pooled slot serves
        /// the next request of about the same size without growing again.</summary>
        private const int ReservationStepTokens = 1024;

        /// <summary>Slot capacities are whole multiples of this many cells, up to the context limit. A decode or verify
        /// row attends the padded cache window, which the capacity caps, and ggml-cuda groups a KV head's queries only
        /// over K whose length is a multiple of 256: a one-shot request's exact 3,736-cell reservation (a 3.6K prompt
        /// and 128 new tokens) kept Qwen3.8 Flash Next's decode in the ungrouped flash-attention kernel, 133 us a
        /// layer against 26 (A40, nsys). Metal's 64-cell windows divide it.</summary>
        private const int SlotCapacityAlignment = 256;

        private int AlignedSlotCapacity(long cells)
            => (int)Math.Min(_maxContextLength, (Math.Max(1, cells) + SlotCapacityAlignment - 1) / SlotCapacityAlignment * SlotCapacityAlignment);

        /// <summary>Slots sized to the device's room (<see cref="SlotTarget"/>: capacity classes with headroom where memory allows,
        /// the largest capacity that fits, reservations fitted to memory, growth a tensor at a time, small spares, pooling only
        /// with room): the slot-bound policies on a device this model accounts (<see cref="DeviceRoomManaged"/>). Elsewhere slots
        /// hold their reservations, rounded, as before.</summary>
        private bool SizesSlotsToRoom => SlotBoundPolicies && DeviceRoomManaged;

        /// <summary>Whole 1,024s up to 4,096 cells, and four capacities an octave above (1.25, 1.5, 1.75 and 2 x a power of
        /// two), up to the context limit: the capacity a slot that must hold <paramref name="cells"/> is made or grown to where
        /// memory allows. A conversation's slots then come in few sizes (one per +12.5-25% of context), so its next turns fit
        /// the slot it had, and the pool's cached blocks of a size serve the next slot of the class.</summary>
        internal int SlotCapacityClass(long cells) => SlotCapacityClass(cells, _maxContextLength);

        /// <inheritdoc cref="SlotCapacityClass(long)"/>
        internal static int SlotCapacityClass(long cells, int maxContextLength)
        {
            long c = Math.Max(1, cells), capacity;
            if (c <= 4096)
                capacity = (c + 1023) / 1024 * 1024;
            else
            {
                long p = 4096;
                while (p * 2 < c) p *= 2;   // p < c <= 2p
                long q = p / 4;
                capacity = c <= p + q ? p + q : c <= p + 2 * q ? p + 2 * q : c <= p + 3 * q ? p + 3 * q : 2 * p;
            }
            return (int)Math.Min(maxContextLength, capacity);
        }

        /// <summary>The capacity <paramref name="required"/> cells round to where no class with headroom fits: whole 1,024s,
        /// as a reservation was rounded.</summary>
        private int ReservationCapacity(long required)
            => AlignedSlotCapacity((Math.Max(1, required) + ReservationStepTokens - 1) / ReservationStepTokens * ReservationStepTokens);

        /// <summary>
        /// The capacity a slot that must hold <paramref name="required"/> cells, and would like <paramref name="wanted"/>
        /// (its request's reservation: prompt and generation budget), is made, copied or grown to on a device this model accounts
        /// (<see cref="SizesSlotsToRoom"/>), its room made by <see cref="EnsureDeviceRoom"/>; the first candidate that gets room:
        /// <list type="number">
        /// <item>the class of <paramref name="wanted"/> and an eighth more (at least 1,024 cells), from pooled slots, parked
        /// recordings and retained window buffers only: headroom never releases a conversation;</item>
        /// <item>the largest whole 1,024 at least <paramref name="required"/> that fits the room those stages reached, else the
        /// required cells rounded to 1,024, then to 256, where those stages make room for them;</item>
        /// <item>with <paramref name="mayEvict"/>, the required cells rounded to 1,024, releasing finished conversations, the
        /// oldest first, then the required cells rounded to 256; a conversation goes whole, so the room it leaves past those cells
        /// is then taken as candidates 1 and 2 take room (up to the class, releasing nothing more).</item>
        /// </list>
        /// 0 where none gets room. A growth (<paramref name="heldCapacity"/>, the slot's cells now) needs the added cells and one
        /// of its tensors, not a second slot (<see cref="GrowSlotRolling"/>). A row's arithmetic does not depend on the capacity
        /// its slot gets: decode and verify rows split their keys by their length on every path (the flash kernels by their
        /// bucket, the GQA decode partitions by their live length), so a request's tokens are the same in a slot of any size.
        /// </summary>
        private int SlotTarget(int required, int wanted, ReadOnlySpan<Slot> keep, string what, int heldCapacity = 0, bool mayEvict = true,
            bool headroom = true)
        {
            int roomy = RoomySlotTarget(required, wanted, keep, what, heldCapacity, headroom);
            if (roomy > 0 || !mayEvict) return roomy;
            return TightSlotTarget(required, wanted, keep, what, heldCapacity, headroom);
        }

        /// <summary>Candidates 1 and 2 of <see cref="SlotTarget"/> (no conversation released); 0 where neither fits.</summary>
        private int RoomySlotTarget(int required, int wanted, ReadOnlySpan<Slot> keep, string what, int heldCapacity, bool headroom = true)
        {
            int floor = AlignedSlotCapacity(required);
            long w = Math.Max(required, wanted);
            int target = Math.Max(floor, headroom ? SlotCapacityClass(w + Math.Max(1024, w / 8)) : ReservationCapacity(w));
            if (target <= heldCapacity) return heldCapacity;
            if (EnsureDeviceRoom(SlotTargetBytes(target, heldCapacity), keep, RoomUse.Slot, mayEvict: false, what, out long room,
                    CacheBlockBytes(target)))
                return target;
            // The largest whole 1,024 the room reached holds, from the bytes a cell and a slot take (no probing).
            long perCell = CacheBytesPerCell();
            if (perCell <= 0 || room == long.MaxValue) return 0;
            long total = DeviceMemory().Total;
            long avail = room - RoomNeed(0, total, RoomUse.Slot) - (heldCapacity > 0 ? CacheTensorBytes(target) : SlotBytesEstimate(0));
            long cells = (heldCapacity > 0 ? heldCapacity : 0) + Math.Max(0, avail) / perCell;
            int fit = (int)Math.Min(target - 1, cells / ReservationStepTokens * ReservationStepTokens);
            int least = ReservationCapacity(required);
            if (fit >= least && fit > heldCapacity
                && EnsureDeviceRoom(SlotTargetBytes(fit, heldCapacity), keep, RoomUse.Slot, mayEvict: false, what, CacheBlockBytes(fit)))
                return fit;
            // The required cells rounded to 1,024, then to 256 (a conversation is worth more than the slack), each asked of the
            // stages that release no conversation: the room they reached was measured for the class's need, and the retained
            // window buffers (stage 4) are given up only where they cover a need's shortfall.
            foreach (int capacity in floor < least ? new[] { least, floor } : new[] { least })
                if (capacity > heldCapacity && capacity < target
                    && EnsureDeviceRoom(SlotTargetBytes(capacity, heldCapacity), keep, RoomUse.Slot, mayEvict: false, what, CacheBlockBytes(capacity)))
                    return capacity;
            return 0;
        }

        /// <summary>Candidate 3 of <see cref="SlotTarget"/>, and a new request's slot where no pooled or retained slot serves it
        /// (<see cref="NewSlotWithRoom"/>): the required cells rounded to 1,024, then to 256, finished conversations released for
        /// them, then, with <paramref name="takeRoom"/>, the room the released ones leave past those cells taken as candidates 1
        /// and 2 take room (up to the class of <paramref name="wanted"/> and an eighth, releasing nothing more); 0 where neither
        /// gets room.</summary>
        private int TightSlotTarget(int required, int wanted, ReadOnlySpan<Slot> keep, string what, int heldCapacity, bool headroom = true,
            bool takeRoom = true)
        {
            foreach (int capacity in new[] { ReservationCapacity(required), AlignedSlotCapacity(required) })
            {
                if (capacity <= heldCapacity) return heldCapacity;
                if (EnsureDeviceRoom(SlotTargetBytes(capacity, heldCapacity), keep, RoomUse.Slot, mayEvict: true, what, CacheBlockBytes(capacity)))
                    // A conversation goes whole: the room it leaves past the required cells, taken now. Left free, the request's
                    // reservation took it at its next step, or its conversation's next turn at that turn's first, by growing the
                    // slot a tensor at a time inside the step (Qwen3.8-27B on an RTX PRO 4500: a 47K clone made at 48,128 cells,
                    // then grown to 56,320 in its first decode step, 107 ms; a new 69K request's slot made at 69,632 cells, then
                    // grown to 81,920 in its next turn's 1,044-row prompt step, 613-616 ms against 510-511 ms).
                    return takeRoom ? Math.Max(capacity, RoomySlotTarget(capacity, wanted, keep, what, heldCapacity, headroom)) : capacity;
            }
            return 0;
        }

        /// <summary>The bytes a slot of <paramref name="capacity"/> cells takes beyond what it holds: a fresh one's
        /// (<see cref="SlotBytesEstimate"/>), or for a slot of <paramref name="heldCapacity"/> cells grown a tensor at a time the
        /// added cells and the largest of its new tensors.</summary>
        private long SlotTargetBytes(int capacity, int heldCapacity)
            => heldCapacity <= 0 ? SlotBytesEstimate(capacity) : CacheBytesPerCell() * (capacity - heldCapacity) + CacheTensorBytes(capacity);

        /// <summary>The largest of a slot's cache tensors at <paramref name="capacity"/> cells (0 where unknown).</summary>
        private long CacheTensorBytes(int capacity)
        {
            Tensor[][] k = _activeKey == null ? _cacheK : _primary?.K;
            Tensor[][] v = _activeKey == null ? _cacheV : _primary?.V;
            int cells = _activeKey == null ? _cacheCapacity : _primary?.Capacity ?? 0;
            if (cells <= 0) return 0;
            long largest = 0;
            foreach (Tensor t in AllCacheTensors(k).Concat(AllCacheTensors(v)))
                largest = Math.Max(largest, t.Storage.ByteLength / cells * capacity);
            return largest;
        }

        /// <summary>A request's reservation as its slot was fitted to it (<see cref="EnsureSequenceSlot(string, int, int)"/>), the
        /// most its later steps grow the slot towards; keyed by request id (slot objects are re-minted).</summary>
        private Dictionary<string, int> _fittedReservations;

        /// <summary>What a step of <paramref name="requestId"/> that needs <paramref name="required"/> cells would like its slot to
        /// hold: its fitted reservation where it has one.</summary>
        private int WantedCells(string requestId, int required)
            => requestId != null && _fittedReservations != null && _fittedReservations.TryGetValue(requestId, out int reserve)
                ? Math.Max(required, reserve) : required;

        /// <summary>Tests: <see cref="SlotTarget"/> for a slot beside the running ones.</summary>
        internal int SlotTargetForTests(int required, int wanted) => SlotTarget(required, wanted, RunningSlots(), "a slot (test)");

        /// <summary>Tests: the free bytes a fresh slot of <paramref name="capacity"/> cells wants (its bytes beside the headroom
        /// and a prompt step's or recordings' room).</summary>
        internal long SlotRoomNeedForTests(int capacity)
            => RoomNeed(SlotBytesEstimate(capacity), DeviceMemory(fresh: true).Total, RoomUse.Slot);

        /// <summary>Tests: the cells a request's slot holds (null where it has none).</summary>
        internal int? SlotCapacityForTests(string requestId) => TryGetSlot(requestId, out Slot slot) ? slot.Capacity : null;

        /// <summary>Tests, probes: the positions a request's slot holds (null where it has none).</summary>
        internal int? SlotLengthForTests(string requestId) => TryGetSlot(requestId, out Slot slot) ? slot.Length : null;

        /// <summary>Tests, probes: the cells of the active cache (the bound request's slot, else the primary).</summary>
        internal int ActiveCacheCapacityForTests => _cacheCapacity;

        /// <summary>Tests: the bytes a cell of a slot's caches takes (every entry's K and V).</summary>
        internal long CacheBytesPerCellForTests => CacheBytesPerCell();

        /// <summary>Tests: the capacities of the pooled slots.</summary>
        internal int[] PooledCapacitiesForTests => _slotPool?.Select(p => p.Capacity).ToArray() ?? Array.Empty<int>();

        /// <summary>Tests: how many cache tensors a rolling growth allocates before the next allocation fails (null: none fails).</summary>
        internal int? RollingGrowthFailAfterForTests { get; set; }

        /// <summary>Bytes of cache rows zero-filled when slots were made or grown (tests: a growth zeroes only the rows past the
        /// slot's length).</summary>
        internal long CacheZeroFillBytes { get; private set; }

        private Dictionary<string, Slot> _slots;            // requests with their own slot

        /// <summary>Probes (eng/GdnDriftProbe): <paramref name="requestId"/>'s recurrent state of <paramref name="layer"/> (rank 0),
        /// or null (no slot, or not a recurrent layer).</summary>
        internal RecurrentState RecurrentStateForProbe(string requestId, int layer)
            => _slots != null && _slots.TryGetValue(requestId, out Slot slot) && slot.Recurrent != null && layer >= 0 && layer < slot.Recurrent.Length
                ? slot.Recurrent[layer]?[0] : null;
        private Dictionary<string, Slot> _retainedSlots;    // finished requests kept for a continuation
        private List<Slot> _slotPool;
        private string _activeKey;                          // request whose slot is in the active fields
        private Slot _primary;                              // the single-stream cache while a request is bound

        /// <summary>Batched steps (two or more requests in one forward) served; tests check the engine used them.</summary>
        internal int BatchedForwardCount { get; private set; }

        // ---------------------------------------------------------------- capability

        /// <summary>A multi-node pipeline runs the serialized stage protocol, which binds no request.</summary>
        public bool SupportsPerSequenceFusedForward => _cacheK != null && LayerSplitGroup == null;

        /// <summary>The paged-storage forward is gone: requests run on their own device-resident slots.</summary>
        public bool BatchedForwardAvailable => false;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
            => throw new NotSupportedException(
                $"{Config.Architecture} serves concurrent requests from per-request slots, not paged storage.");

        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _slots != null && _slots.ContainsKey(requestId);

        public bool SupportsRetainedFusedCache => SupportsPerSequenceFusedForward;

        // ---------------------------------------------------------------- active fields

        private Slot SnapshotActive()
        {
            SettlePendingVerify();
            return SnapshotActiveFields();
        }

        private Slot SnapshotActiveFields() => new()
        {
            K = _cacheK,
            V = _cacheV,
            Capacity = _cacheCapacity,
            Length = _cacheSeqLen,
            Touched = _cacheTouched,
            Recurrent = _recurrentPrimary,
            DeviceDirty = _cacheDeviceDirty,
            DraftStart = _draftStart,
            RopeDelta = _ropeDelta,
            DraftFollowed = _draftFollowed,
            DraftCarry = _draftCarry,
        };

        private void LoadActive(Slot slot)
        {
            _cacheK = slot.K;
            _cacheV = slot.V;
            _cacheCapacity = slot.Capacity;
            _cacheSeqLen = slot.Length;
            _cacheTouched = slot.Touched;
            _recurrentPrimary = slot.Recurrent;
            _cacheDeviceDirty = slot.DeviceDirty;
            _draftStart = slot.DraftStart;
            _ropeDelta = slot.RopeDelta;
            _draftFollowed = slot.DraftFollowed;
            _draftCarry = slot.DraftCarry;
        }

        // The active cache's draft-layer floor (see Slot.DraftStart).
        private int _draftStart;

        // A whole-forward graph wrote the active cache/state in device memory.
        private bool _cacheDeviceDirty;

        /// <summary>Bring a slot's device-written rows and states back to its host tensors, before
        /// host code (the per-op path, snapshots, truncation, growth) reads or rewrites them.</summary>
        private void SyncToHost(Slot slot)
        {
            if (slot == null || !slot.DeviceDirty) return;
            foreach (Tensor t in AllCacheTensors(slot.K)) SyncTensorHostCache(t);
            foreach (Tensor t in AllCacheTensors(slot.V)) SyncTensorHostCache(t);
            if (slot.Recurrent != null)
                foreach (RecurrentState[] layer in slot.Recurrent)
                    if (layer != null)
                        foreach (RecurrentState state in layer)
                            if (state != null)
                                foreach (Tensor t in state.HostTensors) SyncStateTensor(t);
            slot.DeviceDirty = false;
        }

        /// <summary>Sync a state tensor (possibly a view) through its own pointer: device copies are
        /// keyed by the exact host address a graph or kernel bound.</summary>
        private void SyncStateTensor(Tensor t)
        {
            if (t == null || GraphExecutor is not { Capabilities.BindsHostPointers: true } executor) return;
            executor.SyncToHost(TensorComputePrimitives.GetStoragePointer(t), t.ElementCount() * t.ElementType.Size());
        }

        private void SyncActiveToHost()
        {
            if (!_cacheDeviceDirty) return;
            SyncToHost(SnapshotActive());
            _cacheDeviceDirty = false;
        }

        /// <summary>Write the active fields back into the bound request's slot and reinstate the
        /// primary, so every slot object is current (batched steps read slots, not the fields).</summary>
        public void RestorePrimaryCache()
        {
            if (_activeKey == null) return;
            _slots[_activeKey] = SnapshotActive();
            _activeKey = null;
            if (_primary != null)
            {
                LoadActive(_primary);
                _primary = null;
            }
        }

        // ---------------------------------------------------------------- slot lifecycle

        /// <summary>An empty slot for <paramref name="reserveTokens"/> tokens: a pooled one, whose caches and states
        /// keep their device copies, so the graphs cached over it replay for its next request; else a new one of that
        /// size.</summary>
        private Slot NewSlot(int reserveTokens = 0)
        {
            if (TakePooledSlot(reserveTokens, mustFit: false) is { } pooled)
            {
                if (!ResetStatesInPlace(pooled.Recurrent))
                    ClearGraphCache();
                return pooled;
            }
            return FreshSlot(AlignedSlotCapacity(Math.Max(Math.Max(1, reserveTokens), Math.Min(FreshSlotTokens, Math.Max(1, _maxContextLength)))));
        }

        /// <summary>A new empty slot of <paramref name="capacity"/> cells (zeroed caches, reset states).</summary>
        private Slot FreshSlot(int capacity)
        {
            var slot = new Slot { Capacity = capacity, Recurrent = _anyRecurrent ? NewRecurrentStates() : null };
            (slot.K, slot.V) = AllocateCaches(capacity);
            PrimeDeviceCopies(slot, overwritten: false);
            if (_slotStateBytes < 0) MeasureSlotBytes(slot);
            return slot;
        }

        // ---------------------------------------------------------------- device-memory admission

        // A slot's cache bytes per token and its states' bytes, measured on the first slot (-1 until then).
        private long _slotBytesPerToken = -1, _slotStateBytes = -1;

        // Each device's share of a slot (its caches' bytes per token and its states' bytes), by its place in the mesh: a layer
        // split's stage or a tensor-parallel rank holds its layers' or heads' part; one device holds all of it.
        private (int Stage, int Rank, long PerToken, long States)[] _slotShares;

        /// <summary>Whether idle slots' device copies are mirrors of host tensors that <see cref="DropIdleSlotDeviceCopies"/> can
        /// drop: one device, a local layer split's stages or per-rank tensor-parallel graphs' ranks (the native sync and
        /// invalidation visit every device's cache, so each stage's or rank's copies go wherever they live), on an executor
        /// that binds host bytes into separate device copies.</summary>
        private bool DropsIdleSlotMirrors
            => KeepsDeviceCopies && LayerSplitGroup == null && (_mesh.TensorParallel == 1 || RankedSlotCopies)
               && GraphExecutor is { Capabilities: { BindsHostPointers: true, SeparateDeviceCopies: true } };

        /// <summary>Beside a slot's caches and states: its drafter buffers and its share of the graphs' memory (the margin
        /// Qwen4ExpModel keeps for a holder on Metal).</summary>
        private const long SlotExtraBytes = 256L << 20;

        private void MeasureSlotBytes(Slot slot)
        {
            var shares = new SortedDictionary<(int Stage, int Rank), (long Cache, long States)>();
            void Add(int stage, int rank, long cache, long states)
            {
                shares.TryGetValue((stage, rank), out var share);
                shares[(stage, rank)] = (share.Cache + cache, share.States + states);
            }
            for (int e = 0; e < CacheEntries; e++)
                for (int r = 0; r < _mesh.TensorParallel; r++)
                {
                    long bytes = (slot.K?[e] is { } k && r < k.Length && k[r] != null ? k[r].Storage.ByteLength : 0)
                        + (slot.V?[e] is { } v && r < v.Length && v[r] != null ? v[r].Storage.ByteLength : 0);
                    if (bytes <= 0) continue;
                    // A sliding-window ring costs its whole ring a slot, whatever the slot's length.
                    if (RingOf(e) is > 0 and var ring)
                        Add(CacheStage(e), r, 0, bytes / Math.Max(1, CacheCells(e, slot.Capacity)) * ring);
                    else
                        Add(CacheStage(e), r, bytes, 0);
                }
            if (slot.Recurrent != null)
                for (int l = 0; l < slot.Recurrent.Length; l++)
                    if (slot.Recurrent[l] is { } layer)
                        for (int r = 0; r < layer.Length; r++)
                            if (layer[r] is { } state)
                                Add(_mesh.StageOf(l), r, 0, state.StateBytes > 0 ? state.StateBytes
                                    : state.CheckpointTensors.Sum(t => t.ElementCount() * t.ElementType.Size()));
            int capacity = Math.Max(1, slot.Capacity);
            _slotShares = shares.Select(kv => (kv.Key.Stage, kv.Key.Rank, kv.Value.Cache / capacity, kv.Value.States)).ToArray();
            _slotBytesPerToken = shares.Values.Sum(v => v.Cache) / capacity;
            _slotStateBytes = shares.Values.Sum(v => v.States);
        }

        /// <summary>Device bytes a slot of <paramref name="capacity"/> tokens takes, with its margin; -1 before the first
        /// slot was measured.</summary>
        private long SlotDeviceBytes(int capacity)
            => _slotStateBytes < 0 ? -1 : _slotBytesPerToken * capacity + _slotStateBytes + SlotExtraBytes;

        /// <summary>
        /// Whether device memory bounds how many requests hold slots: Metal's working set, one ggml_cuda GPU, or every GPU of
        /// a local mesh (tensor-parallel ranks, layer-split stages, on ggml_cuda or direct CUDA), each judged by its own share
        /// of a slot (<see cref="SlotDeficit"/>), as Qwen4ExpModel and Qwen35Model admit their holders. Without it the runtime
        /// allocated a slot for every concurrent request and kept every finished one: two Qwen3.8-Flash-Next conversations on
        /// a 48 GB Mac reached 38 GB wired, where Qwen4ExpModel stayed at 32 GB; an allocation past a CUDA GPU's memory aborts
        /// the process inside ggml. One direct-CUDA device accounts its memory itself (<see cref="DeviceRoomManaged"/>).
        /// </summary>
        private bool AdmitsSlotsByDeviceMemory => LayerSplitGroup == null && (_mesh.IsSingleDevice
            ? IsGgmlBackend && (_backend == BackendType.GgmlMetal || _backend == BackendType.GgmlCuda)
            : true);   // a mesh whose devices report no memory (host-memory or simulated devices) measures no deficit

        /// <summary>The device memory kept free beside the slots: the larger of 1 GB and 1/32 of it.</summary>
        private static long DeviceHeadroom(long totalBytes) => Math.Max(1L << 30, totalBytes / 32);

        /// <summary>The device's free and total bytes (the GGML backend's report; tests substitute their own).</summary>
        internal Func<(long Free, long Total)?> DeviceMemoryProbe { get; set; } = GgmlDeviceMemory;

        private static readonly Func<(long Free, long Total)?> GgmlDeviceMemory = static ()
            => GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) && total > 0 ? (free, total) : null;

        /// <summary>The bytes claimed ahead of use on the device the GGML backend reports (<see cref="GpuMemoryBudget.Reserve"/>,
        /// <see cref="GpuMemoryBudget.GgmlDevice"/>: a loaded vision tower's weights before their upload and the compute of the
        /// largest image it accepts), which a new slot leaves free beside the headroom. A test's probe sees only the
        /// reservations on the device it names (<see cref="ReservationDeviceForTests"/>), none by default.</summary>
        private long GgmlReservedBytes
        {
            get
            {
                (BackendType Backend, int Device)? device = ReservationDeviceForTests;
                if (device == null && ReferenceEquals(DeviceMemoryProbe, GgmlDeviceMemory))
                    device = (_backend, GpuMemoryBudget.GgmlDevice);
                return device is { } d ? GpuMemoryBudget.ReservedBytes(d.Backend, d.Device) : 0;
            }
        }

        /// <summary>What the device lacks for <paramref name="needBytes"/> beside its headroom and the bytes reserved on it,
        /// after the kept graph arenas are released and then the idle slots' device copies dropped (0 when it fits or is not
        /// measured).</summary>
        private long DeviceDeficit(long needBytes)
        {
            if (needBytes <= 0 || DeviceMemoryProbe() is not { } memory) return 0;
            long reserved = GgmlReservedBytes;
            long Short((long Free, long Total) m) => Math.Max(0, needBytes + DeviceHeadroom(m.Total) + reserved - m.Free);
            long deficit = Short(memory);
            if (deficit > 0 && GgmlBasicOps.ReleaseKeptArenas() > 0 && DeviceMemoryProbe() is { } after)
                deficit = Short(after);
            if (deficit > 0 && DropIdleSlotDeviceCopies() > 0 && DeviceMemoryProbe() is { } dropped)
                deficit = Short(dropped);
            return deficit;
        }

        /// <summary>
        /// What the device shortest of room lacks for one more slot of <paramref name="capacity"/> cells beside its headroom
        /// (0 when every device fits it or none is measured yet). One device: <see cref="DeviceDeficit"/>. A mesh: every device
        /// that holds part of a slot (<see cref="_slotShares"/>: a stage's layers, a rank's heads) against its own free memory,
        /// after the kept graph arenas are released and then the idle slots' device copies dropped. Before, a tensor-parallel or layer-split model gave every concurrent
        /// request a slot, and a GPU that filled aborted the process inside ggml.
        /// </summary>
        private long SlotDeficit(int capacity)
        {
            if (_mesh.IsSingleDevice) return DeviceDeficit(SlotDeviceBytes(capacity));
            if (_slotShares == null || _slotShares.Length == 0) return 0;
            long Worst()
            {
                long worst = 0;
                foreach (var (stage, rank, perToken, states) in _slotShares)
                {
                    if (MeshDeviceMemory(stage, rank) is not { } m) continue;
                    long need = perToken * capacity + states + SlotExtraBytes;
                    worst = Math.Max(worst, need + DeviceHeadroom(m.Total) - m.Free);
                }
                return worst;
            }
            long deficit = Worst();
            if (deficit > 0 && IsGgmlBackend && GgmlBasicOps.ReleaseKeptArenas() > 0)
                deficit = Worst();
            if (deficit > 0 && DropIdleSlotDeviceCopies() > 0)
                deficit = Worst();
            return deficit;
        }

        /// <summary>The free and total bytes of the mesh's device (<paramref name="stage"/>, <paramref name="rank"/>), or null where
        /// it has none of its own (host memory, simulated devices); tests stand in for it (<see cref="SlotDeviceMemoryForTests"/>).</summary>
        private (long Free, long Total)? MeshDeviceMemory(int stage, int rank)
        {
            if (SlotDeviceMemoryForTests?.Invoke(_mesh.GlobalDevice(stage, rank)) is { } stand) return stand;
            IAllocator allocator = _mesh.Allocator(stage, rank);
            if (allocator is IDeviceMemoryInfo memory)
                return !memory.SharesHostMemory && memory.TryGetDeviceMemory(out long free, out long total) && total > 0 ? (free, total) : null;
            // One GGML device keeps its own probe (DeviceMemoryProbe); a mesh's GGML devices are each asked in turn, the backend
            // reporting the calling thread's active device.
            if (_mesh.IsSingleDevice || allocator is not GgmlAllocator ggml || _backend != BackendType.GgmlCuda) return null;
            int previous = GgmlBasicOps.GetActiveRank();
            GgmlBasicOps.SetActiveRank(ggml.DeviceId);
            try
            {
                return GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) && total > 0 ? (free, total) : null;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previous);
            }
        }

        /// <summary>
        /// Drops the device copies of the slots no request runs on (finished requests kept for a continuation, pooled ones)
        /// where the executor keeps separate device copies: their host tensors hold every row and state (synced first), and
        /// a slot's next graph uploads what it binds (<see cref="DropsIdleSlotMirrors"/>). Kept, they filled the device: fifteen finished Qwen3.8-27B DFlash2
        /// requests held 6.6 GB of a 24 GB GPU, and four new ones found no memory for their first step. Returns the slots
        /// dropped.
        /// </summary>
        private int DropIdleSlotDeviceCopies()
        {
            if (!DropsIdleSlotMirrors) return 0;
            int dropped = 0;
            foreach (Slot slot in IdleSlots())
            {
                if (slot?.K == null) continue;
                SyncToHost(slot);
                foreach (Tensor t in AllCacheTensors(slot.K)) InvalidateTensorDeviceCache(t);
                foreach (Tensor t in AllCacheTensors(slot.V)) InvalidateTensorDeviceCache(t);
                if (slot.Recurrent != null)
                    foreach (RecurrentState[] layer in slot.Recurrent)
                        if (layer != null)
                            foreach (RecurrentState state in layer)
                                if (state != null)
                                    foreach (Tensor t in state.HostTensors) InvalidateTensorDeviceCache(t);
                dropped++;
            }
            if (dropped > 0) ClearGraphCache();
            return dropped;

            IEnumerable<Slot> IdleSlots()
            {
                if (_retainedSlots != null) foreach (Slot s in _retainedSlots.Values) yield return s;
                if (_slotPool != null) foreach (Slot s in _slotPool) yield return s;
            }
        }

        /// <summary>The prefix cache evicts finished requests' slots before a new request needs their room: what the device
        /// lacks for one more fresh slot (<see cref="ModelBase.QueryHostMemoryDeficitBytes"/>).</summary>
        /// <remarks>On a device this model manages (<see cref="DeviceRoomManaged"/>, one direct-CUDA device), what the device
        /// lacks of its headroom alone: at admission the incoming request may adopt the very payload a slot's room would evict
        /// (a conversation's next turn takes its finished slot), so a slot's or a prompt step's room is made when it is
        /// allocated (<see cref="EnsureDeviceRoom"/>), which knows what to keep.</remarks>
        protected override long DeviceMemoryDeficitBytes()
        {
            if (AdmitsSlotsByDeviceMemory) return SlotDeficit(FreshSlotTokens);
            if (!DeviceRoomManaged) return 0;
            var (free, total) = DeviceMemory();
            return total <= 0 ? 0 : Math.Max(0, DeviceHeadroom(total) - free);
        }

        /// <summary><see cref="DeviceMemoryDeficitBytes"/> alone (tests).</summary>
        internal long DeviceSlotDeficitBytes => DeviceMemoryDeficitBytes();

        /// <summary>A new request's slot of <paramref name="capacity"/> tokens must fit on the device while other requests
        /// hold theirs; otherwise it waits for one of them to finish (<see cref="SequenceSlotUnavailableException"/>, which
        /// the scheduler re-queues). A lone request always gets its slot.</summary>
        private void EnsureDeviceRoomForSlot(string requestId, int capacity)
        {
            if (!AdmitsSlotsByDeviceMemory || _slots == null || _slots.Count == 0 || (_slotPool?.Count ?? 0) > 0) return;
            long need = SlotDeviceBytes(capacity);
            long deficit = SlotDeficit(capacity);
            if (deficit > 0)
                throw new SequenceSlotUnavailableException(
                    $"{Config.Architecture}: no device memory for another request's slot ({need / 1048576.0:F0} MB, " +
                    $"{deficit / 1048576.0:F0} MB short beside the headroom); {requestId} waits for a running request to finish.");
        }

        /// <summary>
        /// A fresh slot's graph-bound caches and states get their device copies zero-filled on the device, where the executor
        /// keeps separate device copies (ggml_cuda, ggml_vulkan): the first graph binding them then uploads nothing. Their host
        /// bytes are dead rows (past the slot's length) or a reset state's zeros; with <paramref name="overwritten"/>, a copy
        /// on the device writes every bound byte before anything reads it. Those uploads were most of a request's setup:
        /// Qwen3.5-9B on an RTX 2000 Ada uploaded 48-165 MB per graph at ~3 GB/s, twice a request (its slot, then its prefix
        /// checkpoint's), 31-65 ms each. A state whose reset is not all zeros keeps the upload. Each device primes its own
        /// share: a rank's heads under per-rank tensor-parallel graphs, a whole-layer stage's layers on its device. Stages had
        /// primed nothing: on two ggml_cuda GPUs every request's clone of a 3.4K-token prefix uploaded its whole capacity
        /// (Qwen3.5-9B, about 160 MB) before its first token, ~50 ms more than on one GPU.
        /// </summary>
        private void PrimeDeviceCopies(Slot slot, bool overwritten)
        {
            bool ranked = RankedSlotCopies, staged = _mesh.Stages > 1;
            if (!IsGgmlBackend || LayerSplitGroup != null || (_mesh.TensorParallel != 1 && !ranked)
                || GraphExecutor is not { Capabilities: { BindsHostPointers: true, SeparateDeviceCopies: true } })
                return;
            // Each by the bytes a graph binds it with (ggml's size of the leaf): a cache whole, a state's F32 view by its
            // own elements (the state view of a backing is not the whole storage). The caches and the states each share
            // one device allocation (GgmlBasicOps.ZeroDeviceCopies): a slot's ~80 tensors were as many allocations, 0.7 ms
            // each on ggml-vulkan. Apart, a slot that grows frees its old caches' buffer whole.
            var keys = new List<IntPtr>();
            var bytes = new List<long>();
            bool switches = ranked || staged;
            int previous = switches ? GgmlBasicOps.GetActiveRank() : 0;
            try
            {
                for (int stage = 0; stage < _mesh.Stages; stage++)
                for (int rank = 0; rank < _mesh.TensorParallel; rank++)
                {
                    if (switches) GgmlBasicOps.SetActiveRank(_mesh.GlobalDevice(stage, rank));
                    foreach (Tensor t in StageRankTensors(slot.K, stage, rank)) Add(t, t.Storage.ByteLength);
                    foreach (Tensor t in StageRankTensors(slot.V, stage, rank)) Add(t, t.Storage.ByteLength);
                    Prime();
                    if (slot.Recurrent == null) continue;
                    for (int l = 0; l < slot.Recurrent.Length; l++)
                        if (slot.Recurrent[l] is { } layer && StageOfLayer(l) == stage && rank < layer.Length
                            && layer[rank]?.GraphTensors is { } tensors && (overwritten || layer[rank].GraphTensorsResetToZero))
                            foreach (Tensor t in tensors) Add(t, t.ElementCount() * sizeof(float));
                    Prime();
                }
            }
            finally
            {
                if (switches) GgmlBasicOps.SetActiveRank(previous);
            }

            IEnumerable<Tensor> StageRankTensors(Tensor[][] caches, int stage, int rank)
            {
                if (caches == null) yield break;
                for (int e = 0; e < caches.Length; e++)
                    if (CacheStage(e) == stage && caches[e] is { } entry && rank < entry.Length && entry[rank] != null)
                        yield return entry[rank];
            }

            void Add(Tensor t, long size)
            {
                keys.Add(TensorComputePrimitives.GetStoragePointer(t));
                bytes.Add(size);
            }

            void Prime()
            {
                if (keys.Count > 0 && !GgmlBasicOps.ZeroDeviceCopies(
                        System.Runtime.InteropServices.CollectionsMarshal.AsSpan(keys),
                        System.Runtime.InteropServices.CollectionsMarshal.AsSpan(bytes)))
                    for (int i = 0; i < keys.Count; i++) GgmlBasicOps.ZeroDeviceCopy(keys[i], bytes[i]);
                keys.Clear();
                bytes.Clear();
            }
        }

        /// <summary>The pooled slot that best holds <paramref name="tokens"/> (the smallest that does, else the largest;
        /// <paramref name="mustFit"/>: null unless one does), emptied but its states as they were; null when the
        /// pool has none. Among the slots that hold the tokens, a request's own (<paramref name="spare"/> false) takes a slot
        /// a request ran on - its recordings are parked and adopted where they are - before a spare one; a copy that
        /// overwrites every byte (true) takes a spare first.</summary>
        private Slot TakePooledSlot(int tokens, bool mustFit, bool spare = false)
        {
            if (_slotPool == null || _slotPool.Count == 0) return null;
            int pick = 0;
            for (int i = 1; i < _slotPool.Count; i++)
            {
                Slot b = _slotPool[pick], c = _slotPool[i];
                int best = b.Capacity, cap = c.Capacity;
                bool fits = cap >= tokens, bestFits = best >= tokens;
                if (fits && bestFits && b.Spare != c.Spare)
                {
                    if (c.Spare == spare) pick = i;
                    continue;
                }
                if (fits ? !bestFits || cap < best : !bestFits && cap > best) pick = i;
            }
            Slot pooled = _slotPool[pick];
            if (mustFit && pooled.Capacity < tokens) return null;
            _slotPool.RemoveAt(pick);
            pooled.Spare = false;
            pooled.Length = 0;
            pooled.DraftStart = 0;
            pooled.DraftFollowed = -1;
            pooled.RopeDelta = 0;
            pooled.VerifyRows = 0;
            return pooled;
        }

        /// <summary>A slot's caches at <paramref name="capacity"/> cells: K and V per layer and rank, then the extra key
        /// caches (K only).</summary>
        /// <param name="zeroFrom">The rows a copy writes next (<see cref="ZeroCacheRows"/>): only the rows past them are zeroed.</param>
        private (Tensor[][] K, Tensor[][] V) AllocateCaches(int capacity, int zeroFrom = 0)
        {
            var k = new Tensor[CacheEntries][];
            var v = new Tensor[CacheEntries][];
            NoteDeviceMemoryChanged();
            for (int e = 0; e < CacheEntries; e++) AllocateCacheEntry(k, v, e, capacity, zeroFrom);
            return (k, v);
        }

        /// <summary>Cache entry <paramref name="e"/>'s K (and V, for a layer's) at <paramref name="capacity"/> cells, every
        /// rank's, into <paramref name="k"/> and <paramref name="v"/>; nothing for an entry this node does not hold.</summary>
        private void AllocateCacheEntry(Tensor[][] k, Tensor[][] v, int e, int capacity, int zeroFrom = 0)
        {
            Tensor first = NewCacheTensor(e, 0, capacity, value: false, zeroFrom);
            if (first == null) return;
            k[e] = new Tensor[_mesh.TensorParallel];
            k[e][0] = first;
            for (int r = 1; r < _mesh.TensorParallel; r++) k[e][r] = NewCacheTensor(e, r, capacity, value: false, zeroFrom);
            if (e >= TotalLayers) return;
            v[e] = new Tensor[_mesh.TensorParallel];
            for (int r = 0; r < _mesh.TensorParallel; r++) v[e][r] = NewCacheTensor(e, r, capacity, value: true, zeroFrom);
        }

        private void DisposeSlot(Slot slot)
        {
            if (slot == null) return;
            _spareStatesRefused = false;
            // A checkpoint frees its own parts only; a slot whose rows checkpoints share hands them to the longest.
            if (slot.SharesRows)
            {
                DisposeDependent(slot);
                return;
            }
            if (HasDependents(slot))
            {
                HandOverRows(slot);
                return;
            }
            _idleSpareRefused = false;
            _slotMutationDepth++;
            try
            {
                ForgetGraphsOf(slot);
                foreach (Tensor t in AllCacheTensors(slot.K)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
                foreach (Tensor t in AllCacheTensors(slot.V)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
                slot.K = slot.V = null;
                DisposeStates(slot.Recurrent);
                slot.Recurrent = null;
                slot.DraftCarry?.Dispose();
                slot.DraftCarry = null;
            }
            finally
            {
                _slotMutationDepth--;
                NoteDeviceMemoryChanged();
            }
        }

        private void RecycleSlot(Slot slot)
        {
            if (slot?.K == null) return;
            // Never pooled: a checkpoint's rows are another slot's, and a slot whose rows checkpoints share hands them over.
            if (InAnyShare(slot))
            {
                DisposeSlot(slot);
                return;
            }
            _slotPool ??= new List<Slot>(SlotPoolMax);
            if (_slotPool.Count < Math.Min(SlotPoolMax, ExecutionOptions.FromEnvironment().KvHolderPoolMax) && RoomToPool(slot))
                _slotPool.Add(slot);
            else
                DisposeSlot(slot);
        }

        /// <summary>Where slots are sized to the device's room (<see cref="SizesSlotsToRoom"/>), a finished request's slot is
        /// pooled only while the device keeps as much again free beside its headroom and a prompt's step: a pooled slot belongs to
        /// no conversation, and kept on a device near full it is the first thing the next allocation frees.</summary>
        private bool RoomToPool(Slot slot)
        {
            if (!SizesSlotsToRoom) return true;
            var (free, total) = DeviceMemory();
            return total <= 0 || free >= SlotStorageBytes(slot) + DeviceHeadroom(total) + PromptStepBytes;
        }

        /// <summary>
        /// Grow a slot's caches to hold <paramref name="required"/> tokens, keeping its rows. A request's
        /// <paramref name="reservation"/> (its prompt and whole generation) is held as asked, rounded up to
        /// <see cref="ReservationStepTokens"/>; growth past what was reserved doubles, by no more than
        /// <see cref="SlotGrowthHeadroomTokens"/>. Where slots are sized to the device's room (<see cref="SizesSlotsToRoom"/>)
        /// the slot grows to <see cref="SlotTarget"/>'s capacity for <paramref name="required"/> cells and the
        /// <paramref name="wanted"/> ones (its request's reservation), a tensor at a time (<see cref="GrowSlotRolling"/>).
        /// </summary>
        private void EnsureSlotCapacity(Slot slot, int required, bool reservation = false, int wanted = 0)
        {
            if (required > slot.Touched) slot.Touched = required;
            if (required <= slot.Capacity) return;
            if (required > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {required} exceeds configured max context {_maxContextLength}.");
            if (SizesSlotsToRoom)
            {
                Slot[] keep = RunningSlots(slot);
                int target = SlotTarget(required, Math.Max(required, wanted), keep, "a slot's growth", heldCapacity: slot.Capacity);
                if (target < required)
                {
                    // As a growth always could: the required cells where they fit beside the headroom and the narrowest prompt
                    // step (finished conversations released only for that, where releasing them reaches it), else the failure
                    // with what holds the device.
                    target = AlignedSlotCapacity(required);
                    if (!EnsureDeviceRoom(SlotTargetBytes(target, slot.Capacity), keep, RoomUse.SlotFloor, mayEvict: true,
                            "a slot's growth", CacheBlockBytes(target)))
                        throw GrowthExhausted(slot, target, keep);
                }
                GrowSlotRolling(slot, target);
                return;
            }
            int capacity;
            if (reservation)
            {
                // Four Qwen3.8-27B DFlash2 requests of 8K tokens each grew pooled 512-cell slots to 12K cells (16K before
                // the headroom bound): a gigabyte of K/V no request could use, and a 24 GB GPU then had no memory for the
                // step's state copies.
                capacity = (int)Math.Min(_maxContextLength, (required + (long)ReservationStepTokens - 1) / ReservationStepTokens * ReservationStepTokens);
            }
            else
            {
                capacity = Math.Max(slot.Capacity, 1);
                while (capacity < required)
                    capacity = Math.Min(_maxContextLength, capacity * 2);
                // Doubled, but no more than SlotGrowthHeadroomTokens past what is needed: a 16K-token clone that its request's
                // reservation grew by a few dozen tokens became 32K cells, twice the memory and the upload.
                capacity = AlignedSlotCapacity(Math.Min(capacity, (long)required + SlotGrowthHeadroomTokens));
            }
            if (TryGrowSlotOnDevice(slot, capacity)) return;
            // On a device this model manages the grown caches' room is made first, or the growth fails with what holds the
            // device: running out part way left a request failed with CUDA error 2 (issue #376).
            if (DeviceRoomManaged) EnsureRoomToGrow(slot, capacity);
            _slotMutationDepth++;
            try
            {
                SyncToHost(slot);
                (Tensor[][] k, Tensor[][] v) = AllocateCaches(capacity);
                // Only this slot's graphs bound the caches it gives up: every other request's steps and recordings stay. Clearing
                // the whole cache when a pooled 512-cell slot grew for a 4K prompt had every slot's decode trace and record again
                // (Qwen3.8 Flash Next on an A40: five graphs and two parked recordings, 2,200 allocations and a 109 ms recording).
                ForgetGraphsOfGrowing(slot);
                CopyRows(slot.K, k, slot.Length);
                CopyRows(slot.V, v, slot.Length);
                using (ReturningToDriver())
                {
                    foreach (Tensor t in AllCacheTensors(slot.K)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
                    foreach (Tensor t in AllCacheTensors(slot.V)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
                }
                Tensor[][] oldK = slot.K;
                slot.K = k;
                slot.V = v;
                slot.Capacity = capacity;
                RepointDependents(oldK, slot);   // the grown caches hold every row the checkpoints share
            }
            finally
            {
                _slotMutationDepth--;
            }
        }

        /// <summary>
        /// Grow <paramref name="slot"/>'s caches to <paramref name="capacity"/> cells one tensor at a time: each entry's new tensor
        /// (zeroed only past the slot's length), its rows copied, the old one freed and its block handed back to the driver when
        /// the next would not fit. The device holds the added cells and one tensor more at the peak, not a second slot: a
        /// 130K-token Qwen3.8-27B slot (9.3 GB) needed 9.2 GB beside it to grow by 1,024 cells, where it needs about 0.4 GB
        /// (issue #376). An allocation that fails part way fails the request with what holds the device; the slot, its caches of
        /// two capacities, is disposed, so none of it runs.
        /// </summary>
        private void GrowSlotRolling(Slot slot, int capacity)
        {
            if (capacity <= slot.Capacity) return;
            _slotMutationDepth++;
            var cuda = DeviceRoomForTests == null ? _allocator as TensorSharp.Cuda.CudaAllocator : null;
            Tensor[][] k = new Tensor[CacheEntries][], v = new Tensor[CacheEntries][];
            Tensor[][] oldK = slot.K;
            int made = 0;
            try
            {
                SyncToHost(slot);
                // Only this slot's graphs bound the caches it gives up, and they go now: a recording keeps the tensors it
                // writes alive (CudaGraphCapture's KeepAlive), so disposed later, beside the next recorded step
                // (ForgetGraphsOfGrowing), every old tensor stayed allocated through the growth, which then needed a second
                // slot's memory, not the added cells and one tensor its room was made for (Qwen3.8-27B, three 47K
                // conversations at once on an RTX PRO 4500: a growth from 48,128 to 55,296 cells ran out of memory after 29
                // of its 34 tensors, HTTP 500).
                ForgetGraphsOf(slot);
                NoteDeviceMemoryChanged();
                long driverFree = cuda == null ? long.MaxValue : cuda.GetMemoryInfo().freeBytes;
                int rows = slot.Length;
                for (int e = 0; e < CacheEntries; e++)
                {
                    if (slot.K?[e] == null) continue;
                    k[e] = new Tensor[slot.K[e].Length];
                    if (slot.V?[e] != null) v[e] = new Tensor[slot.V[e].Length];
                    for (int r = 0; r < slot.K[e].Length; r++)
                    {
                        if (slot.K[e][r] != null) k[e][r] = Replace(slot.K[e], r, e, value: false);
                        if (slot.V?[e]?[r] != null) v[e][r] = Replace(slot.V[e], r, e, value: true);
                    }
                }
                (_allocator as TensorSharp.Cuda.CudaAllocator)?.LogDeviceRoom(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"{Config.Architecture} a slot of {rows:N0} rows grown a tensor at a time from {slot.Capacity:N0} to {capacity:N0} cells"));
                slot.K = k;
                slot.V = v;
                slot.Capacity = capacity;
                RepointDependents(oldK, slot);   // the grown caches hold every row the checkpoints share
                RollingGrowths++;

                Tensor Replace(Tensor[] old, int r, int e, bool value)
                {
                    // A sliding-window layer's ring holds at most its cells on both sides (CacheCells), and one past its first
                    // lap moves whole: its cells keep their positions (CopyRows, Grow).
                    long bytes = old[r].Storage.ByteLength / Math.Max(1, old[r].Sizes[1]) * CacheCells(e, capacity);
                    int moved = (int)Math.Min(rows, old[r].Sizes[1]);
                    if (cuda != null)
                        driverFree += cuda.TrimForAllocation(bytes, cuda.AllocationBlockBytes(bytes), Math.Max(0, driverFree));
                    if (RollingGrowthFailAfterForTests is int failAfter && made >= failAfter)
                        throw new InvalidOperationException($"{Config.Architecture}: injected allocation failure during a slot's growth (test)");
                    Tensor grown = NewCacheTensor(e, r, capacity, value, zeroFrom: moved);
                    made++;
                    if (cuda != null) driverFree -= cuda.AllocationBlockBytes(bytes);
                    if (moved > 0)
                    {
                        using Tensor src = old[r].Narrow(1, 0, moved);
                        using Tensor dst = grown.Narrow(1, 0, moved);
                        Ops.Copy(dst, src);
                    }
                    InvalidateTensorDeviceCache(old[r]);
                    using (ReturningToDriver()) old[r].Dispose();   // no slot of the old capacity is coming
                    old[r] = null;
                    return grown;
                }
            }
            catch (Exception ex)
            {
                // Caches of two capacities never run: everything the slot held goes, and the request fails.
                foreach (Tensor t in AllCacheTensors(k).Concat(AllCacheTensors(v)).Concat(AllCacheTensors(slot.K)).Concat(AllCacheTensors(slot.V)).ToList())
                {
                    InvalidateTensorDeviceCache(t);
                    t.Dispose();
                }
                slot.K = slot.V = null;
                DisposeStates(slot.Recurrent);
                slot.Recurrent = null;
                slot.DraftCarry?.Dispose();
                slot.DraftCarry = null;
                InvalidateDependents(oldK, Runtime.Scheduling.PrefixCache.InvalidationReason.NativeSlotReclaimed);   // their rows went too
                throw new InvalidOperationException(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"{Config.Architecture}: growing a slot from {slot.Capacity:N0} to {capacity:N0} cells failed after {made} of its " +
                    $"cache tensors ({ex.Message}); the request's slot was released."), ex);
            }
            finally
            {
                _slotMutationDepth--;
                NoteDeviceMemoryChanged();
            }
        }

        /// <summary>Slots grown a tensor at a time (<see cref="GrowSlotRolling"/>; tests check the route).</summary>
        internal int RollingGrowths { get; private set; }

        private static void CopyRows(Tensor[][] from, Tensor[][] to, int rows)
        {
            if (rows <= 0 || from == null) return;
            for (int l = 0; l < from.Length; l++)
            {
                if (from[l] == null) continue;
                for (int r = 0; r < from[l].Length; r++)
                {
                    if (from[l][r] == null) continue;   // a rank that does not run this entry (a draft layer off rank 0)
                    // A ring past its first lap moves whole (same cells on both sides).
                    int n = (int)Math.Min(rows, from[l][r].Sizes[1]);
                    using Tensor src = from[l][r].Narrow(1, 0, n);
                    using Tensor dst = to[l][r].Narrow(1, 0, n);
                    Ops.Copy(dst, src);
                }
            }
        }

        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal))
                return false;
            if (_activeKey == null)
                _primary = SnapshotActive();
            else
                _slots[_activeKey] = SnapshotActive();
            bool fresh = !_slots.TryGetValue(requestId, out Slot slot);
            if (fresh)
            {
                slot = NewSlot();
                _slots[requestId] = slot;
            }
            LoadActive(slot);
            _activeKey = requestId;
            return fresh;
        }

        /// <summary>Hand the primary cache to <paramref name="requestId"/> (zero copy) and give the
        /// primary a fresh allocation, when the request that filled it now needs its own slot.</summary>
        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId) || !SupportsPerSequenceFusedForward) return;
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_activeKey != null || _slots.ContainsKey(requestId)) return;
            _slots[requestId] = SnapshotActive();
            _activeKey = requestId;
            _primary = NewSlot();
        }

        public void OnSequenceReleased(string requestId)
        {
            if (_slots == null || string.IsNullOrEmpty(requestId)) return;
            if (!_slots.TryGetValue(requestId, out Slot slot)) return;
            DropReplaysOf(slot.K, slot.Recurrent);
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) DropReplaysOf(_cacheK, _recurrentPrimary);
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal))
            {
                slot = SnapshotActive();
                _activeKey = null;
                if (_primary != null)
                {
                    LoadActive(_primary);
                    _primary = null;
                }
            }
            _slots.Remove(requestId);
            _fittedReservations?.Remove(requestId);
            RecycleSlot(slot);
            LogDeviceHoldings($"{requestId} released");
        }

        public bool RetainSequenceCache(string requestId) => RetainSequenceCacheAs(requestId, requestId);

        /// <summary>Move <paramref name="requestId"/>'s slot into the retained set under
        /// <paramref name="key"/> (zero copy).</summary>
        public bool RetainSequenceCacheAs(string requestId, string key)
        {
            if (_slots == null || string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(key)) return false;
            if (!_slots.TryGetValue(requestId, out Slot slot)) return false;
            DropReplaysOf(slot.K, slot.Recurrent);
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) DropReplaysOf(_cacheK, _recurrentPrimary);
            _retainedSlots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(key)) return false;
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal))
            {
                slot = SnapshotActive();
                _activeKey = null;
                if (_primary != null)
                {
                    LoadActive(_primary);
                    _primary = null;
                }
            }
            _retainedSlots.Add(key, Retained(slot));
            _slots.Remove(requestId);
            _fittedReservations?.Remove(requestId);
            return true;
        }

        public bool TryRebindRetainedCache(string retainedKey, string newRequestId)
        {
            if (_retainedSlots == null || string.IsNullOrEmpty(retainedKey) || string.IsNullOrEmpty(newRequestId)) return false;
            if (!_retainedSlots.TryGetValue(retainedKey, out Slot slot) || slot.K == null) return false;
            // A checkpoint over another slot's rows is never rebound: it is copied (nothing here may use up its rows' holder).
            if (slot.SharesRows) return MaterializeShared(retainedKey, slot, newRequestId, donate: true, consumable: null, 0);
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_slots.ContainsKey(newRequestId) || string.Equals(_activeKey, newRequestId, StringComparison.Ordinal)) return false;
            _slots.Add(newRequestId, slot);
            _retainedSlots.Remove(retainedKey);
            return true;
        }

        public void DiscardRetainedCache(string requestId)
            => DiscardRetainedCaches(new[] { requestId }, Runtime.Scheduling.PrefixCache.ReleaseReason.Evicted);

        public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, Runtime.Scheduling.PrefixCache.ReleaseReason reason)
        {
            if (_retainedSlots == null) return;
            bool dispose = reason is Runtime.Scheduling.PrefixCache.ReleaseReason.Invalidated
                or Runtime.Scheduling.PrefixCache.ReleaseReason.Pressure
                or Runtime.Scheduling.PrefixCache.ReleaseReason.Reset;
            foreach (string key in payloadKeys)
            {
                if (key == null || !_retainedSlots.Remove(key, out Slot slot)) continue;
                if (dispose) DisposeSlot(slot);
                else RecycleSlot(slot);
            }
        }

        private void DisposeSlots()
        {
            DisposeSpareCheckpointStates();
            // The checkpoints' own parts first, so every slot then frees its rows once.
            if (_rowShares != null)
                foreach (List<Slot> dependents in _rowShares.Values.ToList())
                    foreach (Slot d in dependents.ToList()) DisposeDependent(d);
            _rowShares = null;
            if (_activeKey != null)
            {
                // The active fields are the bound slot: the primary is parked.
                DisposeSlot(_primary);
                _primary = null;
                _slots?.Remove(_activeKey);
                _activeKey = null;
            }
            if (_slots != null)
                foreach (Slot slot in _slots.Values) DisposeSlot(slot);
            if (_retainedSlots != null)
                foreach (Slot slot in _retainedSlots.Values) DisposeSlot(slot);
            if (_slotPool != null)
                foreach (Slot slot in _slotPool) DisposeSlot(slot);
            DisposeSlot(_spareBuild);
            _spareBuild = null;
            _slots = _retainedSlots = null;
            _fittedReservations = null;
            ClearGraphCache();
            _slotPool = null;
        }

        // ---------------------------------------------------------------- batched steps

        public bool CanBatchDecode(string requestId, int position)
            => TryGetSlot(requestId, out Slot slot) && slot.Length == position;

        private bool TryGetSlot(string requestId, out Slot slot)
        {
            slot = null;
            if (requestId == null || _slots == null) return false;
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal))
            {
                // The active fields are newer than the parked object.
                slot = SnapshotActive();
                return slot.K != null;
            }
            return _slots.TryGetValue(requestId, out slot) && slot.K != null;
        }

        public bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
        {
            int n = requestIds.Count;
            if (n == 0 || tokens.Length != n || positions.Length != n) return false;
            var chunks = new int[n][];
            for (int s = 0; s < n; s++) chunks[s] = new[] { tokens[s] };
            return TryForwardSlots(requestIds, chunks, positions, outLogits);
        }

        public bool TryForwardBatchedFusedPrefill(
            IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] positions, float[][] outLogits)
            => TryForwardSlots(requestIds, tokens, positions, outLogits);

        /// <summary>One forward over several requests' slots: every row through one activation,
        /// attention per sequence on its own slot. Declines (nothing written) when a request has no
        /// slot or does not continue its slot at the given position.</summary>
        /// <param name="features">Per request, a buffer for its rows' drafter features, or null (see
        /// <see cref="IUnifiedBatchModel.TryForwardUnified"/>); entries are set to null when no graph captured them.</param>
        /// <remarks>
        /// A step whose graph finds no device memory for its compute buffer (nothing ran) is retried once after the cached
        /// graphs and kept arenas are freed, then run as two halves of its requests, each the same way. Failed, it failed
        /// every request it carried: four Qwen3.8-27B DFlash2 prompts on a 24 GB GPU, where Qwen35Model serves them.
        /// </remarks>
        private bool TryForwardSlots(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> chunks, int[] starts, float[][] outLogits,
            float[][] features = null, SlotRowIO rowIO = null)
        {
            NoteEngineThread();
            ForgetDeferredGraphs();
            if (SlotInvariantChecksForTests) CheckSlotShareInvariants();
            if (TryForwardSlotPromptPipelined(requestIds, chunks, starts, outLogits, features, rowIO)) return true;
            if (_maxSequenceStepRows != int.MaxValue && chunks.Any(c => c != null && c.Length > _maxSequenceStepRows))
                return TryForwardSlotsInRingSteps(requestIds, chunks, starts, outLogits, features, rowIO);
            try { return TryForwardSlotsOnce(requestIds, chunks, starts, outLogits, features, rowIO); }
            catch (InvalidOperationException ex) when (IsGraphAllocationFailure(ex))
            {
                FreeGraphMemory();
                try { return TryForwardSlotsOnce(requestIds, chunks, starts, outLogits, features, rowIO); }
                catch (InvalidOperationException again) when (IsGraphAllocationFailure(again) && requestIds.Count > 1)
                {
                    FreeGraphMemory();
                    int half = requestIds.Count / 2;
                    // The halves pass the same checks the whole did, so each runs or throws: none declines.
                    return ForwardPart(0, half) && ForwardPart(half, requestIds.Count - half);
                }
            }

            bool ForwardPart(int from, int count)
            {
                var ids = new string[count];
                var parts = new int[count][];
                var partStarts = new int[count];
                var logits = new float[count][];
                float[][] partFeatures = features == null ? null : new float[count][];
                for (int i = 0; i < count; i++)
                {
                    ids[i] = requestIds[from + i];
                    parts[i] = chunks[from + i];
                    partStarts[i] = starts[from + i];
                    logits[i] = outLogits[from + i];
                    if (partFeatures != null) partFeatures[i] = features[from + i];
                }
                bool ran = TryForwardSlots(ids, parts, partStarts, logits, partFeatures, rowIO?.Slice(from, count));
                Array.Copy(logits, 0, outLogits, from, count);
                if (partFeatures != null) Array.Copy(partFeatures, 0, features, from, count);
                return ran;
            }
        }

        /// <summary>
        /// A batched step whose chunks hold more rows of a sequence than its sliding-window rings hold beside their window, as
        /// rounds of at most <see cref="MaxSequenceStepRows"/> rows a sequence (StandardDecoderModel.SlidingRings.cs): round k
        /// carries piece k of every chunk that has one, and each sequence's logits come from the round of its last piece.
        /// Declines (nothing run) where the first round does.
        /// </summary>
        private bool TryForwardSlotsInRingSteps(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> chunks, int[] starts, float[][] outLogits,
            float[][] features, SlotRowIO rowIO = null)
        {
            // Drafter features come back round by round, into each sequence's buffer at its rows.
            bool wantsFeatures = features != null && Array.Exists(features, f => f != null);
            int featureWidth = wantsFeatures ? SpecFeatureSize : 0;
            int step = _maxSequenceStepRows, n = requestIds.Count, rounds = 0;
            for (int s = 0; s < n; s++) rounds = Math.Max(rounds, ((chunks[s]?.Length ?? 0) + step - 1) / step);
            // A chunk whose rows its request's drafter follows (FollowPlainPrompt, FollowUnspeculatedPrompt) is followed in every
            // round: registered once for the chunk, the registration the first round took left the later rounds' rows out of
            // the drafter, so a Muse-Glimmer prompt past 512 rows reached its DFlash ring as its first 512 (the prompt's tail
            // missing, no carry held): 302 drafts kept after a 520-token prompt where legacy, from the same text, kept 409.
            int[] followedEnds = TakeFollowedPrompts(requestIds, out bool[] followedAside);
            void FollowRound(List<int> owners)
            {
                if (followedEnds == null) return;
                foreach (int s in owners)
                {
                    if (followedEnds[s] == DFlashNoFollow) continue;
                    if (followedAside != null && followedAside[s]) FollowUnspeculatedPrompt(requestIds[s], followedEnds[s]);
                    else FollowPlainPrompt(requestIds[s], followedEnds[s]);
                }
            }
            // Plain prompts' rounds overlap on the device (IGraphExecutor.OverlapSteps): a round that ends no request's chunk runs
            // no head and hands the host nothing, so its run returns with its work queued while the next round is embedded,
            // traced and built. The slots hold their whole chunks first, so no round grows them.
            IDisposable overlap = null;
            if (OverlapPromptSteps && rounds > 1 && GraphExecutor != null && _mesh.Stages == 1 && !_rankGraphs
                && followedEnds == null && !wantsFeatures && _slots != null
                && requestIds.All(id => _slots.ContainsKey(id)))
            {
                for (int s = 0; s < n; s++)
                    if (chunks[s] != null) EnsureSlotCapacity(_slots[requestIds[s]], checked(starts[s] + chunks[s].Length));
                overlap = GraphExecutor.OverlapSteps();
            }
            try
            {
                for (int round = 0; round < rounds; round++)
                {
                    var ids = new List<string>();
                    var parts = new List<int[]>();
                    var partStarts = new List<int>();
                    var owners = new List<int>();
                    // A request's row input and hidden output belong to its last row, so they go with its last piece.
                    var inputRows = new List<float[]>();
                    var hidden = new List<float[]>();
                    for (int s = 0; s < n; s++)
                    {
                        int at = round * step, length = chunks[s]?.Length ?? 0;
                        if (at >= length) continue;
                        bool last = at + step >= length;
                        ids.Add(requestIds[s]);
                        parts.Add(chunks[s].AsSpan(at, Math.Min(step, length - at)).ToArray());
                        partStarts.Add(starts[s] + at);
                        owners.Add(s);
                        inputRows.Add(last ? rowIO?.InputRows?[s] : null);
                        hidden.Add(last ? rowIO?.Hidden?[s] : null);
                    }
                    bool anyLast = false;
                    foreach (int o in owners) anyLast |= (round + 1) * step >= chunks[o].Length;
                    var roundRowIO = rowIO == null
                        ? overlap != null && !anyLast ? new SlotRowIO { Logits = false } : null
                        : new SlotRowIO
                        {
                            InputRows = rowIO.InputRows == null ? null : inputRows.ToArray(),
                            Logits = rowIO.Logits && (overlap == null || anyLast),
                            Hidden = rowIO.Hidden == null ? null : hidden.ToArray(),
                        };
                    var logits = new float[ids.Count][];
                    // Each piece's rows the sequence's buffer holds (its chunk's last rows), captured by this round.
                    float[][] roundFeatures = null;
                    if (wantsFeatures)
                    {
                        roundFeatures = new float[ids.Count][];
                        for (int i = 0; i < owners.Count; i++)
                            if (features[owners[i]] != null) roundFeatures[i] = new float[(long)parts[i].Length * featureWidth];
                    }
                    FollowRound(owners);
                    if (!TryForwardSlots(ids, parts, partStarts.ToArray(), logits, roundFeatures, roundRowIO))
                    {
                        if (round == 0) return false;
                        throw new InvalidOperationException($"{Config.Architecture}: round {round} of a batched step in ring steps declined after the first ran.");
                    }
                    if (roundFeatures != null)
                        for (int i = 0; i < owners.Count; i++)
                        {
                            float[] all = features[owners[i]], part = roundFeatures[i];
                            if (all == null || part == null) continue;
                            int length = chunks[owners[i]].Length, held = (int)Math.Min(length, all.LongLength / featureWidth);
                            int at = round * step, firstHeld = length - held;
                            for (int r = 0; r < parts[i].Length; r++)
                            {
                                int row = at + r - firstHeld;
                                if (row >= 0) Array.Copy(part, (long)r * featureWidth, all, (long)row * featureWidth, featureWidth);
                            }
                        }
                    for (int i = 0; i < owners.Count; i++)
                        if ((round + 1) * step >= chunks[owners[i]].Length) outLogits[owners[i]] = logits[i];
                }
            }
            finally
            {
                overlap?.Dispose();
                // A request with no rows in this step keeps its registration for its next chunk.
                var idle = new List<int>();
                for (int s = 0; s < n; s++)
                    if ((chunks[s]?.Length ?? 0) == 0) idle.Add(s);
                FollowRound(idle);
            }
            return true;
        }

        /// <summary>Frees the device memory a step can do without: the cached graphs, the kept arenas and the idle slots'
        /// device copies.</summary>
        private void FreeGraphMemory()
        {
            ClearGraphCache();
            if (IsGgmlBackend) GgmlBasicOps.ReleaseKeptArenas();
            DropIdleSlotDeviceCopies();
        }

        /// <summary>A graph's build found no device memory for its compute buffer (ggml_ops_graph_builder.cpp).</summary>
        private static bool IsGraphAllocationFailure(Exception ex)
            => ex.Message.Contains("failed to allocate the graph's compute memory", StringComparison.Ordinal);

        /// <summary>Forwards over several requests that fail as a graph's compute allocation would, before running (tests).</summary>
        internal int InjectedForwardAllocationFailures { get; set; }

        /// <summary>
        /// A step's per-request row input and output (<see cref="IUnifiedBatchModel.TryForwardUnifiedRows"/>): the rows that
        /// replace some requests' last embedded rows, whether the step runs the LM head, and the buffers of the requests' last
        /// final-normed rows.
        /// </summary>
        private sealed class SlotRowIO
        {
            /// <summary>Per request, the embedding row of its last row, or null (its token's).</summary>
            public float[][] InputRows;
            /// <summary>The step runs the head (some request wants logits); without it, every request's logits stay null.</summary>
            public bool Logits = true;
            /// <summary>Per request, a buffer for its last final-normed row, or null.</summary>
            public float[][] Hidden;

            public SlotRowIO Slice(int from, int count) => new()
            {
                InputRows = InputRows?[from..(from + count)],
                Logits = Logits,
                Hidden = Hidden?[from..(from + count)],
            };
        }

        /// <summary>Write the requests' input rows over their last embedded rows (<paramref name="lastRows"/>).</summary>
        private unsafe void InjectRows(Tensor embedded, SlotRowIO rowIO, int[] lastRows)
        {
            if (rowIO?.InputRows == null) return;
            int dim = Config.HiddenSize;
            bool wrote = false;
            float* destination = GetFloatPtr(embedded);
            for (int s = 0; s < lastRows.Length; s++)
            {
                float[] row = rowIO.InputRows[s];
                if (row == null) continue;
                if (row.Length != dim)
                    throw new ArgumentException($"{Config.Architecture}: an input row has {row.Length} values, the embedding {dim}.");
                fixed (float* source = row)
                    Buffer.MemoryCopy(source, destination + (long)lastRows[s] * dim, (long)dim * sizeof(float), (long)dim * sizeof(float));
                wrote = true;
            }
            if (wrote) HostWritten(embedded);
        }

        private bool TryForwardSlotsOnce(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> chunks, int[] starts, float[][] outLogits,
            float[][] features, SlotRowIO rowIO = null)
        {
            if (!SupportsPerSequenceFusedForward || _slots == null || HasPendingMedia || _pendingPositionTable != null) return false;
            int n = requestIds.Count;
            if (n == 0 || chunks.Count != n || starts.Length != n || outLogits.Length != n) return false;

            RestorePrimaryCache();
            var slots = new Slot[n];
            var queryStart = new int[n];
            var counts = new int[n];
            int rows = 0;
            for (int s = 0; s < n; s++)
            {
                if (!_slots.TryGetValue(requestIds[s], out slots[s]) || slots[s].K == null) return false;
                if (slots[s].Length != starts[s] || chunks[s] == null || chunks[s].Length == 0) return false;
                for (int o = 0; o < s; o++)
                    if (ReferenceEquals(slots[o], slots[s])) return false;
                queryStart[s] = rows;
                counts[s] = chunks[s].Length;
                rows += counts[s];
            }
            for (int s = 0; s < n; s++)
            {
                int required = checked(starts[s] + counts[s]);
                EnsureSlotCapacity(slots[s], required, wanted: WantedCells(requestIds[s], required));
            }
            PrepareWrites(slots, starts);   // the graph step's RunTraced does it too; the per-op layers below need it here
            if (n > 1 && InjectedForwardAllocationFailures > 0)
            {
                InjectedForwardAllocationFailures--;
                throw new InvalidOperationException($"{Config.Architecture} graph step failed: graph builder: failed to allocate the graph's compute memory. (injected)");
            }

            var flatTokens = new int[rows];
            var positions = new int[rows];
            var lastRows = new int[n];
            for (int s = 0; s < n; s++)
            {
                Array.Copy(chunks[s], 0, flatTokens, queryStart[s], counts[s]);
                for (int i = 0; i < counts[s]; i++) positions[queryStart[s] + i] = starts[s] + i + slots[s].RopeDelta;
                lastRows[s] = queryStart[s] + counts[s] - 1;
            }
            var batch = new SlotBatch
            {
                NumSequences = n,
                QueryStart = queryStart,
                Counts = counts,
                Starts = (int[])starts.Clone(),
                Slots = slots,
            };

            Tensor embedded = null;
            int[] promptEnds = null;
            bool[] aside = null;
            try
            {
                embedded = EmbedStep(flatTokens);
                InjectRows(embedded, rowIO, lastRows);
                bool capture = rowIO == null && features != null && Array.Exists(features, f => f != null);
                // A lone speculating request's prompt chunk hands the draft layer its rows, and every speculating request's
                // chunk its DFlash ring's (FollowPlainPrompt); a request that does not speculate, its ring's where one is free
                // (FollowUnspeculatedPrompt).
                promptEnds = capture ? null : TakeFollowedPrompts(requestIds, out aside);
                DraftFollow follow = n == 1 && promptEnds != null && promptEnds[0] != DFlashNoFollow && aside?[0] != true
                    ? PlanDraftFollow(slots[0], starts[0], null) : DraftFollow.None;
                float[] draftRows = follow != DraftFollow.None ? PrepareDraftFollow(slots[0], chunks[0], media: false) : null;
                DFlashFollowPlan dflash = promptEnds == null ? null : PlanDFlashFollow(requestIds.ToArray(), starts, counts, promptEnds, aside);
                if (capture ? TryRunCapturingGraph(embedded, slots, queryStart, counts, starts, outLogits, features)
                    : RunFollowingDFlash(ref dflash, plan => TryRunGraph(embedded, slots, queryStart, counts, (int[])starts.Clone(),
                          wantLogits: rowIO?.Logits ?? true, out float[][] graphLogits, rowIO?.Hidden, follow: follow, draftRows: draftRows, dflash: plan, prompt: true)
                          && CopyLogits(graphLogits, outLogits)))
                {
                    BatchedForwardCount++;
                    for (int s = 0; s < n; s++)
                        slots[s].Length = starts[s] + counts[s];
                    if (follow != DraftFollow.None) NoteDraftFollowed(slots[0], starts[0], counts[0]);
                    NoteDFlashFollowed(dflash, starts, counts);
                    return true;
                }
                static bool CopyLogits(float[][] from, float[][] to)
                {
                    if (from != null) Array.Copy(from, to, to.Length);
                    else Array.Clear(to);   // a step no request wanted logits from ran no head
                    return true;
                }
            }
            catch
            {
                // A step that found no device memory is retried (TryForwardSlots): the chunks still follow there.
                RestoreFollowedPrompts(requestIds, promptEnds, aside);
                throw;
            }
            finally
            {
                embedded?.Dispose();
            }
            foreach (Slot slot in slots) SyncToHost(slot);
            ClearGraphCache();
            if (features != null) Array.Clear(features);   // the per-op layers capture no drafter features

            float[] all;
            // A step with a text-LoRA adapter selected must apply it to every planned weight.
            bool loraAudit = OpenLoraForwardAudit();
            _forwardSw.Start();
            try
            {
                BatchedForwardCount++;
                EnterStage(0);
                Tensor hidden = EmbedStep(flatTokens);
                InjectRows(hidden, rowIO, lastRows);
                hidden = RunLayers(hidden, TokenPositions.Explicit(positions), batch);
                all = ProjectRows(hidden, lastRows, rowIO?.Logits ?? true, rowIO?.Hidden);
                hidden.Dispose();
                if (loraAudit) VerifyLoraForwardAudit(producedLogits: true);
            }
            finally
            {
                EnterStage(0);
                _forwardSw.Stop();
                if (loraAudit) CloseLoraForwardAudit();
            }

            int vocab = Config.VocabSize;
            for (int s = 0; s < n; s++)
            {
                slots[s].Length = starts[s] + counts[s];
                if (all == null)
                {
                    outLogits[s] = null;   // no request wanted logits: no head ran
                    continue;
                }
                slots[s].Logits ??= new float[vocab];
                Array.Copy(all, (long)s * vocab, slots[s].Logits, 0, vocab);
                outLogits[s] = slots[s].Logits;
            }
            return true;
        }

        /// <summary>Attention for a batched step: the rows of each sequence against its own slot,
        /// with the kernels a single request uses (decode for one row, prefill for a chunk).</summary>
        private Tensor SlotAttention(RankScope scope, int layer, Tensor q, Tensor k, Tensor v, SlotBatch batch)
        {
            int heads = scope.LocalHeads;
            int rows = (int)q.Sizes[0];
            if (batch.NumSequences == 1)
                return SequenceAttention(scope, layer, batch.Slots[0], q, k, v, batch.Starts[0], rows);

            var output = new Tensor(scope.Allocator, DType.Float32, rows, heads * _valueDim);
            for (int s = 0; s < batch.NumSequences; s++)
            {
                int first = batch.QueryStart[s], count = batch.Counts[s];
                Tensor qs = Rows(q, first, count), ks = Rows(k, first, count), vs = Rows(v, first, count);
                using Tensor part = SequenceAttention(scope, layer, batch.Slots[s], qs, ks, vs, batch.Starts[s], count);
                using Tensor dst = output.Narrow(0, first, count);
                Ops.Copy(dst, part);
            }
            q.Dispose();
            k.Dispose();
            v.Dispose();
            return output;
        }

        private static Tensor Rows(Tensor t, int first, int count)
        {
            using Tensor view = t.Narrow(0, first, count);
            return Ops.NewContiguous(view);
        }

        private Tensor SequenceAttention(RankScope scope, int layer, Slot slot, Tensor q, Tensor k, Tensor v, int start, int count)
        {
            Tensor cacheK = slot.K[layer][scope.Rank], cacheV = slot.V[layer][scope.Rank];
            int band = SlidingWindow(layer);
            if (RingWrapped(layer, start, count))
                return RingAttention(scope, cacheK, cacheV, RingOf(layer), q, k, v, start, count, band);
            return count == 1
                ? DecodeAttention(scope, cacheK, cacheV, (int)cacheK.Sizes[1], q, k, v, start, band)
                : PrefillAttention(scope, cacheK, cacheV, q, k, v, start, count, band);
        }
    }
}
