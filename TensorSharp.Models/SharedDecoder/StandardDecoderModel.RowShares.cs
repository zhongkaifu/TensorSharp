// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    /// <summary>
    /// Shared-row checkpoints, where recorded steps bind a slot's buffers (<see cref="SlotBoundPolicies"/>, the CUDA executor):
    /// a request's automatic prompt-end checkpoint keeps only its recurrent states and shares the request's cache rows below
    /// its length (<see cref="Slot.SharesRows"/>), and a conversation's next turn that adopts it takes over the conversation's
    /// end state's slot instead of copying it (a steal: the slot's length set back, the checkpoint's states written into the
    /// slot's own, so its recordings still bind). Copied whole, Qwen3.8-27B's 47K-token checkpoint was a 3.3 GB copy in its
    /// request's tail and its adopter's clone another (issue #376); shared, each is 166 MB of states.
    /// <para>The rules, keyed by a slot's K array (slot objects are re-minted, <see cref="SnapshotActiveFields"/>):</para>
    /// <list type="bullet">
    /// <item>The slot that holds the rows (the owner: running, retained or pooled never) writes rows only at or past every
    /// checkpoint's length (<see cref="PrepareWrites"/>, at every write entry: graph steps, the per-op forwards, truncation);
    /// a write below one invalidates that checkpoint first, reported to the prefix cache, never changes its rows.</item>
    /// <item>A checkpoint is never pooled, rebound to a request, reclaimed for a new one, moved or reset; its owner is pooled
    /// or disposed only after its rows go to the longest checkpoint (<see cref="HandOverRows"/>), and a copy or growth of the
    /// owner takes its checkpoints along (<see cref="RepointDependents"/>).</item>
    /// <item>Only a same-conversation adopter steals, and only payloads the prefix cache lets it consume
    /// (<see cref="MaterializeRequest.Consumable"/>); otherwise the checkpoint is copied as before.</item>
    /// </list>
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Checkpoints sharing a slot's rows, by that slot's K array (reference identity).</summary>
        private Dictionary<Tensor[][], List<Slot>> _rowShares;

        /// <summary>Checkpoints over one slot's rows at most: the newest serve the conversation's next turns, older ones only
        /// retries of older turns.</summary>
        private const int MaxCheckpointsPerShare = 4;

        /// <summary>Checkpoints that shared their request's rows (tests check the route engaged).</summary>
        internal int SharedCheckpoints { get; private set; }

        /// <summary>Adoptions of a shared checkpoint that took over its rows' slot (no copy).</summary>
        internal int RowSteals { get; private set; }

        /// <summary>Slots whose rows went to their longest checkpoint as they were freed.</summary>
        internal int RowHandOvers { get; private set; }

        /// <summary>Copies or growths of a slot that took its checkpoints along.</summary>
        internal int RowRepoints { get; private set; }

        /// <summary>Shared checkpoints invalidated because their rows' slot wrote below them (tests assert none).</summary>
        internal int SharedCheckpointsInvalidatedByWrite { get; private set; }

        /// <summary>Tests: check the share invariants before every unified step (<see cref="CheckSlotShareInvariants"/>).</summary>
        internal bool SlotInvariantChecksForTests { get; set; }

        /// <summary>Tests: checkpoints copy as before (the reference a shared-row run is compared with, on the CUDA executor).</summary>
        internal bool SharedRowsDisabledForTests { get; set; }

        /// <summary>A family whose steps write a slot's rows below its length with other bytes opts out: its checkpoints copy.
        /// (Qwen3.8's QSA rewrites the last whole block's pooled key below the length with the same bytes, a function of rows
        /// that do not change.)</summary>
        protected virtual bool CheckpointRowsAppendOnly => true;

        private bool _lastMaterializeWantedRoom;

        /// <summary>TS_PREFIX_RELEASE_DEBUG=1 (the prefix cache's decision log): one line a shared checkpoint, steal, hand-over
        /// and copy of a shared checkpoint.</summary>
        private static readonly bool s_shareDebug = Environment.GetEnvironmentVariable("TS_PREFIX_RELEASE_DEBUG") == "1";

        private bool _writeBelowShareWarned;

        private static double Ms(long ticks) => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

        private void ShareDebug(string line)
        {
            if (s_shareDebug) Console.WriteLine($"[prefix-share] {Config.Architecture} {line}");
        }

        bool IPrefixCacheModel.LastMaterializeWantedRoom => _lastMaterializeWantedRoom;

        /// <summary>Whether <paramref name="k"/> holds rows checkpoints share.</summary>
        private bool InShare(Tensor[][] k) => k != null && _rowShares != null && _rowShares.ContainsKey(k);

        /// <summary>A slot that holds rows checkpoints share (not a checkpoint itself).</summary>
        private bool HasDependents(Slot slot) => slot?.K != null && !slot.SharesRows && InShare(slot.K);

        /// <summary>Whether <paramref name="slot"/> is part of a share: a checkpoint, or the slot holding its rows.</summary>
        private bool InAnyShare(Slot slot) => slot != null && (slot.SharesRows || HasDependents(slot));

        // ---------------------------------------------------------------- capture

        bool IPrefixCacheModel.TryCaptureCheckpoint(string requestId, string payloadKey, bool mayShareRows, out PayloadFootprint footprint)
        {
            if (!mayShareRows || !SlotBoundPolicies || !CheckpointRowsAppendOnly || SharedRowsDisabledForTests || !SupportsPrefixCheckpoints
                || string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey)
                || !TryGetSlot(requestId, out Slot slot) || slot.Length <= 0)
                return ((IPrefixCacheModel)this).TryCaptureCopy(requestId, payloadKey, out footprint);
            footprint = default;
            _retainedSlots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(payloadKey)) return false;
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            // The states' room, unless a spare set holds them (a new checkpoint is worth more than the oldest finished
            // conversation, as a copy's was).
            if (_spareCheckpointStates == null && _slotStateBytes > 0 && !EnsureDeviceRoom(_slotStateBytes, RunningSlots(), RoomUse.States, mayEvict: true,
                    "a prompt-end checkpoint's states"))
            {
                CheckpointsDeclinedForRoom++;
                return false;
            }
            _slotMutationDepth++;
            try
            {
                long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
                var checkpoint = new Slot
                {
                    K = slot.K, V = slot.V, Capacity = slot.Capacity, Length = slot.Length, DraftStart = slot.DraftStart,
                    DraftFollowed = -1, RopeDelta = slot.RopeDelta, SharesRows = true,
                    Recurrent = _anyRecurrent ? TakeCheckpointStates() : null,
                };
                NoteDeviceMemoryChanged();
                // The states as the request's last step left them (an executor that keeps device copies of host tensors
                // brings them back first; the rows stay where they are).
                if (_anyRecurrent && slot.DeviceDirty)
                    foreach (RecurrentState[] layer in slot.Recurrent)
                        if (layer != null)
                            foreach (RecurrentState state in layer)
                                if (state != null)
                                    foreach (Tensor t in state.HostTensors) SyncStateTensor(t);
                long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
                if (_anyRecurrent)
                    for (int l = 0; l < slot.Recurrent.Length; l++)
                        if (slot.Recurrent[l] != null)
                            for (int r = 0; r < slot.Recurrent[l].Length; r++)
                                checkpoint.Recurrent[l][r].CopyFrom(slot.Recurrent[l][r]);
                long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
                _retainedSlots.Add(payloadKey, Retained(checkpoint));
                _rowShares ??= new Dictionary<Tensor[][], List<Slot>>(ReferenceEqualityComparer.Instance);
                if (!_rowShares.TryGetValue(slot.K, out List<Slot> dependents))
                    _rowShares.Add(slot.K, dependents = new List<Slot>());
                dependents.Add(checkpoint);
                SharedCheckpoints++;
                footprint = MeasureEndState(payloadKey);
                ShareDebug($"{payloadKey}: a checkpoint of {slot.Length:N0} rows shares {requestId}'s slot of {slot.Capacity:N0} cells " +
                    $"(its states only; {dependents.Count} over the slot; host ms: room {Ms(t1 - t0):F2}, states {Ms(t2 - t1):F2}, copy {Ms(t3 - t2):F2})");
                // Past the cap the oldest goes (it serves only a retry of an older turn).
                while (dependents.Count > MaxCheckpointsPerShare)
                    InvalidateDependent(dependents.OrderBy(d => d.RetainedAt).First(), InvalidationReason.ModelBudgetEviction);
                return true;
            }
            finally
            {
                _slotMutationDepth--;
            }
        }

        // ---------------------------------------------------------------- spare states

        /// <summary>A checkpoint's states made ahead while the engine is idle (<see cref="RunIdleSlice"/>), which the next
        /// shared checkpoint takes instead of allocating them in its request's tail: Qwen3.8-27B's 48 GDN states took 6.4-7.9 ms
        /// of host time there (96 tensors allocated and cleared) of a 47K prompt's 10 ms between its first token and its next
        /// step, on an RTX PRO 4500. One set at most, made only with room beside the headroom and a prompt step, and the first
        /// thing making room gives back.</summary>
        private RecurrentState[][] _spareCheckpointStates;

        // The spare being made, a layer at a time, and the next layer.
        private RecurrentState[][] _spareStatesBuild;
        private int _spareStatesNext;

        // The device had no room for the spare: not asked again until something is freed or a spare is taken.
        private bool _spareStatesRefused;

        /// <summary>Spare checkpoint states made (tests).</summary>
        internal int SpareCheckpointStatesMade { get; private set; }

        /// <summary>Whether a checkpoint's states should be made ahead: shared checkpoints are taken here, and none is spare.</summary>
        private bool WantsSpareCheckpointStates => SlotBoundPolicies && _anyRecurrent && _spareCheckpointStates == null
            && _prefixCacheSink != null && _slotStateBytes > 0 && CheckpointRowsAppendOnly && !SharedRowsDisabledForTests
            && SupportsPrefixCheckpoints;

        /// <summary>
        /// A slice of a spare checkpoint's states: its next layers until <see cref="SpareSliceTicks"/> have gone by. True when it
        /// did any work (the caller asks again); false when there is nothing to do or no room.
        /// </summary>
        private bool BuildSpareCheckpointStates()
        {
            if (_spareStatesBuild == null)
            {
                if (!WantsSpareCheckpointStates || _spareStatesRefused) return false;
                if (DeviceRoomManaged)
                {
                    var (free, total) = DeviceMemory();
                    if (total > 0 && free < _slotStateBytes + DeviceHeadroom(total) + PromptStepBytes)
                    {
                        _spareStatesRefused = true;
                        return false;
                    }
                }
                _spareStatesBuild = new RecurrentState[Config.NumLayers][];
                _spareStatesNext = 0;
            }
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            _slotMutationDepth++;   // a convenience: the allocator's reclaimer releases nothing for it
            NoteDeviceMemoryChanged();
            try
            {
                while (_spareStatesNext < Config.NumLayers)
                {
                    NewRecurrentLayer(_spareStatesBuild, _spareStatesNext++);
                    if (System.Diagnostics.Stopwatch.GetTimestamp() - start >= SpareSliceTicks) break;
                }
            }
            finally
            {
                _slotMutationDepth--;
            }
            if (_spareStatesNext < Config.NumLayers) return true;
            _spareCheckpointStates = _spareStatesBuild;
            _spareStatesBuild = null;
            SpareCheckpointStatesMade++;
            return true;
        }

        /// <summary>Tests: the spare checkpoint states (null: none).</summary>
        internal RecurrentState[][] SpareCheckpointStatesForTests => _spareCheckpointStates;

        /// <summary>Tests: make the spare checkpoint states now, as idle slices would.</summary>
        internal void MakeSpareCheckpointStatesForTests()
        {
            while (_spareCheckpointStates == null && BuildSpareCheckpointStates()) { }
        }

        /// <summary>A new checkpoint's states: the spare, else the one being made (its remaining layers made now), else new
        /// ones.</summary>
        private RecurrentState[][] TakeCheckpointStates()
        {
            RecurrentState[][] states = _spareCheckpointStates;
            _spareCheckpointStates = null;
            _spareStatesRefused = false;
            if (states == null && _spareStatesBuild != null)
            {
                states = _spareStatesBuild;
                _spareStatesBuild = null;
                for (; _spareStatesNext < Config.NumLayers; _spareStatesNext++) NewRecurrentLayer(states, _spareStatesNext);
            }
            return states ?? NewRecurrentStates();
        }

        /// <summary>The spare checkpoint states go (making room); the bytes freed.</summary>
        private long DisposeSpareCheckpointStates()
        {
            if (_spareCheckpointStates == null && _spareStatesBuild == null) return 0;
            long bytes = 0;
            foreach (RecurrentState[][] states in new[] { _spareCheckpointStates, _spareStatesBuild })
                if (states != null)
                    foreach (RecurrentState[] layer in states)
                        if (layer != null)
                            foreach (RecurrentState state in layer)
                                if (state != null) bytes += state.StateBytes;
            DisposeStates(_spareCheckpointStates);
            DisposeStates(_spareStatesBuild);
            _spareCheckpointStates = _spareStatesBuild = null;
            NoteDeviceMemoryChanged();
            return bytes;
        }

        // ---------------------------------------------------------------- the write floor

        /// <summary>
        /// Before a step writes <paramref name="slots"/>' rows from <paramref name="starts"/> on: every checkpoint sharing a slot's
        /// rows past its write is invalidated first (reported to the prefix cache), so a shared row is never rewritten under a
        /// checkpoint. A no-op where no checkpoint shares rows (and where every write is at or past every checkpoint's length,
        /// as a request's prompt chunks, decode rows, verify windows and draft rows are).
        /// </summary>
        private void PrepareWrites(Slot[] slots, int[] starts)
        {
            if (_rowShares is not { Count: > 0 } || slots == null) return;
            for (int s = 0; s < slots.Length && s < starts.Length; s++)
                PrepareWrite(slots[s]?.K, starts[s]);
        }

        /// <inheritdoc cref="PrepareWrites"/>
        private void PrepareWrite(Tensor[][] k, int start)
        {
            if (k == null || _rowShares == null || !_rowShares.TryGetValue(k, out List<Slot> dependents)) return;
            foreach (Slot d in dependents.Where(d => d.Length > start).ToList())
            {
                SharedCheckpointsInvalidatedByWrite++;
                if (!_writeBelowShareWarned)
                {
                    _writeBelowShareWarned = true;
                    Console.Error.WriteLine($"[{Config.Architecture}] a step wrote a slot's rows from {start:N0} on, below a prompt-end " +
                        $"checkpoint's {d.Length:N0} that shares them: the checkpoint was dropped first (its conversation's next turn " +
                        "prefills what it held). Reported once.");
                }
                InvalidateDependent(d, InvalidationReason.TrimmedByModel);
            }
        }

        /// <summary>The prefix-cache key a retained slot is kept under (null where it is not retained).</summary>
        private string RetainedKeyOf(Slot slot)
        {
            if (_retainedSlots != null)
                foreach (var (key, retained) in _retainedSlots)
                    if (ReferenceEquals(retained, slot)) return key;
            return null;
        }

        /// <summary>A checkpoint goes: out of the retained set, its states freed, reported to the prefix cache.</summary>
        private void InvalidateDependent(Slot dependent, InvalidationReason reason)
        {
            string key = RetainedKeyOf(dependent);
            if (key != null) _retainedSlots.Remove(key);
            DisposeDependent(dependent);
            if (key != null) _prefixCacheSink?.OnPayloadInvalidated(key, reason);
        }

        /// <summary>Every checkpoint over <paramref name="k"/> goes (the rows they shared are gone).</summary>
        private void InvalidateDependents(Tensor[][] k, InvalidationReason reason)
        {
            if (k == null || _rowShares == null || !_rowShares.TryGetValue(k, out List<Slot> dependents)) return;
            foreach (Slot d in dependents.ToList()) InvalidateDependent(d, reason);
        }

        // ---------------------------------------------------------------- lifecycle

        /// <summary>A checkpoint's own parts go (its states and carry); the rows it shared stay its owner's.</summary>
        private void DisposeDependent(Slot dependent)
        {
            if (dependent?.K == null) return;
            _slotMutationDepth++;
            try
            {
                Unregister(dependent);
                _spareStatesRefused = false;
                ForgetGraphsOfStates(dependent);
                DisposeStates(dependent.Recurrent);
                dependent.Recurrent = null;
                dependent.DraftCarry?.Dispose();
                dependent.DraftCarry = null;
                dependent.K = dependent.V = null;
            }
            finally
            {
                _slotMutationDepth--;
                NoteDeviceMemoryChanged();
            }
        }

        private void Unregister(Slot dependent)
        {
            if (_rowShares == null || dependent.K == null || !_rowShares.TryGetValue(dependent.K, out List<Slot> dependents)) return;
            dependents.Remove(dependent);
            if (dependents.Count == 0) _rowShares.Remove(dependent.K);
        }

        /// <summary>
        /// The steps and recordings bound to a checkpoint's states go (its rows stay its owner's, whose steps and recordings
        /// stay). A checkpoint never ran, so nothing binds them but what a hand-over made it: <see cref="ForgetGraphsOf"/> would
        /// have dropped every recording over the owner's rows too.
        /// </summary>
        private void ForgetGraphsOfStates(Slot slot)
        {
            if (slot?.Recurrent == null) return;
            DropReplaysOf(null, slot.Recurrent, park: false);
            DisposeParkedBinding(BoundStorages(new[] { new Slot { Recurrent = slot.Recurrent } }));
        }

        /// <summary>
        /// <paramref name="leaving"/> is freed while checkpoints share its rows: the longest takes them over (it holds them and
        /// its own states, the state at its length, as a slot of its own), the others stay over them, and
        /// <paramref name="leaving"/> frees its states only. Its steps and recordings go: they bound states that are gone.
        /// </summary>
        private void HandOverRows(Slot leaving)
        {
            List<Slot> dependents = _rowShares[leaving.K];
            Slot heir = dependents.OrderByDescending(d => d.Length).ThenByDescending(d => d.RetainedAt).First();
            dependents.Remove(heir);
            if (dependents.Count == 0) _rowShares.Remove(leaving.K);
            _slotMutationDepth++;
            try
            {
                ForgetGraphsOf(leaving);
                heir.SharesRows = false;
                heir.Capacity = leaving.Capacity;
                DisposeStates(leaving.Recurrent);
                leaving.Recurrent = null;
                leaving.DraftCarry?.Dispose();
                leaving.DraftCarry = null;
                leaving.K = leaving.V = null;
                RowHandOvers++;
                ShareDebug($"{RetainedKeyOf(heir)}: a freed slot's rows went to its checkpoint of {heir.Length:N0} rows ({dependents.Count} more over them)");
            }
            finally
            {
                _slotMutationDepth--;
                NoteDeviceMemoryChanged();
            }
        }

        /// <summary>The checkpoints over <paramref name="oldK"/>'s rows now share <paramref name="to"/>'s, which hold the same
        /// rows below every checkpoint's length (a copy of the slot, or the slot grown); the slot that held them has none.</summary>
        private void RepointDependents(Tensor[][] oldK, Slot to)
        {
            if (oldK == null || _rowShares == null || ReferenceEquals(oldK, to.K) || !_rowShares.Remove(oldK, out List<Slot> dependents))
                return;
            foreach (Slot d in dependents)
            {
                d.K = to.K;
                d.V = to.V;
                d.Capacity = to.Capacity;
            }
            if (_rowShares.TryGetValue(to.K, out List<Slot> already)) already.AddRange(dependents);
            else _rowShares.Add(to.K, dependents);
            RowRepoints++;
        }

        /// <summary>The active cache's tensors were replaced inside its arrays by larger ones (<see cref="EnsureCacheCapacity"/>):
        /// the checkpoints over them read them at the new capacity.</summary>
        private void NoteShareCapacity(Tensor[][] k, int capacity)
        {
            if (k != null && _rowShares != null && _rowShares.TryGetValue(k, out List<Slot> dependents))
                foreach (Slot d in dependents) d.Capacity = capacity;
        }

        /// <summary>The bytes releasing <paramref name="slot"/> frees: a checkpoint's states, an owner's states (its rows go to
        /// a checkpoint), else its caches and states.</summary>
        private long ReleasedBytes(Slot slot) => InAnyShare(slot) ? RecurrentStateBytes(slot) : SlotStorageBytes(slot);

        // ---------------------------------------------------------------- adoption

        IReadOnlyList<string> IPrefixCacheModel.RowSharers(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out Slot checkpoint) || !checkpoint.SharesRows) return Array.Empty<string>();
            var (ownerKey, _) = RetainedOwner(checkpoint.K);
            return ownerKey == null ? null : StolenKeys(checkpoint, ownerKey);
        }

        /// <summary>The retained slot holding <paramref name="k"/>'s rows and its key; (null, null) where a request runs on them.</summary>
        private (string Key, Slot Owner) RetainedOwner(Tensor[][] k)
        {
            if (_retainedSlots != null)
                foreach (var (key, slot) in _retainedSlots)
                    if (!slot.SharesRows && ReferenceEquals(slot.K, k)) return (key, slot);
            return (null, null);
        }

        /// <summary>The payloads a steal of <paramref name="checkpoint"/>'s rows uses up: the owner's end state and every longer
        /// checkpoint over the rows (the adopter writes from the checkpoint's length on).</summary>
        private List<string> StolenKeys(Slot checkpoint, string ownerKey)
        {
            var keys = new List<string> { ownerKey };
            foreach (Slot d in _rowShares[checkpoint.K])
                if (!ReferenceEquals(d, checkpoint) && d.Length > checkpoint.Length && RetainedKeyOf(d) is { } key)
                    keys.Add(key);
            return keys;
        }

        bool IPrefixCacheModel.TryMaterialize(in MaterializeRequest request)
        {
            _lastMaterializeWantedRoom = false;
            if (!string.IsNullOrEmpty(request.PayloadKey) && TryGetRetained(request.PayloadKey, out Slot slot) && slot.SharesRows)
            {
                if (string.IsNullOrEmpty(request.TargetRequestId)
                    || !CanMaterialize(request.PayloadKey, request.PayloadTokens, request.TargetTokens))
                    return false;
                return MaterializeShared(request.PayloadKey, slot, request.TargetRequestId, request.Op == MaterializeOp.Donate,
                    request.TargetTokens == slot.Length ? request.Consumable : null, request.RequiredContextTokens);
            }
            return HolderPrefixCacheAdapter.TryMaterialize(this, request);
        }

        /// <summary>
        /// A request adopts a shared checkpoint. Where its rows' holder is a retained end state and every payload a write past
        /// the checkpoint uses up is <paramref name="consumable"/> (its own conversation's), the request takes that slot over:
        /// the slot's length goes back to the checkpoint's, the checkpoint's states are written into the slot's own in place
        /// (their recordings still bind them), and the used-up payloads are reported. Otherwise the checkpoint is copied as a
        /// retained slot is cloned (room made first; declined without room, <see cref="IPrefixCacheModel.LastMaterializeWantedRoom"/>).
        /// A donated checkpoint is used up; a cloned one stays over the rows.
        /// </summary>
        private bool MaterializeShared(string payloadKey, Slot checkpoint, string requestId, bool donate,
            IReadOnlyList<string> consumable, int requiredContextTokens)
        {
            _slots ??= new Dictionary<string, Slot>(StringComparer.Ordinal);
            if (_slots.ContainsKey(requestId) || string.Equals(_activeKey, requestId, StringComparison.Ordinal)) return false;
            var (ownerKey, owner) = RetainedOwner(checkpoint.K);
            List<string> stolen = owner == null ? null : StolenKeys(checkpoint, ownerKey);
            if (stolen != null && consumable != null && stolen.TrueForAll(key => consumable.Contains(key)))
            {
                Steal(checkpoint, ownerKey, owner, requestId);
                if (donate) ConsumeCheckpoint(payloadKey, checkpoint);
                return true;
            }
            int capacity = Math.Max(checkpoint.Length, requiredContextTokens);
            ShareDebug($"{requestId} copies checkpoint {payloadKey} of {checkpoint.Length:N0} rows (" +
                (owner == null ? "a request runs on its rows" : $"{stolen.Count - (consumable?.Count(stolen.Contains) ?? 0)} of {stolen.Count} payloads not its to use up") + ")");
            if (!RoomForSlotCopy(checkpoint, capacity, mayEvict: true, needed: true, "a clone of a cached prefix", out int copyCapacity))
            {
                _lastMaterializeWantedRoom = true;
                return false;
            }
            _slots.Add(requestId, CopySlot(checkpoint, copyCapacity));
            if (donate) ConsumeCheckpoint(payloadKey, checkpoint);
            return true;
        }

        /// <summary>The request takes <paramref name="owner"/>'s slot at <paramref name="checkpoint"/>'s state (see
        /// <see cref="MaterializeShared"/>).</summary>
        private void Steal(Slot checkpoint, string ownerKey, Slot owner, string requestId)
        {
            _slotMutationDepth++;
            try
            {
                _retainedSlots.Remove(ownerKey);
                var used = new List<string> { ownerKey };
                foreach (Slot d in _rowShares[checkpoint.K].Where(d => d.Length > checkpoint.Length).ToList())
                {
                    if (RetainedKeyOf(d) is { } key) used.Add(key);
                    InvalidateDependentQuietly(d);
                }
                bool kept = true;
                if (owner.Recurrent != null)
                    for (int l = 0; l < owner.Recurrent.Length; l++)
                        if (owner.Recurrent[l] != null)
                            for (int r = 0; r < owner.Recurrent[l].Length; r++)
                                kept &= owner.Recurrent[l][r].CopyFromInPlace(checkpoint.Recurrent[l][r]);
                if (!kept) ClearGraphCache();
                owner.Length = checkpoint.Length;
                owner.DraftStart = checkpoint.DraftStart;
                owner.RopeDelta = checkpoint.RopeDelta;
                owner.DraftFollowed = -1;
                owner.DraftCarry?.Dispose();
                owner.DraftCarry = null;
                owner.VerifyRows = 0;
                owner.Spare = false;
                _slots[requestId] = owner;
                RowSteals++;
                ShareDebug($"{requestId} took over {ownerKey}'s slot of {owner.Capacity:N0} cells at its checkpoint's {checkpoint.Length:N0} rows " +
                    $"(no copy; {used.Count} payload{(used.Count == 1 ? "" : "s")} used up)");
                foreach (string key in used) _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.TrimmedByModel);
            }
            finally
            {
                _slotMutationDepth--;
            }
        }

        /// <summary>A checkpoint goes from the retained set and its states are freed; the caller reports it.</summary>
        private void InvalidateDependentQuietly(Slot dependent)
        {
            if (RetainedKeyOf(dependent) is { } key) _retainedSlots.Remove(key);
            DisposeDependent(dependent);
        }

        /// <summary>A donated checkpoint leaves the cache (the prefix tree detaches it): its states go.</summary>
        private void ConsumeCheckpoint(string payloadKey, Slot checkpoint)
        {
            _retainedSlots.Remove(payloadKey);
            DisposeDependent(checkpoint);
        }

        // ---------------------------------------------------------------- invariants

        /// <summary>
        /// The share invariants (tests run them before every unified step, <see cref="SlotInvariantChecksForTests"/>): no shared
        /// array is pooled; no checkpoint runs a request; every shared array has exactly one holder that is not a checkpoint
        /// (running or retained); every checkpoint over it is retained, as long as its holder at most, of its capacity.
        /// </summary>
        internal void CheckSlotShareInvariants()
        {
            void Fail(string what) => throw new InvalidOperationException($"{Config.Architecture}: slot share invariant: {what}");
            if (_slots != null)
                foreach (var (id, slot) in _slots)
                    if (slot.SharesRows) Fail($"request {id} runs on a checkpoint");
            if (_retainedSlots != null)
                foreach (var (key, slot) in _retainedSlots)
                    if (slot.SharesRows && (slot.K == null || _rowShares == null || !_rowShares.TryGetValue(slot.K, out var list) || !list.Contains(slot)))
                        Fail($"checkpoint {key} is not registered over its rows");
            if (_rowShares == null) return;
            foreach (var (k, dependents) in _rowShares)
            {
                if (dependents.Count == 0) Fail("a share without checkpoints");
                if (_slotPool != null && _slotPool.Exists(p => ReferenceEquals(p.K, k))) Fail("a shared slot is pooled");
                var holders = new List<Slot>();
                if (_slots != null)
                    foreach (var (id, slot) in _slots)
                        if (!string.Equals(id, _activeKey, StringComparison.Ordinal) && ReferenceEquals(slot.K, k)) holders.Add(slot);
                if (_activeKey != null && ReferenceEquals(_cacheK, k)) holders.Add(SnapshotActiveFields());
                if (_retainedSlots != null)
                    foreach (Slot slot in _retainedSlots.Values)
                        if (!slot.SharesRows && ReferenceEquals(slot.K, k)) holders.Add(slot);
                if (holders.Count != 1) Fail($"{holders.Count} holders of shared rows");
                Slot owner = holders[0];
                foreach (Slot d in dependents)
                {
                    if (!d.SharesRows || !ReferenceEquals(d.K, k)) Fail("a registered checkpoint over other rows");
                    if (RetainedKeyOf(d) == null) Fail("a registered checkpoint is not retained");
                    if (d.Length > owner.Length) Fail($"a checkpoint of {d.Length} rows over a slot of {owner.Length}");
                    if (d.Capacity != owner.Capacity) Fail($"a checkpoint of {d.Capacity} cells over a slot of {owner.Capacity}");
                }
            }
        }

        /// <summary>Tests: whether a retained payload is a checkpoint sharing another slot's rows (null: no such payload).</summary>
        internal bool? RetainedSharesRowsForTests(string payloadKey)
            => TryGetRetained(payloadKey, out Slot slot) ? slot.SharesRows : null;

        /// <summary>Tests: a checksum of a retained payload's or a running request's rows [0, its length) in every cache and its
        /// states (a shared checkpoint's rows must never change while it lasts).</summary>
        internal ulong RowChecksumForTests(string requestOrPayloadKey, int rows = -1)
        {
            if (!TryGetSlot(requestOrPayloadKey, out Slot slot) && !TryGetRetained(requestOrPayloadKey, out slot))
                throw new ArgumentException($"no slot or payload {requestOrPayloadKey}");
            int length = rows < 0 ? slot.Length : rows;
            ulong hash = 1469598103934665603UL;
            void Add(float[] values)
            {
                foreach (float v in values)
                    hash = unchecked((hash ^ (uint)BitConverter.SingleToInt32Bits(v)) * 1099511628211UL);
            }
            foreach (Tensor t in AllCacheTensors(slot.K).Concat(AllCacheTensors(slot.V)))
                for (int h = 0; h < t.Sizes[0] && length > 0; h++)
                {
                    using Tensor head = t.Narrow(0, h, 1);
                    using Tensor part = head.Narrow(1, 0, length);
                    using Tensor flat = Ops.NewContiguous(part);   // F32 caches (the managed CPU executor's)
                    Add(flat.GetElementsAsFloat((int)flat.ElementCount()));
                }
            foreach (Tensor t in CheckpointStateTensors(slot)) Add(t.GetElementsAsFloat((int)t.ElementCount()));
            return hash;
        }
    }
}
