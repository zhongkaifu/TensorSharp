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
            public RecurrentState[][] Recurrent;   // [layer][rank]; hybrid stacks only
            public float[] Logits;                 // reused by batched steps
            public bool DeviceDirty;               // a graph wrote its rows/state in device memory
            public int DraftStart;                 // the draft layers attend to rows from here on
            public int RopeDelta;                  // rotary position = KV index + this past a position table
            public int VerifyRows;                 // a batched verify's window, its recurrent states awaiting the commit
        }

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
        private const int FreshSlotTokens = 512;

        private Dictionary<string, Slot> _slots;            // requests with their own slot
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
            Recurrent = _recurrentPrimary,
            DeviceDirty = _cacheDeviceDirty,
            DraftStart = _draftStart,
            RopeDelta = _ropeDelta,
        };

        private void LoadActive(Slot slot)
        {
            _cacheK = slot.K;
            _cacheV = slot.V;
            _cacheCapacity = slot.Capacity;
            _cacheSeqLen = slot.Length;
            _recurrentPrimary = slot.Recurrent;
            _cacheDeviceDirty = slot.DeviceDirty;
            _draftStart = slot.DraftStart;
            _ropeDelta = slot.RopeDelta;
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
            int capacity = Math.Min(_maxContextLength, Math.Max(Math.Max(1, reserveTokens), Math.Min(FreshSlotTokens, Math.Max(1, _maxContextLength))));
            var slot = new Slot { Capacity = capacity, Recurrent = _anyRecurrent ? NewRecurrentStates() : null };
            (slot.K, slot.V) = AllocateCaches(capacity);
            PrimeDeviceCopies(slot, overwritten: false);
            if (_slotStateBytes < 0) MeasureSlotBytes(slot);
            return slot;
        }

        // ---------------------------------------------------------------- device-memory admission

        // A slot's cache bytes per token and its states' bytes, measured on the first slot (-1 until then).
        private long _slotBytesPerToken = -1, _slotStateBytes = -1;

        /// <summary>Beside a slot's caches and states: its drafter buffers and its share of the graphs' memory (the margin
        /// Qwen4ExpModel keeps for a holder on Metal).</summary>
        private const long SlotExtraBytes = 256L << 20;

        private void MeasureSlotBytes(Slot slot)
        {
            long cache = 0;
            foreach (Tensor t in AllCacheTensors(slot.K).Concat(AllCacheTensors(slot.V))) cache += t.Storage.ByteLength;
            long states = 0;
            if (slot.Recurrent != null)
                foreach (RecurrentState[] layer in slot.Recurrent)
                    if (layer != null)
                        foreach (RecurrentState state in layer)
                            if (state != null)
                                states += state.StateBytes > 0 ? state.StateBytes
                                    : state.CheckpointTensors.Sum(t => t.ElementCount() * t.ElementType.Size());
            _slotBytesPerToken = cache / Math.Max(1, slot.Capacity);
            _slotStateBytes = states;
        }

        /// <summary>Device bytes a slot of <paramref name="capacity"/> tokens takes, with its margin; -1 before the first
        /// slot was measured.</summary>
        private long SlotDeviceBytes(int capacity)
            => _slotStateBytes < 0 ? -1 : _slotBytesPerToken * capacity + _slotStateBytes + SlotExtraBytes;

        /// <summary>
        /// Whether one device's free memory bounds how many requests hold slots: Metal's working set, or one ggml_cuda GPU
        /// (a split or tensor-parallel mesh plans its devices explicitly), as Qwen4ExpModel and Qwen35Model admit their
        /// holders. Without it the runtime allocated a slot for every concurrent request and kept every finished one: two
        /// Qwen3.8-Flash-Next conversations on a 48 GB Mac reached 38 GB wired, where Qwen4ExpModel stayed at 32 GB; an
        /// allocation past a CUDA GPU's memory aborts the process inside ggml.
        /// </summary>
        private bool AdmitsSlotsByDeviceMemory => IsGgmlBackend && _mesh.Stages == 1 && _mesh.TensorParallel == 1
            && LayerSplitGroup == null && (_backend == BackendType.GgmlMetal || _backend == BackendType.GgmlCuda);

        /// <summary>The device memory kept free beside the slots: the larger of 1 GB and 1/32 of it.</summary>
        private static long DeviceHeadroom(long totalBytes) => Math.Max(1L << 30, totalBytes / 32);

        /// <summary>The device's free and total bytes (the GGML backend's report; tests substitute their own).</summary>
        internal Func<(long Free, long Total)?> DeviceMemoryProbe { get; set; } = static ()
            => GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) && total > 0 ? (free, total) : null;

        /// <summary>What the device lacks for <paramref name="needBytes"/> beside its headroom, after the kept graph arenas
        /// are released (0 when it fits or is not measured).</summary>
        private long DeviceDeficit(long needBytes)
        {
            if (needBytes <= 0 || DeviceMemoryProbe() is not { } memory) return 0;
            long deficit = Math.Max(0, needBytes + DeviceHeadroom(memory.Total) - memory.Free);
            if (deficit > 0 && GgmlBasicOps.ReleaseKeptArenas() > 0 && DeviceMemoryProbe() is { } after)
                deficit = Math.Max(0, needBytes + DeviceHeadroom(after.Total) - after.Free);
            return deficit;
        }

        /// <summary>The prefix cache evicts finished requests' slots before a new request needs their room: what the device
        /// lacks for one more fresh slot (<see cref="ModelBase.QueryHostMemoryDeficitBytes"/>).</summary>
        protected override long DeviceMemoryDeficitBytes()
            => AdmitsSlotsByDeviceMemory ? DeviceDeficit(SlotDeviceBytes(FreshSlotTokens)) : 0;

        /// <summary><see cref="DeviceMemoryDeficitBytes"/> alone (tests).</summary>
        internal long DeviceSlotDeficitBytes => DeviceMemoryDeficitBytes();

        /// <summary>A new request's slot of <paramref name="capacity"/> tokens must fit on the device while other requests
        /// hold theirs; otherwise it waits for one of them to finish (<see cref="SequenceSlotUnavailableException"/>, which
        /// the scheduler re-queues). A lone request always gets its slot.</summary>
        private void EnsureDeviceRoomForSlot(string requestId, int capacity)
        {
            if (!AdmitsSlotsByDeviceMemory || _slots == null || _slots.Count == 0 || (_slotPool?.Count ?? 0) > 0) return;
            long need = SlotDeviceBytes(capacity);
            long deficit = DeviceDeficit(need);
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
        /// checkpoint's), 31-65 ms each. A state whose reset is not all zeros keeps the upload.
        /// </summary>
        private void PrimeDeviceCopies(Slot slot, bool overwritten)
        {
            // One device, or the ranks of per-rank tensor-parallel graphs: each rank primes its share on its own device.
            bool ranked = RankedSlotCopies;
            if (!IsGgmlBackend || _mesh.Stages != 1 || (_mesh.TensorParallel != 1 && !ranked)
                || GraphExecutor is not { Capabilities: { BindsHostPointers: true, SeparateDeviceCopies: true } })
                return;
            // Each by the bytes a graph binds it with (ggml's size of the leaf): a cache whole, a state's F32 view by its
            // own elements (the state view of a backing is not the whole storage). The caches and the states each share
            // one device allocation (GgmlBasicOps.ZeroDeviceCopies): a slot's ~80 tensors were as many allocations, 0.7 ms
            // each on ggml-vulkan. Apart, a slot that grows frees its old caches' buffer whole.
            var keys = new List<IntPtr>();
            var bytes = new List<long>();
            int previous = ranked ? GgmlBasicOps.GetActiveRank() : 0;
            try
            {
                for (int rank = 0; rank < _mesh.TensorParallel; rank++)
                {
                    if (ranked) GgmlBasicOps.SetActiveRank(_mesh.GlobalDevice(0, rank));
                    foreach (Tensor t in RankTensors(slot.K, rank)) Add(t, t.Storage.ByteLength);
                    foreach (Tensor t in RankTensors(slot.V, rank)) Add(t, t.Storage.ByteLength);
                    Prime();
                    if (slot.Recurrent == null) continue;
                    foreach (RecurrentState[] layer in slot.Recurrent)
                        if (layer != null && rank < layer.Length && layer[rank]?.GraphTensors is { } tensors
                            && (overwritten || layer[rank].GraphTensorsResetToZero))
                            foreach (Tensor t in tensors) Add(t, t.ElementCount() * sizeof(float));
                    Prime();
                }
            }
            finally
            {
                if (ranked) GgmlBasicOps.SetActiveRank(previous);
            }

            static IEnumerable<Tensor> RankTensors(Tensor[][] caches, int rank)
            {
                if (caches == null) yield break;
                foreach (Tensor[] entry in caches)
                    if (entry != null && rank < entry.Length && entry[rank] != null) yield return entry[rank];
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
        /// pool has none.</summary>
        private Slot TakePooledSlot(int tokens, bool mustFit)
        {
            if (_slotPool == null || _slotPool.Count == 0) return null;
            int pick = 0;
            for (int i = 1; i < _slotPool.Count; i++)
            {
                int best = _slotPool[pick].Capacity, cap = _slotPool[i].Capacity;
                bool fits = cap >= tokens, bestFits = best >= tokens;
                if (fits ? !bestFits || cap < best : !bestFits && cap > best) pick = i;
            }
            Slot pooled = _slotPool[pick];
            if (mustFit && pooled.Capacity < tokens) return null;
            _slotPool.RemoveAt(pick);
            pooled.Length = 0;
            pooled.DraftStart = 0;
            pooled.RopeDelta = 0;
            pooled.VerifyRows = 0;
            return pooled;
        }

        /// <summary>A slot's caches at <paramref name="capacity"/> cells: K and V per layer and rank, then the extra key
        /// caches (K only).</summary>
        private (Tensor[][] K, Tensor[][] V) AllocateCaches(int capacity)
        {
            var k = new Tensor[CacheEntries][];
            var v = new Tensor[CacheEntries][];
            for (int e = 0; e < CacheEntries; e++)
            {
                Tensor first = NewCacheTensor(e, 0, capacity, value: false);
                if (first == null) continue;
                k[e] = new Tensor[_mesh.TensorParallel];
                k[e][0] = first;
                for (int r = 1; r < _mesh.TensorParallel; r++) k[e][r] = NewCacheTensor(e, r, capacity, value: false);
                if (e >= TotalLayers) continue;
                v[e] = new Tensor[_mesh.TensorParallel];
                for (int r = 0; r < _mesh.TensorParallel; r++) v[e][r] = NewCacheTensor(e, r, capacity, value: true);
            }
            return (k, v);
        }

        private void DisposeSlot(Slot slot)
        {
            if (slot == null) return;
            ClearGraphCache();
            foreach (Tensor t in AllCacheTensors(slot.K)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
            foreach (Tensor t in AllCacheTensors(slot.V)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
            slot.K = slot.V = null;
            DisposeStates(slot.Recurrent);
            slot.Recurrent = null;
        }

        private void RecycleSlot(Slot slot)
        {
            if (slot?.K == null) return;
            _slotPool ??= new List<Slot>(SlotPoolMax);
            if (_slotPool.Count < Math.Min(SlotPoolMax, ExecutionOptions.FromEnvironment().KvHolderPoolMax))
                _slotPool.Add(slot);
            else
                DisposeSlot(slot);
        }

        /// <summary>Grow a slot's caches to hold <paramref name="required"/> tokens, keeping its rows.</summary>
        private void EnsureSlotCapacity(Slot slot, int required)
        {
            if (required <= slot.Capacity) return;
            SyncToHost(slot);
            if (required > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {required} exceeds configured max context {_maxContextLength}.");
            int capacity = Math.Max(slot.Capacity, 1);
            while (capacity < required)
                capacity = Math.Min(_maxContextLength, capacity * 2);
            (Tensor[][] k, Tensor[][] v) = AllocateCaches(capacity);
            ClearGraphCache();
            CopyRows(slot.K, k, slot.Length);
            CopyRows(slot.V, v, slot.Length);
            foreach (Tensor t in AllCacheTensors(slot.K)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
            foreach (Tensor t in AllCacheTensors(slot.V)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
            slot.K = k;
            slot.V = v;
            slot.Capacity = capacity;
        }

        private static void CopyRows(Tensor[][] from, Tensor[][] to, int rows)
        {
            if (rows <= 0 || from == null) return;
            for (int l = 0; l < from.Length; l++)
            {
                if (from[l] == null) continue;
                for (int r = 0; r < from[l].Length; r++)
                {
                    using Tensor src = from[l][r].Narrow(1, 0, rows);
                    using Tensor dst = to[l][r].Narrow(1, 0, rows);
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
            RecycleSlot(slot);
        }

        public bool RetainSequenceCache(string requestId) => RetainSequenceCacheAs(requestId, requestId);

        /// <summary>Move <paramref name="requestId"/>'s slot into the retained set under
        /// <paramref name="key"/> (zero copy).</summary>
        public bool RetainSequenceCacheAs(string requestId, string key)
        {
            if (_slots == null || string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(key)) return false;
            if (!_slots.TryGetValue(requestId, out Slot slot)) return false;
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
            return true;
        }

        public bool TryRebindRetainedCache(string retainedKey, string newRequestId)
        {
            if (_retainedSlots == null || string.IsNullOrEmpty(retainedKey) || string.IsNullOrEmpty(newRequestId)) return false;
            if (!_retainedSlots.TryGetValue(retainedKey, out Slot slot) || slot.K == null) return false;
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
            _slots = _retainedSlots = null;
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
        private bool TryForwardSlots(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> chunks, int[] starts, float[][] outLogits,
            float[][] features = null)
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
                EnsureSlotCapacity(slots[s], checked(starts[s] + counts[s]));

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
            try
            {
                embedded = EmbedStep(flatTokens);
                bool capture = features != null && Array.Exists(features, f => f != null);
                if (capture ? TryRunCapturingGraph(embedded, slots, queryStart, counts, starts, outLogits, features)
                    : TryRunGraph(embedded, slots, queryStart, counts, (int[])starts.Clone(), wantLogits: true, out float[][] graphLogits)
                      && CopyLogits(graphLogits, outLogits))
                {
                    BatchedForwardCount++;
                    for (int s = 0; s < n; s++)
                        slots[s].Length = starts[s] + counts[s];
                    return true;
                }
                static bool CopyLogits(float[][] from, float[][] to)
                {
                    Array.Copy(from, to, to.Length);
                    return true;
                }
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
                hidden = RunLayers(hidden, TokenPositions.Explicit(positions), batch);
                all = ProjectRows(hidden, lastRows);
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
            return count == 1
                ? DecodeAttention(scope, cacheK, cacheV, slot.Capacity, q, k, v, start)
                : PrefillAttention(scope, cacheK, cacheV, q, k, v, start, count);
        }
    }
}
