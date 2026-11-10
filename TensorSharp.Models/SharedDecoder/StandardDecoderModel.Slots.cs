// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

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

        private Slot SnapshotActive() => new()
        {
            K = _cacheK,
            V = _cacheV,
            Capacity = _cacheCapacity,
            Length = _cacheSeqLen,
            Recurrent = _recurrentPrimary,
            DeviceDirty = _cacheDeviceDirty,
        };

        private void LoadActive(Slot slot)
        {
            _cacheK = slot.K;
            _cacheV = slot.V;
            _cacheCapacity = slot.Capacity;
            _cacheSeqLen = slot.Length;
            _recurrentPrimary = slot.Recurrent;
            _cacheDeviceDirty = slot.DeviceDirty;
        }

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
                                foreach (Tensor t in state.HostTensors) SyncTensorHostCache(t);
            slot.DeviceDirty = false;
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

        private Slot NewSlot()
        {
            if (_slotPool != null && _slotPool.Count > 0)
            {
                Slot pooled = _slotPool[^1];
                _slotPool.RemoveAt(_slotPool.Count - 1);
                pooled.Length = 0;
                ClearGraphCache();
                ResetStates(pooled.Recurrent);
                return pooled;
            }
            int capacity = Math.Min(FreshSlotTokens, Math.Max(1, _maxContextLength));
            var slot = new Slot { Capacity = capacity, Recurrent = _anyRecurrent ? NewRecurrentStates() : null };
            (slot.K, slot.V) = AllocateCaches(capacity);
            return slot;
        }

        private (Tensor[][] K, Tensor[][] V) AllocateCaches(int capacity)
        {
            DType dtype = _kvCacheDtype.ToDType();
            var k = new Tensor[Config.NumLayers][];
            var v = new Tensor[Config.NumLayers][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!OwnsLayer(l) || _recurrentLayer[l]) continue;
                int stage = _mesh.StageOf(l);
                k[l] = new Tensor[_mesh.TensorParallel];
                v[l] = new Tensor[_mesh.TensorParallel];
                for (int r = 0; r < _mesh.TensorParallel; r++)
                {
                    IAllocator alloc = _mesh.Allocator(stage, r);
                    k[l][r] = new Tensor(alloc, dtype, _localKvHeads, capacity, _keyDim);
                    v[l][r] = new Tensor(alloc, dtype, _localKvHeads, capacity, _valueDim);
                    InitializeCacheTensor(k[l][r]);
                    InitializeCacheTensor(v[l][r]);
                }
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
            _retainedSlots.Add(key, slot);
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
        private bool TryForwardSlots(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> chunks, int[] starts, float[][] outLogits)
        {
            if (!SupportsPerSequenceFusedForward || _slots == null || HasPendingMedia) return false;
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
                for (int i = 0; i < counts[s]; i++) positions[queryStart[s] + i] = starts[s] + i;
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
                embedded = Embedding(flatTokens);
                if (TryRunGraph(embedded, slots, queryStart, counts, (int[])starts.Clone(), wantLogits: true, out float[][] graphLogits))
                {
                    BatchedForwardCount++;
                    for (int s = 0; s < n; s++)
                    {
                        slots[s].Length = starts[s] + counts[s];
                        outLogits[s] = graphLogits[s];
                    }
                    return true;
                }
            }
            finally
            {
                embedded?.Dispose();
            }
            foreach (Slot slot in slots) SyncToHost(slot);
            ClearGraphCache();

            float[] all;
            // A step with a text-LoRA adapter selected must apply it to every planned weight.
            bool loraAudit = OpenLoraForwardAudit();
            _forwardSw.Start();
            try
            {
                BatchedForwardCount++;
                EnterStage(0);
                Tensor hidden = Embedding(flatTokens);
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
