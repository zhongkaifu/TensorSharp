// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Per-request sequence slots for the server's continuous-batching engine
// (the GLM analogue of DeepSeek4Model.PerSeqCache).
//
// The glm-dsa whole-model native executor owns every cache on the device — the
// 576-wide MLA rows and the indexer key cache, per layer — so the per-request
// holders live natively as "slots" (TSGgml_GlmSlotAlloc / SetActiveSlot /
// SlotFree): a slot is a full set of those caches plus its own n_past, sharing
// the model weights and the rope tables. Binding a request is a native
// active-slot switch, so no KV bytes move, and each slot's graphs are cached
// and captured independently (the graph cache keys on the slot id, so
// concurrent requests replay their own captured CUDA graphs rather than
// rebuilding, or replaying another request's baked cache addresses).
//
// ForwardBatch (the token-batched paged path) is deliberately unavailable: MLA
// keeps one compressed row per token and the DSA indexer scores against that
// same contiguous history, neither of which has a paged-KV layout. Native
// token-batched decode shares one graph across independent sequence slots;
// prefill and unsupported decode shapes use per-sequence forwards.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        // requestId -> native slot id. Guarded by _nativeSync, the same lock
        // that serializes ForwardNative / ResetNative.
        private Dictionary<string, int> _slotByRequest;
        // Set when a mid-step chunked batched call failed once; large
        // batches then decline outright (see TryForwardBatchedFusedDecode).
        private bool _batchedChunkingLatched;
        // Native tensor-parallel degree (1 = layer split or one device).
        private int _nativeTp = 1;
        public FusedBatchExecutionEvidence LastAcceptedFusedBatch { get; private set; }
        public GgmlGlmBatchExecution LastNativeBatchedExecution { get; private set; }

        /// <summary>The native executor declines every token-batched decode under tensor parallelism
        /// (TSGgml_GlmForwardBatchedDecode); the executor's once-per-run warning quotes this.</summary>
        public string BatchedFusedDecodeDeclineReason => _nativeTp > 1
            ? $"GLM's token-batched decode runs on a layer split only; under --tp {_nativeTp} each sequence decodes through its own captured graph"
            : null;
        // Native slot serving the single-stream (N==1) path. Slot 0 at load;
        // replaced when AdoptPrimaryCacheToFused hands slot 0 to a request.
        private int _primarySlot;
        // Request whose slot is currently active, or null when the primary is.
        private string _activeSlotKey;

        /// <summary>The batched paged forward has no glm-dsa implementation
        /// (MLA + DSA have no paged-KV layout); the engine's planner routes
        /// around it via the per-sequence fused path.</summary>
        public bool BatchedForwardAvailable => false;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
            => throw new NotSupportedException(
                "GLM serves concurrency through per-sequence slots, not ForwardBatch.");

        /// <summary>Concurrent requests are served by the native executor's
        /// sequence slots. Only the native executor has slots; the managed
        /// per-op path stays on the serial per-sequence route.</summary>
        public bool SupportsPerSequenceFusedForward => _exec != null;

        public bool HasFusedSequenceCache(string requestId)
        {
            lock (_nativeSync)
            {
                return requestId != null
                    && _slotByRequest != null
                    && _slotByRequest.ContainsKey(requestId);
            }
        }

        /// <summary>Make <paramref name="requestId"/>'s slot the native active
        /// slot, allocating an empty one the first time the request is seen.
        /// Returns true when freshly allocated (the sequence starts at
        /// position 0).</summary>
        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            lock (_nativeSync)
            {
                if (_exec == null)
                    throw new InvalidOperationException("Per-request slots require the native GLM executor.");
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);

                bool fresh = false;
                if (!_slotByRequest.TryGetValue(requestId, out int slot))
                {
                    // Memory full: the idle primary first (its contents are nobody's during a fused step),
                    // then a retained conversation's slot (a cached next turn is lost).
                    // Several requests may be admitted before any binds. A
                    // zero-extra-slot request has the unique primary promise;
                    // a full-credit neighbour must not steal it when execution
                    // order changes or the previous claimant is cancelled.
                    bool canTakePrimary = _unifiedPrimarySlotReuse?.Invoke(requestId) ?? true;
                    slot = MemoryPolicy != null && canTakePrimary
                        ? SlotRetention.TakeIdlePrimary(_slotByRequest, ref _primarySlot, _activeSlotKey, new ExecutorGlmSlotStore(_exec)) : -1;
                    if (slot < 0) slot = _exec.SlotAlloc();
                    if (slot < 0 && canTakePrimary)
                        slot = SlotRetention.TakeIdlePrimary(_slotByRequest, ref _primarySlot, _activeSlotKey, new ExecutorGlmSlotStore(_exec));
                    if (slot < 0 && _retainedSlots is { Count: > 0 }) slot = AllocSlotReclaimingRetained();
                    if (slot < 0)
                        throw new SequenceSlotUnavailableException(
                            "GLM has no device memory for another sequence slot.");
                    _slotByRequest[requestId] = slot;
                    fresh = true;
                }

                if (!_exec.SetActiveSlot(slot))
                    throw new InvalidOperationException($"GLM slot {slot} missing for request {requestId}.");
                _activeSlotKey = requestId;
                _cacheSeqLen = _exec.NPast;
                return fresh;
            }
        }

        /// <summary>Hand the live single-stream slot (with its resident KV
        /// state) to <paramref name="requestId"/> without copying, and lazily
        /// allocate a fresh primary for later N==1 use.</summary>
        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            lock (_nativeSync)
            {
                if (_exec == null) return;
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                if (_activeSlotKey != null) return;   // a request slot is already checked out
                if (_slotByRequest.ContainsKey(requestId)) return;

                _slotByRequest[requestId] = _primarySlot;
                _activeSlotKey = requestId;           // primary slot is (and stays) active
                _primarySlot = -1;                    // re-allocated on RestorePrimaryCache
            }
        }

        /// <summary>A new slot, reclaiming idle retained ones when memory has no room (see AllocReclaiming);
        /// the prefix cache learns of each through the sink before its next tree read.</summary>
        private int AllocSlotReclaimingRetained()
        {
            var reclaimed = new List<string>(0);
            int slot = SlotRetention.AllocReclaiming(_retainedSlots, new ExecutorGlmSlotStore(_exec), reclaimed);
            foreach (string key in reclaimed)
                _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
            return slot;
        }

        /// <summary>Reinstate the single-stream slot as the native active slot
        /// before an N==1 step that follows a concurrent episode.</summary>
        public void RestorePrimaryCache()
        {
            lock (_nativeSync)
            {
                if (_exec == null || _activeSlotKey == null) return;
                if (_primarySlot < 0)
                {
                    _primarySlot = AllocSlotReclaimingRetained();
                    if (_primarySlot < 0)
                        throw new SequenceSlotUnavailableException(
                            "GLM primary-slot allocation failed (device memory exhausted?).");
                }
                if (!_exec.SetActiveSlot(_primarySlot))
                    throw new InvalidOperationException($"GLM primary slot {_primarySlot} missing.");
                _activeSlotKey = null;
                _cacheSeqLen = _exec.NPast;
            }
        }

        /// <summary>TRUE token-batched decode: one token for each of N concurrent
        /// requests in a single fused graph, so the dense weights (and each
        /// step's routed experts) are read once per step instead of once per
        /// sequence. Declines (returns false) whenever a request has no slot yet
        /// or a position disagrees with its slot — the engine then falls back to
        /// the per-sequence round-robin loop.</summary>
        public bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
        {
            lock (_nativeSync)
            {
                LastAcceptedFusedBatch = null;
                LastNativeBatchedExecution = null;
                if (_exec == null || _slotByRequest == null) return false;
                int n = requestIds?.Count ?? 0;
                if (n < 2 || tokens == null || positions == null || outLogits == null
                    || tokens.Length != n || positions.Length != n || outLogits.Length != n) return false;

                var slots = new int[n];
                for (int i = 0; i < n; i++)
                {
                    if (requestIds[i] == null || !_slotByRequest.TryGetValue(requestIds[i], out slots[i]))
                        return false;
                    if (tokens[i] < 0 || tokens[i] >= Config.VocabSize || positions[i] < 0 || positions[i] >= _maxContextLength)
                        return false;
                }

                int vocab = Config.VocabSize;
                if ((long) n * vocab > int.MaxValue) return false;
                var flat = new float[n * vocab];
                int context = 0;
                for (int i = 0; i < n; ++i) context = Math.Max(context, checked(positions[i] + 1));
                PrepareUnifiedGraph(context, Math.Min(n, 16), true);
                BeforeGraphCall();
                try
                {
                    if (!ForwardBatchedDecodeChunks(slots, tokens, positions, flat, vocab)) return false;
                }
                finally { AfterGraphCall(); }

                for (int i = 0; i < n; i++)
                {
                    var row = new float[vocab];
                    Array.Copy(flat, (long) i * vocab, row, 0, vocab);
                    outLogits[i] = row;
                }
                // Wider batches execute several graphs. Their last chunk must
                // never be presented as one graph for all original requests.
                if (n <= 16 && _exec is NativeGlmExecutor native && native.LastBatchedExecution is { } actual)
                {
                    if (actual.Rows != n) throw new InvalidOperationException("Native GLM batch evidence has the wrong row count.");
                    for (int i = 0; i < n; ++i)
                        if (actual.Slots[i] != slots[i] || actual.PositionsBefore[i] != positions[i]
                            || actual.PositionsAfter[i] != checked(positions[i] + 1))
                            throw new InvalidOperationException("Native GLM batch evidence disagrees with its request slots or published positions.");
                    LastNativeBatchedExecution = actual;
                    // GLM has no explicit HostMoeSegment seams; its one logical
                    // scheduler graph can still split into many backend spans.
                    LastAcceptedFusedBatch = new(actual.ExecutionId, n, actual.Nodes, 0, 0);
                }
                return true;
            }
        }

        /// <summary>One batched decode step for <paramref name="slots"/>, in windows of at most 16 sequences.</summary>
        private bool ForwardBatchedDecodeChunks(int[] slots, int[] tokens, int[] positions, float[] flat, int vocab)
        {
            // The native batched graph caps at 16 sequences (its per-slot
            // attention forks are O(n) graph nodes). Above that, run the
            // step as near-equal windows of <=16 - two weight sweeps for a
            // double-cap batch still beat that many serial solo sweeps.
            // Windows are sized so none is ever 1 (native needs n>=2). A
            // failure AFTER the first window would leave earlier slots
            // advanced while the engine retries the whole step, and the
            // native position gates would then error those sequences
            // visibly - so on any mid-step failure, latch chunking off and
            // decline.
            const int MaxPerCall = 16;
            int n = slots.Length;
            if (n <= MaxPerCall)
                return _exec.ForwardBatchedDecode(slots, tokens, positions, flat);

            if (_batchedChunkingLatched) return false;
            int chunks = (n + MaxPerCall - 1) / MaxPerCall;
            int baseSize = n / chunks, rem = n % chunks;
            int off = 0;
            for (int c = 0; c < chunks; c++)
            {
                int len = baseSize + (c < rem ? 1 : 0);
                var cs = new int[len]; var ct = new int[len]; var cp = new int[len];
                Array.Copy(slots, off, cs, 0, len);
                Array.Copy(tokens, off, ct, 0, len);
                Array.Copy(positions, off, cp, 0, len);
                var cf = new float[len * vocab];
                if (!_exec.ForwardBatchedDecode(cs, ct, cp, cf))
                {
                    if (c > 0)
                    {
                        _batchedChunkingLatched = true;
                        Console.Error.WriteLine(
                            "[glm batched-decode] chunk " + (c + 1) + "/" + chunks +
                            " failed mid-step; chunked batching disabled");
                    }
                    return false;
                }
                Array.Copy(cf, 0, flat, (long) off * vocab, (long) len * vocab);
                off += len;
            }
            return true;
        }

        /// <summary>Free a finished/aborted request's slot (its caches and any
        /// graphs captured against them).</summary>
        public void OnSequenceReleased(string requestId)
        {
            lock (_nativeSync)
            {
                if (_exec == null
                    || _slotByRequest == null
                    || string.IsNullOrEmpty(requestId)
                    || !_slotByRequest.TryGetValue(requestId, out int slot))
                {
                    return;
                }

                if (string.Equals(_activeSlotKey, requestId, StringComparison.Ordinal))
                {
                    if (_primarySlot < 0)
                    {
                        // A request holds the primary: this slot becomes it (see ReleaseActiveSlotAsPrimary).
                        if (SlotRetention.ReleaseActiveSlotAsPrimary(_slotByRequest, requestId, ref _primarySlot, ref _activeSlotKey,
                                new ExecutorGlmSlotStore(_exec)))
                            _cacheSeqLen = 0;
                        return;
                    }
                    // Reinstate the primary before freeing the active slot.
                    RestorePrimaryCache();
                }

                _slotByRequest.Remove(requestId);
                _exec.SlotFree(slot);
            }
        }
    }
}
