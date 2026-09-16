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
// same contiguous history, neither of which has a paged-KV layout. Concurrent
// requests are served by interleaving whole-graph per-sequence forwards, which
// is what the engine's per-sequence fused contract below expresses.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

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
        public bool SupportsPerSequenceFusedForward => _native != IntPtr.Zero;

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
                if (_native == IntPtr.Zero)
                    throw new InvalidOperationException("Per-request slots require the native GLM executor.");
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);

                bool fresh = false;
                if (!_slotByRequest.TryGetValue(requestId, out int slot))
                {
                    slot = GgmlGlmNative.SlotAlloc(_native);
                    if (slot < 0)
                        throw new InvalidOperationException(
                            "GLM sequence-slot allocation failed (device memory exhausted?).");
                    _slotByRequest[requestId] = slot;
                    fresh = true;
                }

                if (!GgmlGlmNative.SetActiveSlot(_native, slot))
                    throw new InvalidOperationException($"GLM slot {slot} missing for request {requestId}.");
                _activeSlotKey = requestId;
                _cacheSeqLen = GgmlGlmNative.NPast(_native);
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
                if (_native == IntPtr.Zero) return;
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                if (_activeSlotKey != null) return;   // a request slot is already checked out
                if (_slotByRequest.ContainsKey(requestId)) return;

                _slotByRequest[requestId] = _primarySlot;
                _activeSlotKey = requestId;           // primary slot is (and stays) active
                _primarySlot = -1;                    // re-allocated on RestorePrimaryCache
            }
        }

        /// <summary>Reinstate the single-stream slot as the native active slot
        /// before an N==1 step that follows a concurrent episode.</summary>
        public void RestorePrimaryCache()
        {
            lock (_nativeSync)
            {
                if (_native == IntPtr.Zero || _activeSlotKey == null) return;
                if (_primarySlot < 0)
                {
                    _primarySlot = GgmlGlmNative.SlotAlloc(_native);
                    if (_primarySlot < 0)
                        throw new InvalidOperationException(
                            "GLM primary-slot allocation failed (device memory exhausted?).");
                }
                if (!GgmlGlmNative.SetActiveSlot(_native, _primarySlot))
                    throw new InvalidOperationException($"GLM primary slot {_primarySlot} missing.");
                _activeSlotKey = null;
                _cacheSeqLen = GgmlGlmNative.NPast(_native);
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
                if (_native == IntPtr.Zero || _slotByRequest == null) return false;
                int n = requestIds.Count;
                if (n < 2) return false;

                var slots = new int[n];
                for (int i = 0; i < n; i++)
                {
                    if (requestIds[i] == null || !_slotByRequest.TryGetValue(requestIds[i], out slots[i]))
                        return false;
                }

                int vocab = Config.VocabSize;
                if ((long) n * vocab > int.MaxValue) return false;
                var flat = new float[n * vocab];

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
                if (n <= MaxPerCall)
                {
                    if (!GgmlGlmNative.ForwardBatchedDecode(_native, slots, tokens, positions, flat))
                        return false;
                }
                else
                {
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
                        if (!GgmlGlmNative.ForwardBatchedDecode(_native, cs, ct, cp, cf))
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
                }

                for (int i = 0; i < n; i++)
                {
                    var row = new float[vocab];
                    Array.Copy(flat, (long) i * vocab, row, 0, vocab);
                    outLogits[i] = row;
                }
                return true;
            }
        }

        /// <summary>Free a finished/aborted request's slot (its caches and any
        /// graphs captured against them).</summary>
        public void OnSequenceReleased(string requestId)
        {
            lock (_nativeSync)
            {
                if (_native == IntPtr.Zero
                    || _slotByRequest == null
                    || string.IsNullOrEmpty(requestId)
                    || !_slotByRequest.TryGetValue(requestId, out int slot))
                {
                    return;
                }

                if (string.Equals(_activeSlotKey, requestId, StringComparison.Ordinal))
                {
                    // The released slot is active; reinstate the primary first
                    // (the native side refuses to free the active slot).
                    RestorePrimaryCache();
                }

                _slotByRequest.Remove(requestId);
                GgmlGlmNative.SlotFree(_native, slot);
            }
        }
    }
}
