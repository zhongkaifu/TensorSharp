// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Per-request KV-cache holders for the per-sequence fused-decode path
// (the Gemma 4 / Qwen 3.5 pattern, ported to GPT-OSS).
//
// Problem this solves: with N>=2 concurrent requests ExecutionPlanner had no
// PerSequenceFused candidate for this model, so every step fell to the op-by-op
// batched paged forward (ForwardBatch). That path reads Q/K/V back to the host
// with GetElementsAsFloat and keeps the paged pool in managed float[][], so each
// of the 24 layers pays a device->host round trip plus a re-upload of the whole
// K/V history, every token. Measured on gpt-oss-20b / 1x RTX PRO 6000: 3.8 ms
// per token at N=1 (SingleSequenceFused) against 89 ms per token at N=2, i.e.
// aggregate throughput COLLAPSED from 230 tok/s to 14 tok/s the moment a second
// request arrived, and the plan logged
// "rejected: SingleSequenceFused: sequence K/V already committed to paged
// storage" - a one-way latch, so the sequence never recovered.
//
// Fix: give each in-flight request its OWN set of KV-cache tensors and switch
// the model between them with a pointer swap. The engine then runs each
// scheduled sequence through the proven single-graph fused Forward
// (TSGgml_GptOssModelDecode), whose native side ALREADY keeps a per-request
// pool of captured graphs keyed by (model instance, first KV cache pointer) -
// see GptOssDecodeCachePool in ggml_ops_gptoss_decode.cpp, which was written for
// exactly this and had no managed caller. No byte-level extract/inject, no
// paged-storage commitment, no latch.
//
// The single-request (N==1) path is untouched: it keeps using the model's
// primary cache and the engine's live-cache continuation / prefix-cache reuse.
// RestorePrimaryCache() reinstates the primary cache before any N==1 step that
// follows a multi-sequence episode.
using System;
using System.Collections.Generic;

using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class GptOssModel
    {
        private sealed class GptOssKvCacheHolder
        {
            public Tensor[] K;
            public Tensor[] V;
            public int Capacity;
            public int SeqLen;
            public bool HostDirty;
            // Per-request logits buffer for the token-batched decode: reused
            // every step instead of allocating vocab-sized arrays per sequence
            // per token (~800KB x N per step of pure GC churn). Owned by the
            // holder so a reference stored in SequenceState.LastLogits stays
            // valid for exactly the request's lifetime - a shared pool slot
            // could be handed to another request while a preempted sequence
            // still holds the reference.
            public float[] Logits;
        }

        // Per-request fused-decode cache holders, keyed by RequestId.
        private Dictionary<string, GptOssKvCacheHolder> _fusedHolders;
        // RequestId whose holder is currently checked out into the active
        // _kvCacheK/_kvCacheV fields, or null when the primary cache is active.
        private string _activeFusedKey;
        // The primary cache, parked while a fused holder is checked out so
        // RestorePrimaryCache() can reinstate it for the N==1 path.
        private GptOssKvCacheHolder _primaryHolder;
        // Capacity InitKVCache started the primary cache at; fresh holders get
        // the same, and grow on demand through EnsureCacheCapacity.
        private int _initialKvCacheLength = 512;
        // Freelist of released holders. Reusing a holder's tensors hands the next
        // request the SAME host cache pointers, which is what keeps the native
        // decode-graph pools (keyed on those pointers) and their CUDA-graph
        // captures warm across request churn - and means releasing a request does
        // NOT have to drop every captured graph.
        private List<GptOssKvCacheHolder> _holderPool;
        // Sized for the highest concurrency the server is benchmarked at: a
        // parked holder costs only its grown host arrays, while overflow
        // disposes tensors and (via the invalidate hooks) retires native
        // decode state - far more expensive than parking under churn.
        private string _lastBatchedDeclineLogged;
        // Staging for the batched kernel's packed [vocab, N] logits output;
        // grown on demand, reused every step.
        private float[] _batchedLogitsStaging;

        /// <summary>
        /// The per-sequence fused forward is what the engine dispatches for
        /// concurrent (N&gt;=2) requests: each request decodes through its own KV
        /// holder, swapped in with a pointer flip, instead of the op-by-op
        /// batched paged path.
        ///
        /// Gated on the GGML backend (the native per-request decode-graph pool
        /// is what makes the swap cheap) and off under tensor parallelism, where
        /// the KV tensors live in <c>_tpKvCacheK</c>/<c>_tpKvCacheV</c> and these
        /// arrays are null.
        /// </summary>
        public bool SupportsPerSequenceFusedForward =>
            IsGgmlBackend && !IsTensorParallel && _kvCacheK != null;

        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _fusedHolders != null && _fusedHolders.ContainsKey(requestId);

        private GptOssKvCacheHolder SnapshotActiveCache() => new GptOssKvCacheHolder
        {
            K = _kvCacheK,
            V = _kvCacheV,
            Capacity = _kvCacheCapacity,
            SeqLen = _cacheSeqLen,
            HostDirty = _kvCacheHostDirty,
        };

        private void LoadCacheHolder(GptOssKvCacheHolder h)
        {
            _kvCacheK = h.K;
            _kvCacheV = h.V;
            _kvCacheCapacity = h.Capacity;
            _cacheSeqLen = h.SeqLen;
            _kvCacheHostDirty = h.HostDirty;
            // The fused decode/prefill kernels re-read the raw K/V pointers out
            // of _kvCacheK on every call (see TryFusedModelDecode), so there is
            // no cached pointer array to refresh here. The native graph pool
            // keys on those pointers, so the swapped-in holder finds its OWN
            // captured graph instead of rebuilding.
        }

        /// <summary>Allocate one request's worth of per-layer K/V cache tensors.
        /// Zero-filled: the fused decode reads a 256-padded attention window and
        /// rows past the written length are masked but must still be finite, or
        /// the softmax is poisoned.</summary>
        private void AllocateKvCacheArrays(int capacity, out Tensor[] k, out Tensor[] v)
        {
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();
            k = new Tensor[Config.NumLayers];
            v = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                k[l] = new Tensor(_allocator, kvDtype, numKVHeads, capacity, headDim);
                v[l] = new Tensor(_allocator, kvDtype, numKVHeads, capacity, headDim);
                InitializeCacheTensor(k[l]);
                InitializeCacheTensor(v[l]);
            }
        }

        private GptOssKvCacheHolder CreateFreshHolder()
        {
            if (_holderPool != null && _holderPool.Count > 0)
            {
                // Reused allocation: stale device rows beyond the new request's
                // writes are never read (the attention mask bounds every query to
                // rows this request wrote), so no zeroing pass is needed.
                var pooled = _holderPool[_holderPool.Count - 1];
                _holderPool.RemoveAt(_holderPool.Count - 1);
                pooled.SeqLen = 0;
                pooled.HostDirty = false;
                return pooled;
            }
            int capacity = Math.Max(_initialKvCacheLength, 1);
            AllocateKvCacheArrays(capacity, out var k, out var v);
            return new GptOssKvCacheHolder
            {
                K = k,
                V = v,
                Capacity = capacity,
                SeqLen = 0,
                HostDirty = false,
            };
        }

        private void DisposeHolder(GptOssKvCacheHolder h)
        {
            if (h?.K == null) return;
            for (int l = 0; l < h.K.Length; l++)
            {
                if (h.K[l] != null) { InvalidateTensorDeviceCache(h.K[l]); h.K[l].Dispose(); }
                if (h.V != null && h.V[l] != null) { InvalidateTensorDeviceCache(h.V[l]); h.V[l].Dispose(); }
            }
            h.K = null;
            h.V = null;
        }

        /// <summary>Make <paramref name="requestId"/>'s KV cache the model's
        /// active cache, creating an empty one the first time the request is
        /// seen. Returns true when the cache was freshly created, so the caller
        /// knows to inject any prefix-cache-reused prefix before the first
        /// Forward.</summary>
        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            _fusedHolders ??= new Dictionary<string, GptOssKvCacheHolder>(StringComparer.Ordinal);

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
                return false; // already active

            // Save whatever cache is checked out so its (possibly grown) tensors
            // and advanced SeqLen aren't lost when the active fields are repointed.
            if (_activeFusedKey == null)
                _primaryHolder = SnapshotActiveCache();
            else
                _fusedHolders[_activeFusedKey] = SnapshotActiveCache();

            bool fresh;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
            {
                holder = CreateFreshHolder();
                _fusedHolders[requestId] = holder;
                fresh = true;
            }
            else
            {
                fresh = false;
            }

            LoadCacheHolder(holder);
            _activeFusedKey = requestId;
            return fresh;
        }

        /// <summary>Transition the single in-flight N==1 owner - whose live K/V
        /// is in the primary cache - into the fused path without copying any KV
        /// bytes: hand the live primary arrays to the owner's holder and give the
        /// primary a fresh empty allocation.</summary>
        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            if (!SupportsPerSequenceFusedForward) return;
            _fusedHolders ??= new Dictionary<string, GptOssKvCacheHolder>(StringComparer.Ordinal);

            // Only meaningful while the primary cache is the active one. If a
            // fused holder is already checked out there is nothing to adopt.
            if (_activeFusedKey != null)
                return;
            if (_fusedHolders.ContainsKey(requestId))
                return;

            _fusedHolders[requestId] = SnapshotActiveCache();
            _activeFusedKey = requestId;

            // Fresh, zero-filled primary so a later N==1 step for a never-fused
            // request cannot reset the adopted holder's tensors.
            int capacity = Math.Max(_initialKvCacheLength, 1);
            AllocateKvCacheArrays(capacity, out var k, out var v);
            _primaryHolder = new GptOssKvCacheHolder
            {
                K = k,
                V = v,
                Capacity = capacity,
                SeqLen = 0,
                HostDirty = false,
            };
        }

        /// <summary>Reinstate the primary cache as the model's active cache
        /// before an N==1 step that follows a fused episode, so the
        /// single-sequence path (which resets/injects the active cache in place)
        /// never clobbers a still-running concurrent request's holder.</summary>
        public void RestorePrimaryCache()
        {
            if (_activeFusedKey == null)
                return;
            _fusedHolders[_activeFusedKey] = SnapshotActiveCache();
            _activeFusedKey = null;
            if (_primaryHolder != null)
            {
                LoadCacheHolder(_primaryHolder);
                _primaryHolder = null;
            }
        }

        /// <summary>Release a finished/aborted request's per-request cache.</summary>
        public void OnSequenceReleased(string requestId)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
                return;

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
            {
                // The released sequence's cache is the one checked out. Swap the
                // primary back in so the active fields don't dangle.
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }

            _fusedHolders.Remove(requestId);
            // Bounded by the same knob as Qwen's pool (TS_KV_HOLDER_POOL_MAX): a parked
            // holder costs its whole K/V allocation for as long as it waits.
            int poolMax = Runtime.Scheduling.ExecutionOptions.FromEnvironment().KvHolderPoolMax;
            _holderPool ??= new List<GptOssKvCacheHolder>(Math.Max(1, poolMax));
            if (_holderPool.Count < poolMax)
            {
                // Park the allocation for the next request. The tensors stay
                // alive, so the native decode-graph pools (keyed on these host
                // pointers) and their device KV windows remain valid - no reset.
                _holderPool.Add(holder);
            }
            else
            {
                DisposeHolder(holder);
                // The native decode-graph pools key entries on the holder's
                // K-cache pointer. Those bytes are now free and the allocator may
                // hand the same address to the next holder, which would then
                // replay a graph bound to the freed windows. Drop the captured
                // graphs; they rebuild on the next decode.
                ResetFusedModelDecodeCache();
            }
        }

        /// <summary>Free every per-request holder (model teardown).</summary>
        private void DisposeFusedSequenceCaches()
        {
            if (_fusedHolders != null)
            {
                foreach (var kv in _fusedHolders)
                    if (!string.Equals(_activeFusedKey, kv.Key, StringComparison.Ordinal))
                        DisposeHolder(kv.Value);
                _fusedHolders.Clear();
                _fusedHolders = null;
            }
            // The parked primary is not referenced by the active fields, so it
            // has to be freed here; the checked-out holder is freed by the normal
            // KV teardown that follows.
            if (_primaryHolder != null)
            {
                DisposeHolder(_primaryHolder);
                _primaryHolder = null;
            }
            if (_holderPool != null)
            {
                foreach (var h in _holderPool)
                    DisposeHolder(h);
                _holderPool = null;
            }
            _activeFusedKey = null;
        }

        /// <summary>
        /// TRUE token-batched decode: one token for each of N concurrent
        /// sequences in ONE fused graph via
        /// <see cref="GgmlBasicOps.TryGptOssModelDecodeBatched"/>, every weight
        /// loaded once. This is what lifts aggregate throughput above the
        /// round-robin ceiling (N serial fused forwards = N weight sweeps per
        /// step = aggregate ~= single-stream): decode is memory-bandwidth bound,
        /// so one sweep serving N tokens scales aggregate tokens/s with N.
        ///
        /// Returns false (the caller falls back to the round-robin per-sequence
        /// path) when any v1 precondition fails: fused model decode available,
        /// no MoE CPU offload, F32/F16 KV, folded lm_head present, all holders
        /// exist, and no holder needs growth (positions[s]+1 &lt;= capacity).
        /// </summary>
        public unsafe bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
            => ForwardBatchedFusedDecodeCore(requestIds, tokens, positions, outLogits, null);

        /// <summary>Greedy fast path: like <see cref="TryForwardBatchedFusedDecode"/>
        /// but returns only each sequence's argmax token (sampled on-device on
        /// CUDA), skipping the [vocab, N] logits download entirely. Only valid
        /// when every sequence's sampler is a plain argmax.</summary>
        public unsafe bool TryForwardBatchedFusedDecodeSampled(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, int[] outNextTokens)
            => ForwardBatchedFusedDecodeCore(requestIds, tokens, positions, null, outNextTokens);

        /// <summary>A sequence can join the token-batched decode when its holder
        /// exists and needs no growth for this position — growth is the
        /// round-robin path's job, and one growing sequence should not push the
        /// whole batch off the fused path.</summary>
        public bool CanBatchDecode(string requestId, int position)
        {
            if (_fusedHolders == null || !_fusedHolders.TryGetValue(requestId, out var h) || h.K == null)
                return false;
            return position + 1 <= h.Capacity;
        }

        private unsafe bool ForwardBatchedFusedDecodeCore(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions,
            float[][] outLogits, int[] outNextTokens)
        {
            if (!IsGgmlBackend || IsTensorParallel || _fusedHolders == null)
                return false;
            if (!FusedModelDecodeEnabled)
                return false;   // TS_GPTOSS_MODEL_DECODE=0 debugs the whole kernel family
            int n = requestIds.Count;
            if (n < 2 || tokens.Length != n || positions.Length != n)
                return false;
            if (!TryBuildModelDecodeArgs())
                return false;
            // Defensive: SupportsBlockQuantizedKvCache => false already swaps a
            // requested q8_0/q4_0 cache for f16 when InitKVCache runs, so this
            // should not fire. It stays because _kvCacheDtype is seeded from the
            // process-global KvCacheDtypeConfig and the kernel cannot read a
            // block-quantized cache at all.
            int kvType = _kvCacheDtype.GgmlType();
            if (kvType != 0 /* F32 */ && kvType != 1 /* F16 */)
                return false;
            var args = _modelDecodeArgs;
            for (int l = 0; l < args.Length; l++)
                if (args[l].CpuMoe != 0)
                    return false;   // v1: host-MoE segmentation not batched

            // Folded lm_head (the batched kernel requires the fold).
            if (!_quantWeights.TryGetValue("output.weight", out var lmHead) &&
                !_quantWeights.TryGetValue("token_embd.weight", out lmHead))
                return false;
            if (!_weights.TryGetValue("output_norm.weight", out var finalNorm))
                return false;

            // Check in any holder still bound to the active fields FIRST: the
            // dict entry for a checked-out request is a stale snapshot until
            // RestorePrimaryCache replaces it, and reading capacities or
            // writing SeqLen through the stale object desyncs positions.
            RestorePrimaryCache();

            int numLayers = Config.NumLayers;
            var holders = new GptOssKvCacheHolder[n];
            for (int s = 0; s < n; s++)
            {
                if (!_fusedHolders.TryGetValue(requestIds[s], out holders[s]) || holders[s].K == null)
                    return false;
                // Growth is the round-robin path's job (Forward grows the active
                // cache); decline this step and the next one batches again.
                if (positions[s] + 1 > holders[s].Capacity)
                    return false;
            }

            // Canonical order = ascending layer-0 K storage pointer. The native
            // persist pool keys on the ordered SET of cache pointers, and with
            // pooled holder reuse the pointer set outlives any particular request
            // ids - so a stable set of in-flight slots replays its captured graph
            // no matter which requests currently occupy them.
            var order = new int[n];
            for (int s = 0; s < n; s++) order[s] = s;
            var keys = new ulong[n];
            for (int s = 0; s < n; s++)
                keys[s] = (ulong)TensorComputePrimitives.GetStoragePointer(holders[s].K[0]).ToInt64();
            Array.Sort(keys, order);

            var posSorted = new int[n];
            var tokSorted = new int[n];
            var cacheSizes = new int[n];
            for (int s = 0; s < n; s++)
            {
                posSorted[s] = positions[order[s]];
                tokSorted[s] = tokens[order[s]];
                cacheSizes[s] = holders[order[s]].Capacity;
            }

            var kCaches = new IntPtr[numLayers * n];
            var vCaches = new IntPtr[numLayers * n];
            for (int l = 0; l < numLayers; l++)
                for (int s = 0; s < n; s++)
                {
                    var h = holders[order[s]];
                    kCaches[l * n + s] = TensorComputePrimitives.GetStoragePointer(h.K[l]);
                    vCaches[l * n + s] = TensorComputePrimitives.GetStoragePointer(h.V[l]);
                }

            // Embed the N decode tokens -> [n, hidden] host floats = the packed
            // [hidden, n] column-major layout the kernel takes. Dequantize the
            // embedding rows on the HOST: for a handful of decode tokens that is
            // microseconds, where the generic multi-token path routes through a
            // device get_rows plus a readback sync.
            int dim = Config.HiddenSize;
            Tensor hidden;
            if (_quantWeights.TryGetValue("token_embd.weight", out var embW) && embW.HasHostData)
            {
                hidden = new Tensor(_allocator, DType.Float32, n, dim);
                PopulateQuantizedRows(hidden, embW, tokSorted);
            }
            else
            {
                hidden = Embedding(tokSorted);
            }
            int vocab = Config.VocabSize;
            bool wantLogits = outLogits != null;
            long needed = (long)vocab * n;
            if (_batchedLogitsStaging == null || _batchedLogitsStaging.LongLength < needed)
                _batchedLogitsStaging = new float[needed];
            var logitsBuf = _batchedLogitsStaging;
            var sampledSorted = outNextTokens != null ? new int[n] : null;
            bool ok;
            try
            {
                float* hiddenPtr = GetFloatPtr(hidden);
                fixed (float* lp = logitsBuf)
                fixed (int* sp = sampledSorted)
                {
                    ok = GgmlBasicOps.TryGptOssModelDecodeBatched(
                        args, numLayers, n, (IntPtr)hiddenPtr,
                        kCaches, vCaches, cacheSizes, posSorted,
                        (IntPtr)lp, vocab,
                        lmHead.CacheKey, lmHead.GgmlType, lmHead.Ne0, lmHead.Ne1, lmHead.RawBytes,
                        (IntPtr)GetFloatPtr(finalNorm),
                        (IntPtr)sp, wantLogits);
                }
            }
            finally
            {
                hidden.Dispose();
            }
            if (!ok)
            {
                // One line per distinct native error - a silent per-step decline
                // is indistinguishable from the round-robin path being chosen.
                string err = GgmlBasicOps.LastNativeError();
                if (err != _lastBatchedDeclineLogged)
                {
                    _lastBatchedDeclineLogged = err;
                    Console.Error.WriteLine($"[gptoss batched-decode] native declined (n={n}): {err}");
                }
                return false;
            }

            // Distribute per-seq results (un-permute) and advance the holders.
            // The kernel wrote K/V device-side only, so every holder's host
            // mirror is now stale.
            for (int s = 0; s < n; s++)
            {
                var h = holders[order[s]];
                if (wantLogits)
                {
                    h.Logits ??= new float[vocab];
                    Array.Copy(logitsBuf, (long)s * vocab, h.Logits, 0, vocab);
                    outLogits[order[s]] = h.Logits;
                }
                if (outNextTokens != null)
                    outNextTokens[order[s]] = sampledSorted[s];
                h.SeqLen = posSorted[s] + 1;
                h.HostDirty = true;
            }
            return true;
        }
    }
}
