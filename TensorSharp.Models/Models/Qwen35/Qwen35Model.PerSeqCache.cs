// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Per-request KV + GDN-state holders for the per-sequence fused-decode path
// (the Qwen3.5/3.6 analogue of Gemma4Model.PerSeqCache).
//
// Problem this solves: with N>=2 concurrent requests the engine routed every
// decode step through the batched paged forward (ForwardBatch / the true
// token-batched fused decode g_q35bdc). Measured on ggml_cuda that path
// produced WRONG output (the per-slot GDN state + migration corrupt the
// resumed sequence) AND collapsed aggregate throughput to ~24 tok/s (from a
// single-stream ~80). The op-by-op batched fallback was even slower (~10) and
// also wrong; the per-seq KV-swap rotation was fast (~64 agg) but still 1/2
// correct because the model's single fused decode cache (g_q35dc) and the
// single linear GDN state (_convState / _deltaStateTensor / _fdConvScratch)
// were shared across the two sequences — the captured decode graph baked one
// request's device addresses and replayed them for the other.
//
// Fix (mirrors Gemma4): give each in-flight request its OWN set of attention
// KV tensors AND its own GDN recurrent state (host conv ring + device delta
// tensors + the fused-decode conv scratch), and switch the model between them
// with a cheap reference swap. Each sequence then decodes through the proven
// single-graph fused Forward (TryFullModelDecode); the native decode-graph
// pool (g_q35dc_pool) keys each request's captured graph on its first
// attention KV pointer, so concurrent requests each replay their own captured
// graph instead of busting/rebuilding (or, worse, replaying the other
// request's baked addresses). No cross-sequence KV/GDN snapshot, so no
// state-isolation corruption.
//
// The single-request (N==1) path is untouched: it keeps using the model's
// primary cache. RestorePrimaryCache() reinstates it before any N==1 step that
// follows a multi-sequence (fused) episode.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        private sealed class Qwen35KvCacheHolder
        {
            // Attention KV (one entry per layer; recurrent layers are null).
            public Tensor[] K;
            public Tensor[] V;
            public int KvCapacity;
            public int CacheSeqLen;
            public bool KvHostDirty;
            // GDN recurrent state: host conv ring + write idx + device delta state.
            public float[][] ConvState;
            public int[] ConvWriteIdx;
            public Tensor[] DeltaState;
            // Fused-decode conv scratch (ggml [time,channel], cacheable device buffer
            // keyed on this host ptr) + whether this holder's device GDN state is
            // currently seeded (resident) on the device.
            public IntPtr ConvScratch;
            public bool FdStateResident;
            public bool GdnHostDirty;
            // True when the slot-stable batched arena (rather than the normal
            // host/cacheable-buffer path) owns the newest KV + GDN bytes. It can
            // outlive the request id while this holder is retained and re-keyed.
            public bool ArenaStateResident;
            // Reusable full-vocab logits buffer for the arena batched decode:
            // owned by the holder so a SequenceState.LastLogits reference stays
            // valid however the batch composition churns.
            public float[] Logits;
            public bool Retired;
            public bool DisposalStarted;
            // The holder has been an active cache on a GPU backend, so its K/V and
            // recurrent state have device copies besides the host bytes (the prefix
            // cache's MeasureEndState charges both). Copies and imports start without.
            public bool DeviceMirrored;
        }

        // Per-request fused-decode holders, keyed by RequestId.
        private Dictionary<string, Qwen35KvCacheHolder> _fusedHolders;
        // Finished holders kept intact for exact-prefix continuations. Unlike a
        // paged Qwen snapshot, each holder owns both its attention K/V and the
        // matching GDN recurrent state, so it can be re-keyed without rebuilding
        // either half of the hybrid cache.
        private Dictionary<string, Qwen35KvCacheHolder> _retainedFusedHolders;
        // Freelist of released holders. Parking keeps the host allocations and
        // their stable pointer identities reusable; completed arena mappings are
        // retired explicitly, and Metal's aliased state mirrors are evicted at
        // reassignment. Disposing a holder per completed request would fire the
        // broader InvalidateHostBuffer teardown on every completion. Reused
        // holders are re-zeroed by ResetHolderForReuse.
        private List<Qwen35KvCacheHolder> _holderPool;

        private void DiscardArenaSlotForHolder(Qwen35KvCacheHolder h)
        {
            if (h == null)
                return;

            if (!IsGgmlBackend || h.K == null || _isRecurrent == null)
            {
                h.ArenaStateResident = false;
                return;
            }

            for (int l = 0; l < h.K.Length && l < _isRecurrent.Length; l++)
            {
                if (_isRecurrent[l] || h.K[l] == null)
                    continue;

                // The first attention-K pointer is the native registry key for
                // the complete holder slot (attention KV plus recurrent state).
                GgmlBasicOps.Qwen35ArenaDiscardHostPointer(
                    TensorComputePrimitives.GetStoragePointer(h.K[l]));
                break;
            }
            // Clear the managed ownership bit even for a malformed/no-attention
            // holder. Once this method returns, recycle/disposal must never treat
            // an old native arena slot as authoritative for this object.
            h.ArenaStateResident = false;
        }

        private void InvalidateHolderDeviceCopiesForReuse(Qwen35KvCacheHolder h)
        {
            // The stale alias is specific to Metal's in-place GDN layout: its
            // state view and allocation base are separate native cache keys.
            // Keep CUDA's established holder/graph-retention path unchanged;
            // evicting one CUDA mirror also resets every persistent graph.
            if (_backend != BackendType.GgmlMetal || h == null)
                return;

            // A pooled holder keeps stable host pointers, so GGML may still have
            // resident copies and persistent graphs keyed by them. The next
            // request rewrites those host buffers during reset/prefill; evict the
            // old mirrors first so it cannot inherit the completed request's KV
            // or recurrent state. This is paid once per holder reassignment, not
            // per generated token.
            if (h.K != null)
                foreach (Tensor t in h.K)
                    InvalidateTensorDeviceCache(t);
            if (h.V != null)
                foreach (Tensor t in h.V)
                    InvalidateTensorDeviceCache(t);
            if (h.DeltaState != null)
                foreach (Tensor t in h.DeltaState)
                    if (t != null)
                        InvalidateGdnDeltaStateDeviceCaches(t);

            if (h.ConvScratch == IntPtr.Zero || _isRecurrent == null)
                return;
            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            long layerBytes = (long)Math.Max(0, convDim) * qkvDim * sizeof(float);
            int recurrentSlot = 0;
            for (int l = 0; l < _isRecurrent.Length; l++)
            {
                if (!_isRecurrent[l]) continue;
                if (layerBytes > 0)
                {
                    GgmlBasicOps.InvalidateHostBuffer(new IntPtr(
                        h.ConvScratch.ToInt64() + recurrentSlot * layerBytes));
                }
                recurrentSlot++;
            }
        }

        private void ResetHolderForReuse(Qwen35KvCacheHolder h)
        {
            // A completed arena decode can leave newer KV/GDN state resident in
            // its native slot. Retire that mapping before zeroing the host-side
            // holder for another request; a later prefill touch must never flush
            // the previous request over the newly initialized state.
            DiscardArenaSlotForHolder(h);
            InvalidateHolderDeviceCopiesForReuse(h);
            h.CacheSeqLen = 0;
            h.KvHostDirty = false;
            h.GdnHostDirty = false;
            h.FdStateResident = false;
            h.ArenaStateResident = false;
            if (h.ConvState != null)
                for (int l = 0; l < h.ConvState.Length; l++)
                {
                    if (h.ConvState[l] != null) Array.Clear(h.ConvState[l], 0, h.ConvState[l].Length);
                    if (h.ConvWriteIdx != null) h.ConvWriteIdx[l] = 0;
                }
            if (h.DeltaState != null)
                foreach (var t in h.DeltaState)
                    if (t != null) Ops.Fill(t, 0);
            // KV rows beyond the written prefix are mask-bounded on every read
            // path, so stale bytes there are safe (same rule as the GPT-OSS
            // holder pool). The old arena slot was discarded before reset.
        }
        // RequestId whose holder is currently checked out into the active model
        // fields, or null when the primary cache is active.
        private string _activeFusedKey;
        // Snapshot of the primary cache, saved while a fused holder is checked out.
        private Qwen35KvCacheHolder _primaryHolder;
        // Checked-out counterpart of Qwen35KvCacheHolder.ArenaStateResident.
        private bool _arenaStateResident;

        /// <summary>The per-sequence fused forward is the path the engine
        /// dispatches for concurrent (N&gt;=2) requests: each request decodes
        /// through its own KV + GDN holder (swapped in with a cheap reference
        /// flip) instead of the broken/slow batched paged path.
        ///
        /// ggml_cuda: each request decodes through the fast whole-model fused
        /// single-graph decode (<see cref="TryFullModelDecode"/>), so the path
        /// also requires that decode to be available (full-decode on, not
        /// latched unsupported).
        ///
        /// ggml_metal: each holder gets its own persistent whole-model graph,
        /// keyed by the first attention KV pointer. Swapping holders refreshes
        /// the managed descriptors once and selects the matching native graph;
        /// unsupported shapes still fall back to the isolated per-holder
        /// operation path. This preserves both concurrency and per-request
        /// recurrent/attention state.</summary>
        /// <para>Never under tensor parallelism: the per-request holders
        /// snapshot and swap the single-GPU cache arrays (<c>_kvCacheK</c> and
        /// friends), which the TP path never populates - it builds its own
        /// per-rank caches instead. Taking this path with TP active dereferenced
        /// a null cache array the moment a second sequence arrived.</para>
        // Not gated on the speculative session any more: a request that speculates on
        // its BOUND holder used to flip this false after its first speculative step,
        // the planner re-routed the holder-resident sequence to the linear path and
        // it lost its position ("no LastLogits to sample from at position 0"). The
        // fused decode itself leaves the session (ExitSpecSession) when it is next
        // asked to run, so the capability is true whenever the backend supports it.
        public bool SupportsPerSequenceFusedForward =>
            !IsTensorParallel
            && ((_backend == BackendType.GgmlCuda && _fullDecodeEnabled && !_fdUnsupported)
                || _backend == BackendType.GgmlMetal);

        public bool SupportsRetainedFusedCache => true;

        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _fusedHolders != null && _fusedHolders.ContainsKey(requestId);

        private Qwen35KvCacheHolder SnapshotActiveCache() => new Qwen35KvCacheHolder
        {
            K = _kvCacheK,
            V = _kvCacheV,
            KvCapacity = _kvCacheCapacity,
            CacheSeqLen = _cacheSeqLen,
            KvHostDirty = _kvCacheHostDirty,
            ConvState = _convState,
            ConvWriteIdx = _convStateWriteIdx,
            DeltaState = _deltaStateTensor,
            ConvScratch = _fdConvScratch,
            FdStateResident = _fdStateResident,
            GdnHostDirty = _gdnStateHostDirty,
            ArenaStateResident = _arenaStateResident,
            DeviceMirrored = KeepsDeviceKvMirrors,
        };

        private void LoadCacheHolder(Qwen35KvCacheHolder h)
        {
            // The persistent verify cache is keyed by graph shape, not by the
            // per-request K/V holder whose device addresses it captures. Drop it
            // before switching holders so an all-logits/spec verify cannot replay
            // a graph against the previous request's cache.
            if (IsGgmlBackend && !ReferenceEquals(_kvCacheK, h.K))
                InvalidateVerifyCache();

            _kvCacheK = h.K;
            _kvCacheV = h.V;
            _kvCacheCapacity = h.KvCapacity;
            _cacheSeqLen = h.CacheSeqLen;
            _kvCacheHostDirty = h.KvHostDirty;
            _convState = h.ConvState;
            _convStateWriteIdx = h.ConvWriteIdx;
            _deltaStateTensor = h.DeltaState;
            _fdConvScratch = h.ConvScratch;
            _fdStateResident = h.FdStateResident;
            _gdnStateHostDirty = h.GdnHostDirty;
            _arenaStateResident = h.ArenaStateResident;
            // TryFullModelDecode keys its descriptor cache on these storage and
            // conv-scratch pointers, so switching holders refreshes bindings once.
            // The native g_q35dc_pool selects the captured graph by the same first
            // attention-KV pointer.
        }

        private Qwen35KvCacheHolder CreateFreshHolder()
        {
            if (_holderPool != null && _holderPool.Count > 0)
            {
                var reused = _holderPool[_holderPool.Count - 1];
                _holderPool.RemoveAt(_holderPool.Count - 1);
                ResetHolderForReuse(reused);
                return reused;
            }
            return AllocateHolder(_initialKvCacheCapacity > 0 ? _initialKvCacheCapacity : _kvCacheCapacity);
        }

        /// <summary>A brand-new, zeroed holder with attention K/V of <paramref name="cap"/>
        /// rows. The allocation half of <see cref="CreateFreshHolder"/>, on its own so a
        /// copy can be sized to its source (a pooled holder may have grown).</summary>
        private Qwen35KvCacheHolder AllocateHolder(int cap)
        {
            int numLayers = _kvCacheK.Length;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convDim = _convKernel - 1;
            DType kvDtype = _kvCacheDtype.ToDType();

            // Construct the owner before acquiring tensors or unmanaged scratch.
            // Every successful allocation is immediately reachable for rollback.
            var holder = new Qwen35KvCacheHolder
            {
                K = new Tensor[numLayers],
                V = new Tensor[numLayers],
                ConvState = new float[numLayers][],
                ConvWriteIdx = new int[numLayers],
                DeltaState = new Tensor[numLayers],
                KvCapacity = cap,
            };
            bool complete = false;
            try
            {
                int gdnCount = 0;
                for (int l = 0; l < numLayers; l++)
                {
                    if (!_isRecurrent[l])
                    {
                        holder.K[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, cap, Config.HeadDim);
                        holder.V[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, cap, Config.HeadDim);
                        InitializeCacheTensor(holder.K[l]);
                        InitializeCacheTensor(holder.V[l]);
                    }
                    else
                    {
                        holder.ConvState[l] = new float[checked(Math.Max(0, convDim) * qkvDim)];
                        holder.DeltaState[l] = AllocateGdnDeltaStateTensor(
                            _allocator, _useMetalGdnInplaceState, _numVHeads, _headVDim, _headKDim);
                        Ops.Fill(holder.DeltaState[l], 0);
                        gdnCount++;
                    }
                }
                nint scratchBytes = checked((nint)((long)Math.Max(1, gdnCount)
                    * Math.Max(1, convDim) * qkvDim * sizeof(float)));
                holder.ConvScratch = Marshal.AllocHGlobal(scratchBytes);
                complete = true;
                return holder;
            }
            finally
            {
                if (!complete) DisposeHolder(holder);
            }
        }

        /// <summary>Make <paramref name="requestId"/>'s KV + GDN state the model's
        /// active state, creating an empty holder the first time the request is
        /// seen. Cheap: swaps references and the conv-scratch pointer. Returns true
        /// when freshly created, so the caller injects any prefix-cache-reused
        /// prefix before the first forward.</summary>
        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            _fusedHolders ??= new Dictionary<string, Qwen35KvCacheHolder>(StringComparer.Ordinal);

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
                return false; // already active

            // Save whatever cache is currently checked out so its (possibly grown)
            // tensors aren't lost when we repoint the active fields.
            if (_activeFusedKey == null)
                _primaryHolder = SnapshotActiveCache();
            else
                _fusedHolders[_activeFusedKey] = SnapshotActiveCache();

            bool fresh;
            if (_fusedHolders.TryGetValue(requestId, out var holder))
            {
                fresh = false;
            }
            else
            {
                holder = CreateFreshHolder();
                _fusedHolders[requestId] = holder;
                fresh = true;
            }
            LoadCacheHolder(holder);
            _activeFusedKey = requestId;
            // A fresh holder's device GDN state isn't seeded yet; force the first
            // fused decode to re-seed from the (zeroed / prefill-filled) host ring.
            return fresh;
        }

        /// <summary>Transition the single in-flight N==1 owner (whose live state is
        /// in the primary cache) into the fused path without copying KV/GDN bytes:
        /// hand the live primary arrays to the owner's holder and give the primary a
        /// fresh empty allocation for later N==1 use.</summary>
        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            _fusedHolders ??= new Dictionary<string, Qwen35KvCacheHolder>(StringComparer.Ordinal);

            if (_activeFusedKey != null)
                return; // a fused holder is already checked out; nothing to adopt
            if (_fusedHolders.ContainsKey(requestId))
                return;

            // The active fields hold the primary cache with the owner's live state.
            // Move those into the owner's holder (zero copy).
            var holder = SnapshotActiveCache();
            _fusedHolders[requestId] = holder;
            _activeFusedKey = requestId;

            // Give the primary a fresh empty allocation so a future N==1 step for a
            // never-fused request doesn't reset the adopted holder's tensors.
            var fresh = CreateFreshHolder();
            _primaryHolder = fresh;
        }

        /// <summary>Reinstate the primary cache as the model's active cache before
        /// an N==1 step that follows a fused episode. No-op when the primary cache
        /// is already active.</summary>
        public void RestorePrimaryCache()
        {
            if (_activeFusedKey == null)
                return;
            // Null-checked like its four siblings (OnSequenceReleased, RetainSequenceCache,
            // TryRebindRetainedCache, DiscardRetainedCache), and this was the one that was
            // not. DisposeAllFusedHolders frees the dictionary and only clears
            // _activeFusedKey afterwards, so a step racing a teardown found exactly the
            // state this now returns from -- reported from an iPhone as the whole of the
            // failure: "Object reference not set to an instance of an object".
            if (_fusedHolders == null)
            {
                _activeFusedKey = null;
                return;
            }
            _fusedHolders[_activeFusedKey] = SnapshotActiveCache();
            _activeFusedKey = null;
            if (_primaryHolder != null)
            {
                LoadCacheHolder(_primaryHolder);
                _primaryHolder = null;
            }
        }

        /// <summary>Release a finished/aborted request's per-request cache. Called
        /// by the engine when a sequence leaves the scheduler.</summary>
        public void OnSequenceReleased(string requestId)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
                return;

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
            {
                // The released sequence's cache is currently checked out. Swap the
                // primary back in so the active fields don't dangle. Snapshot
                // first: growth or state reseeding may have replaced fields since
                // this holder was loaded, making the dictionary entry stale.
                holder = SnapshotActiveCache();
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }

            _fusedHolders.Remove(requestId);
            RecycleHolder(holder);
        }

        /// <summary>Move a cleanly-finished request's complete attention + GDN
        /// holder out of the active set without touching its state. A dirty native
        /// arena slot intentionally remains registered: it is keyed by the holder's
        /// stable storage pointer, so a later rebind can continue in place; normal
        /// arena eviction flushes it back to the same holder before retiring it.</summary>
        public bool RetainSequenceCache(string requestId) => RetainSequenceCacheAs(requestId, requestId);

        /// <summary>The key-parameterised form of <see cref="RetainSequenceCache"/>: the finished
        /// holder of <paramref name="requestId"/> is retained under <paramref name="key"/>
        /// (the prefix cache's tree-minted payload key, or the request id itself).</summary>
        public bool RetainSequenceCacheAs(string requestId, string key)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(key))
                return false;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
                return false;
            if (_retainedFusedHolders != null && _retainedFusedHolders.ContainsKey(key))
                return false;
            _retainedFusedHolders ??= new Dictionary<string, Qwen35KvCacheHolder>(StringComparer.Ordinal);
            _retainedFusedHolders.EnsureCapacity(checked(_retainedFusedHolders.Count + 1));

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
            {
                // Capture replacements caused by growth/reseeding before checking
                // the holder in, then restore the primary model cache. Do not flush
                // or discard the arena slot: it may contain the newest KV/GDN state.
                holder = SnapshotActiveCache();
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }

            _retainedFusedHolders.Add(key, holder);
            _fusedHolders.Remove(requestId);
            return true;
        }

        /// <summary>Re-key a retained complete holder for a new request. The next
        /// BindSequenceCache observes it as non-fresh and forwards only the new
        /// prompt suffix. Qwen advertises no cache truncation, so the executor only
        /// calls this when the holder's entire token run is an exact prompt prefix.</summary>
        public bool TryRebindRetainedCache(string retainedRequestId, string newRequestId)
        {
            if (_retainedFusedHolders == null
                || string.IsNullOrEmpty(retainedRequestId)
                || string.IsNullOrEmpty(newRequestId))
                return false;
            if (!_retainedFusedHolders.TryGetValue(retainedRequestId, out var holder))
                return false;
            if (holder.Retired) return false;

            _fusedHolders ??= new Dictionary<string, Qwen35KvCacheHolder>(StringComparer.Ordinal);
            if (_fusedHolders.ContainsKey(newRequestId))
                return false;

            _fusedHolders.EnsureCapacity(checked(_fusedHolders.Count + 1));
            _fusedHolders.Add(newRequestId, holder);
            _retainedFusedHolders.Remove(retainedRequestId);
            return true;
        }

        /// <summary>Release an unclaimed retained holder (LRU eviction/reset).
        /// Its state is no longer observable, so an arena-only tail is discarded
        /// before the stable host pointers can be pooled or freed.</summary>
        public void DiscardRetainedCache(string requestId)
        {
            if (_retainedFusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_retainedFusedHolders.TryGetValue(requestId, out var holder))
                return;

            holder.Retired = true;
            RecycleHolder(holder);
            _retainedFusedHolders.Remove(requestId);
        }

        private void RecycleHolder(Qwen35KvCacheHolder holder)
        {
            if (holder == null) return;
            if (holder.DisposalStarted)
            {
                DisposeHolder(holder);
                return;
            }
            // The sequence is complete and not retained, so its device-only arena
            // state is no longer observable. Retire the mapping before this stable
            // pointer can be reassigned to another request.
            DiscardArenaSlotForHolder(holder);
            // Retain stable pointers only while host and device headroom can
            // support the idle set. Count limits alone let evicted conversations
            // accumulate gigabytes of otherwise unobservable CUDA cache copies.
            long? spare = GetCacheMemorySpareBytes();
            long holderBytes = IdleHolderBytes(holder);
            long pooledBytes = PooledHolderBytes();
            bool memoryFits = CanPoolIdleCache(holderBytes, pooledBytes, 0, 1, spare);
            if (!memoryFits)
            {
                DisposeHolder(holder);
                TrimIdleMemory();
                return;
            }
            // A parked Metal holder is re-zeroed and re-seeded at reuse, so its
            // aliased state mirrors can be evicted now. CUDA retains them while
            // the measured memory budget permits pointer/graph reuse.
            InvalidateHolderDeviceCopiesForReuse(holder);
            int poolMax = ExecutionOptions.FromEnvironment().KvHolderPoolMax;
            _holderPool ??= new List<Qwen35KvCacheHolder>(Math.Max(1, poolMax));
            if (CanPoolIdleCache(holderBytes, pooledBytes, _holderPool.Count, poolMax, spare))
            {
                _holderPool.Add(holder);
                holder.Retired = false;
            }
            else
            {
                DisposeHolder(holder);
            }
        }

        /// <summary>Free every parked holder, then the host memory pool (base). What the
        /// pool buys -- a later request skipping allocation, zero-fill and graph
        /// capture -- is worth nothing to a process about to be killed for memory.</summary>
        public override void TrimIdleMemory()
        {
            if (_holderPool != null && _holderPool.Count > 0)
            {
                // Taken out of the pool BEFORE anything is disposed, so a dispose that
                // throws part-way cannot leave a freed holder where CreateFreshHolder
                // would hand it out again. Whatever survives the throw is simply leaked
                // until the model is disposed, which is the safe direction.
                var parked = _holderPool;
                _holderPool = null;
                int count = parked.Count;
                foreach (var holder in parked)
                    DisposeHolder(holder);
                Console.WriteLine($"[memory] Qwen35: freed {count} parked K/V holder(s)");
            }
            base.TrimIdleMemory();
        }

        // ---- Shared-prefix checkpoints (IBatchedPagedModel) ----

        /// <summary>Qwen 3.5 can copy its complete state — attention K/V and the
        /// GatedDeltaNet conv ring and delta state — once every device-resident part
        /// has been brought back to the host. GGML only, and not under tensor
        /// parallelism (the cache lives on the ranks there).</summary>
        public bool SupportsPrefixCheckpoints => IsGgmlBackend && !IsTensorParallel && _kvCacheK != null;

        /// <summary>Deep-copy the ACTIVE cache into the retained set under
        /// <paramref name="key"/>. See <see cref="IBatchedPagedModel.TryCheckpointActiveCache"/>.</summary>
        public bool TryCheckpointActiveCache(string key)
        {
            if (!SupportsPrefixCheckpoints || string.IsNullOrEmpty(key) || _isRecurrent == null)
                return false;
            _retainedFusedHolders ??= new Dictionary<string, Qwen35KvCacheHolder>(StringComparer.Ordinal);
            if (_retainedFusedHolders.ContainsKey(key)
                || (_fusedHolders != null && _fusedHolders.ContainsKey(key)))
                return false;
            _retainedFusedHolders.EnsureCapacity(checked(_retainedFusedHolders.Count + 1));
            // Three places can hold state newer than the host bytes the copy reads: the
            // native arena slot and the K/V device mirrors (EnsureKvCacheHostSynchronized
            // flushes and syncs both), the fused-decode conv scratch and delta mirrors
            // (EnsureFusedDecodeStateHostSynchronized), and verify-owned device slices
            // (DrainDeviceRecurrentState). Same order TryExtractKVBlock uses.
            EnsureKvCacheHostSynchronized();
            EnsureFusedDecodeStateHostSynchronized();
            DrainDeviceRecurrentState();
            var copy = DeepCopyHolder(SnapshotActiveCache());
            bool published = false;
            try
            {
                _retainedFusedHolders.Add(key, copy);
                published = true;
            }
            finally
            {
                if (!published) DisposeHolder(copy);
            }
            return true;
        }

        /// <summary>Deep-copy the retained holder <paramref name="retainedKey"/> into a
        /// fresh active holder for <paramref name="newRequestId"/>; the retained one is
        /// untouched. See <see cref="IBatchedPagedModel.TryCloneRetainedCache"/>.</summary>
        public bool TryCloneRetainedCache(string retainedKey, string newRequestId)
        {
            if (_retainedFusedHolders == null
                || string.IsNullOrEmpty(retainedKey)
                || string.IsNullOrEmpty(newRequestId))
                return false;
            if (!_retainedFusedHolders.TryGetValue(retainedKey, out var source))
                return false;
            _fusedHolders ??= new Dictionary<string, Qwen35KvCacheHolder>(StringComparer.Ordinal);
            if (_fusedHolders.ContainsKey(newRequestId)
                || string.Equals(_activeFusedKey, newRequestId, StringComparison.Ordinal))
                return false;
            // A checkpoint is never bound, so its host bytes stay the truth; a retained
            // conversation holder may be device-dirty and is re-keyed, never copied.
            if (source.KvHostDirty || source.GdnHostDirty || source.ArenaStateResident || source.FdStateResident || source.Retired)
                return false;
            _fusedHolders.EnsureCapacity(checked(_fusedHolders.Count + 1));
            var copy = DeepCopyHolder(source);
            bool published = false;
            try
            {
                _fusedHolders.Add(newRequestId, copy);
                published = true;
            }
            finally
            {
                if (!published) DisposeHolder(copy);
            }
            return true;
        }

        // ---- Checkpoints on disk (IBatchedPagedModel) ----
        //
        // A checkpoint's host bytes are the truth (never bound, every residency flag
        // off), which is exactly what lets them be written to a file: attention K/V
        // rows per layer, and per recurrent layer the conv ring, its write index and
        // the delta state read through the offset pointer the Metal in-place layout
        // requires. The file names the model's K/V identity and every dimension it
        // depends on, and an import that finds anything different creates nothing.

        public bool SupportsRetainedCacheSerialization => SupportsPrefixCheckpoints;

        private const uint CheckpointFileMagic = 0x51354B43;   // "Q5KC"
        private const int CheckpointFileVersion = 1;

        public unsafe bool TryExportRetainedCache(string key, System.IO.Stream destination)
        {
            if (!SupportsPrefixCheckpoints || destination == null || string.IsNullOrEmpty(key)
                || _retainedFusedHolders == null || _isRecurrent == null)
                return false;
            if (!_retainedFusedHolders.TryGetValue(key, out var h) || h.K == null)
                return false;
            // Only host-authoritative state can be written: a checkpoint is never bound.
            if (h.KvHostDirty || h.GdnHostDirty || h.ArenaStateResident || h.FdStateResident || h.Retired)
                return false;

            int rows = Math.Max(0, Math.Min(h.CacheSeqLen, h.KvCapacity));
            int numLayers = h.K.Length;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            var w = new System.IO.BinaryWriter(destination, System.Text.Encoding.UTF8, leaveOpen: true);
            w.Write(CheckpointFileMagic);
            w.Write(CheckpointFileVersion);
            w.Write(KVStateFingerprint ?? string.Empty);
            w.Write(numLayers);
            w.Write(rows);
            w.Write(Config.NumKVHeads);
            w.Write(Config.HeadDim);
            w.Write(_convKernel);
            w.Write(qkvDim);
            for (int l = 0; l < numLayers; l++)
            {
                bool recurrent = l < _isRecurrent.Length && _isRecurrent[l];
                w.Write(recurrent);
                if (!recurrent)
                {
                    WriteCacheRows(w, h.K[l], rows);
                    WriteCacheRows(w, h.V[l], rows);
                    continue;
                }
                float[] conv = h.ConvState[l];
                w.Write(conv?.Length ?? 0);
                if (conv != null && conv.Length > 0)
                {
                    w.Flush();
                    destination.Write(MemoryMarshal.AsBytes(conv.AsSpan()));
                }
                w.Write(h.ConvWriteIdx[l]);
                Tensor delta = h.DeltaState[l];
                long deltaBytes = delta == null ? 0 : GdnDeltaStateBytes(delta);
                w.Write(deltaBytes);
                if (delta != null && deltaBytes > 0)
                {
                    delta.Storage.EnsureHostReadable();
                    w.Flush();
                    destination.Write(new ReadOnlySpan<byte>((void*)GdnDeltaStatePointer(delta), checked((int)deltaBytes)));
                }
            }
            w.Flush();
            return true;
        }

        public unsafe bool TryImportRetainedCache(string key, System.IO.Stream source)
        {
            if (!SupportsPrefixCheckpoints || source == null || string.IsNullOrEmpty(key) || _isRecurrent == null)
                return false;
            _retainedFusedHolders ??= new Dictionary<string, Qwen35KvCacheHolder>(StringComparer.Ordinal);
            if (_retainedFusedHolders.ContainsKey(key) || (_fusedHolders != null && _fusedHolders.ContainsKey(key)))
                return false;
            _retainedFusedHolders.EnsureCapacity(checked(_retainedFusedHolders.Count + 1));

            var r = new System.IO.BinaryReader(source, System.Text.Encoding.UTF8, leaveOpen: true);
            if (r.ReadUInt32() != CheckpointFileMagic || r.ReadInt32() != CheckpointFileVersion)
                return false;
            if (!string.Equals(r.ReadString(), KVStateFingerprint ?? string.Empty, StringComparison.Ordinal))
                return false;
            int numLayers = r.ReadInt32();
            int rows = r.ReadInt32();
            if (numLayers != _kvCacheK.Length || rows < 0 || rows > _maxContextLength)
                return false;
            if (r.ReadInt32() != Config.NumKVHeads || r.ReadInt32() != Config.HeadDim)
                return false;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            if (r.ReadInt32() != _convKernel || r.ReadInt32() != qkvDim)
                return false;

            int cap = Math.Max(1, Math.Min(_maxContextLength, CacheCapacityFor(rows)));
            Qwen35KvCacheHolder h = AllocateHolder(cap);
            bool ok = false;
            try
            {
                for (int l = 0; l < numLayers; l++)
                {
                    bool recurrent = r.ReadBoolean();
                    bool expected = l < _isRecurrent.Length && _isRecurrent[l];
                    if (recurrent != expected)
                        return false;
                    if (!recurrent)
                    {
                        if (!ReadCacheRows(r, h.K[l], rows) || !ReadCacheRows(r, h.V[l], rows))
                            return false;
                        InvalidateTensorDeviceCache(h.K[l]);
                        InvalidateTensorDeviceCache(h.V[l]);
                        continue;
                    }
                    int convLen = r.ReadInt32();
                    if (convLen != (h.ConvState[l]?.Length ?? 0))
                        return false;
                    if (convLen > 0)
                        source.ReadExactly(MemoryMarshal.AsBytes(h.ConvState[l].AsSpan()));
                    h.ConvWriteIdx[l] = r.ReadInt32();
                    long deltaBytes = r.ReadInt64();
                    Tensor delta = h.DeltaState[l];
                    long expectedBytes = delta == null ? 0 : GdnDeltaStateBytes(delta);
                    if (deltaBytes != expectedBytes)
                        return false;
                    if (deltaBytes > 0)
                    {
                        delta.Storage.EnsureHostReadable();
                        source.ReadExactly(new Span<byte>((void*)GdnDeltaStatePointer(delta), checked((int)deltaBytes)));
                        if (IsGgmlBackend)
                            InvalidateGdnDeltaStateDeviceCaches(delta);
                    }
                }
                h.CacheSeqLen = rows;
                h.KvHostDirty = false;
                h.GdnHostDirty = false;
                h.FdStateResident = false;
                h.ArenaStateResident = false;
                h.Logits = null;
                _retainedFusedHolders.Add(key, h);
                ok = true;
                return true;
            }
            catch (System.IO.EndOfStreamException)
            {
                return false;
            }
            finally
            {
                if (!ok)
                    DisposeHolder(h);
            }
        }

        /// <summary>The written rows of one K or V cache tensor, head by head: the
        /// per-head stride is the tensor's CAPACITY, which the reader may size
        /// differently, so only the rows are written.</summary>
        private static unsafe void WriteCacheRows(System.IO.BinaryWriter w, Tensor t, int rows)
        {
            long heads = t.Sizes[0];
            long cap = t.Sizes[1];
            long rowBytes = heads * cap == 0 ? 0 : t.Storage.ByteLength / (heads * cap);
            w.Write((int)heads);
            w.Write(rowBytes);
            if (rows == 0 || heads == 0 || rowBytes == 0)
                return;
            t.Storage.EnsureHostReadable();
            byte* src = (byte*)t.Storage.PtrAtElement(0);
            w.Flush();
            for (long head = 0; head < heads; head++)
                w.BaseStream.Write(new ReadOnlySpan<byte>(src + head * cap * rowBytes, checked((int)(rows * rowBytes))));
        }

        private static unsafe bool ReadCacheRows(System.IO.BinaryReader r, Tensor t, int rows)
        {
            long heads = t.Sizes[0];
            long cap = t.Sizes[1];
            long rowBytes = heads * cap == 0 ? 0 : t.Storage.ByteLength / (heads * cap);
            if (r.ReadInt32() != heads || r.ReadInt64() != rowBytes)
                return false;
            if (rows > cap)
                return false;
            if (rows == 0 || heads == 0 || rowBytes == 0)
                return true;
            t.Storage.EnsureHostReadable();
            byte* dst = (byte*)t.Storage.PtrAtElement(0);
            for (long head = 0; head < heads; head++)
                r.BaseStream.ReadExactly(new Span<byte>(dst + head * cap * rowBytes, checked((int)(rows * rowBytes))));
            return true;
        }

        /// <summary>An independent copy of <paramref name="source"/>, whose host bytes
        /// must be current: fresh tensors sized to the source, every attention K/V
        /// storage and every recurrent layer's conv ring, write index and delta state
        /// copied, and every residency flag off so the next fused decode re-seeds its
        /// device state from the copied host state and builds its own graphs.</summary>
        private unsafe Qwen35KvCacheHolder DeepCopyHolder(Qwen35KvCacheHolder source)
        {
            // Sized to what the source HOLDS, not to what it reserved. The first
            // request's primary cache was reserved for its whole generation budget
            // (PrepareForPrefill), and a checkpoint is kept for the life of the model
            // while every new chat gets a clone of it; copying and pinning that whole
            // reservation each time is hundreds of megabytes a phone does not have.
            // The copy grows on demand like any holder when a chat outlives it.
            int rows = Math.Max(0, Math.Min(source.CacheSeqLen, source.KvCapacity));
            int cap = Math.Min(source.KvCapacity, CacheCapacityFor(rows));
            var dst = AllocateHolder(Math.Max(cap, 1));
            bool complete = false;
            try
            {
                int numLayers = Math.Min(source.K.Length, dst.K.Length);
                for (int l = 0; l < numLayers; l++)
                {
                    bool recurrent = l < _isRecurrent.Length && _isRecurrent[l];
                    if (!recurrent)
                    {
                        if (source.K[l] != null && dst.K[l] != null)
                        {
                            CopyCacheRows(source.K[l], dst.K[l], rows);
                            InvalidateTensorDeviceCache(dst.K[l]);
                        }
                        if (source.V[l] != null && dst.V[l] != null)
                        {
                            CopyCacheRows(source.V[l], dst.V[l], rows);
                            InvalidateTensorDeviceCache(dst.V[l]);
                        }
                        continue;
                    }

                    if (source.ConvState[l] != null && dst.ConvState[l] != null)
                        Array.Copy(source.ConvState[l], dst.ConvState[l], Math.Min(source.ConvState[l].Length, dst.ConvState[l].Length));
                    dst.ConvWriteIdx[l] = source.ConvWriteIdx[l];

                    Tensor from = source.DeltaState[l];
                    Tensor to = dst.DeltaState[l];
                    if (from != null && to != null)
                    {
                        long bytes = GdnDeltaStateBytes(from);
                        if (bytes != GdnDeltaStateBytes(to))
                            throw new InvalidOperationException("delta-state tensors differ in size");
                        from.Storage.EnsureHostReadable();
                        to.Storage.EnsureHostReadable();
                        // The Metal in-place layout is a VIEW at an offset inside a backing
                        // [attention output | state] storage: copy the state slice only,
                        // through the offset pointer, and drop both of the copy's mirrors.
                        Buffer.MemoryCopy((void*)GdnDeltaStatePointer(from), (void*)GdnDeltaStatePointer(to), bytes, bytes);
                        if (IsGgmlBackend)
                            InvalidateGdnDeltaStateDeviceCaches(to);
                    }
                }
                dst.CacheSeqLen = source.CacheSeqLen;
                dst.KvHostDirty = false;
                dst.GdnHostDirty = false;
                dst.FdStateResident = false;
                dst.ArenaStateResident = false;
                dst.Logits = null;
                complete = true;
                return dst;
            }
            finally
            {
                if (!complete) DisposeHolder(dst);
            }
        }

        private void DisposeHolder(Qwen35KvCacheHolder holder)
        {
            if (holder == null) return;
            holder.Retired = true;
            holder.DisposalStarted = true;

            // Persistent whole-model decode graphs capture the device addresses
            // backing this holder's K/V and recurrent-state mirrors.  Drop those
            // graphs before evicting any mirror or freeing its host key; otherwise
            // a recycled host pointer could select a graph that still references
            // freed device memory.  Releases are infrequent (one per completed
            // concurrent request), so rebuilding the surviving holders' graphs on
            // their next token is a small price for deterministic lifetime safety.
            // A batched release (DiscardRetainedCaches) resets once, before its
            // first disposal, and suppresses the per-holder reset here.
            if (IsGgmlBackend && _holderGraphResetSuppressed == 0)
            {
                GgmlBasicOps.Qwen35ResetDecodeCache();
                // Persistent fused prefill/spec verify graphs bind the same
                // holder K/V and recurrent-state buffers. Release them before
                // invalidating those host keys as well.
                InvalidateVerifyCache();
                CountDecodeGraphReset();
            }

            if (holder.K != null)
                for (int i = 0; i < holder.K.Length; i++)
                {
                    var t = holder.K[i];
                    InvalidateTensorDeviceCache(t);
                    t?.Dispose();
                    holder.K[i] = null;
                }
            if (holder.V != null)
                for (int i = 0; i < holder.V.Length; i++)
                {
                    var t = holder.V[i];
                    InvalidateTensorDeviceCache(t);
                    t?.Dispose();
                    holder.V[i] = null;
                }
            if (holder.DeltaState != null)
                for (int i = 0; i < holder.DeltaState.Length; i++)
                {
                    var t = holder.DeltaState[i];
                    if (IsGgmlBackend)
                        InvalidateGdnDeltaStateDeviceCaches(t);
                    t?.Dispose();
                    holder.DeltaState[i] = null;
                }
            if (holder.ConvScratch != IntPtr.Zero)
            {
                // Each recurrent layer binds its offset within ConvScratch as an
                // independent cache key (not the enclosing allocation pointer).
                int convDim = _convKernel - 1;
                int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
                long layerBytes = (long)Math.Max(0, convDim) * qkvDim * sizeof(float);
                int recurrentSlot = 0;
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (!_isRecurrent[l]) continue;
                    if (IsGgmlBackend && layerBytes > 0)
                    {
                        long byteOffset = recurrentSlot * layerBytes;
                        GgmlBasicOps.InvalidateHostBuffer(
                            new IntPtr(holder.ConvScratch.ToInt64() + byteOffset));
                    }
                    recurrentSlot++;
                }
                Marshal.FreeHGlobal(holder.ConvScratch);
                holder.ConvScratch = IntPtr.Zero;
            }
        }

        /// <summary>Free every per-request fused holder (and the saved primary
        /// snapshot). Called on model dispose. Does NOT touch the currently-active
        /// arrays (those are the model's _kvCacheK / _deltaStateTensor / _fdConvScratch,
        /// freed by the normal cache teardown).</summary>
        private void DisposeAllFusedHolders()
        {
            // Read once, before the key is cleared: the primary-holder block below
            // still has to know whether a fused holder was active, and the key itself
            // must not outlive the dictionary it indexes (see RestorePrimaryCache).
            bool fusedWasActive = _activeFusedKey != null;
            if (_fusedHolders != null)
            {
                foreach (var kv in _fusedHolders)
                {
                    if (string.Equals(kv.Key, _activeFusedKey, StringComparison.Ordinal))
                        continue; // active holder shares the model fields
                    DisposeHolder(kv.Value);
                }
                _fusedHolders.Clear();
                _fusedHolders = null;
                // Before anything else can look: _activeFusedKey is a key INTO the
                // dictionary just freed, and every reader that trusts the key to name a
                // live entry is correct only while both are true together. Clearing it
                // last left a window in which the key outlived what it pointed at.
                _activeFusedKey = null;
            }
            if (_retainedFusedHolders != null)
            {
                foreach (var holder in _retainedFusedHolders.Values)
                    DisposeHolder(holder);
                _retainedFusedHolders.Clear();
                _retainedFusedHolders = null;
            }
            if (_holderPool != null)
            {
                foreach (var h in _holderPool)
                    DisposeHolder(h);
                _holderPool = null;
            }
            if (_primaryHolder != null)
            {
                // If a fused holder was active, the primary snapshot owns distinct
                // arrays that must be freed; if the primary is active it shares the
                // model fields and is freed by the main teardown.
                if (fusedWasActive)
                    DisposeHolder(_primaryHolder);
                _primaryHolder = null;
            }
            _activeFusedKey = null;
        }
    }
}
