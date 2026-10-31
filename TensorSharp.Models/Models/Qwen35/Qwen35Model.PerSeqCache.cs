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
// token-batched fused decode). Measured on ggml_cuda that path
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
using TensorSharp.MLX;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

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
            // M-RoPE delta of this sequence: rope position = KV index + RopeDelta for
            // every token past its prompt's position table (non-zero after an image).
            // It travels with the rows it describes - swapped in and out with the
            // holder, copied with a checkpoint, written into a checkpoint file.
            public int RopeDelta;
            public int MtpCacheStart;
            public bool KvHostDirty;
            // GDN recurrent state: host conv ring + write idx + device delta state.
            public float[][] ConvState;
            public int[] ConvWriteIdx;
            public Tensor[] DeltaState;
            // MLX keeps the GDN state in its own per-layer caches, not in the host ring
            // and delta tensor above (which MLX never writes); a holder carries its own
            // set so swapping holders swaps the recurrence too. Null on other backends.
            public MlxFusedOps.GatedDeltaNetCache[] MlxGdn;
            // Bytes those caches hold once they have run (conv tail + delta state).
            public long MlxGdnStateBytes;
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
            // Last use of a retained holder's device K/V copies (clone source, capture,
            // retention); the least recent are demoted to host first under pressure.
            public long DeviceUseStamp;
            // Clones taken from this retained holder. Demotion takes holders nothing has
            // reused (a finished request's end state, a checkpoint never matched) before
            // ones that have served a clone, whatever their recency.
            public int CloneCount;
            // Background download of this idle holder's K/V rows to its host bytes
            // (ScheduleIdleKvSyncs). True on success; every path that reads, frees or
            // binds the holder's K/V waits for it first (WaitIdleKvSync).
            public System.Threading.Tasks.Task<bool> IdleKvSync;
            // IdleKvSync is a checkpoint capture filling this holder's otherwise
            // unwritten host K/V: a failure leaves no copy of the rows to fall back on.
            public bool IdleKvSyncIsCapture;
            // A checkpoint file is being written from this holder's host bytes on another
            // thread; nothing may free or rewrite them until it completes.
            public System.Threading.Tasks.Task ExportTask;
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
            h.RopeDelta = 0;
            h.MtpCacheStart = 0;
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
            if (h.MlxGdn != null)
                foreach (var cache in h.MlxGdn)
                    cache?.Reset();
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
        // MLX: the holders carry their own MLX GDN caches (Qwen35KvCacheHolder.MlxGdn)
        // and the request's forward is the ordinary per-op one, so a swap is the same
        // reference flip; this is also what lets a new chat start from the shared-prefix
        // checkpoint instead of prefilling the system prompt again.
        public bool SupportsPerSequenceFusedForward =>
            !IsTensorParallel
            && ((_backend == BackendType.GgmlCuda && !_fdUnsupported)
                || _backend == BackendType.GgmlMetal
                || _backend == BackendType.Mlx);

        public bool SupportsRetainedFusedCache => true;

        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _fusedHolders != null && _fusedHolders.ContainsKey(requestId);

        private Qwen35KvCacheHolder SnapshotActiveCache() => new Qwen35KvCacheHolder
        {
            K = _kvCacheK,
            V = _kvCacheV,
            KvCapacity = _kvCacheCapacity,
            CacheSeqLen = _cacheSeqLen,
            RopeDelta = _ropeDelta,
            MtpCacheStart = _mtpCacheStart,
            KvHostDirty = _kvCacheHostDirty,
            ConvState = _convState,
            ConvWriteIdx = _convStateWriteIdx,
            DeltaState = _deltaStateTensor,
            MlxGdn = _mlxGdnCache,
            MlxGdnStateBytes = _mlxGdnCache == null ? 0 : MlxGdnStateBytesPerHolder(),
            ConvScratch = _fdConvScratch,
            FdStateResident = _fdStateResident,
            GdnHostDirty = _gdnStateHostDirty,
            ArenaStateResident = _arenaStateResident,
            DeviceMirrored = KeepsDeviceKvMirrors,
        };

        private void LoadCacheHolder(Qwen35KvCacheHolder h)
        {
            WaitIdleKvSync(h);
            WaitExport(h);
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
            _ropeDelta = h.RopeDelta;
            _mtpCacheStart = h.MtpCacheStart;
            _kvCacheHostDirty = h.KvHostDirty;
            _convState = h.ConvState;
            _convStateWriteIdx = h.ConvWriteIdx;
            _deltaStateTensor = h.DeltaState;
            _mlxGdnCache = h.MlxGdn;
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

        // Only active CUDA clones opt into this reuse. Long-lived checkpoints
        // still allocate compact independent storage, and other backends retain
        // their existing host/device ownership rules. Exact capacity prevents an
        // idle large reservation from inflating a smaller request's footprint.
        private Qwen35KvCacheHolder TakePooledCloneHolder(int cap)
        {
            if (_backend != BackendType.GgmlCuda || _holderPool == null)
                return null;
            for (int i = _holderPool.Count - 1; i >= 0; i--)
            {
                var holder = _holderPool[i];
                if (holder.KvCapacity != cap || holder.DisposalStarted || holder.Retired)
                    continue;
                _holderPool.RemoveAt(i); // One owner from here through copy/publication or disposal.
                return holder;
            }
            return null;
        }

        private void PreparePooledHolderForClone(Qwen35KvCacheHolder holder)
        {
            // Discard the old request's arena mapping BEFORE overwriting its host
            // bytes. Invalidation may flush other live slots, but must never flush
            // this completed slot over the new prefix. Existing invalidation hooks
            // retire graphs before freeing their cached device bindings.
            DiscardArenaSlotForHolder(holder);
            WaitCaptureReads(holder.K, holder.V);
            foreach (var set in new[] { holder.K, holder.V })
                if (set != null)
                    foreach (var tensor in set) InvalidateTensorDeviceCache(tensor);
            if (holder.DeltaState != null)
                foreach (var tensor in holder.DeltaState)
                    if (tensor != null) InvalidateGdnDeltaStateDeviceCaches(tensor);
            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            long layerBytes = (long)Math.Max(0, convDim) * qkvDim * sizeof(float);
            int recurrentSlot = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l]) continue;
                if (holder.ConvScratch != IntPtr.Zero && layerBytes > 0)
                    GgmlBasicOps.InvalidateHostBuffer(new IntPtr(holder.ConvScratch.ToInt64() + recurrentSlot * layerBytes));
                recurrentSlot++;
            }
            // Do not ResetHolderForReuse: the copy below overwrites every live
            // attention row and recurrent state, then clears only KV padding.
            holder.DeviceMirrored = false;
        }

        /// <summary>A brand-new, zeroed holder with attention K/V of <paramref name="cap"/>
        /// rows. The allocation half of <see cref="CreateFreshHolder"/>, on its own so a
        /// copy can be sized to its source (a pooled holder may have grown).</summary>
        private Qwen35KvCacheHolder AllocateHolder(int cap)
            => AllocateHolderStorage(cap, overwriteAttention: false);

        /// <param name="overwriteAttention">The caller writes every attention row it reads; skip zeroing K/V.</param>
        /// <param name="overwriteState">The caller copies every recurrent layer's delta state; skip
        /// zeroing it (on fresh memory the fill is a page-faulting pass the copy repeats).</param>
        private Qwen35KvCacheHolder AllocateHolderStorage(int cap, bool overwriteAttention, bool overwriteState = false)
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
                MlxGdn = _backend == BackendType.Mlx ? new MlxFusedOps.GatedDeltaNetCache[numLayers] : null,
                MlxGdnStateBytes = _backend == BackendType.Mlx ? MlxGdnStateBytesPerHolder() : 0,
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
                        if (!overwriteAttention)
                        {
                            InitializeCacheTensor(holder.K[l]);
                            InitializeCacheTensor(holder.V[l]);
                        }
                    }
                    else
                    {
                        holder.ConvState[l] = new float[checked(Math.Max(0, convDim) * qkvDim)];
                        holder.DeltaState[l] = AllocateGdnDeltaStateTensor(
                            _allocator, _useMetalGdnInplaceState, _numVHeads, _headVDim, _headKDim);
                        // 3 MB per layer, 151 MB per Qwen3.8 holder: on a fresh mapping the fill
                        // page-faulted ~0.15 s into every new request once the pool had been trimmed.
                        if (!overwriteState && !holder.DeltaState[l].Storage.TakeAllocationZeroGuarantee())
                            Ops.Fill(holder.DeltaState[l], 0);
                        if (holder.MlxGdn != null)
                            holder.MlxGdn[l] = new MlxFusedOps.GatedDeltaNetCache();
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
            // tensors aren't lost when we repoint the active fields. Its verify state
            // moves into its own device copies first, so the snapshot records where
            // that state now lives.
            SettleVerifyStateOnDevice();
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
            // Allocate the replacement before publishing the move. If allocation
            // fails, the live primary still owns its unchanged KV/recurrent state.
            _fusedHolders.EnsureCapacity(checked(_fusedHolders.Count + 1));
            var fresh = CreateFreshHolder();
            _fusedHolders[requestId] = holder;
            _activeFusedKey = requestId;

            // Give the primary a fresh empty allocation so a future N==1 step for a
            // never-fused request doesn't reset the adopted holder's tensors.
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
            SettleVerifyStateOnDevice();
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
                // this holder was loaded, making the dictionary entry stale. Nothing
                // can read its verify state any more, so it is dropped, not drained.
                DiscardVerifyStateOfReleasedCache();
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
                // A device-held verify state is settled first, so the snapshot's
                // residency flags say where it went (switching back to the primary
                // would otherwise move it after the flags were recorded).
                SettleVerifyStateOnDevice();
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
            TouchDeviceRetained(holder);
            ScheduleIdleKvSyncs();
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
            WaitIdleKvSync(holder);
            WaitExport(holder);

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
            WaitIdleKvSync(holder);
            WaitExport(holder);
            WaitCaptureReads(holder.K, holder.V);
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
                TrimIdleMemoryForPressure();
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
            DisposeParkedHolders();
            base.TrimIdleMemory();
        }

        private void DisposeParkedHolders()
        {
            if (_holderPool == null || _holderPool.Count == 0)
                return;
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

        /// <summary>
        /// <see cref="TrimIdleMemory"/> for a holder that could not be parked: the parked
        /// holders (and their device copies) are freed, but the host memory pool goes back
        /// to the system only when host memory is what ran short. With four long Qwen3.8
        /// requests device memory is the usual limit, and releasing the host pool on every
        /// such discard cost ~0.6 s of unmapping on the worker and turned the next
        /// holders' host allocations into page faults.
        /// </summary>
        private const long HostPoolKeepSpareBytes = 4L * 1024 * 1024 * 1024;

        private void TrimIdleMemoryForPressure()
        {
            long? hostSpare = GetHostMemorySpareBytes();
            if (hostSpare.HasValue && hostSpare.Value >= HostPoolKeepSpareBytes)
                DisposeParkedHolders();
            else
                TrimIdleMemory();
        }

        // ---- Shared-prefix checkpoints (IBatchedPagedModel) ----

        /// <summary>Qwen 3.5 can copy its complete state — attention K/V and the
        /// GatedDeltaNet conv ring and delta state — once every device-resident part
        /// has been brought back to the host. GGML only, and not under tensor
        /// parallelism (the cache lives on the ranks there).</summary>
        public bool SupportsPrefixCheckpoints =>
            (IsGgmlBackend || _backend == BackendType.Mlx) && !IsTensorParallel && _kvCacheK != null;

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
            // native arena slot and the K/V device mirrors (SynchronizeCheckpointKvRows
            // flushes the slot and syncs committed rows), the fused-decode conv scratch and delta mirrors
            // (EnsureFusedDecodeStateHostSynchronized), and verify-owned device slices
            // (DrainDeviceRecurrentState). Same order TryExtractKVBlock uses. On CUDA the
            // committed K/V rows are copied on the device instead: the checkpoint keeps
            // device-authoritative K/V and host recurrent state.
            long captureStarted = PrefixReleaseDebug ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            FlushArenaSlotForActiveHolder();
            EnsureFusedDecodeStateHostSynchronized();
            // Between prefill chunks the verifier holds the newest recurrent state. The device
            // route copies it straight into the checkpoint's host bytes and leaves it live:
            // a drain handed it to the active cache's host copy, which the checkpoint then
            // duplicated and the next chunk uploaded again (151 MB each on Qwen3.8-27B).
            bool stateFromLive = s_captureReadsLiveState && _fvDeviceStateCurrent && _backend == BackendType.GgmlCuda;
            if (!stateFromLive)
                DrainDeviceRecurrentState();
            double syncedMs = PrefixReleaseDebug ? System.Diagnostics.Stopwatch.GetElapsedTime(captureStarted).TotalMilliseconds : 0;
            string route = "host";
            Qwen35KvCacheHolder copy = null;
            Qwen35KvCacheHolder active = SnapshotActiveCache();
            if (CanCopyHolderKvOnDevice(active)
                && EnsureDeviceRetainHeadroom(HolderKvBytes(active, CloneCapacity(active, 0)), requireReserve: true))
            {
                copy = DeepCopyHolderOnDevice(active, 0, recurrentFromLive: stateFromLive);
                route = stateFromLive ? "device+live" : "device";
            }
            if (copy == null && stateFromLive)
            {
                // The other routes copy the active cache's host state: bring it current.
                DrainDeviceRecurrentState();
                active = SnapshotActiveCache();
            }
            if (copy == null && CanCaptureCheckpointInBackground())
            {
                var pairs = new List<(Tensor Source, Tensor Destination)>();
                copy = DeepCopyHolder(active, 0, pairs);
                StartCaptureCopy(copy, pairs, Math.Max(0, Math.Min(active.CacheSeqLen, active.KvCapacity)));
                route = "background";
            }
            double rowSyncMs = 0;
            if (copy == null)
            {
                long rowSyncStarted = PrefixReleaseDebug ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                SynchronizeCheckpointKvRows();
                if (PrefixReleaseDebug) rowSyncMs = System.Diagnostics.Stopwatch.GetElapsedTime(rowSyncStarted).TotalMilliseconds;
                copy = DeepCopyHolder(SnapshotActiveCache());
            }
            if (PrefixReleaseDebug)
                Console.Error.WriteLine($"[prefix-capture] {active.CacheSeqLen} tokens via {route}: state sync {syncedMs:F1} ms, " +
                    $"row sync {rowSyncMs:F1} ms, hostDirty={_kvCacheHostDirty}, " +
                    $"total {System.Diagnostics.Stopwatch.GetElapsedTime(captureStarted).TotalMilliseconds:F1} ms");
            bool published = false;
            try
            {
                _retainedFusedHolders.Add(key, copy);
                TouchDeviceRetained(copy);
                published = true;
                ScheduleIdleKvSyncs();
            }
            finally
            {
                if (!published) DisposeHolder(copy);
            }
            return true;
        }

        /// <summary>A checkpoint copies only the committed rows. Downloading the
        /// entire reserved context first transfers unused capacity on every turn.
        /// Keep the active cache device-dirty: its unwritten rows (including an MTP
        /// draft ahead of the trunk) have deliberately not been synchronized.</summary>
        private void SynchronizeCheckpointKvRows()
        {
            if (_backend != BackendType.GgmlCuda)
            {
                EnsureKvCacheHostSynchronized();
                return;
            }
            FlushArenaSlotForActiveHolder();
            if (!_kvCacheHostDirty) return;
            int rows = Math.Max(0, Math.Min(_cacheSeqLen, _kvCacheCapacity));
            if (rows == 0) return;
            foreach (Tensor[] tensors in new[] { _kvCacheK, _kvCacheV })
                foreach (Tensor tensor in tensors)
                {
                    if (tensor == null) continue;
                    IntPtr pointer = TensorComputePrimitives.GetStorageBasePointer(tensor);
                    // The range API also serves optional mirrors and accepts an
                    // absent or resized entry as a no-op. A device-dirty checkpoint
                    // cannot accept that: use the full sync's checked path instead.
                    if (!GgmlBasicOps.HasResidentDeviceCopy(pointer, tensor.Storage.ByteLength))
                    {
                        SyncTensorHostCache(tensor);
                        continue;
                    }
                    int heads = checked((int)tensor.Sizes[0]);
                    long capacity = tensor.Sizes[1];
                    long rowBytes = tensor.Storage.ByteLength / checked(heads * capacity);
                    var offsets = new long[heads];
                    var lengths = new long[heads];
                    for (int h = 0; h < heads; h++)
                    {
                        offsets[h] = checked(h * capacity * rowBytes);
                        lengths[h] = checked(rows * rowBytes);
                    }
                    GgmlBasicOps.SyncHostBufferRanges(pointer, offsets, lengths);
                }
        }

        /// <summary>Deep-copy the retained holder <paramref name="retainedKey"/> into a
        /// fresh active holder for <paramref name="newRequestId"/>; the retained one is
        /// untouched. See <see cref="IBatchedPagedModel.TryCloneRetainedCache"/>.</summary>
        public bool TryCloneRetainedCache(string retainedKey, string newRequestId)
            => CloneRetainedCacheCore(retainedKey, newRequestId, 0);

        bool IHolderPrefixCacheModel.TryCloneRetainedCache(string retainedKey, string newRequestId, int requiredContextTokens)
            => CloneRetainedCacheCore(retainedKey, newRequestId, requiredContextTokens);

        private bool CloneRetainedCacheCore(string retainedKey, string newRequestId, int requiredContextTokens)
        {
            if (_retainedFusedHolders == null
                || string.IsNullOrEmpty(retainedKey)
                || string.IsNullOrEmpty(newRequestId))
                return false;
            if (!_retainedFusedHolders.TryGetValue(retainedKey, out var source))
                return false;
            long cloneStarted = PrefixReleaseDebug ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            WaitIdleKvSync(source);
            double waitedMs = PrefixReleaseDebug ? System.Diagnostics.Stopwatch.GetElapsedTime(cloneStarted).TotalMilliseconds : 0;
            _fusedHolders ??= new Dictionary<string, Qwen35KvCacheHolder>(StringComparer.Ordinal);
            if (_fusedHolders.ContainsKey(newRequestId)
                || string.Equals(_activeFusedKey, newRequestId, StringComparison.Ordinal))
                return false;
            // A checkpoint is never bound, so its recurrent state stays in host bytes; a
            // retained conversation holder with device-only recurrent state is re-keyed,
            // never copied. CUDA K/V may be device-authoritative and is copied there.
            if (source.GdnHostDirty || source.ArenaStateResident || source.FdStateResident || source.Retired)
                return false;
            // The clone becomes the active cache and needs its K/V on the device either
            // way; make room by demoting idle retained copies, without requiring the
            // reserve. Giving a host-only checkpoint a device copy is optional and does.
            int cloneCapacity = CloneCapacity(source, requiredContextTokens);
            bool onDevice = CanCopyHolderKvOnDevice(source);
            if (!onDevice && !source.KvHostDirty
                && EnsureDeviceRetainHeadroom(HolderKvBytes(source, source.KvCapacity) + HolderKvBytes(source, cloneCapacity), requireReserve: true))
                onDevice = TryUploadHolderKvForDeviceCopies(source);
            double uploadedMs = PrefixReleaseDebug ? System.Diagnostics.Stopwatch.GetElapsedTime(cloneStarted).TotalMilliseconds : 0;
            if (onDevice)
            {
                TouchDeviceRetained(source);
                EnsureDeviceRetainHeadroom(HolderKvBytes(source, cloneCapacity), requireReserve: false, exclude: source);
            }
            double headroomMs = PrefixReleaseDebug ? System.Diagnostics.Stopwatch.GetElapsedTime(cloneStarted).TotalMilliseconds : 0;
            if (source.KvHostDirty && !onDevice)
                return false;
            _fusedHolders.EnsureCapacity(checked(_fusedHolders.Count + 1));
            var copy = onDevice ? DeepCopyHolderOnDevice(source, requiredContextTokens, liveClone: true) : null;
            if (copy == null && source.KvHostDirty)
                return false;
            string route = copy != null ? "device" : "host";
            // A checkpoint that stays in host memory (no room to give it device copies of
            // its own) is uploaded straight into the clone's device copies: the clone is
            // bound next and needs them either way, and the host route first copied every
            // row on the host (~100 ms at 17K tokens) and then uploaded the copy.
            if (copy == null && !onDevice && DeviceHolderCopiesEnabled && !source.ArenaStateResident)
            {
                EnsureDeviceRetainHeadroom(HolderKvBytes(source, cloneCapacity), requireReserve: false, exclude: source);
                copy = DeepCopyHolderOnDevice(source, requiredContextTokens, uploadHostRows: true, liveClone: true);
                if (copy != null) route = "upload";
            }
            copy ??= DeepCopyHolder(source, requiredContextTokens);
            if (PrefixReleaseDebug)
                Console.Error.WriteLine($"[prefix-clone] {source.CacheSeqLen} tokens via {route} " +
                    $"(capacity {copy.KvCapacity}): {System.Diagnostics.Stopwatch.GetElapsedTime(cloneStarted).TotalMilliseconds:F1} ms " +
                    $"(sync wait {waitedMs:F1}, upload {uploadedMs - waitedMs:F1}, headroom {headroomMs - uploadedMs:F1})");
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
            source.CloneCount++;
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
        // 2: the M-RoPE delta follows the row count. A version-1 file names no delta,
        // so it is refused (the prefix is prefilled and saved again in version 2)
        // rather than restored as if its rows had none.
        internal const int CheckpointFileVersion = 2;

        public unsafe bool TryExportRetainedCache(string key, System.IO.Stream destination)
        {
            if (!SupportsPrefixCheckpoints || destination == null || string.IsNullOrEmpty(key)
                || _retainedFusedHolders == null || _isRecurrent == null)
                return false;
            if (!_retainedFusedHolders.TryGetValue(key, out var h) || h.K == null)
                return false;
            WaitIdleKvSync(h);
            // Only host-authoritative state can be written: a checkpoint is never bound.
            // A CUDA checkpoint's K/V rows live on the device; download them first.
            if (h.GdnHostDirty || h.ArenaStateResident || h.FdStateResident || h.Retired)
                return false;
            if (h.KvHostDirty && !TrySyncHolderKvRowsToHost(h))
                return false;
            WriteHolderCheckpoint(h, Math.Max(0, Math.Min(h.CacheSeqLen, h.KvCapacity)), destination);
            return true;
        }

        /// <summary>
        /// Prepare a checkpoint file write that runs off the worker (see
        /// IPrefixCacheModel.TryPrepareBackgroundExport): a device-authoritative holder's
        /// K/V reaches its host bytes through the background idle sync, and the writer
        /// waits for that sync before reading them. Inline, the export downloaded the rows
        /// and serialized ~820 MB on the worker in the middle of the request that
        /// published the shared prefix - 350 ms of an 8K prompt's first token.
        /// </summary>
        public bool TryPrepareBackgroundRetainedExport(string key, out Action<System.IO.Stream> write)
        {
            write = null;
            if (!SupportsPrefixCheckpoints || _backend != BackendType.GgmlCuda || string.IsNullOrEmpty(key)
                || _retainedFusedHolders == null || _isRecurrent == null)
                return false;
            if (!_retainedFusedHolders.TryGetValue(key, out var h) || h.K == null || h.MlxGdn != null
                || h.ExportTask != null)
                return false;
            if (h.GdnHostDirty || h.ArenaStateResident || h.FdStateResident || h.Retired)
                return false;
            var pending = h.IdleKvSync;
            if (pending == null && h.KvHostDirty)
            {
                if (!s_idleKvSyncEnabled)
                    return false;
                StartIdleKvSync(h);
                pending = h.IdleKvSync;
                if (pending == null)
                    return false;
            }
            int rows = Math.Max(0, Math.Min(h.CacheSeqLen, h.KvCapacity));
            write = stream =>
            {
                if (pending != null && !pending.GetAwaiter().GetResult())
                    throw new InvalidOperationException("The checkpoint's K/V rows could not be downloaded for export.");
                WriteHolderCheckpoint(h, rows, stream);
            };
            return true;
        }

        public void NoteBackgroundRetainedExport(string key, System.Threading.Tasks.Task task)
        {
            if (_retainedFusedHolders != null && !string.IsNullOrEmpty(key)
                && _retainedFusedHolders.TryGetValue(key, out var h))
                h.ExportTask = task;
        }

        /// <summary>Serialize a host-authoritative holder's first <paramref name="rows"/>
        /// rows and its recurrent state. Reads host bytes only, so it may run on another
        /// thread while the holder is pinned (ExportTask).</summary>
        private unsafe void WriteHolderCheckpoint(Qwen35KvCacheHolder h, int rows, System.IO.Stream destination)
        {
            int numLayers = h.K.Length;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            var w = new System.IO.BinaryWriter(destination, System.Text.Encoding.UTF8, leaveOpen: true);
            w.Write(CheckpointFileMagic);
            w.Write(CheckpointFileVersion);
            w.Write(CheckpointFingerprint);
            w.Write(numLayers);
            w.Write(rows);
            w.Write(h.RopeDelta);
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
                    if (h.K[l].Storage is MlxStorage)
                    {
                        WriteMlxCacheRows(w, h.K[l], rows);
                        WriteMlxCacheRows(w, h.V[l], rows);
                    }
                    else
                    {
                        WriteCacheRows(w, h.K[l], rows);
                        WriteCacheRows(w, h.V[l], rows);
                    }
                    continue;
                }
                if (h.MlxGdn?[l] != null)
                {
                    WriteMlxGdnState(w, h.MlxGdn[l]);
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
            if (!string.Equals(r.ReadString(), CheckpointFingerprint, StringComparison.Ordinal))
                return false;
            int numLayers = r.ReadInt32();
            int rows = r.ReadInt32();
            int ropeDelta = r.ReadInt32();
            if (numLayers != _kvCacheK.Length || rows < 0 || rows > _maxContextLength)
                return false;
            // A position never precedes zero: rows + delta is the next position.
            if ((long)rows + ropeDelta < 0)
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
                        if (h.K[l].Storage is MlxStorage)
                        {
                            if (!ReadMlxCacheRows(r, h.K[l], rows) || !ReadMlxCacheRows(r, h.V[l], rows))
                                return false;
                            continue;
                        }
                        if (!ReadCacheRows(r, h.K[l], rows) || !ReadCacheRows(r, h.V[l], rows))
                            return false;
                        InvalidateTensorDeviceCache(h.K[l]);
                        InvalidateTensorDeviceCache(h.V[l]);
                        continue;
                    }
                    if (h.MlxGdn?[l] != null)
                    {
                        if (!ReadMlxGdnState(r, h.MlxGdn[l]))
                            return false;
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
                h.RopeDelta = ropeDelta;
                // Version 2 persists trunk state, not the draft cache origin.
                // New speculative executions seed/restart the head after reuse.
                h.MtpCacheStart = rows;
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
            byte* src = (byte*)t.Storage.PtrAtElementForRead(0);   // read only
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

        // A checkpoint file names the backend's state layout as well as the model's K/V
        // identity: MLX writes its own GDN caches (conv tail oldest-first, write index 0)
        // and ggml its host ring and in-place delta layout, and neither reads the other's.
        private string CheckpointFingerprint
            => (KVStateFingerprint ?? string.Empty) + (_backend == BackendType.Mlx ? "|mlx" : string.Empty);

        /// <summary>Bytes of one holder's MLX GDN caches once they have run: per recurrent
        /// layer the conv tail [convKernel - 1, qkvDim] and the delta state [Hv, Dv, Dk].</summary>
        private long MlxGdnStateBytesPerHolder()
        {
            if (_isRecurrent == null)
                return 0;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            long perLayer = ((long)Math.Max(0, _convKernel - 1) * qkvDim + (long)_numVHeads * _headVDim * _headKDim) * sizeof(float);
            long layers = 0;
            foreach (bool recurrent in _isRecurrent)
                if (recurrent) layers++;
            return layers * perLayer;
        }

        /// <summary><see cref="WriteCacheRows"/> for an MLX cache: the same bytes, read
        /// through an MLX view so the storage does not gain a host copy.</summary>
        private static unsafe void WriteMlxCacheRows(System.IO.BinaryWriter w, Tensor t, int rows)
        {
            long heads = t.Sizes[0];
            long cap = t.Sizes[1];
            long rowBytes = heads * cap == 0 ? 0 : t.Storage.ByteLength / (heads * cap);
            w.Write((int)heads);
            w.Write(rowBytes);
            if (rows == 0 || heads == 0 || rowBytes == 0)
                return;
            byte[] bytes = new byte[checked(heads * rows * rowBytes)];
            using (Tensor box = t.Narrow(1, 0, rows))
            fixed (byte* p = bytes)
            {
                if (!MlxFusedOps.TryCopyViewToHost(box, (IntPtr)p, bytes.Length))
                    throw new InvalidOperationException("MLX K/V rows could not be read for the checkpoint.");
            }
            w.Flush();
            w.BaseStream.Write(bytes);
        }

        private static unsafe bool ReadMlxCacheRows(System.IO.BinaryReader r, Tensor t, int rows)
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
            byte[] bytes = new byte[checked(heads * rows * rowBytes)];
            r.BaseStream.ReadExactly(bytes);
            using Tensor box = t.Narrow(1, 0, rows);
            fixed (byte* p = bytes)
                return MlxFusedOps.TryWriteViewFromHost(box, (IntPtr)p, bytes.Length);
        }

        /// <summary>One recurrent layer's MLX GDN state in the file's per-layer layout:
        /// conv length and values, write index (0: the tail is stored oldest-first), delta
        /// byte count and values.</summary>
        private unsafe void WriteMlxGdnState(System.IO.BinaryWriter w, MlxFusedOps.GatedDeltaNetCache cache)
        {
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            float[] conv = new float[Math.Max(0, _convKernel - 1) * qkvDim];
            float[] delta = new float[checked(_numVHeads * _headVDim * _headKDim)];
            fixed (float* c = conv)
            fixed (float* d = delta)
                cache.ExportState(c, conv.Length, d, delta.Length);
            w.Write(conv.Length);
            w.Flush();
            w.BaseStream.Write(MemoryMarshal.AsBytes(conv.AsSpan()));
            w.Write(0);
            w.Write((long)delta.Length * sizeof(float));
            w.Flush();
            w.BaseStream.Write(MemoryMarshal.AsBytes(delta.AsSpan()));
        }

        private unsafe bool ReadMlxGdnState(System.IO.BinaryReader r, MlxFusedOps.GatedDeltaNetCache cache)
        {
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convTail = Math.Max(0, _convKernel - 1);
            if (r.ReadInt32() != convTail * qkvDim)
                return false;
            float[] conv = new float[convTail * qkvDim];
            r.BaseStream.ReadExactly(MemoryMarshal.AsBytes(conv.AsSpan()));
            if (r.ReadInt32() != 0)
                return false;
            float[] delta = new float[checked(_numVHeads * _headVDim * _headKDim)];
            if (r.ReadInt64() != (long)delta.Length * sizeof(float))
                return false;
            r.BaseStream.ReadExactly(MemoryMarshal.AsBytes(delta.AsSpan()));
            fixed (float* c = conv)
            fixed (float* d = delta)
                cache.ImportState(c, convTail, qkvDim, d, _numVHeads, _headVDim, _headKDim);
            return true;
        }

        /// <summary>An independent copy of <paramref name="source"/>, whose host bytes
        /// must be current: fresh tensors sized to the source, every attention K/V
        /// storage and every recurrent layer's conv ring, write index and delta state
        /// copied, and every residency flag off so the next fused decode re-seeds its
        /// device state from the copied host state and builds its own graphs.</summary>
        private unsafe Qwen35KvCacheHolder DeepCopyHolder(Qwen35KvCacheHolder source, int requiredContextTokens = 0,
            List<(Tensor Source, Tensor Destination)> deferredAttention = null)
        {
            // Sized to what the source HOLDS, not to what it reserved. The first
            // request's primary cache was reserved for its whole generation budget
            // (PrepareForPrefill), and a checkpoint is kept for the life of the model
            // while every new chat gets a clone of it; copying and pinning that whole
            // reservation each time is hundreds of megabytes a phone does not have.
            // A checkpoint stays compact. A live clone can reserve the next request's
            // fitted prompt/generation capacity here, so its first prefill does not
            // allocate again and copy the same prefix a second time.
            int rows = Math.Max(0, Math.Min(source.CacheSeqLen, source.KvCapacity));
            int cap = CloneCapacity(source, requiredContextTokens);
            // The CUDA host checkpoint used to fill its entire allocation with
            // zero and immediately overwrite almost every byte. Copy the valid
            // rows once and clear only the padding the attention kernels may read.
            bool overwriteAttention = _backend == BackendType.GgmlCuda;
            var dst = requiredContextTokens > 0 ? TakePooledCloneHolder(cap) : null;
            bool pooled = dst != null;
            dst ??= AllocateHolderStorage(Math.Max(cap, 1), overwriteAttention,
                overwriteState: source.MlxGdn == null && SourceHasEveryDeltaState(source));
            bool complete = false;
            try
            {
                if (pooled) PreparePooledHolderForClone(dst);
                int numLayers = Math.Min(source.K.Length, dst.K.Length);
                for (int l = 0; l < numLayers; l++)
                {
                    bool recurrent = l < _isRecurrent.Length && _isRecurrent[l];
                    if (!recurrent && deferredAttention != null && source.K[l] != null && dst.K[l] != null
                        && source.V[l] != null && dst.V[l] != null
                        && HasKvDeviceCopy(source.K[l]) && HasKvDeviceCopy(source.V[l]))
                    {
                        // The caller fills rows [0, rows) from the source's device copies;
                        // only the padding the attention kernels may read is cleared here.
                        // A layer without device copies (an unbound auxiliary layer) has
                        // authoritative host bytes and takes the host copy below.
                        deferredAttention.Add((source.K[l], dst.K[l]));
                        deferredAttention.Add((source.V[l], dst.V[l]));
                        ClearCheckpointPadding(dst.K[l], rows);
                        ClearCheckpointPadding(dst.V[l], rows);
                        InvalidateTensorDeviceCache(dst.K[l]);
                        InvalidateTensorDeviceCache(dst.V[l]);
                        continue;
                    }
                    if (!recurrent)
                    {
                        if (source.K[l] != null && dst.K[l] != null)
                        {
                            CopyCacheRows(source.K[l], dst.K[l], rows);
                            if (overwriteAttention) ClearCheckpointPadding(dst.K[l], rows);
                            InvalidateTensorDeviceCache(dst.K[l]);
                        }
                        else if (overwriteAttention && dst.K[l] != null) InitializeCacheTensor(dst.K[l]);
                        if (source.V[l] != null && dst.V[l] != null)
                        {
                            CopyCacheRows(source.V[l], dst.V[l], rows);
                            if (overwriteAttention) ClearCheckpointPadding(dst.V[l], rows);
                            InvalidateTensorDeviceCache(dst.V[l]);
                        }
                        else if (overwriteAttention && dst.V[l] != null) InitializeCacheTensor(dst.V[l]);
                        continue;
                    }

                    if (source.MlxGdn?[l] != null && dst.MlxGdn?[l] != null)
                    {
                        // Shares the source's immutable state arrays; see CloneState. On
                        // MLX these caches are the whole recurrent state: the host ring
                        // and delta tensors below are never written, and reading them
                        // would give both storages a host copy for nothing.
                        dst.MlxGdn[l].Dispose();
                        dst.MlxGdn[l] = source.MlxGdn[l].CloneState();
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
                        HostMemory.Copy((void*)GdnDeltaStatePointer(from), (void*)GdnDeltaStatePointer(to), bytes);
                        if (IsGgmlBackend)
                            InvalidateGdnDeltaStateDeviceCaches(to);
                    }
                }
                if (dst.MlxGdn != null)
                    MaterializeMlxHolder(dst);
                dst.CacheSeqLen = source.CacheSeqLen;
                dst.RopeDelta = source.RopeDelta;
                dst.MtpCacheStart = source.MtpCacheStart;
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

        // ---- Device-resident holder copies (CUDA) ----
        //
        // A host deep copy of a long-context holder downloads the committed K/V rows,
        // copies them on the host and uploads them again at the copy's first bind:
        // gigabytes and around a second per checkpoint or clone at 20K+ tokens. On a
        // single CUDA device the rows are copied between the host keys' device copies
        // instead. The copy's K/V is then device-authoritative (KvHostDirty) and its
        // host bytes are not written; every host reader already syncs a dirty holder
        // first. Recurrent state (a few percent of the bytes) keeps the host path.

        // Device copies are keyed by GGML host storage; any other allocator has none to copy.
        private bool DeviceHolderCopiesEnabled =>
            _backend == BackendType.GgmlCuda && !IsTensorParallel && _allocator is GgmlAllocator
            && !string.Equals(Environment.GetEnvironmentVariable("TS_QWEN35_DEVICE_HOLDER_COPY"), "0", StringComparison.Ordinal);

        /// <summary>True when every main-stack attention K/V tensor of <paramref name="holder"/>
        /// has a resident device copy that a device-to-device clone can read. Auxiliary
        /// (NextN) layers that no graph has bound keep their host bytes and are copied
        /// on the host.</summary>
        private bool CanCopyHolderKvOnDevice(Qwen35KvCacheHolder holder)
        {
            if (!DeviceHolderCopiesEnabled || holder?.K == null || holder.V == null || _isRecurrent == null)
                return false;
            bool any = false;
            for (int l = 0; l < holder.K.Length && l < Config.NumLayers; l++)
                foreach (Tensor tensor in new[] { holder.K[l], holder.V[l] })
                {
                    if (tensor == null) continue;
                    if (!HasKvDeviceCopy(tensor))
                    {
                        if (_deviceHolderCopyDeclines < 4)
                        {
                            _deviceHolderCopyDeclines++;
                            Console.Error.WriteLine($"[qwen35] device holder copy declined: layer {l} K/V {tensor.ElementType} " +
                                $"[{string.Join(",", tensor.Sizes.ToArray())}] has no resident device copy (hostDirty={holder.KvHostDirty}).");
                        }
                        return false;
                    }
                    any = true;
                }
            return any;
        }

        private long _deviceUseClock;

        /// <summary>Free device memory a retained (idle) holder's device K/V copies must leave
        /// for activations, graphs and new requests. Env TS_QWEN35_DEVICE_RETAIN_RESERVE_MB.</summary>
        private static long DeviceRetainReserveBytes(long totalBytes)
        {
            string raw = Environment.GetEnvironmentVariable("TS_QWEN35_DEVICE_RETAIN_RESERVE_MB");
            if (long.TryParse(raw, out long mb) && mb >= 0)
                return mb * 1024 * 1024;
            // Covers the verify graph cache (1.5 GB budget), a 4K-token prefill
            // chunk's packed buffer (~1 GB) and a vision encode; 2 GiB measured a
            // packed-buffer OOM mid-request on a 32 GB card.
            return Math.Max(4L * 1024 * 1024 * 1024, totalBytes / 8);
        }

        /// <summary>Attention K/V bytes of <paramref name="holder"/>'s layout at <paramref name="capacity"/> rows.</summary>
        private static long HolderKvBytes(Qwen35KvCacheHolder holder, int capacity)
        {
            long bytes = 0;
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null && t.Sizes[1] > 0)
                            bytes += t.Storage.ByteLength / t.Sizes[1] * capacity;
            return bytes;
        }

        private void TouchDeviceRetained(Qwen35KvCacheHolder holder)
        {
            if (holder != null) holder.DeviceUseStamp = ++_deviceUseClock;
        }

        /// <summary>Make room for <paramref name="neededBytes"/> of new device copies by demoting
        /// the least recently used retained holders' K/V to host memory. True when the free
        /// memory left afterwards covers the retain reserve. Without
        /// <paramref name="requireReserve"/> the caller proceeds either way.</summary>
        private bool EnsureDeviceRetainHeadroom(long neededBytes, bool requireReserve, Qwen35KvCacheHolder exclude = null)
        {
            if (!DeviceHolderCopiesEnabled)
                return false;
            for (int attempt = 0; attempt < 32; attempt++)
            {
                if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) || total <= 0)
                    return !requireReserve;
                if (free - neededBytes >= DeviceRetainReserveBytes(total))
                    return true;
                // Least recently used first, but a holder whose K/V already reached host
                // memory (a finished idle sync) demotes for free, so it goes first. Above
                // both, a holder no clone has used yet goes before one that has served
                // clones: with four Qwen3.8 agent lanes every finished request retained a
                // 25K-token end state that nothing reused, and by recency alone those
                // pushed each lane's shared-prompt checkpoint - cloned on every turn - off
                // the device, so most clones took the ~155 ms host route.
                Qwen35KvCacheHolder victim = null;
                int victimRank = int.MaxValue;
                if (_retainedFusedHolders != null)
                    foreach (var holder in _retainedFusedHolders.Values)
                    {
                        if (ReferenceEquals(holder, exclude) || holder.Retired || holder.ArenaStateResident
                            || !holder.DeviceMirrored || !CanCopyHolderKvOnDevice(holder))
                            continue;
                        int rank = DeviceDemotionRank(holder.CloneCount,
                            !holder.KvHostDirty || holder.IdleKvSync is { IsCompletedSuccessfully: true, Result: true },
                            holder.IdleKvSync != null);
                        if (victim == null || rank < victimRank
                            || (rank == victimRank && holder.DeviceUseStamp < victim.DeviceUseStamp))
                        {
                            victim = holder;
                            victimRank = rank;
                        }
                    }
                if (victim == null)
                    return false;
                long demoteStart = PrefixReleaseDebug ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                if (!DemoteRetainedHolderKv(victim))
                    return false;
                if (PrefixReleaseDebug)
                    Console.Error.WriteLine($"[prefix-demote] free={free >> 20}MiB need={neededBytes >> 20}MiB " +
                        $"reserve={DeviceRetainReserveBytes(total) >> 20}MiB victim={victim.CacheSeqLen}tok " +
                        $"{System.Diagnostics.Stopwatch.GetElapsedTime(demoteStart).TotalMilliseconds:F0}ms");
            }
            return false;
        }

        /// <summary>Order in which retained holders give up their device K/V copies under
        /// pressure, lowest first (least recent first within a rank). Holders no clone has
        /// used come before holders that have served clones; within each, a holder whose
        /// K/V already reached host memory demotes for free, then one whose background sync
        /// is running, then one that still needs a download.</summary>
        internal static int DeviceDemotionRank(int cloneCount, bool kvOnHost, bool hostSyncRunning)
            => (cloneCount > 0 ? 3 : 0) + (kvOnHost ? 0 : hostSyncRunning ? 1 : 2);

        // ---- Background host syncs of idle retained holders (CUDA) ----

        private System.Threading.Tasks.Task _idleKvSyncChain = System.Threading.Tasks.Task.CompletedTask;
        private static readonly bool s_idleKvSyncEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_IDLE_KV_SYNC") != "0";

        /// <summary>
        /// Demoting a device-authoritative holder downloads its K/V on the request's
        /// critical path (~150 ms per GB). A retained holder never changes, so download
        /// each one once, early: on a background thread and stream that overlap the
        /// model's own work, oldest first, one at a time. A later demotion of a synced
        /// holder only frees its device copies. Device copies are never freed here, so
        /// nothing changes for a holder that is reused before memory runs short. Sizing
        /// the queue to a predicted deficit left concurrent requests demoting holders it
        /// had not reached yet.
        /// </summary>
        private void ScheduleIdleKvSyncs()
        {
            if (!s_idleKvSyncEnabled || !DeviceHolderCopiesEnabled || _retainedFusedHolders == null)
                return;
            var candidates = new List<Qwen35KvCacheHolder>();
            foreach (var h in _retainedFusedHolders.Values)
                if (!h.Retired && !h.DisposalStarted && !h.ArenaStateResident && h.DeviceMirrored
                    && h.KvHostDirty && h.IdleKvSync == null && CanCopyHolderKvOnDevice(h))
                    candidates.Add(h);
            candidates.Sort((a, b) => a.DeviceUseStamp.CompareTo(b.DeviceUseStamp));
            foreach (var h in candidates)
                StartIdleKvSync(h);
        }

        private void StartIdleKvSync(Qwen35KvCacheHolder holder)
        {
            int rows = Math.Max(0, Math.Min(holder.CacheSeqLen, holder.KvCapacity));
            var tensors = new List<Tensor>();
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                foreach (Tensor tensor in set)
                    if (tensor != null && HasKvDeviceCopy(tensor)) tensors.Add(tensor);
            if (tensors.Count == 0)
                return;
            // Captured on this thread: the background job touches only these tensors'
            // bytes, and every owner of the holder waits for it before using them.
            var task = _idleKvSyncChain.ContinueWith(_ => SyncIdleHolderKv(tensors, rows),
                System.Threading.CancellationToken.None, System.Threading.Tasks.TaskContinuationOptions.None,
                System.Threading.Tasks.TaskScheduler.Default);
            holder.IdleKvSync = task;
            _idleKvSyncChain = task;
        }

        private static bool SyncIdleHolderKv(List<Tensor> tensors, int rows)
        {
            var planes = new (IntPtr, long)[tensors.Count];
            for (int i = 0; i < tensors.Count; i++)
                planes[i] = (TensorComputePrimitives.GetStorageBasePointer(tensors[i]), tensors[i].Storage.ByteLength);
            HostMemory.Clear(planes);   // padding reads as zero, and the pages are faulted in
            foreach (Tensor tensor in tensors)
            {
                int heads = checked((int)tensor.Sizes[0]);
                long capacity = tensor.Sizes[1];
                long rowBytes = tensor.Storage.ByteLength / checked(heads * capacity);
                var offsets = new long[heads];
                var lengths = new long[heads];
                for (int h = 0; h < heads; h++)
                {
                    offsets[h] = checked(h * capacity * rowBytes);
                    lengths[h] = checked(rows * rowBytes);
                }
                if (GgmlBasicOps.SyncIdleHostBufferRanges(TensorComputePrimitives.GetStorageBasePointer(tensor), offsets, lengths) != 1)
                    return false;
            }
            return true;
        }

        /// <summary>Finish <paramref name="holder"/>'s background K/V sync, if any, before its
        /// bytes are read, freed or bound. A completed sync leaves the host bytes current.</summary>
        private void WaitIdleKvSync(Qwen35KvCacheHolder holder,
            [System.Runtime.CompilerServices.CallerMemberName] string caller = "")
        {
            var task = holder?.IdleKvSync;
            if (task == null)
                return;
            holder.IdleKvSync = null;
            long started = PrefixReleaseDebug && !task.IsCompleted ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            bool synced;
            try { synced = task.GetAwaiter().GetResult(); }
            catch (Exception) { synced = false; }
            if (started != 0)
                Console.Error.WriteLine($"[prefix-idle-wait] {caller} waited " +
                    $"{System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds:F1} ms for a background K/V sync");
            bool capture = holder.IdleKvSyncIsCapture;
            holder.IdleKvSyncIsCapture = false;
            if (synced)
                holder.KvHostDirty = false;
            else if (capture)
                throw new InvalidOperationException(
                    "A background prefix-checkpoint capture failed to download its K/V rows; the checkpoint is unusable.");
        }

        // Background checkpoint captures still reading a source holder's device K/V
        // copies, with the host keys they read. Anything that frees or replaces one of
        // those copies waits for the captures reading it first.
        // Created on first use: tests build this model without running its constructor.
        private List<(HashSet<IntPtr> Sources, System.Threading.Tasks.Task<bool> Task)> _captureReads;

        private void WaitCaptureReads(Tensor[] k, Tensor[] v)
        {
            if (_captureReads == null || _captureReads.Count == 0)
                return;
            for (int i = _captureReads.Count - 1; i >= 0; i--)
            {
                var (sources, task) = _captureReads[i];
                if (!task.IsCompleted && !ReadsAny(sources, k) && !ReadsAny(sources, v))
                    continue;
                try { task.Wait(); }
                catch (Exception) { /* the checkpoint reports it when it is used */ }
                _captureReads.RemoveAt(i);
            }

            static bool ReadsAny(HashSet<IntPtr> sources, Tensor[] tensors)
            {
                if (tensors == null)
                    return false;
                foreach (Tensor t in tensors)
                    if (t != null && sources.Contains(TensorComputePrimitives.GetStorageBasePointer(t)))
                        return true;
                return false;
            }
        }

        /// <summary>Finish a background checkpoint file write reading <paramref name="holder"/>'s
        /// host bytes before they are freed or rewritten.</summary>
        private static void WaitExport(Qwen35KvCacheHolder holder)
        {
            var task = holder?.ExportTask;
            if (task == null)
                return;
            holder.ExportTask = null;
            try { task.Wait(); }
            catch (Exception) { /* the store reports a failed save itself */ }
        }

        private void WaitAllCaptureReads()
        {
            if (_captureReads == null)
                return;
            foreach (var (_, task) in _captureReads)
            {
                try { task.Wait(); }
                catch (Exception) { }
            }
            _captureReads.Clear();
        }

        private static readonly bool s_backgroundCapture =
            Environment.GetEnvironmentVariable("TS_QWEN35_BACKGROUND_CAPTURE") != "0";

        /// <summary>
        /// Fill a host checkpoint's K/V from the active holder's device copies on a
        /// background stream (see TSGgml_DownloadDeviceCopyRanges). Synchronously,
        /// the host path downloaded the committed rows into the ACTIVE holder's host
        /// mirror and then copied them a second time into the checkpoint: 600 ms of a
        /// four-request step for 25K-token Qwen3.8-27B holders, every turn, while the
        /// next request waited. The checkpoint's consumers wait for the copy
        /// (WaitIdleKvSync); whatever frees the active copies waits too (WaitCaptureReads).
        /// </summary>
        private void StartCaptureCopy(Qwen35KvCacheHolder copy, List<(Tensor Source, Tensor Destination)> pairs, int rows)
        {
            var sources = new HashSet<IntPtr>();
            var jobs = new List<(IntPtr Source, IntPtr Destination, long[] SourceOffsets, long[] DestinationOffsets, long[] Lengths)>(pairs.Count);
            foreach (var (src, dst) in pairs)
            {
                int heads = checked((int)src.Sizes[0]);
                long srcCap = src.Sizes[1];
                long dstCap = dst.Sizes[1];
                long rowBytes = src.Storage.ByteLength / checked(heads * srcCap);
                var so = new long[heads];
                var dO = new long[heads];
                var len = new long[heads];
                for (int h = 0; h < heads; h++)
                {
                    so[h] = checked(h * srcCap * rowBytes);
                    dO[h] = checked(h * dstCap * rowBytes);
                    len[h] = checked(rows * rowBytes);
                }
                IntPtr srcKey = TensorComputePrimitives.GetStorageBasePointer(src);
                sources.Add(srcKey);
                jobs.Add((srcKey, TensorComputePrimitives.GetStorageBasePointer(dst), so, dO, len));
            }
            var task = (_idleKvSyncChain ?? System.Threading.Tasks.Task.CompletedTask).ContinueWith(_ =>
            {
                foreach (var job in jobs)
                    if (GgmlBasicOps.DownloadDeviceCopyRanges(job.Source, job.Destination,
                            job.SourceOffsets, job.DestinationOffsets, job.Lengths) != 1)
                        return false;
                return true;
            }, System.Threading.CancellationToken.None, System.Threading.Tasks.TaskContinuationOptions.None,
                System.Threading.Tasks.TaskScheduler.Default);
            _idleKvSyncChain = task;
            copy.IdleKvSync = task;
            copy.IdleKvSyncIsCapture = true;
            (_captureReads ??= new()).Add((sources, task));
        }

        /// <summary>True when every main-stack attention K/V tensor of the active cache has a
        /// resident device copy holding rows its host mirror lacks, so a checkpoint can take
        /// them from the device in the background. An auxiliary (NextN) layer no graph has
        /// bound keeps authoritative host bytes and is copied on the host (DeepCopyHolder):
        /// requiring a device copy of it declined every clone of a checkpoint - which copies
        /// that layer on the host - and sent four-lane Qwen3.8 agent turns through the
        /// synchronous host capture, ~270-500 ms each.</summary>
        private bool CanCaptureCheckpointInBackground()
        {
            if (!s_backgroundCapture || _backend != BackendType.GgmlCuda || !_kvCacheHostDirty
                || _kvCacheK == null || IsTensorParallel)
                return false;
            for (int l = 0; l < _kvCacheK.Length && l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l]) continue;
                if (_kvCacheK[l] == null || _kvCacheV[l] == null
                    || !HasKvDeviceCopy(_kvCacheK[l]) || !HasKvDeviceCopy(_kvCacheV[l]))
                    return false;
            }
            return true;
        }

        /// <summary>Move an idle retained holder's K/V to host memory: download device-
        /// authoritative rows, then drop the device copies (which also retires any graph
        /// still bound to them). Its recurrent state is untouched.</summary>
        private bool DemoteRetainedHolderKv(Qwen35KvCacheHolder holder)
        {
            WaitIdleKvSync(holder);
            WaitCaptureReads(holder.K, holder.V);
            if (holder.KvHostDirty && !TrySyncHolderKvRowsToHost(holder))
                return false;
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                foreach (Tensor tensor in set)
                    if (tensor != null)
                        InvalidateTensorDeviceCache(tensor);
            holder.DeviceMirrored = holder.FdStateResident || holder.GdnHostDirty;
            CountDecodeGraphReset();
            return true;
        }

        /// <summary>True when every recurrent layer of <paramref name="source"/> has a delta state
        /// to copy, so a copy overwrites every destination state it allocates.</summary>
        private bool SourceHasEveryDeltaState(Qwen35KvCacheHolder source)
        {
            if (source.DeltaState == null || _isRecurrent == null)
                return false;
            for (int l = 0; l < _isRecurrent.Length; l++)
                if (_isRecurrent[l] && (l >= source.DeltaState.Length || source.DeltaState[l] == null))
                    return false;
            return true;
        }

        private static bool HasKvDeviceCopy(Tensor tensor)
            => GgmlBasicOps.HasResidentDeviceCopy(TensorComputePrimitives.GetStorageBasePointer(tensor), tensor.Storage.ByteLength);

        /// <summary>Copy one attention tensor's rows into a holder being built on the device
        /// path: between device copies when the source has one, from the source's host rows
        /// straight into the destination's device copy when <paramref name="uploadHostRows"/>
        /// (a host-resident checkpoint), otherwise on the host (an unbound auxiliary layer),
        /// dropping any stale destination mirror.</summary>
        private bool CopyHolderKvTensor(Tensor source, Tensor destination, int rows, bool uploadHostRows = false)
        {
            if (HasKvDeviceCopy(source))
                return CopyCacheRowsOnDevice(source, destination, rows);
            if (uploadHostRows)
                return UploadCacheRowsToDevice(source, destination, rows);
            CopyCacheRows(source, destination, rows);
            ClearCheckpointPadding(destination, rows);
            InvalidateTensorDeviceCache(destination);
            return true;
        }

        private int _deviceHolderCopyDeclines;

        /// <summary>Give a host-authoritative retained holder (a checkpoint imported from
        /// disk or captured on the host path) device copies of its K/V once, so it and
        /// every later clone of it copy on the device.</summary>
        private bool TryUploadHolderKvForDeviceCopies(Qwen35KvCacheHolder holder)
        {
            if (!DeviceHolderCopiesEnabled || holder?.K == null || holder.KvHostDirty || holder.ArenaStateResident)
                return false;
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                foreach (Tensor tensor in set)
                {
                    if (tensor == null) continue;
                    IntPtr key = TensorComputePrimitives.GetStorageBasePointer(tensor);
                    if (GgmlBasicOps.HasResidentDeviceCopy(key, tensor.Storage.ByteLength)) continue;
                    tensor.Storage.EnsureHostReadable();
                    if (!GgmlBasicOps.EnsureDeviceCopyFromHost(key, tensor.Storage.ByteLength))
                        return false;
                }
            holder.DeviceMirrored = true;
            return CanCopyHolderKvOnDevice(holder);
        }

        /// <summary>Download a device-authoritative holder's committed K/V rows and zero its
        /// host padding, making its host bytes current again. The device copies stay.</summary>
        private bool TrySyncHolderKvRowsToHost(Qwen35KvCacheHolder holder)
        {
            WaitIdleKvSync(holder);
            WaitExport(holder);
            if (!CanCopyHolderKvOnDevice(holder))
                return false;
            int rows = Math.Max(0, Math.Min(holder.CacheSeqLen, holder.KvCapacity));
            var synced = new List<Tensor>();
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                foreach (Tensor tensor in set)
                    if (tensor != null && HasKvDeviceCopy(tensor)) synced.Add(tensor);
            // A device-authoritative holder's host planes were never written: zero them
            // (rows and padding) in one parallel pass so the downloads below land in
            // faulted pages instead of faulting 4 KiB at a time on the driver's copy.
            var planes = new (IntPtr, long)[synced.Count];
            for (int i = 0; i < synced.Count; i++)
            {
                synced[i].Storage.EnsureHostReadable();
                planes[i] = (TensorComputePrimitives.GetStorageBasePointer(synced[i]), synced[i].Storage.ByteLength);
            }
            long clearStart = System.Diagnostics.Stopwatch.GetTimestamp();
            HostMemory.Clear(planes);
            double clearMs = System.Diagnostics.Stopwatch.GetElapsedTime(clearStart).TotalMilliseconds;
            long syncStart = System.Diagnostics.Stopwatch.GetTimestamp();
            long syncedBytes = 0;
            foreach (Tensor tensor in synced)
                {
                    int heads = checked((int)tensor.Sizes[0]);
                    long capacity = tensor.Sizes[1];
                    long rowBytes = tensor.Storage.ByteLength / checked(heads * capacity);
                    var offsets = new long[heads];
                    var lengths = new long[heads];
                    for (int h = 0; h < heads; h++)
                    {
                        offsets[h] = checked(h * capacity * rowBytes);
                        lengths[h] = checked(rows * rowBytes);
                    }
                    GgmlBasicOps.SyncHostBufferRanges(TensorComputePrimitives.GetStorageBasePointer(tensor), offsets, lengths);
                    syncedBytes += heads * rows * rowBytes;
                }
            if (PrefixReleaseDebug)
                Console.Error.WriteLine($"[prefix-sync] tensors={synced.Count} clear={clearMs:F0}ms " +
                    $"sync={System.Diagnostics.Stopwatch.GetElapsedTime(syncStart).TotalMilliseconds:F0}ms bytes={syncedBytes >> 20}MiB");
            holder.KvHostDirty = false;
            return true;
        }

        /// <summary>Copy rows [0, <paramref name="rows"/>) of every head plane between two
        /// K/V tensors' device copies (capacities may differ), zeroing the destination's
        /// other rows. The destination copy is created without an upload when absent.</summary>
        private static bool CopyCacheRowsOnDevice(Tensor source, Tensor destination, int rows)
        {
            long heads = source.Sizes[0];
            if (destination.Sizes[0] != heads || destination.ElementType != source.ElementType
                || destination.Sizes[2] != source.Sizes[2])
                return false;
            long sourceCapacity = source.Sizes[1], destinationCapacity = destination.Sizes[1];
            if (rows > sourceCapacity || rows > destinationCapacity)
                return false;
            long rowBytes = source.Storage.ByteLength / checked(heads * sourceCapacity);
            if (rowBytes * heads * destinationCapacity != destination.Storage.ByteLength)
                return false;
            var sourceOffsets = new long[heads];
            var destinationOffsets = new long[heads];
            var lengths = new long[heads];
            for (long h = 0; h < heads; h++)
            {
                sourceOffsets[h] = checked(h * sourceCapacity * rowBytes);
                destinationOffsets[h] = checked(h * destinationCapacity * rowBytes);
                lengths[h] = checked(rows * rowBytes);
            }
            return GgmlBasicOps.CopyDeviceCopyRanges(
                TensorComputePrimitives.GetStorageBasePointer(source), source.Storage.ByteLength,
                TensorComputePrimitives.GetStorageBasePointer(destination), destination.Storage.ByteLength,
                sourceOffsets, destinationOffsets, lengths, zeroDestination: true);
        }

        /// <summary><see cref="CopyCacheRowsOnDevice"/> for a source whose rows are in host
        /// memory only: they are uploaded straight into the destination's device copy.</summary>
        private static bool UploadCacheRowsToDevice(Tensor source, Tensor destination, int rows)
        {
            long heads = source.Sizes[0];
            if (destination.Sizes[0] != heads || destination.ElementType != source.ElementType
                || destination.Sizes[2] != source.Sizes[2])
                return false;
            long sourceCapacity = source.Sizes[1], destinationCapacity = destination.Sizes[1];
            if (rows > sourceCapacity || rows > destinationCapacity)
                return false;
            long rowBytes = source.Storage.ByteLength / checked(heads * sourceCapacity);
            if (rowBytes * heads * destinationCapacity != destination.Storage.ByteLength)
                return false;
            var sourceOffsets = new long[heads];
            var destinationOffsets = new long[heads];
            var lengths = new long[heads];
            for (long h = 0; h < heads; h++)
            {
                sourceOffsets[h] = checked(h * sourceCapacity * rowBytes);
                destinationOffsets[h] = checked(h * destinationCapacity * rowBytes);
                lengths[h] = checked(rows * rowBytes);
            }
            source.Storage.EnsureHostReadable();
            return GgmlBasicOps.UploadToDeviceCopyRanges(
                TensorComputePrimitives.GetStorageBasePointer(source), source.Storage.ByteLength,
                TensorComputePrimitives.GetStorageBasePointer(destination), destination.Storage.ByteLength,
                sourceOffsets, destinationOffsets, lengths, zeroDestination: true);
        }

        /// <summary>The device-copy counterpart of <see cref="DeepCopyHolder"/> for a source whose
        /// K/V device copies are resident and whose recurrent host state is current. With
        /// <paramref name="uploadHostRows"/> the source is a host-resident checkpoint instead,
        /// and its main-stack rows are uploaded straight into the copy's device copies. Returns
        /// null when a device copy fails; the caller then takes the host path.</summary>
        private unsafe Qwen35KvCacheHolder DeepCopyHolderOnDevice(Qwen35KvCacheHolder source, int requiredContextTokens,
            bool uploadHostRows = false, bool liveClone = false, bool recurrentFromLive = false)
        {
            int rows = Math.Max(0, Math.Min(source.CacheSeqLen, source.KvCapacity));
            int cap = CloneCapacity(source, requiredContextTokens);
            // A pooled holder of the same capacity keeps its device copies, so the rows are
            // written in place and graphs keyed by its pointers stay valid.
            var dst = requiredContextTokens > 0 ? TakePooledCloneHolder(cap) : null;
            bool pooled = dst != null;
            dst ??= AllocateHolderStorage(Math.Max(cap, 1), overwriteAttention: true, overwriteState: SourceHasEveryDeltaState(source));
            bool complete = false;
            // A live clone's recurrent state goes straight into its own device copies (the
            // solo decode's conv-scratch slots and delta mirrors) where the verifier can take
            // a state over from them: the clone is bound next, and its first verify then
            // starts on the device. Its host ring and delta bytes are never written, which
            // saves a 151 MB host copy into fresh pages and the upload of that copy. A
            // checkpoint keeps host state: clones and the disk export read it from there.
            bool stateOnDevice = liveClone && DeviceStateHandoffAvailable && dst.ConvScratch != IntPtr.Zero
                && source.ConvState != null && SourceHasEveryDeltaState(source);
            try
            {
                if (pooled) DiscardArenaSlotForHolder(dst);
                int numLayers = Math.Min(source.K.Length, dst.K.Length);
                int recurrentSlot = 0;
                for (int l = 0; l < numLayers; l++)
                {
                    bool recurrent = l < _isRecurrent.Length && _isRecurrent[l];
                    if (!recurrent)
                    {
                        if (source.K[l] == null || dst.K[l] == null || source.V[l] == null || dst.V[l] == null)
                            return null;
                        bool upload = uploadHostRows && l < Config.NumLayers;
                        if (!CopyHolderKvTensor(source.K[l], dst.K[l], rows, upload)
                            || !CopyHolderKvTensor(source.V[l], dst.V[l], rows, upload))
                            return null;
                        continue;
                    }

                    if (stateOnDevice)
                    {
                        if (!UploadRecurrentLayerToDevice(source, dst, l, recurrentSlot++))
                            return null;
                        continue;
                    }
                    if (recurrentFromLive)
                        continue;   // ReadLiveStateInto below writes every recurrent layer

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
                        IntPtr target = GdnDeltaStatePointer(to);
                        HostMemory.Copy((void*)GdnDeltaStatePointer(from), (void*)target, bytes);
                        // A pooled holder's delta mirror is refreshed in place (no-op when it
                        // has none) rather than freed, which would reset every decode graph.
                        if (pooled) GgmlBasicOps.UploadHostBufferRanges(target, new long[] { 0 }, new long[] { bytes });
                        else InvalidateGdnDeltaStateDeviceCaches(to);
                    }
                }
                if (recurrentFromLive && (stateOnDevice || pooled || !ReadLiveStateInto(dst)))
                    return null;
                dst.CacheSeqLen = source.CacheSeqLen;
                dst.RopeDelta = source.RopeDelta;
                dst.MtpCacheStart = source.MtpCacheStart;
                dst.KvHostDirty = true;
                dst.GdnHostDirty = stateOnDevice;
                dst.FdStateResident = stateOnDevice;
                dst.ArenaStateResident = false;
                dst.DeviceMirrored = true;
                dst.Logits = null;
                complete = true;
                return dst;
            }
            finally
            {
                if (!complete) DisposeHolder(dst);
            }
        }

        // TS_QWEN35_CAPTURE_LIVE_STATE=0 drains the live state before every checkpoint, as before.
        private static readonly bool s_captureReadsLiveState =
            Environment.GetEnvironmentVariable("TS_QWEN35_CAPTURE_LIVE_STATE") != "0";

        /// <summary>Copy the verifier's live recurrent state into <paramref name="dst"/>'s host
        /// conv rings and delta bytes, leaving it live on the device. False (nothing
        /// written that the caller keeps) when there is no live state or a size differs.</summary>
        private unsafe bool ReadLiveStateInto(Qwen35KvCacheHolder dst)
        {
            if (!PrepareRecurrentStatePointers() || dst.ConvState == null || dst.DeltaState == null)
                return false;
            int[] layers = _fvRecurrentLayers;
            var deltaOut = new IntPtr[layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                int l = layers[i];
                Tensor to = l < dst.DeltaState.Length ? dst.DeltaState[l] : null;
                if (to == null || l >= dst.ConvState.Length || dst.ConvState[l] == null
                    || GdnDeltaStateBytes(to) != GdnDeltaStateBytes(_deltaStateTensor[l]))
                    return false;
                deltaOut[i] = GdnDeltaStatePointer(to);
            }
            if (!GgmlBasicOps.Qwen35ReadLiveState(_fvSnapConvPtrs, deltaOut, layers.Length, _verifyOwnerId))
                return false;
            // The conv window arrives in the kernel's time-major layout; the ring is
            // channel-major per time step from write index 0 (UnpackRecurrentStateFromNative).
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convDim = _convKernel - 1;
            for (int i = 0; i < layers.Length; i++)
            {
                int l = layers[i];
                float* convSrc = (float*)_fvSnapConvPtrs[i];
                float[] ring = dst.ConvState[l];
                for (int t = 0; t < convDim; t++)
                {
                    int dstBase = t * qkvDim;
                    for (int ch = 0; ch < qkvDim; ch++)
                        ring[dstBase + ch] = convSrc[ch * convDim + t];
                }
                dst.ConvWriteIdx[l] = 0;
                InvalidateGdnDeltaStateDeviceCaches(dst.DeltaState[l]);
            }
            return true;
        }

        /// <summary>Write one recurrent layer of <paramref name="source"/>'s host state into
        /// <paramref name="dst"/>'s device copies under the solo decode's keys: the conv ring
        /// packed into the ggml [time, channel] scratch layout (the decode reseed's transform),
        /// the delta state uploaded from the source's bytes into the destination's key.</summary>
        private unsafe bool UploadRecurrentLayerToDevice(Qwen35KvCacheHolder source, Qwen35KvCacheHolder dst, int layer, int slot)
        {
            Tensor from = source.DeltaState[layer], to = dst.DeltaState[layer];
            float[] ring = source.ConvState[layer];
            if (from == null || to == null || ring == null)
                return false;
            long deltaBytes = GdnDeltaStateBytes(from);
            if (deltaBytes != GdnDeltaStateBytes(to))
                return false;
            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            float* conv = (float*)dst.ConvScratch + (long)slot * convDim * qkvDim;
            int w = source.ConvWriteIdx[layer];
            for (int t = 0; t < convDim; t++)
            {
                int srcBase = ((w + t) % convDim) * qkvDim;
                for (int ch = 0; ch < qkvDim; ch++)
                    conv[ch * convDim + t] = ring[srcBase + ch];
            }
            from.Storage.EnsureHostReadable();
            return GgmlBasicOps.EnsureDeviceCopyFromHost((IntPtr)conv, (long)convDim * qkvDim * sizeof(float))
                && GgmlBasicOps.UploadToDeviceCopyRanges(GdnDeltaStatePointer(from), deltaBytes,
                    GdnDeltaStatePointer(to), deltaBytes, new long[] { 0 }, new long[] { 0 }, new long[] { deltaBytes },
                    zeroDestination: false);
        }

        /// <summary>Keep every unwritten attention row finite without touching the
        /// prefix just copied. All-zero bytes encode zero in F32/F16 and the supported
        /// block-quantized cache formats. Used only for host-backed CUDA checkpoints.</summary>
        internal static unsafe void ClearCheckpointPadding(Tensor tensor, int writtenRows)
        {
            long heads = tensor.Sizes[0], capacity = tensor.Sizes[1];
            if (writtenRows < 0 || writtenRows > capacity)
                throw new ArgumentOutOfRangeException(nameof(writtenRows));
            if (heads == 0 || capacity == 0 || writtenRows == capacity) return;
            long rowBytes = tensor.Storage.ByteLength / checked(heads * capacity);
            tensor.Storage.EnsureHostReadable();
            byte* start = (byte*)TensorComputePrimitives.GetStorageBasePointer(tensor);
            long tailBytes = checked((capacity - writtenRows) * rowBytes);
            for (long h = 0; h < heads; h++)
                HostMemory.Clear(start + checked((h * capacity + writtenRows) * rowBytes), tailBytes);
        }

        /// <summary>
        /// Evaluates an MLX holder's K/V copies and GDN states. They are lazy on MLX, and a
        /// checkpoint kept for the life of the model would otherwise hold the whole graph
        /// that produced them (the copied-from cache, every layer's activations) alive.
        /// </summary>
        private static void MaterializeMlxHolder(Qwen35KvCacheHolder holder)
        {
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null)
                            MlxFusedOps.TryEvaluate(t);
            foreach (var cache in holder.MlxGdn)
                cache?.TryEvaluateState();
        }

        private void DisposeHolder(Qwen35KvCacheHolder holder)
        {
            if (holder == null) return;
            WaitIdleKvSync(holder);
            WaitExport(holder);
            WaitCaptureReads(holder.K, holder.V);
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
            if (holder.MlxGdn != null)
                for (int i = 0; i < holder.MlxGdn.Length; i++)
                {
                    holder.MlxGdn[i]?.Dispose();
                    holder.MlxGdn[i] = null;
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
            WaitAllCaptureReads();
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
