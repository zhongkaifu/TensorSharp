// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Per-request KV-cache holders for the per-sequence fused-decode path.
//
// Problem this solves: with N>=2 concurrent requests the continuous-batching
// engine routed every step through the op-by-op batched paged forward
// (ForwardBatchCore). Each of that path's ~20 Ops.* dispatches per layer does
// a host round-trip + ggml_backend_synchronize on the Metal backend (activation
// tensors bind device-local, so the per-op sync cannot be deferred even with
// async compute). The result: hundreds of Metal queue drains per token, the GPU
// sits idle between dispatches (measured ~30% utilisation), and aggregate
// throughput at N=2 falls BELOW the single-stream rate.
//
// Fix: give each in-flight request its OWN set of KV-cache tensors and switch
// the model between them with a cheap pointer swap (no byte-level extract/inject,
// no sliding-window-cache wrap corruption). The engine then runs each scheduled
// sequence through the proven single-graph fused Forward (NativeGemma4ModelDecode
// for decode), which keeps the GPU saturated — one fused decode graph per token
// per sequence instead of ~840 tiny serialized dispatches for the whole batch.
//
// The single-request (N==1) path is untouched: it keeps using the model's
// primary cache and the engine's live-cache continuation / prefix-cache reuse.
// RestorePrimaryCache() reinstates the primary cache before any N==1 step that
// follows a multi-sequence (fused) episode.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Gemma4Model
    {
        private sealed class Gemma4KvCacheHolder
        {
            public Tensor[] K;
            public Tensor[] V;
            public int[] Sizes;
            public int GlobalCapacity;
            public int SeqLen;
            public bool HostDirty;
            // Cleanup may fail after releasing some tensors. Keep the remaining
            // resources owned, but never expose that holder as reusable state.
            public bool Retired;
            // The holder has been an active cache on a GPU backend, so its K/V
            // has device copies besides the host bytes (the prefix cache's
            // MeasureEndState charges both). Copies and imports start without.
            public bool DeviceMirrored;
        }

        // Per-request fused-decode cache holders, keyed by RequestId.
        private Dictionary<string, Gemma4KvCacheHolder> _fusedHolders;
        // Holders of FINISHED fused requests kept alive for cross-request prefix
        // reuse (multi-turn "请继续"). Keyed by the original RequestId. The executor
        // bounds the count and re-keys one into _fusedHolders via
        // TryRebindRetainedCache when a new request's prompt extends it.
        private Dictionary<string, Gemma4KvCacheHolder> _retainedFusedHolders;
        // RequestId whose holder is currently checked out into the active
        // _kvCacheK/_kvCacheV fields, or null when the primary cache is active.
        private string _activeFusedKey;
        // Snapshot of the primary cache, saved while a fused holder is checked
        // out so RestorePrimaryCache() can reinstate it for the N==1 path.
        private Gemma4KvCacheHolder _primaryHolder;

        /// <summary>The per-sequence fused forward is the path the engine
        /// dispatches for concurrent (N&gt;=2) requests: each request decodes
        /// through its own KV-cache holder (swapped in with a cheap pointer
        /// flip) instead of the round-robin per-step KV extract/inject swap.
        /// It is wired up for any GGML-backed Gemma 4 whose single-token
        /// <c>Forward</c> runs as a (near-)single GPU graph:
        ///   * dense models use the model-wide fused decode kernel
        ///     (<c>NativeGemma4ModelDecode</c>, gated by
        ///     <c>_canUseFusedFullModelDecode</c>);
        ///   * MoE models (<c>gemma-4-26B-A4B</c> etc.) can't use that
        ///     model-wide kernel, but their per-layer fused MoE-decode kernel
        ///     (<c>TryFusedMoELayerDecode</c>) plus the fused per-layer kernel
        ///     for the dense majority keeps each token's <c>Forward</c> down to
        ///     ~one dispatch per layer — far fewer than the op-by-op batched
        ///     paged path, and crucially the batched path can't run MoE at all
        ///     (<c>ForwardBatch</c> throws on MoE layers), so without this the
        ///     engine falls back to the serial KV-swap path and concurrent
        ///     requests decode round-robin.
        /// Both write to the active <c>_kvCacheK</c>/<c>_kvCacheV</c>, which the
        /// per-request holders swap (with <see cref="RefreshDecodeArraysKvCache"/>
        /// repointing the fused-decode pointer arrays), so MoE and dense share
        /// the exact same per-request-cache machinery below.</summary>
        public bool SupportsPerSequenceFusedForward =>
            IsGgmlBackend && (_canUseFusedFullModelDecode || _numExperts > 0);

        public bool SupportsRetainedFusedCache => true;

        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _fusedHolders != null && _fusedHolders.ContainsKey(requestId);

        // Continuous-batching cache-handoff trace, off unless TS_CB_DEBUG=1.
        //
        // Worth keeping: this state machine has four ways in and out of a fused
        // episode and its failures are silent - the model keeps decoding, just
        // against the wrong cache. Reading the handoff order directly is what
        // identified the un-zeroed replacement primary cache that made the first
        // single-stream request after any concurrent burst emit <pad> forever.
        private static readonly bool _cbDebug =
            string.Equals(Environment.GetEnvironmentVariable("TS_CB_DEBUG"), "1", StringComparison.Ordinal);
        private void CbTrace(string what)
        {
            if (!_cbDebug) return;
            Console.Error.WriteLine(
                $"[cb] {what} activeKey={_activeFusedKey ?? "<primary>"} seqLen={_cacheSeqLen} " +
                $"cap={_kvCacheGlobalCapacity} holders={(_fusedHolders?.Count ?? 0)} " +
                $"retained={(_retainedFusedHolders?.Count ?? 0)} primarySaved={(_primaryHolder != null)} " +
                $"k0hash={(_kvCacheK != null && _kvCacheK.Length > 0 && _kvCacheK[0] != null ? _kvCacheK[0].GetHashCode() : 0)}");
        }

        private Gemma4KvCacheHolder SnapshotActiveCache() => new Gemma4KvCacheHolder
        {
            K = _kvCacheK,
            V = _kvCacheV,
            Sizes = _kvCacheSize,
            GlobalCapacity = _kvCacheGlobalCapacity,
            SeqLen = _cacheSeqLen,
            HostDirty = _kvCacheHostDirty,
            DeviceMirrored = KeepsDeviceKvMirrors,
        };

        private void LoadCacheHolder(Gemma4KvCacheHolder h)
        {
            _kvCacheK = h.K;
            _kvCacheV = h.V;
            _kvCacheSize = h.Sizes;
            _kvCacheGlobalCapacity = h.GlobalCapacity;
            _cacheSeqLen = h.SeqLen;
            _kvCacheHostDirty = h.HostDirty;
            // The fused-decode kernels read raw K/V cache pointers cached in
            // _decodeArrays; repoint them at the just-bound holder's tensors.
            RefreshDecodeArraysKvCache();
        }

        private Gemma4KvCacheHolder CreateFreshHolder()
        {
            // Once the prefix cache owns reuse, take a released holder from its pool.
            if (TryTakePooledHolder(out var pooled))
                return pooled;
            // AllocateKvCacheArrays zero-fills: the token-batched fused-decode
            // kernel reads a FIXED 256-padded attention window over each holder's
            // cache, and positions beyond the written length are masked (-inf)
            // but must still be finite or the softmax is poisoned.
            var holder = new Gemma4KvCacheHolder
            {
                GlobalCapacity = _initialGlobalCacheLength,
            };
            AllocateKvCacheArrays(_initialGlobalCacheLength,
                out holder.K, out holder.V, out holder.Sizes, out _);
            return holder;
        }

        /// <summary>Make <paramref name="requestId"/>'s KV cache the model's
        /// active cache, creating an empty one the first time the request is
        /// seen. Cheap: just swaps tensor-array references and refreshes the
        /// fused-decode pointer arrays. Returns true when the cache was freshly
        /// created, so the caller knows to inject any prefix-cache-reused prefix
        /// (NumComputedTokens &gt; 0 at admission) before the first Forward.</summary>
        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            _fusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
                return false; // already active
            CbTrace($"BindSequenceCache({requestId}) ENTER");

            // Save whatever cache is currently checked out so its (possibly
            // grown) tensors aren't lost when we repoint the active fields.
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
            CbTrace($"BindSequenceCache({requestId}) fresh={fresh}");
            return fresh;
        }

        /// <summary>Transition the single in-flight N==1 owner (whose live state
        /// is in the primary cache) into the fused path without copying any KV
        /// bytes: hand the live primary arrays to the owner's holder and give the
        /// primary a fresh empty allocation for later N==1 use. Called by the
        /// executor on the first multi-sequence step when a prior owner exists,
        /// so that owner's history is preserved as its own per-request cache.</summary>
        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            _fusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);

            // Only meaningful when the primary cache is the one currently active
            // (i.e. the N==1 owner ran most recently). If a fused holder is
            // already checked out there is nothing to adopt.
            CbTrace($"AdoptPrimaryCacheToFused({requestId}) ENTER");
            if (_activeFusedKey != null)
                return;
            if (_fusedHolders.ContainsKey(requestId))
                return;

            // The active fields hold the primary cache with the owner's live K/V.
            // Move those arrays into the owner's holder (zero copy).
            var holder = SnapshotActiveCache();
            _fusedHolders[requestId] = holder;
            _activeFusedKey = requestId; // owner's holder is now checked out

            // Give the primary a fresh empty allocation so a future N==1 step for
            // a never-fused request doesn't reset the adopted holder's tensors.
            // AllocateKvCacheArrays zero-fills it - this cache is handed straight
            // to the next single-stream request by RestorePrimaryCache, whose
            // fused decode reads the 256-padded window past the written length.
            AllocateKvCacheArrays(_initialGlobalCacheLength,
                out var k, out var v, out var sizes, out _);
            _primaryHolder = new Gemma4KvCacheHolder
            {
                K = k,
                V = v,
                Sizes = sizes,
                GlobalCapacity = _initialGlobalCacheLength,
                SeqLen = 0,
                HostDirty = false,
            };
            CbTrace($"AdoptPrimaryCacheToFused({requestId}) DONE freshPrimary");
        }

        /// <summary>Reinstate the primary cache as the model's active cache.
        /// Invoked by the executor before an N==1 step that follows a fused
        /// episode so the legacy single-sequence path (which resets/injects the
        /// active cache in place) never clobbers a concurrent request's holder.
        /// No-op when the primary cache is already active.</summary>
        public void RestorePrimaryCache()
        {
            CbTrace("RestorePrimaryCache ENTER");
            if (_activeFusedKey == null)
                return;
            // Save the checked-out fused holder, then swap the primary back in.
            _fusedHolders[_activeFusedKey] = SnapshotActiveCache();
            _activeFusedKey = null;
            if (_primaryHolder != null)
            {
                LoadCacheHolder(_primaryHolder);
                _primaryHolder = null;
            }
            CbTrace("RestorePrimaryCache DONE");
        }

        /// <summary>Release a finished/aborted request's per-request cache. The
        /// engine calls this from InferenceEngine when a sequence leaves the
        /// scheduler. Frees the holder's tensors (after restoring the primary if
        /// the released holder happened to be the active one).</summary>
        public void OnSequenceReleased(string requestId)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
                return;
            CbTrace($"OnSequenceReleased({requestId})");

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
            {
                // The released sequence's cache is the one currently checked out.
                // Capture any growth-replaced arrays before swapping the primary
                // back in, so release disposes the live holder rather than the
                // stale pre-checkout dictionary snapshot.
                holder = SnapshotActiveCache();
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }

            _fusedHolders.Remove(requestId);
            if (_prefixCacheSink != null)
            {
                // Prefix-cache mode (DEC-24): park the allocation instead of freeing
                // it, so releasing one request does not reset every running
                // request's captured decode graphs.
                RecycleOrDisposeHolders(new[] { holder }, dispose: false);
                return;
            }
            DisposeHolder(holder);

            // A captured token-batched decode graph binds this request's KV buffers;
            // now that they're freed, drop all captured batched graphs so a stale
            // entry can't replay against freed memory. They rebuild on next decode.
            if (IsGgmlBackend)
            {
                GgmlBasicOps.Gemma4ResetBatchedDecodeCache();
                GgmlBasicOps.Gemma4ResetMoEBatchedDecodeCache();
                CountDecodeGraphReset();
            }
        }

        /// <summary>Move a FINISHED request's holder out of the active set into the
        /// retained set (keeping its full circular K/V alive) so a later request can
        /// continue from it (see <see cref="TryRebindRetainedCache"/>). Called by the
        /// executor before <see cref="OnSequenceReleased"/>, which then no-ops for the
        /// (already-moved) holder so its buffers are NOT freed. Returns true when a
        /// holder was retained.</summary>
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
            CbTrace($"RetainSequenceCache({requestId} as {key})");
            _retainedFusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);
            if (_retainedFusedHolders.ContainsKey(key)) return false;
            _retainedFusedHolders.EnsureCapacity(checked(_retainedFusedHolders.Count + 1));

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
            {
                // The finishing holder is currently checked out into the model
                // fields. Re-snapshot to capture the (possibly grown) live arrays,
                // then reinstate the primary cache so the active fields don't dangle.
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

        /// <summary>Re-key a retained holder from <paramref name="retainedRequestId"/>
        /// to <paramref name="newRequestId"/> and put it back in the active set, so the
        /// next <see cref="BindSequenceCache"/> for the new request loads it
        /// (fresh==false → no prefix re-inject) and continues from the retained K/V.
        /// Returns false when no retained holder exists for the id.</summary>
        public bool TryRebindRetainedCache(string retainedRequestId, string newRequestId)
        {
            if (_retainedFusedHolders == null
                || string.IsNullOrEmpty(retainedRequestId)
                || string.IsNullOrEmpty(newRequestId))
                return false;
            if (!_retainedFusedHolders.TryGetValue(retainedRequestId, out var holder))
                return false;
            if (holder.Retired) return false;
            _fusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);
            if (_fusedHolders.ContainsKey(newRequestId)) return false;
            _fusedHolders.EnsureCapacity(checked(_fusedHolders.Count + 1));
            _fusedHolders.Add(newRequestId, holder);
            _retainedFusedHolders.Remove(retainedRequestId);
            return true;
        }

        /// <summary>Dispose a retained holder (LRU eviction / shutdown) and free its
        /// KV buffers. Mirrors the free path in <see cref="OnSequenceReleased"/>.</summary>
        public void DiscardRetainedCache(string requestId)
        {
            if (_retainedFusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_retainedFusedHolders.TryGetValue(requestId, out var holder))
                return;
            DisposeHolder(holder);
            _retainedFusedHolders.Remove(requestId);
        }

        // ---- Shared-prefix checkpoints (IBatchedPagedModel) ----

        /// <summary>Gemma 4 can copy its complete cache — every global layer's K/V
        /// and every local layer's circular window — as plain bytes, so a
        /// checkpoint is exact. GGML only: the copy relies on host-side storage.</summary>
        public bool SupportsPrefixCheckpoints => IsGgmlBackend && _kvCacheK != null && _kvCacheV != null;

        /// <summary>Deep-copy the ACTIVE cache (primary or checked-out holder) into the
        /// retained set under <paramref name="key"/>. See
        /// <see cref="IBatchedPagedModel.TryCheckpointActiveCache"/>.</summary>
        public bool TryCheckpointActiveCache(string key)
        {
            if (!SupportsPrefixCheckpoints || string.IsNullOrEmpty(key))
                return false;
            _retainedFusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);
            if (_retainedFusedHolders.ContainsKey(key)
                || (_fusedHolders != null && _fusedHolders.ContainsKey(key)))
                return false;
            _retainedFusedHolders.EnsureCapacity(checked(_retainedFusedHolders.Count + 1));
            // The device copy may be newer than the host bytes the copy reads.
            EnsureKvCacheHostSynchronized();
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
            // Diagnostics must not turn a completed ownership transfer into a
            // reported failure, including failure while formatting the message.
            try { CbTrace($"TryCheckpointActiveCache({key}) seqLen={copy.SeqLen}"); }
            catch { }
            return true;
        }

        /// <summary>Deep-copy the retained holder <paramref name="retainedKey"/> into a
        /// fresh active holder for <paramref name="newRequestId"/>; the retained one
        /// is untouched. See <see cref="IBatchedPagedModel.TryCloneRetainedCache"/>.</summary>
        public bool TryCloneRetainedCache(string retainedKey, string newRequestId)
        {
            if (_retainedFusedHolders == null
                || string.IsNullOrEmpty(retainedKey)
                || string.IsNullOrEmpty(newRequestId))
                return false;
            if (!_retainedFusedHolders.TryGetValue(retainedKey, out var source))
                return false;
            _fusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);
            if (_fusedHolders.ContainsKey(newRequestId)
                || string.Equals(_activeFusedKey, newRequestId, StringComparison.Ordinal))
                return false;
            // A checkpoint is never bound, so its host bytes stay the truth. A retained
            // conversation holder can be device-dirty and is re-keyed, never copied.
            if (source.HostDirty || source.Retired)
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
            try { CbTrace($"TryCloneRetainedCache({retainedKey} -> {newRequestId}) seqLen={source.SeqLen}"); }
            catch { }
            return true;
        }

        /// <summary>An independent copy of <paramref name="source"/>: fresh tensors
        /// (allocated through the same routine as every holder, so donor layers alias
        /// exactly as the original does and unused rows are zero), the bytes of every
        /// unique storage copied, and the copy's device mirrors dropped so the next
        /// forward uploads the copied bytes.</summary>
        // ---- Checkpoints on disk (IBatchedPagedModel) ----
        //
        // Same contract as Qwen 3.5: a checkpoint is host bytes and only host bytes.
        // Local layers are one fixed circular window each and are written whole, so a
        // window that has wrapped round is restored wrapped; global layers write their
        // rows. Donor layers alias another layer's tensors and are not written twice.

        public bool SupportsRetainedCacheSerialization => SupportsPrefixCheckpoints;

        private const uint CheckpointFileMagic = 0x47344B43;   // "G4KC"
        private const int CheckpointFileVersion = 1;

        public unsafe bool TryExportRetainedCache(string key, System.IO.Stream destination)
        {
            if (!SupportsPrefixCheckpoints || destination == null || string.IsNullOrEmpty(key)
                || _retainedFusedHolders == null)
                return false;
            if (!_retainedFusedHolders.TryGetValue(key, out var h) || h.K == null || h.HostDirty || h.Retired)
                return false;

            int rows = Math.Max(0, Math.Min(h.SeqLen, h.GlobalCapacity));
            var w = new System.IO.BinaryWriter(destination, System.Text.Encoding.UTF8, leaveOpen: true);
            w.Write(CheckpointFileMagic);
            w.Write(CheckpointFileVersion);
            w.Write(KVStateFingerprint ?? string.Empty);
            w.Write(Config.NumLayers);
            w.Write(h.SeqLen);
            w.Write(rows);
            w.Write(_slidingWindow);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l))
                {
                    w.Write((byte)2);
                    continue;
                }
                bool local = IsLocalLayer(l);
                w.Write((byte)(local ? 1 : 0));
                w.Write(h.Sizes[l]);
                WriteCacheRows(w, h.K[l], local ? h.Sizes[l] : rows);
                WriteCacheRows(w, h.V[l], local ? h.Sizes[l] : rows);
            }
            w.Flush();
            return true;
        }

        public unsafe bool TryImportRetainedCache(string key, System.IO.Stream source)
        {
            if (!SupportsPrefixCheckpoints || source == null || string.IsNullOrEmpty(key))
                return false;
            _retainedFusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);
            if (_retainedFusedHolders.ContainsKey(key) || (_fusedHolders != null && _fusedHolders.ContainsKey(key)))
                return false;
            _retainedFusedHolders.EnsureCapacity(checked(_retainedFusedHolders.Count + 1));

            var r = new System.IO.BinaryReader(source, System.Text.Encoding.UTF8, leaveOpen: true);
            if (r.ReadUInt32() != CheckpointFileMagic || r.ReadInt32() != CheckpointFileVersion)
                return false;
            if (!string.Equals(r.ReadString(), KVStateFingerprint ?? string.Empty, StringComparison.Ordinal))
                return false;
            if (r.ReadInt32() != Config.NumLayers)
                return false;
            int seqLen = r.ReadInt32();
            int rows = r.ReadInt32();
            if (rows < 0 || seqLen < 0 || rows > seqLen || r.ReadInt32() != _slidingWindow)
                return false;

            // Sized to the rows, never past this model's window: the writer may have run
            // under a larger MAX_CONTEXT than this process does.
            int globalCap = Math.Max(1, _maxContextLength > 0
                ? Math.Min(_maxContextLength, CacheCapacityFor(rows))
                : CacheCapacityFor(rows));
            if (rows > globalCap)
                return false;
            var holder = new Gemma4KvCacheHolder
            {
                GlobalCapacity = globalCap,
                SeqLen = seqLen,
                HostDirty = false,
            };
            AllocateKvCacheArrays(globalCap, out holder.K, out holder.V, out holder.Sizes, out _);
            var k = holder.K;
            var v = holder.V;
            var sizes = holder.Sizes;
            bool ok = false;
            try
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    byte kind = r.ReadByte();
                    bool donor = _kvDonorMap.ContainsKey(l);
                    if (kind == 2 || donor)
                    {
                        if (kind != 2 || !donor)
                            return false;
                        continue;
                    }
                    bool local = IsLocalLayer(l);
                    int storedSize = r.ReadInt32();
                    // A local layer is one fixed window, written whole, so its size must
                    // be this model's window. A global layer's stored size is the
                    // capacity the WRITER happened to hold (its primary cache's, or the
                    // rounded row count, whichever was smaller) and only the rows were
                    // written; this holder has its own capacity, sized to the rows.
                    if (kind != (local ? 1 : 0) || (local ? storedSize != sizes[l] : storedSize < rows))
                        return false;
                    int layerRows = local ? sizes[l] : rows;
                    if (!ReadCacheRows(r, k[l], layerRows) || !ReadCacheRows(r, v[l], layerRows))
                        return false;
                    InvalidateTensorDeviceCache(k[l]);
                    InvalidateTensorDeviceCache(v[l]);
                }
                _retainedFusedHolders.Add(key, holder);
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
                    DisposeHolder(holder);
            }
        }

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

        private Gemma4KvCacheHolder DeepCopyHolder(Gemma4KvCacheHolder source)
        {
            // The global layers are sized to what the source HOLDS (plus a padded
            // window), not to the generation budget its primary cache was reserved
            // for: a checkpoint lives for the life of the model and every new chat
            // clones it. The local layers are one fixed circular window each and are
            // copied whole. A copy grows on demand like any holder.
            int rows = Math.Max(0, Math.Min(source.SeqLen, source.GlobalCapacity));
            int globalCap = Math.Max(1, Math.Min(source.GlobalCapacity, CacheCapacityFor(rows)));
            var copy = new Gemma4KvCacheHolder
            {
                GlobalCapacity = globalCap,
                SeqLen = source.SeqLen,
            };
            AllocateKvCacheArrays(globalCap, out copy.K, out copy.V, out copy.Sizes, out _);
            bool complete = false;
            try
            {
                var seen = new HashSet<Tensor>();
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (_kvDonorMap.ContainsKey(l)) continue;
                    bool local = IsLocalLayer(l);
                    if (source.K[l] != null && copy.K[l] != null && seen.Add(source.K[l]))
                    {
                        if (local) CopyCacheTensorBytes(source.K[l], copy.K[l]);
                        else CopyCacheRows(source.K[l], copy.K[l], rows);
                        InvalidateTensorDeviceCache(copy.K[l]);
                    }
                    if (source.V != null && source.V[l] != null && copy.V[l] != null && seen.Add(source.V[l]))
                    {
                        if (local) CopyCacheTensorBytes(source.V[l], copy.V[l]);
                        else CopyCacheRows(source.V[l], copy.V[l], rows);
                        InvalidateTensorDeviceCache(copy.V[l]);
                    }
                }
                complete = true;
                return copy;
            }
            finally
            {
                if (!complete) DisposeHolder(copy);
            }
        }

        private void DisposeHolder(Gemma4KvCacheHolder holder)
        {
            if (holder == null) return;
            holder.Retired = true;
            // A batched release (DiscardRetainedCaches) resets once, before its
            // first disposal, and suppresses the per-holder reset here.
            if (IsGgmlBackend && _holderGraphResetSuppressed == 0)
            {
                GgmlBasicOps.Gemma4ResetBatchedDecodeCache();
                GgmlBasicOps.Gemma4ResetMoEBatchedDecodeCache();
                CountDecodeGraphReset();
            }
            DisposeKvCacheArrays(holder.K, holder.V);
        }

        // No HashSet allocation on the memory-pressure cleanup path. Clear every
        // alias only after that tensor was released; a retry can see remaining ones.
        private void DisposeKvCacheArrays(Tensor[] k, Tensor[] v)
        {
            void Release(Tensor tensor)
            {
                if (tensor == null) return;
                InvalidateTensorDeviceCache(tensor);
                tensor.Dispose();
                if (k != null) for (int i = 0; i < k.Length; i++)
                    if (ReferenceEquals(k[i], tensor)) k[i] = null;
                if (v != null) for (int i = 0; i < v.Length; i++)
                    if (ReferenceEquals(v[i], tensor)) v[i] = null;
            }
            if (k != null) for (int i = 0; i < k.Length; i++) Release(k[i]);
            if (v != null) for (int i = 0; i < v.Length; i++) Release(v[i]);
        }

        /// <summary>Free every per-request fused cache holder (and the saved
        /// primary snapshot). Called on model dispose. Does not touch the
        /// currently-active arrays (those are the model's _kvCacheK, disposed by
        /// the normal cache teardown).</summary>
        private void DisposeAllFusedHolders()
        {
            if (_fusedHolders != null)
            {
                foreach (var kv in _fusedHolders)
                {
                    // Skip the active holder; its arrays are _kvCacheK and are
                    // disposed by the model's main cache teardown.
                    if (string.Equals(kv.Key, _activeFusedKey, StringComparison.Ordinal))
                        continue;
                    DisposeHolder(kv.Value);
                }
                _fusedHolders.Clear();
                _fusedHolders = null;
            }
            if (_retainedFusedHolders != null)
            {
                foreach (var kv in _retainedFusedHolders)
                    DisposeHolder(kv.Value);
                _retainedFusedHolders.Clear();
                _retainedFusedHolders = null;
            }
            DisposeHolderPool();
            if (_primaryHolder != null)
            {
                // If a fused holder is active, the primary snapshot owns distinct
                // arrays that must be freed; if the primary is active it shares
                // _kvCacheK and is freed by the main teardown.
                if (_activeFusedKey != null)
                    DisposeHolder(_primaryHolder);
                _primaryHolder = null;
            }
            _activeFusedKey = null;
        }
    }
}
