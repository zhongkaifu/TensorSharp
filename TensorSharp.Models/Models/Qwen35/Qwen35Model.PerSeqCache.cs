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
            // GDN recurrent state: host conv ring + write idx + device delta state.
            public float[][] ConvState;
            public int[] ConvWriteIdx;
            public Tensor[] DeltaState;
            // Fused-decode conv scratch (ggml [time,channel], cacheable device buffer
            // keyed on this host ptr) + whether this holder's device GDN state is
            // currently seeded (resident) on the device.
            public IntPtr ConvScratch;
            public bool FdStateResident;
        }

        // Per-request fused-decode holders, keyed by RequestId.
        private Dictionary<string, Qwen35KvCacheHolder> _fusedHolders;
        // RequestId whose holder is currently checked out into the active model
        // fields, or null when the primary cache is active.
        private string _activeFusedKey;
        // Snapshot of the primary cache, saved while a fused holder is checked out.
        private Qwen35KvCacheHolder _primaryHolder;

        /// <summary>The per-sequence fused forward is the path the engine
        /// dispatches for concurrent (N&gt;=2) requests: each request decodes
        /// through its own KV + GDN holder (swapped in with a cheap reference
        /// flip) instead of the broken/slow batched paged path. Wired up for
        /// ggml_cuda where the whole-model fused decode (TryFullModelDecode) is
        /// available.</summary>
        public bool SupportsPerSequenceFusedForward =>
            _backend == BackendType.GgmlCuda && _fullDecodeEnabled && !_fdUnsupported && !_fdSpecSessionActive;

        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _fusedHolders != null && _fusedHolders.ContainsKey(requestId);

        private Qwen35KvCacheHolder SnapshotActiveCache() => new Qwen35KvCacheHolder
        {
            K = _kvCacheK,
            V = _kvCacheV,
            KvCapacity = _kvCacheCapacity,
            CacheSeqLen = _cacheSeqLen,
            ConvState = _convState,
            ConvWriteIdx = _convStateWriteIdx,
            DeltaState = _deltaStateTensor,
            ConvScratch = _fdConvScratch,
            FdStateResident = _fdStateResident,
        };

        private void LoadCacheHolder(Qwen35KvCacheHolder h)
        {
            _kvCacheK = h.K;
            _kvCacheV = h.V;
            _kvCacheCapacity = h.KvCapacity;
            _cacheSeqLen = h.CacheSeqLen;
            _convState = h.ConvState;
            _convStateWriteIdx = h.ConvWriteIdx;
            _deltaStateTensor = h.DeltaState;
            _fdConvScratch = h.ConvScratch;
            _fdStateResident = h.FdStateResident;
            // The fused-decode descriptors are rebuilt from these fields on every
            // TryFullModelDecode call, so no cached pointer array needs refreshing
            // (unlike Gemma4's _decodeArrays). The native g_q35dc_pool selects this
            // holder's captured graph by its first attention KV pointer.
        }

        private Qwen35KvCacheHolder CreateFreshHolder()
        {
            int numLayers = _kvCacheK.Length;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convDim = _convKernel - 1;
            DType kvDtype = _kvCacheDtype.ToDType();
            int cap = _initialKvCacheCapacity > 0 ? _initialKvCacheCapacity : _kvCacheCapacity;

            var k = new Tensor[numLayers];
            var v = new Tensor[numLayers];
            var convState = new float[numLayers][];
            var convWriteIdx = new int[numLayers];
            var deltaState = new Tensor[numLayers];
            int gdnCount = 0;
            for (int l = 0; l < numLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    k[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, cap, Config.HeadDim);
                    v[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, cap, Config.HeadDim);
                    InitializeCacheTensor(k[l]);
                    InitializeCacheTensor(v[l]);
                }
                else
                {
                    convState[l] = new float[Math.Max(0, convDim) * qkvDim];
                    convWriteIdx[l] = 0;
                    deltaState[l] = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, _headKDim);
                    Ops.Fill(deltaState[l], 0);
                    gdnCount++;
                }
            }
            IntPtr convScratch = Marshal.AllocHGlobal(Math.Max(1, gdnCount) * Math.Max(1, convDim) * qkvDim * sizeof(float));

            return new Qwen35KvCacheHolder
            {
                K = k,
                V = v,
                KvCapacity = cap,
                CacheSeqLen = 0,
                ConvState = convState,
                ConvWriteIdx = convWriteIdx,
                DeltaState = deltaState,
                ConvScratch = convScratch,
                FdStateResident = false,
            };
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
                // primary back in so the active fields don't dangle, then the
                // released holder's arrays are distinct and safe to free below.
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }

            _fusedHolders.Remove(requestId);
            DisposeHolder(holder);
        }

        private void DisposeHolder(Qwen35KvCacheHolder holder)
        {
            if (holder == null) return;
            if (holder.K != null)
                foreach (var t in holder.K) t?.Dispose();
            if (holder.V != null)
                foreach (var t in holder.V) t?.Dispose();
            if (holder.DeltaState != null)
                foreach (var t in holder.DeltaState) t?.Dispose();
            if (holder.ConvScratch != IntPtr.Zero)
                Marshal.FreeHGlobal(holder.ConvScratch);
        }

        /// <summary>Free every per-request fused holder (and the saved primary
        /// snapshot). Called on model dispose. Does NOT touch the currently-active
        /// arrays (those are the model's _kvCacheK / _deltaStateTensor / _fdConvScratch,
        /// freed by the normal cache teardown).</summary>
        private void DisposeAllFusedHolders()
        {
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
            }
            if (_primaryHolder != null)
            {
                // If a fused holder is active, the primary snapshot owns distinct
                // arrays that must be freed; if the primary is active it shares the
                // model fields and is freed by the main teardown.
                if (_activeFusedKey != null)
                    DisposeHolder(_primaryHolder);
                _primaryHolder = null;
            }
            _activeFusedKey = null;
        }
    }
}
