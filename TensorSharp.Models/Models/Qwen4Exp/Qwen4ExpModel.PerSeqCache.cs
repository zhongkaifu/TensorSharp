// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Per-request KV + recurrent-state holders for the per-sequence fused path
// (the Qwen3.8-Flash-Next analogue of Qwen35Model.PerSeqCache).
//
// qwen4exp keeps three kinds of per-conversation state: the attention KV (and
// QSA indexer key) caches, the GatedDeltaNet conv + delta-net state, and the
// PLE n-gram/conv history. The fused span kernel holds the GDN and PLE state
// in DEVICE buffers that are updated in place inside persisted graphs, keyed
// on the HOST seed pointers the descriptors carry (see the seq-state map in
// ggml_ops_qwen4exp.cpp). Giving each in-flight request its own set of host
// arrays/tensors therefore gives it its own device state and its own cached
// graphs (the span keys its graph on the descriptor-array addresses), and
// switching requests is a cheap reference swap - no state download/upload, no
// graph invalidation.
//
// Each sequence then decodes through the proven single-graph fused Forward;
// concurrency is served by the engine's per-sequence round-robin. The N==1
// path is untouched: it keeps the model's primary cache, reinstated by
// RestorePrimaryCache() after any multi-sequence episode.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel : IBatchedPagedModel
    {
        private sealed class Qwen4ExpKvCacheHolder
        {
            // Attention KV + QSA indexer keys (null on recurrent layers).
            public Tensor[] K;
            public Tensor[] V;
            public Tensor[] IdxK;
            public Qwen4ExpQsaArgs[] QsaArgs;
            public int[] QsaPositions;
            public int QsaPositionCount;
            public int KvCapacity;
            public int CacheSeqLen;
            public bool KvHostStale;
            // GDN state: host ring (op-by-op path) + the fused path's seed
            // tensors, whose HOST pointers key the native device-state entries.
            public float[][] GdnConvState;
            public int[] GdnConvWriteIdx;
            public Tensor[] GdnConvStateT;
            public Tensor[] GdnStateT;
            // PLE conv history (pinned; seed source + native state key) and the
            // n-gram token window.
            public float[] PleConvState;
            public List<int> PleHistory;
            public int PleNextPos;
            public int MropeCacheGap;
            public bool SpecStateFailed;
            // True once a fused forward has seeded the native per-sequence state
            // entries (GDN conv+ssm, PLE conv, QSA raw keys) from this holder's host
            // seeds: from then on the DEVICE entries are the truth and a copy must
            // download them. False for a fresh, copied or imported holder, whose host
            // bytes are what the next forward uploads (see RetainedCache).
            public bool DeviceStateSeeded;
            // Retained-set bookkeeping (RetainedCache): a shared-prefix checkpoint is
            // a host-authoritative copy that is cloned, never bound; a retired holder
            // is being torn down and must not be re-keyed; RetainedBytes is what it
            // counts against the retention budget; RetainedSerial orders eviction.
            public bool IsCheckpoint;
            public bool Retired;
            public bool Disposed;
            // The holder has been an active cache on a GPU backend, so its caches have
            // device copies besides the host seeds (the prefix cache's MeasureEndState
            // charges both). Copies start without.
            public bool DeviceMirrored;
            public long RetainedBytes;
            public long RetainedSerial;
            // Pinned descriptor arrays. Their addresses are the native graph
            // signature, so per-holder arrays select per-holder graphs.
            public Qwen4ExpAttnArgs[] AttnArgs;
            public Qwen4ExpGdnArgs[] GdnArgs;
            public Qwen4ExpPleArgs[] PleArgs;
            // Span graph-cache slot base for this holder (multiples of 8).
            public int SlotBase;
        }

        private Dictionary<string, Qwen4ExpKvCacheHolder> _fusedHolders;
        private string _activeFusedKey;
        private Qwen4ExpKvCacheHolder _primaryHolder;
        // Span graph-slot base of the ACTIVE holder (0 = primary).
        private int _seqSlotBase;
        // Span slot bases in use (multiples of 16 up to the native slot count).
        private readonly HashSet<int> _usedSlotBases = new() { 0 };

        private int ClaimSlotBase()
        {
            for (int b = 16; b <= 112; b += 16)
            {
                if (_usedSlotBases.Add(b))
                    return b;
            }
            // More concurrent holders than bases: share the last base. Graph
            // signatures still keep correctness; alternation costs rebuilds.
            return 112;
        }

        private void ReleaseSlotBase(int b)
        {
            if (b != 0) _usedSlotBases.Remove(b);
        }

        /// <summary>The batched paged forward has no qwen4exp implementation (the
        /// GDN/PLE state has no paged layout); the planner routes concurrency
        /// through the per-sequence fused path below.</summary>
        public bool BatchedForwardAvailable => false;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
            => throw new NotSupportedException(
                "qwen4exp serves concurrency through per-sequence state holders, not ForwardBatch.");

        /// <summary>Per-sequence holders need the GGML fused span path: it is
        /// where the per-holder graph/state keying lives. The managed op-by-op
        /// path shares scratch that is not per-sequence.</summary>
        public bool SupportsPerSequenceFusedForward =>
            IsGgmlBackend && _tokenGraphEnabled && !_tokenGraphUnsupported;

        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _fusedHolders != null && _fusedHolders.ContainsKey(requestId);

        private Qwen4ExpKvCacheHolder SnapshotActiveCache() => new Qwen4ExpKvCacheHolder
        {
            K = _kCache,
            V = _vCache,
            IdxK = _idxKCache,
            QsaArgs = _qsaArgs, QsaPositions = _qsaPositions, QsaPositionCount = _qsaPositionCount,
            KvCapacity = _kvCacheCapacity,
            CacheSeqLen = _cacheSeqLen,
            KvHostStale = _kvCacheHostStale,
            GdnConvState = _gdnConvState,
            GdnConvWriteIdx = _gdnConvWriteIdx,
            GdnConvStateT = _gdnConvStateT,
            GdnStateT = _gdnStateT,
            PleConvState = _pleConvState,
            PleHistory = _pleHistory,
            PleNextPos = _pleNextPos,
            MropeCacheGap = _mropeCacheGap,
            SpecStateFailed = _specStateFailed,
            DeviceStateSeeded = _deviceStateAuthoritative,
            AttnArgs = _attnArgs,
            GdnArgs = _gdnArgs,
            PleArgs = _pleArgs,
            SlotBase = _seqSlotBase,
            DeviceMirrored = KeepsDeviceKvMirrors,
        };

        private void LoadCacheHolder(Qwen4ExpKvCacheHolder h)
        {
            _kCache = h.K;
            _vCache = h.V;
            _idxKCache = h.IdxK;
            _qsaArgs = h.QsaArgs; _qsaPositions = h.QsaPositions; _qsaPositionCount = h.QsaPositionCount;
            _kvCacheCapacity = h.KvCapacity;
            _cacheSeqLen = h.CacheSeqLen;
            _kvCacheHostStale = h.KvHostStale;
            _gdnConvState = h.GdnConvState;
            _gdnConvWriteIdx = h.GdnConvWriteIdx;
            _gdnConvStateT = h.GdnConvStateT;
            _gdnStateT = h.GdnStateT;
            _pleConvState = h.PleConvState;
            _pleHistory = h.PleHistory;
            _pleNextPos = h.PleNextPos;
            _mropeCacheGap = h.MropeCacheGap;
            _specStateFailed = h.SpecStateFailed;
            _deviceStateAuthoritative = h.DeviceStateSeeded;
            // Null args are lazily rebuilt by Ensure*Args from THIS holder's
            // tensors, which is what keys the native graphs per holder.
            _attnArgs = h.AttnArgs;
            _gdnArgs = h.GdnArgs;
            _pleArgs = h.PleArgs;
            _seqSlotBase = h.SlotBase;
        }

        private Qwen4ExpKvCacheHolder CreateFreshHolder()
            => AllocateHolder(_initialKvCacheCapacity > 0 ? _initialKvCacheCapacity : _kvCacheCapacity);

        /// <summary>A brand-new, zeroed holder with attention K/V and QSA raw-key
        /// caches of <paramref name="cap"/> rows and zeroed recurrent/PLE state. The
        /// allocation half of <see cref="CreateFreshHolder"/>, on its own so a copy
        /// can be sized to what its source holds (RetainedCache).</summary>
        private Qwen4ExpKvCacheHolder AllocateHolder(int cap)
        {
            int nLayer = Config.NumLayers;
            DType kvDtype = _kvCacheDtype.ToDType();
            if (cap <= 0) throw new ArgumentOutOfRangeException(nameof(cap));

            var k = new Tensor[nLayer];
            var v = new Tensor[nLayer];
            var idx = new Tensor[nLayer];
            var convState = new float[nLayer][];
            var convWriteIdx = new int[nLayer];
            var convStateT = new Tensor[nLayer];
            var stateT = new Tensor[nLayer];

            try
            {
                for (int l = 0; l < nLayer; l++)
                {
                    if (_isRecurrent[l])
                    {
                        convState[l] = new float[(long)(_convKernel - 1) * _convDim];
                        convStateT[l] = new Tensor(_allocator, DType.Float32, _convKernel - 1, _convDim);
                        Ops.Fill(convStateT[l], 0f);
                        stateT[l] = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, _headKDim);
                        Ops.Fill(stateT[l], 0f);
                        continue;
                    }
                    k[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, cap, Config.HeadDim);
                    v[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, cap, Config.HeadDim);
                    InitializeCacheTensor(k[l]);
                    InitializeCacheTensor(v[l]);
                    if (UsesQsa(l))
                    {
                        idx[l] = new Tensor(_allocator, kvDtype, 1, cap, _indexerHeadDim);
                        InitializeQsaCache(idx[l]);
                    }
                }

                float[] pleConv = null;
                if (_pleHeads > 0)
                    pleConv = GC.AllocateArray<float>(
                        checked((int)((long)(_pleConvKernel - 1) * _pleNgram * _hcDim)), pinned: true);

                var holder = new Qwen4ExpKvCacheHolder
                {
                    K = k,
                    V = v,
                    IdxK = idx,
                    KvCapacity = cap,
                    CacheSeqLen = 0,
                    KvHostStale = false,
                    GdnConvState = convState,
                    GdnConvWriteIdx = convWriteIdx,
                    GdnConvStateT = convStateT,
                    GdnStateT = stateT,
                    PleConvState = pleConv,
                    PleHistory = new List<int>(),
                    PleNextPos = 0,
                    MropeCacheGap = 0,
                    AttnArgs = null,
                    GdnArgs = null,
                    PleArgs = null,
                    SlotBase = 0,
                };
                holder.SlotBase = ClaimSlotBase();
                return holder;
            }
            catch
            {
                // Allocation can fail after earlier layers already own buffers.
                // None of these tensors has been published to a live holder.
                foreach (Tensor[] set in new[] { k, v, idx, convStateT, stateT })
                    foreach (Tensor tensor in set)
                        if (tensor != null)
                        {
                            InvalidateTensorDeviceCache(tensor);
                            tensor.Dispose();
                        }
                throw;
            }
        }

        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            _fusedHolders ??= new Dictionary<string, Qwen4ExpKvCacheHolder>(StringComparer.Ordinal);

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
                return false;

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
            return fresh;
        }

        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            _fusedHolders ??= new Dictionary<string, Qwen4ExpKvCacheHolder>(StringComparer.Ordinal);
            if (_activeFusedKey != null) return;
            if (_fusedHolders.ContainsKey(requestId)) return;

            var holder = SnapshotActiveCache();
            _fusedHolders[requestId] = holder;
            _activeFusedKey = requestId;

            _primaryHolder = CreateFreshHolder();
            // The adopted primary keeps slot base 0; the fresh primary takes a
            // fused-range base so the two never share span graph slots.
        }

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

        public void OnSequenceReleased(string requestId)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
                return;

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
            {
                // The dictionary entry is stale for the active holder: growth or a
                // re-seed may have replaced fields since it was loaded. Dispose what
                // is live, not what was recorded at bind time.
                holder = SnapshotActiveCache();
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }

            _fusedHolders.Remove(requestId);
            DisposeHolder(holder);
            ReleaseSlotBase(holder.SlotBase);
        }

        private unsafe IntPtr[] HolderStateKeys(Qwen4ExpKvCacheHolder holder)
        {
            var keys = new List<IntPtr>();
            if (holder.IdxK != null)
                foreach (var t in holder.IdxK)
                    if (t != null) keys.Add(TensorComputePrimitives.GetStoragePointer(t));
            if (holder.GdnConvStateT != null)
                foreach (var t in holder.GdnConvStateT)
                    if (t != null) keys.Add((IntPtr)GetFloatPtr(t));
            if (holder.PleConvState != null && holder.PleConvState.Length > 0)
                keys.Add(Marshal.UnsafeAddrOfPinnedArrayElement(holder.PleConvState, 0));
            return keys.ToArray();
        }

        private void DisposeHolder(Qwen4ExpKvCacheHolder holder)
        {
            if (holder == null || holder.Disposed) return;
            holder.Disposed = true;
            holder.Retired = true;
            ReleaseMtpState(holder.GdnConvStateT);
            if (ReferenceEquals(_specSnapshotOwner, holder.GdnConvStateT))
                ReleaseSpecSnapshot();

            // Free the native device-state entries FIRST (that also drops every
            // cached graph, so nothing baked can reference the buffers below);
            // then the tensors.
            // A batched release (DiscardRetainedCaches) frees every holder's native entries in one
            // call, before the first disposal, and suppresses the per-holder release here.
            if (IsGgmlBackend && _holderSeqStateReleaseSuppressed == 0)
            {
                GgmlBasicOps.Qwen4ExpReleaseSeqState(HolderStateKeys(holder));
                CountDecodeGraphReset();
            }

            void DisposeSet(Tensor[] set)
            {
                if (set == null) return;
                foreach (var t in set)
                {
                    if (t == null) continue;
                    InvalidateTensorDeviceCache(t);
                    t.Dispose();
                }
            }
            DisposeSet(holder.K);
            DisposeSet(holder.V);
            DisposeSet(holder.IdxK);
            DisposeSet(holder.GdnConvStateT);
            DisposeSet(holder.GdnStateT);
        }

        private void DisposeAllFusedHolders()
        {
            if (_fusedHolders != null)
            {
                foreach (var kv in _fusedHolders)
                {
                    if (string.Equals(kv.Key, _activeFusedKey, StringComparison.Ordinal))
                        continue;
                    DisposeHolder(kv.Value);
                }
                _fusedHolders.Clear();
                _fusedHolders = null;
            }
            if (_retainedFusedHolders != null)
            {
                foreach (var holder in _retainedFusedHolders.Values)
                {
                    DisposeHolder(holder);
                    ReleaseSlotBase(holder.SlotBase);
                }
                _retainedFusedHolders.Clear();
                _retainedFusedHolders = null;
            }
            if (_primaryHolder != null)
            {
                if (_activeFusedKey != null)
                    DisposeHolder(_primaryHolder);
                _primaryHolder = null;
            }
            _activeFusedKey = null;
        }
    }
}
