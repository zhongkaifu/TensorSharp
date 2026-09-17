// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Gemma 4's radix prefix cache: complete holders copied or donated with
// rewinds limited by the unwrapped sliding window. Released holders reuse the
// pool, and device-authoritative state is settled before copying.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class Gemma4Model : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        // Set by AttachPrefixCache: the prefix cache owns this model's reuse state.
        private IPrefixPayloadSink _prefixCacheSink;
        // Released holders parked for the next request (prefix-cache mode only).
        private List<Gemma4KvCacheHolder> _holderPool;
        // >0 while a batched release has already reset the decode graphs.
        private int _holderGraphResetSuppressed;

        // ---------------------------------------------------------------- capabilities

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            bool holders = SupportsPerSequenceFusedForward;
            return new PrefixCacheCapabilities
            {
                Class = FamilyClass.S,
                Readiness = PrefixCacheMode.Tree,
                NamespaceFingerprint = KVStateFingerprint,
                EndState = holders ? EndStateSupport.CopyAndDonate : EndStateSupport.None,
                CanCaptureCopy = holders && SupportsPrefixCheckpoints,
                AdoptPrimaryOnDisplacement = holders,
                PrimaryResident = true,
                MinRetainTokens = 32,
                Truncation = _slidingWindow > 0 ? TruncationKind.WithinUnwrappedWindow : TruncationKind.Any,
                TruncationParameter = Math.Max(0, _slidingWindow),
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                RewindCapTokens = 16,
                Pages = PageSupport.None,          // DEC-17: end states only
                // Absolute positions and an atomic bidirectional span mask: reuse across
                // a media span is exact (Phase 0 value); M8 calibrates MmReuseMinTokens.
                ReuseAcrossMediaSpan = SupportsReuseAcrossMediaSpan,
                MmReuseMinTokens = 0,
                Persistable = holders && SupportsRetainedCacheSerialization,
            };
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink)
            => _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));

        public void DetachPrefixCache() => _prefixCacheSink = null;

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---------------------------------------------------------------- end states

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return SupportsPerSequenceFusedForward
                && HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
        }

        /// <summary>Holder present and live, holding exactly <paramref name="payloadTokens"/> tokens, and a
        /// rewind (if any) that the ring still allows: <c>target == tokens || tokens ≤ W</c>.</summary>
        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return false;
            if (holder.SeqLen != payloadTokens || targetTokens < 0 || targetTokens > payloadTokens) return false;
            return targetTokens == payloadTokens || CanTruncateKVCache(payloadTokens, targetTokens);
        }

        /// <summary>§6.1: bind the retained holder, run the synchronous host flush, rebind the previous
        /// active cache. A holder whose host bytes are already current needs nothing.</summary>
        public bool SettleForCopy(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return false;
            if (!holder.HostDirty) return true;
            if (!IsGgmlBackend) return false;
            Gemma4KvCacheHolder previous = SnapshotActiveCache();
            LoadCacheHolder(holder);
            try
            {
                EnsureKvCacheHostSynchronized();
                holder.HostDirty = _kvCacheHostDirty;
            }
            finally
            {
                LoadCacheHolder(previous);
            }
            return !holder.HostDirty;
        }

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return default;
            long bytes = HolderBytes(holder);
            var vector = new ResourceVector { HostKv = bytes, DeviceKv = holder.DeviceMirrored ? bytes : 0 };
            return new PayloadFootprint(holder.SeqLen, holder.GlobalCapacity, vector, PositionDelta: 0);
        }

        /// <summary>What <see cref="DeepCopyHolder"/> allocates: local layers are one whole window each,
        /// global layers are sized to the rows the source holds plus a padded window. A clone is
        /// host-authoritative, so nothing is charged to the device until it is bound.</summary>
        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens)
        {
            if (!TryGetRetained(payloadKey, out var source)) return default;
            int rows = Math.Max(0, Math.Min(source.SeqLen, source.GlobalCapacity));
            int globalCap = Math.Max(1, Math.Min(source.GlobalCapacity, CacheCapacityFor(rows)));
            long bytes = 0;
            var seen = new HashSet<Storage>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l)) continue;
                bool local = IsLocalLayer(l);
                foreach (Tensor t in new[] { source.K?[l], source.V?[l] })
                {
                    if (t == null || !seen.Add(t.Storage)) continue;
                    long capacity = t.Sizes[1];
                    bytes += local || capacity <= 0 ? t.Storage.ByteLength : t.Storage.ByteLength / capacity * globalCap;
                }
            }
            return new ResourceVector { HostKv = bytes };
        }

        /// <summary>The batched <see cref="DiscardRetainedCache"/> (DEC-24): every released holder goes to
        /// the pool when the prefix cache owns this model, and a batch resets the decode graphs at most
        /// once, before its first disposal. Invalidation and pressure dispose.</summary>
        public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            if (_retainedFusedHolders == null || payloadKeys.IsEmpty) return;
            List<Gemma4KvCacheHolder> released = null;
            foreach (string key in payloadKeys)
            {
                if (key != null && _retainedFusedHolders.Remove(key, out var holder))
                    (released ??= new List<Gemma4KvCacheHolder>()).Add(holder);
            }
            if (released == null) return;
            bool dispose = reason is ReleaseReason.Invalidated or ReleaseReason.Pressure or ReleaseReason.Reset;
            RecycleOrDisposeHolders(released, dispose);
        }

        // ---------------------------------------------------------------- the holder pool

        private bool TryTakePooledHolder(out Gemma4KvCacheHolder holder)
        {
            holder = null;
            if (_prefixCacheSink == null || _holderPool == null || _holderPool.Count == 0) return false;
            holder = _holderPool[_holderPool.Count - 1];
            _holderPool.RemoveAt(_holderPool.Count - 1);
            return true;
        }

        /// <summary>
        /// Park what fits the pool (TS_KV_HOLDER_POOL_MAX, and half the smaller of host and device
        /// headroom, the Qwen 3.5 pool's rule) and dispose the rest. A parked holder gets the logical
        /// reset the primary cache gets between requests (<see cref="ResetKVCacheCore"/>): on Metal and
        /// Vulkan its device copies stay valid and its bytes stay finite, so nothing is reset; on the other
        /// GGML backends its device copies are dropped, which is a decode-graph reset. Either way the whole
        /// batch resets at most once, before anything is freed or dropped.
        /// </summary>
        private void RecycleOrDisposeHolders(IReadOnlyList<Gemma4KvCacheHolder> holders, bool dispose)
        {
            var toPool = new List<Gemma4KvCacheHolder>();
            var toDispose = new List<Gemma4KvCacheHolder>();
            int poolMax = ExecutionOptions.FromEnvironment().KvHolderPoolMax;
            int pooledCount = _holderPool?.Count ?? 0;
            long pooledBytes = PooledHolderBytes();
            long spare = PoolSpareBytes();
            foreach (var holder in holders)
            {
                if (holder == null) continue;
                long bytes = HolderBytes(holder);
                bool fits = pooledCount + toPool.Count < poolMax
                    && (spare < 0 || pooledBytes + bytes <= spare / 2);
                if (!dispose && _prefixCacheSink != null && !holder.Retired && holder.K != null && fits)
                {
                    toPool.Add(holder);
                    pooledBytes += bytes;
                }
                else
                {
                    toDispose.Add(holder);
                }
            }

            bool recycleDropsDeviceCopies = IsGgmlBackend
                && _backend != BackendType.GgmlMetal && _backend != BackendType.GgmlVulkan;
            if (IsGgmlBackend && (toDispose.Count > 0 || (toPool.Count > 0 && recycleDropsDeviceCopies)))
            {
                GgmlBasicOps.Gemma4ResetBatchedDecodeCache();
                GgmlBasicOps.Gemma4ResetMoEBatchedDecodeCache();
                if (toPool.Count > 0 && recycleDropsDeviceCopies)
                {
                    GgmlBasicOps.Gemma4ResetDecodeCache();
                    GgmlBasicOps.Gemma4MoEResetDecodeCache();
                }
                CountDecodeGraphReset();
            }

            _holderGraphResetSuppressed++;
            try
            {
                foreach (var holder in toDispose)
                    DisposeHolder(holder);
                foreach (var holder in toPool)
                {
                    ResetHolderForReuse(holder);
                    (_holderPool ??= new List<Gemma4KvCacheHolder>(Math.Max(1, poolMax))).Add(holder);
                }
            }
            finally
            {
                _holderGraphResetSuppressed--;
            }
        }

        private void ResetHolderForReuse(Gemma4KvCacheHolder holder)
        {
            holder.SeqLen = 0;
            holder.HostDirty = false;
            var seen = new HashSet<Tensor>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l)) continue;
                if (holder.K[l] != null && seen.Add(holder.K[l])) ResetCacheTensor(holder.K[l]);
                if (holder.V?[l] != null && seen.Add(holder.V[l])) ResetCacheTensor(holder.V[l]);
            }
        }

        private void DisposeHolderPool()
        {
            if (_holderPool == null || _holderPool.Count == 0)
            {
                _holderPool = null;
                return;
            }
            // Taken out before anything is disposed: a throw part-way cannot leave a freed
            // holder where CreateFreshHolder would hand it out.
            var parked = _holderPool;
            _holderPool = null;
            if (IsGgmlBackend)
            {
                GgmlBasicOps.Gemma4ResetBatchedDecodeCache();
                GgmlBasicOps.Gemma4ResetMoEBatchedDecodeCache();
                CountDecodeGraphReset();
            }
            _holderGraphResetSuppressed++;
            try
            {
                foreach (var holder in parked)
                    DisposeHolder(holder);
            }
            finally
            {
                _holderGraphResetSuppressed--;
            }
        }

        /// <summary>Free the parked holders (prefix-cache mode keeps some), then the base trim. In the
        /// unattached mode the pool is empty and this is the base trim.</summary>
        public override void TrimIdleMemory()
        {
            int parked = _holderPool?.Count ?? 0;
            DisposeHolderPool();
            if (parked > 0)
                Console.WriteLine($"[memory] Gemma4: freed {parked} parked K/V holder(s)");
            base.TrimIdleMemory();
        }

        private long PooledHolderBytes()
        {
            long bytes = 0;
            if (_holderPool != null)
                foreach (var holder in _holderPool) bytes += HolderBytes(holder);
            return bytes;
        }

        /// <summary>The smaller of host and device headroom (both the host allocation and a GPU mirror
        /// have to fit), or -1 when neither is known.</summary>
        private long PoolSpareBytes()
        {
            long host = QueryPrefixCacheSpareBytes(ResourceClass.HostKv);
            long device = QueryPrefixCacheSpareBytes(ResourceClass.DeviceKv);
            if (host < 0) return device;
            if (device < 0) return host;
            return Math.Min(host, device);
        }

        /// <summary>Σ bytes of the unique K/V storages (a donor layer aliases its source and is counted once).</summary>
        private static long HolderBytes(Gemma4KvCacheHolder holder)
        {
            long bytes = 0;
            var seen = new HashSet<Storage>();
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null && seen.Add(t.Storage))
                            bytes = checked(bytes + t.Storage.ByteLength);
            return bytes;
        }

        private bool TryGetRetained(string payloadKey, out Gemma4KvCacheHolder holder)
        {
            holder = null;
            return payloadKey != null && _retainedFusedHolders != null
                && _retainedFusedHolders.TryGetValue(payloadKey, out holder)
                && !holder.Retired && holder.K != null;
        }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys =>
            _retainedFusedHolders == null ? Array.Empty<string>() : _retainedFusedHolders.Keys.ToArray();

        public int PrivateHolderCount => _fusedHolders?.Count ?? 0;

        public int PrimaryCacheLength => _activeFusedKey == null ? _cacheSeqLen : (_primaryHolder?.SeqLen ?? 0);

        /// <summary>Test visibility: whether a retained holder's device copy is newer than its host bytes.</summary>
        internal bool IsRetainedHostDirty(string payloadKey) => TryGetRetained(payloadKey, out var holder) && holder.HostDirty;

        /// <summary>Test visibility: holders parked by the pool.</summary>
        internal int PooledHolderCount => _holderPool?.Count ?? 0;
    }
}
