// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen4Exp's radix prefix cache: complete holders copied or donated at exact
// lengths. The tree owns retention and eviction after AttachPrefixCache; model
// budgets may refuse new payloads and report unavoidable reclamation.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        // Set by AttachPrefixCache: the prefix cache owns retention and eviction (DEC-23).
        private IPrefixPayloadSink _prefixCacheSink;
        // >0 while a batched release has already freed the native state entries.
        private int _holderSeqStateReleaseSuppressed;

        // ---------------------------------------------------------------- capabilities

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            bool holders = SupportsRetainedFusedCache && SupportsPerSequenceFusedForward;
            return new PrefixCacheCapabilities
            {
                Class = FamilyClass.R,
                NamespaceFingerprint = KVStateFingerprint,
                EndState = holders ? EndStateSupport.CopyAndDonate : EndStateSupport.None,
                CanCaptureCopy = holders && SupportsPrefixCheckpoints,
                AdoptPrimaryOnDisplacement = holders,
                DeferPrimaryConversion = holders,
                PrimaryResident = true,
                MinRetainTokens = 32,
                Truncation = TruncationKind.None,            // exact length only (IExactFusedCacheReuse)
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                Pages = PageSupport.None,
                // Complete holders preserve the rotary gap and QSA position history,
                // so identical media can be continued at the exact retained length.
                ReuseAcrossMediaSpan = SupportsReuseAcrossMediaSpan,
                Persistable = false,
                // Unset, the tree's own half-spare cap bounds holders (0 = no sub-cap), as it does Qwen 3.5's.
                SubCapBytes = new ResourceVector
                {
                    DeviceKv = _retainedCacheBudgetBytes >= 0 ? _retainedCacheBudgetBytes
                        : GpuMemoryBudget.TryGetReservationSpareBytes(_backend, Math.Max(1, LayerSplitDegree), out _)
                            ? 0 : UnmeasuredRetainedCacheBudgetBytes,
                },
            };
        }

        /// <summary>Tree mode: <see cref="EnsureRetentionBudget"/> refuses instead of evicting and
        /// <see cref="TrimIdleMemory"/> reports every holder it frees through <paramref name="sink"/>.</summary>
        public void AttachPrefixCache(IPrefixPayloadSink sink)
            => _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));

        public void DetachPrefixCache() => _prefixCacheSink = null;

        /// <summary>Device classes under a layer split report the tightest GPU of the
        /// split: a holder's KV and recurrent state are spread across all of them.</summary>
        public long QuerySpareBytes(ResourceClass cls)
            => LayerSplitDegree > 1 && cls is ResourceClass.DeviceKv or ResourceClass.StateSnapshot or ResourceClass.NativeSlot
                ? (GpuMemoryBudget.TryGetReservationSpareBytes(_backend, LayerSplitDegree, out long spare) ? Math.Max(0, spare) : -1)
                : QueryPrefixCacheSpareBytes(cls);

        // ---------------------------------------------------------------- end states

        /// <summary>The complete live primary's eventual holder footprint, without
        /// allocating a replacement or downloading its authoritative native state.</summary>
        public bool TryMeasurePrimaryEndState(int length, out PayloadFootprint footprint)
        {
            footprint = default;
            if (_activeFusedKey != null || _kCache == null || _vCache == null
                || !CompleteSpanPathAvailable || _headKDim != _headVDim
                || _specStateFailed || length <= 0 || _cacheSeqLen != length || length > _kvCacheCapacity)
                return false;
            footprint = MeasureHolderFootprint(SnapshotActiveCache());
            return true;
        }

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            if (!SupportsPerSequenceFusedForward || string.IsNullOrEmpty(payloadKey)
                || HasFusedSequenceCache(payloadKey) || !TryMeasurePrimaryEndState(length, out var measured))
                return false;
            // Adoption needs an empty replacement cache. Do not allocate it when
            // the existing live state already cannot fit the retention budget.
            // RetainSequenceCacheAs rechecks after allocation because headroom
            // can shrink and the prefix cache remains responsible for eviction.
            long retainedBytes = checked(measured.Bytes.HostKv + measured.Bytes.StateSnapshot);
            return EnsureRetentionBudget(retainedBytes, "converting live primary " + payloadKey)
                && HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
        }

        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
            => CanReuseRetainedPrefix(payloadKey, payloadTokens, targetTokens);

        /// <summary>A no-op: <see cref="DeepCopyHolder"/> downloads a retained holder's device state itself
        /// (<see cref="SyncHolderKvToHost"/> and the native state export) before it copies.</summary>
        public bool SettleForCopy(string payloadKey) => TryGetRetained(payloadKey, out _);

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return default;
            return MeasureHolderFootprint(holder);
        }

        private PayloadFootprint MeasureHolderFootprint(Qwen4ExpKvCacheHolder holder)
        {
            long total = RetainedHolderBytes(holder);
            long kv = KvStorageBytes(holder, scaleToCapacity: -1);
            var vector = new ResourceVector
            {
                HostKv = kv,
                StateSnapshot = total - kv,
                // The complete span path keeps every cache and state entry device-resident.
                DeviceKv = holder.DeviceMirrored ? total : 0,
            };
            return new PayloadFootprint(holder.CacheSeqLen, holder.KvCapacity, vector, PositionDelta: 0);
        }

        /// <summary><see cref="DeepCopyHolder"/>'s sizing: attention K/V and QSA raw keys to the rows the
        /// source holds (256-aligned, at least 256), everything else as the source has it.</summary>
        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens)
        {
            if (!TryGetRetained(payloadKey, out var source)) return default;
            int rows = Math.Max(0, Math.Min(source.CacheSeqLen, source.KvCapacity));
            int cap = Math.Min(source.KvCapacity, CopyCapacityFor(rows, _maxContextLength));
            long total = RetainedHolderBytes(source);
            long kvNow = KvStorageBytes(source, scaleToCapacity: -1);
            long kvCopy = KvStorageBytes(source, scaleToCapacity: cap);
            return new ResourceVector { HostKv = kvCopy, StateSnapshot = total - kvNow };
        }

        /// <summary>The batched <see cref="DiscardRetainedCache"/> (DEC-24): one native release for every
        /// holder's state entries (it drops every cached graph once), then each holder's tensors.</summary>
        public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            if (_retainedFusedHolders == null || payloadKeys.IsEmpty) return;
            List<Qwen4ExpKvCacheHolder> released = null;
            foreach (string key in payloadKeys)
            {
                if (key != null && _retainedFusedHolders.Remove(key, out var holder))
                {
                    holder.Retired = true;
                    (released ??= new List<Qwen4ExpKvCacheHolder>()).Add(holder);
                }
            }
            if (released == null) return;
            if (IsGgmlBackend)
            {
                var keys = new List<IntPtr>();
                foreach (var holder in released)
                    if (!holder.Disposed) keys.AddRange(HolderStateKeys(holder));
                if (keys.Count > 0)
                {
                    GgmlBasicOps.Qwen4ExpReleaseSeqState(keys.ToArray());
                    CountDecodeGraphReset();
                }
            }
            _holderSeqStateReleaseSuppressed++;
            try
            {
                foreach (var holder in released)
                {
                    DisposeHolder(holder);
                    ReleaseSlotBase(holder.SlotBase);
                }
            }
            finally
            {
                _holderSeqStateReleaseSuppressed--;
            }
        }

        // ---------------------------------------------------------------- helpers

        /// <summary>Σ attention K/V and QSA raw-key storages; with <paramref name="scaleToCapacity"/> ≥ 0,
        /// what they would be at that row capacity.</summary>
        private static long KvStorageBytes(Qwen4ExpKvCacheHolder holder, int scaleToCapacity)
        {
            long bytes = 0;
            var seen = new HashSet<Storage>();
            foreach (Tensor[] set in new[] { holder.K, holder.V, holder.IdxK })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null && seen.Add(t.Storage))
                            bytes = checked(bytes + (scaleToCapacity < 0 || t.Sizes[1] <= 0
                                ? t.Storage.ByteLength
                                : t.Storage.ByteLength / t.Sizes[1] * scaleToCapacity));
            return bytes;
        }

        private bool TryGetRetained(string payloadKey, out Qwen4ExpKvCacheHolder holder)
        {
            holder = null;
            return payloadKey != null && _retainedFusedHolders != null
                && _retainedFusedHolders.TryGetValue(payloadKey, out holder)
                && !holder.Retired && !holder.Disposed && holder.K != null;
        }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys =>
            _retainedFusedHolders == null ? Array.Empty<string>() : _retainedFusedHolders.Keys.ToArray();

        public int PrivateHolderCount => _fusedHolders?.Count ?? 0;

        public int PrimaryCacheLength => _activeFusedKey == null ? _cacheSeqLen : (_primaryHolder?.CacheSeqLen ?? 0);
    }
}
