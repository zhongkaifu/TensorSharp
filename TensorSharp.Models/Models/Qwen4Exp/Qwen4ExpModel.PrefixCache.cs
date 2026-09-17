// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen 3.8 Flash Next's (qwen4exp) side of the radix prefix cache contract
// (DESIGN §6.4.2, class R).
//
// Inert by default. The capability record says Readiness=Legacy, so an engine
// never calls a state member here, and nothing below changes a code path the
// engine runs today:
//   * RetainSequenceCache is RetainSequenceCacheAs(id, id);
//   * the refuse-and-report mode (EnsureRetentionBudget refuses instead of
//     evicting, TrimIdleMemory reports what it frees, a checkpoint may be
//     donated) is active only after AttachPrefixCache;
//   * DiscardRetainedCaches, SettleForCopy and the measurements are new members.
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
                Readiness = PrefixCacheMode.Legacy,
                NamespaceFingerprint = KVStateFingerprint,
                EndState = holders ? EndStateSupport.CopyAndDonate : EndStateSupport.None,
                CanCaptureCopy = holders && SupportsPrefixCheckpoints,
                AdoptPrimaryOnDisplacement = holders,
                PrimaryResident = true,
                MinRetainTokens = 32,
                Truncation = TruncationKind.None,            // exact length only (IExactFusedCacheReuse)
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                Pages = PageSupport.None,
                // It stores an M-RoPE cache gap; no reference-position test exists yet (§15.2 Q6).
                ReuseAcrossMediaSpan = false,
                Persistable = false,
                SubCapBytes = new ResourceVector { DeviceKv = Math.Max(0, _retainedCacheBudgetBytes) },
            };
        }

        /// <summary>Tree mode: <see cref="EnsureRetentionBudget"/> refuses instead of evicting and
        /// <see cref="TrimIdleMemory"/> reports every holder it frees through <paramref name="sink"/>.</summary>
        public void AttachPrefixCache(IPrefixPayloadSink sink)
            => _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---------------------------------------------------------------- end states

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return SupportsRetainedFusedCache && SupportsPerSequenceFusedForward
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
