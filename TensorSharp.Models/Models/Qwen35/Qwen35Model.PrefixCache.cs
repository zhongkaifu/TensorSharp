// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen 3.5 / 3.6's side of the radix prefix cache contract (DESIGN §6.4.1, class R).
//
// Inert by default. The capability record says Readiness=Legacy, so an engine
// never calls a state member here, and nothing below changes a code path the
// engine runs today: RetainSequenceCache is RetainSequenceCacheAs(id, id), and
// DiscardRetainedCaches, SettleForCopy and the measurements are new members.
//
// What the tree gets once a family PR raises the readiness (M5a):
//   * end states copied (capture, clone) or moved (donate) at zero copy, each
//     holding attention K/V and the matching GatedDeltaNet state together;
//   * exact-length materialization only (no truncation: the recurrence cannot
//     be rewound);
//   * batched releases that recycle into the existing holder pool and reset the
//     decode graphs at most once per batch (DEC-24);
//   * a settle that flushes a donated holder's arena slot, K/V mirrors, fused
//     decode state and verify state before it is cloned (G-12).
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class Qwen35Model : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        // Set by AttachPrefixCache: the prefix cache owns this model's reuse state.
        private IPrefixPayloadSink _prefixCacheSink;
        // >0 while a batched release has already reset the decode graphs.
        private int _holderGraphResetSuppressed;

        // ---------------------------------------------------------------- capabilities

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            bool holders = SupportsPerSequenceFusedForward;
            return new PrefixCacheCapabilities
            {
                Class = FamilyClass.R,
                Readiness = PrefixCacheMode.Legacy,
                NamespaceFingerprint = KVStateFingerprint,
                EndState = holders ? EndStateSupport.CopyAndDonate : EndStateSupport.None,
                // False under tensor parallelism (the cache lives on the ranks).
                CanCaptureCopy = holders && SupportsPrefixCheckpoints,
                AdoptPrimaryOnDisplacement = holders,
                PrimaryResident = true,
                MinRetainTokens = 32,
                Truncation = TruncationKind.None,
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                Pages = PageSupport.None,
                // DEC-11: false until M8 lands a rope delta and flips it (Phase 0's value).
                ReuseAcrossMediaSpan = SupportsReuseAcrossMediaSpan,
                Persistable = holders && SupportsRetainedCacheSerialization,
            };
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink)
            => _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---------------------------------------------------------------- end states

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return SupportsPerSequenceFusedForward
                && HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
        }

        /// <summary>A live holder of exactly <paramref name="payloadTokens"/> tokens, materialized at that
        /// length only: the GatedDeltaNet recurrence has no earlier state to rewind to.</summary>
        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
            => TryGetRetained(payloadKey, out var holder)
               && holder.CacheSeqLen == payloadTokens && targetTokens == payloadTokens;

        /// <summary>
        /// §6.1 for a hybrid holder: bind it (the arena slot is keyed by the holder's pointers), flush in the
        /// order <see cref="TryCheckpointActiveCache"/> uses — arena slot and K/V mirrors, fused-decode conv
        /// scratch and delta mirrors, verify-owned device state — record the flags, rebind the previous
        /// cache. The host copy is then current, so the device state is no longer marked resident: the
        /// holder's next fused decode re-seeds its device state from those same bytes.
        /// </summary>
        public bool SettleForCopy(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return false;
            if (!(holder.KvHostDirty || holder.GdnHostDirty || holder.ArenaStateResident || holder.FdStateResident))
                return true;
            if (!IsGgmlBackend || IsTensorParallel || _isRecurrent == null) return false;
            Qwen35KvCacheHolder previous = SnapshotActiveCache();
            LoadCacheHolder(holder);
            try
            {
                EnsureKvCacheHostSynchronized();
                EnsureFusedDecodeStateHostSynchronized();
                DrainDeviceRecurrentState();
                if (!_gdnStateHostDirty)
                    _fdStateResident = false;
                holder.KvHostDirty = _kvCacheHostDirty;
                holder.GdnHostDirty = _gdnStateHostDirty;
                holder.ArenaStateResident = _arenaStateResident;
                holder.FdStateResident = _fdStateResident;
            }
            finally
            {
                LoadCacheHolder(previous);
            }
            return !(holder.KvHostDirty || holder.GdnHostDirty || holder.ArenaStateResident || holder.FdStateResident);
        }

        /// <summary>Host bytes as <see cref="IdleHolderBytes"/> counts them, split into attention K/V
        /// (<see cref="ResourceVector.HostKv"/>) and recurrent state (<see cref="ResourceVector.StateSnapshot"/>),
        /// plus the device copies of a holder that was bound on a GPU backend.</summary>
        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return default;
            MeasureHolder(holder, out long kvBytes, out long stateBytes, out long deviceBytes);
            var vector = new ResourceVector
            {
                HostKv = kvBytes,
                StateSnapshot = stateBytes,
                DeviceKv = holder.DeviceMirrored ? deviceBytes : 0,
            };
            return new PayloadFootprint(holder.CacheSeqLen, holder.KvCapacity, vector, PositionDelta: 0);
        }

        /// <summary>What <see cref="DeepCopyHolder"/> allocates: attention rows sized to what the source
        /// holds plus a padded window, and a full set of recurrent state.</summary>
        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens)
        {
            if (!TryGetRetained(payloadKey, out var source)) return default;
            int rows = Math.Max(0, Math.Min(source.CacheSeqLen, source.KvCapacity));
            int cap = Math.Max(1, Math.Min(source.KvCapacity, CacheCapacityFor(rows)));
            long kv = 0;
            var seen = new HashSet<Storage>();
            foreach (Tensor[] set in new[] { source.K, source.V })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null && seen.Add(t.Storage))
                            kv += t.Sizes[1] <= 0 ? t.Storage.ByteLength : t.Storage.ByteLength / t.Sizes[1] * cap;
            MeasureHolder(source, out _, out long state, out _);
            if (source.Logits != null) state -= (long)source.Logits.Length * sizeof(float);   // a copy has no logits buffer
            return new ResourceVector { HostKv = kv, StateSnapshot = state };
        }

        /// <summary>The batched <see cref="DiscardRetainedCache"/> (DEC-24): recycle into the holder pool by
        /// <see cref="RecycleHolder"/>'s rules, and reset the decode graphs at most once for the whole batch,
        /// before its first disposal. Invalidation, pressure and reset dispose.</summary>
        public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            if (_retainedFusedHolders == null || payloadKeys.IsEmpty) return;
            List<Qwen35KvCacheHolder> released = null;
            foreach (string key in payloadKeys)
            {
                if (key != null && _retainedFusedHolders.Remove(key, out var holder))
                {
                    holder.Retired = true;
                    (released ??= new List<Qwen35KvCacheHolder>()).Add(holder);
                }
            }
            if (released == null) return;

            bool dispose = reason is ReleaseReason.Invalidated or ReleaseReason.Pressure or ReleaseReason.Reset;
            var toPool = new List<Qwen35KvCacheHolder>();
            var toDispose = new List<Qwen35KvCacheHolder>();
            bool trimPool = false;
            int poolMax = ExecutionOptions.FromEnvironment().KvHolderPoolMax;
            long? spare = GetCacheMemorySpareBytes();
            long pooledBytes = PooledHolderBytes();
            foreach (var holder in released)
            {
                // Its state is no longer observable: retire an arena mapping before the
                // holder's stable pointers can be reassigned or freed.
                DiscardArenaSlotForHolder(holder);
                if (dispose || trimPool || holder.DisposalStarted)
                {
                    toDispose.Add(holder);
                    continue;
                }
                long bytes = IdleHolderBytes(holder);
                if (!CanPoolIdleCache(bytes, pooledBytes, 0, 1, spare))
                {
                    // As RecycleHolder: headroom is gone, so nothing more is parked.
                    trimPool = true;
                    toDispose.AddRange(toPool);
                    toPool.Clear();
                    toDispose.Add(holder);
                    continue;
                }
                if (CanPoolIdleCache(bytes, pooledBytes, (_holderPool?.Count ?? 0) + toPool.Count, poolMax, spare))
                {
                    toPool.Add(holder);
                    pooledBytes += bytes;
                }
                else
                {
                    toDispose.Add(holder);
                }
            }

            if (IsGgmlBackend && (toDispose.Count > 0 || (trimPool && _holderPool is { Count: > 0 })))
            {
                GgmlBasicOps.Qwen35ResetDecodeCache();
                InvalidateVerifyCache();
                CountDecodeGraphReset();
            }
            _holderGraphResetSuppressed++;
            try
            {
                foreach (var holder in toDispose)
                    DisposeHolder(holder);
                foreach (var holder in toPool)
                {
                    InvalidateHolderDeviceCopiesForReuse(holder);
                    _holderPool ??= new List<Qwen35KvCacheHolder>(Math.Max(1, poolMax));
                    _holderPool.Add(holder);
                    holder.Retired = false;
                }
                if (trimPool)
                    TrimIdleMemory();
            }
            finally
            {
                _holderGraphResetSuppressed--;
            }
        }

        // ---------------------------------------------------------------- helpers

        /// <summary>
        /// Every allocation a holder owns, by class: attention K/V storages; the recurrent state (delta-state
        /// storages, host conv rings and their write indices, the fused-decode conv scratch — one float per
        /// ring float — and the holder's logits buffer); and what a bound holder mirrors on the device (K/V,
        /// delta state, conv scratch). Counted independently of <see cref="IdleHolderBytes"/>, which
        /// MeasureEndStateTests holds it to within 1%.
        /// </summary>
        private static void MeasureHolder(Qwen35KvCacheHolder holder, out long kvBytes, out long stateBytes, out long deviceBytes)
        {
            kvBytes = 0;
            long delta = 0, rings = 0;
            var seen = new HashSet<Storage>();
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null && seen.Add(t.Storage))
                            kvBytes = checked(kvBytes + t.Storage.ByteLength);
            if (holder.DeltaState != null)
                foreach (Tensor t in holder.DeltaState)
                    if (t != null && seen.Add(t.Storage))
                        delta = checked(delta + t.Storage.ByteLength);
            if (holder.ConvState != null)
                foreach (float[] ring in holder.ConvState)
                    if (ring != null) rings = checked(rings + (long)ring.Length * sizeof(float));
            long writeIndices = holder.ConvWriteIdx == null ? 0 : (long)holder.ConvWriteIdx.Length * sizeof(int);
            long scratch = holder.ConvScratch != IntPtr.Zero ? rings : 0;
            long logits = holder.Logits == null ? 0 : (long)holder.Logits.Length * sizeof(float);
            stateBytes = checked(delta + rings + writeIndices + scratch + logits);
            deviceBytes = checked(kvBytes + delta + scratch);
        }

        private bool TryGetRetained(string payloadKey, out Qwen35KvCacheHolder holder)
        {
            holder = null;
            return payloadKey != null && _retainedFusedHolders != null
                && _retainedFusedHolders.TryGetValue(payloadKey, out holder)
                && !holder.Retired && !holder.DisposalStarted && holder.K != null;
        }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys =>
            _retainedFusedHolders == null ? Array.Empty<string>() : _retainedFusedHolders.Keys.ToArray();

        public int PrivateHolderCount => _fusedHolders?.Count ?? 0;

        public int PrimaryCacheLength => _activeFusedKey == null ? _cacheSeqLen : (_primaryHolder?.CacheSeqLen ?? 0);

        // ---------------------------------------------------------------- test visibility

        /// <summary>The idle-holder byte count the holder pool budgets with, for a retained payload.</summary>
        internal long RetainedIdleHolderBytes(string payloadKey) => TryGetRetained(payloadKey, out var h) ? IdleHolderBytes(h) : -1;

        /// <summary>The same count for a request's private holder (a clone, before it is bound).</summary>
        internal long PrivateIdleHolderBytes(string requestId)
            => _fusedHolders != null && requestId != null && _fusedHolders.TryGetValue(requestId, out var h) ? IdleHolderBytes(h) : -1;

        /// <summary>Whether a retained holder has device state newer than, or not yet reflected in, its host bytes.</summary>
        internal bool IsRetainedDeviceAuthoritative(string payloadKey)
            => TryGetRetained(payloadKey, out var h) && (h.KvHostDirty || h.GdnHostDirty || h.ArenaStateResident || h.FdStateResident);
    }
}
