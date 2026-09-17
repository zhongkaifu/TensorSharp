// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// DeepSeek V4 / V4.1's radix prefix cache: donate-only native slots. Native
// reuse checks and retention budgets remain authoritative; reclaimed slots are
// reported to the tree through the attached payload sink.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        // Set by AttachPrefixCache: reclaimed slots are reported through it (DEC-23).
        private IPrefixPayloadSink _prefixCacheSink;

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            bool slots = SupportsRetainedFusedCache && SupportsPerSequenceFusedForward;
            return new PrefixCacheCapabilities
            {
                Class = FamilyClass.N,
                Readiness = PrefixCacheMode.Tree,
                NamespaceFingerprint = KVStateFingerprint,
                EndState = slots ? EndStateSupport.DonateOnly : EndStateSupport.None,
                CanCaptureCopy = false,
                AdoptPrimaryOnDisplacement = slots,
                PrimaryResident = true,
                MinRetainTokens = 32,
                // The native slot decides (SlotCanReuse), aligned to its compressor ratio; own scope only.
                Truncation = SupportsKVCacheTruncation ? TruncationKind.ModelDecides : TruncationKind.None,
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                RewindCapTokens = int.MaxValue,
                Pages = PageSupport.None,
                ReuseAcrossMediaSpan = false,
                Persistable = false,
                SubCapBytes = new ResourceVector
                {
                    NativeSlot = (long)Math.Min(_nativeRetentionBudget, (ulong)long.MaxValue),
                },
            };
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink)
        {
            lock (_sync)
                _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));
        }

        public void DetachPrefixCache()
        {
            lock (_sync)
                _prefixCacheSink = null;
        }

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return SupportsRetainedFusedCache
                && HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
        }

        /// <summary>The native slot decides (<see cref="CanReuseRetainedPrefix"/>: its head is
        /// <paramref name="payloadTokens"/> and the rewind to <paramref name="targetTokens"/> is exact).</summary>
        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
            => CanReuseRetainedPrefix(payloadKey, payloadTokens, targetTokens);

        /// <summary>A slot is never copied, so there is nothing to settle and no clone to allow.</summary>
        public bool SettleForCopy(string payloadKey) => false;

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            lock (_sync)
            {
                if (payloadKey == null || _retainedSlotByRequest == null
                    || !_retainedSlotByRequest.TryGetValue(payloadKey, out int slot)
                    || !GgmlDeepSeek4Native.SlotStatus(_handle, slot, out int head, out _, out _))
                    return default;
                // The slot's bytes are native (a full-context cache set plus its graph arena); the managed side
                // has no query for them yet, so the native SlotCanRetain budget bounds retention (M5e measures).
                return new PayloadFootprint(head, _maxContextLength, default, PositionDelta: 0);
            }
        }

        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens) => default;

        public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            lock (_sync)
            {
                foreach (string key in payloadKeys)
                    if (key != null) DiscardRetainedCache(key);
            }
        }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys
        {
            get
            {
                lock (_sync)
                    return _retainedSlotByRequest == null ? Array.Empty<string>() : _retainedSlotByRequest.Keys.ToArray();
            }
        }

        public int PrivateHolderCount
        {
            get
            {
                lock (_sync)
                    return _slotByRequest?.Count ?? 0;
            }
        }

        public int PrimaryCacheLength
        {
            get
            {
                lock (_sync)
                {
                    return _handle != IntPtr.Zero && _primarySlot >= 0
                           && GgmlDeepSeek4Native.SlotStatus(_handle, _primarySlot, out int head, out _, out _)
                        ? head : 0;
                }
            }
        }
    }
}
