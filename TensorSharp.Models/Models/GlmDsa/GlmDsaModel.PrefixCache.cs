// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GLM's radix prefix cache: donate-only native sequence slots. GLM-DSA can
// rewind where the model permits it; GLM5Next keeps exact recurrent states.
// The ownership moves are SlotRetention's, tested without weights.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel : IPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        // payload key -> retained slot, oldest first (the order memory takes them back in). Guarded by _nativeSync.
        private OrderedDictionary<string, SlotRetention.RetainedSlot> _retainedSlots;
        // The retained slots the native side was last told it may free, oldest first (SyncReclaimable).
        private int[] _reclaimableSynced = Array.Empty<int>();
        private readonly int[] _reclaimedIds = new int[16];
        private IPrefixPayloadSink _prefixCacheSink;

        private readonly struct ExecutorGlmSlotStore : SlotRetention.ISlotStore
        {
            private readonly IGlmExecutor _exec;
            public ExecutorGlmSlotStore(IGlmExecutor exec) => _exec = exec;
            public int Alloc() => _exec.SlotAlloc();
            public bool Select(int slot) => _exec.SetActiveSlot(slot);
            public bool Free(int slot) => _exec.SlotFree(slot);
            public int ActiveHead() => _exec.NPast;
            public bool RewindActive(int tokens) => _exec.Rewind(tokens);
            public bool ResetActive() => _exec.ResetChecked();
            public bool SetReclaimable(int[] slots) => _exec.SetReclaimableSlots(slots);
            public int TakeReclaimed(int[] buffer) => _exec.TakeReclaimedSlots(buffer);
        }

        /// <summary>Run <see cref="SyncReclaimable"/> for this model; under _nativeSync.</summary>
        private void BeforeGraphCall()
        {
            if (_exec == null) return;
            _reclaimableSynced = SlotRetention.SyncReclaimable(_retainedSlots, _reclaimableSynced, new ExecutorGlmSlotStore(_exec));
        }

        /// <summary>Run <see cref="DrainReclaimed"/> for this model and tell the prefix cache before its next tree read;
        /// under _nativeSync, after the call whether or not it succeeded.</summary>
        private void AfterGraphCall()
        {
            if (_exec == null) return;
            var reclaimed = new List<string>(0);
            SlotRetention.DrainReclaimed(_retainedSlots, new ExecutorGlmSlotStore(_exec), _reclaimedIds, reclaimed);
            foreach (string key in reclaimed)
                _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
        }

        // ---------------------------------------------------------------- IPrefixCacheModel

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            bool slots = UsesNativeExecutor && !IsDistributedGlm;
            return new PrefixCacheCapabilities
            {
                Class = FamilyClass.N,
                NamespaceFingerprint = KVStateFingerprint,
                EndState = slots ? EndStateSupport.DonateOnly : EndStateSupport.None,
                CanCaptureCopy = false,
                AdoptPrimaryOnDisplacement = slots,
                PrimaryResident = true,
                MinRetainTokens = 32,
                // glm-dsa rewinds rows exactly (own scope, capped at 16 until measured); glm5next's KDA cannot.
                Truncation = SupportsKVCacheTruncation ? TruncationKind.Any : TruncationKind.None,
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                RewindCapTokens = 16,
                Pages = PageSupport.None,
                ReuseAcrossMediaSpan = false,
                Persistable = false,
                // No count of its own: each running request retains its slot for its conversation's next turn,
                // and memory takes them back oldest first when a new slot or a graph needs it (AllocReclaiming,
                // glm_reclaim_for_graph). A cap of 1 left all but one of N parallel conversations re-prefilling.
                MaxRetainedNativeSlots = 0,
            };
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink)
        {
            lock (_nativeSync)
                _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));
        }

        public void DetachPrefixCache()
        {
            lock (_nativeSync)
                _prefixCacheSink = null;
        }

        public bool TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            lock (_nativeSync)
            {
                if (IsDistributedGlm || !UsesNativeExecutor || _slotByRequest == null) return false;
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                if (!SlotRetention.RetainSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, length, SupportsKVCacheTruncation,
                        ref _primarySlot, ref _activeSlotKey, new ExecutorGlmSlotStore(_exec)))
                    return false;
                _cacheSeqLen = _exec.NPast;   // the selection may have moved to the primary
                footprint = Footprint(_retainedSlots[payloadKey]);
                return true;
            }
        }

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            lock (_nativeSync)
            {
                if (IsDistributedGlm || !UsesNativeExecutor) return false;
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                if (!SlotRetention.ConvertPrimarySlot(_retainedSlots, payloadKey, length, ref _primarySlot, ref _activeSlotKey, new ExecutorGlmSlotStore(_exec)))
                    return false;
                _cacheSeqLen = _exec.NPast;
                footprint = Footprint(_retainedSlots[payloadKey]);
                return true;
            }
        }

        public bool TryMaterialize(in MaterializeRequest request)
        {
            if (request.Op != MaterializeOp.Donate) return false;   // slots are moved, never copied
            lock (_nativeSync)
            {
                if (IsDistributedGlm || !UsesNativeExecutor || !SlotRetention.CanDonate(_retainedSlots, request.PayloadKey, request.PayloadTokens, request.TargetTokens, SupportsKVCacheTruncation))
                    return false;
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                return SlotRetention.DonateSlot(_slotByRequest, _retainedSlots, request.PayloadKey, request.TargetRequestId, out _);
            }
        }

        public bool TryReturnDonation(string requestId, string payloadKey)
        {
            lock (_nativeSync)
            {
                if (IsDistributedGlm || !UsesNativeExecutor || _slotByRequest == null) return false;
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                return SlotRetention.ReturnSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, ref _primarySlot, ref _activeSlotKey,
                    new ExecutorGlmSlotStore(_exec));
            }
        }

        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
        {
            lock (_nativeSync)
                return !IsDistributedGlm && UsesNativeExecutor && SlotRetention.CanDonate(_retainedSlots, payloadKey, payloadTokens, targetTokens, SupportsKVCacheTruncation);
        }

        public void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            lock (_nativeSync)
            {
                if (UsesNativeExecutor)
                    SlotRetention.ReleaseSlots(_retainedSlots, payloadKeys, new ExecutorGlmSlotStore(_exec));
            }
        }

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            lock (_nativeSync)
                return payloadKey != null && _retainedSlots != null && _retainedSlots.TryGetValue(payloadKey, out var entry)
                    ? Footprint(entry) : default;
        }

        // The slot's bytes (full-context MLA + indexer rows) are native; M5f measures them.
        private PayloadFootprint Footprint(SlotRetention.RetainedSlot entry) => new(entry.Tokens, _maxContextLength, default, PositionDelta: 0);

        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens) => default;
        public bool TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId) => false;
        public bool TryExport(string payloadKey, System.IO.Stream destination) => false;

        public bool TryImport(string payloadKey, int tokens, System.IO.Stream source, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public bool TryBeginImport(int tokens, out object importTicket)
        {
            importTicket = null;
            return false;
        }

        public bool RunImportRead(object importTicket, System.IO.Stream source) => false;

        public bool TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public void AbortImport(object importTicket) { }

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);
        public void QuerySpareBytes(long[] perClass) => QueryPrefixCacheSpareBytes(perClass);

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys
        {
            get
            {
                lock (_nativeSync)
                    return _retainedSlots == null ? Array.Empty<string>() : _retainedSlots.Keys.ToArray();
            }
        }

        public int PrivateHolderCount
        {
            get
            {
                lock (_nativeSync)
                    return _slotByRequest?.Count ?? 0;
            }
        }

        /// <summary>The primary's tokens while it is the active slot (reading another slot would change the selection).</summary>
        public int PrimaryCacheLength
        {
            get
            {
                lock (_nativeSync)
                    return UsesNativeExecutor && _activeSlotKey == null && _primarySlot >= 0 ? _exec.NPast : 0;
            }
        }
    }
}
