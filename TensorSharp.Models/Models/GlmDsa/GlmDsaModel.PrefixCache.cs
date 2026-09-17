// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GLM's radix prefix cache: donate-only native sequence slots. GLM-DSA can
// rewind where the model permits it; GLM5Next keeps exact recurrent states.
// Ownership moves are tested without weights through IGlmSlotStore.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel : IPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        /// <summary>A retained slot and the tokens it held when it was retained (a retained slot never runs).</summary>
        internal readonly record struct RetainedGlmSlot(int Slot, int Tokens);

        // payload key -> retained slot. Guarded by _nativeSync.
        private Dictionary<string, RetainedGlmSlot> _retainedSlots;
        private IPrefixPayloadSink _prefixCacheSink;

        /// <summary>The native slot operations the retention moves need.</summary>
        internal interface IGlmSlotStore
        {
            /// <summary>A new empty slot, or -1.</summary>
            int Alloc();
            bool Select(int slot);
            bool Free(int slot);
            /// <summary>Tokens in the active slot.</summary>
            int ActiveHead();
            /// <summary>Rewind the active slot to <paramref name="tokens"/>; false when refused.</summary>
            bool RewindActive(int tokens);
        }

        private readonly struct NativeGlmSlotStore : IGlmSlotStore
        {
            private readonly IntPtr _handle;
            public NativeGlmSlotStore(IntPtr handle) => _handle = handle;
            public int Alloc() => GgmlGlmNative.SlotAlloc(_handle);
            public bool Select(int slot) => GgmlGlmNative.SetActiveSlot(_handle, slot);
            public bool Free(int slot) => GgmlGlmNative.SlotFree(_handle, slot);
            public int ActiveHead() => GgmlGlmNative.NPast(_handle);
            public bool RewindActive(int tokens) => GgmlGlmNative.Rewind(_handle, tokens);
        }

        // ---------------------------------------------------------------- the ownership moves

        /// <summary>
        /// Retain <paramref name="requestId"/>'s slot under <paramref name="key"/> at <paramref name="length"/>
        /// tokens, rewinding first where <paramref name="canRewind"/>. A retained slot is never the active one:
        /// if it was, the primary is selected (allocated when the request had adopted it). Any refusal leaves
        /// every owner and the native selection as they were.
        /// </summary>
        internal static bool RetainSlot<TStore>(Dictionary<string, int> requests, Dictionary<string, RetainedGlmSlot> retained,
            string requestId, string key, int length, bool canRewind, ref int primary, ref string active, TStore store)
            where TStore : IGlmSlotStore
        {
            if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(key) || length <= 0 || requests == null
                || !requests.TryGetValue(requestId, out int slot) || retained.ContainsKey(key))
                return false;
            bool wasActive = string.Equals(active, requestId, StringComparison.Ordinal);
            int previous = wasActive ? slot : active == null ? primary : requests.TryGetValue(active, out int s) ? s : -1;
            if (!wasActive && (previous < 0 || !store.Select(slot))) return false;

            int head = store.ActiveHead();
            bool ok = head == length || (head > length && canRewind);
            // A request that adopted the primary needs a fresh one; allocate it before anything changes.
            int freshPrimary = -1;
            if (ok && wasActive && primary < 0)
            {
                freshPrimary = store.Alloc();
                ok = freshPrimary >= 0;
            }
            if (ok && head != length && !store.RewindActive(length))
            {
                if (freshPrimary >= 0) store.Free(freshPrimary);
                ok = false;
            }
            if (!ok)
            {
                if (!wasActive) store.Select(previous);
                return false;
            }

            if (wasActive)
            {
                if (freshPrimary >= 0) primary = freshPrimary;
                if (!store.Select(primary))
                    throw new InvalidOperationException($"GLM primary slot {primary} could not be selected after retaining slot {slot}.");
                active = null;
            }
            else if (!store.Select(previous))
            {
                throw new InvalidOperationException($"GLM slot {previous} could not be reselected after retaining slot {slot}.");
            }
            retained.Add(key, new RetainedGlmSlot(slot, length));
            requests.Remove(requestId);
            return true;
        }

        /// <summary>Donate a retained slot to <paramref name="requestId"/> (not bound yet).</summary>
        internal static bool DonateSlot(Dictionary<string, int> requests, Dictionary<string, RetainedGlmSlot> retained,
            string key, string requestId, out RetainedGlmSlot donated)
        {
            donated = default;
            if (string.IsNullOrEmpty(key) || string.IsNullOrEmpty(requestId) || retained == null
                || !retained.TryGetValue(key, out donated) || requests.ContainsKey(requestId))
                return false;
            requests.Add(requestId, donated.Slot);
            retained.Remove(key);
            return true;
        }

        /// <summary>Take an unbound donation back under its key (admission rollback).</summary>
        internal static bool ReturnSlot<TStore>(Dictionary<string, int> requests, Dictionary<string, RetainedGlmSlot> retained,
            string requestId, string key, ref int primary, ref string active, TStore store)
            where TStore : IGlmSlotStore
        {
            if (string.Equals(active, requestId, StringComparison.Ordinal)) return false;   // bound: no longer the payload
            if (requests == null || string.IsNullOrEmpty(requestId) || !requests.TryGetValue(requestId, out int slot)) return false;
            // Its head has not moved while it was unbound; read it through a select.
            int previous = active == null ? primary : requests.TryGetValue(active, out int s) ? s : -1;
            if (previous < 0 || !store.Select(slot)) return false;
            int head = store.ActiveHead();
            if (!store.Select(previous))
                throw new InvalidOperationException($"GLM slot {previous} could not be reselected after reading slot {slot}.");
            return RetainSlot(requests, retained, requestId, key, head, canRewind: false, ref primary, ref active, store);
        }

        /// <summary>Retain the active primary slot under <paramref name="key"/> and give the primary a fresh slot.</summary>
        internal static bool ConvertPrimarySlot<TStore>(Dictionary<string, RetainedGlmSlot> retained, string key, int length,
            ref int primary, ref string active, TStore store) where TStore : IGlmSlotStore
        {
            if (string.IsNullOrEmpty(key) || active != null || primary < 0 || retained.ContainsKey(key)) return false;
            if (store.ActiveHead() != length || length <= 0) return false;
            int fresh = store.Alloc();
            if (fresh < 0) return false;
            if (!store.Select(fresh))
            {
                store.Free(fresh);
                return false;
            }
            retained.Add(key, new RetainedGlmSlot(primary, length));
            primary = fresh;
            return true;
        }

        /// <summary>Free the listed retained slots (unknown keys ignored). A slot the native side refuses to
        /// free stays retained for a retry.</summary>
        internal static void ReleaseSlots<TStore>(Dictionary<string, RetainedGlmSlot> retained, ReadOnlySpan<string> keys, TStore store)
            where TStore : IGlmSlotStore
        {
            if (retained == null) return;
            foreach (string key in keys)
            {
                if (key == null || !retained.TryGetValue(key, out var entry)) continue;
                if (store.Free(entry.Slot)) retained.Remove(key);
            }
        }

        internal static bool CanDonate(Dictionary<string, RetainedGlmSlot> retained, string key, int tokens, int target, bool canRewind)
            => key != null && retained != null && retained.TryGetValue(key, out var entry)
               && entry.Tokens == tokens && target >= 0 && target <= tokens
               && (target == tokens || canRewind);

        // ---------------------------------------------------------------- IPrefixCacheModel

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            bool slots = UsesNativeExecutor;
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
                // glm-dsa rewinds rows exactly (own scope, capped at 16 until measured); glm5next's KDA cannot.
                Truncation = SupportsKVCacheTruncation ? TruncationKind.Any : TruncationKind.None,
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                RewindCapTokens = 16,
                Pages = PageSupport.None,
                ReuseAcrossMediaSpan = false,
                Persistable = false,
                // One retained slot is today's single live cache in memory (DEC-39); M5f measures its bytes.
                MaxRetainedNativeSlots = 1,
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
                if (!UsesNativeExecutor || _slotByRequest == null) return false;
                _retainedSlots ??= new Dictionary<string, RetainedGlmSlot>(StringComparer.Ordinal);
                if (!RetainSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, length, SupportsKVCacheTruncation,
                        ref _primarySlot, ref _activeSlotKey, new NativeGlmSlotStore(_native)))
                    return false;
                _cacheSeqLen = GgmlGlmNative.NPast(_native);   // the selection may have moved to the primary
                footprint = Footprint(_retainedSlots[payloadKey]);
                return true;
            }
        }

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            lock (_nativeSync)
            {
                if (!UsesNativeExecutor) return false;
                _retainedSlots ??= new Dictionary<string, RetainedGlmSlot>(StringComparer.Ordinal);
                if (!ConvertPrimarySlot(_retainedSlots, payloadKey, length, ref _primarySlot, ref _activeSlotKey, new NativeGlmSlotStore(_native)))
                    return false;
                _cacheSeqLen = GgmlGlmNative.NPast(_native);
                footprint = Footprint(_retainedSlots[payloadKey]);
                return true;
            }
        }

        public bool TryMaterialize(in MaterializeRequest request)
        {
            if (request.Op != MaterializeOp.Donate) return false;   // slots are moved, never copied
            lock (_nativeSync)
            {
                if (!UsesNativeExecutor || !CanDonate(_retainedSlots, request.PayloadKey, request.PayloadTokens, request.TargetTokens, SupportsKVCacheTruncation))
                    return false;
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                return DonateSlot(_slotByRequest, _retainedSlots, request.PayloadKey, request.TargetRequestId, out _);
            }
        }

        public bool TryReturnDonation(string requestId, string payloadKey)
        {
            lock (_nativeSync)
            {
                if (!UsesNativeExecutor || _slotByRequest == null) return false;
                _retainedSlots ??= new Dictionary<string, RetainedGlmSlot>(StringComparer.Ordinal);
                return ReturnSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, ref _primarySlot, ref _activeSlotKey,
                    new NativeGlmSlotStore(_native));
            }
        }

        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
        {
            lock (_nativeSync)
                return UsesNativeExecutor && CanDonate(_retainedSlots, payloadKey, payloadTokens, targetTokens, SupportsKVCacheTruncation);
        }

        public void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            lock (_nativeSync)
            {
                if (UsesNativeExecutor)
                    ReleaseSlots(_retainedSlots, payloadKeys, new NativeGlmSlotStore(_native));
            }
        }

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            lock (_nativeSync)
                return payloadKey != null && _retainedSlots != null && _retainedSlots.TryGetValue(payloadKey, out var entry)
                    ? Footprint(entry) : default;
        }

        // The slot's bytes (full-context MLA + indexer rows) are native; M5f measures them.
        private PayloadFootprint Footprint(RetainedGlmSlot entry) => new(entry.Tokens, _maxContextLength, default, PositionDelta: 0);

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
                    return UsesNativeExecutor && _activeSlotKey == null && _primarySlot >= 0 ? GgmlGlmNative.NPast(_native) : 0;
            }
        }
    }
}
