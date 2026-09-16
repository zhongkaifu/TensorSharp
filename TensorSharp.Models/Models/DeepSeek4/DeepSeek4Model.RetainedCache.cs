// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : IExactFusedCacheReuse
    {
        private Dictionary<string, int> _retainedSlotByRequest;
        // Non-null means native selection belongs to this retained entry, NOT
        // the primary. _activeSlotKey is null in that state. The primary may be
        // absent after the previous request adopted it.
        private string _selectedRetainedKey;
        private readonly bool _nativeRetentionEnabled =
            Environment.GetEnvironmentVariable("TS_DSV41_RETAINED_CACHE") == "1";
        private readonly ulong _nativeRetentionBudget = RetentionBudgetFromEnvironment();

        // Ownership has already committed when these diagnostics run. A broken
        // stderr sink or exhausted formatting allocation must not report a
        // successful transfer as a refusal to the scheduler.
        private static void TraceRetainedCommit(string operation, string request, int slot, int count, string source = null)
        {
            try
            {
                if (operation == "retain")
                    Console.Error.WriteLine($"[dsv41 retained] retain request={request} slot={slot} count={count}");
                else if (operation == "rebind")
                    Console.Error.WriteLine($"[dsv41 retained] rebind source={source} request={request} slot={slot}");
                else
                    Console.Error.WriteLine($"[dsv41 retained] discard request={request} primary={slot} count={count}");
            }
            catch (OutOfMemoryException) { }
            catch (System.IO.IOException) { }
            catch (ObjectDisposedException) { }
        }

        private static ulong RetentionBudgetFromEnvironment()
        {
            string text = Environment.GetEnvironmentVariable("TS_DSV41_RETAINED_CACHE_MB");
            if (text == null) return 2048UL * 1024 * 1024;
            return ulong.TryParse(text, out ulong mb) && mb <= ulong.MaxValue / (1024 * 1024)
                ? mb * 1024 * 1024 : 0; // Invalid/zero budget declines retention.
        }

        public bool SupportsRetainedFusedCache => _nativeRetentionEnabled
            && _handle != IntPtr.Zero && _truncateAlign > 0 && _nativeDsparkBlock == 0;
        public bool SupportsExactFusedCacheReuse => SupportsRetainedFusedCache;

        internal interface INativeSlotRetention : INativeSlotRelease
        {
            bool Status(int slot, out int head, out bool healthy);
            bool CanRetain(int slot, int retainedCount, ulong budget);
            bool ReleaseGraphs(int slot);
        }

        private readonly struct NativeSlotRetention : INativeSlotRetention
        {
            private readonly IntPtr _native;
            public NativeSlotRetention(IntPtr native) => _native = native;
            public bool Reset() => GgmlDeepSeek4Native.ResetChecked(_native);
            public bool Select(int slot) => GgmlDeepSeek4Native.SetActiveSlot(_native, slot);
            public bool Free(int slot) => GgmlDeepSeek4Native.SlotFree(_native, slot);
            public bool Status(int slot, out int head, out bool healthy)
                => GgmlDeepSeek4Native.SlotStatus(_native, slot, out head, out _, out healthy);
            public bool CanRetain(int slot, int retainedCount, ulong budget)
                => GgmlDeepSeek4Native.SlotCanRetain(_native, slot, retainedCount, budget);
            public bool ReleaseGraphs(int slot) => GgmlDeepSeek4Native.SlotReleaseGraphs(_native, slot);
        }

        internal static bool RetainNativeSequence<TNative>(Dictionary<string, int> requests,
            Dictionary<string, int> retained, string key, ref string active, ref string selectedRetained,
            ulong budget, TNative native) where TNative : INativeSlotRetention
        {
            if (string.IsNullOrEmpty(key) || requests == null || !requests.TryGetValue(key, out int slot)
                || retained.ContainsKey(key) || !native.Status(slot, out int head, out bool healthy)
                || !healthy || head <= 0 || !native.CanRetain(slot, retained.Count, budget)) return false;
            // Add first: a managed allocation failure leaves the old owner intact.
            try { retained.Add(key, slot); }
            catch (OutOfMemoryException) { return false; }
            requests.Remove(key);
            if (active == key) { active = null; selectedRetained = key; }
            return true;
        }

        internal static bool RebindNativeSequence<TNative>(Dictionary<string, int> requests,
            Dictionary<string, int> retained, string oldKey, string newKey,
            ref string active, ref string selectedRetained, TNative native)
            where TNative : INativeSlotRetention
        {
            if (string.IsNullOrEmpty(oldKey) || string.IsNullOrEmpty(newKey) || retained == null
                || !retained.TryGetValue(oldKey, out int slot) || requests.ContainsKey(newKey)
                || !native.Status(slot, out _, out bool healthy) || !healthy) return false;
            try { requests.Add(newKey, slot); }
            catch (OutOfMemoryException) { return false; }
            retained.Remove(oldKey);
            if (selectedRetained == oldKey) { selectedRetained = null; active = newKey; }
            return true;
        }

        internal static void DiscardRetainedNativeSequence<TNative>(Dictionary<string, int> retained,
            string key, ref int primary, ref string active, ref string selectedRetained, TNative native)
            where TNative : INativeSlotRetention
        {
            if (string.IsNullOrEmpty(key) || retained == null || !retained.TryGetValue(key, out int slot)) return;
            if (selectedRetained == key)
            {
                if (primary < 0)
                {
                    // An active native slot cannot be freed. Reclaim it without
                    // allocating a replacement; release its captured arenas first.
                    if (!native.ReleaseGraphs(slot)) throw new InvalidOperationException("DSV4 retained graph release failed.");
                    if (!native.Reset())
                        throw new InvalidOperationException("DSV4 retained reset failed; slot remains quarantined.");
                    if (!native.Status(slot, out int head, out bool healthy) || !healthy || head != 0)
                        throw new InvalidOperationException("DSV4 retained slot reset failed; slot remains quarantined.");
                    primary = slot;
                    selectedRetained = null;
                    active = null;
                    retained.Remove(key);
                    return;
                }
                if (!native.Select(primary)) throw new InvalidOperationException("DSV4 retained eviction could not select primary.");
                selectedRetained = null;
                active = null;
            }
            if (!native.Free(slot)) throw new InvalidOperationException("DSV4 retained slot could not be freed.");
            retained.Remove(key);
        }

        public bool RetainSequenceCache(string requestId)
        {
            lock (_sync)
            {
                if (!SupportsRetainedFusedCache) return false;
                try { _retainedSlotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal); }
                catch (OutOfMemoryException) { return false; }
                bool kept = RetainNativeSequence(_slotByRequest, _retainedSlotByRequest, requestId,
                    ref _activeSlotKey, ref _selectedRetainedKey, _nativeRetentionBudget, new NativeSlotRetention(_handle));
                if (kept) TraceRetainedCommit("retain", requestId, _retainedSlotByRequest[requestId], _retainedSlotByRequest.Count);
                return kept;
            }
        }

        public bool TryRebindRetainedCache(string oldRequestId, string newRequestId)
        {
            lock (_sync)
            {
                if (!SupportsRetainedFusedCache) return false;
                try { _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal); }
                catch (OutOfMemoryException) { return false; }
                bool rebound = RebindNativeSequence(_slotByRequest, _retainedSlotByRequest, oldRequestId, newRequestId,
                    ref _activeSlotKey, ref _selectedRetainedKey, new NativeSlotRetention(_handle));
                if (rebound) TraceRetainedCommit("rebind", newRequestId, _slotByRequest[newRequestId], 0, oldRequestId);
                return rebound;
            }
        }

        public void DiscardRetainedCache(string requestId)
        {
            lock (_sync)
            {
                if (_handle == IntPtr.Zero) return;
                bool owned = _retainedSlotByRequest?.ContainsKey(requestId ?? "") == true;
                DiscardRetainedNativeSequence(_retainedSlotByRequest, requestId, ref _primarySlot,
                    ref _activeSlotKey, ref _selectedRetainedKey, new NativeSlotRetention(_handle));
                if (owned) TraceRetainedCommit("discard", requestId, _primarySlot, _retainedSlotByRequest.Count);
            }
        }

        public bool CanReuseLivePrefix(int cachedTokenCount, int targetTokenCount)
        {
            lock (_sync)
            {
                // Live-primary metadata must never refer to an idle retained
                // holder or a checked-out request selected by an earlier step.
                return SupportsExactFusedCacheReuse && _activeSlotKey == null && _selectedRetainedKey == null
                    && _primarySlot >= 0 && GgmlDeepSeek4Native.SlotCanReuse(_handle, _primarySlot, cachedTokenCount, targetTokenCount);
            }
        }

        public bool CanReuseRetainedPrefix(string retainedKey, int cachedTokenCount, int targetTokenCount)
        {
            lock (_sync)
            {
                return SupportsExactFusedCacheReuse && retainedKey != null && _retainedSlotByRequest != null
                    && _retainedSlotByRequest.TryGetValue(retainedKey, out int slot)
                    && GgmlDeepSeek4Native.SlotCanReuse(_handle, slot, cachedTokenCount, targetTokenCount);
            }
        }

        // Reuse an idle allocation for an unrelated sequence before asking for
        // more full-context buffers. Scheduler metadata may remain in its LRU,
        // but the slot-aware query rejects that evicted key thereafter.
        private bool ReclaimRetainedPrimary()
        {
            if (_primarySlot >= 0 || _retainedSlotByRequest == null || _retainedSlotByRequest.Count == 0) return false;
            string key = _selectedRetainedKey;
            if (key == null)
                foreach (var candidate in _retainedSlotByRequest) { key = candidate.Key; break; }
            int slot = _retainedSlotByRequest[key];
            if (_selectedRetainedKey != key)
            {
                if (!GgmlDeepSeek4Native.SetActiveSlot(_handle, slot))
                    throw new InvalidOperationException("DSV4 idle slot could not be selected for reclamation.");
                _activeSlotKey = null;
                _selectedRetainedKey = key;
            }
            DiscardRetainedCache(key);
            return true;
        }
    }
}
