// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Donate-only prefix reuse over an executor's sequence slots (GLM-5.x, and Qwen3.8-Flash-Next on the
// direct-CUDA engine): a finished request's slot is retained under a payload key, moved to the
// request that continues it, and taken back when memory needs it. The ownership moves are pure
// bookkeeping over ISlotStore, so they are tested without weights.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    internal static class SlotRetention
    {
        /// <summary>A retained slot and the tokens it held when it was retained (a retained slot never runs).</summary>
        internal readonly record struct RetainedSlot(int Slot, int Tokens, bool IsCopy = false);

        /// <summary>The native slot operations the retention moves need.</summary>
        internal interface ISlotStore
        {
            /// <summary>A new empty slot, or -1.</summary>
            int Alloc();
            bool Select(int slot);
            bool Free(int slot);
            /// <summary>Tokens in the active slot.</summary>
            int ActiveHead();
            /// <summary>Rewind the active slot to <paramref name="tokens"/>; false when refused.</summary>
            bool RewindActive(int tokens);
            /// <summary>Empty the active slot (position 0, recurrent state cleared); false when it failed.</summary>
            bool ResetActive();
            /// <summary>Replace the list of retained slots, oldest first, that a graph which does not fit may free.</summary>
            bool SetReclaimable(int[] slots);
            /// <summary>Retained slots freed for a graph since the last call, into <paramref name="buffer"/>; how many.</summary>
            int TakeReclaimed(int[] buffer);
        }

        // ---------------------------------------------------------------- the ownership moves

        /// <summary>
        /// Retain <paramref name="requestId"/>'s slot under <paramref name="key"/> at <paramref name="length"/>
        /// tokens, rewinding first where <paramref name="canRewind"/>. A retained slot is never the active one:
        /// if it was, the primary is selected (allocated when the request had adopted it). Any refusal leaves
        /// every owner and the native selection as they were.
        /// </summary>
        internal static bool RetainSlot<TStore>(Dictionary<string, int> requests, IDictionary<string, RetainedSlot> retained,
            string requestId, string key, int length, bool canRewind, ref int primary, ref string active, TStore store)
            where TStore : ISlotStore
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
                    throw new InvalidOperationException($"Primary slot {primary} could not be selected after retaining slot {slot}.");
                active = null;
            }
            else if (!store.Select(previous))
            {
                throw new InvalidOperationException($"Slot {previous} could not be reselected after retaining slot {slot}.");
            }
            retained.Add(key, new RetainedSlot(slot, length));
            requests.Remove(requestId);
            return true;
        }

        /// <summary>Donate a retained slot to <paramref name="requestId"/> (not bound yet).</summary>
        internal static bool DonateSlot(Dictionary<string, int> requests, IDictionary<string, RetainedSlot> retained,
            string key, string requestId, out RetainedSlot donated)
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
        internal static bool ReturnSlot<TStore>(Dictionary<string, int> requests, IDictionary<string, RetainedSlot> retained,
            string requestId, string key, ref int primary, ref string active, TStore store)
            where TStore : ISlotStore
        {
            if (string.Equals(active, requestId, StringComparison.Ordinal)) return false;   // bound: no longer the payload
            if (requests == null || string.IsNullOrEmpty(requestId) || !requests.TryGetValue(requestId, out int slot)) return false;
            // Its head has not moved while it was unbound; read it through a select.
            int previous = active == null ? primary : requests.TryGetValue(active, out int s) ? s : -1;
            if (previous < 0 || !store.Select(slot)) return false;
            int head = store.ActiveHead();
            if (!store.Select(previous))
                throw new InvalidOperationException($"Slot {previous} could not be reselected after reading slot {slot}.");
            return RetainSlot(requests, retained, requestId, key, head, canRewind: false, ref primary, ref active, store);
        }

        /// <summary>Retain the active primary slot under <paramref name="key"/> and give the primary a fresh slot.</summary>
        internal static bool ConvertPrimarySlot<TStore>(IDictionary<string, RetainedSlot> retained, string key, int length,
            ref int primary, ref string active, TStore store) where TStore : ISlotStore
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
            retained.Add(key, new RetainedSlot(primary, length));
            primary = fresh;
            return true;
        }

        /// <summary>Free the listed retained slots (unknown keys ignored). A slot the native side refuses to
        /// free stays retained for a retry.</summary>
        internal static void ReleaseSlots<TStore>(IDictionary<string, RetainedSlot> retained, ReadOnlySpan<string> keys, TStore store)
            where TStore : ISlotStore
        {
            if (retained == null) return;
            foreach (string key in keys)
            {
                if (key == null || !retained.TryGetValue(key, out var entry)) continue;
                if (store.Free(entry.Slot)) retained.Remove(key);
            }
        }

        /// <summary>
        /// A new empty slot, freeing idle retained slots while device memory has no room for one: a retained
        /// slot keeps a finished conversation for its next turn, and a request that needs a slot now outranks
        /// it (DeepSeek's SlotForRequest does the same). Each freed key is added to
        /// <paramref name="reclaimed"/> for the caller to report through the sink (DEC-23). -1 when nothing is
        /// left to free; the caller then reports the missing capacity rather than failing the request.
        /// </summary>
        internal static int AllocReclaiming<TStore>(IDictionary<string, RetainedSlot> retained, TStore store, List<string> reclaimed)
            where TStore : ISlotStore
        {
            int slot = store.Alloc();
            while (slot < 0 && retained != null && retained.Count > 0)
            {
                string key = null;
                foreach (string candidate in retained.Keys) { key = candidate; break; }
                if (!store.Free(retained[key].Slot)) break;
                retained.Remove(key);
                reclaimed.Add(key);
                slot = store.Alloc();
            }
            return slot;
        }

        /// <summary>
        /// The primary slot, emptied, for a request that device memory has no other slot for. While requests
        /// run on their own slots nothing uses the primary: every fused step drops the prefix cache's claim on
        /// it and the executor's live-cache claim, and an N==1 owner that became concurrent has adopted it
        /// already (then <paramref name="primary"/> is -1 and this declines). -1 when declined, with the
        /// previous selection active again.
        /// </summary>
        internal static int TakeIdlePrimary<TStore>(Dictionary<string, int> requests, ref int primary, string active, TStore store)
            where TStore : ISlotStore
        {
            if (primary < 0) return -1;
            int previous = active != null && requests != null && requests.TryGetValue(active, out int held) ? held : primary;
            if (!store.Select(primary)) return -1;
            if (!store.ResetActive())
            {
                store.Select(previous);
                return -1;
            }
            int slot = primary;
            primary = -1;
            return slot;
        }

        /// <summary>
        /// Release the ACTIVE slot of <paramref name="requestId"/> while a request holds the primary: the
        /// released slot becomes the primary, emptied. Allocating a new primary first, with this slot still
        /// held, needed one slot more than memory holds whenever every slot was in use; that threw, the slot
        /// was never freed, and requests waiting for memory never got it. False when the slot could not be
        /// emptied: it is freed instead and the next single-stream step allocates a primary.
        /// </summary>
        internal static bool ReleaseActiveSlotAsPrimary<TStore>(Dictionary<string, int> requests, string requestId,
            ref int primary, ref string active, TStore store) where TStore : ISlotStore
        {
            if (primary >= 0 || !string.Equals(active, requestId, StringComparison.Ordinal)
                || requests == null || !requests.Remove(requestId, out int slot))
                return false;
            if (store.ResetActive())
            {
                primary = slot;
                active = null;
                return true;
            }
            store.Free(slot);
            return false;
        }

        /// <summary>
        /// Before a native call that may build a graph: tell the native side which retained slots it may free,
        /// oldest first, when a graph does not fit (glm_reclaim_for_graph). Every slot the admission rule let in
        /// may be retained at once, so a graph that grows past the reserve (a longer context, a larger batch) must
        /// be able to take their memory back rather than fail a running request. Pushes only a changed list;
        /// returns the list the native side now holds.
        /// </summary>
        internal static int[] SyncReclaimable<TStore>(IDictionary<string, RetainedSlot> retained, int[] synced, TStore store)
            where TStore : ISlotStore
        {
            int count = retained?.Count ?? 0;
            bool same = synced.Length == count;
            if (same && count > 0)
            {
                int i = 0;
                foreach (RetainedSlot entry in retained.Values)
                {
                    if (synced[i++] != entry.Slot) { same = false; break; }
                }
            }
            if (same) return synced;
            var slots = new int[count];
            if (count > 0)
            {
                int k = 0;
                foreach (RetainedSlot entry in retained.Values) slots[k++] = entry.Slot;
            }
            if (!store.SetReclaimable(slots))
                throw new InvalidOperationException("The executor refused the list of retained slots.");
            return slots;
        }

        /// <summary>After such a call: forget the retained slots the native side freed for a graph, adding each key
        /// to <paramref name="reclaimed"/> for the caller to report through the sink (DEC-23).</summary>
        internal static void DrainReclaimed<TStore>(IDictionary<string, RetainedSlot> retained, TStore store, int[] buffer,
            List<string> reclaimed) where TStore : ISlotStore
        {
            int n;
            while ((n = store.TakeReclaimed(buffer)) > 0)
            {
                for (int i = 0; i < n; i++)
                {
                    string key = null;
                    if (retained != null)
                    {
                        foreach (KeyValuePair<string, RetainedSlot> entry in retained)
                        {
                            if (entry.Value.Slot == buffer[i]) { key = entry.Key; break; }
                        }
                    }
                    if (key == null) continue;
                    retained.Remove(key);
                    reclaimed.Add(key);
                }
            }
        }

        internal static bool CanDonate(IDictionary<string, RetainedSlot> retained, string key, int tokens, int target, bool canRewind)
            => key != null && retained != null && retained.TryGetValue(key, out var entry)
               && entry.Tokens == tokens && target >= 0 && target <= tokens
               && (target == tokens || canRewind);
    }
}
