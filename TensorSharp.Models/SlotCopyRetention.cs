// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models;

/// <summary>Copies a slot into a new physical owner before publishing its
/// metadata. The store preserves the source and active selection; the caller
/// serializes dictionary ownership changes. No reclaim or future credit is
/// involved in these operations.</summary>
internal static class SlotCopyRetention
{
    internal interface IStore
    {
        /// <summary>A new, independently owned slot, or -1 on refusal.</summary>
        int Alloc();
        /// <summary>Copy exactly the source's expected head without changing
        /// the source or active selection. A refusal may leave destination
        /// storage partially written; the caller must still free it.</summary>
        bool CopyInto(int src, int dst, int expectedHead);
        /// <summary>Actual owned payload bytes; zero for an unknown slot.</summary>
        long Bytes(int slot);
        /// <summary>Release the physical owner. False retains its payload and
        /// budget ownership in the store, including unpublished slots.</summary>
        bool Free(int slot);
    }

    internal static bool Capture(IStore store, OrderedDictionary<string, SlotRetention.RetainedSlot> retained,
        string key, int source, int tokens, out long bytes)
    {
        bytes = 0;
        if (store == null || retained == null || string.IsNullOrWhiteSpace(key)
            || source < 0 || tokens < 0 || retained.ContainsKey(key) || store.Bytes(source) <= 0)
            return false;

        // Metadata growth must precede the new physical payload. An unexpected
        // publication exception still takes the same physical cleanup path.
        retained.EnsureCapacity(checked(retained.Count + 1));
        int destination = store.Alloc();
        if (destination < 0) return false;
        bool published = false;
        Exception? failure = null;
        try
        {
            if (!store.CopyInto(source, destination, tokens)) return false;
            long actual = store.Bytes(destination);
            if (actual <= 0) return false;
            retained.Add(key, new(destination, tokens, IsCopy: true));
            bytes = actual;
            published = true;
            return true;
        }
        catch (Exception error)
        {
            failure = error;
            throw;
        }
        finally
        {
            if (!published) FreeUnpublished(store, destination, failure);
        }
    }

    internal static bool Clone(IStore store, Dictionary<string, int> requests,
        string requestId, int source, int tokens)
    {
        if (store == null || requests == null || string.IsNullOrWhiteSpace(requestId)
            || source < 0 || tokens < 0 || requests.ContainsKey(requestId) || store.Bytes(source) <= 0)
            return false;

        requests.EnsureCapacity(checked(requests.Count + 1));
        int destination = store.Alloc();
        if (destination < 0) return false;
        bool published = false;
        Exception? failure = null;
        try
        {
            if (!store.CopyInto(source, destination, tokens) || store.Bytes(destination) <= 0) return false;
            requests.Add(requestId, destination);
            published = true;
            return true;
        }
        catch (Exception error)
        {
            failure = error;
            throw;
        }
        finally
        {
            if (!published) FreeUnpublished(store, destination, failure);
        }
    }

    private static void FreeUnpublished(IStore store, int slot, Exception? failure)
    {
        bool freed;
        try { freed = store.Free(slot); }
        catch (Exception error)
        {
            throw CleanupFailure(slot, error);
        }
        if (!freed) throw CleanupFailure(slot, failure);
    }

    private static InvalidOperationException CleanupFailure(int slot, Exception? failure)
        => new($"Copied slot {slot} could not be freed after failed publication. Its unpublished payload and budget ownership remain with the store.", failure);
}
