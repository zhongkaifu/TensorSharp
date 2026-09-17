// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>Protection tiers in eviction order (§5.11): Retired → Ordinary → Breakpoint → ScopeNewest → PublicTop.</summary>
internal enum EvictionTier : byte { Retired = 0, Ordinary = 1, Breakpoint = 2, ScopeNewest = 3, PublicTop = 4 }

/// <summary>The two list kinds: evictable leaves (whole node) and detachable end states.</summary>
internal enum LruKind : byte { Leaf = 1, State = 2 }

/// <summary>
/// Intrusive LRU lists per tier (§5.11): <c>LeafLru[t]</c> holds unlocked payload leaves,
/// <c>StateLru[t]</c> holds end states of internal or path-locked nodes. Each list is ordered by
/// <see cref="RadixNode.LastAccess"/>, oldest first (I9).
/// </summary>
internal sealed class EvictionLists
{
    internal const int TierCount = 5;
    private const int ListCount = 2 * TierCount;

    private readonly RadixNode?[] _head = new RadixNode?[ListCount + 1];
    private readonly RadixNode?[] _tail = new RadixNode?[ListCount + 1];
    private readonly int[] _count = new int[ListCount + 1];

    /// <summary>Elements walked while inserting out of order (a cascade parent re-entering a list).</summary>
    internal long InsertScanSteps { get; private set; }

    /// <summary>List id in [1, 10]; 0 means "in no list".</summary>
    internal static byte ListId(LruKind kind, EvictionTier tier) => (byte)(1 + ((int)kind - 1) * TierCount + (int)tier);

    internal static LruKind KindOf(byte listId) => listId <= TierCount ? LruKind.Leaf : LruKind.State;

    internal RadixNode? First(LruKind kind, EvictionTier tier) => _head[ListId(kind, tier)];

    internal RadixNode? First(byte listId) => _head[listId];

    internal RadixNode? Last(byte listId) => _tail[listId];

    internal int Count(byte listId) => _count[listId];

    /// <summary>Links <paramref name="n"/> into <paramref name="listId"/>, keeping the list ordered by LastAccess.</summary>
    internal void Link(RadixNode n, byte listId)
    {
        if (listId == 0 || listId > ListCount) throw new ArgumentOutOfRangeException(nameof(listId));
        if (n.LruList != 0) throw new InvalidOperationException("Node is already linked.");
        RadixNode? tail = _tail[listId];
        RadixNode? head = _head[listId];
        if (tail is null)
        {
            _head[listId] = _tail[listId] = n;
            n.LruPrev = n.LruNext = null;
        }
        else if (n.LastAccess >= tail.LastAccess)
        {
            n.LruPrev = tail; n.LruNext = null;
            tail.LruNext = n;
            _tail[listId] = n;
        }
        else if (n.LastAccess < head!.LastAccess)
        {
            n.LruPrev = null; n.LruNext = head;
            head.LruPrev = n;
            _head[listId] = n;
        }
        else
        {
            // Walk back from the tail to the first element not newer than n.
            RadixNode cur = tail;
            while (cur.LastAccess > n.LastAccess)
            {
                cur = cur.LruPrev!;
                InsertScanSteps++;
            }
            n.LruPrev = cur; n.LruNext = cur.LruNext;
            cur.LruNext!.LruPrev = n;
            cur.LruNext = n;
        }
        n.LruList = listId;
        _count[listId]++;
    }

    internal void Unlink(RadixNode n)
    {
        byte id = n.LruList;
        if (id == 0) return;
        if (n.LruPrev is null) _head[id] = n.LruNext; else n.LruPrev.LruNext = n.LruNext;
        if (n.LruNext is null) _tail[id] = n.LruPrev; else n.LruNext.LruPrev = n.LruPrev;
        n.LruPrev = n.LruNext = null;
        n.LruList = 0;
        _count[id]--;
    }

    internal void Clear()
    {
        for (int i = 0; i <= ListCount; i++)
        {
            RadixNode? cur = _head[i];
            while (cur is not null)
            {
                RadixNode? next = cur.LruNext;
                cur.LruPrev = cur.LruNext = null;
                cur.LruList = 0;
                cur = next;
            }
            _head[i] = _tail[i] = null;
            _count[i] = 0;
        }
    }

    /// <summary>Test hook for invariant mutation tests: drops a node from its list without clearing its list id.</summary>
    internal void UnsafeDetachKeepId(RadixNode n)
    {
        byte id = n.LruList;
        Unlink(n);
        n.LruList = id;
        _count[id]++;
    }
}
