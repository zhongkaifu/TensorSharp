// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

[Flags]
internal enum NodeFlags : byte
{
    None = 0,
    IsPublicBoundary = 1,    // Depth == some inserter's P; Scope == Public (I4)
    EndsAtBreakpoint = 2,    // Depth == an explicit breakpoint of the inserter (tier Breakpoint)
    PromptEnd = 4,           // page families: the edge ends at the inserter's prompt end (DEC-47)
    DonationPending = 8,     // EndState moved to an admitting request; invisible to Evaluate; open txn only (I18)
    Retired = 16,            // scope retired while the node was locked: delete on last unlock
    PreemptHold = 32,        // inserted by OnPreempted while its request waits (tier ScopeNewest, §5.11)
}

/// <summary>One radix node: an edge of key elements plus the payloads that end inside or at it.</summary>
internal sealed class RadixNode
{
    internal long Id;                        // monotonic, never reused
    internal int Generation;                 // bumped when a pooled node object is recycled (receipts carry it)
    internal RadixNode? Parent;              // null only for Root
    internal KeySlice Edge;                  // tokens [Depth − Edge.Length, Depth)
    internal int Depth;
    internal int ScopeIx;                    // 0 = Public
    internal ChildMap Children;              // (ScopeIx, first key element) → child; inline ≤ 2, Dictionary above
    internal NodeFlags Flags;
    internal int LockRef;                    // durable path locks through this node (I6, I7)
    internal int StateLockRef;               // transient locks on THIS node's EndState only
    internal int PinRef;                     // lane pins (M7c) on this node's EndState
    internal long LastAccess, CreatedTick;   // monotonic engine tick (no clock)
    internal int HitCount;
    internal EndStatePayload? EndState;      // state after token Depth−1; null = state tombstone
    internal PageRef[]? Pages; internal int PageCount;   // pages whose LAST token lies in this edge (§4.4)
    internal MediaSpanRecord[]? MediaSpans;  // spans whose Start lies in this edge
    internal int MediaSpanCount;
    internal ResourceVector Bytes;           // Σ of this node's own payload bytes (I8)
    internal RadixNode? LruPrev, LruNext; internal byte LruList;     // 0 = in no list; else EvictionLists id
    internal RadixNode? ScopePrev, ScopeNext;
    internal RadixNode? RopePrev, RopeNext;  // nodes whose Edge references the same rope (compaction)
    internal bool InTree;                    // false once returned to the pool
    internal bool InPublicTop;               // one of the PublicMax most recently used public boundaries (tier PublicTop)
    internal bool HadEndState;               // an end state was detached (a later attach revives the tombstone)

    internal bool IsRoot => Parent is null;

    internal bool IsPublicBoundary => (Flags & NodeFlags.IsPublicBoundary) != 0;

    internal bool IsDonationPending => (Flags & NodeFlags.DonationPending) != 0;

    internal bool IsLeaf => Children.Count == 0;

    internal bool AnyLock => LockRef != 0 || StateLockRef != 0 || PinRef != 0;

    internal bool HasPayload => EndState is not null || PageCount > 0;

    internal int EdgeStartDepth => Depth - Edge.Length;

    internal ReadOnlySpan<PageRef> PageSpan => Pages is null ? ReadOnlySpan<PageRef>.Empty : new ReadOnlySpan<PageRef>(Pages, 0, PageCount);

    internal ReadOnlySpan<MediaSpanRecord> SpanRecords =>
        MediaSpans is null ? ReadOnlySpan<MediaSpanRecord>.Empty : new ReadOnlySpan<MediaSpanRecord>(MediaSpans, 0, MediaSpanCount);

    internal void AddPage(in PageRef page)
    {
        if (Pages is null) Pages = new PageRef[4];
        else if (PageCount == Pages.Length) Array.Resize(ref Pages, Pages.Length * 2);
        // keep sorted by page index
        int i = PageCount;
        while (i > 0 && Pages[i - 1].PageIndex > page.PageIndex)
        {
            Pages[i] = Pages[i - 1];
            i--;
        }
        Pages[i] = page;
        PageCount++;
    }

    internal void RemovePageAt(int index)
    {
        for (int i = index; i < PageCount - 1; i++)
            Pages![i] = Pages[i + 1];
        PageCount--;
        Pages![PageCount] = default;
    }

    internal void AddSpanRecord(in MediaSpanRecord record)
    {
        if (MediaSpans is null) MediaSpans = new MediaSpanRecord[2];
        else if (MediaSpanCount == MediaSpans.Length) Array.Resize(ref MediaSpans, MediaSpans.Length * 2);
        int i = MediaSpanCount;
        while (i > 0 && MediaSpans[i - 1].Start > record.Start)
        {
            MediaSpans[i] = MediaSpans[i - 1];
            i--;
        }
        MediaSpans[i] = record;
        MediaSpanCount++;
    }

    internal void ResetForPool()
    {
        Parent = null; Edge = default; Depth = 0; ScopeIx = 0; Children = default; Flags = NodeFlags.None;
        LockRef = 0; StateLockRef = 0; PinRef = 0; LastAccess = 0; CreatedTick = 0; HitCount = 0; EndState = null;
        if (Pages is not null) Array.Clear(Pages, 0, PageCount);
        PageCount = 0;
        MediaSpanCount = 0;
        Bytes = default; LruPrev = null; LruNext = null; LruList = 0;
        ScopePrev = null; ScopeNext = null; RopePrev = null; RopeNext = null; InTree = false;
        InPublicTop = false; HadEndState = false;
    }

    public override string ToString() =>
        $"Node#{Id}(depth={Depth}, edge={Edge.Length}, scope={ScopeIx}, flags={Flags}, lock={LockRef}/{StateLockRef}/{PinRef}, end={(EndState is null ? "-" : EndState.Kind.ToString())}, pages={PageCount})";
}

/// <summary>Children keyed by (ScopeIx, first key element): two inline slots, a dictionary from the third.</summary>
internal struct ChildMap
{
    internal readonly record struct Key(int ScopeIx, long First);

    private RadixNode? _c0, _c1;
    private Dictionary<Key, RadixNode>? _many;                   // allocated on the 3rd child

    internal readonly int Count => _many is not null ? _many.Count : (_c0 is null ? 0 : 1) + (_c1 is null ? 0 : 1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal readonly RadixNode? Find(int scopeIx, long first)
    {
        if (_many is not null)
            return _many.TryGetValue(new Key(scopeIx, first), out RadixNode? n) ? n : null;
        if (_c0 is not null && _c0.ScopeIx == scopeIx && _c0.Edge[0] == first) return _c0;
        if (_c1 is not null && _c1.ScopeIx == scopeIx && _c1.Edge[0] == first) return _c1;
        return null;
    }

    internal void Add(RadixNode child)
    {
        var key = new Key(child.ScopeIx, child.Edge[0]);
        if (_many is not null)
        {
            if (!_many.TryAdd(key, child))
                throw new InvalidOperationException("Duplicate child key (I2).");
            return;
        }
        if (Find(key.ScopeIx, key.First) is not null)
            throw new InvalidOperationException("Duplicate child key (I2).");
        if (_c0 is null) { _c0 = child; return; }
        if (_c1 is null) { _c1 = child; return; }
        _many = new Dictionary<Key, RadixNode>(4)
        {
            [new Key(_c0.ScopeIx, _c0.Edge[0])] = _c0,
            [new Key(_c1.ScopeIx, _c1.Edge[0])] = _c1,
            [key] = child,
        };
        _c0 = null; _c1 = null;
    }

    /// <summary>Replaces a child with a node that has the same (scope, first element) key.</summary>
    internal void Replace(RadixNode oldChild, RadixNode newChild)
    {
        if (_many is not null)
        {
            var key = new Key(oldChild.ScopeIx, oldChild.Edge[0]);
            _many[key] = newChild;
            return;
        }
        if (ReferenceEquals(_c0, oldChild)) { _c0 = newChild; return; }
        if (ReferenceEquals(_c1, oldChild)) { _c1 = newChild; return; }
        throw new InvalidOperationException("Replace: child not found.");
    }

    internal void Remove(RadixNode child)
    {
        if (_many is not null)
        {
            _many.Remove(new Key(child.ScopeIx, child.Edge[0]));
            return;
        }
        if (ReferenceEquals(_c0, child)) { _c0 = _c1; _c1 = null; return; }
        if (ReferenceEquals(_c1, child)) { _c1 = null; return; }
    }

    /// <summary>Copies the children into <paramref name="buffer"/> (grown when needed); returns the count.</summary>
    internal readonly int CopyTo(ref RadixNode[] buffer)
    {
        int n = Count;
        if (buffer.Length < n) buffer = new RadixNode[Math.Max(n, buffer.Length * 2)];
        if (_many is not null)
        {
            int i = 0;
            foreach (RadixNode c in _many.Values) buffer[i++] = c;
            return n;
        }
        int k = 0;
        if (_c0 is not null) buffer[k++] = _c0;
        if (_c1 is not null) buffer[k++] = _c1;
        return k;
    }

    public readonly Enumerator GetEnumerator() => new(this);

    public struct Enumerator
    {
        private readonly RadixNode? _c0, _c1;
        private Dictionary<Key, RadixNode>.ValueCollection.Enumerator _it;
        private readonly bool _useMany;
        private int _state;

        internal Enumerator(ChildMap map)
        {
            _c0 = map._c0; _c1 = map._c1; _useMany = map._many is not null;
            _it = _useMany ? map._many!.Values.GetEnumerator() : default;
            _state = 0;
            Current = null!;
        }

        public RadixNode Current { get; private set; }

        public bool MoveNext()
        {
            if (_useMany)
            {
                if (_it.MoveNext()) { Current = _it.Current; return true; }
                return false;
            }
            while (_state < 2)
            {
                RadixNode? c = _state == 0 ? _c0 : _c1;
                _state++;
                if (c is not null) { Current = c; return true; }
            }
            return false;
        }
    }
}

/// <summary>Per-tree node pool; <see cref="RadixNode.Generation"/> is bumped on return.</summary>
internal sealed class RadixNodePool
{
    private readonly Stack<RadixNode> _free = new();
    private readonly int _maxFree;

    internal RadixNodePool(int maxFree = 1 << 16)
    {
        _maxFree = maxFree;
    }

    internal RadixNode Rent()
    {
        if (_free.Count > 0)
            return _free.Pop();
        return new RadixNode();
    }

    internal void Return(RadixNode n)
    {
        n.ResetForPool();
        n.Generation++;
        if (_free.Count < _maxFree)
            _free.Push(n);
    }

    internal int FreeCount => _free.Count;
}
