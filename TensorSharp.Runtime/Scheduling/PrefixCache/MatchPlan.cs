// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>O1/O4 reuse breakdown of one admission.</summary>
public readonly record struct ReuseBreakdown(int PublicTokens, int ScopedTokens, string Source, string Mode,
                                             double MaterializeMs, long MaterializeBytes, int BlockedByScope, double DedupWaitMs);

internal enum CandidateKind : byte { None, EndState, TruncatedEndState, Pages, PrimaryResident }

internal enum MaterializeMode : byte
{
    None,
    CloneEndState,          // model.TryMaterialize(Clone): copy into the request's private holder
    DonateEndState,         // model.TryMaterialize(Donate): re-key, zero copy
    KeepPrimary,            // PrimaryResident donated to this request (EnsureOwnership keeps the primary)
    BindPagesInPlace,       // A2 pages appended to BlockTable; KvStateInPagedStorage = true; route BatchedPaged
    InjectA1Pages,          // A1 slabs injected at first bind
    CopyA2PagesToHolder,    // model.TryCopyPagedToHolder (class P, M7a)
    ConvertPrimaryThenClone,// PrimaryResident not donatable: TryConvertPrimary under a tree key, then clone it
}

internal enum ExpectedRoute : byte { Primary, PerSequenceFused, BatchedPaged }

[Flags]
internal enum ClampReasons : ushort
{
    None = 0,
    Media = 1,
    MediaAcrossSpan = 2,
    Breakpoint = 4,
    Granularity = 8,
    Window = 16,
    LeaveOne = 32,
    RewindCap = 64,
    MmThreshold = 128,
    CloneCost = 256,
}

/// <summary>Why a source did not produce the chosen plan (real reasons for the O1 line).</summary>
internal enum SourceDecline : byte
{
    None,
    Absent,                 // nothing of this kind on the matched path
    NotPermitted,           // only nodes past the public cap or of another scope
    Clamped,                // clamps (media, breakpoint, leave-one) cut every candidate
    PrimaryBusy,            // PrimaryResident while requests run or work is scheduled
    PrimaryClaimed,         // PrimaryResident already state-locked this step
    ModelRefused,           // CanMaterialize == false (queued for invalidation)
    DonateOnlyShared,       // donate-only payload that cannot be donated
    RouteUnreadable,        // page store not readable on the expected route
    Unsupported,            // capability absent
    Shorter,                // a longer candidate won
    CloneCost,              // below MinCloneTokens
    MmThreshold,            // below MmReuseMinTokens with media after it
}

internal readonly record struct MatchRequest(
    KeyRope Key, int KeyLength, int ScopeIx, int PublicBoundary,
    int MatchLimit,                          // λ = min(len−1, explicit ? CacheBreakpointLimit : len−1)   (K6, K7)
    MediaSpanRecord[] Spans,
    ExpectedRoute Route,                     // from ExecutionPlanner.PredictAdmissionRoute (§7.3)
    bool PrimaryAvailable);                  // running == 0 && no scheduled work this step

/// <summary>One matched node on a trail: <c>Matched</c> elements of its edge, starting at <c>StartDepth</c>.</summary>
internal readonly record struct TrailEntry(RadixNode Node, int StartDepth, int Matched, int Prev)
{
    internal int EndDepth => StartDepth + Matched;
    internal bool Full => Matched == Node.Edge.Length;
}

/// <summary>A grow-only list that reuses its buffer across <see cref="Clear"/> (no steady-state allocation).</summary>
internal sealed class PooledList<T>
{
    private T[] _items;
    private int _count;

    internal PooledList(int capacity = 16)
    {
        _items = new T[Math.Max(1, capacity)];
    }

    internal int Count => _count;

    internal ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if ((uint)index >= (uint)_count) throw new ArgumentOutOfRangeException(nameof(index));
            return ref _items[index];
        }
    }

    internal void Add(in T item)
    {
        if (_count == _items.Length)
            Array.Resize(ref _items, _items.Length * 2);
        _items[_count++] = item;
    }

    internal void RemoveLast() => _items[--_count] = default!;

    internal void Clear()
    {
        if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            Array.Clear(_items, 0, _count);
        _count = 0;
    }

    internal ReadOnlySpan<T> AsSpan() => new(_items, 0, _count);
}

/// <summary>
/// The result of <c>PrefixTree.Match</c> + <c>Evaluate</c> for one request at one tree
/// <see cref="Version"/>. Pooled by the caller; immutable after <c>Evaluate</c>.
/// </summary>
internal sealed class MatchPlan
{
    internal const int MaxTrailEnds = 16;

    public long Version;
    public CandidateKind Kind; public MaterializeMode Mode; public int Length;
    public RadixNode? AnchorParent; public int AnchorOffset;  // where Acquire splits (node + offset in its edge)
    public int AnchorTrailEnd;                               // index into TrailEnds of the chosen path
    public RadixNode? PayloadNode;                           // EndState/PrimaryResident source; null for pages
    public int PageCount;                                    // pages [0, PageCount) on the trail
    public int Structural;                                   // LCP over permitted nodes
    public int PublicCap, PublicTokens;
    public int BranchPosition;                               // P13 hint (0 = none)
    public int BlockedByScope;                               // O2 (sampled; never influences the plan)
    public ClampReasons Clamps;
    public SourceDecline EndStateDecline, PageDecline, PrimaryDecline, TruncationDecline;   // real reasons for O1
    public bool SearchCapped;                                // more than MaxTrailEnds alternative paths
    public bool TruncationSearchCapped;                      // the truncation BFS stopped at its node budget
    public int CandidateCount;                               // candidates considered (diagnostics)
    internal readonly PooledList<TrailEntry> Trail = new(32);   // every matched node of every alternative path
    internal readonly PooledList<int> TrailEnds = new(4);       // index of the last entry of each path

    internal void Reset()
    {
        Version = 0; Kind = CandidateKind.None; Mode = MaterializeMode.None; Length = 0;
        AnchorParent = null; AnchorOffset = 0; AnchorTrailEnd = -1; PayloadNode = null; PageCount = 0;
        Structural = 0; PublicCap = 0; PublicTokens = 0; BranchPosition = 0; BlockedByScope = 0;
        Clamps = ClampReasons.None;
        EndStateDecline = SourceDecline.None; PageDecline = SourceDecline.None;
        PrimaryDecline = SourceDecline.None; TruncationDecline = SourceDecline.None;
        SearchCapped = false; TruncationSearchCapped = false; CandidateCount = 0;
        Trail.Clear(); TrailEnds.Clear();
    }

    /// <summary>True when the plan reuses anything.</summary>
    internal bool HasReuse => Kind != CandidateKind.None && Length > 0;

    public override string ToString() =>
        $"plan(v={Version}, {Kind}/{Mode}, L={Length}, structural={Structural}, cap={PublicCap}, public={PublicTokens}, pages={PageCount}, clamps={Clamps})";
}
