// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>Thrown when <see cref="PrefixTree.Acquire"/> receives a plan computed at another tree version (I19).</summary>
public sealed class StalePrefixMatchException : InvalidOperationException
{
    public StalePrefixMatchException(long planVersion, long treeVersion)
        : base($"Prefix match plan is stale (plan version {planVersion}, tree version {treeVersion}).")
    {
        PlanVersion = planVersion;
        TreeVersion = treeVersion;
    }

    public long PlanVersion { get; }
    public long TreeVersion { get; }
}

/// <summary>Side-effect-free payload check (<c>IPrefixCacheModel.CanMaterialize</c>, M2). Tree mode only.</summary>
internal interface IPayloadValidator
{
    bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens);
}

/// <summary>
/// Donation condition (d) (§5.3.4): whether any other waiting request of the scope could reach
/// <paramref name="node"/>. Implemented by the coordinator over its waiting queue (M3).
/// </summary>
internal interface IWaitingPlanView
{
    bool NoOtherWaiterTargets(int scopeIx, RadixNode node, long treeVersion);
}

internal enum AttachResult : byte { Attached, Revived, Duplicate, Refused }

internal enum PressureLevel : byte { Moderate, Critical }

/// <summary>Options read once when the tree is built (M3 fills them from <c>PrefixCacheOptions</c>).</summary>
internal sealed class PrefixTreeOptions
{
    public required PrefixCacheCapabilities Capabilities { get; init; }
    public int BlockSize { get; init; } = 256;
    public int ContextLength { get; init; } = int.MaxValue;
    public long PageHostBytes { get; init; }                           // A1 slab bytes per page (ComputeBlockByteSize)
    public long EngineSerial { get; init; }
    public int PublicMax { get; init; } = 2;                           // TS_PREFIX_CHECKPOINTS_MAX
    public int ScopedEndStateLeavesMax { get; init; } = 4;             // TS_RETAINED_FUSED_CACHE_MAX; 0 = unlimited
    public bool BatchedPagedEnabled { get; init; }
    public int MinCloneTokens { get; init; }                           // DEC-13; 0 = always clone
    public bool BranchSnapshots { get; init; }                         // M7d
    public bool ComputeBlockedByScope { get; init; }                   // O2 sampling decided by the caller
    public bool TrackReceipts { get; init; }                           // ReceiptLedger (Debug and tests)
#if DEBUG
    public bool StrictReceipts { get; init; } = true;                  // a stale receipt throws
#else
    public bool StrictReceipts { get; init; }
#endif
    public IPrefixTreePageHost? PageHost { get; init; }
    public IPayloadValidator? PayloadValidator { get; init; }          // null = Legacy/Shadow (no model calls)
    public Func<ResourceClass, long>? QuerySpareBytes { get; init; }   // −1 = unknown
    public ResourceVector OptionCapBytes { get; init; }                // TS_PREFIX_CACHE_{DEVICE,HOST}_MB; 0 = auto
    public long HostRamBytes { get; init; }                            // auto HostKv cap = 25%; 0 = unknown
    public long PoolPagesCap { get; init; }                            // BlockPool.NumBlocks; 0 = unbounded
    public long ScopeIdleMs { get; init; } = 600_000;
    public Func<long>? ClockMs { get; init; }
    public int TruncationSearchNodes { get; init; } = 64;
    public int DonationWaiterScanMax { get; init; } = 128;
    public int DonateTruncateSlackTokens { get; init; } = 16;
}

/// <summary>Monotonic counters (all MUST-stay-zero counters are named as in BG-18).</summary>
internal struct PrefixTreeCounters
{
    public long StaleMatches, StaleReceipts, MediaHashCollisions, DuplicatesDropped;
    public long Inserts, Splits, NodesCreated, NodesDeleted, Evictions, EvictScanSkips, EvictCalls;
    public long StateDetaches, RopeCompactions, PagesRefused, PagesDuplicate, EndStatesRefused;
    public long Invalidations, ScopesRetired, SearchCapped;
}

/// <summary>
/// The radix prefix tree of one engine (DESIGN §0, §4, §5). Single writer: only the engine worker
/// thread calls it. <see cref="Match"/> and <see cref="Evaluate"/> are pure; <see cref="Acquire"/>
/// splits at the chosen length and locks, asserting the <see cref="Version"/> the plan was made at.
/// </summary>
internal sealed class PrefixTree
{
    private readonly PrefixTreeOptions _options;
    private readonly IPrefixTreePageHost _pages;
    private readonly RadixNodePool _nodePool = new();
    private readonly Dictionary<string, RadixNode> _keyIndex = new(StringComparer.Ordinal);
    private readonly List<RadixNode> _publicBoundaries = new();       // IsPublicBoundary nodes holding an EndState
    private readonly HashSet<string> _refusedKeys = new(StringComparer.Ordinal);
    private readonly Func<long> _clock;
    private long _tick;
    private long _nodeSerial;
    private long _receiptSerial;
    private long _payloadSerial;
    private ResourceVector _cached;
    private ResourceVector _protected;
    private ResourceVector _runningReserve;
    private ResourceVector _optionCap;
    private long _scopeQuota = long.MaxValue;
    private int _scopedEndStates, _publicEndStates, _nativeSlots, _primaryResidents;
    private int[] _matchStack = new int[16];
    private int[] _order = new int[256];                 // pre-sized: probes stay allocation-free (BG-15)
    private RadixNode[] _bfs;
    private RadixNode[] _scratch = new RadixNode[64];
    internal PrefixTreeCounters Counters;

    internal PrefixTree(PrefixTreeOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Rules = new ResumabilityRules(options.Capabilities, options.BlockSize, options.BatchedPagedEnabled);
        _bfs = new RadixNode[Math.Max(64, options.TruncationSearchNodes + 2)];
        _pages = options.PageHost ?? NullPageHost.Instance;
        _clock = options.ClockMs ?? (() => Environment.TickCount64);
        if (options.PublicMax < 0) throw new ArgumentOutOfRangeException(nameof(options), "PublicMax must not be negative.");
        Root = new RadixNode { Id = 0, InTree = true };
        KeyPool = new KeyChunkPool();
        Scopes = new ScopeTable();
        Lists = new EvictionLists();
        Reclaim = new ReclaimQueue();
        Ledger = options.TrackReceipts ? new ReceiptLedger() : null;
        TreeSerial = 1;
        ResolveOptionCaps();
        RefreshBudgets();
    }

    internal PrefixTreeOptions Options => _options;
    internal PrefixCacheCapabilities Caps => _options.Capabilities;
    internal ResumabilityRules Rules { get; }
    internal RadixNode Root { get; }
    internal KeyChunkPool KeyPool { get; }
    internal ScopeTable Scopes { get; }
    internal EvictionLists Lists { get; }
    internal ReclaimQueue Reclaim { get; }
    internal ReceiptLedger? Ledger { get; }
    internal IPrefixTreePageHost PageHost => _pages;

    /// <summary>Bumped by every mutation that can change <see cref="Evaluate"/>'s result (I19).</summary>
    internal long Version { get; private set; } = 1;

    /// <summary>Receipt epoch: bumped by <see cref="Reset"/>, so older receipts are ignored.</summary>
    internal long TreeSerial { get; private set; }

    internal int NodeCount { get; private set; }

    internal ResourceVector Cached => _cached;
    internal ResourceVector Protected => _protected;
    internal ResourceVector Evictable => _cached - _protected;
    internal ResourceVector PendingReclaim => Reclaim.PendingReclaim;
    internal ResourceVector RunningReserve => _runningReserve;
    internal long ScopeQuota => _scopeQuota;
    internal int PublicEndStateCount => _publicEndStates;
    internal int ScopedEndStateCount => _scopedEndStates;
    internal int NativeSlotCount => _nativeSlots;
    internal int PrimaryResidentCount => _primaryResidents;
    internal IReadOnlyList<RadixNode> PublicBoundaryNodes => _publicBoundaries;
    internal IReadOnlyCollection<string> PayloadKeys => _keyIndex.Keys;
    internal int QueuedRefusalCount => _refusedKeys.Count;

    // ------------------------------------------------------------------ keys and scopes

    /// <summary>A tree-minted payload key <c>pc:{engineSerial}:{payloadSerial}</c> (I10).</summary>
    internal string MintKey() =>
        string.Create(CultureInfo.InvariantCulture, $"pc:{_options.EngineSerial}:{++_payloadSerial}");

    internal bool TryGetNodeByKey(string key, out RadixNode node) => _keyIndex.TryGetValue(key, out node!);

    /// <summary>Interns a scope and records request activity for the tier rules.</summary>
    internal int InternScope(ScopeId id, ScopeKind kind) => Scopes.Intern(id, kind);

    /// <summary>Adjusts a scope's waiting and running request counts (§5.1, §5.9).</summary>
    internal void NoteRequests(int scopeIx, int waitingDelta, int runningDelta)
    {
        if (scopeIx == 0) return;
        ScopeRecord rec = Scopes[scopeIx];
        rec.WaitingRequests += waitingDelta;
        rec.RunningRequests += runningDelta;
        if (rec.WaitingRequests < 0 || rec.RunningRequests < 0)
            throw new InvalidOperationException("Scope request counts went negative.");
        rec.LastUseTick = ++_tick;
        rec.LastUseMs = _clock();
        SetScopeActive(rec, true);
        TryRecycleScope(rec);
    }

    /// <summary>Re-evaluates the idle rule of every scope (ScopeNewest tier).</summary>
    internal void RefreshScopeActivity()
    {
        long now = _clock();
        foreach (ScopeRecord rec in Scopes.LiveRecords())
        {
            if (rec.Index == 0) continue;
            bool active = rec.RunningRequests + rec.WaitingRequests > 0 || now - rec.LastUseMs < _options.ScopeIdleMs;
            SetScopeActive(rec, active);
        }
    }

    private void SetScopeActive(ScopeRecord rec, bool active)
    {
        if (rec.Active == active) return;
        rec.Active = active;
        if (rec.NewestLeaf is not null) Relink(rec.NewestLeaf);
    }

    /// <summary>Releases a request's ownership of its key rope; chunks return to the pool once no edge uses them.</summary>
    internal void ReleaseRopeOwner(KeyRope rope)
    {
        if (rope is null || rope.OwnerReleased) return;
        rope.OwnerReleased = true;
        MaybeReclaimRope(rope);
    }

    // ------------------------------------------------------------------ Match (pure)

    /// <summary>
    /// Pure match (§5.2): walks public and own-scope children from the root. Where both a public and
    /// an own-scope child match, both paths are followed (up to <see cref="MatchPlan.MaxTrailEnds"/>),
    /// so a shared public branch never hides a longer own-scope branch. No mutation, no allocation.
    /// </summary>
    internal void Match(in MatchRequest r, MatchPlan plan)
    {
        plan.Reset();
        plan.Version = Version;
        int limit = ResumabilityRules.EffectiveMatchLimit(r);
        if (limit <= 0) return;
        PooledList<TrailEntry> trail = plan.Trail;
        int stackCount = 0;
        // Expand the root.
        ExpandInto(r, plan, Root, 0, -1, limit, ref stackCount);
        while (stackCount > 0)
        {
            int ix = _matchStack[--stackCount];
            TrailEntry e = trail[ix];
            ExpandInto(r, plan, e.Node, e.EndDepth, ix, limit, ref stackCount);
        }
    }

    private void ExpandInto(in MatchRequest r, MatchPlan plan, RadixNode node, int d, int prev, int limit, ref int stackCount)
    {
        if (d >= limit)
        {
            if (prev >= 0) plan.TrailEnds.Add(prev);
            return;
        }
        long e = r.Key[d];
        RadixNode? pub = node.Children.Find(0, e);
        RadixNode? own = r.ScopeIx != 0 ? node.Children.Find(r.ScopeIx, e) : null;
        int lenPub = pub is null ? 0 : MediaVerify(pub, d, KeyCompare.CommonPrefixLength(pub.Edge, 0, r.Key, d, limit - d), r.Spans);
        int lenOwn = own is null ? 0 : MediaVerify(own, d, KeyCompare.CommonPrefixLength(own.Edge, 0, r.Key, d, limit - d), r.Spans);
        if (lenPub == 0 && lenOwn == 0)
        {
            if (prev >= 0) plan.TrailEnds.Add(prev);
            return;
        }
        if (lenPub > 0 && lenOwn > 0 && plan.TrailEnds.Count + stackCount + 2 > MatchPlan.MaxTrailEnds)
        {
            // Search budget exhausted: the design's greedy rule (longer wins, ties → own scope).
            plan.SearchCapped = true;
            if (lenOwn >= lenPub) lenPub = 0; else lenOwn = 0;
        }
        if (lenPub > 0) AddTrail(plan, pub!, d, lenPub, prev, limit, ref stackCount);
        if (lenOwn > 0) AddTrail(plan, own!, d, lenOwn, prev, limit, ref stackCount);
    }

    private void AddTrail(MatchPlan plan, RadixNode c, int d, int len, int prev, int limit, ref int stackCount)
    {
        plan.Trail.Add(new TrailEntry(c, d, len, prev));
        int ix = plan.Trail.Count - 1;
        if (len == c.Edge.Length && d + len < limit)
        {
            if (stackCount == _matchStack.Length) Array.Resize(ref _matchStack, _matchStack.Length * 2);
            _matchStack[stackCount++] = ix;
        }
        else
        {
            plan.TrailEnds.Add(ix);
        }
    }

    /// <summary>
    /// Media verification (§5.2) of <c>[d, d+len)</c> against node <paramref name="c"/>: every node
    /// record starting in range must equal a request span (Start, End and 256-bit id) and vice versa.
    /// The first violation cuts the length to its Start.
    /// </summary>
    internal static int MediaVerify(RadixNode c, int d, int len, MediaSpanRecord[]? spans)
    {
        if (len <= 0) return 0;
        int end = d + len;
        int cut = end;
        ReadOnlySpan<MediaSpanRecord> records = c.SpanRecords;
        for (int i = 0; i < records.Length; i++)
        {
            MediaSpanRecord rec = records[i];
            if (rec.Start < d) continue;
            if (rec.Start >= cut) break;
            if (!ContainsSpan(spans, rec)) { cut = rec.Start; break; }
        }
        if (spans is { Length: > 0 })
        {
            for (int i = LowerBound(spans, d); i < spans.Length; i++)
            {
                MediaSpanRecord s = spans[i];
                if (s.Start >= cut) break;
                if (!ContainsRecord(records, s)) { cut = s.Start; break; }
            }
        }
        return cut - d;
    }

    private static bool ContainsSpan(MediaSpanRecord[]? spans, in MediaSpanRecord rec)
    {
        if (spans is null || spans.Length == 0) return false;
        int i = LowerBound(spans, rec.Start);
        return i < spans.Length && spans[i].Start == rec.Start && spans[i].End == rec.End && spans[i].Id.Equals(rec.Id);
    }

    private static bool ContainsRecord(ReadOnlySpan<MediaSpanRecord> records, in MediaSpanRecord s)
    {
        for (int i = 0; i < records.Length; i++)
        {
            if (records[i].Start == s.Start)
                return records[i].End == s.End && records[i].Id.Equals(s.Id);
            if (records[i].Start > s.Start) break;
        }
        return false;
    }

    private static int LowerBound(MediaSpanRecord[] spans, int start)
    {
        int lo = 0, hi = spans.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (spans[mid].Start < start) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    // ------------------------------------------------------------------ Evaluate (pure)

    private struct Candidate
    {
        public CandidateKind Kind;
        public MaterializeMode Mode;
        public int Length;
        public RadixNode? Payload;
        public int TrailEnd;        // index into plan.TrailEnds of the path that carries Length
        public long TieId;          // deterministic tie break (older node first)

        public readonly bool IsValid => Kind != CandidateKind.None && Length > 0;
    }

    /// <summary>Match + Evaluate in one call (the PlanFor body of §5.2, minus the plan cache).</summary>
    internal void Plan(in MatchRequest r, MatchPlan plan, IWaitingPlanView? waiting = null)
    {
        Match(r, plan);
        Evaluate(r, plan, waiting);
    }

    /// <summary>
    /// Evaluate (§5.3) the trails left by <see cref="Match"/>: exact end states (A), pages (B) and
    /// truncated end states (C); the donation decision; the cost and multimodal thresholds. Pure:
    /// a model refusal is only queued (<see cref="FlushQueuedInvalidations"/>).
    /// </summary>
    internal void Evaluate(in MatchRequest r, MatchPlan plan, IWaitingPlanView? waiting = null)
    {
        plan.Version = Version;
        if (plan.SearchCapped) Counters.SearchCapped++;
        PooledList<TrailEntry> trail = plan.Trail;
        int structural = 0, boundaryDepth = 0;
        for (int i = 0; i < trail.Count; i++)
        {
            TrailEntry te = trail[i];
            if (te.EndDepth > structural) structural = te.EndDepth;
            if (te.Full && te.Node.IsPublicBoundary && te.Node.Depth > boundaryDepth) boundaryDepth = te.Node.Depth;
        }
        int publicCap = Math.Max(r.PublicBoundary, boundaryDepth);
        plan.Structural = structural;
        plan.PublicCap = publicCap;
        ClampReasons clamps = ClampReasons.None;

        Candidate a = EvaluateEndStates(r, plan, publicCap, waiting, ref clamps);
        Candidate b = EvaluatePages(r, plan, publicCap, ref clamps);
        Candidate c = EvaluateTruncation(r, plan, publicCap, Math.Max(a.Length, b.Length), waiting, ref clamps);

        Candidate best = Better(Better(a, b), c);
        if (best.IsValid && ResumabilityRules.IsClone(best.Mode) && best.Length < _options.MinCloneTokens)
        {
            clamps |= ClampReasons.CloneCost;
            SetDecline(plan, best.Kind, SourceDecline.CloneCost);
            Candidate next = default;
            if (!IsCloneOrNone(a) && !SameCandidate(a, best)) next = Better(next, a);
            if (!IsCloneOrNone(b) && !SameCandidate(b, best)) next = Better(next, b);
            if (!IsCloneOrNone(c) && !SameCandidate(c, best)) next = Better(next, c);
            best = next;
        }
        if (best.IsValid && Caps.MmReuseMinTokens > 0 && best.Length < Caps.MmReuseMinTokens && HasSpanAtOrAfter(r.Spans, best.Length))
        {
            clamps |= ClampReasons.MmThreshold;
            SetDecline(plan, best.Kind, SourceDecline.MmThreshold);
            best = default;
        }

        plan.Clamps = clamps;
        if (!best.IsValid)
        {
            plan.Kind = CandidateKind.None;
            plan.Mode = MaterializeMode.None;
            plan.Length = 0;
        }
        else
        {
            plan.Kind = best.Kind;
            plan.Mode = best.Mode;
            plan.Length = best.Length;
            plan.PayloadNode = best.Payload;
            plan.AnchorTrailEnd = best.TrailEnd;
            plan.PageCount = best.Kind == CandidateKind.Pages ? best.Length / _options.BlockSize : 0;
            if (best.Kind == CandidateKind.EndState || best.Kind == CandidateKind.PrimaryResident)
            {
                plan.AnchorParent = best.Payload;
                plan.AnchorOffset = best.Payload!.Edge.Length;
            }
            else
            {
                (RadixNode node, int offset) = LocateOnTrail(plan, best.TrailEnd, best.Length);
                plan.AnchorParent = node;
                plan.AnchorOffset = offset;
            }
            plan.PublicTokens = PublicTokensOnPath(plan.AnchorParent!, best.Length, publicCap);
            MarkShorter(plan, a, best);
            MarkShorter(plan, b, best);
            MarkShorter(plan, c, best);
        }
        if (_options.BranchSnapshots && Caps.Truncation == TruncationKind.None && structural - plan.Length >= 256)
        {
            ClampReasons ignored = ClampReasons.None;
            int bp = Rules.ClampAligned(structural, r, _options.BlockSize, ref ignored);
            plan.BranchPosition = bp > plan.Length ? bp : 0;
        }
        if (_options.ComputeBlockedByScope)
            plan.BlockedByScope = ComputeBlockedByScope(r, plan);
    }

    private static bool IsCloneOrNone(in Candidate c) => !c.IsValid || ResumabilityRules.IsClone(c.Mode);

    private static bool SameCandidate(in Candidate x, in Candidate y) => x.Kind == y.Kind && x.Length == y.Length && ReferenceEquals(x.Payload, y.Payload);

    private static void MarkShorter(MatchPlan plan, in Candidate cand, in Candidate best)
    {
        if (!cand.IsValid || SameCandidate(cand, best)) return;
        SetDeclineIfNone(plan, cand.Kind, SourceDecline.Shorter);
    }

    private static void SetDecline(MatchPlan plan, CandidateKind kind, SourceDecline decline)
    {
        switch (kind)
        {
            case CandidateKind.EndState: plan.EndStateDecline = decline; break;
            case CandidateKind.PrimaryResident: plan.PrimaryDecline = decline; break;
            case CandidateKind.Pages: plan.PageDecline = decline; break;
            case CandidateKind.TruncatedEndState: plan.TruncationDecline = decline; break;
        }
    }

    private static void SetDeclineIfNone(MatchPlan plan, CandidateKind kind, SourceDecline decline)
    {
        switch (kind)
        {
            case CandidateKind.EndState: if (plan.EndStateDecline == SourceDecline.None) plan.EndStateDecline = decline; break;
            case CandidateKind.PrimaryResident: if (plan.PrimaryDecline == SourceDecline.None) plan.PrimaryDecline = decline; break;
            case CandidateKind.Pages: if (plan.PageDecline == SourceDecline.None) plan.PageDecline = decline; break;
            case CandidateKind.TruncatedEndState: if (plan.TruncationDecline == SourceDecline.None) plan.TruncationDecline = decline; break;
        }
    }

    /// <summary>argmax L, then the §5.3.3 tie order, then the older node.</summary>
    private static Candidate Better(in Candidate x, in Candidate y)
    {
        if (!y.IsValid) return x;
        if (!x.IsValid) return y;
        if (x.Length != y.Length) return x.Length > y.Length ? x : y;
        int rx = ResumabilityRules.TieRank(x.Kind, x.Mode), ry = ResumabilityRules.TieRank(y.Kind, y.Mode);
        if (rx != ry) return rx < ry ? x : y;
        return y.TieId < x.TieId ? y : x;
    }

    private static bool HasSpanAtOrAfter(MediaSpanRecord[]? spans, int position)
    {
        if (spans is null) return false;
        for (int i = spans.Length - 1; i >= 0; i--)
            if (spans[i].Start >= position) return true;
        return false;
    }

    /// <summary>(A) Exact end states, deepest first.</summary>
    private Candidate EvaluateEndStates(in MatchRequest r, MatchPlan plan, int publicCap, IWaitingPlanView? waiting, ref ClampReasons clamps)
    {
        PooledList<TrailEntry> trail = plan.Trail;
        int n = 0;
        if (_order.Length < trail.Count) _order = new int[Math.Max(trail.Count, _order.Length * 2)];
        for (int i = 0; i < trail.Count; i++)
        {
            TrailEntry te = trail[i];
            if (te.Full && te.Node.EndState is not null && !te.Node.IsDonationPending)
            {
                // insertion sort by depth, deepest first
                int j = n++;
                while (j > 0 && trail[_order[j - 1]].Node.Depth < te.Node.Depth)
                {
                    _order[j] = _order[j - 1];
                    j--;
                }
                _order[j] = i;
            }
        }
        if (n == 0)
        {
            plan.EndStateDecline = SourceDecline.Absent;
            if (Caps.PrimaryResident) plan.PrimaryDecline = SourceDecline.Absent;
            return default;
        }
        Candidate best = default;
        for (int k = 0; k < n; k++)
        {
            TrailEntry te = trail[_order[k]];
            RadixNode c = te.Node;
            int length = c.Depth;
            if (best.IsValid && length < best.Length) break;
            bool primary = c.EndState!.Kind == EndStateKind.PrimaryResident;
            plan.CandidateCount++;
            if (!ResumabilityRules.Permitted(c.ScopeIx, length, r.ScopeIx, publicCap))
            {
                SetDeclineIfNone(plan, primary ? CandidateKind.PrimaryResident : CandidateKind.EndState, SourceDecline.NotPermitted);
                continue;
            }
            if (Rules.ClampLength(length, r, ref clamps) != length)
            {
                SetDeclineIfNone(plan, primary ? CandidateKind.PrimaryResident : CandidateKind.EndState, SourceDecline.Clamped);
                continue;
            }
            if (primary)
            {
                if (!r.PrimaryAvailable) { SetDeclineIfNone(plan, CandidateKind.PrimaryResident, SourceDecline.PrimaryBusy); continue; }
                if (c.StateLockRef > 0) { SetDeclineIfNone(plan, CandidateKind.PrimaryResident, SourceDecline.PrimaryClaimed); continue; }
            }
            else if (_options.PayloadValidator is not null && !_options.PayloadValidator.CanMaterialize(c.EndState.Key, length, length))
            {
                _refusedKeys.Add(c.EndState.Key);
                SetDeclineIfNone(plan, CandidateKind.EndState, SourceDecline.ModelRefused);
                continue;
            }
            CandidateKind kind = primary ? CandidateKind.PrimaryResident : CandidateKind.EndState;
            MaterializeMode mode = DonationDecision(c, length, primary, waiting);
            if (mode == MaterializeMode.None)
            {
                SetDeclineIfNone(plan, kind, SourceDecline.DonateOnlyShared);
                continue;
            }
            var cand = new Candidate { Kind = kind, Mode = mode, Length = length, Payload = c, TrailEnd = TrailEndOf(plan, _order[k]), TieId = c.Id };
            best = Better(best, cand);
        }
        return best;
    }

    /// <summary>Index into <c>plan.TrailEnds</c> of a path containing trail entry <paramref name="entryIx"/>.</summary>
    private static int TrailEndOf(MatchPlan plan, int entryIx)
    {
        for (int t = 0; t < plan.TrailEnds.Count; t++)
        {
            for (int ix = plan.TrailEnds[t]; ix >= 0; ix = plan.Trail[ix].Prev)
                if (ix == entryIx) return t;
        }
        return -1;
    }

    /// <summary>(B) Pages, per path and per materialization mode.</summary>
    private Candidate EvaluatePages(in MatchRequest r, MatchPlan plan, int publicCap, ref ClampReasons clamps)
    {
        if (Caps.Pages == PageSupport.None && !Caps.SupportsCopyPagedToHolder)
        {
            plan.PageDecline = SourceDecline.Unsupported;
            return default;
        }
        int blockSize = _options.BlockSize;
        Candidate best = default;
        bool sawPage = false, sawReadable = false;
        for (int t = 0; t < plan.TrailEnds.Count; t++)
        {
            int pathLen = CollectPath(plan, plan.TrailEnds[t]);
            int structural = plan.Trail[plan.TrailEnds[t]].EndDepth;
            for (int m = 0; m < 3; m++)
            {
                MaterializeMode mode = m == 0 ? MaterializeMode.InjectA1Pages : m == 1 ? MaterializeMode.BindPagesInPlace : MaterializeMode.CopyA2PagesToHolder;
                int usable = 0;
                int p = 0;
                bool stop = false;
                for (int k = 0; k < pathLen && !stop; k++)
                {
                    TrailEntry te = plan.Trail[_order[k]];
                    RadixNode node = te.Node;
                    ReadOnlySpan<PageRef> pages = node.PageSpan;
                    int pi = 0;
                    while (!stop)
                    {
                        int pageEnd = (p + 1) * blockSize;
                        if (pageEnd - 1 >= node.Depth) break;               // the next page ends in a deeper node
                        while (pi < pages.Length && pages[pi].PageIndex < p) pi++;
                        if (pi >= pages.Length || pages[pi].PageIndex != p) { stop = true; break; }   // absent
                        sawPage = true;
                        PageRef page = pages[pi];
                        if (pageEnd > structural || !ResumabilityRules.Permitted(node.ScopeIx, pageEnd, r.ScopeIx, publicCap)
                            || !Rules.RouteCanRead(page.Store, mode, r.Route))
                        { stop = true; break; }
                        sawReadable = true;
                        if (!Rules.PageInWindow(pageEnd)) { clamps |= ClampReasons.Window; stop = true; break; }
                        if (Rules.PageEndResumable(page) && Rules.ClampLength(pageEnd, r, ref clamps) == pageEnd)
                            usable = pageEnd;
                        p++;
                    }
                    if (te.EndDepth < node.Depth) break;                     // partial node: nothing deeper on this path
                }
                if (usable <= 0) continue;
                plan.CandidateCount++;
                var cand = new Candidate
                {
                    Kind = CandidateKind.Pages, Mode = mode, Length = usable, Payload = null, TrailEnd = t,
                    TieId = ResumabilityRules.PageModePreference(mode, r.Route),
                };
                // B ← the mode with the largest L; ties prefer the route's natural mode (§5.3.2).
                if (!best.IsValid || cand.Length > best.Length || (cand.Length == best.Length && cand.TieId < best.TieId))
                    best = cand;
            }
        }
        if (!best.IsValid)
            plan.PageDecline = !sawPage ? SourceDecline.Absent : !sawReadable ? SourceDecline.RouteUnreadable : SourceDecline.Clamped;
        else
            best.TieId = long.MaxValue;   // across kinds, pages carry no node age
        return best;
    }

    /// <summary>Writes the trail indices of one path into <c>_order</c>, root first; returns the count.</summary>
    private int CollectPath(MatchPlan plan, int endIx)
    {
        int n = 0;
        for (int ix = endIx; ix >= 0; ix = plan.Trail[ix].Prev) n++;
        if (_order.Length < n) _order = new int[Math.Max(n, _order.Length * 2)];
        int k = n;
        for (int ix = endIx; ix >= 0; ix = plan.Trail[ix].Prev) _order[--k] = ix;
        return n;
    }

    /// <summary>(C) Truncated end states in the subtree below each path's stop node.</summary>
    private Candidate EvaluateTruncation(in MatchRequest r, MatchPlan plan, int publicCap, int bestSoFar, IWaitingPlanView? waiting, ref ClampReasons clamps)
    {
        if (Caps.Truncation == TruncationKind.None)
        {
            plan.TruncationDecline = SourceDecline.Unsupported;
            return default;
        }
        Candidate best = default;
        bool considered = false;
        int budget = Math.Max(1, _options.TruncationSearchNodes);
        for (int t = 0; t < plan.TrailEnds.Count; t++)
        {
            TrailEntry end = plan.Trail[plan.TrailEnds[t]];
            int structural = end.EndDepth;
            ClampReasons local = ClampReasons.None;
            int target = Rules.ClampAligned(structural, r, Caps.TruncationGranularity, ref local);
            if (target <= bestSoFar || target <= 0) continue;
            clamps |= local;
            considered = true;
            int head = 0, tail = 0, visited = 0;
            EnsureBfs(1);
            _bfs[tail++] = end.Node;
            while (head < tail)
            {
                RadixNode d = _bfs[head++];
                if (++visited > budget)
                {
                    plan.TruncationSearchCapped = true;
                    break;
                }
                if (d.EndState is not null && !d.IsDonationPending && d.Depth > target)
                {
                    plan.CandidateCount++;
                    Candidate cand = QualifyTruncation(r, plan, d, target, publicCap, waiting, ref clamps);
                    if (cand.IsValid)
                    {
                        cand.TrailEnd = t;
                        best = BetterTruncation(best, cand);
                    }
                }
                foreach (RadixNode child in d.Children)
                {
                    if (child.ScopeIx != 0 && child.ScopeIx != r.ScopeIx) continue;
                    if (tail > budget) break;          // never visited: the loop caps at budget (no queue growth)
                    EnsureBfs(tail + 1);
                    _bfs[tail++] = child;
                }
            }
            Array.Clear(_bfs, 0, tail);
        }
        if (!best.IsValid && plan.TruncationDecline == SourceDecline.None)
            plan.TruncationDecline = considered ? SourceDecline.Absent : SourceDecline.Shorter;
        return best;
    }

    private void EnsureBfs(int n)
    {
        if (_bfs.Length < n) Array.Resize(ref _bfs, Math.Max(n, _bfs.Length * 2));
    }

    private Candidate QualifyTruncation(in MatchRequest r, MatchPlan plan, RadixNode d, int target, int publicCap, IWaitingPlanView? waiting, ref ClampReasons clamps)
    {
        if (!ResumabilityRules.Permitted(d.ScopeIx, target, r.ScopeIx, publicCap))
        {
            SetDeclineIfNone(plan, CandidateKind.TruncatedEndState, SourceDecline.NotPermitted);
            return default;
        }
        if (!Rules.RewindWithinCap(d.Depth, target))
        {
            clamps |= ClampReasons.RewindCap;
            SetDeclineIfNone(plan, CandidateKind.TruncatedEndState, SourceDecline.Clamped);
            return default;
        }
        if (!Rules.TruncationAllows(d.Depth, target))
        {
            SetDeclineIfNone(plan, CandidateKind.TruncatedEndState, SourceDecline.Clamped);
            return default;
        }
        bool primary = d.EndState!.Kind == EndStateKind.PrimaryResident;
        if (primary && (!r.PrimaryAvailable || d.StateLockRef > 0))
        {
            SetDeclineIfNone(plan, CandidateKind.TruncatedEndState, r.PrimaryAvailable ? SourceDecline.PrimaryClaimed : SourceDecline.PrimaryBusy);
            return default;
        }
        if (!primary && _options.PayloadValidator is not null && !_options.PayloadValidator.CanMaterialize(d.EndState.Key, d.Depth, target))
        {
            SetDeclineIfNone(plan, CandidateKind.TruncatedEndState, SourceDecline.ModelRefused);
            return default;
        }
        MaterializeMode mode = DonationDecision(d, target, primary, waiting);
        if (mode == MaterializeMode.None)
        {
            SetDeclineIfNone(plan, CandidateKind.TruncatedEndState, SourceDecline.DonateOnlyShared);
            return default;
        }
        return new Candidate { Kind = CandidateKind.TruncatedEndState, Mode = mode, Length = target, Payload = d, TieId = d.Id };
    }

    /// <summary>Among truncation candidates at one target: the smallest depth, then tie order, then the older node.</summary>
    private static Candidate BetterTruncation(in Candidate x, in Candidate y)
    {
        if (!x.IsValid) return y;
        if (!y.IsValid) return x;
        if (x.Length != y.Length) return x.Length > y.Length ? x : y;
        if (x.Payload!.Depth != y.Payload!.Depth) return x.Payload.Depth < y.Payload.Depth ? x : y;
        int rx = ResumabilityRules.TieRank(x.Kind, x.Mode), ry = ResumabilityRules.TieRank(y.Kind, y.Mode);
        if (rx != ry) return rx < ry ? x : y;
        return y.TieId < x.TieId ? y : x;
    }

    /// <summary>
    /// Donation decision (§5.3.4, P8 + DEC-14). Returns the mode, or <see cref="MaterializeMode.None"/>
    /// when the candidate is rejected (a donate-only payload that cannot be donated).
    /// </summary>
    internal MaterializeMode DonationDecision(RadixNode x, int length, bool primary, IWaitingPlanView? waiting)
    {
        bool donatable = DonationConditionsHold(x, length, primary, waiting);
        if (primary)
        {
            if (donatable) return MaterializeMode.KeepPrimary;
            return Caps.AdoptPrimaryOnDisplacement && Caps.EndState == EndStateSupport.CopyAndDonate
                ? MaterializeMode.ConvertPrimaryThenClone
                : MaterializeMode.None;
        }
        if (donatable) return MaterializeMode.DonateEndState;
        return Caps.EndState == EndStateSupport.CopyAndDonate ? MaterializeMode.CloneEndState : MaterializeMode.None;
    }

    /// <summary>Conditions (a)-(f). For a PrimaryResident, (e) reads <c>Caps.PrimaryResident</c>.</summary>
    internal bool DonationConditionsHold(RadixNode x, int length, bool primary, IWaitingPlanView? waiting)
    {
        if (x.Children.Count != 0) return false;                                              // (a)
        if (x.LockRef != 0 || x.StateLockRef != 0 || x.PinRef != 0) return false;             // (b)
        if (x.ScopeIx == 0) return false;                                                     // (c)
        if (primary ? !Caps.PrimaryResident                                                   // (e)
                    : Caps.EndState != EndStateSupport.DonateOnly && Caps.EndState != EndStateSupport.CopyAndDonate)
            return false;
        if (x.Depth - length > _options.DonateTruncateSlackTokens) return false;              // (f)
        ScopeRecord rec = Scopes[x.ScopeIx];                                                  // (d)
        if (rec.WaitingRequests > 1)
        {
            if (waiting is null || rec.WaitingRequests - 1 > _options.DonationWaiterScanMax) return false;
            if (!waiting.NoOtherWaiterTargets(x.ScopeIx, x, Version)) return false;
        }
        return true;
    }

    /// <summary>
    /// Helper for <see cref="IWaitingPlanView"/> implementations: every other waiting request of the
    /// scope has a plan at the current version whose payload is not <paramref name="x"/> (at most 128
    /// are checked; any unknown fails the condition).
    /// </summary>
    internal static bool OtherWaitersClear(ReadOnlySpan<MatchPlan?> otherPlans, int expectedOthers, RadixNode x, long version, int scanMax = 128)
    {
        if (expectedOthers > scanMax || otherPlans.Length < expectedOthers) return false;
        for (int i = 0; i < expectedOthers; i++)
        {
            MatchPlan? p = otherPlans[i];
            if (p is null || p.Version != version || ReferenceEquals(p.PayloadNode, x)) return false;
        }
        return true;
    }

    private static (RadixNode Node, int Offset) LocateOnTrail(MatchPlan plan, int trailEnd, int length)
    {
        for (int ix = plan.TrailEnds[trailEnd]; ix >= 0; ix = plan.Trail[ix].Prev)
        {
            TrailEntry te = plan.Trail[ix];
            if (te.StartDepth < length && length <= te.EndDepth)
                return (te.Node, length - te.Node.EdgeStartDepth);
        }
        throw new InvalidOperationException($"Length {length} is not on the matched path.");
    }

    private static int PublicTokensOnPath(RadixNode anchor, int length, int publicCap)
    {
        RadixNode? n = anchor;
        while (n is not null && !n.IsRoot && n.ScopeIx != 0) n = n.Parent;
        int publicDepth = n is null || n.IsRoot ? 0 : Math.Min(n.Depth, length);
        return Math.Min(publicDepth, publicCap);
    }

    private int ComputeBlockedByScope(in MatchRequest r, MatchPlan plan)
    {
        int best = 0;
        for (int t = 0; t < plan.TrailEnds.Count; t++)
        {
            TrailEntry te = plan.Trail[plan.TrailEnds[t]];
            RadixNode at = te.Full ? te.Node : te.Node.Parent!;
            int d = te.Full ? te.EndDepth : te.StartDepth;
            if (d >= r.Key.Length) continue;
            long e = r.Key[d];
            int checkedChildren = 0;
            foreach (RadixNode child in at.Children)
            {
                if (checkedChildren >= 8) break;
                if (child.ScopeIx == 0 || child.ScopeIx == r.ScopeIx || child.Edge[0] != e) continue;
                checkedChildren++;
                int len = KeyCompare.CommonPrefixLength(child.Edge, 0, r.Key, d, r.Key.Length - d);
                best = Math.Max(best, d + len - plan.Structural);
            }
        }
        return Math.Max(0, best);
    }

    // ------------------------------------------------------------------ Acquire, SplitAt, locks

    /// <summary>
    /// Acquire (§5.4): split at the plan's length, path-lock the anchor and state-lock the payload
    /// node. Throws <see cref="StalePrefixMatchException"/> when the tree changed since the plan.
    /// </summary>
    internal LockReceipt Acquire(MatchPlan plan)
    {
        if (plan.Version != Version)
        {
            Counters.StaleMatches++;
            throw new StalePrefixMatchException(plan.Version, Version);
        }
        if (plan.Kind == CandidateKind.None || plan.Length <= 0)
            return default;
        RadixNode anchor = plan.AnchorParent ?? throw new InvalidOperationException("Plan has no anchor.");
        if (plan.AnchorOffset < anchor.Edge.Length)
            anchor = SplitAt(anchor, plan.AnchorOffset);
        LockPath(anchor);
        RadixNode? x = plan.PayloadNode;
        if (x is not null)
            LockState(x);
        TouchPath(anchor);
        if (x is not null && !ReferenceEquals(x, anchor)) Touch(x);
        Version++;
        return IssueReceipt(anchor, x);
    }

    /// <summary>Path lock only, anchored at <paramref name="anchor"/> (publication and durable lock moves).</summary>
    internal LockReceipt AcquirePath(RadixNode anchor)
    {
        if (anchor is null || anchor.IsRoot) return default;
        RequireInTree(anchor);
        LockPath(anchor);
        TouchPath(anchor);
        Version++;
        return IssueReceipt(anchor, null);
    }

    /// <summary>State lock only, on <paramref name="node"/>'s end state.</summary>
    internal LockReceipt AcquireState(RadixNode node)
    {
        if (node is null || node.IsRoot) return default;
        RequireInTree(node);
        LockState(node);
        Touch(node);
        Version++;
        return IssueReceipt(null, node);
    }

    /// <summary>MoveDurableLock (P5): acquire the new path first, then release the old one.</summary>
    internal void MoveDurableLock(ref LockReceipt durable, RadixNode newAnchor)
    {
        LockReceipt next = AcquirePath(newAnchor);
        Release(ref durable);
        durable = next;
    }

    private LockReceipt IssueReceipt(RadixNode? path, RadixNode? state)
    {
        var receipt = new LockReceipt(TreeSerial, ++_receiptSerial, path, state);
        Ledger?.Add(receipt);
        return receipt;
    }

    private void RequireInTree(RadixNode n)
    {
        if (!n.InTree) throw new InvalidOperationException("Node is not in the tree.");
    }

    private void LockPath(RadixNode anchor)
    {
        for (RadixNode n = anchor; !n.IsRoot; n = n.Parent!)
        {
            if (n.LockRef++ == 0)
            {
                _protected += PageBytesOf(n);
                Relink(n);
            }
        }
    }

    private void LockState(RadixNode x)
    {
        ResourceVector before = ProtectedOf(x);
        x.StateLockRef++;
        _protected += ProtectedOf(x) - before;
        Relink(x);
    }

    /// <summary>
    /// Release (§4.5): replays the receipt and sets it to <c>default</c>. A default receipt is a
    /// no-op; a receipt of another epoch or a recycled node is ignored and counted (stale_receipts).
    /// </summary>
    internal void Release(ref LockReceipt r)
    {
        if (r.IsEmpty) { r = default; return; }
        if (!ValidateReceipt(r)) { r = default; return; }
        Ledger?.Remove(r.Serial);
        bool changed = false;
        if (r.PathAnchor is not null) changed |= UnlockPath(r.PathAnchor);
        if (r.StateAnchor is not null) changed |= UnlockState(r.StateAnchor);
        RadixNode? pathAnchor = r.PathAnchor, stateAnchor = r.StateAnchor;
        r = default;
        if (stateAnchor is not null) changed |= CollectFrom(stateAnchor);
        if (pathAnchor is not null) changed |= CollectFrom(pathAnchor);
        if (changed) Version++;
    }

    /// <summary>Releases only the state lock of a receipt, keeping its path lock (§5.6 clone).</summary>
    internal void ReleaseState(ref LockReceipt r)
    {
        if (r.StateAnchor is null) return;
        if (!ValidateReceipt(r)) { r = default; return; }
        RadixNode state = r.StateAnchor;
        bool changed = UnlockState(state);
        var remaining = new LockReceipt(r.TreeSerial, r.Serial, r.PathAnchor, null);
        if (r.PathAnchor is null) Ledger?.Remove(r.Serial); else Ledger?.Replace(remaining);
        r = r.PathAnchor is null ? default : remaining;
        changed |= CollectFrom(state);
        if (changed) Version++;
    }

    private bool ValidateReceipt(in LockReceipt r)
    {
        bool stale = r.TreeSerial != TreeSerial
            || (r.PathAnchor is not null && (!r.PathAnchor.InTree || r.PathAnchor.Generation != r.PathGeneration))
            || (r.StateAnchor is not null && (!r.StateAnchor.InTree || r.StateAnchor.Generation != r.StateGeneration))
            || (Ledger is not null && !Ledger.Contains(r.Serial));
        if (!stale) return true;
        Counters.StaleReceipts++;
        if (_options.StrictReceipts)
            throw new InvalidOperationException($"Stale lock receipt {r} (tree serial {TreeSerial}).");
        return false;
    }

    private bool UnlockPath(RadixNode anchor)
    {
        bool changed = false;
        for (RadixNode n = anchor; !n.IsRoot; n = n.Parent!)
        {
            if (n.LockRef <= 0) throw new InvalidOperationException($"LockRef underflow on {n}.");
            if (--n.LockRef == 0)
            {
                _protected -= PageBytesOf(n);
                Relink(n);
                changed = true;
            }
        }
        return changed;
    }

    private bool UnlockState(RadixNode x)
    {
        if (x.StateLockRef <= 0) throw new InvalidOperationException($"StateLockRef underflow on {x}.");
        ResourceVector before = ProtectedOf(x);
        x.StateLockRef--;
        _protected += ProtectedOf(x) - before;
        Relink(x);
        return x.StateLockRef == 0;
    }

    /// <summary>
    /// SplitAt (§5.4): <c>0 &lt; k &lt; c.Edge.Length</c>. The new parent p ends at depth
    /// <c>c.EdgeStart + k</c>; it copies LockRef, HitCount and LastAccess, receives the pages and
    /// media records that end or start before its depth, and is a state tombstone (K4).
    /// </summary>
    internal RadixNode SplitAt(RadixNode c, int k)
    {
        if (c.IsRoot || k <= 0 || k >= c.Edge.Length)
            throw new ArgumentOutOfRangeException(nameof(k), k, $"Split offset must lie inside the edge of {c}.");
        RadixNode parent = c.Parent!;
        RadixNode p = _nodePool.Rent();
        p.Id = ++_nodeSerial;
        p.InTree = true;
        p.Edge = new KeySlice(c.Edge.Rope, c.Edge.Start, k);
        p.Depth = c.EdgeStartDepth + k;
        p.ScopeIx = c.ScopeIx;
        p.LockRef = c.LockRef;
        p.HitCount = c.HitCount;
        p.LastAccess = c.LastAccess;
        p.CreatedTick = c.CreatedTick;
        p.Parent = parent;
        parent.Children.Replace(c, p);
        c.Edge = new KeySlice(c.Edge.Rope, c.Edge.Start + k, c.Edge.Length - k);
        c.Parent = p;
        p.Children.Add(c);

        // Pages whose last token now lies in p: a prefix of c's sorted pages.
        int blockSize = _options.BlockSize;
        int moved = 0;
        ResourceVector movedBytes = default;
        while (moved < c.PageCount && (c.Pages![moved].PageIndex + 1) * blockSize <= p.Depth)
        {
            p.AddPage(c.Pages[moved]);
            _blockOwners[c.Pages[moved].Block] = p;
            movedBytes += PageBytes(c.Pages[moved]);
            moved++;
        }
        for (int i = 0; i < moved; i++) c.RemovePageAt(0);
        p.Bytes = movedBytes;
        c.Bytes -= movedBytes;
        // Media records whose Start lies in p.
        int movedRecords = 0;
        while (movedRecords < c.MediaSpanCount && c.MediaSpans![movedRecords].Start < p.Depth)
        {
            p.AddSpanRecord(c.MediaSpans[movedRecords]);
            movedRecords++;
        }
        if (movedRecords > 0)
        {
            for (int i = movedRecords; i < c.MediaSpanCount; i++) c.MediaSpans![i - movedRecords] = c.MediaSpans[i];
            c.MediaSpanCount -= movedRecords;
        }
        AddToScope(p);                     // p's bytes came from c: the scope total is unchanged
        AttachRopeSlice(p);
        p.Edge.Rope.LiveSliceTokens -= k;  // c's edge shrank by the k tokens p now covers
        NodeCount++;
        Counters.Splits++;
        Counters.NodesCreated++;
        Relink(p);
        Relink(c);
        // A pages-only leaf whose pages all moved to p is now an unlocked payload-less leaf (I5): it only
        // repeated key structure below p, so it goes (p itself keeps its pages and stays).
        if (c.Children.Count == 0 && !c.AnyLock && !c.HasPayload && !c.IsDonationPending)
            DeleteNode(c, ReleaseReason.Evicted);
        Version++;
        return p;
    }

    // ------------------------------------------------------------------ Insert and attach

    /// <summary>
    /// Insert (§5.8): <c>[0, P)</c> goes to public nodes and <c>[P, len)</c> to the scope, with a node
    /// boundary forced at P (flagged <see cref="NodeFlags.IsPublicBoundary"/>). Returns the node ending
    /// at <paramref name="length"/>, or a shallower node when a 63-bit media collision stopped the
    /// walk (counted in media_hash_collisions). The returned node may be a payload-less leaf; call
    /// <see cref="CollectIfEmpty"/> when nothing is attached to it.
    /// </summary>
    internal RadixNode Insert(KeyRope key, int length, int scopeIx, int publicBoundary, NodeFlags endFlags, MediaSpanRecord[]? spans)
    {
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (length <= 0 || length > key.Length)
            throw new ArgumentOutOfRangeException(nameof(length), length, $"Insert length must lie in [1, {key.Length}].");
        if (length > _options.ContextLength)
            throw new ArgumentOutOfRangeException(nameof(length), length, $"Insert length exceeds the context length {_options.ContextLength} (I24).");
        if (publicBoundary < 0) throw new ArgumentOutOfRangeException(nameof(publicBoundary));
        if (scopeIx == 0 && length > publicBoundary)
            throw new ArgumentException("A public insert must end at or before its public boundary (S1).", nameof(scopeIx));
        if (scopeIx != 0 && (!Scopes.IsLive(scopeIx) || Scopes[scopeIx].Retired))
            throw new ArgumentException("Insert into a scope that is not live.", nameof(scopeIx));
        if ((endFlags & (NodeFlags.DonationPending | NodeFlags.Retired | NodeFlags.IsPublicBoundary)) != 0)
            throw new ArgumentException("Insert end flags may not set DonationPending, Retired or IsPublicBoundary.", nameof(endFlags));

        RadixNode node = Root;
        RadixNode? boundary = null;
        int d = 0;
        bool truncated = false;
        while (d < length)
        {
            int want = d < publicBoundary ? 0 : scopeIx;
            int limit = want == 0 ? Math.Min(publicBoundary, length) : length;
            RadixNode? c = node.Children.Find(want, key[d]);
            if (c is null)
            {
                c = NewLeaf(node, key, d, limit - d, want, spans);
                node = c;
                d = limit;
            }
            else
            {
                int cpl = KeyCompare.CommonPrefixLength(c.Edge, 0, key, d, limit - d);
                int m = Math.Min(MediaVerify(c, d, cpl, spans), limit - d);
                if (m == 0)
                {
                    Counters.MediaHashCollisions++;
                    truncated = true;
                    break;
                }
                if (m < c.Edge.Length) c = SplitAt(c, m);
                node = c;
                d += m;
                node.HitCount++;
                Touch(node);
            }
            if (want == 0 && d == publicBoundary) boundary = node;
        }
        Counters.Inserts++;
        if (!truncated)
        {
            if (boundary is not null && !boundary.IsPublicBoundary)
            {
                boundary.Flags |= NodeFlags.IsPublicBoundary;
                if (boundary.EndState is not null) TrackPublicBoundary(boundary);
                Relink(boundary);
            }
            if (endFlags != NodeFlags.None && !node.IsRoot)
            {
                node.Flags |= endFlags;
                Relink(node);
            }
            if (!node.IsRoot && node.ScopeIx != 0)
                SetNewestLeaf(Scopes[node.ScopeIx], node);
        }
        Version++;
        return node;
    }

    private RadixNode NewLeaf(RadixNode parent, KeyRope key, int start, int length, int scopeIx, MediaSpanRecord[]? spans)
    {
        RadixNode n = _nodePool.Rent();
        n.Id = ++_nodeSerial;
        n.InTree = true;
        n.Parent = parent;
        n.Edge = new KeySlice(key, start, length);
        n.Depth = start + length;
        n.ScopeIx = scopeIx;
        n.CreatedTick = n.LastAccess = ++_tick;
        n.HitCount = 1;
        if (spans is not null)
        {
            for (int i = LowerBound(spans, start); i < spans.Length && spans[i].Start < n.Depth; i++)
                n.AddSpanRecord(spans[i]);
        }
        parent.Children.Add(n);
        AddToScope(n);
        AttachRopeSlice(n);
        NodeCount++;
        Counters.NodesCreated++;
        Relink(parent);
        return n;
    }

    /// <summary>
    /// Attaches an end state (§5.7, §5.9, §5.10). A live end state already at the node makes the
    /// incoming key a duplicate (queued for release); a tombstone is revived. An end state strictly
    /// inside a media span is refused (I13).
    /// </summary>
    internal AttachResult AttachEndState(RadixNode node, EndStatePayload payload)
    {
        if (node is null || node.IsRoot) throw new ArgumentException("End states attach to non-root nodes.", nameof(node));
        RequireInTree(node);
        if (payload is null) throw new ArgumentNullException(nameof(payload));
        if (payload.Key is null || !payload.Key.StartsWith("pc:", StringComparison.Ordinal))
            throw new ArgumentException("Payload keys are tree-minted and start with \"pc:\" (I10).", nameof(payload));
        if (_keyIndex.ContainsKey(payload.Key) || Reclaim.Contains(payload.Key))
            throw new ArgumentException($"Payload key {payload.Key} is already in use (I10).", nameof(payload));
        if (payload.Kind == EndStateKind.PrimaryResident && (_primaryResidents > 0 && node.EndState?.Kind != EndStateKind.PrimaryResident))
            throw new InvalidOperationException("At most one PrimaryResident may exist (I14).");
        if (payload.Persisted && (!node.IsPublicBoundary || PathHasMedia(node)))
            throw new ArgumentException("Only public boundary end states without media may be persisted (I20).", nameof(payload));
        if (node.EndState is not null)
        {
            Counters.DuplicatesDropped++;
            Reclaim.Enqueue(payload.Key, ReleaseReason.Duplicate, payload.Bytes);
            return AttachResult.Duplicate;
        }
        if (IsInsideSpan(node, node.Depth))
        {
            Counters.EndStatesRefused++;
            return AttachResult.Refused;
        }
        bool revived = node.HadEndState || node.Children.Count > 0 || node.PageCount > 0;
        ResourceVector before = ProtectedOf(node);
        node.EndState = payload;
        ResourceVector bytes = payload.Bytes;
        node.Bytes += bytes;
        _cached += bytes;
        ScopeBytesAdjust(node.ScopeIx, bytes);
        _protected += ProtectedOf(node) - before;
        _keyIndex.Add(payload.Key, node);
        CountEndState(node, payload, +1);
        Touch(node);
        if (node.IsPublicBoundary) TrackPublicBoundary(node);
        Relink(node);
        Version++;
        return revived ? AttachResult.Revived : AttachResult.Attached;
    }

    /// <summary>
    /// Attaches pages (§4.4 owner rule, §5.10): each page goes to the node on <paramref name="node"/>'s
    /// root path whose edge contains its last token. Pages already owned there are skipped, and a page
    /// whose store of record is not backed by the block flags is refused (I12, G-03). The tree takes
    /// one block reference per attached page. Returns the number attached.
    /// </summary>
    internal int AttachPages(RadixNode node, ReadOnlySpan<PageRef> pages)
    {
        if (node is null || node.IsRoot) throw new ArgumentException("Pages attach to non-root nodes.", nameof(node));
        RequireInTree(node);
        int blockSize = _options.BlockSize;
        int attached = 0;
        for (int i = 0; i < pages.Length; i++)
        {
            PageRef page = pages[i];
            int end = (page.PageIndex + 1) * blockSize;
            if (page.PageIndex < 0 || end > node.Depth || page.Block is null)
                throw new ArgumentOutOfRangeException(nameof(pages), $"Page {page.PageIndex} does not lie on the path of {node}.");
            if (!StoreBacked(page))
            {
                Counters.PagesRefused++;
                continue;
            }
            RadixNode owner = node;
            while (owner.EdgeStartDepth > end - 1) owner = owner.Parent!;
            if (OwnsPage(owner, page.PageIndex) || BlockOwnedElsewhere(page.Block))
            {
                Counters.PagesDuplicate++;
                continue;
            }
            _pages.RetainPage(page.Block);
            ResourceVector before = ProtectedOf(owner);
            owner.AddPage(page);
            ResourceVector bytes = PageBytes(page);
            owner.Bytes += bytes;
            _cached += bytes;
            ScopeBytesAdjust(owner.ScopeIx, bytes);
            _protected += ProtectedOf(owner) - before;
            _blockOwners[page.Block] = owner;
            Relink(owner);
            attached++;
        }
        if (attached > 0) Version++;
        return attached;
    }

    private readonly Dictionary<KvBlock, RadixNode> _blockOwners = new(ReferenceEqualityComparer.Instance);

    internal bool TryGetBlockOwner(KvBlock block, out RadixNode owner) => _blockOwners.TryGetValue(block, out owner!);

    private bool BlockOwnedElsewhere(KvBlock block) => _blockOwners.ContainsKey(block);

    private static bool OwnsPage(RadixNode owner, int pageIndex)
    {
        ReadOnlySpan<PageRef> pages = owner.PageSpan;
        for (int i = 0; i < pages.Length; i++)
            if (pages[i].PageIndex == pageIndex) return true;
        return false;
    }

    internal bool StoreBacked(in PageRef page)
    {
        if (page.Store != PageStore.A1HostSlab && page.Store != PageStore.A2ModelPaged && page.Store != PageStore.Both) return false;
        if (page.HasA1 && (!_pages.HoldsSnapshotBytes(page.Block) || _pages.UsedTokens(page.Block) < _options.BlockSize)) return false;
        if (page.HasA2 && !_pages.HoldsModelPagedKv(page.Block)) return false;
        return true;
    }

    /// <summary>
    /// Detaches an end state (tombstone). The key is queued for a batched release unless the model
    /// already freed the payload (<paramref name="enqueue"/> false: deferred invalidation, §5.17).
    /// </summary>
    internal ResourceVector DetachEndState(RadixNode n, ReleaseReason reason, bool enqueue = true)
    {
        EndStatePayload? es = n.EndState;
        if (es is null) return default;
        ResourceVector before = ProtectedOf(n);
        ResourceVector bytes = es.Bytes;
        n.EndState = null;
        n.HadEndState = true;
        n.Flags &= ~NodeFlags.DonationPending;
        n.Bytes -= bytes;
        _cached -= bytes;
        ScopeBytesAdjust(n.ScopeIx, ResourceVector.Negate(bytes));
        _protected += ProtectedOf(n) - before;
        _keyIndex.Remove(es.Key);
        CountEndState(n, es, -1);
        if (enqueue) Reclaim.Enqueue(es.Key, reason, bytes);
        if (n.IsPublicBoundary) UntrackPublicBoundary(n);
        Relink(n);
        Counters.StateDetaches++;
        Version++;
        return bytes;
    }

    /// <summary>Marks an end state as moved to an admitting request (§5.6). Bytes stay protected until commit.</summary>
    internal void MarkDonationPending(RadixNode x)
    {
        if (x.EndState is null) throw new InvalidOperationException("DonationPending needs an end state.");
        ResourceVector before = ProtectedOf(x);
        x.Flags |= NodeFlags.DonationPending;
        _protected += ProtectedOf(x) - before;
        Relink(x);
        Version++;
    }

    /// <summary>Rollback of a donation: the end state is visible again.</summary>
    internal void CancelDonation(RadixNode x)
    {
        if (!x.IsDonationPending) return;
        ResourceVector before = ProtectedOf(x);
        x.Flags &= ~NodeFlags.DonationPending;
        _protected += ProtectedOf(x) - before;
        Relink(x);
        Version++;
    }

    /// <summary>
    /// Commit of a donation (§5.6): the payload now belongs to the request, so its bytes leave the
    /// tree without a release; the node becomes a tombstone.
    /// </summary>
    internal void CommitDonation(RadixNode x)
    {
        if (!x.IsDonationPending || x.EndState is null) throw new InvalidOperationException("No pending donation on this node.");
        DetachEndState(x, ReleaseReason.Rollback, enqueue: false);
        CollectFrom(x);   // an unlocked payload-less leaf goes now; a locked one at its last unlock
    }

    /// <summary>Deletes <paramref name="node"/> (and payload-less unlocked parents) when it is an unlocked payload-less leaf.</summary>
    internal bool CollectIfEmpty(RadixNode node)
    {
        bool changed = CollectFrom(node);
        if (changed) Version++;
        return changed;
    }

    /// <summary>Lane pin (M7c): protects a node's end state from eviction while a lane job reads it.</summary>
    internal void Pin(RadixNode node)
    {
        if (node.EndState is null) throw new InvalidOperationException("Only an end state can be pinned.");
        ResourceVector before = ProtectedOf(node);
        node.PinRef++;
        _protected += ProtectedOf(node) - before;
        Relink(node);
        Version++;
    }

    internal void Unpin(RadixNode node)
    {
        if (node.PinRef <= 0) throw new InvalidOperationException("PinRef underflow.");
        ResourceVector before = ProtectedOf(node);
        node.PinRef--;
        _protected += ProtectedOf(node) - before;
        Relink(node);
        CollectFrom(node);
        Version++;
    }

    // ------------------------------------------------------------------ eviction

    /// <summary>
    /// Evict (§5.11): leaf-first LRU, tier by tier up to <paramref name="ceiling"/>, until
    /// <paramref name="need"/> units of <paramref name="cls"/> are freed.
    /// </summary>
    internal bool Evict(ResourceClass cls, long need, ReleaseReason why, EvictionTier ceiling)
    {
        Counters.EvictCalls++;
        if (need <= 0) return true;
        long freed = 0;
        for (int t = 0; t <= (int)ceiling && freed < need; t++)
        {
            var tier = (EvictionTier)t;
            while (freed < need)
            {
                RadixNode? a = FirstWithBytes(EvictionLists.ListId(LruKind.Leaf, tier), cls, stateOnly: false);
                RadixNode? b = FirstWithBytes(EvictionLists.ListId(LruKind.State, tier), cls, stateOnly: true);
                if (a is null && b is null) break;
                if (a is not null && (b is null || a.LastAccess <= b.LastAccess))
                {
                    freed += DeleteLeafCascade(a, why)[cls];
                }
                else
                {
                    freed += DetachEndState(b!, why)[cls];
                    CollectFrom(b!);
                }
                Counters.Evictions++;
            }
        }
        return freed >= need;
    }

    private RadixNode? FirstWithBytes(byte listId, ResourceClass cls, bool stateOnly)
    {
        for (RadixNode? n = Lists.First(listId); n is not null; n = n.LruNext)
        {
            long bytes = stateOnly ? n.EndState!.Bytes[cls] : n.Bytes[cls];
            if (bytes > 0) return n;
            Counters.EvictScanSkips++;
        }
        return null;
    }

    /// <summary>
    /// EffectiveCap (§5.11, DEC-31): min(option cap, family sub-cap, cached + spare − running reserve).
    /// </summary>
    internal long EffectiveCap(ResourceClass cls)
    {
        long cap = long.MaxValue;
        if (cls == ResourceClass.PoolPages)
            return _options.PoolPagesCap > 0 ? _options.PoolPagesCap : long.MaxValue;
        if (_optionCap[cls] > 0) cap = Math.Min(cap, _optionCap[cls]);
        long sub = Caps.SubCapBytes[cls];
        if (sub > 0) cap = Math.Min(cap, sub);
        long spare = _options.QuerySpareBytes?.Invoke(cls) ?? -1;
        if (spare >= 0) cap = Math.Min(cap, _cached[cls] + Math.Max(0, spare - _runningReserve[cls]));
        return cap;
    }

    private static readonly ResourceClass[] s_byteClasses =
        { ResourceClass.HostKv, ResourceClass.DeviceKv, ResourceClass.StateSnapshot, ResourceClass.NativeSlot };

    private void ResolveOptionCaps()
    {
        ResourceVector v = _options.OptionCapBytes;
        foreach (ResourceClass c in new[] { ResourceClass.DeviceKv, ResourceClass.StateSnapshot, ResourceClass.NativeSlot })
        {
            if (v[c] > 0) continue;
            long spare = _options.QuerySpareBytes?.Invoke(c) ?? -1;
            if (spare >= 0) v[c] = Math.Max(1, spare / 2);
        }
        if (v.HostKv <= 0 && _options.HostRamBytes > 0) v.HostKv = Math.Max(1, _options.HostRamBytes / 4);
        _optionCap = v;
    }

    /// <summary>The auto-resolved option caps (0 = unbounded).</summary>
    internal ResourceVector OptionCaps => _optionCap;

    /// <summary>Sets the bytes running requests reserve for decode growth (subtracted from spare).</summary>
    internal void SetRunningReserve(ResourceVector reserve)
    {
        _runningReserve = reserve;
    }

    /// <summary>Re-evaluates the per-scope quota and scope activity (tier inputs), relinking what changed.</summary>
    internal void RefreshBudgets()
    {
        long total = 0;
        bool bounded = true;
        foreach (ResourceClass c in s_byteClasses)
        {
            long cap = EffectiveCap(c);
            if (cap == long.MaxValue) { bounded = false; break; }
            total += cap;
        }
        RefreshScopeActivity();
        int active = 0;
        foreach (ScopeRecord rec in Scopes.LiveRecords())
            if (rec.Index != 0 && !rec.Retired && rec.Active) active++;
        long quota = bounded ? total / Math.Max(4, active) : long.MaxValue;
        if (quota != _scopeQuota)
        {
            _scopeQuota = quota;
            foreach (ScopeRecord rec in Scopes.LiveRecords())
                if (rec.NewestLeaf is not null) Relink(rec.NewestLeaf);
        }
    }

    /// <summary>Evicts until every class is within its effective cap (I21). Returns false when a cap still is exceeded.</summary>
    internal bool EnforceCaps(EvictionTier ceiling, ReleaseReason why = ReleaseReason.Evicted)
    {
        RefreshBudgets();
        bool ok = true;
        for (int i = 0; i < ResourceVector.ClassCount; i++)
        {
            var c = (ResourceClass)i;
            long cap = EffectiveCap(c);
            if (_cached[c] > cap)
                ok &= Evict(c, _cached[c] - cap, why, ceiling);
        }
        return ok;
    }

    /// <summary>Count sub-caps (DEC-20): scoped end states, public end states, native slots. Oldest first, ceiling ScopeNewest.</summary>
    internal bool EnforceCountSubCaps()
    {
        bool ok = true;
        ok &= EnforceCount(() => _options.ScopedEndStateLeavesMax > 0 && _scopedEndStates > _options.ScopedEndStateLeavesMax,
                           n => n.ScopeIx != 0 && n.EndState!.Kind != EndStateKind.PrimaryResident);
        ok &= EnforceCount(() => _publicEndStates > _options.PublicMax,
                           n => n.ScopeIx == 0 && n.EndState!.Kind != EndStateKind.PrimaryResident);
        ok &= EnforceCount(() => Caps.MaxRetainedNativeSlots > 0 && _nativeSlots > Caps.MaxRetainedNativeSlots,
                           n => n.EndState!.Kind == EndStateKind.NativeSlot);
        return ok;
    }

    private bool EnforceCount(Func<bool> exceeded, Func<RadixNode, bool> category)
    {
        while (exceeded())
        {
            RadixNode? victim = null;
            for (int t = 0; t <= (int)EvictionTier.ScopeNewest; t++)
            {
                for (int k = 1; k <= 2; k++)
                {
                    for (RadixNode? n = Lists.First((LruKind)k, (EvictionTier)t); n is not null; n = n.LruNext)
                    {
                        if (n.EndState is null || !category(n)) continue;
                        if (victim is null || n.LastAccess < victim.LastAccess) victim = n;
                        break;
                    }
                }
            }
            if (victim is null) return false;
            DetachEndState(victim, ReleaseReason.Evicted);
            CollectFrom(victim);
            Counters.Evictions++;
        }
        return true;
    }

    /// <summary>
    /// RelieveMemoryPressure (§5.11 triggers). Moderate: every class to 50% of its cap, never above
    /// tier Breakpoint. Critical: everything except the newest PublicTop node and the most recently
    /// used scope's newest leaf.
    /// </summary>
    internal void RelieveMemoryPressure(PressureLevel level)
    {
        RefreshBudgets();
        if (level == PressureLevel.Moderate)
        {
            for (int i = 0; i < ResourceVector.ClassCount; i++)
            {
                var c = (ResourceClass)i;
                long cap = EffectiveCap(c);
                long target = cap == long.MaxValue ? long.MaxValue : cap / 2;
                if (_cached[c] > target)
                    Evict(c, _cached[c] - target, ReleaseReason.Pressure, EvictionTier.Breakpoint);
            }
            return;
        }
        RadixNode? keepPublic = null;
        foreach (RadixNode n in _publicBoundaries)
            if (n.InPublicTop && (keepPublic is null || n.LastAccess > keepPublic.LastAccess)) keepPublic = n;
        RadixNode? keepScoped = null;
        long newestUse = long.MinValue;
        foreach (ScopeRecord rec in Scopes.LiveRecords())
        {
            if (rec.Index == 0 || rec.NewestLeaf is null) continue;
            if (rec.LastUseTick > newestUse) { newestUse = rec.LastUseTick; keepScoped = rec.NewestLeaf; }
        }
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (byte id = 1; id <= 2 * EvictionLists.TierCount; id++)
            {
                RadixNode? n = Lists.First(id);
                while (n is not null && (ReferenceEquals(n, keepPublic) || ReferenceEquals(n, keepScoped))) n = n.LruNext;
                if (n is null) continue;
                if (EvictionLists.KindOf(id) == LruKind.Leaf) DeleteLeafCascade(n, ReleaseReason.Pressure);
                else { DetachEndState(n, ReleaseReason.Pressure); CollectFrom(n); }
                Counters.Evictions++;
                progress = true;
                break;
            }
        }
    }

    // ------------------------------------------------------------------ scope retirement, invalidation, reset

    /// <summary>RetireScope (§5.17). Returns false when the scope is unknown.</summary>
    internal bool RetireScope(ScopeId id)
    {
        if (id.IsPublic || !Scopes.TryGetIndex(id, out int ix)) return false;
        ScopeRecord rec = Scopes[ix];
        rec.Retired = true;
        Counters.ScopesRetired++;
        int count = 0;
        for (RadixNode? n = rec.FirstNode; n is not null; n = n.ScopeNext)
        {
            if (_scratch.Length == count) Array.Resize(ref _scratch, _scratch.Length * 2);
            _scratch[count++] = n;
        }
        Array.Sort(_scratch, 0, count, DepthDescending.Instance);
        for (int i = 0; i < count; i++)
        {
            RadixNode n = _scratch[i];
            _scratch[i] = null!;
            if (!n.InTree) continue;
            if (n.AnyLock || n.IsDonationPending || SubtreeLocked(n))
            {
                n.Flags |= NodeFlags.Retired;
                if (!n.AnyLock && !n.IsDonationPending && n.EndState is not null && n.Children.Count > 0)
                    DetachEndState(n, ReleaseReason.ScopeRetired);
                Relink(n);
            }
            else if (n.Children.Count == 0)
            {
                DeleteLeafCascade(n, ReleaseReason.ScopeRetired);
            }
            else
            {
                if (n.EndState is not null) DetachEndState(n, ReleaseReason.ScopeRetired);
                Relink(n);
            }
        }
        if (rec.NewestLeaf is not null) Relink(rec.NewestLeaf);
        Version++;
        TryRecycleScope(rec);
        return true;
    }

    private bool SubtreeLocked(RadixNode n)
    {
        if (n.AnyLock || n.IsDonationPending) return true;
        foreach (RadixNode c in n.Children)
            if (SubtreeLocked(c)) return true;
        return false;
    }

    private sealed class DepthDescending : IComparer<RadixNode>
    {
        internal static readonly DepthDescending Instance = new();
        public int Compare(RadixNode? x, RadixNode? y) => y!.Depth.CompareTo(x!.Depth);
    }

    /// <summary>
    /// A deferred invalidation (§5.17, DEC-23): the model already freed the payload, so the end state is
    /// detached without a release and the node is deleted if it becomes an unlocked payload-less leaf.
    /// </summary>
    internal bool InvalidatePayload(string key)
    {
        if (key is null || !_keyIndex.TryGetValue(key, out RadixNode? node)) return false;
        DetachEndState(node, ReleaseReason.Invalidated, enqueue: false);
        CollectFrom(node);
        Counters.Invalidations++;
        return true;
    }

    /// <summary>Applies CanMaterialize refusals seen during Evaluate (released as Invalidated).</summary>
    internal int FlushQueuedInvalidations()
    {
        int n = 0;
        foreach (string key in _refusedKeys)
        {
            if (!_keyIndex.TryGetValue(key, out RadixNode? node)) continue;
            if (node.StateLockRef > 0 || node.PinRef > 0 || node.IsDonationPending) continue;
            DetachEndState(node, ReleaseReason.Invalidated);
            CollectFrom(node);
            Counters.Invalidations++;
            n++;
        }
        _refusedKeys.Clear();
        return n;
    }

    /// <summary>BackendRecreated (§5.17, DEC-42): every end state and every A2 page is invalid; A1 pages survive.</summary>
    internal void InvalidateDeviceState()
    {
        int count = CollectAllNodes();
        for (int i = 0; i < count; i++)
        {
            RadixNode n = _scratch[i];
            if (!n.InTree) continue;
            if (n.EndState is not null) DetachEndState(n, ReleaseReason.Invalidated);
            for (int p = n.PageCount - 1; p >= 0; p--)
            {
                PageRef page = n.Pages![p];
                if (!page.HasA2) continue;
                if (page.Store == PageStore.Both)
                    n.Pages[p] = page with { Store = PageStore.A1HostSlab };
                else
                    RemovePage(n, p);
            }
            Relink(n);
        }
        for (int i = 0; i < count; i++)
        {
            RadixNode n = _scratch[i];
            _scratch[i] = null!;
            if (n.InTree) CollectFrom(n);
        }
        Version++;
    }

    /// <summary>
    /// Reset (§5.17): every receipt becomes stale (TreeSerial++), every end state key is queued with
    /// reason Reset, every page is freed and the tree is emptied. The caller drains the reclaim queue.
    /// </summary>
    internal void Reset()
    {
        int count = CollectAllNodes();
        for (int i = 0; i < count; i++)
        {
            RadixNode n = _scratch[i];
            if (n.EndState is not null)
            {
                Reclaim.Enqueue(n.EndState.Key, ReleaseReason.Reset, n.EndState.Bytes);
                n.EndState = null;
            }
            for (int p = 0; p < n.PageCount; p++) _pages.FreePage(n.Pages![p].Block);
        }
        Lists.Clear();
        for (int i = 0; i < count; i++)
        {
            RadixNode n = _scratch[i];
            _scratch[i] = null!;
            if (n.Edge.Rope is { } rope)
            {
                rope.SliceRefs--;
                rope.LiveSliceTokens -= n.Edge.Length;
                rope.FirstSliceNode = null;
                if (rope.SliceRefs == 0 && rope.OwnerReleased) rope.ReturnChunks(KeyPool);
            }
            _nodePool.Return(n);
        }
        Root.Children = default;
        _keyIndex.Clear();
        _blockOwners.Clear();
        _publicBoundaries.Clear();
        _refusedKeys.Clear();
        foreach (ScopeRecord rec in Scopes.LiveRecords())
        {
            rec.FirstNode = null; rec.NewestLeaf = null; rec.Bytes = default; rec.NodeCount = 0;
        }
        Counters.NodesDeleted += count;
        NodeCount = 0;
        _cached = default;
        _protected = default;
        _scopedEndStates = _publicEndStates = _nativeSlots = _primaryResidents = 0;
        Ledger?.Clear();
        TreeSerial++;
        Version++;
    }

    /// <summary>One batched release of every queued payload key (DEC-24).</summary>
    internal int DrainReclaimQueue(PayloadReleaseSink? sink) => Reclaim.Drain(sink);

    private int CollectAllNodes()
    {
        int count = 0;
        EnsureBfs(1);
        int head = 0, tail = 0;
        _bfs[tail++] = Root;
        while (head < tail)
        {
            RadixNode n = _bfs[head++];
            foreach (RadixNode c in n.Children)
            {
                EnsureBfs(tail + 1);
                _bfs[tail++] = c;
            }
            if (!n.IsRoot)
            {
                if (_scratch.Length == count) Array.Resize(ref _scratch, _scratch.Length * 2);
                _scratch[count++] = n;
            }
        }
        Array.Clear(_bfs, 0, tail);
        // deepest first so deletions cascade bottom-up
        Array.Sort(_scratch, 0, count, DepthDescending.Instance);
        return count;
    }

    // ------------------------------------------------------------------ enumeration (checker, harness, dump)

    internal IEnumerable<RadixNode> EnumerateNodes()
    {
        var stack = new Stack<RadixNode>();
        foreach (RadixNode c in Root.Children) stack.Push(c);
        while (stack.Count > 0)
        {
            RadixNode n = stack.Pop();
            yield return n;
            foreach (RadixNode c in n.Children) stack.Push(c);
        }
    }

    /// <summary>The full key of a node's root path (tests and dumps).</summary>
    internal static long[] PathKey(RadixNode n)
    {
        var result = new long[n.Depth];
        for (RadixNode? cur = n; cur is not null && !cur.IsRoot; cur = cur.Parent)
        {
            int start = cur.EdgeStartDepth;
            for (int i = 0; i < cur.Edge.Length; i++) result[start + i] = cur.Edge[i];
        }
        return result;
    }

    // ------------------------------------------------------------------ internals: deletion, GC, ropes

    /// <summary>
    /// DeleteLeafCascade (§5.11): deletes an unlocked leaf, then every childless unlocked payload-less
    /// parent (or any childless unlocked parent of a retired scope). Returns the bytes freed.
    /// </summary>
    internal ResourceVector DeleteLeafCascade(RadixNode n, ReleaseReason why)
    {
        if (n.IsRoot || !n.InTree) return default;
        if (n.Children.Count != 0 || n.AnyLock || n.IsDonationPending)
            throw new InvalidOperationException($"Only an unlocked leaf can be deleted: {n}.");
        RadixNode? p = n.Parent;
        ResourceVector freed = DeleteNode(n, why);
        return freed + CascadeParents(p, why);
    }

    private ResourceVector CascadeParents(RadixNode? p, ReleaseReason why)
    {
        ResourceVector freed = default;
        while (p is not null && !p.IsRoot && p.InTree && p.Children.Count == 0 && !p.AnyLock && !p.IsDonationPending
               && (!p.HasPayload || IsRetiredNode(p)))
        {
            RadixNode? next = p.Parent;
            freed += DeleteNode(p, IsRetiredNode(p) ? ReleaseReason.ScopeRetired : why);
            p = next;
        }
        return freed;
    }

    /// <summary>GC from a node: deletes it and cascades when it is an unlocked leaf with no payload (or retired).</summary>
    private bool CollectFrom(RadixNode n)
    {
        if (n.IsRoot || !n.InTree) return false;
        if (n.Children.Count != 0 || n.AnyLock || n.IsDonationPending) return false;
        if (n.HasPayload && !IsRetiredNode(n)) return false;
        RadixNode? parent = n.Parent;
        DeleteNode(n, IsRetiredNode(n) ? ReleaseReason.ScopeRetired : ReleaseReason.Evicted);
        CascadeParents(parent, ReleaseReason.Evicted);
        return true;
    }

    private bool IsRetiredNode(RadixNode n) => n.ScopeIx != 0 && Scopes[n.ScopeIx].Retired;

    private ResourceVector DeleteNode(RadixNode n, ReleaseReason why)
    {
        ResourceVector freed = n.Bytes;
        if (n.EndState is not null) DetachEndState(n, why);
        for (int p = n.PageCount - 1; p >= 0; p--) RemovePage(n, p);
        Lists.Unlink(n);
        RadixNode parent = n.Parent!;
        parent.Children.Remove(n);
        RemoveFromScope(n);
        DetachRopeSlice(n);
        n.InTree = false;
        NodeCount--;
        Counters.NodesDeleted++;
        _nodePool.Return(n);
        Relink(parent);
        Version++;
        return freed;
    }

    private void RemovePage(RadixNode n, int index)
    {
        PageRef page = n.Pages![index];
        ResourceVector before = ProtectedOf(n);
        ResourceVector bytes = PageBytes(page);
        n.RemovePageAt(index);
        n.Bytes -= bytes;
        _cached -= bytes;
        ScopeBytesAdjust(n.ScopeIx, ResourceVector.Negate(bytes));
        _protected += ProtectedOf(n) - before;
        _blockOwners.Remove(page.Block);
        _pages.FreePage(page.Block);
    }

    private void AttachRopeSlice(RadixNode n)
    {
        KeyRope rope = n.Edge.Rope;
        rope.SliceRefs++;
        rope.LiveSliceTokens += n.Edge.Length;
        n.RopePrev = null;
        n.RopeNext = rope.FirstSliceNode;
        if (rope.FirstSliceNode is not null) rope.FirstSliceNode.RopePrev = n;
        rope.FirstSliceNode = n;
    }

    private void DetachRopeSlice(RadixNode n)
    {
        KeyRope rope = n.Edge.Rope;
        if (rope is null) return;
        rope.SliceRefs--;
        rope.LiveSliceTokens -= n.Edge.Length;
        if (n.RopePrev is null) rope.FirstSliceNode = n.RopeNext; else n.RopePrev.RopeNext = n.RopeNext;
        if (n.RopeNext is not null) n.RopeNext.RopePrev = n.RopePrev;
        n.RopePrev = n.RopeNext = null;
        MaybeReclaimRope(rope);
    }

    private void MaybeReclaimRope(KeyRope rope)
    {
        if (!rope.OwnerReleased || rope.Disposed) return;
        if (rope.SliceRefs == 0)
        {
            rope.ReturnChunks(KeyPool);
            return;
        }
        if (rope.LiveSliceTokens < rope.Length / 2)
            CompactRope(rope);
    }

    /// <summary>Copies the surviving slices of a sparsely used rope into fresh chunks (DEC-29).</summary>
    private void CompactRope(KeyRope rope)
    {
        var fresh = new KeyRope { OwnerReleased = true };
        Span<long> buffer = stackalloc long[1024];
        RadixNode? n = rope.FirstSliceNode;
        while (n is not null)
        {
            RadixNode? next = n.RopeNext;
            KeySlice edge = n.Edge;
            int newStart = fresh.Length;
            int copied = 0;
            while (copied < edge.Length)
            {
                ReadOnlySpan<long> seg = rope.Segment(edge.Start + copied, edge.Length - copied);
                fresh.Append(seg, KeyPool);
                copied += seg.Length;
            }
            n.Edge = new KeySlice(fresh, newStart, edge.Length);
            fresh.SliceRefs++;
            fresh.LiveSliceTokens += edge.Length;
            n.RopePrev = null;
            n.RopeNext = fresh.FirstSliceNode;
            if (fresh.FirstSliceNode is not null) fresh.FirstSliceNode.RopePrev = n;
            fresh.FirstSliceNode = n;
            n = next;
        }
        rope.FirstSliceNode = null;
        rope.SliceRefs = 0;
        rope.LiveSliceTokens = 0;
        rope.ReturnChunks(KeyPool);
        Counters.RopeCompactions++;
    }

    // ------------------------------------------------------------------ internals: accounting, lists, tiers

    private ResourceVector PageBytes(in PageRef page)
    {
        var v = new ResourceVector { PoolPages = 1 };
        if (page.HasA1) v.HostKv = _options.PageHostBytes;
        return v;
    }

    internal ResourceVector PageBytesOf(RadixNode n)
    {
        ResourceVector v = n.Bytes;
        if (n.EndState is not null) v -= n.EndState.Bytes;
        return v;
    }

    /// <summary>Protected contribution of one node (I8).</summary>
    internal ResourceVector ProtectedOf(RadixNode n)
    {
        ResourceVector v = default;
        if (n.LockRef > 0) v += PageBytesOf(n);
        if (n.EndState is not null && (n.StateLockRef > 0 || n.PinRef > 0 || n.IsDonationPending)) v += n.EndState.Bytes;
        return v;
    }

    private void ScopeBytesAdjust(int scopeIx, ResourceVector delta)
    {
        if (scopeIx == 0 || delta.IsZero) return;
        ScopeRecord rec = Scopes[scopeIx];
        rec.Bytes += delta;
        if (rec.NewestLeaf is not null) Relink(rec.NewestLeaf);
    }

    private void CountEndState(RadixNode n, EndStatePayload es, int delta)
    {
        switch (es.Kind)
        {
            case EndStateKind.PrimaryResident: _primaryResidents += delta; return;
            case EndStateKind.NativeSlot: _nativeSlots += delta; break;
        }
        if (n.ScopeIx == 0) _publicEndStates += delta; else _scopedEndStates += delta;
    }

    private static bool PathHasMedia(RadixNode n)
    {
        for (RadixNode? cur = n; cur is not null && !cur.IsRoot; cur = cur.Parent)
            if (cur.MediaSpanCount > 0) return true;
        return false;
    }

    /// <summary>Whether <paramref name="depth"/> lies strictly inside a media span recorded on the node's root path.</summary>
    internal static bool IsInsideSpan(RadixNode n, int depth)
    {
        for (RadixNode? cur = n; cur is not null && !cur.IsRoot; cur = cur.Parent)
        {
            ReadOnlySpan<MediaSpanRecord> records = cur.SpanRecords;
            for (int i = 0; i < records.Length; i++)
                if (records[i].Start < depth && depth < records[i].End) return true;
        }
        return false;
    }

    private void AddToScope(RadixNode n)
    {
        ScopeRecord rec = Scopes[n.ScopeIx];
        n.ScopePrev = null;
        n.ScopeNext = rec.FirstNode;
        if (rec.FirstNode is not null) rec.FirstNode.ScopePrev = n;
        rec.FirstNode = n;
        rec.NodeCount++;
    }

    private void RemoveFromScope(RadixNode n)
    {
        ScopeRecord rec = Scopes[n.ScopeIx];
        if (n.ScopePrev is null) rec.FirstNode = n.ScopeNext; else n.ScopePrev.ScopeNext = n.ScopeNext;
        if (n.ScopeNext is not null) n.ScopeNext.ScopePrev = n.ScopePrev;
        n.ScopePrev = n.ScopeNext = null;
        rec.NodeCount--;
        if (ReferenceEquals(rec.NewestLeaf, n)) rec.NewestLeaf = null;
        if (n.ScopeIx != 0) TryRecycleScope(rec);
    }

    private void TryRecycleScope(ScopeRecord rec)
    {
        if (rec.Index != 0 && rec.Live && rec.Retired && rec.NodeCount == 0 && rec.RunningRequests == 0 && rec.WaitingRequests == 0)
            Scopes.Recycle(rec.Index);
    }

    private void SetNewestLeaf(ScopeRecord rec, RadixNode node)
    {
        RadixNode? old = rec.NewestLeaf;
        if (ReferenceEquals(old, node)) return;
        rec.NewestLeaf = node;
        if (old is not null && old.InTree) Relink(old);
        Relink(node);
    }

    private void TrackPublicBoundary(RadixNode n)
    {
        if (!_publicBoundaries.Contains(n)) _publicBoundaries.Add(n);
        RecomputePublicTop();
    }

    private void UntrackPublicBoundary(RadixNode n)
    {
        if (_publicBoundaries.Remove(n))
        {
            n.InPublicTop = false;
            RecomputePublicTop();
        }
    }

    /// <summary>Marks the PublicMax most recently used public boundaries with an end state (I16).</summary>
    private void RecomputePublicTop()
    {
        List<RadixNode> list = _publicBoundaries;
        for (int i = 1; i < list.Count; i++)
        {
            RadixNode item = list[i];
            int j = i - 1;
            while (j >= 0 && list[j].LastAccess < item.LastAccess)
            {
                list[j + 1] = list[j];
                j--;
            }
            list[j + 1] = item;
        }
        for (int i = 0; i < list.Count; i++)
        {
            bool top = i < _options.PublicMax;
            if (list[i].InPublicTop != top)
            {
                list[i].InPublicTop = top;
                Relink(list[i]);
            }
        }
    }

    /// <summary>TierOf (§5.11).</summary>
    internal EvictionTier TierOf(RadixNode n)
    {
        if (n.ScopeIx != 0 && Scopes[n.ScopeIx].Retired) return EvictionTier.Retired;
        if (n.IsPublicBoundary && n.EndState is not null && n.InPublicTop) return EvictionTier.PublicTop;
        if (n.ScopeIx != 0)
        {
            ScopeRecord rec = Scopes[n.ScopeIx];
            if ((ReferenceEquals(rec.NewestLeaf, n) && rec.Active && rec.Bytes.TotalBytes <= _scopeQuota)
                || (n.Flags & NodeFlags.PreemptHold) != 0)
                return EvictionTier.ScopeNewest;
        }
        if ((n.Flags & NodeFlags.EndsAtBreakpoint) != 0) return EvictionTier.Breakpoint;
        return EvictionTier.Ordinary;
    }

    /// <summary>The list a node belongs in (0 = none), from the §5.11 predicates (I9).</summary>
    internal byte DesiredList(RadixNode n)
    {
        if (n.IsRoot || !n.InTree || n.IsDonationPending) return 0;
        bool leaf = n.Children.Count == 0;
        if (leaf && n.LockRef == 0 && n.StateLockRef == 0 && n.PinRef == 0 && n.HasPayload)
            return EvictionLists.ListId(LruKind.Leaf, TierOf(n));
        if (n.EndState is not null && n.StateLockRef == 0 && n.PinRef == 0 && (!leaf || n.LockRef > 0))
            return EvictionLists.ListId(LruKind.State, TierOf(n));
        return 0;
    }

    private void Relink(RadixNode n)
    {
        if (n.IsRoot) return;
        byte want = DesiredList(n);
        if (n.LruList == want) return;
        Lists.Unlink(n);
        if (want != 0) Lists.Link(n, want);
    }

    private void Touch(RadixNode n)
    {
        if (n.IsRoot) return;
        n.LastAccess = ++_tick;
        if (n.LruList != 0)
        {
            byte id = n.LruList;
            Lists.Unlink(n);
            Lists.Link(n, id);
        }
        if (n.IsPublicBoundary && n.EndState is not null) RecomputePublicTop();
    }

    /// <summary>Stamps the root path, parents older than children.</summary>
    private void TouchPath(RadixNode anchor)
    {
        int count = 0;
        for (RadixNode n = anchor; !n.IsRoot; n = n.Parent!)
        {
            if (_scratch.Length == count) Array.Resize(ref _scratch, _scratch.Length * 2);
            _scratch[count++] = n;
        }
        for (int i = count - 1; i >= 0; i--)
        {
            Touch(_scratch[i]);
            _scratch[i] = null!;
        }
    }

    // ------------------------------------------------------------------ test hooks

    /// <summary>Test hook: sets the version without a mutation (invariant mutation tests).</summary>
    internal void UnsafeSetVersion(long version) => Version = version;

    /// <summary>Test hook: overrides the cached counter (invariant mutation tests).</summary>
    internal void UnsafeSetCached(ResourceVector cached) => _cached = cached;

    internal void UnsafeSetProtected(ResourceVector value) => _protected = value;
}
