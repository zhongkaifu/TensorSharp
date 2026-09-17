// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>Guard rails, defensive paths and rarely-taken branches of the tree core.</summary>
public class PrefixTreeEdgeCaseTests
{
    private const int B = Tk.B;

    [Fact]
    public void StaleMatchException_CarriesBothVersions()
    {
        var ex = new StalePrefixMatchException(3, 9);
        Assert.Equal(3, ex.PlanVersion);
        Assert.Equal(9, ex.TreeVersion);
        Assert.Contains("stale", ex.Message);
    }

    [Fact]
    public void DeletionAndCollection_Guards()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 40));
        RadixNode mid = Tk.Put(t, key, 10, s);
        RadixNode leaf = Tk.Put(t, key, 20, s);
        Assert.True(t.DeleteLeafCascade(t.Root, ReleaseReason.Evicted).IsZero);
        Assert.False(t.CollectIfEmpty(t.Root));
        Assert.Throws<InvalidOperationException>(() => t.DeleteLeafCascade(mid, ReleaseReason.Evicted));      // has a child
        LockReceipt r = t.AcquireState(leaf);
        Assert.Throws<InvalidOperationException>(() => t.DeleteLeafCascade(leaf, ReleaseReason.Evicted));     // locked
        t.Release(ref r);
        t.MarkDonationPending(leaf);
        Assert.Throws<InvalidOperationException>(() => t.DeleteLeafCascade(leaf, ReleaseReason.Evicted));     // donation pending
        t.CancelDonation(leaf);
        t.DeleteLeafCascade(leaf, ReleaseReason.Evicted);
        Assert.True(t.DeleteLeafCascade(leaf, ReleaseReason.Evicted).IsZero);                                  // already gone
        Assert.False(t.CollectIfEmpty(leaf));
        Assert.Throws<InvalidOperationException>(() => t.AcquirePath(leaf));                                   // not in the tree
        t.DetachEndState(mid, ReleaseReason.Evicted);
        Assert.True(t.DetachEndState(mid, ReleaseReason.Evicted).IsZero);                                      // tombstone: nothing to detach
        Assert.True(t.CollectIfEmpty(mid));                                                                    // detach leaves collection to the caller
        Tk.Valid(t);
    }

    [Fact]
    public void LockUnderflow_IsReportedNotHidden()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None), strictReceipts: false);
        int s = Tk.Scope(t);
        RadixNode n = Tk.Put(t, Tk.Key(t, Tk.Seq(1, 40)), 20, s);
        LockReceipt path = t.AcquirePath(n);
        n.LockRef = 0;
        Assert.Throws<InvalidOperationException>(() => t.Release(ref path));
        PrefixTree t2 = Tk.Tree(Tk.Caps(pages: PageSupport.None), strictReceipts: false);
        int s2 = Tk.Scope(t2);
        RadixNode n2 = Tk.Put(t2, Tk.Key(t2, Tk.Seq(1, 40)), 20, s2);
        LockReceipt state = t2.AcquireState(n2);
        n2.StateLockRef = 0;
        Assert.Throws<InvalidOperationException>(() => t2.Release(ref state));
    }

    [Fact]
    public void WithoutAReceiptLedger_LocksStillBalance()
    {
        var t = new PrefixTree(new PrefixTreeOptions { Capabilities = Tk.Caps(pages: PageSupport.None), BlockSize = B, TrackReceipts = false, StrictReceipts = false });
        Assert.Null(t.Ledger);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 40));
        Tk.Put(t, key, 30, s);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 20), Tk.Seq(90, 4))), s));
        Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
        LockReceipt r = t.Acquire(plan);
        t.ReleaseState(ref r);
        Assert.NotNull(r.PathAnchor);
        t.Release(ref r);
        LockReceipt state = t.Acquire(Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 40)), s)));
        Assert.NotNull(state.StateAnchor);
        LockReceipt stateOnly = t.AcquireState(state.StateAnchor!);
        t.ReleaseState(ref stateOnly);
        Assert.True(stateOnly.IsEmpty);
        t.Reset();
        t.Release(ref state);                       // stale epoch: ignored
        Assert.Equal(1, t.Counters.StaleReceipts);
        Assert.Equal(0, t.NodeCount);
    }

    [Fact]
    public void MediaVerification_ToleratesACorruptNodeAndNullSpans()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None, mmReuseMin: 8));
        int s = Tk.Scope(t);
        var spans = new[] { Tk.Span(6, 10, new string('4', 64)) };
        int[] tokens = Tk.WithPlaceholders(30, spans);
        KeyRope key = Tk.Key(t, tokens, spans);
        RadixNode n = Tk.Put(t, key, 20, s, spans: spans);
        // A node that lost its span record (an I13 violation) is never trusted past the span start.
        n.MediaSpanCount = 0;
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, spans), s, spans: spans));
        Assert.Equal(6, plan.Structural);
        n.MediaSpanCount = 1;
        // A request with no span array (null) against a media node stops at the record.
        MatchPlan nullSpans = new();
        t.Plan(new MatchRequest(Tk.Key(t, tokens, spans), tokens.Length, s, 0, tokens.Length - 1, null!, ExpectedRoute.Primary, true), nullSpans);
        Assert.Equal(6, nullSpans.Structural);
        // MmReuseMinTokens with a text-only request (null spans) does not apply.
        int[] text = Tk.Seq(500, 20);
        Tk.Put(t, Tk.Key(t, text), 4, s);
        MatchPlan textPlan = new();
        t.Plan(new MatchRequest(Tk.Key(t, text), text.Length, s, 0, text.Length - 1, null!, ExpectedRoute.Primary, true), textPlan);
        Assert.Equal(4, textPlan.Length);
    }

    [Fact]
    public void EqualCandidates_TieBreakByRankThenAge()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None));
        int a = Tk.Scope(t);
        int[] sys = Tk.Seq(1, 16);
        KeyRope key = Tk.Key(t, Tk.Cat(sys, Tk.Seq(40, 20)));
        // Public end state at 16 (a clone for everyone) and an own-scope end state at 16 with a child (a clone too).
        RadixNode pub = Tk.Put(t, key, 16, 0, p: 16);
        RadixNode own = Tk.Put(t, key, 16, a, p: 0);
        Tk.Put(t, key, 30, a, p: 0);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(sys, Tk.Seq(900, 5))), a, p: 16));
        Assert.Equal(16, plan.Length);
        Assert.Equal(MaterializeMode.CloneEndState, plan.Mode);
        Assert.Same(pub, plan.PayloadNode);                 // same rank: the older node
        // Own-scope leaf at 16 (donation, rank 1) beats the public clone (rank 3).
        PrefixTree t2 = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None));
        int a2 = Tk.Scope(t2);
        KeyRope key2 = Tk.Key(t2, Tk.Cat(sys, Tk.Seq(40, 20)));
        Tk.Put(t2, key2, 16, 0, p: 16);
        RadixNode donor = Tk.Put(t2, key2, 16, a2, p: 0);
        plan = Tk.Plan(t2, Tk.Req(Tk.Key(t2, Tk.Cat(sys, Tk.Seq(900, 5))), a2, p: 16));
        Assert.Same(donor, plan.PayloadNode);
        Assert.Equal(MaterializeMode.DonateEndState, plan.Mode);
        Assert.NotSame(own, donor);
    }

    [Fact]
    public void TruncationSearch_StopsAtItsNodeBudget()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.Any, pages: PageSupport.None, rewindCap: 1000), truncationSearchNodes: 3);
        int s = Tk.Scope(t);
        int[] trunk = Tk.Seq(1, 20);
        for (int i = 0; i < 6; i++)
            Tk.Put(t, Tk.Key(t, Tk.Cat(trunk, new[] { 100 + i }, Tk.Seq(200, 10))), 31, s);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(trunk, new[] { 999 }, Tk.Seq(1, 4))), s));
        Assert.True(plan.TruncationSearchCapped);
        Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
        Assert.Equal(20, plan.Length);
    }

    [Fact]
    public void SearchCap_GreedyPrefersTheLongerOwnChild()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None));
        int a = Tk.Scope(t);
        int[] toks = Tk.Seq(1, 100);
        // 20 levels: a public boundary [L, L+2) and a longer own leaf [L, L+3) under the node at L.
        for (int len = 2; len <= 40; len += 2)
        {
            Tk.Put(t, Tk.Key(t, toks), len, 0, p: len);
            Tk.Put(t, Tk.Key(t, toks), len + 3, a, p: len);
        }
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, toks), a));
        Assert.True(plan.SearchCapped);
        Assert.True(plan.Length > 0);
        Tk.Valid(t);
    }

    [Fact]
    public void CostRule_FallsBackFromATruncatedCloneOrAConvertedPrimary()
    {
        // Best = a truncated clone (below MinCloneTokens); A = a shorter donation → A.
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.Any, pages: PageSupport.None), minClone: 100);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        Tk.Put(t, key, 10, s);
        RadixNode deep = Tk.Put(t, key, 40, s);
        Tk.Put(t, key, 60, s);                        // `deep` has a child: truncating it is a clone
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 30), Tk.Seq(900, 3))), s));
        Assert.True((plan.Clamps & ClampReasons.CloneCost) != 0);
        Assert.Equal(SourceDecline.CloneCost, plan.TruncationDecline);
        Assert.Equal(CandidateKind.None, plan.Kind);  // 10 is also a clone (it has a child) → nothing
        Assert.NotNull(deep);

        // Best = a primary resident that must be converted (a clone) below MinCloneTokens.
        PrefixTree t2 = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None), minClone: 100);
        int s2 = Tk.Scope(t2);
        KeyRope key2 = Tk.Key(t2, Tk.Seq(1, 80));
        Tk.Put(t2, key2, 30, s2, payload: Tk.Primary(t2));
        Tk.Put(t2, key2, 50, s2);
        plan = Tk.Plan(t2, Tk.Req(Tk.Key(t2, Tk.Cat(Tk.Seq(1, 31), Tk.Seq(900, 3))), s2));
        Assert.Equal(SourceDecline.CloneCost, plan.PrimaryDecline);
        Assert.Equal(CandidateKind.None, plan.Kind);

        // MmReuseMinTokens drops a pages plan.
        var host = new FakePageHost(B);
        PrefixTree t3 = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, mmReuseMin: 64), host: host);
        int s3 = Tk.Scope(t3);
        var spans = new[] { Tk.Span(40, 44, new string('5', 64)) };
        int[] tokens = Tk.WithPlaceholders(60, spans);
        RadixNode pages = t3.Insert(Tk.Key(t3, tokens, spans), 3 * B, s3, 0, NodeFlags.None, spans);
        t3.AttachPages(pages, Tk.Pages(host, 3));
        plan = Tk.Plan(t3, Tk.Req(Tk.Key(t3, tokens, spans), s3, spans: spans));
        Assert.Equal(SourceDecline.MmThreshold, plan.PageDecline);
        Assert.Equal(CandidateKind.None, plan.Kind);
    }

    [Fact]
    public void BranchPosition_IsZeroWhenClampedBelowTheReuse()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None, reuseAcrossMedia: false), branchSnapshots: true);
        int s = Tk.Scope(t);
        var spans = new[] { Tk.Span(10, 20, new string('6', 64)) };
        int[] tokens = Tk.WithPlaceholders(700, spans);
        KeyRope key = Tk.Key(t, tokens, spans);
        Tk.Put(t, key, 8, s, spans: spans);
        RadixNode tomb = t.Insert(key, 650, s, 0, NodeFlags.None, spans);
        LockReceipt hold = t.AcquirePath(tomb);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, spans), s, spans: spans));
        Assert.Equal(8, plan.Length);
        // structural 650 − 8 ≥ 256, but the branch point clamps to the first span start (10 → page 8) ≤ L.
        Assert.Equal(0, plan.BranchPosition);
        t.Release(ref hold);
    }

    [Fact]
    public void BlockedByScope_AtAPartiallyMatchedNode()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None), blockedByScope: true);
        int a = Tk.Scope(t), b = Tk.Scope(t);
        int[] conv = Tk.Cat(Tk.Seq(1, 8), Tk.Seq(100, 30));
        Tk.Put(t, Tk.Key(t, conv), 38, a, p: 8);
        Tk.Put(t, Tk.Key(t, Tk.Seq(1, 16)), 16, 0, p: 16);         // a public node [8, 16) under [0, 8)
        // b diverges inside the public [8, 16) edge: the stop is partial, siblings are checked at its parent.
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 12), Tk.Seq(900, 4))), b, p: 8));
        Assert.Equal(12, plan.Structural);
        Assert.True(plan.BlockedByScope >= 0);
        // A request whose key ends at the stop has nothing to compare.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 17)), b, p: 8, limit: 16, keyLength: 16));
        Assert.True(plan.BlockedByScope >= 0);
    }

    [Fact]
    public void CountSubCap_FailsWhenEveryVictimIsLocked()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None), publicMax: 1);
        RadixNode p1 = Tk.Put(t, Tk.Key(t, Tk.Seq(1, 20)), 8, 0, p: 8);
        RadixNode p2 = Tk.Put(t, Tk.Key(t, Tk.Seq(100, 20)), 8, 0, p: 8);
        LockReceipt r1 = t.AcquireState(p1), r2 = t.AcquireState(p2);
        Assert.False(t.EnforceCountSubCaps());
        t.Release(ref r1);
        t.Release(ref r2);
        Assert.True(t.EnforceCountSubCaps());
        Tk.Valid(t, new InvariantCheckContext(AfterEvictionTrigger: true));
    }

    [Fact]
    public void DeepPaths_GrowTheScratchBuffers()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None));
        int s = Tk.Scope(t);
        int[] toks = Tk.Seq(1, 400);
        KeyRope key = Tk.Key(t, toks);
        for (int len = 1; len <= 300; len++) Tk.Put(t, key, len, s, host: 1);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, toks), s));
        Assert.Equal(300, plan.Length);
        LockReceipt r = t.Acquire(plan);             // touches a 300-node path
        t.Release(ref r);
        Assert.True(t.RetireScope(t.Scopes[s].Id));   // retires a scope with 300 nodes
        Assert.Equal(0, t.NodeCount);
        Tk.Valid(t);
    }

    [Fact]
    public void Persistence_OnlyOnMediaFreePublicBoundaries()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        RadixNode ok = t.Insert(Tk.Key(t, Tk.Seq(1, 20)), 12, 0, 12, NodeFlags.None, null);
        Assert.Equal(AttachResult.Attached, t.AttachEndState(ok, Tk.Holder(t, persisted: true)));
        var spans = new[] { Tk.Span(2, 6, new string('8', 64)) };
        RadixNode media = t.Insert(Tk.Key(t, Tk.WithPlaceholders(20, spans), spans), 12, 0, 12, NodeFlags.None, spans);
        Assert.True(media.IsPublicBoundary);
        Assert.Throws<ArgumentException>(() => t.AttachEndState(media, Tk.Holder(t, persisted: true)));
        t.AttachEndState(media, Tk.Holder(t));
        Tk.Valid(t);
    }

    [Fact]
    public void RetireScope_DetachesAnInternalEndStateAboveAStateLockedDescendant()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.Any, pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode upper = Tk.Put(t, key, 20, s);
        RadixNode lower = Tk.Put(t, key, 40, s);
        LockReceipt state = t.AcquireState(lower);   // state lock only: `upper` has LockRef 0
        Assert.Equal(0, upper.LockRef);
        t.RetireScope(t.Scopes[s].Id);
        Assert.True(upper.InTree);
        Assert.Null(upper.EndState);
        Assert.True((upper.Flags & NodeFlags.Retired) != 0);
        Tk.Valid(t);
        t.Release(ref state);
        Assert.False(lower.InTree);
        Assert.False(upper.InTree);
        Tk.Valid(t);
    }

    [Fact]
    public void EvictionLists_Guards()
    {
        var lists = new EvictionLists();
        var a = new RadixNode { LastAccess = 5 };
        Assert.Throws<ArgumentOutOfRangeException>(() => lists.Link(a, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => lists.Link(a, 11));
        byte id = EvictionLists.ListId(LruKind.State, EvictionTier.Breakpoint);
        Assert.Equal(LruKind.State, EvictionLists.KindOf(id));
        Assert.Equal(LruKind.Leaf, EvictionLists.KindOf(EvictionLists.ListId(LruKind.Leaf, EvictionTier.PublicTop)));
        lists.Link(a, id);
        Assert.Throws<InvalidOperationException>(() => lists.Link(a, id));
        var older = new RadixNode { LastAccess = 1 };
        var middle = new RadixNode { LastAccess = 3 };
        lists.Link(older, id);
        lists.Link(middle, id);
        Assert.Same(older, lists.First(LruKind.State, EvictionTier.Breakpoint));
        Assert.Same(middle, older.LruNext);
        Assert.Same(a, lists.Last(id));
        Assert.True(lists.InsertScanSteps >= 1);
        lists.Unlink(new RadixNode());               // not linked: no-op
        lists.Clear();
        Assert.Equal(0, lists.Count(id));
        Assert.Equal(0, a.LruList);
    }

    [Fact]
    public void InflightPromotion_SkipsAFollowerThatMovedToAnotherEntry()
    {
        var table = new InflightTable();
        var pool = new KeyChunkPool();
        KeyRope keyA = KeyRope.FromKeys(Enumerable.Range(0, 600).Select(i => (long)i).ToArray(), pool);
        KeyRope keyB = KeyRope.FromKeys(Enumerable.Range(1000, 600).Select(i => (long)i).ToArray(), pool);
        var leaderA = new Req { IsRunning = true, Key = keyA };
        var leaderB = new Req { IsRunning = true, Key = keyB };
        table.RegisterLeader(leaderA, 512, 1);
        table.RegisterLeader(leaderB, 512, 2);
        var mover = new Req { Key = KeyRope.FromKeys(Enumerable.Range(0, 600).Select(i => (long)i).ToArray(), pool) };
        var stayer = new Req { Key = KeyRope.FromKeys(Enumerable.Range(0, 600).Select(i => (long)i).ToArray(), pool) };
        Assert.True(table.TryFollow(mover, 512, 1, 1));
        Assert.True(table.TryFollow(stayer, 512, 1, 1));
        mover.FollowingLeader = null;                // the scheduler released it (admitted elsewhere)
        Assert.Same(stayer, table.OnLeaderGone(leaderA, LeaderGoneReason.Aborted));
        Assert.Null(table.OnLeaderGone(leaderB, LeaderGoneReason.Aborted));   // no followers
    }

    private sealed class Req : IInflightRequest
    {
        public bool IsRunning { get; set; }
        public int NumComputedTokens { get; set; }
        public KeyRope Key { get; init; } = null!;
        public InflightEntry? FollowingLeader { get; set; }
    }
}
