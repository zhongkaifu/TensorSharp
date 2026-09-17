// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class PrefixTreeMatchTests
{
    private static PrefixCacheCapabilities NoTrunc(bool reuseAcrossMedia = true) =>
        Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None, reuseAcrossMedia: reuseAcrossMedia);

    [Fact]
    public void ExactPartialAndMidEdgeMatches()
    {
        PrefixTree t = Tk.Tree(NoTrunc());
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 40);
        Tk.Put(t, Tk.Key(t, conv), 30, s);

        // Exact: the next turn extends the conversation; the end state at 30 is reused.
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s));
        Assert.Equal(CandidateKind.EndState, plan.Kind);
        Assert.Equal(30, plan.Length);
        Assert.Equal(30, plan.Structural);

        // Partial: the prompt diverges at 20, mid-edge. No end state at or before 20 → no reuse,
        // but the structural match is 20 and nothing was split.
        int[] fork = Tk.Cat(Tk.Seq(1, 20), Tk.Seq(500, 20));
        int nodes = t.NodeCount;
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, fork), s));
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.Equal(20, plan.Structural);
        Assert.Equal(nodes, t.NodeCount);
        Assert.Single(plan.TrailEnds.AsSpan().ToArray());
        TrailEntry end = plan.Trail[plan.TrailEnds[0]];
        Assert.Equal(20, end.Matched);
        Assert.False(end.Full);

        // A shorter prompt that ends inside the edge: K6 leaves the last token un-matched.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 30)), s));
        Assert.Equal(29, plan.Structural);
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.Equal(SourceDecline.Absent, plan.EndStateDecline);
        Tk.Valid(t);
    }

    [Fact]
    public void K6_LeavesOneTokenUnmatched()
    {
        PrefixTree t = Tk.Tree(NoTrunc());
        int s = Tk.Scope(t);
        int[] prompt = Tk.Seq(1, 30);
        Tk.Put(t, Tk.Key(t, prompt), 30, s);
        // Exactly the cached prompt again: the end state at 30 would leave nothing to forward.
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, prompt), s));
        Assert.Equal(0, plan.Length);
        Assert.Equal(29, plan.Structural);
        // One more token: usable.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 31)), s));
        Assert.Equal(30, plan.Length);
        // A zero-length request never matches.
        MatchPlan none = new();
        KeyRope one = Tk.Key(t, new[] { 1 });
        t.Plan(Tk.Req(one, s), none);
        Assert.Equal(0, none.Structural);
    }

    [Fact]
    public void BreakpointLimit_CapsTheMatch()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None));
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 50);
        KeyRope key = Tk.Key(t, conv);
        Tk.Put(t, key, 20, s);
        Tk.Put(t, key, 40, s);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s, limit: 25));
        Assert.Equal(20, plan.Length);
        Assert.Equal(25, plan.Structural);
        Assert.True((plan.Clamps & ClampReasons.Breakpoint) != 0 || plan.Clamps == ClampReasons.None);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s, limit: 40));
        Assert.Equal(40, plan.Length);
    }

    [Fact]
    public void AnotherScopesNodesAreNeverMatched()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int a = Tk.Scope(t), b = Tk.Scope(t);
        int[] sys = Tk.Seq(1, 16);
        int[] conv = Tk.Cat(sys, Tk.Seq(100, 30));
        Tk.Put(t, Tk.Key(t, conv), 40, a, p: 16);
        // The public prefix [0,16) has no end state; scope b sees structure only up to 16.
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), b, p: 16));
        Assert.Equal(16, plan.Structural);
        Assert.Equal(CandidateKind.None, plan.Kind);
        // Truncation is scope-bound too: nothing of scope a is truncated for b.
        Assert.NotEqual(CandidateKind.TruncatedEndState, plan.Kind);
        // Scope a reuses everything.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), a, p: 16));
        Assert.Equal(40, plan.Length);
        // A request without a scope index (public only) sees only public nodes.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), 0, p: 16));
        Assert.Equal(16, plan.Structural);
    }

    [Fact]
    public void PublicAndOwnScopeChildren_LongerWins_TieGoesToOwnScope()
    {
        PrefixTree t = Tk.Tree(NoTrunc());
        int a = Tk.Scope(t), b = Tk.Scope(t);
        int[] shared = Tk.Seq(1, 24);
        // Scope b published a public system prompt of 24 tokens with an end state.
        RadixNode pub = Tk.Put(t, Tk.Key(t, shared), 24, 0, p: 24);
        Assert.True(pub.IsPublicBoundary);
        // Scope a (no system prompt: P=0) inserted the same first 12 tokens in its own branch, then diverged.
        int[] own = Tk.Cat(Tk.Seq(1, 12), Tk.Seq(300, 20));
        RadixNode ownNode = Tk.Put(t, Tk.Key(t, own), 32, a, p: 0);
        Assert.Equal(a, ownNode.ScopeIx);
        // a's prompt follows its own branch: the own child matches 32, the public one only 12.
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(own, new[] { 7 })), a));
        Assert.Equal(32, plan.Length);
        Assert.Same(ownNode, plan.PayloadNode);
        // b's prompt equal to the public prompt: the public end state (cap = boundary 24).
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(shared, new[] { 9 })), b));
        Assert.Equal(24, plan.Length);
        Assert.Equal(24, plan.PublicCap);
        Assert.Equal(24, plan.PublicTokens);
    }

    [Fact]
    public void APublicBranchNeverHidesALongerOwnScopeBranch()
    {
        // The design's greedy walk would pick the public child here (it matches its whole 16-token edge
        // while the own child matches its 8-token edge) and lose the own conversation beyond 16.
        PrefixTree t = Tk.Tree(NoTrunc());
        int a = Tk.Scope(t);
        int[] sys = Tk.Seq(1, 16);
        Tk.Put(t, Tk.Key(t, sys), 16, 0, p: 16);
        int[] conv = Tk.Cat(sys, Tk.Seq(200, 30));        // a's own conversation, P = 0, same leading tokens
        RadixNode deep = Tk.Put(t, Tk.Key(t, conv), 46, a, p: 0);
        // A sibling fork in scope a splits a's branch at 8.
        Tk.Put(t, Tk.Key(t, Tk.Cat(Tk.Seq(1, 8), Tk.Seq(900, 10))), 18, a, p: 0);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(conv, new[] { 5 })), a));
        Assert.Equal(46, plan.Length);
        Assert.Same(deep, plan.PayloadNode);
        Assert.Equal(2, plan.TrailEnds.Count);
        Assert.False(plan.SearchCapped);
        Tk.Valid(t);
    }

    [Fact]
    public void PublicCap_AllowsALongerSharedSystemPrompt_S4()
    {
        PrefixTree t = Tk.Tree(NoTrunc());
        int a = Tk.Scope(t), b = Tk.Scope(t);
        int[] longSys = Tk.Seq(1, 40);
        // Scope a published a 40-token public boundary (its system prompt).
        RadixNode boundary40 = Tk.Put(t, Tk.Key(t, longSys), 40, 0, p: 40);
        // Request b declares P = 24 (a shorter system prompt) but its prompt contains the whole 40-token prompt.
        int[] prompt = Tk.Cat(longSys, Tk.Seq(700, 10));
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, prompt), b, p: 24));
        Assert.Equal(40, plan.PublicCap);
        Assert.Equal(40, plan.Length);
        Assert.Same(boundary40, plan.PayloadNode);

        // A public node that continues past b's P without being a boundary is capped at P.
        PrefixTree t2 = Tk.Tree(NoTrunc());
        int a2 = Tk.Scope(t2), b2 = Tk.Scope(t2);
        RadixNode n24 = Tk.Put(t2, Tk.Key(t2, longSys), 24, 0, p: 24);
        // A deeper public end state without a boundary flag cannot be created by Insert; emulate a
        // public node at 40 whose inserter's boundary was elsewhere by attaching at a split point.
        RadixNode n40 = Tk.Put(t2, Tk.Key(t2, longSys), 40, 0, p: 40);
        Assert.True(n40.IsPublicBoundary);
        // b2's key diverges at 38: the 40 boundary is not fully matched, so the cap stays 24.
        int[] diverge = Tk.Cat(Tk.Seq(1, 38), Tk.Seq(900, 10));
        plan = Tk.Plan(t2, Tk.Req(Tk.Key(t2, diverge), b2, p: 24));
        Assert.Equal(24, plan.PublicCap);
        Assert.Equal(24, plan.Length);
        Assert.Same(n24, plan.PayloadNode);
    }

    [Fact]
    public void MediaVerify_SameKeyElementDifferentId_StopsAtTheSpanStart()
    {
        PrefixTree t = Tk.Tree(NoTrunc());
        int s = Tk.Scope(t);
        (string idA, string idB) = Tk.CollidingIds();
        Assert.Equal(KeyElem.Media(MediaId256.FromContentId(idA)), KeyElem.Media(MediaId256.FromContentId(idB)));
        var spanA = new[] { Tk.Span(10, 18, idA) };
        var spanB = new[] { Tk.Span(10, 18, idB) };
        int[] tokens = Tk.WithPlaceholders(30, spanA);
        Tk.Put(t, Tk.Key(t, tokens, spanA), 10, s, spans: spanA);
        Tk.Put(t, Tk.Key(t, tokens, spanA), 25, s, spans: spanA);
        // Same image: full reuse.
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, spanA), s, spans: spanA));
        Assert.Equal(25, plan.Length);
        // Colliding 63-bit key, different 256-bit id: the walk stops at the span start.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, spanB), s, spans: spanB));
        Assert.Equal(10, plan.Structural);
        Assert.Equal(10, plan.Length);
        // Inserting the colliding prompt never overwrites the sibling: it stops at 10.
        RadixNode stopped = t.Insert(Tk.Key(t, tokens, spanB), 25, s, 0, NodeFlags.None, spanB);
        Assert.Equal(10, stopped.Depth);
        Assert.Equal(1, t.Counters.MediaHashCollisions);
        Tk.Valid(t);
    }

    [Fact]
    public void MediaVerify_SameIdDifferentLength_AndAdjacentSpans()
    {
        PrefixTree t = Tk.Tree(NoTrunc());
        int s = Tk.Scope(t);
        string img = new string('a', 64), img2 = new string('b', 64);
        var cached = new[] { Tk.Span(8, 16, img), Tk.Span(16, 20, img2) };   // adjacent spans
        int[] tokens = Tk.WithPlaceholders(32, cached);
        Tk.Put(t, Tk.Key(t, tokens, cached), 28, s, spans: cached);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, cached), s, spans: cached));
        Assert.Equal(28, plan.Length);

        // The same image rendered one token longer: the keys agree up to 16, but the span record
        // differs (End 17 vs 16), so the match stops at the span start 8.
        var longer = new[] { Tk.Span(8, 17, img), Tk.Span(17, 21, img2) };
        int[] tokens2 = Tk.WithPlaceholders(32, longer);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens2, longer), s, spans: longer));
        Assert.Equal(8, plan.Structural);

        // Second adjacent span differs: the match stops exactly at 16.
        var other = new[] { Tk.Span(8, 16, img), Tk.Span(16, 20, new string('c', 64)) };
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, other), s, spans: other));
        Assert.Equal(16, plan.Structural);

        // A request that has a span where the cache has text: text vs media elements differ at the start.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 32)), s));
        Assert.Equal(8, plan.Structural);
    }

    [Fact]
    public void ReuseAcrossMediaSpanFalse_ClampsToTheFirstSpanStart()
    {
        PrefixTree t = Tk.Tree(NoTrunc(reuseAcrossMedia: false));
        int s = Tk.Scope(t);
        var spans = new[] { Tk.Span(12, 20, new string('d', 64)) };
        int[] tokens = Tk.WithPlaceholders(40, spans);
        KeyRope key = Tk.Key(t, tokens, spans);
        Tk.Put(t, key, 12, s, spans: spans);
        Tk.Put(t, key, 30, s, spans: spans);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, spans), s, spans: spans));
        Assert.Equal(12, plan.Length);
        Assert.True((plan.Clamps & ClampReasons.MediaAcrossSpan) != 0);
        // Spans stay atomic even when reuse across them is allowed: a prompt that ends mid-span.
        PrefixTree t2 = Tk.Tree(NoTrunc(reuseAcrossMedia: true));
        int s2 = Tk.Scope(t2);
        KeyRope key2 = Tk.Key(t2, tokens, spans);
        Tk.Put(t2, key2, 12, s2, spans: spans);
        t2.Insert(key2, 16, s2, 0, NodeFlags.None, spans);          // structure inside the span is allowed
        var refused = t2.AttachEndState(t2.Insert(key2, 16, s2, 0, NodeFlags.None, spans), Tk.Holder(t2));
        Assert.Equal(AttachResult.Refused, refused);               // an end state strictly inside a span is not
        MatchPlan mid = Tk.Plan(t2, Tk.Req(Tk.Key(t2, tokens, spans), s2, spans: spans, limit: 16));
        Assert.Equal(12, mid.Length);
        t2.CollectIfEmpty(t2.Insert(key2, 16, s2, 0, NodeFlags.None, spans));
        Tk.Valid(t2);
    }

    [Fact]
    public void Match_IsPure()
    {
        var host = new FakePageHost(Tk.B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 60);
        KeyRope key = Tk.Key(t, conv);
        RadixNode n = Tk.Put(t, key, 40, s);
        t.AttachPages(n, Tk.Pages(host, 5));
        Tk.Put(t, key, 24, s);
        long version = t.Version;
        int nodes = t.NodeCount;
        string lru = LruOrder(t);
        var r = Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 35), Tk.Seq(900, 5))), s);
        MatchPlan p1 = Tk.Plan(t, r), p2 = Tk.Plan(t, r);
        Assert.Equal(p1.ToString(), p2.ToString());
        Assert.Same(p1.PayloadNode, p2.PayloadNode);
        Assert.Same(p1.AnchorParent, p2.AnchorParent);
        Assert.Equal(p1.AnchorOffset, p2.AnchorOffset);
        Assert.Equal(p1.Trail.AsSpan().ToArray(), p2.Trail.AsSpan().ToArray());
        Assert.Equal(version, t.Version);
        Assert.Equal(nodes, t.NodeCount);
        Assert.Equal(lru, LruOrder(t));
        Assert.Equal(version, p1.Version);
    }

    [Fact]
    public void Match_AllocatesNothingOnceWarm()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 5000);
        KeyRope key = Tk.Key(t, conv);
        for (int len = 100; len <= 4000; len += 100) Tk.Put(t, key, len, s);
        var r = Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 3990), Tk.Seq(9000, 20))), s);
        var plan = new MatchPlan();
        // Warm past tiered compilation (call-counting installs allocate once on the calling thread).
        for (int i = 0; i < 2000; i++) t.Plan(r, plan);
        long before = System.GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) t.Match(r, plan);
        long matchAlloc = System.GC.GetAllocatedBytesForCurrentThread() - before;
        before = System.GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) t.Evaluate(r, plan);
        long evalAlloc = System.GC.GetAllocatedBytesForCurrentThread() - before;
        before = System.GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) t.Plan(r, plan);
        long allocated = System.GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated == 0, $"match {matchAlloc} eval {evalAlloc} plan {allocated}");
        Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
        Assert.Equal(3990, plan.Length);
    }

    [Fact]
    public void SearchCap_FallsBackToTheGreedyRule()
    {
        PrefixTree t = Tk.Tree(NoTrunc());
        int a = Tk.Scope(t);
        // Build 20 levels where both a public and an own child match fully: a public boundary at every
        // even length L, and an own-scope leaf [L, L+1) hanging under it.
        int[] toks = Tk.Seq(1, 80);
        for (int len = 2; len <= 40; len += 2)
        {
            Tk.Put(t, Tk.Key(t, toks), len, 0, p: len);
            Tk.Put(t, Tk.Key(t, toks), len + 1, a, p: len);
        }
        var plan = Tk.Plan(t, Tk.Req(Tk.Key(t, toks), a));
        Assert.True(plan.SearchCapped);
        Assert.True(plan.TrailEnds.Count <= MatchPlan.MaxTrailEnds);
        Assert.True(plan.Length > 0);
        Tk.Valid(t);
    }

    [Fact]
    public void BlockedByScope_IsDiagnosticOnly()
    {
        PrefixTree t = Tk.Tree(NoTrunc(), blockedByScope: true);
        int a = Tk.Scope(t), b = Tk.Scope(t);
        int[] sys = Tk.Seq(1, 8);
        int[] conv = Tk.Cat(sys, Tk.Seq(100, 30));
        Tk.Put(t, Tk.Key(t, conv), 38, a, p: 8);
        Tk.Put(t, Tk.Key(t, sys), 8, 0, p: 8);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), b, p: 8));
        Assert.Equal(8, plan.Length);
        Assert.Equal(30, plan.BlockedByScope);
    }

    internal static string LruOrder(PrefixTree t)
    {
        var parts = new List<string>();
        for (byte id = 1; id <= 10; id++)
            for (RadixNode? n = t.Lists.First(id); n is not null; n = n.LruNext)
                parts.Add($"{id}:{n.Id}");
        return string.Join(",", parts);
    }
}
