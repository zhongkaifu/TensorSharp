// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class PrefixTreeEvaluateTests
{
    private const int B = Tk.B;

    private static RadixNode PutPages(PrefixTree t, FakePageHost host, KeyRope key, int pages, int scopeIx, int p = 0,
                                      PageStore store = PageStore.Both, bool stateAtEnd = true)
    {
        RadixNode n = t.Insert(key, pages * B, scopeIx, p, NodeFlags.None, null);
        Assert.Equal(pages, t.AttachPages(n, Tk.Pages(host, pages, store, stateAtEnd: stateAtEnd)));
        return n;
    }

    [Fact]
    public void CandidatesAB_AndC_TheLongestWins()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 64);
        KeyRope key = Tk.Key(t, conv);
        PutPages(t, host, key, 4, s);                  // B: pages to 32
        Tk.Put(t, key, 20, s);                         // A: end state at 20
        RadixNode deep = Tk.Put(t, key, 50, s);        // C source: end state at 50

        // Prompt diverges at 45: A=20, B=32, C=45 (truncate 50 → 45, rewind 5 ≤ 16).
        int[] fork = Tk.Cat(Tk.Seq(1, 45), Tk.Seq(900, 10));
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, fork), s));
        Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
        Assert.Equal(45, plan.Length);
        Assert.Same(deep, plan.PayloadNode);
        Assert.Equal(MaterializeMode.DonateEndState, plan.Mode);   // leaf, own scope, slack 5 ≤ 16

        // Diverge at 30: C would need a 20-token rewind (> 16): pages (32 → aligned ≤ 30 = 24) vs A=20 → pages 24.
        fork = Tk.Cat(Tk.Seq(1, 30), Tk.Seq(900, 10));
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, fork), s));
        Assert.Equal(CandidateKind.Pages, plan.Kind);
        Assert.Equal(24, plan.Length);
        Assert.Equal(3, plan.PageCount);
        Assert.Equal(MaterializeMode.InjectA1Pages, plan.Mode);
        Assert.True((plan.Clamps & ClampReasons.RewindCap) != 0);

        // Diverge at 21: A=20 beats pages (16).
        fork = Tk.Cat(Tk.Seq(1, 21), Tk.Seq(900, 10));
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, fork), s));
        Assert.Equal(CandidateKind.EndState, plan.Kind);
        Assert.Equal(20, plan.Length);
        Assert.Equal(SourceDecline.Shorter, plan.PageDecline);
        Tk.Valid(t);
    }

    [Theory]
    [InlineData((int)PageStore.A1HostSlab, (int)ExpectedRoute.Primary, (int)MaterializeMode.InjectA1Pages)]
    [InlineData((int)PageStore.A1HostSlab, (int)ExpectedRoute.BatchedPaged, (int)MaterializeMode.InjectA1Pages)]
    [InlineData((int)PageStore.A2ModelPaged, (int)ExpectedRoute.BatchedPaged, (int)MaterializeMode.BindPagesInPlace)]
    [InlineData((int)PageStore.A2ModelPaged, (int)ExpectedRoute.PerSequenceFused, (int)MaterializeMode.CopyA2PagesToHolder)]
    [InlineData((int)PageStore.A2ModelPaged, (int)ExpectedRoute.Primary, (int)MaterializeMode.CopyA2PagesToHolder)]
    [InlineData((int)PageStore.Both, (int)ExpectedRoute.BatchedPaged, (int)MaterializeMode.BindPagesInPlace)]
    [InlineData((int)PageStore.Both, (int)ExpectedRoute.PerSequenceFused, (int)MaterializeMode.InjectA1Pages)]
    [InlineData((int)PageStore.Both, (int)ExpectedRoute.Primary, (int)MaterializeMode.InjectA1Pages)]
    public void RouteReadability_PicksTheRoutesNaturalMode(int storeValue, int routeValue, int expectedValue)
    {
        var store = (PageStore)storeValue;
        var route = (ExpectedRoute)routeValue;
        var expected = (MaterializeMode)expectedValue;
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.Both, copyPagedToHolder: true), host: host);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 40);
        PutPages(t, host, Tk.Key(t, conv), 3, s, store: store);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s, route: route));
        Assert.Equal(CandidateKind.Pages, plan.Kind);
        Assert.Equal(expected, plan.Mode);
        Assert.Equal(24, plan.Length);
    }

    [Fact]
    public void RouteReadability_UnreadableStoresProduceNoPages()
    {
        var host = new FakePageHost(B);
        // A2-only pages, no copy-to-holder support, batched paged disabled: nobody can read them.
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.A2ModelPaged), host: host, batchedPaged: false);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 40);
        PutPages(t, host, Tk.Key(t, conv), 3, s, store: PageStore.A2ModelPaged);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s, route: ExpectedRoute.BatchedPaged));
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.Equal(SourceDecline.RouteUnreadable, plan.PageDecline);
        // The family only declares A1: A2 pages on the batched route are not readable either.
        var host2 = new FakePageHost(B);
        PrefixTree t2 = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.A1HostSlab), host: host2);
        int s2 = Tk.Scope(t2);
        PutPages(t2, host2, Tk.Key(t2, conv), 3, s2, store: PageStore.A2ModelPaged);
        plan = Tk.Plan(t2, Tk.Req(Tk.Key(t2, conv), s2, route: ExpectedRoute.BatchedPaged));
        Assert.Equal(CandidateKind.None, plan.Kind);
        // No page support and no copy: pages are not evaluated at all.
        PrefixTree t3 = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None));
        plan = Tk.Plan(t3, Tk.Req(Tk.Key(t3, conv), Tk.Scope(t3)));
        Assert.Equal(SourceDecline.Unsupported, plan.PageDecline);
    }

    [Fact]
    public void PageWindowTokens_StopsTheChain()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pageWindow: 3 * B), host: host);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 60);
        PutPages(t, host, Tk.Key(t, conv), 6, s);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s));
        Assert.Equal(3 * B, plan.Length);
        Assert.True((plan.Clamps & ClampReasons.Window) != 0);
    }

    [Fact]
    public void PagesNeedStateAtEnd_BacktracksToTheLastRestorablePage()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, needStateAtEnd: true), host: host);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 60);
        KeyRope key = Tk.Key(t, conv);
        RadixNode n = t.Insert(key, 5 * B, s, 0, NodeFlags.None, null);
        var pages = new PageRef[5];
        for (int i = 0; i < 5; i++)
            pages[i] = new PageRef(host.NewBlock(PageStore.A1HostSlab), i, PageStore.A1HostSlab, StateAtEnd: i == 1 || i == 4);
        t.AttachPages(n, pages);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s));
        Assert.Equal(5 * B, plan.Length);
        // Diverge inside page 4: the last page with state at its end before it is page 1.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 4 * B + 3), Tk.Seq(900, 10))), s));
        Assert.Equal(2 * B, plan.Length);
    }

    [Fact]
    public void PageChain_StopsAtAMissingPage_AndAtMediaInsideAPage()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None), host: host);
        int s = Tk.Scope(t);
        var spans = new[] { Tk.Span(3 * B - 2, 3 * B + 2, new string('e', 64)) };   // crosses the end of page 2
        int[] conv = Tk.WithPlaceholders(60, spans);
        KeyRope key = Tk.Key(t, conv, spans);
        RadixNode n = t.Insert(key, 5 * B, s, 0, NodeFlags.None, spans);
        var pages = Tk.Pages(host, 5);
        t.AttachPages(n, new[] { pages[0], pages[1], pages[2], pages[4] });   // page 3 missing
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv, spans), s, spans: spans));
        // Page 2 ends inside the span (24 ∈ (22, 26)): not usable; page 3 is absent.
        Assert.Equal(2 * B, plan.Length);
        Assert.True((plan.Clamps & ClampReasons.Media) != 0);
    }

    [Theory]
    [InlineData(TruncationKind.None, 0, 60, 58, false)]
    [InlineData(TruncationKind.Any, 0, 60, 50, true)]
    [InlineData(TruncationKind.Any, 0, 60, 40, false)]                  // rewind 20 > cap 16
    [InlineData(TruncationKind.WithinUnwrappedWindow, 64, 60, 50, true)] // cached ≤ W
    [InlineData(TruncationKind.WithinUnwrappedWindow, 32, 60, 50, false)]// wrapped ring
    [InlineData(TruncationKind.WithinRingSlack, 4, 60, 56, true)]
    [InlineData(TruncationKind.WithinRingSlack, 4, 60, 55, false)]
    public void TruncationKinds_AndRewindCap(TruncationKind kind, int parameter, int cached, int diverge, bool expectTruncation)
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: kind, truncationParameter: parameter, pages: PageSupport.None));
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 80);
        RadixNode src = Tk.Put(t, Tk.Key(t, conv), cached, s);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, diverge), Tk.Seq(900, 5))), s));
        if (expectTruncation)
        {
            Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
            Assert.Equal(diverge, plan.Length);
            Assert.Same(src, plan.PayloadNode);
        }
        else
        {
            Assert.Equal(CandidateKind.None, plan.Kind);
        }
    }

    [Fact]
    public void ModelDecides_HasNoRewindCap_GranularityAligns_ValidatorDecides()
    {
        var validator = new FakeValidator { Rule = (key, payload, target) => payload - target <= 30 };
        PrefixTree t = Tk.Tree(Tk.Caps(endState: EndStateSupport.CopyAndDonate, truncation: TruncationKind.ModelDecides, granularity: 2,
                                       pages: PageSupport.None), validator: validator);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 80);
        Tk.Put(t, Tk.Key(t, conv), 60, s);
        // Diverge at 35 → target aligned to 34; rewind 26 > 16 is fine under ModelDecides (a clone: slack > 16).
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 35), Tk.Seq(900, 5))), s));
        Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
        Assert.Equal(34, plan.Length);
        Assert.Equal(MaterializeMode.CloneEndState, plan.Mode);
        Assert.True((plan.Clamps & ClampReasons.Granularity) != 0);
        // The validator refuses a 40-token rewind.
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 20), Tk.Seq(900, 5))), s));
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.Equal(SourceDecline.ModelRefused, plan.TruncationDecline);
        Assert.Equal(0, t.QueuedRefusalCount);   // a truncation refusal does not invalidate the payload

        // Donate-only (DSV4.1): a rewind beyond the donation slack (DEC-14 (f)) rejects the candidate,
        // one within it donates.
        PrefixTree n = Tk.Tree(Tk.Caps(endState: EndStateSupport.DonateOnly, truncation: TruncationKind.ModelDecides, granularity: 2,
                                       pages: PageSupport.None), validator: validator);
        int sn = Tk.Scope(n);
        Tk.Put(n, Tk.Key(n, conv), 60, sn);
        plan = Tk.Plan(n, Tk.Req(Tk.Key(n, Tk.Cat(Tk.Seq(1, 35), Tk.Seq(900, 5))), sn));
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.Equal(SourceDecline.DonateOnlyShared, plan.TruncationDecline);
        plan = Tk.Plan(n, Tk.Req(Tk.Key(n, Tk.Cat(Tk.Seq(1, 47), Tk.Seq(900, 5))), sn));
        Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
        Assert.Equal(46, plan.Length);
        Assert.Equal(MaterializeMode.DonateEndState, plan.Mode);
    }

    [Fact]
    public void TieOrder_PrimaryThenDonateThenBindThenCloneThenInject()
    {
        Assert.True(ResumabilityRules.TieRank(CandidateKind.PrimaryResident, MaterializeMode.KeepPrimary)
                  < ResumabilityRules.TieRank(CandidateKind.EndState, MaterializeMode.DonateEndState));
        Assert.True(ResumabilityRules.TieRank(CandidateKind.EndState, MaterializeMode.DonateEndState)
                  < ResumabilityRules.TieRank(CandidateKind.Pages, MaterializeMode.BindPagesInPlace));
        Assert.True(ResumabilityRules.TieRank(CandidateKind.Pages, MaterializeMode.BindPagesInPlace)
                  < ResumabilityRules.TieRank(CandidateKind.EndState, MaterializeMode.CloneEndState));
        Assert.True(ResumabilityRules.TieRank(CandidateKind.EndState, MaterializeMode.CloneEndState)
                  < ResumabilityRules.TieRank(CandidateKind.Pages, MaterializeMode.InjectA1Pages));
        Assert.True(ResumabilityRules.TieRank(CandidateKind.Pages, MaterializeMode.CopyA2PagesToHolder)
                  < ResumabilityRules.TieRank(CandidateKind.TruncatedEndState, MaterializeMode.DonateEndState));
        Assert.True(ResumabilityRules.TieRank(CandidateKind.TruncatedEndState, MaterializeMode.DonateEndState)
                  < ResumabilityRules.TieRank(CandidateKind.TruncatedEndState, MaterializeMode.CloneEndState));
        Assert.Equal(7, ResumabilityRules.TieRank(CandidateKind.None, MaterializeMode.None));

        // Pages (bind) and an end state at the same length: a donation beats the pages, a clone does not.
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None), host: host);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 40);
        KeyRope key = Tk.Key(t, conv);
        RadixNode n = PutPages(t, host, key, 3, s, store: PageStore.A2ModelPaged);
        t.AttachEndState(n, Tk.Holder(t));
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s, route: ExpectedRoute.BatchedPaged));
        Assert.Equal(CandidateKind.EndState, plan.Kind);
        Assert.Equal(MaterializeMode.DonateEndState, plan.Mode);
        // Make the end state shared (a child exists) → clone: bind now wins the tie.
        Tk.Put(t, key, 30, s);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 24), Tk.Seq(900, 5))), s, route: ExpectedRoute.BatchedPaged));
        Assert.Equal(CandidateKind.Pages, plan.Kind);
        Assert.Equal(MaterializeMode.BindPagesInPlace, plan.Mode);
        Assert.Equal(24, plan.Length);
    }

    [Fact]
    public void CostRule_SkipsClonesBelowMinCloneTokens()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None), host: host, minClone: 32);
        int s = Tk.Scope(t), other = Tk.Scope(t);
        int[] sys = Tk.Seq(1, 24);
        KeyRope key = Tk.Key(t, Tk.Cat(sys, Tk.Seq(100, 30)));
        // A public end state at 24 (a clone for anyone) plus public pages to 16.
        RadixNode pub = Tk.Put(t, key, 24, 0, p: 24);
        t.AttachPages(pub, Tk.Pages(host, 2));
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(sys, Tk.Seq(500, 5))), other, p: 24));
        Assert.Equal(CandidateKind.Pages, plan.Kind);
        Assert.Equal(16, plan.Length);
        Assert.True((plan.Clamps & ClampReasons.CloneCost) != 0);
        Assert.Equal(SourceDecline.CloneCost, plan.EndStateDecline);
        // Donations are not clones: a 24-token own-scope donation is kept.
        Tk.Put(t, key, 40, s, p: 24);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(sys, Tk.Seq(100, 30))), s, p: 24));
        Assert.Equal(40, plan.Length);
        Assert.Equal(MaterializeMode.DonateEndState, plan.Mode);
        // With no alternative the plan reuses nothing.
        PrefixTree t2 = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None), minClone: 32);
        Tk.Put(t2, Tk.Key(t2, sys), 24, 0, p: 24);
        plan = Tk.Plan(t2, Tk.Req(Tk.Key(t2, Tk.Cat(sys, Tk.Seq(500, 5))), Tk.Scope(t2), p: 24));
        Assert.Equal(CandidateKind.None, plan.Kind);
    }

    [Fact]
    public void MmReuseMinTokens_DropsShortReuseBeforeMedia()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None, mmReuseMin: 30));
        int s = Tk.Scope(t);
        var spans = new[] { Tk.Span(40, 48, new string('f', 64)) };
        int[] tokens = Tk.WithPlaceholders(60, spans);
        KeyRope key = Tk.Key(t, tokens, spans);
        Tk.Put(t, key, 20, s, spans: spans);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, spans), s, spans: spans));
        Assert.Equal(CandidateKind.None, plan.Kind);
        Assert.True((plan.Clamps & ClampReasons.MmThreshold) != 0);
        Assert.Equal(SourceDecline.MmThreshold, plan.EndStateDecline);
        // At or above the threshold, reuse is kept.
        Tk.Put(t, key, 36, s, spans: spans);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, tokens, spans), s, spans: spans));
        Assert.Equal(36, plan.Length);
        // Without media after the reuse point the threshold does not apply.
        int[] text = Tk.Seq(500, 60);
        Tk.Put(t, Tk.Key(t, text), 10, s);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, text), s));
        Assert.Equal(10, plan.Length);
    }

    [Fact]
    public void PrimaryAvailable_GatesThePrimaryResident()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.Any, pages: PageSupport.None));
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 50);
        KeyRope key = Tk.Key(t, conv);
        Tk.Put(t, key, 20, s);
        RadixNode primary = Tk.Put(t, key, 40, s, payload: Tk.Primary(t));
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s, primaryAvailable: true));
        Assert.Equal(CandidateKind.PrimaryResident, plan.Kind);
        Assert.Equal(MaterializeMode.KeepPrimary, plan.Mode);
        Assert.Equal(40, plan.Length);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s, primaryAvailable: false));
        Assert.Equal(CandidateKind.EndState, plan.Kind);
        Assert.Equal(20, plan.Length);
        Assert.Equal(SourceDecline.PrimaryBusy, plan.PrimaryDecline);
        // Claimed this step (state-locked) → not offered.
        LockReceipt claim = t.AcquireState(primary);
        plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s, primaryAvailable: true));
        Assert.Equal(20, plan.Length);
        Assert.Equal(SourceDecline.PrimaryClaimed, plan.PrimaryDecline);
        // Truncation from the primary: the same gates.
        var fork = Tk.Key(t, Tk.Cat(Tk.Seq(1, 35), Tk.Seq(900, 3)));
        plan = Tk.Plan(t, Tk.Req(fork, s, primaryAvailable: true));
        Assert.Equal(CandidateKind.EndState, plan.Kind);
        Assert.Equal(SourceDecline.PrimaryClaimed, plan.TruncationDecline);
        t.Release(ref claim);
        plan = Tk.Plan(t, Tk.Req(fork, s, primaryAvailable: false));
        Assert.Equal(SourceDecline.PrimaryBusy, plan.TruncationDecline);
        plan = Tk.Plan(t, Tk.Req(fork, s, primaryAvailable: true));
        Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
        Assert.Equal(MaterializeMode.KeepPrimary, plan.Mode);
        Assert.Equal(35, plan.Length);
        // A second PrimaryResident is refused (I14).
        Assert.Throws<InvalidOperationException>(() => t.AttachEndState(t.Insert(key, 45, s, 0, NodeFlags.None, null), Tk.Primary(t)));
    }

    [Fact]
    public void ModelRefusal_IsQueuedAndAppliedAfterAdmission()
    {
        var validator = new FakeValidator();
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None), validator: validator);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 50);
        KeyRope key = Tk.Key(t, conv);
        Tk.Put(t, key, 20, s);
        RadixNode deep = Tk.Put(t, key, 40, s);
        validator.Refused.Add(deep.EndState!.Key);
        long version = t.Version;
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, conv), s));
        Assert.Equal(20, plan.Length);
        Assert.Equal(SourceDecline.ModelRefused, plan.EndStateDecline);
        Assert.Equal(version, t.Version);             // Evaluate stayed pure
        Assert.Equal(1, t.QueuedRefusalCount);
        Assert.Equal(1, t.FlushQueuedInvalidations());
        Assert.False(deep.InTree);                     // the refused payload's leaf is gone
        Assert.Equal(1, t.Reclaim.Count);
        Assert.Equal(0, t.FlushQueuedInvalidations());
        Tk.Valid(t);
    }

    [Fact]
    public void BranchPosition_IsAHintWhenEnabled()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.None), branchSnapshots: true);
        int s = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 600);
        KeyRope key = Tk.Key(t, conv);
        Tk.Put(t, key, 16, s);
        t.CollectIfEmpty(t.Insert(key, 590, s, 0, NodeFlags.None, null));
        RadixNode tomb = t.Insert(key, 590, s, 0, NodeFlags.None, null);
        LockReceipt hold = t.AcquirePath(tomb);   // keep the payload-less structure alive
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 500), Tk.Seq(9000, 5))), s));
        Assert.Equal(16, plan.Length);
        Assert.Equal(496, plan.BranchPosition);
        t.Release(ref hold);
        Tk.Valid(t);
    }
}
