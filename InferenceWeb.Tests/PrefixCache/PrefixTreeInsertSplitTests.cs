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

public class PrefixTreeInsertSplitTests
{
    private const int B = Tk.B;

    [Fact]
    public void Insert_SplitsAtP_AndFlagsThePublicBoundary()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 60));
        RadixNode leaf = t.Insert(key, 50, s, 20, NodeFlags.None, null);
        Assert.Equal(50, leaf.Depth);
        Assert.Equal(s, leaf.ScopeIx);
        RadixNode boundary = leaf.Parent!;
        Assert.Equal(20, boundary.Depth);
        Assert.Equal(0, boundary.ScopeIx);
        Assert.True(boundary.IsPublicBoundary);
        Assert.Same(t.Root, boundary.Parent);
        Assert.Same(leaf, t.Scopes[s].NewestLeaf);
        t.AttachEndState(leaf, Tk.Holder(t));
        Tk.Valid(t);
        // P == len: a public insert that ends on the boundary.
        RadixNode pub = t.Insert(Tk.Key(t, Tk.Seq(1, 60)), 20, 0, 20, NodeFlags.None, null);
        Assert.Same(boundary, pub);
        // Public inserts must stay within P; scoped inserts need a live scope; flags are restricted.
        Assert.Throws<ArgumentException>(() => t.Insert(key, 30, 0, 20, NodeFlags.None, null));
        Assert.Throws<ArgumentException>(() => t.Insert(key, 30, 77, 20, NodeFlags.None, null));
        Assert.Throws<ArgumentException>(() => t.Insert(key, 30, s, 20, NodeFlags.DonationPending, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.Insert(key, 0, s, 20, NodeFlags.None, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.Insert(key, 61, s, 20, NodeFlags.None, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.Insert(key, 30, s, -1, NodeFlags.None, null));
        Assert.Throws<ArgumentNullException>(() => t.Insert(null!, 30, s, 0, NodeFlags.None, null));
        PrefixTree small = Tk.Tree(contextLength: 16);
        Assert.Throws<ArgumentOutOfRangeException>(() => small.Insert(Tk.Key(small, Tk.Seq(1, 20)), 20, Tk.Scope(small), 0, NodeFlags.None, null));
    }

    [Fact]
    public void Insert_PublicNodeContinuingPastP_IsSplitAtTheInsertersP()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int a = Tk.Scope(t), b = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode long40 = Tk.Put(t, key, 40, 0, p: 40);          // a 40-token public system prompt
        // Request b declares P = 24 over the same tokens and continues in its own scope.
        RadixNode bLeaf = Tk.Put(t, key, 60, b, p: 24);
        RadixNode at24 = bLeaf.Parent!;
        Assert.Equal(24, at24.Depth);
        Assert.Equal(0, at24.ScopeIx);
        Assert.True(at24.IsPublicBoundary);
        Assert.Same(at24, long40.Parent);                          // the public node was split at 24
        Assert.Equal(b, bLeaf.ScopeIx);
        Assert.Equal(24, bLeaf.EdgeStartDepth);                    // b's own branch duplicates [24, 40)
        Assert.Equal(2, at24.Children.Count);
        // Scope a never sees b's branch; b sees both.
        MatchPlan pa = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 70)), a, p: 40));
        Assert.Equal(40, pa.Length);
        MatchPlan pb = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 70)), b, p: 24));
        Assert.Equal(60, pb.Length);
        Tk.Valid(t);
    }

    [Fact]
    public void ScopedUnderPublic_AndADifferentScopeUnderAScopedNodeIsImpossible()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int a = Tk.Scope(t), b = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 60));
        RadixNode aLeaf = Tk.Put(t, key, 40, a, p: 10);
        // b inserts the same tokens with P = 10: it creates its own branch under the public node at 10.
        RadixNode bLeaf = Tk.Put(t, key, 50, b, p: 10);
        Assert.Equal(10, bLeaf.EdgeStartDepth);
        Assert.Equal(b, bLeaf.ScopeIx);
        Assert.NotSame(aLeaf, bLeaf.Parent);
        // b with P = 20 (past a's scoped start) still never descends into a's node (I3).
        RadixNode b2 = Tk.Put(t, Tk.Key(t, Tk.Cat(Tk.Seq(1, 20), Tk.Seq(300, 20))), 40, b, p: 20);
        for (RadixNode? n = b2; n is not null && !n.IsRoot; n = n.Parent)
            Assert.True(n.ScopeIx == 0 || n.ScopeIx == b);
        Tk.Valid(t);
    }

    [Fact]
    public void DuplicateEndState_IsQueued_TombstoneIsRevived()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 60));
        RadixNode n = Tk.Put(t, key, 40, s);
        EndStatePayload dup = Tk.Holder(t, 500);
        Assert.Equal(AttachResult.Duplicate, t.AttachEndState(t.Insert(key, 40, s, 0, NodeFlags.None, null), dup));
        Assert.Equal(1, t.Counters.DuplicatesDropped);
        Assert.True(t.Reclaim.Contains(dup.Key));
        Assert.Equal(500, t.PendingReclaim.HostKv);
        // Tombstone (end state detached) with a child: an incoming state revives it.
        Tk.Put(t, key, 50, s);
        t.DetachEndState(n, ReleaseReason.Evicted);
        Assert.Null(n.EndState);
        Assert.Equal(AttachResult.Revived, t.AttachEndState(n, Tk.Holder(t)));
        Assert.Throws<ArgumentException>(() => t.AttachEndState(n, new EndStatePayload { Key = "req-1" }));
        Assert.Throws<ArgumentException>(() => t.AttachEndState(n, new EndStatePayload { Key = dup.Key }));
        Assert.Throws<ArgumentException>(() => t.AttachEndState(t.Root, Tk.Holder(t)));
        Assert.Throws<ArgumentNullException>(() => t.AttachEndState(n, null!));
        // I20: only media-free public boundaries persist.
        Assert.Throws<ArgumentException>(() => t.AttachEndState(t.Insert(key, 55, s, 0, NodeFlags.None, null), Tk.Holder(t, persisted: true)));
        t.CollectIfEmpty(t.Insert(key, 55, s, 0, NodeFlags.None, null));
        Tk.Valid(t);
    }

    [Fact]
    public void PageOwnerRule_HoldsAcrossSplits_AndDuplicatesAreSkipped()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode n = t.Insert(key, 6 * B, s, 0, NodeFlags.None, null);
        PageRef[] pages = Tk.Pages(host, 6);
        Assert.Equal(6, t.AttachPages(n, pages));
        Assert.Equal(6, n.PageCount);
        // Split at 3B+3: pages 0..2 end at or before 3B+3 → the new parent; 3..5 stay.
        RadixNode p = t.SplitAt(n, 3 * B + 3);
        Assert.Equal(3, p.PageCount);
        Assert.Equal(3, n.PageCount);
        Assert.Equal(new[] { 0, 1, 2 }, p.PageSpan.ToArray().Select(x => x.PageIndex));
        Assert.True(t.TryGetBlockOwner(pages[1].Block, out RadixNode owner) && ReferenceEquals(owner, p));
        Tk.Valid(t);
        // Attaching the same page indices through the deeper node is a no-op.
        Assert.Equal(0, t.AttachPages(n, Tk.Pages(host, 6)));
        Assert.Equal(6, t.Counters.PagesDuplicate);
        // A page beyond the node is an error; an unbacked page is refused (I12, G-03).
        Assert.Throws<ArgumentOutOfRangeException>(() => t.AttachPages(p, new[] { new PageRef(host.NewBlock(PageStore.Both), 5, PageStore.Both, true) }));
        RadixNode deeper = t.Insert(key, 8 * B, s, 0, NodeFlags.None, null);
        Assert.Equal(0, t.AttachPages(deeper, new[] { new PageRef(host.NewUnbackedBlock(), 6, PageStore.Both, true) }));
        Assert.Equal(0, t.AttachPages(deeper, new[] { new PageRef(host.NewBlock(PageStore.A2ModelPaged), 6, PageStore.Both, true) }));
        Assert.Equal(0, t.AttachPages(deeper, new[] { new PageRef(host.NewBlock(PageStore.Both, fullA1: false), 6, PageStore.A1HostSlab, true) }));
        Assert.Equal(0, t.AttachPages(deeper, new[] { new PageRef(host.NewBlock(PageStore.Both), 6, (PageStore)0, true) }));
        Assert.Equal(4, t.Counters.PagesRefused);
        Assert.Equal(1, t.AttachPages(deeper, new[] { new PageRef(host.NewBlock(PageStore.A2ModelPaged), 6, PageStore.A2ModelPaged, true) }));
        Assert.Throws<ArgumentException>(() => t.AttachPages(t.Root, pages));
        Tk.Valid(t);
    }

    [Fact]
    public void MediaRecords_MoveOnSplit()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        var spans = new[] { Tk.Span(4, 8, new string('1', 64)), Tk.Span(20, 30, new string('2', 64)) };
        int[] tokens = Tk.WithPlaceholders(40, spans);
        KeyRope key = Tk.Key(t, tokens, spans);
        RadixNode n = Tk.Put(t, key, 36, s, spans: spans);
        Assert.Equal(2, n.MediaSpanCount);
        RadixNode p = t.SplitAt(n, 12);
        Assert.Equal(1, p.MediaSpanCount);
        Assert.Equal(4, p.SpanRecords[0].Start);
        Assert.Equal(1, n.MediaSpanCount);
        Assert.Equal(20, n.SpanRecords[0].Start);
        // A split inside a span is structure only; the record stays with the node holding its Start.
        RadixNode q = t.SplitAt(n, 12);          // depth 24, inside [20, 30)
        Assert.Equal(1, q.MediaSpanCount);
        Assert.Equal(0, n.MediaSpanCount);
        Tk.Valid(t);
        Assert.Throws<ArgumentOutOfRangeException>(() => t.SplitAt(n, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.SplitAt(n, n.Edge.Length));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.SplitAt(t.Root, 1));
    }

    [Fact]
    public void SplittingAPagesOnlyLeaf_DropsTheEmptyRemainder()
    {
        // Harness seed 15 (OracleS2): a public leaf [0, 20) holding pages 0-1 is split at 16 by an
        // Acquire; every page moves to the new parent and the remainder [16, 20) must not stay behind
        // as an unlocked payload-less leaf (I5).
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.A1HostSlab, endState: EndStateSupport.None, truncation: TruncationKind.None), host: host);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 40));
        RadixNode leaf = t.Insert(key, 20, 0, 20, NodeFlags.None, null);
        t.AttachPages(leaf, Tk.Pages(host, 2, PageStore.A1HostSlab));
        Tk.Valid(t);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 40)), s, p: 20));
        Assert.Equal(CandidateKind.Pages, plan.Kind);
        Assert.Equal(16, plan.Length);
        LockReceipt r = t.Acquire(plan);
        Assert.False(leaf.InTree);
        Assert.Equal(16, r.PathDepth);
        Assert.Equal(1, t.NodeCount);
        Tk.Valid(t);
        t.Release(ref r);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));
    }

    [Fact]
    public void Split_IsO1_WithSharedRopeSlices()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        int[] tokens = Tk.Seq(1, 20000);
        KeyRope key = Tk.Key(t, tokens);
        RadixNode n = Tk.Put(t, key, 19000, s);
        long chunksBefore = t.KeyPool.Rented;
        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        RadixNode p = t.SplitAt(n, 9000);
        long alloc = GC.GetAllocatedBytesForCurrentThread() - allocBefore;
        Assert.Same(key, p.Edge.Rope);
        Assert.Same(key, n.Edge.Rope);
        Assert.Equal(2, key.SliceRefs);
        Assert.Equal(19000, key.LiveSliceTokens);
        Assert.Equal(chunksBefore, t.KeyPool.Rented);   // no key copy
        Assert.True(alloc < 4096, $"split allocated {alloc} bytes");
        Tk.Valid(t);
    }

    [Fact]
    public void Insert_BreakpointAndPromptEndFlags_AreRecorded()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 60));
        RadixNode bp = Tk.Put(t, key, 30, s, flags: NodeFlags.EndsAtBreakpoint);
        Assert.True((bp.Flags & NodeFlags.EndsAtBreakpoint) != 0);
        Assert.Equal(EvictionTier.ScopeNewest, t.TierOf(bp));        // it is also the scope's newest leaf
        RadixNode later = Tk.Put(t, key, 50, s, flags: NodeFlags.PromptEnd);
        Assert.Equal(EvictionTier.Breakpoint, t.TierOf(bp));
        Assert.Equal(EvictionTier.ScopeNewest, t.TierOf(later));
        Tk.Valid(t);
    }
}
