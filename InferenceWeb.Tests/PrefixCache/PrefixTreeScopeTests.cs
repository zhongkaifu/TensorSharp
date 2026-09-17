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

public class PrefixTreeScopeTests
{
    private const int B = Tk.B;

    [Fact]
    public void RetireScope_DeletesUnlockedNodes_FlagsLockedOnes_DeletesThemOnUnlock()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host);
        int s = Tk.Scope(t), keep = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 100));
        RadixNode pub = Tk.Put(t, key, 16, 0, p: 16);
        RadixNode internalNode = Tk.Put(t, key, 40, s, p: 16);
        RadixNode lockedLeaf = Tk.Put(t, key, 70, s, p: 16);
        RadixNode freeLeaf = Tk.Put(t, Tk.Key(t, Tk.Cat(Tk.Seq(1, 50), Tk.Seq(700, 30))), 60, s, p: 16);
        t.AttachPages(freeLeaf, Tk.Pages(host, 7));
        RadixNode other = Tk.Put(t, key, 50, keep, p: 16);
        LockReceipt run = t.AcquirePath(lockedLeaf);
        ScopeId id = t.Scopes[s].Id;

        Assert.True(t.RetireScope(id));
        Assert.False(freeLeaf.InTree);                               // unlocked leaf: deleted with its pages
        Assert.True(lockedLeaf.InTree);
        Assert.True((lockedLeaf.Flags & NodeFlags.Retired) != 0);
        Assert.True(internalNode.InTree);                            // path-locked through the running request
        Assert.True(pub.InTree);
        Assert.True(other.InTree);
        Assert.False(t.RetireScope(id));                             // already retired: unknown id now
        Assert.False(t.RetireScope(ScopeId.Public));
        Assert.False(t.RetireScope(ScopeId.NewFresh()));
        Tk.Valid(t);
        // A new request with the same id gets a fresh record (index differs); the old nodes stay invisible.
        int again = t.InternScope(id, ScopeKind.Session);
        Assert.NotEqual(s, again);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 100)), again, p: 16));
        Assert.Equal(16, plan.Length);

        t.Release(ref run);
        Assert.False(lockedLeaf.InTree);
        Assert.False(internalNode.InTree);
        Assert.True(pub.InTree);
        Assert.False(t.Scopes.IsLive(s));                            // recycled: no nodes, no requests
        // Pages 0-1 ([0, 16) is public content) stay on the public node; pages 2-6 went with the scope.
        Assert.Equal(5, host.TreeFrees);
        Assert.Equal(2, pub.PageCount);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));
    }

    [Fact]
    public void RetireScope_WithWaitingRequests_RecyclesOnlyAfterThey_Leave()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        Tk.Put(t, Tk.Key(t, Tk.Seq(1, 30)), 20, s);
        t.NoteRequests(s, 1, 0);
        t.RetireScope(t.Scopes[s].Id);
        Assert.Equal(0, t.Scopes[s].NodeCount);
        Assert.True(t.Scopes.IsLive(s));
        t.NoteRequests(s, -1, 0);
        Assert.False(t.Scopes.IsLive(s));
        Tk.Valid(t);
    }

    [Fact]
    public void AnotherScopesNodes_AreNeverMatchedTruncatedClonedOrDonated()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(truncation: TruncationKind.Any, pages: PageSupport.Both, rewindCap: 1000), host: host);
        int a = Tk.Scope(t), b = Tk.Scope(t);
        int[] conv = Tk.Seq(1, 100);
        KeyRope key = Tk.Key(t, conv);
        // Scope a: pages, an end state, a truncation source and a primary resident, all past P = 8.
        RadixNode pagesNode = t.Insert(key, 6 * B, a, 8, NodeFlags.None, null);
        t.AttachPages(pagesNode, Tk.Pages(host, 6));
        Tk.Put(t, key, 60, a, p: 8);
        Tk.Put(t, key, 90, a, p: 8, payload: Tk.Primary(t));
        foreach (ExpectedRoute route in Enum.GetValues<ExpectedRoute>())
        {
            foreach (int diverge in new[] { 30, 55, 60, 75, 95 })
            {
                KeyRope probe = Tk.Key(t, Tk.Cat(Tk.Seq(1, diverge), Tk.Seq(900, 5)));
                MatchPlan plan = Tk.Plan(t, Tk.Req(probe, b, p: 8, route: route, primaryAvailable: true));
                Assert.True(plan.Length <= 8, $"scope b reused {plan.Length} ({plan}) at diverge {diverge}");
                Assert.True(plan.PayloadNode is null || plan.PayloadNode.ScopeIx != a);
                // Scope a reuses beyond P.
                MatchPlan own = Tk.Plan(t, Tk.Req(probe, a, p: 8, route: route, primaryAvailable: true));
                Assert.True(own.Length > 8 || diverge <= 8);
            }
        }
    }

    [Fact]
    public void UnscopedRequests_AreIsolated_I29()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int[] sys = Tk.Seq(1, 16);
        int[] prompt = Tk.Cat(sys, Tk.Seq(100, 40));
        // Two requests without a scope: each gets a fresh random scope.
        int u1 = t.InternScope(ScopeId.NewFresh(), ScopeKind.Unscoped);
        int u2 = t.InternScope(ScopeId.NewFresh(), ScopeKind.Unscoped);
        Assert.NotEqual(u1, u2);
        Tk.Put(t, Tk.Key(t, sys), 16, 0, p: 16);
        Tk.Put(t, Tk.Key(t, prompt), 50, u1, p: 16);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, prompt), u2, p: 16));
        Assert.Equal(16, plan.Length);                                // public prefix only
        Assert.Equal(16, plan.PublicTokens);
        Assert.Throws<InvalidOperationException>(() => t.InternScope(t.Scopes[u1].Id, ScopeKind.Unscoped));
        Tk.Valid(t);
    }

    [Fact]
    public void InvalidatePayload_DetachesWithoutRelease_AndBackendRecreatedDropsDeviceState()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host, pageHostBytes: 10);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 100));
        RadixNode n = t.Insert(key, 6 * B, s, 0, NodeFlags.None, null);
        PageRef[] pages = new[]
        {
            new PageRef(host.NewBlock(PageStore.A1HostSlab), 0, PageStore.A1HostSlab, true),
            new PageRef(host.NewBlock(PageStore.A2ModelPaged), 1, PageStore.A2ModelPaged, true),
            new PageRef(host.NewBlock(PageStore.Both), 2, PageStore.Both, true),
        };
        t.AttachPages(n, pages);
        RadixNode es = Tk.Put(t, key, 70, s);
        string key70 = es.EndState!.Key;
        Assert.True(t.InvalidatePayload(key70));
        Assert.False(es.InTree);
        Assert.Equal(0, t.Reclaim.Count);                            // the model already freed it
        Assert.False(t.InvalidatePayload(key70));
        Assert.False(t.InvalidatePayload(null!));
        RadixNode es2 = Tk.Put(t, key, 80, s);
        t.AttachEndState(n, Tk.Holder(t));
        t.InvalidateDeviceState();
        Assert.False(es2.InTree);
        Assert.Null(n.EndState);
        Assert.Equal(2, n.PageCount);                                // A2-only page freed; Both downgraded to A1
        Assert.All(n.PageSpan.ToArray(), p => Assert.Equal(PageStore.A1HostSlab, p.Store));
        Assert.Equal(2, t.Reclaim.Count);
        Tk.Valid(t);
    }

    [Fact]
    public void Reset_ReleasesEverything_AndInvalidatesReceipts()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host, strictReceipts: false);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 100));
        RadixNode n = t.Insert(key, 4 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(n, Tk.Pages(host, 4));
        RadixNode es = Tk.Put(t, key, 60, s);
        Tk.Put(t, key, 16, 0, p: 16);
        LockReceipt r = t.AcquirePath(es);
        long serial = t.TreeSerial;
        t.Release(ref r);
        r = t.AcquirePath(es);
        t.Reset();
        Assert.Equal(serial + 1, t.TreeSerial);
        Assert.Equal(0, t.NodeCount);
        Assert.Equal(0, host.TreeRefs);
        Assert.True(t.Cached.IsZero);
        Assert.Equal(2, t.Reclaim.Count);                            // the two end states; pages go straight back
        t.Release(ref r);                                            // a pre-reset receipt is stale
        Assert.Equal(1, t.Counters.StaleReceipts);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));
        // The tree is usable after a reset.
        Tk.Put(t, Tk.Key(t, Tk.Seq(1, 30)), 20, s);
        Tk.Valid(t);
    }
}
