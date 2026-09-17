// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class PrefixTreeEvictionTests
{
    private const int B = Tk.B;

    [Fact]
    public void LeafFirst_TombstoneCascade_InternalEndStateDetach()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t), other = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode mid = Tk.Put(t, key, 30, s, host: 100);
        RadixNode leaf = Tk.Put(t, key, 60, s, host: 200);
        Tk.Put(t, Tk.Key(t, Tk.Seq(500, 20)), 10, other, host: 1);   // newest leaf of `s` moves? no: other scope
        Tk.Put(t, Tk.Key(t, Tk.Seq(300, 80)), 5, s, host: 1);        // make `leaf` not the newest of s
        // Ordinary tier, oldest first: `mid`'s end state (internal → StateLru) is older than `leaf`.
        Assert.Equal(EvictionLists.ListId(LruKind.State, EvictionTier.Ordinary), mid.LruList);
        Assert.Equal(EvictionLists.ListId(LruKind.Leaf, EvictionTier.Ordinary), leaf.LruList);
        Assert.True(t.Evict(ResourceClass.HostKv, 50, ReleaseReason.Evicted, EvictionTier.Ordinary));
        Assert.Null(mid.EndState);                    // internal end state detached first (older)
        Assert.True(mid.InTree);                      // tombstone kept: it has a child
        Tk.Valid(t);
        // Evicting the leaf cascades through the payload-less tombstone.
        Assert.True(t.Evict(ResourceClass.HostKv, 150, ReleaseReason.Evicted, EvictionTier.Ordinary));
        Assert.False(leaf.InTree);
        Assert.False(mid.InTree);
        Assert.Equal(2, t.Reclaim.Count);
        Tk.Valid(t);
        // Nothing evictable left below ScopeNewest.
        Assert.False(t.Evict(ResourceClass.HostKv, 1000, ReleaseReason.Evicted, EvictionTier.Ordinary));
        Assert.True(t.Evict(ResourceClass.HostKv, 0, ReleaseReason.Evicted, EvictionTier.Ordinary));
    }

    [Fact]
    public void RetiredTier_IsEvictedFirst()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t), gone = Tk.Scope(t);
        RadixNode ord = Tk.Put(t, Tk.Key(t, Tk.Seq(1, 20)), 10, s, host: 1);
        Tk.Put(t, Tk.Key(t, Tk.Seq(50, 20)), 10, s, host: 1);
        RadixNode retired = Tk.Put(t, Tk.Key(t, Tk.Seq(100, 20)), 10, gone, host: 1);
        LockReceipt path = t.AcquirePath(retired);       // a running request of the retired scope
        Assert.True(t.RetireScope(t.Scopes[gone].Id));
        Assert.True(retired.InTree);
        Assert.True((retired.Flags & NodeFlags.Retired) != 0);
        Assert.Equal(EvictionTier.Retired, t.TierOf(retired));
        Assert.Equal(EvictionLists.ListId(LruKind.State, EvictionTier.Retired), retired.LruList);
        Tk.Valid(t);
        Assert.True(t.Evict(ResourceClass.HostKv, 1, ReleaseReason.Evicted, EvictionTier.Ordinary));
        Assert.Null(retired.EndState);                   // the retired end state went first
        Assert.NotNull(ord.EndState);
        t.Release(ref path);
        Assert.False(retired.InTree);
        Tk.Valid(t);
    }

    [Fact]
    public void EvictionOrder_FollowsTiers()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None), publicMax: 1);
        int s = Tk.Scope(t);
        RadixNode pubTop = Tk.Put(t, Tk.Key(t, Tk.Seq(1, 20)), 10, 0, p: 10, host: 1);
        RadixNode bp = Tk.Put(t, Tk.Key(t, Tk.Seq(100, 20)), 12, s, flags: NodeFlags.EndsAtBreakpoint, host: 1);
        RadixNode ord = Tk.Put(t, Tk.Key(t, Tk.Seq(200, 20)), 12, s, host: 1);
        RadixNode newest = Tk.Put(t, Tk.Key(t, Tk.Seq(300, 20)), 12, s, host: 1);
        var evicted = new List<RadixNode>();
        var all = new[] { pubTop, bp, ord, newest };
        for (int i = 0; i < 4; i++)
        {
            Assert.True(t.Evict(ResourceClass.HostKv, 1, ReleaseReason.Evicted, EvictionTier.PublicTop));
            evicted.Add(all.Single(n => !n.InTree && !evicted.Contains(n)));
            Tk.Valid(t);
        }
        Assert.Equal(new[] { ord, bp, newest, pubTop }, evicted);
        // A ceiling keeps higher tiers.
        PrefixTree t2 = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s2 = Tk.Scope(t2);
        RadixNode keep = Tk.Put(t2, Tk.Key(t2, Tk.Seq(1, 20)), 12, s2, host: 1);
        Assert.False(t2.Evict(ResourceClass.HostKv, 1, ReleaseReason.Evicted, EvictionTier.Breakpoint));
        Assert.True(keep.InTree);
    }

    [Fact]
    public void PublicMax_LimitsTierPublicTop_AndTheCountSubCap()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None), publicMax: 2);
        var boundaries = new List<RadixNode>();
        for (int i = 0; i < 4; i++)
            boundaries.Add(Tk.Put(t, Tk.Key(t, Tk.Seq(1000 * (i + 1), 20)), 10, 0, p: 10, host: 10));
        Assert.Equal(2, boundaries.Count(n => t.TierOf(n) == EvictionTier.PublicTop));
        Assert.Equal(EvictionTier.PublicTop, t.TierOf(boundaries[3]));
        Assert.Equal(EvictionTier.PublicTop, t.TierOf(boundaries[2]));
        Tk.Valid(t);
        // Touching an old boundary promotes it.
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1000, 20)), Tk.Scope(t), p: 10));
        LockReceipt r = t.Acquire(plan);
        t.Release(ref r);
        Assert.Equal(EvictionTier.PublicTop, t.TierOf(boundaries[0]));
        Assert.NotEqual(EvictionTier.PublicTop, t.TierOf(boundaries[2]));
        Tk.Valid(t);
        // Count sub-cap: 4 public end states > PublicMax 2 → the two oldest go.
        Assert.True(t.EnforceCountSubCaps());
        Assert.Equal(2, t.PublicEndStateCount);
        Assert.True(boundaries[0].InTree);
        Assert.True(boundaries[3].InTree);
        Tk.Valid(t, new InvariantCheckContext(AfterEvictionTrigger: true));
    }

    [Fact]
    public void ScopeQuota_AndScopedCountSubCap()
    {
        long spare = 400;
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None), scopedMax: 2, optionCap: new ResourceVector { HostKv = 400, DeviceKv = 400, StateSnapshot = 400, NativeSlot = 400 },
                               spare: c => spare);
        int s = Tk.Scope(t);
        t.RefreshBudgets();
        Assert.Equal(1600 / 4, t.ScopeQuota);
        RadixNode a = Tk.Put(t, Tk.Key(t, Tk.Seq(1, 20)), 10, s, host: 100);
        Assert.Equal(EvictionTier.ScopeNewest, t.TierOf(a));
        RadixNode b = Tk.Put(t, Tk.Key(t, Tk.Seq(100, 20)), 10, s, host: 350);   // scope bytes 450 > quota 400
        Assert.Equal(EvictionTier.Ordinary, t.TierOf(b));
        Tk.Valid(t);
        RadixNode c = Tk.Put(t, Tk.Key(t, Tk.Seq(200, 20)), 10, s, host: 1);
        Assert.Equal(3, t.ScopedEndStateCount);
        Assert.True(t.EnforceCountSubCaps());
        Assert.Equal(2, t.ScopedEndStateCount);
        Assert.False(a.InTree);                       // the oldest in the category
        // An idle scope loses ScopeNewest protection.
        long now = 0;
        PrefixTree idle = Tk.Tree(Tk.Caps(pages: PageSupport.None), clock: () => now);
        int si = Tk.Scope(idle);
        idle.NoteRequests(si, 1, 0);
        RadixNode leaf = Tk.Put(idle, Tk.Key(idle, Tk.Seq(1, 20)), 10, si);
        idle.NoteRequests(si, -1, 0);
        Assert.Equal(EvictionTier.ScopeNewest, idle.TierOf(leaf));
        now = 700_000;
        idle.RefreshScopeActivity();
        Assert.Equal(EvictionTier.Ordinary, idle.TierOf(leaf));
        Tk.Valid(idle);
        idle.NoteRequests(si, 0, 1);
        Assert.Equal(EvictionTier.ScopeNewest, idle.TierOf(leaf));
        Assert.Throws<InvalidOperationException>(() => idle.NoteRequests(si, -5, 0));
        idle.NoteRequests(0, 1, 1);                   // public: ignored
    }

    [Fact]
    public void LockedPinnedAndDonationPending_AreNeverEvicted()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t), other = Tk.Scope(t);
        RadixNode locked = Tk.Put(t, Tk.Key(t, Tk.Seq(1, 20)), 10, s, host: 10);
        RadixNode pinned = Tk.Put(t, Tk.Key(t, Tk.Seq(100, 20)), 10, s, host: 10);
        RadixNode pending = Tk.Put(t, Tk.Key(t, Tk.Seq(200, 20)), 10, s, host: 10);
        Tk.Put(t, Tk.Key(t, Tk.Seq(300, 20)), 10, other, host: 1);
        LockReceipt r = t.AcquireState(locked);
        t.Pin(pinned);
        t.MarkDonationPending(pending);
        Assert.False(t.Evict(ResourceClass.HostKv, 1000, ReleaseReason.Evicted, EvictionTier.PublicTop));
        Assert.True(locked.InTree && pinned.InTree && pending.InTree);
        Assert.NotNull(locked.EndState);
        Assert.NotNull(pinned.EndState);
        Assert.NotNull(pending.EndState);
        Tk.Valid(t, new InvariantCheckContext(TransactionOpen: true));
        t.RelieveMemoryPressure(PressureLevel.Critical);
        Assert.True(locked.InTree && pinned.InTree && pending.InTree);
        // Commit consumes the donation without a release; cancel restores visibility.
        t.CommitDonation(pending);
        Assert.False(pending.InTree);
        Assert.Throws<InvalidOperationException>(() => t.CommitDonation(locked));
        t.CancelDonation(locked);                     // not pending: no-op
        t.Release(ref r);
        t.Unpin(pinned);
        Assert.Throws<InvalidOperationException>(() => t.Unpin(pinned));
        Assert.Throws<InvalidOperationException>(() => t.MarkDonationPending(t.Insert(Tk.Key(t, Tk.Seq(900, 20)), 10, s, 0, NodeFlags.None, null)));
        t.CollectIfEmpty(t.Insert(Tk.Key(t, Tk.Seq(900, 20)), 10, s, 0, NodeFlags.None, null));
        Assert.Throws<InvalidOperationException>(() => t.Pin(t.Root));
        Tk.Valid(t);
    }

    [Fact]
    public void E7_InternalPagesAreFreedOnlyWithTheLeaf()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode parent = t.Insert(key, 2 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(parent, Tk.Pages(host, 2));
        RadixNode child = t.Insert(key, 4 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(child, Tk.Pages(host, 2, first: 2));
        // Only leaves are in LeafLru: the parent's pages cannot be taken while the child exists.
        Assert.Equal(0, parent.LruList);
        Assert.True(t.Evict(ResourceClass.PoolPages, 1, ReleaseReason.Evicted, EvictionTier.PublicTop));
        Assert.False(child.InTree);
        Assert.Equal(2, parent.PageCount);            // untouched until it becomes a leaf
        Assert.Equal(2, host.TreeRefs);
        Assert.True(t.Evict(ResourceClass.PoolPages, 2, ReleaseReason.Evicted, EvictionTier.PublicTop));
        Assert.Equal(0, host.TreeRefs);
        Tk.Valid(t);
    }

    [Fact]
    public void PromptEndSplit_EvictsOutputPagesFirst()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host);
        int s = Tk.Scope(t), other = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        // Finish: pages up to the prompt end (flagged PromptEnd) go in first, then the output pages.
        RadixNode prompt = t.Insert(key, 4 * B, s, 0, NodeFlags.PromptEnd, null);
        t.AttachPages(prompt, Tk.Pages(host, 4));
        RadixNode output = t.Insert(key, 6 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(output, Tk.Pages(host, 2, first: 4));
        Tk.Put(t, Tk.Key(t, Tk.Seq(500, 20)), 10, other, host: 1);
        Assert.True((prompt.Flags & NodeFlags.PromptEnd) != 0);
        // Output is the deeper leaf: evicted first, the prompt pages survive for the next turn.
        t.RefreshBudgets();
        Assert.True(t.Evict(ResourceClass.PoolPages, 2, ReleaseReason.Evicted, EvictionTier.ScopeNewest));
        Assert.False(output.InTree);
        Assert.True(prompt.InTree);
        Assert.Equal(4, prompt.PageCount);
        Tk.Valid(t);
    }

    [Fact]
    public void ReclaimQueue_ReleasesOneBatchPerDrain()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        for (int i = 0; i < 5; i++)
            Tk.Put(t, Tk.Key(t, Tk.Seq(100 * (i + 1), 20)), 10, s, host: 10);
        Tk.Put(t, Tk.Key(t, Tk.Seq(9000, 20)), 10, Tk.Scope(t), host: 10);
        Assert.True(t.Evict(ResourceClass.HostKv, 40, ReleaseReason.Evicted, EvictionTier.ScopeNewest));
        Assert.Equal(4, t.Reclaim.Count);
        Assert.Equal(40, t.PendingReclaim.HostKv);
        Assert.Equal(10 * 6 - 40, t.Cached.HostKv);
        var calls = new List<(string[] Keys, ReleaseReason Reason)>();
        int released = t.DrainReclaimQueue((keys, reason) => calls.Add((keys.ToArray(), reason)));
        Assert.Equal(4, released);
        Assert.Single(calls);
        Assert.Equal(4, calls[0].Keys.Length);
        Assert.Equal(ReleaseReason.Evicted, calls[0].Reason);
        Assert.True(t.PendingReclaim.IsZero);
        Assert.Equal(0, t.DrainReclaimQueue((keys, reason) => calls.Add((keys.ToArray(), reason))));
        Assert.Single(calls);                          // an empty queue makes no call
        // The first reason labels a mixed batch; re-queuing a key is ignored.
        var q = new ReclaimQueue();
        Assert.True(q.Enqueue("pc:0:1", ReleaseReason.Duplicate, new ResourceVector { HostKv = 1 }));
        Assert.False(q.Enqueue("pc:0:1", ReleaseReason.Evicted, new ResourceVector { HostKv = 1 }));
        Assert.True(q.Enqueue("pc:0:2", ReleaseReason.Pressure, default));
        Assert.Throws<ArgumentException>(() => q.Enqueue("", ReleaseReason.Evicted, default));
        ReleaseReason seen = ReleaseReason.Reset;
        q.Drain((keys, reason) => seen = reason);
        Assert.Equal(ReleaseReason.Duplicate, seen);
        Assert.Equal(1, q.DrainCalls);
        Assert.Equal(2, q.ReleasedKeys);
        Assert.Equal(0, q.Drain(null));
        q.Enqueue("pc:0:3", ReleaseReason.Evicted, default);
        Assert.Equal(1, q.Drain(null));                // a null sink drops the batch
        // A sink that throws still clears the queue.
        q.Enqueue("pc:0:4", ReleaseReason.Evicted, default);
        Assert.Throws<InvalidOperationException>(() => q.Drain((k, r) => throw new InvalidOperationException()));
        Assert.Equal(0, q.Count);
    }

    [Fact]
    public void RelieveMemoryPressure_ModerateHalvesCritical_KeepsNewestPublicAndScopeLeaf()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None), publicMax: 2, optionCap: new ResourceVector { HostKv = 1000 });
        int a = Tk.Scope(t), b = Tk.Scope(t);
        RadixNode pub1 = Tk.Put(t, Tk.Key(t, Tk.Seq(1, 20)), 10, 0, p: 10, host: 100);
        RadixNode pub2 = Tk.Put(t, Tk.Key(t, Tk.Seq(100, 20)), 10, 0, p: 10, host: 100);
        RadixNode a1 = Tk.Put(t, Tk.Key(t, Tk.Seq(200, 20)), 10, a, host: 100, flags: NodeFlags.EndsAtBreakpoint);
        RadixNode a2 = Tk.Put(t, Tk.Key(t, Tk.Seq(300, 20)), 10, a, host: 200);
        t.NoteRequests(b, 1, 0);
        t.NoteRequests(b, -1, 0);
        RadixNode b1 = Tk.Put(t, Tk.Key(t, Tk.Seq(400, 20)), 10, b, host: 200);
        RadixNode b0 = Tk.Put(t, Tk.Key(t, Tk.Seq(500, 20)), 10, b, host: 150);
        RadixNode bNew = Tk.Put(t, Tk.Key(t, Tk.Seq(600, 20)), 10, b, host: 50);
        Assert.Equal(900, t.Cached.HostKv);
        // Moderate: HostKv to 50% of the 1000 cap (need 400), never above tier Breakpoint:
        // Ordinary b1 + b0 (350), then Breakpoint a1 (100).
        t.RelieveMemoryPressure(PressureLevel.Moderate);
        Assert.Equal(450, t.Cached.HostKv);
        Assert.False(b1.InTree || b0.InTree || a1.InTree);
        Assert.True(pub1.InTree && pub2.InTree && a2.InTree && bNew.InTree);
        Tk.Valid(t);
        // Critical: everything except the newest PublicTop node and the most recently used scope's newest leaf.
        t.RelieveMemoryPressure(PressureLevel.Critical);
        Assert.True(pub2.InTree);
        Assert.True(bNew.InTree);
        Assert.False(pub1.InTree);
        Assert.False(a2.InTree);
        Assert.Equal(2, t.NodeCount);
        Tk.Valid(t);
    }

    [Fact]
    public void EffectiveCaps_EnforceCaps()
    {
        long spare = 1000;
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None, subCap: new ResourceVector { DeviceKv = 700 }),
                               spare: c => c == ResourceClass.DeviceKv ? spare : -1, poolPagesCap: 10);
        int s = Tk.Scope(t);
        Assert.Equal(500, t.OptionCaps.DeviceKv);    // auto: half of the spare at construction
        Assert.Equal(500, t.EffectiveCap(ResourceClass.DeviceKv));
        Assert.Equal(long.MaxValue, t.EffectiveCap(ResourceClass.HostKv));
        Assert.Equal(10, t.EffectiveCap(ResourceClass.PoolPages));
        for (int i = 0; i < 6; i++)
            Tk.Put(t, Tk.Key(t, Tk.Seq(100 * (i + 1), 20)), 10, s, host: 0, payload: Tk.Holder(t, host: 0, device: 100));
        spare = 50;
        t.SetRunningReserve(new ResourceVector { DeviceKv = 30 });
        Assert.Equal(500, t.EffectiveCap(ResourceClass.DeviceKv));
        spare = 0;
        Assert.Equal(500, t.EffectiveCap(ResourceClass.DeviceKv));
        Assert.True(t.EnforceCaps(EvictionTier.ScopeNewest));
        Assert.True(t.Cached.DeviceKv <= t.EffectiveCap(ResourceClass.DeviceKv));
        Tk.Valid(t, new InvariantCheckContext(AfterEvictionTrigger: true));
        // Spare below the reserve clamps the cap to what is cached.
        spare = 10;
        t.SetRunningReserve(new ResourceVector { DeviceKv = 30 });
        Assert.Equal(500, t.EffectiveCap(ResourceClass.DeviceKv));
        // A locked payload cannot be evicted: the trigger reports failure.
        PrefixTree locked = Tk.Tree(Tk.Caps(pages: PageSupport.None), optionCap: new ResourceVector { HostKv = 10 });
        RadixNode big = Tk.Put(locked, Tk.Key(locked, Tk.Seq(1, 20)), 10, Tk.Scope(locked), host: 100);
        LockReceipt hold = locked.AcquireState(big);
        Assert.False(locked.EnforceCaps(EvictionTier.PublicTop));
        locked.Release(ref hold);
        Assert.True(locked.EnforceCaps(EvictionTier.PublicTop));
        // Option caps from RAM for HostKv.
        PrefixTree ram = new(new PrefixTreeOptions { Capabilities = Tk.Caps(), BlockSize = B, HostRamBytes = 4000 });
        Assert.Equal(1000, ram.OptionCaps.HostKv);
    }
}
