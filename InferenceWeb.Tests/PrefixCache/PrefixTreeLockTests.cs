// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class PrefixTreeLockTests
{
    private const int B = Tk.B;

    [Fact]
    public void Acquire_SplitsAtThePlanLength_AndLocksPathAndState()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode x = Tk.Put(t, key, 50, s);
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Cat(Tk.Seq(1, 40), Tk.Seq(900, 3))), s));
        Assert.Equal(CandidateKind.TruncatedEndState, plan.Kind);
        long version = t.Version;
        int nodes = t.NodeCount;
        LockReceipt r = t.Acquire(plan);
        Assert.True(t.Version > version);
        Assert.Equal(nodes + 1, t.NodeCount);          // split at 40
        Assert.Equal(40, r.PathDepth);
        Assert.Same(x, r.StateAnchor);
        Assert.Equal(1, r.PathAnchor!.LockRef);
        Assert.Equal(0, x.LockRef);                    // the payload below the anchor is state-locked only
        Assert.Equal(1, x.StateLockRef);
        Assert.True(x.LastAccess > r.PathAnchor.LastAccess);
        Tk.Valid(t);
        // A stale plan throws and is counted.
        Assert.Throws<StalePrefixMatchException>(() => t.Acquire(plan));
        Assert.Equal(1, t.Counters.StaleMatches);
        t.Release(ref r);
        Assert.True(r.IsEmpty);
        Assert.Equal(0, t.Protected.HostKv);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));
        // An empty plan acquires nothing.
        MatchPlan none = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(700, 10)), s));
        Assert.True(t.Acquire(none).IsEmpty);
    }

    [Fact]
    public void Receipt_ReplaysThroughNodesCreatedAfterTheLock()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode x = Tk.Put(t, key, 60, s);
        LockReceipt durable = t.AcquirePath(x);
        // Splits above the anchor after the lock: new parents copy LockRef.
        RadixNode p1 = t.SplitAt(x, 30);
        RadixNode p2 = t.SplitAt(p1, 10);
        Assert.Equal(1, p1.LockRef);
        Assert.Equal(1, p2.LockRef);
        Tk.Valid(t);
        t.Release(ref durable);
        Assert.Equal(0, p1.LockRef);
        Assert.Equal(0, p2.LockRef);
        Assert.Equal(0, x.LockRef);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));
    }

    [Fact]
    public void StaleEmptyAndCrossTreeReceipts()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None), strictReceipts: false);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode x = Tk.Put(t, key, 40, s);
        // Empty receipt: no-op.
        LockReceipt empty = default;
        t.Release(ref empty);
        Assert.Equal(0, t.Counters.StaleReceipts);

        // Cross-tree receipt: ignored and counted.
        PrefixTree other = Tk.Tree(Tk.Caps(pages: PageSupport.None), strictReceipts: false);
        int so = Tk.Scope(other);
        RadixNode ox = Tk.Put(other, Tk.Key(other, Tk.Seq(1, 80)), 40, so);
        LockReceipt foreign = other.AcquirePath(ox);
        other.Reset();                         // bumps the other tree's serial
        t.Release(ref foreign);
        Assert.Equal(1, t.Counters.StaleReceipts);
        Assert.True(foreign.IsEmpty);

        // A copy released twice: the ledger catches the second release.
        LockReceipt r = t.AcquirePath(x);
        LockReceipt copy = r;
        t.Release(ref r);
        t.Release(ref copy);
        Assert.Equal(2, t.Counters.StaleReceipts);
        Assert.Equal(0, x.LockRef);

        // Stale generation: the anchor was deleted and its object recycled.
        LockReceipt stateLock = t.AcquireState(x);
        LockReceipt old = stateLock;
        t.Release(ref stateLock);
        t.DeleteLeafCascade(x, ReleaseReason.Evicted);
        RadixNode recycled = Tk.Put(t, Tk.Key(t, Tk.Seq(300, 30)), 20, s);
        t.Release(ref old);
        Assert.Equal(3, t.Counters.StaleReceipts);
        Assert.Equal(0, recycled.StateLockRef);
        Tk.Valid(t);

        // Strict mode throws.
        PrefixTree strict = Tk.Tree(Tk.Caps(pages: PageSupport.None), strictReceipts: true);
        int ss = Tk.Scope(strict);
        RadixNode sx = Tk.Put(strict, Tk.Key(strict, Tk.Seq(1, 40)), 20, ss);
        LockReceipt sr = strict.AcquirePath(sx);
        LockReceipt scopy = sr;
        strict.Release(ref sr);
        Assert.Throws<InvalidOperationException>(() => strict.Release(ref scopy));
    }

    [Fact]
    public void StateLockProtectsTheEndState_PathLockProtectsPages()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host, pageHostBytes: 100);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode x = t.Insert(key, 4 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(x, Tk.Pages(host, 4));
        t.AttachEndState(x, Tk.Holder(t, host: 1000));
        Assert.True(t.Protected.IsZero);

        LockReceipt path = t.AcquirePath(x);
        Assert.Equal(4, t.Protected.PoolPages);
        Assert.Equal(400, t.Protected.HostKv);            // pages only: a path lock never protects the end state
        Assert.Equal(EvictionLists.ListId(LruKind.State, t.TierOf(x)), x.LruList);   // the end state stays detachable
        LockReceipt state = t.AcquireState(x);
        Assert.Equal(1400, t.Protected.HostKv);
        Assert.Equal(0, x.LruList);
        Tk.Valid(t);
        t.ReleaseState(ref state);
        Assert.True(state.IsEmpty);
        Assert.Equal(400, t.Protected.HostKv);
        t.Release(ref path);
        Assert.True(t.Protected.IsZero);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));

        // ReleaseState on a combined receipt keeps the path part.
        MatchPlan plan = Tk.Plan(t, Tk.Req(Tk.Key(t, Tk.Seq(1, 40)), s));
        LockReceipt both = t.Acquire(plan);
        Assert.NotNull(both.StateAnchor);
        t.ReleaseState(ref both);
        Assert.Null(both.StateAnchor);
        Assert.NotNull(both.PathAnchor);
        Assert.Equal(0, x.StateLockRef);
        Tk.Valid(t);
        t.ReleaseState(ref both);                         // nothing state-locked: no-op
        t.Release(ref both);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));
    }

    [Fact]
    public void MoveDurableLock_AcquiresTheNewPathBeforeReleasingTheOld()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode first = t.Insert(key, 2 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(first, Tk.Pages(host, 2));
        LockReceipt durable = t.AcquirePath(first);
        // The request publishes two more pages: a deeper, payload-less-until-attached leaf.
        RadixNode deeper = t.Insert(key, 4 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(deeper, Tk.Pages(host, 2, first: 2));
        t.MoveDurableLock(ref durable, deeper);
        // If the old lock had been released first, `first`'s pages would have been unprotected in between;
        // here they never were: first stays locked through the new path.
        Assert.Same(deeper, durable.PathAnchor);
        Assert.Equal(1, first.LockRef);
        Assert.Equal(1, deeper.LockRef);
        Assert.Equal(4, t.Protected.PoolPages);
        Tk.Valid(t);
        t.Release(ref durable);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));
        // Locking the root is a no-op.
        Assert.True(t.AcquirePath(t.Root).IsEmpty);
        Assert.True(t.AcquireState(t.Root).IsEmpty);
    }

    [Fact]
    public void LastUnlockOfAPayloadlessLeaf_DeletesIt()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 80));
        RadixNode n = t.Insert(key, 30, s, 10, NodeFlags.None, null);
        LockReceipt r = t.AcquirePath(n);
        Tk.Valid(t);                                  // a payload-less leaf may exist while locked
        t.Release(ref r);
        Assert.False(n.InTree);
        Assert.Equal(0, t.NodeCount);                 // the public parent cascaded too
        Tk.Valid(t);
    }
}
