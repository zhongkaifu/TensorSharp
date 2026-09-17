// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class PrefixTreeAccountingTests
{
    private const int B = Tk.B;

    [Fact]
    public void ResourceVector_Arithmetic()
    {
        var a = new ResourceVector { PoolPages = 1, HostKv = 2, DeviceKv = 3, StateSnapshot = 4, NativeSlot = 5 };
        ResourceVector b = a + a;
        Assert.Equal(10, b.NativeSlot);
        Assert.Equal(a, b - a);
        Assert.True((a - b).AnyNegative);
        Assert.False(a.AnyNegative);
        Assert.Equal(14, a.TotalBytes);
        Assert.True(default(ResourceVector).IsZero);
        Assert.True(ResourceVector.Negate(a).AnyNegative);
        for (int i = 0; i < ResourceVector.ClassCount; i++)
        {
            var c = (ResourceClass)i;
            var v = default(ResourceVector);
            v[c] = 7;
            Assert.Equal(7, v[c]);
            Assert.Equal(7, v.PoolPages + v.HostKv + v.DeviceKv + v.StateSnapshot + v.NativeSlot);
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => a[(ResourceClass)9]);
        Assert.Throws<ArgumentOutOfRangeException>(() => { var x = a; x[(ResourceClass)9] = 1; });
        Assert.True(a == (b - a));
        Assert.True(a != b);
        Assert.True(a.Equals((object)(b - a)));
        Assert.Equal(a.GetHashCode(), (b - a).GetHashCode());
        Assert.Contains("native=5", a.ToString());
    }

    [Fact]
    public void EveryZeroOneTransition_MovesProtectedBytes_PerClass()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host, pageHostBytes: 64);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 100));
        RadixNode n = t.Insert(key, 3 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(n, Tk.Pages(host, 3, PageStore.Both));
        EndStatePayload payload = Tk.Holder(t, host: 100, device: 200, state: 300, native: 400);
        t.AttachEndState(n, payload);
        var cached = new ResourceVector { PoolPages = 3, HostKv = 3 * 64 + 100, DeviceKv = 200, StateSnapshot = 300, NativeSlot = 400 };
        Assert.Equal(cached, t.Cached);
        Assert.True(t.Protected.IsZero);
        Assert.Equal(cached, t.Evictable);
        Assert.Equal(cached, t.Scopes[s].Bytes);

        // Path lock 0→1: pages only.
        LockReceipt path = t.AcquirePath(n);
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, t.Protected);
        LockReceipt path2 = t.AcquirePath(n);                        // 1→2: no change
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, t.Protected);
        t.Release(ref path2);
        // State lock 0→1: the end state.
        LockReceipt state = t.AcquireState(n);
        Assert.Equal(cached, t.Protected);
        // Pin on top: no double count.
        t.Pin(n);
        Assert.Equal(cached, t.Protected);
        t.Release(ref state);
        Assert.Equal(cached, t.Protected);                           // still pinned
        t.Unpin(n);
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, t.Protected);
        // DonationPending protects the end state.
        t.MarkDonationPending(n);
        Assert.Equal(cached, t.Protected);
        t.CancelDonation(n);
        t.Release(ref path);
        Assert.True(t.Protected.IsZero);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));

        // Commit of a donation: bytes leave Cached and Protected, no release queued.
        t.MarkDonationPending(n);
        t.CommitDonation(n);
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, t.Cached);
        Assert.True(t.PendingReclaim.IsZero);
        Assert.True(t.Protected.IsZero);
        Tk.Valid(t);
    }

    [Fact]
    public void PendingReclaim_HoldsEvictedBytesUntilTheDrain()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t), other = Tk.Scope(t);
        Tk.Put(t, Tk.Key(t, Tk.Seq(1, 30)), 20, s, payload: Tk.Holder(t, host: 10, device: 20));
        Tk.Put(t, Tk.Key(t, Tk.Seq(100, 30)), 20, other, payload: Tk.Holder(t, host: 1, device: 2));
        Assert.True(t.Evict(ResourceClass.DeviceKv, 20, ReleaseReason.Evicted, EvictionTier.ScopeNewest));
        Assert.Equal(new ResourceVector { HostKv = 10, DeviceKv = 20 }, t.PendingReclaim);
        Assert.Equal(new ResourceVector { HostKv = 1, DeviceKv = 2 }, t.Cached);
        t.DrainReclaimQueue(null);
        Assert.True(t.PendingReclaim.IsZero);
        Tk.Valid(t);
    }

    [Fact]
    public void EffectiveCap_ClampsToSpareWithAFakeSource()
    {
        long spare = -1;
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None, subCap: new ResourceVector { StateSnapshot = 5000 }),
                               spare: _ => spare, optionCap: new ResourceVector { StateSnapshot = 3000 });
        int s = Tk.Scope(t);
        Assert.Equal(3000, t.EffectiveCap(ResourceClass.StateSnapshot));     // unknown spare: option cap
        Tk.Put(t, Tk.Key(t, Tk.Seq(1, 30)), 20, s, payload: Tk.Holder(t, host: 0, state: 1000));
        spare = 500;
        Assert.Equal(1500, t.EffectiveCap(ResourceClass.StateSnapshot));     // cached + spare
        t.SetRunningReserve(new ResourceVector { StateSnapshot = 200 });
        Assert.Equal(1300, t.EffectiveCap(ResourceClass.StateSnapshot));
        Assert.Equal(new ResourceVector { StateSnapshot = 200 }, t.RunningReserve);
        spare = 100;
        Assert.Equal(1000, t.EffectiveCap(ResourceClass.StateSnapshot));     // reserve larger than spare: no headroom
        PrefixTree sub = Tk.Tree(Tk.Caps(pages: PageSupport.None, subCap: new ResourceVector { NativeSlot = 700 }));
        Assert.Equal(700, sub.EffectiveCap(ResourceClass.NativeSlot));
    }
}
