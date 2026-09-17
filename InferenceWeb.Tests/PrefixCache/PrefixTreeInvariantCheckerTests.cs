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

/// <summary>
/// §12.2 (G-20): build a valid tree, apply one corruption per tree-level invariant and assert the
/// checker reports THAT invariant's id. A missed corruption fails the build.
/// </summary>
public class PrefixTreeInvariantCheckerTests
{
    private const int B = Tk.B;

    /// <summary>A small tree exercising every structure the checker reads.</summary>
    private sealed class World
    {
        public PrefixTree T = null!;
        public FakePageHost Host = null!;
        public int ScopeA, ScopeB;
        public RadixNode PublicBoundary = null!;   // [0,16) public, end state
        public RadixNode PublicBoundary2 = null!;  // another public boundary
        public RadixNode PublicBoundary3 = null!;
        public RadixNode ScopedInternal = null!;   // [16,40) scope A, end state, has a child
        public RadixNode ScopedLeaf = null!;       // [40,70) scope A, end state
        public RadixNode PagesLeaf = null!;        // scope B leaf with pages
        public RadixNode MediaLeaf = null!;        // scope A leaf whose edge holds a media span
        public RadixNode PrimaryLeaf = null!;      // scope B PrimaryResident
        public LockReceipt Durable;                // path lock on ScopedLeaf
    }

    private static World Build(int contextLength = int.MaxValue, ResourceVector optionCap = default)
    {
        var w = new World { Host = new FakePageHost(B) };
        w.T = Tk.Tree(Tk.Caps(), host: w.Host, publicMax: 3, contextLength: contextLength, optionCap: optionCap);
        PrefixTree t = w.T;
        w.ScopeA = Tk.Scope(t);
        w.ScopeB = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 100));
        w.PublicBoundary = Tk.Put(t, key, 16, 0, p: 16, host: 10);
        w.ScopedInternal = Tk.Put(t, key, 40, w.ScopeA, p: 16, host: 20);
        w.ScopedLeaf = Tk.Put(t, key, 70, w.ScopeA, p: 16, host: 30);
        w.PublicBoundary2 = Tk.Put(t, Tk.Key(t, Tk.Seq(200, 30)), 12, 0, p: 12, host: 5);
        w.PublicBoundary3 = Tk.Put(t, Tk.Key(t, Tk.Seq(300, 30)), 12, 0, p: 12, host: 5);
        KeyRope pagesKey = Tk.Key(t, Tk.Seq(400, 60));
        w.PagesLeaf = t.Insert(pagesKey, 5 * B, w.ScopeB, 0, NodeFlags.None, null);
        t.AttachPages(w.PagesLeaf, new[]
        {
            new PageRef(w.Host.NewBlock(PageStore.A1HostSlab), 0, PageStore.A1HostSlab, true),
            new PageRef(w.Host.NewBlock(PageStore.A2ModelPaged), 1, PageStore.A2ModelPaged, true),
            new PageRef(w.Host.NewBlock(PageStore.Both), 2, PageStore.Both, true),
            new PageRef(w.Host.NewBlock(PageStore.Both), 3, PageStore.Both, true),
            new PageRef(w.Host.NewBlock(PageStore.Both), 4, PageStore.Both, true),
        });
        var spans = new[] { Tk.Span(20, 30, new string('7', 64)) };
        int[] mediaTokens = Tk.WithPlaceholders(50, spans);
        for (int i = 0; i < mediaTokens.Length; i++) if (mediaTokens[i] != 999) mediaTokens[i] += 1000;
        w.MediaLeaf = Tk.Put(t, Tk.Key(t, mediaTokens, spans), 40, w.ScopeA, spans: spans, host: 7);
        w.PrimaryLeaf = Tk.Put(t, Tk.Key(t, Tk.Seq(600, 40)), 30, w.ScopeB, payload: Tk.Primary(t));
        w.Durable = t.AcquirePath(w.ScopedLeaf);
        Tk.Valid(t);
        return w;
    }

    private static void AssertCaught(PrefixTree t, string id, InvariantCheckContext ctx = default, PrefixTreeInvariantChecker? checker = null)
    {
        IReadOnlyList<InvariantViolation> v = (checker ?? new PrefixTreeInvariantChecker()).Check(t, ctx);
        Assert.True(v.Any(x => x.Id == id), $"corruption for {id} was not caught; reported: [{string.Join("; ", v)}]");
    }

    [Fact]
    public void ValidWorld_HasNoViolations()
    {
        World w = Build();
        Tk.Valid(w.T);
        w.T.Release(ref w.Durable);
        Tk.Valid(w.T, new InvariantCheckContext(Quiescent: true, AfterEvictionTrigger: true));
        Assert.Equal("I7: x", new InvariantViolation("I7", "x").ToString());
    }

    [Fact] public void I1_WrongDepth() { World w = Build(); w.ScopedLeaf.Depth++; AssertCaught(w.T, "I1"); }

    [Fact]
    public void I1_EdgeOutsideRopeAndNodeCount()
    {
        World w = Build();
        w.ScopedLeaf.Edge = new KeySlice(w.ScopedLeaf.Edge.Rope, w.ScopedLeaf.Edge.Rope.Length - 1, 5);
        AssertCaught(w.T, "I1");
        World w2 = Build();
        w2.T.Root.EndState = Tk.Holder(w2.T);
        AssertCaught(w2.T, "I1");
    }

    [Fact]
    public void I2_TwoChildrenShareAKey()
    {
        World w = Build();
        RadixNode a = w.PublicBoundary, b = w.PublicBoundary2;
        b.Edge = new KeySlice(a.Edge.Rope, a.Edge.Start, b.Edge.Length);   // same first element as a
        AssertCaught(w.T, "I2");
    }

    [Fact] public void I3_ScopedChildOfAnotherScope() { World w = Build(); w.ScopedLeaf.ScopeIx = w.ScopeB; AssertCaught(w.T, "I3"); }

    [Fact] public void I4_ScopedPublicBoundary() { World w = Build(); w.ScopedInternal.Flags |= NodeFlags.IsPublicBoundary; AssertCaught(w.T, "I4"); }

    [Fact]
    public void I5_UnlockedPayloadlessLeaf()
    {
        World w = Build();
        w.T.Insert(Tk.Key(w.T, Tk.Seq(900, 30)), 20, w.ScopeA, 0, NodeFlags.None, null);
        AssertCaught(w.T, "I5");
    }

    [Fact] public void I6_LockRefWithoutAReceipt() { World w = Build(); w.ScopedLeaf.LockRef++; w.ScopedInternal.LockRef++; AssertCaught(w.T, "I6"); }

    [Fact] public void I6_StateLockWithoutAReceipt() { World w = Build(); w.PublicBoundary2.StateLockRef++; AssertCaught(w.T, "I6"); }

    [Fact] public void I7_ChildLockedMoreThanParent() { World w = Build(); w.ScopedLeaf.LockRef = 5; AssertCaught(w.T, "I7"); }

    [Fact] public void I8_NodeBytes() { World w = Build(); w.ScopedLeaf.Bytes.HostKv++; AssertCaught(w.T, "I8"); }

    [Fact] public void I8_CachedCounter() { World w = Build(); w.T.UnsafeSetCached(w.T.Cached + new ResourceVector { DeviceKv = 1 }); AssertCaught(w.T, "I8"); }

    [Fact] public void I8_ProtectedCounter() { World w = Build(); w.T.UnsafeSetProtected(new ResourceVector { HostKv = -1 }); AssertCaught(w.T, "I8"); }

    [Fact] public void I9_UnlinkedLruEntry() { World w = Build(); w.T.Lists.Unlink(w.PublicBoundary2); AssertCaught(w.T, "I9"); }

    [Fact] public void I9_DetachedButStillRecorded() { World w = Build(); w.T.Lists.UnsafeDetachKeepId(w.PagesLeaf); AssertCaught(w.T, "I9"); }

    [Fact]
    public void I9_OutOfOrder()
    {
        World w = Build();
        w.PublicBoundary2.LastAccess = long.MaxValue;   // now newer than its successors in the same list
        AssertCaught(w.T, "I9");
    }

    [Fact]
    public void I10_DuplicateKey()
    {
        World w = Build();
        w.ScopedInternal.EndState!.Key = w.PublicBoundary2.EndState!.Key;
        AssertCaught(w.T, "I10");
    }

    [Fact] public void I10_NotTreeMinted() { World w = Build(); w.PublicBoundary3.EndState!.Key = "request-7"; AssertCaught(w.T, "I10"); }

    [Fact]
    public void I11_PageOnTheWrongNode()
    {
        World w = Build();
        RadixNode deeper = w.T.Insert(Tk.Key(w.T, Tk.Seq(400, 60)), 7 * B, w.ScopeB, 0, NodeFlags.None, null);
        w.T.AttachPages(deeper, Tk.Pages(w.Host, 1, first: 6));
        PageRef moved = w.PagesLeaf.Pages![4];
        w.PagesLeaf.RemovePageAt(4);
        deeper.AddPage(moved);                    // page 4 ends at 40, inside PagesLeaf, not in `deeper`
        AssertCaught(w.T, "I11");
    }

    [Fact]
    public void I11_BlockOwnedTwice()
    {
        World w = Build();
        PageRef p0 = w.PagesLeaf.Pages![0];
        w.PublicBoundary.AddPage(p0 with { PageIndex = 1 });
        AssertCaught(w.T, "I11");
    }

    [Fact]
    public void I12_A2PageNotInPagedStorage()
    {
        World w = Build();
        w.Host.Unset(w.PagesLeaf.Pages![1].Block, paged: true);
        AssertCaught(w.T, "I12");
    }

    [Fact]
    public void I12_A1PageWithoutASnapshot()
    {
        World w = Build();
        w.Host.Unset(w.PagesLeaf.Pages![0].Block, snapshot: true);
        AssertCaught(w.T, "I12");
    }

    [Fact]
    public void I13_EndStateInsideASpan()
    {
        World w = Build();
        RadixNode inside = w.T.SplitAt(w.MediaLeaf, 25 - w.MediaLeaf.EdgeStartDepth);   // depth 25 ∈ (20, 30)
        inside.EndState = Tk.Holder(w.T, host: 0);
        AssertCaught(w.T, "I13");
    }

    [Fact]
    public void I13_MediaElementWithoutARecord()
    {
        World w = Build();
        w.MediaLeaf.MediaSpanCount = 0;
        AssertCaught(w.T, "I13");
    }

    [Fact]
    public void I14_SecondPrimaryResident()
    {
        World w = Build();
        w.PublicBoundary3.EndState!.Kind = EndStateKind.PrimaryResident;
        w.PublicBoundary3.EndState.Footprint = default;
        AssertCaught(w.T, "I14");
    }

    [Fact] public void I15_ScopeBytes() { World w = Build(); w.T.Scopes[w.ScopeA].Bytes.HostKv += 3; AssertCaught(w.T, "I15"); }

    [Fact]
    public void I15_NodeMissingFromItsScopeList()
    {
        World w = Build();
        ScopeRecord rec = w.T.Scopes[w.ScopeB];
        rec.FirstNode = rec.FirstNode!.ScopeNext;
        AssertCaught(w.T, "I15");
    }

    [Fact]
    public void I15_RetiredScopeWithAnUnlockedNode()
    {
        World w = Build();
        w.T.Scopes[w.ScopeB].Retired = true;
        AssertCaught(w.T, "I15");
    }

    [Fact]
    public void I16_TooManyPublicTop()
    {
        World w = Build();
        RadixNode fourth = Tk.Put(w.T, Tk.Key(w.T, Tk.Seq(850, 20)), 12, 0, p: 12, host: 1);
        Tk.Valid(w.T);
        foreach (RadixNode n in new[] { w.PublicBoundary, w.PublicBoundary2, w.PublicBoundary3, fourth }) n.InPublicTop = true;
        AssertCaught(w.T, "I16");
    }

    [Fact]
    public void I18_DonationPendingAfterTheStep()
    {
        World w = Build();
        w.T.MarkDonationPending(w.PublicBoundary3);
        Tk.Valid(w.T, new InvariantCheckContext(TransactionOpen: true));
        AssertCaught(w.T, "I18");
    }

    [Fact]
    public void I19_MutationWithoutAVersionBump()
    {
        World w = Build();
        var checker = new PrefixTreeInvariantChecker();
        Assert.Empty(checker.Check(w.T));
        w.PublicBoundary3.Flags |= NodeFlags.EndsAtBreakpoint;
        w.T.Lists.Unlink(w.PublicBoundary3);
        w.T.Lists.Link(w.PublicBoundary3, w.T.DesiredList(w.PublicBoundary3));
        AssertCaught(w.T, "I19", checker: checker);
        // A decreasing version is caught too.
        var c2 = new PrefixTreeInvariantChecker();
        c2.Check(w.T);
        w.T.UnsafeSetVersion(w.T.Version - 1);
        AssertCaught(w.T, "I19", checker: c2);
    }

    [Fact] public void I20_PersistedScopedEndState() { World w = Build(); w.ScopedLeaf.EndState!.Persisted = true; AssertCaught(w.T, "I20"); }

    [Fact]
    public void I21_CapExceededAfterATrigger()
    {
        World w = Build(optionCap: new ResourceVector { HostKv = 50 });
        Tk.Valid(w.T);
        AssertCaught(w.T, "I21", new InvariantCheckContext(AfterEvictionTrigger: true));
    }

    [Fact]
    public void I21_CountSubCapExceeded()
    {
        World w = Build();
        Tk.Put(w.T, Tk.Key(w.T, Tk.Seq(800, 20)), 12, 0, p: 12, host: 1);   // 4 public end states > PublicMax 3
        AssertCaught(w.T, "I21", new InvariantCheckContext(AfterEvictionTrigger: true));
    }

    [Fact] public void I22_LockAtQuiescence() { World w = Build(); AssertCaught(w.T, "I22", new InvariantCheckContext(Quiescent: true)); }

    [Fact]
    public void I24_DepthBeyondTheContext()
    {
        World w = Build(contextLength: 70);
        w.ScopedLeaf.Depth = 71;
        AssertCaught(w.T, "I24");
    }

    [Fact]
    public void I24_ReceiptOfAnotherEpoch()
    {
        World w = Build();
        w.T.Ledger!.UnsafeAdd(new LockReceipt(w.T.TreeSerial + 3, 99_999, null, null));
        AssertCaught(w.T, "I24");
    }

    [Fact]
    public void I25_QueuedKeyStillOnANode()
    {
        World w = Build();
        w.T.Reclaim.UnsafeEnqueueKeyOnly(w.PublicBoundary2.EndState!.Key);
        AssertCaught(w.T, "I25");
    }

    [Fact] public void I28_UnknownScopeIndex() { World w = Build(); w.PagesLeaf.ScopeIx = 999; AssertCaught(w.T, "I28"); }

    [Fact]
    public void I28_RecycledScopeIndex()
    {
        World w = Build();
        int extra = Tk.Scope(w.T);
        RadixNode n = Tk.Put(w.T, Tk.Key(w.T, Tk.Seq(950, 20)), 10, extra);
        w.T.Scopes[extra].Live = false;
        AssertCaught(w.T, "I28");
        Assert.True(n.InTree);
    }

    [Fact]
    public void I29_SharedUnscopedId()
    {
        World w = Build();
        int u1 = w.T.InternScope(ScopeId.NewFresh(), ScopeKind.Unscoped);
        int u2 = w.T.InternScope(ScopeId.NewFresh(), ScopeKind.Unscoped);
        w.T.Scopes[u2].Id = w.T.Scopes[u1].Id;
        AssertCaught(w.T, "I29");
    }

    [Fact]
    public void EveryTreeLevelInvariant_HasAMutationTest()
    {
        string[] treeLevel = { "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13", "I14", "I15", "I16",
                               "I18", "I19", "I20", "I21", "I22", "I24", "I25", "I28", "I29" };
        var names = typeof(PrefixTreeInvariantCheckerTests).GetMethods().Select(m => m.Name).ToArray();
        foreach (string id in treeLevel)
            Assert.Contains(names, n => n.StartsWith(id + "_", StringComparison.Ordinal));
    }
}
