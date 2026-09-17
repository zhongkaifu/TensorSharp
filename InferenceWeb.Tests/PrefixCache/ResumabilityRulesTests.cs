// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class ResumabilityRulesTests
{
    private static ResumabilityRules Rules(PrefixCacheCapabilities caps, bool batched = true) => new(caps, Tk.B, batched);

    [Fact]
    public void Validate_RejectsUnusableRecords()
    {
        Assert.Throws<ArgumentNullException>(() => ResumabilityRules.Validate(null!, 8));
        Assert.Throws<ArgumentException>(() => Rules(Tk.Caps() with { NamespaceFingerprint = "" }));
        Assert.Throws<ArgumentOutOfRangeException>(() => ResumabilityRules.Validate(Tk.Caps(), 0));
        Assert.Throws<ArgumentException>(() => Rules(Tk.Caps(granularity: 0)));
        Assert.Throws<ArgumentException>(() => Rules(Tk.Caps(rewindCap: -1)));
        Assert.Throws<ArgumentException>(() => Rules(Tk.Caps(minRetain: -1)));
        Assert.Throws<ArgumentException>(() => Rules(Tk.Caps(truncation: TruncationKind.WithinUnwrappedWindow)));
        Assert.Throws<ArgumentException>(() => Rules(Tk.Caps(truncation: TruncationKind.WithinRingSlack)));
        Assert.Throws<ArgumentException>(() => Rules(Tk.Caps(subCap: new ResourceVector { HostKv = -1 })));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PrefixTree(new PrefixTreeOptions { Capabilities = Tk.Caps(), PublicMax = -1 }));
        Assert.Throws<ArgumentNullException>(() => new PrefixTree(null!));
        var r = Rules(Tk.Caps());
        Assert.Equal(Tk.B, r.BlockSize);
        Assert.True(r.BatchedPagedEnabled);
        Assert.Equal("test-fp", r.Caps.NamespaceFingerprint);
    }

    [Theory]
    [InlineData((int)TruncationKind.None, 0, 50, 40, false)]
    [InlineData((int)TruncationKind.Any, 0, 50, 40, true)]
    [InlineData((int)TruncationKind.Any, 0, 50, 50, true)]
    [InlineData((int)TruncationKind.Any, 0, 50, 51, false)]
    [InlineData((int)TruncationKind.Any, 0, 50, -1, false)]
    [InlineData((int)TruncationKind.WithinUnwrappedWindow, 64, 64, 10, true)]
    [InlineData((int)TruncationKind.WithinUnwrappedWindow, 64, 65, 60, false)]
    [InlineData((int)TruncationKind.WithinRingSlack, 5, 100, 95, true)]
    [InlineData((int)TruncationKind.WithinRingSlack, 5, 100, 94, false)]
    [InlineData((int)TruncationKind.ModelDecides, 0, 1000, 2, true)]
    [InlineData(99, 0, 10, 5, false)]
    public void TruncationAllows_EveryKind(int kind, int parameter, int cached, int target, bool expected)
    {
        var caps = Tk.Caps() with { Truncation = (TruncationKind)kind, TruncationParameter = parameter };
        // kind 99: an out-of-range value never allows truncation.
        Assert.Equal(expected, Rules(caps).TruncationAllows(cached, target));
    }

    [Fact]
    public void RewindCap_AndGranularityAlignment()
    {
        var any = Rules(Tk.Caps(truncation: TruncationKind.Any, rewindCap: 16));
        Assert.True(any.RewindWithinCap(40, 24));
        Assert.False(any.RewindWithinCap(40, 23));
        var model = Rules(Tk.Caps(truncation: TruncationKind.ModelDecides, rewindCap: 16));
        Assert.True(model.RewindWithinCap(4000, 2));
        Assert.Equal(34, ResumabilityRules.AlignDown(35, 2));
        Assert.Equal(35, ResumabilityRules.AlignDown(35, 1));
        Assert.Equal(32, ResumabilityRules.AlignDown(39, 8));
        Assert.Equal(0, ResumabilityRules.AlignDown(7, 8));
    }

    [Fact]
    public void RouteMatrix()
    {
        var both = Rules(Tk.Caps(pages: PageSupport.Both, copyPagedToHolder: true));
        foreach (ExpectedRoute route in Enum.GetValues<ExpectedRoute>())
        {
            Assert.True(both.RouteCanRead(PageStore.A1HostSlab, MaterializeMode.InjectA1Pages, route));
            Assert.True(both.RouteCanRead(PageStore.Both, MaterializeMode.InjectA1Pages, route));
            Assert.False(both.RouteCanRead(PageStore.A2ModelPaged, MaterializeMode.InjectA1Pages, route));
            Assert.True(both.RouteCanRead(PageStore.A2ModelPaged, MaterializeMode.BindPagesInPlace, route));
            Assert.False(both.RouteCanRead(PageStore.A1HostSlab, MaterializeMode.BindPagesInPlace, route));
            Assert.Equal(route != ExpectedRoute.BatchedPaged, both.RouteCanRead(PageStore.A2ModelPaged, MaterializeMode.CopyA2PagesToHolder, route));
            Assert.False(both.RouteCanRead(PageStore.A1HostSlab, MaterializeMode.CopyA2PagesToHolder, route));
            Assert.False(both.RouteCanRead(PageStore.Both, MaterializeMode.CloneEndState, route));
        }
        var noBatched = Rules(Tk.Caps(pages: PageSupport.Both), batched: false);
        Assert.False(noBatched.RouteCanRead(PageStore.A2ModelPaged, MaterializeMode.BindPagesInPlace, ExpectedRoute.BatchedPaged));
        var a1Only = Rules(Tk.Caps(pages: PageSupport.A1HostSlab));
        Assert.False(a1Only.RouteCanRead(PageStore.Both, MaterializeMode.BindPagesInPlace, ExpectedRoute.BatchedPaged));
        Assert.True(a1Only.RouteCanRead(PageStore.Both, MaterializeMode.InjectA1Pages, ExpectedRoute.BatchedPaged));
        var a2Only = Rules(Tk.Caps(pages: PageSupport.A2ModelPaged));
        Assert.False(a2Only.RouteCanRead(PageStore.Both, MaterializeMode.InjectA1Pages, ExpectedRoute.Primary));
        Assert.True(a2Only.RouteCanRead(PageStore.Both, MaterializeMode.BindPagesInPlace, ExpectedRoute.BatchedPaged));
        var none = Rules(Tk.Caps(pages: PageSupport.None));
        Assert.False(none.RouteCanRead(PageStore.Both, MaterializeMode.InjectA1Pages, ExpectedRoute.Primary));

        // Natural-mode preferences per route.
        Assert.Equal(0, ResumabilityRules.PageModePreference(MaterializeMode.BindPagesInPlace, ExpectedRoute.BatchedPaged));
        Assert.Equal(1, ResumabilityRules.PageModePreference(MaterializeMode.InjectA1Pages, ExpectedRoute.BatchedPaged));
        Assert.Equal(2, ResumabilityRules.PageModePreference(MaterializeMode.CopyA2PagesToHolder, ExpectedRoute.BatchedPaged));
        Assert.Equal(0, ResumabilityRules.PageModePreference(MaterializeMode.InjectA1Pages, ExpectedRoute.Primary));
        Assert.Equal(1, ResumabilityRules.PageModePreference(MaterializeMode.CopyA2PagesToHolder, ExpectedRoute.PerSequenceFused));
        Assert.Equal(2, ResumabilityRules.PageModePreference(MaterializeMode.BindPagesInPlace, ExpectedRoute.Primary));
        Assert.True(ResumabilityRules.IsClone(MaterializeMode.CloneEndState));
        Assert.True(ResumabilityRules.IsClone(MaterializeMode.ConvertPrimaryThenClone));
        Assert.False(ResumabilityRules.IsClone(MaterializeMode.DonateEndState));
    }

    [Fact]
    public void ClampAligned_NeverLandsInsideASpan()
    {
        // Found by the tree trace harness: AlignDown(Clamp(37), 2) = 36 lies inside [34, 37).
        var pool = new KeyChunkPool();
        KeyRope key = KeyRope.FromKeys(new long[100], pool);
        var spans = new[] { Tk.Span(34, 37, "img"), Tk.Span(40, 43, "img2") };
        var rules = Rules(Tk.Caps(truncation: TruncationKind.ModelDecides, granularity: 2));
        var r = new MatchRequest(key, 100, 1, 0, 99, spans, ExpectedRoute.Primary, true);
        Assert.Equal(36, ResumabilityRules.AlignDown(rules.ClampLength(37, r), 2));   // the single pass is wrong
        ClampReasons reasons = ClampReasons.None;
        Assert.Equal(34, rules.ClampAligned(37, r, 2, ref reasons));
        Assert.True((reasons & ClampReasons.Granularity) != 0);
        Assert.True((reasons & ClampReasons.Media) != 0);
        reasons = ClampReasons.None;
        Assert.Equal(38, rules.ClampAligned(39, r, 2, ref reasons));
        Assert.Equal(40, rules.ClampAligned(42, r, 2, ref reasons));   // inside [40, 43) → its start, already aligned
        Assert.Equal(32, rules.ClampAligned(39, r, 4, ref reasons));   // 39 → 36, which lies inside [34, 37) → 34 → 32
        Assert.Equal(32, rules.ClampAligned(36, r, 4, ref reasons));
    }

    [Fact]
    public void Permitted_AndClamp()
    {
        Assert.True(ResumabilityRules.Permitted(0, 16, 3, 16));
        Assert.False(ResumabilityRules.Permitted(0, 17, 3, 16));
        Assert.True(ResumabilityRules.Permitted(3, 1000, 3, 16));
        Assert.False(ResumabilityRules.Permitted(4, 1, 3, 16));

        var pool = new KeyChunkPool();
        KeyRope key = KeyRope.FromKeys(new long[100], pool);
        var spans = new[] { Tk.Span(10, 20, "a"), Tk.Span(30, 40, "b") };
        var rules = Rules(Tk.Caps(reuseAcrossMedia: true));
        var r = new MatchRequest(key, 100, 1, 0, 99, spans, ExpectedRoute.Primary, true);
        ClampReasons reasons = ClampReasons.None;
        Assert.Equal(10, rules.ClampLength(15, r, ref reasons));
        Assert.Equal(ClampReasons.Media, reasons);
        Assert.Equal(20, rules.ClampLength(20, r));
        Assert.Equal(25, rules.ClampLength(25, r));
        Assert.Equal(30, rules.ClampLength(35, r));
        reasons = ClampReasons.None;
        Assert.Equal(99, rules.ClampLength(150, r, ref reasons));
        Assert.Equal(ClampReasons.LeaveOne, reasons);
        var bp = r with { MatchLimit = 50 };
        reasons = ClampReasons.None;
        Assert.Equal(50, rules.ClampLength(60, bp, ref reasons));
        Assert.Equal(ClampReasons.Breakpoint, reasons);
        var noMedia = Rules(Tk.Caps(reuseAcrossMedia: false));
        reasons = ClampReasons.None;
        Assert.Equal(10, noMedia.ClampLength(25, r, ref reasons));
        Assert.Equal(ClampReasons.MediaAcrossSpan, reasons);
        Assert.Equal(10, noMedia.ClampLength(10, r));
        Assert.Equal(0, rules.ClampLength(-3, r));
        Assert.Equal(0, ResumabilityRules.EffectiveMatchLimit(r with { KeyLength = 0 }));

        var window = Rules(Tk.Caps(pageWindow: 24));
        Assert.True(window.PageInWindow(24));
        Assert.False(window.PageInWindow(32));
        Assert.True(Rules(Tk.Caps()).PageInWindow(int.MaxValue));
        var nemotron = Rules(Tk.Caps(needStateAtEnd: true));
        Assert.False(nemotron.PageEndResumable(new PageRef(null!, 0, PageStore.A1HostSlab, false)));
        Assert.True(nemotron.PageEndResumable(new PageRef(null!, 0, PageStore.A1HostSlab, true)));
        Assert.True(Rules(Tk.Caps()).PageEndResumable(new PageRef(null!, 0, PageStore.A1HostSlab, false)));
    }
}
