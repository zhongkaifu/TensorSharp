// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>
/// <see cref="HolderPrefixCacheAdapter"/> against the oracles: an adapter-backed holder fake must pass the
/// same conformance script the oracle's own implementation passes, and the adapter's settle, rewind and
/// import checks must be the ones that make it pass.
/// </summary>
[Trait("Category", "PrefixCacheUnit")]
public sealed class HolderPrefixCacheAdapterTests
{
    private readonly ITestOutputHelper _output;

    public HolderPrefixCacheAdapterTests(ITestOutputHelper output) { _output = output; }

    public static IEnumerable<object[]> HolderFakes => new[] { "P", "S", "R", "N" }.Select(n => new object[] { n });

    private static OracleModel Create(string name) => OracleFakes.All.Single(f => f.Name == name).Create();

    [Theory]
    [MemberData(nameof(HolderFakes))]
    public void AdapterBackedHolderFamily_PassesTheConformanceScript(string name)
    {
        var model = new AdaptedOracleModel(Create(name));
        ConformanceReport report = PrefixCacheConformanceScript.Run(
            PrefixCacheContractConformanceTests.FakeSubject(model, "adapted-" + name, _output.WriteLine));
        Assert.Contains("donate/return/donate", report.Ran);
        Assert.Contains("primary conversion", report.Ran);
    }

    [Theory]
    [InlineData("S")]
    [InlineData("R")]
    public void AdapterWithoutASettle_FailsTheSettleThenCloneStep(string name)
    {
        var model = new AdaptedOracleModel(Create(name), settles: false);
        var failure = Assert.ThrowsAny<Exception>(() => PrefixCacheConformanceScript.Run(
            PrefixCacheContractConformanceTests.FakeSubject(model, "unsettled-" + name, _output.WriteLine)));
        Assert.Contains("settle", failure.Message);
    }

    [Fact]
    public void Donation_ShorterThanTheHolder_RewindsWhereTheFamilyCanAndContinuesExactly()
    {
        var model = new AdaptedOracleModel(OracleFakes.P());
        IPrefixCacheModel pcm = model;
        int[] prompt = Enumerable.Range(0, 30).Select(i => (i * 7 + 3) % OracleModel.DefaultVocab).ToArray();
        int[] other = Enumerable.Range(0, 6).Select(i => (i * 13 + 1) % OracleModel.DefaultVocab).ToArray();
        // Cold: the first 24 prompt tokens, then a different continuation.
        float[] cold = OracleFakes.P().Forward(prompt.Take(24).Concat(other).ToArray());

        Assert.True(model.BindSequenceCache("finished"));
        model.Forward(prompt);
        Assert.True(pcm.TryCaptureDonate("finished", "pc:1:1", 24, out PayloadFootprint fp));
        Assert.Equal(24, fp.Tokens);
        Assert.False(model.HasFusedSequenceCache("finished"));
        Assert.True(pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, "pc:1:1", "next", 24, 24)));
        Assert.False(model.BindSequenceCache("next"));
        Assert.Equal(cold, model.Forward(other));
    }

    [Fact]
    public void Donation_ShorterThanTheHolder_IsRefusedForAFamilyWithoutTruncationAndTheHolderStaysTheRequests()
    {
        var model = new AdaptedOracleModel(OracleFakes.R());
        IPrefixCacheModel pcm = model;
        Assert.True(model.BindSequenceCache("finished"));
        model.Forward(Enumerable.Range(1, 20).ToArray());
        Assert.False(pcm.TryCaptureDonate("finished", "pc:1:1", 16, out _));
        Assert.True(model.HasFusedSequenceCache("finished"), "a refused donation leaves the holder with its request, whose release disposes it");
        Assert.Empty(model.RetainedPayloadKeys);
    }

    [Fact]
    public void Donation_OfAHolderThatWrappedIsRefusedWithoutCorruptingIt()
    {
        var model = new AdaptedOracleModel(OracleFakes.S());
        IPrefixCacheModel pcm = model;
        int[] prompt = Enumerable.Range(0, OracleFakes.GemmaWindow + 8).Select(i => (i * 11 + 5) % OracleModel.DefaultVocab).ToArray();
        Assert.True(model.BindSequenceCache("wrapped"));
        model.Forward(prompt);
        ulong state = model.Inner.ActiveState;
        Assert.False(pcm.TryCaptureDonate("wrapped", "pc:1:1", prompt.Length - 4, out _));
        Assert.True(model.HasFusedSequenceCache("wrapped"));
        Assert.Empty(model.RetainedPayloadKeys);
        model.BindSequenceCache("wrapped");
        Assert.Equal(prompt.Length, model.Inner.ActiveLength);
        Assert.Equal(state, model.Inner.ActiveState);
    }

    [Fact]
    public void Import_OfTheWrongLengthIsDiscarded()
    {
        var model = new AdaptedOracleModel(OracleFakes.R());
        IPrefixCacheModel pcm = model;
        Assert.True(model.BindSequenceCache("a"));
        model.Forward(Enumerable.Range(1, 12).ToArray());
        Assert.True(pcm.TryCaptureCopy("a", "pc:1:1", out _));
        var file = new MemoryStream();
        Assert.True(pcm.TryExport("pc:1:1", file));
        Assert.False(pcm.TryImport("pc:1:2", 11, new MemoryStream(file.ToArray()), out _));
        Assert.DoesNotContain("pc:1:2", model.RetainedPayloadKeys);
        Assert.True(pcm.TryImport("pc:1:3", 12, new MemoryStream(file.ToArray()), out PayloadFootprint fp));
        Assert.Equal(12, fp.Tokens);
    }

    [Fact]
    public void ConvertPrimary_RefusesBeforeAdoptingWhenTheLengthDiffersOrAHolderIsCheckedOut()
    {
        var model = new AdaptedOracleModel(OracleFakes.S());
        IPrefixCacheModel pcm = model;
        model.Forward(Enumerable.Range(1, 10).ToArray());
        Assert.False(pcm.TryConvertPrimary("pc:1:1", 9, out _));
        Assert.Equal(10, model.PrimaryCacheLength);

        Assert.True(model.BindSequenceCache("busy"));
        Assert.False(pcm.TryConvertPrimary("pc:1:1", 10, out _));
        Assert.Equal(10, model.PrimaryCacheLength);

        model.RestorePrimaryCache();
        Assert.True(pcm.TryConvertPrimary("pc:1:1", 10, out PayloadFootprint fp));
        Assert.Equal(10, fp.Tokens);
        Assert.Equal(0, model.PrimaryCacheLength);
        Assert.False(model.HasFusedSequenceCache("pc:1:1"));
    }

    [Fact]
    public void Materialize_RevalidatesWithCanMaterialize()
    {
        var model = new AdaptedOracleModel(OracleFakes.R());
        IPrefixCacheModel pcm = model;
        Assert.True(model.BindSequenceCache("a"));
        model.Forward(Enumerable.Range(1, 12).ToArray());
        Assert.True(pcm.TryCaptureCopy("a", "pc:1:1", out _));
        Assert.False(pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "pc:1:1", "b", 12, 11)));
        Assert.False(pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, "pc:1:1", "b", 13, 13)));
        Assert.False(model.HasFusedSequenceCache("b"));
        Assert.True(pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "pc:1:1", "b", 12, 12)));
    }

    [Fact]
    public void PageOnlyFamily_RefusesEveryEndStateMember()
    {
        IPrefixCacheModel model = new PageOnlyFamily();
        Assert.False(model.TryCaptureCopy("r", "pc:1:1", out _));
        Assert.False(model.TryCaptureDonate("r", "pc:1:1", 4, out _));
        Assert.False(model.TryConvertPrimary("pc:1:1", 4, out _));
        Assert.False(model.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "pc:1:1", "r", 4, 4)));
        Assert.False(model.TryReturnDonation("r", "pc:1:1"));
        Assert.False(model.CanMaterialize("pc:1:1", 4, 4));
        Assert.Equal(default, model.MeasureEndState("pc:1:1"));
        Assert.Equal(default, model.EstimateCloneBytes("pc:1:1", 4));
        Assert.False(model.TryCopyPagedToHolder(new[] { 1 }, 4, "r"));
        Assert.False(model.TryExport("pc:1:1", new MemoryStream()));
        Assert.False(model.TryImport("pc:1:1", 4, new MemoryStream(), out _));
        Assert.False(model.TryBeginImport(4, out object? ticket));
        Assert.Null(ticket);
        model.ReleasePayloads(new[] { "pc:1:1" }, ReleaseReason.Evicted);
        model.AttachPrefixCache(new RecordingPayloadSink());
    }

    private sealed class PageOnlyFamily : IPageOnlyPrefixCacheModel
    {
        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
        {
            Class = FamilyClass.P,
            Readiness = PrefixCacheMode.Legacy,
            NamespaceFingerprint = "page-only",
            Pages = PageSupport.Both,
        };

        public long QuerySpareBytes(ResourceClass cls) => -1;
    }
}
