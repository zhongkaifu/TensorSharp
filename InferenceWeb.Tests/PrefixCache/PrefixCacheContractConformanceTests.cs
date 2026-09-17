// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>
/// The contract conformance script (DESIGN §14.2 M2) on the seven oracle fakes, and proof that each
/// oracle detects the misuse it exists to detect: an oracle that cannot tell a wrong reuse from a
/// right one would let every engine-level test built on it pass vacuously.
/// </summary>
public sealed partial class PrefixCacheContractConformanceTests
{
    private readonly ITestOutputHelper _output;

    public PrefixCacheContractConformanceTests(ITestOutputHelper output) { _output = output; }

    public static IEnumerable<object[]> Fakes => OracleFakes.All.Select(f => new object[] { f.Name });

    private static OracleModel Create(string name) => OracleFakes.All.Single(f => f.Name == name).Create();

    /// <summary>40 shared tokens and a 30-token suffix: 70 &gt; the S oracle's window, so its wrapped
    /// refusal runs, while the script's rewind sequence stays inside the window.</summary>
    internal static ConformanceSubject FakeSubject(IModelArchitecture model, string name, Action<string> log) => new()
    {
        Name = "oracle-" + name,
        Model = model,
        SharedPrefix = Enumerable.Range(0, 40).Select(i => (i * 37 + 11) % OracleModel.DefaultVocab).ToArray(),
        Suffix = Enumerable.Range(0, 30).Select(i => (i * 53 + 7) % OracleModel.DefaultVocab).ToArray(),
        DecodeTokens = 12,
        DonateAfter = 4,
        RewindTokens = 6,
        ExpectedReadiness = PrefixCacheMode.Tree,
        PayloadBytesKnown = true,
        PayloadDeviceDirty = model switch
        {
            OracleModel o => o.IsDeviceDirty,
            AdaptedOracleModel a => a.Inner.IsDeviceDirty,
            _ => null,
        },
        ExpectDirtyDonations = (model as OracleModel ?? (model as AdaptedOracleModel)?.Inner)?.Traits.DeviceDirtyOnForward == true,
        Log = log,
    };

    [Trait("Category", "PrefixCacheUnit")]
    [Theory]
    [MemberData(nameof(Fakes))]
    public void OracleFake_PassesTheConformanceScript(string name)
    {
        OracleModel model = Create(name);
        ConformanceReport report = PrefixCacheConformanceScript.Run(FakeSubject(model, name, _output.WriteLine));
        Assert.Contains("capabilities", report.Ran);
        Assert.Contains("release leaves no keys", report.Ran);
        if (model.Traits.EndState != EndStateSupport.None)
        {
            Assert.Contains("donate/return/donate", report.Ran);
            Assert.Contains(report.Ran, r => r.StartsWith("batched release", StringComparison.Ordinal));
        }
        if (model.Traits.EndState == EndStateSupport.CopyAndDonate)
        {
            Assert.Contains("clone x2", report.Ran);
            Assert.Contains("settle then clone", report.Ran);
        }
        if (model.Traits.Truncation != TruncationKind.None)
            Assert.Contains(report.Ran, r => r.StartsWith("truncate in range", StringComparison.Ordinal));
        if (model.Traits.Persistable)
            Assert.Contains(report.Ran, r => r.StartsWith("export/import", StringComparison.Ordinal));
        if (model.Traits.Truncation == TruncationKind.WithinUnwrappedWindow)
            Assert.Contains("truncate refused (wrapped window)", report.Ran);
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleFakes_AreSevenAndEachEmulatesADistinctFamily()
    {
        Assert.Equal(7, OracleFakes.All.Count);
        var classes = OracleFakes.All.Select(f => f.Create().GetPrefixCacheCapabilities()).ToList();
        Assert.Equal(new[] { FamilyClass.P, FamilyClass.P, FamilyClass.S, FamilyClass.S, FamilyClass.R, FamilyClass.R, FamilyClass.N },
            classes.Select(c => c.Class));
        Assert.All(classes, c => ResumabilityRules.Validate(c, 16));
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleFakes_CountContractCallsButNotTheCapabilityRead()
    {
        foreach (var (name, create) in OracleFakes.All)
        {
            OracleModel model = create();
            model.GetPrefixCacheCapabilities();
            model.Forward(new[] { 1, 2, 3 });
            Assert.Equal(0, model.StateMemberCalls);
            model.CanMaterialize("pc:1:1", 3, 3);
            Assert.Equal(1, model.StateMemberCalls);
        }
    }

    // ------------------------------------------------------------------ the oracles detect misuse

    private static List<int> Greedy(IModelArchitecture model, float[] logits, int n)
    {
        var output = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (i > 0) logits = model.Forward(new[] { output[^1] });
            output.Add(Array.IndexOf(logits, logits.Max()));
        }
        return output;
    }

    private static int[] Tokens(int count, int salt) =>
        Enumerable.Range(0, count).Select(i => (i * 31 + salt) % OracleModel.DefaultVocab).ToArray();

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleS_ARewindOfAWrappedRingSucceedsWithAWrongState()
    {
        OracleModel model = OracleFakes.S();
        int[] prompt = Tokens(OracleFakes.GemmaWindow + 10, 3);
        Assert.True(model.BindSequenceCache("cold"));
        ulong expected = StateAfter(model, prompt.Take(prompt.Length - 4).ToArray());
        model.OnSequenceReleased("cold");

        Assert.True(model.BindSequenceCache("wrapped"));
        model.Forward(prompt);
        Assert.False(model.CanTruncateKVCache(prompt.Length, prompt.Length - 4));
        Assert.True(model.TryTruncateKVCache(prompt.Length - 4), "the oracle does not protect itself; the rules must");
        Assert.NotEqual(expected, model.ActiveState);

        // Within the window the same rewind is exact.
        int[] shortPrompt = prompt.Take(OracleFakes.GemmaWindow - 1).ToArray();
        Assert.True(model.BindSequenceCache("short"));
        model.Forward(shortPrompt);
        Assert.True(model.TryTruncateKVCache(shortPrompt.Length - 4));
        Assert.True(model.BindSequenceCache("short-cold"));
        Assert.Equal(StateAfter(model, shortPrompt.Take(shortPrompt.Length - 4).ToArray()), StateOf(model, "short"));
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Theory]
    [InlineData("S")]
    [InlineData("R")]
    public void DeviceDirtyOracle_RefusesToCloneAnUnsettledHolderUntilMaterializeSettlesIt(string name)
    {
        OracleModel model = Create(name);
        Assert.True(model.BindSequenceCache("a"));
        model.Forward(Tokens(20, 5));
        Assert.True(model.RetainSequenceCache("a"));
        Assert.True(model.IsDeviceDirty("a"));
        Assert.False(model.TryCloneRetainedCache("a", "b"), "a copy of device-authoritative state must be refused");
        Assert.True(model.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "a", "b", 20, 20)));
        Assert.False(model.IsDeviceDirty("a"));
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleR_MaterializesExactLengthsOnlyAndCannotRewind()
    {
        OracleModel model = OracleFakes.R();
        Assert.True(model.BindSequenceCache("a"));
        model.Forward(Tokens(20, 9));
        Assert.True(model.TryCaptureDonate("a", "pc:1:1", 20, out _));
        Assert.True(model.CanMaterialize("pc:1:1", 20, 20));
        Assert.False(model.CanMaterialize("pc:1:1", 20, 19));
        Assert.False(model.TryCaptureDonate("missing", "pc:1:2", 19, out _));
        Assert.True(model.BindSequenceCache("b"));
        model.Forward(Tokens(20, 9));
        Assert.False(model.TryTruncateKVCache(19));
        Assert.False(model.TryCaptureDonate("b", "pc:1:3", 19, out _), "a donation below the holder length needs a rewind the family lacks");
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleN_ReclaimsTheOldestRetainedSlotAndReportsItThroughTheSink()
    {
        OracleModel model = OracleFakes.N();
        var sink = new RecordingPayloadSink();
        model.AttachPrefixCache(sink);
        for (int i = 0; i < 2; i++)
        {
            Assert.True(model.BindSequenceCache("r" + i));
            model.Forward(Tokens(10, i));
            Assert.True(model.TryCaptureDonate("r" + i, "pc:1:" + i, 10, out _));
        }
        Assert.Empty(sink.Reports);
        // Primary + two retained slots fill the limit of three: the next request reclaims the oldest.
        Assert.True(model.BindSequenceCache("next"));
        Assert.Equal(new[] { ("pc:1:0", InvalidationReason.NativeSlotReclaimed) }, sink.Reports);
        Assert.DoesNotContain("pc:1:0", model.RetainedPayloadKeys);
        Assert.False(model.CanMaterialize("pc:1:0", 10, 10));
        // The model decides rewinds: within 8 tokens and aligned to 2.
        Assert.True(model.CanMaterialize("pc:1:1", 10, 4));
        Assert.False(model.CanMaterialize("pc:1:1", 10, 3));
        Assert.False(model.CanMaterialize("pc:1:1", 10, 0));
        Assert.False(model.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "pc:1:1", "clone", 10, 10)), "donate-only");
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleP2_AZeroSlabInjectedAsAPageChangesTheOutput()
    {
        OracleModel model = OracleFakes.P2(blockSize: 16);
        int[] prompt = Tokens(40, 17);
        List<int> cold = Greedy(model, model.Forward(prompt), 6);
        model.ResetKVCache();
        model.Forward(prompt.Take(32).ToArray());
        var slab = new byte[model.ComputeKVBlockByteSize(16)];
        Assert.True(model.TryExtractKVBlock(16, 16, slab));
        var first = new byte[model.ComputeKVBlockByteSize(16)];
        Assert.True(model.TryExtractKVBlock(0, 16, first));

        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 16, first));
        Assert.True(model.TryInjectKVBlock(16, 16, slab));
        Assert.Equal(cold, Greedy(model, model.Forward(prompt.Skip(32).ToArray()), 6));

        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 16, first));
        Assert.True(model.TryInjectKVBlock(16, 16, new byte[slab.Length]));   // an A2-only page's A1 slab (D6)
        Assert.NotEqual(cold, Greedy(model, model.Forward(prompt.Skip(32).ToArray()), 6));
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleP_ModelPagedForwardReadsThroughTheBlockTableAndCopiesToAHolderExactly()
    {
        OracleModel model = OracleFakes.P(blockSize: 16);
        int[] prompt = Tokens(40, 23);
        List<int> cold = Greedy(model, model.Forward(prompt), 6);
        model.ResetKVCache();

        var seq = new SequenceState("paged", prompt, maxNewTokens: 8, blockSize: 16, new SamplingConfig());
        seq.BlockTable.AppendBlock(new KvBlock(3));
        seq.BlockTable.AppendBlock(new KvBlock(7));
        float[] logits = model.ForwardBatch(PrefillContext(seq, 0, 32))[0];
        Assert.Equal(StateAfterCold(prompt.Take(32).ToArray()), PeakOf(logits));

        // A holder copied from the pages continues exactly like the cold run.
        Assert.True(model.TryCopyPagedToHolder(new[] { 3, 7 }, 32, "copied"));
        Assert.False(model.BindSequenceCache("copied"));
        Assert.Equal(cold, Greedy(model, model.Forward(prompt.Skip(32).ToArray()), 6));

        // The same pages read through the wrong block table give another state.
        Assert.True(model.TryCopyPagedToHolder(new[] { 7, 3 }, 32, "swapped"));
        Assert.False(model.BindSequenceCache("swapped"));
        Assert.NotEqual(cold, Greedy(model, model.Forward(prompt.Skip(32).ToArray()), 6));

        static int PeakOf(float[] l) => Array.IndexOf(l, l.Max());
        static int StateAfterCold(int[] tokens)
        {
            OracleModel reference = OracleFakes.P(blockSize: 16);
            float[] l = reference.Forward(tokens);
            return Array.IndexOf(l, l.Max());
        }
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleR2_APageEndingOffAForwardBoundaryIsNotRestorable()
    {
        OracleModel model = OracleFakes.R2(blockSize: 16);
        int[] prompt = Tokens(48, 29);
        List<int> cold = Greedy(model, model.Forward(prompt), 6);

        // Forward boundaries at 16 and 40: the page [0,16) ends at a boundary, [16,32) does not.
        model.ResetKVCache();
        model.Forward(prompt.Take(16).ToArray());
        model.Forward(prompt.Skip(16).Take(24).ToArray());
        var restorable = new byte[model.ComputeKVBlockByteSize(16)];
        var notRestorable = new byte[restorable.Length];
        Assert.True(model.TryExtractKVBlock(0, 16, restorable));
        Assert.True(model.TryExtractKVBlock(16, 16, notRestorable));

        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 16, restorable));
        Assert.Equal(cold, Greedy(model, model.Forward(prompt.Skip(16).ToArray()), 6));

        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 16, restorable));
        Assert.True(model.TryInjectKVBlock(16, 16, notRestorable));
        Assert.NotEqual(cold, Greedy(model, model.Forward(prompt.Skip(32).ToArray()), 6));
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleS2_RefusesARewindBeyondTheRingSlack()
    {
        OracleModel model = OracleFakes.S2();
        int slack = OracleFakes.MuseRingRows - OracleFakes.MuseWindow - 1;
        model.Forward(Tokens(100, 31));
        Assert.False(model.TryTruncateKVCache(100 - slack - 1));
        Assert.Equal(100, model.ActiveLength);
        Assert.True(model.TryTruncateKVCache(100 - slack));
        Assert.Equal(OracleFakes.MuseRingRows, model.MaxReusablePrefixTokens);
    }

    [Trait("Category", "PrefixCacheUnit")]
    [Fact]
    public void OracleFakes_ReleaseIsBatchedAndIdempotent()
    {
        OracleModel model = OracleFakes.S();
        Assert.True(model.BindSequenceCache("a"));
        model.Forward(Tokens(12, 1));
        for (int i = 0; i < 3; i++) Assert.True(model.TryCaptureCopy("a", "pc:1:" + i, out _));
        model.ReleasePayloads(new[] { "pc:1:0", "pc:1:1", "pc:1:2", "unknown" }, ReleaseReason.Evicted);
        Assert.Equal(1, model.DecodeGraphResets);
        model.ReleasePayloads(new[] { "pc:1:0", "unknown" }, ReleaseReason.Evicted);
        Assert.Equal(1, model.DecodeGraphResets);
        Assert.Empty(model.RetainedPayloadKeys);
    }

    private static ulong StateAfter(OracleModel model, int[] tokens)
    {
        model.Forward(tokens);
        return model.ActiveState;
    }

    private static ulong StateOf(OracleModel model, string requestId)
    {
        model.BindSequenceCache(requestId);
        return model.ActiveState;
    }

    private static BatchedForwardContext PrefillContext(SequenceState seq, int start, int count)
    {
        int[] table = seq.BlockTable.Blocks.Select(b => b.Id).ToArray();
        int blockSize = seq.BlockTable.BlockSize;
        var positions = Enumerable.Range(start, count).ToList();
        return new BatchedForwardContext
        {
            Sequences = new List<SequenceState> { seq },
            NumScheduledTokens = new List<int> { count },
            QueryStartLoc = new List<int> { 0, count },
            Positions = positions,
            SlotMapping = positions.Select(p => table[p / blockSize] * blockSize + p % blockSize).ToList(),
            BlockTables = new[] { table },
            MaxQueryLen = count,
            MaxSeqLen = start + count,
        };
    }
}
