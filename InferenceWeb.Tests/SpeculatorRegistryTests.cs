// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The registry is the seam the whole refactor exists for: a new speculation
// algorithm should be a class plus one Register call, with no change to any
// model, executor or scheduler code. These tests pin that contract.
using System;
using System.Collections.Generic;
using System.Linq;

namespace InferenceWeb.Tests;

public class SpeculatorRegistryTests
{
    private static SpeculationOptions Opts(string name, int window = 8, float? pmin = null) => new()
    {
        Enabled = true,
        SpeculatorName = name,
        MaxDraftTokens = window,
        MinDraftProb = pmin,
    };

    [Fact]
    public void Auto_PerTokenHead_ResolvesToTheDraftHeadSpeculator()
    {
        var model = new StubTarget { Kind = DraftHeadKind.PerToken };

        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.Auto), out string decline);

        Assert.Null(decline);
        Assert.IsType<DraftHeadSpeculator>(spec);
        Assert.Equal(DraftHeadSpeculator.DefaultGate, spec.MinDraftProb);
        Assert.True(spec.NeedsHiddenState);
    }

    [Fact]
    public void Auto_BlockHead_ResolvesToTheBlockSpeculatorAndClampsToTheTrainedBlock()
    {
        var model = new StubTarget { Kind = DraftHeadKind.Block, BlockSize = 5 };

        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.Auto, window: 32), out string decline);

        Assert.Null(decline);
        Assert.IsType<BlockDraftSpeculator>(spec);
        // A block drafter can never propose more than it was trained to emit.
        Assert.Equal(5, spec.MaxDraftTokens);
        // The two gates threshold different quantities, so each algorithm brings
        // The two gates threshold different quantities, so each algorithm brings
        // its own default rather than sharing one: the block gate is a CUMULATIVE
        // prefix probability (it decays with every position), the per-token gate a
        // single top-1 probability. They are not on a common scale, so neither
        // ordering between them is meaningful - only that each is its own value.
        Assert.Equal(BlockDraftSpeculator.DefaultGate, spec.MinDraftProb);
        Assert.NotEqual(BlockDraftSpeculator.DefaultGate, DraftHeadSpeculator.DefaultGate);
    }

    [Fact]
    public void Auto_NoDraftHead_DeclinesAndPointsAtTheWeightFreeAlternative()
    {
        var model = new StubTarget { Kind = DraftHeadKind.None };

        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.Auto), out string decline);

        Assert.Null(spec);
        Assert.Contains(SpeculatorRegistry.NGram, decline);
    }

    [Theory]
    [InlineData(3, 8, false, 3)]   // the trunk prefers 3: the default window narrows to it
    [InlineData(3, 8, true, 8)]    // an explicit --spec-draft still wins
    [InlineData(2, 8, false, 2)]  // Metal IQ4 targets prefer a three-row verify
    [InlineData(2, 8, true, 8)]    // the IQ4 preference must not override an explicit window
    [InlineData(0, 8, false, 8)]   // no preference: the default stands
    public void NGram_TakesTheTrunksPreferredWindowLikeEveryOtherAlgorithm(
        int preferred, int requested, bool explicitRequest, int expected)
    {
        // A verify's cost is the trunk's - a recurrent state to snapshot per row,
        // the small-batch matmul kernels' 8-row limit - whoever proposed the rows.
        var model = new StubTarget { Kind = DraftHeadKind.None, PreferredWindow = preferred };
        var opts = new SpeculationOptions
        {
            Enabled = true,
            SpeculatorName = SpeculatorRegistry.NGram,
            MaxDraftTokens = requested,
            MaxDraftTokensExplicit = explicitRequest,
        };
        var spec = SpeculatorRegistry.Create(model, opts, out string decline);
        Assert.Null(decline);
        Assert.Equal(expected, spec.MaxDraftTokens);
    }

    [Theory]
    [InlineData(false, DraftHeadKind.None, 3)]
    [InlineData(true, DraftHeadKind.None, 7)]
    [InlineData(false, DraftHeadKind.PerToken, 3)]
    [InlineData(true, DraftHeadKind.PerToken, 7)]
    public void QwenVerifyLimitPreservesTheDefaultAndCapsExplicitNGramAndMtp(
        bool explicitRequest, DraftHeadKind kind, int expected)
    {
        var model = new StubTarget
        {
            Kind = kind, PreferredWindow = 3,
            HardDraftLimit = 7,   // a GGML verify window's widest exact row kernels
        };
        var options = new SpeculationOptions
        {
            Enabled = true,
            SpeculatorName = kind == DraftHeadKind.None ? SpeculatorRegistry.NGram : SpeculatorRegistry.Auto,
            MaxDraftTokens = 8, MaxDraftTokensExplicit = explicitRequest,
        };
        var speculator = SpeculatorRegistry.Create(model, options, out string decline);
        Assert.Null(decline);
        Assert.Equal(expected, speculator.MaxDraftTokens);
    }

    [Fact]
    public void NGram_ServesAModelWithNoDraftHeadAtAll()
    {
        // The whole point of the layering: an algorithm that needs no trained
        // speculator weights runs on any speculative trunk.
        var model = new StubTarget { Kind = DraftHeadKind.None };

        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.NGram), out string decline);

        Assert.Null(decline);
        Assert.IsType<NGramSpeculator>(spec);
        Assert.False(SpeculatorRegistry.RequiresDraftHead(SpeculatorRegistry.NGram));
        Assert.True(SpeculatorRegistry.RequiresDraftHead(SpeculatorRegistry.Auto));
    }

    [Theory]
    [InlineData(0, 8, false, 3)]
    [InlineData(12, 8, false, 12)]
    [InlineData(12, 2, false, 2)]
    [InlineData(12, 16, false, 12)]
    [InlineData(100, 8, false, 64)]
    [InlineData(12, 1, true, 1)]
    [InlineData(12, 3, true, 3)]
    [InlineData(12, 8, true, 8)]
    [InlineData(12, 12, true, 12)]
    public void NGram_UsesItsOwnDefaultButHonorsEveryExplicitWindow(
        int preferred, int requested, bool explicitRequest, int expected)
    {
        var model = new StubTarget { PreferredWindow = 3, PreferredNGramWindow = preferred };
        var options = new SpeculationOptions
        {
            Enabled = true,
            SpeculatorName = SpeculatorRegistry.NGram,
            MaxDraftTokens = requested,
            MaxDraftTokensExplicit = explicitRequest,
        };
        var spec = SpeculatorRegistry.Create(model, options, out string decline);
        Assert.Null(decline);
        Assert.Equal(expected, spec.MaxDraftTokens);
    }

    [Theory]
    [InlineData(DraftHeadKind.PerToken)]
    [InlineData(DraftHeadKind.Block)]
    public void LearnedDrafters_KeepTheirGeneralPreferenceWhenNGramPrefersTwelve(DraftHeadKind kind)
    {
        var model = new StubTarget
        {
            Kind = kind, BlockSize = 16, PreferredWindow = 3, PreferredNGramWindow = 12,
        };
        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.Auto), out string decline);
        Assert.Null(decline);
        Assert.Equal(3, spec.MaxDraftTokens);
    }

    [Fact]
    public void ExplicitAlgorithmThatTheModelCannotServe_DeclinesWithAReason()
    {
        var model = new StubTarget { Kind = DraftHeadKind.PerToken };

        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.Block), out string decline);

        Assert.Null(spec);
        Assert.Contains(SpeculatorRegistry.Block, decline);
    }

    [Fact]
    public void UnknownAlgorithm_DeclinesAndListsTheKnownOnes()
    {
        var model = new StubTarget { Kind = DraftHeadKind.PerToken };

        var spec = SpeculatorRegistry.Create(model, Opts("medusa-9000"), out string decline);

        Assert.Null(spec);
        Assert.Contains("medusa-9000", decline);
        Assert.Contains(SpeculatorRegistry.NGram, decline);
        Assert.False(SpeculatorRegistry.IsKnown("medusa-9000"));
        Assert.True(SpeculatorRegistry.IsKnown(SpeculatorRegistry.Auto));
    }

    [Fact]
    public void ExplicitPMin_OverridesTheAlgorithmDefault()
    {
        var model = new StubTarget { Kind = DraftHeadKind.PerToken };

        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.Auto, pmin: 0.42f), out _);

        Assert.Equal(0.42f, spec.MinDraftProb);
    }

    [Fact]
    public void Register_AddsANewAlgorithmWithoutTouchingAnyModelCode()
    {
        // This is the extension story in one test: a new speculation algorithm
        // is a class plus one Register call.
        string name = "test-echo-" + Guid.NewGuid().ToString("N")[..8];
        SpeculatorRegistry.Register(name, (target, options) => new EchoSpeculator(options.MaxDraftTokens),
            requiresDraftHead: false);

        Assert.True(SpeculatorRegistry.IsKnown(name));
        Assert.False(SpeculatorRegistry.RequiresDraftHead(name));
        Assert.Contains(name, SpeculatorRegistry.Names);

        var model = new StubTarget { Kind = DraftHeadKind.None };
        var spec = SpeculatorRegistry.Create(model, Opts(name, window: 3), out string decline);

        Assert.Null(decline);
        Assert.IsType<EchoSpeculator>(spec);

        var drafts = new List<int>();
        spec.Propose(new DraftContext { LastToken = 7, Position = 1, MaxTokens = 3 }, drafts);
        Assert.Equal(new[] { 7, 7, 7 }, drafts);
    }

    [Fact]
    public void Register_RefusesToShadowAuto()
    {
        Assert.Throws<ArgumentException>(() =>
            SpeculatorRegistry.Register(SpeculatorRegistry.Auto, (_, _) => null));
    }

    /// <summary>A one-screen speculation algorithm: repeat the last token. It is
    /// useless, which is exactly the point — it exists only to show that adding
    /// an algorithm touches nothing outside itself.</summary>
    private sealed class EchoSpeculator : ISpeculator
    {
        public EchoSpeculator(int maxDraftTokens) => MaxDraftTokens = maxDraftTokens;
        public string Name => "echo";
        public int MaxDraftTokens { get; }
        public float MinDraftProb { get; set; }
        public float DefaultMinDraftProb => 0f;
        public bool NeedsHiddenState => false;
        public bool HandlesOwnPrefill => false;
        public int Propose(in DraftContext ctx, List<int> draftOut)
        {
            for (int i = 0; i < ctx.MaxTokens; i++)
                draftOut.Add(ctx.LastToken);
            return draftOut.Count;
        }
        public void Commit(int[] tokens, float[] hRows, int startPos) { }
        public void Reset() { }
        public void Dispose() { }
    }

    /// <summary>Minimal speculative target: the registry only reads its config
    /// and its draft-head kind.</summary>
    // ---- prefix-reuse arming -------------------------------------------------
    //
    // The executor refuses to arm speculation on a sequence that adopted a KV
    // prefix, because a learned per-position draft head chains its state token by
    // token and a gap ruins it. That refusal used to apply to EVERY algorithm, and
    // since every turn after the first in a chat adopts a prefix, DFlash measured
    // as "no faster than plain" from turn two onward (41.6 vs 40.9 tok/s on
    // Muse-Glimmer; 75.5 once the gate consulted the algorithm - acceptance on the
    // reusing turn was the HIGHEST of the session, 79%).
    //
    // Whether a gap is survivable is the algorithm's own property, so it lives on
    // ISpeculator. These pin which way each algorithm answers.

    [Fact]
    public void BlockDrafter_CanArmAfterPrefixReuse()
    {
        var model = new StubTarget { Kind = DraftHeadKind.Block, BlockSize = 5 };
        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.Auto), out _);
        Assert.IsType<BlockDraftSpeculator>(spec);
        Assert.True(spec.CanArmAfterPrefixReuse,
            "A block drafter reads its own sliding KV ring and refills it from the forwarded "
            + "suffix, so an adopted prefix costs it context for a block or two, not correctness.");
    }

    [Fact]
    public void NGram_CanArmAfterPrefixReuse()
    {
        var model = new StubTarget { Kind = DraftHeadKind.None };
        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.NGram), out _);
        Assert.IsType<NGramSpeculator>(spec);
        Assert.True(spec.CanArmAfterPrefixReuse,
            "n-gram mines the emitted token history, which is complete whatever the KV cache did.");
    }

    [Fact]
    public void PerTokenDraftHead_DoesNotArmAfterPrefixReuse()
    {
        // The one algorithm the original blanket refusal was actually protecting.
        var model = new StubTarget { Kind = DraftHeadKind.PerToken };
        var spec = SpeculatorRegistry.Create(model, Opts(SpeculatorRegistry.Auto), out _);
        Assert.IsType<DraftHeadSpeculator>(spec);
        Assert.False(spec.CanArmAfterPrefixReuse,
            "A NextN/MTP head chains per-position state; a skipped span makes every later "
            + "proposal garbage, so it must stay on the plain path after prefix adoption.");
    }
    private sealed class StubTarget : ISpeculativeTarget, IDraftHead
    {
        public DraftHeadKind Kind { get; set; }
        public int BlockSize { get; set; }
        public int PreferredWindow { get; set; }
        public int HardDraftLimit { get; set; } = int.MaxValue;
        public int SpecMaxDraftTokens => HardDraftLimit;
        public int SpecPreferredDraftWindow => PreferredWindow;
        public int PreferredNGramWindow { get; set; }
        public int SpecPreferredNGramDraftWindow => PreferredNGramWindow;

        public DraftHeadKind DraftHeadKind => Kind;
        public int DraftBlockSize => BlockSize;
        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos) { }

        public ModelConfig Config { get; } = new() { VocabSize = 32, HiddenSize = 4 };
        public ITokenizer Tokenizer => null;
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public void TruncateKVCache(int tokenCount) { }
        public float[] Forward(int[] tokens) => new float[Config.VocabSize];
        public void ResetKVCache() { }
        public void Dispose() { }

        public int CacheSeqLen => 0;
        public int MaxContextLength => 4096;
        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows) { }
        public void SpecEnsureCapacity(int requiredSeqLen) { }
        public void SpecSnapshotRecurrentState() { }
        public void SpecRestoreRecurrentState() { }
        public void SpecRewindCache(int length) { }
    }

}
