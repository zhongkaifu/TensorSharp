// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public partial class SpeculativeExecutionTests
{
    private static FakeSpeculativeModel FixedGovernorTarget() => new()
    { SpecPrefersFixedCostGovernor = true, BlockDraftSize = 7, ResumesAfterGap = true };

    private static BlockDraftSpeculator FixedGovernorBlock(FakeSpeculativeModel head, int k = 7, float p = 0)
        => new(head, 7, k) { MinDraftProb = p };

    [Fact]
    public void GovernorPolicy_OnlyTheResolvedNativeBlockProfileSelectsFixedDrafting()
    {
        var target = FixedGovernorTarget();
        using var block = FixedGovernorBlock(target);
        var profile = SpeculationGovernorFactory.Describe(target, block, true, null);
        Assert.False(profile.Enabled);
        Assert.Equal("qualified-native-ninfer-dflash-k7", profile.Source);
        Assert.Equal(7, profile.MaxDraft);
        Assert.Equal(0f, profile.MinProbability);
        Assert.Same(target, profile.Head);
    }

    [Theory]
    [InlineData("0", false)] [InlineData("1", true)] [InlineData("", true)]
    [InlineData("false", true)] [InlineData(" 0", true)] [InlineData("anything", true)]
    public void GovernorPolicy_ExplicitEnvironmentRetainsOriginalRawSemantics(string raw, bool enabled)
    {
        var target = FixedGovernorTarget();
        using var block = FixedGovernorBlock(target);
        var profile = SpeculationGovernorFactory.Describe(target, block, true, raw);
        Assert.Equal(enabled, profile.Enabled);
        Assert.Equal("explicit-TS_SPEC_ADAPTIVE", profile.Source);
        Assert.Equal(0, target.FixedGovernorHintReads);
    }

    [Theory]
    [InlineData("foreign-head")] [InlineData("non-greedy")] [InlineData("window")]
    [InlineData("confidence")] [InlineData("target-cap")] [InlineData("different-algorithm")]
    public void GovernorPolicy_DeclinedProfilesDoNotQueryBackendCapability(string reason)
    {
        var target = FixedGovernorTarget();
        if (reason == "target-cap") target.SpecMaxDraftTokens = 6;
        using ISpeculator algorithm = reason == "different-algorithm"
            ? new NGramSpeculator(7)
            : FixedGovernorBlock(reason == "foreign-head" ? FixedGovernorTarget() : target,
                reason == "window" ? 6 : 7, reason == "confidence" ? .35f : 0);
        Assert.True(SpeculationGovernorFactory.Describe(target, algorithm, reason != "non-greedy", null).Enabled);
        Assert.Equal(0, target.FixedGovernorHintReads);
        using var native = FixedGovernorBlock(target);
        target.SpecMaxDraftTokens = 7;
        Assert.False(SpeculationGovernorFactory.Describe(target, native, true, null).Enabled);
        Assert.Equal(1, target.FixedGovernorHintReads);
    }

    [Theory]
    [InlineData("foreign-head")] [InlineData("non-greedy")] [InlineData("unqualified-model")]
    [InlineData("window")] [InlineData("confidence")] [InlineData("target-cap")]
    [InlineData("different-algorithm")]
    public void GovernorPolicy_UnmeasuredProfilesRetainAdaptiveDefault(string reason)
    {
        var target = FixedGovernorTarget();
        if (reason == "unqualified-model") target.SpecPrefersFixedCostGovernor = false;
        if (reason == "target-cap") target.SpecMaxDraftTokens = 6;
        using ISpeculator algorithm = reason == "different-algorithm"
            ? new NGramSpeculator(7)
            : FixedGovernorBlock(reason == "foreign-head" ? FixedGovernorTarget() : target,
                reason == "window" ? 6 : 7, reason == "confidence" ? .35f : 0);
        var profile = SpeculationGovernorFactory.Describe(target, algorithm, reason != "non-greedy", null);
        Assert.True(profile.Enabled);
        Assert.Equal("legacy-default", profile.Source);
    }

    [Fact]
    public void GovernorPolicy_RegistryResolvesEffectiveProbabilityBeforeSelection()
    {
        var target = FixedGovernorTarget();
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.Block,
            MaxDraftTokens = 7, MaxDraftTokensExplicit = true, MinDraftProb = 0 };
        using var zero = SpeculatorRegistry.Create(target, options, out var reason);
        Assert.Null(reason);
        Assert.NotNull(zero);
        Assert.False(SpeculationGovernorFactory.Describe(target, zero, true, null).Enabled);
        using var defaultGate = SpeculatorRegistry.Create(target, options with { MinDraftProb = null }, out reason);
        Assert.Null(reason);
        Assert.Equal(BlockDraftSpeculator.DefaultGate, defaultGate!.MinDraftProb);
        Assert.True(SpeculationGovernorFactory.Describe(target, defaultGate, true, null).Enabled);
    }

    [Fact]
    public void GovernorPolicy_ChangingProfileReplacesSharedGovernorWithoutMutatingLiveExecution()
    {
        var target = FixedGovernorTarget();
        using var firstBlock = FixedGovernorBlock(target);
        using var secondBlock = FixedGovernorBlock(target);
        SpeculationCostGovernor? shared = null;
        SpeculationGovernorFactory.Profile? previous = null;
        var first = SpeculationGovernorFactory.Describe(target, firstBlock, true, null);
        var same = SpeculationGovernorFactory.Describe(target, secondBlock, true, null);
        var fixedGovernor = SpeculationGovernorFactory.Share(ref shared, ref previous, first);
        using var live = new SpeculativeExecution(target, firstBlock, governor: fixedGovernor);
        Assert.Same(fixedGovernor, SpeculationGovernorFactory.Share(ref shared, ref previous, same));
        var nonGreedy = SpeculationGovernorFactory.Describe(target, secondBlock, false, null);
        var adaptiveGovernor = SpeculationGovernorFactory.Share(ref shared, ref previous, nonGreedy);
        Assert.NotSame(fixedGovernor, adaptiveGovernor);
        Assert.True(adaptiveGovernor.Enabled);
        Assert.False(live.Governor.Enabled);
        Assert.Same(fixedGovernor, live.Governor);
        using var foreign = FixedGovernorBlock(FixedGovernorTarget());
        var foreignProfile = SpeculationGovernorFactory.Describe(target, foreign, false, null);
        Assert.NotSame(adaptiveGovernor, SpeculationGovernorFactory.Share(ref shared, ref previous, foreignProfile));
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void GovernorPolicy_CallerOwnedGovernorIsAuthoritativeAcrossSequenceAndStatsReset(bool enabled)
    {
        var target = FixedGovernorTarget();
        using var block = FixedGovernorBlock(target);
        var supplied = new SpeculationCostGovernor { Enabled = enabled };
        using var exec = new SpeculativeExecution(target, block, governor: supplied);
        Assert.Same(supplied, exec.Governor);
        Assert.Equal(enabled, exec.AdaptiveSpeculation);
        exec.ResetStatsAndGovernor();
        supplied.NewSequence();
        Assert.Equal(enabled, exec.AdaptiveSpeculation);
    }

    [Fact]
    public void GovernorPolicy_StandaloneConstructorCannotInferTheCallersSampler()
    {
        var target = FixedGovernorTarget();
        using var exec = new SpeculativeExecution(target, FixedGovernorBlock(target));
        Assert.Equal(new SpeculationCostGovernor().Enabled, exec.AdaptiveSpeculation);
        Assert.Equal(SpeculationCostGovernor.AdaptiveEnvironmentSnapshot == null ? "legacy-default"
            : "explicit-TS_SPEC_ADAPTIVE", exec.Governor.SelectionSource);
    }

    [Fact]
    public void GovernorPolicy_FixedDefaultDoesNotReenterCalibrationOnResetOrNewSequence()
    {
        var target = FixedGovernorTarget();
        using var block = FixedGovernorBlock(target);
        var governor = SpeculationGovernorFactory.Create(SpeculationGovernorFactory.Describe(target, block, true, null));
        using var exec = new SpeculativeExecution(target, block, governor: governor);
        for (int sequence = 0; sequence < 3; sequence++)
        {
            governor.NewSequence();
            for (int step = 0; step < 200; step++)
            {
                Assert.True(governor.AllowsSpeculation());
                governor.Record(speculated: step % 2 == 0, tokensEmitted: 1, elapsedTicks: 1000);
            }
            Assert.Equal(0, governor.Wins);
            Assert.Equal(0, governor.Losses);
            Assert.Equal(0, governor.ParkedSteps);
            exec.ResetStatsAndGovernor();
            Assert.False(governor.Enabled);
        }
    }

    [Theory]
    [InlineData("temperature")] [InlineData("repeat")] [InlineData("presence")]
    [InlineData("frequency")] [InlineData("grammar")] [InlineData("observer")]
    public void GovernorPolicy_ArmingRejectsUnmeasuredSamplerOrLogitObserver(string reason)
    {
        var config = SamplingConfig.Greedy;
        switch (reason)
        {
            case "temperature": config.Temperature = .5f; break;
            case "repeat": config.RepetitionPenalty = 1.1f; break;
            case "presence": config.PresencePenalty = 1; break;
            case "frequency": config.FrequencyPenalty = 1; break;
            case "grammar": config.Grammar = new GrammarConstraint(GbnfParser.Parse("root ::= \"a\""), FixedGovernorTarget().Tokenizer); break;
        }
        var seq = new SequenceState("governor", new[] { 1 }, 20, BlockSize, config);
        Assert.False(BatchExecutor.CanUseFixedGovernorGreedyProfile(seq, reason == "observer"));
        seq.AppendOutputToken(4);
        Assert.False(BatchExecutor.CanUseFixedGovernorGreedyProfile(seq, reason == "observer"));
    }

    [Fact]
    public void GovernorPolicy_ArmingKnowsSamplerBeforeFirstOutputWithoutEnablingEarlyGreedyVerify()
    {
        var seq = new SequenceState("governor", new[] { 1 }, 20, BlockSize, SamplingConfig.Greedy);
        Assert.True(BatchExecutor.CanUseFixedGovernorGreedyProfile(seq, false));
        Assert.False(BatchExecutor.CanUseGreedySpecVerify(seq, false));
        seq.AppendOutputToken(4);
        Assert.True(BatchExecutor.CanUseGreedySpecVerify(seq, false));
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void GovernorPolicy_CurrentAndHistoricalMediaRetainAdaptiveQualification(bool generatedOutput)
    {
        var seq = new SequenceState("media", new[] { 1, 2, 3 }, 20, BlockSize, SamplingConfig.Greedy,
            mediaSpans: new[] { new PromptMediaSpan(1, 2, "historical-image") });
        if (generatedOutput) seq.AppendOutputToken(4);
        Assert.False(BatchExecutor.CanUseFixedGovernorGreedyProfile(seq, false));
    }

    [Fact]
    public void GovernorPolicy_BothHttpArmingSitesUseSamePolicyAndSetSpeculationClearsIt()
    {
        var target = FixedGovernorTarget();
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.Block,
            MaxDraftTokens = 7, MaxDraftTokensExplicit = true, MinDraftProb = 0 };
        var pool = new BlockPool(16, BlockSize, BlockSize * 4);
        var scheduler = new ContinuousBatchScheduler(new SchedulerConfig { Speculation = options }, pool);
        var executor = new BatchExecutor(target, pool, scheduler);
        var seq = new SequenceState("linear", new[] { 1 }, 20, BlockSize, SamplingConfig.Greedy);
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var arm = typeof(BatchExecutor).GetMethod("TryArmSpeculation", flags)!;
        using var linear = (SpeculativeExecution)arm.Invoke(executor, new object?[] { target, seq, null, "test" })!;
        var fusedSeq = new SequenceState("fused", new[] { 1 }, 20, BlockSize, SamplingConfig.Greedy);
        object?[] args = { target, fusedSeq, 0, null };
        Assert.True((bool)typeof(BatchExecutor).GetMethod("TryGetFusedSpecContext", flags)!.Invoke(executor, args)!);
        var fused = (SpeculativeExecution)args[3]!.GetType().GetField("Exec")!.GetValue(args[3])!;
        Assert.Same(linear.Governor, fused.Governor);
        var shared = typeof(BatchExecutor).GetField("_sharedGovernor", flags)!;
        Assert.Same(linear.Governor, shared.GetValue(executor));
        var before = linear.AdaptiveSpeculation;
        executor.SetSpeculation(options);
        Assert.Null(shared.GetValue(executor));
        Assert.Null(typeof(BatchExecutor).GetField("_sharedGovernorProfile", flags)!.GetValue(executor));
        using var next = (SpeculativeExecution)arm.Invoke(executor, new object?[] { target, seq, null, "test" })!;
        Assert.NotSame(linear.Governor, next.Governor);
        Assert.Equal(before, next.AdaptiveSpeculation);
    }
}
