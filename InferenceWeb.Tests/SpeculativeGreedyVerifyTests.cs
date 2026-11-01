// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public partial class SpeculativeExecutionTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void GreedyVerify_OnlyExplicitSupportedHookOmitsLogits(bool targetSupports, bool callerOptsIn)
    {
        var model = new FakeSpeculativeModel { EnableGreedyVerify = targetSupports };
        using var exec = NewExec(model, 4);
        int pos = PrefillPrompt(model, exec, 5, out int anchor);
        var accepted = new List<int>();
        var result = exec.DecodeStep(anchor, pos, 4, Argmax, onDraftAccepted: accepted.Add,
            drawGreedy: callerOptsIn ? raw => raw : null);
        bool used = targetSupports && callerOptsIn;
        Assert.Equal(used, result.NextLogits == null);
        Assert.Equal(used ? 1 : 0, model.GreedyVerifyCalls);
        Assert.Equal(used ? 0 : 1, model.FullVerifyCalls);
        Assert.Equal(4, result.AcceptedCount);
        Assert.Equal(model.ExpectedNext(pos + 4), result.NextToken);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void GreedyVerify_PartialAcceptanceAndFollowingStepKeepHiddenCarryExact()
    {
        var model = new FakeSpeculativeModel { EnableGreedyVerify = true };
        using var exec = NewExec(model, 4);
        int pos = PrefillPrompt(model, exec, 5, out int anchor);
        model.DraftWrongPositions.Add(pos + 1);
        var result = exec.DecodeStep(anchor, pos, 4, _ => throw new Exception("host sampler used"), drawGreedy: raw => raw);
        Assert.Equal(1, result.AcceptedCount);
        Assert.Null(result.NextLogits);
        Assert.Equal(pos + 2, model.CacheSeqLen);
        Assert.Equal(model.ExpectedNext(pos + 1), result.NextToken);
        model.DraftWrongPositions.Clear();
        var next = exec.DecodeStep(result.NextToken, model.CacheSeqLen, 4, Argmax, drawGreedy: raw => raw);
        Assert.Equal(4, next.AcceptedCount);
        Assert.Equal(2, model.GreedyVerifyCalls);
        Assert.Equal(0, model.FullVerifyCalls);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void GreedyVerify_FailureIsNeverRetriedAsASecondForward()
    {
        var model = new FakeSpeculativeModel { EnableGreedyVerify = true, ThrowGreedyVerify = true };
        using var exec = NewExec(model, 4);
        int pos = PrefillPrompt(model, exec, 5, out int anchor);
        Assert.Throws<InvalidOperationException>(() => exec.DecodeStep(anchor, pos, 4, Argmax, drawGreedy: raw => raw));
        Assert.Equal(1, model.GreedyVerifyCalls);
        Assert.Equal(0, model.FullVerifyCalls);
        Assert.Equal(pos, model.CacheSeqLen);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void GreedyVerify_NoDraftOrNoBudgetStillReturnsPlainLogits(int budget)
    {
        var model = new FakeSpeculativeModel { EnableGreedyVerify = true };
        using var exec = NewExec(model, 4);
        int pos = PrefillPrompt(model, exec, 5, out int anchor);
        model.LowConfidencePositions.Add(pos);
        var result = exec.DecodeStep(anchor, pos, budget, Argmax, drawGreedy: _ => throw new Exception("unexpected verify"));
        Assert.False(result.UsedSpeculation);
        Assert.NotNull(result.NextLogits);
        Assert.Equal(-1, result.NextToken);
        Assert.Equal(0, model.GreedyVerifyCalls);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(20)]
    [InlineData(40)]
    public void GreedyVerify_EngineBudgetAndPendingTokensMatchFullLogitPath(int budget)
    {
        var baseline = new FakeSpeculativeModel();
        var candidate = new FakeSpeculativeModel { EnableGreedyVerify = true };
        var expected = RunEngineRequest(baseline, 5, budget, true);
        var actual = RunEngineRequest(candidate, 5, budget, true);
        Assert.Equal(expected.OutputTokens, actual.OutputTokens);
        Assert.Equal(expected.Status, actual.Status);
        Assert.Equal(budget, actual.OutputTokens.Count);
        if (budget > 2) Assert.True(candidate.GreedyVerifyCalls > 0);
        Assert.Empty(candidate.ProtocolViolations);
    }

    [Fact]
    public void GreedyVerify_ForcedThinkingAtAnAcceptedRowMatchesHostSampling()
    {
        SamplingConfig Config()
        {
            var config = SamplingConfig.Greedy;
            config.ThinkingBudget = new ThinkingTokenBudget(3, 63, closingTokenIds: new[] { 62, 61 });
            return config;
        }
        var baseline = new FakeSpeculativeModel();
        var candidate = new FakeSpeculativeModel { EnableGreedyVerify = true };
        var expected = RunEngineRequest(baseline, 5, 32, true, sampling: Config());
        var actual = RunEngineRequest(candidate, 5, 32, true, sampling: Config());
        Assert.Equal(expected.OutputTokens, actual.OutputTokens);
        Assert.Equal(new[] { 62, 61, 63 }, actual.OutputTokens.Skip(3).Take(3));
        Assert.True(candidate.GreedyVerifyCalls > 0);
        Assert.Empty(candidate.ProtocolViolations);
    }

    [Fact]
    public void GreedyVerify_EosInAcceptedWindowStopsAtExactCachePrefix()
    {
        var baseline = new FakeSpeculativeModel();
        var candidate = new FakeSpeculativeModel { EnableGreedyVerify = true };
        baseline.EosTokenId = candidate.EosTokenId = candidate.ExpectedNext(9);
        var expected = RunEngineRequest(baseline, 5, 32, true);
        int retainedLength = -1;
        var actual = RunEngineRequest(candidate, 5, 32, true, beforeDispose: () => retainedLength = candidate.CacheSeqLen);
        Assert.Equal(expected.OutputTokens, actual.OutputTokens);
        Assert.Equal(SequenceStatus.FinishedStopped, actual.Status);
        Assert.Equal(actual.OutputTokens.Count + 5, retainedLength);
        Assert.True(candidate.GreedyVerifyCalls > 0);
        Assert.Empty(candidate.ProtocolViolations);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GreedyVerify_RepetitionStopOrThinkingCloseMatchesFullPath(bool closeThinking)
    {
        SamplingConfig Config()
        {
            var config = SamplingConfig.Greedy;
            if (closeThinking) config.ThinkingBudget = new ThinkingTokenBudget(1000, 63, closeOnRepetition: true);
            return config;
        }
        var baseline = new FakeSpeculativeModel { ConstantNextToken = 4 };
        var candidate = new FakeSpeculativeModel { ConstantNextToken = 4, EnableGreedyVerify = true };
        var expected = RunEngineRequest(baseline, 5, 200, true, sampling: Config());
        var actual = RunEngineRequest(candidate, 5, 200, true, sampling: Config());
        Assert.Equal(expected.OutputTokens, actual.OutputTokens);
        Assert.Equal(expected.FinishReason, actual.FinishReason);
        Assert.True(candidate.GreedyVerifyCalls > 0);
        if (closeThinking) Assert.Contains(63, actual.OutputTokens);
        Assert.Empty(candidate.ProtocolViolations);
    }

    [Theory]
    [InlineData("temperature")]
    [InlineData("repeat")]
    [InlineData("presence")]
    [InlineData("frequency")]
    [InlineData("grammar")]
    [InlineData("unopened-end")]
    [InlineData("diagnostics")]
    [InlineData("first-token")]
    public void GreedyVerify_GateRejectsEveryFullLogitConsumer(string reason)
    {
        var config = SamplingConfig.Greedy;
        switch (reason)
        {
            case "temperature": config.Temperature = .5f; break;
            case "repeat": config.RepetitionPenalty = 1.1f; break;
            case "presence": config.PresencePenalty = 1; break;
            case "frequency": config.FrequencyPenalty = 1; break;
            case "grammar": config.Grammar = new GrammarConstraint(GbnfParser.Parse("root ::= \"a\""), new FakeSpeculativeModel().Tokenizer); break;
            case "unopened-end": config.ThinkingBudget = new ThinkingTokenBudget(10, 63, openTokenId: 62, openAtStart: false, suppressUnopenedEnd: true); break;
        }
        var seq = new SequenceState("gate", new[] { 1 }, 20, 8, config);
        if (reason != "first-token") seq.AppendOutputToken(4);
        Assert.False(BatchExecutor.CanUseGreedySpecVerify(seq, reason == "diagnostics"));
    }

    [Fact]
    public void GreedyVerify_FirstTokenAllowListIsAlreadyResolvedBeforeVerify()
    {
        var config = SamplingConfig.Greedy;
        config.FirstTokenAllowList = new[] { 2 };
        var seq = new SequenceState("gate", new[] { 1 }, 20, 8, config);
        Assert.False(BatchExecutor.CanUseGreedySpecVerify(seq, false));
        seq.AppendOutputToken(2);
        Assert.True(BatchExecutor.CanUseGreedySpecVerify(seq, false));
    }

    [Fact]
    public void GreedyVerify_ContextHeadroomCapsTheWindowAndThenUsesPlainStep()
    {
        var model = new FakeSpeculativeModel { EnableGreedyVerify = true, ContextLimit = 8 };
        using var exec = NewExec(model, 4);
        int position = PrefillPrompt(model, exec, 5, out int anchor);
        var first = exec.DecodeStep(anchor, position, 4, Argmax, drawGreedy: raw => raw);
        Assert.Equal(1, first.AcceptedCount);
        Assert.Null(first.NextLogits);
        var last = exec.DecodeStep(first.NextToken, model.CacheSeqLen, 4, Argmax, drawGreedy: raw => raw);
        Assert.False(last.UsedSpeculation);
        Assert.NotNull(last.NextLogits);
        Assert.Equal(8, model.CacheSeqLen);
        Assert.Equal(1, model.GreedyVerifyCalls);
    }

    [Fact]
    public void GreedyVerify_RearmingOverRetainedDeviceCarryKeepsPendingCorrection()
    {
        var model = new FakeSpeculativeModel
        { EnableGreedyVerify = true, CarryOnDevice = true, ResumesAfterGap = true, BlockDraftSize = 3 };
        var firstExec = NewExec(model, 3);
        int position = PrefillPrompt(model, firstExec, 5, out int anchor);
        var first = firstExec.DecodeStep(anchor, position, 3, Argmax, drawGreedy: raw => raw);
        int pending = first.NextToken;
        Assert.Null(first.NextLogits);
        firstExec.Dispose();
        using var resumed = NewExec(model, 3);
        resumed.SeedCommitted(model.Trunk.ToArray());
        position = model.CacheSeqLen;
        var next = resumed.DecodeStep(pending, position, 3, Argmax, drawGreedy: raw => raw);
        Assert.Equal(3, next.AcceptedCount);
        Assert.Equal(model.ExpectedNext(position + 3), next.NextToken);
        Assert.Equal(2, model.GreedyVerifyCalls);
        Assert.Empty(model.ProtocolViolations);
    }
}
