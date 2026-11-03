// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Scheduling;
using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;

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
    public void GreedyVerify_TextRequestStillUsesExplicitTokenOnlyHook()
    {
        var model = new FakeSpeculativeModel { EnableGreedyVerify = true };
        var seq = RunEngineRequest(model, 10, 32, true);
        Assert.Empty(seq.MediaSpans);
        Assert.True(BatchExecutor.CanUseGreedySpecVerify(seq, false));
        Assert.True(model.GreedyVerifyCalls > 0);
        Assert.Empty(model.ProtocolViolations);
    }

    [Theory]
    [InlineData("image", false)]
    [InlineData("image", true)]
    [InlineData("video", false)]
    [InlineData("video", true)]
    [InlineData("audio", false)]
    [InlineData("audio", true)]
    public void GreedyVerify_PreparedCurrentOrHistoricalMediaRetainsFullLogits(string kind, bool historical)
    {
        // Use the real injector/request bucket/ToMediaSpan/SequenceState handoff.
        // Tiny CPU embeddings stand in at the encoder-result seam already used by
        // MultimodalPreparationIsolationTests; this is not a pixel/audio encoder test.
        using var rows = new Tensor(new CpuAllocator(BlasEnum.DotNet), DType.Float32, 2, 1);
        rows.SetElementsAsFloat(new[] { 11f, 12f });
        var preparedModel = (GreedyMediaProbeModel)RuntimeHelpers.GetUninitializedObject(typeof(GreedyMediaProbeModel));
        preparedModel.Rows = rows;
        using var injector = new ModelMultimodalInjector(preparedModel);
        var attachment = new ChatMessage { Role = "user", Content = "describe the attachment" };
        if (kind == "audio") attachment.AudioPaths = new List<string> { "fixture.wav" };
        else attachment.ImagePaths = kind == "video"
            ? new List<string> { "fixture-frame-0.png", "fixture-frame-1.png" }
            : new List<string> { "fixture.png" };
        var history = new List<ChatMessage> { attachment };
        if (historical)
        {
            history.Add(new ChatMessage { Role = "assistant", Content = "I have read the attachment." });
            history.Add(new ChatMessage { Role = "user", Content = "Now answer about the earlier attachment." });
            Assert.Null(history[^1].ImagePaths);
            Assert.Null(history[^1].AudioPaths);
        }
        string requestId = "media-greedy-" + kind;
        var prompt = injector.ProcessPromptTokens(history, Enumerable.Range(1, 10).ToList(), requestId);
        var spans = injector.GetPreparedMediaSpans(requestId);
        Assert.Equal(kind == "video" ? 2 : 1, spans.Count);
        Assert.All(spans, span => Assert.StartsWith("greedy-media-fixture:", span.ContentId));
        var seq = new SequenceState(requestId, prompt, 32, BlockSize, SamplingConfig.Greedy, mediaSpans: spans);

        // Deliver the prompt embeddings. A decode slice beyond the prompt queues
        // nothing, but the complete request still carries its historical media.
        Assert.True(injector.QueuePromptEmbeddingsForSlice(0, prompt.Count, requestId));
        Assert.Equal(spans.Count, preparedModel.Delivered);
        Assert.False(injector.QueuePromptEmbeddingsForSlice(prompt.Count, 1, requestId));
        Assert.Equal(spans, seq.MediaSpans);
        var expectedSpans = spans.ToArray();
        Assert.IsType<List<PromptMediaSpan>>(spans).Clear();
        injector.ClearPreparedPromptState(requestId);
        Assert.Empty(injector.GetPreparedMediaSpans(requestId));
        Assert.Equal(expectedSpans, seq.MediaSpans); // Constructor takes a snapshot.

        var baseline = new FakeSpeculativeModel();
        var expected = RunEngineRequest(baseline, prompt.Count, 32, true);
        // Throw if the new hook is entered at ANY decode step, including after
        // prefill and with no pending embedding queue. This drives BatchExecutor,
        // not a mirrored eligibility predicate.
        var candidate = new FakeSpeculativeModel { EnableGreedyVerify = true, ThrowGreedyVerify = true };
        RunGreedyMediaRequest(candidate, seq);
        Assert.True(seq.NumComputedTokens >= prompt.Count);
        Assert.False(BatchExecutor.CanUseGreedySpecVerify(seq, false));
        Assert.Equal(0, candidate.GreedyVerifyCalls);
        Assert.True(candidate.FullVerifyCalls > 0);
        Assert.Equal(expected.OutputTokens, seq.OutputTokens);
        Assert.Equal(expected.Status, seq.Status);
        Assert.Equal(expected.FinishReason, seq.FinishReason);
        Assert.Equal(expectedSpans, seq.MediaSpans);
        Assert.Empty(candidate.ProtocolViolations);
    }

    private static void RunGreedyMediaRequest(FakeSpeculativeModel model, SequenceState seq)
    {
        var config = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256, MaxNumRunningSequences = 4, MaxPrefillChunkSize = 64,
            SoloPrefillChunkSize = 8192, NumBlocks = 32, BlockSize = BlockSize,
            EnablePrefixCaching = false, DecodeQuantumTokens = 1,
            Speculation = new SpeculationOptions { Enabled = true, MaxDraftTokens = 8, MinDraftProb = .5f },
        };
        using var engine = new InferenceEngine(model, config, NullLogger.Instance);
        var completion = engine.SubmitRequest(seq).Completion.GetAwaiter().GetResult();
        Assert.True(completion.PrefillElapsedTicks > 0);
        Assert.True(completion.DecodeElapsedTicks > 0);
    }

    private sealed class GreedyMediaProbeModel : ModelBase, IMultimodalPromptExpander, IVisionCapableModel, IAudioCapableModel
    {
        private GreedyMediaProbeModel() : base("unused", BackendType.Cpu) { }
        public Tensor Rows;
        public int Delivered;
        public bool IsVisionEncoderLoaded => true;
        public void LoadVisionEncoder(string path) => throw new NotSupportedException();
        public void SetVisionEmbeddings(Tensor embeddings, int position) => Receive(embeddings);
        public void SetAudioEmbeddings(Tensor embeddings, int position) => Receive(embeddings);
        private void Receive(Tensor embeddings)
        {
            using (embeddings) Assert.Equal(new[] { 11f, 12f }, embeddings.GetElementsAsFloat(2));
            Delivered++;
        }
        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(ModelMultimodalInjector injector, List<ChatMessage> history, List<int> tokens)
        {
            int index = 0;
            foreach (var message in history)
            {
                foreach (var path in message.ImagePaths ?? Enumerable.Empty<string>()) Append(injector, path, true, index++);
                foreach (var path in message.AudioPaths ?? Enumerable.Empty<string>()) Append(injector, path, false, index++);
            }
            return tokens;
        }
        private void Append(ModelMultimodalInjector injector, string path, bool vision, int index)
        {
            const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Instance;
            var type = typeof(ModelMultimodalInjector);
            var cacheType = type.GetNestedType("CachedEmbedding", BindingFlags.NonPublic)!;
            var spanType = type.GetNestedType("PreparedEmbeddingSpan", BindingFlags.NonPublic)!;
            object cached = Activator.CreateInstance(cacheType, path, 0L, 0L, Rows, 2, 0, 0)!;
            cacheType.GetProperty("ContentId")!.SetValue(cached, "greedy-media-fixture:" + path);
            int start = 2 + 2 * index;
            object span = Activator.CreateInstance(spanType, cached, start, start, start + 2)!;
            var bucket = (IList)type.GetProperty(vision ? "_preparedVisionEmbeddings" : "_preparedAudioEmbeddings", flags)!.GetValue(injector)!;
            bucket.Add(span);
        }
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
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
