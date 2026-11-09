using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public class DeepSeek41ThinkingBudgetTests
{
    private const int End = 2, Thought = 3, Answer = 4;
    private static readonly string Json = "{\"city\":\"Paris\",\"temperature_c\":19}";

    [Fact]
    public async Task ForcedClose_IsForwardedAndActivatesJsonGrammarBeforeAnswer()
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var config = Config(model.Tokenizer, 2, grammar: true);
        var seq = Sequence("json", config, 8);
        var handle = engine.SubmitRequest(seq);
        int[] output = await Collect(handle);
        Assert.Equal(new[] { Thought, Thought, End, Answer }, output);
        Assert.Equal("eos", (await handle.Completion).FinishReason);
        Assert.Contains(End, model.Forwarded);
        Assert.True(config.Grammar!.IsActive && config.Grammar.IsComplete && !config.Grammar.IsDead);
        var parser = new DeepSeek41OutputParser();
        parser.Init(true, null);
        var content = new StringBuilder();
        foreach (int token in output)
            content.Append(parser.Add(model.Tokenizer.Decode(new List<int> { token }), false).Content);
        content.Append(parser.Add("", true).Content);
        Assert.Equal(Json, content.ToString());
    }

    [Theory]
    [InlineData(2, 2, 2)]
    [InlineData(2, 3, 3)]
    [InlineData(2, 4, 4)]
    public async Task ForcedCloseNeverExtendsOriginalMaxTokens(int budget, int maxTokens, int expected)
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var handle = engine.SubmitRequest(Sequence("cap", Config(model.Tokenizer, budget), maxTokens));
        int[] output = await Collect(handle);
        Assert.Equal(expected, output.Length);
        Assert.Equal("max_tokens", (await handle.Completion).FinishReason);
        Assert.Equal(maxTokens > budget, output.Contains(End));
    }

    [Fact]
    public async Task ParallelRequestsKeepIndependentBudgetAndGrammarState()
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var first = engine.SubmitRequest(Sequence("first", Config(model.Tokenizer, 1, true), 9));
        var second = engine.SubmitRequest(Sequence("second", Config(model.Tokenizer, 3, true), 9));
        int[][] outputs = await Task.WhenAll(Collect(first), Collect(second));
        Assert.Equal(new[] { Thought, End, Answer }, outputs[0]);
        Assert.Equal(new[] { Thought, Thought, Thought, End, Answer }, outputs[1]);
    }

    [Fact]
    public async Task SharedCallerConfigForksGrammarBeforeConcurrentBudgetedRequests()
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var shared = SamplingConfig.Greedy;
        shared.Grammar = new GrammarConstraint(Grammar.JsonObject(), model.Tokenizer);
        shared.Grammar.ActivateAfter("</think>");
        var firstConfig = ChatGenerationPipeline.WithThinkingBudget(shared, model.Tokenizer, "deepseek41", 1, out bool firstInstalled);
        var secondConfig = ChatGenerationPipeline.WithThinkingBudget(shared, model.Tokenizer, "deepseek41", 3, out bool secondInstalled);
        Assert.True(firstInstalled && secondInstalled);
        Assert.NotSame(firstConfig.Grammar, secondConfig.Grammar);
        var first = engine.SubmitRequest(Sequence("shared-first", firstConfig, 9));
        var second = engine.SubmitRequest(Sequence("shared-second", secondConfig, 9));
        int[][] outputs = await Task.WhenAll(Collect(first), Collect(second));
        Assert.Equal(new[] { Thought, End, Answer }, outputs[0]);
        Assert.Equal(new[] { Thought, Thought, Thought, End, Answer }, outputs[1]);
        Assert.False(shared.Grammar.IsActive);
        Assert.Null(shared.ThinkingBudget);
    }

    [Fact]
    public void GrammarForkPreservesPartialTriggerWithoutSharingItsBuffer()
    {
        var tokenizer = new PieceTokenizer();
        tokenizer.Vocab[5] = "</th";
        tokenizer.Vocab[6] = "ink>";
        var source = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        source.ActivateAfter("</think>");
        source.Accept(5);
        var first = source.Fork();
        var second = source.Fork();
        first.Accept(6);
        Assert.True(first.IsActive);
        Assert.False(source.IsActive || second.IsActive);
        second.Accept(6);
        Assert.True(second.IsActive);
        first.Accept(Answer);
        Assert.True(first.IsComplete);
        Assert.False(second.IsComplete);
    }

    [Fact]
    public void GrammarForkPreservesPartialUtf8AndIndependentAnswerPositions()
    {
        var source = new GrammarConstraint(Grammar.JsonObject(), new PieceTokenizer());
        source.AcceptBytes(Encoding.UTF8.GetBytes("{\"city\":\""));
        source.AcceptBytes(new byte[] { 0xE2 });
        var first = source.Fork();
        var second = source.Fork();
        first.AcceptBytes(new byte[] { 0x82, 0xAC }); // euro sign
        second.AcceptBytes(new byte[] { 0x98, 0x83 }); // snowman
        first.AcceptBytes(Encoding.UTF8.GetBytes("\"}"));
        second.AcceptBytes(Encoding.UTF8.GetBytes("\"}"));
        Assert.True(first.IsComplete && second.IsComplete);
        Assert.False(first.IsDead || second.IsDead || source.IsComplete);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(400)]
    public async Task CancellationDuringForcedTokenForwardDoesNotGenerateFinalAnswer(int budget)
    {
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var model = new ReasoningModel { OnClosingForward = () =>
        {
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(5))) throw new TimeoutException();
        } };
        using var engine = Engine(model);
        using var cancellation = new CancellationTokenSource();
        var handle = engine.SubmitRequest(Sequence("cancel", Config(model.Tokenizer, budget), 512), cancellation.Token);
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));
            cancellation.Cancel();
        }
        finally { release.Set(); }
        int[] output = await Collect(handle);
        Assert.DoesNotContain(Answer, output);
        Assert.Equal("aborted", (await handle.Completion).FinishReason);
    }

    [Fact]
    public async Task RepetitionClosesReasoningThroughForwardAndProducesConstrainedAnswer()
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var config = Config(model.Tokenizer, 400, grammar: true);
        var handle = engine.SubmitRequest(Sequence("reasoning-loop", config, 512));
        int[] output = await Collect(handle);
        Assert.Equal(Enumerable.Repeat(Thought, RepetitionGuard.MinSpan).Concat(new[] { End, Answer }), output);
        Assert.Equal("eos", (await handle.Completion).FinishReason);
        Assert.Equal(1, model.Forwarded.Count(x => x == End));
        Assert.True(config.Grammar!.IsActive && config.Grammar.IsComplete && !config.Grammar.IsDead);
        var parser = new DeepSeek41OutputParser();
        parser.Init(true, null);
        var content = new StringBuilder();
        foreach (int token in output)
            content.Append(parser.Add(model.Tokenizer.Decode(new List<int> { token }), false).Content);
        content.Append(parser.Add("", true).Content);
        Assert.Equal(Json, content.ToString());
    }

    [Fact]
    public async Task RepetitionGuardStillStopsLoopingFinalAnswerAfterOneEarlyClose()
    {
        using var model = new ReasoningModel { IgnoreClosure = true };
        using var engine = Engine(model);
        var handle = engine.SubmitRequest(Sequence("repeat", Config(model.Tokenizer, 400), 512));
        int[] output = await Collect(handle);
        Assert.Equal(1, output.Count(x => x == End));
        Assert.Equal(RepetitionGuard.MinSpan, Array.IndexOf(output, End));
        Assert.Equal(2 * RepetitionGuard.MinSpan + 1, output.Length);
        Assert.Equal(RepetitionGuard.FinishReason, (await handle.Completion).FinishReason);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RepetitionWithoutExplicitRecoveryPolicyRetainsHardStop(bool budgetWithoutRecovery)
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var config = SamplingConfig.Greedy;
        if (budgetWithoutRecovery) config.ThinkingBudget = new ThinkingTokenBudget(400, End);
        var handle = engine.SubmitRequest(Sequence("ordinary-repeat", config, 512));
        int[] output = await Collect(handle);
        Assert.Equal(RepetitionGuard.MinSpan, output.Length);
        Assert.DoesNotContain(End, output);
        Assert.Equal(RepetitionGuard.FinishReason, (await handle.Completion).FinishReason);
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public async Task DisablingEitherRepetitionGuardPreventsEarlyClosure(bool requestGuard, bool engineGuard)
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model, engineGuard);
        var config = Config(model.Tokenizer, 400);
        config.StopRepetition = requestGuard;
        var handle = engine.SubmitRequest(Sequence("no-repetition-guard", config, 150));
        int[] output = await Collect(handle);
        Assert.Equal(150, output.Length);
        Assert.DoesNotContain(End, output);
        Assert.Equal("max_tokens", (await handle.Completion).FinishReason);
    }

    [Theory]
    [InlineData(128, false)]
    [InlineData(129, true)]
    public async Task EarlyCloseNeverExtendsOriginalLimit(int maxTokens, bool expectedClose)
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var handle = engine.SubmitRequest(Sequence("loop-limit", Config(model.Tokenizer, 400), maxTokens));
        int[] output = await Collect(handle);
        Assert.Equal(maxTokens, output.Length);
        Assert.Equal(expectedClose, output.Contains(End));
        Assert.DoesNotContain(Answer, output);
        Assert.Equal("max_tokens", (await handle.Completion).FinishReason);
    }

    [Fact]
    public void EarlyClosureIsPerSamplerAndRollsBackWithCommittedHistory()
    {
        var config = Config(new PieceTokenizer(), 400);
        var sampler = new TokenSampler(config);
        var history = Enumerable.Repeat(Thought, 128).ToList();
        Assert.True(sampler.TryRequestThinkingClosure(history));
        Assert.True(sampler.TryGetForcedThinkingToken(history, out int end));
        Assert.Equal(End, end);
        Assert.False(new TokenSampler(config.Clone()).TryGetForcedThinkingToken(history, out _));
        history.RemoveAt(history.Count - 1);
        Assert.False(sampler.TryGetForcedThinkingToken(history, out _));
        history.Add(End);
        Assert.False(sampler.TryRequestThinkingClosure(history));
    }

    /// <summary>A stop drawn before the loop threshold, with the reasoning still open, closes the
    /// reasoning first (issue #338: such a turn used to end with no answer); the model's next stop,
    /// after the close, ends the turn as usual.</summary>
    [Fact]
    public async Task NaturalEosInsideReasoning_ClosesTheReasoningFirst()
    {
        using var model = new ReasoningModel { EosAfterThoughtTokens = RepetitionGuard.MinSpan - 1 };
        using var engine = Engine(model);
        var handle = engine.SubmitRequest(Sequence("natural-eos", Config(model.Tokenizer, 400), 512));
        int[] output = await Collect(handle);
        Assert.Equal(RepetitionGuard.MinSpan, output.Length);
        Assert.Equal(End, output[^1]);
        Assert.Equal("eos", (await handle.Completion).FinishReason);
    }

    [Fact]
    public void EarlyClosureCannotBypassAnAlreadyActivatedAnswerGrammar()
    {
        var tokenizer = new PieceTokenizer();
        var config = Config(tokenizer, 400);
        config.Grammar = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        var sampler = new TokenSampler(config);
        Assert.False(sampler.TryRequestThinkingClosure(Enumerable.Repeat(Thought, 128).ToList()));
    }

    [Fact]
    public void NaturalCloseAndRollbackAreCountedFromCommittedHistory()
    {
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(2, End);
        var sampler = new TokenSampler(config);
        var logits = new float[7]; logits[Thought] = 20;
        var history = new List<int> { End, Answer };
        Assert.Equal(Thought, sampler.Sample(logits, history));
        history.Clear(); history.AddRange(new[] { Thought, Thought });
        Assert.Equal(End, sampler.Sample(logits, history));
        Assert.Equal(End, sampler.Sample(logits, history)); // Peek must not commit the close.
        history.Add(End);
        Assert.Equal(Thought, sampler.Sample(logits, history));
        var independent = new TokenSampler(config.Clone());
        Assert.Equal(End, independent.Sample(logits, new List<int> { Thought, Thought }));
    }

    [Theory]
    [InlineData("TakePendingOrSample", false)]
    [InlineData("PeekPendingOrSample", true)]
    public void ForcedCloseOverridesPendingDeviceArgmaxWithoutHostLogits(string method, bool retainsPending)
    {
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(2, End);
        var seq = Sequence("pending", config, 8);
        seq.AppendOutputToken(Thought); seq.AppendOutputToken(Thought);
        seq.PendingDeviceToken = Thought;
        seq.PendingDevicePosition = seq.NumComputedTokens;
        Assert.True(seq.GetOrCreateSampler().IsPlainGreedyArgmax);
        var operation = typeof(BatchExecutor).GetMethod(method, BindingFlags.Static | BindingFlags.NonPublic)!;
        Assert.Equal(End, (int)operation.Invoke(null, new object[] { seq })!);
        Assert.Equal(retainsPending, seq.PendingDeviceToken.HasValue);
        Assert.Null(seq.LastLogits);
    }

    [Theory]
    [InlineData("TakePendingOrSample", false)]
    [InlineData("PeekPendingOrSample", true)]
    public void EarlyCloseOverridesPendingDeviceArgmaxWithoutHostLogits(string method, bool retainsPending)
    {
        var seq = Sequence("pending-loop", Config(new PieceTokenizer(), 400), 512);
        for (int i = 0; i < RepetitionGuard.MinSpan; ++i) seq.AppendOutputToken(Thought);
        seq.PendingDeviceToken = Thought;
        seq.PendingDevicePosition = seq.NumComputedTokens;
        Assert.True(seq.GetOrCreateSampler().TryRequestThinkingClosure(seq.OutputTokens));
        var operation = typeof(BatchExecutor).GetMethod(method, BindingFlags.Static | BindingFlags.NonPublic)!;
        Assert.Equal(End, (int)operation.Invoke(null, new object[] { seq })!);
        Assert.Equal(retainsPending, seq.PendingDeviceToken.HasValue);
        Assert.Null(seq.LastLogits);
    }

    [Theory]
    [InlineData("deepseek41", 2, true)]
    [InlineData("deepseek_v41", 2, true)]
    [InlineData("deepseek41", 0, true)]   // no cap: installed without a limit, for the closes
    [InlineData("nemotron_h_moe", 2, true)]
    [InlineData("deepseek4", 2, false)]
    // Qwen 3.5/3.6/3.8 and GLM 5.x close their block at the budget too (they used to take the
    // hard stop, which ended the turn with an empty answer).
    [InlineData("qwen35", 2, true)]
    [InlineData("qwen35moe", 2, true)]
    [InlineData("qwen3next", 2, true)]
    [InlineData("qwen4exp", 2, true)]
    [InlineData("glm-dsa", 2, true)]
    [InlineData("glm5next", 2, true)]
    [InlineData("qwen35", 0, true)]
    public void HostInstallsCapabilityOnlyForSupportedThinkingProtocol(string architecture, int budget, bool expected)
    {
        var tokenizer = new PieceTokenizer();
        var original = SamplingConfig.Greedy;
        var result = ChatGenerationPipeline.WithThinkingBudget(original, tokenizer, architecture, budget, out bool installed);
        Assert.Equal(expected, installed);
        Assert.Null(original.ThinkingBudget);
        Assert.Equal(expected, result.ThinkingBudget != null);
        Assert.Equal(expected, result.ThinkingBudget?.CloseOnRepetition ?? false);
        Assert.True(result.StopRepetition);
    }

    [Fact]
    public async Task ClosingText_IsForwardedAheadOfTheEndToken_ThenTheModelAnswers()
    {
        using var model = new ReasoningModel();
        using var engine = Engine(model);
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(2, End, closeOnRepetition: true, closingTokenIds: new[] { 5, 6 });
        var handle = engine.SubmitRequest(Sequence("hand-over", config, 12));
        int[] output = await Collect(handle);
        Assert.Equal(new[] { Thought, Thought, 5, 6, End, Answer }, output);
        Assert.Equal("eos", (await handle.Completion).FinishReason);
        // The text went through the model like any other token, so the cache holds what was streamed.
        Assert.Equal(new[] { 5, 6, End }, model.Forwarded.SkipWhile(t => t != 5).Take(3).ToArray());
    }

    [Theory]
    [InlineData("qwen35", true)]
    [InlineData("qwen4exp", true)]
    [InlineData("glm-dsa", true)]
    [InlineData("glm5next", true)]
    [InlineData("deepseek41", false)]
    public void QwenAndGlm_HandOverToTheAnswerBeforeClosing_OthersCloseBare(string architecture, bool handsOver)
    {
        var tokenizer = new CharTokenizer();
        var result = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, architecture, 64,
            out bool installed);
        Assert.True(installed);
        string text = ChatProtocolRegistry.For(architecture)!.ThinkingBudgetClosingText;
        Assert.Equal(handsOver, text != null);
        Assert.Equal(handsOver ? tokenizer.Encode(text!, false).ToArray() : Array.Empty<int>(),
            result.ThinkingBudget!.ClosingTokenIds.ToArray());
        // An unrequested channel is only capped, never handed over.
        Assert.Empty(ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, "gemma4", 64,
            out _, enableThinking: false)?.ThinkingBudget?.ClosingTokenIds ?? Array.Empty<int>());
    }

    [Theory]
    [InlineData("qwen35", false)]
    [InlineData("qwen4exp", false)]
    [InlineData("glm-dsa", false)]
    [InlineData("glm5next", true)]   // its template opens <think> with thinking off too
    public void QwenAndGlm_BudgetARequestedBlock_AndGlm53FlashsAlwaysOpenOne(string architecture, bool alwaysOpen)
    {
        var protocol = ChatProtocolRegistry.For(architecture)!;
        Assert.Equal("</think>", protocol.ThinkingBudgetEndToken);
        Assert.Null(protocol.ThinkingBudgetOpenToken);   // the prompt opens the block itself
        Assert.Equal(alwaysOpen, protocol.PromptAlwaysOpensThinking);
        Assert.Equal(alwaysOpen, ChatGenerationPipeline.ReasonsWhetherAsked(architecture, enableThinking: false));
        Assert.True(ChatGenerationPipeline.ReasonsWhetherAsked(architecture, enableThinking: true));
        // A block the request did not ask for, and the prompt did not open, is left alone.
        var tokenizer = new PieceTokenizer();
        var original = SamplingConfig.Greedy;
        Assert.Same(original, ChatGenerationPipeline.WithThinkingBudget(original, tokenizer, architecture, 64,
            out bool installed, enableThinking: false));
        Assert.False(installed);
    }

    [Fact]
    public void InvalidEndTokenOrAlreadyActiveGrammarRetainsHardStopFallback()
    {
        var tokenizer = new PieceTokenizer();
        var config = SamplingConfig.Greedy;
        config.Grammar = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        Assert.Same(config, ChatGenerationPipeline.WithThinkingBudget(config, tokenizer, "deepseek41", 2, out bool installed));
        Assert.False(installed);
        config.Grammar = null;
        tokenizer.Vocab[End] = "not-the-trained-close";
        Assert.Same(config, ChatGenerationPipeline.WithThinkingBudget(config, tokenizer, "deepseek41", 2, out installed));
        Assert.False(installed);
    }

    private static SamplingConfig Config(ITokenizer tokenizer, int budget, bool grammar = false)
    {
        var config = SamplingConfig.Greedy;
        if (grammar)
        {
            config.Grammar = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
            config.Grammar.ActivateAfter("</think>");
        }
        var result = ChatGenerationPipeline.WithThinkingBudget(config, tokenizer, "deepseek41", budget, out bool installed);
        Assert.True(installed);
        if (config.Grammar != null) Assert.NotSame(config.Grammar, result.Grammar);
        return result;
    }

    private static SequenceState Sequence(string id, SamplingConfig config, int maxTokens) =>
        new(id, new[] { 6, 1 }, maxTokens, 8, config);

    private static InferenceEngine Engine(ReasoningModel model, bool stopRepetition = true) => new(model, new SchedulerConfig
    {
        MaxNumBatchedTokens = 8, MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 8, NumBlocks = 128, BlockSize = 8,
        EnablePrefixCaching = false, DecodeQuantumTokens = 1,
        StopRepetition = stopRepetition,
    }, NullLogger.Instance);

    private static async Task<int[]> Collect(InferenceRequestHandle handle)
    {
        var result = new List<int>();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await foreach (int token in handle.Tokens.ReadAllAsync(deadline.Token)) result.Add(token);
        await handle.Completion.WaitAsync(deadline.Token);
        return result.ToArray();
    }

    /// <summary>One token per character, plus the reasoning markers as single tokens.</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        private static readonly string[] Specials = { "<eos>", "<think>", "</think>", "<|channel>", "<channel|>" };
        public string[] Vocab { get; } = Specials.Concat(Enumerable.Range(0, 128).Select(c => ((char)c).ToString())).ToArray();
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 0 };
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => Specials.Length + c).ToList();
        public string Decode(List<int> ids) => string.Concat(ids.Select(i => Vocab[i]));
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }

    private sealed class PieceTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        public string[] Vocab { get; } = { "<eos>", "<think>", "</think>", "Reasoning ", Json, "wrong", "User" };
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 0 };
        public int VocabSize => Vocab.Length;
        public IReadOnlyCollection<int> SpecialTokenIds => new[] { 0, 1, 2 };
        public List<int> Encode(string text, bool addSpecial = true) => new() { LookupToken(text) };
        public string Decode(List<int> ids) => string.Concat(ids.Select(i => Vocab[i]));
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }

    private sealed class ReasoningModel : IModelArchitecture
    {
        private readonly List<int> _state = new();
        public List<int> Forwarded { get; } = new();
        public Action? OnClosingForward { get; init; }
        public bool IgnoreClosure { get; init; }
        public int EosAfterThoughtTokens { get; init; } = int.MaxValue;
        public ModelConfig Config { get; } = new() { Architecture = "deepseek41", VocabSize = 7 };
        public ITokenizer Tokenizer { get; } = new PieceTokenizer();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "thinking-budget-test";
        public float[] Forward(int[] tokens)
        {
            _state.AddRange(tokens); Forwarded.AddRange(tokens);
            if (tokens.Contains(End)) OnClosingForward?.Invoke();
            int next = IgnoreClosure || !_state.Contains(End) ? Thought : _state[^1] == Answer ? 0 : Answer;
            if (_state.Count(token => token == Thought) >= EosAfterThoughtTokens) next = 0;
            var logits = new float[7]; logits[next] = 20;
            return logits;
        }
        public void ResetKVCache() => _state.Clear();
        public void TruncateKVCache(int count) => _state.RemoveRange(count, _state.Count - count);
        public long ComputeKVBlockByteSize(int count) => count;
        public bool TryExtractKVBlock(int start, int count, Span<byte> destination)
        {
            if (destination.Length != count || start + count > _state.Count) return false;
            for (int i = 0; i < count; i++) destination[i] = (byte)_state[start + i];
            return true;
        }
        public bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source)
        {
            if (start != _state.Count || source.Length != count) return false;
            foreach (byte token in source) _state.Add(token);
            return true;
        }
        public void Dispose() { }
    }
}
