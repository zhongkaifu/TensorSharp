// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Gemma 4 opens its own thought channel (`<|channel>thought\n ... <channel|>`), and E4B
// does so after a tool result even with thinking OFF, where no template primes anything.
// Campaign 2026-09-16 (B10): the agentic final turn spent all 256 tokens inside that
// channel and returned empty content, and tool_round_trip wrote its answer, closed a
// channel it never opened, and wrote the answer again - a stream delivered both copies.
// These tests pin the sampler policy that bounds the channel and masks the stray close,
// with fake models that behave the way E4B was measured to. They run without a model.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public class Gemma4ThoughtChannelPolicyTests
{
    private const int Eos = 0, Open = 1, Close = 2, Thought = 3, Answer = 4, Newline = 5, ToolResponseEnd = 6, User = 7;

    // ---- Sampler policy ---------------------------------------------------------------

    [Fact]
    public void BudgetCountsFromTheModelsOwnOpener_AndAppliesAgainWhenTheChannelReopens()
    {
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(2, Close, openTokenId: Open, openAtStart: false);
        var sampler = new TokenSampler(config);

        Assert.False(sampler.TryGetForcedThinkingToken(new List<int> { Answer, Answer, Answer }, out _));
        Assert.False(sampler.TryGetForcedThinkingToken(new List<int> { Answer, Open, Thought }, out _));
        Assert.True(sampler.TryGetForcedThinkingToken(new List<int> { Answer, Open, Thought, Thought }, out int end));
        Assert.Equal(Close, end);
        Assert.False(sampler.TryGetForcedThinkingToken(new List<int> { Answer, Open, Thought, Thought, Close, Answer }, out _));
        Assert.True(sampler.TryGetForcedThinkingToken(
            new List<int> { Answer, Open, Thought, Thought, Close, Answer, Open, Thought, Thought }, out _));
    }

    [Fact]
    public void PrimedOpenChannel_CountsFromTheFirstGeneratedToken()
    {
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(2, Close, openTokenId: Open, openAtStart: true);
        Assert.True(new TokenSampler(config).TryGetForcedThinkingToken(new List<int> { Thought, Thought }, out _));
    }

    [Fact]
    public void BoundaryClose_WaitsForALineBreak_ButNeverPastTwiceTheLimit()
    {
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(2, Close, openTokenId: Open, openAtStart: false,
            closeAtBoundary: token => token == Newline);
        var sampler = new TokenSampler(config);

        Assert.False(sampler.TryGetForcedThinkingToken(new List<int> { Open, Thought, Thought }, out _));
        Assert.True(sampler.TryGetForcedThinkingToken(new List<int> { Open, Thought, Newline }, out _));
        Assert.False(new TokenSampler(config).TryGetForcedThinkingToken(new List<int> { Open, Thought, Thought, Thought }, out _));
        Assert.True(new TokenSampler(config).TryGetForcedThinkingToken(new List<int> { Open, Thought, Thought, Thought, Thought }, out _));
    }

    [Fact]
    public void UnopenedClose_IsMasked_OnlyWhileNoChannelIsOpen()
    {
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(int.MaxValue, Close, openTokenId: Open, openAtStart: false,
            suppressUnopenedEnd: true);
        var sampler = new TokenSampler(config);
        // Masking needs host logits: the device-argmax shortcut must be declined.
        Assert.False(sampler.IsPlainGreedyArgmax);

        float[] PreferClose() { var l = new float[8]; l[Close] = 20; l[Eos] = 10; return l; }
        Assert.Equal(Eos, sampler.Sample(PreferClose(), new List<int> { Answer }));
        Assert.Equal(Close, sampler.Sample(PreferClose(), new List<int> { Answer, Open, Thought }));
        Assert.Equal(Eos, sampler.Sample(PreferClose(), new List<int> { Answer, Open, Thought, Close, Answer }));
    }

    [Fact]
    public void LegacyPrimedBudget_StillClosesOnceAndNeverReopens()
    {
        // DeepSeek V4.1's shape: no opener token, channel open from the start.
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(2, Close);
        var sampler = new TokenSampler(config);
        Assert.True(sampler.IsPlainGreedyArgmax);
        Assert.True(sampler.TryGetForcedThinkingToken(new List<int> { Thought, Thought }, out _));
        Assert.False(sampler.TryGetForcedThinkingToken(new List<int> { Thought, Close, Open, Thought, Thought }, out _));
    }

    // ---- Host installation --------------------------------------------------------------

    [Fact]
    public void ThinkingOff_AfterAToolResult_InstallsTheCapAndTheStrayCloseMask()
    {
        var tokenizer = new GemmaLikeTokenizer();
        int budget = ChatGenerationPipeline.UnrequestedThinkingBudgetFor(256);
        var result = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, "gemma4", budget,
            out bool installed, enableThinking: false, promptTokens: new[] { User, ToolResponseEnd });
        Assert.True(installed);
        var policy = result.ThinkingBudget!;
        Assert.Equal(64, policy.TokenLimit);
        Assert.Equal(Close, policy.EndTokenId);
        Assert.Equal(Open, policy.OpenTokenId);
        Assert.False(policy.OpenAtStart);
        Assert.True(policy.SuppressUnopenedEnd);
        Assert.NotNull(policy.CloseAtBoundary);
        Assert.True(policy.CloseAtBoundary!(Newline));
        Assert.False(policy.CloseAtBoundary(Thought));
    }

    [Fact]
    public void ThinkingOff_OrdinaryTurn_CapsTheChannelWithoutMaskingTheClose()
    {
        // Elsewhere a close with no opener separates reasoning the model wrote after a
        // primed empty block from its answer, which the parser relies on.
        var tokenizer = new GemmaLikeTokenizer();
        var result = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, "gemma4", 64,
            out bool installed, enableThinking: false, promptTokens: new[] { User });
        Assert.True(installed);
        Assert.False(result.ThinkingBudget!.SuppressUnopenedEnd);
        Assert.True(new TokenSampler(result).IsPlainGreedyArgmax);
    }

    [Fact]
    public void ThinkingOn_PromptPrimedOpenChannel_CountsFromTheStartAtTheExactBudget()
    {
        var tokenizer = new GemmaLikeTokenizer();
        var result = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, "gemma4", 100,
            out bool installed, enableThinking: true, promptTokens: new[] { ToolResponseEnd, Open, Thought, Newline });
        Assert.True(installed);
        Assert.True(result.ThinkingBudget!.OpenAtStart);
        Assert.False(result.ThinkingBudget.SuppressUnopenedEnd);
        Assert.Null(result.ThinkingBudget.CloseAtBoundary);
        Assert.Equal(100, result.ThinkingBudget.TokenLimit);
    }

    [Theory]
    [InlineData("deepseek41")]
    [InlineData("qwen35")]
    public void ThinkingOff_FamiliesWhoseModelNeverOpensTheChannel_GetNothing(string architecture)
    {
        var original = SamplingConfig.Greedy;
        Assert.Same(original, ChatGenerationPipeline.WithThinkingBudget(original, new GemmaLikeTokenizer(), architecture, 64,
            out bool installed, enableThinking: false, promptTokens: new[] { User }));
        Assert.False(installed);
    }

    [Theory]
    [InlineData(256, 64)]
    [InlineData(2048, 64)]
    [InlineData(100, 25)]
    [InlineData(3, 1)]
    [InlineData(0, 0)]
    public void UnrequestedBudget_IsAQuarterOfTheAllowanceCappedAt64(int maxTokens, int expected)
        => Assert.Equal(expected, ChatGenerationPipeline.UnrequestedThinkingBudgetFor(maxTokens));

    [Fact]
    public void Gemma4Protocol_DeclaresTheChannelTokensAndTheDelayedGrammarTrigger()
    {
        var protocol = ChatProtocolRegistry.For("gemma4")!;
        Assert.Equal("<|channel>", protocol.ThinkingBudgetOpenToken);
        Assert.Equal("<channel|>", protocol.ThinkingBudgetEndToken);
        Assert.Equal("<tool_response|>", protocol.SuppressUnopenedThinkingEndAfter);
        Assert.Equal("<channel|>", OutputParserFactory.GrammarActivationTrigger("gemma4", enableThinking: true));
        // Thinking off: the answer is the reply, so the grammar enforces from token 0.
        Assert.Null(OutputParserFactory.GrammarActivationTrigger("gemma4", enableThinking: false));
    }

    // ---- End to end through the engine --------------------------------------------------

    [Fact]
    public async Task ModelThatNeverClosesItsThought_AnswersOnceTheCapClosesIt()
    {
        using var model = new GemmaLikeModel(GemmaLikeModel.Behaviour.ThinksForever);
        using var engine = Engine(model);

        // Without the policy: every token is thought, and nothing is left for an answer.
        int[] unbounded = await Collect(engine.SubmitRequest(Sequence("unbounded", SamplingConfig.Greedy, 40)));
        Assert.DoesNotContain(Answer, unbounded);

        var config = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, model.Tokenizer, "gemma4", 8,
            out bool installed, enableThinking: false, promptTokens: Prompt);
        Assert.True(installed);
        var handle = engine.SubmitRequest(Sequence("capped", config, 40));
        int[] output = await Collect(handle);
        Assert.Equal("eos", (await handle.Completion).FinishReason);
        int close = Array.IndexOf(output, Close);
        Assert.True(close > 8, "the channel closes at a line break past the cap");
        Assert.Equal(Newline, output[close - 1]);
        Assert.Equal(Answer, output[close + 1]);

        var parser = new Gemma4OutputParser();
        parser.Init(false, null);
        var content = new StringBuilder();
        foreach (int token in output)
            content.Append(parser.Add(model.Tokenizer.Decode(new List<int> { token }), false).Content);
        content.Append(parser.Add("", true).Content);
        Assert.Equal("{\"total\":68.75}", content.ToString());
    }

    [Fact]
    public async Task ModelThatClosesAChannelItNeverOpened_StopsInsteadOfRepeatingItsAnswer()
    {
        using var model = new GemmaLikeModel(GemmaLikeModel.Behaviour.StrayCloseThenRepeat);
        using var engine = Engine(model);

        int[] before = await Collect(engine.SubmitRequest(Sequence("stray", SamplingConfig.Greedy, 12)));
        Assert.Equal(new[] { Answer, Close, Answer }, before.Take(3));

        var config = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, model.Tokenizer, "gemma4", 64,
            out bool installed, enableThinking: false, promptTokens: Prompt);
        Assert.True(installed);
        var handle = engine.SubmitRequest(Sequence("masked", config, 12));
        Assert.Equal(new[] { Answer }, await Collect(handle));
        Assert.Equal("eos", (await handle.Completion).FinishReason);
    }

    private static readonly int[] Prompt = { User, ToolResponseEnd };

    private static SequenceState Sequence(string id, SamplingConfig config, int maxTokens) =>
        new(id, Prompt, maxTokens, 8, config);

    private static InferenceEngine Engine(GemmaLikeModel model) => new(model, new SchedulerConfig
    {
        MaxNumBatchedTokens = 8, MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 8, NumBlocks = 128, BlockSize = 8,
        EnablePrefixCaching = false, DecodeQuantumTokens = 1,
        // The fake thought is one repeated token; the repetition guard is not under test.
        StopRepetition = false,
    }, NullLogger.Instance);

    private static async Task<int[]> Collect(InferenceRequestHandle handle)
    {
        var result = new List<int>();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await foreach (int token in handle.Tokens.ReadAllAsync(deadline.Token)) result.Add(token);
        await handle.Completion.WaitAsync(deadline.Token);
        return result.ToArray();
    }

    private sealed class GemmaLikeTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        public string[] Vocab { get; } =
            { "<turn|>", "<|channel>", "<channel|>", "thought", "{\"total\":68.75}", "\n", "<tool_response|>", "User" };
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { Eos };
        public int VocabSize => Vocab.Length;
        public IReadOnlyCollection<int> SpecialTokenIds => new[] { Eos, Open, Close, ToolResponseEnd };
        public List<int> Encode(string text, bool addSpecial = true) => new() { LookupToken(text) };
        public string Decode(List<int> ids) => string.Concat(ids.Select(i => Vocab[i]));
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == Eos;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }

    /// <summary>Deterministic stand-in for the two measured E4B behaviours after a tool result.</summary>
    private sealed class GemmaLikeModel(GemmaLikeModel.Behaviour behaviour) : IModelArchitecture
    {
        public enum Behaviour { ThinksForever, StrayCloseThenRepeat }

        private readonly List<int> _state = new();
        public ModelConfig Config { get; } = new() { Architecture = "gemma4", VocabSize = 8 };
        public ITokenizer Tokenizer { get; } = new GemmaLikeTokenizer();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "gemma4-thought-channel-test";

        public float[] Forward(int[] tokens)
        {
            _state.AddRange(tokens);
            var generated = _state.Skip(Prompt.Length).ToList();
            var logits = new float[8];
            if (behaviour == Behaviour.ThinksForever)
            {
                if (generated.Count == 0) logits[Open] = 20;
                else if (generated.Contains(Close)) logits[generated[^1] == Answer ? Eos : Answer] = 20;
                else logits[generated.Count % 5 == 0 ? Newline : Thought] = 20;
            }
            else
            {
                // Answer, then a close it prefers over ending the turn, then the answer again.
                if (generated.Count == 0 || generated[^1] == Close) logits[Answer] = 20;
                else if (generated[^1] == Answer) { logits[Close] = 20; logits[Eos] = 10; }
                else logits[Eos] = 20;
            }
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
