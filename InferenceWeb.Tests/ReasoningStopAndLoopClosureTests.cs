// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Issue #338: Qwen3.8-27B NVFP4 under Aider ended 3 of 53 turns on a stop token in the middle of a
/// reasoning loop (line after line the same apart from a number), so the reply was all reasoning and an
/// OpenAI client received an empty <c>stop</c>. A stop drawn inside open reasoning now closes the channel
/// instead (the closing text, then the end token) and the model answers; reasoning that settles into a
/// templated loop is closed the same way before it gets there; and a turn that still ends answerless is
/// reported as the reasoning overrun it is.
/// </summary>
public sealed class ReasoningStopAndLoopClosureTests
{
    private const int Eos = 0, End = 2, Thought = 3, Answer = 4, CloseA = 5, CloseB = 6, FirstLine = 7, Lines = 200;

    // ---- A stop drawn inside open reasoning ---------------------------------------------------------

    [Fact]
    public async Task AStopInsideReasoning_ClosesTheChannel_ThenTheModelAnswers()
    {
        using var model = new Reasoner { StopAfterThoughts = 5 };
        using var engine = Engine(model);
        var handle = engine.SubmitRequest(Sequence("stop", Policy(), 32));
        int[] output = await Collect(handle);
        Assert.Equal(new[] { Thought, Thought, Thought, Thought, Thought, CloseA, CloseB, End, Answer }, output);
        Assert.Equal("eos", (await handle.Completion).FinishReason);
        // The stop inside the reasoning was never forwarded (only the answer's own stop is); the close
        // went through the model like any other token.
        Assert.DoesNotContain(Eos, model.Forwarded.TakeWhile(t => t != End));
        Assert.Equal(new[] { CloseA, CloseB, End }, model.Forwarded.SkipWhile(t => t != CloseA).Take(3).ToArray());
    }

    [Fact]
    public async Task WithoutThePolicy_AStopInsideReasoningEndsTheTurnAsBefore()
    {
        using var model = new Reasoner { StopAfterThoughts = 5 };
        using var engine = Engine(model);
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(int.MaxValue, End, closeOnRepetition: true,
            closingTokenIds: new[] { CloseA, CloseB });
        var handle = engine.SubmitRequest(Sequence("no-policy", config, 32));
        Assert.Equal(Enumerable.Repeat(Thought, 5).ToArray(), await Collect(handle));
        Assert.Equal("eos", (await handle.Completion).FinishReason);
    }

    [Fact]
    public void TheSamplerClosesOnlyAnOpenChannel_AndNeverPastAnActiveGrammar()
    {
        var sampler = new TokenSampler(Policy());
        var history = new List<int> { Thought, Thought };
        Assert.Equal(CloseA, sampler.CloseInsteadOfStop(history, Eos));
        Assert.Equal(1, sampler.StopsClosedInThinking);
        Assert.Equal(Thought, sampler.CloseInsteadOfStop(history, Thought));   // any other draw is kept
        history.AddRange(new[] { CloseA, CloseB, End, Answer });
        Assert.Equal(Eos, sampler.CloseInsteadOfStop(history, Eos));           // closed: a stop is a stop
        Assert.Equal(1, sampler.StopsClosedInThinking);

        var tokenizer = new LineTokenizer();
        var grammar = Policy();
        grammar.Grammar = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        Assert.Equal(Eos, new TokenSampler(grammar).CloseInsteadOfStop(new List<int> { Thought }, Eos));
    }

    [Theory]
    [InlineData("TakePendingOrSample")]
    [InlineData("PeekPendingOrSample")]
    public void AStopTheDeviceDrewIsClosedLikeAHostSample(string method)
    {
        var seq = Sequence("pending", Policy(), 32);
        seq.AppendOutputToken(Thought);
        seq.AppendOutputToken(Thought);
        seq.PendingDeviceToken = Eos;
        seq.PendingDevicePosition = seq.NumComputedTokens;
        Assert.True(seq.GetOrCreateSampler().IsPlainGreedyArgmax);
        var draw = typeof(BatchExecutor).GetMethod(method, BindingFlags.Static | BindingFlags.NonPublic)!;
        Assert.Equal(CloseA, (int)draw.Invoke(null, new object[] { seq })!);
        Assert.Null(seq.LastLogits);
        // A peek's close is history-derived: the draw that follows it returns the same token.
        if (method == "PeekPendingOrSample")
            Assert.Equal(CloseA, (int)typeof(BatchExecutor).GetMethod("TakePendingOrSample",
                BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, new object[] { seq })!);
        Assert.Equal(1, seq.GetOrCreateSampler().StopsClosedInThinking);
    }

    /// <summary>A close requested after a speculative step (the loop detector and the repetition guard ask
    /// at the step's end) replaces the token that step's verify already drew for the next position: it was
    /// never forwarded, and emitting it first would skip the hand-over text.</summary>
    [Fact]
    public void ACloseRequestedAfterAVerify_ReplacesItsPendingDraw()
    {
        var seq = Sequence("verify", Policy(), 32);
        for (int i = 0; i < 4; i++)
            seq.AppendOutputToken(Thought);
        var draw = typeof(BatchExecutor).GetMethod("PendingVerifyDraw", BindingFlags.Static | BindingFlags.NonPublic)!;
        Assert.Equal(Thought, (int)draw.Invoke(null, new object[] { seq, Thought })!);
        Assert.True(seq.GetOrCreateSampler().TryRequestThinkingClosure(seq.OutputTokens));
        Assert.Equal(CloseA, (int)draw.Invoke(null, new object[] { seq, Thought })!);
        // A second request while the close is under way keeps its place.
        seq.AppendOutputToken(CloseA);
        Assert.True(seq.GetOrCreateSampler().TryRequestThinkingClosure(seq.OutputTokens));
        Assert.Equal(CloseB, (int)draw.Invoke(null, new object[] { seq, Thought })!);
    }

    [Theory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    [InlineData("deepseek41")]
    [InlineData("glm5next")]
    public void TheHostInstallsTheClose_EvenWithNoThinkingCap(string architecture)
    {
        var tokenizer = new LineTokenizer();
        var config = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, architecture, 0,
            out bool installed);
        Assert.True(installed);
        Assert.Equal(int.MaxValue, config.ThinkingBudget!.TokenLimit);
        Assert.True(config.ThinkingBudget.CloseOnRepetition);
        Assert.True(config.ThinkingBudget.CloseOnStop!(Eos));
        Assert.False(config.ThinkingBudget.CloseOnStop!(Thought));
        // Thinking off with no cap: a block the prompt closed is still left alone.
        Assert.Null(ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, architecture, 0,
            out _, enableThinking: false).ThinkingBudget);
    }

    // ---- Reasoning that settles into a templated loop -----------------------------------------------

    public static IEnumerable<object[]> IssueLoops() => new[]
    {
        new object[] { "clock", (Func<int, string>)(k => $"Need maybe tests expect to_string for 1:{1500 + k} -> 02:{k % 60:00}. Good.") },
        new object[] { "say", (Func<int, string>)(k => $"Need maybe tests expect \"three hundred {Words(21 + k)}\" for {321 + k}. Good.") },
        new object[] { "go-counting", (Func<int, string>)(k => $"- \"Can identify territory owner for a complex board {100 + k}\"") },
    };

    [Theory]
    [MemberData(nameof(IssueLoops))]
    public void TheIssuesLoopsAreFound_AtTheMinimumRun(string name, Func<int, string> line)
    {
        var detector = new ReasoningLoopDetector();
        for (int k = 0; k < ReasoningLoopDetector.MinLines; k++)
        {
            Assert.False(detector.Looping, $"{name}: looping after only {k} lines");
            Feed(detector, line(k) + "\n");
        }
        Assert.True(detector.Looping, $"{name}: not looping after {ReasoningLoopDetector.MinLines} lines");
    }

    [Fact]
    public void BlankAndShortLines_NeitherExtendNorBreakARun_AndTokensMaySplitLines()
    {
        var detector = new ReasoningLoopDetector();
        var text = new StringBuilder();
        for (int k = 0; k < ReasoningLoopDetector.MinLines; k++)
            text.Append($"Case {k}: input {k * 7}, expected {k * k}.\n\nGood.\n---\n");
        var random = new Random(338);
        byte[] bytes = Encoding.UTF8.GetBytes(text.ToString());
        for (int i = 0; i < bytes.Length;)
        {
            int n = Math.Min(bytes.Length - i, random.Next(1, 9));
            detector.Observe(bytes.AsSpan(i, n));
            i += n;
        }
        Assert.True(detector.Looping);
        Assert.Equal(ReasoningLoopDetector.MinLines, detector.RunLines);
    }

    [Fact]
    public void OrdinaryReasoningAndAShortChecklist_AreNotLoops()
    {
        var detector = new ReasoningLoopDetector();
        foreach (string line in OrdinaryReasoning)
            Feed(detector, line + "\n");
        Assert.True(detector.RunLines <= 2, $"ordinary reasoning ran {detector.RunLines} lines");
        for (int k = 1; k <= 20; k++)
            Feed(detector, $"Check n={k}: {k}*{k} = {k * k}.\n");
        Assert.False(detector.Looping);
        Feed(detector, "So the formula holds for every case we tried, and the edge case n=0 is handled.\n");
        Assert.Equal(1, detector.RunLines);
    }

    [Fact]
    public void TheBandedEditDistance_AgreesWithTheFullOne()
    {
        static int Full(string a, string b)
        {
            var prev = Enumerable.Range(0, b.Length + 1).ToArray();
            for (int i = 1; i <= a.Length; i++)
            {
                var cur = new int[b.Length + 1];
                cur[0] = i;
                for (int j = 1; j <= b.Length; j++)
                    cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
                prev = cur;
            }
            return prev[b.Length];
        }
        var random = new Random(338);
        const string alphabet = "ab# c-";
        for (int t = 0; t < 20000; t++)
        {
            string a = new(Enumerable.Range(0, random.Next(0, 60)).Select(_ => alphabet[random.Next(alphabet.Length)]).ToArray());
            var b = new StringBuilder(a);
            for (int e = random.Next(0, 16); e > 0; e--)
            {
                int op = random.Next(3);
                if (op == 0 && b.Length > 0) b.Remove(random.Next(b.Length), 1);
                else if (op == 1) b.Insert(random.Next(b.Length + 1), alphabet[random.Next(alphabet.Length)]);
                else if (b.Length > 0) b[random.Next(b.Length)] = alphabet[random.Next(alphabet.Length)];
            }
            string bs = b.ToString();
            bool expected = Full(a, bs) <= Math.Max(2, Math.Max(a.Length, bs.Length) / 5);
            Assert.True(expected == ReasoningLoopDetector.Similar(a, bs), $"'{a}' vs '{bs}'");
        }
    }

    [Fact]
    public async Task ALoopingChainOfThought_IsClosedAtTheMinimumRun_ThenTheModelAnswers()
    {
        using var model = new Reasoner { LoopLines = true };
        using var engine = Engine(model);
        var handle = engine.SubmitRequest(Sequence("loop", Policy(), 128));
        int[] output = await Collect(handle);
        int lines = ReasoningLoopDetector.MinLines;
        Assert.Equal(Enumerable.Range(FirstLine, lines).ToArray(), output.Take(lines).ToArray());
        Assert.Equal(new[] { CloseA, CloseB, End, Answer }, output.Skip(lines).ToArray());
        Assert.Equal("eos", (await handle.Completion).FinishReason);
    }

    [Fact]
    public async Task TemplatedLinesInTheAnswer_AreNeitherClosedNorStopped()
    {
        using var model = new Reasoner { LoopLines = true, CloseAtOnce = true };
        using var engine = Engine(model);
        var handle = engine.SubmitRequest(Sequence("content", Policy(), 80));
        int[] output = await Collect(handle);
        Assert.Equal(80, output.Length);
        Assert.Equal(End, output[0]);
        Assert.All(output.Skip(1), t => Assert.True(t >= FirstLine));
        Assert.Equal("max_tokens", (await handle.Completion).FinishReason);
    }

    [Fact]
    public async Task TheRepetitionOptOut_AlsoTurnsOffTheLoopClose()
    {
        using var model = new Reasoner { LoopLines = true };
        using var engine = Engine(model, stopRepetition: false);
        var handle = engine.SubmitRequest(Sequence("opt-out", Policy(), 64));
        int[] output = await Collect(handle);
        Assert.Equal(Enumerable.Range(FirstLine, 64).ToArray(), output);
        Assert.Equal("max_tokens", (await handle.Completion).FinishReason);
    }

    // ---- A turn that still ends with reasoning and no answer ----------------------------------------

    [Theory]
    [InlineData("Let me count the cases.\nCase 1 -> one. Good.\nCase 2 -> two. Good.\nCase", true)]
    [InlineData("Let me count the cases.\n</think>\n\nThe answer is 4.", false)]
    [InlineData("Let me count the cases.\n</think>\n\n", true)]
    public void AnAnswerlessTurn_IsRecognisedFromTheParsedReply(string raw, bool answerless)
    {
        var turn = ChatGenerationPipeline.BuildEmittedTurn("qwen35", raw, enableThinking: true, tools: null,
            generationSuffix: "<think>\n", cancelled: false);
        Assert.False(string.IsNullOrWhiteSpace(turn.Thinking));
        Assert.Equal(answerless, ChatGenerationPipeline.IsAnswerlessReasoning(turn));
    }

    [Fact]
    public void ReasoningIsReportedAsALengthStop()
    {
        Assert.Equal("length", TensorSharp.Server.ProtocolAdapters.FinishReasonMapper.ToOpenAIChat("thinking_budget", false));
        Assert.Equal("stop", TensorSharp.Server.ProtocolAdapters.FinishReasonMapper.ToOpenAIChat("eos", false));
    }

    // ---- Fixtures -----------------------------------------------------------------------------------

    private static readonly string[] OrdinaryReasoning =
    {
        "The user wants a function that converts a number into English words.",
        "First, handle zero separately, since every other branch assumes a positive value.",
        "Numbers below twenty have irregular names, so a lookup table is simplest there.",
        "Tens from twenty to ninety follow a pattern but still need their own words.",
        "A hyphen joins the tens and the ones, as in forty-two.",
        "For hundreds, write the digit's word, then \"hundred\", then the remainder if any.",
        "Thousands, millions and billions repeat the same three-digit grouping.",
        "So the core is a helper that formats any value below one thousand.",
        "The outer loop walks the scale words from the largest down.",
        "Edge cases: exactly one thousand, one million, and the maximum u64 value.",
        "The tests probably check negative input too; the signature takes u64, so no.",
        "Let me make sure the spacing is right when a group is zero, like 1,000,005.",
        "Skipping empty groups avoids double spaces in the output.",
        "Now the Rust specifics: return a String and build it with push_str.",
        "A match on the value is cleaner than nested if statements.",
        "I should avoid recursion depth issues, though the depth here is at most seven.",
        "Performance is irrelevant at this size; clarity matters more.",
        "Double-check: 14 is fourteen, not \"ten four\".",
        "And 100 must print \"one hundred\", with no trailing space.",
        "That covers the behaviour; now write the code and the edit block.",
    };

    private static string Words(int n)
    {
        string[] ones = { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
        string[] tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
        return n % 10 == 0 ? tens[n / 10] : $"{tens[n / 10]}-{ones[n % 10]}";
    }

    private static void Feed(ReasoningLoopDetector detector, string text) => detector.Observe(Encoding.UTF8.GetBytes(text));

    private static SamplingConfig Policy()
    {
        var config = SamplingConfig.Greedy;
        config.ThinkingBudget = new ThinkingTokenBudget(int.MaxValue, End, closeOnRepetition: true,
            closingTokenIds: new[] { CloseA, CloseB }, closeOnStop: t => t == Eos);
        return config;
    }

    private static SequenceState Sequence(string id, SamplingConfig config, int maxTokens) =>
        new(id, new[] { 1, 1 }, maxTokens, 8, config);

    private static InferenceEngine Engine(Reasoner model, bool stopRepetition = true) => new(model, new SchedulerConfig
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

    /// <summary>Single-token specials, a few pieces, and one whole line per loop token (a counter).</summary>
    private sealed class LineTokenizer : ITokenizer
    {
        public string[] Vocab { get; } = new[] { "<eos>", "<think>", "</think>", "Reasoning ", "Answer", "Close ", "now." }
            .Concat(Enumerable.Range(0, Lines).Select(k => $"Need maybe tests expect to_string for 1:{1500 + k} -> 02:{k % 60:00}. Good.\n"))
            .ToArray();
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { Eos };
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => new() { LookupToken(text) };
        public string Decode(List<int> ids) => string.Concat(ids.Select(i => Vocab[i]));
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == Eos;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }

    /// <summary>
    /// Reasons until the channel is closed, then answers and stops. Its reasoning is either plain
    /// thoughts ending in a stop token (<see cref="StopAfterThoughts"/>) or a counter, one line a token
    /// (<see cref="LoopLines"/>). <see cref="CloseAtOnce"/> closes first, so the lines are the answer.
    /// </summary>
    private sealed class Reasoner : IModelArchitecture
    {
        private readonly List<int> _state = new();
        public List<int> Forwarded { get; } = new();
        public int StopAfterThoughts { get; init; } = int.MaxValue;
        public bool LoopLines { get; init; }
        public bool CloseAtOnce { get; init; }
        public ModelConfig Config { get; } = new() { Architecture = "qwen35", VocabSize = FirstLine + Lines };
        public ITokenizer Tokenizer { get; } = new LineTokenizer();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "reasoning-closure-test";

        public float[] Forward(int[] tokens)
        {
            _state.AddRange(tokens);
            Forwarded.AddRange(tokens);
            int lines = _state.Count(t => t >= FirstLine);
            int next;
            if (CloseAtOnce && !_state.Contains(End))
                next = End;
            else if (_state.Contains(End) && !CloseAtOnce)
                next = _state[^1] == Answer ? Eos : Answer;
            else if (LoopLines)
                next = FirstLine + lines % Lines;
            else
                next = _state.Count(t => t == Thought) >= StopAfterThoughts ? Eos : Thought;
            var logits = new float[FirstLine + Lines];
            logits[next] = 20;
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
