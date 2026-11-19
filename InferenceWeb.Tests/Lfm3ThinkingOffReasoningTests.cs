// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LFM3-1.7B with thinking off ends its prompt with "<think></think>", and still reasons in about a
// quarter of its math replies, then closes the block itself and answers (52 of 200 GSM8K replies,
// 5 of 33 instruction prompts on the Q4_K_M, llama.cpp's greedy replies identical; eng/lfm3/e2e).
// Parsed from Init's state the reasoning and the literal "</think>" were the answer the client
// read. The replies below are verbatim greedy replies of LFM3-1.7B-Q4_K_M (ggml_metal).
using System.Text;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

public sealed class Lfm3ThinkingOffReasoningTests
{
    private const string Arch = "lfm3";
    private const string Closed = "<think></think>";

    // "Name three countries in South America, one per line." (thinking off)
    private const string CountriesReply = "Brazil\nArgentina\nChile</think>Brazil\nArgentina\nChile";
    // "How many legs does a spider have? Reply with just the number in digits."
    private const string SpiderReply = "0? No, spiders have eight legs. The question asks \"How many legs does a spider have?\" " +
        "and the answer is 8. The instruction says \"Reply with just the number in digits.\" So I should output only " +
        "the number. Thus, 8.</think>8";
    // "What is the chemical symbol for gold? Answer with one word only." - the reply opens a block of its own.
    private const string GoldReply = "<think>We need to answer the question. The chemical symbol for gold is \"Au\". " +
        "The answer should be just \"Au\". So I'll output \"Au\".</think>Au";
    private const string PlainReply = "The capital of Australia is **Canberra**.";
    // GSM8K test id 38 ("How fast does he run?", thinking off, max_tokens 1,024, ggml_cpu): the result is stated only in
    // the reasoning, and max_tokens cut the answer after the close. The reply's first and last sentences verbatim, its
    // middle (3,300 characters) left out.
    private const string CutReply = "We are given: John runs 60 miles a week. He runs 3 days a week. " +
        "Total hours = 6. Total distance = 60 miles. So speed = 60/6 = 10 miles per hour.\n\nThus final answer: 10.\n\n" +
        "Thus: Answer: 10\n\nWe'll put in the required format.</think>John runs a total of 60 miles";

    private static readonly List<ToolFunction> Tools =
    [
        new()
        {
            Name = "get_weather",
            Parameters = new Dictionary<string, ToolParameter> { ["city"] = new() { Type = "string" } },
            Required = ["city"],
        },
    ];

    private static string Template(string name)
    {
        using Stream stream = typeof(Lfm3ThinkingOffReasoningTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Lfm3." + name)!;
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return reader.ReadToEnd();
    }

    private static (string Content, string Thinking, List<ToolCall> Calls, int Retractions) Drive(
        string reply, int chunk, bool retract, string? suffix = Closed, List<ToolFunction>? tools = null,
        bool appendOnlyStream = false, List<string>? contentPieces = null, string arch = Arch)
    {
        IOutputParser parser = ChatProtocolRegistry.For(arch)!.CreateOutputParser!();
        parser.Init(false, tools ?? Tools);
        if (retract) parser.AcceptRetractions();
        if (appendOnlyStream) parser.StreamWithoutHolding();
        parser.SetGenerationPromptSuffix(suffix);
        var content = new StringBuilder();
        var thinking = new StringBuilder();
        var calls = new List<ToolCall>();
        int retractions = 0;
        void Apply(ParsedOutput p)
        {
            if (!string.IsNullOrEmpty(p.RetractedContent))
            {
                Assert.EndsWith(p.RetractedContent, content.ToString());
                content.Length -= p.RetractedContent.Length;
                retractions++;
            }
            content.Append(p.Content);
            contentPieces?.Add(p.Content ?? "");
            thinking.Append(p.Thinking);
            if (p.ToolCalls != null) calls.AddRange(p.ToolCalls);
        }
        int step = chunk <= 0 ? reply.Length : chunk;
        for (int i = 0; i < reply.Length; i += step)
            Apply(parser.Add(reply.Substring(i, Math.Min(step, reply.Length - i)), false));
        Apply(parser.Add("", true));
        return (content.ToString(), thinking.ToString(), calls, retractions);
    }

    [Theory]
    [InlineData(CountriesReply, "Brazil\nArgentina\nChile", "Brazil\nArgentina\nChile")]
    [InlineData(SpiderReply, "8", null)]
    [InlineData(GoldReply, "Au", null)]
    public void AStrayClose_LeavesTheAnswerAlone(string reply, string answer, string? reasoning)
    {
        foreach (int chunk in new[] { 1, 3, 7, 0 })
        {
            foreach (bool retract in new[] { false, true })
            {
                var run = Drive(reply, chunk, retract);
                Assert.Equal(answer, run.Content.Trim());
                Assert.DoesNotContain("</think>", run.Content);
                Assert.DoesNotContain("<think>", run.Thinking);
                Assert.False(string.IsNullOrWhiteSpace(run.Thinking));
                if (reasoning != null) Assert.Equal(reasoning, run.Thinking.Trim());
                if (!retract) Assert.Equal(0, run.Retractions);
            }
        }
    }

    [Fact]
    public void AnOrdinaryAnswer_IsUnchanged_AndAPageSeesItAsEarlyAsBefore()
    {
        foreach (int chunk in new[] { 1, 5, 0 })
        {
            var held = Drive(PlainReply, chunk, retract: false);
            var shown = Drive(PlainReply, chunk, retract: true);
            var unprimed = Drive(PlainReply, chunk, retract: false, suffix: null);
            Assert.Equal(PlainReply, held.Content);
            Assert.Equal(PlainReply, shown.Content);
            Assert.Equal(PlainReply, unprimed.Content);
            Assert.Empty(held.Thinking);
            Assert.Equal(0, shown.Retractions);
        }
    }

    /// <summary>
    /// An append-only stream (the OpenAI, Ollama and Responses streams, the CLI) is not held while a thinking-off reply is
    /// undecided: it gets the reply as content as it is generated, a stray close and all, as llama.cpp, vLLM and SGLang
    /// stream it; the first piece arrives with the first characters, where it used to wait for the close, the end of the
    /// reply or 4,096 characters. A whole reply and a retracting stream still keep the reasoning apart.
    /// </summary>
    [Theory]
    [InlineData(CountriesReply)]
    [InlineData(SpiderReply)]
    [InlineData(PlainReply)]
    public void AnAppendOnlyStreamGetsTheReplyAsItIsGenerated(string reply)
    {
        foreach (int chunk in new[] { 1, 3, 7 })
        {
            var pieces = new List<string>();
            var streamed = Drive(reply, chunk, retract: false, appendOnlyStream: true, contentPieces: pieces);
            Assert.Equal(reply, streamed.Content);
            Assert.Empty(streamed.Thinking);
            Assert.Equal(0, streamed.Retractions);
            // Nothing is held past a possible partial tag: the first piece's text is out after at most "</think>".Length characters.
            int firstShown = pieces.FindIndex(p => p.Length > 0);
            Assert.InRange(firstShown * chunk, 0, "</think>".Length);

            // The same reply whole, and to a stream that can take text back: the reasoning is told apart as before.
            var retracting = Drive(reply, chunk, retract: true, appendOnlyStream: true);
            var whole = Drive(reply, 0, retract: false, appendOnlyStream: false);
            Assert.Equal(whole.Content, retracting.Content);
            Assert.Equal(whole.Thinking, retracting.Thinking);
        }
    }

    /// <summary>A reply that opens a block of its own is reasoning from its first token, so an append-only stream gets it
    /// as reasoning at once (nothing to hold), and a call is a call.</summary>
    [Fact]
    public void AnAppendOnlyStreamStillSeparatesABlockTheModelOpens_AndParsesCalls()
    {
        var gold = Drive(GoldReply, 3, retract: false, appendOnlyStream: true);
        Assert.Equal("Au", gold.Content.Trim());
        Assert.StartsWith("We need to answer the question.", gold.Thinking);
        const string call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        var run = Drive("Checking.\n" + call, 5, retract: false, appendOnlyStream: true);
        Assert.Equal("Paris", Assert.Single(run.Calls).Arguments["city"]);
        Assert.Equal("Checking.", run.Content.Trim());
    }

    /// <summary>Nemotron-H keeps holding append-only streams (its policy is its own; ChatProtocolRegistry).</summary>
    [Fact]
    public void NemotronHStillHoldsAnAppendOnlyStream()
    {
        const string reply = "Reasoning about it.</think>The answer.";
        var pieces = new List<string>();
        var run = Drive(reply, 4, retract: false, suffix: "<think></think>", appendOnlyStream: true, contentPieces: pieces,
            arch: "nemotron_h");
        Assert.Equal("The answer.", run.Content.Trim());
        Assert.Equal("Reasoning about it.", run.Thinking.Trim());
        Assert.DoesNotContain(pieces.Take(4), p => p.Length > 0);
    }

    [Fact]
    public void ACloseThatStartsPastTheWindow_IsTheAnswersOwnText()
    {
        string longAnswer = new string('x', ChatProtocolRegistry.Lfm3StrayReasoningWindow) + "</think>tail";
        var run = Drive(longAnswer, 64, retract: false);
        Assert.Equal(longAnswer, run.Content);
        Assert.Empty(run.Thinking);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void AThinkingOffCall_IsStillACall(int chunk)
    {
        const string call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        var direct = Drive(call, chunk, retract: false);
        Assert.Equal("Paris", Assert.Single(direct.Calls).Arguments["city"]);
        Assert.True(string.IsNullOrWhiteSpace(direct.Content));

        var reasoned = Drive("The user wants the weather. Use the tool.</think>" + call, chunk, retract: false);
        Assert.Equal("Paris", Assert.Single(reasoned.Calls).Arguments["city"]);
        Assert.Equal("The user wants the weather. Use the tool.", reasoned.Thinking.Trim());
    }

    [Fact]
    public void ThePublishedTemplates_TurnTheWindowOn()
    {
        Assert.True(OutputParserFactory.ThinkingOffReplyMayReason(Arch, Template("text_chat_template.jinja")));
        Assert.True(OutputParserFactory.ThinkingOffReplyMayReason(Arch, Template("omni_chat_template.jinja")));
        Assert.False(OutputParserFactory.ThinkingOffReplyMayReason(Arch, null));
        Assert.True(OutputParserFactory.IsAlwaysRequired(Arch, Template("text_chat_template.jinja")));
        Assert.False(OutputParserFactory.IsAlwaysRequired(Arch, null));
    }

    [Theory]
    [InlineData(false, Closed)]
    [InlineData(true, "<think>")]
    public void TheRecordedSuffix_IsWhatThePromptEndsWith(bool thinking, string expected)
    {
        string template = Template("text_chat_template.jinja");
        var tokenizer = new CharTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "How many legs does a spider have?" } };
        List<int> prompt = renderer.RenderToTokens(tokenizer, template, messages, Arch, true, enableThinking: thinking);
        Assert.EndsWith("assistant\n" + expected, tokenizer.Decode(prompt));
        Assert.Equal(expected, ChatGenerationPipeline.RecordedGenerationSuffix(tokenizer, prompt, Arch, thinking, template));
    }

    [Fact]
    public void TheTranscriptKeepsTheAnswerAndTheRawReply()
    {
        EmittedAssistantTurn turn = ChatGenerationPipeline.BuildEmittedTurn(
            Arch, SpiderReply, false, Tools, generationSuffix: Closed, cancelled: false);
        Assert.Equal("8", turn.Content.Trim());
        Assert.Equal(SpiderReply, turn.RawText);
        // What an append-only stream (and a whole reply cut short) sent instead.
        Assert.Equal(SpiderReply, turn.StreamedContent);

        EmittedAssistantTurn plain = ChatGenerationPipeline.BuildEmittedTurn(
            Arch, PlainReply, false, Tools, generationSuffix: Closed, cancelled: false);
        Assert.Null(plain.StreamedContent);
    }

    // ---------------------------------------------------------------- a whole reply cut short

    /// <summary>
    /// A whole reply cut short (max_tokens) after a stray close reads as its append-only stream did: the answer after the
    /// close is incomplete, and the client keeps the reasoning that states the result, as llama.cpp, vLLM and SGLang return
    /// it. On GSM8K (first 200, thinking off, Q4_K_M) this was 5 replies on ggml_cpu and 7 on the managed CPU, all cut so.
    /// The same reply finished (or a whole reply the parse is not told was cut) keeps the reasoning apart, as before.
    /// </summary>
    [Fact]
    public void AWholeReplyCutInItsAnswer_ReadsAsItsStream()
    {
        ParsedOutput cut = ChatStreamCollector.ParseWhole(Arch, CutReply, false, Tools, Closed, asStreamed: true);
        Assert.Equal(CutReply, cut.Content);
        Assert.True(string.IsNullOrEmpty(cut.Thinking));

        ParsedOutput finished = ChatStreamCollector.ParseWhole(Arch, CutReply, false, Tools, Closed, asStreamed: false);
        Assert.Equal("John runs a total of 60 miles", finished.Content);
        Assert.EndsWith("We'll put in the required format.", finished.Thinking);

        // The non-streaming endpoints' collector: told by the finish reason.
        foreach (bool replyWasCut in new[] { true, false })
        {
            var collector = new ChatStreamCollector();
            collector.Add(new ChatStreamUpdate { Piece = CutReply[..100], RawGenerationSuffix = Closed });
            collector.Add(ChatStreamUpdate.Text(CutReply[100..]));
            ParsedOutput resolved = collector.Resolve(Arch, false, Tools, replyWasCut);
            Assert.Equal(replyWasCut ? cut.Content : finished.Content, resolved.Content);
            Assert.Equal(replyWasCut ? cut.Thinking : finished.Thinking, resolved.Thinking);
        }
        Assert.True(FinishReasonMapper.IsTruncated(FinishReasonMapper.PipelineMaxTokens));
    }

    /// <summary>Reading a cut reply as its stream changes only stray reasoning: a block the model opened itself is still
    /// reasoning, a call still a call, an ordinary answer and a thinking-on reply parse the same, and Nemotron-H (which
    /// holds its streams) still separates its stray reasoning.</summary>
    [Fact]
    public void ReadingACutReplyAsItsStream_ChangesOnlyStrayReasoning()
    {
        const string goldCut = "<think>We need to answer the question. The chemical symbol for gold is \"Au\".</think>A";
        ParsedOutput gold = ChatStreamCollector.ParseWhole(Arch, goldCut, false, Tools, Closed, asStreamed: true);
        Assert.Equal("A", gold.Content);
        Assert.StartsWith("We need to answer the question.", gold.Thinking);

        const string call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        ParsedOutput withCall = ChatStreamCollector.ParseWhole(Arch, "Checking.\n" + call + "\nMore", false, Tools, Closed, asStreamed: true);
        Assert.Equal("Paris", Assert.Single(withCall.ToolCalls!).Arguments["city"]);

        foreach (bool asStreamed in new[] { true, false })
            Assert.Equal(PlainReply, ChatStreamCollector.ParseWhole(Arch, PlainReply, false, Tools, Closed, asStreamed).Content);

        const string thinkingOn = "Reasoning about it.</think>The answ";
        ParsedOutput onStreamed = ChatStreamCollector.ParseWhole(Arch, thinkingOn, true, Tools, "<think>", asStreamed: true);
        ParsedOutput onWhole = ChatStreamCollector.ParseWhole(Arch, thinkingOn, true, Tools, "<think>", asStreamed: false);
        Assert.Equal(onWhole.Content, onStreamed.Content);
        Assert.Equal(onWhole.Thinking, onStreamed.Thinking);
        Assert.Equal("Reasoning about it.", onStreamed.Thinking.Trim());

        const string nemotron = "Reasoning about it.</think>The answ";
        ParsedOutput held = ChatStreamCollector.ParseWhole("nemotron_h", nemotron, false, null, Closed, asStreamed: true);
        Assert.Equal("The answ", held.Content.Trim());
        Assert.Equal("Reasoning about it.", held.Thinking.Trim());
    }

    /// <summary>
    /// The next request sends back whichever form the client got - the reasoning apart (a page, a whole reply the model
    /// finished) or the reply as generated (an API stream, a whole reply cut short), prose before a call included - and
    /// either one splices the turn's generated tokens back, so its KV cache is continued. Edited text still is not.
    /// </summary>
    [Fact]
    public void TheTranscriptMatchesEitherFormTheClientWasSent()
    {
        const string call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        const string reasoning = "The user wants the weather. Use the tool.";
        string raw = reasoning + "</think>\n" + call;
        EmittedAssistantTurn emitted = ChatGenerationPipeline.BuildEmittedTurn(Arch, raw, false, Tools, Closed, cancelled: false);
        Assert.Equal(reasoning, emitted.Thinking.Trim());
        Assert.True(string.IsNullOrWhiteSpace(emitted.Content));
        Assert.Equal(reasoning + "</think>", emitted.StreamedContent!.TrimEnd());

        var store = new ConversationTranscriptStore(maxChains: 16, maxTokens: 10_000);
        var ask = new ChatMessage { Role = "user", Content = "Weather in Paris?" };
        store.Record(new List<ChatMessage> { ask },
            new ChatMessage { Role = "assistant", Content = raw, RawOutputTokens = new List<int> { 7, 8, 9 } },
            emitted, scope: "lfm3");

        List<ChatMessage> Replay(string content, string? thinking) => new()
        {
            new() { Role = "user", Content = ask.Content },
            new()
            {
                Role = "assistant", Content = content, Thinking = thinking,
                ToolCalls = new List<ToolCall>
                {
                    new() { Id = "call_1", Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Paris" } },
                },
            },
            new() { Role = "tool", Content = "18C, clear", ToolCallId = "call_1" },
        };

        Assert.Equal(new[] { 7, 8, 9 }, store.Augment(Replay("", reasoning)).History[1].RawOutputTokens);
        // As the OpenAI stream sent it: the reasoning and the close as content, the blank line before the call included.
        Assert.Equal(new[] { 7, 8, 9 }, store.Augment(Replay(reasoning + "</think>\n", null)).History[1].RawOutputTokens);
        Assert.Equal(new[] { 7, 8, 9 }, store.Augment(Replay(reasoning + "</think>", null)).History[1].RawOutputTokens);
        Assert.Null(store.Augment(Replay("The user wants rain. Use the tool.</think>", null)).History[1].RawOutputTokens);

        // A reply cut short, returned as generated, comes back as its raw text.
        EmittedAssistantTurn cut = ChatGenerationPipeline.BuildEmittedTurn(Arch, CutReply, false, Tools, Closed, cancelled: false);
        var cutStore = new ConversationTranscriptStore(maxChains: 16, maxTokens: 10_000);
        cutStore.Record(new List<ChatMessage> { ask },
            new ChatMessage { Role = "assistant", Content = CutReply, RawOutputTokens = new List<int> { 4, 5 } },
            cut, scope: "lfm3");
        List<ChatMessage> Next(string content, string? thinking) => new()
        {
            new() { Role = "user", Content = ask.Content },
            new() { Role = "assistant", Content = content, Thinking = thinking },
            new() { Role = "user", Content = "And in Rome?" },
        };
        Assert.Equal(new[] { 4, 5 }, cutStore.Augment(Next(CutReply, null)).History[1].RawOutputTokens);
        Assert.Equal(new[] { 4, 5 }, cutStore.Augment(Next(cut.Content, cut.Thinking)).History[1].RawOutputTokens);
        // The stream form went out with no reasoning apart.
        Assert.Equal(CutReply, cut.StreamedContent);
        Assert.Equal("", cut.StreamedThinking);
    }

    /// <summary>
    /// A tool-loop turn whose last round went out as generated (a whole reply cut short, or an API stream): the client got
    /// the earlier rounds' reasoning apart and none of the last round's, and sends back that reasoning or none. Either
    /// expands the turn into the loop's rounds, so the next request continues its KV cache. Before, the reasoning was
    /// compared with every round's split reasoning, the last round's included, and the turn re-prefilled. The stream form
    /// beside the split form's reasoning (the last round's stray reasoning twice) was never sent and does not expand.
    /// </summary>
    [Fact]
    public void ALoopTurnWhoseLastRoundWentOutAsGenerated_MatchesTheReasoningSentWithIt()
    {
        const string plan = "Plan: look the weather up first.";
        var ask = new ChatMessage { Role = "user", Content = "Weather in Paris?" };
        var lookup = new ToolCall { Id = "call_1", Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Paris" } };
        List<int> round1 = new() { 1, 2, 3 }, round2 = new() { 4, 5 };
        var store = new ConversationTranscriptStore(maxChains: 16, maxTokens: 10_000);
        EmittedAssistantTurn last = ChatGenerationPipeline.BuildEmittedTurn(Arch, CutReply, false, Tools, Closed, cancelled: false);
        store.Record(new List<ChatMessage>
            {
                ask,
                new() { Role = "assistant", Content = "Checking.", Thinking = plan, ToolCalls = new List<ToolCall> { lookup }, RawOutputTokens = round1 },
                new() { Role = "tool", Content = "18C, clear", ToolCallId = "call_1" },
            },
            new ChatMessage { Role = "assistant", Content = CutReply, RawOutputTokens = round2 }, last, scope: "lfm3");

        List<ChatMessage> Next(string content, string? thinking) => new()
        {
            new() { Role = "user", Content = ask.Content },
            new() { Role = "assistant", Content = content, Thinking = thinking },
            new() { Role = "user", Content = "And in Rome?" },
        };

        // What the client got: the first round's content and reasoning, the last round as generated.
        foreach (string? thinking in new[] { plan, null })
        {
            List<ChatMessage> expanded = store.Augment(Next("Checking." + CutReply, thinking)).History;
            Assert.Equal(5, expanded.Count);
            Assert.Same(round1, expanded[1].RawOutputTokens);
            Assert.Same(round2, expanded[3].RawOutputTokens);
        }
        // The split form with every round's reasoning, as before.
        Assert.Equal(5, store.Augment(Next("Checking." + last.Content, plan + last.Thinking)).History.Count);
        // Never sent: the stream form beside the last round's split reasoning.
        Assert.Equal(3, store.Augment(Next("Checking." + CutReply, plan + last.Thinking)).History.Count);
    }

    /// <summary>The Responses API's reasoning_tokens counts a reply cut short as its whole reply reads: the stray reasoning
    /// went out as content, so none of it is reasoning. A reply the model finished counts it, as before.</summary>
    [Fact]
    public void AReplyCutShort_CountsNoStrayReasoningTokens()
    {
        var tokenizer = new TagTokenizer();
        List<int> tokens = tokenizer.Encode(CutReply);
        Assert.Equal(0, ReasoningTokenCount.Count(Arch, tokenizer, tokens, false, Tools, Closed, replyWasCut: true));
        int finished = ReasoningTokenCount.Count(Arch, tokenizer, tokens, false, Tools, Closed, replyWasCut: false);
        Assert.Equal(tokenizer.Encode(CutReply[..(CutReply.IndexOf("</think>", StringComparison.Ordinal) + 8)]).Count, finished);
    }

    /// <summary>
    /// With thinking off the 1.7B sometimes writes <c>&lt;think&gt;</c> itself after the prompt's
    /// closed block and reasons until max_tokens (an IFEval-style "essay of at least 150 words"
    /// prompt came back empty). The opener is declared, so such a block is capped like Gemma 4's
    /// unrequested channel; a requested block, opened by the prompt, keeps its ordinary budget.
    /// </summary>
    [Fact]
    public void ABlockTheModelOpensWithThinkingOff_IsCapped_ARequestedOneKeepsItsBudget()
    {
        var tokenizer = new TagTokenizer();
        int open = tokenizer.LookupToken("<think>"), close = tokenizer.LookupToken("</think>");
        List<int> offPrompt = tokenizer.Encode("<|im_start|>assistant\n<think></think>");
        int cap = ChatGenerationPipeline.UnrequestedThinkingBudgetFor(1024);
        var off = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, Arch, cap,
            out bool offInstalled, enableThinking: false, promptTokens: offPrompt);
        Assert.True(offInstalled);
        Assert.Equal(64, off.ThinkingBudget!.TokenLimit);
        Assert.Equal(open, off.ThinkingBudget.OpenTokenId);
        Assert.Equal(close, off.ThinkingBudget.EndTokenId);
        Assert.False(off.ThinkingBudget.OpenAtStart);
        Assert.False(off.ThinkingBudget.SuppressUnopenedEnd);   // a stray close still separates the answer
        Assert.NotNull(off.ThinkingBudget.CloseAtBoundary);
        Assert.True(new TokenSampler(off).IsPlainGreedyArgmax);

        List<int> onPrompt = tokenizer.Encode("<|im_start|>assistant\n<think>");
        var on = ChatGenerationPipeline.WithThinkingBudget(SamplingConfig.Greedy, tokenizer, Arch, 768,
            out bool onInstalled, enableThinking: true, promptTokens: onPrompt);
        Assert.True(onInstalled);
        Assert.True(on.ThinkingBudget!.OpenAtStart);
        Assert.Equal(768, on.ThinkingBudget.TokenLimit);
        Assert.Null(on.ThinkingBudget.CloseAtBoundary);
    }

    /// <summary>The think tags as single vocabulary entries, then one token per character.</summary>
    private sealed class TagTokenizer : ITokenizer
    {
        private static readonly string[] Tags = { "<think>", "</think>" };
        public TagTokenizer()
        {
            var vocab = new List<string> { "<|im_end|>" };
            vocab.AddRange(Tags);
            vocab.Add("\n");
            for (char c = ' '; c <= '~'; c++) vocab.Add(c.ToString());
            Vocab = vocab.ToArray();
        }
        public string[] Vocab { get; }
        public int BosTokenId => -1;
        public int[] EosTokenIds => [0];
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true)
        {
            var ids = new List<int>();
            for (int at = 0; at < text.Length;)
            {
                int tag = Array.FindIndex(Tags, t => string.CompareOrdinal(text, at, t, 0, t.Length) == 0);
                string piece = tag >= 0 ? Tags[tag] : text[at].ToString();
                ids.Add(LookupToken(piece));
                at += piece.Length;
            }
            return ids;
        }
        public string Decode(List<int> ids) => string.Concat(ids.Select(id => Vocab[id]));
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string tokenStr) => Array.IndexOf(Vocab, tokenStr);
    }

    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => [];
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(i => (char)i).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
            => buffer.AddRange(Encoding.UTF8.GetBytes(((char)tokenId).ToString()));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }
}
