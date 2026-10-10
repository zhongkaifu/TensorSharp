// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Nemotron-H 8B Reasoning-128K with thinking off ends its prompt with "<think></think>", and in
// tool rounds (and some first rounds) it reasoned anyway and closed the block itself. The
// parser started in content and never looked for the close, so the reasoning and the literal
// "</think>" were the answer the user read and the one the transcript kept. The replies below
// are verbatim from the Q4_K_M; NemotronHLoggedReplies says where each was logged.
using System.Text;

namespace InferenceWeb.Tests;

public sealed class NemotronHThinkingOffReasoningTests
{
    private const string ReasoningTemplate =
        "{{ '<SPECIAL_10>System\n' }}{% for message in messages %}{{ '\n<SPECIAL_11>Assistant\n' }}{% endfor %}";
    private const string ChatMlTemplate = "{% for message in messages %}<|im_start|>{{ message['role'] }}{% endfor %}";
    private const string Closed = "<think></think>";
    private const string Open = "<think>\n";

    private static readonly List<ToolFunction> Tools =
    [
        new()
        {
            Name = "read_file",
            Parameters = new Dictionary<string, ToolParameter> { ["path"] = new() { Type = "string" } },
            Required = ["path"],
        },
        new()
        {
            Name = "shell",
            Parameters = new Dictionary<string, ToolParameter> { ["command"] = new() { Type = "string" } },
            Required = ["command"],
        },
    ];

    public static TheoryData<string, int> StrayCloses()
    {
        var data = new TheoryData<string, int>();
        foreach (string reply in new[]
                 {
                     NemotronHLoggedReplies.ToolRoundWithCall, NemotronHLoggedReplies.ToolRoundWithAnswer,
                     NemotronHLoggedReplies.FirstRoundWithFencedCommand,
                 })
        {
            foreach (int chunk in new[] { 1, 3, 7, 0 })
                data.Add(reply, chunk);
        }
        return data;
    }

    [Theory]
    [MemberData(nameof(StrayCloses))]
    public void StrayClose_MovesTheReasoningToThinking(string reply, int chunk)
    {
        int close = reply.IndexOf("</think>", StringComparison.Ordinal);
        Run today = Drive(Unprimed(), reply, chunk);
        foreach (bool retract in new[] { false, true })
        {
            Run run = Drive(Registry(), reply, chunk, retract, Closed);

            Assert.False(run.CloseEverShown, "the stray </think> was shown as answer text");
            Assert.Equal(Squash(reply.Substring(0, close)), Squash(run.Thinking));
            Assert.Equal(today.Calls, run.Calls);
            Assert.Equal(0, run.SuffixFailures);
            Assert.DoesNotContain("Okay, the user", run.Content);
            Assert.DoesNotContain("To solve this problem", run.Content);
            if (!retract)
                Assert.Equal(0, run.Retractions);
        }
    }

    [Fact]
    public void TheAnswerAfterTheClose_IsWhatTheUserReads()
    {
        Run run = Drive(Registry(), NemotronHLoggedReplies.ToolRoundWithAnswer, 1, retract: true, suffix: Closed);

        Assert.Equal(
            "The file 'notes.txt' has **7 lines**, as previously determined. No need to read it again since it hasn't changed.",
            run.Content.Trim());
        Assert.Equal(1, run.Retractions);
    }

    [Theory]
    [InlineData("the capital of France is Paris.\n")]
    [InlineData("To determine the number of lines in the `notes.txt` file, I will use the `read_file` tool.\n\n<TOOLCALL>[read_file(path=\"notes.txt\")]</TOOLCALL>\n")]
    [InlineData("Use `echo` here:\n\n```bash\necho hi\n```\nThat prints hi.")]
    public void OrdinaryThinkingOffReplies_AreUnchanged(string reply)
    {
        foreach (int chunk in new[] { 1, 3, 7, 0 })
        {
            Run today = Drive(Unprimed(), reply, chunk);
            Run held = Drive(Registry(), reply, chunk, retract: false, suffix: Closed);
            Run shown = Drive(Registry(), reply, chunk, retract: true, suffix: Closed);

            Assert.Equal(today.Content, held.Content);
            Assert.Equal(today.Content, shown.Content);
            Assert.Equal(today.Calls, held.Calls);
            Assert.Equal(today.Calls, shown.Calls);
            Assert.Empty(held.Thinking);
            Assert.Equal(0, shown.Retractions);
            // A client that can take text back sees the answer exactly as early as before.
            Assert.Equal(today.FirstContentAt, shown.FirstContentAt);
            Assert.Equal(today.FirstCallAt, held.FirstCallAt);
        }
    }

    [Fact]
    public void HeldStream_ReleasesOnlyAtTheDecision()
    {
        // The end of the reply decides an ordinary answer.
        IOutputParser parser = Primed(Closed);
        foreach (char c in "Paris is the capital.")
            Assert.Empty(parser.Add(c.ToString(), false).Content);
        Assert.Equal("Paris is the capital.", parser.Add(string.Empty, true).Content);

        // A complete call decides a preamble, and only once it is complete.
        parser = Primed(Closed);
        const string preamble = "I will read it.\n<TOOLCALL>[read_file(path=\"notes.txt\")]</TOOLCALL>";
        var content = new StringBuilder();
        for (int i = 0; i < preamble.Length; i++)
        {
            ParsedOutput delta = parser.Add(preamble[i].ToString(), false);
            content.Append(delta.Content);
            if (i < preamble.Length - 1)
                Assert.Equal(0, content.Length);
            else
                Assert.Equal("read_file", Assert.Single(delta.ToolCalls!).Name);
        }
        // Streamed, the line break before the tag went out with the text, as it does today.
        Assert.Equal("I will read it.\n", content.ToString());

        // The close decides reasoning, and nothing of it is ever content.
        parser = Primed(Closed);
        content.Clear();
        foreach (char c in NemotronHLoggedReplies.ToolRoundWithAnswer)
            content.Append(parser.Add(c.ToString(), false).Content);
        content.Append(parser.Add(string.Empty, true).Content);
        Assert.StartsWith("The file 'notes.txt' has", content.ToString());
    }

    [Fact]
    public void Retraction_IsExactlyTheTextEarlierResultsShowed()
    {
        IOutputParser parser = Primed(Closed, retract: true);
        var shown = new StringBuilder();
        Assert.Equal("Okay, plan.", parser.Add("Okay, plan.", false).Content);
        shown.Append("Okay, plan.");

        // The close arrives with more undecided text in the same piece: only what was
        // already shown is retracted, and the rest never becomes content.
        ParsedOutput closing = parser.Add(" More.\n</think>\n\nSeven.", false);

        Assert.Equal("Okay, plan.", closing.RetractedContent);
        Assert.Equal("Okay, plan. More.", closing.Thinking);
        Assert.Equal("Seven.", closing.Content);

        // A tag whose body is no call goes back to the undecided text within one piece; a
        // close later in the same piece takes it out of that piece's content again.
        parser = Primed(Closed, retract: true);
        ParsedOutput whole = parser.Add("Plan <TOOLCALL>[1, 2]</TOOLCALL> more\n</think>\nSeven.", false);
        Assert.Empty(whole.RetractedContent);
        Assert.Equal("Seven.", whole.Content);
        Assert.Equal("Plan<TOOLCALL>[1, 2]</TOOLCALL>more", whole.Thinking);
    }

    [Fact]
    public void ACloseSplitAcrossPieces_StillCloses()
    {
        foreach (bool retract in new[] { false, true })
        {
            Run run = DrivePieces(Primed(Closed, retract), "Okay, plan it.\n</th", "ink>\n\nThe file has 7 lines.");
            Assert.Equal("The file has 7 lines.", run.Content);
            Assert.Equal("Okay, plan it.", run.Thinking);
            Assert.False(run.CloseEverShown);
        }
    }

    [Fact]
    public void AReplyThatOpensItsOwnBlock_IsReasoning()
    {
        Run run = DrivePieces(Primed(Closed, retract: true), "<thi", "nk>\nLet me see.", "\n</think>\n\nSeven.");

        Assert.Equal("Seven.", run.Content);
        Assert.Equal("Let me see.", run.Thinking);
        Assert.Equal(0, run.Retractions);
    }

    [Theory]
    [InlineData("Split the reply on the tag:\n\n```python\nanswer = text.split('</think>', 1)[-1]\n```\n\nThat keeps the answer.")]
    [InlineData("Close the block with `</think>` and continue.")]
    [InlineData("A fence of tildes:\n~~~\nprint('</think>')\n~~~\nDone.")]
    public void AQuotedClose_InACodeBlockOrSpan_IsAnswerText(string reply)
    {
        foreach (int chunk in new[] { 1, 3, 0 })
        {
            foreach (bool retract in new[] { false, true })
            {
                Run run = Drive(Registry(), reply, chunk, retract, Closed);
                Assert.Equal(reply, run.Content);
                Assert.Empty(run.Thinking);
                Assert.Equal(0, run.Retractions);
            }
        }
    }

    [Fact]
    public void AStrayCloseAfterAClosedFence_StillCloses()
    {
        // The logged first-round reply quotes a command in a ```bash fence, closes the fence,
        // and only then closes the block.
        Run run = Drive(Registry(), NemotronHLoggedReplies.FirstRoundWithFencedCommand, 1, retract: true, suffix: Closed);

        Assert.Contains("sha256sum | cut -c1-12", run.Thinking);
        Assert.Equal("shell", run.Calls.Split('(')[0]);
        Assert.Empty(run.Content.Trim());
    }

    [Fact]
    public void ACloseThatStartsPastTheWindow_IsTheAnswersOwnText()
    {
        int window = ChatProtocolRegistry.NemotronHStrayReasoningWindow;
        string inside = new string('a', window - 1) + "</think>after";
        string outside = new string('a', window) + "</think>after";
        foreach (int chunk in new[] { 1, 7, 0 })
        {
            foreach (bool retract in new[] { false, true })
            {
                Run decided = Drive(Registry(), inside, chunk, retract, Closed);
                Assert.Equal("after", decided.Content);
                Assert.Equal(new string('a', window - 1), decided.Thinking);

                Run answer = Drive(Registry(), outside, chunk, retract, Closed);
                Assert.Equal(outside, answer.Content);
                Assert.Empty(answer.Thinking);
                Assert.Equal(0, answer.Retractions);
            }
        }

        // A held stream shows a long answer once the window has passed, not at its end.
        IOutputParser parser = Primed(Closed);
        string longAnswer = new string('b', window + 200);
        int firstAt = -1;
        for (int i = 0; i < longAnswer.Length && firstAt < 0; i++)
            if (parser.Add(longAnswer[i].ToString(), false).Content.Length > 0)
                firstAt = i + 1;
        Assert.Equal(window, firstAt);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void AnEmptyCallList_IsNoCallAndNoText(int chunk)
    {
        foreach (bool retract in new[] { false, true })
        {
            Run run = Drive(Registry(), NemotronHLoggedReplies.ToolRoundWithEmptyCallList, chunk, retract, Closed);

            Assert.Empty(run.Calls);
            Assert.Empty(run.Content.Trim());
            Assert.Contains("the answer is still 7", run.Thinking);
        }

        // In reasoning it is not a call either, and the reasoning goes on after it.
        IOutputParser thinking = Registry();
        thinking.Init(true, Tools);
        ParsedOutput parsed = thinking.Add("No call: <TOOLCALL>[]</TOOLCALL> then more.\n</think>\nSeven.", true);
        Assert.Null(parsed.ToolCalls);
        Assert.Equal("Seven.", parsed.Content);
        Assert.Contains("then more.", parsed.Thinking);

        // A parser rule, not a template one: a Nemotron 3 ChatML reply (never primed) that
        // ends on an empty list in the other call tag shows no markup either.
        IOutputParser chatMl = Unprimed();
        chatMl.Init(false, Tools);
        ParsedOutput nano = chatMl.Add("Nothing to call.\n<tool_call>[]</tool_call>\n", true);
        Assert.Null(nano.ToolCalls);
        Assert.Equal("Nothing to call.", nano.Content.Trim());
    }

    [Fact]
    public void ACallBeforeAnyClose_DecidesAnAnswer_AsThinkingOnWould()
    {
        const string reply = "Plan: <TOOLCALL>[shell(command=\"ls\")]</TOOLCALL>\nmore\n</think>\nDone.";
        foreach (bool retract in new[] { false, true })
        {
            Run run = Drive(Registry(), reply, 1, retract, Closed);
            Assert.Equal("shell", run.Calls.Split('(')[0]);
            Assert.StartsWith("Plan:", run.Content);
            Assert.Equal(0, run.Retractions);
        }

        // A tag that is not a call leaves the reply undecided, so the close still decides it.
        Run notACall = Drive(Registry(), "I will use <TOOLCALL> later.\n</think>\nDone.", 1, retract: true, suffix: Closed);
        Assert.Equal("Done.", notACall.Content);
        Assert.Equal(1, notACall.Retractions);
    }

    [Fact]
    public void ThinkingOn_IsUnchanged()
    {
        const string reply = "France's capital.\n</think>\n\nParis.";
        foreach (int chunk in new[] { 1, 3, 0 })
        {
            IOutputParser today = Unprimed();
            today.Init(true, Tools);
            Run expected = Drive(today, reply, chunk, initialized: true);
            IOutputParser primed = Registry();
            primed.Init(true, Tools);
            primed.SetGenerationPromptSuffix(Open);
            Run actual = Drive(primed, reply, chunk, initialized: true);
            Assert.Equal(expected.Content, actual.Content);
            Assert.Equal(expected.Thinking, actual.Thinking);
        }
    }

    [Fact]
    public void ThePromptsTail_DecidesTheStart_NotTheRequestFlag()
    {
        // {'reasoning': True} in a caller's system prompt opened the block on a think-off request.
        IOutputParser parser = Registry();
        parser.Init(false, Tools);
        parser.SetGenerationPromptSuffix(Open);
        ParsedOutput opened = parser.Add("Long reasoning.\n</think>\n\nSeven.", true);
        Assert.Equal(("Seven.", "Long reasoning."), (opened.Content, opened.Thinking));

        // {'reasoning': False} closed it on a think-on request.
        parser = Registry();
        parser.Init(true, Tools);
        parser.SetGenerationPromptSuffix(Closed);
        ParsedOutput closed = parser.Add("Seven.", true);
        Assert.Equal(("Seven.", ""), (closed.Content, closed.Thinking));

        // Once generated text has arrived the tail is history, not a start state.
        parser = Registry();
        parser.Init(false, Tools);
        parser.Add("Okay", false);
        parser.SetGenerationPromptSuffix(Open);
        Assert.Equal("Okay, done.", "Okay" + parser.Add(", done.", true).Content);
    }

    [Fact]
    public void OtherChatMlFamilies_IgnoreThePromptsTail()
    {
        foreach (IOutputParser parser in new IOutputParser[] { new Qwen35OutputParser(), new NemotronOutputParser() })
        {
            parser.Init(false, Tools);
            parser.AcceptRetractions();
            parser.SetGenerationPromptSuffix(Closed);
            ParsedOutput parsed = parser.Add("Okay.\n</think>\nSeven.", true);
            Assert.Equal("Okay.\n</think>\nSeven.", parsed.Content);
            Assert.Empty(parsed.RetractedContent);
        }
    }

    // ---------------------------------------------------------------- scoped to the template

    [Fact]
    public void OnlyTheReasoningTemplate_ReadsThePromptsTail()
    {
        Assert.True(OutputParserFactory.ThinkingOffReplyMayReason("nemotron_h", ReasoningTemplate));
        Assert.False(OutputParserFactory.ThinkingOffReplyMayReason("nemotron_h", ChatMlTemplate));
        Assert.False(OutputParserFactory.ThinkingOffReplyMayReason("nemotron_h", null));
        Assert.False(OutputParserFactory.ThinkingOffReplyMayReason("qwen35", ReasoningTemplate));

        Assert.True(OutputParserFactory.IsAlwaysRequired("nemotron_h", ReasoningTemplate));
        Assert.False(OutputParserFactory.IsAlwaysRequired("nemotron_h", ChatMlTemplate));
        Assert.False(OutputParserFactory.IsAlwaysRequired("nemotron_h"));

        Assert.True(ChatGenerationPipeline.AnnouncesGenerationSuffix("nemotron_h", ReasoningTemplate, Closed));
        Assert.True(ChatGenerationPipeline.AnnouncesGenerationSuffix("nemotron_h", ReasoningTemplate, Open));
        Assert.False(ChatGenerationPipeline.AnnouncesGenerationSuffix("nemotron_h", ReasoningTemplate, string.Empty));
        Assert.False(ChatGenerationPipeline.AnnouncesGenerationSuffix("nemotron_h", ChatMlTemplate, Closed));
        Assert.True(ChatGenerationPipeline.AnnouncesGenerationSuffix("gemma4", null, "<|channel>thought\n"));
        Assert.False(ChatGenerationPipeline.AnnouncesGenerationSuffix("gemma4", null, "<|channel>thought\n<channel|>"));
    }

    [Fact]
    public void NemotronChatMl_RecordsAndAnnouncesNothing_SoItParsesByteForByteAsBefore()
    {
        var tokenizer = new CharTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Lines in notes.txt?" } };
        List<int> prompt = renderer.RenderToTokens(tokenizer, ChatMlTemplate, messages, "nemotron_h", true,
            out _, out _, tools: Tools, enableThinking: false);
        Assert.EndsWith(Closed, tokenizer.Decode(prompt));

        string suffix = ChatGenerationPipeline.RecordedGenerationSuffix(tokenizer, prompt, "nemotron_h", false, ChatMlTemplate);
        Assert.Equal(string.Empty, suffix);
        Assert.False(ChatGenerationPipeline.AnnouncesGenerationSuffix("nemotron_h", ChatMlTemplate, suffix));
        Assert.Null(CliPromptTail(ChatMlTemplate, tokenizer, prompt));

        // So the parser the Web UI, the adapters and the transcript run is never primed.
        foreach (string reply in new[] { NemotronHLoggedReplies.ToolRoundWithAnswer, "Paris.\n" })
        {
            foreach (int chunk in new[] { 1, 0 })
            {
                Run today = Drive(Unprimed(), reply, chunk);
                Run now = Drive(Registry(), reply, chunk, retract: true, suffix: null);
                Assert.Equal(today, now);
            }
        }
        EmittedAssistantTurn emitted = ChatGenerationPipeline.BuildEmittedTurn(
            "nemotron_h", NemotronHLoggedReplies.ToolRoundWithAnswer, false, Tools, generationSuffix: null, cancelled: false);
        Assert.Equal(NemotronHLoggedReplies.ToolRoundWithAnswer, emitted.Content);
    }

    [Theory]
    [InlineData(false, null, Closed)]
    [InlineData(true, null, Open)]
    [InlineData(false, "{'reasoning': True}", Open)]
    [InlineData(true, "{'reasoning': False}", Closed)]
    public void TheRecordedSuffix_IsWhatThePromptEndsWith(bool requestThinking, string? marker, string expected)
    {
        var tokenizer = new CharTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var messages = new List<ChatMessage>();
        if (marker != null)
            messages.Add(new ChatMessage { Role = "system", Content = "You are terse. " + marker });
        messages.Add(new ChatMessage { Role = "user", Content = "What is 6 + 1?" });
        List<int> prompt = renderer.RenderToTokens(tokenizer, ReasoningTemplate, messages, "nemotron_h", true,
            out _, out _, tools: null, enableThinking: requestThinking);
        Assert.EndsWith(expected, tokenizer.Decode(prompt));

        string recorded = ChatGenerationPipeline.RecordedGenerationSuffix(
            tokenizer, prompt, "nemotron_h", requestThinking, ReasoningTemplate);
        Assert.Equal(expected, recorded);
        Assert.Equal(expected, CliPromptTail(ReasoningTemplate, tokenizer, prompt)![^expected.Length..]);
    }

    // ---------------------------------------------------------------- every consumer parses alike

    [Fact]
    public void TheTranscriptAndTheTurnWatcher_ParseLikeTheStreams()
    {
        string reply = NemotronHLoggedReplies.ToolRoundWithCall;
        Run streamed = Drive(Registry(), reply, 1, retract: true, suffix: Closed);

        EmittedAssistantTurn emitted = ChatGenerationPipeline.BuildEmittedTurn(
            "nemotron_h", reply, false, Tools, generationSuffix: Closed, cancelled: false);
        Assert.Equal(Squash(streamed.Content), Squash(emitted.Content));
        Assert.Equal(Squash(streamed.Thinking), Squash(emitted.Thinking));
        Assert.Equal(reply, emitted.RawText);
        Assert.DoesNotContain("</think>", emitted.Content);

        // The turn watcher, primed the same way, ends the turn at the same character as the
        // stream surfaces the call, and before any made-up tool response.
        ToolCallTurnEnd watcher = ToolCallTurnEnd.For("nemotron_h", ReasoningTemplate, false, Tools, Closed)!;
        int firedAt = -1;
        for (int i = 0; i < reply.Length && firedAt < 0; i++)
            if (watcher.Observe(reply[i].ToString())) firedAt = i + 1;
        Assert.Equal(streamed.FirstCallAt, firedAt);
    }

    [Fact]
    public void TheCliPrimesFromThePromptTail()
    {
        var tokenizer = new CharTokenizer();
        List<int> prompt = tokenizer.Encode(new string('x', 500) + "\n<SPECIAL_11>Assistant\n" + Closed);
        string? tail = CliPromptTail(ReasoningTemplate, tokenizer, prompt);
        IOutputParser parser = TensorSharp.Cli.CliOutputParser.Create("nemotron_h", false, Tools, tail);

        ParsedOutput parsed = parser.Add(NemotronHLoggedReplies.ToolRoundWithAnswer, true);
        Assert.StartsWith("The file 'notes.txt' has", parsed.Content);
        Assert.StartsWith("Okay, the user", parsed.Thinking);
        Assert.Equal(64, tail!.Length);
    }

    [Fact]
    public void EveryProtocol_ParserAlwaysRequiredMatchesItsParser()
    {
        foreach (ChatProtocol protocol in ChatProtocolRegistry.All)
        {
            if (protocol.CreateOutputParser == null)
            {
                Assert.False(protocol.OutputParserAlwaysRequired, protocol.Id);
                Assert.Null(protocol.ThinkingOffReplyMayReason);
                continue;
            }
            Assert.True(protocol.OutputParserAlwaysRequired == protocol.CreateOutputParser().AlwaysRequired, protocol.Id);
            foreach (string arch in protocol.Architectures)
            {
                // A template only ever adds a requirement, and only where the protocol says so.
                Assert.Equal(protocol.OutputParserAlwaysRequired, OutputParserFactory.IsAlwaysRequired(arch, null));
                Assert.Equal(
                    protocol.OutputParserAlwaysRequired || protocol.ThinkingOffReplyMayReason?.Invoke(ReasoningTemplate) == true,
                    OutputParserFactory.IsAlwaysRequired(arch, ReasoningTemplate));
            }
        }
    }

    // ---------------------------------------------------------------- the split does not decide

    /// <summary>
    /// The transcript, the collectors and the non-streaming adapters parse a reply whole; the
    /// pages and the SSE adapters parse it as it streams. These replies were split one way
    /// whole and the other way streamed: the Markdown rule read the undecided text as it was
    /// SHOWN, and the call branch had trimmed the line breaks around an empty call list, so
    /// whole the two fence lines ran together into an unclosed six-backtick fence and the
    /// close read as quoted. Streamed, the line breaks survived and the close closed. The
    /// page showed the answer while the transcript kept the reasoning, and the follow-up
    /// re-prefilled the conversation.
    /// </summary>
    [Theory]
    [InlineData("Okay, the user asks whether to call a tool. I could write:\n```\n<TOOLCALL>[]</TOOLCALL>\n```\nbut there is nothing to call.\n</think>\n\nThe file has 7 lines.\n", true)]
    [InlineData("Okay, I could write:\n~~~\n<TOOLCALL>[1, 2]</TOOLCALL>\n~~~\nbut that is no call.\n</think>\n\nThe file has 7 lines.\n", true)]
    [InlineData("Okay, the form is `<TOOLCALL>[]</TOOLCALL>` and nothing else.\n</think>\n\nThe file has 7 lines.\n", true)]
    [InlineData("Quote it as:\n```\n<TOOLCALL>[]</TOOLCALL>\n</think>\n```\nand go on.", false)]
    public void TheDecision_DoesNotDependOnHowTheReplyWasSplit(string reply, bool closes)
    {
        foreach (bool retract in new[] { false, true })
        {
            Run whole = Drive(Registry(), reply, 0, retract, Closed);
            foreach (int chunk in new[] { 1, 2, 4, 7 })
            {
                Run streamed = Drive(Registry(), reply, chunk, retract, Closed);
                Assert.Equal(Squash(whole.Content), Squash(streamed.Content));
                Assert.Equal(Squash(whole.Thinking), Squash(streamed.Thinking));
                Assert.Equal(whole.Calls, streamed.Calls);
                Assert.Equal(0, streamed.SuffixFailures);
            }
            if (closes)
            {
                Assert.Equal("The file has 7 lines.", whole.Content.Trim());
                Assert.StartsWith(reply.Substring(0, 8), whole.Thinking);
            }
            else
            {
                Assert.Empty(whole.Thinking);
                Assert.Contains("</think>", whole.Content);
            }
        }
    }

    /// <summary>
    /// The same property over random replies built from the pieces that decide a reply
    /// (closes, openers, fences, code spans, calls, empty and non-call tag pairs, line
    /// breaks), each fed whole and in random pieces, after a closed, an open and no block,
    /// with thinking on and off, with and without retraction: the split never changes what
    /// is answer, what is reasoning and what is a call, and a retraction is always the end
    /// of what was shown.
    /// </summary>
    [Fact]
    public void RandomReplies_ParseAlikeWholeAndStreamed()
    {
        string[] parts =
        {
            "Okay, the user wants it.", " ", "\n", "\n\n", "```", "~~~", "`", "``", "</think>", "<think>",
            "<TOOLCALL>[]</TOOLCALL>", "<TOOLCALL>[1, 2]</TOOLCALL>", "<TOOLCALL> later", "<TOOLCALL>",
            "<TOOLCALL>[shell(command=\"ls\")]</TOOLCALL>", "<tool_call>[]</tool_call>", "Seven.", "  ",
        };
        var rng = new Random(20261009);
        int checkedReplies = 0;
        for (int n = 0; n < 4000; n++)
        {
            var reply = new StringBuilder();
            int count = rng.Next(1, 14);
            for (int i = 0; i < count; i++)
                reply.Append(parts[rng.Next(parts.Length)]);
            string text = reply.ToString();
            foreach ((bool thinking, string? suffix, bool retract) in Modes)
            {
                IOutputParser wholeParser = Registry();
                wholeParser.Init(thinking, Tools);
                if (retract) wholeParser.AcceptRetractions();
                wholeParser.SetGenerationPromptSuffix(suffix);
                Run whole = Apply(wholeParser, new[] { text }, wholeIsDone: true);
                var pieces = new List<string>();
                for (int i = 0; i < text.Length;)
                {
                    int size = Math.Min(rng.Next(1, 7), text.Length - i);
                    pieces.Add(text.Substring(i, size));
                    i += size;
                }
                IOutputParser parser = Registry();
                parser.Init(thinking, Tools);
                if (retract) parser.AcceptRetractions();
                parser.SetGenerationPromptSuffix(suffix);
                Run streamed = Apply(parser, pieces, wholeIsDone: false);

                string where = $"reply {n} (thinking {thinking}, suffix {suffix?.Replace("\n", "\\n") ?? "none"}, {(retract ? "retract" : "hold")}): {text.Replace("\n", "\\n")}";
                Assert.True(Squash(whole.Content) == Squash(streamed.Content), "content differs, " + where);
                Assert.True(Squash(whole.Thinking) == Squash(streamed.Thinking), "thinking differs, " + where);
                Assert.True(whole.Calls == streamed.Calls, "calls differ, " + where);
                Assert.True(streamed.SuffixFailures == 0, "a retraction was not the end of the answer, " + where);
                checkedReplies++;
            }
        }
        Assert.Equal(4000 * Modes.Length, checkedReplies);
    }

    private static readonly (bool Thinking, string? Suffix, bool Retract)[] Modes =
    {
        (false, Closed, false), (false, Closed, true), (false, Open, false), (false, Open, true),
        (false, null, false), (true, Open, false), (true, Closed, true), (true, null, true),
    };

    // ---------------------------------------------------------------- helpers

    private static IOutputParser Registry() => ChatProtocolRegistry.For("nemotron_h")!.CreateOutputParser!();

    private static IOutputParser Unprimed() => new NemotronOutputParser();

    private static IOutputParser Primed(string suffix, bool retract = false)
    {
        IOutputParser parser = Registry();
        parser.Init(false, Tools);
        if (retract) parser.AcceptRetractions();
        parser.SetGenerationPromptSuffix(suffix);
        return parser;
    }

    private static string? CliPromptTail(string template, ITokenizer tokenizer, List<int> prompt)
        => TensorSharp.Cli.CliOutputParser.PromptTail("nemotron_h", template, tokenizer, prompt);

    internal sealed record Run(
        string Content, string Thinking, string Calls, int FirstContentAt, int FirstCallAt,
        int Retractions, int SuffixFailures, bool CloseEverShown);

    /// <summary>Feeds <paramref name="text"/> the way a stream does (pieces of
    /// <paramref name="chunk"/> characters, or whole when 0) and applies every result the
    /// way a page does: a retraction leaves the shown answer before new content is added.</summary>
    internal static Run Drive(IOutputParser parser, string text, int chunk, bool retract = false,
        string? suffix = null, bool initialized = false)
    {
        if (!initialized)
            parser.Init(false, Tools);
        if (retract) parser.AcceptRetractions();
        if (!initialized)
            parser.SetGenerationPromptSuffix(suffix);
        var pieces = new List<string>();
        if (chunk <= 0)
            pieces.Add(text);
        else
            for (int i = 0; i < text.Length; i += chunk)
                pieces.Add(text.Substring(i, Math.Min(chunk, text.Length - i)));
        return Apply(parser, pieces, wholeIsDone: chunk <= 0);
    }

    private static Run DrivePieces(IOutputParser parser, params string[] pieces) => Apply(parser, pieces, wholeIsDone: false);

    private static Run Apply(IOutputParser parser, IReadOnlyList<string> pieces, bool wholeIsDone)
    {
        var answer = new StringBuilder();
        var thinking = new StringBuilder();
        var calls = new List<string>();
        int consumed = 0, firstContent = -1, firstCall = -1, retractions = 0, suffixFailures = 0;
        bool closeShown = false;
        var feed = new List<(string Piece, bool Done)>();
        foreach (string piece in pieces)
            feed.Add((piece, wholeIsDone));
        if (!wholeIsDone)
            feed.Add((string.Empty, true));
        foreach ((string piece, bool done) in feed)
        {
            consumed += piece.Length;
            ParsedOutput parsed = parser.Add(piece, done);
            if (parsed.RetractedContent.Length > 0)
            {
                retractions++;
                if (answer.ToString().EndsWith(parsed.RetractedContent, StringComparison.Ordinal))
                    answer.Length -= parsed.RetractedContent.Length;
                else
                    suffixFailures++;
            }
            thinking.Append(parsed.Thinking);
            answer.Append(parsed.Content);
            if (parsed.ToolCalls is { Count: > 0 })
            {
                calls.AddRange(parsed.ToolCalls.Select(c => c.ToString()));
                if (firstCall < 0) firstCall = consumed;
            }
            if (firstContent < 0 && answer.Length > 0) firstContent = consumed;
            closeShown |= answer.ToString().Contains("</think>", StringComparison.Ordinal);
        }
        return new Run(answer.ToString(), thinking.ToString(), string.Join("|", calls), firstContent, firstCall,
            retractions, suffixFailures, closeShown);
    }

    private static string Squash(string text) => new(text.Where(c => !char.IsWhiteSpace(c)).ToArray());

    /// <summary>One token per character; enough to decode prompt tails exactly.</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Array.Empty<string>();
        public int VocabSize => char.MaxValue + 1;
        public int BosTokenId => -1;
        public int[] EosTokenIds => Array.Empty<int>();
        public bool IsEos(int id) => false;
        public int LookupToken(string token) => token.Length == 1 ? token[0] : -1;
        public List<int> Encode(string text, bool addSpecial = true) => (text ?? string.Empty).Select(c => (int)c).ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(t => (char)t).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(((char)token).ToString()));
    }
}

/// <summary>
/// Nemotron-H 8B Reasoning-128K (Q4_K_M) replies to thinking-off prompts, verbatim from the
/// logs of the runs that found the leak: TensorAgent tool rounds (nemotron3.log lines 3761,
/// 4043, 4432 and 4648 of the 2026-10-09 end-to-end run) and an OpenAI-API first round
/// (server.log line 169 of the tool-format A/B). The replay tool
/// (eng/validation/NemotronThinkOffReplay) reads the same logs whole.
/// </summary>
internal static class NemotronHLoggedReplies
{
    /// <summary>A tool round: stray reasoning, its close, then the call.</summary>
    public const string ToolRoundWithCall =
        "Okay, the user wants to know how many lines are in the 'notes.txt' file. The available tools include 'read_file', which can read the file and show the lines. Since the file is attached and has 7 lines based on the previous response, I need to confirm that. Using 'read_file' will display the lines, and then I can count them. The tool's parameters require the path, which is 'notes.txt'. No other tools are needed here. Just call 'read_file' with the correct path.\n</think>\n\n<TOOLCALL>[read_file(path=\"notes.txt\")]</TOOLCALL>\n";

    /// <summary>A tool round: stray reasoning, its close, then the answer.</summary>
    public const string ToolRoundWithAnswer =
        "Okay, the user is asking how many lines are in the 'notes.txt' file. They mentioned it's attached and has 7 lines. But the tool response says the file hasn't changed since the last time it was read, which was 7 lines. So instead of reading it again, I should just use the previous result. The user doesn't need to read the file again because it's unchanged. So the answer is still 7 lines. I should respond with that information without running the tool again.\n</think>\n\nThe file 'notes.txt' has **7 lines**, as previously determined. No need to read it again since it hasn't changed.\n";

    /// <summary>A tool round: stray reasoning, its close, then the call (the 4432 round).</summary>
    public const string ToolRoundWithShortReasoning =
        "Okay, the user wants to know how many lines are in notes.txt. The available tool is read_file, which can read the file and return its contents. The tool's parameters require a path, and the file is named notes.txt. Using read_file will give the lines, and I can count them. So I'll call read_file with path=\"notes.txt\".\n</think>\n\n<TOOLCALL>[read_file(path=\"notes.txt\")]</TOOLCALL>\n";

    /// <summary>A tool round that reasoned, closed the block and wrote an empty call list.</summary>
    public const string ToolRoundWithEmptyCallList =
        "Okay, the user previously asked how many lines are in notes.txt and I used the read_file tool to get the answer. Now they're asking again, but the tool's response says the file hasn't changed since the last read. So I should use the previous result instead of reading the file again. The last response mentioned 7 lines, so the answer is still 7. No need to call read_file again.\n</think>\n\n<TOOLCALL>[]</TOOLCALL>\n";

    /// <summary>A first round whose reasoning quotes a command in a closed ```bash fence.</summary>
    public const string FirstRoundWithFencedCommand =
        "To solve this problem, I need to generate the SHA-256 hash of the string \"tensoragent\" and then output the first 12 characters of that hash. Here's how I can do it:\n\nFirst, I'll use the `echo` command to print the string without a newline, which is achieved by using `-n`. Then, I'll pipe this output to the `sha256sum` command to compute the hash. The result will be a hexadecimal digest.\n\nAfter obtaining the full hash, I need to extract only the first 12 characters. This can be done using the `cut` command with the `-c1-12` option, which specifies the first 12 characters.\n\nPutting this all together in a shell command:\n\n```bash\necho -n \"tensoragent\" | sha256sum | cut -c1-12\n```\n\nWhen I run this command, it will output the first 12 hexadecimal characters of the SHA-256 hash of the string \"tensoragent\".\n</think>\n\n<tool_call>[shell(command=\"echo -n 'tensoragent' | sha256sum | cut -c1-12\")]</tool_call>\n";
}
