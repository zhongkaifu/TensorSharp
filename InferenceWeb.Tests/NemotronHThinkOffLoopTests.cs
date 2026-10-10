// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Where the thinking-off stray reasoning of Nemotron-H Reasoning-128K actually happened: in
// the tool loop, rounds two and later, each round after a tool result. These drive the loop
// with the logged replies the way the pipeline feeds it (the prompt's tail first, then the
// text a character at a time), apply its updates the way the Web UI does, and check that the
// answer, the loop's own history and the transcript record all agree, since that agreement
// is what lets the next request reuse the cache.
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

public sealed class NemotronHThinkOffLoopTests : IDisposable
{
    private const string Architecture = "nemotron_h";
    private const string ReasoningTemplate =
        "{{ '<SPECIAL_10>System\n' }}{% for message in messages %}{{ '\n<SPECIAL_11>Assistant\n' }}{% endfor %}";
    private const string Closed = "<think></think>";

    /// <summary>The logged first round of that conversation: a preamble and a call.</summary>
    private const string PreambleAndCall =
        "To determine the number of lines in the `notes.txt` file, I will use the `read_file` tool to read the file and then count the lines.\n\n<TOOLCALL>[read_file(path=\"notes.txt\")]</TOOLCALL>\n";

    private readonly string _skillsDir = Path.Combine(Path.GetTempPath(), "ts-thinkoff-loop-" + Guid.NewGuid().ToString("N"));

    public NemotronHThinkOffLoopTests()
    {
        string dir = Path.Combine(_skillsDir, "notes");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "SKILL.md"), "---\nname: notes\ndescription: reads notes\n---\n\nRead them.\n");
    }

    public void Dispose()
    {
        try { Directory.Delete(_skillsDir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public async Task StrayReasoning_IsRetractedForAClientThatCanTakeItBack()
    {
        var prompts = new List<List<ChatMessage>>();
        List<ChatStreamUpdate> updates = await Run(Plan(retracts: true), prompts,
            NemotronHLoggedReplies.ToolRoundWithCall, NemotronHLoggedReplies.ToolRoundWithAnswer);
        Page page = Page.From(updates);

        Assert.Equal("The file 'notes.txt' has **7 lines**, as previously determined. No need to read it again since it hasn't changed.",
            page.Answer.Trim());
        Assert.Contains("Just call 'read_file' with the correct path.", page.Thinking);
        Assert.Contains("I should respond with that information without running the tool again.", page.Thinking);
        Assert.Contains(updates, u => !string.IsNullOrEmpty(u.RetractedPiece));
        Assert.Equal(2, page.Replaces);

        // The round the next generation is rendered from says what the page shows.
        ChatMessage round1 = prompts[1].Single(m => m.Role == "assistant" && m.RawOutputTokens != null);
        Assert.Empty(round1.Content.Trim());
        Assert.StartsWith("Okay, the user wants to know", round1.Thinking);
        Assert.Equal("read_file", Assert.Single(round1.ToolCalls!).Name);
    }

    [Fact]
    public async Task WithoutARetractingClient_TheLoopHoldsInsteadOfRetracting()
    {
        var prompts = new List<List<ChatMessage>>();
        List<ChatStreamUpdate> updates = await Run(Plan(retracts: false), prompts,
            NemotronHLoggedReplies.ToolRoundWithCall, NemotronHLoggedReplies.ToolRoundWithAnswer);

        Assert.DoesNotContain(updates, u => !string.IsNullOrEmpty(u.RetractedPiece));
        string content = string.Concat(updates.Where(u => !u.Done).Select(u => u.Piece ?? string.Empty));
        Assert.DoesNotContain("Okay, the user", content);
        Assert.DoesNotContain("</think>", content);
        Assert.StartsWith("The file 'notes.txt' has **7 lines**", content.Trim());
    }

    /// <summary>
    /// The logged conversation: a preamble and a call, then stray reasoning and a call, then
    /// stray reasoning and the answer. The page sends the turn back as one message with what
    /// it was left showing, and the transcript must still recognise it, or the follow-up
    /// re-prefills the whole conversation without any error.
    /// </summary>
    [Fact]
    public async Task AFollowUpWithWhatThePageShowed_StillSplicesTheRecordedTurn()
    {
        var store = new ConversationTranscriptStore(maxChains: 64, maxTokens: 1_000_000);
        var user = new ChatMessage { Role = "user", Content = "How many lines does the attached notes.txt have?" };
        var prompts = new List<List<ChatMessage>>();
        List<ChatStreamUpdate> updates = await Run(Plan(retracts: true), prompts,
            new[] { PreambleAndCall, NemotronHLoggedReplies.ToolRoundWithCall, NemotronHLoggedReplies.ToolRoundWithAnswer },
            user, store);
        Page page = Page.From(updates);

        Assert.StartsWith("To determine the number of lines", page.Answer);
        Assert.EndsWith("No need to read it again since it hasn't changed.", page.Answer.Trim());
        Assert.DoesNotContain("Okay, the user", page.Answer);
        Assert.DoesNotContain("</think>", page.Answer);

        TranscriptAugmentation followUp = store.Augment(new List<ChatMessage>
        {
            user,
            new() { Role = "assistant", Content = page.Answer, Thinking = page.Thinking },
            new() { Role = "user", Content = "And what is its last line?" },
        });

        Assert.Equal(1, followUp.SplicedTurns);
        Assert.Equal("scope", followUp.InheritedScope);
        // user, three rounds with their two tool results, the new user message.
        Assert.Equal(7, followUp.History.Count);
        Assert.All(followUp.History.Where(m => m.Role == "assistant"), m => Assert.NotNull(m.RawOutputTokens));

        // The same follow-up with the reasoning still in the answer is not what was sent.
        TranscriptAugmentation leaked = store.Augment(new List<ChatMessage>
        {
            user,
            new() { Role = "assistant", Content = NemotronHLoggedReplies.ToolRoundWithAnswer, Thinking = page.Thinking },
            new() { Role = "user", Content = "And what is its last line?" },
        });
        Assert.Equal(0, leaked.SplicedTurns);
    }

    /// <summary>
    /// The Web UI's rescue after a thinking-budget stop runs the turn again with thinking
    /// off, which is exactly when this template reasons past its closed block. With skills
    /// on, the retry's updates come from the loop already separated: the retry's own parser
    /// must pass them through untouched, retraction included, and the frames must leave the
    /// page showing only the answer.
    /// </summary>
    [Fact]
    public async Task TheThinkingOffRetry_PassesTheLoopsRetractionThrough()
    {
        List<ChatStreamUpdate> updates = await Run(Plan(retracts: true), new List<List<ChatMessage>>(),
            NemotronHLoggedReplies.ToolRoundWithCall, NemotronHLoggedReplies.ToolRoundWithAnswer);
        IOutputParser retryParser = ChatProtocolRegistry.For(Architecture)!.CreateOutputParser!();
        retryParser.Init(false, null);
        retryParser.AcceptRetractions();

        var visible = new StringBuilder();
        string thinking = string.Empty;
        int replaces = 0;
        foreach (ChatStreamUpdate update in updates.Where(u => !u.Done))
        {
            ChatStreamUpdate retried = WebUiChatService.ParseRetryUpdate(update, retryParser);
            Assert.Equal(update, retried);
            foreach (object frame in WebUiChatService.AnswerFrames(
                visible, retried.Piece, retried.ThinkingPiece, retried.RetractedPiece))
            {
                using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(frame));
                if (doc.RootElement.TryGetProperty("thinking", out JsonElement t)) thinking += t.GetString();
                if (doc.RootElement.TryGetProperty("replace", out _)) replaces++;
            }
        }

        Assert.Equal("The file 'notes.txt' has **7 lines**, as previously determined. No need to read it again since it hasn't changed.",
            visible.ToString().Trim());
        Assert.Equal(2, replaces);
        Assert.Contains("I should respond with that information without running the tool again.", thinking);
    }

    [Fact]
    public async Task AnEmptyCallListAfterStrayReasoning_EndsTheTurnWithTheNoAnswerNote()
    {
        List<ChatStreamUpdate> updates = await Run(Plan(retracts: true), new List<List<ChatMessage>>(),
            NemotronHLoggedReplies.ToolRoundWithShortReasoning, NemotronHLoggedReplies.ToolRoundWithEmptyCallList);
        var visible = new StringBuilder();
        int tokens = 0;
        foreach (ChatStreamUpdate update in updates.Where(u => u.IsParsed))
        {
            if (WebUiChatService.HasParsedAnswerContent(update))
                tokens++;
            _ = WebUiChatService.AnswerFrames(visible, update.Piece, update.ThinkingPiece, update.RetractedPiece).ToList();
        }
        Assert.True(tokens > 0, "the reasoning was shown before it was taken back");
        Assert.Empty(visible.ToString().Trim());

        using var session = new ChatSession();
        List<string> answer = Tokens(WebUiChatService.FinalFrames(null, false, null, session, Stopwatch.StartNew(), tokens,
            0, 0, false, visible));
        Assert.Contains(answer, t => t.Contains("ended this turn without writing an answer", StringComparison.Ordinal));
    }

    /// <summary>
    /// A loop whose client cannot take text back holds an undecided reply and forwards
    /// nothing of it until it is decided, so for every piece it holds it hands the client's
    /// stream loop an update with nothing in it: that is what an append-only adapter's
    /// keep-alive runs on (StreamKeepAlive). In both generations the loop runs: a tool
    /// round, and the answer it asks for once the round limit is reached.
    /// </summary>
    [Fact]
    public async Task AHoldingLoop_HandsOverAnUpdateForEveryPieceItHolds()
    {
        List<ChatStreamUpdate> updates = await Run(Plan(retracts: false, maxRounds: 1), new List<List<ChatMessage>>(),
            NemotronHLoggedReplies.ToolRoundWithCall, NemotronHLoggedReplies.ToolRoundWithAnswer);

        static bool Empty(ChatStreamUpdate u) => u.IsParsed && !u.Done && string.IsNullOrEmpty(u.Piece)
            && string.IsNullOrEmpty(u.ThinkingPiece) && string.IsNullOrEmpty(u.RetractedPiece)
            && u.ParsedToolCalls == null && u.ToolProgressPhase == null;
        // The final round starts after the tool round's last progress update; its answer
        // with the first content after that.
        int finalRound = updates.FindLastIndex(u => u.ToolProgressPhase != null) + 1;
        int answerStarts = updates.FindIndex(finalRound, u => !string.IsNullOrEmpty(u.Piece));
        Assert.InRange(finalRound, 1, answerStarts - 1);
        // The tool round's reasoning, held until its close, and the final round's.
        Assert.True(updates.Take(finalRound).Count(Empty) >= NemotronHLoggedReplies.ToolRoundWithCall.IndexOf("</think>", StringComparison.Ordinal));
        Assert.True(updates.Skip(finalRound).Take(answerStarts - finalRound).Count(Empty)
            >= NemotronHLoggedReplies.ToolRoundWithAnswer.IndexOf("</think>", StringComparison.Ordinal));
        Assert.Equal("The file 'notes.txt' has **7 lines**, as previously determined. No need to read it again since it hasn't changed.",
            string.Concat(updates.Where(u => u.IsParsed).Select(u => u.Piece)).Trim());
    }

    /// <summary>
    /// TensorSharp.Runtime and TensorSharp.Chat are published packages: an assembly built
    /// against the earlier signatures (a custom skill generator calling the three-argument
    /// <c>Parsed</c>, a host constructing the parsers) must still bind to them.
    /// </summary>
    [Fact]
    public void TheEarlierPublicSignatures_StillBind()
    {
        Assert.NotNull(typeof(ChatStreamUpdate).GetMethod(nameof(ChatStreamUpdate.Parsed),
            new[] { typeof(string), typeof(string), typeof(IReadOnlyList<ToolCall>) }));
        Assert.NotNull(typeof(NemotronOutputParser).GetConstructor(Type.EmptyTypes));
        Assert.NotNull(typeof(ChatMlOutputParser).GetConstructor(Type.EmptyTypes));
        Assert.NotNull(typeof(OutputParserFactory).GetMethod(nameof(OutputParserFactory.IsAlwaysRequired),
            new[] { typeof(string) }));
    }

    // ---------------------------------------------------------------- the loop, as the pipeline drives it

    private SkillRequestPlan Plan(bool retracts, int maxRounds = 0)
    {
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _skillsDir } });
        var args = new List<string> { "--model", "x.gguf", "--skills-dir", _skillsDir };
        if (maxRounds > 0)
            args.AddRange(new[] { "--skills-max-rounds", maxRounds.ToString(System.Globalization.CultureInfo.InvariantCulture) });
        ServerHostingOptions options = ServerOptionsBuilder.Build(args.ToArray(), _skillsDir);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, new[] { "notes" }, false, null, Architecture,
            contextTokens: 32768, options, out IReadOnlyList<string> unknown);
        Assert.Empty(unknown);
        plan.ClientRetractsAnswerText = retracts;
        return plan;
    }

    private static Task<List<ChatStreamUpdate>> Run(
        SkillRequestPlan plan, List<List<ChatMessage>> prompts, params string[] rounds)
        => Run(plan, prompts, rounds, new ChatMessage { Role = "user", Content = "How many lines?" }, store: null);

    /// <summary>
    /// Runs the loop over canned rounds. Each round is generated the way the pipeline does
    /// it for this template: the prompt's tail announced first, the reply one character at
    /// a time, and the turn recorded in <paramref name="store"/> with what the transcript's
    /// parser makes of it.
    /// </summary>
    private static async Task<List<ChatStreamUpdate>> Run(
        SkillRequestPlan plan, List<List<ChatMessage>> prompts, string[] rounds, ChatMessage user,
        ConversationTranscriptStore? store)
    {
        int round = 0;
        int nextToken = 100;
        async IAsyncEnumerable<ChatStreamUpdate> Generate(
            List<ChatMessage> messages, List<ToolFunction> tools, [EnumeratorCancellation] CancellationToken ct)
        {
            prompts.Add(new List<ChatMessage>(messages));
            string text = rounds[Math.Min(round++, rounds.Length - 1)];
            yield return ChatStreamUpdate.Text(string.Empty) with { RawGenerationSuffix = Closed };
            foreach (char c in text)
            {
                ct.ThrowIfCancellationRequested();
                yield return ChatStreamUpdate.Text(c.ToString());
                await Task.Yield();
            }
            var raw = Enumerable.Range(nextToken, 8).ToList();
            nextToken += 8;
            store?.Record(messages,
                new ChatMessage
                {
                    Role = "assistant", Content = text, RawOutputTokens = raw,
                    RawPromptTrailingWhitespace = string.Empty, RawGenerationSuffix = Closed,
                },
                ChatGenerationPipeline.BuildEmittedTurn(Architecture, text, false, tools, Closed, cancelled: false),
                "scope");
            yield return new ChatStreamUpdate(string.Empty, true, 10, 20, 5, 0, 0, 0, "stop")
            {
                RawOutputTokens = raw,
                RawPromptTrailingWhitespace = string.Empty,
                RawGenerationSuffix = Closed,
            };
        }

        var all = new List<ChatStreamUpdate>();
        await foreach (ChatStreamUpdate update in SkillChatLoop.RunAsync(
            Architecture, new List<ChatMessage> { user }, plan, false, Generate, logger: null, CancellationToken.None))
            all.Add(update);
        Assert.True(ChatGenerationPipeline.AnnouncesGenerationSuffix(Architecture, ReasoningTemplate, Closed));
        return all;
    }

    /// <summary>The answer and reasoning a page is left showing, from the frames the Web UI
    /// sends for the loop's updates, read the way both pages read them.</summary>
    private sealed record Page(string Answer, string Thinking, int Replaces)
    {
        public static Page From(IEnumerable<ChatStreamUpdate> updates)
        {
            var visible = new StringBuilder();
            string answer = string.Empty, thinking = string.Empty;
            int replaces = 0;
            foreach (ChatStreamUpdate update in updates.Where(u => u.IsParsed))
            {
                foreach (object frame in WebUiChatService.AnswerFrames(
                    visible, update.Piece, update.ThinkingPiece, update.RetractedPiece))
                {
                    using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(frame));
                    JsonElement root = doc.RootElement;
                    if (root.TryGetProperty("thinking", out JsonElement t)) thinking += t.GetString();
                    if (root.TryGetProperty("token", out JsonElement token)) answer += token.GetString();
                    if (root.TryGetProperty("replace", out JsonElement replace))
                    {
                        answer = replace.GetString()!;
                        replaces++;
                    }
                }
            }
            Assert.Equal(visible.ToString(), answer);
            return new Page(answer, thinking, replaces);
        }
    }

    private static List<string> Tokens(IEnumerable<object> frames)
    {
        var tokens = new List<string>();
        foreach (object frame in frames)
        {
            using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(frame));
            if (doc.RootElement.TryGetProperty("token", out JsonElement token))
                tokens.Add(token.GetString()!);
        }
        return tokens;
    }
}
