// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Runtime;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the progressive-disclosure loop's streaming contract.
///
/// <para>
/// The loop used to buffer every round and replay only the last one, because a round
/// that ends in a <c>skills_read</c> carries that call's markup and forwarding it would
/// hand a client a tool it has no implementation for. It worked, and it silently cost
/// every skills request its stream — on the Web UI, 26 content frames starting 0.17 s in
/// collapsed to 3 frames all landing at 3.28 s. These tests exist so that regression
/// cannot come back quietly: they assert that content arrives in PIECES, that the pieces
/// are separated rather than raw, and that the tool markup never gets out.
/// </para>
/// <para>
/// The bar is deliberately "more than one update", not a token count. A parser is free
/// to coalesce, and asserting an exact frame count would break on a parser change that
/// is not a regression; a single update, on the other hand, is the buffered behaviour
/// exactly.
/// </para>
/// </summary>
public class SkillChatLoopStreamingTests : IDisposable
{
    private readonly string _baseDir;

    public SkillChatLoopStreamingTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-stream-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- fixtures ----------------------------------------------------------

    private void WriteSkill(string name, string description, params (string Path, string Body)[] extras)
    {
        string dir = Path.Combine(_baseDir, name);
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, "SKILL.md"),
            $"---\nname: {name}\ndescription: {description}\n---\n\nUse the reference.\n");
        foreach ((string path, string body) in extras)
        {
            string full = Path.Combine(dir, path);
            Directory.CreateDirectory(Path.GetDirectoryName(full));
            File.WriteAllText(full, body);
        }
    }

    /// <summary>
    /// Nemotron-H is used throughout: its ChatML markers (<c>&lt;think&gt;</c>,
    /// <c>&lt;tool_call&gt;</c>) are simple enough to write by hand in a fixture, and it
    /// renders both tool declarations and tool results, so the loop takes its full path.
    /// </summary>
    private const string Architecture = "nemotron_h_moe";

    /// <param name="clientTools">
    /// Names the CALLER declared. Passing them matters: the loop tells a tool the client
    /// will answer from one nobody declared by looking them up in the request's tool
    /// list, so a fixture that forwards a client call has to declare it the way a real
    /// client does.
    /// </param>
    private SkillRequestPlan Plan(
        IReadOnlyList<string> selected, bool discovery = false, ICodeRunner codeRunner = null,
        params string[] clientTools)
    {
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        // Built through the real builder rather than hand-constructed: ServerHostingOptions
        // is an all-positional record, so a literal here would silently drift.
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "x.gguf", "--skills-dir", _baseDir }, _baseDir);
        List<ToolFunction> declared = clientTools.Length == 0
            ? null
            : clientTools.Select(n => new ToolFunction { Name = n, Description = n }).ToList();
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, selected, discovery, declared, Architecture,
            contextTokens: 32768, options, out IReadOnlyList<string> unknown,
            codeRunner: codeRunner);
        Assert.Empty(unknown);
        Assert.NotNull(plan);
        return plan;
    }

    private SkillRequestPlan CodePlan(ICodeRunner codeRunner, SessionWorkspace workspace)
    {
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "x.gguf", "--skills-dir", _baseDir }, _baseDir);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, Array.Empty<string>(), false, null, Architecture,
            contextTokens: 32768, options, out IReadOnlyList<string> unknown,
            codeRunner: codeRunner, workspace: workspace);
        Assert.Empty(unknown);
        Assert.NotNull(plan);
        return plan;
    }

    /// <summary>
    /// A generator that replays canned rounds one CHARACTER at a time, which is the
    /// worst case a real token stream can present: every marker is split across updates,
    /// so a loop that only works when a tag arrives whole will fail here.
    /// </summary>
    /// <remarks>
    /// Every fixture opens with a <c>&lt;think&gt;</c> block because that is what a
    /// reasoning model emits and what the ChatML parser expects: with thinking enabled it
    /// starts INSIDE the reasoning channel, so a fixture that skipped the tag would be
    /// read as pure reasoning and assert nothing about content.
    /// </remarks>
    private static SkillChatGeneration Replay(params string[] rounds)
    {
        int round = 0;
        return (messages, tools, ct) => Emit(rounds[Math.Min(round++, rounds.Length - 1)], ct);
    }

    private static async IAsyncEnumerable<ChatStreamUpdate> Emit(
        string text, [EnumeratorCancellation] CancellationToken ct)
    {
        foreach (char c in text)
        {
            ct.ThrowIfCancellationRequested();
            yield return ChatStreamUpdate.Text(c.ToString());
            await Task.Yield();
        }
        yield return new ChatStreamUpdate(string.Empty, true, 10, 20, 5, 0, 0, 0, "stop");
    }

    private static async Task<List<ChatStreamUpdate>> Drain(IAsyncEnumerable<ChatStreamUpdate> stream)
    {
        var all = new List<ChatStreamUpdate>();
        await foreach (ChatStreamUpdate u in stream)
            all.Add(u);
        return all;
    }

    private Task<List<ChatStreamUpdate>> Run(
        SkillRequestPlan plan, SkillChatGeneration generate, bool thinking = true)
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "go" } };
        return Drain(SkillChatLoop.RunAsync(
            Architecture, messages, plan, thinking, generate, logger: null, CancellationToken.None));
    }

    private static string Content(IEnumerable<ChatStreamUpdate> updates) =>
        string.Concat(updates.Where(u => !u.Done).Select(u => u.Piece ?? string.Empty));

    private static string Thinking(IEnumerable<ChatStreamUpdate> updates) =>
        string.Concat(updates.Where(u => !u.Done).Select(u => u.ThinkingPiece ?? string.Empty));

    private static List<ChatStreamUpdate> Payload(IEnumerable<ChatStreamUpdate> updates) =>
        updates.Where(u => !u.Done).ToList();

    // ---- it streams --------------------------------------------------------

    [Fact]
    public async Task SingleRound_StreamsContentInPieces_RatherThanOneBufferedUpdate()
    {
        WriteSkill("alpha", "does alpha things");
        List<ChatStreamUpdate> updates = await Run(Plan(new[] { "alpha" }), Replay("<think>simple</think>The answer is 42."));

        Assert.Equal("The answer is 42.", Content(updates));
        // The whole point: more than one non-terminal update carried the content.
        Assert.True(Payload(updates).Count(u => !string.IsNullOrEmpty(u.Piece)) > 1,
            "content arrived in a single update, which is the buffered behaviour this fix removed");
    }

    [Fact]
    public async Task EveryNonTerminalUpdate_IsMarkedParsed()
    {
        WriteSkill("alpha", "does alpha things");
        List<ChatStreamUpdate> updates = await Run(Plan(new[] { "alpha" }), Replay("<think>hmm</think>hello"));

        Assert.All(Payload(updates), u => Assert.True(u.IsParsed,
            "an adapter keys off IsParsed to skip its own parser; an unmarked update gets parsed twice"));
    }

    [Fact]
    public async Task Reasoning_IsSeparatedFromContent_AndBothStream()
    {
        WriteSkill("alpha", "does alpha things");
        List<ChatStreamUpdate> updates = await Run(
            Plan(new[] { "alpha" }), Replay("<think>weighing it up</think>The answer."));

        Assert.Equal("The answer.", Content(updates));
        Assert.Equal("weighing it up", Thinking(updates));
        Assert.True(Payload(updates).Count(u => !string.IsNullOrEmpty(u.ThinkingPiece)) > 1);
    }

    // ---- and it still keeps the markup to itself ----------------------------

    [Fact]
    public async Task ARoundThatCallsSkillsRead_NeverForwardsTheCall_AndTheNextRoundStreams()
    {
        WriteSkill("alpha", "does alpha things", ("guide.md", "The magic number is 1238."));
        SkillRequestPlan plan = Plan(new[] { "alpha" });

        List<ChatStreamUpdate> updates = await Run(plan, Replay(
            "<think>need the guide</think><tool_call>\n{\"name\": \"skills_read\", \"arguments\": {\"skill\": \"alpha\", \"path\": \"guide.md\"}}\n</tool_call>",
            "<think>got it</think>The magic number is 1238."));

        Assert.Equal("The magic number is 1238.", Content(updates));
        Assert.All(Payload(updates), u => Assert.True(
            u.ParsedToolCalls == null || u.ParsedToolCalls.Count == 0,
            "skills_read is answered in process and must never reach the caller"));
        Assert.DoesNotContain("skills_read", Content(updates), StringComparison.Ordinal);
        // The lookup really happened rather than the model being taken at its word.
        Assert.Single(plan.Invocations);
        Assert.Equal("skills_read", plan.Invocations[0].Tool);
    }

    [Fact]
    public async Task AClientToolCall_IsForwardedOnce_OnItsOwnUpdate()
    {
        WriteSkill("alpha", "does alpha things");
        List<ChatStreamUpdate> updates = await Run(
            Plan(new[] { "alpha" }, discovery: false, codeRunner: null, "get_weather"),
            Replay("<think>ask the tool</think>Checking.<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>"));

        Assert.Equal("Checking.", Content(updates));
        List<ToolCall> forwarded = Payload(updates)
            .Where(u => u.ParsedToolCalls != null)
            .SelectMany(u => u.ParsedToolCalls)
            .ToList();
        ToolCall only = Assert.Single(forwarded);
        Assert.Equal("get_weather", only.Name);

        // The Web UI owns one transient activity panel. Even though this call is
        // handed to the caller rather than executed here, its generation progress
        // still needs a terminal event so the panel cannot remain stuck on screen.
        ChatStreamUpdate lastProgress = Payload(updates)
            .Last(u => u.ToolProgressPhase != null);
        Assert.Equal("finished", lastProgress.ToolProgressPhase);
        Assert.Equal("get_weather", lastProgress.ToolProgressName);
    }

    [Fact]
    public async Task ARoundThatCallsListFiles_IsAnsweredHere_AndTheNextRoundStreamsTheAnswer()
    {
        // The incident's own path, end to end through the loop. shell is DECLARED
        // by this host and was classified as the caller's, so it was handed to a Web UI
        // that has no implementation and the turn ended with nothing rendered — the
        // model's whole reply having been inside <think>. The unknown-name test above
        // covers a different bucket; this one covers the tool that actually broke.
        WriteSkill("alpha", "does alpha things");
        SkillRequestPlan plan = Plan(new[] { "alpha" }, codeRunner: new AlwaysReadyRunner());

        List<ChatStreamUpdate> updates = await Run(plan, Replay(
            "<think>what is here</think><tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
            "<think>now I know</think>Nothing has been written yet."));

        Assert.All(Payload(updates), u => Assert.True(
            u.ParsedToolCalls == null || u.ParsedToolCalls.Count == 0,
            "shell is answered in process and must never reach the caller"));
        Assert.Equal("Nothing has been written yet.", Content(updates));

        SkillToolInvocation invocation = Assert.Single(plan.Invocations);
        Assert.Equal(SkillToolNames.Shell, invocation.Tool);

        // Many writing updates are expected because Replay emits one character at
        // a time. Collapse adjacent equal phases and pin the lifecycle rather than
        // an implementation-dependent frame count: the live Web UI replaces its
        // current activity on each transition and removes it on finished.
        List<ChatStreamUpdate> progress = Payload(updates)
            .Where(u => u.ToolProgressPhase != null)
            .ToList();
        string[] phases = progress.Select(u => u.ToolProgressPhase).ToArray();
        string[] phaseTransitions = phases
            .Where((phase, i) => i == 0 || phase != phases[i - 1])
            .ToArray();
        Assert.Equal(new[] { "writing", "running", "finished" }, phaseTransitions);

        ChatStreamUpdate finished = progress[^1];
        Assert.Equal(SkillToolNames.Shell, finished.ToolProgressName);
    }

    [Fact]
    public void AHostThatRunsCode_GetsMoreRoundsThanOneThatOnlyReadsSkills()
    {
        // Eight rounds was sized for progressive disclosure — read a skill, read two of
        // its references — and it is generous for that. It is not the same activity as
        // writing a program, running it, reading the traceback and fixing it, and one
        // counter gates both. The failure that prompted this spent three rounds reading
        // skills, two producing a document, and three on a deck it was still debugging
        // when the budget ran out.
        WriteSkill("alpha", "does alpha things");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "x.gguf", "--skills-dir", _baseDir }, _baseDir);

        SkillRequestPlan readOnly = SkillRequestPlan.Create(
            registry, new[] { "alpha" }, false, null, Architecture, 32768, options, out _);
        SkillRequestPlan withCode = SkillRequestPlan.Create(
            registry, new[] { "alpha" }, false, null, Architecture, 32768, options, out _,
            codeRunner: new AlwaysReadyRunner());

        Assert.Equal(8, readOnly.LoopOptions.MaxRounds);
        Assert.True(withCode.LoopOptions.MaxRounds > readOnly.LoopOptions.MaxRounds,
            "a plan that can run code needs room to write, run and fix a program");
    }

    [Fact]
    public void AnOperatorsChosenRoundLimit_IsNeverRaised()
    {
        // Both directions. --skills-max-rounds is how an operator bounds what one
        // malfunctioning request costs, and silently overriding it downward or upward
        // would make the flag a suggestion.
        WriteSkill("alpha", "does alpha things");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "x.gguf", "--skills-dir", _baseDir, "--skills-max-rounds", "3" }, _baseDir);

        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, new[] { "alpha" }, false, null, Architecture, 32768, options, out _,
            codeRunner: new AlwaysReadyRunner());

        Assert.Equal(3, plan.LoopOptions.MaxRounds);
    }

    /// <summary>A runner that is available but never asked to run anything.</summary>
    private sealed class AlwaysReadyRunner : ICodeRunner
    {
        public bool CanRun => true;
        public string UnavailableReason => null;
        public bool CanInstallPackages => false;
        public ToolFunction Declare() => new() { Name = SkillToolNames.Shell, Description = "runs commands" };
        public string InstallPackages(
            string language, IReadOnlyList<string> packages, SessionWorkspace workspace,
            Action<string> onOutput = null) => "not used in these tests";
        public SkillToolResult Execute(
            ToolCall call, IReadOnlyList<CodeInputFile> inputFiles = null, Action<string> onOutput = null,
            SessionWorkspace workspace = null, IReadOnlyList<string> skillDirectories = null) =>
            SkillToolResult.Failure("not used in these tests");
    }

    [Fact]
    public async Task Cancellation_DetachesTheRequest_ButDefersWorkspaceDeleteUntilItsWorkerStops()
    {
        string workspaceRoot = Path.Combine(_baseDir, "request-workspaces");
        var manager = new SessionWorkspaceManager(workspaceRoot);
        var runner = new BlockingRunner();
        RequestWorkspaceLease lease = Assert.IsType<RequestWorkspaceLease>(
            RequestWorkspaceLease.Acquire(manager, runner, Architecture));
        string leasedRoot = lease.Workspace.Root;
        SkillRequestPlan plan = CodePlan(runner, lease.Workspace);
        using var cancelled = new CancellationTokenSource();

        var messages = new List<ChatMessage> { new() { Role = "user", Content = "run it" } };
        Task<List<ChatStreamUpdate>> drain = Drain(SkillChatLoop.RunAsync(
            Architecture, messages, plan, enableThinking: true,
            Replay("<think>run</think><tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"wait\"}}\n</tool_call>"),
            logger: null, cancelled.Token));

        try
        {
            await runner.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
            cancelled.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => drain);

            // This is the adapter's method-scope using/finally on an aborted stream.
            lease.Dispose();
            Assert.True(Directory.Exists(leasedRoot),
                "request disposal must not delete a directory under an in-flight tool");

            runner.AllowCompletion();
            await runner.Completed.Task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.True(SpinWait.SpinUntil(() => !Directory.Exists(leasedRoot), TimeSpan.FromSeconds(5)),
                "the deferred release should delete as soon as the worker exits");
        }
        finally
        {
            runner.AllowCompletion();
            lease.Dispose();
        }
    }

    private sealed class BlockingRunner : ICodeRunner
    {
        private readonly ManualResetEventSlim _continue = new(false);

        public TaskCompletionSource<bool> Started { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public TaskCompletionSource<bool> Completed { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public bool CanRun => true;
        public string UnavailableReason => null;

        public ToolFunction Declare() =>
            new() { Name = SkillToolNames.Shell, Description = "runs commands" };

        public SkillToolResult Execute(
            ToolCall call, IReadOnlyList<CodeInputFile> inputFiles = null,
            Action<string> onOutput = null, SessionWorkspace workspace = null,
            IReadOnlyList<string> skillDirectories = null)
        {
            Started.TrySetResult(true);
            try
            {
                _continue.Wait();
                return new SkillToolResult(true, "done", null, null);
            }
            finally
            {
                Completed.TrySetResult(true);
            }
        }

        public void AllowCompletion() => _continue.Set();
    }

    [Fact]
    public async Task AnExhaustedTurn_StillSaysSomething_RatherThanRenderingAnEmptyReply()
    {
        // The other end of the same silence. The model is told the budget is gone and
        // asks for another tool anyway, so its whole reply is markup the loop drops —
        // and the user gets a blank bubble that is indistinguishable from a crash.
        // Observed: a README -> internal-comms -> pptx run produced the document, then
        // spent its last rounds fixing a deck and ended showing nothing at all.
        WriteSkill("alpha", "does alpha things");
        SkillRequestPlan plan = Plan(new[] { "alpha" });

        List<ChatStreamUpdate> updates = await Run(plan, Replay(
            "<think>again</think><tool_call>\n{\"name\": \"skills_read\", \"arguments\": {\"skill\": \"alpha\", \"path\": \"SKILL.md\"}}\n</tool_call>"));

        string content = Content(updates);
        Assert.False(string.IsNullOrWhiteSpace(content), "an exhausted turn must never render as an empty reply");
        Assert.Contains("tool-call budget", content, StringComparison.Ordinal);
        Assert.Contains("continue", content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AToolNobodyDeclared_IsAnsweredInTheLoop_NotHandedToAClientThatCannotRunIt()
    {
        // The turn that made this necessary: the model called a tool the host declares
        // but the classifier had not learned. It was forwarded to the Web UI, which
        // declares no tools and has no handler for the tool-call frame at all — so a
        // reply whose every word was inside <think> rendered as nothing, and the chat
        // simply stopped. Forwarding is only ever right for a name the CALLER declared.
        WriteSkill("alpha", "does alpha things");
        List<ChatStreamUpdate> updates = await Run(Plan(new[] { "alpha" }), Replay(
            "<think>guess</think><tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>",
            "<think>ok</think>It is 4 degrees."));

        Assert.All(Payload(updates), u => Assert.True(
            u.ParsedToolCalls == null || u.ParsedToolCalls.Count == 0,
            "a tool nobody declared must not be handed to a caller that cannot run it"));

        // The model got a second round and used it, so the user sees an answer rather
        // than an empty reply.
        Assert.Equal("It is 4 degrees.", Content(updates));
    }

    // ---- metrics survive the round trip ------------------------------------

    [Fact]
    public async Task TheTerminalUpdate_SumsEveryRoundsMetrics_AndIsNotMarkedParsed()
    {
        WriteSkill("alpha", "does alpha things", ("guide.md", "1238"));
        List<ChatStreamUpdate> updates = await Run(Plan(new[] { "alpha" }), Replay(
            "<think>look</think><tool_call>\n{\"name\": \"skills_read\", \"arguments\": {\"skill\": \"alpha\", \"path\": \"guide.md\"}}\n</tool_call>",
            "<think>done</think>1238."));

        ChatStreamUpdate done = Assert.Single(updates, u => u.Done);
        Assert.False(done.IsParsed, "the terminal update carries metrics, not parsed text");
        Assert.Equal(20, done.PromptTokens);          // two rounds at 10
        Assert.Equal(40, done.EvalTokens);            // two rounds at 20
        Assert.Equal(10, done.KvCacheReusedTokens);   // two rounds at 5
    }

    // ---- the collector the non-streaming endpoints share --------------------

    [Fact]
    public async Task GemmaPromptChannelMetadata_PrimesTheSkillLoopBeforeStreaming()
    {
        WriteSkill("alpha", "does alpha things");
        async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages,
            List<ToolFunction> tools, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            yield return ChatStreamUpdate.Text("") with { RawGenerationSuffix = "<|channel>thought\n" };
            foreach (char value in "Inspect the tool result.<channel|>42")
                yield return ChatStreamUpdate.Text(value.ToString());
            await Task.CompletedTask;
            yield return new ChatStreamUpdate("", true, 10, 20, 0, 0, 0, 0, "eos");
        }
        var updates = new List<ChatStreamUpdate>();
        await foreach (var update in SkillChatLoop.RunAsync("gemma4", new List<ChatMessage>(),
            Plan(new[] { "alpha" }), true, Generate, null, CancellationToken.None))
            updates.Add(update);
        Assert.Equal("42", string.Concat(updates.Where(u => !u.Done).Select(u => u.Piece)));
        Assert.Equal("Inspect the tool result.", string.Concat(updates.Select(u => u.ThinkingPiece)));
    }

    [Fact]
    public void Collector_PreservesPromptOpenedGemmaChannel()
    {
        var collector = new ChatStreamCollector();
        collector.Add(ChatStreamUpdate.Text("") with { RawGenerationSuffix = "<|channel>thought\n" });
        collector.Add(ChatStreamUpdate.Text("Inspect the result.<channel|>42"));
        var parsed = collector.Resolve("gemma4", true, null);
        Assert.Equal("42", parsed.Content);
        Assert.Equal("Inspect the result.", parsed.Thinking);
    }

    [Fact]
    public void Collector_OnAPreParsedStream_ReturnsThePiecesWithoutReParsing()
    {
        var collector = new ChatStreamCollector();
        // Text that would be mangled if a ChatML parser ran over it a second time.
        collector.Add(ChatStreamUpdate.Parsed("Write <think> to open a block.", "why not", null));
        collector.Add(ChatStreamUpdate.Parsed(" Done.", null,
            new List<ToolCall> { new() { Name = "get_weather" } }));

        Assert.True(collector.IsParsed);
        ParsedOutput parsed = collector.Resolve(Architecture, enableThinking: true, tools: null);
        Assert.Equal("Write <think> to open a block. Done.", parsed.Content);
        Assert.Equal("why not", parsed.Thinking);
        Assert.Equal("get_weather", Assert.Single(parsed.ToolCalls).Name);
    }

    [Fact]
    public void Collector_OnARawStream_ParsesItExactlyAsBefore()
    {
        var collector = new ChatStreamCollector();
        collector.Add(ChatStreamUpdate.Text("<think>weighing</think>"));
        collector.Add(ChatStreamUpdate.Text("The answer."));

        Assert.False(collector.IsParsed);
        ParsedOutput parsed = collector.Resolve(Architecture, enableThinking: true, tools: null);
        Assert.Equal("The answer.", parsed.Content);
        Assert.Equal("weighing", parsed.Thinking);
    }
}
