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
using System.Threading;
using System.Threading.Tasks;

namespace InferenceWeb.Tests;

/// <summary>
/// Drives the progressive-disclosure loop with a scripted generator, so every branch
/// is exercised without a model.
///
/// <para>
/// The loop exists so that a client which knows nothing about skills still gets a
/// finished answer: an ordinary OpenAI client sends <c>skills: ["pdf"]</c> and one user
/// message, and if TensorSharp handed <c>skills_read</c> back as a tool call that client
/// would have no implementation to service it and the conversation would simply stall.
/// Answering those calls in process is only safe because they are read-only and confined
/// to a directory the operator already exposed — which is exactly why the caller's OWN
/// tools must never be executed here, and why that case is tested as carefully as the
/// happy path.
/// </para>
/// <para>
/// Two of the cases below are silent-failure guards rather than correctness checks.
/// <see cref="SkillAgentLoopOptions.ToolResultsAreRendered"/> off is the Mistral 3
/// shape: that renderer drops every <c>role: "tool"</c> message, so feeding results back
/// there asks the model to continue from an answer that is not in its prompt, and it
/// calls the same tool again until the round budget is gone — with the request
/// succeeding throughout. And the round limit has to end in one more generation, because
/// returning a tool-call-only turn to the user would show them an empty reply.
/// </para>
/// </summary>
public class SkillAgentLoopTests : IDisposable
{
    private readonly string _baseDir;

    public SkillAgentLoopTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-loop-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
        string dir = Path.Combine(_baseDir, "pdf");
        Directory.CreateDirectory(dir);
        File.WriteAllText(
            Path.Combine(dir, "SKILL.md"),
            "---\nname: pdf\ndescription: does pdfs\n---\n\nRead the form, then fill it.\n");
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    // ---- helpers -----------------------------------------------------------

    private SkillToolContext Context() =>
        new(new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } }).Skills);

    private static List<ChatMessage> Conversation() => new()
    {
        new ChatMessage { Role = "system", Content = "You are terse." },
        new ChatMessage { Role = "user", Content = "Fill in this form." },
    };

    private static ParsedOutput Answer(string content) => new() { Content = content };

    private static ParsedOutput SkillCall(string path = "SKILL.md") => new()
    {
        ToolCalls = new List<ToolCall>
        {
            new()
            {
                Name = SkillTools.ReadToolName,
                Arguments = new Dictionary<string, object> { ["skill"] = "pdf", ["path"] = path },
            },
        },
    };

    private static ParsedOutput ClientCall(string name = "get_weather") => new()
    {
        ToolCalls = new List<ToolCall>
        {
            new() { Name = name, Arguments = new Dictionary<string, object> { ["city"] = "Paris" } },
        },
    };

    /// <summary>
    /// A generator that replays a scripted sequence of turns, repeating the last one
    /// once the script runs out, and counts how many times the host was asked to
    /// generate.
    /// </summary>
    private sealed class ScriptedGenerator
    {
        private readonly SkillTurnOutput[] _turns;
        public int Calls { get; private set; }
        public List<int> MessageCountsSeen { get; } = new();

        public ScriptedGenerator(params SkillTurnOutput[] turns) => _turns = turns;

        public Task<SkillTurnOutput> Generate(
            List<ChatMessage> messages, List<ToolFunction>? tools, CancellationToken cancellationToken)
        {
            MessageCountsSeen.Add(messages.Count);
            SkillTurnOutput turn = _turns[Math.Min(Calls, _turns.Length - 1)];
            Calls++;
            return Task.FromResult(turn);
        }
    }

    // ---- identical rounds ----------------------------------------------------

    [Fact]
    public async Task RunAsync_IdenticalRoundsAtTheLimit_AreAnsweredNotRun_AndOneMoreEndsTheTurn()
    {
        // The same skills_read, forever. With a limit of 3: rounds 1-2 run, round 3 is
        // answered with the refusal instead of being run, round 4 ends the turn.
        var generator = new ScriptedGenerator(new SkillTurnOutput(SkillCall()));
        var ran = new List<SkillToolInvocation>();

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate,
            new SkillAgentLoopOptions { MaxRounds = 10, MaxIdenticalRounds = 3, OnInvocation = ran.Add });

        Assert.Equal(4, generator.Calls);
        Assert.True(result.HitRoundLimit);
        Assert.StartsWith("(Stopped: the same skills_read call was made 4 rounds in a row",
            result.Output.Parsed!.Content, StringComparison.Ordinal);
        // Two real reads, then the refusal, which reaches the model as a tool result.
        Assert.Equal(new[] { true, true, false }, ran.Select(i => i.Ok || i.ResultBytes > 0).ToArray());
        ChatMessage refusal = result.Messages.Last(m => m.Role == "tool");
        Assert.StartsWith("Error: you have made this exact call 3 rounds in a row", refusal.Content, StringComparison.Ordinal);
        // The ending round is not left in the history as if it had been answered.
        Assert.Equal("tool", result.Messages[^1].Role);
    }

    [Fact]
    public async Task RunAsync_ARoundThatChangesItsArguments_ResetsTheCount()
    {
        var generator = new ScriptedGenerator(
            new SkillTurnOutput(SkillCall("a.md")),
            new SkillTurnOutput(SkillCall("a.md")),
            new SkillTurnOutput(SkillCall("b.md")),
            new SkillTurnOutput(SkillCall("b.md")),
            new SkillTurnOutput(Answer("done")));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate,
            new SkillAgentLoopOptions { MaxRounds = 10, MaxIdenticalRounds = 3 });

        Assert.Equal("done", result.Output.Parsed!.Content);
        Assert.False(result.HitRoundLimit);
        Assert.DoesNotContain(result.Messages, m => (m.Content ?? string.Empty).Contains("exact call", StringComparison.Ordinal));
    }

    [Fact]
    public async Task RunAsync_WithoutTheOption_IdenticalRoundsRunToTheRoundLimit()
    {
        // Off by default: the top-level loop's behaviour is unchanged.
        var generator = new ScriptedGenerator(new SkillTurnOutput(SkillCall()));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate, new SkillAgentLoopOptions { MaxRounds = 5 });

        Assert.Equal(6, generator.Calls);
        Assert.DoesNotContain(result.Messages, m => (m.Content ?? string.Empty).Contains("exact call", StringComparison.Ordinal));
        Assert.Equal(0, SkillAgentLoopOptions.Default.MaxIdenticalRounds);
    }

    [Fact]
    public void WithClientTools_KeepsTheIdenticalRoundLimit()
    {
        var options = new SkillAgentLoopOptions { MaxIdenticalRounds = 3 };
        Assert.Equal(3, options.WithClientTools(Array.Empty<ToolFunction>()).MaxIdenticalRounds);
    }

    // ---- stopping ----------------------------------------------------------

    [Fact]
    public async Task RunAsync_AnAnswerWithNoToolCalls_StopsAfterOneRound()
    {
        var generator = new ScriptedGenerator(new SkillTurnOutput(Answer("Here you go.")));
        List<ChatMessage> messages = Conversation();

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            messages, null, Context(), generator.Generate);

        Assert.Equal(1, result.Rounds);
        Assert.Equal(1, generator.Calls);
        Assert.False(result.HitRoundLimit);
        Assert.Empty(result.Invocations);
        Assert.Equal("Here you go.", result.Output.Parsed!.Content);

        // Nothing was appended: a request that needed no lookup must cost exactly what
        // it would have cost without skills.
        Assert.Equal(messages.Count, result.Messages.Count);
    }

    [Fact]
    public async Task RunAsync_OneLookupThenAnAnswer_RunsTwoRoundsAndAppendsAToolMessage()
    {
        var generator = new ScriptedGenerator(
            new SkillTurnOutput(SkillCall()),
            new SkillTurnOutput(Answer("The form needs your name.")));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate);

        Assert.Equal(2, result.Rounds);
        Assert.Equal(2, generator.Calls);
        Assert.False(result.HitRoundLimit);

        // assistant turn (with the call) then the result, so the next render can splice
        // the assistant turn back rather than re-tokenizing it.
        Assert.Equal("assistant", result.Messages[^2].Role);
        Assert.Equal("tool", result.Messages[^1].Role);
        Assert.Contains("Read the form, then fill it.", result.Messages[^1].Content, StringComparison.Ordinal);

        SkillToolInvocation invocation = Assert.Single(result.Invocations);
        Assert.Equal(1, invocation.Round);
        Assert.Equal(SkillTools.ReadToolName, invocation.Tool);
        Assert.Equal("pdf", invocation.SkillId);
        Assert.True(invocation.Ok);
    }

    [Fact]
    public async Task RunAsync_WhenToolResultsAreNotRendered_TheResultComesBackAsAUserTurn()
    {
        // Mistral 3's renderer handles only user and assistant and drops everything else
        // on the floor — silently, with the request still succeeding. Feeding results
        // back as a tool message there is the difference between skills working on that
        // family and appearing to work while doing nothing at all.
        var generator = new ScriptedGenerator(
            new SkillTurnOutput(SkillCall()),
            new SkillTurnOutput(Answer("The form needs your name.")));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate,
            new SkillAgentLoopOptions { ToolResultsAreRendered = false });

        Assert.Equal("user", result.Messages[^1].Role);
        Assert.StartsWith("Result of your skills_read call:", result.Messages[^1].Content, StringComparison.Ordinal);
        Assert.Contains("Read the form, then fill it.", result.Messages[^1].Content, StringComparison.Ordinal);
    }

    // ---- bounds ------------------------------------------------------------

    [Fact]
    public async Task RunAsync_AModelThatNeverStopsAsking_HitsTheRoundLimitAndStillAnswers()
    {
        // Each round is a full generation, so an unbounded loop is unbounded cost. The
        // limit has to end with one more generation: returning the last tool-call-only
        // turn would show the user nothing at all.
        var generator = new ScriptedGenerator(new SkillTurnOutput(SkillCall()));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate,
            new SkillAgentLoopOptions { MaxRounds = 2 });

        Assert.True(result.HitRoundLimit);
        Assert.Equal(3, generator.Calls);          // two rounds, then the forced answer
        Assert.Equal(3, result.Rounds);
        Assert.Equal(2, result.Invocations.Count);

        // The model is told why, in the conversation, so it can answer from what it has.
        Assert.Contains("limit on tool calls", result.Messages[^1].Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RunAsync_MoreCallsInOneTurnThanAllowed_AnswersTheFirstAndSaysSo()
    {
        // A model that emits a pile of reads in one turn is malfunctioning, and
        // answering all of them would blow the context before the next generation runs.
        var flood = new ParsedOutput
        {
            ToolCalls = new List<ToolCall>
            {
                new() { Name = SkillTools.ReadToolName, Arguments = new Dictionary<string, object> { ["skill"] = "pdf", ["path"] = "SKILL.md" } },
                new() { Name = SkillTools.ReadToolName, Arguments = new Dictionary<string, object> { ["skill"] = "pdf", ["path"] = "SKILL.md" } },
                new() { Name = SkillTools.ReadToolName, Arguments = new Dictionary<string, object> { ["skill"] = "pdf", ["path"] = "SKILL.md" } },
            },
        };
        var generator = new ScriptedGenerator(
            new SkillTurnOutput(flood),
            new SkillTurnOutput(Answer("Done.")));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate,
            new SkillAgentLoopOptions { MaxCallsPerRound = 1 });

        Assert.Single(result.Invocations);
        Assert.Contains(result.Messages, m => m.Content.Contains("too many tool calls", StringComparison.Ordinal));
    }

    // ---- the caller's own tools --------------------------------------------

    [Fact]
    public async Task RunAsync_OutOfRoundsAndStillCallingTools_StillReturnsWords()
    {
        // The model is told the budget is gone and asks for another tool anyway. Those
        // calls are dropped — nothing is left to run them — so a reply that was nothing
        // BUT calls used to reach the caller as an empty string, which is the same
        // silence the round cap exists to prevent.
        var generator = new ScriptedGenerator(new SkillTurnOutput(SkillCall()));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate,
            new SkillAgentLoopOptions { MaxRounds = 2 });

        Assert.True(result.HitRoundLimit);
        Assert.False(string.IsNullOrWhiteSpace(result.Output.Parsed!.Content),
            "an exhausted turn must never come back as an empty answer");
        Assert.Contains("tool-call budget", result.Output.Parsed.Content, StringComparison.Ordinal);
        Assert.Contains(SkillTools.ReadToolName, result.Output.Parsed.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RunAsync_AClientToolCall_StopsTheLoopAndIsHandedBackUnexecuted()
    {
        // Only the client knows what its tools do. Executing one here would be a
        // TensorSharp guess at somebody else's side effect.
        var generator = new ScriptedGenerator(new SkillTurnOutput(ClientCall()));
        List<ChatMessage> messages = Conversation();

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            messages, null, Context(), generator.Generate);

        Assert.Equal(1, result.Rounds);
        Assert.Equal(1, generator.Calls);
        Assert.Empty(result.Invocations);
        Assert.Equal(new[] { "get_weather" }, result.PendingClientToolCalls.Select(c => c.Name));
        Assert.Equal(messages.Count, result.Messages.Count);
    }

    [Fact]
    public async Task RunAsync_ASkillCallAlongsideAClientCall_AnswersTheSkillThenHandsBackTheRest()
    {
        var mixed = new ParsedOutput
        {
            ToolCalls = new List<ToolCall>
            {
                new() { Name = SkillTools.ReadToolName, Arguments = new Dictionary<string, object> { ["skill"] = "pdf", ["path"] = "SKILL.md" } },
                new() { Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Paris" } },
            },
        };
        var generator = new ScriptedGenerator(new SkillTurnOutput(mixed));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate);

        // The skill work it already did stays in the returned history, so the host can
        // hand the client tool back without losing the lookup.
        Assert.Single(result.Invocations);
        Assert.Equal(new[] { "get_weather" }, result.PendingClientToolCalls.Select(c => c.Name));
        Assert.Contains(result.Messages, m => m.Role == "tool");
    }

    // ---- what the loop must not touch ---------------------------------------

    [Fact]
    public async Task RunAsync_DoesNotMutateTheCallersMessageList()
    {
        // The caller's list is usually the tracked session history. Appending the loop's
        // internal turns to it would replay them on the next request forever.
        var generator = new ScriptedGenerator(
            new SkillTurnOutput(SkillCall()),
            new SkillTurnOutput(Answer("Done.")));
        List<ChatMessage> messages = Conversation();

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            messages, null, Context(), generator.Generate);

        Assert.Equal(2, messages.Count);
        Assert.NotSame(messages, result.Messages);
        Assert.True(result.Messages.Count > messages.Count);
    }

    [Fact]
    public async Task RunAsync_RawTokensSurviveOntoTheAppendedAssistantMessage()
    {
        // KVCachePromptRenderer splices an assistant turn's recorded tokens back into
        // the next render instead of re-tokenizing its text. Dropping them here still
        // produces a correct answer, but every round re-prefills the whole conversation
        // from the first assistant turn onward — a pure, silent slowdown.
        var generator = new ScriptedGenerator(
            new SkillTurnOutput(SkillCall(), new List<int> { 101, 102, 103 })
            {
                RawPromptTrailingWhitespace = string.Empty,
            },
            new SkillTurnOutput(Answer("Done.")));

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate);

        ChatMessage assistant = result.Messages.Single(m => m.Role == "assistant");
        Assert.Equal(new[] { 101, 102, 103 }, assistant.RawOutputTokens);
        Assert.Equal(string.Empty, assistant.RawPromptTrailingWhitespace);
        Assert.Single(assistant.ToolCalls!);
    }

    [Fact]
    public async Task RunAsync_ACancelledToken_StopsTheLoop()
    {
        // Checked between rounds and before each tool call, so a client that hung up
        // stops at the next boundary rather than after the whole round budget is spent.
        var generator = new ScriptedGenerator(new SkillTurnOutput(SkillCall()));
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate, null, cts.Token));

        Assert.Equal(0, generator.Calls);
    }

    [Fact]
    public async Task RunAsync_ReportsEachInvocationToTheHostAsItHappens()
    {
        // The hook is what a UI streams a "reading pdf/SKILL.md" trace from, and what
        // the server logs; a silent loop is impossible for an operator to reason about.
        var seen = new List<SkillToolInvocation>();
        var generator = new ScriptedGenerator(
            new SkillTurnOutput(SkillCall()),
            new SkillTurnOutput(Answer("Done.")));

        await SkillAgentLoop.RunAsync(
            Conversation(), null, Context(), generator.Generate,
            new SkillAgentLoopOptions { OnInvocation = seen.Add });

        Assert.Single(seen);
        Assert.Equal("pdf", seen[0].SkillId);
        Assert.Equal("SKILL.md", seen[0].ResourcePath);
        Assert.True(seen[0].ResultBytes > 0);
    }
}
