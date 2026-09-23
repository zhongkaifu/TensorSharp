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
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Agents;

namespace InferenceWeb.Tests;

/// <summary>
/// Regressions for the sub-agent fixes that followed the six-lens review: the record a
/// parent is given about what its agent did, what the user and a closing parent are
/// told about answers nobody read, what a turn that ends on a hand-back or at the
/// stream's final update says about agents still out, and which user-role messages are
/// the HOST's rather than the user's. Every test here failed on the code before those
/// fixes; each is driven by scripted agents, so every ordering is forced.
/// </summary>
public class SubAgentReviewFixTests
{
    private static readonly ToolFunction Weather = new() { Name = "get_weather" };

    private static SkillAgentLoopOptions WithWeatherAsClientTool(int maxRounds = 8) => new()
    {
        MaxRounds = maxRounds,
        ClientTools = new[] { Weather },
    };

    private static Task<SkillLoopResult> RunParent(
        SubAgentHarness h, ScriptedAgent parent, SkillAgentLoopOptions? options = null) =>
        SkillAgentLoop.RunAsync(
            new List<ChatMessage>(h.Conversation), h.Tools, h.Runtime.RootContext, parent.Generate, options);

    private static Func<ToolCall> SpawnCall(string task) =>
        () => Steps.Tool(SkillToolNames.SpawnAgent, ("message", task));

    // ---- 1. a steered follow-up keeps the task's work record -----------------------------

    [Fact]
    public async Task SteeredFollowUp_ArrivingWhileTheAnswerIsWritten_KeepsTheRoundsAndToolsAlreadySpent()
    {
        using var h = new SubAgentHarness();
        var answerGate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.ListAgents),
            Steps.After(answerGate.Task, Steps.Answer("first answer")),
            Steps.Answer("final answer"));

        await h.Spawn("Survey the repository.").Within();
        await child.Entered(2).Within();   // round 1 ran list_agents; round 2 is generating

        SkillToolResult sent = await h.Root(SkillToolNames.SendInput,
            ("target", "agent_1"), ("message", "Include the test counts.")).Within(2);
        Assert.Equal("agent_1 is still working; it will read your message before its next step.", sent.Content);
        answerGate.SetResult();

        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within();

        // The follow-up is the same task carrying on, so its record is the whole task's:
        // three rounds, and the list_agents call of round 1 — never "it called no tools".
        Assert.Contains("agent_1 completed in ", waited.Content, StringComparison.Ordinal);
        Assert.Contains("(3 rounds; tools it ran: list_agents)", waited.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("it called no tools", waited.Content, StringComparison.Ordinal);
        Assert.Contains("Its final answer:\nfinal answer", waited.Content, StringComparison.Ordinal);
        Assert.Equal(3, child.CallCount);

        SubAgentSnapshot snapshot = h.Snapshot("agent_1");
        Assert.Equal(1, snapshot.Turns);
        Assert.Equal(3, snapshot.Rounds);
        Assert.Equal(1, snapshot.ToolCalls);
    }

    // ---- 2. client-call re-runs carry earlier tool calls forward ---------------------------

    [Fact]
    public async Task ARefusedClientCall_KeepsTheToolsTheAgentRanBeforeIt_InTheRecord()
    {
        using var h = new SubAgentHarness(loopOptions: WithWeatherAsClientTool());
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.ListAgents),
            Steps.Call("get_weather", ("city", "Paris")),
            Steps.Answer("no weather available"));

        await h.Spawn("Report the weather.").Within();
        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within();

        Assert.StartsWith("Error: 'get_weather' belongs to the application the user is working in",
            child.Call(3).LastContent, StringComparison.Ordinal);
        Assert.Contains("(3 rounds; tools it ran: list_agents)", waited.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("it called no tools", waited.Content, StringComparison.Ordinal);
        Assert.Equal(1, h.Snapshot("agent_1").ToolCalls);
        Assert.Equal(3, h.Snapshot("agent_1").Rounds);
    }

    [Fact]
    public async Task ARefusedClientCall_AfterRealWork_IsNotCorrectedAsAnEmptyTurn()
    {
        // The empty-turn correction is for a turn that ran no tool at all. This agent ran
        // one before the refused client call, so its bare ending is not corrected.
        using var h = new SubAgentHarness(loopOptions: WithWeatherAsClientTool());
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.ListAgents),
            Steps.Call("get_weather", ("city", "Paris")),
            Steps.Answer(string.Empty));

        await h.Spawn("Report the weather.").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        Assert.Equal(3, child.CallCount);
        Assert.DoesNotContain(child.Calls, g => g.AllText.Contains(
            "You ended your turn without doing anything", StringComparison.Ordinal));
        Assert.Equal("(The sub-agent ended without writing a final answer.)", h.Snapshot("agent_1").Result);
        Assert.Equal(3, h.Snapshot("agent_1").Rounds);
    }

    // ---- 3. answers handed over at close / to the user are whole and name no tool ----------

    [Fact]
    public async Task Close_OfAnUnseenLongAnswer_HandsOverTheWholeAnswer_WithoutASendInputHint()
    {
        using var h = new SubAgentHarness();
        string answer = new string('a', SubAgentRuntime.MaxResultChars) + new string('b', 1000);
        h.Script("agent_1", Steps.Answer(answer));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SkillToolResult closed = await h.Root(SkillToolNames.CloseAgent, ("target", "agent_1")).Within(2);

        Assert.True(closed.Ok, closed.Content);
        Assert.Contains("Its final answer:\n" + answer + "\n\n", closed.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("truncated", closed.Content, StringComparison.Ordinal);
        Assert.DoesNotContain(SkillToolNames.SendInput, closed.Content, StringComparison.Ordinal);
        Assert.EndsWith("agent_1 closed; it was completed.", closed.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task EndWithoutRound_ShowsTheUserTheWholeLongAnswer_WithoutASendInputHint()
    {
        using var h = new SubAgentHarness();
        string answer = new string('a', SubAgentRuntime.MaxResultChars) + new string('b', 1000);
        h.Script("agent_1", Steps.Answer(answer));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        string? note = SubAgentConversation.EndWithoutRound(h.Runtime.Root);

        Assert.NotNull(note);
        Assert.StartsWith("(Sub-agent agent_1 finished after this answer was written", note, StringComparison.Ordinal);
        Assert.EndsWith("Its final answer:\n" + answer, note, StringComparison.Ordinal);
        Assert.DoesNotContain("truncated", note, StringComparison.Ordinal);
        Assert.DoesNotContain(SkillToolNames.SendInput, note, StringComparison.Ordinal);
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task Close_OfAnUnseenFailedAgent_HandsOverTheErrorWithoutASendInputHint()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Throw("backend exploded"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Errored);

        SkillToolResult closed = await h.Root(SkillToolNames.CloseAgent, ("target", "agent_1")).Within(2);

        Assert.True(closed.Ok, closed.Content);
        Assert.StartsWith("agent_1 failed after ", closed.Content, StringComparison.Ordinal);
        Assert.Contains(": backend exploded\n\nagent_1 closed; it was failed.", closed.Content, StringComparison.Ordinal);
        Assert.DoesNotContain(SkillToolNames.SendInput, closed.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("If you still need its result", closed.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task EndWithoutRound_OfAnUnseenFailedAgent_NamesNoTool()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Throw("backend exploded"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Errored);

        string? note = SubAgentConversation.EndWithoutRound(h.Runtime.Root);

        Assert.NotNull(note);
        Assert.Contains("agent_1 failed after ", note, StringComparison.Ordinal);
        Assert.EndsWith(": backend exploded", note, StringComparison.Ordinal);
        Assert.DoesNotContain(SkillToolNames.SendInput, note, StringComparison.Ordinal);
    }

    // ---- 4. a top-level client-call hand-back says what became of the agents ---------------

    [Fact]
    public async Task HandBack_WithAFinishedButUnseenChild_EndsTheAnswerWithItsResult()
    {
        using var h = new SubAgentHarness();
        var release = Steps.Gate();
        h.Script("agent_1", Steps.After(release.Task, Steps.Answer("child answer")));
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("look into it")),
            Steps.Do(async _ =>
                {
                    release.TrySetResult();
                    await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed).ConfigureAwait(false);
                },
                _ => Task.FromResult(Steps.Turn("Checking the weather.", Steps.Tool("get_weather", ("city", "Paris"))))));

        SkillLoopResult result = await RunParent(h, parent, WithWeatherAsClientTool()).Within();

        Assert.Equal(new[] { "get_weather" }, result.PendingClientToolCalls.Select(c => c.Name).ToArray());
        string content = result.Output.Parsed!.Content;
        Assert.StartsWith("Checking the weather.\n\n(Sub-agent agent_1 finished after this answer was written, "
            + "so this answer does not use its results. It reported:)\nagent_1 completed in ", content, StringComparison.Ordinal);
        Assert.EndsWith("Its final answer:\nchild answer", content, StringComparison.Ordinal);
        Assert.False(h.Runtime.Root.HasOutstandingWork);
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task HandBack_InARoundThatAlsoRanATool_StopsAWorkingChildAndSaysSo()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Block());
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("forever")),
            Steps.Do(_ => child.Entered(1),
                Steps.Calls(
                    () => Steps.Tool(SkillToolNames.ListAgents),
                    () => Steps.Tool("get_weather", ("city", "Paris")))));

        SkillLoopResult result = await RunParent(h, parent, WithWeatherAsClientTool()).Within();

        Assert.Equal(new[] { "get_weather" }, result.PendingClientToolCalls.Select(c => c.Name).ToArray());
        Assert.Equal(
            "(Sub-agent agent_1 was still working when this turn handed a tool call back to the application "
            + "and was stopped; this answer does not include its results.)",
            result.Output.Parsed!.Content);
        await child.Cancelled.Within();
        Assert.Equal(SubAgentStatus.Closed, h.Snapshot("agent_1").Status);
    }

    [Fact]
    public async Task ASubAgentsOwnHandBack_DoesNotStopItsSubAgents()
    {
        // A depth-1 agent's client call is answered by the runtime (refused) and the agent
        // carries on, so its own sub-agents must still be there when it does.
        using var h = new SubAgentHarness(maxDepth: 2, loopOptions: WithWeatherAsClientTool());
        var grandGate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.SpawnAgent, ("message", "grand task")),
            Steps.Call("get_weather", ("city", "Paris")),
            Steps.Do(_ => { grandGate.TrySetResult(); return Task.CompletedTask; },
                Steps.Call(SkillToolNames.WaitAgent, ("targets", "all"))),
            Steps.Answer("child done"));
        ScriptedAgent grandchild = h.Script("agent_2", Steps.After(grandGate.Task, Steps.Answer("grand done")));

        await h.Spawn("Delegate and check the weather.").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        Assert.False(grandchild.Cancelled.IsCompleted, "the grandchild was stopped by its parent's hand-back");
        Assert.Equal(SubAgentStatus.Completed, h.Snapshot("agent_2").Status);
        Assert.Equal("grand done", h.Snapshot("agent_2").Result);
        Assert.StartsWith("Error: 'get_weather' belongs to the application", child.Call(3).LastContent, StringComparison.Ordinal);
        Assert.Contains("grand done", child.Call(4).LastContent, StringComparison.Ordinal);
        Assert.Equal("child done", h.Snapshot("agent_1").Result);
    }

    // ---- 5. the server settles agents at the stream's final update -------------------------

    private static ChatStreamUpdate Final() => new(string.Empty, true, 3, 4, 0, 0, 0, 0, "stop");

    private static async IAsyncEnumerable<ChatStreamUpdate> Stream(
        IEnumerable<ChatStreamUpdate> updates, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (ChatStreamUpdate update in updates)
        {
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            yield return update;
        }
    }

    private static async Task<List<ChatStreamUpdate>> Drain(IAsyncEnumerable<ChatStreamUpdate> stream)
    {
        var all = new List<ChatStreamUpdate>();
        await foreach (ChatStreamUpdate update in stream)
            all.Add(update);
        return all;
    }

    [Fact]
    public async Task SettleAtEnd_AWorkingChild_IsStoppedAndSaidJustBeforeTheFinalUpdate()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Block());
        await h.Spawn("forever").Within();
        await child.Entered(1).Within();
        ChatStreamUpdate text = ChatStreamUpdate.Parsed("the answer", null, null);
        ChatStreamUpdate final = Final();

        List<ChatStreamUpdate> seen = await Drain(
            ModelService.SettleSubAgentsAtEnd(Stream(new[] { text, final }), h.Runtime.Root)).Within();

        Assert.Equal(3, seen.Count);
        Assert.Equal(text, seen[0]);
        Assert.True(seen[1].IsParsed);
        Assert.False(seen[1].Done);
        Assert.Equal(
            "\n\n(Sub-agent agent_1 was still working when this turn ended and was stopped; "
            + "this answer does not include its results.)",
            seen[1].Piece);
        Assert.Equal(final, seen[2]);
        await child.Cancelled.Within();
        Assert.Equal(SubAgentStatus.Closed, h.Snapshot("agent_1").Status);
    }

    [Fact]
    public async Task SettleAtEnd_AFinishedButUnseenChild_IsShownJustBeforeTheFinalUpdate()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("child answer"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        ChatStreamUpdate final = Final();

        List<ChatStreamUpdate> seen = await Drain(
            ModelService.SettleSubAgentsAtEnd(Stream(new[] { ChatStreamUpdate.Text("x"), final }), h.Runtime.Root)).Within();

        Assert.Equal(3, seen.Count);
        Assert.True(seen[1].IsParsed);
        Assert.StartsWith("\n\n(Sub-agent agent_1 finished after this answer was written", seen[1].Piece, StringComparison.Ordinal);
        Assert.EndsWith("Its final answer:\nchild answer", seen[1].Piece, StringComparison.Ordinal);
        Assert.Equal(final, seen[2]);
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task SettleAtEnd_WithNothingOutstanding_PassesTheStreamThroughUnchanged()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("child answer"));
        await h.Spawn("one").Within();
        await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within();   // delivered
        var updates = new[]
        {
            ChatStreamUpdate.Text("a"),
            ChatStreamUpdate.Parsed("b", "thinking", null),
            Final(),
        };

        List<ChatStreamUpdate> seen = await Drain(
            ModelService.SettleSubAgentsAtEnd(Stream(updates), h.Runtime.Root)).Within();

        Assert.Equal(updates, seen);
    }

    // ---- 6. handovers are the host's, never the user's request -----------------------------

    [Fact]
    public async Task EndOfTurnGuard_Handover_IsAHostAuthoredUserMessage()
    {
        using var h = new SubAgentHarness();
        var release = Steps.Gate();
        h.Script("agent_1", Steps.After(release.Task, Steps.Answer("one done")));
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("first half")),
            Steps.Do(_ => { release.TrySetResult(); return Task.CompletedTask; }, Steps.Answer("premature answer")),
            Steps.Answer("final with results"));

        SkillLoopResult result = await RunParent(h, parent).Within();

        Assert.Equal("final with results", result.Output.Parsed!.Content);
        ChatMessage handover = parent.Call(3).Messages[^1];
        Assert.IsAssignableFrom<HostAuthoredUserMessage>(handover);
        Assert.Equal("user", handover.Role);
        Assert.StartsWith("Before you finish:", handover.Content, StringComparison.Ordinal);
        Assert.Contains(result.Messages, m => ReferenceEquals(m, handover));
    }

    [Fact]
    public async Task AppendDeliveries_TheFallbackUserNotification_IsAHostAuthoredUserMessage()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("the answer"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        var working = new List<ChatMessage>
        {
            new() { Role = "user", Content = "u" },
            new() { Role = "assistant", Content = "done" },
        };

        Assert.True(SubAgentConversation.AppendDeliveries(working, h.Runtime.Root));

        Assert.Equal(3, working.Count);
        Assert.IsAssignableFrom<HostAuthoredUserMessage>(working[2]);
        Assert.Equal("user", working[2].Role);
        Assert.StartsWith("<subagent_notification>", working[2].Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task EmptyTurnCorrection_IsAHostAuthoredUserMessage()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Answer(string.Empty), Steps.Answer("did it"));
        await h.Spawn("Do the thing.").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        ChatMessage correction = child.Call(2).Messages[^1];
        Assert.IsAssignableFrom<HostAuthoredUserMessage>(correction);
        Assert.Equal("user", correction.Role);
        Assert.StartsWith("You ended your turn without doing anything", correction.Content, StringComparison.Ordinal);
        Assert.Equal("did it", h.Snapshot("agent_1").Result);
    }

    [Fact]
    public void Compaction_OfAHistoryEndingOnAHostHandover_KeepsTheGenuineRequestAsItsAnchor()
    {
        const string task = "GENUINE_USER_REQUEST_SENTINEL";
        const string premature = "PREMATURE_ANSWER_SENTINEL";
        const string handover = "HOST_HANDOVER_SENTINEL";
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "SYSTEM_SENTINEL" },
            new() { Role = "user", Content = "OLD_TASK_" + new string('u', 4_000) },
            new() { Role = "assistant", Content = "OLD_ANSWER_" + new string('a', 4_000) },
            new() { Role = "user", Content = task },
            new() { Role = "assistant", Content = "OLD_SPAWN_ROUND_" + new string('r', 4_000) },
            new() { Role = "tool", Content = "OLD_SPAWN_RESULT_" + new string('f', 4_000) },
            new() { Role = "assistant", Content = premature },
            HostAuthoredUserMessage.Create(handover),
        };

        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContext(history, Cost(history), promptTokenLimit: 1_000, Cost);

        Assert.Contains(window.History, message => message.Content == task);
        Assert.Contains(window.History, message => message.Content == premature);
        Assert.Contains(window.History, message => message.Content == handover);
        Assert.DoesNotContain(window.History, message =>
            message.Content.StartsWith("OLD_TASK_", StringComparison.Ordinal));
        Assert.True(window.FinalPromptTokens <= 1_000);
    }

    [Fact]
    public void TheChatLoopsCompletionCorrection_IsAHostAuthoredUserMessage()
    {
        // One marker for every host note: compaction checks the base type, so the chat
        // loop's correction must be one.
        Assert.True(typeof(HostAuthoredUserMessage).IsAssignableFrom(
            typeof(TensorSharp.Server.Skills.SkillChatLoop.HostCompletionCorrectionMessage)));
        HostAuthoredUserMessage note = HostAuthoredUserMessage.Create("note");
        Assert.Equal("user", note.Role);
        Assert.Equal("note", note.Content);
    }
}
