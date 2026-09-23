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
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Agents;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="SkillAgentLoop"/> as the parent agent's loop: it binds the conversation
/// the runtime copies from, awaits agent tools, folds finished sub-agents' answers into
/// the next tool result, and — the end-of-turn guard — never lets the parent answer
/// while answers it asked for are still on their way.
/// </summary>
public class SubAgentLoopTests
{
    private static Task<SkillLoopResult> RunParent(
        SubAgentHarness h, ScriptedAgent parent, SkillAgentLoopOptions? options = null) =>
        SkillAgentLoop.RunAsync(
            new List<ChatMessage>(h.Conversation), h.Tools, h.Runtime.RootContext, parent.Generate, options);

    private static Func<ToolCall> SpawnCall(string task, bool fork = false) =>
        () => fork
            ? Steps.Tool(SkillToolNames.SpawnAgent, ("message", task), ("fork_context", true))
            : Steps.Tool(SkillToolNames.SpawnAgent, ("message", task));

    [Fact]
    public async Task SpawnTwo_WaitForAll_ThenAnswer()
    {
        using var h = new SubAgentHarness();
        TaskCompletionSource release = Steps.Gate();
        h.Script("agent_1", Steps.After(release.Task, Steps.Answer("one done")));
        h.Script("agent_2", Steps.After(release.Task, Steps.Answer("two done")));
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("first half"), SpawnCall("second half")),
            // Both children are held until the parent's second generation, so neither can
            // be delivered at the first round's boundary; both are finished before the
            // wait is even called, so one wait returns both.
            Steps.Do(async _ =>
                {
                    release.TrySetResult();
                    await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed).ConfigureAwait(false);
                    await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed).ConfigureAwait(false);
                },
                Steps.Call(SkillToolNames.WaitAgent, ("targets", "all"))),
            Steps.Answer("final: combined"));

        SkillLoopResult result = await RunParent(h, parent).Within();

        Assert.Equal("final: combined", result.Output.Parsed!.Content);
        Assert.Equal(3, result.Rounds);
        Assert.False(result.HitRoundLimit);
        Assert.Equal(new[] { "spawn_agent", "spawn_agent", "wait_agent" }, result.Invocations.Select(i => i.Tool).ToArray());
        Assert.All(result.Invocations, i => Assert.True(i.Ok));

        SubAgentGeneration second = parent.Call(2);
        Assert.StartsWith("agent_1 started", second.Contents[^2], StringComparison.Ordinal);
        Assert.StartsWith("agent_2 started", second.Contents[^1], StringComparison.Ordinal);
        Assert.DoesNotContain("<subagent_notification>", second.AllText, StringComparison.Ordinal);

        SubAgentGeneration third = parent.Call(3);
        Assert.Equal("tool", third.LastRole);
        Assert.Contains("agent_1 completed in", third.LastContent, StringComparison.Ordinal);
        Assert.Contains("one done", third.LastContent, StringComparison.Ordinal);
        Assert.Contains("agent_2 completed in", third.LastContent, StringComparison.Ordinal);
        Assert.Contains("two done", third.LastContent, StringComparison.Ordinal);
        Assert.Equal(1, third.AllText.Occurrences("one done"));
        Assert.Equal(1, third.AllText.Occurrences("two done"));
    }

    [Fact]
    public async Task TheLoopBindsItsOwnWorkingConversation_SoAForkSeesTheSpawningTurn()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Answer("forked answer"));
        var parent = new ScriptedAgent(
            _ => Task.FromResult(new SkillTurnOutput(
                new ParsedOutput
                {
                    Content = "Delegating.",
                    ToolCalls = new List<ToolCall> { SpawnCall("Keep going from here.", fork: true)() },
                },
                RawTokens: new[] { 7, 8, 9 })),
            Steps.Call(SkillToolNames.WaitAgent, ("targets", "all")),
            Steps.Answer("done"));

        SkillLoopResult result = await RunParent(h, parent).Within();

        Assert.Equal("done", result.Output.Parsed!.Content);
        SubAgentGeneration first = child.Call(1);
        Assert.Equal(new[] { "system", "developer", "user", "assistant", "tool", "user" }, first.Roles);
        Assert.Equal(SubAgentHarness.UserRequest, first.Contents[2]);
        Assert.Equal(new[] { 7, 8, 9 }, first.Messages[3].RawOutputTokens);
        Assert.Equal("Delegating.", first.Contents[3]);
        Assert.Contains("You were forked from the conversation above", first.LastContent, StringComparison.Ordinal);
        Assert.True(h.Launches.Single().ForkContext);
    }

    [Fact]
    public async Task EndOfTurnGuard_AParentThatAnswersWithoutWaiting_GetsTheResultsAndOneMoreRound()
    {
        using var h = new SubAgentHarness();
        TaskCompletionSource release = Steps.Gate();
        h.Script("agent_1", Steps.After(release.Task, Steps.Answer("one done")));
        h.Script("agent_2", Steps.After(release.Task, Steps.Answer("two done")));
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("first half"), SpawnCall("second half")),
            Steps.Do(_ => { release.TrySetResult(); return Task.CompletedTask; }, Steps.Answer("premature answer")),
            Steps.Answer("final with results"));

        SkillLoopResult result = await RunParent(h, parent).Within();

        Assert.Equal("final with results", result.Output.Parsed!.Content);
        Assert.Equal(3, result.Rounds);

        SubAgentGeneration third = parent.Call(3);
        Assert.Equal("assistant", third.Roles[^2]);
        Assert.Equal("premature answer", third.Contents[^2]);
        Assert.Equal("user", third.LastRole);
        Assert.StartsWith("Before you finish: sub-agents you started had not reported back to you. They have finished, and their results are below.",
            third.LastContent, StringComparison.Ordinal);
        Assert.Contains("<subagent_notification>", third.LastContent, StringComparison.Ordinal);
        Assert.Contains("one done", third.LastContent, StringComparison.Ordinal);
        Assert.Contains("two done", third.LastContent, StringComparison.Ordinal);
        Assert.EndsWith("Now write your final answer to the user, using these results. Do not redo work the sub-agents already did.",
            third.LastContent, StringComparison.Ordinal);
        Assert.False(h.Runtime.Root.HasOutstandingWork);
    }

    [Fact]
    public async Task EndOfTurnGuard_WithNoRoundLeft_StopsTheChildrenAndSaysSo()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent one = h.Script("agent_1", Steps.Block());
        ScriptedAgent two = h.Script("agent_2", Steps.Block());
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("first half"), SpawnCall("second half")),
            // Answer only once both are generating, so "stopped" means a live generation
            // was cancelled, not that a worker was closed before it began.
            Steps.Do(async _ =>
                {
                    await one.Entered(1).ConfigureAwait(false);
                    await two.Entered(1).ConfigureAwait(false);
                },
                Steps.Answer("done")));

        SkillLoopResult result = await RunParent(h, parent, new SkillAgentLoopOptions { MaxRounds = 2 }).Within();

        Assert.Equal(
            "done\n\n(Sub-agents agent_1, agent_2 were still working when this turn ran out of rounds and were stopped; "
            + "this answer does not include their results.)",
            result.Output.Parsed!.Content);
        Assert.Equal(2, result.Rounds);
        await one.Cancelled.Within();
        await two.Cancelled.Within();
        Assert.Equal(SubAgentStatus.Closed, h.Snapshot("agent_1").Status);
        Assert.Equal(SubAgentStatus.Closed, h.Snapshot("agent_2").Status);
    }

    [Fact]
    public async Task EndOfTurnGuard_WithNoRoundLeft_OneChild_UsesTheSingular()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Block());
        var parent = new ScriptedAgent(Steps.Calls(SpawnCall("only")), Steps.Answer(""));

        SkillLoopResult result = await RunParent(h, parent, new SkillAgentLoopOptions { MaxRounds = 2 }).Within();

        Assert.Equal(
            "(Sub-agent agent_1 was still working when this turn ran out of rounds and was stopped; "
            + "this answer does not include its results.)",
            result.Output.Parsed!.Content);
    }

    /// <summary>
    /// The design's end-of-turn rule is "never silent": with no round left, children are
    /// stopped AND the answer says so. A child that FINISHED during the last generation is
    /// not stopped, so today it is neither delivered nor mentioned — its result is dropped
    /// without a word when the runtime is disposed. Whether the fix is a note (as for a
    /// stopped child) or one extra forced generation (as the tool-call round limit already
    /// does) is the author's call; this pins only that the answer is not silent about it.
    /// </summary>
    [Fact]
    public async Task EndOfTurnGuard_WithNoRoundLeft_ACompletedButUnseenChildIsNotDroppedSilently()
    {
        using var h = new SubAgentHarness();
        TaskCompletionSource release = Steps.Gate();
        h.Script("agent_1", Steps.After(release.Task, Steps.Answer("child answer")));
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("look into it")),
            Steps.Do(async _ =>
                {
                    release.TrySetResult();
                    await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed).ConfigureAwait(false);
                },
                Steps.Answer("done")));

        SkillLoopResult result = await RunParent(h, parent, new SkillAgentLoopOptions { MaxRounds = 2 }).Within();

        // Its result exists and was paid for: the user is shown it, and told the answer
        // above does not use it — and it is delivered exactly once, here.
        string answer = result.Output.Parsed!.Content;
        Assert.StartsWith("done\n\n(Sub-agent agent_1 finished after this answer was written", answer, StringComparison.Ordinal);
        Assert.Contains("child answer", answer, StringComparison.Ordinal);
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task AFinishedChildsAnswer_IsFoldedIntoTheParentsNextToolResult_Once()
    {
        using var h = new SubAgentHarness();
        TaskCompletionSource release = Steps.Gate();
        h.Script("agent_1", Steps.After(release.Task, Steps.Answer("child answer")));
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("look into it")),
            Steps.Do(async _ =>
                {
                    release.TrySetResult();
                    await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed).ConfigureAwait(false);
                },
                Steps.Call("no_such_tool")),
            Steps.Answer("ok"));

        SkillLoopResult result = await RunParent(
            h, parent, new SkillAgentLoopOptions { ClientTools = Array.Empty<ToolFunction>() }).Within();

        Assert.Equal("ok", result.Output.Parsed!.Content);
        Assert.Equal(3, result.Rounds);

        SubAgentGeneration third = parent.Call(3);
        Assert.Equal("tool", third.LastRole);
        Assert.StartsWith("Error: there is no tool called 'no_such_tool'", third.LastContent, StringComparison.Ordinal);
        Assert.Contains("\n\n<subagent_notification>\nagent_1 completed in ", third.LastContent, StringComparison.Ordinal);
        Assert.EndsWith("child answer\n</subagent_notification>", third.LastContent, StringComparison.Ordinal);
        Assert.Equal(1, third.AllText.Occurrences("child answer"));
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task RoundLimit_WithAChildStillOut_TheForcedAnswerSeesItsResultExactlyOnce()
    {
        using var h = new SubAgentHarness();
        TaskCompletionSource parentAtRoundTwo = Steps.Gate();
        // The child finishes shortly after the parent's last tool round; the loop then
        // collects it before forcing the answer. (The delay only makes that ordering
        // likely: the assertions hold whichever way the delivery went.)
        h.Script("agent_1", Steps.After(parentAtRoundTwo.Task,
            Steps.Do(g => Task.Delay(200, g.Token), Steps.Answer("child answer"))));
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("look into it")),
            Steps.Do(_ => { parentAtRoundTwo.TrySetResult(); return Task.CompletedTask; }, Steps.Call("no_such_tool")),
            Steps.Answer("forced answer"));

        SkillLoopResult result = await RunParent(h, parent, new SkillAgentLoopOptions
        {
            MaxRounds = 2,
            ClientTools = Array.Empty<ToolFunction>(),
        }).Within();

        Assert.True(result.HitRoundLimit);
        Assert.Equal("forced answer", result.Output.Parsed!.Content);
        SubAgentGeneration last = parent.Call(3);
        Assert.Equal("user", last.LastRole);
        Assert.StartsWith("Error: the limit on tool calls for this turn has been reached.", last.LastContent, StringComparison.Ordinal);
        Assert.Equal(1, last.AllText.Occurrences("child answer"));
        Assert.False(h.Runtime.Root.HasOutstandingWork);
    }

    [Fact]
    public async Task AnAgentCallWithoutARuntime_IsAnsweredInConversation()
    {
        var parent = new ScriptedAgent(
            Steps.Call(SkillToolNames.SpawnAgent, ("message", "do it")),
            Steps.Answer("did it myself"));
        var conversation = new List<ChatMessage> { new() { Role = "user", Content = "hi" } };

        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            conversation, SubAgentTools.Declare(), new SkillToolContext(Array.Empty<Skill>()), parent.Generate).Within();

        Assert.Equal("did it myself", result.Output.Parsed!.Content);
        Assert.Equal("Error: sub-agents are not enabled on this host. Do the task yourself.", parent.Call(2).LastContent);
    }

    [Fact]
    public async Task ACallersOwnToolNamedLikeAnAgentTool_IsHandedBackToTheCaller()
    {
        using var h = new SubAgentHarness();
        var parent = new ScriptedAgent(Steps.Call(SkillToolNames.SpawnAgent, ("message", "x")));

        SkillLoopResult result = await RunParent(h, parent, new SkillAgentLoopOptions
        {
            ClientTools = new[] { new ToolFunction { Name = SkillToolNames.SpawnAgent } },
        }).Within();

        Assert.Equal(new[] { "spawn_agent" }, result.PendingClientToolCalls.Select(c => c.Name).ToArray());
        Assert.Equal(0, h.Runtime.SpawnedCount);
    }

    [Fact]
    public async Task ACancelledTurn_StopsAParentBlockedInWaitAgent()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Block());
        var parent = new ScriptedAgent(
            Steps.Calls(SpawnCall("forever")),
            Steps.Call(SkillToolNames.WaitAgent, ("targets", "all")));
        using var cts = new CancellationTokenSource();

        Task<SkillLoopResult> running = SkillAgentLoop.RunAsync(
            new List<ChatMessage>(h.Conversation), h.Tools, h.Runtime.RootContext, parent.Generate, null, cts.Token);
        await parent.Entered(2).Within();
        await child.Entered(1).Within();
        Assert.False(running.IsCompleted);

        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => running.Within());
    }
}
