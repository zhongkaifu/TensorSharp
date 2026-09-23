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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Agents;

namespace InferenceWeb.Tests;

/// <summary>
/// The sub-agent runtime driven through the top-level agent's scope, with every child
/// generating through a scripted agent. Children that must still be "working" at some
/// point are gated on a <see cref="TaskCompletionSource"/> the test releases, so every
/// ordering asserted here is forced rather than hoped for.
/// </summary>
public class SubAgentRuntimeTests
{
    // ---- spawn ---------------------------------------------------------------------

    [Fact]
    public async Task Spawn_ReturnsAtOnce_WhileTheChildIsStillGenerating()
    {
        using var h = new SubAgentHarness();
        var gate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1", Steps.After(gate.Task, Steps.Answer("four files")));

        SkillToolResult spawned = await h.Spawn("Count the files.").Within();

        Assert.True(spawned.Ok, spawned.Content);
        Assert.StartsWith("agent_1 started. It is working in the background;", spawned.Content, StringComparison.Ordinal);
        await child.Entered(1).Within();
        Assert.Equal(SubAgentStatus.Running, h.Snapshot("agent_1").Status);

        gate.SetResult();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        Assert.Equal("four files", h.Snapshot("agent_1").Result);
        Assert.False(h.Snapshot("agent_1").ResultDelivered);
    }

    [Fact]
    public async Task Spawn_AssignsSequentialShortIds_AndTellsTheHostWhoItIs()
    {
        using var h = new SubAgentHarness();

        for (int i = 1; i <= 3; i++)
        {
            SkillToolResult spawned = await h.Spawn("task " + i).Within();
            Assert.True(spawned.Ok, spawned.Content);
            Assert.StartsWith($"agent_{i} started", spawned.Content, StringComparison.Ordinal);
        }

        Assert.Equal(new[] { "agent_1", "agent_2", "agent_3" }, h.Launches.Select(l => l.AgentId).ToArray());
        Assert.All(h.Launches, l =>
        {
            Assert.Null(l.ParentId);
            Assert.Equal(1, l.Depth);
            Assert.False(l.ForkContext);
        });
        Assert.Equal(3, h.Runtime.SpawnedCount);
        Assert.Equal(new[] { "task 1", "task 2", "task 3" }, h.Runtime.Snapshot().Select(s => s.Task).ToArray());
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   \n ")]
    public async Task Spawn_ABlankMessage_IsRefused(string? message)
    {
        using var h = new SubAgentHarness();
        SkillToolResult result = message == null
            ? await h.Root(SkillToolNames.SpawnAgent).Within()
            : await h.Root(SkillToolNames.SpawnAgent, ("message", message)).Within();

        Assert.False(result.Ok);
        Assert.Equal("Error: Empty message can't be sent to an agent. Put the agent's task in 'message'.", result.Content);
        Assert.Equal(0, h.Runtime.SpawnedCount);
        Assert.Empty(h.Launches);
    }

    [Fact]
    public async Task Spawn_AcceptsTheTaskUnderACommonAlias()
    {
        using var h = new SubAgentHarness();
        SkillToolResult result = await h.Root(SkillToolNames.SpawnAgent, ("task", "  Summarise the log.  ")).Within();
        Assert.True(result.Ok, result.Content);
        Assert.Equal("Summarise the log.", h.Snapshot("agent_1").Task);
    }

    [Fact]
    public async Task Spawn_BeforeTheConversationIsBound_IsRefused()
    {
        var runtime = new SubAgentRuntime(
            new SubAgentOptions { Enabled = true },
            new SubAgentHostBinding { CreateGenerator = _ => new ScriptedAgent().Generate },
            new SkillToolContext(Array.Empty<Skill>()));
        using (runtime)
        {
            SkillToolResult result = await runtime.Root.ExecuteAsync(
                Steps.Tool(SkillToolNames.SpawnAgent, ("message", "x")), null, CancellationToken.None).Within();
            Assert.False(result.Ok);
            Assert.Equal("Error: sub-agents are not available in this conversation. Do the task yourself.", result.Content);
            Assert.Equal(0, runtime.SpawnedCount);
        }
    }

    [Fact]
    public async Task Spawn_WhenTheHostCannotStartTheAgent_IsAnErrorResult()
    {
        var runtime = new SubAgentRuntime(
            new SubAgentOptions { Enabled = true },
            new SubAgentHostBinding { CreateGenerator = _ => throw new InvalidOperationException("no engine") },
            new SkillToolContext(Array.Empty<Skill>()));
        using (runtime)
        {
            runtime.Root.Bind(new List<ChatMessage> { new() { Role = "system", Content = "s" } }, null);
            SkillToolResult result = await runtime.Root.ExecuteAsync(
                Steps.Tool(SkillToolNames.SpawnAgent, ("message", "x")), null, CancellationToken.None).Within();
            Assert.False(result.Ok);
            Assert.Equal("Error: the agent could not be started: no engine", result.Content);
            Assert.Equal(SubAgentStatus.Closed, runtime.Snapshot().Single().Status);
            Assert.False(runtime.Root.HasOutstandingWork);
        }
    }

    // ---- depth -----------------------------------------------------------------------

    [Fact]
    public async Task Spawn_FromAChildAtTheDepthLimit_IsRefusedWithTheExactMessage()
    {
        using var h = new SubAgentHarness(maxDepth: 1);
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.SpawnAgent, ("message", "help me")),
            Steps.Answer("did it myself"));

        await h.Spawn("Do the thing.").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SubAgentGeneration second = child.Call(2);
        Assert.Equal("tool", second.LastRole);
        Assert.Equal("Error: Agent depth limit reached. Solve the task yourself.", second.LastContent);
        Assert.Equal(1, h.Runtime.SpawnedCount);
        Assert.Contains("You cannot start sub-agents of your own", child.Call(1).LastContent, StringComparison.Ordinal);
        Assert.Equal("did it myself", h.Snapshot("agent_1").Result);
    }

    [Fact]
    public async Task Spawn_FromAChild_IsAllowedWhenTheDepthLimitIsTwo()
    {
        using var h = new SubAgentHarness(maxDepth: 2);
        // The grandchild is held until the child's second generation, so its answer
        // cannot be folded into the spawn result at the child's first round boundary.
        var grandGate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.SpawnAgent, ("message", "grand task")),
            Steps.Do(_ => { grandGate.TrySetResult(); return Task.CompletedTask; },
                Steps.Call(SkillToolNames.WaitAgent, ("targets", "all"))),
            Steps.Answer("child done"));
        ScriptedAgent grandchild = h.Script("agent_2", Steps.After(grandGate.Task, Steps.Answer("grand done")));

        await h.Spawn("Delegate further.").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SubAgentLaunch launch = h.Launches.Single(l => l.AgentId == "agent_2");
        Assert.Equal("agent_1", launch.ParentId);
        Assert.Equal(2, launch.Depth);

        Assert.StartsWith("agent_2 started", child.Call(2).LastContent, StringComparison.Ordinal);
        Assert.Contains("agent_2 completed in", child.Call(3).LastContent, StringComparison.Ordinal);
        Assert.Contains("grand done", child.Call(3).LastContent, StringComparison.Ordinal);

        // Only the agent at the limit is told it cannot spawn.
        Assert.DoesNotContain("You cannot start sub-agents", child.Call(1).LastContent, StringComparison.Ordinal);
        Assert.Contains("You cannot start sub-agents of your own", grandchild.Call(1).LastContent, StringComparison.Ordinal);
        Assert.Equal("child done", h.Snapshot("agent_1").Result);
    }

    // ---- the open-agent limit ----------------------------------------------------------

    [Fact]
    public async Task Spawn_AtTheLimitWithEveryAgentWorking_IsRefused()
    {
        using var h = new SubAgentHarness(maxThreads: 2);
        h.Script("agent_1", Steps.Block());
        h.Script("agent_2", Steps.Block());

        Assert.True((await h.Spawn("one").Within()).Ok);
        Assert.True((await h.Spawn("two").Within()).Ok);
        SkillToolResult third = await h.Spawn("three").Within();

        Assert.False(third.Ok);
        Assert.StartsWith("Error: agent thread limit reached", third.Content, StringComparison.Ordinal);
        Assert.Contains("2 agents are already open", third.Content, StringComparison.Ordinal);
        Assert.Equal(2, h.Runtime.SpawnedCount);
    }

    [Fact]
    public async Task Spawn_AtTheLimit_EvictsAFinishedAgentWhoseAnswerWasDelivered()
    {
        using var h = new SubAgentHarness(maxThreads: 2);
        h.Script("agent_1", Steps.Answer("one done"));
        h.Script("agent_2", Steps.Block());

        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        SkillToolResult collected = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within();
        Assert.Contains("one done", collected.Content, StringComparison.Ordinal);
        await h.Spawn("two").Within();

        SkillToolResult third = await h.Spawn("three").Within();

        Assert.True(third.Ok, third.Content);
        Assert.StartsWith("agent_3 started", third.Content, StringComparison.Ordinal);
        Assert.Contains("(agent_1, which had finished and already reported, was closed to make room.)",
            third.Content, StringComparison.Ordinal);
        Assert.Equal(SubAgentStatus.Closed, h.Snapshot("agent_1").Status);
        Assert.Equal(SubAgentStatus.Running, h.Snapshot("agent_2").Status);
    }

    [Fact]
    public async Task Spawn_AtTheLimit_NeverEvictsAnAnswerNobodyHasSeen()
    {
        using var h = new SubAgentHarness(maxThreads: 2);
        h.Script("agent_1", Steps.Answer("unseen answer"));
        h.Script("agent_2", Steps.Block());

        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await h.Spawn("two").Within();

        SkillToolResult third = await h.Spawn("three").Within();

        Assert.False(third.Ok);
        Assert.StartsWith("Error: agent thread limit reached", third.Content, StringComparison.Ordinal);
        Assert.Equal(SubAgentStatus.Completed, h.Snapshot("agent_1").Status);
        SubAgentDeliveries pending = h.Runtime.Root.TakePendingDeliveries();
        Assert.Contains("unseen answer", pending.Notification, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Spawn_AtTheLimit_FromASubAgent_NeverEvictsAnAgentItDidNotStart()
    {
        // agent_1 is the root's, finished and already reported to the root. agent_2 (also
        // the root's) then spawns at the limit: closing agent_1 would take an agent from
        // under the root, which still addresses it, and tell agent_2 about an id it
        // cannot even name.
        using var h = new SubAgentHarness(maxThreads: 2, maxDepth: 2);
        h.Script("agent_1", Steps.Answer("one"));
        await h.Spawn("first").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2);
        ScriptedAgent child = h.Script("agent_2",
            Steps.Call(SkillToolNames.SpawnAgent, ("message", "grand task")),
            Steps.Answer("did it myself"));

        await h.Spawn("second").Within();
        await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed);

        Assert.StartsWith("Error: agent thread limit reached", child.Call(2).LastContent, StringComparison.Ordinal);
        Assert.DoesNotContain("agent_1", child.Call(2).LastContent, StringComparison.Ordinal);
        Assert.Equal(SubAgentStatus.Completed, h.Snapshot("agent_1").Status);

        SkillToolResult sent = await h.Root(SkillToolNames.SendInput, ("target", "agent_1"), ("message", "one more")).Within(2);
        Assert.True(sent.Ok, sent.Content);
    }

    // ---- what a child is started with ----------------------------------------------------

    [Fact]
    public async Task FreshChild_GetsTheParentsInstructionsVerbatimThenOneTaskEnvelope_AndTheSameTools()
    {
        var conversation = new List<ChatMessage>
        {
            new() { Role = "system", Content = SubAgentHarness.SystemPrompt, RawOutputTokens = null },
            new() { Role = "developer", Content = SubAgentHarness.DeveloperPrompt },
            new() { Role = "user", Content = "the user's own request" },
            new() { Role = "assistant", Content = "working on it" },
        };
        using var h = new SubAgentHarness(conversation: conversation);
        ScriptedAgent child = h.Script("agent_1", Steps.Answer("ok"));

        await h.Spawn("List every TODO in src/.").Within();
        await child.Entered(1).Within();

        SubAgentGeneration first = child.Call(1);
        Assert.Equal(new[] { "system", "developer", "user" }, first.Roles);
        Assert.Equal(SubAgentHarness.SystemPrompt, first.Contents[0]);
        Assert.Equal(SubAgentHarness.DeveloperPrompt, first.Contents[1]);
        Assert.NotSame(conversation[0], first.Messages[0]);
        Assert.NotSame(conversation[1], first.Messages[1]);

        string envelope = first.Contents[2];
        Assert.StartsWith("You are agent_1, a sub-agent. Another agent started you", envelope, StringComparison.Ordinal);
        Assert.EndsWith("\n\nTask:\nList every TODO in src/.", envelope, StringComparison.Ordinal);
        Assert.DoesNotContain("the user's own request", first.AllText, StringComparison.Ordinal);

        // The KV prefix-sharing contract: the tool block is the parent's, unchanged.
        Assert.NotNull(first.Tools);
        Assert.Equal(h.Tools.Select(t => t.Name), first.Tools!.Select(t => t.Name));
        Assert.Equal(h.Tools.Select(t => t.Description), first.Tools!.Select(t => t.Description));
        for (int i = 0; i < h.Tools.Count; i++)
            Assert.Same(h.Tools[i], first.Tools[i]);

        // A later edit to the child's copy cannot reach the parent's conversation.
        first.Messages[0].Content = "mutated";
        Assert.Equal(SubAgentHarness.SystemPrompt, conversation[0].Content);
    }

    [Fact]
    public async Task ForkedChild_GetsTheConversationUpToTheSpawningTurn_WithEveryCallOfThatTurnAnswered()
    {
        ToolCall before = Steps.Tool("lookup", ("query", "a"));
        ToolCall spawn = Steps.Tool(SkillToolNames.SpawnAgent, ("message", "Fix the parser."), ("fork_context", true));
        ToolCall after = Steps.Tool("lookup", ("query", "c"));
        var assistant = new ChatMessage
        {
            Role = "assistant",
            Content = "Splitting the work.",
            Thinking = "two parts",
            ToolCalls = new List<ToolCall> { before, spawn, after },
            RawOutputTokens = new List<int> { 11, 12, 13 },
            RawGenerationSuffix = "<|end|>",
        };
        var conversation = new List<ChatMessage>
        {
            new() { Role = "system", Content = SubAgentHarness.SystemPrompt },
            new() { Role = "user", Content = "the user's own request" },
            assistant,
            new() { Role = "tool", ToolCallId = before.Id, Content = "lookup result A" },
        };
        using var h = new SubAgentHarness(conversation: conversation);
        ScriptedAgent child = h.Script("agent_1", Steps.Answer("fixed"));

        SkillToolResult spawned = await h.Root(spawn).Within();
        Assert.True(spawned.Ok, spawned.Content);
        Assert.StartsWith("agent_1 started with a copy of this conversation.", spawned.Content, StringComparison.Ordinal);
        Assert.True(h.Launches.Single().ForkContext);
        await child.Entered(1).Within();

        // The copied conversation, every call of the spawning turn answered, and then the
        // task as a USER turn: a model reads a tool result as the end of a step it took,
        // and a fork whose task arrived there ended its turn on its first token.
        SubAgentGeneration first = child.Call(1);
        Assert.Equal(new[] { "system", "user", "assistant", "tool", "tool", "tool", "user" }, first.Roles);
        Assert.Equal("the user's own request", first.Contents[1]);

        ChatMessage copied = first.Messages[2];
        Assert.NotSame(assistant, copied);
        Assert.Equal("Splitting the work.", copied.Content);
        Assert.Equal("two parts", copied.Thinking);
        Assert.Equal(new[] { 11, 12, 13 }, copied.RawOutputTokens);
        Assert.NotSame(assistant.RawOutputTokens, copied.RawOutputTokens);
        Assert.Equal("<|end|>", copied.RawGenerationSuffix);
        Assert.Equal(new[] { before, spawn, after }, copied.ToolCalls);
        Assert.NotSame(assistant.ToolCalls, copied.ToolCalls);

        Assert.Equal(before.Id, first.Messages[3].ToolCallId);
        Assert.Equal("lookup result A", first.Contents[3]);
        Assert.Equal(spawn.Id, first.Messages[4].ToolCallId);
        Assert.Equal("agent_1 started with a copy of this conversation.", first.Contents[4]);
        Assert.Equal(after.Id, first.Messages[5].ToolCallId);
        Assert.Equal("(The result of this call is not shown to the sub-agent.)", first.Contents[5]);
        Assert.StartsWith("You are agent_1, a sub-agent. You were forked from the conversation above", first.Contents[6], StringComparison.Ordinal);
        Assert.EndsWith("Task:\nFix the parser.", first.Contents[6], StringComparison.Ordinal);
    }

    [Fact]
    public async Task ForkedChild_OnAFamilyWithoutToolMessages_GetsItsResultsAsUserTurns()
    {
        ToolCall spawn = Steps.Tool(SkillToolNames.SpawnAgent, ("message", "Fix it."), ("fork_context", "true"));
        var conversation = new List<ChatMessage>
        {
            new() { Role = "system", Content = SubAgentHarness.SystemPrompt },
            new() { Role = "user", Content = "request" },
            new() { Role = "assistant", ToolCalls = new List<ToolCall> { spawn } },
        };
        using var h = new SubAgentHarness(
            conversation: conversation,
            loopOptions: new SkillAgentLoopOptions { ToolResultsAreRendered = false });
        ScriptedAgent child = h.Script("agent_1", Steps.Answer("fixed"));

        await h.Root(spawn).Within();
        await child.Entered(1).Within();

        SubAgentGeneration first = child.Call(1);
        Assert.Equal(new[] { "system", "user", "assistant", "user", "user" }, first.Roles);
        Assert.Equal("Result of your spawn_agent call:\n\nagent_1 started with a copy of this conversation.", first.Contents[3]);
        Assert.StartsWith("You are agent_1, a sub-agent. You were forked", first.LastContent, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ForkedChild_WhoseSpawningTurnIsNotInTheConversation_StartsFresh()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Answer("ok"));

        SkillToolResult spawned = await h.Root(SkillToolNames.SpawnAgent, ("message", "Do it."), ("fork_context", true)).Within();
        await child.Entered(1).Within();

        Assert.Equal(new[] { "system", "developer", "user" }, child.Call(1).Roles);
        Assert.StartsWith("You are agent_1, a sub-agent. Another agent started you", child.Call(1).LastContent, StringComparison.Ordinal);

        // Neither the model nor the host may be told it was forked: the model would think
        // the agent has context it does not, and the host would run a fresh agent in the
        // parent's cache scope (SubAgentLaunch.ForkContext: "started with a copy").
        Assert.DoesNotContain("with a copy of this conversation", spawned.Content, StringComparison.Ordinal);
        Assert.False(h.Launches.Single().ForkContext);
    }

    // ---- wait_agent ------------------------------------------------------------------------

    [Fact]
    public async Task Wait_ReturnsWhenTheFirstAgentFinishes_WithOnlyThatAnswer()
    {
        using var h = new SubAgentHarness();
        var gate1 = Steps.Gate();
        var gate2 = Steps.Gate();
        ScriptedAgent one = h.Script("agent_1", Steps.After(gate1.Task, Steps.Answer("answer one")));
        ScriptedAgent two = h.Script("agent_2", Steps.After(gate2.Task, Steps.Answer("answer two")));
        await h.Spawn("one").Within();
        await h.Spawn("two").Within();
        await one.Entered(1).Within();
        await two.Entered(1).Within();

        Task<SkillToolResult> waiting = h.Root(SkillToolNames.WaitAgent, ("targets", "all"));
        Assert.False(waiting.IsCompleted, "wait_agent returned while both agents were still working");

        gate2.SetResult();
        SkillToolResult first = await waiting.Within();

        Assert.True(first.Ok, first.Content);
        Assert.StartsWith("agent_2 completed in ", first.Content, StringComparison.Ordinal);
        Assert.Contains("Its final answer:\nanswer two", first.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("answer one", first.Content, StringComparison.Ordinal);
        Assert.Contains("\n\nStill running: agent_1 (", first.Content, StringComparison.Ordinal);

        gate1.SetResult();
        SkillToolResult second = await h.Root(SkillToolNames.WaitAgent, ("targets", "all")).Within();
        Assert.Contains("answer one", second.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("answer two", second.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("Still running", second.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Wait_ForAnAlreadyFinishedAgent_ReturnsAtOnce_AndOnlyOnce()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("the answer"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SkillToolResult first = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2);
        Assert.True(first.Ok, first.Content);
        Assert.Contains("Its final answer:\nthe answer", first.Content, StringComparison.Ordinal);
        Assert.True(h.Snapshot("agent_1").ResultDelivered);

        SkillToolResult again = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2);
        Assert.True(again.Ok, again.Content);
        Assert.Equal(
            "Nothing to wait for: none of those agents is working, and every answer was already delivered to you (agent_1: completed).",
            again.Content);

        SkillToolResult any = await h.Root(SkillToolNames.WaitAgent, ("targets", "all")).Within(2);
        Assert.DoesNotContain("the answer", any.Content, StringComparison.Ordinal);
        Assert.StartsWith("Nothing to wait for", any.Content, StringComparison.Ordinal);

        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task Wait_WithNoAgents_SaysSo()
    {
        using var h = new SubAgentHarness();
        SkillToolResult result = await h.Root(SkillToolNames.WaitAgent, ("targets", "all")).Within(2);
        Assert.True(result.Ok);
        Assert.Equal("You have no sub-agents to wait for. Start one with spawn_agent.", result.Content);
    }

    public static IEnumerable<object?[]> TargetForms()
    {
        yield return new object?[] { "all", true };
        yield return new object?[] { "ALL", true };
        yield return new object?[] { "any", true };
        yield return new object?[] { "*", true };
        yield return new object?[] { null, true };  // omitted
        yield return new object?[] { "", true };
        yield return new object?[] { "agent_1, agent_2", true };
        yield return new object?[] { "agent_1 agent_2", true };
        yield return new object?[] { "[\"agent_1\", \"agent_2\"]", true };
        yield return new object?[] { "1", false };  // numeric shorthand for agent_1
        yield return new object?[] { " agent_1 ", false };
    }

    [Theory]
    [MemberData(nameof(TargetForms))]
    public async Task Wait_AcceptsEveryTargetShape(string? targets, bool both)
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("answer one"));
        h.Script("agent_2", Steps.Answer("answer two"));
        await h.Spawn("one").Within();
        await h.Spawn("two").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed);

        SkillToolResult result = targets == null
            ? await h.Root(SkillToolNames.WaitAgent).Within(2)
            : await h.Root(SkillToolNames.WaitAgent, ("targets", targets)).Within(2);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("answer one", result.Content, StringComparison.Ordinal);
        Assert.Equal(both, result.Content.Contains("answer two", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Wait_AcceptsAJsonArrayAndAJsonNumber()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("answer one"));
        h.Script("agent_2", Steps.Answer("answer two"));
        await h.Spawn("one").Within();
        await h.Spawn("two").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed);

        SkillToolResult byNumber = await h.Root(SkillToolNames.WaitAgent,
            ("targets", JsonDocument.Parse("2").RootElement.Clone())).Within(2);
        Assert.Contains("answer two", byNumber.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("answer one", byNumber.Content, StringComparison.Ordinal);

        SkillToolResult byArray = await h.Root(SkillToolNames.WaitAgent,
            ("targets", JsonDocument.Parse("[\"agent_1\",\"agent_2\"]").RootElement.Clone())).Within(2);
        Assert.Contains("answer one", byArray.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("answer two", byArray.Content, StringComparison.Ordinal); // already delivered
    }

    [Fact]
    public async Task Wait_ForAnUnknownId_NamesTheCallersAgents()
    {
        using var h = new SubAgentHarness();
        SkillToolResult none = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_9")).Within(2);
        Assert.False(none.Ok);
        Assert.Equal("Error: agent with id agent_9 not found. You have no sub-agents.", none.Content);

        h.Script("agent_1", Steps.Block());
        h.Script("agent_2", Steps.Block());
        await h.Spawn("one").Within();
        await h.Spawn("two").Within();

        SkillToolResult unknown = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1, agent_9")).Within(2);
        Assert.False(unknown.Ok);
        Assert.Equal("Error: agent with id agent_9 not found. Your agents are: agent_1, agent_2.", unknown.Content);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public async Task Wait_ANonPositiveTimeout_IsAnError(int timeout)
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Block());
        await h.Spawn("one").Within();

        SkillToolResult result = await h.Root(SkillToolNames.WaitAgent, ("targets", "all"), ("timeout_ms", timeout)).Within(2);

        Assert.False(result.Ok);
        Assert.Equal("Error: timeout_ms must be greater than zero", result.Content);
    }

    [Theory]
    [InlineData(5L, "Requested timeout of 5 ms was clamped to 10000 ms.")]
    [InlineData(7_200_000L, "Requested timeout of 7200000 ms was clamped to 3600000 ms.")]
    public async Task Wait_AnOutOfRangeTimeout_IsClampedAndSaysSo(long timeout, string note)
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("the answer"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SkillToolResult result = await h.Root(SkillToolNames.WaitAgent, ("targets", "all"), ("timeout_ms", timeout)).Within(2);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("the answer", result.Content, StringComparison.Ordinal);
        Assert.EndsWith("\n" + note, result.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// SLOW (about 10 s): the one test that lets a wait run out. The shortest honoured
    /// timeout is <see cref="SubAgentTools.MinWaitMs"/>, so a 1 ms request waits ten seconds.
    /// </summary>
    [Fact]
    public async Task Wait_TimesOut_AfterTheClampedMinimum_TakesTenSeconds()
    {
        using var h = new SubAgentHarness();
        var gate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1", Steps.After(gate.Task, Steps.Answer("late")));
        await h.Spawn("one").Within();
        await child.Entered(1).Within();

        var stopwatch = Stopwatch.StartNew();
        SkillToolResult result = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1"), ("timeout_ms", 1)).Within(30);
        stopwatch.Stop();

        Assert.True(result.Ok, result.Content);
        Assert.True(stopwatch.Elapsed >= TimeSpan.FromSeconds(9.5), $"returned after {stopwatch.Elapsed}");
        Assert.StartsWith("No agent finished within ", result.Content, StringComparison.Ordinal);
        Assert.Contains(". Still running: agent_1 (", result.Content, StringComparison.Ordinal);
        Assert.EndsWith("\nRequested timeout of 1 ms was clamped to 10000 ms.", result.Content, StringComparison.Ordinal);
        Assert.Equal(SubAgentStatus.Running, h.Snapshot("agent_1").Status);

        gate.SetResult();
        SkillToolResult late = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within();
        Assert.Contains("late", late.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Wait_StreamsTheChildrensActivityToTheTap()
    {
        using var h = new SubAgentHarness();
        var gate1 = Steps.Gate();
        var gate2 = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1",
            Steps.After(gate1.Task, Steps.Call(SkillToolNames.ListAgents)),
            Steps.After(gate2.Task, Steps.Answer("tapped")));
        await h.Spawn("one").Within();
        await child.Entered(1).Within();

        var lines = new ConcurrentQueue<string>();
        Task<SkillToolResult> waiting = h.Root(
            Steps.Tool(SkillToolNames.WaitAgent, ("targets", "agent_1")), line => lines.Enqueue(line));
        Assert.False(waiting.IsCompleted);

        gate1.SetResult();
        await child.Entered(2).Within();
        Assert.Contains("agent_1: round 1: list_agents", lines);

        gate2.SetResult();
        SkillToolResult result = await waiting.Within();
        Assert.Contains("tapped", result.Content, StringComparison.Ordinal);
        Assert.Contains(lines, l => l.StartsWith("agent_1: finished (", StringComparison.Ordinal));

        // The tap is only listened to while the wait lasts.
        int count = lines.Count;
        h.Script("agent_2", Steps.Answer("unobserved"));
        await h.Spawn("two").Within();
        await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed);
        Assert.Equal(count, lines.Count);
    }

    [Fact]
    public async Task Wait_CancelledByTheCaller_Throws_AndLeavesTheChildRunning()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Block());
        await h.Spawn("one").Within();
        await child.Entered(1).Within();

        using var cts = new CancellationTokenSource();
        Task<SkillToolResult> waiting = h.Root(Steps.Tool(SkillToolNames.WaitAgent, ("targets", "all")), null, cts.Token);
        Assert.False(waiting.IsCompleted);
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting.Within());
        Assert.Equal(SubAgentStatus.Running, h.Snapshot("agent_1").Status);
        Assert.False(child.Cancelled.IsCompleted);
    }

    // ---- exactly-once delivery ---------------------------------------------------------------

    [Fact]
    public async Task PendingDeliveries_HandAFinishedAnswerOverExactlyOnce()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("the answer"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        Assert.True(h.Runtime.Root.HasOutstandingWork);
        SubAgentDeliveries first = h.Runtime.Root.TakePendingDeliveries();

        Assert.False(first.IsEmpty);
        Assert.Empty(first.Messages);
        Assert.NotNull(first.Notification);
        Assert.StartsWith("<subagent_notification>\nagent_1 completed in ", first.Notification, StringComparison.Ordinal);
        Assert.EndsWith("Its final answer:\nthe answer\n</subagent_notification>", first.Notification, StringComparison.Ordinal);

        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
        Assert.False(h.Runtime.Root.HasOutstandingWork);

        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2);
        Assert.DoesNotContain("the answer", waited.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ALongAnswer_IsCutAndSaysHowToGetTheRest()
    {
        using var h = new SubAgentHarness();
        string answer = new string('a', SubAgentRuntime.MaxResultChars) + new string('b', 1000);
        h.Script("agent_1", Steps.Answer(answer));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        string content = (await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2)).Content;

        Assert.Contains("Its final answer:\n" + new string('a', SubAgentRuntime.MaxResultChars) + "\n[... truncated: the full answer is 9000 characters. "
            + "Ask agent_1 for the part you need with send_input.]", content, StringComparison.Ordinal);
        Assert.DoesNotContain("ab", content, StringComparison.Ordinal);
        Assert.Equal(answer, h.Snapshot("agent_1").Result); // kept whole; only the delivery is cut
    }

    [Fact]
    public async Task AChildThatAnswersOnlyInItsReasoning_IsReportedAsSuch()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", _ => Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "  ", Thinking = "The total is 42." })));
        h.Script("agent_2", Steps.Answer(""));
        await h.Spawn("one").Within();
        await h.Spawn("two").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed);

        Assert.Equal("(The sub-agent wrote no final answer. The end of its reasoning was:)\nThe total is 42.", h.Snapshot("agent_1").Result);
        Assert.Equal("(The sub-agent ended without writing a final answer.)", h.Snapshot("agent_2").Result);
    }

    [Fact]
    public async Task PendingDeliveries_AfterWaitDeliveredTheAnswer_AreEmpty()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("the answer"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "all")).Within(2);
        Assert.Contains("the answer", waited.Content, StringComparison.Ordinal);

        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task PendingDeliveries_CarryEveryFinishedAnswerInOneNotification()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("answer one"));
        h.Script("agent_2", Steps.Answer("answer two"));
        await h.Spawn("one").Within();
        await h.Spawn("two").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed);

        string notification = h.Runtime.Root.TakePendingDeliveries().Notification!;
        Assert.Equal(1, notification.Occurrences("<subagent_notification>"));
        Assert.True(notification.IndexOf("answer one", StringComparison.Ordinal)
                    < notification.IndexOf("answer two", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AppendDeliveries_FoldsTheNotificationIntoTheLastResultOfTheRound()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("the answer"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        var working = new List<ChatMessage>
        {
            new() { Role = "system", Content = "s" },
            new() { Role = "user", Content = "u" },
            new() { Role = "assistant", ToolCalls = new List<ToolCall> { new() { Name = "a" }, new() { Name = "b" } } },
            new() { Role = "tool", Content = "first result" },
            new() { Role = "tool", Content = "second result  \n" },
        };

        Assert.True(SubAgentConversation.AppendDeliveries(working, h.Runtime.Root));

        Assert.Equal(5, working.Count);
        Assert.Equal("first result", working[3].Content);
        Assert.StartsWith("second result\n\n<subagent_notification>\nagent_1 completed", working[4].Content, StringComparison.Ordinal);
        Assert.EndsWith("the answer\n</subagent_notification>", working[4].Content, StringComparison.Ordinal);

        Assert.False(SubAgentConversation.AppendDeliveries(working, h.Runtime.Root));
        Assert.Equal(1, working[4].Content!.Occurrences("the answer"));
    }

    [Fact]
    public async Task AppendDeliveries_AddsAUserTurnWhenTheConversationEndsOnTheAssistant()
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
        Assert.Equal("done", working[1].Content);
        Assert.Equal("user", working[2].Role);
        Assert.StartsWith("<subagent_notification>", working[2].Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AppendDeliveries_WithNoScopeOrNothingPending_ChangesNothing()
    {
        var working = new List<ChatMessage> { new() { Role = "tool", Content = "r" } };
        Assert.False(SubAgentConversation.AppendDeliveries(working, null));

        using var h = new SubAgentHarness();
        Assert.False(SubAgentConversation.AppendDeliveries(working, h.Runtime.Root));
        Assert.Single(working);
        Assert.Equal("r", working[0].Content);
    }

    // ---- send_input --------------------------------------------------------------------------

    [Fact]
    public async Task SendInput_ToAWorkingChild_IsDeliveredAtItsNextRoundBoundary()
    {
        using var h = new SubAgentHarness();
        var gate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1",
            Steps.After(gate.Task, Steps.Call(SkillToolNames.ListAgents)),
            Steps.Answer("saw it"));
        await h.Spawn("one").Within();
        await child.Entered(1).Within();

        SkillToolResult sent = await h.Root(SkillToolNames.SendInput, ("target", "agent_1"), ("message", "Also check the tests.")).Within(2);
        Assert.True(sent.Ok, sent.Content);
        Assert.Equal("agent_1 is still working; it will read your message before its next step.", sent.Content);

        gate.SetResult();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SubAgentGeneration second = child.Call(2);
        Assert.Equal("tool", second.Roles[^2]);
        Assert.Equal("user", second.LastRole);
        Assert.Equal(
            "Message from the agent that started you:\n\nAlso check the tests.\n\nWhen you are done, reply with your final answer, as before.",
            second.LastContent);
        Assert.Equal("saw it", h.Snapshot("agent_1").Result);
    }

    [Fact]
    public async Task SendInput_ArrivingWhileTheChildWritesItsAnswer_MakesItCarryOn()
    {
        using var h = new SubAgentHarness();
        var gate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1",
            Steps.After(gate.Task, Steps.Answer("first answer")),
            Steps.Answer("second answer"));
        await h.Spawn("one").Within();
        await child.Entered(1).Within();

        await h.Root(SkillToolNames.SendInput, ("target", "agent_1"), ("message", "Include the totals.")).Within(2);
        gate.SetResult();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SubAgentGeneration second = child.Call(2);
        Assert.Equal("assistant", second.Roles[^2]);
        Assert.Equal("first answer", second.Contents[^2]);
        Assert.Contains("Include the totals.", second.LastContent, StringComparison.Ordinal);

        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2);
        Assert.Contains("second answer", waited.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("first answer", waited.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SendInput_ToAFinishedChild_RestartsItWithItsHistory_AndHandsOverAnUnseenAnswer()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Answer("first"), Steps.Answer("second"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        SkillToolResult sent = await h.Root(SkillToolNames.SendInput, ("target", "agent_1"), ("message", "Now the other half.")).Within(2);

        Assert.True(sent.Ok, sent.Content);
        Assert.StartsWith("agent_1 completed in ", sent.Content, StringComparison.Ordinal);
        Assert.Contains("Its final answer:\nfirst\n\n", sent.Content, StringComparison.Ordinal);
        Assert.EndsWith("agent_1 is working on your message, with its earlier context kept.", sent.Content, StringComparison.Ordinal);

        await child.Entered(2).Within();
        SubAgentGeneration second = child.Call(2);
        Assert.Equal(new[] { "system", "developer", "user", "assistant", "user" }, second.Roles);
        Assert.Equal("first", second.Contents[3]);
        Assert.Contains("Now the other half.", second.LastContent, StringComparison.Ordinal);

        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within();
        Assert.Contains("Its final answer:\nsecond", waited.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("first", waited.Content, StringComparison.Ordinal);
        Assert.Equal(2, h.Snapshot("agent_1").Turns);
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task SendInput_ToAFinishedChildWhoseAnswerWasSeen_OnlyRestartsIt()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("first"), Steps.Answer("second"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2);

        SkillToolResult sent = await h.Root(SkillToolNames.SendInput, ("target", "1"), ("message", "More.")).Within(2);
        Assert.Equal("agent_1 is working on your message, with its earlier context kept.", sent.Content);
    }

    [Fact]
    public async Task SendInput_ToAClosedChild_IsAnError()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Block());
        await h.Spawn("one").Within();
        await h.Root(SkillToolNames.CloseAgent, ("target", "agent_1")).Within(2);

        SkillToolResult sent = await h.Root(SkillToolNames.SendInput, ("target", "agent_1"), ("message", "hello")).Within(2);
        Assert.False(sent.Ok);
        Assert.Equal("Error: agent with id agent_1 is closed", sent.Content);
    }

    [Fact]
    public async Task SendInput_MissingArguments_AreErrors()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Block());
        await h.Spawn("one").Within();

        SkillToolResult noTarget = await h.Root(SkillToolNames.SendInput, ("message", "hello")).Within(2);
        Assert.Equal("Error: send_input needs 'target': the id of the agent to message.", noTarget.Content);

        SkillToolResult noMessage = await h.Root(SkillToolNames.SendInput, ("target", "agent_1"), ("message", "  ")).Within(2);
        Assert.Equal("Error: Empty message can't be sent to an agent", noMessage.Content);

        SkillToolResult unknown = await h.Root(SkillToolNames.SendInput, ("target", "agent_7"), ("message", "hi")).Within(2);
        Assert.Equal("Error: agent with id agent_7 not found. Your agents are: agent_1.", unknown.Content);
    }

    /// <summary>
    /// A message sent to a child while that child is at the end of its own turn, waiting
    /// for ITS sub-agents (the end-of-turn guard), must still reach it. The guard hands
    /// the grandchildren's results over; it must not swallow the parent's message on the
    /// way.
    /// </summary>
    [Fact]
    public async Task SendInput_QueuedWhileTheChildWaitsForItsOwnSubAgentsAtTheEndOfItsTurn_IsNotLost()
    {
        using var h = new SubAgentHarness(maxDepth: 2);
        var grandGate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.SpawnAgent, ("message", "grand task")),
            Steps.Answer("draft without the grandchild"),
            Steps.Answer("final"));
        h.Script("agent_2", Steps.After(grandGate.Task, Steps.Answer("grand done")));

        await h.Spawn("one").Within();
        await child.Entered(2).Within();   // agent_1 has answered; its end-of-turn guard now waits for agent_2

        SkillToolResult sent = await h.Root(SkillToolNames.SendInput, ("target", "agent_1"), ("message", "Use metric units.")).Within(2);
        Assert.Equal("agent_1 is still working; it will read your message before its next step.", sent.Content);

        grandGate.SetResult();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        Assert.Contains(child.Calls.Skip(2), g => g.Contents.Any(c => c.Contains("Use metric units.", StringComparison.Ordinal)));
        Assert.Contains(child.Calls, g => g.LastContent.StartsWith("Before you finish:", StringComparison.Ordinal)
                                          && g.LastContent.Contains("grand done", StringComparison.Ordinal));
    }

    // ---- close_agent --------------------------------------------------------------------------

    [Fact]
    public async Task Close_StopsAWorkingChild_AndIsIdempotent()
    {
        using var h = new SubAgentHarness();
        ScriptedAgent child = h.Script("agent_1", Steps.Block());
        await h.Spawn("one").Within();
        await child.Entered(1).Within();

        SkillToolResult closed = await h.Root(SkillToolNames.CloseAgent, ("target", "agent_1")).Within(2);
        Assert.True(closed.Ok, closed.Content);
        Assert.Equal("agent_1 closed; it was running.", closed.Content);
        await child.Cancelled.Within();
        Assert.Equal(SubAgentStatus.Closed, h.Snapshot("agent_1").Status);
        Assert.False(h.Runtime.Root.HasOutstandingWork);

        SkillToolResult again = await h.Root(SkillToolNames.CloseAgent, ("target", "agent_1")).Within(2);
        Assert.True(again.Ok);
        Assert.Equal("agent_1 was already closed.", again.Content);

        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "all")).Within(2);
        Assert.Equal("None of your sub-agents is open, so there is nothing to wait for.", waited.Content);
    }

    [Fact]
    public async Task Close_CascadesToTheChildsOwnSubAgents()
    {
        using var h = new SubAgentHarness(maxDepth: 2);
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.SpawnAgent, ("message", "grand task")),
            Steps.Block());
        ScriptedAgent grandchild = h.Script("agent_2", Steps.Block());
        await h.Spawn("one").Within();
        await grandchild.Entered(1).Within();
        await child.Entered(2).Within();

        SkillToolResult closed = await h.Root(SkillToolNames.CloseAgent, ("target", "agent_1")).Within(2);

        Assert.Equal("agent_1 closed; it was running. 1 agent it had started was closed too.", closed.Content);
        await child.Cancelled.Within();
        await grandchild.Cancelled.Within();
        Assert.Equal(SubAgentStatus.Closed, h.Snapshot("agent_1").Status);
        Assert.Equal(SubAgentStatus.Closed, h.Snapshot("agent_2").Status);
    }

    [Fact]
    public async Task Close_FreesTheSlot()
    {
        using var h = new SubAgentHarness(maxThreads: 1);
        h.Script("agent_1", Steps.Block());
        h.Script("agent_2", Steps.Answer("second"));
        await h.Spawn("one").Within();

        Assert.False((await h.Spawn("two").Within()).Ok);
        await h.Root(SkillToolNames.CloseAgent, ("target", "agent_1")).Within(2);

        SkillToolResult spawned = await h.Spawn("two").Within();
        Assert.True(spawned.Ok, spawned.Content);
        Assert.StartsWith("agent_2 started", spawned.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Close_OfAFinishedChildWhoseAnswerWasNeverDelivered_HandsTheAnswerOver()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Answer("the unseen answer"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed).Within();

        // Closing means "stop working", not "discard what was already produced".
        SkillToolResult closed = await h.Root(SkillToolNames.CloseAgent, ("target", "agent_1")).Within(2);
        Assert.True(closed.Ok, closed.Content);
        Assert.StartsWith("agent_1 completed in ", closed.Content, StringComparison.Ordinal);
        Assert.Contains("the unseen answer", closed.Content, StringComparison.Ordinal);
        Assert.EndsWith("agent_1 closed; it was completed.", closed.Content, StringComparison.Ordinal);
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
    }

    [Fact]
    public async Task Close_WithoutATarget_IsAnError()
    {
        using var h = new SubAgentHarness();
        SkillToolResult result = await h.Root(SkillToolNames.CloseAgent).Within(2);
        Assert.False(result.Ok);
        Assert.Equal("Error: close_agent needs 'target': the id of the agent to close.", result.Content);
    }

    // ---- list_agents -----------------------------------------------------------------------------

    [Fact]
    public async Task List_ShowsOnlyTheCallersOwnChildren()
    {
        using var h = new SubAgentHarness(maxDepth: 2);
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call(SkillToolNames.SpawnAgent, ("message", "grand task")),
            Steps.Block());
        ScriptedAgent grandchild = h.Script("agent_2", Steps.Block());
        await h.Spawn("Investigate the crash.").Within();
        await grandchild.Entered(1).Within();
        await child.Entered(2).Within();

        SkillToolResult listed = await h.Root(SkillToolNames.ListAgents).Within(2);

        Assert.True(listed.Ok);
        Assert.StartsWith("1 sub-agent:\nagent_1: running for ", listed.Content, StringComparison.Ordinal);
        Assert.EndsWith(" - task: Investigate the crash.", listed.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("agent_2", listed.Content, StringComparison.Ordinal);

        // Nor can the root reach the grandchild by id.
        SkillToolResult reach = await h.Root(SkillToolNames.CloseAgent, ("target", "agent_2")).Within(2);
        Assert.Equal("Error: agent with id agent_2 not found. Your agents are: agent_1.", reach.Content);
        SkillToolResult reachByNumber = await h.Root(SkillToolNames.WaitAgent, ("targets", "2")).Within(2);
        Assert.Equal("Error: agent with id 2 not found. Your agents are: agent_1.", reachByNumber.Content);
    }

    [Fact]
    public async Task List_SaysWhetherEachAnswerWasCollected()
    {
        using var h = new SubAgentHarness();
        Assert.Equal("You have no sub-agents. Start one with spawn_agent.", (await h.Root(SkillToolNames.ListAgents).Within(2)).Content);

        h.Script("agent_1", Steps.Answer("one"));
        h.Script("agent_2", Steps.Answer("two"));
        await h.Spawn("first task").Within();
        await h.Spawn("second task").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed);
        await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2);

        string listed = (await h.Root(SkillToolNames.ListAgents).Within(2)).Content;

        Assert.Equal(
            "2 sub-agents:\n"
            + "agent_1: completed (answer already delivered) - task: first task\n"
            + "agent_2: completed (answer not yet collected: call wait_agent) - task: second task",
            listed);
    }

    // ---- failures ------------------------------------------------------------------------------

    [Fact]
    public async Task AChildWhoseGenerationFails_IsFailed_AndItsErrorIsDeliveredOnce()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Throw("backend exploded"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Errored);

        string listed = (await h.Root(SkillToolNames.ListAgents).Within(2)).Content;
        Assert.Contains("agent_1: failed (answer not yet collected", listed, StringComparison.Ordinal);

        SubAgentDeliveries first = h.Runtime.Root.TakePendingDeliveries();
        Assert.Contains("agent_1 failed after ", first.Notification, StringComparison.Ordinal);
        Assert.Contains(": backend exploded\nIf you still need its result, give it the task again with send_input or do the task yourself.",
            first.Notification, StringComparison.Ordinal);
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);

        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within(2);
        Assert.DoesNotContain("backend exploded", waited.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AFailedChild_CanBeGivenTheTaskAgain()
    {
        using var h = new SubAgentHarness();
        h.Script("agent_1", Steps.Throw("transient"), Steps.Answer("worked the second time"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Errored);

        SkillToolResult sent = await h.Root(SkillToolNames.SendInput, ("target", "agent_1"), ("message", "Try again.")).Within(2);
        Assert.StartsWith("agent_1 failed after ", sent.Content, StringComparison.Ordinal);

        SkillToolResult waited = await h.Root(SkillToolNames.WaitAgent, ("targets", "agent_1")).Within();
        Assert.Contains("worked the second time", waited.Content, StringComparison.Ordinal);
    }

    // ---- disposal and the turn token --------------------------------------------------------------

    [Fact]
    public async Task Dispose_StopsEveryWorkingChildPromptly_AndRefusesNewSpawns()
    {
        var h = new SubAgentHarness();
        ScriptedAgent one = h.Script("agent_1", Steps.Block());
        ScriptedAgent two = h.Script("agent_2", Steps.Block());
        await h.Spawn("one").Within();
        await h.Spawn("two").Within();
        await one.Entered(1).Within();
        await two.Entered(1).Within();

        var stopwatch = Stopwatch.StartNew();
        h.Dispose();
        stopwatch.Stop();

        Assert.True(stopwatch.Elapsed < TimeSpan.FromSeconds(3), $"Dispose took {stopwatch.Elapsed}");
        Assert.True(one.Cancelled.IsCompleted);
        Assert.True(two.Cancelled.IsCompleted);
        Assert.All(h.Runtime.Snapshot(), s => Assert.Equal(SubAgentStatus.Closed, s.Status));

        SkillToolResult late = await h.Spawn("three").Within(2);
        Assert.False(late.Ok);
        Assert.Equal("Error: this turn is ending, so no new agent can start.", late.Content);

        h.Dispose(); // idempotent
    }

    [Fact]
    public async Task CancellingTheTurn_StopsTheChildren_AndEndsAWait()
    {
        using var turn = new CancellationTokenSource();
        using var h = new SubAgentHarness(turnToken: turn.Token);
        ScriptedAgent child = h.Script("agent_1", Steps.Block());
        await h.Spawn("one").Within();
        await child.Entered(1).Within();

        Task<SkillToolResult> waiting = h.Root(SkillToolNames.WaitAgent, ("targets", "all"));
        turn.Cancel();

        SkillToolResult result = await waiting.Within();
        Assert.False(result.Ok);
        Assert.Equal("Error: this turn is ending; the agents were stopped.", result.Content);
        await child.Cancelled.Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Closed);
    }

    // ---- concurrency ------------------------------------------------------------------------------

    [Fact]
    public async Task Children_GenerateConcurrently()
    {
        using var h = new SubAgentHarness();
        var bothInside = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int inside = 0;
        SubAgentStep rendezvous = Steps.Do(
            async generation =>
            {
                if (Interlocked.Increment(ref inside) == 2)
                    bothInside.TrySetResult();
                // Serialized children would leave the first one here alone until this
                // throws, failing it.
                await bothInside.Task.WaitAsync(TimeSpan.FromSeconds(10), generation.Token).ConfigureAwait(false);
            },
            Steps.Answer("met"));
        h.Script("agent_1", rendezvous);
        h.Script("agent_2", rendezvous);

        await h.Spawn("one").Within();
        await h.Spawn("two").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed, timeoutSeconds: 15);
        await h.WaitForStatusAsync("agent_2", SubAgentStatus.Completed, timeoutSeconds: 15);

        Assert.Equal("met", h.Snapshot("agent_1").Result);
        Assert.Equal("met", h.Snapshot("agent_2").Result);
    }

    [Fact]
    public async Task AChildsCallToTheCallersOwnTool_IsRefusedAndItCarriesOn()
    {
        using var h = new SubAgentHarness(loopOptions: new SkillAgentLoopOptions
        {
            ClientTools = new[] { new ToolFunction { Name = "get_weather" } },
        });
        ScriptedAgent child = h.Script("agent_1",
            Steps.Call("get_weather", ("city", "Paris")),
            Steps.Answer("no weather available"));
        await h.Spawn("one").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        Assert.StartsWith("Error: 'get_weather' belongs to the application the user is working in",
            child.Call(2).LastContent, StringComparison.Ordinal);
        Assert.Equal("no weather available", h.Snapshot("agent_1").Result);
    }
}
