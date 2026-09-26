using System.Collections.Concurrent;
using System.Text.Json;
using TensorSharp.AgentHost.Agents;

namespace InferenceWeb.Tests;

/// <summary>Exercises orchestration with deterministic generators, without a model or device.</summary>
public sealed class MultiAgentTests
{
    private static ToolCall Call(string name, params (string Key, object Value)[] arguments) => new()
    {
        Id = Guid.NewGuid().ToString("N"),
        Name = name,
        Arguments = arguments.ToDictionary(p => p.Key, p => p.Value),
    };

    private static ToolCall Spawn(string name, string role = "explorer", string? task = null) =>
        Call("spawn_agent", ("task_name", name), ("agent_type", role), ("task", task ?? "Inspect " + name));

    private static Task<SkillTurnOutput> Answer(string content) =>
        Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = content }));

    private static Task<SkillTurnOutput> Calls(params ToolCall[] calls) =>
        Task.FromResult(new SkillTurnOutput(new ParsedOutput { ToolCalls = calls.ToList() }));

    private static string Id(SkillToolResult result)
    {
        Assert.True(result.Ok, result.Content);
        return JsonDocument.Parse(result.Content).RootElement.GetProperty("agent_id").GetString()!;
    }

    private static JsonElement Json(SkillToolResult result)
    {
        Assert.True(result.Ok, result.Content);
        using JsonDocument doc = JsonDocument.Parse(result.Content);
        return doc.RootElement.Clone();
    }

    private static TaskCompletionSource<bool> Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static MultiAgentSession Session(
        Func<string, SkillTurnGenerator> factory, MultiAgentOptions? options = null,
        SkillToolContext? context = null, IReadOnlyList<ToolFunction>? tools = null,
        CancellationToken cancellationToken = default) =>
        new([new() { Role = "system", Content = "Preserve source evidence." },
             new() { Role = "developer", Content = "Respect workspace boundaries." },
             new() { Role = "user", Content = "Parent private conversation." },
             new() { Role = "assistant", Content = "Parent private reasoning." }],
            tools ?? SkillTools.BuiltIn(), context ?? new SkillToolContext([]), factory,
            options ?? new MultiAgentOptions { Enabled = true }, cancellationToken: cancellationToken);

    private static async Task<JsonElement> Wait(MultiAgentSession session, string? id = null)
    {
        ToolCall call = id is null
            ? Call("wait_agent", ("timeout_ms", 10000))
            : Call("wait_agent", ("agent_id", id), ("timeout_ms", 10000));
        return Json(await session.ExecuteAsync(call));
    }

    [Fact]
    public async Task IndependentChildrenOverlap_AndConcurrencyLimitQueuesExtraWork()
    {
        var release = Signal();
        var bothStarted = Signal();
        int active = 0, maximum = 0;
        await using var session = Session(_ => async (_, _, token) =>
        {
            int current = Interlocked.Increment(ref active);
            InterlockedExtensions.Max(ref maximum, current);
            if (current == 2) bothStarted.TrySetResult(true);
            try { await release.Task.WaitAsync(token); return new(new ParsedOutput { Content = "evidence" }); }
            finally { Interlocked.Decrement(ref active); }
        }, new() { Enabled = true, MaxConcurrentAgents = 2 });

        string first = Id(await session.ExecuteAsync(Spawn("first")));
        string second = Id(await session.ExecuteAsync(Spawn("second")));
        try
        {
            await bothStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal("queued", Json(await session.ExecuteAsync(Spawn("third"))).GetProperty("status").GetString());
            Assert.Equal(2, Volatile.Read(ref maximum));
        }
        finally { release.TrySetResult(true); }

        JsonElement results = await Wait(session);
        Assert.Equal(3, results.GetProperty("agents").GetArrayLength());
        Assert.All(results.GetProperty("agents").EnumerateArray(), item =>
        {
            Assert.Equal("completed", item.GetProperty("status").GetString());
            Assert.Equal("evidence", item.GetProperty("result").GetString());
        });
        Assert.NotEqual(first, second);
        Assert.Equal(0, Volatile.Read(ref active));
    }

    [Fact]
    public async Task ChildReceivesTaskAndPolicy_WithoutParentsConversationOrClientTools()
    {
        List<ChatMessage>? seenMessages = null;
        List<ToolFunction>? seenTools = null;
        var runner = new CountingRunner();
        var toolSet = SkillTools.BuiltIn();
        toolSet.Add(runner.Declare());
        toolSet.Add(new() { Name = "send_email" });
        await using var session = Session(_ => (messages, tools, _) =>
        {
            seenMessages = messages;
            seenTools = tools;
            return Answer("inspected");
        }, context: new SkillToolContext([]) { CodeRunner = runner }, tools: toolSet);

        string child = Id(await session.ExecuteAsync(Spawn("scope", task: "Only inspect this module.")));
        await Wait(session, child);
        Assert.Contains(seenMessages!, m => m.Role == "system" && m.Content!.Contains("Preserve source evidence."));
        Assert.Single(seenMessages!, m => m.Role == "system");
        Assert.Contains("You are a subagent, role explorer.", seenMessages![0].Content);
        Assert.DoesNotContain("/root/scope", seenMessages[0].Content);
        Assert.Contains("Your agent ID is /root/scope. Your parent is /root.",
            Assert.Single(seenMessages, m => m.Role == "user").Content);
        Assert.Contains("You are read-only.", seenMessages[0].Content);
        Assert.Contains(seenMessages!, m => m.Content!.Contains("Respect workspace boundaries."));
        Assert.Contains(seenMessages!, m => m.Content!.Contains("Only inspect this module."));
        Assert.DoesNotContain(seenMessages!, m => (m.Content ?? "").Contains("Parent private"));
        Assert.Contains(seenTools!, t => t.Name == "skills_read");
        Assert.DoesNotContain(seenTools!, t => t.Name is "shell" or "send_email" or "skills_run");
    }

    [Theory]
    [InlineData("explorer")]
    [InlineData("reviewer")]
    [InlineData("worker")]
    public async Task SameRoleChildrenShareRenderedPreamble_WhileIdentityAndTasksStayPrivate(string role)
    {
        var captured = new ConcurrentDictionary<string, (List<ChatMessage> Messages, List<ToolFunction> Tools)>();
        await using var session = Session(id => (messages, tools, _) =>
        {
            captured[id] = (new(messages), new(tools!));
            return Answer("checked");
        }, new() { Enabled = true, AllowWorkerTools = true });
        string first = Id(await session.ExecuteAsync(Spawn("proposal_a", role, "Sell 120 units at $25. Costs: $13 per unit and $300 fixed. Claimed profit: $1,300.")));
        string second = Id(await session.ExecuteAsync(Spawn("proposal_b", role, "Serve 200 customers paying $18. Costs: $7 per customer and $400 fixed. Claimed profit: $1,900.")));
        await Wait(session);

        var a = captured[first];
        var b = captured[second];
        static List<ChatMessage> Preamble(List<ChatMessage> messages) =>
            messages.TakeWhile(m => m.Role is "system" or "developer").ToList();
        Assert.Equal(JsonSerializer.Serialize(Preamble(a.Messages)), JsonSerializer.Serialize(Preamble(b.Messages)));
        string renderedA = ChatTemplate.RenderQwen35(a.Messages, tools: a.Tools);
        string renderedB = ChatTemplate.RenderQwen35(b.Messages, tools: b.Tools);
        const string userHeader = "<|im_start|>user\n";
        int boundaryA = renderedA.IndexOf(userHeader, StringComparison.Ordinal);
        int boundaryB = renderedB.IndexOf(userHeader, StringComparison.Ordinal);
        Assert.True(boundaryA > 0 && boundaryB > 0);
        string prefix = renderedA[..boundaryA];
        Assert.Equal(prefix, renderedB[..boundaryB]);
        Assert.Contains("<tools>", prefix);
        Assert.Contains("Preserve source evidence.", prefix);
        Assert.Contains("Respect workspace boundaries.", prefix);
        Assert.DoesNotContain(first, prefix);
        Assert.DoesNotContain(second, prefix);
        Assert.DoesNotContain("Sell 120", prefix);
        Assert.DoesNotContain("Serve 200", prefix);

        string firstTask = Assert.Single(a.Messages, m => m.Role == "user").Content!;
        string secondTask = Assert.Single(b.Messages, m => m.Role == "user").Content!;
        Assert.Contains($"Your agent ID is {first}. Your parent is /root.", firstTask);
        Assert.Contains($"Your agent ID is {second}. Your parent is /root.", secondTask);
        Assert.Contains("[Assigned task]\nSell 120 units at $25. Costs: $13 per unit and $300 fixed. Claimed profit: $1,300.\n\n[Workspace]", firstTask);
        Assert.Contains("[Assigned task]\nServe 200 customers paying $18. Costs: $7 per customer and $400 fixed. Claimed profit: $1,900.\n\n[Workspace]", secondTask);
        Assert.DoesNotContain(second, firstTask);
        Assert.DoesNotContain(first, secondTask);
        Assert.DoesNotContain("Parent private", prefix + firstTask + secondTask);
    }

    [Theory]
    [InlineData("explorer", false)]
    [InlineData("reviewer", true)]
    [InlineData("worker", false)]
    [InlineData("worker", true)]
    public async Task PredictedPublicProfilesExactlyMatchExecutedChildPrompts(string role, bool allowWorkerTools)
    {
        List<ChatMessage>? seenMessages = null;
        List<ToolFunction>? seenTools = null;
        await using var session = Session(_ => (messages, tools, _) =>
        {
            seenMessages = messages;
            seenTools = tools;
            return Answer("checked");
        }, new() { Enabled = true, AllowWorkerTools = allowWorkerTools });
        IReadOnlyList<MultiAgentPromptProfile> profiles = session.GetPromptProfiles();
        Assert.Equal(allowWorkerTools ? 4 : 3, profiles.Count);
        Assert.All(profiles, profile => Assert.All(profile.Messages, message =>
        {
            Assert.True(message.Role is "system" or "developer");
            Assert.DoesNotContain("Parent private", message.Content);
            Assert.DoesNotContain("/root/", message.Content);
        }));

        string child = Id(await session.ExecuteAsync(Spawn("predicted", role, "Private assigned task.")));
        await Wait(session, child);
        string actual = ChatTemplate.RenderQwen35(
            seenMessages!.TakeWhile(message => message.Role is "system" or "developer").ToList(),
            addGenerationPrompt: false, tools: seenTools);
        MultiAgentPromptProfile match = Assert.Single(profiles, profile =>
            ChatTemplate.RenderQwen35(profile.Messages.ToList(), addGenerationPrompt: false,
                tools: profile.Tools.ToList()) == actual);
        Assert.Equal(JsonSerializer.Serialize(seenTools), JsonSerializer.Serialize(match.Tools));
        Assert.DoesNotContain("Private assigned task.", actual);
        if (!allowWorkerTools || role != "worker")
            Assert.DoesNotContain(match.Tools, tool => tool.Name == SkillTools.RunToolName);
    }

    [Fact]
    public async Task DifferentChildRolesKeepDistinctPolicyAndToolPermissions()
    {
        var captured = new ConcurrentDictionary<string, (List<ChatMessage> Messages, List<ToolFunction> Tools)>();
        var runner = new CountingRunner();
        await using var session = Session(id => (messages, tools, _) =>
        {
            captured[id] = (new(messages), new(tools!));
            return Answer("checked");
        }, new() { Enabled = true, AllowWorkerTools = true },
            context: new SkillToolContext([]) { CodeRunner = runner }, tools: [runner.Declare()]);
        string reviewer = Id(await session.ExecuteAsync(Spawn("review", "reviewer")));
        string worker = Id(await session.ExecuteAsync(Spawn("edit", "worker")));
        await Wait(session);

        var readOnly = captured[reviewer];
        var mutable = captured[worker];
        Assert.Contains("role reviewer.", readOnly.Messages[0].Content);
        Assert.Contains("You are read-only.", readOnly.Messages[0].Content);
        Assert.Contains("role worker.", mutable.Messages[0].Content);
        Assert.Contains("Edit only assigned files.", mutable.Messages[0].Content);
        Assert.Contains("private workspace", mutable.Messages[0].Content);
        Assert.NotEqual(readOnly.Messages[0].Content, mutable.Messages[0].Content);
        Assert.DoesNotContain(readOnly.Tools, t => t.Name == "shell");
        Assert.Contains(mutable.Tools, t => t.Name == "shell");
    }

    [Theory]
    [InlineData("explorer")]
    [InlineData("reviewer")]
    public async Task ReadOnlyRolesCannotExecuteHallucinatedMutationTools(string role)
    {
        var runner = new CountingRunner();
        int round = 0;
        await using var session = Session(_ => (messages, _, _) => ++round == 1
            ? Calls(Call("shell", ("command", "must-never-run")))
            : Answer(string.Join("\n", messages.Where(m => m.Role is "tool" or "user").Select(m => m.Content))),
            context: new SkillToolContext([]) { CodeRunner = runner }, tools: [runner.Declare()]);
        string child = Id(await session.ExecuteAsync(Spawn("readonly", role)));
        JsonElement result = (await Wait(session, child)).GetProperty("agents")[0];
        Assert.Equal(0, runner.Calls);
        Assert.Contains("error", result.GetProperty("result").GetString()!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task IsolatedWorkerToolsCanOverlapWithParentHostTools()
    {
        var runner = new CountingRunner(delayMilliseconds: 35);
        await using var session = Session(_ =>
        {
            int round = 0;
            return (_, _, _) => ++round == 1 ? Calls(Call("shell", ("command", "fixture"))) : Answer("done");
        }, new() { Enabled = true, AllowWorkerTools = true },
            context: new SkillToolContext([]) { CodeRunner = runner }, tools: [runner.Declare()]);
        Id(await session.ExecuteAsync(Spawn("one", "worker")));
        Id(await session.ExecuteAsync(Spawn("two", "worker")));
        Task<SkillToolResult> parent = session.ExecuteHostToolAsync(Call("shell", ("command", "fixture")));
        await Wait(session);
        Assert.True((await parent).Ok);
        Assert.Equal(3, runner.Calls);
        Assert.InRange(runner.MaximumActive, 2, 3);
    }

    [Fact]
    public async Task DepthAndTotalAgentBudgetsApplyAcrossTree()
    {
        var release = Signal();
        await using var session = Session(_ => async (_, _, ct) =>
        {
            await release.Task.WaitAsync(ct);
            return new(new ParsedOutput { Content = "done" });
        }, new() { Enabled = true, MaxConcurrentAgents = 3, MaxDepth = 2, MaxAgents = 2 });
        string child = Id(await session.ExecuteAsync(Spawn("child")));
        string grandchild = Id(await session.ExecuteAsync(Spawn("grandchild"), child));
        Assert.False((await session.ExecuteAsync(Spawn("too_deep"), grandchild)).Ok);
        release.TrySetResult(true);
        await Wait(session);
        Assert.False((await session.ExecuteAsync(Spawn("over_total"))).Ok);
    }

    [Fact]
    public async Task GlobalGenerationBudgetStopsRunawayChildren()
    {
        int generations = 0;
        await using var session = Session(_ => (_, _, _) =>
        {
            Interlocked.Increment(ref generations);
            return Calls(Call("skills_list"));
        }, new() { Enabled = true, MaxTotalChildGenerations = 3, MaxRoundsPerAgent = 20 });
        string child = Id(await session.ExecuteAsync(Spawn("loop")));
        JsonElement result = (await Wait(session, child)).GetProperty("agents")[0];
        Assert.InRange(Volatile.Read(ref generations), 1, 3);
        Assert.NotEqual("running", result.GetProperty("status").GetString());
        Assert.Contains("budget", result.ToString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RootCancellationReachesChildGeneration()
    {
        using var cancellation = new CancellationTokenSource();
        var started = Signal();
        var cancelled = Signal();
        await using var session = Session(_ => async (_, _, ct) =>
        {
            started.TrySetResult(true);
            try { await Task.Delay(Timeout.Infinite, ct); }
            finally { if (ct.IsCancellationRequested) cancelled.TrySetResult(true); }
            return new(new ParsedOutput());
        }, cancellationToken: cancellation.Token);
        Id(await session.ExecuteAsync(Spawn("cancel")));
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        await cancelled.Task.WaitAsync(TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task TimeoutIsReportedAsFailure_NotSuccessfulEmptyAnswer()
    {
        await using var session = Session(_ => async (_, _, ct) =>
        {
            await Task.Delay(Timeout.Infinite, ct);
            return new(new ParsedOutput());
        }, new() { Enabled = true, AgentTimeoutSeconds = 1 });
        string child = Id(await session.ExecuteAsync(Spawn("timeout")));
        JsonElement result = (await Wait(session, child)).GetProperty("agents")[0];
        Assert.NotEqual("completed", result.GetProperty("status").GetString());
        Assert.False(string.IsNullOrWhiteSpace(result.GetProperty("error").GetString()));
    }

    [Fact]
    public async Task CloseCancelsChildAndKeepsSiblingAvailable()
    {
        var release = Signal();
        await using var session = Session(_ => async (_, _, ct) =>
        {
            await release.Task.WaitAsync(ct);
            return new(new ParsedOutput { Content = "survived" });
        });
        string first = Id(await session.ExecuteAsync(Spawn("first")));
        string second = Id(await session.ExecuteAsync(Spawn("second")));
        Assert.True((await session.ExecuteAsync(Call("close_agent", ("agent_id", first)))).Ok);
        release.TrySetResult(true);
        JsonElement sibling = (await Wait(session, second)).GetProperty("agents")[0];
        Assert.Equal("completed", sibling.GetProperty("status").GetString());
        Assert.Equal("survived", sibling.GetProperty("result").GetString());
        JsonElement closed = (await Wait(session, first)).GetProperty("agents")[0];
        Assert.NotEqual("completed", closed.GetProperty("status").GetString());
    }

    [Fact]
    public async Task ChildrenCannotControlUnrelatedSiblings()
    {
        var release = Signal();
        await using var session = Session(_ => async (_, _, ct) =>
        {
            await release.Task.WaitAsync(ct);
            return new(new ParsedOutput { Content = "done" });
        });
        string first = Id(await session.ExecuteAsync(Spawn("first")));
        string second = Id(await session.ExecuteAsync(Spawn("second")));
        Assert.False((await session.ExecuteAsync(Call("close_agent", ("agent_id", second)), first)).Ok);
        Assert.False((await session.ExecuteAsync(Call("send_input", ("agent_id", second), ("message", "stop")), first)).Ok);
        release.TrySetResult(true);
        Assert.All((await Wait(session)).GetProperty("agents").EnumerateArray(),
            row => Assert.Equal("completed", row.GetProperty("status").GetString()));
    }

    [Fact]
    public async Task FailedChildDoesNotLoseSuccessfulSiblingEvidence()
    {
        await using var session = Session(id => (_, _, _) => id.EndsWith("bad", StringComparison.Ordinal)
            ? Task.FromException<SkillTurnOutput>(new InvalidOperationException("fixture failed"))
            : Answer("verified evidence"));
        Id(await session.ExecuteAsync(Spawn("bad")));
        Id(await session.ExecuteAsync(Spawn("good")));
        JsonElement results = (await Wait(session)).GetProperty("agents");
        Assert.Contains(results.EnumerateArray(), row => row.GetProperty("status").GetString() == "failed");
        Assert.Contains(results.EnumerateArray(), row => row.GetProperty("result").GetString() == "verified evidence");
    }

    [Fact]
    public async Task FinalAnswerWaitsForChildrenAndSynthesizesTheirEvidence()
    {
        int parentRounds = 0;
        var options = new SkillAgentLoopOptions
        {
            MultiAgent = new() { Enabled = true },
            SubagentGeneratorFactory = _ => async (_, _, ct) =>
            {
                await Task.Delay(30, ct);
                return new(new ParsedOutput { Content = "UNIQUE_CHILD_EVIDENCE" });
            },
        };
        SkillLoopResult result = await SkillAgentLoop.RunAsync([new() { Role = "user", Content = "Inspect and report." }],
            null, new SkillToolContext([]), (messages, _, _) =>
            {
                int round = ++parentRounds;
                if (round == 1) return Calls(Spawn("inspect"));
                if (!messages.Any(m => (m.Content ?? "").Contains("UNIQUE_CHILD_EVIDENCE")))
                    return Task.FromResult(new SkillTurnOutput(new ParsedOutput
                    { Content = "Premature parent answer", Thinking = "Independent parent analysis" }, [31, 32])
                    { RawPromptTrailingWhitespace = "\n", RawGenerationSuffix = "assistant:" });
                ChatMessage provisional = Assert.Single(messages, m => m.Content == "Premature parent answer");
                Assert.Equal("Independent parent analysis", provisional.Thinking);
                Assert.Equal(new[] { 31, 32 }, provisional.RawOutputTokens);
                Assert.Equal("\n", provisional.RawPromptTrailingWhitespace);
                Assert.Equal("assistant:", provisional.RawGenerationSuffix);
                return Answer("Synthesis: UNIQUE_CHILD_EVIDENCE");
            }, options);

        Assert.Equal("Synthesis: UNIQUE_CHILD_EVIDENCE", result.Output.Parsed.Content);
        Assert.True(parentRounds >= 3);
        Assert.Empty(result.PendingClientToolCalls);
        Assert.False(result.HitRoundLimit);
    }

    [Fact]
    public async Task ClientToolWithAgentNameKeepsClientOwnership()
    {
        int childGenerations = 0;
        var clientTool = new ToolFunction { Name = "spawn_agent", Description = "Client operation" };
        SkillLoopResult result = await SkillAgentLoop.RunAsync([new() { Role = "user", Content = "Do client work." }],
            [clientTool], new SkillToolContext([]), (_, _, _) => Calls(Spawn("client")), new()
            {
                ClientTools = [clientTool],
                MultiAgent = new() { Enabled = true },
                SubagentGeneratorFactory = _ => (_, _, _) => { childGenerations++; return Answer("wrong"); },
            });
        Assert.Equal("spawn_agent", Assert.Single(result.PendingClientToolCalls).Name);
        Assert.Equal(0, childGenerations);
        Assert.DoesNotContain(result.Invocations, call => call.Tool == "spawn_agent");
    }

    [Fact]
    public async Task SimpleTaskDoesNotCreateChildren()
    {
        int factories = 0;
        SkillLoopResult result = await SkillAgentLoop.RunAsync([new() { Role = "user", Content = "Say hello." }],
            null, new SkillToolContext([]), (_, _, _) => Answer("Hello."), new()
            {
                MultiAgent = new() { Enabled = true },
                SubagentGeneratorFactory = _ => { factories++; return (_, _, _) => Answer("unnecessary"); },
            });
        Assert.Equal("Hello.", result.Output.Parsed.Content);
        Assert.Equal(1, result.Rounds);
        Assert.Equal(0, factories);
    }

    [Fact]
    public async Task MalformedAndOversizedTasksAreRejectedBeforeGeneration()
    {
        int generated = 0;
        await using var session = Session(_ => (_, _, _) => { generated++; return Answer("wrong"); },
            new() { Enabled = true, MaxTaskCharacters = 256 });
        Assert.False((await session.ExecuteAsync(Call("spawn_agent", ("task_name", "missing")))).Ok);
        Assert.False((await session.ExecuteAsync(Spawn("large", task: new string('x', 257)))).Ok);
        Assert.False((await session.ExecuteAsync(Spawn("role", "unknown"))).Ok);
        Assert.Equal(0, generated);
    }

    [Fact]
    public async Task FollowUpRetainsRawGenerationAndFullContext_WhileParentReportIsBounded()
    {
        string full = new('x', 400);
        int turns = 0;
        await using var session = Session(_ => (messages, _, _) =>
        {
            if (++turns == 1)
                return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = full, Thinking = "check" }, new[] { 17, 18 })
                { RawPromptTrailingWhitespace = "\n", RawGenerationSuffix = "suffix" });
            ChatMessage previous = messages.Last(m => m.Role == "assistant");
            Assert.Equal(full, previous.Content);
            Assert.Equal(new[] { 17, 18 }, previous.RawOutputTokens);
            Assert.Equal("\n", previous.RawPromptTrailingWhitespace);
            Assert.Equal("suffix", previous.RawGenerationSuffix);
            return Answer("follow-up checked");
        }, new() { MaxResultCharacters = 256 });
        string id = Id(await session.ExecuteAsync(Spawn("retained")));
        JsonElement first = await Wait(session, id);
        Assert.True(first.GetProperty("agents")[0].GetProperty("result").GetString()!.Length <= 256);
        Assert.True((await session.ExecuteAsync(Call("send_input", ("agent_id", id), ("message", "Follow up")))).Ok);
        JsonElement second = await Wait(session, id);
        Assert.Equal("follow-up checked", second.GetProperty("agents")[0].GetProperty("result").GetString());
    }

    [Fact]
    public async Task FollowUpResumesCompletedChildWithItsOwnHistoryAndGenerator()
    {
        int factories = 0;
        List<ChatMessage>? resumed = null;
        await using var session = Session(_ =>
        {
            factories++;
            return (messages, _, _) =>
            {
                resumed = messages;
                return Answer("Report: " + messages.Last(m => m.Role == "user").Content);
            };
        });
        string child = Id(await session.ExecuteAsync(Spawn("follow", task: "Initial inspection")));
        await Wait(session, child);
        Assert.True((await session.ExecuteAsync(Call("send_input", ("agent_id", child), ("message", "Check the edge case")))).Ok);
        JsonElement result = (await Wait(session, child)).GetProperty("agents")[0];
        Assert.Equal("Report: Check the edge case", result.GetProperty("result").GetString());
        Assert.Contains(resumed!, m => m.Role == "assistant"
            && m.Content!.Contains("[Assigned task]\nInitial inspection\n\n[Workspace]", StringComparison.Ordinal));
        ChatMessage initialTask = Assert.Single(resumed!, m => m.Role == "user"
            && m.Content!.Contains("[TensorSharp subagent identity]", StringComparison.Ordinal));
        Assert.Contains($"Your agent ID is {child}. Your parent is /root.", initialTask.Content);
        Assert.Equal("Check the edge case", resumed!.Last(m => m.Role == "user").Content);
        Assert.Equal(1, factories);
    }

    [Fact]
    public async Task FollowUpDuringGenerationIsDeliveredBeforeChildCompletes()
    {
        var started = Signal();
        var release = Signal();
        int rounds = 0;
        await using var session = Session(_ => async (messages, _, ct) =>
        {
            if (Interlocked.Increment(ref rounds) == 1)
            {
                started.TrySetResult(true);
                await release.Task.WaitAsync(ct);
                return new(new ParsedOutput { Content = "Initial report" });
            }
            return new(new ParsedOutput { Content = messages.Last(m => m.Role == "user").Content });
        });
        string child = Id(await session.ExecuteAsync(Spawn("mailbox")));
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True((await session.ExecuteAsync(Call("send_input", ("agent_id", child), ("message", "Include this correction")))).Ok);
        release.TrySetResult(true);
        JsonElement result = (await Wait(session, child)).GetProperty("agents")[0];
        Assert.Equal("Include this correction", result.GetProperty("result").GetString());
        Assert.Equal(2, rounds);
    }

    [Fact]
    public async Task CancellingWaitDoesNotCancelIndependentChildWork()
    {
        var release = Signal();
        await using var session = Session(_ => async (_, _, ct) =>
        {
            await release.Task.WaitAsync(ct);
            return new(new ParsedOutput { Content = "still completed" });
        });
        string child = Id(await session.ExecuteAsync(Spawn("wait")));
        using var cancelWait = new CancellationTokenSource();
        Task<SkillToolResult> waiting = session.ExecuteAsync(Call("wait_agent", ("agent_id", child), ("timeout_ms", 10000)),
            cancellationToken: cancelWait.Token);
        cancelWait.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting);
        release.TrySetResult(true);
        Assert.Equal("still completed", (await Wait(session, child)).GetProperty("agents")[0].GetProperty("result").GetString());
    }

    [Fact]
    public async Task ChildReportsAreBoundedAndCollectedOnce()
    {
        await using var session = Session(_ => (_, _, _) => Answer(new string('x', 2000)),
            new() { Enabled = true, MaxResultCharacters = 256 });
        Id(await session.ExecuteAsync(Spawn("bounded")));
        Assert.True(session.HasPendingResults());
        string report = await session.CollectResultsAsync();
        using JsonDocument json = JsonDocument.Parse(report[report.IndexOf('{')..]);
        string answer = json.RootElement.GetProperty("agents")[0].GetProperty("result").GetString()!;
        Assert.True(answer.Length <= 256);
        Assert.Contains("truncated", answer, StringComparison.OrdinalIgnoreCase);
        Assert.False(session.HasPendingResults());
    }

    [Fact]
    public async Task ReadOnlyParentCannotRegainMutationToolsThroughWorkerDescendant()
    {
        var release = Signal();
        var runner = new CountingRunner();
        await using var session = Session(id =>
        {
            int round = 0;
            return async (_, _, ct) =>
            {
                if (id.EndsWith("reader", StringComparison.Ordinal))
                {
                    await release.Task.WaitAsync(ct);
                    return new(new ParsedOutput { Content = "reader done" });
                }
                return ++round == 1 ? await Calls(Call("shell", ("command", "must-never-run"))) : await Answer("worker done");
            };
        }, new() { Enabled = true, AllowWorkerTools = true },
            context: new SkillToolContext([]) { CodeRunner = runner }, tools: [runner.Declare()]);
        string reader = Id(await session.ExecuteAsync(Spawn("reader")));
        string worker = Id(await session.ExecuteAsync(Spawn("worker", "worker"), reader));
        Assert.True((await session.ExecuteAsync(Call("wait_agent", ("agent_id", worker), ("timeout_ms", 10000)), reader)).Ok);
        release.TrySetResult(true);
        await Wait(session, reader);
        Assert.Equal(0, runner.Calls);
    }

    [Fact]
    public async Task CancellationStillWinsWhenGeneratorReturnsSuccessAfterCancellation()
    {
        var started = Signal();
        var release = Signal();
        await using var session = Session(_ => async (_, _, _) =>
        {
            started.TrySetResult(true);
            await release.Task;
            return new(new ParsedOutput { Content = "late success must not win" });
        });
        string child = Id(await session.ExecuteAsync(Spawn("late")));
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True((await session.ExecuteAsync(Call("close_agent", ("agent_id", child)))).Ok);
        release.TrySetResult(true);
        JsonElement result = (await Wait(session, child)).GetProperty("agents")[0];
        Assert.Equal("cancelled", result.GetProperty("status").GetString());
        Assert.NotEqual("late success must not win", result.GetProperty("result").GetString());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ExplorerCanReadWorkspaceOnlyWhenParentDeclaresReadFile(bool declared)
    {
        var runner = new CountingRunner();
        List<ToolFunction>? offered = null;
        int round = 0;
        await using var session = Session(_ => (_, tools, _) =>
        {
            offered = tools;
            return ++round == 1 ? Calls(Call("read_file", ("path", "fixture.txt"))) : Answer("done");
        }, context: new SkillToolContext([]) { CodeRunner = runner },
            tools: declared ? [new() { Name = "read_file" }, runner.Declare()] : [runner.Declare()]);
        string child = Id(await session.ExecuteAsync(Spawn("read")));
        await Wait(session, child);
        Assert.Equal(declared, offered!.Any(t => t.Name == "read_file"));
        Assert.DoesNotContain(offered!, t => t.Name == "shell");
        Assert.Equal(declared ? 1 : 0, runner.Calls);
    }

    [Fact]
    public async Task ConcurrentDisposalAndNewWorkLeaveNoRunningChildCallbacks()
    {
        var started = Signal();
        int active = 0;
        var session = Session(_ => async (_, _, token) =>
        {
            Interlocked.Increment(ref active);
            started.TrySetResult(true);
            try
            {
                await Task.Delay(Timeout.Infinite, token);
                return new(new ParsedOutput());
            }
            finally { Interlocked.Decrement(ref active); }
        });
        string child = Id(await session.ExecuteAsync(Spawn("initial")));
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Task[] attempts = Enumerable.Range(0, 12).Select(index => Task.Run(async () =>
        {
            try
            {
                await session.ExecuteAsync(index % 2 == 0 ? Spawn("race" + index)
                    : Call("send_input", ("agent_id", child), ("message", "follow-up")));
            }
            catch (OperationCanceledException) { }
            catch (ObjectDisposedException) { }
        })).ToArray();
        Task disposal = session.DisposeAsync().AsTask();
        await Task.WhenAll(attempts.Append(disposal)).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(0, Volatile.Read(ref active));
    }

    [Theory]
    [InlineData("length", "limit_reached")]
    [InlineData("max_tokens", "limit_reached")]
    [InlineData("thinking_budget", "limit_reached")]
    [InlineData("repetition", "limit_reached")]
    [InlineData("aborted", "failed")]
    [InlineData("error", "failed")]
    [InlineData("content_filter", "failed")]
    [InlineData("cancelled", "cancelled")]
    public async Task IncompleteGenerationsCannotExecutePartiallyEmittedTools(string finishReason, string expectedStatus)
    {
        var runner = new CountingRunner();
        await using var session = Session(_ => (_, _, _) => Task.FromResult(new SkillTurnOutput(new ParsedOutput
        {
            Content = "Incomplete report",
            ToolCalls = [Call("shell", ("command", "must-never-run"))],
        }) { FinishReason = finishReason }), new() { Enabled = true, AllowWorkerTools = true },
            context: new SkillToolContext([]) { CodeRunner = runner }, tools: [runner.Declare()]);
        string child = Id(await session.ExecuteAsync(Spawn("incomplete", "worker")));
        JsonElement result = (await Wait(session, child)).GetProperty("agents")[0];
        Assert.Equal(expectedStatus, result.GetProperty("status").GetString());
        Assert.Contains(finishReason, result.GetProperty("error").GetString()!);
        Assert.Equal(0, runner.Calls);
    }

    [Fact]
    public async Task EmptyChildReportIsFailureWithAnExplanation()
    {
        await using var session = Session(_ => (_, _, _) => Answer("  "));
        string child = Id(await session.ExecuteAsync(Spawn("empty")));
        JsonElement result = (await Wait(session, child)).GetProperty("agents")[0];
        Assert.Equal("failed", result.GetProperty("status").GetString());
        Assert.False(string.IsNullOrWhiteSpace(result.GetProperty("error").GetString()));
    }

    private sealed class CountingRunner(int delayMilliseconds = 0) : ICodeRunner
    {
        private int _calls, _active, _maximumActive;
        public int Calls => Volatile.Read(ref _calls);
        public int MaximumActive => Volatile.Read(ref _maximumActive);
        public bool CanRun => true;
        public string? UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = "shell" };
        public IReadOnlyList<ToolFunction> DeclareWorkspaceTools(bool allowWrite) => allowWrite
            ? [Declare(), new() { Name = "read_file" }] : [new() { Name = "read_file" }];
        public ICodeRunner ForkForWorkspace(SessionWorkspace workspace, bool allowWrite) => new ScopedRunner(this, workspace, allowWrite);
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null)
        {
            Interlocked.Increment(ref _calls);
            InterlockedExtensions.Max(ref _maximumActive, Interlocked.Increment(ref _active));
            try
            {
                if (delayMilliseconds > 0) Thread.Sleep(delayMilliseconds);
                return new(true, "fixture output", null, null);
            }
            finally { Interlocked.Decrement(ref _active); }
        }

        private sealed class ScopedRunner(CountingRunner owner, SessionWorkspace scope, bool allowWrite) : ICodeRunner
        {
            public bool CanRun => true;
            public string? UnavailableReason => null;
            public ToolFunction Declare() => DeclareTools()[0];
            public IReadOnlyList<ToolFunction> DeclareTools() => owner.DeclareWorkspaceTools(allowWrite);
            public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
                Action<string>? onOutput = null, SessionWorkspace? workspace = null,
                IReadOnlyList<string>? skillDirectories = null)
            {
                if (!ReferenceEquals(workspace, scope) || (!allowWrite && call.Name != "read_file"))
                    return SkillToolResult.Failure("Fixture scope or permission mismatch.");
                return owner.Execute(call, onOutput: onOutput, workspace: scope);
            }
        }
    }

    private static class InterlockedExtensions
    {
        public static void Max(ref int location, int value)
        {
            int current;
            do { current = Volatile.Read(ref location); if (current >= value) return; }
            while (Interlocked.CompareExchange(ref location, value, current) != current);
        }
    }
}
