using System.Collections.Concurrent;
using System.Text.Json;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.CodeExec;

namespace InferenceWeb.Tests;

/// <summary>Deterministic DAG and permission checks; these do not measure model quality.</summary>
public sealed class MultiAgentDependencyTests
{
    private static ToolCall Call(string name, params (string Key, object Value)[] arguments) => new()
    {
        Id = Guid.NewGuid().ToString("N"), Name = name,
        Arguments = arguments.ToDictionary(pair => pair.Key, pair => pair.Value),
    };

    private static ToolCall Spawn(string name, string? dependencies = null, string role = "explorer", string? permissions = null)
    {
        var call = Call("spawn_agent", ("task_name", name), ("agent_type", role), ("task", "Complete " + name));
        if (dependencies != null) call.Arguments!["depends_on"] = dependencies;
        if (permissions != null) call.Arguments!["permissions"] = permissions;
        return call;
    }

    private static TaskCompletionSource<bool> Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);
    private static Task<SkillTurnOutput> Answer(string text) => Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = text }));
    private static Task<SkillTurnOutput> Calls(params ToolCall[] calls) => Task.FromResult(new SkillTurnOutput(new ParsedOutput { ToolCalls = calls.ToList() }));
    private static JsonElement Json(SkillToolResult result)
    {
        Assert.True(result.Ok, result.Content);
        using JsonDocument document = JsonDocument.Parse(result.Content);
        return document.RootElement.Clone();
    }

    private static string Id(SkillToolResult result) => Json(result).GetProperty("agent_id").GetString()!;
    private static MultiAgentSession Session(Func<string, SkillTurnGenerator> factory, int capacity = 2,
        bool allowWorkers = false, CancellationToken cancellationToken = default) => new(
            [new() { Role = "system", Content = "Preserve evidence and file ownership." }],
            SkillTools.BuiltIn(), new SkillToolContext([]), factory,
            new() { Enabled = true, MaxConcurrentAgents = capacity, MaxAgents = 16, AllowWorkerTools = allowWorkers },
            cancellationToken: cancellationToken);

    private static async Task<JsonElement> Wait(MultiAgentSession session, string? id = null, string caller = MultiAgentSession.RootId)
    {
        ToolCall call = Call("wait_agent", ("timeout_ms", 10000));
        if (id != null) call.Arguments!["agent_id"] = id;
        JsonElement result = Json(await session.ExecuteAsync(call, caller).WaitAsync(TimeSpan.FromSeconds(12)));
        Assert.False(result.GetProperty("timed_out").GetBoolean(), result.ToString());
        return result.GetProperty("agents");
    }

    private static JsonElement Row(JsonElement rows, string id) =>
        Assert.Single(rows.EnumerateArray(), row => row.GetProperty("agent_id").GetString() == id);

    [Fact]
    public async Task ReadyTasksBeyondCapacityQueueAndEveryTaskEventuallyRuns()
    {
        var release = Signal();
        var full = Signal();
        var started = new ConcurrentDictionary<string, bool>();
        int active = 0, maximum = 0;
        await using var session = Session(id => async (_, _, ct) =>
        {
            started.TryAdd(id, true);
            int current = Interlocked.Increment(ref active);
            int previous;
            do { previous = Volatile.Read(ref maximum); }
            while (current > previous && Interlocked.CompareExchange(ref maximum, current, previous) != previous);
            if (current == 2) full.TrySetResult(true);
            try { await release.Task.WaitAsync(ct); return new(new ParsedOutput { Content = id }); }
            finally { Interlocked.Decrement(ref active); }
        });
        string[] ids = new string[6];
        for (int i = 0; i < ids.Length; i++) ids[i] = Id(await session.ExecuteAsync(Spawn("task" + i)));
        await full.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(2, started.Count);
        Assert.Equal(4, session.GetProgress().Count(agent => agent.Status == "queued"));
        Assert.True(session.HasPendingResults());
        release.TrySetResult(true);
        JsonElement rows = await Wait(session);
        Assert.Equal(ids.Length, rows.GetArrayLength());
        Assert.All(rows.EnumerateArray(), row => Assert.Equal("completed", row.GetProperty("status").GetString()));
        Assert.Equal(ids.Length, started.Count);
        Assert.Equal(2, maximum);
        Assert.Equal(0, active);
    }

    [Fact]
    public async Task DependentWaitsForEveryPrerequisiteAndReceivesEvidence_IndependentTaskStillRuns()
    {
        var releaseA = Signal();
        var releaseB = Signal();
        var independentStarted = Signal();
        var dependentStarted = Signal();
        List<ChatMessage>? dependentMessages = null;
        await using var session = Session(id => async (messages, _, ct) =>
        {
            if (id.EndsWith("/a", StringComparison.Ordinal)) await releaseA.Task.WaitAsync(ct);
            if (id.EndsWith("/b", StringComparison.Ordinal)) await releaseB.Task.WaitAsync(ct);
            if (id.EndsWith("/independent", StringComparison.Ordinal)) independentStarted.TrySetResult(true);
            if (id.EndsWith("/merge", StringComparison.Ordinal))
            {
                dependentMessages = new(messages);
                dependentStarted.TrySetResult(true);
            }
            return new(new ParsedOutput { Content = "EVIDENCE:" + id });
        }, capacity: 3);
        string a = Id(await session.ExecuteAsync(Spawn("a")));
        string b = Id(await session.ExecuteAsync(Spawn("b")));
        string merge = Id(await session.ExecuteAsync(Spawn("merge", a + ",\n" + b)));
        Id(await session.ExecuteAsync(Spawn("independent")));
        await independentStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(dependentStarted.Task.IsCompleted);
        Assert.Equal("waiting", Assert.Single(session.GetProgress(), agent => agent.AgentId == merge).Status);
        releaseA.TrySetResult(true);
        await Wait(session, a);
        Assert.False(dependentStarted.Task.IsCompleted);
        releaseB.TrySetResult(true);
        JsonElement rows = await Wait(session);
        Assert.Equal("completed", Row(rows, merge).GetProperty("status").GetString());
        string context = string.Join("\n", dependentMessages!.Select(message => message.Content));
        Assert.Contains("EVIDENCE:" + a, context);
        Assert.Contains("EVIDENCE:" + b, context);
    }

    [Fact]
    public async Task FailedPrerequisiteBlocksItsTransitiveDependentsAndPreservesIndependentWork()
    {
        var release = Signal();
        var started = new ConcurrentDictionary<string, bool>();
        await using var session = Session(id => async (_, _, ct) =>
        {
            started.TryAdd(id, true);
            if (id.EndsWith("/fail", StringComparison.Ordinal))
            {
                await release.Task.WaitAsync(ct);
                throw new InvalidOperationException("fixture prerequisite failure");
            }
            return new(new ParsedOutput { Content = "independent result" });
        });
        string failed = Id(await session.ExecuteAsync(Spawn("fail")));
        string blocked = Id(await session.ExecuteAsync(Spawn("blocked", failed)));
        string downstream = Id(await session.ExecuteAsync(Spawn("downstream", blocked)));
        string independent = Id(await session.ExecuteAsync(Spawn("independent")));
        release.TrySetResult(true);
        JsonElement rows = await Wait(session);
        Assert.Equal("failed", Row(rows, failed).GetProperty("status").GetString());
        foreach (string id in new[] { blocked, downstream })
        {
            JsonElement row = Row(rows, id);
            Assert.Equal("blocked", row.GetProperty("status").GetString());
            Assert.False(string.IsNullOrWhiteSpace(row.GetProperty("error").GetString()));
            Assert.False(started.ContainsKey(id));
        }
        Assert.Equal("completed", Row(rows, independent).GetProperty("status").GetString());
    }

    [Fact]
    public async Task ClosingPrerequisiteBlocksDependentWithoutStartingIt()
    {
        var sourceStarted = Signal();
        int dependentRuns = 0;
        await using var session = Session(id => async (_, _, ct) =>
        {
            if (id.EndsWith("/source", StringComparison.Ordinal))
            {
                sourceStarted.TrySetResult(true);
                await Task.Delay(Timeout.Infinite, ct);
            }
            else Interlocked.Increment(ref dependentRuns);
            return new(new ParsedOutput { Content = "unexpected" });
        });
        string source = Id(await session.ExecuteAsync(Spawn("source")));
        string dependent = Id(await session.ExecuteAsync(Spawn("dependent", source)));
        await sourceStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True((await session.ExecuteAsync(Call("close_agent", ("agent_id", source)))).Ok);
        JsonElement rows = await Wait(session);
        Assert.Equal("cancelled", Row(rows, source).GetProperty("status").GetString());
        Assert.Equal("blocked", Row(rows, dependent).GetProperty("status").GetString());
        Assert.Equal(0, dependentRuns);
    }

    [Fact]
    public async Task DependencyReferencesMustNameExistingDirectSiblings()
    {
        var release = Signal();
        await using var session = Session(_ => async (_, _, ct) =>
        {
            await release.Task.WaitAsync(ct);
            return new(new ParsedOutput { Content = "done" });
        }, capacity: 4);
        string owner = Id(await session.ExecuteAsync(Spawn("owner")));
        string sibling = Id(await session.ExecuteAsync(Spawn("sibling")));
        string nested = Id(await session.ExecuteAsync(Spawn("nested"), owner));
        Assert.False((await session.ExecuteAsync(Spawn("unknown", "/root/missing"))).Ok);
        Assert.False((await session.ExecuteAsync(Spawn("self", "/root/self"))).Ok);
        Assert.False((await session.ExecuteAsync(Spawn("foreign", nested))).Ok);
        Assert.False((await session.ExecuteAsync(Spawn("foreign", sibling), owner)).Ok);
        Assert.Equal(3, session.GetProgress().Count);
        release.TrySetResult(true);
        await Wait(session);
    }

    [Fact]
    public async Task ClosingQueuedTaskNeverCreatesItsGenerator()
    {
        var started = Signal();
        var release = Signal();
        var generators = new ConcurrentBag<string>();
        await using var session = Session(id =>
        {
            generators.Add(id);
            return async (_, _, ct) =>
            {
                started.TrySetResult(true);
                await release.Task.WaitAsync(ct);
                return new(new ParsedOutput { Content = "done" });
            };
        }, capacity: 1);
        Id(await session.ExecuteAsync(Spawn("active")));
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        string queued = Id(await session.ExecuteAsync(Spawn("queued")));
        Assert.True((await session.ExecuteAsync(Call("close_agent", ("agent_id", queued)))).Ok);
        JsonElement row = Row(await Wait(session, queued), queued);
        Assert.Equal("cancelled", row.GetProperty("status").GetString());
        release.TrySetResult(true);
        await Wait(session);
        Assert.DoesNotContain(queued, generators);
    }

    [Fact]
    public async Task RootCancellationDrainsRunningQueuedAndWaitingTasksWithoutStartingQueuedWork()
    {
        using var cancellation = new CancellationTokenSource();
        var started = Signal();
        var generators = new ConcurrentBag<string>();
        await using var session = Session(id =>
        {
            generators.Add(id);
            return async (_, _, ct) =>
            {
                started.TrySetResult(true);
                await Task.Delay(Timeout.Infinite, ct);
                return new(new ParsedOutput { Content = "unexpected" });
            };
        }, capacity: 1, cancellationToken: cancellation.Token);
        string active = Id(await session.ExecuteAsync(Spawn("active")));
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Id(await session.ExecuteAsync(Spawn("queued")));
        Id(await session.ExecuteAsync(Spawn("dependent", active)));
        cancellation.Cancel();
        await session.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Single(generators);
        Assert.All(session.GetProgress(), agent => Assert.Equal("cancelled", agent.Status));
    }

    [Fact]
    public async Task FollowUpAtCapacityQueuesAndRetainsChildHistory()
    {
        var release = Signal();
        int turns = 0;
        await using var session = Session(id => async (messages, _, ct) =>
        {
            if (id.EndsWith("/busy", StringComparison.Ordinal)) await release.Task.WaitAsync(ct);
            else if (Interlocked.Increment(ref turns) == 2)
            {
                Assert.Contains(messages, message => message.Role == "assistant" && message.Content == "first report");
                Assert.Equal("follow-up", messages.Last(message => message.Role == "user").Content);
                return new(new ParsedOutput { Content = "second report" });
            }
            return new(new ParsedOutput { Content = "first report" });
        }, capacity: 1);
        string follow = Id(await session.ExecuteAsync(Spawn("follow")));
        await Wait(session, follow);
        Id(await session.ExecuteAsync(Spawn("busy")));
        JsonElement accepted = Json(await session.ExecuteAsync(Call("send_input", ("agent_id", follow), ("message", "follow-up"))));
        Assert.Equal("queued", accepted.GetProperty("status").GetString());
        Assert.Equal(1, turns);
        release.TrySetResult(true);
        Assert.Equal("second report", Row(await Wait(session), follow).GetProperty("result").GetString());
        Assert.Equal(2, turns);
    }

    [Fact]
    public async Task PrerequisiteCannotBeRestartedWhileItsDependentConsumesItsReport()
    {
        var started = Signal();
        var release = Signal();
        await using var session = Session(id => async (_, _, ct) =>
        {
            if (id.EndsWith("/consumer", StringComparison.Ordinal))
            {
                started.TrySetResult(true);
                await release.Task.WaitAsync(ct);
            }
            return new(new ParsedOutput { Content = id });
        });
        string source = Id(await session.ExecuteAsync(Spawn("source")));
        await Wait(session, source);
        string consumer = Id(await session.ExecuteAsync(Spawn("consumer", source)));
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False((await session.ExecuteAsync(Call("send_input", ("agent_id", source), ("message", "rewrite evidence")))).Ok);
        release.TrySetResult(true);
        await Wait(session, consumer);
        Assert.True((await session.ExecuteAsync(Call("send_input", ("agent_id", source), ("message", "new independent task")))).Ok);
        await Wait(session, source);
    }

    [Fact]
    public async Task FinalSynthesisWaitsForQueuedAndDependentTasks()
    {
        int rounds = 0;
        SkillLoopResult result = await SkillAgentLoop.RunAsync([new() { Role = "user", Content = "Combine all evidence." }],
            null, new SkillToolContext([]), (messages, _, _) =>
            {
                if (++rounds == 1) return Calls(Spawn("first"), Spawn("second"), Spawn("merge", "/root/first, /root/second"));
                bool collected = messages.Any(message => (message.Content ?? "").Contains("EVIDENCE:/root/merge", StringComparison.Ordinal));
                return Answer(collected ? "Complete synthesis" : "Premature answer");
            }, new()
            {
                MaxRounds = 8,
                MultiAgent = new() { Enabled = true, MaxConcurrentAgents = 1 },
                SubagentGeneratorFactory = id => (_, _, _) => Answer("EVIDENCE:" + id),
            }).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal("Complete synthesis", result.Output.Parsed.Content);
        Assert.True(rounds >= 3);
        Assert.False(result.HitRoundLimit);
        Assert.DoesNotContain(result.Invocations, invocation => !invocation.Ok);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task NestedAgentReleasesCapacityWhileWaitingOrAutomaticallyCollecting(bool explicitWait)
    {
        int parentTurns = 0;
        await using var session = Session(id =>
        {
            if (id.EndsWith("/leaf", StringComparison.Ordinal)) return (_, _, _) => Answer("LEAF_EVIDENCE");
            return (messages, _, _) =>
            {
                int turn = Interlocked.Increment(ref parentTurns);
                if (turn == 1) return Calls(Spawn("leaf"));
                if (turn == 2 && explicitWait) return Calls(Call("wait_agent", ("timeout_ms", 5000)));
                bool evidence = messages.Any(message => (message.Content ?? "").Contains("LEAF_EVIDENCE", StringComparison.Ordinal));
                return Answer(evidence ? "Parent incorporated LEAF_EVIDENCE" : "Premature parent");
            };
        }, capacity: 1);
        string parent = Id(await session.ExecuteAsync(Spawn("parent")));
        JsonElement row = Row(await Wait(session, parent), parent);
        Assert.Equal("completed", row.GetProperty("status").GetString());
        Assert.Equal("Parent incorporated LEAF_EVIDENCE", row.GetProperty("result").GetString());
        Assert.Equal(2, session.GetProgress().Count);
    }

    [Theory]
    [InlineData("explorer", "workspace-write", true)]
    [InlineData("reviewer", "workspace-write", true)]
    [InlineData("worker", "workspace-write", false)]
    [InlineData("worker", "full-access", true)]
    public async Task ExplicitPermissionsRejectRoleOrHostEscalation(string role, string permission, bool allowWorkers)
    {
        int generated = 0;
        await using var session = Session(_ => (_, _, _) =>
        {
            Interlocked.Increment(ref generated);
            return Answer("unexpected");
        }, allowWorkers: allowWorkers);
        Assert.False((await session.ExecuteAsync(Spawn("escalation", role: role, permissions: permission))).Ok);
        Assert.Equal(0, generated);
        Assert.Empty(session.GetProgress());
    }

    [Fact]
    public async Task ExplicitReadOnlyWorkerCannotGrantWritePermissionToDescendant()
    {
        var release = Signal();
        List<ToolFunction>? offered = null;
        await using var session = Session(_ => async (_, tools, ct) =>
        {
            offered = tools;
            await release.Task.WaitAsync(ct);
            return new(new ParsedOutput { Content = "read only" });
        }, allowWorkers: true);
        JsonElement parent = Json(await session.ExecuteAsync(Spawn("reader", role: "worker", permissions: "read-only")));
        string id = parent.GetProperty("agent_id").GetString()!;
        Assert.False(parent.GetProperty("mutable_tools").GetBoolean());
        Assert.False((await session.ExecuteAsync(Spawn("writer", role: "worker", permissions: "workspace-write"), id)).Ok);
        release.TrySetResult(true);
        await Wait(session);
        Assert.DoesNotContain(offered!, tool => tool.Name is "shell" or "write_file" or "apply_patch" or "skills_run");
    }

    [Fact]
    public async Task WorkerFileToolsUsePrivateInputsAndReviewerReadsExportedDependencyCopies()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-agent-loop-tests-" + Guid.NewGuid().ToString("N"));
        var manager = new SessionWorkspaceManager(root);
        SessionWorkspace parent = manager.GetOrCreate("parent");
        using var shell = new ShellRunner(new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off });
        var runner = new CodeRunnerAdapter(shell);
        Assert.True(parent.TryWriteFile("input.txt", "original\n", out string? error), error);
        Assert.True(parent.TryWriteFile("private.txt", "parent-only", out error), error);
        var context = new SkillToolContext([]) { Workspace = parent, CodeRunner = runner };
        var invocations = new ConcurrentQueue<SkillToolInvocation>();
        try
        {
            await using var session = new MultiAgentSession([], runner.DeclareTools(persists: true), context, id =>
            {
                int turn = 0;
                return (messages, tools, _) =>
                {
                    Assert.DoesNotContain(tools!, tool => tool.Name == "shell");
                    if (id.EndsWith("/producer", StringComparison.Ordinal))
                    {
                        if (++turn == 1) return Calls(Call("read_file", ("path", "input.txt")));
                        if (turn == 2)
                        {
                            Assert.Contains(messages, message => message.Role == "tool" && message.Content!.Contains("original"));
                            return Calls(
                                Call("apply_patch", ("patch", "*** Begin Patch\n*** Update File: input.txt\n@@\n-original\n+edited\n*** End Patch")),
                                Call("write_file", ("path", "output.txt"), ("content", "VERIFIED_CHILD_OUTPUT")));
                        }
                        return turn == 3 ? Answer("Produced edited input and VERIFIED_CHILD_OUTPUT.")
                            : Task.FromException<SkillTurnOutput>(new InvalidOperationException("Follow-up fixture failure"));
                    }
                    Assert.DoesNotContain(tools!, tool => tool.Name is "write_file" or "apply_patch");
                    if (++turn == 1)
                    {
                        Assert.Contains(messages, message => message.Content!.Contains("Produced edited input"));
                        return Calls(Call("read_file", ("path", "dependencies/producer/output.txt")),
                            Call("read_file", ("path", "dependencies/producer/input.txt")));
                    }
                    Assert.Contains(messages, message => message.Role == "tool" && message.Content!.Contains("VERIFIED_CHILD_OUTPUT"));
                    Assert.Contains(messages, message => message.Role == "tool" && message.Content!.Contains("edited"));
                    return Answer("Reviewed dependency files and confirmed VERIFIED_CHILD_OUTPUT.");
                };
            }, new() { Enabled = true, AllowWorkerTools = true, MaxConcurrentAgents = 1 },
                new() { OnInvocation = invocations.Enqueue });
            ToolCall producer = Spawn("producer", role: "worker", permissions: "workspace-write");
            producer.Arguments!["input_files"] = "input.txt";
            string producerId = Id(await session.ExecuteAsync(producer));
            string reviewerId = Id(await session.ExecuteAsync(Spawn("reviewer", producerId, "reviewer", "read-only")));
            SkillToolResult result = await session.ExecuteAsync(Call("wait_agent", ("timeout_ms", 10000)));
            JsonElement rows = Json(result).GetProperty("agents");
            Assert.Equal("completed", Row(rows, producerId).GetProperty("status").GetString());
            Assert.Equal("completed", Row(rows, reviewerId).GetProperty("status").GetString());
            Assert.Contains("VERIFIED_CHILD_OUTPUT", Row(rows, reviewerId).GetProperty("result").GetString());
            Assert.NotEqual(Row(rows, producerId).GetProperty("workspace_id").GetString(),
                Row(rows, reviewerId).GetProperty("workspace_id").GetString());
            Assert.DoesNotContain(invocations, invocation => !invocation.Ok);
            Assert.Equal(5, invocations.Count);
            Assert.True(parent.TryReadFile("input.txt", out string original, out error), error);
            Assert.Equal("original\n", original);
            Assert.True(parent.TryReadFile("private.txt", out string secret, out error), error);
            Assert.Equal("parent-only", secret);
            Assert.False(File.Exists(Path.Combine(parent.WorkDirectory, "output.txt")));
            JsonElement exported = Row(rows, producerId).GetProperty("files");
            Assert.Equal(2, exported.GetArrayLength());
            string output = Assert.Single(exported.EnumerateArray(), file => file.GetProperty("path").GetString()!
                .EndsWith("/output.txt", StringComparison.Ordinal)).GetProperty("path").GetString()!;
            Assert.True(parent.TryReadFile(output, out string produced, out error), error);
            Assert.Equal("VERIFIED_CHILD_OUTPUT", produced);
            string edited = Assert.Single(exported.EnumerateArray(), file => file.GetProperty("path").GetString()!
                .EndsWith("/input.txt", StringComparison.Ordinal)).GetProperty("path").GetString()!;
            Assert.True(parent.TryReadFile(edited, out string changed, out error), error);
            Assert.Equal("edited\n", changed);
            Assert.Empty(result.Files!); // Workspace paths are not public download URLs.
            Assert.True((await session.ExecuteAsync(Call("send_input", ("agent_id", producerId), ("message", "Attempt another task")))).Ok);
            SkillToolResult failedFollowUp = await session.ExecuteAsync(Call("wait_agent", ("agent_id", producerId), ("timeout_ms", 10000)));
            JsonElement failed = Row(Json(failedFollowUp).GetProperty("agents"), producerId);
            Assert.Equal("failed", failed.GetProperty("status").GetString());
            Assert.Empty(failedFollowUp.Files!);
            Assert.Equal(0, failed.GetProperty("files").GetArrayLength());
            Assert.True(parent.TryReadFile(output, out produced, out error), error);
            Assert.Equal("VERIFIED_CHILD_OUTPUT", produced);
        }
        finally
        {
            manager.Release("parent");
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
