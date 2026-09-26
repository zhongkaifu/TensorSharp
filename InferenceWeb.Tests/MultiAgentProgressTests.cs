using System.Text.Json;
using TensorSharp.AgentHost.Agents;
using TensorSharp.Server.ResponseSerializers;

namespace InferenceWeb.Tests;

/// <summary>Live progress must remain independent of the model's result-consumption protocol.</summary>
public sealed class MultiAgentProgressTests
{
    private static ToolCall Call(string name, params (string Key, object Value)[] arguments) => new()
    {
        Name = name,
        Arguments = arguments.ToDictionary(p => p.Key, p => p.Value),
    };

    private static TaskCompletionSource<bool> Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static MultiAgentSession Session(Func<string, SkillTurnGenerator> factory,
        SkillToolContext? context = null, IReadOnlyList<ToolFunction>? tools = null) =>
        new([], tools ?? SkillTools.BuiltIn(), context ?? new SkillToolContext([]), factory,
            new() { Enabled = true, AllowWorkerTools = true, MaxResultCharacters = 256 });

    private static async Task Spawn(MultiAgentSession session, string name, string role = "explorer",
        string parent = MultiAgentSession.RootId) =>
        Assert.True((await session.ExecuteAsync(Call(MultiAgentTools.Spawn,
            ("task_name", name), ("task", "Inspect " + name), ("agent_type", role)), parent)).Ok);

    private static async Task<MultiAgentProgress> TerminalProgress(MultiAgentSession session)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        while (true)
        {
            MultiAgentProgress progress = Assert.Single(session.GetProgress());
            if (progress.Status is not ("running" or "cancelling")) return progress;
            await Task.Delay(10, timeout.Token);
        }
    }

    [Fact]
    public async Task SnapshotsAreStableAndDoNotConsumeBoundedCompletedReports()
    {
        var release = Signal();
        await using var session = Session(_ => async (_, _, ct) =>
        {
            await release.Task.WaitAsync(ct);
            return new(new ParsedOutput { Content = new string('x', 1000), Thinking = "private reasoning" });
        });
        await Spawn(session, "review", "reviewer");
        MultiAgentProgress running = Assert.Single(session.GetProgress());
        Assert.Equal("/root/review", running.AgentId);
        Assert.Equal("/root", running.ParentId);
        Assert.Equal("Inspect review", running.Task);
        Assert.Equal("reviewer", running.AgentType);
        Assert.Equal("running", running.Status);
        Assert.Null(running.Result);

        release.TrySetResult(true);
        MultiAgentProgress completed = await TerminalProgress(session);
        Assert.Equal("completed", completed.Status);
        Assert.InRange(completed.Result!.Length, 1, 256);
        Assert.Contains("truncated", completed.Result, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("running", running.Status);
        Assert.Null(running.Result);
        Assert.True(session.HasPendingResults());
        string reports = await session.CollectResultsAsync();
        Assert.Contains("/root/review", reports);
        Assert.False(session.HasPendingResults());
        Assert.Equal(completed, Assert.Single(session.GetProgress()));
    }

    [Fact]
    public async Task SnapshotIncludesDescendantsWithoutLeakingAnotherRequestsAgents()
    {
        await using var session = Session(_ => async (_, _, ct) =>
        {
            await Task.Delay(Timeout.Infinite, ct);
            return new(new ParsedOutput());
        });
        await using var otherSession = Session(_ => (_, _, _) => Task.FromResult(new SkillTurnOutput(new ParsedOutput())));
        await Spawn(session, "parent");
        await Spawn(session, "child", parent: "/root/parent");
        IReadOnlyList<MultiAgentProgress> progress = session.GetProgress();
        Assert.Equal(2, progress.Count);
        Assert.Contains(progress, p => p.AgentId == "/root/parent/child" && p.ParentId == "/root/parent");
        Assert.Empty(otherSession.GetProgress());
        MultiAgentProgress parent = Assert.Single(progress, p => p.AgentId == "/root/parent");
        Assert.Equal(MultiAgentTools.Spawn, parent.Tool);
        Assert.Equal("completed", parent.ToolStatus);

        using var cancelWait = new CancellationTokenSource();
        Task<SkillToolResult> waiting = session.ExecuteAsync(Call(MultiAgentTools.Wait,
            ("agent_id", "/root/parent/child"), ("timeout_ms", 60000)), "/root/parent", cancelWait.Token);
        parent = Assert.Single(session.GetProgress(), p => p.AgentId == "/root/parent");
        Assert.Equal(MultiAgentTools.Wait, parent.Tool);
        Assert.Equal("running", parent.ToolStatus);
        cancelWait.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting);
        parent = Assert.Single(session.GetProgress(), p => p.AgentId == "/root/parent");
        Assert.Equal("interrupted", parent.ToolStatus);
        Assert.Equal("running", parent.Status);
    }

    [Fact]
    public async Task RunningHostToolAndItsOutcomeAppearInSnapshots()
    {
        var runner = new BlockingRunner();
        int round = 0;
        await using var session = Session(_ => (_, _, _) => Task.FromResult(new SkillTurnOutput(
            Interlocked.Increment(ref round) == 1
                ? new ParsedOutput { ToolCalls = [Call("shell", ("command", "fixture"))] }
                : new ParsedOutput { Content = "Tool inspected." })),
            new SkillToolContext([]) { CodeRunner = runner }, [runner.Declare()]);
        await Spawn(session, "worker", "worker");
        try
        {
            await runner.Started.Task.WaitAsync(TimeSpan.FromSeconds(5));
            MultiAgentProgress running = Assert.Single(session.GetProgress());
            Assert.Equal("shell", running.Tool);
            Assert.Equal("running", running.ToolStatus);
        }
        finally { runner.Release.TrySetResult(true); }
        MultiAgentProgress completed = await TerminalProgress(session);
        Assert.Equal("shell", completed.Tool);
        Assert.Equal("completed", completed.ToolStatus);
        Assert.Equal("fixture.txt", completed.Detail);
    }

    [Fact]
    public async Task FailureAndRestartClearThePreviousReportAndActivity()
    {
        var restartStarted = Signal();
        int round = 0;
        await using var session = Session(_ => async (_, _, ct) =>
        {
            if (Interlocked.Increment(ref round) == 1) throw new InvalidOperationException("fixture failed");
            restartStarted.TrySetResult(true);
            await Task.Delay(Timeout.Infinite, ct);
            return new(new ParsedOutput());
        });
        await Spawn(session, "retry");
        MultiAgentProgress failed = await TerminalProgress(session);
        Assert.Equal("failed", failed.Status);
        Assert.Equal("fixture failed", failed.Error);
        Assert.True((await session.ExecuteAsync(Call(MultiAgentTools.Send,
            ("agent_id", "/root/retry"), ("message", "Retry the inspection")))).Ok);
        await restartStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        MultiAgentProgress restarted = Assert.Single(session.GetProgress());
        Assert.Equal("running", restarted.Status);
        Assert.Equal("Retry the inspection", restarted.Task);
        Assert.Null(restarted.Result);
        Assert.Null(restarted.Error);
        Assert.Null(restarted.Tool);
    }

    [Theory]
    [InlineData("str_replace", "edit_file")]
    [InlineData("apply-patch", "apply_patch")]
    [InlineData("create_file", "write_file")]
    [InlineData("read", "read_file")]
    [InlineData("shell", "shell")]
    [InlineData("skills_run", "skills_run")]
    [InlineData("some_client_tool", "some_client_tool")]
    public void ToolProgressNamesTheToolThatRuns_NotTheAliasTheModelWrote(string called, string reported)
    {
        // The pages label progress by tool name; an accepted alias used to show up raw
        // ("Running str replace") in the live status and the kept trace line.
        object frame = WebUiSseEvents.ToolProgress("running", called, null, 1.0, null,
            [new("/root/a", "/root", "task", "worker", "running", called, "running", null, null, null)]);
        using JsonDocument json = JsonDocument.Parse(JsonSerializer.Serialize(frame));

        Assert.Equal(reported, json.RootElement.GetProperty("tool").GetString());
        Assert.Equal(reported, Assert.Single(json.RootElement.GetProperty("agents").EnumerateArray())
            .GetProperty("tool").GetString());
    }

    [Fact]
    public void ToolProgressSerializesAgentDetailsWithWebUiFieldNames()
    {
        object frame = WebUiSseEvents.ToolProgress("running", MultiAgentTools.Wait, null, 2.5, null,
            [new("/root/review", "/root", "Inspect <scope>", "reviewer", "running", "read_file",
                "completed", "module.cs", null, null)]);
        using JsonDocument json = JsonDocument.Parse(JsonSerializer.Serialize(frame));
        JsonElement payload = json.RootElement;
        Assert.Equal("running", payload.GetProperty("tool_progress").GetString());
        Assert.Equal("wait_agent", payload.GetProperty("tool").GetString());
        Assert.Equal(2.5, payload.GetProperty("seconds").GetDouble());
        JsonElement agent = Assert.Single(payload.GetProperty("agents").EnumerateArray());
        Assert.Equal("/root/review", agent.GetProperty("agent_id").GetString());
        Assert.Equal("/root", agent.GetProperty("parent_id").GetString());
        Assert.Equal("Inspect <scope>", agent.GetProperty("task").GetString());
        Assert.Equal("reviewer", agent.GetProperty("agent_type").GetString());
        Assert.Equal("read_file", agent.GetProperty("tool").GetString());
        Assert.Equal("completed", agent.GetProperty("tool_status").GetString());
        Assert.Equal("module.cs", agent.GetProperty("detail").GetString());
        Assert.Equal(JsonValueKind.Null, agent.GetProperty("result").ValueKind);
        Assert.Equal(JsonValueKind.Null, agent.GetProperty("error").ValueKind);
    }

    private sealed class BlockingRunner : ICodeRunner
    {
        public TaskCompletionSource<bool> Started { get; } = Signal();
        public TaskCompletionSource<bool> Release { get; } = Signal();
        public bool CanRun => true;
        public string? UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = "shell" };
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null)
        {
            Started.TrySetResult(true);
            Release.Task.WaitAsync(TimeSpan.FromSeconds(5)).GetAwaiter().GetResult();
            return new(true, "fixture output", null, "fixture.txt");
        }
    }
}
