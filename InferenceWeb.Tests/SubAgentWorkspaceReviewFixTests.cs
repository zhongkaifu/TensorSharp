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
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.CodeExec;

namespace InferenceWeb.Tests;

/// <summary>
/// Regressions for the workspace half of the sub-agent review fixes, against a real
/// <see cref="SessionWorkspaceManager"/> in a temp directory:
/// <list type="bullet">
/// <item>a sub-agent's code or script call that waited behind another agent's command
/// never starts once its agent was stopped (the stop token is checked with the
/// workspace's execution lock held);</item>
/// <item>a running sub-agent holds the workspace open, so releasing the conversation
/// does not delete the directory under it;</item>
/// <item>a finished agent's lane is retired — its shell session, read ledger and state
/// directory — while the conversation's own state stays.</item>
/// </list>
/// </summary>
public class SubAgentWorkspaceReviewFixTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public SubAgentWorkspaceReviewFixTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-agent-ws-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private SessionWorkspace Workspace(string id = "agents") => _workspaces.GetOrCreate(id);

    private CodeExecOptions ShellOptions() => new()
    {
        Enabled = true,
        Sandbox = SkillSandboxMode.Off,
        Timeout = TimeSpan.FromSeconds(30),
        ScratchDirectory = _base,
    };

    private static bool HavePosixShell =>
        ShellProgram.TryResolve(null, out ShellProgram? shell, out _) && shell is { Kind: ShellKind.Posix };

    private static SkillToolContext ContextWith(SessionWorkspace workspace) =>
        new(Array.Empty<Skill>()) { Workspace = workspace };

    /// <summary>A condition poll with a ceiling; never a fixed sleep.</summary>
    private static async Task PollUntil(Func<bool> condition, string what, int timeoutSeconds = 10)
    {
        var stopwatch = Stopwatch.StartNew();
        while (!condition())
        {
            if (stopwatch.Elapsed > TimeSpan.FromSeconds(timeoutSeconds))
                throw new TimeoutException(what + " did not happen within " + timeoutSeconds + " s");
            await Task.Delay(5).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Whether <paramref name="task"/> finishes within a short bound: used only to show a
    /// call is BLOCKED on the execution lock, so a false result is the expected one.
    /// </summary>
    private static async Task<bool> Finishes(Task task)
    {
        await Task.WhenAny(task, Task.Delay(TimeSpan.FromMilliseconds(250))).ConfigureAwait(false);
        return task.IsCompleted;
    }

    /// <summary>The lane directory a runtime created for <paramref name="agentId"/> (its lane id carries a per-turn tag).</summary>
    private static string LaneName(SessionWorkspace owner, string agentId) =>
        Directory.GetDirectories(Path.Combine(owner.StateDirectory, SessionWorkspace.LanesDirectoryName))
            .Select(Path.GetFileName)
            .Single(name => name!.StartsWith(agentId + "-", StringComparison.Ordinal))!;

    // ---- 7. a stopped agent's queued command never starts ------------------------------------

    [Fact]
    public async Task CodeRunner_ACallStoppedWhileItWaitedForTheWorkspace_NeverReachesTheRunner()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");
        var inner = new CountingRunner();
        using var stop = new CancellationTokenSource();
        var runner = new SubAgentCodeRunner(inner, stop.Token);

        SkillToolResult result;
        using (var otherAgent = new ExecutionHolder(owner))
        {
            Task<SkillToolResult> call = Task.Run(() =>
                runner.Execute(Steps.Tool(SkillToolNames.Shell, ("command", "echo hi")), workspace: lane));
            Assert.False(await Finishes(call), "the call did not wait for the workspace lock");

            stop.Cancel();
            otherAgent.Release();
            result = await call.Within();
        }

        Assert.False(result.Ok);
        Assert.Equal("Error: " + SubAgentCodeRunner.StoppedMessage, result.Content);
        Assert.Equal(0, inner.Executions);
    }

    [Fact]
    public async Task CodeRunner_ALiveCallThatWaitedForTheWorkspace_RunsOnceItIsFree()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");
        var inner = new CountingRunner();
        using var stop = new CancellationTokenSource();
        var runner = new SubAgentCodeRunner(inner, stop.Token);

        SkillToolResult result;
        using (var otherAgent = new ExecutionHolder(owner))
        {
            Task<SkillToolResult> call = Task.Run(() =>
                runner.Execute(Steps.Tool(SkillToolNames.Shell, ("command", "echo hi")), workspace: lane));
            Assert.False(await Finishes(call), "the call did not wait for the workspace lock");
            otherAgent.Release();
            result = await call.Within();
        }

        Assert.True(result.Ok, result.Content);
        Assert.Equal("ran", result.Content);
        Assert.Equal(1, inner.Executions);
        Assert.Same(lane, inner.LastWorkspace);
    }

    [Fact]
    public async Task CodeRunner_AnInstallStoppedWhileItWaited_NeverReachesTheRunner()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");
        var inner = new CountingRunner();
        using var stop = new CancellationTokenSource();
        var runner = new SubAgentCodeRunner(inner, stop.Token);

        string? error;
        using (var otherAgent = new ExecutionHolder(owner))
        {
            Task<string?> install = Task.Run(() => runner.InstallPackages("python", new[] { "rich" }, lane));
            Assert.False(await Finishes(install), "the install did not wait for the workspace lock");
            stop.Cancel();
            otherAgent.Release();
            error = await install.Within();
        }

        Assert.Equal(SubAgentCodeRunner.StoppedMessage, error);
        Assert.Equal(0, inner.Installs);
    }

    private const string ScriptStopped = "Error: this agent was stopped before the script could start, so it did not run.";

    private Skill MakeSkill()
    {
        string root = Path.Combine(_base, "skill-" + Guid.NewGuid().ToString("N"));
        string dir = Path.Combine(root, "tester");
        Directory.CreateDirectory(Path.Combine(dir, "scripts"));
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            "---\nname: tester\ndescription: stop-token test skill\n---\nBody.");
        File.WriteAllText(Path.Combine(dir, "scripts", "tool.py"), "print('script ran')\n");
        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { root } }).Skills.Single();
    }

    [Fact]
    public async Task ScriptRunner_AScriptStoppedWhileItWaitedForTheWorkspace_DoesNotStart()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");
        Skill skill = MakeSkill();
        var parentRunner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Off,
            Workspace = owner,
        });
        var installer = new CountingRunner();
        using var stop = new CancellationTokenSource();
        SkillScriptRunner agentRunner = parentRunner.ForAgent(lane, installer, stop.Token);

        SkillToolResult result;
        using (var otherAgent = new ExecutionHolder(owner))
        {
            Task<SkillToolResult> run = Task.Run(() =>
                agentRunner.Run(skill, "scripts/tool.py", Array.Empty<string>()));
            Assert.False(await Finishes(run), "the script did not wait for the workspace lock");
            stop.Cancel();
            otherAgent.Release();
            result = await run.Within();
        }

        Assert.False(result.Ok);
        Assert.Equal(ScriptStopped, result.Content);
        Assert.DoesNotContain("script ran", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ScriptRunner_ALiveScriptThatWaitedForTheWorkspace_GoesOnToRun()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");
        Skill skill = MakeSkill();
        var parentRunner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Off,
            Workspace = owner,
        });
        using var stop = new CancellationTokenSource();
        SkillScriptRunner agentRunner = parentRunner.ForAgent(lane, new CountingRunner(), stop.Token);

        SkillToolResult missing;
        using (var otherAgent = new ExecutionHolder(owner))
        {
            // A path that does not exist: it gets past the stop check to the path guard,
            // which is all this asserts on a host with or without an interpreter.
            Task<SkillToolResult> run = Task.Run(() =>
                agentRunner.Run(skill, "scripts/missing.py", Array.Empty<string>()));
            Assert.False(await Finishes(run), "the script did not wait for the workspace lock");
            otherAgent.Release();
            missing = await run.Within();
        }

        Assert.False(missing.Ok);
        Assert.NotEqual(ScriptStopped, missing.Content);
        Assert.Contains("missing.py", missing.Content, StringComparison.Ordinal);

        if (!TensorSharp.AgentHost.CodeExec.CodeEnvironment.TryResolveInterpreter(
                TensorSharp.AgentHost.CodeExec.CodeLanguage.Python, out _, out _))
        {
            return;
        }
        SkillToolResult ran = agentRunner.Run(skill, "scripts/tool.py", Array.Empty<string>());
        Assert.True(ran.Ok, ran.Content);
        Assert.Contains("script ran", ran.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ForSubAgent_WrapsTheCodeAndScriptRunnersWithTheAgentsStopToken_InItsOwnLane()
    {
        SessionWorkspace owner = Workspace();
        var inner = new CountingRunner();
        var scripts = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off, Workspace = owner });
        var parent = new SkillToolContext(Array.Empty<Skill>()) { Workspace = owner, CodeRunner = inner, ScriptRunner = scripts };
        using var h = new SubAgentHarness();
        using var stop = new CancellationTokenSource();

        SkillToolContext agent = parent.ForSubAgent(h.Runtime.Root, "agent_7", stop.Token);

        Assert.Same(owner.ForAgent("agent_7"), agent.Workspace);
        Assert.IsType<SubAgentCodeRunner>(agent.CodeRunner);
        SkillScriptRunner agentScripts = Assert.IsType<SkillScriptRunner>(agent.ScriptRunner);
        Assert.NotSame(scripts, agentScripts);

        Assert.True(agent.CodeRunner!.Execute(Steps.Tool(SkillToolNames.Shell), workspace: agent.Workspace).Ok);
        Assert.Equal(1, inner.Executions);

        stop.Cancel();
        Assert.Equal("Error: " + SubAgentCodeRunner.StoppedMessage,
            agent.CodeRunner.Execute(Steps.Tool(SkillToolNames.Shell), workspace: agent.Workspace).Content);
        Assert.Equal(ScriptStopped, agentScripts.Run(MakeSkill(), "scripts/tool.py", Array.Empty<string>()).Content);
        Assert.Equal(1, inner.Executions);
    }

    // ---- 8. a running sub-agent holds the workspace open --------------------------------------

    [Fact]
    public async Task Release_WhileASubAgentRuns_KeepsTheWorkspace_UntilTheAgentFinishes()
    {
        SessionWorkspace owner = Workspace("leased");
        using var h = new SubAgentHarness(context: ContextWith(owner));
        var gate = Steps.Gate();
        ScriptedAgent child = h.Script("agent_1", Steps.After(gate.Task, Steps.Answer("done")));
        await h.Spawn("work").Within();
        await child.Entered(1).Within();

        _workspaces.Release("leased");

        Assert.True(Directory.Exists(owner.Root), "the workspace was deleted under a running sub-agent");
        Assert.True(Directory.Exists(owner.WorkDirectory));

        // Nothing new starts in a conversation that has been let go — and the spawn says
        // so itself, rather than "started" followed by a failure at the next delivery.
        SkillToolResult late = await h.Spawn("late work").Within();
        Assert.False(late.Ok);
        Assert.Equal("Error: agent_2 could not start: this conversation's workspace was released before the agent could start.",
            late.Content);
        Assert.Equal(SubAgentStatus.Errored, h.Snapshot("agent_2").Status);
        Assert.True(h.Runtime.Root.TakePendingDeliveries().IsEmpty);
        Assert.True(Directory.Exists(owner.Root));

        gate.SetResult();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        Assert.Equal("done", h.Snapshot("agent_1").Result);
        await PollUntil(() => !Directory.Exists(owner.Root), "deleting the released workspace after the agent finished");
    }

    [Fact]
    public async Task Release_WhileASubAgentRuns_DeletesTheWorkspaceOnceTheRuntimeStopsIt()
    {
        SessionWorkspace owner = Workspace("disposed");
        using var h = new SubAgentHarness(context: ContextWith(owner));   // a second dispose is a no-op
        ScriptedAgent child = h.Script("agent_1", Steps.Block());
        await h.Spawn("work").Within();
        await child.Entered(1).Within();

        _workspaces.Release("disposed");
        Assert.True(Directory.Exists(owner.Root), "the workspace was deleted under a running sub-agent");

        await h.Runtime.DisposeAsync().AsTask().Within();
        await child.Cancelled.Within();
        await PollUntil(() => !Directory.Exists(owner.Root), "deleting the released workspace after the runtime stopped its agent");
    }

    // ---- 9. lanes of finished agents are retired -----------------------------------------------

    [Fact]
    public void RetireLane_LetsGoOfTheLanesOwnState_AndOnlyThat()
    {
        SessionWorkspace owner = Workspace("retire");
        SessionWorkspace lane = owner.ForAgent("agent_1");
        SessionWorkspace other = owner.ForAgent("agent_2");
        string path = Path.Combine(owner.WorkDirectory, "notes.txt");
        const string content = "one\ntwo\n";
        File.WriteAllText(path, content);
        lane.Reads.Record(path, content, 1, int.MaxValue, complete: true);
        other.Reads.Record(path, content, 1, int.MaxValue, complete: true);
        owner.Reads.Record(path, content, 1, int.MaxValue, complete: true);
        string laneDirectory = Path.Combine(owner.StateDirectory, SessionWorkspace.LanesDirectoryName, "agent_1");
        Assert.True(Directory.Exists(laneDirectory));

        var conversationCleanup = new CountingCleanup();
        var registeredThroughTheLane = new CountingCleanup();
        var laneOnly = new CountingCleanup();
        var otherLaneOnly = new CountingCleanup();
        owner.RegisterCleanup(conversationCleanup);
        lane.RegisterCleanup(registeredThroughTheLane);   // a lane's ordinary cleanup is the conversation's
        lane.RegisterLaneCleanup(laneOnly);
        other.RegisterLaneCleanup(otherLaneOnly);

        owner.RetireLane("agent_1");

        // The lane's own state is gone: its directory, its ledger and its lane-only cleanups.
        Assert.False(Directory.Exists(laneDirectory));
        Assert.Equal(0, lane.Reads.Count);
        Assert.Equal(1, laneOnly.DisposeCount);

        // Nothing else is: the conversation's cleanups, the other lane, the owner's ledger
        // and the shared files.
        Assert.Equal(0, conversationCleanup.DisposeCount);
        Assert.Equal(0, registeredThroughTheLane.DisposeCount);
        Assert.Equal(0, otherLaneOnly.DisposeCount);
        Assert.Same(other, owner.ForAgent("agent_2"));
        Assert.True(Directory.Exists(other.ShellStateDirectory));
        Assert.Equal(ReadFreshness.Fresh, other.Reads.Check(path, content).Freshness);
        Assert.Equal(ReadFreshness.Fresh, owner.Reads.Check(path, content).Freshness);
        Assert.Equal(content, File.ReadAllText(path));

        // Asked for again, the lane is a fresh one; asked through a lane, the owner retires.
        SessionWorkspace again = owner.ForAgent("agent_1");
        Assert.NotSame(lane, again);
        Assert.Equal(0, again.Reads.Count);
        Assert.Equal(ReadFreshness.Unread, again.Reads.Check(path, content).Freshness);
        Assert.True(Directory.Exists(again.ShellStateDirectory));

        // A cleanup registered on the retired lane afterwards is run at once, not kept.
        var late = new CountingCleanup();
        lane.RegisterLaneCleanup(late);
        Assert.Equal(1, late.DisposeCount);

        // Asked through a lane, the owner retires; unknown and repeated retirements change nothing.
        other.RetireLane("agent_2");
        Assert.Equal(1, otherLaneOnly.DisposeCount);
        other.RetireLane("agent_2");
        owner.RetireLane("agent_9");
        Assert.Equal(1, otherLaneOnly.DisposeCount);
        Assert.Same(again, owner.ForAgent("agent_1"));

        // A lane never retired lets go with the conversation; the conversation's own end
        // runs everything that remained, once.
        var neverRetired = new CountingCleanup();
        owner.ForAgent("agent_3").RegisterLaneCleanup(neverRetired);
        _workspaces.Release("retire");
        Assert.Equal(1, conversationCleanup.DisposeCount);
        Assert.Equal(1, registeredThroughTheLane.DisposeCount);
        Assert.Equal(1, neverRetired.DisposeCount);
        Assert.Equal(1, laneOnly.DisposeCount);
        Assert.Equal(1, otherLaneOnly.DisposeCount);
        Assert.False(Directory.Exists(owner.Root));
    }

    [Fact]
    public void RetireLane_ForgetsTheLanesShellSession_AndKeepsTheOwnersShellAndJobTable()
    {
        if (!HavePosixShell) return;

        SessionWorkspace owner = Workspace("retire-shell");
        SessionWorkspace lane = owner.ForAgent("agent_1");
        using var runner = new ShellRunner(ShellOptions());
        const string where = "echo \"[cwd=$(basename \"$PWD\")][probe=$LANE_PROBE]\"";

        CodeExecResult moved = runner.Run(new ShellRequest("mkdir sub && cd sub && export LANE_PROBE=a"), lane);
        Assert.True(moved.Ok, moved.Content);
        CodeExecResult ownerMoved = runner.Run(new ShellRequest("mkdir own && cd own"), owner);
        Assert.True(ownerMoved.Ok, ownerMoved.Content);
        CodeExecResult job = runner.Run(new ShellRequest("true") { Background = true }, lane);
        Assert.True(job.Ok, job.Content);
        Assert.Contains("job-1", job.Content, StringComparison.Ordinal);
        ShellSession laneShell = runner.SessionFor(lane);
        ShellSession ownerShell = runner.SessionFor(owner);
        Assert.Equal("sub", laneShell.CurrentDirectoryLabel);

        owner.RetireLane("agent_1");

        // The lane asked for again is a new shell, back in the work directory with
        // nothing exported — and its wrapper scripts count from 1 again.
        SessionWorkspace again = owner.ForAgent("agent_1");
        Assert.NotSame(laneShell, runner.SessionFor(again));
        CodeExecResult fresh = runner.Run(new ShellRequest(where), again);
        Assert.True(fresh.Ok, fresh.Content);
        Assert.Contains("[cwd=work][probe=]", fresh.Content, StringComparison.Ordinal);
        string script = Assert.Single(Directory.GetFiles(again.ShellScriptDirectory, "cmd-*"));
        Assert.StartsWith("cmd-1.", Path.GetFileName(script), StringComparison.Ordinal);

        // The owner's shell is the same one, still where it left itself.
        Assert.Same(ownerShell, runner.SessionFor(owner));
        CodeExecResult ownerWhere = runner.Run(new ShellRequest(where), owner);
        Assert.True(ownerWhere.Ok, ownerWhere.Content);
        Assert.Contains("[cwd=own][probe=]", ownerWhere.Content, StringComparison.Ordinal);

        // The job table is the conversation's: its numbering does not restart.
        CodeExecResult next = runner.Run(new ShellRequest("true") { Background = true }, owner);
        Assert.True(next.Ok, next.Content);
        Assert.Contains("job-2", next.Content, StringComparison.Ordinal);

        _workspaces.Release("retire-shell");
    }

    [Fact]
    public async Task DisposeAsync_RetiresTheLanesOfFinishedAgents()
    {
        SessionWorkspace owner = Workspace("turn-end");
        using var h = new SubAgentHarness(context: ContextWith(owner));   // a second dispose is a no-op
        h.Script("agent_1", Steps.Answer("done"));
        await h.Spawn("work").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);

        string name = LaneName(owner, "agent_1");
        string directory = Path.Combine(owner.StateDirectory, SessionWorkspace.LanesDirectoryName, name);
        SessionWorkspace lane = owner.ForAgent(name);
        string path = Path.Combine(owner.WorkDirectory, "seen.txt");
        File.WriteAllText(path, "x\n");
        lane.Reads.Record(path, "x\n", 1, int.MaxValue, complete: true);

        await h.Runtime.DisposeAsync().AsTask().Within();

        Assert.False(Directory.Exists(directory), "the finished agent's lane directory was kept");
        Assert.Equal(0, lane.Reads.Count);
        Assert.NotSame(lane, owner.ForAgent(name));
        Assert.True(Directory.Exists(owner.WorkDirectory));
    }

    /// <summary>
    /// SLOW (about 5 s): the one test that lets a worker outlive the disposal grace period,
    /// which is what decides that its lane is kept.
    /// </summary>
    [Fact]
    public async Task DisposeAsync_KeepsTheLaneOfAWorkerStillRunningPastTheGracePeriod_TakesFiveSeconds()
    {
        SessionWorkspace owner = Workspace("turn-end-slow");
        var h = new SubAgentHarness(context: ContextWith(owner));
        var stuck = Steps.Gate();
        h.Script("agent_1", Steps.Answer("done"));
        // Ignores its cancellation token, as a generation stuck in native code would.
        ScriptedAgent slow = h.Script("agent_2", async _ =>
        {
            await stuck.Task.ConfigureAwait(false);
            return Steps.Turn("late");
        });
        await h.Spawn("quick").Within();
        await h.Spawn("slow").Within();
        await h.WaitForStatusAsync("agent_1", SubAgentStatus.Completed);
        await slow.Entered(1).Within();

        string finishedName = LaneName(owner, "agent_1");
        string runningName = LaneName(owner, "agent_2");
        SessionWorkspace finished = owner.ForAgent(finishedName);
        SessionWorkspace running = owner.ForAgent(runningName);
        string path = Path.Combine(owner.WorkDirectory, "seen.txt");
        File.WriteAllText(path, "x\n");
        running.Reads.Record(path, "x\n", 1, int.MaxValue, complete: true);

        try
        {
            await h.Runtime.DisposeAsync().AsTask().Within(30);

            string lanes = Path.Combine(owner.StateDirectory, SessionWorkspace.LanesDirectoryName);
            Assert.False(Directory.Exists(Path.Combine(lanes, finishedName)));
            Assert.NotSame(finished, owner.ForAgent(finishedName));

            // Still running: its lane is left for the conversation's release to take.
            Assert.True(Directory.Exists(Path.Combine(lanes, runningName)));
            Assert.Same(running, owner.ForAgent(runningName));
            Assert.Equal(ReadFreshness.Fresh, running.Reads.Check(path, "x\n").Freshness);
        }
        finally
        {
            stuck.TrySetResult();
            await PollUntil(
                () => h.Runtime.Snapshot().Single(s => s.Id == "agent_2").Status
                    is not (SubAgentStatus.Starting or SubAgentStatus.Running),
                "the stuck agent's worker ending");
            h.Dispose();
        }
    }

    // ---- helpers ---------------------------------------------------------------------------

    /// <summary>
    /// Another agent's command, holding the workspace's execution lock on a thread of its
    /// own (the lock is a monitor, so it must be released on the thread that took it).
    /// </summary>
    private sealed class ExecutionHolder : IDisposable
    {
        private readonly ManualResetEventSlim _held = new();
        private readonly ManualResetEventSlim _release = new();
        private readonly Thread _thread;

        public ExecutionHolder(SessionWorkspace workspace)
        {
            _thread = new Thread(() =>
            {
                using (workspace.EnterExecution())
                {
                    _held.Set();
                    _release.Wait();
                }
            })
            { IsBackground = true, Name = "other agent's command" };
            _thread.Start();
            Assert.True(_held.Wait(TimeSpan.FromSeconds(10)), "the holder never took the execution lock");
        }

        public void Release()
        {
            _release.Set();
            Assert.True(_thread.Join(TimeSpan.FromSeconds(10)), "the holder never let go of the execution lock");
        }

        public void Dispose()
        {
            _release.Set();
            _thread.Join(TimeSpan.FromSeconds(10));
            _held.Dispose();
            _release.Dispose();
        }
    }

    /// <summary>Counts what reaches it; runs nothing.</summary>
    private sealed class CountingRunner : ICodeRunner
    {
        private int _executions;
        private int _installs;

        public int Executions => Volatile.Read(ref _executions);
        public int Installs => Volatile.Read(ref _installs);
        public SessionWorkspace? LastWorkspace { get; private set; }

        public bool CanRun => true;
        public string? UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = SkillToolNames.Shell };
        public bool CanInstallPackages => true;

        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null)
        {
            Interlocked.Increment(ref _executions);
            LastWorkspace = workspace;
            return SkillToolResult.Success("ran");
        }

        public string? InstallPackages(string language, IReadOnlyList<string> packages,
            SessionWorkspace workspace, Action<string>? onOutput = null)
        {
            Interlocked.Increment(ref _installs);
            return null;
        }
    }

    private sealed class CountingCleanup : IDisposable
    {
        private int _disposeCount;

        public int DisposeCount => Volatile.Read(ref _disposeCount);

        public void Dispose() => Interlocked.Increment(ref _disposeCount);
    }
}
