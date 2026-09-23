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
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins workspace LANES: several agents — a parent and the sub-agents it spawned —
/// working in one conversation's workspace at the same time.
///
/// <para>
/// They share the files, because that is what a helper spawned into the same
/// conversation is for. Before lanes they also shared everything else a workspace
/// carried: the shell's saved directory and exports lived in one <c>state/shell</c>, the
/// runner kept one shell session per root, and one read ledger hung off the workspace.
/// So a sub-agent's <c>cd sub</c> moved the parent's next command into <c>sub</c>
/// without either of them being told, and a file the sub-agent had read counted as read
/// by the parent — an edit gate satisfied by a context window that was not the one
/// making the edit.
/// </para>
/// <para>
/// What is asserted is the split: files, packages, the execution gate and the lifetime
/// are the owner's; the shell's memory and the read ledger are each lane's own. The
/// end-to-end shell test runs a real POSIX shell with the sandbox off (confinement has
/// its own tests), and gates out on a host without one; the same property is also
/// pinned hermetically through the fake backend, which needs no shell at all.
/// </para>
/// </summary>
public class SessionWorkspaceLaneTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public SessionWorkspaceLaneTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-lanes-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private SessionWorkspace Workspace(string id = "lanes") => _workspaces.GetOrCreate(id);

    /// <summary>Enabled, unsandboxed, and rooted in this test's own temp directory.</summary>
    private CodeExecOptions Options(SkillSandboxMode sandbox = SkillSandboxMode.Off) => new()
    {
        Enabled = true,
        Sandbox = sandbox,
        Timeout = TimeSpan.FromSeconds(30),
        ScratchDirectory = _base,
    };

    private static bool HavePosixShell =>
        ShellProgram.TryResolve(null, out ShellProgram? shell, out _) && shell is { Kind: ShellKind.Posix };

    private static readonly SkillSandboxCapabilities Honest = new(
        ConfinesWrites: true, ConfinesNetwork: true, ConfinesHomeReads: true, BoundsProcessTree: true);

    // ---- what is shared ---------------------------------------------------------------

    [Fact]
    public void ALane_SharesItsOwnersFiles_InBothDirections()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");

        Assert.Equal(owner.Id, lane.Id);
        Assert.Equal(owner.Root, lane.Root);
        Assert.Equal(owner.WorkDirectory, lane.WorkDirectory);
        Assert.Equal(owner.EnvDirectory, lane.EnvDirectory);
        Assert.Equal(owner.StateDirectory, lane.StateDirectory);
        Assert.Equal(owner.TempDirectory, lane.TempDirectory);
        Assert.Equal(owner.RuntimeTempDirectory, lane.RuntimeTempDirectory);

        File.WriteAllText(Path.Combine(owner.WorkDirectory, "from-owner.txt"), "written by the parent");
        Assert.True(lane.TryReadFile("from-owner.txt", out string seen, out string? readError), readError);
        Assert.Equal("written by the parent", seen);

        Assert.True(lane.TryWriteFile("reports/from-lane.txt", "written by the sub-agent", out string? writeError), writeError);
        Assert.Equal("written by the sub-agent",
            File.ReadAllText(Path.Combine(owner.WorkDirectory, "reports", "from-lane.txt")));
        Assert.Contains(owner.ListFiles(), f => f.Path == "reports/from-lane.txt");
    }

    [Fact]
    public void ALane_SharesThePackageLedgers_AndTheHostRepairSet()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");

        // One package tree, so one record of what is in it: a sub-agent's install
        // must not be repeated by its parent, nor claimed for the other registry.
        lane.MarkInstalled("python", new[] { "rich" });
        Assert.True(owner.IsInstalled("python", "rich"));
        Assert.False(owner.IsInstalled("javascript", "rich"));

        Assert.True(owner.TryMarkApplied("requirements:tester"));
        Assert.False(lane.TryMarkApplied("requirements:tester"));

        Assert.True(lane.TryWriteFile("skill-repairs/tool.py", "print('repaired')\n", out string? error), error);
        lane.MarkHostRepairArtifacts("skill-repairs/tool.py");
        Assert.True(owner.IsHostRepairArtifact("skill-repairs/tool.py"));
    }

    [Fact]
    public void ALanesExecution_WaitsForTheOwner_AndTheOwnersForALane()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace laneA = owner.ForAgent("agent_1");
        SessionWorkspace laneB = owner.ForAgent("agent_2");

        // Every agent's code tools still run one at a time: they share the files and the
        // package tree, and a tool that observes another agent's half-written file or
        // half-extracted wheel is the exact hazard the gate exists for.
        AssertBlockedWhileHeld(holder: owner, waiter: laneA);
        AssertBlockedWhileHeld(holder: laneA, waiter: owner);
        AssertBlockedWhileHeld(holder: laneA, waiter: laneB);

        static void AssertBlockedWhileHeld(SessionWorkspace holder, SessionWorkspace waiter)
        {
            using var entered = new ManualResetEventSlim();
            IDisposable held = holder.EnterExecution();
            Task waiting;
            try
            {
                waiting = Task.Run(() =>
                {
                    using (waiter.EnterExecution())
                        entered.Set();
                });
                Assert.False(entered.Wait(TimeSpan.FromMilliseconds(300)),
                    "a second agent entered execution while another agent held it");
            }
            finally
            {
                held.Dispose();
            }
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)), "the waiting agent never entered");
            Assert.True(waiting.Wait(TimeSpan.FromSeconds(10)));
        }
    }

    [Fact]
    public void ALane_KeepsItsOwnerAlive_AndIsUnusableOnceTheOwnerIsReleased()
    {
        SessionWorkspace owner = Workspace("released");
        SessionWorkspace lane = owner.ForAgent("agent_1");
        var cleanup = new CountingCleanup();
        lane.RegisterCleanup(cleanup);

        IDisposable operation = lane.BeginOperation();
        _workspaces.Release("released");

        // The sub-agent's tool is still running, so the one directory is not deleted
        // under it — and nothing new may start in a conversation that has ended.
        Assert.True(Directory.Exists(owner.WorkDirectory));
        Assert.Equal(0, cleanup.DisposeCount);
        Assert.Throws<ObjectDisposedException>(() => lane.BeginOperation());
        Assert.Throws<ObjectDisposedException>(() => owner.BeginOperation());
        Assert.Throws<ObjectDisposedException>(() => owner.ForAgent("agent_2"));
        Assert.Throws<ObjectDisposedException>(() => lane.ForAgent("agent_2"));

        operation.Dispose();

        // A lane has no end of its own: what it registered ends with the conversation.
        Assert.Equal(1, cleanup.DisposeCount);
        Assert.False(Directory.Exists(owner.Root));
    }

    // ---- what is per lane ---------------------------------------------------------------

    [Fact]
    public void ALane_HasItsOwnShellStateDirectory_InsideStateAndCreated()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");

        Assert.Equal(Path.Combine(owner.StateDirectory, "shell"), owner.ShellStateDirectory);
        Assert.Equal(Path.Combine(owner.StateDirectory, "lanes", "agent_1", "shell"), lane.ShellStateDirectory);
        Assert.NotEqual(owner.ShellStateDirectory, lane.ShellStateDirectory);
        Assert.True(Directory.Exists(lane.ShellStateDirectory));

        // The owner's wrapper scripts stay where they were; a lane's go beside its own
        // shell state, where another agent's numbering cannot reach them.
        Assert.Equal(owner.StateDirectory, owner.ShellScriptDirectory);
        Assert.Equal(Path.Combine(owner.StateDirectory, "lanes", "agent_1"), lane.ShellScriptDirectory);
        Assert.True(Directory.Exists(lane.ShellScriptDirectory));
    }

    [Fact]
    public void ShellKey_IsTheRootForTheOwner_AndDistinctForEachLane()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace laneA = owner.ForAgent("agent_1");
        SessionWorkspace laneB = owner.ForAgent("agent_2");

        // The owner keys exactly as every workspace did before lanes existed.
        Assert.Equal(owner.Root, owner.ShellKey);
        Assert.Equal(string.Empty, owner.LaneId);
        Assert.Equal("agent_1", laneA.LaneId);
        Assert.Equal(3, new[] { owner.ShellKey, laneA.ShellKey, laneB.ShellKey }.Distinct(StringComparer.Ordinal).Count());
    }

    [Fact]
    public void TheSameId_IsTheSameLane_FromAnyThread_AndLanesAreFlat()
    {
        SessionWorkspace owner = Workspace();

        var seen = new SessionWorkspace[32];
        Parallel.For(0, seen.Length, i => seen[i] = owner.ForAgent("agent_1"));
        Assert.All(seen, lane => Assert.Same(seen[0], lane));

        // Asking a lane for a lane asks its owner.
        SessionWorkspace other = owner.ForAgent("agent_2");
        Assert.Same(other, seen[0].ForAgent("agent_2"));
        Assert.Same(seen[0], other.ForAgent("agent_1"));

        // Two spellings that would be one directory — on macOS and Windows, or after
        // sanitizing — are one lane, never two lanes silently sharing a shell.
        Assert.Same(seen[0], owner.ForAgent("AGENT_1"));
        Assert.Same(owner.ForAgent("agent9"), owner.ForAgent("agent.9"));
        Assert.NotSame(seen[0], owner.ForAgent("agent1"));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("../..")]
    [InlineData("/")]
    [InlineData("!!!")]
    public void AnIdWithNothingToNameADirectoryWith_IsRejected(string? laneId)
    {
        SessionWorkspace owner = Workspace();

        Assert.ThrowsAny<ArgumentException>(() => owner.ForAgent(laneId!));
        Assert.False(Directory.Exists(Path.Combine(owner.StateDirectory, "lanes"))
                     && Directory.EnumerateDirectories(Path.Combine(owner.StateDirectory, "lanes")).Any());
    }

    [Fact]
    public void AnId_IsReducedToASafeDirectoryName()
    {
        SessionWorkspace owner = Workspace();

        SessionWorkspace traversal = owner.ForAgent("../../etc/agent 1");
        Assert.Equal("etcagent1", traversal.LaneId);
        Assert.StartsWith(
            Path.Combine(owner.StateDirectory, "lanes") + Path.DirectorySeparatorChar,
            traversal.ShellStateDirectory, StringComparison.Ordinal);

        SessionWorkspace longId = owner.ForAgent(new string('a', 100) + "-tail");
        Assert.Equal(new string('a', 64), longId.LaneId);
    }

    [Fact]
    public void EachLane_HasItsOwnReadLedger()
    {
        SessionWorkspace owner = Workspace();
        SessionWorkspace laneA = owner.ForAgent("agent_1");
        SessionWorkspace laneB = owner.ForAgent("agent_2");
        string path = Path.Combine(owner.WorkDirectory, "notes.txt");
        const string content = "one\ntwo\n";
        File.WriteAllText(path, content);

        laneA.Reads.Record(path, content, 1, int.MaxValue, complete: true);

        Assert.Equal(ReadFreshness.Fresh, laneA.Reads.Check(path, content).Freshness);
        Assert.Equal(ReadFreshness.Unread, owner.Reads.Check(path, content).Freshness);
        Assert.Equal(ReadFreshness.Unread, laneB.Reads.Check(path, content).Freshness);
        Assert.Same(laneA.Reads, owner.ForAgent("agent_1").Reads);
    }

    [Fact]
    public void ASubAgentsRead_DoesNotAuthorizeTheParentsEdit()
    {
        // The read gate end to end, through the file tools: a replace_all whose extent
        // the model has not seen is refused. The sub-agent saw the whole file; the
        // parent did not, and must be refused exactly as if nobody had.
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");
        using var runner = new ShellRunner(Options());
        string path = Path.Combine(owner.WorkDirectory, "names.txt");
        File.WriteAllText(path, "old\nkeep\nold\n");

        CodeExecResult read = runner.ReadFile(new ShellTools.ReadRequest("names.txt", 0, 0), lane);
        Assert.True(read.Ok, read.Content);

        CodeExecResult blind = runner.EditFile(new ShellTools.EditRequest("names.txt", "old", "new", true), owner);
        Assert.False(blind.Ok);
        Assert.Contains("you have not seen all", blind.Content, StringComparison.Ordinal);
        Assert.Equal("old\nkeep\nold\n", File.ReadAllText(path));

        CodeExecResult informed = runner.EditFile(new ShellTools.EditRequest("names.txt", "old", "new", true), lane);
        Assert.True(informed.Ok, informed.Content);
        Assert.Equal("new\nkeep\nnew\n", File.ReadAllText(path));
    }

    [Fact]
    public void ARebuild_IsToldToEveryAgentOnce_AndTheOwnersRepairIsUnchanged()
    {
        SessionWorkspace owner = Workspace("wiped");
        SessionWorkspace laneA = owner.ForAgent("agent_1");
        SessionWorkspace laneB = owner.ForAgent("agent_2");

        // The owner's repair is the owner's layout and nothing else, as it always was.
        Directory.Delete(owner.Root, recursive: true);
        Assert.True(owner.EnsureDirectories());
        Assert.True(Directory.Exists(owner.WorkDirectory));
        Assert.False(Directory.Exists(Path.Combine(owner.StateDirectory, "lanes")));
        Assert.True(owner.ConsumeRebuiltNotice());
        Assert.False(owner.ConsumeRebuiltNotice());

        // Each lane repairs its own shell state on its next call and is told once, the
        // way ShellRunner.RunIn asks.
        Assert.True(laneA.TryEnsureDirectories() | laneA.ConsumeRebuiltNotice());
        Assert.True(Directory.Exists(laneA.ShellStateDirectory));
        Assert.False(laneA.ConsumeRebuiltNotice());
        Assert.True(laneB.TryEnsureDirectories() | laneB.ConsumeRebuiltNotice());
        Assert.False(laneB.ConsumeRebuiltNotice());

        // A lane that finds the shared layout gone repairs it for everyone — and the
        // parent, which wrote most of the files and outlives every sub-agent, is still
        // told in its own next result rather than losing the notice to the lane.
        Directory.Delete(owner.Root, recursive: true);
        Assert.True(laneA.EnsureDirectories());
        Assert.True(Directory.Exists(owner.WorkDirectory));
        Assert.True(Directory.Exists(laneA.ShellStateDirectory));
        Assert.True(laneA.ConsumeRebuiltNotice());
        Assert.True(owner.ConsumeRebuiltNotice());
        Assert.True(laneB.ConsumeRebuiltNotice());
        Assert.False(owner.ConsumeRebuiltNotice());
    }

    // ---- the shell, per lane ----------------------------------------------------------

    [Fact]
    public void ALanesShell_IsGrantedOnlyItsOwnState_AndResumesWhereItLeftOff_WithoutAShell()
    {
        // Hermetic: the fake backend persists through the session the way an in-process
        // shell does, so this half runs on every host.
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        using var runner = new ShellRunner(Options(SkillSandboxMode.Required), backend: backend);
        SessionWorkspace owner = Workspace();
        SessionWorkspace lane = owner.ForAgent("agent_1");
        string sub = Path.Combine(owner.WorkDirectory, "sub");
        Directory.CreateDirectory(sub);

        backend.Answer = launch =>
        {
            launch.Session!.Save(sub, new Dictionary<string, string> { ["LANE_PROBE"] = "lane" });
            return FakeShellBackend.Ok(string.Empty);
        };
        CodeExecResult moved = runner.Run(new ShellRequest("cd sub && export LANE_PROBE=lane"), lane);
        Assert.True(moved.Ok, moved.Content);

        // Writable: the lane's own shell state and the shared temp — never the owner's
        // shell state, and never the state directory that holds host-written scripts.
        ShellLaunch first = backend.Launches[0];
        Assert.Contains(lane.ShellStateDirectory, first.WritablePaths);
        Assert.DoesNotContain(owner.ShellStateDirectory, first.WritablePaths);
        Assert.DoesNotContain(owner.StateDirectory, first.WritablePaths);
        Assert.Equal(owner.WorkDirectory, first.WriteDirectory);
        Assert.Equal(owner.Root, first.ReadOnlyDirectory);

        backend.Answer = null;
        runner.Run(new ShellRequest("pwd"), owner);
        runner.Run(new ShellRequest("pwd"), lane);
        ShellLaunch ownerNext = backend.Launches[1];
        ShellLaunch laneNext = backend.Launches[2];

        Assert.Equal(owner.WorkDirectory, ownerNext.WorkingDirectory);
        Assert.False(ownerNext.Session!.Load().Environment.ContainsKey("LANE_PROBE"));
        Assert.Equal(sub, laneNext.WorkingDirectory);
        Assert.Equal("lane", laneNext.Session!.Load().Environment["LANE_PROBE"]);
        Assert.Same(first.Session, laneNext.Session);
        Assert.NotSame(first.Session, ownerNext.Session);
    }

    [Fact]
    public void TwoLanes_KeepIndependentWorkingDirectories_OverTheSameFiles()
    {
        if (!HavePosixShell) return;

        SessionWorkspace owner = Workspace();
        SessionWorkspace laneA = owner.ForAgent("agent_1");
        SessionWorkspace laneB = owner.ForAgent("agent_2");
        using var runner = new ShellRunner(Options());

        Assert.NotSame(runner.SessionFor(owner), runner.SessionFor(laneA));
        Assert.NotSame(runner.SessionFor(laneA), runner.SessionFor(laneB));
        Assert.Same(runner.SessionFor(laneA), runner.SessionFor(owner.ForAgent("agent_1")));

        const string where = "echo \"[cwd=$(basename \"$PWD\")][probe=$LANE_PROBE]\"";

        // Sub-agent A moves, exports and writes.
        CodeExecResult moved = runner.Run(new ShellRequest(
            "mkdir sub && cd sub && printf 'from-a\\n' > made-by-a.txt && export LANE_PROBE=a"), laneA);
        Assert.True(moved.Ok, moved.Content);
        Assert.Contains("Working directory is now sub", moved.Content, StringComparison.Ordinal);

        // Sub-agent B has not moved and inherited nothing — but sees A's file.
        CodeExecResult b = runner.Run(new ShellRequest(where + " && cat sub/made-by-a.txt"), laneB);
        Assert.True(b.Ok, b.Content);
        Assert.Contains("[cwd=work][probe=]", b.Content, StringComparison.Ordinal);
        Assert.Contains("from-a", b.Content, StringComparison.Ordinal);

        // Nor has the parent.
        CodeExecResult parent = runner.Run(new ShellRequest(where), owner);
        Assert.True(parent.Ok, parent.Content);
        Assert.Contains("[cwd=work][probe=]", parent.Content, StringComparison.Ordinal);

        // A is still where it left itself, and sees what B writes next.
        CodeExecResult bWrites = runner.Run(new ShellRequest("printf 'from-b\\n' > sub/made-by-b.txt"), laneB);
        Assert.True(bWrites.Ok, bWrites.Content);
        CodeExecResult a = runner.Run(new ShellRequest(where + " && cat made-by-b.txt"), laneA);
        Assert.True(a.Ok, a.Content);
        Assert.Contains("[cwd=sub][probe=a]", a.Content, StringComparison.Ordinal);
        Assert.Contains("from-b", a.Content, StringComparison.Ordinal);

        Assert.Equal("sub", runner.SessionFor(laneA).CurrentDirectoryLabel);
        Assert.Equal(".", runner.SessionFor(laneB).CurrentDirectoryLabel);
        Assert.Equal(".", runner.SessionFor(owner).CurrentDirectoryLabel);

        // The two halves of the tool surface agree per lane: a relative path means the
        // file where THAT agent's shell is.
        Assert.True(runner.ReadFile(new ShellTools.ReadRequest("made-by-a.txt", 0, 0), laneA).Ok);
        Assert.False(runner.ReadFile(new ShellTools.ReadRequest("made-by-a.txt", 0, 0), owner).Ok);

        // Each shell numbered its own wrapper scripts in its own directory: A ran two
        // commands, B two, the parent one — none of them rewrote another's cmd-1.
        Assert.Equal(2, Directory.GetFiles(laneA.ShellScriptDirectory, "cmd-*").Length);
        Assert.Equal(2, Directory.GetFiles(laneB.ShellScriptDirectory, "cmd-*").Length);
        Assert.Single(Directory.GetFiles(owner.StateDirectory, "cmd-*"));
    }

    private sealed class CountingCleanup : IDisposable
    {
        private int _disposeCount;

        public int DisposeCount => Volatile.Read(ref _disposeCount);

        public void Dispose() => Interlocked.Increment(ref _disposeCount);
    }
}
