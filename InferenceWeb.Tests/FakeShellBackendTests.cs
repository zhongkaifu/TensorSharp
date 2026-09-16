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
using System.Text;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// An <see cref="IShellBackend"/> that runs nothing: it records every launch it is
/// handed and answers with whatever the test canned. This is the shape an iOS host
/// plugs in — an in-process runtime that has no child process — so what these tests
/// pin is the contract across the seam: what the host hands over, and what it does
/// with the answer.
/// </summary>
internal sealed class FakeShellBackend : IShellBackend
{
    public List<ShellLaunch> Launches { get; } = new();

    /// <summary>What to answer a launch with. Null answers "exit 0, printed the command".</summary>
    public Func<ShellLaunch, ConfinedResult>? Answer { get; set; }

    public ISkillSandbox? Sandbox { get; init; }

    public ShellProgram? Shell { get; init; } = ShellProgram.InProcess();

    public string Name => "fake";

    public bool CanRun => Shell != null;

    public string? UnavailableReason => Shell == null ? "the fake backend was built without a shell" : null;

    public static ConfinedResult Ok(string stdout, string stderr = "", int exitCode = 0) =>
        new(true, false, exitCode, stdout, stderr, TimeSpan.FromMilliseconds(12), "fake", null);

    public bool TryStart(ShellLaunch launch, out IShellJob? job, out ConfinedResult failure)
    {
        Launches.Add(launch);
        ConfinedResult result = Answer?.Invoke(launch)
            ?? Ok((launch.Command ?? string.Join(" ", launch.Argv!)) + "\n");
        if (!result.Started)
        {
            job = null;
            failure = result;
            return false;
        }
        job = new FakeJob(result);
        failure = default;
        return true;
    }

    private sealed class FakeJob : IShellJob
    {
        private readonly ConfinedResult _result;

        public FakeJob(ConfinedResult result) => _result = result;

        public int ProcessId => -1;

        public string SandboxName => _result.SandboxName;

        public bool HasExited => true;

        public bool Killed { get; private set; }

        public ConfinedResult WaitForExit(TimeSpan timeout) => _result;

        public void Kill() => Killed = true;

        public void Dispose() { }
    }
}

/// <summary>Records what the host asked it to install, and "installs" it by saying yes.</summary>
internal sealed class RecordingInstaller : IPackageInstaller
{
    public List<(string Language, string[] Packages)> Requests { get; } = new();

    public bool CanInstall { get; init; } = true;

    public string? FailWith { get; init; }

    public string? Install(
        SessionWorkspace workspace, CodeLanguage language,
        IReadOnlyList<string> packages, Action<string>? onOutput, out bool performed)
    {
        Requests.Add((CodeExecOptions.NameOf(language), packages.ToArray()));
        performed = FailWith == null;
        return FailWith;
    }
}

/// <summary>
/// The shell tool driven through a backend that starts no process. Every sentence of
/// the result — the exit line, the working directory, the environment-reset note, the
/// install notes — is produced by the host from the backend's answer and the session's
/// state files, so a backend that persists state through
/// <see cref="ShellSession.Save"/> gets exactly the result a wrapper script gets.
/// </summary>
public sealed class FakeShellBackendTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public FakeShellBackendTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-fakebackend-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static readonly SkillSandboxCapabilities Honest = new(
        ConfinesWrites: true, ConfinesNetwork: true, ConfinesHomeReads: true, BoundsProcessTree: true);

    private static FakeShellBackend Backend(SkillSandboxCapabilities? capabilities = null) =>
        new() { Sandbox = new InProcessSandbox(capabilities ?? Honest) };

    /// <summary>Enabled, sandbox REQUIRED and never unconfined: the fake has to earn CanRun.</summary>
    private CodeExecOptions Options(Action<CodeExecOptions>? tweak = null)
    {
        var options = new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Required,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        };
        tweak?.Invoke(options);
        return options;
    }

    private SessionWorkspace Workspace(string id = "fake") => _workspaces.GetOrCreate(id);

    private ShellRunner Runner(
        FakeShellBackend backend, Action<CodeExecOptions>? tweak = null,
        IPackageInstaller? installer = null, ISyntaxVerifier? syntax = null) =>
        new(Options(tweak), backend: backend, installer: installer, syntax: syntax);

    // ---- confinement is judged from the backend's sandbox ------------------------

    [Fact]
    public void AnHonestInProcessSandbox_MakesCanRunTrue_UnderRequired_WithoutUnconfined()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);

        Assert.True(runner.CanRun, runner.UnavailableReason);
        Assert.Null(runner.UnavailableReason);
        Assert.True(runner.NetworkConfinementGuaranteed);
        Assert.Same(backend, runner.Backend);
        Assert.Equal("in-process", runner.Sandbox!.Name);
    }

    [Fact]
    public void ABackendWhoseSandboxLeavesTheNetworkOpen_IsRefusedUnderRequired_AndTheGapIsNamed()
    {
        FakeShellBackend backend = Backend(new SkillSandboxCapabilities(
            ConfinesWrites: true, ConfinesNetwork: false, ConfinesHomeReads: true, BoundsProcessTree: true));
        using var runner = Runner(backend);

        Assert.False(runner.CanRun);
        Assert.Contains("in-process", runner.UnavailableReason!, StringComparison.Ordinal);
        Assert.Contains("reach the network", runner.UnavailableReason!, StringComparison.Ordinal);

        CodeExecResult result = runner.Run(new ShellRequest("echo hi"), Workspace());
        Assert.False(result.Ok);
        Assert.Empty(backend.Launches);
    }

    [Fact]
    public void ABackendWhoseSandboxLeavesTheNetworkOpen_RunsWhenTheOperatorOpenedTheNetworkAnyway()
    {
        // Network confinement is not required after the operator explicitly enables
        // network access; write confinement still is. Same rule as the OS sandboxes.
        FakeShellBackend backend = Backend(new SkillSandboxCapabilities(
            ConfinesWrites: true, ConfinesNetwork: false, ConfinesHomeReads: true, BoundsProcessTree: true));
        using var runner = Runner(backend, o => o.AllowNetwork = true);

        Assert.True(runner.CanRun, runner.UnavailableReason);
        Assert.False(runner.NetworkConfinementGuaranteed);
    }

    [Fact]
    public void ABackendWithNoShell_IsUnavailable_AndSaysWhy()
    {
        var backend = new FakeShellBackend { Shell = null, Sandbox = new InProcessSandbox(Honest) };
        using var runner = Runner(backend);

        Assert.False(runner.CanRun);
        Assert.Contains("without a shell", runner.UnavailableReason!, StringComparison.Ordinal);
        Assert.Empty(new CodeRunnerAdapter(runner).DeclareTools());
    }

    [Fact]
    public void TheDeclarationNamesTheInProcessDialect()
    {
        using var runner = Runner(Backend());
        var declared = new CodeRunnerAdapter(runner).DeclareTools();

        Assert.Contains(declared, t => t.Name == ShellTools.ShellToolName);
        string shell = declared.Single(t => t.Name == ShellTools.ShellToolName).Description!;
        Assert.Contains("Run a sh command", shell, StringComparison.Ordinal);
    }

    // ---- the launch, and the result made from its answer --------------------------

    [Fact]
    public void ACommandIsHandedAcrossTheSeam_WithTheHostsTerms_AndTheResultReadsTheAnswer()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        SessionWorkspace workspace = Workspace();

        CodeExecResult result = runner.Run(new ShellRequest("echo hi"), workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("exit 0 (0.01s, sandbox: fake)", result.Content, StringComparison.Ordinal);
        Assert.Contains("\necho hi\n", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("Not confined on this host", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("Working directory is now", result.Content, StringComparison.Ordinal);

        ShellLaunch launch = Assert.Single(backend.Launches);
        Assert.Equal("echo hi", launch.Command);
        Assert.Null(launch.Argv);
        Assert.Equal(ShellLaunch.Purposes.Shell, launch.Purpose);
        Assert.NotNull(launch.Session);
        Assert.Null(launch.CallWorkDirectory);
        Assert.Equal(workspace.WorkDirectory, launch.WorkingDirectory);
        Assert.Equal(workspace.WorkDirectory, launch.WriteDirectory);
        Assert.Equal(workspace.Root, launch.ReadOnlyDirectory);
        Assert.Contains(workspace.TempDirectory, launch.WritablePaths);
        Assert.Contains(workspace.ShellStateDirectory, launch.WritablePaths);
        Assert.Contains(workspace.EnvDirectory, launch.ReadablePaths);
        Assert.False(launch.AllowNetwork);
        Assert.Equal(TimeSpan.FromSeconds(30), launch.Timeout);
        Assert.Equal(workspace.WorkDirectory, launch.Environment["HOME"]);
        Assert.Equal(workspace.TempDirectory, launch.Environment["TMPDIR"]);
        Assert.Equal(workspace.EnvDirectory, launch.Environment["PYTHONPATH"]);
        Assert.True(launch.Environment.ContainsKey("PATH"));

        // Nothing was written for a backend that has no wrapper script to run.
        Assert.Empty(Directory.GetFiles(workspace.StateDirectory, "cmd-*"));
    }

    [Fact]
    public void ABackendThatSavesTheSessionState_GetsTheWorkingDirectoryLine_AndTheNextLaunchStartsThere()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        SessionWorkspace workspace = Workspace();
        string sub = Path.Combine(workspace.WorkDirectory, "sub");
        Directory.CreateDirectory(sub);

        backend.Answer = launch =>
        {
            // What an in-process shell does after `cd sub; export FOO=bar`: persist
            // through the session, the way the wrapper's EXIT trap does.
            launch.Session!.Save(sub, new Dictionary<string, string> { ["FOO"] = "bar" });
            return FakeShellBackend.Ok(string.Empty);
        };
        CodeExecResult first = runner.Run(new ShellRequest("cd sub && export FOO=bar"), workspace);

        Assert.True(first.Ok, first.Content);
        Assert.Contains("Working directory is now sub\n", first.Content, StringComparison.Ordinal);
        Assert.Contains("(no output)", first.Content, StringComparison.Ordinal);

        backend.Answer = null;
        CodeExecResult second = runner.Run(new ShellRequest("pwd"), workspace);

        Assert.True(second.Ok, second.Content);
        ShellLaunch next = backend.Launches[1];
        Assert.Equal(sub, next.WorkingDirectory);
        ShellState state = next.Session!.Load();
        Assert.Equal(sub, state.CurrentDirectory);
        Assert.Equal("bar", state.Environment["FOO"]);
        Assert.Contains("Working directory is now sub\n", second.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void APerCallWorkDirectory_IsHandedOverAsSuch_AndDoesNotMoveTheSession()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        SessionWorkspace workspace = Workspace();
        string build = Path.Combine(workspace.WorkDirectory, "build");
        Directory.CreateDirectory(build);

        CodeExecResult result = runner.Run(new ShellRequest("make") { WorkDirectory = "build" }, workspace);

        Assert.True(result.Ok, result.Content);
        ShellLaunch launch = Assert.Single(backend.Launches);
        Assert.Equal(build, launch.WorkingDirectory);
        Assert.Equal(build, launch.CallWorkDirectory);
        // The backend saved nothing, so the session is still where it was.
        Assert.DoesNotContain("Working directory is now", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ABackendThatResetsTheEnvironment_GetsTheResetNote_Once()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        SessionWorkspace workspace = Workspace();

        backend.Answer = launch =>
        {
            Assert.True(launch.Session!.MarkEnvironmentReset());
            return FakeShellBackend.Ok("ok\n");
        };
        CodeExecResult first = runner.Run(new ShellRequest("export X=$(printf 'a\\nb')"), workspace);
        Assert.Contains("saved environment could not be read back and was reset", first.Content, StringComparison.Ordinal);

        backend.Answer = null;
        CodeExecResult second = runner.Run(new ShellRequest("echo again"), workspace);
        Assert.DoesNotContain("could not be read back", second.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AFailingAnswer_IsNotOk_AndCarriesTheOutput_AndTheGapsOfAWeakerSandbox()
    {
        FakeShellBackend backend = Backend(new SkillSandboxCapabilities(
            ConfinesWrites: true, ConfinesNetwork: true, ConfinesHomeReads: false, BoundsProcessTree: true));
        using var runner = Runner(backend);
        backend.Answer = _ => FakeShellBackend.Ok(string.Empty, "boom: no such file\n", exitCode: 2);

        CodeExecResult result = runner.Run(new ShellRequest("cat missing.txt"), Workspace());

        Assert.False(result.Ok);
        Assert.Contains("exit 2 (0.01s, sandbox: fake)", result.Content, StringComparison.Ordinal);
        Assert.Contains("boom: no such file", result.Content, StringComparison.Ordinal);
        Assert.Contains("Not confined on this host: commands may read the user's home directory.", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ABackendThatCannotStart_ProducesARefusalWithItsReason()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        backend.Answer = _ => new ConfinedResult(
            false, false, -1, string.Empty, string.Empty, TimeSpan.Zero, "none", "the interpreter is not loaded yet");

        CodeExecResult result = runner.Run(new ShellRequest("python3 x.py"), Workspace());

        Assert.False(result.Ok);
        Assert.Contains("The command was not run: the interpreter is not loaded yet", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ATimedOutAnswer_SaysSo_AndKeepsWhatWasPrinted()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        backend.Answer = _ => new ConfinedResult(
            true, true, -1, "partial\n", string.Empty, TimeSpan.FromSeconds(30), "fake", null);

        CodeExecResult result = runner.Run(new ShellRequest("sleep 100") { Timeout = TimeSpan.FromSeconds(5) }, Workspace());

        Assert.False(result.Ok);
        Assert.Contains("did not finish within 5s", result.Content, StringComparison.Ordinal);
        Assert.Contains("partial", result.Content, StringComparison.Ordinal);
        Assert.Equal(TimeSpan.FromSeconds(5), Assert.Single(backend.Launches).Timeout);
    }

    [Fact]
    public void ABackgroundJob_IsStartedWithoutADeadline_AndTheResultNamesItsLog()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        SessionWorkspace workspace = Workspace();

        CodeExecResult result = runner.Run(new ShellRequest("python3 -m http.server") { Background = true }, workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("Started in the background as job-1", result.Content, StringComparison.Ordinal);
        Assert.Contains("tail -n 40 ../state/.jobs/job-1.log", result.Content, StringComparison.Ordinal);
        ShellLaunch launch = Assert.Single(backend.Launches);
        Assert.Equal(System.Threading.Timeout.InfiniteTimeSpan, launch.Timeout);
        Assert.True(File.Exists(Path.Combine(workspace.StateDirectory, ".jobs", "job-1.log")));
    }

    // ---- installs are the host's, and go to whatever installer it was given ------

    [Fact]
    public void AnInstallInTheLine_IsReadOut_HandedToTheInstaller_AndSubstitutedBeforeTheBackendSeesIt()
    {
        FakeShellBackend backend = Backend();
        var installer = new RecordingInstaller();
        using var runner = Runner(backend, o => o.AllowInstall = true, installer);

        CodeExecResult result = runner.Run(
            new ShellRequest("pip install requests && python3 fetch.py"), Workspace());

        Assert.True(result.Ok, result.Content);
        (string language, string[] packages) = Assert.Single(installer.Requests);
        Assert.Equal("python", language);
        Assert.Equal(new[] { "requests" }, packages);
        Assert.Contains("Installed: requests", result.Content, StringComparison.Ordinal);
        Assert.Equal("true && python3 fetch.py", Assert.Single(backend.Launches).Command);
    }

    [Fact]
    public void ARefusedInstall_BecomesFalse_AndTheCallIsNotOk_WhateverTheRestOfTheLineExited()
    {
        FakeShellBackend backend = Backend();
        var installer = new RecordingInstaller { FailWith = "'requests' is not on this host's allowed-package list" };
        using var runner = Runner(backend, o => o.AllowInstall = true, installer);

        CodeExecResult result = runner.Run(
            new ShellRequest("pip install requests; python3 fetch.py"), Workspace());

        Assert.False(result.Ok);
        Assert.Contains("The install above did not happen", result.Content, StringComparison.Ordinal);
        Assert.Contains("allowed-package list", result.Content, StringComparison.Ordinal);
        Assert.Equal("false; python3 fetch.py", Assert.Single(backend.Launches).Command);
    }

    [Fact]
    public void AMissingModule_IsInstalledThroughTheInstaller_AndTheCommandRunAgain_InOneCall()
    {
        FakeShellBackend backend = Backend();
        var installer = new RecordingInstaller();
        using var runner = Runner(backend, o => o.AllowInstall = true, installer);
        int attempt = 0;
        backend.Answer = _ => ++attempt == 1
            ? FakeShellBackend.Ok(string.Empty, "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nModuleNotFoundError: No module named 'yaml'\n", exitCode: 1)
            : FakeShellBackend.Ok("parsed\n");

        CodeExecResult result = runner.Run(new ShellRequest("python3 x.py"), Workspace());

        Assert.True(result.Ok, result.Content);
        Assert.Equal(2, backend.Launches.Count);
        Assert.Equal(new[] { "PyYAML" }, Assert.Single(installer.Requests).Packages);
        Assert.Contains("'yaml' was missing, so PyYAML was installed and the command was run again.", result.Content, StringComparison.Ordinal);
        Assert.Contains("parsed", result.Content, StringComparison.Ordinal);
    }

    // ---- the file tools are the host's alone ------------------------------------

    [Fact]
    public void ReadEditWriteAndPatch_NeverTouchTheBackend()
    {
        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        SessionWorkspace workspace = Workspace();

        Assert.True(runner.WriteFile(new ShellTools.WriteRequest("notes.txt", "alpha\nbeta\n"), workspace).Ok);
        Assert.True(runner.ReadFile(new ShellTools.ReadRequest("notes.txt", 0, 0), workspace).Ok);
        Assert.True(runner.EditFile(new ShellTools.EditRequest("notes.txt", "beta", "gamma", false), workspace).Ok);
        CodeExecResult patched = runner.ApplyPatch(
            "*** Begin Patch\n*** Add File: extra.txt\n+one\n+two\n*** End Patch\n", workspace);
        Assert.True(patched.Ok, patched.Content);

        Assert.Equal("alpha\ngamma\n", File.ReadAllText(Path.Combine(workspace.WorkDirectory, "notes.txt")));
        Assert.Equal("one\ntwo\n", File.ReadAllText(Path.Combine(workspace.WorkDirectory, "extra.txt")));
        Assert.Empty(backend.Launches);
    }

    [Fact]
    public void TheSyntaxCheckAfterAWrite_IsAnArgvLaunchThroughTheBackend()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _) || python == null)
            return;   // The default verifier launches the resolved interpreter; nothing to launch here.

        FakeShellBackend backend = Backend();
        using var runner = Runner(backend);
        SessionWorkspace workspace = Workspace();
        backend.Answer = _ => FakeShellBackend.Ok(string.Empty);

        Assert.True(runner.WriteFile(new ShellTools.WriteRequest("ok.py", "print(1)\n"), workspace).Ok);

        ShellLaunch launch = Assert.Single(backend.Launches);
        Assert.Null(launch.Command);
        Assert.Equal(ShellLaunch.Purposes.SyntaxCheck, launch.Purpose);
        Assert.Equal(python, launch.Argv![0]);
        Assert.Equal("-c", launch.Argv[1]);
        Assert.Contains(Path.Combine(workspace.WorkDirectory, "ok.py"), launch.Argv);
        Assert.Equal(workspace.TempDirectory, launch.WriteDirectory);
        Assert.False(launch.AllowNetwork);
    }

    [Fact]
    public void ASuppliedVerifier_ReplacesTheDefault_AndTheBackendSeesNothing()
    {
        FakeShellBackend backend = Backend();
        var verifier = new RecordingVerifier();
        using var runner = Runner(backend, syntax: verifier);
        SessionWorkspace workspace = Workspace();

        CodeExecResult result = runner.WriteFile(new ShellTools.WriteRequest("ok.py", "print(1)\n"), workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Equal(new[] { "ok.py" }, Assert.Single(verifier.Asked));
        Assert.Empty(backend.Launches);
    }

    private sealed class RecordingVerifier : ISyntaxVerifier
    {
        public List<string[]> Asked { get; } = new();

        public string? Verify(IReadOnlyList<string> relativePaths, SessionWorkspace workspace, string? ranIn = null)
        {
            Asked.Add(relativePaths.ToArray());
            return null;
        }
    }

    // ---- ProcessShellBackend is the desktop behaviour, unchanged --------------------

    [Fact]
    public void TheDefaultBackend_IsTheProcessOne_WithTheDetectedShellAndSandbox()
    {
        using var runner = new ShellRunner(Options(o => o.Sandbox = SkillSandboxMode.Off));

        var backend = Assert.IsType<ProcessShellBackend>(runner.Backend);
        Assert.Equal("process", backend.Name);
        Assert.Null(backend.Sandbox);
        Assert.Equal(SkillSandboxMode.Off, backend.Mode);
        Assert.Same(backend.Shell, runner.Shell);
        if (runner.Shell != null)
            Assert.True(backend.CanRun);
        else
            Assert.NotNull(backend.UnavailableReason);
    }

    [Fact]
    public void TheProcessBackend_RefusesALaunchThatIsNotWellFormed_InsteadOfGuessing()
    {
        var backend = new ProcessShellBackend(null, SkillSandboxMode.Off, ShellProgram.InProcess());
        SessionWorkspace workspace = Workspace();

        var both = new ShellLaunch
        {
            Command = "echo", Argv = new[] { "echo" },
            WorkingDirectory = workspace.WorkDirectory, WriteDirectory = workspace.WorkDirectory,
            ReadOnlyDirectory = workspace.Root,
        };
        Assert.False(backend.TryStart(both, out IShellJob? job, out ConfinedResult failure));
        Assert.Null(job);
        Assert.False(failure.Started);
        Assert.Contains("never both", failure.Error!, StringComparison.Ordinal);

        var noSession = both with { Argv = null };
        Assert.False(backend.TryStart(noSession, out _, out failure));
        Assert.Contains("needs the session", failure.Error!, StringComparison.Ordinal);
    }

    [Fact]
    public void TheProcessBackend_RewritesTheWrapperScriptIntoTheModelsFrame()
    {
        var script = new ShellSession.ShellScript("/tmp/state/cmd-7.sh", 20);
        var raw = new ConfinedResult(
            true, false, 2, "ran /tmp/state/cmd-7.sh\n",
            "/tmp/state/cmd-7.sh: line 24: foo: command not found\n",
            TimeSpan.FromMilliseconds(5), "none", null);

        ConfinedResult fixedUp = ProcessShellBackend.Rewrite(raw, script);

        Assert.Equal("ran command\n", fixedUp.Stdout);
        Assert.Equal("command line 4: foo: command not found\n", fixedUp.Stderr);
    }
}

/// <summary>
/// Skill scripts through the same seam: an argv launch, no shell, the skill's own
/// directory read-only, the session's environment importable.
/// </summary>
public sealed class SkillScriptRunnerBackendTests : IDisposable
{
    private readonly string _base;

    public SkillScriptRunnerBackendTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-skillbackend-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static readonly SkillSandboxCapabilities Honest = new(
        ConfinesWrites: true, ConfinesNetwork: true, ConfinesHomeReads: true, BoundsProcessTree: true);

    private SessionWorkspace Workspace(string id) =>
        new SessionWorkspaceManager(Path.Combine(_base, "sessions")).GetOrCreate(id);

    private Skill MakeSkill()
    {
        string root = Path.Combine(_base, "skill-" + Guid.NewGuid().ToString("N"));
        string dir = Path.Combine(root, "tester");
        Directory.CreateDirectory(Path.Combine(dir, "scripts"));
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            "---\nname: tester\ndescription: a skill whose script runs through the seam\n---\nBody.");
        File.WriteAllText(Path.Combine(dir, "scripts", "tool.py"), "print('hello')\n");
        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { root } }).Skills.Single();
    }

    private Skill MakeSkill(string relativePath, string source)
    {
        string root = Path.Combine(_base, "skill-" + Guid.NewGuid().ToString("N"));
        string dir = Path.Combine(root, "tester");
        string full = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            "---\nname: tester\ndescription: a skill whose script runs through the seam\n---\nBody.");
        File.WriteAllText(full, source);
        return new SkillRegistry(new SkillRegistryOptions { Roots = new[] { root } }).Skills.Single();
    }

    private static string[] RepairCopies(SessionWorkspace workspace) =>
        Directory.GetFiles(workspace.WorkDirectory, "*.py", SearchOption.AllDirectories);

    [Fact]
    public void AScriptIsAnArgvLaunch_WithTheInterpreterFirst_AndTheSessionEnvironmentImportable()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("a");
        var options = new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required,
            Backend = backend,
            Workspace = workspace,
        };
        var runner = new SkillScriptRunner(options);
        Skill skill = MakeSkill();

        Assert.True(runner.CanRun, runner.UnavailableReason);
        Assert.Same(backend, runner.Backend);

        backend.Answer = _ => FakeShellBackend.Ok("hello\n");
        SkillToolResult result = runner.Run(skill, "scripts/tool.py", new[] { "--n", "3" });

        Assert.True(result.Ok, result.Content);
        Assert.Contains("Ran scripts/tool.py (exit code 0, sandbox: fake)", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("Not confined on this host", result.Content, StringComparison.Ordinal);
        Assert.Contains("stdout:\nhello", result.Content, StringComparison.Ordinal);

        ShellLaunch launch = Assert.Single(backend.Launches);
        Assert.Null(launch.Command);
        Assert.Equal(ShellLaunch.Purposes.Script, launch.Purpose);
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        Assert.Equal(new[] { options.Interpreters[".py"], scriptPath, "--n", "3" }, launch.Argv);
        Assert.Equal(skill.RootDirectory, launch.ReadOnlyDirectory);
        Assert.Equal(workspace.WorkDirectory, launch.WriteDirectory);
        Assert.Equal(workspace.WorkDirectory, launch.WorkingDirectory);
        Assert.Contains(workspace.EnvDirectory, launch.ReadablePaths);
        Assert.False(launch.AllowNetwork);
        Assert.Equal(TimeSpan.FromSeconds(60), launch.Timeout);

        // Windows keeps interpreter caches beneath the session's hidden home;
        // POSIX redirects both variables to the writable working directory.
        string expectedHome = OperatingSystem.IsWindows()
            ? Path.Combine(workspace.WorkDirectory, ".home") : workspace.WorkDirectory;
        string expectedTemp = OperatingSystem.IsWindows()
            ? Path.Combine(expectedHome, "Temp") : workspace.WorkDirectory;
        Assert.Equal(expectedHome, launch.Environment["HOME"]);
        Assert.Equal(expectedTemp, launch.Environment["TMPDIR"]);
        Assert.Equal(workspace.WorkDirectory, launch.Environment["PWD"]);
        Assert.Equal(Path.Combine(workspace.EnvDirectory, "node_modules"), launch.Environment["NODE_PATH"]);
        Assert.Equal(
            workspace.EnvDirectory + Path.PathSeparator + skill.RootDirectory,
            launch.Environment["PYTHONPATH"]);
        Assert.Equal("1", launch.Environment["PYTHONDONTWRITEBYTECODE"]);
    }

    [Fact]
    public void SkillsRun_SeesAnAttachedCsvOnItsFirstCall()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("attached-csv");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required,
            Backend = backend,
            Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string source = Path.Combine(_base, "stored-form.csv");
        File.WriteAllText(source, "row,value\n1,FIRST_SKILL_RUN_SEES_ME\n");

        // The fake backend starts no process, but reads from the exact working
        // directory the real script receives. Before central staging this callback
        // throws on the first skills_run call because only CodeRunnerAdapter staged
        // attachments, and skill scripts bypass that adapter.
        backend.Answer = launch => FakeShellBackend.Ok(
            File.ReadAllText(Path.Combine(launch.WorkingDirectory, "form.csv")));

        var context = new SkillToolContext(new[] { skill })
        {
            ScriptRunner = runner,
            Workspace = workspace,
            CodeInputFiles = new[] { new CodeInputFile("form.csv", source) },
        };
        SkillToolResult result = SkillTools.Execute(
            new ToolCall
            {
                Name = SkillTools.RunToolName,
                Arguments = new Dictionary<string, object>
                {
                    ["skill"] = "tester",
                    ["path"] = "scripts/tool.py",
                },
            },
            context);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("FIRST_SKILL_RUN_SEES_ME", result.Content, StringComparison.Ordinal);
        Assert.Equal(File.ReadAllBytes(source), File.ReadAllBytes(Path.Combine(workspace.WorkDirectory, "form.csv")));
    }

    [Fact]
    public void SkillsRun_RepairsAnExactAttachedNameWithoutChangingOrdinaryScalarSplitting()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("attached-csv-spaces");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required,
            Backend = backend,
            Workspace = workspace,
        });
        Skill skill = MakeSkill();
        const string displayName = "webresults-20251106 2.csv";
        string source = Path.Combine(_base, "stored-results.csv");
        File.WriteAllText(source, "candidate,votes\nSam,10\n");

        backend.Answer = _ => FakeShellBackend.Ok("analyzed\n");
        var context = new SkillToolContext(new[] { skill })
        {
            ScriptRunner = runner,
            Workspace = workspace,
            CodeInputFiles = new[] { new CodeInputFile(displayName, source) },
        };

        SkillToolResult result = SkillTools.Execute(
            new ToolCall
            {
                Name = SkillTools.RunToolName,
                Arguments = new Dictionary<string, object>
                {
                    ["skill"] = "tester",
                    ["path"] = "scripts/tool.py",
                    // Exact shape observed in the live upload-to-chat run: the model
                    // copied the right friendly name but did not add nested quotes.
                    ["args"] = displayName,
                },
            },
            context);

        Assert.True(result.Ok, result.Content);
        ShellLaunch launch = Assert.Single(backend.Launches);
        Assert.Equal(displayName, Assert.Single(launch.Argv!.Skip(2)));
        Assert.True(File.Exists(Path.Combine(workspace.WorkDirectory, displayName)));

        SkillToolResult ordinary = SkillTools.Execute(
            new ToolCall
            {
                Name = SkillTools.RunToolName,
                Arguments = new Dictionary<string, object>
                {
                    ["skill"] = "tester",
                    ["path"] = "scripts/tool.py",
                    ["args"] = "--out summary.json 2400",
                },
            },
            context);

        Assert.True(ordinary.Ok, ordinary.Content);
        Assert.Equal(new[] { "--out", "summary.json", "2400" }, backend.Launches[1].Argv!.Skip(2));
    }

    [Fact]
    public void AMissingPackageFailure_IsNotMisrepresentedAsABrokenSkillScript()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("b");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n  File \"{scriptPath}\", line 1, in <module>\n"
            + "ModuleNotFoundError: No module named 'lxml'\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Contains("(exit code 1, sandbox: fake)", result.Content, StringComparison.Ordinal);
        Assert.Contains("pip install lxml", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.Empty(RepairCopies(workspace));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnInvalidWorkspaceJsonSpec_IsExcerptedAndAuthorized_InsteadOfStagingTheSkillScript(
        bool joinedOption)
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace(joinedOption ? "bad-json-equals" : "bad-json-pair");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();

        string directory = Path.Combine(workspace.WorkDirectory, "reports");
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "apple-chips.json");
        const string spec = """
            {
              "title": "Apple M6 vs M5",
              "slides": [
                {"layout": "title", "title": "Apple M6 vs M5"},
                {"layout": "table", "rows": [["M5", "known"], ["M6", "reported"]]},
              ]
            }
            """;
        File.WriteAllText(path, spec);
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            "make_pptx: the spec is not valid JSON — Expecting value at line 6, column 3\n",
            exitCode: 1);

        string[] arguments = joinedOption
            ? new[] { "--spec=reports/apple-chips.json", "--out", "apple-chips.pptx" }
            : new[] { "--spec", "reports/apple-chips.json", "--out", "apple-chips.pptx" };
        SkillToolResult result = runner.Run(skill, "scripts/tool.py", arguments);

        Assert.False(result.Ok);
        Assert.Contains("input file 'reports/apple-chips.json'", result.Content, StringComparison.Ordinal);
        Assert.Contains("not from the bundled skill script", result.Content, StringComparison.Ordinal);
        Assert.Contains("Apple M6 vs M5", result.Content, StringComparison.Ordinal);
        Assert.Contains("with `edit_file`", result.Content, StringComparison.Ordinal);
        Assert.Contains("Do not use `write_file` or re-type the whole spec", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.Empty(RepairCopies(workspace));

        FileLedger.ReadState seen = workspace.Reads.Check(
            path, spec.Replace("\r\n", "\n", StringComparison.Ordinal));
        Assert.Equal(ReadFreshness.Partial, seen.Freshness);
        Assert.True(seen.Covers(6, 6),
            $"the invalid line was shown but not authorized: {seen.FirstLine}-{seen.LastLine}");
    }

    [Fact]
    public void AValidJsonSpec_DoesNotHideAGenuineSkillScriptFailure()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("valid-json-script-failure");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "deck.json"),
            "{\"title\":\"Apple M6 vs M5\",\"slides\":[]}");
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n  File \"{scriptPath}\", line 1\nRuntimeError: writer bug\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(
            skill, "scripts/tool.py", new[] { "--spec", "deck.json", "--out", "deck.pptx" });

        Assert.False(result.Ok);
        Assert.Contains("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.Contains("edit_file", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("apply_patch", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("input file 'deck.json'", result.Content, StringComparison.Ordinal);
        string overlay = Assert.Single(RepairCopies(workspace));
        Assert.Equal(File.ReadAllBytes(scriptPath), File.ReadAllBytes(overlay));
    }

    [Fact]
    public void ARepeatedScriptFailure_DoesNotOverwriteAnExistingRepairCopy()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("existing-script-overlay");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n  File \"{scriptPath}\", line 1\nRuntimeError: writer bug\n",
            exitCode: 1);

        SkillToolResult first = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());
        Assert.False(first.Ok);
        string overlay = Assert.Single(RepairCopies(workspace));
        const string repaired = "print('my local repair')\n";
        File.WriteAllText(overlay, repaired);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Equal(repaired, File.ReadAllText(overlay));
        Assert.Single(RepairCopies(workspace));
        Assert.Contains("editable repair copy already exists", result.Content, StringComparison.Ordinal);
        Assert.Contains("was not overwritten", result.Content, StringComparison.Ordinal);
        Assert.Contains("edit_file", result.Content, StringComparison.Ordinal);
        Assert.Contains("Do not rewrite the complete file", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnprovenOverlayCollision_StagesVerifiedBytesAtAFreshPath()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("unproven-script-overlay");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        string occupied = Path.Combine(
            workspace.WorkDirectory, "skill-repairs", "tester", "scripts", "tool.py");
        Directory.CreateDirectory(Path.GetDirectoryName(occupied)!);
        const string unrelated = "print('unrelated workspace file')\n";
        File.WriteAllText(occupied, unrelated);
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n  File \"{scriptPath}\", line 1\nRuntimeError: writer bug\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Equal(unrelated, File.ReadAllText(occupied));
        string staged = Assert.Single(RepairCopies(workspace), path => path != occupied);
        Assert.Equal(File.ReadAllBytes(scriptPath), File.ReadAllBytes(staged));
        Assert.Contains(
            Path.GetRelativePath(workspace.WorkDirectory, staged).Replace('\\', '/'),
            result.Content,
            StringComparison.Ordinal);
        Assert.DoesNotContain("already exists", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void SameNamedScriptsInDifferentResourceDirectories_GetTheirOwnExactRepairCopies()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("same-basename-overlays");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string firstPath = Path.Combine(skill.RootDirectory, "first", "tool.py");
        string secondPath = Path.Combine(skill.RootDirectory, "second", "tool.py");
        Directory.CreateDirectory(Path.GetDirectoryName(firstPath)!);
        Directory.CreateDirectory(Path.GetDirectoryName(secondPath)!);
        byte[] firstSource = new byte[] { 0xef, 0xbb, 0xbf }
            .Concat(Encoding.UTF8.GetBytes("print('first')\n"))
            .ToArray();
        byte[] secondSource = Encoding.UTF8.GetBytes("print('second')\n");
        File.WriteAllBytes(firstPath, firstSource);
        File.WriteAllBytes(secondPath, secondSource);
        backend.Answer = launch => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n  File \"{launch.Argv![1]}\", line 1\nRuntimeError: writer bug\n",
            exitCode: 1);

        Assert.False(runner.Run(skill, "first/tool.py", Array.Empty<string>()).Ok);
        Assert.False(runner.Run(skill, "second/tool.py", Array.Empty<string>()).Ok);

        string[] copies = RepairCopies(workspace);
        Assert.Equal(2, copies.Length);
        Assert.Contains(copies, path => File.ReadAllBytes(path).SequenceEqual(firstSource));
        Assert.Contains(copies, path => File.ReadAllBytes(path).SequenceEqual(secondSource));
    }

    [Fact]
    public void PythonTracebackPathContainingParentheses_StillStagesTheAttributedScript()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("parenthesized-python-path");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts (dev)", "tool.py");
        Directory.CreateDirectory(Path.GetDirectoryName(scriptPath)!);
        File.WriteAllText(scriptPath, "raise RuntimeError('writer bug')\n");
        backend.Answer = launch => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n  File \"{launch.Argv![1]}\", line 1\nRuntimeError: writer bug\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(
            skill, "scripts (dev)/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Contains("A copy of this script", result.Content, StringComparison.Ordinal);
        string overlay = Assert.Single(RepairCopies(workspace));
        Assert.Equal(File.ReadAllBytes(scriptPath), File.ReadAllBytes(overlay));
    }

    [Fact]
    public void AnOrdinaryPythonSyntaxDefect_IsStagedEvenOnAnOldInterpreter()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("ordinary-python-syntax");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"  File \"{scriptPath}\", line 1\n    print(\n         ^\n"
            + "SyntaxError: '(' was never closed\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Contains("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("host's Python", result.Content, StringComparison.Ordinal);
        Assert.Single(RepairCopies(workspace));
    }

    [Fact]
    public void AFrameAppendedInsideAnExceptionMessage_CannotForgeEntryPointAttribution()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("forged-python-frame");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        string helperPath = Path.Combine(skill.RootDirectory, "scripts", "helper.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            "Traceback (most recent call last):\n"
            + $"  File \"{helperPath}\", line 8, in parse\n"
            + "ValueError: rejected workspace text follows\n"
            + $"  File \"{scriptPath}\", line 1, in <module>\n"
            + "RuntimeError: wrapper message\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.Empty(RepairCopies(workspace));
    }

    [Theory]
    [InlineData("scripts (dev)/tool.js")]
    [InlineData("scripts (dev)/tool.mjs")]
    [InlineData("scripts (dev)/tool.sh")]
    [InlineData("scripts (dev)/tool.bash")]
    public void ARelocatedNonPythonEntryPoint_IsNotStagedWithChangedSiblingSemantics(
        string relativePath)
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("non-python-repair-" + Guid.NewGuid().ToString("N"));
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill(relativePath, "broken entry point\n");
        string scriptPath = Path.Combine(
            skill.RootDirectory, relativePath.Replace('/', Path.DirectorySeparatorChar));
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            relativePath.EndsWith(".js", StringComparison.Ordinal)
                || relativePath.EndsWith(".mjs", StringComparison.Ordinal)
                ? $"Error: entry bug\n    at main ({scriptPath}:1:1)\n"
                : $"{scriptPath}: line 1: syntax error\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(skill, relativePath, Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.False(Directory.Exists(Path.Combine(workspace.WorkDirectory, "skill-repairs")));
    }

    [Fact]
    public void NodeMalformedJsonDiagnostic_PointsToTheWorkspaceSpec()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("node-invalid-json");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill("scripts/writer.mjs", "JSON.parse('{}')\n");
        const string spec = "{\n  \"title\": \"Apple M6 vs M5\",\n  \"slides\": [],\n}\n";
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "deck.json"), spec);
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            "SyntaxError: Unexpected token } in JSON at position 54\n"
            + "    at JSON.parse (<anonymous>)\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(
            skill,
            "scripts/writer.mjs",
            new[] { "--spec", "deck.json", "--out", "deck.pptx" });

        Assert.False(result.Ok);
        Assert.Contains("input file 'deck.json'", result.Content, StringComparison.Ordinal);
        Assert.Contains("with `edit_file`", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async System.Threading.Tasks.Task ConcurrentFailuresConvergeOnOneProvenRepairOverlay()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("concurrent-script-overlay");
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n  File \"{scriptPath}\", line 1\n"
            + "RuntimeError: writer bug\n",
            exitCode: 1);
        SkillScriptRunner MakeRunner() => new(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });

        System.Threading.Tasks.Task<SkillToolResult> first = System.Threading.Tasks.Task.Run(
            () => MakeRunner().Run(skill, "scripts/tool.py", Array.Empty<string>()));
        System.Threading.Tasks.Task<SkillToolResult> second = System.Threading.Tasks.Task.Run(
            () => MakeRunner().Run(skill, "scripts/tool.py", Array.Empty<string>()));
        SkillToolResult[] results = await System.Threading.Tasks.Task.WhenAll(first, second);

        Assert.All(results, result => Assert.False(result.Ok));
        Assert.Single(RepairCopies(workspace));
        Assert.Single(Directory.GetFiles(
            Path.Combine(workspace.StateDirectory, "skill-repair-overlays"),
            "*.path",
            SearchOption.TopDirectoryOnly));
        Assert.Equal(1, results.Count(result => result.Content.Contains(
            "A copy of this script is now", StringComparison.Ordinal)));
        Assert.Equal(1, results.Count(result => result.Content.Contains(
            "editable repair copy already exists", StringComparison.Ordinal)));
    }

    [Theory]
    [InlineData("usage: tool.py [-h] --spec SPEC\ntool.py: error: the following arguments are required: --spec\n")]
    [InlineData("Traceback (most recent call last):\n  File \"{script}\", line 4, in <module>\nFileNotFoundError: [Errno 2] No such file or directory: 'input.csv'\n")]
    public void ArgumentAndInputFailures_DoNotStageTheBundledScript(string diagnosticTemplate)
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("skill-input-" + Guid.NewGuid().ToString("N"));
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        string diagnostic = diagnosticTemplate.Replace("{script}", scriptPath, StringComparison.Ordinal);
        backend.Answer = _ => FakeShellBackend.Ok(string.Empty, diagnostic, exitCode: 2);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.Empty(RepairCopies(workspace));
    }

    [Fact]
    public void ATracebackWhoseDeepestFrameIsAHelper_DoesNotStageTheEntryScript()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("skill-helper-failure");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        string helperPath = Path.Combine(skill.RootDirectory, "scripts", "helper.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n"
            + $"  File \"{scriptPath}\", line 1, in <module>\n"
            + $"  File \"{helperPath}\", line 8, in build\n"
            + "RuntimeError: helper bug\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.Empty(RepairCopies(workspace));
    }

    [Theory]
    [InlineData("ValueError: invalid slide specification")]
    [InlineData("KeyError: 'title'")]
    [InlineData("PIL.UnidentifiedImageError: cannot identify image file 'cover.bin'")]
    public void WorkspaceDataExceptionsAtTheEntryPoint_DoNotStageTrustedSkillCode(
        string terminalException)
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("skill-data-" + Guid.NewGuid().ToString("N"));
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"Traceback (most recent call last):\n  File \"{scriptPath}\", line 9, in <module>\n"
            + terminalException + "\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.Empty(RepairCopies(workspace));
    }

    [Fact]
    public void FrameShapedInputTextWithoutATraceback_DoesNotStageTheEntryScript()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        SessionWorkspace workspace = Workspace("skill-echoed-frame");
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = workspace,
        });
        Skill skill = MakeSkill();
        string scriptPath = Path.Combine(skill.RootDirectory, "scripts", "tool.py");
        backend.Answer = _ => FakeShellBackend.Ok(
            string.Empty,
            $"Rejected input text: File \"{scriptPath}\", line 1\n",
            exitCode: 1);

        SkillToolResult result = runner.Run(skill, "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.DoesNotContain("A copy of this script", result.Content, StringComparison.Ordinal);
        Assert.Empty(RepairCopies(workspace));
    }

    [Fact]
    public void ABackendThatCannotStartTheScript_SaysSo()
    {
        var backend = new FakeShellBackend { Sandbox = new InProcessSandbox(Honest) };
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = Workspace("c"),
        });
        backend.Answer = _ => new ConfinedResult(
            false, false, -1, string.Empty, string.Empty, TimeSpan.Zero, "none", "the sandbox could not be prepared (no profile)");

        SkillToolResult result = runner.Run(MakeSkill(), "scripts/tool.py", Array.Empty<string>());

        Assert.False(result.Ok);
        Assert.Contains("'scripts/tool.py' was not run: the sandbox could not be prepared (no profile), and skill scripts are configured", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ABackendWhoseSandboxDoesNotConfine_IsRefusedUnderRequired()
    {
        var backend = new FakeShellBackend
        {
            Sandbox = new InProcessSandbox(new SkillSandboxCapabilities(
                ConfinesWrites: true, ConfinesNetwork: false, ConfinesHomeReads: true, BoundsProcessTree: true)),
        };
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = SkillSandboxMode.Required, Backend = backend, Workspace = Workspace("d"),
        });

        Assert.False(runner.CanRun);
        Assert.Contains("in-process", runner.UnavailableReason!, StringComparison.Ordinal);
        Assert.Contains("network access", runner.UnavailableReason!, StringComparison.Ordinal);
        SkillToolResult result = runner.Run(MakeSkill(), "scripts/tool.py", Array.Empty<string>());
        Assert.False(result.Ok);
        Assert.Empty(backend.Launches);
    }

    [Fact]
    public void TheDefaultBackend_IsTheProcessOne_UnderTheOptionsMode()
    {
        var runner = new SkillScriptRunner(new SkillScriptRunnerOptions { Sandbox = SkillSandboxMode.Off });

        var backend = Assert.IsType<ProcessShellBackend>(runner.Backend);
        Assert.Equal(SkillSandboxMode.Off, backend.Mode);
        Assert.Null(backend.Sandbox);
        Assert.Null(runner.Sandbox);
    }
}
