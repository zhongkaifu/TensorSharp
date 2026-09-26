using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.CodeExec;

namespace InferenceWeb.Tests;

public sealed class MultiAgentWorkspaceTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-agent-tests-" + Guid.NewGuid().ToString("N"));
    private readonly SessionWorkspaceManager _manager;
    private readonly ShellRunner _shell;
    private readonly SkillToolContext _parent;

    public MultiAgentWorkspaceTests()
    {
        _manager = new SessionWorkspaceManager(_root);
        _shell = new ShellRunner(new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off });
        _parent = new SkillToolContext([])
        {
            Workspace = _manager.GetOrCreate("parent"),
            CodeRunner = new CodeRunnerAdapter(_shell),
        };
    }

    private static ToolCall Call(string name, params (string Key, object Value)[] arguments) => new()
    {
        Id = Guid.NewGuid().ToString("N"), Name = name,
        Arguments = arguments.ToDictionary(a => a.Key, a => a.Value),
    };

    private static void Write(SessionWorkspace workspace, string path, string content) =>
        Assert.True(workspace.TryWriteFile(path, content, out string? error), error);

    private static string Read(SessionWorkspace workspace, string path)
    {
        Assert.True(workspace.TryReadFile(path, out string content, out string? error), error);
        return content;
    }

    [Fact]
    public void EachChildHasIndependentFilesAndReadToolsSeeOnlySelectedInputs()
    {
        Write(_parent.Workspace!, "src/input.txt", "parent content");
        Write(_parent.Workspace!, "unassigned.txt", "private context");
        using var first = AgentWorkspace.Create(_parent, "first", ["src/input.txt"]);
        using var second = AgentWorkspace.Create(_parent, "second", ["src/input.txt"]);

        Assert.NotEqual(first.Workspace.Root, second.Workspace.Root);
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        Assert.False(Path.GetRelativePath(home, first.Workspace.Root).StartsWith("..", StringComparison.Ordinal));
        if (!OperatingSystem.IsWindows())
            Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute,
                File.GetUnixFileMode(Path.GetDirectoryName(first.Workspace.Root)!));
        Assert.NotEqual(first.Workspace.WorkDirectory, _parent.Workspace!.WorkDirectory);
        Assert.Empty(first.Context.CodeInputFiles);
        Assert.False(File.Exists(Path.Combine(first.Workspace.WorkDirectory, "unassigned.txt")));
        SkillToolResult result = SkillTools.Execute(Call("read_file", ("path", "src/input.txt")), first.Context);
        Assert.True(result.Ok, result.Content);
        Assert.Contains("parent content", result.Content);
        Write(first.Workspace, "src/input.txt", "first change");
        Assert.Equal("parent content", Read(second.Workspace, "src/input.txt"));
        Assert.Equal("parent content", Read(_parent.Workspace, "src/input.txt"));
    }

    [Fact]
    public void ReadOnlyRunnerRejectsMutationAndCannotElevateNestedChild()
    {
        using var child = AgentWorkspace.Create(_parent, "reader");
        foreach (string name in new[] { "write_file", "write", "edit_file", "apply_patch", "apply-patch", "shell" })
        {
            Assert.False(child.AllowsTool(name));
            Assert.False(child.Context.CodeRunner!.Execute(Call(name), workspace: child.Workspace).Ok);
        }
        using var nested = AgentWorkspace.Create(child.Context, "reader/nested", mutable: true);
        Assert.True(nested.AllowsTool("read_file"));
        Assert.False(nested.AllowsTool("write_file"));
        Assert.False(nested.AllowsTool("shell"));
        Assert.Null(nested.Context.ScriptRunner);
    }

    [Fact]
    public void WorkerWritesAndPatchesOnlyItsOwnWorkspace()
    {
        using var worker = AgentWorkspace.Create(_parent, "worker", mutable: true);
        var write = SkillTools.Execute(Call("write_file", ("path", "output.txt"), ("content", "initial\n")), worker.Context);
        Assert.True(write.Ok, write.Content);
        Assert.False(File.Exists(Path.Combine(_parent.Workspace!.WorkDirectory, "output.txt")));
        Assert.True(SkillTools.Execute(Call("read_file", ("path", "output.txt")), worker.Context).Ok);
        var patch = SkillTools.Execute(Call("apply_patch", ("patch", "*** Begin Patch\n*** Update File: output.txt\n@@\n-initial\n+updated\n*** End Patch")), worker.Context);
        Assert.True(patch.Ok, patch.Content);
        Assert.Equal("updated\n", Read(worker.Workspace, "output.txt"));
        Assert.False(SkillTools.Execute(Call("write_file", ("path", "../escape.txt"), ("content", "escape")), worker.Context).Ok);
        Assert.False(worker.Context.CodeRunner!.Execute(Call("write_file", ("path", "escape.txt"), ("content", "escape")),
            workspace: _parent.Workspace).Ok);
        Assert.False(worker.AllowsTool("shell")); // An unconfined parent cannot delegate an unconfined shell.
    }

    [Fact]
    public void NestedWorkerUsesItsOwnWorkspaceWithoutLosingWritePermission()
    {
        using var parent = AgentWorkspace.Create(_parent, "worker", mutable: true);
        using var child = AgentWorkspace.Create(parent.Context, "worker/child", mutable: true);
        var write = SkillTools.Execute(Call("write_file", ("path", "nested.txt"), ("content", "child")), child.Context);
        Assert.True(write.Ok, write.Content);
        Assert.False(File.Exists(Path.Combine(parent.Workspace.WorkDirectory, "nested.txt")));
        Assert.Equal("child", Read(child.Workspace, "nested.txt"));
    }

    [Fact]
    public void AttachedInputsAreCopiedOnlyWhenSelected()
    {
        string source = Path.Combine(_root, "upload.bin");
        File.WriteAllBytes(source, [0, 1, 255]);
        var context = new SkillToolContext([]) { CodeInputFiles = [new("upload.bin", source)] };
        using var empty = AgentWorkspace.Create(context, "empty");
        using var selected = AgentWorkspace.Create(context, "selected", ["upload.bin"]);
        Assert.Empty(empty.Workspace.ListFiles());
        Assert.Equal(new byte[] { 0, 1, 255 }, File.ReadAllBytes(Path.Combine(selected.Workspace.WorkDirectory, "upload.bin")));
    }

    [Theory]
    [InlineData("../escape")]
    [InlineData("sub/../../escape")]
    [InlineData("sub/../input")]
    [InlineData("/absolute")]
    [InlineData("C:\\file")]
    [InlineData("sub\\file")]
    [InlineData("./input")]
    public void TraversalAndAbsoluteInputPathsAreRejected(string path)
    {
        Assert.Throws<ArgumentException>(() => AgentWorkspace.Create(_parent, "bad", [path]));
    }

    [Fact]
    public void SymbolicLinksAreRejectedEvenWhenTheirTargetsRemainInsideParent()
    {
        Write(_parent.Workspace!, "real/file.txt", "evidence");
        File.CreateSymbolicLink(Path.Combine(_parent.Workspace.WorkDirectory, "link.txt"), "real/file.txt");
        Directory.CreateSymbolicLink(Path.Combine(_parent.Workspace.WorkDirectory, "link-dir"), "real");
        Assert.Throws<IOException>(() => AgentWorkspace.Create(_parent, "leaf", ["link.txt"]));
        Assert.Throws<IOException>(() => AgentWorkspace.Create(_parent, "ancestor", ["link-dir/file.txt"]));
    }

    [Fact]
    public void InputCountAndByteLimitsAreEnforcedBeforeUnboundedCopies()
    {
        Assert.Throws<ArgumentException>(() => AgentWorkspace.Create(_parent, "count",
            Enumerable.Repeat("input", AgentWorkspace.MaxTransferFiles + 1).ToArray()));
        using (var stream = File.Create(Path.Combine(_parent.Workspace!.WorkDirectory, "large.bin")))
            stream.SetLength(AgentWorkspace.MaxTransferBytes + 1L);
        Assert.Throws<IOException>(() => AgentWorkspace.Create(_parent, "bytes", ["large.bin"]));
    }

    [Fact]
    public void HandoffPublishesChangedFilesAndDependenciesAsCopiesWithoutOverwritingParent()
    {
        Write(_parent.Workspace!, "input.txt", "before");
        using var producer = AgentWorkspace.Create(_parent, "producer", ["input.txt"], mutable: true);
        Write(producer.Workspace, "input.txt", "edited");
        Write(producer.Workspace, "new.txt", "result");
        var artifacts = producer.Handoff();
        Assert.Equal(2, artifacts.Count);
        Assert.Equal("before", Read(_parent.Workspace, "input.txt"));
        Assert.All(artifacts, artifact => Assert.True(File.Exists(artifact.Url)));
        Assert.Empty(producer.Handoff());
        using var consumer = AgentWorkspace.Create(_parent, "consumer");
        var imported = consumer.ImportDependencyFiles("producer", producer);
        Assert.Contains("dependencies/producer/new.txt", imported);
        Assert.Equal("edited", Read(consumer.Workspace, "dependencies/producer/input.txt"));
        Assert.Empty(consumer.Handoff()); // Prerequisite inputs are not the consumer's output.
        Write(producer.Workspace, "new.txt", "changed afterwards");
        Assert.Equal("result", Read(consumer.Workspace, "dependencies/producer/new.txt"));
    }

    [Fact]
    public void DependencyImportCollisionPreservesExplicitInputAndItsBaseline()
    {
        const string inputPath = "dependencies/totals/totals.json";
        Write(_parent.Workspace!, inputPath, "explicit input");
        using var producer = AgentWorkspace.Create(_parent, "totals", mutable: true);
        Write(producer.Workspace, "totals.json", "dependency result");
        using var consumer = AgentWorkspace.Create(_parent, "consumer", [inputPath]);

        Assert.Contains("already exists", Assert.Throws<IOException>(() =>
            consumer.ImportDependencyFiles("totals", producer)).Message);
        Assert.Equal("explicit input", Read(consumer.Workspace, inputPath));
        Assert.Equal("explicit input", Read(_parent.Workspace, inputPath));
        Assert.Empty(consumer.Handoff()); // A failed import cannot change the baseline digest.
    }

    [Fact]
    public void FullInputAllowanceDoesNotConsumeOutputAllowance()
    {
        string[] inputs = Enumerable.Range(0, AgentWorkspace.MaxTransferFiles).Select(i => $"input-{i}.txt").ToArray();
        foreach (string name in inputs) Write(_parent.Workspace!, name, "input");
        using var child = AgentWorkspace.Create(_parent, "many-inputs", inputs, mutable: true);
        Write(child.Workspace, "output.txt", "result");
        Assert.EndsWith("/output.txt", Assert.Single(child.Handoff()).Name);
    }

    [Fact]
    public void FailedOutputPreflightPublishesNoFilesOrPartialManifest()
    {
        using var child = AgentWorkspace.Create(_parent, "too-many", mutable: true);
        for (int i = 0; i <= AgentWorkspace.MaxTransferFiles; i++)
            Write(child.Workspace, $"output-{i}.txt", "output");
        Assert.Throws<IOException>(() => child.Handoff());
        Assert.Empty(_parent.Workspace!.ListFiles());
        File.Delete(Path.Combine(child.Workspace.WorkDirectory, "output-128.txt"));
        Assert.Equal(AgentWorkspace.MaxTransferFiles, child.Handoff().Count);
    }

    [Fact]
    public void BuiltInRequiredSandboxDoesNotClaimCompleteWorkspaceReadIsolation()
    {
        using var shell = new ShellRunner(new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Required });
        var runner = new CodeRunnerAdapter(shell);
        Assert.DoesNotContain(runner.DeclareWorkspaceTools(allowWrite: true), t => t.Name == "shell");
        var scripts = new SkillScriptRunner(new() { Sandbox = SkillSandboxMode.Required });
        Assert.False(scripts.CanForkForWorkspace);
        using var child = AgentWorkspace.Create(_parent, "capability-test", mutable: true);
        Assert.Null(scripts.ForkForWorkspace(child.Workspace, child.Context.CodeRunner));
    }

    [Fact]
    public void MissingParentWorkspaceCannotSilentlyDiscardProducedFiles()
    {
        using var child = AgentWorkspace.Create(new SkillToolContext([]), "no-parent-workspace");
        Assert.Empty(child.Handoff());
        Write(child.Workspace, "result.txt", "work that must be reported");
        Assert.Contains("parent has no workspace", Assert.Throws<IOException>(() => child.Handoff()).Message);
    }

    [Fact]
    public void DisposalRemovesChildWorkspaceAndDefersUntilActiveOperationsExit()
    {
        var child = AgentWorkspace.Create(_parent, "cleanup");
        string path = child.Workspace.Root;
        IDisposable operation = child.Workspace.BeginOperation();
        child.Dispose();
        Assert.True(Directory.Exists(path));
        operation.Dispose();
        Assert.False(Directory.Exists(path));
        Assert.True(Directory.Exists(_parent.Workspace!.Root));
    }

    [Fact]
    public void DeclarationPredictionMatchesActualToolsAndUnscopedScriptRunnersAreWithheld()
    {
        var context = new SkillToolContext([])
        {
            CodeRunner = _parent.CodeRunner, Workspace = _parent.Workspace,
            ScriptRunner = new SkillScriptRunner(new() { Sandbox = SkillSandboxMode.Off, Workspace = _parent.Workspace }),
        };
        foreach (bool mutable in new[] { false, true })
        {
            using var child = AgentWorkspace.Create(context, "profile", mutable: mutable);
            foreach (string tool in new[] { "skills_read", "skills_list", "skills_run", "read_file", "write_file", "apply_patch", "shell" })
                Assert.Equal(AgentWorkspace.AllowsTool(context, mutable, tool), child.AllowsTool(tool));
            Assert.Null(child.Context.ScriptRunner);
        }
    }

    public void Dispose()
    {
        _shell.Dispose();
        _manager.Release("parent");
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }
}
