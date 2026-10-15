// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.CompilerServices;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

public sealed class FileReadVisibilityTests : IDisposable
{
    private const string Original = "def parity(n):\n    return 'even'\n";
    private const string Fixed = "def parity(n):\n    return 'even' if n % 2 == 0 else 'odd'\n";
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-read-visibility-" + Guid.NewGuid().ToString("N"));
    private readonly SessionWorkspaceManager _manager;
    private readonly SessionWorkspace _workspace;
    private readonly ShellRunner _runner;

    public FileReadVisibilityTests()
    {
        Directory.CreateDirectory(_directory);
        _manager = new SessionWorkspaceManager(Path.Combine(_directory, "workspaces"));
        _workspace = _manager.GetOrCreate("s");
        _runner = new ShellRunner(new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off });
    }

    public void Dispose()
    {
        _runner.Dispose();
        _manager.Release("s");
        Directory.Delete(_directory, recursive: true);
    }

    private string FilePath => Path.Combine(_workspace.WorkDirectory, "parity.py");
    private CodeExecResult Read(int offset = 0, int limit = 0) =>
        _runner.ReadFile(new ShellTools.ReadRequest("parity.py", offset, limit), _workspace);
    private void Write() => Assert.True(_runner.WriteFile(new ShellTools.WriteRequest("parity.py", Original), _workspace).Ok);

    [Fact]
    public void CompactionPreservesProvenanceButPartialRereadCannotRestoreFullVisibility()
    {
        Write();
        Assert.Contains("unchanged since", Read().Content);
        _workspace.Reads.InvalidateReadVisibility();

        Assert.Equal(ReadFreshness.Fresh, _workspace.Reads.Check(FilePath, Original).Freshness);
        Assert.True(_workspace.Reads.TryGetKnownText(FilePath, out string text));
        Assert.Equal(Original, text);
        Assert.Equal(ReadFreshness.Stale, _workspace.Reads.Check(FilePath, Fixed).Freshness);
        Assert.Contains("def parity(n):", Read(1, 1).Content);
        Assert.False(_workspace.Reads.CanReuseVisibleRead(FilePath, Original));

        Assert.Contains("return 'even'", Read().Content);
        Assert.Contains("unchanged since", Read().Content);
    }

    [Fact]
    public void ChangedFilePartialReadDoesNotReuseItsOldCompleteResult()
    {
        Write();
        File.WriteAllText(FilePath, Fixed);
        Assert.Contains("def parity(n):", Read(1, 1).Content);
        Assert.False(_workspace.Reads.CanReuseVisibleRead(FilePath, Fixed));
        Assert.Contains("else 'odd'", Read().Content);
        Assert.Contains("unchanged since", Read().Content);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedPatchAllowsExactRereadAndRepairWithoutDisablingOtherFileCaching(bool compacted)
    {
        Write();
        string other = Path.Combine(_workspace.WorkDirectory, "other.py");
        _workspace.Reads.Record(other, "unchanged", 1, 1, complete: true);
        if (compacted) _workspace.Reads.InvalidateReadVisibility(FilePath);

        CodeExecResult failed = _runner.ApplyPatch("*** Begin Patch\n*** Update File: parity.py\n@@\n-def parity(n): return 'even'\n+def parity(n): return 'odd'\n*** End Patch", _workspace);
        Assert.False(failed.Ok);
        Assert.Equal(Original, File.ReadAllText(FilePath));
        Assert.True(_workspace.Reads.CanReuseVisibleRead(other, "unchanged"));
        Assert.Contains("return 'even'", Read().Content);
        CodeExecResult repaired = _runner.ApplyPatch("*** Begin Patch\n*** Update File: parity.py\n@@\n-    return 'even'\n+    return 'even' if n % 2 == 0 else 'odd'\n*** End Patch", _workspace);
        Assert.True(repaired.Ok, repaired.Content);
        Assert.Equal(Fixed, File.ReadAllText(FilePath).Replace("\r\n", "\n"));
        Assert.Contains("unchanged since", Read().Content);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void NarrowMutationPreservesVisibilityOnlyWhenEarlierBodyIsStillAvailable(bool patch, bool compacted)
    {
        Write();
        if (compacted) _workspace.Reads.InvalidateReadVisibility();
        CodeExecResult result = patch
            ? _runner.ApplyPatch("*** Begin Patch\n*** Update File: parity.py\n@@\n-    return 'even'\n+    return 'odd'\n*** End Patch", _workspace)
            : _runner.EditFile(new ShellTools.EditRequest("parity.py", "'even'", "'odd'", false), _workspace);
        Assert.True(result.Ok, result.Content);
        CodeExecResult read = Read();
        if (compacted)
        {
            Assert.Contains("def parity(n):", read.Content);
            Assert.Contains("return 'odd'", read.Content);
        }
        else Assert.Contains("unchanged since", read.Content);
        Assert.Contains("unchanged since", Read().Content);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task StreamingToolLoopConsumesCompactionBeforeExecutingRead(bool compacted)
    {
        Write();
        var runner = new CodeRunnerAdapter(_runner);
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _directory } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(new[] { "--model", "x.gguf" }, _directory);
        SkillRequestPlan plan = SkillRequestPlan.Create(registry, Array.Empty<string>(), false, null,
            "nemotron_h_moe", 32768, options, out _, codeRunner: runner, workspace: _workspace);
        Assert.NotNull(plan);
        int calls = 0;
        async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
            [EnumeratorCancellation] CancellationToken cancellation)
        {
            cancellation.ThrowIfCancellationRequested();
            if (calls++ == 0)
            {
                yield return ChatStreamUpdate.Text(string.Empty) with { HistoryCompacted = compacted };
                yield return ChatStreamUpdate.Text("<think>read</think><tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"parity.py\"}}</tool_call>");
            }
            else
            {
                string result = messages.Last(message => message.Role == "tool").Content;
                Assert.Contains(compacted ? "return 'even'" : "unchanged since", result);
                yield return ChatStreamUpdate.Text("<think>done</think>Read completed.");
            }
            await Task.Yield();
            yield return new ChatStreamUpdate("", true, 0, 0, 0, 0, 0, 0, "stop");
        }
        await foreach (var _ in SkillChatLoop.RunAsync("nemotron_h_moe",
            new List<ChatMessage> { new() { Role = "user", Content = "Inspect the file." } },
            plan, true, Generate, null, CancellationToken.None)) { }
        Assert.Equal(2, calls);
        Assert.Contains("unchanged since", Read().Content);
    }

    [Fact]
    public async Task NonStreamingChildLoopInvalidatesItsOwnWorkspaceBeforeReading()
    {
        Write();
        var context = new SkillToolContext(Array.Empty<Skill>())
        { Workspace = _workspace, CodeRunner = new CodeRunnerAdapter(_runner) };
        int calls = 0;
        Task<SkillTurnOutput> Generate(List<ChatMessage> messages, List<ToolFunction>? tools, CancellationToken cancellation)
        {
            if (calls++ == 0)
                return Task.FromResult(new SkillTurnOutput(new ParsedOutput
                {
                    ToolCalls = new List<ToolCall> { new() { Name = "read_file",
                        Arguments = new Dictionary<string, object> { ["path"] = "parity.py" } } },
                }) { HistoryCompacted = true });
            Assert.Contains("return 'even'", messages.Last(message => message.Role == "tool").Content);
            return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "Read completed." }));
        }
        await SkillAgentLoop.RunAsync(new List<ChatMessage> { new() { Role = "user", Content = "Inspect." } },
            null, context, Generate);
        Assert.Equal(2, calls);
        Assert.Contains("unchanged since", Read().Content);
    }
}
