// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// A skill script, a file read or a hub install that failed is answered from its first
/// failure while nothing it depends on changed — the cycle that ran an LFM3-Omni 2B turn
/// to its round cap (analyze.py before research.py, then a read of the file that was
/// never written, over and over).
/// </summary>
public sealed class SkillRunRepeatGuardTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-repeat-guard-" + Guid.NewGuid().ToString("N"));

    public SkillRunRepeatGuardTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "find-skills"));
        File.WriteAllText(Path.Combine(_root, "bundled", "find-skills", "SKILL.md"),
            "---\nname: find-skills\ndescription: Find, install and use a hub skill.\n---\nUse skills_search.\n");
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private SkillRegistry Registry() => new(new SkillRegistryOptions
    {
        Roots = new[] { Path.Combine(_root, "bundled") },
        InstallDirectory = Path.Combine(_root, "installed"),
    });

    private SkillRequestPlan Plan(SkillRegistry registry, SessionWorkspace workspace)
    {
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "x.gguf", "--skills-dir", Path.Combine(_root, "bundled"), "--skills-allow-network" }, _root);
        return Assert.IsType<SkillRequestPlan>(SkillRequestPlan.Create(
            registry, null, true, null, "qwen35", 32768, options, out _, workspace: workspace));
    }

    private static ToolCall Call(string name, params (string Key, object Value)[] arguments) => new()
    {
        Name = name,
        Arguments = arguments.ToDictionary(pair => pair.Key, pair => (object?)pair.Value),
    };

    private const string Package = "fixture/hub@stock-report";

    [Fact]
    public void AFailedSkillRunIsAnsweredFromTheFirstFailureOnlyWhileTheWorkspaceIsUnchanged()
    {
        SkillRegistry registry = Registry();
        var manager = new SessionWorkspaceManager(Path.Combine(_root, "sessions"));
        SessionWorkspace workspace = manager.GetOrCreate("repeat");
        workspace.EnsureDirectories();
        SkillRequestPlan plan = Plan(registry, workspace);
        ToolCall run = Call("skills_run", ("skill", "find-skills"), ("path", "scripts/x.py"), ("args", "--out notes.md"));
        string? first = SkillChatLoop.RepeatKey(run, plan);
        Assert.NotNull(first);
        Assert.Equal(first, SkillChatLoop.RepeatKey(run, plan));
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "notes.md"), "now it exists");
        Assert.NotEqual(first, SkillChatLoop.RepeatKey(run, plan));
        Assert.NotNull(SkillChatLoop.RepeatKey(Call("skills_install", ("package", Package)), plan));
        Assert.NotEqual(SkillChatLoop.RepeatKey(Call("skills_install", ("package", Package)), plan),
            SkillChatLoop.RepeatKey(Call("skills_install", ("package", "other/repo@thing")), plan));
        // The same install twice in a turn (an "Already installed" answered fourteen times) is one call.
        Assert.Equal(SkillChatLoop.RepeatKey(Call("skills_install", ("package", Package)), plan),
            SkillChatLoop.RepeatKey(Call("skills_install", ("package", Package)), plan));
        // A search, and a shell command over the same workspace, are answered from their first result.
        Assert.Equal(SkillChatLoop.RepeatKey(Call("skills_search", ("query", "psi to bar")), plan),
            SkillChatLoop.RepeatKey(Call("skills_search", ("query", "psi to bar")), plan));
        Assert.NotEqual(SkillChatLoop.RepeatKey(Call("skills_search", ("query", "psi to bar")), plan),
            SkillChatLoop.RepeatKey(Call("skills_search", ("query", "unit converter")), plan));
        string? shell = SkillChatLoop.RepeatKey(Call("shell", ("command", "curl -s https://example.com | head")), plan);
        Assert.NotNull(shell);
        Assert.Equal(shell, SkillChatLoop.RepeatKey(Call("shell", ("command", "curl -s https://example.com | head ")), plan));
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "page.html"), "<html/>");
        Assert.NotEqual(shell, SkillChatLoop.RepeatKey(Call("shell", ("command", "curl -s https://example.com | head")), plan));
    }
}
