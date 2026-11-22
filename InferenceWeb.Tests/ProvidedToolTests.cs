// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// A tool a reachable skill provides (provides-tools + provides-call) is offered as a real
/// tool with one string parameter, and runs the declared call through skills_run.
/// </summary>
public sealed class ProvidedToolTests : IDisposable
{
    private const string Call = "skills_run(skill=\"research\", path=\"scripts/research.py\", args=[\"<question>\", \"--out\", \"notes.md\"]), then read notes.md";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-provided-" + Guid.NewGuid().ToString("N"));

    public ProvidedToolTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "research", "scripts"));
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "SKILL.md"),
            "---\nname: research\ndescription: Use for web searches and current information lookups.\nmetadata:\n"
            + "  provides-tools: WebSearch, WebFetch\n  provides-call: '" + Call + "'\n---\nRun it.\n");
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "scripts", "research.py"), "print('notes')\n");
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "plain"));
        File.WriteAllText(Path.Combine(_root, "bundled", "plain", "SKILL.md"), "---\nname: plain\ndescription: Plain.\n---\nDo it.\n");
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private SkillRegistry Registry() => new(new SkillRegistryOptions { Roots = new[] { Path.Combine(_root, "bundled") } });

    private SkillRequestPlan Plan(bool exec)
    {
        var flags = new List<string> { "--model", "x.gguf", "--skills-dir", Path.Combine(_root, "bundled") };
        if (exec) flags.Add("--skills-allow-exec");
        ServerHostingOptions options = ServerOptionsBuilder.Build(flags.ToArray(), _root);
        return Assert.IsType<SkillRequestPlan>(SkillRequestPlan.Create(
            Registry(), null, true, null, "qwen35", 32768, options, out _));
    }

    [Fact]
    public void ADeclaredProvidedToolIsARealToolOnlyWhereSkillScriptsRun()
    {
        SkillRequestPlan plan = Plan(exec: true);
        ToolFunction search = Assert.Single(plan.Tools, tool => tool.Name == "web_search");
        Assert.Equal("Use for web searches and current information lookups. (WebSearch, provided by the 'research' skill; returns notes.md.)",
            search.Description);
        Assert.Equal(new[] { "query" }, search.Required);
        ToolFunction fetch = Assert.Single(plan.Tools, tool => tool.Name == "web_fetch");
        Assert.Equal(new[] { "url" }, fetch.Required);
        Assert.True(plan.ToolContext.IsProvidedTool("web_search"));

        // Without skill scripts there is nothing to run them with, so they are not offered.
        SkillRequestPlan noExec = Plan(exec: false);
        Assert.DoesNotContain(noExec.Tools, tool => tool.Name is "web_search" or "web_fetch");
        Assert.False(noExec.ToolContext.IsProvidedTool("web_search"));

        // An identical call in one turn fetches the same pages: it is answered from its first result.
        ToolCall Search(string query) => new() { Name = "web_search", Arguments = new Dictionary<string, object?> { ["query"] = query } };
        string? key = SkillChatLoop.RepeatKey(Search("Nvidia stock"), plan);
        Assert.NotNull(key);
        Assert.Equal(key, SkillChatLoop.RepeatKey(Search("Nvidia stock"), plan));
        Assert.NotEqual(key, SkillChatLoop.RepeatKey(Search("Nvidia earnings"), plan));

        Assert.Equal("web_search", ProvidedTool.SnakeCase("WebSearch"));
        Assert.Equal("web_fetch", ProvidedTool.SnakeCase("WebFetch"));
        Assert.Equal("read_url", ProvidedTool.SnakeCase("read-URL"));
    }

    [Fact]
    public void AProvidedToolRunsTheDeclaredCallWithItsInputAndReturnsTheFileItNames()
    {
        SkillRegistry registry = Registry();
        SessionWorkspace workspace = new SessionWorkspaceManager(Path.Combine(_root, "sessions")).GetOrCreate("p");
        var runner = new WritingRunner(workspace);
        var context = new SkillToolContext(registry.Skills) { ScriptRunner = runner, Workspace = workspace };
        Assert.Equal(new[] { "web_search", "web_fetch" }, context.OfferProvidedTools(new[] { "shell" }).Select(t => t.Name));

        SkillToolResult searched = SkillTools.Execute(new ToolCall
        {
            Name = "web_search",
            Arguments = new Dictionary<string, object?> { ["query"] = "Nvidia stock" },
        }, context);
        Assert.True(searched.Ok, searched.Content);
        Assert.Equal(new[] { "Nvidia stock", "--out", "notes.md" }, runner.Runs.Single());
        Assert.Contains("notes.md:\n# Research: Nvidia stock\n", searched.Content, StringComparison.Ordinal);

        SkillToolResult fetched = SkillTools.Execute(new ToolCall
        {
            Name = "web_fetch",
            Arguments = new Dictionary<string, object?> { ["url"] = "https://example.org/a" },
        }, context);
        Assert.True(fetched.Ok, fetched.Content);
        Assert.Equal(new[] { "https://example.org/a", "--out", "notes.md" }, runner.Runs[1]);

        Assert.False(SkillTools.Execute(new ToolCall { Name = "web_search", Arguments = new Dictionary<string, object?>() }, context).Ok);
        Assert.Equal(2, runner.Runs.Count);

        // The loop answers it as the host's own tool.
        SkillTools.Partition(new[] { new ToolCall { Name = "web_search" } }, Array.Empty<ToolFunction>(),
            out var ours, out var theirs, out var unknown, context);
        Assert.Single(ours);
        Assert.Empty(unknown);
        SkillTools.Partition(new[] { new ToolCall { Name = "web_search" } }, Array.Empty<ToolFunction>(),
            out _, out _, out var withoutContext);
        Assert.Single(withoutContext);
    }

    [Fact]
    public void AProvidedToolNeverTakesANameAnotherToolHas()
    {
        var context = new SkillToolContext(Registry().Skills) { ScriptRunner = new WritingRunner(null) };
        Assert.Equal(new[] { "web_fetch" }, context.OfferProvidedTools(new[] { "web_search" }).Select(t => t.Name));
    }

    /// <summary>Records each run and writes the dossier the declared call names.</summary>
    private sealed class WritingRunner(SessionWorkspace? workspace) : ISkillScriptRunner
    {
        public List<string[]> Runs { get; } = new();

        public SkillToolResult Run(Skill skill, string relativePath, IReadOnlyList<string> arguments,
            Action<string>? onOutput = null, IReadOnlyList<string>? packages = null)
        {
            Runs.Add(arguments.ToArray());
            if (workspace != null)
                File.WriteAllText(Path.Combine(workspace.WorkDirectory, "notes.md"), "# Research: " + arguments[0] + "\n");
            return new SkillToolResult(true, "Ran " + relativePath + " (exit code 0)", skill.Id, relativePath);
        }
    }
}
