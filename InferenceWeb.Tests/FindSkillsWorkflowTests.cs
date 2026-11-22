using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

public sealed class FindSkillsWorkflowTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-find-skills-" + Guid.NewGuid().ToString("N"));

    public FindSkillsWorkflowTests()
    {
        WriteSkill("find-skills");
        WriteSkill("private-skill");
    }

    private string WriteSkill(string name)
    {
        string directory = Path.Combine(_root, name);
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "SKILL.md"), $"---\nname: {name}\ndescription: {name} workflow\n---\nUse this workflow.\n");
        return directory;
    }

    private SkillRegistry Registry(bool writable = true, Func<string, bool>? policy = null) => new(new()
    {
        Roots = new[] { _root }, InstallDirectory = writable ? Path.Combine(_root, "installed") : null,
        MarketplaceHostAllowed = policy,
    });

    private SkillRequestPlan? Plan(SkillRegistry registry, bool network, string[]? requested = null, bool discovery = false)
    {
        var flags = new List<string> { "--model", "x.gguf", "--skills-dir", _root };
        if (network) flags.Add("--skills-allow-network");
        var options = ServerOptionsBuilder.Build(flags.ToArray(), _root);
        return SkillRequestPlan.Create(registry, requested ?? new[] { "find-skills" }, discovery, null,
            "gemma4", 32768, options, out _);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void SharedHostHonorsNetworkPermission(bool network, bool offered)
    {
        var plan = Assert.IsType<SkillRequestPlan>(Plan(Registry(), network));
        Assert.Equal(offered, plan.Tools.Any(t => t.Name == SkillMarketplaceTools.SearchToolName));
        Assert.Equal(offered, plan.ToolContext.Marketplace != null);
        if (!network)
        {
            var result = SkillTools.Execute(new() { Name = "skills_search", Arguments = new() { ["query"] = "testing" } }, plan.ToolContext);
            Assert.False(result.Ok);
            Assert.Contains("network", result.Content);
        }
    }

    [Fact]
    public void ExplicitSelectionAndEmptySelectionDoNotAcquireMarketplace()
    {
        var registry = Registry();
        var restricted = Assert.IsType<SkillRequestPlan>(Plan(registry, true, new[] { "private-skill" }));
        Assert.Null(restricted.ToolContext.Marketplace);
        Assert.DoesNotContain(restricted.Tools, t => SkillMarketplaceTools.IsTool(t.Name));
        var empty = Plan(registry, true, Array.Empty<string>(), discovery: true);
        Assert.True(empty == null || !empty.Tools.Any(t => SkillMarketplaceTools.IsTool(t.Name)));
    }

    [Fact]
    public void ReadOnlyRegistryOffersDiscoveryButCannotInstall()
    {
        var plan = Assert.IsType<SkillRequestPlan>(Plan(Registry(writable: false), true));
        Assert.Contains(plan.Tools, t => t.Name == "skills_search");
        Assert.Contains(plan.Tools, t => t.Name == "skills_inspect");
        Assert.DoesNotContain(plan.Tools, t => t.Name == "skills_install");
        var result = SkillTools.Execute(new() { Name = "skills_install", Arguments = new() { ["package"] = "owner/repo@demo" } }, plan.ToolContext);
        Assert.False(result.Ok);
        Assert.Contains("read-only", result.Content);
    }

    [Fact]
    public void LiveHostPolicyIsCheckedBeforeAnyNetworkOperation()
    {
        bool allowed = false;
        var plan = Assert.IsType<SkillRequestPlan>(Plan(Registry(policy: _ => allowed), true));
        foreach (string name in new[] { "skills_search", "skills_inspect", "skills_install" })
        {
            var result = SkillTools.Execute(new() { Name = name }, plan.ToolContext);
            Assert.False(result.Ok);
            Assert.Contains("network policy", result.Content);
        }
        allowed = true;
        // Once the policy allows it the call proceeds to its own validation (no network here:
        // the package is not one), which names the exact form a package takes.
        var invalid = SkillTools.Execute(new() { Name = "skills_install", Arguments = new() { ["package"] = "demo" } }, plan.ToolContext);
        Assert.False(invalid.Ok);
        Assert.Contains("owner/repository@skill-name", invalid.Content);
    }

    [Fact]
    public void InstalledGrantDoesNotExposeUnselectedSkillsOrChangeOtherRequests()
    {
        var registry = Registry();
        var current = Plan(registry, true)!.ToolContext;
        var concurrent = Plan(registry, true)!.ToolContext;
        var snapshot = current.Reachable;
        string source = WriteSkill("new-workflow");
        var skill = registry.InstallFromDirectory(source);
        current.GrantInstalled(skill);
        Assert.True(current.TryResolve("new-workflow", out _, out _));
        Assert.False(current.TryResolve("private-skill", out _, out _));
        Assert.False(concurrent.TryResolve("new-workflow", out _, out _));
        Assert.DoesNotContain(snapshot, s => s.Id == "new-workflow");
        Assert.True(registry.TryGet("new-workflow", out _));
    }

    [Fact]
    public void ClientShadowWinsAndMarketplaceCallsStayInsideHost()
    {
        var session = Plan(Registry(), true)!.ToolContext.Marketplace;
        var custom = new ToolFunction { Name = "skills_search", Description = "Client search" };
        var tools = SkillTools.Merge(new() { custom }, false, out var shadowed, session);
        Assert.Contains("skills_search", shadowed);
        Assert.Same(custom, tools.Single(t => t.Name == "skills_search"));
        SkillTools.Partition(new[] { new ToolCall { Name = "skills_search" }, new ToolCall { Name = "skills_inspect" } },
            new[] { custom }, out var builtIn, out var client, out var unknown);
        Assert.Single(client);
        Assert.Equal("skills_inspect", Assert.Single(builtIn).Name);
        Assert.Empty(unknown);
    }

    [Fact]
    public void InspectionBoundsEncodedMetadataAndFilePathsForPhoneContexts()
    {
        string oversized = new('\u4e2d', 20_000);
        var inspection = new SkillPackageInspection("owner/repo@sample", "owner/repo", new string('a', 40),
            new SkillManifest
            {
                Name = "sample", Description = oversized, RawDescription = oversized, Body = oversized,
                Compatibility = oversized, License = oversized, AllowedTools = new[] { oversized }, Warnings = new[] { oversized },
            }, Enumerable.Repeat(oversized + ".txt", 100).ToArray(), 12345);
        string output = SkillMarketplaceTools.FormatInspection(inspection, 8192);
        Assert.True(System.Text.Encoding.UTF8.GetByteCount(output) <= 8192);
        using var parsed = System.Text.Json.JsonDocument.Parse(output);
        Assert.True(parsed.RootElement.GetProperty("MetadataTruncated").GetBoolean());
        Assert.True(parsed.RootElement.GetProperty("ManifestTruncated").GetBoolean());
        Assert.True(parsed.RootElement.GetProperty("FilesTruncated").GetBoolean());
        Assert.Equal(inspection.Revision, parsed.RootElement.GetProperty("Revision").GetString());
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);
}
