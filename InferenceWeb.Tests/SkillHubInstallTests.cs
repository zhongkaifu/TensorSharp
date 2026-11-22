// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
using System.IO.Compression;
using System.Net;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Chat;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// The hub chain a model has to be able to complete in one turn — search, install, use —
/// through the shared request planner and tools, with a fixture transport in place of
/// skills.sh and GitHub. Each result must name the exact next call, because that is what a
/// small model follows.
/// </summary>
public sealed class SkillHubInstallTests : IDisposable
{
    private const string Revision = "abcdefabcdefabcdefabcdefabcdefabcdefabcd";
    private const string Package = "fixture/hub@stock-report";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-hub-install-" + Guid.NewGuid().ToString("N"));
    private readonly Hub _hub = new();
    private readonly HttpClient _http;
    private readonly SkillMarketplace _marketplace;

    public SkillHubInstallTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "find-skills"));
        File.WriteAllText(Path.Combine(_root, "bundled", "find-skills", "SKILL.md"),
            "---\nname: find-skills\ndescription: Find, install and use a hub skill.\n---\nUse skills_search.\n");
        _http = new HttpClient(_hub);
        _marketplace = new SkillMarketplace(_http, new SkillMarketplaceOptions { TemporaryDirectory = _root });
    }

    public void Dispose()
    {
        _marketplace.Dispose();
        _http.Dispose();
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private SkillRegistry Registry(SkillMarketplace? marketplace = null) => new(new SkillRegistryOptions
    {
        Roots = new[] { Path.Combine(_root, "bundled") },
        InstallDirectory = Path.Combine(_root, "installed"),
        Marketplace = marketplace ?? _marketplace,
    });

    private const string ResearchCall = "skills_run(skill=\"research\", path=\"scripts/research.py\", args=[\"<question>\"])";

    /// <summary>A bundled skill that declares the host's web search call, as TensorAgent's research skill does.</summary>
    private void BundleResearch(string description = "Use for web searches and current information lookups.")
    {
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "research", "scripts"));
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "SKILL.md"),
            "---\nname: research\ndescription: " + description + "\nmetadata:\n  provides-tools: WebSearch, WebFetch\n"
            + "  provides-call: '" + ResearchCall + "'\n---\nRun it.\n");
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "scripts", "research.py"), "print('notes')\n");
    }

    private void Bundle(string name, string description)
    {
        Directory.CreateDirectory(Path.Combine(_root, "bundled", name));
        File.WriteAllText(Path.Combine(_root, "bundled", name, "SKILL.md"),
            "---\nname: " + name + "\ndescription: " + description + "\n---\nDo it.\n");
    }

    /// <summary>A request context that can run skill scripts, as on a host with Run code on.</summary>
    private SkillToolContext RunnableContext(SkillRegistry registry, SkillMarketplace? marketplace = null) =>
        new(registry.Skills)
        {
            Marketplace = new SkillMarketplaceSession(marketplace ?? _marketplace, registry),
            ScriptRunner = new RecordingScriptRunner(),
        };

    /// <summary>A server request that names no skill: discovery on, network on, the default planner.</summary>
    private SkillRequestPlan Plan(SkillRegistry registry, bool network = true, SessionWorkspace? workspace = null)
    {
        var flags = new List<string> { "--model", "x.gguf", "--skills-dir", Path.Combine(_root, "bundled") };
        if (network) flags.Add("--skills-allow-network");
        ServerHostingOptions options = ServerOptionsBuilder.Build(flags.ToArray(), _root);
        return Assert.IsType<SkillRequestPlan>(SkillRequestPlan.Create(
            registry, null, true, null, "qwen35", 32768, options, out IReadOnlyList<string> unknown, workspace: workspace));
    }

    private static ToolCall Call(string name, params (string Key, object Value)[] arguments) => new()
    {
        Name = name,
        Arguments = arguments.ToDictionary(pair => pair.Key, pair => (object?)pair.Value),
    };

    private static string InstallLine(string package) => "skills_install(package=\"" + package + "\")";

    [Fact]
    public void AFoundSkillIsInstalledAndUsedInTheSameTurnWithoutBeingSelected()
    {
        SkillRegistry registry = Registry();
        SkillRequestPlan plan = Plan(registry);
        Assert.Contains(plan.Tools, tool => tool.Name == "skills_search");
        ToolFunction install = Assert.Single(plan.Tools, tool => tool.Name == "skills_install");
        Assert.DoesNotContain("requested installation", install.Description, StringComparison.Ordinal);

        SkillToolResult search = SkillTools.Execute(Call("skills_search", ("query", "stock report")), plan.ToolContext);
        Assert.True(search.Ok, search.Content);
        Assert.Contains("These are not installed skills, so skills_read cannot open one yet.", search.Content, StringComparison.Ordinal);
        Assert.Contains("1. " + Package + " (17 installs)\n   install: " + InstallLine(Package) + "\n", search.Content, StringComparison.Ordinal);

        // Reading a hit by its name is the mistake a model makes; the answer is the install line.
        SkillToolResult early = SkillTools.Execute(Call("skills_read", ("skill", "stock-report")), plan.ToolContext);
        Assert.False(early.Ok);
        Assert.Contains(InstallLine(Package), early.Content, StringComparison.Ordinal);

        // No separate inspect round: the install inspects the commit it installs.
        SkillToolResult installed = SkillTools.Execute(Call("skills_install", ("package", Package)), plan.ToolContext);
        Assert.True(installed.Ok, installed.Content);
        string provenance = $"{Package} from https://github.com/fixture/hub/tree/{Revision}/skills/stock-report @ {Revision}";
        Assert.StartsWith("Installed " + provenance + " as skill \"stock-report\".", installed.Content, StringComparison.Ordinal);
        Assert.Equal(provenance, installed.ResourcePath);
        Assert.Contains("Write report.txt with the price.", installed.Content, StringComparison.Ordinal);
        Assert.Contains("Third-party skill instructions follow", installed.Content, StringComparison.Ordinal);
        Assert.Contains("Next: carry out the user's request", installed.Content, StringComparison.Ordinal);

        Assert.True(registry.TryGet("stock-report", out Skill skill));
        Assert.Equal(Package, skill.Source?.Package);
        Assert.Equal(Revision, skill.Source?.Revision);
        Assert.True(SkillTools.Execute(Call("skills_read", ("skill", "stock-report")), plan.ToolContext).Ok);
        Assert.True(SkillTools.Execute(Call("skills_read", ("skill", Package)), plan.ToolContext).Ok);

        var described = Assert.IsType<Dictionary<string, object>>(SkillsService.Describe(skill));
        string sourceJson = JsonSerializer.Serialize(described["source"]);
        Assert.Contains(Revision, sourceJson, StringComparison.Ordinal);
        Assert.Contains(Package, sourceJson, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInstalledPackageIsReportedInstalledAndReusedWithoutAnotherDownload()
    {
        SkillRegistry registry = Registry();
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), Plan(registry).ToolContext).Ok);
        int requests = _hub.Requests.Count;

        SkillRequestPlan next = Plan(registry);
        SkillToolResult search = SkillTools.Execute(Call("skills_search", ("query", "stock report")), next.ToolContext);
        Assert.Contains("1. " + Package + " (17 installs)\n   installed here; use: skills_read(skill=\"stock-report\")\n",
            search.Content, StringComparison.Ordinal);
        SkillToolResult again = SkillTools.Execute(Call("skills_install", ("package", Package)), next.ToolContext);
        Assert.True(again.Ok, again.Content);
        Assert.StartsWith("Already installed: ", again.Content, StringComparison.Ordinal);
        Assert.Equal(requests + 1, _hub.Requests.Count);   // the search only
    }

    [Fact]
    public void ANameTakenByASkillThatRunsHereIsAnsweredWithThatSkillAndSaysHowToGetTheHubOne()
    {
        BundleResearch();
        _hub.Hits = """[{"name":"research","source":"acme/tools","installs":900}]""";
        _hub.Skill = ("acme/tools", "research");
        SkillRegistry registry = Registry();
        SkillToolContext context = RunnableContext(registry);
        context.UserRequest = "Please do research on Nvidia and tell me if I should buy its stock now.";

        SkillToolResult answer = SkillTools.Execute(Call("skills_install", ("package", "acme/tools@research")), context);
        Assert.False(answer.Ok);
        Assert.Contains("this host already has a skill called 'research' that runs here. Use it: " + ResearchCall + ".",
            answer.Content, StringComparison.Ordinal);
        Assert.Contains("The hub skill acme/tools@research is a different one; to install it anyway, the user can ask for "
            + "acme/tools@research by name.", answer.Content, StringComparison.Ordinal);
        Assert.False(registry.TryGet("acme-research", out _));
        Assert.DoesNotContain(_hub.Requests, uri => uri.Host is "api.github.com" or "codeload.github.com");
    }

    [Theory]
    [InlineData("Install acme/tools@research and use it.")]
    [InlineData("Use the research skill from acme/tools.")]
    public void ANameTakenByADifferentSkillInstallsUnderThePublisherQualifiedNameWhenTheUserNamesThePackage(string request)
    {
        BundleResearch();
        _hub.Hits = """[{"name":"research","source":"acme/tools","installs":900}]""";
        _hub.Skill = ("acme/tools", "research");
        SkillRegistry registry = Registry();
        SkillToolContext context = RunnableContext(registry);
        context.UserRequest = request;

        SkillToolResult installed = SkillTools.Execute(Call("skills_install", ("package", "acme/tools@research")), context);
        Assert.True(installed.Ok, installed.Content);
        Assert.Contains("as skill \"acme-research\"", installed.Content, StringComparison.Ordinal);
        Assert.Contains("already called 'research'", installed.Content, StringComparison.Ordinal);
        Assert.True(registry.TryGet("acme-research", out Skill renamed));
        Assert.Equal("research", renamed.Source?.RenamedFrom);
        Assert.StartsWith("---\nname: acme-research\n", File.ReadAllText(Path.Combine(renamed.RootDirectory, "SKILL.md")));
        Assert.True(registry.TryGet("research", out Skill local));
        Assert.Null(local.Source);
        Assert.Equal("Run it.", local.Manifest.Body.Trim());

        // A later turn that does not name it is pointed at the skill that runs here, and told the copy exists.
        SkillToolContext later = RunnableContext(registry);
        later.UserRequest = "Research Nvidia's stock.";
        SkillToolResult again = SkillTools.Execute(Call("skills_install", ("package", "acme/tools@research")), later);
        Assert.False(again.Ok);
        Assert.Contains("Use it: " + ResearchCall, again.Content, StringComparison.Ordinal);
        Assert.Contains("is a different one, installed here as 'acme-research'", again.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ANameTakenByASkillThatCannotRunHereStillInstallsUnderThePublisherQualifiedName()
    {
        // The skill already here needs a program this in-process host can never launch.
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "research", "scripts"));
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "SKILL.md"),
            "---\nname: research\ndescription: Research with a browser.\nmetadata: {\"openclaw\":{\"requires\":{\"bins\":[\"chromium\"]}}}\n---\nRun it.\n");
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "scripts", "run.sh"), "chromium --headless\n");
        _hub.Hits = """[{"name":"research","source":"acme/tools","installs":900}]""";
        _hub.Skill = ("acme/tools", "research");
        SkillRegistry registry = Registry();
        SessionWorkspace workspace = new SessionWorkspaceManager(Path.Combine(_root, "sessions")).GetOrCreate("in-process");
        var context = new SkillToolContext(registry.Skills)
        {
            Marketplace = new SkillMarketplaceSession(_marketplace, registry),
            Workspace = workspace,
            ScriptRunner = new SkillScriptRunner(new SkillScriptRunnerOptions
            {
                Backend = new FakeShellBackend { Sandbox = new InProcessSandbox(new SkillSandboxCapabilities(true, true, true, true)) },
                Workspace = workspace,
            }),
            UserRequest = "Research Nvidia.",
        };
        SkillToolResult installed = SkillTools.Execute(Call("skills_install", ("package", "acme/tools@research")), context);
        Assert.True(installed.Ok, installed.Content);
        Assert.Contains("as skill \"acme-research\"", installed.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheChatLoopGivesTheToolsTheUsersLatestMessage()
    {
        BundleResearch();
        _hub.Hits = """[{"name":"research","source":"acme/tools","installs":900}]""";
        _hub.Skill = ("acme/tools", "research");
        SkillRegistry registry = Registry();
        SkillRequestPlan plan = Plan(registry);
        int round = 0;
        async IAsyncEnumerable<ChatStreamUpdate> Generate(List<ChatMessage> messages, List<ToolFunction> tools,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
        {
            string text = round++ == 0
                ? "<tool_call>{\"name\":\"skills_install\",\"arguments\":{\"package\":\"acme/tools@research\"}}</tool_call>"
                : "Installed.";
            yield return ChatStreamUpdate.Text(text);
            await Task.Yield();
            yield return new ChatStreamUpdate("", true, 10, 20, 5, 0, 0, 0, "stop");
        }
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Research Nvidia." },
            new() { Role = "assistant", Content = "Which source?" },
            new() { Role = "user", Content = "Install acme/tools@research and use that." },
        };
        await foreach (var _ in SkillChatLoop.RunAsync("nemotron_h_moe", messages, plan, true, Generate, null, CancellationToken.None)) { }
        Assert.Equal("Install acme/tools@research and use that.", plan.ToolContext.UserRequest);
        Assert.True(registry.TryGet("acme-research", out _));
    }

    [Fact]
    public void AStaleHubEntryNamesTheNextResultAndThatCallInstalls()
    {
        _hub.Hits = """[{"name":"gone","source":"vanished/repo","installs":50},{"name":"stock-report","source":"fixture/hub","installs":10}]""";
        SkillRequestPlan plan = Plan(Registry());
        Assert.True(SkillTools.Execute(Call("skills_search", ("query", "stock report")), plan.ToolContext).Ok);

        SkillToolResult stale = SkillTools.Execute(Call("skills_install", ("package", "vanished/repo@gone")), plan.ToolContext);
        Assert.False(stale.Ok);
        Assert.Contains("HTTP 404", stale.Content, StringComparison.Ordinal);
        Assert.Contains("Next result, if it fits the task: " + InstallLine(Package) + "; otherwise", stale.Content, StringComparison.Ordinal);
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), plan.ToolContext).Ok);
    }

    [Fact]
    public void OneTurnInstallsAtMostTwoSkillsAndIsToldToUseThem()
    {
        _hub.Hits = """[{"name":"stock-report","source":"fixture/hub","installs":30},{"name":"video-tool","source":"fixture/hub","installs":20},{"name":"third","source":"fixture/hub","installs":10}]""";
        _hub.ExtraSkill = ("video-tool", "---\nname: video-tool\ndescription: Cut a video.\n---\nCut it.\n", "notes.md", "notes\n");
        SkillRequestPlan plan = Plan(Registry());
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), plan.ToolContext).Ok);
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", "fixture/hub@video-tool")), plan.ToolContext).Ok);
        SkillToolResult third = SkillTools.Execute(Call("skills_install", ("package", "fixture/hub@third")), plan.ToolContext);
        Assert.False(third.Ok);
        Assert.Contains("already installed 2 skills (stock-report, video-tool)", third.Content, StringComparison.Ordinal);
        Assert.Contains("skills_read(skill=\"video-tool\")", third.Content, StringComparison.Ordinal);
        // Reusing an installed package is not a new install.
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), plan.ToolContext).Ok);
    }

    [Fact]
    public void AnOwnerAndNameMeanThatPublishersHit()
    {
        _hub.Hits = """[{"name":"stock-report","source":"other/place","installs":90},{"name":"stock-report","source":"fixture/hub","installs":10}]""";
        SkillRequestPlan plan = Plan(Registry());
        Assert.True(SkillTools.Execute(Call("skills_search", ("query", "stock report")), plan.ToolContext).Ok);
        SkillToolResult installed = SkillTools.Execute(Call("skills_install", ("package", "fixture/stock-report")), plan.ToolContext);
        Assert.True(installed.Ok, installed.Content);
        Assert.StartsWith("Installed " + Package + " from ", installed.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOwnerFilterGivenAsOwnerAndRepositoryFiltersByTheOwner()
    {
        SkillRequestPlan plan = Plan(Registry());
        SkillToolResult search = SkillTools.Execute(Call("skills_search", ("query", "stock report"), ("owner", "fixture/hub")), plan.ToolContext);
        Assert.True(search.Ok, search.Content);
        Assert.Contains(Package, search.Content, StringComparison.Ordinal);
        Assert.Contains("owner=fixture", _hub.Requests.Last().Query, StringComparison.Ordinal);
    }

    [Fact]
    public void AToolNameTypedIntoTheShellIsCalledATool()
    {
        Assert.True(SkillTools.IsBuiltInTool("skills_run"));
        Assert.True(SkillTools.IsBuiltInTool("skills_install"));
    }

    [Fact]
    public void AnArchiveLabelledWithAnotherCommitIsRefused()
    {
        _hub.ArchiveComment = new string('9', 40);
        SkillRegistry registry = Registry();
        SkillToolResult result = SkillTools.Execute(Call("skills_install", ("package", Package)), Plan(registry).ToolContext);
        Assert.False(result.Ok);
        Assert.Contains("is not commit " + Revision, result.Content, StringComparison.Ordinal);
        Assert.False(registry.TryGet("stock-report", out _));
    }

    [Fact]
    public void WithoutSkillNetworkAccessNoHubToolIsOffered()
    {
        SkillRequestPlan plan = Plan(Registry(), network: false);
        Assert.DoesNotContain(plan.Tools, tool => SkillMarketplaceTools.IsTool(tool.Name));
        Assert.Empty(_hub.Requests);
    }

    [Fact]
    public void AHubSkillsTextNeverEntersTheSystemPrompt()
    {
        SkillRegistry registry = Registry();
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), Plan(registry).ToolContext).Ok);
        string system = Plan(registry).Prompt.Instructions;
        Assert.DoesNotContain("Fetch a quote and write a report", system, StringComparison.Ordinal);
        Assert.Contains("stock-report: third-party skill installed from the skill hub (" + Package + ")", system, StringComparison.Ordinal);

        SkillPlan selected = SkillPrompt.Plan(new[] { registry.Skills.Single(s => s.Id == "stock-report") }, Array.Empty<Skill>(),
            new SkillPromptOptions { ContextTokens = 32768, ToolsAvailable = true, InlineSelectedBodies = true });
        Assert.Empty(selected.Inlined);
        Assert.DoesNotContain("Write report.txt with the price.", selected.Instructions, StringComparison.Ordinal);
    }

    [Fact]
    public void AnotherAgentsToolsAreMappedToThisHostsCalls()
    {
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "research", "scripts"));
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "SKILL.md"),
            "---\nname: research\ndescription: Web research.\nmetadata:\n  provides-tools: WebSearch, WebFetch\n"
            + "  provides-call: 'skills_run(skill=\"research\", path=\"scripts/research.py\", args=[\"<question>\"])'\n---\nRun it.\n");
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "scripts", "research.py"), "print('notes')\n");
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "deep-dive"));
        File.WriteAllText(Path.Combine(_root, "bundled", "deep-dive", "SKILL.md"),
            "---\nname: deep-dive\ndescription: Research a company.\nallowed-tools: WebSearch, Task, skills_read\n---\nSearch the web.\n");
        SkillRegistry registry = Registry();
        var context = new SkillToolContext(registry.Skills)
        {
            Marketplace = new SkillMarketplaceSession(_marketplace, registry),
            ScriptRunner = new RecordingScriptRunner(),
        };
        SkillToolResult read = SkillTools.Execute(Call("skills_read", ("skill", "deep-dive")), context);
        Assert.True(read.Ok, read.Content);
        Assert.Contains("WebSearch -> skills_run(skill=\"research\", path=\"scripts/research.py\", args=[\"<question>\"])",
            read.Content, StringComparison.Ordinal);
        Assert.Contains("Not available here: Task.", read.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("skills_read ->", read.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInstructionsOnlyHubSkillPointsAtTheHostsWebSearchForCurrentFigures()
    {
        Directory.CreateDirectory(Path.Combine(_root, "bundled", "research", "scripts"));
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "SKILL.md"),
            "---\nname: research\ndescription: Web research.\nmetadata:\n  provides-tools: WebSearch, WebFetch\n"
            + "  provides-call: 'skills_run(skill=\"research\", path=\"scripts/research.py\", args=[\"<question>\"])'\n---\nRun it.\n");
        File.WriteAllText(Path.Combine(_root, "bundled", "research", "scripts", "research.py"), "print('notes')\n");
        _hub.Hits = """[{"name":"advice","source":"fixture/hub","installs":30}]""";
        _hub.ExtraSkill = ("advice", "---\nname: advice\ndescription: Stock advice.\n---\nWeigh valuation and momentum.\n", "notes.md", "n\n");
        SkillRegistry registry = Registry();
        var withSearch = new SkillToolContext(registry.Skills)
        {
            Marketplace = new SkillMarketplaceSession(_marketplace, registry),
            ScriptRunner = new RecordingScriptRunner(),
        };
        SkillToolResult installed = SkillTools.Execute(Call("skills_install", ("package", "fixture/hub@advice")), withSearch);
        Assert.True(installed.Ok, installed.Content);
        Assert.Contains("This skill ships no scripts and no data source", installed.Content, StringComparison.Ordinal);
        Assert.Contains("If the request needs any, get them with skills_run(skill=\"research\", path=\"scripts/research.py\"", installed.Content, StringComparison.Ordinal);
        Assert.Equal(1, CountOf(installed.Content, "no data source"));

        // A skill with scripts holds its own data path; and without a web tool the note says it cannot be checked.
        var withoutSearch = new SkillToolContext(registry.Skills) { Marketplace = new SkillMarketplaceSession(_marketplace, registry) };
        SkillToolResult read = SkillTools.Execute(Call("skills_read", ("skill", "advice")), withoutSearch);
        Assert.Contains("cannot be checked here", read.Content, StringComparison.Ordinal);

        // Running it is answered with what it is and where the figures come from.
        SkillToolResult run = SkillTools.Execute(Call("skills_run", ("skill", "advice"), ("path", "SKILL.md")), withSearch);
        Assert.False(run.Ok);
        Assert.Contains("'advice' ships no scripts, so there is nothing to run", run.Content, StringComparison.Ordinal);
        Assert.Contains("get them with skills_run(skill=\"research\"", run.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ARepositoryOverTheArchiveLimitSkipsItsOtherResultsAndNamesAnotherRepositorysHit()
    {
        using var limited = new SkillMarketplace(_http, new SkillMarketplaceOptions
        {
            TemporaryDirectory = _root,
            ArchiveLimits = new SkillArchiveLimits { MaxEntries = 16 },
        });
        // Three skills of one monorepo whose archive is over the entry limit, then one elsewhere.
        // The siblings share the word "stock" with the failed one, which used to make the next
        // of them the "next result".
        _hub.Hits = """[{"name":"stock-quote","source":"big/mono","installs":90},{"name":"stock-chart","source":"big/mono","installs":80},{"name":"stock-news","source":"big/mono","installs":70},{"name":"stock-report","source":"fixture/hub","installs":10}]""";
        _hub.OversizedRepository = "mono";
        SkillRegistry registry = Registry(limited);
        SkillToolContext context = RunnableContext(registry, limited);
        Assert.True(SkillTools.Execute(Call("skills_search", ("query", "stock quote")), context).Ok);

        SkillToolResult first = SkillTools.Execute(Call("skills_install", ("package", "big/mono@stock-quote")), context);
        Assert.False(first.Ok);
        Assert.Contains("exceeds the entry limit", first.Content, StringComparison.Ordinal);
        Assert.Contains("The other 2 results are in big/mono too, in the same repository archive, so they would fail the "
            + "same way and were skipped.", first.Content, StringComparison.Ordinal);
        Assert.Contains("Next result, if it fits the task: " + InstallLine(Package) + ";", first.Content, StringComparison.Ordinal);

        // A sibling asked for anyway is answered from this turn's failure, without fetching that archive again.
        int archives = _hub.Requests.Count(uri => uri.Host == "codeload.github.com");
        SkillToolResult sibling = SkillTools.Execute(Call("skills_install", ("package", "big/mono@stock-chart")), context);
        Assert.False(sibling.Ok);
        Assert.Contains("big/mono already failed earlier in this turn", sibling.Content, StringComparison.Ordinal);
        Assert.Contains("Next result, if it fits the task: " + InstallLine(Package) + ";", sibling.Content, StringComparison.Ordinal);
        Assert.Equal(archives, _hub.Requests.Count(uri => uri.Host == "codeload.github.com"));
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), context).Ok);

        // The memory is the turn's: a new request tries the repository again.
        archives = _hub.Requests.Count(uri => uri.Host == "codeload.github.com");
        SkillToolContext nextTurn = RunnableContext(registry, limited);
        Assert.True(SkillTools.Execute(Call("skills_search", ("query", "stock quote")), nextTurn).Ok);
        SkillToolResult retried = SkillTools.Execute(Call("skills_install", ("package", "big/mono@stock-chart")), nextTurn);
        Assert.Contains("exceeds the entry limit", retried.Content, StringComparison.Ordinal);
        Assert.Equal(archives + 1, _hub.Requests.Count(uri => uri.Host == "codeload.github.com"));
    }

    [Fact]
    public void WhenNoHubResultIsLeftTheResultNamesTheHostsWebSearch()
    {
        BundleResearch();
        _hub.Hits = """[{"name":"gone","source":"vanished/repo","installs":50}]""";
        SkillRegistry registry = Registry();
        SkillToolContext context = RunnableContext(registry);
        Assert.True(SkillTools.Execute(Call("skills_search", ("query", "stock quote")), context).Ok);
        SkillToolResult stale = SkillTools.Execute(Call("skills_install", ("package", "vanished/repo@gone")), context);
        Assert.False(stale.Ok);
        Assert.Contains("No other search result is left; search again with words for the task, or continue without a hub "
            + "skill. Web search here: " + ResearchCall + ".", stale.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ASearchListsTheInstalledSkillsThatFitBeforeTheHubsAndANameMatchOutranksAGenericWord()
    {
        BundleResearch();
        // Shares only "analysis" with the query, in its description.
        Bundle("chart-maker", "Draw a chart for any analysis.");
        Bundle("unit-converter", "Convert units of measure.");
        SkillRegistry registry = Registry();
        SkillToolContext context = RunnableContext(registry);

        SkillToolResult search = SkillTools.Execute(Call("skills_search", ("query", "Nvidia stock research analysis skill")), context);
        Assert.True(search.Ok, search.Content);
        string[] lines = search.Content.Split('\n');
        Assert.Equal("Installed here, no install needed; use one if it fits the task:", lines[0]);
        Assert.StartsWith("- research: Use for web searches and current information lookups. Use: " + ResearchCall, lines[1], StringComparison.Ordinal);
        Assert.StartsWith("- chart-maker: Draw a chart for any analysis. Use: skills_read(skill=\"chart-maker\")", lines[2], StringComparison.Ordinal);
        Assert.DoesNotContain("unit-converter", search.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("- find-skills", search.Content, StringComparison.Ordinal);   // the search itself
        Assert.Contains("1. " + Package + " (17 installs)", search.Content, StringComparison.Ordinal);
        Assert.True(search.Content.IndexOf("- research:", StringComparison.Ordinal)
            < search.Content.IndexOf("1. " + Package, StringComparison.Ordinal));
    }

    [Fact]
    public void ASearchListsAtMostThreeInstalledSkillsAndIgnoresWordsThatSayNothing()
    {
        foreach (string name in new[] { "stock-a", "stock-b", "stock-c", "stock-d" })
            Bundle(name, "Stock tool " + name + ".");
        Bundle("helper", "An agent skill for the user, with tools.");
        SkillRegistry registry = Registry();
        SkillToolContext context = RunnableContext(registry);

        SkillToolResult search = SkillTools.Execute(Call("skills_search", ("query", "stocks")), context);
        Assert.Equal(3, search.Content.Split('\n').Count(line => line.StartsWith("- stock-", StringComparison.Ordinal)));

        SkillToolResult generic = SkillTools.Execute(Call("skills_search", ("query", "an agent skill for the user with tools")), context);
        Assert.DoesNotContain("Installed here", generic.Content, StringComparison.Ordinal);

        Assert.Equal(new[] { "nvidia", "research", "searches", "searche", "search" }.ToHashSet(),
            SkillMarketplaceTools.MeaningfulWords("Nvidia: an agent's research skills & searches, to do"));
    }

    [Fact]
    public void AZeroHitSearchNamesTheHostsWebSearchAndToSearchForTheTask()
    {
        BundleResearch();
        _hub.Hits = "[]";
        SkillRegistry registry = Registry();

        // No installed skill shares a word with the query: the host's web search is named.
        SkillToolResult none = SkillTools.Execute(Call("skills_search", ("query", "Nvidia stock")), RunnableContext(registry));
        Assert.True(none.Ok, none.Content);
        Assert.Equal("No hub skill matched. Web search here: " + ResearchCall + ".\n"
            + "The hub matches what a skill does: search with words for the task, not its subject, or continue without a hub skill.",
            none.Content);

        // The research skill itself matches: it is listed once, with its call, and not named twice.
        SkillToolResult listed = SkillTools.Execute(Call("skills_search", ("query", "Nvidia stock research")), RunnableContext(registry));
        Assert.StartsWith("Installed here, no install needed; use one if it fits the task:\n- research: ", listed.Content, StringComparison.Ordinal);
        Assert.Equal(1, CountOf(listed.Content, ResearchCall));
        Assert.Contains("No hub skill matched.\nThe hub matches what a skill does", listed.Content, StringComparison.Ordinal);

        // Without skill scripts there is no call to name.
        var noScripts = new SkillToolContext(registry.Skills) { Marketplace = new SkillMarketplaceSession(_marketplace, registry) };
        SkillToolResult plain = SkillTools.Execute(Call("skills_search", ("query", "Nvidia stock")), noScripts);
        Assert.DoesNotContain("Web search here", plain.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOptionAndItsValueInOneArgumentAreSplitForTheScript()
    {
        BundleResearch();
        SkillRegistry registry = Registry();
        var runner = new RecordingScriptRunner();
        var context = new SkillToolContext(registry.Skills) { ScriptRunner = runner };
        // As a small model wrote it: the array inside a string, "--out notes.md" as one element.
        SkillToolResult ran = SkillTools.Execute(Call("skills_run", ("skill", "research"), ("path", "scripts/research.py"),
            ("args", "[\"Nvidia stock\", \"--out notes.md\"]")), context);
        Assert.True(ran.Ok, ran.Content);
        Assert.Equal(new[] { "Nvidia stock", "--out", "notes.md" }, runner.Runs.Single().Arguments);
        // The result says so, with the shape to use.
        Assert.Contains("[TensorSharp host] Each of \"--out notes.md\" was split into an option and its value, two arguments: "
            + "give each argument its own array element, as in [\"--out\",\"notes.md\"].", ran.Content, StringComparison.Ordinal);

        // Only the first whitespace after a bare option token; the rest is one value.
        Assert.Equal(new[] { "--title", "My Report" }, SkillTools.SplitJoinedOptions(new[] { "--title My Report" }));
        Assert.Equal(new[] { "-o", "x y" }, SkillTools.SplitJoinedOptions(new[] { "-o x y" }));
        // Kept whole: '=' or a quote before the space, no whitespace, not an option, a value.
        string[] whole = { "--title=A B", "--name\"x y\"", "\"--out notes.md\"", "--verbose", "-5 apples", "plain words", "--" };
        Assert.Equal(whole, SkillTools.SplitJoinedOptions(whole, out IReadOnlyList<string> joined));
        Assert.Empty(joined);
        // Nothing to split: the result is the script's own.
        runner.Runs.Clear();
        SkillToolResult plain = SkillTools.Execute(Call("skills_run", ("skill", "research"), ("path", "scripts/research.py"),
            ("args", new[] { "Nvidia stock", "--out", "notes.md" })), context);
        Assert.Equal("price=1", plain.Content);
    }

    [Fact]
    public void AShellCommandThatNamesASkillsScriptOutsideTheWorkspaceIsToldTheExactCall()
    {
        BundleResearch();
        SkillRegistry registry = Registry();
        SessionWorkspace workspace = new SessionWorkspaceManager(Path.Combine(_root, "sessions")).GetOrCreate("shell");
        var context = new SkillToolContext(registry.Skills)
        {
            ScriptRunner = new RecordingScriptRunner(),
            CodeRunner = new RefusingCodeRunner(),
            Workspace = workspace,
        };
        SkillToolResult shell = SkillTools.Execute(Call("shell", ("command", "python3 research.py \"Nvidia stock\" --out notes.md")), context);
        Assert.Contains("[TensorSharp host] research.py is not in this conversation's working directory: it is the 'research' "
            + "skill's script scripts/research.py. Run it with skills_run(skill=\"research\", path=\"scripts/research.py\", "
            + "args=[\"Nvidia stock\",\"--out\",\"notes.md\"]).", shell.Content, StringComparison.Ordinal);

        // The model's own file of that name is its own program.
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "research.py"), "print(1)\n");
        SkillToolResult own = SkillTools.Execute(Call("shell", ("command", "python3 research.py")), context);
        Assert.DoesNotContain("[TensorSharp host]", own.Content, StringComparison.Ordinal);
        Assert.False(SkillCommandTranslation.TryFindMissingSkillScript("python3 other.py", context, out _));
        Assert.False(SkillCommandTranslation.TryFindMissingSkillScript("cd x && python3 research.py", context, out _));
    }

    [Fact]
    public void AnInstructionsOnlyPackageInspectedNamesWhereCurrentFiguresComeFrom()
    {
        BundleResearch();
        _hub.Hits = """[{"name":"advice","source":"fixture/hub","installs":30}]""";
        _hub.ExtraSkill = ("advice", "---\nname: advice\ndescription: Stock advice.\n---\nWeigh valuation and momentum.\n", "notes.md", "n\n");
        SkillRegistry registry = Registry();
        SkillToolContext context = RunnableContext(registry);
        SkillToolResult advice = SkillTools.Execute(Call("skills_inspect", ("package", "fixture/hub@advice")), context);
        Assert.True(advice.Ok, advice.Content);
        Assert.Contains("This skill ships no scripts and no data source", advice.Content, StringComparison.Ordinal);
        Assert.Contains("get them with " + ResearchCall, advice.Content, StringComparison.Ordinal);
        // A package with scripts brings its own data path.
        SkillToolResult scripted = SkillTools.Execute(Call("skills_inspect", ("package", Package)), context);
        Assert.True(scripted.Ok, scripted.Content);
        Assert.DoesNotContain("no data source", scripted.Content, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("skills_inspect", "research")]
    [InlineData("skills_install", "research@skill-hub")]
    public void AnInstalledSkillsNameGivenAsAPackageIsAnsweredWithHowToUseIt(string tool, string package)
    {
        BundleResearch();
        SkillRegistry registry = Registry();
        SkillToolContext context = RunnableContext(registry);
        Assert.True(SkillTools.Execute(Call("skills_search", ("query", "stock report")), context).Ok);
        SkillToolResult result = SkillTools.Execute(Call(tool, ("package", package)), context);
        Assert.False(result.Ok);
        Assert.Contains("'research' is a skill installed here, not a hub package, so it needs no install or inspection. "
            + "Use it: " + ResearchCall + ".", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain(Package, result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AFailedImportOfASkillsScriptNamesTheCallThatRunsIt()
    {
        BundleResearch();
        SkillRegistry registry = Registry();
        var context = new SkillToolContext(registry.Skills)
        {
            ScriptRunner = new RecordingScriptRunner(),
            CodeRunner = new TracebackCodeRunner("ModuleNotFoundError: No module named 'research'"),
        };
        SkillToolResult shell = SkillTools.Execute(Call("shell", ("command", "python3 -c \"import research\"")), context);
        Assert.Contains("[TensorSharp host] research is the 'research' skill's script scripts/research.py, not a module to import. "
            + "Run it with " + ResearchCall + ".", shell.Content, StringComparison.Ordinal);

        // Another module is not a skill's script.
        var other = new SkillToolContext(registry.Skills)
        {
            ScriptRunner = new RecordingScriptRunner(),
            CodeRunner = new TracebackCodeRunner("ModuleNotFoundError: No module named 'yaml'"),
        };
        Assert.DoesNotContain("[TensorSharp host]", SkillTools.Execute(Call("shell", ("command", "python3 x.py")), other).Content, StringComparison.Ordinal);
    }

    private static int CountOf(string text, string part)
    {
        int count = 0;
        for (int at = text.IndexOf(part, StringComparison.Ordinal); at >= 0; at = text.IndexOf(part, at + part.Length, StringComparison.Ordinal))
            count++;
        return count;
    }

    [Theory]
    [InlineData("# /// script\n# requires-python = \">=3.10\"\n# dependencies = [\n#     \"yfinance>=0.2.40\",\n#     \"pandas==2.2.3\",\n#     'feedparser',\n#     \"pywin32; sys_platform == 'win32'\",\n# ]\n# ///\nimport yfinance\n",
        "yfinance|pandas==2.2.3|feedparser")]
    [InlineData("#!/usr/bin/env python3\nprint('no metadata')\n", "")]
    [InlineData("# /// script\n# dependencies = [\"rich[jupyter]>=13\", \"https://example.com/x.whl\"]\n# ///\n", "rich")]
    public void InlineScriptMetadataYieldsTheDependenciesAHostCanInstall(string script, string expected)
    {
        Assert.Equal(expected, string.Join("|", InlineScriptMetadata.Dependencies(script)));
    }

    [Fact]
    public void ASkillsOwnUvRunLineInTheShellRunsAsThatSkillsScript()
    {
        SkillRegistry registry = Registry();
        SkillToolContext installedContext = Plan(registry).ToolContext;
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), installedContext).Ok);
        var scripts = new RecordingScriptRunner();
        var code = new RefusingCodeRunner();
        var context = new SkillToolContext(registry.Skills) { ScriptRunner = scripts, CodeRunner = code };

        SkillToolResult ran = SkillTools.Execute(
            Call("shell", ("command", "uv run --with requests {baseDir}/scripts/quote.py NVDA --output json")), context);
        Assert.True(ran.Ok, ran.Content);
        Assert.StartsWith("Ran this as skills_run(skill=\"stock-report\", path=\"scripts/quote.py\")", ran.Content, StringComparison.Ordinal);
        var run = Assert.Single(scripts.Runs);
        Assert.Equal(("stock-report", "scripts/quote.py", "NVDA|--output|json", "requests"),
            (run.Skill, run.Path, string.Join("|", run.Arguments), string.Join("|", run.Packages)));
        Assert.Equal(0, code.Calls);

        // Anything beyond one simple command is the shell's.
        SkillTools.Execute(Call("shell", ("command", "uv run {baseDir}/scripts/quote.py NVDA | head")), context);
        Assert.Equal(1, code.Calls);
        Assert.Single(scripts.Runs);
    }

    [Fact]
    public void AnInProcessHostIsToldWhichNeedsItCanNeverMeet()
    {
        string skill = Path.Combine(_root, "needs");
        Directory.CreateDirectory(Path.Combine(skill, "scripts"));
        File.WriteAllText(Path.Combine(skill, "SKILL.md"),
            "---\nname: needs\ndescription: Needs things.\nmetadata: {\"clawdbot\":{\"requires\":{\"bins\":[\"uv\",\"ffmpeg\"]}}}\n---\nRun it.\n");
        File.WriteAllText(Path.Combine(skill, "scripts", "run.py"),
            "# /// script\n# dependencies = [\"pandas\", \"rich\"]\n# ///\nimport pandas\n");
        File.WriteAllText(Path.Combine(skill, "scripts", "tool.rb"), "puts 1\n");
        Assert.True(SkillManifestParser.TryParse(File.ReadAllText(Path.Combine(skill, "SKILL.md")), "needs", out SkillManifest? manifest, out _));

        var device = new SkillHostProfile
        {
            Name = "iOS", CanRunScripts = true, CanLaunchPrograms = false, InstallsPackages = true,
            PackageUnavailable = (_, package) => package == "pandas" ? "every published wheel is compiled code" : null,
        };
        SkillCompatibilityReport onDevice = SkillCompatibility.Assess(skill, manifest!, device);
        Assert.False(onDevice.CanRun);
        string verdict = onDevice.Describe(device.Name);
        Assert.StartsWith("Cannot run on iOS: ", verdict, StringComparison.Ordinal);
        Assert.Contains("`ffmpeg` program, and iOS cannot launch programs", verdict, StringComparison.Ordinal);
        Assert.DoesNotContain("`uv`", verdict, StringComparison.Ordinal);   // the host stands in for uv run
        Assert.Contains("scripts/tool.rb is a .rb script", verdict, StringComparison.Ordinal);
        Assert.Contains("Python package pandas, which cannot be used on iOS: every published wheel is compiled code", verdict, StringComparison.Ordinal);

        var desktop = new SkillHostProfile { CanRunScripts = true, CanLaunchPrograms = true, InstallsPackages = true };
        SkillCompatibilityReport onDesktop = SkillCompatibility.Assess(skill, manifest!, desktop);
        Assert.True(onDesktop.CanRun);
        Assert.Contains(onDesktop.Notes, note => note.Contains("`ffmpeg` program, which is not on this host's PATH", StringComparison.Ordinal));
        Assert.Contains(onDesktop.Notes, note => note.Contains("pandas, rich); skills_run installs them", StringComparison.Ordinal));
    }

    [Fact]
    public void AnInProcessHostRefusesToInstallASkillItCannotRunAndNamesTheNextResult()
    {
        _hub.Hits = """[{"name":"video-tool","source":"fixture/hub","installs":70},{"name":"stock-report","source":"fixture/hub","installs":10}]""";
        _hub.ExtraSkill = ("video-tool", "---\nname: video-tool\ndescription: Cut a video.\nmetadata: {\"clawdbot\":{\"requires\":{\"bins\":[\"ffmpeg\"]}}}\n---\nRun scripts/cut.py.\n",
            "scripts/cut.py", "import subprocess\nsubprocess.run(['ffmpeg'])\n");
        SkillRegistry registry = Registry();
        var manager = new SessionWorkspaceManager(Path.Combine(_root, "sessions"));
        SessionWorkspace workspace = manager.GetOrCreate("in-process");
        var context = new SkillToolContext(registry.Skills)
        {
            Marketplace = new SkillMarketplaceSession(_marketplace, registry),
            Workspace = workspace,
            ScriptRunner = new SkillScriptRunner(new SkillScriptRunnerOptions
            {
                Backend = new FakeShellBackend { Sandbox = new InProcessSandbox(new SkillSandboxCapabilities(true, true, true, true)) },
                Workspace = workspace,
            }),
        };
        Assert.True(SkillTools.Execute(Call("skills_search", ("query", "video")), context).Ok);
        SkillToolResult refused = SkillTools.Execute(Call("skills_install", ("package", "fixture/hub@video-tool")), context);
        Assert.False(refused.Ok);
        Assert.Contains("Not installed: fixture/hub@video-tool Cannot run on ", refused.Content, StringComparison.Ordinal);
        Assert.Contains("`ffmpeg` program", refused.Content, StringComparison.Ordinal);
        Assert.Contains("Next result, if it fits the task: " + InstallLine(Package) + "; otherwise", refused.Content, StringComparison.Ordinal);
        Assert.False(registry.TryGet("video-tool", out _));
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), context).Ok);
    }

    private sealed class RecordingScriptRunner : ISkillScriptRunner
    {
        public List<(string Skill, string Path, IReadOnlyList<string> Arguments, IReadOnlyList<string> Packages)> Runs { get; } = new();

        public SkillToolResult Run(Skill skill, string relativePath, IReadOnlyList<string> arguments,
            Action<string>? onOutput = null, IReadOnlyList<string>? packages = null)
        {
            Runs.Add((skill.Id, relativePath, arguments.ToArray(), packages?.ToArray() ?? Array.Empty<string>()));
            return new SkillToolResult(true, "price=1", skill.Id, relativePath);
        }
    }

    /// <summary>A shell whose every command fails with the given traceback line.</summary>
    private sealed class TracebackCodeRunner(string error) : ICodeRunner
    {
        public bool CanRun => true;
        public string? UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = "shell" };
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null, IReadOnlyList<string>? skillDirectories = null) =>
            new(false, "exit 1 (0s)\n\nTraceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n" + error + "\n", null, null);
    }

    private sealed class RefusingCodeRunner : ICodeRunner
    {
        public int Calls { get; private set; }
        public bool CanRun => true;
        public string? UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = "shell" };
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null, SessionWorkspace? workspace = null, IReadOnlyList<string>? skillDirectories = null)
        {
            Calls++;
            return new SkillToolResult(true, "shell ran", null, null);
        }
    }

    /// <summary>skills.sh and GitHub, answered from memory.</summary>
    private sealed class Hub : HttpMessageHandler
    {
        public List<Uri> Requests { get; } = new();
        public string Hits { get; set; } = """[{"name":"stock-report","source":"fixture/hub","installs":17}]""";
        public (string Source, string Name) Skill { get; set; } = ("fixture/hub", "stock-report");
        public (string Name, string Manifest, string ScriptPath, string Script)? ExtraSkill { get; set; }
        public string ArchiveComment { get; set; } = Revision;
        /// <summary>A repository whose archive holds more entries than a small limit (a skills monorepo).</summary>
        public string? OversizedRepository { get; set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Uri uri = request.RequestUri!;
            Requests.Add(uri);
            HttpResponseMessage response = uri.Host switch
            {
                "skills.sh" => Json("{\"skills\":" + Hits + "}"),
                "api.github.com" when uri.AbsolutePath.StartsWith("/repos/vanished/", StringComparison.Ordinal) =>
                    new HttpResponseMessage(HttpStatusCode.NotFound) { Content = new StringContent("{}") },
                // GitHub answers the commit id alone when asked for application/vnd.github.sha.
                "api.github.com" when request.Headers.Accept.ToString().Contains("application/vnd.github.sha", StringComparison.Ordinal) =>
                    new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(Revision + "\n") },
                "api.github.com" => Json("{\"sha\":\"" + Revision + "\"}"),
                "codeload.github.com" => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Archive(uri)) },
                _ => throw new InvalidOperationException("Unexpected request " + uri),
            };
            return Task.FromResult(response);
        }

        private static HttpResponseMessage Json(string text) =>
            new(HttpStatusCode.OK) { Content = new StringContent(text, Encoding.UTF8, "application/json") };

        private byte[] Archive(Uri uri)
        {
            string repository = uri.AbsolutePath.Split('/')[2];
            using var output = new MemoryStream();
            using (var zip = new ZipArchive(output, ZipArchiveMode.Create, leaveOpen: true))
            {
                zip.Comment = ArchiveComment;
                void Add(string path, string text)
                {
                    using var writer = new StreamWriter(zip.CreateEntry(repository + "-" + Revision + "/" + path).Open(), new UTF8Encoding(false));
                    writer.Write(text);
                }
                Add("skills/" + Skill.Name + "/SKILL.md",
                    "---\nname: " + Skill.Name + "\ndescription: Fetch a quote and write a report.\n---\n"
                    + "Run `uv run {baseDir}/scripts/quote.py TICKER`. Write report.txt with the price.\n");
                Add("skills/" + Skill.Name + "/scripts/quote.py",
                    "# /// script\n# dependencies = [\"requests\"]\n# ///\nimport sys\nprint('price=1')\n");
                if (ExtraSkill is { } extra)
                {
                    Add("skills/" + extra.Name + "/SKILL.md", extra.Manifest);
                    Add("skills/" + extra.Name + "/" + extra.ScriptPath, extra.Script);
                }
                if (repository == OversizedRepository)
                    for (int i = 0; i < 32; i++)
                        Add("docs/page-" + i.ToString(System.Globalization.CultureInfo.InvariantCulture) + ".md", "page\n");
            }
            return output.ToArray();
        }
    }
}
