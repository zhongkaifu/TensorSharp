using System.IO.Compression;
using System.Net;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;
using TensorSharp.Server.Skills;

namespace TensorAgent.Tests;

/// <summary>
/// Real host, registry, request planner, tool loop, in-process shell and HTTP catalog.
/// Only the remote marketplace and token producer are fixtures; model quality and
/// public network availability are separate live validations.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class FindSkillsHostTests : IDisposable
{
    private const string Package = "fixture/workflows@portable-report";
    private const string Revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ta-find-skills-" + Guid.NewGuid().ToString("N"));
    private readonly MarketplaceTransport _transport = new();
    private AgentAppHost? _host;
    private SkillMarketplace? _marketplace;
    private HttpClient? _marketplaceHttp;

    private AgentAppHost Start(bool network = true, bool scripts = true)
    {
        DirectoryInfo? repo = new(AppContext.BaseDirectory);
        while (repo != null && !Directory.Exists(Path.Combine(repo.FullName, "TensorAgent", "skills")))
            repo = repo.Parent;
        Assert.NotNull(repo);
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"))
        {
            BundledSkillsDirectory = Path.Combine(repo!.FullName, "TensorAgent", "skills"),
            ExecutionMode = AgentExecutionMode.InProcess,
        };
        paths.EnsureCreated();
        var store = new SettingsStore(paths.SettingsFile);
        AppSettings settings = store.Load();
        settings.AllowNetwork = network;
        settings.AllowCodeExecution = scripts;
        settings.MultiAgentEnabled = false;
        store.Save(settings);
        _marketplaceHttp = new HttpClient(_transport);
        _marketplace = new SkillMarketplace(_marketplaceHttp);
        _host = new AgentAppHost(paths, marketplace: _marketplace);
        return _host;
    }

    private static SkillRequestPlan Plan(AgentAppHost host, IReadOnlyList<string>? skills = null)
    {
        SkillRequestPlan plan = SkillRequestPlan.Create(
            host.Skills, skills, skills == null, new List<ToolFunction>(), "qwen35", 8192,
            host.Options, out IReadOnlyList<string> unknown, codeRunner: host.CodeRunner,
            workspace: host.Workspaces.GetOrCreate("find-skills"), multiAgent: false);
        Assert.Empty(unknown);
        return plan;
    }

    private static ToolCall Call(string name, params (string Key, object Value)[] arguments) => new()
    {
        Name = name,
        Arguments = arguments.ToDictionary(pair => pair.Key, pair => (object?)pair.Value),
    };

    [Fact]
    public async Task DiscoveredPackageInstallsAndRunsInTheSameTurnThroughTheInProcessHost()
    {
        AgentAppHost host = Start();
        SkillRequestPlan plan = Plan(host, ["find-skills"]);
        SkillRequestPlan independent = Plan(host);
        Assert.Equal("in-process", host.Backend.Name);
        Assert.NotNull(plan.ToolContext.Marketplace);
        Assert.Contains(plan.Tools, tool => tool.Name == "skills_install");
        ToolCall[] workflow =
        [
            Call("skills_read", ("skill", "find-skills"), ("path", "SKILL.md")),
            Call("skills_search", ("query", "portable arithmetic report")),
            Call("skills_inspect", ("package", Package)),
            Call("skills_install", ("package", Package), ("revision", Revision)),
            Call("skills_read", ("skill", "portable-report"), ("path", "SKILL.md")),
            Call("skills_run", ("skill", "portable-report"), ("path", "scripts/report.sh"), ("args", "6 7")),
        ];
        int round = 0;
        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            [new ChatMessage { Role = "user", Content = "Find, install and use a skill to multiply 6 by 7 and save the report." }],
            plan.Tools, plan.ToolContext,
            (messages, _, _) =>
            {
                if (messages.LastOrDefault() is { Role: "tool" } toolResult)
                    Assert.False(toolResult.Content.StartsWith("Error:", StringComparison.Ordinal), toolResult.Content);
                if (round < workflow.Length)
                    return Task.FromResult(new SkillTurnOutput(new ParsedOutput { ToolCalls = [workflow[round++]] }));
                Assert.Contains(messages, message => message.Role == "tool" && message.Content.Contains("product.txt", StringComparison.Ordinal));
                string scriptResult = messages.Last(message => message.Role == "tool").Content;
                Assert.True(scriptResult.Contains("answer=42", StringComparison.Ordinal), scriptResult);
                return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "Saved the verified arithmetic report." }));
            }, plan.LoopOptions);

        Assert.False(result.HitRoundLimit);
        Assert.Equal(workflow.Length + 1, result.Rounds);
        Assert.Equal(workflow.Length, result.Invocations.Count);
        Assert.All(result.Invocations, invocation => Assert.True(invocation.Ok, invocation.Tool));
        Assert.Equal("answer=42\n", File.ReadAllText(Path.Combine(plan.ToolContext.Workspace!.WorkDirectory, "product.txt")));
        Assert.True(host.Skills.TryGet("portable-report", out Skill installed));
        Assert.Equal(SkillOrigin.Installed, installed.Origin);
        Assert.True(File.Exists(Path.Combine(installed.RootDirectory, "references", "contract.txt")));
        Assert.StartsWith(host.Paths.InstalledSkillsDirectory, installed.RootDirectory, StringComparison.Ordinal);
        Assert.Contains(plan.ToolContext.Reachable, skill => skill.Id == "portable-report");
        Assert.DoesNotContain(independent.ToolContext.Reachable, skill => skill.Id == "portable-report");
        Assert.False(SkillTools.Execute(Call("skills_read", ("skill", "portable-report"), ("path", "SKILL.md")), independent.ToolContext).Ok);
        Assert.Contains(Plan(host).ToolContext.Reachable, skill => skill.Id == "portable-report");

        // Inspection and install share the exact downloaded archive, avoiding a second
        // network fetch or branch resolution between review and execution.
        Assert.Equal(3, _transport.Requests.Count);
        Assert.Single(_transport.Requests, uri => uri.Host == "codeload.github.com");
        host.Server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={host.Server.Token}");
        using JsonDocument catalog = JsonDocument.Parse(await client.GetStringAsync("/api/skills"));
        Assert.Contains(catalog.RootElement.GetProperty("skills").EnumerateArray(), skill => skill.GetProperty("id").GetString() == "portable-report");
        Assert.Contains("skills_run", await client.GetStringAsync("/api/skills/portable-report/files/SKILL.md"), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public void NetworkOffOrExplicitEmptySelectionCannotReachTheMarketplace(bool network, bool emptySelection)
    {
        AgentAppHost host = Start(network: network);
        SkillRequestPlan plan = Plan(host, emptySelection ? [] : ["find-skills"]);
        Assert.Null(plan.ToolContext.Marketplace);
        Assert.DoesNotContain(plan.Tools, tool => SkillMarketplaceTools.IsTool(tool.Name));
        SkillToolResult result = SkillTools.Execute(Call("skills_search", ("query", "portable report")), plan.ToolContext);
        Assert.False(result.Ok);
        Assert.Empty(_transport.Requests);
    }

    [Fact]
    public void ExistingRequestObservesNetworkSwitchAndAllowedHostsBeforeUsingCachedResults()
    {
        AgentAppHost host = Start();
        SkillRequestPlan plan = Plan(host, ["find-skills"]);
        ToolCall search = Call("skills_search", ("query", "portable report"));
        Assert.True(SkillTools.Execute(search, plan.ToolContext).Ok);
        Assert.Single(_transport.Requests);

        AppSettings settings = host.Settings.Load();
        settings.AllowNetwork = false;
        host.ApplySettings(settings);
        Assert.False(SkillTools.Execute(search, plan.ToolContext).Ok);
        settings.AllowNetwork = true;
        settings.NetworkHosts = ["example.org"];
        host.ApplySettings(settings);
        Assert.False(SkillTools.Execute(search, plan.ToolContext).Ok);
        settings.NetworkHosts = ["skills.sh"];
        host.ApplySettings(settings);
        Assert.True(SkillTools.Execute(search, plan.ToolContext).Ok);
        Assert.False(SkillTools.Execute(Call("skills_inspect", ("package", Package)), plan.ToolContext).Ok);
        Assert.Single(_transport.Requests);
    }

    [Fact]
    public void NativeDiscoveryDoesNotRequireCodeExecutionAndDoesNotEnableScripts()
    {
        AgentAppHost host = Start(scripts: false);
        SkillRequestPlan plan = Plan(host, ["find-skills"]);
        Assert.Contains(plan.Tools, tool => tool.Name == "skills_search");
        Assert.DoesNotContain(plan.Tools, tool => tool.Name == "skills_run");
        SkillToolResult inspection = SkillTools.Execute(Call("skills_inspect", ("package", Package)), plan.ToolContext);
        Assert.True(inspection.Ok, inspection.Content);
        Assert.True(SkillTools.Execute(Call("skills_install", ("package", Package)), plan.ToolContext).Ok);
        Assert.True(SkillTools.Execute(Call("skills_read", ("skill", "portable-report"), ("path", "SKILL.md")), plan.ToolContext).Ok);
        Assert.False(SkillTools.Execute(Call("skills_run", ("skill", "portable-report"), ("path", "scripts/report.sh")), plan.ToolContext).Ok);
        Assert.False(File.Exists(Path.Combine(plan.ToolContext.Workspace!.WorkDirectory, "product.txt")));
    }

    [Fact]
    public async Task UnselectedDiscoveryInstallsAFoundSkillAndRunsItInTheSameTurn()
    {
        // The user names no skill and never asks for an installation: an ordinary task.
        AgentAppHost host = Start();
        SkillRequestPlan plan = Plan(host);
        Assert.Contains(plan.Tools, tool => tool.Name == "skills_install");
        var results = new List<string>();
        ToolCall[] workflow =
        [
            Call("skills_search", ("query", "arithmetic report")),
            // A small model reads a hit as if it were installed; the answer names the install call.
            Call("skills_read", ("skill", "portable-report"), ("path", "SKILL.md")),
            Call("skills_install", ("package", Package)),
            Call("skills_run", ("skill", "portable-report"), ("path", "scripts/report.sh"), ("args", "6 7")),
        ];
        int round = 0;
        SkillLoopResult result = await SkillAgentLoop.RunAsync(
            [new ChatMessage { Role = "user", Content = "Multiply 6 by 7 and save a report of it." }],
            plan.Tools, plan.ToolContext,
            (messages, _, _) =>
            {
                if (messages.LastOrDefault() is { Role: "tool" } toolResult)
                    results.Add(toolResult.Content);
                return Task.FromResult(round < workflow.Length
                    ? new SkillTurnOutput(new ParsedOutput { ToolCalls = [workflow[round++]] })
                    : new SkillTurnOutput(new ParsedOutput { Content = "Saved the report: 6 x 7 = 42." }));
            }, plan.LoopOptions);

        Assert.False(result.HitRoundLimit);
        Assert.Contains("skills_install(package=\"" + Package + "\")", results[0], StringComparison.Ordinal);
        Assert.StartsWith("Error: 'portable-report' is a skills_search result, not an installed skill. Install it first with skills_install(package=\""
            + Package + "\")", results[1], StringComparison.Ordinal);
        Assert.StartsWith("Installed " + Package + " from https://github.com/fixture/workflows/tree/" + Revision
            + "/skills/portable-report @ " + Revision, results[2], StringComparison.Ordinal);
        Assert.Contains("Run skills_run with skill portable-report", results[2], StringComparison.Ordinal);
        Assert.Contains("answer=42", results[3], StringComparison.Ordinal);
        Assert.True(host.Skills.TryGet("portable-report", out Skill installed));
        Assert.Equal(Revision, installed.Source?.Revision);
        // Inspection happened inside the install: one search, one commit lookup, one archive.
        Assert.Equal(3, _transport.Requests.Count);
    }

    [Fact]
    public void TheInProcessHostRefusesASkillThatNeedsAProgramAndNamesTheNextResult()
    {
        AgentAppHost host = Start();
        SkillRequestPlan plan = Plan(host);
        Assert.True(SkillTools.Execute(Call("skills_search", ("query", "cut a video")), plan.ToolContext).Ok);
        SkillToolResult refused = SkillTools.Execute(Call("skills_install", ("package", "fixture/workflows@video-cutter")), plan.ToolContext);
        Assert.False(refused.Ok);
        Assert.Contains("Not installed: fixture/workflows@video-cutter Cannot run on ", refused.Content, StringComparison.Ordinal);
        Assert.Contains("It needs the `ffmpeg` program, and ", refused.Content, StringComparison.Ordinal);
        Assert.Contains("cannot launch programs", refused.Content, StringComparison.Ordinal);
        Assert.Contains("Next result, if it fits the task: skills_install(package=\"" + Package + "\"); otherwise", refused.Content, StringComparison.Ordinal);
        Assert.False(host.Skills.TryGet("video-cutter", out _));
        Assert.False(Directory.Exists(Path.Combine(host.Paths.InstalledSkillsDirectory, "video-cutter")));
    }

    [Fact]
    public void TheBundledResearchSkillIsTheDevicesDataPathAndRunsInProcess()
    {
        // On iOS the research skill is where current figures come from, so it has to be runnable
        // in the in-process runtime: its scripts import only the standard library and each other.
        AgentAppHost host = Start();
        SkillRequestPlan plan = Plan(host);
        SkillHostProfile device = SkillHostProfile.From(plan.ToolContext);
        Assert.False(device.CanLaunchPrograms);
        Assert.True(host.Skills.TryGet("research", out Skill research));
        SkillCompatibilityReport report = SkillCompatibility.Assess(research, device);
        Assert.True(report.CanRun, string.Join(" ", report.Blockers));
        Assert.Empty(report.Packages);

        string scripts = Path.Combine(research.RootDirectory, "scripts");
        var local = Directory.GetFiles(scripts, "*.py").Select(Path.GetFileNameWithoutExtension).ToHashSet(StringComparer.Ordinal);
        string[] standardLibrary =
        [
            "__future__", "argparse", "collections", "concurrent", "csv", "dataclasses", "datetime", "email", "gzip", "hashlib",
            "html", "http", "io", "itertools", "json", "math", "os", "pathlib", "re", "socket", "ssl", "string", "sys",
            "textwrap", "time", "typing", "unicodedata", "urllib", "zlib",
        ];
        foreach (string file in Directory.GetFiles(scripts, "*.py"))
        {
            foreach (string line in File.ReadLines(file))
            {
                var match = System.Text.RegularExpressions.Regex.Match(line, @"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))");
                if (!match.Success)
                    continue;
                string module = (match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value).Split('.')[0];
                Assert.True(local.Contains(module) || standardLibrary.Contains(module),
                    $"{Path.GetFileName(file)} imports '{module}', which the iOS in-process runtime may not have.");
            }
        }

        // What it declares it stands in for, and the exact call the host hands out for it.
        Assert.Equal("WebSearch, WebFetch", research.Manifest.Metadata["provides-tools"]);
        Assert.StartsWith("skills_run(skill=\"research\", path=\"scripts/research.py\", args=[\"<question>\"",
            research.Manifest.Metadata["provides-call"], StringComparison.Ordinal);
    }

    [Fact]
    public void ASearchWithTheSubjectStillLeadsToTheBundledResearchSkillOnTheInProcessHost()
    {
        // The phone's traced turns opened with skills_search("Nvidia stock research"): the hub had
        // nothing for the subject, and the bundled research skill - the device's data path - was
        // never used. The search now answers with it first, as its exact call.
        AgentAppHost host = Start();
        SkillRequestPlan plan = Plan(host);
        Assert.True(host.Skills.TryGet("research", out Skill research));
        SkillToolResult search = SkillTools.Execute(Call("skills_search", ("query", "Nvidia stock research")), plan.ToolContext);
        Assert.True(search.Ok, search.Content);
        string[] lines = search.Content.Split('\n');
        Assert.Equal("Installed here, no install needed; use one if it fits the task:", lines[0]);
        Assert.StartsWith("- research: ", lines[1], StringComparison.Ordinal);
        Assert.EndsWith(" Use: " + research.Manifest.Metadata["provides-call"].Trim(), lines[1], StringComparison.Ordinal);
        Assert.DoesNotContain("- find-skills", search.Content, StringComparison.Ordinal);
    }

    public void Dispose()
    {
        _host?.Dispose();
        _marketplace?.Dispose();
        _marketplaceHttp?.Dispose();
        CodeEnvironment.Reset();
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private sealed class MarketplaceTransport : HttpMessageHandler
    {
        public List<Uri> Requests { get; } = new();

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Uri uri = request.RequestUri!;
            Requests.Add(uri);
            HttpContent body = uri.Host switch
            {
                "skills.sh" => new StringContent("""{"skills":[{"name":"video-cutter","source":"fixture/workflows","installs":40},{"name":"portable-report","source":"fixture/workflows","installs":17}]}""", Encoding.UTF8, "application/json"),
                "api.github.com" when uri.AbsolutePath == "/repos/fixture/workflows/commits/HEAD" => new StringContent("{\"sha\":\"" + Revision + "\"}", Encoding.UTF8, "application/json"),
                "codeload.github.com" when uri.AbsolutePath == "/fixture/workflows/zip/" + Revision => new ByteArrayContent(Archive()),
                _ => throw new InvalidOperationException("Unexpected marketplace request: " + uri),
            };
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = body });
        }

        private static byte[] Archive()
        {
            using var output = new MemoryStream();
            using (var zip = new ZipArchive(output, ZipArchiveMode.Create, leaveOpen: true))
            {
                zip.Comment = Revision;   // GitHub labels an archive with its commit
                void Add(string name, string text)
                {
                    using var writer = new StreamWriter(zip.CreateEntry("workflows-" + Revision + "/skills/portable-report/" + name).Open(), new UTF8Encoding(false));
                    writer.Write(text);
                }
                Add("SKILL.md", """
                    ---
                    name: portable-report
                    description: Multiply two integer arguments and save a plain-text arithmetic report in the workspace.
                    ---
                    Run skills_run with skill portable-report, path scripts/report.sh and the two integer args.
                    Verify product.txt contains answer=<product>. The script uses the host shell and needs no Python or Node.
                    """);
                Add("scripts/report.sh", "printf 'answer=%s\\n' \"$(( $1 * $2 ))\" > product.txt\ncat product.txt\n");
                Add("references/contract.txt", "The report contains the product of both supplied integers.\n");
                void AddVideo(string name, string text)
                {
                    using var writer = new StreamWriter(zip.CreateEntry("workflows-" + Revision + "/skills/video-cutter/" + name).Open(), new UTF8Encoding(false));
                    writer.Write(text);
                }
                AddVideo("SKILL.md", """
                    ---
                    name: video-cutter
                    description: Cut a clip out of a video file with ffmpeg.
                    metadata: {"clawdbot":{"requires":{"bins":["ffmpeg"]}}}
                    ---
                    Run scripts/cut.sh with the input file and the two timestamps.
                    """);
                AddVideo("scripts/cut.sh", "ffmpeg -i \"$1\" -ss \"$2\" -to \"$3\" -c copy clip.mp4\n");
            }
            return output.ToArray();
        }
    }
}
