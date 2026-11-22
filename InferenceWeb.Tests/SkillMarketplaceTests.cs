// Copyright (c) Zhongkai Fu. Licensed under the repository's BSD-3-Clause license.
using System.IO.Compression;
using System.Net;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class MarketplacePythonFactAttribute : FactAttribute
{
    public MarketplacePythonFactAttribute()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _))
            Skip = "Requires Python for the real discovered-skill execution test.";
    }
}

public sealed class SkillMarketplaceTests : IDisposable
{
    private const string Revision = "1234567890abcdef1234567890abcdef12345678";
    private const string Package = "example/workflows@calculate-total";
    private const string Manifest = "---\nname: calculate-total\ndescription: Calculate a total using the bundled helper.\nlicense: MIT\n---\nRun scripts/calculate.py with integer arguments.\n";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-marketplace-tests-" + Guid.NewGuid().ToString("N"));

    public SkillMarketplaceTests() => Directory.CreateDirectory(_root);
    public void Dispose() { if (Directory.Exists(_root)) Directory.Delete(_root, true); }

    private SkillRegistry Registry() => new(new SkillRegistryOptions { InstallDirectory = Path.Combine(_root, "installed") });

    private static byte[] Archive(params (string Path, string Text, int? Attributes)[] files)
    {
        using var memory = new MemoryStream();
        using (var archive = new ZipArchive(memory, ZipArchiveMode.Create, leaveOpen: true))
        {
            // GitHub labels an archive with its commit; the client checks it.
            archive.Comment = Revision;
            foreach (var file in files)
            {
                ZipArchiveEntry entry = archive.CreateEntry(file.Path, CompressionLevel.Fastest);
                if (file.Attributes is int attributes) entry.ExternalAttributes = attributes;
                using var writer = new StreamWriter(entry.Open(), new UTF8Encoding(false));
                writer.Write(file.Text);
            }
        }
        return memory.ToArray();
    }

    private static byte[] GoodArchive() => Archive(
        ("workflows-sha/skills/different-folder/SKILL.md", Manifest, null),
        ("workflows-sha/skills/different-folder/scripts/calculate.py", "import sys\nfrom helper import total\nprint('verified-total=' + str(total([int(x) for x in sys.argv[1:]])))\n", null),
        ("workflows-sha/skills/different-folder/scripts/helper.py", "def total(values):\n    return sum(values)\n", null),
        ("workflows-sha/skills/different-folder/references/example.md", "Example: 9 + 12 = 21.\n", null),
        ("workflows-sha/skills/other/SKILL.md", "---\nname: unrelated\ndescription: Another workflow.\n---\n", null));

    private static HttpResponseMessage Json(object value) => new(HttpStatusCode.OK)
    {
        Content = new StringContent(JsonSerializer.Serialize(value), Encoding.UTF8, "application/json"),
    };

    private static HttpResponseMessage Search(string name = "calculate-total") => Json(new
    {
        skills = new[] { new { id = "example/workflows/" + name, name, source = "example/workflows", installs = 123 } },
    });

    private sealed class Handler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> respond) : HttpMessageHandler
    {
        public List<Uri> Requests { get; } = new();
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token)
        {
            Requests.Add(request.RequestUri!);
            return respond(request, token);
        }
    }

    private static Handler Fixture(byte[]? archive = null) => new((request, _) =>
    {
        HttpResponseMessage response = request.RequestUri!.Host switch
        {
            "skills.sh" => Search(),
            "api.github.com" => Json(new { sha = Revision }),
            "codeload.github.com" => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(archive ?? GoodArchive()) },
            _ => throw new InvalidOperationException("Unexpected host " + request.RequestUri.Host),
        };
        return Task.FromResult(response);
    });

    private SkillMarketplaceOptions Options() => new() { TemporaryDirectory = _root };

    [Fact]
    public async Task SearchInspectAndInstallKeepResourcesAndRefreshTheCatalogWithoutAnotherDownload()
    {
        using var handler = Fixture();
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, Options());
        SkillRegistry registry = Registry();
        SkillSearchResult found = Assert.Single(await marketplace.SearchAsync("calculate total"));
        Assert.Equal(Package, found.Package);
        Assert.Equal(123, found.Installs);
        SkillPackageInspection inspected = await marketplace.InspectAsync(found.Package);
        Assert.Equal(Revision, inspected.Revision);
        Assert.Equal("calculate-total", inspected.Manifest.Name);
        Assert.Contains("scripts/helper.py", inspected.Files);
        Assert.DoesNotContain(inspected.Files, p => p.Contains("unrelated", StringComparison.Ordinal));
        SkillMarketplaceInstall installed = await marketplace.InstallAsync(inspected, registry);
        Assert.Equal(3, handler.Requests.Count);
        Assert.Equal(Revision, installed.Revision);
        Assert.True(registry.TryGet("calculate-total", out Skill skill));
        Assert.Equal(SkillOrigin.Installed, skill.Origin);
        Assert.False(registry.TryGet("unrelated", out _));
        Assert.Equal(Manifest, File.ReadAllText(Path.Combine(skill.RootDirectory, "SKILL.md")));
        SkillToolResult read = SkillTools.Execute(new ToolCall { Name = SkillTools.ReadToolName,
            Arguments = new Dictionary<string, object> { ["skill"] = skill.Id, ["path"] = "references/example.md" } },
            new SkillToolContext([skill]));
        Assert.True(read.Ok, read.Content);
        Assert.Contains("9 + 12 = 21", read.Content);
        Assert.Empty(Directory.GetDirectories(_root, "ts-marketplace-*"));
        await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.InstallAsync(inspected, registry));
        Assert.Equal(3, handler.Requests.Count);
        Assert.Equal(Manifest, File.ReadAllText(Path.Combine(skill.RootDirectory, "SKILL.md")));
    }

    [MarketplacePythonFact]
    public async Task ADiscoveredSkillRunsItsRealHelperThroughTheNormalSkillTool()
    {
        using var handler = Fixture();
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, Options());
        SkillSearchResult found = Assert.Single(await marketplace.SearchAsync("calculate"));
        SkillMarketplaceInstall installed = await marketplace.InstallAsync(found.Package, Registry());
        var manager = new SessionWorkspaceManager(Path.Combine(_root, "sessions"));
        SessionWorkspace workspace = manager.GetOrCreate("marketplace-run");
        var context = new SkillToolContext([installed.Skill])
        {
            Workspace = workspace,
            ScriptRunner = new SkillScriptRunner(new SkillScriptRunnerOptions
            {
                Workspace = workspace,
                Sandbox = SkillSandboxMode.Off,
                AllowNetwork = false,
            }),
        };
        SkillToolResult run = SkillTools.Execute(new ToolCall { Name = SkillTools.RunToolName,
            Arguments = new Dictionary<string, object> { ["skill"] = installed.Skill.Id,
                ["path"] = "scripts/calculate.py", ["args"] = new[] { "9", "12", "21" } } }, context);
        Assert.True(run.Ok, run.Content);
        Assert.Contains("verified-total=42", run.Content);
        Assert.Equal(3, handler.Requests.Count);
    }

    [Fact]
    public async Task SearchEncodesTheQueryFiltersTheOwnerAndCachesAnImmutableSortedBoundedResult()
    {
        using var handler = new Handler((request, _) => Task.FromResult(Json(new { skills = new[]
        {
            new { name = "slow", source = "example/workflows", installs = 1 },
            new { name = "fast", source = "example/workflows", installs = 9 },
            new { name = "ignored", source = "someone/workflows", installs = 100 },
        } })));
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http);
        IReadOnlyList<SkillSearchResult> first = await marketplace.SearchAsync("a&b / tasks", "EXAMPLE");
        IReadOnlyList<SkillSearchResult> cached = await marketplace.SearchAsync("a&b / tasks", "example");
        Assert.Same(first, cached);
        Assert.Equal(new[] { "fast", "slow" }, first.Select(s => s.Name));
        Assert.Single(handler.Requests);
        Assert.Contains("q=a%26b%20%2F%20tasks&limit=20&owner=example", handler.Requests[0].OriginalString);
        Assert.Throws<NotSupportedException>(() => ((IList<SkillSearchResult>)first).Add(first[0]));
    }

    [Fact]
    public async Task SearchCacheEvictionAndExpiryPermitFreshResults()
    {
        using var handler = Fixture();
        using var http = new HttpClient(handler);
        using (var bounded = new SkillMarketplace(http, new SkillMarketplaceOptions { SearchCacheEntries = 1 }))
        {
            await bounded.SearchAsync("first");
            await bounded.SearchAsync("second");
            await bounded.SearchAsync("first");
            Assert.Equal(3, handler.Requests.Count);
        }
        using var expired = new SkillMarketplace(http, new SkillMarketplaceOptions { CacheLifetime = TimeSpan.Zero });
        await expired.SearchAsync("same");
        await expired.SearchAsync("same");
        Assert.Equal(5, handler.Requests.Count);
    }

    [Fact]
    public async Task ASearchCountWithTheWrongJsonTypeDoesNotBreakOtherValidResults()
    {
        using var handler = new Handler((_, _) => Task.FromResult(Json(new
        {
            skills = new[] { new { name = "calculate-total", source = "example/workflows", installs = "unknown" } },
        })));
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http);
        Assert.Equal(0, Assert.Single(await marketplace.SearchAsync("total")).Installs);
    }

    [Theory]
    [InlineData("https://github.com/example/workflows@calculate-total")]
    [InlineData("example/workflows@calculate-total --yes")]
    [InlineData("example/../workflows@calculate-total")]
    [InlineData("example/workflows@../../escape")]
    [InlineData("example/workflows")]
    [InlineData("example/workflows@calculate-total\n")]
    [InlineData("example\n/workflows@calculate-total")]
    [InlineData("example/workflows\n@calculate-total")]
    [InlineData("example/workflows@con")]
    public async Task UnsafeAndUnsupportedPackagesAreRejectedBeforeNetwork(string package)
    {
        using var handler = Fixture();
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http);
        await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.InspectAsync(package));
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task AHostPolicyChangeClosesBothCachedAndUncachedOperations()
    {
        bool allowed = true;
        using var handler = Fixture();
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, new SkillMarketplaceOptions { IsRequestAllowed = _ => allowed });
        await marketplace.SearchAsync("total");
        SkillPackageInspection inspection = await marketplace.InspectAsync(Package);
        allowed = false;
        await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.SearchAsync("total"));
        await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.InspectAsync(Package));
        await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.InstallAsync(inspection, Registry()));
        Assert.Equal(3, handler.Requests.Count);
    }

    [Fact]
    public async Task InstallingAnInspectionAfterEvictionStillFetchesTheOriginalImmutableCommit()
    {
        using var handler = Fixture();
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, new SkillMarketplaceOptions { PackageCacheEntries = 0 });
        SkillPackageInspection inspection = await marketplace.InspectAsync(Package);
        await marketplace.InstallAsync(inspection, Registry());
        Assert.Equal(3, handler.Requests.Count);
        Assert.Single(handler.Requests, uri => uri.Host == "api.github.com");
        Assert.All(handler.Requests.Where(uri => uri.Host == "codeload.github.com"),
            uri => Assert.EndsWith("/zip/" + Revision, uri.AbsolutePath));
    }

    [Theory]
    [InlineData("../escape")]
    [InlineData("/absolute")]
    [InlineData("C:/drive")]
    [InlineData("repo/../../escape")]
    [InlineData("repo/trailing. /file")]
    [InlineData("repo/skill/report.txt:payload")]
    [InlineData("repo/skill/NUL.txt")]
    [InlineData("repo/skill/COM1")]
    public async Task RepositoryTraversalIsRejectedWithoutWritingOutsideTheInstallDirectory(string path)
    {
        using var handler = Fixture(Archive(("repo/skill/SKILL.md", Manifest, null), (path, "bad", null)));
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, Options());
        await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.InstallAsync(Package, Registry()));
        Assert.Empty(Directory.EnumerateFileSystemEntries(_root));
    }

    [Fact]
    public async Task SymlinksAndCaseCollidingEntriesAreRejectedBeforeExtraction()
    {
        foreach (byte[] archive in new[]
        {
            Archive(("repo/skill/SKILL.md", Manifest, null), ("repo/skill/link", "../../outside", unchecked((int)0xA1FF0000))),
            Archive(("repo/skill/SKILL.md", Manifest, null), ("repo/skill/link/", "../../outside", unchecked((int)0xA1FF0000))),
            Archive(("repo/skill/SKILL.md", Manifest, null), ("repo/skill/Readme", "a", null), ("repo/skill/README", "b", null)),
        })
        {
            using var handler = Fixture(archive);
            using var http = new HttpClient(handler);
            using var marketplace = new SkillMarketplace(http, Options());
            await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.InstallAsync(Package, Registry()));
        }
        Assert.Empty(Directory.EnumerateFileSystemEntries(_root));
    }

    [Fact]
    public async Task AnUnrelatedRepositorySymlinkDoesNotBlockOrGetCopiedIntoTheSelectedSkill()
    {
        byte[] archive = Archive(("repo/CLAUDE.md", "AGENTS.md", unchecked((int)0xA1FF0000)),
            ("repo/AGENTS.md", "Repository instructions.", null),
            ("repo/skill/SKILL.md", Manifest, null),
            ("repo/skill/references/instructions.md", "A retained skill reference.", null));
        using var handler = Fixture(archive);
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, Options());
        SkillMarketplaceInstall installed = await marketplace.InstallAsync(Package, Registry());
        Assert.Equal(new[] { "SKILL.md", "references/instructions.md" }, installed.Skill.Files.Select(f => f.Path));
        Assert.False(File.Exists(Path.Combine(installed.Skill.RootDirectory, "CLAUDE.md")));
        Assert.Empty(Directory.GetDirectories(_root, "ts-marketplace-*"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task MissingAndAmbiguousManifestNamesDoNotInstallAnArbitrarySkill(bool duplicate)
    {
        byte[] archive = duplicate
            ? Archive(("repo/one/SKILL.md", Manifest, null), ("repo/two/SKILL.md", Manifest, null))
            : Archive(("repo/SKILL.md", Manifest.Replace("name: calculate-total", "name: different"), null));
        using var handler = Fixture(archive);
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, Options());
        SkillInstallException error = await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.InstallAsync(Package, Registry()));
        Assert.Contains(duplicate ? "ambiguous" : "No skill", error.Message);
        Assert.Empty(Directory.EnumerateFileSystemEntries(_root));
    }

    [Fact]
    public async Task NestedExampleManifestsRemainResourcesAndDoNotMakeARealSkillAmbiguous()
    {
        byte[] archive = Archive(("repo/skill/SKILL.md", Manifest, null),
            ("repo/skill/examples/sample/SKILL.md", Manifest, null));
        using var handler = Fixture(archive);
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, Options());
        SkillMarketplaceInstall installed = await marketplace.InstallAsync(Package, Registry());
        Assert.True(File.Exists(Path.Combine(installed.Skill.RootDirectory, "examples/sample/SKILL.md")));
    }

    [Fact]
    public async Task HttpErrorsMalformedJsonAndInvalidArchivesRemainFailuresInsteadOfEmptySearchSuccess()
    {
        using var httpError = new HttpClient(new Handler((_, _) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.TooManyRequests))));
        using var unavailable = new SkillMarketplace(httpError);
        SkillInstallException rate = await Assert.ThrowsAsync<SkillInstallException>(() => unavailable.SearchAsync("testing"));
        Assert.Contains("429", rate.Message);
        using var jsonError = new HttpClient(new Handler((_, _) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            { Content = new StringContent("<html>Unavailable</html>") })));
        using var malformed = new SkillMarketplace(jsonError);
        await Assert.ThrowsAsync<SkillInstallException>(() => malformed.SearchAsync("testing"));
        using var badZip = new HttpClient(Fixture(Encoding.UTF8.GetBytes("not a zip")));
        using var corrupt = new SkillMarketplace(badZip);
        await Assert.ThrowsAsync<SkillInstallException>(() => corrupt.InspectAsync(Package));
    }

    [Fact]
    public async Task DownloadAndArchiveLimitsFailBeforeRegistryMutation()
    {
        using var handler = Fixture();
        using var http = new HttpClient(handler);
        using (var downloadBound = new SkillMarketplace(http, new SkillMarketplaceOptions { MaxArchiveBytes = 20 }))
            await Assert.ThrowsAsync<SkillInstallException>(() => downloadBound.InstallAsync(Package, Registry()));
        using (var entryBound = new SkillMarketplace(http, new SkillMarketplaceOptions
            { ArchiveLimits = new SkillArchiveLimits { MaxEntries = 1 } }))
            await Assert.ThrowsAsync<SkillInstallException>(() => entryBound.InstallAsync(Package, Registry()));
        using (var manifestBound = new SkillMarketplace(http, new SkillMarketplaceOptions { MaxManifestBytes = 10 }))
            await Assert.ThrowsAsync<SkillInstallException>(() => manifestBound.InstallAsync(Package, Registry()));
        Assert.Empty(Directory.EnumerateFileSystemEntries(_root));
    }

    [Fact]
    public async Task CancellationStopsTheFetchAndNeverCreatesAnInstallOrTemporaryDirectory()
    {
        using var cancellation = new CancellationTokenSource();
        using var handler = new Handler(async (_, token) =>
        {
            cancellation.Cancel();
            await Task.Delay(Timeout.Infinite, token);
            return Search();
        });
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, Options());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => marketplace.InstallAsync(Package, Registry(), cancellation.Token));
        Assert.Empty(Directory.EnumerateFileSystemEntries(_root));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => marketplace.SearchAsync("total", cancellationToken: cancellation.Token));
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task NetworkFailuresAndTimeoutsKeepTheirActionableErrorInsteadOfReportingNoMatches()
    {
        using var failedHttp = new HttpClient(new Handler((_, _) => throw new HttpRequestException("connection unavailable")));
        using var failed = new SkillMarketplace(failedHttp);
        Assert.Contains("connection unavailable", (await Assert.ThrowsAsync<SkillInstallException>(() => failed.SearchAsync("testing"))).Message);

        using var delayedHttp = new HttpClient(new Handler(async (_, token) =>
        {
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return Search();
        }));
        using var delayed = new SkillMarketplace(delayedHttp, new SkillMarketplaceOptions { RequestTimeout = TimeSpan.FromMilliseconds(20) });
        Assert.Contains("timed out", (await Assert.ThrowsAsync<SkillInstallException>(() => delayed.SearchAsync("testing"))).Message);
    }

    [Fact]
    public async Task ARegistryFailureStillCleansTheTemporaryExtraction()
    {
        using var handler = Fixture();
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http, Options());
        var registry = new SkillRegistry(new SkillRegistryOptions
        {
            InstallDirectory = Path.Combine(_root, "installed"),
            MaxSkills = 0,
        });
        await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.InstallAsync(Package, registry));
        Assert.Empty(Directory.GetDirectories(_root, "ts-marketplace-*"));
        Assert.Empty(registry.Skills);
    }

    [Fact]
    public async Task ARedirectIsRejectedAndTheClientDoesNotFollowItsExternalTarget()
    {
        using var handler = new Handler((_, _) =>
        {
            var response = new HttpResponseMessage(HttpStatusCode.Redirect);
            response.Headers.Location = new Uri("http://127.0.0.1/private");
            return Task.FromResult(response);
        });
        using var http = new HttpClient(handler);
        using var marketplace = new SkillMarketplace(http);
        await Assert.ThrowsAsync<SkillInstallException>(() => marketplace.SearchAsync("testing"));
        Assert.Single(handler.Requests);
    }
}
