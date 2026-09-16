using System.Diagnostics;
using System.Net;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace RemoteAgentBench;

internal static class Program
{
    internal static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase, PropertyNameCaseInsensitive = true };

    public static async Task<int> Main(string[] args)
    {
        Options options = Options.Parse(args);
        if (!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("Run this client in the qualified WSL/Linux bubblewrap host");
        if (File.Exists(Path.Combine(options.Output, "actual-agent-workflows.json")))
            throw new IOException("Use a new output directory; existing reports are immutable");
        Directory.CreateDirectory(options.Output);
        Directory.CreateDirectory(options.ArtifactRoot);
        using var stop = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; stop.Cancel(); };
        using var fixturesDocument = JsonDocument.Parse(await File.ReadAllTextAsync(options.Fixtures, stop.Token));
        using var identityDocument = JsonDocument.Parse(await File.ReadAllTextAsync(options.Identity, stop.Token));
        if (options.ScriptedFixture)
        {
            if (!identityDocument.RootElement.TryGetProperty("scripted_fixture", out var scripted) || !scripted.GetBoolean() ||
                options.Model != "scripted-loopback-fixture")
                throw new InvalidDataException("Scripted plumbing checks require an explicitly labeled fixture identity and model");
        }
        else ValidateRemoteIdentity(identityDocument.RootElement, options.ExpectedNativeSha256);
        List<Fixture> fixtures = fixturesDocument.RootElement.GetProperty("cases").Deserialize<List<Fixture>>(Json)!;
        if (fixtures.Count == 0 || fixtures.Any(f => f.Variant is not ("original" or "distinct")) ||
            fixtures.Select(f => (f.Variant, f.Scenario, f.Trial)).Distinct().Count() != fixtures.Count)
            throw new InvalidDataException("Missing or duplicate fixture case identities");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = [options.SkillsRoot] });
        if (registry.Errors.Count > 0 || !registry.TryGet("release-validation", out _))
            throw new InvalidDataException("The original release-validation skill is absent or failed to load");
        var store = new CodeArtifactStore(options.ArtifactRoot);
        await using var artifactServer = new ArtifactServer(store, options.ArtifactPort);
        artifactServer.Start();
        using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        var workspaces = new SessionWorkspaceManager(Path.Combine(options.Output, "workspaces"));
        var cases = new List<CaseResult>();
        var report = new Dictionary<string, object?>
        {
            ["format_version"] = 1, ["topology"] = options.ScriptedFixture
                ? "Scripted loopback responses; real local AgentHost tools and required bubblewrap; no model or native binding proof"
                : "WSL AgentHost tools and required bubblewrap; remote TensorSharp model over loopback tunnel",
            ["scripted_fixture"] = options.ScriptedFixture,
            ["event_source"] = "local SkillAgentLoop invocations, not Web UI SSE",
            ["endpoint"] = options.Endpoint, ["model"] = options.Model, ["thinking"] = options.Thinking,
            ["started_at_unix"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
            ["fixture_sha256"] = Hash(options.Fixtures), ["fixture_source"] = fixturesDocument.RootElement.Clone(),
            ["remote_identity"] = identityDocument.RootElement.Clone(), ["remote_identity_sha256"] = Hash(options.Identity),
            ["remote_binding_scope"] = options.ScriptedFixture ? "None: explicitly synthetic fixture identity"
                : "Supplied root-stage /proc evidence, not observed by this client; post-run remote identity check remains required",
            ["managed_files_sha256"] = Directory.EnumerateFiles(AppContext.BaseDirectory, "*.dll").ToDictionary(path => Path.GetFileName(path)!, Hash),
            ["skill_files_sha256"] = Directory.EnumerateFiles(options.SkillsRoot, "*", SearchOption.AllDirectories).ToDictionary(p => Path.GetRelativePath(options.SkillsRoot, p), Hash),
            ["sandbox_required"] = true, ["max_rounds"] = options.MaxRounds, ["max_tokens"] = options.MaxTokens,
            ["cases"] = cases, ["run_complete"] = false, ["release_qualified"] = false,
            ["independent_code_verification"] = "Pending verify-agent-code-artifacts.py in a separate required sandbox",
        };
        await Save(report, options.Output);
        foreach (var wave in fixtures.GroupBy(f => f.Wave))
        {
            if (wave.Any(f => f.Concurrency is not (1 or 4)) || wave.Count() != wave.First().Concurrency)
                throw new InvalidDataException("Fixture wave must contain exactly its declared 1 or 4 cases");
            var completed = await Task.WhenAll(wave.Select(f => Run(f, options, registry, store, workspaces, artifactServer, http, stop.Token)));
            cases.AddRange(completed);
            await Save(report, options.Output);
            Console.WriteLine($"{wave.Key}: {string.Join(", ", completed.Select(c => c.Status))}");
        }
        report["run_complete"] = true;
        report["finished_at_unix"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
        report["workflow_status"] = cases.All(c => c.Status == "ok") ? "ok" : "failed";
        await Save(report, options.Output);
        return cases.All(c => c.Status == "ok") ? 0 : 1;
    }

    private static async Task<CaseResult> Run(Fixture fixture, Options options, SkillRegistry registry, CodeArtifactStore store,
        SessionWorkspaceManager workspaces, ArtifactServer artifactServer, HttpClient http, CancellationToken stop)
    {
        var result = new CaseResult { Variant = fixture.Variant, Scenario = fixture.Scenario, Trial = fixture.Trial, Spec = fixture.Spec.Clone() };
        var watch = Stopwatch.StartNew();
        string session = "remote-agent-" + Guid.NewGuid().ToString("N");
        var workspace = workspaces.GetOrCreate(session);
        result.Session = session;
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(stop);
        timeout.CancelAfter(TimeSpan.FromSeconds(options.TimeoutSeconds));
        try
        {
            var codeOptions = new CodeExecOptions
            {
                Enabled = true, Sandbox = SkillSandboxMode.Required, AllowNetwork = false, AllowInstall = false,
                Timeout = TimeSpan.FromSeconds(120), MaxTimeout = TimeSpan.FromSeconds(600),
                ScratchDirectory = Path.Combine(options.Output, "scratch"), ArtifactDirectory = options.ArtifactRoot,
                ArtifactUriPrefix = artifactServer.Prefix,
            };
            using var shell = new ShellRunner(codeOptions, artifacts: store);
            var adapter = new CodeRunnerAdapter(shell, codeOptions, onCompleted: completed => result.CodeResults.Add(completed));
            var script = new SkillScriptRunner(new SkillScriptRunnerOptions
            {
                Sandbox = SkillSandboxMode.Required, AllowNetwork = false, Workspace = workspace, Backend = adapter.Backend,
                Timeout = TimeSpan.FromSeconds(60),
            });
            var sandbox = script.Sandbox;
            result.Sandbox = new { name = sandbox?.Name, capabilities = sandbox?.Capabilities, script.CanRun,
                shellCanRun = shell.CanRun, script.UnavailableReason, shellUnavailableReason = shell.UnavailableReason };
            if (!script.CanRun || !shell.CanRun || sandbox?.Name != "bubblewrap" ||
                !sandbox.Capabilities.ConfinesWrites || !sandbox.Capabilities.ConfinesNetwork || !sandbox.Capabilities.ConfinesHomeReads)
                throw new InvalidOperationException("Required WSL bubblewrap confinement is unavailable");
            var selectedIds = fixture.Spec.TryGetProperty("skills", out var skills)
                ? skills.EnumerateArray().Select(value => value.GetString()!).ToArray() : Array.Empty<string>();
            var selected = registry.Resolve(selectedIds, out var unknown);
            if (unknown.Count != 0) throw new InvalidDataException("Unknown selected skill");
            var plan = SkillPrompt.Plan(selected, registry.Skills, new SkillPromptOptions { ToolsAvailable = true, ContextTokens = 32768 });
            List<ToolFunction> tools = SkillTools.Merge(null, allowScripts: true, out _);
            tools.AddRange(adapter.DeclareTools(persists: true));
            var messages = SkillPrompt.Apply([new ChatMessage { Role = "user", Content = fixture.Spec.GetProperty("prompt").GetString()! }], plan);
            messages = SkillPrompt.Apply(messages, CodePrompt.Block(fileTools: true, hasPatch: true));
            var context = new SkillToolContext(plan.Reachable.ToList()) { ScriptRunner = script, CodeRunner = adapter, Workspace = workspace };
            var transport = new OpenAiTurn(http, options, result);
            var loop = await SkillAgentLoop.RunAsync(messages, tools, context, transport.Generate,
                new SkillAgentLoopOptions
                {
                    MaxRounds = options.MaxRounds, MaxCallsPerRound = 8, ToolResultsAreRendered = true,
                    // These declarations belong to THIS host. Passing them as client
                    // tools here would prevent the local AgentHost from executing them.
                    ClientTools = Array.Empty<ToolFunction>(),
                    OnInvocation = invocation => result.Events.Add(new
                    {
                        skill_step = invocation.Tool, ok = invocation.Ok, round = invocation.Round,
                        skill = invocation.SkillId, path = invocation.ResourcePath, bytes = invocation.ResultBytes,
                        at_unix_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                        files = invocation.Files.Select(file => new { name = file.Name, url = file.Url, bytes = file.Bytes }).ToArray(),
                    }),
                }, timeout.Token);
            result.Messages = loop.Messages;
            result.Answer = loop.Output.Parsed.Content;
            result.Rounds = loop.Rounds;
            if (loop.HitRoundLimit || loop.PendingClientToolCalls.Count > 0 || loop.Output.Parsed.ToolCalls is { Count: > 0 })
                throw new InvalidDataException("Agent did not terminate with a completed answer within the round limit");
            var succeeded = loop.Invocations.Where(call => call.Ok).Select(call => call.Tool).ToHashSet(StringComparer.Ordinal);
            result.SuccessfulTools = succeeded.Order().ToArray();
            var required = fixture.Spec.GetProperty("tools").EnumerateArray().Select(value => value.GetString()!).ToHashSet();
            if (!required.IsSubsetOf(succeeded)) throw new InvalidDataException("Missing successful real tool calls: " + string.Join(",", required.Except(succeeded)));
            if (fixture.Spec.TryGetProperty("expected", out var expected) && result.Answer.Trim() != expected.GetString())
                throw new InvalidDataException("Final answer differs from independent fixture value");
            if (fixture.Spec.TryGetProperty("artifact", out var expectedArtifact))
            {
                var versions = loop.Invocations.SelectMany(call => call.Files).Where(file => Path.GetFileName(file.Name) == "result.json").ToArray();
                result.ResultArtifactVersions = versions;
                if (versions.Length == 0) throw new InvalidDataException("No result.json artifact was advertised");
                // Independent HTTP fetch of the latest immutable artifact version;
                // model prose and model-written tests_passed alone are not sufficient.
                byte[] bytes = await http.GetByteArrayAsync(versions[^1].Url, timeout.Token);
                using var artifact = JsonDocument.Parse(bytes);
                result.Artifacts.Add(new { name = "result.json", url = versions[^1].Url, sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)), content = artifact.RootElement.Clone() });
                if (!System.Text.Json.Nodes.JsonNode.DeepEquals(System.Text.Json.Nodes.JsonNode.Parse(expectedArtifact.GetRawText()), System.Text.Json.Nodes.JsonNode.Parse(bytes)))
                    throw new InvalidDataException("Downloaded final result artifact failed the original oracle");
            }
            result.Status = "ok";
        }
        catch (Exception error)
        {
            result.Detail = error.ToString();
        }
        finally
        {
            result.WallSeconds = watch.Elapsed.TotalSeconds;
            // Only this case's workspace is released; captured artifacts and traces
            // remain for independent validation. No broad orphan sweep is performed.
            workspaces.Release(session);
        }
        return result;
    }

    private static void ValidateRemoteIdentity(JsonElement identity, string expected)
    {
        if (!System.Text.RegularExpressions.Regex.IsMatch(expected, "^[0-9a-f]{64}$")) throw new ArgumentException("Expected lowercase native SHA256");
        if (identity.GetProperty("remote_pid").GetInt32() <= 0 || !identity.TryGetProperty("remote_start_ticks", out _))
            throw new InvalidDataException("Root-stage remote process identity is required");
        var mapped = identity.GetProperty("mapped_native_libraries").EnumerateObject().ToArray();
        if (mapped.Length == 0 || mapped.Any(item => item.Value.GetString() != expected))
            throw new InvalidDataException("Root-stage native mapping does not match the explicitly pinned candidate");
    }

    private static string Hash(string path)
    {
        using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(file));
    }

    private static Task Save(object report, string directory) => File.WriteAllTextAsync(Path.Combine(directory, "actual-agent-workflows.json"), JsonSerializer.Serialize(report, Json));
}

internal sealed record Fixture(string Variant, string Scenario, string Trial, string Wave, int Concurrency, JsonElement Spec);

internal sealed class CaseResult
{
    public string Variant { get; init; } = "";
    public string Scenario { get; init; } = "";
    public string Trial { get; init; } = "";
    public string Session { get; set; } = "";
    public JsonElement Spec { get; init; }
    public string Status { get; set; } = "fail";
    public string? Detail { get; set; }
    public string Answer { get; set; } = "";
    public string? Finish { get; set; }
    public int Rounds { get; set; }
    public double WallSeconds { get; set; }
    public object? Sandbox { get; set; }
    public object? Messages { get; set; }
    public object? ResultArtifactVersions { get; set; }
    public string[] SuccessfulTools { get; set; } = [];
    public List<object> Events { get; } = [];
    public List<object> Artifacts { get; } = [];
    public List<Dictionary<string, object?>> Requests { get; } = [];
    public List<CodeExecResult> CodeResults { get; } = [];
}

internal sealed record Options(string Endpoint, string Model, string SkillsRoot, string ArtifactRoot, string Output,
    string Fixtures, string Identity, string ExpectedNativeSha256, int ArtifactPort, bool Thinking, bool ScriptedFixture, int MaxRounds, int MaxTokens, int TimeoutSeconds)
{
    public static Options Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        bool thinking = false, scriptedFixture = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--thinking") { thinking = true; continue; }
            if (args[i] == "--scripted-fixture") { scriptedFixture = true; continue; }
            if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 == args.Length) throw new ArgumentException("Expected --option value");
            if (!values.TryAdd(args[i], args[++i])) throw new ArgumentException("Duplicate option");
        }
        string Required(string name) => values.Remove(name, out var value) ? value : throw new ArgumentException("Missing " + name);
        int Number(string name, int fallback) => values.Remove(name, out var value) ? int.Parse(value, System.Globalization.CultureInfo.InvariantCulture) : fallback;
        var result = new Options(Required("--endpoint"), Required("--model"), Path.GetFullPath(Required("--skills-root")),
            Path.GetFullPath(Required("--artifact-root")), Path.GetFullPath(Required("--output")), Path.GetFullPath(Required("--fixtures")),
            Path.GetFullPath(Required("--identity")), Required("--expected-native-sha256"), Number("--artifact-port", 18481),
            thinking, scriptedFixture, Number("--max-rounds", 24), Number("--max-tokens", 4096), Number("--timeout", 1200));
        if (values.Count != 0) throw new ArgumentException("Unknown options: " + string.Join(", ", values.Keys));
        var endpoint = new Uri(result.Endpoint);
        if (!endpoint.IsLoopback || endpoint.Scheme != "http" || !endpoint.AbsolutePath.EndsWith("/chat/completions", StringComparison.Ordinal))
            throw new ArgumentException("--endpoint must be an HTTP loopback tunnel URL ending /chat/completions");
        if (result.ArtifactPort is < 1024 or > 65535 || result.MaxRounds is < 1 or > 64 || result.MaxTokens is < 1 or > 16384 || result.TimeoutSeconds < 1)
            throw new ArgumentException("Invalid port or work bounds");
        return result;
    }
}
