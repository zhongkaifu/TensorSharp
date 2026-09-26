// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Python;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.Skills;

namespace TensorAgent.Tests;

/// <summary>
/// The one xUnit collection every live-model test class joins.
///
/// <para>
/// xUnit runs test classes in parallel by default, and two of these in parallel is
/// two multi-gigabyte models resident at once, two engines competing for the same
/// GPU, and — because CPython is one interpreter per process — two hosts sharing an
/// interpreter that only one of them configured. Naming the collection and
/// disabling parallelism inside it is what makes "the orchestrator runs them
/// serially" true of the test runner rather than only of the person invoking it.
/// </para>
/// </summary>

/// <summary>
/// A fact that needs real weights on this machine, and says how to supply them
/// rather than passing silently when they are absent.
/// </summary>
public sealed class LiveModelFactAttribute : FactAttribute
{
    public LiveModelFactAttribute() => Skip = LiveModelHarness.Unavailable(out _, out _);
}

/// <summary>
/// A fact whose model is asked to WRITE AND RUN a program, so it needs an
/// interpreter as well as weights.
///
/// <para>
/// Separate from <see cref="LiveModelFactAttribute"/> because the failure is
/// unrecognisable otherwise: without CPython the shell answers every
/// <c>python3</c> with "no embedded Python", the model apologises, and the test
/// fails on an assertion about arithmetic. Naming the missing interpreter up front
/// is the difference between a skip a person can act on and a red test they cannot.
/// </para>
/// </summary>
public sealed class LiveCodeFactAttribute : FactAttribute
{
    public LiveCodeFactAttribute() =>
        Skip = LiveModelHarness.Unavailable(out _, out _) ?? LiveModelHarness.NoInterpreterReason();
}

/// <summary>A live code scenario that also reaches current public internet data.</summary>
public sealed class LiveNetworkCodeFactAttribute : FactAttribute
{
    public LiveNetworkCodeFactAttribute()
    {
        Skip = LiveModelHarness.Unavailable(out _, out _)
            ?? LiveModelHarness.NoInterpreterReason()
            ?? (NetworkFactAttribute.Enabled
                ? null
                : $"reaches the real internet: set {NetworkFactAttribute.Variable}=1 and re-run");
    }
}

/// <summary>
/// A fact that needs the bundled <c>documents</c> skill and the two packages its
/// scripts import.
///
/// <para>
/// That skill is built alongside these tests rather than by them, so its absence is
/// a machine that is not ready, not a regression: the skip names the directory that
/// is missing. The same is true of reportlab and openpyxl — the skill's SKILL.md
/// contract is that PDFs come from one and spreadsheets from the other, and an
/// interpreter without them can only produce a traceback.
/// </para>
/// </summary>
public sealed class LiveDocumentsFactAttribute : FactAttribute
{
    public LiveDocumentsFactAttribute() =>
        Skip = LiveModelHarness.Unavailable(out _, out _)
            ?? LiveModelHarness.NoInterpreterReason()
            ?? LiveModelHarness.NoDocumentsSkillReason();
}

/// <summary>The numbers the final frame reports about one turn.</summary>
public readonly record struct TurnStats(
    int PromptTokens, int ReusedTokens, double ReusePercent, double Seconds, int Tokens)
{
    public override string ToString() =>
        $"prompt {PromptTokens}, reused {ReusedTokens} ({ReusePercent:0.0}%), {Tokens} tokens in {Seconds:0.0}s";
}

/// <summary>
/// A real model, a real turn, over the real API — the parts every live test needs.
///
/// <para>
/// Every hermetic test in this project checks a part in isolation, which is what
/// makes them fast — and also what makes them unable to answer the only question
/// that matters: does a person typing into this app get an answer back. That needs
/// weights, so it needs a machine that has some. Point
/// <c>TENSORAGENT_TEST_MODEL_DIR</c> at a directory holding the catalog's own file
/// names and the live tests run; otherwise they skip, saying what to set.
/// </para>
/// <para>
/// The harness lives here rather than in one of the two test classes because both
/// drive the same app the same way, and a second copy of "open a session, read the
/// SSE frames, pull the stats off the last one" is a second place for the frame
/// names to drift from what the page actually reads.
/// </para>
/// </summary>
public abstract class LiveModelHarness : IDisposable
{
    /// <summary>The xUnit collection that serialises everything holding a model.</summary>
    /// <summary>The one at-a-time queue; defined in <see cref="LiveModelCollection"/>.</summary>
    public const string Collection = LiveModelCollection.Name;

    /// <summary>Where the catalog's GGUF files can be found on this machine.</summary>
    public const string ModelDirVariable = "TENSORAGENT_TEST_MODEL_DIR";

    /// <summary>An alternative file name for the weights, when the local copy is named differently.</summary>
    public const string ModelFileVariable = "TENSORAGENT_TEST_MODEL_FILE";

    /// <summary>The skill whose scripts produce documents, per its SKILL.md contract.</summary>
    public const string DocumentsSkillId = "documents";

    /// <summary>
    /// Long enough for a whole agent turn, not just one generation.
    ///
    /// <para>
    /// A turn that writes a program, runs it, reads the traceback and runs it again
    /// is a dozen generations and a dozen executions inside a single HTTP request.
    /// A timeout sized for one generation cancels those halfway, and a cancelled
    /// stream looks exactly like a model that stopped talking.
    /// </para>
    /// </summary>
    protected static readonly TimeSpan RequestTimeout = TimeSpan.FromMinutes(30);

    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-live-" + Guid.NewGuid().ToString("N"));
    private AgentAppHost? _host;
    private HttpClient? _client;
    private string? _hostedModelFileName;

    /// <summary>The running app. Valid after <see cref="Start"/>.</summary>
    protected AgentAppHost Host =>
        _host ?? throw new InvalidOperationException("Start() has not run yet, so there is no host to talk to");

    /// <summary>An HTTP client holding the launch token, pointed at the loopback server.</summary>
    protected HttpClient Client =>
        _client ?? throw new InvalidOperationException("Start() has not run yet, so there is no server to talk to");

    /// <summary>Which backend actually answered the load, for the numbers a test prints.</summary>
    protected string LoadedBackend { get; private set; } = string.Empty;

    public virtual void Dispose()
    {
        _client?.Dispose();
        _host?.Dispose();
        try { Directory.Delete(_root, true); } catch { }
        GC.SuppressFinalize(this);
    }

    // =====================================================================================
    // what this machine can and cannot run
    // =====================================================================================

    /// <summary>Why the live tests cannot run here, or null when they can.</summary>
    internal static string? Unavailable(out CatalogModel model, out string weights)
    {
        model = null!;
        weights = string.Empty;

        string? directory = Environment.GetEnvironmentVariable(ModelDirVariable);
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return $"set {ModelDirVariable} to a directory holding one of the catalog's GGUF files";

        string? named = Environment.GetEnvironmentVariable(ModelFileVariable);
        foreach (CatalogModel candidate in ModelCatalog.BuiltIn)
        {
            if (candidate.Kind == CatalogArchitectureKind.Diffusion)
                continue;
            string path = Path.Combine(directory, named ?? candidate.Weights.FileName);
            if (!File.Exists(path))
                continue;
            // A truncated or unrelated file would fail deep inside the loader with a
            // message about tensors; check the size the catalog expects instead.
            if (named is null && new FileInfo(path).Length != candidate.Weights.Bytes)
                continue;
            model = candidate;
            weights = path;
            return null;
        }
        return $"no catalog GGUF found in {directory}; set {ModelFileVariable} to use a differently named file";
    }

    /// <summary>Why nothing here can run a program, or null when an interpreter was named.</summary>
    ///
    /// <para>
    /// Naming a root is not the same as having one that works, and the gap between the
    /// two costs three minutes of a live model and then an assertion about a missing
    /// PDF. The staged <c>python-runtime/simulator</c> is the trap: it holds a complete
    /// CPython, so every filesystem check passes, but its <c>Python.framework</c> is
    /// built for the iOS SIMULATOR (Mach-O platform 7) and dyld refuses to load it into
    /// a native macOS process. The shell then answers every <c>python3</c> with exit
    /// 127 and the model spends the turn working around it.
    /// </para>
    /// <para>
    /// So the check is the real one: resolve the layout the way the app does and ask
    /// the loader to open the library. That is dlopen, not <c>Py_Initialize</c> — this
    /// runs while xUnit is still enumerating tests, and initializing CPython here would
    /// burn the one interpreter this process gets before any test has chosen its root.
    /// </para>
    internal static string? NoInterpreterReason()
    {
        if (string.IsNullOrWhiteSpace(InterpreterRoot))
            return "this test asks the model to run a program, which needs an interpreter: set "
                + $"{LivePythonFactAttribute.RootVariable}=<a CPython 3.13 prefix with lib/python3.13 and "
                + "lib/libpython3.13.dylib> and re-run. On macOS, TensorAgent/python-runtime/simulator will "
                + "NOT do: it is built for the iOS simulator and cannot be loaded by a native test host.";

        if (!PythonRuntimeLayout.TryDiscover(InterpreterRoot!, out PythonRuntimeLayout? layout, out string? error)
            || layout is null)
        {
            return $"{LivePythonFactAttribute.RootVariable}={InterpreterRoot} is not a usable runtime: {error}";
        }

        string? library = layout.FindLibrary();
        if (library is null)
            return $"{LivePythonFactAttribute.RootVariable}={InterpreterRoot} has no loadable libpython; on a "
                + "test host the symbols cannot come from the app image the way they do on iOS";

        if (!NativeLibrary.TryLoad(library, out nint handle))
            return $"{library} cannot be loaded into this process -- on macOS this is what an iOS or "
                + "iOS-simulator build of CPython looks like. Point "
                + $"{LivePythonFactAttribute.RootVariable} at a CPython built for this machine.";
        NativeLibrary.Free(handle);
        return null;
    }

    /// <summary>Why the document scenarios cannot run here, or null when everything they need is present.</summary>
    internal static string? NoDocumentsSkillReason()
    {
        string directory = Path.Combine(RepoSkillsDirectory, DocumentsSkillId);
        string manifest = Path.Combine(directory, SkillManifestParser.SkillFileName);
        if (!File.Exists(manifest))
            return $"the '{DocumentsSkillId}' skill is not bundled: {manifest} does not exist";

        // Parse it the way the registry does, not by looking at the directory. A gate
        // that only checks for a file passes while the runtime finds no skill at all —
        // the frontmatter 'name' is what the model asks for, and the parser REJECTS a
        // manifest whose name disagrees with its directory rather than tolerating it.
        // Checked here, that is a skip naming the problem; unchecked, it is a live test
        // failing on an assertion about a PDF for reasons nothing mentions.
        if (!SkillManifestParser.TryParse(File.ReadAllText(manifest), DocumentsSkillId,
                out SkillManifest? parsed, out string? error))
            return $"the '{DocumentsSkillId}' skill does not load: {error}";
        if (!string.Equals(parsed!.Name, DocumentsSkillId, StringComparison.Ordinal))
            return $"the skill in {directory} calls itself '{parsed.Name}', so the model has no "
                + $"'{DocumentsSkillId}' skill to invoke";

        string[] missing = MissingPackages(InterpreterRoot!, "reportlab", "openpyxl");
        return missing.Length == 0
            ? null
            : $"the '{DocumentsSkillId}' skill writes PDFs with reportlab and spreadsheets with openpyxl, and "
                + $"{string.Join(" and ", missing)} could not be found under {InterpreterRoot}. Stage a runtime with "
                + "TensorAgent/scripts/prepare-python.sh, or install them into that prefix.";
    }

    /// <summary>The interpreter this machine offers, or null when it offers none.</summary>
    private static string? InterpreterRoot =>
        Environment.GetEnvironmentVariable(LivePythonFactAttribute.RootVariable) is { Length: > 0 } root
            ? root
            : null;

    /// <summary>
    /// Which of <paramref name="names"/> is not importable from <paramref name="root"/>.
    ///
    /// <para>
    /// A directory search rather than an <c>import</c>, because deciding whether to
    /// skip happens while xUnit is still enumerating tests and initializing CPython
    /// there would start the one interpreter this process gets before any test has
    /// chosen its runtime root. The depth bound keeps a mistyped root from turning
    /// into a walk of the disk; every layout the interpreter itself accepts —
    /// <c>lib/python3.x/site-packages</c>, <c>app_packages</c>, and the same pair
    /// under <c>python/</c> — is within it.
    /// </para>
    /// </summary>
    private static string[] MissingPackages(string root, params string[] names)
    {
        var options = new EnumerationOptions
        {
            RecurseSubdirectories = true,
            MaxRecursionDepth = 6,
            IgnoreInaccessible = true,
        };
        return names.Where(name =>
        {
            try { return !Directory.EnumerateDirectories(root, name, options).Any(); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return true; }
        }).ToArray();
    }

    /// <summary>The skills the app ships, read out of the repository the tests were built from.</summary>
    internal static string RepoSkillsDirectory => Path.Combine(RepoRoot, "TensorAgent", "skills");

    private static string RepoRoot
    {
        get
        {
            var directory = new DirectoryInfo(AppContext.BaseDirectory);
            while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
                directory = directory.Parent;
            return directory?.FullName
                ?? throw new InvalidOperationException(
                    $"no TensorSharp.slnx above {AppContext.BaseDirectory}, so the bundled skills cannot be located");
        }
    }

    // =====================================================================================
    // the app, started and driven
    // =====================================================================================

    /// <param name="model">The catalog entry whose weights were found.</param>
    /// <param name="weights">Where those weights actually are on this machine.</param>
    /// <param name="skills">
    /// Load the skills in TensorAgent/skills (the directory, which also holds the two the
    /// phone does not bundle). Off by default because every skill declares
    /// itself in the prompt, which costs thousands of tokens on every turn of every
    /// test that has nothing to do with skills.
    /// </param>
    /// <param name="interpreter">
    /// Stage the CPython named by <c>TENSORAGENT_PYTHON_ROOT</c>. Off by default so a
    /// machine without one still runs every test that never asks the model to compute
    /// anything.
    /// </param>
    /// <param name="maxTokens">The budget stored in settings; a request may still name its own.</param>
    protected AgentAppHost Start(
        CatalogModel model,
        string weights,
        bool skills = false,
        bool interpreter = false,
        int maxTokens = 64)
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"))
        {
            DeviceMemoryGB = 16,
            ExecutionMode = AgentExecutionMode.InProcess,
            BundledSkillsDirectory = skills ? RepoSkillsDirectory : string.Empty,
            PythonRuntimeDirectory = interpreter ? InterpreterRoot ?? string.Empty : string.Empty,
        };
        paths.EnsureCreated();

        // Link, do not copy: these files are gigabytes. Keep the SOURCE basename,
        // though. ModelFileVariable deliberately permits a local quantization whose
        // name differs from the catalog's preferred file; hiding Q8_0 behind the
        // catalog's IQ4_XS name made the live output claim that the wrong checkpoint
        // had loaded and left the acceptance test unable to prove which one it ran.
        string target = Path.Combine(paths.ModelsDirectory, model.Id);
        Directory.CreateDirectory(target);
        _hostedModelFileName = Path.GetFileName(weights);
        string hostedModelPath = Path.Combine(target, _hostedModelFileName);
        File.CreateSymbolicLink(hostedModelPath, Path.GetFullPath(weights));

        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = model.Id;
        chosen.MaxTokens = maxTokens;
        settings.Save(chosen);

        // UseModel normally applies these immediately before loading. This harness
        // loads through the public HTTP API instead, so apply the same catalog budget
        // here before the model is constructed.
        EngineMemoryPolicy.Apply(model, chosen);

        _host = new AgentAppHost(paths);
        _host.Options.RepointHostedModel(hostedModelPath, _host.Options.StartupMmProjPath);
        // The test below deliberately owns the one load through /api/models/load.
        // AgentAppHost.Start would also schedule its remembered-model load, which
        // both restores the catalog alias and races this explicit request.
        _host.Server.Start();
        _client = Connect(_host);
        return _host;
    }

    /// <summary>
    /// Close this host and open another over the same directories, which is what a
    /// relaunch is: the engine session and its KV cache are gone, and only what was
    /// written to disk survives.
    /// </summary>
    protected AgentAppHost Reopen(AgentPaths paths)
    {
        _client?.Dispose();
        _host?.Dispose();
        _host = new AgentAppHost(paths);
        if (_hostedModelFileName is { Length: > 0 })
        {
            string hostedModelPath = Path.Combine(
                paths.ModelsDirectory,
                _host.Settings.Load().SelectedModelId!,
                _hostedModelFileName);
            _host.Options.RepointHostedModel(hostedModelPath, _host.Options.StartupMmProjPath);
        }
        _host.Server.Start();
        _client = Connect(_host);
        return _host;
    }

    private static HttpClient Connect(AgentAppHost host)
    {
        var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl), Timeout = RequestTimeout };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={host.Server.Token}");
        return client;
    }

    /// <summary>
    /// Load the selected model on the best backend this build actually has.
    ///
    /// <para>
    /// Metal first, because it is what the phone uses and what makes a scenario that
    /// writes and runs a program finish in minutes rather than in an hour. A build
    /// whose Metal slice is missing must still be able to run these, so a refused
    /// load falls back to the CPU — and SAYS so, naming the backend and the reason it
    /// was refused. A silent fallback turns "Metal is broken" into "the tests are
    /// slow today", which is the one thing nobody investigates.
    /// </para>
    /// </summary>
    /// <returns>The backend that answered, for the numbers a test prints.</returns>
    protected async Task<string> LoadAsync(CatalogModel model)
    {
        string modelFileName = _hostedModelFileName
            ?? throw new InvalidOperationException("Start() did not publish model weights");
        var refusals = new List<string>();
        foreach (string backend in new[] { "ggml_metal", "ggml_cpu" })
        {
            HttpResponseMessage response = await Client.PostAsJsonAsync("/api/models/load", new
            {
                model = modelFileName,
                backend,
            });
            string payload = await response.Content.ReadAsStringAsync();
            if (response.IsSuccessStatusCode)
            {
                JsonElement loaded = JsonSerializer.Deserialize<JsonElement>(payload);
                Assert.Equal(modelFileName, loaded.GetProperty("model").GetString());
                LoadedBackend = backend;
                Console.WriteLine($"live model: {modelFileName} ({model.DisplayName}) loaded on {backend}");
                return backend;
            }

            refusals.Add($"{backend}: {(int)response.StatusCode} {payload}");
            Console.WriteLine($"live model: {backend} refused the load ({(int)response.StatusCode} {payload})");
        }

        throw new InvalidOperationException(
            "the model could not be loaded on any backend this build offers:"
            + Environment.NewLine + "  " + string.Join(Environment.NewLine + "  ", refusals));
    }

    /// <summary>Open a session bound to a conversation, exactly as the page does before its first message.</summary>
    protected async Task<JsonElement> OpenSessionAsync(string conversation = "new")
    {
        using HttpResponseMessage response = await Client.PostAsync($"/api/sessions?conversation={conversation}", null);
        string payload = await response.Content.ReadAsStringAsync();
        Assert.True(response.IsSuccessStatusCode, $"opening a session failed: {(int)response.StatusCode} {payload}");
        return JsonSerializer.Deserialize<JsonElement>(payload);
    }

    /// <summary>The directory the shell tool and a skill's scripts share for one session.</summary>
    protected string WorkspaceOf(string sessionId) => Host.Workspaces.GetOrCreate(sessionId).WorkDirectory;

    /// <summary>Read a server-sent-event stream into its frames, as the page does.</summary>
    protected async Task<List<JsonElement>> StreamAsync(object body, CancellationToken ct = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json"),
        };
        using HttpResponseMessage response = await Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        Assert.True(response.IsSuccessStatusCode, $"chat failed: {(int)response.StatusCode} {await response.Content.ReadAsStringAsync(ct)}");
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);

        var frames = new List<JsonElement>();
        await using Stream stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream);
        while (await reader.ReadLineAsync(ct) is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal))
                continue;
            frames.Add(JsonSerializer.Deserialize<JsonElement>(line[6..]));
        }
        return frames;
    }

    /// <summary>The assistant message the page would have rendered from these frames.</summary>
    protected static string TextOf(IEnumerable<JsonElement> frames)
    {
        var text = new StringBuilder();
        foreach (JsonElement frame in frames)
        {
            if (frame.TryGetProperty("token", out JsonElement token) && token.GetString() is { } piece)
                text.Append(piece);
            else if (frame.TryGetProperty("replace", out JsonElement replace) && replace.GetString() is { } whole)
                text.Clear().Append(whole);
        }
        return text.ToString();
    }

    /// <summary>The stats line the page reads off the terminal frame.</summary>
    protected static TurnStats StatsOf(IEnumerable<JsonElement> frames)
    {
        JsonElement done = frames.Last(f => f.TryGetProperty("done", out _));
        return new TurnStats(
            done.GetProperty("promptTokens").GetInt32(),
            done.GetProperty("kvReusedTokens").GetInt32(),
            done.GetProperty("kvReusePercent").GetDouble(),
            done.GetProperty("elapsed").GetDouble(),
            done.GetProperty("tokenCount").GetInt32());
    }
}
