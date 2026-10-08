using System.Net;
using System.Net.Sockets;
using System.Net.Http.Json;
using System.Text.Json;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.JavaScript;
using TensorAgent.Core.Python;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Sandbox;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Skills;

namespace TensorAgent.Tests;

/// <summary>
/// The whole app, assembled and driven over HTTP.
///
/// <para>
/// Every other test in this project checks one part. This one checks that the parts
/// were connected: that the settings the user sees actually reach the sandbox, that
/// a session created by the page is filed under a conversation that survives a
/// restart, and that the code runner underneath the chat tool is the in-process one
/// rather than a process launcher that would not exist on a phone.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class AgentAppHostTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-host-" + Guid.NewGuid().ToString("N"));
    private AgentAppHost? _host;
    private HttpClient? _client;

    private AgentPaths Paths => new(Path.Combine(_root, "data"), Path.Combine(_root, "cache"))
    {
        ExecutionMode = AgentExecutionMode.InProcess,
    };

    private AgentAppHost Start(Action<SettingsStore>? configure = null)
    {
        if (configure is not null)
        {
            AgentPaths paths = Paths;
            paths.EnsureCreated();
            configure(new SettingsStore(paths.SettingsFile));
        }

        _host = new AgentAppHost(Paths);
        _host.Start();
        _client = new HttpClient { BaseAddress = new Uri(_host.Server.BaseUrl) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{Core.Hosting.LoopbackServer.TokenCookie}={_host.Server.Token}");
        return _host;
    }

    public void Dispose()
    {
        _client?.Dispose();
        _host?.Dispose();
        CodeEnvironment.Reset();
        try { Directory.Delete(_root, true); } catch { }
    }

    private async Task<JsonElement> Get(string path)
        => JsonSerializer.Deserialize<JsonElement>(await _client!.GetStringAsync(path));

    /// <summary>
    /// A remembered model the picker would not offer this device is not loaded either.
    ///
    /// <para>
    /// The Models list is built from <c>ForDevice</c>, while startup can find a saved id
    /// in the whole catalog. Without a second device-tier check, a choice restored from
    /// a larger device could auto-load with no row in the list to explain or undo it.
    /// The retained catalog starts at 12 GB, so an 8 GB device exercises that boundary
    /// with a real catalog entry.
    /// </para>
    /// </summary>
    [Fact]
    public void AModelGatedAboveThisDeviceIsNotLoadedAtStartupAndTheChoiceIsCleared()
    {
        CatalogModel tooBig = ModelCatalog.BuiltIn.First(m => m.MinDeviceMemoryGB > 8);
        AgentPaths paths = Paths with { DeviceMemoryGB = 8 };
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = tooBig.Id;
        settings.Save(chosen);

        _host = new AgentAppHost(paths);
        _host.Start();

        Assert.Equal(AgentAppHost.ModelLoadState.None, _host.ModelLoad);
        Assert.Null(new SettingsStore(paths.SettingsFile).Load().SelectedModelId);
    }

    [Theory]
    [InlineData("qwen3.8-flash-next-iq1m", true)]
    [InlineData("qwen3.8-flash-next-q2kxl", false)]
    public void A32GbStartupKeepsTheIq1MChoiceButStillClearsThe48GbQ2Choice(string id, bool keepChoice)
    {
        AgentPaths paths = Paths with { DeviceMemoryGB = 32, DeviceClass = DeviceClass.Desktop };
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = id;
        settings.Save(chosen);

        // Only the small metadata shard is present. Startup must restore the IQ1_M
        // choice and skip an incomplete download before any native model load occurs.
        CatalogModel model = ModelCatalog.Find(id)!;
        var store = new ModelStore(paths.ModelsDirectory);
        Directory.CreateDirectory(store.DirectoryFor(model));
        using (FileStream stream = File.Create(store.PathFor(model, model.Weights)))
            stream.SetLength(model.Weights.Bytes);

        _host = new AgentAppHost(paths);
        _host.Start();

        Assert.Contains(_host.Catalog, m => m.Id == "qwen3.8-flash-next-iq1m");
        Assert.DoesNotContain(_host.Catalog, m => m.Id == "qwen3.8-flash-next-q2kxl");
        Assert.Equal(AgentAppHost.ModelLoadState.None, _host.ModelLoad);
        Assert.Equal(keepChoice ? id : null, new SettingsStore(paths.SettingsFile).Load().SelectedModelId);
        Assert.Equal(InstallState.Partial, _host.Models.StateOf(model));
        if (keepChoice)
            Assert.Equal(store.PathFor(model, model.Weights), _host.Options.StartupModelPath);
    }

    /// <summary>
    /// A remembered model the catalog no longer has is cleared, not merely skipped.
    ///
    /// <para>
    /// The id encodes the quantization, so re-pointing an entry at a better file
    /// renames it — <c>gemma-4-12b-iq3xxs</c> becomes <c>gemma-4-12b-iq2m</c> — and
    /// every install that had the old one selected wakes up holding a name that
    /// resolves to nothing. Leaving it in settings.json means the app carries a dead
    /// choice for the rest of its life, presenting it to the picker as the model in
    /// use. The gated-above-this-device case beside it has always cleared; this is the
    /// same argument for the case that vanished instead of growing too big.
    /// </para>
    /// </summary>
    [Fact]
    public void AModelTheCatalogNoLongerHasIsClearedFromTheSavedChoice()
    {
        AgentPaths paths = Paths with { DeviceMemoryGB = 12 };
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = "gemma-4-12b-iq3xxs";
        settings.Save(chosen);

        _host = new AgentAppHost(paths);
        _host.Start();

        Assert.Equal(AgentAppHost.ModelLoadState.None, _host.ModelLoad);
        Assert.Null(new SettingsStore(paths.SettingsFile).Load().SelectedModelId);
    }

    /// <summary>
    /// A remembered model this build does not know and never retired is kept, unloaded,
    /// rather than cleared: it is most likely a newer build's entry, chosen in the Debug or
    /// Release build that shares these settings, and clearing it would leave that build
    /// starting with no model the next time it opens.
    /// </summary>
    [Fact]
    public void AModelANewerBuildChoseIsKeptForIt()
    {
        AgentPaths paths = Paths with { DeviceMemoryGB = 12 };
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = "a-newer-builds-entry-q4";
        settings.Save(chosen);

        _host = new AgentAppHost(paths);
        _host.Start();

        Assert.Equal(AgentAppHost.ModelLoadState.None, _host.ModelLoad);
        Assert.Equal("a-newer-builds-entry-q4", new SettingsStore(paths.SettingsFile).Load().SelectedModelId);
    }

    /// <summary>
    /// A launch reclaims the weights and prefix checkpoints of a RETIRED entry and keeps
    /// those of an id it does not know, which a newer build sharing the directories may
    /// have installed (see CatalogTests.ASweepByAnOlderBuildKeepsTheModelsANewerBuildInstalled).
    /// </summary>
    [Fact]
    public void StartupReclaimsRetiredModelsAndKeepsAnUnknownIdsFiles()
    {
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        string prefixCache = Path.Combine(paths.CacheRoot, "prefix-cache");
        const string retired = "gemma-4-12b-iq3xxs";
        const string newer = "a-newer-builds-entry-q4";
        Assert.True(ModelCatalog.IsRetired(retired));
        foreach (string id in new[] { retired, newer })
        {
            Directory.CreateDirectory(Path.Combine(paths.ModelsDirectory, id));
            File.WriteAllBytes(Path.Combine(paths.ModelsDirectory, id, "weights.gguf"), new byte[64]);
            Directory.CreateDirectory(Path.Combine(prefixCache, id));
            File.WriteAllBytes(Path.Combine(prefixCache, id, "checkpoint.bin"), new byte[64]);
        }

        _host = new AgentAppHost(paths);

        Assert.False(Directory.Exists(Path.Combine(paths.ModelsDirectory, retired)));
        Assert.False(Directory.Exists(Path.Combine(prefixCache, retired)));
        Assert.True(File.Exists(Path.Combine(paths.ModelsDirectory, newer, "weights.gguf")));
        Assert.True(File.Exists(Path.Combine(prefixCache, newer, "checkpoint.bin")));
    }

    /// <summary>
    /// The app chooses LoRA plug-ins per picture. A TS_LORAS exported in the shell that
    /// launched it (run-mac.sh passes the environment through) would apply plug-ins nobody
    /// chose to every image load, and make the engine warn on every other one.
    /// </summary>
    [Fact]
    public void ALoraListInheritedFromTheEnvironmentIsNotApplied()
    {
        string? before = Environment.GetEnvironmentVariable("TS_LORAS");
        try
        {
            Environment.SetEnvironmentVariable("TS_LORAS", "[{\"path\":\"/somewhere/else.safetensors\"}]");
            _host = new AgentAppHost(Paths);
            Assert.Null(Environment.GetEnvironmentVariable("TS_LORAS"));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_LORAS", before);
        }
    }

    /// <summary>Puts correctly sized (sparse) files in place, so the store reads the plug-in as installed.</summary>
    private static void FakeInstall(LoraStore store, string id)
    {
        CatalogLora lora = LoraCatalog.Find(id)!;
        Directory.CreateDirectory(store.DirectoryFor(lora));
        foreach (LoraFile file in lora.Files)
        {
            using FileStream stream = File.Create(store.PathFor(lora, file));
            stream.SetLength(file.Bytes);
        }
    }

    [Fact]
    public async Task TheLoraSheetListsThePlugInsAndSavesOnlyAChoiceThatCanBeHonoured()
    {
        AgentPaths paths = Paths with { DeviceMemoryGB = 48 };
        _host = new AgentAppHost(paths);
        _host.Start();
        _client = new HttpClient { BaseAddress = new Uri(_host.Server.BaseUrl) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{Core.Hosting.LoopbackServer.TokenCookie}={_host.Server.Token}");

        JsonElement listed = await Get("/api/agent/loras");
        Assert.Equal(LoraCatalog.BuiltIn.Count, listed.GetProperty("loras").GetArrayLength());
        Assert.All(listed.GetProperty("loras").EnumerateArray(), l => Assert.Equal("NotInstalled", l.GetProperty("state").GetString()));
        Assert.Equal(0, listed.GetProperty("chosen").GetArrayLength());

        const string viggle = "qwen-image-2.1-viggle-turbo", pruna = "qwen-image-2.1-pruna-8step", film = "qwen-image-2.1-film-stills";
        async Task<HttpResponseMessage> Choose(params (string Id, float Strength)[] loras) =>
            await _client!.PostAsJsonAsync("/api/agent/loras/choice",
                new { loras = loras.Select(l => new { id = l.Id, strength = l.Strength }).ToArray() });

        HttpResponseMessage refused = await Choose((viggle, 1f));
        Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
        Assert.Contains("not downloaded", (await refused.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("error").GetString());

        FakeInstall(_host.Loras, viggle);
        FakeInstall(_host.Loras, pruna);
        FakeInstall(_host.Loras, film);
        refused = await Choose((viggle, 1f), (pruna, 1f));
        Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
        Assert.Contains("both set the number of steps", (await refused.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("error").GetString());

        HttpResponseMessage saved = await Choose((film, 0.55f), (viggle, 1f));
        Assert.Equal(HttpStatusCode.OK, saved.StatusCode);
        Assert.Equal(new[] { new ImageLoraChoice(film, 0.55f), new ImageLoraChoice(viggle, 1f) },
            new SettingsStore(paths.SettingsFile).Load().ImageLoras);
        JsonElement state = await Get("/api/agent/loras");
        JsonElement filmRow = state.GetProperty("loras").EnumerateArray().Single(l => l.GetProperty("id").GetString() == film);
        Assert.True(filmRow.GetProperty("chosen").GetBoolean());
        Assert.Equal(0.55f, filmRow.GetProperty("strength").GetSingle());
        Assert.Equal("Installed", filmRow.GetProperty("state").GetString());

        // Removing a plug-in turns it off as well.
        HttpResponseMessage removed = await _client!.DeleteAsync($"/api/agent/loras/{film}");
        Assert.Equal(HttpStatusCode.OK, removed.StatusCode);
        Assert.Equal(new[] { new ImageLoraChoice(viggle, 1f) }, new SettingsStore(paths.SettingsFile).Load().ImageLoras);
        Assert.False(Directory.Exists(_host.Loras.DirectoryFor(LoraCatalog.Find(film)!)));

        Assert.Equal(HttpStatusCode.NotFound, (await _client!.DeleteAsync("/api/agent/loras/no-such-plug-in")).StatusCode);
    }

    private void StartServing(AgentPaths paths)
    {
        _host = new AgentAppHost(paths);
        _host.Start();
        _client = new HttpClient { BaseAddress = new Uri(_host.Server.BaseUrl) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{Core.Hosting.LoopbackServer.TokenCookie}={_host.Server.Token}");
    }

    /// <summary>
    /// The page saves its other settings by posting back the whole copy it read when it
    /// loaded. That copy's plug-in choice is older than anything changed in the LoRA sheet
    /// since, so the save keeps the stored choice: a plug-in turned on stays on, and one
    /// removed with its files does not come back to refuse every later picture.
    /// </summary>
    [Fact]
    public async Task ASettingsSaveFromThePageKeepsThePlugInsChosenSince()
    {
        AgentPaths paths = Paths with { DeviceMemoryGB = 48 };
        StartServing(paths);
        const string viggle = "qwen-image-2.1-viggle-turbo", film = "qwen-image-2.1-film-stills";
        FakeInstall(_host!.Loras, viggle);
        FakeInstall(_host.Loras, film);
        JsonElement loaded = await Get("/api/agent/settings");
        Assert.Equal(0, loaded.GetProperty("imageLoras").GetArrayLength());

        Assert.Equal(HttpStatusCode.OK, (await _client!.PostAsJsonAsync("/api/agent/loras/choice",
            new { loras = new[] { new { id = viggle, strength = 1f }, new { id = film, strength = 0.6f } } })).StatusCode);

        // The page's copy, from before the choice, with one switch flipped.
        var stale = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(loaded.GetRawText())!;
        stale["skillsEnabled"] = JsonSerializer.SerializeToElement(false);
        HttpClient client = _client!;
        HttpResponseMessage response = await client.PostAsJsonAsync("/api/agent/settings", stale);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        AppSettings stored = new SettingsStore(paths.SettingsFile).Load();
        Assert.False(stored.SkillsEnabled);
        var both = new[] { new ImageLoraChoice(viggle, 1f), new ImageLoraChoice(film, 0.6f) };
        Assert.Equal(both, stored.ImageLoras);
        Assert.Equal(2, (await response.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("imageLoras").GetArrayLength());

        // A copy that leaves the choice out, or says null, changes nothing either.
        stale.Remove("imageLoras");
        Assert.Equal(HttpStatusCode.OK, (await client.PostAsJsonAsync("/api/agent/settings", stale)).StatusCode);
        stale["imageLoras"] = JsonSerializer.SerializeToElement<object?>(null);
        Assert.Equal(HttpStatusCode.OK, (await client.PostAsJsonAsync("/api/agent/settings", stale)).StatusCode);
        Assert.Equal(both, new SettingsStore(paths.SettingsFile).Load().ImageLoras);

        // Removed in the sheet, then a copy taken while it was on is saved: it stays removed.
        JsonElement withBoth = await Get("/api/agent/settings");
        Assert.Equal(HttpStatusCode.OK, (await client.DeleteAsync($"/api/agent/loras/{viggle}")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await client.PostAsJsonAsync("/api/agent/settings", withBoth)).StatusCode);
        Assert.Equal(new[] { new ImageLoraChoice(film, 0.6f) }, new SettingsStore(paths.SettingsFile).Load().ImageLoras);
    }

    /// <summary>
    /// Every change in the sheet sends the whole choice back, including what this build cannot
    /// check: a plug-in a newer build sharing the settings chose, and one that is on but whose
    /// files have gone. Neither blocks the change, and the one whose files are gone can be
    /// turned off; only what the change adds has to be downloaded.
    /// </summary>
    [Fact]
    public async Task TheSheetCanChangeAChoiceHoldingPlugInsItCannotCheck()
    {
        AgentPaths paths = Paths with { DeviceMemoryGB = 48 };
        const string viggle = "qwen-image-2.1-viggle-turbo", film = "qwen-image-2.1-film-stills", grain = "qwen-image-2.1-grainscape";
        const string newer = "a-plug-in-from-a-newer-build";
        var store = new SettingsStore(paths.SettingsFile);
        AppSettings seeded = store.Load();
        seeded.ImageLoras = new() { new ImageLoraChoice(newer, 0.8f), new ImageLoraChoice(viggle, 1f) };
        store.Save(seeded);
        StartServing(paths);
        FakeInstall(_host!.Loras, film);

        JsonElement sheet = await Get("/api/agent/loras");
        JsonElement viggleRow = sheet.GetProperty("loras").EnumerateArray().Single(l => l.GetProperty("id").GetString() == viggle);
        Assert.True(viggleRow.GetProperty("chosen").GetBoolean());
        Assert.Equal("NotInstalled", viggleRow.GetProperty("state").GetString());

        // What the page sends to turn Film Stills on: the saved choice, then the change.
        async Task<HttpResponseMessage> Choose(params object[] loras) =>
            await _client!.PostAsJsonAsync("/api/agent/loras/choice", new { loras });
        HttpResponseMessage on = await Choose(new { id = newer, strength = 0.8f }, new { id = viggle, strength = 1f }, new { id = film, strength = 0.7f });
        Assert.Equal(HttpStatusCode.OK, on.StatusCode);
        Assert.Equal(new[] { new ImageLoraChoice(viggle, 1f), new ImageLoraChoice(film, 0.7f), new ImageLoraChoice(newer, 0.8f) },
            store.Load().ImageLoras);

        // Turning off the one whose files are gone.
        Assert.Equal(HttpStatusCode.OK, (await Choose(new { id = newer, strength = 0.8f }, new { id = film, strength = 0.7f })).StatusCode);
        Assert.Equal(new[] { new ImageLoraChoice(film, 0.7f), new ImageLoraChoice(newer, 0.8f) }, store.Load().ImageLoras);

        // Something newly turned on still has to be there.
        HttpResponseMessage refused = await Choose(new { id = film, strength = 0.7f }, new { id = grain, strength = 0.7f });
        Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
        Assert.Contains("not downloaded", (await refused.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("error").GetString());

        // A strength that is not a number is the client's mistake, said as one; none at all is the plug-in's own.
        refused = await Choose(new { id = film, strength = "0.8" });
        Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
        Assert.Contains("must be a number", (await refused.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("error").GetString());
        Assert.Equal(HttpStatusCode.OK, (await Choose(new { id = film, strength = (float?)null })).StatusCode);
        Assert.Equal(new[] { new ImageLoraChoice(film, 0.7f), new ImageLoraChoice(newer, 0.8f) }, store.Load().ImageLoras);
    }

    [Fact]
    public async Task ADeviceWithoutTheImageModelIsOfferedNoPlugIns()
    {
        _host = new AgentAppHost(Paths with { DeviceMemoryGB = 12 });
        _host.Start();
        _client = new HttpClient { BaseAddress = new Uri(_host.Server.BaseUrl) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{Core.Hosting.LoopbackServer.TokenCookie}={_host.Server.Token}");

        Assert.Equal(0, (await Get("/api/agent/loras")).GetProperty("loras").GetArrayLength());
    }

    /// <summary>
    /// A picture whose plug-ins cannot be honoured ends with the reason, not with a picture
    /// made without them; and the preparation is told whether the picture is an edit.
    /// </summary>
    [Fact]
    public async Task APictureItsPlugInsCannotBeHonouredForIsRefusedWithTheReason()
    {
        _host = new AgentAppHost(Paths);
        var asked = new List<bool>();
        JsonElement body = JsonDocument.Parse(
            "{\"sessionId\":\"s-1\",\"messages\":[{\"role\":\"user\",\"content\":\"a red apple\",\"stillImagePaths\":[\"/tmp/photo.png\"]}]}")
            .RootElement.Clone();

        var frames = new List<JsonElement>();
        await foreach (object frame in ImageTurns.StreamAsync(_host.Chat, body, CancellationToken.None, editing =>
        {
            asked.Add(editing);
            return ImageTurns.Preparation.Refused("Viggle Turbo is turned on but its files are missing.");
        }))
        {
            frames.Add(JsonSerializer.SerializeToElement(frame));
        }

        Assert.Equal(new[] { true }, asked);
        JsonElement only = Assert.Single(frames);
        Assert.True(only.GetProperty("done").GetBoolean());
        Assert.Equal("Viggle Turbo is turned on but its files are missing.", only.GetProperty("error").GetString());
        Assert.Equal("s-1", only.GetProperty("sessionId").GetString());
    }

    [Fact]
    public void ItCreatesEverythingItNeedsAndSeparatesBackedUpDataFromRefetchableFiles()
    {
        AgentAppHost host = Start();
        Assert.True(Directory.Exists(host.Paths.ConversationsDirectory));
        Assert.True(Directory.Exists(host.Paths.ModelsDirectory));
        Assert.True(Directory.Exists(host.Paths.ScratchDirectory));

        // Weights can always be downloaded again; a transcript cannot. They must not
        // share a root, or the backup either carries gigabytes or loses the chats.
        Assert.StartsWith(host.Paths.CacheRoot, host.Paths.ModelsDirectory, StringComparison.Ordinal);
        Assert.StartsWith(host.Paths.DataRoot, host.Paths.ConversationsDirectory, StringComparison.Ordinal);
        Assert.DoesNotContain(host.Paths.DataRoot, host.Paths.ModelsDirectory, StringComparison.Ordinal);
    }

    /// <summary>
    /// A skill's own script runs on the same thing the model's own programs run on.
    ///
    /// <para>
    /// It did not. The request planner built its script runner with no backend, which
    /// falls back to launching a child process — so on a desktop <c>skills_run</c>
    /// quietly used the SYSTEM python3, without the packages this app staged, and
    /// answered "No module named 'reportlab'" for a script the shell tool could run
    /// perfectly; and on iOS, where no process can be started at all, no skill script
    /// could ever have run. The symptom was "the documents skill does not work",
    /// which points at the skill and not at the wiring.
    /// </para>
    /// </summary>
    [Fact]
    public void ASkillsScriptRunsOnTheSameInterpreterTheModelsOwnProgramsDo()
    {
        AgentAppHost host = Start();
        Assert.NotNull(host.CodeRunner);
        Assert.Same(host.Backend, host.CodeRunner!.Backend);
        Assert.Equal("in-process", host.CodeRunner.Backend!.Name);
    }

    [Fact]
    public void TheCodeRunnerRunsInProcessBecauseNothingCanBeSpawned()
    {
        AgentAppHost host = Start();
        Assert.NotNull(host.CodeRunner);
        Assert.True(host.CodeRunner!.CanRun);
        Assert.Equal("in-process", host.Backend.Name);
        Assert.Equal("sh", host.Backend.Shell!.Name);
    }

    [Fact]
    public void TheUnsetSkillsRoundCapLetsCodePlanningUseTwentyFourRounds()
    {
        AgentAppHost host = Start();

        // ServerHostingOptions stores eight as the ordinary skills fallback, but the
        // separate "specified" bit is what lets SkillRequestPlan.RoundsFor promote an
        // agent that can also write, run and repair code to the larger default.
        Assert.False(host.Options.SkillsMaxRoundsSpecified);
        Assert.Equal(8, host.Options.SkillsMaxRounds);

        SkillRequestPlan plan = SkillRequestPlan.Create(
            host.Skills,
            Array.Empty<string>(),
            discovery: false,
            clientTools: new List<ToolFunction>(),
            architecture: "qwen35",
            contextTokens: 32768,
            options: host.Options,
            out IReadOnlyList<string> unknown,
            codeRunner: host.CodeRunner)!;

        Assert.Empty(unknown);
        Assert.NotNull(plan);
        Assert.Equal(SkillHostOptions.CodeExecutionRounds, plan.LoopOptions.MaxRounds);
        Assert.Equal(24, plan.LoopOptions.MaxRounds);
    }

    [Fact]
    public void WithNetworkOn_TheModelIsToldTheInProcessInstallerActualLimits()
    {
        AgentAppHost host = Start(settings =>
        {
            AppSettings enabled = settings.Load();
            enabled.AllowNetwork = true;
            settings.Save(enabled);
        });

        ToolFunction persistent = host.CodeRunner!.DeclareTools(persists: true)
            .Single(tool => tool.Name == ShellTools.ShellToolName);
        ToolFunction stateless = Assert.Single(host.CodeRunner.DeclareTools(persists: false));

        Assert.True(host.CodeRunner.CanInstallPackages);
        Assert.True(host.CodeRunner.CanInstallPackagesFor("python"));
        Assert.False(host.CodeRunner.CanInstallPackagesFor("javascript"));

        foreach (ToolFunction shell in new[] { persistent, stateless })
        {
            // The network guidance is present, and every sentence of it is true of any
            // request. It used to carry a whole Yahoo Finance screener program, which is
            // what made the model reach for finance APIs on unrelated turns.
            Assert.Contains("quoted heredoc", shell.Description, StringComparison.Ordinal);
            Assert.Contains("python3 - <<'PY'", shell.Description, StringComparison.Ordinal);
            Assert.Contains("standard library first", shell.Description, StringComparison.Ordinal);
            Assert.Contains("instead of inventing it", shell.Description, StringComparison.Ordinal);
            // What may not be INVENTED, never a ban on explaining or deriving: the turns
            // where the answer IS an interpretation are turns too.
            Assert.DoesNotContain("no figure you did not fetch", shell.Description, StringComparison.Ordinal);
            Assert.Contains("pure-Python wheels tagged `none-any`", shell.Description, StringComparison.Ordinal);
            Assert.Contains("does not resolve dependencies", shell.Description, StringComparison.Ordinal);
            Assert.Contains("compiled/native extensions", shell.Description, StringComparison.Ordinal);
            Assert.DoesNotContain("npm install", shell.Description, StringComparison.Ordinal);
            // And what it need not install at all: the model that produced an HTML file
            // instead of a deck did so because nothing told it lxml was already here.
            Assert.Contains("already built into this app", shell.Description, StringComparison.Ordinal);
            Assert.Contains("lxml", shell.Description, StringComparison.Ordinal);
            Assert.Contains("python-pptx (import pptx)", shell.Description, StringComparison.Ordinal);
            Assert.Contains("cannot be replaced by another version", shell.Description, StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// The network switch defaults to off, and off is where most users live. The
    /// install guidance is rightly hidden then -- but the bundle is not something the
    /// switch changes, and a model that is not told lxml and python-pptx are here
    /// reimplements OOXML by hand, badly, for the rest of the turn.
    /// </summary>
    [Fact]
    public void WithNetworkOff_TheModelIsStillToldWhatIsBuiltIn()
    {
        AgentAppHost host = Start(settings =>
        {
            AppSettings offline = settings.Load();
            offline.AllowNetwork = false;
            settings.Save(offline);
        });

        ToolFunction shell = host.CodeRunner!.DeclareTools(persists: true)
            .Single(tool => tool.Name == ShellTools.ShellToolName);

        Assert.False(host.CodeRunner.CanInstallPackages);
        // The quoting advice is about writing Python, not about the network, so it is
        // here with the switch off too. It used to be inside the network paragraph.
        Assert.Contains("quoted heredoc", shell.Description, StringComparison.Ordinal);
        Assert.Contains("Already available:", shell.Description, StringComparison.Ordinal);
        Assert.Contains("already built into this app", shell.Description, StringComparison.Ordinal);
        Assert.Contains("lxml", shell.Description, StringComparison.Ordinal);
        Assert.Contains("python-pptx (import pptx)", shell.Description, StringComparison.Ordinal);
        Assert.DoesNotContain("pure-Python wheels tagged `none-any`", shell.Description, StringComparison.Ordinal);
    }

    // =====================================================================================
    // the bundle answers before the index
    // =====================================================================================

    private static readonly BundledDistribution[] PhoneBundle =
    {
        new("lxml", "lxml", "6.1.3", Compiled: true),
        new("python-pptx", "python-pptx", "1.0.2", Compiled: false),
    };

    private AgentAppHost StartWithBundle(bool network, RecordingInstallHook installer, out PackageProbePython python)
    {
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings enabled = settings.Load();
        enabled.AllowCodeExecution = true;
        enabled.AllowNetwork = network;
        settings.Save(enabled);

        python = new PackageProbePython();
        python.Bundled.AddRange(PhoneBundle);
        _host = new AgentAppHost(paths, python: python, installer: installer);
        return _host;
    }

    /// <summary>
    /// The scenario off the phone: <c>pip install lxml</c>, typed out of habit before
    /// <c>import lxml</c>. The index was consulted, found only compiled wheels, and
    /// refused in words that said lxml was unavailable — on a device where lxml was
    /// importable the whole time. Now the bundle is asked first, nothing is fetched,
    /// and the rest of the line runs.
    /// </summary>
    [Theory]
    [InlineData("pip install lxml")]
    [InlineData("python3 -m pip install lxml==6.1.3")]
    [InlineData("pip install python-pptx lxml")]
    [InlineData("pip install Python_PPTX")]
    public void ABundledPackageIsReportedAsAlreadyThereWithoutTouchingTheIndex(string command)
    {
        var installer = new RecordingInstallHook();
        AgentAppHost host = StartWithBundle(network: true, installer, out _);
        SessionWorkspace workspace = host.Workspaces.GetOrCreate("bundled");

        SkillToolResult result = host.CodeRunner!.Execute(
            ShellCall(command + " && echo AFTER"), workspace: workspace);

        Assert.True(result.Ok, result.Content);
        // The sentence the change exists to deliver, not the ledger's generic one: the
        // model must read that the package is part of the app, not that it installed
        // it earlier in this session.
        Assert.Contains("Built into this host, nothing to install", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("Already installed this session", result.Content, StringComparison.Ordinal);
        Assert.Contains("AFTER", result.Content, StringComparison.Ordinal);
        Assert.Empty(installer.Requests);
    }

    /// <summary>
    /// The network switch is about fetching, and a bundled package fetches nothing:
    /// with the switch OFF the same request is still answered from the bundle, while a
    /// package the bundle does not have is still refused for the switch.
    /// </summary>
    [Fact]
    public void WithTheNetworkOffABundledPackageIsStillAnsweredFromTheBundle()
    {
        var installer = new RecordingInstallHook();
        AgentAppHost host = StartWithBundle(network: false, installer, out _);
        SessionWorkspace workspace = host.Workspaces.GetOrCreate("offline");

        SkillToolResult bundled = host.CodeRunner!.Execute(
            ShellCall("pip install lxml && echo AFTER"), workspace: workspace);
        Assert.True(bundled.Ok, bundled.Content);
        Assert.Contains("Built into this host, nothing to install: lxml", bundled.Content, StringComparison.Ordinal);
        Assert.Contains("AFTER", bundled.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("installing is not enabled", bundled.Content, StringComparison.Ordinal);

        // A pin to another version of a COMPILED bundled package fetches nothing either,
        // so with the switch off it is still answered by name rather than by the switch.
        SkillToolResult pinned = host.CodeRunner!.Execute(
            ShellCall("pip install lxml==5.3.0 && echo AFTER"), workspace: workspace);
        Assert.False(pinned.Ok);
        Assert.Contains("lxml 6.1.3 is compiled into this app", pinned.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("installing is not enabled", pinned.Content, StringComparison.Ordinal);

        SkillToolResult fetched = host.CodeRunner!.Execute(
            ShellCall("pip install requests && echo AFTER"), workspace: workspace);
        Assert.False(fetched.Ok);
        Assert.Contains("installing is not enabled", fetched.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("AFTER", fetched.Content, StringComparison.Ordinal);

        // Half bundled is not bundled: the fetch half decides.
        SkillToolResult mixed = host.CodeRunner!.Execute(
            ShellCall("pip install lxml requests && echo AFTER"), workspace: workspace);
        Assert.False(mixed.Ok);
        Assert.Contains("installing is not enabled", mixed.Content, StringComparison.Ordinal);

        Assert.Empty(installer.Requests);
    }

    /// <summary>
    /// A compiled distribution is the one that shipped, full stop: a pin to another
    /// version is refused by name rather than sent to an index that cannot help. A
    /// pure one can be shadowed by a session install, so the same pin goes through.
    /// </summary>
    [Fact]
    public void ReplacingABundledPackageIsRefusedWhenCompiledAndAllowedWhenPure()
    {
        var installer = new RecordingInstallHook();
        AgentAppHost host = StartWithBundle(network: true, installer, out _);
        SessionWorkspace workspace = host.Workspaces.GetOrCreate("replace");

        SkillToolResult compiled = host.CodeRunner!.Execute(
            ShellCall("pip install lxml==5.3.0 && echo AFTER"), workspace: workspace);
        Assert.False(compiled.Ok);
        Assert.Contains("lxml 6.1.3 is compiled into this app", compiled.Content, StringComparison.Ordinal);
        Assert.Contains("cannot be replaced by version 5.3.0", compiled.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("AFTER", compiled.Content, StringComparison.Ordinal);
        Assert.Empty(installer.Requests);

        SkillToolResult pure = host.CodeRunner!.Execute(
            ShellCall("pip install python-pptx==0.6.23"), workspace: workspace);
        Assert.True(pure.Ok, pure.Content);
        InstallRequest request = Assert.Single(installer.Requests);
        Assert.Equal(new[] { "python-pptx==0.6.23" }, request.Packages);
    }

    /// <summary>
    /// <c>pip list</c> is how a model finds out what is here, and an answer that omits
    /// the bundle says lxml is absent. The session's own installs come first; a bundled
    /// distribution the session has shadowed is listed once, as the session's.
    /// </summary>
    [Fact]
    public void PipListShowsTheBundleBehindTheSessionsOwnInstalls()
    {
        var installer = new RecordingInstallHook();
        AgentAppHost host = StartWithBundle(network: true, installer, out _);
        SessionWorkspace workspace = host.Workspaces.GetOrCreate("listing");
        Directory.CreateDirectory(Path.Combine(workspace.EnvDirectory, "python_pptx-0.6.23.dist-info"));

        SkillToolResult listed = host.CodeRunner!.Execute(ShellCall("pip list"), workspace: workspace);

        Assert.True(listed.Ok, listed.Content);
        Assert.Contains("python_pptx==0.6.23", listed.Content, StringComparison.Ordinal);
        Assert.Contains("lxml==6.1.3", listed.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("python-pptx==1.0.2", listed.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// No tool description may name a vendor, product, market or worked example.
    ///
    /// <para>
    /// This is the test for a reported bug, and the bug was a prompt: a complete Yahoo
    /// Finance screener program — 3,204 characters of url, response fields and a
    /// ten-row table — had been added to the shell tool's description to make one
    /// stock-gainers request come out right. A tool description is read on EVERY turn,
    /// so it did not read as guidance, it read as a demonstration of what code here
    /// looks like, and the model reached for finance APIs on requests that had nothing
    /// to do with finance. Whatever is in a declaration has to be true of every task;
    /// a specific one belongs in a skill, which is injected only when it is selected.
    /// </para>
    /// <para>
    /// The vocabulary below is a sample, not a definition — the real rule is the
    /// sentence above, and this catches the shape it takes in practice.
    /// </para>
    /// </summary>
    [Fact]
    public void NoToolDescriptionNamesAVendorProductOrWorkedExample()
    {
        AgentAppHost host = Start(settings =>
        {
            AppSettings everything = settings.Load();
            everything.AllowCodeExecution = true;
            everything.AllowNetwork = true;      // the widest declaration this host emits
            settings.Save(everything);
        });

        string[] mustNotAppear =
        {
            // The one that caused the report, and its neighbours.
            "yahoo", "yfinance", "day_gainers", "scrIds", "quoteType", "regularMarket",
            "screener", "gainers", "ticker", "NASDAQ", "S&P",
            // Other domains a future well-meaning fix might paste in whole.
            "openai", "github.com/", "api_key", "bitcoin", "weather.com",
        };

        foreach (ToolFunction tool in host.CodeRunner!.DeclareTools(persists: true)
                     .Concat(host.CodeRunner.DeclareTools(persists: false)))
        {
            foreach (string banned in mustNotAppear)
            {
                Assert.False(
                    tool.Description.Contains(banned, StringComparison.OrdinalIgnoreCase),
                    $"the '{tool.Name}' tool description names '{banned}'. A declaration is read on every "
                    + "turn, so a task-specific example in it biases every unrelated request; put it in a "
                    + "skill instead.");
            }
        }
    }

    /// <summary>
    /// A narrowed allow-list narrows what the declaration RECOMMENDS, and nothing else.
    ///
    /// <para>
    /// The general execution guidance used to be shown only when one particular finance
    /// host was allow-listed, because the guidance WAS that host's API. Now that every
    /// sentence of it is true of any request, an operator restricting egress must not
    /// also lose the advice about heredoc quoting and not inventing data.
    /// </para>
    /// </summary>
    [Fact]
    public void ARestrictedNetworkDeclarationStillCarriesTheGeneralGuidanceAndNamesNoHost()
    {
        AgentAppHost host = Start(settings =>
        {
            AppSettings enabled = settings.Load();
            enabled.AllowNetwork = true;
            enabled.NetworkHosts = new List<string> { "pypi.org" };
            settings.Save(enabled);
        });

        string declaration = host.CodeRunner!.Declare().Description;
        Assert.Contains("ENABLED only for these host suffixes: pypi.org", declaration,
            StringComparison.Ordinal);
        Assert.DoesNotContain("ENABLED and unrestricted", declaration, StringComparison.Ordinal);
        Assert.Contains("standard library first", declaration, StringComparison.Ordinal);
        Assert.DoesNotContain("query1.finance.yahoo.com", declaration, StringComparison.Ordinal);
    }

    [Fact]
    public void TheEmbeddedPythonIsPublishedWithoutAdvertisingProgramsFromTheHostPath()
    {
        var python = new PackageProbePython();
        _host = new AgentAppHost(
            Paths,
            python: python,
            javaScript: new UnavailableJavaScript());

        Assert.True(CodeEnvironment.IsConfigured);
        Assert.True(
            CodeEnvironment.TryResolveInterpreter(
                CodeLanguage.Python, out string? interpreter, out string? error),
            error);
        Assert.Equal("python3", interpreter);
        Assert.Equal(new Version(3, 13), CodeEnvironment.PythonVersionOf(interpreter!));

        // The development machine has several of the programs ProbeAvailable checks for,
        // but none exists in the iOS in-process shell. Configure must replace that PATH
        // inventory rather than append the embedded runtime to it.
        Assert.Equal(new[] { "python3 3.13-test" }, CodeEnvironment.AvailableTools);
        string inventory = _host.CodeRunner!.Declare().Description
            .Split('\n')
            .Single(line => line.StartsWith("On this host:", StringComparison.Ordinal));
        Assert.StartsWith("On this host: python3 3.13-test.", inventory, StringComparison.Ordinal);
    }

    [Fact]
    public void APackageInstallHiddenInANestedShellCannotBypassTheHostBridge()
    {
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings enabled = settings.Load();
        enabled.AllowNetwork = true;
        settings.Save(enabled);

        var installer = new RecordingInstallHook();
        _host = new AgentAppHost(paths, python: new PackageProbePython(), installer: installer);
        SessionWorkspace workspace = _host.Workspaces.GetOrCreate("nested-install");

        SkillToolResult result = _host.CodeRunner!.Execute(
            ShellCall("sh -c 'pip install img2pdf'"), workspace: workspace);

        Assert.False(result.Ok);
        Assert.Contains(ExecutionPolicy.InstallsByHostMessage, result.Content, StringComparison.Ordinal);
        Assert.Empty(installer.Requests);
        Assert.False(File.Exists(Path.Combine(workspace.EnvDirectory, "img2pdf.py")));
    }

    [Fact]
    public async Task OneWorkspaceWaitsForItsInstallWhileAnotherWorkspaceCanKeepRunning()
    {
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings enabled = settings.Load();
        enabled.AllowNetwork = true;
        settings.Save(enabled);

        var python = new PackageProbePython();
        var installer = new BlockingInstallHook();
        _host = new AgentAppHost(paths, python: python, installer: installer);
        SessionWorkspace installing = _host.Workspaces.GetOrCreate("installing");
        SessionWorkspace independent = _host.Workspaces.GetOrCreate("independent");

        Task<SkillToolResult> install = Task.Run(() =>
            _host.CodeRunner!.Execute(ShellCall("pip install img2pdf"), workspace: installing));
        await installer.Entered.Task.WaitAsync(TimeSpan.FromSeconds(5));

        var importStarted = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<SkillToolResult> import = Task.Factory.StartNew(
            () =>
            {
                importStarted.TrySetResult(true);
                return _host.CodeRunner!.Execute(
                    ShellCall("python3 -c \"import img2pdf; print(img2pdf.answer)\""),
                    workspace: installing);
            },
            CancellationToken.None,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);
        await importStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));

        Task<SkillToolResult> otherWorkspace = Task.Run(() =>
            _host.CodeRunner!.Execute(
                new ToolCall
                {
                    Name = SkillToolNames.WriteFile,
                    Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
                    {
                        ["path"] = "unblocked.txt",
                        ["content"] = "other workspace ran\n",
                    },
                },
                workspace: independent));

        try
        {
            SkillToolResult independentResult =
                await otherWorkspace.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.True(independentResult.Ok, independentResult.Content);
            Assert.False(import.IsCompleted, "a second call entered the same workspace during an install");
            Assert.False(
                python.CodeEntered.Task.IsCompleted,
                "Python observed the package directory while the installer was still writing it");
        }
        finally
        {
            installer.Release.TrySetResult(true);
        }

        SkillToolResult installed = await install.WaitAsync(TimeSpan.FromSeconds(5));
        SkillToolResult imported = await import.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(installed.Ok, installed.Content);
        Assert.True(imported.Ok, imported.Content);
        Assert.Contains("731", imported.Content, StringComparison.Ordinal);
        Assert.Equal("other workspace ran\n", File.ReadAllText(
            Path.Combine(independent.WorkDirectory, "unblocked.txt")));
    }

    /// <summary>
    /// The package command has to cross both seams in the assembled app: the agent host
    /// reads it out of the model's shell line, then TensorAgent hands the validated names
    /// to its in-process wheel installer. Testing either half by itself missed the wiring
    /// between them, so every install on iOS fell through to <c>python3 -m pip</c> even
    /// though the embedded runtime deliberately ships no pip module.
    /// </summary>
    [Theory]
    [InlineData("pip install img2pdf -q && python3 -c \"import img2pdf; print(img2pdf.answer)\"")]
    [InlineData("python3 -m pip install img2pdf && python3 -c \"import img2pdf; print(img2pdf.answer)\"")]
    public void ModelWrittenPipCommandsReachTheInProcessInstallerAndTheNextRunCanImportThePackage(
        string command)
    {
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings enabled = settings.Load();
        enabled.AllowCodeExecution = true;
        enabled.AllowNetwork = true;
        settings.Save(enabled);

        var python = new PackageProbePython();
        var installer = new RecordingInstallHook();
        _host = new AgentAppHost(paths, python: python, installer: installer);
        string declarationBeforeInstall = _host.CodeRunner!.Declare().Description;

        SessionWorkspace workspace = _host.Workspaces.GetOrCreate("install-spelling");
        SkillToolResult result = _host.CodeRunner!.Execute(
            new ToolCall
            {
                Name = "shell",
                Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    ["command"] = command,
                },
            },
            workspace: workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("Installed: img2pdf", result.Content, StringComparison.Ordinal);
        Assert.Contains("731", result.Content, StringComparison.Ordinal);

        InstallRequest request = Assert.Single(installer.Requests);
        Assert.Equal("python", request.Language);
        Assert.Equal(new[] { "img2pdf" }, request.Packages);
        Assert.Equal(workspace.EnvDirectory, request.TargetDirectory);
        Assert.Equal(workspace.EnvDirectory, request.Policy.PackageRoot);
        Assert.True(request.Policy.AllowNetwork);
        Assert.True(File.Exists(Path.Combine(workspace.EnvDirectory, "img2pdf.py")));
        Assert.Equal(declarationBeforeInstall, _host.CodeRunner.Declare().Description);

        // `python -m pip` is a spelling the host accepts from the MODEL, not a module
        // the embedded interpreter should ever be asked to run. Its packages are
        // installed by the hook above.
        Assert.DoesNotContain("pip", python.ModuleCalls, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public void AFailedPackageDoesNotForgetPackagesAlreadyCommittedByTheSameCommand()
    {
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings enabled = settings.Load();
        enabled.AllowNetwork = true;
        settings.Save(enabled);

        var installer = new PartiallyFailingInstallHook();
        _host = new AgentAppHost(paths, python: new PackageProbePython(), installer: installer);
        SessionWorkspace workspace = _host.Workspaces.GetOrCreate("partial-install");

        SkillToolResult first = _host.CodeRunner!.Execute(
            ShellCall("pip install alpha bravo"), workspace: workspace);

        Assert.False(first.Ok);
        Assert.Contains("Installed alpha before this failure", first.Content, StringComparison.Ordinal);
        Assert.Contains("Could not install bravo", first.Content, StringComparison.Ordinal);
        Assert.True(workspace.IsInstalled("python", "alpha"));
        Assert.False(workspace.IsInstalled("python", "bravo"));
        Assert.Equal(new[] { "alpha", "bravo" }, installer.Requested);

        SkillToolResult retry = _host.CodeRunner.Execute(
            ShellCall("pip install alpha"), workspace: workspace);

        Assert.True(retry.Ok, retry.Content);
        Assert.Contains("Already installed this session: alpha", retry.Content, StringComparison.Ordinal);
        Assert.Equal(new[] { "alpha", "bravo" }, installer.Requested);
    }

    [Fact]
    public void EquivalentDistributionNamesAreInstalledOnlyOncePerSession()
    {
        AgentPaths paths = Paths;
        paths.EnsureCreated();
        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings enabled = settings.Load();
        enabled.AllowCodeExecution = true;
        enabled.AllowNetwork = true;
        settings.Save(enabled);

        var installer = new RecordingInstallHook();
        _host = new AgentAppHost(paths, python: new PackageProbePython(), installer: installer);
        SessionWorkspace workspace = _host.Workspaces.GetOrCreate("canonical-package-name");

        SkillToolResult first = _host.CodeRunner!.Execute(
            ShellCall("pip install zope.interface zope-interface ZOPE_interface"),
            workspace: workspace);

        Assert.True(first.Ok, first.Content);
        InstallRequest request = Assert.Single(installer.Requests);
        Assert.Equal(new[] { "zope.interface" }, request.Packages);
        Assert.True(workspace.IsInstalled("python", "zope-interface"));

        SkillToolResult retry = _host.CodeRunner.Execute(
            ShellCall("python3 -m pip install zope_interface"), workspace: workspace);

        Assert.True(retry.Ok, retry.Content);
        Assert.Contains("Already installed this session: zope_interface", retry.Content, StringComparison.Ordinal);
        Assert.Single(installer.Requests);
    }

    private static ToolCall ShellCall(string command) => new()
    {
        Name = SkillToolNames.Shell,
        Arguments = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["command"] = command,
        },
    };

    /// <summary>
    /// With code execution off the runner exists and refuses, rather than not existing.
    ///
    /// <para>
    /// The difference is invisible to the model — the request planner offers the code
    /// tools only for a runner that says <c>CanRun</c>, so a refusing runner and no
    /// runner declare exactly the same thing — and it is the whole of why the switch can
    /// now be flipped without relaunching. Built conditionally, both sandbox switches
    /// took effect "next time TensorAgent starts", which on a phone means never: leaving
    /// an app does not restart it.
    /// </para>
    /// </summary>
    [Fact]
    public void TurningCodeExecutionOffLeavesARunnerThatRefuses()
    {
        AgentAppHost host = Start(settings =>
        {
            AppSettings off = settings.Load();
            off.AllowCodeExecution = false;
            settings.Save(off);
        });

        Assert.NotNull(host.CodeRunner);
        Assert.False(host.CodeRunner!.CanRun);
        Assert.False(host.CodeExec.Enabled);
        Assert.Contains("code execution off", host.DescribeEngine(), StringComparison.Ordinal);
    }

    /// <summary>
    /// Both sandbox switches reach the running app, not merely the settings file.
    ///
    /// <para>
    /// This is the reported bug: "Allow network access" was turned on and <c>curl</c>
    /// went on answering "network access is disabled by the user". Everything a run's
    /// permissions are read from has to move together — the code runner's options, the
    /// installer's standing policy, the shell's host list, and the terms a skill's
    /// scripts are planned against — because a model that finds any one of them stale
    /// reports the switch as broken.
    /// </para>
    /// </summary>
    [Fact]
    public void ChangingTheSandboxSwitchesTakesEffectWithoutARestart()
    {
        AgentAppHost host = Start();
        Assert.False(host.CodeExec.AllowNetwork);
        Assert.False(host.Options.SkillsAllowNetwork);
        Assert.False(host.Installer!.CanInstall);
        Assert.Contains("network off", host.DescribeEngine(), StringComparison.Ordinal);
        Assert.DoesNotContain("standard library first", host.CodeRunner!.Declare().Description,
            StringComparison.Ordinal);

        AppSettings on = host.Settings.Load();
        on.AllowNetwork = true;
        host.Settings.Save(on);
        host.ApplySettings(on);

        Assert.True(host.CodeExec.AllowNetwork);
        Assert.True(host.CodeExec.AllowInstall);
        Assert.True(host.Options.SkillsAllowNetwork);
        Assert.True(host.Installer.CanInstall);
        Assert.Contains("network on", host.DescribeEngine(), StringComparison.Ordinal);
        Assert.Contains("standard library first", host.CodeRunner!.Declare().Description,
            StringComparison.Ordinal);

        // And back off again, because a switch that can only be turned on is half a
        // switch: a user who changes their mind has to be able to.
        AppSettings off = host.Settings.Load();
        off.AllowNetwork = false;
        off.AllowCodeExecution = false;
        host.Settings.Save(off);
        host.ApplySettings(off);

        Assert.False(host.CodeExec.AllowNetwork);
        Assert.False(host.Options.SkillsAllowNetwork);
        Assert.False(host.Installer.CanInstall);
        Assert.False(host.CodeRunner!.CanRun);
        Assert.Contains("code execution off", host.DescribeEngine(), StringComparison.Ordinal);
        Assert.DoesNotContain("standard library first", host.CodeRunner.Declare().Description,
            StringComparison.Ordinal);
    }

    [Fact]
    public void TheToolTimeoutAlsoBoundsHostPerformedPackageInstalls()
    {
        AgentAppHost host = Start(settings =>
        {
            AppSettings configured = settings.Load();
            configured.ToolTimeoutSeconds = 9;
            settings.Save(configured);
        });

        Assert.Equal(TimeSpan.FromSeconds(9), host.CodeExec.Timeout);
        Assert.Equal(host.CodeExec.Timeout, host.CodeExec.InstallTimeout);

        AppSettings changed = host.Settings.Load();
        changed.ToolTimeoutSeconds = 480;
        host.ApplySettings(changed);

        Assert.Equal(TimeSpan.FromSeconds(480), host.CodeExec.Timeout);
        Assert.Equal(host.CodeExec.Timeout, host.CodeExec.InstallTimeout);
    }

    /// <summary>
    /// The skills switch is a switch, not a filter over a list.
    ///
    /// <para>
    /// Off has to mean the request planner builds no plan at all — no skill declared,
    /// none reachable — because the cost the user is turning off is the declaration
    /// itself: twelve skills announce themselves in every prompt, which on a phone is
    /// thousands of tokens per turn of every chat. A page that merely stopped ticking
    /// them would still pay for all of it.
    /// </para>
    /// </summary>
    [Fact]
    public void TurningSkillsOffStopsThemBeingDeclaredAtAllAndTakesEffectAtOnce()
    {
        AgentAppHost host = Start();
        Assert.True(host.Options.SkillsEnabled);

        AppSettings off = host.Settings.Load();
        off.SkillsEnabled = false;
        host.Settings.Save(off);
        host.ApplySettings(off);
        Assert.False(host.Options.SkillsEnabled);

        // And back: a switch that only turns off is half a switch.
        AppSettings on = host.Settings.Load();
        on.SkillsEnabled = true;
        host.Settings.Save(on);
        host.ApplySettings(on);
        Assert.True(host.Options.SkillsEnabled);
    }

    /// <summary>
    /// A host started with the switch already off starts with it off, and says so on
    /// the route the page reads before it paints the list.
    /// </summary>
    [Fact]
    public async Task AHostStartedWithSkillsOffReportsThemOffToThePage()
    {
        Start(settings =>
        {
            AppSettings s = settings.Load();
            s.SkillsEnabled = false;
            settings.Save(s);
        });

        Assert.False(_host!.Options.SkillsEnabled);
        JsonElement skills = await Get("/api/skills");
        Assert.False(skills.GetProperty("enabled").GetBoolean());

        // Saving it back through the route the switch uses applies it live.
        HttpResponseMessage saved = await _client!.PostAsJsonAsync("/api/agent/settings", new
        {
            skillsEnabled = true,
            allowCodeExecution = true,
            allowNetwork = false,
            maxTokens = 2048,
        });
        Assert.True(saved.IsSuccessStatusCode);
        Assert.True(_host.Options.SkillsEnabled);
        Assert.True((await Get("/api/skills")).GetProperty("enabled").GetBoolean());
    }

    [Fact]
    public void TheNetworkSwitchReachesTheSandboxAndNotJustTheSettingsFile()
    {
        AgentAppHost allowed = Start(settings =>
        {
            AppSettings on = settings.Load();
            on.AllowNetwork = true;
            settings.Save(on);
        });

        Assert.True(allowed.CodeExec.AllowNetwork);
        Assert.Contains("network on", allowed.DescribeEngine(), StringComparison.Ordinal);
    }

    [Fact]
    public void ByDefaultTheNetworkIsOffAndACommandThatReachesForItIsRefused()
    {
        AgentAppHost host = Start();
        Assert.False(host.CodeExec.AllowNetwork);

        string work = Path.Combine(_root, "work");
        Directory.CreateDirectory(work);
        ConfinedResult result = ((IShellBackend)host.Backend).Run(new ShellLaunch
        {
            Argv = new[] { "sh", "-c", "curl https://example.com" },
            WorkingDirectory = work,
            WriteDirectory = work,
            ReadOnlyDirectory = work,
            Timeout = TimeSpan.FromSeconds(10),
        });
        Assert.False(result.Ok);
        Assert.Contains("network", result.Stderr, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TheEngineDescriptionReachesThePageAndNamesWhatIsMissing()
    {
        Start();
        JsonElement body = await Get("/api/agent/engine");
        string engine = body.GetProperty("engine").GetString()!;
        Assert.Contains("sh (in-process)", engine, StringComparison.Ordinal);
        Assert.Contains("no python", engine, StringComparison.Ordinal);
        Assert.Contains("network off", engine, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CreatingASessionMintsAConversationAndSaysWhichOne()
    {
        AgentAppHost host = Start();
        JsonElement created = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());

        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;
        Assert.NotEmpty(sessionId);
        Assert.NotEmpty(conversationId);
        Assert.Equal(conversationId, host.Recorder.ConversationFor(sessionId));
        Assert.NotNull(host.Conversations.Load(conversationId));
    }

    [Fact]
    public async Task ResumingASessionHandsBackTheSavedTranscript()
    {
        AgentAppHost host = Start();
        Conversation saved = host.Conversations.Create();
        saved.Messages.Add(new StoredMessage { Role = "user", Content = "what is a tensor" });
        saved.Messages.Add(new StoredMessage { Role = "assistant", Content = "an array with a shape" });
        host.Conversations.Save(saved);

        JsonElement resumed = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync($"/api/sessions?conversation={saved.Id}", null)).Content.ReadAsStringAsync());

        Assert.Equal(saved.Id, resumed.GetProperty("conversationId").GetString());
        JsonElement messages = resumed.GetProperty("messages");
        Assert.Equal(2, messages.GetArrayLength());
        Assert.Equal("what is a tensor", messages[0].GetProperty("content").GetString());
        Assert.Equal("what is a tensor", resumed.GetProperty("title").GetString());
    }

    [Fact]
    public async Task ATurnIsWrittenToDiskSoTheChatSurvivesTheAppBeingKilled()
    {
        AgentAppHost host = Start();
        JsonElement created = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;

        // The recorder is what /api/chat calls once a request is accepted. Driving it
        // directly is deliberate: no model is loaded here, so a real chat request would
        // be refused before it ever reached this hook.
        host.Recorder.Record(sessionId, JsonSerializer.Deserialize<JsonElement>("""
            {
              "model": "gemma-4-e2b-q8",
              "think": true,
              "skills": ["pdf"],
              "messages": [
                { "role": "user", "content": "summarise this", "textFileNames": ["report.pdf"] },
                { "role": "assistant", "content": "here is the summary", "thinking": "reading it" }
              ]
            }
            """));

        // Read it back through a second store over the same directory, which is what a
        // relaunch does.
        var reopened = new ConversationStore(host.Paths.ConversationsDirectory);
        Conversation? conversation = reopened.Load(conversationId);
        Assert.NotNull(conversation);
        Assert.Equal(2, conversation!.Messages.Count);
        Assert.Equal("summarise this", conversation.Messages[0].Content);
        Assert.Equal(new List<string> { "report.pdf" }, conversation.Messages[0].TextFileNames);
        Assert.Equal("reading it", conversation.Messages[1].Thinking);
        Assert.Equal("gemma-4-e2b-q8", conversation.ModelId);
        Assert.True(conversation.Think);
        Assert.Equal(new List<string> { "pdf" }, conversation.Skills);
        Assert.Equal("summarise this", conversation.Title);
    }

    [Fact]
    public async Task TheAnswerIsSavedWhenTheTurnEndsAndNotOnlyWhenTheNextOneStarts()
    {
        // The gap that loses exactly the message the user is reading. Recording the
        // request saves the history the page sent, which does not yet contain the
        // reply being generated; if the app is killed after the answer appears and
        // before another question is asked, that answer is gone.
        AgentAppHost host = Start();
        JsonElement created = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;

        host.Recorder.Record(sessionId, JsonSerializer.Deserialize<JsonElement>(
            """{ "messages": [ { "role": "user", "content": "what is 2 + 2" } ] }"""));
        host.Recorder.Complete(sessionId, "It is 4.", "adding them");

        Conversation saved = new ConversationStore(host.Paths.ConversationsDirectory).Load(conversationId)!;
        Assert.Equal(2, saved.Messages.Count);
        Assert.Equal("assistant", saved.Messages[1].Role);
        Assert.Equal("It is 4.", saved.Messages[1].Content);
        Assert.Equal("adding them", saved.Messages[1].Thinking);
    }

    [Fact]
    public async Task RegeneratingAnAnswerReplacesItRatherThanAppendingASecond()
    {
        AgentAppHost host = Start();
        JsonElement created = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;

        host.Recorder.Record(sessionId, JsonSerializer.Deserialize<JsonElement>(
            """{ "messages": [ { "role": "user", "content": "hello" } ] }"""));
        host.Recorder.Complete(sessionId, "first attempt");
        host.Recorder.Complete(sessionId, "second attempt");

        Conversation saved = new ConversationStore(host.Paths.ConversationsDirectory).Load(conversationId)!;
        Assert.Equal(2, saved.Messages.Count);
        Assert.Equal("second attempt", saved.Messages[1].Content);
    }

    [Fact]
    public async Task AnAnswerForAnUnboundSessionIsIgnoredRatherThanMisfiled()
    {
        AgentAppHost host = Start();
        Conversation other = host.Conversations.Create();
        host.Recorder.Complete("a-session-nobody-bound", "this belongs to nothing");
        Assert.Empty(host.Conversations.Load(other.Id)!.Messages);
    }

    [Fact]
    public async Task OpeningTheAppRepeatedlyWithoutTypingLeavesOneEmptyChatRatherThanMany()
    {
        // The page creates a session on every load and a session binds a
        // conversation, so twenty launches used to leave twenty "Chat Sep 2, 07:38"
        // rows pushing the real conversations off the screen. An empty conversation
        // is not a chat the user had; it is one they were about to have.
        AgentAppHost host = Start();
        var ids = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < 5; i++)
        {
            JsonElement created = JsonSerializer.Deserialize<JsonElement>(
                await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
            ids.Add(created.GetProperty("conversationId").GetString()!);
        }

        Assert.Single(ids);
        Assert.Single(host.Conversations.List(includeEmpty: true));
        // And an untouched one is not offered as a saved chat at all.
        Assert.Empty(host.Conversations.List());
    }

    [Fact]
    public async Task OnceAChatHasAMessageTheNextLaunchStartsAFreshOne()
    {
        AgentAppHost host = Start();
        JsonElement first = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
        string used = first.GetProperty("conversationId").GetString()!;
        host.Recorder.Record(first.GetProperty("sessionId").GetString()!, JsonSerializer.Deserialize<JsonElement>(
            """{ "messages": [ { "role": "user", "content": "hello" } ] }"""));

        JsonElement second = JsonSerializer.Deserialize<JsonElement>(
            await (await _client.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
        Assert.NotEqual(used, second.GetProperty("conversationId").GetString());

        // The one with a message is listed; the fresh empty one is not.
        IReadOnlyList<ConversationSummary> listed = host.Conversations.List();
        Assert.Single(listed);
        Assert.Equal(used, listed[0].Id);
    }

    [Fact]
    public async Task DisposingASessionUnbindsItWithoutDeletingTheConversation()
    {
        AgentAppHost host = Start();
        JsonElement created = JsonSerializer.Deserialize<JsonElement>(
            await (await _client!.PostAsync("/api/sessions?conversation=new", null)).Content.ReadAsStringAsync());
        string sessionId = created.GetProperty("sessionId").GetString()!;
        string conversationId = created.GetProperty("conversationId").GetString()!;

        await _client.DeleteAsync($"/api/sessions/{sessionId}");

        Assert.Null(host.Recorder.ConversationFor(sessionId));
        Assert.NotNull(host.Conversations.Load(conversationId));
    }

    [Fact]
    public async Task TheCatalogOfferedIsTheOneThisDeviceCanActuallyRun()
    {
        Start();
        JsonElement body = await Get("/api/agent/catalog");
        foreach (JsonElement model in body.GetProperty("models").EnumerateArray())
            Assert.True(model.GetProperty("minDeviceMemoryGB").GetInt32() <= 12,
                $"{model.GetProperty("id").GetString()} needs more memory than the device tier allows");
    }

    [Fact]
    public async Task The32GbCatalogApiOffersTheVisionIq1MEntryAndKeepsQ2Gated()
    {
        StartServing(Paths with { DeviceMemoryGB = 32, DeviceClass = DeviceClass.Desktop });
        JsonElement body = await Get("/api/agent/catalog");
        JsonElement[] models = body.GetProperty("models").EnumerateArray().ToArray();
        JsonElement iq1m = Assert.Single(models, m => m.GetProperty("id").GetString() == "qwen3.8-flash-next-iq1m");
        Assert.Equal(32, iq1m.GetProperty("minDeviceMemoryGB").GetInt32());
        Assert.Equal("UD-IQ1_M", iq1m.GetProperty("quantization").GetString());
        Assert.Equal("Image, Video", iq1m.GetProperty("modalities").GetString());
        Assert.True(iq1m.GetProperty("experimental").GetBoolean());
        Assert.Equal(32768, iq1m.GetProperty("contextLength").GetInt32());
        Assert.Equal(74_538_755_776, iq1m.GetProperty("totalBytes").GetInt64());
        Assert.Equal("Qwen Community License 1.0", iq1m.GetProperty("license").GetString());
        Assert.DoesNotContain(models, m => m.GetProperty("id").GetString() == "qwen3.8-flash-next-q2kxl");
        Assert.Equal(AgentAppHost.ModelLoadState.None, _host!.ModelLoad);
    }

    [Fact]
    public async Task TheModelsRouteAdvertisesMetalFirstBecauseThatIsWhyItIsOnAPhone()
    {
        Start();
        JsonElement body = await Get("/api/models");
        Assert.Equal("ggml_metal", body.GetProperty("defaultBackend").GetString());
        JsonElement backends = body.GetProperty("supportedBackends");
        Assert.Equal("ggml_metal", backends[0].GetProperty("Value").GetString());
        Assert.Contains(backends.EnumerateArray(), b => b.GetProperty("Value").GetString() == "ggml_cpu");
    }

    [Fact]
    public async Task ChangingASettingThroughTheApiIsWhatTheNextLaunchReads()
    {
        AgentAppHost host = Start();
        await _client!.PostAsJsonAsync("/api/agent/settings", new
        {
            allowCodeExecution = true,
            allowNetwork = true,
            maxTokens = 1024,
        });

        Assert.True(new SettingsStore(host.Paths.SettingsFile).Load().AllowNetwork);

        // A restart is what applies it: the running host built its options at startup.
        host.Dispose();
        using var restarted = new AgentAppHost(Paths);
        Assert.True(restarted.CodeExec.AllowNetwork);
        Assert.Equal(1024, restarted.Options.DefaultMaxTokens);
        _host = null;
    }

    [Fact]
    public async Task ShuttingDownStopsServingBeforeItReleasesTheEngine()
    {
        // The order is the whole test. A request in flight is usually inside the
        // model, so releasing the engine first unmaps weights that native compute
        // threads are still reading and the process dies with a segmentation fault in
        // an unrelated-looking kernel. This is what that cost, found by an end-to-end
        // test that stopped a generation partway.
        AgentAppHost host = Start();
        var entered = new TaskCompletionSource();
        var finished = new TaskCompletionSource();

        // A handler that does NOT stop when asked, which is what a native compute
        // already inside a graph looks like: cancellation is delivered between
        // tokens, and until the next one it keeps reading the weights.
        host.Server.MapGet("/api/agent/slow", async (_, _) =>
        {
            entered.TrySetResult();
            await Task.Delay(TimeSpan.FromSeconds(2), CancellationToken.None);
            finished.TrySetResult();
            return LoopbackResponse.Json(new { ok = true });
        });

        Task pending = _client!.GetAsync("/api/agent/slow");
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(10));

        var clock = System.Diagnostics.Stopwatch.StartNew();
        host.Dispose();
        clock.Stop();
        _host = null;

        Assert.True(finished.Task.IsCompleted, "shutdown returned while a request was still running");
        Assert.True(clock.Elapsed > TimeSpan.FromSeconds(1),
            $"shutdown returned in {clock.Elapsed.TotalMilliseconds:0}ms without waiting for the request");
        try { await pending; } catch { /* the connection closes with the server */ }
    }

    [Fact]
    public void AnAllowListInSettingsReachesThePolicyEveryRuntimeChecks()
    {
        // Three runtimes check ExecutionPolicy.NetworkHosts, and until this was wired
        // nothing in the app could ever set it: the list was always empty and
        // IsHostAllowed short-circuited to true, so the check read as enforcement and
        // enforced nothing.
        AgentAppHost host = Start(settings =>
        {
            AppSettings s = settings.Load();
            s.AllowNetwork = true;
            s.NetworkHosts = new List<string> { "pypi.org" };
            settings.Save(s);
        });

        string work = Path.Combine(_root, "hosts");
        Directory.CreateDirectory(work);
        ConfinedResult refused = ((IShellBackend)host.Backend).Run(new ShellLaunch
        {
            Argv = new[] { "sh", "-c", "curl https://example.com" },
            WorkingDirectory = work,
            WriteDirectory = work,
            ReadOnlyDirectory = work,
            AllowNetwork = true,
            Timeout = TimeSpan.FromSeconds(15),
        });

        Assert.False(refused.Ok);
        Assert.Contains(ExecutionPolicy.HostNotAllowedSuffix, refused.Stderr, StringComparison.Ordinal);
        // And it must say so rather than reporting the network as off, which is the
        // advice that sends someone to a switch that is already on.
        Assert.DoesNotContain(ExecutionPolicy.NetworkDisabledMessage, refused.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheApiIsClosedToAnythingWithoutTheLaunchToken()
    {
        AgentAppHost host = Start();
        using var bare = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl) };
        Assert.Equal(HttpStatusCode.Forbidden, (await bare.GetAsync("/api/agent/settings")).StatusCode);
    }
    [Fact]
    public async Task ShuttingDownWaitsForTheEngineItselfAndNotOnlyForTheRequests()
    {
        // Draining the requests is not the same as draining the engine, which is what
        // the neighbouring test covers and where the segmentation fault came back from.
        // Cancellation reaches a generation between tokens, so the HTTP request can be
        // finished while the engine's own threads are still inside a graph compute, and
        // releasing the model at that moment unmaps the weights a ggml kernel is
        // reading. The engine's live counters exist only once a model is loaded, which
        // is not something a unit test can have: what the shutdown asks is substituted
        // here, what it does with the answer is the real thing.
        using var idle = new ManualResetEventSlim(initialState: false);
        // Both are completed from inside the shutdown itself, so their continuations
        // have to go elsewhere: run inline, the rest of this test would execute on the
        // thread that is trying to shut the host down.
        var polled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var released = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        var host = new AgentAppHost(Paths, modelService: new ReleaseRecordingModelService(() => released.TrySetResult()));
        _host = host;
        host.EngineHasWorkInFlight = () =>
        {
            polled.TrySetResult();
            return !idle.IsSet;
        };
        host.Start();

        try
        {
            Task shutdown = Task.Run(host.Dispose);

            Task asked = await Task.WhenAny(polled.Task, shutdown);
            Assert.True(ReferenceEquals(asked, polled.Task),
                "the shutdown finished without ever asking the engine whether it was still working");

            // Long enough that a shutdown which is not waiting has finished by now.
            await Task.Delay(TimeSpan.FromMilliseconds(250));
            Assert.False(released.Task.IsCompleted, "the weights were freed while the engine was still computing");
            Assert.False(shutdown.IsCompleted, "the shutdown returned while the engine was still computing");

            idle.Set();
            Task finished = await Task.WhenAny(shutdown, Task.Delay(TimeSpan.FromSeconds(20)));
            Assert.True(ReferenceEquals(finished, shutdown), "the engine went idle and the shutdown never returned");
            await shutdown;
            _host = null;

            Assert.True(released.Task.IsCompleted, "the engine went idle and the model was never released");
        }
        finally
        {
            // However this ends, the stand-in stops claiming work, so tearing the
            // fixture down cannot sit in the drain for the full timeout.
            idle.Set();
        }
    }

    private sealed class ReleaseRecordingModelService(Action onRelease) : ModelService, IDisposable
    {
        void IDisposable.Dispose()
        {
            onRelease();
            base.Dispose();
        }
    }

    /// <summary>
    /// A deterministic stand-in for the app's network wheel installer. It writes one
    /// importable module into the exact target it was handed, making the test about the
    /// assembled routing and environment rather than PyPI availability.
    /// </summary>
    private sealed class RecordingInstallHook : IInstallHook
    {
        public List<InstallRequest> Requests { get; } = new();

        public bool CanInstall => true;

        public string? UnavailableReason => null;

        public Task<ExecutionResult> InstallAsync(
            InstallRequest request, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Requests.Add(request);
            Directory.CreateDirectory(request.TargetDirectory);
            File.WriteAllText(Path.Combine(request.TargetDirectory, "img2pdf.py"), "answer = 731\n");
            return Task.FromResult(new ExecutionResult(
                0,
                "installed img2pdf probe\n",
                string.Empty,
                false,
                request.TargetDirectory,
                new Dictionary<string, string>(StringComparer.Ordinal),
                TimeSpan.Zero));
        }
    }

    private sealed class BlockingInstallHook : IInstallHook
    {
        public TaskCompletionSource<bool> Entered { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public TaskCompletionSource<bool> Release { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public bool CanInstall => true;

        public string? UnavailableReason => null;

        public async Task<ExecutionResult> InstallAsync(
            InstallRequest request, CancellationToken cancellationToken)
        {
            Entered.TrySetResult(true);
            await Release.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
            Directory.CreateDirectory(request.TargetDirectory);
            File.WriteAllText(Path.Combine(request.TargetDirectory, "img2pdf.py"), "answer = 731\n");
            return new ExecutionResult(
                0,
                "installed img2pdf probe\n",
                string.Empty,
                false,
                request.TargetDirectory,
                new Dictionary<string, string>(StringComparer.Ordinal),
                TimeSpan.Zero);
        }
    }

    private sealed class PartiallyFailingInstallHook : IInstallHook
    {
        public List<string> Requested { get; } = new();

        public bool CanInstall => true;

        public string? UnavailableReason => null;

        public Task<ExecutionResult> InstallAsync(
            InstallRequest request, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string package = Assert.Single(request.Packages);
            Requested.Add(package);
            return Task.FromResult(string.Equals(package, "bravo", StringComparison.Ordinal)
                ? ExecutionResult.Failed(
                    "the probe rejects bravo", request.TargetDirectory,
                    new Dictionary<string, string>(StringComparer.Ordinal))
                : new ExecutionResult(
                    0,
                    "installed alpha probe\n",
                    string.Empty,
                    false,
                    request.TargetDirectory,
                    new Dictionary<string, string>(StringComparer.Ordinal),
                    TimeSpan.Zero));
        }
    }

    /// <summary>
    /// Enough Python to prove that a package in the session's PYTHONPATH is visible.
    /// Asking it to run a module always fails deliberately, so a regression that sends
    /// <c>-m pip</c> into embedded CPython cannot be mistaken for an install success.
    /// </summary>
    private sealed class PackageProbePython : IPythonRuntime
    {
        public List<string> ModuleCalls { get; } = new();

        /// <summary>What this probe claims the app bundle ships; empty by default.</summary>
        public List<BundledDistribution> Bundled { get; } = new();

        public IReadOnlyList<BundledDistribution> BundledDistributions => Bundled;

        public TaskCompletionSource<bool> CodeEntered { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public bool IsAvailable => true;

        public string? UnavailableReason => null;

        public string Version => "3.13-test";

        public Task<ExecutionResult> RunScriptAsync(
            string scriptPath,
            IReadOnlyList<string> arguments,
            InterpreterContext context,
            CancellationToken cancellationToken) =>
            Task.FromResult(ExecutionResult.Failed(
                $"the probe does not run scripts: {scriptPath}", context.WorkingDirectory, context.Environment));

        public Task<ExecutionResult> RunCodeAsync(
            string source,
            IReadOnlyList<string> arguments,
            InterpreterContext context,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CodeEntered.TrySetResult(true);
            string? installed = context.PathEntries("PYTHONPATH")
                .Select(directory => Path.Combine(directory, "img2pdf.py"))
                .FirstOrDefault(File.Exists);
            if (installed is null || !source.Contains("import img2pdf", StringComparison.Ordinal))
            {
                return Task.FromResult(ExecutionResult.Failed(
                    "ModuleNotFoundError: No module named 'img2pdf'",
                    context.WorkingDirectory,
                    context.Environment));
            }

            string module = File.ReadAllText(installed);
            string stdout = module.Contains("answer = 731", StringComparison.Ordinal)
                ? "731\n"
                : string.Empty;
            return Task.FromResult(new ExecutionResult(
                0,
                stdout,
                string.Empty,
                false,
                context.WorkingDirectory,
                context.Environment,
                TimeSpan.Zero));
        }

        public Task<ExecutionResult> RunModuleAsync(
            string module,
            IReadOnlyList<string> arguments,
            InterpreterContext context,
            CancellationToken cancellationToken)
        {
            ModuleCalls.Add(module);
            return Task.FromResult(ExecutionResult.Failed(
                $"No module named '{module}'", context.WorkingDirectory, context.Environment));
        }

        public Task<SyntaxCheckResult> CheckSyntaxAsync(
            string scriptPath, CancellationToken cancellationToken) =>
            Task.FromResult(SyntaxCheckResult.Passed);
    }

    private sealed class UnavailableJavaScript : IJavaScriptRuntime
    {
        public bool IsAvailable => false;

        public string? UnavailableReason => "JavaScript is deliberately absent in this test";

        public Task<ExecutionResult> RunScriptAsync(
            string scriptPath,
            IReadOnlyList<string> arguments,
            InterpreterContext context,
            CancellationToken cancellationToken) =>
            throw new InvalidOperationException(UnavailableReason);

        public Task<ExecutionResult> RunCodeAsync(
            string source,
            IReadOnlyList<string> arguments,
            InterpreterContext context,
            CancellationToken cancellationToken) =>
            throw new InvalidOperationException(UnavailableReason);

        public Task<SyntaxCheckResult> CheckSyntaxAsync(
            string scriptPath, CancellationToken cancellationToken) =>
            throw new InvalidOperationException(UnavailableReason);
    }

    /// <summary>
    /// Choosing a model repoints the engine, so the choice is real before the method
    /// returns.
    ///
    /// <para>
    /// The engine allows one hosted model per process because the desktop server is
    /// launched against one --model, and the app inherited that: picking a model in
    /// TensorAgent's own list saved a setting and nothing else, so the Models page said
    /// "selected" while /api/chat kept answering "No model is configured" until the app
    /// was restarted. On a phone that is indistinguishable from a broken button, and it
    /// is what a user reported. This pins the half that is testable without weights:
    /// after UseModel, the guard the chat route consults resolves the NEW file, and
    /// before it, it does not.
    /// </para>
    /// </summary>
    [Fact]
    public void ChoosingAModelRepointsTheHostedModelRatherThanWaitingForARestart()
    {
        AgentAppHost host = Start();

        // The guard the chat route consults. Before anything is chosen it holds the
        // placeholder path, so a request naming a real model is refused.
        const string wanted = "gemma-4-E2B-it-Q8_0.gguf";
        Assert.False(
            TensorSharp.Server.Hosting.HostedModelGuard.TryResolveHostedModelRequest(
                wanted, host.Options.StartupModelPath, out _, out string before),
            "nothing has been chosen yet, so this model must not resolve");
        Assert.Contains("not hosted", before, StringComparison.OrdinalIgnoreCase);

        // UseModel loads weights, which a unit test has none of; repointing is the part
        // that decides whether the choice is visible, and it is what is checked here.
        CatalogModel model = ModelCatalog.BuiltIn.Single(m => m.Id == "gemma-4-e2b-q8");
        AppSettings settings = host.Settings.Load();
        settings.SelectedModelId = model.Id;
        host.Settings.Save(settings);
        host.Options.RepointHostedModel(
            host.Paths.SelectedModelPath(settings), host.Paths.SelectedProjectorPath(settings));

        Assert.True(
            TensorSharp.Server.Hosting.HostedModelGuard.TryResolveHostedModelRequest(
                wanted, host.Options.StartupModelPath, out string resolved, out string after),
            $"after choosing {model.Id} the chat route must accept it: {after}");
        Assert.EndsWith(wanted, resolved, StringComparison.Ordinal);
    }


    // ---- coming back to the foreground ---------------------------------------------
    //
    // iOS reclaims a suspended app's sockets and the managed HttpListener cannot tell
    // (LoopbackServerTests has the transport's half). This is the host's half: on every
    // return to the foreground the listener is probed, rebuilt if dead, the outcome is
    // written to background.log — the one record readable after the fact on a phone —
    // and whoever owns the WebView is told, including when the page's origin moved.

    private static HttpClient ClientWithToken(AgentAppHost host)
    {
        var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{Core.Hosting.LoopbackServer.TokenCookie}={host.Server.Token}");
        return client;
    }

    private static readonly TimeSpan Bound = TimeSpan.FromSeconds(10);

    private static string BackgroundTrace(AgentAppHost host)
        => File.ReadAllText(Path.Combine(host.Paths.LogsDirectory, "background.log"));

    [Fact]
    public async Task ComingToTheForegroundWritesTheListenerCheckToTheBackgroundTrace()
    {
        AgentAppHost host = Start();
        var checks = new List<AgentAppHost.ForegroundReport>();
        var moves = new List<string>();
        host.ForegroundChecked += checks.Add;
        host.EntryUrlChanged += moves.Add;

        AgentAppHost.ForegroundReport report = await host.OnForegroundAsync().WaitAsync(Bound);

        Assert.Equal(Core.Hosting.LoopbackServer.ListenerHealth.Alive, report.Listener);
        Assert.False(report.EntryUrlChanged);
        Assert.Equal(host.Server.Port, report.Port);
        Assert.Equal(0, host.ListenerRestarts);
        Assert.Equal(new[] { report }, checks);
        Assert.Empty(moves);
        Assert.Contains("foreground: loopback listener alive", BackgroundTrace(host), StringComparison.Ordinal);
    }

    [SkippableFact]
    public async Task ComingToTheForegroundRebuildsADeadListener()
    {
        Skip.If(!ListeningSockets.ManagedHttpListenerInUse, ListeningSockets.WhyNotManaged);
        AgentAppHost host = Start();
        host.ListenerProbeTimeout = TimeSpan.FromSeconds(2);
        int port = host.Server.Port;
        string entry = host.EntryUrl;
        JsonElement before = await Get("/api/agent/engine").WaitAsync(Bound);
        Assert.True(before.TryGetProperty("engine", out _));
        var checks = new List<AgentAppHost.ForegroundReport>();
        var moves = new List<string>();
        host.ForegroundChecked += checks.Add;
        host.EntryUrlChanged += moves.Add;

        ListeningSockets.Kill(port);

        AgentAppHost.ForegroundReport report = await host.OnForegroundAsync().WaitAsync(Bound);

        Assert.Equal(Core.Hosting.LoopbackServer.ListenerHealth.Restarted, report.Listener);
        Assert.Equal(port, report.Port);
        Assert.Equal(port, host.Server.Port);
        Assert.False(report.EntryUrlChanged);
        Assert.Equal(entry, host.EntryUrl);
        Assert.Equal(1, host.ListenerRestarts);
        Assert.Equal(new[] { report }, checks);
        Assert.Empty(moves);
        Assert.Contains("foreground: loopback listener DEAD", BackgroundTrace(host), StringComparison.Ordinal);

        // The page's requests go through again; a fresh client, because the pooled
        // connection belonged to the incarnation that was just retired.
        using HttpClient fresh = ClientWithToken(host);
        JsonElement after = JsonSerializer.Deserialize<JsonElement>(
            await fresh.GetStringAsync("/api/agent/engine").WaitAsync(Bound));
        Assert.True(after.TryGetProperty("engine", out _));
    }

    [SkippableFact]
    public async Task ComingToTheForegroundMovesPortsOnlyWhenItMust()
    {
        Skip.If(!ListeningSockets.ManagedHttpListenerInUse, ListeningSockets.WhyNotManaged);
        AgentAppHost host = Start();
        host.ListenerProbeTimeout = TimeSpan.FromSeconds(1);
        int oldPort = host.Server.Port;
        string token = host.Server.Token;
        string oldEntry = host.EntryUrl;
        var checks = new List<AgentAppHost.ForegroundReport>();
        var moves = new List<string>();
        host.ForegroundChecked += checks.Add;
        host.EntryUrlChanged += moves.Add;

        ListeningSockets.Kill(oldPort);
        var squatter = new TcpListener(IPAddress.Loopback, oldPort);
        squatter.Start();
        try
        {
            AgentAppHost.ForegroundReport report = await host.OnForegroundAsync().WaitAsync(Bound);

            Assert.Equal(Core.Hosting.LoopbackServer.ListenerHealth.Relisted, report.Listener);
            Assert.True(report.EntryUrlChanged);
            Assert.NotEqual(oldPort, host.Server.Port);
            Assert.Equal(host.Server.Port, report.Port);
            Assert.Equal(1, host.ListenerRestarts);

            // The WebView's owner was told where the page now lives: same token, new port.
            string moved = Assert.Single(moves);
            Assert.Equal(host.EntryUrl, moved);
            Assert.Equal($"http://127.0.0.1:{host.Server.Port}/?token={token}", moved);
            Assert.NotEqual(oldEntry, moved);
            Assert.Equal(new[] { report }, checks);

            string trace = BackgroundTrace(host);
            Assert.Contains("foreground: loopback listener DEAD", trace, StringComparison.Ordinal);
            Assert.Contains("the page will be reloaded at " + host.EntryUrl, trace, StringComparison.Ordinal);

            using HttpClient fresh = ClientWithToken(host);
            JsonElement after = JsonSerializer.Deserialize<JsonElement>(
                await fresh.GetStringAsync("/api/agent/engine").WaitAsync(Bound));
            Assert.True(after.TryGetProperty("engine", out _));
        }
        finally
        {
            squatter.Dispose();
        }
    }
}
