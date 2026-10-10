// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Net;
using System.Security.Cryptography;
using System.Text.Json;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Settings;
using TensorSharp.Runtime;

namespace TensorAgent.Tests;

/// <summary>
/// The LoRA plug-ins the app offers (<see cref="LoraCatalog"/>), the store that keeps them
/// (<see cref="LoraStore"/>) and the rules a choice obeys (<see cref="LoraSelection"/>).
/// </summary>
public sealed class LoraCatalogTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ta-lora-" + Guid.NewGuid().ToString("n"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    private static readonly JsonDocumentOptions Jsonc = new() { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !Directory.Exists(Path.Combine(directory.FullName, "config", "lora")))
            directory = directory.Parent;
        Assert.NotNull(directory);
        return directory!.FullName;
    }

    private static string EngineConfigPath(string id) => Path.Combine(RepoRoot(), "config", "lora", id + ".json");

    [Fact]
    public void EveryPlugInIsWellFormed()
    {
        var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (CatalogLora lora in LoraCatalog.BuiltIn)
        {
            Assert.True(ids.Add(lora.Id), $"duplicate id {lora.Id}");
            Assert.Matches("^[a-z0-9.-]+$", lora.Id);
            CatalogModel? model = ModelCatalog.Find(lora.BaseModelId);
            Assert.True(model is { Family: CatalogFamily.QwenImage }, $"{lora.Id}: the base model must be a Qwen-Image entry");
            Assert.NotEmpty(lora.DisplayName);
            Assert.NotEmpty(lora.Purpose);
            Assert.NotEmpty(lora.License);
            Assert.EndsWith(".safetensors", lora.Weights.FileName);
            Assert.Equal(lora.Files.Count, lora.Files.Select(f => f.FileName).Distinct(StringComparer.OrdinalIgnoreCase).Count());
            foreach (LoraFile file in lora.Files)
            {
                Assert.False(file.FileName.Contains('/'), $"{lora.Id}: file names are bare ({file.FileName})");
                // Pinned to a commit, not a branch: a plug-in is small enough that its
                // publisher re-uploads it, and the hash below would then refuse the download.
                Assert.Matches("^https://huggingface\\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/.+$", file.Url);
                Assert.EndsWith("/" + file.FileName, file.Url);
                Assert.Matches("^[0-9a-f]{64}$", file.Sha256);
                Assert.True(file.Bytes > 1_000, $"{lora.Id}/{file.FileName}: {file.Bytes} bytes");
            }
            Assert.InRange(lora.DefaultStrength, LoraCatalog.MinStrength, LoraCatalog.MaxStrength);
            if (lora.Kind == LoraKind.Speed)
            {
                Assert.True(lora.Steps is > 0, $"{lora.Id}: a speed plug-in says its step count");
                Assert.True(lora.RecipeConfig is not null ^ lora.ConfigFile is not null,
                    $"{lora.Id}: a speed plug-in's schedule comes from exactly one config");
                Assert.False(lora.StrengthAdjustable);
            }
            else
            {
                Assert.Null(lora.Steps);
                Assert.Null(lora.RecipeConfig);
                Assert.Null(lora.ConfigFile);
                Assert.True(lora.StrengthAdjustable);
            }
            if (lora.ConfigFile is { } config)
                Assert.Contains(lora.Files, f => f.FileName == config);
            if (lora.RecipeConfig is not null)
                Assert.Contains("\"sampling\"", LoraCatalog.RecipeText(lora));
        }
    }

    /// <summary>
    /// The app offers exactly the plug-ins the engine ships configs for, at the same files:
    /// the CLI and server read <c>config/lora/</c>, and a plug-in that answered differently in
    /// the app would be a second, unvalidated recipe. Both directions are checked.
    /// </summary>
    [Fact]
    public void EachPlugInIsTheEnginesOwnConfigAndEveryConfigIsOffered()
    {
        string[] configs = Directory.GetFiles(Path.Combine(RepoRoot(), "config", "lora"), "qwen-image-2.1-*.json")
            .Select(Path.GetFileNameWithoutExtension).OrderBy(id => id, StringComparer.Ordinal).ToArray()!;
        Assert.Equal(configs, LoraCatalog.BuiltIn.Select(l => l.Id).OrderBy(id => id, StringComparer.Ordinal));

        foreach (CatalogLora lora in LoraCatalog.BuiltIn)
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(EngineConfigPath(lora.Id)), Jsonc);
            JsonElement root = document.RootElement;
            Assert.Equal("qwen-image-2.1-lora", root.GetProperty("type").GetString());

            JsonElement weights = root.GetProperty("weights");
            Assert.Equal(lora.Weights.Url, weights.GetProperty("urls")[0].GetString());
            Assert.Equal(lora.Weights.Sha256, weights.GetProperty("sha256").GetString());
            Assert.EndsWith("/" + lora.Weights.FileName, weights.GetProperty("path").GetString());

            float scale = root.TryGetProperty("scale", out JsonElement s) ? s.GetSingle() : 1f;
            Assert.Equal(scale, lora.DefaultStrength);

            bool forwards = root.TryGetProperty("config", out JsonElement forwarded);
            bool samples = root.TryGetProperty("sampling", out JsonElement sampling);
            Assert.Equal(forwards || samples, lora.Kind == LoraKind.Speed);
            if (samples)
            {
                Assert.Equal(lora.Id + ".json", lora.RecipeConfig);
                Assert.Equal(sampling.GetProperty("steps").GetInt32(), lora.Steps);
                // What the engine is handed is the repository's file itself, byte for byte.
                Assert.Equal(File.ReadAllText(EngineConfigPath(lora.Id)), LoraCatalog.RecipeText(lora));
            }
            if (forwards)
            {
                LoraFile config = Assert.Single(lora.Files, f => f.FileName == lora.ConfigFile);
                Assert.Equal(config.Url, forwarded.GetProperty("urls")[0].GetString());
                Assert.Equal(config.Sha256, forwarded.GetProperty("sha256").GetString());
            }
        }
    }

    [Fact]
    public void ThePlugInsAreOfferedOnlyWhereTheirModelIs()
    {
        Assert.All(LoraCatalog.BuiltIn, l => Assert.Equal(LoraCatalog.QwenImage, l.BaseModelId));
        Assert.Equal(LoraCatalog.BuiltIn.Count, LoraCatalog.For(LoraCatalog.QwenImage).Count);
        Assert.Empty(LoraCatalog.For("gemma-4-e2b-q8"));
        // No phone or tablet reaches the image model's tier, so none is offered a plug-in.
        Assert.DoesNotContain(ModelCatalog.ForDevice(16), m => m.Id == LoraCatalog.QwenImage);
        Assert.All(LoraCatalog.QwenImageTurbo, id => Assert.DoesNotContain(ModelCatalog.ForDevice(16), m => m.Id == id));
    }

    /// <summary>
    /// Qwen-Image 2.1 Turbo is step-distilled already, and the engine refuses a plug-in that
    /// brings a second schedule: no speed plug-in applies to a Turbo entry. A plug-in reaches a
    /// Turbo entry only by naming it (<see cref="CatalogLora.AlsoFor"/>), which is done for the
    /// style plug-ins a real picture validated there: Film Stills and Grainscape.
    /// </summary>
    [Fact]
    public void ATurboEntryIsOfferedOnlyThePlugInsValidatedOnItAndNeverASpeedOne()
    {
        string[] turbos = ModelCatalog.BuiltIn
            .Where(m => m.Family == CatalogFamily.QwenImage && m.ImageVariant == QwenImageVariant.Turbo).Select(m => m.Id).ToArray();
        Assert.Equal(turbos, LoraCatalog.QwenImageTurbo);
        foreach (CatalogLora lora in LoraCatalog.BuiltIn)
        {
            foreach (string id in lora.AlsoFor)
                Assert.Contains(id, turbos);
            if (lora.Kind == LoraKind.Speed || lora.NeedsModelSteps)
                Assert.All(turbos, id => Assert.False(lora.AppliesTo(id), $"{lora.Id} must not apply to {id}"));
        }
        foreach (string id in turbos)
        {
            Assert.Equal(new[] { Film, Grain }, LoraCatalog.For(id).Select(l => l.Id));
            // What the LoRA sheet lists while that entry is loaded.
            Assert.Equal(new[] { Film, Grain }, LoraCatalog.Offered(ModelCatalog.Find(id), ModelCatalog.BuiltIn).Select(l => l.Id));
        }
        // The base model, a chat model or nothing loaded: every plug-in of an offered model.
        Assert.Equal(LoraCatalog.BuiltIn, LoraCatalog.Offered(ModelCatalog.Find(LoraCatalog.QwenImage), ModelCatalog.BuiltIn));
        Assert.Equal(LoraCatalog.BuiltIn, LoraCatalog.Offered(ModelCatalog.Find("gemma-4-e2b-q8"), ModelCatalog.BuiltIn));
        Assert.Equal(LoraCatalog.BuiltIn, LoraCatalog.Offered(null, ModelCatalog.BuiltIn));
        Assert.Empty(LoraCatalog.Offered(null, ModelCatalog.ForDevice(16)));
    }

    [Fact]
    public void ATurboPictureLeavesTheSpeedPlugInOutAndSaysWhy()
    {
        LoraStore store = StoreWith(Viggle, Film, Grain);
        string turbo = LoraCatalog.QwenImageTurbo[0];

        LoraPlan plan = LoraSelection.Plan(
            new[] { new ImageLoraChoice(Viggle, 1f), new ImageLoraChoice(Film, 0.7f) }, turbo, store, editing: false, out string? error)!;
        Assert.Null(error);
        Assert.Equal(new[] { "Film Stills" }, plan.Names);
        Assert.Equal(new float?[] { 0.7f }, plan.Specs.Select(s => s.Scale));
        Assert.Equal("Viggle Turbo sits out: Qwen-Image 2.1 Turbo is step-distilled already", plan.SatOut);

        // Alone, the speed plug-in leaves nothing to apply, and the plan still says why.
        plan = LoraSelection.Plan(new[] { new ImageLoraChoice(Viggle, 1f) }, turbo, store, editing: false, out error)!;
        Assert.Null(error);
        Assert.Empty(plan.Specs);
        Assert.Equal("Viggle Turbo sits out: Qwen-Image 2.1 Turbo is step-distilled already", plan.SatOut);

        // The base model takes the same choice whole.
        Assert.Equal(new[] { "Viggle Turbo", "Film Stills" },
            LoraSelection.Plan(new[] { new ImageLoraChoice(Viggle, 1f), new ImageLoraChoice(Film, 0.7f) },
                LoraCatalog.QwenImage, store, editing: false, out _)!.Names);
    }

    [Fact]
    public void ATurboPictureNamesEveryChosenPlugInNotValidatedThere()
    {
        // Turned on while the base model was loaded, then the Turbo entry: the sheet no longer
        // lists them, and the picture is made without them, so the plan says why for each.
        const string detail = "qwen-image-2.1-detail-enhancer";
        const string remover = "qwen-image-2.1-object-remover";
        LoraStore store = StoreWith(Viggle, Film, detail, remover);
        var choices = new[]
        {
            new ImageLoraChoice(detail, 0.8f), new ImageLoraChoice(Viggle, 1f),
            new ImageLoraChoice(Film, 0.7f), new ImageLoraChoice(remover, 1f),
        };

        LoraPlan plan = LoraSelection.Plan(choices, LoraCatalog.QwenImageTurbo[1], store, editing: true, out string? error)!;
        Assert.Null(error);
        Assert.Equal(new[] { "Film Stills" }, plan.Names);
        Assert.Equal("Detail Enhancer sits out: not validated on Qwen-Image 2.1 Turbo; " +
            "Viggle Turbo sits out: Qwen-Image 2.1 Turbo is step-distilled already; " +
            "Object Remover sits out: not validated on Qwen-Image 2.1 Turbo", plan.SatOut);

        // Nothing left to apply: still said.
        plan = LoraSelection.Plan(new[] { new ImageLoraChoice(detail, 0.8f) }, LoraCatalog.QwenImageTurbo[0], store, editing: false, out _)!;
        Assert.Empty(plan.Specs);
        Assert.Equal("Detail Enhancer sits out: not validated on Qwen-Image 2.1 Turbo", plan.SatOut);

        // A model that makes no pictures has no plug-ins to name.
        Assert.Same(LoraPlan.None, LoraSelection.Plan(choices, "gemma-4-e2b-q8", store, editing: true, out _));
    }

    // ---- the store ---------------------------------------------------------------------

    private sealed class Bytes : HttpMessageHandler
    {
        private readonly Dictionary<string, byte[]> _files;
        public int Requests;
        public Bytes(Dictionary<string, byte[]> files) => _files = files;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref Requests);
            return Task.FromResult(_files.TryGetValue(request.RequestUri!.AbsoluteUri, out byte[]? body)
                ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) }
                : new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }

    private static LoraFile FileOf(string name, byte[] body, string repo = "test/lora") => new(
        name, $"https://huggingface.co/{repo}/resolve/{new string('a', 40)}/{name}", body.Length,
        Convert.ToHexStringLower(SHA256.HashData(body)));

    private static byte[] Body(int length, byte seed) => Enumerable.Range(0, length).Select(i => (byte)(i * 31 + seed)).ToArray();

    [Fact]
    public async Task ADownloadVerifiesEveryFileAndASpeedPlugInIsHandedItsSchedule()
    {
        byte[] weights = Body(4096, 1);
        CatalogLora viggle = LoraCatalog.Find("qwen-image-2.1-viggle-turbo")! with
        {
            Files = new[] { FileOf("viggle.safetensors", weights) },
        };
        var handler = new Bytes(new() { [viggle.Weights.Url] = weights });
        var store = new LoraStore(_root, new ResumableDownloader(new HttpClient(handler)));
        Assert.Equal(InstallState.NotInstalled, store.StateOf(viggle));

        await store.DownloadAsync(viggle, null, CancellationToken.None);

        Assert.Equal(InstallState.Installed, store.StateOf(viggle));
        LoraSpec spec = store.SpecFor(viggle, strength: 0.3f);
        Assert.True(Path.IsPathFullyQualified(spec.Path));
        Assert.Equal(store.PathFor(viggle, viggle.Weights), spec.Path);
        // A speed plug-in is used at the strength it was trained at, whatever was asked.
        Assert.Equal(viggle.DefaultStrength, spec.Scale);
        // Its schedule is the engine's own config, written beside the weights.
        Assert.NotNull(spec.ConfigPath);
        Assert.Equal(Path.GetDirectoryName(spec.Path), Path.GetDirectoryName(spec.ConfigPath));
        Assert.Equal(LoraCatalog.RecipeText(viggle), File.ReadAllText(spec.ConfigPath!));
        // And asking again is the same spec, which is how the engine sees no change.
        Assert.Equal(spec, store.SpecFor(viggle, 0.3f));

        // A second download of an installed plug-in fetches nothing.
        int requests = handler.Requests;
        await store.DownloadAsync(viggle, null, CancellationToken.None);
        Assert.Equal(requests, handler.Requests);

        store.Delete(viggle);
        Assert.Equal(InstallState.NotInstalled, store.StateOf(viggle));
        Assert.False(Directory.Exists(store.DirectoryFor(viggle)));
    }

    [Fact]
    public async Task AFileThatIsNotThePinnedOneIsNeverInstalled()
    {
        byte[] weights = Body(2048, 2);
        CatalogLora film = LoraCatalog.Find("qwen-image-2.1-film-stills")! with
        {
            Files = new[] { FileOf("film.safetensors", weights) with { Sha256 = new string('0', 64) } },
        };
        var store = new LoraStore(_root, new ResumableDownloader(new HttpClient(new Bytes(new() { [FileOf("film.safetensors", weights).Url] = weights })), maxAttempts: 1));

        await Assert.ThrowsAnyAsync<Exception>(() => store.DownloadAsync(film, null, CancellationToken.None));

        Assert.NotEqual(InstallState.Installed, store.StateOf(film));
        Assert.False(File.Exists(store.PathFor(film, film.Weights)));
    }

    [Fact]
    public async Task ABundleIsHandedTheConfigThatShipsWithIt()
    {
        byte[] weights = Body(3000, 3), pdd = Body(1500, 4);
        CatalogLora funAcc = LoraCatalog.Find("qwen-image-2.1-fun-acc-4step")! with
        {
            Files = new[] { FileOf("bundle.safetensors", weights), FileOf("pdd_config.json", pdd) },
        };
        var store = new LoraStore(_root, new ResumableDownloader(new HttpClient(new Bytes(new()
        {
            [funAcc.Files[0].Url] = weights,
            [funAcc.Files[1].Url] = pdd,
        }))));

        await store.DownloadAsync(funAcc, null, CancellationToken.None);

        LoraSpec spec = store.SpecFor(funAcc, 1f);
        Assert.Equal(Path.Combine(store.DirectoryFor(funAcc), "pdd_config.json"), spec.ConfigPath);
        Assert.Equal(pdd, File.ReadAllBytes(spec.ConfigPath!));
    }

    [Fact]
    public void AStylePlugInTakesTheStrengthChosenAndNeedsNoConfig()
    {
        var store = new LoraStore(_root);
        CatalogLora film = LoraCatalog.Find("qwen-image-2.1-film-stills")!;
        LoraSpec spec = store.SpecFor(film, 0.45f);
        Assert.Equal(0.45f, spec.Scale);
        Assert.Null(spec.ConfigPath);
    }

    // ---- the rules a choice obeys --------------------------------------------------------

    /// <summary>Puts empty-but-correctly-sized files in place, so the store reads the plug-ins as installed.</summary>
    private LoraStore StoreWith(params string[] installed)
    {
        var store = new LoraStore(_root);
        foreach (string id in installed)
        {
            CatalogLora lora = LoraCatalog.Find(id)!;
            Directory.CreateDirectory(store.DirectoryFor(lora));
            foreach (LoraFile file in lora.Files)
            {
                SparseFileFixture.Create(store.PathFor(lora, file), file.Bytes);
            }
        }
        return store;
    }

    private const string Viggle = "qwen-image-2.1-viggle-turbo";
    private const string Pruna = "qwen-image-2.1-pruna-8step";
    private const string Film = "qwen-image-2.1-film-stills";
    private const string Grain = "qwen-image-2.1-grainscape";

    [Fact]
    public void AChoiceIsCheckedAndNormalisedBeforeItIsSaved()
    {
        LoraStore store = StoreWith(Viggle, Pruna, Film);
        var none = Array.Empty<ImageLoraChoice>();

        List<ImageLoraChoice>? saved = LoraSelection.Validate(new ImageLoraChoice[]
        {
            new(Film, 9f), new(Viggle, 0.2f), new(Film, 0.5f),
        }, none, store, out string? error);
        Assert.Null(error);
        Assert.Equal(new[] { new ImageLoraChoice(Film, LoraCatalog.MaxStrength), new ImageLoraChoice(Viggle, 1f) }, saved);

        Assert.Null(LoraSelection.Validate(new ImageLoraChoice[] { new(Viggle, 1f), new(Pruna, 1f) }, none, store, out error));
        Assert.Contains("both set the number of steps", error);

        Assert.Null(LoraSelection.Validate(new ImageLoraChoice[] { new(Grain, 0.7f) }, none, store, out error));
        Assert.Contains("not downloaded", error);

        Assert.Null(LoraSelection.Validate(new ImageLoraChoice[] { new("no-such-plug-in", 1f) }, none, store, out error));
        Assert.Contains("no LoRA plug-in", error);
    }

    [Fact]
    public void AChoiceKeepsWhatANewerBuildChoseAndThisOneCannotShow()
    {
        LoraStore store = StoreWith(Film);
        var saved = new[] { new ImageLoraChoice("a-plug-in-from-a-newer-build", 0.8f), new ImageLoraChoice(Viggle, 1f) };

        List<ImageLoraChoice>? chosen = LoraSelection.Validate(new[] { new ImageLoraChoice(Film, 0.7f) }, saved, store, out _);

        Assert.Equal(new[] { new ImageLoraChoice(Film, 0.7f), new ImageLoraChoice("a-plug-in-from-a-newer-build", 0.8f) }, chosen);
    }

    /// <summary>
    /// The page sends the whole choice with every change. What it repeats from the saved choice
    /// is kept even when this build cannot check it (a newer build's id, a plug-in whose files
    /// have gone); only what the change adds must be known and downloaded.
    /// </summary>
    [Fact]
    public void AChangeIsCheckedOnlyForWhatItAdds()
    {
        LoraStore store = StoreWith(Film);
        const string newer = "a-plug-in-from-a-newer-build";
        var saved = new[] { new ImageLoraChoice(newer, 0.8f), new ImageLoraChoice(Viggle, 1f) };

        List<ImageLoraChoice>? chosen = LoraSelection.Validate(
            new[] { new ImageLoraChoice(newer, 0.8f), new ImageLoraChoice(Viggle, 1f), new ImageLoraChoice(Film, 0.7f) },
            saved, store, out string? error);
        Assert.Null(error);
        Assert.Equal(new[] { new ImageLoraChoice(Viggle, 1f), new ImageLoraChoice(Film, 0.7f), new ImageLoraChoice(newer, 0.8f) }, chosen);

        Assert.Null(LoraSelection.Validate(new[] { new ImageLoraChoice(Viggle, 1f), new ImageLoraChoice(Grain, 0.7f) }, saved, store, out error));
        Assert.Contains("Grainscape is not downloaded", error);
        Assert.Null(LoraSelection.Validate(new[] { new ImageLoraChoice("another-unknown-id", 1f) }, saved, store, out error));
        Assert.Contains("no LoRA plug-in called 'another-unknown-id'", error);
    }

    [Fact]
    public void APictureAppliesTheSpeedPlugInFirstAndNamesWhatItApplies()
    {
        LoraStore store = StoreWith(Viggle, Film, Grain);
        var choices = new[]
        {
            new ImageLoraChoice(Grain, 0.6f), new ImageLoraChoice("unknown-to-this-build", 1f),
            new ImageLoraChoice(Viggle, 0.1f), new ImageLoraChoice(Film, 0.7f),
        };

        LoraPlan? plan = LoraSelection.Plan(choices, LoraCatalog.QwenImage, store, editing: true, out string? error);

        Assert.Null(error);
        Assert.Equal(new[] { "Viggle Turbo", "Grainscape", "Film Stills" }, plan!.Names);
        Assert.Equal(new float?[] { 1f, 0.6f, 0.7f }, plan.Specs.Select(s => s.Scale));
        Assert.All(plan.Specs, s => Assert.True(Path.IsPathFullyQualified(s.Path)));

        // Another model's picture applies none of them.
        Assert.Same(LoraPlan.None, LoraSelection.Plan(choices, "gemma-4-e2b-q8", store, editing: true, out _));
    }

    [Fact]
    public void AnEditOnlyPlugInAppliesToEditsAndNotToPicturesMadeFromWords()
    {
        const string exposure = "qwen-image-2.1-natural-exposure";
        LoraStore store = StoreWith(Viggle, exposure);
        var choices = new[] { new ImageLoraChoice(exposure, 1f), new ImageLoraChoice(Viggle, 1f) };

        Assert.Equal(new[] { "Viggle Turbo", "Natural Exposure" },
            LoraSelection.Plan(choices, LoraCatalog.QwenImage, store, editing: true, out _)!.Names);
        Assert.Equal(new[] { "Viggle Turbo" },
            LoraSelection.Plan(choices, LoraCatalog.QwenImage, store, editing: false, out _)!.Names);
    }

    /// <summary>
    /// Object Remover works only at the model's own 40 steps. An edit it applies to is made
    /// without the speed plug-in, and the plan says why; a picture made from words, which it
    /// does not apply to, keeps the speed plug-in.
    /// </summary>
    [Fact]
    public void ASpeedPlugInSitsOutAnEditThatNeedsTheModelsOwnSteps()
    {
        const string remover = "qwen-image-2.1-object-remover";
        LoraStore store = StoreWith(Viggle, remover, Film);
        var choices = new[] { new ImageLoraChoice(Viggle, 1f), new ImageLoraChoice(remover, 1f), new ImageLoraChoice(Film, 0.7f) };

        LoraPlan edit = LoraSelection.Plan(choices, LoraCatalog.QwenImage, store, editing: true, out string? error)!;
        Assert.Null(error);
        Assert.Equal(new[] { "Object Remover", "Film Stills" }, edit.Names);
        Assert.Equal("Viggle Turbo sits out: Object Remover works only at the model's own steps", edit.SatOut);
        Assert.All(edit.Specs, s => Assert.Null(s.ConfigPath));

        LoraPlan drawing = LoraSelection.Plan(choices, LoraCatalog.QwenImage, store, editing: false, out _)!;
        Assert.Equal(new[] { "Viggle Turbo", "Film Stills" }, drawing.Names);
        Assert.Null(drawing.SatOut);
        Assert.Equal(remover, Assert.Single(LoraCatalog.BuiltIn, l => l.NeedsModelSteps).Id);
    }

    /// <summary>
    /// Pictures in two chats can be prepared at once, and each writes the speed plug-in's
    /// schedule beside its weights: neither may move away the file the other is writing.
    /// </summary>
    [Fact]
    public async Task PicturesPreparedAtOnceAllGetTheSchedule()
    {
        LoraStore store = StoreWith(Viggle);
        CatalogLora viggle = LoraCatalog.Find(Viggle)!;
        using var start = new ManualResetEventSlim();
        Task<LoraSpec>[] prepared = Enumerable.Range(0, 16).Select(i => Task.Run(() =>
        {
            start.Wait();
            // Separate store instances can share the same on-device plug-in directory.
            return (i % 2 == 0 ? store : new LoraStore(store.Root)).SpecFor(viggle, 1f);
        })).ToArray();
        start.Set();
        LoraSpec[] specs = await Task.WhenAll(prepared);

        Assert.Single(specs.Select(s => s.ConfigPath).Distinct());
        Assert.Equal(LoraCatalog.RecipeText(viggle), File.ReadAllText(specs[0].ConfigPath!));
        Assert.Empty(Directory.EnumerateFiles(store.DirectoryFor(viggle), "*.tmp"));
    }

    [Fact]
    public void APlugInTurnedOnButGoneRefusesThePictureInsteadOfSkippingIt()
    {
        LoraStore store = StoreWith(Film);
        Assert.Null(LoraSelection.Plan(new[] { new ImageLoraChoice(Viggle, 1f) }, LoraCatalog.QwenImage, store, editing: false, out string? error));
        Assert.Contains("files are missing", error);

        store = StoreWith(Viggle, Pruna);
        Assert.Null(LoraSelection.Plan(new[] { new ImageLoraChoice(Viggle, 1f), new ImageLoraChoice(Pruna, 1f) },
            LoraCatalog.QwenImage, store, editing: false, out error));
        Assert.Contains("both set the number of steps", error);
    }

    [Fact]
    public void TheChoiceIsSavedWithTheSettingsAndAnOldFileReadsAsNone()
    {
        string path = Path.Combine(_root, "settings.json");
        var settings = new SettingsStore(path);
        Assert.Empty(settings.Load().ImageLoras);

        AppSettings chosen = settings.Load();
        chosen.ImageLoras = new() { new ImageLoraChoice(Viggle, 1f), new ImageLoraChoice(Film, 0.7f) };
        settings.Save(chosen);
        Assert.Contains("\"imageLoras\"", File.ReadAllText(path));
        Assert.Equal(chosen.ImageLoras, settings.Load().ImageLoras);

        // A clone is a copy: changing it leaves the original as it was.
        AppSettings clone = chosen.Clone();
        clone.ImageLoras.Clear();
        Assert.Equal(2, chosen.ImageLoras.Count);

        File.WriteAllText(path, "{\"selectedModelId\":\"gemma-4-e2b-q8\"}");
        Assert.Empty(settings.Load().ImageLoras);
    }

    /// <summary>A hand-edited file's null choice, or null or nameless entries in it, read as nothing chosen.</summary>
    [Fact]
    public void AHandEditedChoiceWithNullsReadsAsWhatItCanSay()
    {
        string path = Path.Combine(_root, "settings.json");
        Directory.CreateDirectory(_root);
        var settings = new SettingsStore(path);

        File.WriteAllText(path, "{\"imageLoras\": null}");
        Assert.Empty(settings.Load().ImageLoras);
        Assert.Empty(settings.Load().Clone().ImageLoras);

        File.WriteAllText(path, "{\"imageLoras\": [null, {\"strength\": 1}, {\"id\": \"" + Film + "\", \"strength\": 0.5}]}");
        Assert.Equal(new[] { new ImageLoraChoice(Film, 0.5f) }, settings.Load().ImageLoras);
    }

    /// <summary>
    /// The page's routes run at once. Each change is a load, a change and a save, and two of
    /// them interleaved used to lose one; as one step under the store's lock, none is lost.
    /// A change that returns null leaves the file alone.
    /// </summary>
    [Fact]
    public async Task ChangesMadeAtOnceAreAllKept()
    {
        var settings = new SettingsStore(Path.Combine(_root, "settings.json"));
        Assert.Equal(settings.Load().MaxTokens, settings.Update(_ => null).MaxTokens);
        Assert.False(File.Exists(settings.Path));

        await Task.WhenAll(Enumerable.Range(0, 32).Select(i => Task.Run(() => settings.Update(s =>
        {
            s.DefaultSkills.Add("skill-" + i);
            return s;
        }))));

        Assert.Equal(32, settings.Load().DefaultSkills.Distinct().Count());
    }
}
