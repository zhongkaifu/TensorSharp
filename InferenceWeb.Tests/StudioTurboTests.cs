// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Chat.Studio;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Studio;
using Adapter = InferenceWeb.Tests.MiniMaxH3LoraFixtures.Adapter;

namespace InferenceWeb.Tests;

/// <summary>
/// Video Studio under a MiniMax-H3 sampling recipe (the LightX2V Turbo plug-in):
/// <list type="bullet">
/// <item>the capability advertises the recipe and fixes its steps, base MiniMax-H3 keeps its 20, and the 3-argument
/// <c>CreateCapability</c> other callers use still reads the model's own schedule;</item>
/// <item>a host whose <c>--video-steps</c> contradicts the recipe, or a recipe on the Ref2VA checkpoint, leaves the
/// model unavailable rather than offering a run that is refused;</item>
/// <item>the host's per-clip plug-in choice is read without side effects for capabilities and validation and taken
/// once for the run, whose bound recipe the chat service checks after applying it;</item>
/// <item>validation refuses a shot planned for another recipe (409) or another step count (by name);</item>
/// <item>provenance names each adapter file, its strength and SHA-256, and the exact sigma grid.</item>
/// </list>
/// Capabilities use metadata-only model objects (StudioGenerationParityTests' pattern) with the shipped plug-in
/// configs; the adapter runs against the tiny fixture transformer and adapters (MiniMaxH3LoraFixtures). No clip is
/// generated: that needs the real model, so provenance is checked through the helpers GenerateAsync assembles it with.
/// </summary>
public sealed class StudioTurboTests : IDisposable
{
    private const string Name8 = "LightX2V Turbo 8-step (FL2V v1.0, 768p)", Name4 = "LightX2V Turbo 4-step (FL2V v1.2, 768p)";
    private const string W8 = MiniMaxH3LoraFixtures.Weights8, W4 = MiniMaxH3LoraFixtures.Weights4;
    private const string Fixture8 = "minimax-h3-lora:turbo-8step", Fixture4 = "minimax-h3-lora:turbo-4step";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-studio-turbo-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new();
    private readonly MiniMaxH3LoraFixtures _fx = new();
    private readonly List<bool> _providerCalls = new();
    private ModelService _models;
    private WebUiChatService _chat;
    private MiniMaxH3Model _model;

    public StudioTurboTests()
    {
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
        // The model and the hosts read these; a developer's shell must not leak into the fixture.
        foreach (string name in new[]
                 {
                     LoraCliFlags.EnvironmentVariable, QwenImageVariantFlag.EnvironmentVariable, "TS_VIDEO_TEXT_ENCODER", "TS_VIDEO_VAE",
                     "TS_VIDEO_AUDIO_VAE", "TS_H3_DIT_LAYERS", "TENSORSHARP_MODELS",
                 })
            _env.Set(name, null);
    }

    public void Dispose()
    {
        _models?.Dispose();
        _fx.Dispose();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ---- metadata-only capabilities ----------------------------------------------------------

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.True(dir != null, "The tests run from a source checkout; config/lora/ must be reachable.");
        return dir!.FullName;
    }

    private static string ShippedConfig(int steps) => Path.Combine(FindRepoRoot(), "config", "lora", $"minimax-h3-fl2v-turbo-{steps}step.json");
    private static MiniMaxH3SamplingRecipe Shipped(int steps) => MiniMaxH3LoraConfig.Load(ShippedConfig(steps)).Recipe;

    private static T WithProperty<T>(T model, string property, object value)
    {
        typeof(T).GetField($"<{property}>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, value);
        return model;
    }

    /// <summary>A MiniMax-H3 model object with no weights, its plug-in's recipe injected as a loaded set would leave it.</summary>
    private static MiniMaxH3Model H3(MiniMaxH3Partition partition, MiniMaxH3SamplingRecipe recipe = null)
    {
        var model = WithProperty((MiniMaxH3Model)RuntimeHelpers.GetUninitializedObject(typeof(MiniMaxH3Model)), nameof(MiniMaxH3Model.Partition), partition);
        return recipe == null ? model : WithProperty(model, nameof(MiniMaxH3Model.SamplingRecipe), recipe);
    }

    private static StudioModelCapability Capability(IVideoGenerationModel model, VideoGenerationParams defaults = null) =>
        NativeStudioGenerationAdapter.CreateCapability(model, "h3", defaults ?? new());

    [Fact]
    public void TheThreeArgumentCapabilityKeepsEachModelsOwnSchedule()
    {
        Assert.Equal(VideoSamplingContract.NativeId, StudioModelCapability.NativeSamplingRecipe);
        var h3 = Capability(H3(MiniMaxH3Partition.FirstLastFrame));
        Assert.Equal((MiniMaxH3Scheduler.DefaultSteps, (int?)null), (h3.DefaultSteps, h3.FixedSteps));
        var ltx = NativeStudioGenerationAdapter.CreateCapability((Ltx25Model)RuntimeHelpers.GetUninitializedObject(typeof(Ltx25Model)), "ltx", new());
        Assert.Equal((8, (int?)8), (ltx.DefaultSteps, ltx.FixedSteps)); // LTX's own fixed schedule is not a recipe
        var wan = NativeStudioGenerationAdapter.CreateCapability(WithProperty(WithProperty(
            (WanVideoModel)RuntimeHelpers.GetUninitializedObject(typeof(WanVideoModel)), nameof(WanVideoModel.Variant), WanVariant.T2V),
            nameof(WanVideoModel.InputChannels), 16), "wan", new());
        foreach (var capability in new[] { h3, ltx, wan })
        {
            Assert.True(capability.Available, capability.UnavailableReason);
            Assert.Equal("native", capability.SamplingRecipe);
            Assert.Null(capability.SamplingRecipeName);
            Assert.Equal(capability.ModelId, capability.DisplayName);
        }
    }

    [Theory]
    [InlineData(8, Name8)]
    [InlineData(4, Name4)]
    public void ATurboRecipeFixesItsStepsAndNamesItself(int steps, string name)
    {
        var recipe = Shipped(steps);
        var turbo = Capability(H3(MiniMaxH3Partition.FirstLastFrame, recipe));
        Assert.True(turbo.Available, turbo.UnavailableReason);
        Assert.Equal((steps, (int?)steps), (turbo.DefaultSteps, turbo.FixedSteps));
        Assert.Equal((1f, (float?)1f), (turbo.DefaultCfgScale, turbo.FixedCfgScale));
        Assert.Equal($"minimax-h3-lora:minimax-h3-fl2v-turbo-{steps}step", turbo.SamplingRecipe);
        Assert.Equal(name, turbo.SamplingRecipeName);
        Assert.Equal("h3 · " + name, turbo.DisplayName);

        // The explicit-contract overload is the same capability; a host that repeats the recipe's count still offers it.
        var explicitContract = NativeStudioGenerationAdapter.CreateCapability(H3(MiniMaxH3Partition.FirstLastFrame), "h3", new(), recipe.ToContract());
        Assert.Equal(JsonSerializer.Serialize(turbo, StudioJson.Options), JsonSerializer.Serialize(explicitContract, StudioJson.Options));
        Assert.True(Capability(H3(MiniMaxH3Partition.FirstLastFrame, recipe), new() { Steps = steps }).Available);

        // Everything but the sampling is the base FL2VA checkpoint's.
        var basis = Capability(H3(MiniMaxH3Partition.FirstLastFrame));
        Assert.Equal(basis.Modes, turbo.Modes);
        Assert.Equal(basis.ImageCanvasModes, turbo.ImageCanvasModes);
        Assert.Equal((basis.DefaultWidth, basis.DefaultHeight, basis.FixedFps, basis.DefaultFrameCount),
            (turbo.DefaultWidth, turbo.DefaultHeight, turbo.FixedFps, turbo.DefaultFrameCount));
    }

    [Theory]
    [InlineData(8, 20)]
    [InlineData(8, 4)]
    [InlineData(4, 8)]
    public void AHostStepDefaultThatContradictsTheRecipeLeavesTheModelUnavailable(int steps, int hostSteps)
    {
        // Through the server's own flag (the startup check refuses this first; Studio is the backstop).
        var defaults = VideoGenerationParamsParser.CreateDefaults(ServerOptionsBuilder.Build(["--video-steps", hostSteps.ToString()], _dir));
        Assert.Equal(hostSteps, defaults.Steps);
        var turbo = Capability(H3(MiniMaxH3Partition.FirstLastFrame, Shipped(steps)), defaults);
        string name = steps == 8 ? Name8 : Name4;
        Assert.False(turbo.Available);
        Assert.Equal($"--video-steps {hostSteps} contradicts {name}, which samples exactly {steps} steps. " +
                     $"Start the host without --video-steps or with --video-steps {steps}.", turbo.UnavailableReason);
        Assert.Equal(steps, turbo.DefaultSteps);

        // The base checkpoint takes the host's count as its default, as before.
        var basis = Capability(H3(MiniMaxH3Partition.FirstLastFrame), defaults);
        Assert.True(basis.Available);
        Assert.Equal((hostSteps, (int?)null), (basis.DefaultSteps, basis.FixedSteps));
    }

    [Fact]
    public void ARecipeOnTheRef2vaCheckpointIsUnavailable()
    {
        var capability = Capability(H3(MiniMaxH3Partition.Reference, Shipped(8)));
        Assert.False(capability.Available);
        Assert.Equal(Name8 + " (minimax-h3-fl2v-turbo-8step.json) is a MiniMax-H3 FL2VA plug-in; the loaded Ref2VA checkpoint does not take it, " +
                     "and Ref2VA Turbo is not supported. Load minimax_h3_fl2va_pruned-*.gguf or drop the plug-in.", capability.UnavailableReason);
    }

    [Fact]
    public void AModelThatTakesNoPlugInsRefusesAChoiceOfThem()
    {
        IVideoGenerationModel ltx = (Ltx25Model)RuntimeHelpers.GetUninitializedObject(typeof(Ltx25Model));
        var error = Assert.Throws<InvalidOperationException>(() => NativeStudioGenerationAdapter.PlannedContract(ltx, [new LoraSpec(ShippedConfig(8))]));
        Assert.Equal($"{ltx.VideoModelFamily} takes no LoRA plug-ins.", error.Message);
        Assert.Null(NativeStudioGenerationAdapter.PlannedContract(ltx, []));
        Assert.Null(NativeStudioGenerationAdapter.PlannedContract(ltx, null));
        Assert.Equal(409, StudioHostBoundary.Status(new StudioException("sampling_recipe_conflict", "")));
    }

    // ---- provenance --------------------------------------------------------------------------

    private static string Grid(float[] sigmas) => string.Join(", ", sigmas.Select(s => s.ToString("0.#####", System.Globalization.CultureInfo.InvariantCulture)));

    [Fact]
    public void ProvenanceNamesTheGridTheClipSampled()
    {
        var basis = NativeStudioGenerationAdapter.MiniMaxH3Sampling(null, 20);
        Assert.Equal("MiniMax-H3: 20 steps, sd.cpp discrete grid, video shift 12 / audio 3, euler, cfg 1", basis["recipe"]);
        Assert.Equal(("12", "3"), (basis["videoShift"], basis["audioShift"]));
        Assert.Equal(Grid(MiniMaxH3Scheduler.BuildSigmas(20, 12f)), basis["sigmaGrid"]);
        Assert.False(basis.ContainsKey("samplingRecipeName"));

        var turbo8 = NativeStudioGenerationAdapter.MiniMaxH3Sampling(Shipped(8), 8);
        Assert.Equal(Name8 + ": 8 steps, endpoint-inclusive linspace, video shift 6 / audio 3, euler, cfg 1", turbo8["recipe"]);
        Assert.Equal(("6", "3", Name8), (turbo8["videoShift"], turbo8["audioShift"], turbo8["samplingRecipeName"]));
        // The design's pinned grids (MiniMaxH3Scheduler.BuildUniformSigmas at shift 6).
        Assert.Equal("1, 0.97674, 0.94737, 0.90909, 0.85714, 0.78261, 0.66667, 0.46154, 0", turbo8["sigmaGrid"]);
        Assert.Equal("1, 0.94737, 0.85714, 0.66667, 0", NativeStudioGenerationAdapter.MiniMaxH3Sampling(Shipped(4), 4)["sigmaGrid"]);
    }

    private sealed class RecordingProgress : IProgress<StudioProgress>
    {
        public readonly List<StudioProgress> Reports = new();
        public void Report(StudioProgress value) { lock (Reports) Reports.Add(value); }
    }

    [Fact]
    public async Task ProvenanceNamesEachAdapterFileItsStrengthAndHash()
    {
        string weights = _fx.WriteAdapter(new Adapter().AllTargets(blocks: 2, refinerBlocks: 1, rank: 4), W8);
        string config = _fx.WriteManifest(scale: 0.5f);
        string bare = _fx.WriteText("other.safetensors", "not an adapter");
        // The forms a spec takes: the config alone (--lora), the weights with their config (the app's and the hosts'
        // resolved form), bare weights. A spec's own strength wins over the config's "scale".
        Assert.Equal((weights, 0.5f), NativeStudioGenerationAdapter.PluginWeights(new LoraSpec(config)));
        Assert.Equal((weights, 0.75f), NativeStudioGenerationAdapter.PluginWeights(new LoraSpec(config, 0.75f)));
        Assert.Equal((weights, 0.5f), NativeStudioGenerationAdapter.PluginWeights(new LoraSpec(weights, null, config)));
        Assert.Equal((bare, 1f), NativeStudioGenerationAdapter.PluginWeights(new LoraSpec(bare)));
        // A shipped config names its adapter in the models folder without downloading it.
        var shipped = NativeStudioGenerationAdapter.PluginWeights(new LoraSpec(ShippedConfig(8)));
        Assert.Equal((W8, 1f), (Path.GetFileName(shipped.Weights), shipped.Strength));

        // The hash is the file's, computed once per file version with a progress line the first time.
        var progress = new RecordingProgress();
        string hash = await NativeStudioGenerationAdapter.Sha256Async(new FileInfo(weights), progress, 0, "hashing", CancellationToken.None);
        Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(weights))), hash);
        Assert.Equal("model-identity", Assert.Single(progress.Reports).Stage);
        Assert.Equal(hash, await NativeStudioGenerationAdapter.Sha256Async(new FileInfo(weights), progress, 0, "hashing", CancellationToken.None));
        Assert.Single(progress.Reports);

        Assert.Equal($"; lora={W8}@0.5; lora_sha256={hash}; recipe={Fixture8}",
            NativeStudioGenerationAdapter.PluginIdentity([(weights, 0.5f)], [hash], Fixture8));
        Assert.Equal("; recipe=native", NativeStudioGenerationAdapter.PluginIdentity([], [], VideoSamplingContract.NativeId));
    }

    // ---- the adapter on a loaded model ---------------------------------------------------------

    /// <summary>Load the tiny FL2VA transformer through the host's model service, as a real host would.</summary>
    private MiniMaxH3Model LoadFixtureModel()
    {
        string path = _fx.WriteTransformer();
        // The companions' preferred names, so the constructor never scans the temp folder.
        foreach (string name in new[] { "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors" })
            _fx.WriteText(name, "");
        _models = new ModelService(NullLogger<ModelService>.Instance, (p, _, _, _) => _model = new MiniMaxH3Model(p, BackendType.Cpu));
        _chat = new WebUiChatService(_models, new SessionManager(), WebUiVideoLoraTests.Options(_dir),
            new UploadStoragePolicy(Path.Combine(_dir, "uploads")),
            new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() }),
            codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);
        _models.LoadModel(path, null, "cpu");
        Assert.Same(_model, _models.Model);
        return _model;
    }

    /// <summary>A tiny adapter and its plug-in config: (config, weights).</summary>
    private (string Config, string Weights) Plugin(int steps)
    {
        string weights = _fx.WriteAdapter(new Adapter().AllTargets(blocks: 2, refinerBlocks: 1, rank: 4, seed: steps), steps == 8 ? W8 : W4);
        return (_fx.WriteManifest($"turbo-{steps}step.json", Path.GetFileName(weights), steps: steps, name: $"Test Turbo {steps}"), weights);
    }

    /// <summary>The adapter with a host that chooses plug-ins per clip, recording whether each call was for the clip.</summary>
    private NativeStudioGenerationAdapter Studio(Func<bool, IReadOnlyList<LoraSpec>> choice) =>
        new(_models, _chat, videoLoras: forClip => { _providerCalls.Add(forClip); return choice(forClip); });

    private StudioGenerationRequest Shot(int steps = 0, string recipe = null) =>
        new() { ModelId = _models.LoadedModelName, Prompt = "A kite rises over the dunes.", Steps = steps, SamplingRecipe = recipe };

    private static readonly IReadOnlyDictionary<string, string> NoAssets = new Dictionary<string, string>();

    [Fact]
    public async Task TheCapabilityFollowsTheHostsChoiceWithoutApplyingIt()
    {
        var model = LoadFixtureModel();
        var (config8, weights8) = Plugin(8);
        var (config4, _) = Plugin(4);
        IReadOnlyList<LoraSpec> chosen = [new LoraSpec(weights8, null, config8)]; // the app's form: weights with their config
        var studio = Studio(_ => chosen);

        var turbo = Assert.Single(await studio.GetCapabilitiesAsync(CancellationToken.None));
        Assert.True(turbo.Available, turbo.UnavailableReason);
        Assert.Equal((Fixture8, "Test Turbo 8", 8, (int?)8), (turbo.SamplingRecipe, turbo.SamplingRecipeName, turbo.DefaultSteps, turbo.FixedSteps));
        Assert.Equal(_models.LoadedModelName + " · Test Turbo 8", turbo.DisplayName);
        chosen = [new LoraSpec(config4)]; // the CLI form: the config alone
        var four = Assert.Single(await studio.GetCapabilitiesAsync(CancellationToken.None));
        Assert.Equal((Fixture4, (int?)4), (four.SamplingRecipe, four.FixedSteps));
        chosen = []; // the plug-in was removed
        var native = Assert.Single(await studio.GetCapabilitiesAsync(CancellationToken.None));
        Assert.Equal(("native", (string)null, MiniMaxH3Scheduler.DefaultSteps, (int?)null), (native.SamplingRecipe, native.SamplingRecipeName, native.DefaultSteps, native.FixedSteps));

        // Capabilities only plan: every call was a query, and nothing reached the model.
        Assert.Equal(new[] { false, false, false }, _providerCalls);
        Assert.Empty(model.LoraSpecs);
        Assert.Null(model.SamplingRecipe);
    }

    [Fact]
    public async Task WithoutAProviderTheCapabilityIsTheModelsOwnPlugIn()
    {
        // The server: --lora is applied at load (TS_LORAS), and Studio reads the recipe the model holds.
        var (config8, _) = Plugin(8);
        _env.Set(LoraCliFlags.EnvironmentVariable, LoraCliFlags.ToJson([new LoraSpec(config8)]));
        var model = LoadFixtureModel();
        Assert.Equal(8, model.SamplingRecipe.Steps);
        var capability = Assert.Single(await new NativeStudioGenerationAdapter(_models, _chat).GetCapabilitiesAsync(CancellationToken.None));
        Assert.Equal((Fixture8, (int?)8), (capability.SamplingRecipe, capability.FixedSteps));
    }

    [Fact]
    public async Task AChoiceTheModelRefusesLeavesItUnavailableWithTheModelsText()
    {
        LoadFixtureModel();
        var (_, weights8) = Plugin(8);
        string qwen = _fx.WriteText("qwen-plugin.json", "{ \"type\": \"qwen-image-2.1-lora\", \"name\": \"q\", \"weights\": \"q.safetensors\" }");
        foreach (var (choice, reason) in new (IReadOnlyList<LoraSpec>, string)[]
                 {
                     ([new LoraSpec(weights8)], MiniMaxH3LoraConfig.BareWeightsRefusal(W8)),
                     ([new LoraSpec(qwen)], MiniMaxH3LoraConfig.ForeignTypeRefusal("qwen-plugin.json", LoraCliFlags.QwenImagePluginType,
                         _models.LoadedModelName, MiniMaxH3Model.ArchitectureId, LoraCliFlags.MiniMaxH3PluginType)),
                 })
        {
            var capability = Assert.Single(await Studio(_ => choice).GetCapabilitiesAsync(CancellationToken.None));
            Assert.False(capability.Available);
            Assert.Equal(reason, capability.UnavailableReason);
        }
        // A host that cannot plan the clip's plug-ins (the app's still downloading) is reported the same way.
        var pending = Assert.Single(await Studio(_ => throw new InvalidOperationException("The Turbo plug-in is still downloading."))
            .GetCapabilitiesAsync(CancellationToken.None));
        Assert.False(pending.Available);
        Assert.Equal("The Turbo plug-in is still downloading.", pending.UnavailableReason);
    }

    [Fact]
    public async Task ValidationHoldsTheShotToThePlannedRecipe()
    {
        LoadFixtureModel();
        var (config8, weights8) = Plugin(8);
        var studio = Studio(_ => [new LoraSpec(weights8, null, config8)]);

        var conflict = await Assert.ThrowsAsync<StudioException>(() => studio.ValidateAsync(Shot(recipe: "native"), NoAssets, CancellationToken.None));
        Assert.Equal("sampling_recipe_conflict", conflict.Code);
        Assert.Equal(409, StudioHostBoundary.Status(conflict));
        Assert.Equal("This shot was planned for the model's own schedule but the loaded model now samples with Test Turbo 8. " +
                     "Refresh the model settings and review the steps.", conflict.Message);
        var steps = await Assert.ThrowsAsync<StudioException>(() => studio.ValidateAsync(Shot(20), NoAssets, CancellationToken.None));
        Assert.Equal("invalid_generation", steps.Code);
        Assert.Equal("This model samples exactly 8 steps (Test Turbo 8); leave steps 0 for its default.", steps.Message);

        var unset = Shot();
        await studio.ValidateAsync(unset, NoAssets, CancellationToken.None);
        Assert.Equal(8, unset.Steps);
        await studio.ValidateAsync(Shot(8, Fixture8), NoAssets, CancellationToken.None);
        Assert.All(_providerCalls, forClip => Assert.False(forClip));
        Assert.Empty(_model.LoraSpecs);
    }

    [Fact]
    public async Task ThePlugInsRecipeHasTheLastWordInValidation()
    {
        // The choice changed between the capability read and the final check: the recipe the clip would run refuses the
        // step count the capability allowed, with the model's own text.
        LoadFixtureModel();
        var (config8, weights8) = Plugin(8);
        var (config4, weights4) = Plugin(4);
        int calls = 0;
        var studio = Studio(_ => ++calls == 1 ? [new LoraSpec(weights8, null, config8)] : [new LoraSpec(weights4, null, config4)]);
        var error = await Assert.ThrowsAsync<StudioException>(() => studio.ValidateAsync(Shot(), NoAssets, CancellationToken.None));
        Assert.Equal("invalid_generation", error.Code);
        Assert.StartsWith($"Test Turbo 4 ({W4}) samples exactly 4 steps on its own schedule; 8 steps contradicts it.", error.Message);
    }

    [Fact]
    public async Task TheRunTakesTheHostsChoiceOnceAndTheServiceChecksTheBoundRecipe()
    {
        // The shot was planned (and bound) under the 8-step plug-in; the user picked the 4-step one before it ran.
        var model = LoadFixtureModel();
        var (config8, weights8) = Plugin(8);
        var (config4, weights4) = Plugin(4);
        var studio = Studio(forClip => forClip ? [new LoraSpec(weights4, null, config4)] : [new LoraSpec(weights8, null, config8)]);
        var progress = new RecordingProgress();
        string output = Path.Combine(_dir, "out");

        var error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            studio.GenerateAsync(Shot(8, Fixture8), output, NoAssets, progress, CancellationToken.None));
        Assert.Equal($"The video model's sampling recipe is {Fixture4}; this job was admitted for {Fixture8}. Refresh and retry.", error.Message);
        // Validation asked twice without side effects, then the clip's choice was taken once and applied by the service
        // (under the model's GPU lock) before the recipe check refused the clip.
        Assert.Equal(new[] { false, false, true }, _providerCalls);
        Assert.Equal(weights4, Assert.Single(model.LoraSpecs).Path);
        Assert.Equal(4, model.SamplingRecipe.Steps);
        // The checkpoint and the clip's adapter were hashed for provenance before it; nothing was produced.
        Assert.Equal(2, progress.Reports.Count(p => p.Stage == "model-identity"));
        Assert.False(Directory.Exists(output));
    }
}
