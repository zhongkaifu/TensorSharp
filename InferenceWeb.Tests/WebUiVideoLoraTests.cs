// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// The chat service's side of a video model's LoRA plug-ins (the MiniMax-H3 Turbo plug-in):
/// <list type="bullet">
/// <item>a clip request's plug-ins are applied, and the sampling recipe it was admitted for is
/// checked, under the guard every clip takes on both the plain and the streamed path: the
/// model it found still loaded, a use of it held, its GPU lock held;</item>
/// <item>a request the plug-in's recipe contradicts is refused with the model's text, a 400
/// or a <c>{ done, error }</c> frame;</item>
/// <item>an unload waits for plug-ins being applied, and the set they built is disposed once;</item>
/// <item>the warnings a load gives for inherited <c>TS_LORAS</c> plug-ins the model skips.</item>
/// </list>
/// The models are stand-ins with the plug-in contract and no weights, and the plug-in configs
/// are synthetic (their weights files are never read): this is evidence about the service, not
/// about an adapter or a clip.
/// </summary>
public sealed class WebUiVideoLoraTests : IDisposable
{
    internal static readonly TimeSpan TestTimeout = TimeSpan.FromSeconds(30);
    internal const string Prompt = "a red circle rolls across the frame";

    /// <summary>The refusals of <see cref="MiniMaxH3SamplingRecipe.Validate"/> for the
    /// fixture's 8-step plug-in, spelled out so the text itself is pinned.</summary>
    internal const string StepsRefusal =
        "Test Turbo 8 (minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors) samples exactly 8 steps on its own schedule; " +
        "20 steps contradicts it. Leave the step count unset (--diffusion-steps on the CLI, --video-steps on the server, " +
        "\"steps\" in a request) or set it to 8.";

    internal const string ShiftRefusal =
        "Test Turbo 8 (minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors) samples with video flow shift 6 (audio 3); " +
        "flow shift 12 contradicts it. Leave it unset (--flow-shift, \"flowShift\") or set it to 6.";

    private const string Unloading = "The selected video model is being unloaded.";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-video-lora-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new();
    private readonly MiniMaxH3LoraFixtures _fixtures = new();
    private readonly Dictionary<string, ModelBase> _byFile = new(StringComparer.Ordinal);
    private readonly List<TurboStandIn> _turbos = new();
    private readonly RecordingLog _log = new();
    private readonly ModelService _models;
    private readonly WebUiChatService _chat;
    private readonly string _manifest8, _manifest4, _qwenManifest;

    public WebUiVideoLoraTests()
    {
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
        _env.Set(LoraCliFlags.EnvironmentVariable, null);
        _env.Set(QwenImageVariantFlag.EnvironmentVariable, null);
        // The plug-ins' weights only have to exist: the service and the stand-in resolve the
        // specs as the model does, and resolving a plug-in whose weights are missing refuses it.
        _fixtures.WriteText(MiniMaxH3LoraFixtures.Weights8, "stand-in weights, never read");
        _fixtures.WriteText(MiniMaxH3LoraFixtures.Weights4, "stand-in weights, never read");
        _manifest8 = _fixtures.WriteManifest();
        _manifest4 = _fixtures.WriteManifest("turbo-4step.json", MiniMaxH3LoraFixtures.Weights4, steps: 4, name: "Test Turbo 4");
        _fixtures.WriteText("qwen-adapter.safetensors", "stand-in weights, never read");
        _qwenManifest = _fixtures.WriteText("qwen-plugin.json",
            $$"""{ "type": "{{LoraCliFlags.QwenImagePluginType}}", "weights": "qwen-adapter.safetensors" }""");
        _models = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, _) => _byFile[Path.GetFileName(path)]);
        _chat = new WebUiChatService(_models, new SessionManager(), Options(_dir),
            new UploadStoragePolicy(Path.Combine(_dir, "uploads")),
            new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() }),
            codeRunner: null, workspaces: null, codeArtifacts: null, _log);
    }

    public void Dispose()
    {
        // A failed assertion must not leave a request parked inside the process-wide video lock.
        foreach (var turbo in _turbos)
            turbo.ReleaseAll();
        _models.Dispose();
        _fixtures.Dispose();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ---- request refusals ----------------------------------------------------------------

    [Theory]
    [InlineData("steps", 20f, StepsRefusal)]
    [InlineData("flowShift", 12f, ShiftRefusal)]
    public async Task TheStreamEndsWithTheRecipesRefusal(string field, float value, string refusal)
    {
        var model = LoadTurbo();
        var frames = await Stream(Body(field, value), Plugin8);
        Assert.Equal(refusal, FinalError(frames));
        Assert.Equal(1, model.SetLorasCalls);
        // The plug-ins were applied and the clip begun with the model's GPU lock held.
        Assert.True(model.SetLorasHeldGpuLock);
        Assert.True(model.GenerateHeldGpuLock);
    }

    [Theory]
    [InlineData("steps", 20f, StepsRefusal)]
    [InlineData("flowShift", 12f, ShiftRefusal)]
    public async Task ThePlainRouteAnswersTheRecipesRefusalWith400(string field, float value, string refusal)
    {
        LoadTurbo();
        var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
            () => _chat.VideoGenerateAsync(Body(field, value), Plugin8, CancellationToken.None).WaitAsync(TestTimeout));
        Assert.Equal(400, ex.StatusCode);
        Assert.Equal(refusal, Error(ex.Payload));
    }

    [Fact]
    public async Task GenerateVideoAsyncPropagatesTheRecipesRefusalUnwrapped()
    {
        // Studio's path: its caller shapes the refusal, so it arrives as the model threw it.
        var model = LoadTurbo();
        var ex = await Assert.ThrowsAsync<ArgumentException>(() => _chat.GenerateVideoAsync(
            Prompt, new VideoGenerationParams { Steps = 20 }, model, Plugin8, expectedSamplingRecipe: null).WaitAsync(TestTimeout));
        Assert.Equal(StepsRefusal, ex.Message);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0)]
    [InlineData(8)]
    public async Task UnsetZeroOrTheRecipesOwnStepCountReachTheModel(int? steps)
    {
        var model = LoadTurbo();
        object body = steps is { } n ? new { prompt = Prompt, steps = n } : new { prompt = Prompt };
        var frames = await Stream(JsonSerializer.SerializeToElement(body), Plugin8);
        Assert.Equal(TurboStandIn.Reached, FinalError(frames));
        Assert.Equal(steps ?? 0, model.LastParams!.Steps);
        Assert.Equal(1, model.GenerateCalls);
    }

    // ---- which plug-ins a clip is made with ------------------------------------------------

    [Fact]
    public async Task ASetIsAppliedOnceKeptForNullAndRemovedByAnEmptyList()
    {
        var model = LoadTurbo();

        Assert.Equal(TurboStandIn.Reached, await PlainError(Plugin8));
        Assert.Equal(1, model.SetLorasCalls);
        Assert.Equal("minimax-h3-lora:turbo-8step", model.SamplingContract?.Id);
        var resolved = Assert.Single(model.LoraSpecs);
        Assert.Equal(Path.Combine(_fixtures.Dir, MiniMaxH3LoraFixtures.Weights8), resolved.Path);
        Assert.Equal(_manifest8, resolved.ConfigPath);

        // The same plug-in, given as its config again: the model holds it resolved, and it is
        // compared in that form rather than packed again for every clip.
        Assert.Equal(TurboStandIn.Reached, await PlainError(Plugin8));
        Assert.Equal(TurboStandIn.Reached, FinalError(await Stream(Body(), Plugin8)));
        Assert.Equal(1, model.SetLorasCalls);

        // Null: the caller does not choose, and the model keeps what it has.
        Assert.Equal(TurboStandIn.Reached, await PlainError(null));
        Assert.Equal(TurboStandIn.Reached, FinalError(await Stream(Body(), null)));
        Assert.Equal(1, model.SetLorasCalls);
        Assert.Equal("minimax-h3-lora:turbo-8step", model.SamplingContract?.Id);

        // Empty: the clip is made without plug-ins, and the set is released.
        Assert.Equal(TurboStandIn.Reached, FinalError(await Stream(Body(), Array.Empty<LoraSpec>())));
        Assert.Equal(2, model.SetLorasCalls);
        Assert.Empty(model.LoraSpecs);
        Assert.Null(model.SamplingContract);
        Assert.Equal(1, Assert.Single(model.Sets).DisposeCount);
        Assert.Equal(6, model.GenerateCalls);

        var applied = _log.Messages(LogLevel.Information).Where(m => m.StartsWith("Video request: LoRA plug-ins ", StringComparison.Ordinal)).ToList();
        Assert.Equal(2, applied.Count);
        Assert.StartsWith("Video request: LoRA plug-ins turbo-8step.json@ applied in ", applied[0]);
        Assert.StartsWith("Video request: LoRA plug-ins (none) applied in ", applied[1]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ASetTheModelRefusesKeepsThePreviousOne(bool stream)
    {
        var model = LoadTurbo();
        Assert.Equal(TurboStandIn.Reached, await PlainError(Plugin8));
        var before = model.LoraSpecs;

        var qwen = new[] { new LoraSpec(_qwenManifest) };
        string error;
        if (stream)
        {
            error = FinalError(await Stream(Body(), qwen));
        }
        else
        {
            var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
                () => _chat.VideoGenerateAsync(Body(), qwen, CancellationToken.None).WaitAsync(TestTimeout));
            Assert.Equal(500, ex.StatusCode);
            error = Error(ex.Payload);
        }

        Assert.Equal("The LoRA plug-ins could not be applied: --lora qwen-plugin.json is a qwen-image-2.1-lora plug-in; " +
            "MiniMax-H3 (minimax-h3) takes minimax-h3-lora plug-ins only.", error);
        Assert.Equal(2, model.SetLorasCalls);
        Assert.Same(before, model.LoraSpecs);
        Assert.Equal("minimax-h3-lora:turbo-8step", model.SamplingContract?.Id);
        Assert.Equal(0, Assert.Single(model.Sets).DisposeCount);
        // Nothing was generated with the set the request did not get.
        Assert.Equal(1, model.GenerateCalls);
    }

    [Fact]
    public async Task AVideoModelWithoutPluginsRefusesASetButNotAnEmptyOrNullOne()
    {
        var model = Load("plain.gguf", path => new PlainVideoStandIn(path));

        var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
            () => _chat.VideoGenerateAsync(Body(), Plugin8, CancellationToken.None).WaitAsync(TestTimeout));
        Assert.Equal(500, ex.StatusCode);
        Assert.Equal("wan takes no LoRA plug-ins.", Error(ex.Payload));
        Assert.Equal("wan takes no LoRA plug-ins.", FinalError(await Stream(Body(), Plugin8)));
        Assert.Equal(0, model.GenerateCalls);

        Assert.Equal(TurboStandIn.Reached, FinalError(await Stream(Body(), Array.Empty<LoraSpec>())));
        Assert.Equal(TurboStandIn.Reached, await PlainError(null));
        Assert.Equal(2, model.GenerateCalls);
    }

    // ---- the admitted sampling recipe --------------------------------------------------------

    [Theory]
    [InlineData(null, "minimax-h3-lora:turbo-8step", "native")]
    [InlineData("8", VideoSamplingContract.NativeId, "minimax-h3-lora:turbo-8step")]
    [InlineData("4", "minimax-h3-lora:turbo-8step", "minimax-h3-lora:turbo-4step")]
    public async Task AJobAdmittedForAnotherRecipeIsRefusedBeforeTheClip(string? plugin, string admitted, string actual)
    {
        var model = LoadTurbo();
        IReadOnlyList<LoraSpec> loras = plugin switch { "8" => Plugin8, "4" => [new LoraSpec(_manifest4)], _ => null };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => _chat.GenerateVideoAsync(
            Prompt, new VideoGenerationParams(), model, loras, admitted).WaitAsync(TestTimeout));

        Assert.Equal($"The video model's sampling recipe is {actual}; this job was admitted for {admitted}. Refresh and retry.", ex.Message);
        Assert.Equal(0, model.GenerateCalls);
        // The request's plug-ins were applied first: the recipe checked is the one the clip would use.
        Assert.Equal(loras == null ? 0 : 1, model.SetLorasCalls);
    }

    [Theory]
    [InlineData(null, VideoSamplingContract.NativeId)]
    [InlineData("8", "minimax-h3-lora:turbo-8step")]
    public async Task AJobAdmittedForTheModelsRecipeRuns(string? plugin, string admitted)
    {
        var model = LoadTurbo();
        IReadOnlyList<LoraSpec> loras = plugin == "8" ? Plugin8 : null;
        var ex = await Assert.ThrowsAsync<NotSupportedException>(() => _chat.GenerateVideoAsync(
            Prompt, new VideoGenerationParams(), model, loras, admitted).WaitAsync(TestTimeout));
        Assert.Equal(TurboStandIn.Reached, ex.Message);
        Assert.Equal(1, model.GenerateCalls);
    }

    // ---- the guard on the streamed path ------------------------------------------------------

    [Fact]
    public async Task TheStreamRefusesAModelSwappedWhileItWasQueued()
    {
        var first = LoadTurbo("first.gguf");
        var second = Register("second.gguf", path => new TurboStandIn(path));
        first.HoldGenerate = new ManualResetEventSlim();

        // A clip on the first model holds the video lock; a streamed request queues behind it,
        // having found the first model loaded.
        var running = _chat.GenerateVideoAsync(Prompt, new VideoGenerationParams());
        Assert.True(first.GenerateEntered.Wait(TestTimeout));
        var queued = _chat.VideoGenerateStreamAsync(Body(), null, CancellationToken.None).GetAsyncEnumerator();
        var queuedFirst = queued.MoveNextAsync();

        // The second model is loaded meanwhile: the first is unpublished at once and disposed
        // once the running clip leaves it.
        var swap = Task.Run(() => _models.LoadModel(Path.Combine(_dir, "second.gguf"), null, "cpu"));
        Assert.True(SpinWait.SpinUntil(() => !ReferenceEquals(_models.Model, first), TestTimeout));
        first.HoldGenerate.Set();

        var done = await Assert.ThrowsAsync<NotSupportedException>(() => running.WaitAsync(TestTimeout));
        Assert.Equal(TurboStandIn.Reached, done.Message);
        Assert.Equal("The selected video model changed while generation was queued.", FinalError(await Drain(queued, queuedFirst)));
        await swap.WaitAsync(TestTimeout);

        Assert.Same(second, _models.Model);
        Assert.Equal(1, first.GenerateCalls);
        Assert.Equal(0, second.GenerateCalls);
        Assert.Equal(1, first.DisposeCalls);
        Assert.False(first.DisposedWhileBusy);
    }

    [Fact]
    public async Task TheStreamRefusesAModelAlreadyBeingUnloaded()
    {
        var model = LoadTurbo();
        model.BeginRetirement();
        Assert.Equal("The video model is being unloaded; retry after loading it.", FinalError(await Stream(Body(), Plugin8)));
        Assert.Equal(0, model.SetLorasCalls);
        Assert.Equal(0, model.GenerateCalls);
    }

    [Fact]
    public async Task TheStreamRefusesAModelThatBeganUnloadingWhileItWaitedForTheGpu()
    {
        var model = LoadTurbo();
        IAsyncEnumerator<object> frames;
        ValueTask<bool> first;
        lock (model.GpuComputeLock)
        {
            frames = _chat.VideoGenerateStreamAsync(Body(), Plugin8, CancellationToken.None).GetAsyncEnumerator();
            first = frames.MoveNextAsync();
            // The request holds a use of the model and waits for its GPU lock.
            Assert.True(SpinWait.SpinUntil(() => !model.WaitForUsesToDrain(TimeSpan.Zero), TestTimeout));
            model.BeginRetirement();
        }
        Assert.Equal(Unloading, FinalError(await Drain(frames, first)));
        Assert.Equal(0, model.SetLorasCalls);
        Assert.Equal(0, model.GenerateCalls);
        Assert.True(model.WaitForUsesToDrain(TimeSpan.Zero));
    }

    /// <summary>
    /// The B1 race: an unload issued while a clip's plug-ins are being applied (a MiniMax-H3
    /// adapter takes seconds to pack) waits for them under the model's GPU lock, the clip is then
    /// refused rather than run, and the set that was built is disposed once, with the model.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task AnUnloadWaitsForPluginsBeingAppliedAndTheSetIsDisposedOnce(bool stream)
    {
        var model = LoadTurbo();
        model.HoldSetLoras = new ManualResetEventSlim();
        // The unload stops waiting for the request's use almost at once, so only the model's
        // GPU lock, held by the request, keeps the dispose out.
        TimeSpan drain = ModelLifecycleService.UseDrainTimeout;
        ModelLifecycleService.UseDrainTimeout = TimeSpan.FromMilliseconds(50);
        try
        {
            Task<string> request = stream
                ? Task.Run(async () => FinalError(await Stream(Body(), Plugin8)))
                : PlainError(Plugin8);
            Assert.True(model.SetLorasEntered.Wait(TestTimeout));
            Assert.True(model.SetLorasHeldGpuLock);
            Assert.False(TryTake(model.GpuComputeLock));

            var unload = Task.Run(() => _models.UnloadModel());
            Assert.True(SpinWait.SpinUntil(() => model.IsRetiring, TestTimeout));
            Assert.Null(_models.Model);
            await Task.Delay(300);
            Assert.False(unload.IsCompleted);
            Assert.Equal(0, model.DisposeCalls);

            model.HoldSetLoras.Set();
            Assert.Equal(Unloading, await request.WaitAsync(TestTimeout));
            await unload.WaitAsync(TestTimeout);
        }
        finally
        {
            ModelLifecycleService.UseDrainTimeout = drain;
        }

        Assert.Equal(0, model.GenerateCalls);
        Assert.Equal(1, model.DisposeCalls);
        Assert.False(model.DisposedWhileBusy);
        Assert.Equal(1, Assert.Single(model.Sets).DisposeCount);
    }

    // ---- the load's warnings for inherited plug-ins -----------------------------------------

    [Fact]
    public void ALoadNamesInheritedPluginsOfAnotherFamilyOneWarningPerType()
    {
        string qwen2 = _fixtures.WriteText("qwen-plugin-2.json",
            $$"""{ "type": "{{LoraCliFlags.QwenImagePluginType}}", "weights": "qwen-adapter.safetensors" }""");
        string bare = _fixtures.WriteText("adapter.safetensors", "bare weights");
        _env.Set(LoraCliFlags.EnvironmentVariable, LoraCliFlags.ToJson(new[]
        {
            new LoraSpec(_qwenManifest), new LoraSpec(bare), new LoraSpec(_manifest8), new LoraSpec(qwen2, 0.5f),
        }));

        Assert.Equal(new[]
        {
            "TS_LORAS names LoRA plug-ins qwen-plugin.json, qwen-plugin-2.json of type qwen-image-2.1-lora; " +
            "turbo.gguf (minimax-h3) takes minimax-h3-lora plug-ins and runs without them.",
            "TS_LORAS names LoRA plug-ins adapter.safetensors of type none (bare weights); " +
            "turbo.gguf (minimax-h3) takes minimax-h3-lora plug-ins and runs without them.",
        }, LoadWarnings("turbo.gguf", path => new TurboStandIn(path)));
    }

    [Fact]
    public void ALoadNamesEveryInheritedPluginForAModelThatTakesNoneWithoutReadingThem()
    {
        string missing = Path.Combine(_dir, "missing-plugin.json");
        _env.Set(LoraCliFlags.EnvironmentVariable, LoraCliFlags.ToJson(new[] { new LoraSpec(_manifest8), new LoraSpec(missing) }));
        Assert.Equal(new[]
        {
            "TS_LORAS names LoRA plug-ins (turbo-8step.json, missing-plugin.json), but plain.gguf (wan) takes none; it runs without them.",
        }, LoadWarnings("plain.gguf", path => new PlainVideoStandIn(path)));
    }

    [Fact]
    public void ALoadReportsAnUnreadablePluginByNameInsteadOfFailing()
    {
        string missing = Path.Combine(_dir, "missing-plugin.json");
        string broken = _fixtures.WriteText("broken.json", "{ \"type\": ");
        _env.Set(LoraCliFlags.EnvironmentVariable, LoraCliFlags.ToJson(new[] { new LoraSpec(missing), new LoraSpec(broken), new LoraSpec(_manifest8) }));

        var warnings = LoadWarnings("turbo.gguf", path => new TurboStandIn(path));

        Assert.Equal(2, warnings.Count);
        Assert.Equal($"TS_LORAS names LoRA plug-in missing-plugin.json, which cannot be read: --lora file not found: {missing}", warnings[0]);
        Assert.StartsWith($"TS_LORAS names LoRA plug-in broken.json, which cannot be read: LoRA config {broken} is not valid JSON: ", warnings[1]);
    }

    [Fact]
    public void ALoadSaysNothingOfPluginsTheModelTakes()
    {
        _env.Set(LoraCliFlags.EnvironmentVariable, LoraCliFlags.ToJson(new[] { new LoraSpec(_manifest8) }));
        Assert.Empty(LoadWarnings("turbo.gguf", path => new TurboStandIn(path)));
    }

    [Fact]
    public void ALoadKeepsTheQwenVariantWarning()
    {
        _env.Set(QwenImageVariantFlag.EnvironmentVariable, "turbo");
        Assert.Equal(new[]
        {
            "--qwen-image-variant / TS_QWEN_IMAGE_VARIANT (turbo) applies to Qwen-Image-2.1 models only; plain.gguf (wan) is not one and ignores it.",
        }, LoadWarnings("plain.gguf", path => new PlainVideoStandIn(path)));
    }

    // ---- helpers -----------------------------------------------------------------------------

    private IReadOnlyList<LoraSpec> Plugin8 => [new LoraSpec(_manifest8)];

    internal static ServerHostingOptions Options(string dir) => new(
        startupModelPath: null, startupMmProjPath: null, defaultBackend: "cpu",
        supportedBackends: null, defaultMaxTokens: 100, maxTokensPinned: false,
        defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
        defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
        uploadDirectory: Path.Combine(dir, "uploads"), logDirectory: Path.Combine(dir, "logs"), fileLoggingEnabled: false,
        samplingDefaults: null, prefixCacheEnabled: false);

    /// <summary>An empty GGUF: enough for a load's header check and a stand-in model.</summary>
    internal static string WriteStandInGguf(string dir, string name)
    {
        string path = Path.Combine(dir, name);
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);
        writer.Write(0UL);
        writer.Write(0UL);
        writer.Write(new byte[8]);
        return path;
    }

    private T Register<T>(string name, Func<string, T> create) where T : ModelBase
    {
        var model = create(WriteStandInGguf(_dir, name));
        _byFile[name] = model;
        if (model is TurboStandIn turbo) _turbos.Add(turbo);
        return model;
    }

    private T Load<T>(string name, Func<string, T> create) where T : ModelBase
    {
        var model = Register(name, create);
        _models.LoadModel(Path.Combine(_dir, name), null, "cpu");
        Assert.Same(model, _models.Model);
        return model;
    }

    private TurboStandIn LoadTurbo(string name = "turbo.gguf") => Load(name, path => new TurboStandIn(path));

    private List<string> LoadWarnings(string name, Func<string, ModelBase> create)
    {
        var log = new RecordingLog();
        string path = WriteStandInGguf(_dir, name);
        using var lifecycle = new ModelLifecycleService(log, (p, _, _, _) => create(p));
        lifecycle.LoadModel(path, null, "cpu");
        Assert.True(lifecycle.IsLoaded);
        return log.Messages(LogLevel.Warning);
    }

    private static JsonElement Body() => JsonSerializer.SerializeToElement(new { prompt = Prompt });

    private static JsonElement Body(string field, float value) =>
        JsonSerializer.SerializeToElement(new Dictionary<string, object> { ["prompt"] = Prompt, [field] = value });

    private async Task<List<JsonElement>> Stream(JsonElement body, IReadOnlyList<LoraSpec> loras)
    {
        var frames = _chat.VideoGenerateStreamAsync(body, loras, CancellationToken.None).GetAsyncEnumerator();
        return await Drain(frames, frames.MoveNextAsync());
    }

    private static async Task<List<JsonElement>> Drain(IAsyncEnumerator<object> frames, ValueTask<bool> first)
    {
        var result = new List<JsonElement>();
        try
        {
            for (bool more = await first.AsTask().WaitAsync(TestTimeout); more; more = await frames.MoveNextAsync().AsTask().WaitAsync(TestTimeout))
                result.Add(JsonSerializer.SerializeToElement(frames.Current));
        }
        finally
        {
            await frames.DisposeAsync();
        }
        return result;
    }

    /// <summary>The <c>{ done, error }</c> frame a refused stream ends with.</summary>
    private static string FinalError(List<JsonElement> frames)
    {
        var last = Assert.Single(frames);
        Assert.True(last.GetProperty("done").GetBoolean());
        return last.GetProperty("error").GetString();
    }

    private static string Error(object payload) =>
        JsonSerializer.SerializeToElement(payload).GetProperty("error").GetString();

    /// <summary>What a clip on the plain Studio path threw (every clip here stops at the model).</summary>
    private async Task<string> PlainError(IReadOnlyList<LoraSpec> loras)
    {
        var ex = await Assert.ThrowsAnyAsync<Exception>(() => _chat.GenerateVideoAsync(
            Prompt, new VideoGenerationParams(), null, loras, expectedSamplingRecipe: null).WaitAsync(TestTimeout));
        Assert.True(ex is NotSupportedException or InvalidOperationException, ex.ToString());
        return ex.Message;
    }

    private static bool TryTake(object gate)
    {
        if (!Monitor.TryEnter(gate)) return false;
        Monitor.Exit(gate);
        return true;
    }

    // ---- stand-ins ---------------------------------------------------------------------------

    /// <summary>
    /// A MiniMax-H3 with the plug-in contract of the real model and no weights: SetLoras takes
    /// the recipe <see cref="MiniMaxH3LoraConfig.RecipeFor(IReadOnlyList{LoraSpec}, MiniMaxH3Partition)"/>
    /// reads from the configs (which refuses another family's plug-in or bare weights with the
    /// canonical texts), resolves the set and keeps it; GenerateVideo checks the
    /// request against that recipe, or the base rules, as the pipeline does, then stops with
    /// <see cref="Reached"/>. It takes no lock of its own around SetLoras, GenerateVideo and
    /// Dispose, so only the chat service's guard keeps an unload out of them, and it records an
    /// unload that got in. Gates hold SetLoras (the adapter packing) or GenerateVideo open.
    /// </summary>
    internal sealed class TurboStandIn : ModelBase, IVideoGenerationModel, ILoraPluginHost
    {
        internal const string Reached = "The clip request reached the model.";

        private StandInSet _set;
        private int _busy;
        private bool _disposed;

        internal TurboStandIn(string path) : base(path, BackendType.Cpu) =>
            Config = new ModelConfig { Architecture = MiniMaxH3Model.ArchitectureId };

        internal int SetLorasCalls, GenerateCalls, DisposeCalls;
        internal bool SetLorasHeldGpuLock, GenerateHeldGpuLock, DisposedWhileBusy;
        internal VideoGenerationParams LastParams;
        internal Exception ThrowOnGenerate;
        internal readonly List<StandInSet> Sets = new();
        internal readonly ManualResetEventSlim SetLorasEntered = new(), GenerateEntered = new();
        internal ManualResetEventSlim HoldSetLoras, HoldGenerate;

        internal void ReleaseAll()
        {
            HoldSetLoras?.Set();
            HoldGenerate?.Set();
        }

        public string VideoModelFamily => "minimax-h3";
        public bool SupportsAudio => true;
        public bool SupportsImageConditioning => true;
        public bool SupportsEndImageConditioning => true;
        public bool SupportsReferenceConditioning => false;
        public VideoSamplingContract SamplingContract => _set?.Recipe.ToContract();

        public string LoraPluginType => LoraCliFlags.MiniMaxH3PluginType;
        public IReadOnlyList<LoraSpec> LoraSpecs { get; private set; } = Array.Empty<LoraSpec>();
        public bool AcceptsLora(LoraSpec spec) => LoraCliFlags.DeclaredPluginType(spec) == LoraCliFlags.MiniMaxH3PluginType;

        public void SetLoras(IReadOnlyList<LoraSpec> specs)
        {
            SetLorasCalls++;
            SetLorasHeldGpuLock = Monitor.IsEntered(GpuComputeLock);
            Interlocked.Increment(ref _busy);
            try
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(TurboStandIn), "MiniMax-H3 was unloaded; its LoRA plug-ins were not applied.");
                var recipe = MiniMaxH3LoraConfig.RecipeFor(specs, MiniMaxH3Partition.FirstLastFrame);
                var resolved = LoraCliFlags.Resolve(specs);
                SetLorasEntered.Set();
                HoldSetLoras?.Wait(TestTimeout);
                var next = recipe == null ? null : new StandInSet(recipe);
                if (next != null) Sets.Add(next);
                var old = _set;
                _set = next;
                LoraSpecs = resolved;
                old?.Dispose();
            }
            finally
            {
                Interlocked.Decrement(ref _busy);
            }
        }

        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams p = null)
        {
            GenerateCalls++;
            GenerateHeldGpuLock = Monitor.IsEntered(GpuComputeLock);
            Interlocked.Increment(ref _busy);
            try
            {
                LastParams = p;
                GenerateEntered.Set();
                HoldGenerate?.Wait(TestTimeout);
                if (ThrowOnGenerate != null)
                    throw ThrowOnGenerate;
                if (_set != null)
                    _set.Recipe.Validate(p ?? new VideoGenerationParams());
                else
                    MiniMaxH3Pipeline.ValidateSamplingParameters(p ?? new VideoGenerationParams());
                throw new NotSupportedException(Reached);
            }
            finally
            {
                Interlocked.Decrement(ref _busy);
            }
        }

        public override void Dispose()
        {
            DisposeCalls++;
            if (Volatile.Read(ref _busy) > 0)
                DisposedWhileBusy = true;
            if (!_disposed)
            {
                _disposed = true;
                _set?.Dispose();
                _set = null;
            }
            base.Dispose();
        }

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    /// <summary>The plug-in set a <see cref="TurboStandIn"/> built, counting its disposals.</summary>
    internal sealed class StandInSet(MiniMaxH3SamplingRecipe recipe) : IDisposable
    {
        internal MiniMaxH3SamplingRecipe Recipe { get; } = recipe;
        internal int DisposeCount;
        public void Dispose() => Interlocked.Increment(ref DisposeCount);
    }

    /// <summary>A video model that takes no LoRA plug-ins.</summary>
    internal sealed class PlainVideoStandIn : ModelBase, IVideoGenerationModel
    {
        internal PlainVideoStandIn(string path) : base(path, BackendType.Cpu) => Config = new ModelConfig { Architecture = "wan" };

        internal int GenerateCalls;
        public string VideoModelFamily => "wan";
        public bool SupportsAudio => false;
        public bool SupportsImageConditioning => true;
        public bool SupportsEndImageConditioning => false;
        public bool SupportsReferenceConditioning => false;

        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams p = null)
        {
            GenerateCalls++;
            throw new NotSupportedException(TurboStandIn.Reached);
        }

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    /// <summary>A logger and logger factory that keep every formatted message.</summary>
    internal sealed class RecordingLog : ILoggerFactory, ILogger
    {
        private readonly ConcurrentQueue<(LogLevel Level, string Message)> _entries = new();

        internal List<string> Messages(LogLevel level) => _entries.Where(e => e.Level == level).Select(e => e.Message).ToList();

        public ILogger CreateLogger(string categoryName) => this;
        public void AddProvider(ILoggerProvider provider) { }
        public void Dispose() { }
        public IDisposable BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception,
            Func<TState, Exception, string> formatter) => _entries.Enqueue((logLevel, formatter(state, exception)));
    }
}
