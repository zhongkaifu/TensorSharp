using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorSharp.Chat.Studio;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Studio;

namespace InferenceWeb.Tests;

/// <summary>Contract parity only: these tests do not load weights or claim visual model quality.</summary>
public sealed class StudioGenerationParityTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-video-parity-" + Guid.NewGuid().ToString("N"));

    // Capability getters are metadata only. No native model is created, used, or disposed.
    static Ltx25Model MetadataModel() => (Ltx25Model)RuntimeHelpers.GetUninitializedObject(typeof(Ltx25Model));

    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }

    [Fact]
    public void LtxCapabilityUsesTheModelQualityDefaultWhenHostLeavesSizeUnset()
    {
        var capability = NativeStudioGenerationAdapter.CreateCapability(MetadataModel(), "ltx", new());

        var native = Ltx25GenerationSettings.Resolve(new() { Seed = 42 });
        Assert.Equal(native.Width, capability.DefaultWidth);
        Assert.Equal(native.Height, capability.DefaultHeight);
        Assert.Equal(1024, capability.DefaultWidth);
        Assert.Equal(native.Fps, capability.DefaultFps);
        Assert.Equal(native.Frames, capability.DefaultFrameCount);
        Assert.True(capability.DefaultGenerateAudio);
        Assert.Equal(Ltx25GenerationSettings.Steps, capability.DefaultSteps);
        Assert.Equal(1, capability.DefaultCfgScale);
    }

    [Theory]
    [InlineData(1024, 1024, 144, 24, 145)]
    [InlineData(1280, 768, 17, 30, 17)]
    [InlineData(512, 512, 9, 24, 9)] // An explicit host choice remains authoritative.
    public void StudioCapabilitiesInheritChatHostDefaults(int width, int height, int frames, int fps, int effectiveFrames)
    {
        var options = Options(width, height, frames, fps);
        var shared = VideoGenerationParamsParser.CreateDefaults(options);
        var capability = NativeStudioGenerationAdapter.CreateCapability(MetadataModel(), "ltx", shared);

        Assert.Equal(width, capability.DefaultWidth);
        Assert.Equal(height, capability.DefaultHeight);
        Assert.Equal(fps, capability.DefaultFps);
        Assert.Equal(effectiveFrames, capability.DefaultFrameCount);
        Assert.Equal(8, capability.DefaultSteps);
        Assert.Equal(frames, shared.Frames); // Capability snapping never rewrites chat defaults.
    }

    [Fact]
    public void FixedLtxSamplingScheduleOverridesAnIncompatibleHostDefault()
    {
        var defaults = new VideoGenerationParams { Steps = 20 };
        var capability = NativeStudioGenerationAdapter.CreateCapability(MetadataModel(), "ltx", defaults);
        Assert.Equal(8, capability.DefaultSteps);
        Assert.Equal(capability.FixedSteps, capability.DefaultSteps);
        Assert.Equal(20, defaults.Steps);
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new() { Steps = 20 }));
    }

    [Theory]
    [InlineData("minimax", 641, 385, 672, 416)]
    [InlineData("minimax", 0, 385, 416, 416)]
    [InlineData("minimax", 641, 0, 672, 672)]
    [InlineData("wan", 641, 385, 640, 384)]
    [InlineData("wan", 0, 385, 832, 384)]
    [InlineData("ltx", 641, 385, 641, 385)]
    public void InheritedDimensionsUseEachModelsNativeAlignment(string family, int width, int height, int expectedWidth, int expectedHeight)
    {
        IVideoGenerationModel model = family switch
        {
            "minimax" => (MiniMaxH3Model)RuntimeHelpers.GetUninitializedObject(typeof(MiniMaxH3Model)),
            "wan" => (WanVideoModel)RuntimeHelpers.GetUninitializedObject(typeof(WanVideoModel)),
            _ => MetadataModel(),
        };
        var capability = NativeStudioGenerationAdapter.CreateCapability(model, family, new() { Width = width, Height = height });
        Assert.Equal(expectedWidth, capability.DefaultWidth);
        Assert.Equal(expectedHeight, capability.DefaultHeight);
        if (family == "minimax")
        {
            var canvas = MiniMaxH3Pipeline.ResolveCanvas(new() { Width = width, Height = height }, MiniMaxH3Mode.TextToVideo);
            Assert.Equal(MiniMaxH3Geometry.AlignSpatial(canvas.Width), capability.DefaultWidth);
            Assert.Equal(MiniMaxH3Geometry.AlignSpatial(canvas.Height), capability.DefaultHeight);
        }
        if (family == "wan") Assert.Equal(WanVideoPipeline.SnapDim(height, capability.SpatialMultiple), capability.DefaultHeight);
        if (family == "ltx") Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new() { Width = capability.DefaultWidth, Height = capability.DefaultHeight }));
    }

    [Fact]
    public void StudioImageToVideoMatchesChatRecipeConditioningAndNoiseForTheSameSeed()
    {
        var options = Options(1024, 1024, 144, 24);
        Directory.CreateDirectory(options.UploadDirectory);
        string image = Path.Combine(options.UploadDirectory, "banner.png");
        File.WriteAllBytes(image, [1, 2, 3, 4]); // Neither contract path decodes pixels here.
        var capability = NativeStudioGenerationAdapter.CreateCapability(MetadataModel(), "ltx", VideoGenerationParamsParser.CreateDefaults(options));
        var request = new StudioGenerationRequest
        {
            ModelId = "ltx", Prompt = "A person smiles gently.", ConditioningAssetId = "image",
            Width = capability.DefaultWidth, Height = capability.DefaultHeight, FrameCount = capability.DefaultFrameCount,
            Fps = capability.DefaultFps, Steps = capability.DefaultSteps, CfgScale = capability.DefaultCfgScale,
            GenerateAudio = capability.DefaultGenerateAudio, Seed = 42,
        };
        var studio = NativeStudioGenerationAdapter.Parameters(request, new Dictionary<string, string> { ["image"] = image }, null, default);
        var body = JsonSerializer.SerializeToElement(new { prompt = request.Prompt, frames = 145, imagePath = "banner.png", videoMode = "i2v", seed = 42 });
        var chat = VideoGenerationParamsParser.Parse(body, options, out string error);

        Assert.Null(error);
        Assert.Equal(chat.Mode, studio.Mode);
        Assert.Equal(chat.GenerateAudio, studio.GenerateAudio);
        Assert.Equal(chat.ImageBytes, File.ReadAllBytes(studio.ImagePath));
        var chatGuide = Assert.Single(Ltx25ImageConditioning.Resolve(chat, 145));
        var studioGuide = Assert.Single(Ltx25ImageConditioning.Resolve(studio, 145));
        Assert.Equal(chatGuide.FrameIndex, studioGuide.FrameIndex);
        Assert.Equal(1, studioGuide.Strength);
        var chatSettings = Ltx25GenerationSettings.Resolve(chat);
        var studioSettings = Ltx25GenerationSettings.Resolve(studio);
        Assert.Equal(chatSettings, studioSettings);
        Assert.True(studioSettings.Ancestral);
        Assert.Equal(11, studioSettings.TotalSteps);
        Assert.Equal(512, studioSettings.HalfResolution().Width);
        Assert.Equal(chatSettings.HalfResolution().InitialNoise().Video, studioSettings.HalfResolution().InitialNoise().Video);
    }

    static T WithProperty<T>(T model, string property, object value)
    {
        typeof(T).GetField($"<{property}>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, value);
        return model;
    }
    static MiniMaxH3Model H3(MiniMaxH3Partition partition) =>
        WithProperty((MiniMaxH3Model)RuntimeHelpers.GetUninitializedObject(typeof(MiniMaxH3Model)), nameof(MiniMaxH3Model.Partition), partition);
    static WanVideoModel Wan(WanVariant variant, int inputChannels) => WithProperty(
        WithProperty((WanVideoModel)RuntimeHelpers.GetUninitializedObject(typeof(WanVideoModel)), nameof(WanVideoModel.Variant), variant),
        nameof(WanVideoModel.InputChannels), inputChannels);

    [Theory]
    [InlineData("h3-fl2va", "t2v,i2v,fl2v", "ref:Ref2VA")]
    [InlineData("h3-ref2va", "t2v,ref", "i2v:FL2VA,fl2v:FL2VA")]
    [InlineData("wan-t2v", "t2v", "i2v:text-to-video only,fl2v:last frame,ref:identity-reference")]
    [InlineData("wan-ti2v", "t2v,i2v", "fl2v:last frame,ref:identity-reference")]
    [InlineData("wan-i2v-a14b", "i2v", "t2v:first-frame image,fl2v:last frame,ref:identity-reference")]
    public void EachCheckpointAdvertisesTheModesItRunsAndExplainsTheOthers(string checkpoint, string modes, string reasons)
    {
        IVideoGenerationModel model = checkpoint switch
        {
            "h3-fl2va" => H3(MiniMaxH3Partition.FirstLastFrame),
            "h3-ref2va" => H3(MiniMaxH3Partition.Reference),
            "wan-t2v" => Wan(WanVariant.T2V, 16),
            "wan-ti2v" => Wan(WanVariant.TI2V, 48),
            _ => Wan(WanVariant.A14B, 36),
        };
        var capability = NativeStudioGenerationAdapter.CreateCapability(model, checkpoint, new());
        Assert.Equal(modes, string.Join(",", capability.Modes));
        // Each of t2v/i2v/fl2v/ref either runs or says why, so Studio never offers a dead control.
        var explained = reasons.Split(',').Select(r => r.Split(':')).ToDictionary(r => r[0], r => r[1]);
        Assert.Equal(explained.Keys.Order(), capability.UnavailableModes.Keys.Order());
        foreach (var (mode, reason) in explained) Assert.Contains(reason, capability.UnavailableModes[mode]);
        Assert.Null(capability.ReferenceMinFrames);
        // MiniMax publishes the clip lengths H3 was trained on (4-15 s); Wan states none here.
        double? min = checkpoint.StartsWith("h3") ? 4 : null, max = checkpoint.StartsWith("h3") ? 15 : null;
        Assert.Equal(min, capability.RecommendedMinSeconds); Assert.Equal(max, capability.RecommendedMaxSeconds);
    }

    [Theory]
    [InlineData("recipe", "i2v,fl2v")]
    [InlineData("pinned", "")]
    [InlineData("compact", "")]
    [InlineData("ref2va", "")]
    public void H3KeyframeModesSizeTheirCanvasFromTheFirstFrameOnlyOnARecipeHostThatPinsNoSize(string host, string expected)
    {
        // A host that pins a size keeps it for every mode, as chat does; a compact host (the TensorAgent app) keeps
        // 640x384; Ref2VA has no keyframe modes. Only an unpinned recipe host lets i2v/fl2v follow the first frame.
        var defaults = new VideoGenerationParams
        {
            Width = host == "pinned" ? 640 : 0, Height = host == "pinned" ? 384 : 0,
            CanvasPolicy = host == "compact" ? VideoCanvasPolicy.CompactPreview : VideoCanvasPolicy.ModelRecipe,
        };
        var model = H3(host == "ref2va" ? MiniMaxH3Partition.Reference : MiniMaxH3Partition.FirstLastFrame);
        var capability = NativeStudioGenerationAdapter.CreateCapability(model, host, defaults);
        Assert.Equal(expected, string.Join(",", capability.ImageCanvasModes));
        Assert.Equal(expected.Length > 0 ? MiniMaxH3Geometry.RecipeShortEdge : 0, capability.ImageCanvasShortEdge);
        Assert.Equal(expected.Length > 0 ? MiniMaxH3Geometry.RecipeMaxPixels : 0, capability.ImageCanvasMaxPixels);
        Assert.Equal((640, 384), (capability.DefaultWidth, capability.DefaultHeight)); // text-to-video keeps the compact canvas
    }

    [Fact]
    public void StudioFirstFrameCanvasIsTheCanvasChatGivesTheSameImage()
    {
        Directory.CreateDirectory(root);
        string path = Path.Combine(root, "banner.png");
        TensorSharp.Models.QwenImage.ImageIO.SavePng(path, new(313, 209, new float[313 * 209 * 3]));
        var studio = NativeStudioGenerationAdapter.FirstFrameCanvas(path);
        var chat = MiniMaxH3Pipeline.ResolveCanvas(new() { ImagePath = path }, MiniMaxH3Mode.ImageToVideo);
        Assert.Equal((1152, 768), studio);
        Assert.Equal(chat, studio);
        // A shape beyond the recipe's 4:1 is clamped, exactly as chat clamps it.
        string strip = Path.Combine(root, "strip.png");
        TensorSharp.Models.QwenImage.ImageIO.SavePng(strip, new(500, 100, new float[500 * 100 * 3]));
        Assert.Equal((2016, 512), NativeStudioGenerationAdapter.FirstFrameCanvas(strip));
        Assert.Equal(MiniMaxH3Pipeline.ResolveCanvas(new() { ImagePath = strip }, MiniMaxH3Mode.ImageToVideo), NativeStudioGenerationAdapter.FirstFrameCanvas(strip));
        Assert.Equal("invalid_generation", Assert.Throws<StudioException>(() => NativeStudioGenerationAdapter.FirstFrameCanvas(null)).Code);
    }

    [Theory]
    [InlineData("i2v", true, 1152, 768)]
    [InlineData("fl2v", true, 1152, 768)]  // the FIRST frame sizes it, not the portrait last frame
    [InlineData("t2v", true, 0, 0)]        // no image: the host default fills it later
    [InlineData("i2v", false, 0, 0)]       // a host without first-frame sizing keeps the zeros for its default
    public void StudioFillsAZeroSizeFromTheFirstFrameOnlyInAnImageCanvasMode(string mode, bool canvasHost, int width, int height)
    {
        Directory.CreateDirectory(root);
        string first = Path.Combine(root, "first.png"), last = Path.Combine(root, "last.png");
        TensorSharp.Models.QwenImage.ImageIO.SavePng(first, new(313, 209, new float[313 * 209 * 3]));
        TensorSharp.Models.QwenImage.ImageIO.SavePng(last, new(119, 178, new float[119 * 178 * 3]));
        var capability = NativeStudioGenerationAdapter.CreateCapability(H3(MiniMaxH3Partition.FirstLastFrame), "h3",
            new() { CanvasPolicy = canvasHost ? VideoCanvasPolicy.ModelRecipe : VideoCanvasPolicy.CompactPreview });
        var request = new StudioGenerationRequest { ModelId = "h3", Prompt = "p", Mode = mode,
            ConditioningAssetId = mode == "t2v" ? null : "first", EndConditioningAssetId = mode == "fl2v" ? "last" : null };
        Assert.Equal(mode, NativeStudioGenerationAdapter.ApplyImageCanvas(capability, request,
            mode == "t2v" ? null : first, mode == "fl2v" ? last : null));
        Assert.Equal((width, height), (request.Width, request.Height));
        // An explicit size is never replaced.
        var typed = new StudioGenerationRequest { ModelId = "h3", Prompt = "p", Mode = "i2v", ConditioningAssetId = "first", Width = 640, Height = 384 };
        NativeStudioGenerationAdapter.ApplyImageCanvas(capability, typed, first, null);
        Assert.Equal((640, 384), (typed.Width, typed.Height));
    }

    [Fact]
    public void StudioSizesTheCanvasFromTheLastFrameWhenItIsTheOnlyKeyframe()
    {
        Directory.CreateDirectory(root);
        string last = Path.Combine(root, "portrait-last.png");
        TensorSharp.Models.QwenImage.ImageIO.SavePng(last, new(119, 178, new float[119 * 178 * 3]));
        var capability = NativeStudioGenerationAdapter.CreateCapability(H3(MiniMaxH3Partition.FirstLastFrame), "h3",
            new() { CanvasPolicy = VideoCanvasPolicy.ModelRecipe });
        var request = new StudioGenerationRequest { ModelId = "h3", Prompt = "p", Mode = "fl2v", EndConditioningAssetId = "last" };
        Assert.Equal("fl2v", NativeStudioGenerationAdapter.ApplyImageCanvas(capability, request, null, last));
        // The portrait keyframe's own aspect at the recipe's 768 short edge, exactly the canvas chat gives it.
        Assert.Equal((768, 1152), (request.Width, request.Height));
        Assert.Equal(MiniMaxH3Pipeline.ResolveCanvas(new() { EndImagePath = last }, MiniMaxH3Mode.FirstLastFrame), (request.Width, request.Height));
    }

    [Theory]
    [InlineData(WanVariant.TI2V, 48)]
    [InlineData(WanVariant.A14B, 36)]
    public void WanRefusesALastFrameInsteadOfDroppingIt(WanVariant variant, int inputChannels)
    {
        var wan = Wan(variant, inputChannels);
        Assert.DoesNotContain("fl2v", NativeStudioGenerationAdapter.CreateCapability(wan, "wan", new()).Modes);
        var error = Assert.Throws<NotSupportedException>(() => WanVideoPipeline.ValidateConditioning(
            new() { EndImagePath = "last.png" }, hasImage: false, wan.SupportsImage, wan.RequiresImage));
        Assert.Contains("cannot pin the last frame", error.Message);
        Assert.Throws<NotSupportedException>(() => WanVideoPipeline.ValidateConditioning(
            new() { ImagePath = "first.png", EndImagePath = "last.png" }, hasImage: true, wan.SupportsImage, wan.RequiresImage));
        WanVideoPipeline.ValidateConditioning(new() { ImagePath = "first.png" }, hasImage: true, wan.SupportsImage, wan.RequiresImage);
        // A text-to-video-only checkpoint must not invite a photo it then refuses.
        Assert.False(((IVideoGenerationModel)Wan(WanVariant.T2V, 16)).SupportsImageConditioning);
        Assert.True(((IVideoGenerationModel)wan).SupportsImageConditioning);
    }

    // Studio asset IDs and the matching chat uploads, so both paths name the same files.
    Dictionary<string, string> ModeAssets(ServerHostingOptions options)
    {
        Directory.CreateDirectory(options.UploadDirectory);
        var assets = new Dictionary<string, string>();
        foreach (string name in new[] { "first", "last", "ref-a", "ref-b", "guide" })
        {
            string path = Path.Combine(options.UploadDirectory, name + ".png");
            File.WriteAllBytes(path, [1, 2, 3, 4]); // Mode resolution never decodes pixels.
            assets[name] = path;
        }
        return assets;
    }

    /// <summary>"fl2v-last" is fl2v with only its last frame, the first frame left empty.</summary>
    static StudioGenerationRequest ModeRequest(string mode) => new()
    {
        ModelId = "model", Prompt = "The person from Picture 1 walks through a market.", Mode = mode == "fl2v-last" ? "fl2v" : mode,
        Width = 1024, Height = 1024, FrameCount = 121, Fps = 24, Steps = 8, CfgScale = 1, Seed = 42,
        ConditioningAssetId = mode is "i2v" or "fl2v" or "keyframes" ? "first" : null,
        EndConditioningAssetId = mode is "fl2v" or "fl2v-last" ? "last" : null,
        TemporalConditions = mode == "keyframes" ? [new("guide", 64, .8f)] : [],
        ReferenceAssetIds = mode == "ref" ? ["ref-a", "ref-b"] : [],
    };

    static JsonElement ChatBody(string mode) => JsonSerializer.SerializeToElement(mode switch
    {
        "i2v" => (object)new { prompt = "p", videoMode = mode, imagePath = "first.png" },
        "fl2v" => new { prompt = "p", videoMode = mode, imagePath = "first.png", endImage = "last.png" },
        "fl2v-last" => new { prompt = "p", videoMode = "fl2v", endImage = "last.png" },
        "ref" => new { prompt = "p", videoMode = mode, referenceImages = new[] { "ref-a.png", "ref-b.png" } },
        _ => new { prompt = "p", videoMode = mode },
    });

    [Theory]
    [InlineData(MiniMaxH3Partition.FirstLastFrame, "t2v", MiniMaxH3Mode.TextToVideo)]
    [InlineData(MiniMaxH3Partition.FirstLastFrame, "i2v", MiniMaxH3Mode.ImageToVideo)]
    [InlineData(MiniMaxH3Partition.FirstLastFrame, "fl2v", MiniMaxH3Mode.FirstLastFrame)]
    [InlineData(MiniMaxH3Partition.FirstLastFrame, "fl2v-last", MiniMaxH3Mode.FirstLastFrame)]
    [InlineData(MiniMaxH3Partition.Reference, "t2v", MiniMaxH3Mode.TextToVideo)]
    [InlineData(MiniMaxH3Partition.Reference, "ref", MiniMaxH3Mode.Reference)]
    public void StudioModesReachMiniMaxH3AsTheSameModeChatUses(MiniMaxH3Partition partition, string mode, MiniMaxH3Mode expected)
    {
        var options = Options(640, 384, 22, 24);
        var assets = ModeAssets(options);
        var request = ModeRequest(mode);
        Assert.Equal(request.Mode, StudioGenerationModes.Resolve(request));
        Assert.Contains(request.Mode, NativeStudioGenerationAdapter.CreateCapability(H3(partition), "h3", new()).Modes);
        var studio = NativeStudioGenerationAdapter.Parameters(request, assets, null, default);
        var chat = VideoGenerationParamsParser.Parse(ChatBody(mode), options, out string error);

        Assert.Null(error);
        Assert.Equal(expected, MiniMaxH3ModeResolver.Resolve(studio, partition));
        Assert.Equal(expected, MiniMaxH3ModeResolver.Resolve(chat, partition));
        Assert.Equal(chat.ReferenceImagePaths ?? [], studio.ReferenceImagePaths); // Same files, same order: <Picture 1>, <Picture 2>.
        Assert.Equal(chat.EndImagePath, studio.EndImagePath);
        if (mode == "fl2v-last") // a last frame alone leaves the first frame empty on both paths: nothing stands in for it
            foreach (var p in new[] { studio, chat })
                Assert.True(p.Image == null && p.ImageBytes == null && p.ImagePath == null && p.EndImagePath != null);
    }

    [Theory]
    [InlineData(MiniMaxH3Partition.FirstLastFrame, "ref")]
    [InlineData(MiniMaxH3Partition.Reference, "i2v")]
    [InlineData(MiniMaxH3Partition.Reference, "fl2v")]
    public void MiniMaxH3RefusesAModeThatNeedsTheOtherCheckpoint(MiniMaxH3Partition partition, string mode)
    {
        Assert.DoesNotContain(mode, NativeStudioGenerationAdapter.CreateCapability(H3(partition), "h3", new()).Modes);
        var studio = NativeStudioGenerationAdapter.Parameters(ModeRequest(mode), ModeAssets(Options(640, 384, 22, 24)), null, default);
        Assert.Throws<InvalidOperationException>(() => MiniMaxH3ModeResolver.Resolve(studio, partition));
    }

    [Theory]
    [InlineData("t2v", new int[0])]
    [InlineData("i2v", new[] { 0 })]
    [InlineData("fl2v", new[] { 0, 120 })]
    [InlineData("fl2v-last", new[] { 120 })]   // a last frame alone: one appended guide at the final frame
    [InlineData("keyframes", new[] { 0, 64 })]
    public void StudioModesReachLtxAsTheIntendedFrameGuides(string mode, int[] frames)
    {
        var parameters = NativeStudioGenerationAdapter.Parameters(ModeRequest(mode), ModeAssets(Options(1024, 1024, 121, 24)), null, default);
        var settings = Ltx25GenerationSettings.Resolve(parameters);
        Assert.True(settings.TwoStage);
        Assert.Equal(frames, Ltx25ImageConditioning.Resolve(parameters, settings.Frames).Select(image => image.FrameIndex));
        // Only a lone first frame gets the publisher's softened first-stage strength.
        Assert.Equal(mode == "i2v" ? .7f : null, Ltx25Pipeline.ResolveFirstStageImageStrength(parameters, twoStage: true,
            Ltx25ImageConditioning.Resolve(parameters, settings.Frames)));
    }

    [Fact]
    public void StudioReferenceModeReachesLtxIngredientsAsIdentityImagesNotFrames()
    {
        var parameters = NativeStudioGenerationAdapter.Parameters(ModeRequest("ref"), ModeAssets(Options(1024, 1024, 121, 24)), null, default);
        var settings = Ltx25GenerationSettings.Resolve(parameters, supportsIdentityReferences: true);
        Assert.False(settings.TwoStage);
        Assert.Equal(["ref-a.png", "ref-b.png"], Ltx25ReferenceConditioning.ResolveSources(parameters).Select(source => Path.GetFileName(source.ImagePath)));
        Assert.Throws<NotSupportedException>(() => Ltx25GenerationSettings.Resolve(parameters, supportsIdentityReferences: false));
        var shortClip = ModeRequest("ref"); shortClip.FrameCount = 97;
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(
            NativeStudioGenerationAdapter.Parameters(shortClip, ModeAssets(Options(1024, 1024, 121, 24)), null, default), supportsIdentityReferences: true));
    }

    ServerHostingOptions Options(int width, int height, int frames, int fps) => ServerOptionsBuilder.Build(
        ["--video-width", width.ToString(), "--video-height", height.ToString(), "--video-frames", frames.ToString(), "--fps", fps.ToString(), "--video-steps", "8"], root);
}
