// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Studio;

namespace TensorSharp.Chat.Studio;

/// <summary>Uses the host's active model and existing chat video pipeline. Never loads or substitutes a model.</summary>
public sealed class NativeStudioGenerationAdapter : IStudioGenerationAdapter
{
    private static readonly ConcurrentDictionary<string, string> ModelHashes = new(StringComparer.Ordinal);
    private readonly ModelService _models;
    private readonly WebUiChatService _chat;
    private readonly Func<CancellationToken, Task> _prepare;
    private readonly TensorSharp.Runtime.Scheduling.ComputeGate _compute;
    private readonly Func<bool, IReadOnlyList<LoraSpec>> _videoLoras;
    /// <param name="videoLoras">The LoRA plug-ins each clip runs with, for a host that chooses them per clip (the
    /// TensorAgent app); null leaves the loaded model's own (the server's <c>--lora</c>). It is called with false for a
    /// capability or validation query, which must have no side effects, and with true once for the clip that runs.</param>
    public NativeStudioGenerationAdapter(ModelService models, WebUiChatService chat, Func<CancellationToken, Task> prepare = null, TensorSharp.Runtime.Scheduling.ComputeGate compute = null,
        Func<bool, IReadOnlyList<LoraSpec>> videoLoras = null)
    { _models = models; _chat = chat; _prepare = prepare; _compute = compute; _videoLoras = videoLoras; }

    public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_models.Model is not IVideoGenerationModel video)
            return Task.FromResult<IReadOnlyList<StudioModelCapability>>(Array.Empty<StudioModelCapability>());
        // The contract the next clip would sample with: the host's plug-in choice, read from the plug-in configs alone,
        // else the model's own. A choice the model would refuse leaves nothing to offer, so it is the unavailable reason.
        VideoSamplingContract contract;
        try { contract = PlannedContract(video, _videoLoras?.Invoke(false)); }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            var refused = CreateCapability(video, _models.LoadedModelName, _chat.VideoDefaults, null);
            refused.Available = false; refused.UnavailableReason = e.Message;
            return Task.FromResult<IReadOnlyList<StudioModelCapability>>(new[] { refused });
        }
        return Task.FromResult<IReadOnlyList<StudioModelCapability>>(new[] { CreateCapability(video, _models.LoadedModelName, _chat.VideoDefaults, contract) });
    }

    /// <summary>The sampling contract <paramref name="specs"/> bind <paramref name="video"/> to, read from the plug-in
    /// configs alone (never a download); null specs keep the model's own plug-ins. Null for the model's own schedule.</summary>
    /// <exception cref="ArgumentException">The model refuses the plug-ins, with its canonical text.</exception>
    /// <exception cref="InvalidOperationException">Plug-ins for a model that takes none.</exception>
    internal static VideoSamplingContract PlannedContract(IVideoGenerationModel video, IReadOnlyList<LoraSpec> specs) =>
        specs == null ? video.SamplingContract
        : video is MiniMaxH3Model h3 ? h3.ContractFor(specs)
        : specs.Count > 0 ? throw new InvalidOperationException($"{video.VideoModelFamily} takes no LoRA plug-ins.")
        : video.SamplingContract;

    internal static StudioModelCapability CreateCapability(IVideoGenerationModel video, string modelId, VideoGenerationParams defaults) =>
        CreateCapability(video, modelId, defaults, video.SamplingContract);

    /// <summary>The Studio capability of <paramref name="video"/> sampling under <paramref name="contract"/> (null: its own
    /// schedule). A contract fixes the steps and guidance, names the recipe a job is bound to, and leaves the model
    /// unavailable on a host whose <c>--video-steps</c> contradicts it.</summary>
    internal static StudioModelCapability CreateCapability(IVideoGenerationModel video, string modelId, VideoGenerationParams defaults,
                                                           VideoSamplingContract contract)
    {
        var capability = new StudioModelCapability
        {
            ModelId = modelId, DisplayName = modelId,
            Family = video.VideoModelFamily, Available = true, TextToVideo = true,
            ImageConditioning = video.SupportsImageConditioning, LastFrameConditioning = video.SupportsEndImageConditioning,
            AudioOutput = video.SupportsAudio, DefaultFrameCount = video.DefaultPreviewFrames,
            TemporalGuidance = video.SupportsKeyframeConditioning, IdentityReferences = video.SupportsReferenceConditioning, MaxReferences = video.MaxReferenceImages,
            DeferredCapabilities = new[] { "LoRA selection, native retake and extension are not exposed by the Studio generation adapter. Existing model APIs remain available." }
        };
        switch (video)
        {
            case Ltx25Model ltx:
                capability.SpatialMultiple = 64; capability.TemporalMultiple = 8;
                capability.DefaultWidth = capability.DefaultHeight = Ltx25GenerationSettings.DefaultSize;
                capability.DefaultSteps = 8; capability.FixedSteps = 8;
                capability.DefaultCfgScale = 1; capability.FixedCfgScale = 1;
                if (!ltx.SupportsImageConditioning)
                {
                    const string encoder = "LTX image conditioning needs the desktop FFmpeg/libx264 image transform on this host; install FFmpeg on PATH or set TS_FFMPEG.";
                    capability.UnavailableModes[StudioGenerationModes.ImageToVideo] = encoder;
                    capability.UnavailableModes[StudioGenerationModes.FirstLastFrame] = encoder;
                    capability.UnavailableModes[StudioGenerationModes.Keyframes] = encoder;
                }
                if (ltx.SupportsReferenceConditioning) capability.ReferenceMinFrames = Ltx25ReferenceConditioning.MinimumFrames;
                else capability.UnavailableModes[StudioGenerationModes.Reference] = "LTX identity/appearance references need the Ingredients IC-LoRA " +
                    "(loras/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors beside the checkpoint, or TS_LTX25_INGREDIENTS_LORA). Install it and reload the model.";
                break;
            case MiniMaxH3Model h3:
                capability.SpatialMultiple = 32; capability.TemporalMultiple = 17; capability.TemporalOffset = 5;
                capability.DefaultWidth = 640; capability.DefaultHeight = 384;
                capability.ImageConditioning = h3.Partition != MiniMaxH3Partition.Reference;
                capability.FixedFps = 24; capability.DefaultCfgScale = 1; capability.FixedCfgScale = 1;
                capability.DefaultSteps = MiniMaxH3Scheduler.DefaultSteps;
                // MiniMax's README (and the vllm-omni / sglang recipes) give 4-15 s at 24 fps; the default stays a short preview.
                capability.RecommendedMinSeconds = 4; capability.RecommendedMaxSeconds = 15;
                // FL2VA and Ref2VA are separate checkpoint files, not settings.
                if (h3.Partition == MiniMaxH3Partition.Reference)
                {
                    const string keyframes = "MiniMax-H3 first- or last-frame conditioning needs the FL2VA checkpoint (minimax_h3_fl2va_pruned-*.gguf); " +
                        "the loaded Ref2VA checkpoint treats images as identity references. Load the FL2VA file to pin frames.";
                    capability.UnavailableModes[StudioGenerationModes.ImageToVideo] = keyframes;
                    capability.UnavailableModes[StudioGenerationModes.FirstLastFrame] = keyframes;
                }
                else capability.UnavailableModes[StudioGenerationModes.Reference] = "MiniMax-H3 identity/appearance references need the Ref2VA checkpoint " +
                    "(minimax_h3_ref2va_pruned-*.gguf); the loaded FL2VA checkpoint pins frames. Load the Ref2VA file to use references.";
                break;
            case WanVideoModel wan:
                capability.SpatialMultiple = wan.Variant == WanVariant.TI2V ? 32 : 16;
                capability.DefaultWidth = wan.Variant == WanVariant.TI2V ? 1280 : 832;
                capability.DefaultHeight = wan.Variant == WanVariant.TI2V ? 704 : 480;
                capability.DefaultFps = wan.Variant == WanVariant.TI2V ? 24 : 16;
                capability.ImageConditioning = wan.SupportsImage; capability.TextToVideo = !wan.RequiresImage;
                capability.DefaultSteps = wan.DistilledSteps > 0 ? wan.DistilledSteps : wan.Variant == WanVariant.TI2V ? 50 : wan.Variant == WanVariant.A14B ? 40 : 30;
                capability.DefaultCfgScale = wan.DistilledSteps > 0 ? 1 : 5;
                if (wan.RequiresImage) capability.UnavailableModes[StudioGenerationModes.TextToVideo] = "This Wan2.2 image-to-video checkpoint always animates a first-frame image.";
                if (!wan.SupportsImage) capability.UnavailableModes[StudioGenerationModes.ImageToVideo] = "This Wan checkpoint is text-to-video only. Wan2.2-TI2V-5B or Wan2.2-I2V-A14B animates a first frame.";
                capability.UnavailableModes[StudioGenerationModes.FirstLastFrame] = "Wan takes at most a first-frame image; it cannot pin a last frame, alone or with a first. " +
                    "MiniMax-H3 FL2VA or LTX 2.5 can end a clip on an image.";
                capability.UnavailableModes[StudioGenerationModes.Reference] = "Wan has no identity-reference conditioning. MiniMax-H3 Ref2VA or LTX 2.5 with the Ingredients IC-LoRA supports ref.";
                break;
            default:
                capability.Available = false;
                capability.UnavailableReason = "This model family has no validated Studio parameter adapter yet.";
                break;
        }
        // Studio must inherit the same host-configured recipe as chat. In particular,
        // overriding LTX's 1024 output with a generic 512 canvas reduces its first
        // denoising pass to 256 pixels, destroying fine face and hand detail.
        if (defaults.Width > 0) capability.DefaultWidth = defaults.Width;
        if (defaults.Height > 0) capability.DefaultHeight = defaults.Height;
        // These families accept arbitrary host dimensions and align them inside
        // native generation. Advertise that effective canvas so Studio's stricter
        // admission accepts its own defaults. LTX deliberately rejects off-grid sizes.
        if (video is MiniMaxH3Model h3Model)
        {
            // Without a conditioning image, H3 uses a square for a one-sided
            // size override. An image's aspect cannot be known at capability time.
            if (defaults.Width > 0 && defaults.Height <= 0) capability.DefaultHeight = defaults.Width;
            else if (defaults.Height > 0 && defaults.Width <= 0) capability.DefaultWidth = defaults.Height;
            capability.DefaultWidth = MiniMaxH3Geometry.AlignSpatial(capability.DefaultWidth);
            capability.DefaultHeight = MiniMaxH3Geometry.AlignSpatial(capability.DefaultHeight);
            // i2v/fl2v keep the first frame's framing, so its canvas comes from that image by MiniMax's 768p
            // recipe, exactly as chat sizes it. A host that pins both sides keeps that size for every mode, as chat
            // does; a one-sided pin squares here (above) while chat derives the other side from the photo.
            if (h3Model.Partition != MiniMaxH3Partition.Reference && defaults.CanvasPolicy == VideoCanvasPolicy.ModelRecipe
                && defaults.Width <= 0 && defaults.Height <= 0)
            {
                capability.ImageCanvasModes = [StudioGenerationModes.ImageToVideo, StudioGenerationModes.FirstLastFrame];
                capability.ImageCanvasShortEdge = MiniMaxH3Geometry.RecipeShortEdge;
                capability.ImageCanvasMaxPixels = MiniMaxH3Geometry.RecipeMaxPixels;
                capability.ImageCanvasMaxAspect = MiniMaxH3Geometry.RecipeMaxAspect;
            }
        }
        else if (video is WanVideoModel)
        {
            int grid = capability.SpatialMultiple;
            capability.DefaultWidth = Math.Max(grid, (capability.DefaultWidth + grid / 2) / grid * grid);
            capability.DefaultHeight = Math.Max(grid, (capability.DefaultHeight + grid / 2) / grid * grid);
        }
        if (defaults.Fps > 0) capability.DefaultFps = defaults.Fps;
        if (capability.FixedFps is int fixedFps) capability.DefaultFps = fixedFps;
        // A recipe (a step-distillation plug-in's) fixes the step count, so the host's --video-steps default cannot
        // override it: a host started with another count offers the model as unavailable rather than a refused run.
        if (contract != null)
        {
            capability.FixedSteps = capability.DefaultSteps = contract.Steps;
            capability.FixedCfgScale = capability.DefaultCfgScale = contract.CfgScale;
            capability.SamplingRecipe = contract.Id; capability.SamplingRecipeName = contract.DisplayName;
            capability.DisplayName += " · " + contract.DisplayName;
            string reason = video is MiniMaxH3Model { Partition: MiniMaxH3Partition.Reference }
                ? $"{contract.DisplayName} ({contract.Source}) is a MiniMax-H3 FL2VA plug-in; the loaded Ref2VA checkpoint does not take it, " +
                  "and Ref2VA Turbo is not supported. Load minimax_h3_fl2va_pruned-*.gguf or drop the plug-in."
                : defaults.Steps > 0 && defaults.Steps != contract.Steps
                ? $"{LoraHostPreflight.ServerStepsFlag} {defaults.Steps} contradicts {contract.DisplayName}, which samples exactly {contract.Steps} steps. " +
                  $"Start the host without {LoraHostPreflight.ServerStepsFlag} or with {LoraHostPreflight.ServerStepsFlag} {contract.Steps}."
                : null;
            if (reason != null && capability.Available) { capability.Available = false; capability.UnavailableReason = reason; }
        }
        if (defaults.Steps > 0) capability.DefaultSteps = defaults.Steps;
        if (capability.FixedSteps is int fixedSteps) capability.DefaultSteps = fixedSteps;
        capability.DefaultGenerateAudio = capability.AudioOutput && defaults.GenerateAudio;
        if (defaults.Frames > 0)
        {
            // Chat's composer snaps duration to the model grid before generation.
            // Expose that effective count, so e.g. --video-frames 144 is usable as 145.
            int frames = defaults.Frames;
            capability.DefaultFrameCount = video switch
            {
                MiniMaxH3Model => MiniMaxH3Geometry.AlignFrames(frames),
                WanVideoModel => Math.Max(1, (frames + 2) / 4 * 4 + 1),
                _ => capability.TemporalOffset + Math.Max(0, checked(frames - capability.TemporalOffset + capability.TemporalMultiple - 1)) / capability.TemporalMultiple * capability.TemporalMultiple
            };
        }
        return capability;
    }

    public async Task ValidateAsync(StudioGenerationRequest request, IReadOnlyDictionary<string, string> conditioningAssets, CancellationToken cancellationToken)
    {
        string conditioningPath = Resolve(conditioningAssets, request.ConditioningAssetId);
        string endConditioningPath = Resolve(conditioningAssets, request.EndConditioningAssetId);
        var capabilities = await GetCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
        var capability = capabilities.SingleOrDefault(c => c.ModelId == request.ModelId);
        if (capability == null || !capability.Available)
            throw new StudioException("model_unavailable", capability?.UnavailableReason ?? "The requested video model is no longer loaded. Select and load it explicitly before retrying.");
        capability.RequireSamplingRecipe(request);
        string mode = ApplyImageCanvas(capability, request, conditioningPath, endConditioningPath);
        if (request.Width == 0) request.Width = capability.DefaultWidth;
        if (request.Height == 0) request.Height = capability.DefaultHeight;
        if (request.Fps == 0) request.Fps = capability.DefaultFps;
        request.GenerateAudio ??= capability.DefaultGenerateAudio;
        if (request.Steps == 0) request.Steps = capability.DefaultSteps;
        if (request.CfgScale == 0) request.CfgScale = capability.DefaultCfgScale;
        if (request.FrameCount == 0) request.FrameCount = capability.DefaultFrameCount;
        if (capability.FixedSteps is int fixedSteps && request.Steps != fixedSteps)
            throw new StudioException("invalid_generation", capability.FixedStepsRefusal());
        if (capability.FixedCfgScale is float fixedCfg && request.CfgScale != fixedCfg)
            throw new StudioException("invalid_generation", "This checkpoint requires its fixed guidance. Refresh the model's capability settings.");
        if (request.Width <= 0 || request.Height <= 0 || request.Width % capability.SpatialMultiple != 0 || request.Height % capability.SpatialMultiple != 0 || request.Width > capability.MaxDimension || request.Height > capability.MaxDimension)
            throw new StudioException("invalid_generation", $"Dimensions must be positive multiples of {capability.SpatialMultiple}, at most {capability.MaxDimension}.");
        if (request.FrameCount < capability.TemporalOffset || (request.FrameCount - capability.TemporalOffset) % capability.TemporalMultiple != 0 || request.FrameCount > capability.MaxFrames)
            throw new StudioException("invalid_generation", $"Frames must follow {capability.TemporalMultiple}n+{capability.TemporalOffset}, at most {capability.MaxFrames}.");
        if (request.Fps is < 1 or > 240 || capability.FixedFps is int fps && request.Fps != fps || request.Steps is < 1 or > 100 || !float.IsFinite(request.CfgScale) || request.CfgScale is < 1 or > 30)
            throw new StudioException("invalid_generation", "Unsupported frame rate, sampling steps, or guidance.");
        if (!capability.Modes.Contains(mode))
            throw new StudioException("unsupported_mode", capability.UnavailableModes.TryGetValue(mode, out var reason) ? reason
                : $"The loaded checkpoint does not support {mode} generation. Supported: {string.Join(", ", capability.Modes)}.");
        if (mode == StudioGenerationModes.Reference && capability.ReferenceMinFrames is int minimumFrames && request.FrameCount < minimumFrames)
            throw new StudioException("invalid_generation", $"This model's reference workflow requires at least {minimumFrames} frames.");
        if (conditioningPath != null && !capability.ImageConditioning || endConditioningPath != null && !capability.LastFrameConditioning || request.GenerateAudio == true && !capability.AudioOutput)
            throw new StudioException("unsupported_conditioning", "The loaded checkpoint does not support the requested conditioning or audio output.");
        if (request.CfgScale == 1 && !string.IsNullOrWhiteSpace(request.NegativePrompt))
            throw new StudioException("unsupported_negative_prompt", "Negative prompts do not run at CFG 1. Clear the negative prompt rather than silently ignoring it.");
        if (request.TemporalConditions.Count > 0 && !capability.TemporalGuidance || request.ReferenceAssetIds.Count > 0 && !capability.IdentityReferences)
            throw new StudioException("unsupported_conditioning", "The loaded checkpoint does not support these temporal guides or identity references.");
        if (request.TemporalConditions.Count + request.ReferenceAssetIds.Count > Math.Max(0, capability.MaxReferences))
            throw new StudioException("invalid_conditioning", $"This model accepts at most {capability.MaxReferences} guides/references.");
        foreach (var guide in request.TemporalConditions)
            if (guide.FrameIndex < -1 || guide.FrameIndex >= request.FrameCount || !float.IsFinite(guide.Strength) || guide.Strength is < 0 or > 1)
                throw new StudioException("invalid_conditioning", "Temporal guide position or strength is invalid.");
        // The native resolvers are the final authority on each family's combination rules.
        try
        {
            if (_models.Model is Ltx25Model ltx)
                _ = Ltx25GenerationSettings.Resolve(Parameters(request, conditioningAssets, null, cancellationToken), ltx.SupportsReferenceConditioning);
            if (_models.Model is MiniMaxH3Model h3)
                _ = MiniMaxH3ModeResolver.Resolve(Parameters(request, conditioningAssets, null, cancellationToken), h3.Partition);
            // So is a plug-in's recipe, which holds Studio to the request rules it holds chat to.
            if (_models.Model is MiniMaxH3Model plugged)
                (_videoLoras == null ? plugged.SamplingRecipe : MiniMaxH3LoraConfig.RecipeFor(_videoLoras(false), plugged.Partition))
                    ?.Validate(Parameters(request, conditioningAssets, null, cancellationToken));
        }
        catch (Exception e) when (e is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            throw new StudioException("invalid_generation", e.Message);
        }
    }

    /// <summary>Resolve the request's mode and, for width = height = 0 in one of the capability's
    /// <see cref="StudioModelCapability.ImageCanvasModes"/>, replace the zeros with the canvas the first-frame image
    /// gives (the last-frame image when there is no first). Returns the mode.</summary>
    internal static string ApplyImageCanvas(StudioModelCapability capability, StudioGenerationRequest request,
                                            string conditioningPath, string endConditioningPath)
    {
        string mode = StudioGenerationModes.Resolve(request);
        if (request.Width == 0 && request.Height == 0 && capability.ImageCanvasModes.Contains(mode))
            (request.Width, request.Height) = FirstFrameCanvas(conditioningPath ?? endConditioningPath);
        return mode;
    }

    /// <summary>The canvas MiniMax-H3's recipe gives this keyframe (the first frame, or the last when it is alone, as
    /// sglang's "last_keyframe" geometry and diffusers' anchor do), read from the image as the pipeline decodes it
    /// (EXIF orientation applied), so a phone photo stored on its side is sized upright, as chat sizes it.</summary>
    internal static (int Width, int Height) FirstFrameCanvas(string path)
    {
        if (path == null)
            throw new StudioException("invalid_generation", "Sizing the canvas from the image needs a first-frame or last-frame image.");
        TensorSharp.Models.QwenImage.RgbImage image;
        try { image = TensorSharp.Models.QwenImage.ImageIO.Load(path); }
        catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or ArgumentException)
        {
            throw new StudioException("invalid_generation", "The keyframe image could not be read to size the canvas: " + e.Message);
        }
        if (image.Width <= 0 || image.Height <= 0)
            throw new StudioException("invalid_generation", "The keyframe image has no pixels to size the canvas from.");
        return MiniMaxH3Geometry.KeyframeCanvas(image.Width, image.Height);
    }

    public async Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest request, string outputDirectory,
        IReadOnlyDictionary<string, string> conditioningAssets, IProgress<StudioProgress> progress, CancellationToken cancellationToken)
    {
        if (_prepare != null) await _prepare(cancellationToken).ConfigureAwait(false);
        var expectedModel = _models.Model as IVideoGenerationModel;
        string expectedName = _models.LoadedModelName;
        string expectedPath = _models.LoadedModelPath;
        string expectedBackend = _models.LoadedBackend;
        await ValidateAsync(request, conditioningAssets, cancellationToken).ConfigureAwait(false);
        if (expectedModel == null || expectedName != request.ModelId || !ReferenceEquals(expectedModel, _models.Model))
            throw new StudioException("model_changed", "The selected model changed during admission. Select it explicitly and retry.");
        // The plug-ins this clip runs with: the host's choice, made once for the run, or (null) the model's own. The chat
        // service applies them and re-checks the job's bound recipe under the model's GPU lock; these copies are for
        // provenance, so it names the files and the schedule the clip actually sampled.
        var loras = _videoLoras?.Invoke(true);
        var plugins = (loras ?? (expectedModel as ILoraPluginHost)?.LoraSpecs ?? Array.Empty<LoraSpec>()).Select(PluginWeights).ToList();
        var h3Recipe = expectedModel is MiniMaxH3Model turbo
            ? loras == null ? turbo.SamplingRecipe : MiniMaxH3LoraConfig.RecipeFor(loras, turbo.Partition) : null;
        string recipeId = (expectedModel is MiniMaxH3Model ? h3Recipe?.ToContract() : PlannedContract(expectedModel, loras))?.Id
            ?? VideoSamplingContract.NativeId;
        using var interrupted = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        void ForegroundChanged(bool open) { if (!open) interrupted.Cancel(); }
        if (_compute != null) _compute.Changed += ForegroundChanged;
        try
        {
            if (_compute != null && !_compute.IsOpen) interrupted.Cancel();
            cancellationToken = interrupted.Token;
            string identity = expectedName + "; backend=" + expectedBackend;
            var source = new FileInfo(expectedPath);
            string checksum = await Sha256Async(source, progress, 0, "Computing the installed checkpoint SHA-256 once for generation provenance.", cancellationToken).ConfigureAwait(false);
            identity += "; sha256=" + checksum + "; bytes=" + source.Length.ToString(CultureInfo.InvariantCulture);
            if (expectedModel is MiniMaxH3Model h3)
                identity += "; checkpoint=" + (h3.Partition == MiniMaxH3Partition.Reference ? "ref2va" : "fl2va");
            // Each plug-in's adapter file is hashed like the checkpoint: before the clip when it is on disk, after it when
            // the run fetched it (a plug-in config names its download).
            var pluginHashes = new string[plugins.Count];
            async Task HashPluginsAsync(bool fetched, double fraction)
            {
                for (int i = 0; i < plugins.Count; i++)
                    if (pluginHashes[i] == null && (fetched || File.Exists(plugins[i].Weights)))
                        pluginHashes[i] = await Sha256Async(new FileInfo(plugins[i].Weights), progress, fraction,
                            "Computing the LoRA plug-in's SHA-256 once for generation provenance.", cancellationToken).ConfigureAwait(false);
            }
            await HashPluginsAsync(fetched: false, 0).ConfigureAwait(false);
            var parameters = Parameters(request, conditioningAssets, progress, cancellationToken);
            var result = await _chat.GenerateVideoAsync(request.Prompt, parameters, expectedModel, loras, request.SamplingRecipe).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            if (result.Width != request.Width || result.Height != request.Height || result.Frames != request.FrameCount || result.Fps != request.Fps)
                throw new StudioException("unexpected_output_geometry", "Native generation returned different dimensions, frame count, or frame rate. The output was not accepted into the project; inspect model capabilities before retrying.");
            if (request.GenerateAudio == true && !result.AudioMuxed)
                throw new StudioException("audio_not_muxed", "The native encoder produced separate audio. Studio requires muxed audio to preserve preview/export parity; use an available local muxing encoder.");
            await HashPluginsAsync(fetched: true, .95).ConfigureAwait(false);
            identity += PluginIdentity(plugins, pluginHashes, recipeId);
            Directory.CreateDirectory(outputDirectory);
            string path = Path.Combine(outputDirectory, "generated.mp4");
            File.Copy(result.OutputPath, path, overwrite: false);
            var effective = new Dictionary<string, string>
            {
                ["width"] = result.Width.ToString(CultureInfo.InvariantCulture), ["height"] = result.Height.ToString(CultureInfo.InvariantCulture),
                ["frames"] = result.Frames.ToString(CultureInfo.InvariantCulture), ["fps"] = result.Fps.ToString(CultureInfo.InvariantCulture),
                ["seed"] = result.Seed.ToString(CultureInfo.InvariantCulture), ["steps"] = request.Steps.ToString(CultureInfo.InvariantCulture),
                ["cfgScale"] = request.CfgScale.ToString(CultureInfo.InvariantCulture), ["codec"] = result.Codec,
                ["audioMuxed"] = result.AudioMuxed.ToString(), ["mode"] = parameters.Mode,
                ["recipe"] = expectedModel is Ltx25Model ? parameters.Mode == StudioGenerationModes.Reference ? "LTX Ingredients: 8 full-size steps; euler_ancestral" : "LTX distilled: 8 half-size steps + 3 full-size refinement steps; euler_ancestral" : "native model sampling"
            };
            effective["samplingRecipe"] = recipeId;
            // MiniMax-H3 names the exact schedule it sampled, in place of the generic recipe text.
            if (expectedModel is MiniMaxH3Model)
                foreach (var (key, value) in MiniMaxH3Sampling(h3Recipe, request.Steps)) effective[key] = value;
            if (plugins.Count > 0) effective["lora"] = string.Join(", ", plugins.Select(PluginLabel));
            return new StudioGenerationResult(path, identity, effective);
        }
        finally { if (_compute != null) _compute.Changed -= ForegroundChanged; }
    }

    /// <summary>A file's SHA-256, computed once per path, size and write time; the progress line says it is happening.</summary>
    internal static async Task<string> Sha256Async(FileInfo source, IProgress<StudioProgress> progress, double fraction, string detail, CancellationToken cancellationToken)
    {
        string cacheKey = source.FullName + "|" + source.Length + "|" + source.LastWriteTimeUtc.Ticks;
        if (ModelHashes.TryGetValue(cacheKey, out string checksum)) return checksum;
        progress.Report(new StudioProgress("model-identity", fraction, detail));
        await using var file = new FileStream(source.FullName, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        checksum = Convert.ToHexStringLower(await SHA256.HashDataAsync(file, cancellationToken).ConfigureAwait(false));
        ModelHashes.TryAdd(cacheKey, checksum);
        return checksum;
    }

    /// <summary>The adapter file a plug-in spec applies and the strength it runs at, read from its plug-in config alone
    /// (never a download): <c>--lora config.json</c> names its weights, and a MiniMax-H3 config's <c>"scale"</c> is the
    /// strength when the spec sets none.</summary>
    internal static (string Weights, float Strength) PluginWeights(LoraSpec spec)
    {
        string manifest = LoraCliFlags.ManifestPath(spec);
        bool h3 = manifest != null && LoraCliFlags.DeclaredPluginType(new LoraSpec(manifest)) == LoraCliFlags.MiniMaxH3PluginType;
        string weights = manifest != null && spec.Path.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? ConfigFileArgs.ResolveFileEntryPath(manifest, "weights") ?? spec.Path : spec.Path;
        return (weights, spec.Scale ?? (h3 ? MiniMaxH3LoraConfig.Load(manifest).Scale : null) ?? 1f);
    }

    private static string PluginLabel((string Weights, float Strength) plugin) =>
        Path.GetFileName(plugin.Weights) + "@" + plugin.Strength.ToString(CultureInfo.InvariantCulture);

    /// <summary>The provenance identity's tail: each plug-in's adapter file at its strength with the file's SHA-256, then
    /// the sampling recipe the clip was bound to.</summary>
    internal static string PluginIdentity(IReadOnlyList<(string Weights, float Strength)> plugins, IReadOnlyList<string> hashes, string recipeId) =>
        string.Concat(plugins.Select((plugin, i) => "; lora=" + PluginLabel(plugin) + "; lora_sha256=" + hashes[i])) + "; recipe=" + recipeId;

    /// <summary>The schedule a MiniMax-H3 clip sampled, for provenance, as the pipeline resolves a Studio request (which
    /// sets no flow shift or sampler): the plug-in's recipe, else the base checkpoint's stable-diffusion.cpp grid at the
    /// request's steps, video shift 12 / audio 3.</summary>
    internal static Dictionary<string, string> MiniMaxH3Sampling(MiniMaxH3SamplingRecipe recipe, int steps)
    {
        static string Num(float value) => value.ToString(CultureInfo.InvariantCulture);
        float[] sigmas = recipe?.VideoSigmas() ?? MiniMaxH3Scheduler.BuildSigmas(steps);
        var sampling = new Dictionary<string, string>
        {
            ["recipe"] = recipe?.Describe() ?? $"MiniMax-H3: {steps} steps, sd.cpp discrete grid, video shift {Num(MiniMaxH3Scheduler.VideoShift)} / " +
                $"audio {Num(MiniMaxH3Scheduler.AudioShift)}, euler, cfg 1",
            ["videoShift"] = Num(recipe?.VideoShift ?? MiniMaxH3Scheduler.VideoShift),
            ["audioShift"] = Num(recipe?.AudioShift ?? MiniMaxH3Scheduler.AudioShift),
            ["sigmaGrid"] = string.Join(", ", sigmas.Select(s => s.ToString("0.#####", CultureInfo.InvariantCulture))),
        };
        if (recipe != null) sampling["samplingRecipeName"] = recipe.Name;
        return sampling;
    }

    private static string Resolve(IReadOnlyDictionary<string, string> assets, string id) => id == null ? null :
        assets.TryGetValue(id, out var path) ? path : throw new StudioException("conditioning_not_found", "Conditioning asset is not available in this project.");

    internal static VideoGenerationParams Parameters(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct)
    {
        // Steps finished, from OnStep: a progress report's Step can name the step still running (a Wan heartbeat does).
        int completed = 0;
        void Report(VideoGenerationProgress p)
        {
            int done = Math.Min(Volatile.Read(ref completed), Math.Max(p.TotalSteps, 0));
            // The model's "done" ends its pipeline; the host still encodes the MP4 before Studio imports it.
            string stage = p.Phase == "done" ? "encoding" : p.Phase ?? "inference";
            progress.Report(new StudioProgress(stage, p.TotalSteps > 0 ? Math.Min(.95, .1 + .8 * done / p.TotalSteps) : 0, p.Detail) { Step = done, TotalSteps = p.TotalSteps });
        }
        return new()
        {
            Width = r.Width, Height = r.Height, Frames = r.FrameCount, Fps = r.Fps, Steps = r.Steps,
            CfgScale = r.CfgScale, Seed = r.Seed, NegativePrompt = r.NegativePrompt,
            ImagePath = Resolve(assets, r.ConditioningAssetId), EndImagePath = Resolve(assets, r.EndConditioningAssetId),
            // Studio's mode names are the native mode names for every family that reads one; Wan infers from the image.
            Mode = r.Mode ?? StudioGenerationModes.Infer(r),
            ReferenceImagePaths = r.ReferenceAssetIds.Select(id => Resolve(assets, id)).ToList(),
            ImageConditions = r.TemporalConditions.Select(c => new VideoImageCondition { ImagePath = Resolve(assets, c.AssetId), FrameIndex = c.FrameIndex, Strength = c.Strength }).ToList(),
            GenerateAudio = r.GenerateAudio ?? true, CancellationToken = ct,
            // Every video pipeline follows a finished step with a progress report, which carries the new count.
            OnStep = (step, total) => Volatile.Write(ref completed, step),
            OnProgress = p =>
            {
                if (progress == null) return;
                if (!p.Heartbeat) { Report(p); return; }
                // A heartbeat comes from a timer thread mid-pass, where an exception would take the process down. A full
                // disk is raised by the next report from the generating thread instead.
                try { Report(p); } catch (Exception) { }
            }
        };
    }
}
