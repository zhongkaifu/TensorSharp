// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LTX 2.5 Distilled on the diffusion runtime's video pipeline (DiffusionVideoPipeline): the family's hooks. A clip follows the
// publisher's two-stage DistilledPipeline: eight ancestral Euler steps at half the output size, the x2 latent upscaler, then
// three refinement steps at full size from the upscaled latent mixed with fresh noise, each pass re-applying the image guides
// encoded at its own size (an Ingredients identity request keeps one full-size pass with its reference sheet appended). The
// joint audio stream is stepped beside the video on the same sigmas. Each pass draws its own Philox noise, keyed by the seed
// and the publisher's offsets, for its initial latent (stream 0, then 1) and for every ancestral step (seed + 10000, then +
// 20000).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Models.Video;
using TensorSharp.Runtime.Scheduling.Diffusion;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Models.Ltx25;

/// <summary>An LTX clip as the family resolved it, and what its phases produce for the next.</summary>
internal sealed class Ltx25Job : DiffusionVideoJob
{
    public required VideoGenerationParams Params { get; init; }
    public required Ltx25GenerationSettings Settings { get; init; }
    /// <summary>The first pass's settings: half the output size for two stages, else the output's.</summary>
    public required Ltx25GenerationSettings First { get; init; }
    public required bool Identity { get; init; }
    public required string ConditioningPrompt { get; init; }
    public required IReadOnlyList<VideoImageCondition> Images { get; init; }
    public float? FirstStageStrength { get; init; }
    public int IdentityImages { get; init; }
    public int IdentityVideos { get; init; }

    // ---- Condition
    public int PromptTokens;

    // ---- EncodeVideo
    public RgbImage[] Sources;   // each guide recompressed once (CRF 18); both passes resize it to their size
    public List<Ltx25ReferenceConditioning.ReferenceLatent> References = new();
    public List<Ltx25EncodedImage> FirstImages = new();

    // ---- Denoise: the pass's conditioned template (masks, clean latents, positions) and its step-noise seed offset
    public Ltx25DenoiserInput Pass;
    public long NoiseSeedOffset;
    public Ltx25GenerationSettings PassSettings;
    public bool FinalPass;

    public Action<string, string, string, IReadOnlyDictionary<string, object>> Report;
}

/// <summary>The LTX 2.5 family: the hooks <see cref="DiffusionVideoPipeline"/> runs a clip through (see the file header).</summary>
internal sealed class Ltx25Family : IVideoDiffusionFamily
{
    internal const string FirstPass = "first-pass", SinglePass = "single-pass", Refinement = "refinement";
    private const int Channels = 128;
    private readonly Ltx25Model _model;

    public Ltx25Family(Ltx25Model model) => _model = model ?? throw new ArgumentNullException(nameof(model));

    public string Id => Ltx25Model.ArchitectureId;
    public string DisplayName => "LTX 2.5";

    // ---- components

    public IConditioner CreateConditioner(DiffusionLoad load) =>
        new Ltx25Conditioner(load.File(DiffusionLoad.TextEncoder), load.File(DiffusionLoad.Denoiser), _model.DirectAllocator);

    public IDenoiser CreateDenoiser(DiffusionLoad load, string file) =>
        new Ltx25Denoiser(file, _model.DirectAllocator) { AdapterBytes = () => _model.Ingredients?.FactorBytes ?? 0 };

    public IResidentComponent CreateVideoCodec(DiffusionLoad load) =>
        new Ltx25VideoCodec(load.File(DiffusionLoad.Vae), _model.DiffusionVaePath, _model.LatentUpscalerPath, _model.DirectAllocator);

    public IResidentComponent CreateAudioCodec(DiffusionLoad load) => new Ltx25AudioCodec(_model.AudioVaePath, _model.DirectAllocator);

    // ---- Prepare

    public DiffusionVideoJob Prepare(DiffusionVideoRequest request, DiffusionVideoComponents components)
    {
        var p = (request.FamilyRequest as VideoGenerationParams ?? new VideoGenerationParams()).ResolveVideoContinuations(ltxReferenceVideos: true);
        var settings = Ltx25GenerationSettings.Resolve(p, _model.SupportsReferenceConditioning);
        bool identity = p.Mode == "ref";
        bool twoStage = settings.TwoStage;
        var first = twoStage ? settings.HalfResolution() : settings;
        // Identity references may start on a pinned first frame (a chain's previous segment); no other guide joins them.
        IReadOnlyList<VideoImageCondition> images = identity
            ? Ltx25ReferenceConditioning.ResolveFirstFrameGuide(p) is { } firstFrame ? new[] { firstFrame } : Array.Empty<VideoImageCondition>()
            : Ltx25ImageConditioning.Resolve(p, settings.Frames);
        float? firstStrength = ResolveFirstStageImageStrength(p, twoStage, images);
        // The adapter's factors are read while the clip conditions and encodes (its first step would wait for them otherwise).
        if (identity) _model.Ingredients?.Preload();
        int identityImages = identity ? Ltx25ReferenceConditioning.ResolveSources(p).Count : 0;
        int identityVideos = identity ? p.ReferenceVideoPaths?.Count ?? 0 : 0;
        // Format cannot throw here: the reference counts it checks were refused above.
        string prompt = request.Prompt ?? string.Empty;
        string conditioningPrompt = identity ? Ltx25ReferencePrompt.Format(prompt, identityImages, identityVideos) : prompt;

        var stages = new List<DiffusionVideoStage> { new(twoStage ? FirstPass : SinglePass, Ltx25Scheduler.DistilledSigmas()) };
        if (twoStage) stages.Add(new DiffusionVideoStage(Refinement, Ltx25Scheduler.RefinementSigmas()));
        int steps = stages.Sum(s => s.Steps);

        // Residency: the denoiser's widest pass (its guides or references appended), and the codec's decode at full size.
        long Tokens(Ltx25GenerationSettings s) => s.VideoTokens +
            (identity ? s.VideoTokens * (long)(identityImages > 0 ? 1 : 0) + identityVideos * (long)s.VideoTokens
                      : images.Count(i => i.FrameIndex != 0) * (long)(s.LatentHeight * s.LatentWidth));
        long widest = Math.Max(Tokens(first), Tokens(settings));
        var target = new LatentShape(Channels, settings.LatentFrames, settings.LatentHeight, settings.LatentWidth);
        var workload = new DiffusionWorkload(target, Array.Empty<LatentShape>(), conditioningPrompt);
        IResidentComponent codec = components.VideoCodec;
        long denoiseBytes = (_model.LoadedDenoiser as Ltx25Denoiser)?.WorkingBytesFor(widest) ?? widest * (512L << 10) + (1L << 30);
        var demands = new List<PhaseDemand>
        {
            new(DiffusionPhase.Condition, DiffusionComponent.Conditioner, components.Conditioner.WorkingBytes(workload)),
        };
        if (identity || images.Count > 0)
            demands.Add(new(DiffusionPhase.EncodeReferences, DiffusionComponent.Vae, codec.WorkingBytes(workload)));
        demands.Add(new(DiffusionPhase.Denoise, DiffusionComponent.Denoiser, denoiseBytes));
        demands.Add(new(DiffusionPhase.Decode, DiffusionComponent.Vae, codec.WorkingBytes(workload)));
        if (request.DecodeAudio && components.AudioCodec is { } audioCodec)
            demands.Add(new(DiffusionPhase.DecodeAudio, DiffusionComponent.AudioVae, audioCodec.WorkingBytes(workload)));

        return new Ltx25Job
        {
            Recipe = new SamplingRecipe
            {
                Schedule = DiffusionSchedule.Explicit,
                Steps = stages[0].Steps,
                Sigmas = stages[0].Sigmas,
                Sampler = settings.Ancestral ? DiffusionSamplerKind.EulerAncestral : DiffusionSamplerKind.EulerDenoised,
                Owner = "LTX 2.5 Distilled",
            },
            Steps = steps,
            Sigmas = stages[0].Sigmas,
            Stages = stages,
            DenoiserFile = _model.CheckpointPath,
            Demands = demands,
            GeometryClass = $"{settings.Width}x{settings.Height}x{settings.Frames}",
            Describe = $"{(identity ? "ref" : images.Count > 0 ? "guided" : "t2v")}, {steps} steps, seed {settings.Seed}",
            Params = p,
            Settings = settings,
            First = first,
            Identity = identity,
            ConditioningPrompt = conditioningPrompt,
            Images = images,
            FirstStageStrength = firstStrength,
            IdentityImages = identityImages,
            IdentityVideos = identityVideos,
            Report = (phase, detail, stage, attributes) => _model.Report(p, phase, detail, stage, attributes),
        };
    }

    // The publisher's two-stage I2V workflow uses .7 at half resolution and 1
    // during refinement. Only shorthand first-image requests select this default:
    // explicit guides, ordered/end images and other modes retain their strengths.
    // A null result means that the first pass uses the resolved request strengths.
    internal static float? ResolveFirstStageImageStrength(VideoGenerationParams parameters, bool twoStage,
        IReadOnlyList<VideoImageCondition> images)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(images);
        bool shorthand = parameters.Image != null || parameters.ImageBytes is { Length: > 0 } || !string.IsNullOrWhiteSpace(parameters.ImagePath);
        bool firstImageMode = string.IsNullOrWhiteSpace(parameters.Mode) || parameters.Mode == "i2v";
        return twoStage && shorthand && firstImageMode && images.Count == 1 && images[0].FrameIndex == 0 &&
            parameters.EndImage == null && string.IsNullOrWhiteSpace(parameters.EndImagePath) &&
            parameters.ReferenceImagePaths is not { Count: > 0 } && parameters.ImageConditions is not { Count: > 0 }
            ? .7f : null;
    }

    internal static List<Ltx25EncodedImage> WithFirstStageImageStrength(IReadOnlyList<Ltx25EncodedImage> images, float strength)
    {
        if (!float.IsFinite(strength) || strength < 0 || strength > 1)
            throw new ArgumentOutOfRangeException(nameof(strength), "Image strength must be finite and in [0,1].");
        if (!images.Any(image => image.FrameIndex == 0))
            throw new ArgumentException("A first-stage strength requires a frame-zero image guide.", nameof(images));
        return images.Select(image => image.FrameIndex == 0 ? image with { Strength = strength } : image).ToList();
    }

    // ---- Condition

    public ConditioningSet Condition(DiffusionVideoJob job, IConditioner conditioner, CancellationToken ct)
    {
        var j = (Ltx25Job)job;
        var c = (Ltx25Conditioner)conditioner;
        var clock = Stopwatch.StartNew();
        // A repeated prompt comes from the model's small prompt cache and makes this phase
        // almost instant, which would otherwise look like an anomaly in a trace.
        j.Report("text-encode", "Gemma 4 and learned AV conditioning", null, j.Params.OnProgress == null ? null
            : new Dictionary<string, object> { [TelemetryAttributes.VideoPromptCacheHit] = c.HasCached(j.ConditioningPrompt) });
        ConditioningSet set = c.Encode(j.ConditioningPrompt, Array.Empty<PresentedImage>(), ct);
        j.PromptTokens = set.Rows;
        Trim();
        Timing("text-encode", clock);
        return set;
    }

    // ---- EncodeRefs

    public void EncodeVideo(DiffusionVideoJob job, IResidentComponent videoCodec, Action<DiffusionVideoProgress> report, CancellationToken ct)
    {
        var j = (Ltx25Job)job;
        var codec = (Ltx25VideoCodec)videoCodec;
        var clock = Stopwatch.StartNew();
        if (j.Identity)
        {
            var sources = Ltx25ReferenceConditioning.ResolveSources(j.Params);
            Ltx25GenerationSettings settings = j.Settings;
            if (sources.Count > 0)
            {
                j.Report("reference-encode", $"encoding {sources.Count} identity images as a {settings.Frames}-frame Ingredients reference sheet", null, null);
                RgbImage sheet = Ltx25ReferenceConditioning.BuildSheet(sources.Select(source => source.ResolveImage(preserveAlpha: true)).ToArray(),
                    settings.Width, settings.Height);
                float[] encodedSheet = codec.EncodeStaticReferenceVideo(sheet.Pixels, settings.Frames, settings.Height, settings.Width, ct);
                j.References.Add(new(encodedSheet, settings.LatentFrames, settings.LatentHeight, settings.LatentWidth));
            }
            if (j.Params.ReferenceVideoPaths is { Count: > 0 } referenceVideos)
                for (int i = 0; i < referenceVideos.Count; ++i)
                    EncodeReferenceVideo(j, codec, referenceVideos[i], i, referenceVideos.Count, ct);
            Trim();
            Timing("reference-encode", clock);
            // One full-size pass, so the first-frame guide is encoded once, at the output size.
            j.FirstImages = EncodeImages(j, codec, settings, "image-encode", null, "image-encode", ct);
        }
        else
        {
            j.FirstImages = EncodeImages(j, codec, j.First, "image-encode", null, "image-encode", ct);
            if (j.FirstStageStrength is float strength)
                j.FirstImages = WithFirstStageImageStrength(j.FirstImages, strength);
        }
    }

    private void EncodeReferenceVideo(Ltx25Job j, Ltx25VideoCodec codec, string path, int index, int count, CancellationToken ct)
    {
        Ltx25GenerationSettings settings = j.Settings;
        j.Report("reference-encode", $"encoding reference video {index + 1}/{count} at {settings.Width}x{settings.Height}", null, null);
        var (files, sourceFps) = MediaHelper.ExtractFramesAtRate(path, settings.Fps, maxFrames: settings.Frames);
        string extracted = File.Exists(path) && files.Count > 0 ? Path.GetDirectoryName(files[0]) : null;
        if (extracted != null && !Path.GetFileName(extracted).StartsWith("refvid_", StringComparison.Ordinal)) extracted = null;
        try
        {
            if (files.Count == 0) throw new ArgumentException($"Reference video '{path}' contains no decodable frames.");
            // The causal VAE needs 8n+1 frames and the Ingredients adapter reserves at least 16 latent
            // frames. Hold the last decoded frame when a short reference does not fill that window.
            int referenceFrames = Math.Max(settings.Frames, 1 + ((files.Count - 1 + 7) / 8) * 8);
            var videoPixels = new float[checked(referenceFrames * settings.Height * settings.Width * 3)];
            RgbImage last = null;
            for (int frame = 0; frame < referenceFrames; ++frame)
            {
                ct.ThrowIfCancellationRequested();
                if (frame < files.Count)
                {
                    RgbImage decoded = ImageIO.Load(files[frame]);
                    last = Ltx25ImagePreprocessing.ResizeAndCenterCrop(decoded, settings.Width, settings.Height);
                }
                int offset = checked(frame * settings.Height * settings.Width * 3);
                last.Pixels.CopyTo(videoPixels, offset);
            }
            float[] encoded = codec.EncodeReferenceVideo(videoPixels, referenceFrames, settings.Height, settings.Width, ct);
            int latentFrames = (referenceFrames - 1) / 8 + 1;
            j.References.Add(new(encoded, latentFrames, settings.LatentHeight, settings.LatentWidth));
            Console.WriteLine($"  [ltx25] reference video: {files.Count} frames at {settings.Fps} fps (source {sourceFps:F2} fps) -> {referenceFrames} frames");
        }
        finally
        {
            if (extracted != null)
            {
                try { Directory.Delete(extracted, recursive: true); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }

    /// <summary>The temporal guides encoded at <paramref name="stage"/>'s size: each source recompressed once (CRF 18), then
    /// resized and centre-cropped for the pass, as the publisher's image loader does.</summary>
    private List<Ltx25EncodedImage> EncodeImages(Ltx25Job j, Ltx25VideoCodec codec, Ltx25GenerationSettings stage,
        string phase, string progressStage, string timingPhase, CancellationToken ct)
    {
        var encoded = new List<Ltx25EncodedImage>(j.Images.Count);
        if (j.Images.Count == 0) return encoded;
        var clock = Stopwatch.StartNew();
        j.Sources ??= j.Images.Select(image => Ltx25ImagePreprocessing.PrepareSource(image.ResolveImage(), ct)).ToArray();
        j.Report(phase, $"encoding {j.Images.Count} temporal image guides at {stage.Width}x{stage.Height}", progressStage, null);
        for (int i = 0; i < j.Images.Count; ++i)
        {
            ct.ThrowIfCancellationRequested();
            RgbImage resized = Ltx25ImagePreprocessing.ResizeAndCenterCrop(j.Sources[i], stage.Width, stage.Height);
            encoded.Add(new(codec.EncodeImage(resized.Pixels, stage.Height, stage.Width, ct), j.Images[i].FrameIndex, j.Images[i].Strength));
        }
        Trim();
        Timing(timingPhase, clock);
        return encoded;
    }

    // ---- Denoise

    public DiffusionStreams InitialStreams(DiffusionVideoJob job)
    {
        var j = (Ltx25Job)job;
        return BeginPass(j, j.First, j.FirstImages, Ltx25GenerationSettings.AncestralNoiseSeedOffset, prior: null);
    }

    /// <summary>
    /// A pass's conditioned initial state. The first draws all-noise latents (Philox stream 0); the refinement mixes the
    /// upsampled latent and the first pass's audio with fresh noise (stream 1) at its first sigma. Then the pass's guides: the
    /// identity references appended clean (and a pinned first frame), or the temporal image guides (frame 0 in place, later
    /// ones appended with their pixel-time positions).
    /// </summary>
    private DiffusionStreams BeginPass(Ltx25Job j, Ltx25GenerationSettings stage, List<Ltx25EncodedImage> images, long seedOffset,
        (float[] Video, float[] Audio)? prior)
    {
        Ltx25GenerationSettings settings = j.Settings;
        bool twoStage = settings.TwoStage;
        float[] sigmas = prior == null ? Ltx25Scheduler.DistilledSigmas() : Ltx25Scheduler.RefinementSigmas();
        int guideTokens = j.Identity ? checked(j.References.Sum(reference => reference.Frames * reference.Height * reference.Width))
            : checked(images.Count(image => image.FrameIndex != 0) * stage.LatentHeight * stage.LatentWidth);
        var noise = stage.InitialNoise(guideTokens, prior == null ? 0UL : 1UL);
        float[] video = noise.Video, audio = noise.Audio, rawNoise = null;
        float noiseScale = 1;
        if (prior is var (priorVideo, priorAudio))
        {
            noiseScale = sigmas[0];
            rawNoise = noise.Video;
            video = (float[])noise.Video.Clone();
            for (int i = 0; i < priorVideo.Length; ++i) video[i] = priorVideo[i] + (rawNoise[i] - priorVideo[i]) * noiseScale;
            audio = new float[priorAudio.Length];
            for (int i = 0; i < audio.Length; ++i) audio[i] = priorAudio[i] + (noise.Audio[i] - priorAudio[i]) * noiseScale;
        }
        var input = new Ltx25DenoiserInput
        {
            Video = video, VideoTokens = checked(stage.VideoTokens + guideTokens),
            Audio = audio, AudioTokens = stage.AudioTokens,
            // The contexts arrive with each step's conditioning.
            VideoContext = Array.Empty<float>(), VideoContextTokens = 0, AudioContext = Array.Empty<float>(), AudioContextTokens = 0,
            VideoPositions = Ltx25Geometry.VideoPositions(stage.LatentFrames, stage.LatentHeight, stage.LatentWidth, stage.Fps),
            AudioPositions = Ltx25Geometry.AudioPositions(stage.AudioTokens),
            VideoSigma = 1, AudioSigma = 1, FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
            KeyframeMask = Enumerable.Range(0, checked(stage.VideoTokens + guideTokens))
                .Select(token => token < stage.LatentHeight * stage.LatentWidth ? 1f : 0f).ToArray(),
        };
        input = j.Identity ? Ltx25ReferenceConditioning.Apply(input, stage, j.References)
            : Ltx25ImageConditioning.Apply(input, stage, images, rawNoise, noiseScale);
        // The guide and the references occupy disjoint tokens (canvas frame 0, the appended tail), so pinning frame 0
        // after the references equals upstream's images-then-references order.
        if (j.Identity && images.Count > 0) input = Ltx25ImageConditioning.ApplyFirstFrame(input, stage, images[0]);
        j.Pass = input;
        j.PassSettings = stage;
        j.NoiseSeedOffset = seedOffset;
        j.FinalPass = !twoStage || prior != null;
        string pass = prior == null ? twoStage ? "first pass" : "single pass" : "refinement pass";
        string progressStage = prior == null ? twoStage ? FirstPass : SinglePass : Refinement;
        j.Report("denoise", $"{pass} at {stage.Width}x{stage.Height}: {stage.VideoTokens} video + {guideTokens} guide, {stage.AudioTokens} audio, " +
            $"{j.PromptTokens} prompt tokens, " +
            (stage.Ancestral ? Ltx25GenerationSettings.AncestralSampler
                : $"{Ltx25GenerationSettings.EulerSampler} (requested; the publisher samples LTX 2.5 with {Ltx25GenerationSettings.AncestralSampler})") +
            (images.Count == 0 ? "" : (j.Identity ? "; first-frame guide strength: " : "; temporal guide strengths: ") + string.Join(", ", images.Select(image =>
                $"frame {image.FrameIndex}={image.Strength.ToString("R", CultureInfo.InvariantCulture)}"))), progressStage, null);
        return new DiffusionStreams(new Dictionary<string, float[]>
        {
            [DiffusionStreams.Primary] = (float[])input.Video.Clone(),
            [Ltx25Denoiser.AudioStream] = (float[])input.Audio.Clone(),
        });
    }

    /// <summary>A step's denoiser input: the pass's layout (<paramref name="pass"/>: tokens, positions, guide mask, keyframe
    /// markers) over the current latents at <paramref name="sigma"/>, each video token's timestep its denoise weight times sigma
    /// (a guide conditioned at strength s denoises with weight 1 - s), the audio stream's the global sigma.</summary>
    internal static Ltx25DenoiserInput StepInput(Ltx25DenoiserInput pass, float[] video, float[] audio, float sigma,
        float[] videoContext, float[] audioContext, int contextTokens, bool referenceLora)
    {
        float[] mask = pass.VideoDenoiseMask;
        return new Ltx25DenoiserInput
        {
            Video = video, VideoTokens = pass.VideoTokens,
            Audio = audio, AudioTokens = pass.AudioTokens,
            VideoContext = videoContext, VideoContextTokens = contextTokens,
            AudioContext = audioContext, AudioContextTokens = contextTokens,
            VideoPositions = pass.VideoPositions, AudioPositions = pass.AudioPositions,
            FrequencyPrecision = pass.FrequencyPrecision,
            VideoSigma = sigma, AudioSigma = sigma,
            VideoTimesteps = mask == null ? null : Array.ConvertAll(mask, value => value * sigma),
            KeyframeMask = pass.KeyframeMask,
            UseReferenceLora = referenceLora,
        };
    }

    public DenoiseStepInputs StepInputs(DiffusionVideoJob job, ConditioningSet conditioning, DiffusionStreams state, StepState step)
    {
        var j = (Ltx25Job)job;
        var encoding = (Ltx25PromptEncoding)conditioning.FamilyState;
        Ltx25DenoiserInput input = StepInput(j.Pass, state[DiffusionStreams.Primary], state[Ltx25Denoiser.AudioStream], step.Sigma,
            encoding.VideoContext, encoding.AudioContext, encoding.Tokens, j.Identity);
        return new DenoiseStepInputs
        {
            Latent = input.Video,
            Conditioning = conditioning,
            Layout = new DiffusionLayout { Rows = input.VideoTokens, TargetStart = 0, TargetRows = input.VideoTokens, FamilyLayout = input },
            Step = step,
            Recipe = j.Recipe,
            Streams = new Dictionary<string, float[]> { [Ltx25Denoiser.AudioStream] = input.Audio },
        };
    }

    public DiffusionStreams SamplerVelocities(DiffusionVideoJob job, DiffusionStreams state, DenoiseStepInputs inputs, StepOutput output,
        StepState step)
    {
        var j = (Ltx25Job)job;
        // One stage ahead of each companion's first use: the upscaler during the last first-pass step, the video decoder
        // during the final pass, the audio decoder during the video decode.
        int completed = step.Index + 1;
        if (!j.FinalPass && completed == step.Count - 1) _model.PrefetchCompanion(Ltx25Model.CompanionFile.Upscaler);
        if (j.FinalPass && completed == Math.Min(1, step.Count - 1)) _model.PrefetchCompanion(Ltx25Model.CompanionFile.VideoDecoder);
        return new DiffusionStreams(new Dictionary<string, float[]>
        {
            [DiffusionStreams.Primary] = output.Velocity,
            [Ltx25Denoiser.AudioStream] = output.Streams[Ltx25Denoiser.AudioStream],
        });
    }

    /// <summary>The step's ancestral noise (one Philox draw a step, video tokens then audio: the pass's settings'
    /// <see cref="Ltx25GenerationSettings.AncestralNoise"/>) and the video stream's guide mask with its clean latents.</summary>
    public DiffusionSamplerInputs SamplerInputs(DiffusionVideoJob job, DiffusionStreams state, StepState step)
    {
        var j = (Ltx25Job)job;
        float[] video = state[DiffusionStreams.Primary], audio = state[Ltx25Denoiser.AudioStream];
        float[] videoNoise = null, audioNoise = null;
        if (j.PassSettings.Ancestral && step.Next != 0)
            (videoNoise, audioNoise) = j.PassSettings.AncestralNoise(step.Index, video.Length, audio.Length, j.NoiseSeedOffset);
        return new DiffusionSamplerInputs(new Dictionary<string, SamplerStreamInputs>
        {
            [DiffusionStreams.Primary] = new() { Noise = videoNoise, DenoiseMask = j.Pass.VideoDenoiseMask, Clean = j.Pass.CleanVideo },
            [Ltx25Denoiser.AudioStream] = new() { Noise = audioNoise },
        });
    }

    /// <summary>The refinement pass's state: the first pass's video latent upscaled x2 (the upscaler normalizes with the VAE's
    /// statistics), the guides encoded again at full size, then mixed with fresh noise at the refinement's first sigma.</summary>
    public DiffusionStreams NextStage(DiffusionVideoJob job, int stage, DiffusionStreams previous, DiffusionVideoComponents components,
        Action<DiffusionVideoProgress> report, CancellationToken ct)
    {
        var j = (Ltx25Job)job;
        var codec = (Ltx25VideoCodec)components.VideoCodec;
        Ltx25GenerationSettings first = j.First, settings = j.Settings;
        var clock = Stopwatch.StartNew();
        j.Report("denoise", $"x2 latent upscale to {settings.Width}x{settings.Height}", "upscale", null);
        float[] firstLatents = Ltx25Geometry.UnpackVideo(previous[DiffusionStreams.Primary].AsSpan(0, checked(first.VideoTokens * Channels)),
            first.VideoTokens, Channels);
        float[] upsampled = codec.Upsample(firstLatents, first.LatentFrames, first.LatentHeight, first.LatentWidth, ct);
        Trim();
        Timing("upsample", clock);
        List<Ltx25EncodedImage> images = EncodeImages(j, codec, settings, "denoise", "refine-image-encode", "refine-image-encode", ct);
        return BeginPass(j, settings, images, Ltx25GenerationSettings.AncestralStage2NoiseSeedOffset,
            (Ltx25Geometry.PackVideo(upsampled, settings.VideoTokens, Channels), previous[Ltx25Denoiser.AudioStream]));
    }

    // ---- Decode

    public RgbImage[] DecodeVideo(DiffusionVideoJob job, DiffusionStreams state, IResidentComponent videoCodec,
        Action<DiffusionVideoProgress> report, CancellationToken ct)
    {
        var j = (Ltx25Job)job;
        var codec = (Ltx25VideoCodec)videoCodec;
        Ltx25GenerationSettings settings = j.Settings;
        var clock = Stopwatch.StartNew();
        float[] videoLatents = Ltx25Geometry.UnpackVideo(state[DiffusionStreams.Primary].AsSpan(0, checked(settings.VideoTokens * Channels)),
            settings.VideoTokens, Channels);
        if (j.Params.GenerateAudio) _model.PrefetchCompanion(Ltx25Model.CompanionFile.AudioDecoder);
        j.Report("vae-decode", codec.UsesDiffusionDecoder ? "diffusion video decoder" : "convolutional video decoder", null, null);
        // The denoiser released its device copies when its phase ended; the pool's free buffers go back before the decode.
        Trim();
        float[] pixels = codec.Decode(videoLatents, settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, settings.Seed, ct);
        Trim();
        int frameSize = checked(settings.Width * settings.Height * 3);
        if (pixels.Length != checked(frameSize * settings.Frames)) throw new InvalidOperationException("LTX video decoder returned an unexpected frame geometry.");
        var frames = new RgbImage[settings.Frames];
        for (int frame = 0; frame < frames.Length; ++frame)
        {
            var rgb = new float[frameSize];
            Array.Copy(pixels, frame * frameSize, rgb, 0, frameSize);
            foreach (float value in rgb)
                if (!float.IsFinite(value) || value < 0 || value > 1) throw new InvalidOperationException("LTX video decoder returned invalid RGB pixels.");
            frames[frame] = new RgbImage(settings.Width, settings.Height, rgb);
        }
        Timing("vae-decode", clock);
        return frames;
    }

    public (float[][] Channels, int SampleRate) DecodeAudio(DiffusionVideoJob job, DiffusionStreams state, IResidentComponent audioCodec,
        CancellationToken ct)
    {
        var j = (Ltx25Job)job;
        Ltx25GenerationSettings settings = j.Settings;
        var clock = Stopwatch.StartNew();
        j.Report("audio-decode", "audio VAE, vocoder and 48 kHz bandwidth extension", null, null);
        GeneratedVideoAudio audio = ((Ltx25AudioCodec)audioCodec).Decode(state[Ltx25Denoiser.AudioStream], settings.AudioTokens,
            checked((int)Math.Round(settings.Frames / (double)settings.Fps * 48000, MidpointRounding.ToEven)), ct);
        Trim();
        Timing("audio-decode", clock);
        return (audio.Channels, audio.SampleRate);
    }

    public void EncodeAudio(DiffusionVideoJob job, IResidentComponent audioCodec, CancellationToken ct) =>
        throw new NotSupportedException("LTX audio-reference workflows are not implemented by this distilled preset.");

    // ---- helpers

    /// <summary>Return the allocator's free buffers to the device between phases, as the direct pipeline did (CUDA's pool
    /// trimmed after a sync, MLX's cache cleared).</summary>
    private void Trim()
    {
        if (_model.DirectAllocator is CudaAllocator cuda) { cuda.Synchronize(); cuda.TrimPool(); }
        else if (_model.DirectAllocator is MlxAllocator) MlxBackend.ClearCache();
    }

    private static void Timing(string phase, Stopwatch clock) =>
        Console.WriteLine($"  [ltx25-timing] {phase}: {clock.Elapsed.TotalMilliseconds:F3}ms");
}
