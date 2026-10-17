using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;

namespace TensorSharp.Models.Ltx25;

internal static class Ltx25Pipeline
{
    internal static GeneratedVideo Generate(Ltx25Model model, string prompt, VideoGenerationParams parameters)
        => Generate(model, prompt, parameters, null);

    // Diagnostic callers can observe the final normalized [128,T,H,W] video
    // latents and token-major audio before either decoder. Buffers are borrowed
    // and must not be mutated. The ordinary generation path has no observer.
    internal static GeneratedVideo Generate(Ltx25Model model, string prompt, VideoGenerationParams parameters,
        Action<Ltx25GenerationSettings, float[], float[]> finalLatents)
        => Generate(model, prompt, parameters, finalLatents, null);

    // Diagnostic-only override for the first pass's frame-zero guide. The
    // refinement pass continues to use the request's explicit strength.
    internal static GeneratedVideo Generate(Ltx25Model model, string prompt, VideoGenerationParams parameters,
        Action<Ltx25GenerationSettings, float[], float[]> finalLatents, float? firstStageImageStrength)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (firstStageImageStrength is float strength && (!float.IsFinite(strength) || strength < 0 || strength > 1))
            throw new ArgumentOutOfRangeException(nameof(firstStageImageStrength), "Image strength must be finite and in [0,1].");
        var settings = Ltx25GenerationSettings.Resolve(parameters, model.SupportsReferenceConditioning);
        var timer = Stopwatch.StartNew();
        var phase = Stopwatch.StartNew();
        int completed = 0, totalSteps = settings.TotalSteps;
        void Report(string name, string detail)
        {
            parameters.CancellationToken.ThrowIfCancellationRequested();
            Console.WriteLine($"  [ltx25] {name}: {detail}");
            parameters.OnProgress?.Invoke(new VideoGenerationProgress
            {
                Phase = name, Detail = detail, Step = completed, TotalSteps = totalSteps,
                ElapsedSeconds = timer.Elapsed.TotalSeconds,
            });
        }
        void EndPhase(string name)
        {
            Console.WriteLine($"  [ltx25-timing] {name}: {phase.Elapsed.TotalMilliseconds:F3}ms");
            phase.Restart();
        }
        void Trim()
        {
            if (model.DirectAllocator is CudaAllocator cuda) { cuda.Synchronize(); cuda.TrimPool(); }
            else if (model.DirectAllocator is MlxAllocator) MlxBackend.ClearCache();
        }
        try
        {
            bool identity = parameters.Mode == "ref";
            bool twoStage = settings.TwoStage;
            var first = twoStage ? settings.HalfResolution() : settings;
            var images = identity ? Array.Empty<VideoImageCondition>() : Ltx25ImageConditioning.Resolve(parameters, settings.Frames);
            float? effectiveFirstStageStrength = ResolveFirstStageImageStrength(parameters, twoStage, images, firstStageImageStrength);
            // Recompress each original once; both stages resize the same conditioned
            // source independently, as the publisher's image loader does.
            var resolvedImages = images.Select(image => Ltx25ImagePreprocessing.PrepareSource(
                image.ResolveImage(), parameters.CancellationToken)).ToArray();
            float[] encodedReference = null;
            int identitySourceCount = 0;
            // Progress phases follow the hosts' video contract (text-encode, denoise,
            // vae-decode, ...): everything between the two passes reports as denoise, so
            // the step count and ETA keep advancing. Timing lines keep the finer names.
            List<Ltx25EncodedImage> EncodeImages(Ltx25GenerationSettings stage, string progressPhase, string timingPhase)
            {
                var encoded = new List<Ltx25EncodedImage>(images.Count);
                if (images.Count == 0) return encoded;
                Report(progressPhase, $"encoding {images.Count} temporal image guides at {stage.Width}x{stage.Height}");
                for (int i = 0; i < images.Count; ++i)
                {
                    parameters.CancellationToken.ThrowIfCancellationRequested();
                    RgbImage resized = Ltx25ImagePreprocessing.ResizeAndCenterCrop(resolvedImages[i], stage.Width, stage.Height);
                    encoded.Add(new(model.EncodeImage(resized.Pixels, stage.Height, stage.Width, parameters.CancellationToken),
                        images[i].FrameIndex, images[i].Strength));
                }
                Trim();
                EndPhase(timingPhase);
                return encoded;
            }
            List<Ltx25EncodedImage> encodedImages = null;
            if (identity)
            {
                var sources = Ltx25ReferenceConditioning.ResolveSources(parameters);
                identitySourceCount = sources.Count;
                Report("reference-encode", $"encoding {sources.Count} identity images as a {settings.Frames}-frame Ingredients reference sheet");
                RgbImage sheet = Ltx25ReferenceConditioning.BuildSheet(sources.Select(source => source.ResolveImage(preserveAlpha: true)).ToArray(), settings.Width, settings.Height);
                encodedReference = model.EncodeStaticReferenceVideo(sheet.Pixels, settings.Frames, settings.Height, settings.Width, parameters.CancellationToken);
                Trim(); EndPhase("reference-encode");
            }
            else
            {
                encodedImages = EncodeImages(first, "image-encode", "image-encode");
                if (effectiveFirstStageStrength.HasValue)
                    encodedImages = WithFirstStageImageStrength(encodedImages, effectiveFirstStageStrength.Value);
            }
            Report("text-encode", "Gemma 4 and learned AV conditioning");
            Ltx25PromptEncoding conditioning = model.Conditioning(identity
                ? Ltx25ReferencePrompt.Format(prompt, identitySourceCount) : prompt, parameters.CancellationToken);
            Trim();
            EndPhase("text-encode");

            // One denoising pass. The first draws all-noise latents; the refinement
            // pass mixes the upsampled latent and stage-one audio with fresh noise.
            Ltx25SamplingOutput RunStage(Ltx25GenerationSettings stage, List<Ltx25EncodedImage> stageImages, float[] sigmas,
                long noiseSeedOffset, int stepOffset, string phaseName, float[] priorVideo, float[] priorAudio)
            {
                int guideTokens = identity ? stage.VideoTokens : checked(stageImages.Count(image => image.FrameIndex != 0) * stage.LatentHeight * stage.LatentWidth);
                var noise = stage.InitialNoise(guideTokens, priorVideo == null ? 0UL : 1UL);
                float[] video = noise.Video, audio = noise.Audio, rawNoise = null;
                float noiseScale = 1;
                if (priorVideo != null)
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
                    VideoContext = conditioning.VideoContext, VideoContextTokens = conditioning.Tokens,
                    AudioContext = conditioning.AudioContext, AudioContextTokens = conditioning.Tokens,
                    VideoPositions = Ltx25Geometry.VideoPositions(stage.LatentFrames, stage.LatentHeight, stage.LatentWidth, stage.Fps),
                    AudioPositions = Ltx25Geometry.AudioPositions(stage.AudioTokens),
                    VideoSigma = 1, AudioSigma = 1, FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
                    KeyframeMask = Enumerable.Range(0, checked(stage.VideoTokens + guideTokens))
                        .Select(token => token < stage.LatentHeight * stage.LatentWidth ? 1f : 0f).ToArray(),
                };
                input = identity ? Ltx25ReferenceConditioning.Apply(input, stage, encodedReference,
                    stage.LatentFrames, stage.LatentHeight, stage.LatentWidth)
                    : Ltx25ImageConditioning.Apply(input, stage, stageImages, rawNoise, noiseScale);
                Report("denoise", $"{(priorVideo == null ? twoStage ? "first pass" : "single pass" : "refinement pass")} at {stage.Width}x{stage.Height}: {stage.VideoTokens} video + {guideTokens} guide, {stage.AudioTokens} audio, {conditioning.Tokens} prompt tokens, " +
                    (stage.Ancestral ? Ltx25GenerationSettings.AncestralSampler
                        : $"{Ltx25GenerationSettings.EulerSampler} (requested; the publisher samples LTX 2.5 with {Ltx25GenerationSettings.AncestralSampler})") +
                    (identity || stageImages.Count == 0 ? "" : "; temporal guide strengths: " + string.Join(", ", stageImages.Select(image =>
                        $"frame {image.FrameIndex}={image.Strength.ToString("R", CultureInfo.InvariantCulture)}"))));
                Ltx25SamplingOutput result = model.Sample(input, stage, (step, total) =>
                    {
                        completed = stepOffset + step;
                        parameters.OnStep?.Invoke(completed, totalSteps);
                        Report("denoise", $"step {completed}/{totalSteps}");
                    }, parameters.CancellationToken, sigmas, noiseSeedOffset);
                if (result.BlocksPerStep != 48) throw new InvalidOperationException("LTX generation did not execute all transformer blocks.");
                Trim();
                EndPhase(phaseName);
                return result;
            }

            Ltx25SamplingOutput latents = RunStage(first, encodedImages, Ltx25Scheduler.DistilledSigmas(),
                Ltx25GenerationSettings.AncestralNoiseSeedOffset, 0, "denoise", null, null);
            if (twoStage)
            {
                Report("denoise", $"x2 latent upscale to {settings.Width}x{settings.Height}");
                float[] firstLatents = Ltx25Geometry.UnpackVideo(latents.Video.AsSpan(0, checked(first.VideoTokens * 128)), first.VideoTokens, 128);
                float[] upsampled = model.UpsampleLatents(firstLatents, first.LatentFrames, first.LatentHeight, first.LatentWidth, parameters.CancellationToken);
                Trim();
                EndPhase("upsample");
                var refinementImages = EncodeImages(settings, "denoise", "refine-image-encode");
                latents = RunStage(settings, refinementImages, Ltx25Scheduler.RefinementSigmas(),
                    Ltx25GenerationSettings.AncestralStage2NoiseSeedOffset, Ltx25GenerationSettings.Steps, "refine",
                    Ltx25Geometry.PackVideo(upsampled, settings.VideoTokens, 128), latents.Audio);
            }

            float[] videoLatents = Ltx25Geometry.UnpackVideo(latents.Video.AsSpan(0, checked(settings.VideoTokens * 128)), settings.VideoTokens, 128);
            finalLatents?.Invoke(settings, videoLatents, latents.Audio);
            Report("vae-decode", model.UsesDiffusionDecoder ? "diffusion video decoder" : "convolutional video decoder");
            float[] pixels = model.DecodeVideo(videoLatents,
                settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, settings.Seed, parameters.CancellationToken);
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
            EndPhase("vae-decode");

            GeneratedVideoAudio audio = null;
            if (parameters.GenerateAudio)
            {
                Report("audio-decode", "audio VAE, vocoder and 48 kHz bandwidth extension");
                audio = model.DecodeAudio(latents.Audio, settings.AudioTokens,
                    checked((int)Math.Round(settings.Frames / (double)settings.Fps * 48000, MidpointRounding.ToEven)),
                    parameters.CancellationToken);
                Trim();
                EndPhase("audio-decode");
            }
            Report("done", $"{settings.Width}x{settings.Height}, {settings.Frames} frames at {settings.Fps} fps");
            return new GeneratedVideo { Frames = frames, Fps = settings.Fps, Seed = settings.Seed, Audio = audio };
        }
        finally { Trim(); }
    }

    // The publisher's two-stage I2V workflow uses .7 at half resolution and 1
    // during refinement. Only shorthand first-image requests select this default:
    // explicit guides, ordered/end images and other modes retain their strengths.
    // A null result means that the first pass uses the resolved request strengths.
    internal static float? ResolveFirstStageImageStrength(VideoGenerationParams parameters, bool twoStage,
        IReadOnlyList<VideoImageCondition> images, float? diagnosticOverride = null)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(images);
        if (diagnosticOverride is float strength)
        {
            if (!float.IsFinite(strength) || strength < 0 || strength > 1)
                throw new ArgumentOutOfRangeException(nameof(diagnosticOverride), "Image strength must be finite and in [0,1].");
            if (!twoStage || !images.Any(image => image.FrameIndex == 0))
                throw new ArgumentException("A first-stage image strength requires two-stage generation with a frame-zero image guide.", nameof(diagnosticOverride));
            return strength;
        }
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
}
