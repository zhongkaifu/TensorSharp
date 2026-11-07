using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime.Telemetry;

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
        parameters = (parameters ?? new VideoGenerationParams()).ResolveVideoContinuations(ltxReferenceVideos: true);
        if (firstStageImageStrength is float strength && (!float.IsFinite(strength) || strength < 0 || strength > 1))
            throw new ArgumentOutOfRangeException(nameof(firstStageImageStrength), "Image strength must be finite and in [0,1].");
        var settings = Ltx25GenerationSettings.Resolve(parameters, model.SupportsReferenceConditioning);
        var timer = Stopwatch.StartNew();
        var phase = Stopwatch.StartNew();
        int completed = 0, totalSteps = settings.TotalSteps;
        // stage names the part of the phase a report belongs to, for telemetry (the two passes
        // and what runs between them all report as denoise); attributes are facts about it.
        void Report(string name, string detail, string stage = null, IReadOnlyDictionary<string, object> attributes = null)
        {
            parameters.CancellationToken.ThrowIfCancellationRequested();
            Console.WriteLine($"  [ltx25] {name}: {detail}");
            parameters.OnProgress?.Invoke(new VideoGenerationProgress
            {
                Phase = name, Detail = detail, Step = completed, TotalSteps = totalSteps,
                ElapsedSeconds = timer.Elapsed.TotalSeconds,
                Stage = stage, Attributes = attributes,
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
            // Identity references may start on a pinned first frame (a chain's previous segment); no other guide joins them.
            IReadOnlyList<VideoImageCondition> images = identity
                ? Ltx25ReferenceConditioning.ResolveFirstFrameGuide(parameters) is { } firstFrame ? new[] { firstFrame } : Array.Empty<VideoImageCondition>()
                : Ltx25ImageConditioning.Resolve(parameters, settings.Frames);
            float? effectiveFirstStageStrength = ResolveFirstStageImageStrength(parameters, twoStage, images, firstStageImageStrength);
            // Recompress each original once; both stages resize the same conditioned
            // source independently, as the publisher's image loader does.
            var resolvedImages = images.Select(image => Ltx25ImagePreprocessing.PrepareSource(
                image.ResolveImage(), parameters.CancellationToken)).ToArray();
            var encodedReferences = new List<Ltx25ReferenceConditioning.ReferenceLatent>();
            int identityImageCount = 0, identityVideoCount = 0;
            // Progress phases follow the hosts' video contract (text-encode, denoise,
            // vae-decode, ...): everything between the two passes reports as denoise, so
            // the step count and ETA keep advancing. Timing lines keep the finer names.
            List<Ltx25EncodedImage> EncodeImages(Ltx25GenerationSettings stage, string progressPhase, string timingPhase,
                string progressStage = null)
            {
                var encoded = new List<Ltx25EncodedImage>(images.Count);
                if (images.Count == 0) return encoded;
                Report(progressPhase, $"encoding {images.Count} temporal image guides at {stage.Width}x{stage.Height}", progressStage);
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
                identityImageCount = sources.Count;
                if (sources.Count > 0)
                {
                    Report("reference-encode", $"encoding {sources.Count} identity images as a {settings.Frames}-frame Ingredients reference sheet");
                    RgbImage sheet = Ltx25ReferenceConditioning.BuildSheet(sources.Select(source => source.ResolveImage(preserveAlpha: true)).ToArray(), settings.Width, settings.Height);
                    float[] encodedSheet = model.EncodeStaticReferenceVideo(sheet.Pixels, settings.Frames, settings.Height, settings.Width, parameters.CancellationToken);
                    encodedReferences.Add(new(encodedSheet, settings.LatentFrames, settings.LatentHeight, settings.LatentWidth));
                }
                if (parameters.ReferenceVideoPaths is { Count: > 0 } referenceVideos)
                {
                    identityVideoCount = referenceVideos.Count;
                    for (int i = 0; i < referenceVideos.Count; ++i)
                    {
                        parameters.CancellationToken.ThrowIfCancellationRequested();
                        string path = referenceVideos[i];
                        Report("reference-encode", $"encoding reference video {i + 1}/{referenceVideos.Count} at {settings.Width}x{settings.Height}");
                        var (files, sourceFps) = TensorSharp.Models.MediaHelper.ExtractFramesAtRate(path, settings.Fps, maxFrames: settings.Frames);
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
                                parameters.CancellationToken.ThrowIfCancellationRequested();
                                if (frame < files.Count)
                                {
                                    RgbImage decoded = TensorSharp.Models.QwenImage.ImageIO.Load(files[frame]);
                                    last = Ltx25ImagePreprocessing.ResizeAndCenterCrop(decoded, settings.Width, settings.Height);
                                }
                                int offset = checked(frame * settings.Height * settings.Width * 3);
                                last.Pixels.CopyTo(videoPixels, offset);
                            }
                            float[] encoded = model.EncodeReferenceVideo(videoPixels, referenceFrames, settings.Height, settings.Width, parameters.CancellationToken);
                            int latentFrames = (referenceFrames - 1) / 8 + 1;
                            encodedReferences.Add(new(encoded, latentFrames, settings.LatentHeight, settings.LatentWidth));
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
                }
                Trim(); EndPhase("reference-encode");
                // One full-size pass, so the first-frame guide is encoded once, at the output size.
                encodedImages = EncodeImages(settings, "image-encode", "image-encode");
            }
            else
            {
                encodedImages = EncodeImages(first, "image-encode", "image-encode");
                if (effectiveFirstStageStrength.HasValue)
                    encodedImages = WithFirstStageImageStrength(encodedImages, effectiveFirstStageStrength.Value);
            }
            // Format cannot throw here: the reference counts it checks were refused above.
            string conditioningPrompt = identity
                ? Ltx25ReferencePrompt.Format(prompt, identityImageCount, identityVideoCount) : prompt;
            // A repeated prompt comes from the model's small prompt cache and makes this phase
            // almost instant, which would otherwise look like an anomaly in a trace.
            Report("text-encode", "Gemma 4 and learned AV conditioning", attributes: parameters.OnProgress == null ? null
                : new Dictionary<string, object> { [TelemetryAttributes.VideoPromptCacheHit] = model.HasCachedConditioning(conditioningPrompt) });
            Ltx25PromptEncoding conditioning = model.Conditioning(conditioningPrompt, parameters.CancellationToken);
            Trim();
            EndPhase("text-encode");

            // One denoising pass. The first draws all-noise latents; the refinement
            // pass mixes the upsampled latent and stage-one audio with fresh noise.
            Ltx25SamplingOutput RunStage(Ltx25GenerationSettings stage, List<Ltx25EncodedImage> stageImages, float[] sigmas,
                long noiseSeedOffset, int stepOffset, string phaseName, float[] priorVideo, float[] priorAudio)
            {
                int guideTokens = identity ? checked(encodedReferences.Sum(reference => reference.Frames * reference.Height * reference.Width))
                    : checked(stageImages.Count(image => image.FrameIndex != 0) * stage.LatentHeight * stage.LatentWidth);
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
                input = identity ? Ltx25ReferenceConditioning.Apply(input, stage, encodedReferences)
                    : Ltx25ImageConditioning.Apply(input, stage, stageImages, rawNoise, noiseScale);
                // The guide and the references occupy disjoint tokens (canvas frame 0, the appended tail), so pinning frame 0
                // after the references equals upstream's images-then-references order.
                if (identity && stageImages.Count > 0) input = Ltx25ImageConditioning.ApplyFirstFrame(input, stage, stageImages[0]);
                string progressStage = priorVideo == null ? twoStage ? "first-pass" : "single-pass" : "refinement";
                Report("denoise", $"{(priorVideo == null ? twoStage ? "first pass" : "single pass" : "refinement pass")} at {stage.Width}x{stage.Height}: {stage.VideoTokens} video + {guideTokens} guide, {stage.AudioTokens} audio, {conditioning.Tokens} prompt tokens, " +
                    (stage.Ancestral ? Ltx25GenerationSettings.AncestralSampler
                        : $"{Ltx25GenerationSettings.EulerSampler} (requested; the publisher samples LTX 2.5 with {Ltx25GenerationSettings.AncestralSampler})") +
                    (stageImages.Count == 0 ? "" : (identity ? "; first-frame guide strength: " : "; temporal guide strengths: ") + string.Join(", ", stageImages.Select(image =>
                        $"frame {image.FrameIndex}={image.Strength.ToString("R", CultureInfo.InvariantCulture)}"))), progressStage);
                Ltx25SamplingOutput result = model.Sample(input, stage, (step, total) =>
                    {
                        // One stage ahead of each companion's first use: the upscaler
                        // during the last first-pass step, the video decoder during the
                        // final pass, the audio decoder during the video decode.
                        bool finalPass = !twoStage || stepOffset != 0;
                        if (!finalPass && step == total - 1) model.PrefetchCompanion(Ltx25Model.CompanionFile.Upscaler);
                        if (finalPass && step == Math.Min(1, total - 1)) model.PrefetchCompanion(Ltx25Model.CompanionFile.VideoDecoder);
                        completed = stepOffset + step;
                        parameters.OnStep?.Invoke(completed, totalSteps);
                        Report("denoise", $"step {completed}/{totalSteps}", progressStage);
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
                Report("denoise", $"x2 latent upscale to {settings.Width}x{settings.Height}", "upscale");
                float[] firstLatents = Ltx25Geometry.UnpackVideo(latents.Video.AsSpan(0, checked(first.VideoTokens * 128)), first.VideoTokens, 128);
                float[] upsampled = model.UpsampleLatents(firstLatents, first.LatentFrames, first.LatentHeight, first.LatentWidth, parameters.CancellationToken);
                Trim();
                EndPhase("upsample");
                var refinementImages = EncodeImages(settings, "denoise", "refine-image-encode", "refine-image-encode");
                latents = RunStage(settings, refinementImages, Ltx25Scheduler.RefinementSigmas(),
                    Ltx25GenerationSettings.AncestralStage2NoiseSeedOffset, Ltx25GenerationSettings.Steps, "refine",
                    Ltx25Geometry.PackVideo(upsampled, settings.VideoTokens, 128), latents.Audio);
            }

            float[] videoLatents = Ltx25Geometry.UnpackVideo(latents.Video.AsSpan(0, checked(settings.VideoTokens * 128)), settings.VideoTokens, 128);
            finalLatents?.Invoke(settings, videoLatents, latents.Audio);
            if (parameters.GenerateAudio) model.PrefetchCompanion(Ltx25Model.CompanionFile.AudioDecoder);
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
            if (model.StopCompanionPrefetch() is string prefetched) Console.WriteLine($"  [ltx25] companion prefetch: {prefetched}");
            Report("done", $"{settings.Width}x{settings.Height}, {settings.Frames} frames at {settings.Fps} fps");
            return new GeneratedVideo { Frames = frames, Fps = settings.Fps, Seed = settings.Seed, Audio = audio };
        }
        finally { model.StopCompanionPrefetch(); Trim(); }
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
