// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 (joint video and 32 kHz stereo audio) on the diffusion runtime's video pipeline (DiffusionVideoPipeline): the
// family's hooks. A clip's mode chooses its checkpoint (FL2VA for text and keyframes, Ref2VA for references) and what it is
// conditioned on; a keyframe is conditioned twice, as Qwen3-VL vision features in the prompt and as a near-clean video latent
// pinned to its end of the timeline, and a Ref2VA reference takes its own stretch of the timeline before the clip. Both
// streams are stepped on the video schedule: the sampler carries (sigma_video / sigma_audio) * audio, restored to physical
// audio before each step and converted back (with its drift term) after it, and the final carrier scaled by
// audioShift / videoShift before the audio VAE.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>A MiniMax-H3 clip as the family resolved it, and what its phases produce for the next.</summary>
    internal sealed class MiniMaxH3Job : DiffusionVideoJob
    {
        public required VideoGenerationParams Params { get; init; }
        public required string Prompt { get; init; }
        public required long Seed { get; init; }
        public required MiniMaxH3Mode Mode { get; init; }
        public required MiniMaxH3Shape Shape { get; init; }
        public required float VideoShift { get; init; }
        public MiniMaxH3SamplingRecipe PluginRecipe { get; init; }
        public List<RgbImage> Keyframes { get; init; } = new();
        public List<bool> KeyframeAtEnd { get; init; } = new();
        public List<MiniMaxH3Requests.MiniMaxH3Reference> References { get; init; }

        // ---- Condition
        public int TextLength;
        public List<int> TextModalities;

        // ---- EncodeRefs / EncodeAudio
        public float[] ConditionTokens;
        public int ConditionCount;
        public List<int> ConditionFrames = new();
        public List<bool> ConditionAtEnd = new();
        public List<MiniMaxH3ReferenceBlock> ReferenceBlocks;
        public float[] ConditionAudio;
        public int ConditionAudioCount;
        public List<(int Kind, int Count)> Chunks;
        // Per reference: its encoded video patches (or null) and audio tokens (or null), joined in reference order.
        public float[][] ReferenceVideo, ReferenceAudio;
        public (int LatentT, int LatentH, int LatentW)[] ReferenceVideoShape;
        public int[] ReferenceAudioFrames;

        // ---- Denoise
        public Stopwatch Clock;
        public System.Threading.Tasks.Task<double> Prefault;

        /// <summary>Where the family's stage reports go (the request's progress sink).</summary>
        public Action<DiffusionVideoProgress> Report;
    }

    /// <summary>The MiniMax-H3 family: the hooks <see cref="DiffusionVideoPipeline"/> runs a clip through (see the file header).</summary>
    internal sealed class MiniMaxH3Family : IVideoDiffusionFamily
    {
        private readonly MiniMaxH3Model _model;

        public MiniMaxH3Family(MiniMaxH3Model model) => _model = model ?? throw new ArgumentNullException(nameof(model));

        public string Id => MiniMaxH3Model.ArchitectureId;
        public string DisplayName => "MiniMax-H3";

        private BackendType Backend => _model.Backend;

        // ---- components

        public IConditioner CreateConditioner(DiffusionLoad load) =>
            new MiniMaxH3Conditioner(load.File(DiffusionLoad.TextEncoder), load.Backend, _model.DirectAllocator);

        /// <summary>The denoiser of <paramref name="file"/>: on one device, or over the load's devices as whole-block stages
        /// (<see cref="DiffusionLoad.DenoiserDevices"/>, <c>--layer-split N</c>), the first of which is the model's own.</summary>
        public IDenoiser CreateDenoiser(DiffusionLoad load, string file) =>
            new MiniMaxH3Denoiser(file, load.Backend, _model.DirectAllocator, load.DenoiserDevices)
            {
                AdapterBytes = () => _model.Loras?.FactorBytes ?? 0,
            };

        public IResidentComponent CreateVideoCodec(DiffusionLoad load) =>
            new MiniMaxH3VideoCodec(load.File(DiffusionLoad.Vae), load.Backend, _model.DirectAllocator);

        public IResidentComponent CreateAudioCodec(DiffusionLoad load) =>
            load.Files.TryGetValue(MiniMaxH3Model.AudioVaeRole, out string path) && !string.IsNullOrWhiteSpace(path)
                ? new MiniMaxH3AudioCodec(path, load.Backend, _model.UsesDirectBackend, _model.DirectAllocator)
                : null;

        // ---- Prepare

        public DiffusionVideoJob Prepare(DiffusionVideoRequest request, DiffusionVideoComponents components)
        {
            var p = (request.FamilyRequest as VideoGenerationParams ?? new VideoGenerationParams()).ResolveVideoContinuations();
            // The LoRA plug-in is read once: the model swaps it only between clips.
            MiniMaxH3Partition partition = _model.PartitionFor(p);
            MiniMaxH3LoraSet lora = _model.LoraFor(partition);
            MiniMaxH3SamplingRecipe recipe = lora?.Recipe;
            MiniMaxH3Requests.ValidateSamplingParameters(p, recipe);

            // The mode decides what an image MEANS, and so whether it may size the canvas: a keyframe is a frame of the clip,
            // a reference is not.
            MiniMaxH3Mode mode = MiniMaxH3ModeResolver.Resolve(p, partition);
            MiniMaxH3Requests.RequirePluginMode(recipe, mode);
            var keyframes = new List<RgbImage>();
            var atEnd = new List<bool>();
            var (width, height, shape) = MiniMaxH3Requests.ResolveCanvasAndKeyframes(p, mode, keyframes, atEnd);

            // The base checkpoint samples 20 steps, stable-diffusion.cpp's CLI default; a fast clip takes the LightX2V Turbo
            // plug-in, whose recipe fixes the steps, the shift and the grid the adapter was distilled on.
            var (steps, videoShift, sigmas) = MiniMaxH3Requests.ResolveSampling(p, recipe);
            if (recipe != null)
            {
                Console.WriteLine(recipe.LogLine());
                string canvasNote = MiniMaxH3Requests.DescribePluginCanvas(recipe, shape);
                if (canvasNote != null) Console.WriteLine(canvasNote);
            }

            // Ref2VA reference images are loaded once and used twice: the vision tower reads them for the prompt, and the
            // video VAE encodes them into conditioning latents. Both must see the same pixels.
            List<MiniMaxH3Requests.MiniMaxH3Reference> references = mode == MiniMaxH3Mode.Reference
                ? MiniMaxH3Requests.LoadReferences(p, width, height, shape.Frames)
                : null;

            bool encodesVideo = keyframes.Count > 0 || references?.Any(r => r.Kind != MiniMaxH3ReferenceKind.Audio) == true;
            bool encodesAudio = references?.Any(r => r.Audio != null) == true;
            var conditioner = (MiniMaxH3Conditioner)components.Conditioner;
            long rows = (long)shape.VideoTokenCount + shape.AudioTokenCount + Math.Max(1, (request.Prompt?.Length ?? 0) / 3);
            if (keyframes.Count > 0) rows += (long)keyframes.Count * shape.TokenGridWidth * shape.TokenGridHeight;
            if (references != null)
                foreach (var r in references)
                    if (r.Frames is { Count: > 0 } f)
                        rows += (long)MiniMaxH3VideoVae.ChunkLatentFrames(f.Count) * (f[0].Height / 32) * (f[0].Width / 32);
            var demands = new List<PhaseDemand>
            {
                new(DiffusionPhase.Condition, DiffusionComponent.Conditioner, conditioner.WorkingBytesFor(request.Prompt, 0)),
            };
            IResidentComponent videoCodec = components.VideoCodec;
            if (encodesVideo)
                demands.Add(new(DiffusionPhase.EncodeReferences, DiffusionComponent.Vae, videoCodec.WorkingBytes(default)));
            if (encodesAudio)
                demands.Add(new(DiffusionPhase.EncodeAudio, DiffusionComponent.AudioVae, components.AudioCodec?.WorkingBytes(default) ?? 0));
            demands.Add(new(DiffusionPhase.Denoise, DiffusionComponent.Denoiser,
                MiniMaxH3Image.MiniMaxH3ImageDenoiser.WorkingBytesFor(_model.DitConfig, rows, fused: Backend == BackendType.GgmlCuda,
                    quantized: true, bf16Route: true, edit: keyframes.Count > 0 || references is { Count: > 0 })));
            demands.Add(new(DiffusionPhase.Decode, DiffusionComponent.Vae, videoCodec.WorkingBytes(default)));
            if (request.DecodeAudio && components.AudioCodec is { } audioCodec)
                demands.Add(new(DiffusionPhase.DecodeAudio, DiffusionComponent.AudioVae, audioCodec.WorkingBytes(default)));

            return new MiniMaxH3Job
            {
                Recipe = recipe != null ? PluginSampling(recipe) : BaseSampling(steps, videoShift),
                Steps = steps,
                Sigmas = sigmas,
                DenoiserFile = _model.DenoiserPathFor(partition),
                Demands = demands,
                GeometryClass = $"{shape.Width}x{shape.Height}x{shape.Frames}",
                Describe = $"{mode}, {steps} steps, seed {request.Seed}",
                Params = p,
                Prompt = request.Prompt ?? string.Empty,
                Seed = request.Seed,
                Mode = mode,
                Shape = shape,
                VideoShift = videoShift,
                PluginRecipe = recipe,
                Keyframes = keyframes,
                KeyframeAtEnd = atEnd,
                References = references,
                Clock = Stopwatch.StartNew(),
                Report = request.Progress,
            };
        }

        /// <summary>The base checkpoint's schedule as a runtime recipe: stable-diffusion.cpp's sigmas at the video shift (the
        /// audio stream follows them through the carrier).</summary>
        private static SamplingRecipe BaseSampling(int steps, float videoShift) => new()
        {
            Schedule = DiffusionSchedule.Explicit,
            Steps = steps,
            Sigmas = MiniMaxH3Scheduler.BuildSigmas(steps, videoShift),
            Owner = "MiniMax-H3",
        };

        /// <summary>A Turbo plug-in's recipe: its linspace grid at the video shift, the audio stream at its own.</summary>
        private static SamplingRecipe PluginSampling(MiniMaxH3SamplingRecipe recipe) => new()
        {
            Schedule = DiffusionSchedule.LinspaceShift,
            Steps = recipe.Steps,
            Shift = recipe.VideoShift,
            StreamShifts = new Dictionary<string, float> { [MiniMaxH3Denoiser.AudioStream] = recipe.AudioShift },
            Owner = recipe.Name,
        };

        // ---- Condition

        public ConditioningSet Condition(DiffusionVideoJob job, IConditioner conditioner, CancellationToken ct)
        {
            var j = (MiniMaxH3Job)job;
            var c = (MiniMaxH3Conditioner)conditioner;
            var clock = Stopwatch.StartNew();
            Report(j, DiffusionPhase.Condition, "text-encoder");
            // A CUDA upload warms the encoder's mapped pages by touching them (eight workers, one fault per page), which on an
            // A40 pod's network volume read the 15.5 GB at ~1 GB/s; concurrent 8 MiB reads reach ~2.3 GiB/s there, and the
            // touches then find the pages cached. Metal binds the same mapped pages zero-copy, so it needs none.
            bool directRead = MiniMaxH3DirectRead.Supported(Backend);
            using var teRead = Backend == BackendType.GgmlCuda && !directRead
                ? MappedFileReadAhead.TryStart(c.Path, MappedFileReadAhead.WholeFile(c.Path),
                    message => Console.WriteLine("  [h3] warning: text-encoder " + message), long.MaxValue)
                : null;
            MiniMaxH3TextEncoder te = c.Encoder;
            double openSeconds = c.LastOpenSeconds;
            string presented = j.Prompt;
            List<MiniMaxH3PromptImage> promptImages = null;
            if (j.Keyframes.Count > 0)
            {
                if (!te.HasVision)
                    throw new InvalidOperationException(
                        "image conditioning needs the Qwen3-VL vision tower, and this text-encoder checkpoint does not carry one. " +
                        "Point --video-text-encoder at the full qwen3vl_32b_minimax_h3 GGUF.");
                // A KEYFRAME is conditioned twice: the VAE latent pins its frame, and the same picture through the vision tower
                // describes the scene for the whole clip (without it long clips drift off the keyframe).
                var shown = new List<MiniMaxH3Requests.MiniMaxH3Reference>(j.Keyframes.Count);
                foreach (RgbImage frame in j.Keyframes)
                    shown.Add(new MiniMaxH3Requests.MiniMaxH3Reference
                    {
                        Kind = MiniMaxH3ReferenceKind.Image,
                        Frames = new List<RgbImage> { frame },
                        Presented = new List<RgbImage> { frame },
                        Timestamps = new List<float> { 0f },
                    });
                (presented, promptImages) = MiniMaxH3Requests.BuildReferencePrompt(te, shown, j.Prompt);
                Console.WriteLine($"  [h3] {j.Keyframes.Count} keyframe(s) presented to the vision tower -> {promptImages.Count} vision span(s)");
            }
            else if (j.References is { Count: > 0 })
            {
                MiniMaxH3Requests.RequireReferenceVision(te.HasVision, j.References);
                (presented, promptImages) = MiniMaxH3Requests.BuildReferencePrompt(te, j.References, j.Prompt);
                Console.WriteLine($"  [h3] ref2va: {j.References.Count} reference(s) -> {promptImages.Count} vision span(s)");
            }
            ct.ThrowIfCancellationRequested();
            List<int> ids = te.Tokenize(presented);
            if (ids.Count == 0) ids.Add(te.Tokenizer.BosTokenId);
            var runClock = Stopwatch.StartNew();
            float[] hidden = te.Encode(ids, promptImages);
            double runSeconds = runClock.Elapsed.TotalSeconds;
            teRead?.Stop();
            // The trunk is finished with, so the denoiser read starts now and runs alongside the encoder's teardown and the
            // denoiser's upload (it walks the file in the order the upload does, ahead of it).
            if (!directRead)
            {
                string ditPath = j.DenoiserFile;
                long avail = AvailablePhysicalBytes();
                j.Prefault = System.Threading.Tasks.Task.Run(() => MiniMaxH3Model.PrefaultWeightFile(ditPath, avail));
            }
            j.TextLength = ids.Count;
            if (promptImages is { Count: > 0 })
                j.TextModalities = MiniMaxH3Requests.BuildTextModalities(ids.Count, promptImages);
            Console.WriteLine($"  [h3] text conditioning: {ids.Count} tokens in {clock.Elapsed.TotalSeconds:F1}s");
            Report(j, DiffusionPhase.Condition, "text-encoder", attributes: new Dictionary<string, object>
            {
                ["tensorsharp.video.open_s"] = Math.Round(openSeconds, 3),
                ["tensorsharp.video.run_s"] = Math.Round(runSeconds, 3),
                ["tensorsharp.video.teardown_s"] = Math.Round(Math.Max(0, clock.Elapsed.TotalSeconds - openSeconds - runSeconds), 3),
            });
            return new ConditioningSet(hidden, ids.Count, te.Config.Hidden,
                j.TextModalities?.ToArray());
        }

        /// <summary>Physical memory free right now, or 0 when it cannot be read.</summary>
        private static long AvailablePhysicalBytes()
        {
            try
            {
                var info = GC.GetGCMemoryInfo();
                long avail = info.TotalAvailableMemoryBytes - info.MemoryLoadBytes;
                return avail > 0 ? avail : 0;
            }
            catch (PlatformNotSupportedException) { return 0; }
        }

        // ---- EncodeRefs / EncodeAudio

        public void EncodeVideo(DiffusionVideoJob job, IResidentComponent videoCodec, Action<DiffusionVideoProgress> report,
            CancellationToken ct)
        {
            var j = (MiniMaxH3Job)job;
            MiniMaxH3VideoVae vae = ((MiniMaxH3VideoCodec)videoCodec).Vae;
            MiniMaxH3Shape shape = j.Shape;
            if (j.Mode is MiniMaxH3Mode.ImageToVideo or MiniMaxH3Mode.FirstLastFrame)
            {
                // A keyframe is NOT a mask: it becomes extra sequence tokens at a near-clean timestep, pinned to the start or
                // end of the timeline, fitted to the generation canvas so it shares the target's spatial axes.
                Report(j, DiffusionPhase.EncodeReferences, "image-encode");
                var parts = new List<float[]>();
                for (int i = 0; i < j.Keyframes.Count; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    float[] latent = vae.EncodeFrame(j.Keyframes[i]);
                    vae.NormalizeLatent(latent);
                    MiniMaxH3Noise.AddVisual(latent, j.Seed, 1, shape.LatentHeight, shape.LatentWidth);
                    parts.Add(MiniMaxH3Latents.PatchifyVideo(latent, 1, shape.LatentHeight, shape.LatentWidth, MiniMaxH3Geometry.VideoLatentChannels));
                    j.ConditionFrames.Add(1);
                    j.ConditionAtEnd.Add(j.KeyframeAtEnd[i]);
                }
                int perFrame = shape.TokenGridWidth * shape.TokenGridHeight;
                j.ConditionCount = parts.Count * perFrame;
                j.ConditionTokens = MiniMaxH3Requests.Concat(parts);
                Console.WriteLine($"  [h3] {j.Mode}: {parts.Count} keyframe(s) -> {j.ConditionCount} conditioning tokens");
                return;
            }
            if (j.References is not { Count: > 0 }) return;

            // A reference is not a frame the clip must reproduce: it takes its own stretch of the shared timeline before the
            // generated video, keeps its own aspect ratio, and is noised to the same near-clean timestep as a keyframe.
            Report(j, DiffusionPhase.EncodeReferences, "reference-encode");
            int n = j.References.Count;
            j.ReferenceVideo ??= new float[n][];
            j.ReferenceVideoShape ??= new (int, int, int)[n];
            for (int i = 0; i < n; i++)
            {
                MiniMaxH3Requests.MiniMaxH3Reference reference = j.References[i];
                if (reference.Kind == MiniMaxH3ReferenceKind.Audio) continue;
                ct.ThrowIfCancellationRequested();
                var encodeClock = Stopwatch.StartNew();
                RgbImage first = reference.Frames[0];
                int latentH = first.Height / MiniMaxH3Geometry.VaeSpatialRatio, latentW = first.Width / MiniMaxH3Geometry.VaeSpatialRatio;
                int latentT;
                // A lone reference frame goes through the 2-D reduction of the encoder, which is exact for a single frame.
                float[] refLatent;
                if (reference.Frames.Count == 1)
                {
                    latentT = 1;
                    refLatent = vae.EncodeFrame(first);
                }
                else refLatent = vae.EncodeClip(reference.Frames, out latentT);
                vae.NormalizeLatent(refLatent);
                MiniMaxH3Noise.AddVisual(refLatent, j.Seed, latentT, latentH, latentW);
                j.ReferenceVideo[i] = MiniMaxH3Latents.PatchifyVideo(refLatent, latentT, latentH, latentW, MiniMaxH3Geometry.VideoLatentChannels);
                j.ReferenceVideoShape[i] = (latentT, latentH, latentW);
                Console.WriteLine($"  [h3] encoded reference {i}: {reference.Frames.Count}x{first.Width}x{first.Height} -> " +
                                  $"{latentT} latent frame(s) in {encodeClock.Elapsed.TotalSeconds:F1}s");
            }
            if (!j.EncodesAudio) AssembleReferences(j);
        }

        public void EncodeAudio(DiffusionVideoJob job, IResidentComponent audioCodec, CancellationToken ct)
        {
            var j = (MiniMaxH3Job)job;
            if (j.References is not { Count: > 0 }) return;
            MiniMaxH3AudioVae audioVae = ((MiniMaxH3AudioCodec)audioCodec).Vae;
            if (!audioVae.HasEncoder)
                throw new InvalidOperationException(
                    "reference audio needs the audio VAE and its encoder; point --audio-vae at minimax_h3_audio_vae_fp32.safetensors.");
            int n = j.References.Count;
            j.ReferenceAudio = new float[n][];
            j.ReferenceAudioFrames = new int[n];
            for (int i = 0; i < n; i++)
            {
                MiniMaxH3Requests.MiniMaxH3Reference reference = j.References[i];
                if (reference.Audio == null) continue;
                ct.ThrowIfCancellationRequested();
                int frames = MiniMaxH3AudioVae.FramesFor(reference.Audio[0].Length);
                var planes = new List<float[]>(reference.Audio.Length);
                foreach (float[] channel in reference.Audio)
                {
                    float[] plane = audioVae.EncodeMono(channel);
                    audioVae.NormalizePlane(plane, frames);
                    planes.Add(plane);
                }
                // The DiT wants [channel][t][32]; the encoder emits [32][t]. No visual noise: an encoded soundtrack conditions
                // exactly as it came out.
                var tokens = new float[(long)planes.Count * frames * MiniMaxH3AudioVae.LatentChannels];
                for (int ch = 0; ch < planes.Count; ch++)
                    for (int t = 0; t < frames; t++)
                        for (int c = 0; c < MiniMaxH3AudioVae.LatentChannels; c++)
                            tokens[((long)ch * frames + t) * MiniMaxH3AudioVae.LatentChannels + c] = planes[ch][(long)c * frames + t];
                j.ReferenceAudio[i] = tokens;
                j.ReferenceAudioFrames[i] = frames;
            }
            AssembleReferences(j);
        }

        /// <summary>The references' conditioning rows in reference order (a clip with a soundtrack: its audio, then its video),
        /// the chunk list that places them, and the blocks the layout packs.</summary>
        private static void AssembleReferences(MiniMaxH3Job j)
        {
            j.ReferenceBlocks = new List<MiniMaxH3ReferenceBlock>(j.References.Count);
            j.Chunks = new List<(int Kind, int Count)>();
            var videoParts = new List<float[]>();
            var audioParts = new List<float[]>();
            int videoIndex = 0, audioIndex = 0;
            for (int i = 0; i < j.References.Count; i++)
            {
                MiniMaxH3Requests.MiniMaxH3Reference reference = j.References[i];
                int thisAudio = -1, thisVideo = -1, audioFrames = 0;
                var (latentT, latentH, latentW) = j.ReferenceVideoShape?[i] ?? (0, 0, 0);
                if (reference.Audio != null)
                {
                    audioFrames = j.ReferenceAudioFrames[i];
                    audioParts.Add(j.ReferenceAudio[i]);
                    thisAudio = audioIndex++;
                    j.ConditionAudioCount += audioFrames * MiniMaxH3Geometry.AudioChannels;
                    j.Chunks.Add((1, audioFrames * MiniMaxH3Geometry.AudioChannels));
                }
                if (reference.Kind != MiniMaxH3ReferenceKind.Audio)
                {
                    videoParts.Add(j.ReferenceVideo[i]);
                    thisVideo = videoIndex++;
                    j.Chunks.Add((0, latentT * (latentH / MiniMaxH3Geometry.PatchH) * (latentW / MiniMaxH3Geometry.PatchW)));
                }
                j.ReferenceBlocks.Add(new MiniMaxH3ReferenceBlock
                {
                    Kind = reference.Kind,
                    VideoIndex = thisVideo,
                    AudioIndex = thisAudio,
                    LatentFrames = Math.Max(1, latentT),
                    LatentHeight = latentH,
                    LatentWidth = latentW,
                    AudioLatentFrames = audioFrames,
                });
            }
            j.ConditionTokens = MiniMaxH3Requests.Concat(videoParts);
            j.ConditionCount = j.ConditionTokens.Length / (MiniMaxH3Geometry.VideoLatentChannels * MiniMaxH3Geometry.PatchH * MiniMaxH3Geometry.PatchW);
            j.ConditionAudio = MiniMaxH3Requests.Concat(audioParts);
            if (j.ConditionAudio.Length == 0) { j.ConditionAudio = null; j.ConditionAudioCount = 0; }
            Console.WriteLine($"  [h3] {j.Mode}: {j.References.Count} reference(s) -> {j.ConditionCount} visual + " +
                              $"{j.ConditionAudioCount} audio conditioning tokens");
        }

        // ---- Denoise

        public DiffusionStreams InitialStreams(DiffusionVideoJob job)
        {
            var j = (MiniMaxH3Job)job;
            // A reference set with no picture needs its blocks too (EncodeVideo did not run).
            if (j.References is { Count: > 0 } && j.ReferenceBlocks == null) AssembleReferences(j);
            // sigma[0] is 1, so the carrier at the first step is the physical audio noise.
            var (video, audio) = MiniMaxH3Noise.Initial(j.Shape, j.Seed);
            return new DiffusionStreams(new Dictionary<string, float[]>
            {
                [DiffusionStreams.Primary] = video,
                [MiniMaxH3Denoiser.AudioStream] = audio,
            });
        }

        public DenoiseStepInputs StepInputs(DiffusionVideoJob job, ConditioningSet conditioning, DiffusionStreams state, StepState step)
        {
            var j = (MiniMaxH3Job)job;
            float[] video = state[DiffusionStreams.Primary], carrier = state[MiniMaxH3Denoiser.AudioStream];
            if (step.Index > 0)
            {
                // The previous step's update, checked before it is read (a diverged clip decodes to black with a buzz).
                MiniMaxH3Requests.RequireFinite(video, "video latent", step.Index - 1, step.Count, 0);
                MiniMaxH3Requests.RequireFinite(carrier, "audio carrier", step.Index - 1, step.Count, 0);
            }
            if (step.Index == 0) Report(j, DiffusionPhase.Denoise, null);
            MiniMaxH3PackedLayout layout = MiniMaxH3Layout.Build(j.TextLength, j.Shape, step.Sigma,
                conditionFrames: j.ConditionFrames.Count > 0 ? j.ConditionFrames : null,
                keyframeAtEnd: j.ConditionAtEnd.Count > 0 ? j.ConditionAtEnd : null,
                textTokenModalities: j.TextModalities,
                videoShift: j.VideoShift, audioShift: MiniMaxH3Scheduler.AudioShift,
                references: j.ReferenceBlocks);
            if (step.Index == 0) MiniMaxH3Requests.RequireChunksMatchLayout(j.Chunks, layout);
            // The shared sigma schedule carries scaled audio, as in the reference sampler; the DiT reads physical audio.
            float[] audioInput = MiniMaxH3Scheduler.AudioInputFromCarrier(carrier, step.Sigma, j.VideoShift);
            return new DenoiseStepInputs
            {
                Latent = video,
                Conditioning = conditioning,
                Layout = new DiffusionLayout
                {
                    Rows = layout.TokenCount,
                    TargetStart = layout.VideoSpan.Start,
                    TargetRows = layout.VideoSpan.Length,
                    Timesteps = layout.Timesteps,
                    Positions = layout.Positions,
                    PositionAxes = 3,
                    FamilyLayout = new MiniMaxH3StepLayout
                    {
                        Layout = layout,
                        ConditionTokens = j.ConditionTokens,
                        ConditionCount = j.ConditionCount,
                        ConditionAudio = j.ConditionAudio,
                        ConditionAudioCount = j.ConditionAudioCount,
                        Chunks = j.Chunks,
                        VideoShift = j.VideoShift,
                        AudioShift = MiniMaxH3Scheduler.AudioShift,
                    },
                },
                Step = step,
                Recipe = j.Recipe,
                Streams = new Dictionary<string, float[]> { [MiniMaxH3Denoiser.AudioStream] = audioInput },
            };
        }

        public DiffusionStreams SamplerVelocities(DiffusionVideoJob job, DiffusionStreams state, DenoiseStepInputs inputs, StepOutput output,
            StepState step)
        {
            var j = (MiniMaxH3Job)job;
            int tokens = inputs.Layout.Rows;
            float[] video = output.Velocity, audio = output.Streams[MiniMaxH3Denoiser.AudioStream];
            // Both streams share one trunk, so one bad number takes the whole clip down; check them before they are integrated.
            MiniMaxH3Requests.RequireFinite(video, "video velocity", step.Index, step.Count, tokens);
            MiniMaxH3Requests.RequireFinite(audio, "audio velocity", step.Index, step.Count, tokens);
            MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(inputs.Streams[MiniMaxH3Denoiser.AudioStream], audio, step.Sigma, j.VideoShift);
            MiniMaxH3Requests.RequireFinite(audio, "audio carrier velocity", step.Index, step.Count, tokens);
            if (step.Index == 0)
                Console.WriteLine($"  [h3] {tokens} packed tokens; first step in {j.Clock.Elapsed.TotalSeconds:F1}s");
            return new DiffusionStreams(new Dictionary<string, float[]>
            {
                [DiffusionStreams.Primary] = video,
                [MiniMaxH3Denoiser.AudioStream] = audio,
            });
        }

        // ---- Decode

        public RgbImage[] DecodeVideo(DiffusionVideoJob job, DiffusionStreams state, IResidentComponent videoCodec,
            Action<DiffusionVideoProgress> report, CancellationToken ct)
        {
            var j = (MiniMaxH3Job)job;
            MiniMaxH3Shape shape = j.Shape;
            MiniMaxH3Requests.RequireFinite(state[DiffusionStreams.Primary], "video latent", j.Steps - 1, j.Steps, 0);
            MiniMaxH3Requests.RequireFinite(state[MiniMaxH3Denoiser.AudioStream], "audio carrier", j.Steps - 1, j.Steps, 0);
            // The pipelined denoiser read is collected here: the denoiser has finished with the file either way.
            j.Prefault?.GetAwaiter().GetResult();
            // The denoise loop's reuse gallocr and the CUDA attention/projection scratch are of no use to the decode or to the
            // next clip's 17 GB text encoder.
            if (Backend == BackendType.GgmlCuda)
                _model.RunWhileOtherGgmlModelsAreIdle(GgmlBasicOps.ReleaseReuseComputeBuffers);
            var codec = (MiniMaxH3VideoCodec)videoCodec;
            Report(j, DiffusionPhase.Decode, "vae-open",
                attributes: new Dictionary<string, object> { [TensorSharp.Runtime.Telemetry.TelemetryAttributes.VideoCached] = codec.Made });
            MiniMaxH3VideoVae vae = codec.Vae;
            Report(j, DiffusionPhase.Decode, "vae-decode");
            float[] latent = MiniMaxH3Latents.UnpatchifyVideo(state[DiffusionStreams.Primary], shape.LatentFrames, shape.LatentHeight,
                shape.LatentWidth, MiniMaxH3Geometry.VideoLatentChannels);
            vae.DenormalizeLatent(latent);
            MiniMaxH3Requests.RequireFiniteOutput(latent, "video VAE input");
            // Straight to RGB frames: the decoder resolves the warm-up frames and chunk overlaps, checks the values are finite
            // and converts each frame in one pass.
            return vae.DecodeFrames(latent, shape.LatentFrames, shape.LatentHeight, shape.LatentWidth, shape.Frames,
                (tile, tiles) => Report(j, DiffusionPhase.Decode, "vae-decode", $"tile {tile}/{tiles}",
                    new Dictionary<string, object> { [TensorSharp.Runtime.Telemetry.TelemetryAttributes.VideoTiles] = tiles }));
        }

        public (float[][] Channels, int SampleRate) DecodeAudio(DiffusionVideoJob job, DiffusionStreams state, IResidentComponent audioCodec,
            CancellationToken ct)
        {
            var j = (MiniMaxH3Job)job;
            // The final sampler carrier has a videoShift/audioShift factor; undo it before the VAE's denormalization
            // (H3AVFlowDenoiser.process_latent_out).
            float[] audioLatent = (float[])state[MiniMaxH3Denoiser.AudioStream].Clone();
            MiniMaxH3Scheduler.AudioOutputFromCarrierInPlace(audioLatent, j.VideoShift);
            MiniMaxH3Requests.RequireFiniteOutput(audioLatent, "audio VAE input");
            GeneratedVideoAudio audio = ((MiniMaxH3AudioCodec)audioCodec).Vae.DecodeStereo(audioLatent, j.Shape.AudioLatentFrames);
            foreach (float[] channel in audio.Channels)
                MiniMaxH3Requests.RequireFiniteOutput(channel, "audio VAE output");
            // The audio grid is 40 Hz and the video grid 24 fps, so the tracks rarely land on the same length; trim to the
            // video's duration.
            int wanted = (int)Math.Round(j.Shape.Frames / (double)j.Shape.Fps * audio.SampleRate);
            float[][] channels = audio.Channels;
            if (audio.SampleCount > wanted && wanted > 0)
            {
                channels = new float[audio.ChannelCount][];
                for (int c = 0; c < audio.ChannelCount; c++)
                {
                    channels[c] = new float[wanted];
                    Array.Copy(audio.Channels[c], channels[c], wanted);
                }
            }
            return (channels, audio.SampleRate);
        }

        // ---- progress

        /// <summary>A report of the clip's stage; the pipeline's reports carry the phases and the steps.</summary>
        private static void Report(MiniMaxH3Job j, DiffusionPhase phase, string stage, string detail = null,
            IReadOnlyDictionary<string, object> attributes = null) =>
            j.Report?.Invoke(new DiffusionVideoProgress(phase, 0, j.Steps, stage, detail, attributes));
    }
}
