// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LTX 2.5 Distilled: text/image-to-video with jointly generated 48 kHz stereo audio. A clip runs on the diffusion runtime's
// video pipeline (SharedDiffusion/DiffusionVideoPipeline) with the LTX family's hooks (Ltx25Family): the denoiser's steps are
// graph programs (Ltx25Denoiser) on a tensor executor (CUDA, MLX or the managed CPU executor), and the optional Ingredients
// IC-LoRA (identity references) is the step's adapter set (Ltx25LoraSet).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Media;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.Ltx25;

/// <summary>Native LTX 2.5 Distilled text/image-to-video with joint stereo audio.</summary>
public sealed class Ltx25Model : ModelBase, IVideoGenerationModel
{
    public const string ArchitectureId = "ltx-2.5";
    internal string CheckpointPath { get; }
    internal string TextEncoderPath { get; }
    internal string VideoVaePath { get; }
    /// <summary>The publisher's diffusion video VAE when configured or installed beside
    /// the model; it decodes on MLX and CPU. Null decodes with <see cref="VideoVaePath"/>.</summary>
    internal string DiffusionVaePath { get; }
    /// <summary>True when frames come from the diffusion decoder rather than the convolutional one.</summary>
    internal bool UsesDiffusionDecoder => DiffusionVaePath != null;
    internal string AudioVaePath { get; }
    private string _referenceLoraPath;
    internal string ReferenceLoraPath => Volatile.Read(ref _referenceLoraPath);
    /// <summary>The x2 spatial latent upscaler between the half- and full-size passes.</summary>
    internal string LatentUpscalerPath { get; }
    internal IAllocator DirectAllocator => _allocator;
    // The devices of the denoiser's whole-block stages (--layer-split N), the first DirectAllocator itself; null for one device.
    private readonly IReadOnlyList<IAllocator> _denoiserDevices;
    // One clip at a time, and never across an adapter swap or an unload.
    private readonly object _generationLock = new();
    private DiffusionVideoPipeline _pipeline;
    private Ltx25LoraSet _ingredients;
    private readonly List<MappedFileReadAhead> _companionReads = new();
    private bool _disposed;

    /// <summary>Prompts the conditioner answered from its cache.</summary>
    public long PromptCacheHits => (_pipeline?.ConditionerOrNull as Ltx25Conditioner)?.CacheHits ?? 0;

    /// <summary>The loaded Ingredients adapter set, or null (it is opened when the adapter is attached and applied only to
    /// identity reference clips).</summary>
    internal Ltx25LoraSet Ingredients => Volatile.Read(ref _ingredients);

    /// <summary>The pipeline's denoiser, or null before the first clip.</summary>
    internal IDenoiser LoadedDenoiser => _pipeline?.LoadedDenoiser;

    /// <param name="layerSplitDegree">Devices the denoiser's 48 blocks are split over as whole-block stages (--layer-split N,
    /// --backend cuda): each holds its share of the blocks; the text encoder, the VAEs and the upscaler run on the first.</param>
    public Ltx25Model(string checkpoint, BackendType backend, int layerSplitDegree = 1)
        : base(checkpoint, DirectBackend(backend), layerSplitDegree: layerSplitDegree)
    {
        try
        {
            var geometry = Ltx25Config.FromGguf(_gguf);
            if (geometry.VideoDim != 4096 || geometry.AudioDim != 2048 || geometry.Layers != 48 ||
                geometry.VideoChannels != 128 || geometry.AudioChannels != 128)
                throw new NotSupportedException("LTX generation requires the authentic 48-layer LTX 2.5 AV checkpoint.");
            // The stage devices, made (and a split past the visible CUDA devices refused) before any companion is resolved.
            _denoiserDevices = StageDeviceAllocators();
            if (IsDevCheckpointName(checkpoint))
                throw new ModelLoadRefusedException(
                    $"'{Path.GetFileName(checkpoint)}' is LTX 2.5's dev checkpoint, which is sampled with 30 steps at CFG 3. " +
                    "TensorSharp runs the distilled recipe (8 + 3 steps at CFG 1) and needs a distilled GGUF, such as " +
                    "distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf from unsloth/LTX-2.5-GGUF.");
            CheckpointPath = Path.GetFullPath(checkpoint);
            string directory = Path.GetDirectoryName(CheckpointPath);
            TextEncoderPath = Companion(directory, "TS_VIDEO_TEXT_ENCODER", "text_encoders", "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors");
            VideoVaePath = Companion(directory, "TS_VIDEO_VAE", "vae", "ltx-2.5-video-vae-conv-bf16.safetensors");
            AudioVaePath = Companion(directory, "TS_VIDEO_AUDIO_VAE", "vae", "ltx-2.5-audio-vae-bf16.safetensors");
            LatentUpscalerPath = Companion(directory, "TS_VIDEO_UPSCALER", "latent_upscale_models", LatentUpscalerFileName);
            DiffusionVaePath = ResolveDiffusionVae(VideoVaePath, _allocator is CudaAllocator);
            Console.WriteLine($"  [ltx25] video decoder: {(UsesDiffusionDecoder ? "diffusion" : "convolutional")} " +
                $"({Path.GetFileName(DiffusionVaePath ?? VideoVaePath)})");
            if (_denoiserDevices != null)
                Console.WriteLine($"  placement    = denoiser blocks over {_denoiserDevices.Count} GPUs as whole-block stages " +
                    "(the text encoder, VAEs and upscaler on the first)");
            string configuredReference = Environment.GetEnvironmentVariable("TS_LTX25_INGREDIENTS_LORA");
            string reference = null;
            if (!string.IsNullOrWhiteSpace(configuredReference))
            {
                if (!File.Exists(configuredReference)) throw new FileNotFoundException("The configured LTX Ingredients adapter does not exist.", configuredReference);
                reference = Path.GetFullPath(configuredReference);
            }
            else
            {
                string sibling = Path.Combine(directory, "loras", IngredientsFileName);
                if (File.Exists(sibling)) reference = Path.GetFullPath(sibling);
            }
            if (reference != null)
            {
                _ingredients = Ltx25LoraSet.Open(reference, _gguf, geometry);
                _referenceLoraPath = reference;
            }
            Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = 0 };
        }
        catch { _ingredients?.Dispose(); base.Dispose(); throw; }
    }

    internal const string LatentUpscalerFileName = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
    internal const string IngredientsFileName = "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors";

    /// <summary>
    /// Whether <paramref name="checkpoint"/> is named as LTX 2.5's dev (full, non-distilled)
    /// checkpoint: a "dev" word in the file name, as in the publisher's
    /// ltx-2.5-22b-dev-transformer-bf16 and the ltx-2.5-22b-dev-*.gguf files at the root of
    /// unsloth/LTX-2.5-GGUF, beside its distilled/ folder. The two checkpoints have the same
    /// tensors, shapes and embedded config, so the name is all that tells them apart, and the
    /// distilled recipe this model runs turns the dev weights into noise without an error.
    /// A name that also says "distill" is the distilled checkpoint whatever else it says.
    /// </summary>
    internal static bool IsDevCheckpointName(string checkpoint)
    {
        string name = Path.GetFileNameWithoutExtension(checkpoint);
        return System.Text.RegularExpressions.Regex.IsMatch(name, @"(^|[-_. ])dev([-_. ]|$)",
                   System.Text.RegularExpressions.RegexOptions.IgnoreCase)
               && name.IndexOf("distill", StringComparison.OrdinalIgnoreCase) < 0;
    }

    /// <summary>The diffusion VAE when explicitly selected: TS_VIDEO_DIFFUSION_VAE, or
    /// the video VAE itself when that is the diffusion checkpoint. It is never picked up
    /// beside the denoiser: it renders cleaner but softer frames than the convolutional
    /// default, so it is a choice, not an upgrade. CUDA has no neighborhood-attention
    /// kernel and refuses it.</summary>
    private static string ResolveDiffusionVae(string videoVae, bool cuda)
    {
        bool IsDiffusion(string path)
        {
            using var file = new SafetensorsFile(path);
            return Ltx25DiffusionVaeConfig.IsDiffusionVae(file.Metadata);
        }
        string configured = Environment.GetEnvironmentVariable("TS_VIDEO_DIFFUSION_VAE");
        string path;
        if (IsDiffusion(videoVae)) path = videoVae;
        else if (string.IsNullOrWhiteSpace(configured)) return null;
        else
        {
            if (!File.Exists(configured)) throw new FileNotFoundException("The configured LTX diffusion video VAE does not exist.", configured);
            path = Path.GetFullPath(configured);
            if (!IsDiffusion(path)) throw new InvalidDataException($"'{path}' is not a CausalDiffusionVAE checkpoint.");
        }
        if (cuda)
            throw new NotSupportedException("The LTX 2.5 diffusion video decoder runs on MLX and CPU. On CUDA, remove " +
                "--video-diffusion-vae (TS_VIDEO_DIFFUSION_VAE) and use the convolutional ltx-2.5-video-vae-conv-bf16.safetensors as --video-vae.");
        return path;
    }

    // The family runs on the tensor executors of the CUDA, MLX and CPU backends; the GGML option names stay usable by hosts
    // whose device picker uses them and select the same executors.
    internal static BackendType DirectBackend(BackendType backend) => backend switch
    {
        BackendType.GgmlCuda or BackendType.Cuda => BackendType.Cuda,
        BackendType.GgmlMetal or BackendType.Mlx => BackendType.Mlx,
        BackendType.GgmlCpu or BackendType.Cpu => BackendType.Cpu,
        _ => throw new NotSupportedException("LTX 2.5 supports CUDA, MLX and CPU execution; Vulkan is not implemented."),
    };

    private static string Companion(string directory, string variable, string subdirectory, string filename)
    {
        string configured = Environment.GetEnvironmentVariable(variable);
        if (!string.IsNullOrWhiteSpace(configured))
        {
            if (File.Exists(configured)) return Path.GetFullPath(configured);
            throw new FileNotFoundException($"The LTX companion specified by {variable} does not exist.", configured);
        }
        foreach (string path in new[] { Path.Combine(directory, subdirectory, filename), Path.Combine(directory, filename) })
            if (File.Exists(path)) return path;
        throw new FileNotFoundException($"LTX 2.5 requires {filename}; configure its companion path or install it beside the denoiser.", filename);
    }

    public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams parameters = null) => Generate(prompt, parameters, null);

    /// <summary>A clip through <paramref name="family"/> on a pipeline of its own (a validation tool's family that wraps this
    /// model's <see cref="Ltx25Family"/> to read its state between the hooks), or, null, through the model's pipeline.</summary>
    internal GeneratedVideo Generate(string prompt, VideoGenerationParams parameters, IVideoDiffusionFamily family)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
            return MlxWorker.Shared.Invoke(() => Generate(prompt, parameters, family));
        lock (_generationLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            parameters ??= new VideoGenerationParams();
            DiffusionVideoPipeline own = family == null ? null : new DiffusionVideoPipeline(family, Load());
            DiffusionVideoPipeline pipeline = own ?? (_pipeline ??= new DiffusionVideoPipeline(new Ltx25Family(this), Load()));
            var clock = Stopwatch.StartNew();
            Ltx25GenerationSettings settings = null;
            try
            {
                DiffusionVideoResult result = pipeline.Run(new DiffusionVideoRequest
                {
                    Prompt = prompt,
                    Seed = parameters.Seed,
                    // The Ingredients adapter conditions identity references only; every other clip runs the base checkpoint.
                    Adapters = parameters.Mode == "ref" && Ingredients is { } set ? set.AdapterSet : AdapterSet.Empty,
                    DecodeAudio = parameters.GenerateAudio,
                    FamilyRequest = parameters,
                    Progress = progress => Report(parameters, progress, clock),
                }, parameters.CancellationToken);
                settings = ((Ltx25Job)result.Job).Settings;
                if (StopCompanionPrefetch() is string prefetched) Console.WriteLine($"  [ltx25] companion prefetch: {prefetched}");
                Report(parameters, "done", $"{settings.Width}x{settings.Height}, {settings.Frames} frames at {settings.Fps} fps", null, null, clock);
                return new GeneratedVideo
                {
                    Frames = result.Frames,
                    Fps = settings.Fps,
                    Seed = settings.Seed,
                    Audio = result.Audio == null ? null : new GeneratedVideoAudio { Channels = result.Audio, SampleRate = result.AudioSampleRate },
                };
            }
            finally
            {
                own?.Dispose();
                StopCompanionPrefetch();
                TrimDevicePools();
            }
        }
    }

    /// <summary>Return the free buffers of the model's devices to them (CUDA's pools trimmed after a sync, MLX's cache cleared):
    /// the first device's, and under a split every stage device's too, whose pools hold the denoiser's freed working buffers.</summary>
    internal void TrimDevicePools()
    {
        if (_allocator is CudaAllocator cuda) { cuda.Synchronize(); cuda.TrimPool(); }
        else if (_allocator is MlxAllocator) MlxBackend.ClearCache();
        if (_denoiserDevices == null) return;
        foreach (IAllocator device in _denoiserDevices)
            if (!ReferenceEquals(device, _allocator) && device is CudaAllocator stage) { stage.Synchronize(); stage.TrimPool(); }
    }

    /// <summary>The runtime's load: the denoiser, the text encoder and the VAE this model resolved (the upscaler, the diffusion
    /// decoder and the audio VAE travel with the family), and the denoiser's stage devices under a split.</summary>
    private DiffusionLoad Load() => new()
    {
        Backend = DirectBackend(_backend),
        DenoiserDevices = _denoiserDevices,
        Files = new Dictionary<string, string>
        {
            [DiffusionLoad.Denoiser] = CheckpointPath,
            [DiffusionLoad.TextEncoder] = TextEncoderPath,
            [DiffusionLoad.Vae] = VideoVaePath,
        },
    };

    // ---- progress, as the hosts' video contract has always read LTX's: text-encode, (reference-encode, image-encode), denoise
    // with its stages and steps, vae-decode, audio-decode, done.

    private Stopwatch _clipClock;

    /// <summary>A family report (a phase entered, or a stage of the denoise phase), in the hosts' phase names.</summary>
    internal void Report(VideoGenerationParams p, string phase, string detail, string stage, IReadOnlyDictionary<string, object> attributes)
        => Report(p, phase, detail, stage, attributes, _clipClock);

    private int _reportedSteps, _reportedTotal;

    private void Report(VideoGenerationParams p, string phase, string detail, string stage, IReadOnlyDictionary<string, object> attributes,
        Stopwatch clock)
    {
        p.CancellationToken.ThrowIfCancellationRequested();
        Console.WriteLine($"  [ltx25] {phase}: {detail}");
        p.OnProgress?.Invoke(new VideoGenerationProgress
        {
            Phase = phase, Detail = detail, Step = _reportedSteps, TotalSteps = _reportedTotal,
            ElapsedSeconds = clock?.Elapsed.TotalSeconds ?? 0, Stage = stage, Attributes = attributes,
        });
    }

    /// <summary>The pipeline's reports: its steps as the hosts' "denoise" steps (and the step callback); the phases themselves
    /// are the family's reports.</summary>
    private void Report(VideoGenerationParams p, DiffusionVideoProgress progress, Stopwatch clock)
    {
        _clipClock = clock;
        if (progress.Phase == DiffusionPhase.Prepare)
        {
            _reportedSteps = 0;
            _reportedTotal = 0;
            return;
        }
        if (progress.Total > 0) _reportedTotal = progress.Total;
        if (!progress.IsStep) return;
        _reportedSteps = progress.Step;
        p.OnStep?.Invoke(progress.Step, progress.Total);
        Report(p, "denoise", $"step {progress.Step}/{progress.Total}", progress.Stage, null, clock);
    }

    internal enum CompanionFile { Upscaler, VideoDecoder, AudioDecoder }

    /// <summary>Read a companion file a later stage will open into the file cache in the
    /// background, one stage ahead of its use. Each is otherwise read through page faults
    /// when its network is first built, which on an A40 pod's network volume was most of
    /// 4.4 s of audio decode. Not earlier: pages read once are the first a memory-limited
    /// container evicts, and a prefetch at the first denoise step was gone again by the
    /// audio decode. Networks this process already holds are skipped.</summary>
    internal void PrefetchCompanion(CompanionFile companion)
    {
        var video = _pipeline?.VideoCodecOrNull as Ltx25VideoCodec;
        var audio = _pipeline?.AudioCodecOrNull as Ltx25AudioCodec;
        string path = companion switch
        {
            CompanionFile.Upscaler when video?.HasUpsampler != true => LatentUpscalerPath,
            CompanionFile.VideoDecoder when video?.HasDecoder != true => UsesDiffusionDecoder ? DiffusionVaePath : VideoVaePath,
            CompanionFile.AudioDecoder when audio?.HasDecoder != true => AudioVaePath,
            _ => null,
        };
        if (path != null && File.Exists(path) && MappedFileReadAhead.TryStart(path, MappedFileReadAhead.WholeFile(path),
                windowBytes: long.MaxValue) is { } read)
            _companionReads.Add(read);
    }

    /// <summary>Stops any companion prefetch and describes what it read, or returns null.</summary>
    internal string StopCompanionPrefetch()
    {
        if (_companionReads.Count == 0) return null;
        foreach (var read in _companionReads) read.Stop();
        string description = string.Join(", ", _companionReads.Select(read => read.Describe()));
        foreach (var read in _companionReads) read.Dispose();
        _companionReads.Clear();
        return description;
    }

    public override void TrimIdleMemory()
    {
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
        {
            MlxWorker.Shared.Invoke(TrimIdleMemory);
            return;
        }
        lock (_generationLock)
        {
            _pipeline?.Dispose();
            _pipeline = null;
            base.TrimIdleMemory();
            TrimDevicePools();
        }
    }

    public string VideoModelFamily => ArchitectureId;
    public int DefaultPreviewFrames => 17;
    public int FrameCountMultiple => 8;
    public int FrameCountOffset => 1;
    public int DefaultPreviewFps => 24;
    public int DefaultOutputWidth => Ltx25GenerationSettings.DefaultSize;
    public int DefaultOutputHeight => Ltx25GenerationSettings.DefaultSize;
    public int ResolutionMultiple => 64;
    public int MinimumReferenceFrames => Ltx25ReferenceConditioning.MinimumFrames;
    public bool SupportsAudio => true;
    public bool SupportsImageConditioning => MediaCodecs.VideoEncoder.SupportsH264ImageConditioning;
    public bool SupportsEndImageConditioning => SupportsImageConditioning;
    public bool SupportsKeyframeConditioning => SupportsImageConditioning;
    public bool SupportsReferenceConditioning => ReferenceLoraPath != null;

    /// <summary>
    /// Attach the Ingredients adapter to this loaded model without rebuilding the
    /// checkpoint. Returns false when a generation currently owns the model; Studio
    /// can retry on its next capability refresh.
    /// </summary>
    public bool TryLoadIngredientsAdapter(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath)) throw new FileNotFoundException("The configured LTX Ingredients adapter does not exist.", fullPath);
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread && ReferenceLoraPath is { } currentPath &&
            !string.Equals(currentPath, fullPath, StringComparison.Ordinal))
        {
            // Replacing an adapter releases its MLX copies. First reject a live generation without waiting behind its worker;
            // otherwise perform the replacement on that worker.
            if (!Monitor.TryEnter(_generationLock)) return false;
            Monitor.Exit(_generationLock);
            return MlxWorker.Shared.Invoke(() => TryLoadIngredientsAdapter(fullPath));
        }
        if (!Monitor.TryEnter(_generationLock)) return false;
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (string.Equals(ReferenceLoraPath, fullPath, StringComparison.Ordinal)) return true;
            Ltx25LoraSet loaded = Ltx25LoraSet.Open(fullPath, _gguf, Ltx25Config.FromGguf(_gguf));
            Ltx25LoraSet previous = _ingredients;
            (_pipeline?.LoadedDenoiser as Ltx25Denoiser)?.ReleaseLora();
            Volatile.Write(ref _ingredients, loaded);
            Volatile.Write(ref _referenceLoraPath, fullPath);
            previous?.Dispose();
            return true;
        }
        finally { Monitor.Exit(_generationLock); }
    }

    /// <summary>Attach a configured or conventional sibling adapter when it has appeared after model load.</summary>
    public bool TryLoadIngredientsAdapterIfPresent()
    {
        string configured = Environment.GetEnvironmentVariable("TS_LTX25_INGREDIENTS_LORA");
        string path = !string.IsNullOrWhiteSpace(configured)
            ? configured
            : Path.Combine(Path.GetDirectoryName(CheckpointPath)!, "loras", IngredientsFileName);
        return File.Exists(path) && TryLoadIngredientsAdapter(path);
    }

    public int MaxReferenceImages => Ltx25ImageConditioning.MaxImages;
    protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException("LTX is a video-generation model; use GenerateVideo().");
    protected override void ResetKVCacheCore() { }
    public override void WarmUpKernels() { }
    public override void Dispose()
    {
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
        {
            MlxWorker.Shared.Invoke(Dispose);
            return;
        }
        lock (_generationLock)
        {
            if (_disposed) return;
            _disposed = true;
            _pipeline?.Dispose();
            _pipeline = null;
            _ingredients?.Dispose();
            _ingredients = null;
            base.Dispose();
        }
    }
}
