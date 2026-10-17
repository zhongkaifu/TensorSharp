using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Media;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;

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
    internal string ReferenceLoraPath { get; }
    /// <summary>The x2 spatial latent upscaler between the half- and full-size passes.</summary>
    internal string LatentUpscalerPath { get; }
    internal IAllocator DirectAllocator => _allocator;
    private readonly object _generationLock = new();
    private readonly Dictionary<string, (Ltx25PromptEncoding Encoding, LinkedListNode<string> Node)> _promptCache = new(StringComparer.Ordinal);
    private readonly LinkedList<string> _promptOrder = new();
    private Ltx25Denoiser _denoiser;
    private Ltx25PromptEncoder _promptEncoder;
    private Ltx25VideoVaeDecoder _videoDecoder;
    private Ltx25DiffusionVideoDecoder _diffusionDecoder;
    private Ltx25VideoVaeEncoder _videoEncoder;
    private Ltx25AudioDecoder _audioDecoder;
    private Ltx25LatentUpsampler _latentUpsampler;
    public long PromptCacheHits { get; private set; }
    private bool _disposed;

    public Ltx25Model(string checkpoint, BackendType backend) : base(checkpoint, DirectBackend(backend))
    {
        try
        {
            var geometry = Ltx25Config.FromGguf(_gguf);
            if (geometry.VideoDim != 4096 || geometry.AudioDim != 2048 || geometry.Layers != 48 ||
                geometry.VideoChannels != 128 || geometry.AudioChannels != 128)
                throw new NotSupportedException("LTX generation requires the authentic 48-layer LTX 2.5 AV checkpoint.");
            CheckpointPath = Path.GetFullPath(checkpoint);
            string directory = Path.GetDirectoryName(CheckpointPath);
            TextEncoderPath = Companion(directory, "TS_VIDEO_TEXT_ENCODER", "text_encoders", "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors");
            VideoVaePath = Companion(directory, "TS_VIDEO_VAE", "vae", "ltx-2.5-video-vae-conv-bf16.safetensors");
            AudioVaePath = Companion(directory, "TS_VIDEO_AUDIO_VAE", "vae", "ltx-2.5-audio-vae-bf16.safetensors");
            LatentUpscalerPath = Companion(directory, "TS_VIDEO_UPSCALER", "latent_upscale_models", LatentUpscalerFileName);
            DiffusionVaePath = ResolveDiffusionVae(VideoVaePath, _allocator is CudaAllocator);
            Console.WriteLine($"  [ltx25] video decoder: {(UsesDiffusionDecoder ? "diffusion" : "convolutional")} " +
                $"({Path.GetFileName(DiffusionVaePath ?? VideoVaePath)})");
            string configuredReference = Environment.GetEnvironmentVariable("TS_LTX25_INGREDIENTS_LORA");
            if (!string.IsNullOrWhiteSpace(configuredReference))
            {
                if (!File.Exists(configuredReference)) throw new FileNotFoundException("The configured LTX Ingredients adapter does not exist.", configuredReference);
                ReferenceLoraPath = Path.GetFullPath(configuredReference);
            }
            else
            {
                string reference = Path.Combine(directory, "loras", "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors");
                if (File.Exists(reference)) ReferenceLoraPath = reference;
            }
            Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = 0 };
        }
        catch { base.Dispose(); throw; }
    }

    internal const string LatentUpscalerFileName = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";

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

    // GGML option names remain usable by hosts whose device picker uses them;
    // this family executes its owned direct CUDA/MLX kernels, not a GGML graph.
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

    public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams parameters = null)
    {
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
            return MlxWorker.Shared.Invoke(() => GenerateVideo(prompt, parameters));
        lock (_generationLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Ltx25Pipeline.Generate(this, prompt, parameters ?? new VideoGenerationParams());
        }
    }

    internal Ltx25PromptEncoding Conditioning(string prompt, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_promptCache.TryGetValue(prompt, out var cached))
        {
            _promptOrder.Remove(cached.Node);
            _promptOrder.AddLast(cached.Node);
            ++PromptCacheHits;
            return cached.Encoding;
        }
        Ltx25PromptEncoding encoding;
        // Keep the large mapped files and tokenizer across distinct prompts, but
        // release the connector's large device weights before denoising.
        var encoder = _promptEncoder ??= new Ltx25PromptEncoder(TextEncoderPath, CheckpointPath, _allocator);
        long mlxBf16 = encoder.NativeBf16Projections, cudaMapped = encoder.CudaMappedProjections,
            cudaBf16 = encoder.CudaBf16Projections, connectorProjections = encoder.CudaConnectorProjections,
            connectorAttentions = encoder.CudaConnectorAttentions;
        try
        {
            try
            {
                encoding = encoder.Encode(prompt, cancellationToken);
                Console.WriteLine($"  [ltx25-conditioning] gemma={encoder.LastGemmaMilliseconds:F3}ms aggregate={encoder.LastAggregationMilliseconds:F3}ms " +
                    $"connectors={encoder.LastConnectorMilliseconds:F3}ms mlx_bf16={encoder.NativeBf16Projections - mlxBf16} " +
                    $"cuda_mapped={encoder.CudaMappedProjections - cudaMapped} cuda_bf16={encoder.CudaBf16Projections - cudaBf16} " +
                    $"cuda_connector_projections={encoder.CudaConnectorProjections - connectorProjections} " +
                    $"cuda_connector_attentions={encoder.CudaConnectorAttentions - connectorAttentions}");
            }
            finally { encoder.ReleaseDeviceResidency(); }
        }
        catch { DisposeOwned(ref _promptEncoder); throw; }
        cancellationToken.ThrowIfCancellationRequested();
        _promptCache.Add(prompt, (encoding, _promptOrder.AddLast(prompt)));
        if (_promptCache.Count > 4)
        {
            _promptCache.Remove(_promptOrder.First.Value);
            _promptOrder.RemoveFirst();
        }
        return encoding;
    }

    internal Ltx25SamplingOutput Sample(Ltx25DenoiserInput input, Ltx25GenerationSettings settings, Action<int, int> onStep,
        CancellationToken cancellationToken = default, float[] sigmas = null,
        long noiseSeedOffset = Ltx25GenerationSettings.AncestralNoiseSeedOffset)
    {
        sigmas ??= Ltx25Scheduler.DistilledSigmas();
        ArgumentNullException.ThrowIfNull(settings);
        // Reuse device weights across both sampling stages. Mapped checkpoint and
        // descriptor identities survive across requests; DecodeVideo releases their
        // device representation to leave room for full-resolution VAE activations.
        // The serialized generation lock also protects this single device model.
        _denoiser ??= new Ltx25Denoiser(CheckpointPath, _allocator, ReferenceLoraPath);
        try
        {
            void Progress(int step, int total)
            {
                cancellationToken.ThrowIfCancellationRequested();
                onStep?.Invoke(step, total);
            }
            return settings.Ancestral
                ? Ltx25Sampling.RunEulerAncestral(_denoiser, input, sigmas,
                    step => settings.AncestralNoise(step, input.Video.Length, input.Audio.Length, noiseSeedOffset), Progress, cancellationToken)
                : Ltx25Sampling.RunEuler(_denoiser, input, sigmas, Progress, cancellationToken);
        }
        catch { DisposeOwned(ref _denoiser); throw; }
    }

    internal float[] DecodeVideo(float[] latents, int frames, int height, int width, long seed,
        CancellationToken cancellationToken = default)
    {
        // Denoising is finished. Its live quantized weights and position tables
        // otherwise remain resident beside the VAE's full-resolution activations;
        // ClearCache alone only frees inactive allocator buffers. Retain the mapped
        // checkpoint and descriptors so the next request can rebind the same weights.
        _denoiser?.ReleaseDeviceResidency();
        if (_allocator is MlxAllocator) MlxBackend.ClearCache();
        else if (_allocator is CudaAllocator cuda) { cuda.Synchronize(); cuda.TrimPool(); }
        if (UsesDiffusionDecoder)
        {
            var diffusion = _diffusionDecoder ??= new Ltx25DiffusionVideoDecoder(DiffusionVaePath, _allocator);
            try { return diffusion.Decode(latents, frames, height, width, seed, cancellationToken); }
            catch { DisposeOwned(ref _diffusionDecoder); throw; }
        }
        var decoder = _videoDecoder ??= new Ltx25VideoVaeDecoder(VideoVaePath, _allocator);
        try { return decoder.Decode(latents, frames, height, width, cancellationToken); }
        catch { DisposeOwned(ref _videoDecoder); throw; }
    }

    /// <summary>Stage-1 to stage-2 spatial upsampling of channel-major normalized
    /// [128,T,H,W] latents, normalized with the video VAE's per-channel statistics.</summary>
    internal float[] UpsampleLatents(float[] latents, int frames, int height, int width, CancellationToken cancellationToken = default)
    {
        float[] mean, std;
        using (var vae = new SafetensorsFile(VideoVaePath))
        {
            mean = vae.ReadFloat32("per_channel_statistics.mean-of-means");
            std = vae.ReadFloat32("per_channel_statistics.std-of-means");
        }
        var upsampler = _latentUpsampler ??= new Ltx25LatentUpsampler(LatentUpscalerPath, _allocator);
        try
        {
            if (upsampler.LatentChannels != 128 || mean.Length != 128 || std.Length != 128)
                throw new InvalidDataException("The LTX 2.5 spatial upsampler and video VAE must use 128 latent channels.");
            return upsampler.Upsample(latents, frames, height, width, mean, std, cancellationToken);
        }
        catch { DisposeOwned(ref _latentUpsampler); throw; }
    }

    internal float[] EncodeImage(float[] rgb, int height, int width, CancellationToken cancellationToken = default)
    {
        var encoder = _videoEncoder ??= new Ltx25VideoVaeEncoder(VideoVaePath, _allocator);
        try { return encoder.EncodeImage(rgb, height, width, cancellationToken); }
        catch { DisposeOwned(ref _videoEncoder); throw; }
    }

    internal float[] EncodeReferenceVideo(float[] rgb, int frames, int height, int width, CancellationToken cancellationToken = default)
    {
        var encoder = _videoEncoder ??= new Ltx25VideoVaeEncoder(VideoVaePath, _allocator);
        try { return encoder.Encode(rgb, frames, height, width, cancellationToken); }
        catch { DisposeOwned(ref _videoEncoder); throw; }
    }

    internal float[] EncodeStaticReferenceVideo(float[] rgbFrame, int frames, int height, int width, CancellationToken cancellationToken = default)
    {
        var encoder = _videoEncoder ??= new Ltx25VideoVaeEncoder(VideoVaePath, _allocator);
        try { return encoder.EncodeStaticVideo(rgbFrame, frames, height, width, cancellationToken); }
        catch { DisposeOwned(ref _videoEncoder); throw; }
    }

    internal GeneratedVideoAudio DecodeAudio(float[] latents, int tokens, int samples,
        CancellationToken cancellationToken = default)
    {
        var decoder = _audioDecoder ??= new Ltx25AudioDecoder(AudioVaePath, _allocator);
        try { return decoder.Decode(latents, tokens, samples, cancellationToken); }
        catch { DisposeOwned(ref _audioDecoder); throw; }
    }

    private static void DisposeOwned<T>(ref T component) where T : class, IDisposable
    {
        var previous = component;
        component = null;
        previous?.Dispose();
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
            _promptCache.Clear(); _promptOrder.Clear();
            DisposeOwned(ref _promptEncoder);
            DisposeOwned(ref _videoDecoder);
            DisposeOwned(ref _diffusionDecoder);
            DisposeOwned(ref _videoEncoder);
            DisposeOwned(ref _audioDecoder);
            DisposeOwned(ref _latentUpsampler);
            DisposeOwned(ref _denoiser);
            base.TrimIdleMemory();
            if (_allocator is MlxAllocator) MlxBackend.ClearCache();
            else if (_allocator is CudaAllocator cuda) { cuda.Synchronize(); cuda.TrimPool(); }
        }
    }

    public string VideoModelFamily => ArchitectureId;
    public int DefaultPreviewFrames => 17;
    public bool SupportsAudio => true;
    public bool SupportsImageConditioning => MediaCodecs.VideoEncoder.SupportsH264ImageConditioning;
    public bool SupportsEndImageConditioning => SupportsImageConditioning;
    public bool SupportsKeyframeConditioning => SupportsImageConditioning;
    public bool SupportsReferenceConditioning => ReferenceLoraPath != null;
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
            _promptCache.Clear();
            _promptOrder.Clear();
            DisposeOwned(ref _promptEncoder);
            DisposeOwned(ref _videoDecoder);
            DisposeOwned(ref _diffusionDecoder);
            DisposeOwned(ref _videoEncoder);
            DisposeOwned(ref _audioDecoder);
            DisposeOwned(ref _latentUpsampler);
            DisposeOwned(ref _denoiser);
            base.Dispose();
        }
    }
}
