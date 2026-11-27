// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LTX 2.5's components on the diffusion runtime (IResidentComponent): the prompt encoder (Gemma 4 and the learned AV
// connectors, with the model's small prompt cache), the video codec (the convolutional video VAE's encoder and decoder, the
// publisher's diffusion decoder when selected, and the x2 spatial latent upscaler, which normalizes with the VAE's statistics)
// and the audio codec (the audio VAE, vocoder and bandwidth extension). Each makes its networks on first use and keeps them
// between clips, as the model kept them; device memory goes where it went before (the connectors' weights after each new
// prompt, the denoiser's before each decode).
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.Ltx25;

/// <summary>The prompt encoder and its cache of the last four prompts' contexts.</summary>
internal sealed class Ltx25Conditioner : IConditioner
{
    private const int CachedPrompts = 4;
    private readonly string _textEncoder, _checkpoint;
    private readonly IAllocator _allocator;
    private readonly Dictionary<string, (Ltx25PromptEncoding Encoding, LinkedListNode<string> Node)> _cache = new(StringComparer.Ordinal);
    private readonly LinkedList<string> _order = new();
    private Ltx25PromptEncoder _encoder;
    private bool _disposed;

    public Ltx25Conditioner(string textEncoder, string checkpoint, IAllocator allocator)
    {
        _textEncoder = textEncoder ?? throw new ArgumentNullException(nameof(textEncoder));
        _checkpoint = checkpoint ?? throw new ArgumentNullException(nameof(checkpoint));
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
    }

    public long CacheHits { get; private set; }

    /// <summary>Whether <see cref="Encode(string, CancellationToken)"/> would answer <paramref name="prompt"/> from the cache.</summary>
    public bool HasCached(string prompt) => prompt != null && _cache.ContainsKey(prompt);

    /// <summary>The prompt's video and audio contexts: from the cache, or Gemma's 49 hidden states through the learned
    /// aggregation and the two connectors (whose device weights are released after each new prompt, before the denoise).</summary>
    public Ltx25PromptEncoding Encode(string prompt, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();
        if (_cache.TryGetValue(prompt, out var cached))
        {
            _order.Remove(cached.Node);
            _order.AddLast(cached.Node);
            ++CacheHits;
            return cached.Encoding;
        }
        Ltx25PromptEncoding encoding;
        var encoder = _encoder ??= new Ltx25PromptEncoder(_textEncoder, _checkpoint, _allocator);
        long cudaMapped = encoder.CudaMappedProjections,
            cudaBf16 = encoder.CudaBf16Projections, connectorProjections = encoder.CudaConnectorProjections,
            connectorAttentions = encoder.CudaConnectorAttentions;
        try
        {
            try
            {
                encoding = encoder.Encode(prompt, cancellationToken);
                Console.WriteLine($"  [ltx25-conditioning] gemma={encoder.LastGemmaMilliseconds:F3}ms aggregate={encoder.LastAggregationMilliseconds:F3}ms " +
                    $"connectors={encoder.LastConnectorMilliseconds:F3}ms " +
                    $"cuda_mapped={encoder.CudaMappedProjections - cudaMapped} cuda_bf16={encoder.CudaBf16Projections - cudaBf16} " +
                    $"cuda_connector_projections={encoder.CudaConnectorProjections - connectorProjections} " +
                    $"cuda_connector_attentions={encoder.CudaConnectorAttentions - connectorAttentions}" +
                    (encoder.LastReadAhead == null ? "" : " " + encoder.LastReadAhead));
            }
            finally { encoder.ReleaseDeviceResidency(); }
        }
        catch
        {
            DisposeEncoder();
            throw;
        }
        cancellationToken.ThrowIfCancellationRequested();
        _cache.Add(prompt, (encoding, _order.AddLast(prompt)));
        if (_cache.Count > CachedPrompts)
        {
            _cache.Remove(_order.First.Value);
            _order.RemoveFirst();
        }
        return encoding;
    }

    public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken cancellationToken)
    {
        Ltx25PromptEncoding encoding = Encode(prompt, cancellationToken);
        return new ConditioningSet(encoding.VideoContext, encoding.Tokens, encoding.VideoContext.Length / encoding.Tokens)
        {
            FamilyState = encoding,
        };
    }

    /// <summary>The encoder streams its projections from the mapping and releases its connectors after each prompt: what it
    /// keeps on the device between phases is small.</summary>
    public long WeightBytes => 0;

    public long WorkingBytes(DiffusionWorkload workload) => 2L << 30;

    /// <summary>The connectors' device weights (the encoder releases them itself after every new prompt).</summary>
    public void ReleaseDevice()
    {
        if (!_disposed) _encoder?.ReleaseDeviceResidency();
    }

    public void ReleaseWorkspace() { }

    /// <summary>The cache and the encoder, mappings and all (the model's TrimIdleMemory).</summary>
    public void Trim()
    {
        _cache.Clear();
        _order.Clear();
        DisposeEncoder();
    }

    private void DisposeEncoder()
    {
        var previous = _encoder;
        _encoder = null;
        previous?.Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Trim();
    }
}

/// <summary>The video codec: the VAE's encoder and decoder (convolutional, or the diffusion decoder when selected) and the x2
/// spatial latent upscaler the two passes meet at.</summary>
internal sealed class Ltx25VideoCodec : IResidentComponent
{
    private readonly string _vae, _diffusionVae, _upscaler;
    private readonly IAllocator _allocator;
    private Ltx25VideoVaeEncoder _encoder;
    private Ltx25VideoVaeDecoder _decoder;
    private Ltx25DiffusionVideoDecoder _diffusionDecoder;
    private Ltx25LatentUpsampler _upsampler;
    private (float[] Mean, float[] Std)? _statistics;
    private bool _disposed;

    public Ltx25VideoCodec(string vae, string diffusionVae, string upscaler, IAllocator allocator)
    {
        _vae = vae ?? throw new ArgumentNullException(nameof(vae));
        _diffusionVae = diffusionVae;
        _upscaler = upscaler ?? throw new ArgumentNullException(nameof(upscaler));
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
    }

    public bool UsesDiffusionDecoder => _diffusionVae != null;
    public bool HasUpsampler => _upsampler != null;
    public bool HasDecoder => UsesDiffusionDecoder ? _diffusionDecoder != null : _decoder != null;

    public float[] EncodeImage(float[] rgb, int height, int width, CancellationToken ct)
    {
        var encoder = _encoder ??= new Ltx25VideoVaeEncoder(_vae, _allocator);
        try { return encoder.EncodeImage(rgb, height, width, ct); }
        catch { Dispose(ref _encoder); throw; }
    }

    public float[] EncodeReferenceVideo(float[] rgb, int frames, int height, int width, CancellationToken ct)
    {
        var encoder = _encoder ??= new Ltx25VideoVaeEncoder(_vae, _allocator);
        try { return encoder.Encode(rgb, frames, height, width, ct); }
        catch { Dispose(ref _encoder); throw; }
    }

    public float[] EncodeStaticReferenceVideo(float[] rgbFrame, int frames, int height, int width, CancellationToken ct)
    {
        var encoder = _encoder ??= new Ltx25VideoVaeEncoder(_vae, _allocator);
        try { return encoder.EncodeStaticVideo(rgbFrame, frames, height, width, ct); }
        catch { Dispose(ref _encoder); throw; }
    }

    /// <summary>Stage-1 to stage-2 spatial upsampling of channel-major normalized [128,T,H,W] latents, normalized with the video
    /// VAE's per-channel statistics.</summary>
    public float[] Upsample(float[] latents, int frames, int height, int width, CancellationToken ct)
    {
        if (_statistics == null)
        {
            using var vae = new SafetensorsFile(_vae);
            _statistics = (vae.ReadFloat32("per_channel_statistics.mean-of-means"), vae.ReadFloat32("per_channel_statistics.std-of-means"));
        }
        var (mean, std) = _statistics.Value;
        var upsampler = _upsampler ??= new Ltx25LatentUpsampler(_upscaler, _allocator);
        try
        {
            if (upsampler.LatentChannels != 128 || mean.Length != 128 || std.Length != 128)
                throw new InvalidDataException("The LTX 2.5 spatial upsampler and video VAE must use 128 latent channels.");
            return upsampler.Upsample(latents, frames, height, width, mean, std, ct);
        }
        catch { Dispose(ref _upsampler); throw; }
    }

    public float[] Decode(float[] latents, int frames, int height, int width, long seed, CancellationToken ct)
    {
        if (UsesDiffusionDecoder)
        {
            var diffusion = _diffusionDecoder ??= new Ltx25DiffusionVideoDecoder(_diffusionVae, _allocator);
            try { return diffusion.Decode(latents, frames, height, width, seed, ct); }
            catch { Dispose(ref _diffusionDecoder); throw; }
        }
        var decoder = _decoder ??= new Ltx25VideoVaeDecoder(_vae, _allocator);
        try { return decoder.Decode(latents, frames, height, width, ct); }
        catch { Dispose(ref _decoder); throw; }
    }

    /// <summary>The checkpoints' bytes as their networks hold them (BF16 widened to F32: twice the files').</summary>
    public long WeightBytes => 2 * (Size(_vae) + Size(_upscaler) + (_diffusionVae != null ? Size(_diffusionVae) : 0));

    public long WorkingBytes(DiffusionWorkload workload)
    {
        // A decode's activations: the full-size frames at the decoder's widest stage, F32.
        long cells = workload.Target.Frames * (long)workload.Target.Height * workload.Target.Width;
        return Math.Max(4L << 30, cells * 1024 * 8 * sizeof(float));
    }

    private static long Size(string path)
    {
        try { return new FileInfo(path).Length; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return 0; }
    }

    /// <summary>The encoder's device weights; the decoders and the upscaler (which keep theirs while they live) are dropped
    /// and made again at their next use.</summary>
    public void ReleaseDevice()
    {
        if (_disposed) return;
        _encoder?.ReleaseDeviceResidency();
        Dispose(ref _decoder);
        Dispose(ref _diffusionDecoder);
        Dispose(ref _upsampler);
    }

    public void ReleaseWorkspace() { }

    /// <summary>Every network (the model's TrimIdleMemory).</summary>
    public void Trim()
    {
        Dispose(ref _encoder);
        Dispose(ref _decoder);
        Dispose(ref _diffusionDecoder);
        Dispose(ref _upsampler);
    }

    private static void Dispose<T>(ref T component) where T : class, IDisposable
    {
        var previous = component;
        component = null;
        previous?.Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Trim();
    }
}

/// <summary>The audio codec: the audio VAE's decoder, the BigVGAN vocoder and the 48 kHz bandwidth extension.</summary>
internal sealed class Ltx25AudioCodec : IResidentComponent
{
    private readonly string _path;
    private readonly IAllocator _allocator;
    private Ltx25AudioDecoder _decoder;
    private bool _disposed;

    public Ltx25AudioCodec(string path, IAllocator allocator)
    {
        _path = path ?? throw new ArgumentNullException(nameof(path));
        _allocator = allocator;
    }

    public bool HasDecoder => _decoder != null;

    public GeneratedVideoAudio Decode(float[] latents, int tokens, int samples, CancellationToken ct)
    {
        var decoder = _decoder ??= new Ltx25AudioDecoder(_path, _allocator);
        try { return decoder.Decode(latents, tokens, samples, ct); }
        catch
        {
            var previous = _decoder;
            _decoder = null;
            previous?.Dispose();
            throw;
        }
    }

    public long WeightBytes
    {
        get
        {
            try { return 2 * new FileInfo(_path).Length; }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return 0; }
        }
    }

    public long WorkingBytes(DiffusionWorkload workload) => 1L << 30;

    public void ReleaseDevice() { }

    public void ReleaseWorkspace() { }

    public void Trim()
    {
        var previous = _decoder;
        _decoder = null;
        previous?.Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Trim();
    }
}
