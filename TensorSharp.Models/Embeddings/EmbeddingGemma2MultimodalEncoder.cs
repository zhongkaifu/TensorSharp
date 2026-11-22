// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.GGML;
using TensorSharp.Cpu;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Embeddings;

/// <summary>An ordered text fragment or local media file in one embedding input.</summary>
public sealed record EmbeddingGemma2InputPart(string Type, string Value)
{
    public static EmbeddingGemma2InputPart Text(string text) => new("text", text);
    public static EmbeddingGemma2InputPart Image(string path) => new("image", path);
    public static EmbeddingGemma2InputPart Audio(string path) => new("audio", path);
    public static EmbeddingGemma2InputPart Video(string path) => new("video", path);
}

public sealed class EmbeddingGemma2MultimodalOptions
{
    public bool LoadVision { get; init; } = true;
    public bool LoadAudio { get; init; } = true;
    public int ImageSoftTokens { get; init; } = 280;
    public int VideoSoftTokens { get; init; } = 140;
    public double VideoFramesPerSecond { get; init; } = 1;
    public int MaxVideoFrames { get; init; } = 32;
    public bool IncludeVideoTimestamps { get; init; }
    /// <summary>Optional text before sampled frames; empty matches Hugging Face, "Video:" matches llama.cpp's video helper.</summary>
    public string VideoPrefix { get; init; } = "";
    /// <summary>The trained Hugging Face frontend is the default. The clamp
    /// option is intended for explicitly controlled llama.cpp comparisons.</summary>
    public Gemma4AudioLogPolicy AudioLogPolicy { get; init; } = Gemma4AudioLogPolicy.HuggingFaceAddFloor;
}

/// <summary>
/// EmbeddingGemma 2 text, image, audio and video encoding with a shared input window.
/// Media parts are expanded in order; task prefixes belong in the text parts.
/// </summary>
public sealed class EmbeddingGemma2MultimodalEncoder : IEmbeddingModel
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly EmbeddingGemma2MultimodalOptions _options;
    private readonly Gemma4AudioLogPolicy _audioLogPolicy;
    private readonly ITokenizer _tokenizer;
    private readonly GgmlContext _context;
    private readonly Gemma4VisionEncoder _vision;
    private readonly Gemma4AudioEncoder _audio;
    private readonly Gemma4ImageProcessor _imageProcessor;
    private readonly int _imageStart, _imageEnd, _audioStart, _audioEnd;
    private bool _disposed;

    public EmbeddingModel TextModel { get; }
    public string ModelName => TextModel.ModelName;
    public string Architecture => TextModel.Architecture;
    public int Dimensions => TextModel.Dimensions;
    public int MaxTokens => TextModel.MaxTokens;
    public int VocabularySize => TextModel.VocabularySize;
    public bool SupportsImages => _vision != null;
    public bool SupportsAudio => _audio != null;
    public bool SupportsVideo => _vision != null;
    public bool HasVision => _vision != null;
    public bool HasAudio => _audio != null;
    public Gemma4AudioLogPolicy AudioLogPolicy => _audioLogPolicy;

    public int[] Tokenize(string text, bool truncate = false) => TextModel.Tokenize(text, truncate);

    public Task<EmbeddingBatchResult> EmbedTokensAsync(IReadOnlyList<int[]> inputs,
        CancellationToken cancellationToken = default) => TextModel.EmbedTokensAsync(inputs, cancellationToken);

    private EmbeddingGemma2MultimodalEncoder(string modelPath, string mmprojPath,
        EmbeddingModelOptions modelOptions, EmbeddingGemma2MultimodalOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mmprojPath);
        if (!File.Exists(mmprojPath)) throw new FileNotFoundException("Multimodal projector file was not found.", mmprojPath);
        ValidateSoftTokens(options.ImageSoftTokens, nameof(options.ImageSoftTokens));
        ValidateSoftTokens(options.VideoSoftTokens, nameof(options.VideoSoftTokens));
        if (!double.IsFinite(options.VideoFramesPerSecond) || options.VideoFramesPerSecond <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.VideoFramesPerSecond));
        if (options.MaxVideoFrames is < 1 or > 128)
            throw new ArgumentOutOfRangeException(nameof(options.MaxVideoFrames), "Video frame limit must be between 1 and 128.");
        ArgumentNullException.ThrowIfNull(options.VideoPrefix);
        if (ContainsMediaDelimiter(options.VideoPrefix))
            throw new ArgumentException("Video prefix must not contain unmatched media delimiters.", nameof(options.VideoPrefix));
        if (!Enum.IsDefined(options.AudioLogPolicy)) throw new ArgumentOutOfRangeException(nameof(options.AudioLogPolicy));
        _audioLogPolicy = Environment.GetEnvironmentVariable("TS_EMBEDDING_AUDIO_LOG_POLICY") switch
        {
            null or "" => options.AudioLogPolicy,
            "llama_clamp" => Gemma4AudioLogPolicy.LlamaClampFloor,
            "huggingface_add" => Gemma4AudioLogPolicy.HuggingFaceAddFloor,
            _ => throw new ArgumentException("TS_EMBEDDING_AUDIO_LOG_POLICY must be huggingface_add or llama_clamp."),
        };
        _options = options;
        try
        {
            TextModel = EmbeddingModel.Load(modelPath, modelOptions);
            if (TextModel.Architecture != "gemma-embedding2")
                throw new NotSupportedException("This multimodal encoder requires an EmbeddingGemma 2 text model.");
            using (var file = new GgufFile(modelPath)) _tokenizer = EmbeddingTokenizer.Create(file);
            _imageStart = FindToken("<|image>");
            _imageEnd = FindToken("<image|>");
            _audioStart = FindToken("<|audio>");
            _audioEnd = FindToken("<audio|>");
            var allocator = TextModel.InputAllocator;
            if (allocator == null && TextModel.Backend == "CPU") allocator = new CpuAllocator(BlasEnum.DotNet);
            if (allocator == null)
            {
                var backend = TextModel.Backend switch
                {
                    "GGML_METAL" => GgmlBackendType.Metal,
                    "GGML_CUDA" => GgmlBackendType.Cuda,
                    _ => GgmlBackendType.Cpu,
                };
                _context = new GgmlContext([modelOptions.Device], backend);
                allocator = new GgmlAllocator(_context, modelOptions.Device);
            }
            using var projector = new GgufFile(mmprojPath);
            if (options.LoadVision)
            {
                if (projector.GetString("clip.vision.projector_type", "") != "gemma4v")
                    throw new InvalidDataException("EmbeddingGemma 2 requires a gemma4v vision projector.");
                _vision = new Gemma4VisionEncoder(mmprojPath, allocator, inputPixelsInZeroToOneRange: true,
                    requireValidatedFloat32Precision: true);
                if (_vision.ProjectionDim != TextModel.HiddenSize)
                    throw new InvalidDataException("Vision projector dimension does not match the text model.");
                _imageProcessor = new Gemma4ImageProcessor(maxTokens: options.ImageSoftTokens,
                    minTokens: options.ImageSoftTokens, imageMean: [0, 0, 0], imageStd: [1, 1, 1], referenceSizing: true);
            }
            if (options.LoadAudio)
            {
                if (projector.GetString("clip.audio.projector_type", "") != "gemma4a")
                    throw new InvalidDataException("EmbeddingGemma 2 requires a gemma4a audio projector.");
                _audio = new Gemma4AudioEncoder(mmprojPath, allocator, requireValidatedFloat32Precision: true);
                if (_audio.ProjectionDim != TextModel.HiddenSize)
                    throw new InvalidDataException("Audio projector dimension does not match the text model.");
            }
        }
        catch
        {
            _vision?.Dispose();
            _audio?.Dispose();
            TextModel?.Dispose();
            if (_context != null)
            {
                ReleaseDeviceMemory();
                _context.ReleasePoolForDisposal();
            }
            throw;
        }
    }

    public static EmbeddingGemma2MultimodalEncoder Load(string modelPath, string mmprojPath,
        EmbeddingModelOptions modelOptions = null, EmbeddingGemma2MultimodalOptions options = null)
        => new(modelPath, mmprojPath, modelOptions ?? new(), options ?? new());

    public async Task<EmbeddingBatchResult> EmbedAsync(IReadOnlyList<EmbeddingGemma2InputPart> parts,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(parts);
        if (parts.Count == 0) throw new ArgumentException("Provide at least one text or media part.", nameof(parts));
        var owned = new EmbeddingGemma2InputPart[parts.Count];
        for (int i = 0; i < owned.Length; ++i)
        {
            var part = parts[i] ?? throw new ArgumentException("Input parts must not be null.", nameof(parts));
            ArgumentNullException.ThrowIfNull(part.Value);
            if (part.Type != "text" && part.Type != "image" && part.Type != "audio" && part.Type != "video")
                throw new ArgumentException($"Unsupported embedding input part '{part.Type}'.", nameof(parts));
            if (part.Type == "text" && ContainsMediaDelimiter(part.Value))
                throw new ArgumentException("Supply image, audio, and video as media parts instead of unmatched placeholder tokens in text.", nameof(parts));
            if (part.Type != "text" && !File.Exists(part.Value))
                throw new FileNotFoundException("Embedding media file was not found.", part.Value);
            if ((part.Type == "image" || part.Type == "video") && _vision == null)
                throw new NotSupportedException("Load the vision encoder to embed images or video.");
            if (part.Type == "audio" && _audio == null)
                throw new NotSupportedException("Load the audio encoder to embed audio.");
            owned[i] = part;
        }
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var prepared = await Task.Run(() => Prepare(owned, cancellationToken), CancellationToken.None).ConfigureAwait(false);
            return await TextModel.EmbedInputsAsync(prepared, prepared.Length / TextModel.HiddenSize, cancellationToken).ConfigureAwait(false);
        }
        finally { _gate.Release(); }
    }

    private float[] Prepare(EmbeddingGemma2InputPart[] parts, CancellationToken cancellationToken)
    {
        var blocks = new List<float[]>();
        int tokens = 0;
        void CheckRoom(int count)
        {
            if (count <= 0 || tokens + count > TextModel.MaxTokens)
                throw new ArgumentException($"Input exceeds the shared {TextModel.MaxTokens}-token embedding context.", nameof(parts));
        }
        void AddTextTokens(int[] ids)
        {
            if (ids.Length == 0) return;
            CheckRoom(ids.Length);
            blocks.Add(TextModel.GetTokenEmbeddings(ids));
            tokens += ids.Length;
        }
        void AddMedia(Tensor embeddings, int start, int end, int validRows = 0)
        {
            if (embeddings.Sizes.Length != 2 || embeddings.Sizes[1] != TextModel.HiddenSize)
                throw new InvalidDataException("Media encoder returned invalid embedding dimensions.");
            int rows = checked((int)embeddings.Sizes[0]);
            int count = validRows == 0 ? rows : validRows;
            if (count == 0 || rows < count) throw new InvalidDataException("Media encoder returned invalid embedding dimensions.");
            CheckRoom(count + 2);
            AddTextTokens([start]);
            blocks.Add(embeddings.GetElementsAsFloat(checked(count * TextModel.HiddenSize)));
            tokens += count;
            AddTextTokens([end]);
        }
        AddTextTokens([_tokenizer.BosTokenId]);
        var text = new StringBuilder();
        void FlushText()
        {
            if (text.Length == 0) return;
            AddTextTokens(_tokenizer.Encode(text.ToString(), addSpecial: false).ToArray());
            text.Clear();
        }
        foreach (var part in parts)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (part.Type == "text") { text.Append(part.Value); continue; }
            FlushText();
            if (part.Type == "image")
            {
                var (pixels, width, height) = DecodeMedia(() => _imageProcessor.ProcessImage(part.Value), "image");
                CheckRoom(_imageProcessor.ComputeOutputTokens(width, height) + 2);
                using var embeddings = _vision.Encode(pixels, width, height);
                AddMedia(embeddings, _imageStart, _imageEnd);
            }
            else if (part.Type == "audio")
            {
                var samples = DecodeMedia(() => Gemma4AudioPreprocessor.DecodeAudioFile(part.Value), "audio");
                int validTokens = Gemma4AudioPreprocessor.ComputeAudioTokenCount(samples, padToMultiple: false);
                if (validTokens == 0) throw new InvalidDataException("Audio input is too short to produce a valid audio token.");
                if (samples.Length % 128 != 0) Array.Resize(ref samples, checked((samples.Length + 127) / 128 * 128));
                CheckRoom(validTokens + 2);
                var (mel, frames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples, _audioLogPolicy);
                if (mel == null || frames <= 0) throw new InvalidDataException("Audio input is too short to produce a mel spectrogram.");
                using var embeddings = _audio.Encode(mel, frames);
                AddMedia(embeddings, _audioStart, _audioEnd, validTokens);
            }
            else
            {
                if (_options.VideoPrefix.Length != 0)
                    AddTextTokens(_tokenizer.Encode(_options.VideoPrefix, addSpecial: false).ToArray());
                var info = DecodeMedia(() => MediaCodecs.Video.Probe(part.Value), "video");
                var indices = SampleVideoFrames(info, _options.VideoFramesPerSecond, _options.MaxVideoFrames);
                int received = 0;
                MediaCodecs.Video.ReadFrames(part.Value, indices, (index, pixels, width, height, stride, layout) =>
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (received >= indices.Length || index != indices[received])
                        throw new InvalidDataException("Video decoder returned frames out of order.");
                    if (_options.IncludeVideoTimestamps)
                    {
                        int seconds = (int)(index / info.Fps);
                        AddTextTokens(_tokenizer.Encode((received == 0 ? "" : " ") + $"{seconds / 60:00}:{seconds % 60:00} ", false).ToArray());
                    }
                    Gemma4ImageProcessor.CalcAspectRatioPreservingSize(width, height, 16, 3,
                        _options.VideoSoftTokens, out int resizedWidth, out int resizedHeight);
                    CheckRoom((resizedWidth / 48) * (resizedHeight / 48) + 2);
                    var rgba = FrameToRgba(pixels, width, height, stride, layout);
                    var normalized = ImageProcessorUtils.ResizeRgbaToChannelFirstBicubic(rgba, width, height,
                        resizedWidth, resizedHeight, [0, 0, 0], [1, 1, 1]);
                    using var embeddings = _vision.Encode(normalized, resizedWidth, resizedHeight);
                    AddMedia(embeddings, _imageStart, _imageEnd);
                    ++received;
                });
                if (received != indices.Length) throw new InvalidDataException("Video decoder did not return every sampled frame.");
            }
        }
        FlushText();
        AddTextTokens([_tokenizer.EosTokenIds[0]]);
        var result = new float[checked(tokens * TextModel.HiddenSize)];
        int offset = 0;
        foreach (var block in blocks) { block.CopyTo(result, offset); offset += block.Length; }
        return result;
    }

    internal static int[] SampleVideoFrames(VideoInfo info, double fps, int maxFrames)
    {
        ArgumentNullException.ThrowIfNull(info);
        if (!double.IsFinite(info.Fps) || info.Fps <= 0 || info.FrameCount <= 0)
            throw new InvalidDataException("Video requires a valid source frame rate and frame count.");
        if (!double.IsFinite(fps) || fps <= 0 || maxFrames is < 1 or > 128)
            throw new ArgumentOutOfRangeException(nameof(fps), "Requested frame rate and frame budget must be positive.");
        double sampled = Math.Max(1, Math.Floor(info.FrameCount / info.Fps * fps));
        if (sampled > int.MaxValue) throw new InvalidDataException("Video duration exceeds the supported frame index range.");
        int count = (int)sampled;
        int selected = Math.Min(count, maxFrames);
        var result = new int[selected];
        for (int i = 0; i < selected; ++i)
        {
            int sample = count <= maxFrames ? i : selected == 1 ? 0 : (int)Math.Floor((double)i * (count - 1) / (selected - 1));
            result[i] = Math.Min(info.FrameCount - 1, (int)Math.Floor(sample * info.Fps / fps));
        }
        return result;
    }

    private static byte[] FrameToRgba(byte[] pixels, int width, int height, int stride, PixelLayout layout)
    {
        int channels = layout is PixelLayout.Bgra or PixelLayout.Rgba ? 4 : 3;
        if (width <= 0 || height <= 0 || stride < checked(width * channels) || pixels.Length < checked(stride * height))
            throw new InvalidDataException("Video decoder returned an invalid pixel buffer.");
        var result = new byte[checked(width * height * 4)];
        bool bgr = layout is PixelLayout.Bgr or PixelLayout.Bgra;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
            {
                int source = y * stride + x * channels, target = (y * width + x) * 4;
                result[target] = pixels[source + (bgr ? 2 : 0)];
                result[target + 1] = pixels[source + 1];
                result[target + 2] = pixels[source + (bgr ? 0 : 2)];
                result[target + 3] = 255;
            }
        return result;
    }

    private int FindToken(string spelling)
    {
        int id = _tokenizer.LookupToken(spelling);
        if (id < 0) throw new InvalidDataException($"EmbeddingGemma 2 tokenizer is missing '{spelling}'.");
        return id;
    }

    internal static bool ContainsMediaDelimiter(string text)
    {
        foreach (string delimiter in new[] { "<|image>", "<image|>", "<|audio>", "<audio|>",
            "<|image|>", "<|audio|>", "<|video|>" })
            if (text.Contains(delimiter, StringComparison.Ordinal)) return true;
        return false;
    }

    private static T DecodeMedia<T>(Func<T> decode, string kind)
    {
        try { return decode(); }
        catch (Exception ex) when (ex.GetType() == typeof(Exception) ||
            ex is InvalidOperationException or FormatException or IndexOutOfRangeException or OverflowException ||
            ex.GetType().Namespace?.StartsWith("NLayer", StringComparison.Ordinal) == true ||
            ex.GetType().Namespace?.StartsWith("NVorbis", StringComparison.Ordinal) == true ||
            ex.GetType().Namespace?.StartsWith("OpenCvSharp", StringComparison.Ordinal) == true)
        {
            throw new InvalidDataException($"Cannot decode {kind} input.", ex);
        }
    }

    private static void ValidateSoftTokens(int value, string name)
    {
        if (Array.IndexOf(Gemma4ImageProcessor.SupportedSoftTokens, value) < 0)
            throw new ArgumentOutOfRangeException(name, "Vision token budget must be 70, 140, 280, 560, or 1120.");
    }

    /// <summary>This encoder's GGUF device memory (its context's pool: the vision and audio encoders' tensors; they invalidate
    /// their own weights): per owner while a model of the process is live, which keeps its own; the whole-process wipe and
    /// the shared scratch only when none is.</summary>
    private void ReleaseDeviceMemory()
    {
        if (ModelBase.LiveGgmlModels > 0)
        {
            // While no live model runs a step: the shared graph caches go too.
            ModelBase.RunWhileGgmlModelsAreIdle(null, () =>
            {
                GgmlBasicOps.ReleaseHostRanges(_context.PoolRanges(), Array.Empty<IntPtr>());
                GgmlBasicOps.ResetSharedGraphCaches();
            });
            return;
        }
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        GgmlBasicOps.ClearHostBufferCache();
    }

    public void Dispose()
    {
        _gate.Wait();
        try
        {
            if (_disposed) return;
            _disposed = true;
            _vision?.Dispose();
            _audio?.Dispose();
            TextModel.Dispose();
            if (_context != null)
            {
                ReleaseDeviceMemory();
                _context.ReleasePoolForDisposal();
            }
        }
        finally { _gate.Release(); }
    }
}
