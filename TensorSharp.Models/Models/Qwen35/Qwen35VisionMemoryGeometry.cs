// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Memory.Planning;

namespace TensorSharp.Models;

internal sealed class QwenVisionConstructionCleanupException : Exception
{
    internal Qwen35VisionEncoder UnreleasedEncoder { get; }
    internal QwenVisionConstructionCleanupException(Qwen35VisionEncoder encoder, Exception creation, Exception cleanup)
        : base("Vision construction and cleanup both failed; its model retains the encoder for cleanup retry.", new AggregateException(creation, cleanup))
        => UnreleasedEncoder = encoder;
}

/// <summary>Forecast for the existing GGML vision path, not a claim about GC,
/// codec internals, file-cache residency or vendor/driver physical ownership.</summary>
internal sealed record Qwen35VisionMemoryGeometry(int PatchSize, int MergeSize, int Hidden,
    int Intermediate, int Heads, int Blocks, int Projection, int MergerIntermediate,
    long WeightBytes, long LoadingTemporaryBytes, long PatchWeightBytes)
{
    internal long DeviceWeightBytes => checked(WeightBytes + 3 * PatchWeightBytes);
    internal long HostWeightBytes => DeviceWeightBytes;
    internal long ResidentHostBytes => HostWeightBytes;

    internal static Qwen35VisionMemoryGeometry Read(string path)
    {
        using var file = GgufFile.OpenForMemoryPlanning(path);
        return Read(file);
    }

    internal static Qwen35VisionMemoryGeometry Read(GgufFile file)
    {
        int Int(string name, int fallback)
        {
            uint value = file.GetUint32("clip.vision." + name, (uint)fallback);
            if (value == 0 || value > int.MaxValue) throw new NotSupportedException("Invalid vision geometry: " + name);
            return (int)value;
        }
        int patch = Int("patch_size", 16), merge = Int("spatial_merge_size", 2);
        int hidden = Int("embedding_length", 1152), intermediate = Int("feed_forward_length", 4304);
        int heads = Int("attention.head_count", 16), blocks = Int("block_count", 27), projection = Int("projection_dim", 4096);
        if (hidden % heads != 0 || (hidden / heads) % 2 != 0) throw new NotSupportedException("Invalid vision head width.");
        long weights = 0, temporary = 0, patchWeight = 0;
        foreach (var tensor in file.Tensors.Values)
        {
            long bytes = checked(tensor.NumElements * 4);
            weights = checked(weights + bytes + (64L << 10));
            temporary = Math.Max(temporary, checked(file.GetTensorFileRegion(tensor.Name).ByteLength + bytes));
            if (tensor.Name == "v.patch_embd.weight") patchWeight = bytes;
        }
        if (!file.Tensors.TryGetValue("mm.0.weight", out var merger) || merger.Shape.Length != 2
            || merger.Shape[1] == 0 || merger.Shape[1] > int.MaxValue)
            throw new NotSupportedException("Missing vision merger geometry.");
        return new(patch, merge, hidden, intermediate, heads, blocks, projection,
            (int)merger.Shape[1], weights, temporary, patchWeight);
    }

    internal long GeometryCacheBytes(int resizedHeight, int resizedWidth)
    {
        checked
        {
            long patches = (long)(resizedHeight / PatchSize) * (resizedWidth / PatchSize);
            return patches * (4L * Hidden + 4L * Hidden / Heads + 4L);
        }
    }

    internal InferenceMemoryBytes ExecutionPeak(int resizedHeight, int resizedWidth)
    {
        int factor = checked(PatchSize * MergeSize);
        if (resizedHeight <= 0 || resizedWidth <= 0 || resizedHeight % factor != 0 || resizedWidth % factor != 0)
            throw new ArgumentOutOfRangeException(nameof(resizedHeight));
        checked
        {
            long patches = (long)(resizedHeight / PatchSize) * (resizedWidth / PatchSize);
            long merged = patches / MergeSize / MergeSize;
            if (patches > int.MaxValue || merged * Projection > int.MaxValue)
                throw new NotSupportedException("Vision tensor exceeds the supported managed element range.");
            long geometryCache = GeometryCacheBytes(resizedHeight, resizedWidth);
            long pixelsAndIm2Col = 24L * resizedHeight * resizedWidth;
            long block = 4L * patches * (12L * Hidden + 4L * Intermediate);
            // Include the bounded score fallback, even though the native CUDA
            // path normally needs only streaming score tiles.
            long scores = 4L * Heads * patches * Math.Min(patches, Math.Max(64L, (256L << 20) / Math.Max(1, 4L * Heads * patches)));
            long projector = 4L * merged * (2L * Hidden * MergeSize * MergeSize + 2L * MergerIntermediate + Projection);
            long graph = (64L << 20) + Math.Max(block + scores, projector);
            long payload = geometryCache + pixelsAndIm2Col + graph;
            return new(payload, payload);
        }
    }
}

internal sealed record QwenVisionImagePlan(string Path, long FileBytes, long LastWriteUtcTicks,
    int OriginalHeight, int OriginalWidth, int ResizedHeight, int ResizedWidth,
    int Tokens, long EmbeddingBytes, long HostPreparationBytes, InferenceMemoryBytes ExecutionPeak);

internal sealed record QwenVisionPreparationPlan(IReadOnlyList<QwenVisionImagePlan> Images,
    int ExpandedPromptTokens, long RetainedEmbeddingBytes, long HostPeakBytes, long DevicePeakBytes)
{
    internal static QwenVisionPreparationPlan Create(IReadOnlyList<string> paths, int promptTokens,
        int placeholderCount, int maximumContext, Qwen35VisionMemoryGeometry geometry)
    {
        ArgumentNullException.ThrowIfNull(paths);
        ArgumentNullException.ThrowIfNull(geometry);
        if (promptTokens <= 0 || maximumContext <= 0 || placeholderCount != paths.Count)
            throw new ArgumentException("Every admitted image must have exactly one prompt placeholder.");
        var images = new List<QwenVisionImagePlan>(paths.Count);
        var processor = new Qwen35ImageProcessor(geometry.PatchSize, geometry.MergeSize);
        long expanded = promptTokens, retained = 0, hostPreparation = 0, hostExecution = 0, deviceExecution = 0;
        long geometryCaches = 0;
        var distinctGrids = new HashSet<(int Height, int Width)>();
        foreach (string name in paths)
        {
            string path = System.IO.Path.GetFullPath(name);
            var info = new FileInfo(path);
            long bytes = info.Length, ticks = info.LastWriteTimeUtc.Ticks;
            if (bytes <= 0 || bytes > int.MaxValue) throw new NotSupportedException("Image file exceeds the supported managed range.");
            var (width, height) = ReadBoundedDimensions(path);
            long sourcePixels = checked((long)height * width);
            if (sourcePixels > int.MaxValue / 4) throw new NotSupportedException("Decoded RGBA image exceeds the managed range.");
            var (h, w) = processor.SmartResize(height, width);
            int tokens = checked((h / geometry.PatchSize / geometry.MergeSize) * (w / geometry.PatchSize / geometry.MergeSize));
            expanded = checked(expanded + tokens - 1L);
            if (expanded > maximumContext) throw new InvalidOperationException("Images exceed the admitted context before pixel decoding.");
            long embedding = checked((long)tokens * geometry.Projection * 4);
            // Bound known encoded/RGBA/normalized representations, with codec
            // scratch allowance. GC timing and opaque codec allocations remain
            // uninstrumented; native/Tensor children are separately accounted.
            // Byte-domain bicubic retains a horizontal uint8 intermediate and
            // both axes' bounds/double/22-bit coefficient arrays. Account them
            // explicitly before decoding, rather than hiding them inside the
            // codec allowance or assuming a GC between resampling passes.
            long resizeScratch = ImageProcessorUtils.GetByteBicubicResizeScratchBytes(width, height, w, h);
            long preparation = checked(bytes + sourcePixels * 32 + 12L * h * w + resizeScratch);
            var execution = geometry.ExecutionPeak(h, w);
            images.Add(new(path, bytes, ticks, height, width, h, w, tokens, embedding, preparation, execution));
            retained = checked(retained + embedding);
            hostPreparation = checked(hostPreparation + preparation); // do not assume a GC between images
            long cache = geometry.GeometryCacheBytes(h, w);
            if (distinctGrids.Add((h, w))) geometryCaches = checked(geometryCaches + cache);
            // Position/RoPE/block-order tables survive every image. Their
            // distinct geometries accumulate even though execution is serial.
            hostExecution = Math.Max(hostExecution, execution.Host - cache);
            deviceExecution = Math.Max(deviceExecution, execution.Device - cache);
            info.Refresh();
            if (info.Length != bytes || info.LastWriteTimeUtc.Ticks != ticks) throw new IOException("Image changed during admission planning.");
        }
        // Cold vision weights may bind on the first encode. These are upper
        // bounds, so the caller may subtract only identified existing owners.
        return new(images, checked((int)expanded), retained,
            checked(hostPreparation + retained + geometryCaches + hostExecution),
            checked(geometry.DeviceWeightBytes + retained + geometryCaches + deviceExecution));
    }

    internal void ValidateFilesUnchanged()
    {
        foreach (var image in Images)
        {
            var info = new FileInfo(image.Path);
            if (info.Length != image.FileBytes || info.LastWriteTimeUtc.Ticks != image.LastWriteUtcTicks)
                throw new IOException("Admitted image changed before preparation.");
        }
    }

    /// <summary>Header-only PNG/JPEG probe. Never calls a decoder or reads the
    /// complete file. Other formats are rejected by this initial adapter.</summary>
    internal static (int Width, int Height) ReadBoundedDimensions(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096);
        Span<byte> header = stackalloc byte[24];
        stream.ReadExactly(header);
        if (header[..8].SequenceEqual(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }))
        {
            if (BinaryPrimitives.ReadUInt32BigEndian(header[8..12]) != 13
                || !header[12..16].SequenceEqual("IHDR"u8)) throw new InvalidDataException("Invalid PNG header.");
            return Dimensions(BinaryPrimitives.ReadUInt32BigEndian(header[16..20]), BinaryPrimitives.ReadUInt32BigEndian(header[20..24]));
        }
        if (header[0] != 255 || header[1] != 216) throw new NotSupportedException("Unified vision admission currently accepts PNG/JPEG still images only.");
        stream.Position = 2;
        const int maxHeader = 1 << 20;
        Span<byte> lengthBytes = stackalloc byte[2];
        Span<byte> frame = stackalloc byte[5];
        for (int segment = 0; segment < 4096 && stream.Position < Math.Min(stream.Length, maxHeader); segment++)
        {
            if (stream.ReadByte() != 255) throw new InvalidDataException("Invalid JPEG marker.");
            int marker; do { marker = stream.ReadByte(); } while (marker == 255 && stream.Position < maxHeader);
            if (marker < 0 || marker is 0 or 217 or 218) throw new InvalidDataException("JPEG has no bounded frame header.");
            if (marker is 216 or 1 || marker is >= 208 and <= 215) continue;
            stream.ReadExactly(lengthBytes);
            int length = BinaryPrimitives.ReadUInt16BigEndian(lengthBytes);
            if (length < 2 || stream.Position + length - 2 > Math.Min(stream.Length, maxHeader))
                throw new InvalidDataException("JPEG header exceeds admission scan limit.");
            if (marker is 192 or 193 or 194)
            {
                if (length < 8) throw new InvalidDataException("Invalid JPEG frame header.");
                stream.ReadExactly(frame);
                return Dimensions(BinaryPrimitives.ReadUInt16BigEndian(frame[3..5]), BinaryPrimitives.ReadUInt16BigEndian(frame[1..3]));
            }
            stream.Seek(length - 2, SeekOrigin.Current);
        }
        throw new InvalidDataException("No PNG/JPEG dimensions inside bounded admission header.");
    }

    private static (int, int) Dimensions(uint width, uint height)
    {
        if (width == 0 || height == 0 || width > int.MaxValue || height > int.MaxValue)
            throw new InvalidDataException("Invalid image dimensions.");
        return ((int)width, (int)height);
    }
}

public partial class Qwen35VisionEncoder
{
    private string _memoryGeometryPath;
    private Qwen35VisionMemoryGeometry _memoryGeometry;
    internal Qwen35VisionMemoryGeometry UnifiedMemoryGeometry => _memoryGeometry ??= Qwen35VisionMemoryGeometry.Read(_memoryGeometryPath);
    // Only changed while the owning model's GPU lock is held. The default path
    // still cooperatively yields; admitted synchronous preparation must not.
    internal bool UnifiedPreparationActive { get; set; }
    private void YieldVisionCompute()
    {
        if (!UnifiedPreparationActive) _hostModel?.YieldGpuComputeLock();
    }
}
