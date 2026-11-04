// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.Runtime;

public enum NinferTensorFormat
{
    BFloat16, Float32, Int32, Int64, Q4G64Float16Scale, Q5G64Float16Scale,
    Q6G64Float16Scale, W8G32Float16Scale, NvFp4, Fp8E4M3FnRowBFloat16Scale
}

public enum NinferTensorLayout
{
    ContiguousLittleEndianV1, RowSplitK128V1, BlockScaleK16M128x4V1, RowScaleV1
}

public sealed record NinferArtifactIdentity(string ModelId, string WeightsId);

/// <summary>Offsets and lengths describe original encoded bytes, never GGML types or converted weights.</summary>
public abstract class NinferObjectDescriptor
{
    internal object Owner { get; }
    public string Name { get; }
    public ulong PayloadOffset { get; }
    public long AbsoluteOffset { get; }
    public ulong ByteLength { get; }
    internal NinferObjectDescriptor(object owner, string name, ulong offset, long absolute, ulong bytes)
        => (Owner, Name, PayloadOffset, AbsoluteOffset, ByteLength) = (owner, name, offset, absolute, bytes);
}

/// <summary>Exact encoded-plane geometry. Null offsets mean that plane does not exist.</summary>
public sealed record NinferTensorGeometry(
    ulong EncodedBytes, ulong Alignment, ulong? PaddedK = null, int? GroupSize = null,
    ulong? LowPlaneBytes = null, ulong? HighPlaneOffset = null, ulong? HighPlaneBytes = null,
    ulong? ScalePlaneOffset = null, ulong? ScalePlaneBytes = null, ulong? DivisorOffset = null);

public sealed class NinferTensorDescriptor : NinferObjectDescriptor
{
    public NinferTensorFormat Format { get; }
    public NinferTensorLayout Layout { get; }
    public IReadOnlyList<ulong> Shape { get; }
    public NinferTensorGeometry Geometry { get; }
    public float? WeightDivisor { get; }
    public uint? WeightDivisorBits { get; }
    internal NinferTensorDescriptor(object owner, string name, ulong offset, long absolute, ulong bytes,
        NinferTensorFormat format, NinferTensorLayout layout, ulong[] shape,
        NinferTensorGeometry geometry, float? divisor, uint? divisorBits)
        : base(owner, name, offset, absolute, bytes)
    {
        Format = format; Layout = layout; Shape = Array.AsReadOnly(shape);
        Geometry = geometry; WeightDivisor = divisor; WeightDivisorBits = divisorBits;
    }
}

public sealed class NinferResourceDescriptor : NinferObjectDescriptor
{
    public string Encoding => "raw-bytes-v1";
    internal NinferResourceDescriptor(object owner, string name, ulong offset, long absolute, ulong bytes)
        : base(owner, name, offset, absolute, bytes) { }
}

/// <summary>
/// Read-only, positional NInfer v2 metadata reader. No factory/native model integration,
/// quantization, repacking, tokenizer binding, CUDA calls or full-artifact checksum.
/// Geometry follows Ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4,
/// crates/artifact/src/lib.rs; this is an independent TensorSharp implementation.
/// Initialization reads only the 16-byte prefix, bounded JSON, and each NVFP4 divisor (4 bytes).
/// Code/scale/resource contents are not validated by a successful constructor.
/// </summary>
public sealed class NinferArtifactReader : IDisposable
{
    /// <summary>Recognizes the NInfer family before selecting a format parser; the reader still validates its version and directory.</summary>
    public static bool IsNinferArtifact(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        bool hint = string.Equals(Path.GetExtension(path), ".ninfer", StringComparison.OrdinalIgnoreCase);
        try
        {
            using var stream = File.OpenRead(path);
            Span<byte> magic = stackalloc byte[8];
            int count = stream.ReadAtLeast(magic, magic.Length, throwOnEndOfStream: false);
            if (count >= 4 && magic[..4].SequenceEqual("GGUF"u8)) return false;
            if (count >= 7 && magic[..7].SequenceEqual("NINFER\0"u8)) return true;
            return hint; // A damaged explicitly named artifact gets this parser's refusal.
        }
        catch (FileNotFoundException) when (hint) { return true; }
        catch (DirectoryNotFoundException) when (hint) { return true; }
    }

    public const string GeometrySpecificationRevision = "e600a7d9e0d5921d767a6cb464373583b3cdcaf4";
    public const string GeometrySpecificationSourceSha256 = "e264fe3522e4fa07b21f66bf31e9877987396b59c55ae71e7a639b79ad434a73";
    private readonly SafeFileHandle _handle;
    private readonly object _owner = new();
    private readonly Dictionary<string, NinferObjectDescriptor> _byName = new(StringComparer.Ordinal);
    private long _bytesRead, _readOperations;
    public string FullPath { get; }
    public long FileBytes { get; }
    /// <summary>Original open-handle timestamp for cache identity; not a payload checksum.</summary>
    public long FileLastWriteUtcTicks { get; }
    public int MetadataBytes { get; }
    public long PayloadStart { get; }
    public NinferArtifactIdentity Identity { get; }
    public string PrefixSha256 { get; }
    public string DirectorySha256 { get; }
    public IReadOnlyList<NinferObjectDescriptor> Objects { get; }
    public IReadOnlyList<NinferTensorDescriptor> Tensors { get; }
    public IReadOnlyList<NinferResourceDescriptor> Resources { get; }
    public ulong TrailingUnclaimedPayloadBytes { get; }
    public long InitializationBytesRead { get; }
    public long InitializationReadOperations { get; }
    public long TotalBytesRead => Interlocked.Read(ref _bytesRead);
    public long ReadOperations => Interlocked.Read(ref _readOperations);
    public bool FullArtifactSha256Recomputed => false;

    public NinferArtifactReader(string path, int maxMetadataBytes = 16 << 20, int maxObjects = 100_000)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxMetadataBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxObjects);
        FullPath = Path.GetFullPath(path);
        _handle = File.OpenHandle(FullPath, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
        try
        {
            FileBytes = RandomAccess.GetLength(_handle);
            FileLastWriteUtcTicks = File.GetLastWriteTimeUtc(_handle).Ticks;
            Require(FileBytes >= 8, "Artifact shorter than magic.");
            Span<byte> prefix = stackalloc byte[16];
            ReadRange(0, prefix[..8]);
            if (prefix[..8].SequenceEqual("NINFER\0\x03"u8)) Unsupported("UNSUPPORTED_SCHEMA: NInfer v3");
            if (prefix[..8].SequenceEqual("VELOXVI\x01"u8)) Unsupported("UNSUPPORTED_SCHEMA: Velox VI v1");
            if (prefix[..8].SequenceEqual("NINFER\0\x01"u8)) Unsupported("UNSUPPORTED_SCHEMA: NInfer v1");
            Require(prefix[..8].SequenceEqual("NINFER\0\x02"u8), "Unknown NInfer magic.");
            Require(FileBytes >= 16, "Artifact shorter than v2 prefix.");
            ReadRange(8, prefix[8..]);
            ulong length = BinaryPrimitives.ReadUInt64LittleEndian(prefix[8..]);
            Require(length > 0, "Empty JSON directory.");
            ulong metadataEnd = Add(16, length, "metadata end");
            ulong payloadStart = Align(metadataEnd, 4096, "payload start");
            Require(metadataEnd <= (ulong)FileBytes && payloadStart <= (ulong)FileBytes, "Directory/payload start past EOF.");
            Require(length <= (ulong)maxMetadataBytes, "JSON directory exceeds configured inspection limit.");
            MetadataBytes = checked((int)length);
            PayloadStart = checked((long)payloadStart);
            PrefixSha256 = Hex(SHA256.HashData(prefix));
            byte[] metadata = new byte[MetadataBytes];
            ReadRange(16, metadata);
            DirectorySha256 = Hex(SHA256.HashData(metadata));
            using var doc = JsonDocument.Parse(metadata, new JsonDocumentOptions { MaxDepth = 64 });
            var directory = Members(doc.RootElement, "directory", "identity", "objects");
            var identity = Members(directory["identity"], "identity", "model_id", "weights_id");
            Identity = new(Text(identity["model_id"], "model_id"), Text(identity["weights_id"], "weights_id"));
            JsonElement rows = directory["objects"];
            Require(rows.ValueKind == JsonValueKind.Array, "Objects must be an array.");
            Require(rows.GetArrayLength() > 0 && rows.GetArrayLength() <= maxObjects, "Invalid or excessive object count.");
            var objects = new List<NinferObjectDescriptor>();
            var tensors = new List<NinferTensorDescriptor>();
            var resources = new List<NinferResourceDescriptor>();
            ulong cursor = 0;
            Span<byte> divisorBytes = stackalloc byte[4];
            foreach (var row in rows.EnumerateArray())
            {
                Require(row.ValueKind == JsonValueKind.Object && row.TryGetProperty("kind", out _), "Object kind missing.");
                string kind = Text(row.GetProperty("kind"), "kind");
                Dictionary<string, JsonElement> fields;
                NinferTensorGeometry? geometry = null;
                NinferTensorFormat format = default; NinferTensorLayout layout = default;
                ulong[] shape = Array.Empty<ulong>();
                if (kind == "tensor")
                {
                    fields = Members(row, "tensor", "name", "kind", "shape", "format", "layout", "offset", "bytes");
                    format = ParseFormat(Text(fields["format"], "format"));
                    layout = ParseLayout(Text(fields["layout"], "layout"));
                    var rawShape = fields["shape"];
                    Require(rawShape.ValueKind == JsonValueKind.Array && rawShape.GetArrayLength() <= 16, "Shape must have rank 0..16.");
                    shape = new ulong[rawShape.GetArrayLength()];
                    for (int i = 0; i < shape.Length; i++) shape[i] = UInt(rawShape[i], "shape dimension", true);
                    geometry = Geometry(format, layout, shape);
                }
                else
                {
                    Require(kind == "resource", "Unknown object kind.");
                    fields = Members(row, "resource", "name", "kind", "encoding", "offset", "bytes");
                    Require(Text(fields["encoding"], "encoding") == "raw-bytes-v1", "Unknown resource encoding.");
                }
                string name = Text(fields["name"], "name");
                Require(!_byName.ContainsKey(name), "Duplicate object name: " + name);
                ulong offset = UInt(fields["offset"], "offset"), bytes = UInt(fields["bytes"], "bytes", true);
                Require(offset >= cursor, "Objects overlap or are out of order: " + name);
                Require(offset % (geometry?.Alignment ?? 1) == 0, "Object alignment: " + name);
                ulong end = Add(offset, bytes, "object end");
                Require(end <= (ulong)FileBytes - payloadStart, "Object past EOF: " + name);
                Require(geometry == null || bytes == geometry.EncodedBytes, "Encoded tensor bytes mismatch: " + name);
                long absolute = checked((long)Add(payloadStart, offset, "absolute offset"));
                NinferObjectDescriptor descriptor;
                if (geometry != null)
                {
                    float? divisor = null; uint? divisorBits = null;
                    if (geometry.DivisorOffset is ulong divisorOffset)
                    {
                        ReadRange(checked((long)Add((ulong)absolute, divisorOffset, "absolute divisor offset")), divisorBytes);
                        uint encoded = BinaryPrimitives.ReadUInt32LittleEndian(divisorBytes);
                        float value = BitConverter.Int32BitsToSingle(unchecked((int)encoded));
                        Require(float.IsFinite(value) && value > 0, "Non-positive/nonfinite NVFP4 divisor: " + name);
                        divisor = value; divisorBits = encoded;
                    }
                    var tensor = new NinferTensorDescriptor(_owner, name, offset, absolute, bytes, format, layout, shape, geometry, divisor, divisorBits);
                    descriptor = tensor; tensors.Add(tensor);
                }
                else
                {
                    var resource = new NinferResourceDescriptor(_owner, name, offset, absolute, bytes);
                    descriptor = resource; resources.Add(resource);
                }
                _byName.Add(name, descriptor); objects.Add(descriptor); cursor = end;
            }
            Objects = objects.AsReadOnly(); Tensors = tensors.AsReadOnly(); Resources = resources.AsReadOnly();
            TrailingUnclaimedPayloadBytes = (ulong)FileBytes - payloadStart - cursor;
            Require(RandomAccess.GetLength(_handle) == FileBytes, "Artifact length changed during metadata inspection.");
            InitializationBytesRead = TotalBytesRead;
            InitializationReadOperations = ReadOperations;
        }
        catch (JsonException error) { _handle.Dispose(); throw new InvalidDataException("Invalid JSON directory.", error); }
        catch { _handle.Dispose(); throw; }
    }

    /// <summary>Exact absolute file-range read; concurrent calls never share a seek position.</summary>
    public void ReadRange(long absoluteOffset, Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_handle.IsClosed, this);
        if (absoluteOffset < 0 || absoluteOffset > FileBytes || destination.Length > FileBytes - absoluteOffset)
            throw new ArgumentOutOfRangeException(nameof(absoluteOffset), "Read range past file bounds.");
        int read = 0;
        while (read < destination.Length)
        {
            int count = RandomAccess.Read(_handle, destination[read..], checked(absoluteOffset + read));
            Interlocked.Increment(ref _readOperations);
            Interlocked.Add(ref _bytesRead, count);
            if (count == 0) throw new EndOfStreamException("Short NInfer range read.");
            read += count;
        }
    }

    public void ReadObjectRange(NinferObjectDescriptor descriptor, ulong relativeOffset, Span<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        if (!ReferenceEquals(descriptor.Owner, _owner)) throw new ArgumentException("Descriptor belongs to another reader.", nameof(descriptor));
        if (relativeOffset > descriptor.ByteLength || (ulong)destination.Length > descriptor.ByteLength - relativeOffset)
            throw new ArgumentOutOfRangeException(nameof(relativeOffset), "Read range past object bounds.");
        ReadRange(checked(descriptor.AbsoluteOffset + (long)relativeOffset), destination);
    }

    /// <summary>Explicit bounded raw bytes only: tokenizer/config/template interpretation belongs to a later integration.</summary>
    public byte[] ReadResource(NinferResourceDescriptor descriptor, int maxBytes = 16 << 20)
    {
        ObjectDisposedException.ThrowIf(_handle.IsClosed, this);
        ArgumentNullException.ThrowIfNull(descriptor);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBytes);
        if (!ReferenceEquals(descriptor.Owner, _owner)) throw new ArgumentException("Descriptor belongs to another reader.", nameof(descriptor));
        if (descriptor.ByteLength > (ulong)maxBytes) throw new InvalidOperationException("Resource exceeds configured read limit.");
        byte[] result = new byte[checked((int)descriptor.ByteLength)];
        ReadObjectRange(descriptor, 0, result);
        return result;
    }

    public NinferObjectDescriptor GetObject(string name) => _byName[name];
    public void Dispose() => _handle.Dispose();

    private static string Hex(byte[] bytes) => Convert.ToHexString(bytes).ToLowerInvariant();
    private static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    private static void Unsupported(string message) => throw new NotSupportedException(message);
    private static string Text(JsonElement value, string label)
    {
        Require(value.ValueKind == JsonValueKind.String && value.GetString() is { Length: > 0 }, label + " must be a nonempty string.");
        return value.GetString()!;
    }
    private static ulong UInt(JsonElement value, string label, bool positive = false)
    {
        Require(value.ValueKind == JsonValueKind.Number, label + " must be an unsigned integer.");
        foreach (char c in value.GetRawText()) Require(c >= '0' && c <= '9', label + " must be an unsigned integer token.");
        Require(value.TryGetUInt64(out ulong result) && (!positive || result != 0), label + " invalid u64/positive integer.");
        return result;
    }
    private static Dictionary<string, JsonElement> Members(JsonElement element, string label, params string[] keys)
    {
        Require(element.ValueKind == JsonValueKind.Object, label + " must be an object.");
        var result = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject()) Require(result.TryAdd(property.Name, property.Value), "Duplicate JSON member: " + property.Name);
        Require(result.Count == keys.Length && keys.All(result.ContainsKey), label + " missing/extra members.");
        return result;
    }
    private static ulong Add(ulong a, ulong b, string label)
    {
        try { return checked(a + b); }
        catch (OverflowException error) { throw new InvalidDataException(label + " overflows u64.", error); }
    }
    private static ulong Mul(ulong a, ulong b, string label)
    {
        try { return checked(a * b); }
        catch (OverflowException error) { throw new InvalidDataException(label + " overflows u64.", error); }
    }
    private static ulong Align(ulong value, ulong alignment, string label) => Add(value, alignment - 1, label) / alignment * alignment;

    private static NinferTensorFormat ParseFormat(string format) => format switch
    {
        "BF16" => NinferTensorFormat.BFloat16, "FP32" => NinferTensorFormat.Float32,
        "I32" => NinferTensorFormat.Int32, "I64" => NinferTensorFormat.Int64,
        "Q4G64_F16S" => NinferTensorFormat.Q4G64Float16Scale, "Q5G64_F16S" => NinferTensorFormat.Q5G64Float16Scale,
        "Q6G64_F16S" => NinferTensorFormat.Q6G64Float16Scale, "W8G32_F16S" => NinferTensorFormat.W8G32Float16Scale,
        "NVFP4" => NinferTensorFormat.NvFp4, "FP8_E4M3FN_ROW_BF16S" => NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale,
        "TRELLIS_MUL1_K2" or "TRELLIS_MUL1_K2P5" or "TRELLIS_MUL1_K3" or "TRELLIS_MUL1_K4" or "TRELLIS_MUL1_K5" or "TRELLIS_MUL1_K6" or "Q4G32_F16S"
            => throw new NotSupportedException("UNSUPPORTED_TENSOR_GEOMETRY: " + format),
        _ => throw new InvalidDataException("Unknown v2 numeric format: " + format)
    };
    private static NinferTensorLayout ParseLayout(string layout) => layout switch
    {
        "contiguous-le-v1" => NinferTensorLayout.ContiguousLittleEndianV1,
        "row-split-k128-v1" => NinferTensorLayout.RowSplitK128V1,
        "blockscale-k16-m128x4-v1" => NinferTensorLayout.BlockScaleK16M128x4V1,
        "row-scale-v1" => NinferTensorLayout.RowScaleV1,
        "trellis-tile16-v1" or "row-interleaved-v1" => throw new NotSupportedException("UNSUPPORTED_TENSOR_GEOMETRY: " + layout),
        _ => throw new InvalidDataException("Unknown v2 storage layout: " + layout)
    };
    private static NinferTensorGeometry Geometry(NinferTensorFormat format, NinferTensorLayout layout, ulong[] shape)
    {
        if (layout == NinferTensorLayout.ContiguousLittleEndianV1)
        {
            ulong width = format switch { NinferTensorFormat.BFloat16 => 2, NinferTensorFormat.Float32 or NinferTensorFormat.Int32 => 4, NinferTensorFormat.Int64 => 8, _ => 0UL };
            Require(width != 0, "Contiguous format/layout mismatch.");
            ulong count = 1;
            foreach (ulong dim in shape) count = Mul(count, dim, "element count");
            return new(Mul(count, width, "direct bytes"), 256);
        }
        Require(shape.Length == 2, "Quantized layout requires rank two [N,K].");
        ulong n = shape[0], k = shape[1];
        if (layout == NinferTensorLayout.RowSplitK128V1)
        {
            var (group, low, high) = format switch
            {
                NinferTensorFormat.Q4G64Float16Scale => (64, 32, 0), NinferTensorFormat.Q5G64Float16Scale => (64, 32, 8),
                NinferTensorFormat.Q6G64Float16Scale => (64, 32, 16), NinferTensorFormat.W8G32Float16Scale => (32, 32, 0),
                _ => throw new InvalidDataException("Row-split format/layout mismatch.")
            };
            ulong kp = Align(k, 128, "padded K"), groups = Mul(n, kp / (ulong)group, "physical groups");
            ulong lowBytes = Mul(groups, (ulong)low, "low bytes"), highBytes = Mul(groups, (ulong)high, "high bytes");
            ulong highOffset = Align(lowBytes, 256, "high offset"), scaleOffset = Add(highOffset, Align(highBytes, 256, "high aligned"), "scale offset");
            ulong scales = Mul(groups, 2, "scale bytes");
            return new(Add(scaleOffset, scales, "encoded bytes"), 256, kp, group, lowBytes, highOffset, highBytes, scaleOffset, scales);
        }
        if (layout == NinferTensorLayout.BlockScaleK16M128x4V1)
        {
            Require(format == NinferTensorFormat.NvFp4 && n % 128 == 0 && k % 64 == 0, "NVFP4 requires N%128=0, K%64=0 and matching layout.");
            ulong elements = Mul(n, k, "NVFP4 elements"), codes = elements / 2, scales = elements / 16;
            ulong scaleOffset = Align(codes, 256, "scale offset"), divisor = Add(scaleOffset, scales, "divisor offset");
            return new(Add(divisor, 4, "NVFP4 bytes"), 256, LowPlaneBytes: codes, ScalePlaneOffset: scaleOffset, ScalePlaneBytes: scales, DivisorOffset: divisor);
        }
        Require(layout == NinferTensorLayout.RowScaleV1 && format == NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale, "FP8 format/layout mismatch.");
        ulong codeBytes = Mul(n, k, "FP8 elements"), scale = Align(codeBytes, 256, "FP8 scale offset"), scaleBytes = Mul(n, 2, "FP8 scales");
        return new(Add(scale, scaleBytes, "FP8 bytes"), 256, LowPlaneBytes: codeBytes, ScalePlaneOffset: scale, ScalePlaneBytes: scaleBytes);
    }
}
