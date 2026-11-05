// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.Runtime;

public enum NinferTensorFormat
{
    BFloat16, Float32, Int32, Q4G64Float16Scale, Q5G64Float16Scale,
    Q6G64Float16Scale, W8G32Float16Scale, NvFp4, Fp8E4M3FnRowBFloat16Scale
}

public enum NinferTensorLayout
{
    ContiguousLittleEndianV1, RowSplitK128V1, BlockScaleK16M128x4V1, RowScaleV1
}

/// <summary>Activation representations a use position permits (NInfer v3 <c>activation_policy</c>).</summary>
public enum NinferActivationPolicy { A16Only, AllowA8, AllowA4 }

/// <summary>Model name (metadata.name) and conversion recipe (provenance.recipe).</summary>
public sealed record NinferArtifactIdentity(string ModelId, string WeightsId);

/// <summary>Offsets and lengths describe original encoded bytes, never GGML types or converted weights.</summary>
public abstract class NinferObjectDescriptor
{
    internal object Owner { get; }
    /// <summary>The directory object id, e.g. <c>weight/000123</c>.</summary>
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
    /// <summary>NVFP4 matrix divisor stored after the scale plane (FP32), null for other formats.</summary>
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
    public string Encoding => "raw_bytes_v1";
    internal NinferResourceDescriptor(object owner, string name, ulong offset, long absolute, ulong bytes)
        : base(owner, name, offset, absolute, bytes) { }
}

/// <summary>A logical parameter: a whole object, or one contiguous element range of it (whole rows of a matrix).</summary>
public sealed class NinferBinding
{
    public string Name { get; }
    public NinferTensorDescriptor Object { get; }
    public ulong ElementBegin { get; }
    public ulong ElementEnd { get; }
    public bool IsWholeObject { get; }
    internal NinferBinding(string name, NinferTensorDescriptor tensor, ulong begin, ulong end, bool whole)
        => (Name, Object, ElementBegin, ElementEnd, IsWholeObject) = (name, tensor, begin, end, whole);

    /// <summary>First row and row count of a range over a rank-2 [rows, columns] object.</summary>
    public (int FirstRow, int Rows) RowRange
    {
        get
        {
            if (Object.Shape.Count != 2) throw new InvalidOperationException("Row ranges need a rank-2 object: " + Name);
            ulong columns = Object.Shape[1];
            if (ElementBegin % columns != 0 || ElementEnd % columns != 0)
                throw new InvalidDataException("Binding range does not cover whole rows: " + Name);
            return (checked((int)(ElementBegin / columns)), checked((int)((ElementEnd - ElementBegin) / columns)));
        }
    }
}

/// <summary>One use position of a parameter: its input, permitted activation representation and calibration.</summary>
public sealed class NinferUse
{
    public string Parameter { get; }
    public string Input { get; }
    public NinferActivationPolicy Policy { get; }
    /// <summary>FP32 scalar activation divisor of an A4 use, null otherwise.</summary>
    public NinferTensorDescriptor? ActivationInputDivisor { get; }
    internal NinferUse(string parameter, string input, NinferActivationPolicy policy, NinferTensorDescriptor? divisor)
        => (Parameter, Input, Policy, ActivationInputDivisor) = (parameter, input, policy, divisor);
}

/// <summary>
/// Read-only NInfer v3 container reader (single-file artifacts): the 32-byte entry header
/// (magic, JSON length, artifact id), the JSON directory (components, objects, bindings,
/// uses, metadata, provenance, files) and a 4096-aligned payload. Geometry follows
/// Neroued/ninfer 1882c6ad src/artifact/{framing.h,formats.cpp,layouts.cpp}; this is an
/// independent TensorSharp implementation. Initialization reads the header, the bounded
/// JSON and each NVFP4 divisor (4 bytes); code/scale/resource contents are not validated.
/// </summary>
public sealed class NinferArtifactReader : IDisposable
{
    public const int HeaderBytes = 32;
    public const ulong PayloadAlignment = 4096;
    public const string FormatSpecificationRevision = "1882c6ad";

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

    private readonly SafeFileHandle _handle;
    private readonly object _owner = new();
    private readonly Dictionary<string, NinferObjectDescriptor> _byId = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<NinferUse>> _usesByParameter = new(StringComparer.Ordinal);
    private long _bytesRead, _readOperations;
    public string FullPath { get; }
    public long FileBytes { get; }
    /// <summary>Original open-handle timestamp for cache identity; not a payload checksum.</summary>
    public long FileLastWriteUtcTicks { get; }
    public int MetadataBytes { get; }
    public long PayloadStart { get; }
    public string ArtifactId { get; }
    public NinferArtifactIdentity Identity { get; }
    public string PrefixSha256 { get; }
    public string DirectorySha256 { get; }
    public IReadOnlyList<NinferObjectDescriptor> Objects { get; }
    public IReadOnlyList<NinferTensorDescriptor> Tensors { get; }
    public IReadOnlyList<NinferResourceDescriptor> Resources { get; }
    public IReadOnlyDictionary<string, NinferBinding> Bindings { get; }
    public IReadOnlyList<NinferUse> Uses { get; }
    /// <summary>Component name to its resource map (file name to descriptor).</summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, NinferResourceDescriptor>> ComponentResources { get; }
    /// <summary>Raw component configuration JSON (component name to <c>config</c> object text).</summary>
    public IReadOnlyDictionary<string, string> ComponentConfigs { get; }
    public long InitializationBytesRead { get; }
    public long InitializationReadOperations { get; }
    public long TotalBytesRead => Interlocked.Read(ref _bytesRead);
    public long ReadOperations => Interlocked.Read(ref _readOperations);

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
            Span<byte> prefix = stackalloc byte[HeaderBytes];
            ReadRange(0, prefix[..8]);
            if (prefix[..8].SequenceEqual("NINFER\0\x02"u8))
                Unsupported("UNSUPPORTED_SCHEMA: NInfer v2 artifacts are no longer supported; use the NInfer v3 .ninfer release.");
            if (prefix[..8].SequenceEqual("NINFER\0\x01"u8)) Unsupported("UNSUPPORTED_SCHEMA: NInfer v1");
            if (prefix[..8].SequenceEqual("NINPRT\0\x03"u8)) Unsupported("UNSUPPORTED_SCHEMA: open the NInfer v3 entry file, not a continuation part.");
            Require(prefix[..8].SequenceEqual("NINFER\0\x03"u8), "Unknown NInfer magic.");
            Require(FileBytes >= HeaderBytes, "Artifact shorter than the v3 header.");
            ReadRange(8, prefix[8..]);
            ulong length = BinaryPrimitives.ReadUInt64LittleEndian(prefix[8..16]);
            Require(length > 0, "Empty JSON directory.");
            ulong metadataEnd = Add(HeaderBytes, length, "metadata end");
            ulong payloadStart = Align(metadataEnd, PayloadAlignment, "payload start");
            Require(metadataEnd <= (ulong)FileBytes && payloadStart <= (ulong)FileBytes, "Directory/payload start past EOF.");
            Require(length <= (ulong)maxMetadataBytes, "JSON directory exceeds configured inspection limit.");
            MetadataBytes = checked((int)length);
            PayloadStart = checked((long)payloadStart);
            ArtifactId = Hex(prefix[16..32].ToArray());
            PrefixSha256 = Hex(SHA256.HashData(prefix));
            byte[] metadata = new byte[MetadataBytes];
            ReadRange(HeaderBytes, metadata);
            DirectorySha256 = Hex(SHA256.HashData(metadata));
            using var doc = JsonDocument.Parse(metadata, new JsonDocumentOptions { MaxDepth = 64 });
            var directory = Members(doc.RootElement, "directory", "components", "objects", "bindings", "uses", "metadata", "provenance", "files");

            var files = directory["files"];
            Require(files.ValueKind == JsonValueKind.Array && files.GetArrayLength() == 1, "Multi-file NInfer v3 artifacts are not supported.");
            var file = Members(files[0], "file", "path", "payload_bytes");
            Require(file["path"].ValueKind == JsonValueKind.Null, "The single v3 file entry must be the entry file.");
            ulong payloadBytes = UInt(file["payload_bytes"], "payload_bytes");
            Require(payloadBytes <= (ulong)FileBytes - payloadStart, "Declared payload exceeds the file.");

            string model = Text(Members(directory["metadata"], "metadata", "name")["name"], "metadata.name");
            var provenance = directory["provenance"];
            Require(provenance.ValueKind == JsonValueKind.Object, "provenance must be an object.");
            Require(provenance.TryGetProperty("recipe", out var recipe), "provenance.recipe missing.");
            Identity = new(model, Text(recipe, "provenance.recipe"));

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
                    fields = Members(row, "tensor", "id", "kind", "shape", "format", "layout", "offset", "bytes");
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
                    fields = Members(row, "resource", "id", "kind", "encoding", "offset", "bytes");
                    Require(Text(fields["encoding"], "encoding") == "raw_bytes_v1", "Unknown resource encoding.");
                }
                string id = Text(fields["id"], "id");
                Require(!_byId.ContainsKey(id), "Duplicate object id: " + id);
                ulong offset = UInt(fields["offset"], "offset"), bytes = UInt(fields["bytes"], "bytes", true);
                Require(offset >= cursor, "Objects overlap or are out of order: " + id);
                Require(offset % (geometry?.Alignment ?? 1) == 0, "Object alignment: " + id);
                ulong end = Add(offset, bytes, "object end");
                Require(end <= payloadBytes, "Object past the declared payload: " + id);
                Require(geometry == null || bytes == geometry.EncodedBytes, "Encoded tensor bytes mismatch: " + id);
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
                        Require(float.IsFinite(value) && value > 0, "Non-positive/nonfinite NVFP4 divisor: " + id);
                        divisor = value; divisorBits = encoded;
                    }
                    var tensor = new NinferTensorDescriptor(_owner, id, offset, absolute, bytes, format, layout, shape, geometry, divisor, divisorBits);
                    descriptor = tensor; tensors.Add(tensor);
                }
                else
                {
                    var resource = new NinferResourceDescriptor(_owner, id, offset, absolute, bytes);
                    descriptor = resource; resources.Add(resource);
                }
                _byId.Add(id, descriptor); objects.Add(descriptor); cursor = end;
            }
            Objects = objects.AsReadOnly(); Tensors = tensors.AsReadOnly(); Resources = resources.AsReadOnly();

            var bindings = new Dictionary<string, NinferBinding>(StringComparer.Ordinal);
            foreach (var property in Object(directory["bindings"], "bindings").EnumerateObject())
            {
                Require(!bindings.ContainsKey(property.Name), "Duplicate binding: " + property.Name);
                bindings.Add(property.Name, ParseBinding(property.Name, property.Value));
            }
            Bindings = new ReadOnlyDictionary<string, NinferBinding>(bindings);

            var uses = new List<NinferUse>();
            Require(directory["uses"].ValueKind == JsonValueKind.Array, "Uses must be an array.");
            foreach (var row in directory["uses"].EnumerateArray())
            {
                Require(row.ValueKind == JsonValueKind.Object, "Use must be an object.");
                bool hasAuxiliaries = row.TryGetProperty("auxiliaries", out var auxiliaries);
                var fields = hasAuxiliaries ? Members(row, "use", "parameter", "input", "activation_policy", "auxiliaries")
                    : Members(row, "use", "parameter", "input", "activation_policy");
                string parameter = Text(fields["parameter"], "use parameter");
                Require(bindings.ContainsKey(parameter), "Use names an unbound parameter: " + parameter);
                var policy = Text(fields["activation_policy"], "activation_policy") switch
                {
                    "A16Only" => NinferActivationPolicy.A16Only,
                    "AllowA8" => NinferActivationPolicy.AllowA8,
                    "AllowA4" => NinferActivationPolicy.AllowA4,
                    string other => throw new InvalidDataException("Unknown activation policy: " + other)
                };
                NinferTensorDescriptor? divisor = null;
                if (hasAuxiliaries)
                {
                    foreach (var auxiliary in Object(auxiliaries, "auxiliaries").EnumerateObject())
                    {
                        Require(auxiliary.Name == "activation_input_divisor", "Unknown use auxiliary: " + auxiliary.Name);
                        string id = Text(Members(auxiliary.Value, "auxiliary", "object")["object"], "auxiliary object");
                        divisor = Tensor(id);
                        Require(divisor.Format == NinferTensorFormat.Float32 && divisor.Shape.Count == 0, "Activation divisor must be an FP32 scalar: " + id);
                    }
                }
                var use = new NinferUse(parameter, Text(fields["input"], "use input"), policy, divisor);
                uses.Add(use);
                if (!_usesByParameter.TryGetValue(parameter, out var list)) _usesByParameter[parameter] = list = new();
                list.Add(use);
            }
            Uses = uses.AsReadOnly();

            var componentResources = new Dictionary<string, IReadOnlyDictionary<string, NinferResourceDescriptor>>(StringComparer.Ordinal);
            var componentConfigs = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var component in Object(directory["components"], "components").EnumerateObject())
            {
                var map = new Dictionary<string, NinferResourceDescriptor>(StringComparer.Ordinal);
                if (component.Value.TryGetProperty("resources", out var resourceMap))
                    foreach (var entry in Object(resourceMap, "resources").EnumerateObject())
                    {
                        string id = Text(entry.Value, "resource id");
                        Require(_byId.TryGetValue(id, out var resource) && resource is NinferResourceDescriptor, "Component resource is not a resource object: " + id);
                        map.Add(entry.Name, (NinferResourceDescriptor)resource!);
                    }
                componentResources.Add(component.Name, new ReadOnlyDictionary<string, NinferResourceDescriptor>(map));
                if (component.Value.TryGetProperty("config", out var config)) componentConfigs.Add(component.Name, config.GetRawText());
            }
            ComponentResources = new ReadOnlyDictionary<string, IReadOnlyDictionary<string, NinferResourceDescriptor>>(componentResources);
            ComponentConfigs = new ReadOnlyDictionary<string, string>(componentConfigs);
            Require(RandomAccess.GetLength(_handle) == FileBytes, "Artifact length changed during metadata inspection.");
            InitializationBytesRead = TotalBytesRead;
            InitializationReadOperations = ReadOperations;
        }
        catch (JsonException error) { _handle.Dispose(); throw new InvalidDataException("Invalid JSON directory.", error); }
        catch { _handle.Dispose(); throw; }
    }

    private NinferBinding ParseBinding(string name, JsonElement value)
    {
        Require(value.ValueKind == JsonValueKind.Object, "Binding must be an object: " + name);
        if (value.TryGetProperty("object", out _))
        {
            var tensor = Tensor(Text(Members(value, "binding", "object")["object"], "binding object"));
            ulong count = tensor.Shape.Aggregate(1UL, (a, b) => checked(a * b));
            return new(name, tensor, 0, count, true);
        }
        var parts = Members(value, "binding", "parts")["parts"];
        Require(parts.ValueKind == JsonValueKind.Array && parts.GetArrayLength() == 1, "Only single-part bindings are supported: " + name);
        var part = Members(parts[0], "binding part", "object", "range");
        var t = Tensor(Text(part["object"], "part object"));
        var range = part["range"];
        Require(range.ValueKind == JsonValueKind.Array && range.GetArrayLength() == 2, "Part range must be [begin, end): " + name);
        ulong begin = UInt(range[0], "range begin"), end = UInt(range[1], "range end");
        ulong total = t.Shape.Aggregate(1UL, (a, b) => checked(a * b));
        Require(begin < end && end <= total, "Part range outside its object: " + name);
        return new(name, t, begin, end, begin == 0 && end == total);
    }

    public NinferTensorDescriptor Tensor(string id)
    {
        Require(_byId.TryGetValue(id, out var o) && o is NinferTensorDescriptor, "Unknown tensor object: " + id);
        return (NinferTensorDescriptor)o!;
    }

    public NinferBinding GetBinding(string name)
    {
        if (!Bindings.TryGetValue(name, out var binding)) throw new InvalidDataException("Missing NInfer binding: " + name);
        return binding;
    }

    public IReadOnlyList<NinferUse> UsesOf(string parameter)
        => _usesByParameter.TryGetValue(parameter, out var list) ? list.AsReadOnly() : Array.Empty<NinferUse>();

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

    /// <summary>Explicit bounded raw bytes only: tokenizer/config/template interpretation belongs to the frontend.</summary>
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

    /// <summary>Reads one FP32 scalar tensor (for example an activation divisor).</summary>
    public float ReadScalar(NinferTensorDescriptor descriptor)
    {
        Require(descriptor.Format == NinferTensorFormat.Float32 && descriptor.Shape.Count == 0 && descriptor.ByteLength == 4, "Expected an FP32 scalar: " + descriptor.Name);
        Span<byte> raw = stackalloc byte[4];
        ReadObjectRange(descriptor, 0, raw);
        return BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(raw));
    }

    public NinferObjectDescriptor GetObject(string id) => _byId[id];
    public void Dispose() => _handle.Dispose();

    private static string Hex(byte[] bytes) => Convert.ToHexString(bytes).ToLowerInvariant();
    private static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    private static void Unsupported(string message) => throw new NotSupportedException(message);
    private static string Text(JsonElement value, string label)
    {
        Require(value.ValueKind == JsonValueKind.String && value.GetString() is { Length: > 0 }, label + " must be a nonempty string.");
        return value.GetString()!;
    }
    private static JsonElement Object(JsonElement value, string label)
    {
        Require(value.ValueKind == JsonValueKind.Object, label + " must be an object.");
        return value;
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
        "bf16" => NinferTensorFormat.BFloat16, "fp32" => NinferTensorFormat.Float32, "int32" => NinferTensorFormat.Int32,
        "q4_g64_fp16" => NinferTensorFormat.Q4G64Float16Scale, "q5_g64_fp16" => NinferTensorFormat.Q5G64Float16Scale,
        "q6_g64_fp16" => NinferTensorFormat.Q6G64Float16Scale, "q8_g32_fp16" => NinferTensorFormat.W8G32Float16Scale,
        "nvfp4" => NinferTensorFormat.NvFp4, "fp8_e4m3fn_row_bf16" => NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale,
        _ => throw new NotSupportedException("UNSUPPORTED_TENSOR_FORMAT: " + format)
    };
    private static NinferTensorLayout ParseLayout(string layout) => layout switch
    {
        "contiguous_le_v1" => NinferTensorLayout.ContiguousLittleEndianV1,
        "row_split_k128_v1" => NinferTensorLayout.RowSplitK128V1,
        "block_scale_k16_m128x4_v1" => NinferTensorLayout.BlockScaleK16M128x4V1,
        "row_scale_v1" => NinferTensorLayout.RowScaleV1,
        _ => throw new NotSupportedException("UNSUPPORTED_TENSOR_LAYOUT: " + layout)
    };
    private static NinferTensorGeometry Geometry(NinferTensorFormat format, NinferTensorLayout layout, ulong[] shape)
    {
        if (layout == NinferTensorLayout.ContiguousLittleEndianV1)
        {
            ulong width = format switch { NinferTensorFormat.BFloat16 => 2, NinferTensorFormat.Float32 or NinferTensorFormat.Int32 => 4, _ => 0UL };
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
