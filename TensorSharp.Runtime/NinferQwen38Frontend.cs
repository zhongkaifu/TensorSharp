// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime;

/// <summary>Original HF generation metadata, not sampler settings applied by this loader.</summary>
public sealed record NinferQwen38GenerationMetadata(int BosTokenId, bool DoSample,
    IReadOnlyList<int> EosTokenIds, int PadTokenId, double Temperature, int TopK, double TopP);

/// <summary>Validated processor configuration only; no image/video processing is implemented.</summary>
public sealed record NinferQwen38ProcessorMetadata(int LongestEdge, int ShortestEdge,
    int PatchSize, int TemporalPatchSize, int MergeSize, IReadOnlyList<double> Mean,
    IReadOnlyList<double> StandardDeviation, string ProcessorClass, string Implementation);

/// <summary>Preserved original UTF-8 resource. Copies never expose the stored byte array.</summary>
public sealed class NinferFrontendResource
{
    private readonly byte[] _bytes;
    public string Name { get; }
    public int ByteLength => _bytes.Length;
    public string Sha256 { get; }
    public string Utf8Text { get; }
    public byte[] CopyBytes() => (byte[])_bytes.Clone();
    internal NinferFrontendResource(string name, byte[] bytes)
    {
        Name = name; _bytes = (byte[])bytes.Clone();
        Utf8Text = new UTF8Encoding(false, true).GetString(_bytes);
        Sha256 = Convert.ToHexString(SHA256.HashData(_bytes)).ToLowerInvariant();
    }
}

/// <summary>
/// Read-only known-family HF frontend. It preserves the original chat template but does not
/// execute it, bind a factory/drafter/vision pipeline, apply sampling, or enable inference.
/// </summary>
public sealed class NinferQwen38Frontend
{
    private static readonly IReadOnlyDictionary<string, int> ResourceLimits = new ReadOnlyDictionary<string, int>(new Dictionary<string, int>(StringComparer.Ordinal)
    {
        ["frontend/tokenizer.json"] = 16 << 20,
        ["frontend/tokenizer_config.json"] = 1 << 20,
        ["frontend/chat_template.jinja"] = 64 << 10,
        ["frontend/generation_config.json"] = 64 << 10,
        ["frontend/preprocessor_config.json"] = 64 << 10,
        ["frontend/video_preprocessor_config.json"] = 64 << 10
    });
    public NinferQwen38Tokenizer Tokenizer { get; }
    public NinferQwen38GenerationMetadata Generation { get; }
    public NinferQwen38ProcessorMetadata ImageProcessor { get; }
    public NinferQwen38ProcessorMetadata VideoProcessor { get; }
    public IReadOnlyDictionary<string, NinferFrontendResource> Resources { get; }
    public string ChatTemplate => Resources["frontend/chat_template.jinja"].Utf8Text;
    public int MaximumContextTokens => 262144;
    public bool InferenceSupported => false;
    public bool ChatTemplateRenderingImplemented => false;
    public bool VisionProcessingImplemented => false;
    public bool SamplingSettingsApplied => false;
    public string? ArtifactPath { get; }
    public string? ArtifactDirectorySha256 { get; }

    public static NinferQwen38Frontend FromReader(NinferArtifactReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        reader.ReadRange(0, Span<byte>.Empty); // Disposed-reader check, no tensor content read.
        NinferFrontendJson.Require(reader.Identity.ModelId == "qwen3.8-27b" && reader.Identity.WeightsId == "nvfp4full", "Unsupported NInfer frontend family.");
        NinferFrontendJson.Require(reader.Objects.Count == 1325 && reader.Tensors.Count == 1319 && reader.Resources.Count == 6, "Known artifact inventory count differs.");
        NinferFrontendJson.Require(reader.Resources.Select(x => x.Name).ToHashSet(StringComparer.Ordinal).SetEquals(ResourceLimits.Keys), "Known frontend resource names differ.");
        ulong total = 0;
        foreach (var resource in reader.Resources)
        {
            NinferFrontendJson.Require(ReferenceEquals(reader.GetObject(resource.Name), resource) && resource.Encoding == "raw-bytes-v1" && resource.ByteLength <= (ulong)ResourceLimits[resource.Name], "Frontend resource ownership/encoding/bounds differ.");
            total = checked(total + resource.ByteLength);
        }
        NinferFrontendJson.Require(total <= 16UL << 20, "Frontend resources exceed total read budget.");
        var bytes = reader.Resources.ToDictionary(x => x.Name, x => reader.ReadResource(x, ResourceLimits[x.Name]), StringComparer.Ordinal);
        return ParseResources(bytes, NinferFrontendProfile.Actual, reader.FullPath, reader.DirectorySha256);
    }

    // Tiny generated CPU fixtures only. The public reader path cannot select reduced counts.
    internal static NinferQwen38Frontend ParseTinyForTests(IReadOnlyDictionary<string, byte[]> bytes) =>
        ParseResources(bytes, NinferFrontendProfile.TinyTests, null, null);

    private NinferQwen38Frontend(NinferQwen38Tokenizer tokenizer, NinferQwen38GenerationMetadata generation,
        NinferQwen38ProcessorMetadata image, NinferQwen38ProcessorMetadata video,
        IReadOnlyDictionary<string, NinferFrontendResource> resources, string? path, string? directorySha)
    {
        Tokenizer = tokenizer; Generation = generation; ImageProcessor = image; VideoProcessor = video;
        Resources = resources; ArtifactPath = path; ArtifactDirectorySha256 = directorySha;
    }
    private static NinferQwen38Frontend ParseResources(IReadOnlyDictionary<string, byte[]> bytes,
        NinferFrontendProfile profile, string? path, string? directorySha)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        NinferFrontendJson.Require(bytes.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(ResourceLimits.Keys), "Exactly six known resources are required.");
        long total = 0;
        var resources = new Dictionary<string, NinferFrontendResource>(StringComparer.Ordinal);
        foreach (var limit in ResourceLimits)
        {
            byte[] value = bytes[limit.Key];
            NinferFrontendJson.Require(value != null && value.Length > 0 && value.Length <= limit.Value, "Frontend resource empty or exceeds bound.");
            total = checked(total + value!.Length); resources.Add(limit.Key, new NinferFrontendResource(limit.Key, value));
        }
        NinferFrontendJson.Require(total <= 16L << 20, "Frontend resources exceed total read budget.");
        using var tokenizer = NinferFrontendJson.Parse(resources["frontend/tokenizer.json"].Utf8Text);
        using var config = NinferFrontendJson.Parse(resources["frontend/tokenizer_config.json"].Utf8Text);
        using var generation = NinferFrontendJson.Parse(resources["frontend/generation_config.json"].Utf8Text);
        using var image = NinferFrontendJson.Parse(resources["frontend/preprocessor_config.json"].Utf8Text);
        using var video = NinferFrontendJson.Parse(resources["frontend/video_preprocessor_config.json"].Utf8Text);
        var g = ParseGeneration(generation.RootElement);
        var t = NinferQwen38Tokenizer.Parse(tokenizer.RootElement, config.RootElement, g, profile);
        NinferFrontendJson.Require(NinferFrontendJson.Text(config.RootElement.GetProperty("chat_template")) == resources["frontend/chat_template.jinja"].Utf8Text, "Chat template resource and tokenizer config differ.");
        return new(t, g, ParseProcessor(image.RootElement, false), ParseProcessor(video.RootElement, true),
            new ReadOnlyDictionary<string, NinferFrontendResource>(resources), path, directorySha);
    }
    private static NinferQwen38GenerationMetadata ParseGeneration(JsonElement root)
    {
        NinferFrontendJson.Exact(root, "bos_token_id", "do_sample", "eos_token_id", "pad_token_id", "temperature", "top_k", "top_p");
        var eos = root.GetProperty("eos_token_id"); NinferFrontendJson.Require(eos.ValueKind == JsonValueKind.Array && eos.GetArrayLength() == 2, "Explicit ordered two EOS IDs required.");
        int[] ids = eos.EnumerateArray().Select(NinferFrontendJson.Integer).ToArray();
        double temp = NinferFrontendJson.Number(root.GetProperty("temperature")), topP = NinferFrontendJson.Number(root.GetProperty("top_p")); int topK = NinferFrontendJson.Integer(root.GetProperty("top_k"));
        NinferFrontendJson.Require(temp >= 0 && topP > 0 && topP <= 1 && topK >= 0, "Generation sampling metadata outside supported domain.");
        return new(NinferFrontendJson.Integer(root.GetProperty("bos_token_id")), NinferFrontendJson.Bool(root.GetProperty("do_sample")), Array.AsReadOnly(ids), NinferFrontendJson.Integer(root.GetProperty("pad_token_id")), temp, topK, topP);
    }
    private static NinferQwen38ProcessorMetadata ParseProcessor(JsonElement root, bool video)
    {
        string impl = video ? "video_processor_type" : "image_processor_type";
        NinferFrontendJson.Exact(root, "size", "patch_size", "temporal_patch_size", "merge_size", "image_mean", "image_std", "processor_class", impl);
        var size = root.GetProperty("size"); NinferFrontendJson.Exact(size, "longest_edge", "shortest_edge");
        int longest = NinferFrontendJson.Integer(size.GetProperty("longest_edge")), shortest = NinferFrontendJson.Integer(size.GetProperty("shortest_edge"));
        int patch = NinferFrontendJson.Integer(root.GetProperty("patch_size")), temporal = NinferFrontendJson.Integer(root.GetProperty("temporal_patch_size")), merge = NinferFrontendJson.Integer(root.GetProperty("merge_size"));
        NinferFrontendJson.Require(longest == (video ? 25165824 : 16777216) && shortest == (video ? 4096 : 65536) && patch == 16 && temporal == 2 && merge == 2, "Unknown image/video geometry.");
        static IReadOnlyList<double> Triple(JsonElement e)
        {
            NinferFrontendJson.Require(e.ValueKind == JsonValueKind.Array && e.GetArrayLength() == 3, "Processor channel statistics must have three entries.");
            double[] a = e.EnumerateArray().Select(NinferFrontendJson.Number).ToArray();
            NinferFrontendJson.Require(a.All(x => x == 0.5), "Unknown processor channel statistics."); return Array.AsReadOnly(a);
        }
        string cls = NinferFrontendJson.Text(root.GetProperty("processor_class")), implementation = NinferFrontendJson.Text(root.GetProperty(impl));
        NinferFrontendJson.Require(cls == "Qwen3VLProcessor" && implementation == (video ? "Qwen3VLVideoProcessor" : "Qwen2VLImageProcessorFast"), "Unknown processor implementation.");
        return new(longest, shortest, patch, temporal, merge, Triple(root.GetProperty("image_mean")), Triple(root.GetProperty("image_std")), cls, implementation);
    }
}

internal static class NinferFrontendJson
{
    internal static JsonDocument Parse(string text)
    {
        var doc = JsonDocument.Parse(text, new JsonDocumentOptions { MaxDepth = 64 });
        try { CheckDuplicates(doc.RootElement); return doc; } catch { doc.Dispose(); throw; }
    }
    private static void CheckDuplicates(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            var keys = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in value.EnumerateObject()) { Require(keys.Add(p.Name), "Duplicate JSON property: " + p.Name); CheckDuplicates(p.Value); }
        }
        else if (value.ValueKind == JsonValueKind.Array) foreach (var x in value.EnumerateArray()) CheckDuplicates(x);
    }
    internal static void Exact(JsonElement value, params string[] keys) => Require(value.ValueKind == JsonValueKind.Object && value.EnumerateObject().Select(x => x.Name).ToHashSet(StringComparer.Ordinal).SetEquals(keys), "Unknown/missing JSON behavior fields.");
    internal static string Text(JsonElement value) { Require(value.ValueKind == JsonValueKind.String, "Expected JSON string."); return value.GetString()!; }
    internal static bool Bool(JsonElement value) { Require(value.ValueKind is JsonValueKind.True or JsonValueKind.False, "Expected JSON boolean."); return value.GetBoolean(); }
    internal static int Integer(JsonElement value) { Require(value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out _), "Expected exact Int32 JSON integer."); return value.GetInt32(); }
    internal static double Number(JsonElement value) { Require(value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out double d) && double.IsFinite(d), "Expected finite JSON number."); return value.GetDouble(); }
    internal static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
}
