// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class QwenImageArchitectureTests
{
    [Theory]
    [InlineData("", true)]
    [InlineData("", false)]
    [InlineData("model.diffusion_model.", true)]
    [InlineData("model.diffusion_model.", false)]
    public void MetadataFreeVersion21ResolvesWithBothSupportedNamespacesAndMlpLayouts(string prefix, bool fused)
    {
        // The downloaded Unsloth Q8_0 file has zero metadata entries, a mixed
        // BF16/F32/Q8_0 tensor table and a fused gate/up projection in each block.
        // An opaque filename and no tensor payload ensure recognition uses only
        // the header, independent of filename, native libraries or weight reads.
        using var fixture = new TensorHeader(Version21Tensors(prefix, fused));
        Assert.Empty(fixture.File.Metadata);
        Assert.Equal(fused ? 265 : 297, fixture.File.Tensors.Count);
        Assert.Equal("qwen_image", Resolve(fixture.File).Id);
    }

    [Theory]
    [InlineData("img_in.weight")]
    [InlineData("txt_in.text_norm.weight")]
    [InlineData("modulation.1.weight")]
    [InlineData("transformer_blocks.0.img_mlp.gate_up.weight")]
    public void IncompleteSignatureIsNotClaimed(string missing)
    {
        var tensors = Version21Tensors("", fused: true);
        tensors.Remove(missing);
        using var fixture = new TensorHeader(tensors);
        Assert.Throws<NotSupportedException>(() => Resolve(fixture.File));
    }

    [Theory]
    [InlineData("img_in.weight", 64, 3072)] // Original double-stream Qwen-Image.
    [InlineData("txt_in.in_layer.weight", 3584, 4096)] // Earlier text conditioner.
    [InlineData("modulation.1.weight", 4096, 24576)]
    [InlineData("transformer_blocks.0.img_mlp.gate_up.weight", 4096, 12288)]
    [InlineData("proj_out.weight", 4096, 128)]
    public void IncompatibleDimensionsAreNotClaimed(string name, ulong input, ulong output)
    {
        var tensors = Version21Tensors("", fused: true);
        tensors[name] = (new[] { input, output }, GgmlTensorType.Q8_0);
        using var fixture = new TensorHeader(tensors);
        Assert.Throws<NotSupportedException>(() => Resolve(fixture.File));
    }

    [Fact]
    public void UnsupportedRankIsNotClaimed()
    {
        var tensors = Version21Tensors("", fused: true);
        tensors["txt_in.text_norm.weight"] = (new ulong[] { 4096, 1 }, GgmlTensorType.BF16);
        using var fixture = new TensorHeader(tensors);
        Assert.Throws<NotSupportedException>(() => Resolve(fixture.File));
    }

    [Fact]
    public void TensorsFromDifferentNamespacesAreNotCombined()
    {
        var tensors = Version21Tensors("model.diffusion_model.", fused: true);
        tensors["txt_in.text_norm.weight"] = tensors["model.diffusion_model.txt_in.text_norm.weight"];
        tensors.Remove("model.diffusion_model.txt_in.text_norm.weight");
        using var fixture = new TensorHeader(tensors);
        Assert.Throws<NotSupportedException>(() => Resolve(fixture.File));
    }

    [Fact]
    public void AModelFilenameAloneCannotSelectAnArchitecture()
    {
        using var fixture = new TensorHeader(new(), "qwen-image-2.1-Q8_0.gguf");
        Assert.Throws<NotSupportedException>(() => Resolve(fixture.File));
    }

    [Fact]
    public void ExplicitArchitectureStillTakesPrecedence()
    {
        using var fixture = new TensorHeader(Version21Tensors("", fused: true));
        Assert.Equal("qwen35", ModelArchitectureRegistry.Resolve("qwen35", fixture.File).Id);
        Assert.Throws<NotSupportedException>(() => ModelArchitectureRegistry.Resolve("unknown-architecture", fixture.File));
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(null)] // the GGML backend this test process pins
    public void EarlierQwenImageTransformersAreRefusedWithAMigrationNote(BackendType? requested)
    {
        // A Qwen-Image-Edit-2511-layout DiT (60 double-stream blocks, Qwen2.5-VL 3584-wide
        // text input, no txt_in.text_norm) tagged general.architecture=qwen_image. The tag
        // routes it to QwenImageModel, which must refuse it by name before anything else:
        // not with "requires a GGML backend" and not with a missing-companion error.
        BackendType backend = requested ?? TestGates.PinnedGgmlBackend;
        using var fixture = new TensorHeader(LegacyEditTensors(), "qwen-image-edit-2511-Q4_K_M.gguf",
            new() { ["general.architecture"] = "qwen_image" });
        Assert.Equal("qwen_image", Resolve(fixture.File).Id);

        bool preferManaged = NativeDequant.PreferManaged;
        try
        {
            var error = Assert.Throws<ModelLoadRefusedException>(() => new QwenImageModel(fixture.Path, backend));
            Assert.Contains("'qwen-image-edit-2511-Q4_K_M.gguf' is not a Qwen-Image-2.1 diffusion transformer", error.Message);
            Assert.Contains("docs/models/qwenimage21.md", error.Message);
            Assert.True(ModelLoadRefusal.TryDescribe(error, out string reason));
            Assert.DoesNotContain("\n", reason);
        }
        finally
        {
            // Constructing a model pins the process-global dequant route; restore it.
            NativeDequant.PreferManaged = preferManaged;
        }
    }

    [ModelFact("TENSORSHARP_QWEN21_DIT")]
    public void RealVersion21CheckpointResolvesFromItsTensorTable()
    {
        using var file = new GgufFile(Environment.GetEnvironmentVariable("TENSORSHARP_QWEN21_DIT")!);
        Assert.Equal("qwen_image", Resolve(file).Id);
    }

    private static ModelArchitectureDescriptor Resolve(GgufFile file) =>
        ModelArchitectureRegistry.Resolve(file.GetString("general.architecture"), file);

    private static Dictionary<string, (ulong[] Shape, GgmlTensorType Type)> LegacyEditTensors()
    {
        var tensors = new Dictionary<string, (ulong[] Shape, GgmlTensorType Type)>
        {
            ["img_in.weight"] = (new ulong[] { 64, 3072 }, GgmlTensorType.BF16),
            ["txt_norm.weight"] = (new ulong[] { 3584 }, GgmlTensorType.F32),
            ["txt_in.weight"] = (new ulong[] { 3584, 3072 }, GgmlTensorType.BF16),
            ["time_text_embed.timestep_embedder.linear_1.weight"] = (new ulong[] { 256, 3072 }, GgmlTensorType.BF16),
            ["norm_out.linear.weight"] = (new ulong[] { 3072, 6144 }, GgmlTensorType.BF16),
            ["proj_out.weight"] = (new ulong[] { 3072, 64 }, GgmlTensorType.BF16),
        };
        for (int layer = 0; layer < 60; layer++)
        {
            string block = $"transformer_blocks.{layer}.";
            tensors[block + "img_mod.1.weight"] = (new ulong[] { 3072, 18432 }, GgmlTensorType.Q4_K);
            tensors[block + "txt_mod.1.weight"] = (new ulong[] { 3072, 18432 }, GgmlTensorType.Q4_K);
            tensors[block + "attn.to_q.weight"] = (new ulong[] { 3072, 3072 }, GgmlTensorType.Q4_K);
            tensors[block + "attn.add_q_proj.weight"] = (new ulong[] { 3072, 3072 }, GgmlTensorType.Q4_K);
        }
        return tensors;
    }

    internal static Dictionary<string, (ulong[] Shape, GgmlTensorType Type)> Version21Tensors(string prefix, bool fused)
    {
        var tensors = new Dictionary<string, (ulong[] Shape, GgmlTensorType Type)>();
        void Add(string name, GgmlTensorType type, params ulong[] shape) => tensors.Add(prefix + name, (shape, type));
        Add("img_in.weight", GgmlTensorType.BF16, 64, 4096);
        Add("txt_in.text_norm.weight", GgmlTensorType.BF16, 4096);
        Add("txt_in.in_layer.weight", GgmlTensorType.BF16, 4096, 4096);
        Add("txt_in.out_layer.weight", GgmlTensorType.BF16, 4096, 4096);
        Add("time_text_embed.timestep_embedder.linear_1.weight", GgmlTensorType.Q8_0, 256, 4096);
        Add("time_text_embed.timestep_embedder.linear_2.weight", GgmlTensorType.Q8_0, 4096, 4096);
        Add("modulation.1.weight", GgmlTensorType.Q8_0, 4096, 16384);
        Add("norm_out.linear.weight", GgmlTensorType.F32, 4096, 4096);
        Add("proj_out.weight", GgmlTensorType.Q8_0, 4096, 64);
        for (int layer = 0; layer < 32; layer++)
        {
            string block = $"transformer_blocks.{layer}.";
            foreach (string projection in new[] { "to_q", "to_k", "to_v", "to_out.0" })
                Add(block + "attn." + projection + ".weight", GgmlTensorType.Q8_0, 4096, 4096);
            Add(block + "attn.norm_q.weight", GgmlTensorType.F32, 128);
            Add(block + "attn.norm_k.weight", GgmlTensorType.F32, 128);
            if (fused)
                Add(block + "img_mlp.gate_up.weight", GgmlTensorType.Q8_0, 4096, 24576);
            else
            {
                Add(block + "img_mlp.gate_layer.weight", GgmlTensorType.Q8_0, 4096, 12288);
                Add(block + "img_mlp.proj.weight", GgmlTensorType.Q8_0, 4096, 12288);
            }
            Add(block + "img_mlp.out.weight", GgmlTensorType.Q8_0, 12288, 4096);
        }
        return tensors;
    }

    // Header-only GGUF exercises the real reader and registry, without needing
    // gigabytes of payload or allocating a native inference backend.
    internal sealed class TensorHeader : IDisposable
    {
        private readonly string _directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        internal GgufFile File { get; }
        internal string Path { get; }

        internal TensorHeader(Dictionary<string, (ulong[] Shape, GgmlTensorType Type)> tensors, string name = "weights.gguf",
            Dictionary<string, string>? metadata = null)
        {
            Directory.CreateDirectory(_directory);
            string path = System.IO.Path.Combine(_directory, name);
            Path = path;
            metadata ??= new();
            using (var writer = new BinaryWriter(System.IO.File.Create(path)))
            {
                void WriteString(string value)
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(value);
                    writer.Write((ulong)bytes.Length);
                    writer.Write(bytes);
                }
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write((ulong)tensors.Count);
                writer.Write((ulong)metadata.Count);
                foreach (var (key, value) in metadata)
                {
                    WriteString(key);
                    writer.Write(8u); // GGUF string value
                    WriteString(value);
                }
                foreach (var (key, tensor) in tensors)
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(key);
                    writer.Write((ulong)bytes.Length);
                    writer.Write(bytes);
                    writer.Write((uint)tensor.Shape.Length);
                    foreach (ulong dimension in tensor.Shape) writer.Write(dimension);
                    writer.Write((uint)tensor.Type);
                    writer.Write(0UL);
                }
            }
            File = new GgufFile(path);
        }

        public void Dispose()
        {
            File.Dispose();
            Directory.Delete(_directory, recursive: true);
        }
    }
}
