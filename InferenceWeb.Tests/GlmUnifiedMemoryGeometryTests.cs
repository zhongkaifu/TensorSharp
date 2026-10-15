// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using TensorSharp;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

/// <summary>Metadata/policy admission checks only; no GPU qualification.</summary>
public sealed class GlmUnifiedMemoryGeometryTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-glm-budget-" + Guid.NewGuid());
    public GlmUnifiedMemoryGeometryTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Theory]
    [InlineData("glm5next")]
    [InlineData("glm5-next")]
    public void BothArchitectureSpellingsForecastIndependentKdaAndF16MlaSlots(string architecture)
    {
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(Path.Combine(_directory, "tiny.gguf"),
            4, true, numLayers: 4, mixedAttention: true, routedExperts: true,
            quantizeExperts: true, architecture: architecture);
        using var file = GgufFile.OpenForMemoryPlanning(path);
        var geometry = GlmUnifiedMemoryGeometry.Read(file);
        Assert.Equal(architecture, geometry.Architecture);
        Assert.Equal(4, geometry.Layers);
        Assert.Equal(2, geometry.RecurrentLayers);
        Assert.Equal(2, geometry.AttentionLayers);
        // Native MLA and pooled indexer rows use F16; KDA convolution and
        // recurrent states use F32, independently for every sequence slot.
        Assert.Equal(2L * 2 * (32 + 2 * 64), geometry.KvBytesPerToken);
        Assert.Equal(4L * 2 * (3L * 4 * 16 * 3 + 4L * 16 * 16), geometry.FixedStateBytes);
        Assert.Equal(geometry.KvBytesPerToken * 128, geometry.SlotBytes(256) - geometry.SlotBytes(128));
        Assert.Equal(file.Tensors.Values.Sum(t => file.GetTensorFileRegion(t.Name).ByteLength), geometry.SourceBytes);
        Assert.All(geometry.ExpertLayerBytes, bytes => Assert.True(bytes > 0));
        Assert.True(geometry.ExpertSourceBytes > 0);
        Assert.Equal(4, geometry.ExpertCount);
        Assert.All(geometry.ExpertLayerLayouts, layout =>
        {
            Assert.NotNull(layout);
            Assert.Equal((GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, GgmlTensorType.Q8_0), layout.Types);
            Assert.Equal((64L * 64 * 3 / 32) * 34, layout.BytesPerExpert);
            Assert.True(layout.SupportsSelectedTier);
            Assert.Equal(3L * 448 / 32 * 34, layout.TensorTailBytes);
        });
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.SlotBytes(257));
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.Workspace(256, 1025));
        Assert.True(geometry.Workspace(256, 3).Device > geometry.Workspace(256, 1).Device);
    }

    [Fact]
    public void HeaderOnlyMixedExpertTriplesPreserveQuantizedPayloadAndRejectIncompleteTriples()
    {
        const string a = "glm5-next";
        string path = Path.Combine(_directory, "mixed-experts.gguf");
        Write(path, [new Str { Key = "general.architecture", V = a },
            new U32 { Key = a + ".block_count", V = 2 }, new U32 { Key = a + ".context_length", V = 32 },
            new U32 { Key = a + ".embedding_length", V = 256 }, new U32 { Key = a + ".vocab_size", V = 256 },
            new U32 { Key = a + ".attention.head_count", V = 2 }, new U32Arr { Key = a + ".attention.head_count_kv", V = [0, 1] },
            new U32 { Key = a + ".attention.kv_lora_rank", V = 512 }, new U32 { Key = a + ".kda.head_dim", V = 128 },
            new U32 { Key = a + ".ssm.conv_kernel", V = 4 }, new U32 { Key = a + ".attention.indexer.key_length", V = 128 },
            new U32 { Key = a + ".hyper_connection.count", V = 4 }, new U32 { Key = a + ".expert_feed_forward_length", V = 256 },
            new U32 { Key = a + ".expert_count", V = 4 }, new U32 { Key = a + ".expert_used_count", V = 2 },
            new U32 { Key = a + ".attention.indexer.kpool", V = 4 }],
            [Gen("token_embd.weight", .1f, 256, 256),
                Blocks("blk.0.ffn_gate_exps.weight", SyntheticGguf.GgmlType.IQ2_XS, .1f, 256, 256, 4),
                Blocks("blk.0.ffn_up_exps.weight", SyntheticGguf.GgmlType.IQ2_XS, .1f, 256, 256, 4),
                Blocks("blk.0.ffn_down_exps.weight", SyntheticGguf.GgmlType.IQ3_XXS, .1f, 256, 256, 4),
                Blocks("blk.1.ffn_gate_exps.weight", SyntheticGguf.GgmlType.IQ3_XXS, .1f, 256, 256, 4),
                Blocks("blk.1.ffn_up_exps.weight", SyntheticGguf.GgmlType.IQ3_XXS, .1f, 256, 256, 4),
                Blocks("blk.1.ffn_down_exps.weight", SyntheticGguf.GgmlType.IQ4_XS, .1f, 256, 256, 4)]);
        using var file = GgufFile.OpenForMemoryPlanning(path);
        var geometry = GlmUnifiedMemoryGeometry.Read(file);
        Assert.Equal((GgmlTensorType.IQ2_XS, GgmlTensorType.IQ2_XS, GgmlTensorType.IQ3_XXS), geometry.ExpertLayerLayouts[0]!.Types);
        Assert.Equal((GgmlTensorType.IQ3_XXS, GgmlTensorType.IQ3_XXS, GgmlTensorType.IQ4_XS), geometry.ExpertLayerLayouts[1]!.Types);
        Assert.Equal(256L * (74 + 74 + 98), geometry.ExpertLayerLayouts[0]!.BytesPerExpert);
        Assert.Equal(256L * (98 + 98 + 136), geometry.ExpertLayerLayouts[1]!.BytesPerExpert);
        Assert.Equal(98L + 98 + 136, geometry.ExpertLayerLayouts[1]!.TensorTailBytes);
        Assert.All(geometry.ExpertLayerLayouts, l => Assert.True(l!.SupportsSelectedTier));
        file.Tensors.Remove("blk.1.ffn_up_exps.weight");
        Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryGeometry.Read(file));
    }

    [Fact]
    public void PlacementCannotSpendSlotWorkspaceOrVisionCapacityOnExperts()
    {
        var geometry = new GlmUnifiedMemoryGeometry
        { Layers = 4, DenseDeviceBytes = 100, ExpertLayerBytes = [0, 50, 70, 80] };
        Assert.Equal(0, geometry.SelectCpuExpertLayers(500, 100, 100));
        Assert.Equal(2, geometry.SelectCpuExpertLayers(450, 100, 100));
        Assert.Equal(3, geometry.SelectCpuExpertLayers(380, 100, 100));
        Assert.Equal(4, geometry.SelectCpuExpertLayers(300, 100, 100));
        Assert.Throws<MemoryPressureException>(() => geometry.SelectCpuExpertLayers(299, 100, 100));
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.SelectCpuExpertLayers(300, -1, 100));
    }

    [Fact]
    public void GlobalGraphPlacementRetainsOnlyOneCacheAndReplacementAcrossRequests()
    {
        var geometry = new GlmUnifiedMemoryGeometry
        {
            Layers = 45, DeclaredContextTokens = 32768, RecurrentLayers = 34,
            Hidden = 4096, Heads = 64, Vocab = 154880, KvRow = 512, IndexerDim = 128,
            KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048, UsedExperts = 8
        };
        var solo = geometry.GraphPeak(32768, 1, false);
        var prefill = geometry.GraphPeak(32768, 128, false);
        var joint = geometry.GraphPeak(32768, 3, true);
        long expected = geometry.SlotBytes(32768) * 3 + Math.Max(solo.Device, Math.Max(prefill.Device, joint.Device));
        Assert.Equal(expected, geometry.ExecutionPlacementReserve(32768, 128, 3));
        long duplicated = (geometry.SlotBytes(32768) + solo.Device) * 3 + prefill.Device + joint.Device;
        Assert.True(expected < duplicated);
        Assert.Equal(2L * 2 * geometry.FixedStateBytes,
            joint.Device - 2L * geometry.Workspace(32768, 3).Device);
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.ExecutionPlacementReserve(32768, 128, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.GraphPeak(32768, 3, true, 0));
        Assert.True(geometry.ExecutionPlacementReserve(32768, 128, 1) < expected);
    }

    [Fact]
    public void OrdinaryModelPolicyIsOptionalAndUnifiedRejectsPartialAdapters()
    {
        GlmDsaModel.ValidateUnifiedMemoryPolicy(null, BackendType.Cpu, 4, 2, true);
        var policy = new ModelMemoryPolicy(1024, 128) { OmitEmbeddedDraftWeights = true, NativeMoeCpuLayers = 3 };
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy, BackendType.Cpu, 1, 1, false));
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 2, 1, false));
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 1, 2, false));
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 1, 1, true));
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy with { OmitEmbeddedDraftWeights = false }, BackendType.GgmlCuda, 1, 1, false));
        Assert.Throws<NotSupportedException>(() => GlmDsaModel.ValidateUnifiedMemoryPolicy(policy with { NativeMoeCpuLayers = null }, BackendType.GgmlCuda, 1, 1, false));
    }

    [Theory]
    [InlineData("glm5next")]
    [InlineData("glm5-next")]
    public void FactoryAllowsPublishedGlmOnlyWithCompleteSingleCudaPolicy(string architecture)
    {
        // Exercise the production factory guard, in addition to the model's
        // own guard. Metadata tests must not construct a CUDA model on CPU CI.
        ModelBase.ValidateExplicitModelMemoryPolicy("glm-dsa", architecture, BackendType.GgmlCuda, 1, false, 1);
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateExplicitModelMemoryPolicy(
            "glm-dsa", architecture, BackendType.Cpu, 1, false, 1));
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateExplicitModelMemoryPolicy(
            "glm-dsa", architecture, BackendType.GgmlCuda, 2, false, 1));
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateExplicitModelMemoryPolicy(
            "glm-dsa", architecture, BackendType.GgmlCuda, 1, true, 1));
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateExplicitModelMemoryPolicy(
            "glm-dsa", architecture, BackendType.GgmlCuda, 1, false, 2));
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateExplicitModelMemoryPolicy(
            "glm-dsa", "glm-dsa", BackendType.GgmlCuda, 1, false, 1));
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateExplicitModelMemoryPolicy(
            "unknown", architecture, BackendType.GgmlCuda, 1, false, 1));
    }

    [Fact]
    public void PublishedGlm5VProjectorUsesOneFusedEncoderDeviceRepresentation()
    {
        string path = Path.Combine(_directory, "vision.gguf");
        Write(path, [new Str { Key = "clip.projector_type", V = "glm5v" },
            new U32 { Key = "clip.vision.embedding_length", V = 64 },
            new U32 { Key = "clip.vision.attention.head_count", V = 4 }],
            [Gen("v.blk.0.attn_qkv.weight", .1f, 64, 192), Gen("mm.patch_merger.weight", .1f, 4, 64, 32)]);
        var geometry = GlmUnifiedVisionMemoryGeometry.Read(path);
        long bytes = (64 * 192 + 4 * 64 * 32) * 4L, allowance = 2 * (64L << 10), copies = 3L * 4 * 64 * 32 * 4;
        Assert.Equal(bytes + allowance + copies, geometry.DeviceWeightBytes);
        Assert.Equal(2 * (bytes + allowance) + copies, geometry.HostWeightBytes);
        Assert.True(geometry.ExecutionPeak(56, 56).Device > geometry.ExecutionPeak(28, 28).Device);
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.ExecutionPeak(29, 28));
    }

    [Fact]
    public void ExperimentalExpertCachePreservesSlotsGraphVisionAndOperatorCeilings()
    {
        const long gib = 1L << 30;
        Assert.Equal(3 * gib, GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(14 * gib, 6 * gib, 3 * gib, 2 * gib, long.MaxValue));
        Assert.Equal(gib, GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(12 * gib, 6 * gib, 3 * gib, 2 * gib, long.MaxValue));
        Assert.Equal(gib / 2, GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(14 * gib, 6 * gib, 3 * gib, 2 * gib, gib / 2));
        Assert.Equal(0, GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(10 * gib, 6 * gib, 3 * gib, 2 * gib, long.MaxValue));
        Assert.Equal(0, GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(long.MaxValue, long.MaxValue, long.MaxValue, 0, long.MaxValue));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(14 * gib, 6 * gib, 3 * gib, 0, -1));
    }

    [Fact]
    public void ImageGeometryAndFileIdentityAreCheckedBeforePixelsAreDecoded()
    {
        var geometry = new GlmUnifiedVisionMemoryGeometry(14, 2, 64, 128, 4, 32, 1024, 512, 128);
        byte[] header = new byte[24];
        new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }.CopyTo(header, 0);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(8), 13);
        "IHDR"u8.CopyTo(header.AsSpan(12));
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(16), 56);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(20), 56);
        string path = Path.Combine(_directory, "header-only.png"); File.WriteAllBytes(path, header);
        var plan = GlmVisionPreparationPlan.Create([path], 10, 1, 1024, geometry);
        Assert.Equal(25, plan.ExpandedPromptTokens); // GLM's minimum is 16 merged tokens.
        Assert.True(plan.HostPeakBytes > geometry.HostWeightBytes);
        Assert.True(plan.DevicePeakBytes > geometry.DeviceWeightBytes);
        var coldTransposes = GlmVisionPreparationPlan.Create([path], 10, 1, 1024, geometry, 64);
        Assert.Equal(plan.HostPeakBytes - geometry.HostWeightBytes + 64, coldTransposes.HostPeakBytes);
        Assert.Equal(plan.DevicePeakBytes, coldTransposes.DevicePeakBytes); // cold device bindings are still forecast
        var warmTransposes = GlmVisionPreparationPlan.Create([path], 10, 1, 1024, geometry, 0);
        Assert.Equal(coldTransposes.HostPeakBytes - 64, warmTransposes.HostPeakBytes);
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmVisionPreparationPlan.Create([path], 10, 1, 1024, geometry, -1));
        plan.ValidateFilesUnchanged();
        Assert.Throws<ArgumentException>(() => GlmVisionPreparationPlan.Create([path], 10, 2, 1024, geometry));
        Assert.Throws<InvalidOperationException>(() => GlmVisionPreparationPlan.Create([path], 10, 1, 24, geometry));
        File.WriteAllBytes(path, [1, 2]);
        Assert.Throws<IOException>(() => plan.ValidateFilesUnchanged());
    }
}
