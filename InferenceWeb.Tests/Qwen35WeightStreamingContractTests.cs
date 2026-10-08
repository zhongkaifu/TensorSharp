// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Memory;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public sealed class Qwen35WeightStreamingContractTests
{
    [Fact]
    public void MetadataCountsOnlyWhitelistedResidentConstants()
    {
        Assert.Equal(40, Validate(new[] {
            Weight("token_embd.weight", GgmlTensorType.Q8_0, 32, 128),
            Weight("blk.0.attn_q.weight", GgmlTensorType.Q8_0, 32, 64),
            Weight("blk.0.ssm_conv1d.weight", GgmlTensorType.F32, 3, 2),
            Weight("output_norm.weight", GgmlTensorType.F32, 3),
            Weight("blk.0.attn_q.scale", GgmlTensorType.F32, 1),
        }));
    }

    [Theory]
    [InlineData(GgmlTensorType.F32)]
    [InlineData(GgmlTensorType.F16)]
    [InlineData(GgmlTensorType.Q4_K)]
    [InlineData(GgmlTensorType.PQ2_0)]
    public void DenseMatricesCannotSilentlyMaterializeOrUseAnUnimplementedFormat(GgmlTensorType type)
        => Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.ffn_down.weight", type, 32, 32)
        }));

    [Fact]
    public void UnnamedConstantsLargeConstantsAndAggregateOvercommitAreRejected()
    {
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.unknown.weight", GgmlTensorType.F32, 1)
        }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.ssm_conv1d.weight", GgmlTensorType.F32, 262145)
        }));
        Assert.Throws<NotSupportedException>(() => Validate(
            new[] { Embedding() }.Concat(Enumerable.Range(0, 33)
                .Select(i => Weight($"blk.{i}.ssm_conv1d.weight", GgmlTensorType.F32, 262144)))));
    }

    [Fact]
    public void BlockAlignmentEmbeddingAndScalarScaleAreRequired()
    {
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Weight("token_embd.weight", GgmlTensorType.Q8_0, 31, 32)
        }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Weight("output.weight", GgmlTensorType.Q8_0, 32, 32)
        }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.attn_q.scale", GgmlTensorType.F32, 2)
        }));
    }

    [Fact]
    public void UnsupportedExecutionModesFailBeforeWeightLoading()
    {
        foreach (var backend in new[] { BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.Cuda })
            Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, backend: backend));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, tp: 2));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, split: 2));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, experts: 1));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, mtp: 1));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, draft: "draft.gguf"));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, architecture: "qwen3next"));
    }

    [Fact]
    public void StreamingModeRefusesGraphAndMediaEntryPointsBeforeDereferencingWeights()
    {
        var model = UninitializedModel(streaming: true);
        Assert.False(model.BatchedForwardAvailable);
        Assert.False(model.SupportsBatchedMultimodal);
        Assert.False(model.SupportsPerSequenceFusedForward);
        Assert.False(model.SupportsRetainedFusedCache);
        Assert.False(model.SupportsLinearKVMigration);
        Assert.False(model.SupportsPrefixCheckpoints);
        Assert.False(model.SupportsRetainedCacheSerialization);
        Assert.False(model.TryFullModelDecode(null, 0, Array.Empty<float>()));
        Assert.False(model.TryFullModelVerify(null, 0, 1, null, null));
        Assert.False(model.CanBatchDecode("request", 0));
        Assert.False(model.TryForwardBatchedFusedDecode(null, null, null, null));
        Assert.False(model.TryForwardBatchedFusedDecodeSampled(null, null, null, null));
        Assert.Throws<NotSupportedException>(() => model.ForwardBatch(null));
        Assert.Throws<NotSupportedException>(() => model.BindSequenceCache("request"));
        Assert.Throws<NotSupportedException>(() => model.AdoptPrimaryCacheToFused("request"));
        Assert.Throws<NotSupportedException>(() => model.SubmitGreedyDecodeStep(1));
        Assert.False(model.RetainSequenceCacheAs("request", "prefix"));
        Assert.False(model.TryRebindRetainedCache("prefix", "request"));
        Assert.False(model.TryCloneRetainedCache("prefix", "request"));
        Assert.False(model.TryCheckpointActiveCache("prefix"));
        Assert.False(model.TryImportRetainedCache("prefix", new MemoryStream()));
        Assert.False(model.TryExportRetainedCache("prefix", new MemoryStream()));
        Assert.Throws<NotSupportedException>(() => model.LoadVisionEncoder("must-not-be-opened.gguf"));
        Assert.Throws<NotSupportedException>(() => model.SetVisionEmbeddings(null, 0));
    }

    [Fact]
    public void WeightFreeSpeculationCannotBypassTheStreamedForwardFailureBoundary()
    {
        var model = UninitializedModel(streaming: true);
        ISpeculativeTarget target = model;
        Assert.NotNull(target.SpeculationRefusal);
        Assert.False(target.SpeculationProfitable);
        var speculator = SpeculatorRegistry.Create(target, new SpeculationOptions
        {
            Enabled = true,
            SpeculatorName = SpeculatorRegistry.NGram,
        }, out string decline);
        Assert.Null(speculator);
        Assert.Equal(target.SpeculationRefusal, decline);
        // No model tensors/state are initialized: direct callers must be refused
        // before touching them, even without consulting the capability first.
        Assert.Equal(target.SpeculationRefusal,
            Assert.Throws<NotSupportedException>(() => target.SpecForward(null, null, null, true)).Message);

        var resident = UninitializedModel(streaming: false);
        Assert.Null(((ISpeculativeTarget)resident).SpeculationRefusal);
        Assert.True(resident.SpeculationProfitable);
    }

    private static Qwen35Model UninitializedModel(bool streaming)
    {
        var model = (Qwen35Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35Model));
        GC.SuppressFinalize(model);
        if (streaming)
        {
            var budget = new MemoryBudget(new[] { new MemoryCharge("host", 1 << 20), new MemoryCharge("gpu", 2 << 20) });
            typeof(ModelBase).GetField("WeightStreaming", BindingFlags.Instance | BindingFlags.NonPublic)!
                .SetValue(model, new WeightStreamingOptions(budget, "host", new[] { "gpu" }));
        }
        return model;
    }

    private static long Validate(IEnumerable<GgufTensorInfo> tensors, BackendType backend = BackendType.GgmlCuda,
        int tp = 1, int split = 1, int experts = 0, int mtp = 0, string draft = null, string architecture = "qwen35")
        => Qwen35Model.ValidateStreamingWeightMetadata(architecture, backend, tp, split, experts, mtp, draft, tensors);
    private static GgufTensorInfo Embedding() => Weight("token_embd.weight", GgmlTensorType.Q8_0, 32, 128);
    private static GgufTensorInfo Weight(string name, GgmlTensorType type, params ulong[] shape)
        => new() { Name = name, Type = type, Shape = shape };
}
