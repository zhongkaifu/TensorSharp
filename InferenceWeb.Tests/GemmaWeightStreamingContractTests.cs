// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Memory;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public sealed class GemmaWeightStreamingContractTests
{
    [Fact]
    public void DiagnosticFileFailureDoesNotThrowOrChangeTheInput()
    {
        var model = UninitializedModel();
        string filePath = Path.GetTempFileName();
        try
        {
            File.WriteAllText(filePath, "existing file must remain intact");
            typeof(Gemma4Model).GetField("_gemmaDiagnosticDirectory", BindingFlags.Instance | BindingFlags.NonPublic)!
                .SetValue(model, filePath); // A file cannot become the output directory.
            typeof(Gemma4Model).GetField("_gemmaDiagnosticTag", BindingFlags.Instance | BindingFlags.NonPublic)!
                .SetValue(model, "failure-test");
            using var values = Tensor.FromArray(new CpuAllocator(BlasEnum.DotNet), new float[,] { { 1, -2, 3, 4 } });
            typeof(Gemma4Model).GetMethod("DumpStreamingTensor", BindingFlags.Instance | BindingFlags.NonPublic)!
                .Invoke(model, new object[] { values, "input", -1 });
            Assert.Equal(new float[] { 1, -2, 3, 4 }, values.GetElementsAsFloat(4));
            Assert.Equal("existing file must remain intact", File.ReadAllText(filePath));
            Assert.Null(typeof(Gemma4Model).GetField("_gemmaDiagnosticTag", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
        }
        finally { File.Delete(filePath); }
    }

    [Fact]
    public void PleMatricesStayFileBackedAndOnlySmallNamedF32ParametersCountAsResident()
    {
        Assert.Equal(44, Validate(new[] {
            Embedding(), Weight("per_layer_token_embd.weight", GgmlTensorType.Q8_0, 10752, 262144),
            Weight("per_layer_model_proj.weight", GgmlTensorType.F16, 2560, 10752),
            Weight("blk.0.attn_q.weight", GgmlTensorType.Q8_0, 32, 64),
            Weight("output_norm.weight", GgmlTensorType.F32, 3),
            Weight("per_layer_proj_norm.weight", GgmlTensorType.F32, 2),
            Weight("rope_freqs.weight", GgmlTensorType.F32, 4),
            Weight("blk.0.layer_output_scale.weight", GgmlTensorType.F32, 1),
            Weight("blk.0.attn_q.scale", GgmlTensorType.F32, 1),
        }));
        // Gemma's resident CUDA arithmetic has a narrower shape contract than
        // the generic full-precision F16 streaming kernel.
        Assert.Equal(0, Validate(new[] { Embedding(),
            Weight("per_layer_model_proj.weight", GgmlTensorType.F16, 64, 32) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding(),
            Weight("per_layer_model_proj.weight", GgmlTensorType.F16, 35, 17) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding(),
            Weight("per_layer_model_proj.weight", GgmlTensorType.F16, 32, 32) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding(),
            Weight("per_layer_model_proj.weight", GgmlTensorType.F16, 64, 17) }));
    }

    [Theory]
    [InlineData(GgmlTensorType.F32)]
    [InlineData(GgmlTensorType.F16)]
    [InlineData(GgmlTensorType.Q4_K)]
    [InlineData(GgmlTensorType.PQ2_0)]
    public void OtherMatrixFormatsCannotSilentlyMaterialize(GgmlTensorType type)
        => Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.ffn_down.weight", type, 32, 32)
        }));

    [Fact]
    public void LargeOrUnnamedConstantsAndF16PretendingToBeASmallParameterAreRejected()
    {
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.unknown.weight", GgmlTensorType.F32, 1) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("output_norm.weight", GgmlTensorType.F16, 32) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("per_layer_model_proj.weight", GgmlTensorType.F32, 32, 32) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("output_norm.weight", GgmlTensorType.F32, 262145) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }.Concat(
            Enumerable.Range(0, 33).Select(i => Weight($"blk.{i}.attn_norm.weight", GgmlTensorType.F32, 262144)))));
    }

    [Fact]
    public void QuantizedAlignmentDimensionsEmbeddingAndScalarShapeAreRequired()
    {
        foreach (ulong width in new ulong[] { 0, 31, (ulong)int.MaxValue + 1 })
            Assert.Throws<NotSupportedException>(() => Validate(new[] {
                Weight("token_embd.weight", GgmlTensorType.Q8_0, width, 128) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Weight("output.weight", GgmlTensorType.Q8_0, 32, 128) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.layer_output_scale.weight", GgmlTensorType.F32, 2) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.attn_q.scale", GgmlTensorType.F32, 2) }));
        Assert.Throws<NotSupportedException>(() => Validate(new[] {
            Embedding(), Weight("blk.0.attn_norm.weight", GgmlTensorType.F32, 2, 2) }));
    }

    [Fact]
    public void UnsupportedModesRefuseBeforeLoadingMatrices()
    {
        foreach (var backend in new[] { BackendType.Cpu, BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.Cuda })
            Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, backend: backend));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, tp: 2));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, split: 2));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, experts: 1));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, mtp: 1));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, draft: "draft.gguf"));
        Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, architecture: "gemma4-assistant"));
    }

    [Fact]
    public void StreamingAttentionRequiresUnquantizedKvStorage()
    {
        foreach (KvCacheDtype dtype in Enum.GetValues<KvCacheDtype>())
        {
            if (dtype is KvCacheDtype.F16 or KvCacheDtype.F32)
                Assert.Equal(0, Validate(new[] { Embedding() }, kvCacheDtype: dtype));
            else
                Assert.Throws<NotSupportedException>(() => Validate(new[] { Embedding() }, kvCacheDtype: dtype));
        }
    }

    [Fact]
    public void DirectGraphHolderAndMediaEntryPointsCannotBypassTheForwardFailureBoundary()
    {
        var model = UninitializedModel();
        Assert.False(model.BatchedForwardAvailable);
        Assert.False(model.SupportsPerSequenceFusedForward);
        Assert.False(model.SupportsRetainedFusedCache);
        Assert.False(model.SupportsLinearKVMigration);
        Assert.False(model.SupportsPrefixCheckpoints);
        Assert.False(model.SupportsRetainedCacheSerialization);
        Assert.False(model.SupportsPipelinedGreedy);
        Assert.False(model.CanBatchDecode("request", 0));
        Assert.False(model.TryForwardBatchedFusedDecode(null, null, null, null));
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
        Assert.Throws<NotSupportedException>(() => model.LoadAudioEncoder("must-not-be-opened.gguf"));
        Assert.Throws<NotSupportedException>(() => model.SetVisionEmbeddings(null, 0));
        Assert.Throws<NotSupportedException>(() => model.SetAudioEmbeddings(null, 0));
        Assert.Throws<NotSupportedException>(() => model.LoadMtpDraftWeights("must-not-be-opened.gguf"));
        Assert.Throws<NotSupportedException>(() => model.LoadDFlashDraftWeights("must-not-be-opened.gguf"));
        Assert.Throws<NotSupportedException>(() => model.DraftStep(0, null, 0, null, null));
    }

    [Fact]
    public void WeightFreeSpeculationCannotBypassTheStreamedForwardFailureBoundary()
    {
        ISpeculativeTarget target = UninitializedModel();
        Assert.NotNull(target.SpeculationRefusal);
        Assert.False(target.SpeculationProfitable);
        var speculator = SpeculatorRegistry.Create(target, new SpeculationOptions {
            Enabled = true, SpeculatorName = SpeculatorRegistry.NGram,
        }, out string decline);
        Assert.Null(speculator);
        Assert.Equal(target.SpeculationRefusal, decline);
        Assert.Equal(target.SpeculationRefusal,
            Assert.Throws<NotSupportedException>(() => target.SpecForward(null, null, null, true)).Message);
    }

    private static Gemma4Model UninitializedModel()
    {
        var model = (Gemma4Model)RuntimeHelpers.GetUninitializedObject(typeof(Gemma4Model));
        GC.SuppressFinalize(model);
        var budget = new MemoryBudget(new[] { new MemoryCharge("host", 1 << 20), new MemoryCharge("gpu", 2 << 20) });
        typeof(ModelBase).GetField("WeightStreaming", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new WeightStreamingOptions(budget, "host", new[] { "gpu" }));
        return model;
    }

    private static long Validate(IEnumerable<GgufTensorInfo> tensors, BackendType backend = BackendType.GgmlCuda,
        int tp = 1, int split = 1, int experts = 0, int mtp = 0, string draft = null, string architecture = "gemma4",
        KvCacheDtype kvCacheDtype = KvCacheDtype.F16)
        => Gemma4Model.ValidateStreamingWeightMetadata(architecture, backend, tp, split, experts, mtp, draft, tensors, kvCacheDtype);
    private static GgufTensorInfo Embedding() => Weight("token_embd.weight", GgmlTensorType.Q8_0, 32, 128);
    private static GgufTensorInfo Weight(string name, GgmlTensorType type, params ulong[] shape)
        => new() { Name = name, Type = type, Shape = shape };
}
