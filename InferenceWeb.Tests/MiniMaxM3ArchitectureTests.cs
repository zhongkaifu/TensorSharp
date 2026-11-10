// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The minimax-m3 descriptor and every refusal it makes before a checkpoint is
// opened: unsupported backends, --tp off ggml_cuda or with a layer split or a
// drafter, distributed groups, block-quantized K/V, a --draft-model that is not
// there and an unknown TS_MINIMAX_M3_MSA value.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

/// <summary>Tests that set TS_MINIMAX_M3_* variables or the process-wide KV dtype; they run alone.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class MiniMaxM3EnvironmentCollection
{
    public const string Name = "MiniMax-M3 environment";
}

[Collection(MiniMaxM3EnvironmentCollection.Name)]
public sealed class MiniMaxM3ArchitectureTests : IDisposable
{
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly string _restoreMsa = Environment.GetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable);

    public MiniMaxM3ArchitectureTests()
    {
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, null);
        // No choice made: the request a fresh process has (no model writes its own default back).
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, false);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, _restoreMsa);
        KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
    }

    private static ModelArchitectureDescriptor Arch()
    {
        Assert.True(ModelArchitectureRegistry.TryGet("minimax-m3", out var descriptor));
        return descriptor;
    }

    private static ModelCreateContext Context(BackendType backend, ITensorParallelGroup group = null, string draft = null)
        => new("/model-is-not-opened.gguf", backend, probe: null, tpDegree: 1, tpGroup: group, draftModelPath: draft);

    [Fact]
    public void DescriptorIsRegisteredUnderBothSpellings()
    {
        var arch = Arch();
        Assert.Equal("minimax-m3", arch.Id);
        Assert.Equal("MiniMax-M3.1", arch.DisplayName);
        Assert.True(ModelArchitectureRegistry.TryGet("minimax_m3", out var alias));
        Assert.Same(arch, alias);
        Assert.True(ModelArchitectureRegistry.TryGet("MiniMax-M3", out var upper));
        Assert.Same(arch, upper);
        // MiniMax-M2 (no sparse attention, no vision) and MiniMax-H3 are other families.
        Assert.False(ModelArchitectureRegistry.TryGet("minimax-m2", out var m2) && ReferenceEquals(m2, arch));
        Assert.True(ModelArchitectureRegistry.TryGet("minimax-h3", out var h3));
        Assert.NotSame(arch, h3);
        Assert.Contains("*mmproj*MiniMax*M3*.gguf", arch.ProjectorFileHints);
    }

    [Fact]
    public void TensorParallelAndLayerSplitBothRunOnCuda()
    {
        var arch = Arch();
        Assert.Equal(new[] { BackendType.GgmlCuda }, arch.Serving.TensorParallel.Backends);
        Assert.Equal(new[] { BackendType.GgmlCuda }, arch.Serving.LayerSplit.Backends);
        Assert.False(arch.SupportsDistributedTensorParallel);
        Assert.Null(arch.Serving.TensorParallel.LoopbackBackend);
        Assert.Empty(arch.Serving.HybridParallel.Backends);
        Assert.Contains("MiniMax-M3", arch.Serving.HybridParallel.GapReason, StringComparison.Ordinal);

        ITensorParallelGroup group = null;
        int tp = ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlCuda, 1, ref group, out int split, 8);
        Assert.Equal(1, tp);
        Assert.Equal(8, split);
        Assert.Null(group);

        // --tp N is its own mode: every layer split over N GPUs, no layer split.
        Assert.Equal(8, ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlCuda, 8, ref group, out split));
        Assert.Equal(1, split);
        Assert.Null(group);
        // ...and never combined with a layer split, nor on a backend without several devices.
        Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlCuda, 2, ref group, out _, 2));
        Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlCpu, 2, ref group, out _));

        Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlCpu, 1, ref group, out _, 2));
        Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlVulkan, 1, ref group, out _, 2));
    }

    [Theory]
    [InlineData(BackendType.Mlx, "MLX")]
    [InlineData(BackendType.GgmlMetal, "Metal")]
    [InlineData(BackendType.GgmlVulkan, "Vulkan")]
    [InlineData(BackendType.Cuda, "direct-CUDA")]
    [InlineData(BackendType.Cpu, "pure C#")]
    public void UnsupportedBackendsAreRefusedBeforeTheCheckpointOpens(BackendType backend, string named)
    {
        var error = Assert.Throws<NotSupportedException>(() => Arch().ApplyNativeTunables(Context(backend)));
        Assert.Contains(named, error.Message, StringComparison.Ordinal);
        Assert.Contains("--backend ggml_cuda", error.Message, StringComparison.Ordinal);
        // A direct construction refuses the same way, before the base constructor maps the file.
        var direct = Assert.Throws<NotSupportedException>(() => new MiniMaxM3Model("/model-is-not-opened.gguf", backend));
        Assert.Equal(error.Message, direct.Message);
    }

    [Theory]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCpu)]
    public void NativeExecutorBackendsPass(BackendType backend)
        => Arch().ApplyNativeTunables(Context(backend));

    [Fact]
    public void DistributedGroupIsRefused()
    {
        var group = new StubTpGroup();
        var error = Assert.Throws<NotSupportedException>(() => Arch().ApplyNativeTunables(Context(BackendType.GgmlCuda, group)));
        Assert.Contains("--tp-node-id/--tp-peers", error.Message, StringComparison.Ordinal);

        ITensorParallelGroup live = group;
        var gate = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(Arch(), BackendType.GgmlCuda, 2, ref live, out _));
        Assert.Contains("minimax-m3", gate.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorParallelConstructionRefusesWhatItCannotRun()
    {
        // Off ggml_cuda (the meta backend spans CUDA GPUs), with a layer split, or with a drafter:
        // refused before the checkpoint is opened.
        var cpu = Assert.Throws<NotSupportedException>(() =>
            new MiniMaxM3Model("/model-is-not-opened.gguf", BackendType.GgmlCpu, tpDegree: 2));
        Assert.Equal(MiniMaxM3Architecture.TensorParallelBackendRefusal, cpu.Message);
        var split = Assert.Throws<NotSupportedException>(() =>
            new MiniMaxM3Model("/model-is-not-opened.gguf", BackendType.GgmlCuda, tpDegree: 2, layerSplitDegree: 2));
        Assert.Contains("--layer-split and --tp", split.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DraftModelUnderTensorParallelismIsRefused()
    {
        var error = Assert.Throws<NotSupportedException>(() =>
            new MiniMaxM3Model("/model-is-not-opened.gguf", BackendType.GgmlCuda, tpDegree: 2,
                draftModelPath: "/MiniMax-M3.1-Flash-DSpark-BF16.gguf"));
        // Today every drafter is refused first; once DSpark lands, --tp still refuses it.
        Assert.Contains("--draft-model", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(KvCacheDtype.Q8_0)]
    [InlineData(KvCacheDtype.Q4_0)]
    public void BlockQuantizedKvCacheIsRefused(KvCacheDtype dtype)
    {
        KvCacheDtypeConfig.Set(dtype);
        var error = Assert.Throws<NotSupportedException>(() => Arch().ApplyNativeTunables(Context(BackendType.GgmlCuda)));
        Assert.Contains($"KV_CACHE_DTYPE={dtype.ToShortString()}", error.Message, StringComparison.Ordinal);
        Assert.Contains("index-key", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void FloatKvCacheRequestsPass()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        Arch().ApplyNativeTunables(Context(BackendType.GgmlCuda));
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        Arch().ApplyNativeTunables(Context(BackendType.GgmlCuda));
    }

    [Fact]
    public void DraftModelThatIsNotThereIsRefusedBeforeTheTargetIsMapped()
    {
        // The drafter itself is validated in MiniMaxM3DsparkTests; here only that the
        // descriptor checks it before the 232 GiB target is opened.
        var error = Assert.Throws<NotSupportedException>(() =>
            Arch().ApplyNativeTunables(Context(BackendType.GgmlCuda, draft: "/MiniMax-M3.1-Flash-DSpark-BF16.gguf")));
        Assert.Contains("--draft-model", error.Message, StringComparison.Ordinal);
        Assert.Contains("does not exist", error.Message, StringComparison.Ordinal);
        Arch().ApplyNativeTunables(Context(BackendType.GgmlCuda, draft: "  "));
    }

    [Theory]
    [InlineData("auto")]
    [InlineData("dense")]
    [InlineData("ggml")]
    [InlineData("kernel")]
    public void KnownAttentionModesPass(string mode)
    {
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, mode);
        Arch().ApplyNativeTunables(Context(BackendType.GgmlCuda));
    }

    [Fact]
    public void UnknownAttentionModeIsRefused()
    {
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, "sparse");
        var error = Assert.Throws<NotSupportedException>(() => Arch().ApplyNativeTunables(Context(BackendType.GgmlCuda)));
        Assert.Contains("TS_MINIMAX_M3_MSA=sparse", error.Message, StringComparison.Ordinal);
        Assert.Contains("auto, dense, ggml, kernel", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("minimax-m3")]
    [InlineData("minimax_m3")]
    public void AudioInputIsRefused(string architecture)
        => Assert.Equal(AudioInputSupport.MiniMaxM3Message, AudioInputSupport.UnsupportedReasonFor(architecture));

    [Fact]
    public void RefusalsCountAsLoadRefusals()
    {
        var error = Assert.Throws<NotSupportedException>(() => Arch().ApplyNativeTunables(Context(BackendType.Mlx)));
        Assert.True(TensorSharp.Runtime.ModelLoadRefusal.TryDescribe(error, out string line));
        Assert.Contains("MLX", line, StringComparison.Ordinal);
    }

    /// <summary>Minimal live group: the gate only reads whether one exists.</summary>
    private sealed class StubTpGroup : ITensorParallelGroup
    {
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }
}
