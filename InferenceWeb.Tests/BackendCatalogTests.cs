using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public class BackendCatalogTests
{
    [Fact]
    public void GetSupportedBackends_OnlyReturnsAvailableBackendsInUiOrder()
    {
        var backends = BackendCatalog.GetSupportedBackends(
            backendType => backendType switch
            {
                GgmlBackendType.Metal => false,
                GgmlBackendType.Cuda => true,
                GgmlBackendType.Vulkan => true,
                GgmlBackendType.Cpu => true,
                _ => false,
            },
            () => true,
            () => true);

        Assert.Collection(backends,
            backend =>
            {
                Assert.Equal("mlx", backend.Value);
                Assert.Equal("MLX Metal (GPU)", backend.Label);
            },
            backend =>
            {
                Assert.Equal("cuda", backend.Value);
                Assert.Equal("CUDA (cuBLAS GPU)", backend.Label);
            },
            backend =>
            {
                Assert.Equal("ggml_cuda", backend.Value);
                Assert.Equal("GGML CUDA (GPU)", backend.Label);
            },
            backend =>
            {
                Assert.Equal("ggml_vulkan", backend.Value);
                Assert.Equal("GGML Vulkan (GPU)", backend.Label);
            },
            backend =>
            {
                Assert.Equal("ggml_cpu", backend.Value);
                Assert.Equal("GGML CPU", backend.Label);
            },
            backend =>
            {
                Assert.Equal("cpu", backend.Value);
                Assert.Equal("CPU (Pure C#)", backend.Label);
            });
    }

    [Fact]
    public void GetSupportedBackends_AlwaysIncludesBothCpuBackends()
    {
        var backends = BackendCatalog.GetSupportedBackends(_ => false, () => false, () => false);

        Assert.Collection(backends,
            backend =>
            {
                Assert.Equal("ggml_cpu", backend.Value);
                Assert.Equal("GGML CPU", backend.Label);
            },
            backend =>
            {
                Assert.Equal("cpu", backend.Value);
                Assert.Equal("CPU (Pure C#)", backend.Label);
            });
    }

    [Fact]
    public void ResolveDefaultBackend_PrefersConfiguredBackendWhenSupported()
    {
        var supportedBackends = new[]
        {
            new BackendOption("cuda", "CUDA (cuBLAS GPU)"),
            new BackendOption("ggml_cpu", "GGML CPU"),
        };

        string backend = BackendCatalog.ResolveDefaultBackend("cuda", supportedBackends);

        Assert.Equal("cuda", backend);
    }

    [Fact]
    public void ResolveDefaultBackend_FallsBackToFirstSupportedBackend()
    {
        var supportedBackends = new[]
        {
            new BackendOption("ggml_cpu", "GGML CPU"),
            new BackendOption("cpu", "CPU (Pure C#)"),
        };

        string backend = BackendCatalog.ResolveDefaultBackend("ggml_metal", supportedBackends);

        Assert.Equal("ggml_cpu", backend);
    }

    [Theory]
    [InlineData(BackendType.Mlx, "mlx")]
    [InlineData(BackendType.Cuda, "cuda")]
    [InlineData(BackendType.GgmlMetal, "ggml_metal")]
    [InlineData(BackendType.GgmlCuda, "ggml_cuda")]
    [InlineData(BackendType.GgmlVulkan, "ggml_vulkan")]
    [InlineData(BackendType.GgmlCpu, "ggml_cpu")]
    [InlineData(BackendType.Cpu, "cpu")]
    public void ToBackendValue_ReturnsCanonicalBackendString(BackendType backendType, string expected)
    {
        Assert.Equal(expected, BackendCatalog.ToBackendValue(backendType));
    }

    [Theory]
    [InlineData("mlx", "mlx")]
    [InlineData("mlx_metal", "mlx")]
    [InlineData("mlx-metal", "mlx")]
    public void Canonicalize_MapsMlxAliases(string backend, string expected)
    {
        Assert.Equal(expected, BackendCatalog.Canonicalize(backend));
    }

    [Fact]
    public void ShouldStoreWeightQuantized_PureCpuBackendKeepsSupportedWeightsCompressed()
    {
        var info = new GgufTensorInfo
        {
            Name = "blk.0.attn_q.weight",
            Type = GgmlTensorType.Q8_0,
            Shape = new ulong[] { 128, 256 }
        };

        bool shouldStoreQuantized = ModelBase.ShouldStoreWeightQuantized(BackendType.Cpu, info);

        Assert.True(shouldStoreQuantized);
    }

    [Fact]
    public void ShouldStoreWeightQuantized_GgmlBackendsKeepQuantizedWeights()
    {
        var info = new GgufTensorInfo
        {
            Name = "blk.0.attn_q.weight",
            Type = GgmlTensorType.Q8_0,
            Shape = new ulong[] { 128, 256 }
        };

        bool shouldStoreQuantized = ModelBase.ShouldStoreWeightQuantized(BackendType.GgmlCpu, info);

        Assert.True(shouldStoreQuantized);
    }

    [Fact]
    public void ShouldStoreWeightQuantized_DirectCudaKeepsSupportedWeightsCompressed()
    {
        var info = new GgufTensorInfo
        {
            Name = "blk.0.attn_q.weight",
            Type = GgmlTensorType.Q8_0,
            Shape = new ulong[] { 128, 256 }
        };

        bool shouldStoreQuantized = ModelBase.ShouldStoreWeightQuantized(BackendType.Cuda, info);

        Assert.True(shouldStoreQuantized);
    }

    [Theory]
    [InlineData(GgmlTensorType.IQ2_XXS)]
    [InlineData(GgmlTensorType.IQ2_S)]
    [InlineData(GgmlTensorType.MXFP4)]
    public void ShouldStoreWeightQuantized_DirectCudaKeepsHostFallbackWeightsCompressed(GgmlTensorType type)
    {
        var info = new GgufTensorInfo
        {
            Name = "blk.0.ffn_gate_exps.weight",
            Type = type,
            Shape = new ulong[] { 256, 512, 8 }
        };

        bool shouldStoreQuantized = ModelBase.ShouldStoreWeightQuantized(BackendType.Cuda, info);

        Assert.True(shouldStoreQuantized);
    }

    [Fact]
    public void ShouldStoreWeightQuantized_DirectCudaDoesNotKeepIntegerWeightsCompressed()
    {
        var info = new GgufTensorInfo
        {
            Name = "blk.0.some_index.weight",
            Type = GgmlTensorType.I32,
            Shape = new ulong[] { 128, 256 }
        };

        bool shouldStoreQuantized = ModelBase.ShouldStoreWeightQuantized(BackendType.Cuda, info);

        Assert.False(shouldStoreQuantized);
    }

    [Fact]
    public void ShouldStoreWeightQuantized_MlxKeepsSupportedWeightsCompressed()
    {
        var info = new GgufTensorInfo
        {
            Name = "blk.0.attn_q.weight",
            Type = GgmlTensorType.Q8_0,
            Shape = new ulong[] { 128, 256 }
        };

        bool shouldStoreQuantized = ModelBase.ShouldStoreWeightQuantized(BackendType.Mlx, info);

        Assert.True(shouldStoreQuantized);
    }

    [Fact]
    public void ShouldStoreWeightQuantized_MlxKeepsIq4XsCompressed()
    {
        var info = new GgufTensorInfo
        {
            Name = "blk.0.attn_q.weight",
            Type = GgmlTensorType.IQ4_XS,
            Shape = new ulong[] { 256, 512 }
        };

        bool shouldStoreQuantized = ModelBase.ShouldStoreWeightQuantized(BackendType.Mlx, info);

        Assert.True(shouldStoreQuantized);
    }
}


