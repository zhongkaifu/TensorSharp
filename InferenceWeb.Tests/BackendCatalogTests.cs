using InferenceEngine;
using InferenceWeb;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public class BackendCatalogTests
{
    [Fact]
    public void GetSupportedBackends_OnlyReturnsAvailableBackendsInUiOrder()
    {
        var backends = BackendCatalog.GetSupportedBackends(backendType => backendType switch
        {
            GgmlBackendType.Metal => false,
            GgmlBackendType.Cuda => true,
            GgmlBackendType.Cpu => true,
            _ => false,
        });

        Assert.Collection(backends,
            backend =>
            {
                Assert.Equal("ggml_cuda", backend.Value);
                Assert.Equal("GGML CUDA (GPU)", backend.Label);
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
        var backends = BackendCatalog.GetSupportedBackends(_ => false);

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
            new BackendOption("ggml_cuda", "GGML CUDA (GPU)"),
            new BackendOption("ggml_cpu", "GGML CPU"),
        };

        string backend = BackendCatalog.ResolveDefaultBackend("cuda", supportedBackends);

        Assert.Equal("ggml_cuda", backend);
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
    [InlineData(BackendType.GgmlMetal, "ggml_metal")]
    [InlineData(BackendType.GgmlCuda, "ggml_cuda")]
    [InlineData(BackendType.GgmlCpu, "ggml_cpu")]
    [InlineData(BackendType.Cpu, "cpu")]
    public void ToBackendValue_ReturnsCanonicalBackendString(BackendType backendType, string expected)
    {
        Assert.Equal(expected, BackendCatalog.ToBackendValue(backendType));
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
}
