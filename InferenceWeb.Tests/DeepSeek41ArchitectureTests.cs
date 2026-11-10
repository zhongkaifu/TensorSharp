// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

public class DeepSeek41ArchitectureTests : IDisposable
{
    private readonly EnvScope _env = new();

    public DeepSeek41ArchitectureTests()
    {
        _env.ClearSpeculationVars();
        _env.Set("TS_DSV41_ENGRAM_DEVICE", null);
        _env.Set("TS_DSV41_ALLOW_NON_CUDA_GPU", null);
    }

    public void Dispose() => _env.Dispose();

    /// <summary>
    /// TS_DSV4_UBATCH unset: V4.1 on a ggml GPU backend asks the native loader to
    /// choose (1024/512/256, never trading an extra host routed-expert layer for
    /// prefill width); its CPU executors and the direct-CUDA engine keep 256.
    /// </summary>
    [Theory]
    [InlineData(BackendType.GgmlCuda, GgmlDeepSeek4Native.UBatchAuto)]
    [InlineData(BackendType.GgmlVulkan, GgmlDeepSeek4Native.UBatchAuto)]
    [InlineData(BackendType.GgmlMetal, GgmlDeepSeek4Native.UBatchAuto)]
    [InlineData(BackendType.GgmlCpu, 256)]
    [InlineData(BackendType.Cpu, 256)]
    [InlineData(BackendType.Cuda, 256)]
    public void V41PrefillWidthIsAutomaticOnlyOnGgmlGpuBackends(BackendType backend, int expected)
    {
        Assert.Equal(-1, GgmlDeepSeek4Native.UBatchAuto);
        foreach (string unset in new[] { null, "", "  " })
        {
            Assert.Equal(expected, DeepSeek4Model.ResolveNativeUbatch(true, backend, unset, out string warning));
            Assert.Null(warning);
        }
    }

    /// <summary>Any explicit positive TS_DSV4_UBATCH is used verbatim and turns the
    /// automatic choice off, on every backend and for both architectures.</summary>
    [Theory]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCpu)]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.GgmlVulkan)]
    public void AnExplicitPrefillWidthIsUsedVerbatim(BackendType backend)
    {
        foreach (bool v41 in new[] { true, false })
            foreach (int width in new[] { 32, 256, 512, 1024, 2048 })
            {
                Assert.Equal(width, DeepSeek4Model.ResolveNativeUbatch(v41, backend, width.ToString(), out string warning));
                Assert.Null(warning);
            }
    }

    /// <summary>Plain V4 keeps its defaults: 512 on the pure C# executor, 1024 elsewhere.</summary>
    [Theory]
    [InlineData(BackendType.Cpu, 512)]
    [InlineData(BackendType.GgmlCuda, 1024)]
    [InlineData(BackendType.GgmlCpu, 1024)]
    [InlineData(BackendType.Cuda, 1024)]
    public void V4PrefillWidthDefaultsAreUnchanged(BackendType backend, int expected)
        => Assert.Equal(expected, DeepSeek4Model.ResolveNativeUbatch(false, backend, null, out _));

    /// <summary>A set value that is not a positive width is reported, not silently
    /// replaced -- and "-1" is not a way to spell the automatic choice, which is
    /// what leaving the variable unset means.</summary>
    [Theory]
    [InlineData("abc")]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("1.5")]
    public void AnInvalidPrefillWidthIsReportedAndIgnored(string configured)
    {
        Assert.Equal(GgmlDeepSeek4Native.UBatchAuto,
            DeepSeek4Model.ResolveNativeUbatch(true, BackendType.GgmlCuda, configured, out string warning));
        Assert.NotNull(warning);
        Assert.Contains("TS_DSV4_UBATCH", warning);
        Assert.Contains(configured, warning);
        Assert.Equal(256, DeepSeek4Model.ResolveNativeUbatch(true, BackendType.GgmlCpu, configured, out warning));
        Assert.NotNull(warning);
    }

    [Fact]
    public void DescriptorIsIndependentAndDeclaresTensorParallelismWithExplicitLayerSplit()
    {
        Assert.True(ModelArchitectureRegistry.TryGet("deepseek41", out var v41));
        Assert.True(ModelArchitectureRegistry.TryGet("deepseek4", out var v4));
        Assert.NotSame(v4, v41);
        Assert.True(v41.Serving.TensorParallel.IsSupportedOn(BackendType.GgmlCuda));
        Assert.True(v41.Serving.LayerSplit.IsSupportedOn(BackendType.GgmlCuda));
        Assert.Empty(v41.Serving.HybridParallel.Backends);
        TensorSharp.ITensorParallelGroup group = null;
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(v41, BackendType.GgmlCuda, 1, ref group, out int split, 4));
        Assert.Equal(4, split);
        Assert.Contains("LAYER SPLIT", v41.DescribeMultiGpuPlacement(4));
    }

    /// <summary>
    /// What is left without an implementation: MLX, and the ggml GPU backends
    /// until the opt-in is set. BackendType.Cpu and BackendType.Cuda both run
    /// V4.1 now -- the pure C# executor and the direct-CUDA engine respectively.
    /// </summary>
    [Theory]
    [InlineData(BackendType.Mlx)]
    [InlineData(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlVulkan)]
    public void UnsupportedBackendsRefusedBeforeLoadingWeights(BackendType backend)
    {
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", backend, null));
        Assert.Contains("ggml_cuda", error.Message);
        Assert.Contains("ggml_cpu", error.Message);
    }

    /// <summary>
    /// The refusal names ggml_cuda as the one serving backend. It used to file
    /// the direct-CUDA engine under "serving" too, although that engine has no
    /// numerical gate yet and the model card calls it a correctness and
    /// portability path like the two CPU executors.
    /// </summary>
    [Fact]
    public void BackendRefusalNamesOnlyGgmlCudaAsTheServingBackend()
    {
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.Mlx, null));
        Assert.Contains("--backend ggml_cuda (the serving backend)", error.Message);
        Assert.Contains("--backend cuda (the direct-CUDA engine, not yet held to a numerical gate)", error.Message);
        Assert.Contains("correctness and portability paths only", error.Message);
        Assert.DoesNotContain("--backend cuda (serving)", error.Message);
    }

    /// <summary>
    /// A non-CUDA ggml GPU backend runs V4.1 with this architecture's ops on the
    /// CPU backend: correct, but a host round trip per occurrence. It is opt-in
    /// because what the refusal originally closed was a SILENT fallback onto
    /// whichever GPU enumerated first, not an explicit request.
    /// </summary>
    [Theory]
    [InlineData(BackendType.GgmlVulkan)]
    [InlineData(BackendType.GgmlMetal)]
    public void NonCudaGpuBackendsReachTheLoadOnlyWhenAskedFor(BackendType backend)
    {
        var refused = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", backend, null));
        Assert.Contains("TS_DSV41_ALLOW_NON_CUDA_GPU", refused.Message);

        _env.Set("TS_DSV41_ALLOW_NON_CUDA_GPU", "1");
        // Backend validation succeeds before the loader reads embedded metadata.
        DeepSeek41Architecture.ValidateLoad("missing.gguf", backend, null);
    }

    /// <summary>The opt-in covers ggml GPU backends only; it does not open a
    /// backend that has no V4.1 implementation at all.</summary>
    [Theory]
    [InlineData(BackendType.Mlx)]
    public void TheNonCudaOptInDoesNotOpenABackendWithoutAnImplementation(BackendType backend)
    {
        _env.Set("TS_DSV41_ALLOW_NON_CUDA_GPU", "1");
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", backend, null));
        Assert.Contains("ggml_cuda", error.Message);
    }

    /// <summary>
    /// The direct-CUDA engine implements V4.1 with its own kernels, so it
    /// reaches the load like the ggml one. It is a separate backend that shares
    /// nothing with ggml_cuda.
    /// </summary>
    [Fact]
    public void TheDirectCudaEngineReachesTheLoad()
    {
        string directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            DeepSeek41Architecture.ValidateLoad(Path.Combine(directory, "model.gguf"), BackendType.Cuda, null);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    /// <summary>
    /// The pure C# executor implements V4.1's graph -- the compressors at
    /// ratios 1 and 2, the shared compressed caches and sparse selection, the
    /// Engram tables, the delayed hyper-connection gates and the trained cache
    /// quantization -- and is checked against the PyTorch reference by
    /// Dsv41CpuExecutorTests. It is a portability path, not a serving one, so
    /// it reaches the load rather than being turned away.
    /// </summary>
    [Fact]
    public void TheManagedCpuExecutorReachesTheLoad()
    {
        string directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            DeepSeek41Architecture.ValidateLoad(Path.Combine(directory, "model.gguf"), BackendType.Cpu, null);
            Assert.Contains("--backend cpu", DeepSeek41Architecture.DescribeCpuBackendChoice(BackendType.Cpu));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    /// <summary>
    /// A q8_0/q4_0 cache used to survive the load: the native graph allocated
    /// F16 caches regardless and <c>KvCacheDtype</c> reported the requested
    /// tier. It is now refused before the checkpoint is opened, on every
    /// executor, with the reason.
    /// </summary>
    [Theory]
    [InlineData(KvCacheDtype.Q8_0, BackendType.GgmlCuda)]
    [InlineData(KvCacheDtype.Q8_0, BackendType.Cuda)]
    [InlineData(KvCacheDtype.Q8_0, BackendType.GgmlCpu)]
    [InlineData(KvCacheDtype.Q8_0, BackendType.Cpu)]
    [InlineData(KvCacheDtype.Q4_0, BackendType.GgmlCuda)]
    [InlineData(KvCacheDtype.Q4_0, BackendType.Cpu)]
    public void BlockQuantizedKvCacheIsRefusedBeforeTheLoad(KvCacheDtype dtype, BackendType backend)
    {
        KvCacheDtype restoreDtype = KvCacheDtypeConfig.Current;
        bool restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        KvCacheDtypeConfig.Set(dtype);
        try
        {
            var error = Assert.Throws<NotSupportedException>(() =>
                DeepSeek41Architecture.ValidateLoad("missing.gguf", backend, null));
            Assert.StartsWith($"KV_CACHE_DTYPE={dtype.ToShortString()} is not supported by DeepSeek V4.1 Flash", error.Message);
            Assert.Contains("F16", error.Message);
            Assert.Contains("no block-dequantize step", error.Message);
            Assert.Contains("NVFP4", error.Message);
            Assert.Contains("Unset KV_CACHE_DTYPE or set it to f16", error.Message);
            Assert.Equal(error.Message, DeepSeek4Architecture.BlockQuantizedKvCacheError("DeepSeek V4.1 Flash", dtype, v41: true));
        }
        finally { KvCacheDtypeConfig.RestoreForTests(restoreDtype, restoreExplicit); }
    }

    /// <summary>The V4 message must not claim V4.1's trained cache quantization.</summary>
    [Fact]
    public void PlainV4RefusalDoesNotClaimTheV41CacheQuantization()
    {
        string message = DeepSeek4Architecture.BlockQuantizedKvCacheError("DeepSeek V4 (Flash)", KvCacheDtype.Q8_0, v41: false);
        Assert.StartsWith("KV_CACHE_DTYPE=q8_0 is not supported by DeepSeek V4 (Flash)", message);
        Assert.DoesNotContain("NVFP4", message);
        Assert.Contains("Unset KV_CACHE_DTYPE or set it to f16", message);
    }

    [Theory]
    [InlineData(KvCacheDtype.F16)]
    [InlineData(KvCacheDtype.F32)]
    public void FloatKvCacheRequestsReachTheLoad(KvCacheDtype dtype)
    {
        KvCacheDtype restoreDtype = KvCacheDtypeConfig.Current;
        bool restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        KvCacheDtypeConfig.Set(dtype);
        string directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            DeepSeek41Architecture.ValidateLoad(Path.Combine(directory, "model.gguf"), BackendType.GgmlCuda, null);
        }
        finally
        {
            KvCacheDtypeConfig.RestoreForTests(restoreDtype, restoreExplicit);
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// ggml_cpu drives the native loader's cpu_only branch, where every V4.1
    /// fused op runs its scalar CPU implementation. It has to reach the load,
    /// not be turned away with the V4 executors.
    /// </summary>
    [Theory]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCpu)]
    public void SupportedBackendsReachTheLoad(BackendType backend)
    {
        string directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            DeepSeek41Architecture.ValidateLoad(Path.Combine(directory, "model.gguf"), backend, null);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    /// <summary>
    /// The server's default backend is ggml_cpu off macOS, so omitting
    /// --backend on a GPU box used to be a refusal and now loads. It must not
    /// load silently.
    /// </summary>
    [Fact]
    public void CpuBackendSaysWhatItIsBeforeTheLoad()
    {
        string note = DeepSeek41Architecture.DescribeCpuBackendChoice(BackendType.GgmlCpu);
        Assert.Contains("ONE CPU device", note);
        Assert.Contains("not a serving path", note);
        Assert.Contains("--backend ggml_cuda", note);
        Assert.Contains("TS_DSV4_THREADS", note);
        Assert.Null(DeepSeek41Architecture.DescribeCpuBackendChoice(BackendType.GgmlCuda));
    }

    /// <summary>
    /// Routed-MoE tensor parallelism shards expert dimensions across GPUs, so
    /// the native loader rejects it under cpu_only. Reject it here first, before
    /// a 246 GiB checkpoint is opened.
    /// </summary>
    [Fact]
    public void RoutedMoeTensorParallelIsRefusedOnTheCpuBackend()
    {
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.GgmlCpu, null, tpDegree: 4));
        Assert.Contains("--tp", error.Message);
        Assert.Contains("ggml_cuda", error.Message);
    }

    /// <summary>
    /// TS_DSV41_ENGRAM_DEVICE places the Engram tables in VRAM; the native
    /// loader reads it only off its cpu_only branch. Accepting it here would
    /// leave "=1" -- require GPU residency, fail if it does not fit -- silently
    /// ignored, and would accept a value ggml_cuda rejects. "=0" asks for the
    /// host mappings this backend always uses, so it stays legal.
    /// </summary>
    [Theory]
    [InlineData("1")]
    [InlineData("2")]
    public void GpuResidentEngramTablesAreRefusedOnTheCpuBackend(string value)
    {
        _env.Set("TS_DSV41_ENGRAM_DEVICE", value);
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.GgmlCpu, null));
        Assert.Contains("TS_DSV41_ENGRAM_DEVICE", error.Message);
    }

    [Fact]
    public void HostMappedEngramTablesRemainSelectableOnTheCpuBackend()
    {
        _env.Set("TS_DSV41_ENGRAM_DEVICE", "0");
        DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.GgmlCpu, null);
    }

    /// <summary>
    /// Placement stays native's decision on ggml_cuda: mirroring its rule twice
    /// is how the two spellings drift apart.
    /// </summary>
    [Theory]
    [InlineData("1")]
    [InlineData("0")]
    public void EngramPlacementIsLeftToTheLoaderOnTheGpuBackend(string value)
    {
        _env.Set("TS_DSV41_ENGRAM_DEVICE", value);
        DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.GgmlCuda, null);
    }

    /// <summary>
    /// V4 reaches the same loader, and ggml_cpu used to run it on the GPUs
    /// there. It is also the backend the server picks when --backend is
    /// omitted off macOS, so the switch must announce itself.
    /// </summary>
    [Fact]
    public void V4AlsoSaysWhenTheCpuBackendTakesItOffTheGpus()
    {
        string note = DeepSeek4Model.DescribeCpuBackendChoice(BackendType.GgmlCpu);
        Assert.Contains("ONE CPU device", note);
        Assert.Contains("not on any GPU", note);
        Assert.Contains("--backend ggml_cuda", note);
        Assert.Null(DeepSeek4Model.DescribeCpuBackendChoice(BackendType.GgmlCuda));
        Assert.Null(DeepSeek4Model.DescribeCpuBackendChoice(BackendType.Cpu));
    }

    [Theory]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCpu)]
    public void BackendValidationDoesNotRequireExternalEngramFiles(BackendType backend)
    {
        string model = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "model.gguf");
        DeepSeek41Architecture.ValidateLoad(model, backend, null);
    }

    [Fact]
    public void V4DraftCannotBeAppliedToV41()
    {
        var error = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateDsparkArchitecture("deepseek4-dspark"));
        Assert.Contains("DSpark", error.Message);
    }

    [Fact]
    public void V4DraftFileCannotBeAppliedToV41()
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0UL);
                writer.Write(1UL);
                static void WriteString(BinaryWriter w, string text)
                {
                    byte[] bytes = System.Text.Encoding.UTF8.GetBytes(text);
                    w.Write((ulong)bytes.Length);
                    w.Write(bytes);
                }
                WriteString(writer, "general.architecture");
                writer.Write(8u);
                WriteString(writer, "deepseek4-dspark");
                while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
            }
            var error = Assert.Throws<NotSupportedException>(() =>
                DeepSeek41Architecture.ValidateLoad("missing.gguf", BackendType.GgmlCuda, path));
            Assert.Contains("DSpark", error.Message);
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(null)]
    [InlineData("deepseek_v4_flash_dspark_draft")]
    [InlineData("qwen4exp")]
    [InlineData("deepseek41")]
    public void DraftArchitectureMustBeExplicitlyV41(string? architecture)
        => Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ValidateDsparkArchitecture(architecture));

    [Fact]
    public void V41DraftArchitectureReachesNativeTensorValidation()
        => DeepSeek41Architecture.ValidateDsparkArchitecture("deepseek41-dspark");

    [Fact]
    public void LlamaCppV41DflashExportReachesNativeTensorValidation()
    {
        var keys = new HashSet<string> { "dflash.hyper_connection.count", "dflash.attention.output_group_count" };
        DeepSeek41Architecture.ValidateDsparkDraft("dflash", keys.Contains);
        DeepSeek41Architecture.ValidateDsparkDraft("deepseek41-dspark", _ => false);
    }

    [Theory]
    [InlineData("dflash.hyper_connection.count")]
    [InlineData("dflash.attention.output_group_count")]
    [InlineData(null)]
    public void OtherDflashDraftersAreRefusedForV41(string? onlyKey)
        => Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateDsparkDraft("dflash", key => key == onlyKey));

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.Cuda)]
    public void V41DraftRefusesExecutorsWithoutItsDraftGraph(BackendType backend)
        => Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("missing.gguf", backend, "v41-draft.gguf"));

    [Theory]
    [InlineData(1, 0)]
    [InlineData(2, 2)]
    [InlineData(4, 4)]
    [InlineData(8, 8)]
    public void TensorParallelRanksComeFromTheTpDegree(int degree, int expected)
        => Assert.Equal(expected, DeepSeek41Architecture.ResolveTensorParallelRanks(degree));

    [Fact]
    public void TensorParallelDegreeAboveEightIsRefused()
        => Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ResolveTensorParallelRanks(9));

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    public void ExplicitTensorParallelDegreeReachesTheLoader(int degree)
    {
        Assert.Equal(degree, DeepSeek41Architecture.ResolveTensorParallelRanks(degree));
        Assert.True(DeepSeek41Architecture.Descriptor.Serving.TensorParallel.IsSupportedOn(BackendType.GgmlCuda));
        Assert.Null(DeepSeek41Architecture.Descriptor.Serving.TensorParallel.Precondition(degree, BackendType.GgmlCuda));
        Assert.NotNull(DeepSeek41Architecture.Descriptor.Serving.TensorParallel.Precondition(9, BackendType.GgmlCuda));
        DeepSeek41Architecture.ValidateLoad("not-opened.gguf", BackendType.GgmlCuda, null, tpDegree: degree);
    }

    [Theory]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlVulkan)]
    [InlineData(BackendType.GgmlMetal)]
    public void ExplicitTensorParallelDegreeRefusesUnsupportedBackendsBeforeOpeningWeights(BackendType backend)
    {
        Assert.False(DeepSeek41Architecture.Descriptor.Serving.TensorParallel.IsSupportedOn(backend));
        Assert.Throws<NotSupportedException>(() =>
            new DeepSeek41Model("not-opened.gguf", backend, tpDegree: 6));
    }

    [Fact]
    public void PlacementMessagesNameTheMode()
    {
        Assert.Contains("4 GPUs by LAYER SPLIT", DeepSeek41Architecture.Descriptor.DescribeMultiGpuPlacement(4));
        string message = DeepSeek41Architecture.DescribeTensorParallelPlacement(2);
        Assert.Contains("across 2 GPUs", message);
        Assert.Contains("gate/up/down", message);
        Assert.Contains("F32 activation/output gathers", message);
        Assert.Contains("CPU-offloaded layers", message);
        Assert.Contains("Attention and shared experts retain layer placement", message);
        Assert.DoesNotContain("shards no weights", message);
    }

    [Fact]
    public void NativeRoutedMoeShardingDoesNotCreateManagedCollectives()
    {
        TensorSharp.ITensorParallelGroup group = null;
        Assert.Equal(4, ModelBase.ResolveTensorParallelSupport(DeepSeek41Architecture.Descriptor,
            BackendType.GgmlCuda, 4, ref group, out int split));
        Assert.Equal(1, split);
        Assert.Null(group);
    }
}
