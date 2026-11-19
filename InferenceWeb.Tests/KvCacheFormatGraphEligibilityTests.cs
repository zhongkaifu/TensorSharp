// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Which K/V cache formats a model's whole-forward graphs may hold (IGraphExecutor.TakesKvCacheFormat): F32 and F16 on every
/// executor; Q8_0 and Q4_0 where every view and write of a cache addresses its blocks (GGML's QuantizedKvCache), and q8_0 on
/// direct CUDA, whose flash kernels and row writes take it (heads of 128 or 256, K as wide as V) - for the families whose
/// attention is those kernels' alone (Qwen3.8 Flash Next's QSA layers keep it off). Off the graphs the model says why.
/// </summary>
public sealed class KvCacheFormatGraphEligibilityTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("kv-format").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private sealed class Executor : TensorGraphExecutor
    {
        private readonly GraphCapabilities _caps;

        public Executor(bool quantizedKv) : base(new CpuAllocator(BlasEnum.DotNet))
            => _caps = new GraphCapabilities { QuantizedKvCache = quantizedKv };

        public override GraphCapabilities Capabilities => _caps;

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x) => throw new NotSupportedException();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BlockFormatsNeedAnExecutorThatTakesThem(bool quantizedKv)
    {
        IGraphExecutor executor = new Executor(quantizedKv);
        Assert.True(executor.TakesKvCacheFormat(GraphFormat.F32, 128, 128));
        Assert.True(executor.TakesKvCacheFormat(GraphFormat.F16, 256, 256));
        Assert.Equal(quantizedKv, executor.TakesKvCacheFormat(GraphFormat.Q8_0, 256, 256));
        Assert.Equal(quantizedKv, executor.TakesKvCacheFormat(GraphFormat.Q4_0, 128, 128));
        Assert.False(executor.TakesKvCacheFormat(GraphFormat.BF16, 128, 128));
    }

    /// <summary>The managed executor takes no block format: a q8_0 cache keeps the model off its graphs, and says so. (The
    /// managed backend cannot hold a q4_0 cache at all: its fill of a narrowed q4_0 tensor throws at load.)</summary>
    [Fact]
    public void TheManagedExecutorRefusesABlockCacheByName()
    {
        using IDisposable kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q8_0);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 4 }.Write(Path.Combine(_dir, "q35.gguf"));
        using var model = new Qwen35RuntimeModel(path, BackendType.Cpu);
        Assert.Equal("KV dtype Q8_0 on this backend's executor", model.GraphRefusalForTests);
    }

    /// <summary>Direct CUDA takes q8_0 exactly where its flash kernels run it (an image with their sm_80 code), never for heads
    /// they do not have or K and V of different widths, and never q4_0; F16 always.</summary>
    [CudaFact]
    public void DirectCudaTakesQ8OnlyWhereItsFlashKernelsRunIt()
    {
        using var allocator = new CudaAllocator();
        using var executor = new CudaGraphExecutor(allocator);
        bool runs = allocator.FlashPrefillKernels is { } flash && flash.RunsAttentionQ8(256);
        Assert.Equal(runs, executor.TakesKvCacheFormat(GraphFormat.Q8_0, 256, 256));
        Assert.Equal(runs, executor.AttendsCapacityWindows(256, GraphFormat.Q8_0));
        Assert.False(executor.TakesKvCacheFormat(GraphFormat.Q8_0, 64, 64));
        Assert.False(executor.TakesKvCacheFormat(GraphFormat.Q8_0, 256, 128));
        Assert.False(executor.TakesKvCacheFormat(GraphFormat.Q4_0, 256, 256));
        Assert.True(executor.TakesKvCacheFormat(GraphFormat.F16, 64, 64));
    }

    /// <summary>Direct CUDA with its flash kernels' sm_80 code: q8_0 for 128- and 256-wide heads whose K and V widths agree.
    /// Where the loaded module has no such code (the committed compute_75 PTX, an sm_7x build) the gate skips it.</summary>
    [CudaFact(TensorCoreModule = "tensorsharp_flash_prefill")]
    public void DirectCudaTakesQ8ForItsFlashKernelsHeads()
    {
        using var allocator = new CudaAllocator();
        using var executor = new CudaGraphExecutor(allocator);
        Assert.True(allocator.FlashPrefillKernels is { } flash && flash.RunsAttentionQ8(256),
            "the loaded tensorsharp_flash_prefill module carries sm_80 code but runs no q8_0 attention (a stale image?)");
        Assert.True(executor.TakesKvCacheFormat(GraphFormat.Q8_0, 256, 256));
        Assert.True(executor.TakesKvCacheFormat(GraphFormat.Q8_0, 128, 128));
        Assert.True(executor.AttendsCapacityWindows(256, GraphFormat.Q8_0));
    }

    /// <summary>On direct CUDA q4_0 keeps a model off whole-forward graphs, and q8_0 does where the flash kernels cannot run it,
    /// saying so.</summary>
    [CudaFact]
    public void DirectCudaGraphsRefuseQ4AndQ8WithoutItsKernels()
    {
        string dense = new Qwen35DenseSyntheticModelBuilder { Layers = 4 }.Write(Path.Combine(_dir, "q35.gguf"));
        using (KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q4_0))
        using (var model = new Qwen35RuntimeModel(dense, BackendType.Cuda))
            Assert.Equal("KV dtype Q4_0 on this backend's executor", model.GraphRefusalForTests);
        bool runs;
        using (var allocator = new CudaAllocator())
            runs = allocator.FlashPrefillKernels is { } flash && flash.RunsAttentionQ8(Qwen35DenseSyntheticModelBuilder.HeadDim);
        using (KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q8_0))
        using (var model = new Qwen35RuntimeModel(dense, BackendType.Cuda))
            Assert.Equal(runs ? null : "KV dtype Q8_0 on this backend's executor", model.GraphRefusalForTests);
    }

    /// <summary>On direct CUDA with its flash kernels' sm_80 code a q8_0 Qwen 3.5 runs as graphs; q8_0 keeps a model with QSA
    /// layers off them (their indexer cache and selection kernels read F16). Where the loaded module has no such code the gate
    /// skips it.</summary>
    [CudaFact(TensorCoreModule = "tensorsharp_flash_prefill")]
    public void DirectCudaGraphsTakeQ8ForPlainAttentionFamilies()
    {
        string dense = new Qwen35DenseSyntheticModelBuilder { Layers = 4 }.Write(Path.Combine(_dir, "q35.gguf"));
        using (KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q8_0))
        using (var model = new Qwen35RuntimeModel(dense, BackendType.Cuda))
            Assert.Null(model.GraphRefusalForTests);
        string qsa = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 8, qsa: true);
        using (KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q8_0))
        using (var model = new Qwen4ExpRuntimeModel(qsa, BackendType.Cuda))
            Assert.Equal($"KV dtype Q8_0 with {model.Config.Architecture}'s attention routes on this backend's executor", model.GraphRefusalForTests);
    }
}
