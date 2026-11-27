// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron 3.5 Super VL stores every dense projection in BF16 (Mamba-2 in/out, attention, the latent MoE projections, the
// LM head). At its widths those products take the GPU executors' BF16 kernels; each backend must follow the managed
// backend's F32 trajectory over BF16 projections. Direct CUDA keeps a product's activations F32 (its BF16 decode kernels,
// and the prompt steps' F32 products): within 2e-2. ggml-cuda rounds a multi-row BF16 product's activations to BF16 for its
// GEMM (llama.cpp's arithmetic): up to 0.038 in a logit of the prompt steps here.
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

[Collection("MoeCpuOffloadConfig")]
public sealed class NemotronBf16ProjectionTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-nemotron-bf16-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public NemotronBf16ProjectionTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        try { Directory.Delete(_directory, true); } catch { /* best effort */ }
    }

    private string Write(bool latent, bool hostExperts)
        => new NemotronSyntheticModelBuilder { LinearType = SyntheticGguf.GgmlType.BF16, Width = 512, Latent = latent, Scaled = hostExperts }
            .Write(Path.Combine(_directory, $"bf16-{latent}-{hostExperts}.gguf"));

    [CudaTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Bf16Projections_OnDirectCuda_FollowTheManagedBackend(bool latent, bool hostExperts) => Follow(BackendType.Cuda, latent, hostExperts, 2e-2f);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Bf16Projections_OnGgmlCuda_FollowTheManagedBackend(bool latent, bool hostExperts)
        => Follow(TestGates.PinnedGgmlBackend, latent, hostExperts, 6e-2f);

    /// <summary>Super VL routes 22 of its 512 experts a token: the device's top-k over the router's scores must pick what the
    /// managed backend's does past 16 routes too.</summary>
    [CudaTheory]
    [InlineData(22, false)]
    [InlineData(22, true)]
    public void WideTopK_OnDirectCuda_FollowsTheManagedBackend(int topK, bool hostExperts) => FollowWide(BackendType.Cuda, topK, hostExperts, 2e-2f);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(22, false)]
    [InlineData(22, true)]
    public void WideTopK_OnGgmlCuda_FollowsTheManagedBackend(int topK, bool hostExperts) => FollowWide(TestGates.PinnedGgmlBackend, topK, hostExperts, 6e-2f);

    private void FollowWide(BackendType backend, int topK, bool hostExperts, float tolerance)
    {
        string path = new NemotronSyntheticModelBuilder { Experts = 64, TopK = topK, Latent = true, Scaled = true }
            .Write(Path.Combine(_directory, $"wide-{topK}-{hostExperts}.gguf"));
        Follow(backend, path, hostExperts, tolerance);
    }

    private void Follow(BackendType backend, bool latent, bool hostExperts, float tolerance)
        => Follow(backend, Write(latent, hostExperts), hostExperts, tolerance);

    private static void Follow(BackendType backend, string path, bool hostExperts, float tolerance)
    {
        using var reference = new NemotronRuntimeModel(path, BackendType.Cpu);
        if (hostExperts) MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            using var model = new NemotronRuntimeModel(path, backend);
            NemotronExpertScaleTests.AssertSameTrajectory(reference, model, tolerance);
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }
}
