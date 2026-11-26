// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron's per-layer MoE geometry on the shared runtime's model (NemotronRuntimeModel). Bolt stores expert_used_count as
// a per-layer array whose element [0] - a Mamba2 layer - is 0, and expert_count as a scalar larger than any layer's stacked
// expert tensor. Each MoE layer takes its expert count from its stacked tensor and its top-k from the array, a top-k no
// layer can route is refused naming the layer, the load line reports the resolved values, and tensor parallelism shards and
// routes each layer by its own counts.
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NemotronPerLayerMoEConfigTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-nemotron-moecfg-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();

    public NemotronPerLayerMoEConfigTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
    }

    public void Dispose()
    {
        _environment.Dispose();
        try { Directory.Delete(_directory, true); } catch { /* best effort */ }
    }

    [Fact]
    public void PerLayerFixture_LoadsWithResolvedArraysAndLoadLine()
    {
        // Layer 1: 8 experts top-3; layer 3: 4 experts top-2; expert_used_count is an array whose element [0] (an
        // attention layer) is 0.
        string path = new NemotronSyntheticModelBuilder
        {
            Layout = "AMAM",
            LayerExperts = new[] { 8, 4 },
            UsedPerLayer = new uint[] { 0, 3, 0, 2 },
        }.Write(Path.Combine(_directory, "per-layer.gguf"));

        var console = new StringWriter();
        var original = Console.Out;
        NemotronRuntimeModel model;
        Console.SetOut(console);
        try { model = new NemotronRuntimeModel(path, BackendType.Cpu); }
        finally { Console.SetOut(original); }

        using (model)
        {
            Assert.Equal((0, 0), model.ExpertLayoutForTests(0));
            Assert.Equal((8, 3), model.ExpertLayoutForTests(1));
            Assert.Equal((0, 0), model.ExpertLayoutForTests(2));
            Assert.Equal((4, 2), model.ExpertLayoutForTests(3));
            Assert.Contains("MoE: 2 layers, 4-8 experts, top-2-3", console.ToString());

            float[] logits = model.Forward(new[] { 3, 17, 42, 9, 100, 7, 64, 11, 5 });
            Assert.All(logits, v => Assert.True(float.IsFinite(v)));
            float[] decode = model.Forward(new[] { 12 });
            Assert.All(decode, v => Assert.True(float.IsFinite(v)));
        }
    }

    [Theory]
    [InlineData(0u)]
    [InlineData(9u)]
    public void TopKOutsideOneToExperts_IsRefusedNamingTheLayer(uint badUsed)
    {
        string path = new NemotronSyntheticModelBuilder
        {
            Layout = "AMAM",
            LayerExperts = new[] { 8, 8 },
            UsedPerLayer = new uint[] { 0, 2, 0, badUsed },
        }.Write(Path.Combine(_directory, $"bad-{badUsed}.gguf"));

        var ex = Assert.Throws<InvalidDataException>(() => new NemotronRuntimeModel(path, BackendType.Cpu));
        Assert.Contains("layer 3", ex.Message);
        Assert.Contains($"expert_used_count {badUsed}", ex.Message);
    }

    [Fact]
    public void PerLayerFixture_UnderTensorParallelism_FollowsTheSingleDeviceTrajectory()
    {
        // Bolt-shaped: each MoE block has its own expert count and top-k, with scalar .scale sidecars and NVFP4 expert
        // scales; every rank routes and shards each layer by its own counts. Both models run F32 K/V, as in
        // NemotronExpertScaleTests, whose trajectory check this is.
        string path = new NemotronSyntheticModelBuilder
        {
            Layout = "AMAM",
            LayerExperts = new[] { 8, 4 },
            UsedPerLayer = new uint[] { 0, 3, 0, 2 },
            Scaled = true,
        }.Write(Path.Combine(_directory, "per-layer-tp.gguf"));

        KvCacheDtype restoreDtype = KvCacheDtypeConfig.Current;
        bool restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        try
        {
            using var single = new NemotronRuntimeModel(path, BackendType.Cpu);
            using var tp = new NemotronRuntimeModel(path, BackendType.Cpu, 2, new SimulatedDeviceGroup(2));
            Assert.Equal(2, tp.Mesh.TensorParallel);
            NemotronExpertScaleTests.AssertSameTrajectory(single, tp, 2e-3f);
        }
        finally
        {
            KvCacheDtypeConfig.RestoreForTests(restoreDtype, restoreExplicit);
        }
    }
}
