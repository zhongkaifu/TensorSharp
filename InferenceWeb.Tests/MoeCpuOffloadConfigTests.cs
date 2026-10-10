// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Coverage for the MoE CPU-offload policy layer (<c>--n-cpu-moe</c> /
/// <c>--cpu-moe</c>, llama.cpp's <c>-ncmoe</c> / <c>-cmoe</c> equivalent).
///
/// The two things that must be exactly right are (a) which layers are placed on
/// the host and (b) which GGUF tensor names are consequently kept out of VRAM.
/// Getting (b) wrong in the permissive direction silently spends the VRAM the
/// flag exists to save; getting it wrong in the restrictive direction strands a
/// weight the GPU path still needs.
///
/// <see cref="MoeCpuOffloadConfig"/> is process-wide state, so every test resets
/// it on the way in and out. The collection attribute serializes them against
/// each other.
/// </summary>
[Collection("MoeCpuOffloadConfig")]
public sealed class MoeCpuOffloadConfigTests : IDisposable
{
    public MoeCpuOffloadConfigTests() => MoeCpuOffloadConfig.Reset();

    public void Dispose() => MoeCpuOffloadConfig.Reset();

    [Fact]
    public void NativeCapacityPlan_DistinguishesUnsetFromExplicitZeroAndReset()
    {
        Assert.Equal(-1, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: true));
        Assert.Equal(0, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: false));
        Assert.False(MoeCpuOffloadConfig.IsEnabled); // other loaders are unchanged
        MoeCpuOffloadConfig.SetLayers(0);
        Assert.Equal(0, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: true));
        MoeCpuOffloadConfig.SetLayers(7);
        Assert.Equal(7, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: true));
        Assert.Equal(7, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: false));
        MoeCpuOffloadConfig.SetAllLayers();
        Assert.Equal(int.MaxValue, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: true));
        MoeCpuOffloadConfig.Reset();
        Assert.Equal(-1, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: true));
        Assert.Equal(0, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: false));
    }

    [Fact]
    public void Default_IsDisabled()
    {
        Assert.False(MoeCpuOffloadConfig.IsEnabled);
        Assert.False(MoeCpuOffloadConfig.IsExplicitlySet);
        Assert.Equal(0, MoeCpuOffloadConfig.CpuMoeLayers);
        Assert.False(MoeCpuOffloadConfig.AllLayers);
        Assert.Null(MoeCpuOffloadConfig.Describe());
        // With the feature off, nothing is withheld from the accelerator.
        Assert.False(MoeCpuOffloadConfig.IsLayerOnCpu(0));
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.0.ffn_gate_exps.weight"));
    }

    [Fact]
    public void SetLayers_OffloadsTheFirstNLayersOnly()
    {
        MoeCpuOffloadConfig.SetLayers(32);

        Assert.True(MoeCpuOffloadConfig.IsEnabled);
        Assert.True(MoeCpuOffloadConfig.IsExplicitlySet);
        Assert.Equal(32, MoeCpuOffloadConfig.CpuMoeLayers);

        Assert.True(MoeCpuOffloadConfig.IsLayerOnCpu(0));
        Assert.True(MoeCpuOffloadConfig.IsLayerOnCpu(31));
        // Boundary: llama.cpp's --n-cpu-moe N covers layers [0, N), so layer N
        // itself stays resident.
        Assert.False(MoeCpuOffloadConfig.IsLayerOnCpu(32));
        Assert.False(MoeCpuOffloadConfig.IsLayerOnCpu(47));
    }

    [Fact]
    public void SetLayers_Zero_IsAValidNoOp()
    {
        MoeCpuOffloadConfig.SetLayers(0);
        Assert.False(MoeCpuOffloadConfig.IsEnabled);
        Assert.False(MoeCpuOffloadConfig.IsLayerOnCpu(0));
    }

    [Fact]
    public void SetLayers_Negative_Throws()
        => Assert.Throws<ArgumentOutOfRangeException>(() => MoeCpuOffloadConfig.SetLayers(-1));

    [Fact]
    public void SetAllLayers_OffloadsEveryLayer()
    {
        MoeCpuOffloadConfig.SetAllLayers();

        Assert.True(MoeCpuOffloadConfig.IsEnabled);
        Assert.True(MoeCpuOffloadConfig.AllLayers);
        Assert.True(MoeCpuOffloadConfig.IsLayerOnCpu(0));
        Assert.True(MoeCpuOffloadConfig.IsLayerOnCpu(1000));
        Assert.Equal("all layers", MoeCpuOffloadConfig.Describe());
    }

    [Fact]
    public void SetAllLayers_ThenSetLayers_ReplacesThePolicy()
    {
        MoeCpuOffloadConfig.SetAllLayers();
        MoeCpuOffloadConfig.SetLayers(4);

        Assert.False(MoeCpuOffloadConfig.AllLayers);
        Assert.True(MoeCpuOffloadConfig.IsLayerOnCpu(3));
        Assert.False(MoeCpuOffloadConfig.IsLayerOnCpu(4));
    }

    [Theory]
    [InlineData("0", 0, false)]
    [InlineData("32", 32, false)]
    [InlineData("  8  ", 8, false)]
    [InlineData("all", 0, true)]
    [InlineData("ALL", 0, true)]
    [InlineData("max", 0, true)]
    public void TryParse_AcceptsCountsAndAllAliases(string value, int expectedLayers, bool expectedAll)
    {
        Assert.True(MoeCpuOffloadConfig.TryParse(value, out int layers, out bool all));
        Assert.Equal(expectedLayers, layers);
        Assert.Equal(expectedAll, all);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("-1")]
    [InlineData("half")]
    [InlineData("3.5")]
    public void TryParse_RejectsGarbage(string? value)
        => Assert.False(MoeCpuOffloadConfig.TryParse(value, out _, out _));

    [Theory]
    [InlineData("blk.0.ffn_gate_exps.weight", 0)]
    [InlineData("blk.7.ffn_up_exps.weight", 7)]
    [InlineData("blk.31.ffn_down_exps.3.weight", 31)]
    [InlineData("blk.127.attn_q.weight", 127)]
    public void TryParseLayerIndex_ReadsTheGgufBlockPrefix(string name, int expected)
    {
        Assert.True(MoeCpuOffloadConfig.TryParseLayerIndex(name, out int layer));
        Assert.Equal(expected, layer);
    }

    [Theory]
    [InlineData("token_embd.weight")]
    [InlineData("output_norm.weight")]
    [InlineData("v.blk.0.attn_q.weight")]   // vision tower: not a trunk layer
    [InlineData("blk..weight")]
    [InlineData("blk.x.attn_q.weight")]
    public void TryParseLayerIndex_RejectsNamesWithoutABlockIndex(string name)
        => Assert.False(MoeCpuOffloadConfig.TryParseLayerIndex(name, out _));

    [Theory]
    [InlineData("blk.0.ffn_gate_exps.weight")]
    [InlineData("blk.0.ffn_up_exps.weight")]
    [InlineData("blk.0.ffn_down_exps.weight")]
    [InlineData("blk.0.ffn_gate_exps.5.weight")]   // per-expert split view
    public void IsRoutedExpertWeightName_MatchesTheExpsInfix(string name)
        => Assert.True(MoeCpuOffloadConfig.IsRoutedExpertWeightName(name));

    [Theory]
    [InlineData("blk.0.ffn_gate_shexp.weight")]    // shared expert: always on GPU
    [InlineData("blk.0.ffn_up_shexp.weight")]
    [InlineData("blk.0.ffn_gate_inp.weight")]      // router: always on GPU
    [InlineData("blk.0.attn_q.weight")]
    [InlineData("blk.0.ffn_gate.weight")]          // dense FFN
    [InlineData("token_embd.weight")]
    [InlineData("")]
    [InlineData(null)]
    public void IsRoutedExpertWeightName_LeavesEverythingElseAlone(string? name)
        => Assert.False(MoeCpuOffloadConfig.IsRoutedExpertWeightName(name));

    [Fact]
    public void IsOffloadedExpertWeightName_CombinesLayerAndTensorRole()
    {
        MoeCpuOffloadConfig.SetLayers(4);

        // Routed experts inside the offloaded range: kept off the accelerator.
        Assert.True(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.0.ffn_gate_exps.weight"));
        Assert.True(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.3.ffn_down_exps.7.weight"));

        // Same tensor role, outside the range: stays resident.
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.4.ffn_gate_exps.weight"));
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.40.ffn_gate_exps.weight"));

        // Inside the range but NOT a routed expert: attention, router and the
        // always-active shared expert all keep their device copy, otherwise the
        // host would be doing work the GPU should.
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.0.attn_q.weight"));
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.0.ffn_gate_inp.weight"));
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.0.ffn_gate_shexp.weight"));

        // Weights with no layer index (embeddings, LM head) are never offloaded.
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("token_embd.weight"));
    }

    [Fact]
    public void IsOffloadedExpertWeightName_AllLayers_CoversEveryBlock()
    {
        MoeCpuOffloadConfig.SetAllLayers();

        Assert.True(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.0.ffn_gate_exps.weight"));
        Assert.True(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.93.ffn_down_exps.weight"));
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("blk.93.attn_k.weight"));
        // No block prefix at all: still nothing to offload even in "all" mode,
        // because there is no MoE layer to place it on.
        Assert.False(MoeCpuOffloadConfig.IsOffloadedExpertWeightName("output.weight"));
    }

    [Fact]
    public void Describe_ReportsTheActivePolicy()
    {
        MoeCpuOffloadConfig.SetLayers(12);
        Assert.Equal("first 12 layer(s)", MoeCpuOffloadConfig.Describe());
    }

    [Fact]
    public void SetCpuThreads_PublishesTheNativeEnvironmentVariable()
    {
        try
        {
            MoeCpuOffloadConfig.SetCpuThreads(9);
            Assert.Equal(9, MoeCpuOffloadConfig.CpuThreads);
            // The host matmul runs inside GgmlOps.dll and reads the count from
            // the environment, so the flag has to land there too.
            Assert.Equal("9", Environment.GetEnvironmentVariable("TS_CPU_MOE_THREADS"));

            MoeCpuOffloadConfig.SetCpuThreads(0);
            Assert.Equal(0, MoeCpuOffloadConfig.CpuThreads);
            Assert.Null(Environment.GetEnvironmentVariable("TS_CPU_MOE_THREADS"));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_CPU_MOE_THREADS", null);
        }
    }

    [Fact]
    public void ConfigureFromEnvironment_ReadsLayerCount()
    {
        try
        {
            Environment.SetEnvironmentVariable("TS_N_CPU_MOE", "16");
            MoeCpuOffloadConfig.ConfigureFromEnvironment();

            Assert.True(MoeCpuOffloadConfig.IsEnabled);
            Assert.Equal(16, MoeCpuOffloadConfig.CpuMoeLayers);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_N_CPU_MOE", null);
        }
    }

    [Fact]
    public void ConfigureFromEnvironment_ReadsAllLayersSwitch()
    {
        try
        {
            Environment.SetEnvironmentVariable("TS_CPU_MOE", "1");
            MoeCpuOffloadConfig.ConfigureFromEnvironment();

            Assert.True(MoeCpuOffloadConfig.AllLayers);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_CPU_MOE", null);
        }
    }

    [Fact]
    public void ConfigureFromEnvironment_TreatsZeroAsOff()
    {
        try
        {
            Environment.SetEnvironmentVariable("TS_CPU_MOE", "0");
            Environment.SetEnvironmentVariable("TS_N_CPU_MOE", "0");
            MoeCpuOffloadConfig.ConfigureFromEnvironment();

            Assert.False(MoeCpuOffloadConfig.IsEnabled);
            Assert.True(MoeCpuOffloadConfig.IsExplicitlySet);
            Assert.Equal(0, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: true));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_CPU_MOE", null);
            Environment.SetEnvironmentVariable("TS_N_CPU_MOE", null);
        }
    }

    [Fact]
    public void ConfigureFromEnvironment_DoesNotOverrideAnExplicitFlag()
    {
        try
        {
            // The CLI parses --n-cpu-moe before this runs; the env var must lose.
            MoeCpuOffloadConfig.SetLayers(4);
            Environment.SetEnvironmentVariable("TS_N_CPU_MOE", "40");
            MoeCpuOffloadConfig.ConfigureFromEnvironment();

            Assert.Equal(4, MoeCpuOffloadConfig.CpuMoeLayers);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_N_CPU_MOE", null);
        }
    }

    [Theory]
    [InlineData("0", null, 0)]
    [InlineData(null, "0", 0)]
    [InlineData("8", "0", 8)]
    public void EnvironmentPolicy_PreservesExplicitZeroAndLayerPrecedence(string? layers, string? all, int expected)
    {
        using var env = new NativeEnvScope();
        env.Set("TS_N_CPU_MOE", layers);
        env.Set("TS_CPU_MOE", all);
        MoeCpuOffloadConfig.ConfigureFromEnvironment();
        Assert.True(MoeCpuOffloadConfig.IsExplicitlySet);
        Assert.Equal(expected, MoeCpuOffloadConfig.ResolveNativeLayers(planWhenUnspecified: true));
    }
}
