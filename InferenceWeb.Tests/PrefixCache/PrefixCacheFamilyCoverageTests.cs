// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>
/// Every engine-served family implements the prefix-cache contract and its diagnostics (DESIGN §6,
/// "common to every family"), and holder families take their end-state members from the Runtime's
/// adapter. A new engine-served family that does not fails here instead of silently staying outside
/// the default radix tree.
/// </summary>
[Trait("Category", "PrefixCacheUnit")]
public sealed class PrefixCacheFamilyCoverageTests
{
    [Theory]
    [InlineData(typeof(DeepSeek41Model))]
    [InlineData(typeof(DeepSeek4Model))]
    [InlineData(typeof(Gemma4Model))]
    [InlineData(typeof(GlmDsaModel))]
    [InlineData(typeof(GptOssModel))]
    [InlineData(typeof(HunyuanDenseModel))]
    [InlineData(typeof(Mistral3Model))]
    [InlineData(typeof(MuseGlimmerModel))]
    [InlineData(typeof(NemotronModel))]
    [InlineData(typeof(Qwen35Model))]
    [InlineData(typeof(Qwen3Model))]
    [InlineData(typeof(Qwen4ExpModel))]
    public void EveryEngineServedFamily_EnablesTreeWithoutAdvertisingUnavailableEndStates(Type family)
    {
        // Capabilities describe construction-time geometry and backend availability.
        // An unloaded CPU instance must enable the owner while refusing operations
        // that need GPU holders or native slots. Block-quantized K/V also avoids
        // Qwen3's paged path, whose availability inspects loaded projection weights.
        var model = (ModelBase)RuntimeHelpers.GetUninitializedObject(family);
        typeof(ModelBase).GetField("<Config>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new ModelConfig
            {
                Architecture = family.Name, NumLayers = 2, NumHeads = 4, NumKVHeads = 2, HiddenSize = 128,
            });
        typeof(ModelBase).GetField("_kvCacheDtype", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, KvCacheDtype.Q8_0);
        typeof(ModelBase).GetField("<ExecutionPlan>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new BackendExecutionPlan(BackendType.Cpu));

        PrefixCacheCapabilities caps = ((IPrefixCacheModel)model).GetPrefixCacheCapabilities();

        Assert.Equal(PrefixCacheMode.Tree, caps.Readiness);
        Assert.Equal(model.KVStateFingerprint, caps.NamespaceFingerprint);
        Assert.False(string.IsNullOrWhiteSpace(caps.NamespaceFingerprint));
        Assert.Equal(EndStateSupport.None, caps.EndState);
        Assert.False(caps.CanCaptureCopy);
        Assert.False(caps.AdoptPrimaryOnDisplacement);
        Assert.False(caps.Persistable);

        // Native paged attention rows lack Nemotron's matching recurrent state;
        // enabling the radix owner must never turn them into reusable prefixes.
        if (family == typeof(NemotronModel))
        {
            Assert.True(caps.PagesNeedStateAtEnd);
            Assert.Equal(PageSupport.None, caps.Pages);
        }
    }

    [Fact]
    public void EveryEngineServedFamily_ImplementsTheContractAndItsDiagnostics()
    {
        Type[] engineServed = typeof(ModelBase).Assembly.GetTypes()
            .Where(t => !t.IsAbstract && typeof(ModelBase).IsAssignableFrom(t)
                        && (typeof(IBatchedPagedModel).IsAssignableFrom(t) || t == typeof(MuseGlimmerModel)))
            .OrderBy(t => t.Name)
            .ToArray();
        Assert.NotEmpty(engineServed);
        foreach (Type family in engineServed)
        {
            Assert.True(typeof(IPrefixCacheModel).IsAssignableFrom(family), $"{family.Name} does not implement IPrefixCacheModel");
            Assert.True(typeof(IPrefixCacheModelDiagnostics).IsAssignableFrom(family), $"{family.Name} does not implement IPrefixCacheModelDiagnostics");
        }
        Assert.Equal(
            new[] { "DeepSeek41Model", "DeepSeek4Model", "Gemma4Model", "GlmDsaModel", "GptOssModel", "HunyuanDenseModel",
                    "Mistral3Model", "MuseGlimmerModel", "NemotronModel", "Qwen35Model", "Qwen3Model", "Qwen4ExpModel" },
            engineServed.Select(t => t.Name).ToArray());
    }

    [Theory]
    [InlineData(typeof(Gemma4Model))]
    [InlineData(typeof(Qwen35Model))]
    [InlineData(typeof(Qwen4ExpModel))]
    [InlineData(typeof(DeepSeek4Model))]
    public void HolderFamilies_UseTheHolderAdapter(Type family)
        => Assert.True(typeof(IHolderPrefixCacheModel).IsAssignableFrom(family), $"{family.Name} is not an IHolderPrefixCacheModel");

    [Theory]
    [InlineData(typeof(Qwen3Model))]
    [InlineData(typeof(GptOssModel))]
    [InlineData(typeof(Mistral3Model))]
    [InlineData(typeof(HunyuanDenseModel))]
    [InlineData(typeof(NemotronModel))]
    [InlineData(typeof(MuseGlimmerModel))]
    public void PageFamilies_RefuseEveryEndStateMember(Type family)
        => Assert.True(typeof(IPageOnlyPrefixCacheModel).IsAssignableFrom(family), $"{family.Name} is not an IPageOnlyPrefixCacheModel");
}
