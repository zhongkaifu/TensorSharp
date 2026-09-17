// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>
/// Every engine-served family implements the prefix-cache contract and its diagnostics (DESIGN §6,
/// "common to every family"), and holder families take their end-state members from the Runtime's
/// adapter. A new engine-served family that does not fails here instead of silently staying outside
/// the tree when M3 raises every family to Shadow.
/// </summary>
[Trait("Category", "PrefixCacheUnit")]
public sealed class PrefixCacheFamilyCoverageTests
{
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
