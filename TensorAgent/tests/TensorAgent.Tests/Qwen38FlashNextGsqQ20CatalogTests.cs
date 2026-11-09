// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using TensorAgent.Core.Catalog;

namespace TensorAgent.Tests;

/// <summary>
/// The fastest Qwen3.8 Flash Next on a 48 GB Mac (config/qwen3.8-flash-next.json): ISTA-DASLab's GSQ-RCO Q2_0 in two
/// shards with the DFlash2 drafter. Its files are pinned by size and sha256 as read at the publishers' revisions.
/// </summary>
public sealed class Qwen38FlashNextGsqQ20CatalogTests
{
    private const string Id = "qwen3.8-flash-next-gsq-q2-0";

    private static CatalogModel Model => Assert.Single(ModelCatalog.BuiltIn, m => m.Id == Id);

    [Fact]
    public void TheEntryPinsTheTwoShards_AndTheDrafterAtItsRevision()
    {
        CatalogModel model = Model;
        Assert.Equal(CatalogFamily.Qwen38FlashNext, model.Family);
        Assert.Equal(CatalogArchitectureKind.MixtureOfExperts, model.Kind);
        Assert.Equal("GSQ-RCO Q2_0", model.Quantization);

        CatalogFile[] weights = model.WeightFiles.ToArray();
        Assert.Equal(2, weights.Length);
        Assert.Equal(("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", 37_623_740_192L,
                "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
            (weights[0].FileName, weights[0].Bytes, weights[0].Sha256));
        Assert.Equal(("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf", 28_800_138_432L,
                "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
            (weights[1].FileName, weights[1].Bytes, weights[1].Sha256));
        Assert.All(weights, w => Assert.StartsWith("https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/", w.Url));

        CatalogFile draft = Assert.Single(model.Files, f => f.Role == CatalogFileRole.Draft);
        Assert.Equal("Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf", draft.FileName);
        Assert.Equal(362_124_704L, draft.Bytes);
        Assert.Equal("2f174e593998c77df0eb7ca13961915a26f25f7944d784a1838e1856e857ff0d", draft.Sha256);
        Assert.Contains("/resolve/4f5a39be8f0a9e9a573ac9824251830a8f4096d9/", draft.Url);
        Assert.True(draft.Optional);
        Assert.True(Assert.Single(model.Files, f => f.Role == CatalogFileRole.Projector).Optional);
    }

    [Fact]
    public void ItIsADesktopEntry_WithTheEnginesCacheSettings()
    {
        CatalogModel model = Model;
        Assert.Equal(48, model.MinDeviceMemoryGB);
        Assert.Equal("f16", model.KvCacheDtype);
        Assert.True(model.LeanCaches);
        Assert.True(model.Experimental);
        Assert.True(model.SupportsThinking);
        // The n-gram table and the experts outside the GPU slots are paged; the rest is held.
        Assert.InRange(model.WeightsPagedFromDiskBytes, 28_800_138_432L, model.WeightsBytes - 1);
        Assert.Null(model.DraftTuning);
    }
}
