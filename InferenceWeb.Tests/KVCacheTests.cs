// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace InferenceWeb.Tests;

/// <summary>
/// Unit tests for the conversation-level <see cref="KVCache"/> bookkeeping object.
/// These tests intentionally exercise only the pure-data behaviour: prefix matching,
/// truncation, append, and reuse-plan generation. They do not touch any model.
/// </summary>
public class KVCacheTests
{
    [Fact]
    public void NewCache_IsEmpty()
    {
        var cache = new KVCache();

        Assert.True(cache.IsEmpty);
        Assert.Equal(0, cache.Count);
        Assert.Empty(cache.Tokens);
        Assert.Null(cache.NextLogits);
    }

    [Fact]
    public void RecordAppend_SingleToken_StoresTokenAndLogits()
    {
        var cache = new KVCache();
        var logits = new float[] { 0.1f, 0.2f, 0.3f };

        cache.RecordAppend(42, logits);

        Assert.Equal(1, cache.Count);
        Assert.Equal(new[] { 42 }, cache.Tokens);
        Assert.Same(logits, cache.NextLogits);
    }

    [Fact]
    public void RecordAppend_TokenList_StoresAllTokensAndLatestLogits()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });
        cache.RecordAppend(new[] { 4, 5 }, new float[] { 0.7f });

        Assert.Equal(5, cache.Count);
        Assert.Equal(new[] { 1, 2, 3, 4, 5 }, cache.Tokens);
        Assert.Equal(new float[] { 0.7f }, cache.NextLogits);
    }

    [Fact]
    public void RecordAppend_NullTokens_IsNoOp()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2 }, new float[] { 0.0f });

        cache.RecordAppend((IReadOnlyList<int>?)null, new float[] { 1.0f });

        Assert.Equal(2, cache.Count);
    }

    [Fact]
    public void RecordAppend_NullLogits_ClearsCachedLogits()
    {
        var cache = new KVCache();
        cache.RecordAppend(1, new float[] { 0.1f });
        Assert.NotNull(cache.NextLogits);

        cache.RecordAppend(2, null);

        Assert.Equal(2, cache.Count);
        Assert.Null(cache.NextLogits);
    }

    [Fact]
    public void Reset_ClearsTokensAndLogits()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });

        cache.Reset();

        Assert.True(cache.IsEmpty);
        Assert.Null(cache.NextLogits);
    }

    [Fact]
    public void TruncateTo_ShorterLength_DropsTrailingTokensAndLogits()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3, 4, 5 }, new float[] { 0.5f });

        cache.TruncateTo(3);

        Assert.Equal(3, cache.Count);
        Assert.Equal(new[] { 1, 2, 3 }, cache.Tokens);
        Assert.Null(cache.NextLogits);
    }

    [Fact]
    public void TruncateTo_SameLength_KeepsLogits()
    {
        var cache = new KVCache();
        var logits = new float[] { 0.5f };
        cache.RecordAppend(new[] { 1, 2, 3 }, logits);

        cache.TruncateTo(3);

        Assert.Equal(3, cache.Count);
        Assert.Same(logits, cache.NextLogits);
    }

    [Fact]
    public void TruncateTo_Zero_EmptiesCache()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });

        cache.TruncateTo(0);

        Assert.True(cache.IsEmpty);
    }

    [Fact]
    public void TruncateTo_Negative_Throws()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2 }, null);

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.TruncateTo(-1));
    }

    [Fact]
    public void TruncateTo_BeyondCount_Throws()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2 }, null);

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.TruncateTo(3));
    }

    [Fact]
    public void CommonPrefixLength_NoOverlap_ReturnsZero()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, null);

        Assert.Equal(0, cache.CommonPrefixLength(new[] { 4, 5, 6 }));
    }

    [Fact]
    public void CommonPrefixLength_PartialMatch_ReturnsCommonLength()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3, 4, 5 }, null);

        Assert.Equal(3, cache.CommonPrefixLength(new[] { 1, 2, 3, 99, 100 }));
    }

    [Fact]
    public void CommonPrefixLength_NewIsLonger_ReturnsCacheLength()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, null);

        Assert.Equal(3, cache.CommonPrefixLength(new[] { 1, 2, 3, 4, 5 }));
    }

    [Fact]
    public void CommonPrefixLength_NewIsShorter_ReturnsNewLength()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3, 4, 5 }, null);

        Assert.Equal(2, cache.CommonPrefixLength(new[] { 1, 2 }));
    }

    [Fact]
    public void CommonPrefixLength_NullOrEmpty_ReturnsZero()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, null);

        Assert.Equal(0, cache.CommonPrefixLength(null));
        Assert.Equal(0, cache.CommonPrefixLength(Array.Empty<int>()));
    }

    [Fact]
    public void IsPrefixOf_ExactMatch_ReturnsTrue()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, null);

        Assert.True(cache.IsPrefixOf(new[] { 1, 2, 3 }));
    }

    [Fact]
    public void IsPrefixOf_CacheIsPrefix_ReturnsTrue()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2 }, null);

        Assert.True(cache.IsPrefixOf(new[] { 1, 2, 3, 4 }));
    }

    [Fact]
    public void IsPrefixOf_DifferentTokens_ReturnsFalse()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, null);

        Assert.False(cache.IsPrefixOf(new[] { 1, 99, 3 }));
    }

    [Fact]
    public void IsPrefixOf_InputShorter_ReturnsFalse()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3, 4 }, null);

        Assert.False(cache.IsPrefixOf(new[] { 1, 2 }));
    }

    [Fact]
    public void TryGetExactMatchLogits_ExactMatch_ReturnsLogitsCopy()
    {
        var cache = new KVCache();
        var stored = new float[] { 1.0f, 2.0f };
        cache.RecordAppend(new[] { 1, 2, 3 }, stored);

        bool ok = cache.TryGetExactMatchLogits(new[] { 1, 2, 3 }, out float[] logits);

        Assert.True(ok);
        Assert.Equal(stored, logits);
        // Returned array should be an independent copy so caller mutations don't poison the cache.
        logits[0] = 999f;
        Assert.Equal(1.0f, stored[0]);
    }

    [Fact]
    public void TryGetExactMatchLogits_NoLogitsCached_ReturnsFalse()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, null);

        Assert.False(cache.TryGetExactMatchLogits(new[] { 1, 2, 3 }, out _));
    }

    [Fact]
    public void TryGetExactMatchLogits_DifferentLength_ReturnsFalse()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });

        Assert.False(cache.TryGetExactMatchLogits(new[] { 1, 2 }, out _));
        Assert.False(cache.TryGetExactMatchLogits(new[] { 1, 2, 3, 4 }, out _));
    }

    [Fact]
    public void PlanReuse_EmptyCache_ReturnsResetForFullPrompt()
    {
        var cache = new KVCache();

        var plan = cache.PlanReuse(new[] { 1, 2, 3, 4 }, supportsTruncation: true);

        Assert.Equal(ReusePlanKind.Reset, plan.Kind);
        Assert.Equal(4, plan.TokensToForward);
        Assert.Equal(0, plan.ReusedPrefixLength);
        Assert.Null(plan.CachedLogits);
    }

    [Fact]
    public void PlanReuse_EmptyInput_ReturnsResetWithZeroForward()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });

        var plan = cache.PlanReuse(Array.Empty<int>(), supportsTruncation: true);

        Assert.Equal(ReusePlanKind.Reset, plan.Kind);
        Assert.Equal(0, plan.TokensToForward);
    }

    [Fact]
    public void PlanReuse_ExactMatchWithLogits_ReturnsExactMatch()
    {
        var cache = new KVCache();
        var logits = new float[] { 0.5f, 0.3f };
        cache.RecordAppend(new[] { 1, 2, 3 }, logits);

        var plan = cache.PlanReuse(new[] { 1, 2, 3 }, supportsTruncation: true);

        Assert.Equal(ReusePlanKind.ExactMatch, plan.Kind);
        Assert.Equal(0, plan.TokensToForward);
        Assert.Equal(logits, plan.CachedLogits);
    }

    [Fact]
    public void PlanReuse_ExactMatchWithoutLogits_ForwardsLastTokenForFreshLogits()
    {
        var cache = new KVCache();
        // Logits not cached: last RecordAppend passed null
        cache.RecordAppend(new[] { 1, 2, 3 }, null);

        var plan = cache.PlanReuse(new[] { 1, 2, 3 }, supportsTruncation: true);

        Assert.Equal(ReusePlanKind.PartialReuse, plan.Kind);
        Assert.Equal(2, plan.ReusedPrefixLength);
        Assert.Equal(1, plan.TokensToForward);
    }

    [Fact]
    public void PlanReuse_PartialMatch_TruncatableModel_ReturnsPartialReuse()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3, 4, 5 }, new float[] { 0.5f });

        var plan = cache.PlanReuse(new[] { 1, 2, 3, 99, 100 }, supportsTruncation: true);

        Assert.Equal(ReusePlanKind.PartialReuse, plan.Kind);
        Assert.Equal(3, plan.ReusedPrefixLength);
        Assert.Equal(2, plan.TokensToForward);
    }

    [Fact]
    public void PlanReuse_CacheIsPrefixOfInput_ReusesCacheLength()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });

        var plan = cache.PlanReuse(new[] { 1, 2, 3, 4, 5 }, supportsTruncation: true);

        Assert.Equal(ReusePlanKind.PartialReuse, plan.Kind);
        Assert.Equal(3, plan.ReusedPrefixLength);
        Assert.Equal(2, plan.TokensToForward);
    }

    [Fact]
    public void PlanReuse_PartialMatch_NonTruncatableModel_ReturnsReset()
    {
        // For recurrent / SSM models (Qwen3.5, Nemotron-H) we cannot rewind state to an
        // earlier position, so any divergence forces a full reset.
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3, 4, 5 }, new float[] { 0.5f });

        var plan = cache.PlanReuse(new[] { 1, 2, 3, 99, 100 }, supportsTruncation: false);

        Assert.Equal(ReusePlanKind.Reset, plan.Kind);
        Assert.Equal(5, plan.TokensToForward);
    }

    [Fact]
    public void PlanReuse_CacheIsPrefixOfInput_NonTruncatableModel_ReusesCache()
    {
        // Recurrent models can still reuse the cache when the new input is an EXTENSION
        // of the cached prefix (no rewind required).
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });

        var plan = cache.PlanReuse(new[] { 1, 2, 3, 4, 5, 6 }, supportsTruncation: false);

        Assert.Equal(ReusePlanKind.PartialReuse, plan.Kind);
        Assert.Equal(3, plan.ReusedPrefixLength);
        Assert.Equal(3, plan.TokensToForward);
    }

    [Fact]
    public void PlanReuse_NoCommonPrefix_ReturnsReset()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });

        var plan = cache.PlanReuse(new[] { 99, 100 }, supportsTruncation: true);

        Assert.Equal(ReusePlanKind.Reset, plan.Kind);
        Assert.Equal(2, plan.TokensToForward);
    }

    [Fact]
    public void PlanReuse_FirstTokenDiffers_ReturnsReset()
    {
        // Different BOS / system prompt → no usable prefix at all.
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3, 4 }, new float[] { 0.5f });

        var plan = cache.PlanReuse(new[] { 99, 2, 3, 4 }, supportsTruncation: true);

        Assert.Equal(ReusePlanKind.Reset, plan.Kind);
        Assert.Equal(4, plan.TokensToForward);
    }

    [Fact]
    public void PlanReuse_NewInputShorterThanCache_TruncatableTreatsAsPartialReuse()
    {
        var cache = new KVCache();
        cache.RecordAppend(new[] { 1, 2, 3, 4, 5 }, new float[] { 0.5f });

        var plan = cache.PlanReuse(new[] { 1, 2, 3 }, supportsTruncation: true);

        // Input matches the first 3 tokens of cache. Since input == 3 tokens and cache
        // has matching tokens for the entire input, common == input.Count. We back off
        // by one to ensure we forward at least one token to get fresh logits.
        Assert.Equal(ReusePlanKind.PartialReuse, plan.Kind);
        Assert.Equal(2, plan.ReusedPrefixLength);
        Assert.Equal(1, plan.TokensToForward);
    }
}
