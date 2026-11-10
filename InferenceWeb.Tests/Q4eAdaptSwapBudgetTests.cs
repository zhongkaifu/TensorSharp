// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Adaptive residency's swap budget a round on the direct CUDA engine (Q4eCudaEngine.AdaptSwapBudget): 256 when the
/// swaps overlap the decode, 8 when they block it, and TS_MOE_TIER_ADAPT_SWAPS, when set, for every round.
/// </summary>
public sealed class Q4eAdaptSwapBudgetTests
{
    [Theory]
    [InlineData(null, true, 256)]
    [InlineData(null, false, 8)]
    [InlineData("", true, 256)]
    [InlineData("64", true, 64)]
    [InlineData("64", false, 64)]
    [InlineData("0", true, 1)]
    [InlineData("not a number", true, 256)]
    public void TheBudget_IsTheSettingOrTheDefaultForTheSwapKind(string setting, bool overlapped, int want)
        => Assert.Equal(want, Q4eCudaEngine.AdaptSwapBudget(setting, overlapped));
}
