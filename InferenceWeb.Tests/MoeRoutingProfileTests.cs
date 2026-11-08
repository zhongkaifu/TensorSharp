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
using System.IO;
using System.Linq;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The routing profile decides which experts a host-offloaded layer keeps in its
/// accelerator tier. These pin its ranking, the budgeted selection across layers,
/// and the persistence contract (a cached profile must match the model exactly;
/// an explicitly named one may come from another quantization of it).
/// </summary>
[Collection("EnvironmentVariables")]
public sealed class MoeRoutingProfileTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-moe-profile-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new();

    public MoeRoutingProfileTests()
    {
        _env.Set("TENSORSHARP_MOE_PROFILE", null);
        _env.Set("TENSORSHARP_MOE_PROFILE_DIR", _dir);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (DirectoryNotFoundException) { }
    }

    private static MoeRoutingProfile Profile(params ulong[][] counts)
    {
        var p = new MoeRoutingProfile("model.gguf|1|test|" + counts.Length, counts.Length, counts[0].Length);
        for (int l = 0; l < counts.Length; l++) p.Add(l, counts[l]);
        return p;
    }

    [Fact]
    public void Rank_OrdersByCount_TiesKeepTheLowerId()
    {
        var p = Profile(new ulong[] { 5, 9, 5, 0, 9 });
        Assert.Equal(new[] { 1, 4, 0, 2, 3 }, p.Rank(0));
    }

    [Fact]
    public void SelectTiers_SpendsTheBudgetWhereItCoversTheMostRoutes()
    {
        // Layer 0 is skewed (one expert takes most routes), layer 1 is flat. With
        // room for three experts the skewed layer's head and the flat layer's two
        // best are worth more than the skewed layer's tail.
        var p = Profile(
            new ulong[] { 70, 10, 10, 10 },
            new ulong[] { 30, 30, 20, 20 });
        int[][] tiers = p.SelectTiers(new long[] { 100, 100 }, budgetBytes: 300);
        Assert.Equal(new[] { 0 }, tiers[0]);
        Assert.Equal(new[] { 0, 1 }, tiers[1]);
    }

    [Fact]
    public void SelectTiers_TakesEachLayersPrefixAndNeverPicksUnroutedExperts()
    {
        var p = Profile(new ulong[] { 0, 3, 0, 1 }, new ulong[] { 2, 2, 2, 2 });
        int[][] tiers = p.SelectTiers(new long[] { 10, 10 }, budgetBytes: 1000);
        Assert.Equal(new[] { 1, 3 }, tiers[0]);            // experts 0 and 2 were never routed
        Assert.Equal(new[] { 0, 1, 2, 3 }, tiers[1]);
        foreach (int[] tier in tiers)
            Assert.Equal(tier.Length, tier.Distinct().Count());
    }

    [Fact]
    public void SelectTiers_RespectsTheBudgetAndSkipsLayersWithoutBytes()
    {
        var p = Profile(new ulong[] { 4, 3, 2, 1 }, new ulong[] { 4, 3, 2, 1 });
        int[][] tiers = p.SelectTiers(new long[] { 30, 0 }, budgetBytes: 70);
        Assert.Equal(new[] { 0, 1 }, tiers[0]);             // 2 x 30 bytes fit in 70, a third does not
        Assert.Empty(tiers[1]);                             // no bytes: not tiered
    }

    [Fact]
    public void CoversLayers_NeedsEnoughRoutesInEveryLayer()
    {
        var p = new MoeRoutingProfile("m|1|t|2", 2, 2);
        p.Add(0, new ulong[] { (ulong)MoeRoutingProfile.MinimumRoutesPerLayer, 0 });
        Assert.False(p.CoversLayers(new[] { 0, 1 }));
        p.Add(1, new ulong[] { 1, (ulong)MoeRoutingProfile.MinimumRoutesPerLayer - 1 });
        Assert.True(p.CoversLayers(new[] { 0, 1 }));
    }

    [Fact]
    public void SaveAndLoad_RoundTripsAndMergesAcrossLoads()
    {
        var p = Profile(new ulong[] { 1, 2, 3 }, new ulong[] { 4, 5, 6 });
        string path = MoeRoutingProfile.PathFor(p.Identity);
        Assert.StartsWith(_dir, path);
        Assert.True(p.TrySave(path));

        var loaded = MoeRoutingProfile.TryLoad(path, p.Identity, 2, 3);
        Assert.NotNull(loaded);
        Assert.Equal(new ulong[] { 4, 5, 6 }, loaded.Counts(1).ToArray());
        loaded.Add(0, new ulong[] { 10, 0, 0 });
        Assert.True(loaded.TrySave(path));
        Assert.Equal(new ulong[] { 11, 2, 3 }, MoeRoutingProfile.TryLoad(path, p.Identity, 2, 3).Counts(0).ToArray());
    }

    [Fact]
    public void Load_RefusesAnotherModelOrShape_UnlessTheFileIsNamedExplicitly()
    {
        var p = Profile(new ulong[] { 1, 2, 3 });
        string path = MoeRoutingProfile.PathFor(p.Identity);
        Assert.True(p.TrySave(path));
        Assert.Null(MoeRoutingProfile.TryLoad(path, "other.gguf|1|test|1", 1, 3));
        Assert.Null(MoeRoutingProfile.TryLoad(path, p.Identity, 1, 4));
        Assert.Null(MoeRoutingProfile.TryLoad(Path.Combine(_dir, "missing.json"), p.Identity, 1, 3));

        // An explicitly named profile may come from another quantization of the
        // same model: same routed shape, different file.
        _env.Set("TENSORSHARP_MOE_PROFILE", path);
        Assert.Equal(path, MoeRoutingProfile.PathFor("anything"));
        Assert.NotNull(MoeRoutingProfile.TryLoad(path, "other.gguf|1|test|1", 1, 3));
        Assert.Null(MoeRoutingProfile.TryLoad(path, "other.gguf|1|test|1", 1, 4));
    }

    [Fact]
    public void Load_ReturnsNullForACorruptFile()
    {
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, "bad.json");
        File.WriteAllText(path, "{ \"version\": 1, \"layers\": ");
        Assert.Null(MoeRoutingProfile.TryLoad(path, "m", 1, 3));
    }

    [Theory]
    [InlineData("2.5", 2_500_000_000L)]
    [InlineData("0", 0L)]
    [InlineData("-1", null)]
    [InlineData("abc", null)]
    [InlineData(null, null)]
    public void BudgetOverrides_ParseGigabytes(string value, long? expected)
    {
        Assert.Equal(expected, Qwen4ExpModel.ExpertTierBudgetOverride(value));
        Assert.Equal(expected, ModelBase.HostExpertIoBudgetOverride(value));
    }

    /// <summary>TS_MOE_WINDOW_SHARE: the share of a unified-memory tier budget given to window slots, 0 to 0.9; anything
    /// else (unset, negative, past 0.9, not a number) is the default.</summary>
    [Theory]
    [InlineData("0.25", 0.25)]
    [InlineData("0", 0.0)]
    [InlineData("0.9", 0.9)]
    [InlineData("0.95", Qwen4ExpModel.DefaultWindowShare)]
    [InlineData("-0.1", Qwen4ExpModel.DefaultWindowShare)]
    [InlineData("abc", Qwen4ExpModel.DefaultWindowShare)]
    [InlineData(null, Qwen4ExpModel.DefaultWindowShare)]
    public void WindowShare_ParsesAFraction(string value, double expected)
        => Assert.Equal(expected, Qwen4ExpModel.WindowShare(value));
}
