// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The legacy Muse-Glimmer code's outputs on the synthetic fixtures that no other frozen file holds, frozen on the managed
/// backend before stage S9 of docs/design/muse-glimmer-unified.md deletes that code
/// (Fixtures/MuseGlimmer/LegacySynthetic/managed.json):
/// <list type="bullet">
/// <item>the legacy per-op vision tower's rows for every grid of <see cref="MuseGlimmerVisionTowerTests.Grids"/>, over the
/// F32-product and the Q8_0-product fixtures (<see cref="MuseGlimmerVisionSynthetic"/>), and for Muse Spark's width;</item>
/// <item>the legacy model's DFlash drafter features (the residual entering each target layer) over the speculation tests'
/// prompt (<see cref="MuseGlimmerRuntimeSpeculationTests.LongPrompt"/>), F32 keys and values.</item>
/// </list>
/// The managed backend multiplies F32 values, so these are the reference the other executors were held to. The writer
/// (MuseGlimmerLegacySyntheticWriterTests, d492bfe9) went with the legacy code.
/// </summary>
internal static class MuseGlimmerLegacySynthetic
{
    internal const string ResourceName = "InferenceWeb.Tests.Fixtures.MuseGlimmer.LegacySynthetic.managed.json";

    /// <summary>The towers' fixtures: name, projection width and product type.</summary>
    internal static readonly (string Name, int Projection, SyntheticGguf.GgmlType Products)[] TowerFixtures =
    {
        ("f32", 48, SyntheticGguf.GgmlType.F32),
        ("q8", 48, SyntheticGguf.GgmlType.Q8_0),
        ("spark-f32", MuseSparkSyntheticModelBuilder.Hidden, SyntheticGguf.GgmlType.F32),
    };

    /// <summary>The images frozen on each fixture: the tower tests' grids (seeds 1, 2, ... in order) on the Glimmer fixtures,
    /// and Spark's 6 x 10 at seed 5.</summary>
    internal static IEnumerable<(int W, int H, uint Seed)> ImagesOf(string fixture) => fixture == "spark-f32"
        ? new[] { (6, 10, 5u) }
        : MuseGlimmerVisionTowerTests.Grids.Select((g, i) => (g.W, g.H, (uint)(i + 1)));

    internal static string TowerKey(string fixture, int w, int h, uint seed) => $"{fixture}/{w}x{h}/{seed}";

    internal static string WriteTowerFixture(string fixture, string dir)
    {
        var f = TowerFixtures.Single(t => t.Name == fixture);
        Directory.CreateDirectory(dir);
        return MuseGlimmerVisionSynthetic.Write(Path.Combine(dir, $"mmproj-{fixture}.gguf"), projection: f.Projection, products: f.Products);
    }

    internal sealed record Store(IReadOnlyDictionary<string, string> FixtureSha256, IReadOnlyDictionary<string, float[]> Towers,
        float[] DrafterFeatures, string TargetSha256, string DraftSha256);

    private static readonly Lazy<Store> s_store = new(() =>
    {
        using Stream s = typeof(MuseGlimmerLegacySynthetic).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new FileNotFoundException("no frozen legacy synthetic outputs embedded", ResourceName);
        using JsonDocument doc = JsonDocument.Parse(s);
        JsonElement root = doc.RootElement;
        var shas = root.GetProperty("towerFixtures").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString(), StringComparer.Ordinal);
        var towers = root.GetProperty("towers").EnumerateObject()
            .ToDictionary(p => p.Name, p => MuseGlimmerGoldens.Frozen.FromJson(p.Value).Values, StringComparer.Ordinal);
        JsonElement drafter = root.GetProperty("drafter");
        return new Store(shas, towers, MuseGlimmerGoldens.Frozen.FromJson(drafter.GetProperty("features")).Values,
            drafter.GetProperty("targetSha256").GetString(), drafter.GetProperty("draftSha256").GetString());
    });

    internal static Store Frozen => s_store.Value;

    /// <summary>The legacy tower's frozen rows for one image of <paramref name="fixture"/>, after checking that the fixture the
    /// generator writes today is the one they were frozen from.</summary>
    internal static float[] TowerRows(string fixture, string mmprojPath, int w, int h, uint seed)
    {
        Assert.Equal(Frozen.FixtureSha256[fixture], MuseGlimmerGoldens.FileSha(mmprojPath));
        Assert.True(Frozen.Towers.TryGetValue(TowerKey(fixture, w, h, seed), out float[] rows), $"no frozen legacy rows for {TowerKey(fixture, w, h, seed)}");
        return rows;
    }
}
