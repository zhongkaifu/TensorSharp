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
/// The managed backend multiplies F32 values, so these are the reference the other executors were held to.
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

/// <summary>
/// Writes <see cref="MuseGlimmerLegacySynthetic"/>'s file from the legacy code on the managed backend, into the directory
/// TS_TEST_FREEZE_MUSE_GLIMMER_SYNTHETIC names (Fixtures/MuseGlimmer/LegacySynthetic); TS_TEST_FREEZE_PROVENANCE, when set,
/// is recorded as the build. Each output is computed twice and must repeat bit for bit. Goes with the legacy code in S9.
/// </summary>
[Trait("Category", "Bench")]
public sealed class MuseGlimmerLegacySyntheticWriterTests(ITestOutputHelper output)
{
    private const string OutEnv = "TS_TEST_FREEZE_MUSE_GLIMMER_SYNTHETIC";

    [FreezeFact(OutEnv)]
    public void FreezeTheLegacyTowerRowsAndDrafterFeatures()
    {
        string outDir = Environment.GetEnvironmentVariable(OutEnv);
        string scratch = Path.Combine(Path.GetTempPath(), "ts-mg-freeze-synth-" + Guid.NewGuid().ToString("N"));
        try
        {
            var cpu = new CpuAllocator(BlasEnum.DotNet);
            var fixtures = new JsonObject();
            var towers = new JsonObject();
            foreach (var f in MuseGlimmerLegacySynthetic.TowerFixtures)
            {
                string mmproj = MuseGlimmerLegacySynthetic.WriteTowerFixture(f.Name, scratch);
                fixtures[f.Name] = MuseGlimmerGoldens.FileSha(mmproj);
                foreach (var (w, h, seed) in MuseGlimmerLegacySynthetic.ImagesOf(f.Name))
                {
                    int width = w * MuseGlimmerVisionSynthetic.Patch, height = h * MuseGlimmerVisionSynthetic.Patch;
                    float[] px = MuseGlimmerVisionSynthetic.Pixels(width, height, seed);
                    float[] Rows()
                    {
                        using var encoder = new MuseGlimmerVisionEncoder(mmproj, cpu);
                        return encoder.EncodeToHostArray(px, width, height, out _);
                    }
                    float[] rows = Rows(), again = Rows();
                    Assert.Equal(MuseGlimmerGoldens.Sha(rows), MuseGlimmerGoldens.Sha(again));
                    string key = MuseGlimmerLegacySynthetic.TowerKey(f.Name, w, h, seed);
                    towers[key] = MuseGlimmerGoldens.Frozen.Of(rows).ToJson();
                    output.WriteLine($"{key}: {rows.Length} values, repeatable");
                }
            }

            var (target, draft) = MuseGlimmerRuntimeSpeculationTests.WriteFiles(Path.Combine(scratch, "spec"));
            float[] Features()
            {
                using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
                using ModelBase model = new MuseGlimmerModel(target, BackendType.Cpu, 1, null, draft);
                var spec = (ISpeculativeModel)model;
                int[] prompt = MuseGlimmerRuntimeSpeculationTests.LongPrompt;
                var rows = new float[prompt.Length * spec.SpecFeatureSize];
                spec.SpecForward(prompt, rows, new float[model.Config.VocabSize], allLogitsRows: false);
                return rows;
            }
            float[] features = Features(), repeat = Features();
            Assert.Equal(MuseGlimmerGoldens.Sha(features), MuseGlimmerGoldens.Sha(repeat));
            output.WriteLine($"drafter features: {features.Length} values, repeatable");

            var file = new JsonObject
            {
                ["lane"] = MuseGlimmerGoldens.LaneKey(BackendType.Cpu),
                ["about"] = "The legacy Muse-Glimmer per-op vision tower's rows and the legacy model's DFlash drafter features on the synthetic " +
                    "fixtures, frozen on the managed backend before stage S9 deleted the legacy code (InferenceWeb.Tests/MuseGlimmerLegacySynthetic.cs reads them).",
                ["producer"] = "MuseGlimmerVisionEncoder and MuseGlimmerModel (legacy)",
                ["provenance"] = Environment.GetEnvironmentVariable("TS_TEST_FREEZE_PROVENANCE") ?? "",
                ["frozenUtc"] = DateTime.UtcNow.ToString("O"),
                ["towerFixtures"] = fixtures,
                ["towers"] = towers,
                ["drafter"] = new JsonObject
                {
                    ["targetSha256"] = MuseGlimmerGoldens.FileSha(target),
                    ["draftSha256"] = MuseGlimmerGoldens.FileSha(draft),
                    ["kvDtype"] = KvCacheDtype.F32.ToString(),
                    ["features"] = MuseGlimmerGoldens.Frozen.Of(features).ToJson(),
                },
            };
            Directory.CreateDirectory(outDir);
            string path = Path.Combine(outDir, "managed.json");
            File.WriteAllText(path, file.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
            output.WriteLine($"wrote {path}");
        }
        finally
        {
            try { Directory.Delete(scratch, recursive: true); } catch (IOException) { }
        }
    }
}
