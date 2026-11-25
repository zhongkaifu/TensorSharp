// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using Frozen = InferenceWeb.Tests.MiniMaxH3ImageGoldens.Frozen;

namespace InferenceWeb.Tests;

/// <summary>
/// The retired Qwen-Image-2.1 transformer's velocities on the synthetic fixtures (QwenImage21DenoiserTests' tiny transformer
/// and LoRA), frozen before the legacy engines were deleted: the native graph (TSGgml_QwenImage21Forward) per GGML lane and
/// the pure-C# transformer (QwenImage21ManagedDiT) on the host lane, one file a lane (Fixtures/QwenImage21/LegacyGoldens/*.json,
/// MiniMaxH3ImageGoldens' format). The runtime is held to the bytes on a lane where it gave them when they were frozen, and to
/// the sample within a bound elsewhere.
/// </summary>
internal static class QwenImage21Goldens
{
    internal const string ResourcePrefix = "InferenceWeb.Tests.Fixtures.QwenImage21.LegacyGoldens.";

    /// <summary>Set to a directory, the freeze test writes this lane's frozen legacy outputs there.</summary>
    internal const string FreezeVariable = "TS_QWEN21_FREEZE_DIR";

    internal sealed record Case(string Lane, string FixtureSha256, bool RuntimeBitExact, Frozen[] Outputs);

    private static readonly Lazy<Dictionary<string, Dictionary<string, Case>>> s_lanes = new(Load);

    private static Dictionary<string, Dictionary<string, Case>> Load()
    {
        var lanes = new Dictionary<string, Dictionary<string, Case>>(StringComparer.Ordinal);
        var asm = typeof(QwenImage21Goldens).Assembly;
        foreach (string name in asm.GetManifestResourceNames().Where(n => n.StartsWith(ResourcePrefix, StringComparison.Ordinal)))
        {
            using Stream s = asm.GetManifestResourceStream(name)!;
            using JsonDocument doc = JsonDocument.Parse(s);
            string lane = doc.RootElement.GetProperty("lane").GetString();
            var cases = new Dictionary<string, Case>(StringComparer.Ordinal);
            foreach (JsonProperty c in doc.RootElement.GetProperty("cases").EnumerateObject())
                cases[c.Name] = new Case(lane, c.Value.GetProperty("fixtureSha256").GetString(), c.Value.GetProperty("runtimeBitExact").GetBoolean(),
                    c.Value.GetProperty("outputs").EnumerateArray().Select(Frozen.FromJson).ToArray());
            lanes[lane] = cases;
        }
        return lanes;
    }

    internal static IReadOnlyCollection<string> Lanes => s_lanes.Value.Keys;

    /// <summary>This lane's frozen case, else the same backend's on any machine (another CPU, another GPU), else any lane's.</summary>
    internal static Case Find(string caseName, string lane)
    {
        var all = s_lanes.Value;
        if (all.TryGetValue(lane, out var mine) && mine.TryGetValue(caseName, out Case exact)) return exact;
        string backend = lane.Split('|')[0];
        foreach (var (key, cases) in all.OrderBy(kv => kv.Key.Split('|')[0] == backend ? 0 : 1).ThenBy(kv => kv.Key, StringComparer.Ordinal))
            if (cases.TryGetValue(caseName, out Case c)) return c;
        return null;
    }

    /// <summary>The runtime's outputs of a case held to the frozen legacy ones: bytes on the lane that froze them where the
    /// runtime gave the bytes then, else the sampled values within <paramref name="bound"/>.</summary>
    internal static string Check(string caseName, string lane, string fixtureSha256, IReadOnlyList<float[]> got, double bound)
    {
        Case frozen = Find(caseName, lane);
        Assert.True(frozen != null, $"no frozen legacy output for {caseName} (lanes: {string.Join("; ", Lanes)})");
        Assert.True(frozen.FixtureSha256 == fixtureSha256,
            $"{caseName}: the synthetic fixture is not the one the legacy outputs were frozen from; the legacy engines are deleted, " +
            "so a change of the fixture generator needs reference outputs from another oracle");
        Assert.Equal(frozen.Outputs.Length, got.Count);
        bool bytes = frozen.Lane == lane && frozen.RuntimeBitExact;
        double worst = 0;
        for (int i = 0; i < got.Count; i++)
        {
            Frozen want = frozen.Outputs[i];
            float[] g = got[i];
            Assert.Equal(want.Count, g.Length);
            if (bytes)
            {
                Assert.True(want.Sha256 == MiniMaxH3ImageGoldens.Sha(g), $"{caseName} output {i + 1}: not the legacy bytes frozen on {lane}");
                continue;
            }
            double err2 = 0, ref2 = 0;
            for (int k = 0; k < want.Sample.Length; k++)
            {
                double w = want.Sample[k], v = g[(long)k * want.Stride];
                err2 += (v - w) * (v - w);
                ref2 += w * w;
            }
            double rel = ref2 > 0 ? Math.Sqrt(err2 / ref2) : Math.Sqrt(err2);
            worst = Math.Max(worst, rel);
            Assert.True(rel <= bound, $"{caseName} output {i + 1}: sample relative L2 {rel:E3} from the legacy output frozen on {frozen.Lane}, bound {bound:E1}");
        }
        return bytes ? $"{caseName}: byte for byte the legacy outputs frozen on this lane"
            : $"{caseName}: within {bound:E1} of the legacy outputs frozen on {frozen.Lane} (worst {worst:E3})";
    }

    /// <summary>A lane's frozen cases as the JSON file the store reads.</summary>
    internal static JsonObject File(string lane, IReadOnlyDictionary<string, (string FixtureSha256, bool RuntimeBitExact, IReadOnlyList<float[]> Outputs)> cases)
    {
        var json = new JsonObject();
        foreach (var (name, c) in cases.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            json[name] = new JsonObject
            {
                ["fixtureSha256"] = c.FixtureSha256,
                ["runtimeBitExact"] = c.RuntimeBitExact,
                ["outputs"] = new JsonArray(c.Outputs.Select(o => (JsonNode)Frozen.Of(o).ToJson()).ToArray()),
            };
        return new JsonObject
        {
            ["lane"] = lane,
            ["about"] = "Qwen-Image-2.1's retired transformer on QwenImage21DenoiserTests' synthetic fixtures (QwenImage21Goldens)",
            ["frozenUtc"] = DateTime.UtcNow.ToString("O"),
            ["cases"] = json,
        };
    }

    /// <summary>The SHA-256 of the files a case's fixture is (the transformer, a LoRA).</summary>
    internal static string FixtureSha(params string[] paths)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (string p in paths) hash.AppendData(System.IO.File.ReadAllBytes(p));
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    /// <summary>The steps of a case: (sigma, step index), and how the state moves with a step's velocity (deterministic, so the
    /// frozen velocities drive every engine's inputs).</summary>
    internal static readonly (float Sigma, int Step)[] Steps = { (1f, 0), (0.75f, 1), (0.3f, 2) };

    internal static void Advance(float[] x, float[] velocity, float sigma)
    {
        for (int i = 0; i < x.Length; i++) x[i] += (sigma - 0.5f) * velocity[i] * 0.1f;
    }
}
