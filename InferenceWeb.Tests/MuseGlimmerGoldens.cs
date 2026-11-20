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
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The legacy Muse-Glimmer model's logits on the synthetic fixtures, frozen per lane in stage S0 of the unified-runtime
/// migration (docs/design/muse-glimmer-unified.md section 4.2), before any stage changes shared code: one file per lane
/// (backend, runtime identifier and device) under Fixtures/MuseGlimmer/LegacyGoldens. A case is a fixture variant and a
/// script of forwards over one cache (a long prefill, prompt steps, the ring wrapping, the full layers' cache growing, a
/// rewind); its outputs are the logits every forward returned, each kept whole (base64 of the float bits) with its
/// SHA-256, its largest magnitude and its L2 norm. The fixtures are regenerated from their options at test time and held
/// to the SHA-256 the outputs were frozen from.
///
/// The legacy model is held to its own frozen bytes on the lane that froze them (<see cref="MuseGlimmerFrozenLegacyTests"/>)
/// until stage S9 deletes it; the runtime family (stage S2 on) is held to them within a bound measured between the lanes.
/// </summary>
internal static class MuseGlimmerGoldens
{
    internal const string ResourcePrefix = "InferenceWeb.Tests.Fixtures.MuseGlimmer.LegacyGoldens.";

    // ---------------------------------------------------------------- fixtures

    /// <summary>A fixture variant's builder. Both have eight blocks (six sliding-window, two full NoPE layers at 3 and 7), a
    /// 64-token window, the 30B's logit scale and softcap with a head large enough to put the top logits on the softcap's
    /// curve (about 0.6-0.8 of the cap), and block output projections small enough (RMS about 6e-3) for the post norms' 1e-8
    /// epsilon to differ visibly from 1e-5.
    /// "tied" reads its head from token_embd.weight, whose rows are then large (the input norm brings them to unit RMS).</summary>
    internal static MuseGlimmerSyntheticModelBuilder Builder(string fixture) => fixture switch
    {
        "untied" => new MuseGlimmerSyntheticModelBuilder
        {
            BlockCount = 8, SlidingWindow = 64, LogitScale = 0.19612f, Softcap = 20f, HeadScale = 3f, BlockOutputScale = 0.004f,
        },
        "tied" => new MuseGlimmerSyntheticModelBuilder
        {
            BlockCount = 8, SlidingWindow = 64, LogitScale = 0.19612f, Softcap = 20f, TiedOutput = true, EmbeddingScale = 3f,
            BlockOutputScale = 0.004f,
        },
        _ => throw new ArgumentOutOfRangeException(nameof(fixture)),
    };

    /// <summary>Writes the fixture into <paramref name="dir"/> and returns its path.</summary>
    internal static string WriteFixture(string fixture, string dir)
        => Builder(fixture).Write(Path.Combine(dir, $"muse-glimmer-{fixture}.gguf"));

    // ---------------------------------------------------------------- cases

    /// <summary>A frozen case: the fixture variant, and the script of forwards over one cache.</summary>
    internal sealed record CaseSpec(string Name, string Fixture, Func<ModelBase, List<float[]>> Script);

    /// <summary>Deterministic token ids over the byte vocabulary (no model's argmax: the scripts are teacher-forced, so a
    /// near-tie cannot change what a later forward reads).</summary>
    internal static int[] Tokens(int count, uint seed = 0x9E3779B9u)
    {
        var ids = new int[count];
        uint state = seed;
        for (int i = 0; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            ids[i] = (int)(state % MuseGlimmerSyntheticModelBuilder.ByteTokens);
        }
        return ids;
    }

    private static float[] Keep(float[] logits) => (float[])logits.Clone();

    /// <summary>Forward <paramref name="tokens"/>[from..to) one token at a time, keeping each forward's logits.</summary>
    private static void Decode(ModelBase model, int[] tokens, int from, int to, List<float[]> outputs)
    {
        for (int p = from; p < to; p++)
            outputs.Add(Keep(model.Forward(new[] { tokens[p] })));
    }

    /// <summary>
    /// The cases. Positions are absolute; the window is 64 and legacy's ring (where a backend arms it: ggml_metal here)
    /// is pad256(64 + 2048 + 1) = 2304 rows, while the full layers start at 2048 rows on ggml_metal and MLX and grow on
    /// demand (ggml_cpu and the managed backend allocate the whole 8192-token context):
    /// <list type="bullet">
    /// <item><c>decode</c>: a 100-token prompt (past the window) in one forward, then 12 decode steps.</item>
    /// <item><c>prompt-steps</c>: a prompt forwarded in three uneven steps that end on both sides of the window, then 4 decode
    /// steps.</item>
    /// <item><c>ring-wrap</c>: a 2500-token prompt in one forward (legacy prefills it in 2048-row chunks; it passes the ring
    /// and the full layers' first grow), then 16 decode steps on the wrapped ring.</item>
    /// <item><c>kv-grow</c>: a 2030-token prompt, then 24 decode steps; the full layers grow at position 2048 on a decode
    /// step.</item>
    /// <item><c>rewind</c>: a 300-token prompt and 4 decode steps, a truncation to 290, the 10 rows forwarded again, then 4
    /// decode steps.</item>
    /// </list>
    /// The tied fixture repeats <c>decode</c> and <c>ring-wrap</c>.
    /// </summary>
    internal static readonly CaseSpec[] Cases =
    {
        new("untied/decode", "untied", DecodeScript),
        new("untied/prompt-steps", "untied", PromptStepsScript),
        new("untied/ring-wrap", "untied", RingWrapScript),
        new("untied/kv-grow", "untied", KvGrowScript),
        new("untied/rewind", "untied", RewindScript),
        new("tied/decode", "tied", DecodeScript),
        new("tied/ring-wrap", "tied", RingWrapScript),
    };

    private static List<float[]> DecodeScript(ModelBase model)
    {
        int[] t = Tokens(112);
        model.ResetKVCache();
        var outputs = new List<float[]> { Keep(model.Forward(t[..100])) };
        Decode(model, t, 100, 112, outputs);
        return outputs;
    }

    private static List<float[]> PromptStepsScript(ModelBase model)
    {
        int[] t = Tokens(255, seed: 0x2545F491u);
        model.ResetKVCache();
        var outputs = new List<float[]>
        {
            Keep(model.Forward(t[..37])),
            Keep(model.Forward(t[37..150])),
            Keep(model.Forward(t[150..251])),
        };
        Decode(model, t, 251, 255, outputs);
        return outputs;
    }

    private static List<float[]> RingWrapScript(ModelBase model)
    {
        int[] t = Tokens(2516, seed: 0x6C8E9CF5u);
        model.ResetKVCache();
        var outputs = new List<float[]> { Keep(model.Forward(t[..2500])) };
        Decode(model, t, 2500, 2516, outputs);
        return outputs;
    }

    private static List<float[]> KvGrowScript(ModelBase model)
    {
        int[] t = Tokens(2054, seed: 0x1B873593u);
        model.ResetKVCache();
        var outputs = new List<float[]> { Keep(model.Forward(t[..2030])) };
        Decode(model, t, 2030, 2054, outputs);
        return outputs;
    }

    private static List<float[]> RewindScript(ModelBase model)
    {
        int[] t = Tokens(310, seed: 0x85EBCA6Bu);
        model.ResetKVCache();
        var outputs = new List<float[]> { Keep(model.Forward(t[..300])) };
        int[] extra = Tokens(8, seed: 0xC2B2AE35u);
        for (int i = 0; i < 4; i++) outputs.Add(Keep(model.Forward(new[] { extra[i] })));
        Assert.True(model.TryTruncateKVCache(290), "the rewind to 290 was refused");
        outputs.Add(Keep(model.Forward(t[290..300])));
        for (int i = 4; i < 8; i++) outputs.Add(Keep(model.Forward(new[] { extra[i] })));
        return outputs;
    }

    /// <summary>The KV cache type every case runs with (flow-scoped, so no other test sees it).</summary>
    internal const KvCacheDtype CaseKvDtype = KvCacheDtype.F16;

    // ---------------------------------------------------------------- lanes

    internal static string BackendName(BackendType backend) => backend switch
    {
        BackendType.Cpu => "cpu",
        BackendType.GgmlCpu => "ggml_cpu",
        BackendType.GgmlMetal => "ggml_metal",
        BackendType.GgmlCuda => "ggml_cuda",
        BackendType.GgmlVulkan => "ggml_vulkan",
        BackendType.Mlx => "mlx",
        BackendType.Cuda => "cuda",
        _ => backend.ToString(),
    };

    /// <summary>The lane a backend runs on here: backend, runtime identifier and device (the GPU's name, or the CPU's model;
    /// MLX's device is the chip, named like the CPU).</summary>
    internal static string LaneKey(BackendType backend)
    {
        string device = backend switch
        {
            BackendType.GgmlMetal => GpuName(GgmlBackendType.Metal),
            BackendType.GgmlCuda => GpuName(GgmlBackendType.Cuda),
            BackendType.GgmlVulkan => GpuName(GgmlBackendType.Vulkan),
            BackendType.Cuda => DirectCudaName(),
            _ => CpuName(),
        };
        return $"{BackendName(backend)}|{RuntimeInformation.RuntimeIdentifier}|{device}";
    }

    private static string GpuName(GgmlBackendType type)
    {
        try { return GgmlBasicOps.GetGpuDeviceDescription(type, 0) ?? "?"; }
        catch (Exception) { return "?"; }
    }

    private static string DirectCudaName()
    {
        try { return TensorSharp.Cuda.CudaDevice.GetDevice(0).Name ?? "?"; }
        catch (Exception) { return "?"; }
    }

    [DllImport("libc", EntryPoint = "sysctlbyname")]
    private static extern int SysctlByName(string name, byte[] oldp, ref IntPtr oldlenp, IntPtr newp, IntPtr newlen);

    private static string CpuName()
    {
        try
        {
            if (OperatingSystem.IsMacOS())
            {
                var buffer = new byte[256];
                IntPtr len = buffer.Length;
                if (SysctlByName("machdep.cpu.brand_string", buffer, ref len, IntPtr.Zero, IntPtr.Zero) == 0)
                    return System.Text.Encoding.ASCII.GetString(buffer, 0, Math.Max(0, (int)len - 1)).Trim();
            }
            else if (OperatingSystem.IsLinux() && File.Exists("/proc/cpuinfo"))
            {
                string line = File.ReadLines("/proc/cpuinfo").FirstOrDefault(l => l.StartsWith("model name", StringComparison.Ordinal)
                    || l.StartsWith("CPU part", StringComparison.Ordinal));
                if (line != null) return line[(line.IndexOf(':') + 1)..].Trim();
            }
        }
        catch (Exception) { }
        return "?";
    }

    /// <summary>The file a lane's frozen cases go to.</summary>
    internal static string FileName(string lane) =>
        new string(lane.Select(ch => char.IsLetterOrDigit(ch) || ch is '-' or '_' ? ch : '.').ToArray()).Trim('.') + ".json";

    // ---------------------------------------------------------------- one output

    /// <summary>One frozen output: its bytes' SHA-256, its largest magnitude, its L2 norm and its values.</summary>
    internal sealed record Frozen(string Sha256, double MaxAbs, double L2, float[] Values)
    {
        internal static Frozen Of(float[] values)
        {
            double maxAbs = 0, l2 = 0;
            foreach (float v in values)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs((double)v));
                l2 += (double)v * v;
            }
            return new Frozen(Sha(values), maxAbs, Math.Sqrt(l2), (float[])values.Clone());
        }

        internal JsonObject ToJson()
        {
            byte[] bytes = new byte[Values.Length * sizeof(float)];
            Buffer.BlockCopy(Values, 0, bytes, 0, bytes.Length);
            return new JsonObject
            {
                ["sha256"] = Sha256,
                ["count"] = Values.Length,
                ["maxAbs"] = MaxAbs,
                ["l2"] = L2,
                ["values"] = Convert.ToBase64String(bytes),
            };
        }

        internal static Frozen FromJson(JsonElement e)
        {
            byte[] bytes = Convert.FromBase64String(e.GetProperty("values").GetString());
            var values = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            Assert.Equal(e.GetProperty("count").GetInt32(), values.Length);
            return new Frozen(e.GetProperty("sha256").GetString(), e.GetProperty("maxAbs").GetDouble(), e.GetProperty("l2").GetDouble(), values);
        }
    }

    internal static string Sha(float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexStringLower(SHA256.HashData(bytes));
    }

    internal static string FileSha(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));

    // ---------------------------------------------------------------- the store

    /// <summary>A frozen case on one lane: the fixture it ran on, the KV cache type, and its outputs.</summary>
    internal sealed record Case(string Lane, string FixtureSha256, string KvDtype, Frozen[] Outputs);

    private static readonly Lazy<Dictionary<string, Dictionary<string, Case>>> s_lanes = new(Load);

    private static Dictionary<string, Dictionary<string, Case>> Load()
    {
        var lanes = new Dictionary<string, Dictionary<string, Case>>(StringComparer.Ordinal);
        var asm = typeof(MuseGlimmerGoldens).Assembly;
        foreach (string name in asm.GetManifestResourceNames().Where(n => n.StartsWith(ResourcePrefix, StringComparison.Ordinal)))
        {
            using Stream s = asm.GetManifestResourceStream(name)!;
            using JsonDocument doc = JsonDocument.Parse(s);
            string lane = doc.RootElement.GetProperty("lane").GetString();
            var cases = new Dictionary<string, Case>(StringComparer.Ordinal);
            foreach (JsonProperty c in doc.RootElement.GetProperty("cases").EnumerateObject())
                cases[c.Name] = new Case(lane, c.Value.GetProperty("fixtureSha256").GetString(), c.Value.GetProperty("kvDtype").GetString(),
                    c.Value.GetProperty("outputs").EnumerateArray().Select(Frozen.FromJson).ToArray());
            lanes[lane] = cases;
        }
        return lanes;
    }

    internal static IReadOnlyCollection<string> Lanes => s_lanes.Value.Keys;

    internal static IReadOnlyDictionary<string, Case> CasesOf(string lane) => s_lanes.Value[lane];

    /// <summary>The frozen case for this lane; where this lane was never frozen, the same backend's on this runtime identifier
    /// first, then the same backend's anywhere, then ggml_cpu's on the Mac that froze them. Null when none exists.</summary>
    internal static Case Find(string caseName, string lane)
    {
        var all = s_lanes.Value;
        if (all.TryGetValue(lane, out var mine) && mine.TryGetValue(caseName, out Case exact)) return exact;
        string[] parts = lane.Split('|');
        int Rank(string key)
        {
            string[] k = key.Split('|');
            if (k[0] == parts[0]) return k.ElementAtOrDefault(1) == parts.ElementAtOrDefault(1) ? 0 : 1;
            return key.StartsWith("ggml_cpu|osx-arm64|", StringComparison.Ordinal) ? 2 : 3;
        }
        foreach (var (key, cases) in all.OrderBy(kv => Rank(kv.Key)).ThenBy(kv => kv.Key, StringComparer.Ordinal))
            if (cases.TryGetValue(caseName, out Case c)) return c;
        return null;
    }

    /// <summary>The relative L2 distance of <paramref name="got"/> from <paramref name="want"/>, and the largest difference
    /// over the largest magnitude of <paramref name="want"/>.</summary>
    internal static (double RelL2, double RelMax) Distance(float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        double err2 = 0, ref2 = 0, maxErr = 0, maxRef = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double d = (double)got[i] - want[i];
            err2 += d * d;
            ref2 += (double)want[i] * want[i];
            maxErr = Math.Max(maxErr, Math.Abs(d));
            maxRef = Math.Max(maxRef, Math.Abs((double)want[i]));
        }
        return (ref2 > 0 ? Math.Sqrt(err2 / ref2) : Math.Sqrt(err2), maxRef > 0 ? maxErr / maxRef : maxErr);
    }

    /// <summary>
    /// Holds <paramref name="got"/> (a case's outputs, in order) to the frozen case: byte for byte when
    /// <paramref name="bytes"/> (the frozen lane is this one, and the producer is the one that froze it), else every output
    /// within <paramref name="bound"/> in relative L2 and in largest difference over the largest frozen magnitude. Returns a
    /// line saying what was checked.
    /// </summary>
    internal static string Check(Case frozen, string caseName, string fixtureSha256, IReadOnlyList<float[]> got, bool bytes, double bound)
    {
        Assert.True(frozen != null, $"no frozen legacy output for {caseName} (lanes: {string.Join("; ", Lanes)})");
        Assert.True(frozen.FixtureSha256 == fixtureSha256,
            $"{caseName}: the synthetic fixture is not the one the legacy outputs were frozen from ({fixtureSha256} vs {frozen.FixtureSha256}); " +
            "a fixture generator change needs new reference outputs");
        Assert.Equal(frozen.Outputs.Length, got.Count);
        double worstL2 = 0, worstMax = 0;
        for (int i = 0; i < got.Count; i++)
        {
            Frozen want = frozen.Outputs[i];
            Assert.All(got[i], v => Assert.True(float.IsFinite(v), $"{caseName} output {i}: a non-finite logit"));
            if (bytes)
            {
                Assert.True(want.Sha256 == Sha(got[i]),
                    $"{caseName} output {i}: not the bytes frozen on {frozen.Lane} (relL2 {Distance(want.Values, got[i]).RelL2:E3})");
                continue;
            }
            var (relL2, relMax) = Distance(want.Values, got[i]);
            worstL2 = Math.Max(worstL2, relL2);
            worstMax = Math.Max(worstMax, relMax);
            Assert.True(relL2 <= bound && relMax <= bound,
                $"{caseName} output {i}: relL2 {relL2:E3}, largest difference {relMax:E3} of the largest frozen value, from the " +
                $"outputs frozen on {frozen.Lane}; bound {bound:E1}");
        }
        return bytes
            ? $"{caseName}: {got.Count} outputs byte for byte the ones frozen on this lane"
            : $"{caseName}: {got.Count} outputs within {bound:E1} of the ones frozen on {frozen.Lane} (worst relL2 {worstL2:E3}, largest difference {worstMax:E3})";
    }

    // ---------------------------------------------------------------- writing (the freeze)

    internal static JsonObject CaseJson(string fixtureSha256, IEnumerable<float[]> outputs) => new()
    {
        ["fixtureSha256"] = fixtureSha256,
        ["kvDtype"] = CaseKvDtype.ToString(),
        ["outputs"] = new JsonArray(outputs.Select(o => (JsonNode)Frozen.Of(o).ToJson()).ToArray()),
    };
}
