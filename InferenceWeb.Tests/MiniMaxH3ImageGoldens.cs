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
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The legacy MiniMax-H3-Image oracle's numbers, frozen before stage S10 deleted it (docs/design/diffusion-runtime.md section
/// 1.6): the steps of the legacy denoiser (MiniMaxH3DiT.ForwardImage, TSGgml_MiniMaxH3DitForward's graph) and the pixels of
/// the legacy decoder (MiniMaxH3ImageVae, TSGgml_MiniMaxH3VitDecodeBatch) on the synthetic fixtures, one file per lane
/// (backend, runtime identifier and device: Fixtures/MiniMaxH3Image/LegacyGoldens/*.json). Each output keeps its SHA-256, its
/// size, largest magnitude and L2 norm and a strided sample of its values (bit patterns), and each case records whether the
/// runtime gave the legacy bytes on that lane when it was frozen. On a lane where it did, the runtime is held to the bytes;
/// elsewhere (another machine, or a lane where the runtime takes products legacy did not: ggml_cpu on ARM64 after main's
/// repacked and BF16 products) to the sample within a bound. The fixtures are regenerated from their seeds at test time and
/// held to the SHA-256 the frozen outputs were made from.
/// </summary>
internal static class MiniMaxH3ImageGoldens
{
    internal const string ResourcePrefix = "InferenceWeb.Tests.Fixtures.MiniMaxH3Image.LegacyGoldens.";
    internal const int SampleCount = 128;

    // ---------------------------------------------------------------- lanes

    /// <summary>The lane a backend runs on here: backend, runtime identifier and device (the GPU's name, or the CPU's model).</summary>
    internal static string LaneKey(BackendType backend)
    {
        string device = backend switch
        {
            BackendType.GgmlMetal => GpuName(GgmlBackendType.Metal),
            BackendType.GgmlCuda => GpuName(GgmlBackendType.Cuda),
            _ => CpuName(),
        };
        string name = backend switch
        {
            BackendType.GgmlMetal => "ggml_metal",
            BackendType.GgmlCuda => "ggml_cuda",
            BackendType.GgmlCpu => "ggml_cpu",
            _ => backend.ToString(),
        };
        return $"{name}|{RuntimeInformation.RuntimeIdentifier}|{device}";
    }

    /// <summary>The lane of host arithmetic (C#): the runtime identifier.</summary>
    internal static string HostLaneKey() => "host|" + RuntimeInformation.RuntimeIdentifier;

    private static string GpuName(GgmlBackendType type)
    {
        try { return GgmlBasicOps.GetGpuDeviceDescription(type, 0) ?? "?"; }
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

    // ---------------------------------------------------------------- one output

    /// <summary>One frozen output: its bytes' SHA-256, size, largest magnitude, L2 norm and a strided sample.</summary>
    internal sealed record Frozen(string Sha256, int Count, double MaxAbs, double L2, int Stride, float[] Sample)
    {
        internal static Frozen Of(float[] values)
        {
            int stride = Math.Max(1, values.Length / SampleCount);
            var sample = new List<float>();
            for (int i = 0; i < values.Length && sample.Count < SampleCount; i += stride) sample.Add(values[i]);
            double maxAbs = 0, l2 = 0;
            foreach (float v in values)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs((double)v));
                l2 += (double)v * v;
            }
            return new Frozen(Sha(values), values.Length, maxAbs, Math.Sqrt(l2), stride, sample.ToArray());
        }

        internal JsonObject ToJson() => new()
        {
            ["sha256"] = Sha256,
            ["count"] = Count,
            ["maxAbs"] = MaxAbs,
            ["l2"] = L2,
            ["stride"] = Stride,
            ["sample"] = new JsonArray(Sample.Select(v => (JsonNode)BitConverter.SingleToInt32Bits(v).ToString("x8")).ToArray()),
        };

        internal static Frozen FromJson(JsonElement e) => new(
            e.GetProperty("sha256").GetString(), e.GetProperty("count").GetInt32(), e.GetProperty("maxAbs").GetDouble(),
            e.GetProperty("l2").GetDouble(), e.GetProperty("stride").GetInt32(),
            e.GetProperty("sample").EnumerateArray().Select(s => BitConverter.Int32BitsToSingle(Convert.ToInt32(s.GetString(), 16))).ToArray());
    }

    internal static string Sha(float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexStringLower(SHA256.HashData(bytes));
    }

    internal static string FileSha(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));

    // ---------------------------------------------------------------- the store

    /// <summary>A frozen case on one lane: the fixture it ran on, its outputs, and whether the runtime gave their bytes there.</summary>
    internal sealed record Case(string Lane, string FixtureSha256, bool RuntimeBitExact, Frozen[] Outputs);

    private static readonly Lazy<Dictionary<string, Dictionary<string, Case>>> s_lanes = new(Load);

    private static Dictionary<string, Dictionary<string, Case>> Load()
    {
        var lanes = new Dictionary<string, Dictionary<string, Case>>(StringComparer.Ordinal);
        var asm = typeof(MiniMaxH3ImageGoldens).Assembly;
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

    /// <summary>The frozen case for this lane; when this lane was never frozen, one frozen from the same fixture bytes (the
    /// synthetic Q4_K fixture quantizes to other bytes on x64 than on ARM64): the same backend's, on this runtime identifier
    /// first, else the reference lane's (ggml_cpu on the Mac that froze them), else any lane's. Null when none exists.</summary>
    internal static Case Find(string caseName, string lane, string fixtureSha256 = null)
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
            if (cases.TryGetValue(caseName, out Case c) && (fixtureSha256 == null || c.FixtureSha256 == fixtureSha256)) return c;
        return null;
    }

    /// <summary>
    /// Holds <paramref name="got"/> (the runtime's outputs of the case, in order) to the frozen legacy outputs: byte for byte
    /// where they were frozen on this lane and the runtime gave their bytes then, else within <paramref name="bound"/> (the
    /// largest sampled difference over the output's largest magnitude, and the sample's relative L2). Returns what was checked.
    /// </summary>
    internal static string Check(string caseName, string lane, string fixtureSha256, IReadOnlyList<float[]> got, double bound,
        Action<string> log) => CheckAgainst(Find(caseName, lane, fixtureSha256) ?? Find(caseName, lane), caseName, lane, fixtureSha256, got, bound, log);

    /// <summary><see cref="Check"/> against a given frozen case (bytes only when it was frozen on <paramref name="lane"/>).</summary>
    internal static string CheckAgainst(Case frozen, string caseName, string lane, string fixtureSha256, IReadOnlyList<float[]> got,
        double bound, Action<string> log)
    {
        Assert.True(frozen != null, $"no frozen legacy output for {caseName} (lanes: {string.Join("; ", Lanes)})");
        Assert.True(frozen.FixtureSha256 == fixtureSha256,
            $"{caseName}: the synthetic fixture is not the one the legacy outputs were frozen from ({fixtureSha256} vs {frozen.FixtureSha256}); " +
            "the legacy oracle is deleted, so a fixture generator change needs new reference outputs from another oracle");
        Assert.Equal(frozen.Outputs.Length, got.Count);
        bool bytes = frozen.Lane == lane && frozen.RuntimeBitExact;
        double worst = 0, worstL2 = 0;
        for (int i = 0; i < got.Count; i++)
        {
            Frozen want = frozen.Outputs[i];
            float[] g = got[i];
            Assert.Equal(want.Count, g.Length);
            Assert.All(g, v => Assert.True(float.IsFinite(v)));
            if (bytes)
            {
                Assert.True(want.Sha256 == Sha(g), $"{caseName} output {i + 1}: the runtime's bytes are not the legacy ones frozen on {lane}");
                continue;
            }
            double maxErr = 0, err2 = 0, ref2 = 0;
            for (int k = 0; k < want.Sample.Length; k++)
            {
                double w = want.Sample[k], v = g[(long)k * want.Stride];
                maxErr = Math.Max(maxErr, Math.Abs(v - w));
                err2 += (v - w) * (v - w);
                ref2 += w * w;
            }
            double rel = want.MaxAbs > 0 ? maxErr / want.MaxAbs : maxErr, relL2 = ref2 > 0 ? Math.Sqrt(err2 / ref2) : Math.Sqrt(err2);
            worst = Math.Max(worst, rel);
            worstL2 = Math.Max(worstL2, relL2);
            Assert.True(rel <= bound && relL2 <= bound,
                $"{caseName} output {i + 1}: {rel:E3} of the largest legacy value (sample L2 {relL2:E3}) from the legacy output frozen on {frozen.Lane}, bound {bound:E1}");
        }
        string what = bytes ? $"{caseName}: {got.Count} outputs byte for byte the legacy ones frozen on this lane"
            : $"{caseName}: {got.Count} outputs within {bound:E1} of the legacy ones frozen on {frozen.Lane} (worst {worst:E3}, sample L2 {worstL2:E3})";
        log?.Invoke(what);
        return what;
    }

    // ---------------------------------------------------------------- the denoiser cases

    /// <summary>A request's packed sequence: the target latent, the text rows (their modality tags, null all text) and the
    /// references' latents.</summary>
    internal sealed record Request(int LatentHeight, int LatentWidth, int TextRows, int[] Modalities = null, (int H, int W)[] References = null)
    {
        /// <summary>Text to image at 272 x 272 (289 image tokens, 310 rows: past 256, so V's scale is 2).</summary>
        public static readonly Request TextToImage = new(34, 34, 21);

        /// <summary>Text with an image's tags, two references (a row at t = 0.999 of its own) and an odd-shaped target.</summary>
        public static readonly Request WithReferences = new(16, 20, 12, new[] { 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1 }, new[] { (8, 8), (12, 8) });

        /// <summary>Over a thousand rows, where ggml_cuda's wide routes take the products (cuBLAS BF16, h3_q4k_gemm_takes).</summary>
        public static readonly Request Wide = new(64, 64, 21);

        /// <summary>Wide, with text tags alternating row by row and a reference: more modulation segments than the fused
        /// kernels take (tsg_h3_fused_max_segments, 8).</summary>
        public static readonly Request ManySegments = new(64, 64, 12, Enumerable.Range(0, 12).Select(i => i % 2).ToArray(), new[] { (8, 8) });
    }

    internal static string DenoiserCase(string request, int format) => $"denoiser/{request}/{MiniMaxH3ImageFixture.Name(format)}";

    internal static Request Named(string request) => request switch
    {
        "TextToImage" => Request.TextToImage,
        "WithReferences" => Request.WithReferences,
        "Wide" => Request.Wide,
        "ManySegments" => Request.ManySegments,
        _ => throw new ArgumentException(request),
    };

    private static float[] Random(Random rng, long n, float scale = 1f)
    {
        var v = new float[n];
        for (long i = 0; i < n; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    /// <summary>Six Euler steps from the same noise: each step's velocity from <paramref name="advance"/> on the state, which
    /// moves with that velocity; <paramref name="also"/> (when given) gets the same inputs at each step, its velocities
    /// returned second.</summary>
    internal static (List<float[]> Steps, List<float[]> Also) Run(MiniMaxH3ImageFamily family, IDenoiser advance, Request rq,
        IDenoiser also = null, int seed = 7, int steps = 6)
    {
        var rng = new Random(seed);
        var text = new ConditioningSet(Random(rng, (long)rq.TextRows * MiniMaxH3ImageFixture.TextDim, 2f), rq.TextRows,
            MiniMaxH3ImageFixture.TextDim, rq.Modalities);
        var references = (rq.References ?? Array.Empty<(int, int)>()).Select((r, i) =>
        {
            var shape = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, r.H, r.W);
            return new LatentRef(i, shape, family.ToRows(new LatentTensor(shape, Random(rng, shape.Count))));
        }).ToArray();
        var target = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, rq.LatentHeight, rq.LatentWidth);
        float[] x = family.ToRows(new LatentTensor(target, family.Noise(seed).Next(target.Count)));
        float[] sigmas = family.DefaultRecipe.SigmasFor(steps);
        var outA = new List<float[]>();
        var outB = new List<float[]>();
        for (int i = 0; i < steps; i++)
        {
            float sigma = sigmas[i], next = sigmas[i + 1];
            var (start, stop) = family.DefaultRecipe.HeadsFor(i, sigma, next);
            var state = new StepState(i, steps, sigma, next, start, stop);
            var inputs = new DenoiseStepInputs
            {
                Latent = x,
                Conditioning = text,
                References = references,
                Layout = family.Pack(text, references, target, state),
                Step = state,
                Recipe = family.DefaultRecipe,
            };
            float[] v = (float[])advance.Step(inputs, AdapterSet.Empty, System.Threading.CancellationToken.None).Velocity.Clone();
            outA.Add(v);
            if (also != null) outB.Add((float[])also.Step(inputs, AdapterSet.Empty, System.Threading.CancellationToken.None).Velocity.Clone());
            for (long k = 0; k < x.Length; k++) x[k] += v[k] * (next - sigma);
        }
        return (outA, outB);
    }

    /// <summary>The packed request the host tables (the AdaLN curve and the RoPE tables) are frozen for: WithReferences at its
    /// third step.</summary>
    internal static TensorSharp.Models.MiniMaxH3.MiniMaxH3PackedLayout HostTablesLayout(MiniMaxH3ImageFamily family)
    {
        Request rq = Request.WithReferences;
        var text = new ConditioningSet(new float[rq.TextRows * MiniMaxH3ImageFixture.TextDim], rq.TextRows, MiniMaxH3ImageFixture.TextDim, rq.Modalities);
        var references = rq.References.Select((r, i) =>
        {
            var shape = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, r.H, r.W);
            return new LatentRef(i, shape, new float[shape.Count]);
        }).ToArray();
        var target = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, rq.LatentHeight, rq.LatentWidth);
        DiffusionLayout layout = family.Pack(text, references, target, new StepState(2, 6, 0.75f, 0.53364f, 2, 3));
        return (TensorSharp.Models.MiniMaxH3.MiniMaxH3PackedLayout)layout.FamilyLayout;
    }

    // ---------------------------------------------------------------- the decoder cases

    /// <summary>A normalized latent [H, W, 24] of uniform values in [-1, 1) from <paramref name="seed"/>.</summary>
    internal static float[] NormalizedLatent(int h, int w, int seed)
    {
        var rng = new Random(seed);
        var v = new float[(long)h * w * MiniMaxH3ImageVaeCodec.LatentChannels];
        for (long i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    /// <summary>A decoder case: the fixture's blocks, the latent's size and seed, and the tiles a call (0 a row) both decoders
    /// took (on ggml_cuda cuDNN's batched attention plans by batch, so the pixels depend on it there).</summary>
    internal sealed record VaeCase(string Name, int Blocks, int LatentH, int LatentW, int Seed, int Batch, int CudaBatch);

    internal static readonly VaeCase[] VaeCases =
    {
        new("vae/eleven-tiles", 8, 16, 128, 11, Batch: 4, CudaBatch: 0),
        new("vae/canvas-32x40", 2, 32, 40, 11, Batch: 0, CudaBatch: 0),
        new("vae/canvas-16x12", 2, 16, 12, 11, Batch: 0, CudaBatch: 0),
        new("vae/canvas-20x36", 2, 20, 36, 11, Batch: 0, CudaBatch: 0),
        new("vae/split-4", 2, 16, 128, 5, Batch: 4, CudaBatch: 4),
    };

    internal static int BatchOn(VaeCase c, BackendType backend) => backend == BackendType.GgmlCuda ? c.CudaBatch : c.Batch;

    // ---------------------------------------------------------------- writing (the freeze)

    /// <summary>The file a lane's frozen cases go to.</summary>
    internal static string FileName(string lane) =>
        new string(lane.Select(ch => char.IsLetterOrDigit(ch) || ch is '-' or '_' ? ch : '.').ToArray()).Trim('.') + ".json";

    internal static JsonObject CaseJson(string fixtureSha256, bool runtimeBitExact, double runtimeError, IEnumerable<float[]> outputs) => new()
    {
        ["fixtureSha256"] = fixtureSha256,
        ["runtimeBitExact"] = runtimeBitExact,
        ["runtimeErrorAtFreeze"] = runtimeError,
        ["outputs"] = new JsonArray(outputs.Select(o => (JsonNode)Frozen.Of(o).ToJson()).ToArray()),
    };

    /// <summary>The largest difference between two output lists, over the first's largest magnitude.</summary>
    internal static double RelativeError(IReadOnlyList<float[]> want, IReadOnlyList<float[]> got)
    {
        double worst = 0;
        for (int i = 0; i < want.Count; i++)
        {
            double scale = want[i].Max(v => Math.Abs((double)v)), err = 0;
            for (int k = 0; k < want[i].Length; k++) err = Math.Max(err, Math.Abs((double)want[i][k] - got[i][k]));
            worst = Math.Max(worst, scale > 0 ? err / scale : err);
        }
        return worst;
    }
}
