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
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The legacy Muse-Glimmer model and vision tower on the real weights, frozen per lane next to the lane's llama.cpp image
/// goldens (Fixtures/MuseGlimmer/LlamaCpp/[set/]legacy_vision.&lt;lane&gt;.json), so the image gates that are relative to legacy
/// outlive stage S9, which deletes it:
/// <list type="bullet">
/// <item>its teacher-forced argmax agreement with the lane's set (<see cref="MuseGlimmerParity.GoldenSet"/>), and the
/// distribution of every position's <see cref="MuseGlimmerParity.Position.TopDelta"/> (mean, p50, p90, p99, max);</item>
/// <item>on the lanes whose products quantize the activations (ggml_cuda, ggml_vulkan), each picture's 1 - cosine between the
/// legacy tower's rows and the F32 tower's (<see cref="F32Tower"/>).</item>
/// </list>
/// <see cref="MuseGlimmerLegacyVisionWriterTests"/> writes them.
/// </summary>
internal static class MuseGlimmerLegacyVision
{
    /// <summary>A distribution's mean and nearest-rank quantiles.</summary>
    internal sealed record Quantiles(double Mean, double P50, double P90, double P99, double Max);

    internal sealed record Lane(string Key, string Set, int Agree, int Positions, double Agreement, Quantiles TopDelta,
        IReadOnlyDictionary<string, double> RowsOneMinusCos);

    internal static string FileName(string lane) => "legacy_vision." + MuseGlimmerGoldens.FileName(lane);

    /// <summary>The frozen figures of <paramref name="lane"/> next to set <paramref name="set"/>'s goldens, or null.</summary>
    internal static Lane Find(string lane, string set)
    {
        string resource = MuseGlimmerParity.ResourcePrefix + (set == "metal" ? "" : set + ".") + FileName(lane);
        using Stream s = typeof(MuseGlimmerLegacyVision).Assembly.GetManifestResourceStream(resource);
        if (s == null) return null;
        using JsonDocument doc = JsonDocument.Parse(s);
        JsonElement root = doc.RootElement;
        Assert.Equal(lane, root.GetProperty("lane").GetString());
        JsonElement prompts = root.GetProperty("imagePrompts");
        JsonElement q = prompts.GetProperty("topDelta");
        var rows = new Dictionary<string, double>(StringComparer.Ordinal);
        if (root.TryGetProperty("rowsVsF32", out JsonElement r) && r.ValueKind == JsonValueKind.Object)
            foreach (JsonElement image in r.GetProperty("images").EnumerateArray())
                rows[image.GetProperty("name").GetString()] = image.GetProperty("oneMinusCos").GetDouble();
        return new Lane(lane, root.GetProperty("set").GetString(), prompts.GetProperty("agree").GetInt32(), prompts.GetProperty("positions").GetInt32(),
            prompts.GetProperty("agreement").GetDouble(),
            new Quantiles(q.GetProperty("mean").GetDouble(), q.GetProperty("p50").GetDouble(), q.GetProperty("p90").GetDouble(),
                q.GetProperty("p99").GetDouble(), q.GetProperty("max").GetDouble()),
            rows);
    }

    /// <summary>The mean and nearest-rank quantiles of <paramref name="values"/>.</summary>
    internal static Quantiles QuantilesOf(IEnumerable<double> values)
    {
        double[] v = values.OrderBy(x => x).ToArray();
        Assert.NotEmpty(v);
        double At(double q) => v[Math.Clamp((int)Math.Ceiling(q * v.Length) - 1, 0, v.Length - 1)];
        return new Quantiles(v.Average(), At(0.50), At(0.90), At(0.99), v[^1]);
    }

    /// <summary>
    /// The F32 tower the quantized lanes are measured against: the graph tower with F32 products
    /// (<see cref="TensorSharp.Graph.GraphProgram.FloatProducts"/>) on direct CUDA where the process can open it (0.1-1.6 s an
    /// image on an RTX PRO 4500 Blackwell), else on the managed executor (8-95 s an image on 28 cores). Both multiply the q8_0
    /// weights' values in F32: they are within relative L2 3e-5 of the per-op tower that dequantized its weights at load, and
    /// within 1 - cosine 1e-9 of each other.
    /// </summary>
    internal static (MuseGlimmerVisionTower Tower, IDisposable Owner, string Name) F32Tower(GgufFile mmproj, bool preferDirectCuda = true)
    {
        if (preferDirectCuda && TestGates.CudaSkip == null)
        {
            var cuda = new CudaAllocator();
            return (new MuseGlimmerVisionTower(mmproj, cuda, BackendType.Cuda), cuda, "the graph tower with F32 products on direct CUDA");
        }
        // The managed tower multiplies with its integer dot unless asked (MuseGlimmerVisionTower).
        var managed = new MuseGlimmerVisionTower(mmproj, new CpuAllocator(BlasEnum.DotNet), BackendType.Cpu);
        managed.Graph.FloatProducts = true;
        return (managed, null, "the graph tower with F32 products on the managed executor");
    }

    /// <summary>Whether a lane's products quantize the activations, so its towers are measured against the F32 tower.</summary>
    internal static bool QuantizedProducts(BackendType backend) => backend is BackendType.GgmlCuda or BackendType.GgmlVulkan;

    /// <summary>1 - cosine of two row sets.</summary>
    internal static double OneMinusCos(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return 1 - dot / Math.Sqrt(na * nb);
    }
}

/// <summary>Writes <see cref="MuseGlimmerLegacyVision"/>'s file for the lane the parity tests run on, into the directory
/// TS_TEST_FREEZE_MUSE_GLIMMER_REAL names (the lane's set folder under Fixtures/MuseGlimmer/LlamaCpp). Needs the weights and
/// the mmproj (TS_TEST_MODEL_DIR).</summary>
[Trait("Category", "Bench")]
public sealed class MuseGlimmerLegacyVisionWriterTests(ITestOutputHelper output)
{
    private const string OutEnv = "TS_TEST_FREEZE_MUSE_GLIMMER_REAL";

    [FreezeFact(OutEnv)]
    public void FreezeTheLegacyVisionFigures()
    {
        string skip = MuseGlimmerVisionParityTests.Skip(needsModel: true, BackendType.GgmlCpu);
        Assert.True(skip == null, skip);
        BackendType backend = MuseGlimmerParity.Backend();
        string lane = MuseGlimmerGoldens.LaneKey(backend), set = MuseGlimmerParity.GoldenSet(backend);
        var goldens = MuseGlimmerVisionParityTests.GoldensOf(set);
        string mmproj = MuseGlimmerVisionParityTests.FindMmproj(), modelPath = MuseGlimmerParity.FindModel();
        string modelSha = File.Exists(modelPath + ".done") ? File.ReadAllText(modelPath + ".done").Split(' ')[0] : MuseGlimmerGoldens.FileSha(modelPath);

        List<MuseGlimmerVisionParityTests.PromptRun> runs;
        using (var model = new MuseGlimmerModel(modelPath, backend))
        {
            ((IVisionCapableModel)model).LoadVisionEncoder(mmproj);
            runs = MuseGlimmerVisionParityTests.RunImagePrompts(model, model.VisionEncoder, goldens, output.WriteLine);
        }
        int agree = runs.Sum(r => r.Positions.Count(p => p.Agrees)), total = runs.Sum(r => r.Positions.Count);
        var q = MuseGlimmerLegacyVision.QuantilesOf(runs.SelectMany(r => r.Positions).Select(p => p.TopDelta));
        output.WriteLine($"legacy on {lane}: {agree}/{total} against the {set} set; top-k |dlogprob| mean {q.Mean:F4} p50 {q.P50:F4} " +
            $"p90 {q.P90:F4} p99 {q.P99:F4} max {q.Max:F4}");

        JsonNode rows = null;
        if (MuseGlimmerLegacyVision.QuantizedProducts(backend))
        {
            using var laneAllocator = new MuseGlimmerVisionTowerTests.Lane(backend);
            using var legacy = new MuseGlimmerVisionEncoder(mmproj, laneAllocator.Allocator);
            using var gguf = new GgufFile(mmproj);
            var (reference, owner, name) = MuseGlimmerLegacyVision.F32Tower(gguf);
            using (owner)
            using (reference)
            {
                var images = new JsonArray();
                foreach (var (image, px, w, h) in MuseGlimmerVisionParityTests.Pictures(legacy.ImageProcessor))
                {
                    double d = MuseGlimmerLegacyVision.OneMinusCos(reference.EncodeToHost(px, w, h, out _), legacy.EncodeToHostArray(px, w, h, out _));
                    output.WriteLine($"  {image}: legacy 1 - cosine to the F32 tower {d:E3}");
                    images.Add(new JsonObject { ["name"] = image, ["width"] = w, ["height"] = h, ["oneMinusCos"] = d });
                }
                rows = new JsonObject { ["reference"] = name, ["images"] = images };
            }
        }

        var file = new JsonObject
        {
            ["lane"] = lane,
            ["set"] = set,
            ["about"] = "The legacy Muse-Glimmer model and vision tower on this lane: its teacher-forced argmax agreement with the " +
                "lane's llama.cpp image goldens (and the distribution of every position's largest |our log-probability - llama.cpp's| " +
                "over llama.cpp's listed top tokens), and on the quantized-product lanes each picture's 1 - cosine between the legacy " +
                "tower's rows and the F32 tower's. InferenceWeb.Tests/MuseGlimmerVisionParityTests.cs holds the runtime family to them.",
            ["producer"] = "MuseGlimmerModel and MuseGlimmerVisionEncoder (legacy)",
            ["provenance"] = Environment.GetEnvironmentVariable("TS_TEST_FREEZE_PROVENANCE") ?? "",
            ["frozenUtc"] = DateTime.UtcNow.ToString("O"),
            ["model"] = new JsonObject { ["file"] = Path.GetFileName(modelPath), ["sha256"] = modelSha },
            ["mmproj"] = new JsonObject { ["file"] = Path.GetFileName(mmproj) },
            ["llamaCpp"] = goldens.LlamaCppCommit,
            ["imagePrompts"] = new JsonObject
            {
                ["agree"] = agree,
                ["positions"] = total,
                ["agreement"] = (double)agree / total,
                ["topDelta"] = new JsonObject { ["mean"] = q.Mean, ["p50"] = q.P50, ["p90"] = q.P90, ["p99"] = q.P99, ["max"] = q.Max },
            },
            ["rowsVsF32"] = rows,
        };
        string path = Path.Combine(Environment.GetEnvironmentVariable(OutEnv), MuseGlimmerLegacyVision.FileName(lane));
        File.WriteAllText(path, file.ToJsonString(new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }) + "\n");
        output.WriteLine($"wrote {path}");
    }
}
