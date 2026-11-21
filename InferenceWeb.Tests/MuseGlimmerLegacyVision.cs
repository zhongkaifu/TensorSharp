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
/// goldens (Fixtures/MuseGlimmer/LlamaCpp/[set/]legacy_vision.&lt;lane&gt;.json) before stage S9 deleted them, for the image
/// gates that are relative to legacy:
/// <list type="bullet">
/// <item>its teacher-forced argmax agreement with the lane's set (<see cref="MuseGlimmerParity.GoldenSet"/>), and the
/// distribution of every position's <see cref="MuseGlimmerParity.Position.TopDelta"/> (mean, p50, p90, p99, max);</item>
/// <item>on the lanes whose products quantize the activations (ggml_cuda, ggml_vulkan), each picture's 1 - cosine between the
/// legacy tower's rows and the F32 tower's (<see cref="F32Tower"/>).</item>
/// </list>
/// They were written on 2026-11-20 by a writer test that went with the legacy code (Fixtures/MuseGlimmer/README.md).
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
