// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>[Fact] for the real-weight Muse-Glimmer vision tests: skips without the weights or without the mmproj the image
/// goldens were made with (both under TS_TEST_MODEL_DIR).</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class MuseGlimmerVisionFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Models";

    public MuseGlimmerVisionFactAttribute() => Skip = MuseGlimmerVisionParityTests.Skip(needsModel: true, BackendType.GgmlCpu);

    /// <summary>Only the mmproj is read (the towers' rows), on this backend (GgmlCpu: the pinned GGML backend).</summary>
    public BackendType TowerOnly
    {
        get => _towerOnly;
        set { _towerOnly = value; Skip = MuseGlimmerVisionParityTests.Skip(needsModel: false, value); }
    }
    private BackendType _towerOnly;
}

/// <summary>
/// Muse-Glimmer's image prompts against llama.cpp (Fixtures/MuseGlimmer/LlamaCpp/ref_vision.json, llama.cpp d3c3e242c: its
/// greedy continuations of nine image prompts with every token's top log-probabilities, made by
/// eng/validation/muse-glimmer-llamacpp-vision-goldens.py), teacher-forced through the legacy model with its tower and the
/// unified-runtime family with the graph tower; and the graph tower's rows against the legacy tower's on the real mmproj.
/// </summary>
public sealed class MuseGlimmerVisionParityTests(ITestOutputHelper output)
{
    private const string Golden = "ref_vision.json";

    /// <summary>As the text parity tests (<see cref="MuseGlimmerParityTests"/>): each engine finds the other's choice at most
    /// this many nats below its own where they disagree, llama.cpp listing ours among its top four.</summary>
    private const double NearTieNats = 0.25;

    private const double MinAgreement = 0.97;

    /// <summary>The bound on |our log-probability - llama.cpp's| over llama.cpp's top ten at the first generated position, on
    /// the GGML backends (S0's text bound; measured on ggml_metal: 0.06 or less, both families, but for the image past the
    /// table, <see cref="FirstPositionLogprobBoundPastTheTable"/>).</summary>
    private const double FirstPositionLogprobBound = 0.15;

    /// <summary>The first-position bound on MLX, whose text numerics sit further from llama.cpp's Metal ones on these
    /// prompts: measured legacy 0.311 and the runtime family 0.378 nat on number-7319's three-token answer (the towers' rows
    /// for it agree at cosine 0.999998), 0.18 or less elsewhere.</summary>
    private const double FirstPositionLogprobBoundMlx = 0.5;

    /// <summary>
    /// The first-position bound for an image with a side of more patches than the position table (word-ocean.jpg: 40 x 14).
    /// There the tower follows the reference's zero padding (<see cref="MuseGlimmerVisionGrid.ResamplePositionTable"/>) and
    /// llama.cpp d3c3e242c clamps, so the two read different position rows at the long side's edges. Measured on ggml_metal:
    /// the runtime family 0.160 / 0.180 nat on the two ocean prompts (0.103 / 0.120 with llama.cpp's clamping), legacy 0.150 /
    /// 0.161 (0.087 / 0.088); every other prompt 0.06 or less. The near-tie bound.
    /// </summary>
    private const double FirstPositionLogprobBoundPastTheTable = 0.25;

    /// <summary>The graph tower's rows against the legacy tower's on the same executor (design §4.5).</summary>
    private const double MinCosine = 0.9999;

    internal sealed record Record(string Name, string Prompt, string Image, int PromptN, int[] Generated, MuseGlimmerParity.Step[] Steps);

    internal sealed record File(string LlamaCppCommit, string ModelFile, string MmprojFile, IReadOnlyList<Record> Records);

    private static readonly Lazy<File> s_file = new(() =>
    {
        using Stream s = typeof(MuseGlimmerVisionParityTests).Assembly.GetManifestResourceStream(MuseGlimmerParity.ResourcePrefix + Golden)
            ?? throw new FileNotFoundException($"golden {Golden} is not embedded (Fixtures/MuseGlimmer/LlamaCpp)");
        using JsonDocument doc = JsonDocument.Parse(s);
        JsonElement root = doc.RootElement;
        var records = new List<Record>();
        foreach (JsonElement r in root.GetProperty("records").EnumerateArray())
        {
            var steps = r.GetProperty("steps").EnumerateArray().Select(x => new MuseGlimmerParity.Step(x.GetProperty("logprob").GetDouble(),
                x.GetProperty("top").EnumerateArray().Select(t => (t[0].GetInt32(), t[1].GetDouble())).ToArray())).ToArray();
            records.Add(new Record(r.GetProperty("name").GetString(), r.GetProperty("prompt").GetString(), r.GetProperty("image").GetString(),
                r.GetProperty("prompt_n").GetInt32(), r.GetProperty("generated_tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray(), steps));
        }
        return new File(root.GetProperty("llama_cpp").GetProperty("commit").GetString(), root.GetProperty("model").GetProperty("file").GetString(),
            root.GetProperty("mmproj").GetProperty("file").GetString(), records);
    });

    internal static File Goldens => s_file.Value;

    /// <summary>The position table's side in patches (sqrt of v.position_embd.weight's rows: 32).</summary>
    private static int TableSide => s_tableSide.Value;

    private static readonly Lazy<int> s_tableSide = new(() =>
    {
        using var gguf = new GgufFile(FindMmproj());
        return (int)Math.Round(Math.Sqrt((double)gguf.Tensors["v.position_embd.weight"].Shape[1]));
    });

    internal static string Skip(bool needsModel, BackendType towerBackend)
    {
        string skip = TestGates.ModelSkip(MuseGlimmerParity.ModelDirEnv, "muse-glimmer");
        if (skip != null) return skip;
        if (FindMmproj() == null) return $"Requires the vision projector (no {Goldens.MmprojFile} under {MuseGlimmerParity.ModelDirEnv}).";
        if (needsModel) return MuseGlimmerParity.Skip(needsDraft: false);
        return towerBackend switch
        {
            BackendType.Mlx => TestGates.MlxSkip,
            BackendType.Cuda => TestGates.CudaSkip,
            _ => TestGates.GgmlPinSkip(TestGates.PinnedGgmlBackend),
        };
    }

    internal static string FindMmproj()
    {
        string dir = Environment.GetEnvironmentVariable(MuseGlimmerParity.ModelDirEnv);
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir)) return null;
        string path = Path.Combine(dir, Goldens.MmprojFile);
        return System.IO.File.Exists(path) ? path : null;
    }

    internal static string RepoPath(string relative)
    {
        var here = new DirectoryInfo(AppContext.BaseDirectory);
        while (here != null && !Directory.Exists(Path.Combine(here.FullName, "TensorSharp.Runtime")))
            here = here.Parent;
        if (here == null) throw new DirectoryNotFoundException("no repository root above " + AppContext.BaseDirectory);
        return Path.Combine(here.FullName, relative.Replace('/', Path.DirectorySeparatorChar));
    }

    private static ModelBase Load(bool runtime)
    {
        BackendType backend = MuseGlimmerParity.Backend();
        ModelBase model = runtime
            ? MuseGlimmerArchitecture.CreateRuntimeModel(MuseGlimmerParity.FindModel(), backend)
            : ModelBase.Create(MuseGlimmerParity.FindModel(), backend);
        ((IVisionCapableModel)model).LoadVisionEncoder(FindMmproj());
        return model;
    }

    private static IMuseGlimmerVisionTower TowerOf(ModelBase model) => model switch
    {
        MuseGlimmerModel legacy => legacy.VisionEncoder,
        MuseGlimmerRuntimeModel runtime => runtime.VisionTower,
        _ => throw new ArgumentException(model.GetType().Name),
    };

    [MuseGlimmerVisionFact]
    public void LegacyImagePromptsMatchLlamaCpp()
    {
        using ModelBase model = Load(runtime: false);
        CheckAgainstLlamaCpp(model);
    }

    [MuseGlimmerVisionFact]
    public void RuntimeImagePromptsMatchLlamaCpp()
    {
        using ModelBase model = Load(runtime: true);
        CheckAgainstLlamaCpp(model);
    }

    private void CheckAgainstLlamaCpp(ModelBase model)
    {
        IMuseGlimmerVisionTower tower = TowerOf(model);
        output.WriteLine($"backend {MuseGlimmerParity.Backend()}; model {model.GetType().Name}; tower {tower.GetType().Name}; llama.cpp {Goldens.LlamaCppCommit}");
        int patch = model.Tokenizer.LookupToken("<|patch|>"), start = model.Tokenizer.LookupToken("<|image_start|>"), end = model.Tokenizer.LookupToken("<|image_end|>");
        int total = 0, agree = 0;
        double worstFirst = 0;
        var failures = new List<string>();
        foreach (Record r in Goldens.Records)
        {
            var (pixels, w, h) = tower.ImageProcessor.ProcessImage(RepoPath(r.Image));
            var sw = Stopwatch.StartNew();
            using Tensor encoded = tower.Encode(pixels, w, h);
            int count = (int)encoded.Sizes[0];
            double encodeMs = sw.Elapsed.TotalMilliseconds;
            List<int> tokens = model.Tokenizer.Encode(r.Prompt, addSpecial: true);
            int at = tokens.IndexOf(patch);
            Assert.True(at >= 0, $"{r.Name}: the prompt has no <|patch|>");
            var expanded = new List<int>(tokens.Take(at)) { start };
            expanded.AddRange(Enumerable.Repeat(0, count));
            expanded.Add(end);
            expanded.AddRange(tokens.Skip(at + 1));
            Assert.True(r.PromptN == expanded.Count, $"{r.Name}: {expanded.Count} prompt positions, llama.cpp's {r.PromptN} ({count} image rows)");
            // The model takes ownership of what it is given: a copy of the rows a request (each pass) would queue.
            void Queue(ModelBase m) => ((IVisionCapableModel)m).SetVisionEmbeddings(Ops.NewContiguous(encoded), at + 1);
            var golden = new MuseGlimmerParity.Golden(r.Name, r.Prompt, expanded.ToArray(), r.Generated, r.Steps);
            var positions = MuseGlimmerParity.TeacherForce(model, golden, out double[] first, Queue);
            int[] greedy = MuseGlimmerParity.GreedyContinue(model, expanded.ToArray(), r.Generated.Length, Queue);
            int firstDiff = positions.FindIndex(p => !p.Agrees);
            total += positions.Count;
            agree += positions.Count(p => p.Agrees);
            double firstErr = r.Steps[0].Top.Max(t => Math.Abs(first[t.Id] - t.Logprob));
            worstFirst = Math.Max(worstFirst, firstErr);
            bool pastTheTable = w / tower.ImageProcessor.PatchSize > TableSide || h / tower.ImageProcessor.PatchSize > TableSide;
            double firstBound = MuseGlimmerParity.Backend() == BackendType.Mlx ? FirstPositionLogprobBoundMlx
                : pastTheTable ? FirstPositionLogprobBoundPastTheTable : FirstPositionLogprobBound;
            string text = model.Tokenizer.Decode(greedy.TakeWhile(t => !model.Tokenizer.IsEos(t)).ToList());
            string reference = model.Tokenizer.Decode(r.Generated.ToList());
            output.WriteLine($"{r.Name} ({w}x{h}, {count} rows, encode {encodeMs:F0} ms): argmax = llama.cpp at {positions.Count(p => p.Agrees)}/{positions.Count} " +
                $"(greedy prefix {(firstDiff < 0 ? positions.Count : firstDiff)}); first position top-10 |dlogprob| <= {firstErr:F4}");
            output.WriteLine($"  ours:      {text.Replace("\n", "\\n")}");
            output.WriteLine($"  llama.cpp: {reference.Replace("\n", "\\n")}");
            if (pastTheTable)
                output.WriteLine($"  (a side of more than {TableSide} patches: llama.cpp clamps the position table's edges where the reference zero-pads; bound {firstBound})");
            if (firstErr > firstBound)
                failures.Add($"{r.Name}: first-position log-probabilities differ by {firstErr:F4} (bound {firstBound})");
            foreach (var p in positions.Where(p => !p.Agrees))
            {
                string what = $"{r.Name} @{p.Index}: llama.cpp {p.Reference} '{MuseGlimmerParity.Decode(model, p.Reference)}', ours {p.Ours} " +
                    $"'{MuseGlimmerParity.Decode(model, p.Ours)}'; our gap {p.OurGap:F4}, llama.cpp's gap " + (p.ReferenceGap is { } g ? g.ToString("F4") : "(not in its top 4)");
                output.WriteLine("  " + what);
                if (p.OurGap > NearTieNats || p.ReferenceGap is not { } gap || gap > NearTieNats)
                    failures.Add(what);
            }
        }
        double agreement = (double)agree / total;
        output.WriteLine($"overall: {agree}/{total} = {agreement:P2}; worst first-position |dlogprob| {worstFirst:F4}");
        Assert.True(failures.Count == 0, "positions that are not near-ties:\n" + string.Join("\n", failures));
        Assert.True(agreement >= MinAgreement, $"argmax agreement with llama.cpp {agreement:P2} < {MinAgreement:P0}");
    }

    // ================================================================ the towers' rows

    /// <summary>Pictures for the rows comparison: the golden images, and synthetic pixels at the design's sizes (336 x 336:
    /// 144 rows, 1036 x 1036: 1369 rows, a 588 x 868 portrait: 651 rows).</summary>
    private static IEnumerable<(string Name, float[] Pixels, int Width, int Height)> Pictures(MuseGlimmerImageProcessor processor)
    {
        foreach (string image in Goldens.Records.Select(r => r.Image).Distinct())
        {
            var (px, w, h) = processor.ProcessImage(RepoPath(image));
            yield return (Path.GetFileName(image), px, w, h);
        }
        foreach (var (w, h) in new[] { (336, 336), (1036, 1036), (588, 868) })
            yield return ($"synthetic {w}x{h}", MuseGlimmerVisionSynthetic.Pixels(w, h, (uint)(w + h)), w, h);
    }

    [MuseGlimmerVisionFact(TowerOnly = BackendType.GgmlCpu)]
    public void TheGraphTowerGivesTheLegacyTowersRowsOnThePinnedGgmlBackend() => CompareTowers(TestGates.PinnedGgmlBackend);

    [MuseGlimmerVisionFact(TowerOnly = BackendType.Mlx)]
    public void TheGraphTowerGivesTheLegacyTowersRowsOnMlx() => CompareTowers(BackendType.Mlx);

    [MuseGlimmerVisionFact(TowerOnly = BackendType.Cuda)]
    public void TheGraphTowerGivesTheLegacyTowersRowsOnDirectCuda() => CompareTowers(BackendType.Cuda);

    private void CompareTowers(BackendType backend)
    {
        using var lane = new MuseGlimmerVisionTowerTests.Lane(backend);
        string mmproj = FindMmproj();
        using var legacy = new MuseGlimmerVisionEncoder(mmproj, lane.Allocator);
        using var gguf = new GgufFile(mmproj);
        using var tower = new MuseGlimmerVisionTower(gguf, lane.Allocator, backend);
        output.WriteLine($"{backend}: graph tower on {tower.Graph.Executor.GetType().Name}");
        double worstCos = 1;
        foreach (var (name, px, w, h) in Pictures(tower.ImageProcessor))
        {
            var sw = Stopwatch.StartNew();
            float[] a = legacy.EncodeToHostArray(px, w, h, out _);
            double legacyMs = sw.Elapsed.TotalMilliseconds;
            sw.Restart();
            float[] b = tower.EncodeToHost(px, w, h, out int tokens);
            double towerMs = sw.Elapsed.TotalMilliseconds;
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += (double)a[i] * b[i];
                na += (double)a[i] * a[i];
                nb += (double)b[i] * b[i];
            }
            double cos = dot / Math.Sqrt(na * nb);
            // The worst row: what one image token's embedding is off by.
            int width = tower.ProjectionDim;
            double worstRow = 1;
            for (int row = 0; row < tokens; row++)
            {
                double d = 0, x = 0, y = 0;
                for (int c = 0; c < width; c++)
                {
                    double p = a[row * width + c], q = b[row * width + c];
                    d += p * q;
                    x += p * p;
                    y += q * q;
                }
                worstRow = Math.Min(worstRow, d / Math.Sqrt(x * y));
            }
            var dist = MuseGlimmerVisionTowerTests.Distance(a, b);
            output.WriteLine($"  {name} ({w}x{h}, {tokens} rows): cosine {cos:F7} (worst row {worstRow:F6}), relL2 {dist.RelL2:E2}, " +
                $"max {dist.RelMax:E2}; legacy {legacyMs:F0} ms, graph tower {towerMs:F0} ms (one run each, informational)");
            worstCos = Math.Min(worstCos, cos);
        }

        // A large image as one program and as programs over layer ranges (what a caller that yields the GPU lock gets):
        // alternating, after a warm-up of each (informational).
        var (_, pxL, wL, hL) = Pictures(tower.ImageProcessor).First(p => p.Width == 1036);
        int defaultYield = tower.YieldPatches;
        var whole = new List<double>();
        var split = new Dictionary<int, List<double>> { [1024] = new(), [2048] = new() };
        float[] oneProgram = tower.EncodeToHost(pxL, wL, hL, out _);
        foreach (int y in split.Keys)
        {
            tower.YieldPatches = y;
            var rows = tower.EncodeToHost(pxL, wL, hL, out _, betweenParts: () => { });
            var d = MuseGlimmerVisionTowerTests.Distance(oneProgram, rows);
            output.WriteLine($"  1036x1036 in {tower.LastParts} programs ({y} patches each): {Math.Max(d.RelL2, d.RelMax):E2} from one program's rows");
        }
        for (int round = 0; round < 2; round++)
        {
            var sw = Stopwatch.StartNew();
            tower.EncodeToHost(pxL, wL, hL, out _);
            whole.Add(sw.Elapsed.TotalMilliseconds);
            foreach (int y in split.Keys)
            {
                tower.YieldPatches = y;
                sw.Restart();
                tower.EncodeToHost(pxL, wL, hL, out _, betweenParts: () => { });
                split[y].Add(sw.Elapsed.TotalMilliseconds);
            }
        }
        tower.YieldPatches = defaultYield;
        output.WriteLine($"  1036x1036: one program {string.Join(", ", whole.Select(x => x.ToString("F0")))} ms; " +
            string.Join("; ", split.Select(kv => $"{kv.Key}-patch parts {string.Join(", ", kv.Value.Select(x => x.ToString("F0")))} ms")));
        Assert.True(worstCos >= MinCosine, $"{backend}: cosine {worstCos:F7} < {MinCosine}");
    }
}
