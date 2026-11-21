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
/// As the text parity tests (<see cref="MuseGlimmerParityTests"/>), each lane is held to the argmax agreement with its own ggml
/// backend's llama.cpp set (<see cref="MuseGlimmerParity.GoldenSet"/>: LlamaCpp/ for Metal, LlamaCpp/cuda/, LlamaCpp/vulkan/),
/// and its first generated position is checked against llama.cpp only on the float-activation lanes
/// (<see cref="MuseGlimmerParityTests.FirstPositionRuleOf"/>: ggml_metal, MLX, direct CUDA); elsewhere it is logged.
/// Where a lane's rule is relative to legacy (<see cref="ImageRuleOf"/>, the rows on the quantized-product lanes), it reads
/// legacy's figures frozen on the lane next to its set (<see cref="MuseGlimmerLegacyVision"/>), so the rule outlives stage S9.
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

    internal sealed record File(string LlamaCppCommit, string ModelFile, string MmprojFile, IReadOnlyList<Record> Records, string Backend,
        IReadOnlyList<string> ImageSha256);

    private static readonly Dictionary<string, File> s_files = new(StringComparer.Ordinal);

    /// <summary>The image goldens of set <paramref name="set"/> (<see cref="MuseGlimmerParity.GoldenSets"/>; a set made before the
    /// provenance named its backend is Metal's).</summary>
    internal static File GoldensOf(string set)
    {
        lock (s_files)
        {
            if (s_files.TryGetValue(set, out File cached)) return cached;
            string resource = MuseGlimmerParity.ResourcePrefix + (set == "metal" ? "" : set + ".") + Golden;
            using Stream s = typeof(MuseGlimmerVisionParityTests).Assembly.GetManifestResourceStream(resource)
                ?? throw new FileNotFoundException($"golden {set}/{Golden} is not embedded (Fixtures/MuseGlimmer/LlamaCpp)");
            using JsonDocument doc = JsonDocument.Parse(s);
            JsonElement root = doc.RootElement;
            var records = new List<Record>();
            var shas = new List<string>();
            foreach (JsonElement r in root.GetProperty("records").EnumerateArray())
            {
                var steps = r.GetProperty("steps").EnumerateArray().Select(x => new MuseGlimmerParity.Step(x.GetProperty("logprob").GetDouble(),
                    x.GetProperty("top").EnumerateArray().Select(t => (t[0].GetInt32(), t[1].GetDouble())).ToArray())).ToArray();
                records.Add(new Record(r.GetProperty("name").GetString(), r.GetProperty("prompt").GetString(), r.GetProperty("image").GetString(),
                    r.GetProperty("prompt_n").GetInt32(), r.GetProperty("generated_tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray(), steps));
                shas.Add(r.GetProperty("image_sha256").GetString());
            }
            JsonElement llama = root.GetProperty("llama_cpp");
            string backend = llama.TryGetProperty("backend", out JsonElement b) ? b.GetString() : "metal";
            return s_files[set] = new File(llama.GetProperty("commit").GetString(), root.GetProperty("model").GetProperty("file").GetString(),
                root.GetProperty("mmproj").GetProperty("file").GetString(), records, backend, shas);
        }
    }

    /// <summary>The Metal set: its prompts, images and mmproj are every set's.</summary>
    internal static File Goldens => GoldensOf("metal");

    /// <summary>The set the lane under test is held to.</summary>
    private static File LaneGoldens => GoldensOf(MuseGlimmerParity.GoldenSet(MuseGlimmerParity.Backend()));

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

    /// <summary>The goldens themselves (no weights), every set: the Metal set's prompts, images and prompt lengths, the same
    /// llama.cpp commit, model and mmproj, each backend's own label, and a step with ten log-probabilities at the first generated
    /// position and four after it for every generated token, llama.cpp's greedy token at the top.</summary>
    [Fact]
    public void ImageGoldensAreWellFormed()
    {
        File metal = Goldens;
        foreach (string set in MuseGlimmerParity.GoldenSets)
        {
            File file = GoldensOf(set);
            Assert.Equal(set, file.Backend);
            Assert.Equal(metal.LlamaCppCommit, file.LlamaCppCommit);
            Assert.Equal(metal.ModelFile, file.ModelFile);
            Assert.Equal(metal.MmprojFile, file.MmprojFile);
            Assert.Equal(metal.Records.Select(r => (r.Name, r.Prompt, r.Image, r.PromptN)), file.Records.Select(r => (r.Name, r.Prompt, r.Image, r.PromptN)));
            Assert.Equal(metal.ImageSha256, file.ImageSha256);
            foreach (Record r in file.Records)
            {
                Assert.True(r.Generated.Length > 0, $"{set}/{r.Name}: nothing generated");
                Assert.Equal(r.Generated.Length, r.Steps.Length);
                for (int i = 0; i < r.Steps.Length; i++)
                {
                    Assert.Equal(r.Generated[i], r.Steps[i].Top[0].Id);
                    Assert.Equal(i == 0 ? 10 : 4, r.Steps[i].Top.Length);
                }
            }
        }
    }

    /// <summary>How a lane's image prompts are held (<see cref="ImageRuleOf"/>).</summary>
    internal enum ImageRule
    {
        /// <summary>Argmax agreement with the lane's set of at least <see cref="MinAgreement"/>, every parting a near-tie.</summary>
        NearTies,
        /// <summary>Argmax agreement with the lane's set of at least <see cref="MinAgreement"/>, and the distribution of every
        /// position's <see cref="MuseGlimmerParity.Position.TopDelta"/> no worse than the legacy model's frozen one by more than
        /// <see cref="DistributionSlack"/> in its mean, p50 and p90; its p99 and max, and the partings, are logged.</summary>
        Distribution,
        /// <summary>Argmax agreement with the lane's set no more than <see cref="LegacyAgreementSlack"/> under the legacy model's,
        /// frozen on the lane; partings are logged.</summary>
        LegacyRelative,
    }

    /// <summary>
    /// The image-prompt rule of a lane:
    /// <list type="bullet">
    /// <item>ggml_cuda and ggml_vulkan (<see cref="ImageRule.LegacyRelative"/>): their products quantize the activations, and
    /// both families part from the backend's own llama.cpp set where an engine's step is not a near-tie. Measured on an RTX PRO
    /// 4500 Blackwell: on ggml_cuda legacy 207/212 (frozen) and the runtime 208/212, each with two partings past the near-tie
    /// bound; on ggml_vulkan 210/217 (frozen) and 209/217, under 97%.</item>
    /// <item>direct CUDA (<see cref="ImageRule.Distribution"/>): one position, wide-describe @32, flips with the rows. The runtime
    /// family takes llama.cpp's token there with the legacy tower's rows and parts with the graph tower's (relative L2 1.3e-5
    /// apart), where llama.cpp's own gap is 0.277, past the near-tie bound. Legacy flips another position the same way
    /// (ocean-describe @20, a near-tie) when given the graph tower's rows. Over the 216 positions the runtime family is nearer
    /// llama.cpp than legacy at 130 and further at 86: mean 0.116 against 0.125, p50 0.083 against 0.090, p90 0.253 against
    /// 0.260 (RTX PRO 4500 Blackwell, IQ2_XXS). The tail is not held: it is the third-largest of 216 values at p99, six
    /// ocean-describe positions (a picture past the position table, where llama.cpp clamps and the reference zero-pads) where
    /// every family and tower sits at 0.3-0.7, and legacy's own p99 moves from 0.458 to 0.548 given the graph tower's rows.</item>
    /// <item>ggml_metal, MLX and the CPU lanes (<see cref="ImageRule.NearTies"/>).</item>
    /// </list>
    /// </summary>
    internal static ImageRule ImageRuleOf(BackendType backend) => backend switch
    {
        BackendType.GgmlCuda or BackendType.GgmlVulkan => ImageRule.LegacyRelative,
        BackendType.Cuda => ImageRule.Distribution,
        _ => ImageRule.NearTies,
    };

    /// <summary>How far under the legacy model's frozen agreement the runtime family's may sit (one percentage point).</summary>
    private const double LegacyAgreementSlack = 0.01;

    /// <summary>How much worse than the legacy model's frozen distribution the runtime family's may be, in each statistic held.</summary>
    private const double DistributionSlack = 0.10;

    /// <summary>One image prompt teacher-forced through a model: its positions, the first position's top-10 distance, whether its
    /// picture is past the position table, and the greedy continuation.</summary>
    internal sealed record PromptRun(Record Record, List<MuseGlimmerParity.Position> Positions, double FirstErr, bool PastTheTable,
        string Text, string Reference, int Width, int Height, int Rows, double EncodeMs);

    /// <summary>Every record of <paramref name="goldens"/> through <paramref name="model"/> with <paramref name="tower"/>'s rows:
    /// teacher-forced against llama.cpp's continuation, and greedy (logged).</summary>
    internal static List<PromptRun> RunImagePrompts(ModelBase model, IMuseGlimmerVisionTower tower, File goldens, Action<string> log)
    {
        int patch = model.Tokenizer.LookupToken("<|patch|>"), start = model.Tokenizer.LookupToken("<|image_start|>"), end = model.Tokenizer.LookupToken("<|image_end|>");
        var runs = new List<PromptRun>();
        foreach (Record r in goldens.Records)
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
            double firstErr = r.Steps[0].Top.Max(t => Math.Abs(first[t.Id] - t.Logprob));
            bool pastTheTable = w / tower.ImageProcessor.PatchSize > TableSide || h / tower.ImageProcessor.PatchSize > TableSide;
            string text = model.Tokenizer.Decode(greedy.TakeWhile(t => !model.Tokenizer.IsEos(t)).ToList());
            string reference = model.Tokenizer.Decode(r.Generated.ToList());
            log($"{r.Name} ({w}x{h}, {count} rows, encode {encodeMs:F0} ms): argmax = llama.cpp at {positions.Count(p => p.Agrees)}/{positions.Count} " +
                $"(greedy prefix {(firstDiff < 0 ? positions.Count : firstDiff)}); first position top-10 |dlogprob| <= {firstErr:F4}");
            log($"  ours:      {text.Replace("\n", "\\n")}");
            log($"  llama.cpp: {reference.Replace("\n", "\\n")}");
            runs.Add(new PromptRun(r, positions, firstErr, pastTheTable, text, reference, w, h, count, encodeMs));
        }
        return runs;
    }

    private void CheckAgainstLlamaCpp(ModelBase model)
    {
        IMuseGlimmerVisionTower tower = TowerOf(model);
        BackendType backend = MuseGlimmerParity.Backend();
        File goldens = LaneGoldens;
        ImageRule rule = ImageRuleOf(backend);
        bool holdFirst = MuseGlimmerParityTests.FirstPositionRuleOf(backend) == MuseGlimmerParityTests.FirstPositionRule.AgainstLlamaCpp;
        string lane = MuseGlimmerGoldens.LaneKey(backend);
        MuseGlimmerLegacyVision.Lane frozen = rule == ImageRule.NearTies ? null : MuseGlimmerLegacyVision.Find(lane, goldens.Backend);
        output.WriteLine($"backend {backend}; model {model.GetType().Name}; tower {tower.GetType().Name}; llama.cpp {goldens.LlamaCppCommit} " +
            $"({goldens.Backend} goldens); rule {rule}; first position {(holdFirst ? "against llama.cpp" : "logged")}");
        Assert.True(rule == ImageRule.NearTies || frozen != null,
            $"no legacy image figures frozen for {lane} next to the {goldens.Backend} goldens (MuseGlimmerLegacyVisionWriterTests writes " +
            $"Fixtures/MuseGlimmer/LlamaCpp/{(goldens.Backend == "metal" ? "" : goldens.Backend + "/")}{MuseGlimmerLegacyVision.FileName(lane)})");
        var runs = RunImagePrompts(model, tower, goldens, output.WriteLine);
        var failures = new List<string>();
        foreach (PromptRun run in runs)
        {
            double firstBound = backend == BackendType.Mlx ? FirstPositionLogprobBoundMlx
                : run.PastTheTable ? FirstPositionLogprobBoundPastTheTable : FirstPositionLogprobBound;
            if (run.PastTheTable)
                output.WriteLine($"  ({run.Record.Name}: a side of more than {TableSide} patches; llama.cpp clamps the position table's edges where the reference zero-pads; bound {firstBound})");
            if (run.FirstErr > firstBound && holdFirst)
                failures.Add($"{run.Record.Name}: first-position log-probabilities differ by {run.FirstErr:F4} (bound {firstBound})");
            else if (run.FirstErr > firstBound)
                output.WriteLine($"  (logged, not held on {backend}: {run.Record.Name}'s first-position log-probabilities {run.FirstErr:F4} from llama.cpp's, past {firstBound})");
            foreach (var p in run.Positions.Where(p => !p.Agrees))
            {
                string what = $"{run.Record.Name} @{p.Index}: llama.cpp {p.Reference} '{MuseGlimmerParity.Decode(model, p.Reference)}', ours {p.Ours} " +
                    $"'{MuseGlimmerParity.Decode(model, p.Ours)}'; our gap {p.OurGap:F4}, llama.cpp's gap " + (p.ReferenceGap is { } g ? g.ToString("F4") : "(not in its top 4)");
                bool nearTie = p.OurGap <= NearTieNats && p.ReferenceGap is { } gap && gap <= NearTieNats;
                output.WriteLine("  parting " + what + (nearTie ? "" : " (not a near-tie)"));
                if (rule == ImageRule.NearTies && !nearTie)
                    failures.Add(what);
            }
        }
        int total = runs.Sum(r => r.Positions.Count), agree = runs.Sum(r => r.Positions.Count(p => p.Agrees));
        double agreement = (double)agree / total;
        var q = MuseGlimmerLegacyVision.QuantilesOf(runs.SelectMany(r => r.Positions).Select(p => p.TopDelta));
        output.WriteLine($"overall: {agree}/{total} = {agreement:P2}; worst first-position |dlogprob| {runs.Max(r => r.FirstErr):F4}; " +
            $"top-k |dlogprob| over every position: mean {q.Mean:F4}, p50 {q.P50:F4}, p90 {q.P90:F4}, p99 {q.P99:F4}, max {q.Max:F4}" +
            (frozen == null ? "" : $"; legacy frozen on this lane {frozen.Agree}/{frozen.Positions} = {frozen.Agreement:P2}, mean " +
                $"{frozen.TopDelta.Mean:F4}, p50 {frozen.TopDelta.P50:F4}, p90 {frozen.TopDelta.P90:F4}, p99 {frozen.TopDelta.P99:F4}, max {frozen.TopDelta.Max:F4}"));
        Assert.True(failures.Count == 0, "positions that fail the lane's rule:\n" + string.Join("\n", failures));
        if (rule == ImageRule.LegacyRelative)
        {
            Assert.Equal(frozen.Positions, total);
            Assert.True(agreement >= frozen.Agreement - LegacyAgreementSlack,
                $"argmax agreement with llama.cpp {agreement:P2} is more than {LegacyAgreementSlack:P0} under the legacy model's frozen {frozen.Agreement:P2}");
            return;
        }
        Assert.True(agreement >= MinAgreement, $"argmax agreement with llama.cpp {agreement:P2} < {MinAgreement:P0}");
        if (rule == ImageRule.Distribution)
        {
            Assert.Equal(frozen.Positions, total);
            foreach (var (label, ours, legacy) in new[] { ("mean", q.Mean, frozen.TopDelta.Mean), ("p50", q.P50, frozen.TopDelta.P50), ("p90", q.P90, frozen.TopDelta.P90) })
                Assert.True(ours <= legacy * (1 + DistributionSlack),
                    $"top-k |dlogprob| {label} {ours:F4} is more than {DistributionSlack:P0} past the legacy model's frozen {legacy:F4}");
        }
    }

    // ================================================================ the towers' rows

    /// <summary>Pictures for the rows comparison: the golden images, and synthetic pixels at the design's sizes (336 x 336:
    /// 144 rows, 1036 x 1036: 1369 rows, a 588 x 868 portrait: 651 rows).</summary>
    internal static IEnumerable<(string Name, float[] Pixels, int Width, int Height)> Pictures(MuseGlimmerImageProcessor processor)
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

    /// <summary>On the quantized-product lanes (ggml_cuda, ggml_vulkan), each picture's 1 - cosine between the graph tower's rows
    /// and the F32 tower's (<see cref="MuseGlimmerLegacyVision.F32Tower"/>) may be at most this many times the legacy tower's,
    /// frozen on the lane. Both towers' products quantize the activations there: measured on an RTX PRO 4500 Blackwell, each sits
    /// at 1 - cosine 6e-5 to 1.7e-3 from the F32 tower on ggml_vulkan and 8e-4 to 4.7e-3 on ggml_cuda, and from each other at up
    /// to 3.9e-4 on ggml_vulkan, where legacy's per-op tower and the graph tower round independently. There the graph tower's
    /// distance is 0.74-1.67x legacy's picture by picture and 1.00x summed over the nine; on ggml_cuda, where the two share
    /// their rounding, 1.00x on every picture.</summary>
    private const double F32DistanceRatio = 1.25;

    /// <summary>Direct CUDA's tower multiplies F32 copies of its weights' values: it is held to the managed executor's F32 tower
    /// (cosine 1.0000000 to the per-op F32 tower measured).</summary>
    private const double MinCosineF32 = 0.99999;

    private void CompareTowers(BackendType backend)
    {
        using var lane = new MuseGlimmerVisionTowerTests.Lane(backend);
        string mmproj = FindMmproj();
        using var legacy = new MuseGlimmerVisionEncoder(mmproj, lane.Allocator);
        using var gguf = new GgufFile(mmproj);
        using var tower = new MuseGlimmerVisionTower(gguf, lane.Allocator, backend);
        bool quantized = MuseGlimmerLegacyVision.QuantizedProducts(backend);
        string laneKey = MuseGlimmerGoldens.LaneKey(backend), set = MuseGlimmerParity.GoldenSet(backend);
        MuseGlimmerLegacyVision.Lane frozen = quantized ? MuseGlimmerLegacyVision.Find(laneKey, set) : null;
        Assert.True(!quantized || frozen is { RowsOneMinusCos.Count: > 0 },
            $"no legacy tower figures frozen for {laneKey} (MuseGlimmerLegacyVisionWriterTests writes Fixtures/MuseGlimmer/LlamaCpp/{set}/" +
            $"{MuseGlimmerLegacyVision.FileName(laneKey)})");
        // The F32 reference: on the quantized lanes the cheaper F32 tower, on direct CUDA the managed one (another executor's
        // arithmetic than the tower under test).
        var (reference, owner, referenceName) = quantized || backend == BackendType.Cuda
            ? MuseGlimmerLegacyVision.F32Tower(gguf, preferDirectCuda: backend != BackendType.Cuda)
            : ((MuseGlimmerVisionTower)null, (IDisposable)null, (string)null);
        using (owner)
        using (reference)
        {
            output.WriteLine($"{backend}: graph tower on {tower.Graph.Executor.GetType().Name}" + (reference == null ? "; held to the legacy tower"
                : quantized ? $"; held to {F32DistanceRatio}x the legacy tower's frozen distance to {referenceName}" : $"; held to {referenceName}"));
            var failures = new List<string>();
            foreach (var (name, px, w, h) in Pictures(tower.ImageProcessor))
            {
                var sw = Stopwatch.StartNew();
                float[] a = legacy.EncodeToHostArray(px, w, h, out _);
                double legacyMs = sw.Elapsed.TotalMilliseconds;
                sw.Restart();
                float[] b = tower.EncodeToHost(px, w, h, out int tokens);
                double towerMs = sw.Elapsed.TotalMilliseconds;
                var (cos, worstRow) = Cosines(a, b, tower.ProjectionDim);
                var dist = MuseGlimmerVisionTowerTests.Distance(a, b);
                output.WriteLine($"  {name} ({w}x{h}, {tokens} rows): vs legacy cosine {cos:F7} (worst row {worstRow:F6}), relL2 {dist.RelL2:E2}, " +
                    $"max {dist.RelMax:E2}; legacy {legacyMs:F0} ms, graph tower {towerMs:F0} ms (one run each, informational)");
                if (reference == null)
                {
                    if (cos < MinCosine) failures.Add($"{name}: cosine {cos:F7} to the legacy tower < {MinCosine}");
                    continue;
                }
                float[] f = reference.EncodeToHost(px, w, h, out _);
                var (fCos, fWorst) = Cosines(f, b, tower.ProjectionDim);
                double graph = 1 - fCos, legacyNow = MuseGlimmerLegacyVision.OneMinusCos(f, a);
                if (!quantized)
                {
                    output.WriteLine($"    vs the F32 tower: cosine {fCos:F7} (worst row {fWorst:F6}); legacy {1 - legacyNow:F7}");
                    if (fCos < MinCosineF32) failures.Add($"{name}: cosine {fCos:F7} to the F32 tower < {MinCosineF32}");
                    continue;
                }
                Assert.True(frozen.RowsOneMinusCos.TryGetValue(name, out double legacyFrozen), $"{name}: no frozen legacy figure");
                output.WriteLine($"    1 - cosine to the F32 tower: graph {graph:E3} (worst row {fWorst:F6}), legacy {legacyNow:E3} now, {legacyFrozen:E3} frozen " +
                    $"({graph / legacyFrozen:F2}x, bound {F32DistanceRatio}x)");
                if (graph > F32DistanceRatio * legacyFrozen)
                    failures.Add($"{name}: 1 - cosine to the F32 tower {graph:E3} > {F32DistanceRatio} x legacy's {legacyFrozen:E3}");
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
            Assert.True(failures.Count == 0, $"{backend}:\n" + string.Join("\n", failures));
        }
    }

    /// <summary>The cosine of two row sets, and of their least similar row.</summary>
    private static (double Cos, double WorstRow) Cosines(float[] a, float[] b, int width)
    {
        double worstRow = 1;
        for (int row = 0; row < a.Length / width; row++)
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
        return (1 - MuseGlimmerLegacyVision.OneMinusCos(a, b), worstRow);
    }
}
