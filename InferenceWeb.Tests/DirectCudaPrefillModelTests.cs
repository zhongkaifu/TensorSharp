// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Model-level tests of the direct-CUDA backend (BackendType.Cuda) for the families the direct-CUDA
// prefill plan rewrites (docs/validation/direct-cuda-prefill/design/PLAN.md): gpt-oss, Qwen3.5 MoE
// and dense (Gated DeltaNet), Gemma-4 MoE (sliding-window + global attention) and Nemotron-H
// (Mamba2 + attention, dense ReLU^2 or MoE). The plan's packages replace the direct-CUDA prefill
// attention, MoE, Mamba2, GDN and dense-projection paths; these tests pin what the base does there:
//
//   * prefill rows at 17..1100 tokens and decode rows against the same file on ggml_cpu (the
//     reference every lane has), and against the managed cpu backend where direct CUDA tracks it
//     much more closely than ggml_cpu does;
//   * chunked prefill against single-pass prefill and the decode steps after it (KV, Mamba2 and
//     GDN state continuation across chunk boundaries);
//   * the same prefill repeated three times on one loaded model at lengths that are and are not
//     multiples of 3/8/16/32/64 and on both sides of the 512-token prefill-graph limit (the first
//     run plans a prefill CUDA graph, the second captures it, the third replays it);
//   * a K/V + recurrent-state snapshot restored into an empty cache against the straight run
//     (including Qwen3.5 prefill and decode graphs captured or replayed right after the inject),
//     and host reads of the K/V cache (host decode attention, block extracts) that must not
//     queue it for a full re-upload;
//   * a 16-row verify window (SpecForward, every row) against 16 single-token decode steps;
//   * the continuous-batching engine's concurrent greedy streams against solo ones, on files
//     whose greedy continuation provably depends on the context.
//
// Tolerances. Every number is relative L2 of a logits row (||x - ref|| / ||ref||). Each one is
// derived the way Stage 0 derived the G3 floors (docs/validation/direct-cuda-prefill/stage0): the
// base revision's distance A, measured on these files on the laptop RTX 3080 (sm_86), and the
// floor, the distance between two ggml_cpu configurations that should agree (the prompt whole vs
// in L/4 pieces, Stage 0's floor_chunk); the tolerance is max(1.5 x A, floor) rounded up to two
// significant digits (Stage 0's gate is 1.10 x A; 1.5 leaves room for another GPU's rounding).
// Where the base is bit-identical the test demands bit-identity. The values and the mutation
// checks that justify them are in docs/validation/direct-cuda-prefill/correctness/setup/. These
// tests catch faults (a stale cache row or state, a dropped sink, a wrong route, a cross-sequence
// read); the PARITY gate for the prefill packages is the harness's G3a/G3b against ggml_cuda on
// real weights (eng/validation/README-direct-cuda-prefill.md), not these files.
//
// Rows marked [KnownDefectData] fail on the base revision for reasons outside the prefill plan
// (DirectCudaKnownDefects, KnownDefectDataAttribute): they skip VISIBLY and are not passes;
// TS_TEST_RUN_KNOWN_DEFECTS=<id>|all runs them.
//
// The class changes process-wide settings (MAX_CONTEXT, TS_SCHED_DISABLE_BATCHED, the KV cache
// dtype, the MoE offload config), so it runs in the non-parallel engine-environment collection.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.DirectCudaModelChecks;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class DirectCudaPrefillModelTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-direct-cuda-prefill-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public DirectCudaPrefillModelTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "2048");     // the 1100-token rows cross 1024
        _environment.Set("TS_SCHED_DISABLE_BATCHED", null);
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);   // what --backend cuda serves with
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        try { Directory.Delete(_directory, true); } catch { /* best effort */ }
    }

    // ---- files ------------------------------------------------------------------------------

    private static readonly bool[] AllGlobal = { false, false, false, false, false, false };

    /// <summary>Synthetic file per family. gpt-oss uses sinks large enough that a path which drops
    /// or misapplies them moves the rows 5-10x more than the backends' rounding (P0's file: 1.5-2x,
    /// so the review's sink-dropping mutation passed every ggml_cpu row). gemma4moe keeps the
    /// 26B's 32-row sliding window and its local/global layer mix; gemma4moe_global has only
    /// global layers.</summary>
    private static readonly Dictionary<string, Func<string, string>> Files = new()
    {
        ["gptoss"] = p => GptOssSyntheticModelBuilder.SensitiveSinks().Write(p),
        ["qwen35moe"] = p => new Qwen35MoeSyntheticModelBuilder().Write(p),
        ["qwen35"] = p => new Qwen35DenseSyntheticModelBuilder().Write(p),
        ["gemma4moe"] = p => new Gemma4MoeSyntheticModelBuilder().Write(p),
        ["gemma4moe_global"] = p => new Gemma4MoeSyntheticModelBuilder { Pattern = AllGlobal }.Write(p),
        ["nemotron_h"] = p => Nemotron("SFAFSF").Write(p),
        ["nemotron_h_moe"] = p => Nemotron("SMAMSM").Write(p),
        // The real models' storage types at a width whose K holds whole 256-value super-blocks:
        // Nemotron-H 8B Q4_K_M (Q4_K projections and head) and Lightning MXFP4_MOE (MXFP4 up,
        // Q5_1 down experts). The width-64 files above never reach these routes.
        ["nemotron_h_q4k"] = p => WideNemotron("SFAFSF", SyntheticGguf.GgmlType.Q4_K).Write(p),
        ["nemotron_h_moe_mx"] = p => WideNemotron("SMAMSM", SyntheticGguf.GgmlType.Q8_0, SyntheticGguf.GgmlType.MXFP4, SyntheticGguf.GgmlType.Q5_1).Write(p),
    };

    private static NemotronSyntheticModelBuilder WideNemotron(string layout, SyntheticGguf.GgmlType linear,
        SyntheticGguf.GgmlType experts = SyntheticGguf.GgmlType.Q8_0, SyntheticGguf.GgmlType? downExperts = null) => new()
    {
        Layout = layout,
        AttentionHeads = 16,
        AttentionKvHeads = 1,
        AttentionHeadDim = 128,
        Width = 256,
        SsmInnerWidth = 256,
        DenseFfWidth = 256,
        SharedFfWidth = 256,
        ExpertFf = 256,
        LinearType = linear,
        ExpertType = experts,
        DownExpertType = downExperts,
    };

    private static NemotronSyntheticModelBuilder Nemotron(string layout) => new()
    {
        Layout = layout,
        AttentionHeads = 16,
        AttentionKvHeads = 1,
        AttentionHeadDim = 128,
    };

    private string WriteModel(string family) => Files[family](Path.Combine(_directory, family + ".gguf"));

    private static ModelBase Load(string path, BackendType backend) => ModelBase.Create(path, backend);

    private void Report(string what, IEnumerable<double> values) => _output.WriteLine($"{what}: {Fmt(values)}");

    private static void AssertFinite(IEnumerable<float[]> rows, string where)
    {
        int i = 0;
        foreach (float[] row in rows)
        {
            Assert.True(row.All(float.IsFinite), $"{where}: row {i} has a non-finite logit");
            i++;
        }
    }

    private static int[] Quarters(int length)
    {
        int q = Math.Max(1, length / 4);
        return new[] { q, q, q };
    }

    /// <summary>Asserts every row against its tolerance; a tolerance of 0 demands bit-identity.</summary>
    private static void AssertRows(string what, IReadOnlyList<float[]> actual, IReadOnlyList<float[]> expected, Func<int, double> tolerance)
    {
        for (int i = 0; i < actual.Count; i++)
        {
            double tol = tolerance(i);
            if (tol == 0)
                Assert.True(BitIdentical(actual[i], expected[i]),
                    $"{what}: row {i} is {RelL2(actual[i], expected[i]):E3} away, but the base is bit-identical here");
            else
                Assert.True(RelL2(actual[i], expected[i]) <= tol,
                    $"{what}: row {i} is {RelL2(actual[i], expected[i]):E3} away (tolerance {tol:E2})");
        }
    }

    /// <summary>
    /// The first decode step of a model's first trajectory runs per-op; every later one replays
    /// the captured decode graph, whose kernels round slightly differently (measured: 0 on the MoE
    /// and Nemotron files, 1.25e-7 on the dense BF16 Qwen3.5 file; 1-3e-2 on real Qwen3.5-0.8B
    /// Q8_0, see DirectCudaRealModelTests). So a repeated trajectory must be bit-identical from
    /// its second run on, and the first run may differ on its decode rows by this much (the
    /// Qwen3.5 prefill-graph ring-phase defect D5, fixed, moved them 0.13-0.75, D3 by 0.07 and more).
    /// The sliding-window Gemma file's first run differs from later runs on a few decode rows
    /// (with the D3 fixes: 1.7e-3 at 64 tokens, 4.75e-3 at 511, 0 at the other lengths; runs 2 and
    /// 3 bit-identical): rounding between its per-op and graph-replayed decode steps, not traced
    /// further.
    /// </summary>
    private static double FirstRunDecodeBand(string family) => family == "gemma4moe" ? 0.0072 : 1e-4;

    private void AssertReproduces(string what, IReadOnlyList<List<float[]>> runs, string family)
    {
        for (int r = 1; r < runs.Count; r++)
        {
            var rel = runs[r].Select((row, i) => RelL2(row, runs[0][i])).ToList();
            int same = runs[r].Select((row, i) => BitIdentical(row, runs[0][i])).Count(x => x);
            Report($"{what}: run {r + 1} vs run 1 (bit-identical rows {same}/{rel.Count})", rel);
        }
        for (int r = 2; r < runs.Count; r++)
            AssertRows($"{what} run {r + 1} vs run 2", runs[r], runs[1], _ => 0);
        AssertRows($"{what} run 2 vs run 1", runs[1], runs[0], row => row == 0 ? 0 : FirstRunDecodeBand(family));
    }

    // ---- 1. prefill rows against ggml_cpu ---------------------------------------------------

    /// <summary>max(1.5 x A, floor) of the last prompt row against ggml_cpu, per family and length
    /// (A on the base; e.g. gptoss 17 tokens: A 2.1e-2; the sink-dropping mutation moves it 0.26).</summary>
    private static readonly Dictionary<string, Dictionary<int, double>> PrefillVsGgml = new()
    {
        ["gptoss"] = new() { [17] = 0.031, [97] = 0.047, [160] = 0.028, [257] = 0.067, [513] = 0.048, [1100] = 0.030 },
        ["qwen35moe"] = new() { [17] = 0.068, [97] = 0.049, [160] = 0.13, [257] = 0.098, [513] = 0.092, [1100] = 0.062 },
        ["qwen35"] = new() { [17] = 0.0034, [97] = 0.0029, [160] = 0.0025, [257] = 0.0022, [513] = 0.0026, [1100] = 0.0021 },
        // 257: prompt salt 1 (PrefillSalt); A 6.2e-3, floor 3.6e-3.
        ["gemma4moe"] = new() { [17] = 0.0077, [97] = 0.0078, [160] = 0.0086, [257] = 0.0093, [513] = 0.0072, [1100] = 0.0075 },
        ["gemma4moe_global"] = new() { [17] = 0.0083, [97] = 0.0074, [160] = 0.0088, [257] = 0.0075, [513] = 0.0078, [1100] = 0.0069 },
        ["nemotron_h"] = new() { [17] = 0.015, [97] = 0.017, [160] = 0.016, [257] = 0.017, [513] = 0.028, [1100] = 0.021 },
        // Above 512 rows in one forward the expert matmuls take another route (513-2000: 1.4e-2 to
        // 2.7e-2 from managed cpu, against 6e-8 at <= 512); 1100 is D6 (0.33), its tolerance the
        // neighbours'.
        ["nemotron_h_moe"] = new() { [17] = 0.017, [97] = 0.017, [160] = 0.013, [257] = 0.015, [513] = 0.036, [1100] = 0.041 },
        ["nemotron_h_q4k"] = new() { [17] = 0.023, [97] = 0.024, [160] = 0.023, [257] = 0.022, [513] = 0.020, [1100] = 0.033 },
        // 513: D6 (1.05 from ggml_cpu); the tolerance is the neighbours'.
        ["nemotron_h_moe_mx"] = new() { [17] = 0.017, [97] = 0.027, [160] = 0.020, [257] = 0.024, [513] = 0.027, [1100] = 0.048 },
    };

    [CudaTheory(GgmlBackend = BackendType.GgmlCpu)]
    [MemberData(nameof(PrefillRowCases))]
    [KnownDefectData("D6", "nemotron_h_moe", 1100)]
    [KnownDefectData("D6", "nemotron_h_moe_mx", 513)]
    public void PrefillRows_TrackGgmlCpu(string family, int length)
    {
        // 17: below the 32-row expert and F16-GEMM thresholds (scalar quantized matmuls);
        // 97: past every sliding window, not a multiple of 3/8/16/32; 160: two flash-style KV tiles;
        // 257: past gpt-oss's 256-row ForwardRefill chunk; 513: past the 512-token prefill-graph
        // limit; 1100: past 1024 rows.
        string path = WriteModel(family);
        int[] prompt = Prompt(length, PrefillSalt(family, length));
        float[] reference;
        using (var ggml = Load(path, BackendType.GgmlCpu))
            reference = PrefillLogits(ggml, prompt);
        float[] row;
        using (var cuda = Load(path, BackendType.Cuda))
            row = PrefillLogits(cuda, prompt);
        AssertFinite(new[] { row }, $"{family} prefill {length}");
        double rel = RelL2(row, reference), tol = PrefillVsGgml[family][length];
        _output.WriteLine($"{family} {length}-token prefill row vs ggml_cpu: {rel:E3} (tolerance {tol:E2})");
        Assert.True(rel <= tol, $"{family}: the {length}-token prefill row is {rel:E3} from ggml_cpu (tolerance {tol:E2})");
    }

    public static TheoryData<string, int> PrefillRowCases()
    {
        var data = new TheoryData<string, int>();
        foreach (var (family, lengths) in PrefillVsGgml)
            foreach (int n in lengths.Keys)
                if (!IsKnownPrefillDefect(family, n))
                    data.Add(family, n);
        return data;
    }

    private static bool IsKnownPrefillDefect(string family, int length)
        => (family, length) is ("nemotron_h_moe", 1100) or ("nemotron_h_moe_mx", 513);

    /// <summary>
    /// The prompt salt of a prefill row. 0 everywhere except the sliding-window Gemma file at 257
    /// tokens, whose salt-0 last row is ill-conditioned (formerly listed as D4, not a code defect):
    /// it amplifies rounding about tenfold, so CUDA's GEMM precision setting alone moves it from
    /// 7.1e-2 to 3.3e-2 off ggml_cpu (TS_CUDA_QMM_F16GEMM_MIN_ROWS=100000) and managed cpu sits
    /// 2.0e-2 off ggml_cpu, while the ggml floor is 6.7e-3 and every layer picks the same experts
    /// on every backend. Two more of the 30 prompts measured at 255-259 tokens on this file behave
    /// alike (255/salt 4, 256/salt 1, a router near-tie). Salt 1 at 257 is well-conditioned (cuda
    /// 6.2e-3 default, 5.2e-3 precise; cpu 5.4e-3; floor 3.6e-3) and runs the same code path.
    /// </summary>
    private static int PrefillSalt(string family, int length) => (family, length) is ("gemma4moe", 257) ? 1 : 0;

    // ---- 2. decode rows against ggml_cpu ----------------------------------------------------

    /// <summary>(prefill row, median decode row, worst decode row) tolerances against ggml_cpu.</summary>
    private static readonly Dictionary<(string, int), (double Prefill, double Median, double Max)> DecodeVsGgml = new()
    {
        [("gptoss", 96)] = (0.037, 0.061, 0.11),
        [("gptoss", 97)] = (0.047, 0.056, 0.063),
        [("qwen35moe", 96)] = (0.072, 0.090, 0.19),
        [("qwen35moe", 97)] = (0.049, 0.11, 0.23),
        [("qwen35", 96)] = (0.0025, 0.0032, 0.0042),
        [("qwen35", 97)] = (0.0029, 0.0029, 0.0042),
        [("gemma4moe_global", 96)] = (0.0056, 0.0082, 0.017),
        [("gemma4moe_global", 97)] = (0.0074, 0.0080, 0.014),
        [("gemma4moe", 96)] = (0.0057, 0.0075, 0.024),     // A 3.7e-3 / 5.0e-3 / 1.57e-2 (with the D3 fixes)
        [("gemma4moe", 97)] = (0.0078, 0.0082, 0.012),     // A 5.2e-3 / 5.4e-3 / 7.8e-3
        [("nemotron_h", 96)] = (0.021, 0.021, 0.023),
        [("nemotron_h", 97)] = (0.017, 0.019, 0.027),
        [("nemotron_h_moe", 96)] = (0.013, 0.019, 0.021),
        [("nemotron_h_moe", 97)] = (0.017, 0.018, 0.020),
        [("nemotron_h_q4k", 96)] = (0.032, 0.032, 0.042),
        [("nemotron_h_q4k", 97)] = (0.024, 0.036, 0.047),
        [("nemotron_h_moe_mx", 96)] = (0.026, 0.024, 0.029),
        [("nemotron_h_moe_mx", 97)] = (0.027, 0.023, 0.029),
    };

    [CudaTheory(GgmlBackend = BackendType.GgmlCpu)]
    [InlineData("gptoss", 96)]
    [InlineData("gptoss", 97)]
    [InlineData("qwen35moe", 96)]
    [InlineData("qwen35moe", 97)]
    [InlineData("qwen35", 96)]
    [InlineData("qwen35", 97)]
    [InlineData("gemma4moe_global", 96)]
    [InlineData("gemma4moe_global", 97)]
    [InlineData("gemma4moe", 96)]
    [InlineData("gemma4moe", 97)]
    [InlineData("nemotron_h", 96)]
    [InlineData("nemotron_h", 97)]
    [InlineData("nemotron_h_moe", 96)]
    [InlineData("nemotron_h_moe", 97)]
    [InlineData("nemotron_h_q4k", 96)]
    [InlineData("nemotron_h_q4k", 97)]
    [InlineData("nemotron_h_moe_mx", 96)]
    [InlineData("nemotron_h_moe_mx", 97)]
    public void DecodeRows_AfterPrefill_TrackGgmlCpu(string family, int length)
    {
        // A fresh model: the prompt's prefill (never a graph replay), then 10 teacher-forced decode
        // steps through the default decode path (Qwen3.5 and Gemma-4 capture a decode graph on
        // the second step and replay it from the third). 97 is not a multiple of the GDN conv
        // ring (3): a prefill that leaves the ring phase stale shows up from the first decode row.
        string path = WriteModel(family);
        int[] prompt = Prompt(length), forced = Forced(10);
        List<float[]> reference;
        using (var ggml = Load(path, BackendType.GgmlCpu))
            reference = Trajectory(ggml, prompt, forced);
        List<float[]> actual;
        using (var cuda = Load(path, BackendType.Cuda))
            actual = Trajectory(cuda, prompt, forced);
        AssertFinite(actual, family);
        var rel = actual.Select((row, i) => RelL2(row, reference[i])).ToList();
        Report($"{family} {length}-token prefill + 10 decode rows vs ggml_cpu", rel);
        var decode = rel.Skip(1).ToList();
        var tol = DecodeVsGgml[(family, length)];
        double median = Quantile(decode, 0.5);
        Assert.True(rel[0] <= tol.Prefill, $"{family}: prefill row {rel[0]:E3} > {tol.Prefill:E2}");
        Assert.True(median <= tol.Median, $"{family}: median decode row {median:E3} > {tol.Median:E2}");
        Assert.True(decode.Max() <= tol.Max, $"{family}: worst decode row {decode.Max():E3} > {tol.Max:E2}");
    }

    // ---- 3. chunked against single-pass prefill (state continuation) ------------------------

    private static readonly int[][] Chunkings = { new[] { 1, 15, 33 }, new[] { 16, 16, 16, 16, 16 }, new[] { 31, 2 }, null, new[] { 64 } };

    /// <summary>Per-row tolerance of a chunked run against the single pass (0 = bit-identical, as
    /// the base is: Nemotron except the rows its 16/24/31-row chunks route differently).</summary>
    private static double ChunkTolerance(string family, int length, int[] chunks, int row) => family switch
    {
        "gptoss" => row == 0 ? 0.0075 : 0.0068,
        "qwen35moe" => row == 0 ? 0.0075 : 0.015,
        "qwen35" => row == 0 ? 0.0011 : 0.00031,
        "gemma4moe" => row == 0 ? 0.0089 : 0.017,  // A 5.9e-3 / 1.08e-2 (with the D3 fixes)
        "gemma4moe_global" => row == 0 ? 0.0074 : 0.017,
        "nemotron_h" => chunks[0] is 1 or 64 ? 0 : row == 0 ? 0.015 : row == 1 && length == 96 ? 0.013 : 0,
        "nemotron_h_moe" => 0,
        "nemotron_h_q4k" => row == 0 ? 0.0036 : 0.021,
        "nemotron_h_moe_mx" => row == 0 ? 0.0041 : 0.0035,
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    [CudaTheory]
    [InlineData("gptoss", 96)]
    [InlineData("gptoss", 97)]
    [InlineData("qwen35moe", 96)]
    [InlineData("qwen35moe", 97)]
    [InlineData("qwen35", 96)]
    [InlineData("qwen35", 97)]
    [InlineData("gemma4moe", 96)]
    [InlineData("gemma4moe", 97)]
    [InlineData("gemma4moe_global", 96)]
    [InlineData("gemma4moe_global", 97)]
    [InlineData("nemotron_h", 96)]
    [InlineData("nemotron_h", 97)]
    [InlineData("nemotron_h_moe", 96)]
    [InlineData("nemotron_h_moe", 97)]
    [InlineData("nemotron_h_q4k", 97)]
    [InlineData("nemotron_h_moe_mx", 97)]
    public void ChunkedPrefill_ContinuesLikeSinglePass(string family, int length)
    {
        // Chunk boundaries at odd offsets: a 1-row chunk (decode-shaped prefill), sub-32-row
        // chunks (scalar matmul routes), a 33-row chunk past the expert threshold, a 2-row tail,
        // quarters, and a 64-row head. The KV rows, the Mamba2 conv/SSM state and the GDN
        // conv/delta state must carry across every boundary; the decode rows after the prompt
        // read all of it. No chunk shape repeats a key within a row, so no prefill graph is
        // captured here (RepeatedChunkedPrefill_Reproduces covers that).
        string path = WriteModel(family);
        int[] prompt = Prompt(length);
        int[] forced = Forced(4);
        using var cuda = Load(path, BackendType.Cuda);
        List<float[]> single = Trajectory(cuda, prompt, forced);
        AssertFinite(single, family);
        foreach (int[] given in Chunkings)
        {
            int[] chunks = given ?? Quarters(length);
            List<float[]> chunked = Trajectory(cuda, prompt, forced, chunks);
            AssertFinite(chunked, family);
            var rel = chunked.Select((row, i) => RelL2(row, single[i])).ToList();
            int bitIdentical = chunked.Select((row, i) => BitIdentical(row, single[i])).Count(x => x);
            Report($"{family} {length}: chunks [{string.Join(",", chunks)},rest] vs single pass (bit-identical rows {bitIdentical}/{rel.Count})", rel);
            AssertRows($"{family} chunks [{string.Join(",", chunks)}]", chunked, single, row => ChunkTolerance(family, length, chunks, row));
        }
    }

    // ---- 3b. the server's refill path ---------------------------------------------------------

    /// <summary>Refill (the server's re-prefill path) vs one ForwardPrompt, 600 tokens: per-row
    /// tolerance for (prompt row, decode rows); 0 = bit-identical, as the base is.</summary>
    private static readonly Dictionary<string, (double Prefill, double Decode)> RefillTolerance = new()
    {
        ["gptoss"] = (0.056, 0.032),     // 256-row chunks: 3.7e-2 / 1.1e-2..2.1e-2
        ["qwen35moe"] = (0, 0),
        ["qwen35"] = (0, 0),
        ["gemma4moe"] = (0, 0),
        ["gemma4moe_global"] = (0, 0),
        ["nemotron_h"] = (0, 0),
        ["nemotron_h_moe"] = (0, 0),
        ["nemotron_h_q4k"] = (0, 0),
        ["nemotron_h_moe_mx"] = (0, 0),
    };

    [CudaTheory]
    [InlineData("gptoss")]
    [InlineData("qwen35moe")]
    [InlineData("qwen35")]
    [InlineData("gemma4moe")]
    [InlineData("gemma4moe_global")]
    [InlineData("nemotron_h")]
    [InlineData("nemotron_h_moe")]
    [InlineData("nemotron_h_q4k")]
    [InlineData("nemotron_h_moe_mx")]
    public void ForwardRefill_TracksSinglePass(string family)
    {
        // ForwardRefill is what the server runs to (re)build a sequence's cache: gpt-oss splits
        // it into 256-row chunks (TS_GPTOSS_PREFILL_CHUNK), Nemotron feeds the last prompt token
        // through the decode path. 600 tokens cross 256, 512 and both sliding windows; the decode
        // rows after it read the state it left.
        string path = WriteModel(family);
        int[] prompt = Prompt(600);
        int[] forced = Forced(4);
        using var cuda = Load(path, BackendType.Cuda);
        List<float[]> single = Trajectory(cuda, prompt, forced);
        cuda.ResetKVCache();
        var refill = new List<float[]> { (float[])cuda.ForwardRefill(prompt).Clone() };
        foreach (int token in forced)
            refill.Add((float[])cuda.Forward(new[] { token }).Clone());
        AssertFinite(refill, family);
        var tol = RefillTolerance[family];
        Report($"{family} ForwardRefill(600) vs ForwardPrompt (bit-identical rows {refill.Select((r, i) => BitIdentical(r, single[i])).Count(x => x)}/{refill.Count})",
            refill.Select((r, i) => RelL2(r, single[i])));
        AssertRows($"{family} refill", refill, single, row => row == 0 ? tol.Prefill : tol.Decode);
    }

    // ---- 4. repeated prefill on one loaded model (graph plan / capture / replay) -------------

    [CudaTheory]
    [MemberData(nameof(RepeatedPrefillCases))]
    public void RepeatedPrefill_Reproduces(string family, int length)
    {
        // The same prompt and steps three times on one loaded model. Direct-CUDA Qwen3.5 plans a
        // per-op prefill CUDA graph on a key's first sighting, captures it on the second and
        // replays it from the third (up to 512 tokens); the decode path captures its own graph.
        // Every kernel the base uses is deterministic for a fixed call sequence (AssertReproduces
        // for the one first-run allowance). Lengths: multiples of 3/16/32/64 and not, and both
        // sides of the 512-token graph limit. Qwen3.5 at lengths up to 512 that are not multiples
        // of the GDN conv ring (3) was D5: a replay left the host ring phase behind the device
        // ring, and an aborted capture (the MoE file's host expert routing syncs) advanced the
        // layers it reached twice; the ring-phase check fails at the run that lost track.
        string path = WriteModel(family);
        int[] prompt = Prompt(length);
        int[] forced = Forced(6);
        using var cuda = Load(path, BackendType.Cuda);
        var runs = Enumerable.Range(0, 3).Select(r => TrajectoryKeepingRingPhase(cuda, prompt, forced, null, $"{family} {length} run {r + 1}")).ToList();
        AssertReproduces($"{family} {length}", runs, family);
    }

    private static readonly int[] RepeatLengths = { 17, 64, 96, 97, 130, 257, 511, 513 };

    public static IEnumerable<object[]> RepeatedPrefillCases()
    {
        foreach (string family in Files.Keys)
            foreach (int n in RepeatLengths)
                yield return new object[] { family, n };
    }

    [CudaTheory]
    [InlineData("gptoss")]
    [InlineData("qwen35moe")]
    [InlineData("qwen35")]
    [InlineData("gemma4moe_global")]
    [InlineData("nemotron_h")]
    [InlineData("nemotron_h_moe")]
    public void RepeatedChunkedPrefill_Reproduces(string family)
    {
        // A server splits prompts into the same chunk shapes over and over: the same 97-token
        // prompt in chunks of 31, 2 and 64, three times. 31 and 64 are not multiples of 3, so a
        // prefill graph captured or replayed for a chunk must leave the GDN ring phase where the
        // per-op chunk does.
        string path = WriteModel(family);
        int[] prompt = Prompt(97), forced = Forced(6);
        int[] chunks = { 31, 2, 64 };
        using var cuda = Load(path, BackendType.Cuda);
        var runs = Enumerable.Range(0, 3).Select(r => TrajectoryKeepingRingPhase(cuda, prompt, forced, chunks, $"{family} chunks [31,2,64] run {r + 1}")).ToList();
        AssertReproduces($"{family} chunks [31,2,64]", runs, family);
    }

    /// <summary>A <see cref="Trajectory"/> that, on direct-CUDA Qwen3.5, also checks the GDN conv
    /// ring bookkeeping: every recurrent layer's host ring phase (the write index its device kernel
    /// is launched with, and the phase the decode graph applies to every layer) must equal the
    /// tokens fed since the reset modulo conv_kernel - 1. A path that forgets the bookkeeping fails
    /// here at the run that forgot it, even where the logits happen to survive.</summary>
    private static List<float[]> TrajectoryKeepingRingPhase(ModelBase model, int[] prompt, int[] forced, int[] chunks, string what)
    {
        List<float[]> rows = Trajectory(model, prompt, forced, chunks);
        if (GdnRingPhases(model) is { } ring)
        {
            int expected = (prompt.Length + forced.Length) % ring.Ring;
            Assert.True(ring.Phases.All(p => p == expected),
                $"{what}: GDN conv ring phases [{string.Join(",", ring.Phases)}] after {prompt.Length + forced.Length} tokens, expected {expected} on every recurrent layer");
        }
        return rows;
    }

    [CudaTheory]
    [InlineData("gemma4moe")]
    [InlineData("gemma4moe_global")]
    public void DefaultDecode_Reproduces(string family)
    {
        // Gemma-4's default decode replays a captured decode graph from the third step. It must
        // repeat bit for bit, inside the window too (20 + 10 tokens < 32). D3 (cause 1) failed it on
        // the local/global file: the unweighted V/router norm kept ONE all-ones weight vector and
        // reallocated it whenever the width changed, so a global layer freed it in the middle of
        // the decode-graph capture, later captured kernels reused its block as scratch, and every
        // replay read the previous replay's activations as the first layers' norm weights.
        string path = WriteModel(family);
        int[] prompt = Prompt(20), forced = Forced(10);
        using var cuda = Load(path, BackendType.Cuda);
        var runs = Enumerable.Range(0, 3).Select(_ => Trajectory(cuda, prompt, forced)).ToList();
        AssertReproduces($"{family} default decode", runs, family);
    }

    [CudaTheory]
    [InlineData("cuda", 97)]
    [InlineData("cuda", 257)]
    [InlineData("cpu", 97)]
    [InlineData("cpu", 257)]
    public void SlidingWindowRing_AfterAPrefillPastTheWindow_HoldsItsLastRows(string backend, int length)
    {
        // A prefill longer than the 32-row sliding window writes into the window's ring, where rows
        // s and s + 32 share a slot and only the later may survive. Layer 0's K/V rows depend only
        // on (token, position), so two prompts of the same length that share their last 32 tokens
        // must leave bit-identical layer-0 rings (same GEMM shapes, so the same rounding too). D3
        // (cause 2): every row was written at once and the CUDA copy kernel's threads (the managed
        // Parallel.For too, more rarely) raced for each slot, so the ring kept an arbitrary mix of
        // rows from different windows (97 tokens: 97-106 of 128 slots wrong, varying run to run).
        string path = WriteModel("gemma4moe");
        const int window = 32;
        int[] a = Prompt(length), b = Prompt(length, salt: 7);
        Array.Copy(a, length - window, b, length - window, window);
        Assert.NotEqual(a[0], b[0]);
        using var model = Load(path, backend == "cpu" ? BackendType.Cpu : BackendType.Cuda);
        for (int repeat = 0; repeat < 3; repeat++)
        {
            model.ResetKVCache();
            model.ForwardPrompt(a);
            float[] ka = LayerCache(model, 0, value: false), va = LayerCache(model, 0, value: true);
            model.ResetKVCache();
            model.ForwardPrompt(b);
            float[] kb = LayerCache(model, 0, value: false), vb = LayerCache(model, 0, value: true);
            Assert.Equal(Gemma4MoeSyntheticModelBuilder.LocalKvHeads * window * Gemma4MoeSyntheticModelBuilder.LocalHeadDim, ka.Length);
            Assert.Contains(ka, x => x != 0);
            int kDiff = ka.Zip(kb, (x, y) => x != y).Count(d => d), vDiff = va.Zip(vb, (x, y) => x != y).Count(d => d);
            _output.WriteLine($"{backend} {length} tokens, repeat {repeat}: layer-0 ring elements differing between the two prompts K {kDiff}/{ka.Length}, V {vDiff}/{va.Length}");
            Assert.True(kDiff == 0 && vDiff == 0, $"{backend} {length} tokens: the layer-0 ring keeps rows from before the last window (K {kDiff}, V {vDiff} elements differ)");
        }
    }

    /// <summary>Layer <paramref name="layer"/>'s K or V cache of a Gemma-4 model as floats (the host
    /// mirror, synchronised from the device), read by reflection.</summary>
    private static unsafe float[] LayerCache(ModelBase model, int layer, bool value)
    {
        var field = typeof(Gemma4Model).GetField(value ? "_kvCacheV" : "_kvCacheK",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?? throw new InvalidOperationException("Gemma4Model's _kvCacheK/_kvCacheV were renamed: update LayerCache");
        var tensor = ((TensorSharp.Tensor[])field.GetValue(model))[layer];
        long n = tensor.ElementCount();
        var result = new float[n];
        if (tensor.ElementType == TensorSharp.DType.Float16)
        {
            ushort* p = TensorSharp.TensorComputePrimitives.GetHalfPointer(tensor);
            for (long i = 0; i < n; i++) result[i] = (float)BitConverter.UInt16BitsToHalf(p[i]);
        }
        else
        {
            float* p = TensorSharp.TensorComputePrimitives.GetFloatPointer(tensor);
            for (long i = 0; i < n; i++) result[i] = p[i];
        }
        return result;
    }

    // ---- 5. snapshot restore ---------------------------------------------------------------

    [CudaTheory]
    [InlineData("gptoss", "cuda")]
    [InlineData("qwen35moe", "cuda")]
    [InlineData("qwen35", "cuda")]
    [InlineData("gemma4moe", "cuda")]
    [InlineData("nemotron_h", "cuda")]
    [InlineData("nemotron_h_moe", "cuda")]
    [InlineData("gptoss", "cpu")]
    [InlineData("qwen35moe", "cpu")]
    [InlineData("qwen35", "cpu")]
    [InlineData("gemma4moe", "cpu")]
    [InlineData("nemotron_h", "cpu")]
    [InlineData("nemotron_h_moe", "cpu")]
    public void SnapshotRestore_ReproducesTheStraightRun(string family, string backend)
    {
        // The prefix block (K/V rows and, for the hybrids, the Mamba2/GDN state as of its end)
        // is extracted after a prefill and injected into an empty cache: the next forwards then
        // attend over and continue from exactly the same bytes, so the rows must be
        // bit-identical. Gemma-4 restores a pooled snapshot only within one sliding window
        // (MaxReusablePrefixTokens), so its prefix is 24 rows. The managed cpu rows are the
        // control. On direct CUDA this was D1: the inject writes the host mirror of the caches,
        // and the next forward's K/V append treated the cache as write-only, so the injected
        // rows never reached the device and attention ran over a zero prefix.
        string path = WriteModel(family);
        int prefixLength = family.StartsWith("gemma4") ? 24 : 48;
        int[] prompt = Prompt(prefixLength + 16);
        int[] forced = Forced(4);
        using var model = Load(path, backend == "cpu" ? BackendType.Cpu : BackendType.Cuda);
        // Warm-up: a model's first decode step runs per-op and later ones replay the captured
        // decode graph, which rounds differently (1.3e-7 on the dense Qwen3.5 file); without it
        // the straight run's first decode row would be per-op and the restored run's a replay.
        Trajectory(model, prompt, forced);
        SnapshotResult result = SnapshotRestore(model, prompt[..prefixLength], prompt[prefixLength..], forced);
        Assert.True(result != null, $"{family} on {backend}: the model refused to snapshot a {prefixLength}-token prefix");
        var rel = result.Restored.Select((row, i) => RelL2(row, result.Straight[i])).ToList();
        Report($"{family} restored vs straight ({result.BlockBytes} B block)", rel);
        AssertRows($"{family} on {backend}, after restoring the snapshot", result.Restored, result.Straight, _ => 0);
    }

    [CudaTheory]
    [InlineData("gptoss", "cuda")]
    [InlineData("qwen35moe", "cuda")]
    [InlineData("qwen35", "cuda")]
    [InlineData("gemma4moe", "cuda")]
    [InlineData("nemotron_h", "cuda")]
    [InlineData("nemotron_h_moe", "cuda")]
    [InlineData("qwen35moe", "cpu")]
    public void SnapshotBlock_RoundTripsThroughInject(string family, string backend)
    {
        // Narrows D1: a block injected into an empty cache and extracted again comes back byte
        // for byte on direct CUDA too, so what the cache STORES is right and D1 lives in what the
        // continuation reads after an inject (the device copies / host mirror the next forward
        // attends over), not in the block format or the copy into the cache.
        string path = WriteModel(family);
        int prefixLength = family.StartsWith("gemma4") ? 24 : 48;
        using var model = Load(path, backend == "cpu" ? BackendType.Cpu : BackendType.Cuda);
        model.ResetKVCache();
        model.ForwardPrompt(Prompt(prefixLength));
        Assert.True(model.SupportsKVStateSnapshot, $"{family} on {backend} does not snapshot");
        long size = model.ComputeKVBlockByteSize(prefixLength);
        var block = new byte[size];
        Assert.True(model.TryExtractKVBlock(0, prefixLength, block));
        Assert.Contains(block, b => b != 0);
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, prefixLength, block));
        var again = new byte[size];
        Assert.True(model.TryExtractKVBlock(0, prefixLength, again));
        int differing = block.Zip(again, (a, b) => a != b).Count(d => d);
        _output.WriteLine($"{family} on {backend}: {size} B block, {differing} bytes differ after inject + extract");
        Assert.Equal(0, differing);
    }

    [CudaTheory]
    [InlineData("replay-after-inject")]
    [InlineData("capture-after-inject")]
    [InlineData("capture-after-extract")]
    [InlineData("no-inject")]
    public void Qwen35PrefillGraph_AroundSnapshots_MatchesAFreshModel(string scenario)
    {
        // Direct-CUDA Qwen3.5 captures its per-op prefill on a key's second sighting and replays it
        // from the third, reading the K/V and GDN state buffers in place. So a replay right after a
        // snapshot inject must see the injected rows, and a capture must not record the upload of
        // a host-dirty buffer (left by an inject, or by an extract, which checks the state out to
        // the host): a recorded upload copies that stale host image over the state on every later
        // replay, and an unrelated request then continues from the snapshot's prefix. Every row
        // here is the 16-token suffix after a 48-token prefix (key 16|48), compared with a freshly
        // loaded model's plain run of the same prefix. The base (D1) was 0.11 off on the replay
        // after an inject and 7.7e-3 / 1.8e-2 off on the replays after a capture that followed an
        // extract / an inject; no-inject is the control.
        string path = WriteModel("qwen35");
        int[] p = Prompt(48), q = Prompt(48, salt: 3), s = Prompt(16, salt: 5);
        float[] Suffix(ModelBase m) => (float[])m.ForwardPrompt(s).Clone();
        float[] Row(ModelBase m, int[] prefix)
        {
            m.ResetKVCache();
            m.ForwardPrompt(prefix);
            return Suffix(m);
        }
        float[] refP, refQ;
        using (var fresh = Load(path, BackendType.Cuda)) refP = Row(fresh, p);
        using (var fresh = Load(path, BackendType.Cuda)) refQ = Row(fresh, q);
        Assert.False(BitIdentical(refP, refQ), "the two prefixes give the same suffix row: this file cannot tell them apart");

        using var model = Load(path, BackendType.Cuda);
        model.ResetKVCache();
        model.ForwardPrompt(p);
        var block = new byte[model.ComputeKVBlockByteSize(48)];
        Assert.True(model.TryExtractKVBlock(0, 48, block));
        var checks = new List<(string What, float[] Row, float[] Reference)> { ("run 1 (plain, after an extract): P", Suffix(model), refP) };
        switch (scenario)
        {
            case "replay-after-inject":
                checks.Add(("run 2 (capture): Q", Row(model, q), refQ));
                model.ResetKVCache();
                Assert.True(model.TryInjectKVBlock(0, 48, block));
                checks.Add(("run 3 (replay right after injecting P): P", Suffix(model), refP));
                break;
            case "capture-after-inject":
                model.ResetKVCache();
                Assert.True(model.TryInjectKVBlock(0, 48, block));
                checks.Add(("run 2 (capture right after injecting P): P", Suffix(model), refP));
                checks.Add(("run 3 (replay, Q prefilled normally): Q", Row(model, q), refQ));
                break;
            case "capture-after-extract":
                model.ResetKVCache();
                model.ForwardPrompt(p);
                Assert.True(model.TryExtractKVBlock(0, 48, block));
                checks.Add(("run 2 (capture right after extracting P): P", Suffix(model), refP));
                checks.Add(("run 3 (replay, Q prefilled normally): Q", Row(model, q), refQ));
                break;
            case "no-inject":
                checks.Add(("run 2 (capture): Q", Row(model, q), refQ));
                checks.Add(("run 3 (replay): P", Row(model, p), refP));
                checks.Add(("run 4 (replay): Q", Row(model, q), refQ));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(scenario));
        }
        foreach (var (what, row, reference) in checks)
            _output.WriteLine($"{scenario} {what} vs a fresh model: {RelL2(row, reference):E3} (bit-identical {BitIdentical(row, reference)})");
        foreach (var (what, row, reference) in checks)
            Assert.True(BitIdentical(row, reference), $"{scenario} {what} is {RelL2(row, reference):E3} from a fresh model's row");
    }

    [CudaTheory]
    [InlineData("qwen35")]
    [InlineData("qwen35moe")]
    public void Qwen35DecodeGraph_CapturedRightAfterInject_MatchesAFreshModel(string family)
    {
        // The decode-side twin of the prefill-graph scenarios above. Direct-CUDA Qwen3.5 captures
        // its decode layer loop on a key's second sighting and replays it after that, reading the
        // K/V and GDN state buffers in place. Here the capture is the first forward after a
        // snapshot inject (what a prefix-cache hit that matched all but the last prompt token does:
        // it feeds that token as a one-token forward), so the injected host rows must be uploaded
        // BEFORE the capture (EnsureDecodeGraphInputsResident in RunCudaDecodeLayerLoop): an upload
        // recorded inside the graph copies that host image over the state on every replay. The
        // review of the D1 fix removed only that call and nothing failed; this scenario then put
        // the rows 0.07-0.24 (dense) and 0.78-1.21 (MoE) from the fresh model's.
        string path = WriteModel(family);
        int[] prefix = Prompt(48);
        int[] forced = Forced(4);
        float[][] straight;
        using (var fresh = Load(path, BackendType.Cuda))
        {
            fresh.ResetKVCache();
            fresh.ForwardPrompt(prefix);
            straight = forced.Select(token => (float[])fresh.Forward(new[] { token }).Clone()).ToArray();
        }

        using var model = Load(path, BackendType.Cuda);
        model.ResetKVCache();
        model.ForwardPrompt(prefix);
        var block = new byte[model.ComputeKVBlockByteSize(prefix.Length)];
        Assert.True(model.TryExtractKVBlock(0, prefix.Length, block));
        // The decode key's first sighting (a plain per-op step), as in the fresh model.
        float[] first = (float[])model.Forward(new[] { forced[0] }).Clone();
        Assert.True(BitIdentical(first, straight[0]), $"{family}: the step before the inject is {RelL2(first, straight[0]):E3} from the fresh model's");
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, prefix.Length, block));
        // Second sighting, right after the inject: row 0 is the CAPTURE, rows 1.. replay it.
        float[][] restored = forced.Select(token => (float[])model.Forward(new[] { token }).Clone()).ToArray();
        var rel = restored.Select((row, i) => RelL2(row, straight[i])).ToList();
        Report($"{family} decode after inject (row 0 captured, then replays) vs a fresh model", rel);
        // Row 0 is the captured graph against the fresh model's per-op step: the same kernels,
        // measured 1.3e-7 apart on the dense file (the per-op vs graph rounding the snapshot test's
        // warm-up avoids) and bit-identical on MoE. Every replayed row is bit-identical.
        AssertRows($"{family} decode captured right after an inject", restored, straight, row => row == 0 ? 1e-6 : 0);
    }

    // ---- 5b. host reads of the K/V cache ----------------------------------------------------

    /// <summary>Families whose direct-CUDA decode attention runs on the host over the K/V cache
    /// (AttentionDecodePureCS and its relatives: Nemotron-H, Mistral 3, Hunyuan dense,
    /// Muse-Glimmer), and gpt-oss, whose prefix-cache block capture (TryExtractKVBlock) reads the
    /// cache on the host.</summary>
    private static readonly Dictionary<string, Func<string, string>> HostReadFiles = new()
    {
        ["gptoss"] = p => GptOssSyntheticModelBuilder.SensitiveSinks().Write(p),
        ["nemotron_h"] = p => Nemotron("SFAFSF").Write(p),
        ["nemotron_h_moe"] = p => Nemotron("SMAMSM").Write(p),
        ["mistral3"] = p => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3" }.Write(p),
        ["hunyuan"] = p => new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense",
            IncludeMistralControlTokens = false, IncludeQkNorms = true, RopeBase = 10000f,
        }.Write(p),
        ["museglimmer"] = p => new MuseGlimmerSyntheticModelBuilder().Write(p),
    };

    /// <summary>How many of the model's direct-CUDA K/V cache storages (_kvCacheK/_kvCacheV, by
    /// reflection) hold host writes not yet uploaded.</summary>
    private static (int Dirty, int Count) HostDirtyKvCaches(ModelBase model)
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance
            | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.DeclaredOnly;
        int dirty = 0, count = 0;
        foreach (string name in new[] { "_kvCacheK", "_kvCacheV" })
        {
            System.Reflection.FieldInfo field = null;
            for (Type type = model.GetType(); type != null && field == null; type = type.BaseType)
                field = type.GetField(name, flags);
            if (field?.GetValue(model) is not TensorSharp.Tensor[] caches)
                continue;
            foreach (TensorSharp.Tensor cache in caches)
            {
                if (cache?.Storage is not TensorSharp.Cuda.CudaStorage storage)
                    continue;
                count++;
                if (storage.IsHostDirty)
                    dirty++;
            }
        }
        return (dirty, count);
    }

    [CudaTheory]
    [InlineData("gptoss")]
    [InlineData("nemotron_h")]
    [InlineData("nemotron_h_moe")]
    [InlineData("mistral3")]
    [InlineData("hunyuan")]
    [InlineData("museglimmer")]
    public void HostReadsOfTheKvCache_DoNotQueueItForUpload(string family)
    {
        // The D1 fix makes the K/V append upload a host-dirty cache before it writes its rows. A
        // host read that checks the cache out as if it were a write (PtrAtElement) then costs a
        // full-cache upload on the next append: on every decode step for the families whose
        // decode attention runs on the host (the D1 review measured 9.7 KB -> 1.06 MB uploaded
        // per step on nemotron_h at 2K of cache, 64 MiB and +5-6 ms per step with 8 KV heads at
        // 8K), and after every prefix-cache block capture on gpt-oss (292 B -> 4.19 MB). So after
        // each decode step, and after a block extract, no K/V cache may be host-dirty; and the
        // extract must not change what the next steps compute.
        string path = HostReadFiles[family](Path.Combine(_directory, family + ".gguf"));
        int[] prompt = Prompt(64);
        int[] forced = Forced(4);
        List<float[]> straight;
        using (var reference = Load(path, BackendType.Cuda))
            straight = Trajectory(reference, prompt, forced);

        using var model = Load(path, BackendType.Cuda);
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardPrompt(prompt).Clone() };
        var states = new List<(string What, int Dirty, int Count)>();
        bool extracted = false;
        for (int step = 0; step < forced.Length; step++)
        {
            if (step == 2 && model.SupportsKVStateSnapshot)
            {
                var block = new byte[model.ComputeKVBlockByteSize(48)];
                extracted = model.TryExtractKVBlock(0, 48, block);
                var afterExtract = HostDirtyKvCaches(model);
                states.Add(("after extracting a 48-token block", afterExtract.Dirty, afterExtract.Count));
            }
            rows.Add((float[])model.Forward(new[] { forced[step] }).Clone());
            var afterStep = HostDirtyKvCaches(model);
            states.Add(($"after decode step {step}", afterStep.Dirty, afterStep.Count));
        }
        foreach (var (what, dirty, count) in states)
            _output.WriteLine($"{family} {what}: {dirty} of {count} K/V cache storages host-dirty");
        _output.WriteLine($"{family}: block extract {(extracted ? "ran" : "not supported")}");
        Assert.All(states, s => Assert.True(s.Count > 0, $"{family}: no direct-CUDA K/V cache found by reflection"));
        Assert.All(states, s => Assert.True(s.Dirty == 0, $"{family} {s.What}: {s.Dirty} of {s.Count} K/V cache storages are queued for a full upload"));
        AssertRows($"{family} decode with an extract in between", rows, straight, _ => 0);
    }

    // ---- 6. verify windows -----------------------------------------------------------------

    /// <summary>Verify row vs the same token decoded alone (0 = bit-identical, as the base is).</summary>
    private static readonly Dictionary<string, double> VerifyTolerance = new()
    {
        ["qwen35moe"] = 0.010,          // measured 4.5e-3..6.7e-3
        ["qwen35"] = 0.0037,            // 1.9e-3..2.4e-3 (BF16 GEMM rows vs BF16 matvec rows)
        ["gemma4moe_global"] = 0.0087,  // 9e-8..5.8e-3, against the DEFAULT (graph) decode
        ["gemma4moe"] = 0.017,          // past the window: 3.3e-3..1.1e-2 (cuda's decode row 11 is 9.4e-3 from ggml_cpu)
        ["nemotron_h"] = 1e-6,          // 7e-8..9.5e-8: the same arithmetic per row
        ["nemotron_h_moe"] = 1e-6,
        ["nemotron_h_q4k"] = 0.023,      // 6.2e-3..1.5e-2: Q4_K GEMM rows vs Q4_K matvec rows
        ["nemotron_h_moe_mx"] = 0.0016,  // 8.5e-8..1.0e-7, one row 1.0e-3
    };

    [CudaTheory]
    [InlineData("qwen35moe")]
    [InlineData("qwen35")]
    [InlineData("gemma4moe_global")]
    [InlineData("gemma4moe")]
    [InlineData("nemotron_h")]
    [InlineData("nemotron_h_moe")]
    [InlineData("nemotron_h_q4k")]
    [InlineData("nemotron_h_moe_mx")]
    public void VerifyWindow16_MatchesSequentialDecode(string family)
    {
        // A speculative verify forwards 16 rows at once (SpecForward, every row's logits); each
        // row must be what decoding that token alone, through the default decode path, gives.
        // The plan keeps verify windows on their current kernels below its prefill row
        // thresholds, which these rows pin. Sliding-window Gemma: past the window (D4 put verify
        // and decode rows ~6% apart; it was D3's two causes, now fixed).
        string path = WriteModel(family);
        int[] prompt = Prompt(64);
        using var cuda = Load(path, BackendType.Cuda);
        VerifyResult result = VerifyWindow(cuda, prompt[..48], prompt[48..64]);
        Assert.True(result.AllRows, $"{family} is not a speculative target");
        AssertFinite(result.Window, family);
        var rel = result.Window.Select((row, i) => RelL2(row, result.Sequential[i])).ToList();
        int bitIdentical = result.Window.Select((row, i) => BitIdentical(row, result.Sequential[i])).Count(x => x);
        Report($"{family} verify rows vs sequential decode (bit-identical {bitIdentical}/{rel.Count})", rel);
        AssertRows($"{family} verify", result.Window, result.Sequential, _ => VerifyTolerance[family]);
    }

    [CudaFact]
    public void GptOss_LastRowOf16TokenForward_MatchesSequentialDecode()
    {
        // gpt-oss has no speculative path, so this is not a verify window: after a 48-token
        // prefill, the logits row a 16-token Forward returns (prefill-shaped attention over the
        // sliding window and the sinks) against the 16 tokens decoded one at a time. The base is
        // bit-identical, which makes this the most sensitive gpt-oss attention check (the P0
        // review's off-by-one window and dropped-sink mutations both failed it).
        string path = WriteModel("gptoss");
        int[] prompt = Prompt(64);
        using var cuda = Load(path, BackendType.Cuda);
        VerifyResult result = VerifyWindow(cuda, prompt[..48], prompt[48..64]);
        Assert.False(result.AllRows);
        _output.WriteLine($"gptoss last row of a 16-token forward vs decode: {RelL2(result.Window[0], result.Sequential[0]):E3}");
        AssertRows("gptoss 16-token forward", result.Window, result.Sequential, _ => 0);
    }

    // ---- 7. continuous batching: concurrent against solo -----------------------------------

    /// <summary>Files whose greedy continuation depends on the history (the sliding-window Gemma
    /// and the tied dense Qwen files repeat their last token: the P0 review found the Gemma row
    /// could not fail). gemma4moe here is the all-global context-sensitive file.</summary>
    private static readonly Dictionary<string, Func<string, string>> ContextFiles = new()
    {
        ["gptoss"] = p => GptOssSyntheticModelBuilder.SensitiveSinks().Write(p),
        ["qwen35moe"] = p => new Qwen35MoeSyntheticModelBuilder().Write(p),
        ["qwen35"] = p => Qwen35DenseSyntheticModelBuilder.ContextSensitive().Write(p),
        ["gemma4moe"] = p => Gemma4MoeSyntheticModelBuilder.ContextSensitive().Write(p),
        ["nemotron_h"] = p => Nemotron("SFAFSF").Write(p),
        ["nemotron_h_moe"] = p => Nemotron("SMAMSM").Write(p),
    };

    [CudaTheory]
    [InlineData("gptoss", "cuda")]
    [InlineData("qwen35moe", "cuda")]
    [InlineData("qwen35", "cuda")]
    [InlineData("gemma4moe", "cuda")]
    [InlineData("nemotron_h", "cuda")]
    [InlineData("nemotron_h_moe", "cuda")]
    [InlineData("qwen35moe", "cpu")]
    [InlineData("qwen35", "cpu")]
    public async Task ConcurrentEngineStreams_MatchSolo(string family, string backend)
    {
        // Three prompts of different lengths through the continuous-batching engine, first one
        // at a time, then together (batched prefill/decode where the family has a batched path,
        // per-sequence cache swaps otherwise). A concurrent stream may leave its solo twin only
        // at a near-tie: where the two tokens' logits under the solo history are within 10% of
        // the row's RMS (measured near-ties: 0.03%-7% of the RMS; a sequence reading another's
        // state lands several RMS away: D2, Qwen3.5 on direct CUDA, where every sequence of a batch
        // shared one device GDN conv ring per layer). The managed cpu rows are the control.
        // Precondition: the file's next-token argmax must move with the history, or a stream
        // that read another sequence's state would still emit the same tokens.
        string path = ContextFiles[family](Path.Combine(_directory, family + "-context.gguf"));
        int[][] prompts = { Prompt(40, 1), Prompt(23, 2), Prompt(57, 3) };
        const int tokens = 24;
        using var model = Load(path, backend == "cpu" ? BackendType.Cpu : BackendType.Cuda);
        int sensitive = ContextSensitivity(model, out string detail);
        _output.WriteLine($"{family}: {detail}");
        Assert.True(sensitive >= 3, $"{family}: the next-token argmax moved with the history in only {sensitive}/12 probes: this file cannot detect a cross-sequence read");
        EngineStreams streams = await SoloAndConcurrent(model, prompts, tokens);
        // A stream that stops early sampled end-of-sequence (not emitted): compare it as that token.
        int eos = model.Tokenizer.EosTokenIds.FirstOrDefault(2);
        static int At(List<int> s, int k, int stop) => k < s.Count ? s[k] : stop;
        for (int i = 0; i < prompts.Length; i++)
        {
            List<int> solo = streams.Solo[i], concurrent = streams.Concurrent[i];
            Assert.True(solo.Count >= 8 && solo.Distinct().Count() > 1, $"{family} prompt {i}: the solo stream [{string.Join(",", solo)}] is too short or repeats one token");
            int length = Math.Max(solo.Count, concurrent.Count);
            int diff = Enumerable.Range(0, length).FirstOrDefault(k => At(solo, k, eos) != At(concurrent, k, eos), -1);
            _output.WriteLine($"{family} prompt {i}: solo [{string.Join(",", solo)}] ({solo.Count}), concurrent ({concurrent.Count}) first difference {diff}");
            if (diff < 0)
                continue;
            float[] logits = LogitsBefore(model, prompts[i], solo, diff);
            double gap = logits[At(solo, diff, eos)] - logits[At(concurrent, diff, eos)];
            double rms = Rms(logits);
            _output.WriteLine($"  at step {diff}: solo-token minus concurrent-token logit {gap:F4}, row RMS {rms:F4}");
            Assert.True(Math.Abs(gap) <= 0.10 * rms,
                $"{family} prompt {i}: the concurrent stream leaves the solo one at step {diff} by {gap:F4} logits (RMS {rms:F4}): not a near-tie");
        }
    }

    // ---- 7b. the batched forward itself: logits rows, below the scheduler --------------------

    /// <summary>A sequence in a 3-sequence ForwardBatch against the same sequence alone (N=1
    /// ForwardBatch from a reset cache), per family: the worst row's relative L2. What remains on
    /// direct CUDA is rounding: the batched matmuls take other routes at other row counts (expert
    /// GEMM rows at M=1 vs M=2..8 differ by 1e-7 and grow through the layers; measured 2.3e-3..3.2e-3
    /// dense, 5e-3..7e-3 MoE); D2 (a shared device conv ring) put them 0.23-1.33 away.</summary>
    private static readonly Dictionary<string, double> BatchedVsSolo = new()
    {
        ["qwen35"] = 0.0048,
        ["qwen35moe"] = 0.011,
    };

    [CudaTheory]
    [InlineData("qwen35moe", "cuda")]
    [InlineData("qwen35", "cuda")]
    [InlineData("qwen35moe", "cpu")]
    [InlineData("qwen35", "cpu")]
    public void ForwardBatch_SequencesKeepTheirOwnState(string family, string backend)
    {
        // Below the scheduler: three prompts (40, 23 and 57 tokens) prefilled in one ForwardBatch,
        // then 8 decode batches, teacher-forced, against each prompt alone. Then each prompt alone
        // AGAIN without a reset (the engine never resets between requests: a new sequence reuses a
        // freed slot, which must start from zero state): that must equal the first solo run bit for
        // bit. D2 failed both on direct CUDA: the Gated DeltaNet conv ring lived in one device
        // tensor per layer that the batched path never swapped per sequence, so sequences read each
        // other's last qkv rows and a new sequence read the previous one's tail.
        string path = ContextFiles[family](Path.Combine(_directory, family + "-context.gguf"));
        int[][] prompts = { Prompt(40, 1), Prompt(23, 2), Prompt(57, 3) };
        int[][] forced = { Forced(8, 1), Forced(8, 2), Forced(8, 3) };
        using var model = Load(path, backend == "cpu" ? BackendType.Cpu : BackendType.Cuda);
        var solo = new List<float[]>[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
        {
            model.ResetKVCache();
            solo[i] = BatchedTrajectories(model, new[] { prompts[i] }, new[] { forced[i] })[0];
        }
        model.ResetKVCache();
        List<float[]>[] together = BatchedTrajectories(model, prompts, forced);
        var soloAgain = new List<float[]>[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
            soloAgain[i] = BatchedTrajectories(model, new[] { prompts[i] }, new[] { forced[i] })[0];
        double tolerance = backend == "cpu" ? 1e-5 : BatchedVsSolo[family];
        for (int i = 0; i < prompts.Length; i++)
        {
            AssertFinite(together[i], $"{family} sequence {i}");
            Report($"{family} on {backend}, sequence {i} in the batch vs alone", together[i].Select((row, r) => RelL2(row, solo[i][r])));
            Report($"{family} on {backend}, sequence {i} alone after the batch (no reset) vs alone", soloAgain[i].Select((row, r) => RelL2(row, solo[i][r])));
        }
        for (int i = 0; i < prompts.Length; i++)
        {
            AssertRows($"{family} on {backend}, sequence {i} alone again without a reset", soloAgain[i], solo[i], _ => 0);
            AssertRows($"{family} on {backend}, sequence {i} in a 3-sequence batch", together[i], solo[i], _ => tolerance);
        }
    }

    [CudaTheory]
    [InlineData("cuda")]
    [InlineData("cpu")]
    public void ForwardBatch_NeighbourContent_DoesNotReachTheMiddleSequence(string backend)
    {
        // Shapes held fixed: the middle sequence of batch [p0, p1, p2] against batch [q0, p1, q2],
        // where q0 and q2 have p0's and p2's lengths but other tokens. Every GEMM and attention
        // shape is the same in both batches, so on the dense file the middle sequence's rows must
        // be bit-identical; any difference is a read of a neighbour's state (D2 moved them 0.43).
        string path = ContextFiles["qwen35"](Path.Combine(_directory, "qwen35-context.gguf"));
        int[][] a = { Prompt(40, 1), Prompt(23, 2), Prompt(57, 3) };
        int[][] b = { Prompt(40, 11), Prompt(23, 2), Prompt(57, 13) };
        int[][] fa = { Forced(8, 1), Forced(8, 2), Forced(8, 3) };
        int[][] fb = { Forced(8, 21), Forced(8, 2), Forced(8, 23) };
        using var model = Load(path, backend == "cpu" ? BackendType.Cpu : BackendType.Cuda);
        model.ResetKVCache();
        List<float[]>[] withA = BatchedTrajectories(model, a, fa);
        model.ResetKVCache();
        List<float[]>[] withB = BatchedTrajectories(model, b, fb);
        Assert.False(BitIdentical(withA[0][^1], withB[0][^1]), "the neighbours' own rows did not change: the swap tests nothing");
        Report($"qwen35 on {backend}: middle sequence with neighbours A vs B", withA[1].Select((row, r) => RelL2(row, withB[1][r])));
        AssertRows($"qwen35 on {backend}: the middle sequence with other neighbours", withB[1], withA[1], _ => 0);
    }

    /// <summary>Of 12 probes, how often the next-token argmax changes when the history changes and
    /// the last token does not.</summary>
    private static int ContextSensitivity(ModelBase model, out string detail)
    {
        int differ = 0;
        var rels = new List<double>();
        for (int salt = 1; salt <= 12; salt++)
        {
            int[] a = Prompt(40, salt), b = Prompt(40, salt + 50);
            b[^1] = a[^1];
            model.ResetKVCache();
            float[] la = (float[])model.ForwardPrompt(a).Clone();
            model.ResetKVCache();
            float[] lb = (float[])model.ForwardPrompt(b).Clone();
            if (ArgMax(la) != ArgMax(lb)) differ++;
            rels.Add(RelL2(lb, la));
        }
        detail = $"argmax moved with the history in {differ}/12 probes (logits relative L2 {Fmt(rels)})";
        return differ;
    }
}
