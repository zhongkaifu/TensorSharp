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
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Muse-Glimmer on the shared decoder runtime (<see cref="MuseGlimmerRuntimeModel"/>, stage S2 of
/// docs/design/muse-glimmer-unified.md), reached through its test seam (<see cref="MuseGlimmerArchitecture.CreateRuntimeModel"/>),
/// against the legacy model's logits frozen per lane on the synthetic fixtures (<see cref="MuseGlimmerGoldens"/>: a decode, uneven
/// prompt steps, a long prompt that wraps the rings and grows the full layers' caches, a decode across the growth, a rewind;
/// a tied head), on every executor, as programs and per op.
///
/// The bound (<see cref="ParityBound"/>) is the legacy model's own spread between its lanes on the same fixtures: the fused
/// graphs (ggml_cpu, ggml_metal, ggml_cuda, ggml_vulkan) against the per-op paths (managed, MLX, direct CUDA), measured at the
/// freezes (<see cref="MuseGlimmerFrozenGoldensTests.TheFrozenLanesAgreeWithinTheBound"/>: 1.37e-3 over the Mac's four lanes,
/// 3.29e-3 with the RTX PRO 4500 box's, direct CUDA against ggml_vulkan). Bit for bit is not expected: the
/// runtime attends rings of the window plus 512 rows in prompt steps of at most 512 rows, where legacy reads 2048-row chunks
/// over a 2304-row ring (ggml_metal) or uniform caches, and pads decode windows differently, so the attention's reductions run
/// over other key counts in another order.
/// </summary>
public sealed class MuseGlimmerRuntimeTests(ITestOutputHelper output) : IDisposable
{
    /// <summary>
    /// The bound on every output's relative L2 distance and largest difference over the largest frozen value, against legacy's
    /// logits frozen on the same lane: legacy's own worst spread between its fused and per-op lanes on these fixtures (3.29e-3
    /// once the GPU box's lanes were frozen, rounded up to 4e-3; it was 2e-3 over the Mac's lanes alone, 1.37e-3). On the GPU
    /// box the runtime measured 1.4e-3 to 3.5e-3 from legacy on ggml_cuda and ggml_vulkan, which is the GPU lanes' own noise:
    /// legacy ggml_cuda is 2.8e-3 from the managed reference and the runtime there within 2.8e-3 of it too. Off the frozen lanes
    /// (a backend the freeze has not covered) the cross-lane bound applies.
    /// </summary>
    internal const double ParityBound = 4e-3;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-mg-runtime-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    public static TheoryData<string, string> CasesAndPaths()
    {
        var data = new TheoryData<string, string>();
        foreach (var c in MuseGlimmerGoldens.Cases)
        {
            data.Add(c.Name, "graphs");
            data.Add(c.Name, "per-op");
        }
        return data;
    }

    /// <summary>Runs one case on the runtime family, its steps as programs or per op (held to the path asked for), and returns
    /// its fixture's SHA-256 and its outputs.</summary>
    internal static (string FixtureSha256, List<float[]> Outputs) RunRuntime(MuseGlimmerGoldens.CaseSpec spec, BackendType backend,
        bool graphs, string dir)
    {
        Directory.CreateDirectory(dir);
        string path = MuseGlimmerGoldens.WriteFixture(spec.Fixture, dir);
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(MuseGlimmerGoldens.CaseKvDtype);
        using ModelBase model = MuseGlimmerArchitecture.CreateRuntimeModel(path, backend);
        var decoder = Assert.IsType<MuseGlimmerRuntimeModel>(model);
        decoder.GraphLoweringEnabled = graphs;
        List<float[]> outputs = spec.Script(model);
        if (graphs) Assert.True(decoder.GraphRuns > 0, $"{spec.Name} on {backend}: no step ran as a program ({decoder.GraphRefusalForTests})");
        else Assert.Equal(0, decoder.GraphRuns);
        return (MuseGlimmerGoldens.FileSha(path), outputs);
    }

    private void Verify(BackendType backend, string caseName, string path)
    {
        var spec = MuseGlimmerGoldens.Cases.Single(c => c.Name == caseName);
        string lane = MuseGlimmerGoldens.LaneKey(backend);
        var (sha, outputs) = RunRuntime(spec, backend, path == "graphs", _dir);
        MuseGlimmerGoldens.Case frozen = MuseGlimmerGoldens.Find(caseName, lane);
        bool sameLane = frozen != null && frozen.Lane == lane;
        output.WriteLine($"[{path}] " + MuseGlimmerGoldens.Check(frozen, caseName, sha, outputs, bytes: false,
            sameLane ? ParityBound : MuseGlimmerFrozenGoldensTests.CrossLaneBound));
        // For the record: the distance from every lane legacy was frozen on (the worst output's relL2 and largest difference).
        foreach (string other in MuseGlimmerGoldens.Lanes.OrderBy(l => l, StringComparer.Ordinal))
        {
            var o = MuseGlimmerGoldens.CasesOf(other)[caseName].Outputs;
            double l2 = 0, max = 0;
            for (int i = 0; i < outputs.Count; i++)
            {
                var (relL2, relMax) = MuseGlimmerGoldens.Distance(o[i].Values, outputs[i]);
                l2 = Math.Max(l2, relL2);
                max = Math.Max(max, relMax);
            }
            output.WriteLine($"  vs legacy on {other.Split('|')[0]}: worst relL2 {l2:E3}, largest difference {max:E3}");
        }
        // The argmax is legacy's at every output (the fixtures' top logits are far apart at the bound).
        for (int i = 0; i < outputs.Count; i++)
            Assert.Equal(MuseGlimmerParity.ArgMax(frozen.Outputs[i].Values, frozen.Outputs[i].Values.Length),
                MuseGlimmerParity.ArgMax(outputs[i], outputs[i].Length));
    }

    [Theory]
    [MemberData(nameof(CasesAndPaths))]
    public void RuntimeOnTheManagedBackendGivesTheFrozenLegacyLogits(string caseName, string path) => Verify(BackendType.Cpu, caseName, path);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(CasesAndPaths))]
    public void RuntimeOnGgmlCpuGivesTheFrozenLegacyLogits(string caseName, string path) => Verify(BackendType.GgmlCpu, caseName, path);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(CasesAndPaths))]
    public void RuntimeOnGgmlMetalGivesTheFrozenLegacyLogits(string caseName, string path) => Verify(BackendType.GgmlMetal, caseName, path);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(CasesAndPaths))]
    public void RuntimeOnGgmlCudaGivesTheFrozenLegacyLogits(string caseName, string path) => Verify(BackendType.GgmlCuda, caseName, path);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [MemberData(nameof(CasesAndPaths))]
    public void RuntimeOnGgmlVulkanGivesTheFrozenLegacyLogits(string caseName, string path) => Verify(BackendType.GgmlVulkan, caseName, path);

    [MlxTheory]
    [MemberData(nameof(CasesAndPaths))]
    public void RuntimeOnMlxGivesTheFrozenLegacyLogits(string caseName, string path) => Verify(BackendType.Mlx, caseName, path);

    [CudaTheory]
    [MemberData(nameof(CasesAndPaths))]
    public void RuntimeOnDirectCudaGivesTheFrozenLegacyLogits(string caseName, string path) => Verify(BackendType.Cuda, caseName, path);

    /// <summary><see cref="ParityBound"/> is legacy's own spread: no smaller than the worst distance between legacy's frozen
    /// lanes on these fixtures (its fused graphs against its per-op paths), and at most one and a half times it.</summary>
    [Fact]
    public void TheParityBoundIsLegacysOwnSpread()
    {
        string[] lanes = MuseGlimmerGoldens.Lanes.OrderBy(l => l, StringComparer.Ordinal).ToArray();
        Assert.True(lanes.Length >= 2, "at least two frozen lanes are needed to measure legacy's spread");
        double worst = 0;
        string where = "";
        foreach (var spec in MuseGlimmerGoldens.Cases)
            for (int a = 0; a < lanes.Length; a++)
                for (int b = a + 1; b < lanes.Length; b++)
                {
                    var x = MuseGlimmerGoldens.CasesOf(lanes[a])[spec.Name].Outputs;
                    var y = MuseGlimmerGoldens.CasesOf(lanes[b])[spec.Name].Outputs;
                    for (int i = 0; i < x.Length; i++)
                    {
                        var (relL2, relMax) = MuseGlimmerGoldens.Distance(x[i].Values, y[i].Values);
                        double d = Math.Max(relL2, relMax);
                        if (d > worst) (worst, where) = (d, $"{spec.Name} output {i}, {lanes[a].Split('|')[0]} vs {lanes[b].Split('|')[0]}");
                    }
                }
        output.WriteLine($"legacy's worst spread between its lanes: {worst:E3} ({where}); the runtime's same-lane bound {ParityBound:E1}");
        Assert.InRange(ParityBound, worst, 1.5 * worst);
    }

    // ---------------------------------------------------------------- the family's declarations

    /// <summary>The 30B's layer pattern from the scalar period (39 sliding-window layers, every fourth layer full and NoPE),
    /// the fixture's from its own: the sliding-window layers keep rings of the window plus a 512-row step (aligned to 256),
    /// the full layers linear caches; the runtime's KV fingerprint names the family's pattern and post-norm epsilon.</summary>
    [Fact]
    public void TheSlidingWindowLayersKeepRingsAndTheFullLayersLinearCaches()
    {
        Directory.CreateDirectory(_dir);
        string path = MuseGlimmerGoldens.WriteFixture("untied", _dir);
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        using var model = (MuseGlimmerRuntimeModel)MuseGlimmerArchitecture.CreateRuntimeModel(path, BackendType.Cpu);
        int window = MuseGlimmerGoldens.Builder("untied").SlidingWindow;
        int ring = (window + StandardDecoderModel.RingStepRows + StandardDecoderModel.RingAlignment - 1)
            / StandardDecoderModel.RingAlignment * StandardDecoderModel.RingAlignment;
        for (int l = 0; l < model.Config.NumLayers; l++)
        {
            bool swa = l % 4 < 3;
            Assert.Equal(swa, model.IsSlidingWindowLayerForTests(l));
            Assert.Equal(swa ? ring : 0, model.RingCellsForTests(l));
        }
        Assert.Equal(StandardDecoderModel.RingStepRows, model.MaxSequenceStepRows);
        Assert.Null(model.GraphRefusalForTests);
        Assert.Contains("|muse-glimmer|swa=64:11101110|", model.KVStateFingerprint, StringComparison.Ordinal);
        Assert.Contains("post=1E-08", model.KVStateFingerprint, StringComparison.Ordinal);
    }

    /// <summary>Tensor parallelism (per-rank programs: one tensor-parallel stage always runs them; the output gate's rows
    /// sharded with the query heads, the Q/K norms replicated), whole-layer stages (programs and per op) and their hybrid (whose
    /// tensor-parallel stages run per op) on simulated devices give the one-device logits of the decode and ring-wrap cases.</summary>
    [Theory]
    [InlineData(2, 1, true)]
    [InlineData(1, 2, true)]
    [InlineData(1, 2, false)]
    [InlineData(2, 2, false)]
    public void PlacementsOnSimulatedDevicesGiveTheOneDeviceLogits(int tp, int stages, bool graphs)
    {
        Directory.CreateDirectory(_dir);
        foreach (string caseName in new[] { "untied/decode", "untied/ring-wrap" })
        {
            var spec = MuseGlimmerGoldens.Cases.Single(c => c.Name == caseName);
            string path = MuseGlimmerGoldens.WriteFixture(spec.Fixture, _dir);
            List<float[]> Run(Func<ModelBase> create)
            {
                using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(MuseGlimmerGoldens.CaseKvDtype);
                using ModelBase model = create();
                ((StandardDecoderModel)model).GraphLoweringEnabled = graphs;
                return spec.Script(model);
            }
            List<float[]> one = Run(() => MuseGlimmerArchitecture.CreateRuntimeModel(path, BackendType.Cpu));
            List<float[]> placed = Run(() => MuseGlimmerArchitecture.CreateRuntimeModel(path, BackendType.Cpu, tp,
                tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null, stages));
            double worst = 0;
            for (int i = 0; i < one.Count; i++)
                worst = Math.Max(worst, MuseGlimmerGoldens.Distance(one[i], placed[i]).RelL2);
            output.WriteLine($"{caseName}: tp {tp} x stages {stages} ({(graphs ? "graphs" : "per-op")}): worst relL2 {worst:E2} over {one.Count} outputs");
            Assert.True(worst <= 1e-4, $"{caseName}: tp {tp} x stages {stages}: relL2 {worst:E2}");
        }
    }

    /// <summary>
    /// Tensor parallelism over two GPUs as per-rank graphs (ggml_cuda; ggml_vulkan since design D4) and whole-layer stages on two
    /// GPUs give the one-GPU logits of the decode and ring-wrap cases on the same backend. Each rank holds one of the fixture's
    /// two KV heads, as a rank of the 30B does under --tp 2, so the ring-wrap case reads wrapped rings whole with one KV head a
    /// rank (G12: legacy turned its rings off under --tp after ggml-cuda misread that shape). Where the ranks' sums cross the
    /// host (ggml-vulkan) the placement takes 2048-row prompt steps (StandardDecoderModel.RingStepRowsForPlacement), and the
    /// one-GPU reference runs at the same width.
    /// </summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData(2, 1)]
    [InlineData(1, 2)]
    public void PlacementsOnTwoGgmlCudaGpusGiveTheOneGpuLogits(int tp, int stages) => PlacementsOnTwoGpus(BackendType.GgmlCuda, tp, stages);

    /// <summary>The same on ggml_vulkan.</summary>
    [GgmlTheory(BackendType.GgmlVulkan, Devices = 2)]
    [InlineData(2, 1)]
    [InlineData(1, 2)]
    public void PlacementsOnTwoGgmlVulkanGpusGiveTheOneGpuLogits(int tp, int stages) => PlacementsOnTwoGpus(BackendType.GgmlVulkan, tp, stages);

    /// <summary>The bound of <see cref="PlacementsOnTwoGpus"/> (relative L2 and largest difference over the largest logit):
    /// 1.5x legacy's own --tp 2 spread from its one-GPU run on these cases and GPUs (2x RTX PRO 4500, ggml d9fe7bff), since
    /// ggml's kernels split each row-parallel product per rank in both families. Legacy's worst: 4.6e-3 on ggml_cuda (the
    /// ring-wrap cases; its device sums), 3.0e-3 on ggml_vulkan (host-staged F32 sums). The runtime measured 5.2e-3 and
    /// 2.6e-3 there; whole-layer stages gave the one-GPU bytes on both.</summary>
    internal static double TwoGpuPlacementBound(BackendType backend) => backend == BackendType.GgmlVulkan ? 4.5e-3 : 7e-3;

    private void PlacementsOnTwoGpus(BackendType backend, int tp, int stages)
    {
        Directory.CreateDirectory(_dir);
        var problems = new List<string>();
        foreach (string caseName in new[] { "untied/decode", "untied/ring-wrap", "tied/ring-wrap" })
        {
            var spec = MuseGlimmerGoldens.Cases.Single(c => c.Name == caseName);
            string path = MuseGlimmerGoldens.WriteFixture(spec.Fixture, _dir);
            int placedStep = 0;
            List<float[]> Run(int t, int s, int? stepRows = null)
            {
                using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(MuseGlimmerGoldens.CaseKvDtype);
                StandardDecoderModel.RingGeometryForTests.Value = stepRows is int rows ? (rows, StandardDecoderModel.RingAlignment) : null;
                try
                {
                    using ModelBase model = MuseGlimmerArchitecture.CreateRuntimeModel(path, backend, t, null, s);
                    var decoder = (StandardDecoderModel)model;
                    Assert.Equal(t, decoder.Mesh.TensorParallel);
                    Assert.Equal(s, decoder.Mesh.Stages);
                    if (stepRows == null && t > 1)
                    {
                        // The placement picks the prompt step: 2048 rows where the ranks' sums cross the host (ggml-vulkan's
                        // collective goes through host memory), the runtime's 512 where they stay on the devices.
                        if (backend == BackendType.GgmlVulkan) Assert.True(decoder.RankSumsCrossHost, $"{caseName}: ggml-vulkan's ranks sum on the host");
                        Assert.Equal(decoder.RankSumsCrossHost ? StandardDecoderModel.HostStagedRingStepRows : StandardDecoderModel.RingStepRows,
                            decoder.RingStepRowsInUse);
                    }
                    else if (stepRows == null)
                    {
                        Assert.False(decoder.RankSumsCrossHost);
                        Assert.Equal(StandardDecoderModel.RingStepRows, decoder.RingStepRowsInUse);
                    }
                    placedStep = decoder.RingStepRowsInUse;
                    List<float[]> outputs = spec.Script(model);
                    // Every step ran as graphs: per-rank graphs under tensor parallelism (its per-op path runs no program).
                    Assert.True(decoder.GraphRuns > 0 && decoder.GraphReplays > 0,
                        $"{caseName} on {backend}, tp {t} x stages {s}: the steps did not run and replay as graphs ({decoder.GraphRefusalForTests})");
                    return outputs;
                }
                finally
                {
                    StandardDecoderModel.RingGeometryForTests.Value = null;
                }
            }
            List<float[]> placed = Run(tp, stages);
            // One GPU at the placed model's step width, so the comparison isolates the placement's arithmetic.
            List<float[]> one = Run(1, 1, placedStep == StandardDecoderModel.RingStepRows ? null : placedStep);
            double worst = 0, worstMax = 0;
            for (int i = 0; i < one.Count; i++)
            {
                var (relL2, relMax) = MuseGlimmerGoldens.Distance(one[i], placed[i]);
                worst = Math.Max(worst, relL2);
                worstMax = Math.Max(worstMax, relMax);
                int want = MuseGlimmerParity.ArgMax(one[i], one[i].Length), got = MuseGlimmerParity.ArgMax(placed[i], placed[i].Length);
                if (want != got) problems.Add($"{caseName} output {i}: argmax {got}, one GPU {want}");
            }
            output.WriteLine($"{caseName} on {backend}: tp {tp} x stages {stages}: worst relL2 {worst:E2}, largest difference {worstMax:E2} over {one.Count} outputs");
            if (worst > TwoGpuPlacementBound(backend) || worstMax > TwoGpuPlacementBound(backend))
                problems.Add($"{caseName}: relL2 {worst:E2}, largest difference {worstMax:E2}");
        }
        Assert.True(problems.Count == 0, $"{backend}, tp {tp} x stages {stages}: {string.Join("; ", problems)}");
    }

    /// <summary>A tensor-parallel degree that does not divide the KV heads (the fixture's 2, as the 30B's) is refused at load,
    /// naming them.</summary>
    [Fact]
    public void ATensorParallelDegreePastTheKvHeadsIsRefused()
    {
        Directory.CreateDirectory(_dir);
        string path = MuseGlimmerGoldens.WriteFixture("untied", _dir);
        var e = Assert.Throws<NotSupportedException>(() =>
        {
            using ModelBase model = MuseGlimmerArchitecture.CreateRuntimeModel(path, BackendType.Cpu, 4, new SimulatedDeviceGroup(4));
        });
        output.WriteLine(e.Message);
        Assert.Contains("2 KV heads", e.Message, StringComparison.Ordinal);
    }

    /// <summary>The descriptor builds the shared-runtime family (the switch, stage S8; the legacy model went in S9).</summary>
    [Fact]
    public void TheDescriptorBuildsTheRuntimeFamily()
    {
        Directory.CreateDirectory(_dir);
        string path = MuseGlimmerGoldens.WriteFixture("untied", _dir);
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        using (ModelBase created = ModelBase.Create(path, BackendType.Cpu))
            Assert.IsType<MuseGlimmerRuntimeModel>(created);
        Assert.Equal(TensorSharp.Models.Architecture.ServingImplementation.SharedRuntime, MuseGlimmerArchitecture.Descriptor.Serving.Implementation);
    }
}
