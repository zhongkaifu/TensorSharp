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
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The legacy Muse-Glimmer model (<see cref="MuseGlimmerModel"/>) against its own logits on the synthetic fixtures, frozen
/// per lane before the unified-runtime migration changes shared code (<see cref="MuseGlimmerGoldens"/>): byte for byte on
/// the lane that froze them, within <see cref="CrossLaneBound"/> of the nearest frozen lane elsewhere. Until stage S9 deletes
/// legacy, this keeps the oracle the runtime family is compared with from moving under it.
/// </summary>
public sealed class MuseGlimmerFrozenLegacyTests(ITestOutputHelper output) : IDisposable
{
    /// <summary>
    /// Off the lane a case was frozen on, the bound on every output's relative L2 distance and on its largest difference
    /// over the largest frozen value. Measured at the freeze (ggml d9fe7bff, M5 Pro) between the four lanes' legacy outputs:
    /// see <see cref="TheFrozenLanesAgreeWithinTheBound"/>, which prints the spread.
    /// </summary>
    internal const double CrossLaneBound = 1e-2;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-mg-frozen-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    public static TheoryData<string> CaseNames()
    {
        var data = new TheoryData<string>();
        foreach (var c in MuseGlimmerGoldens.Cases) data.Add(c.Name);
        return data;
    }

    /// <summary>Runs one case on the legacy model and returns its fixture's SHA-256 and its outputs.</summary>
    internal static (string FixtureSha256, List<float[]> Outputs) RunLegacy(MuseGlimmerGoldens.CaseSpec spec, BackendType backend, string dir)
    {
        Directory.CreateDirectory(dir);
        string path = MuseGlimmerGoldens.WriteFixture(spec.Fixture, dir);
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(MuseGlimmerGoldens.CaseKvDtype);
        using var model = new MuseGlimmerModel(path, backend);
        return (MuseGlimmerGoldens.FileSha(path), spec.Script(model));
    }

    private void Verify(BackendType backend, string caseName)
    {
        var spec = MuseGlimmerGoldens.Cases.Single(c => c.Name == caseName);
        string lane = MuseGlimmerGoldens.LaneKey(backend);
        var (sha, outputs) = RunLegacy(spec, backend, _dir);
        MuseGlimmerGoldens.Case frozen = MuseGlimmerGoldens.Find(caseName, lane);
        bool bytes = frozen != null && frozen.Lane == lane;
        output.WriteLine(MuseGlimmerGoldens.Check(frozen, caseName, sha, outputs, bytes, CrossLaneBound));
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void LegacyOnTheManagedBackendReproducesItsFrozenLogits(string caseName) => Verify(BackendType.Cpu, caseName);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(CaseNames))]
    public void LegacyOnGgmlCpuReproducesItsFrozenLogits(string caseName) => Verify(BackendType.GgmlCpu, caseName);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(CaseNames))]
    public void LegacyOnGgmlMetalReproducesItsFrozenLogits(string caseName) => Verify(BackendType.GgmlMetal, caseName);

    [MlxTheory]
    [MemberData(nameof(CaseNames))]
    public void LegacyOnMlxReproducesItsFrozenLogits(string caseName) => Verify(BackendType.Mlx, caseName);

    [CudaTheory]
    [MemberData(nameof(CaseNames))]
    public void LegacyOnDirectCudaReproducesItsFrozenLogits(string caseName) => Verify(BackendType.Cuda, caseName);

    /// <summary>Every frozen lane holds every case, each from the fixture bytes the generator writes today, with as many
    /// outputs as the case's script returns.</summary>
    [Fact]
    public void EveryFrozenLaneHoldsEveryCaseFromTheSameFixtures()
    {
        Assert.NotEmpty(MuseGlimmerGoldens.Lanes);
        Directory.CreateDirectory(_dir);
        var shas = MuseGlimmerGoldens.Cases.Select(c => c.Fixture).Distinct()
            .ToDictionary(f => f, f => MuseGlimmerGoldens.FileSha(MuseGlimmerGoldens.WriteFixture(f, _dir)));
        foreach (string lane in MuseGlimmerGoldens.Lanes)
        {
            var cases = MuseGlimmerGoldens.CasesOf(lane);
            foreach (var spec in MuseGlimmerGoldens.Cases)
            {
                Assert.True(cases.TryGetValue(spec.Name, out var c), $"{lane} has no {spec.Name}");
                Assert.Equal(shas[spec.Fixture], c.FixtureSha256);
                Assert.Equal(MuseGlimmerGoldens.CaseKvDtype.ToString(), c.KvDtype);
                Assert.All(c.Outputs, o => Assert.Equal(MuseGlimmerSyntheticModelBuilder.ByteTokens, o.Values.Length));
            }
            output.WriteLine($"{lane}: {cases.Count} cases, {cases.Values.Sum(c => c.Outputs.Length)} outputs");
        }
        // The four lanes the freeze covers on the Mac.
        foreach (string backend in new[] { "cpu|", "ggml_cpu|", "ggml_metal|", "mlx|" })
            Assert.Contains(MuseGlimmerGoldens.Lanes, l => l.StartsWith(backend, StringComparison.Ordinal));
    }

    /// <summary>The frozen lanes agree with each other within <see cref="CrossLaneBound"/>: the bound another lane is held to is
    /// wider than what separates the backends' own legacy outputs (printed, per case, with the cases' largest logits and the
    /// share of outputs whose argmax all lanes share).</summary>
    [Fact]
    public void TheFrozenLanesAgreeWithinTheBound()
    {
        string[] lanes = MuseGlimmerGoldens.Lanes.OrderBy(l => l, StringComparer.Ordinal).ToArray();
        Assert.True(lanes.Length >= 2, "at least two frozen lanes are needed to measure their spread");
        double worst = 0;
        foreach (var spec in MuseGlimmerGoldens.Cases)
        {
            double caseWorst = 0;
            int outputs = MuseGlimmerGoldens.CasesOf(lanes[0])[spec.Name].Outputs.Length, sameArgmax = 0;
            for (int o = 0; o < outputs; o++)
            {
                var values = lanes.Select(l => MuseGlimmerGoldens.CasesOf(l)[spec.Name].Outputs[o].Values).ToArray();
                if (values.Select(v => MuseGlimmerParity.ArgMax(v, v.Length)).Distinct().Count() == 1) sameArgmax++;
                for (int a = 0; a < values.Length; a++)
                    for (int b = 0; b < values.Length; b++)
                    {
                        if (a == b) continue;
                        var (relL2, relMax) = MuseGlimmerGoldens.Distance(values[a], values[b]);
                        caseWorst = Math.Max(caseWorst, Math.Max(relL2, relMax));
                    }
            }
            double maxAbs = MuseGlimmerGoldens.CasesOf(lanes[0])[spec.Name].Outputs.Max(x => x.MaxAbs);
            output.WriteLine($"{spec.Name}: {outputs} outputs, largest |logit| {maxAbs:F3}, worst cross-lane distance {caseWorst:E3}, " +
                $"argmax shared by every lane at {sameArgmax}/{outputs}");
            worst = Math.Max(worst, caseWorst);
        }
        output.WriteLine($"lanes: {string.Join("; ", lanes)}; worst {worst:E3} (bound {CrossLaneBound:E1})");
        Assert.True(worst <= CrossLaneBound / 2, $"the frozen lanes are {worst:E3} apart, too close to the bound {CrossLaneBound:E1} other lanes are held to");
    }
}

/// <summary>Skips unless <paramref name="envVar"/> names an existing directory: the writers below run only when asked.</summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class FreezeFactAttribute : FactAttribute
{
    public FreezeFactAttribute(string envVar)
    {
        string dir = Environment.GetEnvironmentVariable(envVar);
        Skip = string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir)
            ? $"A freeze writer: runs only when {envVar} names the directory to write into."
            : null;
    }
}

/// <summary>
/// Writes <see cref="MuseGlimmerGoldens"/>' files: every case on the legacy model, on each lane this process can run (the
/// managed backend, the pinned GGML backend, MLX and direct CUDA where they are there), into the directory TS_TEST_FREEZE_MUSE_GLIMMER_GOLDENS
/// names (Fixtures/MuseGlimmer/LegacyGoldens). TS_TEST_FREEZE_PROVENANCE, when set, is recorded as the build the outputs came
/// from. Goes with the legacy code in stage S9.
/// </summary>
[Trait("Category", "Bench")]
public sealed class MuseGlimmerGoldenWriterTests(ITestOutputHelper output)
{
    private const string OutEnv = "TS_TEST_FREEZE_MUSE_GLIMMER_GOLDENS";

    [FreezeFact(OutEnv)]
    public void FreezeTheLegacyLogits()
    {
        string outDir = Environment.GetEnvironmentVariable(OutEnv);
        string scratch = Path.Combine(Path.GetTempPath(), "ts-mg-freeze-" + Guid.NewGuid().ToString("N"));
        try
        {
            var backends = new List<BackendType> { BackendType.Cpu, TestGates.PinnedGgmlBackend };
            if (TestGates.MlxSkip == null) backends.Add(BackendType.Mlx);
            if (TestGates.CudaSkip == null) backends.Add(BackendType.Cuda);
            foreach (BackendType backend in backends)
            {
                string lane = MuseGlimmerGoldens.LaneKey(backend);
                var cases = new JsonObject();
                foreach (var spec in MuseGlimmerGoldens.Cases)
                {
                    var (sha, outputs) = MuseGlimmerFrozenLegacyTests.RunLegacy(spec, backend, scratch);
                    var (again, repeat) = MuseGlimmerFrozenLegacyTests.RunLegacy(spec, backend, scratch);
                    Assert.Equal(sha, again);
                    for (int i = 0; i < outputs.Count; i++)
                        Assert.True(MuseGlimmerGoldens.Sha(outputs[i]) == MuseGlimmerGoldens.Sha(repeat[i]),
                            $"{lane} {spec.Name} output {i}: two legacy runs differ, so the lane cannot be frozen byte for byte");
                    cases[spec.Name] = MuseGlimmerGoldens.CaseJson(sha, outputs);
                    output.WriteLine($"{lane} {spec.Name}: {outputs.Count} outputs, repeatable");
                }
                var file = new JsonObject
                {
                    ["lane"] = lane,
                    ["about"] = "The legacy Muse-Glimmer model's logits on the synthetic fixtures, frozen on this lane in stage S0 of the " +
                        "unified-runtime migration (InferenceWeb.Tests/MuseGlimmerGoldens.cs reads them).",
                    ["producer"] = "MuseGlimmerModel (legacy)",
                    ["provenance"] = Environment.GetEnvironmentVariable("TS_TEST_FREEZE_PROVENANCE") ?? "",
                    ["frozenUtc"] = DateTime.UtcNow.ToString("O"),
                    ["cases"] = cases,
                };
                string path = Path.Combine(outDir, MuseGlimmerGoldens.FileName(lane));
                File.WriteAllText(path, file.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
                output.WriteLine($"wrote {path}");
            }
        }
        finally
        {
            try { Directory.Delete(scratch, recursive: true); } catch (IOException) { }
        }
    }
}
