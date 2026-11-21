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
/// The legacy Muse-Glimmer model's logits on the synthetic fixtures, frozen per lane in stage S0 before the unified-runtime
/// migration changed shared code (<see cref="MuseGlimmerGoldens"/>), which the runtime family is held to
/// (<see cref="MuseGlimmerRuntimeTests"/>) since stage S9 deleted the legacy model: the files themselves, every lane holding
/// every case from the fixtures the generator writes today, and the lanes' spread under <see cref="CrossLaneBound"/>.
/// </summary>
public sealed class MuseGlimmerFrozenGoldensTests(ITestOutputHelper output) : IDisposable
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
