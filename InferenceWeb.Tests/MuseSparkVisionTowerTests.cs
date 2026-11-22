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
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Models.Graph;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Muse Spark on the shared vision tower (stage S4 of docs/design/muse-glimmer-unified.md): its mmproj (projector type
/// muse-glimmer at Spark's width) loads as <see cref="MuseGlimmerVisionTower"/>, the graph program Muse-Glimmer runs, whose
/// rows are a standalone tower's, and the legacy per-op tower's (frozen) within the towers' parity bound, and they enter Spark's
/// prompt as its own vision rows do (normalized with the text, <see cref="MuseSparkModelTests"/>).
/// </summary>
[Collection("Muse Spark model")]
public sealed class MuseSparkVisionTowerTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-muse-spark-vision-" + Guid.NewGuid().ToString("N"));
    private readonly MuseSparkSyntheticModelBuilder _fixture = new();

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    [Fact]
    public void SparkLoadsTheSharedGraphTowerAndTakesItsRows()
    {
        Directory.CreateDirectory(_dir);
        string mmproj = MuseGlimmerVisionSynthetic.Write(Path.Combine(_dir, "mmproj-spark.gguf"), projection: MuseSparkSyntheticModelBuilder.Hidden,
            products: SyntheticGguf.GgmlType.F32);
        // F32 K/V, as Spark's reference-decoder comparisons run (MuseSparkModelTests), for this flow only.
        using var kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        using var model = new MuseSparkModel(_fixture.Write(Path.Combine(_dir, "spark.gguf")), BackendType.Cpu);
        model.LoadVisionEncoder(mmproj);
        MuseGlimmerVisionTower tower = Assert.IsType<MuseGlimmerVisionTower>(model.VisionEncoder);
        Assert.IsType<ManagedGraphExecutor>(tower.Graph.Executor);
        Assert.Equal(MuseSparkSyntheticModelBuilder.Hidden, tower.ProjectionDim);

        // 6 x 10 patches: six windows, 15 rows.
        int w = 6 * MuseGlimmerVisionSynthetic.Patch, h = 10 * MuseGlimmerVisionSynthetic.Patch;
        float[] px = MuseGlimmerVisionSynthetic.Pixels(w, h, 5);
        float[] rows = tower.EncodeToHost(px, w, h, out int count);
        Assert.Equal(15, count);
        var cpu = new CpuAllocator(BlasEnum.DotNet);
        using (var gguf = new GgufFile(mmproj))
        using (var standalone = new MuseGlimmerVisionTower(gguf, cpu, BackendType.Cpu))
            Assert.Equal(standalone.EncodeToHost(px, w, h, out _), rows);
        // The legacy per-op tower's rows at Spark's width, frozen on the managed backend before stage S9 deleted it.
        float[] legacy = MuseGlimmerLegacySynthetic.TowerRows("spark-f32", mmproj, 6, 10, 5);
        var d = MuseGlimmerVisionTowerTests.Distance(legacy, rows);
        output.WriteLine($"Spark's tower vs the legacy tower's frozen rows: relL2 {d.RelL2:E2}, max {d.RelMax:E2}");
        Assert.True(Math.Max(d.RelL2, d.RelMax) <= MuseGlimmerVisionTowerTests.ParityBound);

        // The rows replace their placeholders (positions 1 .. 15) as Spark's reference decoder takes replacements.
        int[] tokens = new[] { 7 }.Concat(Enumerable.Repeat(9, count)).Concat(new[] { 21, 42, 11 }).ToArray();
        var replacements = new Dictionary<int, float[]>();
        int hidden = MuseSparkSyntheticModelBuilder.Hidden;
        for (int r = 0; r < count; r++) replacements[r + 1] = rows.AsSpan(r * hidden, hidden).ToArray();
        var embeddings = new Tensor(cpu, DType.Float32, count, hidden);
        embeddings.SetElementsAsFloat(rows);
        model.SetVisionEmbeddings(embeddings, 1);
        float[] expected = _fixture.Reference(tokens, replacements: replacements), actual = model.Forward(tokens);
        float error = expected.Zip(actual, (x, y) => MathF.Abs(x - y)).Max();
        output.WriteLine($"logits with the tower's rows vs the reference decoder: max |d| {error:E2}");
        Assert.True(actual.All(float.IsFinite));
        Assert.True(error <= 3e-5f, $"maximum logit error {error}");
        Assert.False(model.HasPendingVisionEmbeddings);
    }
}
