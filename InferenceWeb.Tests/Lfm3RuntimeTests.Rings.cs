// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Sliding-window ring caches (StandardDecoderModel.SlidingRings.cs) on LFM3: the synthetic window of 8 keys gets rings of 16
// cells (8 + 8 rows a step), so every script below laps them several times. Each mechanism the linear caches are tested for
// is tested again over rings that have wrapped: the reference forward, prompts read whole / token by token / in chunks,
// decode steps advanced in place, verify windows across the wrap, checkpoints (clones, device copies, files), speculation,
// slot batches, continuous batching and every placement, plus the rings' sizes and the truncations they allow.
using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed partial class Lfm3RuntimeTests
{
    private const int SmallRingStepRows = 8, SmallRingCells = 16;

    /// <summary>Rings of 16 cells for models constructed inside the scope (the geometry is read at construction).</summary>
    private static IDisposable SmallRings()
    {
        StandardDecoderModel.RingGeometryForTests.Value = (SmallRingStepRows, SmallRingStepRows);
        return new RingScope();
    }

    private sealed class RingScope : IDisposable
    {
        public void Dispose() => StandardDecoderModel.RingGeometryForTests.Value = null;
    }

    /// <summary>Under <see cref="SmallRings"/>: the fixture's sliding layers keep 16-cell rings, the others none, and the script
    /// that ran reached past the rings' first lap (<paramref name="positions"/>, 0: not checked).</summary>
    private static void AssertRingsWrapped(Lfm3RuntimeModel model, int positions)
    {
        if (StandardDecoderModel.RingGeometryForTests.Value == null) return;
        // "csfcsfsc": layers 1, 4 and 6 slide, 2 and 5 attend fully.
        foreach (int l in new[] { 1, 4, 6 }) Assert.Equal(SmallRingCells, model.RingCellsForTests(l));
        foreach (int l in new[] { 0, 2, 5 }) Assert.Equal(0, model.RingCellsForTests(l));
        Assert.Equal(SmallRingStepRows, model.MaxSequenceStepRows);
        if (positions > 0) Assert.True(positions > SmallRingCells, $"the script reaches {positions} positions: the 16-cell rings never wrap");
    }

    [Fact]
    public void SlidingWindowLayersKeepRingsOfTheirWindowAndAStep()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var rings = SmallRings();
        using var model = Create(Fixture("dense"));
        AssertRingsWrapped(model, 0);
        model.ForwardRefill(Enumerable.Range(0, 45).Select(i => 40 + (i * 11) % 150).ToArray());
        Tensor[][] k = model.KeyCachesForTests;
        // A ring is never larger than its slot, and keeps its cells as the slot grows past them.
        Assert.Equal(SmallRingCells, k[1][0].Sizes[1]);
        Assert.Equal(SmallRingCells, k[4][0].Sizes[1]);
        Assert.True(k[2][0].Sizes[1] >= 45, "a full layer keeps every position");
        Assert.Null(k[0]);   // a short-conv layer has no K/V

        // The default geometry: the window plus llama.cpp's 512-row ubatch, aligned to 256 (768 cells for LFM3's 256 keys).
        StandardDecoderModel.RingGeometryForTests.Value = null;
        using var defaults = Create(Fixture("dense"));
        int expected = (Lfm3SyntheticModelBuilder.Window + StandardDecoderModel.RingStepRows + StandardDecoderModel.RingAlignment - 1)
            / StandardDecoderModel.RingAlignment * StandardDecoderModel.RingAlignment;
        Assert.Equal(expected, defaults.RingCellsForTests(1));
        Assert.Equal(StandardDecoderModel.RingStepRows, defaults.MaxSequenceStepRows);
    }

    [Theory]
    [InlineData("graphs")]
    [InlineData("per-op")]
    public void RingsMatchTheReferenceForward(string path)
    {
        using var rings = SmallRings();
        AssertMatchesReference(BackendType.Cpu, path == "graphs", 2e-5);
    }

    [Theory]
    [InlineData("graphs")]
    [InlineData("per-op")]
    public void RingsOnThePinnedGgmlBackendMatchTheReferenceForward(string path)
    {
        using var rings = SmallRings();
        AssertMatchesReference(TestGates.PinnedGgmlBackend, path == "graphs", TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 1e-4 : 5e-3);
    }

    [MlxFact]
    public void RingsOnMlxMatchTheReferenceForward()
    {
        using var rings = SmallRings();
        AssertMatchesReference(BackendType.Mlx, graphs: true, 5e-3);
    }

    /// <summary>
    /// F16 K/V (the default cache type: on ggml_metal every prompt chunk then takes the indexed path over the ring, on F32 the
    /// copy path): rings that wrapped give the logits linear caches give, for a long prompt read in ring steps and for decode
    /// steps that lap the ring, on every executor.
    /// </summary>
    [Theory]
    [InlineData("managed", "f16")]
    [InlineData("managed", "f32")]
    [InlineData("ggml", "f16")]
    [InlineData("ggml", "f32")]
    [InlineData("per-op", "f16")]
    [InlineData("per-op", "f32")]
    public void RingsGiveTheLogitsOfLinearCaches(string executor, string kv)
        => AssertRingsMatchLinear(executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu, executor != "per-op",
            kv == "f16" ? KvCacheDtype.F16 : KvCacheDtype.F32);

    [MlxFact]
    public void RingsOnMlxGiveTheLogitsOfLinearCaches() => AssertRingsMatchLinear(BackendType.Mlx, graphs: true, KvCacheDtype.F16);

    private void AssertRingsMatchLinear(BackendType backend, bool graphs, KvCacheDtype kv)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture("dense");
        int[] prompt = Enumerable.Range(0, 45).Select(i => 40 + (i * 13) % 170).ToArray();
        List<float[]> Run(bool rings)
        {
            using IDisposable scope = rings ? SmallRings() : null;
            using var model = Create(path, backend: backend);
            model.GraphLoweringEnabled = graphs;
            if (rings) AssertRingsWrapped(model, prompt.Length);
            else Assert.True(model.RingCellsForTests(1) == 0 || model.RingCellsForTests(1) > prompt.Length + 40, "the linear run must not wrap");
            // The prompt in the 8-row steps the rings take it in (the linear run too: products of other row counts round
            // apart, and the managed executor's F16 prompt attention rounds its operands from 17 rows on).
            float[] last = null;
            for (int at = 0; at < prompt.Length; at += SmallRingStepRows)
                last = model.Forward(prompt.AsSpan(at, Math.Min(SmallRingStepRows, prompt.Length - at)).ToArray());
            var rows = new List<float[]> { (float[])last.Clone() };
            int next = ArgMax(rows[0]);
            for (int i = 0; i < 40; i++)
            {
                rows.Add((float[])model.Forward(new[] { next }).Clone());
                next = ArgMax(rows[^1]);
            }
            Assert.True(graphs ? model.GraphRuns > 0 : model.GraphRuns == 0);
            return rows;
        }
        List<float[]> linear = Run(rings: false);
        List<float[]> ringed = Run(rings: true);
        // Over a wrapped ring attention reads the window's keys in ring order under a mask: the softmax sums in another order.
        // With F16 K/V the managed executor's causal-window kernels round their operands to F16 as ggml's flash attention does,
        // while a ring's mask (cells out of position order) takes its dense path over the widened rows, and the per-op path's
        // ring window is widened to F32 likewise: 3-6e-5 apart (F32 K/V: 1e-6).
        double tolerance = backend == BackendType.Cpu ? (kv == KvCacheDtype.F16 ? 1e-4 : 1e-5) : RoundsActivations(backend) ? 2.5e-2 : 5e-3;
        AssertRows(ringed, linear, tolerance, $"rings vs linear on {backend} graphs={graphs} kv={kv}");
    }

    /// <summary>
    /// The production geometry (rings of 768 cells, 512-row steps) on a 1,300-token prompt: its steps start at 0, 512 and
    /// 1,024, the second already reaches past the ring's first lap, so a step of the same width runs on either side of the wrap
    /// (a streaming executor must not run the second on the first's template), then decode steps over the wrapped ring.
    /// Against the double-precision reference forward.
    /// </summary>
    [Theory]
    [InlineData("managed", "graphs")]
    [InlineData("managed", "per-op")]
    [InlineData("ggml", "graphs")]
    public void LongPromptsOverProductionRingsMatchTheReferenceForward(string executor, string path)
        => AssertLongPromptMatchesReference(executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu, path == "graphs");

    [MlxFact]
    public void LongPromptsOverProductionRingsOnMlxMatchTheReferenceForward() => AssertLongPromptMatchesReference(BackendType.Mlx, graphs: true);

    private void AssertLongPromptMatchesReference(BackendType backend, bool graphs, KvCacheDtype kv = KvCacheDtype.F32, double tolerance = 0)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        Lfm3SyntheticModelBuilder builder = Builder("f32");
        string path = builder.Write(Path.Combine(_dir, "lfm3-f32-long.gguf"));
        var reference = new Lfm3Reference(builder);
        int[] prompt = Enumerable.Range(0, 1300).Select(i => 40 + (i * 37 + i / 7) % 200).ToArray();
        int[] decode = { 71, 72, 73, 74 };
        var expected = new List<float[]> { reference.Prompt(prompt) };
        foreach (int token in decode) expected.Add(reference.Step(token));

        using var model = Create(path, backend: backend);
        Assert.True(model.RingCellsForTests(1) is > 0 and < 1300, $"ring of {model.RingCellsForTests(1)} cells");
        model.GraphLoweringEnabled = graphs;
        // Twice: the second prompt's steps run as streams of the first's kinds on a streaming executor.
        for (int round = 0; round < 2; round++)
        {
            model.ResetKVCache();
            var actual = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
            foreach (int token in decode) actual.Add((float[])model.Forward(new[] { token }).Clone());
            AssertRows(actual, expected, tolerance > 0 ? tolerance : backend == BackendType.Cpu ? 2e-5 : 5e-3, $"{backend} graphs={graphs} kv={kv} round {round}");
        }
    }

    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void PromptsOverRingsMatchTokenByTokenAndChunked(string executor)
    {
        using var rings = SmallRings();
        AssertPromptPaths(executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu);
    }

    [MlxFact]
    public void PromptsOverRingsMatchTokenByTokenAndChunkedOnMlx()
    {
        using var rings = SmallRings();
        AssertPromptPaths(BackendType.Mlx);
    }

    /// <summary>Decode steps over rings that lap every 16 positions advance in place (their write cells and masks move round
    /// the ring) and give the traced steps' logits bit for bit, for one request and two.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void AdvancedDecodeStepsOverRingsMatchTracedOnes(string executor)
    {
        using var rings = SmallRings();
        AdvancedDecodeStepsMatchTracedOnes(executor);
    }

    /// <summary>A verify window across the ring's wrap (rows 14..18 of a 16-cell ring): every row as plain decoding gives it,
    /// and a commit of part of it continues exactly (the rejected rows' cells hold no position the window reads).</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    [InlineData("per-op")]
    public void VerifyWindowsAcrossTheRingWrapCommitTheKeptRows(string executor)
    {
        using var rings = SmallRings();
        AssertVerifyWindowCommits(executor, Enumerable.Range(0, 14).Select(i => 41 + (i * 11) % 90).ToArray());
    }

    /// <summary>A checkpoint of wrapped rings (23 positions, 16 cells) - cloned, copied on the device, into a pooled slot, and
    /// written to a file (format 6: the rings whole) - continues exactly.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void CheckpointsOfWrappedRingsContinueExactly(string executor)
    {
        using var rings = SmallRings();
        CheckpointsContinueExactly(executor);
    }

    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void SpeculationOverRingsMatchesPlainGreedy(string executor)
    {
        using var rings = SmallRings();
        SpeculativeDecodingMatchesPlainGreedy(executor);
    }

    [Theory]
    [MemberData(nameof(Placements))]
    public void PlacementsOverRingsMatchOneDevice(int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var rings = SmallRings();
        string path = Fixture("dense");
        // A longer script than Script's: 45 prompt tokens in ring steps, 30 decode steps.
        List<float[]> Run(Lfm3RuntimeModel model)
        {
            var rows = new List<float[]> { (float[])model.ForwardRefill(Enumerable.Range(0, 45).Select(i => 40 + (i * 11) % 150).ToArray()).Clone() };
            for (int i = 0; i < 30; i++) rows.Add((float[])model.Forward(new[] { 60 + i % 40 }).Clone());
            return rows;
        }
        List<float[]> reference;
        using (var single = Create(path))
            reference = Run(single);
        var group = tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        using var placed = new Lfm3RuntimeModel(path, BackendType.Cpu, tp, group, stages);
        AssertRingsWrapped(placed, 75);
        AssertRows(Run(placed), reference, tp == 1 ? 0 : 1e-4, $"rings tp={tp} stages={stages}");
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(1, 2)]
    public void SlotBatchesOverRingsMatchEachSequenceAlone(int tp, int stages)
    {
        using var rings = SmallRings();
        CompareSlotBatches(tp, stages, BackendType.Cpu);
    }

    [Fact]
    public void SlotBatchesOverRingsMatchEachSequenceAloneOnThePinnedGgmlBackend()
    {
        using var rings = SmallRings();
        CompareSlotBatches(1, 1, TestGates.PinnedGgmlBackend);
    }

    [MlxFact]
    public void SlotBatchesOverRingsMatchEachSequenceAloneOnMlx()
    {
        using var rings = SmallRings();
        CompareSlotBatches(1, 1, BackendType.Mlx);
    }

    /// <summary>The engine's prompt chunks (8 rows) and batched decode steps over rings: the greedy streams of each request
    /// alone. A batched prefill of chunks longer than a ring step is split into rounds (TryForwardSlotsInRingSteps).</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    public async Task ConcurrentRequestsOverRingsAreBatched(int tp, int stages)
    {
        using var rings = SmallRings();
        await ConcurrentRequestsAreBatched(tp, stages);
    }

    /// <summary>A batched prefill whose chunks are longer than a ring step runs as rounds and gives each sequence's lone logits.</summary>
    [Fact]
    public void BatchedPrefillLongerThanARingStepRunsInRounds()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var rings = SmallRings();
        string path = Fixture("dense");
        int[][] prompts =
        {
            Enumerable.Range(0, 37).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 5).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 21).Select(i => 48 + (i * 3) % 10).ToArray(),
        };
        var alone = new float[prompts.Length][];
        using (var reference = Create(path))
            for (int s = 0; s < prompts.Length; s++)
            {
                reference.ResetKVCache();
                alone[s] = (float[])reference.ForwardRefill(prompts[s]).Clone();
            }
        using var model = Create(path);
        var slots = (IBatchedPagedModel)model;
        string[] ids = { "x", "y", "z" };
        foreach (string id in ids) slots.BindSequenceCache(id);
        slots.RestorePrimaryCache();
        var logits = new float[ids.Length][];
        Assert.True(slots.TryForwardBatchedFusedPrefill(ids, prompts, new int[ids.Length], logits));
        for (int s = 0; s < ids.Length; s++)
            Assert.True(RelL2(logits[s], alone[s]) <= 1e-5, $"sequence {s}: relL2 {RelL2(logits[s], alone[s]):E2}");
    }

    /// <summary>Media rows (an image's embeddings) in a forward that rings take in steps: each step carries the rows of the
    /// spans it holds (a span cut across two steps included), giving the logits of linear caches reading it whole.</summary>
    [Theory]
    [InlineData("graphs")]
    [InlineData("per-op")]
    public void MediaRowsFollowTheirRingSteps(string path)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string file = Fixture("dense");
        int[] prompt = Enumerable.Range(0, 45).Select(i => 40 + (i * 13) % 170).ToArray();
        var random = new Random(5);
        float[] Rows(int n) => Enumerable.Range(0, n * Lfm3SyntheticModelBuilder.Hidden).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
        float[] first = Rows(12), second = Rows(9);   // rows [5, 17) cross the steps at 8 and 16; [30, 39) the one at 32
        float[] Run(bool rings)
        {
            using IDisposable scope = rings ? SmallRings() : null;
            using var model = Create(file);
            model.GraphLoweringEnabled = path == "graphs";
            foreach ((float[] values, int at) in new[] { (first, 5), (second, 30) })
            {
                var media = new Tensor(model.DirectAllocatorForTelemetry, DType.Float32, values.Length / Lfm3SyntheticModelBuilder.Hidden, Lfm3SyntheticModelBuilder.Hidden);
                media.SetElementsAsFloat(values);
                model.QueueMediaForTests(media, at);
            }
            float[] logits = (float[])model.Forward(prompt).Clone();
            if (rings) AssertRingsWrapped(model, prompt.Length);
            return logits;
        }
        float[] linear = Run(rings: false), ringed = Run(rings: true);
        Assert.True(RelL2(ringed, linear) <= 1e-5, $"media over ring steps vs whole: relL2 {RelL2(ringed, linear):E2}");
    }

    /// <summary>A stack without recurrent layers rewinds as far as its rings keep the next row's window: within the ring's
    /// slack past its window, never to positions it overwrote; the cache truncated there continues as a fresh run.</summary>
    [Fact]
    public void RingsRewindOnlyWhereTheyKeepTheWindow()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var rings = SmallRings();
        string path = new Lfm3SyntheticModelBuilder { Layout = "sfsf", UntiedOutput = true }.Write(Path.Combine(_dir, "lfm3-attn.gguf"));
        int[] tokens = Enumerable.Range(0, 40).Select(i => 40 + (i * 11) % 150).ToArray();
        using var model = Create(path);
        Assert.Equal(SmallRingCells, model.RingCellsForTests(0));
        Assert.False(model.SupportsKVStateSnapshot);
        model.ForwardRefill(tokens);
        // 40 positions in a 16-cell ring keep 24..39; position t's window reads (t - 8, t).
        Assert.True(model.CanTruncateKVCache(40, 32));    // reads 25..31
        Assert.True(model.CanTruncateKVCache(40, 31));    // reads 24..30, the oldest position kept
        Assert.False(model.CanTruncateKVCache(40, 30));   // reads 23..29: 23 is overwritten
        Assert.False(model.CanTruncateKVCache(40, 20));   // reads 13..19
        Assert.True(model.CanTruncateKVCache(40, 0));
        Assert.False(model.TryTruncateKVCache(20));
        Assert.True(model.TryTruncateKVCache(32));
        float[] continued = model.Forward(new[] { 77 });

        using var fresh = Create(path);
        fresh.ForwardRefill(tokens.Take(32).ToArray());
        float[] expected = fresh.Forward(new[] { 77 });
        Assert.True(RelL2(continued, expected) <= 1e-5, $"truncated ring vs fresh run: relL2 {RelL2(continued, expected):E2}");
    }
}
