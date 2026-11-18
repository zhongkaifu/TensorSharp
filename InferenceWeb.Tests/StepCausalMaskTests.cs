// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The causal masks a shared-runtime step emits: one input a sequence, whatever the attention layers, with the values the
// layers read before (the managed CPU executor, synthetic Qwen3.5 fixture).
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class StepCausalMaskTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-step-masks-" + Guid.NewGuid().ToString("N"));

    /// <summary>The fixture's layers; every fourth attends (full_attention_interval 4).</summary>
    private const int Layers = 8, AttentionLayers = Layers / 4;

    public StepCausalMaskTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Qwen35Fixture()
        => new Qwen35DenseSyntheticModelBuilder { Layers = Layers, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));

    /// <summary>
    /// A step of two sequences (one decode row and another request's prompt) emits each sequence's mask once, read by every
    /// attention layer: the step had kept one mask, which each sequence's evicted, so every layer emitted both again. The
    /// logits of that step, of the prompt before it and of the two sequences' decode steps after it (advanced a position
    /// on, each sequence's mask by its own rows) are the bits the per-layer masks gave.
    /// </summary>
    [Fact]
    public void AStepOfTwoSequencesEmitsEachSequencesMaskOnce()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = Qwen35Fixture();
        var cached = RunTwoSequences(path, singleMaskCache: false);
        var perLayer = RunTwoSequences(path, singleMaskCache: true);

        Assert.Equal(2, cached.MixedStepMasks);
        Assert.Equal(2 * AttentionLayers, perLayer.MixedStepMasks);
        Assert.Equal(2, cached.DecodeStepMasks);
        Assert.Equal(2 * AttentionLayers, perLayer.DecodeStepMasks);
        Assert.True(cached.Advances > 0, "the two sequences' decode steps did not advance");
        Assert.Equal(perLayer.Logits.Count, cached.Logits.Count);
        for (int i = 0; i < cached.Logits.Count; i++)
            Assert.True(cached.Logits[i].AsSpan().SequenceEqual(perLayer.Logits[i]), $"step {i}: the per-sequence masks moved the logits");
    }

    private static (List<float[]> Logits, long MixedStepMasks, long DecodeStepMasks, int Advances) RunTwoSequences(string path,
        bool singleMaskCache)
    {
        using var model = new Qwen35RuntimeModel(path, BackendType.Cpu, 1, null, 1);
        model.SingleMaskCacheForTests = singleMaskCache;
        var unified = (IUnifiedBatchModel)model;
        foreach (string id in new[] { "a", "b" }) unified.EnsureSequenceSlot(id, 128);
        var trace = new List<float[]>();
        int[] promptA = Enumerable.Range(0, 20).Select(i => 40 + (i * 11) % 150).ToArray();
        int[] promptB = Enumerable.Range(0, 24).Select(i => 30 + (i * 7) % 180).ToArray();

        var one = new float[1][];
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { promptA }, new[] { 0 }, one));
        trace.Add((float[])one[0].Clone());
        int nextA = ArgMax(one[0]);

        // A's decode row and B's prompt in one step.
        var two = new float[2][];
        int runs = model.GraphRuns;
        long before = model.CausalMaskInputs;
        Assert.True(unified.TryForwardUnified(new[] { "a", "b" }, new[] { new[] { nextA }, promptB }, new[] { promptA.Length, 0 }, two));
        long mixed = model.CausalMaskInputs - before;
        Assert.True(model.GraphRuns > runs, "the mixed step did not run as a graph");
        trace.AddRange(two.Select(r => (float[])r.Clone()));

        // Both decode: the first step traces, the later ones advance the kept step.
        int[] starts = { promptA.Length + 1, promptB.Length };
        int[][] tokens = { new[] { ArgMax(two[0]) }, new[] { ArgMax(two[1]) } };
        long decodeMasks = -1;
        int advances = model.DecodeAdvances;
        for (int step = 0; step < 6; step++)
        {
            before = model.CausalMaskInputs;
            Assert.True(unified.TryForwardUnified(new[] { "a", "b" }, tokens, starts, two));
            if (step == 0) decodeMasks = model.CausalMaskInputs - before;
            trace.AddRange(two.Select(r => (float[])r.Clone()));
            for (int s = 0; s < 2; s++) starts[s]++;
            tokens = new[] { new[] { ArgMax(two[0]) }, new[] { ArgMax(two[1]) } };
        }
        return (trace, mixed, decodeMasks, model.DecodeAdvances - advances);
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int v = 1; v < row.Length; v++) if (row[v] > row[best]) best = v;
        return best;
    }
}
