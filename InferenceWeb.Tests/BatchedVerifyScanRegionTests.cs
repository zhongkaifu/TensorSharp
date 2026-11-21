// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A batched verify's sequences' gated delta-net scans in one launch a layer (GraphFusion.DeltaNetScans, the direct CUDA
/// executor's multi-sequence scan), their delta-rule preps with their conv histories' copies and their gated norms with the
/// rows' concatenation (GraphFusion.DeltaNetPreps, GraphFusion.GatedRmsNorms) and their F16 cache writes and attention windows
/// in one launch a layer each (GraphFusion.SequenceAttentions, the multi-sequence write and window kernels): the same math as each sequence's own launches, so every row's logits, and every row after
/// the commits (which re-step each sequence's kept rows on its state), are the bits of the step that launches them a sequence
/// at a time, and those of each sequence run alone.
/// </summary>
public sealed class BatchedVerifyScanRegionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-scan-region-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public BatchedVerifyScanRegionTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly int[] Prompts = { 21, 30, 14, 9 };
    private static readonly int[][] Kept = { new[] { 3, 8, 1, 5 }, new[] { 6, 2, 8, 4 } };
    private const int Window = 8;

    private static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    private static int[] Prompt(int r, int length) => Enumerable.Range(0, length).Select(i => 3 + (i * 37 + r * 11 + i / 7) % 250).ToArray();

    /// <summary>
    /// Requests <paramref name="lanes"/> on one model: each prompt alone, then two verify windows of all of them in one step
    /// each (window tokens fixed by the request and the step, the kept counts of <see cref="Kept"/>), then two decode rows of
    /// each alone. Every row's logits, per request in order.
    /// </summary>
    private static List<float[]>[] Run(StandardDecoderModel model, int[] lanes)
    {
        var unified = (IUnifiedBatchModel)model;
        var batched = (IBatchedSpeculativeTarget)model;
        int vocab = model.Config.VocabSize;
        var rows = lanes.Select(_ => new List<float[]>()).ToArray();
        var ids = lanes.Select(r => $"r{r}").ToArray();
        var length = new int[lanes.Length];
        var next = new int[lanes.Length];
        var logits = new float[1][];
        for (int j = 0; j < lanes.Length; j++)
        {
            int r = lanes[j];
            Assert.Equal(0, unified.EnsureSequenceSlot(ids[j], 256));
            Assert.True(unified.TryForwardUnified(new[] { ids[j] }, new[] { Prompt(r, Prompts[r]) }, new[] { 0 }, logits));
            rows[j].Add((float[])logits[0].Clone());
            length[j] = Prompts[r];
            next[j] = ArgMax(logits[0]);
        }
        for (int step = 0; step < Kept.Length; step++)
        {
            var tokens = new int[lanes.Length * Window];
            for (int j = 0; j < lanes.Length; j++)
                for (int i = 0; i < Window; i++)
                    tokens[j * Window + i] = i == 0 ? next[j] : 3 + (next[j] * 13 + i * 29 + lanes[j] * 7 + step) % 250;
            var windowRows = new float[lanes.Length * Window * vocab];
            Assert.True(batched.TryBatchedSpecVerifyLogits(ids, (int[])length.Clone(), Window, tokens, windowRows, null));
            var kept = lanes.Select(r => Kept[step][r]).ToArray();
            batched.CommitBatchedSpecVerify(ids, (int[])length.Clone(), kept);
            for (int j = 0; j < lanes.Length; j++)
            {
                for (int i = 0; i < Window; i++) rows[j].Add(windowRows.AsSpan((j * Window + i) * vocab, vocab).ToArray());
                next[j] = ArgMax(windowRows.AsSpan((j * Window + kept[j] - 1) * vocab, vocab));
                length[j] += kept[j];
            }
        }
        for (int t = 0; t < 2; t++)
            for (int j = 0; j < lanes.Length; j++)
            {
                Assert.True(unified.TryForwardUnified(new[] { ids[j] }, new[] { new[] { next[j] } }, new[] { length[j] }, logits));
                rows[j].Add((float[])logits[0].Clone());
                next[j] = ArgMax(logits[0]);
                length[j]++;
            }
        foreach (string id in ids) model.OnSequenceReleased(id);
        return rows;
    }

    private int Apart(List<float[]> a, List<float[]> b)
    {
        Assert.Equal(a.Count, b.Count);
        int apart = 0;
        for (int i = 0; i < a.Count; i++)
            if (!a[i].AsSpan().SequenceEqual(b[i])) apart++;
        return apart;
    }

    /// <summary>Over an F16 cache and a q8_0 one, which the flash window kernels read (an F32 cache takes the generic attention).</summary>
    [CudaTheory]
    [InlineData(KvCacheDtype.F16)]
    [InlineData(KvCacheDtype.Q8_0)]
    public void SequencesScannedAndAttendedInOneLaunchAreEachSequencesOwnBits(KvCacheDtype kv)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(kv);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35.gguf"));
        int[] all = Enumerable.Range(0, Prompts.Length).ToArray();
        List<float[]>[] together, apart;
        long regions, windows, preps, norms, writes;
        using (var model = (StandardDecoderModel)ModelBase.Create(path, BackendType.Cuda))
        {
            together = Run(model, all);
            var executor = (CudaGraphExecutor)model.GraphExecutorForTests;
            (regions, windows, preps, norms, writes) = (executor.DeltaNetScanRegions, executor.SequenceWindowLaunches, executor.PrepSequenceLaunches,
                executor.GatedNormSequenceLaunches, executor.SequenceWriteLaunches);
        }
        using (var model = (StandardDecoderModel)ModelBase.Create(path, BackendType.Cuda))
        {
            ((Qwen35RuntimeModel)model).DeltaNetScanRegionsDisabledForTests = true;
            ((Qwen35RuntimeModel)model).SequencePrepRegionsDisabledForTests = true;
            model.SequenceAttentionRegionsDisabledForTests = true;
            apart = Run(model, all);
            var executor = (CudaGraphExecutor)model.GraphExecutorForTests;
            Assert.Equal(0, executor.DeltaNetScanRegions);
            Assert.Equal(0, executor.SequenceWindowLaunches);
            Assert.Equal(0, executor.PrepSequenceLaunches);
            Assert.Equal(0, executor.GatedNormSequenceLaunches);
            Assert.Equal(0, executor.SequenceWriteLaunches);
        }
        Assert.True(regions > 0, "no verify step scanned its sequences in one launch");
        Assert.True(windows > 0, "no verify step attended its sequences' windows in one launch");
        Assert.True(preps > 0, "no verify step prepared its sequences' delta-rule inputs in one launch");
        Assert.True(norms > 0, "no verify step normed its sequences' outputs in one launch");
        // An F16 cache's rows are written for every sequence in one launch; a q8_0 cache's each as before.
        if (kv == KvCacheDtype.F16) Assert.True(writes > 0, "no verify step wrote its sequences' cache rows in one launch");
        var alone = new List<float[]>[Prompts.Length];
        for (int r = 0; r < Prompts.Length; r++)
            using (var model = (StandardDecoderModel)ModelBase.Create(path, BackendType.Cuda))
                alone[r] = Run(model, new[] { r })[0];
        for (int r = 0; r < Prompts.Length; r++)
        {
            int vsOps = Apart(together[r], apart[r]), vsAlone = Apart(together[r], alone[r]);
            _output.WriteLine($"request {r}: {together[r].Count} rows; {vsOps} apart from the preps, scans, norms and windows launched a sequence each, {vsAlone} from it alone");
            Assert.Equal(0, vsOps);
            Assert.Equal(0, vsAlone);
        }
        _output.WriteLine($"{regions} scan regions, {preps} prep regions, {norms} norm regions, {writes} write launches, {windows} window regions");
    }
}
