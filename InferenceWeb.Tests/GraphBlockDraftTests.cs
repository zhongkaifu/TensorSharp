// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>
/// The DFlash / DFlash2 block drafter as the runtime's graph programs (StandardDecoderModel.BlockDraft.cs) drafts what the
/// per-op drafter (ModelBase.DFlash.cs) does. Two models attach the same drafter to a Qwen 3.5 dense trunk; one drafts
/// through the graphs, the other through the per-op path (GraphBlockDraftDisabledForTests), and both drafters take the
/// first model's trunk rows through a prompt and draft / verify / commit cycles whose accept counts walk every catch-up
/// width. The drafter's window (16) is shorter than the context, so the window slides and the 32-row ring wraps. Each block
/// must give the same ids with confidences within 1e-3, and the rings the same rows over the window within 1e-4 relative L2.
/// On ggml_metal the seam's model drafts on the fused GGML graph (CanUseFusedDFlash), so there the reference is that graph.
/// </summary>
public sealed class GraphBlockDraftTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-graph-block-draft-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public GraphBlockDraftTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private const int Cycles = 24;
    private const double ConfidenceTolerance = 1e-3, RingTolerance = 1e-4;

    [Theory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void GraphDraftsMatchThePerOpDrafterOnTheManagedCpu(string variant) => DraftsMatch(BackendType.Cpu, variant);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void GraphDraftsMatchThePerOpDrafterOnGgmlCpu(string variant) => DraftsMatch(BackendType.GgmlCpu, variant);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void GraphDraftsMatchThePerOpDrafterOnMetal(string variant) => DraftsMatch(BackendType.GgmlMetal, variant);

    [MlxTheory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void GraphDraftsMatchThePerOpDrafterOnMlx(string variant) => DraftsMatch(BackendType.Mlx, variant);

    /// <summary>The same on the direct CUDA backend (the CUDA graph executor: the ring written and the selector's codebooks
    /// gathered on the device), against the managed CPU's per-op drafter: CUDA's own per-op selector drifts from it (its
    /// DFlash2 confidences past the first slot), where the graph drafter matches it to the last digit.</summary>
    [CudaTheory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void GraphDraftsMatchTheManagedPerOpDrafterOnDirectCuda(string variant) => DraftsMatch(BackendType.Cuda, variant, BackendType.Cpu);

    /// <summary>The same on ggml_cuda, where the old path is the fused native drafter.</summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void GraphDraftsMatchTheFusedDrafterOnGgmlCuda(string variant) => DraftsMatch(BackendType.GgmlCuda, variant);

    /// <summary>Speculation with the graph drafter emits plain greedy decoding's stream, on the Qwen 3.5 trunk and on Qwen 3.8
    /// (qwen4exp: the drafter reads its hyper-connection streams' mean, QSA attention), and the graphs served every block.</summary>
    [Theory]
    [InlineData("qwen35", "dflash")]
    [InlineData("qwen35", "dflash2")]
    [InlineData("qwen4exp", "dflash2")]
    public void GraphDraftSpeculationMatchesPlainGreedyOnTheManagedCpu(string family, string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Fixtures(variant);
        if (family == "qwen4exp") target = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 16, qsa: true);
        StandardDecoderModel created = null;
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() => created = family == "qwen4exp"
                ? new Qwen4ExpRuntimeModel(target, BackendType.Cpu, draft)
                : new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft),
            $"{family} cpu {variant} graph drafter");
        Assert.Null(created.GraphBlockDraftDeclineReason);
        Assert.True(created.GraphBlockDrafts > 0, $"{family} {variant}: no block ran as a graph");
    }

    private (string Target, string Draft) Fixtures(string variant)
    {
        string target = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));
        int hidden = Qwen35DenseSyntheticModelBuilder.Hidden;
        string path = Path.Combine(_dir, variant + ".gguf");
        string draft = variant switch
        {
            "dflash" => Qwen4ExpSyntheticModelBuilder.WriteDFlash(path, blockSize: 4, hidden: hidden, slidingWindow: 16),
            "dflash2" => Qwen4ExpSyntheticModelBuilder.WriteDFlash(path, blockSize: 5, hidden: hidden, layers: 2, slidingWindow: 16, dflash2: true),
            "dflash2-q8" => Qwen4ExpSyntheticModelBuilder.WriteDFlash(path, blockSize: 5, hidden: hidden, slidingWindow: 16, dflash2: true,
                codebookType: SyntheticGguf.GgmlType.Q8_0),
            _ => throw new ArgumentOutOfRangeException(nameof(variant)),
        };
        return (target, draft);
    }

    /// <summary>The graph drafter on <paramref name="backend"/> against the per-op drafter on <paramref name="reference"/>
    /// (the same backend unless named).</summary>
    private void DraftsMatch(BackendType backend, string variant, BackendType? reference = null)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Fixtures(variant);
        using var graph = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft);
        using var perOp = new Qwen35RuntimeModel(target, reference ?? backend, 1, null, 1, draft) { GraphBlockDraftDisabledForTests = true };
        string label = $"{backend} {variant}";
        Assert.True(graph.HasDFlash && perOp.HasDFlash, $"{label}: the drafter did not attach");
        Assert.Equal(32, graph.DFlashRingRowsForTests);

        var trunk = (ISpeculativeModel)graph;
        var drafters = new IDraftHead[] { graph, perOp };
        int feat = trunk.SpecFeatureSize, vocab = graph.Config.VocabSize, gamma = trunk.DraftBlockSize;
        int[] prompt = Enumerable.Range(0, 21).Select(i => 40 + (i * 7) % 23).ToArray();

        // The prompt: token k pairs with the features of the token before it (zeros before the first).
        graph.ResetKVCache();
        var hAll = new float[prompt.Length * feat];
        var logits = new float[vocab];
        trunk.SpecForward(prompt, hAll, logits, allLogitsRows: false);
        var shifted = new float[hAll.Length];
        Array.Copy(hAll, 0, shifted, feat, hAll.Length - feat);
        foreach (IDraftHead d in drafters) d.DraftCatchUp(prompt, shifted, 0);
        float[] pending = hAll[^feat..];
        int last = Argmax(logits, 0, vocab), position = prompt.Length;
        var mismatches = new List<string>();
        double worstConfidence = 0, worstRing = 0;
        worstRing = Math.Max(worstRing, CompareRings(graph, perOp, label, "prompt", mismatches));

        for (int cycle = 0; cycle < Cycles; cycle++)
        {
            var ids = new int[2][];
            var conf = new float[2][];
            for (int i = 0; i < 2; i++)
            {
                ids[i] = new int[gamma];
                conf[i] = new float[gamma];
                Assert.Equal(gamma, drafters[i].DraftBlock(last, pending, position, ids[i], conf[i]));
            }
            for (int e = 0; e < gamma; e++)
            {
                if (ids[0][e] != ids[1][e])
                    mismatches.Add($"cycle {cycle} slot {e}: graph drafted {ids[0][e]} ({conf[0][e]:F5}), per-op {ids[1][e]} ({conf[1][e]:F5})");
                worstConfidence = Math.Max(worstConfidence, Math.Abs(conf[0][e] - conf[1][e]));
                if (Math.Abs(conf[0][e] - conf[1][e]) > ConfidenceTolerance)
                    mismatches.Add($"cycle {cycle} slot {e}: confidence {conf[0][e]:F6} against the per-op {conf[1][e]:F6}");
            }
            worstRing = Math.Max(worstRing, CompareRings(graph, perOp, label, $"cycle {cycle} draft", mismatches));

            // Verify [last, drafts] on the trunk and keep an accept count that walks 0 .. gamma, so the catch-ups take
            // every width (the tokens kept are the window's whatever the trunk predicted: the drafters see one stream).
            int[] window = new[] { last }.Concat(ids[0]).ToArray();
            var hWindow = new float[window.Length * feat];
            var windowLogits = new float[window.Length * vocab];
            trunk.SpecForward(window, hWindow, windowLogits, allLogitsRows: true);
            int accepted = cycle % (gamma + 1);
            trunk.SpecOnVerifyAccepted(accepted, gamma);
            trunk.SpecRewindCache(position + accepted + 1);
            var catchUp = new float[(accepted + 1) * feat];
            Array.Copy(pending, catchUp, feat);
            Array.Copy(hWindow, 0, catchUp, feat, (long)accepted * feat);
            foreach (IDraftHead d in drafters) d.DraftCatchUp(window[..(accepted + 1)], catchUp, position);
            worstRing = Math.Max(worstRing, CompareRings(graph, perOp, label, $"cycle {cycle} commit", mismatches));

            pending = hWindow[(accepted * feat)..((accepted + 1) * feat)];
            last = Argmax(windowLogits, (long)accepted * vocab, vocab);
            position += accepted + 1;
        }

        _output.WriteLine($"{label}: {Cycles} blocks to position {position}; worst confidence difference {worstConfidence:E2}, " +
            $"worst ring relative L2 {worstRing:E2}; {graph.GraphBlockDrafts} graph blocks, {graph.GraphDFlashInjects} graph injects, " +
            $"{graph.DraftAdvances} advanced draft steps");
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Null(graph.GraphBlockDraftDeclineReason);
        Assert.Equal(Cycles, graph.GraphBlockDrafts);
        Assert.True(graph.GraphDFlashInjects > 0, $"{label}: no ring inject ran as a graph");
        // Where the executor replays its plans, the family's kept steps advance: later blocks and one-row injects ran as the
        // kept step moved on, their ring rows, mask and anchor refilled (MLX's executor runs each step as recorded instead).
        if (graph.GraphExecutorForTests.Capabilities.Replays)
            Assert.True(graph.DraftAdvances > 0, $"{label}: no draft step advanced");
        Assert.Equal(0, perOp.GraphBlockDrafts);
        Assert.Equal(0, perOp.GraphDFlashInjects);
    }

    /// <summary>
    /// The two rings' rows for the positions a draft can attend (the sliding window behind the frontier; K and V of every
    /// draft layer); returns the worst relative L2. Older positions are not compared: a block writes its own rows into
    /// the slots of positions a ring's length behind it, which are past every query's window (as the fused graph's block
    /// does), and the per-op drafter leaves them.
    /// </summary>
    private static double CompareRings(ModelBase graph, ModelBase perOp, string label, string when, List<string> mismatches)
    {
        double worst = 0;
        DFlashConfig cfg = graph.DFlashSettings;
        int row = cfg.NumKVHeads * cfg.HeadDim;
        for (int layer = 0; layer < cfg.NumLayers; layer++)
        {
            var (gk, gv) = graph.DFlashRingContextForTests(layer, out int gFirst);
            var (pk, pv) = perOp.DFlashRingContextForTests(layer, out int pFirst);
            Assert.True(gFirst == pFirst && gk.Length == pk.Length, $"{label} {when}: the rings hold different positions");
            int held = gk.Length / row, window = Math.Min(held, cfg.SlidingWindow);
            Assert.True(window > 0, $"{label} {when}: the ring holds no context");
            foreach ((float[] a, float[] b, string what) in new[] { (gk, pk, "K"), (gv, pv, "V") })
            {
                Assert.Contains(b[((held - window) * row)..], value => value != 0);
                double rel = RelativeL2(a[((held - window) * row)..], b[((held - window) * row)..]);
                worst = Math.Max(worst, rel);
                if (rel > RingTolerance) mismatches.Add($"{when}: layer {layer} {what} relative L2 {rel:E2}");
            }
        }
        return worst;
    }

    private static double RelativeL2(float[] a, float[] b)
    {
        double diff = 0, norm = 0;
        for (int i = 0; i < a.Length; i++)
        {
            diff += (double)(a[i] - b[i]) * (a[i] - b[i]);
            norm += (double)b[i] * b[i];
        }
        return norm == 0 ? Math.Sqrt(diff) : Math.Sqrt(diff / norm);
    }

    private static int Argmax(float[] values, long offset, int count)
    {
        int best = 0;
        for (int i = 1; i < count; i++)
            if (values[offset + i] > values[offset + best]) best = i;
        return best;
    }
}
