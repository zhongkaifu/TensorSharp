// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Muse-Glimmer's speculation on the shared runtime (stage S5 of docs/design/muse-glimmer-unified.md): the family declares
// nothing of a drafter; the DFlash block drafter (StandardDecoderModel.BlockDraft.cs over ModelBase.DFlash*), n-gram drafts,
// the batched verify over slots and the cost governor are the runtime's, once for every family. These tests hold the family to
// that layer on the goldens' untied fixture (8 blocks: 6 sliding-window, 2 full NoPE; a real logit scale and a head that
// reaches the softcap) with a window of 8 keys, and a synthetic DFlash drafter (blocks of 4, a window of 16, so its 32-row
// ring wraps): speculation emits plain greedy decoding's stream on every executor, alone and for several requests at once
// (continuous batching, which legacy lacked), across a whole-layer split, and the drafter is declined under --tp (D5).
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class MuseGlimmerRuntimeSpeculationTests(ITestOutputHelper output) : IDisposable
{
    private const int Window = 8;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-mg-spec-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>The goldens' untied fixture with a window of <see cref="Window"/> keys, and the DFlash drafter (target layers 2,
    /// 5 and 7, so features of a sliding-window layer, a full NoPE layer's neighbour and the last layer's input).</summary>
    private (string Target, string Draft) Files() => WriteFiles(_dir);

    /// <summary><see cref="Files"/> into <paramref name="dir"/> (the frozen legacy features came from the same fixtures).</summary>
    internal static (string Target, string Draft) WriteFiles(string dir)
    {
        Directory.CreateDirectory(dir);
        string target = new MuseGlimmerSyntheticModelBuilder
        {
            BlockCount = 8, SlidingWindow = Window, LogitScale = 0.19612f, Softcap = 20f, HeadScale = 3f, BlockOutputScale = 0.004f,
        }.Write(Path.Combine(dir, "muse-glimmer.gguf"));
        string draft = Qwen4ExpSyntheticModelBuilder.WriteDFlash(Path.Combine(dir, "dflash.gguf"), blockSize: 4,
            hidden: MuseGlimmerSyntheticModelBuilder.Hidden, slidingWindow: 16);
        return (target, draft);
    }

    private static StandardDecoderModel Create(string target, BackendType backend, string draft = null, int tp = 1, int stages = 1)
        => (StandardDecoderModel)MuseGlimmerArchitecture.CreateRuntimeModel(target, backend, tp,
            tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null, stages, draft);

    private static BackendType BackendOf(string executor) => executor switch
    {
        "managed" => BackendType.Cpu,
        "ggml" => TestGates.PinnedGgmlBackend,
        "mlx" => BackendType.Mlx,
        "cuda" => BackendType.Cuda,
        _ => throw new ArgumentOutOfRangeException(nameof(executor)),
    };

    /// <summary>Where a verify window's rows and a lone decode row take other products, a DFlash stream may part from plain
    /// greedy's only at a near-tie: plain's top two logits within this share. That covers ggml's CPU F16 dot products,
    /// ggml-cuda, MLX's batched kernels, and the CUDA executor. The CUDA executor sends this fixture's F16 weights to cuBLAS
    /// at every row count, and its rows round one way for one row and another for two or more. On an RTX PRO 4500 a verify
    /// window's rows are 3-8e-4 of the largest logit from the same rows decoded alone. Every width from 2 to 8 gives the
    /// same row, and a one-row window gives the decode step's bits. Plain greedy's token 7 here is a 9.3e-5 tie on that
    /// executor.
    /// The exact rule stays on two backends:
    /// <list type="bullet">
    /// <item>ggml_metal, whose kernels are row-shape invariant.</item>
    /// <item>The managed CPU. A window's rows are the decode step's bits up to four rows there, and within 1e-6 of the
    /// largest logit past that.</item>
    /// </list></summary>
    private static double TieTolerance(BackendType backend)
        => backend is BackendType.GgmlCpu or BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.Mlx or BackendType.Cuda
            ? 1e-2 : 0;

    /// <summary>The engine's near-tie bound (a share of the largest logit), for the GGML backends and MLX. The engine tests on
    /// the CUDA executor still ask for plain decoding's stream token for token.</summary>
    private static double EngineNearTie(BackendType backend)
        => backend is BackendType.GgmlCpu or BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.Mlx ? 5e-3 : 0;

    internal static readonly int[] LongPrompt = Enumerable.Range(0, 40).Select(i => 40 + (i * 7) % 23).ToArray();

    // ---------------------------------------------------------------- DFlash: the block drafter

    /// <summary>
    /// Greedy DFlash on the runtime family (the drafter's graph programs where the executor runs them, its per-op path
    /// elsewhere) emits plain greedy decoding's stream: a 40-token prompt (five windows), then 48 tokens drafted and verified a
    /// whole block at a time, so the drafter's 32-row ring wraps and its window slides. This is the real-weight gate's rule
    /// (MuseGlimmerParityTests.DFlashDraftingIsLossless), with the near-tie as a share of the largest logit
    /// (<see cref="TieTolerance"/>):
    /// <list type="bullet">
    /// <item>The DFlash stream is plain greedy's.</item>
    /// <item>Or it first leaves plain greedy where plain's top two are a near-tie, by taking plain's runner-up there.</item>
    /// <item>Then it agrees again under teacher forcing. Resumed from the prompt and plain's tokens through that position, it
    /// is plain's continuation, or it leaves it again only at another such near-tie.</item>
    /// <item>A parting that is not a near-tie still passes when the stream's token is the argmax of the verify row that drew it
    /// in this run (<see cref="VerifyWindowLog"/>), as in the real-weight gate.</item>
    /// <item>Over the 48 tokens, the stream takes plain greedy's token at <see cref="VerifyWindowLog.MinAgreement"/> of the
    /// positions or more: a parting is one position where it does not.</item>
    /// </list>
    /// </summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void DFlashMatchesPlainGreedy(string executor) => AssertDFlashMatchesPlainGreedy(BackendOf(executor));

    [MlxFact]
    public void DFlashMatchesPlainGreedyOnMlx() => AssertDFlashMatchesPlainGreedy(BackendType.Mlx);

    [CudaFact]
    public void DFlashMatchesPlainGreedyOnDirectCuda() => AssertDFlashMatchesPlainGreedy(BackendType.Cuda);

    private void AssertDFlashMatchesPlainGreedy(BackendType backend)
    {
        const int maxNew = 48;
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Files();
        double tolerance = TieTolerance(backend);
        string label = $"muse-glimmer {backend} dflash";

        // Plain greedy, one row a step: each token with its runner-up and their relative margin.
        var plain = new List<(int Token, int RunnerUp, double Margin)>();
        using (StandardDecoderModel model = Create(target, backend, draft))
        {
            float[] logits = model.ForwardRefill(LongPrompt);
            for (int i = 0; i < maxNew; i++)
            {
                int best = ArgMax(logits), second = -1;
                for (int v = 0; v < logits.Length; v++)
                    if (v != best && (second < 0 || logits[v] > logits[second])) second = v;
                plain.Add((best, second, (logits[best] - logits[second]) / Math.Max(Math.Abs(logits[best]), 1e-6)));
                logits = model.Forward(new[] { best });
            }
        }

        // DFlash from the prompt, then (past each parting) from the prompt and plain's tokens through the parting.
        var partings = new List<string>();
        int from = 0, blocks = 0, injects = 0;
        long verifySteps = 0, drafted = 0;
        string declined = null;
        while (from < maxNew)
        {
            int[] prompt = LongPrompt.Concat(plain.Take(from).Select(p => p.Token)).ToArray();
            using StandardDecoderModel model = Create(target, backend, draft);
            var spec = Assert.IsAssignableFrom<ISpeculativeModel>(model);
            Assert.Null(spec.SpeculationRefusal);
            Assert.Equal(DraftHeadKind.Block, spec.DraftHeadKind);
            // Gate 0 and no cost governor: every step drafts the whole block, whatever the (random) drafter's confidence.
            var speculator = new BlockDraftSpeculator(spec, spec.DraftBlockSize, spec.DraftBlockSize) { MinDraftProb = 0 };
            var decoder = new SpeculativeDecoder(spec, speculator) { AdaptiveSpeculation = false, PrefillChunkSize = prompt.Length };
            var windows = new VerifyWindowLog(decoder, model.Config.VocabSize);
            List<int> stream = decoder.GenerateGreedy(prompt, maxNew - from);
            Assert.Equal(maxNew - from, stream.Count);
            verifySteps += decoder.VerifySteps;
            drafted += decoder.TokensDrafted;
            blocks += model.GraphBlockDrafts;
            injects += model.GraphDFlashInjects;
            declined ??= model.GraphBlockDraftDeclineReason;
            int d = 0;
            while (d < stream.Count && stream[d] == plain[from + d].Token) d++;
            if (d == stream.Count) break;
            var at = plain[from + d];
            string parting = $"token {from + d}: {stream[d]} for plain's {at.Token} (relative margin {at.Margin:E2})";
            if (!(tolerance > 0 && at.Margin <= tolerance && stream[d] == at.RunnerUp))
            {
                // Not a near-tie: the token must be the argmax of the verify row that drew it in this run.
                string what = $"{label}: token {from + d} is {stream[d]}, plain greedy {at.Token} (runner-up {at.RunnerUp}, relative " +
                    $"margin {at.Margin:E2}{(from > 0 ? $"; resumed at {from}" : "")}), not a near-tie";
                VerifyWindowLog.Row? row = windows.Drawing(prompt.Length + d);
                Assert.True(row != null, $"{what}, and no verify row drew it");
                string over = VerifyWindowLog.Describe(row.Value, stream[d], at.Token);
                Assert.True(row.Value.Top[0].Id == stream[d], $"{what}, and {over}");
                parting += "; " + over;
            }
            partings.Add(parting);
            from += d + 1;
        }
        Assert.True(verifySteps > 0, $"{label}: no verify ran");
        Assert.True(drafted > 0, $"{label}: the drafter drafted nothing");
        double agreement = 1.0 - (double)partings.Count / maxNew;
        Assert.True(agreement >= VerifyWindowLog.MinAgreement,
            $"{label}: {partings.Count} parting(s) in {maxNew} tokens, an agreement of {agreement:P1} with plain greedy, under {VerifyWindowLog.MinAgreement:P0}");
        output.WriteLine($"{backend}: {(partings.Count == 0 ? "plain greedy's stream" : $"{partings.Count} parting(s), " +
            $"each agreeing again under teacher forcing: {string.Join("; ", partings)}")}; {blocks} blocks and {injects} injects " +
            $"as programs; declined: {declined ?? "no"}");
        Assert.Null(declined);
    }

    /// <summary>
    /// The graph drafter drafts what the per-op drafter does over the family's trunk rows (the features are the residuals
    /// entering the target layers, after the weightless input norm): the same ids, confidences within 1e-3, through draft /
    /// verify / commit cycles whose accept counts walk every catch-up width. The confidences are the softmax of the head's raw
    /// logits, without the trunk's logit scale (0.19612 here) or its softcap (llama.cpp's dflash graph), on both paths.
    /// </summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void TheGraphDrafterDraftsAsThePerOpDrafter(string executor)
    {
        BackendType backend = BackendOf(executor);
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Files();
        using StandardDecoderModel graph = Create(target, backend, draft);
        using StandardDecoderModel perOp = Create(target, backend, draft);
        graph.GraphBlockDraftOnMetalForTests = true;
        perOp.GraphBlockDraftDisabledForTests = true;
        Assert.True(graph.HasDFlash && perOp.HasDFlash);
        var trunk = (ISpeculativeModel)graph;
        var drafters = new IDraftHead[] { graph, perOp };
        int feat = trunk.SpecFeatureSize, vocab = graph.Config.VocabSize, gamma = trunk.DraftBlockSize;
        Assert.Equal(3 * MuseGlimmerSyntheticModelBuilder.Hidden, feat);

        var hAll = new float[LongPrompt.Length * feat];
        var logits = new float[vocab];
        trunk.SpecForward(LongPrompt, hAll, logits, allLogitsRows: false);
        var shifted = new float[hAll.Length];
        Array.Copy(hAll, 0, shifted, feat, hAll.Length - feat);
        foreach (IDraftHead d in drafters) d.DraftCatchUp(LongPrompt, shifted, 0);
        float[] pending = hAll[^feat..];
        int lastToken = ArgMax(logits), position = LongPrompt.Length;
        double worst = 0;
        int drafted = 0;
        for (int cycle = 0; cycle < 16; cycle++)
        {
            var ids = new int[2][];
            var conf = new float[2][];
            for (int i = 0; i < 2; i++)
            {
                ids[i] = new int[gamma];
                conf[i] = new float[gamma];
                Assert.Equal(gamma, drafters[i].DraftBlock(lastToken, pending, position, ids[i], conf[i]));
            }
            Assert.True(ids[0].SequenceEqual(ids[1]), $"cycle {cycle}: graph drafted [{string.Join(",", ids[0])}], per-op [{string.Join(",", ids[1])}]");
            for (int e = 0; e < gamma; e++) worst = Math.Max(worst, Math.Abs(conf[0][e] - conf[1][e]));
            drafted += gamma;

            int[] window = new[] { lastToken }.Concat(ids[0]).ToArray();
            var hWindow = new float[window.Length * feat];
            var windowLogits = new float[window.Length * vocab];
            trunk.SpecForward(window, hWindow, windowLogits, allLogitsRows: true);
            int accepted = cycle % (gamma + 1);
            trunk.SpecOnVerifyAccepted(accepted, gamma);
            trunk.SpecRewindCache(position + accepted + 1);
            var catchUp = new float[(accepted + 1) * feat];
            Array.Copy(pending, catchUp, feat);
            Array.Copy(hWindow, 0, catchUp, feat, (long)accepted * feat);
            int[] keep = window.Take(accepted + 1).ToArray();
            foreach (IDraftHead d in drafters) d.DraftCatchUp(keep, catchUp, position);
            pending = hWindow.AsSpan(accepted * feat, feat).ToArray();
            position += accepted + 1;
            lastToken = ArgMax(windowLogits.AsSpan(accepted * vocab, vocab).ToArray());
        }
        output.WriteLine($"{backend}: {drafted} drafts equal, worst confidence difference {worst:E2}; graph blocks {graph.GraphBlockDrafts}, " +
            $"per-op {perOp.GraphBlockDrafts}");
        Assert.True(worst <= 1e-3, $"confidences {worst:E2} apart");
        Assert.True(graph.GraphBlockDrafts > 0, $"no block ran as a program ({graph.GraphBlockDraftDeclineReason})");
        Assert.Equal(0, perOp.GraphBlockDrafts);
    }

    /// <summary>The runtime's drafter features (the residual entering each target layer) are legacy's, to the families'
    /// parity bound: the prompt's every row, on the managed CPU, against legacy's features frozen there before stage S9
    /// deleted it (<see cref="MuseGlimmerLegacySynthetic"/>).</summary>
    [Fact]
    public void TheDrafterFeaturesAreLegacys()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Files();
        var frozen = MuseGlimmerLegacySynthetic.Frozen;
        Assert.Equal(frozen.TargetSha256, MuseGlimmerGoldens.FileSha(target));
        Assert.Equal(frozen.DraftSha256, MuseGlimmerGoldens.FileSha(draft));
        float[] runtime;
        using (ModelBase model = Create(target, BackendType.Cpu, draft))
        {
            var spec = (ISpeculativeModel)model;
            runtime = new float[LongPrompt.Length * spec.SpecFeatureSize];
            spec.SpecForward(LongPrompt, runtime, new float[model.Config.VocabSize], allLogitsRows: false);
        }
        var (relL2, relMax) = MuseGlimmerGoldens.Distance(frozen.DrafterFeatures, runtime);
        output.WriteLine($"features of {LongPrompt.Length} rows: relL2 {relL2:E2}, largest difference {relMax:E2} of the largest");
        Assert.True(relL2 <= MuseGlimmerRuntimeTests.ParityBound && relMax <= MuseGlimmerRuntimeTests.ParityBound, $"relL2 {relL2:E2}, max {relMax:E2}");
    }

    /// <summary>
    /// A speculating request's prompt chunk longer than a ring step (24 rows over the tests' 8-row steps: three rounds) reaches
    /// its DFlash ring whole, as the same prompt in chunks of one step each does: every round's rows (the ring's frontier is the
    /// prompt's end, its rows the same) and the prompt's last feature row held as the first draft's carry. The registration the
    /// first round took had left the later rounds' rows out (the 30B's prompts past 512 rows lost their tail).
    /// </summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void APromptChunkPastARingStepReachesTheDrafterWhole(string executor)
    {
        BackendType backend = BackendOf(executor);
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Files();
        int[] prompt = LongPrompt[..24];
        StandardDecoderModel.RingGeometryForTests.Value = (8, 8);
        try
        {
            (int, int)? run(StandardDecoderModel model, string id) => model.DFlashRingRunForTests(id) is { } r ? (r.ValidFrom, r.Filled) : null;
            using StandardDecoderModel whole = Create(target, backend, draft);
            using StandardDecoderModel stepped = Create(target, backend, draft);
            Assert.Equal(8, whole.MaxSequenceStepRows);
            var logits = new float[1][];
            whole.EnsureSequenceSlot("whole", prompt.Length + 1);
            whole.FollowPlainPrompt("whole", prompt.Length);
            Assert.True(whole.TryForwardUnified(new[] { "whole" }, new[] { prompt }, new[] { 0 }, logits));
            stepped.EnsureSequenceSlot("stepped", prompt.Length + 1);
            for (int at = 0; at < prompt.Length; at += 8)
            {
                stepped.FollowPlainPrompt("stepped", prompt.Length);
                Assert.True(stepped.TryForwardUnified(new[] { "stepped" }, new[] { prompt[at..(at + 8)] }, new[] { at }, logits));
            }
            Assert.Equal(((int, int)?)(0, prompt.Length), run(stepped, "stepped"));
            Assert.Equal(((int, int)?)(0, prompt.Length), run(whole, "whole"));
            int feat = ((ISpeculativeModel)whole).SpecFeatureSize;
            var carry = new float[2][];
            for (int i = 0; i < 2; i++)
            {
                StandardDecoderModel model = i == 0 ? whole : stepped;
                string id = i == 0 ? "whole" : "stepped";
                model.BindDraftState(id);
                carry[i] = new float[feat];
                Assert.True(((IDraftHead)model).TryTakeHeldCarry(prompt.Length, carry[i]), $"{id}: no carry held after the prompt");
            }
            double worst = MuseGlimmerGoldens.Distance(carry[1], carry[0]).RelL2;
            for (int layer = 0; layer < whole.DFlashSettings.NumLayers; layer++)
            {
                var (wk, wv) = whole.DFlashRingContextForTests("whole", layer, out int wFirst);
                var (sk, sv) = stepped.DFlashRingContextForTests("stepped", layer, out int sFirst);
                Assert.Equal(sFirst, wFirst);
                Assert.Equal(sk.Length, wk.Length);
                worst = Math.Max(worst, Math.Max(MuseGlimmerGoldens.Distance(sk, wk).RelL2, MuseGlimmerGoldens.Distance(sv, wv).RelL2));
            }
            output.WriteLine($"{backend}: the ring holds 0..{prompt.Length}; ring rows and carry within {worst:E2} of the stepped prompt's");
            Assert.True(worst <= 1e-5, $"{backend}: {worst:E2} apart");
        }
        finally
        {
            StandardDecoderModel.RingGeometryForTests.Value = null;
        }
    }

    // ---------------------------------------------------------------- through the engine: one request and several at once

    /// <summary>
    /// Speculation through the engine's unified step (DFlash blocks over each request's own ring, or n-gram drafts) emits what
    /// plain decoding does, for one request and for three at once: each drafts over its own drafter state and every window is
    /// verified in one forward over the requests' slots (the batched verify ran). Legacy served one request at a time.
    /// </summary>
    [Theory]
    [InlineData("managed", SpeculatorRegistry.Block)]
    [InlineData("managed", SpeculatorRegistry.NGram)]
    [InlineData("ggml", SpeculatorRegistry.Block)]
    [InlineData("ggml", SpeculatorRegistry.NGram)]
    public Task EngineSpeculationMatchesPlain(string executor, string speculator) => AssertEngine(BackendOf(executor), speculator);

    [MlxTheory]
    [InlineData(SpeculatorRegistry.Block)]
    [InlineData(SpeculatorRegistry.NGram)]
    public Task EngineSpeculationMatchesPlainOnMlx(string speculator) => AssertEngine(BackendType.Mlx, speculator);

    [CudaTheory]
    [InlineData(SpeculatorRegistry.Block)]
    [InlineData(SpeculatorRegistry.NGram)]
    public Task EngineSpeculationMatchesPlainOnDirectCuda(string speculator) => AssertEngine(BackendType.Cuda, speculator);

    private Task AssertEngine(BackendType backend, string speculator, int stages = 1)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Files();
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = speculator, MaxDraftTokens = 3, MinDraftProb = 0 };
        return SpeculativeRuntimeChecks.AssertEngineSpeculationMatchesPlain(
            () => Create(target, backend, speculator == SpeculatorRegistry.Block ? draft : null, stages: stages),
            options, $"muse-glimmer {backend} {speculator} stages {stages}", nearTie: EngineNearTie(backend));
    }

    /// <summary>n-gram drafts need no drafter: the family speculates with a weight-free speculator, and the trunk's verify and
    /// rollback (drafts both accepted and rejected) emit plain greedy decoding's stream.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void ScriptedDraftsMatchPlainGreedy(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, _) = Files();
        BackendType backend = BackendOf(executor);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(target, backend), $"muse-glimmer {backend} scripted",
            tieTolerance: TieTolerance(backend));
    }

    // ---------------------------------------------------------------- placements

    /// <summary>
    /// DFlash across a whole-layer split (two stages on simulated devices, as staged programs; the last stage holds the head,
    /// the drafter keeps its per-op path there) emits one device's plain greedy stream: new for the family (legacy refused
    /// --layer-split). Per op, the runtime speculates on one device only, and says so.
    /// </summary>
    [Fact]
    public void DFlashAcrossALayerSplitMatchesOneDevice()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Files();
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() =>
            {
                StandardDecoderModel model = Create(target, BackendType.Cpu, draft, stages: 2);
                Assert.True(model.HasDFlash, "the drafter did not attach to the layer split");
                return model;
            }, "muse-glimmer layer split 2 dflash", maxNew: 32, prompt: LongPrompt, reference: () => Create(target, BackendType.Cpu));

        using StandardDecoderModel perOp = Create(target, BackendType.Cpu, draft, stages: 2);
        perOp.GraphLoweringEnabled = false;
        string refusal = ((ISpeculativeTarget)perOp).SpeculationRefusal;
        output.WriteLine($"per op: {refusal}");
        Assert.Contains("runs on one device", refusal ?? "", StringComparison.Ordinal);
    }

    /// <summary>The engine's unified step over the layer split, one request and three at once.</summary>
    [Fact]
    public Task EngineDFlashAcrossALayerSplitMatchesPlain() => AssertEngine(BackendType.Cpu, SpeculatorRegistry.Block, stages: 2);

    /// <summary>Under --tp the drafter is declined, as legacy declined it (design D5): it borrows the head and embedding the
    /// ranks shard. The model loads without it (the host's attach then refuses it as the model's own decline: the server exits
    /// 2, the CLI warns), and still speculates with n-gram drafts.</summary>
    [Fact]
    public void TheDrafterIsDeclinedUnderTensorParallelism()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Files();
        using StandardDecoderModel model = Create(target, BackendType.Cpu, draft, tp: 2);
        Assert.False(model.HasDFlash);
        Assert.True(model.TensorParallelActive);
        Assert.Equal(DraftHeadKind.None, ((IDraftHead)model).DraftHeadKind);
        Assert.Null(((ISpeculativeTarget)model).SpeculationRefusal);
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }
}
