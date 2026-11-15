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

    // ---------------------------------------------------------------- the trunk keeps the ring (DraftSelfCatchUp)

    /// <summary>Speculation where the trunk's own steps write the ring (the prompt read without a feature row on the host:
    /// the standalone decoder's prompt step returns only its last row's) emits plain greedy decoding's stream, and the
    /// prompt's rows reached the ring through the trunk's graph.</summary>
    [Theory]
    [InlineData("qwen35", "dflash", BackendType.Cpu)]
    [InlineData("qwen35", "dflash2", BackendType.Cpu)]
    [InlineData("qwen4exp", "dflash2", BackendType.Cpu)]
    public void SelfCatchUpSpeculationMatchesPlainGreedy(string family, string variant, BackendType backend)
        => SelfCatchUpMatchesPlainGreedy(family, variant, backend);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("qwen35", "dflash", BackendType.GgmlCpu)]
    [InlineData("qwen35", "dflash2", BackendType.GgmlCpu)]
    [InlineData("qwen4exp", "dflash2", BackendType.GgmlCpu)]
    public void SelfCatchUpSpeculationMatchesPlainGreedyOnGgmlCpu(string family, string variant, BackendType backend)
        => SelfCatchUpMatchesPlainGreedy(family, variant, backend);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("qwen35", "dflash", BackendType.GgmlMetal)]
    [InlineData("qwen35", "dflash2", BackendType.GgmlMetal)]
    public void SelfCatchUpSpeculationMatchesPlainGreedyOnMetal(string family, string variant, BackendType backend)
        => SelfCatchUpMatchesPlainGreedy(family, variant, backend);

    [MlxTheory]
    [InlineData("qwen35", "dflash", BackendType.Mlx)]
    [InlineData("qwen35", "dflash2", BackendType.Mlx)]
    public void SelfCatchUpSpeculationMatchesPlainGreedyOnMlx(string family, string variant, BackendType backend)
        => SelfCatchUpMatchesPlainGreedy(family, variant, backend);

    /// <summary>Where the drafter keeps its other path (ggml_metal's fused DFlash2 selector, the per-op drafter), the trunk
    /// does not write the ring: the prompt goes through the speculative context, as before.</summary>
    [Fact]
    public void RefusedGraphDrafterKeepsTheContextPath() => RefusedGraphDrafterKeepsTheContext(BackendType.Cpu, "per-op drafter");

    [GgmlFact(BackendType.GgmlMetal)]
    public void FusedMetalSelectorKeepsTheContextPath() => RefusedGraphDrafterKeepsTheContext(BackendType.GgmlMetal, "ggml_metal fused selector");

    private void RefusedGraphDrafterKeepsTheContext(BackendType backend, string why)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        using StandardDecoderModel model = Create(Files("qwen35", "dflash2"), backend, graphOnMetal: false);
        if (backend == BackendType.Cpu) model.GraphBlockDraftDisabledForTests = true;
        Assert.True(model.HasDFlash, why);
        Assert.False(((IDraftHead)model).DraftSelfCatchUp, $"{why}: the trunk writes the ring of a drafter on its other path");
        var execution = new SpeculativeExecution((ISpeculativeModel)model,
            new BlockDraftSpeculator((IDraftHead)model, model.DraftBlockSize, model.DraftBlockSize));
        Assert.True(execution.PrefillsThroughContext, $"{why}: the prompt does not go through the context");
        Assert.False(execution.PrefillSelfCatchUp);
        // A prompt registered to follow writes nothing.
        var head = (ISpeculativeModel)model;
        head.FollowPlainPrompt(null, 21);
        model.ForwardPrompt(Enumerable.Range(0, 21).Select(i => 40 + (i * 7) % 23).ToArray(), morePromptFollows: false);
        Assert.Equal(0, model.GraphDFlashFollows);
    }

    /// <summary>The prompt through the speculative context (the path every backend the graph drafter refuses keeps) still
    /// emits plain greedy decoding's stream, and no trunk step wrote the ring.</summary>
    [Theory]
    [InlineData("qwen35", "dflash2")]
    [InlineData("qwen4exp", "dflash2")]
    public void ContextCatchUpSpeculationMatchesPlainGreedy(string family, string variant)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var files = Files(family, variant);
        StandardDecoderModel created = null;
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() =>
            {
                created = Create(files, BackendType.Cpu);
                created.DFlashSelfCatchUpDisabledForTests = true;
                return created;
            }, $"{family} cpu {variant} through the context");
        Assert.False(((IDraftHead)created).DraftSelfCatchUp);
        Assert.Equal(0, created.GraphDFlashFollows);
        Assert.True(created.GraphBlockDrafts > 0, $"{family} {variant}: no block ran as a graph");
    }

    private void SelfCatchUpMatchesPlainGreedy(string family, string variant, BackendType backend)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var files = Files(family, variant);
        StandardDecoderModel created = null;
        // ggml_cpu rounds a verify window's rows otherwise than one row's: near-ties of plain greedy may part (as its other
        // speculation checks allow).
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() => created = Create(files, backend),
            $"{family} {backend} {variant} self catch-up", tieTolerance: backend == BackendType.Cpu ? 0 : 1e-2);
        Assert.True(((IDraftHead)created).DraftSelfCatchUp, $"{family} {variant}: the trunk does not keep the ring");
        Assert.Null(created.GraphBlockDraftDeclineReason);
        Assert.True(created.GraphDFlashFollows > 0, $"{family} {variant}: no trunk step wrote the ring");
        Assert.True(created.GraphBlockDrafts > 0, $"{family} {variant}: no block ran as a graph");
    }

    /// <summary>
    /// The engine's unified step with block speculation (prompts in 16-token chunks, one request and three at once: each
    /// step's chunks of several requests, each written into its own request's ring) emits what plain decoding does, its
    /// prompts reached the rings through the trunk's steps, and the batched verify ran.
    /// </summary>
    [Theory]
    [InlineData("qwen35", "dflash", BackendType.Cpu)]
    [InlineData("qwen35", "dflash2", BackendType.Cpu)]
    [InlineData("qwen4exp", "dflash2", BackendType.Cpu)]
    public Task SelfCatchUpEngineSpeculationMatchesPlain(string family, string variant, BackendType backend)
        => EngineSelfCatchUpMatchesPlain(family, variant, backend);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("qwen35", "dflash2", BackendType.GgmlCpu)]
    public Task SelfCatchUpEngineSpeculationMatchesPlainOnGgmlCpu(string family, string variant, BackendType backend)
        => EngineSelfCatchUpMatchesPlain(family, variant, backend);

    private async Task EngineSelfCatchUpMatchesPlain(string family, string variant, BackendType backend)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var files = Files(family, variant);
        var created = new List<StandardDecoderModel>();
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.Block, MaxDraftTokens = 3, MinDraftProb = 0 };
        await SpeculativeRuntimeChecks.AssertEngineSpeculationMatchesPlain(() =>
            {
                StandardDecoderModel model = Create(files, backend);
                created.Add(model);
                return model;
            }, options, $"{family} {backend} {variant} self catch-up");
        Assert.All(created, m => Assert.True(((IDraftHead)m).DraftSelfCatchUp));
        Assert.True(created.Exists(m => m.GraphDFlashFollows > 0), $"{family} {variant}: no trunk step wrote a ring");
        Assert.All(created, m => Assert.Null(m.GraphBlockDraftDeclineReason));
    }

    /// <summary>
    /// After a prompt read on the plain path whose steps write the ring (in one chunk, in several, longer than the 32-row
    /// ring, with the prompt's end known or not, and two requests' chunks in each step of the unified path), the ring
    /// holds the rows the speculative context's catch-up writes for the same prompt (every row's features to the host,
    /// token k paired with row k - 1), a null-row catch-up over the followed rows leaves it as it is, and the first block
    /// drafts the same ids.
    /// </summary>
    [Theory]
    [InlineData("dflash", "21", true, 1)]
    [InlineData("dflash2", "21", true, 1)]
    [InlineData("dflash2", "20,25", true, 1)]       // longer than the ring
    [InlineData("dflash2", "17,18,20", true, 1)]    // the first chunk older than the ring at the prompt's end
    [InlineData("dflash2", "17,18,20", false, 1)]   // the prompt's end unknown: every chunk's rows written
    [InlineData("dflash", "17,25", true, 2)]        // two requests' chunks in each unified step
    [InlineData("dflash2", "17,18,20", true, 2)]
    public void FollowedPromptRingMatchesTheContextsCatchUp(string variant, string chunks, bool knownEnd, int requests)
        => FollowedRingMatches(BackendType.Cpu, variant, chunks, knownEnd, requests);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("dflash2", "21", true, 1)]
    [InlineData("dflash2", "17,18,20", true, 1)]
    [InlineData("dflash2", "17,18,20", true, 2)]
    public void FollowedPromptRingMatchesTheContextsCatchUpOnGgmlCpu(string variant, string chunks, bool knownEnd, int requests)
        => FollowedRingMatches(BackendType.GgmlCpu, variant, chunks, knownEnd, requests);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("dflash", "21", true, 1)]
    [InlineData("dflash2", "17,18,20", true, 1)]
    [InlineData("dflash", "17,18,20", true, 2)]
    public void FollowedPromptRingMatchesTheContextsCatchUpOnMetal(string variant, string chunks, bool knownEnd, int requests)
        => FollowedRingMatches(BackendType.GgmlMetal, variant, chunks, knownEnd, requests);

    [MlxTheory]
    [InlineData("dflash", "21", true, 1)]
    [InlineData("dflash2", "17,18,20", true, 1)]
    [InlineData("dflash2", "17,18,20", true, 2)]
    public void FollowedPromptRingMatchesTheContextsCatchUpOnMlx(string variant, string chunks, bool knownEnd, int requests)
        => FollowedRingMatches(BackendType.Mlx, variant, chunks, knownEnd, requests);

    private void FollowedRingMatches(BackendType backend, string variant, string chunkList, bool knownEnd, int requests)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        int[] chunks = chunkList.Split(',').Select(int.Parse).ToArray();
        int n = chunks.Sum();
        string label = $"{backend} {variant} chunks {chunkList} end {(knownEnd ? "known" : "unknown")} x{requests}";
        var files = Files("qwen35", variant);
        using var followed = (Qwen35RuntimeModel)Create(files, backend);
        // Each request's reference, alive through the test (a model disposed mid-test would hand its host addresses to the
        // next one, whose GGML device copies are found by address).
        var references = new Qwen35RuntimeModel[requests];
        using var disposeReferences = new DisposeAll(references);
        for (int r = 0; r < requests; r++)
        {
            references[r] = (Qwen35RuntimeModel)Create(files, backend);
            references[r].DFlashSelfCatchUpDisabledForTests = true;
        }
        var head = (ISpeculativeModel)followed;
        Assert.True(head.DraftSelfCatchUp, $"{label}: the trunk does not keep the ring");
        int feat = head.SpecFeatureSize, vocab = followed.Config.VocabSize, gamma = head.DraftBlockSize, ring = followed.DFlashRingRowsForTests;
        int[][] prompts = Enumerable.Range(0, requests)
            .Select(r => Enumerable.Range(0, n).Select(i => 40 + (i * 7 + r * 5) % 23).ToArray()).ToArray();
        string[] ids = Enumerable.Range(0, requests).Select(r => $"req{r}").ToArray();
        var mismatches = new List<string>();

        // Followed: the plain prompt chunks, each registered first (FollowPlainPrompt), on the primary cache for one request
        // and on per-request slots, every request's chunk in one step, for two.
        var unified = (TensorSharp.Runtime.Scheduling.IUnifiedBatchModel)followed;
        var batched = (IBatchedSpeculativeTarget)followed;
        if (requests > 1)
            foreach (string id in ids) unified.EnsureSequenceSlot(id, n + 32);
        int start = 0;
        foreach (int size in chunks)
        {
            if (requests == 1)
            {
                head.FollowPlainPrompt(null, knownEnd ? n : -1);
                followed.ForwardPrompt(prompts[0][start..(start + size)], morePromptFollows: start + size < n);
            }
            else
            {
                foreach (string id in ids) head.FollowPlainPrompt(id, knownEnd ? n : -1);
                Assert.True(unified.TryForwardUnified(ids, prompts.Select(p => p[start..(start + size)]).ToArray(),
                    Enumerable.Repeat(start, requests).ToArray(), new float[requests][]));
            }
            start += size;
        }
        Assert.True(followed.GraphDFlashFollows > 0, $"{label}: no trunk step wrote the ring");

        for (int r = 0; r < requests; r++)
        {
            // Through the context: each prompt chunk's every row's features, the speculator's pairing and catch-up.
            Qwen35RuntimeModel caught = references[r];
            var reference = (ISpeculativeModel)caught;
            Assert.False(reference.DraftSelfCatchUp);
            var rows = new float[n * feat];
            var logits = new float[vocab];
            start = 0;
            foreach (int size in chunks)
            {
                var part = new float[size * feat];
                reference.SpecForward(prompts[r][start..(start + size)], part, logits, allLogitsRows: false);
                Array.Copy(part, 0, rows, (long)start * feat, part.Length);
                start += size;
            }
            var shifted = new float[rows.Length];
            Array.Copy(rows, 0, shifted, feat, rows.Length - feat);
            reference.DraftCatchUp(prompts[r], shifted, 0);
            Assert.Equal(0, caught.GraphDFlashFollows);

            using (batched.BeginBatchedSpeculation())
            {
                if (requests > 1) batched.BindDraftState(ids[r]);
                // The speculator's null-row catch-up over the followed prompt (the executor arms with it) reports no gap.
                head.DraftCatchUp(prompts[r], null, 0);
                CompareContexts(followed, caught, $"{label} request {r} prompt", mismatches, FollowTolerance(backend));

                // The first block, each drafter from the features of the prompt's last row (the speculative context's carry).
                float[] pending = rows[((n - 1) * feat)..];
                int last = Argmax(logits, 0, vocab);
                var draftIds = new int[2][];
                var draftConf = new float[2][];
                IDraftHead[] drafters = { followed, caught };
                for (int d = 0; d < 2; d++)
                {
                    draftIds[d] = new int[gamma];
                    draftConf[d] = new float[gamma];
                    Assert.Equal(gamma, drafters[d].DraftBlock(last, pending, n, draftIds[d], draftConf[d]));
                }
                for (int e = 0; e < gamma; e++)
                {
                    if (draftIds[0][e] != draftIds[1][e])
                        mismatches.Add($"request {r} slot {e}: followed drafted {draftIds[0][e]} ({draftConf[0][e]:F5}), caught up {draftIds[1][e]} ({draftConf[1][e]:F5})");
                    else if (Math.Abs(draftConf[0][e] - draftConf[1][e]) > ConfidenceTolerance)
                        mismatches.Add($"request {r} slot {e}: confidence {draftConf[0][e]:F6} against {draftConf[1][e]:F6}");
                }
                CompareRings(followed, caught, label, $"request {r} first block", mismatches, FollowTolerance(backend));
            }
        }
        _output.WriteLine($"{label}: ring {ring} rows, {followed.GraphDFlashFollows} followed chunks");
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Null(followed.GraphBlockDraftDeclineReason);
    }

    /// <summary>
    /// A conversation's second turn on the same cache (its prompt continues the first turn's rows, as a reused prefix
    /// does): the ring keeps the first turn as context (the row the first turn's last step left out is a zero row, as
    /// the speculative context's catch-up writes it), and every block of both turns drafts what the context's catch-up
    /// drafts, each trunk running its own drafts' verifies.
    /// </summary>
    [Theory]
    [InlineData("dflash", BackendType.Cpu)]
    [InlineData("dflash2", BackendType.Cpu)]
    public void SecondTurnDraftsAsTheContextsCatchUp(string variant, BackendType backend) => SecondTurnMatches(variant, backend);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("dflash2", BackendType.GgmlCpu)]
    public void SecondTurnDraftsAsTheContextsCatchUpOnGgmlCpu(string variant, BackendType backend) => SecondTurnMatches(variant, backend);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("dflash", BackendType.GgmlMetal)]
    [InlineData("dflash2", BackendType.GgmlMetal)]
    public void SecondTurnDraftsAsTheContextsCatchUpOnMetal(string variant, BackendType backend) => SecondTurnMatches(variant, backend);

    private void SecondTurnMatches(string variant, BackendType backend)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string label = $"{backend} {variant} second turn";
        var files = Files("qwen35", variant);
        using var followed = (Qwen35RuntimeModel)Create(files, backend);
        using var caught = (Qwen35RuntimeModel)Create(files, backend);
        caught.DFlashSelfCatchUpDisabledForTests = true;
        var models = new[] { followed, caught };
        var heads = new ISpeculativeModel[] { followed, caught };
        int feat = heads[0].SpecFeatureSize, vocab = followed.Config.VocabSize, gamma = heads[0].DraftBlockSize;
        var mismatches = new List<string>();
        var pending = new float[2][];
        var last = new int[2];
        int position = 0;

        // Each turn: its prompt, its first decode step plain (no carry before it, as the executor arms), then
        // draft / verify / commit cycles; the second turn's prompt continues the first turn's rows.
        int[] first = Enumerable.Range(0, 21).Select(i => 40 + (i * 7) % 23).ToArray();
        int[] second = Enumerable.Range(0, 18).Select(i => 44 + (i * 5) % 19).ToArray();
        foreach (int[] prompt in new[] { first, second })
        {
            int end = position + prompt.Length;
            for (int m = 0; m < 2; m++)
            {
                float[] logits;
                if (m == 0)
                {
                    heads[0].FollowPlainPrompt(null, end);
                    logits = followed.ForwardPrompt(prompt, morePromptFollows: false);
                    // The executor arms over the trunk's tokens with a null-row catch-up.
                    heads[0].DraftCatchUp(prompt, null, position);
                }
                else
                {
                    // A request armed after a reused prefix starts with a zero carry.
                    var rows = new float[prompt.Length * feat];
                    logits = new float[vocab];
                    heads[1].SpecForward(prompt, rows, logits, allLogitsRows: false);
                    var paired = new float[rows.Length];
                    Array.Copy(rows, 0, paired, feat, rows.Length - feat);
                    heads[1].DraftCatchUp(prompt, paired, position);
                    pending[1] = rows[((prompt.Length - 1) * feat)..];
                }
                last[m] = Argmax(logits, 0, vocab);
            }
            Assert.Equal(last[0], last[1]);
            CompareContexts(followed, caught, $"{label} prompt at {position}", mismatches, FollowTolerance(backend));
            position = end;

            // The first decode step runs plain and captures the carry: the followed head has none of the prompt's last row
            // on the host, the context's head commits it.
            for (int m = 0; m < 2; m++)
            {
                var carry = new float[feat];
                var logits = new float[vocab];
                heads[m].SpecForward(new[] { last[m] }, carry, logits, allLogitsRows: false);
                heads[m].DraftCatchUp(new[] { last[m] }, m == 0 ? null : pending[1], position);
                pending[m] = carry;
                last[m] = Argmax(logits, 0, vocab);
            }
            Assert.Equal(last[0], last[1]);
            position++;

            // Draft / verify / commit cycles, each model on its own trunk, accept counts walking 0 .. gamma.
            for (int cycle = 0; cycle < 6; cycle++)
            {
                var ids = new int[2][];
                var conf = new float[2][];
                for (int m = 0; m < 2; m++)
                {
                    ids[m] = new int[gamma];
                    conf[m] = new float[gamma];
                    Assert.Equal(gamma, heads[m].DraftBlock(last[m], pending[m], position, ids[m], conf[m]));
                }
                for (int e = 0; e < gamma; e++)
                    if (ids[0][e] != ids[1][e])
                        mismatches.Add($"position {position} slot {e}: followed drafted {ids[0][e]} ({conf[0][e]:F5}), caught up {ids[1][e]} ({conf[1][e]:F5})");
                    else if (Math.Abs(conf[0][e] - conf[1][e]) > ConfidenceTolerance)
                        mismatches.Add($"position {position} slot {e}: confidence {conf[0][e]:F6} against {conf[1][e]:F6}");
                CompareRings(followed, caught, label, $"block at {position}", mismatches, FollowTolerance(backend));
                int accepted = cycle % (gamma + 1);
                for (int m = 0; m < 2; m++)
                {
                    int[] window = new[] { last[m] }.Concat(ids[0]).ToArray();
                    var hWindow = new float[window.Length * feat];
                    var windowLogits = new float[window.Length * vocab];
                    heads[m].SpecForward(window, hWindow, windowLogits, allLogitsRows: true);
                    heads[m].SpecOnVerifyAccepted(accepted, gamma);
                    heads[m].SpecRewindCache(position + accepted + 1);
                    var catchUp = new float[(accepted + 1) * feat];
                    Array.Copy(pending[m], catchUp, feat);
                    Array.Copy(hWindow, 0, catchUp, feat, (long)accepted * feat);
                    heads[m].DraftCatchUp(window[..(accepted + 1)], catchUp, position);
                    pending[m] = hWindow[(accepted * feat)..((accepted + 1) * feat)];
                    last[m] = Argmax(windowLogits, (long)accepted * vocab, vocab);
                }
                Assert.Equal(last[0], last[1]);
                CompareRings(followed, caught, label, $"commit at {position}", mismatches, FollowTolerance(backend));
                position += accepted + 1;
            }
        }
        _output.WriteLine($"{label}: {followed.GraphDFlashFollows} followed chunks, ended at position {position}");
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Equal(2, followed.GraphDFlashFollows);
        Assert.Equal(0, caught.GraphDFlashFollows);
        Assert.Null(followed.GraphBlockDraftDeclineReason);
    }

    /// <summary>
    /// The followed ring's context against the context catch-up's over the positions both hold: the followed ring holds
    /// the prompt's last row too (its own features: a ring row needs no other), which the catch-up writes only with the
    /// first block's carry. Both runs must start at the same position (a restarted run would not).
    /// </summary>
    private static void CompareContexts(ModelBase followed, ModelBase caught, string when, List<string> mismatches, double tolerance)
    {
        DFlashConfig cfg = followed.DFlashSettings;
        int row = cfg.NumKVHeads * cfg.HeadDim;
        for (int layer = 0; layer < cfg.NumLayers; layer++)
        {
            var (fk, fv) = followed.DFlashRingContextForTests(layer, out int fFirst);
            var (ck, cv) = caught.DFlashRingContextForTests(layer, out int cFirst);
            int fEnd = fFirst + fk.Length / row, cEnd = cFirst + ck.Length / row;
            int lo = Math.Max(fFirst, cFirst), hi = Math.Min(fEnd, cEnd);
            if (fEnd - cEnd is < 0 or > 1)
                mismatches.Add($"{when}: the followed ring ends at {fEnd}, the caught-up one at {cEnd}");
            if (fFirst != cFirst && !(fEnd - fFirst == cEnd - cFirst && fFirst == cFirst + 1))
                mismatches.Add($"{when}: the followed ring's context starts at {fFirst}, the caught-up one's at {cFirst}");
            if (hi - lo <= 0)
            {
                mismatches.Add($"{when}: the rings share no context");
                continue;
            }
            foreach ((float[] a, float[] b, int aFirst, int bFirst, string what) in new[] { (fk, ck, fFirst, cFirst, "K"), (fv, cv, fFirst, cFirst, "V") })
            {
                float[] x = a[((lo - aFirst) * row)..((hi - aFirst) * row)], y = b[((lo - bFirst) * row)..((hi - bFirst) * row)];
                double rel = RelativeL2(x, y);
                if (rel > tolerance) mismatches.Add($"{when}: layer {layer} {what} relative L2 {rel:E2} over positions {lo}..{hi - 1}");
            }
        }
    }

    private sealed class DisposeAll(IDisposable[] items) : IDisposable
    {
        public void Dispose()
        {
            foreach (IDisposable item in items) item?.Dispose();
        }
    }

    /// <summary>
    /// The ring tolerance of a written-by-the-trunk ring against the context's: the trunk's rows are the same bits, but the
    /// drafter's products over them run at other row counts (the chunk's kept rows against the catch-up's, which the
    /// speculative context batches otherwise), and a GPU backend's products round by row count: on ggml_metal the inject
    /// program alone gives 2.6e-4 (relative L2 of the K rows) for 32 rows in one inject against 13 + 19.
    /// </summary>
    private static double FollowTolerance(BackendType backend) => backend is BackendType.Cpu or BackendType.GgmlCpu ? RingTolerance : 1e-3;

    /// <summary>The family's trunk and the drafter, written once a test (a model maps its file: rewriting it under a live
    /// model would pull its weights away).</summary>
    private (string Family, string Target, string Draft) Files(string family, string variant)
    {
        var (target, draft) = Fixtures(variant);
        if (family == "qwen4exp") target = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 16, qsa: true);
        return (family, target, draft);
    }

    /// <summary>The model, its drafter on the graph programs wherever they run (ggml_metal's DFlash2 selector too, where the
    /// fused drafter is kept for speed unless <paramref name="graphOnMetal"/> is false).</summary>
    private static StandardDecoderModel Create((string Family, string Target, string Draft) files, BackendType backend, bool graphOnMetal = true)
    {
        StandardDecoderModel model = files.Family == "qwen4exp"
            ? new Qwen4ExpRuntimeModel(files.Target, backend, files.Draft)
            : new Qwen35RuntimeModel(files.Target, backend, 1, null, 1, files.Draft);
        model.GraphBlockDraftOnMetalForTests = graphOnMetal;
        return model;
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
        using var graph = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft) { GraphBlockDraftOnMetalForTests = true };
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
    private static double CompareRings(ModelBase graph, ModelBase perOp, string label, string when, List<string> mismatches,
        double tolerance = RingTolerance)
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
                if (rel > tolerance) mismatches.Add($"{when}: layer {layer} {what} relative L2 {rel:E2}");
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
