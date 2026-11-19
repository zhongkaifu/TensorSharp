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
    /// DFlash2 confidences past the first slot), where the graph drafter matches it to the last digit. The ring attention takes
    /// its F32 products here: its default, the engine's F16 tensor-core kernel, is checked against a reference within F16
    /// tolerance (CudaRingAttentionKernelTests) and for lossless speculation (below).</summary>
    [CudaTheory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void GraphDraftsMatchTheManagedPerOpDrafterOnDirectCuda(string variant)
        => DraftsMatch(BackendType.Cuda, variant, BackendType.Cpu, ringKernel: false);

    /// <summary>Speculation with the graph drafter on direct CUDA, its ring attention on the engine's tensor-core kernel
    /// (ts_graph_ring_attn_f16, the default for draft steps), emits plain greedy decoding's stream: the drafts are verified,
    /// so the kernel's F16 rounding moves acceptance only. The kernel served the blocks' attention and the graphs every block.</summary>
    [CudaTheory]
    [InlineData("qwen35", "dflash")]
    [InlineData("qwen35", "dflash2")]
    [InlineData("qwen4exp", "dflash2")]
    public void RingKernelDraftSpeculationMatchesPlainGreedyOnDirectCuda(string family, string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var files = Files(family, variant);
        StandardDecoderModel created = null;
        TensorSharp.Cuda.CudaGraphExecutor executor = null;
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() =>
            {
                created = Create(files, BackendType.Cuda);
                executor = Assert.IsType<TensorSharp.Cuda.CudaGraphExecutor>(created.GraphExecutorForTests);
                return created;
            }, $"{family} cuda {variant} ring kernel");
        Assert.Null(created.GraphBlockDraftDeclineReason);
        Assert.True(created.GraphBlockDrafts > 0, $"{family} {variant}: no block ran as a graph");
        Assert.True(executor.RingAttentions > 0, $"{family} {variant}: no draft attention ran on the ring kernel");
    }

    /// <summary>
    /// A later request's drafts on the model's ring at positions behind the earlier request's last ones (another, shorter
    /// conversation), after another state's buffers changed in between (a new request's first verify window grows its own):
    /// the steps kept over the ring move there (no trace, no build, and where the executor records its steps, no new
    /// recording), and every block drafts bit for bit what steps traced at each call draft, the ring's rows too.
    /// </summary>
    [Theory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void RingStepsMoveBehindForALaterRequest(string variant) => RingStepsMoveBehind(BackendType.Cpu, variant);

    /// <summary>The same on direct CUDA, whose kept steps replay their recordings: a recorded step moved behind its last
    /// position answers as the eager steps of a model that traces every call.</summary>
    [CudaTheory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void RingStepsMoveBehindForALaterRequestOnDirectCuda(string variant) => RingStepsMoveBehind(BackendType.Cuda, variant);

    /// <summary>The same on the GGML builders, whose plans replay too (GraphCapabilities.Replays): their kept ring steps move
    /// behind as well, with the inputs rewritten on the host.</summary>
    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void RingStepsMoveBehindForALaterRequestOnGgmlCpu(string variant) => RingStepsMoveBehind(BackendType.GgmlCpu, variant);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void RingStepsMoveBehindForALaterRequestOnMetal(string variant) => RingStepsMoveBehind(BackendType.GgmlMetal, variant);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void RingStepsMoveBehindForALaterRequestOnGgmlCuda(string variant) => RingStepsMoveBehind(BackendType.GgmlCuda, variant);

    private void RingStepsMoveBehind(BackendType backend, string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Fixtures(variant);
        // (ggml_metal keeps DFlash2 on its fused selector tail unless a test asks for the graph programs.)
        using var kept = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft) { GraphBlockDraftOnMetalForTests = true };
        using var traced = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft)
            { DecodeAdvanceDisabled = true, GraphBlockDraftOnMetalForTests = true };
        string label = $"{backend} {variant}";
        int recorded = 0, built = 0, moved = 0;
        // The first request runs past position 40; the second starts again at 9, after another state's buffers changed.
        List<string> mismatches = DraftBitwise(kept, traced, new[] { 21, 9 },
            beforeRequest: request =>
            {
                if (request == 1) Interlocked.Increment(ref StandardDecoderModel.RecurrentState.BindingChanges);
            },
            afterPrompt: request =>
            {
                // (The prompt's rows reach the ring in one program of their width; the server's prompt step writes them itself.)
                if (request == 1) (recorded, built, moved) = (kept.DecodeRecordings, kept.GraphBuilds, kept.DraftRebases);
            });
        _output.WriteLine($"{label}: {kept.DraftRebases - moved} steps moved behind in the second request, {kept.DraftAdvances} advanced " +
            $"in all; {kept.GraphBuilds - built} builds and {kept.DecodeRecordings - recorded} recordings in the second request");
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Null(kept.GraphBlockDraftDeclineReason);
        Assert.Null(traced.GraphBlockDraftDeclineReason);
        Assert.True(kept.DraftRebases > moved, $"{label}: no kept step moved behind for the second request");
        Assert.Equal(built, kept.GraphBuilds);
        Assert.Equal(recorded, kept.DecodeRecordings);
        Assert.Equal(0, traced.DraftAdvances);
        Assert.Equal(0, traced.DraftRebases);
    }

    /// <summary>
    /// A block step over a ring whose recording failed to launch (the device out of memory) runs unrecorded for the rest of
    /// that request, and records again once the ring changes hands: the next request's blocks launch their recording. Kept
    /// across requests (the steps over a ring move behind for the next request), the failed step had stayed unrecorded for
    /// the life of the process. Every block drafts bit for bit what steps traced at each call draft.
    /// </summary>
    [CudaTheory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void ARingStepWhoseLaunchFailedRecordsAgainForTheRingsNextOwnerOnDirectCuda(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Fixtures(variant);
        using var kept = new Qwen35RuntimeModel(target, BackendType.Cuda, 1, null, 1, draft);
        using var traced = new Qwen35RuntimeModel(target, BackendType.Cuda, 1, null, 1, draft) { DecodeAdvanceDisabled = true };
        string label = $"cuda {variant}";
        // The block's recorded launches each request makes, the first one after the prompt of request 0 failing.
        var launches = new int[2];
        int request = 0, fail = 0;
        kept.RecordedLaunchFailsForTests = shape =>
        {
            // (A block's kept step is keyed by its width too: "draft-dflash-block|w5|".)
            if (!shape.Contains("dflash-block", StringComparison.Ordinal)) return false;
            launches[request]++;
            return Interlocked.Exchange(ref fail, 0) == 1;
        };
        var models = new IPerRequestDraftState[] { kept, traced };
        List<string> mismatches = DraftBitwise(kept, traced, new[] { 21, 9 },
            beforeRequest: r =>
            {
                request = r;
                // Each request on its own ring: the first request's is given back, and the second takes it anew.
                foreach (var model in models)
                {
                    if (r > 0) model.ReleaseDraftState($"req{r - 1}");
                    model.BindDraftState($"req{r}");
                }
            },
            afterPrompt: r =>
            {
                if (r == 0) fail = 1;
            });
        _output.WriteLine($"{label}: the block's recorded launches {launches[0]} in the first request (its first one failed), " +
            $"{launches[1]} in the second; {kept.DecodeRecordings} recordings in all");
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Null(kept.GraphBlockDraftDeclineReason);
        Assert.Equal(0, fail);
        Assert.Equal(1, launches[0]);
        Assert.True(launches[1] > 0, $"{label}: the block stayed unrecorded for the ring's next owner");
    }

    /// <summary>
    /// DFlash2's convolutions as a region (GraphFusion.DFlashConv) draft bit for bit what their ops draft alone, over 2, 3 and
    /// 4 taps (a block as wide as its taps too: the drafter refuses a wider convolution): on direct CUDA the region's kernel
    /// (ts_graph_dflash_conv_f32) serves them,
    /// elsewhere (the managed CPU here) the region's ops run as they did.
    /// </summary>
    [Theory]
    [InlineData("dflash2")]
    [InlineData("dflash2-k4b5")]
    [InlineData("dflash2-k2b2")]
    [InlineData("dflash2-k4b16")]
    public void DFlashConvRegionDraftsAsItsOps(string variant) => ConvRegionDraftsAsItsOps(BackendType.Cpu, variant);

    [CudaTheory]
    [InlineData("dflash2")]
    [InlineData("dflash2-k3b5")]
    [InlineData("dflash2-k4b5")]
    [InlineData("dflash2-k4b4")]
    // The narrowest block, Qwen3.8-27B DFlash2's 8 rows (--spec-draft 7) and the widest replayed block (SmallChunkRows).
    [InlineData("dflash2-k2b2")]
    [InlineData("dflash2-k4b8")]
    [InlineData("dflash2-k4b16")]
    public void DFlashConvRegionDraftsAsItsOpsOnDirectCuda(string variant) => ConvRegionDraftsAsItsOps(BackendType.Cuda, variant);

    private void ConvRegionDraftsAsItsOps(BackendType backend, string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Fixtures(variant);
        using var region = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft);
        using var ops = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft) { DFlashConvRegionsDisabledForTests = true };
        string label = $"{backend} {variant}";
        Assert.True(region.DFlashSettings.HasConv, $"{label}: the drafter has no convolution");
        List<string> mismatches = DraftBitwise(region, ops, new[] { 21, 9 });
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Null(region.GraphBlockDraftDeclineReason);
        Assert.Null(ops.GraphBlockDraftDeclineReason);
        Assert.True(region.GraphBlockDrafts > 0, $"{label}: no block ran as a graph");
        if (region.GraphExecutorForTests is TensorSharp.Cuda.CudaGraphExecutor cuda)
        {
            var opsExecutor = Assert.IsType<TensorSharp.Cuda.CudaGraphExecutor>(ops.GraphExecutorForTests);
            _output.WriteLine($"{label}: {cuda.DFlashConvRegions} convolutions on the region's kernel");
            Assert.True(cuda.DFlashConvRegions > 0, $"{label}: no convolution ran on the region's kernel");
            Assert.Equal(0, opsExecutor.DFlashConvRegions);
        }
    }

    /// <summary>
    /// Drafts through two models' graph drafters from the same synthetic feature rows: per request (one a prompt length in
    /// <paramref name="prompts"/>, each on the model's ring from position 0) the prompt's rows, then draft / commit cycles
    /// whose accept counts walk 0 .. gamma three times (every catch-up width runs, and advances, in each request); every
    /// block's ids and confidences and every draft layer's ring rows are compared bit for bit. Returns the mismatches.
    /// </summary>
    private static List<string> DraftBitwise(StandardDecoderModel a, StandardDecoderModel b, int[] prompts,
        Action<int> beforeRequest = null, Action<int> afterPrompt = null)
    {
        var heads = new IDraftHead[] { a, b };
        var spec = (ISpeculativeModel)a;
        int feat = spec.SpecFeatureSize, gamma = spec.DraftBlockSize, vocab = a.Config.VocabSize;
        var rng = new Random(376);
        float[] Features(int count)
        {
            var values = new float[count];
            for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
            return values;
        }
        var mismatches = new List<string>();
        for (int request = 0; request < prompts.Length; request++)
        {
            beforeRequest?.Invoke(request);
            int promptLength = prompts[request];
            int[] prompt = Enumerable.Range(0, promptLength).Select(i => 40 + (i * 7 + request) % 23).ToArray();
            float[] rows = Features(promptLength * feat);
            Array.Clear(rows, 0, feat);   // the token before the prompt has no features
            foreach (IDraftHead head in heads) head.DraftCatchUp(prompt, rows, 0);
            afterPrompt?.Invoke(request);
            int position = promptLength, last = prompt[^1];
            float[] pending = Features(feat);
            for (int cycle = 0; cycle < 3 * (gamma + 1); cycle++)
            {
                var ids = new int[2][];
                var conf = new float[2][];
                for (int m = 0; m < 2; m++)
                {
                    ids[m] = new int[gamma];
                    conf[m] = new float[gamma];
                    Assert.Equal(gamma, heads[m].DraftBlock(last, pending, position, ids[m], conf[m]));
                }
                for (int e = 0; e < gamma; e++)
                    if (ids[0][e] != ids[1][e] || BitConverter.SingleToInt32Bits(conf[0][e]) != BitConverter.SingleToInt32Bits(conf[1][e]))
                        mismatches.Add($"request {request} position {position} slot {e}: {ids[0][e]} ({conf[0][e]:R}) against " +
                            $"{ids[1][e]} ({conf[1][e]:R})");
                for (int layer = 0; layer < a.DFlashSettings.NumLayers; layer++)
                {
                    var (ak, av) = a.DFlashRingContextForTests(layer, out int aFirst);
                    var (bk, bv) = b.DFlashRingContextForTests(layer, out int bFirst);
                    if (aFirst != bFirst || !ak.AsSpan().SequenceEqual(bk) || !av.AsSpan().SequenceEqual(bv))
                        mismatches.Add($"request {request} position {position}: layer {layer}'s ring rows differ");
                }
                int accepted = cycle % (gamma + 1);
                int[] window = new[] { last }.Concat(ids[0].Take(accepted)).ToArray();
                float[] catchUp = Features((accepted + 1) * feat);
                Array.Copy(pending, catchUp, feat);
                foreach (IDraftHead head in heads) head.DraftCatchUp(window, catchUp, position);
                pending = Features(feat);
                last = (ids[0][Math.Min(accepted, gamma - 1)] + cycle) % vocab;
                position += accepted + 1;
            }
        }
        return mismatches;
    }

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

    /// <summary>
    /// The first decode step after a prompt drafts a block: the prompt's plain steps write the ring and, in the ring's
    /// carry, the last row's taps, which the first draft chains from (where the context, armed without a carry, had decoded
    /// that step plainly); the speculating streams are plain decoding's. Through the engine's unified step and through a
    /// context armed after a plain prompt.
    /// </summary>
    [Theory]
    [InlineData("qwen35", "dflash")]
    [InlineData("qwen35", "dflash2")]
    [InlineData("qwen4exp", "dflash2")]
    public Task FirstDecodeStepAfterThePromptDrafts(string family, string variant) => FirstDecodeStepDrafts(family, variant, BackendType.Cpu);

    /// <summary>The same on the direct CUDA backend (the CUDA graph executor), where the prompt's steps must stay exactly
    /// the plain ones for the speculating streams to be plain decoding's.</summary>
    [CudaTheory]
    [InlineData("qwen35", "dflash")]
    [InlineData("qwen35", "dflash2")]
    [InlineData("qwen4exp", "dflash2")]
    public Task FirstDecodeStepAfterThePromptDraftsOnDirectCuda(string family, string variant)
        => FirstDecodeStepDrafts(family, variant, BackendType.Cuda);

    /// <summary>The drafter-follow warm-up runs its products at every row count up to 64 and a ladder above it, up to the
    /// widest a chunk keeps.</summary>
    [Fact]
    public void TheFollowWarmupRowsCoverEveryCountTo64AndALadderAbove()
    {
        Assert.Equal(Enumerable.Range(1, 10), StandardDecoderModel.FollowWarmRows(10));
        int[] rows = StandardDecoderModel.FollowWarmRows(2081);
        Assert.Equal(Enumerable.Range(1, 64), rows.Take(64));
        Assert.Equal(2081, rows[^1]);
        Assert.True(rows.Zip(rows.Skip(1)).All(pair => pair.Second > pair.First));
        Assert.All(new[] { 80, 128, 256, 512, 1024, 2048 }, r => Assert.Contains(r, rows));
        Assert.True(rows.Length < 120, $"{rows.Length} row counts");
    }

    /// <summary>On direct CUDA the drafter-follow warm-up runs the follow's products, one per weight shape (a drafter with
    /// DFlash2 Q4_K_M's formats: Q4_K encoder and K projections, Q6_K V projections), where the managed executor runs none, and
    /// a model warmed so still drafts what plain decoding emits.</summary>
    [CudaFact]
    public void TheFollowWarmupRunsTheFollowsProductsOnDirectCuda()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var files = Files("qwen35", "dflash2-kquant");
        using (StandardDecoderModel cpu = Create(files, BackendType.Cpu))
            Assert.Equal(0, cpu.WarmDFlashFollowProductsCounted());
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() =>
        {
            StandardDecoderModel model = Create(files, BackendType.Cuda);
            Assert.Equal(3, model.WarmDFlashFollowProductsCounted());
            return model;
        }, "qwen35 dflash2-kquant cuda warmed follow");
    }

    /// <summary>
    /// A prompt that arrives while other requests speculate is read before their drafts and verify windows in the step that
    /// admits it, and its first token streams then (BatchExecutor.PromptFinished), not after their verify; every stream is
    /// still plain decoding's.
    /// </summary>
    [Theory]
    [InlineData("qwen35", "dflash2", BackendType.Cpu)]
    public Task APromptBesideSpeculatingRequestsStreamsItsFirstTokenBeforeTheirVerify(string family, string variant, BackendType backend)
        => PromptBesideSpeculatingRequests(family, variant, backend);

    [CudaTheory]
    [InlineData("qwen35", "dflash2", BackendType.Cuda)]
    public Task APromptBesideSpeculatingRequestsStreamsItsFirstTokenBeforeTheirVerifyOnDirectCuda(string family, string variant,
        BackendType backend)
        => PromptBesideSpeculatingRequests(family, variant, backend);

    private async Task PromptBesideSpeculatingRequests(string family, string variant, BackendType backend)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var files = Files(family, variant);
        int[][] prompts =
        {
            Enumerable.Range(0, 24).Select(i => 40 + i % 6).ToArray(),
            Enumerable.Range(0, 19).Select(i => 50 + (i * 5) % 9).ToArray(),
            Enumerable.Range(0, 27).Select(i => 44 + i % 4).ToArray(),
        };
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.Block, MaxDraftTokens = 3, MinDraftProb = 0 };
        var (plain, _) = await ServeWithALatePrompt(files, backend, prompts, null);
        var (drafted, published) = await ServeWithALatePrompt(files, backend, prompts, options);
        for (int r = 0; r < prompts.Length; r++)
            Assert.Equal(plain[r], drafted[r]);
        // Only the late request's prompt shared a step with speculating requests, and it streamed before their verify ran.
        var late = Assert.Single(published);
        Assert.Equal("req2", late.Id);
        Assert.True(late.BeforeVerify, "the late prompt's first token waited for the step's verify windows");
        Assert.True(late.VerifiedInItsStep, "the step that read the late prompt verified no window");
    }

    /// <summary>Two requests, then (once both decode) a third: each request's tokens, and each prompt the executor finished
    /// before the rest of its step (whether that step's verify windows had run yet, and whether they ran after).</summary>
    private static async Task<(int[][] Outputs, List<(string Id, bool BeforeVerify, bool VerifiedInItsStep)> Published)> ServeWithALatePrompt(
        (string Family, string Target, string Draft) files, BackendType backend, int[][] prompts, SpeculationOptions speculation)
    {
        using StandardDecoderModel model = Create(files, backend);
        var cfg = new TensorSharp.Runtime.Scheduling.SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 32,
            NumBlocks = 64,
            BlockSize = 16,
            EnablePrefixCaching = false,
            Speculation = speculation ?? new SpeculationOptions { Enabled = false },
        };
        using var engine = new TensorSharp.Runtime.Scheduling.InferenceEngine(model, cfg,
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance);
        var gate = new TensorSharp.Runtime.Scheduling.ComputeGate();
        gate.Close();
        engine.ComputeGate = gate;
        var published = new List<(string Id, bool BeforeVerify, bool VerifiedInItsStep)>();
        int verifiesAtStepStart = 0, pending = -1;
        var bothDecode = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        engine.ScheduledForTests = output =>
        {
            // The previous step's published prompt: did that step verify after it?
            if (pending >= 0)
            {
                var last = published[pending];
                published[pending] = last with { VerifiedInItsStep = model.BatchedSpecVerifySteps > verifiesAtStepStart };
                pending = -1;
            }
            verifiesAtStepStart = model.BatchedSpecVerifySteps;
            // Once the first two requests both decode, the loop parks before its next step for the third to arrive.
            if (output.ScheduledWork.Count == 2 && output.ScheduledWork.TrueForAll(work => !work.IsPrefill) && bothDecode.TrySetResult())
                gate.Close();
        };
        var executor = engine.ExecutorForTests;
        Action<TensorSharp.Runtime.Scheduling.SequenceState> publish = executor.PromptFinished;
        executor.PromptFinished = seq =>
        {
            published.Add((seq.RequestId, model.BatchedSpecVerifySteps == verifiesAtStepStart, false));
            pending = published.Count - 1;
            publish(seq);
        };
        var sequences = prompts.Select((p, i) => new TensorSharp.Runtime.Scheduling.SequenceState($"req{i}", p, 24, cfg.BlockSize,
            TensorSharp.Runtime.SamplingConfig.Greedy)).ToList();
        var handles = new List<TensorSharp.Runtime.Scheduling.InferenceRequestHandle>
        {
            engine.SubmitRequest(sequences[0]),
            engine.SubmitRequest(sequences[1]),
        };
        gate.Open();
        await bothDecode.Task.WaitAsync(TimeSpan.FromMinutes(1));
        // Parked before the step after the first decode step: the third request joins the next one.
        var parked = System.Diagnostics.Stopwatch.StartNew();
        while (engine.StepsHeldByGate == 0 && parked.Elapsed < TimeSpan.FromMinutes(1)) await Task.Delay(1);
        Assert.True(engine.StepsHeldByGate > 0, "the engine did not park for the third request");
        handles.Add(engine.SubmitRequest(sequences[2]));
        gate.Open();
        int[][] outputs = await Task.WhenAll(handles.Select(async h =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync())
                tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        })).WaitAsync(TimeSpan.FromMinutes(3));
        return (outputs, published);
    }

    private async Task FirstDecodeStepDrafts(string family, string variant, BackendType backend)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var files = Files(family, variant);
        var created = new List<StandardDecoderModel>();
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.Block, MaxDraftTokens = 3, MinDraftProb = 0 };
        await SpeculativeRuntimeChecks.AssertFirstDecodeStepDrafts(() =>
            {
                StandardDecoderModel model = Create(files, backend);
                created.Add(model);
                return model;
            }, options, $"{family} {backend} {variant}");
        Assert.True(created.Exists(m => m.GraphDFlashFollows > 0), $"{family} {variant}: no trunk step wrote the ring");
        Assert.All(created, m => Assert.Null(m.GraphBlockDraftDeclineReason));
        SpeculativeRuntimeChecks.AssertPlainPromptCarriesIntoTheFirstDraft(() => Create(files, backend), $"{family} {backend} {variant} plain prompt");
    }

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
            "dflash2-kquant" => Qwen4ExpSyntheticModelBuilder.WriteDFlash(path, blockSize: 5, hidden: hidden, layers: 2, slidingWindow: 16,
                dflash2: true, kQuantProjections: true),
            // dflash2-k{taps}b{block}: DFlash2 with a convolution of that many taps over blocks of that width.
            _ when variant.StartsWith("dflash2-k", StringComparison.Ordinal) && variant.IndexOf('b', 9) is int at and > 9
                => Qwen4ExpSyntheticModelBuilder.WriteDFlash(path, blockSize: int.Parse(variant[(at + 1)..]), hidden: hidden, layers: 2,
                    slidingWindow: 16, dflash2: true, convTaps: int.Parse(variant[9..at])),
            _ => throw new ArgumentOutOfRangeException(nameof(variant)),
        };
        return (target, draft);
    }

    /// <summary>The graph drafter on <paramref name="backend"/> against the per-op drafter on <paramref name="reference"/>
    /// (the same backend unless named).</summary>
    private void DraftsMatch(BackendType backend, string variant, BackendType? reference = null, bool ringKernel = true)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Fixtures(variant);
        using var graph = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft) { GraphBlockDraftOnMetalForTests = true };
        if (!ringKernel && graph.GraphExecutorForTests is TensorSharp.Cuda.CudaGraphExecutor cuda) cuda.RingAttentionDisabledForTests = true;
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
