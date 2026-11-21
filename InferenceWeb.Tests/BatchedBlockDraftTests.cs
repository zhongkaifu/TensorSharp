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
/// Several requests' DFlash / DFlash2 blocks drafted in one graph program (StandardDecoderModel.BlockDraftLanes.cs,
/// IBatchedSpeculativeTarget.TryDraftBatch) and their commits' catch-ups written in one (a batched step's scope) are each
/// request's own: four requests at different positions, on their own rings, draft what each drafts alone on a twin model
/// (its carried row injected, then its block, the graph programs a lone request runs), cycle after cycle with accept counts
/// that differ by request and walk every catch-up width, while the 32-row rings wrap; their rings hold the same rows after
/// every draft and every commit. A request's drafts beside copies of itself are its drafts beside other requests (nothing of
/// another lane reaches its rows): the same bits on every executor, the two runs' programs having the same shapes.
/// <para>Against a lone draft a product's rounding follows its row count (the lanes' 4 x b rows against a lone block's b).
/// ggml's CPU gives a row its bits whatever the rows beside it: the blocks and rings are the lone drafts' bits. The managed
/// CPU rounds the two within ~1e-6: every id agrees, confidences and ring rows within 1e-5. A GPU's products round them apart
/// by up to ~1e-3 (ggml-metal's products of more than 8 rows take its half-precision tiles; MLX's ring rows are 3e-4 apart):
/// ring rows within 1e-3, confidences within 1e-3 (ggml_cuda, whose F32 products of more rows than its matrix-vector kernel
/// takes run on the TF32 tensor cores: 5e-3, ring rows 2e-3), and a block may part from its lone draft only at a slot the
/// lone block decided at a
/// near-tie (DFlashNearTies: its head's k-th logit against the (k+1)-th, or its walk's best score against the second, within
/// 1e-2 of the largest), each such parting reported with its gaps. The walk is verified, so a parting moves acceptance only.</para>
/// </summary>
public sealed class BatchedBlockDraftTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-batched-block-draft-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public BatchedBlockDraftTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private const double GpuTolerance = 1e-3, CpuTolerance = 1e-5;

    /// <summary>ggml_cuda's bounds: its F32 products of more rows than its matrix-vector kernel takes run on the TF32 tensor
    /// cores (a 10-bit mantissa, 4.9e-4 a product), where a lone block's run on F32. Measured on an RTX PRO 4500: confidences
    /// of unparted slots 1.0e-3 .. 1.8e-3 apart, ring rows 8.5e-4 .. 8.8e-4 (relative L2).</summary>
    private const double Tf32Tolerance = 5e-3, Tf32RingTolerance = 2e-3;

    // ---------------------------------------------------------------- the lanes draft as each request alone

    [Theory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void LanesDraftAsEachRequestAlone(string variant) => LanesDraftAsAlone(BackendType.Cpu, variant);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void LanesDraftAsEachRequestAloneOnGgmlCpu(string variant) => LanesDraftAsAlone(BackendType.GgmlCpu, variant);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    public void LanesDraftAsEachRequestAloneOnMetal(string variant) => LanesDraftAsAlone(BackendType.GgmlMetal, variant);

    [MlxTheory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void LanesDraftAsEachRequestAloneOnMlx(string variant) => LanesDraftAsAlone(BackendType.Mlx, variant);

    [CudaTheory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    [InlineData("dflash2-q8")]
    [InlineData("dflash2-k3b5")]
    public void LanesDraftAsEachRequestAloneOnDirectCuda(string variant) => LanesDraftAsAlone(BackendType.Cuda, variant);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void LanesDraftAsEachRequestAloneOnGgmlCuda(string variant) => LanesDraftAsAlone(BackendType.GgmlCuda, variant);

    /// <summary>DFlash2's convolution over wider taps and blocks drafts each lane's own block (its taps never reach into the
    /// lane before it), as the region's ops and, on direct CUDA, its kernel.</summary>
    [Theory]
    [InlineData("dflash2-k3b5")]
    [InlineData("dflash2-k4b8")]
    [InlineData("dflash2-k2b2")]
    public void LanesConvolveEachBlockOnItsOwn(string variant) => LanesDraftAsAlone(BackendType.Cpu, variant);

    // ---------------------------------------------------------------- a lane's rows are its own

    [Theory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void ALaneDraftsTheSameBesideCopiesOfItself(string variant) => CopiesDraftAsOthers(BackendType.Cpu, variant);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("dflash2")]
    public void ALaneDraftsTheSameBesideCopiesOfItselfOnGgmlCpu(string variant) => CopiesDraftAsOthers(BackendType.GgmlCpu, variant);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void ALaneDraftsTheSameBesideCopiesOfItselfOnMetal(string variant) => CopiesDraftAsOthers(BackendType.GgmlMetal, variant);

    [MlxTheory]
    [InlineData("dflash2")]
    public void ALaneDraftsTheSameBesideCopiesOfItselfOnMlx(string variant) => CopiesDraftAsOthers(BackendType.Mlx, variant);

    [CudaTheory]
    [InlineData("dflash")]
    [InlineData("dflash2")]
    public void ALaneDraftsTheSameBesideCopiesOfItselfOnDirectCuda(string variant) => CopiesDraftAsOthers(BackendType.Cuda, variant);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("dflash2")]
    public void ALaneDraftsTheSameBesideCopiesOfItselfOnGgmlCuda(string variant) => CopiesDraftAsOthers(BackendType.GgmlCuda, variant);

    // ---------------------------------------------------------------- the engine's step

    /// <summary>The engine's unified step, three requests at once speculating through the drafter (prompts in 16-token
    /// chunks): their blocks drafted in one program a step and their catch-ups written in one, their streams plain
    /// decoding's and those of the requests drafting alone, with the same drafts kept.</summary>
    [Theory]
    [InlineData("qwen35", "dflash", BackendType.Cpu)]
    [InlineData("qwen35", "dflash2", BackendType.Cpu)]
    [InlineData("qwen4exp", "dflash2", BackendType.Cpu)]
    public Task EngineLanesDraftAsEachRequest(string family, string variant, BackendType backend) => EngineDraftsAsEachRequest(family, variant, backend);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("qwen35", "dflash2", BackendType.GgmlCpu)]
    public Task EngineLanesDraftAsEachRequestOnGgmlCpu(string family, string variant, BackendType backend) => EngineDraftsAsEachRequest(family, variant, backend);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("qwen35", "dflash2", BackendType.GgmlMetal)]
    public Task EngineLanesDraftAsEachRequestOnMetal(string family, string variant, BackendType backend) => EngineDraftsAsEachRequest(family, variant, backend);

    [MlxTheory]
    [InlineData("qwen35", "dflash2", BackendType.Mlx)]
    public Task EngineLanesDraftAsEachRequestOnMlx(string family, string variant, BackendType backend) => EngineDraftsAsEachRequest(family, variant, backend);

    [CudaTheory]
    [InlineData("qwen35", "dflash2", BackendType.Cuda)]
    [InlineData("qwen4exp", "dflash2", BackendType.Cuda)]
    public Task EngineLanesDraftAsEachRequestOnDirectCuda(string family, string variant, BackendType backend) => EngineDraftsAsEachRequest(family, variant, backend);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("qwen35", "dflash2", BackendType.GgmlCuda)]
    public Task EngineLanesDraftAsEachRequestOnGgmlCuda(string family, string variant, BackendType backend) => EngineDraftsAsEachRequest(family, variant, backend);

    private async Task EngineDraftsAsEachRequest(string family, string variant, BackendType backend)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures(variant);
        if (family == "qwen4exp") target = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 16, qsa: true);
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.Block, MaxDraftTokens = 3, MinDraftProb = 0 };
        // The managed CPU and ggml's CPU give a decode-size row its bits whatever the rows beside it; ggml's GPU kernels and
        // direct CUDA's products round by their row count, where a synthetic checkpoint's near-tied logits pick either token.
        double nearTie = backend is BackendType.Cpu or BackendType.GgmlCpu ? 0 : 1e-2;
        await SpeculativeRuntimeChecks.AssertBatchedBlockDraftsDraftAsEachRequest(() => Create(family, target, draft, backend),
            options, $"{family} {backend} {variant}", nearTie: nearTie);
    }

    // ---------------------------------------------------------------- the checks

    private static readonly int[] Prompts = { 21, 9, 30, 14 };

    /// <summary>Four requests drafted together on one model and each alone on its twin, cycle after cycle (see the class).</summary>
    private void LanesDraftAsAlone(BackendType backend, string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Fixtures(variant);
        using StandardDecoderModel together = Create("qwen35", target, draft, backend), alone = Create("qwen35", target, draft, backend);
        // ggml's CPU gives a row its bits whatever the rows beside it: there the blocks and rings are the lone drafts' bits.
        bool bitwise = backend == BackendType.GgmlCpu, cpu = backend is BackendType.Cpu or BackendType.GgmlCpu;
        double tolerance = cpu ? CpuTolerance : backend == BackendType.GgmlCuda ? Tf32Tolerance : GpuTolerance;
        double ringTolerance = backend == BackendType.GgmlCuda ? Tf32RingTolerance : tolerance;
        string label = $"{backend} {variant}";
        alone.DFlashTraceForTests = true;   // each lone block's slots, to tell a near-tie from a real difference
        var lanes = new Lanes(together, alone, Prompts, uniformAccepts: false, seed: 391);
        var mismatches = new List<string>();
        var partings = new List<string>();
        double worstConfidence = 0, worstRing = 0;
        int blocks = 0;
        lanes.Run(3, (cycle, tIds, tConf, aIds, aConf, aTrace) =>
        {
            for (int r = 0; r < Prompts.Length; r++)
            {
                blocks++;
                for (int e = 0; e < tIds[r].Length; e++)
                {
                    string slot = $"cycle {cycle} lane {r} slot {e}: drafted {tIds[r][e]} ({tConf[r][e]:R}) together, {aIds[r][e]} ({aConf[r][e]:R}) alone";
                    bool sameBits = BitConverter.SingleToInt32Bits(tConf[r][e]) == BitConverter.SingleToInt32Bits(aConf[r][e]);
                    if (tIds[r][e] != aIds[r][e] || Math.Abs(tConf[r][e] - aConf[r][e]) > tolerance || (bitwise && !sameBits))
                    {
                        // The walk parted (its later slots follow other predecessors): on a CPU never; on a GPU only where the
                        // lone block's slot decided at a near-tie.
                        (double boundary, double walk) margins = DFlashNearTies.Margins(aTrace[r][e]);
                        string why = $"{slot}; the lone slot's boundary gap {margins.boundary:E2}, walk gap {margins.walk:E2}";
                        if (cpu || !DFlashNearTies.IsNearTie(margins)) mismatches.Add(why);
                        else partings.Add(why);
                        break;
                    }
                    worstConfidence = Math.Max(worstConfidence, Math.Abs(tConf[r][e] - aConf[r][e]));
                }
            }
        }, (when, r, rel, sameBits) =>
        {
            worstRing = Math.Max(worstRing, rel);
            if (rel > ringTolerance || (bitwise && !sameBits)) mismatches.Add($"{when}: lane {r}'s ring rows {rel:E2} (relative L2) apart");
        });
        _output.WriteLine($"{label}: {lanes.Cycles} cycles of {Prompts.Length} lanes; worst confidence difference {worstConfidence:E2}, " +
            $"worst ring relative L2 {worstRing:E2}; {partings.Count} of {blocks} blocks parted ({string.Join("; ", partings)}); " +
            $"{together.BatchedBlockDrafts} lane programs, {together.DFlashLaneInjects} rings' catch-ups written together, " +
            $"{together.DraftAdvances} advanced draft steps");
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Null(together.BatchedBlockDraftDeclineReason);
        Assert.Null(together.GraphBlockDraftDeclineReason);
        Assert.Equal(lanes.Cycles, together.BatchedBlockDrafts);
        Assert.Equal(lanes.Cycles * Prompts.Length, together.DFlashLaneInjects);
        Assert.Equal(0, alone.BatchedBlockDrafts);
        // Where the executor replays its plans, the lanes' kept steps advance (their rings' rows, masks and anchors refilled).
        if (together.GraphExecutorForTests.Capabilities.Replays)
            Assert.True(together.DraftAdvances > 0, $"{label}: no lanes step advanced");
    }

    /// <summary>
    /// Lane 0 drafted beside three other requests (model one) and beside three copies of itself (model two), every lane of
    /// both keeping the same count each cycle so the two runs' programs have the same shapes: lane 0's blocks and ring rows
    /// are the same bits in both, on every executor.
    /// </summary>
    private void CopiesDraftAsOthers(BackendType backend, string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var (target, draft) = Fixtures(variant);
        using StandardDecoderModel others = Create("qwen35", target, draft, backend), copies = Create("qwen35", target, draft, backend);
        string label = $"{backend} {variant}";
        var mismatches = new List<string>();
        var beside = new Lanes(others, null, Prompts, uniformAccepts: true, seed: 391);
        var selves = new Lanes(copies, null, Enumerable.Repeat(Prompts[0], Prompts.Length).ToArray(), uniformAccepts: true, seed: 391, copiesOfLane0: true);
        var blocks = new List<(int[] Ids, float[] Conf)>();
        beside.Run(2, (cycle, tIds, tConf, _, _, _) => blocks.Add((tIds[0], tConf[0])), null);
        int at = 0;
        selves.Run(2, (cycle, tIds, tConf, _, _, _) =>
        {
            (int[] ids, float[] conf) = blocks[at++];
            for (int e = 0; e < ids.Length; e++)
                if (ids[e] != tIds[0][e] || BitConverter.SingleToInt32Bits(conf[e]) != BitConverter.SingleToInt32Bits(tConf[0][e]))
                    mismatches.Add($"cycle {cycle} slot {e}: drafted {ids[e]} ({conf[e]:R}) beside others, {tIds[0][e]} ({tConf[0][e]:R}) beside copies");
            for (int r = 1; r < Prompts.Length; r++)
                if (!tIds[r].AsSpan().SequenceEqual(tIds[0]))
                    mismatches.Add($"cycle {cycle}: copy {r} drafted [{string.Join(",", tIds[r])}], lane 0 [{string.Join(",", tIds[0])}]");
        }, null);
        CompareLaneRing(others, copies, "lane0", label, mismatches);
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Equal(beside.Cycles, blocks.Count);
        Assert.Equal(beside.Cycles, others.BatchedBlockDrafts);
        Assert.Equal(selves.Cycles, copies.BatchedBlockDrafts);
    }

    private static void CompareLaneRing(StandardDecoderModel a, StandardDecoderModel b, string requestId, string label, List<string> mismatches)
    {
        for (int layer = 0; layer < a.DFlashSettings.NumLayers; layer++)
        {
            var (ak, av) = a.DFlashRingContextForTests(requestId, layer, out int aFirst);
            var (bk, bv) = b.DFlashRingContextForTests(requestId, layer, out int bFirst);
            if (aFirst != bFirst || !ak.AsSpan().SequenceEqual(bk) || !av.AsSpan().SequenceEqual(bv))
                mismatches.Add($"{label}: {requestId}'s ring rows of layer {layer} differ");
        }
    }

    /// <summary>
    /// Requests drafting on one model together (<see cref="IBatchedSpeculativeTarget.TryDraftBatch"/>, the commits in a
    /// batched step's scope) and, when a twin is given, on the twin each alone (<see cref="IDraftHead.DraftBlock"/> and
    /// <see cref="IDraftHead.DraftCatchUp"/> on its bound ring), from the same synthetic feature rows: each request's prompt
    /// rows, then draft / commit cycles whose accept counts walk 0 .. gamma (each request offset by its index unless
    /// <c>uniformAccepts</c>), the kept tokens the drafted ones (the drafters see one stream).
    /// </summary>
    private sealed class Lanes
    {
        private readonly StandardDecoderModel _together, _alone;
        private readonly int[] _prompts;
        private readonly bool _uniform;
        private readonly string[] _ids;
        private readonly int _feat, _gamma, _vocab;
        private readonly Random[] _rngs;
        private readonly bool _copies;

        public Lanes(StandardDecoderModel together, StandardDecoderModel alone, int[] prompts, bool uniformAccepts, int seed,
            bool copiesOfLane0 = false)
        {
            (_together, _alone, _prompts, _uniform, _copies) = (together, alone, prompts, uniformAccepts, copiesOfLane0);
            _ids = Enumerable.Range(0, prompts.Length).Select(r => $"lane{r}").ToArray();
            var spec = (ISpeculativeModel)together;
            (_feat, _gamma, _vocab) = (spec.SpecFeatureSize, spec.DraftBlockSize, together.Config.VocabSize);
            // Each lane draws its rows from its own generator; a copy of lane 0 from lane 0's seed (the same rows, as its
            // draws come in the same counts).
            _rngs = Enumerable.Range(0, prompts.Length).Select(r => new Random(seed * 31 + (copiesOfLane0 ? 0 : r))).ToArray();
        }

        public int Cycles { get; private set; }

        private float[] Features(int lane, int count)
        {
            var values = new float[count];
            for (int i = 0; i < count; i++) values[i] = (float)(_rngs[lane].NextDouble() * 2 - 1);
            return values;
        }

        /// <summary><paramref name="passes"/> walks of the accept counts; <paramref name="blocks"/> gets each cycle's blocks
        /// (together, then alone: null without a twin), <paramref name="rings"/> each lane's rings compared after every draft and
        /// commit (relative L2, bit for bit).</summary>
        public void Run(int passes, Action<int, int[][], float[][], int[][], float[][], ModelBase.DFlashSlotTrace[][]> blocks,
            Action<string, int, double, bool> rings)
        {
            int lanes = _prompts.Length;
            var last = new int[lanes];
            var position = new int[lanes];
            var pending = new float[lanes][];
            var models = _alone == null ? new[] { _together } : new[] { _together, _alone };
            for (int r = 0; r < lanes; r++)
            {
                // Lane r's prompt (a copy of lane 0's draws lane 0's rows, so it holds the same rows).
                int source = _copies ? 0 : r;
                int length = _prompts[r];
                int[] prompt = Enumerable.Range(0, length).Select(i => 40 + (i * 7 + source) % 23).ToArray();
                float[] rows = Features(r, length * _feat);
                Array.Clear(rows, 0, _feat);   // the token before the prompt has no features
                foreach (StandardDecoderModel model in models)
                {
                    model.BindDraftState(_ids[r]);
                    ((IDraftHead)model).DraftCatchUp(prompt, rows, 0);
                }
                position[r] = length;
                last[r] = prompt[^1];
                pending[r] = Features(r, _feat);
            }
            CompareRings("prompt", rings);
            int cycles = passes * (_gamma + 1);
            for (int cycle = 0; cycle < cycles; cycle++)
            {
                var tIds = new int[lanes][];
                var tConf = new float[lanes][];
                for (int r = 0; r < lanes; r++)
                {
                    tIds[r] = new int[_gamma];
                    tConf[r] = new float[_gamma];
                }
                Assert.True(_together.TryDraftBatch(_ids, last, pending, position, tIds, tConf),
                    $"the lanes program declined: {_together.BatchedBlockDraftDeclineReason}");
                int[][] aIds = null;
                float[][] aConf = null;
                ModelBase.DFlashSlotTrace[][] aTrace = null;
                if (_alone != null)
                {
                    aIds = new int[lanes][];
                    aConf = new float[lanes][];
                    aTrace = new ModelBase.DFlashSlotTrace[lanes][];
                    for (int r = 0; r < lanes; r++)
                    {
                        aIds[r] = new int[_gamma];
                        aConf[r] = new float[_gamma];
                        _alone.BindDraftState(_ids[r]);
                        Assert.Equal(_gamma, ((IDraftHead)_alone).DraftBlock(last[r], pending[r], position[r], aIds[r], aConf[r]));
                        aTrace[r] = _alone.LastDFlashTraceForTests;
                    }
                }
                blocks?.Invoke(cycle, tIds, tConf, aIds, aConf, aTrace);
                CompareRings($"cycle {cycle} draft", rings);

                // Commit: lane r keeps [anchor, its first accepted drafts], their catch-up rows the carried one and fresh ones.
                var windows = new int[lanes][];
                var catchUps = new float[lanes][];
                var accepted = new int[lanes];
                for (int r = 0; r < lanes; r++)
                {
                    accepted[r] = (cycle + (_uniform ? 0 : r)) % (_gamma + 1);
                    windows[r] = new[] { last[r] }.Concat(tIds[r].Take(accepted[r])).ToArray();
                    catchUps[r] = new float[(accepted[r] + 1) * _feat];
                    Array.Copy(pending[r], catchUps[r], _feat);
                    Array.Copy(Features(r, accepted[r] * _feat), 0, catchUps[r], _feat, (long)accepted[r] * _feat);
                }
                using (_together.BeginBatchedSpeculation())
                    for (int r = 0; r < lanes; r++)
                    {
                        _together.BindDraftState(_ids[r]);
                        ((IDraftHead)_together).DraftCatchUp(windows[r], catchUps[r], position[r]);
                    }
                if (_alone != null)
                    for (int r = 0; r < lanes; r++)
                    {
                        _alone.BindDraftState(_ids[r]);
                        ((IDraftHead)_alone).DraftCatchUp(windows[r], catchUps[r], position[r]);
                    }
                CompareRings($"cycle {cycle} commit", rings);
                for (int r = 0; r < lanes; r++)
                {
                    pending[r] = Features(r, _feat);
                    last[r] = (tIds[r][Math.Min(accepted[r], _gamma - 1)] + cycle) % _vocab;
                    position[r] += accepted[r] + 1;
                }
                Cycles++;
            }
        }

        private void CompareRings(string when, Action<string, int, double, bool> rings)
        {
            if (rings == null || _alone == null) return;
            DFlashConfig cfg = _together.DFlashSettings;
            int row = cfg.NumKVHeads * cfg.HeadDim;
            for (int r = 0; r < _prompts.Length; r++)
            {
                double worst = 0;
                bool same = true;
                for (int layer = 0; layer < cfg.NumLayers; layer++)
                {
                    var (tk, tv) = _together.DFlashRingContextForTests(_ids[r], layer, out int tFirst);
                    var (ak, av) = _alone.DFlashRingContextForTests(_ids[r], layer, out int aFirst);
                    Assert.True(tFirst == aFirst && tk.Length == ak.Length, $"{when}: lane {r}'s rings hold different positions");
                    // The positions a draft can attend: the sliding window behind the frontier (a block writes its own rows into
                    // the slots of positions a ring's length behind it, which no window reaches).
                    int held = tk.Length / row, window = Math.Min(held, cfg.SlidingWindow);
                    foreach ((float[] a, float[] b) in new[] { (tk, ak), (tv, av) })
                    {
                        ReadOnlySpan<float> x = a.AsSpan((held - window) * row), y = b.AsSpan((held - window) * row);
                        same &= x.SequenceEqual(y);
                        worst = Math.Max(worst, RelativeL2(x, y));
                    }
                }
                rings(when, r, worst, same);
            }
        }
    }

    private static double RelativeL2(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double diff = 0, norm = 0;
        for (int i = 0; i < a.Length; i++)
        {
            diff += (double)(a[i] - b[i]) * (a[i] - b[i]);
            norm += (double)b[i] * b[i];
        }
        return norm == 0 ? Math.Sqrt(diff) : Math.Sqrt(diff / norm);
    }

    /// <summary>The model, its drafter on the graph programs wherever they run (ggml_metal's DFlash2 selector too, where a
    /// lone request keeps the fused drafter for speed).</summary>
    private StandardDecoderModel Create(string family, string target, string draft, BackendType backend)
    {
        StandardDecoderModel model = family == "qwen4exp"
            ? new Qwen4ExpRuntimeModel(target, backend, draft)
            : new Qwen35RuntimeModel(target, backend, 1, null, 1, draft);
        model.GraphBlockDraftOnMetalForTests = true;
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
            // dflash2-k{taps}b{block}: DFlash2 with a convolution of that many taps over blocks of that width.
            _ when variant.StartsWith("dflash2-k", StringComparison.Ordinal) && variant.IndexOf('b', 9) is int at and > 9
                => Qwen4ExpSyntheticModelBuilder.WriteDFlash(path, blockSize: int.Parse(variant[(at + 1)..]), hidden: hidden, layers: 2,
                    slidingWindow: 16, dflash2: true, convTaps: int.Parse(variant[9..at])),
            _ => throw new ArgumentOutOfRangeException(nameof(variant)),
        };
        return (target, draft);
    }
}
