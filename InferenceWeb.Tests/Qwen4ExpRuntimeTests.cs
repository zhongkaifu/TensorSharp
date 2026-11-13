// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.8-Flash-Next (qwen4exp) on the shared decoder runtime against goldens of the deleted Qwen4ExpModel
// (Fixtures/Qwen4ExpLegacy) on the synthetic checkpoint:
// hyper-connections, the sigmoid-gated recurrence, gated attention, the PLE layer (its n-gram and conv
// histories across steps) and the shared-expert MoE.

using TensorSharp;
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Qwen4ExpRuntimeTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("q4e-runtime").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // An EOS (255) inside the prompt cuts the n-gram context after it.
    private static readonly int[] Prompt = { 1, 17, 33, 2, 99, 5, 255, 45, 12, 200, 7, 63, 88, 31 };
    private static readonly int[] Decode = { 41, 9, 130, 77 };

    // A top-k of 64 blocks keeps the QSA layers dense (exact) for these lengths. Qwen4ExpModel's per-layer path
    // (the managed CPU) has no QSA layers at all.
    private string Fixture(bool qsa = true) => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, $"q4e-{qsa}.gguf"), indexerTopK: 64, qsa: qsa);

    /// <summary>Rows the deleted Qwen4ExpModel produced on these fixtures (Fixtures/Qwen4ExpLegacy).</summary>
    private static float[][] LegacyGolden(string name)
    {
        using Stream stream = typeof(Qwen4ExpRuntimeTests).Assembly.GetManifestResourceStream(
            $"InferenceWeb.Tests.Fixtures.Qwen4ExpLegacy.{name}.json") ?? throw new FileNotFoundException(name);
        return System.Text.Json.JsonSerializer.Deserialize<float[][]>(stream)!;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeMatchesQwen4ExpModelOnTheManagedCpu(bool chunkedPrompt)
    {
        string path = Fixture(qsa: false);
        float[][] expected = LegacyGolden(chunkedPrompt ? "script_chunked_cpu" : "script_cpu");
        float[][] actual = Run(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu), chunkedPrompt);
        // The managed executor runs the recurrent mixer's regions as fused host kernels (ManagedGraphExecutor), whose
        // results stay within 2.4e-7 of the region's own ops but round differently; through this fixture's routing
        // (16 random experts, top 4) that moves one prompt by ~3.3e-3 on arm64. A chunked prompt's managed products
        // pick their kernels by row count, so either model moves by ~5e-3 against its own unchunked run
        // (Qwen4ExpModel: 4.7e-3 measured), and x64's products differ by 6.4e-3 in one prompt (Linux).
        AssertClose(expected, actual, 1e-2, "managed cpu");
    }

    /// <summary>MLX (F16 attention over F16 caches, MLX's own kernels) against Qwen4ExpModel's managed run.</summary>
    [MlxTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeOnMlxMatchesQwen4ExpModel(bool chunkedPrompt)
    {
        string path = Fixture(qsa: false);
        float[][] expected = LegacyGolden(chunkedPrompt ? "script_chunked_cpu" : "script_cpu");
        float[][] actual = Run(() => new Qwen4ExpRuntimeModel(path, BackendType.Mlx), chunkedPrompt);
        AssertClose(expected, actual, 2e-2, "mlx");
    }

    /// <summary>TS_Q4E_HC_Q8=1 on the shared runtime: the BF16 hyper-connection matrices become Q8_0 at load for every
    /// backend (the managed CPU here), and the answers stay within Q8_0's rounding of the BF16 read.</summary>
    [Fact]
    public void HyperConnectionsReadAsQ8_0StayCloseToBf16()
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-hc-bf16.gguf"), indexerTopK: 64, qsa: false,
            projectionType: SyntheticGguf.GgmlType.BF16);
        float[][] bf16 = Run(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu), chunkedPrompt: false);
        float[][] q8;
        using (Qwen4ExpFormat.HcQ8ForCurrentFlowInTests(true))
        {
            q8 = Run(() =>
            {
                var model = new Qwen4ExpRuntimeModel(path, BackendType.Cpu);
                Assert.Equal(WeightRequant.GgmlTypeQ8_0, model.QuantizedWeightTypeForTests("output_hc_down.weight"));
                Assert.Equal(WeightRequant.GgmlTypeQ8_0, model.QuantizedWeightTypeForTests("blk.0.hc_ffn_up.weight"));
                return model;
            }, chunkedPrompt: false);
        }
        double[] errors = RowErrors(bf16, q8);
        Assert.True(errors.Max() > 0, "the Q8_0 read changed nothing: the matrices stayed BF16");
        AssertClose(bf16, q8, 2e-2, "hc q8_0 vs bf16");
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeMatchesQwen4ExpModelOnGgmlCpu(bool chunkedPrompt) => MatchesOnGgml(BackendType.GgmlCpu, chunkedPrompt);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeMatchesQwen4ExpModelOnGgmlMetal(bool chunkedPrompt) => MatchesOnGgml(BackendType.GgmlMetal, chunkedPrompt);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeMatchesQwen4ExpModelOnGgmlCuda(bool chunkedPrompt) => MatchesOnGgml(BackendType.GgmlCuda, chunkedPrompt);

    /// <summary>Qwen4ExpModel's token-span graphs on ggml_cpu (QSA layers dense below the budget) against the runtime.</summary>
    private void MatchesOnGgml(BackendType backend, bool chunkedPrompt)
    {
        string path = Fixture();
        float[][] expected = LegacyGolden(chunkedPrompt ? "script_qsa_chunked_ggmlcpu" : "script_qsa_ggmlcpu");
        float[][] actual = Run(() => new Qwen4ExpRuntimeModel(path, backend), chunkedPrompt);
        AssertClose(expected, actual, 2e-2, backend.ToString());
    }

    // ---- the prefix cache: every reuse continues as a cold run of the same tokens

    /// <summary>The prefix-cache contract script (every capture, clone, donation, truncation and primary conversion
    /// compared token for token with a cold run of the same tokens) on the runtime's qwen4exp: the slot's copies carry
    /// the PLE layer's conv and n-gram histories and the QSA indexer cache, with QSA dense and selecting past a top-k of
    /// 8 (Qwen4ExpModel's holders were held to this by Qwen4ExpRetainedCacheTests). The accelerators run the selecting
    /// fixture only through SparseQsaMatches: a prompt chunked where the script splits it (34 + 9 cells) picks a near-tied
    /// block differently from one forward on Metal (1.6e-2 at the next row, with or without any reuse; exact on the CPU
    /// backends), and this fixture's top two logits tie to 1e-4, so a greedy comparison would flip.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PrefixCacheReuseContinuesAsColdRunsOnTheManagedCpu(bool selecting) => PrefixCacheConformance(BackendType.Cpu, selecting);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(false)]
    [InlineData(true)]
    public void PrefixCacheReuseContinuesAsColdRunsOnGgmlCpu(bool selecting) => PrefixCacheConformance(BackendType.GgmlCpu, selecting);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(false)]
    public void PrefixCacheReuseContinuesAsColdRunsOnGgmlMetal(bool selecting) => PrefixCacheConformance(BackendType.GgmlMetal, selecting);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false)]
    public void PrefixCacheReuseContinuesAsColdRunsOnGgmlCuda(bool selecting) => PrefixCacheConformance(BackendType.GgmlCuda, selecting);

    private void PrefixCacheConformance(BackendType backend, bool selecting)
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, $"q4e-prefix-{selecting}.gguf"), indexerTopK: selecting ? 8 : 64, qsa: true);
        using var model = new Qwen4ExpRuntimeModel(path, backend);
        model.AttachPrefixCache(new InferenceWeb.Tests.PrefixCache.Fakes.RecordingPayloadSink());
        // An EOS inside the shared prefix cuts the n-gram context after it; 34 + 9 cells cross a top-k of 8.
        int[] shared = Enumerable.Range(0, 34).Select(i => i == 20 ? 255 : 3 + (i * 37) % 240).ToArray();
        int[] suffix = Enumerable.Range(0, 9).Select(i => 7 + (i * 53) % 230).ToArray();
        var report = InferenceWeb.Tests.PrefixCache.PrefixCacheConformanceScript.Run(new InferenceWeb.Tests.PrefixCache.ConformanceSubject
        {
            Name = $"qwen4exp {backend}{(selecting ? " selecting" : "")}",
            Model = model,
            SharedPrefix = shared,
            Suffix = suffix,
            DecodeTokens = 8,
            DonateAfter = 3,
            RewindTokens = 4,
            // A graph leaves a slot's rows and recurrent state on the device until a sync.
            ExpectDirtyDonations = backend != BackendType.Cpu,
        });
        Assert.Contains("capture", report.Ran);
        Assert.Contains("clone x2", report.Ran);
        Assert.Contains("donate/return/donate", report.Ran);
        Assert.Contains("settle then clone", report.Ran);
        Assert.Contains("primary conversion", report.Ran);
    }

    // ---- QSA past its dense budget: the block selection

    /// <summary>How a sparse QSA run reaches its sparse rows.</summary>
    public enum QsaRun
    {
        /// <summary>A 40-token prompt in one step, then 12 decode steps.</summary>
        Whole,
        /// <summary>The same prompt in two steps (23 tokens, then 17 continuing at 23).</summary>
        Chunked,
        /// <summary>A 6-token prompt, then 18 decode steps: the first 5 attend densely, the rest select.</summary>
        Crossing,
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(QsaRun.Whole)]
    [InlineData(QsaRun.Chunked)]
    [InlineData(QsaRun.Crossing)]
    public void SparseQsaOnGgmlCpuMatchesTheManagedSelection(QsaRun run) => SparseQsaMatches(BackendType.GgmlCpu, run);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(QsaRun.Whole)]
    [InlineData(QsaRun.Chunked)]
    [InlineData(QsaRun.Crossing)]
    public void SparseQsaOnGgmlMetalMatchesTheManagedSelection(QsaRun run) => SparseQsaMatches(BackendType.GgmlMetal, run);

    [MlxTheory]
    [InlineData(QsaRun.Whole)]
    [InlineData(QsaRun.Chunked)]
    [InlineData(QsaRun.Crossing)]
    public void SparseQsaOnMlxMatchesTheManagedSelection(QsaRun run) => SparseQsaMatches(BackendType.Mlx, run);

    /// <summary>
    /// A top-k of 8 cells keeps QSA dense up to 11 cells; past them a row attends only its top two blocks and its
    /// incomplete tail. ggml's lightning indexer and top-k mask (the GGML lowering) select what the interpreter's
    /// host ops do on the managed CPU. Qwen4ExpModel is no reference here: its native selection keeps
    /// <c>top_k + ratio - 1</c> single cells (llama.cpp before #29751), which takes part of a third block where a
    /// row's tail is short; the runtime keeps whole blocks plus the tail, as llama.cpp does since. The backends
    /// agree to under 9e-3 per row where they pick the same blocks; one block more (a top-k of 12) moves rows by
    /// 2.1e-2 or more and dense attention by 3.4e-2 or more, so the tolerance tells a wrong pick from rounding (one
    /// near-tied pick a run aside), and the dense run shows the selection engages (and, decoding across the
    /// budget, that the rows within it are dense).
    /// </summary>
    private void SparseQsaMatches(BackendType backend, QsaRun run)
    {
        const double tolerance = 1.5e-2;
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-sparse.gguf"), indexerTopK: 8, qsa: true);
        string densePath = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-dense.gguf"), indexerTopK: 64, qsa: true);
        int[] prompt = Enumerable.Range(0, run == QsaRun.Crossing ? 6 : 40).Select(i => 3 + (i * 37) % 240).ToArray();
        int[] decode = Enumerable.Range(0, run == QsaRun.Crossing ? 18 : 12).Select(i => 5 + (i * 11) % 200).ToArray();
        float[][] Rows(Func<ModelBase> create)
        {
            using ModelBase model = create();
            model.ResetKVCache();
            var rows = new List<float[]>();
            if (run == QsaRun.Chunked)
            {
                model.ForwardRefill(prompt[..23]);
                rows.Add((float[])model.Forward(prompt[23..]).Clone());
            }
            else
            {
                rows.Add((float[])model.ForwardRefill(prompt).Clone());
            }
            foreach (int token in decode)
                rows.Add((float[])model.Forward(new[] { token }).Clone());
            return rows.ToArray();
        }
        float[][] expected = Rows(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu));
        float[][] actual = Rows(() => new Qwen4ExpRuntimeModel(path, backend));
        // The selection is discrete: where two blocks score within rounding of the cutoff, the backends may keep
        // different ones (the interpreter sums in double, ggml in float; ggml_cpu on x64 flipped one row of the
        // crossing run, 2.85e-2). One such row a run is allowed, within a block's worth (one block more moves a row
        // by 2.1e-2 to 3.6e-2); a wrong selection moves most rows.
        double[] errors = RowErrors(expected, actual);
        Assert.True(errors.Count(e => e >= tolerance) <= 1 && errors.All(e => e < 5e-2),
            $"{backend} sparse qsa ({run}) relL2 per row: {string.Join(", ", errors.Select(e => e.ToString("E2")))}");
        double[] fromDense = RowErrors(expected, Rows(() => new Qwen4ExpRuntimeModel(densePath, BackendType.Cpu)));
        string detail = string.Join(", ", fromDense.Select(e => e.ToString("E2")));
        if (run == QsaRun.Crossing)
        {
            // Row r ends at position 5 + r: rows 0..5 hold at most 11 cells.
            Assert.True(fromDense.Take(6).All(e => e < 1e-6), $"rows within the budget are not dense: {detail}");
            Assert.True(fromDense.Skip(6).Max() > tolerance, $"the selection barely moves the rows past the budget: {detail}");
        }
        else
        {
            Assert.True(fromDense.All(e => e > tolerance), $"the selection barely moves the rows: {detail}");
        }
    }

    // ---- QSA past its budget over media positions

    /// <summary>
    /// The block selection with rotary positions that are not the KV index (media spans and the text after them): a
    /// block's pooled key turns at its first cell's recorded position and a row's indexer query at its own. Rotary scores
    /// depend only on relative positions, so a text prompt fed at every position plus 7 (a position table, then the slot's
    /// delta) answers as the plain prompt, through the selection (a top-k of 8 over a 40-token prompt and decode steps).
    /// Qwen4ExpModel kept a position history for this; the runtime refused the selection once any position moved. The
    /// rotations at other absolute angles round differently and this fixture's routing (16 random experts, top 4)
    /// amplifies that: without QSA layers the same shift moves the rows by 4-6e-3 on the managed CPU (a table at the
    /// plain positions answers bit for bit), and a selection misread by the shift would move them by 2e-2 or more.
    /// </summary>
    [Theory]
    [InlineData(BackendType.Cpu)]
    public void ShiftedPositionsSelectAsThePlainPrompt(BackendType backend) => ShiftedPositions(backend, 1e-2);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void ShiftedPositionsSelectAsThePlainPromptOnGgmlCpu(BackendType backend) => ShiftedPositions(backend, 1.5e-2);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlMetal)]
    public void ShiftedPositionsSelectAsThePlainPromptOnGgmlMetal(BackendType backend) => ShiftedPositions(backend, 1.5e-2);

    private void ShiftedPositions(BackendType backend, double tolerance)
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-shift.gguf"), indexerTopK: 8, qsa: true);
        int[] prompt = Enumerable.Range(0, 40).Select(i => 3 + (i * 37) % 240).ToArray();
        int[] decode = Enumerable.Range(0, 10).Select(i => 5 + (i * 11) % 200).ToArray();
        float[][] plain = PositionedRun(() => new Qwen4ExpRuntimeModel(path, backend), prompt, decode, null);
        float[][] shifted = PositionedRun(() => new Qwen4ExpRuntimeModel(path, backend), prompt, decode, i => (i + 7, i + 7, i + 7));
        AssertClose(plain, shifted, tolerance, $"{backend} shifted positions");
    }

    /// <summary>An image-like span (a 2 x 4 grid at one temporal position, then text continuing past it) through the
    /// selection: the GGML backends' multi-axis rotary of the pooled keys and queries answers as the managed CPU's.</summary>
    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void MediaPositionsSelectAsTheManagedCpuOnGgmlCpu(BackendType backend) => MediaPositions(backend);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlMetal)]
    public void MediaPositionsSelectAsTheManagedCpuOnGgmlMetal(BackendType backend) => MediaPositions(backend);

    private void MediaPositions(BackendType backend)
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-media.gguf"), indexerTopK: 8, qsa: true);
        int[] prompt = Enumerable.Range(0, 40).Select(i => 3 + (i * 37) % 240).ToArray();
        int[] decode = Enumerable.Range(0, 10).Select(i => 5 + (i * 11) % 200).ToArray();
        // Rows 6..13 are the grid (T 6, H 6..7, W 6..9); the text after it continues at T 10 (the grid's extent).
        (int, int, int) Position(int i) => i < 6 ? (i, i, i) : i < 14 ? (6, 6 + (i - 6) / 4, 6 + (i - 6) % 4) : (i - 4, i - 4, i - 4);
        float[][] expected = PositionedRun(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu), prompt, decode, Position);
        float[][] actual = PositionedRun(() => new Qwen4ExpRuntimeModel(path, backend), prompt, decode, Position);
        double[] errors = RowErrors(expected, actual);
        // As SparseQsaMatches: one near-tied block pick a run is allowed, within a block's worth.
        Assert.True(errors.Count(e => e >= 1.5e-2) <= 1 && errors.All(e => e < 5e-2),
            $"{backend} media positions relL2 per row: {string.Join(", ", errors.Select(e => e.ToString("E2")))}");
    }

    private static float[][] PositionedRun(Func<ModelBase> create, int[] prompt, int[] decode, Func<int, (int T, int H, int W)>? position)
    {
        using ModelBase model = create();
        model.ResetKVCache();
        if (position != null)
        {
            var flat = new int[3 * prompt.Length];
            for (int i = 0; i < prompt.Length; i++)
                (flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]) = position(i);
            ((TensorSharp.Models.Architecture.IMRoPEPositionSink)model).SetMRoPEPositions(flat);
        }
        var rows = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
        foreach (int token in decode)
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        return rows.ToArray();
    }

    // ---- speculation: the PLE histories follow a verify's accept count; the MTP head chains the stream

    private (string Target, string Mtp) SpecFixtures(bool qsa)
        => (Fixture(qsa), Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, $"q4e-mtp-{qsa}.gguf"), indexerTopK: 64));

    /// <summary>n-gram drafts (accepted and rejected): the verify keeps each row's recurrent and PLE state, and the
    /// accept count commits the kept prefix's n-gram and conv histories.</summary>
    /// <summary>Tensor parallelism as per-rank graphs on two simulated devices: the heads and experts split, the
    /// hyper-connections, PLE and QSA indexer whole on each rank; the same logits as one device, with QSA dense and
    /// selecting (a top-k of 8 over a 40-token prompt: every rank must pick the same blocks).</summary>
    /// <summary>Whole-layer stages (--layer-split 2) on the managed CPU: the PLE layer, the QSA layers and the hyper-connection
    /// streams pass stage to stage as the residual rows, and every row answers as one device's (Qwen4ExpModel's layer
    /// placement was held to this by Qwen4ExpLayerSplitTests).</summary>
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void StagesMatchOneDeviceOnTheManagedCpu(bool qsa, bool chunkedPrompt)
    {
        string path = Fixture(qsa);
        ModelBase Staged()
        {
            var model = new Qwen4ExpRuntimeModel(path, BackendType.Cpu, layerSplitDegree: 2);
            Assert.Equal(2, model.Mesh.Stages);
            return model;
        }
        AssertClose(Run(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu), chunkedPrompt), Run(Staged, chunkedPrompt), 1e-5,
            $"stages2 qsa={qsa} chunked={chunkedPrompt}");
    }

    /// <summary>A decode step one position on runs again without a trace, its PLE rows and QSA pooled blocks refilled:
    /// every logit is the traced step's, while QSA attends densely and once it selects past its budget (~67 cells on
    /// this fixture: those steps trace again), on one device and over two layer-split stages (each keeps its own step).</summary>
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void AdvancedDecodeStepsMatchTracedOnes(bool stages, bool managedCpu)
    {
        using IDisposable f32 = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture(qsa: true);
        BackendType backend = managedCpu ? BackendType.Cpu : TestGates.PinnedGgmlBackend;
        List<float[]> Run(bool advance, out int advances)
        {
            using var model = stages ? new Qwen4ExpRuntimeModel(path, backend, layerSplitDegree: 2) : new Qwen4ExpRuntimeModel(path, backend);
            Assert.Equal(stages ? 2 : 1, model.Mesh.Stages);
            model.DecodeAdvanceDisabled = !advance;
            var unified = (TensorSharp.Runtime.Scheduling.IUnifiedBatchModel)model;
            unified.EnsureSequenceSlot("a", 320);
            int[] prompt = Enumerable.Range(0, 20).Select(i => 3 + (i * 37) % 240).ToArray();
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, logits));
            var trace = new List<float[]>();
            int length = prompt.Length, next = ArgMax(logits[0]);
            for (int t = 0; t < 90; t++)
            {
                Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { next } }, new[] { length }, logits));
                trace.Add((float[])logits[0].Clone());
                next = ArgMax(logits[0]);
                length++;
            }
            advances = model.DecodeAdvances;
            return trace;
        }

        List<float[]> traced = Run(advance: false, out int none);
        List<float[]> advanced = Run(advance: true, out int advances);
        Assert.Equal(0, none);
        // The dense steps after the first two advance (each stage's), the selecting ones trace.
        Assert.True(advances >= (stages ? 80 : 40), $"{advances} decode steps advanced");
        Assert.True(advances < (stages ? 180 : 90), $"{advances} decode steps advanced: the selecting ones should trace");
        for (int i = 0; i < traced.Count; i++)
            Assert.True(traced[i].AsSpan().SequenceEqual(advanced[i]), $"step {i}: advanced logits differ");
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void RankGraphsMatchOneDeviceOnTheManagedCpu(bool qsa, bool chunkedPrompt)
    {
        // BF16 projections and experts: Q8_0 ones quantize the activations, and the ranks' repacked in-projections
        // round apart from one device's by that noise (8.5e-3 for Qwen 3.5's dense fixture, 1.7e-5 in BF16).
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, $"q4e-tp-{qsa}.gguf"), indexerTopK: 64, qsa: qsa,
            projectionType: SyntheticGguf.GgmlType.BF16, expertGateUpType: SyntheticGguf.GgmlType.BF16, expertDownType: SyntheticGguf.GgmlType.BF16);
        var group = new SimulatedDeviceGroup(2);
        ModelBase Placed()
        {
            var model = new Qwen4ExpRuntimeModel(path, BackendType.Cpu, tpDegree: 2, tpGroup: group);
            Assert.Equal(2, model.Mesh.TensorParallel);
            return model;
        }
        AssertClose(Run(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu), chunkedPrompt), Run(Placed, chunkedPrompt), 1e-4,
            $"tp2 qsa={qsa} chunked={chunkedPrompt}");
        Assert.True(group.AllReduceCount > 0, "the ranks never reduced");
    }

    [Fact]
    public void RankGraphsSelectTheSameQsaBlocksAsOneDevice()
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-sparse-tp.gguf"), indexerTopK: 8, qsa: true,
            projectionType: SyntheticGguf.GgmlType.BF16, expertGateUpType: SyntheticGguf.GgmlType.BF16, expertDownType: SyntheticGguf.GgmlType.BF16);
        int[] prompt = Enumerable.Range(0, 40).Select(i => 3 + (i * 37) % 240).ToArray();
        int[] decode = Enumerable.Range(0, 12).Select(i => 5 + (i * 11) % 200).ToArray();
        float[][] Rows(ModelBase model)
        {
            using (model)
            {
                var rows = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
                foreach (int token in decode) rows.Add((float[])model.Forward(new[] { token }).Clone());
                return rows.ToArray();
            }
        }
        double[] errors = RowErrors(Rows(new Qwen4ExpRuntimeModel(path, BackendType.Cpu)),
            Rows(new Qwen4ExpRuntimeModel(path, BackendType.Cpu, tpDegree: 2, tpGroup: new SimulatedDeviceGroup(2))));
        // Partial sums only reorder the additions: one near-tied block pick a run is allowed, within a block's worth.
        Assert.True(errors.Count(e => e >= 1e-4) <= 1 && errors.All(e => e < 5e-2),
            $"tp2 sparse qsa relL2 per row: {string.Join(", ", errors.Select(e => e.ToString("E2")))}");
    }

    [Fact]
    public void SpeculationMatchesPlainGreedyOnTheManagedCpu()
    {
        var (target, mtp) = SpecFixtures(qsa: false);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Cpu), "qwen4exp cpu");
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Cpu, mtp), "qwen4exp cpu mtp");
    }

    [MlxFact]
    public void SpeculationMatchesPlainGreedyOnMlx()
    {
        var (target, mtp) = SpecFixtures(qsa: false);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Mlx), "qwen4exp mlx");
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Mlx, mtp), "qwen4exp mlx mtp");
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(1e-2)]
    public void SpeculationMatchesPlainGreedyOnGgmlCpu(double tieTolerance) => SpeculationOnGgml(BackendType.GgmlCpu, tieTolerance);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(0)]
    public void SpeculationMatchesPlainGreedyOnGgmlMetal(double tieTolerance) => SpeculationOnGgml(BackendType.GgmlMetal, tieTolerance);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(1e-2)]
    public void SpeculationMatchesPlainGreedyOnGgmlCuda(double tieTolerance) => SpeculationOnGgml(BackendType.GgmlCuda, tieTolerance);

    /// <summary>ggml_cpu and ggml_cuda round a verify window's activations unlike one row's: the streams may part
    /// only at near-ties, to the runner-up.</summary>
    private void SpeculationOnGgml(BackendType backend, double tieTolerance)
    {
        var (target, mtp) = SpecFixtures(qsa: true);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, backend), $"qwen4exp {backend}", tieTolerance: tieTolerance);
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, backend, mtp), $"qwen4exp {backend} mtp",
            tieTolerance: tieTolerance);
    }

    /// <summary>The runtime's draft layer computes what Qwen4ExpModel's MTP head does (which runs on GGML only):
    /// prompt catch-up and two chained draft steps.</summary>
    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlMetal)]
    public void MtpDraftsMatchQwen4ExpModelOnMetal(BackendType backend) => MtpDraftsMatch(backend);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void MtpDraftsMatchQwen4ExpModelOnGgmlCpu(BackendType backend) => MtpDraftsMatch(backend);

    private void MtpDraftsMatch(BackendType backend)
    {
        var (target, mtp) = SpecFixtures(qsa: true);
        float[][] legacy = LegacyGolden("mtp_drafts_ggmlcpu");
        List<float[]> actual;
        using (var runtime = new Qwen4ExpRuntimeModel(target, backend, mtp))
            actual = DraftTrace(runtime);
        double[] rel = Errors(legacy, actual.ToArray());
        Assert.True(rel.All(r => r <= 3e-2), $"{backend}: draft rows relL2 from Qwen4ExpModel: {string.Join(", ", rel.Select(r => r.ToString("E2")))}");
    }

    // ---- DFlash: the drafter reads the hc-stream mean of the residual entering its target layers

    private (string Target, string Draft) DFlashFixtures(bool qsa)
        => (Fixture(qsa), Qwen4ExpSyntheticModelBuilder.WriteDFlash(Path.Combine(_dir, "q4e-dflash.gguf")));

    /// <summary>The feature rows a prompt and a verify capture, and their logits, match Qwen4ExpModel's native taps
    /// (which run on GGML only).</summary>
    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlMetal)]
    public void DFlashFeaturesMatchQwen4ExpModelOnMetal(BackendType backend) => DFlashFeaturesMatch(backend);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void DFlashFeaturesMatchQwen4ExpModelOnGgmlCpu(BackendType backend) => DFlashFeaturesMatch(backend);

    private void DFlashFeaturesMatch(BackendType backend)
    {
        var (target, draft) = DFlashFixtures(qsa: true);
        float[][] legacy = LegacyGolden("dflash_features_ggmlcpu"), actual;
        using (var runtime = new Qwen4ExpRuntimeModel(target, backend, draft))
            actual = FeatureTrace(runtime);
        double[] rel = Errors(legacy, actual);
        Assert.True(rel.All(r => r <= 2e-2), $"{backend}: DFlash rows relL2 from Qwen4ExpModel: {string.Join(", ", rel.Select(r => r.ToString("E2")))}");
    }

    /// <summary>Each prompt row's features, then a 3-row verify's features and logits.</summary>
    private static float[][] FeatureTrace(ModelBase model)
    {
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        var head = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.IDraftHead>(model);
        Assert.Equal(TensorSharp.Runtime.Speculative.DraftHeadKind.Block, head.DraftHeadKind);
        int width = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        Assert.Equal(3 * Qwen4ExpSyntheticModelBuilder.Hidden, width);
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        int[] window = { 41, 9, 130 };
        var rows = new List<float[]>();
        model.ResetKVCache();
        var hAll = new float[prompt.Length * width];
        var logits = new float[vocab];
        spec.SpecForward(prompt, hAll, logits, false);
        for (int r = 0; r < prompt.Length; r++) rows.Add(hAll[(r * width)..((r + 1) * width)]);
        rows.Add((float[])logits.Clone());
        var hWindow = new float[window.Length * width];
        var windowLogits = new float[window.Length * vocab];
        spec.SpecForward(window, hWindow, windowLogits, true);
        for (int r = 0; r < window.Length; r++)
        {
            rows.Add(hWindow[(r * width)..((r + 1) * width)]);
            rows.Add(windowLogits[(r * vocab)..((r + 1) * vocab)]);
        }
        // Every tapped layer of every row was written (zeros would compare equal to zeros).
        foreach (float[] features in hAll.Chunk(width).Concat(hWindow.Chunk(width)))
            for (int tap = 0; tap < 3; tap++)
                Assert.Contains(features.Skip(tap * Qwen4ExpSyntheticModelBuilder.Hidden).Take(Qwen4ExpSyntheticModelBuilder.Hidden), v => v != 0);
        return rows.ToArray();
    }

    [Fact]
    public void DFlashSpeculationMatchesPlainGreedyOnTheManagedCpu()
    {
        var (target, draft) = DFlashFixtures(qsa: false);
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Cpu, draft), "qwen4exp cpu dflash");
    }

    /// <summary>The engine's unified step drafts each request's blocks over its own DFlash ring (the prompt's
    /// feature rows captured by the batched prefill) and verifies every window in one forward: one request or several
    /// at once give the plain greedy streams; so do n-gram drafts.</summary>
    [Theory]
    [InlineData(SpeculatorRegistry.Block)]
    [InlineData(SpeculatorRegistry.NGram)]
    public async Task EngineSpeculationMatchesPlainOnTheManagedCpu(string speculator)
    {
        var (target, draft) = DFlashFixtures(qsa: false);
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = speculator, MaxDraftTokens = 3, MinDraftProb = 0 };
        await SpeculativeRuntimeChecks.AssertEngineSpeculationMatchesPlain(() => new Qwen4ExpRuntimeModel(target, BackendType.Cpu, draft),
            options, $"qwen4exp cpu {speculator}");
    }

    [MlxFact]
    public void DFlashSpeculationMatchesPlainGreedyOnMlx()
    {
        var (target, draft) = DFlashFixtures(qsa: false);
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, BackendType.Mlx, draft), "qwen4exp mlx dflash");
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlCpu, 1e-2)]
    public void DFlashSpeculationMatchesPlainGreedyOnGgmlCpu(BackendType backend, double tieTolerance) => DFlashSpeculationOnGgml(backend, tieTolerance);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlMetal, 0)]
    public void DFlashSpeculationMatchesPlainGreedyOnGgmlMetal(BackendType backend, double tieTolerance) => DFlashSpeculationOnGgml(backend, tieTolerance);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCuda, 1e-2)]
    public void DFlashSpeculationMatchesPlainGreedyOnGgmlCuda(BackendType backend, double tieTolerance) => DFlashSpeculationOnGgml(backend, tieTolerance);

    private void DFlashSpeculationOnGgml(BackendType backend, double tieTolerance)
    {
        var (target, draft) = DFlashFixtures(qsa: true);
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() => new Qwen4ExpRuntimeModel(target, backend, draft), $"qwen4exp {backend} dflash",
            tieTolerance: tieTolerance);
    }

    /// <summary>Logits of the trunk and of the head's prompt catch-up and two chained drafts.</summary>
    private static List<float[]> DraftTrace(ModelBase model)
    {
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        var head = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.IDraftHead>(model);
        Assert.Equal(TensorSharp.Runtime.Speculative.DraftHeadKind.PerToken, head.DraftHeadKind);
        int width = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        int n = prompt.Length;
        var rows = new List<float[]>();
        var hAll = new float[n * width];
        var logits = new float[vocab];
        model.ResetKVCache();
        spec.SpecForward(prompt, hAll, logits, false);
        rows.Add((float[])logits.Clone());
        // As the speculative executor commits a prompt: from position 0, row k the hidden state of the token before
        // tokens[k] (zeros before the first).
        head.DraftCatchUp(prompt, new float[width].Concat(hAll[..((n - 1) * width)]).ToArray(), 0);
        var draft = new float[vocab];
        var h1 = new float[width];
        head.DraftStep(ArgMax(logits), hAll[((n - 1) * width)..], n, draft, h1);
        rows.Add((float[])draft.Clone());
        var h2 = new float[width];
        head.DraftStep(ArgMax(draft), h1, n + 1, draft, h2);
        rows.Add((float[])draft.Clone());
        return rows;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static double[] Errors(float[][] expected, float[][] actual)
    {
        var errors = new double[expected.Length];
        for (int i = 0; i < expected.Length; i++)
        {
            double num = 0, den = 0;
            for (int j = 0; j < expected[i].Length; j++)
            {
                double d = expected[i][j] - actual[i][j];
                num += d * d;
                den += (double)expected[i][j] * expected[i][j];
            }
            errors[i] = Math.Sqrt(num / Math.Max(den, 1e-30));
        }
        return errors;
    }

    private static float[][] Run(Func<ModelBase> create, bool chunkedPrompt)
    {
        using ModelBase model = create();
        model.ResetKVCache();
        var rows = new List<float[]>();
        if (chunkedPrompt)
        {
            model.ForwardRefill(Prompt[..6]);
            rows.Add((float[])model.Forward(Prompt[6..]).Clone());
        }
        else
        {
            rows.Add((float[])model.ForwardRefill(Prompt).Clone());
        }
        foreach (int token in Decode)
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        return rows.ToArray();
    }

    internal static void AssertClose(float[][] expected, float[][] actual, double tolerance, string label)
    {
        double[] errors = RowErrors(expected, actual);
        Assert.True(errors.All(e => e < tolerance), $"{label} relL2 per row: {string.Join(", ", errors.Select(e => e.ToString("E2")))}");
    }

    internal static double[] RowErrors(float[][] expected, float[][] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        var errors = new double[expected.Length];
        for (int i = 0; i < expected.Length; i++)
        {
            double num = 0, den = 0;
            for (int j = 0; j < expected[i].Length; j++)
            {
                double d = expected[i][j] - actual[i][j];
                num += d * d;
                den += (double)expected[i][j] * expected[i][j];
            }
            errors[i] = Math.Sqrt(num / Math.Max(den, 1e-30));
        }
        return errors;
    }
}

/// <summary>
/// The runtime's prompt chunks on Metal's prompt-width kernels (TensorSharp's tsm kernels as device operations between
/// graph segments, as Qwen4ExpModel runs them): a 66-row prompt is wide enough for them (64 rows,
/// TS_METAL_PROMPT_OPS_MIN_ROWS) and within the dense QSA budget of a top-k of 64 (67 cells). In the collection of
/// the other tests that count the kernels' passes.
/// </summary>
[Collection("MoeCpuOffloadConfig")]
public sealed class Qwen4ExpRuntimePromptKernelTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("q4e-prompt-kernels").FullName;
    private readonly ITestOutputHelper _output;

    public Qwen4ExpRuntimePromptKernelTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly int[] Prompt = Enumerable.Range(0, 66).Select(i => 3 + (i * 37) % 240).ToArray();
    private static readonly int[] Decode = { 41, 9, 130, 77 };

    private static float[][] Rows(ModelBase model, int firstChunk)
    {
        using (model)
        {
            model.ResetKVCache();
            var rows = new List<float[]>();
            if (firstChunk < Prompt.Length)
            {
                model.ForwardRefill(Prompt[..firstChunk]);
                rows.Add((float[])model.Forward(Prompt[firstChunk..]).Clone());
            }
            else
            {
                rows.Add((float[])model.ForwardRefill(Prompt).Clone());
            }
            foreach (int token in Decode)
                rows.Add((float[])model.Forward(new[] { token }).Clone());
            return rows.ToArray();
        }
    }

    /// <summary>The delta-net layers' mixer (GraphFusion.DeltaNetPrompt) on tsm_gdn_prompt and every half-layer's
    /// hyper-connection read (GraphFusion.HcRead) on tsm_hc_read: the same rows as the ops on Metal (the prompt in
    /// two 33-row chunks, below the kernels' width), and as the managed runtime.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void PromptKernelsAnswerAsTheirOps()
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 64, qsa: true);
        float[][] ops = Rows(new Qwen4ExpRuntimeModel(path, BackendType.GgmlMetal), 33);
        long gdnBefore = GgmlBasicOps.TsmGdnPromptCount(), readBefore = GgmlBasicOps.TsmHcReadCount();
        float[][] kernel = Rows(new Qwen4ExpRuntimeModel(path, BackendType.GgmlMetal), Prompt.Length);
        long layers = GgmlBasicOps.TsmGdnPromptCount() - gdnBefore, reads = GgmlBasicOps.TsmHcReadCount() - readBefore;
        float[][] managed = Rows(new Qwen4ExpRuntimeModel(path, BackendType.Cpu), Prompt.Length);
        _output.WriteLine($"gdn kernel layers {layers}, hc reads {reads}; kernel vs ops {string.Join(" ", Qwen4ExpRuntimeTests.RowErrors(ops, kernel).Select(e => e.ToString("E1")))}; "
            + $"kernel vs managed {string.Join(" ", Qwen4ExpRuntimeTests.RowErrors(managed, kernel).Select(e => e.ToString("E1")))}");
        Assert.Equal(6, layers);   // the fixture's six delta-net layers, once each
        Assert.Equal(16, reads);   // its eight layers' two half-layers
        Qwen4ExpRuntimeTests.AssertClose(ops, kernel, 1e-3, "metal kernel vs ops");   // 4.5e-5 measured (M5 Pro)
        Qwen4ExpRuntimeTests.AssertClose(managed, kernel, 2e-2, "metal kernel vs managed");
    }

    /// <summary>A prompt chunk past the QSA budget attends only each row's chosen cells on TensorSharp's kernels
    /// (tsm_qsa_compact, tsm_qsa_attn) where its window is wide enough (8K cells on the tensor API; the test hook lowers
    /// it to the fixture's): the same rows as ggml's flash attention under the selection's mask. The kernels read an
    /// F16 cache, pinned for this test's models only.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void SelectedAttentionAnswersAsTheMaskedFlashAttention()
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-selected.gguf"), indexerTopK: 8, qsa: true);   // dense up to 11
        using IDisposable f16 = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        float[][] masked = Rows(new Qwen4ExpRuntimeModel(path, BackendType.GgmlMetal), Prompt.Length);
        float[][] selected;
        long passes;
        GgmlBasicOps.TsmQsaAttnSetMinCells(1);
        try
        {
            long before = GgmlBasicOps.TsmQsaAttnCount();
            selected = Rows(new Qwen4ExpRuntimeModel(path, BackendType.GgmlMetal), Prompt.Length);
            passes = GgmlBasicOps.TsmQsaAttnCount() - before;
        }
        finally
        {
            GgmlBasicOps.TsmQsaAttnSetMinCells(0);
        }
        _output.WriteLine($"selected attention passes {passes}; vs the masked flash attention "
            + string.Join(" ", Qwen4ExpRuntimeTests.RowErrors(masked, selected).Select(e => e.ToString("E1"))));
        Assert.Equal(2, passes);   // the fixture's two QSA layers in the prompt chunk (decode rows take ggml's sparse path)
        Qwen4ExpRuntimeTests.AssertClose(masked, selected, 1e-3, "selected attention vs masked flash attention");   // 4.8e-5 measured (M5 Pro)
    }
}
