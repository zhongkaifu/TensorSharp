// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// LFM3 on the direct CUDA executor (CudaGraphExecutor, BackendType.Cuda): the mechanisms the rest of Lfm3RuntimeTests proves on
// the managed and GGML executors, on the executor that records decode steps as CUDA graphs, attends F16 caches through
// capacity windows under band masks (the flash decode skipping the tiles below a one-row band's floor), runs NVFP4 products
// without FP4 tensor cores below sm_120, and keeps short-convolution windows in device states. F32 and F16 K/V each, the
// latter the default cache type. The pinned GGML lanes (TS_TEST_GGML_BACKEND=cuda / vulkan) run the "ggml" cases of the
// other files on ggml-cuda and ggml-vulkan.
using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed partial class Lfm3RuntimeTests
{
    /// <summary>The backend an executor name of these tests runs on: "ggml" the lane's pinned GGML backend, "cuda" /
    /// "cuda-per-op" direct CUDA, the rest ("managed", "per-op") the managed CPU executor.</summary>
    private static BackendType BackendOf(string executor) => executor switch
    {
        "ggml" => TestGates.PinnedGgmlBackend,
        "cuda" or "cuda-per-op" => BackendType.Cuda,
        _ => BackendType.Cpu,
    };

    /// <summary>Whether an executor name of these tests runs the per-op layers instead of graphs.</summary>
    private static bool PerOp(string executor) => executor.EndsWith("per-op", StringComparison.Ordinal);

    // Direct CUDA's distances, measured on an A40 (sm_86), driver 595.91, CUDA 13.2, 2026-11-17. F32 weights against the
    // double-precision reference forward: 2e-7 .. 1.7e-6 where every step is a few rows, 3-7e-4 after a prompt step of 21+
    // rows (its products and attention take half operands); BF16 fixtures where row counts differ (prompt vs decode, graphs vs
    // the per-op products) 1-1.5e-2 (BF16 activations, RoundsActivations).
    private const double CudaReferenceTolerance = 2e-3;
    private const double CudaDenseTolerance = 2.5e-2;

    // ---------------------------------------------------------------- the recipe

    [CudaTheory]
    [InlineData("graphs", KvCacheDtype.F32)]
    [InlineData("graphs", KvCacheDtype.F16)]
    [InlineData("per-op", KvCacheDtype.F32)]
    [InlineData("per-op", KvCacheDtype.F16)]
    public void DirectCudaMatchesTheReferenceForward(string path, KvCacheDtype kv)
        => AssertMatchesReference(BackendType.Cuda, path == "graphs", CudaReferenceTolerance, kv);

    [CudaTheory]
    [InlineData("graphs", KvCacheDtype.F32)]
    [InlineData("graphs", KvCacheDtype.F16)]
    [InlineData("per-op", KvCacheDtype.F16)]
    public void DirectCudaRingsMatchTheReferenceForward(string path, KvCacheDtype kv)
    {
        using var rings = SmallRings();
        AssertMatchesReference(BackendType.Cuda, path == "graphs", CudaReferenceTolerance, kv);
    }

    /// <summary>The production rings (768 cells, 512-row steps) under a 1,300-token prompt and decode steps past the wrap.</summary>
    [CudaTheory]
    [InlineData(KvCacheDtype.F32)]
    [InlineData(KvCacheDtype.F16)]
    public void DirectCudaLongPromptsOverProductionRingsMatchTheReferenceForward(KvCacheDtype kv)
        => AssertLongPromptMatchesReference(BackendType.Cuda, graphs: true, kv, CudaReferenceTolerance);

    [CudaTheory]
    [InlineData("dense")]
    [InlineData("q8")]
    public void DirectCudaMatchesTheManagedReference(string variant) => CompareBackend(variant, BackendType.Cuda);

    [CudaTheory]
    [InlineData("dense")]
    [InlineData("q8")]
    public void DirectCudaGraphsMatchThePerOpPath(string variant) => AssertGraphMatchesPerOp(variant, BackendType.Cuda);

    [CudaFact]
    public void GraphAndPerOpStepsShareTheStateOnDirectCuda() => AssertGraphAndPerOpShareTheState(BackendType.Cuda);

    /// <summary>An explicit q8_0 K/V cache on direct CUDA: a stack with sliding-window layers keeps it off the graphs (their
    /// bands are selections the executor reads only over F16 caches, and its q8_0 row writes are not taken modulo a ring's
    /// cells), and runs the per-op path; a stack without them runs q8_0 as graphs on the flash kernels.</summary>
    [CudaFact(TensorCoreModule = "tensorsharp_flash_prefill")]
    public void DirectCudaKeepsQ8KvOffTheGraphsOfAStackWithSlidingWindows()
    {
        using IDisposable q8 = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q8_0);
        string sliding = new Lfm3SyntheticModelBuilder { HeadDimension = 128 }.Write(Path.Combine(_dir, "lfm3-q8-sliding.gguf"));
        using (Lfm3RuntimeModel model = Create(sliding, backend: BackendType.Cuda))
            Assert.Equal($"KV dtype Q8_0 with {model.Config.Architecture}'s attention routes on this backend's executor", model.GraphRefusalForTests);
        string full = new Lfm3SyntheticModelBuilder { Layout = "cfcfcf", HeadDimension = 128 }.Write(Path.Combine(_dir, "lfm3-q8-full.gguf"));
        using (Lfm3RuntimeModel model = Create(full, backend: BackendType.Cuda))
            Assert.Null(model.GraphRefusalForTests);
    }

    // ---------------------------------------------------------------- the sliding window

    [CudaTheory]
    [InlineData(KvCacheDtype.F32, false)]
    [InlineData(KvCacheDtype.F16, false)]
    [InlineData(KvCacheDtype.F32, true)]
    [InlineData(KvCacheDtype.F16, true)]
    public void DirectCudaPromptMatchesTokenByTokenAndChunkedAcrossTheWindow(KvCacheDtype kv, bool rings)
    {
        using IDisposable scope = rings ? SmallRings() : null;
        AssertPromptPaths(BackendType.Cuda, kv);
    }

    /// <summary>The same on F32 weights, where no product rounds BF16 activations: the band masks of whole prompts, chunks that
    /// straddle the window and decode rows agree to the half-operand prompt products' rounding.</summary>
    [CudaTheory]
    [InlineData(KvCacheDtype.F32, false)]
    [InlineData(KvCacheDtype.F16, false)]
    [InlineData(KvCacheDtype.F16, true)]
    public void DirectCudaPromptOnF32WeightsMatchesTokenByTokenAndChunkedAcrossTheWindow(KvCacheDtype kv, bool rings)
    {
        using IDisposable scope = rings ? SmallRings() : null;
        AssertPromptPaths(BackendType.Cuda, kv, "f32", CudaReferenceTolerance);
    }

    [CudaTheory]
    [InlineData(KvCacheDtype.F32, false)]
    [InlineData(KvCacheDtype.F16, false)]
    [InlineData(KvCacheDtype.F16, true)]
    public void DirectCudaRingsGiveTheLogitsOfLinearCaches(KvCacheDtype kv, bool perOp)
        => AssertRingsMatchLinear(BackendType.Cuda, graphs: !perOp, kv);

    /// <summary>
    /// Decode steps of one request and of two together, advanced in place, give the traced steps' logits bit for bit, past the
    /// window and (with rings) round the ring. Direct CUDA records an advanced step as a CUDA graph and replays it. Where its
    /// flash kernels attend the caches (F16, 128-wide heads: capacity windows) the sliding layers' bands are read on the device
    /// (GraphCapabilities.BandMasksOnDevice), so the steps past the window record and replay too - with rings, their writes
    /// taken round the ring at every replay. Elsewhere a band is classified on the host, and a step whose band closes a cell
    /// inside its window runs as its plan (the runtime counts it).
    /// </summary>
    [CudaTheory]
    [InlineData(KvCacheDtype.F32, false, 64)]
    [InlineData(KvCacheDtype.F16, false, 64)]
    [InlineData(KvCacheDtype.F16, true, 64)]
    [InlineData(KvCacheDtype.F16, false, 128)]
    [InlineData(KvCacheDtype.F16, true, 128)]
    public void DirectCudaAdvancedDecodeStepsMatchTracedOnes(KvCacheDtype kv, bool rings, int headDim)
    {
        using IDisposable scope = rings ? SmallRings() : null;
        bool deviceBands = kv == KvCacheDtype.F16 && headDim == 128;
        AssertAdvancedDecodeMatchesTraced(BackendType.Cuda, kv, Fixture(headDim == 128 ? "dense128" : "dense"), advanced: model =>
        {
            _output.WriteLine($"kv={kv} rings={rings} d={headDim}: {model.DecodeAdvances} advances, {model.DecodeRecordings} recordings, " +
                $"{model.RecordedDecodeReplays} replays, {model.SlidingWindowUnrecordedSteps} sliding-window steps unrecorded");
            if (!deviceBands)
            {
                Assert.True(model.SlidingWindowUnrecordedSteps > 250, $"{model.SlidingWindowUnrecordedSteps} steps past the window ran unrecorded");
                Assert.Equal(0, model.RecordedDecodeReplays);
            }
            else
            {
                Assert.Equal(0, model.SlidingWindowUnrecordedSteps);
                Assert.True(model.DecodeRecordings >= 2, $"{model.DecodeRecordings} recordings (one lone, one batched expected)");
                Assert.True(model.RecordedDecodeReplays > 250, $"{model.RecordedDecodeReplays} steps replayed a recording");
            }
        });
    }

    /// <summary>
    /// A stack without sliding-window layers (short convolutions and full attention) records its advanced decode steps on
    /// direct CUDA - one sequence's and two sequences' together - and replays them: the logits are the traced steps', bit for
    /// bit, with the short-convolution windows advanced inside the recordings.
    /// </summary>
    [CudaTheory]
    [InlineData(KvCacheDtype.F32)]
    [InlineData(KvCacheDtype.F16)]
    public void DirectCudaRecordsTheDecodeStepsOfAStackWithoutSlidingWindows(KvCacheDtype kv)
    {
        string path = new Lfm3SyntheticModelBuilder { Layout = "cfcfcf", UntiedOutput = true }.Write(Path.Combine(_dir, "lfm3-full.gguf"));
        AssertAdvancedDecodeMatchesTraced(BackendType.Cuda, kv, path, advanced: model =>
        {
            _output.WriteLine($"kv={kv}: {model.DecodeAdvances} advances, {model.DecodeRecordings} recordings, {model.RecordedDecodeReplays} replays");
            Assert.Equal(0, model.SlidingWindowUnrecordedSteps);
            Assert.True(model.DecodeRecordings >= 2, $"{model.DecodeRecordings} recordings (one lone, one batched expected)");
            Assert.True(model.RecordedDecodeReplays > 250, $"{model.RecordedDecodeReplays} steps replayed a recording");
        });
    }

    /// <summary>
    /// A request whose decode starts below the window: its advanced steps record and replay; until the sliding layers' band
    /// closes a cell inside its window (position W), then they run as plans (reported once) - except where the flash kernels
    /// attend the caches (F16, 128-wide heads), whose bands are read on the device: the one recording replays through the
    /// window. Every logit is the traced steps'.
    /// </summary>
    [CudaTheory]
    [InlineData(KvCacheDtype.F32, 64)]
    [InlineData(KvCacheDtype.F16, 64)]
    [InlineData(KvCacheDtype.F16, 128)]
    public void DirectCudaRecordsDecodeStepsUntilTheBandSlides(KvCacheDtype kv, int headDim)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture(headDim == 128 ? "dense128" : "dense");
        const int steps = 30;
        int[] prompt = { 41, 52 };
        (List<float[]> Rows, int Recordings, int Replays, int Unrecorded) Run(bool advance)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.DecodeAdvanceDisabled = !advance;
            var unified = (IUnifiedBatchModel)model;
            unified.EnsureSequenceSlot("a", prompt.Length + steps + 8);
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, logits));
            var rows = new List<float[]> { (float[])logits[0].Clone() };
            int next = ArgMax(logits[0]), length = prompt.Length;
            for (int t = 0; t < steps; t++)
            {
                Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { next } }, new[] { length }, logits));
                rows.Add((float[])logits[0].Clone());
                next = ArgMax(logits[0]);
                length++;
            }
            return (rows, model.DecodeRecordings, model.RecordedDecodeReplays, model.SlidingWindowUnrecordedSteps);
        }
        var traced = Run(advance: false);
        var advanced = Run(advance: true);
        _output.WriteLine($"kv={kv}: {advanced.Recordings} recordings, {advanced.Replays} replays, {advanced.Unrecorded} unrecorded past the band");
        // Steps at positions 2 (traced), 3 (advanced: recorded), 4.. (replayed): over F32 caches to 7, as from 8 on the band
        // closes cell 0; over F16 caches to the last.
        Assert.Equal(1, advanced.Recordings);
        if (kv == KvCacheDtype.F32 || headDim != 128)
        {
            Assert.Equal(Lfm3SyntheticModelBuilder.Window - 4, advanced.Replays);
            Assert.Equal(prompt.Length + steps - Lfm3SyntheticModelBuilder.Window, advanced.Unrecorded);
        }
        else
        {
            Assert.Equal(steps - 2, advanced.Replays);
            Assert.Equal(0, advanced.Unrecorded);
        }
        for (int i = 0; i < traced.Rows.Count; i++)
            Assert.True(traced.Rows[i].AsSpan().SequenceEqual(advanced.Rows[i]),
                $"kv={kv} row {i}: advanced logits differ (relL2 {RelL2(advanced.Rows[i], traced.Rows[i]):E2})");
    }

    [CudaTheory]
    [InlineData("cuda", KvCacheDtype.F32)]
    [InlineData("cuda", KvCacheDtype.F16)]
    [InlineData("cuda-per-op", KvCacheDtype.F16)]
    public void DirectCudaVerifyWindowsAcrossTheWindowCommitTheKeptRows(string executor, KvCacheDtype kv)
        => AssertVerifyWindowCommits(executor, new[] { 41, 52, 63, 74, 85, 96 }, kv, CudaDenseTolerance);

    [CudaTheory]
    [InlineData("cuda", KvCacheDtype.F32)]
    [InlineData("cuda", KvCacheDtype.F16)]
    public void DirectCudaVerifyWindowsAcrossTheRingWrapCommitTheKeptRows(string executor, KvCacheDtype kv)
    {
        using var rings = SmallRings();
        AssertVerifyWindowCommits(executor, Enumerable.Range(0, 14).Select(i => 41 + (i * 11) % 90).ToArray(), kv, CudaDenseTolerance);
    }

    /// <summary>
    /// Verify windows of 4 rows starting at every position from 1 to W + 2 (each keeps one row, so the next starts a position
    /// on): the window advances in place, and direct CUDA records it. Over F32 caches it records while its rows sit below the
    /// band (start + 4 &lt;= W): from start W - 3 its LAST rows' band closes cells inside the window while its first row's does
    /// not, so a replay would attend those rows under the floor the recording classified (0), and such steps run as plans.
    /// Where the flash kernels attend the caches (F16, 128-wide heads) each row's band is read on the device, so the recording
    /// replays across the band edge. Every row is plain decoding's.
    /// </summary>
    [CudaTheory]
    [InlineData(KvCacheDtype.F32, 64)]
    [InlineData(KvCacheDtype.F16, 64)]
    [InlineData(KvCacheDtype.F16, 128)]
    public void DirectCudaVerifyWindowsCrossingTheBandRowByRowMatchPlainDecoding(KvCacheDtype kv, int headDim)
        => AssertVerifyWindowsAcrossTheBandEdge(BackendType.Cuda, kv, CudaDenseTolerance, model =>
        {
            _output.WriteLine($"kv={kv} d={headDim}: {model.WindowAdvances} window advances, {model.DecodeRecordings} recordings, " +
                $"{model.RecordedDecodeReplays} replays, {model.SlidingWindowUnrecordedSteps} unrecorded");
            Assert.True(model.DecodeRecordings >= 1, "no verify window was recorded");
            if (kv == KvCacheDtype.F32 || headDim != 128)
            {
                Assert.True(model.RecordedDecodeReplays >= 1, "no recorded verify window was replayed below the band");
                Assert.True(model.SlidingWindowUnrecordedSteps >= 3, $"{model.SlidingWindowUnrecordedSteps} windows crossing the band ran unrecorded");
            }
            else
            {
                Assert.Equal(0, model.SlidingWindowUnrecordedSteps);
                Assert.True(model.RecordedDecodeReplays >= Lfm3SyntheticModelBuilder.Window - 2,
                    $"{model.RecordedDecodeReplays} recorded verify windows replayed (across the band edge expected)");
            }
        }, headDim == 128 ? "dense128" : "dense");

    /// <summary><see cref="DirectCudaVerifyWindowsCrossingTheBandRowByRowMatchPlainDecoding"/> on the managed executor and the
    /// pinned GGML backend (whose steps advance without recordings).</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void VerifyWindowsCrossingTheBandRowByRowMatchPlainDecoding(string executor)
    {
        BackendType backend = BackendOf(executor);
        AssertVerifyWindowsAcrossTheBandEdge(backend, KvCacheDtype.F32, RoundsActivations(backend) ? 2.5e-2 : 2e-3,
            model => Assert.True(model.WindowAdvances > 0, "no verify window advanced in place"));
    }

    private void AssertVerifyWindowsAcrossTheBandEdge(BackendType backend, KvCacheDtype kv, double tolerance, Action<Lfm3RuntimeModel> check,
        string variant = "dense")
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture(variant);
        const int count = 4, last = Lfm3SyntheticModelBuilder.Window + 2;
        int[] tokens = Enumerable.Range(0, last + count).Select(i => 41 + (i * 11) % 90).ToArray();
        var plain = new List<float[]>();
        using (var reference = Create(path, backend: backend))
        {
            plain.Add((float[])reference.ForwardRefill(new[] { tokens[0] }).Clone());
            for (int p = 1; p < tokens.Length; p++) plain.Add((float[])reference.Forward(new[] { tokens[p] }).Clone());
        }

        using var model = Create(path, backend: backend);
        var spec = (ISpeculativeModel)model;
        int vocab = model.Config.VocabSize;
        model.ForwardRefill(new[] { tokens[0] });
        var logits = new float[count * vocab];
        var worst = new double[last + 1];
        for (int start = 1; start <= last; start++)
        {
            spec.SpecForward(tokens.AsSpan(start, count).ToArray(), null, logits, allLogitsRows: true);
            for (int r = 0; r < count; r++)
                worst[start] = Math.Max(worst[start], RelL2(logits.AsSpan(r * vocab, vocab).ToArray(), plain[start + r]));
            spec.SpecOnVerifyAccepted(0, count);
            spec.SpecRewindCache(start + 1);
        }
        _output.WriteLine($"{backend} kv={kv}: worst relL2 per window start {string.Join(" ", worst.Skip(1).Select(w => w.ToString("E1")))}");
        for (int start = 1; start <= last; start++)
            Assert.True(worst[start] <= tolerance, $"{backend} kv={kv}: the verify window at {start} sits {worst[start]:E2} from plain decoding");
        check(model);
    }

    // ---------------------------------------------------------------- the runtime's mechanisms

    [CudaTheory]
    [InlineData(KvCacheDtype.F32, false)]
    [InlineData(KvCacheDtype.F16, false)]
    [InlineData(KvCacheDtype.F16, true)]
    public void DirectCudaCheckpointsContinueExactly(KvCacheDtype kv, bool rings)
    {
        using IDisposable scope = rings ? SmallRings() : null;
        AssertCheckpointsContinue(BackendType.Cuda, kv, deviceCopies: false);
    }

    [CudaTheory]
    [InlineData(KvCacheDtype.F32, false)]
    [InlineData(KvCacheDtype.F16, false)]
    [InlineData(KvCacheDtype.F16, true)]
    public void DirectCudaSlotBatchesMatchEachSequenceAlone(KvCacheDtype kv, bool rings)
    {
        using IDisposable scope = rings ? SmallRings() : null;
        CompareSlotBatches(1, 1, BackendType.Cuda, kv, CudaDenseTolerance);
    }

    [CudaTheory]
    [InlineData(KvCacheDtype.F16, false)]
    [InlineData(KvCacheDtype.F16, true)]
    public async Task DirectCudaConcurrentRequestsAreBatched(KvCacheDtype kv, bool rings)
    {
        using IDisposable scope = rings ? SmallRings() : null;
        await AssertConcurrentRequestsAreBatched(1, 1, BackendType.Cuda, kv, nearTie: 1e-2);
    }

    /// <summary>Greedy n-gram speculation emits plain greedy's tokens on direct CUDA, past the window and round wrapped rings.</summary>
    [CudaTheory]
    [InlineData(KvCacheDtype.F32, false)]
    [InlineData(KvCacheDtype.F16, false)]
    [InlineData(KvCacheDtype.F16, true)]
    public void DirectCudaSpeculativeDecodingMatchesPlainGreedy(KvCacheDtype kv, bool rings)
    {
        using IDisposable scope = rings ? SmallRings() : null;
        AssertSpeculationMatchesPlainGreedy(BackendType.Cuda, kv, tieTolerance: 1e-2);
    }

    // ---------------------------------------------------------------- NVFP4

    /// <summary>LiquidAI's NVFP4 layout on direct CUDA: the NVFP4 blocks stay as they are (no widening) and their .scale sidecars
    /// apply; below sm_120 (no FP4 tensor cores) the products run on the NVFP4 matvec and the dequantizing GEMM.</summary>
    [CudaFact]
    public void Nvfp4FeedForwardsOnDirectCudaMatchTheManagedReference() => AssertNvfp4(BackendType.Cuda);
}
