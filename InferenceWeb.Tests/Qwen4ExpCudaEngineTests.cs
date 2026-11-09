// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen3.8-Flash-Next (qwen4exp) on the direct-CUDA engine (<c>--backend cuda</c>), held to the native
/// token span <c>--backend ggml_cuda</c> runs, on the tiny checkpoint of
/// <see cref="Qwen4ExpSyntheticModelBuilder"/>: GDN and QSA attention layers, the PLE block, the
/// low-rank hyper-connections and every expert storage type of the shipped file.
/// </summary>
public sealed class Qwen4ExpCudaEngineTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qwen4exp-cuda-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();

    private readonly KvCacheDtype _savedKv = KvCacheDtypeConfig.Current;
    private readonly bool _savedKvExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public Qwen4ExpCudaEngineTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", "1024");
        _env.ClearSpeculationVars();
        // The direct engine's defaults (q8_0 KV, adaptive residency) are pinned off here: these tests hold it to the
        // native span (F16 KV) and to fixed placements bit for bit. The tests of those features turn them on.
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        _env.Set("TS_MOE_TIER_ADAPT", "0");
        // The route-tail skip (on by default) leaves rank-7+ misses out: these tests hold every route computed.
        _env.Set("TS_MOE_ROUTE_TAIL_SKIP", "0");
        // The GPU's share of the missed experts (on by default) moves host-computed answers: these tests leave every miss
        // to the CPU; its own tests set it.
        _env.Set("TS_Q4E_HYBRID_MISSES", "0");
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_savedKv, _savedKvExplicit);
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(int indexerTopK = 16, bool q2kxlExperts = false, bool q4kxlExperts = false,
        SyntheticGguf.GgmlType? expertDownType = null, float expertDownGain = 0.25f,
        SyntheticGguf.GgmlType? expertGateUpType = null, int expertFf = Qwen4ExpSyntheticModelBuilder.ExpertFf)
        => Qwen4ExpSyntheticModelBuilder.Write(
        Path.Combine(_dir, $"qwen4exp-{indexerTopK}{(q2kxlExperts ? "-q2kxl" : "")}{(q4kxlExperts ? "-q4kxl" : "")}" +
            $"{(expertGateUpType is { } gateUp ? "-gateup" + gateUp : "")}" +
            $"{(expertDownType is { } down ? "-down" + down : "")}-ff{expertFf}-g{expertDownGain}.gguf"), indexerTopK,
        q2kxlExperts: q2kxlExperts, q4kxlExperts: q4kxlExperts, expertDownType: expertDownType,
        expertDownGain: expertDownGain, expertGateUpType: expertGateUpType, expertFf: expertFf);

    private static readonly int[] Prompt = Enumerable.Range(0, 40).Select(i => (i * 37 + 11) % 250).ToArray();
    private static readonly int[] Steps = { 17, 203, 99, 4, 150, 61 };

    /// <summary>The prompt's logits, then each decode step's.</summary>
    private static float[][] Run(ModelBase model)
    {
        var rows = new float[Steps.Length + 1][];
        rows[0] = (float[])model.ForwardRefill(Prompt).Clone();
        for (int i = 0; i < Steps.Length; i++)
            rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
        return rows;
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void TheFixture_RunsOnTheNativeTokenSpan_Cpu() => NativeSmoke(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void TheFixture_RunsOnTheNativeTokenSpan_Metal() => NativeSmoke(BackendType.GgmlMetal);

    /// <summary>The fixture is the oracle for the direct engine, so first it has to be one the native
    /// executor takes whole: the token span, the in-span PLE block and QSA past its 19-cell width.</summary>
    private void NativeSmoke(BackendType backend)
    {
        string path = Fixture();
        using ModelBase model = ModelBase.Create(path, backend);
        Assert.IsType<Qwen4ExpModel>(model);
        float[][] rows = Run(model);
        for (int i = 0; i < rows.Length; i++)
        {
            Assert.Equal(Qwen4ExpSyntheticModelBuilder.Vocab, rows[i].Length);
            Assert.All(rows[i], v => Assert.True(float.IsFinite(v)));
            double norm = Math.Sqrt(rows[i].Sum(v => (double)v * v));
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: argmax {ArgMax(rows[i])} |logits| {norm:F3}");
            Assert.True(norm > 1e-3, "degenerate logits");
            if (i > 0)
                Assert.True(RelativeError(rows[i], rows[i - 1]) > 1e-3, "the step did not change the logits");
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void TheFeatureTaps_AverageTheStreamsOfACompletedLayer_Cpu() => NativeFeatureTaps(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void TheFeatureTaps_AverageTheStreamsOfACompletedLayer_Metal() => NativeFeatureTaps(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void TheFeatureTaps_AverageTheStreamsOfACompletedLayer_Cuda() => NativeFeatureTaps(BackendType.GgmlCuda);

    /// <summary>The DFlash feature taps of the native token span (TSGgml_Qwen4ExpSetFeatureTaps): after a tapped layer,
    /// the mean of its hyper-connection streams lands in the tap's column block of each row. The last layer's completed
    /// output is the residual a speculative forward already exports, so its tap must be that residual's stream mean; a
    /// middle layer's tap fills its own block, and a block no tap names stays untouched. A prompt, a decode step and a
    /// verify-width window alike.</summary>
    private void NativeFeatureTaps(BackendType backend)
    {
        string path = Fixture(indexerTopK: 2048);
        using ModelBase model = ModelBase.Create(path, backend);
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        int hcDim = spec.SpecFeatureSize, nEmbd = model.Config.HiddenSize, hc = hcDim / nEmbd;
        int last = model.Config.NumLayers - 1, vocab = model.Config.VocabSize, stride = 3 * nEmbd;
        Assert.Equal(hcDim, hc * nEmbd);
        foreach (int[] tokens in new[] { Prompt, new[] { Steps[0] }, new[] { Steps[1], 77, 123 } })
        {
            var hidden = new float[tokens.Length * hcDim];
            float[] taps = GC.AllocateArray<float>(tokens.Length * stride, pinned: true);
            Array.Fill(taps, float.NaN);
            IntPtr at = System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(taps, 0);
            Assert.True(TensorSharp.GGML.GgmlBasicOps.Qwen4ExpSetFeatureTaps(new[] { 1, last }, at, stride));
            try
            {
                spec.SpecForward(tokens, hidden, new float[tokens.Length * vocab], true);
            }
            finally
            {
                TensorSharp.GGML.GgmlBasicOps.Qwen4ExpSetFeatureTaps(null, IntPtr.Zero, 0);
            }
            double worst = 0;
            for (int r = 0; r < tokens.Length; r++)
            {
                bool middleDiffers = false;
                for (int d = 0; d < nEmbd; d++)
                {
                    float mean = hidden[(long)r * hcDim + d];
                    for (int c = 1; c < hc; c++)
                        mean += hidden[(long)r * hcDim + (long)c * nEmbd + d];
                    mean *= 1.0f / hc;
                    float lastTap = taps[(long)r * stride + nEmbd + d], middle = taps[(long)r * stride + d];
                    worst = Math.Max(worst, Math.Abs(lastTap - mean) / Math.Max(1.0, Math.Abs(mean)));
                    Assert.True(float.IsFinite(middle), $"T={tokens.Length} row {r}: the middle tap left its block unwritten");
                    Assert.True(float.IsNaN(taps[(long)r * stride + 2 * nEmbd + d]), $"T={tokens.Length} row {r}: a tap wrote past its slot");
                    middleDiffers |= middle != lastTap;
                }
                Assert.True(middleDiffers, $"T={tokens.Length} row {r}: the middle tap repeats the last layer's");
            }
            _output.WriteLine($"T={tokens.Length}: last-layer tap vs the exported streams' mean, worst relative error {worst:E2}");
            Assert.True(worst <= 1e-6, $"T={tokens.Length}: the last layer's tap is not its streams' mean ({worst:E2})");
        }
    }

    /// <summary>A prompt chunk older than a block drafter's ring asks for its last row's features only (a one-row
    /// buffer from SpeculativeExecution.PrefillStep); it gets exactly the last row of the full capture.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void ASpeculativePromptChunk_CanAskForItsLastRowOnly_Cpu()
    {
        string path = Fixture(indexerTopK: 2048);
        using ModelBase model = ModelBase.Create(path, BackendType.GgmlCpu);
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        var all = new float[Prompt.Length * feat];
        spec.SpecForward(Prompt, all, new float[vocab], false);
        model.ResetKVCache();
        var last = new float[feat];
        spec.SpecForward(Prompt, last, new float[vocab], false);
        Assert.True(last.AsSpan().SequenceEqual(all.AsSpan((Prompt.Length - 1) * feat, feat)),
            "the one-row capture is not the full capture's last row");
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void VerifyWindowsOfChangingWidth_CommitAsPlainDecode_Cpu() => NativeChangingWidths(BackendType.GgmlCpu, 1e-5);

    /// <summary>Metal verifies a window as one batched forward, which does not round like its decode steps.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void VerifyWindowsOfChangingWidth_CommitAsPlainDecode_Metal() => NativeChangingWidths(BackendType.GgmlMetal, 2e-2);

    /// <summary>ggml_cuda verifies with the one-token kernels per row (exact).</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void VerifyWindowsOfChangingWidth_CommitAsPlainDecode_Cuda() => NativeChangingWidths(BackendType.GgmlCuda, 1e-5);

    /// <summary>Verify windows of changing width on the native token span, some partly accepted (committed from the
    /// window's own per-row states), against a plain decode of the same tokens. Each width keeps its own persisted graph,
    /// and a wider window grows the buffers the narrower graphs captured into: those graphs must rebuild, never replay
    /// against a freed buffer.</summary>
    private void NativeChangingWidths(BackendType backend, double bound)
    {
        string path = Fixture(indexerTopK: 2048);
        int[] seq = Enumerable.Range(0, 15).Select(i => (i * 29 + 5) % 250).ToArray();
        (int Width, int Keep)[] windows = { (2, 2), (4, 2), (2, 1), (4, 4), (3, 2), (4, 1), (2, 2) };
        using ModelBase model = ModelBase.Create(path, backend);
        int vocab = model.Config.VocabSize;
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        var plain = seq.Select(t => (float[])model.Forward(new[] { t }).Clone()).ToArray();

        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        int pos = 0;
        foreach (var (width, keep) in windows)
        {
            int[] window = seq[pos..(pos + width)];
            var all = new float[width * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(window, null, all, true);
            spec.SpecOnVerifyAccepted(keep - 1, width - 1);
            // A fully accepted window keeps the verify's state as it is (the executor does not rewind).
            if (keep < width)
            {
                Assert.True(spec.SpecVerifyPersistsAcceptedKv, $"window {width}/{keep} at {pos}: the verify did not commit its prefix");
                spec.SpecRewindCache(Prompt.Length + pos + keep);
            }
            for (int r = 0; r < keep; r++)
            {
                double err = RelativeError(all.AsSpan(r * vocab, vocab).ToArray(), plain[pos + r]);
                Assert.True(err <= bound, $"window {width}/{keep} at {pos}: row {r} differs from its decode by {err:E2}");
            }
            pos += keep;
        }
        double next = RelativeError(model.Forward(new[] { seq[pos] }), plain[pos]);
        _output.WriteLine($"after {windows.Length} windows: next decode relative error {next:E2}");
        Assert.True(next <= bound, $"the committed state differs from a plain decode by {next:E2}");
    }

    // ---------------------------------------------------------------- the direct-CUDA engine

    /// <summary>Dense attention (top_k 2048, far above the 46 cells this runs): the prompt's logits and
    /// every decode step's against the native token span. Both engines read the same quantized bytes;
    /// they differ in kernels and in how activations are quantized for them.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheDirectCudaEngine_AnswersAsTheNativeSpan()
    {
        string path = Fixture(indexerTopK: 2048);
        float[][] native, cuda;
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
            native = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            Assert.IsType<Qwen4ExpCudaModel>(model);
            cuda = Run(model);
        }
        double worst = 0;
        for (int i = 0; i < native.Length; i++)
        {
            double err = RelativeError(cuda[i], native[i]);
            worst = Math.Max(worst, err);
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: relative error {err:E3}, argmax {ArgMax(cuda[i])} vs {ArgMax(native[i])}");
        }
        Assert.True(worst <= 2e-2, $"the direct-CUDA engine strayed {worst:E3} from the native token span");
    }

    /// <summary>The UD-Q2_K_XL expert mix - IQ2_XS gate/up, one IQ3_XXS layer, IQ4_NL down - on the
    /// direct engine's per-token and grouped expert kernels, against ggml-cuda's own kernels for
    /// those formats in the native span. Before the engine learned the two formats it refused the
    /// shipped file at load ("unsupported expert quant type 17").</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheDirectCudaEngine_AnswersAsTheNativeSpan_Q2KXLExperts()
    {
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        float[][] native, cuda;
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
            native = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            Assert.IsType<Qwen4ExpCudaModel>(model);
            cuda = Run(model);
        }
        double worst = 0;
        for (int i = 0; i < native.Length; i++)
        {
            double err = RelativeError(cuda[i], native[i]);
            worst = Math.Max(worst, err);
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: relative error {err:E3}, argmax {ArgMax(cuda[i])} vs {ArgMax(native[i])}");
        }
        Assert.True(worst <= 2e-2, $"the direct-CUDA engine strayed {worst:E3} from the native token span");
    }

    /// <summary>The UD-Q4_K_XL expert mix - Q4_K gate/up over Q5_1 down, the layout 43 of the shipped
    /// file's 48 down projections take - on the direct engine's per-token kernels (the decode steps)
    /// and tensor-core kernels (the 40-token prompt), against ggml-cuda's own Q5_1 kernels in the
    /// native span. Before the engine learned Q5_1 it refused the file at load ("unsupported expert
    /// quant type 7"), and with <c>--layer-split 2</c> that refusal aborted the CLI.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheDirectCudaEngine_AnswersAsTheNativeSpan_Q4KXLExperts()
    {
        // The routed down projection at full gain (the fixture's default writes a quarter-size update):
        // at a quarter, even a decoder that drops every fifth bit stayed under the bound (1.8e-2,
        // against 6.6e-3 here); at full gain it strays 6.3e-2, while this decoder still answers within
        // the 6.6e-3 it does at a quarter.
        string path = Fixture(indexerTopK: 2048, q4kxlExperts: true, expertDownGain: 1f);
        float[][] native, cuda;
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
            native = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            Assert.IsType<Qwen4ExpCudaModel>(model);
            cuda = Run(model);
        }
        double worst = 0;
        for (int i = 0; i < native.Length; i++)
        {
            double err = RelativeError(cuda[i], native[i]);
            worst = Math.Max(worst, err);
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: relative error {err:E3}, argmax {ArgMax(cuda[i])} vs {ArgMax(native[i])}");
        }
        Assert.True(worst <= 2e-2, $"the direct-CUDA engine strayed {worst:E3} from the native token span");
    }

    /// <summary>
    /// Low-bit IQ routed experts - IQ2_XXS gate/up and down, as unsloth's UD-IQ2_XXS and UD-IQ1_S files keep
    /// many, and the DeepSeek V4.1 1017's IQ1_S gate/up over IQ2_S down - on the direct engine against
    /// ggml-cuda's own kernels for those formats in the native span, at an expert width of 256 so the down
    /// projection holds whole 256-value blocks. The 40-token prompt is past the per-slot bound, so it takes the
    /// grouped route <see cref="TensorSharp.Cuda.CudaMoe.SelectRoute"/> picks for the loaded module: with the
    /// low-bit IQ decoders, the tensor-core kernel (the register-staged ones on a module without its body) for
    /// IQ2_XXS too, which used to run on the warp-per-row kernels alone; the six decode steps run the per-slot
    /// ones.
    /// <para>
    /// The decode steps compared with ggml-cuda are the captured ones, as served. The routers are read back on a
    /// second run, whose decode steps the probe keeps uncaptured; that run must answer as the captured one to
    /// the bit, so what the probe reads is the computation under test.
    /// </para>
    /// <para>
    /// Every step is held to <see cref="LogitBound"/>, with one exception on a module without the int8 Q8_0
    /// tensor-core GEMM (<c>ts_quant_matmul_q8_0_mmq*</c>: the committed compute_75 PTX, or a GPU below sm_80).
    /// There the prompt's dense Q8_0 projections take the dequantize-to-F16 cuBLAS route. Against a float64
    /// oracle on the engine's own inputs that route is the more exact one: median relative error 2.9e-4
    /// (2.8e-4 to 3.0e-4) against 5.5e-3 (1.0e-3 to 8.6e-3) for the int8 GEMM's q8_1 activations, and lower on
    /// each of the prompt's 84 projections. But ggml-cuda rounds the activations to q8_1 as the int8 GEMM does,
    /// and the choice of route alone moves the direct engine's router logits by up to 4.0e-3, so a layer whose
    /// fourth and fifth experts stand closer than that can go either way. On the IQ1_S/IQ2_S fixture, decode
    /// step 2's layer 4 (margin 7.5e-4 on an RTX 3080 Laptop, 4.9e-4 with every kernel module from its committed
    /// PTX, 2.31e-4 on an A40 with the same PTX) picks expert 7 on the F16 route and expert 12 on the int8 route:
    /// the only one of 368 router rows that differs. That step strays 2.1e-2 from ggml-cuda on the F16 route
    /// (2.115e-2 and 2.140e-2 there, 2.143e-2 on an A40; 7.3e-3 on the int8 route), and every other step stays at
    /// 6e-3 to 8e-3 on both. The prompt itself never flipped.
    /// </para>
    /// <para>
    /// So on such a module one decode step may pass the bound as a routing flip, and only one that is measured:
    /// the direct engine is run once more with the prompt in chunks below
    /// <see cref="TensorSharp.Cuda.CudaQuantizedOps.F16GemmMinRows"/> rows, where the dense projections round
    /// their activations to q8_1 as ggml-cuda's do. That run must answer the step within
    /// <see cref="LogitBound"/>, and the lowest layer where its router chose differently must be a swap of the
    /// direct engine's fourth expert for its fifth at a margin of at most <see cref="RouterTieMargin"/>. The
    /// step must also keep its argmax and stay within <see cref="FlipLogitCap"/>. Every layer's chosen experts
    /// must be the top of its own logits on both runs.
    /// </para>
    /// </summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData((int)SyntheticGguf.GgmlType.IQ2_XXS, (int)SyntheticGguf.GgmlType.IQ2_XXS)]
    [InlineData((int)SyntheticGguf.GgmlType.IQ1_S, (int)SyntheticGguf.GgmlType.IQ2_S)]
    // ISTA-DASLab's Q2_0 pack: every routed expert in Q2_0, decoded on the warp-per-row routes.
    [InlineData((int)SyntheticGguf.GgmlType.Q2_0, (int)SyntheticGguf.GgmlType.Q2_0)]
    public void TheDirectCudaEngine_AnswersAsTheNativeSpan_LowBitIqExperts(int gateUpType, int downType)
    {
        var gateUp = (SyntheticGguf.GgmlType)gateUpType;
        var down = (SyntheticGguf.GgmlType)downType;
        // The routed down projection at full gain, as for the Q5_1 layout above.
        const int ff = 256;
        string path = Fixture(indexerTopK: 2048, expertGateUpType: gateUp, expertDownType: down, expertDownGain: 1f,
            expertFf: ff);
        // Decode steps captured into CUDA graphs, as served (the default), for the run held to ggml-cuda.
        _env.Set("TS_Q4E_GRAPHS", null);
        float[][] native, cuda, probed;
        var routers = new List<TensorSharp.Cuda.Q4eCudaEngine.RouterLayerProbe>();
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
            native = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            cuda = Run(model);
            Assert.True(engine.CapturedDecodeGraphs > 0, "the decode steps were not captured");
        }
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            // The probe reads the device mid-step, so this run's decode steps are not captured.
            Assert.IsType<Qwen4ExpCudaModel>(model).Engine.RouterProbe = routers.Add;
            probed = Run(model);
        }
        for (int i = 0; i < cuda.Length; i++)
            Assert.True(System.Runtime.InteropServices.MemoryMarshal.AsBytes(cuda[i].AsSpan())
                    .SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes(probed[i].AsSpan())),
                $"{(i == 0 ? "prefill" : $"decode {i}")}: the uncaptured run the routers were read from differs from the captured one");
        TensorSharp.Cuda.CudaMoe.ExpertRoute route;
        bool int8Gemm;
        using (var allocator = new TensorSharp.Cuda.CudaAllocator(0))
        {
            allocator.Context.MakeCurrent();
            int8Gemm = allocator.Kernels.HasQ80Mmq;
            using var dk = TensorSharp.Cuda.Dsv4Kernels.Create();
            Assert.True(dk.DecodesLowBitIqExperts, "the loaded kernel module predates the low-bit IQ decoders");
            route = TensorSharp.Cuda.CudaMoe.SelectRoute(Prompt.Length, TensorSharp.Cuda.Q4eCudaEngine.PerSlotMaxRows,
                dk.HasMoeMma, dk.DecodesLowBitIqExperts, gateUpType, downType,
                Qwen4ExpSyntheticModelBuilder.Hidden, ff, dk.DecodesLegacyQuants);
            Assert.Equal(dk.HasMoeMma ? TensorSharp.Cuda.CudaMoe.ExpertRoute.TensorCore : TensorSharp.Cuda.CudaMoe.ExpertRoute.Staged,
                route);
        }
        var failures = new List<string>();
        // Only on a module without the int8 GEMM: the direct engine with the prompt's dense projections on
        // q8_1 activations, as ggml-cuda's, for the routing flips the F16 route can leave.
        float[][] q81 = null;
        var q81Routers = new List<TensorSharp.Cuda.Q4eCudaEngine.RouterLayerProbe>();
        int q81Chunk = 0;
        if (!int8Gemm)
        {
            q81Chunk = Q81PromptChunk();
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            Assert.IsType<Qwen4ExpCudaModel>(model).Engine.RouterProbe = q81Routers.Add;
            q81 = Run(model, q81Chunk);
        }
        int flips = 0;
        for (int i = 0; i < native.Length; i++)
        {
            string step = i == 0 ? "prefill" : $"decode {i}";
            int p0 = i == 0 ? 0 : Prompt.Length + i - 1;
            double err = RelativeError(cuda[i], native[i]);
            int argmax = ArgMax(cuda[i]), nativeArgmax = ArgMax(native[i]);
            double margin = SmallestRouterMargin(routers, p0, step, failures);
            _output.WriteLine($"{gateUp} gate/up, {down} down, {(i == 0 ? $"prefill ({route})" : step)}: " +
                $"relative error {err:E3}, argmax {argmax} vs {nativeArgmax}, smallest router margin {margin:E2}");
            List<RouterSwap> swaps = null;
            double q81Err = double.NaN;
            if (q81 != null)
            {
                q81Err = RelativeError(q81[i], native[i]);
                string where = $"{step} with the prompt in {q81Chunk}-row chunks";
                if (i == 0)
                {
                    for (int c = 0; c < Prompt.Length; c += q81Chunk)
                        SmallestRouterMargin(q81Routers, c, where, failures);
                }
                else
                {
                    SmallestRouterMargin(q81Routers, p0, where, failures);
                    swaps = SelectionDifferences(routers, q81Routers, p0);
                }
                _output.WriteLine($"  prompt in {q81Chunk}-row chunks (q8_1 activations): relative error {q81Err:E3}" +
                    (swaps == null ? "" : swaps.Count == 0 ? ", same experts" : ", experts differ at " + string.Join(", ", swaps)));
            }
            if (err <= LogitBound)
                continue;
            // A measured routing flip: a decode step on a module without the int8 GEMM, whose q8_1 twin answers within
            // the bound and first chose differently where the direct engine's fourth and fifth experts nearly tie.
            if (swaps is { Count: > 0 } && swaps[0].FourthForFifth && swaps[0].Margin <= RouterTieMargin
                && q81Err <= LogitBound && argmax == nativeArgmax && err <= FlipLogitCap)
            {
                flips++;
                _output.WriteLine($"  {step}: a routing flip at a near tie of the direct engine's router ({swaps[0]})");
                continue;
            }
            failures.Add($"{step}: the direct-CUDA engine strayed {err:E3} from the native token span " +
                $"(argmax {argmax} vs {nativeArgmax}, smallest router margin {margin:E2}" +
                (q81 == null ? ")" : $"; with q8_1 prompt activations {q81Err:E3}, experts differ at " +
                    (swaps is { Count: > 0 } ? string.Join(", ", swaps) : "no layer") + ")"));
        }
        if (flips > 1)
            failures.Add($"{flips} steps past {LogitBound:E0}: more than the one routing flip a near tie explains");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>The logits' relative L2 bound for every step.</summary>
    private const double LogitBound = 2e-2;

    /// <summary>The largest fourth-minus-fifth router margin a measured flip may stand at: the flips measured stood at
    /// 4.9e-4 and 7.5e-4 on an RTX 3080 Laptop and 2.31e-4 on an A40, under the 4.0e-3 by which the dense prefill
    /// route moves the router logits.</summary>
    private const double RouterTieMargin = 2e-3;

    /// <summary>What one measured routing flip may leave in the step's logits: it moved them 2.115e-2 to 2.143e-2.</summary>
    private const double FlipLogitCap = 2.5e-2;

    /// <summary>The prompt's logits, then each decode step's, with the prompt run as forwards of at most
    /// <paramref name="promptChunk"/> tokens.</summary>
    private static float[][] Run(ModelBase model, int promptChunk)
    {
        var rows = new float[Steps.Length + 1][];
        for (int c = 0; c < Prompt.Length; c += promptChunk)
            rows[0] = (float[])model.Forward(Prompt[c..Math.Min(Prompt.Length, c + promptChunk)]).Clone();
        for (int i = 0; i < Steps.Length; i++)
            rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
        return rows;
    }

    /// <summary>Equal prompt chunks below <see cref="TensorSharp.Cuda.CudaQuantizedOps.F16GemmMinRows"/> rows: the dense
    /// Q8_0 projections then quantize their activations to q8_1 (dp4a), as ggml-cuda's do, instead of taking the
    /// dequantize-to-F16 cuBLAS route.</summary>
    private static int Q81PromptChunk()
    {
        int below = TensorSharp.Cuda.CudaQuantizedOps.F16GemmMinRows - 1;
        Assert.True(below >= 2, $"TS_CUDA_QMM_F16GEMM_MIN_ROWS={below + 1} leaves no multi-row q8_1 prompt chunk");
        int chunks = (Prompt.Length + below - 1) / below;
        return (Prompt.Length + chunks - 1) / chunks;
    }

    /// <summary>A row where two runs' routers chose different experts: this run's choice and the other's, this run's
    /// fourth-minus-fifth margin, and whether the other took exactly this run's fifth expert instead of its fourth.</summary>
    private readonly record struct RouterSwap(int Layer, int Row, int[] Chosen, int[] Other, double Margin, bool FourthForFifth)
    {
        public override string ToString() =>
            $"layer {Layer} row {Row} [{string.Join(",", Chosen.Order())}] vs [{string.Join(",", Other.Order())}] " +
            $"margin {Margin:E2}{(FourthForFifth ? "" : ", not a fourth-for-fifth swap")}";
    }

    /// <summary>Every layer and row of the forward at <paramref name="p0"/> where <paramref name="other"/> chose other
    /// experts than <paramref name="run"/>, lowest layer first.</summary>
    private static List<RouterSwap> SelectionDifferences(List<TensorSharp.Cuda.Q4eCudaEngine.RouterLayerProbe> run,
        List<TensorSharp.Cuda.Q4eCudaEngine.RouterLayerProbe> other, int p0)
    {
        var theirs = other.Where(p => p.P0 == p0).ToDictionary(p => p.Layer);
        var swaps = new List<RouterSwap>();
        foreach (var p in run.Where(p => p.P0 == p0).OrderBy(p => p.Layer))
        {
            Assert.True(theirs.TryGetValue(p.Layer, out var q) && q.Nt == p.Nt, $"layer {p.Layer} at {p0}: no router to compare");
            for (int t = 0; t < p.Nt; t++)
            {
                int[] chosen = p.Selected.AsSpan(t * p.Used, p.Used).ToArray();
                int[] otherChosen = q.Selected.AsSpan(t * q.Used, q.Used).ToArray();
                if (chosen.ToHashSet().SetEquals(otherChosen))
                    continue;
                float[] logits = p.Logits.AsSpan(t * p.Experts, p.Experts).ToArray();
                int[] ranked = Enumerable.Range(0, p.Experts).OrderByDescending(e => logits[e]).ThenBy(e => e).ToArray();
                double margin = (double)logits[ranked[p.Used - 1]] - logits[ranked[p.Used]];
                bool fourthForFifth = otherChosen.ToHashSet().SetEquals(ranked.Take(p.Used - 1).Append(ranked[p.Used]));
                swaps.Add(new RouterSwap(p.Layer, t, chosen, otherChosen, margin, fourthForFifth));
            }
        }
        return swaps;
    }

    /// <summary>The smallest fourth-to-fifth margin of the direct engine's router logits over every layer and row of
    /// the forward at <paramref name="p0"/>. Each layer's chosen experts must be the top of its own logits (the
    /// kernel ranks in float, this in double: only a tie within float rounding may differ).</summary>
    private static double SmallestRouterMargin(List<TensorSharp.Cuda.Q4eCudaEngine.RouterLayerProbe> probes, int p0,
        string step, List<string> failures)
    {
        var forward = probes.Where(p => p.P0 == p0).ToList();
        Assert.True(forward.Count == Qwen4ExpSyntheticModelBuilder.Layers, $"{step}: {forward.Count} router probes");
        double smallest = double.MaxValue;
        foreach (var p in forward)
        {
            for (int t = 0; t < p.Nt; t++)
            {
                float[] logits = p.Logits.AsSpan(t * p.Experts, p.Experts).ToArray();
                int[] ranked = Enumerable.Range(0, p.Experts).OrderByDescending(e => logits[e]).ThenBy(e => e).ToArray();
                double margin = (double)logits[ranked[p.Used - 1]] - logits[ranked[p.Used]];
                smallest = Math.Min(smallest, margin);
                int[] chosen = p.Selected.AsSpan(t * p.Used, p.Used).ToArray();
                if (!ranked.Take(p.Used).ToHashSet().SetEquals(chosen) && margin > 1e-5)
                    failures.Add($"{step}, layer {p.Layer} row {t}: the router chose [{string.Join(",", chosen)}], " +
                        $"not the top {p.Used} of its own logits [{string.Join(",", ranked.Take(p.Used))}]");
            }
        }
        return smallest;
    }

    /// <summary>An expert layout no direct-CUDA kernel decodes (Q4_0 down) is refused at load as one
    /// <see cref="NotSupportedException"/> naming the tensor, which the hosts report as a load refusal,
    /// and not from inside the per-device upload.</summary>
    /// <summary>
    /// Expert residency (Q4eCudaEngine.Residency.cs): with every routed expert in pinned host memory, and
    /// with only some of them in VRAM slots, each expert's rows are read through a per-expert table the
    /// GPU follows into VRAM or over the bus. The kernels' arithmetic does not depend on where the bytes
    /// live, so prefill (grouped kernels) and every decode step (per-slot kernels, captured graphs) answer
    /// as the all-resident engine does, bit for bit.
    /// </summary>
    [CudaFact]
    public void ExpertResidency_HostAndPartialSlots_AnswerAsAllResidentBitForBit()
    {
        // Staging only: every expert's product comes from the same GPU kernels wherever its bytes live.
        _env.Set("TS_Q4E_HOST_EXPERTS", "0");
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);

        float[][] Load(string vramGb, out int resident)
        {
            _env.Set("TS_Q4E_EXPERT_VRAM_GB", vramGb);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            resident = Assert.IsType<Qwen4ExpCudaModel>(model).Engine.ResidentExperts;
            return Run(model);
        }

        float[][] all = Load(null, out int allResident);
        float[][] host = Load("0", out int hostResident);
        float[][] partial = Load((expertBytes / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            out int partialResident);
        _output.WriteLine($"expert bytes {expertBytes}; slots: all-resident {allResident}, host {hostResident}, partial {partialResident}");
        Assert.Equal(-1, allResident);
        Assert.Equal(0, hostResident);
        Assert.InRange(partialResident, 1, Qwen4ExpSyntheticModelBuilder.Layers * Qwen4ExpSyntheticModelBuilder.Experts - 1);
        for (int i = 0; i < all.Length; i++)
        {
            Assert.Equal(all[i], host[i]);
            Assert.Equal(all[i], partial[i]);
        }
    }

    /// <summary>
    /// A prompt read layer by layer (TS_Q4E_PROMPT_GROUP) runs each layer over all the group's ubatches before the
    /// next layer, its missed experts staged once per layer: every ubatch computes what it computes in ubatch order,
    /// so the prompt's logits and the decode steps after it answer bit for bit as ubatch order does. Twelve-row
    /// ubatches of the 40-token prompt and groups of 24 rows: the groups end inside the prompt and on its last,
    /// four-row ubatch (the decode kernels' size).
    /// </summary>
    [CudaFact]
    public void APromptReadLayerByLayer_AnswersBitForBitAsUbatchOrder()
    {
        // Staging only: a four-row ubatch would otherwise leave its missed experts to the CPU in ubatch order.
        _env.Set("TS_Q4E_HOST_EXPERTS", "0");
        _env.Set("TS_Q4E_UBATCH", "12");
        string path = Fixture(indexerTopK: 16, q2kxlExperts: true);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);
        // Two thirds resident: a layer's host experts then fit half the staging, so the next layer can be staged ahead.
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", (expertBytes * 2 / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture));

        (float[][] Rows, long Groups, long Prefetched) Answer(string group)
        {
            _env.Set("TS_Q4E_PROMPT_GROUP", group);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            var rows = Run(model);
            return (rows, engine.PromptGroupsRun, engine.PromptPrefetchGroups);
        }
        var ubatchOrder = Answer("0");
        _env.Set("TS_Q4E_PROMPT_PREFETCH", "0");
        var grouped = Answer("24");
        _env.Set("TS_Q4E_PROMPT_PREFETCH", null);
        var prefetched = Answer("24");
        _output.WriteLine($"prompt groups read layer by layer: ubatch order {ubatchOrder.Groups}, grouped {grouped.Groups}, "
            + $"grouped with the next layer staged ahead {prefetched.Groups}");
        Assert.Equal(0, ubatchOrder.Groups);
        Assert.Equal(2, grouped.Groups);
        Assert.Equal(2, prefetched.Groups);
        Assert.Equal(0, grouped.Prefetched);
        Assert.Equal(2, prefetched.Prefetched);
        for (int i = 0; i < grouped.Rows.Length; i++)
        {
            Assert.True(grouped.Rows[i].AsSpan().SequenceEqual(ubatchOrder.Rows[i]),
                $"{(i == 0 ? "prefill" : $"decode {i}")}: reading the prompt layer by layer changed the logits");
            Assert.True(prefetched.Rows[i].AsSpan().SequenceEqual(ubatchOrder.Rows[i]),
                $"{(i == 0 ? "prefill" : $"decode {i}")}: staging the next layer ahead changed the logits");
        }
    }

    /// <summary>
    /// A prompt group's ubatches share their layer's dense weights as F16 (TS_Q4E_GROUP_F16): the first ubatch of a layer
    /// dequantizes each quantized projection into the group's arena and the others reuse that copy, which is the same
    /// kernel's output, so the prompt and the decode steps after it answer bit for bit as dequantizing per ubatch does.
    /// 520-row ubatches (past the Q8_0 MMQ's 512 rows, so the fixture's Q8_0 projections take the F16 route) of an
    /// 1,100-token prompt in groups of 1,040 rows: two ubatches a layer, then a group of one.
    /// </summary>
    [CudaFact]
    public void APromptGroupsDenseWeights_DequantizedOncePerLayer_AnswerBitForBit()
    {
        _env.Set("TS_Q4E_HOST_EXPERTS", "0");
        _env.Set("MAX_CONTEXT", "2048");    // the class keeps 1024; the prompt is 1,100 tokens
        _env.Set("TS_Q4E_UBATCH", "520");
        _env.Set("TS_Q4E_PROMPT_GROUP", "1040");
        string path = Fixture(indexerTopK: 16, q2kxlExperts: true);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);
        // Prompt groups need expert residency: two thirds resident.
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", (expertBytes * 2 / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        int[] prompt = Enumerable.Range(0, 1100).Select(i => (i * 37 + 11) % 250).ToArray();

        (float[][] Rows, long Groups, long Dequantized, long Reused) Answer(string f16)
        {
            _env.Set("TS_Q4E_GROUP_F16", f16);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            var rows = new float[Steps.Length + 1][];
            rows[0] = (float[])model.ForwardRefill(prompt).Clone();
            for (int i = 0; i < Steps.Length; i++)
                rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
            return (rows, engine.PromptGroupsRun, engine.GroupF16Dequantized, engine.GroupF16Reused);
        }
        var perUbatch = Answer("0");
        var shared = Answer(null);
        _output.WriteLine($"prompt groups {shared.Groups}: F16 copies dequantized {shared.Dequantized}, reused {shared.Reused} " +
            $"(per ubatch: {perUbatch.Dequantized} / {perUbatch.Reused})");
        Assert.Equal(2, perUbatch.Groups);
        Assert.Equal(2, shared.Groups);
        Assert.Equal(0, perUbatch.Dequantized + perUbatch.Reused);
        Assert.True(shared.Dequantized > 0, "no layer dequantized a weight into the group's arena");
        Assert.Equal(shared.Dequantized, shared.Reused);    // two ubatches a layer in the first group, the second alone
        for (int i = 0; i < shared.Rows.Length; i++)
            Assert.True(shared.Rows[i].AsSpan().SequenceEqual(perUbatch.Rows[i]),
                $"{(i == 0 ? "prefill" : $"decode {i}")}: sharing the layer's F16 weights changed the logits");
    }

    /// <summary>
    /// A prompt group runs its routed experts' gate/up once over all its ubatches (TS_Q4E_GROUP_EXPERTS): each ubatch
    /// stages its missed experts and leaves its selections, int8 activations, shared-expert outputs and scatter logits to
    /// the group; after the last, gate/up runs over the group and each ubatch's down reads its slots' H through the group's
    /// plan. Every slot's products are ubatch order's, so the prompt and the decode steps after it answer bit for bit as
    /// the per-ubatch experts do. Q2_0 experts on the int8 route, 64-row ubatches of a 300-token prompt in groups of 192
    /// rows: three ubatches, then 64 and 44.
    /// </summary>
    [CudaFact]
    public void APromptGroupsExpertGateUp_RunOncePerLayer_AnswersBitForBit()
    {
        _env.Set("TS_Q4E_HOST_EXPERTS", "0");
        _env.Set("TS_Q4E_UBATCH", "64");
        _env.Set("TS_Q4E_PROMPT_GROUP", "192");
        string path = Fixture(indexerTopK: 16, expertGateUpType: SyntheticGguf.GgmlType.Q2_0, expertDownType: SyntheticGguf.GgmlType.Q2_0);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);
        // Prompt groups need expert residency: two thirds resident.
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", (expertBytes * 2 / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        int[] prompt = Enumerable.Range(0, 300).Select(i => (i * 37 + 11) % 250).ToArray();

        (float[][] Rows, long Groups, long Deferred) Answer(string grouped)
        {
            _env.Set("TS_Q4E_GROUP_EXPERTS", grouped);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            var rows = new float[Steps.Length + 1][];
            rows[0] = (float[])model.ForwardRefill(prompt).Clone();
            for (int i = 0; i < Steps.Length; i++)
                rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
            return (rows, engine.PromptGroupsRun, engine.GroupExpertsRun);
        }
        var perUbatch = Answer("0");
        var grouped = Answer(null);
        _output.WriteLine($"prompt groups {grouped.Groups}: layers whose gate/up ran over the group {grouped.Deferred} " +
            $"(per ubatch: {perUbatch.Deferred})");
        Assert.Equal(2, perUbatch.Groups);
        Assert.Equal(2, grouped.Groups);
        Assert.Equal(0, perUbatch.Deferred);
        Assert.Equal(2L * Qwen4ExpSyntheticModelBuilder.Layers, grouped.Deferred);
        for (int i = 0; i < grouped.Rows.Length; i++)
            Assert.True(grouped.Rows[i].AsSpan().SequenceEqual(perUbatch.Rows[i]),
                $"{(i == 0 ? "prefill" : $"decode {i}")}: running the group's gate/up at once changed the logits");
    }

    /// <summary>
    /// The route-tail skip (TS_MOE_ROUTE_TAIL_SKIP; Strata's STRATA_ROUTE_TAIL_SKIP): in decode steps and verify windows a
    /// host expert every row routes at the given rank or lower is left out. The fixture routes 4 of 16 experts, so rank 2
    /// leaves out misses at ranks 2-3: some slots are skipped and the answers move; a rank as wide as the selection skips
    /// nothing and answers bit for bit as off; the prompt (staged) is untouched.
    /// </summary>
    [CudaFact]
    public void TheRouteTailSkip_LeavesOutOnlyLowRankedMisses()
    {
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);
        // Two thirds resident: a layer's host experts then fit half the staging, so the next layer can be staged ahead.
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", (expertBytes / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        (float[][] Rows, long Skipped, long Misses) Answer(string rank)
        {
            _env.Set("TS_MOE_ROUTE_TAIL_SKIP", rank);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            float[][] rows = Run(model);
            return (rows, engine.TailSkipped, engine.HostExpertMisses);
        }
        var off = Answer("0");
        var wide = Answer(Qwen4ExpSyntheticModelBuilder.ExpertsUsed.ToString(System.Globalization.CultureInfo.InvariantCulture));
        var tail = Answer("2");
        _output.WriteLine($"host experts computed: off {off.Misses}, rank 4 {wide.Misses}, rank 2 {tail.Misses} ({tail.Skipped} skipped)");
        Assert.Equal(0, off.Skipped);
        Assert.Equal(0, wide.Skipped);
        // A layer whose misses are all in the tail rings no doorbell, so the host's count of skipped slots covers only the
        // layers that rang; the computed misses fall either way.
        Assert.True(tail.Misses < off.Misses, "the tail skip did not lower the host's share");
        Assert.True(off.Rows[0].AsSpan().SequenceEqual(tail.Rows[0]), "the tail skip changed the prompt");
        bool moved = false;
        for (int i = 0; i < off.Rows.Length; i++)
        {
            Assert.True(off.Rows[i].AsSpan().SequenceEqual(wide.Rows[i]), $"{(i == 0 ? "prefill" : $"decode {i}")}: a rank as wide as the selection changed the logits");
            Assert.All(tail.Rows[i], v => Assert.True(float.IsFinite(v)));
            moved |= !off.Rows[i].AsSpan().SequenceEqual(tail.Rows[i]);
        }
        Assert.True(moved, "skipping routes left every decode step as it was");
    }

    /// <summary>Host experts (Strata's doorbell): decode steps leave the experts without a VRAM slot to the
    /// CPU inside the captured step, prompts stage them. The CPU's dot products quantize activations as
    /// ggml's CPU kernels do, so the answer matches the all-resident engine closely rather than bit for bit.</summary>
    [CudaFact]
    public void ExpertResidency_HostComputedMisses_AnswerAsAllResident()
    {
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);

        float[][] Load(string vramGb, out long misses)
        {
            _env.Set("TS_Q4E_EXPERT_VRAM_GB", vramGb);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            float[][] r = Run(model);
            misses = engine.HostExpertMisses;
            return r;
        }

        float[][] all = Load(null, out long allMisses);
        float[][] host = Load("0", out long hostMisses);
        string third = (expertBytes / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        float[][] partial = Load(third, out long partialMisses);
        _output.WriteLine($"host-computed experts: all-resident {allMisses}, none resident {hostMisses}, a third resident {partialMisses}");
        Assert.Equal(-1, allMisses);
        Assert.True(hostMisses > 0 && partialMisses > 0 && partialMisses < hostMisses);
        for (int i = 0; i < all.Length; i++)
        {
            double eHost = RelativeError(host[i], all[i]), ePartial = RelativeError(partial[i], all[i]);
            _output.WriteLine($"forward {i}: relative error none resident {eHost:E2}, a third resident {ePartial:E2}");
            Assert.True(eHost <= 2e-2, $"forward {i}: host-computed experts differ by {eHost:E2}");
            Assert.True(ePartial <= 2e-2, $"forward {i}: partly host-computed experts differ by {ePartial:E2}");
        }
    }

    /// <summary>A layer whose selected experts all have VRAM slots rings no doorbell (the GPU skips the host's echo):
    /// with a third of the experts resident some layers ring and some do not, and the answers are bit for bit those of
    /// ringing every layer, decode steps and a verify window alike, with the same experts left to the CPU.</summary>
    [CudaFact]
    public void ExpertResidency_DoorbellSkip_AnswersBitForBitAsRingingEveryLayer()
    {
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", (expertBytes / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        int vocab = Qwen4ExpSyntheticModelBuilder.Vocab;
        (float[][] Rows, float[] Window, long Misses) Answer()
        {
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            float[][] rows = Run(model);
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            var window = new float[4 * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(new[] { 5, 77, 123, 9 }, null, window, true);
            return (rows, window, Assert.IsType<Qwen4ExpCudaModel>(model).Engine.HostExpertMisses);
        }
        var skip = Answer();
        _env.Set("TS_Q4E_DOORBELL_SKIP", "0");
        var ring = Answer();
        _output.WriteLine($"experts on the CPU: skipping {skip.Misses}, ringing every layer {ring.Misses}");
        Assert.True(skip.Misses > 0, "no expert was left to the CPU: the fixture did not exercise the doorbell");
        Assert.Equal(ring.Misses, skip.Misses);
        for (int i = 0; i < skip.Rows.Length; i++)
            Assert.True(skip.Rows[i].AsSpan().SequenceEqual(ring.Rows[i]), $"{(i == 0 ? "prefill" : $"decode {i}")}: skipping changed the logits");
        Assert.True(skip.Window.AsSpan().SequenceEqual(ring.Window), "skipping changed the verify window's logits");
    }

    /// <summary>The GPU's share of a decode step's missed experts (TS_Q4E_HYBRID_MISSES): the GPU stages up to that many
    /// over the bus and computes them itself while the CPU computes the rest. The fixture routes 4 of 16 experts, none
    /// resident here. A share of four in slot order takes all of a decode step's experts: none is left to the CPU and the
    /// steps answer bit for bit as the all-resident engine. The odd-numbered split and a share of two in slot order (the
    /// default) leave fewer experts to the CPU than no share does, within the host-compute tolerance.</summary>
    [CudaFact]
    public void ExpertResidency_GpuShareOfMisses_AnswersAsAllResident()
    {
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        float[][] Load(string vramGb, string share, string split, out long misses)
        {
            _env.Set("TS_Q4E_EXPERT_VRAM_GB", vramGb);
            _env.Set("TS_Q4E_HYBRID_MISSES", share);
            _env.Set("TS_Q4E_HYBRID_SPLIT", split);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            float[][] r = Run(model);
            misses = engine.HostExpertMisses;
            return r;
        }
        float[][] all = Load(null, "0", null, out _);
        float[][] cpu = Load("0", "0", null, out long cpuMisses);
        float[][] whole = Load("0", "4", "first", out long wholeMisses);
        float[][] odd = Load("0", "4", "odd", out long oddMisses);
        float[][] two = Load("0", "2", "first", out long twoMisses);
        _output.WriteLine($"experts on the CPU: no share {cpuMisses}, four in slot order {wholeMisses}, odd-numbered {oddMisses}, "
            + $"two in slot order {twoMisses}");
        Assert.True(cpuMisses > 0, "no expert was left to the CPU: the fixture did not exercise the doorbell");
        Assert.Equal(0, wholeMisses);
        Assert.True(oddMisses > 0 && oddMisses < cpuMisses, "the odd-numbered split left the CPU's share as it was");
        Assert.True(twoMisses > 0 && twoMisses < cpuMisses, "a share of two left the CPU's share as it was");
        for (int i = 0; i < all.Length; i++)
        {
            string at = i == 0 ? "prefill" : $"decode {i}";
            Assert.True(whole[i].AsSpan().SequenceEqual(all[i]), $"{at}: the GPU computing every miss differs from all-resident");
            double eCpu = RelativeError(cpu[i], all[i]), eOdd = RelativeError(odd[i], all[i]), eTwo = RelativeError(two[i], all[i]);
            _output.WriteLine($"{at}: relative error to all-resident: no share {eCpu:E2}, odd-numbered {eOdd:E2}, two {eTwo:E2}");
            Assert.True(eOdd <= 2e-2, $"{at}: the odd-numbered split differs by {eOdd:E2}");
            Assert.True(eTwo <= 2e-2, $"{at}: a share of two differs by {eTwo:E2}");
        }
    }

    /// <summary>The odd-numbered split gives an expert to the same side whatever else a call selects (while the share
    /// holds them all: here eight, every odd-numbered expert of the fixture's 16), so a verify window's rows compute their
    /// own decode steps as they do with no share. Slot order depends on the window; its row errors are only reported.</summary>
    [CudaFact]
    public void ExpertResidency_GpuShareOfMisses_VerifiesAsItsOwnDecode()
    {
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", "0");
        int[] window = { Steps[0], Steps[1], Steps[2], Steps[3] };
        int vocab = Qwen4ExpSyntheticModelBuilder.Vocab;
        double[] RowErrors(string share, string split)
        {
            _env.Set("TS_Q4E_HYBRID_MISSES", share);
            _env.Set("TS_Q4E_HYBRID_SPLIT", split);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            model.ForwardRefill(Prompt);
            var plain = new float[window.Length][];
            for (int i = 0; i < window.Length; i++)
                plain[i] = (float[])model.Forward(new[] { window[i] }).Clone();
            model.ResetKVCache();
            model.ForwardRefill(Prompt);
            var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
            var all = new float[window.Length * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(window, null, all, true);
            return Enumerable.Range(0, window.Length).Select(i => RelativeError(all.AsSpan(i * vocab, vocab).ToArray(), plain[i])).ToArray();
        }
        double[] cpu = RowErrors("0", null), odd = RowErrors("8", "odd"), first = RowErrors("2", "first");
        for (int i = 0; i < window.Length; i++)
        {
            _output.WriteLine($"verify row {i} vs its decode step: no share {cpu[i]:E2}, odd-numbered {odd[i]:E2}, two in slot order {first[i]:E2}");
            Assert.True(cpu[i] <= 1e-5, $"verify row {i} differs from its decode step by {cpu[i]:E2} with no share");
            Assert.True(odd[i] <= 1e-5, $"verify row {i} differs from its decode step by {odd[i]:E2} under the odd-numbered split");
        }
    }

    /// <summary>The resident KV window (TS_Q4E_KV_RESIDENT): a QSA layer's Q8_0 K/V lives in pinned host memory with a window
    /// of pages in VRAM, its raw indexer keys in a ring. Asked for 1 cell, the window takes one call's floor (well under the
    /// fixture's 1,024). A 600-token prompt read in 128-row ubatches (each layer staged from the host copy), 150 decode steps
    /// whose selections move over the whole context (pages fetched over the bus, the window evicting), and a verify window
    /// answer bit for bit as the resident caches.</summary>
    [CudaFact]
    public void AResidentKvWindow_AnswersBitForBitAsResidentCaches()
    {
        string path = Fixture(indexerTopK: 16);
        int[] prompt = Enumerable.Range(0, 600).Select(i => (i * 37 + 11) % 250).ToArray();
        int[] steps = Enumerable.Range(0, 150).Select(i => (i * 53 + 7) % 250).ToArray();
        int[] window = { 5, 77, 123, 9 };
        int vocab = Qwen4ExpSyntheticModelBuilder.Vocab;
        KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0);
        _env.Set("TS_Q4E_UBATCH", "128");
        (List<float[]> Rows, float[] Window, (long Misses, long Calls, long Overflows) Counters, int Cells, long Staged) Answer(string resident)
        {
            _env.Set("TS_Q4E_KV_RESIDENT", resident);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            var rows = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
            foreach (int token in steps)
                rows.Add((float[])model.Forward(new[] { token }).Clone());
            var counters = engine.KvStreamCounters();
            var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
            var all = new float[window.Length * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(window, null, all, true);
            return (rows, all, counters, engine.KvWindowCells, engine.KvStagedBytes);
        }
        var resident = Answer("0");
        var streamed = Answer("1");
        _output.WriteLine($"window {streamed.Cells} cells; pages fetched {streamed.Counters.Misses} in {streamed.Counters.Calls} resolves, "
            + $"{streamed.Counters.Overflows} overflowing; {streamed.Staged / 1048576.0:F1} MiB staged for the prompt");
        Assert.Equal(0, resident.Cells);
        Assert.True(streamed.Cells > 0 && streamed.Cells < 600, $"the window ({streamed.Cells} cells) does not stream the prompt");
        Assert.True(streamed.Counters.Misses > 0, "no page was fetched over the bus");
        Assert.Equal(0, streamed.Counters.Overflows);
        Assert.True(streamed.Staged > 0, "no prompt ubatch was staged from the host copy");
        for (int i = 0; i < resident.Rows.Count; i++)
            Assert.True(streamed.Rows[i].AsSpan().SequenceEqual(resident.Rows[i]),
                $"{(i == 0 ? "prefill" : $"decode {i}")}: the streamed caches changed the logits");
        Assert.True(streamed.Window.AsSpan().SequenceEqual(resident.Window), "the streamed caches changed the verify window's logits");
    }

    /// <summary>Adaptive residency (TS_MOE_TIER_ADAPT=1 under host experts): between decode steps the most-selected
    /// host experts trade places with the least-selected VRAM experts. The swaps keep every expert reachable, so a
    /// long decode keeps answering as the all-resident engine does within the host-compute tolerance.</summary>
    /// <remarks>Both the blocking swaps and the overlapped rounds (TS_MOE_TIER_ADAPT_ASYNC, the default: copies on a stream of
    /// their own, the tables switched a forward boundary later), the overlapped ones every 2 steps so rounds follow each
    /// other while decode steps run beside their copies.</remarks>
    [CudaTheory]
    [InlineData("0")]
    [InlineData("1")]
    public void ExpertResidency_AdaptiveSwaps_KeepAnswering(string overlapped)
    {
        _env.Set("TS_MOE_TIER_ADAPT_ASYNC", overlapped);
        if (overlapped == "1")
            _env.Set("TS_MOE_TIER_ADAPT_EVERY", "2");
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);
        int[] steps = Enumerable.Range(0, 40).Select(i => (i * 53 + 7) % 250).ToArray();
        float[][] Decode(ModelBase model)
        {
            var rows = new float[steps.Length + 1][];
            rows[0] = (float[])model.ForwardRefill(Prompt).Clone();
            for (int i = 0; i < steps.Length; i++)
                rows[i + 1] = (float[])model.Forward(new[] { steps[i] }).Clone();
            return rows;
        }

        float[][] all;
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
            all = Decode(model);
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", (expertBytes / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        _env.Set("TS_MOE_TIER_ADAPT", "1");
        float[][] adaptive;
        long swaps;
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            adaptive = Decode(model);
            swaps = Assert.IsType<Qwen4ExpCudaModel>(model).Engine.AdaptiveSwaps;
        }
        _output.WriteLine($"adaptive swaps over {steps.Length} decode steps: {swaps}");
        Assert.True(swaps > 0, "no expert changed places");
        // A swapped expert moves between the host's arithmetic and the GPU's, so a near-tie route can flip: one flip's
        // measured mark on a step's logits (FlipLogitCap). 36 swaps over these 40 steps flipped one at forward 22
        // (2.47e-2) once the F16 attention kernels were rebuilt; within a flip's mark either way.
        for (int i = 0; i < all.Length; i++)
        {
            double err = RelativeError(adaptive[i], all[i]);
            Assert.True(err <= FlipLogitCap, $"forward {i}: adaptive residency differs by {err:E2}");
        }
    }

    /// <summary>Speculative verify on the direct engine: a window's accepted prefix is committed by replaying the
    /// recurrences over it, which must leave the state a plain decode of those tokens leaves - the next token's
    /// logits included - and every kept row's verify logits must be that decode's. A restore returns to the window's
    /// start. Rows past the accepted prefix are arbitrary drafts.</summary>
    /// <remarks>With top_k 16 the QSA selection runs from the prompt on, so a captured window pools the blocks its
    /// rows complete one row at a time.</remarks>
    [CudaTheory]
    [InlineData(2048)]
    [InlineData(16)]
    public void AVerifyWindow_CommitsItsAcceptedPrefix_AsAPlainDecodeOfThoseTokens(int indexerTopK)
    {
        string path = Fixture(indexerTopK: indexerTopK);
        int[] window = { Steps[0], 77, 123, 9 };
        int next = Steps[1];
        int vocab = Qwen4ExpSyntheticModelBuilder.Vocab;
        using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);

        foreach (int keep in new[] { 1, 2, 4 })
        {
            model.ResetKVCache();
            model.ForwardRefill(Prompt);
            var plain = new List<float[]>();
            for (int i = 0; i < keep; i++)
                plain.Add((float[])model.Forward(new[] { window[i] }).Clone());
            float[] plainNext = (float[])model.Forward(new[] { next }).Clone();

            model.ResetKVCache();
            model.ForwardRefill(Prompt);
            var all = new float[window.Length * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(window, null, all, true);
            spec.SpecOnVerifyAccepted(keep - 1, window.Length - 1);
            Assert.True(spec.SpecVerifyPersistsAcceptedKv, "the verify did not commit its prefix");
            spec.SpecRewindCache(Prompt.Length + keep);
            Assert.Equal(Prompt.Length + keep, model.CacheSeqLen);
            for (int i = 0; i < keep; i++)
            {
                double rowErr = RelativeError(all.AsSpan(i * vocab, vocab).ToArray(), plain[i]);
                _output.WriteLine($"keep {keep}: verify row {i} relative error {rowErr:E2}");
                Assert.True(rowErr <= 1e-5, $"keep {keep}: verify row {i} differs from its decode by {rowErr:E2}");
            }
            float[] specNext = model.Forward(new[] { next });
            double err = RelativeError(specNext, plainNext);
            _output.WriteLine($"keep {keep}: next token after the commit, relative error {err:E2}");
            Assert.True(err <= 1e-5, $"keep {keep}: the committed state differs from a plain decode by {err:E2}");
        }

        // Every draft rejected through the restore path: the head returns to the window's start.
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        float[] plainFirst = (float[])model.Forward(new[] { window[0] }).Clone();
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        spec.SpecSnapshotRecurrentState();
        spec.SpecForward(window, null, new float[window.Length * vocab], true);
        spec.SpecRestoreRecurrentState();
        spec.SpecRewindCache(Prompt.Length);
        double restoreErr = RelativeError(model.Forward(new[] { window[0] }), plainFirst);
        Assert.True(restoreErr <= 1e-5, $"after a restore the window's first token differs by {restoreErr:E2}");
    }

    /// <summary>
    /// Split attention (flash decoding, TS_Q4E_ATTN_SPLITS): a decode-size row of more than 256 cells spreads them over
    /// blocks whose partials the last block merges. Past a 600-token prompt (dense rows of 601 cells and more) the
    /// decode steps answer as the one-block kernel does within float reassociation, the prompt itself (prefill-size
    /// rows) bit for bit, and a verify window's rows compute their own decode steps (as AVerifyWindow_... holds them):
    /// the split depends on a row's cells alone.
    /// </summary>
    [CudaFact]
    public void SplitAttention_AnswersAsOneBlockPerRow_AndVerifiesAsItsOwnDecode()
    {
        string path = Fixture(indexerTopK: 2048);
        int[] prompt = Enumerable.Range(0, 600).Select(i => (i * 37 + 11) % 250).ToArray();
        int[] window = { Steps[0], Steps[1], Steps[2], Steps[3] };
        int vocab = Qwen4ExpSyntheticModelBuilder.Vocab;
        (float[][] Rows, float[] Window) Answer(string splits)
        {
            _env.Set("TS_Q4E_ATTN_SPLITS", splits);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var rows = new float[window.Length + 1][];
            rows[0] = (float[])model.ForwardRefill(prompt).Clone();
            for (int i = 0; i < window.Length; i++)
                rows[i + 1] = (float[])model.Forward(new[] { window[i] }).Clone();
            model.ResetKVCache();
            model.ForwardRefill(prompt);
            var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
            var all = new float[window.Length * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(window, null, all, true);
            return (rows, all);
        }
        var split = Answer("16");
        var whole = Answer("1");
        Assert.True(split.Rows[0].AsSpan().SequenceEqual(whole.Rows[0]), "the prompt's logits moved");
        for (int i = 1; i < split.Rows.Length; i++)
        {
            double err = RelativeError(split.Rows[i], whole.Rows[i]);
            _output.WriteLine($"decode {i}: split vs one block per row, relative error {err:E2}");
            Assert.True(err <= 1e-4, $"decode {i}: split attention differs by {err:E2}");
            double rowErr = RelativeError(split.Window.AsSpan((i - 1) * vocab, vocab).ToArray(), split.Rows[i]);
            Assert.True(rowErr <= 1e-5, $"verify row {i - 1} differs from its own decode step by {rowErr:E2} under split attention");
        }
    }

    /// <summary>
    /// Prompt-size attention on tensor cores (TS_Q4E_ATTN_MMA: a block per row and KV head, the group's query heads one
    /// 16-row mma tile, F16 operands) against the one-block-per-(head, row) kernel: a 600-row prompt's logits and the
    /// decode steps after it, dense (top_k 2048) and past the QSA width (top_k 16, every row from 19 on over its listed
    /// cells), over the F16 and the Q8_0 cache. The operands round to F16, so the answers move a little (as much as the
    /// Q8_0 cache moves them from F16); decode steps keep their kernels.
    /// </summary>
    [CudaTheory]
    [InlineData(2048, false)]
    [InlineData(16, false)]
    [InlineData(16, true)]
    public void TensorCorePromptAttention_AnswersAsTheScalarKernel(int indexerTopK, bool q8)
    {
        string path = Fixture(indexerTopK: indexerTopK);
        int[] prompt = Enumerable.Range(0, 600).Select(i => (i * 37 + 11) % 250).ToArray();
        KvCacheDtype saved = KvCacheDtypeConfig.Current;
        bool savedExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            KvCacheDtypeConfig.Set(q8 ? KvCacheDtype.Q8_0 : KvCacheDtype.F16);
            (float[][] Rows, long Calls) Answer(string mma)
            {
                _env.Set("TS_Q4E_ATTN_MMA", mma);
                using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
                var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
                var rows = new float[Steps.Length + 1][];
                rows[0] = (float[])model.ForwardRefill(prompt).Clone();
                for (int i = 0; i < Steps.Length; i++)
                    rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
                return (rows, engine.PromptAttentionMmaCalls);
            }
            var scalar = Answer("0");
            var mma = Answer(null);
            Assert.Equal(0, scalar.Calls);
            Assert.True(mma.Calls > 0, "the prompt's attention did not run on tensor cores");
            for (int i = 0; i < mma.Rows.Length; i++)
            {
                double err = RelativeError(mma.Rows[i], scalar.Rows[i]);
                _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: tensor cores vs scalar kernel, relative error {err:E2}");
                // Measured 4.2e-3 .. 5.4e-3 on the prefill and 3.0e-3 .. 5.2e-3 on the decode steps after it on an RTX 5090,
                // through eight layers of the fixture (the Q8_0 cache moves them 3.7e-3 .. 4.8e-3 from F16);
                // Q4eAttentionMmaTests holds the kernel itself to 1e-3 a row.
                Assert.True(err <= 1e-2, $"{(i == 0 ? "prefill" : $"decode {i}")}: the tensor-core attention moved the logits by {err:E2}");
            }
        }
        finally
        {
            KvCacheDtypeConfig.RestoreForTests(saved, savedExplicit);
            _env.Set("TS_Q4E_ATTN_MMA", null);
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void TheMtpFixture_DraftsOnTheNativeHead_Cpu() => NativeMtpSmoke(BackendType.GgmlCpu);

    /// <summary>
    /// A draft-head catch-up whose graph does not fit the device goes in halves (on ggml_cuda after four concurrent
    /// requests, a 1,045-row prompt chunk's catch-up asked 1.29 GB and the request failed). The seam takes that branch
    /// for every forward past 4 rows: the 40-token catch-up runs as sixteen forwards of 2 or 3 rows, the head's position
    /// covers the prompt (the next step is no gap), and the chained draft answers as after one 40-row forward.
    /// </summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheDraftHeadCatchUp_InHalves_AnswersAsOneForward_GgmlCuda()
    {
        string path = Fixture(indexerTopK: 2048);
        string mtp = Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, "qwen4exp-mtp.gguf"), indexerTopK: 2048);
        (float[] Logits, float[] Streams) Draft(int maxRows)
        {
            Qwen4ExpModel.MtpForwardRowsForTest = maxRows;
            try
            {
                using ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda, draftModelPath: mtp);
                var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
                var head = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.IDraftHead>(model);
                int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
                var hidden = new float[Prompt.Length * feat];
                spec.SpecForward(Prompt, hidden, new float[Prompt.Length * vocab], true);
                var rows = new float[Prompt.Length * feat];
                Array.Copy(hidden, 0, rows, feat, (Prompt.Length - 1) * feat);
                head.DraftCatchUp(Prompt, rows, 0);
                var logits = new float[vocab];
                var streams = new float[feat];
                head.DraftStep(Steps[0], hidden.AsSpan((Prompt.Length - 1) * feat, feat).ToArray(), Prompt.Length, logits, streams);
                return (logits, streams);
            }
            finally { Qwen4ExpModel.MtpForwardRowsForTest = 0; }
        }
        var whole = Draft(0);
        var halves = Draft(4);
        double num = 0, den = 0;
        for (int i = 0; i < whole.Logits.Length; i++) { double d = halves.Logits[i] - whole.Logits[i]; num += d * d; den += (double)whole.Logits[i] * whole.Logits[i]; }
        double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
        _output.WriteLine($"draft logits after a halved catch-up vs one forward: relative L2 {rel:E3}, argmax {ArgMax(halves.Logits)} vs {ArgMax(whole.Logits)}");
        Assert.All(halves.Logits, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(ArgMax(whole.Logits), ArgMax(halves.Logits));
        Assert.True(rel < 1e-2, $"a catch-up in halves drafts {rel:E3} away from one forward");
    }

    [GgmlFact(BackendType.GgmlMetal)]
    public void TheMtpFixture_DraftsOnTheNativeHead_Metal() => NativeMtpSmoke(BackendType.GgmlMetal);

    /// <summary>The synthetic NextN file passes the published head's validation and the native head drafts from it.</summary>
    private void NativeMtpSmoke(BackendType backend)
    {
        string path = Fixture(indexerTopK: 2048);
        string mtp = Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, "qwen4exp-mtp.gguf"), indexerTopK: 2048);
        using ModelBase model = ModelBase.Create(path, backend, draftModelPath: mtp);
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
        var head = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.IDraftHead>(model);
        Assert.True(head.HasDraftHead);
        int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        var hidden = new float[Prompt.Length * feat];
        spec.SpecForward(Prompt, hidden, new float[Prompt.Length * vocab], true);
        var rows = new float[Prompt.Length * feat];
        Array.Copy(hidden, 0, rows, feat, (Prompt.Length - 1) * feat);
        head.DraftCatchUp(Prompt, rows, 0);
        var logits = new float[vocab];
        var streams = new float[feat];
        head.DraftStep(Steps[0], hidden.AsSpan((Prompt.Length - 1) * feat, feat).ToArray(), Prompt.Length, logits, streams);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        Assert.All(streams, v => Assert.True(float.IsFinite(v)));
        _output.WriteLine($"draft argmax {ArgMax(logits)}, |streams| {Math.Sqrt(streams.Sum(v => (double)v * v)):F3}");
        Assert.True(streams.Any(v => v != 0f), "degenerate draft streams");
    }

    /// <summary>
    /// The MTP block of a shared-target NextN file on the direct engine, against the native MTP executor ggml_cuda
    /// runs (ggml_ops_qwen4exp_mtp.cpp): the trunk's hidden streams of a prompt (the drafter's input, before the
    /// output mixer), a catch-up over the prompt (token k with the streams of token k - 1), then three chained draft
    /// steps, each step's logits and streams. Then the fused catch-up-and-step answers as a catch-up and a step.
    /// </summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheMtpBlock_DraftsAsTheNativeHead()
    {
        string path = Fixture(indexerTopK: 2048);
        string mtp = Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, "qwen4exp-mtp.gguf"), indexerTopK: 2048);
        (float[] Hidden, float[][] Logits, float[][] Streams) Drafts(BackendType backend)
        {
            using ModelBase model = ModelBase.Create(path, backend, draftModelPath: mtp);
            var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeTarget>(model);
            var head = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.IDraftHead>(model);
            Assert.True(head.HasDraftHead, $"{backend}: no draft head");
            int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
            Assert.Equal(Qwen4ExpSyntheticModelBuilder.Hc * Qwen4ExpSyntheticModelBuilder.Hidden, feat);
            var hidden = new float[Prompt.Length * feat];
            spec.SpecForward(Prompt, hidden, new float[vocab], false);
            var rows = new float[Prompt.Length * feat];
            Array.Copy(hidden, 0, rows, feat, (Prompt.Length - 1) * feat);
            head.DraftCatchUp(Prompt, rows, 0);
            var logits = new float[3][];
            var streams = new float[3][];
            float[] h = hidden.AsSpan((Prompt.Length - 1) * feat, feat).ToArray();
            int token = Steps[0];
            for (int i = 0; i < 3; i++)
            {
                logits[i] = new float[vocab];
                streams[i] = new float[feat];
                head.DraftStep(token, h, Prompt.Length + i, logits[i], streams[i]);
                token = ArgMax(logits[i]);
                h = streams[i];
            }
            return (hidden, logits, streams);
        }
        var native = Drafts(BackendType.GgmlCuda);
        var cuda = Drafts(BackendType.Cuda);
        double hiddenErr = RelativeError(cuda.Hidden, native.Hidden);
        _output.WriteLine($"trunk streams: relative error {hiddenErr:E3}");
        Assert.True(hiddenErr <= 2e-2, $"the trunk's hidden streams strayed {hiddenErr:E3} from the native span");
        for (int i = 0; i < 3; i++)
        {
            double le = RelativeError(cuda.Logits[i], native.Logits[i]), se = RelativeError(cuda.Streams[i], native.Streams[i]);
            _output.WriteLine($"draft {i}: logits relative error {le:E3} (argmax {ArgMax(cuda.Logits[i])} vs {ArgMax(native.Logits[i])}), streams {se:E3}");
            Assert.True(le <= 2e-2 && se <= 2e-2, $"draft step {i} strayed from the native head: logits {le:E3}, streams {se:E3}");
        }

        // One call over the catch-up rows and the first draft row answers as the two calls.
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda, draftModelPath: mtp))
        {
            // The server's attach step after load takes the head the constructor already attached (it used to refuse
            // it: "does not use a separate draft GGUF").
            _env.Set(TensorSharp.Runtime.Speculative.SpeculationEnvVars.DraftModel, mtp);
            Assert.True(SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(model, out string attachError), attachError);
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            var head = (TensorSharp.Runtime.Speculative.IDraftHead)model;
            int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
            var hidden = new float[Prompt.Length * feat];
            spec.SpecForward(Prompt, hidden, new float[vocab], false);
            var rows = new float[(Prompt.Length + 1) * feat];
            Array.Copy(hidden, 0, rows, feat, Prompt.Length * feat);
            var tokens = Prompt.Append(Steps[0]).ToArray();
            var fusedLogits = new float[vocab];
            var fusedStreams = new float[feat];
            head.DraftCatchUpAndStep(tokens, rows, 0, fusedLogits, fusedStreams);
            double le = RelativeError(fusedLogits, cuda.Logits[0]), se = RelativeError(fusedStreams, cuda.Streams[0]);
            _output.WriteLine($"fused catch-up and step: logits relative error {le:E3}, streams {se:E3}");
            // Its join rows (tokens x streams) take the prompt-size matmul route instead of the decode kernels.
            Assert.True(le <= 1e-2 && se <= 1e-2, $"the fused call differs from catch-up + step: logits {le:E3}, streams {se:E3}");
        }
    }

    /// <summary>
    /// The MTP draft chain (Q4eCudaEngine.MtpDraftChain): one call folds the catch-up rows with the first step and runs
    /// the next steps on the device, each fed the previous step's first-max token and the head's own streams. It picks
    /// the tokens the step-by-step calls pick (a host argmax over their logits) and reports the per-token gate's
    /// confidence (top-1 over the 10 largest logits) as the host computes it. With a draft vocabulary
    /// (TS_Q4E_DRAFT_VOCAB) the chain scores the first ids only and picks what the host picks over those ids.
    /// </summary>
    [CudaTheory]
    [InlineData(0)]
    [InlineData(97)]
    public void TheMtpDraftChain_PicksAsStepByStepCalls(int draftVocab)
    {
        _env.Set("TS_Q4E_DRAFT_VOCAB", draftVocab.ToString(System.Globalization.CultureInfo.InvariantCulture));
        string path = Fixture(indexerTopK: 2048);
        string mtp = Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, "qwen4exp-mtp.gguf"), indexerTopK: 2048);
        const int Steps3 = 3;
        (int[] Ids, float[] Probs) Run(bool chain)
        {
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda, draftModelPath: mtp);
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            var head = (TensorSharp.Runtime.Speculative.IDraftHead)model;
            Assert.True(head.SupportsDraftChain, "the direct engine offers no draft chain");
            int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
            var hidden = new float[Prompt.Length * feat];
            spec.SpecForward(Prompt, hidden, new float[vocab], false);
            var rows = new float[(Prompt.Length + 1) * feat];
            Array.Copy(hidden, 0, rows, feat, Prompt.Length * feat);
            var tokens = Prompt.Append(Steps[0]).ToArray();
            var ids = new int[Steps3];
            var probs = new float[Steps3];
            if (chain)
            {
                Assert.Equal(Steps3, head.DraftChain(tokens, rows, 0, Steps3, ids, probs));
                return (ids, probs);
            }
            var logits = new float[vocab];
            var h = new float[feat];
            var next = new float[feat];
            head.DraftCatchUpAndStep(tokens, rows, 0, logits, h);
            for (int i = 0; i < Steps3; i++)
            {
                ids[i] = TensorSharp.Runtime.Speculative.DraftHeadSpeculator.ArgmaxWithTopKConfidence(logits,
                    draftVocab > 0 ? draftVocab : vocab, out probs[i]);
                if (i + 1 < Steps3)
                {
                    head.DraftStep(ids[i], h, tokens.Length + i, logits, next);
                    (h, next) = (next, h);
                }
            }
            return (ids, probs);
        }
        var stepwise = Run(chain: false);
        var chained = Run(chain: true);
        _output.WriteLine($"step by step: [{string.Join(",", stepwise.Ids)}] p [{string.Join(",", stepwise.Probs.Select(p => p.ToString("F5")))}]");
        _output.WriteLine($"chained:      [{string.Join(",", chained.Ids)}] p [{string.Join(",", chained.Probs.Select(p => p.ToString("F5")))}]");
        Assert.Equal(stepwise.Ids, chained.Ids);
        if (draftVocab > 0)
            Assert.All(chained.Ids, id => Assert.True(id < draftVocab, $"drafted {id}, outside the first {draftVocab} ids"));
        for (int i = 0; i < Steps3; i++)
            Assert.True(Math.Abs(stepwise.Probs[i] - chained.Probs[i]) <= 1e-4f, $"step {i}: confidence {chained.Probs[i]} vs {stepwise.Probs[i]}");
    }

    /// <summary>The MTP block restarts its cache after a gap (IDraftHead.DraftHeadResumesAfterGap): a catch-up without hidden
    /// rows moves its row 0 to the end of the gap, and the rows replayed from there answer bit for bit as a block that
    /// started with them at position 0 (its cache counts from the restart, and RoPE scores depend on distance alone). A
    /// draft from before the restart is refused. A lone request still reads its prompt through the context
    /// (DraftHeadConditionsOnPrompt).</summary>
    [CudaFact]
    public void TheMtpBlock_RestartsAfterAGap_AsABlockThatStartedThere()
    {
        string path = Fixture(indexerTopK: 2048);
        string mtp = Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, "qwen4exp-mtp.gguf"), indexerTopK: 2048);
        using ModelBase model = ModelBase.Create(path, BackendType.Cuda, draftModelPath: mtp);
        var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
        var head = (TensorSharp.Runtime.Speculative.IDraftHead)model;
        Assert.True(head.DraftHeadResumesAfterGap);
        Assert.True(head.DraftHeadConditionsOnPrompt);
        int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        var hidden = new float[Prompt.Length * feat];
        spec.SpecForward(Prompt, hidden, new float[vocab], false);
        // Rows from position S: token S + t with the streams of the token before it.
        const int S = 20;
        int k = Prompt.Length - S;
        int[] tokens = Prompt[S..];
        var rows = new float[k * feat];
        Array.Copy(hidden, (long)(S - 1) * feat, rows, 0, (long)k * feat);

        var fresh = new float[vocab];
        var hFresh = new float[feat];
        head.DraftCatchUpAndStep(tokens, rows, 0, fresh, hFresh);
        // The gap [0, S) reported, then the same rows at their own positions.
        head.DraftCatchUp(Prompt[..S], null, 0);
        var restarted = new float[vocab];
        var hRestarted = new float[feat];
        head.DraftCatchUpAndStep(tokens, rows, S, restarted, hRestarted);
        Assert.True(restarted.AsSpan().SequenceEqual(fresh), "the restarted block's draft logits differ from a block that started there");
        Assert.True(hRestarted.AsSpan().SequenceEqual(hFresh), "the restarted block's streams differ from a block that started there");
        Assert.ThrowsAny<InvalidOperationException>(
            () => head.DraftCatchUpAndStep(tokens[..1], rows[..feat], S - 1, new float[vocab], new float[feat]));
    }

    /// <summary>
    /// The MTP catch-up after a grouped prompt reads the trunk streams of the last group's positions from the group's
    /// buffer (Q4eCudaEngine.MtpUploadRows), the values the host's copy holds, so the draft chain picks the same tokens with
    /// the same confidences, bit for bit, as with every row from the host (TS_Q4E_MTP_D2D=0). A 300-token prompt in groups
    /// of 192 rows (64-row ubatches), the catch-up from position 150: rows for positions 150..191 come from the host, the
    /// 108 rows of the last group from the device.
    /// </summary>
    [CudaFact]
    public void TheMtpCatchUpAfterAGroupedPrompt_ReadsTheLastGroupsStreams_BitForBit()
    {
        _env.Set("TS_Q4E_HOST_EXPERTS", "0");
        _env.Set("TS_Q4E_UBATCH", "64");
        _env.Set("TS_Q4E_PROMPT_GROUP", "192");
        string path = Fixture(indexerTopK: 16, expertGateUpType: SyntheticGguf.GgmlType.Q2_0, expertDownType: SyntheticGguf.GgmlType.Q2_0);
        string mtp = Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, "qwen4exp-mtp.gguf"), indexerTopK: 16);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);
        // Prompt groups need expert residency: two thirds resident.
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", (expertBytes * 2 / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        int[] prompt = Enumerable.Range(0, 300).Select(i => (i * 37 + 11) % 250).ToArray();
        const int S = 150, Steps3 = 3;

        (int[] Ids, float[] Probs, long FromDevice, long Groups) Run(string d2d)
        {
            _env.Set("TS_Q4E_MTP_D2D", d2d);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda, draftModelPath: mtp);
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            var head = (TensorSharp.Runtime.Speculative.IDraftHead)model;
            int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
            var hidden = new float[prompt.Length * feat];
            spec.SpecForward(prompt, hidden, new float[vocab], false);
            // The block restarts at S (the gap before it reported), then catches up on the prompt's rows from S and the
            // next token, row t with the streams of the token before it.
            head.DraftCatchUp(prompt[..S], null, 0);
            int[] tokens = prompt[S..].Append(Steps[0]).ToArray();
            var rows = new float[tokens.Length * feat];
            Array.Copy(hidden, (long)(S - 1) * feat, rows, 0, (long)tokens.Length * feat);
            var ids = new int[Steps3];
            var probs = new float[Steps3];
            Assert.Equal(Steps3, head.DraftChain(tokens, rows, S, Steps3, ids, probs));
            return (ids, probs, engine.MtpRowsFromDevice, engine.PromptGroupsRun);
        }
        var host = Run("0");
        var device = Run(null);
        _output.WriteLine($"host rows:   [{string.Join(",", host.Ids)}] p [{string.Join(",", host.Probs.Select(p => p.ToString("R")))}]");
        _output.WriteLine($"device rows: [{string.Join(",", device.Ids)}] p [{string.Join(",", device.Probs.Select(p => p.ToString("R")))}], " +
            $"{device.FromDevice} rows from the group's buffer");
        Assert.Equal(2, device.Groups);
        Assert.Equal(0, host.FromDevice);
        Assert.Equal(prompt.Length - 192, device.FromDevice);
        Assert.Equal(host.Ids, device.Ids);
        for (int i = 0; i < Steps3; i++)
            Assert.True(BitConverter.SingleToInt32Bits(host.Probs[i]) == BitConverter.SingleToInt32Bits(device.Probs[i]),
                $"step {i}: confidence {device.Probs[i]:R} vs {host.Probs[i]:R}");
    }

    /// <summary>
    /// The last prompt group's streams stop standing for their positions once the slot's position falls below them: a
    /// grouped 150-token prompt, then the sequence reset and a short (ungrouped) prompt over positions the group had held.
    /// Its catch-up takes every row from the host - none from the stale group buffer - and drafts what a fresh engine
    /// drafts for that prompt alone, bit for bit.
    /// </summary>
    [CudaFact]
    public void TheMtpCatchUp_AfterTheSlotIsReset_DoesNotReadTheOldGroupsStreams()
    {
        _env.Set("TS_Q4E_HOST_EXPERTS", "0");
        _env.Set("TS_Q4E_UBATCH", "64");
        _env.Set("TS_Q4E_PROMPT_GROUP", "192");
        string path = Fixture(indexerTopK: 16, expertGateUpType: SyntheticGguf.GgmlType.Q2_0, expertDownType: SyntheticGguf.GgmlType.Q2_0);
        string mtp = Qwen4ExpSyntheticModelBuilder.WriteMtp(Path.Combine(_dir, "qwen4exp-mtp.gguf"), indexerTopK: 16);
        long expertBytes = 0;
        using (var g = new GgufFile(path))
            foreach (var (name, t) in g.Tensors)
                if (name.Contains("_exps.", StringComparison.Ordinal))
                    expertBytes += g.GetTensorByteCount(t);
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", (expertBytes * 2 / 3 / 1e9).ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        int[] first = Enumerable.Range(0, 150).Select(i => (i * 37 + 11) % 250).ToArray();
        int[] second = Enumerable.Range(0, 40).Select(i => (i * 53 + 7) % 250).ToArray();
        const int Steps3 = 3;

        // The second prompt's catch-up: every row from position 0 with the streams of the token before it.
        (int[] Ids, float[] Probs, long FromDevice) Draft(ModelBase model, int[] prompt)
        {
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            var head = (TensorSharp.Runtime.Speculative.IDraftHead)model;
            int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
            var hidden = new float[prompt.Length * feat];
            spec.SpecForward(prompt, hidden, new float[vocab], false);
            int[] tokens = prompt.Append(Steps[0]).ToArray();
            var rows = new float[tokens.Length * feat];
            Array.Copy(hidden, 0, rows, feat, (long)prompt.Length * feat);
            var ids = new int[Steps3];
            var probs = new float[Steps3];
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            long before = engine.MtpRowsFromDevice;
            Assert.Equal(Steps3, head.DraftChain(tokens, rows, 0, Steps3, ids, probs));
            return (ids, probs, engine.MtpRowsFromDevice - before);
        }

        (int[] Ids, float[] Probs, long FromDevice) afterReset, fresh;
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda, draftModelPath: mtp))
        {
            var engine = Assert.IsType<Qwen4ExpCudaModel>(model).Engine;
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            spec.SpecForward(first, new float[first.Length * spec.SpecFeatureSize], new float[model.Config.VocabSize], false);
            Assert.True(engine.PromptGroupsRun > 0, "the first prompt was not read in a group");
            engine.Reset();
            afterReset = Draft(model, second);
        }
        _env.Set("TS_Q4E_MTP_D2D", "0");
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda, draftModelPath: mtp))
            fresh = Draft(model, second);
        _output.WriteLine($"after the reset: [{string.Join(",", afterReset.Ids)}], {afterReset.FromDevice} rows from the group's buffer; " +
            $"fresh: [{string.Join(",", fresh.Ids)}]");
        Assert.Equal(0, afterReset.FromDevice);
        Assert.Equal(fresh.Ids, afterReset.Ids);
        for (int i = 0; i < Steps3; i++)
            Assert.True(BitConverter.SingleToInt32Bits(fresh.Probs[i]) == BitConverter.SingleToInt32Bits(afterReset.Probs[i]),
                $"step {i}: confidence {afterReset.Probs[i]:R} vs {fresh.Probs[i]:R}");
    }

    /// <summary>A decode step's hyper-connection scatter fused with the next mixer's first pass computes what the two
    /// launches compute: the prompt and every decode step, bit for bit, with QSA past its width.</summary>
    [CudaFact]
    public void TheFusedScatterAndMixerPass_AnswersBitForBitAsTwoLaunches()
    {
        string path = Fixture();
        float[][] fused, split;
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
            fused = Run(model);
        _env.Set("TS_Q4E_FUSE_HC", "0");
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
            split = Run(model);
        for (int i = 0; i < fused.Length; i++)
            Assert.True(fused[i].AsSpan().SequenceEqual(split[i]), $"{(i == 0 ? "prefill" : $"decode {i}")}: the fused kernel changed the logits");
    }

    /// <summary>The shared expert on its own stream beside the router and the routed experts (forked and joined inside the
    /// captured step) computes what the serial chain computes, bit for bit, decode steps and a verify window alike.</summary>
    [CudaFact]
    public void TheForkedSharedExpert_AnswersBitForBitAsTheSerialChain()
    {
        string path = Fixture();
        int vocab = Qwen4ExpSyntheticModelBuilder.Vocab;
        (float[][] Rows, float[] Window) Answer()
        {
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            float[][] rows = Run(model);
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            var window = new float[4 * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(new[] { 5, 77, 123, 9 }, null, window, true);
            return (rows, window);
        }
        var forked = Answer();
        _env.Set("TS_Q4E_FORK_SHEXP", "0");
        var serial = Answer();
        for (int i = 0; i < forked.Rows.Length; i++)
            Assert.True(forked.Rows[i].AsSpan().SequenceEqual(serial.Rows[i]), $"{(i == 0 ? "prefill" : $"decode {i}")}: the fork changed the logits");
        Assert.True(forked.Window.AsSpan().SequenceEqual(serial.Window), "the fork changed the verify window's logits");
    }

    /// <summary>--kv-cache-dtype q8_0 on the direct engine: the attention cache as int8 codes with an F16 scale per 32
    /// values (ggml's Q8_0 arithmetic), half the F16 cache's bytes. The prompt and every decode step stay close to the F16
    /// cache's answers, and a verify window over the Q8_0 cache answers as a plain decode on that same cache does.</summary>
    [CudaFact]
    public void AQ8KvCache_AnswersCloseToF16_AndVerifiesAsItsOwnDecode()
    {
        string path = Fixture();
        int vocab = Qwen4ExpSyntheticModelBuilder.Vocab;
        KvCacheDtype saved = KvCacheDtypeConfig.Current;
        bool savedExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            float[][] f16;
            using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
                f16 = Run(model);
            // q8_0 is the direct engine's default when the operator names no dtype.
            KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, explicitlySet: false);
            using (ModelBase byDefault = ModelBase.Create(path, BackendType.Cuda))
                Assert.Equal(KvCacheDtype.Q8_0, byDefault.KvCacheDtype);
            KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0);
            using ModelBase q8 = ModelBase.Create(path, BackendType.Cuda);
            Assert.Equal(KvCacheDtype.Q8_0, q8.KvCacheDtype);
            float[][] rows = Run(q8);
            for (int i = 0; i < rows.Length; i++)
            {
                double err = RelativeError(rows[i], f16[i]);
                _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: q8_0 vs f16 KV relative error {err:E2}");
                // Measured 3.7e-3 .. 4.8e-3 on an RTX 5090 (the engine-vs-native-span difference is 4.5e-3 .. 6.4e-3).
                Assert.True(err <= 1e-2, $"{(i == 0 ? "prefill" : $"decode {i}")}: the q8_0 cache moved the logits by {err:E2}");
            }

            int[] window = { Steps[0], 77, 123, 9 };
            q8.ResetKVCache();
            q8.ForwardRefill(Prompt);
            var plain = new List<float[]>();
            foreach (int token in window)
                plain.Add((float[])q8.Forward(new[] { token }).Clone());
            q8.ResetKVCache();
            q8.ForwardRefill(Prompt);
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)q8;
            var all = new float[window.Length * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(window, null, all, true);
            for (int i = 0; i < window.Length; i++)
            {
                double err = RelativeError(all.AsSpan(i * vocab, vocab).ToArray(), plain[i]);
                Assert.True(err <= 1e-5, $"verify row {i} over the q8_0 cache differs from its decode by {err:E2}");
            }
        }
        finally
        {
            KvCacheDtypeConfig.RestoreForTests(saved, savedExplicit);
        }
    }

    /// <summary>A router the file stores as BF16 (ISTA's) is read as stored at decode size, half the bytes of its F32
    /// widening, with the same sums: decode steps and a verify window bit for bit as the F32 router's, and a prompt (which
    /// widens it for its GEMM) as well.</summary>
    [CudaFact]
    public void ABf16Router_AnswersBitForBitAsItsF32Widening()
    {
        // 256 experts: the BF16 GEMV is the wide (warp-per-output) form a real router takes.
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "qwen4exp-bf16router.gguf"), 2048,
            routerType: SyntheticGguf.GgmlType.BF16, expertCount: 256, expertUsedCount: 8);
        int vocab = Qwen4ExpSyntheticModelBuilder.Vocab;
        (float[][] Rows, float[] Window) Answer()
        {
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            float[][] rows = Run(model);
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            var window = new float[4 * vocab];
            spec.SpecSnapshotRecurrentState();
            spec.SpecForward(new[] { 5, 77, 123, 9 }, null, window, true);
            return (rows, window);
        }
        var bf16 = Answer();
        _env.Set("TS_Q4E_BF16_ROUTER", "0");
        var f32 = Answer();
        for (int i = 0; i < bf16.Rows.Length; i++)
            Assert.True(bf16.Rows[i].AsSpan().SequenceEqual(f32.Rows[i]), $"{(i == 0 ? "prefill" : $"decode {i}")}: the BF16 router changed the logits");
        Assert.True(bf16.Window.AsSpan().SequenceEqual(f32.Window), "the BF16 router changed the verify window's logits");
    }

    [CudaFact]
    public void AnUndecodableExpertLayout_IsRefusedAtLoadByTensorName()
    {
        string path = Fixture(q4kxlExperts: true, expertDownType: SyntheticGguf.GgmlType.Q4_0);
        var ex = Assert.Throws<NotSupportedException>(() => ModelBase.Create(path, BackendType.Cuda));
        Assert.StartsWith("[q4e-cuda] blk.0.ffn_down_exps.weight is Q4_0 (ggml type 2)", ex.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out string reason));
        Assert.Contains("--backend ggml_cuda", reason);
    }

    /// <summary>Two devices answer as one: the streams cross the boundary through pinned memory, and
    /// each layer keeps its caches, recurrent state and PLE history on its own device.</summary>
    [MultiCudaFact(2)]
    public void ALayerSplit_AnswersAsOneDevice()
    {
        string path = Fixture();
        float[][] single, split;
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
            single = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda, layerSplitDegree: 2))
            split = Run(model);
        for (int i = 0; i < single.Length; i++)
            Assert.True(RelativeError(split[i], single[i]) <= 1e-5, $"forward {i} differs across the split");
    }

    /// <summary>A reset clears the recurrent states, the QSA keys and the PLE history as well as the
    /// attention cache.</summary>
    [CudaFact]
    public void AReset_StartsAFreshSequence()
    {
        using ModelBase model = ModelBase.Create(Fixture(), BackendType.Cuda);
        float[][] first = Run(model);
        model.ResetKVCache();
        float[][] again = Run(model);
        for (int i = 0; i < first.Length; i++)
            Assert.True(RelativeError(again[i], first[i]) <= 1e-6, $"forward {i} differs after a reset");
    }

    // ---------------------------------------------------------------- QSA

    /// <summary>
    /// Every sparse QSA step (top_k 16: past 19 cells each query keeps its best blocks of four) held
    /// to the rules of the native executor's indexer, computed from the engine's own inputs:
    /// <list type="bullet">
    /// <item>a block's key is the mean of its members' cached raw keys, RMS-normed times k_norm and
    /// rotated to the block's first position;</item>
    /// <item>a query scores the complete blocks it has seen as the sum over indexer heads of
    /// ReLU'd dot products;</item>
    /// <item>it attends the cells of its own unfinished block, then the best blocks' cells until
    /// top_k + ratio - 1 cells are listed, ties to the lower block.</item>
    /// </list>
    /// The native top-k leaves the order among equal scores unspecified, so an end-to-end comparison
    /// cannot pin a selection down; the dense comparison above covers everything else.
    /// </summary>
    [CudaFact]
    public void TheSparseSelection_FollowsTheIndexerRules()
    {
        string path = Fixture();
        var probes = new System.Collections.Generic.List<TensorSharp.Cuda.Q4eCudaEngine.SparseLayerProbe>();
        using (var ex = OpenExecutor(path))
        {
            ex.Engine.SparseProbe = probes.Add;
            Step(ex, Prompt);
            foreach (int token in Steps)
                Step(ex, new[] { token });
        }
        // Two QSA layers, each past the width in the prompt and in every step.
        Assert.Equal(2 * (1 + Steps.Length), probes.Count);
        using var gguf = new GgufFile(path);
        foreach (var probe in probes)
            CheckSparseLayer(probe, gguf);
    }

    private static void CheckSparseLayer(TensorSharp.Cuda.Q4eCudaEngine.SparseLayerProbe p, GgufFile gguf)
    {
        int d = p.IdxDim, h = p.IdxHeads, r = p.Ratio;
        string where = $"layer {p.Layer} at {p.P0}";
        float[] kNorm = F32(gguf, $"blk.{p.Layer}.indexer.k_norm.weight");

        // Block keys from the cached raw keys, as stored.
        int blocks = (p.P0 + p.Nt) / r;
        var keys = new float[blocks * d];
        for (int b = 0; b < blocks; b++)
        {
            var mean = new double[d];
            for (int c = 0; c < d; c++)
            {
                double acc = 0;
                for (int j = 0; j < r; j++)
                    acc += (float)p.RawKeys[(long)(b * r + j) * d + c];
                mean[c] = acc / r;
            }
            double ss = mean.Sum(v => v * v);
            double inv = 1.0 / Math.Sqrt(ss / d + p.Eps);
            var x = new double[d];
            for (int c = 0; c < d; c++)
                x[c] = mean[c] * inv * kNorm[c];
            Rope(x, b * r, p.NRot, p.RopeBase, p.RopeFreqScale);
            for (int c = 0; c < d; c++)
                keys[b * d + c] = (float)x[c];
        }
        AssertClose(keys, p.Pooled, 1e-4, $"block keys, {where}");

        int width = p.TopK + r - 1;
        for (int t = 0; t < p.Nt; t++)
        {
            int q = p.P0 + t;
            if (q + 1 <= width)
            {
                Assert.Equal(-1, p.CellCnt[t]);
                continue;
            }
            int nb = (q + 1) / r;
            var want = new float[nb];
            var got = new float[nb];
            for (int b = 0; b < nb; b++)
            {
                double score = 0;
                for (int k = 0; k < h; k++)
                {
                    double dot = 0;
                    for (int c = 0; c < d; c++)
                        dot += (double)p.Queries[((long)t * h + k) * d + c] * p.Pooled[b * d + c];
                    score += Math.Max(dot, 0);
                }
                want[b] = (float)score;
                got[b] = p.Scores[(long)t * p.ScoreStride + b];
            }
            AssertClose(want, got, 1e-4, $"block scores, {where}, query {q}");

            // The rules over the ENGINE's scores (so the check is exact), in the kernel's float order.
            int tail = (q + 1) / r * r, t0 = q + 1 - tail, need = width - t0, full = need / r, part = need % r;
            int[] ranked = Enumerable.Range(0, nb).OrderByDescending(b => OrderKey(got[b])).ThenBy(b => b)
                .Take(full + (part > 0 ? 1 : 0)).ToArray();
            var cells = Enumerable.Range(tail, t0).ToList();
            for (int i = 0; i < ranked.Length; i++)
                cells.AddRange(Enumerable.Range(ranked[i] * r, i < full ? r : part));
            Assert.Equal(width, p.CellCnt[t]);
            Assert.Equal(cells.OrderBy(c => c), p.Cells.Skip(t * p.CellStride).Take(width).OrderBy(c => c));
        }
    }

    /// <summary>NEOX rotary over the first nRot values: pair (i, i + nRot/2) at angle
    /// freqScale * pos * base^(-2i/nRot).</summary>
    private static void Rope(double[] x, int pos, int nRot, float ropeBase, float freqScale)
    {
        int half = nRot / 2;
        for (int i = 0; i < half; i++)
        {
            double theta = freqScale * (pos * Math.Pow(ropeBase, -2.0 * i / nRot));
            double c = Math.Cos(theta), s = Math.Sin(theta);
            double a = x[i], b = x[i + half];
            x[i] = a * c - b * s;
            x[i + half] = a * s + b * c;
        }
    }

    /// <summary>The select kernel's total order on floats (the IEEE bits made unsigned-comparable).</summary>
    private static uint OrderKey(float v)
    {
        uint k = BitConverter.SingleToUInt32Bits(v);
        return (k & 0x80000000u) != 0 ? ~k : k | 0x80000000u;
    }

    private static float[] F32(GgufFile gguf, string name)
    {
        byte[] raw = gguf.ReadTensorData(gguf.Tensors[name]);
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        return f;
    }

    /// <summary>Worst absolute difference within <paramref name="tol"/> times the expected values'
    /// largest magnitude (at least 1).</summary>
    private static void AssertClose(float[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1.0, expected.Length == 0 ? 0 : expected.Max(v => Math.Abs((double)v)));
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]) / scale);
        Assert.True(worst <= tol, $"{what}: off by {worst:E3} (tolerance {tol:E1})");
    }

    // ---------------------------------------------------------------- sequence slots

    private static Qwen4ExpCudaExecutor OpenExecutor(string path) => new Qwen4ExpCudaExecutor(path, 1024, 64, 1);

    private static float[] Step(Qwen4ExpCudaExecutor ex, int[] tokens)
    {
        var logits = new float[ex.VocabSize];
        ex.Forward(tokens, logits);
        return logits;
    }

    private static int NewActiveSlot(Qwen4ExpCudaExecutor ex)
    {
        int slot = ex.SlotAlloc();
        Assert.True(slot >= 0);
        Assert.True(ex.SetActiveSlot(slot));
        return slot;
    }

    /// <summary>Two sequences interleaved chunk by chunk answer exactly as each alone: every cache,
    /// recurrent state and PLE history belongs to its slot.</summary>
    [CudaFact]
    public void InterleavedSlots_AnswerExactlyAsEachSequenceAlone()
    {
        using var ex = OpenExecutor(Fixture());
        int[][] a = { Prompt[..24], Steps[..1], Steps[1..2] };
        int[][] b = { Prompt[16..40], Steps[3..4], Steps[4..5] };
        var aloneA = a.Select(c => Step(ex, c)).ToList();
        ex.ResetChecked();
        var aloneB = b.Select(c => Step(ex, c)).ToList();

        int sa = NewActiveSlot(ex);
        int sb = NewActiveSlot(ex);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(ex.SetActiveSlot(sa));
            Assert.True(RelativeError(Step(ex, a[i]), aloneA[i]) <= 1e-6, $"sequence A chunk {i}");
            Assert.True(ex.SetActiveSlot(sb));
            Assert.True(RelativeError(Step(ex, b[i]), aloneB[i]) <= 1e-6, $"sequence B chunk {i}");
        }
    }

    /// <summary>A captured decode step answers bit for bit as the same step issued kernel by kernel:
    /// the token, the position and the PLE rows are read at launch, blocks complete inside the
    /// graph, and crossing the QSA width (19 cells) captures the sparse regime beside the dense one.
    /// A freed sequence takes its graphs with it.</summary>
    [CudaFact]
    public void CapturedDecodeSteps_AnswerAsUncapturedSteps()
    {
        string path = Fixture();
        // Positions 12..27: dense to 18, sparse from 19, four blocks completing on the way.
        int[] tokens = Enumerable.Range(0, 16).Select(i => Steps[i % Steps.Length] + i).ToArray();
        float[][] Decode(Qwen4ExpCudaExecutor ex)
        {
            Step(ex, Prompt[..12]);
            return tokens.Select(t => Step(ex, new[] { t })).ToArray();
        }

        float[][] captured, plain;
        using (var ex = OpenExecutor(path))
        {
            captured = Decode(ex);
            Assert.Equal(2, ex.Engine.CapturedDecodeGraphs);

            int first = ex.ActiveSlot;
            int other = NewActiveSlot(ex);
            Step(ex, Prompt[..4]);
            Step(ex, new[] { 7 });
            Assert.Equal(3, ex.Engine.CapturedDecodeGraphs);
            Assert.True(ex.SetActiveSlot(first));
            Assert.True(ex.SlotFree(other));
            Assert.Equal(2, ex.Engine.CapturedDecodeGraphs);
        }
        _env.Set("TS_Q4E_GRAPHS", "0");
        using (var ex = OpenExecutor(path))
        {
            plain = Decode(ex);
            Assert.Equal(0, ex.Engine.CapturedDecodeGraphs);
        }
        for (int i = 0; i < tokens.Length; i++)
            Assert.Equal(plain[i], captured[i]);
    }

    /// <summary>A slot that starts small and grows (prefill chunks, then decode steps past its
    /// capacity, its captured graphs rebuilt each time) answers bit for bit as a slot that held the
    /// whole context from the start.</summary>
    [CudaFact]
    public void AGrowingSlot_AnswersAsOneThatHeldTheWholeContext()
    {
        string path = Fixture();
        int[] tokens = Enumerable.Range(0, 16).Select(i => Steps[i % Steps.Length] + i).ToArray();
        float[][] Sequence(Qwen4ExpCudaExecutor ex)
        {
            var rows = new List<float[]> { Step(ex, Prompt[..24]), Step(ex, Prompt[24..40]) };
            rows.AddRange(tokens.Select(t => Step(ex, new[] { t })));
            return rows.ToArray();
        }

        float[][] whole;
        using (var ex = OpenExecutor(path))
        {
            whole = Sequence(ex);
            Assert.Equal(0, ex.ReadTelemetry().Grows);
        }
        _env.Set("TS_Q4E_SLOT_TOKENS", "8");
        float[][] grown;
        using (var ex = OpenExecutor(path))
        {
            grown = Sequence(ex);
            // 8 positions grew to hold the 56 it forwarded: counted, with no failure.
            var counters = ex.ReadTelemetry();
            Assert.InRange(counters.Grows, 1, 64);
            Assert.Equal(0, counters.GrowFailures);
            Assert.True(counters.SlotCapacityTokens >= 56, $"{counters.SlotCapacityTokens} positions held");
            Assert.True(counters.SlotBytes > 0);
        }
        Assert.Equal(whole.Length, grown.Length);
        for (int i = 0; i < whole.Length; i++)
            Assert.Equal(whole[i], grown[i]);
    }

    /// <summary>A slot rewinds to its prompt-boundary checkpoint, or past it by replaying its own
    /// tokens, and continues exactly as a slot that never went further: the checkpoint is the
    /// recurrent states at the end of the last multi-token forward, and the attention rows below the
    /// head still hold their positions. Positions below the checkpoint stay refused.</summary>
    [CudaFact]
    public void ARewindFromTheCheckpoint_ContinuesAsTheSequenceThatStoppedThere()
    {
        using var ex = OpenExecutor(Fixture());
        int[] next = Prompt[24..31];
        int[] reply = Steps[..4];

        // The reference: the prompt, two of the reply's tokens forwarded as one chunk, the rest.
        int reference = NewActiveSlot(ex);
        Step(ex, Prompt[..24]);
        float[] wantAtBoundary = Step(ex, next);
        int replayed = NewActiveSlot(ex);
        Step(ex, Prompt[..24]);
        Step(ex, reply[..2]);
        float[] wantAfterReplay = Step(ex, next);

        // The slot under test: the prompt (a checkpoint at 24), then the reply decoded token by token.
        int slot = NewActiveSlot(ex);
        Step(ex, Prompt[..24]);
        foreach (int t in reply)
            Step(ex, new[] { t });
        Assert.True(ex.CanRewind(slot, 28, 24));
        Assert.True(ex.CanRewind(slot, 28, 26));
        Assert.False(ex.CanRewind(slot, 28, 23));
        Assert.False(ex.CanRewind(slot, 27, 24));   // not the slot's head

        var before = ex.ReadTelemetry();
        Assert.True(ex.Rewind(24));
        Assert.Equal(24, ex.NPast);
        Assert.Equal(wantAtBoundary, Step(ex, next));
        var atCheckpoint = ex.ReadTelemetry();
        Assert.Equal(before.RewindsToCheckpoint + 1, atCheckpoint.RewindsToCheckpoint);
        Assert.Equal(before.ReplayedTokens, atCheckpoint.ReplayedTokens);

        // That chunk moved the checkpoint to 24 + next.Length; go back to it, then nothing earlier.
        int head = 24 + next.Length;
        Assert.False(ex.Rewind(24));
        Assert.Equal(head, ex.NPast);
        Assert.Equal(atCheckpoint.RewindsRefused + 1, ex.ReadTelemetry().RewindsRefused);

        // Past the checkpoint: a second slot replays two of the reply's tokens.
        int second = NewActiveSlot(ex);
        Step(ex, Prompt[..24]);
        foreach (int t in reply)
            Step(ex, new[] { t });
        var beforeReplay = ex.ReadTelemetry();
        Assert.True(ex.Rewind(26));
        Assert.Equal(26, ex.NPast);
        Assert.Equal(wantAfterReplay, Step(ex, next));
        var afterReplay = ex.ReadTelemetry();
        Assert.Equal(beforeReplay.RewindsWithReplay + 1, afterReplay.RewindsWithReplay);
        Assert.Equal(beforeReplay.ReplayedTokens + 2, afterReplay.ReplayedTokens);
        Assert.Equal(beforeReplay.RewindsToCheckpoint, afterReplay.RewindsToCheckpoint);
        _ = reference; _ = replayed; _ = second;
    }

    /// <summary>A batched decode step computes each sequence's row exactly as that sequence's own step
    /// does, over several steps, with the rows at different positions on both sides of the QSA
    /// width.</summary>
    [CudaFact]
    public void BatchedDecode_ComputesEverySequencesOwnStep()
    {
        using var ex = OpenExecutor(Fixture());
        int[] prompts = { 20, 13, 31 };
        int n = prompts.Length, steps = 4;
        var batched = new int[n];
        var serial = new int[n];
        for (int i = 0; i < n; i++)
        {
            batched[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
            serial[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
        }
        for (int step = 0; step < steps; step++)
        {
            int[] tokens = Enumerable.Range(0, n).Select(i => Steps[(i + step) % Steps.Length]).ToArray();
            int[] positions = prompts.Select(p => p + step).ToArray();
            var rows = new float[n * ex.VocabSize];
            Assert.True(ex.ForwardBatchedDecode(batched, tokens, positions, rows));
            for (int i = 0; i < n; i++)
            {
                Assert.True(ex.SetActiveSlot(serial[i]));
                float[] want = Step(ex, new[] { tokens[i] });
                Assert.Equal(want, rows.AsSpan(i * ex.VocabSize, ex.VocabSize).ToArray());
            }
        }
        // The steps ran as captured graphs (one per set of rows and their QSA regimes), and freeing a
        // slot drops the graphs holding it.
        int graphs = ex.Engine.CapturedBatchedDecodeGraphs;
        Assert.True(graphs > 0);
        // The batched steps launched captured graphs, and the counters saw every capture.
        var counters = ex.ReadTelemetry();
        Assert.True(counters.BatchedGraphCaptures >= graphs, $"{counters.BatchedGraphCaptures} captures, {graphs} graphs");
        Assert.InRange(counters.BatchedGraphLaunches, 1, steps);
        Assert.Equal(0, counters.GraphCaptureFailures);
        Assert.True(counters.ForwardIssueSeconds > 0);
        Assert.True(ex.SlotFree(batched[0]));
        Assert.True(ex.Engine.CapturedBatchedDecodeGraphs < graphs);
    }

    /// <summary>A batched verify (several sequences' draft windows in one forward, Q4eCudaEngine.ForwardBatchedVerify) computes
    /// every sequence's own window: each row's first-max token and hidden streams as that sequence's own verify, and after
    /// the commit (a different number of rows kept per sequence) the next step's logits as after its own commit. Dense
    /// rows (top_k 2048) and QSA rows past the width (top_k 16).</summary>
    [CudaTheory]
    [InlineData(2048)]
    [InlineData(16)]
    public void BatchedVerify_ComputesEverySequencesOwnWindow(int indexerTopK)
    {
        using var ex = OpenExecutor(Fixture(indexerTopK: indexerTopK));
        var engine = ex.Engine;
        Assert.True(engine.SupportsBatchedVerify);
        int[] prompts = { 30, 21, 37 };
        int[][] windows = { new[] { 17, 203, 99, 4 }, new[] { 150, 61, 5, 77 }, new[] { 123, 9, 31, 200 } };
        int[] keep = { 4, 2, 1 };
        int n = prompts.Length, w = 4, h = engine.HiddenWidth;
        var batched = new int[n];
        var serial = new int[n];
        for (int i = 0; i < n; i++)
        {
            batched[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
            serial[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
        }
        var tokens = windows.SelectMany(x => x).ToArray();
        var drawn = new int[n * w];
        var hidden = new float[n * w * h];
        Assert.True(engine.ForwardBatchedVerify(batched, w, tokens, drawn, hidden));
        Assert.True(engine.CommitBatchedVerify(batched, keep));
        for (int i = 0; i < n; i++)
        {
            Assert.True(ex.SetActiveSlot(serial[i]));
            var wantDrawn = new int[w];
            var wantHidden = new float[w * h];
            engine.ForwardVerifyGreedy(windows[i], wantDrawn, wantHidden);
            Assert.True(engine.CommitVerify(keep[i]));
            Assert.Equal(wantDrawn, drawn.AsSpan(i * w, w).ToArray());
            double hErr = RelativeError(hidden.AsSpan(i * w * h, w * h).ToArray(), wantHidden);
            _output.WriteLine($"sequence {i}: hidden streams relative error {hErr:E2}");
            Assert.True(hErr <= 1e-5, $"sequence {i}: the batched verify's hidden streams differ by {hErr:E2}");
            float[] want = Step(ex, new[] { Steps[i] });
            Assert.True(ex.SetActiveSlot(batched[i]));
            Assert.Equal(prompts[i] + keep[i], engine.NPast);
            float[] got = Step(ex, new[] { Steps[i] });
            double err = RelativeError(got, want);
            _output.WriteLine($"sequence {i}: next step after keeping {keep[i]} rows, relative error {err:E2}");
            Assert.True(err <= 1e-5, $"sequence {i}: the step after the batched commit differs by {err:E2}");
        }
    }

    /// <summary>A batched verify over the resident KV window (TS_Q4E_KV_RESIDENT, a window of one call's floor): every
    /// sequence's QSA rows read their own slot's pages, fetched over the bus as the selections move, and each sequence's
    /// rows and next step are its own verify's.</summary>
    [CudaFact]
    public void BatchedVerify_UnderAResidentKvWindow_ComputesEverySequencesOwnWindow()
    {
        _env.Set("TS_Q4E_KV_RESIDENT", "1");
        // The window streams Q8_0 K/V; prompts in 128-row ubatches, each staged from the host copy.
        using var ex = new Qwen4ExpCudaExecutor(Fixture(indexerTopK: 16), 1024, 128, 1, kvQ8: true);
        var engine = ex.Engine;
        Assert.True(engine.SupportsBatchedVerify);
        Assert.True(engine.KvWindowCells > 0, "the KV window is off");
        int[] lengths = { 600, 420, 520 };
        int[][] windows = { new[] { 17, 203, 99, 4 }, new[] { 150, 61, 5, 77 }, new[] { 123, 9, 31, 200 } };
        int[] keep = { 2, 4, 1 };
        int n = lengths.Length, w = 4, h = engine.HiddenWidth;
        var batched = new int[n];
        var serial = new int[n];
        for (int i = 0; i < n; i++)
        {
            int seed = i * 13;
            int[] prompt = Enumerable.Range(0, lengths[i]).Select(t => (t * 37 + 11 + seed) % 250).ToArray();
            batched[i] = NewActiveSlot(ex);
            Step(ex, prompt);
            serial[i] = NewActiveSlot(ex);
            Step(ex, prompt);
        }
        var drawn = new int[n * w];
        var hidden = new float[n * w * h];
        // The batched slots' streamed layers, summed (the counters read the active slot).
        (long Misses, long Overflows) Counters()
        {
            long misses = 0, overflows = 0;
            foreach (int id in batched)
            {
                Assert.True(ex.SetActiveSlot(id));
                var c = engine.KvStreamCounters();
                misses += c.Misses;
                overflows += c.Overflows;
            }
            return (misses, overflows);
        }
        var before = Counters();
        Assert.True(engine.ForwardBatchedVerify(batched, w, windows.SelectMany(x => x).ToArray(), drawn, hidden));
        var after = Counters();
        _output.WriteLine($"window {engine.KvWindowCells} cells; pages fetched {before.Misses} before the batched verify, "
            + $"{after.Misses - before.Misses} by it; {after.Overflows} overflowing resolves");
        Assert.True(after.Misses > before.Misses, "the batched verify fetched no page over the bus");
        Assert.Equal(0, after.Overflows);
        Assert.True(engine.CommitBatchedVerify(batched, keep));
        for (int i = 0; i < n; i++)
        {
            Assert.True(ex.SetActiveSlot(serial[i]));
            var wantDrawn = new int[w];
            var wantHidden = new float[w * h];
            engine.ForwardVerifyGreedy(windows[i], wantDrawn, wantHidden);
            Assert.True(engine.CommitVerify(keep[i]));
            Assert.Equal(wantDrawn, drawn.AsSpan(i * w, w).ToArray());
            double hErr = RelativeError(hidden.AsSpan(i * w * h, w * h).ToArray(), wantHidden);
            _output.WriteLine($"sequence {i}: hidden streams relative error {hErr:E2}");
            Assert.True(hErr <= 1e-5, $"sequence {i}: the batched verify's hidden streams differ by {hErr:E2}");
            float[] want = Step(ex, new[] { Steps[i] });
            Assert.True(ex.SetActiveSlot(batched[i]));
            Assert.Equal(lengths[i] + keep[i], engine.NPast);
            double err = RelativeError(Step(ex, new[] { Steps[i] }), want);
            _output.WriteLine($"sequence {i}: next step after keeping {keep[i]} rows, relative error {err:E2}");
            Assert.True(err <= 1e-5, $"sequence {i}: the step after the batched commit differs by {err:E2}");
        }
    }

    /// <summary>Four sequences' 4-row windows are a 16-row batched verify. Under expert residency with
    /// TS_Q4E_HOST_EXPERT_ROWS=16 the CPU computes its missed experts inside the step (as it does for a solo window)
    /// instead of the step staging them over the bus; each sequence's rows are still its own verify's, and the commit
    /// leaves each sequence where its own commit would.</summary>
    [CudaFact]
    public void BatchedVerify_SixteenRowsWithHostComputedMisses_AreEachSequencesOwnWindow()
    {
        _env.Set("TS_Q4E_EXPERT_VRAM_GB", "0");
        _env.Set("TS_Q4E_HOST_EXPERT_ROWS", "16");
        using var ex = OpenExecutor(Fixture(indexerTopK: 2048, q2kxlExperts: true));
        var engine = ex.Engine;
        Assert.True(engine.SupportsBatchedVerify);
        Assert.Equal(0, engine.ResidentExperts);
        int[] prompts = { 30, 21, 37, 26 };
        int[][] windows = { new[] { 17, 203, 99, 4 }, new[] { 150, 61, 5, 77 }, new[] { 123, 9, 31, 200 }, new[] { 8, 64, 180, 2 } };
        int[] keep = { 4, 2, 1, 3 };
        int n = prompts.Length, w = 4, h = engine.HiddenWidth;
        var batched = new int[n];
        var serial = new int[n];
        for (int i = 0; i < n; i++)
        {
            batched[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
            serial[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
        }
        var drawn = new int[n * w];
        var hidden = new float[n * w * h];
        long before = engine.HostExpertMisses;
        Assert.True(engine.ForwardBatchedVerify(batched, w, windows.SelectMany(x => x).ToArray(), drawn, hidden));
        long cpu = engine.HostExpertMisses - before;
        _output.WriteLine($"16-row batched verify: {cpu} experts computed by the CPU");
        Assert.True(cpu > 0, "the 16-row step left no expert to the CPU: it staged them instead");
        Assert.True(engine.CommitBatchedVerify(batched, keep));
        for (int i = 0; i < n; i++)
        {
            Assert.True(ex.SetActiveSlot(serial[i]));
            var wantDrawn = new int[w];
            var wantHidden = new float[w * h];
            engine.ForwardVerifyGreedy(windows[i], wantDrawn, wantHidden);
            Assert.True(engine.CommitVerify(keep[i]));
            Assert.Equal(wantDrawn, drawn.AsSpan(i * w, w).ToArray());
            double hErr = RelativeError(hidden.AsSpan(i * w * h, w * h).ToArray(), wantHidden);
            _output.WriteLine($"sequence {i}: hidden streams relative error {hErr:E2}");
            Assert.True(hErr <= 1e-5, $"sequence {i}: the batched verify's hidden streams differ by {hErr:E2}");
            float[] want = Step(ex, new[] { Steps[i] });
            Assert.True(ex.SetActiveSlot(batched[i]));
            Assert.Equal(prompts[i] + keep[i], engine.NPast);
            double err = RelativeError(Step(ex, new[] { Steps[i] }), want);
            _output.WriteLine($"sequence {i}: next step after keeping {keep[i]} rows, relative error {err:E2}");
            Assert.True(err <= 1e-5, $"sequence {i}: the step after the batched commit differs by {err:E2}");
        }
    }

    /// <summary>The batched verify for sequences that sample returns every row's logits: each sequence's rows as its own
    /// verify window's.</summary>
    [CudaFact]
    public void BatchedVerifyLogits_AreEachSequencesOwnVerifyLogits()
    {
        using var ex = OpenExecutor(Fixture(indexerTopK: 16));
        var engine = ex.Engine;
        int[] prompts = { 26, 33 };
        int[][] windows = { new[] { 17, 203, 99, 4 }, new[] { 150, 61, 5, 77 } };
        int n = prompts.Length, w = 4, vocab = ex.VocabSize;
        var batched = new int[n];
        var serial = new int[n];
        for (int i = 0; i < n; i++)
        {
            batched[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
            serial[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
        }
        var logits = new float[n * w * vocab];
        Assert.True(engine.ForwardBatchedVerify(batched, w, windows.SelectMany(x => x).ToArray(), null, null, logits));
        Assert.True(engine.CommitBatchedVerify(batched, new[] { 3, 1 }));
        for (int i = 0; i < n; i++)
        {
            Assert.True(ex.SetActiveSlot(serial[i]));
            var want = new float[w * vocab];
            engine.ForwardVerify(windows[i], want, null);
            for (int r = 0; r < w; r++)
            {
                double err = RelativeError(logits.AsSpan((i * w + r) * vocab, vocab).ToArray(), want.AsSpan(r * vocab, vocab).ToArray());
                Assert.True(err <= 1e-5, $"sequence {i} row {r}: the batched verify's logits differ by {err:E2}");
            }
        }
    }

    /// <summary>Batched steps run through captured graphs (Q4eCudaEngine.BatchGraphs.cs), a key captured on its third run:
    /// three batched verifies with their commits and four batched decode steps (the third captured, the fourth replaying it
    /// at the next positions) answer bit for bit as the same steps uncaptured (TS_Q4E_BATCH_GRAPHS=0). Dense rows (top_k
    /// 2048) and QSA rows past the width (top_k 16).</summary>
    [CudaTheory]
    [InlineData(2048)]
    [InlineData(16)]
    public void BatchedSteps_CapturedAnswerBitForBitAsUncaptured(int indexerTopK)
    {
        string path = Fixture(indexerTopK: indexerTopK);
        int[] prompts = { 30, 21, 37 };
        int[][] windows = { new[] { 17, 203, 99, 4 }, new[] { 150, 61, 5, 77 }, new[] { 123, 9, 31, 200 } };
        int[] keep = { 4, 2, 1 };
        (List<int[]> Drawn, List<float[]> Rows, int Graphs) Run(string graphs)
        {
            _env.Set("TS_Q4E_BATCH_GRAPHS", graphs);
            using var ex = OpenExecutor(path);
            var engine = ex.Engine;
            int n = prompts.Length, w = 4, h = engine.HiddenWidth, vocab = ex.VocabSize;
            var slots = new int[n];
            for (int i = 0; i < n; i++)
            {
                slots[i] = NewActiveSlot(ex);
                Step(ex, Prompt[..prompts[i]]);
            }
            var drawnAll = new List<int[]>();
            var rows = new List<float[]>();
            for (int round = 0; round < 3; round++)
            {
                var drawn = new int[n * w];
                var hidden = new float[n * w * h];
                Assert.True(engine.ForwardBatchedVerify(slots, w, windows.SelectMany(x => x).ToArray(), drawn, hidden));
                Assert.True(engine.CommitBatchedVerify(slots, keep));
                drawnAll.Add(drawn);
                rows.Add(hidden);
            }
            var tokens = new int[n];
            var positions = new int[n];
            for (int i = 0; i < n; i++)
                positions[i] = prompts[i] + 3 * keep[i];
            for (int step = 0; step < 4; step++)
            {
                for (int i = 0; i < n; i++)
                    tokens[i] = Steps[(i + step) % Steps.Length];
                var logits = new float[n * vocab];
                Assert.True(engine.ForwardBatchedDecode(slots, tokens, positions, logits));
                rows.Add(logits);
                for (int i = 0; i < n; i++)
                    positions[i]++;
            }
            return (drawnAll, rows, engine.CapturedBatchGraphs);
        }
        var plain = Run("0");
        var captured = Run(null);
        _output.WriteLine($"batched graphs: uncaptured {plain.Graphs}, captured {captured.Graphs}");
        Assert.Equal(0, plain.Graphs);
        Assert.Equal(2, captured.Graphs);
        for (int i = 0; i < plain.Drawn.Count; i++)
            Assert.Equal(plain.Drawn[i], captured.Drawn[i]);
        for (int i = 0; i < plain.Rows.Count; i++)
            Assert.True(captured.Rows[i].AsSpan().SequenceEqual(plain.Rows[i]),
                i < 3 ? $"batched verify {i}: capturing changed the hidden streams" : $"batched decode step {i - 3}: capturing changed the logits");
    }

    /// <summary>The engine frees the slots its owner no longer holds (Q4eCudaEngine.SlotFreeAllExcept): every slot but the
    /// active one and those kept, never a kept or the selected one; nothing more on a second call.</summary>
    [CudaFact]
    public void SlotFreeAllExcept_FreesOnlyTheSlotsNothingHolds()
    {
        using var ex = OpenExecutor(Fixture());
        int first = ex.ActiveSlot;
        int a = NewActiveSlot(ex);
        Step(ex, Prompt[..8]);
        int lost = NewActiveSlot(ex);
        Step(ex, Prompt[..8]);
        int active = NewActiveSlot(ex);
        Step(ex, Prompt[..8]);
        Assert.Equal(1, ex.SlotFreeAllExcept(new[] { first, a }));
        Assert.True(ex.SlotHead(first, out _));
        Assert.True(ex.SlotHead(a, out _));
        Assert.True(ex.SlotHead(active, out _));
        Assert.False(ex.SlotHead(lost, out _));
        Assert.Equal(active, ex.ActiveSlot);
        Assert.Equal(0, ex.SlotFreeAllExcept(new[] { first, a }));
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    private static double RelativeError(float[] actual, float[] expected)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
