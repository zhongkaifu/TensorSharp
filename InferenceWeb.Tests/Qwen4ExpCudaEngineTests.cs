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

    public Qwen4ExpCudaEngineTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", "1024");
        _env.ClearSpeculationVars();
    }

    public void Dispose()
    {
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
                Qwen4ExpSyntheticModelBuilder.Hidden, ff);
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
