// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A MiniMax-H3 LoRA plug-in applied unmerged is the same denoiser as the checkpoint with the
// update merged into its weights. Checked on a tiny transformer (MiniMaxH3LoraFixtures: an
// 8-wide residual stream, 2 heads of 8, ffn 16, two trunk blocks and one token-refiner block)
// over one full step of the runtime's denoiser (MiniMaxH3Denoiser: the step's graph program) with
// text, a conditioning keyframe, audio and video in the sequence:
//
// - the managed tensor executor with the plug-in against the same executor on a GGUF with
//   strength * alpha / rank * B * A added to its weights in the checkpoint's row order (fc1's
//   halves swapped from diffusers' [value; gate]), for every target and for the gate-only,
//   value-only, q-only and refiner-only updates, with the unswapped merge as the negative
//   control;
// - the GGML graph executor on the process's pinned GGML backend (TS_TEST_GGML_BACKEND; GgmlCpu
//   by default) with the packed factors against the managed executor with the exact ones, after
//   measuring the two executors' disagreement without a plug-in;
// - strength 0 (every term kept with zero up factors) bit-identical to no plug-in;
// - the MLX tensor executor against the managed one;
// - the refusals of a set packed for the other executor, or disposed.
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Models.MiniMaxH3;
using Xunit.Abstractions;
using Adapter = InferenceWeb.Tests.MiniMaxH3LoraFixtures.Adapter;

namespace InferenceWeb.Tests;

/// <summary>One denoiser step of the tiny fixture transformer: 5 text tokens, one conditioning
/// keyframe (4 tokens), 6 audio tokens and 8 video tokens, every segment kind the trunk
/// modulates. Inputs are deterministic, so two runs see identical buffers.</summary>
internal static class TinyH3Step
{
    internal static readonly MiniMaxH3Shape Shape = new()
    {
        Width = 64, Height = 64, Frames = 6, Fps = 24,
        LatentWidth = 4, LatentHeight = 4, LatentFrames = 2, AudioLatentFrames = 3,
    };

    internal const int TextTokens = 5, ConditionTokens = 4;

    private static float[] Wave(int count, float step, float phase, Func<float, float> f) =>
        Enumerable.Range(0, count).Select(i => f(i * step + phase)).ToArray();

    internal static float[] Text => Wave(TextTokens * MiniMaxH3LoraFixtures.TextDim, .31f, 0f, MathF.Sin);
    internal static float[] Video => Wave(Shape.VideoTokenCount * MiniMaxH3LoraFixtures.VideoPatchDim, .17f, 0f, MathF.Sin);
    internal static float[] Audio => Wave(Shape.AudioTokenCount * MiniMaxH3LoraFixtures.AudioPatchDim, .23f, 0f, MathF.Cos);
    internal static float[] Condition => Wave(ConditionTokens * MiniMaxH3LoraFixtures.VideoPatchDim, .29f, 1f, MathF.Sin);

    internal static MiniMaxH3PackedLayout Layout(float sigma) =>
        MiniMaxH3Layout.Build(TextTokens, Shape, sigma, conditionFrames: new[] { 1 }, keyframeAtEnd: new[] { false });

    /// <summary>The step's video velocities followed by its audio velocities.</summary>
    internal static float[] Run(MiniMaxH3Denoiser dit, MiniMaxH3LoraSet lora, float sigma = .75f)
    {
        var (video, audio) = dit.Forward(lora, Video, Shape.VideoTokenCount, Audio, Shape.AudioTokenCount,
            Text, TextTokens, Layout(sigma), sigma, Condition, ConditionTokens);
        return video.Concat(audio).ToArray();
    }

    /// <summary>The distinct factor pointers of a packed set.</summary>
    internal static HashSet<IntPtr> FactorPointers(MiniMaxH3LoraSet set) => set.FactorPointers().ToHashSet();

    /// <summary>max |a - b| over max |b|.</summary>
    internal static double Rel(float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0, scale = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs((double)actual[i] - expected[i]));
            scale = Math.Max(scale, Math.Abs((double)expected[i]));
        }
        return worst / Math.Max(scale, 1e-30);
    }

    internal static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return dot / Math.Sqrt(na * nb);
    }

    internal static float[] Minus(float[] a, float[] b) => a.Select((v, i) => v - b[i]).ToArray();

    internal static void AssertBits(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }
}

public sealed class MiniMaxH3LoraParityTests : IDisposable
{
    private const int Rank = 4;
    // The manifest's alpha 8 at rank 4, strength 1.
    private const float Scale = 8f / Rank;
    private const int Ffn = MiniMaxH3LoraFixtures.Ffn, Hidden = MiniMaxH3LoraFixtures.Hidden;

    private readonly ITestOutputHelper _out;
    private readonly MiniMaxH3LoraFixtures _fx = new();
    private readonly string _base;

    public MiniMaxH3LoraParityTests(ITestOutputHelper output)
    {
        _out = output;
        _base = _fx.WriteTransformer();
    }

    public void Dispose() => _fx.Dispose();

    public static TheoryData<string> Cases => new() { "all", "fc1-gate", "fc1-value", "q", "refiner" };

    /// <summary>
    /// The update a case adds, as F32 factors (the merge then sees exactly what the loader reads).
    /// "fc1-gate" and "fc1-value" update one half of fc1 only: diffusers' ff.net.0.proj rows are
    /// [value; gate], so the gate-only update lives in its lower rows and lands in fc1's first
    /// half after the swap.
    /// </summary>
    private static Adapter Build(string id)
    {
        var adapter = new Adapter();
        switch (id)
        {
            case "all":
                adapter.AllTargets(blocks: 2, refinerBlocks: 1, rank: Rank, seed: 11, dtype: "F32");
                break;
            case "fc1-gate":
            case "fc1-value":
            {
                float[] a = MiniMaxH3LoraFixtures.Rand(Rank * Hidden, 21, .5f);
                float[] b = MiniMaxH3LoraFixtures.Rand(2 * Ffn * Rank, 22, .5f);
                // Zero the other half's rows of lora_B [2 ffn, rank]: [0, ffn) is value, [ffn, 2 ffn) gate.
                int zeroFrom = id == "fc1-gate" ? 0 : Ffn;
                Array.Clear(b, zeroFrom * Rank, Ffn * Rank);
                adapter.Term(false, 1, MiniMaxH3LoraTarget.Fc1, Rank, a, b);
                break;
            }
            case "q":
                adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, Rank, seed: 31, dtype: "F32");
                break;
            case "refiner":
                adapter.AllTargets(blocks: 0, refinerBlocks: 1, rank: Rank, seed: 41, dtype: "F32");
                break;
            default:
                throw new ArgumentException(id);
        }
        return adapter;
    }

    private string Plugin(Adapter adapter, string manifest = "turbo-8step.json")
    {
        _fx.WriteAdapter(adapter);
        return _fx.WriteManifest(manifest);
    }

    private static MiniMaxH3LoraSet LoadSet(LoraSpec spec, string transformer, BackendType backend, bool tensorExecutor) =>
        MiniMaxH3LoraSet.Load(new[] { spec }, MiniMaxH3LoraFixtures.ConfigOf(transformer),
            MiniMaxH3Partition.FirstLastFrame, backend, tensorExecutor);

    /// <summary>The denoiser on the managed tensor executor (the pure-C# cpu backend).</summary>
    private static MiniMaxH3Denoiser Managed(string path) => new(path, BackendType.Cpu);

    /// <summary>The denoiser on the GGML graph executor of the process's pinned backend.</summary>
    private static MiniMaxH3Denoiser Graph(string path) => new(path, TestGates.PinnedGgmlBackend);

    // ---- the managed executor against a merged checkpoint -----------------------------------

    [Theory]
    [MemberData(nameof(Cases))]
    public void ManagedLora_EqualsTheMergedCheckpoint(string id)
    {
        var adapter = Build(id);
        string merged = _fx.WriteTransformer(deltas: adapter.MergedDeltas(Scale), fileName: "merged.gguf");
        using var set = LoadSet(new LoraSpec(Plugin(adapter)), _base, BackendType.Cpu, tensorExecutor: true);
        Assert.Equal(new[] { Scale }, set.Scales);
        using var dit = Managed(_base);
        using var mergedDit = Managed(merged);

        float[] baseline = TinyH3Step.Run(dit, null);
        float[] unmerged = TinyH3Step.Run(dit, set);
        float[] expected = TinyH3Step.Run(mergedDit, null);
        double rel = TinyH3Step.Rel(unmerged, expected), moved = TinyH3Step.Rel(baseline, expected);
        _out.WriteLine($"[h3-lora-parity] managed {id}: unmerged vs merged rel {rel:E3}; the update moves the step by rel {moved:E3}");
        Assert.True(moved > 1e-3, $"{id}: the update moved the step by only rel {moved:E3}; the case proves nothing.");
        Assert.True(rel <= 1e-5, $"{id}: unmerged vs merged rel {rel:E3} > 1e-5");
    }

    [Fact]
    public void ManagedLora_DiffersFromAMergeThatKeepsDiffusersFc1Order()
    {
        // fc1 updates with both halves populated in every block: applying them without the
        // [value; gate] -> [gate | value] swap is a different model, by far more than the tolerance.
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Fc1, Rank, seed: 51, dtype: "F32");
        adapter.Term(false, 1, MiniMaxH3LoraTarget.Fc1, Rank, seed: 52, dtype: "F32");
        adapter.Term(true, 0, MiniMaxH3LoraTarget.Fc1, Rank, seed: 53, dtype: "F32");
        string swapped = _fx.WriteTransformer(deltas: adapter.MergedDeltas(Scale), fileName: "swapped.gguf");
        string unswapped = _fx.WriteTransformer(deltas: adapter.MergedDeltas(Scale, swapFc1: false), fileName: "unswapped.gguf");
        using var set = LoadSet(new LoraSpec(Plugin(adapter)), _base, BackendType.Cpu, tensorExecutor: true);
        using var dit = Managed(_base);
        using var swappedDit = Managed(swapped);
        using var unswappedDit = Managed(unswapped);

        float[] unmerged = TinyH3Step.Run(dit, set);
        double right = TinyH3Step.Rel(unmerged, TinyH3Step.Run(swappedDit, null));
        double wrong = TinyH3Step.Rel(unmerged, TinyH3Step.Run(unswappedDit, null));
        _out.WriteLine($"[h3-lora-parity] fc1 order: swapped merge rel {right:E3}, unswapped merge rel {wrong:E3}");
        Assert.True(right <= 1e-5, $"swapped merge rel {right:E3}");
        Assert.True(wrong > 1e-2, $"the unswapped merge is only rel {wrong:E3} away; the control cannot tell the orders apart");
    }

    // ---- the MLX tensor executor ---------------------------------------------------------------

    /// <summary>The MLX tensor executor applies the plug-in as the managed one does.</summary>
    [MlxFact]
    public void MlxLora_MatchesManaged()
    {
        var adapter = Build("all");
        using var set = LoadSet(new LoraSpec(Plugin(adapter)), _base, BackendType.Mlx, tensorExecutor: true);
        using var cpu = Managed(_base);
        float[] cpuBase = TinyH3Step.Run(cpu, null), cpuLora = TinyH3Step.Run(cpu, set);

        using var allocator = new MlxAllocator();
        var (mlxLora, mlxBase) = MlxWorker.Shared.Invoke(() =>
        {
            using var dit = new MiniMaxH3Denoiser(_base, BackendType.Mlx, allocator);
            float[] withLora = TinyH3Step.Run(dit, set);
            float[] without = TinyH3Step.Run(dit, null);
            return (withLora, without);
        });

        double rel = TinyH3Step.Rel(mlxLora, cpuLora), baseRel = TinyH3Step.Rel(mlxBase, cpuBase);
        double moved = TinyH3Step.Rel(cpuLora, cpuBase);
        double cosine = TinyH3Step.Cosine(TinyH3Step.Minus(mlxLora, mlxBase), TinyH3Step.Minus(cpuLora, cpuBase));
        // MLX and the CPU disagree by ~1e-3 here without any plug-in, so the bound is the device one the GGML lanes use,
        // never tighter than that floor allows; the update's direction is the sharp check.
        double bound = Math.Max(2e-3, 2.5 * baseRel);
        _out.WriteLine($"[h3-lora-parity] mlx all: vs managed rel {rel:E3} (no plug-in floor {baseRel:E3}, bound {bound:E3}), " +
            $"the update moves the step by rel {moved:E3}, update cosine {cosine:F7}");
        Assert.True(moved > 10 * bound, $"the update moved the step by only rel {moved:E3}; the case proves nothing on MLX.");
        Assert.True(rel <= bound, $"MLX vs managed rel {rel:E3} > {bound:E3}");
        Assert.True(cosine >= .9999, $"update cosine {cosine:F7} < 0.9999");
    }

    // ---- the GGML graph executor against the managed one ---------------------------------------

    /// <summary>
    /// The GGML graph executor with the packed factors against the managed executor with the exact ones. Both run the
    /// program's attention with K and V cast to F16, so their disagreement without a plug-in is the executors' product
    /// arithmetic alone; an update as large as the synthetic "all" case (it moves the step by ~38%) scales that rounding up
    /// with q/k/v. GgmlCpu is held to rel 1e-4, any device to rel 2e-3 and an update cosine of 0.9999.
    ///
    /// GgmlCuda runs its float products on cuBLAS handles ggml-cuda sets to TF32 tensor-op math, so there the default path
    /// gets a bound derived from its own floor, and the 0.9999 cosine is held under TS_H3_CUDA_F32_PRODUCTS=1, which moves
    /// the step's weight products to TensorSharp's pedantic F32 GEMM.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void GraphLoraOnThePinnedGgmlBackend_MatchesManaged(string id)
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var products = new NativeEnvScope();
        // The default path first, whatever the process environment says.
        products.Set("TS_H3_CUDA_F32_PRODUCTS", null);
        var adapter = Build(id);
        string manifest = Plugin(adapter);
        using var packed = LoadSet(new LoraSpec(manifest), _base, backend, tensorExecutor: false);
        using var exact = LoadSet(new LoraSpec(manifest), _base, BackendType.Cpu, tensorExecutor: true);
        using var graph = Graph(_base);
        using var managed = Managed(_base);

        // The two executors' disagreement without a plug-in: the floor the comparison stands on.
        float[] managedBase = TinyH3Step.Run(managed, null), managedLora = TinyH3Step.Run(managed, exact);
        double moved = TinyH3Step.Rel(managedLora, managedBase);
        (double floor, double rel, double cosine) Compare(string label)
        {
            float[] graphBase = TinyH3Step.Run(graph, null), graphLora = TinyH3Step.Run(graph, packed);
            double f = TinyH3Step.Rel(graphBase, managedBase);
            double r = TinyH3Step.Rel(graphLora, managedLora);
            double c = TinyH3Step.Cosine(TinyH3Step.Minus(graphLora, graphBase), TinyH3Step.Minus(managedLora, managedBase));
            _out.WriteLine($"[h3-lora-parity] {backend}{label} vs managed {id}: no-plug-in floor rel {f:E3}; " +
                $"with the plug-in rel {r:E3}; update cosine {c:F7}; the update moves the step by rel {moved:E3}");
            return (f, r, c);
        }
        var (floor, rel, cosine) = Compare("");

        Assert.True(moved > 1e-3, $"{id}: the update moved the step by only rel {moved:E3}");
        if (backend == BackendType.GgmlCuda)
        {
            // The default (TF32) path: the update, a difference of two noisy steps, is off by up to about floor / moved in
            // direction, and the F16 factors add (2 u)^2 = 2^-20 of their own (see the H3-Image LoRA tests).
            double bound = Math.Max((floor / moved) * (floor / moved), Math.Pow(2, -20));
            Assert.True(rel <= 2e-3, $"{id}: {backend} (TF32) vs managed rel {rel:E3} > 2e-3 (no-plug-in floor {floor:E3})");
            Assert.True(1 - cosine <= bound, $"{id}: {backend} (TF32) update 1 - cosine {1 - cosine:E3} > {bound:E3}");
            products.Set("TS_H3_CUDA_F32_PRODUCTS", "1");
            (floor, rel, cosine) = Compare(", TS_H3_CUDA_F32_PRODUCTS=1");
        }
        if (backend == BackendType.GgmlCpu)
        {
            Assert.True(rel <= 1e-4, $"{id}: GgmlCpu vs managed rel {rel:E3} > 1e-4 (no-plug-in floor {floor:E3})");
        }
        else
        {
            Assert.True(rel <= 2e-3, $"{id}: {backend} vs managed rel {rel:E3} > 2e-3 (no-plug-in floor {floor:E3})");
            Assert.True(cosine >= 0.9999, $"{id}: {backend} update cosine {cosine:F7} < 0.9999");
        }
    }

    // ---- strength 0 ----------------------------------------------------------------------------

    [Fact]
    public void StrengthZero_IsBitIdenticalToNoPlugIn_OnBothExecutors()
    {
        var adapter = Build("all");
        var spec = new LoraSpec(Plugin(adapter), 0f);
        using var packed = LoadSet(spec, _base, TestGates.PinnedGgmlBackend, tensorExecutor: false);
        using var exact = LoadSet(spec, _base, BackendType.Cpu, tensorExecutor: true);
        // Every term is kept, with zero up factors: the identity probe exercises the whole wiring.
        Assert.Equal(adapter.Targets.Count(), packed.TermCount);
        Assert.Equal(new[] { 0f }, packed.Scales);
        using var graph = Graph(_base);
        using var managed = Managed(_base);

        TinyH3Step.AssertBits(TinyH3Step.Run(graph, null), TinyH3Step.Run(graph, packed), $"{TestGates.PinnedGgmlBackend} strength 0");
        TinyH3Step.AssertBits(TinyH3Step.Run(managed, null), TinyH3Step.Run(managed, exact), "managed strength 0");
    }

    /// <summary>A plug-in switched off, on and off again: each step binds the set it is given (its plan is recompiled for
    /// it), so "on" never runs with the plain checkpoint's plan nor "off" with the plug-in's.</summary>
    [Fact]
    public void ThePlanFollowsThePlugIn_OffOnOff()
    {
        using var set = LoadSet(new LoraSpec(Plugin(Build("refiner"))), _base, BackendType.Cpu, tensorExecutor: true);
        using var dit = Managed(_base);
        float[] off = TinyH3Step.Run(dit, null);
        float[] on = TinyH3Step.Run(dit, set);
        float[] offAgain = TinyH3Step.Run(dit, null);
        float[] onAgain = TinyH3Step.Run(dit, set);
        Assert.True(TinyH3Step.Rel(on, off) > 1e-3, "the refiner-only plug-in did not change the step");
        TinyH3Step.AssertBits(off, offAgain, "off after on");
        TinyH3Step.AssertBits(on, onAgain, "on after off");

        using var fresh = Managed(_base);
        TinyH3Step.AssertBits(on, TinyH3Step.Run(fresh, set), "on, a fresh denoiser");

        // A residency release drops the plan; the next step compiles it again with the same factors.
        dit.ReleaseDevice();
        TinyH3Step.AssertBits(on, TinyH3Step.Run(dit, set), "on after a residency release");
    }

    // ---- refusals ----------------------------------------------------------------------------

    [Fact]
    public void ASetPackedForTheOtherExecutor_OrDisposed_IsRefused()
    {
        string manifest = Plugin(Build("q"));
        using var packed = LoadSet(new LoraSpec(manifest), _base, TestGates.PinnedGgmlBackend, tensorExecutor: false);
        using var exact = LoadSet(new LoraSpec(manifest), _base, BackendType.Cpu, tensorExecutor: true);
        using var graph = Graph(_base);
        using var managed = Managed(_base);
        Assert.Contains("packed for a tensor executor", Assert.Throws<ArgumentException>(() => TinyH3Step.Run(graph, exact)).Message);
        Assert.Contains("packed for the GGML graph executor", Assert.Throws<ArgumentException>(() => TinyH3Step.Run(managed, packed)).Message);

        packed.Dispose();
        Assert.Throws<ObjectDisposedException>(() => TinyH3Step.Run(graph, packed));
        exact.Dispose();
        Assert.Throws<ObjectDisposedException>(() => TinyH3Step.Run(managed, exact));
    }
}
