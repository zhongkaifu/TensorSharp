// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A MiniMax-H3 LoRA plug-in applied unmerged is the same denoiser as the checkpoint with the
// update merged into its weights. Checked on a tiny transformer (MiniMaxH3LoraFixtures: an
// 8-wide residual stream, 2 heads of 8, ffn 16, two trunk blocks and one token-refiner block)
// over one full DiT step with text, a conditioning keyframe, audio and video in the sequence:
//
// - the direct CPU path with the plug-in against the direct CPU path on a GGUF with
//   strength * alpha / rank * B * A added to its weights in the checkpoint's row order (fc1's
//   halves swapped from diffusers' [value; gate]), for every target and for the gate-only,
//   value-only, q-only and refiner-only updates, with the unswapped merge as the negative
//   control;
// - the native graph on the process's pinned GGML backend (TS_TEST_GGML_BACKEND; GgmlCpu by
//   default) with the packed factors against the direct CPU path, on both native attention
//   paths, after measuring the two paths' disagreement without a plug-in;
// - strength 0 (every term kept with zero up factors) bit-identical to no plug-in;
// - the direct path's refined-text cache across a plug-in switched off, on and off;
// - the direct MLX path against the direct CPU path, its factors held once, on the device;
// - the native refusals of a corrupted adapter.
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Models.MiniMaxH3;
using Xunit.Abstractions;
using Adapter = InferenceWeb.Tests.MiniMaxH3LoraFixtures.Adapter;

namespace InferenceWeb.Tests;

/// <summary>One DiT step of the tiny fixture transformer: 5 text tokens, one conditioning
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
    internal static float[] Run(MiniMaxH3DiT dit, MiniMaxH3LoraSet lora, float sigma = .75f)
    {
        var (video, audio) = dit.Forward(lora, Video, Shape.VideoTokenCount, Audio, Shape.AudioTokenCount,
            Text, TextTokens, Layout(sigma), sigma, Condition, ConditionTokens);
        return video.Concat(audio).ToArray();
    }

    /// <summary>The distinct factor pointers of a packed set, read from its native descriptors.</summary>
    internal static HashSet<IntPtr> FactorPointers(MiniMaxH3LoraSet set)
    {
        var pointers = new HashSet<IntPtr>();
        var adapter = Marshal.PtrToStructure<H3DitAdapter>(set.NativeAdapter);
        void Collect(IntPtr blocks, int count)
        {
            for (int b = 0; b < count; b++)
            {
                var block = Marshal.PtrToStructure<H3BlockLora>(blocks + b * Marshal.SizeOf<H3BlockLora>());
                foreach (var l in new[] { block.Q, block.K, block.V, block.Out, block.Fc1, block.Fc2 })
                {
                    if (l.Rank == 0) continue;
                    pointers.Add(l.Down);
                    pointers.Add(l.Up);
                }
            }
        }
        Collect(adapter.Blocks, adapter.NumBlocks);
        Collect(adapter.Refiner, adapter.NumRefinerBlocks);
        return pointers;
    }

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
    private readonly CpuAllocator _allocator = new(BlasEnum.DotNet);
    private readonly EnvScope _env = new();
    private readonly string _base;

    public MiniMaxH3LoraParityTests(ITestOutputHelper output)
    {
        _out = output;
        _env.Set("TS_H3_DIT_LAYERS", null);
        _base = _fx.WriteTransformer();
    }

    public void Dispose()
    {
        _env.Dispose();
        _fx.Dispose();
    }

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

    private static MiniMaxH3LoraSet LoadSet(LoraSpec spec, string transformer, BackendType backend, bool direct) =>
        MiniMaxH3LoraSet.Load(new[] { spec }, MiniMaxH3LoraFixtures.ConfigOf(transformer),
            MiniMaxH3Partition.FirstLastFrame, backend, direct);

    private MiniMaxH3DiT Direct(string path) => new(path, BackendType.Cpu, _allocator);

    private static MiniMaxH3DiT Native(string path) => new(path, TestGates.PinnedGgmlBackend, null);

    // ---- direct CPU against a merged checkpoint --------------------------------------------

    [Theory]
    [MemberData(nameof(Cases))]
    public void DirectCpuLora_EqualsTheMergedCheckpoint(string id)
    {
        var adapter = Build(id);
        string merged = _fx.WriteTransformer(deltas: adapter.MergedDeltas(Scale), fileName: "merged.gguf");
        using var set = LoadSet(new LoraSpec(Plugin(adapter)), _base, BackendType.Cpu, direct: true);
        Assert.Equal(new[] { Scale }, set.Scales);
        using var dit = Direct(_base);
        using var mergedDit = Direct(merged);

        float[] baseline = TinyH3Step.Run(dit, null);
        float[] unmerged = TinyH3Step.Run(dit, set);
        float[] expected = TinyH3Step.Run(mergedDit, null);
        double rel = TinyH3Step.Rel(unmerged, expected), moved = TinyH3Step.Rel(baseline, expected);
        _out.WriteLine($"[h3-lora-parity] direct-cpu {id}: unmerged vs merged rel {rel:E3}; the update moves the step by rel {moved:E3}");
        Assert.True(moved > 1e-3, $"{id}: the update moved the step by only rel {moved:E3}; the case proves nothing.");
        Assert.True(rel <= 1e-5, $"{id}: unmerged vs merged rel {rel:E3} > 1e-5");
    }

    [Fact]
    public void DirectCpuLora_DiffersFromAMergeThatKeepsDiffusersFc1Order()
    {
        // fc1 updates with both halves populated in every block: applying them without the
        // [value; gate] -> [gate | value] swap is a different model, by far more than the tolerance.
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Fc1, Rank, seed: 51, dtype: "F32");
        adapter.Term(false, 1, MiniMaxH3LoraTarget.Fc1, Rank, seed: 52, dtype: "F32");
        adapter.Term(true, 0, MiniMaxH3LoraTarget.Fc1, Rank, seed: 53, dtype: "F32");
        string swapped = _fx.WriteTransformer(deltas: adapter.MergedDeltas(Scale), fileName: "swapped.gguf");
        string unswapped = _fx.WriteTransformer(deltas: adapter.MergedDeltas(Scale, swapFc1: false), fileName: "unswapped.gguf");
        using var set = LoadSet(new LoraSpec(Plugin(adapter)), _base, BackendType.Cpu, direct: true);
        using var dit = Direct(_base);
        using var swappedDit = Direct(swapped);
        using var unswappedDit = Direct(unswapped);

        float[] unmerged = TinyH3Step.Run(dit, set);
        double right = TinyH3Step.Rel(unmerged, TinyH3Step.Run(swappedDit, null));
        double wrong = TinyH3Step.Rel(unmerged, TinyH3Step.Run(unswappedDit, null));
        _out.WriteLine($"[h3-lora-parity] fc1 order: swapped merge rel {right:E3}, unswapped merge rel {wrong:E3}");
        Assert.True(right <= 1e-5, $"swapped merge rel {right:E3}");
        Assert.True(wrong > 1e-2, $"the unswapped merge is only rel {wrong:E3} away; the control cannot tell the orders apart");
    }

    // ---- the direct MLX path -----------------------------------------------------------------

    private static readonly FieldInfo DitDirect = typeof(MiniMaxH3DiT).GetField("_direct", BindingFlags.NonPublic | BindingFlags.Instance)!;
    private static readonly FieldInfo DirectLora = typeof(MiniMaxH3DirectDiT).GetField("_lora", BindingFlags.NonPublic | BindingFlags.Instance)!;
    private static readonly FieldInfo LinearWeight = typeof(DirectLinear).GetField("_wCpu", BindingFlags.NonPublic | BindingFlags.Instance)!;
    private static readonly FieldInfo MlxHostBuffer = typeof(MlxStorage).GetField("buffer", BindingFlags.NonPublic | BindingFlags.Instance)!;

    /// <summary>
    /// The MLX direct path applies the plug-in as the CPU one does, and keeps each factor once:
    /// as a device array, with no host copy beside it (a host-written F32 tensor would hold
    /// both for as long as the set is applied, doubling the 8-step adapter's 2.77 GB).
    /// </summary>
    [MlxFact]
    public void DirectMlxLora_MatchesDirectCpu_AndKeepsNoHostCopyOfItsFactors()
    {
        var adapter = Build("all");
        using var set = LoadSet(new LoraSpec(Plugin(adapter)), _base, BackendType.Mlx, direct: true);
        using var cpu = Direct(_base);
        float[] cpuBase = TinyH3Step.Run(cpu, null), cpuLora = TinyH3Step.Run(cpu, set);

        using var allocator = new MlxAllocator();
        var (mlxLora, mlxBase, factors, hostCopies) = MlxWorker.Shared.Invoke(() =>
        {
            using var dit = new MiniMaxH3DiT(_base, BackendType.Mlx, allocator);
            float[] withLora = TinyH3Step.Run(dit, set);
            // Before the plug-in-free step below, which drops the built factors.
            var linears = BuiltLoraLinears(dit);
            int host = linears.Count(l => MlxHostBuffer.GetValue(((Tensor)LinearWeight.GetValue(l)!).Storage) is IntPtr p && p != IntPtr.Zero);
            float[] without = TinyH3Step.Run(dit, null);
            return (withLora, without, linears.Count, host);
        });

        double rel = TinyH3Step.Rel(mlxLora, cpuLora), baseRel = TinyH3Step.Rel(mlxBase, cpuBase);
        double moved = TinyH3Step.Rel(cpuLora, cpuBase);
        double cosine = TinyH3Step.Cosine(TinyH3Step.Minus(mlxLora, mlxBase), TinyH3Step.Minus(cpuLora, cpuBase));
        // MLX and CPU disagree by ~1e-3 here without any plug-in (9.5e-4 measured on Apple silicon),
        // so the bound is the device one the native lanes use, never tighter than that floor
        // allows; the update's direction is the sharp check.
        double bound = Math.Max(2e-3, 2.5 * baseRel);
        _out.WriteLine($"[h3-lora-parity] direct-mlx all: vs direct cpu rel {rel:E3} (no plug-in floor {baseRel:E3}, " +
            $"bound {bound:E3}), the update moves the step by rel {moved:E3}, update cosine {cosine:F7}; " +
            $"{factors} factor linears, {hostCopies} with a host copy");
        Assert.True(factors > 0, "the MLX forward built no LoRA factors; the plug-in was not applied there.");
        Assert.Equal(0, hostCopies);
        Assert.True(moved > 10 * bound, $"the update moved the step by only rel {moved:E3}; the case proves nothing on MLX.");
        Assert.True(rel <= bound, $"direct MLX vs direct CPU rel {rel:E3} > {bound:E3}");
        Assert.True(cosine >= .9999, $"update cosine {cosine:F7} < 0.9999");
    }

    /// <summary>Every factor linear the direct DiT has built for its current plug-in.</summary>
    private static List<DirectLinear> BuiltLoraLinears(MiniMaxH3DiT dit)
    {
        object state = DirectLora.GetValue(DitDirect.GetValue(dit)!)!;
        var linears = new List<DirectLinear>();
        foreach (string arrayName in new[] { "Trunk", "Refiner" })
        {
            foreach (object? block in (Array)state.GetType().GetField(arrayName)!.GetValue(state)!)
            {
                if (block == null) continue;
                foreach (FieldInfo field in block.GetType().GetFields())
                {
                    object? value = field.GetValue(block);
                    if (value is DirectLinear linear) linears.Add(linear);
                    else if (value is DirectLinear[] many) linears.AddRange(many.Where(l => l != null));
                }
            }
        }
        return linears;
    }

    // ---- the native graph against the direct path ---------------------------------------------

    public static TheoryData<string, bool> NativeCases
    {
        get
        {
            var data = new TheoryData<string, bool>();
            foreach (string id in new[] { "all", "fc1-gate", "fc1-value", "q", "refiner" })
            {
                data.Add(id, true);
                data.Add(id, false);
            }
            return data;
        }
    }

    /// <summary>
    /// The native graph with the packed factors against the direct CPU path, on both of the
    /// native attention paths. The flash kernel takes K and V as F16, which is the whole of
    /// the two paths' disagreement without a plug-in (measured on GgmlCpu: rel 1.5e-5 with
    /// flash, 2.5e-7 with the explicit softmax, TS_H3_NO_FLASH=1), and an update as large as
    /// the synthetic "all" case (it moves the step by 38%) scales that rounding up with q/k/v.
    /// So the explicit softmax is where GgmlCpu is held to 1e-4: it isolates the LoRA wiring.
    /// The flash path, and any device, are held to rel 2e-3 and an update cosine of 0.9999.
    /// </summary>
    [Theory]
    [MemberData(nameof(NativeCases))]
    public void NativeLoraOnThePinnedGgmlBackend_MatchesDirectCpu(string id, bool flash)
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var attentionPath = new NativeEnvScope();
        attentionPath.Set("TS_H3_NO_FLASH", flash ? null : "1");
        var adapter = Build(id);
        string manifest = Plugin(adapter);
        using var packed = LoadSet(new LoraSpec(manifest), _base, backend, direct: false);
        using var exact = LoadSet(new LoraSpec(manifest), _base, BackendType.Cpu, direct: true);
        using var native = Native(_base);
        using var direct = Direct(_base);

        // The two paths' disagreement without a plug-in: the floor the comparison stands on.
        float[] nativeBase = TinyH3Step.Run(native, null), directBase = TinyH3Step.Run(direct, null);
        float[] nativeLora = TinyH3Step.Run(native, packed), directLora = TinyH3Step.Run(direct, exact);
        double floor = TinyH3Step.Rel(nativeBase, directBase);
        double rel = TinyH3Step.Rel(nativeLora, directLora);
        double cosine = TinyH3Step.Cosine(TinyH3Step.Minus(nativeLora, nativeBase), TinyH3Step.Minus(directLora, directBase));
        double moved = TinyH3Step.Rel(directLora, directBase);
        string attention = flash ? "flash attention" : "explicit softmax";
        _out.WriteLine($"[h3-lora-parity] {backend} ({attention}) vs direct-cpu {id}: no-plug-in floor rel {floor:E3}; " +
            $"with the plug-in rel {rel:E3}; update cosine {cosine:F7}; the update moves the step by rel {moved:E3}");

        Assert.True(moved > 1e-3, $"{id}: the update moved the step by only rel {moved:E3}");
        if (backend == BackendType.GgmlCpu && !flash)
        {
            Assert.True(rel <= 1e-4, $"{id}: GgmlCpu ({attention}) vs direct CPU rel {rel:E3} > 1e-4 (no-plug-in floor {floor:E3})");
        }
        else
        {
            Assert.True(rel <= 2e-3, $"{id}: {backend} ({attention}) vs direct CPU rel {rel:E3} > 2e-3 (no-plug-in floor {floor:E3})");
            Assert.True(cosine >= 0.9999, $"{id}: {backend} ({attention}) update cosine {cosine:F7} < 0.9999");
        }
    }

    // ---- strength 0 ----------------------------------------------------------------------------

    [Fact]
    public void StrengthZero_IsBitIdenticalToNoPlugIn_OnBothPaths()
    {
        var adapter = Build("all");
        var spec = new LoraSpec(Plugin(adapter), 0f);
        using var packed = LoadSet(spec, _base, TestGates.PinnedGgmlBackend, direct: false);
        using var exact = LoadSet(spec, _base, BackendType.Cpu, direct: true);
        // Every term is kept, with zero up factors: the identity probe exercises the whole wiring.
        Assert.Equal(adapter.Targets.Count(), packed.TermCount);
        Assert.Equal(new[] { 0f }, packed.Scales);
        using var native = Native(_base);
        using var direct = Direct(_base);

        TinyH3Step.AssertBits(TinyH3Step.Run(native, null), TinyH3Step.Run(native, packed), $"{TestGates.PinnedGgmlBackend} strength 0");
        TinyH3Step.AssertBits(TinyH3Step.Run(direct, null), TinyH3Step.Run(direct, exact), "direct CPU strength 0");
    }

    // ---- the direct path's refined-text cache --------------------------------------------------

    [Fact]
    public void RefinedTextCache_FollowsThePlugIn_OffOnOff()
    {
        // Refiner-only terms change the step through the refined text alone, so a cache that
        // ignored the plug-in would hand the plain tokens back and the "on" step would match "off".
        using var set = LoadSet(new LoraSpec(Plugin(Build("refiner"))), _base, BackendType.Cpu, direct: true);
        using var dit = Direct(_base);
        float[] off = TinyH3Step.Run(dit, null);
        float[] on = TinyH3Step.Run(dit, set);
        float[] offAgain = TinyH3Step.Run(dit, null);
        float[] onAgain = TinyH3Step.Run(dit, set);
        float[] onAgainSameSet = TinyH3Step.Run(dit, set, sigma: .75f);
        Assert.True(TinyH3Step.Rel(on, off) > 1e-3, "the refiner-only plug-in did not change the step");
        TinyH3Step.AssertBits(off, offAgain, "off after on");
        TinyH3Step.AssertBits(on, onAgain, "on after off");
        TinyH3Step.AssertBits(on, onAgainSameSet, "on, cached");

        using var fresh = Direct(_base);
        TinyH3Step.AssertBits(on, TinyH3Step.Run(fresh, set), "on, uncached");

        // A residency release drops the built factors and the cache; the next step rebuilds both.
        dit.ReleaseDeviceResidency();
        TinyH3Step.AssertBits(on, TinyH3Step.Run(dit, set), "on after a residency release");
    }

    // ---- refusals ----------------------------------------------------------------------------

    [Fact]
    public void NativeForward_RefusesACorruptedAdapter_WithTheNativeMessage()
    {
        using var packed = LoadSet(new LoraSpec(Plugin(Build("all"))), _base, TestGates.PinnedGgmlBackend, direct: false);
        using var native = Native(_base);
        float[] good = TinyH3Step.Run(native, packed);

        // A managed build paired with a GgmlOps of another adapter size.
        IntPtr adapter = packed.NativeAdapter;
        int structBytes = Marshal.ReadInt32(adapter);
        Assert.Equal(Marshal.SizeOf<H3DitAdapter>(), structBytes);
        Marshal.WriteInt32(adapter, 31);
        try
        {
            var e = Assert.Throws<InvalidOperationException>(() => TinyH3Step.Run(native, packed));
            Assert.Equal("MiniMax-H3 DiT forward failed: MiniMaxH3DitForward: LoRA adapter is 31 bytes, this GgmlOps expects 32; " +
                "the managed TensorSharp build and the native library differ, rebuild GgmlOps.", e.Message);
        }
        finally { Marshal.WriteInt32(adapter, structBytes); }

        // One part's factor type, named by block and diffusers module.
        var descriptor = Marshal.PtrToStructure<H3DitAdapter>(adapter);
        IntPtr vType = descriptor.Blocks + Marshal.SizeOf<H3BlockLora>()
            + (int)Marshal.OffsetOf<H3BlockLora>(nameof(H3BlockLora.V)) + (int)Marshal.OffsetOf<H3Lora>(nameof(H3Lora.Type));
        int type = Marshal.ReadInt32(vType);
        Marshal.WriteInt32(vType, 7);
        try
        {
            var e = Assert.Throws<InvalidOperationException>(() => TinyH3Step.Run(native, packed));
            Assert.Equal("MiniMax-H3 DiT forward failed: MiniMaxH3DitForward: LoRA block 1 attn.to_v: factor type 7 is " +
                "neither F32 (0) nor F16 (1).", e.Message);
        }
        finally { Marshal.WriteInt32(vType, type); }

        TinyH3Step.AssertBits(good, TinyH3Step.Run(native, packed), "after the repairs");
    }

    [Fact]
    public void ASetLoadedForTheOtherPath_OrDisposed_IsRefused()
    {
        string manifest = Plugin(Build("q"));
        using var packed = LoadSet(new LoraSpec(manifest), _base, TestGates.PinnedGgmlBackend, direct: false);
        using var exact = LoadSet(new LoraSpec(manifest), _base, BackendType.Cpu, direct: true);
        using var native = Native(_base);
        using var direct = Direct(_base);
        Assert.Contains("loaded for a direct backend", Assert.Throws<ArgumentException>(() => TinyH3Step.Run(native, exact)).Message);
        Assert.Contains("packed for the native graph", Assert.Throws<ArgumentException>(() => TinyH3Step.Run(direct, packed)).Message);

        packed.Dispose();
        Assert.Throws<ObjectDisposedException>(() => TinyH3Step.Run(native, packed));
        exact.Dispose();
        Assert.Throws<ObjectDisposedException>(() => TinyH3Step.Run(direct, exact));
    }
}
