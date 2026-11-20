// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.SharedDiffusion;
using Xunit;
using Xunit.Abstractions;
using G = InferenceWeb.Tests.MiniMaxH3ImageLoraFixture.Grammar;
using P = InferenceWeb.Tests.MiniMaxH3ImageLoraFixture.Proj;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image's denoiser with LoRA adapters applied unmerged through LoraLinear regions (docs/design/diffusion-runtime.md
/// sections 3.7 and 5.3), on every backend the denoiser runs on, against an independent merge: the same synthetic checkpoint
/// with every adapter's strength * alpha / rank * B A added to its F32 projections in the checkpoint's own layout
/// (<see cref="MiniMaxH3ImageLoraFixture.Deltas"/>), denoised without adapters. Also: strength 0 is the base denoiser bit for
/// bit, and switching adapters between requests leaves nothing of the previous set behind (no stale plan, no stale device
/// copy), the bug class of PR #282.
/// </summary>
public sealed class MiniMaxH3ImageLoraDenoiserTests(ITestOutputHelper output) : IDisposable
{
    private const int F32 = MiniMaxH3ImageFixture.F32, BF16 = MiniMaxH3ImageFixture.BF16, Q8_0 = MiniMaxH3ImageFixture.Q8_0,
        Q4_K = MiniMaxH3ImageFixture.Q4_K;

    private readonly MiniMaxH3ImageLoraFixture _fx = new();

    public void Dispose() => _fx.Dispose();

    private static float[] Random(Random rng, long n, float scale = 1f)
    {
        var v = new float[n];
        for (long i = 0; i < n; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    /// <summary>Text to image at 272 x 272 (289 image tokens, 310 rows) or, wide, 1024 x 1024 (4096 tokens: ggml_cuda's wide
    /// routes and served regions).</summary>
    private static (int H, int W, int Text) Geometry(bool wide) => wide ? (64, 64, 21) : (34, 34, 21);

    /// <summary>Two Euler steps of the speed recipe's kind from one seed's noise: each step's velocity, the state advanced
    /// with <paramref name="advanceWith"/>'s (the reference's) so both denoisers see the same inputs.</summary>
    private static List<float[]> Steps(MiniMaxH3ImageFamily family, IDenoiser denoiser, AdapterSet adapters, bool wide, int seed = 11,
        List<float[]> advanceWith = null)
    {
        var (h, w, textRows) = Geometry(wide);
        var rng = new Random(seed);
        var text = new ConditioningSet(Random(rng, (long)textRows * MiniMaxH3ImageFixture.TextDim, 2f), textRows, MiniMaxH3ImageFixture.TextDim);
        var target = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, h, w);
        float[] x = family.ToRows(new LatentTensor(target, family.Noise(seed).Next(target.Count)));
        var contract = new SamplingContract { Name = "test", Sigmas = new[] { 1f, 12f / 13f, 0f }, Heads = HeadPolicy.ByStep, MinGenerationArea = 1024 };
        SamplingRecipe recipe = contract.Apply(family.DefaultRecipe);
        float[] sigmas = recipe.SigmasFor(recipe.Steps);
        var result = new List<float[]>();
        for (int i = 0; i < recipe.Steps; i++)
        {
            var (start, stop) = recipe.HeadsFor(i, sigmas[i], sigmas[i + 1]);
            var state = new StepState(i, recipe.Steps, sigmas[i], sigmas[i + 1], start, stop);
            var inputs = new DenoiseStepInputs
            {
                Latent = x, Conditioning = text, References = Array.Empty<LatentRef>(),
                Layout = family.Pack(text, Array.Empty<LatentRef>(), target, state), Step = state, Recipe = recipe,
            };
            float[] v = (float[])denoiser.Step(inputs, adapters, CancellationToken.None).Velocity.Clone();
            result.Add(v);
            float[] advance = advanceWith?[i] ?? v;
            for (long k = 0; k < x.LongLength; k++) x[k] += advance[k] * (sigmas[i + 1] - sigmas[i]);
        }
        return result;
    }

    private static (double Rel, double RelL2) Compare(float[] want, float[] got)
    {
        double scale = want.Max(v => Math.Abs((double)v)), worst = 0, l2 = 0, n2 = 0;
        for (int k = 0; k < want.Length; k++)
        {
            Assert.True(float.IsFinite(got[k]), $"value {k} is {got[k]}");
            double d = got[k] - (double)want[k];
            worst = Math.Max(worst, Math.Abs(d));
            l2 += d * d;
            n2 += (double)want[k] * want[k];
        }
        return (worst / scale, Math.Sqrt(l2 / n2));
    }

    private static void AssertBitExact(List<float[]> want, List<float[]> got, string what)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            int differ = 0;
            for (int k = 0; k < want[i].Length; k++)
                if (BitConverter.SingleToInt32Bits(want[i][k]) != BitConverter.SingleToInt32Bits(got[i][k])) differ++;
            Assert.True(differ == 0, $"{what} step {i + 1}: {differ} of {want[i].Length} values differ");
        }
    }

    /// <summary>The adapters of the parity cases: a diffusers speed adapter on every projection of every block (q, k and v
    /// apart; the refiner too), and a kohya style adapter (dense fused qkv and fc1, trunk only) at strength 0.6.</summary>
    private (List<LoraSpec> Specs, List<(IReadOnlyList<MiniMaxH3ImageLoraFixture.Term>, float)> Merge) Adapters(int seed = 21)
    {
        var speed = MiniMaxH3ImageLoraFixture.Random(seed, 8, 0.5f, perThirdQkv: true, magnitude: 1f);
        var style = MiniMaxH3ImageLoraFixture.Random(seed + 1, 4, 1f, perThirdQkv: false, magnitude: 0.8f, trunkOnly: true,
            projections: new[] { P.Qkv, P.Fc1 });
        string a = _fx.Write(MiniMaxH3ImageLoraFixture.Build(speed, G.Diffusers, prefix: "transformer."), $"speed-{seed}.safetensors");
        string b = _fx.Write(MiniMaxH3ImageLoraFixture.Build(style, G.Kohya), $"style-{seed}.safetensors");
        string config = _fx.WriteConfig($"speed-{seed}.json", a, name: "Speed", steps: 2);
        return (new List<LoraSpec> { new(config), new(b, 0.6f) },
            new List<(IReadOnlyList<MiniMaxH3ImageLoraFixture.Term>, float)> { (speed, 1f), (style, 0.6f) });
    }

    /// <summary>Unmerged on <paramref name="backend"/> against the merged checkpoint denoised without adapters (on the same
    /// backend, so only the LoRA path differs), F32 projections.</summary>
    private void VerifyAgainstTheMerge(BackendType backend, double bound, bool wide = false)
    {
        if (backend != BackendType.Cpu) GgmlBasicOps.EnsureBackendAvailable(GgmlBackend(backend));
        var (specs, merge) = Adapters();
        using var fx = new MiniMaxH3ImageFixture(F32);
        using var merged = new MiniMaxH3ImageFixture(F32, edit: MiniMaxH3ImageLoraFixture.Merge(MiniMaxH3ImageLoraFixture.Deltas(merge)));
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var reference = new MiniMaxH3ImageDenoiser(merged.DitPath, backend);
        using var unmerged = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
        using var set = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), unmerged.Config, backend);
        List<float[]> want = Steps(family, reference, AdapterSet.Empty, wide);
        List<float[]> baseline = Steps(family, unmerged, AdapterSet.Empty, wide, advanceWith: want);
        List<float[]> got = Steps(family, unmerged, set.AdapterSet, wide, advanceWith: want);
        for (int i = 0; i < want.Count; i++)
        {
            var (rel, relL2) = Compare(want[i], got[i]);
            var (moved, movedL2) = Compare(want[i], baseline[i]);
            output.WriteLine($"{backend}{(wide ? " wide" : "")} step {i + 1}: unmerged vs merged max {rel:E3}, L2 {relL2:E3}; " +
                             $"the adapters move the velocity by {movedL2:E3} (L2)");
            Assert.True(rel <= bound && relL2 <= bound, $"step {i + 1}: {rel:E3} (L2 {relL2:E3}) exceeds {bound:E1}");
            Assert.True(movedL2 > 10 * bound, $"step {i + 1}: the adapters move the velocity by only {movedL2:E3}");
        }
        Assert.Same(set, unmerged.BoundLora);
    }

    private static GgmlBackendType GgmlBackend(BackendType b) => b switch
    {
        BackendType.GgmlMetal => GgmlBackendType.Metal,
        BackendType.GgmlCuda => GgmlBackendType.Cuda,
        _ => GgmlBackendType.Cpu,
    };

    // Bounds: the largest relative error measured on each lane (max |err| / max |v|), with headroom. The managed executor
    // and ggml_cpu keep the factors in F32; ggml_metal and ggml_cuda in F16, and ggml_cuda's products round their
    // activations to F16 (TF32 or BF16 on its wide routes).
    [Fact]
    public void UnmergedAdaptersMatchTheMergedCheckpointOnTheManagedExecutor() => VerifyAgainstTheMerge(BackendType.Cpu, 1e-4);

    [GgmlFact(BackendType.GgmlCpu)]
    public void UnmergedAdaptersMatchTheMergedCheckpointOnGgmlCpu() => VerifyAgainstTheMerge(BackendType.GgmlCpu, 1e-4);

    [GgmlFact(BackendType.GgmlMetal)]
    public void UnmergedAdaptersMatchTheMergedCheckpointOnGgmlMetal() => VerifyAgainstTheMerge(BackendType.GgmlMetal, 1e-3);

    [GgmlFact(BackendType.GgmlCuda)]
    public void UnmergedAdaptersMatchTheMergedCheckpointOnGgmlCuda() => VerifyAgainstTheMerge(BackendType.GgmlCuda, 3e-3);

    [GgmlFact(BackendType.GgmlCuda)]
    public void WideUnmergedAdaptersMatchTheMergedCheckpointOnGgmlCuda() => VerifyAgainstTheMerge(BackendType.GgmlCuda, 3e-3, wide: true);

    /// <summary>Block-quantized and BF16 bases on ggml_cuda's wide routes (a Q4_K projection's LoraLinear served as one
    /// node), against the same checkpoint and adapters on the managed executor (exact F32 factors, F32 products).</summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(BF16, 2e-2)]
    [InlineData(Q4_K, 3e-2)]
    public void WideAdaptersOnQuantizedBasesMatchTheManagedExecutorOnGgmlCuda(int format, double bound)
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        var (specs, _) = Adapters();
        using var fx = new MiniMaxH3ImageFixture(format);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var managed = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu);
        using var cuda = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.GgmlCuda);
        using var managedSet = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), managed.Config, BackendType.Cpu);
        using var cudaSet = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), cuda.Config, BackendType.GgmlCuda);
        List<float[]> want = Steps(family, managed, managedSet.AdapterSet, wide: true);
        List<float[]> got = Steps(family, cuda, cudaSet.AdapterSet, wide: true, advanceWith: want);
        int served = GgmlGraphExecutor.ServedRegions(cuda.LastPlan);
        for (int i = 0; i < want.Count; i++)
        {
            var (rel, relL2) = Compare(want[i], got[i]);
            output.WriteLine($"ggml_cuda wide {MiniMaxH3ImageFixture.Name(format)} step {i + 1}: vs managed max {rel:E3}, L2 {relL2:E3} ({served} regions served)");
            Assert.True(rel <= bound && relL2 <= bound, $"step {i + 1}: {rel:E3} (L2 {relL2:E3}) exceeds {bound:E1}");
        }
        Assert.Equal(1, cuda.Compiles);
    }

    // ---- strength 0 and switching -------------------------------------------------------------------------------------

    /// <summary>Strength 0 keeps every term with zero up factors: the base denoiser's velocities bit for bit.</summary>
    private void VerifyStrengthZero(BackendType backend, int format)
    {
        if (backend != BackendType.Cpu) GgmlBasicOps.EnsureBackendAvailable(GgmlBackend(backend));
        var (specs, _) = Adapters();
        using var fx = new MiniMaxH3ImageFixture(format);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var denoiser = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
        using var zero = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs.Select(s => s with { Scale = 0f }).ToList()), denoiser.Config, backend);
        List<float[]> want = Steps(family, denoiser, AdapterSet.Empty, wide: false);
        List<float[]> got = Steps(family, denoiser, zero.AdapterSet, wide: false, advanceWith: want);
        AssertBitExact(want, got, $"{backend} {MiniMaxH3ImageFixture.Name(format)} strength 0");
    }

    [Theory]
    [InlineData(F32)]
    [InlineData(Q8_0)]
    public void StrengthZeroIsTheBaseDenoiserBitForBitOnTheManagedExecutor(int format) => VerifyStrengthZero(BackendType.Cpu, format);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(F32)]
    [InlineData(Q4_K)]
    public void StrengthZeroIsTheBaseDenoiserBitForBitOnGgmlCpu(int format) => VerifyStrengthZero(BackendType.GgmlCpu, format);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(F32)]
    [InlineData(Q8_0)]
    public void StrengthZeroIsTheBaseDenoiserBitForBitOnGgmlMetal(int format) => VerifyStrengthZero(BackendType.GgmlMetal, format);

    /// <summary>
    /// One denoiser serves A, then no adapters, then B, then A again, then A reloaded (a new set: new factors, possibly at the
    /// freed set's addresses): each request's velocities are those of a fresh denoiser given that set alone, bit for bit, so
    /// neither a plan nor a device copy of an earlier set survives into a later request.
    /// </summary>
    private void VerifySwitching(BackendType backend)
    {
        if (backend != BackendType.Cpu) GgmlBasicOps.EnsureBackendAvailable(GgmlBackend(backend));
        var (specsA, _) = Adapters(31);
        var (specsB, _) = Adapters(41);
        using var fx = new MiniMaxH3ImageFixture(Q8_0);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        MiniMaxH3Config cfg;
        using (var probe = new MiniMaxH3ImageDenoiser(fx.DitPath, backend)) cfg = probe.Config;
        List<float[]> Fresh(IReadOnlyList<LoraSpec> specs)
        {
            using var d = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
            if (specs == null) return Steps(family, d, AdapterSet.Empty, wide: false);
            using var set = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), cfg, backend);
            return Steps(family, d, set.AdapterSet, wide: false);
        }
        List<float[]> freshA = Fresh(specsA), freshB = Fresh(specsB), freshNone = Fresh(null);

        using var served = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
        var setA = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specsA), cfg, backend);
        AssertBitExact(freshA, Steps(family, served, setA.AdapterSet, wide: false), $"{backend} A");
        AssertBitExact(freshNone, Steps(family, served, AdapterSet.Empty, wide: false), $"{backend} A, then none");
        Assert.Null(served.BoundLora);
        using (var setB = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specsB), cfg, backend))
            AssertBitExact(freshB, Steps(family, served, setB.AdapterSet, wide: false), $"{backend} none, then B");
        AssertBitExact(freshA, Steps(family, served, setA.AdapterSet, wide: false), $"{backend} B, then A");
        // The model's swap: release the binding, free the set, load the next.
        served.ReleaseLora();
        setA.Dispose();
        using var reloaded = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specsA), cfg, backend);
        AssertBitExact(freshA, Steps(family, served, reloaded.AdapterSet, wide: false), $"{backend} A reloaded");
        Assert.Same(reloaded, served.BoundLora);
    }

    [Fact]
    public void SwitchingAdaptersLeavesNothingBehindOnTheManagedExecutor() => VerifySwitching(BackendType.Cpu);

    [GgmlFact(BackendType.GgmlCpu)]
    public void SwitchingAdaptersLeavesNothingBehindOnGgmlCpu() => VerifySwitching(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void SwitchingAdaptersLeavesNothingBehindOnGgmlMetal() => VerifySwitching(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void SwitchingAdaptersLeavesNothingBehindOnGgmlCuda() => VerifySwitching(BackendType.GgmlCuda);

    /// <summary>A geometry's steps with adapters share one plan; a set loaded for another backend or checkpoint is refused.</summary>
    [Fact]
    public void AdaptedStepsCompileOnceAndAForeignSetIsRefused()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        var (specs, _) = Adapters();
        using var fx = new MiniMaxH3ImageFixture(Q8_0);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var denoiser = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
        using var set = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), denoiser.Config, backend);
        Steps(family, denoiser, set.AdapterSet, wide: false);
        Assert.Equal(1, denoiser.Compiles);
        Assert.Equal(set.FactorBytes > 0, true);
        using var managedSet = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), denoiser.Config, BackendType.Cpu);
        Assert.Contains("is packed for Cpu", Assert.Throws<ArgumentException>(() => Steps(family, denoiser, managedSet.AdapterSet, wide: false)).Message);
        var foreign = new AdapterSet(new[] { new DiffusionAdapter { Name = "other", Payload = new object() } });
        Assert.Contains("was not loaded by MiniMax-H3-Image's LoRA loader",
            Assert.Throws<ArgumentException>(() => Steps(family, denoiser, foreign, wide: false)).Message);
    }
}
