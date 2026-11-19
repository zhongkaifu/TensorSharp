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
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image's denoiser on the diffusion runtime (MiniMaxH3ImageDenoiser: one GraphProgram a step) against the legacy
/// denoiser (MiniMaxH3DiT.ForwardImage, TSGgml_MiniMaxH3DitForward's native graph) on a tiny synthetic checkpoint
/// (<see cref="MiniMaxH3ImageFixture"/>): the same velocities bit for bit on every GGML backend the step lowers the way the
/// native graph is built (ggml_cpu, ggml_metal, ggml_cuda with its fused nodes), within a measured bound on the managed
/// executor, over a whole six-step loop that visits every PDD head.
/// </summary>
public sealed class MiniMaxH3ImageDenoiserTests(ITestOutputHelper output)
{
    private const int F32 = MiniMaxH3ImageFixture.F32, BF16 = MiniMaxH3ImageFixture.BF16, Q8_0 = MiniMaxH3ImageFixture.Q8_0,
        Q4_K = MiniMaxH3ImageFixture.Q4_K;

    /// <summary>A request's packed sequence: the target latent, the text rows (their modality tags, null all text) and the
    /// references' latents.</summary>
    private sealed record Request(int LatentHeight, int LatentWidth, int TextRows, int[] Modalities = null, (int H, int W)[] References = null)
    {
        /// <summary>Text to image at 272 x 272 (289 image tokens, 310 rows: past 256, so V's scale is 2).</summary>
        public static readonly Request TextToImage = new(34, 34, 21);

        /// <summary>Text with an image's tags, two references (a row at t = 0.999 of its own) and an odd-shaped target.</summary>
        public static readonly Request WithReferences = new(16, 20, 12, new[] { 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1 }, new[] { (8, 8), (12, 8) });

        /// <summary>Over a thousand rows, where ggml_cuda's wide routes take the products (cuBLAS BF16, h3_q4k_gemm_takes).</summary>
        public static readonly Request Wide = new(64, 64, 21);

        /// <summary>Wide, with text tags alternating row by row and a reference: more modulation segments than the fused
        /// kernels take (tsg_h3_fused_max_segments, 8), so the native DiT builds every block unfused, its BF16
        /// SwiGLU-in-GEMM included.</summary>
        public static readonly Request ManySegments = new(64, 64, 12, Enumerable.Range(0, 12).Select(i => i % 2).ToArray(), new[] { (8, 8) });
    }

    private static float[] Random(Random rng, long n, float scale = 1f)
    {
        var v = new float[n];
        for (long i = 0; i < n; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private sealed class Loop
    {
        public readonly List<float[]> Legacy = new(), Runtime = new();
    }

    /// <summary>Six Euler steps from the same noise, each step's velocity from both denoisers on the same inputs (the state
    /// advanced with the legacy one's).</summary>
    private static Loop Run(MiniMaxH3ImageFamily family, IDenoiser legacy, IDenoiser runtime, Request rq, int seed = 7, int steps = 6)
    {
        var rng = new Random(seed);
        var text = new ConditioningSet(Random(rng, (long)rq.TextRows * MiniMaxH3ImageFixture.TextDim, 2f), rq.TextRows,
            MiniMaxH3ImageFixture.TextDim, rq.Modalities);
        var references = (rq.References ?? Array.Empty<(int, int)>()).Select((r, i) =>
        {
            var shape = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, r.H, r.W);
            return new LatentRef(i, shape, family.ToRows(new LatentTensor(shape, Random(rng, shape.Count))));
        }).ToArray();
        var target = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, rq.LatentHeight, rq.LatentWidth);
        float[] x = family.ToRows(new LatentTensor(target, family.Noise(seed).Next(target.Count)));
        float[] sigmas = family.DefaultRecipe.SigmasFor(steps);
        var loop = new Loop();
        for (int i = 0; i < steps; i++)
        {
            float sigma = sigmas[i], next = sigmas[i + 1];
            var (start, stop) = family.DefaultRecipe.HeadsFor(i, sigma, next);
            var state = new StepState(i, steps, sigma, next, start, stop);
            var inputs = new DenoiseStepInputs
            {
                Latent = x,
                Conditioning = text,
                References = references,
                Layout = family.Pack(text, references, target, state),
                Step = state,
                Recipe = family.DefaultRecipe,
            };
            float[] want = (float[])legacy.Step(inputs, AdapterSet.Empty, CancellationToken.None).Velocity.Clone();
            float[] got = (float[])runtime?.Step(inputs, AdapterSet.Empty, CancellationToken.None).Velocity.Clone();
            loop.Legacy.Add(want);
            loop.Runtime.Add(got);
            for (long k = 0; k < x.Length; k++) x[k] += want[k] * (next - sigma);
        }
        return loop;
    }

    private void AssertBitExact(Loop loop, string what)
    {
        for (int i = 0; i < loop.Legacy.Count; i++)
        {
            float[] want = loop.Legacy[i], got = loop.Runtime[i];
            Assert.Equal(want.Length, got.Length);
            Assert.All(want, v => Assert.True(float.IsFinite(v)));
            int differ = 0, first = -1;
            for (int k = 0; k < want.Length; k++)
                if (BitConverter.SingleToInt32Bits(want[k]) != BitConverter.SingleToInt32Bits(got[k]))
                {
                    if (first < 0) first = k;
                    differ++;
                }
            double scale = want.Max(v => Math.Abs((double)v));
            output.WriteLine($"{what} step {i + 1}: {differ} of {want.Length} values differ (max |v| {scale:G4})");
            Assert.True(differ == 0, $"{what} step {i + 1}: {differ} values differ, the first at {first} (legacy {(first >= 0 ? want[first] : 0)}, runtime {(first >= 0 ? got[first] : 0)})");
        }
    }

    /// <summary>The bit-exactness check on one GGML backend: legacy and runtime in one process, the runtime over the legacy DiT's
    /// mapping (<paramref name="share"/>, one device copy) or its own.</summary>
    private void VerifyBitExact(BackendType backend, int format, Request rq, bool share)
    {
        GgmlBasicOps.EnsureBackendAvailable(backend switch
        {
            BackendType.GgmlMetal => GgmlBackendType.Metal,
            BackendType.GgmlCuda => GgmlBackendType.Cuda,
            _ => GgmlBackendType.Cpu,
        });
        using var fx = new MiniMaxH3ImageFixture(format);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var legacy = new MiniMaxH3ImageLegacyDenoiser(fx.DitPath, backend);
        using MiniMaxH3ImageDenoiser runtime = share
            ? MiniMaxH3ImageDenoiser.SharingLegacyWeights(legacy.Dit, backend)
            : new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
        Loop loop = Run(family, legacy, runtime, rq);
        string what = $"{backend} {MiniMaxH3ImageFixture.Name(format)} {rq.LatentHeight}x{rq.LatentWidth}";
        AssertBitExact(loop, what);
        // One program for the request's six steps (the head, latent and time embedding are its inputs).
        Assert.Equal(1, runtime.Compiles);
        int served = GgmlGraphExecutor.ServedRegions(runtime.LastPlan), lowered = GgmlGraphExecutor.LoweredRegions(runtime.LastPlan);
        output.WriteLine($"{what}: {served} regions served, {lowered} lowered");
        if (backend != BackendType.GgmlCuda)
        {
            // Off ggml_cuda nothing is served, and every attention (the refiner's and each block's) is the native graph's.
            Assert.Equal(0, served);
            Assert.Equal(MiniMaxH3ImageFixture.RefinerBlocks + MiniMaxH3ImageFixture.Blocks, lowered);
        }
    }

    public static IEnumerable<object[]> Formats() => new[] { F32, BF16, Q8_0, Q4_K }.Select(f => new object[] { f });

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(Formats))]
    public void TextToImageStepsMatchTheLegacyDenoiserBitForBitOnGgmlCpu(int format)
        => VerifyBitExact(BackendType.GgmlCpu, format, Request.TextToImage, share: format != F32);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(Formats))]
    public void ReferencedStepsMatchTheLegacyDenoiserBitForBitOnGgmlCpu(int format)
        => VerifyBitExact(BackendType.GgmlCpu, format, Request.WithReferences, share: true);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(Formats))]
    public void TextToImageStepsMatchTheLegacyDenoiserBitForBitOnGgmlMetal(int format)
        => VerifyBitExact(BackendType.GgmlMetal, format, Request.TextToImage, share: format != F32);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(Formats))]
    public void ReferencedStepsMatchTheLegacyDenoiserBitForBitOnGgmlMetal(int format)
        => VerifyBitExact(BackendType.GgmlMetal, format, Request.WithReferences, share: true);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(Formats))]
    public void TextToImageStepsMatchTheLegacyDenoiserBitForBitOnGgmlCuda(int format)
        => VerifyBitExact(BackendType.GgmlCuda, format, Request.TextToImage, share: format != F32);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(Formats))]
    public void ReferencedStepsMatchTheLegacyDenoiserBitForBitOnGgmlCuda(int format)
        => VerifyBitExact(BackendType.GgmlCuda, format, Request.WithReferences, share: true);

    /// <summary>Past a thousand rows ggml_cuda's wide routes take the products: cuBLAS BF16 (with the q8_1 guard pair
    /// dropped around a Q4_K weight) and the BF16 fc2 that applies the SwiGLU as it narrows.</summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(BF16)]
    [InlineData(Q4_K)]
    public void WideStepsMatchTheLegacyDenoiserBitForBitOnGgmlCuda(int format)
        => VerifyBitExact(BackendType.GgmlCuda, format, Request.Wide, share: true);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(Q4_K)]
    public void WideStepsMatchTheLegacyDenoiserBitForBitOnGgmlMetal(int format)
        => VerifyBitExact(BackendType.GgmlMetal, format, Request.Wide, share: true);

    /// <summary>Past the fused kernels' segment count ggml_cuda serves no modulation or gated residual, and then (as the native
    /// DiT does) no BF16 SwiGLU-in-GEMM either: fc1, the SwiGLU and fc2 run as the region's ops on the wide routes.</summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(BF16)]
    [InlineData(Q4_K)]
    public void ManySegmentStepsMatchTheLegacyDenoiserBitForBitOnGgmlCuda(int format)
        => VerifyBitExact(BackendType.GgmlCuda, format, Request.ManySegments, share: true);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(BF16)]
    public void ManySegmentStepsMatchTheLegacyDenoiserBitForBitOnGgmlCpu(int format)
        => VerifyBitExact(BackendType.GgmlCpu, format, Request.ManySegments, share: true);

    [Fact]
    public void TheManySegmentRequestHasMoreSegmentsThanTheFusedKernelsTake()
    {
        using var fx = new MiniMaxH3ImageFixture(BF16);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        Request rq = Request.ManySegments;
        var text = new ConditioningSet(new float[rq.TextRows * MiniMaxH3ImageFixture.TextDim], rq.TextRows, MiniMaxH3ImageFixture.TextDim, rq.Modalities);
        var shape = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, 8, 8);
        var references = new[] { new LatentRef(0, shape, family.ToRows(new LatentTensor(shape, new float[shape.Count]))) };
        var target = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, rq.LatentHeight, rq.LatentWidth);
        DiffusionLayout layout = family.Pack(text, references, target, new StepState(0, 6, 1f, 0.9f, 0, 1));
        Assert.True(layout.Segments.Count > 8, $"{layout.Segments.Count} segments");
        Assert.True(layout.Rows >= 1024, $"{layout.Rows} rows");
    }

    /// <summary>The managed executor runs the same program through the regions' reference ops (its own products and
    /// attention): against the legacy denoiser on the process's GGML backend, every step within a bound about five times
    /// the largest difference measured on the lanes' backends (largest error over the largest velocity; the relative L2 is
    /// a little below it): F32 9e-5 on ggml_cpu and 6.2e-4 on ggml_metal (its F32 products), BF16 4.3e-4 and 5.8e-4, Q8_0
    /// 1.4e-3 and 1.8e-3 (ggml rounds the activations of a BF16 product to BF16 and quantizes those of a Q8_0 product to
    /// q8_0; the managed products read them in F32).</summary>
    [Theory]
    [InlineData(F32, 3e-3)]
    [InlineData(BF16, 3e-3)]
    [InlineData(Q8_0, 1e-2)]
    public void ManagedExecutorStepsMatchTheLegacyDenoiserWithinTheBound(int format, double bound)
    {
        BackendType ggml = TestGates.PinnedGgmlBackend;
        using var fx = new MiniMaxH3ImageFixture(format);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var legacy = new MiniMaxH3ImageLegacyDenoiser(fx.DitPath, ggml);
        using var managed = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu);
        Assert.IsType<TensorSharp.Models.Graph.ManagedGraphExecutor>(managed.Executor);
        Loop loop = Run(family, legacy, managed, Request.WithReferences);
        for (int i = 0; i < loop.Legacy.Count; i++)
        {
            float[] want = loop.Legacy[i], got = loop.Runtime[i];
            double scale = want.Max(v => Math.Abs((double)v)), worst = 0, l2 = 0, n2 = 0;
            for (int k = 0; k < want.Length; k++)
            {
                Assert.True(float.IsFinite(got[k]), $"step {i + 1}: value {k} is {got[k]}");
                double d = got[k] - (double)want[k];
                worst = Math.Max(worst, Math.Abs(d));
                l2 += d * d;
                n2 += (double)want[k] * want[k];
            }
            double rel = worst / scale, relL2 = Math.Sqrt(l2 / n2);
            output.WriteLine($"managed vs legacy ({ggml}) {MiniMaxH3ImageFixture.Name(format)} step {i + 1}: max |err| {worst:E3} " +
                             $"({rel:E3} of max |v| {scale:G4}), relative L2 {relL2:E3}");
            Assert.True(rel <= bound && relL2 <= bound, $"step {i + 1}: relative error {rel:E3} (L2 {relL2:E3}) exceeds {bound:E1}");
        }
        Assert.Equal(1, managed.Compiles);
    }

    // ================================================================ plans, residency and the step's host values

    /// <summary>A geometry's steps share one plan; another text length records and compiles a new one in its place, and a
    /// released workspace compiles again with the same results.</summary>
    [Fact]
    public void AGeometryCompilesOnceAndAReleasedWorkspaceRecompilesToTheSameBytes()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var fx = new MiniMaxH3ImageFixture(Q8_0);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var legacy = new MiniMaxH3ImageLegacyDenoiser(fx.DitPath, backend);
        using var runtime = MiniMaxH3ImageDenoiser.SharingLegacyWeights(legacy.Dit, backend);
        Assert.True(runtime.Preload() >= 0);
        Assert.Equal(0, runtime.Preload());   // once
        var small = new Request(8, 8, 5);
        Loop first = Run(family, legacy, runtime, small, steps: 3);
        Assert.Equal(1, runtime.Compiles);
        Run(family, legacy, runtime, small with { TextRows = 6 }, steps: 2);
        Assert.Equal(2, runtime.Compiles);
        runtime.ReleaseWorkspace();
        Assert.Null(runtime.LastPlan);
        Loop again = Run(family, legacy, runtime, small, steps: 3);
        Assert.Equal(3, runtime.Compiles);
        for (int i = 0; i < first.Runtime.Count; i++) Assert.Equal(first.Runtime[i], again.Runtime[i]);
        AssertBitExact(again, $"{backend} after a released workspace");
        // Released device copies are uploaded again from the mapping by the next step.
        runtime.ReleaseDevice();
        Loop resident = Run(family, legacy, runtime, small, steps: 3);
        for (int i = 0; i < first.Runtime.Count; i++) Assert.Equal(first.Runtime[i], resident.Runtime[i]);
    }

    [Fact]
    public void TheStepsHostTablesAreTheLegacyDenoisersBitForBit()
    {
        using var fx = new MiniMaxH3ImageFixture(F32);
        using var legacy = new MiniMaxH3DiT(fx.DitPath, TestGates.PinnedGgmlBackend, null);
        using var runtime = MiniMaxH3ImageDenoiser.SharingLegacyWeights(legacy, TestGates.PinnedGgmlBackend);
        MiniMaxH3PackedLayout layout = MiniMaxH3ImageLayout.Build(7, 10, 12, 0.6f, null, new[] { (4, 6) });
        Assert.Equal(legacy.TimeEmbedding(layout.Timesteps), runtime.TimeEmbedding(layout.Timesteps));
        var (cos, sin) = legacy.BuildRope(layout.Positions);
        var (rc, rs) = runtime.Rope(layout.Positions);
        Assert.Equal(cos, rc);
        Assert.Equal(sin, rs);
        Assert.Same(rc, runtime.Rope((float[])layout.Positions.Clone()).Cos);   // kept for the request's steps
    }

    [Fact]
    public void TheDenoiserRefusesAdaptersAndAForeignLayout()
    {
        using var fx = new MiniMaxH3ImageFixture(F32);
        using var runtime = new MiniMaxH3ImageDenoiser(fx.DitPath, TestGates.PinnedGgmlBackend);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        var text = new ConditioningSet(new float[3 * MiniMaxH3ImageFixture.TextDim], 3, MiniMaxH3ImageFixture.TextDim);
        var target = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, 4, 4);
        var state = new StepState(0, 6, 1f, 0.9f, 0, 1);
        var inputs = new DenoiseStepInputs
        {
            Latent = new float[target.Count],
            Conditioning = text,
            Layout = family.Pack(text, Array.Empty<LatentRef>(), target, state),
            Step = state,
            Recipe = family.DefaultRecipe,
        };
        var adapters = new AdapterSet(new[] { new DiffusionAdapter { Name = "style" } });
        Assert.Throws<NotSupportedException>(() => runtime.Step(inputs, adapters, CancellationToken.None));
        var foreign = new DenoiseStepInputs
        {
            Latent = inputs.Latent, Conditioning = text, Step = state, Recipe = family.DefaultRecipe,
            Layout = new DiffusionLayout { Rows = 7, TargetStart = 3, TargetRows = 4 },
        };
        Assert.Throws<ArgumentException>(() => runtime.Step(foreign, AdapterSet.Empty, CancellationToken.None));
        Assert.Equal(0, runtime.Compiles);
    }

    [Fact]
    public void TheDenoiserRefusesAnAudioVideoCheckpoint()
    {
        using var fixtures = new MiniMaxH3LoraFixtures();
        string video = fixtures.WriteTransformer();
        Assert.Throws<NotSupportedException>(() => new MiniMaxH3ImageDenoiser(video, TestGates.PinnedGgmlBackend));
        Assert.Throws<NotSupportedException>(() => MiniMaxH3ImageFamily.ForDenoiser(video));
    }

    [Fact]
    public void WeightAndWorkingBytesAreTheLeavesAndAPositiveEstimate()
    {
        using var fx = new MiniMaxH3ImageFixture(Q4_K);
        using var runtime = new MiniMaxH3ImageDenoiser(fx.DitPath, TestGates.PinnedGgmlBackend);
        const int H = MiniMaxH3ImageFixture.Hidden, I = MiniMaxH3ImageFixture.Inner, F = MiniMaxH3ImageFixture.Ffn;
        long q4k(long input, long output) => input / 256 * 144 * output;
        long block = q4k(H, 3 * I) + q4k(I, H) + q4k(H, 2 * F) + q4k(F, H) + 4L * (2 * H + 2 * MiniMaxH3ImageFixture.HeadDim);
        long modulation = 2L * MiniMaxH3ImageFixture.TimeDim * 18 * H + 4L * 18 * H;
        long expected = MiniMaxH3ImageFixture.RefinerBlocks * block + MiniMaxH3ImageFixture.Blocks * (block + modulation)
            + 2L * MiniMaxH3ImageFixture.TextDim * H + 4L * H                 // condition_proj (BF16) and its bias
            + 4L * MiniMaxH3ImageFixture.PatchDim * H + 4L * H                // video_patch_proj (F32) and its bias
            + 2L * MiniMaxH3ImageFixture.TimeDim * 2 * H + 4L * 2 * H         // the final AdaLN projection and its bias
            + 4L * H * 2;                                                      // the final and the refiner's last norm
        Assert.Equal(expected, runtime.WeightBytes);
        var small = new DiffusionWorkload(new LatentShape(24, 1, 8, 8), Array.Empty<LatentShape>(), "a");
        var large = new DiffusionWorkload(new LatentShape(24, 1, 64, 64), Array.Empty<LatentShape>(), "a");
        Assert.True(runtime.WorkingBytes(small) > 0);
        Assert.True(runtime.WorkingBytes(large) > 50 * runtime.WorkingBytes(small));
    }

    // ================================================================ the family

    [Fact]
    public void TheFamilysRecipeIsTheLegacySchedulesAndHeads()
    {
        var family = new MiniMaxH3ImageFamily(MiniMaxH3ImageSampling.ReleaseSigmas);
        for (int steps = 1; steps <= 12; steps++)
        {
            float[] sigmas = family.DefaultRecipe.SigmasFor(steps);
            Assert.Equal(MiniMaxH3ImageSampling.Schedule(MiniMaxH3ImageSampling.ReleaseSigmas, steps), sigmas);
            for (int i = 0; i < steps; i++)
                Assert.Equal(MiniMaxH3ImageSampling.Interval(MiniMaxH3ImageSampling.ReleaseSigmas, sigmas[i], sigmas[i + 1]),
                    family.DefaultRecipe.HeadsFor(i, sigmas[i], sigmas[i + 1]));
        }
        Assert.Equal(6, family.DefaultRecipe.Steps);
        Assert.Equal(DiffusionGuidance.None, family.Guidance);
        Assert.Equal(96, family.Geometry.ValuesPerToken);
        Assert.Equal(TorchCpuRandn.Normal(1000, 42), family.Noise(42).Next(1000));
        Assert.Equal(MiniMaxH3ImageFamily.ArchitectureId, MiniMaxH3ImageModel.ArchitectureId);
    }

    [Fact]
    public void PackIsMiniMaxH3ImageLayoutWithItsSegmentsAndTimesteps()
    {
        var family = new MiniMaxH3ImageFamily(MiniMaxH3ImageSampling.ReleaseSigmas);
        var text = new ConditioningSet(new float[5 * 8], 5, 8, new[] { 1, 0, 0, 1, 1 });
        var refShape = new LatentShape(24, 1, 6, 4);
        var references = new[] { new LatentRef(0, refShape, new float[refShape.Count]) };
        var target = new LatentShape(24, 1, 8, 10);
        DiffusionLayout layout = family.Pack(text, references, target, new StepState(1, 6, 0.75f, 0.5f, 3, 4));
        MiniMaxH3PackedLayout want = MiniMaxH3ImageLayout.Build(5, 8, 10, 0.75f, text.Modalities, new[] { (6, 4) });
        var packed = Assert.IsType<MiniMaxH3PackedLayout>(layout.FamilyLayout);
        Assert.Equal(want.Positions, packed.Positions);
        Assert.Equal(want.Positions, layout.Positions);
        Assert.Equal(want.Timesteps, layout.Timesteps);
        Assert.Equal(want.TokenCount, layout.Rows);
        Assert.Equal(5 + 6, layout.TargetStart);
        Assert.Equal(20, layout.TargetRows);
        Assert.Equal(3, layout.PositionAxes);
        Assert.Equal(want.ModulationSpans.Select(s => (s.Start, s.End - s.Start, s.Row % 3, s.Row / 3)),
            layout.Segments.Select(s => (s.Start, s.Length, s.Modality, s.Timestep)));
    }

    [Fact]
    public void TokenRowsAreTheLegacyPatchifyAndRoundTrip()
    {
        var family = new MiniMaxH3ImageFamily(MiniMaxH3ImageSampling.ReleaseSigmas);
        var shape = new LatentShape(24, 1, 6, 8);
        float[] chw = Random(new Random(3), shape.Count);
        float[] rows = family.ToRows(new LatentTensor(shape, chw));
        var hwc = new float[chw.Length];
        for (int c = 0; c < 24; c++)
            for (int i = 0; i < 48; i++)
                hwc[i * 24 + c] = chw[c * 48 + i];
        Assert.Equal(MiniMaxH3DiT.PatchifyVideo(hwc, 1, 6, 8, 24), rows);
        Assert.Equal(chw, family.FromRows(rows, shape).Data);
        // A per-cell mask in the rows' layout: every value of a token carries its own cell's weight.
        float[] cells = Enumerable.Range(0, 48).Select(i => (float)i).ToArray();
        var (weights, per) = family.MaskRows(cells, shape);
        Assert.Equal(1, per);
        float[] cellOfValue = family.ToRows(new LatentTensor(shape, Enumerable.Range(0, (int)shape.Count).Select(i => (float)(i % 48)).ToArray()));
        Assert.Equal(cellOfValue, weights);
    }

    [Fact]
    public void TheFamilyRefusesEditsAndTheComponentsLaterStagesBring()
    {
        var family = new MiniMaxH3ImageFamily(MiniMaxH3ImageSampling.ReleaseSigmas);
        var image = new TensorSharp.Models.QwenImage.RgbImage(32, 32, new float[32 * 32 * 3]);
        Assert.Throws<ArgumentException>(() => family.ResolveCanvas(new DiffusionRequest
        {
            Prompt = "x",
            References = new[] { new DiffusionReference(image) },
        }, null));
        Assert.Equal(DiffusionCanvas.Of(2048, 2048), family.ResolveCanvas(new DiffusionRequest { Prompt = "x" }, null));
        Assert.Equal(DiffusionCanvas.Of(1024, 512), family.ResolveCanvas(new DiffusionRequest { Prompt = "x", Width = 1024, Height = 512 }, null));
        var load = new DiffusionLoad { Backend = TestGates.PinnedGgmlBackend };
        Assert.Throws<NotSupportedException>(() => family.CreateConditioner(load));
        Assert.Throws<NotSupportedException>(() => family.CreateVae(load));
        Assert.Throws<NotSupportedException>(() => family.References.ForVae(image, 0));
    }
}
