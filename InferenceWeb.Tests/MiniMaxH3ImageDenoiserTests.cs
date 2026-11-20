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
/// MiniMax-H3-Image's denoiser on the diffusion runtime (MiniMaxH3ImageDenoiser: one GraphProgram a step) on a tiny synthetic
/// checkpoint (<see cref="MiniMaxH3ImageFixture"/>): its plans, residency, refusals and host values, and the family's recipe,
/// layout and token rows. Its velocities against the legacy denoiser's, frozen before stage S10 deleted it, are
/// <see cref="MiniMaxH3ImageFrozenLegacyTests"/>'.
/// </summary>
public sealed class MiniMaxH3ImageDenoiserTests(ITestOutputHelper output)
{
    private const int F32 = MiniMaxH3ImageFixture.F32, BF16 = MiniMaxH3ImageFixture.BF16, Q8_0 = MiniMaxH3ImageFixture.Q8_0,
        Q4_K = MiniMaxH3ImageFixture.Q4_K;

    private static MiniMaxH3ImageGoldens.Request Request(int h, int w, int text) => new(h, w, text);

    private static float[] Random(Random rng, long n, float scale = 1f)
    {
        var v = new float[n];
        for (long i = 0; i < n; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    [Fact]
    public void TheManySegmentRequestHasMoreSegmentsThanTheFusedKernelsTake()
    {
        using var fx = new MiniMaxH3ImageFixture(BF16);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        MiniMaxH3ImageGoldens.Request rq = MiniMaxH3ImageGoldens.Request.ManySegments;
        var text = new ConditioningSet(new float[rq.TextRows * MiniMaxH3ImageFixture.TextDim], rq.TextRows, MiniMaxH3ImageFixture.TextDim, rq.Modalities);
        var shape = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, 8, 8);
        var references = new[] { new LatentRef(0, shape, family.ToRows(new LatentTensor(shape, new float[shape.Count]))) };
        var target = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, rq.LatentHeight, rq.LatentWidth);
        DiffusionLayout layout = family.Pack(text, references, target, new StepState(0, 6, 1f, 0.9f, 0, 1));
        Assert.True(layout.Segments.Count > 8, $"{layout.Segments.Count} segments");
        Assert.True(layout.Rows >= 1024, $"{layout.Rows} rows");
    }

    // ================================================================ plans, residency and the step's host values

    /// <summary>A geometry's steps share one plan; another text length records and compiles a new one in its place, and a
    /// released workspace compiles again with the same results, as do released device copies.</summary>
    [Fact]
    public void AGeometryCompilesOnceAndAReleasedWorkspaceRecompilesToTheSameBytes()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var fx = new MiniMaxH3ImageFixture(Q8_0);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var runtime = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
        Assert.True(runtime.Preload() >= 0);
        Assert.Equal(0, runtime.Preload());   // once
        var small = Request(8, 8, 5);
        var (first, _) = MiniMaxH3ImageGoldens.Run(family, runtime, small, steps: 3);
        Assert.Equal(1, runtime.Compiles);
        MiniMaxH3ImageGoldens.Run(family, runtime, small with { TextRows = 6 }, steps: 2);
        Assert.Equal(2, runtime.Compiles);
        runtime.ReleaseWorkspace();
        Assert.Null(runtime.LastPlan);
        var (again, _) = MiniMaxH3ImageGoldens.Run(family, runtime, small, steps: 3);
        Assert.Equal(3, runtime.Compiles);
        for (int i = 0; i < first.Count; i++) Assert.Equal(first[i], again[i]);
        // Released device copies are uploaded again from the mapping by the next step.
        runtime.ReleaseDevice();
        var (resident, _) = MiniMaxH3ImageGoldens.Run(family, runtime, small, steps: 3);
        for (int i = 0; i < first.Count; i++) Assert.Equal(first[i], resident[i]);
    }

    [Fact]
    public void TheDenoiserRefusesForeignAdaptersAndAForeignLayout()
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
        // Adapters the family's loader did not load (MiniMaxH3ImageLoraDenoiserTests covers the ones it did).
        var adapters = new AdapterSet(new[] { new DiffusionAdapter { Name = "style" } });
        Assert.Throws<ArgumentException>(() => runtime.Step(inputs, adapters, CancellationToken.None));
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
        // A fixed part, then the same bytes a row (2x2 patches and a token every three characters of the prompt).
        Assert.Equal(64 * 64 / 4 + 1, MiniMaxH3ImageDenoiser.RowsOf(large));
        Assert.Equal(MiniMaxH3ImageDenoiser.WorkingBytesFor(runtime.Config, 1025, TestGates.PinnedGgmlBackend == BackendType.GgmlCuda, quantized: true),
            runtime.WorkingBytes(large));
        long perRow = (runtime.WorkingBytes(large) - runtime.WorkingBytes(small)) / (1025 - 17);
        Assert.True(perRow > 0 && runtime.WorkingBytes(small) > 17 * perRow);
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
        Assert.Equal("minimax-h3-image", MiniMaxH3ImageFamily.ArchitectureId);
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

    /// <summary>Edits (stage S5) and masked edits (stage S6) are the family's; a mask needs a picture to edit; its components
    /// need their files (a load without them names the missing role).</summary>
    [Fact]
    public void TheFamilyTakesMasksAndRefusesComponentsWithoutTheirFiles()
    {
        var family = new MiniMaxH3ImageFamily(MiniMaxH3ImageSampling.ReleaseSigmas);
        var image = new TensorSharp.Models.QwenImage.RgbImage(32, 32, new float[32 * 32 * 3]);
        // Edits are taken (MiniMaxH3ImageEditTests), masked ones on the source's own canvas.
        Assert.Equal(new DiffusionCanvas((2048, 2048), (1024, 1024)), family.ResolveCanvas(new DiffusionRequest
        {
            Prompt = "x",
            References = new[] { new DiffusionReference(image) },
        }, null));
        EditMask mask = EditMask.Create(new EditMaskOptions { Mask = image }, image);
        Assert.Equal(new DiffusionCanvas((2048, 2048), (32, 32)), family.ResolveCanvas(new DiffusionRequest
        {
            Prompt = "x",
            References = new[] { new DiffusionReference(image) },
        }, mask));
        Assert.Contains("requires a first reference image", Assert.Throws<ArgumentException>(() =>
            family.ResolveCanvas(new DiffusionRequest { Prompt = "x" }, mask)).Message);
        Assert.Equal(DiffusionCanvas.Of(2048, 2048), family.ResolveCanvas(new DiffusionRequest { Prompt = "x" }, null));
        Assert.Equal(DiffusionCanvas.Of(1024, 512), family.ResolveCanvas(new DiffusionRequest { Prompt = "x", Width = 1024, Height = 512 }, null));
        var load = new DiffusionLoad { Backend = TestGates.PinnedGgmlBackend };
        Assert.Contains("text-encoder", Assert.Throws<InvalidOperationException>(() => family.CreateConditioner(load)).Message);
        Assert.Contains("vae", Assert.Throws<InvalidOperationException>(() => family.CreateVae(load)).Message);
        Assert.Equal((1024, 1024), (family.References.ForVae(image, 0).Width, family.References.ForVae(image, 0).Height));
    }
}
