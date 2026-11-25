// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The diffusion runtime's sampling recipes, head policies, Euler sampler and sampling contracts: pure arithmetic, no
// weights. The golden schedule bits are ComfyUI 0.45.0's (MiniMaxH3ImageModelSampling + simple_scheduler), which a
// numpy float32 replay of the algorithm reproduces bit for bit; the legacy MiniMaxH3ImageSampling and
// MiniMaxH3Scheduler are the live oracles until they are deleted (docs/design/diffusion-runtime.md stage S10).
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;

namespace InferenceWeb.Tests;

public sealed class SharedDiffusionSamplingTests
{
    private static readonly float[] H3Grid = MiniMaxH3ImageSampling.ReleaseSigmas;

    private static SamplingRecipe H3Recipe => new()
    {
        Schedule = DiffusionSchedule.ComfySimpleOverGrid,
        Steps = 6,
        Grid = H3Grid,
        Heads = HeadPolicy.GridInterval,
    };

    private static uint[] Bits(float[] values) => values.Select(BitConverter.SingleToUInt32Bits).ToArray();

    [Fact]
    public void ComfySimpleOverTheH3GridIsBitExactWithComfyUiAtSixAndEightSteps()
    {
        // 1, 0.96765357, 0.88770843, 0.75, 0.53364283, 0.23261039, 0 (and the eight-step list), as float32 bits.
        uint[] six = { 0x3F800000, 0x3F77B825, 0x3F6340DC, 0x3F400000, 0x3F089CD1, 0x3E6E316B, 0x00000000 };
        uint[] eight = { 0x3F800000, 0x3F79C3BA, 0x3F6D7302, 0x3F5A68C9, 0x3F400000, 0x3F164AE7, 0x3EC3C8CE, 0x3E322ECD, 0x00000000 };
        Assert.Equal(six, Bits(H3Recipe.SigmasFor(6)));
        Assert.Equal(eight, Bits(H3Recipe.SigmasFor(8)));
        Assert.Equal(six, Bits(DiffusionSchedules.ComfySimpleOverGrid(H3Grid, 6)));
    }

    [Fact]
    public void ComfySimpleOverGridIsTheLegacyH3ScheduleAtEveryStepCount()
    {
        for (int steps = 1; steps <= 64; steps++)
            Assert.Equal(Bits(MiniMaxH3ImageSampling.Schedule(H3Grid, steps)), Bits(H3Recipe.SigmasFor(steps)));
    }

    [Fact]
    public void GridIntervalHeadsAreTheLegacyIntervalsAndOneHeadAStepAtSix()
    {
        SamplingRecipe recipe = H3Recipe;
        float[] six = recipe.SigmasFor(6);
        for (int i = 0; i < 6; i++)
            Assert.Equal((i, i + 1), recipe.HeadsFor(i, six[i], six[i + 1]));
        for (int steps = 1; steps <= 20; steps++)
        {
            float[] s = recipe.SigmasFor(steps);
            for (int i = 0; i < steps; i++)
                Assert.Equal(MiniMaxH3ImageSampling.Interval(H3Grid, s[i], s[i + 1]), recipe.HeadsFor(i, s[i], s[i + 1]));
        }
    }

    [Fact]
    public void ByStepHeadsTakeHeadIAtStepIAndRefuseAStepWithoutAHead()
    {
        SamplingRecipe recipe = H3Recipe with { Heads = HeadPolicy.ByStep };
        Assert.Equal((0, 1), recipe.HeadsFor(0, 1f, 0.923f));
        Assert.Equal((1, 2), recipe.HeadsFor(1, 0.923f, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => recipe.HeadsFor(6, 0.1f, 0f));
        Assert.Equal((0, 0), (H3Recipe with { Heads = HeadPolicy.None }).HeadsFor(3, 0.5f, 0.2f));
    }

    [Theory]
    [InlineData(4, 12f)]
    [InlineData(8, 6f)]
    [InlineData(20, 3f)]
    [InlineData(1, 1f)]
    public void LinspaceShiftIsTheTurboGridBitForBit(int steps, float shift)
    {
        var recipe = new SamplingRecipe { Schedule = DiffusionSchedule.LinspaceShift, Steps = steps, Shift = shift };
        Assert.Equal(Bits(MiniMaxH3Scheduler.BuildUniformSigmas(steps, shift)), Bits(recipe.SigmasFor(steps)));
    }

    [Fact]
    public void AStreamTakesItsOwnShiftOverTheSameLinspace()
    {
        var recipe = new SamplingRecipe
        {
            Schedule = DiffusionSchedule.LinspaceShift, Steps = 4, Shift = 6f,
            StreamShifts = new Dictionary<string, float> { ["audio"] = 3f },
        };
        recipe.Validate();
        Assert.Equal(Bits(MiniMaxH3Scheduler.BuildUniformSigmas(4, 3f)), Bits(recipe.StreamSigmas("audio", 4)));
        // The H3 video recipe derives its audio grid by mapping each video sigma; it is the same grid.
        float[] mapped = recipe.SigmasFor(4).Select((s, i) => i == 4 ? 0f : MiniMaxH3Scheduler.MapSigma(s, 6f, 3f)).ToArray();
        Assert.Equal(mapped, recipe.StreamSigmas("audio", 4), new Tolerance(1e-6f));
        Assert.Throws<ArgumentException>(() => recipe.StreamSigmas("video2", 4));
    }

    [Fact]
    public void AnExplicitListFixesItsStepCount()
    {
        float[] pdmd = { 1f, 12f / 13f, 0f };
        var recipe = new SamplingRecipe { Schedule = DiffusionSchedule.Explicit, Steps = 2, Sigmas = pdmd };
        recipe.Validate();
        Assert.Equal(2, recipe.ResolveSteps(0));
        Assert.Equal(2, recipe.ResolveSteps(2));
        Assert.Throws<ArgumentException>(() => recipe.ResolveSteps(6));
        Assert.Equal(Bits(pdmd), Bits(recipe.SigmasFor(2)));
        Assert.NotSame(pdmd, recipe.SigmasFor(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => recipe.ResolveSteps(-1));
        // A grid recipe takes any count.
        Assert.Equal(9, H3Recipe.ResolveSteps(9));
    }

    [Fact]
    public void ValidationRefusesMalformedRecipes()
    {
        Assert.Throws<ArgumentException>(() => new SamplingRecipe { Schedule = DiffusionSchedule.Explicit, Steps = 2, Sigmas = new[] { 1f, 0.5f } }.Validate());
        Assert.Throws<ArgumentException>(() => new SamplingRecipe { Schedule = DiffusionSchedule.Explicit, Steps = 2, Sigmas = new[] { 1f, 0.5f, 0.1f } }.Validate());
        Assert.Throws<ArgumentException>(() => new SamplingRecipe { Schedule = DiffusionSchedule.Explicit, Steps = 2, Sigmas = new[] { 1f, 1f, 0f } }.Validate());
        Assert.Throws<ArgumentException>(() => new SamplingRecipe { Schedule = DiffusionSchedule.ComfySimpleOverGrid, Steps = 6 }.Validate());
        Assert.Throws<ArgumentException>(() => new SamplingRecipe { Schedule = DiffusionSchedule.LinspaceShift, Steps = 4, Shift = 0f }.Validate());
        Assert.Throws<ArgumentException>(() => new SamplingRecipe { Schedule = DiffusionSchedule.LinspaceShift, Steps = 4, Heads = HeadPolicy.ByStep }.Validate());
        Assert.Throws<ArgumentException>(() => new SamplingRecipe { Schedule = DiffusionSchedule.LinspaceShift, Steps = 0 }.Validate());
        H3Recipe.Validate();
    }

    [Fact]
    public void EulerFlowIsTheLegacyHostStepBitForBit()
    {
        var random = new Random(17);
        float[] x = Enumerable.Range(0, 4099).Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
        float[] v = Enumerable.Range(0, x.Length).Select(_ => (float)(random.NextDouble() * 6 - 3)).ToArray();
        float[] legacy = (float[])x.Clone(), shared = (float[])x.Clone();
        float[] sigmas = H3Recipe.SigmasFor(6);
        var sampler = new EulerFlowSampler();
        for (int i = 0; i < 6; i++)
        {
            float sigma = sigmas[i], next = sigmas[i + 1];
            // the legacy MiniMaxH3ImageModel.Generate's update (deleted in stage S10).
            float dt = next - sigma;
            for (long k = 0; k < legacy.Length; k++) legacy[k] += v[k] * dt;
            sampler.Step(shared, v, sigma, next);
        }
        Assert.Equal(Bits(legacy), Bits(shared));
        Assert.Throws<ArgumentException>(() => sampler.Step(new float[3], new float[2], 1f, 0.5f));
    }

    [Fact]
    public void TheDenoisedEstimateIsTheFlowsCleanLatent()
    {
        // x - sigma * v in float32, element by element (Qwen-Image-2.1's retired pipeline previewed exactly this).
        var random = new Random(3);
        float[] x = Enumerable.Range(0, 257).Select(_ => (float)random.NextDouble()).ToArray();
        float[] v = Enumerable.Range(0, 257).Select(_ => (float)random.NextDouble() - 0.5f).ToArray();
        Assert.Equal(Bits(x.Zip(v, (a, b) => a - 0.37f * b).ToArray()), Bits(EulerFlowSampler.Denoised(x, v, 0.37f)));
        Assert.Throws<ArgumentOutOfRangeException>(() => EulerFlowSampler.Denoised(x, v, 1.5f));
    }

    [Fact]
    public void ASpeedContractFixesTheSamplingAndRefusesContradictions()
    {
        var contract = new SamplingContract
        {
            Name = "pdmd-2step", Sigmas = new[] { 1f, 12f / 13f, 0f }, Heads = HeadPolicy.ByStep, MinGenerationArea = 1024 * 1024,
        };
        SamplingRecipe recipe = contract.Apply(H3Recipe);
        Assert.Equal(DiffusionSchedule.Explicit, recipe.Schedule);
        Assert.Equal(2, recipe.Steps);
        Assert.Equal(HeadPolicy.ByStep, recipe.Heads);
        Assert.Same(H3Grid, recipe.Grid);
        Assert.Equal(new[] { (0, 1), (1, 2) }, new[] { recipe.HeadsFor(0, 1f, 12f / 13f), recipe.HeadsFor(1, 12f / 13f, 0f) });
        contract.Validate(0, 0f);
        contract.Validate(2, 1f);
        Assert.Contains("pdmd-2step samples exactly 2 steps", Assert.Throws<ArgumentException>(() => contract.Validate(6, 0f)).Message);
        Assert.Contains("guidance 1", Assert.Throws<ArgumentException>(() => contract.Validate(0, 4f)).Message);
    }

    [Fact]
    public void AnAdapterSetTakesStyleAdaptersAndAtMostOneSpeedAdapter()
    {
        var speed = new DiffusionAdapter { Name = "pdmd", Contract = new SamplingContract { Name = "pdmd", Sigmas = new[] { 1f, 0.5f, 0f } } };
        var style = new DiffusionAdapter { Name = "realem", Scale = 0.8f };
        var set = new AdapterSet(new[] { style, speed, style with { Name = "other" } });
        Assert.Same(speed.Contract, set.Contract);
        Assert.Equal(3, set.Adapters.Count);
        Assert.Null(new AdapterSet(new[] { style }).Contract);
        Assert.True(AdapterSet.Empty.IsEmpty);
        var two = Assert.Throws<ArgumentException>(() => new AdapterSet(new[] { speed, speed with { Name = "lightning" } }));
        Assert.Contains("Two speed adapters (pdmd, lightning)", two.Message);
        Assert.Throws<ArgumentException>(() => new AdapterSet(new[] { style with { Scale = float.NaN } }));
    }

    private sealed class Tolerance(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float a, float b) => Math.Abs(a - b) <= tolerance;
        public int GetHashCode(float v) => 0;
    }
}
