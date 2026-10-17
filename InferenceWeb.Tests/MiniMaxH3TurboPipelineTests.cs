// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The pipeline's side of a MiniMax-H3 LoRA plug-in's sampling contract, against the shipped
// Turbo configs (config/lora/minimax-h3-fl2v-turbo-{8,4}step.json): which requests a loaded
// plug-in refuses (MiniMaxH3Pipeline.ValidateSamplingParameters), which steps, shift and sigma
// grid a clip then samples with (ResolveSampling), the Reference-mode refusal and the 768p note.
// Nothing here loads weights: the recipe comes from the configs alone.
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3TurboPipelineTests
{
    private const string Name8 = "LightX2V Turbo 8-step (FL2V v1.0, 768p)";
    private const string Name4 = "LightX2V Turbo 4-step (FL2V v1.2, 768p)";
    private const string Weights8 = MiniMaxH3LoraFixtures.Weights8, Weights4 = MiniMaxH3LoraFixtures.Weights4;

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.True(dir != null, "The tests run from a source checkout; config/lora/ must be reachable.");
        return dir!.FullName;
    }

    private static MiniMaxH3SamplingRecipe Recipe(int steps) =>
        MiniMaxH3LoraConfig.Load(Path.Combine(FindRepoRoot(), "config", "lora", $"minimax-h3-fl2v-turbo-{steps}step.json")).Recipe;

    private static void AssertGrid(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6, $"sigma[{i}] = {actual[i]:R}, expected {expected[i]:R}");
    }

    private static string StepsRefusal(string name, string weights, int steps, int requested) =>
        $"{name} ({weights}) samples exactly {steps} steps on its own schedule; {requested} steps contradicts it. " +
        "Leave the step count unset (--diffusion-steps on the CLI, --video-steps on the server, \"steps\" in a request) " +
        $"or set it to {steps}.";

    private const string CfgRefusal = "MiniMax-H3 is CFG-distilled and requires guidance 1.0. Use --cfg 1.0 (the default).";

    // ---- what a loaded plug-in accepts ----------------------------------------------------

    [Theory]
    [InlineData(8, 0, 0f, null, 0f)]
    [InlineData(8, 8, 6f, "euler", 1f)]
    [InlineData(8, 0, 0f, "", 0f)]
    [InlineData(8, 0, 0f, " Euler ", 0f)]
    [InlineData(4, 4, 0f, null, 1f)]
    [InlineData(4, 0, 6f, "euler", 0f)]
    public void ARequestThatLeavesTheRecipeAloneOrRepeatsItIsAccepted(int recipeSteps, int steps, float shift, string? sampler, float cfg)
    {
        var p = new VideoGenerationParams { Steps = steps, FlowShift = shift, Sampler = sampler, CfgScale = cfg };
        MiniMaxH3Pipeline.ValidateSamplingParameters(p, Recipe(recipeSteps));
        // Nothing is overridden: the request still says what it said.
        Assert.Equal(steps, p.Steps);
        Assert.Equal(shift, p.FlowShift);
        Assert.Equal(sampler, p.Sampler);
    }

    [Theory]
    [InlineData(8, 20)]
    [InlineData(8, 4)]
    [InlineData(8, 1)]
    [InlineData(4, 8)]
    [InlineData(4, 20)]
    public void ADifferentStepCountIsRefusedWithTheCanonicalText(int recipeSteps, int requested)
    {
        var e = Assert.Throws<ArgumentException>(() => MiniMaxH3Pipeline.ValidateSamplingParameters(
            new VideoGenerationParams { Steps = requested }, Recipe(recipeSteps)));
        Assert.Equal(recipeSteps == 8
            ? StepsRefusal(Name8, Weights8, 8, requested)
            : StepsRefusal(Name4, Weights4, 4, requested), e.Message);
    }

    [Theory]
    [InlineData(12f)]
    [InlineData(3f)]
    [InlineData(6.5f)]
    public void ADifferentFlowShiftIsRefused(float shift)
    {
        var e = Assert.Throws<ArgumentException>(() => MiniMaxH3Pipeline.ValidateSamplingParameters(
            new VideoGenerationParams { FlowShift = shift }, Recipe(8)));
        Assert.Equal(
            $"{Name8} ({Weights8}) samples with video flow shift 6 (audio 3); flow shift " +
            $"{shift.ToString(System.Globalization.CultureInfo.InvariantCulture)} contradicts it. " +
            "Leave it unset (--flow-shift, \"flowShift\") or set it to 6.", e.Message);
    }

    [Theory]
    [InlineData("unipc")]
    [InlineData("euler_ancestral")]
    public void AnotherSamplerIsRefused(string sampler)
    {
        var e = Assert.Throws<ArgumentException>(() => MiniMaxH3Pipeline.ValidateSamplingParameters(
            new VideoGenerationParams { Sampler = sampler }, Recipe(4)));
        Assert.Equal(
            $"{Name4} ({Weights4}) samples with Euler; sampler '{sampler}' contradicts it. " +
            "Leave it unset (--sampler, \"sampler\") or pass euler.", e.Message);
    }

    [Theory]
    [InlineData(2f)]
    [InlineData(float.NaN)]
    public void GuidanceOtherThanOneIsTheBaseRefusal(float cfg)
    {
        var e = Assert.Throws<ArgumentException>(() => MiniMaxH3Pipeline.ValidateSamplingParameters(
            new VideoGenerationParams { CfgScale = cfg }, Recipe(8)));
        Assert.StartsWith(CfgRefusal, e.Message);
    }

    [Fact]
    public void NegativeStepsAreTheBaseRefusalEvenUnderARecipe()
    {
        var e = Assert.Throws<ArgumentException>(() => MiniMaxH3Pipeline.ValidateSamplingParameters(
            new VideoGenerationParams { Steps = -1 }, Recipe(8)));
        Assert.StartsWith("MiniMax-H3 steps must be positive, or 0 for the default.", e.Message);
    }

    [Theory]
    [InlineData(20, 12f, null)]
    [InlineData(8, 6f, "unipc")]
    [InlineData(4, 0f, null)]
    public void WithoutAPlugInTheBaseChecksAreUnchanged(int steps, float shift, string? sampler)
    {
        var p = new VideoGenerationParams { Steps = steps, FlowShift = shift, Sampler = sampler };
        MiniMaxH3Pipeline.ValidateSamplingParameters(p, null);
        MiniMaxH3Pipeline.ValidateSamplingParameters(p);
        var e = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3Pipeline.ValidateSamplingParameters(new VideoGenerationParams { CfgScale = 3f }, null));
        Assert.StartsWith(CfgRefusal, e.Message);
    }

    // ---- what a clip samples with -------------------------------------------------------

    [Theory]
    [InlineData(0, 0f)]
    [InlineData(8, 6f)]
    public void EightStepPlugIn_SamplesItsLinspaceGridAtShift6(int requestSteps, float requestShift)
    {
        var (steps, shift, sigmas) = MiniMaxH3Pipeline.ResolveSampling(
            new VideoGenerationParams { Steps = requestSteps, FlowShift = requestShift }, Recipe(8));
        Assert.Equal(8, steps);
        Assert.Equal(6f, shift);
        // TimeSnrShift(6, 1 - i/8), endpoint-inclusive, then 0.
        AssertGrid(new[] { 1, 42 / 43.0, 18 / 19.0, 10 / 11.0, 6 / 7.0, 18 / 23.0, 2 / 3.0, 6 / 13.0, 0 }, sigmas);
    }

    [Fact]
    public void FourStepPlugIn_SamplesItsLinspaceGridAtShift6()
    {
        var (steps, shift, sigmas) = MiniMaxH3Pipeline.ResolveSampling(new VideoGenerationParams(), Recipe(4));
        Assert.Equal(4, steps);
        Assert.Equal(6f, shift);
        AssertGrid(new[] { 1, 18 / 19.0, 6 / 7.0, 2 / 3.0, 0 }, sigmas);
    }

    [Theory]
    [InlineData(0, 0f, 20, 12f)]
    [InlineData(8, 6f, 8, 6f)]
    [InlineData(30, 0f, 30, 12f)]
    public void WithoutAPlugIn_TheRequestOrTheBaseDefaultsUseSdCppsGrid(int requestSteps, float requestShift, int steps, float shift)
    {
        var resolved = MiniMaxH3Pipeline.ResolveSampling(
            new VideoGenerationParams { Steps = requestSteps, FlowShift = requestShift }, null);
        Assert.Equal(steps, resolved.Steps);
        Assert.Equal(shift, resolved.VideoShift);
        Assert.Equal(MiniMaxH3Scheduler.BuildSigmas(steps, shift), resolved.Sigmas);
        // Base 8 steps at shift 6 is NOT the Turbo grid: the plug-in is what selects linspace.
        if (steps == 8)
            Assert.NotEqual(Recipe(8).VideoSigmas(), resolved.Sigmas);
    }

    // ---- modes and canvas -----------------------------------------------------------------

    [Fact]
    public void AReferenceClipIsRefusedWithAPlugIn()
    {
        var e = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3Pipeline.RequirePluginMode(Recipe(8), MiniMaxH3Mode.Reference));
        Assert.Equal(
            $"{Name8} ({Weights8}) is a MiniMax-H3 FL2VA plug-in for text, first-frame and first/last-frame clips; " +
            "reference conditioning does not take it. Drop the plug-in for a reference clip.", e.Message);
    }

    [Theory]
    [InlineData(MiniMaxH3Mode.TextToVideo)]
    [InlineData(MiniMaxH3Mode.ImageToVideo)]
    [InlineData(MiniMaxH3Mode.FirstLastFrame)]
    public void KeyframeAndTextClipsTakeThePlugIn(MiniMaxH3Mode mode)
    {
        MiniMaxH3Pipeline.RequirePluginMode(Recipe(4), mode);
        MiniMaxH3Pipeline.RequirePluginMode(null, MiniMaxH3Mode.Reference);
        MiniMaxH3Pipeline.RequirePluginMode(null, mode);
    }

    [Theory]
    [InlineData(1344, 768, null)]
    [InlineData(768, 1344, null)]
    [InlineData(640, 384, "  [lora] " + Name8 + " was trained at 768p; this clip is 640x384 (short edge 384).")]
    [InlineData(1280, 704, "  [lora] " + Name8 + " was trained at 768p; this clip is 1280x704 (short edge 704).")]
    public void APlugInRunOffItsTrainingCanvasSaysSoOnce(int width, int height, string? expected)
    {
        var shape = new MiniMaxH3Shape { Width = width, Height = height };
        Assert.Equal(expected, MiniMaxH3Pipeline.DescribePluginCanvas(Recipe(8), shape));
        Assert.Null(MiniMaxH3Pipeline.DescribePluginCanvas(null, shape));
    }
}
