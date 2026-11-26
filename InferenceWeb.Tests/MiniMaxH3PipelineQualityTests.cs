// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;

namespace InferenceWeb.Tests;

public class MiniMaxH3PipelineQualityTests
{
    [Fact]
    public void AudioOnlyReferencesWorkWithoutATextEncoderVisionTower()
    {
        var references = new[]
        {
            new MiniMaxH3Requests.MiniMaxH3Reference { Kind = MiniMaxH3ReferenceKind.Audio },
            new MiniMaxH3Requests.MiniMaxH3Reference { Kind = MiniMaxH3ReferenceKind.Audio },
        };
        MiniMaxH3Requests.RequireReferenceVision(false, references);
    }

    [Theory]
    [InlineData(MiniMaxH3ReferenceKind.Image)]
    [InlineData(MiniMaxH3ReferenceKind.Video)]
    [InlineData(MiniMaxH3ReferenceKind.VideoAudio)]
    public void VisualReferencesStillRequireATextEncoderVisionTower(MiniMaxH3ReferenceKind visualKind)
    {
        var references = new[]
        {
            new MiniMaxH3Requests.MiniMaxH3Reference { Kind = MiniMaxH3ReferenceKind.Audio },
            new MiniMaxH3Requests.MiniMaxH3Reference { Kind = visualKind },
        };
        var error = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3Requests.RequireReferenceVision(false, references));
        Assert.Contains("vision tower", error.Message);
        MiniMaxH3Requests.RequireReferenceVision(true, references);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void DecoderFailuresAreRejectedBeforeMediaConversion(float invalid)
    {
        var pixels = new[] { 0.25f, invalid, 0.75f };
        var error = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3Requests.RequireFiniteOutput(pixels, "video VAE output"));
        Assert.Contains("video VAE output", error.Message);
        Assert.Contains("element 1", error.Message);
    }

    [Fact]
    public void DecoderInfinityWouldOtherwiseBecomeAValidSaturatedPixel()
    {
        var pixels = new[] { 0f, float.PositiveInfinity, float.NegativeInfinity };
        Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3Requests.RequireFiniteOutput(pixels, "video VAE output"));
        MiniMaxH3VideoVae.ToRgb(pixels, frames: 1, height: 1, width: 1);
        Assert.All(pixels, value => Assert.True(float.IsFinite(value)));
        Assert.Equal(1f, pixels[1]);
        Assert.Equal(0f, pixels[2]);
    }

    [Fact]
    public void FiniteVelocityCanOverflowTheFinalEulerUpdate()
    {
        var latent = new[] { float.MaxValue };
        var velocity = new[] { -float.MaxValue };
        MiniMaxH3Requests.RequireFinite(velocity, "video velocity", 7, 8, 118);
        MiniMaxH3Scheduler.EulerStep(latent, velocity, 1f, 0f);
        var error = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3Requests.RequireFinite(latent, "video latent", 7, 8, 118));
        Assert.Contains("video latent", error.Message);
        Assert.Contains("step 8/8", error.Message);
    }

    [Fact]
    public void FiniteDecoderValuesIncludingHighlightsRemainAccepted()
    {
        MiniMaxH3Requests.RequireFiniteOutput(new[] { -1000f, 0f, 1000f }, "video VAE output");
        MiniMaxH3Requests.RequireFiniteOutput(new[] { -1f, 0f, 1f }, "audio VAE output");
    }

    [Theory]
    [InlineData(-1f)]
    [InlineData(0.5f)]
    [InlineData(2f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void UnsupportedGuidanceIsRejectedBeforeLoadingWeights(float cfg)
    {
        var error = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3Requests.ValidateSamplingParameters(new VideoGenerationParams { CfgScale = cfg }));
        Assert.Contains("guidance 1.0", error.Message);
    }

    [Theory]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void InvalidFlowShiftIsRejectedBeforeLoadingWeights(float shift)
    {
        var error = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3Requests.ValidateSamplingParameters(new VideoGenerationParams { FlowShift = shift }));
        Assert.Contains("flow shift", error.Message);
    }

    [Fact]
    public void DefaultsAndCustomPositiveFlowShiftsRemainAccepted()
    {
        MiniMaxH3Requests.ValidateSamplingParameters(new VideoGenerationParams());
        MiniMaxH3Requests.ValidateSamplingParameters(new VideoGenerationParams { CfgScale = 1f, Steps = 8, FlowShift = 6f });
        var error = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3Requests.ValidateSamplingParameters(new VideoGenerationParams { Steps = -1 }));
        Assert.Contains("steps", error.Message);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void SchedulerRefusesAnUndefinedShift(float shift)
        => Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3Scheduler.BuildSigmas(8, shift));
}
