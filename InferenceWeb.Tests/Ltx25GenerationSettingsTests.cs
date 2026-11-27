using TensorSharp.Models.Ltx25;
using TensorSharp.Models.Video;

namespace InferenceWeb.Tests;

public sealed class Ltx25GenerationSettingsTests
{
    [Fact]
    public void DefaultRequestUsesTheInteractiveSeventeenFramePreview()
    {
        var settings = Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Seed = 42 });
        Assert.Equal((1024, 1024, 17, 24), (settings.Width, settings.Height, settings.Frames, settings.Fps));
        Assert.Equal(3072, settings.VideoTokens);
        Assert.Equal(18, settings.AudioTokens);
    }

    [Fact]
    public void RequestsSampleThePublishersTwoStageRecipeAtHalfTheOutputSize()
    {
        var settings = Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Width = 1024, Height = 768, Frames = 97, Seed = 42 });
        Assert.True(settings.TwoStage);
        Assert.Equal(11, settings.TotalSteps);
        var first = settings.HalfResolution();
        Assert.Equal((512, 384, 16, 12, 97, 13), (first.Width, first.Height, first.LatentWidth, first.LatentHeight, first.Frames, first.LatentFrames));
        Assert.Equal((settings.Seed, settings.AudioTokens, settings.Fps, settings.Ancestral), (first.Seed, first.AudioTokens, first.Fps, first.Ancestral));
        Assert.Equal(new[] { .909375f, .725f, .421875f, 0f }, Ltx25Scheduler.RefinementSigmas());
        Assert.Equal(Ltx25Scheduler.RefinementSigmas()[0], Ltx25GenerationSettings.RefinementNoiseScale);
        // Stage two draws its own initial noise and step noise.
        Assert.NotEqual(settings.InitialNoise().Video, settings.InitialNoise(stream: 1).Video);
        Assert.NotEqual(settings.AncestralNoise(0, 128, 16).Video,
            settings.AncestralNoise(0, 128, 16, Ltx25GenerationSettings.AncestralStage2NoiseSeedOffset).Video);
    }

    [Theory]
    [InlineData(512, 480)]
    [InlineData(544, 512)]
    public void OutputSizesMustHalveOntoTheLatentGrid(int width, int height)
    {
        var error = Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Width = width, Height = height }));
        Assert.Contains("multiples of 64", error.Message);
    }

    [Theory]
    [InlineData(17, 24, 18)]
    [InlineData(33, 24, 34)]
    [InlineData(9, 20, 11)]
    [InlineData(17, 20, 21)]
    [InlineData(17, 10, 42)]
    [InlineData(9, 10, 22)]
    public void AudioGeometryFollowsOriginalDurationRounding(int frames, int fps, int audioTokens)
    {
        var settings = Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Width = 64, Height = 128, Frames = frames, Fps = fps, Seed = 42 });
        Assert.Equal(audioTokens, settings.AudioTokens);
        Assert.Equal((frames - 1) / 8 + 1, settings.LatentFrames);
        Assert.Equal(2, settings.LatentWidth);
        Assert.Equal(4, settings.LatentHeight);
    }

    [Theory]
    [InlineData(16, 64, 64)]
    [InlineData(17, 63, 64)]
    [InlineData(17, 64, -1)]
    public void InvalidExplicitGeometryFailsBeforeLoadingNetworks(int frames, int width, int height)
    {
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Width = width, Height = height, Frames = frames }));
    }

    [Fact]
    public void UnsupportedRecipeOrConditioningDoesNotSilentlyGenerateADifferentRequest()
    {
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Steps = 4 }));
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { CfgScale = 3 }));
        Assert.Throws<ArgumentException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Sampler = "unipc" }));
        Assert.NotNull(Ltx25GenerationSettings.Resolve(new VideoGenerationParams { ImagePath = "image.png" }));
        Assert.Throws<NotSupportedException>(() => Ltx25GenerationSettings.Resolve(new VideoGenerationParams { ReferenceAudioPaths = ["reference.wav"] }));
    }

    [Theory]
    // The publisher's DistilledPipeline samples LTX 2.5+ with ancestral Euler; euler is an explicit diagnostic.
    [InlineData(null, true)]
    [InlineData("", true)]
    [InlineData("euler_ancestral", true)]
    [InlineData("Euler_Ancestral", true)]
    [InlineData("euler", false)]
    public void SamplerDefaultsToThePublishersAncestralRecipe(string? sampler, bool ancestral)
        => Assert.Equal(ancestral, Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Sampler = sampler, Seed = 42 }).Ancestral);

    [Fact]
    public void AncestralNoiseIsDeterministicIndependentPerStepAndApartFromInitialNoise()
    {
        var settings = Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Width = 256, Height = 256, Frames = 17, Seed = 42 });
        int video = settings.VideoTokens * 128, audio = settings.AudioTokens * 128;
        var first = settings.AncestralNoise(0, video, audio);
        var again = settings.AncestralNoise(0, video, audio);
        var second = settings.AncestralNoise(1, video, audio);
        Assert.Equal(first.Video, again.Video);
        Assert.Equal(first.Audio, again.Audio);
        Assert.Equal((video, audio), (first.Video.Length, first.Audio.Length));
        var initial = settings.InitialNoise();
        // Distinct draws are uncorrelated standard normals, not shifted copies of one sequence.
        Assert.True(Math.Abs(Correlation(first.Video, second.Video)) < .05);
        Assert.True(Math.Abs(Correlation(first.Video, initial.Video)) < .05);
        Assert.True(Math.Abs(Correlation(first.Video[..^1], first.Video[1..])) < .05);
        double mean = first.Video.Average(value => (double)value);
        double variance = first.Video.Average(value => (value - mean) * (value - mean));
        Assert.InRange(mean, -.05, .05);
        Assert.InRange(variance, .9, 1.1);
        var otherSeed = Ltx25GenerationSettings.Resolve(new VideoGenerationParams { Width = 256, Height = 256, Frames = 17, Seed = 43 });
        Assert.NotEqual(first.Video, otherSeed.AncestralNoise(0, video, audio).Video);
    }

    private static double Correlation(float[] x, float[] y)
    {
        double mx = x.Average(value => (double)value), my = y.Average(value => (double)value), sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < x.Length; ++i)
        {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx) * (x[i] - mx);
            syy += (y[i] - my) * (y[i] - my);
        }
        return sxy / Math.Sqrt(sxx * syy);
    }
}
