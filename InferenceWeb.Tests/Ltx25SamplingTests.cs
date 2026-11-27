using System.Text.Json;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.SharedDiffusion;

namespace InferenceWeb.Tests;

/// <summary>LTX 2.5's samplers on the diffusion runtime (<see cref="EulerDenoisedSampler"/>, <see cref="EulerAncestralSampler"/>)
/// against the unchanged official Euler and euler_ancestral loops, through the family's step inputs.</summary>
public sealed class Ltx25SamplingTests
{
    private static Ltx25DenoiserInput Input(float[]? video = null, float[]? audio = null,
        float[]? videoTimesteps = null, float[]? audioTimesteps = null, float[]? keyframeMask = null) => new()
    {
        Video = video ?? [1f, 2f], VideoTokens = 1,
        Audio = audio ?? [3f, 4f], AudioTokens = 1,
        VideoContext = [5f, 6f], VideoContextTokens = 1,
        AudioContext = [7f, 8f], AudioContextTokens = 1,
        VideoPositions = Ltx25Geometry.VideoPositions(1, 1, 1, 24),
        AudioPositions = Ltx25Geometry.AudioPositions(1),
        FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
        VideoSigma = .25f, AudioSigma = .5f,
        VideoTimesteps = videoTimesteps, AudioTimesteps = audioTimesteps, KeyframeMask = keyframeMask,
    };

    private static float[] Values(JsonElement value) => value.EnumerateArray().Select(item => item.GetSingle()).ToArray();

    private static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 3e-7f,
                $"Element {i}: expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Fact]
    public void JointEulerMatchesPinnedOriginalFloat32X0AndStepArithmetic()
    {
        using Stream stream = typeof(Ltx25SamplingTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Ltx25.euler-float32.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        JsonElement root = fixture.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", root.GetProperty("source_revision").GetString());
        float[] initialVideo = Values(root.GetProperty("initial_video"));
        float[] initialAudio = Values(root.GetProperty("initial_audio"));
        var initial = Input(initialVideo, initialAudio);
        var progress = new List<(int Step, int Total)>();
        int called = 0;
        var output = Ltx25SamplerLoop.Run(input =>
        {
            JsonElement[] expected = root.GetProperty("steps").EnumerateArray().ToArray();
            Close(called == 0 ? initialVideo : Values(expected[called - 1].GetProperty("video")), input.Video);
            Close(called == 0 ? initialAudio : Values(expected[called - 1].GetProperty("audio")), input.Audio);
            Assert.NotSame(initialVideo, input.Video);
            Assert.NotSame(initialAudio, input.Audio);
            Assert.Same(initial.VideoContext, input.VideoContext);
            Assert.Same(initial.AudioContext, input.AudioContext);
            Assert.Same(initial.VideoPositions, input.VideoPositions);
            Assert.Same(initial.AudioPositions, input.AudioPositions);
            Assert.Equal(initial.FrequencyPrecision, input.FrequencyPrecision);
            Assert.Equal(root.GetProperty("sigmas")[called].GetSingle(), input.VideoSigma);
            Assert.Equal(input.VideoSigma, input.AudioSigma);
            Assert.Null(input.VideoTimesteps);
            JsonElement step = expected[called++];
            return new(Values(step.GetProperty("video_velocity")), Values(step.GetProperty("audio_velocity")), 2);
        }, initial, Values(root.GetProperty("sigmas")), onStep: (step, total) => progress.Add((step, total)));

        JsonElement last = root.GetProperty("steps")[2];
        Close(Values(last.GetProperty("video")), output.Video);
        Close(Values(last.GetProperty("audio")), output.Audio);
        Assert.Equal(3, output.Steps);
        Assert.Equal(3, called);
        Assert.Equal(new[] { (1, 3), (2, 3), (3, 3) }, progress);
        Close(Values(root.GetProperty("initial_video")), initialVideo);
        Close(Values(root.GetProperty("initial_audio")), initialAudio);
    }

    [Fact]
    public void AncestralEulerMatchesPinnedOriginalLoopWithConditioningMasks()
    {
        using Stream stream = typeof(Ltx25SamplingTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Ltx25.euler-ancestral-float32.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        JsonElement root = fixture.RootElement;
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", root.GetProperty("source_revision").GetString());
        Assert.Equal(EulerAncestralSampler.Eta, root.GetProperty("eta").GetSingle());
        Assert.Equal(EulerAncestralSampler.NoiseScale, root.GetProperty("s_noise").GetSingle());
        JsonElement[] steps = root.GetProperty("steps").EnumerateArray().ToArray();
        float[] sigmas = Values(root.GetProperty("sigmas"));
        Assert.Equal(Ltx25Scheduler.DistilledSigmas(), sigmas);
        var initial = new Ltx25DenoiserInput
        {
            Video = Values(root.GetProperty("initial_video")), VideoTokens = root.GetProperty("video_tokens").GetInt32(),
            Audio = Values(root.GetProperty("initial_audio")), AudioTokens = root.GetProperty("audio_tokens").GetInt32(),
            VideoContext = [5f, 6f], VideoContextTokens = 1, AudioContext = [7f, 8f], AudioContextTokens = 1,
            VideoPositions = Ltx25Geometry.VideoPositions(1, 1, 3, 24), AudioPositions = Ltx25Geometry.AudioPositions(2),
            FrequencyPrecision = Ltx25FrequencyPrecision.Float64, VideoSigma = 1, AudioSigma = 1,
            VideoDenoiseMask = Values(root.GetProperty("video_mask")), CleanVideo = Values(root.GetProperty("clean_video")),
        };
        int called = 0;
        var drawn = new List<int>();
        var output = Ltx25SamplerLoop.Run(input =>
        {
            JsonElement step = steps[called];
            // Each prediction sees exactly the original loop's latents, so every step is checked.
            Close(Values(step.GetProperty("video")), input.Video);
            Close(Values(step.GetProperty("audio")), input.Audio);
            Assert.Equal(step.GetProperty("sigma").GetSingle(), input.VideoSigma);
            float[] mask = Values(root.GetProperty("video_mask"));
            Assert.Equal(mask.Select(value => value * input.VideoSigma).ToArray(), input.VideoTimesteps);
            called++;
            return new(Values(step.GetProperty("video_velocity")), Values(step.GetProperty("audio_velocity")), 2);
        }, initial, sigmas, step =>
        {
            drawn.Add(step);
            return (Values(steps[step].GetProperty("video_noise")), Values(steps[step].GetProperty("audio_noise")));
        });
        Close(Values(root.GetProperty("final_video")), output.Video);
        Close(Values(root.GetProperty("final_audio")), output.Audio);
        Assert.Equal(8, called);
        // Seven renoised steps; the terminal step returns X0 without drawing.
        Assert.Equal(Enumerable.Range(0, 7), drawn);
        // The strength-1 first frame stays exactly clean throughout.
        Assert.Equal(Values(root.GetProperty("clean_video"))[..2], output.Video[..2]);
    }

    [Fact]
    public void TheRecipesSamplersAreTheConditionedEulerSamplers()
    {
        SamplingRecipe Recipe(DiffusionSamplerKind kind) => new()
        {
            Schedule = DiffusionSchedule.Explicit, Steps = 8, Sigmas = Ltx25Scheduler.DistilledSigmas(), Sampler = kind,
        };
        Assert.IsType<EulerAncestralSampler>(Recipe(DiffusionSamplerKind.EulerAncestral).CreateSampler());
        Assert.IsType<EulerDenoisedSampler>(Recipe(DiffusionSamplerKind.EulerDenoised).CreateSampler());
        Assert.True(new EulerAncestralSampler().NeedsNoise);
        Assert.False(new EulerDenoisedSampler().NeedsNoise);
    }

    [Fact]
    public void AncestralNoiseMustCoverTheStream()
    {
        var sampler = new EulerAncestralSampler();
        Assert.Throws<ArgumentException>(() => sampler.Step(new float[2], new float[2], 1f, .5f, null));
        Assert.Throws<ArgumentException>(() => sampler.Step(new float[2], new float[2], 1f, .5f, new SamplerStreamInputs { Noise = [0f] }));
        Assert.Throws<InvalidOperationException>(() => sampler.Step(new float[2], new float[2], 1f, .5f,
            new SamplerStreamInputs { Noise = [0f, float.NaN] }));
        // The terminal step returns the x0 estimate and reads no noise.
        float[] x = [1f, 2f];
        sampler.Step(x, [.5f, .25f], .5f, 0f, null);
        Assert.Equal(new[] { 1f - .5f * .5f, 2f - .25f * .5f }, x);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void InvalidVelocityCannotProduceSuccessfulSampling(bool dimensions)
    {
        foreach (IConditionedSampler sampler in new IConditionedSampler[] { new EulerDenoisedSampler(), new EulerAncestralSampler() })
        {
            float[] velocity = dimensions ? [0f] : [float.NaN, 0f];
            if (dimensions) Assert.Throws<ArgumentException>(() => sampler.Step(new float[2], velocity, 1f, 0f, null));
            else Assert.Throws<InvalidOperationException>(() => sampler.Step(new float[2], velocity, 1f, 0f, null));
        }
    }

    [Fact]
    public void MaskAndCleanLatentsComeTogetherAndCoverTheStream()
    {
        var sampler = new EulerDenoisedSampler();
        Assert.Throws<ArgumentException>(() => sampler.Step(new float[4], new float[4], 1f, 0f,
            new SamplerStreamInputs { DenoiseMask = [1f, 1f] }));
        Assert.Throws<ArgumentException>(() => sampler.Step(new float[4], new float[4], 1f, 0f,
            new SamplerStreamInputs { Clean = new float[4] }));
        Assert.Throws<ArgumentException>(() => sampler.Step(new float[4], new float[4], 1f, 0f,
            new SamplerStreamInputs { DenoiseMask = [1f, 1f, 1f], Clean = new float[4] }));
        Assert.Throws<ArgumentException>(() => sampler.Step(new float[4], new float[4], 1f, 0f,
            new SamplerStreamInputs { DenoiseMask = [1f, 1f], Clean = new float[2] }));
    }
}
