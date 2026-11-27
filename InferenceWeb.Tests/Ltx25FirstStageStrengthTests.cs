using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

/// <summary>The first pass's frame-zero guide strength (the publisher's .7 for a shorthand first image) as the family applies it:
/// only the first pass's frame-zero guide changes; the refinement and the other guides keep their strengths.</summary>
public sealed class Ltx25FirstStageStrengthTests
{
    private static List<Ltx25EncodedImage> Override(IReadOnlyList<Ltx25EncodedImage> images, float strength)
        => Ltx25Family.WithFirstStageImageStrength(images, strength);

    [Theory]
    [InlineData(1f)]
    [InlineData(.85f)]
    public void FirstStageOverrideAllowsDenoisingWithoutChangingRefinementOrOtherGuides(float requestedStrength)
    {
        float[] firstImage = Enumerable.Repeat(.25f, 128).ToArray();
        var original = new List<Ltx25EncodedImage> { new(firstImage, 0, requestedStrength), new(new float[128], 16, .4f) };
        List<Ltx25EncodedImage> firstPass = Override(original, .7f);
        Assert.Equal(requestedStrength, original[0].Strength);
        Assert.Same(firstImage, firstPass[0].Latent);
        Assert.Same(original[1], firstPass[1]);

        // Exercise the real per-token conditioning, including a prior/noise
        // refinement blend, rather than treating strength as metadata only.
        var settings = new Ltx25GenerationSettings(32, 32, 17, 24, 1, 1, 3, 18, 42);
        Ltx25DenoiserInput Input(float value) => new()
        {
            Video = Enumerable.Repeat(value, 4 * 128).ToArray(), VideoTokens = 4,
            Audio = new float[18 * 128], AudioTokens = 18,
            VideoContext = [0], VideoContextTokens = 1, AudioContext = [0], AudioContextTokens = 1,
            VideoPositions = Ltx25Geometry.VideoPositions(3, 1, 1, 24), AudioPositions = Ltx25Geometry.AudioPositions(18),
            VideoSigma = 1, AudioSigma = 1,
            FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
        };
        var first = Ltx25ImageConditioning.Apply(Input(1), settings, firstPass);
        Assert.InRange(Math.Abs(first.VideoDenoiseMask[0] - .3f), 0, 1e-7f);
        Assert.InRange(Math.Abs(first.Video[0] - .475f), 0, 1e-7f);

        var refinement = Ltx25ImageConditioning.Apply(Input(.8f), settings, original,
            Enumerable.Repeat(1f, 4 * 128).ToArray(), Ltx25GenerationSettings.RefinementNoiseScale);
        Assert.InRange(Math.Abs(refinement.VideoDenoiseMask[0] - (1 - requestedStrength)), 0, 1e-7f);
        Assert.InRange(Math.Abs(refinement.Video[0] - (.25f + (.8f - .25f) * (1 - requestedStrength))), 0, 1e-7f);
        Assert.Equal(first.VideoDenoiseMask[3], refinement.VideoDenoiseMask[3]);
    }

    [Theory]
    [InlineData(-.1f)]
    [InlineData(1.1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void InvalidStrengthIsRejected(float strength)
        => Assert.Throws<ArgumentOutOfRangeException>(() => Override([new(new float[128], 0)], strength));

    [Fact]
    public void OverrideRequiresFirstFrameGuide()
        => Assert.Throws<ArgumentException>(() => Override([new(new float[128], 8)], .7f));
}
