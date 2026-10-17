using System.Reflection;
using System.Text.Json;
using TensorSharp.Chat.Studio;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Studio;

namespace InferenceWeb.Tests;

public sealed class Ltx25ImageStrengthPolicyTests
{
    static readonly Type Pipeline = typeof(Ltx25Model).Assembly.GetType("TensorSharp.Models.Ltx25.Ltx25Pipeline", throwOnError: true)!;

    static float? FirstStageStrength(VideoGenerationParams request, bool twoStage = true,
        float? diagnosticOverride = null, IReadOnlyList<VideoImageCondition>? resolved = null)
    {
        var images = resolved ?? Ltx25ImageConditioning.Resolve(request, 17);
        var method = Pipeline.GetMethod("ResolveFirstStageImageStrength", BindingFlags.Static | BindingFlags.NonPublic)!;
        return (float?)method.Invoke(null, [request, twoStage, images, diagnosticOverride]);
    }

    static VideoGenerationParams Shorthand(string source, string? mode = null) => new()
    {
        Image = source == "image" ? new RgbImage(1, 1, [.1f, .2f, .3f]) : null,
        ImageBytes = source == "bytes" ? [1, 2, 3] : null,
        ImagePath = source == "path" ? "first.png" : null,
        Mode = mode,
    };

    [Theory]
    [InlineData("image", null)]
    [InlineData("bytes", null)]
    [InlineData("path", null)]
    [InlineData("image", "i2v")]
    [InlineData("bytes", "i2v")]
    [InlineData("path", "i2v")]
    public void ShorthandSourcesUsePublisherFirstPassAndPreserveFullStrengthRefinement(string source, string? mode)
    {
        var request = Shorthand(source, mode);
        request.ImageConditions = []; // Studio supplies empty lists for unused guide types.
        request.ReferenceImagePaths = [];
        var guides = Ltx25ImageConditioning.Resolve(request, 17);
        float strength = FirstStageStrength(request, resolved: guides)!.Value;
        Assert.Equal(.7f, strength);
        Assert.Equal(1, Assert.Single(guides).Strength);

        // Verify that the policy changes the actual first-frame denoising mask
        // and blend, while the independent refinement encoding stays fully pinned.
        var refinementImages = new List<Ltx25EncodedImage> { new(Enumerable.Repeat(.25f, 128).ToArray(), 0, guides[0].Strength) };
        var firstImages = (List<Ltx25EncodedImage>)Pipeline.GetMethod("WithFirstStageImageStrength", BindingFlags.Static | BindingFlags.NonPublic)!
            .Invoke(null, [refinementImages, strength])!;
        var settings = new Ltx25GenerationSettings(32, 32, 17, 24, 1, 1, 3, 18, 42);
        Ltx25DenoiserInput Input(float value) => new()
        {
            Video = Enumerable.Repeat(value, 3 * 128).ToArray(), VideoTokens = 3,
            Audio = new float[18 * 128], AudioTokens = 18,
            VideoContext = [0], VideoContextTokens = 1, AudioContext = [0], AudioContextTokens = 1,
            VideoPositions = Ltx25Geometry.VideoPositions(3, 1, 1, 24), AudioPositions = Ltx25Geometry.AudioPositions(18),
            VideoSigma = 1, AudioSigma = 1, FrequencyPrecision = Ltx25FrequencyPrecision.Float64,
        };
        var first = Ltx25ImageConditioning.Apply(Input(1), settings, firstImages);
        Assert.InRange(Math.Abs(first.VideoDenoiseMask[0] - .3f), 0, 1e-7f);
        Assert.InRange(Math.Abs(first.Video[0] - .475f), 0, 1e-7f);
        var refinement = Ltx25ImageConditioning.Apply(Input(.8f), settings, refinementImages,
            Enumerable.Repeat(1f, 3 * 128).ToArray(), Ltx25GenerationSettings.RefinementNoiseScale);
        Assert.Equal(0, refinement.VideoDenoiseMask[0]);
        Assert.Equal(.25f, refinement.Video[0]);
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(.85f)]
    public void ExplicitFirstFrameStrengthIsPreserved(float strength)
    {
        var request = new VideoGenerationParams { ImageConditions = [new() { ImagePath = "first.png", Strength = strength }] };
        Assert.Null(FirstStageStrength(request));
        Assert.Equal(strength, Assert.Single(Ltx25ImageConditioning.Resolve(request, 17)).Strength);
    }

    [Theory]
    [InlineData("end-image")]
    [InlineData("end-path")]
    [InlineData("explicit-keyframe")]
    [InlineData("ordered-single")]
    [InlineData("ordered-multiple")]
    [InlineData("keyframes-mode")]
    public void OtherTemporalGuideRequestsPreserveTheirStrengths(string scenario)
    {
        var request = Shorthand("path");
        switch (scenario)
        {
            case "end-image": request.EndImage = new RgbImage(1, 1, [0, 0, 0]); break;
            case "end-path": request.EndImagePath = "end.png"; request.Mode = "fl2v"; break;
            case "explicit-keyframe": request.ImageConditions = [new() { ImagePath = "middle.png", FrameIndex = 8, Strength = .85f }]; break;
            case "ordered-single": request.ImagePath = null; request.ReferenceImagePaths = ["first.png"]; break;
            case "ordered-multiple": request.ImagePath = null; request.ReferenceImagePaths = ["first.png", "end.png"]; break;
            case "keyframes-mode": request.Mode = "keyframes"; break;
        }
        Assert.Null(FirstStageStrength(request));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("t2v")]
    [InlineData("t2va")]
    public void TextOnlyRequestsDoNotSelectAnImageStrength(string? mode) =>
        Assert.Null(FirstStageStrength(new() { Mode = mode }));

    [Fact]
    public void IdentityAndSingleStageRequestsRemainUnchanged()
    {
        Assert.Null(FirstStageStrength(Shorthand("path"), twoStage: false));
        Assert.Null(FirstStageStrength(Shorthand("path", "ref"), twoStage: false, resolved: []));
        Assert.Null(FirstStageStrength(Shorthand("path", "ref"), resolved: []));
    }

    [Theory]
    [InlineData(false, 1f)]
    [InlineData(false, .85f)]
    [InlineData(true, 1f)]
    [InlineData(true, .85f)]
    public void DiagnosticOverrideWinsWithoutChangingRequestedRefinement(bool explicitGuide, float diagnosticStrength)
    {
        var request = explicitGuide
            ? new VideoGenerationParams { ImageConditions = [new() { ImagePath = "first.png", Strength = .6f }] }
            : Shorthand("path");
        Assert.Equal(diagnosticStrength, FirstStageStrength(request, diagnosticOverride: diagnosticStrength));
        Assert.Equal(explicitGuide ? .6f : 1f, Assert.Single(Ltx25ImageConditioning.Resolve(request, 17)).Strength);
    }

    [Theory]
    [InlineData(-.1f)]
    [InlineData(1.1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void InvalidDiagnosticOverrideIsRejected(float strength)
    {
        var error = Assert.Throws<TargetInvocationException>(() => FirstStageStrength(Shorthand("path"), diagnosticOverride: strength));
        Assert.IsType<ArgumentOutOfRangeException>(error.InnerException);
    }

    [Fact]
    public void DiagnosticOverrideStillRequiresTwoStagesAndAFrameZeroGuide()
    {
        Assert.IsType<ArgumentException>(Assert.Throws<TargetInvocationException>(() =>
            FirstStageStrength(Shorthand("path"), twoStage: false, diagnosticOverride: .7f)).InnerException);
        Assert.IsType<ArgumentException>(Assert.Throws<TargetInvocationException>(() =>
            FirstStageStrength(new() { EndImagePath = "end.png" }, diagnosticOverride: .7f)).InnerException);
    }

    [Fact]
    public void StudioUploadAndChatUploadSelectTheSameSharedImagePolicy()
    {
        string uploadRoot = Path.Combine(Path.GetTempPath(), "ltx-image-policy-" + Guid.NewGuid().ToString("N"));
        try
        {
            var options = ServerOptionsBuilder.Build([], uploadRoot);
            Directory.CreateDirectory(options.UploadDirectory);
            string file = Path.Combine(options.UploadDirectory, "first.png");
            File.WriteAllBytes(file, [1, 2, 3]); // Request parsing never decodes pixels.
            var studio = NativeStudioGenerationAdapter.Parameters(new StudioGenerationRequest { ConditioningAssetId = "first" },
                new Dictionary<string, string> { ["first"] = file }, null, default);
            var chat = VideoGenerationParamsParser.Parse(JsonSerializer.SerializeToElement(new { imagePath = "first.png", videoMode = "i2v" }), options, out string error);
            Assert.Null(error);
            Assert.NotNull(studio.ImagePath);
            Assert.NotNull(chat.ImageBytes);
            Assert.Equal(.7f, FirstStageStrength(studio));
            Assert.Equal(FirstStageStrength(studio), FirstStageStrength(chat));
        }
        finally { if (Directory.Exists(uploadRoot)) Directory.Delete(uploadRoot, true); }
    }
}
