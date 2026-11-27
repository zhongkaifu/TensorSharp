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
    static float? FirstStageStrength(VideoGenerationParams request, bool twoStage = true,
        IReadOnlyList<VideoImageCondition>? resolved = null)
        => Ltx25Family.ResolveFirstStageImageStrength(request, twoStage, resolved ?? Ltx25ImageConditioning.Resolve(request, 17));

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
        var firstImages = Ltx25Family.WithFirstStageImageStrength(refinementImages, strength);
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
