// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>
/// Tests that set or read the Qwen-Image server default size (TS_QWEN_IMAGE_WIDTH/HEIGHT), which
/// every automatic geometry consults: one class setting it while another resolved a size changed
/// that size. They run alone.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class QwenImageDefaultSizeCollection
{
    public const string Name = "Qwen-Image default size";
}

/// <summary>
/// An edit that keeps its source picture's size (<see cref="QwenImageParams.KeepSourceSize"/>).
/// Sized from an area alone, an unmasked edit of a 1600 x 1200 photo came back at 1184 x 896 in
/// TensorAgent (1 MP) and 2368 x 1760 in the Web UI (4 MP), while a selection edit returned the
/// photo's own canvas; so a selection edit followed by any other edit of its result shrank.
/// The geometry is read from the server default size's variables, which
/// <see cref="QwenImageResolutionTests"/> sets too, so both run alone.
/// </summary>
[Collection(QwenImageDefaultSizeCollection.Name)]
public sealed class QwenImageKeepSourceSizeTests : IDisposable
{
    private const long TensorAgentArea = 1024L * 1024;   // ImageTurns.DefaultTargetArea
    private const long NativeArea = 2048L * 2048;         // what the server resolves an omitted targetArea to

    private readonly string _width = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH");
    private readonly string _height = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT");

    public QwenImageKeepSourceSizeTests()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", null);
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", null);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", _width);
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", _height);
    }

    private static RgbImage Picture(int width, int height) => new(width, height, new float[checked(width * height * 3)]);

    /// <summary>A selection over part of the picture: white in its left half.</summary>
    private static RgbImage HalfSelection(int width, int height)
    {
        var pixels = new float[checked(width * height * 3)];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width / 2; x++)
                pixels[(y * width + x) * 3] = pixels[(y * width + x) * 3 + 1] = pixels[(y * width + x) * 3 + 2] = 1f;
        return new RgbImage(width, height, pixels);
    }

    /// <summary>The request a page sends for an edit, as the server parses it.</summary>
    private static QwenImageParams Edit(long area, bool keep, RgbImage? mask = null) =>
        new() { TargetArea = area, KeepSourceSize = keep, Mask = mask };

    /// <summary>The size an edit of <paramref name="source"/> comes back at.</summary>
    private static (int Width, int Height) EditedSize(RgbImage source, QwenImageParams p) =>
        QwenImage21Pipeline.ResolveGeometry(p, source, QwenImageEditMask.Create(p, source)).Output;

    // ---- the report, reproduced ----------------------------------------------------------

    [Theory]
    [InlineData(TensorAgentArea, 1184, 896)]
    [InlineData(NativeArea, 2368, 1760)]
    public void WithoutTheOption_ASelectionEditKeepsThePhotoAndTheNextEditShrinksIt(long area, int shrunkWidth, int shrunkHeight)
    {
        var photo = Picture(1600, 1200);
        var selected = EditedSize(photo, Edit(area, keep: false, HalfSelection(1600, 1200)));
        Assert.Equal((1600, 1200), selected);

        var next = EditedSize(Picture(selected.Width, selected.Height), Edit(area, keep: false));
        Assert.Equal((shrunkWidth, shrunkHeight), next);
    }

    [Theory]
    [InlineData(TensorAgentArea)]
    [InlineData(NativeArea)]
    public void WithTheOption_EveryEditOfAnEditKeepsTheSize(long area)
    {
        // Selection, change, change, selection, change: the size never moves.
        var size = (Width: 1600, Height: 1200);
        foreach (bool masked in new[] { true, false, false, true, false })
        {
            var p = Edit(area, keep: true, masked ? HalfSelection(size.Width, size.Height) : null);
            size = EditedSize(Picture(size.Width, size.Height), p);
            Assert.Equal((1600, 1200), size);
        }
    }

    // ---- where it samples ----------------------------------------------------------------

    [Theory]
    // Larger than the budget: sampled exactly where an automatic size would, enlarged back.
    [InlineData(1600, 1200, TensorAgentArea, 1184, 896)]
    [InlineData(4032, 3024, TensorAgentArea, 1184, 896)]
    [InlineData(3024, 4032, TensorAgentArea, 896, 1184)]
    [InlineData(4032, 3024, NativeArea, 2368, 1760)]
    // On the budget: sampled at its own size, so nothing is resized.
    [InlineData(1024, 1024, TensorAgentArea, 1024, 1024)]
    [InlineData(1184, 896, TensorAgentArea, 1184, 896)]
    [InlineData(2048, 2048, NativeArea, 2048, 2048)]
    // Between the floor and the budget: sampled at its own size, never enlarged to the budget.
    [InlineData(1600, 1200, NativeArea, 1600, 1216)]
    [InlineData(1024, 1024, NativeArea, 1024, 1024)]
    // Below the floor: sampled at about the floor and resized down.
    [InlineData(800, 600, TensorAgentArea, 1184, 896)]
    [InlineData(1024, 768, NativeArea, 1184, 896)]
    public void SamplingStaysWithinTheBudgetAndTheSourcesOwnArea(int width, int height, long area, int sampledWidth, int sampledHeight)
    {
        var p = Edit(area, keep: true);
        var (sampling, output) = QwenImage21Pipeline.ResolveGeometry(p, Picture(width, height), null);

        Assert.Equal((sampledWidth, sampledHeight), sampling);
        Assert.Equal((width, height), output);
        Assert.True(sampling.Width % 32 == 0 && sampling.Height % 32 == 0);
        // Within the budget and the source's own area, give or take the 32-pixel grid (half a
        // cell on each side), and at the source's aspect ratio to within one cell.
        long sampled = (long)sampling.Width * sampling.Height, own = (long)width * height;
        long expected = Math.Min(area, Math.Max(own, QwenImage21Pipeline.KeptSourceSamplingFloor));
        Assert.True(sampled <= expected * 1.1 && sampled >= expected / 1.1, $"{sampling} is not about the area it should spend ({expected})");
        double aspect = (double)width / height;
        Assert.InRange(sampling.Width / (double)sampling.Height, aspect * (1 - 32.0 / sampling.Height), aspect * (1 + 32.0 / sampling.Height));
        // Larger than the budget, it is exactly the automatic geometry.
        if (own >= area)
            Assert.Equal(QwenImage21Pipeline.ResolveDimensions(Edit(area, keep: false), Picture(width, height)), sampling);
    }

    [Fact]
    public void TheServersDefaultSizeIsABudgetAtTheSourcesShape()
    {
        // A landscape default: kept for a portrait photo, it would sample landscape and be
        // stretched back to portrait. Only its area is spent.
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "1536");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "1024");
        var p = new QwenImageParams { KeepSourceSize = true };
        p.TargetArea = p.ResolveTargetArea();   // as the server parses an omitted area
        var (sampling, output) = QwenImage21Pipeline.ResolveGeometry(p, Picture(3024, 4032), null);
        Assert.Equal((3024, 4032), output);
        Assert.Equal((1088, 1440), sampling);
        // Without the option the default size wins outright, as before.
        var plain = new QwenImageParams();
        plain.TargetArea = plain.ResolveTargetArea();
        Assert.Equal(((1536, 1024), (1536, 1024)), QwenImage21Pipeline.ResolveGeometry(plain, Picture(3024, 4032), null));
    }

    [Fact]
    public void TheCpuBackendsAutomaticAreaIsTheBudget()
    {
        var p = new QwenImageParams { KeepSourceSize = true };
        p.TargetArea = p.ResolveTargetArea();
        Assert.Equal(((1184, 896), (1600, 1200)),
            QwenImage21Pipeline.ResolveGeometry(p, Picture(1600, 1200), null, BackendType.Cpu));
        Assert.Equal(((2368, 1760), (4032, 3024)),
            QwenImage21Pipeline.ResolveGeometry(p, Picture(4032, 3024), null, BackendType.GgmlMetal));
    }

    [Fact]
    public void TheResizedResultIsExactlyTheSourceSize()
    {
        // What the pipeline does with the decoded image (FinishOutput, called from Run): 1184 x
        // 896 decoded for a 1600 x 1200 photo comes back at 1600 x 1200.
        var p = Edit(TensorAgentArea, keep: true);
        var (sampling, output) = QwenImage21Pipeline.ResolveGeometry(p, Picture(1600, 1200), null);
        var decoded = new RgbImage(sampling.Width, sampling.Height,
            Enumerable.Range(0, sampling.Width * sampling.Height * 3).Select(i => (i % 7) / 7f).ToArray());
        var result = QwenImage21Pipeline.FinishOutput(decoded, null, output);
        Assert.Equal((1600, 1200), (result.Width, result.Height));
        Assert.Equal(1600 * 1200 * 3, result.Pixels.Length);
    }

    [Fact]
    public void WithoutKeepingTheSourceSizeTheDecodedPictureIsReturnedAsItIs()
    {
        // The same instance: no resize, so outputs without the option are byte-identical to before.
        var p = Edit(TensorAgentArea, keep: false);
        var (sampling, output) = QwenImage21Pipeline.ResolveGeometry(p, Picture(1600, 1200), null);
        var decoded = new RgbImage(sampling.Width, sampling.Height, new float[sampling.Width * sampling.Height * 3]);
        Assert.Same(decoded, QwenImage21Pipeline.FinishOutput(decoded, null, output));
    }

    [Theory]
    [InlineData(16, 16)]
    [InlineData(100, 100)]
    [InlineData(256, 256)]
    [InlineData(640, 480)]
    public void ASmallSourceSamplesAtTheFloorAndComesBackAtItsOwnSize(int width, int height)
    {
        // A 100 x 100 icon would have been edited from 36 image tokens; it samples at about the
        // floor, where references are conditioned, and is resized down to its own size.
        var p = Edit(TensorAgentArea, keep: true);
        var (sampling, output) = QwenImage21Pipeline.ResolveGeometry(p, Picture(width, height), null);
        Assert.Equal((width, height), output);
        long sampled = (long)sampling.Width * sampling.Height;
        Assert.InRange(sampled, QwenImage21Pipeline.KeptSourceSamplingFloor / 1.1, QwenImage21Pipeline.KeptSourceSamplingFloor * 1.1);
        // A smaller budget than the floor still bounds it.
        var tight = QwenImage21Pipeline.ResolveGeometry(Edit(256L * 256, keep: true), Picture(width, height), null).Sampling;
        Assert.InRange((long)tight.Width * tight.Height, 0, 256L * 256 * 1.2);
    }

    // ---- what does not change ------------------------------------------------------------

    [Theory]
    [InlineData(1600, 1200, TensorAgentArea, null)]
    [InlineData(1600, 1200, NativeArea, null)]
    [InlineData(1089, 1024, 512L * 512, null)]
    [InlineData(4, 3, NativeArea, BackendType.Cpu)]
    [InlineData(800, 600, TensorAgentArea, BackendType.GgmlMetal)]
    public void WithoutTheOption_TheOutputIsTheSampledImageAsBefore(int width, int height, long area, BackendType? backend)
    {
        var p = Edit(area, keep: false);
        var reference = Picture(width, height);
        var (sampling, output) = QwenImage21Pipeline.ResolveGeometry(p, reference, null, backend);
        Assert.Equal(QwenImage21Pipeline.ResolveDimensions(p, reference, backend), sampling);
        Assert.Equal(sampling, output);
    }

    [Fact]
    public void TextToImageIsUnchanged() =>
        Assert.Equal(((2048, 2048), (2048, 2048)), QwenImage21Pipeline.ResolveGeometry(new QwenImageParams(), null, null));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AMaskedEditSamplesAndReturnsTheSameWithOrWithoutTheOption(bool crop)
    {
        var source = Picture(1600, 1200);
        QwenImageParams Masked(bool keep) =>
            new() { TargetArea = TensorAgentArea, KeepSourceSize = keep, Mask = HalfSelection(1600, 1200), MaskCrop = crop };
        var plain = Masked(false);
        var kept = Masked(true);
        var expected = QwenImage21Pipeline.ResolveGeometry(plain, source, QwenImageEditMask.Create(plain, source));
        Assert.Equal(expected, QwenImage21Pipeline.ResolveGeometry(kept, source, QwenImageEditMask.Create(kept, source)));
        Assert.Equal((1600, 1200), expected.Output);
    }

    // ---- refusals ------------------------------------------------------------------------

    [Fact]
    public void TheOptionAndAnExplicitSizeAreTwoAnswersToOneQuestion()
    {
        var p = new QwenImageParams { KeepSourceSize = true, Width = 1024, Height = 768 };
        var error = Assert.Throws<ArgumentException>(() => QwenImage21Pipeline.ResolveGeometry(p, Picture(1600, 1200), null));
        Assert.Contains("width/height", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheOptionNeedsAPictureToKeepTheSizeOf()
    {
        var error = Assert.Throws<ArgumentException>(() =>
            QwenImage21Pipeline.ResolveGeometry(new QwenImageParams { KeepSourceSize = true }, null, null));
        Assert.Contains("input image", error.Message, StringComparison.Ordinal);
    }
}
