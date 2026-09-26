// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public sealed class QwenImageResolutionTests : IDisposable
{
    private readonly string _width = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH");
    private readonly string _height = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT");

    public QwenImageResolutionTests()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", null);
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", null);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", _width);
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", _height);
    }

    [Fact]
    public void AutomaticTextToImageUsesNativeTwoKGeometry()
    {
        Assert.Equal((2048, 2048), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null));
    }

    [Theory]
    [InlineData(4, 3, 2368, 1760)]
    [InlineData(3, 4, 1760, 2368)]
    public void AutomaticEditRetainsAspectRatioAtNativeArea(int width, int height, int expectedWidth, int expectedHeight)
    {
        var reference = new RgbImage(width, height, new float[width * height * 3]);
        Assert.Equal((expectedWidth, expectedHeight),
            QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), reference));
    }

    [Fact]
    public void ExplicitDraftAreaKeepsOneKAvailable()
    {
        Assert.Equal((1024, 1024), QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { TargetArea = 1024L * 1024 }, null));
    }

    [Fact]
    public void ExplicitGeometryWinsOverAreaAndEnvironment()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "2048");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "2048");
        Assert.Equal((1024, 768), QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { Width = 1024, Height = 768, TargetArea = 512L * 512 }, null));
    }

    [Fact]
    public void EnvironmentGeometryStillOverridesAutomaticSize()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "1536");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "1024");
        Assert.Equal((1536, 1024), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null));
    }

    [Fact]
    public void ExplicitAreaWinsOverTheServerDefaultSize()
    {
        // A request that names an area keeps its own geometry; the default size is
        // only for requests that named neither a size nor an area.
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "1536");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "1024");
        Assert.Equal((1024, 1024), QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { TargetArea = 1024L * 1024 }, null));
    }

    [Fact]
    public void PreResolvedAutomaticAreaStillTakesTheServerDefaultSize()
    {
        // The Web UI / API layer resolves an omitted targetArea to the native area
        // before the pipeline sees it; that must still count as "no area".
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "1536");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "1024");
        var p = new QwenImageParams();
        p.TargetArea = p.ResolveTargetArea();
        Assert.Equal((1536, 1024), QwenImage21Pipeline.ResolveDimensions(p, null));
    }

    [Fact]
    public void DefaultSizeOffTheGrid_SnapsDownAndWarnsOnce()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "1000");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "700");

        string warnings = CaptureStderr(() =>
        {
            // Every request that falls back to the default used to fail here.
            Assert.Equal((992, 672), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null));
            Assert.Equal((992, 672), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null));
        });

        Assert.Equal(1, Occurrences(warnings, "992x672"));
        Assert.Contains("1000x700", warnings, StringComparison.Ordinal);
    }

    [Fact]
    public void DefaultSizeBelowOneTile_SnapsToTheMinimum()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "20");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "33");
        string warnings = CaptureStderr(() =>
            Assert.Equal((32, 32), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null)));
        Assert.Contains("32x32", warnings, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("TS_QWEN_IMAGE_WIDTH", "TS_QWEN_IMAGE_HEIGHT", "1248")]
    [InlineData("TS_QWEN_IMAGE_HEIGHT", "TS_QWEN_IMAGE_WIDTH", "1184")]
    public void HalfConfiguredDefault_IsIgnoredWithAWarning(string set, string missing, string value)
    {
        Environment.SetEnvironmentVariable(set, value);

        string warnings = CaptureStderr(() =>
        {
            Assert.Equal((2048, 2048), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null));
            Assert.Equal((2048, 2048), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null));
        });

        Assert.Equal(1, Occurrences(warnings, "WARNING"));
        Assert.Contains($"{set} is set without {missing}", warnings, StringComparison.Ordinal);
    }

    [Fact]
    public void UnparsableDefault_IsIgnoredWithAWarning()
    {
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "wide");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "-64");
        string warnings = CaptureStderr(() =>
            Assert.Equal((2048, 2048), QwenImage21Pipeline.ResolveDimensions(new QwenImageParams(), null)));
        Assert.Contains("TS_QWEN_IMAGE_WIDTH=wide", warnings, StringComparison.Ordinal);
    }

    [Fact]
    public void ExplicitRequestSizeKeepsItsStrictValidation()
    {
        // Snapping is for the operator's fallback default only; a request that asks
        // for an off-grid size is still told so.
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH", "1536");
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT", "1024");
        Assert.Throws<ArgumentException>(() => QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { Width = 1000, Height = 1000 }, null));
        Assert.Throws<ArgumentException>(() => QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { Width = 1024 }, null));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    public void TensorParallelDegreesThatDivideTheHeadsAreAccepted(int ranks)
    {
        Assert.Null(QwenImageModel.DitTensorParallelRefusal(ranks));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(6)]
    [InlineData(12)]
    public void TensorParallelRefusalStatesTheRuleTheCheckApplies(int ranks)
    {
        string refusal = QwenImageModel.DitTensorParallelRefusal(ranks);
        Assert.NotNull(refusal);
        Assert.Contains("divides its 32 attention heads", refusal, StringComparison.Ordinal);
        Assert.Contains($"--tp {ranks}", refusal, StringComparison.Ordinal);
        // The old text listed "(2, 4 or 8)" while the check also accepted 16 and 32.
        Assert.DoesNotContain("2, 4 or 8", refusal, StringComparison.Ordinal);
    }

    private static string CaptureStderr(Action action)
    {
        var captured = new StringWriter();
        TextWriter saved = Console.Error;
        try
        {
            Console.SetError(captured);
            action();
        }
        finally { Console.SetError(saved); }
        return captured.ToString();
    }

    private static int Occurrences(string text, string value)
    {
        int count = 0;
        for (int i = text.IndexOf(value, StringComparison.Ordinal); i >= 0;
             i = text.IndexOf(value, i + value.Length, StringComparison.Ordinal))
            count++;
        return count;
    }
}
