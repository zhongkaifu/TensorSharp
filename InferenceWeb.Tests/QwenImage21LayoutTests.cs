// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;

namespace InferenceWeb.Tests;

/// <summary>The joint sequence of a Qwen-Image-2.1 prediction (QwenImage21Layout): text and vision-slot segments, the
/// target, their 3-axis RoPE tables, and the automatic geometry.</summary>
public sealed class QwenImage21LayoutTests
{
    private const int HeadDim = 128;

    [Fact]
    public void AutomaticReferenceGeometryMatchesHalfAwayFromZeroRounding()
    {
        // sqrt(512² * 1089/1024) = 528 = 16.5 grid cells.
        var reference = new RgbImage(1089, 1024, new float[1089 * 1024 * 3]);
        var geometry = QwenImage21Canvas.ResolveDimensions(new DiffusionRequest { TargetArea = 512 * 512 }, reference);
        Assert.Equal((544, 512), geometry);
    }

    [Fact]
    public void AutomaticGeometryRejectsIntegerOverflowInsteadOfClampingTo32()
    {
        Assert.Throws<OverflowException>(() => QwenImage21Canvas.ResolveDimensions(
            new DiffusionRequest { TargetArea = long.MaxValue }, null));
    }

    [Fact]
    public void TextThenTargetUsesCausalTextAndCenteredImageCoordinates()
    {
        var layout = QwenImage21Layout.Build(2, null, new[] { (2, 3) }, HeadDim);
        Assert.Equal(2, layout.Prefix);
        Assert.Equal(new[] { 0, 2 }, layout.Segments.Select(s => s.Start));
        Assert.Equal(new[] { 2, 8 }, layout.Segments.Select(s => s.End));
        Assert.Equal(new[] { false, true }, layout.Segments.Select(s => s.IsImage));
        AssertPosition(layout.Cos, layout.Sin, 0, 0, 0, 0);
        AssertPosition(layout.Cos, layout.Sin, 1, 1, 1, 1);
        // Width=3 uses [-2,-1,0], matching upstream's ceil-centered coordinates.
        AssertPosition(layout.Cos, layout.Sin, 2, 2, -1, -2);
        AssertPosition(layout.Cos, layout.Sin, 7, 2, 0, 0);
    }

    [Fact]
    public void VisionSlotExpandsToFourLatentsAndResumesTextCoordinates()
    {
        var layout = QwenImage21Layout.Build(5, new[] { 0, 0, 1, 0, 0 }, new[] { (2, 2), (2, 3) }, HeadDim);
        Assert.Equal(8, layout.Prefix);
        Assert.Equal(new[] { 0, 2, 6, 8 }, layout.Segments.Select(s => s.Start));
        Assert.Equal(new[] { 2, 6, 8, 14 }, layout.Segments.Select(s => s.End));
        Assert.Equal(new[] { 0, 0, 3, 4 }, layout.Segments.Select(s => s.SourceStart));
        Assert.Equal(new[] { false, true, false, true }, layout.Segments.Select(s => s.IsImage));
        AssertPosition(layout.Cos, layout.Sin, 2, 2, -1, -1);
        AssertPosition(layout.Cos, layout.Sin, 6, 4, 4, 4);
        AssertPosition(layout.Cos, layout.Sin, 8, 6, -1, -2);
    }

    [Fact]
    public void MultipleReferencesPreserveTheirOrderAndDistinctSources()
    {
        var layout = QwenImage21Layout.Build(5, new[] { 0, 1, 0, 2, 0 }, new[] { (2, 2), (2, 2), (2, 2) }, HeadDim);
        Assert.Equal(11, layout.Prefix);
        Assert.Equal(new[] { 0, 1, 5, 6, 10, 11 }, layout.Segments.Select(s => s.Start));
        Assert.Equal(new[] { 0, 4, 8 }, layout.Segments.Where(s => s.IsImage).Select(s => s.SourceStart));
        AssertPosition(layout.Cos, layout.Sin, 1, 1, -1, -1);
        AssertPosition(layout.Cos, layout.Sin, 6, 4, -1, -1);
        AssertPosition(layout.Cos, layout.Sin, 11, 7, -1, -1);
    }

    [Theory]
    [InlineData(new[] { 0, 0, 0 })] // missing reference
    [InlineData(new[] { 0, 2, 0 })] // out of order
    [InlineData(new[] { 0, -1, 0 })] // invalid tag
    [InlineData(new[] { 1, 1, 0 })] // two vision slots cannot represent four latents
    [InlineData(new[] { 1, 0, 1 })] // repeated noncontiguous reference
    public void InvalidReferenceSlotsAreRefused(int[] slots)
    {
        Assert.Throws<ArgumentException>(() => QwenImage21Layout.Build(3, slots, new[] { (2, 2), (2, 2) }, HeadDim));
    }

    private static void AssertPosition(float[] cos, float[] sin, int token, int t, int h, int w)
    {
        // The first pair of each RoPE axis has frequency one. These checks use
        // analytic coordinates rather than reproducing the layout implementation.
        foreach (var (channel, coordinate) in new[] { (0, t), (8, h), (36, w) })
        {
            Assert.InRange(Math.Abs(cos[token * 64 + channel] - Math.Cos(coordinate)), 0, 1e-7);
            Assert.InRange(Math.Abs(sin[token * 64 + channel] - Math.Sin(coordinate)), 0, 1e-7);
        }
    }
}
