// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;

namespace InferenceWeb.Tests;

public class MiniMaxH3UploadedImageTests
{
    [Theory]
    [InlineData(null, MiniMaxH3Mode.ImageToVideo)]
    [InlineData("i2v", MiniMaxH3Mode.ImageToVideo)]
    [InlineData("fl2v", MiniMaxH3Mode.FirstLastFrame)]
    public void UploadedBytesBecomeTheFirstFrameWithoutAFilePath(string? mode, MiniMaxH3Mode expected)
    {
        var request = new VideoGenerationParams { Mode = mode, ImageBytes = UploadedPng() };
        Assert.Equal(expected,
            MiniMaxH3ModeResolver.Resolve(request, MiniMaxH3Partition.FirstLastFrame));

        var frames = new List<RgbImage>();
        var atEnd = new List<bool>();
        MiniMaxH3Pipeline.ResolveKeyframes(request, MiniMaxH3Geometry.Resolve(64, 32, 22), frames, atEnd);

        AssertUploadedPixels(Assert.Single(frames));
        Assert.False(Assert.Single(atEnd));
        Assert.Null(request.ImagePath);
    }

    [Fact]
    public void UploadedFirstFrameAndLastFrameKeepTheirTimelineOrder()
    {
        var last = SolidImage(0f, 0f, 1f);
        var request = new VideoGenerationParams { Mode = "fl2v", ImageBytes = UploadedPng(), EndImage = last };
        Assert.Equal(MiniMaxH3Mode.FirstLastFrame,
            MiniMaxH3ModeResolver.Resolve(request, MiniMaxH3Partition.FirstLastFrame));

        var frames = new List<RgbImage>();
        var atEnd = new List<bool>();
        MiniMaxH3Pipeline.ResolveKeyframes(request, MiniMaxH3Geometry.Resolve(64, 32, 22), frames, atEnd);

        Assert.Equal(2, frames.Count);
        AssertUploadedPixels(frames[0]);
        Assert.Same(last, frames[1]);
        Assert.Equal(new[] { false, true }, atEnd);
    }

    [Fact]
    public void UploadedBytesSetTheCanvasAspectWhenOnlyWidthIsSupplied()
    {
        var request = new VideoGenerationParams { Width = 640, ImageBytes = UploadedPng() };
        Assert.Equal((640, 320), MiniMaxH3Pipeline.ResolveCanvas(request, MiniMaxH3Mode.ImageToVideo));
    }

    [Fact]
    public void APlainUploadIsOneReferenceOnRef2Va()
    {
        var request = new VideoGenerationParams { ImageBytes = UploadedPng() };
        Assert.Equal(MiniMaxH3Mode.Reference,
            MiniMaxH3ModeResolver.Resolve(request, MiniMaxH3Partition.Reference));

        var references = MiniMaxH3Pipeline.LoadReferences(request, 64, 32, 22);
        var reference = Assert.Single(references);
        Assert.Equal(MiniMaxH3ReferenceKind.Image, reference.Kind);
        var frame = Assert.Single(reference.Frames);
        AssertUploadedPixels(frame);
        Assert.Same(frame, Assert.Single(reference.Presented));
    }

    [Theory]
    [InlineData("i2v")]
    [InlineData("fl2v")]
    public void ExplicitKeyframeModeWithAnUploadStillRequiresFl2Va(string mode)
    {
        var request = new VideoGenerationParams { Mode = mode, ImageBytes = UploadedPng() };
        var error = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3ModeResolver.Resolve(request, MiniMaxH3Partition.Reference));
        Assert.Contains("keyframe conditioning needs the FL2VA checkpoint", error.Message);
    }

    [Fact]
    public void I2vWithAReferenceUploadReportsCheckpointMismatch()
    {
        // Web clients send their upload as a named reference on Ref2VA. A server
        // default of i2v must name the incompatible checkpoint rather than claim
        // that the user failed to provide an image.
        var request = new VideoGenerationParams
        {
            Mode = "i2v",
            ReferenceImagePaths = new List<string> { "uploaded-reference.png" },
        };
        var error = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3ModeResolver.Resolve(request, MiniMaxH3Partition.Reference));
        Assert.Contains("keyframe conditioning needs the FL2VA checkpoint", error.Message);
        Assert.DoesNotContain("needs a conditioning image", error.Message);
    }

    [Fact]
    public void UploadedKeyframeCannotBeCombinedWithNamedReferences()
    {
        var request = new VideoGenerationParams
        {
            Mode = "ref",
            ImageBytes = UploadedPng(),
            ReferenceImagePaths = new List<string> { "reference.png" },
        };
        var error = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3ModeResolver.Resolve(request, MiniMaxH3Partition.Reference));
        Assert.Contains("keyframes and references", error.Message);
    }

    [Fact]
    public void EmptyUploadDoesNotCountAsAConditioningImage()
    {
        var request = new VideoGenerationParams { Mode = "i2v", ImageBytes = Array.Empty<byte>() };
        var error = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3ModeResolver.Resolve(request, MiniMaxH3Partition.FirstLastFrame));
        Assert.Contains("Upload an image", error.Message);
    }

    private static byte[] UploadedPng() => ImageIO.EncodePng(SolidImage(1f, 0f, 0f));

    private static RgbImage SolidImage(float r, float g, float b)
    {
        var pixels = new float[64 * 32 * 3];
        for (int i = 0; i < pixels.Length; i += 3)
        {
            pixels[i] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
        }
        return new RgbImage(64, 32, pixels);
    }

    private static void AssertUploadedPixels(RgbImage image)
    {
        Assert.Equal(64, image.Width);
        Assert.Equal(32, image.Height);
        for (int i = 0; i < image.Pixels.Length; i += 3)
        {
            Assert.Equal(1f, image.Pixels[i]);
            Assert.Equal(0f, image.Pixels[i + 1]);
            Assert.Equal(0f, image.Pixels[i + 2]);
        }
    }
}
