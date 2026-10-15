// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class QwenByteBicubicMemoryTests
{
    [Theory]
    // Independent payload sums: horizontal RGB bytes; per-axis bound int pairs;
    // and full double + 22-bit int coefficient tables at 2*ceil(2*scale)+1.
    [InlineData(7, 5, 4, 3, 872L)]
    [InlineData(3, 2, 7, 5, 858L)]
    [InlineData(7, 3, 3, 7, 923L)]
    [InlineData(640, 480, 560, 420, 896560L)]
    public void ForecastIncludesHorizontalBoundsAndBothCoefficientRepresentations(
        int sourceWidth, int sourceHeight, int width, int height, long expected)
        => Assert.Equal(expected, ImageProcessorUtils.GetByteBicubicResizeScratchBytes(sourceWidth, sourceHeight, width, height));

    [Fact]
    public void UnchangedSizeNeedsNoResizeIntermediateOrCoefficients()
        => Assert.Equal(0, ImageProcessorUtils.GetByteBicubicResizeScratchBytes(640, 480, 640, 480));

    [Fact]
    public void OpaqueUnchangedSizePreservesEveryExistingNormalizedFloatBit()
    {
        const int width = 32, height = 24;
        var rgba = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            rgba[4 * i] = (byte)(i * 37);
            rgba[4 * i + 1] = (byte)(i * 71);
            rgba[4 * i + 2] = (byte)(255 - i * 13);
            rgba[4 * i + 3] = 255;
        }
        float[] previous = ImageProcessorUtils.ResizeRgbaToChannelFirstNormalized(rgba, width, height, width, height);
        float[] actual = ImageProcessorUtils.ResizeRgbaToChannelFirstBicubicByte(rgba, width, height, width, height);
        Assert.Equal(previous.Select(BitConverter.SingleToInt32Bits), actual.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void InvalidOrUnsupportedScratchIsRejectedBeforeAnyPixelAllocation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageProcessorUtils.GetByteBicubicResizeScratchBytes(0, 3, 4, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageProcessorUtils.GetByteBicubicResizeScratchBytes(3, 4, -1, 5));
        Assert.Throws<NotSupportedException>(() => ImageProcessorUtils.GetByteBicubicResizeScratchBytes(int.MaxValue, 1, 1, 1));
        Assert.Throws<NotSupportedException>(() => ImageProcessorUtils.GetByteBicubicResizeScratchBytes(1, 1, int.MaxValue, 1));
        Assert.Throws<NotSupportedException>(() => ImageProcessorUtils.GetByteBicubicResizeScratchBytes(1, 500000000, 700000000, 1));
        Assert.Throws<ArgumentException>(() => ImageProcessorUtils.ResizeRgbaToChannelFirstBicubicByte([], 3, 2, 7, 5));
    }

    [Theory]
    [InlineData(513, 257, 512, 256, 465408L)]
    [InlineData(512, 256, 512, 256, 0L)]
    public void HeaderOnlyVisionAdmissionAddsKnownResizeScratchBeforeDecoding(
        int sourceWidth, int sourceHeight, int width, int height, long scratch)
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-qwen-byte-bicubic-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string path = Path.Combine(directory, "header-only.png");
            // Intentionally no compressed pixels: the planner must stay header-only.
            byte[] header = new byte[24];
            new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }.CopyTo(header, 0);
            BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(8), 13);
            "IHDR"u8.CopyTo(header.AsSpan(12));
            BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(16), (uint)sourceWidth);
            BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(20), (uint)sourceHeight);
            File.WriteAllBytes(path, header);
            var geometry = new Qwen35VisionMemoryGeometry(16, 2, 64, 128, 4, 2, 32, 64, 4096, 2048, 128);
            var plan = QwenVisionPreparationPlan.Create([path], 10, 1, 1000, geometry);
            var image = Assert.Single(plan.Images);
            Assert.Equal((height, width), (image.ResizedHeight, image.ResizedWidth));
            long previousKnownRepresentations = 24L + 32L * sourceWidth * sourceHeight + 12L * width * height;
            Assert.Equal(previousKnownRepresentations + scratch, image.HostPreparationBytes);
            Assert.True(plan.HostPeakBytes >= image.HostPreparationBytes + image.EmbeddingBytes);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }
}
