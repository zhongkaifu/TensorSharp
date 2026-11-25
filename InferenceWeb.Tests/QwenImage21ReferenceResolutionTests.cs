// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using Xunit;

namespace InferenceWeb.Tests;

public class QwenImage21ReferenceResolutionTests
{
    [Theory]
    [InlineData(2048, 2048, 1024, 1024)]
    [InlineData(1024, 1024, 1024, 1024)]
    [InlineData(256, 256, 256, 256)]
    public void ReferencesAreBoundedIndependentlyOfOutput(int width, int height, int expectedWidth, int expectedHeight)
    {
        var image = new RgbImage(32, 32, new float[32 * 32 * 3]);
        Assert.Equal((expectedWidth, expectedHeight), QwenImage21TestGeometry.ResolveReferenceDimensions(image, width, height));
    }

    [Fact]
    public void ReferenceRetainsItsOwnAspectRatio()
    {
        var image = new RgbImage(64, 32, new float[64 * 32 * 3]);
        var (width, height) = QwenImage21TestGeometry.ResolveReferenceDimensions(image, 2048, 2048);
        Assert.Equal((1440, 736), (width, height));
    }
}
