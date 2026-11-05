using System;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NinferDecodeGraphPolicyTests
{
    [Theory]
    [InlineData(null, false)] [InlineData("0", false)] [InlineData("1", true)]
    public void GraphsRequireAnExplicitChoice(string value, bool expected)
        => Assert.Equal(expected, Q35CudaEngine.NinferDecodeGraphsRequested(value));

    [Theory]
    [InlineData("")] [InlineData("true")] [InlineData("2")] [InlineData(" 1")]
    public void InvalidGraphChoiceIsRefused(string value)
        => Assert.Throws<ArgumentException>(() => Q35CudaEngine.NinferDecodeGraphsRequested(value));

    [Fact]
    public void OnlyCompleteSingleRowGroupsWithStableInputsCanBeCaptured()
    {
        Assert.True(Q35CudaEngine.NinferDecodeGraphShapeSupported(1, 1, 1, false, false, false, false, false));
        Assert.True(Q35CudaEngine.NinferDecodeGraphShapeSupported(1, 1, 16, false, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferDecodeGraphShapeSupported(2, 1, 1, false, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferDecodeGraphShapeSupported(1, 2, 1, false, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferDecodeGraphShapeSupported(1, 1, 0, false, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferDecodeGraphShapeSupported(1, 1, 17, false, false, false, false, false));
        for (int unsupported = 0; unsupported < 5; unsupported++)
            Assert.False(Q35CudaEngine.NinferDecodeGraphShapeSupported(1, 1, 1,
                unsupported == 0, unsupported == 1, unsupported == 2, unsupported == 3, unsupported == 4));
    }

    [Fact]
    public void DraftVerificationGraphsIncludeTheAnchorWithinTheirEightRowWindow()
    {
        Assert.True(Q35CudaEngine.NinferVerifyGraphShapeSupported(1, 8, 8, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferVerifyGraphShapeSupported(1, 8, 9, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferVerifyGraphShapeSupported(1, 9, 1, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferVerifyGraphShapeSupported(1, 8, 17, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferVerifyGraphShapeSupported(2, 8, 1, false, false, false, false));
        Assert.False(Q35CudaEngine.NinferVerifyGraphShapeSupported(1, 8, 1, true, false, false, false));
    }
}
