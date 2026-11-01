// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class Qwen35VerifyGraphTests
{
    // Exact geometry used both in the native launcher and in the captured graph key.
    [Theory]
    [InlineData(1, 128, 1)]
    [InlineData(127, 128, 1)]
    [InlineData(128, 128, 1)]
    [InlineData(129, 128, 2)]
    [InlineData(255, 128, 2)]
    [InlineData(256, 128, 2)]
    [InlineData(257, 128, 3)]
    [InlineData(8192, 128, 64)]
    [InlineData(8193, 256, 33)]
    [InlineData(16384, 256, 64)]
    [InlineData(16385, 512, 33)]
    [InlineData(131072, 2048, 64)]
    [InlineData(131073, 4096, 33)]
    [InlineData(250000, 4096, 62)]
    [InlineData(262144, 4096, 64)]
    [InlineData(262145, 8192, 33)]
    public void VerifyAttentionPlan_ChangesAtRealGridAndReductionBoundaries(int keys, int chunk, int splits)
    {
        var plan = Q35Kernels.GetSplitAttentionPlan(keys);
        Assert.Equal(chunk, plan.Chunk);
        Assert.Equal(splits, plan.Splits);
    }

    [Theory]
    [InlineData(1, 8, 24, 4, true)]
    [InlineData(2, 8, 24, 4, true)]
    [InlineData(4, 8, 24, 4, true)]
    [InlineData(8, 8, 24, 4, true)]
    [InlineData(9, 8, 24, 4, false)]
    [InlineData(16, 4, 24, 4, true)]
    [InlineData(1, 10, 24, 4, true)]
    [InlineData(1, 11, 24, 4, false)]
    [InlineData(1, 64, 4, 4, true)]
    [InlineData(1, 65, 4, 4, false)]
    [InlineData(0, 8, 24, 4, false)]
    [InlineData(17, 1, 24, 4, false)]
    [InlineData(1, 0, 24, 4, false)]
    [InlineData(1, 8, 25, 4, false)]
    [InlineData(1, 8, 24, 0, false)]
    public void VerifyCapture_RefusesShapesOnTheOtherAttentionPath(int lanes, int window, int heads, int kvHeads, bool supported)
    {
        Assert.Equal(supported, Q35CudaEngine.VerifyGraphShapeSupported(lanes, window, heads, kvHeads));
    }
}
