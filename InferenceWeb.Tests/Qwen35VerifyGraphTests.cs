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
    [InlineData(20480, 128, 160)]
    [InlineData(20481, 256, 81)]
    [InlineData(33205, 256, 130)]
    [InlineData(40960, 256, 160)]
    [InlineData(40961, 512, 81)]
    [InlineData(61675, 512, 121)]
    [InlineData(131072, 1024, 128)]
    [InlineData(163840, 1024, 160)]
    [InlineData(163841, 2048, 81)]
    [InlineData(262144, 2048, 128)]
    [InlineData(327681, 4096, 81)]
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
