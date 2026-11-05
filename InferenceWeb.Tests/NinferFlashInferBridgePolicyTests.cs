// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferFlashInferBridgePolicyTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(139264)]
    public void CastCapacityCoversAllEightDownProjectionInputs(long count)
        => Assert.Equal(count * 4, CudaNinferFlashInferBridge.ValidateCast(count));

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(139265)]
    [InlineData(long.MaxValue)]
    public void CastNeverTruncatesOrOverflows(long count)
        => Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferFlashInferBridge.ValidateCast(count));

    [Theory]
    [InlineData(34816, 1, CudaNinferOriginalRowMapping.GateUpRows8)]
    [InlineData(34816, 8, CudaNinferOriginalRowMapping.GateUpRows8)]
    [InlineData(14336, 7, CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue)]
    [InlineData(5120, 8, CudaNinferOriginalRowMapping.Identity)]
    public void ExactMappedShapesHaveBoundedOutput(int n, int rows, CudaNinferOriginalRowMapping mapping)
        => Assert.Equal((long)n * rows * 4, CudaNinferFlashInferBridge.ValidateScatter(n, rows, mapping));

    [Theory]
    [InlineData(34816, 0, CudaNinferOriginalRowMapping.GateUpRows8)]
    [InlineData(34816, 9, CudaNinferOriginalRowMapping.GateUpRows8)]
    [InlineData(5120, 8, CudaNinferOriginalRowMapping.GateUpRows8)]
    [InlineData(34816, 1, CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue)]
    [InlineData(14335, 1, CudaNinferOriginalRowMapping.Identity)]
    [InlineData(14336, 1, (CudaNinferOriginalRowMapping)3)]
    public void IncorrectMappedShapeCannotLaunch(int n, int rows, CudaNinferOriginalRowMapping mapping)
        => Assert.ThrowsAny<ArgumentException>(() => CudaNinferFlashInferBridge.ValidateScatter(n, rows, mapping));

    [Fact]
    public void DisjointAdjacentBuffersAreAllowedButAnyLiveOverlapOrOverflowIsRejected()
    {
        CudaNinferFlashInferBridge.ValidateSpans((IntPtr)0x1000, 16, 4, (IntPtr)0x1010, 8, 2);
        Assert.Throws<ArgumentException>(() => CudaNinferFlashInferBridge.ValidateSpans((IntPtr)0x1000, 16, 4, (IntPtr)0x100e, 8, 2));
        Assert.Throws<ArgumentException>(() => CudaNinferFlashInferBridge.ValidateSpans((IntPtr)0x1002, 16, 4, (IntPtr)0x2000, 8, 2));
        Assert.Throws<ArgumentException>(() => CudaNinferFlashInferBridge.ValidateSpans((IntPtr)(long.MaxValue - 3), 16, 4, (IntPtr)0x2000, 8, 2));
        Assert.Throws<ArgumentException>(() => CudaNinferFlashInferBridge.ValidateSpans(IntPtr.Zero, 16, 4, (IntPtr)0x2000, 8, 2));
    }
}
