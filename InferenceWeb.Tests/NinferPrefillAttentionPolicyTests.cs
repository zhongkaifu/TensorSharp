// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferPrefillAttentionPolicyTests
{
    [Theory]
    [InlineData(64, 4)]
    [InlineData(65, 3)]
    [InlineData(129, 2)]
    [InlineData(416, 1)]
    [InlineData(512, 4)]
    public void Original170SmWavePolicyIsPreserved(int rows, int expected)
        => Assert.Equal(expected, CudaNinferPrefillAttentionModule.SplitCount(rows, 170));

    [Theory]
    [InlineData(1, 1, false, true, false)]
    [InlineData(8, 1, false, true, false)]
    [InlineData(63, 1, false, true, false)]
    [InlineData(64, 1, false, true, true)]
    [InlineData(512, 1, false, true, true)]
    [InlineData(8192, 1, false, true, true)]
    [InlineData(8193, 1, false, true, false)]
    [InlineData(64, 2, false, true, false)]
    [InlineData(64, 1, true, true, false)]
    [InlineData(64, 1, false, false, false)]
    public void DecodeVerificationBatchAndPaddedGroupsAreExcluded(int rows, int sequences, bool verify, bool valid, bool expected)
        => Assert.Equal(expected, CudaNinferPrefillAttentionModule.Supports(rows, sequences, verify, valid));

    [Theory]
    [InlineData(255, 1)]
    [InlineData(256, 1)]
    [InlineData(257, 2)]
    [InlineData(31744, 124)]
    [InlineData(32256, 126)]
    [InlineData(262144, 1024)]
    public void Native256TokenBlocksCoverExactHistoryAndTail(int visible, int blocks)
    {
        var plan = CudaNinferPrefillAttentionModule.Plan(64, visible, blocks, 170);
        Assert.Equal((visible + 63) / 64, plan.Pages);
        if (blocks > 1)
            Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.Plan(64, visible, blocks - 1, 170));
    }

    [Fact]
    public void FixedWorkspaceContainsEveryOriginalPartialAndBothBf16Planes()
    {
        const int rows = 512, context = 262144;
        var plan = CudaNinferPrefillAttentionModule.Plan(rows, context, 1025, 170);
        Assert.Equal(4, plan.Splits);
        // Independently count original tensor shapes, not a full KV-history copy.
        long q = rows * 24L * 256 * 2, acc = rows * 24L * 256 * 4 * 4, stat = rows * 24L * 4 * 4;
        Assert.Equal(6291456, q); Assert.Equal(50724864, acc + 2 * stat);
        long[] offsets = { plan.Query, plan.Output, plan.PartialAcc, plan.PartialM, plan.PartialL, plan.Planes, plan.Positions, plan.Table, plan.Bytes };
        long[] sizes = { q, q, acc, stat, stat, 32, 4, context / 64 * 4L };
        Assert.All(offsets, x => Assert.Equal(0, x % 256));
        for (int i = 0; i < sizes.Length; i++) Assert.InRange(offsets[i] + sizes[i], offsets[i], offsets[i + 1]);
        Assert.Equal(plan.Bytes, CudaNinferPrefillAttentionModule.WorkspaceCapacity(rows, context));
        Assert.True(plan.Bytes < 64L << 20);
    }

    [Fact]
    public void OnePreallocatedCapacityCoversEveryActualShorterChunkAndEveryOriginalSplit()
    {
        long capacity = CudaNinferPrefillAttentionModule.WorkspaceCapacity(512, 262144);
        foreach (int rows in Enumerable.Range(64, 449))
            foreach (int history in new[] { rows, 31744 + rows, 262144 })
                Assert.InRange(CudaNinferPrefillAttentionModule.Plan(rows, history, (history + 255) / 256, 170).Bytes, 1, capacity);
    }

    [Theory]
    [InlineData(-1, 524288)]
    [InlineData(1, 524288)]
    [InlineData(0, 16)]
    [InlineData(0, 0)]
    [InlineData(16777216, 0)]
    [InlineData(long.MaxValue, 0)]
    public void OverlappingUnalignedOrOverflowingCachePlanesAreRefused(long k, long v)
        => Assert.Throws<ArgumentException>(() => CudaNinferPrefillAttentionModule.ValidatePlaneOffsets(k, v));

    [Fact]
    public void AllSixteenNativeAttentionLayerPlanesAndLegalGuardOffsetsAreAdmitted()
    {
        for (int layer = 0; layer < 16; layer++)
        {
            long k = layer * 2 * 524288L;
            CudaNinferPrefillAttentionModule.ValidatePlaneOffsets(k, k + 524288);
        }
        CudaNinferPrefillAttentionModule.ValidatePlaneOffsets(256, 524288 + 256);
    }

    [Fact]
    public void InvalidVisibleExtentAndCapacityFailBeforeAnyLaunch()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.Plan(512, 511, 2, 170));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.Plan(512, 262145, 1025, 170));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.Plan(512, 512, 2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.WorkspaceCapacity(63, 262144));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.WorkspaceCapacity(512, 511));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.WorkspaceCapacity(512, 262145));
    }
}
