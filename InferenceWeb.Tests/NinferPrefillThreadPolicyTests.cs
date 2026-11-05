// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferPrefillThreadPolicyTests
{
    [Theory]
    [InlineData(null, false, 256)]
    [InlineData(null, true, 256)] // Independent original-module option remains independent.
    [InlineData("256", false, 256)]
    [InlineData("128", false, 128)]
    public void OnlyAnExplicit128SelectsTheNewSchedule(string? raw, bool originalModule, int expected)
        => Assert.Equal(expected, CudaNinferAttention.ResolvePrefillThreads(raw!, originalModule));

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData(" 128")]
    [InlineData("128 ")]
    [InlineData("128\n")]
    [InlineData("0")]
    [InlineData("1")]
    [InlineData("64")]
    [InlineData("129")]
    [InlineData("512")]
    [InlineData("auto")]
    [InlineData("1.28e2")]
    [InlineData("１２８")]
    public void InvalidOrAmbiguousValuesNeverSilentlyEnableOrFallBack(string raw)
        => Assert.Throws<ArgumentException>(() => CudaNinferAttention.ResolvePrefillThreads(raw, false));

    [Theory]
    [InlineData("128")]
    [InlineData("256")]
    [InlineData("")]
    public void ExplicitScheduleAndOriginalKernelModuleCannotBeCombined(string raw)
        => Assert.Throws<NotSupportedException>(() => CudaNinferAttention.ResolvePrefillThreads(raw, true));

    [Theory]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(129)]
    [InlineData(416)]
    [InlineData(512)]
    public void OnlyCompleteOrdinarySingleSequencePrefillUses128(int rows)
        => Assert.True(CudaNinferAttention.Use128Prefill(128, rows, 1, false, true, 24, 4, 512));

    [Theory]
    [InlineData(256, 64, 1, false, true, 24, 4, 512)]
    [InlineData(0, 64, 1, false, true, 24, 4, 512)]
    [InlineData(129, 64, 1, false, true, 24, 4, 512)]
    [InlineData(128, -1, 1, false, true, 24, 4, 512)]
    [InlineData(128, 0, 1, false, true, 24, 4, 512)]
    [InlineData(128, 1, 1, false, true, 24, 4, 512)]
    [InlineData(128, 8, 1, false, true, 24, 4, 512)]
    [InlineData(128, 63, 1, false, true, 24, 4, 512)]
    [InlineData(128, 513, 1, false, true, 24, 4, 512)]
    [InlineData(128, 64, 0, false, true, 24, 4, 512)]
    [InlineData(128, 64, 2, false, true, 24, 4, 512)]
    [InlineData(128, 64, 1, true, true, 24, 4, 512)]
    [InlineData(128, 64, 1, false, false, 24, 4, 512)]
    [InlineData(128, 64, 1, false, true, 16, 4, 512)]
    [InlineData(128, 64, 1, false, true, 24, 2, 512)]
    [InlineData(128, 64, 1, false, true, 24, 4, 63)]
    public void DecodeVerifyPaddingBatchAndForeignGeometryKeepTheOldSchedule(int mode, int rows, int sequences,
        bool verify, bool valid, int heads, int kvHeads, int capacity)
        => Assert.False(CudaNinferAttention.Use128Prefill(mode, rows, sequences, verify, valid, heads, kvHeads, capacity));

    [Fact]
    public void VerifyAndPaddedGroupsStayExcludedAtEverySupportedBoundary()
    {
        foreach (int rows in new[] { 64, 65, 129, 255, 256, 257, 416, 512 })
        {
            Assert.False(CudaNinferAttention.Use128Prefill(128, rows, 1, true, true, 24, 4, 512));
            Assert.False(CudaNinferAttention.Use128Prefill(128, rows, 1, false, false, 24, 4, 512));
            Assert.False(CudaNinferAttention.Use128Prefill(128, rows, 2, false, true, 24, 4, 512));
            Assert.False(CudaNinferAttention.Use128Prefill(256, rows, 1, false, true, 24, 4, 512));
        }
    }
}
