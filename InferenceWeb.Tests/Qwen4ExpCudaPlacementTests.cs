// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class Qwen4ExpCudaPlacementTests
{
    private const long GiB = 1L << 30;

    [Fact]
    public void Plan_UsesFreeMemoryAfterPreloadAndKeepsOnlyFittingTrailingLayers()
    {
        // A 16 GiB card with 10 GiB free, 1 GiB of pending cache/float
        // bindings, 1 GiB headroom and the 3 GiB graph scratch reserve.
        long[] layers = { 4 * GiB, 3 * GiB, 2 * GiB, GiB };
        Assert.Equal(2, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB, 0));
        Assert.Equal(0, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 4 * GiB, GiB, 0));
    }

    [Fact]
    public void Plan_ExpertCacheServesEveryLayerWhenWholeModelDoesNotFit()
    {
        long[] layers = { 4 * GiB, 3 * GiB, 2 * GiB, GiB };
        long[] minimum = { GiB, GiB, GiB, GiB };
        Assert.Equal(0, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB, 4 * GiB, minimum, 4));
        // An all-resident model keeps its single graph even with a cache request.
        Assert.Equal(4, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 15 * GiB, GiB, 4 * GiB, minimum, 4));
    }

    [Fact]
    public void Plan_InsufficientCachePreservesResidentTrailingLayers()
    {
        long[] layers = { 4 * GiB, 3 * GiB, 2 * GiB, GiB };
        long minimum = CacheMinimum();
        long[] cache = { minimum, minimum, minimum, minimum };
        Assert.Equal(2, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB, 1L << 20, cache, 4));
        Assert.Equal(2, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB, minimum * 4 - 1, cache, 4));
        Assert.Equal(0, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB, minimum * 4, cache, 4));
    }

    [Fact]
    public void Plan_UnsupportedOrDisabledCacheKeepsWholeResidentLayers()
    {
        long[] layers = { 4 * GiB, 3 * GiB, 2 * GiB, GiB };
        Assert.Equal(2, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB, 4 * GiB, new long[4], 4));
        Assert.Equal(2, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB, 4 * GiB, new[] { GiB, GiB, GiB, GiB }, 0));
        Assert.Equal(2, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB, 4 * GiB));
    }

    [Fact]
    public void Plan_HeterogeneousCacheAndResidentLayersShareTheAvailableMemory()
    {
        long[] layers = { 4 * GiB, 3 * GiB, 2 * GiB, GiB };
        // Only the first layer can cache. Its 2 GiB quota must be reserved
        // before keeping the 1+2 GiB trailing layers in the 5 GiB available.
        Assert.Equal(2, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 10 * GiB, GiB,
            8 * GiB, new[] { GiB, 0L, 0L, 0L }, 4));
        // With one less GiB available, retaining both tiers would overcommit.
        Assert.Equal(1, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 9 * GiB, GiB,
            8 * GiB, new[] { GiB, 0L, 0L, 0L }, 4));
        // Offloading that third layer adds another eligible cache quota, so
        // the plan must repeat the reservation and release the final layer too.
        Assert.Equal(0, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(layers, GiB, 9 * GiB, GiB,
            8 * GiB, new[] { GiB, 0L, GiB, 0L }, 4));
    }

    [Fact]
    public void Plan_ReservesPossibleNativeCacheEvenBelowConservativeFitThreshold()
    {
        // Four GiB remains. Both trailing layers fit without a cache. The native
        // allocator may reuse buffers and fit the first layer into its 2 GiB
        // quota although the conservative graph estimate is 3 GiB. Retaining
        // both layers as well would spend 5 GiB, so only the last can remain.
        Assert.Equal(1, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(
            new[] { 4 * GiB, 3 * GiB, 2 * GiB, GiB }, GiB, 9 * GiB, GiB,
            8 * GiB, new[] { 3 * GiB, 0L, 0L, 0L }, 4,
            new[] { GiB, 0L, 0L, 0L }));
    }

    [Fact]
    public void CacheMinimum_RequiresQuantizedSplitWeightsAndCompleteSelectedSlots()
    {
        Assert.InRange(CacheMinimum(), (1L << 20) + 1, 2L << 20);
        var gate = Expert(GgmlTensorType.IQ3_S, 256, 128);
        var up = Expert(GgmlTensorType.IQ4_XS, 256, 128);
        var down = Expert(GgmlTensorType.IQ4_NL, 128, 256);
        foreach (var type in new[] { GgmlTensorType.F16, GgmlTensorType.BF16, GgmlTensorType.I8 })
            Assert.Equal(0, ExpertPlacementPlanner.MinimumCudaExpertCacheBytes(Expert(type, 256, 128), up, down, 256, 128, 4, 16));
        Assert.Equal(0, ExpertPlacementPlanner.MinimumCudaExpertCacheBytes(null!, null!, down, 256, 128, 4, 16));
        Assert.Equal(0, ExpertPlacementPlanner.MinimumCudaExpertCacheBytes(gate, up, down, 256, 128, 17, 16));
        Assert.Equal(0, ExpertPlacementPlanner.MinimumCudaExpertCacheBytes(gate, up, down, 320, 128, 4, 16));
        Assert.Equal(0, ExpertPlacementPlanner.MinimumCudaExpertCacheBytes(
            Expert(GgmlTensorType.IQ3_S, 256, 128, gate.TotalRawBytes + 1), up, down, 256, 128, 4, 16));
    }

    private static StackedExpertWeights Expert(GgmlTensorType type, int width, int rows, long? bytes = null)
        => new(new IntPtr(1), (int)type, width, rows, 16,
            bytes ?? width / GgufFile.GetBlockSize(type) * GgufFile.GetTypeSize(type) * rows * 16,
            isExternalView: true, ownerToken: null, ownedBuffer: IntPtr.Zero);

    private static long CacheMinimum() => ExpertPlacementPlanner.MinimumCudaExpertCacheBytes(
        Expert(GgmlTensorType.IQ3_S, 256, 128), Expert(GgmlTensorType.IQ4_XS, 256, 128),
        Expert(GgmlTensorType.IQ4_NL, 128, 256), 256, 128, 4, 16);

    [Fact]
    public void Plan_DoesNotOverflowWhenReportedMemoryOrWeightsReachInt64Limit()
    {
        Assert.Equal(0, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(
            new[] { long.MaxValue }, long.MaxValue, long.MaxValue, GiB, 0));
        Assert.Equal(0, ExpertPlacementPlanner.PlanCudaDeviceExpertLayers(new[] { GiB }, 0, -1, GiB, 0));
    }

    [Theory]
    [InlineData(null, 0)]
    [InlineData("", 0)]
    [InlineData("invalid", 0)]
    [InlineData("8192 ", 0)]
    [InlineData(" 8192", 0)]
    [InlineData("+8192", 0)]
    [InlineData("-1", 0)]
    [InlineData("0", 0)]
    [InlineData("1", 1048576)]
    [InlineData("8192", 8589934592)]
    [InlineData("9223372036854775807", 0)]
    public void CacheBudget_RejectsInvalidOrOverflowingValues(string? input, long expected)
        => Assert.Equal(expected, ExpertPlacementPlanner.ResolveCudaExpertCacheBudget(input!));

    [Theory]
    [InlineData(null, 48)]
    [InlineData("", 48)]
    [InlineData("8", 8)]
    [InlineData("0008", 8)]
    [InlineData("0", 0)]
    [InlineData("invalid", 0)]
    [InlineData(" 8", 0)]
    [InlineData("8 ", 0)]
    [InlineData("+8", 0)]
    [InlineData("-8", 0)]
    [InlineData("18446744073709551615", ulong.MaxValue)]
    [InlineData("18446744073709551616", 0)]
    public void CacheLayers_MatchNativeDefaultsAndRejectDisabledOrMalformedQuotas(string? input, ulong expected)
        => Assert.Equal(expected, ExpertPlacementPlanner.ResolveCudaExpertCacheLayers(input!));
}
