// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Pure metadata/policy tests. No native allocator, CUDA initialization,
/// model execution or performance qualification is implied.</summary>
public sealed class GlmSelectedExpertMemoryPlanTests
{
    private const long GiB = 1L << 30;
    private static GlmUnifiedExpertLayout Layout(GgmlTensorType gate, GgmlTensorType up,
        GgmlTensorType down, long bytes) => new(new(gate, bytes / 3, 0, true),
            new(up, bytes / 3, 0, true), new(down, bytes - 2 * (bytes / 3), 0, true));

    private static GlmUnifiedMemoryGeometry Geometry() => new()
    {
        Layers = 3, DeclaredContextTokens = 32768, RecurrentLayers = 1,
        Hidden = 4096, Heads = 64, Vocab = 154880, KvRow = 512, IndexerDim = 128,
        KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048,
        UsedExperts = 8, ExpertCount = 288, DenseDeviceBytes = GiB,
        ExpertLayerBytes = [30 * GiB, 30 * GiB, 30 * GiB],
        ExpertLayerLayouts = [Layout(GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ3_XXS, 7_536_640),
            Layout(GgmlTensorType.IQ1_S, GgmlTensorType.IQ1_S, GgmlTensorType.IQ3_XXS, 6_488_064),
            Layout(GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ4_XS, 8_781_824)]
    };

    private static UnifiedMemoryModelOptions Options(int prefill = 128, int requests = 3, long maximum = long.MaxValue)
        => new() { EnableGlmSelectedExperts = true, PrefillChunkTokens = prefill,
            MaximumRunningRequests = requests, MaximumExpertCacheBytes = maximum };

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void PhaseFloorProtectsEachWideGroupAndEveryOtherNarrowGroup(int rows)
    {
        var geometry = Geometry();
        long forecast = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, rows);
        long narrowRows = Math.Max(3, Math.Min(rows, 8));
        long perRow = 4 * (4096 + 2L * 8 + 6L * 2048 * 8 + 4L * 4096 * 8);
        long narrowScratch = narrowRows * (narrowRows + 1) / 2 * perRow
            + 128 * (3 + 256 * narrowRows) + (1L << 20);
        long wideScratch = narrowScratch + (rows > 8 ? rows * perRow + 128 * 256 : 0);
        long union = Math.Min(288, 8L * (rows > 8 ? rows : 3));
        long[] payload = [7_536_640, 6_488_064, 8_781_824];
        long[] narrow = payload.Select(b => 24 * b + narrowScratch).ToArray();
        long[] wide = payload.Select(b => union * b + wideScratch).ToArray();
        long exact = Enumerable.Range(0, 3).Max(i => wide[i] + narrow.Sum() - narrow[i]);
        Assert.Equal(exact, forecast);
        Assert.All(Enumerable.Range(0, 3), i => Assert.True(forecast >= wide[i] + narrow.Sum() - narrow[i]));
        if (rows > 8) Assert.True(forecast < wide.Sum());
        else Assert.Equal(wide.Sum(), forecast);
    }

    [Fact]
    public void ResourceGrowthAndShrinkSelectLargestFeasiblePrefillWithoutMutatingCaller()
    {
        var geometry = Geometry();
        var options = Options();
        var original = options with { };
        var small = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, 4 * GiB, GiB / 2);
        var large = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, 11 * GiB, GiB / 2);
        var shrunk = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, 4 * GiB, GiB / 2);
        Assert.True(small.PrefillRows > 8);
        Assert.True(small.PrefillRows < large.PrefillRows);
        Assert.Equal(128, large.PrefillRows);
        Assert.Equal(small, shrunk);
        Assert.Equal(original, options);
        Assert.Equal(128, options.PrefillChunkTokens);
        Assert.Equal(3, small.CpuExpertLayers);
        Assert.True(small.CacheMinimumForecastBytes <= small.CacheCapacityBytes);
        long nextReserve = geometry.ExecutionPlacementReserve(32768, small.PrefillRows + 1, 3);
        long nextCapacity = GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(4 * GiB,
            geometry.DenseDeviceBytes, nextReserve, GiB / 2, options.MaximumExpertCacheBytes);
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, small.PrefillRows + 1) > nextCapacity);
        Assert.Equal(small.CacheCapacityBytes + geometry.ExecutionPlacementReserve(32768, small.PrefillRows, 3),
            small.ExecutionReserveBytes);
        Assert.True(small.ExecutionReserveBytes + geometry.DenseDeviceBytes + GiB / 2 <= 4 * GiB);
    }

    [Theory]
    [InlineData(129)]
    [InlineData(256)]
    [InlineData(384)]
    [InlineData(512)]
    public void WidePrefillFallsBackAtTheExactCombinedSlotGraphAndExpertFloor(int rows)
    {
        var geometry = Geometry();
        var options = Options(prefill: 512);
        const long vision = GiB / 2;
        long graphAndSlots = geometry.ExecutionPlacementReserve(32768, rows, 3);
        long expertFloor = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, rows);
        long ceiling = checked(geometry.DenseDeviceBytes + vision + graphAndSlots + expertFloor);

        var exact = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, ceiling, vision);
        var below = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, ceiling - 1, vision);
        Assert.Equal(rows, exact.PrefillRows);
        Assert.Equal(rows - 1, below.PrefillRows);
        Assert.Equal(expertFloor, exact.CacheCapacityBytes);
        Assert.Equal(expertFloor, exact.CacheMinimumForecastBytes);
        Assert.Equal(graphAndSlots + expertFloor, exact.ExecutionReserveBytes);
        Assert.Equal(3, exact.CpuExpertLayers);
        Assert.True(geometry.SlotBytes(32768) * 3 <= graphAndSlots);
        Assert.True(below.CacheMinimumForecastBytes <= below.CacheCapacityBytes);
        Assert.Equal(512, options.PrefillChunkTokens);
    }

    [Fact]
    public void WidePrefillPreservesSmallCallerAndLogicalContextCaps()
    {
        var geometry = Geometry();
        var capped = GlmSelectedExpertMemoryPlan.Select(geometry, Options(prefill: 129), 32768, 20 * GiB, 0);
        Assert.Equal(129, capped.PrefillRows);
        var shortContext = GlmSelectedExpertMemoryPlan.Select(geometry, Options(prefill: 512), 200, 20 * GiB, 0);
        Assert.Equal(200, shortContext.PrefillRows);
        Assert.Equal(geometry.ExecutionPlacementReserve(200, 200, 3) + shortContext.CacheCapacityBytes,
            shortContext.ExecutionReserveBytes);
        Assert.True(shortContext.CacheMinimumForecastBytes <= shortContext.CacheCapacityBytes);
    }

    [Theory]
    [InlineData(8, 512)]
    [InlineData(16, 256)]
    [InlineData(64, 64)]
    public void LargerSelectedExpertCountsKeepTheFixedRouteScratchBound(int used, int cap)
    {
        var geometry = Geometry() with { UsedExperts = used };
        var plan = GlmSelectedExpertMemoryPlan.Select(geometry, Options(prefill: 512), 32768, 20 * GiB, 0);
        Assert.Equal(cap, plan.PrefillRows);
        Assert.InRange(plan.PrefillRows * used, 1, 4096);
        if (cap < 512)
            Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, cap + 1, maximumRouteElements: 4096));
        // Fully resident experts do not use the compact cache's route scratch.
        var resident = GlmSelectedExpertMemoryPlan.Select(geometry, Options(prefill: 512), 32768, 200 * GiB, 0);
        Assert.Equal(512, resident.PrefillRows);
        Assert.Equal(0, resident.CpuExpertLayers);
    }

    [Fact]
    public void OperatorAndCallerCapsArePreservedWhileTwoGiBIsNoLongerAnImplicitCap()
    {
        var geometry = Geometry();
        var automatic = GlmSelectedExpertMemoryPlan.Select(geometry, Options(), 32768, 10 * GiB, GiB / 2);
        var limited = GlmSelectedExpertMemoryPlan.Select(geometry, Options(maximum: 2 * GiB), 32768, 10 * GiB, GiB / 2);
        Assert.True(automatic.CacheCapacityBytes > 2 * GiB);
        Assert.Equal(2 * GiB, limited.CacheCapacityBytes);
        Assert.InRange(limited.PrefillRows, 9, 127);
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, limited.PrefillRows) <= 2 * GiB);
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, limited.PrefillRows + 1) > 2 * GiB);
        var capped = GlmSelectedExpertMemoryPlan.Select(geometry, Options(prefill: 7), 32768, 10 * GiB, 0);
        Assert.Equal(7, capped.PrefillRows);
        Assert.Equal(512, GlmSelectedExpertMemoryPlan.Select(geometry, Options(prefill: 512), 32768, 10 * GiB, 0).PrefillRows);
        Assert.Equal(4, GlmSelectedExpertMemoryPlan.Select(geometry, Options(), 4, 10 * GiB, 0).PrefillRows);
    }

    [Fact]
    public void NarrowVariantsProtectMaximumDecodeUnionAndFailClosedBelowItsFloor()
    {
        var geometry = Geometry();
        long narrow = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 1);
        Assert.Equal(narrow, GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 3));
        Assert.True(narrow < GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 8));
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 8)
            > GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 1, 8));
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 9) > narrow);
        Assert.Equal(3, GlmSelectedExpertMemoryPlan.Select(geometry, Options(maximum: narrow), 32768, 10 * GiB, 0).PrefillRows);
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(geometry,
            Options(maximum: narrow - 1), 32768, 10 * GiB, 0));
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(geometry,
            Options(maximum: 0), 32768, 10 * GiB, 0));
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(geometry,
            Options(), 32768, geometry.DenseDeviceBytes, GiB));
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(3, 8)]
    [InlineData(1, 32)]
    [InlineData(3, 128)]
    [InlineData(1, 256)]
    [InlineData(3, 512)]
    public void CompactPrefillRetainsEveryNarrowGraphBesideTheSingleWideOwner(int decodeRows, int prefillRows)
    {
        var geometry = Geometry();
        long narrowRows = Math.Max(decodeRows, Math.Min(prefillRows, 8));
        long retainedNarrowRows = Enumerable.Range(1, checked((int)narrowRows)).Sum();
        long wideRows = prefillRows > 8 ? prefillRows : 0;
        long union = Math.Min(geometry.ExpertCount,
            (wideRows > 0 ? Math.Max(decodeRows, prefillRows) : decodeRows) * (long)geometry.UsedExperts);
        long perRow = 4 * (geometry.Hidden + 2L * geometry.UsedExperts
            + 6L * geometry.ExpertFf * geometry.UsedExperts + 4L * geometry.Hidden * geometry.UsedExperts);
        long descriptorPadding = 128 * (3 + 256 * (narrowRows + (wideRows > 0 ? 1 : 0)));
        long narrowUnion = Math.Min(geometry.ExpertCount, decodeRows * (long)geometry.UsedExperts);
        long narrowPadding = 128 * (3 + 256 * narrowRows);
        long expected = narrowUnion * (7_536_640 + 6_488_064)
            + 2 * (retainedNarrowRows * perRow + narrowPadding + (1L << 20))
            + union * 8_781_824 + (retainedNarrowRows + wideRows) * perRow + descriptorPadding + (1L << 20);
        Assert.Equal(expected, GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, decodeRows, prefillRows));
        // The old decode-only forecast misses these real, concurrently retained
        // widths. Wide replacement frees its previous owner before allocation,
        // so no second wide payload belongs in the steady minimum.
        long missingRows = retainedNarrowRows - decodeRows * (decodeRows + 1L) / 2;
        Assert.True(missingRows > 0);
        Assert.True(expected - 3 * missingRows * perRow < expected);
    }

    [Fact]
    public void NarrowScratchCapChoosesLargestFittingWidthWithoutInventingWideWeights()
    {
        var geometry = Geometry();
        long cap = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 1, 7);
        var chosen = GlmSelectedExpertMemoryPlan.Select(geometry,
            Options(prefill: 8, requests: 1, maximum: cap), 32768, 10 * GiB, 0);
        Assert.Equal(7, chosen.PrefillRows);
        Assert.Equal(cap, chosen.CacheMinimumForecastBytes);
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 1, 8) > cap);
        long wide = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 1, 9);
        Assert.True(wide > GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 1, 8));
    }

    [Fact]
    public void GroupFloorsUseEachActualPayloadAndIgnoreDuplicateTypeTriples()
    {
        var geometry = Geometry();
        var duplicate = geometry with { Layers = 4, ExpertLayerLayouts = [.. geometry.ExpertLayerLayouts, geometry.ExpertLayerLayouts[0]] };
        Assert.Equal(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 16),
            GlmSelectedExpertMemoryPlan.ForecastMinimum(duplicate, 3, 16));
        var uniformLargest = geometry with { ExpertLayerLayouts = geometry.ExpertLayerLayouts.Select(l =>
            Layout(l.Gate.Type, l.Up.Type, l.Down.Type, 8_781_824)).ToArray() };
        Assert.Equal(24L * (2 * 8_781_824L - (7_536_640 + 6_488_064)),
            GlmSelectedExpertMemoryPlan.ForecastMinimum(uniformLargest, 3, 16)
            - GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 16));
        var withTails = geometry with { ExpertLayerLayouts = geometry.ExpertLayerLayouts.Select(l =>
            l with { Gate = l.Gate with { TensorTailBytes = 512 } }).ToArray() };
        Assert.Equal(3 * 512, GlmSelectedExpertMemoryPlan.ForecastMinimum(withTails, 3, 16)
            - GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 16));
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void PhaseMinimumProtectsOtherDecodeFloorsWithoutAllWideResidency(int rows)
    {
        var geometry = Geometry();
        long maximum = 7000L << 20;
        long floor = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 1, rows);
        Assert.True(floor < maximum);
        long otherGroupsWideWeight = 288L * (7_536_640 + 6_488_064);
        long otherGroupsNarrowWeight = 8L * (7_536_640 + 6_488_064);
        Assert.True(floor > 288L * 8_781_824 + otherGroupsNarrowWeight);
        Assert.True(floor < 288L * 8_781_824 + otherGroupsWideWeight);
        var plan = GlmSelectedExpertMemoryPlan.Select(geometry,
            Options(prefill: rows, requests: 1, maximum: maximum), 1024, 14 * GiB, 0);
        Assert.Equal(rows, plan.PrefillRows);
        Assert.Equal(floor, plan.CacheMinimumForecastBytes);
        Assert.Equal(maximum, plan.CacheCapacityBytes);
    }

    [Fact]
    public void DisabledAndFullyResidentPlansPreserveOriginalPrefillAndUseNoExpertEnvelope()
    {
        var geometry = Geometry();
        var disabled = GlmSelectedExpertMemoryPlan.Select(geometry, Options() with { EnableGlmSelectedExperts = false }, 32768, 10 * GiB, 0);
        Assert.Equal(128, disabled.PrefillRows);
        Assert.Equal(0, disabled.CacheCapacityBytes);
        Assert.Equal(0, disabled.CacheMinimumForecastBytes);
        var resident = GlmSelectedExpertMemoryPlan.Select(geometry with { ExpertLayerLayouts = [] }, Options(), 32768, 100 * GiB, 0);
        Assert.Equal(128, resident.PrefillRows);
        Assert.Equal(0, resident.CpuExpertLayers);
        Assert.Equal(0, resident.CacheCapacityBytes);
        Assert.False(new UnifiedMemoryModelOptions().EnableGlmSelectedExperts);
    }

    [Theory]
    [InlineData(8)]
    [InlineData(32)]
    [InlineData(127)]
    public void DefaultExpertModeSelectsLargestPrefillAtExactPlacementBudget(int rows)
    {
        var geometry = Geometry();
        var options = Options(prefill: 512) with { EnableGlmSelectedExperts = false };
        long vision = GiB / 2;
        long reserve = geometry.ExecutionPlacementReserve(32768, rows, 3);
        long ceiling = checked(geometry.DenseDeviceBytes + vision + reserve);
        // The larger chunk really cannot fit even after every expert layer is
        // offloaded. No selected-expert envelope is needed in this mode.
        Assert.Throws<MemoryPressureException>(() => geometry.SelectCpuExpertLayers(ceiling,
            geometry.ExecutionPlacementReserve(32768, rows + 1, 3), vision));

        var exact = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, ceiling, vision);
        var below = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, ceiling - 1, vision);
        Assert.Equal(rows, exact.PrefillRows);
        Assert.Equal(rows - 1, below.PrefillRows);
        Assert.Equal(reserve, exact.ExecutionReserveBytes);
        Assert.Equal(3, exact.CpuExpertLayers);
        Assert.Equal(0, exact.CacheCapacityBytes);
        Assert.Equal(0, exact.CacheMinimumForecastBytes);
        Assert.Equal(new UnifiedMemoryModelOptions { PrefillChunkTokens = 512 }, options);
    }

    [Fact]
    public void DefaultExpertModePreservesDenseVisionAndConcurrentSlotFloorAtSmallestPrefill()
    {
        var geometry = Geometry();
        var options = Options(prefill: 1) with { EnableGlmSelectedExperts = false };
        long vision = GiB / 2;
        long reserve = geometry.ExecutionPlacementReserve(32768, 1, 3);
        long ceiling = checked(geometry.DenseDeviceBytes + vision + reserve);
        var exact = GlmSelectedExpertMemoryPlan.Select(geometry, options, 32768, ceiling, vision);
        Assert.Equal(1, exact.PrefillRows);
        Assert.Equal(reserve, exact.ExecutionReserveBytes);
        Assert.Equal(3, exact.CpuExpertLayers);
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(
            geometry, options, 32768, ceiling - 1, vision));
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(
            geometry, options, 32768, ceiling, vision + 1));
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(
            geometry with { DenseDeviceBytes = geometry.DenseDeviceBytes + 1 }, options, 32768, ceiling, vision));
        long soloCeiling = checked(geometry.DenseDeviceBytes + vision
            + geometry.ExecutionPlacementReserve(32768, 1, 1));
        Assert.True(soloCeiling < ceiling);
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.Select(
            geometry, options, 32768, soloCeiling, vision));
        Assert.Equal(3, options.MaximumRunningRequests);
        Assert.False(options.EnableGlmSelectedExperts);
    }

    [Fact]
    public void CheckedSixtyFourBitMathRejectsOverflowAndUnsupportedOrIncompleteLayouts()
    {
        var geometry = Geometry();
        var overflow = geometry with { ExpertLayerLayouts = [Layout(GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, long.MaxValue), null, null] };
        Assert.Throws<OverflowException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(overflow, 3, 128));
        Assert.Throws<OverflowException>(() => GlmSelectedExpertMemoryPlan.Select(geometry with { DenseDeviceBytes = long.MaxValue }, Options(), 32768, long.MaxValue, 0));
        var unsupported = geometry with { ExpertLayerLayouts = [geometry.ExpertLayerLayouts[0] with
            { Gate = geometry.ExpertLayerLayouts[0].Gate with { SupportsSelectedTier = false } }, null, null] };
        Assert.Throws<NotSupportedException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(unsupported, 3, 8));
        Assert.Throws<NotSupportedException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry with { ExpertLayerLayouts = [] }, 3, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 4, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 1025));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 513,
            4096));
        Assert.True(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 1024)
            > GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 512));
    }

    [Fact]
    public void OptionalPrefixReplacementCannotPermanentlyReduceTheAdmittedPrefillWidth()
    {
        var geometry = Geometry();
        var options = Options(prefill: 512) with
        {
            EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = 1024,
        };
        const int context = 32768;
        long slot = geometry.SlotBytes(context);
        long mandatoryExecution = geometry.AdaptiveExecutionPlacementReserve(context, 3);
        long fullPrefillFloor = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 1024);
        // Three live requests, one retained public checkpoint and the idle
        // primary coexist at this exact metadata-derived placement boundary.
        // A future capture is optional and has not allocated any owner yet.
        long ceiling = checked(geometry.DenseDeviceBytes + mandatoryExecution + fullPrefillFloor + 2 * slot);
        var prefix = UnifiedMemoryPrefixCapacity.Select(1, slot, ceiling);
        Assert.Equal(1, prefix.RetainedCopies);
        var plan = GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, context, ceiling,
            checked(prefix.PlacementReserveBytes + slot));
        Assert.Equal(1024, plan.PrefillRows);
        Assert.Equal(fullPrefillFloor, plan.CacheMinimumForecastBytes);
        Assert.Equal(fullPrefillFloor, plan.CacheCapacityBytes);
        Assert.Equal(3, plan.CpuExpertLayers);
        Assert.Equal(ceiling, geometry.DenseDeviceBytes + plan.ExecutionReserveBytes + prefix.PlacementReserveBytes + slot);

        // The removed policy reserves an additional, nonexistent replacement
        // for the session's entire lifetime and selects a smaller phase floor.
        var oldPolicy = GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, context, ceiling, 3 * slot);
        Assert.True(oldPolicy.PrefillRows < plan.PrefillRows);
        Assert.Equal(options.AdaptiveGlmPrefillChunkTokens, plan.PrefillRows);
    }
}
