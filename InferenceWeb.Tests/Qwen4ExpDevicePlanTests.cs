// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The VRAM-aware placement behind qwen4exp's ggml_cuda layer split (issue #256):
// two 20 GB RTX 3080s asked to hold 62 GB of routed experts failed in warmup,
// because the split priced every expert as resident and nothing planned an
// offload per GPU. These tests drive the pure planner with the sizes from that
// report: IQ4_XS gate/up stacks of 445,644,800 B and a 471,859,200 B down stack
// per layer (1300 MiB), 48 layers, roughly 46 MiB of dense weights a layer.
using System;
using System.Linq;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class Qwen4ExpDevicePlanTests
{
    private const long MiB = 1L << 20;
    private const long GiB = 1L << 30;
    private const int Layers = 48;
    private const long ExpertLayer = 445_644_800L * 2 + 471_859_200L;   // 1300 MiB

    private static long[] Uniform(long bytes) => Enumerable.Repeat(bytes, Layers).ToArray();

    /// <summary>Dense weights per layer, plus F16 KV/QSA on every 4th (attention)
    /// layer at <paramref name="contextRows"/> rows.</summary>
    private static long[] DenseWithCaches(int contextRows)
    {
        var bytes = Uniform(46 * MiB);
        for (int l = 3; l < Layers; l += 4)
            bytes[l] += 2L * 2 * contextRows * 256 * 2 + (long)contextRows * 128 * 2;
        return bytes;
    }

    private static Qwen4ExpSpanScratch Scratch(int tokens = 2048, int context = 16384, int streamMin = 128)
        => Qwen4ExpModel.SpanScratchFor(tokens, context, streamMin);

    private static void AssertContiguousCoverage(int[] map, int devices)
    {
        Qwen4ExpModel.ValidateContiguousMap(map, map.Length, devices);
        Assert.Equal(devices, map.Distinct().Count());
    }

    private static int[] RunEnds(int[] map, int devices)
    {
        var ends = new int[devices];
        for (int l = 0; l < map.Length; l++) ends[map[l]] = l + 1;
        return ends;
    }

    [Fact]
    public void Issue256_TwoTwentyGigCards_OffloadPerGpuAndFit()
    {
        // ~19.3 GiB free on each 20 GB card, the default 1/16 headroom.
        long free = (long)(19.3 * GiB);
        long capacity = free - 20L * GiB / 16;
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(16384), Uniform(ExpertLayer),
            new[] { capacity, capacity }, firstDeviceFixed: 0, lastDeviceFixed: 600 * MiB, Scratch());

        Assert.True(plan.Fits);
        AssertContiguousCoverage(plan.LayerDevice, 2);
        for (int d = 0; d < 2; d++)
            Assert.True(plan.DeviceBytes[d] <= capacity, $"gpu{d} planned {plan.DeviceBytes[d]} > {capacity}");
        int resident = Layers - plan.HostLayers;
        // Each card holds ~10 layers of experts beside its dense weights, caches and
        // span scratch - nowhere near the 20-28 the old split asked for.
        Assert.InRange(resident, 16, 26);
        Assert.InRange(plan.HostLayers, 22, 32);
    }

    [Fact]
    public void HostLayersAreTheLeadingLayersOfEachGpusRun()
    {
        long capacity = 18 * GiB;
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(8192), Uniform(ExpertLayer),
            new[] { capacity, capacity }, 0, 600 * MiB, Scratch());
        int[] ends = RunEnds(plan.LayerDevice, 2);
        for (int d = 0, begin = 0; d < 2; d++)
        {
            bool sawResident = false;
            for (int l = begin; l < ends[d]; l++)
            {
                if (!plan.ExpertOnHost[l]) sawResident = true;
                else Assert.False(sawResident, $"layer {l} is host-routed after a resident layer on gpu{d}");
            }
            begin = ends[d];
        }
    }

    [Fact]
    public void ModelThatFits_KeepsEveryExpertResidentAndBalancesTheGpus()
    {
        long capacity = 60 * GiB;
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(8192), Uniform(ExpertLayer),
            new[] { capacity, capacity }, 0, 600 * MiB, Scratch(tokens: 4096));
        Assert.True(plan.Fits);
        Assert.Equal(0, plan.HostLayers);
        AssertContiguousCoverage(plan.LayerDevice, 2);
        // Balanced to within about one layer.
        Assert.True(Math.Abs(plan.DeviceBytes[0] - plan.DeviceBytes[1]) <= ExpertLayer + 700 * MiB,
            $"{plan.DeviceBytes[0]} vs {plan.DeviceBytes[1]}");
    }

    [Fact]
    public void LargerGpuTakesMoreResidentLayers()
    {
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(8192), Uniform(ExpertLayer),
            new[] { 22 * GiB, 10 * GiB }, 0, 600 * MiB, Scratch());
        Assert.True(plan.Fits);
        int[] ends = RunEnds(plan.LayerDevice, 2);
        int resident0 = Enumerable.Range(0, ends[0]).Count(l => !plan.ExpertOnHost[l]);
        int resident1 = Enumerable.Range(ends[0], Layers - ends[0]).Count(l => !plan.ExpertOnHost[l]);
        Assert.True(resident0 > resident1, $"{resident0} vs {resident1}");
        Assert.True(plan.DeviceBytes[0] <= 22 * GiB && plan.DeviceBytes[1] <= 10 * GiB);
    }

    [Fact]
    public void PinnedRuns_AreHonouredAndOffloadedPerGpu()
    {
        // TS_Q4E_LAYER_SPLIT=20,28 from the report.
        int[] map = Qwen4ExpModel.ParseLayerSplitOverride("20,28", Layers, 2);
        long capacity = 18 * GiB;
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(16384), Uniform(ExpertLayer),
            new[] { capacity, capacity }, 0, 600 * MiB, Scratch(), fixedMap: map);
        Assert.True(plan.Fits);
        Assert.Equal(map, plan.LayerDevice);
        // The 28-layer GPU routes more of its run to the host than the 20-layer one.
        int host0 = Enumerable.Range(0, 20).Count(l => plan.ExpertOnHost[l]);
        int host1 = Enumerable.Range(20, 28).Count(l => plan.ExpertOnHost[l]);
        Assert.True(host1 > host0, $"{host0} vs {host1}");
        Assert.All(plan.DeviceBytes, b => Assert.True(b <= capacity));
    }

    [Fact]
    public void ExplicitHostSet_IsKeptAndTheRunsArePackedAroundIt()
    {
        // --n-cpu-moe 30: the first 30 layers' experts on the host. Their GPU pays
        // only for their dense weights, so the split must give that GPU MORE layers
        // instead of the byte balance that priced their experts as resident.
        var forced = Enumerable.Range(0, Layers).Select(l => l < 30).ToArray();
        long capacity = 18 * GiB;
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(8192), Uniform(ExpertLayer),
            new[] { capacity, capacity }, 0, 600 * MiB, Scratch(), forcedHost: forced);
        Assert.True(plan.Fits);
        Assert.Equal(forced, plan.ExpertOnHost);
        AssertContiguousCoverage(plan.LayerDevice, 2);
        Assert.All(plan.DeviceBytes, b => Assert.True(b <= capacity));
    }

    [Fact]
    public void ExplicitHostSetThatCannotFit_IsReported()
    {
        var forced = new bool[Layers];   // --n-cpu-moe 0
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(8192), Uniform(ExpertLayer),
            new[] { 18 * GiB, 18 * GiB }, 0, 600 * MiB, Scratch(), forcedHost: forced);
        Assert.False(plan.Fits);
    }

    [Fact]
    public void NothingFitsWhenDenseWeightsAloneExceedTheGpus()
    {
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(Uniform(1 * GiB), Uniform(ExpertLayer),
            new[] { 8 * GiB, 8 * GiB }, 0, 600 * MiB, Scratch());
        Assert.False(plan.Fits);
        Assert.True(plan.ExpertOnHost.All(h => h));
        AssertContiguousCoverage(plan.LayerDevice, 2);
    }

    [Fact]
    public void VisionReserveOnGpu0ShiftsResidentLayersToGpu1()
    {
        long capacity = 18 * GiB;
        var without = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(8192), Uniform(ExpertLayer),
            new[] { capacity, capacity }, 0, 600 * MiB, Scratch());
        var with = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(8192), Uniform(ExpertLayer),
            new[] { capacity, capacity }, (long)(2.2 * GiB), 600 * MiB, Scratch());
        Assert.True(with.Fits);
        Assert.True(with.DeviceBytes[0] <= capacity);
        Assert.True(with.HostLayers >= without.HostLayers);
    }

    [Fact]
    public void SingleGpu_IsTheDegenerateCase()
    {
        var plan = Qwen4ExpModel.PlanQwen4ExpPlacement(DenseWithCaches(8192), Uniform(ExpertLayer),
            new[] { 14 * GiB }, 0, 600 * MiB, Scratch());
        Assert.True(plan.Fits);
        Assert.All(plan.LayerDevice, d => Assert.Equal(0, d));
        int resident = Layers - plan.HostLayers;
        Assert.InRange(resident, 1, 12);
        // Leading layers on the host, trailing ones resident.
        Assert.True(plan.ExpertOnHost.Take(plan.HostLayers).All(h => h));
        Assert.True(plan.ExpertOnHost.Skip(plan.HostLayers).All(h => !h));
    }

    [Fact]
    public void FirstHostLayerCostsAStreamBuffer_SoOneHostLayerIsNeverChosenForNothing()
    {
        // Two layers. All resident needs 2 x 1300 MiB + scratch. Routing ONE layer
        // to the host frees 1300 MiB but buys a 1300 MiB stream buffer plus seam
        // scratch: it never helps, so a budget just short of all-resident must
        // route both layers, not one.
        var dense = new[] { 46 * MiB, 46 * MiB };
        var experts = new[] { ExpertLayer, ExpertLayer };
        var scratch = Scratch();
        long allResident = 2 * 46 * MiB + 2 * ExpertLayer + scratch.Bytes(0, 0);
        int r = Qwen4ExpModel.FitResidentLayers(dense, experts, 0, 2, allResident - 1, 0, scratch);
        Assert.Equal(0, r);
        Assert.Equal(2, Qwen4ExpModel.FitResidentLayers(dense, experts, 0, 2, allResident, 0, scratch));
    }

    [Fact]
    public void FitResidentLayers_ReportsWhenEvenAllHostDoesNotFit()
    {
        var dense = Uniform(46 * MiB);
        var experts = Uniform(ExpertLayer);
        int r = Qwen4ExpModel.FitResidentLayers(dense, experts, 0, 24, 1 * GiB, 0, Scratch(), null, out long bytes);
        Assert.Equal(-1, r);
        Assert.True(bytes > GiB);
    }

    [Fact]
    public void ForcedHostSetFit_ReturnsItsResidentCountOrMinusOne()
    {
        var dense = Uniform(46 * MiB);
        var experts = Uniform(ExpertLayer);
        var forced = Enumerable.Range(0, Layers).Select(l => l < 40).ToArray();
        Assert.Equal(8, Qwen4ExpModel.FitResidentLayers(dense, experts, 0, Layers, 40 * GiB, 0, Scratch(), forced));
        Assert.Equal(-1, Qwen4ExpModel.FitResidentLayers(dense, experts, 0, Layers, 4 * GiB, 0, Scratch(), forced));
    }

    [Fact]
    public void Scratch_GrowsWithWidthAndHostLayers_AndStreamsOnlyAboveTheThreshold()
    {
        var narrow = Scratch(tokens: 2048);
        var wide = Scratch(tokens: 4096);
        Assert.True(wide.Bytes(0, 0) > narrow.Bytes(0, 0));
        Assert.True(narrow.Bytes(10, ExpertLayer) > narrow.Bytes(1, ExpertLayer));
        // The first host layer adds the stream buffer (a layer of experts).
        Assert.True(narrow.Bytes(1, ExpertLayer) - narrow.Bytes(0, 0) >= ExpertLayer);
        var noStream = Scratch(tokens: 2048, streamMin: 0);
        Assert.True(noStream.Bytes(1, ExpertLayer) - noStream.Bytes(0, 0) < ExpertLayer);
        var belowThreshold = Scratch(tokens: 2048, streamMin: 4096);
        Assert.False(belowThreshold.Streams);
    }

    [Theory]
    [InlineData(0, 2048)]
    [InlineData(8192, 2048)]
    [InlineData(14336, 2048)]
    [InlineData(30720, 1024)]
    [InlineData(1_000_000, 128)]
    public void SpanWidth_NarrowsAsTheKvItReadsGrows(int startPos, int expected)
    {
        long budget = 2048L * Qwen4ExpModel.KvRowsAtFullSpan;
        int width = Qwen4ExpModel.SpanTokensAt(startPos, 2048, budget, 2_000_000);
        Assert.Equal(expected, width);
        Assert.Equal(0, width % 64);
    }

    [Fact]
    public void SpanWidth_Uncapped_IsUnbounded()
        => Assert.Equal(int.MaxValue, Qwen4ExpModel.SpanTokensAt(1000, int.MaxValue, long.MaxValue, 8192));

    [Theory]
    [InlineData(null, 128)]
    [InlineData("0", 0)]
    [InlineData("256", 256)]
    [InlineData(" 64", 64)]
    [InlineData("-5", 128)]
    [InlineData("abc", 0)]
    [InlineData("512x", 512)]
    public void StreamThreshold_ParsesExactlyLikeTheNativeSeam(string value, int expected)
        => Assert.Equal(expected, Qwen4ExpModel.ParseStreamMinTokens(value));

    [Theory]
    [InlineData(null, null)]
    [InlineData("", null)]
    [InlineData("2048", 2048)]
    [InlineData(" 1024 ", 1024)]
    public void SpanOverride_Parses(string value, int? expected)
        => Assert.Equal(expected, Qwen4ExpModel.ParseSpanTokenOverride(value));

    [Theory]
    [InlineData("100")]
    [InlineData("abc")]
    [InlineData("-2048")]
    public void SpanOverride_RefusesValuesItCannotHonour(string value)
        => Assert.Throws<ArgumentException>(() => Qwen4ExpModel.ParseSpanTokenOverride(value));

    [Theory]
    [InlineData("qwen4exp token span: failed to allocate graph tensors. ggml: ggml_backend_cuda_buffer_type_alloc_buffer: allocating 425.00 MiB on device 0: cudaMalloc failed: out of memory", true)]
    [InlineData("qwen4exp token span: qwen4exp: could not persist mutable cache buffer", true)]
    [InlineData("qwen4exp token span: invalid speculative output shape.", false)]
    [InlineData("ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 8184608000", true)]
    [InlineData("ggml_gallocr_reserve_n_impl: failed to allocate CPU buffer of size 8184608000", false)]
    [InlineData("ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate CUDA_Host buffer of size 1024", false)]
    [InlineData("Insufficient memory to continue the execution of the program.", false)]
    [InlineData(null, false)]
    public void AllocationFailures_AreRecognized(string message, bool expected)
        => Assert.Equal(expected, Qwen4ExpModel.IsDeviceAllocationFailure(message));

    [Theory]
    [InlineData("qwen4exp: required token-span path declined; this configuration cannot use the per-layer fallback. Native token span layers [12, 48) on device 1 failed at position 0, tokens 1: qwen4exp token span: failed to allocate graph tensors.", 1)]
    [InlineData("Native token span layers [0, 20) on device 0 failed at position 0, tokens 2048: ...", 0)]
    [InlineData("qwen4exp token span: failed to allocate graph tensors.", null)]
    [InlineData(null, null)]
    public void WarmupFailure_NamesTheDeviceThatRanOut(string message, int? expected)
        => Assert.Equal(expected, Qwen4ExpModel.ParseFailedDevice(message));

    [Fact]
    public void ProjectorEstimate_IsZeroWithoutAReadableProjector()
    {
        Assert.Equal(0, Qwen4ExpModel.EstimateProjectorDeviceBytes(null));
        Assert.Equal(0, Qwen4ExpModel.EstimateProjectorDeviceBytes(""));
        Assert.Equal(0, Qwen4ExpModel.EstimateProjectorDeviceBytes(System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".gguf")));
    }

    [Fact]
    public void SingleGpuPlan_WithSpanScratch_KeepsFewerLayersAsHostLayersAddScratch()
    {
        long[] layers = Uniform(ExpertLayer);
        long free = 15 * GiB, headroom = GiB;
        int flat = Qwen4ExpModel.PlanCudaDeviceExpertLayers(layers, GiB, free, headroom, 0);
        int modelled = Qwen4ExpModel.PlanCudaDeviceExpertLayers(layers, GiB, free, headroom, 0,
            spanScratch: Scratch());
        // Both leave room; the modelled reserve is never larger than all-resident.
        Assert.InRange(flat, 1, Layers - 1);
        Assert.InRange(modelled, 0, Layers - 1);
        // Enough memory for everything keeps everything, with no stream buffer charged.
        long everything = Layers * ExpertLayer + GiB + headroom + Scratch(4096).Bytes(0, 0);
        Assert.Equal(Layers, Qwen4ExpModel.PlanCudaDeviceExpertLayers(layers, GiB, everything, headroom, 0,
            spanScratch: Scratch(4096)));
    }
}
