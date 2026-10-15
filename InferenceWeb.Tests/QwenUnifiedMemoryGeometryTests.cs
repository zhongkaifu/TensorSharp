// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

public sealed class QwenUnifiedMemoryGeometryTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-qwen-geometry-" + Guid.NewGuid());
    public QwenUnifiedMemoryGeometryTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void OrdinaryConstructorAndPlacementRemainAvailable()
    {
        Assert.NotNull(typeof(Qwen4ExpModel).GetConstructor(new[] { typeof(string), typeof(BackendType),
            typeof(int), typeof(ITensorParallelGroup), typeof(int), typeof(string) }));
        Qwen4ExpModel.ValidateUnifiedMemoryPolicy(null, BackendType.Cpu, 4, 2, "ordinary-draft");
        const long gib = 1L << 30;
        long[] layers = [gib, gib, gib];
        int ordinary = Qwen4ExpModel.PlanCudaDeviceExpertLayers(layers, gib, 6 * gib, gib, 0);
        Assert.Equal(1, ordinary);
        Assert.Equal(ordinary, Qwen4ExpModel.PlanCudaDeviceExpertLayers(layers, gib, 6 * gib, gib,
            0, null, 0, null, 3 * gib));
        Assert.Equal(3, Qwen4ExpModel.PlanCudaDeviceExpertLayers(layers, gib, 6 * gib, gib,
            0, null, 0, null, gib));
    }

    [Fact]
    public void ExplicitPolicyRequiresSupportedDeviceGeometryAndTrunkOnly()
    {
        var policy = new ModelMemoryPolicy(1024, 8) { OmitEmbeddedDraftWeights = true };
        Qwen4ExpModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 1, 1, null);
        Assert.Throws<NotSupportedException>(() => Qwen4ExpModel.ValidateUnifiedMemoryPolicy(policy, BackendType.Cpu, 1, 1, null));
        Assert.Throws<NotSupportedException>(() => Qwen4ExpModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 2, 1, null));
        Assert.Throws<NotSupportedException>(() => Qwen4ExpModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 1, 2, null));
        Assert.Throws<NotSupportedException>(() => Qwen4ExpModel.ValidateUnifiedMemoryPolicy(policy, BackendType.GgmlCuda, 1, 1, "draft"));
        Assert.Throws<NotSupportedException>(() => Qwen4ExpModel.ValidateUnifiedMemoryPolicy(policy with { OmitEmbeddedDraftWeights = false }, BackendType.GgmlCuda, 1, 1, null));
        foreach (var invalid in new[] { policy with { ExpertCacheBytes = -1 }, policy with { DeviceExecutionReserveBytes = -1 },
            policy with { ExpertCachePrefillRows = 0 }, policy with { ExpertCachePrefillRows = 129 } })
            Assert.Throws<ArgumentOutOfRangeException>(() => Qwen4ExpModel.ValidateUnifiedMemoryPolicy(invalid, BackendType.GgmlCuda, 1, 1, null));
    }

    [Fact]
    public void MetadataForecastSeparatesMappedExpertsAndLiveRecurrentState()
    {
        string path = WriteGeometry();
        using var file = GgufFile.OpenForMemoryPlanning(path);
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        Assert.Equal(8, geometry.Layers);
        Assert.Equal(6, geometry.RecurrentLayers);
        Assert.Equal(2, geometry.AttentionLayers);
        Assert.Equal(2, geometry.QsaLayers);
        Assert.Equal((2L * 2 * 2 * 32 + 2 * 8) * 2, geometry.KvBytesPerToken(2));
        Assert.True(geometry.ExpertSourceBytes > 0);
        Assert.True(geometry.MappedSourceBytes > geometry.ExpertSourceBytes);
        Assert.Equal(geometry.ExpertCacheMinimumBytes.Max() * 8, geometry.AllLayerExpertCacheMinimumBytes);
        // One quantized embedding plus one F16 norm promoted to F32. PLE and
        // expert file mappings cannot masquerade as preloaded dense payload.
        Assert.Equal(32 * 8 / 32 * 34 + 32 * 4 + 2 * (64L << 10), geometry.DenseDeviceBytes);
        Assert.Equal(32 * 4, geometry.ResidentHostPayloadBytes);
        var f16 = geometry.RequestPeak(512, 8, 2);
        var f32 = geometry.RequestPeak(512, 8, 4);
        long extraBothGenerations = 2 * 512 * geometry.KvBytesPerToken(2);
        Assert.Equal(extraBothGenerations, f32.Host - f16.Host);
        Assert.Equal(extraBothGenerations, f32.Device - f16.Device);
        var wide = geometry.RequestPeak(512, 64);
        Assert.True(wide.Host > f16.Host);
        Assert.True(wide.Device > f16.Device);
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.RequestPeak(512, 513));
        Assert.Throws<NotSupportedException>(() => geometry.RequestPeak(512, 8, 1));
        Assert.Throws<OverflowException>(() => (geometry with { PleStateBytes = long.MaxValue }).RequestPeak(512, 8));
    }

    [Fact]
    public void UnsupportedCacheGeometryCannotClaimAllLayerCacheFit()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        var minimum = (long[])geometry.ExpertCacheMinimumBytes.Clone();
        minimum[3] = 0;
        Assert.Equal(0, (geometry with { ExpertCacheMinimumBytes = minimum }).AllLayerExpertCacheMinimumBytes);
        minimum[3] = long.MaxValue;
        Assert.Throws<OverflowException>(() => (geometry with { ExpertCacheMinimumBytes = minimum }).AllLayerExpertCacheMinimumBytes);
    }

    [Fact]
    public void CheckpointForecastIncludesBothRetentionCapsAndPublicationTransient()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        var none = geometry.RequestPeak(512, 8);
        var one = geometry.RequestPeak(512, 8, checkpointCopies: 1);
        var bothCapsAndTransient = geometry.RequestPeak(512, 8, checkpointCopies: 5);
        Assert.True(one.Host > none.Host);
        Assert.Equal(5 * (one.Host - none.Host), bothCapsAndTransient.Host - none.Host);
        Assert.Equal(none.Device, bothCapsAndTransient.Device);
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.RequestPeak(512, 8, checkpointCopies: -1));
        Assert.Throws<OverflowException>(() => (geometry with { PleStateBytes = long.MaxValue / 4 }).RequestPeak(512, 8, checkpointCopies: 5));
    }

    [Fact]
    public void JointArenaHighWaterRequiresBothContextAndLaneAxes()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        var twoLong = geometry.JointDecodePeak(4096, 2);
        var eightShort = geometry.JointDecodePeak(256, 8);
        var eightAtRetainedCapacity = geometry.JointDecodePeak(4096, 8);
        // A replacement arena retains the old 4096-row capacity but has nine
        // state lanes. Taking the maximum of the two old byte totals omits
        // that combination; it is not a valid reservation for the new shape.
        Assert.True(eightAtRetainedCapacity.Device > Math.Max(twoLong.Device, eightShort.Device));
        Assert.True(eightAtRetainedCapacity.Host > Math.Max(twoLong.Host, eightShort.Host));
        Assert.Equal(geometry.JointDecodePeak(257, 2), geometry.JointDecodePeak(512, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.JointDecodePeak(0, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.JointDecodePeak(256, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.JointDecodePeak(256, 9));
        Assert.Throws<OverflowException>(() => (geometry with { PleStateBytes = long.MaxValue }).JointDecodePeak(256, 2));
    }

    [Fact]
    public void EveryRequestForecastIncludesCpuFallbackWithout128RowDiscontinuity()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        Assert.Equal(4, geometry.ExpertCount);
        // A one-row cache miss stages only its two selected experts, not zero
        // and not the full four-expert layer.
        var decode = geometry.HostPrefillPeak(1, 8);
        Assert.Equal(6528 + 63, decode.ExpertStageBytes);
        Assert.True(decode.GraphTensorBytes > 0 && decode.GraphWorkBytes > 0 && decode.AuxiliaryBytes > 0);
        var at128 = geometry.HostPrefillPeak(128, 8);
        var wide = geometry.HostPrefillPeak(129, 8);
        // 4 Q8_0 experts * 32x32 * 34/32, three projections; one arena tail.
        Assert.Equal(13056 + 63, wide.ExpertStageBytes);
        Assert.Equal(at128.ExpertStageBytes, wide.ExpertStageBytes);
        Assert.Equal(wide.TotalBytes - at128.TotalBytes,
            geometry.HostPrefillPeak(130, 8).TotalBytes - wide.TotalBytes);
        Assert.True(wide.GraphTensorBytes > 0);
        Assert.True(wide.GraphWorkBytes > (4L << 20));
        long oldHostPerRow = 4L * (3L * geometry.Hidden + geometry.HyperHidden) + 12 + 2L * 512;
        var narrow = geometry.RequestPeak(512, 128);
        var request = geometry.RequestPeak(512, 129);
        Assert.Equal(oldHostPerRow + geometry.HostPrefillPeak(129).TotalBytes - geometry.HostPrefillPeak(128).TotalBytes,
            request.Host - narrow.Host);
        Assert.True(request.Device > narrow.Device);
    }

    [Fact]
    public void HostStageBoundsDistinctRoutesAndReusesLargestLayer()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file) with
        {
            Layers = 2, ExpertCount = 1024, UsedExperts = 2,
            ExpertProjectionBytes = [[1024L * 64, 1024L * 64, 1024L * 64],
                [1024L * 128, 1024L * 192, 1024L * 256]]
        };
        // At 129 rows only 258 distinct experts can be touched. Parts are
        // already aligned. The first layer is not retained alongside the second.
        Assert.Equal(2L * (128 + 192 + 256) + 63, geometry.HostPrefillPeak(1, 8).ExpertStageBytes);
        Assert.Equal(256L * (128 + 192 + 256) + 63, geometry.HostPrefillPeak(128, 8).ExpertStageBytes);
        Assert.Equal(258L * (128 + 192 + 256) + 63, geometry.HostPrefillPeak(129, 8).ExpertStageBytes);
        Assert.Equal(1024L * (128 + 192 + 256) + 63, geometry.HostPrefillPeak(512, 8).ExpertStageBytes);
        Assert.Equal(geometry.HostPrefillPeak(512, 8).ExpertStageBytes, geometry.HostPrefillPeak(1024, 8).ExpertStageBytes);
        var reversed = geometry with { ExpertProjectionBytes = geometry.ExpertProjectionBytes.Reverse().ToArray() };
        Assert.Equal(geometry.HostPrefillPeak(512, 8), reversed.HostPrefillPeak(512, 8));
    }

    [Fact]
    public void HostWorkIncludesExplicitThreadsAndSeparateArenaAlignment()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file) with
        {
            Layers = 1, ExpertCount = 2, UsedExperts = 1,
            ExpertProjectionBytes = [[130, 132, 134]]
        };
        Assert.Equal(3 * 128 + 63, geometry.HostPrefillPeak(1, 1).ExpertStageBytes);
        Assert.Equal(3 * 192 + 63, geometry.HostPrefillPeak(129, 1).ExpertStageBytes);
        var eight = geometry.HostPrefillPeak(129, 8);
        var sixtyFive = geometry.HostPrefillPeak(129, 65);
        Assert.Equal(57 * ((512L << 10) + 128), sixtyFive.GraphWorkBytes - eight.GraphWorkBytes);
        Assert.Equal(eight.ExpertStageBytes, sixtyFive.ExpertStageBytes);
        Assert.Equal(eight.GraphTensorBytes, sixtyFive.GraphTensorBytes);
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.HostPrefillPeak(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.HostPrefillPeak(129, -1));
    }

    [Fact]
    public void CpuFallbackMetadataKeepsMixedQuantizationSizes()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry(mixedExpertQuantization: true));
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        // Q8_0 / IQ4_NL / F16 have different actual row sizes.
        Assert.Equal(new long[] {4352, 2304, 8192}, geometry.ExpertProjectionBytes[7]);
        Assert.Equal(14848, geometry.ExpertLayerBytes[7]);
        Assert.Equal(7424 + 63, geometry.HostPrefillPeak(1, 8).ExpertStageBytes);
        Assert.Equal(14848 + 63, geometry.HostPrefillPeak(512, 8).ExpertStageBytes);
    }

    [Fact]
    public void CpuFallbackFailsClosedForMissingGeometryAndOverflow()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        Assert.Throws<NotSupportedException>(() => (geometry with { ExpertCount = 0 }).HostPrefillPeak(1));
        Assert.Throws<NotSupportedException>(() => (geometry with { ExpertCount = 0 }).HostPrefillPeak(129));
        Assert.Throws<NotSupportedException>(() => (geometry with { ExpertProjectionBytes = [] }).HostPrefillPeak(129));
        Assert.Throws<NotSupportedException>(() => (geometry with { Layers = 1, ExpertProjectionBytes = [[1, 2, 3]] }).HostPrefillPeak(129));
        Assert.Throws<OverflowException>(() => (geometry with
        { Layers = 1, ExpertCount = 1, UsedExperts = 1, ExpertProjectionBytes = [[long.MaxValue, 1, 1]] }).HostPrefillPeak(129));
        Assert.Throws<OverflowException>(() => (geometry with { Hidden = int.MaxValue, ExpertFf = int.MaxValue,
            UsedExperts = int.MaxValue, ExpertCount = int.MaxValue, Layers = 1,
            ExpertProjectionBytes = [[int.MaxValue, int.MaxValue, int.MaxValue]] }).HostPrefillPeak(int.MaxValue));
        Assert.Throws<OverflowException>(() => new Qwen4ExpHostPrefillMemory(long.MaxValue, 1, 0, 0).TotalBytes);
    }

    [Fact]
    public void CudaStreamForecastCoversShortOverridesAndDoesNotMultiplyQuantizedTails()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        Assert.Equal(new long[] {510, 510, 510}, geometry.ExpertProjectionCudaTailBytes[0]);
        var one = geometry.DeviceExpertStreamPeak(1);
        // Two Q8_0 experts per projection: align256(2176 + one 510-byte tail).
        Assert.Equal(3 * 2816, one.ExpertWeightBytes);
        Assert.Equal(2L << 20, one.AllocationSlackBytes);
        var at127 = geometry.DeviceExpertStreamPeak(127);
        var at128 = geometry.DeviceExpertStreamPeak(128);
        var at129 = geometry.DeviceExpertStreamPeak(129);
        Assert.Equal(3 * 4864, at128.ExpertWeightBytes);
        Assert.Equal(at127.ExpertWeightBytes, at128.ExpertWeightBytes);
        Assert.Equal(at128.ExpertWeightBytes, at129.ExpertWeightBytes);
        Assert.Equal(at128.GraphTensorBytes - at127.GraphTensorBytes, at129.GraphTensorBytes - at128.GraphTensorBytes);
        Assert.True(one.GraphTensorBytes > 0);
        // Do not skip the forecast below the native default 128: its cached
        // environment override may enable the CUDA stream at smaller batches.
        Assert.True(one.TotalBytes > one.ExpertWeightBytes);
    }

    [Fact]
    public void CudaStreamBoundsUniqueRoutesAndUsesOneLargestLayerArena()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file) with
        {
            Layers = 2, ExpertCount = 1024, UsedExperts = 2,
            ExpertProjectionBytes = [[1024L * 256, 1024L * 256, 1024L * 256],
                [1024L * 512, 1024L * 768, 1024L * 1024]],
            ExpertProjectionCudaTailBytes = [[0, 0, 0], [0, 0, 0]]
        };
        Assert.Equal(2L * (512 + 768 + 1024), geometry.DeviceExpertStreamPeak(1).ExpertWeightBytes);
        Assert.Equal(258L * (512 + 768 + 1024), geometry.DeviceExpertStreamPeak(129).ExpertWeightBytes);
        Assert.Equal(1024L * (512 + 768 + 1024), geometry.DeviceExpertStreamPeak(512).ExpertWeightBytes);
        Assert.Equal(geometry.DeviceExpertStreamPeak(512).ExpertWeightBytes, geometry.DeviceExpertStreamPeak(1024).ExpertWeightBytes);
        var reverse = geometry with { ExpertProjectionBytes = geometry.ExpertProjectionBytes.Reverse().ToArray(),
            ExpertProjectionCudaTailBytes = geometry.ExpertProjectionCudaTailBytes.Reverse().ToArray() };
        Assert.Equal(geometry.DeviceExpertStreamPeak(129), reverse.DeviceExpertStreamPeak(129));
    }

    [Fact]
    public void CudaStreamMetadataKeepsMixedQuantizedAndUnquantizedTails()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry(mixedExpertQuantization: true));
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        Assert.Equal(new long[] {510, 270, 0}, geometry.ExpertProjectionCudaTailBytes[7]);
        // Largest layer: Q8_0 + IQ4_NL + F16, padded/aligned independently.
        Assert.Equal(4864 + 2816 + 8192, geometry.DeviceExpertStreamPeak(512).ExpertWeightBytes);
        Assert.Equal(14848 + 63, geometry.HostPrefillPeak(512, 8).ExpertStageBytes);
    }

    [Fact]
    public void RequestDeviceBoundNeverDecreasesOrCountsRoutedActivationsTwice()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        // Fixed pre-stream bounds for this eight-layer synthetic checkpoint,
        // F16 KV and 512-token capacity. These include the retained decode graph.
        foreach (var (rows, previousDeviceBytes) in new (int, long)[]
            { (1, 135021344), (128, 138109984), (512, 147448864) })
        {
            var stream = geometry.DeviceExpertStreamPeak(rows);
            long oldRoutedAllowance = 1792L * rows;
            long extra = stream.ExpertWeightBytes + stream.AllocationSlackBytes
                + Math.Max(0, stream.GraphTensorBytes - oldRoutedAllowance);
            long current = geometry.RequestPeak(512, rows).Device;
            Assert.Equal(previousDeviceBytes + extra, current);
            Assert.True(current >= previousDeviceBytes);
        }
    }

    [Fact]
    public void CudaStreamFailsClosedForMissingPaddingAndArithmeticOverflow()
    {
        using var file = GgufFile.OpenForMemoryPlanning(WriteGeometry());
        var geometry = Qwen4ExpUnifiedGeometry.Read(file);
        Assert.Throws<ArgumentOutOfRangeException>(() => geometry.DeviceExpertStreamPeak(0));
        Assert.Throws<NotSupportedException>(() => (geometry with { ExpertProjectionCudaTailBytes = [] }).DeviceExpertStreamPeak(128));
        Assert.Throws<NotSupportedException>(() => (geometry with { Layers = 1, ExpertProjectionBytes = [[4, 4, 4]],
            ExpertProjectionCudaTailBytes = [[0, -1, 0]] }).DeviceExpertStreamPeak(128));
        Assert.Throws<OverflowException>(() => (geometry with { Layers = 1, ExpertProjectionBytes = [[4, 4, 4]],
            ExpertProjectionCudaTailBytes = [[long.MaxValue, 0, 0]] }).DeviceExpertStreamPeak(128));
        Assert.Throws<OverflowException>(() => (geometry with { Hidden = int.MaxValue, ExpertFf = int.MaxValue,
            ExpertCount = int.MaxValue, UsedExperts = int.MaxValue, Layers = 1,
            ExpertProjectionBytes = [[int.MaxValue, int.MaxValue, int.MaxValue]],
            ExpertProjectionCudaTailBytes = [[0, 0, 0]] }).DeviceExpertStreamPeak(int.MaxValue));
        Assert.Throws<OverflowException>(() => new Qwen4ExpDeviceExpertStreamMemory(long.MaxValue, 1, 0).TotalBytes);
    }

    [Fact]
    public void HeaderOnlyPlanBoundsAllImagesBeforeAnyDecoderRuns()
    {
        string a = Png("a.png", 256, 256), b = Png("b.png", 256, 256);
        // These files intentionally contain only headers; planning must not
        // decode pixels or require the complete PNG payload.
        var one = QwenVisionPreparationPlan.Create([a], 10, 1, 1000, Vision());
        var two = QwenVisionPreparationPlan.Create([a, b], 10, 2, 1000, Vision());
        Assert.Equal(73, one.ExpandedPromptTokens);
        Assert.Equal(136, two.ExpandedPromptTokens);
        Assert.Equal(2 * one.RetainedEmbeddingBytes, two.RetainedEmbeddingBytes);
        Assert.Equal(one.HostPeakBytes + one.Images[0].HostPreparationBytes + one.RetainedEmbeddingBytes, two.HostPeakBytes);
        Assert.Equal(one.DevicePeakBytes + one.RetainedEmbeddingBytes, two.DevicePeakBytes);
        Assert.Throws<InvalidOperationException>(() => QwenVisionPreparationPlan.Create([a, b], 10, 2, 100, Vision()));
        Assert.Throws<ArgumentException>(() => QwenVisionPreparationPlan.Create([a, b], 10, 1, 1000, Vision()));
        one.ValidateFilesUnchanged();
        using (var output = File.Open(a, FileMode.Append)) output.WriteByte(0);
        Assert.Throws<IOException>(() => one.ValidateFilesUnchanged());
    }

    [Fact]
    public void DifferentImageGridsRetainBothGeometryCaches()
    {
        string square = Png("square.png", 256, 256), wide = Png("wide.png", 512, 256);
        var geometry = Vision();
        var a = QwenVisionPreparationPlan.Create([square], 10, 1, 1000, geometry);
        var b = QwenVisionPreparationPlan.Create([wide], 10, 1, 1000, geometry);
        var both = QwenVisionPreparationPlan.Create([square, wide], 10, 2, 1000, geometry);
        Assert.Equal(b.DevicePeakBytes + a.RetainedEmbeddingBytes + geometry.GeometryCacheBytes(256, 256), both.DevicePeakBytes);
        Assert.Equal(b.HostPeakBytes + a.Images[0].HostPreparationBytes + a.RetainedEmbeddingBytes
            + geometry.GeometryCacheBytes(256, 256), both.HostPeakBytes);
    }

    [Fact]
    public void HeaderParserAcceptsBoundedJpegAndRejectsMalformedSources()
    {
        string path = Path.Combine(_directory, "header.jpg");
        // SOI, APP0, SOF0: a complete bounded header without compressed pixels.
        File.WriteAllBytes(path, [255,216, 255,224,0,10, 0,0,0,0,0,0,0,0,
            255,192,0,11, 8,1,0,2,0, 1,1,0x11,0]);
        Assert.Equal((512, 256), QwenVisionPreparationPlan.ReadBoundedDimensions(path));
        Assert.Equal((256, 256), QwenVisionPreparationPlan.ReadBoundedDimensions(Png("valid.png", 256, 256)));
        Assert.Throws<InvalidDataException>(() => QwenVisionPreparationPlan.ReadBoundedDimensions(Png("zero.png", 0, 256)));
        Assert.Throws<InvalidDataException>(() => QwenVisionPreparationPlan.ReadBoundedDimensions(Png("overflow.png", uint.MaxValue, 256)));
        File.WriteAllBytes(path, new byte[24]);
        Assert.Throws<NotSupportedException>(() => QwenVisionPreparationPlan.ReadBoundedDimensions(path));
        File.WriteAllBytes(path, [255,216]);
        Assert.Throws<EndOfStreamException>(() => QwenVisionPreparationPlan.ReadBoundedDimensions(path));
        File.WriteAllBytes(path, [255,216,255,224,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
        Assert.Throws<InvalidDataException>(() => QwenVisionPreparationPlan.ReadBoundedDimensions(path));
    }

    [Fact]
    public void ImageGeometryRejectsOverflowBeforePixelAllocation()
    {
        Assert.Throws<NotSupportedException>(() => QwenVisionPreparationPlan.Create([Png("huge.png", 100000, 100000)], 10, 1, 1000, Vision()));
        Assert.Throws<ArgumentOutOfRangeException>(() => Vision().ExecutionPeak(255, 256));
        Assert.Throws<NotSupportedException>(() => Vision().ExecutionPeak(int.MaxValue - 31, int.MaxValue - 31));
        Assert.Throws<OverflowException>(() => (Vision() with { WeightBytes = long.MaxValue }).DeviceWeightBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void VisionConstructorFailureReleasesPublishedAndUnpublishedTensors(bool afterAllocation)
    {
        string path = WriteTinyVision();
        var allocator = new FaultAllocator(2, afterAllocation);
        Assert.Throws<OutOfMemoryException>(() => new Qwen35VisionEncoder(path, allocator));
        Assert.Equal(afterAllocation ? 2 : 1, allocator.Created.Count);
        Assert.All(allocator.Created, storage => Assert.Equal(IntPtr.Zero, storage.buffer));
        using var exclusive = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
    }

    [Fact]
    public void VisionDisposalReleasesEveryWeightAndIsIdempotent()
    {
        var allocator = new FaultAllocator(0, false);
        var encoder = new Qwen35VisionEncoder(WriteTinyVision(), allocator);
        Assert.All(allocator.Created, storage => Assert.NotEqual(IntPtr.Zero, storage.buffer));
        encoder.Dispose();
        encoder.Dispose();
        Assert.All(allocator.Created, storage => Assert.Equal(IntPtr.Zero, storage.buffer));
    }

    private string WriteTinyVision()
    {
        string path = Path.Combine(_directory, "tiny-vision.gguf");
        Write(path, [], [Gen("test.first.weight", .1f, 4), Gen("test.second.weight", .1f, 4)]);
        return path;
    }

    private sealed class FaultAllocator(int failAt, bool afterAllocation) : IAllocator
    {
        private int _attempt;
        public List<CpuStorage> Created { get; } = [];
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public float GetAllocatedMemoryRatio() => 0;
        public Storage Allocate(DType type, long count)
        {
            bool fail = ++_attempt == failAt;
            if (fail && !afterAllocation) throw new OutOfMemoryException("Injected vision payload refusal.");
            var storage = new FaultStorage(this, type, count, fail);
            Created.Add(storage);
            return storage;
        }
    }

    private sealed class FaultStorage(IAllocator owner, DType type, long count, bool failOnWrite)
        : CpuStorage(owner, type, count)
    {
        public override void SetElementsAsFloat(long index, float[] values)
        {
            if (failOnWrite) throw new OutOfMemoryException("Injected vision payload initialization failure.");
            base.SetElementsAsFloat(index, values);
        }
    }

    private static Qwen35VisionMemoryGeometry Vision() => new(16, 2, 64, 128, 4, 2, 32, 64, 4096, 2048, 128);
    private string Png(string name, uint width, uint height)
    {
        byte[] header = new byte[24];
        new byte[] {137,80,78,71,13,10,26,10}.CopyTo(header, 0);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(8, 4), 13);
        "IHDR"u8.CopyTo(header.AsSpan(12));
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(16, 4), width);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(20, 4), height);
        string path = Path.Combine(_directory, name); File.WriteAllBytes(path, header); return path;
    }

    private string WriteGeometry(bool mixedExpertQuantization = false)
    {
        const string a = "qwen4exp.";
        var kv = new List<Kv> { new Str { Key = "general.architecture", V = "qwen4exp" },
            new I32Arr { Key = a + "attention.compress_ratios", V = new[] {0,0,0,4,0,0,0,4} } };
        foreach (var (key, value) in new (string, uint)[] { ("block_count",8),("context_length",4096),
            ("embedding_length",32),("attention.head_count",4),("attention.head_count_kv",2),("attention.key_length",32),
            ("hyper_connection.count",4),("ssm.conv_kernel",4),("ssm.state_size",8),("ssm.group_count",2),
            ("ssm.time_step_rank",4),("expert_feed_forward_length",32),("expert_used_count",2),("expert_count",4),
            ("attention.indexer.head_count",2),("attention.indexer.key_length",8) }) kv.Add(new U32 { Key = a + key, V = value });
        var tensors = new List<SyntheticGguf.Tensor>();
        var embedding = Gen("token_embd.weight", 0.1f, 32, 8); embedding.Type = GgmlType.Q8_0; tensors.Add(embedding);
        var norm = Gen("output_norm.weight", 0.1f, 32); norm.Type = GgmlType.F16; tensors.Add(norm);
        tensors.Add(Gen("per_layer_token_embd.weight", 0.1f, 32, 16));
        for (int layer = 0; layer < 8; layer++) foreach (string part in new[] { "gate", "up", "down" })
        {
            string name = $"blk.{layer}.ffn_{part}_exps.weight";
            var tensor = mixedExpertQuantization && layer == 7 && part == "up"
                ? Blocks(name, GgmlType.IQ4_NL, .1f, 32, 32, 4) : Gen(name, 0.1f, 32, 32, 4);
            if (tensor.RawBlocks is null)
                tensor.Type = mixedExpertQuantization && layer == 7 && part == "down" ? GgmlType.F16 : GgmlType.Q8_0;
            tensors.Add(tensor);
        }
        string path = Path.Combine(_directory, "geometry.gguf"); Write(path, kv, tensors); return path;
    }
}
