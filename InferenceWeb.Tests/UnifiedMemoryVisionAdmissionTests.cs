// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Real reservation arithmetic and header-derived image plans, without
/// starting a native backend or claiming codec/device execution coverage.</summary>
public sealed class UnifiedMemoryVisionAdmissionTests
{
    private const long MiB = 1L << 20;
    private const long GiB = 1L << 30;
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const string Device = UnifiedMemoryModelSession.DevicePool;

    [Fact]
    public void TextRequestsWithProjectorDoNotReserveUnusedTowerPerRequest()
    {
        var model = Session(vision: true);
        var plain = Session(vision: false);
        var peak = new InferenceMemoryBytes(128 * MiB, 256 * MiB);
        var charges = model.IncrementalPeak(peak, 2 * GiB);
        Assert.Equal(plain.IncrementalPeak(peak, 2 * GiB), charges);
        var budget = new MemoryBudget([new(Host, 8 * GiB), new(Device, 3 * GiB)]);
        using var existingTower = budget.TryReserve([new(Device, 2 * GiB)])!;
        existingTower.Commit();
        using var first = budget.TryReserve(charges);
        using var second = budget.TryReserve(charges);
        Assert.NotNull(first); Assert.NotNull(second);
        var usage = budget.Snapshot().Single(p => p.Pool == Device);
        Assert.Equal(2 * GiB, usage.Committed);
        Assert.Equal(512 * MiB, usage.Reserved);
    }

    [Fact]
    public void HeaderPlannedMediaRetainsFullColdTowerAndExecutionPeak()
    {
        var plan = ImagePlan();
        Assert.True(plan.DevicePeakBytes > Vision().DeviceWeightBytes);
        var peak = UnifiedMemoryModelSession.VisionPreparationPeak(plan);
        Assert.Equal(plan.DevicePeakBytes, peak.Device);
        Assert.Equal(plan.HostPeakBytes, peak.Host);
        var charges = Session(vision: true).IncrementalPeak(peak, GiB);
        long needed = charges.Single(c => c.Pool == Device).Bytes;
        Assert.Equal(GiB + plan.DevicePeakBytes, needed);
        var tooSmall = new MemoryBudget([new(Host, 8 * GiB), new(Device, needed - 1)]);
        Assert.Null(tooSmall.TryReserve(charges));
        var exact = new MemoryBudget([new(Host, 8 * GiB), new(Device, needed)]);
        using var admitted = exact.TryReserve(charges);
        Assert.NotNull(admitted);
    }

    [Fact]
    public void MediaForecastDoesNotDeductLiveTowerOwnersFromLedger()
    {
        var peak = UnifiedMemoryModelSession.VisionPreparationPeak(ImagePlan());
        var charges = Session(vision: true).IncrementalPeak(peak, 2 * GiB);
        var budget = new MemoryBudget([new(Host, 8 * GiB), new(Device, peak.Device + GiB - 1)]);
        using var boundTower = budget.TryReserve([new(Device, GiB)])!;
        boundTower.Commit();
        Assert.Null(budget.TryReserve(charges));
        Assert.Equal(GiB, budget.Snapshot().Single(p => p.Pool == Device).Committed);
    }

    [Theory]
    [InlineData(0L, 2304L)]
    [InlineData(2048L, 256L)]
    [InlineData(3072L, 256L)]
    public void OnlyExplicitTrunkPreloadCreditReducesFutureTextBindings(long boundMiB, long expectedMiB)
    {
        var charges = Session(vision: true).IncrementalPeak(new(0, 256 * MiB), boundMiB * MiB);
        Assert.Equal(expectedMiB * MiB, charges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(64 * MiB, charges.Single(c => c.Pool == Host).Bytes);
    }

    [Fact]
    public void ReservationArithmeticRejectsOverflowAndInvalidPreloadObservation()
    {
        var model = Session(vision: true);
        Assert.Throws<OverflowException>(() => model.IncrementalPeak(new(0, long.MaxValue), 0));
        Assert.Throws<OverflowException>(() => model.IncrementalPeak(new(long.MaxValue, 0), 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.IncrementalPeak(new(0, 0), -1));
    }

    private static UnifiedMemoryModelSession Session(bool vision)
    {
        // Exercise the actual instance planner with explicit cache observations;
        // constructor hardware discovery/native scopes are deliberately omitted.
        var session = (UnifiedMemoryModelSession)RuntimeHelpers.GetUninitializedObject(typeof(UnifiedMemoryModelSession));
        const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
        typeof(UnifiedMemoryModelSession).GetField("_geometry", fields)!.SetValue(session,
            new Qwen4ExpUnifiedGeometry { LoadingDeviceBoundBytes = 2 * GiB });
        typeof(UnifiedMemoryModelSession).GetField("_vision", fields)!.SetValue(session, vision ? Vision() : null);
        return session;
    }

    private static Qwen35VisionMemoryGeometry Vision() => new(16, 2, 64, 128, 4, 2, 32, 64,
        2 * GiB, 2048, 128);

    private static QwenVisionPreparationPlan ImagePlan()
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-vision-admission-" + Guid.NewGuid() + ".png");
        byte[] header = new byte[24];
        new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }.CopyTo(header, 0);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(8, 4), 13);
        "IHDR"u8.CopyTo(header.AsSpan(12));
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(16, 4), 32);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(20, 4), 32);
        try
        {
            File.WriteAllBytes(path, header);
            return QwenVisionPreparationPlan.Create([path], 10, 1, 1024, Vision());
        }
        finally { File.Delete(path); }
    }
}
