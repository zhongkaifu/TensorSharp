// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp.Memory;
using TensorSharp.MLX;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class HostMemoryPortabilityTests
{
    private const long GiB = 1L << 30;

    [Fact]
    public void AppleUsesKernelPageSizeAndCreditsFreeAndPurgeableOnly()
    {
        Assert.Equal((8 * GiB, 3 * GiB), HostMemoryInfo.FromAppleCounters(
            8 * (ulong)GiB, 16384, 131072, 65536, null));
        Assert.Equal((8 * GiB, 3 * GiB), HostMemoryInfo.FromAppleCounters(
            8 * (ulong)GiB, 4096, 524288, 262144, null));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1073741824, 1073741824)]
    [InlineData(17179869184, 3221225472)]
    public void AppleAppRemainingAllowanceIsAnIndependentCeiling(long remaining, long expected)
    {
        var sample = HostMemoryInfo.FromAppleCounters(8 * (ulong)GiB, 16384, 131072, 65536, (ulong)remaining);
        Assert.Equal(8 * GiB, sample.Total);
        Assert.Equal(expected, sample.Available);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(12345)]
    [InlineData(4294967296)]
    public void InvalidApplePageSizeFailsClosed(long pageSize) => Assert.Throws<IOException>(() =>
        HostMemoryInfo.FromAppleCounters(8 * (ulong)GiB, (ulong)pageSize, 1, 0, null));

    [Fact]
    public void InvalidCapacityFailsClosedBeforeAppLimitCanHideIt()
    {
        Assert.Throws<IOException>(() => HostMemoryInfo.FromAppleCounters(4096, 4096, 2, 0, 0));
        Assert.Throws<IOException>(() => HostMemoryInfo.Validate(0, 0));
        Assert.Throws<IOException>(() => HostMemoryInfo.Validate(ulong.MaxValue, 0));
        Assert.Throws<IOException>(() => HostMemoryInfo.Validate(1024, 1025));
    }

    [Fact]
    public void AppleVmStatisticsMatchesStableMachAbi()
    {
        Assert.Equal(15 * 4, Marshal.SizeOf<HostMemoryInfo.AppleVmStatistics>());
        Assert.Equal(12 * 4, (int)Marshal.OffsetOf<HostMemoryInfo.AppleVmStatistics>("Purgeable"));
        Assert.Equal(14 * 4, (int)Marshal.OffsetOf<HostMemoryInfo.AppleVmStatistics>("Speculative"));
    }

    [Theory]
    [InlineData(1, 0)]
    [InlineData(4, 2048)]
    [InlineData(8, 6144)]
    [InlineData(32, 27852)]
    public void MlxWiredDefaultPreservesBothHeadroomBounds(long gib, long expectedMiB)
    {
        long bytes = gib * GiB;
        long limit = MlxNative.DefaultWiredLimitMiB(bytes);
        Assert.Equal(expectedMiB, limit);
        Assert.True(limit == 0 || bytes / (1 << 20) - limit >= 2048);
        Assert.True(limit <= bytes / (1 << 20) * 85 / 100);
    }

    [Fact]
    public void CurrentHostObservationIsFiniteWithoutInitializingAnAccelerator()
    {
        var sample = HostMemoryInfo.Capture();
        Assert.True(sample.Total > 0);
        Assert.InRange(sample.Available, 0, sample.Total);
    }
}
