// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Hardware;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using Xunit;
using System.Runtime.InteropServices;

namespace InferenceWeb.Tests;

public sealed class HostCommitPlanningTests
{
    private const long GiB = 1L << 30;

    [Theory]
    [InlineData(8, 0, 3, 5)]
    [InlineData(0, 7, 3, 4)]
    [InlineData(8, 7, 3, 4)]
    [InlineData(8, 7, 9, 0)]
    public void ImmediateJobLimitsUseCurrentPrivateCommit(long processLimit, long jobLimit, long used, long free)
    {
        var system = HostMemoryInfo.FromWindowsCounters(32, 17, 40, 12);
        var actual = HostMemoryInfo.ApplyWindowsJobCommit(system,
            processLimit == 0 ? null : (ulong)processLimit,
            jobLimit == 0 ? null : (ulong)jobLimit, (ulong)used, true);
        Assert.Equal(free, actual.CommitAvailable);
        Assert.Equal(jobLimit == 0 ? processLimit : jobLimit, actual.CommitLimit);
        Assert.Equal(system.Total, actual.Total);
        Assert.Equal(system.Available, actual.Available);
        Assert.Contains("ancestors unverified", actual.CommitConstraintScope!);
    }

    [Fact]
    public void JobIntersectionNeverGrantsMoreThanSystemAllowance()
    {
        var actual = HostMemoryInfo.ApplyWindowsJobCommit(
            HostMemoryInfo.FromWindowsCounters(32, 17, 6, 2), 8, 7, 1, true);
        Assert.Equal(6, actual.CommitLimit);
        Assert.Equal(2, actual.CommitAvailable);
    }

    [Fact]
    public void SharedJobAggregateCannotBeGuessedFromThisProcess()
    {
        var system = HostMemoryInfo.FromWindowsCounters(32, 17, 40, 12);
        var error = Assert.Throws<IOException>(() => HostMemoryInfo.ApplyWindowsJobCommit(system, 8, 7, 1, false));
        Assert.Contains("aggregate commitment is unknown", error.Message);
        // A per-process limit is observable even when other processes share the Job.
        Assert.Equal(7, HostMemoryInfo.ApplyWindowsJobCommit(system, 8, null, 1, false).CommitAvailable);
    }

    [Theory]
    [InlineData(0UL, 1UL)]
    [InlineData(ulong.MaxValue, 1UL)]
    [InlineData(8UL, ulong.MaxValue)]
    public void InvalidJobCounterCannotBecomeUnlimited(ulong limit, ulong used)
        => Assert.Throws<IOException>(() => HostMemoryInfo.ApplyWindowsJobCommit(
            HostMemoryInfo.FromWindowsCounters(32, 17, 40, 12), limit, null, used, true));

    [Fact]
    public void WindowsNativeStructuresMatchPointerWidth()
    {
        // These layouts are used by documented Windows Job and process APIs.
        Assert.Equal(IntPtr.Size == 8 ? 64 : 48, Marshal.SizeOf<HostMemoryInfo.WindowsJobBasicLimits>());
        Assert.Equal(IntPtr.Size == 8 ? 144 : 112, Marshal.SizeOf<HostMemoryInfo.WindowsJobLimits>());
        Assert.Equal(IntPtr.Size == 8 ? 80 : 44, Marshal.SizeOf<HostMemoryInfo.WindowsProcessMemory>());
        Assert.Equal(IntPtr.Size == 8 ? 112 : 96,
            (int)Marshal.OffsetOf<HostMemoryInfo.WindowsJobLimits>(nameof(HostMemoryInfo.WindowsJobLimits.ProcessLimit)));
        Assert.Equal(IntPtr.Size == 8 ? 72 : 40,
            (int)Marshal.OffsetOf<HostMemoryInfo.WindowsProcessMemory>(nameof(HostMemoryInfo.WindowsProcessMemory.PrivateUsage)));
    }

    [Fact]
    public void CommitPressureDoesNotRewritePhysicalOrPrefaultCapacity()
    {
        var sample = HostMemoryInfo.FromWindowsCounters(32UL << 30, 17UL << 30, 40UL << 30, 4UL << 30);
        Assert.Equal(32 * GiB, sample.Total);
        Assert.Equal(17 * GiB, sample.Available);
        Assert.Equal(40 * GiB, sample.CommitLimit);
        Assert.Equal(4 * GiB, sample.CommitAvailable);
        var hw = new InferenceHardwareMemory(sample.Total, sample.Available, 16 * GiB, 12 * GiB)
        { HostCommitLimit = sample.CommitLimit, HostCommitAvailable = sample.CommitAvailable };
        Assert.Equal((32 * GiB, 4 * GiB), hw.HostAllocationCapacity());
    }

    [Fact]
    public void ExhaustedCommitIsZeroAllowanceRatherThanUnknown()
    {
        var sample = HostMemoryInfo.FromWindowsCounters(32UL << 30, 17UL << 30, 40UL << 30, 0);
        var hw = new InferenceHardwareMemory(sample.Total, sample.Available, 16 * GiB, 12 * GiB)
        { HostCommitLimit = sample.CommitLimit, HostCommitAvailable = sample.CommitAvailable };
        Assert.Equal(0, hw.HostAllocationCapacity().Available);
    }

    [Fact]
    public void UnknownCommitPreservesExistingCrossPlatformPlan()
    {
        var hw = new InferenceHardwareMemory(32 * GiB, 17 * GiB, 16 * GiB, 12 * GiB);
        Assert.Equal((32 * GiB, 17 * GiB), hw.HostAllocationCapacity());
        var observations = new List<PhysicalMemoryObservation>();
        AdaptiveModelSession.AddHostCommitObservation(observations, hw, DateTimeOffset.UtcNow, 8 * GiB, GiB, 0);
        Assert.Empty(observations);
    }

    [Fact]
    public void InvalidWindowsCommitObservationFailsClosed()
    {
        Assert.Throws<IOException>(() => HostMemoryInfo.FromWindowsCounters(32, 17, 0, 0));
        Assert.Throws<IOException>(() => HostMemoryInfo.FromWindowsCounters(32, 17, ulong.MaxValue, 0));
        Assert.Throws<IOException>(() => HostMemoryInfo.FromWindowsCounters(32, 17, 40, 41));
        Assert.Throws<IOException>(() => HostMemoryInfo.FromWindowsCounters(32, 33, 40, 4));
    }

    [Theory]
    [InlineData(32, 33)]
    [InlineData(32, -1)]
    [InlineData(0, 0)]
    public void CommitCeilingCannotHideInvalidPhysicalObservation(long total, long available)
    {
        var hw = new InferenceHardwareMemory(total, available, 16, 12)
        { HostCommitLimit = 40, HostCommitAvailable = 4 };
        Assert.Throws<ArgumentException>(() => hw.HostAllocationCapacity());
    }

    [Theory]
    [InlineData(null, 1L)]
    [InlineData(1L, null)]
    [InlineData(0L, 0L)]
    [InlineData(8L, -1L)]
    [InlineData(8L, 9L)]
    public void MalformedInjectedPairCannotGrantAllowance(long? total, long? available)
    {
        var hw = new InferenceHardwareMemory(32, 17, 16, 12)
        { HostCommitLimit = total, HostCommitAvailable = available };
        Assert.Throws<ArgumentException>(() => hw.HostAllocationCapacity());
    }

    [Fact]
    public void ResidentLoadThatFitsRamCanStillFailCommitAdmission()
    {
        var profile = new DenseMemoryProfile(new()
        {
            DenseWeights = new(4 * GiB, 4 * GiB), Persistent = new(16 << 20, 16 << 20),
            KvCaches = [new(65536, AllocationBlockTokens: 256)]
        }, GiB, 64 << 20, 4096, 14336, 32, 65536, null);
        var options = new AdaptiveModelMemoryOptions(2048, 512) { AllowWeightStreaming = false };
        var budget = new MemoryBudget([new(AdaptiveModelSession.HostPool, 0), new(AdaptiveModelSession.DevicePool, 0)]);
        var hw = new InferenceHardwareMemory(32 * GiB, 17 * GiB, 16 * GiB, 12 * GiB);
        Assert.True(AdaptiveModelSession.PlanLoad(profile, options, hw, budget).Accepted);
        // Resident device weights do not imply another 4 GiB host allocation.
        // The 4 GiB commit allowance minus 2 GiB default headroom still covers
        // 1 GiB fusion + 128 MiB loader + 16 MiB persistent host loading peak.
        var fits = AdaptiveModelSession.PlanLoad(profile, options,
            hw with { HostCommitLimit = 40 * GiB, HostCommitAvailable = 4 * GiB }, budget);
        Assert.True(fits.Accepted);
        Assert.Equal(GiB + (128L << 20) + (16L << 20),
            fits.PoolPeaks.Single(p => p.Pool == AdaptiveModelSession.HostPool).Loading);
        var limited = AdaptiveModelSession.PlanLoad(profile, options,
            hw with { HostCommitLimit = 40 * GiB, HostCommitAvailable = 3 * GiB }, budget);
        Assert.False(limited.Accepted);
        Assert.Contains(limited.Rejections, r => r.Pool == AdaptiveModelSession.HostPool);
    }

    [Fact]
    public void RefreshKeepsCommitSeparateWithoutSummingOrClaimingEnforcement()
    {
        const string pool = AdaptiveModelSession.HostPool;
        var now = DateTimeOffset.UtcNow;
        var hw = new InferenceHardwareMemory(32 * GiB, 17 * GiB, 16 * GiB, 12 * GiB)
        { HostCommitLimit = 40 * GiB, HostCommitAvailable = 4 * GiB };
        var observations = new List<PhysicalMemoryObservation>
        {
            new("ram", pool, "host-ram", PhysicalMemoryMetric.HostAvailable, hw.HostTotal, hw.HostAvailable, now)
            { MaximumBudgetBytes = 8 * GiB, Headroom = new() { MinimumBytes = GiB } }
        };
        AdaptiveModelSession.AddHostCommitObservation(observations, hw, now, 8 * GiB, GiB, 0);
        var commit = observations.Single(o => o.Constraint == "host-commit-available");
        Assert.Null(commit.IncludedCommittedBytes);
        Assert.Null(commit.Enforcement);
        Assert.Equal(MemoryHardLimitCoverage.None, commit.RequiredHardLimitCoverage);
        var plan = new PhysicalMemoryCapacityPlanner().Plan(observations, [new(pool, 8 * GiB, 0, 0)], now);
        Assert.True(plan.Accepted);
        Assert.Equal(3 * GiB, plan.DesiredCapacities.Single().Bytes);
        Assert.Equal(2, plan.Attribution.Count);
    }

    [Fact]
    public void CurrentPlatformProvidesValidOrExplicitlyUnknownCommit()
    {
        var sample = HostMemoryInfo.CaptureAllocationConstraints();
        Assert.True(sample.Total > 0);
        Assert.InRange(sample.Available, 0, sample.Total);
        if (OperatingSystem.IsWindows())
        {
            Assert.True(sample.CommitLimit > 0);
            Assert.InRange(sample.CommitAvailable!.Value, 0, sample.CommitLimit!.Value);
        }
        else
        {
            Assert.Null(sample.CommitLimit);
            Assert.Null(sample.CommitAvailable);
        }
    }
}
