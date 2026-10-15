// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Globalization;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Real ledger and injected hardware observations; no CUDA/model load.</summary>
public sealed class GlmUnifiedAdmissionDiagnosticsTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const string Device = UnifiedMemoryModelSession.DevicePool;

    [Fact]
    public void CommitConstrainedHostRefusalReportsTheFailingPoolAndDoesNotAddReservedCreditBack()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var root = budget.Reserve([new(Host, 300), new(Device, 100)]);
        using var payload = root.TryTake([new(Host, 200), new(Device, 80)])!;
        payload.Commit();
        var before = budget.Snapshot();
        var hardware = new InferenceHardwareMemory(2000, 1000, 1600, 800)
        { HostCommitLimit = 1500, HostCommitAvailable = 180 };
        var sample = GlmUnifiedPhysicalObservation.Create(hardware, before, 1000, 1000, 50, 100);
        Assert.Equal(330, sample.Limits.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(780, sample.Limits.Single(c => c.Pool == Device).Bytes);
        MemoryCharge[] growth = [new(Host, 31), new(Device, 0)];
        Assert.False(sample.Fits(growth));
        string text = sample.Describe(growth);
        Assert.Contains($"{Host}: reserved=100, committed=200, capacity=1000, physical-limit=330, growth=31, combined-post=331, shortfall=1", text);
        Assert.Contains($"{Device}: reserved=20, committed=80, capacity=1000, physical-limit=780, growth=0, combined-post=100, shortfall=0", text);
        Assert.Contains("host-available=1000, host-commit-limit=1500, host-commit-available=180, host-allocation-available=180", text);
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void ExistingJointOwnershipIsReplacedOnceAndCommittedChildrenAreNeverCreditedAsFreed()
    {
        var budget = new MemoryBudget([new(Host, 2000), new(Device, 2000)]);
        using var joint = budget.Reserve([new(Host, 600), new(Device, 900)]);
        using var liveGraph = joint.TryTake([new(Host, 200), new(Device, 500)])!;
        liveGraph.Commit();
        var before = budget.Snapshot();
        var sample = GlmUnifiedPhysicalObservation.Create(new(3000, 500, 3000, 500), before, 2000, 2000, 0, 0);
        // New joint target (400/700) plus request (100/200), less the
        // already-owned 600/900. Live graph ownership is part of that root.
        MemoryCharge[] growth = [new(Host, 400 - 600 + 100), new(Device, 700 - 900 + 200)];
        Assert.True(sample.Fits(growth));
        string text = sample.Describe(growth);
        Assert.Contains("growth=-100, combined-post=500, shortfall=0", text);
        Assert.Contains("growth=0, combined-post=900, shortfall=0", text);
        Assert.Equal(500, joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void DeviceDeficitAndFreshHardwareChangeRemainDistinctFromHostDeficit()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var live = budget.Reserve([new(Host, 100), new(Device, 500)]);
        live.Commit();
        var ledger = budget.Snapshot();
        var initial = GlmUnifiedPhysicalObservation.Create(new(2000, 1000, 1600, 200), ledger, 1000, 1000, 50, 100);
        MemoryCharge[] growth = [new(Host, 20), new(Device, 101)];
        Assert.False(initial.Fits(growth));
        Assert.Contains("physical-limit=600, growth=101, combined-post=601, shortfall=1", initial.Describe(growth));
        var fresh = GlmUnifiedPhysicalObservation.Create(new(2000, 1000, 1600, 201), ledger, 1000, 1000, 50, 100);
        Assert.True(fresh.Fits(growth));
        // Retaining an observation for diagnostics must not resample or
        // rewrite the original refusal after the hardware allowance changes.
        Assert.Contains("device-available=200", initial.Describe(growth));
        Assert.Contains("shortfall=1", initial.Describe(growth));
        Assert.Equal(ledger.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void HeadroomExhaustionNeverSubtractsExistingCommitmentOrGrantsReservedCredit()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var owner = budget.Reserve([new(Host, 200), new(Device, 300)]);
        using var live = owner.TryTake([new(Host, 80), new(Device, 90)])!;
        live.Commit();
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 10, 1600, 20), budget.Snapshot(), 1000, 1000, 50, 100);
        Assert.Equal(80, sample.Limits.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(90, sample.Limits.Single(c => c.Pool == Device).Bytes);
        Assert.False(sample.Fits([new(Host, 0), new(Device, 0)]));
        Assert.Contains("growth=0, combined-post=200, shortfall=120", sample.Describe([new(Host, 0)]));
    }

    [Fact]
    public void UnknownCommitIsExplicitAndConfiguredLedgerCeilingStillLimitsAdmission()
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 50, 30, 0), new(Device, 70, 0, 0) };
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 1000, 1600, 800), ledger, 1000, 1000, 0, 0);
        Assert.False(sample.Fits([new(Host, 21)]));
        Assert.Contains("host-commit-limit=unknown, host-commit-available=unknown", sample.Describe([new(Host, 21)]));
        Assert.Contains("physical-limit=1000, growth=21, combined-post=51, shortfall=1", sample.Describe([new(Host, 21)]));
    }

    [Fact]
    public void DiagnosticArithmeticDoesNotOverflowOrDependOnProcessCulture()
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
            var ledger = new MemoryPoolSnapshot[] { new(Host, long.MaxValue, long.MaxValue, 0), new(Device, 1, 0, 0) };
            var sample = GlmUnifiedPhysicalObservation.Create(new(long.MaxValue, long.MaxValue, 1, 1), ledger,
                long.MaxValue, 1, 0, 0);
            Assert.False(sample.Fits([new(Host, long.MaxValue)]));
            string text = sample.Describe([new(Host, long.MaxValue)]);
            Assert.Contains("combined-post=18446744073709551614", text);
            Assert.Contains("shortfall=9223372036854775807", text);
        }
        finally { CultureInfo.CurrentCulture = original; }
    }

    [Fact]
    public void SuccessfulMappedHintWithoutFreshAllowanceStillRefusesAndDoesNotRefundLiveChildren()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var root = budget.Reserve([new(Host, 200), new(Device, 200)]);
        using var state = root.TryTake([new(Host, 100), new(Device, 100)])!;
        state.Commit();
        var before = budget.Snapshot();
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 50, 1600, 800), before, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        int hints = 0, samples = 0;
        Assert.True(sample.TryRefreshMappedHostPressure(growth, () => { hints++; return true; },
            () => { samples++; return sample; }, out var refreshed, out var hinted));
        Assert.True(hinted);
        Assert.Equal(1, hints); Assert.Equal(1, samples);
        Assert.False(refreshed.Fits(growth));
        Assert.Equal(75, refreshed.Shortfall(Host, growth));
        Assert.Equal(100, root.LiveChildCharges.Single(c => c.Pool == Host).Bytes);
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void MappedHostRetryUsesOneFreshObservationAndExactLimitWithoutAnyBudgetIncrease()
    {
        var budget = new MemoryBudget([new(Host, 1000), new(Device, 1000)]);
        using var root = budget.Reserve([new(Host, 200), new(Device, 200)]);
        using var state = root.TryTake([new(Host, 100), new(Device, 100)])!;
        state.Commit();
        var before = budget.Snapshot();
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 50, 1600, 800), before, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        int hints = 0, samples = 0;
        Assert.True(sample.TryRefreshMappedHostPressure(growth, () => { hints++; return true; },
            () => { samples++; return GlmUnifiedPhysicalObservation.Create(new(2000, 125, 1600, 800), budget.Snapshot(), 1000, 1000, 0, 0); },
            out var refreshed, out var hinted));
        Assert.True(hinted); Assert.Equal(1, hints); Assert.Equal(1, samples);
        Assert.True(refreshed.Fits(growth)); // committed 100 + available 125 == owned 200 + requested 25.
        Assert.Equal(225, refreshed.Limits.Single(c => c.Pool == Host).Bytes);
        Assert.False(sample.Fits(growth));
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(0, 0)]
    public void DeviceOrCombinedDeficitAndAlreadyFittingRequestNeverHintsMappings(int hostDeficit, int deviceDeficit)
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 1000, 100, 100), new(Device, 1000, 100, 100) };
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 100 - hostDeficit, 1600, 100 - deviceDeficit),
            ledger, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 0), new(Device, 0)];
        int hints = 0, samples = 0;
        Assert.False(sample.TryRefreshMappedHostPressure(growth, () => { hints++; return true; },
            () => { samples++; return sample; }, out var refreshed, out var hinted));
        Assert.Null(hinted); Assert.Equal(0, hints); Assert.Equal(0, samples);
        Assert.Equal(sample, refreshed);
    }

    [Fact]
    public void FalseMappedHintCannotAuthorizeGrowthButAnIndependentlyFreshHardwareAllowanceCan()
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 1000, 100, 100), new(Device, 1000, 100, 100) };
        var sample = GlmUnifiedPhysicalObservation.Create(new(2000, 50, 1600, 800), ledger, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        Assert.True(sample.TryRefreshMappedHostPressure(growth, () => false,
            () => GlmUnifiedPhysicalObservation.Create(new(2000, 125, 1600, 800), ledger, 1000, 1000, 0, 0),
            out var refreshed, out var hinted));
        Assert.False(hinted);
        Assert.True(refreshed.Fits(growth));
        Assert.Equal(ledger, refreshed.Ledger);
    }

    [Fact]
    public void FreeingMappedPhysicalRamCannotConcealAnUnchangedHostCommitShortfall()
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 1000, 100, 100), new(Device, 1000, 100, 100) };
        var hardware = new InferenceHardwareMemory(2000, 50, 1600, 800)
        { HostCommitLimit = 3000, HostCommitAvailable = 50 };
        var sample = GlmUnifiedPhysicalObservation.Create(hardware, ledger, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(Host, 25), new(Device, 25)];
        Assert.True(sample.TryRefreshMappedHostPressure(growth, () => true,
            () => GlmUnifiedPhysicalObservation.Create(hardware with { HostAvailable = 1000 }, ledger, 1000, 1000, 0, 0),
            out var refreshed, out _));
        Assert.False(refreshed.Fits(growth));
        Assert.Equal(75, refreshed.Shortfall(Host, growth));
        Assert.Contains("host-available=1000, host-commit-limit=3000, host-commit-available=50, host-allocation-available=50",
            refreshed.Describe(growth));
    }

    [Theory]
    [InlineData(-1, 100)]
    [InlineData(101, 100)]
    [InlineData(0, 0)]
    public void InvalidDeviceObservationCannotConcealARefusal(long available, long total)
    {
        var ledger = new MemoryPoolSnapshot[] { new(Host, 100, 0, 0), new(Device, 100, 0, 0) };
        Assert.Throws<ArgumentException>(() => GlmUnifiedPhysicalObservation.Create(new(200, 100, total, available),
            ledger, 100, 100, 0, 0));
    }
}
