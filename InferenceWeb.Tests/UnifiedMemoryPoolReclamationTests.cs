// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>The production session's real pool-selection/re-sampling loop with
/// deterministic physical advice and real ownership transactions. No native
/// backend, model checkpoint or GPU is loaded by these fixtures.</summary>
public sealed class UnifiedMemoryPoolReclamationTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const string Device = UnifiedMemoryModelSession.DevicePool;

    private static MemoryCharge[] Charges(long host, long device) => [new(Host, host), new(Device, device)];
    private static MemoryBudget Budget(long host = 100, long device = 100) => new(Charges(host, device));

    private static BudgetReservation Committed(MemoryBudget budget, long host, long device)
    {
        var owner = budget.Reserve(Charges(host, device)); owner.Commit(); return owner;
    }

    private static BudgetReservation ClosedRootChild(MemoryBudget budget, long device)
    {
        using var root = budget.Reserve(Charges(0, device));
        var child = root.TryTake(Charges(0, device))!; child.Commit(); return child;
    }

    [Theory]
    [InlineData(40, 10)] // Host-only pressure with Device headroom.
    [InlineData(40, 80)] // Host-only pressure with Device exactly at its limit.
    [InlineData(30, 80)] // Both pools exactly fit: no reclamation at all.
    public void HostShortageOrExactDeviceLimitDoesNotAuthorizeDeviceDemotion(long hostPeak, long devicePeak)
    {
        var budget = Budget();
        using var live = Committed(budget, 70, 0);
        using var retained = ClosedRootChild(budget, 20);
        var before = budget.Snapshot().ToArray();
        int demotions = 0, physicalSamples = 0, estimates = 0;
        var initial = new UnifiedMemoryCapacitySample(Charges(hostPeak, devicePeak), Charges(100, 100));
        var result = UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(budget, Device, initial,
            () => { physicalSamples++; return Charges(100, 100); },
            () => { estimates++; return initial.Needed; },
            _ => { demotions++; return false; });
        Assert.Equal(0, demotions);
        Assert.Equal(0, physicalSamples); Assert.Equal(0, estimates);
        Assert.Equal(initial, result);
        Assert.Equal(before, budget.Snapshot());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(70)] // Host remains unfit after Device has become ready.
    public void OneSufficientDeviceVictimStopsBeforeTouchingTheNextHolder(long liveHost)
    {
        var budget = Budget();
        using var live = Committed(budget, liveHost, 30);
        using var first = ClosedRootChild(budget, 20);
        using var second = ClosedRootChild(budget, 20);
        int demotions = 0, physicalSamples = 0, estimates = 0;
        var needed = Charges(40, 40); var limits = Charges(100, 100);
        var result = UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(budget, Device, new(needed, limits),
            () => { physicalSamples++; return limits; }, () => { estimates++; return needed; },
            _ =>
            {
                demotions++;
                if (demotions == 1) { first.Dispose(); return true; }
                return false;
            });
        Assert.Equal(1, demotions); Assert.Equal(1, physicalSamples); Assert.Equal(1, estimates);
        Assert.Equal(50, budget.Snapshot().Single(c => c.Pool == Device).Committed);
        Assert.Equal(20, second.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.False(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Device, result));
        Assert.Equal(liveHost == 70, UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Host, result));
    }

    [Fact]
    public void EveryVictimUsesFreshPhysicalLimitsAndReestimatesTheActualRequest()
    {
        var budget = Budget();
        using var live = Committed(budget, 0, 30);
        using var first = ClosedRootChild(budget, 20);
        using var second = ClosedRootChild(budget, 20);
        int physicalSamples = 0, estimates = 0, demotions = 0;
        var seenLimits = new List<long>(); var seenLedger = new List<long>();
        var order = new List<string>();
        var result = UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(budget, Device,
            new(Charges(0, 40), Charges(100, 100)),
            () => { physicalSamples++; order.Add("physical"); return Charges(100, 80); },
            () => { estimates++; order.Add("estimate"); return Charges(0, 50); },
            limits =>
            {
                order.Add("demote");
                seenLimits.Add(limits.Single(c => c.Pool == Device).Bytes);
                seenLedger.Add(budget.Snapshot().Single(c => c.Pool == Device).Committed);
                if (++demotions == 1) first.Dispose(); else second.Dispose();
                return true;
            });
        Assert.Equal(2, demotions); Assert.Equal(2, physicalSamples); Assert.Equal(2, estimates);
        Assert.Equal(new long[] { 100, 80 }, seenLimits);
        Assert.Equal(new long[] { 70, 50 }, seenLedger);
        Assert.Equal(new[] { "demote", "physical", "estimate", "demote", "physical", "estimate" }, order);
        Assert.Equal(50, result.Needed.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(80, result.Limits.Single(c => c.Pool == Device).Bytes);
        Assert.False(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Device, result));
    }

    [Fact]
    public void ARefusedDemotionRefreshesAdviceButDoesNotRefundAnyLiveOrPendingChild()
    {
        var budget = Budget();
        using var sibling = Committed(budget, 0, 30);
        using var retained = ClosedRootChild(budget, 20);
        using var liveRoot = budget.Reserve(Charges(0, 20));
        using var pending = liveRoot.TryTake(Charges(0, 20))!;
        var before = budget.Snapshot().ToArray();
        int samples = 0, estimates = 0, demotions = 0;
        var result = UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(budget, Device,
            new(Charges(0, 40), Charges(100, 100)),
            () => { samples++; return Charges(100, 90); },
            () => { estimates++; return Charges(0, 45); },
            _ => { demotions++; return false; });
        Assert.Equal(1, demotions); Assert.Equal(1, samples); Assert.Equal(1, estimates);
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(20, retained.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(20, pending.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(30, sibling.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(90, result.Limits.Single(c => c.Pool == Device).Bytes);
        Assert.True(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Device, result));
    }

    [Fact]
    public void FailedPhysicalReleasePropagatesWithoutAdvertisingCreditOrRetryingAnotherHolder()
    {
        var budget = Budget();
        using var retained = ClosedRootChild(budget, 70);
        var before = budget.Snapshot().ToArray();
        var failure = new IOException("physical release failed");
        int attempts = 0, samples = 0;
        var observed = Assert.Throws<IOException>(() => UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(
            budget, Device, new(Charges(0, 40), Charges(100, 100)),
            () => { samples++; return Charges(100, 100); }, () => Charges(0, 40),
            _ => { attempts++; throw failure; }));
        Assert.Same(failure, observed); Assert.Equal(1, attempts); Assert.Equal(0, samples);
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(70, retained.OwnedCharges.Single(c => c.Pool == Device).Bytes);
    }

    [Fact]
    public void ScratchChildRemainsChargedAfterItsMaintenanceRootCloses()
    {
        var budget = Budget();
        using var retained = ClosedRootChild(budget, 70);
        BudgetReservation? scratch = null;
        try
        {
            var result = UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(budget, Device,
                new(Charges(0, 40), Charges(100, 100)), () => Charges(100, 100), () => Charges(0, 40),
                limits =>
                {
                    using var maintenance = budget.TryReserve(Charges(10, 0), limits)!;
                    scratch = maintenance.TryTake(Charges(10, 0))!; scratch.Commit();
                    // Only the physical Device release returns this credit.
                    Assert.Equal(70, budget.Snapshot().Single(c => c.Pool == Device).Committed);
                    retained.Dispose(); return true;
                });
            Assert.False(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Device, result));
            Assert.Equal(10, budget.Snapshot().Single(c => c.Pool == Host).Committed);
            Assert.Equal(0, budget.Snapshot().Single(c => c.Pool == Host).Reserved);
            Assert.Equal(10, scratch!.OwnedCharges.Single(c => c.Pool == Host).Bytes);
        }
        finally { scratch?.Dispose(); }
        Assert.All(budget.Snapshot(), c => Assert.Equal(0, c.Reserved + c.Committed));
    }

    [Fact]
    public void AdvisoryReadinessDoesNotAuthorizeSpendingCreditAcquiredByAnotherOwner()
    {
        var budget = Budget();
        using var retained = ClosedRootChild(budget, 40);
        var before = budget.Snapshot().ToArray();
        var advice = UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(budget, Device,
            new(Charges(0, 50), Charges(100, 100)), () => throw new Exception("unneeded sample"),
            () => throw new Exception("unneeded estimate"), _ => throw new Exception("unneeded release"));
        Assert.Equal(before, budget.Snapshot());
        using var rival = Committed(budget, 0, 20);
        Assert.Null(budget.TryReserve(advice.Needed, advice.Limits));
        Assert.Equal(40, retained.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(60, budget.Snapshot().Single(c => c.Pool == Device).Committed);
    }

    [Fact]
    public void DeficitArithmeticDoesNotOverflowAtTheConfiguredLongCeiling()
    {
        var budget = Budget(device: long.MaxValue);
        using var retained = Committed(budget, 0, long.MaxValue - 1);
        var sample = new UnifiedMemoryCapacitySample(Charges(0, 2), Charges(100, long.MaxValue));
        Assert.True(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Device, sample));
        Assert.False(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Host, sample));
    }

    [Fact]
    public void RecordedRoundedV27FinalHostRefusalDoesNotAuthorizeDeviceDemotion()
    {
        // Exact ledger counts and rounded MiB charge/constraint messages from
        // v27's final CompletePrefixMaterialization refusal after successful
        // vision encoding. This is a policy regression, not a
        // replay of model execution or evidence about its earlier pressure.
        const long mib = 1L << 20;
        var budget = Budget(6156 * mib, 13332 * mib);
        using var physical = Committed(budget, 2392199168, 7985351564);
        using var pending = budget.Reserve(Charges(3283307290, 4266437200));
        var sample = new UnifiedMemoryCapacitySample(Charges(1008 * mib, 1520 * mib), Charges(6156 * mib, 13332 * mib));
        Assert.True(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Host, sample));
        Assert.False(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Device, sample));
        var before = budget.Snapshot().ToArray();
        int demotions = 0;
        var result = UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(budget, Device, sample,
            () => throw new Exception("unneeded physical sample"),
            () => throw new Exception("unneeded estimate"), _ => { demotions++; return false; });
        Assert.Equal(0, demotions);
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(sample, result);
        // The classification is unchanged even with conservative 1 MiB bounds
        // around the logged rounded fields. No actual peak is relaxed.
        Assert.True(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Host,
            new(Charges(1007 * mib, 1521 * mib), Charges(6157 * mib, 13331 * mib))));
        Assert.False(UnifiedMemoryReclamationPolicy.HasPoolDeficit(budget, Device,
            new(Charges(1007 * mib, 1521 * mib), Charges(6157 * mib, 13331 * mib))));
    }
}
