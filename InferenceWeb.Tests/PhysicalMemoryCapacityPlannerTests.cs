// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Hardware;
using TensorSharp.Memory.Planning;

namespace InferenceWeb.Tests;

public sealed class PhysicalMemoryCapacityPlannerTests
{
    private static readonly DateTimeOffset Now = new(2026, 10, 10, 0, 0, 0, TimeSpan.Zero);
    private static PhysicalMemoryCapacityPlanner Planner(long minimum = 10, long burst = 5)
        => new(new() { MinimumBytes = minimum, FutureBurstBytes = burst });
    private static PhysicalMemoryObservation Observation(long free, long? included = null)
        => new("host", "ram", "system-ram", PhysicalMemoryMetric.HostAvailable, 1000, free, Now)
        { IncludedCommittedBytes = included };
    private static readonly MemoryPoolSnapshot[] Ledger = [new("ram", 1000, 20, 200)];

    [Fact]
    public void PageableCommittedAndDiagnosticGaugesDoNotInventResidentCredit()
    {
        var plan = Planner().Plan([Observation(600)], Ledger, Now);
        Assert.True(plan.Accepted);
        Assert.Equal(590, Assert.Single(plan.DesiredCapacities).Bytes);
        var attribution = Assert.Single(plan.Attribution);
        Assert.Equal(400, attribution.UnattributedUnavailableUpperBound);
        Assert.Equal(200, attribution.UnprovenCommittedBytes);
        Assert.Equal(0, attribution.ProvenIncludedCommittedBytes);
        Assert.Equal(370, InferenceMemoryPlanner.GetCapacity(Assert.Single(plan.Pools)).AdditionalAvailable);
    }

    [Fact]
    public void OnlyCertifiedSubsetOfSharedLedgerIsAddedBack()
    {
        var plan = Planner().Plan([Observation(600, 80)], Ledger, Now);
        Assert.Equal(670, Assert.Single(plan.DesiredCapacities).Bytes);
        var attribution = Assert.Single(plan.Attribution);
        Assert.Equal(320, attribution.UnattributedUnavailableUpperBound);
        Assert.Equal(120, attribution.UnprovenCommittedBytes);
        Assert.Equal(450, InferenceMemoryPlanner.GetCapacity(Assert.Single(plan.Pools)).AdditionalAvailable);
        // All current unattributed use is already absent from free; it is not
        // subtracted again as headroom. Only the configured future floor is.
        Assert.Equal(10, attribution.HeadroomBytes);
    }

    [Fact]
    public void ObservedGrowthAndPeakReboundReserveFutureSpaceWithoutSummingGauges()
    {
        var planner = Planner();
        planner.Plan([Observation(600, 100)], Ledger, Now); // unattributed 300
        var pressure = planner.Plan([Observation(500, 100) with { ObservedAt = Now.AddSeconds(1) }], Ledger, Now.AddSeconds(1));
        Assert.Equal(100, Assert.Single(pressure.Attribution).RecentGrowthBytes);
        Assert.Equal(105, Assert.Single(pressure.Attribution).HeadroomBytes);
        Assert.Equal(495, Assert.Single(pressure.DesiredCapacities).Bytes);
        var recovered = planner.Plan([Observation(650, 100) with { ObservedAt = Now.AddSeconds(2) }], Ledger, Now.AddSeconds(2));
        Assert.Equal(150, Assert.Single(recovered.Attribution).PeakReboundBytes);
        Assert.Equal(155, Assert.Single(recovered.Attribution).HeadroomBytes);
        Assert.Equal(595, Assert.Single(recovered.DesiredCapacities).Bytes);
    }

    [Fact]
    public void InstrumentedGrowthDoesNotBecomeUninstrumentedGrowthAgain()
    {
        var planner = Planner();
        planner.Plan([Observation(600, 100)], Ledger, Now);
        var grown = planner.Plan([Observation(500, 200) with { ObservedAt = Now.AddSeconds(1) }], Ledger, Now.AddSeconds(1));
        Assert.Equal(0, Assert.Single(grown.Attribution).RecentGrowthBytes);
        Assert.Equal(690, Assert.Single(grown.DesiredCapacities).Bytes);
    }

    [Fact]
    public void SharedUmaConstraintsAreNotAddedAndTightestConstraintWinsWithinOnePool()
    {
        var observations = new[]
        {
            Observation(600, 100),
            Observation(300, 20) with { Constraint = "job-commit", Metric = PhysicalMemoryMetric.PrivateCommitAvailable },
            Observation(400, 100) with { Constraint = "uma-device", Pool = "gpu", Metric = PhysicalMemoryMetric.AllocatorAvailable }
        };
        var plan = Planner().Plan(observations, [Ledger[0], new("gpu", 1000, 0, 100)], Now);
        Assert.True(plan.Accepted);
        Assert.Equal(2, plan.Pools.Count);
        Assert.Equal(310, plan.DesiredCapacities.Single(c => c.Pool == "ram").Bytes);
        Assert.Equal(490, plan.DesiredCapacities.Single(c => c.Pool == "gpu").Bytes);
        Assert.All(plan.Attribution, p => Assert.Equal("system-ram", p.Domain));
    }

    [Theory]
    [InlineData(MemoryHardLimitCoverage.PrivateCommit, MemoryHardLimitCoverage.HostPhysical, false)]
    [InlineData(MemoryHardLimitCoverage.HostPhysical, MemoryHardLimitCoverage.AllPhysical, false)]
    [InlineData(MemoryHardLimitCoverage.None, MemoryHardLimitCoverage.HostPhysical, false)]
    [InlineData(MemoryHardLimitCoverage.HostPhysical, MemoryHardLimitCoverage.HostPhysical, true)]
    public void HardLimitRequirementsCannotPromotePartialOrAdvisoryPlatformCoverage(
        MemoryHardLimitCoverage actual, MemoryHardLimitCoverage required, bool accepted)
    {
        var observation = Observation(600) with
        {
            RequiredHardLimitCoverage = required,
            Enforcement = new("test verified OS mechanism", actual, 1000, true)
        };
        var plan = Planner().Plan([observation], Ledger, Now);
        Assert.Equal(accepted, plan.Accepted);
        if (!accepted) { Assert.Empty(plan.Pools); Assert.Empty(plan.DesiredCapacities); Assert.Single(plan.Refusals); }
        Assert.False(Planner().Plan([observation with { Enforcement = observation.Enforcement with { Active = false } }], Ledger, Now).Accepted);
    }

    [Fact]
    public void SmallerEnforcedLimitMustBeReflectedInTheObservedConstraint()
    {
        var observation = Observation(600) with
        {
            RequiredHardLimitCoverage = MemoryHardLimitCoverage.PrivateCommit,
            Enforcement = new("Job", MemoryHardLimitCoverage.PrivateCommit, 500, true)
        };
        Assert.False(Planner().Plan([observation], Ledger, Now).Accepted);
    }

    [Fact]
    public void StaleMissingInconsistentOrReorderedSamplesCannotRefreshCapacities()
    {
        var planner = Planner();
        Assert.False(planner.Plan([], Ledger, Now).Accepted);
        Assert.False(planner.Plan([Observation(600) with { ObservedAt = Now.AddSeconds(-11) }], Ledger, Now).Accepted);
        Assert.False(planner.Plan([Observation(600) with { ObservedAt = Now.AddSeconds(1) }], Ledger, Now).Accepted);
        Assert.False(planner.Plan([Observation(990, 20)], Ledger, Now).Accepted);
        Assert.False(planner.Plan([Observation(600, 201)], Ledger, Now).Accepted);
        Assert.False(planner.Plan([Observation(0) with { TotalBytes = 0 }], Ledger, Now).Accepted);
        Assert.True(planner.Plan([Observation(600)], Ledger, Now).Accepted);
        Assert.False(planner.Plan([Observation(900) with { ObservedAt = Now.AddSeconds(-1) }], Ledger, Now).Accepted);
    }

    [Fact]
    public void ChangedMetricOrExpiredHistoryDoesNotCreateFalsePeakHeadroom()
    {
        var planner = Planner();
        planner.Plan([Observation(100)], Ledger, Now);
        var changed = planner.Plan([Observation(900) with { Metric = PhysicalMemoryMetric.PrivateCommitAvailable,
            ObservedAt = Now.AddSeconds(1) }], Ledger, Now.AddSeconds(1));
        Assert.Equal(10, Assert.Single(changed.Attribution).HeadroomBytes);
        var expired = planner.Plan([Observation(900) with { ObservedAt = Now.AddMinutes(6) }], Ledger, Now.AddMinutes(6));
        Assert.Equal(10, Assert.Single(expired.Attribution).HeadroomBytes);
    }

    [Fact]
    public void HeadroomArithmeticSaturatesInsteadOfOverflowingAndExplicitCeilingStillApplies()
    {
        var planner = Planner(long.MaxValue, long.MaxValue);
        var plan = planner.Plan([Observation(600) with { MaximumBudgetBytes = 25 }], Ledger, Now);
        Assert.True(plan.Accepted);
        Assert.Equal(0, Assert.Single(plan.DesiredCapacities).Bytes);
        var capacity = InferenceMemoryPlanner.GetCapacity(Assert.Single(plan.Pools));
        Assert.Equal(220, capacity.ProtectedCapacity);
        Assert.Equal(220, capacity.OverTargetBytes);
    }

    [Fact]
    public void LegacyPlannerAlsoDefaultsToZeroOverlapAndRejectsInventedCredit()
    {
        var pool = new InferenceMemoryPool("ram", 1000, 600, 10, Ledger[0]);
        Assert.Equal(590, InferenceMemoryPlanner.GetCapacity(pool).DesiredCapacity);
        Assert.Equal(670, InferenceMemoryPlanner.GetCapacity(pool with { IncludedCommittedBytes = 80 }).DesiredCapacity);
        Assert.Throws<ArgumentOutOfRangeException>(() => InferenceMemoryPlanner.GetCapacity(pool with { IncludedCommittedBytes = 201 }));
    }
}
