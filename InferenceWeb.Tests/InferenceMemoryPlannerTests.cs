// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;

namespace InferenceWeb.Tests;

public sealed class InferenceMemoryPlannerTests
{
    [Fact]
    public void FittingWeightsAloneDoesNotAdmitKvAndGraphPeak()
    {
        var input = Input(1000, 100) with
        {
            Model = new()
            {
                DenseWeights = new(90, 90),
                KvCaches = [new(BytesPerToken: 2)]
            },
            Candidates = [Candidate("resident"), Candidate("host", InferenceWeightPlacement.HostCached) with
                { PrefillWorkspace = new(Device: 5) }]
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.Equal("host", plan.SelectedCandidate!.Name);
        Assert.Equal(25, plan.PoolPeaks.Single(p => p.Pool == "gpu").Peak);
        var rejected = Assert.Single(plan.Rejections);
        Assert.Equal("resident", rejected.Candidate);
        Assert.Equal(110, rejected.RequiredBytes);
        Assert.Equal(100, rejected.AvailableBytes);
    }

    [Fact]
    public void SmallerQualifiedPrefillChunkKeepsResidentGraphBeforeAnyOffload()
    {
        var input = Input(1000, 100) with
        {
            Model = new() { DenseWeights = new(60, 60), Persistent = new(Device: 10) },
            Workload = new(2048, 2048, 1, 1),
            Candidates =
            [
                Candidate("host2048", InferenceWeightPlacement.HostCached) with { PrefillChunkTokens = 2048 },
                Candidate("resident2048") with { PrefillChunkTokens = 2048, PrefillWorkspace = new(Device: 40) },
                Candidate("resident256") with { PrefillChunkTokens = 256, PrefillWorkspace = new(Device: 20) }
            ]
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.True(plan.Accepted);
        Assert.True(plan.SelectedCandidate!.PreservesExecutionGraph);
        Assert.Equal("resident256", plan.SelectedCandidate.Name);
        Assert.Equal(256, plan.SelectedChunkTokens);
        Assert.Equal("resident2048", Assert.Single(plan.Rejections).Candidate);
    }

    [Fact]
    public void QuantizationAndModalityRefusalCannotSilentlySelectStreaming()
    {
        var input = Input(10, 10) with
        {
            Candidates =
            [
                Candidate("resident"), Candidate("host", InferenceWeightPlacement.HostCached),
                Candidate("ssd", InferenceWeightPlacement.SsdStreaming) with
                    { CapabilityRefusal = "IQ2/multimodal adapter has no compatible streamed projection." }
            ]
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.False(plan.Accepted);
        Assert.Empty(plan.AdditionalCharges);
        Assert.Contains(plan.Rejections, r => r.Candidate == "ssd"
            && r.Kind == InferenceMemoryRejectionKind.Unsupported && r.Reason.Contains("IQ2"));
    }

    [Fact]
    public void KvRoundsEachPhysicalLayerWindowAndEveryConcurrentSequence()
    {
        var input = Input(10000, 10000) with
        {
            Model = new()
            {
                KvCaches = [new(2, LayerCount: 3, AllocationBlockTokens: 8),
                    new(4, WindowTokens: 5, AllocationBlockTokens: 4, Tier: MemoryTier.Host)],
                RecurrentStatePerSequence = new(Host: 7, Device: 11)
            },
            Workload = new(10, 10, 2, 3)
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.True(plan.Accepted);
        Assert.Equal(16 * 2 * 3 * 3 + 11 * 3, plan.PoolPeaks.Single(p => p.Pool == "gpu").Peak);
        Assert.Equal(8 * 4 * 3 + 7 * 3, plan.PoolPeaks.Single(p => p.Pool == "ram").Peak);
    }

    [Fact]
    public void ResidentExpertsIncludeEveryExpertNotOnlyTheActiveRoutes()
    {
        var input = Input(2000, 500) with
        {
            Model = new()
            {
                DenseWeights = new(100, 100), ExpertWeights = new(800, 800),
                ExpertCount = 8, ActiveExpertsPerToken = 1
            },
            Candidates = [Candidate("resident"), Candidate("host", InferenceWeightPlacement.HostCached) with
                { DeviceWeightCacheBytes = 150, PrefillWorkspace = new(Device: 50) }]
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.Equal("host", plan.SelectedCandidate!.Name);
        Assert.Equal(900, Assert.Single(plan.Rejections).RequiredBytes);
        Assert.Equal(900, plan.PoolPeaks.Single(p => p.Pool == "ram").Peak);
        Assert.Equal(200, plan.PoolPeaks.Single(p => p.Pool == "gpu").Peak);
    }

    [Fact]
    public void SsdStreamingChargesPartialRamCacheAndBothTransferBuffers()
    {
        var input = Input(40, 60) with
        {
            Pools = [Pool("ram", 40), Pool("gpu", 60), Pool("ssd", 100)], SsdPools = ["ssd"],
            Candidates = [Candidate("resident"), Candidate("host", InferenceWeightPlacement.HostCached),
                Candidate("ssd", InferenceWeightPlacement.SsdStreaming) with
                {
                    HostWeightCacheBytes = 10, DeviceWeightCacheBytes = 20,
                    TransferBuffer = new(Host: 8, Device: 12), TransferBufferCount = 2,
                    PrefillWorkspace = new(Device: 15), Persistent = new(Ssd: 7)
                }]
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.Equal("ssd", plan.SelectedCandidate!.Name);
        Assert.Equal(26, plan.PoolPeaks.Single(p => p.Pool == "ram").Peak);
        Assert.Equal(59, plan.PoolPeaks.Single(p => p.Pool == "gpu").Peak);
        Assert.Equal(7, plan.PoolPeaks.Single(p => p.Pool == "ssd").Peak);
        Assert.Equal(new InferenceMemoryBytes(16, 24),
            plan.Components.Single(c => c.Name == "transfer-buffers").Bytes);
        // The existing 100-byte source file does not require another 100 bytes free.
    }

    [Fact]
    public void UmaAddsHostCopiesToPhysicalRamButDoesNotSumDistinctConstraintCapacities()
    {
        var candidate = Candidate("resident") with { KeepResidentHostWeights = true };
        var input = Input(100, 100) with
        {
            Model = new() { DenseWeights = new(60, 60) },
            DevicePools = ["ram", "gpu"], Candidates = [candidate]
        };
        var rejected = InferenceMemoryPlanner.Plan(input);
        Assert.False(rejected.Accepted);
        var pressure = Assert.Single(rejected.Rejections);
        Assert.Equal("ram", pressure.Pool);
        Assert.Equal(120, pressure.RequiredBytes);
        var accepted = InferenceMemoryPlanner.Plan(input with
        { Candidates = [candidate with { KeepResidentHostWeights = false }] });
        Assert.True(accepted.Accepted);
        Assert.All(accepted.PoolPeaks, p => Assert.Equal(60, p.Peak));
    }

    [Fact]
    public void HardwareFreeAndReservationsAreCountedExactlyOnce()
    {
        var budget = new MemoryBudget([new MemoryCharge("ram", 100), new MemoryCharge("gpu", 100)]);
        using var allocated = budget.Reserve([new MemoryCharge("gpu", 40)]);
        allocated.Commit();
        using var pending = budget.Reserve([new MemoryCharge("gpu", 20)]);
        var snapshot = budget.Snapshot().Single(p => p.Pool == "gpu");
        Assert.Equal(40, snapshot.Available); // not the 60 bytes physically free
        var input = Input(100, 100) with
        {
            Pools = [Pool("ram", 100), new("gpu", 100, 60, 10, snapshot)],
            Model = new() { DenseWeights = new(30, 30) }
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.True(plan.Accepted);
        var capacity = plan.Capacities.Single(p => p.Pool == "gpu");
        Assert.Equal(90, capacity.DesiredCapacity);
        Assert.Equal(30, capacity.AdditionalAvailable);
        Assert.Equal(snapshot, budget.Snapshot().Single(p => p.Pool == "gpu"));
        Assert.True(budget.TrySetCapacity("gpu", capacity.ProtectedCapacity));
        using var admitted = budget.Reserve(plan.AdditionalCharges);
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "gpu").Available);
    }

    [Fact]
    public void ShrinkBelowInFlightOwnersPausesAdmissionWithoutRevokingCredit()
    {
        var budget = new MemoryBudget([new MemoryCharge("ram", 1000), new MemoryCharge("gpu", 200)]);
        using var allocated = budget.Reserve([new MemoryCharge("gpu", 120)]);
        allocated.Commit();
        using var pending = budget.Reserve([new MemoryCharge("gpu", 30)]);
        var input = Input(1000, 200) with
        {
            Pools = [Pool("ram", 1000), new("gpu", 200, 10, 20, budget.Snapshot().Single(p => p.Pool == "gpu"))],
            Candidates = [Candidate("reuse") with { ReusableCommittedCharges = [new("gpu", 100)] }]
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.False(plan.Accepted);
        var capacity = plan.Capacities.Single(p => p.Pool == "gpu");
        Assert.Equal(110, capacity.DesiredCapacity);
        Assert.Equal(150, capacity.ProtectedCapacity);
        Assert.Equal(40, capacity.OverTargetBytes);
        Assert.Equal(InferenceMemoryRejectionKind.ProtectedOwners, Assert.Single(plan.Rejections).Kind);
        Assert.False(budget.TrySetCapacity("gpu", capacity.DesiredCapacity));
        Assert.Equal(120, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        pending.Dispose();
        allocated.Dispose(); // only physical-owner release unlocks the smaller limit
        var recovered = InferenceMemoryPlanner.Plan(input with
        {
            Pools = [Pool("ram", 1000), new("gpu", 200, 150, 20, budget.Snapshot().Single(p => p.Pool == "gpu"))],
            Candidates = [Candidate("new")]
        });
        Assert.True(recovered.Accepted);
    }

    [Fact]
    public void ReuseSubtractsOnlyIdentifiedCommittedModelStorage()
    {
        var input = Input(1000, 200) with
        { Pools = [Pool("ram", 1000), new("gpu", 200, 40, 0, new("gpu", 200, 0, 80))] };
        Assert.False(InferenceMemoryPlanner.Plan(input).Accepted);
        var plan = InferenceMemoryPlanner.Plan(input with
        { Candidates = [Candidate("reuse") with { ReusableCommittedCharges = [new("gpu", 80)] }] });
        Assert.True(plan.Accepted);
        var peak = plan.PoolPeaks.Single(p => p.Pool == "gpu");
        Assert.Equal(80, peak.ReusedCommitted);
        Assert.Equal(20, peak.Additional);
        Assert.Throws<ArgumentException>(() => InferenceMemoryPlanner.Plan(input with
        { Candidates = [Candidate("invent-credit") with { ReusableCommittedCharges = [new("gpu", 81)] }] }));
    }

    [Fact]
    public void RecoveredHardwareMayGrowBudgetButHonorsExplicitOperatorLimit()
    {
        var input = Input(1000, 200) with
        { Pools = [Pool("ram", 1000), new("gpu", 200, 150, 20, new("gpu", 50, 0, 10), MaximumBudgetBytes: 120)] };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.True(plan.Accepted);
        var capacity = plan.Capacities.Single(p => p.Pool == "gpu");
        Assert.Equal(120, capacity.DesiredCapacity);
        Assert.Equal(110, capacity.AdditionalAvailable);
    }

    [Fact]
    public void LoadingCopiesAndDecodePeakCanRejectAnOtherwiseSmallPrefill()
    {
        var input = Input(100, 100) with
        {
            Model = new() { DenseWeights = new(50, 50) },
            Candidates =
            [
                Candidate("load-copy") with { LoadingWorkspace = new(Device: 51) },
                Candidate("decode-peak") with { DecodeWorkspace = new(Device: 51) },
                Candidate("phases") with
                    { LoadingWorkspace = new(Device: 40), PrefillWorkspace = new(Device: 30), DecodeWorkspace = new(Device: 20) }
            ]
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.Equal("phases", plan.SelectedCandidate!.Name);
        Assert.Equal(2, plan.Rejections.Count);
        var peak = plan.PoolPeaks.Single(p => p.Pool == "gpu");
        Assert.Equal((90L, 80L, 70L, 90L), (peak.Loading, peak.Prefill, peak.Decode, peak.Peak));
    }

    [Fact]
    public void OverflowIsARejectionRatherThanSmallWrappedPeak()
    {
        var input = Input(long.MaxValue, long.MaxValue) with
        {
            Model = new()
            {
                DenseWeights = new(1, 1),
                KvCaches = [new(long.MaxValue, LayerCount: 2)]
            }
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.False(plan.Accepted);
        Assert.Equal(InferenceMemoryRejectionKind.SizeOverflow, Assert.Single(plan.Rejections).Kind);
    }

    [Fact]
    public void PhysicalCapacityArithmeticCannotOverflowAtInt64Limit()
    {
        var input = Input(1000, long.MaxValue) with
        {
            Pools = [Pool("ram", 1000), new("gpu", long.MaxValue, long.MaxValue, 1,
                new("gpu", long.MaxValue, 0, 100))]
        };
        var plan = InferenceMemoryPlanner.Plan(input);
        Assert.True(plan.Accepted);
        Assert.Equal(long.MaxValue - 1, plan.Capacities.Single(p => p.Pool == "gpu").DesiredCapacity);
    }

    [Fact]
    public void MissingSpillMappingCannotEraseAnSsdAllocation()
    {
        var input = Input(1000, 1000) with
        { Candidates = [Candidate("spill") with { Persistent = new(Ssd: 1) }] };
        Assert.Throws<ArgumentException>(() => InferenceMemoryPlanner.Plan(input));
    }

    [Fact]
    public void CompetingOwnerAfterPlanningStillRequiresAtomicRealReservation()
    {
        var budget = new MemoryBudget([new MemoryCharge("ram", 100), new MemoryCharge("gpu", 100)]);
        var plan = InferenceMemoryPlanner.Plan(Input(100, 100));
        Assert.True(plan.Accepted);
        using var other = budget.Reserve([new MemoryCharge("gpu", 1)]);
        Assert.Null(budget.TryReserve(plan.AdditionalCharges));
        Assert.Equal(1, budget.Snapshot().Single(p => p.Pool == "gpu").Reserved);
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "ram").Reserved);
    }

    [Fact]
    public void InvalidGeometryAndDuplicateConstraintsAreRejected()
    {
        var input = Input(1000, 1000);
        Assert.Throws<ArgumentOutOfRangeException>(() => InferenceMemoryPlanner.Plan(input with
        { Workload = new(10, 11, 1, 1) }));
        Assert.Throws<ArgumentOutOfRangeException>(() => InferenceMemoryPlanner.Plan(input with
        { Model = new() { ExpertWeights = new(1, 1), ExpertCount = 0 } }));
        Assert.Throws<ArgumentException>(() => InferenceMemoryPlanner.Plan(input with { DevicePools = ["gpu", "gpu"] }));
        Assert.Throws<ArgumentException>(() => InferenceMemoryPlanner.Plan(input with
        { Candidates = [Candidate("lost-transfer") with { TransferBuffer = new(Host: 1) }] }));
    }

    private static InferenceMemoryPool Pool(string name, long bytes)
        => new(name, bytes, bytes, 0, new(name, bytes, 0, 0));

    private static InferenceExecutionCandidate Candidate(string name,
        InferenceWeightPlacement placement = InferenceWeightPlacement.Resident)
        => new() { Name = name, Placement = placement, PrefillChunkTokens = 10, PreservesExecutionGraph = placement == InferenceWeightPlacement.Resident };

    private static InferenceMemoryPlanningInput Input(long host, long device)
        => new()
        {
            Pools = [Pool("ram", host), Pool("gpu", device)], HostPools = ["ram"], DevicePools = ["gpu"],
            Model = new() { DenseWeights = new(100, 100) }, Workload = new(10, 10, 1, 1),
            Candidates = [Candidate("resident")]
        };
}
