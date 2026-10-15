// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Real reservation/child ownership transactions; cache callbacks here
/// retire ledger children, not GPU buffers. Native fencing/physical reclamation
/// still requires separate native and real-model qualification.</summary>
public sealed class UnifiedMemoryPrefillPerformanceTierTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool, Device = UnifiedMemoryModelSession.DevicePool;
    private const long MiB = 1L << 20;
    private static MemoryCharge[] Charges(long hostMiB, long deviceMiB = 0)
        => [new(Host, checked(hostMiB * MiB)), new(Device, checked(deviceMiB * MiB))];
    private static long DeviceBytes(BudgetReservation owner) => owner.OwnedCharges.Single(c => c.Pool == Device).Bytes;

    internal static MemoryCharge[] CliffEstimate(int rows) => Charges(100, rows switch
    { >= 256 => 900, >= 128 => 400, >= 64 => 350, >= 32 => 250, >= 16 => 220, >= 8 => 210, _ => 200 });

    [Fact]
    public void WidestAffordableShapeReclaimsRealCacheBeforeAcceptingAnAlreadyFitting128Shape()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 700);
        using var request = budget.Reserve(Charges(10, 10));
        using var joint = budget.Reserve(Charges(10, 10));
        MemoryCharge[] Estimate(int rows) => Charges(100, rows >= 512 ? 400 : rows >= 256 ? 350 : 200);
        var limits = Charges(1000, 1000);
        var jointTarget = new InferenceMemoryBytes(50 * MiB, 100 * MiB);

        // This is a real ownership transaction, not an assumed fitting flag:
        // accepting the already-fitting lower tier spends no cache credit.
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelect(budget, request, joint, jointTarget,
            512, Estimate, limits, out var pure));
        Assert.Equal(128, pure.Rows);
        Assert.Equal(700 * MiB, cache.Bytes); Assert.Empty(cache.Events);

        // Restore exactly the same initial owners before the performance pass.
        Assert.True(budget.TryResizeTotals([(request, Charges(10, 10)), (joint, Charges(10, 10))],
            limits, requirePhysicalFit: true));
        int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, jointTarget,
            512, Estimate, () => limits, cache.Bytes, 100 * MiB, (plan, _) =>
            {
                calls++;
                Assert.Equal(512, plan.PreferredRows);
                Assert.Equal(200 * MiB, plan.DeviceShortfallBytes);
                Assert.Equal(436 * MiB, plan.TargetCacheBytes);
                cache.Trim(plan.TargetCacheBytes);
            }, ref attempted, out var selected));
        Assert.Equal(512, selected.Rows); Assert.Equal(1, calls); Assert.True(attempted);
        Assert.Equal(436 * MiB, cache.Bytes);
        Assert.Equal(400 * MiB, DeviceBytes(request)); Assert.Equal(100 * MiB, DeviceBytes(joint));
        Assert.Equal(new[] { "child-released", "root-shrunk", "replacement-child-committed" }, cache.Events);
        var device = budget.Snapshot().Single(p => p.Pool == Device);
        Assert.Equal(cache.Bytes + DeviceBytes(request) + DeviceBytes(joint), device.Committed + device.Reserved);
    }

    [Theory]
    [InlineData(200)] // The widest shape only exceeds Host.
    [InlineData(400)] // Both pools exceed their limits for the widest shape.
    public void WidestHostShortageFallsBackTo256WithoutRetiringCudaCache(int wideDevice)
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 700);
        using var request = budget.Reserve(Charges(10, 10)); using var joint = budget.Reserve(Charges(10, 10));
        int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            512, rows => rows == 512 ? Charges(400, wideDevice) : Charges(100, 200),
            () => Charges(300, 1000), cache.Bytes, 100 * MiB, (_, _) => calls++, ref attempted, out var selected));
        Assert.Equal(256, selected.Rows); Assert.Equal(0, calls); Assert.False(attempted);
        Assert.Equal(700 * MiB, cache.Bytes); Assert.Empty(cache.Events);
        Assert.Equal(200 * MiB, DeviceBytes(request)); Assert.Equal(100 * MiB, DeviceBytes(joint));
    }

    [Fact]
    public void RealCacheChildRetirementMakes128FitBeforeTheLegacyFourRowCliff()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 700);
        using var request = budget.Reserve(Charges(10, 10));
        using var joint = budget.Reserve(Charges(10, 10));
        int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            512, CliffEstimate, () => Charges(1000, 2000), cache.Bytes, 100 * MiB, (plan, limits) =>
            {
                calls++;
                Assert.Equal(128, plan.PreferredRows);
                Assert.Equal(200 * MiB, plan.DeviceShortfallBytes);
                Assert.Equal(436 * MiB, plan.TargetCacheBytes);
                // Observed hardware room exceeds the configured ledger ceiling;
                // deficit planning must still enforce min(hardware, capacity).
                Assert.Equal(2000 * MiB, limits.Single(c => c.Pool == Device).Bytes);
                cache.Trim(plan.TargetCacheBytes);
            }, ref attempted, out var selected));
        Assert.Equal(128, selected.Rows); Assert.Equal(1, calls);
        Assert.Equal(436 * MiB, cache.Bytes);
        Assert.Equal(400 * MiB, DeviceBytes(request));
        Assert.Equal(100 * MiB, DeviceBytes(joint));
        Assert.True(attempted);
        Assert.Equal(new[] { "child-released", "root-shrunk", "replacement-child-committed" }, cache.Events);
    }

    [Fact]
    public void Unaffordable128CanReclaimOnceFor32WithoutCrossingThePositiveCacheMinimum()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 400);
        using var request = budget.Reserve(Charges(10, 10)); using var joint = budget.Reserve(Charges(10, 10));
        int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            512, rows => Charges(100, rows >= 128 ? 900 : rows >= 64 ? 700 : rows >= 32 ? 540 : 400),
            () => Charges(1000, 1000), cache.Bytes, 300 * MiB, (plan, _) =>
            {
                calls++; Assert.Equal(32, plan.PreferredRows);
                Assert.Equal(40 * MiB, plan.DeviceShortfallBytes);
                Assert.Equal(300 * MiB, plan.TargetCacheBytes); // Hysteresis clips at the positive minimum.
                cache.Trim(plan.TargetCacheBytes);
            }, ref attempted, out var selected));
        Assert.Equal(32, selected.Rows); Assert.Equal(1, calls); Assert.Equal(300 * MiB, cache.Bytes);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(96)]
    [InlineData(127)]
    public void IntermediateMaximumProtectsItsWidestAffordableShapeBeforeAnAlreadyFitting32(int maximum)
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 400);
        using var request = budget.Reserve(Charges(10, 10)); using var joint = budget.Reserve(Charges(10, 10));
        int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            maximum, rows => Charges(100, rows == maximum ? 540 : rows >= 32 ? 500 : 400),
            () => Charges(1000, 1000), cache.Bytes, 300 * MiB, (plan, _) =>
            {
                calls++; Assert.Equal(maximum, plan.PreferredRows);
                Assert.Equal(40 * MiB, plan.DeviceShortfallBytes);
                Assert.Equal(300 * MiB, plan.TargetCacheBytes); cache.Trim(plan.TargetCacheBytes);
            }, ref attempted, out var selected));
        Assert.Equal(maximum, selected.Rows); Assert.Equal(1, calls); Assert.True(attempted);
        Assert.Equal(300 * MiB, cache.Bytes); Assert.Equal(540 * MiB, DeviceBytes(request));
        Assert.Equal(new[] { "child-released", "root-shrunk", "replacement-child-committed" }, cache.Events);
    }

    [Theory]
    [InlineData(200)] // Host-only shortage.
    [InlineData(700)] // Both pools short; a device trim cannot fix Host.
    public void HostShortageNeverTriggersTheGpuCacheCallback(int wideDevice)
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 400);
        using var request = budget.Reserve(Charges(10, 10)); using var joint = budget.Reserve(Charges(10, 10));
        int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            512, rows => rows >= 32 ? Charges(400, wideDevice) : Charges(100, 100),
            () => Charges(300, 1000), cache.Bytes, 100 * MiB, (_, _) => calls++, ref attempted, out var selected));
        Assert.Equal(16, selected.Rows); Assert.Equal(0, calls); Assert.Equal(400 * MiB, cache.Bytes);
    }

    [Fact]
    public void SeventeenTokenPromptKeepsThePureBoundedPassAndNeverReclaimsCache()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 400);
        using var request = budget.Reserve(Charges(10, 10)); using var joint = budget.Reserve(Charges(10, 10));
        var seen = new List<int>(); int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            17, rows => { seen.Add(rows); return Charges(100, rows >= 16 ? 800 : rows >= 8 ? 600 : 500); },
            () => Charges(1000, 1000), cache.Bytes, 100 * MiB, (_, _) => calls++, ref attempted, out var selected));
        Assert.Equal(4, selected.Rows); Assert.Equal(new[] { 17, 16, 8, 4 }, seen);
        Assert.Equal(0, calls); Assert.Equal(400 * MiB, cache.Bytes);
    }

    [Fact]
    public void AMinimumEqualToTheCurrentCacheCannotDisableTheCacheToBuyRows()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 400);
        using var request = budget.Reserve(Charges(10, 10)); using var joint = budget.Reserve(Charges(10, 10));
        int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            512, rows => Charges(100, rows > 4 ? 600 : 500), () => Charges(1000, 1000),
            cache.Bytes, cache.Bytes, (_, _) => calls++, ref attempted, out var selected));
        Assert.Equal(4, selected.Rows); Assert.Equal(0, calls); Assert.Equal(400 * MiB, cache.Bytes);
        Assert.True(cache.Bytes > 0);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void FreshPhysicalLimitOrCompetingOwnerAfterReclaimCannotRevokeLiveChildFloors(bool physicalLimitDrops)
    {
        var budget = new MemoryBudget(Charges(1000, 2000));
        using var cache = new CacheOwner(budget, 700);
        using var request = budget.Reserve(Charges(200, 200));
        using var child = request.TryTake(Charges(180, 180))!; child.Commit();
        using var joint = budget.Reserve(Charges(300, 300));
        using var arena = joint.TryTake(Charges(280, 280))!; arena.Commit();
        var originalRequest = request.OwnedCharges.ToArray(); var originalJoint = joint.OwnedCharges.ToArray();
        var originalChild = child.OwnedCharges.ToArray(); var originalArena = arena.OwnedCharges.ToArray();
        BudgetReservation? competitor = null; bool reclaimed = false; int calls = 0, postReclaimSamples = 0; bool attempted = false;
        try
        {
            Assert.False(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(100 * MiB, 100 * MiB),
                512, rows => Charges(100, rows >= 256 ? 900 : rows >= 128 ? 400 : 1),
                () => { if (reclaimed) postReclaimSamples++; return Charges(1000, reclaimed && physicalLimitDrops ? 500 : 1000); },
                cache.Bytes, 100 * MiB, (plan, _) =>
                {
                    calls++; Assert.Equal(128, plan.PreferredRows);
                    cache.Trim(plan.TargetCacheBytes); reclaimed = true;
                    if (!physicalLimitDrops) competitor = budget.Reserve(Charges(0, 600));
                }, ref attempted, out var selected));
            Assert.Equal(default, selected); Assert.Equal(1, calls); Assert.True(postReclaimSamples > 0);
            Assert.Equal(originalRequest, request.OwnedCharges); Assert.Equal(originalJoint, joint.OwnedCharges);
            Assert.Equal(originalChild, child.OwnedCharges); Assert.Equal(originalArena, arena.OwnedCharges);
            Assert.True(attempted);
            Assert.False(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(100 * MiB, 100 * MiB),
                512, rows => Charges(100, rows >= 256 ? 900 : rows >= 128 ? 400 : 1),
                () => Charges(1000, physicalLimitDrops ? 700 : 1000), cache.Bytes, 100 * MiB,
                (_, _) => throw new InvalidOperationException("A second trim in the same admission is forbidden"),
                ref attempted, out var retried));
            Assert.Equal(default, retried); Assert.Equal(1, calls); Assert.Equal(3, cache.Events.Count);
            Assert.Equal(originalRequest, request.OwnedCharges); Assert.Equal(originalJoint, joint.OwnedCharges);
            Assert.Equal(originalChild, child.OwnedCharges); Assert.Equal(originalArena, arena.OwnedCharges);
            long actualOwners = cache.Bytes + DeviceBytes(request) + DeviceBytes(joint) + (competitor == null ? 0 : DeviceBytes(competitor));
            var device = budget.Snapshot().Single(c => c.Pool == Device);
            Assert.Equal(actualOwners, device.Committed + device.Reserved);
            Assert.True(cache.Bytes >= 100 * MiB);
        }
        finally { competitor?.Dispose(); }
    }

    [Fact]
    public void FirstFullFitDoesNotReclaimOrSearchSmallerShapes()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 200);
        using var request = budget.Reserve(Charges(10, 10)); using var joint = budget.Reserve(Charges(10, 10));
        var seen = new List<int>(); int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            512, rows => { seen.Add(rows); return Charges(100, 100); }, () => Charges(1000, 1000),
            cache.Bytes, 100 * MiB, (_, _) => calls++, ref attempted, out var selected));
        Assert.Equal(512, selected.Rows); Assert.Equal(new[] { 512 }, seen); Assert.Equal(0, calls);
        Assert.Equal(200 * MiB, cache.Bytes);
    }

    [Fact]
    public void NonMonotonicLargerShapesAreMeasuredIndependentlyBeforeAnyCallback()
    {
        var budget = new MemoryBudget(Charges(1000, 1000));
        using var cache = new CacheOwner(budget, 300);
        using var request = budget.Reserve(Charges(10, 10)); using var joint = budget.Reserve(Charges(10, 10));
        var seen = new List<int>(); int calls = 0; bool attempted = false;
        Assert.True(UnifiedMemoryPrefillAdmission.TrySelectPerformance(budget, request, joint, new(50 * MiB, 100 * MiB),
            512, rows => { seen.Add(rows); return Charges(100, rows == 256 ? 300 : 1000); },
            () => Charges(1000, 1000), cache.Bytes, 100 * MiB, (_, _) => calls++, ref attempted, out var selected));
        Assert.Equal(256, selected.Rows); Assert.Equal(new[] { 512, 256 }, seen); Assert.Equal(0, calls);
    }

    private sealed class CacheOwner : IDisposable
    {
        private readonly MemoryBudget _budget;
        private readonly BudgetReservation _root;
        private BudgetReservation? _payload;
        internal readonly List<string> Events = [];
        internal long Bytes => DeviceBytes(_root);
        internal CacheOwner(MemoryBudget budget, long mib)
        {
            _budget = budget; _root = budget.Reserve(Charges(0, mib));
            _payload = _root.TryTake(Charges(0, mib))!; _payload.Commit();
        }
        internal void Trim(long bytes)
        {
            long before = _budget.Snapshot().Single(c => c.Pool == Device).Available;
            _payload!.Dispose(); _payload = null;
            // A retired child's credit first returns to its root. It is NOT
            // globally spendable before the owning cache root actually shrinks.
            Assert.Equal(before, _budget.Snapshot().Single(c => c.Pool == Device).Available);
            Events.Add("child-released");
            Assert.True(_root.TryResizeTotal([new(Host, 0), new(Device, bytes)]));
            Assert.Equal(bytes, Bytes); Events.Add("root-shrunk");
            _payload = _root.TryTake([new(Host, 0), new(Device, bytes)])!; _payload.Commit();
            Events.Add("replacement-child-committed");
        }
        public void Dispose() { _payload?.Dispose(); _root.Dispose(); }
    }
}
