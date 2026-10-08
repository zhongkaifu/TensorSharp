// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory;

/// <summary>Pure placement policy extracted from the existing Qwen4Exp offload
/// implementation. Groups can be layers, experts or any executor-defined contiguous
/// units. Layout support and scratch requirements are supplied by the adapter.</summary>
public static class TieredPlacementPlanner
{
    /// <summary>Keep a suffix of whole groups resident. When every group can use an
    /// exact selected-unit cache and all weights do not fit, prefer that cache. A
    /// heterogeneous cache and whole resident groups must share the same free bytes.</summary>
    public static int PlanDiscreteResidency(long[] groupBytes, long pendingBytes, long freeBytes,
        long headroomBytes, long scratchBytes, long cacheBytes, long[]? cacheMinimumBytes = null,
        ulong cachePartitions = 48, long[]? cacheAllocationFloorBytes = null)
    {
        ArgumentNullException.ThrowIfNull(groupBytes);
        if (pendingBytes < 0 || headroomBytes < 0 || scratchBytes < 0 || cacheBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(pendingBytes));
        if (cacheMinimumBytes != null && cacheMinimumBytes.Length != groupBytes.Length)
            throw new ArgumentException("Cache layouts must match the group count.", nameof(cacheMinimumBytes));
        if (cacheAllocationFloorBytes != null && (cacheMinimumBytes == null || cacheAllocationFloorBytes.Length != groupBytes.Length))
            throw new ArgumentException("Cache allocation floors must match the cache layouts.", nameof(cacheAllocationFloorBytes));
        foreach (long bytes in groupBytes) ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        if (cacheMinimumBytes != null)
            foreach (long bytes in cacheMinimumBytes) ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        if (cacheAllocationFloorBytes != null)
            for (int i = 0; i < cacheAllocationFloorBytes.Length; i++)
                if (cacheAllocationFloorBytes[i] < 0 || cacheAllocationFloorBytes[i] > cacheMinimumBytes![i])
                    throw new ArgumentOutOfRangeException(nameof(cacheAllocationFloorBytes));
        long available = Math.Max(0, freeBytes);
        foreach (long reserve in new[] { pendingBytes, headroomBytes, scratchBytes })
            available = reserve >= available ? 0 : available - reserve;
        int Fit(long budget)
        {
            int resident = 0;
            for (int i = groupBytes.Length - 1; i >= 0; i--)
            {
                if (groupBytes[i] > budget) break;
                budget -= groupBytes[i];
                resident++;
            }
            return resident;
        }
        int count = Fit(available);
        if (count == groupBytes.Length || cacheBytes == 0 || cachePartitions == 0 || cacheMinimumBytes == null) return count;
        long quota = (long)((ulong)cacheBytes / cachePartitions);
        bool CacheFits(int i) => cacheMinimumBytes[i] > 0 && cacheMinimumBytes[i] <= quota;
        bool CacheMayAllocate(int i)
        {
            long floor = cacheAllocationFloorBytes == null ? cacheMinimumBytes[i] : cacheAllocationFloorBytes[i];
            return floor > 0 && floor <= quota;
        }
        bool all = cachePartitions >= (ulong)groupBytes.Length;
        for (int i = 0; i < groupBytes.Length; i++) all &= CacheFits(i);
        // This selects an execution path, not permission to allocate cacheBytes:
        // the runtime cache must still acquire the actual physical reservation.
        if (all) return 0;
        for (;;)
        {
            int eligible = 0;
            for (int i = 0; i < groupBytes.Length - count; i++) if (CacheMayAllocate(i)) eligible++;
            long reserve = quota == 0 || eligible == 0 ? 0
                : eligible > cacheBytes / quota ? cacheBytes : quota * eligible;
            int next = Fit(reserve >= available ? 0 : available - reserve);
            if (next == count) return count;
            count = next;
        }
    }

    /// <summary>Unified-memory devices have BOTH a physical RAM constraint and a
    /// device working-set constraint. Wiring a group removes RAM available for hot
    /// offloaded pages. Hot fraction is a measured policy input, not a quality knob.</summary>
    public static int PlanUnifiedResidency(long[] groupBytes, long otherDeviceBytes, long workingSetBytes,
        long ramBytes, long hostReserveBytes, long deviceHeadroomBytes, long scratchBytes, double offloadedHotFraction)
    {
        ArgumentNullException.ThrowIfNull(groupBytes);
        if (otherDeviceBytes < 0 || workingSetBytes < 0 || ramBytes < 0 || hostReserveBytes < 0 || deviceHeadroomBytes < 0 || scratchBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(otherDeviceBytes));
        if (!double.IsFinite(offloadedHotFraction) || offloadedHotFraction < 0 || offloadedHotFraction > 1)
            throw new ArgumentOutOfRangeException(nameof(offloadedHotFraction));
        foreach (long bytes in groupBytes) ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        static long SaturatingAdd(long a, long b) => b > long.MaxValue - a ? long.MaxValue : a + b;
        long Hot(long bytes) => offloadedHotFraction == 1 ? bytes : (long)(bytes * offloadedHotFraction);
        long wired = SaturatingAdd(otherDeviceBytes, scratchBytes);
        // Use decimal only for this load-time aggregate; a saturated aggregate
        // cannot later be decremented correctly when moving groups to the device.
        decimal hot = groupBytes.Sum(b => (decimal)Hot(b));
        long workingBudget = Math.Max(0, workingSetBytes - deviceHeadroomBytes);
        int count = 0;
        for (int i = groupBytes.Length - 1; i >= 0; i--)
        {
            if (wired > workingBudget || groupBytes[i] > workingBudget - wired) break;
            long nextWired = wired + groupBytes[i];
            decimal nextHot = hot - Hot(groupBytes[i]);
            if (nextWired > ramBytes || (decimal)(ramBytes - nextWired) < hostReserveBytes + nextHot) break;
            wired = nextWired;
            hot = nextHot;
            count++;
        }
        return count;
    }
}
