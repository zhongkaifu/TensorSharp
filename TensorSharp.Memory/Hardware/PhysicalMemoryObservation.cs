// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;

namespace TensorSharp.Memory.Hardware;

/// <summary>These metrics are not interchangeable, even on one physical device.</summary>
public enum PhysicalMemoryMetric
{
    HostAvailable, AllocatorAvailable, PrivateCommitAvailable, ApplicationFootprintAvailable
}

[Flags]
public enum MemoryHardLimitCoverage
{
    None = 0,
    PrivateCommit = 1,
    AnonymousResident = 2,
    FilePageCache = 4,
    KernelMemory = 8,
    DeviceAllocations = 16,
    DriverAllocations = 32,
    HostPhysical = AnonymousResident | FilePageCache | KernelMemory,
    AllPhysical = HostPhysical | DeviceAllocations | DriverAllocations
}

/// <summary>Evidence supplied by the platform enforcement owner, not inferred
/// from an observed limit or ledger. Active means the mechanism is installed and
/// verified for this observation's scope. A Windows Job private-commit limit does
/// not cover process RSS, mapped file cache or VRAM; a GC heap limit does not cover
/// native allocations; an Apple availability hint does not establish enforcement.</summary>
public sealed record MemoryHardLimitEvidence(
    string Mechanism, MemoryHardLimitCoverage Coverage, long LimitBytes, bool Active);

/// <summary>A single allocation constraint. Domain identifies shared physical
/// storage (e.g. UMA RAM), not a capacity to sum with other constraints. The caller
/// captures platform/allocator free space and the ledger close together. None of
/// these observations makes unhooked allocations strictly budget-owned.</summary>
public sealed record PhysicalMemoryObservation(
    string Constraint, string Pool, string Domain, PhysicalMemoryMetric Metric,
    long TotalBytes, long AvailableBytes, DateTimeOffset ObservedAt)
{
    /// <summary>Allocator-certified ledger bytes excluded by AvailableBytes.
    /// Unknown is zero; aggregate process/GC/mmap gauges cannot certify overlap.</summary>
    public long? IncludedCommittedBytes { get; init; }
    public long MaximumBudgetBytes { get; init; } = long.MaxValue;
    public Planning.AutomaticMemoryHeadroom? Headroom { get; init; }
    public MemoryHardLimitCoverage RequiredHardLimitCoverage { get; init; }
    public MemoryHardLimitEvidence? Enforcement { get; init; }
}

/// <summary>Overlapping diagnostic gauges, never additive ownership categories.
/// WorkingSet includes resident GC/native/mapped pages. GCCommitted is virtual GC
/// backing and HeapSize is from the last GC; neither proves currently resident
/// bytes. Missing private-commit support is null, not zero. Use host availability
/// or allocator observations for admission, not a sum of these gauges.</summary>
public sealed record ProcessMemoryGauges(
    DateTimeOffset ObservedAt, long WorkingSetBytes, long? PrivateCommittedBytes,
    long GcCommittedBytes, long LastGcHeapBytes, long ManagedLiveEstimateBytes)
{
    public static ProcessMemoryGauges Capture()
    {
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        long? privateBytes;
        try { privateBytes = process.PrivateMemorySize64; }
        catch (PlatformNotSupportedException) { privateBytes = null; }
        var gc = GC.GetGCMemoryInfo();
        return new(DateTimeOffset.UtcNow, process.WorkingSet64, privateBytes,
            gc.TotalCommittedBytes, gc.HeapSizeBytes, GC.GetTotalMemory(false));
    }
}
