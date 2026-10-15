// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;

namespace TensorSharp.Memory;

/// <summary>Physical capacity and an optional independent allowance for new
/// committed allocations. Commit is not resident RAM and must not be added to it.
/// Null means this provider does not establish a commit allowance.</summary>
public readonly record struct HostMemorySnapshot(long Total, long Available,
    long? CommitLimit = null, long? CommitAvailable = null)
{
    public string? CommitConstraintScope { get; init; }
}

/// <summary>Contemporaneous host capacity observations for planning, independent
/// of model, accelerator and ledger. Availability is not an atomic reservation
/// or a hard process limit. Re-read at admission boundaries; leave headroom for
/// other owners. Unknown/invalid observations throw instead of granting credit.</summary>
public static partial class HostMemoryInfo
{
    /// <summary>Windows reports physical RAM; Linux additionally applies visible
    /// cgroup constraints. Apple reports physical RAM and conservative free plus
    /// purgeable pages, further bounded by the app's remaining limit on iOS and
    /// Mac Catalyst. No GC heap limit is treated as physical capacity. Windows
    /// Job limits, hidden cgroup ancestors and future system pressure are not
    /// certified by this observation.</summary>
    public static (long Total, long Available) Capture()
    {
        if (OperatingSystem.IsWindows()) return CaptureWindows();
        if (OperatingSystem.IsLinux()) return CaptureLinux();
        if (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS() || OperatingSystem.IsMacCatalyst())
            return CaptureApple(OperatingSystem.IsIOS() || OperatingSystem.IsMacCatalyst());
        throw new PlatformNotSupportedException("No host memory observation provider is available on this platform.");
    }

    /// <summary>Read allocation constraints without changing the physical-only
    /// semantics used by mapped-file prefault. Windows commit fields come from
    /// GlobalMemoryStatusEx, intersected with observable immediate Job limits.
    /// Shared Job usage and hidden ancestor limits are not established; unknown
    /// shared Job commitment refuses planning. Other platforms
    /// retain their existing cgroup/app-constrained physical observation and
    /// report commit as unknown. Observations are not atomic with allocation and
    /// do not install a process hard limit.</summary>
    public static HostMemorySnapshot CaptureAllocationConstraints()
    {
        if (OperatingSystem.IsWindows())
        {
            var status = ReadWindowsStatus();
            return ConstrainWindowsJob(FromWindowsCounters(status.TotalPhysical, status.AvailablePhysical,
                status.TotalPageFile, status.AvailablePageFile));
        }
        var physical = Capture();
        return new(physical.Total, physical.Available);
    }

    private static (long Total, long Available) CaptureWindows()
    {
        var status = ReadWindowsStatus();
        return Validate(status.TotalPhysical, status.AvailablePhysical);
    }

    private static MemoryStatus ReadWindowsStatus()
    {
        var status = new MemoryStatus { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
        if (!GlobalMemoryStatusEx(ref status))
            throw new IOException("GlobalMemoryStatusEx could not read memory availability.");
        return status;
    }

    internal static HostMemorySnapshot FromWindowsCounters(ulong physicalTotal, ulong physicalAvailable,
        ulong commitLimit, ulong commitAvailable)
    {
        var physical = Validate(physicalTotal, physicalAvailable);
        // ullTotalPageFile is the smaller of the system/process commit limits;
        // ullAvailPageFile is the amount this process can currently commit.
        // These are neither the on-disk pagefile size nor extra physical RAM.
        var commit = Validate(commitLimit, commitAvailable);
        return new(physical.Total, physical.Available, commit.Total, commit.Available)
        { CommitConstraintScope = "Windows process allowance reported by GlobalMemoryStatusEx" };
    }

    internal static (long Total, long Available) Validate(ulong total, ulong available)
    {
        if (total == 0 || total > long.MaxValue || available > total)
            throw new IOException("Invalid host memory capacity observation.");
        return ((long)total, (long)available);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatus
    {
        public uint Length, Load;
        public ulong TotalPhysical, AvailablePhysical, TotalPageFile, AvailablePageFile,
            TotalVirtual, AvailableVirtual, AvailableExtendedVirtual;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
}
