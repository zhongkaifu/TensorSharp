// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;

namespace TensorSharp.Memory;

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

    private static (long Total, long Available) CaptureWindows()
    {
        var status = new MemoryStatus { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
        if (!GlobalMemoryStatusEx(ref status))
            throw new IOException("GlobalMemoryStatusEx could not read physical memory.");
        return Validate(status.TotalPhysical, status.AvailablePhysical);
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
