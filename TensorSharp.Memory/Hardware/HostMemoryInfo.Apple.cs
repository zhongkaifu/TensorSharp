// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;

namespace TensorSharp.Memory;

public static partial class HostMemoryInfo
{
    private static (long Total, long Available) CaptureApple(bool appLimited)
    {
        try
        {
            nuint size = sizeof(ulong);
            if (AppleNative.sysctlbyname("hw.memsize", out ulong total, ref size, IntPtr.Zero, 0) != 0
                || size != sizeof(ulong))
                throw new IOException("sysctlbyname(hw.memsize) could not read physical memory.");
            uint host = AppleNative.Host.Value;
            if (host == 0 || AppleNative.host_page_size(host, out nuint pageSize) != 0)
                throw new IOException("host_page_size could not read the VM page size.");
            // HOST_VM_INFO, including revision 2. Use the stable 15 x uint ABI
            // instead of guessing the size of a newer vm_statistics64 revision.
            uint count = 15;
            if (AppleNative.host_statistics(host, 2, out AppleVmStatistics stats, ref count) != 0 || count < 15)
                throw new IOException("host_statistics could not read complete VM statistics.");
            ulong? appAvailable = appLimited ? AppleNative.os_proc_available_memory() : null;
            return FromAppleCounters(total, pageSize, stats.Free, stats.Purgeable, appAvailable);
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            throw new IOException("The Apple host memory APIs are unavailable; capacity is unknown.", ex);
        }
    }

    // free_count already includes speculative_count. Inactive pages can contain
    // dirty/anonymous data and are not all free: don't credit them, compressor
    // contents or all file-backed pages. The app limit is a remaining allowance,
    // not total physical RAM. In particular, zero does not mean 'unlimited'.
    internal static (long Total, long Available) FromAppleCounters(ulong total, ulong pageSize,
        uint freePages, uint purgeablePages, ulong? appAvailable)
    {
        if (pageSize == 0 || pageSize > int.MaxValue || (pageSize & (pageSize - 1)) != 0)
            throw new IOException("Invalid Apple VM page size.");
        ulong available = ((ulong)freePages + purgeablePages) * pageSize;
        var observation = Validate(total, available);
        if (appAvailable is { } remaining)
            observation.Available = (long)Math.Min(available, remaining);
        return observation;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct AppleVmStatistics
    {
        public uint Free, Active, Inactive, Wired;
        public uint ZeroFill, Reactivations, PageIns, PageOuts, Faults, CopyOnWrite, Lookups, Hits;
        public uint Purgeable, Purges, Speculative;
    }

    private static class AppleNative
    {
        private const string Library = "/usr/lib/libSystem.B.dylib";
        // One host send right for the process lifetime, not one leaked right per
        // refresh. Nested lazy initialization never loads libSystem on other OSes.
        internal static readonly Lazy<uint> Host = new(mach_host_self);

        [DllImport(Library)] internal static extern int sysctlbyname(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out ulong value, ref nuint length,
            IntPtr newValue, nuint newLength);
        [DllImport(Library)] private static extern uint mach_host_self();
        [DllImport(Library)] internal static extern int host_page_size(uint host, out nuint size);
        [DllImport(Library)] internal static extern int host_statistics(
            uint host, int flavor, out AppleVmStatistics statistics, ref uint count);
        [DllImport(Library)] internal static extern nuint os_proc_available_memory();
    }
}
