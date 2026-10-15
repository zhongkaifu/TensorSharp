// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;

// Diagnostic phase snapshots, outside inference timers. PageFaultCount includes
// soft faults; IO counters do not establish physical reads of mapped GGUF pages.
internal static class ProbeProcessCounters
{
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryCounters
    {
        public uint Size, PageFaultCount;
        public nuint PeakWorkingSet, WorkingSet, PeakPagedPool, PagedPool;
        public nuint PeakNonpagedPool, NonpagedPool, Pagefile, PeakPagefile, PrivateUsage;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperations, WriteOperations, OtherOperations;
        public ulong ReadBytes, WriteBytes, OtherBytes;
    }

    [DllImport("psapi.dll", SetLastError = true)]
    private static extern bool GetProcessMemoryInfo(IntPtr process, ref MemoryCounters counters, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetProcessIoCounters(IntPtr process, out IoCounters counters);

    internal static object Capture(string stage)
    {
        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        if (!OperatingSystem.IsWindows()) return new { stage, unix_ms = timestamp, status = "unavailable: Windows only" };
        using var process = Process.GetCurrentProcess();
        var memory = new MemoryCounters { Size = (uint)Marshal.SizeOf<MemoryCounters>() };
        if (!GetProcessMemoryInfo(process.Handle, ref memory, memory.Size))
            return new { stage, unix_ms = timestamp, status = "memory-counter-error", error = Marshal.GetLastWin32Error() };
        if (!GetProcessIoCounters(process.Handle, out var io))
            return new { stage, unix_ms = timestamp, status = "io-counter-error", error = Marshal.GetLastWin32Error() };
        return new { stage, unix_ms = timestamp, status = "available", pid = process.Id,
            page_faults_including_soft = memory.PageFaultCount,
            working_set_bytes = (ulong)memory.WorkingSet, peak_working_set_bytes = (ulong)memory.PeakWorkingSet,
            private_commit_bytes = (ulong)memory.PrivateUsage,
            io_read_operations = io.ReadOperations, io_read_bytes = io.ReadBytes,
            io_write_operations = io.WriteOperations, io_write_bytes = io.WriteBytes,
            cpu_ms = process.TotalProcessorTime.TotalMilliseconds };
    }
}
