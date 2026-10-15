// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace TensorSharp.Memory;

public static partial class HostMemoryInfo
{
    private const uint JobProcessMemory = 0x100, JobTotalMemory = 0x200;

    // GlobalMemoryStatusEx does not reliably reflect Job commit ceilings. Only
    // public, documented APIs are used here. A null Job handle observes the
    // immediate Job, never a proof that every hidden ancestor was inspected.
    private static HostMemorySnapshot ConstrainWindowsJob(HostMemorySnapshot system)
    {
        nint process = new(-1); // GetCurrentProcess pseudo-handle; never close it.
        if (!IsProcessInJob(process, 0, out bool inJob)) throw JobReadError("job membership");
        if (!inJob) return system;
        if (!QueryJobLimits(0, 9, out var limits, (uint)Marshal.SizeOf<WindowsJobLimits>(), 0))
            throw JobReadError("immediate job limits");
        uint flags = limits.Basic.Flags;
        if ((flags & (JobProcessMemory | JobTotalMemory)) == 0) return system;

        bool soleProcess = false;
        if ((flags & JobTotalMemory) != 0)
        {
            if (!QueryJobProcesses(0, 3, out var members, (uint)Marshal.SizeOf<WindowsJobProcesses>(), 0))
                throw JobReadError("job process membership (shared Job commitment is not observable)");
            soleProcess = members.Assigned == 1 && members.Listed == 1 && members.First == (nuint)Environment.ProcessId;
        }
        var usage = new WindowsProcessMemory { Size = (uint)Marshal.SizeOf<WindowsProcessMemory>() };
        if (!K32GetProcessMemoryInfo(process, ref usage, usage.Size)) throw JobReadError("process private commitment");
        return ApplyWindowsJobCommit(system,
            (flags & JobProcessMemory) != 0 ? (ulong)limits.ProcessLimit : null,
            (flags & JobTotalMemory) != 0 ? (ulong)limits.JobLimit : null,
            (ulong)usage.PrivateUsage, soleProcess);
    }

    // A single-process Job's current usage can be attributed to this process.
    // For shared Jobs, neither a notification snapshot nor historical peak is
    // current aggregate usage. Refuse that unknown constraint instead of granting
    // the full limit or permanently subtracting the historical peak.
    internal static HostMemorySnapshot ApplyWindowsJobCommit(HostMemorySnapshot system,
        ulong? processLimit, ulong? jobLimit, ulong privateBytes, bool soleProcess)
    {
        if (system.Total <= 0 || system.Available < 0 || system.Available > system.Total)
            throw new IOException("Invalid Windows physical memory observation.");
        if (system.CommitLimit is not > 0 || system.CommitAvailable is not >= 0
            || system.CommitAvailable > system.CommitLimit || privateBytes > long.MaxValue)
            throw new IOException("Invalid Windows commit observation.");
        if (!processLimit.HasValue && !jobLimit.HasValue) return system;
        if (jobLimit.HasValue && !soleProcess)
            throw new IOException("The immediate Windows Job has a shared commit limit; current aggregate commitment is unknown.");
        long total = system.CommitLimit.Value, available = system.CommitAvailable.Value;
        void Apply(ulong limit)
        {
            if (limit == 0 || limit > long.MaxValue) throw new IOException("Invalid Windows Job commit limit.");
            total = Math.Min(total, (long)limit);
            // A limit can shrink below existing usage. That is real pressure,
            // not unsigned underflow or a reason to grant unlimited credit.
            long free = limit > privateBytes ? (long)(limit - privateBytes) : 0;
            available = Math.Min(available, free);
        }
        if (processLimit is ulong perProcess) Apply(perProcess);
        if (jobLimit is ulong perJob) Apply(perJob);
        return system with
        {
            CommitLimit = total, CommitAvailable = available,
            CommitConstraintScope = jobLimit.HasValue
                ? "Windows allowance intersected with immediate single-process Job commitment; ancestors unverified"
                : "Windows allowance intersected with immediate Job per-process commitment; ancestors unverified"
        };
    }

    private static IOException JobReadError(string operation)
        => new("Could not observe Windows " + operation + ".", new Win32Exception(Marshal.GetLastPInvokeError()));

    [StructLayout(LayoutKind.Sequential)]
    internal struct WindowsJobBasicLimits
    {
        public long ProcessTime, JobTime;
        public uint Flags;
        public nuint MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcessLimit;
        public nuint Affinity;
        public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    internal struct WindowsJobLimits
    {
        public WindowsJobBasicLimits Basic;
        public ulong Reads, Writes, Other, ReadBytes, WriteBytes, OtherBytes;
        public nuint ProcessLimit, JobLimit, PeakProcess, PeakJob;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct WindowsJobProcesses
    {
        public uint Assigned, Listed;
        public nuint First, Second;
    }
    [StructLayout(LayoutKind.Sequential)]
    internal struct WindowsProcessMemory
    {
        public uint Size, PageFaultCount;
        public nuint PeakWorkingSet, WorkingSet, PeakPagedPool, PagedPool,
            PeakNonpagedPool, NonpagedPool, PagefileUsage, PeakPagefileUsage, PrivateUsage;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool IsProcessInJob(nint process, nint job, [MarshalAs(UnmanagedType.Bool)] out bool result);
    [DllImport("kernel32.dll", EntryPoint = "QueryInformationJobObject", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool QueryJobLimits(nint job, int kind, out WindowsJobLimits data, uint size, nint returned);
    [DllImport("kernel32.dll", EntryPoint = "QueryInformationJobObject", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool QueryJobProcesses(nint job, int kind, out WindowsJobProcesses data, uint size, nint returned);
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool K32GetProcessMemoryInfo(nint process, ref WindowsProcessMemory data, uint size);
}
