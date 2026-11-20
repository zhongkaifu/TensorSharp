// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// What the CUDA allocator and the models' room checks take a device's free and total memory to be (issue #383). On Windows
// (WDDM) cuMemGetInfo counts shared system memory in its total and only the calling process's allocations in its free, and a
// cuMemAlloc past dedicated memory succeeds by spilling to system memory, so every room check passed while dedicated memory
// was full (an RTX Spark N1X: cuMemGetInfo 50,803 MiB total, NVML 38,912 MiB; a 69K-token turn spilled 11.5 GiB).
using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>A device's dedicated memory: device-wide use and the dedicated total, as NVML reports them.</summary>
    public interface IDedicatedMemoryProbe
    {
        /// <summary>Device-wide used bytes (every process) and the dedicated total; false where not known now.</summary>
        bool TryRead(out long usedBytes, out long totalBytes);
    }

    /// <summary>
    /// A device's free and total memory for room checks: cuMemGetInfo, bounded by NVML's dedicated figures where NVML knows the
    /// device (the smaller free and the smaller total) and by a deployment budget (TENSORSHARP_CUDA_DEVICE_BUDGET_MB: the
    /// total is at most the budget, the free at most the budget less the device-wide use). Pure over its delegates, so tests
    /// run it over fake drivers.
    /// <para>Where allocations past the total can succeed (<see cref="AllocationsMaySpill"/>: Windows, where the driver places
    /// them in shared system memory, or a budget under the device's memory) the allocator gives memory back before an
    /// allocation would pass the total (<see cref="ShortfallBytes"/>), since no out-of-memory error comes to start its reclaim
    /// ladder.</para>
    /// </summary>
    public sealed class DeviceMemoryView
    {
        private readonly Func<(long Free, long Total)?> _driver;
        private readonly IDedicatedMemoryProbe _dedicated;
        private readonly long _budgetBytes;
        private readonly bool _driverMaySpill;

        // The last NVML reading and when it was taken (Stopwatch ticks), and the bytes this process allocated (+) and freed (-)
        // since: ShortfallBytes is asked at every backing allocation, and between readings the device-wide use moves by our own
        // allocations (another process's change shows at the next reading).
        private long _usedAtRead, _totalAtRead, _readAt, _ownSinceRead;
        private readonly object _sync = new();
        private static readonly long ReadTicks = System.Diagnostics.Stopwatch.Frequency / 10;

        /// <param name="driver">cuMemGetInfo's (free, total); null where it failed.</param>
        /// <param name="dedicated">NVML's device-wide use and dedicated total, or null where NVML is absent.</param>
        /// <param name="budgetBytes">A deployment's budget for this device (0: none).</param>
        /// <param name="driverMaySpill">Whether the driver may place an allocation past dedicated memory in system memory
        /// instead of failing it (Windows).</param>
        public DeviceMemoryView(Func<(long Free, long Total)?> driver, IDedicatedMemoryProbe dedicated, long budgetBytes, bool driverMaySpill)
        {
            _driver = driver ?? throw new ArgumentNullException(nameof(driver));
            _dedicated = dedicated;
            _budgetBytes = Math.Max(0, budgetBytes);
            _driverMaySpill = driverMaySpill;
        }

        /// <summary>The deployment budget in bytes (0: none).</summary>
        public long BudgetBytes => _budgetBytes;

        /// <summary>Whether NVML's dedicated figures bound the reading.</summary>
        public bool HasDedicatedProbe => _dedicated != null;

        /// <summary>Whether an allocation past <see cref="TryRead"/>'s total may succeed instead of failing: the driver spills
        /// (Windows) with a dedicated total to keep to, or a budget is set.</summary>
        public bool AllocationsMaySpill => _budgetBytes > 0 || (_driverMaySpill && _dedicated != null);

        /// <summary>Free and total bytes for room checks (see the class); false where the driver does not say.</summary>
        public bool TryRead(out long freeBytes, out long totalBytes)
        {
            freeBytes = totalBytes = 0;
            if (_driver() is not { } d || d.Total <= 0) return false;
            long free = Math.Max(0, d.Free), total = d.Total;
            long deviceUsed = Math.Max(0, d.Total - d.Free);
            if (ReadDedicated(out long used, out long dedicatedTotal))
            {
                total = Math.Min(total, dedicatedTotal);
                free = Math.Min(free, Math.Max(0, dedicatedTotal - used));
                deviceUsed = used;
            }
            if (_budgetBytes > 0)
            {
                total = Math.Min(total, _budgetBytes);
                free = Math.Min(free, Math.Max(0, _budgetBytes - deviceUsed));
            }
            freeBytes = Math.Min(free, total);
            totalBytes = total;
            return true;
        }

        /// <summary>
        /// The bytes by which an allocation of <paramref name="bytes"/> would leave less than <paramref name="reserveBytes"/> of
        /// the total free; 0 where it fits, and always 0 unless <see cref="AllocationsMaySpill"/> (elsewhere a full device fails
        /// the allocation itself). Reads NVML at most every 100 ms and counts this process's allocations and frees since
        /// (<see cref="NoteAllocated"/>).
        /// </summary>
        public long ShortfallBytes(long bytes, long reserveBytes)
        {
            if (!AllocationsMaySpill) return 0;
            long total, used;
            if (_dedicated != null)
            {
                lock (_sync)
                {
                    long now = System.Diagnostics.Stopwatch.GetTimestamp();
                    if (_readAt == 0 || now - _readAt >= ReadTicks)
                    {
                        if (!ReadDedicated(out long u, out long t)) return 0;
                        _usedAtRead = u;
                        _totalAtRead = t;
                        _ownSinceRead = 0;
                        _readAt = now;
                    }
                    used = _usedAtRead + _ownSinceRead;
                    total = _totalAtRead;
                }
            }
            else
            {
                if (_driver() is not { } d || d.Total <= 0) return 0;
                used = d.Total - d.Free;
                total = d.Total;
            }
            if (_budgetBytes > 0) total = Math.Min(total, _budgetBytes);
            return Math.Max(0, used + bytes + Math.Max(0, reserveBytes) - total);
        }

        /// <summary>This process allocated (positive) or freed (negative) device bytes: counted against the last NVML reading
        /// until the next one.</summary>
        public void NoteAllocated(long bytes)
        {
            if (_dedicated == null || !AllocationsMaySpill) return;
            lock (_sync) _ownSinceRead += bytes;
        }

        /// <summary>The next <see cref="ShortfallBytes"/> reads NVML anew (memory was given back).</summary>
        public void Invalidate()
        {
            lock (_sync) _readAt = 0;
        }

        private bool ReadDedicated(out long used, out long total)
        {
            used = total = 0;
            if (_dedicated == null) return false;
            try
            {
                return _dedicated.TryRead(out used, out total) && total > 0;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>One line for diagnostics: where the figures come from.</summary>
        public string Describe()
        {
            string driver = _driver() is { } d ? $"cuMemGetInfo free={d.Free >> 20} MiB of {d.Total >> 20} MiB" : "cuMemGetInfo n/a";
            string nvml = ReadDedicated(out long used, out long total) ? $"NVML used={used >> 20} MiB of {total >> 20} MiB" : "NVML n/a";
            string budget = _budgetBytes > 0 ? $", budget {_budgetBytes >> 20} MiB" : "";
            TryRead(out long free, out long room);
            return string.Create(CultureInfo.InvariantCulture,
                $"{driver}, {nvml}{budget}: room free={free >> 20} MiB of {room >> 20} MiB, spill guard {(AllocationsMaySpill ? "on" : "off")}");
        }

        /// <summary>TENSORSHARP_CUDA_DEVICE_BUDGET_MB: the device memory a deployment lets this process's room checks count on
        /// (0 or unset: the device's dedicated memory).</summary>
        public static long BudgetFromEnvironment()
        {
            string raw = Environment.GetEnvironmentVariable("TENSORSHARP_CUDA_DEVICE_BUDGET_MB");
            return long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out long mb) && mb > 0 ? mb << 20 : 0;
        }
    }

    /// <summary>NVML's reading of one CUDA device, matched by its UUID (then its PCI bus id, so CUDA_VISIBLE_DEVICES and NVML's
    /// own order do not matter); <see cref="Create"/> returns null where NVML or the device is not found.</summary>
    internal sealed class NvmlDedicatedMemoryProbe : IDedicatedMemoryProbe
    {
        private readonly IntPtr _device;

        private NvmlDedicatedMemoryProbe(IntPtr device) => _device = device;

        private static int s_init;   // 0: not tried, 1: initialized, 2: unavailable

        internal static NvmlDedicatedMemoryProbe Create(int cudaOrdinal)
        {
            if (string.Equals(Environment.GetEnvironmentVariable("TENSORSHARP_CUDA_NVML"), "0", StringComparison.Ordinal)) return null;
            try
            {
                if (Volatile.Read(ref s_init) == 0)
                    Interlocked.CompareExchange(ref s_init, NvmlApi.nvmlInit() == NvmlApi.Success ? 1 : 2, 0);
                if (Volatile.Read(ref s_init) != 1 || CudaDriverApi.cuDeviceGet(out int device, cudaOrdinal) != 0) return null;
                IntPtr handle = IntPtr.Zero;
                var uuid = new byte[16];
                if (CudaDriverApi.cuDeviceGetUuid(uuid, device) == 0)
                    foreach (string prefix in new[] { "GPU-", "MIG-" })
                        if (NvmlApi.nvmlDeviceGetHandleByUUID(NvmlApi.UuidString(uuid, prefix), out handle) == NvmlApi.Success) break;
                        else handle = IntPtr.Zero;
                if (handle == IntPtr.Zero)
                {
                    var bus = new byte[32];
                    if (CudaDriverApi.cuDeviceGetPCIBusId(bus, bus.Length, device) == 0)
                    {
                        string id = System.Text.Encoding.ASCII.GetString(bus).TrimEnd('\0');
                        if (NvmlApi.nvmlDeviceGetHandleByPciBusId(id, out IntPtr byBus) == NvmlApi.Success) handle = byBus;
                    }
                }
                if (handle == IntPtr.Zero) return null;
                var probe = new NvmlDedicatedMemoryProbe(handle);
                return probe.TryRead(out _, out long total) && total > 0 ? probe : null;   // GB10-class devices: not supported
            }
            catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException
                or MarshalDirectiveException)
            {
                Volatile.Write(ref s_init, 2);
                return null;
            }
        }

        public bool TryRead(out long usedBytes, out long totalBytes)
        {
            usedBytes = totalBytes = 0;
            if (NvmlApi.nvmlDeviceGetMemoryInfo(_device, out NvmlApi.Memory m) != NvmlApi.Success || m.Total == 0) return false;
            usedBytes = (long)m.Used;
            totalBytes = (long)m.Total;
            return true;
        }
    }
}
