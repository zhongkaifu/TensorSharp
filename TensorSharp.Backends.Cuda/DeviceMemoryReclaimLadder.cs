// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// What a CUDA allocator does when cuMemAlloc finds the device full. It used to run two induced GCs, wait for finalizers and
// free every block its pool cached, one synchronizing cuMemFree each, before retrying: in one 1,044-row prompt step of
// Qwen3.8-27B NVFP4 at a 24K context (issue #376) that was 1,614 cuMemFree calls and 1,091 ms of a 3.1 s step, and every
// later step allocated its buffers again. The ladder frees the least that makes room, the cheapest loss first.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// After a failed device allocation, climb one rung at a time and retry the allocation after each rung that freed
    /// something:
    /// <list type="number">
    /// <item>synchronize the stream, then give back cached large blocks, the least recently used size first, until the
    /// allocation, <see cref="TrimSlackBytes"/> and the pool's free-memory reserve are freed (<see cref="TrimTarget"/>);</item>
    /// <item>give back the pool shards' cached small blocks;</item>
    /// <item>ask each registered <see cref="IDeviceMemoryReclaimer"/>, then trim again what it released into the cache (not
    /// while a CUDA graph is being captured on the allocator, nor inside another climb on this thread: a reclaimer's own
    /// allocation that fails never asks the reclaimers again);</item>
    /// <item>the last resort, the old path: a full GC (storages whose last reference is an unreachable view are released by
    /// finalization) and a drain of every cached block, warned about once per process.</item>
    /// </list>
    /// While a capture runs on the allocator, the first synchronize aborts it (CudaGraphCaptureAbortedException) before
    /// anything is freed, as it did before: a captured graph may own blocks the cache holds; the capture site runs the
    /// region again unrecorded. Pure: the device, the pool and the warning come in as delegates, so tests run it over a
    /// fake device.
    /// </summary>
    internal sealed class DeviceMemoryReclaimLadder
    {
        /// <summary>What a trim frees beyond the allocation and the reserve: room for the driver's own granularity and the
        /// step's next few allocations, so the next one does not climb again at once.</summary>
        internal const long TrimSlackBytes = 64L << 20;

        private readonly Func<long, IntPtr> _tryAllocate;
        private readonly Action _synchronize;
        private readonly Func<long, long> _trimLarge;
        private readonly Func<long> _drainSmall;
        private readonly Func<IReadOnlyList<IDeviceMemoryReclaimer>> _reclaimers;
        private readonly Func<bool> _capturing;
        private readonly Action _collectAndDrain;
        private readonly ReportOnce _fullDrainReport;
        private readonly int _deviceId;
        private readonly Action<string> _trace;
        private readonly long _freeReserveBytes;

        // Climbs in progress on this thread (on any allocator): a nested one skips the reclaimers.
        [ThreadStatic] private static int t_climbs;

        private long _oomTrims, _oomReclaims, _fullDrains;

        /// <param name="tryAllocate">One device allocation: the pointer, <see cref="IntPtr.Zero"/> when the device is out of
        /// memory; any other failure throws.</param>
        /// <param name="synchronize">Wait for the work queued on the allocating thread's stream (the cached blocks it used).</param>
        /// <param name="trimLarge">Give back cached large blocks, the least recently used first, until the bytes asked are freed; returns
        /// the bytes freed.</param>
        /// <param name="drainSmall">Give back every cached small block; returns the bytes freed.</param>
        /// <param name="reclaimers">The reclaimers registered now (null or empty: none).</param>
        /// <param name="capturing">Whether a CUDA graph is being captured on the allocator.</param>
        /// <param name="collectAndDrain">The last resort: a full GC, the finalizers, and a drain of every cached block.</param>
        /// <param name="fullDrainReport">Where the last resort's warning goes, once.</param>
        /// <param name="deviceId">The GPU the warning names.</param>
        /// <param name="trace">Where a line per climb goes (what each rung freed and how long it took), or null: the CUDA
        /// allocator's TS_CUDA_LOG_VRAM=1 diagnostics.</param>
        /// <param name="freeReserveBytes">The free memory below which the pool hands returned blocks to the driver instead of
        /// caching them (TENSORSHARP_CUDA_POOL_MIN_FREE_MB): a trim leaves at least this much beside the allocation. One that
        /// left the device below it turned every later return into a synchronizing cuMemFree and every rent of that size into
        /// a cuMemAlloc (about 1,100 such frees within two seconds of a 24K-token prompt on an RTX PRO 4500).</param>
        /// <param name="budgetShortfall">Where an allocation past the device's dedicated memory or budget succeeds instead of
        /// failing (WDDM spills it to system memory; DeviceMemoryView.AllocationsMaySpill): the bytes an allocation of the
        /// given size would pass it by, with <paramref name="freeReserveBytes"/> kept free (0: it fits). Null elsewhere: there a
        /// full device fails the allocation and the ladder climbs from that.</param>
        /// <param name="budgetPassReport">Where the warning of an allocation placed past it goes, once (default: standard error,
        /// once per process).</param>
        internal DeviceMemoryReclaimLadder(Func<long, IntPtr> tryAllocate, Action synchronize, Func<long, long> trimLarge,
            Func<long> drainSmall, Func<IReadOnlyList<IDeviceMemoryReclaimer>> reclaimers, Func<bool> capturing,
            Action collectAndDrain, ReportOnce fullDrainReport, int deviceId, Action<string> trace = null,
            long freeReserveBytes = 0, Func<long, long> budgetShortfall = null, ReportOnce budgetPassReport = null)
        {
            _budgetShortfall = budgetShortfall;
            _budgetPassReport = budgetPassReport ?? ReportOnce.BudgetPass;
            _trace = trace;
            _freeReserveBytes = Math.Max(0, freeReserveBytes);
            _tryAllocate = tryAllocate ?? throw new ArgumentNullException(nameof(tryAllocate));
            _synchronize = synchronize ?? throw new ArgumentNullException(nameof(synchronize));
            _trimLarge = trimLarge ?? throw new ArgumentNullException(nameof(trimLarge));
            _drainSmall = drainSmall ?? throw new ArgumentNullException(nameof(drainSmall));
            _reclaimers = reclaimers ?? (() => null);
            _capturing = capturing ?? (() => false);
            _collectAndDrain = collectAndDrain ?? throw new ArgumentNullException(nameof(collectAndDrain));
            _fullDrainReport = fullDrainReport ?? throw new ArgumentNullException(nameof(fullDrainReport));
            _deviceId = deviceId;
        }

        /// <summary>Out-of-memory allocations that succeeded once cached blocks went back (rungs 1 and 2).</summary>
        internal long OomTrims => Interlocked.Read(ref _oomTrims);

        /// <summary>Out-of-memory allocations that succeeded once a reclaimer released memory (rung 3).</summary>
        internal long OomReclaims => Interlocked.Read(ref _oomReclaims);

        /// <summary>Out-of-memory allocations that reached the last resort (rung 4), whether it then succeeded or not.</summary>
        internal long FullDrains => Interlocked.Read(ref _fullDrains);

        /// <summary>The allocation of <paramref name="bytes"/>, climbing the ladder when the device is full;
        /// <see cref="IntPtr.Zero"/> when even the last resort made no room. Where an allocation past dedicated memory would
        /// succeed (<c>budgetShortfall</c>), the cheap rungs give memory back first (<see cref="KeepWithinBudget"/>).</summary>
        internal IntPtr Allocate(long bytes)
        {
            if (_budgetShortfall != null) KeepWithinBudget(bytes);
            IntPtr ptr = _tryAllocate(bytes);
            return ptr != IntPtr.Zero ? ptr : Climb(bytes);
        }

        private readonly Func<long, long> _budgetShortfall;
        private long _budgetTrims, _budgetReclaims, _budgetPasses;

        // Guard climbs to skip after one that freed nothing (nothing cached, no holder gave anything): an allocation past the
        // budget is then placed anyway, and asking again at once would only synchronize the stream for nothing.
        private int _budgetSkip;
        private const int BudgetSkipAfterBarren = 32;

        /// <summary>An allocation goes ahead past dedicated memory (or the budget): counted, and warned about once, since on WDDM
        /// it lands in shared system memory and runs at its speed (never silently).</summary>
        private void PassBudget(long bytes, long shortBy)
        {
            Interlocked.Increment(ref _budgetPasses);
            _budgetPassReport.Report(() => BudgetPassWarning(_deviceId, bytes, shortBy));
        }

        private readonly ReportOnce _budgetPassReport;

        /// <summary>The once-per-process warning of an allocation placed past dedicated memory.</summary>
        internal static string BudgetPassWarning(int deviceId, long bytes, long shortBy)
            => $"WARNING: [cuda] GPU {deviceId}: a {Mib(bytes)} MiB allocation passes the device's dedicated memory (or "
               + $"TENSORSHARP_CUDA_DEVICE_BUDGET_MB) by {Mib(shortBy)} MiB with nothing left to give back (no cached block, no "
               + "finished conversation): on Windows the driver places it in shared system memory, which is far slower. A shorter "
               + "context, q8_0 K/V or fewer concurrent requests fit. Reported once.";

        /// <summary>Allocations that would have passed dedicated memory and fit once cached blocks went back (rungs 1-2).</summary>
        internal long BudgetTrims => Interlocked.Read(ref _budgetTrims);

        /// <summary>Allocations that would have passed dedicated memory and fit once a reclaimer released memory (rung 3).</summary>
        internal long BudgetReclaims => Interlocked.Read(ref _budgetReclaims);

        /// <summary>Allocations placed past dedicated memory (or the budget): nothing left to give back made room.</summary>
        internal long BudgetPasses => Interlocked.Read(ref _budgetPasses);

        /// <summary>
        /// Before an allocation that would pass the device's dedicated memory (or budget) less the free reserve, where such an
        /// allocation succeeds instead of failing (issue #383: WDDM placed it in shared system memory, so no out-of-memory error
        /// ever started the climb and every room check passed with dedicated memory full): rungs 1-3, each only while still
        /// short - the cached large blocks the work stopped using, the shards' small blocks, then what registered holders give
        /// up - and no last resort (the allocation then goes ahead; on a real out-of-memory error <see cref="Climb"/> runs as
        /// before). Not while a graph is captured (the synchronize would abort it) nor inside another climb on this thread.
        /// </summary>
        private void KeepWithinBudget(long bytes)
        {
            long shortBy = _budgetShortfall(bytes);
            if (shortBy <= 0) return;
            if (_budgetSkip > 0)
            {
                _budgetSkip--;
                PassBudget(bytes, shortBy);
                return;
            }
            if (t_climbs > 0 || _capturing())
            {
                PassBudget(bytes, shortBy);
                return;
            }
            t_climbs++;
            long start = Stopwatch.GetTimestamp();
            long trimmed = 0, drained = 0, released = 0;
            int rung = 0;
            bool made = false;
            try
            {
                rung = 1;
                _synchronize();
                trimmed = _trimLarge(shortBy + TrimSlackBytes);
                if (trimmed > 0 && (shortBy = _budgetShortfall(bytes)) <= 0)
                {
                    Interlocked.Increment(ref _budgetTrims);
                    made = true;
                    return;
                }
                rung = 2;
                drained = _drainSmall();
                if (drained > 0 && (shortBy = _budgetShortfall(bytes)) <= 0)
                {
                    Interlocked.Increment(ref _budgetTrims);
                    made = true;
                    return;
                }
                if (_reclaimers() is { Count: > 0 } reclaimers)
                {
                    rung = 3;
                    foreach (IDeviceMemoryReclaimer reclaimer in reclaimers)
                    {
                        long got = reclaimer.Reclaim(shortBy);
                        if (got > 0) released += got;
                        _synchronize();
                        long again = _trimLarge(shortBy + TrimSlackBytes);
                        trimmed += again;
                        if ((got > 0 || again > 0) && (shortBy = _budgetShortfall(bytes)) <= 0)
                        {
                            Interlocked.Increment(ref _budgetReclaims);
                            made = true;
                            return;
                        }
                    }
                }
                PassBudget(bytes, shortBy);
                if (trimmed == 0 && drained == 0 && released == 0) _budgetSkip = BudgetSkipAfterBarren;
            }
            finally
            {
                t_climbs--;
                if (_trace != null)
                {
                    try
                    {
                        _trace($"[vram] GPU {_deviceId}: {Mib(bytes)} MiB would pass dedicated memory; rung {rung}"
                            + (made ? " kept it within" : $" left it {Mib(Math.Max(0, shortBy))} MiB past")
                            + $" (trimmed {Mib(trimmed)} MiB of large blocks, {Mib(drained)} MiB of small blocks"
                            + $", holders released {Mib(released)} MiB) in {Ms(start):F1} ms");
                    }
                    catch (Exception)
                    {
                        // Diagnostics must never break inference.
                    }
                }
            }
        }

        private IntPtr Climb(long bytes)
        {
            bool nested = t_climbs > 0;
            t_climbs++;
            long start = Stopwatch.GetTimestamp();
            long trimmed = 0, drained = 0, released = 0;
            int rung = 0;
            bool done = false;   // false in the finally: a rung threw (a capture aborted by the synchronize, say)
            IntPtr ptr = IntPtr.Zero;
            try
            {
                long target = TrimTarget(bytes, _freeReserveBytes);

                // 1. Cached large blocks, the least recently used size first (a pool rent found none of its own size).
                rung = 1;
                _synchronize();
                trimmed = _trimLarge(target);
                if (trimmed > 0 && (ptr = _tryAllocate(bytes)) != IntPtr.Zero)
                {
                    Interlocked.Increment(ref _oomTrims);
                    done = true;
                    return ptr;
                }

                // 2. The shards' small blocks.
                rung = 2;
                drained = _drainSmall();
                if (drained > 0 && (ptr = _tryAllocate(bytes)) != IntPtr.Zero)
                {
                    Interlocked.Increment(ref _oomTrims);
                    done = true;
                    return ptr;
                }

                // 3. What registered holders can give up (they decide themselves where and when they may act).
                if (!nested && !_capturing() && _reclaimers() is { Count: > 0 } reclaimers)
                {
                    rung = 3;
                    foreach (IDeviceMemoryReclaimer reclaimer in reclaimers)
                    {
                        long got = reclaimer.Reclaim(bytes);
                        if (got > 0) released += got;
                        _synchronize();
                        long again = _trimLarge(target);
                        trimmed += again;
                        if ((got > 0 || again > 0) && (ptr = _tryAllocate(bytes)) != IntPtr.Zero)
                        {
                            Interlocked.Increment(ref _oomReclaims);
                            done = true;
                            return ptr;
                        }
                    }
                }

                // 4. The last resort.
                rung = 4;
                Interlocked.Increment(ref _fullDrains);
                long drainStart = Stopwatch.GetTimestamp();
                _collectAndDrain();
                ptr = _tryAllocate(bytes);
                double ms = Ms(drainStart);
                bool succeeded = ptr != IntPtr.Zero;
                _fullDrainReport.Report(() => FullDrainWarning(_deviceId, bytes, released, ms, succeeded));
                done = true;
                return ptr;
            }
            finally
            {
                t_climbs--;
                if (_trace != null)
                {
                    try
                    {
                        _trace($"[vram] GPU {_deviceId}: {Mib(bytes)} MiB found the device full; rung {rung}"
                            + (!done ? " threw" : ptr != IntPtr.Zero ? " made room" : " made no room")
                            + $" (trimmed {Mib(trimmed)} MiB of large blocks, {Mib(drained)} MiB of small blocks"
                            + $", holders released {Mib(released)} MiB{(rung == 4 ? ", full GC and drain" : "")}) in {Ms(start):F1} ms");
                    }
                    catch (Exception)
                    {
                        // Diagnostics must never break inference.
                    }
                }
            }
        }

        /// <summary>What a climb's trim frees: the allocation, <see cref="TrimSlackBytes"/> and the pool's free-memory
        /// reserve.</summary>
        internal static long TrimTarget(long bytes, long freeReserveBytes)
        {
            long margin = TrimSlackBytes + Math.Max(0, freeReserveBytes);
            return bytes > long.MaxValue - margin ? long.MaxValue : bytes + margin;
        }

        private static double Ms(long since) => (Stopwatch.GetTimestamp() - since) * 1000.0 / Stopwatch.Frequency;

        /// <summary>The last resort's warning (no silent fallback: it names what was lost).</summary>
        internal static string FullDrainWarning(int deviceId, long bytes, long released, double ms, bool succeeded)
            => $"WARNING: [cuda] device memory ran out on GPU {deviceId} ({Mib(bytes)} MiB asked) after the pool's cached blocks"
               + (released > 0 ? $" and {Mib(released)} MiB its holders released" : "")
               + $" went back to the driver: ran a full GC and drained the pool ({ms:F0} ms)"
               + (succeeded ? "" : ", and the allocation still failed")
               + "; later steps allocate their buffers again. Reported once.";

        private static long Mib(long bytes) => (bytes + (1L << 20) - 1) >> 20;
    }

    /// <summary>A warning written once in the process (or once per instance, for tests).</summary>
    internal sealed class ReportOnce
    {
        /// <summary>The allocators' last-resort warning, on standard error.</summary>
        internal static ReportOnce FullDrain { get; } = new(line => Console.Error.WriteLine(line));

        /// <summary>The budget guard's warning (an allocation placed past dedicated memory), on standard error.</summary>
        internal static ReportOnce BudgetPass { get; } = new(line => Console.Error.WriteLine(line));

        /// <summary>The warning of a Windows device NVML does not report (room checks read cuMemGetInfo alone).</summary>
        internal static ReportOnce NoDedicatedReading { get; } = new(line => Console.Error.WriteLine(line));

        private readonly Action<string> _write;
        private int _reported;

        /// <param name="write">Where the line goes (tests capture it).</param>
        internal ReportOnce(Action<string> write) => _write = write ?? throw new ArgumentNullException(nameof(write));

        /// <summary>Write <paramref name="line"/>'s text unless this already wrote one; true when this call wrote it.</summary>
        internal bool Report(Func<string> line)
        {
            if (Interlocked.Exchange(ref _reported, 1) != 0)
                return false;
            try
            {
                _write(line());
            }
            catch (Exception)
            {
                // Diagnostics must never break inference.
            }
            return true;
        }
    }
}
