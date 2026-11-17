// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Page-locked ranges of a model file's mapping that the CUDA executor stages experts from (a prompt chunk's experts the
    /// device tier does not hold). A copy from pageable memory goes through the driver's small bounce buffer at a fraction
    /// of the link's speed (9.3 against 55.6 GB/s on PCIe 5, ggml_ops_host_pin.cu); registering the pages the experts
    /// already occupy gives the DMA rate without a second copy of them. Pinned pages cannot be evicted, so the registered
    /// total stays within a budget (60% of the machine's memory); past it, or where the driver refuses a range, the copies
    /// stay pageable.
    ///
    /// Registrations are process-wide and page-aligned, kept disjoint (the driver refuses one overlapping another), so
    /// adjacent stacks that share a page split there - and the driver refuses a copy whose source spans two registrations,
    /// or a registration and unregistered memory. Every copy from a model file's mapping therefore goes through
    /// <see cref="CopyToDevice"/>, which cuts it at the registrations' edges. An instance owns the ranges it registered and
    /// unregisters them when disposed (with its model, before the mapping goes).
    /// </summary>
    internal sealed class CudaHostPins : IDisposable
    {
        private const long Page = 4096;
        private static readonly object Sync = new();
        // Disjoint, sorted by start (so by end too): found by binary search, as a host's experts pin as thousands of runs.
        private static readonly SortedList<long, (long End, CudaHostPins Owner, bool Mapped)> Ranges = new();
        private static readonly long Budget = Math.Max(0, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 10 * 6);
        // What a planner allowed beyond the budget (Allow): the host memory it measured free for the locks.
        private static long _allowed;
        private static long _pinned;
        private static bool _refused;
        // Ranges whose driver calls are in flight outside the lock (LockOutside, Split): held in Ranges so nothing registers
        // over them, never mapped for a kernel (DevicePointer), and their bytes reserved against the limit.
        private static readonly HashSet<long> Pending = new();
        private static long _reserved;

        /// <summary>Most bytes the process registers: 60% of the managed heap's limit, or what a planner allowed (the host
        /// memory it measured free, less a reserve), whichever is more.</summary>
        private static long Limit => Math.Max(Budget, _allowed);

        /// <summary>Let the process register up to <paramref name="bytes"/> (a planner's measure of the host memory free for
        /// the locks); never lowers the limit.</summary>
        public static void Allow(long bytes)
        {
            lock (Sync) _allowed = Math.Max(_allowed, bytes);
        }

        /// <summary>Bytes registered in the process.</summary>
        public static long PinnedBytes { get { lock (Sync) return _pinned; } }

        /// <summary>Bytes this instance registered.</summary>
        public long OwnBytes { get; private set; }

        /// <summary>Page-lock [<paramref name="ptr"/>, + <paramref name="bytes"/>) as far as the budget allows; true when all of
        /// it is locked (then copies from it run at DMA speed).</summary>
        public bool Pin(IntPtr ptr, long bytes)
        {
            if (ptr == IntPtr.Zero || bytes <= 0) return false;
            long start = ptr.ToInt64() / Page * Page, end = (ptr.ToInt64() + bytes + Page - 1) / Page * Page;
            lock (Sync)
            {
                // The gaps of [start, end) between the registered ranges.
                long at = start;
                var gaps = new List<(long, long)>();
                for (int i = FirstEndingAfter(start); i < Ranges.Count; i++)
                {
                    long s = Ranges.Keys[i], rEnd = Ranges.Values[i].End;
                    if (s >= end) break;
                    if (s > at) gaps.Add((at, s));
                    at = Math.Max(at, rEnd);
                }
                if (at < end) gaps.Add((at, end));
                if (gaps.Count == 0) return true;
                if (_refused) return false;
                bool all = true;
                foreach ((long s, long e) in gaps)
                {
                    if (_pinned + _reserved + (e - s) > Limit) { all = false; continue; }
                    // Mapped into the device's address space as well, so a kernel can read an expert where it lies (DevicePointer);
                    // a driver that refuses the mapping still locks the pages for copies.
                    bool mapped = true;
                    int rc = CudaDriverApi.cuMemHostRegister(new IntPtr(s), new UIntPtr((ulong)(e - s)),
                        CudaDriverApi.CU_MEMHOSTREGISTER_PORTABLE | CudaDriverApi.CU_MEMHOSTREGISTER_READ_ONLY | CudaDriverApi.CU_MEMHOSTREGISTER_DEVICEMAP);
                    if (rc != 0)
                    {
                        mapped = false;
                        rc = CudaDriverApi.cuMemHostRegister(new IntPtr(s), new UIntPtr((ulong)(e - s)),
                            CudaDriverApi.CU_MEMHOSTREGISTER_PORTABLE | CudaDriverApi.CU_MEMHOSTREGISTER_READ_ONLY);
                    }
                    if (rc != 0)
                    {
                        // A platform or driver that will not register mapped file pages: stay pageable from now on.
                        Console.Error.WriteLine($"[cuda] the experts' file pages could not be page-locked (CUDA error {rc}); " +
                            "staging copies run at the pageable rate.");
                        _refused = true;
                        return false;
                    }
                    Ranges[s] = (e, this, mapped);
                    _pinned += e - s;
                    OwnBytes += e - s;
                }
                return all;
            }
        }

        /// <summary>Register [<paramref name="s"/>, <paramref name="e"/>) mapped into the device's address space where the
        /// driver allows, else locked for copies only (<see cref="Pin"/>'s flags); the CUDA error, or 0.</summary>
        private static int Register(long s, long e, out bool mapped)
        {
            mapped = true;
            int rc = CudaDriverApi.cuMemHostRegister(new IntPtr(s), new UIntPtr((ulong)(e - s)),
                CudaDriverApi.CU_MEMHOSTREGISTER_PORTABLE | CudaDriverApi.CU_MEMHOSTREGISTER_READ_ONLY | CudaDriverApi.CU_MEMHOSTREGISTER_DEVICEMAP);
            if (rc == 0) return 0;
            mapped = false;
            return CudaDriverApi.cuMemHostRegister(new IntPtr(s), new UIntPtr((ulong)(e - s)),
                CudaDriverApi.CU_MEMHOSTREGISTER_PORTABLE | CudaDriverApi.CU_MEMHOSTREGISTER_READ_ONLY);
        }

        /// <summary>Placeholders [s, e) registered now, outside the lock; each made a registration of this instance, or
        /// dropped where the driver refused it. The bytes registered.</summary>
        private long RegisterPending(List<(long Start, long End)> pending, out bool all)
        {
            all = true;
            long registered = 0;
            foreach ((long s, long e) in pending)
            {
                int rc = Register(s, e, out bool mapped);
                lock (Sync)
                {
                    Pending.Remove(s);
                    _reserved -= e - s;
                    if (rc != 0)
                    {
                        Ranges.Remove(s);
                        all = false;
                        continue;
                    }
                    Ranges[s] = (e, this, mapped);
                    _pinned += e - s;
                    OwnBytes += e - s;
                    registered += e - s;
                }
            }
            return registered;
        }

        /// <summary>
        /// <see cref="Pin"/> with the driver's calls outside the process-wide lock: the gaps are held by placeholders while they
        /// register, so a thread that stages and maps experts meanwhile (CopyToDevice, DevicePointer) never waits for a
        /// registration (0.1-1 ms an expert's range on an A40). A worker locks the experts an adaptive trade is about to move
        /// to the host this way. True when all of it is locked (and mapped where the driver maps); <paramref name="registered"/>
        /// the bytes this call registered.
        /// </summary>
        internal bool LockOutside(IntPtr ptr, long bytes, out long registered)
        {
            registered = 0;
            if (ptr == IntPtr.Zero || bytes <= 0) return false;
            long start = ptr.ToInt64() / Page * Page, end = (ptr.ToInt64() + bytes + Page - 1) / Page * Page;
            var gaps = new List<(long Start, long End)>();
            bool all = true;
            lock (Sync)
            {
                long at = start;
                var found = new List<(long, long)>();
                for (int i = FirstEndingAfter(start); i < Ranges.Count; i++)
                {
                    long s = Ranges.Keys[i], rEnd = Ranges.Values[i].End;
                    if (s >= end) break;
                    if (s > at) found.Add((at, s));
                    at = Math.Max(at, rEnd);
                }
                if (at < end) found.Add((at, end));
                if (found.Count == 0) return true;
                if (_refused) return false;
                foreach ((long s, long e) in found)
                {
                    if (_pinned + _reserved + (e - s) > Limit) { all = false; continue; }
                    Ranges[s] = (e, this, false);
                    Pending.Add(s);
                    _reserved += e - s;
                    gaps.Add((s, e));
                }
            }
            registered = RegisterPending(gaps, out bool registeredAll);
            return all && registeredAll;
        }

        /// <summary>
        /// Registration [<paramref name="start"/>, <paramref name="end"/>) replaced by the page-aligned pieces
        /// <paramref name="keep"/> (each within it, disjoint, in order), registered by this instance. Between the unregistration
        /// and the pieces' registrations the kept pages are pageable, so nothing may copy from the range or read it on the
        /// device meanwhile: never while a model serves (a test's executor unlocks the tier its load locked before any step
        /// read it). False where the registration is not there or a piece could not be registered (it stays pageable).
        /// </summary>
        internal bool Split(long start, long end, IReadOnlyList<(long Start, long End)> keep)
        {
            (long End, CudaHostPins Owner, bool Mapped) r;
            lock (Sync)
            {
                if (!Ranges.TryGetValue(start, out r) || r.End != end || Pending.Contains(start)) return false;
                if (CudaDriverApi.cuMemHostUnregister(new IntPtr(start)) != 0) return false;
                Ranges.Remove(start);
                _pinned -= end - start;
                r.Owner.OwnBytes -= end - start;
            }
            var pieces = new List<(long Start, long End)>();
            lock (Sync)
                foreach ((long s, long e) in keep)
                {
                    if (s < start || e > end || s >= e) continue;
                    Ranges[s] = (e, this, false);
                    Pending.Add(s);
                    _reserved += e - s;
                    pieces.Add((s, e));
                }
            RegisterPending(pieces, out bool all);
            return all;
        }

        /// <summary>The registrations (not ones in flight) overlapping [<paramref name="start"/>, <paramref name="end"/>), in
        /// order.</summary>
        internal static List<(long Start, long End)> RegistrationsIn(long start, long end)
        {
            var found = new List<(long Start, long End)>();
            lock (Sync)
                for (int i = FirstEndingAfter(start); i < Ranges.Count && Ranges.Keys[i] < end; i++)
                    if (!Pending.Contains(Ranges.Keys[i])) found.Add((Ranges.Keys[i], Ranges.Values[i].End));
            return found;
        }

        /// <summary>Registered bytes within [<paramref name="start"/>, <paramref name="end"/>) (tests).</summary>
        internal static long LockedBytesIn(long start, long end)
        {
            long locked = 0;
            foreach ((long s, long e) in RegistrationsIn(start, end))
                locked += Math.Min(e, end) - Math.Max(s, start);
            return locked;
        }

        /// <summary>
        /// Copy [<paramref name="src"/>, + <paramref name="bytes"/>) of host memory to <paramref name="dst"/> on
        /// <paramref name="stream"/> (zero: a synchronous copy), one copy per registration the range crosses. Returns the
        /// first CUDA error, or 0.
        /// </summary>
        public static int CopyToDevice(IntPtr dst, IntPtr src, long bytes, IntPtr stream)
        {
            long at = src.ToInt64(), end = at + bytes;
            var cuts = new List<long>();
            lock (Sync)
                for (int i = FirstEndingAfter(at); i < Ranges.Count; i++)
                {
                    long s = Ranges.Keys[i], rEnd = Ranges.Values[i].End;
                    if (s >= end) break;
                    if (s > at) cuts.Add(s);
                    if (rEnd < end) cuts.Add(rEnd);
                }
            cuts.Add(end);
            cuts.Sort();
            foreach (long cut in cuts)
            {
                if (cut <= at) continue;
                IntPtr to = dst + (nint)(at - src.ToInt64());
                var n = new UIntPtr((ulong)(cut - at));
                int rc = stream == IntPtr.Zero
                    ? CudaDriverApi.cuMemcpyHtoD(to, new IntPtr(at), n)
                    : CudaDriverApi.cuMemcpyHtoDAsync(to, new IntPtr(at), n, stream);
                if (rc != 0) return rc;
                at = cut;
            }
            return 0;
        }

        /// <summary>
        /// The device's address of [<paramref name="host"/>, + <paramref name="bytes"/>) for a kernel to read in place, or zero:
        /// every byte must lie in registrations mapped into the device's address space, which unified addressing places at the
        /// host's own addresses (a range across registrations is then contiguous on the device as well). Only an expert wholly
        /// covered may be read so: one that began in the last page of a neighbour's lock and ran on into unlocked pages had
        /// faulted (CUDA error 700) when its first byte alone was checked.
        /// </summary>
        public static IntPtr DevicePointer(IntPtr host, long bytes)
        {
            if (host == IntPtr.Zero || bytes <= 0) return IntPtr.Zero;
            long at = host.ToInt64(), end = at + bytes;
            lock (Sync)
            {
                for (int i = FirstEndingAfter(at); i < Ranges.Count && at < end; i++)
                {
                    long s = Ranges.Keys[i];
                    var (rEnd, _, mapped) = Ranges.Values[i];
                    if (s > at || !mapped || Pending.Contains(s)) return IntPtr.Zero;
                    at = rEnd;
                }
                if (at < end) return IntPtr.Zero;
            }
            return CudaDriverApi.cuMemHostGetDevicePointer(out IntPtr device, host, 0) == 0 && device == host ? device : IntPtr.Zero;
        }

        /// <summary>The index of the first registered range ending past <paramref name="at"/> (Count if none); under the lock.</summary>
        private static int FirstEndingAfter(long at)
        {
            IList<long> keys = Ranges.Keys;
            int lo = 0, hi = keys.Count - 1, last = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) / 2;
                if (keys[mid] <= at) { last = mid; lo = mid + 1; }
                else hi = mid - 1;
            }
            // The range starting at or before at covers it when it ends past it; the next one starts after it.
            return last >= 0 && Ranges.Values[last].End > at ? last : last + 1;
        }

        public void Dispose()
        {
            lock (Sync)
            {
                var mine = new List<long>();
                // (A range in flight is its registering thread's: an owner disposes only once that thread is done.)
                foreach (var (s, r) in Ranges)
                    if (ReferenceEquals(r.Owner, this) && !Pending.Contains(s)) mine.Add(s);
                foreach (long s in mine)
                {
                    CudaDriverApi.cuMemHostUnregister(new IntPtr(s));
                    _pinned -= Ranges[s].End - s;
                    Ranges.Remove(s);
                }
            }
        }
    }
}
