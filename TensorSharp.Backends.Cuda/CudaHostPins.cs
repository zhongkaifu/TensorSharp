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
        private static readonly SortedList<long, (long End, CudaHostPins Owner)> Ranges = new();
        private static readonly long Budget = Math.Max(0, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 10 * 6);
        // What a planner allowed beyond the budget (Allow): the host memory it measured free for the locks.
        private static long _allowed;
        private static long _pinned;
        private static bool _refused;

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
                    if (_pinned + (e - s) > Limit) { all = false; continue; }
                    int rc = CudaDriverApi.cuMemHostRegister(new IntPtr(s), new UIntPtr((ulong)(e - s)),
                        CudaDriverApi.CU_MEMHOSTREGISTER_PORTABLE | CudaDriverApi.CU_MEMHOSTREGISTER_READ_ONLY);
                    if (rc != 0)
                    {
                        // A platform or driver that will not register mapped file pages: stay pageable from now on.
                        Console.Error.WriteLine($"[cuda] the experts' file pages could not be page-locked (CUDA error {rc}); " +
                            "staging copies run at the pageable rate.");
                        _refused = true;
                        return false;
                    }
                    Ranges[s] = (e, this);
                    _pinned += e - s;
                    OwnBytes += e - s;
                }
                return all;
            }
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
                foreach (var (s, r) in Ranges)
                    if (ReferenceEquals(r.Owner, this)) mine.Add(s);
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
