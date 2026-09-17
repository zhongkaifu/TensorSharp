// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// What every family shares for the radix prefix cache's model contract
// (TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModel, DESIGN §6): a
// decode-graph reset counter and the spare-memory query. Neither changes what a
// model does; both only report.
using System;
using System.Diagnostics;
using System.Threading;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public abstract partial class ModelBase
    {
        private long _decodeGraphResets;

        /// <summary>
        /// How many times this model dropped its captured decode graphs (a
        /// <c>GgmlBasicOps.*Reset*DecodeCache</c> group, or a native release that
        /// drops every cached graph). One count per reset, however many native
        /// caches it clears. A release that disposes a holder has reset every
        /// running request's graphs; the prefix cache batches releases so this stays
        /// at most one per batch (DEC-24), and gate BG-11 watches it.
        /// </summary>
        public long DecodeGraphResets => Interlocked.Read(ref _decodeGraphResets);

        /// <summary>Record one decode-graph reset (see <see cref="DecodeGraphResets"/>).</summary>
        protected void CountDecodeGraphReset() => Interlocked.Increment(ref _decodeGraphResets);

        /// <summary>Whether a cache that has been active keeps device copies of its K/V besides the host
        /// bytes (a GGML GPU backend; Metal's double charge). Reads only the backend field, so a holder
        /// snapshot stays safe on a partially constructed model.</summary>
        protected bool KeepsDeviceKvMirrors =>
            _backend is BackendType.GgmlMetal or BackendType.GgmlCuda or BackendType.GgmlVulkan;

        /// <summary>
        /// <see cref="IPrefixCacheModel.QuerySpareBytes"/> for a model on this backend:
        /// the device classes report the measured device budget's spare (the same query
        /// <c>ResolvePrefillReservationLength</c> trims against), host K/V reports the
        /// process's available memory past its working set and a reserve, and page
        /// counts are the block pool's business. -1 when unknown.
        /// </summary>
        protected long QueryPrefixCacheSpareBytes(ResourceClass cls)
        {
            switch (cls)
            {
                case ResourceClass.DeviceKv:
                case ResourceClass.StateSnapshot:
                case ResourceClass.NativeSlot:
                    return GpuMemoryBudget.TryGetReservationSpareBytes(_backend, out long spare) ? Math.Max(0, spare) : -1;
                case ResourceClass.HostKv:
                    return HostSpareBytes();
                default:
                    return -1;
            }
        }

        /// <summary>Host headroom by the rule the Qwen 3.5 holder pool already applies: the GC's
        /// available-memory limit, less the process working set and a reserve. -1 when unknown.</summary>
        private static long HostSpareBytes()
        {
            try
            {
                long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                if (available <= 0) return -1;
                using var process = Process.GetCurrentProcess();
                long reserve = Math.Max(GpuMemoryBudget.MinHeadroomBytes, available / 16);
                return Math.Max(0, available - Math.Min(available, process.WorkingSet64) - reserve);
            }
            catch (InvalidOperationException) { return -1; }
            catch (System.ComponentModel.Win32Exception) { return -1; }
            catch (NotSupportedException) { return -1; }
        }
    }
}
