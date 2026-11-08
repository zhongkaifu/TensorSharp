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
using TensorSharp.GGML;
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
                    return ClampToMacMemory(HostSpareBytes());
                default:
                    return -1;
            }
        }

        /// <summary>
        /// <see cref="IPrefixCacheModel.QueryHostMemoryDeficitBytes"/>: on a Mac, how far what macOS can
        /// give without compressing has fallen below what it should keep (<see cref="UnifiedMemorySpareBytes"/>),
        /// or how far the device is from room for one more request (<see cref="DeviceMemoryDeficitBytes"/>),
        /// whichever is more; 0 when neither.
        /// </summary>
        public long QueryHostMemoryDeficitBytes()
        {
            long host = OperatingSystem.IsMacOS() && UnifiedMemorySpareBytes() is long spare && spare < 0 ? -spare : 0;
            return Math.Max(host, DeviceMemoryDeficitBytes());
        }

        /// <summary>
        /// Bytes the device is short of room for one more request's state beside a headroom, which the prefix
        /// cache then evicts from retained state; 0 by default. The host figure alone cannot see this on a Mac
        /// whose weights are mapped: their clean file pages count as available, so a model whose GPU working
        /// set was full never reported a deficit.
        /// </summary>
        protected virtual long DeviceMemoryDeficitBytes() => 0;

        /// <summary>The larger of these and 15% of RAM is what <see cref="UnifiedMemorySpareBytes"/> leaves to
        /// macOS and other programs.</summary>
        protected const long UnifiedMemoryMinReserveBytes = 7_000_000_000L;

        /// <summary>
        /// Apple silicon has one memory for the GPU and everything else, and what the GPU reads is
        /// wired, so retained K/V there comes out of the machine itself; past what macOS can give, it
        /// compresses and swaps and every step slows down. Qwen3.8-27B on MLX on a 48 GB Mac: sixteen
        /// retained conversations (each sized for the whole context) took the server past 34 GB, and
        /// decode fell from 25 to 7-12 tok/s. This is what macOS can give without compressing
        /// (<see cref="GgmlBasicOps.HostAvailableMemoryBytes"/>: free and purgeable pages and the clean
        /// file cache) less the larger of 7 GB and 15% of RAM - the rule the Qwen4Exp expert tiers are
        /// planned by. Negative when the machine is already short; null where it is not measured
        /// (the native query reports -1 off Apple).
        /// </summary>
        protected static long? UnifiedMemorySpareBytes()
        {
            try
            {
                long available = GgmlBasicOps.HostAvailableMemoryBytes();
                long ram = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                if (available < 0 || ram <= 0) return null;
                return available - Math.Max(UnifiedMemoryMinReserveBytes, ram / 100 * 15);
            }
            catch (DllNotFoundException) { return null; }
            catch (EntryPointNotFoundException) { return null; }
        }

        /// <summary>A host spare figure (-1 unknown) bounded, on macOS, by <see cref="UnifiedMemorySpareBytes"/>.</summary>
        private static long ClampToMacMemory(long spare)
        {
            if (!OperatingSystem.IsMacOS() || UnifiedMemorySpareBytes() is not long mac) return spare;
            mac = Math.Max(0, mac);
            return spare < 0 ? mac : Math.Min(spare, mac);
        }

        /// <summary>
        /// The capability record of a family with no end states (class P parity, MuseGlimmer, Nemotron,
        /// HunyuanDense; DESIGN §6.2-§6.4.3): its pages as the model serves them today — A1 host slabs where
        /// it restores snapshots across sequences, A2 model-paged storage where its batched paged forward is
        /// available — the window its pooled path is capped at, and the primary cache as a resident payload.
        /// Tree mode uses only the storage and truncation operations available on this instance.
        /// </summary>
        protected PrefixCacheCapabilities PageFamilyCapabilities(FamilyClass familyClass, TruncationKind truncation,
            int truncationParameter = 0, bool pagesNeedStateAtEnd = false, bool reuseAcrossMediaSpan = false)
        {
            bool a1 = SupportsKVStateSnapshot && SupportsCrossSequenceKvReuse;
            // Model-paged attention rows do not contain the recurrent state required
            // at a restorable boundary. Those families must use complete host slabs.
            // Nothing writes model-paged rows while the batched path is switched off
            // (--no-continuous-batching).
            bool a2 = !pagesNeedStateAtEnd
                && !TensorSharp.Runtime.Scheduling.ExecutionOptions.FromEnvironment().BatchedPathDisabled
                && this is TensorSharp.Runtime.Scheduling.IBatchedPagedModel paged && paged.BatchedForwardAvailable;
            PageSupport pages = a1 && a2 ? PageSupport.Both : a1 ? PageSupport.A1HostSlab : a2 ? PageSupport.A2ModelPaged : PageSupport.None;
            return new PrefixCacheCapabilities
            {
                Class = familyClass,
                NamespaceFingerprint = KVStateFingerprint,
                EndState = EndStateSupport.None,
                PrimaryResident = true,
                Truncation = SupportsKVCacheTruncation ? truncation : TruncationKind.None,
                TruncationParameter = truncationParameter,
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                RewindCapTokens = 16,
                Pages = pages,
                PagesNeedStateAtEnd = pagesNeedStateAtEnd,
                PageWindowTokens = MaxReusablePrefixTokens == int.MaxValue ? 0 : MaxReusablePrefixTokens,
                ReuseAcrossMediaSpan = reuseAcrossMediaSpan && SupportsReuseAcrossMediaSpan,
            };
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
