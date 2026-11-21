// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// VRAM headroom policy for the GGML GPU backends.
    ///
    /// Running a GPU backend to zero free bytes does not fail loudly. On
    /// Windows/WDDM the driver silently pages device allocations out to system
    /// RAM instead, and a weight tensor that lands in host memory turns a
    /// bandwidth-bound decode into a PCIe-bound one. Measured on a 16 GB RTX 3080
    /// Laptop with Qwen3.8-27B-UD-IQ4_XS (13.2 GB of resident weights): with no
    /// headroom left, a 1 GB allocation by any other process — a browser showing
    /// the built-in web UI is enough — dropped decode from 18.0 to 3.5 tok/s, and
    /// it did not recover when that process exited. llama.cpp on the same card and
    /// model keeps ~2 GB free and loses 2% under identical pressure.
    ///
    /// The buffers sized by policy rather than by the model — how many KV tokens
    /// to reserve up front, how wide a prefill chunk to build — are therefore
    /// capped against what is actually free, instead of by a constant that
    /// happened to fit the card the default was chosen on. Everything here is a
    /// cap: it can only ever hand back something at or below the caller's own
    /// default, so a card with room keeps today's behaviour exactly.
    ///
    /// Memory an owner will allocate later can be claimed ahead of use
    /// (<see cref="Reserve"/>): the spare readings here, and the shared decoder's
    /// slot admission and room checks, leave it out, so a cache that grows on
    /// demand does not take it first. A vision tower whose weights upload with the
    /// first image claims them, and its largest image's compute, when it loads.
    /// </summary>
    internal static class GpuMemoryBudget
    {
        /// <summary>Never reserve less than this much VRAM, however small the card.</summary>
        internal const long MinHeadroomBytes = 512L * 1024 * 1024;

        /// <summary>
        /// Backends whose allocations come out of a fixed device pool that a
        /// competing process can push into host memory. Metal is unified-memory
        /// and already has its own conservative caps; CPU has no device budget.
        /// </summary>
        internal static bool AppliesTo(BackendType backend) =>
            backend == BackendType.GgmlCuda || backend == BackendType.GgmlVulkan;

        /// <summary>
        /// Backends whose device budget also bounds a one-off <em>reservation</em> —
        /// a buffer sized once from a request's declared budget and then kept for
        /// the whole request, rather than a steady-state working buffer.
        ///
        /// Metal is on this list even though it is unified-memory, because
        /// <c>recommendedMaxWorkingSetSize</c> is a hard ceiling and not a soft one.
        /// Past it a command buffer does not degrade, it fails outright with
        /// <c>kIOGPUCommandBufferCallbackErrorOutOfMemory</c>, and ggml-metal then
        /// latches a sticky error state (ggml-metal-context.m: <c>ctx-&gt;has_error</c>)
        /// that fails EVERY later graph until the backend is recreated — so the
        /// process is dead, not slow, and the op that reports it is an innocent
        /// bystander. Measured: Qwen3.8-27B-Q8_0 (29.0 GB of weights) on a 48 GB
        /// M5 Pro whose working set is 40.2 GB, served with --max-tokens 256000,
        /// reserved 260,864 KV tokens x 64 KiB = 17.1 GB on the first request and
        /// killed the backend.
        ///
        /// It stays out of <see cref="AppliesTo"/> because the Metal numbers the
        /// tuned defaults there carry (initial KV allocation, prefill chunk width)
        /// were measured against their own fixed caps, and must not start moving
        /// with whatever happens to be free.
        /// </summary>
        internal static bool AppliesToReservations(BackendType backend) =>
            AppliesTo(backend) || backend == BackendType.GgmlMetal;

        /// <summary>
        /// VRAM to keep free for the driver, other processes, and the transient
        /// allocations our own graph rebuilds make. TS_VRAM_HEADROOM_MB overrides.
        /// </summary>
        internal static long ResolveHeadroomBytes(long totalBytes)
        {
            string ov = Environment.GetEnvironmentVariable("TS_VRAM_HEADROOM_MB");
            if (!string.IsNullOrWhiteSpace(ov) && long.TryParse(ov, out long mb) && mb >= 0)
                return mb * 1024 * 1024;
            // ~6% of the card, floored: enough to absorb a desktop compositor plus
            // one browser without the driver starting to evict our weights.
            return Math.Max(MinHeadroomBytes, totalBytes / 16);
        }

        /// <summary>
        /// Device bytes still available for policy-sized buffers once the headroom
        /// reserve is set aside. False when the backend has no device budget, the
        /// query fails — callers then keep their own default untouched.
        /// </summary>
        internal static bool TryGetSpareBytes(BackendType backend, out long spareBytes)
            => TryGetSpareBytes(backend, AppliesTo(backend), out spareBytes);

        /// <summary>
        /// Same as <see cref="TryGetSpareBytes(BackendType, out long)"/>, but for a
        /// buffer a request reserves up front — see <see cref="AppliesToReservations"/>
        /// for why Metal is included here and not there.
        /// </summary>
        internal static bool TryGetReservationSpareBytes(BackendType backend, out long spareBytes)
            => TryGetSpareBytes(backend, AppliesToReservations(backend), out spareBytes);

        /// <summary>
        /// The device's free bytes with no headroom set aside, on the backends <see cref="AppliesToReservations"/> covers: the
        /// most one phase of a request can have, which the diffusion runtime refuses a request past (its residency plan
        /// already keeps the headroom where it can). False when the backend has no device budget or the query fails.
        /// </summary>
        internal static bool TryGetReservationFreeBytes(BackendType backend, out long freeBytes)
        {
            freeBytes = 0;
            if (!AppliesToReservations(backend))
                return false;
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long totalBytes) || totalBytes <= 0)
                return false;
            // Bytes another owner claimed ahead of use (Reserve) are not this request's to take.
            freeBytes = Math.Max(0, free - Math.Max(0, ReservedBytes(backend, GgmlDevice)));
            return true;
        }

        private static bool TryGetSpareBytes(BackendType backend, bool backendApplies, out long spareBytes)
        {
            spareBytes = 0;
            if (!backendApplies)
                return false;
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long freeBytes, out long totalBytes) || totalBytes <= 0)
                return false;
            spareBytes = SpareBytes(freeBytes, totalBytes, ReservedBytes(backend, GgmlDevice));
            return true;
        }

        /// <summary>The spare of a device reading: its free bytes less the headroom and the bytes reserved on it.</summary>
        internal static long SpareBytes(long freeBytes, long totalBytes, long reservedBytes)
            => Math.Max(0, freeBytes - ResolveHeadroomBytes(totalBytes) - Math.Max(0, reservedBytes));

        // ---------------------------------------------------------------- reservations

        /// <summary>The device a GGML backend's readings describe (<see cref="GgmlBasicOps.TryGetDeviceMemoryInfo"/>: the
        /// primary backend's, rank 0 under tensor parallelism), and so the one its reservations are made on.</summary>
        internal const int GgmlDevice = 0;

        private static readonly object s_reservationsLock = new();
        private static readonly List<DeviceReservation> s_reservations = new();

        /// <summary>
        /// Claim <paramref name="bytes"/> of <paramref name="device"/>'s memory for <paramref name="owner"/> before it allocates
        /// them: every spare reading of that device leaves them out (<see cref="TryGetSpareBytes"/>, and the shared decoder's
        /// slot admission and room checks), so caches sized from what is free, a KV cache growing on demand above all, leave
        /// the owner its room. The owner sets <see cref="DeviceReservation.Bytes"/> to what it still needs as it allocates
        /// (memory it holds is no longer free, and stays out of the readings by itself) and disposes the reservation when it
        /// lets the memory go. <paramref name="backend"/> names the device's family: a GGML backend's device is
        /// <see cref="GgmlDevice"/>, direct CUDA's the allocator's ordinal.
        /// </summary>
        internal static DeviceReservation Reserve(BackendType backend, int device, string owner, long bytes)
        {
            ArgumentException.ThrowIfNullOrEmpty(owner);
            var reservation = new DeviceReservation(backend, device, owner, bytes);
            lock (s_reservationsLock) s_reservations.Add(reservation);
            return reservation;
        }

        /// <summary>The bytes reserved on <paramref name="device"/> of <paramref name="backend"/>'s family, by every owner.</summary>
        internal static long ReservedBytes(BackendType backend, int device)
        {
            long sum = 0;
            lock (s_reservationsLock)
                foreach (DeviceReservation r in s_reservations)
                    if (r.Backend == backend && r.Device == device) sum += r.Bytes;
            return sum;
        }

        internal static void Release(DeviceReservation reservation)
        {
            lock (s_reservationsLock) s_reservations.Remove(reservation);
        }

        /// <summary>
        /// Largest token count at or below <paramref name="desiredTokens"/> whose
        /// per-token cost fits <paramref name="budgetBytes"/>, snapped down to a
        /// multiple of <paramref name="granularity"/> and never below
        /// <paramref name="minTokens"/>. Snapping matters as much as the cap does:
        /// a chunk width is also a graph shape, and letting it vary per prompt
        /// turns every prompt length into a cold shape whose graph (and, on
        /// Vulkan, whose pipelines) is rebuilt inside the measured TTFT.
        /// </summary>
        internal static int FitTokens(
            long budgetBytes, long bytesPerToken, int desiredTokens, int minTokens, int granularity)
        {
            if (bytesPerToken <= 0 || desiredTokens <= minTokens)
                return desiredTokens;
            long affordable = budgetBytes / bytesPerToken;
            if (affordable >= desiredTokens)
                return desiredTokens;
            if (granularity > 1)
                affordable -= affordable % granularity;
            if (affordable < minTokens)
                return minTokens;
            return (int)Math.Min(affordable, desiredTokens);
        }
    }

    /// <summary>A claim on a device's memory (<see cref="GpuMemoryBudget.Reserve"/>): the bytes its owner may still allocate
    /// there. Disposing it gives the claim up.</summary>
    internal sealed class DeviceReservation : IDisposable
    {
        private long _bytes;

        internal DeviceReservation(BackendType backend, int device, string owner, long bytes)
        {
            Backend = backend;
            Device = device;
            Owner = owner;
            _bytes = Math.Max(0, bytes);
        }

        public BackendType Backend { get; }

        public int Device { get; }

        public string Owner { get; }

        /// <summary>The bytes still claimed: the owner lowers it as it allocates them (never below 0).</summary>
        public long Bytes
        {
            get => Interlocked.Read(ref _bytes);
            set => Interlocked.Exchange(ref _bytes, Math.Max(0, value));
        }

        public void Dispose() => GpuMemoryBudget.Release(this);
    }
}
