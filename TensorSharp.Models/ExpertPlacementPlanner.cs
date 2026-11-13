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
using System.IO;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Where a MoE model's routed experts run when they do not all fit the accelerator: the budget
    /// arithmetic shared by every family's expert plan (first measured on Qwen 3.8 Flash Next). Whole
    /// trailing layers stay resident, as llama.cpp's <c>--n-cpu-moe</c> keeps them; the rest run at a host
    /// seam, optionally with an accelerator tier of their most-routed experts (ModelBase.ExpertTiers.cs).
    /// </summary>
    internal static class ExpertPlacementPlanner
    {
        /// <summary>RAM the host keeps for itself - the OS, other processes, and
        /// the PLE table's hot pages - before any wired weight is counted.</summary>
        internal const long HostReserveFloorBytes = 8L << 30;

        /// <summary>Share of an offloaded layer's experts the page cache must be able to
        /// hold. A wired layer holds all 512 experts, used or not, and every byte it
        /// wires is a byte the offloaded layers can no longer cache, so past a point
        /// wiring more makes both prefill and decode SLOWER. Measured on a 48 GiB
        /// M5 Pro, Qwen3.8-Flash-Next UD-Q2_K_XL, random-token pp512 / tg128, in one
        /// session: 15 / 16 / 18 / 20 / 22 layers wired gave 147.6 / 153.4 / 183.9 /
        /// 171.9 / 80.7 t/s prefill and 21.1 / 21.2 / 20.6 / 20.3 / 19.0 t/s decode; an
        /// earlier one had 0 / 4 / 8 / 12 wired at 17.5 / 18.3 / 19.0 / 19.4 t/s decode
        /// and 26 wired falling to 11.9. Two thirds is the share that stops that Mac at
        /// 15 layers, within 0.1 t/s of the best decode measured (21.2 with 16).</summary>
        internal const double OffloadedHotFraction = 0.67;

        /// <summary>Graph scratch, KV growth and the driver's own allocations, on top
        /// of the weights, inside the Metal working set.</summary>
        internal const long DeviceScratchBytes = 3L << 30;

        internal static long PendingDFlashDraftBytes(string path)
        {
            try
            {
                return new FileInfo(path).Length + DFlashRingAllowanceBytes;
            }
            catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException || ex is ArgumentException)
            {
                return 0;   // the attach reports the file itself
            }
        }

        /// <summary>One request's DFlash KV ring (Flash Next DFlash2: 2080 rows x 5 layers, 81 MB F32), rounded up.</summary>
        internal const long DFlashRingAllowanceBytes = 128L << 20;

        /// <summary>Concurrent sequences ggml_cuda plans VRAM for (<c>TS_Q4E_LANES</c>, as on the direct CUDA engine):
        /// 1 to 16, 1 when unset or unreadable. Requests beyond what fits wait for a running one either way.</summary>
        internal static int GgmlCudaLanes(string value)
            => int.TryParse(value, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int lanes)
                ? Math.Clamp(lanes, 1, 16) : 1;

        /// <summary>VRAM the expert plan keeps for the sequences beside the first: each a holder of
        /// <paramref name="holderBytes"/> plus <paramref name="extraBytes"/> of recurrent state and arena share.</summary>
        internal static long ConcurrentSequenceReserveBytes(int lanes, long holderBytes, long extraBytes)
        {
            if (lanes < 1 || holderBytes < 0 || extraBytes < 0)
                throw new ArgumentOutOfRangeException(nameof(lanes));
            return checked((lanes - 1) * (holderBytes + extraBytes));
        }

        internal static long ResolveCudaExpertCacheBudget(string value)
            => long.TryParse(value, System.Globalization.NumberStyles.None,
                System.Globalization.CultureInfo.InvariantCulture, out long mb)
                && mb > 0 && mb <= long.MaxValue / (1L << 20)
                ? mb * (1L << 20) : 0;

        internal static ulong ResolveCudaExpertCacheLayers(string value)
            => string.IsNullOrEmpty(value) ? 48
                : ulong.TryParse(value, System.Globalization.NumberStyles.None,
                    System.Globalization.CultureInfo.InvariantCulture, out ulong layers) && layers > 0 ? layers : 0;

        /// <summary>Conservative fit threshold for one native scalar expert-cache graph.
        /// Zero means the native cache cannot serve this layout. The native helper
        /// still measures its actual allocator size and validates kernel support.</summary>
        internal static long MinimumCudaExpertCacheBytes(StackedExpertWeights gate, StackedExpertWeights up,
            StackedExpertWeights down, int hidden, int ff, int used, int experts)
            => MinimumCudaExpertCacheBytes(gate, up, down, hidden, ff, used, experts, out _);

        internal static long MinimumCudaExpertCacheBytes(StackedExpertWeights gate, StackedExpertWeights up,
            StackedExpertWeights down, int hidden, int ff, int used, int experts, out long allocationFloor)
        {
            allocationFloor = 0;
            if (hidden <= 0 || ff <= 0 || used <= 0 || used > 64 || used > experts)
                return 0;
            var weights = new[] { gate, up, down };
            var widths = new[] { hidden, hidden, ff };
            var rows = new[] { ff, ff, hidden };
            try
            {
                long selectedBytes = 0, tails = 0;
                for (int i = 0; i < weights.Length; i++)
                {
                    var weight = weights[i];
                    if (weight == null || weight.Data == IntPtr.Zero || weight.NumExperts != experts
                        || weight.PerExpertNe0 != widths[i] || weight.PerExpertNe1 != rows[i]
                        || !Enum.IsDefined(typeof(GgmlTensorType), (GgmlTensorType)weight.GgmlType)
                        || weight.GgmlType >= 128)
                        return 0;
                    var type = (GgmlTensorType)weight.GgmlType;
                    long block = GgufFile.GetBlockSize(type), size = GgufFile.GetTypeSize(type);
                    if (block <= 1 || widths[i] % block != 0) return 0;
                    long perExpert = checked((long)widths[i] / block * size * rows[i]);
                    if (checked(perExpert * experts) != weight.TotalRawBytes) return 0;
                    selectedBytes = checked(selectedBytes + perExpert * used);
                    // Upstream CUDA pads the end of quantized matrices to 512
                    // elements, then aligns every graph allocation to 128 bytes.
                    long padding = (512 - widths[i] % 512) % 512;
                    tails = checked(tails + padding / block * size);
                }
                // Count all graph tensors without assuming any allocator reuse:
                // gate/up/SiLU/product, down/weighted/add-chain, input/output,
                // I32 IDs and F32 routes. Reserve alignment for all 256 nodes.
                long scalars = checked(4L * ff * used + (3L * used + 1) * hidden + 2L * used);
                allocationFloor = checked((1L << 20) + selectedBytes);
                return checked(allocationFloor + tails + scalars * sizeof(float) + 256L * 128);
            }
            catch (OverflowException) { allocationFloor = 0; return 0; }
        }

        /// <summary>Trailing whole expert layers that fit CUDA's remaining memory.
        /// When an opt-in expert cache can serve every layer and the complete
        /// model does not fit, all layers use the seam. Insufficient or unsupported
        /// cache layouts preserve whole resident layers, reserving cache quota only
        /// for offloaded layers whose selected experts can fit it.
        /// Models that fit keep the uninterrupted all-device graph.</summary>
        internal static int PlanCudaDeviceExpertLayers(long[] layerExpertBytes, long pendingBytes,
            long freeBytes, long headroomBytes, long expertCacheBytes,
            long[] layerCacheMinimumBytes = null, ulong cacheLayers = 48,
            long[] layerCacheAllocationFloorBytes = null)
        {
            ArgumentNullException.ThrowIfNull(layerExpertBytes);
            if (pendingBytes < 0 || headroomBytes < 0 || expertCacheBytes < 0)
                throw new ArgumentOutOfRangeException(nameof(pendingBytes));
            if (layerCacheMinimumBytes != null && layerCacheMinimumBytes.Length != layerExpertBytes.Length)
                throw new ArgumentException("Cache layouts must match the expert layer count.", nameof(layerCacheMinimumBytes));
            if (layerCacheAllocationFloorBytes != null && (layerCacheMinimumBytes == null
                || layerCacheAllocationFloorBytes.Length != layerExpertBytes.Length))
                throw new ArgumentException("Cache allocation floors must match the cache layouts.", nameof(layerCacheAllocationFloorBytes));
            foreach (long bytes in layerExpertBytes)
                if (bytes < 0) throw new ArgumentOutOfRangeException(nameof(layerExpertBytes));
            if (layerCacheMinimumBytes != null)
                foreach (long bytes in layerCacheMinimumBytes)
                    if (bytes < 0) throw new ArgumentOutOfRangeException(nameof(layerCacheMinimumBytes));
            if (layerCacheAllocationFloorBytes != null)
                for (int l = 0; l < layerCacheAllocationFloorBytes.Length; l++)
                    if (layerCacheAllocationFloorBytes[l] < 0 || layerCacheAllocationFloorBytes[l] > layerCacheMinimumBytes[l])
                        throw new ArgumentOutOfRangeException(nameof(layerCacheAllocationFloorBytes));
            long available = Math.Max(0, freeBytes);
            foreach (long reserve in new[] { pendingBytes, headroomBytes, DeviceScratchBytes })
                available = reserve >= available ? 0 : available - reserve;
            int Fit(long budget)
            {
                int resident = 0;
                for (int l = layerExpertBytes.Length - 1; l >= 0; l--)
                {
                    long bytes = layerExpertBytes[l];
                    if (bytes > budget) break;
                    budget -= bytes;
                    resident++;
                }
                return resident;
            }
            int count = Fit(available);
            if (count == layerExpertBytes.Length || expertCacheBytes == 0 || cacheLayers == 0 || layerCacheMinimumBytes == null)
                return count;
            long quota = (long)((ulong)expertCacheBytes / cacheLayers);
            bool CacheFits(int layer) => layerCacheMinimumBytes[layer] > 0 && layerCacheMinimumBytes[layer] <= quota;
            bool CacheMayAllocate(int layer)
            {
                long floor = layerCacheAllocationFloorBytes == null ? layerCacheMinimumBytes[layer] : layerCacheAllocationFloorBytes[layer];
                return floor > 0 && floor <= quota;
            }
            bool all = cacheLayers >= (ulong)layerExpertBytes.Length;
            for (int l = 0; l < layerExpertBytes.Length; l++) all &= CacheFits(l);
            if (all) return 0;
            // A heterogeneous model can cache only some host layers. Charge their
            // quotas before retaining whole GPU layers, and repeat when that charge
            // moves another eligible layer to the host. Even a quota below our
            // conservative threshold can fit the native allocator with reuse;
            // charge any layer above the necessary payload+workspace floor.
            // This prevents both tiers spending the same remaining device bytes.
            for (;;)
            {
                int eligible = 0;
                for (int l = 0; l < layerExpertBytes.Length - count; l++)
                    if (CacheMayAllocate(l)) eligible++;
                long reserve = quota == 0 || eligible == 0 ? 0
                    : eligible > expertCacheBytes / quota ? expertCacheBytes : quota * eligible;
                int next = Fit(reserve >= available ? 0 : available - reserve);
                if (next == count) return count;
                count = next;
            }
        }

        /// <summary>How many trailing layers' experts the accelerator holds. A layer
        /// joins only when its experts fit the working set left after every other
        /// bound weight and the scratch, AND the RAM left after wiring still covers
        /// the host reserve plus the hot share of every layer that stays offloaded.</summary>
        internal static int PlanDeviceExpertLayers(long[] layerExpertBytes, long otherDeviceBytes,
            long workingSetBytes, long ramBytes)
        {
            int n = layerExpertBytes.Length;
            long hostReserve = Math.Max(HostReserveFloorBytes, ramBytes / 6);
            long wired = otherDeviceBytes + DeviceScratchBytes;
            long offloadedHot = 0;
            for (int l = 0; l < n; l++)
                offloadedHot += (long)(layerExpertBytes[l] * OffloadedHotFraction);

            int onDevice = 0;
            // Walk from the LAST layer down: those are the ones that stay resident.
            for (int l = n - 1; l >= 0; l--)
            {
                long nextWired = wired + layerExpertBytes[l];
                long nextHot = offloadedHot - (long)(layerExpertBytes[l] * OffloadedHotFraction);
                if (nextWired > workingSetBytes - Math.Max(GpuMemoryBudget.MinHeadroomBytes, workingSetBytes / 16))
                    break;
                if (ramBytes - nextWired < hostReserve + nextHot)
                    break;
                wired = nextWired;
                offloadedHot = nextHot;
                onDevice++;
            }
            return onDevice;
        }
    }
}
