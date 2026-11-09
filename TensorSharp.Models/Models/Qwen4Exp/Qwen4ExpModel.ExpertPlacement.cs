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
    /// Where each layer's routed experts run: on the accelerator, or on the host
    /// straight from the GGUF mapping (MoE CPU offload).
    ///
    /// <para>The decode span binds every layer's stacked expert tensors. On
    /// ggml-metal each bound weight is one no-copy MTLBuffer over the mmap, and a
    /// command buffer that uses it needs ALL of it resident - every one of the 512
    /// experts, not the 10 a token routes to. Qwen3.8-Flash-Next UD-Q2_K_XL has
    /// 46 GB of experts; a 48 GiB Mac's Metal working set is 40.2 GB, so the
    /// all-accelerator span fails with kIOGPUCommandBufferCallbackErrorOutOfMemory.
    /// An offloaded layer's experts are never wrapped: the host reads only the
    /// rows its tokens route to, and the OS page cache keeps the hot ones.</para>
    ///
    /// <para>Explicit <c>--n-cpu-moe N</c> / <c>--cpu-moe</c> always win. Without
    /// them, a Metal or single-device CUDA run whose experts do not fit is
    /// planned here. Metal also reserves RAM for the host's hot expert pages.
    /// CUDA can serve offloaded layers through the opt-in selected-expert cache.
    /// Whole resident layers are kept at the end, as llama.cpp's
    /// <c>--n-cpu-moe</c> does.</para>
    /// </summary>
    public partial class Qwen4ExpModel
    {
        /// <summary>Per layer: true when its routed experts run on the host.
        /// Null until <see cref="PlanExpertPlacement"/> runs; all false when nothing
        /// is offloaded.</summary>
        private bool[] _expertOnHost;

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

        /// <summary>What a DFlash drafter named for this load takes once attached (its weights and the KV ring of the
        /// first request), counted by the Metal plan like the MTP head: the drafter is attached last, after the experts
        /// took the working set. Unplanned, the 0.45 GB of Flash Next's DFlash2 Q4_K_M was what a 4096-token prompt
        /// chunk lacked whenever a quiet machine at load sized the tier past ~25.5 GB of a 48 GB M5 Pro: its 2.84 GB
        /// arena found 3.87-4.0 GB free of the 4.1 GB it needs beside the kept headroom, every 4K chunk ran in halves,
        /// and 16K-token prompts took a second longer.</summary>
        private long _pendingDraftBytes;

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

        /// <summary>
        /// With host-served experts (offloaded or tiered layers), the 2048-token kernel warm-up
        /// routes to nearly every expert of every layer and streams them all: about 46 GB of
        /// Flash Next's experts read before the first request, minutes on network storage (a
        /// geesefs mount: over 75 s and still reading). 64 rows stay on the tier path
        /// (<c>kHostMoeHotTierMaxRows</c>), whose graphs short prompts use; a long prompt then
        /// streams only the experts it routes.
        /// </summary>
        protected override int GgmlPrefillChunkWarmupLength
            => _expertOnHost != null && Array.IndexOf(_expertOnHost, true) >= 0 ? 64 : 0;

        private bool IsExpertLayerOnHost(int layer)
            => _expertOnHost != null && layer >= 0 && layer < _expertOnHost.Length && _expertOnHost[layer];

        /// <summary>Decide <see cref="_expertOnHost"/>. Runs once, after the weights
        /// and caches exist and before the first forward builds a span.</summary>
        private void PlanExpertPlacement()
        {
            int n = Config.NumLayers;
            _expertOnHost = new bool[n];

            if (MoeCpuOffloadConfig.IsExplicitlySet)
            {
                if (!MoeCpuOffloadConfig.IsEnabled)
                    return;
                if (!IsGgmlBackend || IsTensorParallel)
                {
                    MoeCpuOffloadConfig.WarnUnsupportedBackend(ArchitectureId,
                        IsTensorParallel ? _backend + " with tensor parallelism" : _backend.ToString());
                    return;
                }
                if (_backend == BackendType.GgmlCpu)
                {
                    // Every expert already runs on the host; a seam would only add syncs.
                    Console.WriteLine("[moe-offload] qwen4exp: ggml_cpu already runs every expert on the host; " +
                        "--n-cpu-moe / --cpu-moe change nothing.");
                    return;
                }
                for (int l = 0; l < n; l++)
                    _expertOnHost[l] = MoeCpuOffloadConfig.IsLayerOnCpu(l);
                // Explicit offload still records routing, so a --cpu-moe run is also
                // how a complete profile is collected for the tiered plan.
                AttachRoutingProfile();
                ReportExpertPlacement("--n-cpu-moe / --cpu-moe", null);
                if (_backend == BackendType.GgmlMetal)
                {
                    long resident = 0;
                    for (int l = 0; l < n; l++)
                        if (!_expertOnHost[l]) resident += LayerExpertBytes(l);
                    ConfigureHostExpertIo(HostExpertIoBudget(resident));
                }
                return;
            }

            // CUDA has a separate VRAM pool. Quantized non-expert weights were
            // preloaded above, but the span still has to bind float weights and
            // caches. Plan against current free memory rather than total VRAM.
            // A layer split needs a per-device plan and is left to explicit flags.
            if (_backend == BackendType.GgmlCuda && !IsTensorParallel && LayerSplitDegree <= 1 && _q4eStage == null)
            {
                if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) || total <= 0)
                    return;
                var layerBytes = new long[n];
                var cacheMinimum = new long[n];
                var cacheAllocationFloor = new long[n];
                for (int l = 0; l < n; l++)
                {
                    layerBytes[l] = LayerExpertBytes(l);
                    _stackedExpertWeights.TryGetValue($"blk.{l}.ffn_gate_exps.weight", out var gate);
                    _stackedExpertWeights.TryGetValue($"blk.{l}.ffn_up_exps.weight", out var up);
                    _stackedExpertWeights.TryGetValue($"blk.{l}.ffn_down_exps.weight", out var down);
                    if (!_fusedGateUpExperts)
                        cacheMinimum[l] = MinimumCudaExpertCacheBytes(gate, up, down,
                            Config.HiddenSize, _expertFf, _numExpertsUsed, _numExperts, out cacheAllocationFloor[l]);
                }
                // These tensors were filled directly on the host; their first
                // device bindings remain pending. Dense quantized preloads are
                // already reflected in measured free VRAM and are not counted
                // again. The optional MTP head is preloaded AFTER this plan.
                long pending = checked(CacheBytes() + (_mtpReady ? 0 : _mtpResidentBytes));
                foreach (var weight in _weights.Values) pending = checked(pending + weight.Storage.ByteLength);
                // Concurrent requests' holders come out of the same VRAM later: without a reserve the tier
                // (19.1 GB on the RTX 5090) left ~4.6 GB, and four requests' state plus the batched arena did not fit.
                int lanes = GgmlCudaLanes(Environment.GetEnvironmentVariable("TS_Q4E_LANES"));
                long laneReserve = ConcurrentSequenceReserveBytes(lanes,
                    HolderCacheBytes(Math.Min(_maxContextLength, CudaSequenceGrowthTokens)), MetalSequenceExtraBytes);
                pending = checked(pending + laneReserve);
                if (laneReserve > 0)
                    Console.WriteLine($"[qwen4exp] {lanes} concurrent sequences on ggml_cuda: {Gb(laneReserve)} of VRAM kept beside the " +
                        $"experts for the {lanes - 1} beside the first ({CudaSequenceGrowthTokens} K/V rows and their state each; TS_Q4E_LANES).");
                long cacheBudget = ResolveCudaExpertCacheBudget(
                    Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_MB"));
                ulong cacheLayers = ResolveCudaExpertCacheLayers(
                    Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_LAYERS"));
                int resident = PlanCudaDeviceExpertLayers(layerBytes, pending, free,
                    GpuMemoryBudget.ResolveHeadroomBytes(total), cacheBudget, cacheMinimum, cacheLayers, cacheAllocationFloor);
                if (resident < n && cacheBudget == 0)
                {
                    // NInfer's RTX 5090 Flash Next layout: the VRAM the whole-layer plan
                    // would spend holds every layer's most-routed experts instead, and
                    // the host computes the rest. Needs a routing profile.
                    long vram = 0;
                    for (int l = n - resident; l < n; l++) vram += layerBytes[l];
                    if (TryPlanExpertTiers(vram))
                        return;
                }
                for (int l = 0; l < n - resident; l++) _expertOnHost[l] = true;
                ReportExpertPlacement("planned", $"CUDA free {Gb(free)}, reserved scratch {Gb(DeviceScratchBytes)}" +
                    (cacheBudget > 0 ? $", expert cache ceiling {Gb(cacheBudget)}" : "") +
                    "; --n-cpu-moe N overrides");
                return;
            }

            if (_backend != BackendType.GgmlMetal || IsTensorParallel)
                return;
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out _, out long workingSet) || workingSet <= 0)
                return;
            long ram = (long)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            if (ram <= 0)
                return;

            long[] experts = new long[n];
            for (int l = 0; l < n; l++)
                experts[l] = LayerExpertBytes(l);
            // The MTP head and a DFlash drafter are loaded after this plan, into the same working set.
            long other = AcceleratorBoundNonExpertBytes() + CacheBytes() + (_mtpReady ? 0 : _mtpResidentBytes) + _pendingDraftBytes;

            long deviceExpertBytes = workingSet - Math.Max(GpuMemoryBudget.MinHeadroomBytes, workingSet / 16)
                - other - DeviceScratchBytes;
            // The working set is Metal's ceiling, not what this Mac can give: the expert tier is
            // wired while the GPU runs, and macOS keeps a floor of file cache besides other
            // programs' memory. ISTA Q2_0 on a 48 GB Mac planned 29.7 GB of tier against a 40.2 GB
            // working set and Metal ran out of memory on the first long prompt (an unrecoverable
            // command-buffer failure). So, as Strata sizes its arena, the experts also stay within
            // what macOS can give without compressing anything, less max(7 GB, 15% of the RAM)
            // kept for macOS, the scratch, and the MTP head still to come.
            // That snapshot depends on the moment (just after another model exits most of the RAM
            // is free), so the whole wired set - experts, the other accelerator buffers and the
            // scratch - also stays within 65% of the RAM: on the 48 GB Mac with an editor and a
            // browser open, ~32 GB wired ran clean and 38.6 GB (the MTP head beside a 26.9 GB tier)
            // ran Metal out of memory. Strata leaves 30-40% of a Mac's RAM likewise.
            long available = GgmlBasicOps.HostAvailableMemoryBytes();
            long keep = Math.Max(7_000_000_000L, ram / 100 * 15);
            long fits = Math.Min(ram / 100 * 65 - other - DeviceScratchBytes,
                available > 0 ? available - keep - DeviceScratchBytes - (_mtpReady ? 0 : _mtpResidentBytes) - _pendingDraftBytes
                    : long.MaxValue);
            if (fits < deviceExpertBytes)
                Console.WriteLine($"[moe-offload] qwen4exp: experts on the accelerator stay within {Gb(Math.Max(0, fits))}: " +
                    $"65% of the RAM wired at most, and macOS can give {(available > 0 ? Gb(available) : "?")} without " +
                    $"compressing and keeps {Gb(keep)} (Metal working set {Gb(workingSet)}).");
            deviceExpertBytes = Math.Min(deviceExpertBytes, Math.Max(0, fits));

            int onDevice = PlanDeviceExpertLayers(experts, other, workingSet, ram);
            long layerPlanBytes = 0;
            for (int l = n - onDevice; l < n; l++) layerPlanBytes += experts[l];
            // The whole layers kept resident stay under the same cap as a tier.
            while (onDevice > 0 && layerPlanBytes > deviceExpertBytes)
                layerPlanBytes -= experts[n - onDevice--];
            if (onDevice >= n)
                return;
            // A tier puts a host seam in every MoE layer; whole layers need one only where they are
            // offloaded. On Metal a seam is a command buffer submitted and waited for (~0.17 ms) and the
            // host computes the misses, so when the whole-layer plan keeps nearly every layer resident it
            // wins: ISTA Q2_0 on the 48 GB Mac, 42 of 48 layers resident, decoded 25.0 / 19.8 / 24.8 tok/s
            // against 22.6 / 15.9 / 23.2 tiered (code / prose / list). With most layers offloaded (UD-Q2_K_XL
            // there) the tier wins. TS_MOE_HOT_TIER=1 keeps the tier anyway, =0 the whole layers.
            bool preferLayers = Environment.GetEnvironmentVariable("TS_MOE_HOT_TIER") is not "1"
                && (long)onDevice * 100 >= (long)n * 85;
            if (!preferLayers && TryPlanExpertTiers(layerPlanBytes, deviceExpertBytes))
                return;
            // Offload the FIRST layers, keep the last ones resident (llama.cpp's
            // --n-cpu-moe order): a token then ends on the accelerator without a
            // final host round trip.
            for (int l = 0; l < n - onDevice; l++)
                _expertOnHost[l] = true;
            ReportExpertPlacement("planned", $"Metal working set {Gb(workingSet)}, RAM {Gb(ram)}; " +
                "--n-cpu-moe N overrides");
            ConfigureHostExpertIo(HostExpertIoBudget(layerPlanBytes));
        }

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

        private static long MinimumCudaExpertCacheBytes(StackedExpertWeights gate, StackedExpertWeights up,
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

        private long LayerExpertBytes(int layer)
        {
            long bytes = 0;
            foreach (string part in new[] { "gate", "up", "down", "gate_up" })
                if (_stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_{part}_exps.weight", out var w))
                    bytes += w.TotalRawBytes;
            return bytes;
        }

        /// <summary>Bytes of every weight a span binds besides the routed experts:
        /// projections, norms, routers, shared experts, embeddings and the head. The
        /// PLE table is excluded - it is gathered on the host and never bound.</summary>
        private long AcceleratorBoundNonExpertBytes()
        {
            long bytes = 0;
            foreach (var kv in _quantWeights)
            {
                if (_stackedExpertMemberNames.Contains(kv.Key)
                    || MoeCpuOffloadConfig.IsRoutedExpertWeightName(kv.Key)
                    || string.Equals(kv.Key, "per_layer_token_embd.weight", StringComparison.Ordinal))
                    continue;
                bytes += kv.Value.RawBytes;
                // TS_Q4E_HC_Q8=1: the Q8_0 copy of a BF16 hyper-connection matrix is resident besides the file's
                // pages (without this a 16K-token prompt ran the 48 GB Mac's Metal out of memory).
                if (HcQ8 && kv.Value.GgmlType == WeightRequant.GgmlTypeBf16 && IsHcMatrix(kv.Key) && kv.Value.Ne0 % 32 == 0)
                    bytes += WeightRequant.Q8_0Bytes(kv.Value.Ne1, kv.Value.Ne0);
            }
            foreach (var kv in _weights)
                bytes += kv.Value.Storage.ByteLength;
            return bytes;
        }

        /// <summary>The KV and QSA key caches as allocated now; each has a device copy.</summary>
        private long CacheBytes()
        {
            long bytes = 0;
            foreach (Tensor[] caches in new[] { _kCache, _vCache, _idxKCache })
                if (caches != null)
                    foreach (Tensor t in caches)
                        if (t != null)
                            bytes += t.Storage.ByteLength;
            return bytes;
        }

        private void ReportExpertPlacement(string how, string detail)
        {
            int n = _expertOnHost.Length, onHost = 0;
            long hostBytes = 0, deviceBytes = 0;
            for (int l = 0; l < n; l++)
            {
                long b = LayerExpertBytes(l);
                if (_expertOnHost[l]) { onHost++; hostBytes += b; }
                else deviceBytes += b;
            }
            if (onHost == 0)
                return;
            Console.WriteLine($"[moe-offload] qwen4exp ({how}): routed experts of {onHost} of {n} layers run on the " +
                $"host from the GGUF mapping ({Gb(hostBytes)} read on demand); the accelerator holds {n - onHost} " +
                $"layers' ({Gb(deviceBytes)})" + (detail != null ? $". {detail}." : "."));
        }

        private static string Gb(long bytes) => $"{bytes / 1e9:F1} GB";
    }
}
