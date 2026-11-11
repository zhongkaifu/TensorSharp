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
using static TensorSharp.Models.ExpertPlacementPlanner;

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


        /// <summary>What a DFlash drafter named for this load takes once attached (its weights and the KV ring of the
        /// first request), counted by the Metal plan like the MTP head: the drafter is attached last, after the experts
        /// took the working set. Unplanned, the 0.45 GB of Flash Next's DFlash2 Q4_K_M was what a 4096-token prompt
        /// chunk lacked whenever a quiet machine at load sized the tier past ~25.5 GB of a 48 GB M5 Pro: its 2.84 GB
        /// arena found 3.87-4.0 GB free of the 4.1 GB it needs beside the kept headroom, every 4K chunk ran in halves,
        /// and 16K-token prompts took a second longer.</summary>
        private long _pendingDraftBytes;


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
                AttachRoutingProfile(ArchitectureId, _fusedGateUpExperts ? 0 : _numExperts);
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
