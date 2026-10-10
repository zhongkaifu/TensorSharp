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
                ReportExpertPlacement("--n-cpu-moe / --cpu-moe", null);
                return;
            }

            // CUDA has a separate VRAM pool. Quantized non-expert weights were
            // preloaded above, but the span still has to bind float weights and
            // caches. Plan against current free memory rather than total VRAM.
            // A layer split needs a per-device plan and is left to explicit flags.
            if (_backend == BackendType.GgmlCuda && !IsTensorParallel && LayerSplitDegree <= 1)
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
                long cacheBudget = ResolveCudaExpertCacheBudget(
                    Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_MB"));
                ulong cacheLayers = ResolveCudaExpertCacheLayers(
                    Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_LAYERS"));
                int resident = PlanCudaDeviceExpertLayers(layerBytes, pending, free,
                    GpuMemoryBudget.ResolveHeadroomBytes(total), cacheBudget, cacheMinimum, cacheLayers, cacheAllocationFloor);
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
            long other = AcceleratorBoundNonExpertBytes() + CacheBytes();

            int onDevice = PlanDeviceExpertLayers(experts, other, workingSet, ram);
            if (onDevice >= n)
                return;
            // Offload the FIRST layers, keep the last ones resident (llama.cpp's
            // --n-cpu-moe order): a token then ends on the accelerator without a
            // final host round trip.
            for (int l = 0; l < n - onDevice; l++)
                _expertOnHost[l] = true;
            ReportExpertPlacement("planned", $"Metal working set {Gb(workingSet)}, RAM {Gb(ram)}; " +
                "--n-cpu-moe N overrides");
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
            return TensorSharp.Memory.TieredPlacementPlanner.PlanDiscreteResidency(
                layerExpertBytes, pendingBytes, freeBytes, headroomBytes, DeviceScratchBytes,
                expertCacheBytes, layerCacheMinimumBytes, cacheLayers, layerCacheAllocationFloorBytes);
        }

        /// <summary>How many trailing layers' experts the accelerator holds. A layer
        /// joins only when its experts fit the working set left after every other
        /// bound weight and the scratch, AND the RAM left after wiring still covers
        /// the host reserve plus the hot share of every layer that stays offloaded.</summary>
        internal static int PlanDeviceExpertLayers(long[] layerExpertBytes, long otherDeviceBytes,
            long workingSetBytes, long ramBytes)
        {
            return TensorSharp.Memory.TieredPlacementPlanner.PlanUnifiedResidency(
                layerExpertBytes, otherDeviceBytes, workingSetBytes, ramBytes,
                Math.Max(HostReserveFloorBytes, ramBytes / 6),
                Math.Max(GpuMemoryBudget.MinHeadroomBytes, workingSetBytes / 16),
                DeviceScratchBytes, OffloadedHotFraction);
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
