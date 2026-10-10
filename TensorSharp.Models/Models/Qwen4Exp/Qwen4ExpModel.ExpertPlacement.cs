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
            // ggml_cuda plans against what each GPU has free, and sub-chunks prefill to
            // the span width that plan reserved scratch for.
            bool cudaPlans = _backend == BackendType.GgmlCuda && !IsTensorParallel;

            if (MoeCpuOffloadConfig.IsExplicitlySet)
            {
                if (!MoeCpuOffloadConfig.IsEnabled)
                {
                    // An explicit "no offload" under a split is still checked per GPU:
                    // failing here names the fix, failing in warmup does not.
                    if (cudaPlans && LayerSplitDegree > 1)
                        PlanCudaSplitExpertPlacement();
                    else if (cudaPlans)
                    {
                        ApplySingleDeviceSpanCap(anyHost: false);
                        CapSingleDeviceContext();
                    }
                    return;
                }
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
                if (cudaPlans && LayerSplitDegree > 1)
                {
                    PlanCudaSplitExpertPlacement();
                    return;
                }
                bool anyHost = false;
                for (int l = 0; l < n; l++)
                    anyHost |= _expertOnHost[l] = MoeCpuOffloadConfig.IsLayerOnCpu(l);
                if (cudaPlans)
                    ApplySingleDeviceSpanCap(anyHost);
                ReportExpertPlacement("--n-cpu-moe / --cpu-moe", null);
                if (cudaPlans)
                    CapSingleDeviceContext();
                return;
            }

            // A layer split was sized against each GPU before the preload; now re-fit
            // each GPU's host set against what it really has free.
            if (cudaPlans && LayerSplitDegree > 1)
            {
                PlanCudaSplitExpertPlacement();
                return;
            }

            // CUDA has a separate VRAM pool. Quantized non-expert weights were
            // preloaded above, but the span still has to bind float weights and
            // caches. Plan against current free memory rather than total VRAM.
            if (cudaPlans)
            {
                if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) || total <= 0)
                {
                    ApplySingleDeviceSpanCap(anyHost: false);
                    return;
                }
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
                // again. The optional MTP head is preloaded AFTER this plan, the
                // recurrent state is created by the first forward, and the vision
                // tower (when the host said it will load one) after construction.
                int kvElement = _kvCacheDtype == KvCacheDtype.F32 ? 4 : 2;
                long pending = checked((_mtpReady ? 0 : _mtpResidentBytes) + _visionReserveBytes);
                for (int l = 0; l < n; l++) pending = checked(pending + AllocatedLayerCacheBytes(l, kvElement));
                foreach (var weight in _weights.Values) pending = checked(pending + weight.Storage.ByteLength);
                long cacheBudget = ResolveCudaExpertCacheBudget(
                    Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_MB"));
                ulong cacheLayers = ResolveCudaExpertCacheLayers(
                    Environment.GetEnvironmentVariable("TS_HOST_MOE_EXPERT_CACHE_LAYERS"));
                long headroom = GpuMemoryBudget.ResolveHeadroomBytes(total);
                // Plan at the full span width; once anything has to go to the host,
                // plan again at the narrower width that keeps more layers resident.
                int? pinned = ParseSpanTokenOverride(Environment.GetEnvironmentVariable("TS_Q4E_PREFILL_CHUNK"));
                int tokens = pinned ?? SpanTokensWhenResident;
                int resident = PlanCudaDeviceExpertLayers(layerBytes, pending, free, headroom, cacheBudget,
                    cacheMinimum, cacheLayers, cacheAllocationFloor, SpanScratch(tokens, _maxContextLength, kvElement));
                if (pinned == null && resident < n)
                {
                    tokens = SpanTokensWhenOffloaded;
                    resident = PlanCudaDeviceExpertLayers(layerBytes, pending, free, headroom, cacheBudget,
                        cacheMinimum, cacheLayers, cacheAllocationFloor, SpanScratch(tokens, _maxContextLength, kvElement));
                }
                _spanTokenCapPinned = pinned != null;
                SetSpanTokenCap(tokens, _maxContextLength);
                long largestHost = 0, residentExperts = 0;
                for (int l = 0; l < n; l++)
                {
                    if (l < n - resident) { _expertOnHost[l] = true; largestHost = Math.Max(largestHost, layerBytes[l]); }
                    else residentExperts += layerBytes[l];
                }
                long scratch = SpanScratch(tokens, _maxContextLength, kvElement).Bytes(n - resident, largestHost);
                ReportExpertPlacement("planned", $"CUDA free {Gb(free)}, reserved span scratch {Gb(scratch)} " +
                    $"for prefill spans of up to {tokens} tokens" +
                    (cacheBudget > 0 ? $", expert cache ceiling {Gb(cacheBudget)}" : "") +
                    "; --n-cpu-moe N and TS_Q4E_PREFILL_CHUNK override");
                // Room the expert cache may claim is not room the KV cache can grow into.
                long used = checked(pending + residentExperts + scratch + Math.Min(cacheBudget, Math.Max(0, free)));
                CapContextToDeviceRoom(new[] { Math.Max(0, free - headroom) }, new[] { used },
                    _kvCacheDtype == KvCacheDtype.F32 ? 4 : 2);
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

        /// <summary>Span width for a single-GPU ggml_cuda run whose host set was not
        /// planned here (explicit flags, or no memory query).</summary>
        private void ApplySingleDeviceSpanCap(bool anyHost)
        {
            int? pinned = ParseSpanTokenOverride(Environment.GetEnvironmentVariable("TS_Q4E_PREFILL_CHUNK"));
            _spanTokenCapPinned = pinned != null;
            SetSpanTokenCap(pinned ?? (anyHost ? SpanTokensWhenOffloaded : SpanTokensWhenResident), _maxContextLength);
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
        /// <param name="spanScratch">Scratch a span needs, by how many layers route
        /// their experts to the host. Null keeps the flat <see cref="DeviceScratchBytes"/>
        /// reserve.</param>
        internal static int PlanCudaDeviceExpertLayers(long[] layerExpertBytes, long pendingBytes,
            long freeBytes, long headroomBytes, long expertCacheBytes,
            long[] layerCacheMinimumBytes = null, ulong cacheLayers = 48,
            long[] layerCacheAllocationFloorBytes = null, Qwen4ExpSpanScratch? spanScratch = null)
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
            foreach (long reserve in spanScratch == null
                         ? new[] { pendingBytes, headroomBytes, DeviceScratchBytes }
                         : new[] { pendingBytes, headroomBytes })
                available = reserve >= available ? 0 : available - reserve;
            int Fit(long budget)
            {
                if (spanScratch is Qwen4ExpSpanScratch scratch)
                {
                    // The scratch grows with every layer routed to the host (seam
                    // tensors, and the stream buffer the first one adds), so each split
                    // is checked as a whole. All-host is the floor whether it fits or not.
                    int n = layerExpertBytes.Length;
                    long residentBytes = 0;
                    foreach (long b in layerExpertBytes) residentBytes = checked(residentBytes + b);
                    long largestHost = 0;
                    for (int split = 0; split <= n; split++)
                    {
                        if (split > 0)
                        {
                            residentBytes -= layerExpertBytes[split - 1];
                            largestHost = Math.Max(largestHost, layerExpertBytes[split - 1]);
                        }
                        if (checked(residentBytes + scratch.Bytes(split, largestHost)) <= budget)
                            return n - split;
                    }
                    return 0;
                }
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
            _placementSummary = $"the GPU holds the experts of {n - onHost} of {n} layers ({Gb(deviceBytes)}), "
                + $"{onHost} run on the host";
            Console.WriteLine($"[moe-offload] qwen4exp ({how}): routed experts of {onHost} of {n} layers run on the " +
                $"host from the GGUF mapping ({Gb(hostBytes)} read on demand); the accelerator holds {n - onHost} " +
                $"layers' ({Gb(deviceBytes)})" + (detail != null ? $". {detail}." : "."));
        }

        private static string Gb(long bytes) => $"{bytes / 1e9:F1} GB";
    }
}
