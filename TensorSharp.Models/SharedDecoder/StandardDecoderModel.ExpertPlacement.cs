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
using TensorSharp.GGML;
using TensorSharp.Runtime;
using static TensorSharp.Models.ExpertPlacementPlanner;

namespace TensorSharp.Models
{
    /// <summary>
    /// Where each layer's routed experts run on one GGML device whose memory cannot hold them all: whole trailing
    /// layers stay on the accelerator and the first ones run at a host seam (<see cref="GraphOp.MoeHostExperts"/>),
    /// or - with a routing profile from earlier loads - every layer runs at a seam and keeps its most-routed experts
    /// in an accelerator tier (ModelBase.ExpertTiers.cs). The budgets are <see cref="ExpertPlacementPlanner"/>'s,
    /// measured on Qwen 3.8 Flash Next. Explicit <c>--n-cpu-moe N</c> / <c>--cpu-moe</c> always win; a model that
    /// fits keeps every expert on the accelerator.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Per layer: its routed experts run at a host seam. Null: the explicit flags decide.</summary>
        private bool[] _expertOnHost;

        /// <summary>What a draft model named for this load puts on the accelerator once attached after the plan (a
        /// DFlash drafter's weights and ring). An MTP head loads with the trunk and is counted with it.</summary>
        private protected long _pendingDraftBytes;

        private protected override bool IsExpertLayerOffloaded(int layer)
            => _expertOnHost == null ? MoeCpuOffloadConfig.IsLayerOnCpu(layer)
                : layer >= 0 && layer < _expertOnHost.Length && _expertOnHost[layer];

        /// <summary>A weight the host reads and the accelerator never binds (Flash Next's PLE table).</summary>
        protected virtual bool IsHostOnlyWeight(string name) => false;

        /// <summary>Decide <see cref="_expertOnHost"/> once the weights and caches exist, before the first forward.</summary>
        private void PlanExpertPlacement()
        {
            if (!_anyMoe || !IsGgmlBackend || _mesh.TensorParallel > 1 || _mesh.Stages > 1 || LayerSplitGroup != null)
                return;
            int n = Config.NumLayers;
            if (MoeCpuOffloadConfig.IsExplicitlySet)
            {
                if (!MoeCpuOffloadConfig.IsEnabled || _backend == BackendType.GgmlCpu)
                    return;
                // An explicit offload still records routing: a --cpu-moe run is how a complete profile is collected.
                AttachRoutingProfile(Config.Architecture, _numExperts);
                if (_backend == BackendType.GgmlMetal)
                {
                    long resident = 0;
                    for (int l = 0; l < n; l++)
                        if (_moeLayer[l] && !MoeCpuOffloadConfig.IsLayerOnCpu(l)) resident += LayerExpertBytes(l);
                    ConfigureHostExpertIo(HostExpertIoBudget(resident));
                }
                return;
            }
            if (_backend == BackendType.GgmlCuda)
                PlanCuda(n);
            else if (_backend == BackendType.GgmlMetal)
                PlanMetal(n);
        }

        /// <summary>Discrete VRAM: whole trailing layers within the free memory, or a tier of the same bytes.</summary>
        private void PlanCuda(int n)
        {
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) || total <= 0)
                return;
            var layerBytes = new long[n];
            for (int l = 0; l < n; l++)
                layerBytes[l] = _moeLayer[l] ? LayerExpertBytes(l) : 0;
            // The caches and float weights were filled on the host; their first device bindings are still to come.
            long pending = checked(CacheBytes() + _pendingDraftBytes);
            foreach (var weight in _weights.Values) pending = checked(pending + weight.Storage.ByteLength);
            int resident = PlanCudaDeviceExpertLayers(layerBytes, pending, free, GpuMemoryBudget.ResolveHeadroomBytes(total), 0);
            if (resident >= n)
                return;
            long vram = 0;
            for (int l = n - resident; l < n; l++) vram += layerBytes[l];
            if (TryPlanExpertTiers(vram, long.MaxValue, unified: false))
                return;
            _expertOnHost = new bool[n];
            for (int l = 0; l < n - resident; l++) _expertOnHost[l] = _moeLayer[l];
            ReportExpertPlacement("planned", $"CUDA free {Gb(free)}, reserved scratch {Gb(DeviceScratchBytes)}; --n-cpu-moe N overrides");
        }

        /// <summary>Unified memory: the experts stay within the Metal working set and what macOS can wire (see
        /// Qwen4ExpModel.PlanExpertPlacement, whose measurements set these bounds).</summary>
        private void PlanMetal(int n)
        {
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out _, out long workingSet) || workingSet <= 0)
                return;
            long ram = (long)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            if (ram <= 0)
                return;
            var experts = new long[n];
            for (int l = 0; l < n; l++)
                experts[l] = _moeLayer[l] ? LayerExpertBytes(l) : 0;
            long other = AcceleratorBoundNonExpertBytes() + CacheBytes() + _pendingDraftBytes;
            long deviceExpertBytes = workingSet - Math.Max(GpuMemoryBudget.MinHeadroomBytes, workingSet / 16) - other - DeviceScratchBytes;
            // Within what macOS can give without compressing, less max(7 GB, 15% of the RAM) for itself, and within 65%
            // of the RAM wired in all.
            long available = GgmlBasicOps.HostAvailableMemoryBytes();
            long keep = Math.Max(7_000_000_000L, ram / 100 * 15);
            long fits = Math.Min(ram / 100 * 65 - other - DeviceScratchBytes,
                available > 0 ? available - keep - DeviceScratchBytes - _pendingDraftBytes : long.MaxValue);
            if (fits < deviceExpertBytes)
                Console.WriteLine($"[moe-offload] {Config.Architecture}: experts on the accelerator stay within {Gb(Math.Max(0, fits))}: " +
                    $"65% of the RAM wired at most, and macOS can give {(available > 0 ? Gb(available) : "?")} without " +
                    $"compressing and keeps {Gb(keep)} (Metal working set {Gb(workingSet)}).");
            deviceExpertBytes = Math.Min(deviceExpertBytes, Math.Max(0, fits));

            int onDevice = PlanDeviceExpertLayers(experts, other, workingSet, ram);
            long layerPlanBytes = 0;
            for (int l = n - onDevice; l < n; l++) layerPlanBytes += experts[l];
            while (onDevice > 0 && layerPlanBytes > deviceExpertBytes)
                layerPlanBytes -= experts[n - onDevice--];
            if (onDevice >= n)
                return;
            // A tier puts a host seam in every MoE layer; whole layers only where offloaded. With nearly every layer
            // resident the whole-layer plan wins, with most offloaded the tier (TS_MOE_HOT_TIER=1 / =0 force either).
            bool preferLayers = Environment.GetEnvironmentVariable("TS_MOE_HOT_TIER") is not "1" && (long)onDevice * 100 >= (long)n * 85;
            if (!preferLayers && TryPlanExpertTiers(layerPlanBytes, deviceExpertBytes, unified: true))
                return;
            // Offload the FIRST layers (llama.cpp's --n-cpu-moe order): a token ends on the accelerator.
            _expertOnHost = new bool[n];
            for (int l = 0; l < n - onDevice; l++) _expertOnHost[l] = _moeLayer[l];
            ReportExpertPlacement("planned", $"Metal working set {Gb(workingSet)}, RAM {Gb(ram)}; --n-cpu-moe N overrides");
            ConfigureHostExpertIo(HostExpertIoBudget(layerPlanBytes));
        }

        /// <summary>Every MoE layer at a host seam with its most-routed experts in an accelerator tier of
        /// <paramref name="layerPlanBytes"/> (fixed slots, and on unified memory window slots filled on demand).
        /// False keeps the whole-layer plan: tiers off, or no routing profile to rank the experts by.</summary>
        private bool TryPlanExpertTiers(long layerPlanBytes, long deviceExpertBytes, bool unified)
        {
            if (!ExpertTiersEnabled())
                return false;
            AttachRoutingProfile(Config.Architecture, _numExperts);
            if (_routingProfile == null)
                return false;
            int n = Config.NumLayers;
            var moeLayers = new List<int>(n);
            var expertBytes = new long[n];
            for (int l = 0; l < n; l++)
            {
                if (!_moeLayer[l] || !TryGetLayerExperts(l, out var gate, out var up, out var down))
                    continue;
                moeLayers.Add(l);
                expertBytes[l] = gate.PerExpertRawBytes + up.PerExpertRawBytes + down.PerExpertRawBytes;
            }
            if (moeLayers.Count == 0)
                return false;
            bool profiled = _routingProfile.CoversLayers(moeLayers);
            // Discrete VRAM has no host-writable window: without a profile its tiers would be empty.
            if (!profiled && !unified)
            {
                Console.WriteLine($"[moe-offload] {Config.Architecture} (profiling): no routing profile covers this model yet " +
                    $"({_routingProfilePath}); this load keeps the layer plan, and its offloaded layers record their routing.");
                return false;
            }
            if (!profiled)
                Console.WriteLine($"[moe-offload] {Config.Architecture} (profiling): no routing profile covers this model yet " +
                    $"({_routingProfilePath}); this load gives every layer window slots only and records the routing.");
            _expertOnHost = new bool[n];
            foreach (int l in moeLayers) _expertOnHost[l] = true;

            long? tierOverride = ExpertTierBudgetOverride(Environment.GetEnvironmentVariable("TS_MOE_HOT_TIER_GB"));
            long budget = !profiled ? 0 : tierOverride ?? Math.Min(layerPlanBytes, deviceExpertBytes);
            if (profiled && unified && tierOverride == null)
                budget -= (long)(budget * WindowShare(Environment.GetEnvironmentVariable("TS_MOE_WINDOW_SHARE")));
            // On unified memory the explicitly read budget becomes each layer's window (a cold expert is read once into a
            // slot and computed on the accelerator); an eighth stays a host cache for routes wider than a window.
            long ioBudget = unified ? HostExpertIoBudget(budget) : 0;
            bool explicitIo = unified && ConfigureHostExpertIo(Math.Max(ioBudget / 8, 1L << 30));
            long windowBudget = explicitIo ? ioBudget - ioBudget / 8 : 0;
            int[][] tiers = profiled ? _routingProfile.SelectTiers(expertBytes, budget) : new int[n][];
            for (int l = 0; l < n; l++) tiers[l] ??= Array.Empty<int>();
            long tierBytes = 0;
            foreach (int l in moeLayers) tierBytes += (long)tiers[l].Length * expertBytes[l];
            windowBudget = Math.Min(windowBudget, Math.Max(0, deviceExpertBytes - tierBytes));
            long perLayerWindow = windowBudget / Math.Max(1, moeLayers.Count);
            RegisterExpertTiers(Config.Architecture, moeLayers, expertBytes, _numExperts, tiers, perLayerWindow, profiled, budget);
            return true;
        }

        /// <summary>RAM for explicitly read experts: <c>TS_HOST_MOE_IO_CACHE_GB</c>, or three quarters of what is left
        /// after the host reserve and everything wired for the accelerator.</summary>
        private long HostExpertIoBudget(long wiredExpertBytes)
        {
            if (HostExpertIoBudgetOverride(Environment.GetEnvironmentVariable("TS_HOST_MOE_IO_CACHE_GB")) is long requested)
                return requested;
            long ram = (long)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            long hostReserve = Math.Max(HostReserveFloorBytes, ram / 6);
            long wired = AcceleratorBoundNonExpertBytes() + CacheBytes() + DeviceScratchBytes + wiredExpertBytes;
            return Math.Max(0, (ram - hostReserve - wired) / 4 * 3);
        }

        /// <summary>Read the host layers' experts through the explicitly read host expert cache (Apple only).</summary>
        private bool ConfigureHostExpertIo(long budget)
        {
            bool anyHost = false;
            for (int l = 0; l < Config.NumLayers && !anyHost; l++)
                anyHost = _moeLayer[l] && IsExpertLayerOffloaded(l);
            if (!anyHost || budget <= 0 || !EnableHostExpertIo(budget))
                return false;
            Console.WriteLine($"[moe-offload] {Config.Architecture}: host experts are read with explicit I/O " +
                $"({Gb(budget)} host expert cache; TS_HOST_MOE_IO_CACHE_GB overrides, 0 reads the mapping).");
            return true;
        }

        private long LayerExpertBytes(int layer)
        {
            long bytes = 0;
            foreach (string part in new[] { "gate", "up", "down", "gate_up" })
                if (_stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_{part}_exps.weight", out var w))
                    bytes += w.TotalRawBytes;
            return bytes;
        }

        /// <summary>Every weight the accelerator binds besides the trunk's routed experts (a draft layer's experts
        /// are resident).</summary>
        private long AcceleratorBoundNonExpertBytes()
        {
            long bytes = 0;
            foreach (var kv in _quantWeights)
            {
                if (_stackedExpertMemberNames.Contains(kv.Key) || MoeCpuOffloadConfig.IsRoutedExpertWeightName(kv.Key) || IsHostOnlyWeight(kv.Key))
                    continue;
                bytes += kv.Value.RawBytes;
            }
            foreach (var kv in _weights)
                if (!IsHostOnlyWeight(kv.Key)) bytes += kv.Value.Storage.ByteLength;
            for (int l = Config.NumLayers; l < TotalLayers; l++)
                bytes += LayerExpertBytes(l);
            return bytes;
        }

        /// <summary>The attention caches as allocated now.</summary>
        private long CacheBytes()
        {
            long bytes = 0;
            if (_cacheK == null) return 0;
            for (int l = 0; l < _cacheK.Length; l++)
                for (int r = 0; _cacheK[l] != null && r < _cacheK[l].Length; r++)
                    bytes += _cacheK[l][r].Storage.ByteLength + _cacheV[l][r].Storage.ByteLength;
            return bytes;
        }

        private void ReportExpertPlacement(string how, string detail)
        {
            int n = _expertOnHost.Length, onHost = 0;
            long hostBytes = 0, deviceBytes = 0;
            for (int l = 0; l < n; l++)
            {
                long b = _moeLayer[l] ? LayerExpertBytes(l) : 0;
                if (_expertOnHost[l]) { onHost++; hostBytes += b; }
                else deviceBytes += b;
            }
            if (onHost == 0)
                return;
            Console.WriteLine($"[moe-offload] {Config.Architecture} ({how}): routed experts of {onHost} of {n} layers run on the " +
                $"host from the GGUF mapping ({Gb(hostBytes)} read on demand); the accelerator holds {Gb(deviceBytes)} of experts" +
                (detail != null ? $". {detail}." : "."));
        }

        private static string Gb(long bytes) => $"{bytes / 1e9:F1} GB";
    }
}
