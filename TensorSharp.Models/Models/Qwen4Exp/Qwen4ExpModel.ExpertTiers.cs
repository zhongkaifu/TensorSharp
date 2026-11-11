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
using System.Globalization;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using static TensorSharp.Models.ExpertPlacementPlanner;

namespace TensorSharp.Models
{
    /// <summary>
    /// Accelerator tiers for host-offloaded experts, ranked by routing frequency.
    ///
    /// <para>The layer plan in <see cref="PlanExpertPlacement"/> wires whole trailing
    /// layers: every one of their 512 experts stays resident, used or not, while
    /// the other layers page their experts in from the GGUF mapping. A tiered plan
    /// spends the same bytes on the experts that are routed most, in every layer:
    /// all layers run their routed experts at a host seam, and the routes whose
    /// expert sits in the layer's tier run on the accelerator from a compact copy
    /// inside the same graph (ggml_ops_moe_hot_tier.cpp). On Qwen3.8-Flash-Next the
    /// most-routed third of a layer's experts takes about half of its routes, and
    /// the RAM that wiring their cold neighbours used to cost goes to the page
    /// cache instead.</para>
    ///
    /// <para>The ranking is a <see cref="MoeRoutingProfile"/> collected from the host
    /// seams of earlier loads. With none yet, every layer runs on the host for one
    /// load (decode speed measured on par with the layer plan) and records its
    /// routing; the tiers come with the next load. <c>TS_MOE_HOT_TIER=0</c> keeps
    /// the layer plan; <c>TS_MOE_HOT_TIER_GB</c> sets the tier budget (default: the
    /// bytes the layer plan would have wired).</para>
    /// </summary>
    /// <summary>
    /// Flash Next's tier budget (see <see cref="ModelBase"/>'s expert tiers, which register,
    /// profile and release them for every family). The layer plan in
    /// <see cref="PlanExpertPlacement"/> wires whole trailing layers: every one of their 512
    /// experts stays resident, used or not. A tiered plan spends the same bytes on the experts that
    /// are routed most, in every layer; on Qwen3.8-Flash-Next the most-routed third of a layer's
    /// experts takes about half of its routes.
    /// </summary>
    public partial class Qwen4ExpModel
    {
        private protected override bool IsExpertLayerOffloaded(int layer) => IsExpertLayerOnHost(layer);

        /// <summary>RAM for explicitly read experts: <c>TS_HOST_MOE_IO_CACHE_GB</c>, or three
        /// quarters of what is left after the host reserve and everything wired for the
        /// accelerator (the rest stays page cache for the PLE rows, prefill streaming
        /// and the process).</summary>
        private long HostExpertIoBudget(long wiredExpertBytes)
        {
            long? requested = HostExpertIoBudgetOverride(Environment.GetEnvironmentVariable("TS_HOST_MOE_IO_CACHE_GB"));
            if (requested is long r)
                return r;
            long ram = (long)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            long hostReserve = Math.Max(HostReserveFloorBytes, ram / 6);
            long wired = AcceleratorBoundNonExpertBytes() + CacheBytes() + DeviceScratchBytes + wiredExpertBytes;
            return Math.Max(0, (ram - hostReserve - wired) / 4 * 3);
        }

        /// <summary>Read the host layers' experts through the explicitly read host
        /// expert cache of <paramref name="budget"/> bytes. Returns false when it does
        /// not apply (budget 0, not Apple, no mapping).</summary>
        private bool ConfigureHostExpertIo(long budget)
        {
            if (_expertOnHost == null || Array.IndexOf(_expertOnHost, true) < 0 || budget <= 0)
                return false;
            if (!EnableHostExpertIo(budget))
                return false;
            Console.WriteLine($"[moe-offload] qwen4exp: host experts are read with explicit I/O " +
                $"({Gb(budget)} host expert cache; TS_HOST_MOE_IO_CACHE_GB overrides, 0 reads the mapping).");
            return true;
        }

        /// <summary>Test hook: serve the GGUF mapping through the explicit-read I/O layer (as a tiered Mac load does),
        /// with a host expert cache of <paramref name="budgetBytes"/>.</summary>
        internal bool EnableHostExpertIoForTests(long budgetBytes) => EnableHostExpertIo(budgetBytes);

        internal int ExpertCountForTests => _numExperts;

        /// <summary>The tiered plan for a Metal run whose experts do not fit. Returns
        /// false to keep the layer plan. <paramref name="deviceExpertBytes"/> bounds the
        /// tier and the window slots together: on unified memory both are accelerator
        /// buffers inside the Metal working set.</summary>
        private bool TryPlanExpertTiers(long layerPlanBytes, long deviceExpertBytes = long.MaxValue)
        {
            bool unified = _backend == BackendType.GgmlMetal;
            if (!ExpertTiersEnabled() || _fusedGateUpExperts || !(unified || _backend == BackendType.GgmlCuda))
                return false;
            AttachRoutingProfile(ArchitectureId, _fusedGateUpExperts ? 0 : _numExperts);
            if (_routingProfile == null)
                return false;
            int n = Config.NumLayers;
            var moeLayers = new List<int>(n);
            var expertBytes = new long[n];
            for (int l = 0; l < n; l++)
            {
                if (!TryGetLayerExperts(l, out var gate, out var up, out var down))
                    continue;
                moeLayers.Add(l);
                expertBytes[l] = gate.PerExpertRawBytes + up.PerExpertRawBytes + down.PerExpertRawBytes;
            }
            if (moeLayers.Count == 0)
                return false;
            foreach (int l in moeLayers)
                _expertOnHost[l] = true;

            // Without a profile there is nothing to rank fixed slots by: every slot is a
            // window slot (least recently used), which already follows the text being
            // generated, while the seams record the profile for the next load.
            bool profiled = _routingProfile.CoversLayers(moeLayers);
            // Discrete VRAM has no host-writable window: there the tiers are fixed slots
            // only, so without a profile the whole-layer plan stays. Its resident layers
            // have no host seam to count at, so their graphs record their router ids too:
            // the profile saved at dispose then covers every layer, and the next load
            // keeps each layer's most-routed experts in VRAM.
            if (!profiled && !unified)
            {
                foreach (int l in moeLayers) _expertOnHost[l] = false;
                GgmlBasicOps.MoeRoutingRecordDevice(true);
                _recordingDeviceRoutes = true;
                Console.WriteLine($"[moe-offload] qwen4exp (profiling): no routing profile covers this model yet " +
                    $"({_routingProfilePath}); this load keeps the layer plan and records every layer's routing, " +
                    $"so a later load keeps each layer's most-routed experts in VRAM once {MoeRoutingProfile.MinimumRoutesPerLayer:N0} " +
                    "routes a layer are recorded.");
                return false;
            }
            if (!profiled)
                Console.WriteLine($"[moe-offload] qwen4exp (profiling): no routing profile covers this model yet " +
                    $"({_routingProfilePath}); this load gives every layer window slots only and records the routing, " +
                    "the next load also keeps each layer's most-routed experts in fixed slots.");

            long? tierOverride = ExpertTierBudgetOverride(Environment.GetEnvironmentVariable("TS_MOE_HOT_TIER_GB"));
            long budget = !profiled ? 0 : tierOverride ?? Math.Min(layerPlanBytes, deviceExpertBytes);
            // On unified memory a share of the budget goes to window slots (see DefaultWindowShare).
            if (profiled && unified && tierOverride == null)
                budget -= (long)(budget * WindowShare(Environment.GetEnvironmentVariable("TS_MOE_WINDOW_SHARE")));
            // The explicitly read budget becomes each layer's window: accelerator slots
            // the seams fill on demand, so a cold expert is read once into a slot and
            // computed on the accelerator, never on the host. A small share stays a
            // host cache for routes wider than a window.
            long ioBudget = unified ? HostExpertIoBudget(budget) : 0;
            bool explicitIo = unified && ConfigureHostExpertIo(Math.Max(ioBudget / 8, 1L << 30));
            long windowBudget = explicitIo ? ioBudget - ioBudget / 8 : 0;
            int[][] tiers = profiled ? _routingProfile.SelectTiers(expertBytes, budget) : new int[n][];
            for (int l = 0; l < n; l++) tiers[l] ??= Array.Empty<int>();
            // The window is sized from host RAM, but its slots are accelerator buffers too:
            // what the tier leaves of the working set bounds it. A layer plan that already
            // fills the working set (ISTA Q2_0: 29.7 GB of experts on a 40.2 GB set) left the
            // 3.1 GB of window slots it was given out of every account, and Metal ran out of
            // memory on the first long answer.
            long tierBytes = 0;
            foreach (int l in moeLayers) tierBytes += (long)tiers[l].Length * expertBytes[l];
            windowBudget = Math.Min(windowBudget, Math.Max(0, deviceExpertBytes - tierBytes));
            long perLayerWindow = windowBudget / Math.Max(1, moeLayers.Count);
            RegisterExpertTiers("qwen4exp", moeLayers, expertBytes, _numExperts, tiers, perLayerWindow, profiled, budget);
            return true;
        }
    }
}
