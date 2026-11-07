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
    public partial class Qwen4ExpModel
    {
        private MoeRoutingProfile _routingProfile;
        private string _routingProfilePath;
        private bool _routingProfileExitHook;
        // The accelerator-resident layers' graphs also count their routing (see
        // TryPlanExpertTiers): a discrete-VRAM plan without a covering profile.
        private bool _recordingDeviceRoutes;

        internal static bool ExpertTiersEnabled()
            => Environment.GetEnvironmentVariable("TS_MOE_HOT_TIER") is not "0";

        internal static long? ExpertTierBudgetOverride(string value)
            => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double gb)
               && gb >= 0 && gb < 1e6 ? (long)(gb * 1e9) : null;

        private bool TryGetLayerExperts(int layer, out StackedExpertWeights gate, out StackedExpertWeights up,
            out StackedExpertWeights down)
        {
            up = down = null;
            return _stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_gate_exps.weight", out gate)
                && _stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_up_exps.weight", out up)
                && _stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_down_exps.weight", out down);
        }

        /// <summary>Find (or start) this model's routing profile and zero the native
        /// counters, so what is saved at dispose is exactly this load's routing.</summary>
        private void AttachRoutingProfile()
        {
            if (_routingProfile != null || !IsGgmlBackend || IsTensorParallel || _fusedGateUpExperts)
                return;
            int n = Config.NumLayers;
            string identity = MoeRoutingProfile.IdentityFor(ModelFilePaths, ArchitectureId, n, _numExperts);
            string path = identity == null ? null : MoeRoutingProfile.PathFor(identity);
            if (path == null)
                return;
            _routingProfilePath = path;
            _routingProfile = MoeRoutingProfile.TryLoad(path, identity, n, _numExperts)
                ?? new MoeRoutingProfile(identity, n, _numExperts);
            for (int l = 0; l < n; l++)
                if (TryGetLayerExperts(l, out var gate, out _, out _))
                    GgmlBasicOps.MoeRoutingReset(gate.Data);
            if (!_routingProfileExitHook)
            {
                // A CLI run can end without disposing the model; the counters are
                // in native memory, so harvest them while it still exists.
                AppDomain.CurrentDomain.ProcessExit += OnProcessExitSaveRoutingProfile;
                _routingProfileExitHook = true;
            }
        }

        private void OnProcessExitSaveRoutingProfile(object sender, EventArgs e) => SaveRoutingProfile();

        /// <summary>Merge this load's routing into the profile file. Re-reads the file
        /// first so concurrent processes on the same model add up rather than
        /// overwrite each other. Never throws: a profile is an optimization.</summary>
        private void SaveRoutingProfile()
        {
            if (_routingProfile == null || _routingProfilePath == null)
                return;
            try
            {
                int n = Config.NumLayers;
                MoeRoutingProfile merged = null;
                for (int l = 0; l < n; l++)
                {
                    if (!TryGetLayerExperts(l, out var gate, out _, out _))
                        continue;
                    ulong[] counts = GgmlBasicOps.MoeRoutingCounts(gate.Data);
                    if (counts == null || counts.Length != _numExperts)
                        continue;
                    ulong sum = 0;
                    foreach (ulong c in counts) sum += c;
                    if (sum == 0)
                        continue;
                    merged ??= MoeRoutingProfile.TryLoad(_routingProfilePath, _routingProfile.Identity, n, _numExperts)
                        ?? new MoeRoutingProfile(_routingProfile.Identity, n, _numExperts);
                    merged.Add(l, counts);
                    GgmlBasicOps.MoeRoutingReset(gate.Data);
                }
                if (merged != null && merged.TrySave(_routingProfilePath))
                    _routingProfile = merged;
            }
            catch (Exception ex) when (ex is OverflowException or InvalidOperationException or DllNotFoundException
                or EntryPointNotFoundException)
            {
                Console.Error.WriteLine($"[moe-offload] routing profile not saved: {ex.Message}");
            }
        }

        private void DetachRoutingProfile()
        {
            if (IsGgmlBackend && Environment.GetEnvironmentVariable("TS_MOE_TIER_STATS") == "1")
            {
                var t = GgmlBasicOps.MoeHotTierStats();
                var io = GgmlBasicOps.HostExpertIoStats();
                Console.Error.WriteLine($"[moe-offload] tier slots: {t.Hits} hits, {t.Misses} window reads " +
                    $"({t.Deferred} in the background), {t.Overflow} overflow steps; host cache: {io.Hits} hits, " +
                    $"{io.Misses} misses, {io.ReadBytes / 1e9:F1} GB read.");
            }
            SaveRoutingProfile();
            if (_recordingDeviceRoutes)
            {
                GgmlBasicOps.MoeRoutingRecordDevice(false);
                _recordingDeviceRoutes = false;
            }
            if (_routingProfileExitHook)
            {
                AppDomain.CurrentDomain.ProcessExit -= OnProcessExitSaveRoutingProfile;
                _routingProfileExitHook = false;
            }
            _routingProfile = null;
        }

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

        /// <summary>Test hook: give every host layer a tier holding experts 0..fixed-1 plus
        /// <paramref name="windowPerLayer"/> window slots. Call before the first forward.
        /// Returns the number of layers tiered.</summary>
        internal int RegisterExpertTiersForTests(int fixedPerLayer, int windowPerLayer)
        {
            int tiered = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!IsExpertLayerOnHost(l) || !TryGetLayerExperts(l, out var gate, out var up, out var down))
                    continue;
                int fixedCount = Math.Clamp(fixedPerLayer, 0, _numExperts);
                int window = Math.Clamp(windowPerLayer, 0, _numExperts - fixedCount);
                var ids = new int[fixedCount];
                for (int i = 0; i < ids.Length; i++) ids[i] = i;
                if (GgmlBasicOps.MoeHotTierRegister(gate.Data, up.Data, down.Data,
                        gate.GgmlType, up.GgmlType, down.GgmlType,
                        gate.TotalRawBytes, up.TotalRawBytes, down.TotalRawBytes,
                        _numExperts, ids, window, releaseSourcePages: false))
                    tiered++;
            }
            return tiered;
        }

        internal int ExpertCountForTests => _numExperts;

        /// <summary>Test hook: the native routing counters of layer <paramref name="layer"/>.</summary>
        internal ulong[] RoutingCountsForTests(int layer)
            => TryGetLayerExperts(layer, out var gate, out _, out _) ? GgmlBasicOps.MoeRoutingCounts(gate.Data) : null;

        /// <summary>Test hook: record (or stop recording) accelerator-resident layers' routing.</summary>
        internal void RecordDeviceRoutesForTests(bool on)
        {
            for (int l = 0; l < Config.NumLayers; l++)
                if (TryGetLayerExperts(l, out var gate, out _, out _))
                    GgmlBasicOps.MoeRoutingReset(gate.Data);
            GgmlBasicOps.MoeRoutingRecordDevice(on);
            _recordingDeviceRoutes = on;
        }

        /// <summary>Free this model's accelerator tiers; its graphs are already gone.</summary>
        private void ReleaseExpertTiers()
        {
            if (!IsGgmlBackend || _stackedExpertWeights == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
                if (TryGetLayerExperts(l, out var gate, out _, out _))
                    GgmlBasicOps.MoeHotTierRelease(gate.Data);
        }

        /// <summary>The tiered plan for a Metal run whose experts do not fit. Returns
        /// false to keep the layer plan.</summary>
        private bool TryPlanExpertTiers(long layerPlanBytes)
        {
            bool unified = _backend == BackendType.GgmlMetal;
            if (!ExpertTiersEnabled() || _fusedGateUpExperts || !(unified || _backend == BackendType.GgmlCuda))
                return false;
            AttachRoutingProfile();
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

            long budget = !profiled ? 0
                : ExpertTierBudgetOverride(Environment.GetEnvironmentVariable("TS_MOE_HOT_TIER_GB")) ?? layerPlanBytes;
            // The explicitly read budget becomes each layer's window: accelerator slots
            // the seams fill on demand, so a cold expert is read once into a slot and
            // computed on the accelerator, never on the host. A small share stays a
            // host cache for routes wider than a window.
            long ioBudget = unified ? HostExpertIoBudget(budget) : 0;
            bool explicitIo = unified && ConfigureHostExpertIo(Math.Max(ioBudget / 8, 1L << 30));
            long windowBudget = explicitIo ? ioBudget - ioBudget / 8 : 0;
            long perLayerWindow = windowBudget / Math.Max(1, moeLayers.Count);
            int[][] tiers = profiled ? _routingProfile.SelectTiers(expertBytes, budget) : new int[n][];
            for (int l = 0; l < n; l++) tiers[l] ??= Array.Empty<int>();
            int tiered = 0, experts = 0, windowSlots = 0;
            long bytes = 0;
            double covered = 0;
            foreach (int l in moeLayers)
            {
                int[] tier = tiers[l];
                int window = (int)Math.Min(_numExperts - tier.Length, perLayerWindow / expertBytes[l]);
                if ((tier.Length == 0 && window == 0) || !TryGetLayerExperts(l, out var gate, out var up, out var down))
                    continue;
                if (!GgmlBasicOps.MoeHotTierRegister(gate.Data, up.Data, down.Data,
                        gate.GgmlType, up.GgmlType, down.GgmlType,
                        gate.TotalRawBytes, up.TotalRawBytes, down.TotalRawBytes,
                        _numExperts, tier, window, releaseSourcePages: true))
                {
                    Console.Error.WriteLine($"[moe-offload] qwen4exp: layer {l} keeps every route on the host " +
                        $"(accelerator tier not created: {GgmlBasicOps.LastNativeError()})");
                    continue;
                }
                tiered++;
                experts += tier.Length;
                bytes += tier.Length * expertBytes[l];
                windowSlots += window;
                ReadOnlySpan<ulong> counts = _routingProfile.Counts(l);
                ulong hit = 0;
                foreach (int e in tier) hit += counts[e];
                if (profiled) covered += (double)hit / _routingProfile.Routes(l);
            }
            Console.WriteLine($"[moe-offload] qwen4exp (tiered): all {moeLayers.Count} layers' routed experts run at " +
                $"a host seam; {tiered} layers keep their most-routed experts on the accelerator ({experts} experts, " +
                $"{Gb(bytes)} of a {Gb(budget)} budget) covering {covered / Math.Max(1, moeLayers.Count) * 100:F1}% " +
                $"of profiled routes, plus {windowSlots} window slots ({Gb(windowSlots * (long)(expertBytes[moeLayers[0]]))}) " +
                $"filled on demand. Profile: {_routingProfilePath}.");
            return true;
        }
    }
}
