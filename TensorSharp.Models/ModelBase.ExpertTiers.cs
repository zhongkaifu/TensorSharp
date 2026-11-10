// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Accelerator expert tiers for host-offloaded routed experts, for every family on the GGML
    /// backends (first built for Qwen 3.8 Flash Next). An offloaded layer runs its routed experts
    /// at a host seam; the routes whose expert sits in the layer's tier run on the accelerator
    /// from a compact copy (ggml_ops_moe_hot_tier.cpp): fixed slots hold the layer's most-routed
    /// experts, ranked by a <see cref="MoeRoutingProfile"/> collected at the seams of earlier
    /// loads, and - on unified memory - window slots are filled on demand by explicit reads, least
    /// recently used first. The native side adapts fixed slots to the conversation, reads the next
    /// layer's predicted misses ahead and skips missed low-rank routes (route-tail skip).
    ///
    /// <para><c>TS_MOE_HOT_TIER=0</c> keeps every offloaded route on the host;
    /// <c>TS_MOE_HOT_TIER_GB</c> sets the fixed-slot budget; <c>TS_MOE_WINDOW_SHARE</c> the share
    /// of a unified-memory budget given to window slots.</para>
    /// </summary>
    public abstract partial class ModelBase
    {
        private protected MoeRoutingProfile _routingProfile;

        /// <summary>Per layer, the experts its accelerator tier holds in fixed slots (null for a
        /// layer without a tier). The host serves every other route of an offloaded layer.</summary>
        private protected int[][] _tierExperts;
        private protected string _routingProfilePath;
        private bool _routingProfileExitHook;
        private int _routingProfileExperts;
        // Accelerator-resident layers' graphs also count their routing: a discrete-VRAM plan
        // without a covering profile records it for the next load.
        private protected bool _recordingDeviceRoutes;

        /// <summary>Window slots the registered tiers fill on demand (0 without tiers or on discrete VRAM).</summary>
        private protected int _tierWindowSlots;

        internal static bool ExpertTiersEnabled()
            => Environment.GetEnvironmentVariable("TS_MOE_HOT_TIER") is not "0";

        internal static long? ExpertTierBudgetOverride(string value)
            => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double gb)
               && gb >= 0 && gb < 1e6 ? (long)(gb * 1e9) : null;

        /// <summary>The share of a unified-memory tier budget given to window slots (least recently used, filled as the
        /// text routes) instead of profile-ranked fixed slots. The profile ranks experts over other prompts; the window
        /// follows this one. ISTA Q2_0 on the 48 GB M5 Pro, the same 25 GB either way: 0.3 GB of window (what the fixed
        /// slots left) read 26-47 missed experts a 4-row verify pass and computed up to 13 routes on the host, 3.1 GB
        /// (this share) read 14-19 and none on the host - the pass 52.8-59.0 -> 49.4-51.0 ms, a 1-row step 32.5-33.9 ->
        /// 31.9-32.5 ms. <c>TS_MOE_WINDOW_SHARE</c> (0 to 0.9) sets it; an explicit <c>TS_MOE_HOT_TIER_GB</c> keeps its
        /// own split (the window gets what the fixed slots leave).</summary>
        internal const double DefaultWindowShare = 0.12;

        internal static double WindowShare(string value)
            => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double share)
               && share >= 0 && share <= 0.9 ? share : DefaultWindowShare;

        private protected bool TryGetLayerExperts(int layer, out StackedExpertWeights gate, out StackedExpertWeights up,
            out StackedExpertWeights down)
        {
            up = down = null;
            gate = null;
            return _stackedExpertWeights != null
                && _stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_gate_exps.weight", out gate)
                && _stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_up_exps.weight", out up)
                && _stackedExpertWeights.TryGetValue($"blk.{layer}.ffn_down_exps.weight", out down);
        }

        /// <summary>Find (or start) this model's routing profile and zero the native counters,
        /// so what is saved at dispose is exactly this load's routing. <paramref name="architectureId"/>
        /// is part of the profile's identity: keep it stable, or earlier profiles stop matching.</summary>
        private protected void AttachRoutingProfile(string architectureId, int numExperts)
        {
            if (_routingProfile != null || !IsGgmlBackend || IsTensorParallel || numExperts <= 0)
                return;
            int n = Config.NumLayers;
            string identity = MoeRoutingProfile.IdentityFor(ModelFilePaths, architectureId, n, numExperts);
            string path = identity == null ? null : MoeRoutingProfile.PathFor(identity);
            if (path == null)
                return;
            _routingProfilePath = path;
            _routingProfileExperts = numExperts;
            _routingProfile = MoeRoutingProfile.TryLoad(path, identity, n, numExperts)
                ?? new MoeRoutingProfile(identity, n, numExperts);
            for (int l = 0; l < n; l++)
                if (TryGetLayerExperts(l, out var gate, out _, out _))
                    GgmlBasicOps.MoeRoutingReset(gate.Data);
            if (!_routingProfileExitHook)
            {
                // A CLI run can end without disposing the model; the counters are in native
                // memory, so harvest them while it still exists.
                AppDomain.CurrentDomain.ProcessExit += OnProcessExitSaveRoutingProfile;
                _routingProfileExitHook = true;
            }
        }

        private void OnProcessExitSaveRoutingProfile(object sender, EventArgs e) => SaveRoutingProfile();

        /// <summary>Merge this load's routing into the profile file. Re-reads the file first so
        /// concurrent processes on the same model add up rather than overwrite each other. Never
        /// throws: a profile is an optimization.</summary>
        private protected void SaveRoutingProfile()
        {
            if (_routingProfile == null || _routingProfilePath == null)
                return;
            try
            {
                int n = Config.NumLayers;
                int experts = _routingProfileExperts;
                MoeRoutingProfile merged = null;
                for (int l = 0; l < n; l++)
                {
                    if (!TryGetLayerExperts(l, out var gate, out _, out _))
                        continue;
                    ulong[] counts = GgmlBasicOps.MoeRoutingCounts(gate.Data);
                    if (counts == null || counts.Length != experts)
                        continue;
                    ulong sum = 0;
                    foreach (ulong c in counts) sum += c;
                    if (sum == 0)
                        continue;
                    merged ??= MoeRoutingProfile.TryLoad(_routingProfilePath, _routingProfile.Identity, n, experts)
                        ?? new MoeRoutingProfile(_routingProfile.Identity, n, experts);
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

        /// <summary>Report the tier statistics (<c>TS_MOE_TIER_STATS=1</c>), save the profile and
        /// stop recording. Call before the stacked experts are released.</summary>
        private protected void DetachRoutingProfile()
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

        /// <summary>Free this model's accelerator tiers; its graphs are already gone.</summary>
        private protected void ReleaseExpertTiers()
        {
            if (!IsGgmlBackend || _stackedExpertWeights == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
                if (TryGetLayerExperts(l, out var gate, out _, out _))
                    GgmlBasicOps.MoeHotTierRelease(gate.Data);
        }

        /// <summary>
        /// Read every fixed slot's expert slices into the page cache with parallel large reads
        /// before the tiers copy them. The native fill faults its sources in on 16 threads, a page
        /// at a time; on a network mount each fault is a round trip: measured on a geesefs mount,
        /// the 23.0 GB tier of Qwen3.8-Flash-Next took 363 s that way (~63 MB/s).
        /// </summary>
        private protected void WarmTierExperts(string family, List<int> layers, int[][] tiers)
        {
            if (_gguf == null)
                return;
            var ranges = new List<(IntPtr Pointer, long Bytes)>();
            foreach (int l in layers)
            {
                if (tiers[l].Length == 0 || !TryGetLayerExperts(l, out var gate, out var up, out var down))
                    continue;
                foreach (var w in new[] { gate, up, down })
                {
                    long per = w.PerExpertRawBytes;
                    foreach (int e in tiers[l])
                        ranges.Add((w.Data + (nint)(e * per), per));
                }
            }
            if (ranges.Count == 0)
                return;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            int threads = GgufFile.DefaultWarmThreads();
            long read = _gguf.WarmMappedRanges(ranges, threads);
            if (sw.Elapsed.TotalSeconds >= 1)
                Console.WriteLine($"[moe-offload] {family}: read the accelerator tier's experts ({TierGb(read)}) in " +
                    $"{sw.Elapsed.TotalSeconds:F1}s with {threads} parallel reads.");
        }

        /// <summary>
        /// Register each layer's tier: <paramref name="tiers"/>[l] in fixed slots plus up to
        /// <paramref name="perLayerWindowBytes"/> of window slots. Returns the number of layers
        /// tiered; a layer whose tier the native side declines keeps every route on the host.
        /// </summary>
        private protected int RegisterExpertTiers(string family, List<int> moeLayers, long[] expertBytes, int numExperts,
            int[][] tiers, long perLayerWindowBytes, bool profiled, long budget)
        {
            int n = Config.NumLayers;
            WarmTierExperts(family, moeLayers, tiers);
            _tierExperts = new int[n][];
            int tiered = 0, experts = 0, windowSlots = 0;
            long bytes = 0;
            double covered = 0;
            foreach (int l in moeLayers)
            {
                int[] tier = tiers[l];
                int window = (int)Math.Min(numExperts - tier.Length, perLayerWindowBytes / expertBytes[l]);
                if ((tier.Length == 0 && window == 0) || !TryGetLayerExperts(l, out var gate, out var up, out var down))
                    continue;
                if (!GgmlBasicOps.MoeHotTierRegister(gate.Data, up.Data, down.Data,
                        gate.GgmlType, up.GgmlType, down.GgmlType,
                        gate.TotalRawBytes, up.TotalRawBytes, down.TotalRawBytes,
                        numExperts, tier, window, releaseSourcePages: true))
                {
                    Console.Error.WriteLine($"[moe-offload] {family}: layer {l} keeps every route on the host " +
                        $"(accelerator tier not created: {GgmlBasicOps.LastNativeError()})");
                    continue;
                }
                tiered++;
                _tierExperts[l] = tier;
                experts += tier.Length;
                bytes += tier.Length * expertBytes[l];
                windowSlots += window;
                if (profiled)
                {
                    ReadOnlySpan<ulong> counts = _routingProfile.Counts(l);
                    ulong hit = 0;
                    foreach (int e in tier) hit += counts[e];
                    covered += (double)hit / _routingProfile.Routes(l);
                }
            }
            _tierWindowSlots = windowSlots;
            Console.WriteLine($"[moe-offload] {family} (tiered): all {moeLayers.Count} layers' routed experts run at " +
                $"a host seam; {tiered} layers keep their most-routed experts on the accelerator ({experts} experts, " +
                $"{TierGb(bytes)} of a {TierGb(budget)} budget) covering {covered / Math.Max(1, moeLayers.Count) * 100:F1}% " +
                $"of profiled routes, plus {windowSlots} window slots ({TierGb(windowSlots * (long)(expertBytes[moeLayers[0]]))}) " +
                $"filled on demand. Profile: {_routingProfilePath}.");
            return tiered;
        }

        /// <summary>Test hook: give every offloaded layer a tier holding experts 0..fixed-1 plus
        /// <paramref name="windowPerLayer"/> window slots. Call before the first forward. Returns the
        /// number of layers tiered.</summary>
        internal int RegisterExpertTiersForTests(int fixedPerLayer, int windowPerLayer)
        {
            int tiered = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!IsExpertLayerOffloaded(l) || !TryGetLayerExperts(l, out var gate, out var up, out var down))
                    continue;
                int numExperts = gate.NumExperts;
                int fixedCount = Math.Clamp(fixedPerLayer, 0, numExperts);
                int window = Math.Clamp(windowPerLayer, 0, numExperts - fixedCount);
                var ids = new int[fixedCount];
                for (int i = 0; i < ids.Length; i++) ids[i] = i;
                if (GgmlBasicOps.MoeHotTierRegister(gate.Data, up.Data, down.Data,
                        gate.GgmlType, up.GgmlType, down.GgmlType,
                        gate.TotalRawBytes, up.TotalRawBytes, down.TotalRawBytes,
                        numExperts, ids, window, releaseSourcePages: false))
                    tiered++;
            }
            return tiered;
        }

        /// <summary>Whether layer <paramref name="layer"/>'s routed experts run at a host seam (the
        /// family's offload plan; <c>--n-cpu-moe</c> / <c>--cpu-moe</c> by default).</summary>
        private protected virtual bool IsExpertLayerOffloaded(int layer) => MoeCpuOffloadConfig.IsLayerOnCpu(layer);

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

        private static string TierGb(long bytes) => $"{bytes / 1e9:F1} GB";
    }
}
