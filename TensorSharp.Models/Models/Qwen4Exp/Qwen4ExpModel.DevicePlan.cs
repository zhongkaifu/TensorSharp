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

namespace TensorSharp.Models
{
    /// <summary>
    /// What one GPU needs, beyond weights and caches, while it runs its share of a
    /// qwen4exp token span. Everything here grows with the widest span the model
    /// runs (<see cref="MaxTokens"/>), which is why the model sub-chunks prefill to
    /// that width instead of taking whatever the scheduler hands it.
    ///
    /// <para>Measured with TS_GGML_LOG_VRAM=1 on Qwen3.8-Flash-Next UD-IQ1_M (RTX 3080
    /// Laptop 16 GB, 40 host-routed layers, MAX_CONTEXT=8192): the span graph buffer
    /// was 5.3 MB at T=1, 572 MB at T=2048 and 1561 MB at T=4096 (~300-330 KB a token
    /// plus the [rows, T] attention terms); the expert-stream buffer one layer of
    /// experts plus ~60-85 KB a token; the ggml-cuda pool ~200-280 KB a token at
    /// prefill. Peak use during the 4096-token span was 3.74 GB. Before the seam
    /// reused its buffers the same spans needed 2164 / 4705 MB: each host-routed
    /// layer held 20 KB a token for the whole pass.</para>
    /// </summary>
    internal readonly struct Qwen4ExpSpanScratch
    {
        /// <summary>Allocator pools, cuBLAS / precise-backend workspaces and CUDA
        /// graphs that exist on any device that ran a span.</summary>
        internal const long FixedBytes = 256L << 20;

        /// <summary>Span activations (reused layer to layer) plus the ggml-cuda
        /// pool's prefill growth, per token of span width.</summary>
        internal const long BytesPerToken = 352L * 1024 + 256L * 1024;

        /// <summary>What a host-routed layer's seam still holds for the whole pass, per
        /// token: its top-k ids and weights. moe_in / moe_out share one buffer that
        /// is freed after the layer (ggml_ops_qwen4exp.cpp, kQ4eSeamReuseMinTokens).</summary>
        internal const long SeamBytesPerTokenPerHostLayer = 256;

        /// <summary>Expert-stream graph activations per token, on top of the layer of
        /// experts the stream buffer holds.</summary>
        internal const long StreamBytesPerToken = 96L * 1024;

        /// <summary>Per (KV row x token): the F16 attention mask input plus the F32
        /// [rows, T] temporaries QSA's sparse mask builds per attention layer.</summary>
        internal const long BytesPerKvRowToken = 18;

        /// <param name="extraBytesPerKvRowToken">Per (KV row x token) on top of
        /// <see cref="BytesPerKvRowToken"/>: an F32 KV cache cannot use flash
        /// attention, and the soft_max path materializes F32 scores and probabilities
        /// for every head.</param>
        internal Qwen4ExpSpanScratch(int maxTokens, long kvRowTokens, int streamMinTokens,
            long seamBytesPerTokenPerHostLayer = SeamBytesPerTokenPerHostLayer, long extraBytesPerKvRowToken = 0)
        {
            if (maxTokens < 1) throw new ArgumentOutOfRangeException(nameof(maxTokens));
            if (kvRowTokens < 0) throw new ArgumentOutOfRangeException(nameof(kvRowTokens));
            if (seamBytesPerTokenPerHostLayer < 0) throw new ArgumentOutOfRangeException(nameof(seamBytesPerTokenPerHostLayer));
            if (extraBytesPerKvRowToken < 0) throw new ArgumentOutOfRangeException(nameof(extraBytesPerKvRowToken));
            MaxTokens = maxTokens;
            KvRowTokens = kvRowTokens;
            StreamMinTokens = Math.Max(0, streamMinTokens);
            SeamPerTokenPerHostLayer = seamBytesPerTokenPerHostLayer;
            KvRowTokenBytes = BytesPerKvRowToken + extraBytesPerKvRowToken;
        }

        /// <summary>Widest span (tokens) a device runs.</summary>
        internal int MaxTokens { get; }

        /// <summary>Largest KV-rows x tokens product a span reads.</summary>
        internal long KvRowTokens { get; }

        /// <summary>TS_HOST_MOE_DEVICE_MIN_BATCH: a host-routed layer's experts are
        /// streamed onto the GPU for spans at least this wide. 0 = never.</summary>
        internal int StreamMinTokens { get; }

        internal long SeamPerTokenPerHostLayer { get; }

        /// <summary>Bytes per (KV row x token) the span's attention terms take.</summary>
        internal long KvRowTokenBytes { get; }

        internal bool Streams => StreamMinTokens > 0 && MaxTokens >= StreamMinTokens;

        /// <summary>Scratch for a device running <paramref name="hostLayers"/>
        /// host-routed layers, the largest of which has
        /// <paramref name="largestHostLayerExperts"/> bytes of experts.</summary>
        internal long Bytes(int hostLayers, long largestHostLayerExperts)
        {
            long bytes = checked(FixedBytes + BytesPerToken * MaxTokens + KvRowTokenBytes * KvRowTokens);
            if (hostLayers > 0)
            {
                bytes = checked(bytes + SeamPerTokenPerHostLayer * MaxTokens * hostLayers);
                // The stream buffer is sized to a whole layer of experts, all of them,
                // and it is persistent: the first wide prefill keeps it for good.
                if (Streams)
                    bytes = checked(bytes + largestHostLayerExperts + StreamBytesPerToken * MaxTokens);
            }
            return bytes;
        }
    }

    /// <summary>A layer -> GPU map plus which layers' routed experts run on the host.</summary>
    internal sealed class Qwen4ExpPlacementPlan
    {
        internal Qwen4ExpPlacementPlan(int[] layerDevice, bool[] expertOnHost, long[] deviceBytes, bool fits)
        {
            LayerDevice = layerDevice;
            ExpertOnHost = expertOnHost;
            DeviceBytes = deviceBytes;
            Fits = fits;
        }

        internal int[] LayerDevice { get; }
        internal bool[] ExpertOnHost { get; }

        /// <summary>Bytes the plan charges each device, scratch included.</summary>
        internal long[] DeviceBytes { get; }

        /// <summary>False when no assignment fits even with every expert on the host;
        /// the arrays then hold the best-effort map.</summary>
        internal bool Fits { get; }

        internal int HostLayers
        {
            get { int n = 0; foreach (bool h in ExpertOnHost) if (h) n++; return n; }
        }
    }

    public partial class Qwen4ExpModel
    {
        /// <summary>
        /// Place a layer split and its expert offload together, against what each GPU
        /// can actually hold. Issue #256: two 20 GB cards cannot hold 62 GB of routed
        /// experts, and a split that prices every expert as resident asks each card for
        /// 27-38 GB.
        ///
        /// <para>Each device holds a CONTIGUOUS, non-empty run of layers (see
        /// <see cref="PackLayersOntoDevices"/> for why). Within its run, a device keeps
        /// the experts of its trailing layers and routes the leading ones to the host -
        /// the same rule the single-GPU plan and llama.cpp's --n-cpu-moe follow. The
        /// plan maximizes the number of resident layers (each host-routed layer costs a
        /// seam and a host expert pass every token), then spreads the remaining room so
        /// no device is fuller, relative to its budget, than it has to be.</para>
        ///
        /// <para><paramref name="fixedMap"/> pins the runs (TS_Q4E_LAYER_SPLIT);
        /// <paramref name="forcedHost"/> pins the host set (--n-cpu-moe / --cpu-moe)
        /// and the runs are then packed around it, so host-routed layers no longer
        /// count their experts against the device that runs them.</para>
        /// </summary>
        /// <param name="layerBytes">Per layer: bytes that are device-resident whatever
        /// happens to its experts (dense weights, caches, recurrent state).</param>
        /// <param name="expertBytes">Per layer: routed expert bytes.</param>
        /// <param name="deviceCapacity">Per device: bytes the plan may spend
        /// (free memory less headroom). Scratch is charged by <paramref name="scratch"/>.</param>
        /// <param name="firstDeviceFixed">Bytes only device 0 holds (embedding, vision).</param>
        /// <param name="lastDeviceFixed">Bytes only the last device holds (head, MTP).</param>
        internal static Qwen4ExpPlacementPlan PlanQwen4ExpPlacement(
            long[] layerBytes, long[] expertBytes, long[] deviceCapacity,
            long firstDeviceFixed, long lastDeviceFixed, Qwen4ExpSpanScratch scratch,
            int[] fixedMap = null, bool[] forcedHost = null)
        {
            ArgumentNullException.ThrowIfNull(layerBytes);
            ArgumentNullException.ThrowIfNull(expertBytes);
            ArgumentNullException.ThrowIfNull(deviceCapacity);
            int n = layerBytes.Length, devices = deviceCapacity.Length;
            if (expertBytes.Length != n)
                throw new ArgumentException("Expert sizes must match the layer count.", nameof(expertBytes));
            if (devices < 1 || n < devices)
                throw new ArgumentException($"{n} layer(s) cannot cover {devices} device(s).", nameof(deviceCapacity));
            if (firstDeviceFixed < 0 || lastDeviceFixed < 0)
                throw new ArgumentOutOfRangeException(nameof(firstDeviceFixed));
            for (int l = 0; l < n; l++)
                if (layerBytes[l] < 0 || expertBytes[l] < 0)
                    throw new ArgumentOutOfRangeException(nameof(layerBytes));
            if (forcedHost != null && forcedHost.Length != n)
                throw new ArgumentException("The host set must match the layer count.", nameof(forcedHost));
            if (fixedMap != null)
                ValidateContiguousMap(fixedMap, n, devices);

            var costs = new SpanCosts(layerBytes, expertBytes, firstDeviceFixed, lastDeviceFixed, scratch, forcedHost, devices);

            if (fixedMap != null)
            {
                // The runs are pinned: each device only chooses its own host set.
                var host = new bool[n];
                var used = new long[devices];
                bool fits = true;
                for (int d = 0, begin = 0; d < devices; d++)
                {
                    int end = begin;
                    while (end < n && fixedMap[end] == d) end++;
                    int resident = costs.BestResident(d, begin, end, deviceCapacity[d]);
                    if (resident < 0) { fits = false; resident = 0; }
                    used[d] = costs.Apply(d, begin, end, resident, host);
                    begin = end;
                }
                return new Qwen4ExpPlacementPlan((int[])fixedMap.Clone(), host, used, fits);
            }

            // Maximize resident layers at full budgets, then find the smallest common
            // fill fraction that still reaches that count: the same balance DSV4 and
            // GLM-DSA bisect for, on top of an exact search instead of a greedy fill.
            int best = costs.Solve(deviceCapacity, 1.0, null);
            if (best < 0)
            {
                // Nothing fits, not even with every expert on the host. Report the
                // byte-balanced runs with everything routed to the host, so the caller
                // can name what each device would still need.
                int[] map = PackLayersOntoDevices(layerBytes, firstDeviceFixed, lastDeviceFixed, devices);
                var host = new bool[n];
                var used = new long[devices];
                for (int d = 0, begin = 0; d < devices; d++)
                {
                    int end = begin;
                    while (end < n && map[end] == d) end++;
                    used[d] = costs.Apply(d, begin, end, 0, host);
                    begin = end;
                }
                return new Qwen4ExpPlacementPlan(map, host, used, false);
            }

            double lo = 0.0, hi = 1.0;
            for (int i = 0; i < 40 && hi - lo > 1e-4; i++)
            {
                double mid = (lo + hi) / 2;
                if (costs.Solve(deviceCapacity, mid, null) >= best) hi = mid;
                else lo = mid;
            }
            var ends = new int[devices];
            int resolved = costs.Solve(deviceCapacity, hi, ends);
            if (resolved < best)
                throw new InvalidOperationException("qwen4exp placement: the balanced solution lost resident layers.");

            var finalMap = new int[n];
            var finalHost = new bool[n];
            var finalUsed = new long[devices];
            for (int d = 0, begin = 0; d < devices; d++)
            {
                int end = ends[d];
                for (int l = begin; l < end; l++) finalMap[l] = d;
                int resident = costs.BestResident(d, begin, end, (long)(deviceCapacity[d] * hi));
                finalUsed[d] = costs.Apply(d, begin, end, Math.Max(0, resident), finalHost);
                begin = end;
            }
            return new Qwen4ExpPlacementPlan(finalMap, finalHost, finalUsed, true);
        }

        /// <summary>How many of a run's trailing layers keep their experts on the device
        /// with <paramref name="available"/> bytes to spend, or -1 when even routing
        /// every expert of the run to the host does not fit. The phase that runs after
        /// the preload, against measured free memory.</summary>
        internal static int FitResidentLayers(long[] layerBytes, long[] expertBytes, int begin, int end,
            long available, long fixedBytes, Qwen4ExpSpanScratch scratch, bool[] forcedHost = null)
            => FitResidentLayers(layerBytes, expertBytes, begin, end, available, fixedBytes, scratch, forcedHost, out _);

        /// <param name="bytes">What the chosen placement costs the device; when nothing
        /// fits, what the cheapest placement (every expert on the host, or the forced
        /// host set) would have needed.</param>
        internal static int FitResidentLayers(long[] layerBytes, long[] expertBytes, int begin, int end,
            long available, long fixedBytes, Qwen4ExpSpanScratch scratch, bool[] forcedHost, out long bytes)
        {
            ArgumentNullException.ThrowIfNull(layerBytes);
            ArgumentNullException.ThrowIfNull(expertBytes);
            if (expertBytes.Length != layerBytes.Length || (forcedHost != null && forcedHost.Length != layerBytes.Length))
                throw new ArgumentException("Per-layer arrays must have one entry per layer.", nameof(expertBytes));
            if (begin < 0 || end > layerBytes.Length || begin >= end)
                throw new ArgumentOutOfRangeException(nameof(begin));
            var costs = new SpanCosts(layerBytes, expertBytes, fixedBytes, 0, scratch, forcedHost, 1);
            int resident = costs.BestResident(0, begin, end, available);
            bytes = costs.Cost(0, begin, end, Math.Max(0, resident));
            return resident;
        }

        internal static void ValidateContiguousMap(int[] map, int layers, int devices)
        {
            ArgumentNullException.ThrowIfNull(map);
            if (map.Length != layers)
                throw new ArgumentException("The layer map must cover every layer.", nameof(map));
            if (map[0] != 0 || map[layers - 1] != devices - 1)
                throw new ArgumentException("The layer map must start on device 0 and end on the last device.", nameof(map));
            for (int l = 1; l < layers; l++)
                if (map[l] != map[l - 1] && map[l] != map[l - 1] + 1)
                    throw new ArgumentException("The layer map must be contiguous and visit every device in order.", nameof(map));
        }

        /// <summary>Byte costs of placing a run of layers on one device.</summary>
        private sealed class SpanCosts
        {
            private readonly long[] _layer, _expert;
            private readonly long[] _layerPrefix, _expertPrefix;
            private readonly long _first, _last;
            private readonly Qwen4ExpSpanScratch _scratch;
            private readonly bool[] _forced;
            private readonly int _devices;

            internal SpanCosts(long[] layer, long[] expert, long first, long last,
                Qwen4ExpSpanScratch scratch, bool[] forced, int devices)
            {
                _layer = layer; _expert = expert; _first = first; _last = last;
                _scratch = scratch; _forced = forced; _devices = devices;
                int n = layer.Length;
                _layerPrefix = new long[n + 1];
                _expertPrefix = new long[n + 1];
                for (int l = 0; l < n; l++)
                {
                    _layerPrefix[l + 1] = checked(_layerPrefix[l] + layer[l]);
                    _expertPrefix[l + 1] = checked(_expertPrefix[l] + expert[l]);
                }
            }

            private long Fixed(int device)
                => (device == 0 ? _first : 0) + (device == _devices - 1 ? _last : 0);

            /// <summary>Bytes device <paramref name="device"/> needs for [begin, end)
            /// with the trailing <paramref name="resident"/> layers' experts on it.
            /// Under a forced host set, <paramref name="resident"/> is ignored.</summary>
            internal long Cost(int device, int begin, int end, int resident)
            {
                long bytes = checked(Fixed(device) + _layerPrefix[end] - _layerPrefix[begin]);
                int hostLayers = 0;
                long largestHost = 0;
                if (_forced != null)
                {
                    for (int l = begin; l < end; l++)
                    {
                        if (_forced[l]) { hostLayers++; largestHost = Math.Max(largestHost, _expert[l]); }
                        else bytes = checked(bytes + _expert[l]);
                    }
                }
                else
                {
                    int split = end - resident;
                    bytes = checked(bytes + _expertPrefix[end] - _expertPrefix[split]);
                    hostLayers = split - begin;
                    for (int l = begin; l < split; l++) largestHost = Math.Max(largestHost, _expert[l]);
                }
                return checked(bytes + _scratch.Bytes(hostLayers, largestHost));
            }

            /// <summary>Most trailing layers [begin, end) can keep resident within
            /// <paramref name="cap"/> bytes on <paramref name="device"/>; -1 if none.</summary>
            internal int BestResident(int device, int begin, int end, long cap)
            {
                if (_forced != null)
                {
                    int resident = 0;
                    for (int l = begin; l < end; l++) if (!_forced[l]) resident++;
                    return Cost(device, begin, end, 0) <= cap ? resident : -1;
                }
                // Not monotone: routing the FIRST layer to the host buys a whole layer of
                // stream buffer and seam scratch, so each count is checked on its own.
                // Walk the host/resident split forwards so the largest host layer is
                // tracked incrementally - this runs inside the placement search.
                long dense = checked(Fixed(device) + _layerPrefix[end] - _layerPrefix[begin]);
                long largestHost = 0;
                for (int split = begin; split <= end; split++)
                {
                    if (split > begin) largestHost = Math.Max(largestHost, _expert[split - 1]);
                    long bytes = checked(dense + _expertPrefix[end] - _expertPrefix[split]
                        + _scratch.Bytes(split - begin, largestHost));
                    if (bytes <= cap)
                        return end - split;
                }
                return -1;
            }

            /// <summary>Mark the host-routed layers of [begin, end) and return its cost.</summary>
            internal long Apply(int device, int begin, int end, int resident, bool[] host)
            {
                if (_forced != null)
                {
                    for (int l = begin; l < end; l++) host[l] = _forced[l];
                    return Cost(device, begin, end, 0);
                }
                for (int l = begin; l < end; l++) host[l] = l < end - resident;
                return Cost(device, begin, end, resident);
            }

            /// <summary>Largest total of resident layers over every contiguous,
            /// non-empty assignment with device d spending at most
            /// <paramref name="fraction"/> x capacity[d]; -1 when none fits. When
            /// <paramref name="ends"/> is given it receives each device's run end.</summary>
            internal int Solve(long[] capacity, double fraction, int[] ends)
            {
                int n = _layer.Length, devices = capacity.Length;
                // best[d, m]: most resident layers with devices 0..d holding [0, m).
                var best = new int[devices, n + 1];
                var from = new int[devices, n + 1];
                for (int d = 0; d < devices; d++)
                    for (int m = 0; m <= n; m++) best[d, m] = -1;
                for (int d = 0; d < devices; d++)
                {
                    long cap = (long)(capacity[d] * fraction);
                    // Device d needs at least one layer, and must leave one for each later device.
                    int minEnd = d + 1, maxEnd = n - (devices - 1 - d);
                    for (int m = minEnd; m <= maxEnd; m++)
                    {
                        if (d == devices - 1 && m != n) continue;
                        int lo = d == 0 ? 0 : d, hi = d == 0 ? 0 : m - 1;
                        for (int l = lo; l <= hi; l++)
                        {
                            int before = d == 0 ? 0 : best[d - 1, l];
                            if (before < 0) continue;
                            int resident = BestResident(d, l, m, cap);
                            if (resident < 0) continue;
                            if (before + resident > best[d, m])
                            {
                                best[d, m] = before + resident;
                                from[d, m] = l;
                            }
                        }
                    }
                }
                int total = best[devices - 1, n];
                if (total >= 0 && ends != null)
                {
                    for (int d = devices - 1, m = n; d >= 0; d--)
                    {
                        ends[d] = m;
                        m = from[d, m];
                    }
                }
                return total;
            }
        }
    }
}
