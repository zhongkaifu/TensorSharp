// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    public abstract partial class StandardDecoderModel
    {
        /// <summary>The runtime's expert host on the tensor executors (null on the GGML backends, whose plugin runs its
        /// own host seams natively).</summary>
        private ModelExpertHost _expertHost;

        /// <summary>
        /// Expert offload for every tensor executor (direct CUDA, MLX, the managed CPU), one implementation whatever the
        /// backend: an offloaded layer's routed experts stay in host memory (the GGUF mapping, read through the explicit
        /// SSD reader where it is registered), its tier's experts are kept on the executor's device, and the host computes
        /// the routes the tier does not hold with ggml's CPU kernels (the kernel, dot products and summation order of an
        /// offloaded GGML layer). A step wider than the host kernel takes - a prompt chunk - runs every route on the
        /// device, the experts outside the tier read in for it and dropped after.
        /// </summary>
        private sealed class ModelExpertHost : IGraphExpertHost
        {
            /// <summary>Rows one host call computes (TSGgml_HostExpertRows' bound): a wider step streams its experts.</summary>
            internal const int HostRowsPerCall = 16;

            private readonly StandardDecoderModel _m;
            private readonly bool _deviceIsHost;
            private readonly bool[][] _tier;

            private readonly int _tailRank, _tailMaxCachedPercent;

            public ModelExpertHost(StandardDecoderModel model, bool deviceIsHost)
            {
                _m = model;
                _deviceIsHost = deviceIsHost;
                _tier = new bool[model.Config.NumLayers][];
                _recent = new float[model.Config.NumLayers][];
                (_tailRank, _tailMaxCachedPercent) = model.RouteTailSkip;
            }

            /// <summary>Off where every expert is the device's or the layer's tier holds the switch's share of them (the
            /// misses are rare there, and the skip would only cost quality).</summary>
            public int TailSkipRank(int layer)
            {
                if (_deviceIsHost || _tailRank <= 0) return 0;
                if (_tier[layer] is { } held)
                {
                    int count = 0;
                    foreach (bool h in held) if (h) count++;
                    if (count * 100L >= (long)_tailMaxCachedPercent * held.Length) return 0;
                }
                return _tailRank;
            }

            /// <summary>Keep <paramref name="experts"/> of <paramref name="layer"/> on the executor's device.</summary>
            public void SetTier(int layer, int[] experts)
            {
                if (experts == null || experts.Length == 0) { _tier[layer] = null; return; }
                var held = new bool[_m._numExperts];
                foreach (int e in experts)
                    if (e >= 0 && e < held.Length) held[e] = true;
                _tier[layer] = held;
            }

            /// <summary>On an executor whose device is the host every expert is where the device reads it.</summary>
            public bool IsResident(int layer, int expert)
                => _deviceIsHost || (_tier[layer] is { } held && expert >= 0 && expert < held.Length && held[expert]);

            public bool StreamsStep(int layer, int rows)
            {
                bool stream = !_deviceIsHost && rows > HostRowsPerCall;
                if (stream) System.Threading.Interlocked.Increment(ref StreamedSteps);
                return stream;
            }

            /// <summary>Routes the host computed and steps whose experts were streamed (tests).</summary>
            internal long HostRoutes, StreamedSteps;

            public void Record(int layer, ReadOnlySpan<int> ids, float adaptWeight = 1f)
            {
                if (_m.TryGetLayerExperts(layer, out StackedExpertWeights gate, out _, out _))
                    GgmlBasicOps.MoeRoutingCount(gate.Data, _m._numExperts, ids);
                if (!_adapt || _tier[layer] == null) return;
                lock (_recentSync)
                {
                    float[] recent = _recent[layer] ??= new float[_m._numExperts];
                    foreach (int e in ids)
                        if (e >= 0 && e < recent.Length) recent[e] += adaptWeight;
                }
            }

            // ---- adaptive tiers: each tiered layer's decayed recent route counts (decode steps count one a route, a prompt
            // chunk's routes PromptAdaptRows rows' worth in all, as the Qwen3.8 engines weighed them).

            private readonly bool _adapt = Environment.GetEnvironmentVariable("TS_MOE_TIER_ADAPT") != "0";
            private readonly float[][] _recent;
            private readonly object _recentSync = new();

            /// <summary>A prompt chunk's selections weigh as this many rows' worth of decode selections.</summary>
            internal const float PromptAdaptRows = 128f;

            public IReadOnlyList<(int Layer, int Out, int In)> ProposeSwaps(int budget)
            {
                var trades = new List<(float Gain, int Layer, int Out, int In)>();
                if (!_adapt || budget <= 0) return Array.Empty<(int, int, int)>();
                lock (_recentSync)
                {
                    for (int l = 0; l < _tier.Length; l++)
                    {
                        bool[] held = _tier[l];
                        float[] recent = _recent[l];
                        if (held == null || recent == null) continue;
                        // The layer's hottest experts outside the tier against its coldest members, pairwise.
                        var outside = new List<int>();
                        var inside = new List<int>();
                        for (int e = 0; e < held.Length; e++) (held[e] ? inside : outside).Add(e);
                        outside.Sort((a, b) => recent[b].CompareTo(recent[a]));
                        inside.Sort((a, b) => recent[a].CompareTo(recent[b]));
                        for (int i = 0; i < Math.Min(outside.Count, inside.Count); i++)
                        {
                            float hot = recent[outside[i]], cold = recent[inside[i]];
                            if (hot <= 2f * cold + 1f) break;
                            trades.Add((hot - cold, l, inside[i], outside[i]));
                        }
                        // Decayed each round: recent routing outweighs old.
                        for (int e = 0; e < recent.Length; e++) recent[e] *= 0.5f;
                    }
                }
                trades.Sort((a, b) => b.Gain.CompareTo(a.Gain));
                var result = new List<(int, int, int)>(Math.Min(budget, trades.Count));
                for (int i = 0; i < trades.Count && result.Count < budget; i++)
                    result.Add((trades[i].Layer, trades[i].Out, trades[i].In));
                return result;
            }

            public void CommitSwap(int layer, int outOf, int into)
            {
                if (_tier[layer] is not { } held) return;
                held[outOf] = false;
                held[into] = true;
                System.Threading.Interlocked.Increment(ref Swaps);
            }

            /// <summary>Trades adaptive tiers made (tests, telemetry).</summary>
            internal long Swaps;

            public unsafe bool Compute(int layer, ReadOnlySpan<float> x, ReadOnlySpan<int> ids, ReadOnlySpan<float> weights,
                ReadOnlySpan<byte> skip, int rows, Span<float> output)
            {
                if (!_m.TryGetLayerExperts(layer, out StackedExpertWeights gate, out StackedExpertWeights up, out StackedExpertWeights down))
                    return false;
                int hidden = (int)gate.PerExpertNe0, ff = (int)gate.PerExpertNe1, k = ids.Length / rows;
                long hostRoutes = 0;
                foreach (byte s in skip) if (s == 0) hostRoutes++;
                System.Threading.Interlocked.Add(ref HostRoutes, hostRoutes);
                fixed (float* xp = x, wp = weights, op = output)
                fixed (int* ip = ids)
                fixed (byte* sp = skip)
                {
                    for (int r0 = 0; r0 < rows; r0 += HostRowsPerCall)
                    {
                        int n = Math.Min(HostRowsPerCall, rows - r0);
                        if (!GgmlBasicOps.HostExpertRows(gate.Data, up.Data, down.Data, gate.GgmlType, up.GgmlType, down.GgmlType,
                                _m._numExperts, hidden, ff, k, n, xp + (long)r0 * hidden, ip + (long)r0 * k, wp + (long)r0 * k,
                                sp + (long)r0 * k, op + (long)r0 * hidden))
                            return false;
                    }
                }
                return true;
            }
        }

        /// <summary>The route-tail skip's rank (0 off) and the tier share past which a layer computes every route
        /// (TS_MOE_ROUTE_TAIL_SKIP, default 7; TS_MOE_TAIL_SKIP_MAX_CACHED, default 90 percent): the switches ggml's tiers
        /// read. Tests that compare exact sums pass their own.</summary>
        private (int Rank, int MaxCachedPercent) RouteTailSkip
        {
            get
            {
                if (RouteTailSkipForTests is { } fixedSkip) return fixedSkip;
                static int Env(string name, int fallback)
                    => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) ? v : fallback;
                return (Math.Max(0, Env("TS_MOE_ROUTE_TAIL_SKIP", 7)), Math.Clamp(Env("TS_MOE_TAIL_SKIP_MAX_CACHED", 90), 0, 100));
            }
        }

        /// <summary>Test seam: the route-tail skip this model's expert host applies (set before the first offloaded step).</summary>
        internal (int Rank, int MaxCachedPercent)? RouteTailSkipForTests { get; set; }

        /// <summary>The expert host of an offloaded <paramref name="layer"/> on a tensor executor, or null (a resident layer,
        /// or the GGML backends).</summary>
        private ModelExpertHost HostFor(int layer)
        {
            if (IsGgmlBackend || !_anyMoe || !_moeLayer[layer] || !IsExpertLayerOffloaded(layer)) return null;
            if (_expertHost == null)
            {
                _expertHost = new ModelExpertHost(this, deviceIsHost: _backend == BackendType.Cpu);
                // An executor that records host experts waits for them inside its steps: the team is sized for that.
                if (_backend != BackendType.Cpu && GraphExecutor?.Capabilities.RecordsHostExperts == true)
                    MoeCpuOffloadConfig.SizeTeamForDoorbell();
                // The planned tiers (TryPlanTensorExecutorTiers), resident since the preload.
                for (int l = 0; _tierExperts != null && l < _tierExperts.Length; l++)
                    _expertHost.SetTier(l, _tierExperts[l]);
            }
            return _expertHost;
        }

        /// <summary>Routes the expert host computed on the host, and steps it streamed to the device (tests).</summary>
        internal (long HostRoutes, long StreamedSteps) ExpertHostCountsForTests
            => _expertHost == null ? (0, 0) : (_expertHost.HostRoutes, _expertHost.StreamedSteps);

        /// <summary>Test seam: keep <paramref name="experts"/> of <paramref name="layer"/> in the executor's tier.</summary>
        internal void SetExpertTierForTests(int layer, int[] experts) => (HostFor(layer) ?? throw new InvalidOperationException(
            $"layer {layer} is not an offloaded layer on a tensor executor")).SetTier(layer, experts);

        /// <summary>A layer's stacked experts as the graph reads them, carrying the model's expert host.</summary>
        private sealed class HostedExpertSet : IGraphHostedExperts
        {
            private readonly StackedExpertWeights _w;
            private readonly QuantizedWeight[] _members;

            public HostedExpertSet(StackedExpertWeights w, QuantizedWeight[] members, IGraphExpertHost host)
            {
                _w = w;
                _members = members;
                Host = host;
            }

            public IGraphExpertHost Host { get; }
            public int Format => _w.GgmlType;
            public long Ne0 => _w.PerExpertNe0;
            public long Ne1 => _w.PerExpertNe1;
            public long Ne2 => _w.NumExperts;
            public IntPtr Data => _w.Data;
            public IntPtr CacheKey => _w.Data;
            public long Bytes => _w.TotalRawBytes;

            public IGraphWeight Expert(int index) => _members[index]
                ?? throw new NotSupportedException($"expert {index} has no member weight");
        }
    }
}
