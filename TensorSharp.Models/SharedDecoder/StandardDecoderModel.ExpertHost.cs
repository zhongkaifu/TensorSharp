// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
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

            public void Record(int layer, ReadOnlySpan<int> ids)
            {
                if (_m.TryGetLayerExperts(layer, out StackedExpertWeights gate, out _, out _))
                    GgmlBasicOps.MoeRoutingCount(gate.Data, _m._numExperts, ids);
            }

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
