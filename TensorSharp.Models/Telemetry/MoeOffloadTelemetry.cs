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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Models.Telemetry
{
    /// <summary>
    /// Feeds the <c>tensorsharp.moe.*</c> instruments (<see cref="TensorSharpMetrics.RegisterMoeOffloadSource"/>): where
    /// routed experts ran (accelerator slots, slots that read them from storage, the host's expert kernel), the host
    /// kernel's time, adaptive residency swaps and expert I/O bytes, for any MoE model whose experts do not all live on
    /// the accelerator: the GGML expert tiers and host seams (every model with <c>--n-cpu-moe</c>, Qwen3.8-Flash-Next's
    /// tiers) and the direct CUDA engines' expert residency.
    ///
    /// <para>
    /// The figures are process totals the native layer and the CUDA backend already keep, monotonic across model
    /// unloads. They are read by <see cref="Snapshot"/> on the engine's thread (the hosts call it with the GPU memory
    /// snapshot after engine steps), at most four times a second and only while something collects the instruments,
    /// so telemetry off costs one property read a step. The collection thread reads only the published snapshot.
    /// </para>
    /// </summary>
    public static class MoeOffloadTelemetry
    {
        private static readonly TimeSpan s_minInterval = TimeSpan.FromMilliseconds(250);
        private static readonly object s_gate = new();
        private static readonly List<(WeakReference<ModelBase> Model, string Architecture)> s_tracked = new();
        private static readonly HashSet<string> s_registered = new();
        private static readonly ConcurrentDictionary<string, MoeOffloadReading> s_latest = new();
        private static readonly object s_snapshotGate = new();
        private static long s_lastSnapshot;

        /// <summary>
        /// Report MoE offload for <paramref name="model"/>'s architecture. Held weakly; idempotent, cheap and never throws.
        /// A model on host memory alone is not tracked (its experts are all host experts by construction).
        /// </summary>
        public static void Track(ModelBase model)
        {
            if (model == null)
                return;
            try
            {
                if (model.DirectAllocatorForTelemetry is not (GgmlAllocator or CudaAllocator))
                    return;
                string architecture = string.IsNullOrEmpty(model.Config?.Architecture) ? "unknown" : model.Config.Architecture;
                lock (s_gate)
                {
                    for (int i = s_tracked.Count - 1; i >= 0; i--)
                    {
                        if (!s_tracked[i].Model.TryGetTarget(out ModelBase existing))
                            s_tracked.RemoveAt(i);
                        else if (ReferenceEquals(existing, model))
                            return;
                    }
                    s_tracked.Add((new WeakReference<ModelBase>(model), architecture));
                    // Process lifetime, as the GPU memory sources.
                    if (s_registered.Add(architecture))
                        TensorSharpMetrics.RegisterMoeOffloadSource(architecture,
                            () => s_latest.TryGetValue(architecture, out MoeOffloadReading r) ? r : null);
                }
            }
            catch (Exception)
            {
                // Telemetry must never fail a model load.
            }
        }

        /// <summary>
        /// Refresh the published figures. Call it from the engine's thread between steps (the hosts' engine step
        /// observer); it never blocks, runs at most every 250 ms and never throws.
        /// </summary>
        public static void Snapshot()
        {
            if (!TensorSharpMetrics.MoeOffloadObserved)
                return;
            long now = Stopwatch.GetTimestamp();
            long last = Volatile.Read(ref s_lastSnapshot);
            if (last != 0 && Stopwatch.GetElapsedTime(last, now) < s_minInterval)
                return;
            if (!Monitor.TryEnter(s_snapshotGate))
                return;
            try
            {
                Volatile.Write(ref s_lastSnapshot, now);
                var live = new List<string>();
                lock (s_gate)
                {
                    s_tracked.RemoveAll(t => !t.Model.TryGetTarget(out _));
                    foreach (var t in s_tracked)
                        if (t.Model.TryGetTarget(out ModelBase m) && !m.IsRetiring && !live.Contains(t.Architecture))
                            live.Add(t.Architecture);
                }
                if (live.Count == 0)
                    return;
                MoeOffloadReading? reading = Read();
                if (reading is not MoeOffloadReading r)
                    return;
                foreach (string architecture in live)
                    s_latest[architecture] = r;
            }
            catch (Exception)
            {
                // Telemetry must never fail a generation.
            }
            finally
            {
                Monitor.Exit(s_snapshotGate);
            }
        }

        /// <summary>The process totals now, or null while nothing has been offloaded.</summary>
        internal static MoeOffloadReading? Read()
        {
            long[] n = null;
            try
            {
                // TensorSharp's own counters in GgmlOps (no backend is touched); the direct CUDA engine's host experts
                // run the same host kernel, so they count there too.
                n = GgmlBasicOps.MoeOffloadCounters();
            }
            catch (Exception)
            {
                // No GGML native library in this process.
            }
            long device = (n?[0] ?? 0) + CudaMoeOffloadCounters.DeviceRoutes;
            long streamed = n?[1] ?? 0, host = n?[7] ?? 0, io = n?[8] ?? 0;
            long swaps = (n?[4] ?? 0) + CudaMoeOffloadCounters.Swaps;
            double hostSeconds = (n?[5] ?? 0) / 1e9;
            if (device == 0 && streamed == 0 && host == 0 && swaps == 0 && io == 0)
                return null;
            return new MoeOffloadReading(device, streamed, host, hostSeconds, swaps, io);
        }
    }
}
