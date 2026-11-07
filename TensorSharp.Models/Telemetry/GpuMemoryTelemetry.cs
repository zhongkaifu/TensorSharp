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
using System.Diagnostics;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Models.Telemetry
{
    /// <summary>
    /// Device memory for the <c>tensorsharp.gpu.memory.used</c> / <c>.total</c> / <c>.peak</c>
    /// gauges, for the devices the tracked models' allocators are on.
    ///
    /// <para>
    /// The gauges are read on the metrics collection thread, at whatever moment an exporter
    /// collects, so no native query runs there. <see cref="Snapshot"/> takes each backend's
    /// figures on a thread that is already doing that backend's work, and the gauges report
    /// the latest snapshot. Two places call it: the inference engine's worker between steps
    /// (the hosts register it with <see cref="TensorSharpMetrics.SetEngineStepObserver"/> when
    /// telemetry is on), and the video helper at phase and stage boundaries on the thread that
    /// reports progress. The server's model lifecycle <see cref="Track"/>s every model it
    /// loads; the video helper tracks the model of each clip.
    /// </para>
    /// <para>
    /// The query is picked by the type of the model's allocator, not by the backend it was
    /// asked for (LTX runs a <c>ggml_cuda</c> request on direct CUDA, for one). Each has its
    /// own reason for being taken where it is:
    /// </para>
    /// <list type="bullet">
    /// <item><b>ggml</b> (<see cref="GgmlAllocator"/>, <see cref="GgmlBasicOps.TryGetDeviceMemoryInfo"/>):
    /// the native call starts with <c>ensure_backend()</c>, which in a process that has not
    /// brought a GGML backend up yet latches Metal as the process's backend type and tries to
    /// initialize it, so a later <c>ggml_cuda</c> load would be refused; and
    /// <c>TSGgml_RecreateBackend</c> / shutdown free <c>g_backend</c> under no lock a reader
    /// takes. The query itself (<c>cudaMemGetInfo</c>, Metal's working-set counters) does not
    /// wait for a running graph, so it is only asked for while a tracked model's allocator is
    /// on a GGML GPU backend (that model's <see cref="GgmlContext"/> brought the backend up),
    /// never through a model being unloaded, and only from a thread the host already
    /// serializes with backend teardown: the engine's worker is stopped before an unload or a
    /// <c>RecreateBackend</c>. It reports the calling thread's rank, which between engine steps
    /// is rank 0.</item>
    /// <item><b>CUDA</b> (<see cref="CudaAllocator.GetMemoryInfo"/>): <c>cuMemGetInfo</c> makes the
    /// device's primary context current on the calling thread (the one the engine's worker
    /// already uses for a direct CUDA model), and an allocator disposed under a collection
    /// would hand the driver a released context. The allocator pool counters
    /// (<see cref="CudaAllocatorStats.CachedBytes"/>) are safe anywhere but count only idle pooled
    /// blocks (512 MB + 1 GB at most by default), so they read near zero while a 22B denoiser holds
    /// most of the card; the gauges report device bytes in use instead.</item>
    /// <item><b>MLX</b> (<see cref="MlxBackend.GetMemorySnapshot"/>): every query goes through
    /// <c>MlxWorker.Shared.Invoke</c>, which queues behind the running MLX operation (on MLX a
    /// video clip is one <c>Invoke</c>, minutes long). The engine's worker is a thread of its
    /// own, not the MLX worker, so an <c>Invoke</c> there would stall a step. On the MLX worker
    /// <c>Invoke</c> runs inline and the figure is read at once (MLX video models report
    /// progress there); anywhere else the read is queued with <c>MlxWorker.Dispatch</c>, which
    /// does not wait, and the gauges report it once the worker has run it.</item>
    /// </list>
    /// <para>
    /// Peak is the highest in-use figure any snapshot has seen (MLX reports its own). A figure
    /// stops being reported once no tracked model on that backend is alive, and a model stops
    /// counting once its unload begins (<see cref="ModelBase.IsRetiring"/>). Readings are for
    /// device 0 (ggml: the calling thread's rank, normally rank 0), except CUDA, which reports
    /// each tracked allocator's device.
    /// </para>
    /// </summary>
    public static class GpuMemoryTelemetry
    {
        // The value of tensorsharp.backend, spelled as --backend takes it. The CPU backends
        // have no device memory to report.
        private const string GgmlCuda = "ggml_cuda";
        private const string GgmlMetal = "ggml_metal";
        private const string GgmlVulkan = "ggml_vulkan";
        private const string Cuda = "cuda";
        private const string Mlx = "mlx";
        private static readonly string[] s_backends = { GgmlCuda, GgmlMetal, GgmlVulkan, Cuda, Mlx };

        // Phase and stage boundaries can come close together (LTX reports one per reference
        // video); a snapshot is a few native queries, but nothing needs them more often.
        private static readonly TimeSpan s_minInterval = TimeSpan.FromMilliseconds(250);

        private readonly record struct Tracked(WeakReference<ModelBase> Model, string Backend);
        private readonly record struct Reading(string Backend, GpuMemoryReading Value);

        private static readonly object s_gate = new();
        private static readonly List<Tracked> s_tracked = new();
        private static readonly Dictionary<(string Backend, int Device), long> s_peaks = new();
        private static readonly object s_snapshotGate = new();
        // The ggml and CUDA figures, replaced by each snapshot.
        private static Reading[] s_readings = Array.Empty<Reading>();
        // The MLX figure, written on the MLX worker (see RequestMlxReading).
        private static Reading[] s_mlxReadings = Array.Empty<Reading>();
        private static int s_mlxReadQueued;
        private static long s_lastSnapshot;
        private static int s_registered;

        /// <summary>
        /// Register the GPU memory sources with <see cref="TensorSharpMetrics"/>. Idempotent and
        /// cheap; the sources report nothing until a tracked model has been snapshotted.
        /// </summary>
        public static void EnsureRegistered()
        {
            if (Interlocked.Exchange(ref s_registered, 1) != 0)
                return;
            // Process lifetime: the registrations are never disposed.
            foreach (string backend in s_backends)
                TensorSharpMetrics.RegisterGpuMemorySource(backend, () => Read(backend));
        }

        /// <summary>
        /// Report device memory for the device <paramref name="model"/>'s allocator is on. Held
        /// weakly, so an unloaded model drops out once it is collected. Idempotent, cheap and
        /// never throws: a host may call it after every load, telemetry on or off (nothing is
        /// queried until something calls <see cref="Snapshot"/>). A model on host memory (cpu,
        /// ggml_cpu) is not tracked.
        /// </summary>
        public static void Track(ModelBase model)
        {
            if (model == null)
                return;
            try
            {
                string backend = SourceOf(model.DirectAllocatorForTelemetry);
                if (backend == null)
                    return;
                EnsureRegistered();
                lock (s_gate)
                {
                    for (int i = s_tracked.Count - 1; i >= 0; i--)
                    {
                        if (!s_tracked[i].Model.TryGetTarget(out ModelBase existing))
                            s_tracked.RemoveAt(i);
                        else if (ReferenceEquals(existing, model))
                            return;
                    }
                    s_tracked.Add(new Tracked(new WeakReference<ModelBase>(model), backend));
                }
            }
            catch (Exception)
            {
                // Telemetry must never fail a model load.
            }
        }

        /// <summary>
        /// Refresh the figures the gauges report. Call it only from a thread that may use the
        /// tracked models' backend right now: one running their work (the engine's worker
        /// between steps), or holding the host's GPU lock. It never blocks (a snapshot already
        /// in progress on another thread wins, and MLX is asked without waiting), runs at most
        /// every 250 ms, and never throws.
        /// </summary>
        public static void Snapshot()
        {
            // Native device queries for gauges nobody collects would be work for nothing:
            // with telemetry (or metrics) off, every video phase and engine step returns here.
            if (!TensorSharpMetrics.GpuMemoryObserved)
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
                Tracked[] tracked;
                lock (s_gate)
                {
                    s_tracked.RemoveAll(t => !t.Model.TryGetTarget(out _));
                    tracked = s_tracked.ToArray();
                }

                var readings = new List<Reading>();
                bool ggmlAsked = false, mlxAsked = false;
                HashSet<int> cudaDevices = null;
                foreach (Tracked t in tracked)
                {
                    // An unload begins before the model's allocator is freed (and, with
                    // RecreateBackend, the ggml backend after it), so never ask through it.
                    if (!t.Model.TryGetTarget(out ModelBase model) || model.IsRetiring)
                        continue;
                    switch (model.DirectAllocatorForTelemetry)
                    {
                        case GgmlAllocator ggml when !ggmlAsked && SourceOf(ggml) is string ggmlBackend:
                            // One GGML backend per process, so one device-wide query covers
                            // every model on it.
                            ggmlAsked = true;
                            if (TryReadGgml(out long free, out long total))
                                readings.Add(InUse(ggmlBackend, 0, total - free, total));
                            break;
                        case CudaAllocator cuda:
                            if ((cudaDevices ??= new HashSet<int>()).Add(cuda.DeviceId)
                                && TryReadCuda(cuda, out long cudaFree, out long cudaTotal))
                                readings.Add(InUse(Cuda, cuda.DeviceId, cudaTotal - cudaFree, cudaTotal));
                            break;
                        case MlxAllocator when !mlxAsked:
                            mlxAsked = true;
                            RequestMlxReading();
                            break;
                    }
                }
                Volatile.Write(ref s_readings, readings.ToArray());
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

        private static Reading InUse(string backend, int device, long used, long total)
        {
            if (used < 0) used = 0;
            long peak;
            lock (s_gate)
            {
                peak = s_peaks.TryGetValue((backend, device), out long seen) && seen > used ? seen : used;
                s_peaks[(backend, device)] = peak;
            }
            return new Reading(backend, new GpuMemoryReading(device, used, total > 0 ? total : null, peak));
        }

        private static bool TryReadGgml(out long freeBytes, out long totalBytes)
        {
            try
            {
                return GgmlBasicOps.TryGetDeviceMemoryInfo(out freeBytes, out totalBytes) && totalBytes > 0;
            }
            catch (Exception)
            {
                freeBytes = totalBytes = 0;
                return false;
            }
        }

        private static bool TryReadCuda(CudaAllocator allocator, out long freeBytes, out long totalBytes)
        {
            try
            {
                (freeBytes, totalBytes) = allocator.GetMemoryInfo();
                return totalBytes > 0;
            }
            catch (Exception)
            {
                // Disposed with its model, or the driver refused: no figure this time.
                freeBytes = totalBytes = 0;
                return false;
            }
        }

        /// <summary>
        /// Refresh the MLX figure without waiting for the MLX worker: inline on the worker,
        /// otherwise queued to it with <c>Dispatch</c>, which returns at once. One read is queued
        /// at a time, so a worker busy with a long clip gets one, not one per snapshot.
        /// </summary>
        private static void RequestMlxReading()
        {
            MlxWorker worker = MlxWorker.Shared;
            if (worker.IsOnWorkerThread)
            {
                ReadMlx();
                return;
            }
            if (Interlocked.Exchange(ref s_mlxReadQueued, 1) != 0)
                return;
            try
            {
                worker.Dispatch(static () =>
                {
                    try { ReadMlx(); }
                    finally { Volatile.Write(ref s_mlxReadQueued, 0); }
                });
            }
            catch (Exception)
            {
                // The worker has shut down, so nothing will run the read.
                Volatile.Write(ref s_mlxReadQueued, 0);
            }
        }

        // On the MLX worker, where GetMemorySnapshot's Invoke runs inline.
        private static void ReadMlx()
        {
            try
            {
                MlxMemorySnapshot mlx = MlxBackend.GetMemorySnapshot();
                Volatile.Write(ref s_mlxReadings,
                    new[] { new Reading(Mlx, new GpuMemoryReading(0, (long)mlx.ActiveBytes, null, (long)mlx.PeakBytes)) });
            }
            catch (Exception)
            {
                // Keep the previous figure.
            }
        }

        /// <summary>
        /// The <c>tensorsharp.backend</c> value of the device <paramref name="allocator"/> allocates
        /// on, or null for host memory (cpu, ggml_cpu) and for a model with no allocator (a test fake).
        /// </summary>
        private static string SourceOf(IAllocator allocator) => allocator switch
        {
            CudaAllocator => Cuda,
            MlxAllocator => Mlx,
            GgmlAllocator ggml => ggml.Context.BackendType switch
            {
                GgmlBackendType.Cuda => GgmlCuda,
                GgmlBackendType.Metal => GgmlMetal,
                GgmlBackendType.Vulkan => GgmlVulkan,
                _ => null,
            },
            _ => null,
        };

        // Collected models drop out of s_tracked; one being unloaded stops counting here.
        private static bool IsLive(Tracked t) => t.Model.TryGetTarget(out ModelBase model) && !model.IsRetiring;

        // Runs on the metrics collection thread: reads the published snapshot and the weak
        // references, never a backend.
        private static IEnumerable<GpuMemoryReading> Read(string backend)
        {
            bool live = false;
            lock (s_gate)
            {
                foreach (Tracked t in s_tracked)
                {
                    if (t.Backend == backend && IsLive(t))
                    {
                        live = true;
                        break;
                    }
                }
            }
            if (!live)
                return Array.Empty<GpuMemoryReading>();
            List<GpuMemoryReading> result = null;
            Reading[] published = backend == Mlx ? Volatile.Read(ref s_mlxReadings) : Volatile.Read(ref s_readings);
            foreach (Reading reading in published)
            {
                if (reading.Backend == backend)
                    (result ??= new List<GpuMemoryReading>()).Add(reading.Value);
            }
            return (IEnumerable<GpuMemoryReading>)result ?? Array.Empty<GpuMemoryReading>();
        }
    }
}
