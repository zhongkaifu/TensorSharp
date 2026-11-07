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
using System.Diagnostics.Metrics;
using System.Threading;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Runtime.Telemetry
{
    /// <summary>
    /// Every instrument TensorSharp records on <see cref="TensorSharpTelemetry.Meter"/>.
    ///
    /// <para>
    /// Recording on an instrument nobody listens to is a no-op, so call sites record
    /// unconditionally. The observable instruments read counters the engine already keeps,
    /// at collection time, so they add nothing to the hot path.
    /// </para>
    /// </summary>
    public static class TensorSharpMetrics
    {
        // Bucket boundaries recommended by the GenAI semantic conventions.
        private static readonly double[] s_requestDurationBuckets =
            { 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92 };
        private static readonly double[] s_timeToFirstTokenBuckets =
            { 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0 };
        private static readonly double[] s_timePerOutputTokenBuckets =
            { 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5 };
        private static readonly double[] s_stepBuckets =
            { 0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12 };
        private static readonly int[] s_batchBuckets = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
        // Video clips run from seconds (a distilled 9-frame test) to the better part of an hour.
        private static readonly double[] s_longBuckets =
            { 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200 };

        private static Meter M => TensorSharpTelemetry.Meter;

        // GenAI server metrics ----------------------------------------------------------

        public static readonly Histogram<double> RequestDuration = M.CreateHistogram(
            "gen_ai.server.request.duration", "s",
            "Time from the start of a generation request to its last token, including queueing.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_requestDurationBuckets });

        public static readonly Histogram<double> TimeToFirstToken = M.CreateHistogram(
            "gen_ai.server.time_to_first_token", "s",
            "Time from the start of a generation request to its first output token.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_timeToFirstTokenBuckets });

        public static readonly Histogram<double> TimePerOutputToken = M.CreateHistogram(
            "gen_ai.server.time_per_output_token", "s",
            "Wall time per output token after the first, for completed requests.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_timePerOutputTokenBuckets });

        public static readonly Counter<long> Tokens = M.CreateCounter<long>(
            "tensorsharp.tokens", "{token}",
            "Prompt and output tokens, by tensorsharp.token.type (input, output).");

        public static readonly Counter<long> TokensCacheRead = M.CreateCounter<long>(
            "tensorsharp.tokens.cache_read", "{token}",
            "Prompt tokens served from the prefix cache; a subset of input tokens.");

        // Engine ------------------------------------------------------------------------

        public static readonly Histogram<double> EngineStepDuration = M.CreateHistogram(
            "tensorsharp.engine.step.duration", "s",
            "Wall time of one engine step (one prefill chunk or one batched decode).",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_stepBuckets });

        public static readonly Histogram<int> EngineBatchSize = M.CreateHistogram(
            "tensorsharp.engine.batch.size", "{sequence}",
            "Sequences in one engine step.",
            advice: new InstrumentAdvice<int> { HistogramBucketBoundaries = s_batchBuckets });

        public static readonly Counter<long> SpeculativeTokensDrafted = M.CreateCounter<long>(
            "tensorsharp.speculative.tokens.drafted", "{token}", "Tokens a speculator proposed.");

        public static readonly Counter<long> SpeculativeTokensAccepted = M.CreateCounter<long>(
            "tensorsharp.speculative.tokens.accepted", "{token}", "Proposed tokens the target model accepted.");

        // Models ------------------------------------------------------------------------

        public static readonly Histogram<double> ModelLoadDuration = M.CreateHistogram(
            "tensorsharp.model.load.duration", "s", "Time to load a model, by outcome.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_longBuckets });

        // Agent -------------------------------------------------------------------------

        public static readonly Histogram<double> ToolDuration = M.CreateHistogram(
            "tensorsharp.tool.duration", "s", "Time to run one tool call, by tool and outcome.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_requestDurationBuckets });

        // Media -------------------------------------------------------------------------

        public static readonly Histogram<double> VideoGenerationDuration = M.CreateHistogram(
            "tensorsharp.video.generation.duration", "s",
            "Time to generate and save one clip, including the wait for the GPU, by outcome.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_longBuckets });

        public static readonly Histogram<double> VideoPhaseDuration = M.CreateHistogram(
            "tensorsharp.video.phase.duration", "s",
            "Time in one phase (and stage) of a clip: text-encode, denoise, vae-decode, export, ...",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_longBuckets });

        public static readonly Histogram<double> VideoDenoiseStepDuration = M.CreateHistogram(
            "tensorsharp.video.denoise.step.duration", "s",
            "Time per denoising step, by model and stage.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_longBuckets });

        public static readonly Counter<long> VideoDitLoads = M.CreateCounter<long>(
            "tensorsharp.video.dit.loads", "{load}",
            "Times a clip found its denoiser resident (cached=true) or had to build it (cached=false).");

        public static readonly Histogram<double> ImageGenerationDuration = M.CreateHistogram(
            "tensorsharp.image.generation.duration", "s",
            "Time to generate one image request, by outcome.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_longBuckets });

        // Observable --------------------------------------------------------------------

        private static readonly ConcurrentDictionary<InferenceEngine, string> s_engines = new();
        private static readonly ConcurrentDictionary<string, Func<int>> s_mediaJobCounters = new();
        private static readonly ConcurrentDictionary<string, Func<int>> s_videoQueueCounters = new();
        private static readonly ConcurrentDictionary<string, Func<IEnumerable<GpuMemoryReading>>> s_gpuMemorySources = new();

        static TensorSharpMetrics()
        {
            M.CreateObservableGauge("tensorsharp.engine.requests.running",
                () => EngineMeasurements(e => e.RunningCount), "{request}", "Sequences the engine is running.");
            M.CreateObservableGauge("tensorsharp.engine.requests.waiting",
                () => EngineMeasurements(e => e.WaitingCount), "{request}", "Sequences waiting for the engine.");
            // Monotonic across model swaps: an engine's totals are folded into s_retired when it
            // is disposed, so a rebuilt engine starting from zero does not look like a reset.
            M.CreateObservableCounter("tensorsharp.engine.steps",
                () => EngineTotals(e => e.TotalStepsRun, r => r.Steps), "{step}", "Engine steps run.");
            M.CreateObservableCounter("tensorsharp.engine.forward.time",
                () => EngineTotals(e => e.TotalForwardTime.TotalSeconds, r => r.ForwardSeconds), "s", "Time spent in model forwards.");
            M.CreateObservableGauge("tensorsharp.kv_cache.blocks.used",
                () => EngineMeasurements(e => { var s = e.PoolStats; return s.totalBlocks - s.freeBlocks; }),
                "{block}", "Paged KV cache blocks in use.");
            M.CreateObservableGauge("tensorsharp.kv_cache.blocks.total",
                () => EngineMeasurements(e => e.PoolStats.totalBlocks), "{block}", "Paged KV cache blocks in the pool.");
            M.CreateObservableGauge("tensorsharp.media.generations.active",
                () => CounterMeasurements(s_mediaJobCounters), "{generation}", "Image and video generations in progress.");
            M.CreateObservableGauge("tensorsharp.video.queue.waiting",
                () => CounterMeasurements(s_videoQueueCounters), "{clip}", "Clips waiting for the one-clip-at-a-time lock.");
            s_gpuMemoryUsed = M.CreateObservableGauge("tensorsharp.gpu.memory.used",
                () => GpuMeasurements(r => r.UsedBytes), "By", "Device memory in use (or cached by the allocator).");
            M.CreateObservableGauge("tensorsharp.gpu.memory.total",
                () => GpuMeasurements(r => r.TotalBytes), "By", "Device memory on the device.");
            M.CreateObservableGauge("tensorsharp.gpu.memory.peak",
                () => GpuMeasurements(r => r.PeakBytes), "By", "Peak device memory the allocator has held.");
        }

        /// <summary>Touch the class so its observable instruments exist before an exporter starts.</summary>
        public static void EnsureInitialized() { }

        private static readonly ObservableGauge<long> s_gpuMemoryUsed;

        /// <summary>
        /// Whether anything collects the GPU memory gauges (a meter provider that exports
        /// metrics, or a test listener). Device-memory snapshots are native queries; with nobody
        /// collecting them they are skipped entirely, so telemetry off (or metrics off) costs
        /// nothing here.
        /// </summary>
        public static bool GpuMemoryObserved => s_gpuMemoryUsed.Enabled;

        private static Action? s_engineStepObserver;
        private static long s_lastEngineStepObservation;
        private static readonly long s_engineStepObserverInterval = System.Diagnostics.Stopwatch.Frequency / 4;

        /// <summary>
        /// A callback the inference engine's worker thread runs between steps, at most four
        /// times a second: the place to take device-memory snapshots for LLM serving, on the
        /// thread that owns the backend's work (TensorSharp.Models registers
        /// <c>GpuMemoryTelemetry.Snapshot</c> here). Null removes it. It must be cheap and must
        /// not throw; exceptions are swallowed.
        /// </summary>
        public static void SetEngineStepObserver(Action? observer) => Volatile.Write(ref s_engineStepObserver, observer);

        /// <summary>Called by the engine worker after each step; a no-op unless an observer is set.</summary>
        internal static void OnEngineStep()
        {
            Action? observer = Volatile.Read(ref s_engineStepObserver);
            if (observer is null)
                return;
            long now = System.Diagnostics.Stopwatch.GetTimestamp();
            long last = Interlocked.Read(ref s_lastEngineStepObservation);
            if (now - last < s_engineStepObserverInterval || Interlocked.CompareExchange(ref s_lastEngineStepObservation, now, last) != last)
                return;
            try { observer(); }
            catch (Exception) { }
        }

        // Moving an engine from live to retired, and reading both for a cumulative total, happen
        // under one lock: otherwise a collection that read the retired totals just before an
        // unload, and the live engines just after it, would miss that engine entirely and
        // report a total lower than the previous one (a false counter reset). The work under
        // the lock is a few counter reads.
        private static readonly object s_engineGate = new();

        /// <summary>Called by <see cref="InferenceEngine"/> on construction.</summary>
        internal static void RegisterEngine(InferenceEngine engine, string? architecture)
        {
            lock (s_engineGate)
                s_engines[engine] = string.IsNullOrEmpty(architecture) ? "unknown" : architecture;
        }

        /// <summary>Called by <see cref="InferenceEngine"/> from <c>Dispose</c>.</summary>
        internal static void UnregisterEngine(InferenceEngine engine)
        {
            lock (s_engineGate)
            {
                if (!s_engines.TryRemove(engine, out string? architecture))
                    return;
                long steps = engine.TotalStepsRun;
                double forward = engine.TotalForwardTime.TotalSeconds;
                s_retired.AddOrUpdate(architecture,
                    _ => new RetiredTotals(steps, forward),
                    (_, r) => new RetiredTotals(r.Steps + steps, r.ForwardSeconds + forward));
            }
        }

        private readonly record struct RetiredTotals(long Steps, double ForwardSeconds);

        private static readonly ConcurrentDictionary<string, RetiredTotals> s_retired = new();

        private static IEnumerable<Measurement<T>> EngineTotals<T>(Func<InferenceEngine, T> live, Func<RetiredTotals, T> retired)
            where T : struct, System.Numerics.INumber<T>
        {
            var sums = new Dictionary<string, T>(StringComparer.Ordinal);
            lock (s_engineGate)
            {
                foreach (var entry in s_retired)
                    sums[entry.Key] = retired(entry.Value);
                foreach (var entry in s_engines)
                {
                    T value;
                    try { value = live(entry.Key); }
                    catch (ObjectDisposedException) { continue; }
                    sums[entry.Value] = sums.TryGetValue(entry.Value, out T sum) ? sum + value : value;
                }
            }
            var measurements = new List<Measurement<T>>(sums.Count);
            foreach (var pair in sums)
                measurements.Add(new Measurement<T>(pair.Value, new KeyValuePair<string, object?>(TelemetryAttributes.ModelArchitecture, pair.Key)));
            return measurements;
        }

        /// <summary>
        /// Report a count of in-progress media generations. <paramref name="read"/> runs on the
        /// metrics collection thread and must be cheap and thread-safe. Returns a handle that
        /// unregisters it.
        /// </summary>
        public static IDisposable RegisterMediaGenerationCounter(string owner, Func<int> read) =>
            Register(s_mediaJobCounters, owner, read);

        /// <summary>Report how many clips wait for the video lock. Same rules as <see cref="RegisterMediaGenerationCounter"/>.</summary>
        public static IDisposable RegisterVideoQueueCounter(string owner, Func<int> read) =>
            Register(s_videoQueueCounters, owner, read);

        /// <summary>
        /// Report device memory for one backend. <paramref name="read"/> runs on the metrics
        /// collection thread: it must not block behind GPU work (for MLX, return a snapshot
        /// taken elsewhere rather than calling into the MLX worker).
        /// </summary>
        public static IDisposable RegisterGpuMemorySource(string backend, Func<IEnumerable<GpuMemoryReading>> read) =>
            Register(s_gpuMemorySources, backend, read);

        private static IDisposable Register<T>(ConcurrentDictionary<string, T> map, string key, T value)
        {
            map[key] = value;
            return new Registration(() => map.TryRemove(new KeyValuePair<string, T>(key, value)));
        }

        private static IEnumerable<Measurement<T>> EngineMeasurements<T>(Func<InferenceEngine, T> read) where T : struct
        {
            foreach (var entry in s_engines)
            {
                T value;
                try { value = read(entry.Key); }
                catch (ObjectDisposedException) { continue; }
                yield return new Measurement<T>(value, new KeyValuePair<string, object?>(TelemetryAttributes.ModelArchitecture, entry.Value));
            }
        }

        private static IEnumerable<Measurement<int>> CounterMeasurements(ConcurrentDictionary<string, Func<int>> map)
        {
            foreach (var entry in map)
            {
                int value;
                try { value = entry.Value(); }
                catch (Exception) { continue; }
                yield return new Measurement<int>(value, new KeyValuePair<string, object?>("tensorsharp.owner", entry.Key));
            }
        }

        private static IEnumerable<Measurement<long>> GpuMeasurements(Func<GpuMemoryReading, long?> pick)
        {
            foreach (var source in s_gpuMemorySources)
            {
                IEnumerable<GpuMemoryReading> readings;
                try { readings = source.Value() ?? Array.Empty<GpuMemoryReading>(); }
                catch (Exception) { continue; }
                foreach (var reading in readings)
                {
                    if (pick(reading) is not long value)
                        continue;
                    yield return new Measurement<long>(value,
                        new KeyValuePair<string, object?>(TelemetryAttributes.Backend, source.Key),
                        new KeyValuePair<string, object?>(TelemetryAttributes.GpuDevice, reading.Device));
                }
            }
        }

        private sealed class Registration : IDisposable
        {
            private Action? _remove;
            public Registration(Action remove) => _remove = remove;
            public void Dispose() => System.Threading.Interlocked.Exchange(ref _remove, null)?.Invoke();
        }
    }

    /// <summary>One device's memory figures; a figure a backend cannot report is <c>null</c>.</summary>
    public readonly record struct GpuMemoryReading(int Device, long? UsedBytes, long? TotalBytes, long? PeakBytes);
}
