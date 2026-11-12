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

        // Serving signals: what a step's batch and a sequence's cadence looked like, for finding
        // where an engine loses time under concurrent requests.

        private static readonly double[] s_interTokenBuckets =
            { 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 5, 10, 30 };
        private static readonly int[] s_prefillTokenBuckets =
            { 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
        private static readonly double[] s_stageBuckets =
            { 0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024 };

        public static readonly Histogram<double> InterTokenLatency = M.CreateHistogram(
            "tensorsharp.engine.inter_token.duration", "s",
            "Time between a sequence's consecutive output tokens, by the kind of step that emitted the later one. A mixed step that prefills other requests' prompts stretches it for every sequence it runs.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_interTokenBuckets });

        public static readonly Histogram<int> StepPrefillTokens = M.CreateHistogram(
            "tensorsharp.engine.step.prefill_tokens", "{token}",
            "Prompt tokens one step forwarded, by step kind (prefill or mixed).",
            advice: new InstrumentAdvice<int> { HistogramBucketBoundaries = s_prefillTokenBuckets });

        public static readonly Counter<long> PrefixCacheLookups = M.CreateCounter<long>(
            "tensorsharp.prefix_cache.lookups", "{lookup}",
            "Prefix-cache lookups at admission, by outcome (full, partial, miss) and source (where a reused prefix came from, or why a miss reused nothing).");

        public static readonly Histogram<double> EngineStageDuration = M.CreateHistogram(
            "tensorsharp.engine.stage.duration", "s",
            "GPU time of one stage of a sampled forward (TS_Q4E_PERF_SAMPLE on the Qwen3.8-Flash-Next direct-CUDA engine), by stage, step kind and rows.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_stageBuckets });

        public static readonly Histogram<double> EngineStepPhaseDuration = M.CreateHistogram(
            "tensorsharp.engine.step.phase.duration", "s",
            "Host time of one part of a batched decode step, by phase (finish: waiting for the previous step's logits and copying them; sample; issue: queueing the step's GPU work, or the whole forward where the step waits for itself; draft and verify on speculative steps) and rows.",
            advice: new InstrumentAdvice<double> { HistogramBucketBoundaries = s_stageBuckets });

        private static readonly int[] s_acceptedBuckets = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

        public static readonly Histogram<int> SpeculativeAcceptedLength = M.CreateHistogram(
            "tensorsharp.speculative.accepted_length", "{token}",
            "Drafted tokens the target accepted in one verify window, by the window's drafted tokens (tensorsharp.speculative.drafted).",
            advice: new InstrumentAdvice<int> { HistogramBucketBoundaries = s_acceptedBuckets });

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
        private static readonly ConcurrentDictionary<string, Func<MoeOffloadReading?>> s_moeOffloadSources = new();

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
            // MoE expert offload: where routed experts ran and what the host did for them (see MoeOffloadReading).
            s_moeRoutes = M.CreateObservableCounter("tensorsharp.moe.expert.routes", MoeRouteMeasurements, "{route}",
                "Routed-expert selections by where the expert ran (tensorsharp.moe.placement: device, streamed, host).");
            M.CreateObservableCounter("tensorsharp.moe.host.compute.time",
                () => MoeMeasurements(r => r.HostComputeSeconds), "s",
                "Time the host's expert kernel spent computing experts the accelerator did not hold.");
            M.CreateObservableCounter("tensorsharp.moe.expert.swaps",
                () => MoeMeasurements(r => r.Swaps), "{swap}",
                "Experts adaptive residency moved between accelerator slots and the host.");
            M.CreateObservableCounter("tensorsharp.moe.expert.io",
                () => MoeMeasurements(r => r.IoBytes), "By", "Expert bytes read from storage by explicit expert I/O.");

            // Engines that keep sequence slots of their own (IServingTelemetrySource). Per engine:
            // reloading the model starts its counters again.
            M.CreateObservableGauge("tensorsharp.slots",
                () => ServingMeasurements(
                    (s => s.RunningSlots, TelemetryAttributes.SlotState, "running"),
                    (s => s.RetainedSlots, TelemetryAttributes.SlotState, "retained")),
                "{slot}", "Sequence slots, by state: bound to a request, or kept for a conversation's next turn.");
            M.CreateObservableGauge("tensorsharp.slots.memory",
                () => ServingMeasurements((s => s.SlotBytes, null, null)), "By", "Device memory every sequence slot holds.");
            M.CreateObservableGauge("tensorsharp.slots.capacity",
                () => ServingMeasurements((s => s.SlotCapacityTokens, null, null)), "{token}",
                "Positions every sequence slot's caches hold together (capacity, not use).");
            M.CreateObservableCounter("tensorsharp.slots.grows",
                () => ServingMeasurements((s => s.SlotGrows, null, null)), "{grow}", "Times a slot's caches grew.");
            M.CreateObservableCounter("tensorsharp.slots.grow_failures",
                () => ServingMeasurements((s => s.SlotGrowFailures, null, null)), "{failure}",
                "Times a slot could not grow for lack of device memory, even after retained slots were given back.");
            M.CreateObservableCounter("tensorsharp.slots.reclaims",
                () => ServingMeasurements(
                    (s => s.ReclaimsForNewSlot, TelemetryAttributes.SlotReclaimReason, "new_slot"),
                    (s => s.ReclaimsForGrowth, TelemetryAttributes.SlotReclaimReason, "growth")),
                "{slot}", "Retained slots given back, by what needed the memory. A steady rate means conversations evict each other.");
            M.CreateObservableCounter("tensorsharp.slots.rewinds",
                () => ServingMeasurements(
                    (s => s.RewindsToCheckpoint, TelemetryAttributes.RewindRoute, "checkpoint"),
                    (s => s.RewindsWithReplay, TelemetryAttributes.RewindRoute, "replay"),
                    (s => s.RewindsToZero, TelemetryAttributes.RewindRoute, "reset"),
                    (s => s.RewindsRefused, TelemetryAttributes.RewindRoute, "refused")),
                "{rewind}", "Slot rewinds, by route.");
            M.CreateObservableCounter("tensorsharp.slots.rewind.replayed_tokens",
                () => ServingMeasurements((s => s.ReplayedTokens, null, null)), "{token}",
                "Tokens a rewind past its checkpoint forwarded again.");
            M.CreateObservableCounter("tensorsharp.cuda_graph.captures",
                () => ServingMeasurements(
                    (s => s.SoloGraphCaptures, TelemetryAttributes.GraphKind, "solo"),
                    (s => s.BatchedGraphCaptures, TelemetryAttributes.GraphKind, "batched")),
                "{graph}", "CUDA graphs captured, by kind. Captures that keep pace with launches mean the graphs do not get reused.");
            M.CreateObservableCounter("tensorsharp.cuda_graph.launches",
                () => ServingMeasurements(
                    (s => s.SoloGraphLaunches, TelemetryAttributes.GraphKind, "solo"),
                    (s => s.BatchedGraphLaunches, TelemetryAttributes.GraphKind, "batched")),
                "{launch}", "Decode steps launched as CUDA graphs, by kind.");
            M.CreateObservableCounter("tensorsharp.cuda_graph.capture_failures",
                () => ServingMeasurements((s => s.GraphCaptureFailures, null, null)), "{failure}",
                "CUDA graph captures that failed (those steps then run uncaptured).");
            M.CreateObservableCounter("tensorsharp.engine.forward.issue_time",
                () => ServingMeasurements((s => s.ForwardIssueSeconds, null, null)), "s",
                "Host time spent issuing forwards' GPU work. Close to tensorsharp.engine.forward.time: bound by the host.");
            M.CreateObservableCounter("tensorsharp.engine.forward.wait_time",
                () => ServingMeasurements((s => s.ForwardWaitSeconds, null, null)), "s",
                "Host time spent waiting for forwards' GPU work after issuing it.");
            M.CreateObservableCounter("tensorsharp.engine.lane.steps",
                () => ServingMeasurements(
                    (s => s.DecodeLaneFinishesReady, TelemetryAttributes.LaneFinish, "ready"),
                    (s => s.DecodeLaneFinishesWaited, TelemetryAttributes.LaneFinish, "waited")),
                "{step}",
                "Decode steps kept in flight on a decode lane, by how they ended for the host. ready: the GPUs had finished before the host came back for the logits, so the host (sampling, scheduling) bounds the rate; waited: the GPUs do.");
            M.CreateObservableCounter("tensorsharp.engine.lane.rows",
                () => ServingMeasurements((s => s.DecodeLaneRows, null, null)), "{row}",
                "Rows (sequences) the decode lanes' steps carried.");
            M.CreateObservableCounter("tensorsharp.engine.lane.time",
                () => ServingMeasurements(
                    (s => s.DecodeLaneIssueSeconds, TelemetryAttributes.LaneState, "issue"),
                    (s => s.DecodeLaneWaitSeconds, TelemetryAttributes.LaneState, "wait"),
                    (s => s.DecodeLaneIdleSeconds, TelemetryAttributes.LaneState, "idle")),
                "s",
                "Host time per decode lane state, summed over lanes: issue (queueing a lane's step), wait (blocked on a lane's step), idle (a lane with no step in flight between its finish and its next issue).");
        }

        private static readonly ObservableCounter<long> s_moeRoutes;

        /// <summary>Whether anything collects the MoE offload instruments; the models' counter snapshots are skipped
        /// otherwise, as the GPU memory ones are.</summary>
        public static bool MoeOffloadObserved => s_moeRoutes.Enabled;

        /// <summary>
        /// Report MoE expert offload counters for one model architecture. <paramref name="read"/> runs on the metrics
        /// collection thread: return a snapshot taken on the engine's thread, never query a backend there. Returns a
        /// handle that unregisters it.
        /// </summary>
        public static IDisposable RegisterMoeOffloadSource(string architecture, Func<MoeOffloadReading?> read) =>
            Register(s_moeOffloadSources, architecture, read);

        private static IEnumerable<Measurement<long>> MoeRouteMeasurements()
        {
            foreach (var source in s_moeOffloadSources)
            {
                MoeOffloadReading? r;
                try { r = source.Value(); }
                catch (Exception) { continue; }
                if (r is not MoeOffloadReading reading)
                    continue;
                var arch = new KeyValuePair<string, object?>(TelemetryAttributes.ModelArchitecture, source.Key);
                yield return new Measurement<long>(reading.DeviceRoutes, arch,
                    new KeyValuePair<string, object?>(TelemetryAttributes.MoePlacement, TelemetryAttributes.MoePlacementDevice));
                yield return new Measurement<long>(reading.StreamedRoutes, arch,
                    new KeyValuePair<string, object?>(TelemetryAttributes.MoePlacement, TelemetryAttributes.MoePlacementStreamed));
                yield return new Measurement<long>(reading.HostRoutes, arch,
                    new KeyValuePair<string, object?>(TelemetryAttributes.MoePlacement, TelemetryAttributes.MoePlacementHost));
            }
        }

        private static IEnumerable<Measurement<T>> MoeMeasurements<T>(Func<MoeOffloadReading, T> pick) where T : struct
        {
            foreach (var source in s_moeOffloadSources)
            {
                MoeOffloadReading? r;
                try { r = source.Value(); }
                catch (Exception) { continue; }
                if (r is MoeOffloadReading reading)
                    yield return new Measurement<T>(pick(reading),
                        new KeyValuePair<string, object?>(TelemetryAttributes.ModelArchitecture, source.Key));
            }
        }

        private static IEnumerable<Measurement<T>> ServingMeasurements<T>(
            params (Func<ServingTelemetrySnapshot, T> Pick, string? Key, string? Value)[] series) where T : struct
        {
            foreach (var entry in s_engines)
            {
                ServingTelemetrySnapshot snapshot;
                try
                {
                    if (entry.Key.ServingTelemetrySource is not { } source)
                        continue;
                    snapshot = source.ReadServingTelemetry();
                }
                catch (Exception) { continue; }
                var architecture = new KeyValuePair<string, object?>(TelemetryAttributes.ModelArchitecture, entry.Value);
                foreach (var (pick, key, value) in series)
                    yield return key == null
                        ? new Measurement<T>(pick(snapshot), architecture)
                        : new Measurement<T>(pick(snapshot), architecture, new KeyValuePair<string, object?>(key, value));
            }
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

    /// <summary>
    /// Process totals of MoE expert offload, monotonic (they survive a model's unload): routes by where the expert ran
    /// (an accelerator slot already holding it, a slot that read it from storage first, the host's expert kernel), the
    /// host kernel's time, adaptive residency swaps, and bytes explicit expert I/O read.
    /// </summary>
    public readonly record struct MoeOffloadReading(long DeviceRoutes, long StreamedRoutes, long HostRoutes,
        double HostComputeSeconds, long Swaps, long IoBytes);

    /// <summary>One device's memory figures; a figure a backend cannot report is <c>null</c>.</summary>
    public readonly record struct GpuMemoryReading(int Device, long? UsedBytes, long? TotalBytes, long? PeakBytes);
}
