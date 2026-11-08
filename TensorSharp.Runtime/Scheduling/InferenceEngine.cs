// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Top-level inference engine that ties together the paged KV pool, the
    /// continuous-batching scheduler, and the batch executor. Owns a single
    /// dedicated worker thread that runs the step loop; clients enqueue
    /// requests via <see cref="SubmitRequest"/> and consume per-token output
    /// via the returned <see cref="InferenceRequestHandle"/>.
    ///
    /// The engine is the single coordination point for everything that needs
    /// the model's KV state, and request lifecycle is per-request rather than
    /// per-session.
    /// </summary>
    public sealed class InferenceEngine : IDisposable
    {
        private readonly IModelArchitecture _model;
        private readonly ILogger _logger;
        private readonly bool _stopRepetition;
        // One emitted token's bytes for ReasoningLoopDetector (engine thread only).
        private readonly List<byte> _loopBytes = new();
        private readonly int _nativeSlotContextLimit;
        // The model's context window (MAX_CONTEXT), or 0 when it advertises none. A request
        // must leave its prompt one token of room and stops generating where it ends.
        private readonly int _contextLimit;
        private readonly BlockPool _pool;
        private readonly ContinuousBatchScheduler _scheduler;
        private readonly BatchExecutor _executor;

        private readonly ConcurrentDictionary<string, InferenceRequestHandle> _handles = new();
        // SubmitRequest is callable from multiple client threads. Serialize the
        // short admission critical section so an in-flight RequestId is reserved
        // before another submit can construct/queue a replacement handle.
        private readonly object _submissionGate = new();
        private readonly Channel<EngineCommand> _commands = Channel.CreateUnbounded<EngineCommand>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
        private readonly Thread _worker;
        private readonly CancellationTokenSource _shutdownCts = new();
        private long _totalCompleted;
        private long _totalSubmitted;
        private long _totalStepsRun;
        private long _totalForwardTicks;
        private bool _disposed;

        // Tags of the per-step histograms, built once so recording a step allocates nothing.
        // The step kind says whether a step's duration is a prompt chunk, a batched decode, or
        // both; the values are TensorSharp's own (TelemetryAttributes.StepKind).
        private const string StepKindPrefill = "prefill";
        private const string StepKindDecode = "decode";
        private const string StepKindMixed = "mixed";
        private static readonly KeyValuePair<string, object?> s_prefillStepTag = new(Telemetry.TelemetryAttributes.StepKind, StepKindPrefill);
        private static readonly KeyValuePair<string, object?> s_decodeStepTag = new(Telemetry.TelemetryAttributes.StepKind, StepKindDecode);
        private static readonly KeyValuePair<string, object?> s_mixedStepTag = new(Telemetry.TelemetryAttributes.StepKind, StepKindMixed);
        private readonly KeyValuePair<string, object?> _architectureTag;

        // TENSORSHARP_OTEL_TRACE_STEPS: the step span, its link cap, and the one attribute only
        // it carries (a count, so it stays off the metrics).
        private const string StepSpanName = "inference.step";
        private const int MaxStepSpanLinks = 32;
        private const string ScheduledTokensAttribute = "tensorsharp.engine.scheduled_tokens";

        /// <summary>Whether the radix prefix cache serves this loaded model (prefix
        /// caching is on and the model implements the prefix-cache contract).</summary>
        public bool PrefixCacheActive => _executor.RadixPrefixCacheEnabled;

        public InferenceEngine(IModelArchitecture model, SchedulerConfig cfg, ILogger? logger = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            ArgumentNullException.ThrowIfNull(cfg);
            // TS_DUMP_LOGITS: an earlier run's vector must not survive a run that never samples.
            FirstTokenLogitsDump.ClearStaleOutputs();
            if (cfg.BlockSize <= 0) cfg = cfg.WithBlockSize(PreferredBlockSize(model));
            _logger = logger ?? NullLogger.Instance;
            if (!cfg.MixedStepPrefillBudgetChosen
                && model is Speculative.IMixedStepPrefillTarget { PreferredMixedStepPrefillTokens: > 0 } mixedTarget)
            {
                cfg = cfg.WithMixedStepPrefillBudget(mixedTarget.PreferredMixedStepPrefillTokens);
                _logger.LogInformation(
                    "Steps that also decode carry at most {Tokens} prompt tokens (the model's bound; " +
                    "TS_SCHED_MIXED_PREFILL_TOKENS overrides it, 0 for none)", cfg.MixedStepPrefillBudget);
            }
            _stopRepetition = cfg.StopRepetition;   // cfg is null-checked above
            _nativeSlotContextLimit = UsesNativeDeepSeek41Slots(model) ? Math.Max(0, model.MaxContextLength) : 0;
            _contextLimit = Math.Max(0, model.MaxContextLength);
            // The same name the engine gauges use (TensorSharpMetrics.RegisterEngine).
            string? architecture = model.Config?.Architecture;
            _architectureTag = new(Telemetry.TelemetryAttributes.ModelArchitecture,
                string.IsNullOrEmpty(architecture) ? "unknown" : architecture);

            long blockBytes = ComputeBlockByteSize(model, cfg.BlockSize);
            int numBlocks = ResolveEffectiveNumBlocks(model, cfg, _logger);
            _pool = new BlockPool(numBlocks, cfg.BlockSize, blockBytes);
            _scheduler = new ContinuousBatchScheduler(cfg, _pool, logger,
                supportsCrossSequenceKvReuse: model.SupportsCrossSequenceKvReuse,
                requiresPerBlockCapture: model.RequiresPerBlockCapture);
            AttachSoloPrefillPolicy(model, cfg, _scheduler, _logger);
            _executor = new BatchExecutor(model, _pool, _scheduler, logger);
            _executor.InitializeRadixCache(cfg);
            // So admission can say WHY a turn reused nothing.
            _scheduler.AttachReuseDiagnostics(() => _executor.LastFusedContinuationDeclineReason);
            // Shared-prefix checkpoints: end a prefill chunk exactly where the chat
            // layer says the shared prompt ends, so the executor can copy the model's
            // state there and start every later new chat from that copy.
            if (_executor.PrefixCheckpointsSupported)
            {
                _scheduler.EnablePrefixCheckpoints();
                if (model is ISoloPrefillBudgetModel { CheckpointGapTokens: > 0 } gapModel)
                {
                    _scheduler.SetCheckpointGapTokens(gapModel.CheckpointGapTokens);
                    _logger.LogInformation(
                        "Shared-prefix checkpoint boundaries at most {Tokens} tokens past a reused prefix do not end a " +
                        "prefill step (the model's forward pass costs more than those tokens save a later request)",
                        gapModel.CheckpointGapTokens);
                }
            }

            // One-time capability report: which execution paths are statically
            // available for this model+backend under the current configuration,
            // and why the unavailable ones are unavailable. Per-step routing
            // (selected path, fallback chain, rejection reasons) is logged by
            // BatchExecutor whenever the plan changes.
            _logger.LogInformation(
                "InferenceEngine[{Arch}] execution capability report:\n{Report}",
                model.Config?.Architecture ?? "model",
                ExecutionPlanner.BuildCapabilityReport(
                    ExecutionCapabilities.FromModel(model),
                    ExecutionOptions.FromEnvironment(),
                    cfg));

            _worker = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = $"TensorSharp.InferenceEngine[{model.Config?.Architecture ?? "model"}]",
            };
            _worker.Start();
            // The engine gauges (running, waiting, steps, KV blocks) read the counters above at
            // collection time; registering costs nothing when telemetry is off.
            Telemetry.TensorSharpMetrics.RegisterEngine(this, model.Config?.Architecture);
        }

        public IModelArchitecture Model => _model;
        public BlockPoolStats PoolStats => _pool.GetStats();
        /// <summary>The block pool itself, for tests that inspect its slabs.</summary>
        internal BlockPool Pool => _pool;
        /// <summary>The radix prefix cache, or null when it does not serve this model.</summary>
        internal PrefixCache.PrefixCacheCoordinator? RadixCache => _executor.RadixCache;
        public long TotalCompleted => Interlocked.Read(ref _totalCompleted);
        public long TotalSubmitted => Interlocked.Read(ref _totalSubmitted);
        public long TotalStepsRun => Interlocked.Read(ref _totalStepsRun);
        public TimeSpan TotalForwardTime => TimeSpan.FromMilliseconds(
            (double)Interlocked.Read(ref _totalForwardTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);
        public int RunningCount => _scheduler.RunningCount;
        public int WaitingCount => _scheduler.WaitingCount;

        /// <summary>
        /// Whether the step loop may run right now, or null for "always".
        ///
        /// <para>
        /// Consulted on the worker thread between two steps, which is the one place
        /// where no command buffer is in flight. A closed gate parks the loop there:
        /// nothing is scheduled, nothing is submitted to the GPU, and the sequences keep
        /// their place and their cache until it opens. It exists for iOS, where an app
        /// that is not frontmost may not submit GPU work and ggml-metal treats a refused
        /// command buffer as terminal — but nothing here knows that; a host that never
        /// closes it pays one volatile read per step. See <see cref="ComputeGate"/>.
        /// </para>
        /// <para>
        /// The engine cannot enforce this by being pulled less: tokens go out through an
        /// unbounded channel and the loop runs whether or not anyone reads them, which is
        /// what makes a reader that walks away harmless — and is also exactly why a
        /// reader that merely stops reading cannot stop the GPU.
        /// </para>
        /// </summary>
        public ComputeGate? ComputeGate
        {
            get => Volatile.Read(ref _computeGate);
            set => Volatile.Write(ref _computeGate, value);
        }

        private ComputeGate? _computeGate;
        private long _stepsHeldByGate;

        /// <summary>
        /// Where shared-prefix checkpoints outlive the process, or null for nowhere;
        /// forwarded to the executor, which reads it on its own thread. See
        /// <see cref="IPrefixCheckpointStore"/>.
        /// </summary>
        public IPrefixCheckpointStore? PrefixCheckpointStore
        {
            get => _executor.PrefixCheckpointStore;
            set => _executor.PrefixCheckpointStore = value;
        }

        /// <summary>Wait for the checkpoint writes to <see cref="PrefixCheckpointStore"/> still in
        /// flight: on hosts with memory to spare they run in the background, after the request that
        /// produced them has completed.</summary>
        public void FlushPrefixCheckpointSaves() => _executor.RadixCache?.FlushPendingSaves();

        /// <summary>
        /// How many times the step loop was actually held by a closed
        /// <see cref="ComputeGate"/>. Zero on any host that never closes it; a check
        /// reads it to prove the loop parked rather than merely that the gate closed.
        /// </summary>
        public long StepsHeldByGate => Interlocked.Read(ref _stepsHeldByGate);

        /// <summary>Submit a sequence for inference. Returns immediately with a
        /// handle whose <see cref="InferenceRequestHandle.Tokens"/> channel
        /// streams sampled tokens.</summary>
        public InferenceRequestHandle SubmitRequest(SequenceState seq, CancellationToken ct = default)
        {
            if (seq == null) throw new ArgumentNullException(nameof(seq));
            lock (_submissionGate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_handles.ContainsKey(seq.RequestId))
                {
                    throw new InvalidOperationException(
                        $"Sequence {seq.RequestId} is already submitted.");
                }

                var handle = new InferenceRequestHandle(seq, this, ct);
                if (!_handles.TryAdd(seq.RequestId, handle))
                {
                    // All submitters take _submissionGate, so this is defensive
                    // against a future registry writer outside this method.
                    var ex = new InvalidOperationException(
                        $"Sequence {seq.RequestId} is already submitted.");
                    handle.CompleteWithError(ex);
                    throw ex;
                }

                if (!_commands.Writer.TryWrite(new EngineCommand
                    {
                        Kind = EngineCommandKind.Submit,
                        Sequence = seq,
                    }))
                {
                    _handles.TryRemove(seq.RequestId, out _);
                    var ex = new ObjectDisposedException(nameof(InferenceEngine));
                    handle.CompleteWithError(ex);
                    throw ex;
                }

                Interlocked.Increment(ref _totalSubmitted);
                return handle;
            }
        }

        /// <summary>Cancel a submitted request. Idempotent.</summary>
        public void Abort(string requestId)
        {
            _commands.Writer.TryWrite(new EngineCommand
            {
                Kind = EngineCommandKind.Abort,
                RequestId = requestId,
            });
        }

        /// <summary>
        /// Release memory that only serves the next request's speed (retained
        /// conversation holders beyond the newest, parked per-request holders, pooled
        /// host buffers). Queued like an abort and applied on the engine thread between
        /// steps, because those buffers belong to the model and a forward may be reading
        /// them right now. Safe to call at any time; a no-op after disposal.
        /// </summary>
        public void TrimIdleMemory()
        {
            _commands.Writer.TryWrite(new EngineCommand { Kind = EngineCommandKind.Trim });
        }

        /// <summary>
        /// Switch the speculation policy for every step from now on - what a settings
        /// switch does while a model is loaded, instead of waiting for the next load.
        /// Queued like a trim and applied on the engine thread between steps; the
        /// executor drops its armed contexts and re-arms under the new policy on the
        /// next turn. Safe to call at any time; a no-op after disposal.
        /// </summary>
        public void UpdateSpeculation(SpeculationOptions options)
        {
            _commands.Writer.TryWrite(new EngineCommand
            {
                Kind = EngineCommandKind.Speculation,
                Speculation = options ?? SpeculationOptions.Disabled,
            });
        }

        /// <summary>
        /// Release the model's copy of every retired request adapter (see
        /// <see cref="IRequestAdapter.IsRetired"/>) that no request in flight still runs with;
        /// the rest go as their last request finishes. Queued like a trim and applied on the
        /// engine thread between steps, because a step may be reading the adapter's factors
        /// right now. Safe to call at any time. Returns false once the engine has shut down,
        /// when nothing it ran is left to finish.
        /// </summary>
        public bool ReleaseRetiredAdapters()
            => _commands.Writer.TryWrite(new EngineCommand { Kind = EngineCommandKind.ReleaseAdapters });

        /// <summary>How many submitted, unfinished requests run with an adapter that
        /// <paramref name="match"/> accepts. Safe from any thread.</summary>
        public int CountInFlightRequests(Func<IRequestAdapter, bool> match)
        {
            ArgumentNullException.ThrowIfNull(match);
            int count = 0;
            foreach (var entry in _handles)
            {
                if (entry.Value.Sequence.Adapter is { } adapter && match(adapter))
                    count++;
            }
            return count;
        }

        public void Dispose()
        {
            lock (_submissionGate)
            {
                if (_disposed) return;
                _disposed = true;
                _shutdownCts.Cancel();
                _commands.Writer.TryComplete();
            }
            // Wait for the worker to actually leave its step. The caller is about to
            // free the model's buffers and, on a recovery, the GPU backend itself; a
            // worker still inside a graph compute when that happens is a use-after-free
            // in a kernel, not an error. A step is bounded (one decode, or one prefill
            // chunk), so this returns; the cap only stops a wedged native call from
            // holding a shutdown forever. A worker parked on the compute gate leaves at
            // once, because the gate wait uses the shutdown token.
            if (_worker.IsAlive && Thread.CurrentThread != _worker)
            {
                bool left;
                try { left = _worker.Join(TimeSpan.FromSeconds(60)); } catch { left = true; }
                if (!left)
                    _logger.LogWarning("InferenceEngine worker did not leave its step within 60s of shutdown; releasing anyway");
            }
            // Only now: a step that was running when Dispose began has counted itself in
            // TotalStepsRun and TotalForwardTime (and the step histograms), and unregistering
            // folds those totals into the retired ones, so the step counters match the histograms.
            // Before the resources below are freed, so a metrics collection never reads them.
            Telemetry.TensorSharpMetrics.UnregisterEngine(this);
            // Nobody is going to finish these now. A consumer awaiting one of them --
            // another conversation's turn, on a phone -- would otherwise wait forever.
            var abandoned = new ObjectDisposedException(nameof(InferenceEngine),
                "The inference engine was shut down while this request was in flight.");
            foreach (var entry in _handles)
            {
                if (_handles.TryRemove(entry.Key, out var handle))
                    handle.CompleteWithError(abandoned);
            }
            if (!_worker.IsAlive)
            {
                lock (_model.GpuComputeLock)
                {
                    foreach (var sequence in _scheduler.GetInFlightSequencesSnapshot())
                    {
                        _scheduler.Abort(sequence.RequestId);
                        NotifyReleasedSequence(_model as IBatchedPagedModel, sequence.RequestId,
                            seen: null, retainFusedCache: false);
                    }
                    _executor.Reset();
                    _executor.RadixCache?.Detach();
                }
            }
        }

        private void WorkerLoop()
        {
            // Thread.Start flowed the ExecutionContext of whichever request lazily built the
            // engine, and the engine serves every request, so nothing it traces should inherit
            // that request's span. (UnsafeStart would also drop the inherited logging scope and
            // change what the engine's log lines print.)
            System.Diagnostics.Activity.Current = null;
            Telemetry.TelemetryContext.Clear();
            var sw = new System.Diagnostics.Stopwatch();
            while (!_shutdownCts.IsCancellationRequested)
            {
                // Drain queued commands (non-blocking).
                while (_commands.Reader.TryRead(out var cmd))
                {
                    lock (_model.GpuComputeLock)
                        ApplyCommand(cmd);
                }

                // A retired adapter's factors go once its last request has left (finished,
                // failed or aborted); base-only traffic never sets the flag.
                if (_adapterSweepPending)
                {
                    lock (_model.GpuComputeLock)
                        SweepRetiredAdapters();
                }

                // If there's nothing in flight, block on command channel.
                if (_scheduler.RunningCount == 0 && _scheduler.WaitingCount == 0)
                {
                    try
                    {
                        // Wait for at least one command to arrive.
                        if (!_commands.Reader.WaitToReadAsync(_shutdownCts.Token).AsTask().GetAwaiter().GetResult())
                            break;
                    }
                    catch (OperationCanceledException) { break; }
                    continue;
                }

                // Not while the host says the GPU is not ours. Between steps, so no
                // command buffer is in flight when the loop parks; and BEFORE Schedule,
                // so nothing is admitted or preempted on the strength of a step that is
                // not about to run. Commands that queue while the loop is held (an Abort,
                // a Submit) are drained at the top of the next pass, before the step they
                // would have changed.
                if (Volatile.Read(ref _computeGate) is ComputeGate gate && !gate.IsOpen)
                {
                    Interlocked.Increment(ref _stepsHeldByGate);
                    try { gate.Wait(_shutdownCts.Token); }
                    catch (OperationCanceledException) { break; }
                    continue;
                }

                // Admission materializes model payloads, and completion captures them.
                // They share the same compute gate as forwarding and media encoders.
                lock (_model.GpuComputeLock)
                {
                    // Run one scheduler step.
                    sw.Restart();
                    SchedulerOutput? output = null;
                    List<SequenceStepResult> results;
                    try
                    {
                        output = _scheduler.Schedule();
                    }
                    catch (Exception ex)
                    {
                        FailStepSequences(ex, output, "scheduler");
                        continue;
                    }

                    // A request that resolved an adapter before it was unloaded binds it
                    // again for its own steps; sweep it once that request is done.
                    if (output.Adapter is { IsRetired: true })
                        _adapterSweepPending = true;

                    if (output.IsEmpty)
                    {
                        // A preemption may have happened while trying to make room.
                        // Release its model-owned state even though no forward pass
                        // was produced this iteration.
                        NotifyReleasedSequences(output);

                        // A non-empty running set cannot become schedulable without
                        // completing a step or freeing blocks. If Schedule returned
                        // no work, neither can happen: continuing would busy-spin the
                        // worker forever with an open client stream and an idle GPU.
                        // Schedule may need more than one pass to preempt enough
                        // small victims for a large allocation. A preemption is
                        // real progress even if this pass produced no forward work.
                        if (_scheduler.RunningCount > 0
                            && output.PreemptedRequestIds.Count == 0)
                            FailStalledSequences();
                        continue;
                    }

                    try
                    {
                        results = _executor.ExecuteStep(output);
                    }
                    catch (Exception ex)
                    {
                        FailStepSequences(ex, output, "executor");
                        continue;
                    }

                    long stepTicks = sw.ElapsedTicks;
                    Interlocked.Increment(ref _totalStepsRun);
                    Interlocked.Add(ref _totalForwardTicks, stepTicks);
                    RecordStepMetrics(output, stepTicks);

                    // Post-step: emit tokens, detect stop conditions, finish sequences.
                    ApplyResults(results, output);

                    // Notify the model about sequences whose per-request state can
                    // now be reclaimed (finished, preempted, errored). Hybrid
                    // models (Nemotron-H, Qwen 3.5) allocate Mamba2 / GatedDeltaNet
                    // recurrent-state slots keyed by RequestId; without this
                    // notification the slot pool grows unbounded and slot indices
                    // get reused incorrectly across abandoned sequences.
                    NotifyReleasedSequences(output);

                    // The optional step span is built here, from the step's measured times
                    // and after its tokens went out, so the step itself pays nothing for it.
                    // A listener check first: TraceSteps may read the environment.
                    if (Telemetry.TensorSharpTelemetry.Inference.HasListeners() && Telemetry.TensorSharpTelemetry.TraceSteps)
                        TraceStep(output, stepTicks, sw.ElapsedTicks);

                    // The step observer (device-memory snapshots) still runs under the GPU lock,
                    // which a snapshot needs, but after the step's tokens went out, so it never
                    // delays one. A null check unless an observer is registered.
                    Telemetry.TensorSharpMetrics.OnEngineStep();
                }
            }
        }

        /// <summary>Step duration (scheduling plus execution, the time <see cref="TotalForwardTime"/>
        /// sums) and batch size of a step that ran. Engine thread, once per step; with no meter
        /// listener attached it is two flag reads.</summary>
        private void RecordStepMetrics(SchedulerOutput output, long stepTicks)
        {
            var duration = Telemetry.TensorSharpMetrics.EngineStepDuration;
            var batchSize = Telemetry.TensorSharpMetrics.EngineBatchSize;
            if (!duration.Enabled && !batchSize.Enabled)
                return;
            var kind = StepKindTag(output.ScheduledWork);
            duration.Record((double)stepTicks / System.Diagnostics.Stopwatch.Frequency, _architectureTag, kind);
            batchSize.Record(output.ScheduledWork.Count, _architectureTag, kind);
        }

        /// <summary>Whether a step forwarded prompt chunks, decode tokens, or both.</summary>
        private static KeyValuePair<string, object?> StepKindTag(List<ScheduledSequenceWork> work)
        {
            bool prefill = false, decode = false;
            for (int i = 0; i < work.Count; i++)
            {
                if (work[i].IsPrefill) prefill = true;
                else decode = true;
            }
            return prefill ? (decode ? s_mixedStepTag : s_prefillStepTag) : s_decodeStepTag;
        }

        /// <summary>
        /// One <c>inference.step</c> span for a step that ran (TENSORSHARP_OTEL_TRACE_STEPS). A
        /// batched step belongs to every request in it and a span has one parent, so the span is
        /// a root in a trace of its own, linked to the request spans of the sequences it ran. It
        /// is built after the fact: <paramref name="sinceStartTicks"/> is the step stopwatch now,
        /// <paramref name="stepTicks"/> where the step itself ended.
        /// </summary>
        private void TraceStep(SchedulerOutput output, long stepTicks, long sinceStartTicks)
        {
            try
            {
                var work = output.ScheduledWork;
                List<System.Diagnostics.ActivityLink>? links = null;
                for (int i = 0; i < work.Count && (links == null || links.Count < MaxStepSpanLinks); i++)
                {
                    var context = work[i].Sequence.TelemetryContext;
                    if (context == default)
                        continue;
                    links ??= new List<System.Diagnostics.ActivityLink>(Math.Min(work.Count, MaxStepSpanLinks));
                    links.Add(new System.Diagnostics.ActivityLink(context));
                }

                DateTimeOffset start = DateTimeOffset.UtcNow - StopwatchTicksToTimeSpan(sinceStartTicks);
                // parentContext default with Activity.Current null (WorkerLoop cleared it): a root.
                var activity = Telemetry.TensorSharpTelemetry.Inference.StartActivity(
                    StepSpanName, System.Diagnostics.ActivityKind.Internal, parentContext: default,
                    links: links, startTime: start);
                if (activity == null)
                    return;
                if (activity.IsAllDataRequested)
                {
                    activity.SetTag(Telemetry.TelemetryAttributes.ModelArchitecture, _architectureTag.Value);
                    activity.SetTag(Telemetry.TelemetryAttributes.BatchSize, work.Count);
                    activity.SetTag(Telemetry.TelemetryAttributes.StepKind, StepKindTag(work).Value);
                    activity.SetTag(ScheduledTokensAttribute, output.TotalScheduledTokens);
                }
                activity.SetEndTime((start + StopwatchTicksToTimeSpan(stepTicks)).UtcDateTime);
                activity.Stop();
            }
            catch (Exception ex)
            {
                // Profiling output must never fail the step loop that serves every request.
                _logger.LogDebug(ex, "Recording the engine step span failed");
            }
        }

        private static TimeSpan StopwatchTicksToTimeSpan(long ticks)
            => TimeSpan.FromTicks((long)(ticks * ((double)TimeSpan.TicksPerSecond / System.Diagnostics.Stopwatch.Frequency)));

        private void NotifyReleasedSequences(SchedulerOutput output)
        {
            var batched = _model as Runtime.Scheduling.IBatchedPagedModel;
            var seen = new HashSet<string>(StringComparer.Ordinal);
            if (output.FinishedRequestIds != null)
            {
                foreach (var id in output.FinishedRequestIds)
                    NotifyReleasedSequence(batched, id, seen);
            }
            if (output.PreemptedRequestIds != null)
            {
                foreach (var id in output.PreemptedRequestIds)
                    NotifyReleasedSequence(batched, id, seen);
            }
        }

        /// <summary>Fail running requests after the scheduler reports no work.
        /// This is an invariant guard for an exhausted fixed-size KV pool: it
        /// converts an otherwise permanent empty-plan spin into a clear error and
        /// releases the blocks so waiting requests can continue.</summary>
        private void FailStalledSequences()
        {
            var stalled = _scheduler.GetRunningSequencesSnapshot();
            if (stalled.Count == 0) return;

            var ex = new InvalidOperationException(
                "KV cache capacity exceeded: no running sequence can make progress " +
                "within the configured KV block pool. Shorten the prompt or generated " +
                "output, or enlarge the pool with TS_SCHED_NUM_BLOCKS.");

            _logger.LogError(
                "Scheduler stalled: failing {Count} running request(s) because the KV " +
                "pool is exhausted ({Free}/{Total} blocks free).",
                stalled.Count, _pool.NumFreeBlocks, _pool.NumBlocks);

            var released = new HashSet<string>(StringComparer.Ordinal);
            foreach (var seq in stalled)
            {
                if (seq == null) continue;

                try
                {
                    if (_scheduler.NotifyError(seq, ex))
                        released.Add(seq.RequestId);
                }
                catch (Exception cleanupEx)
                {
                    _logger.LogError(
                        cleanupEx,
                        "Failed to release scheduler state for stalled sequence {EngineRequestId}",
                        seq.RequestId);
                    released.Add(seq.RequestId);
                }

                if (_handles.TryRemove(seq.RequestId, out var handle))
                {
                    LogSpeculationStatsIfAny(seq);
                    handle.CompleteWithError(ex);
                    Interlocked.Increment(ref _totalCompleted);
                }
            }

            NotifyReleasedSequences(released);
        }

        private void FailStepSequences(Exception ex, SchedulerOutput? output, string phase)
        {
            var affected = GetAffectedSequences(output);
            if (affected.Count == 0)
            {
                _logger.LogError(ex, "Engine {Phase} step failed with no affected requests", phase);
                if (output != null)
                    NotifyReleasedSequences(output);
                return;
            }

            _logger.LogError(
                ex,
                "Engine {Phase} step failed; failing {Count} affected request(s)",
                phase,
                affected.Count);

            var released = new HashSet<string>(StringComparer.Ordinal);
            if (output?.PreemptedRequestIds != null)
            {
                foreach (var id in output.PreemptedRequestIds)
                {
                    if (!string.IsNullOrEmpty(id))
                        released.Add(id);
                }
            }

            foreach (var seq in affected)
            {
                if (seq == null) continue;

                try
                {
                    if (_scheduler.NotifyError(seq, ex, output))
                        released.Add(seq.RequestId);
                }
                catch (Exception cleanupEx)
                {
                    _logger.LogError(
                        cleanupEx,
                        "Failed to release scheduler state for errored sequence {EngineRequestId}",
                        seq.RequestId);
                    released.Add(seq.RequestId);
                }

                if (_handles.TryRemove(seq.RequestId, out var handle))
                {
                    handle.CompleteWithError(ex);
                    Interlocked.Increment(ref _totalCompleted);
                    released.Add(seq.RequestId);
                }
            }

            if (output?.FinishedRequestIds != null)
            {
                foreach (var id in output.FinishedRequestIds)
                {
                    if (!string.IsNullOrEmpty(id))
                        released.Add(id);
                }
            }

            NotifyReleasedSequences(released);
        }

        private List<SequenceState> GetAffectedSequences(SchedulerOutput? output)
        {
            var affected = new List<SequenceState>();
            var seen = new HashSet<string>(StringComparer.Ordinal);

            if (output?.ScheduledWork != null)
            {
                foreach (var work in output.ScheduledWork)
                {
                    var seq = work?.Sequence;
                    if (seq == null) continue;
                    if (seen.Add(seq.RequestId))
                        affected.Add(seq);
                }
            }

            if (affected.Count > 0)
                return affected;

            return _scheduler.GetInFlightSequencesSnapshot();
        }

        private void NotifyReleasedSequences(IEnumerable<string> requestIds)
        {
            var batched = _model as Runtime.Scheduling.IBatchedPagedModel;
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var id in requestIds)
                NotifyReleasedSequence(batched, id, seen);
        }

        private void NotifyReleasedSequence(
            Runtime.Scheduling.IBatchedPagedModel? batched,
            string requestId,
            HashSet<string>? seen,
            bool retainFusedCache = true)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            if (seen != null && !seen.Add(requestId)) return;

            try
            {
                if (retainFusedCache)
                {
                    // Give the executor first refusal: a fused sequence that finished
                    // cleanly has its complete per-request state holder RETAINED
                    // (re-keyed out of the active set) for cross-request prefix reuse,
                    // so the model release below no-ops instead of disposing it.
                    _executor.TryRetainReleasedFusedCache(requestId);
                }
                else
                {
                    // Aborted requests are never retention candidates, but they may
                    // already have executor tracking (or a retained-holder rewind that
                    // was planned while waiting for suffix capacity). Clear both before
                    // the model frees the holder.
                    _executor.DiscardReleasedFusedCacheBookkeeping(requestId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(
                    ex,
                    retainFusedCache
                        ? "Fused-cache retention failed for sequence {EngineRequestId}"
                        : "Fused-cache bookkeeping cleanup failed for sequence {EngineRequestId}",
                    requestId);
            }

            try
            {
                batched?.OnSequenceReleased(requestId);
                _executor.RadixCache?.Drain();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Model release hook failed for sequence {EngineRequestId}", requestId);
            }
        }

        private void ApplyCommand(EngineCommand cmd)
        {
            switch (cmd.Kind)
            {
                case EngineCommandKind.Submit:
                    try
                    {
                        // Every model: the prompt must leave room for one generated token.
                        // The pool check in Submit below cannot catch this - the pool is
                        // sized for the whole context or more - so a 6k-token prompt under
                        // MAX_CONTEXT=4096 used to be admitted and fail inside the model's
                        // prefill ("Requested sequence length ... exceeds configured max
                        // context"), after spending a forward on it.
                        int promptTokens = cmd.Sequence.PromptTokens.Count;
                        if (_contextLimit > 0 && promptTokens >= _contextLimit)
                            throw RefuseForContext(cmd.Sequence,
                                new ContextLengthExceededException(promptTokens, _contextLimit));
                        // A larger metadata pool accounts for independent native
                        // slots; it must not enlarge any individual slot's context.
                        long requested = (long)promptTokens + cmd.Sequence.MaxNewTokens;
                        if (_nativeSlotContextLimit > 0 && requested > _nativeSlotContextLimit)
                            throw RefuseForContext(cmd.Sequence, new ContextLengthExceededException(
                                promptTokens, _nativeSlotContextLimit,
                                $"DeepSeek V4.1 request requires {requested} tokens, but each native sequence slot " +
                                $"has a context limit of {_nativeSlotContextLimit}. Shorten the prompt or output allowance."));
                        _scheduler.Submit(cmd.Sequence);
                    }
                    catch (Exception ex)
                    {
                        if (_handles.TryRemove(cmd.Sequence.RequestId, out var h))
                            h.CompleteWithError(ex);
                    }
                    break;

                case EngineCommandKind.Trim:
                    try
                    {
                        _logger.LogInformation("Idle memory trimmed on the host's request: {Summary}",
                            _executor.TrimIdleMemory());
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Trimming idle memory failed");
                    }
                    break;

                case EngineCommandKind.ReleaseAdapters:
                    SweepRetiredAdapters();
                    break;

                case EngineCommandKind.Speculation:
                    _executor.SetSpeculation(cmd.Speculation);
                    _logger.LogInformation(
                        "Speculation policy updated on the host's request: enabled={Enabled} algorithm={Algorithm} maxDraft={MaxDraft}",
                        cmd.Speculation.Enabled, cmd.Speculation.SpeculatorName, cmd.Speculation.MaxDraftTokens);
                    break;

                case EngineCommandKind.Abort:
                    _scheduler.Abort(cmd.RequestId);
                    // Every radix family owns request keys, including primary-only
                    // models which have no IBatchedPagedModel release hook.
                    NotifyReleasedSequence(
                        _model as IBatchedPagedModel,
                        cmd.RequestId,
                        seen: null,
                        retainFusedCache: true);
                    if (_handles.TryRemove(cmd.RequestId, out var handle))
                    {
                        // Aborted requests (stop button, client disconnect,
                        // stop-sequence hit in the adapter) never reach the
                        // ApplyResults finish paths, so surface speculative
                        // stats here too.
                        LogSpeculationStatsIfAny(handle.Sequence);
                        handle.CompleteAborted();
                    }
                    break;
            }
        }

        /// <summary>Mark a sequence refused at submission (nothing was scheduled for it) and
        /// return the error for the submit path to deliver to its handle.</summary>
        private static ContextLengthExceededException RefuseForContext(
            SequenceState seq, ContextLengthExceededException error)
        {
            seq.Status = SequenceStatus.FinishedError;
            seq.FinishReason = "error";
            seq.Error = error;
            return error;
        }

        // Set when an adapter was retired (ReleaseRetiredAdapters) or a retired one ran a
        // step, and cleared once no retired adapter is still bound. Engine thread only.
        private bool _adapterSweepPending;

        /// <summary>Release the model's state for every retired adapter that no request in
        /// flight runs with. Engine thread, under the GPU compute lock, between steps.</summary>
        private void SweepRetiredAdapters()
        {
            _adapterSweepPending = false;
            if (_model is not IAdapterAwareModel aware)
                return;
            foreach (var adapter in aware.BoundAdapters)
            {
                if (!adapter.IsRetired)
                    continue;
                if (_scheduler.ReferencesAdapter(adapter) || HandlesReference(adapter))
                {
                    _adapterSweepPending = true;
                    continue;
                }
                try
                {
                    aware.ReleaseAdapter(adapter);
                    _logger.LogInformation("Released LoRA adapter {Adapter}: unloaded, and no request runs with it.", adapter.Id);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Releasing LoRA adapter {Adapter} failed", adapter.Id);
                }
            }
        }

        // Submissions still queued as commands are in _handles before the scheduler sees them.
        private bool HandlesReference(IRequestAdapter adapter)
        {
            foreach (var entry in _handles)
            {
                if (ReferenceEquals(entry.Value.Sequence.Adapter, adapter))
                    return true;
            }
            return false;
        }

        private void ApplyResults(List<SequenceStepResult> results, SchedulerOutput output)
        {
            for (int i = 0; i < results.Count; i++)
            {
                var r = results[i];
                var seq = r.Sequence;
                var handle = _handles.TryGetValue(seq.RequestId, out var h) ? h : null;
                handle?.RecordForwardTime(r);

                if (r.SlotUnavailable)
                {
                    _scheduler.DeferForSequenceSlot(seq, output, _model.MaxContextLength);
                    continue;
                }

                if (r.Error != null)
                {
                    LogSpeculationStatsIfAny(seq);
                    _scheduler.NotifyError(seq, r.Error, output);
                    handle?.CompleteWithError(r.Error);
                    _handles.TryRemove(seq.RequestId, out _);
                    Interlocked.Increment(ref _totalCompleted);
                    continue;
                }

                if (r.SampledToken < 0 && r.IsPrefill && handle != null)
                    PublishEarlyFirstToken(seq, handle);

                if (r.SampledToken >= 0)
                {
                    // A speculative step emits the sampled token plus the
                    // accepted draft tokens (ExtraTokens); each gets the same
                    // per-token EOS / length checks the one-token path applied.
                    var extraTokens = r.ExtraTokens ?? Array.Empty<int>();
                    int extraCount = extraTokens.Count;
                    int totalNew = 1 + extraCount;
                    // Tokens already in OutputTokens before this step's batch;
                    // OutputTokens may not be consulted directly mid-loop
                    // because the executor appended the whole batch up front.
                    int baseCount = seq.OutputTokens.Count - totalNew;

                    // A stop the sampler turned into a reasoning close (TokenSampler.CloseInsteadOfStop).
                    var stepSampler = seq.GetOrCreateSampler();
                    if (stepSampler.StopsClosedInThinking > seq.LoggedStopClosures)
                    {
                        seq.LoggedStopClosures = stepSampler.StopsClosedInThinking;
                        _logger.LogWarning(
                            "Request {EngineRequestId} drew a stop token inside its reasoning after {Emitted} tokens; " +
                            "closing the reasoning so the model answers",
                            seq.RequestId, baseCount);
                    }

                    // Only reasoning is fed to the templated-loop detector: the channel's state after
                    // this step's tokens (a channel that closed or opened inside one step is rare,
                    // and the detector only ever asks to close an open one).
                    var reasoningLoops = _stopRepetition && (seq.SamplingConfig?.StopRepetition ?? true)
                        && _model.Tokenizer != null ? seq.ReasoningLoops : null;
                    if (reasoningLoops != null && !stepSampler.InOpenReasoning(seq.OutputTokens))
                    {
                        reasoningLoops.Reset();
                        reasoningLoops = null;
                    }

                    bool finished = false;
                    for (int t = 0; t < totalNew && !finished; t++)
                    {
                        int token = t == 0 ? r.SampledToken : extraTokens[t - 1];
                        int emittedCount = baseCount + t + 1;

                        // Stop on EOS. Do NOT publish the EOS token to the
                        // consumer channel: its textual form is a special
                        // marker (e.g. <|im_end|>) that would
                        // otherwise be decoded by AppendTokenBytes and leak
                        // into the streamed assistant output.
                        if (_model.Tokenizer != null && _model.Tokenizer.IsEos(token))
                        {
                            TruncateUnpublishedTail(seq, emittedCount);
                            LogSpeculationStatsIfAny(seq);
                            _scheduler.NotifyStop(seq, SequenceStatus.FinishedStopped, "eos", output);
                            handle?.CompleteFinished();
                            _handles.TryRemove(seq.RequestId, out _);
                            Interlocked.Increment(ref _totalCompleted);
                            finished = true;
                            break;
                        }

                        if (t == 0 && baseCount == 0 && seq.EarlyFirstTokenPublished)
                        {
                            // Streamed when the prompt finished (PublishEarlyFirstToken);
                            // every draw of the first token returns that same token.
                            if (token != seq.EarlyFirstToken)
                                throw new InvalidOperationException(
                                    $"Request {seq.RequestId} emitted first token {token} after streaming {seq.EarlyFirstToken}.");
                        }
                        else
                            handle?.PublishToken(token);

                        // Stop on max-new-tokens, or where the context window ends. Every
                        // path forwards a decode token in the step that samples it, so the
                        // token after this one would be written at position prompt+emitted:
                        // past the window once that reaches it, which the model refuses
                        // ("Requested sequence length N+1 exceeds configured max context N")
                        // and which used to fail the request mid-reply. Reported as the
                        // length cap it is, the same "max_tokens" a host clamping the reply
                        // to the room its prompt leaves (ClampGenerationReserve) would see.
                        bool contextFull = _contextLimit > 0
                            && (long)seq.PromptTokens.Count + emittedCount >= _contextLimit;
                        if (emittedCount >= seq.MaxNewTokens || contextFull)
                        {
                            if (emittedCount < seq.MaxNewTokens)
                                _logger.LogInformation(
                                    "Request {EngineRequestId} reached the model's {Context}-token context window after " +
                                    "{Emitted} of {Requested} requested tokens; finishing it as a length stop",
                                    seq.RequestId, _contextLimit, emittedCount, seq.MaxNewTokens);
                            TruncateUnpublishedTail(seq, emittedCount);
                            LogSpeculationStatsIfAny(seq);
                            _scheduler.NotifyStop(seq, SequenceStatus.FinishedLengthCapped, "max_tokens", output);
                            handle?.CompleteFinished();
                            _handles.TryRemove(seq.RequestId, out _);
                            Interlocked.Increment(ref _totalCompleted);
                            finished = true;
                            break;
                        }

                        // Stop a generation that has locked into a loop. It would
                        // otherwise run to max-new-tokens -- hundreds of thousands
                        // of tokens on a phone whose reply limit the user raised --
                        // streaming the same phrase until somebody presses Stop.
                        // Reported as its own reason so the layers above can say
                        // what happened rather than "the answer was cut off".
                        if (_stopRepetition
                            && (seq.SamplingConfig?.StopRepetition ?? true)
                            && RepetitionGuard.IsLooping(seq.OutputTokens, emittedCount, out int period, out int repeats))
                        {
                            // V4.1's trained reasoning close can recover a looping
                            // thought and leave room for the answer. The next normal
                            // sample/forward commits it, respecting cancellation,
                            // EOS, the original length cap and delayed grammars.
                            // Never discard an already-forwarded speculative tail.
                            if (emittedCount == seq.OutputTokens.Count &&
                                seq.GetOrCreateSampler().TryRequestThinkingClosure(seq.OutputTokens))
                            {
                                _logger.LogWarning(
                                    "Request {EngineRequestId} closing looping reasoning after {Emitted} tokens (period {Period})",
                                    seq.RequestId, emittedCount, period);
                                continue;
                            }
                            TruncateUnpublishedTail(seq, emittedCount);
                            LogSpeculationStatsIfAny(seq);
                            _logger.LogWarning(
                                "Request {EngineRequestId} stopped after {Emitted} tokens: {What} (period {Period})",
                                seq.RequestId, emittedCount,
                                // Quotes the repeated output, decoded: content, kept out of exported logs.
                                LogContent.Of(RepetitionGuard.Describe(seq.OutputTokens, emittedCount, period, repeats,
                                    ids => _model.Tokenizer?.Decode(ids))),
                                period);
                            _scheduler.NotifyStop(seq, SequenceStatus.FinishedStopped, RepetitionGuard.FinishReason, output);
                            handle?.CompleteFinished();
                            _handles.TryRemove(seq.RequestId, out _);
                            Interlocked.Increment(ref _totalCompleted);
                            finished = true;
                            break;
                        }

                        // Reasoning that settled into a templated loop (a line repeated with a
                        // number or a word changed: ReasoningLoopDetector) is closed so the model
                        // answers. Never a stop. Asked at the newest token only, as above.
                        if (reasoningLoops is { } loops)
                        {
                            _loopBytes.Clear();
                            _model.Tokenizer!.AppendTokenBytes(token, _loopBytes);
                            loops.Observe(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_loopBytes));
                            if (loops.Looping && emittedCount == seq.OutputTokens.Count)
                            {
                                if (stepSampler.TryRequestThinkingClosure(seq.OutputTokens))
                                    _logger.LogWarning(
                                        "Request {EngineRequestId} closing reasoning that repeated a templated line {Lines} times " +
                                        "after {Emitted} tokens: {Line}",
                                        seq.RequestId, loops.RunLines, emittedCount,
                                        // The repeated line, decoded: content, kept out of exported logs.
                                        LogContent.Of(loops.LastLine ?? string.Empty));
                                loops.Reset();
                            }
                        }
                    }
                }
            }
        }

        /// <summary>Stream a request's first token the moment its prompt is prefilled. The
        /// step that forwards it -- with speculation a whole draft and verify, ~15 ms on a 27B
        /// target -- used to run before the token was published, so every request's time to
        /// first token included one extra step. EOS is not streamed; the forwarding step
        /// still applies every stop and length check to the token.</summary>
        private void PublishEarlyFirstToken(SequenceState seq, InferenceRequestHandle handle)
        {
            if (seq.MaxNewTokens < 1 || !BatchExecutor.TryDrawEarlyFirstToken(seq, out int token))
                return;
            if (_model.Tokenizer != null && _model.Tokenizer.IsEos(token))
                return;
            handle.PublishToken(token);
            seq.EarlyFirstTokenPublished = true;
        }

        /// <summary>Drop speculatively accepted tokens past a mid-batch stop
        /// point so the sequence's recorded output matches what was streamed.</summary>
        private static void TruncateUnpublishedTail(SequenceState seq, int keepCount)
        {
            if (seq.OutputTokens.Count > keepCount)
                seq.OutputTokens.RemoveRange(keepCount, seq.OutputTokens.Count - keepCount);
            // The speculative step advanced NumComputedTokens over the whole
            // forwarded batch; the tokens just dropped are no longer part of the
            // sequence, so bring the committed-token count back in line or the
            // scheduler's prefix-cache block registration will hash positions
            // past the end of the (now shorter) token list and throw.
            seq.TrimComputedToTotalTokens();
        }

        /// <summary>Log cumulative NextN/MTP speculative-decoding counters when
        /// a request that ran speculatively finishes.</summary>
        private void LogSpeculationStatsIfAny(SequenceState seq)
        {
            var st = seq.SpecStats;
            if (st == null || (st.VerifySteps + st.PlainSteps) == 0)
                return;
            _logger.LogInformation(
                "Speculative decoding stats for {EngineRequestId}: drafted={Drafted} accepted={Accepted} acceptance={Acceptance:P0} verifySteps={VerifySteps} plainSteps={PlainSteps} rollbacks={Rollbacks} | phaseMs draft={DraftMs:F0} verify={VerifyMs:F0} snapshot={SnapshotMs:F0} rollback={RollbackMs:F0} catchUp={CatchUpMs:F0} plain={PlainMs:F0} | governor plain={GovPlain:F1}ms/tok spec={GovSpec:F1}ms/tok wins={GovWins} losses={GovLosses} parked={GovParked}",
                seq.RequestId, st.TokensDrafted, st.TokensAccepted, st.AcceptanceRate,
                st.VerifySteps, st.PlainSteps, st.RollbackSteps,
                st.DraftMs, st.VerifyMs, st.SnapshotMs, st.RollbackMs, st.CatchUpMs, st.PlainMs,
                st.PlainMsPerToken, st.SpecMsPerToken, st.GovernorWins, st.GovernorLosses, st.GovernorParkedSteps);
        }

        internal static long ComputeBlockByteSize(IModelArchitecture model, int blockSize)
        {
            if (!model.SupportsKVStateSnapshot) return 0;
            long size = model.ComputeKVBlockByteSize(blockSize);
            return Math.Max(size, 0);
        }

        /// <summary>Resolve the physical block-table capacity. By default it is
        /// large enough for one full model context, or one context per allowed
        /// native V4.1 sequence slot. Only metadata is allocated up front and the
        /// comparatively large snapshot slabs remain lazy. An
        /// explicit TS_SCHED_NUM_BLOCKS value is a hard operator limit.</summary>
        private static int ResolveEffectiveNumBlocks(
            IModelArchitecture model,
            SchedulerConfig cfg,
            ILogger logger)
        {
            int numBlocks = cfg.NumBlocks;
            string? rawOverride = Environment.GetEnvironmentVariable("TS_SCHED_NUM_BLOCKS");
            bool explicitOverride = int.TryParse(rawOverride, out int overrideBlocks)
                && overrideBlocks > 0;
            if (explicitOverride || cfg.BlockSize <= 0)
                return numBlocks;

            int contextLength = model.MaxContextLength;
            if (contextLength <= 0)
                return numBlocks;

            // V4.1 stores complete per-request KV state in native slots. Its
            // zero-byte block pool only tracks scheduling reservations; limiting
            // that pool to one aggregate context needlessly evicts healthy slots
            // and recomputes their prompts. Size metadata for all admitted slots.
            // Other architectures retain their existing shared-pool semantics.
            int contexts = UsesNativeDeepSeek41Slots(model) ? Math.Max(1, cfg.MaxNumRunningSequences) : 1;
            long neededLong = ((long)contextLength + cfg.BlockSize - 1) / cfg.BlockSize * contexts;
            if (neededLong > int.MaxValue)
                throw new InvalidOperationException(
                    $"Model context {contextLength} across {contexts} sequence context(s) requires too many KV blocks " +
                    $"at block size {cfg.BlockSize}.");

            int neededBlocks = (int)neededLong;
            if (neededBlocks > numBlocks)
            {
                logger?.LogInformation(
                    "Sizing KV block pool to model context: {ContextTokens} tokens x {SequenceContexts} context(s) -> " +
                    "{Blocks} blocks of {BlockSize} (configured default was {ConfiguredBlocks}). " +
                    "Set TS_SCHED_NUM_BLOCKS to impose an explicit hard limit.",
                    contextLength, contexts, neededBlocks, cfg.BlockSize, numBlocks);
                numBlocks = neededBlocks;
            }

            return numBlocks;
        }

        /// <summary>Let a model with its own lone-prompt steps (<see cref="ISoloPrefillBudgetModel"/>)
        /// size them, unless the configuration chose <see cref="SchedulerConfig.SoloPrefillChunkSize"/>.
        /// Returns the policy attached, or null.</summary>
        internal static ISoloPrefillBudgetModel? AttachSoloPrefillPolicy(object model, SchedulerConfig cfg,
            ContinuousBatchScheduler scheduler, ILogger logger)
        {
            if (model is not ISoloPrefillBudgetModel solo)
                return null;
            if (cfg.SoloPrefillChunkSizeChosen)
            {
                logger.LogInformation(
                    "Lone prompts prefill in the configured steps of up to {Tokens} tokens (TS_SCHED_SOLO_PREFILL_CHUNK, " +
                    "bounded by TS_SCHED_MAX_BATCHED_TOKENS); the model's own lone-prompt steps are not used",
                    Math.Min(cfg.SoloPrefillChunkSize, cfg.MaxNumBatchedTokens));
                return null;
            }
            int step = solo.PreferredSoloPrefillTokens(0, int.MaxValue / 2);
            if (step <= 0)
            {
                logger.LogInformation(
                    "The model has no lone-prompt step of its own here (its prefill does not pipeline across devices); " +
                    "lone prompts prefill in the scheduler's steps of up to {Tokens} tokens",
                    Math.Min(cfg.SoloPrefillChunkSize, cfg.MaxNumBatchedTokens));
                return null;
            }
            scheduler.SetSoloPrefillPolicy(solo);
            logger.LogInformation(
                "Lone prompts prefill in steps of up to about {Tokens} tokens chosen by the model (it pipelines each " +
                "step across its devices); TS_SCHED_SOLO_PREFILL_CHUNK sets the scheduler's own step instead " +
                "(bounded by TS_SCHED_MAX_BATCHED_TOKENS)", step);
            return solo;
        }

        /// <summary>
        /// The pool block size when the configuration leaves it to the model. A family whose only
        /// cross-request reuse is its pages (no per-conversation end state: Hunyuan Dense, Mistral 3,
        /// Nemotron-H, Muse-Glimmer) reuses whole blocks only, so at 256 tokens every turn recomputed its
        /// last partial block and a turn shorter than one reused nothing: eight parallel Hunyuan
        /// conversations reused 0 tokens on their second turn, and at 16 all but the last partial block,
        /// in 26% less time. A family with end states keeps each conversation's exact state, and its pool
        /// only accounts for reservations.
        /// </summary>
        internal static int PreferredBlockSize(IModelArchitecture model)
            => model is PrefixCache.IPrefixCacheModel cache
               && cache.GetPrefixCacheCapabilities() is { EndState: PrefixCache.EndStateSupport.None } caps
               && caps.Pages != PrefixCache.PageSupport.None
                ? SchedulerConfig.PageFamilyBlockSize
                : SchedulerConfig.DefaultBlockSize;

        private static bool UsesNativeDeepSeek41Slots(IModelArchitecture model)
            => string.Equals(model.Config?.Architecture, "deepseek41", StringComparison.OrdinalIgnoreCase)
                && !model.SupportsKVStateSnapshot
                && model is IBatchedPagedModel { SupportsPerSequenceFusedForward: true };

        private struct EngineCommand
        {
            public EngineCommandKind Kind;
            public SequenceState Sequence;
            public string RequestId;
            public SpeculationOptions Speculation;
        }

        private enum EngineCommandKind
        {
            Submit,
            Abort,
            Trim,
            Speculation,
            ReleaseAdapters,
        }
    }
}
