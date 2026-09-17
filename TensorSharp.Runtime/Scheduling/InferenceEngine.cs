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
using TensorSharp.Runtime.Paged;
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
    /// Replaces the old FIFO queue plus per-session KV manager: the engine is
    /// the single coordination point for everything that needs the model's KV
    /// state, and request lifecycle is per-request rather than per-session.
    /// </summary>
    public sealed class InferenceEngine : IDisposable
    {
        private readonly IModelArchitecture _model;
        private readonly ILogger _logger;
        private readonly bool _stopRepetition;
        private readonly int _nativeSlotContextLimit;
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

        public InferenceEngine(IModelArchitecture model, SchedulerConfig cfg, ILogger logger = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            ArgumentNullException.ThrowIfNull(cfg);
            _logger = logger ?? NullLogger.Instance;
            _stopRepetition = cfg.StopRepetition;   // cfg is null-checked above
            _nativeSlotContextLimit = UsesNativeDeepSeek41Slots(model) ? Math.Max(0, model.MaxContextLength) : 0;

            long blockBytes = ComputeBlockByteSize(model, cfg.BlockSize);
            int numBlocks = ResolveEffectiveNumBlocks(model, cfg, _logger);
            _pool = new BlockPool(numBlocks, cfg.BlockSize, blockBytes);
            _scheduler = new ContinuousBatchScheduler(cfg, _pool, model.KVStateFingerprint ?? string.Empty, logger,
                supportsCrossSequenceKvReuse: model.SupportsCrossSequenceKvReuse,
                maxReusablePrefixTokens: model.MaxReusablePrefixTokens,
                requiresPerBlockCapture: model.RequiresPerBlockCapture,
                supportsReuseAcrossMediaSpan: model.SupportsReuseAcrossMediaSpan,
                canPrefillMediaAfterReusedPrefix: model.CanPrefillMediaAfterReusedPrefix);
            _executor = new BatchExecutor(model, _pool, _scheduler, logger);
            // Let the scheduler plan same-session live-cache continuations through the
            // executor (which owns the model's live KV-cache state).
            _scheduler.AttachLiveCacheContinuation(
                _executor.ComputeLiveContinuationLcp,
                _executor.TryAdoptLiveCache);
            // Cross-request prefix reuse for concurrent (per-seq fused) decode:
            // re-adopt a finished request's complete retained holder (K/V and,
            // for hybrid models, recurrent state) for a follow-up turn.
            _scheduler.AttachFusedCacheContinuation(
                _executor.ComputeFusedContinuationLcp,
                _executor.TryAdoptFusedContinuation);
            // So admission can say WHY a turn reused nothing. The mechanisms record
            // their reasons; only the scheduler knows which one ended up serving the
            // request, so only it can report the outcome without guessing.
            _scheduler.AttachReuseDiagnostics(
                () => _executor.LastLiveContinuationDeclineReason,
                () => _executor.LastFusedContinuationDeclineReason,
                () => _executor.LastFusedAdoptionSource,
                () => _executor.LastBlockedByScopeTokens);
            // Shared-prefix checkpoints: end a prefill chunk exactly where the chat
            // layer says the shared prompt ends, so the executor can copy the model's
            // state there and start every later new chat from that copy.
            if (_executor.PrefixCheckpointsSupported)
                _scheduler.EnablePrefixCheckpoints();

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
        }

        public IModelArchitecture Model => _model;
        public BlockPoolStats PoolStats => _pool.GetStats();
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
        public ComputeGate ComputeGate
        {
            get => Volatile.Read(ref _computeGate);
            set => Volatile.Write(ref _computeGate, value);
        }

        private ComputeGate _computeGate;
        private long _stepsHeldByGate;

        /// <summary>
        /// Where shared-prefix checkpoints outlive the process, or null for nowhere;
        /// forwarded to the executor, which reads it on its own thread. See
        /// <see cref="IPrefixCheckpointStore"/>.
        /// </summary>
        public IPrefixCheckpointStore PrefixCheckpointStore
        {
            get => _executor.PrefixCheckpointStore;
            set => _executor.PrefixCheckpointStore = value;
        }

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
            // Nobody is going to finish these now. A consumer awaiting one of them --
            // another conversation's turn, on a phone -- would otherwise wait forever.
            var abandoned = new ObjectDisposedException(nameof(InferenceEngine),
                "The inference engine was shut down while this request was in flight.");
            foreach (var entry in _handles)
            {
                if (_handles.TryRemove(entry.Key, out var handle))
                    handle.CompleteWithError(abandoned);
            }
        }

        private void WorkerLoop()
        {
            var sw = new System.Diagnostics.Stopwatch();
            while (!_shutdownCts.IsCancellationRequested)
            {
                // Drain queued commands (non-blocking).
                while (_commands.Reader.TryRead(out var cmd))
                {
                    ApplyCommand(cmd);
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

                // Run one scheduler step.
                sw.Restart();
                SchedulerOutput output = null;
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

                Interlocked.Increment(ref _totalStepsRun);
                Interlocked.Add(ref _totalForwardTicks, sw.ElapsedTicks);

                // Post-step: emit tokens, detect stop conditions, finish sequences.
                ApplyResults(results, output);

                // Notify the model about sequences whose per-request state can
                // now be reclaimed (finished, preempted, errored). Hybrid
                // models (Nemotron-H, Qwen 3.5) allocate Mamba2 / GatedDeltaNet
                // recurrent-state slots keyed by RequestId; without this
                // notification the slot pool grows unbounded and slot indices
                // get reused incorrectly across abandoned sequences.
                NotifyReleasedSequences(output);
            }
        }

        private void NotifyReleasedSequences(SchedulerOutput output)
        {
            if (_model is not Runtime.Scheduling.IBatchedPagedModel batched) return;
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
                        "Failed to release scheduler state for stalled sequence {RequestId}",
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

        private void FailStepSequences(Exception ex, SchedulerOutput output, string phase)
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
                        "Failed to release scheduler state for errored sequence {RequestId}",
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

        private List<SequenceState> GetAffectedSequences(SchedulerOutput output)
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
            if (_model is not Runtime.Scheduling.IBatchedPagedModel batched) return;
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var id in requestIds)
                NotifyReleasedSequence(batched, id, seen);
        }

        private void NotifyReleasedSequence(
            Runtime.Scheduling.IBatchedPagedModel batched,
            string requestId,
            HashSet<string> seen,
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
                        ? "Fused-cache retention failed for sequence {RequestId}"
                        : "Fused-cache bookkeeping cleanup failed for sequence {RequestId}",
                    requestId);
            }

            try
            {
                batched.OnSequenceReleased(requestId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Model release hook failed for sequence {RequestId}", requestId);
            }
        }

        private void ApplyCommand(EngineCommand cmd)
        {
            switch (cmd.Kind)
            {
                case EngineCommandKind.Submit:
                    try
                    {
                        // A larger metadata pool accounts for independent native
                        // slots; it must not enlarge any individual slot's context.
                        long requested = (long)cmd.Sequence.PromptTokens.Count + cmd.Sequence.MaxNewTokens;
                        if (_nativeSlotContextLimit > 0 && requested > _nativeSlotContextLimit)
                        {
                            var error = new InvalidOperationException(
                                $"DeepSeek V4.1 request requires {requested} tokens, but each native sequence slot " +
                                $"has a context limit of {_nativeSlotContextLimit}. Shorten the prompt or output allowance.");
                            cmd.Sequence.Status = SequenceStatus.FinishedError;
                            cmd.Sequence.FinishReason = "error";
                            cmd.Sequence.Error = error;
                            throw error;
                        }
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

                case EngineCommandKind.Speculation:
                    _executor.SetSpeculation(cmd.Speculation);
                    _logger.LogInformation(
                        "Speculation policy updated on the host's request: enabled={Enabled} algorithm={Algorithm} maxDraft={MaxDraft}",
                        cmd.Speculation.Enabled, cmd.Speculation.SpeculatorName, cmd.Speculation.MaxDraftTokens);
                    break;

                case EngineCommandKind.Abort:
                    _scheduler.Abort(cmd.RequestId);
                    if (_model is Runtime.Scheduling.IBatchedPagedModel batchedAbort)
                    {
                        // Abort bypasses ApplyResults/NotifyReleasedSequences, so run
                        // the same executor-first cleanup before freeing model state.
                        // The executor keeps a cleanly stopped sequence's holder (the
                        // Stop button is the ordinary way a phone turn ends) and
                        // declines anything inconsistent itself.
                        NotifyReleasedSequence(
                            batchedAbort,
                            cmd.RequestId,
                            seen: null,
                            retainFusedCache: true);
                    }
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

        private void ApplyResults(List<SequenceStepResult> results, SchedulerOutput output)
        {
            for (int i = 0; i < results.Count; i++)
            {
                var r = results[i];
                var seq = r.Sequence;
                var handle = _handles.TryGetValue(seq.RequestId, out var h) ? h : null;

                if (r.Error != null)
                {
                    LogSpeculationStatsIfAny(seq);
                    _scheduler.NotifyError(seq, r.Error, output);
                    handle?.CompleteWithError(r.Error);
                    _handles.TryRemove(seq.RequestId, out _);
                    Interlocked.Increment(ref _totalCompleted);
                    continue;
                }

                if (r.SampledToken >= 0)
                {
                    // A speculative step emits the sampled token plus the
                    // accepted draft tokens (ExtraTokens); each gets the same
                    // per-token EOS / length checks the one-token path applied.
                    int extraCount = r.ExtraTokens?.Count ?? 0;
                    int totalNew = 1 + extraCount;
                    // Tokens already in OutputTokens before this step's batch;
                    // OutputTokens may not be consulted directly mid-loop
                    // because the executor appended the whole batch up front.
                    int baseCount = seq.OutputTokens.Count - totalNew;

                    bool finished = false;
                    for (int t = 0; t < totalNew && !finished; t++)
                    {
                        int token = t == 0 ? r.SampledToken : r.ExtraTokens[t - 1];
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

                        handle?.PublishToken(token);

                        // Stop on max-new-tokens.
                        if (emittedCount >= seq.MaxNewTokens)
                        {
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
                                    "Request {RequestId} closing looping reasoning after {Emitted} tokens (period {Period})",
                                    seq.RequestId, emittedCount, period);
                                continue;
                            }
                            TruncateUnpublishedTail(seq, emittedCount);
                            LogSpeculationStatsIfAny(seq);
                            _logger.LogWarning(
                                "Request {RequestId} stopped after {Emitted} tokens: {What} (period {Period})",
                                seq.RequestId, emittedCount,
                                RepetitionGuard.Describe(seq.OutputTokens, emittedCount, period, repeats,
                                    ids => _model.Tokenizer?.Decode(ids)),
                                period);
                            _scheduler.NotifyStop(seq, SequenceStatus.FinishedStopped, RepetitionGuard.FinishReason, output);
                            handle?.CompleteFinished();
                            _handles.TryRemove(seq.RequestId, out _);
                            Interlocked.Increment(ref _totalCompleted);
                            finished = true;
                            break;
                        }
                    }
                }
            }
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
                "Speculative decoding stats for {RequestId}: drafted={Drafted} accepted={Accepted} acceptance={Acceptance:P0} verifySteps={VerifySteps} plainSteps={PlainSteps} rollbacks={Rollbacks} | phaseMs draft={DraftMs:F0} verify={VerifyMs:F0} snapshot={SnapshotMs:F0} rollback={RollbackMs:F0} catchUp={CatchUpMs:F0} plain={PlainMs:F0} | governor plain={GovPlain:F1}ms/tok spec={GovSpec:F1}ms/tok wins={GovWins} losses={GovLosses} parked={GovParked}",
                seq.RequestId, st.TokensDrafted, st.TokensAccepted, st.AcceptanceRate,
                st.VerifySteps, st.PlainSteps, st.RollbackSteps,
                st.DraftMs, st.VerifyMs, st.SnapshotMs, st.RollbackMs, st.CatchUpMs, st.PlainMs,
                st.PlainMsPerToken, st.SpecMsPerToken, st.GovernorWins, st.GovernorLosses, st.GovernorParkedSteps);
        }

        private static long ComputeBlockByteSize(IModelArchitecture model, int blockSize)
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
            string rawOverride = Environment.GetEnvironmentVariable("TS_SCHED_NUM_BLOCKS");
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
        }
    }
}
