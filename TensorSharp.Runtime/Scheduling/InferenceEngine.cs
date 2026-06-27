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
        private readonly BlockPool _pool;
        private readonly ContinuousBatchScheduler _scheduler;
        private readonly BatchExecutor _executor;

        private readonly ConcurrentDictionary<string, InferenceRequestHandle> _handles = new();
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

            long blockBytes = ComputeBlockByteSize(model, cfg.BlockSize);
            _pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, blockBytes);
            _scheduler = new ContinuousBatchScheduler(cfg, _pool, model.KVStateFingerprint ?? string.Empty, logger,
                supportsCrossSequenceKvReuse: model.SupportsCrossSequenceKvReuse,
                maxReusablePrefixTokens: model.MaxReusablePrefixTokens);
            _executor = new BatchExecutor(model, _pool, _scheduler, logger);
            // Let the scheduler plan same-session live-cache continuations through the
            // executor (which owns the model's live KV-cache state).
            _scheduler.AttachLiveCacheContinuation(
                _executor.ComputeLiveContinuationLcp,
                _executor.TryAdoptLiveCache);
            // Cross-request prefix reuse for concurrent (per-seq fused) decode:
            // re-adopt a finished request's retained KV holder for a follow-up turn.
            _scheduler.AttachFusedCacheContinuation(
                _executor.ComputeFusedContinuationLcp,
                _executor.TryAdoptFusedContinuation);

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

        /// <summary>Submit a sequence for inference. Returns immediately with a
        /// handle whose <see cref="InferenceRequestHandle.Tokens"/> channel
        /// streams sampled tokens.</summary>
        public InferenceRequestHandle SubmitRequest(SequenceState seq, CancellationToken ct = default)
        {
            if (seq == null) throw new ArgumentNullException(nameof(seq));
            var handle = new InferenceRequestHandle(seq, this, ct);
            _handles[seq.RequestId] = handle;
            Interlocked.Increment(ref _totalSubmitted);

            _commands.Writer.TryWrite(new EngineCommand
            {
                Kind = EngineCommandKind.Submit,
                Sequence = seq,
            });
            return handle;
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

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _shutdownCts.Cancel();
            _commands.Writer.TryComplete();
            try { _worker.Join(2000); } catch { /* best effort */ }
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

                // Run one scheduler step.
                sw.Restart();
                SchedulerOutput output = null;
                List<SequenceStepResult> results;
                try
                {
                    output = _scheduler.Schedule();
                    if (output.IsEmpty && _scheduler.RunningCount == 0)
                        continue;
                }
                catch (Exception ex)
                {
                    FailStepSequences(ex, output, "scheduler");
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
            HashSet<string> seen)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            if (seen != null && !seen.Add(requestId)) return;

            try
            {
                // Give the executor first refusal: a fused sequence that finished
                // cleanly has its per-request KV holder RETAINED (re-keyed out of the
                // active set) for cross-request prefix reuse, so the model release
                // below no-ops for it instead of disposing the still-useful K/V.
                _executor.TryRetainReleasedFusedCache(requestId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Fused-cache retention failed for sequence {RequestId}", requestId);
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
                    try { _scheduler.Submit(cmd.Sequence); }
                    catch (Exception ex)
                    {
                        if (_handles.TryRemove(cmd.Sequence.RequestId, out var h))
                            h.CompleteWithError(ex);
                    }
                    break;

                case EngineCommandKind.Abort:
                    _scheduler.Abort(cmd.RequestId);
                    if (_handles.TryRemove(cmd.RequestId, out var handle))
                    {
                        // Aborted requests (stop button, client disconnect,
                        // stop-sequence hit in the adapter) never reach the
                        // ApplyResults finish paths, so surface speculative
                        // stats here too.
                        LogMtpStatsIfAny(handle.Sequence);
                        handle.CompleteAborted();
                    }
                    if (_model is Runtime.Scheduling.IBatchedPagedModel batchedAbort)
                        batchedAbort.OnSequenceReleased(cmd.RequestId);
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
                    LogMtpStatsIfAny(seq);
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
                        // marker (e.g. <end_of_turn>, <|im_end|>) that would
                        // otherwise be decoded by AppendTokenBytes and leak
                        // into the streamed assistant output.
                        if (_model.Tokenizer != null && _model.Tokenizer.IsEos(token))
                        {
                            TruncateUnpublishedTail(seq, emittedCount);
                            LogMtpStatsIfAny(seq);
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
                            LogMtpStatsIfAny(seq);
                            _scheduler.NotifyStop(seq, SequenceStatus.FinishedLengthCapped, "max_tokens", output);
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
        private void LogMtpStatsIfAny(SequenceState seq)
        {
            var st = seq.SpecStats;
            if (st == null || (st.VerifySteps + st.PlainSteps) == 0)
                return;
            _logger.LogInformation(
                "MTP speculative stats for {RequestId}: drafted={Drafted} accepted={Accepted} acceptance={Acceptance:P0} verifySteps={VerifySteps} plainSteps={PlainSteps} rollbacks={Rollbacks} | phaseMs draft={DraftMs:F0} verify={VerifyMs:F0} snapshot={SnapshotMs:F0} rollback={RollbackMs:F0} catchUp={CatchUpMs:F0} plain={PlainMs:F0}",
                seq.RequestId, st.TokensDrafted, st.TokensAccepted, st.AcceptanceRate,
                st.VerifySteps, st.PlainSteps, st.RollbackSteps,
                st.DraftMs, st.VerifyMs, st.SnapshotMs, st.RollbackMs, st.CatchUpMs, st.PlainMs);
        }

        private static long ComputeBlockByteSize(IModelArchitecture model, int blockSize)
        {
            if (!model.SupportsKVStateSnapshot) return 0;
            long size = model.ComputeKVBlockByteSize(blockSize);
            return Math.Max(size, 0);
        }

        private struct EngineCommand
        {
            public EngineCommandKind Kind;
            public SequenceState Sequence;
            public string RequestId;
        }

        private enum EngineCommandKind
        {
            Submit,
            Abort,
        }
    }
}
