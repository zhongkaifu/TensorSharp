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
    /// Replaces the old <c>InferenceQueue</c> + <c>SessionKvCacheManager</c>:
    /// the engine is the single coordination point for everything that needs
    /// the model's KV state, and request lifecycle is per-request not per-
    /// session.
    /// </summary>
    public sealed class InferenceEngine : IDisposable
    {
        private readonly IModelArchitecture _model;
        private readonly SchedulerConfig _cfg;
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
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            _logger = logger ?? NullLogger.Instance;

            long blockBytes = ComputeBlockByteSize(model, cfg.BlockSize);
            _pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, blockBytes);
            _scheduler = new ContinuousBatchScheduler(cfg, _pool, model.KVStateFingerprint ?? string.Empty, logger);
            _executor = new BatchExecutor(model, _pool, _scheduler, logger);

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
                SchedulerOutput output;
                List<SequenceStepResult> results;
                try
                {
                    output = _scheduler.Schedule();
                    if (output.IsEmpty && _scheduler.RunningCount == 0)
                        continue;

                    results = _executor.ExecuteStep(output);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Engine worker step failed");
                    continue;
                }
                Interlocked.Increment(ref _totalStepsRun);
                Interlocked.Add(ref _totalForwardTicks, sw.ElapsedTicks);

                // Post-step: emit tokens, detect stop conditions, finish sequences.
                ApplyResults(results, output);
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
                        handle.CompleteAborted();
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
                    _scheduler.NotifyStop(seq, SequenceStatus.FinishedError, "error", output);
                    handle?.CompleteWithError(r.Error);
                    _handles.TryRemove(seq.RequestId, out _);
                    Interlocked.Increment(ref _totalCompleted);
                    continue;
                }

                if (r.SampledToken >= 0)
                {
                    handle?.PublishToken(r.SampledToken);

                    // Stop on EOS.
                    if (_model.Tokenizer != null && _model.Tokenizer.IsEos(r.SampledToken))
                    {
                        _scheduler.NotifyStop(seq, SequenceStatus.FinishedStopped, "eos", output);
                        handle?.CompleteFinished();
                        _handles.TryRemove(seq.RequestId, out _);
                        Interlocked.Increment(ref _totalCompleted);
                        continue;
                    }

                    // Stop on max-new-tokens.
                    if (seq.OutputTokens.Count >= seq.MaxNewTokens)
                    {
                        _scheduler.NotifyStop(seq, SequenceStatus.FinishedLengthCapped, "max_tokens", output);
                        handle?.CompleteFinished();
                        _handles.TryRemove(seq.RequestId, out _);
                        Interlocked.Increment(ref _totalCompleted);
                        continue;
                    }
                }
            }
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
