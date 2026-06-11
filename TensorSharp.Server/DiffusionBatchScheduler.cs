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
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Models;

namespace TensorSharp.Server
{
    /// <summary>A live preview canvas from a batched diffusion request: the request's committed response so
    /// far plus its current block's best-guess (argmax) canvas, with "replace" semantics.</summary>
    internal readonly record struct DiffusionPreview(int Block, int Step, int TotalSteps, int[] Tokens);

    /// <summary>Streaming handle returned by <see cref="DiffusionBatchScheduler.Submit"/>: per-step previews
    /// plus a task that completes with the final committed token sequence.</summary>
    internal sealed class DiffusionRequestHandle
    {
        public ChannelReader<DiffusionPreview> Previews { get; }
        public Task<List<int>> Completion { get; }

        public DiffusionRequestHandle(ChannelReader<DiffusionPreview> previews, Task<List<int>> completion)
        {
            Previews = previews;
            Completion = completion;
        }
    }

    /// <summary>
    /// The continuous-batching scheduler for DiffusionGemma — the diffusion analog of the autoregressive
    /// <see cref="TensorSharp.Runtime.Scheduling.InferenceEngine"/>. DiffusionGemma is not thread-safe on the
    /// GPU (concurrent GGML compute from two threads aborts the process), so true parallelism is achieved by
    /// BATCHING within a single compute thread: one background worker owns <c>model.GpuComputeLock</c> and
    /// denoises every in-flight request's canvas together (one batched forward per step). The weight-bound
    /// work (embedding, dense MLP, 128-expert MoE, lm_head) runs once over all sequences' canvas tokens, so
    /// aggregate throughput scales with the batch size; only the per-sequence attention loops.
    ///
    /// Requests are admitted and retired at BLOCK boundaries (block-granular continuous batching): a request
    /// arriving while a block is in flight joins on the next block; a request that ends (or hits its block
    /// budget) is retired and its result published. This fixes the previous behaviour where a second parallel
    /// request produced no output until the first fully finished (the per-request worker held the GPU lock for
    /// the entire multi-minute generation).
    /// </summary>
    internal sealed class DiffusionBatchScheduler : IDisposable
    {
        private readonly DiffusionGemmaModel _model;
        private readonly DiffusionGemmaSampler _sampler;
        private readonly ILogger _logger;
        private readonly int _maxBatch;

        private readonly object _pendingLock = new();
        private readonly Queue<PendingRequest> _pending = new();
        private readonly SemaphoreSlim _signal = new(0);
        private readonly CancellationTokenSource _stop = new();
        private readonly Thread _worker;
        private volatile int _activeCount;

        public DiffusionBatchScheduler(DiffusionGemmaModel model, ILogger logger, int maxBatch)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _logger = logger ?? NullLogger.Instance;
            _sampler = new DiffusionGemmaSampler(model);
            _maxBatch = Math.Max(1, maxBatch);
            _worker = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = "diffusion-batch-scheduler",
            };
            _worker.Start();
        }

        /// <summary>Number of requests currently being denoised in the active batch (for status / metrics).</summary>
        public int ActiveCount => _activeCount;

        /// <summary>Submit a request. Returns immediately with a handle that streams per-step previews and a
        /// task that completes with the final token sequence. Thread-safe; callable from any request thread.</summary>
        public DiffusionRequestHandle Submit(int[] promptTokens, DiffusionEbParams p, CancellationToken ct)
        {
            var channel = Channel.CreateUnbounded<DiffusionPreview>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = false,   // the worker writes; completion may race the writer
            });
            var tcs = new TaskCompletionSource<List<int>>(TaskCreationOptions.RunContinuationsAsynchronously);
            var req = new PendingRequest(promptTokens, p, ct, channel, tcs);

            if (_stop.IsCancellationRequested)
            {
                tcs.TrySetCanceled(ct);
                channel.Writer.TryComplete();
                return new DiffusionRequestHandle(channel.Reader, tcs.Task);
            }

            lock (_pendingLock) { _pending.Enqueue(req); }
            _signal.Release();
            return new DiffusionRequestHandle(channel.Reader, tcs.Task);
        }

        private void WorkerLoop()
        {
            var active = new List<ActiveRequest>();
            var stopCt = _stop.Token;
            try
            {
                while (!stopCt.IsCancellationRequested)
                {
                    AdmitPending(active);

                    if (active.Count == 0)
                    {
                        // Idle: block until a request arrives (or shutdown).
                        try { _signal.Wait(stopCt); }
                        catch (OperationCanceledException) { break; }
                        continue;
                    }

                    _activeCount = active.Count;

                    var runs = new List<DiffusionSeqRun>(active.Count);
                    foreach (var x in active) runs.Add(x.Run);

                    try
                    {
                        lock (_model.GpuComputeLock)
                        {
                            _sampler.RunBlockBatched(runs, stopCt);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "DiffusionGemma batched block failed for {Count} active request(s)", active.Count);
                        foreach (var x in active)
                        {
                            x.Req.Tcs.TrySetException(ex);
                            x.Req.Channel.Writer.TryComplete(ex);
                            SafeDispose(x.Run.State);
                        }
                        active.Clear();
                        _activeCount = 0;
                        continue;
                    }

                    // Retire finished / cancelled requests; the rest carry over to the next block.
                    for (int i = active.Count - 1; i >= 0; i--)
                    {
                        var x = active[i];
                        bool cancelled = x.Req.Ct.IsCancellationRequested;
                        if (x.Run.Done || cancelled)
                        {
                            if (cancelled) x.Req.Tcs.TrySetCanceled(x.Req.Ct);
                            else x.Req.Tcs.TrySetResult(x.Run.Response);
                            x.Req.Channel.Writer.TryComplete();
                            SafeDispose(x.Run.State);
                            active.RemoveAt(i);
                        }
                    }
                    _activeCount = active.Count;
                }
            }
            finally
            {
                foreach (var x in active)
                {
                    x.Req.Tcs.TrySetCanceled();
                    x.Req.Channel.Writer.TryComplete();
                    SafeDispose(x.Run.State);
                }
                lock (_pendingLock)
                {
                    while (_pending.Count > 0)
                    {
                        var r = _pending.Dequeue();
                        r.Tcs.TrySetCanceled();
                        r.Channel.Writer.TryComplete();
                    }
                }
                _activeCount = 0;
            }
        }

        private void AdmitPending(List<ActiveRequest> active)
        {
            lock (_pendingLock)
            {
                while (active.Count < _maxBatch && _pending.Count > 0)
                {
                    var req = _pending.Dequeue();
                    if (req.Ct.IsCancellationRequested)
                    {
                        req.Tcs.TrySetCanceled(req.Ct);
                        req.Channel.Writer.TryComplete();
                        continue;
                    }

                    var state = _model.CreateSeqState();
                    var capturedReq = req;
                    var run = new DiffusionSeqRun(req.PromptTokens, req.Params, state, req.Ct,
                        (r, step, total, preview) =>
                            capturedReq.Channel.Writer.TryWrite(new DiffusionPreview(r.BlockIndex, step, total, preview)));
                    active.Add(new ActiveRequest(run, req));
                }
            }
        }

        private void SafeDispose(DiffusionSeqState state)
        {
            try
            {
                lock (_model.GpuComputeLock)
                    _model.DisposeSeqState(state);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "DiffusionGemma sequence-state disposal failed");
            }
        }

        public void Dispose()
        {
            _stop.Cancel();
            _signal.Release();
            try { _worker.Join(TimeSpan.FromSeconds(10)); }
            catch { /* best effort */ }
            _signal.Dispose();
            _stop.Dispose();
        }

        private sealed class PendingRequest
        {
            public int[] PromptTokens { get; }
            public DiffusionEbParams Params { get; }
            public CancellationToken Ct { get; }
            public Channel<DiffusionPreview> Channel { get; }
            public TaskCompletionSource<List<int>> Tcs { get; }

            public PendingRequest(int[] promptTokens, DiffusionEbParams p, CancellationToken ct,
                Channel<DiffusionPreview> channel, TaskCompletionSource<List<int>> tcs)
            {
                PromptTokens = promptTokens;
                Params = p;
                Ct = ct;
                Channel = channel;
                Tcs = tcs;
            }
        }

        private readonly struct ActiveRequest
        {
            public DiffusionSeqRun Run { get; }
            public PendingRequest Req { get; }
            public ActiveRequest(DiffusionSeqRun run, PendingRequest req) { Run = run; Req = req; }
        }
    }
}
