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
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Server
{
    /// <summary>
    /// FIFO request queue that ensures only one inference runs at a time,
    /// preserving KV cache stability. Provides real-time position tracking
    /// so clients can see how many requests are ahead of theirs.
    /// </summary>
    public class InferenceQueue
    {
        private readonly object _sync = new();
        private readonly LinkedList<QueueTicket> _waiters = new();
        private readonly ILogger<InferenceQueue> _logger;
        private bool _busy;
        private long _totalProcessed;
        private string _currentRequestId;

        public InferenceQueue()
            : this(NullLogger<InferenceQueue>.Instance)
        {
        }

        public InferenceQueue(ILogger<InferenceQueue> logger)
        {
            _logger = logger ?? NullLogger<InferenceQueue>.Instance;
        }

        public int PendingCount { get { lock (_sync) return _waiters.Count; } }
        public long TotalProcessed => Interlocked.Read(ref _totalProcessed);
        public bool IsBusy { get { lock (_sync) return _busy; } }

        public QueueTicket Enqueue(CancellationToken ct, string requestId = null)
        {
            lock (_sync)
            {
                var ticket = new QueueTicket(this, ct, requestId);
                if (!_busy)
                {
                    _busy = true;
                    _currentRequestId = requestId;
                    ticket.SetPosition(0);
                    ticket.MarkReady();
                    _logger.LogDebug(LogEventIds.QueueReady,
                        "Inference slot acquired immediately for request {RequestId}",
                        requestId ?? "(none)");
                    return ticket;
                }
                var node = _waiters.AddLast(ticket);
                ticket.Node = node;
                ticket.SetPosition(_waiters.Count);
                _logger.LogInformation(LogEventIds.QueueEnqueued,
                    "Request {RequestId} queued at position {Position} (pending={PendingCount}, current={CurrentRequestId})",
                    requestId ?? "(none)", _waiters.Count, _waiters.Count, _currentRequestId ?? "(none)");
                return ticket;
            }
        }

        internal void Release(QueueTicket completed)
        {
            lock (_sync)
            {
                Interlocked.Increment(ref _totalProcessed);
                if (completed.Node != null)
                {
                    _waiters.Remove(completed.Node);
                    completed.Node = null;
                }
                _logger.LogDebug(LogEventIds.QueueReleased,
                    "Inference slot released for request {RequestId} (totalProcessed={TotalProcessed}, pending={PendingCount})",
                    completed.RequestId ?? "(none)", _totalProcessed, _waiters.Count);
                ActivateNext();
            }
        }

        internal void RemoveCancelled(QueueTicket ticket)
        {
            lock (_sync)
            {
                if (ticket.Node != null)
                {
                    _waiters.Remove(ticket.Node);
                    ticket.Node = null;
                    UpdatePositions();
                    _logger.LogInformation(LogEventIds.QueueCancelled,
                        "Cancelled queued request {RequestId} (pending={PendingCount})",
                        ticket.RequestId ?? "(none)", _waiters.Count);
                }
            }
        }

        private void ActivateNext()
        {
            while (_waiters.Count > 0)
            {
                var node = _waiters.First;
                _waiters.RemoveFirst();
                var next = node.Value;
                next.Node = null;

                if (next.IsCancelled)
                {
                    _logger.LogDebug(LogEventIds.QueueCancelled,
                        "Skipping cancelled queued request {RequestId}",
                        next.RequestId ?? "(none)");
                    continue;
                }

                _currentRequestId = next.RequestId;
                next.SetPosition(0);
                next.MarkReady();
                UpdatePositions();
                _logger.LogInformation(LogEventIds.QueueReady,
                    "Activated queued request {RequestId} (pending={PendingCount})",
                    next.RequestId ?? "(none)", _waiters.Count);
                return;
            }
            _busy = false;
            _currentRequestId = null;
        }

        private void UpdatePositions()
        {
            int pos = 1;
            foreach (var t in _waiters)
                t.SetPosition(pos++);
        }

        public QueueStatus GetStatus()
        {
            lock (_sync)
            {
                return new QueueStatus
                {
                    Busy = _busy,
                    PendingRequests = _waiters.Count,
                    TotalProcessed = Interlocked.Read(ref _totalProcessed),
                    CurrentRequestId = _currentRequestId
                };
            }
        }
    }

    public class QueueStatus
    {
        public bool Busy { get; set; }
        public int PendingRequests { get; set; }
        public long TotalProcessed { get; set; }
        public string CurrentRequestId { get; set; }
    }

    /// <summary>
    /// Represents a request's place in the inference queue.
    /// Dispose when inference is complete to release the slot.
    /// </summary>
    public class QueueTicket : IDisposable
    {
        private readonly InferenceQueue _queue;
        private readonly CancellationToken _ct;
        private readonly CancellationTokenRegistration _ctReg;
        private readonly TaskCompletionSource _readyTcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _position;
        private bool _disposed;

        public string RequestId { get; }
        public int Position => Volatile.Read(ref _position);
        public bool IsReady => _readyTcs.Task.IsCompletedSuccessfully;
        public bool IsCancelled => _ct.IsCancellationRequested;
        internal LinkedListNode<QueueTicket> Node { get; set; }

        internal QueueTicket(InferenceQueue queue, CancellationToken ct, string requestId)
        {
            _queue = queue;
            _ct = ct;
            RequestId = requestId;
            _ctReg = ct.Register(() =>
            {
                _readyTcs.TrySetCanceled();
                _queue.RemoveCancelled(this);
            });
        }

        internal void MarkReady() => _readyTcs.TrySetResult();
        internal void SetPosition(int pos) => Volatile.Write(ref _position, pos);

        /// <summary>
        /// Wait until this ticket's turn arrives, or until the timeout elapses.
        /// Use in a loop with position checks to send updates to the client.
        /// </summary>
        public async Task WaitAsync(TimeSpan timeout)
        {
            if (IsReady) return;
            try
            {
                using var delayCts = CancellationTokenSource.CreateLinkedTokenSource(_ct);
                var delay = Task.Delay(timeout, delayCts.Token);
                var completed = await Task.WhenAny(_readyTcs.Task, delay);
                if (completed == _readyTcs.Task)
                    delayCts.Cancel();
            }
            catch (OperationCanceledException) { }
        }

        /// <summary>
        /// Block until this ticket's turn arrives. No position updates.
        /// </summary>
        public async Task WaitUntilReadyAsync()
        {
            try { await _readyTcs.Task; }
            catch (OperationCanceledException) { }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                _ctReg.Dispose();
                if (IsReady)
                    _queue.Release(this);
                else
                    _queue.RemoveCancelled(this);
            }
        }
    }
}

