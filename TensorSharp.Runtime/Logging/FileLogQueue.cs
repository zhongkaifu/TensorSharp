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
using System.Threading;

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// Bounded MPSC handoff queue between <see cref="FileLogger"/> producers and the
    /// background writer thread inside <see cref="FileLoggerProvider"/>. Drops the
    /// oldest entry when the buffer is full so that a misbehaving sink can never
    /// pin unbounded memory in a long-running server.
    /// </summary>
    internal sealed class FileLogQueue
    {
        private readonly ConcurrentQueue<LogEntry> _entries = new();
        private readonly SemaphoreSlim _signal = new(0, int.MaxValue);
        private readonly int _maxQueued;
        private long _enqueued;
        private long _dropped;

        public FileLogQueue(int maxQueued)
        {
            if (maxQueued <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxQueued));
            _maxQueued = maxQueued;
        }

        public long Enqueued => Interlocked.Read(ref _enqueued);
        public long Dropped => Interlocked.Read(ref _dropped);
        public int Pending => _entries.Count;

        public void Enqueue(LogEntry entry)
        {
            if (entry == null)
                return;

            // Trim from the head while we're over the bound. We don't try to be
            // exact under contention - dropping is best-effort by design.
            while (_entries.Count >= _maxQueued && _entries.TryDequeue(out _))
            {
                Interlocked.Increment(ref _dropped);
                try { _signal.Wait(0); }
                catch (ObjectDisposedException) { return; }
            }

            _entries.Enqueue(entry);
            Interlocked.Increment(ref _enqueued);

            try { _signal.Release(); }
            catch (ObjectDisposedException) { /* shutting down */ }
        }

        /// <summary>
        /// Block until at least one entry is queued or <paramref name="ct"/> fires,
        /// then return without dequeueing. The caller should then drain entries with
        /// <see cref="TryDequeue"/> until it returns false.
        /// </summary>
        public bool WaitForEntry(TimeSpan timeout, CancellationToken ct)
        {
            try
            {
                return _signal.Wait(timeout, ct);
            }
            catch (OperationCanceledException)
            {
                return false;
            }
            catch (ObjectDisposedException)
            {
                return false;
            }
        }

        public bool TryDequeue(out LogEntry entry) => _entries.TryDequeue(out entry);

        public void SignalShutdown()
        {
            try { _signal.Release(); }
            catch (ObjectDisposedException) { /* already disposed */ }
        }

        public void DisposeSignal() => _signal.Dispose();
    }
}
