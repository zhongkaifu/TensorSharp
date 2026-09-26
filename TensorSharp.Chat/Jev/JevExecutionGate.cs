// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Server.Jev;

/// <summary>Bounded, cancellable admission shared with model load, unload and disposal.</summary>
internal sealed class JevExecutionGate(int capacity)
{
    private readonly SemaphoreSlim _execution = new(1, 1);
    private readonly object _state = new();
    private int _admitted;
    private bool _stopped;

    internal Task<T> ExecuteAsync<T>(Func<CancellationToken, T> execute, CancellationToken cancellation)
        => ExecuteAwaitedAsync(ct => Task.Run(() => execute(ct), ct), cancellation);

    internal async Task<T> ExecuteAwaitedAsync<T>(Func<CancellationToken, Task<T>> execute, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        lock (_state)
        {
            ObjectDisposedException.ThrowIf(_stopped, this);
            if (_admitted >= capacity) throw new JevQueueFullException();
            _admitted++;
        }
        bool entered = false;
        try
        {
            await _execution.WaitAsync(cancellation).ConfigureAwait(false);
            entered = true;
            lock (_state) ObjectDisposedException.ThrowIf(_stopped, this);
            // Only the admitted owner occupies a worker. Waiting HTTP requests remain asynchronous.
            return await execute(cancellation).ConfigureAwait(false);
        }
        finally
        {
            lock (_state) _admitted--;
            if (entered) _execution.Release();
        }
    }

    internal IDisposable BeginChange(bool shutdown = false)
    {
        // Close admission before waiting for active native work. Queued requests
        // then fail when they acquire the gate instead of prolonging shutdown.
        if (shutdown) lock (_state) _stopped = true;
        _execution.Wait();
        lock (_state)
        {
            if (!shutdown && _stopped)
            {
                _execution.Release();
                throw new ObjectDisposedException(nameof(JevExecutionGate));
            }
        }
        return new Lease(_execution);
    }

    private sealed class Lease(SemaphoreSlim gate) : IDisposable
    {
        private SemaphoreSlim? _gate = gate;
        public void Dispose() => Interlocked.Exchange(ref _gate, null)?.Release();
    }
}
