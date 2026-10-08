// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory;

/// <summary>Bounded FIFO admission by the entire request's conservative physical
/// peak, including KV growth and scratch. The model adapter supplies costs; this class
/// never infers KV or recurrent-state sizes from model names. An impossible request
/// fails at enqueue, temporary pressure leaves it queued. Explicit cancellation
/// removes waiters. The engine calls TryAdmit after completions or memory changes.</summary>
public sealed class MemoryRequestQueue
{
    private sealed record Request(string Id, MemoryCharge[] Charges);
    private readonly object _gate = new();
    private readonly MemoryBudget _budget;
    private readonly LinkedList<Request> _waiting = new();
    private readonly Dictionary<string, LinkedListNode<Request>> _index = new(StringComparer.Ordinal);
    private readonly HashSet<string> _active = new(StringComparer.Ordinal);
    private readonly int _maxQueued;

    public MemoryRequestQueue(MemoryBudget budget, int maxQueuedRequests = 1024)
    {
        _budget = budget ?? throw new ArgumentNullException(nameof(budget));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxQueuedRequests);
        _maxQueued = maxQueuedRequests;
    }
    public int WaitingCount { get { lock (_gate) return _waiting.Count; } }
    public int ActiveCount { get { lock (_gate) return _active.Count; } }

    public void Enqueue(string requestId, IEnumerable<MemoryCharge> peakCharges)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(requestId);
        var charges = _budget.Normalize(peakCharges);
        if (!_budget.CanEverFit(charges)) throw new MemoryPressureException("Request peak exceeds the configured capacity even in isolation.");
        lock (_gate)
        {
            if (_index.ContainsKey(requestId) || _active.Contains(requestId)) throw new ArgumentException("Duplicate request id.");
            if (_waiting.Count == _maxQueued) throw new MemoryPressureException("Admission queue is full.");
            _index.Add(requestId, _waiting.AddLast(new Request(requestId, charges)));
        }
    }

    public bool CancelWaiting(string requestId)
    {
        lock (_gate)
        {
            if (!_index.Remove(requestId, out var node)) return false;
            _waiting.Remove(node);
            return true;
        }
    }

    public AdmittedMemoryRequest? TryAdmit()
    {
        lock (_gate)
        {
            if (_waiting.First is not { } first) return null;
            var reservation = _budget.TryReserve(first.Value.Charges);
            if (reservation == null) return null;
            _waiting.RemoveFirst();
            _index.Remove(first.Value.Id);
            _active.Add(first.Value.Id);
            return new AdmittedMemoryRequest(first.Value.Id, reservation, () =>
            {
                lock (_gate) _active.Remove(first.Value.Id);
            });
        }
    }
}

/// <summary>Pass Envelope to acquisitions for request-owned resources. Shared model
/// weights have a separate lifetime/budget. Unregister request state AFTER execution
/// fences, then dispose; any surviving cached allocation remains charged globally.</summary>
public sealed class AdmittedMemoryRequest : IDisposable
{
    private readonly Action _finish;
    private int _disposed;
    internal AdmittedMemoryRequest(string requestId, BudgetReservation envelope, Action finish)
    { RequestId = requestId; Envelope = envelope; _finish = finish; }
    public string RequestId { get; }
    public BudgetReservation Envelope { get; }
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Envelope.Dispose();
        _finish();
    }
}
