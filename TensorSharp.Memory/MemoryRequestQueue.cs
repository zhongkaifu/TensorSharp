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
    internal enum FaultPoint { EnqueueIndex, EnqueueIndexed, AdmissionOwner, AdmissionActive, AdmissionRegistered }
    // Instance-local deterministic failure injection. No default callback.
    internal Action<FaultPoint>? FaultInjector { get; set; }

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
            // Prepare both objects before publication. An index resize must not
            // leave an unindexed linked-list entry that cannot be cancelled.
            var node = new LinkedListNode<Request>(new Request(requestId, charges));
            try
            {
                FaultInjector?.Invoke(FaultPoint.EnqueueIndex);
                _index.Add(requestId, node);
                FaultInjector?.Invoke(FaultPoint.EnqueueIndexed);
            }
            catch
            {
                // Also handles a dictionary failure after it inserted its entry.
                _index.Remove(requestId);
                throw;
            }
            _waiting.AddLast(node); // Existing node: no allocation after indexing.
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

    /// <summary>Rejects and removes a head request that no longer fits after a
    /// capacity reduction, so it cannot permanently block the remaining queue.</summary>
    public AdmittedMemoryRequest? TryAdmit()
    {
        lock (_gate)
        {
            if (_waiting.First is not { } first) return null;
            if (!_budget.CanEverFit(first.Value.Charges))
            {
                _waiting.RemoveFirst();
                _index.Remove(first.Value.Id);
                throw new MemoryPressureException($"Request '{first.Value.Id}' peak exceeds the reduced capacity even in isolation.");
            }
            var reservation = _budget.TryReserve(first.Value.Charges);
            if (reservation == null) return null;
            try
            {
                string requestId = first.Value.Id;
                FaultInjector?.Invoke(FaultPoint.AdmissionOwner);
                var admitted = new AdmittedMemoryRequest(requestId, reservation, () =>
                {
                    lock (_gate) _active.Remove(requestId);
                });
                FaultInjector?.Invoke(FaultPoint.AdmissionActive);
                _active.Add(requestId);
                FaultInjector?.Invoke(FaultPoint.AdmissionRegistered);
                // Owner and active registration exist. Neither removal allocates.
                _waiting.RemoveFirst();
                _index.Remove(requestId);
                return admitted;
            }
            catch (Exception admissionError)
            {
                _active.Remove(first.Value.Id);
                try { reservation.Dispose(); }
                catch (Exception releaseError)
                {
                    throw new AggregateException("Request admission and reservation rollback both failed.",
                        admissionError, releaseError);
                }
                throw;
            }
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
        // 0=retryable, 1=disposing, 2=complete. A notification can fail after the
        // ledger released its owner: still retire the active ID in that case.
        // If actual release or the finish callback failed, a later Dispose can
        // retry without pretending ownership was freed or notifying twice.
        if (Interlocked.CompareExchange(ref _disposed, 1, 0) != 0) return;
        try
        {
            Exception? releaseError = null;
            try { Envelope.Dispose(); }
            catch (Exception error)
            {
                releaseError = error;
                throw;
            }
            finally
            {
                if (Volatile.Read(ref Envelope.State) == 2)
                {
                    try
                    {
                        _finish();
                        Volatile.Write(ref _disposed, 2);
                    }
                    catch (Exception finishError) when (releaseError != null)
                    {
                        throw new AggregateException("Request reservation release and completion both failed.",
                            releaseError, finishError);
                    }
                }
            }
        }
        finally { Interlocked.CompareExchange(ref _disposed, 0, 1); }
    }
}
