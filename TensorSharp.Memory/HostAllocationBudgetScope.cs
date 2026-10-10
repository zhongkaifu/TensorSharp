// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory;

/// <summary>Process-wide ownership for explicitly instrumented anonymous host
/// allocations. Install at a quiescent boundary before constructing the model.
/// Existing allocations, file mappings, GC/driver heaps and allocator metadata
/// are not retroactively charged. A retained pool block keeps its charge until
/// it is physically freed. This is an allocation quota, not an RSS limit.</summary>
public sealed class HostAllocationBudgetScope : IDisposable
{
    private static readonly object Gate = new();
    private static HostAllocationBudgetScope? _active;
    private readonly MemoryBudget _budget;
    private readonly string[] _pools;
    private BudgetReservation? _envelope;
    private long _bytes, _peak;
    private int _allocations;
    private bool _disposed;

    public HostAllocationBudgetScope(MemoryBudget budget, IEnumerable<string> pools)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ArgumentNullException.ThrowIfNull(pools);
        _budget = budget;
        _pools = pools.ToArray();
        if (_pools.Length == 0 || _pools.Distinct(StringComparer.Ordinal).Count() != _pools.Length)
            throw new ArgumentException("Map each host allocation to distinct physical constraints.", nameof(pools));
        budget.CanEverFit(_pools.Select(p => new MemoryCharge(p, 0)));
        lock (Gate)
        {
            if (_active != null) throw new InvalidOperationException("A host allocation budget is already attached.");
            _active = this;
        }
    }

    public (long Bytes, long PeakBytes, int Allocations) Usage
    { get { lock (Gate) return (_bytes, _peak, _allocations); } }

    /// <summary>Reserve before calling the allocator. Null means instrumentation
    /// is disabled; quota refusal throws. Commit only after allocation succeeds,
    /// dispose after a failed allocation or a successful physical free.</summary>
    public static Allocation? Reserve(long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bytes);
        lock (Gate)
        {
            if (_active is not { } scope) return null;
            var charges = scope._pools.Select(p => new MemoryCharge(p, bytes));
            var reservation = scope._envelope == null ? scope._budget.TryReserve(charges) : scope._envelope.TryTake(charges);
            if (reservation == null) throw new MemoryPressureException($"Host allocation of {bytes} bytes exceeds its allocation envelope.");
            scope._allocations++;
            scope._bytes = checked(scope._bytes + bytes);
            scope._peak = Math.Max(scope._peak, scope._bytes);
            return new(scope, reservation, bytes);
        }
    }

    /// <summary>Only for serialized model execution. Native/managed allocations
    /// in this process share the active model lane; never use across an await.</summary>
    public IDisposable EnterExecution(BudgetReservation envelope)
    {
        ArgumentNullException.ThrowIfNull(envelope);
        lock (Gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!ReferenceEquals(envelope.Budget, _budget)) throw new ArgumentException("Request belongs to a different budget.", nameof(envelope));
            if (_envelope != null) throw new InvalidOperationException("Concurrent or nested host execution envelopes are unsupported.");
            _envelope = envelope;
            return new Execution(this);
        }
    }

    public void Dispose()
    {
        lock (Gate)
        {
            if (_disposed) return;
            if (_allocations != 0 || _envelope != null)
                throw new InvalidOperationException("Host allocations or execution still own budget credit. Release them and retry disposal.");
            _active = null;
            _disposed = true;
        }
    }

    private sealed class Execution(HostAllocationBudgetScope owner) : IDisposable
    {
        private bool _disposed;
        public void Dispose() { lock (Gate) { if (_disposed) return; owner._envelope = null; _disposed = true; } }
    }

    public sealed class Allocation : IDisposable
    {
        private readonly HostAllocationBudgetScope _scope;
        private readonly BudgetReservation _reservation;
        private readonly long _bytes;
        private bool _disposed;
        internal Allocation(HostAllocationBudgetScope scope, BudgetReservation reservation, long bytes)
        { _scope = scope; _reservation = reservation; _bytes = bytes; }
        public void Commit() { lock (Gate) { ObjectDisposedException.ThrowIf(_disposed, this); _reservation.Commit(); } }
        public void Dispose()
        {
            lock (Gate)
            {
                if (_disposed) return;
                _reservation.Dispose();
                _scope._bytes -= _bytes;
                _scope._allocations--;
                _disposed = true;
            }
        }
    }
}
