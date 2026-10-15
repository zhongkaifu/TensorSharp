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
    // Instance-local fault seams; the normal path invokes no delegate.
    internal Action? BeforeAllocationOwnerAllocation { get; set; }
    internal Action? BeforeExecutionOwnerAllocation { get; set; }

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
            // Allocate the final owner and validate all bookkeeping before
            // taking credit. After TryReserve/TryTake succeeds only no-throw
            // assignments remain, so OOM cannot orphan a reservation.
            int nextAllocations = checked(scope._allocations + 1);
            long nextBytes = checked(scope._bytes + bytes);
            scope.BeforeAllocationOwnerAllocation?.Invoke();
            var owner = new Allocation(scope, bytes);
            var charges = scope._pools.Select(p => new MemoryCharge(p, bytes));
            var reservation = scope._envelope == null ? scope._budget.TryReserve(charges) : scope._envelope.TryTake(charges);
            if (reservation == null) throw new MemoryPressureException($"Host allocation of {bytes} bytes exceeds its allocation envelope.");
            owner.Bind(reservation);
            scope._allocations = nextAllocations;
            scope._bytes = nextBytes;
            scope._peak = Math.Max(scope._peak, nextBytes);
            return owner;
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
            BeforeExecutionOwnerAllocation?.Invoke();
            var execution = new Execution(this);
            _envelope = envelope;
            return execution;
        }
    }

    public void Dispose()
    {
        lock (Gate)
        {
            if (_disposed) return;
            if (_allocations != 0 || _envelope != null)
                throw new InvalidOperationException($"Host allocations ({_allocations}, {_bytes} bytes) or execution still own budget credit. Release them and retry disposal.");
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
        private BudgetReservation _reservation = null!;
        private readonly long _bytes;
        private bool _disposed;
        internal Allocation(HostAllocationBudgetScope scope, long bytes)
        { _scope = scope; _bytes = bytes; }
        // Only called before publication, after atomic reservation succeeds.
        internal void Bind(BudgetReservation reservation) => _reservation = reservation;
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
