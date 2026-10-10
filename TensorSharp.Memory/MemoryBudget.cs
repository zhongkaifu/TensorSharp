// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory;

/// <summary>A physical capacity constraint, e.g. node0/ram, node0/gpu0, or node0/ssd.
/// UMA allocations charge both the physical RAM pool and the GPU working-set constraint;
/// these are constraints on the same allocation, not two copies to sum.</summary>
public readonly record struct MemoryCharge(string Pool, long Bytes);
public readonly record struct MemoryPoolSnapshot(string Pool, long Capacity, long Reserved, long Committed)
{
    public long Available => Capacity - Reserved - Committed;
}
public readonly record struct MemoryPoolHighWatermark(string Pool, long Owned, long Committed);

public sealed class MemoryPressureException : InvalidOperationException
{
    public MemoryPressureException(string message) : base(message) { }
}

/// <summary>Atomic multi-pool reservations. Every allocation, including transfers and
/// spill files, reserves BEFORE allocating. Shared by all schedulers in one process.
/// Capacity is an explicit budget after OS/driver/external-allocation headroom.</summary>
public sealed class MemoryBudget
{
    private sealed class Pool(long capacity)
    {
        public long Capacity = capacity;
        public long Reserved;
        public long Committed;
        public long PeakOwned, PeakCommitted;
    }
    private readonly object _gate = new();
    private readonly Dictionary<string, Pool> _pools = new(StringComparer.Ordinal);
    private TaskCompletionSource? _changed;
    private static TaskCompletionSource NewSignal() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    /// <summary>Capture BEFORE trying admission, then wait if it fails. The signal
    /// completes when releases/capacity changes could allow progress; no polling or
    /// callbacks under the accounting lock are required.</summary>
    public Task ChangeSignal { get { lock (_gate) return (_changed ??= NewSignal()).Task; } }
    private void Pulse()
    {
        var old = _changed;
        _changed = null;
        old?.TrySetResult();
    }

    public MemoryBudget(IEnumerable<MemoryCharge> capacities)
    {
        ArgumentNullException.ThrowIfNull(capacities);
        foreach (var c in capacities)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(c.Pool);
            ArgumentOutOfRangeException.ThrowIfNegative(c.Bytes);
            if (!_pools.TryAdd(c.Pool, new Pool(c.Bytes)))
                throw new ArgumentException($"Duplicate pool: {c.Pool}", nameof(capacities));
        }
        if (_pools.Count == 0) throw new ArgumentException("At least one pool is required.", nameof(capacities));
    }

    internal MemoryCharge[] Normalize(IEnumerable<MemoryCharge> charges)
    {
        ArgumentNullException.ThrowIfNull(charges);
        var result = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var c in charges)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(c.Pool);
            ArgumentOutOfRangeException.ThrowIfNegative(c.Bytes);
            if (!_pools.ContainsKey(c.Pool)) throw new ArgumentException($"Unknown pool: {c.Pool}");
            result[c.Pool] = checked(result.GetValueOrDefault(c.Pool) + c.Bytes);
        }
        return result.OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(x => new MemoryCharge(x.Key, x.Value)).ToArray();
    }

    public bool CanEverFit(IEnumerable<MemoryCharge> charges)
    {
        var normalized = Normalize(charges);
        lock (_gate) return normalized.All(c => c.Bytes <= _pools[c.Pool].Capacity);
    }

    public BudgetReservation? TryReserve(IEnumerable<MemoryCharge> charges)
    {
        var normalized = Normalize(charges);
        lock (_gate)
        {
            if (normalized.Any(c => c.Bytes > _pools[c.Pool].Capacity - _pools[c.Pool].Reserved - _pools[c.Pool].Committed))
                return null;
            foreach (var c in normalized)
            {
                var pool = _pools[c.Pool];
                pool.Reserved += c.Bytes;
                pool.PeakOwned = Math.Max(pool.PeakOwned, pool.Reserved + pool.Committed);
            }
            return new BudgetReservation(this, normalized);
        }
    }

    public BudgetReservation Reserve(IEnumerable<MemoryCharge> charges) => TryReserve(charges)
        ?? throw new MemoryPressureException("The requested allocation does not fit the current multi-pool budget.");

    public IReadOnlyList<MemoryPoolSnapshot> Snapshot()
    {
        lock (_gate) return _pools.Select(x => new MemoryPoolSnapshot(x.Key, x.Value.Capacity,
            x.Value.Reserved, x.Value.Committed)).ToArray();
    }

    /// <summary>Lowering a budget never silently invalidates existing leases.</summary>
    public bool TrySetCapacity(string pool, long capacity)
        => TrySetCapacities(new[] { new MemoryCharge(pool, capacity) });

    /// <summary>Apply a hardware refresh as one transaction. A live owner in any
    /// pool prevents every change; unchanged capacities do not wake waiters.</summary>
    public bool TrySetCapacities(IEnumerable<MemoryCharge> capacities)
    {
        ArgumentNullException.ThrowIfNull(capacities);
        var updates = capacities.ToArray();
        foreach (var c in updates)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(c.Pool);
            ArgumentOutOfRangeException.ThrowIfNegative(c.Bytes);
            if (!_pools.ContainsKey(c.Pool)) throw new ArgumentException($"Unknown pool: {c.Pool}");
        }
        if (updates.Select(c => c.Pool).Distinct(StringComparer.Ordinal).Count() != updates.Length)
            throw new ArgumentException("A capacity update must name each pool at most once.", nameof(capacities));
        lock (_gate)
        {
            if (updates.Any(c => c.Bytes < _pools[c.Pool].Reserved + _pools[c.Pool].Committed)) return false;
            bool changed = updates.Any(c => c.Bytes != _pools[c.Pool].Capacity);
            foreach (var c in updates) _pools[c.Pool].Capacity = c.Bytes;
            if (changed) Pulse();
            return true;
        }
    }

    /// <summary>Exact ledger high-water marks since construction, including
    /// reservations that outlive a sampler interval. Committed is instrumented
    /// payload, Owned includes unused request credit; neither is process RSS.</summary>
    public IReadOnlyList<MemoryPoolHighWatermark> HighWatermarks()
    {
        lock (_gate) return _pools.Select(p => new MemoryPoolHighWatermark(p.Key,
            p.Value.PeakOwned, p.Value.PeakCommitted)).ToArray();
    }

    /// <summary>Check current headroom without owning it. Intended for bounded cache
    /// reclamation; admission must still use TryReserve because other owners can race.</summary>
    public bool CanReserve(IEnumerable<MemoryCharge> charges)
    {
        var normalized = Normalize(charges);
        lock (_gate) return normalized.All(c => c.Bytes <=
            _pools[c.Pool].Capacity - _pools[c.Pool].Reserved - _pools[c.Pool].Committed);
    }

    // A spilled/demoted child must use its request's reserved credit while that
    // request is alive. Retained prefixes outlive the envelope and then compete
    // directly for free pool capacity. Check and reserve under the same lock.
    internal BudgetReservation? TryReserveFollowing(BudgetReservation origin, IEnumerable<MemoryCharge> charges)
    {
        if (!ReferenceEquals(origin.Owner, this)) throw new ArgumentException("Allocation belongs to a different budget.");
        lock (_gate)
            return origin.Parent is { State: 0 } parent ? Take(parent, charges) : TryReserve(charges);
    }

    internal void Commit(BudgetReservation reservation)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(reservation.State == 2, reservation);
            if (reservation.State == 1) return;
            if (reservation.HasChildren) throw new InvalidOperationException("An envelope with child allocations cannot be committed as one allocation.");
            foreach (var c in reservation.Items)
            {
                _pools[c.Pool].Reserved -= c.Bytes;
                _pools[c.Pool].Committed += c.Bytes;
                _pools[c.Pool].PeakCommitted = Math.Max(_pools[c.Pool].PeakCommitted, _pools[c.Pool].Committed);
            }
            reservation.State = 1;
        }
    }

    internal void Release(BudgetReservation reservation)
    {
        lock (_gate)
        {
            if (reservation.State == 2) return;
            bool releasedToPool = false;
            foreach (var c in reservation.Items)
            {
                if (reservation.State == 0) _pools[c.Pool].Reserved -= c.Bytes;
                else _pools[c.Pool].Committed -= c.Bytes;
                // Allocations drawn from a live request envelope return their
                // credit to that request. Closing an envelope releases only its
                // unused credit; its live children remain physically charged.
                if (reservation.Parent is { State: 0 } parent)
                {
                    int index = Array.FindIndex(parent.Items, x => x.Pool == c.Pool);
                    parent.Items[index] = parent.Items[index] with { Bytes = checked(parent.Items[index].Bytes + c.Bytes) };
                    _pools[c.Pool].Reserved += c.Bytes;
                }
                else if (c.Bytes > 0) releasedToPool = true;
            }
            reservation.State = 2;
            // Returning a page to its live request envelope changes no global
            // availability. Do not wake every engine blocked on this pool.
            if (releasedToPool) Pulse();
        }
    }

    internal IReadOnlyList<MemoryCharge> Charges(BudgetReservation reservation)
    { lock (_gate) return Array.AsReadOnly((MemoryCharge[])reservation.Items.Clone()); }

    internal BudgetReservation? Take(BudgetReservation envelope, IEnumerable<MemoryCharge> charges)
    {
        if (!ReferenceEquals(envelope.Owner, this)) throw new ArgumentException("Reservation belongs to a different budget.");
        var requested = Normalize(charges);
        lock (_gate)
        {
            if (envelope.State != 0) throw new InvalidOperationException("An allocation envelope must remain reserved and open.");
            if (requested.Any(c => c.Bytes > envelope.Items.FirstOrDefault(x => x.Pool == c.Pool).Bytes)) return null;
            foreach (var c in requested)
            {
                int index = Array.FindIndex(envelope.Items, x => x.Pool == c.Pool);
                if (index >= 0) envelope.Items[index] = envelope.Items[index] with { Bytes = envelope.Items[index].Bytes - c.Bytes };
            }
            envelope.HasChildren = true;
            // Accounting is unchanged until the child allocation commits.
            return new BudgetReservation(this, requested.Where(c => c.Bytes > 0).ToArray()) { Parent = envelope };
        }
    }
}

/// <summary>Own until the physical allocation is actually freed, not merely enqueued
/// for freeing. Commit after successful allocation; Dispose rolls back or releases.</summary>
public sealed class BudgetReservation : IDisposable
{
    internal readonly MemoryBudget Owner;
    internal readonly MemoryCharge[] Items;
    internal int State;
    internal BudgetReservation? Parent;
    internal bool HasChildren;
    internal BudgetReservation(MemoryBudget owner, MemoryCharge[] items)
    {
        Owner = owner;
        Items = items;
    }
    public IReadOnlyList<MemoryCharge> Charges => Owner.Charges(this);
    public MemoryBudget Budget => Owner;
    /// <summary>Draw allocation credit without double counting a request's reserved
    /// peak. Disposing the child returns credit while this envelope stays open.</summary>
    public BudgetReservation? TryTake(IEnumerable<MemoryCharge> charges) => Owner.Take(this, charges);
    public void Commit() => Owner.Commit(this);
    public void Dispose() => Owner.Release(this);
}
