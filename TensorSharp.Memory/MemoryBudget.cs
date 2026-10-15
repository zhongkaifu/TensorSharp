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
/// <summary>The configured ceiling is changed only by an explicit capacity setter.
/// A physical refresh target may fall below existing owners; effective capacity
/// follows those owners down as they release, without revoking their credit.</summary>
public readonly record struct MemoryPoolCapacityTarget(
    string Pool, long Ceiling, long Desired, long Effective, long OverTarget);

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
        public long Ceiling = capacity;
        public long? RefreshTarget;
        public long Reserved;
        public long Committed;
        public long PeakOwned, PeakCommitted;
    }
    private readonly object _gate = new();
    private readonly Dictionary<string, Pool> _pools = new(StringComparer.Ordinal);
    private TaskCompletionSource? _changed;
    // Instance-local fault seam for deterministic allocation-failure tests. The
    // default path has no factory/delegate allocation and never invokes a hook.
    internal enum ReservationAllocationPoint { Owner, ChildCharges }
    internal Action<ReservationAllocationPoint>? BeforeReservationAllocation { get; set; }
    // Synthetic notification-failure seam, after ownership is already released.
    // It does not model CLR thread-pool recovery or promise a completed old signal.
    internal Action? BeforeChangeNotification { get; set; }
    private static TaskCompletionSource NewSignal() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private BudgetReservation NewReservation(MemoryCharge[] charges)
    {
        BeforeReservationAllocation?.Invoke(ReservationAllocationPoint.Owner);
        return new BudgetReservation(this, charges);
    }

    private static int FindCharge(MemoryCharge[] charges, string pool)
    {
        for (int i = 0; i < charges.Length; i++)
            if (charges[i].Pool == pool) return i;
        return -1;
    }

    /// <summary>Capture BEFORE trying admission, then wait if it fails. The signal
    /// completes when releases/capacity changes could allow progress; no polling or
    /// callbacks under the accounting lock are required.</summary>
    public Task ChangeSignal { get { lock (_gate) return (_changed ??= NewSignal()).Task; } }
    private void Pulse()
    {
        var old = _changed;
        _changed = null;
        if (old != null)
        {
            BeforeChangeNotification?.Invoke();
            old.TrySetResult();
        }
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

    /// <summary>Validate a queued request against configured ceilings, ignoring
    /// a previous physical refresh target. Only use when a fresh physical check
    /// runs later at admission; this neither reserves nor authorizes allocation.</summary>
    public bool CanEverFitConfiguredCeiling(IEnumerable<MemoryCharge> charges)
    {
        var normalized = Normalize(charges);
        lock (_gate) return normalized.All(c => c.Bytes <= _pools[c.Pool].Ceiling);
    }

    /// <summary>Distinguish ordinary live-owner contention from a lower physical
    /// refresh target. This ignores only the refresh target, never live owners.
    /// It is a readiness hint for a fresh constrained admission attempt, not an
    /// allocation permission; TryReserve still enforces the effective ledger.</summary>
    public bool CanReserveConfiguredCeiling(IEnumerable<MemoryCharge> charges)
    {
        var normalized = Normalize(charges);
        lock (_gate) return normalized.All(c => c.Bytes <=
            _pools[c.Pool].Ceiling - _pools[c.Pool].Reserved - _pools[c.Pool].Committed);
    }

    /// <summary>Check whether a request could fit in isolation within the ledger
    /// and per-call absolute constraints. Existing owners are deliberately ignored:
    /// ordinary contention may wait, while an impossible peak must be refused.
    /// This check does not reserve; TryReserve with the same fresh constraints
    /// still checks all current owners atomically before admitting the request.</summary>
    public bool CanEverFit(IEnumerable<MemoryCharge> charges,
        IEnumerable<MemoryCharge> capacityConstraints)
    {
        var normalized = Normalize(charges);
        var limits = NormalizeCapacityConstraints(capacityConstraints);
        lock (_gate) return normalized.All(c => c.Bytes <= Math.Min(_pools[c.Pool].Capacity,
            limits.GetValueOrDefault(c.Pool, long.MaxValue)));
    }

    public BudgetReservation? TryReserve(IEnumerable<MemoryCharge> charges)
        => TryReserveCore(Normalize(charges), null);

    /// <summary>Reserve against both the ledger and fresh per-call absolute
    /// capacity constraints, without changing shared ceilings or refresh targets.
    /// Repeated constraints on a pool use their minimum; charges still add.
    /// Constraints include current reserved and committed owners, so callers
    /// must not add those owners again to a physical planner's desired capacity.
    /// Ordinary contention against configured ceilings returns null, including
    /// when another caller reserved after the observation. If configured space
    /// exists, a physical constraint violation throws MemoryPressureException.
    /// Effective refresh targets still constrain successful reservations. All
    /// checks and reservation are one transaction.
    /// Physical observations remain forecasts, not OS allocation guarantees.</summary>
    public BudgetReservation? TryReserve(IEnumerable<MemoryCharge> charges,
        IEnumerable<MemoryCharge> capacityConstraints)
    {
        var normalized = Normalize(charges);
        return TryReserveCore(normalized, NormalizeCapacityConstraints(capacityConstraints));
    }

    private Dictionary<string, long> NormalizeCapacityConstraints(IEnumerable<MemoryCharge> capacityConstraints)
    {
        ArgumentNullException.ThrowIfNull(capacityConstraints);
        var limits = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var limit in capacityConstraints)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(limit.Pool);
            ArgumentOutOfRangeException.ThrowIfNegative(limit.Bytes);
            if (!_pools.ContainsKey(limit.Pool)) throw new ArgumentException($"Unknown pool: {limit.Pool}", nameof(capacityConstraints));
            limits[limit.Pool] = limits.TryGetValue(limit.Pool, out long previous)
                ? Math.Min(previous, limit.Bytes) : limit.Bytes;
        }
        return limits;
    }

    private BudgetReservation? TryReserveCore(MemoryCharge[] normalized, Dictionary<string, long>? limits)
    {
        lock (_gate)
        {
            // Readiness and sampling happen outside this lock. A competing
            // engine or unrelated owner can consume configured credit before
            // this call; that remains ordinary waitable ledger contention.
            // Do not use effective Capacity here: a lowered physical target
            // must still reach the constraint check and fail explicitly when
            // configured credit is available, rather than wait for a nonexistent
            // release signal from a permanent owner.
            if (normalized.Any(c => c.Bytes > _pools[c.Pool].Ceiling - _pools[c.Pool].Reserved - _pools[c.Pool].Committed))
                return null;
            if (limits != null)
                foreach (var (name, capacity) in limits)
                {
                    var pool = _pools[name];
                    long requested = normalized.FirstOrDefault(c => c.Pool == name).Bytes;
                    long available = capacity - pool.Reserved - pool.Committed;
                    if (requested > available)
                        throw new MemoryPressureException($"Pool '{name}' cannot admit {requested} bytes within the current physical capacity constraint of {capacity} bytes.");
                }
            if (normalized.Any(c => c.Bytes > _pools[c.Pool].Capacity - _pools[c.Pool].Reserved - _pools[c.Pool].Committed))
                return null;
            // Allocate the only object capable of releasing this credit before
            // changing the ledger. OOM must leave no ownerless reservation.
            var reservation = NewReservation(normalized);
            foreach (var c in normalized)
            {
                var pool = _pools[c.Pool];
                pool.Reserved += c.Bytes;
                pool.PeakOwned = Math.Max(pool.PeakOwned, pool.Reserved + pool.Committed);
            }
            return reservation;
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

    /// <summary>Explicitly replace configured ceilings as one transaction. A live owner in any
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
            foreach (var c in updates)
            {
                var pool = _pools[c.Pool];
                pool.Capacity = pool.Ceiling = c.Bytes;
                pool.RefreshTarget = null;
            }
            if (changed) Pulse();
            return true;
        }
    }

    /// <summary>Refresh desired admission capacities atomically, clamped to each
    /// configured ceiling. Existing owners remain valid, even below the target.
    /// Unlike a one-shot max(target, owners), this target persists: each later
    /// physical release lowers the effective capacity until the target is reached.
    /// Recovery can grow back to the configured ceiling, never above it. Use an
    /// explicit successful TrySetCapacities to replace ceilings or clear targets.
    /// A plan is advisory; TryReserve remains the atomic allocation gate.</summary>
    public IReadOnlyList<MemoryPoolCapacityTarget> RefreshCapacities(IEnumerable<MemoryCharge> desiredCapacities)
    {
        ArgumentNullException.ThrowIfNull(desiredCapacities);
        var updates = desiredCapacities.ToArray();
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var c in updates)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(c.Pool);
            ArgumentOutOfRangeException.ThrowIfNegative(c.Bytes);
            if (!_pools.ContainsKey(c.Pool)) throw new ArgumentException($"Unknown pool: {c.Pool}");
            if (!names.Add(c.Pool)) throw new ArgumentException("A refresh must name each pool at most once.", nameof(desiredCapacities));
        }
        lock (_gate)
        {
            bool changed = false;
            foreach (var c in updates)
            {
                var pool = _pools[c.Pool];
                long previous = pool.Capacity;
                pool.RefreshTarget = Math.Min(c.Bytes, pool.Ceiling);
                RecomputeCapacity(pool);
                changed |= previous != pool.Capacity;
            }
            if (changed) Pulse();
            return CapacityTargetsLocked();
        }
    }

    public IReadOnlyList<MemoryPoolCapacityTarget> CapacityTargets()
    { lock (_gate) return CapacityTargetsLocked(); }

    private MemoryPoolCapacityTarget[] CapacityTargetsLocked() => _pools.Select(p =>
        new MemoryPoolCapacityTarget(p.Key, p.Value.Ceiling, p.Value.RefreshTarget ?? p.Value.Ceiling,
            p.Value.Capacity, Math.Max(0, p.Value.Reserved + p.Value.Committed -
                (p.Value.RefreshTarget ?? p.Value.Ceiling)))).ToArray();

    private static void RecomputeCapacity(Pool pool)
    {
        if (pool.RefreshTarget is long target)
            pool.Capacity = Math.Max(target, pool.Reserved + pool.Committed);
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
                    // No capturing predicate allocation after changing a pool:
                    // failure partway through a multi-pool release is not safe.
                    int index = FindCharge(parent.Items, c.Pool);
                    parent.Items[index] = parent.Items[index] with { Bytes = checked(parent.Items[index].Bytes + c.Bytes) };
                    _pools[c.Pool].Reserved += c.Bytes;
                }
                else if (c.Bytes > 0) releasedToPool = true;
                // An immediate live parent receives this credit back, so its
                // own total and every ancestor's ownership stay unchanged.
                // With a closed parent the physical credit goes to the pool;
                // every ancestor must lose this part of its live subtree too.
                for (var ancestor = reservation.Parent; ancestor != null; ancestor = ancestor.Parent)
                {
                    int index = FindCharge(ancestor.Items, c.Pool);
                    ancestor.DescendantBytes[index] -= c.Bytes;
                    if (ReferenceEquals(ancestor, reservation.Parent) && ancestor.State == 0) break;
                }
                RecomputeCapacity(_pools[c.Pool]);
            }
            reservation.State = 2;
            // Returning a page to its live request envelope changes no global
            // availability. Do not wake every engine blocked on this pool.
            if (releasedToPool) Pulse();
        }
    }

    internal IReadOnlyList<MemoryCharge> Charges(BudgetReservation reservation)
    { lock (_gate) return Array.AsReadOnly((MemoryCharge[])reservation.Items.Clone()); }

    internal IReadOnlyList<MemoryCharge> OwnedCharges(BudgetReservation reservation, bool descendantsOnly)
    {
        lock (_gate)
        {
            var result = new MemoryCharge[reservation.Items.Length];
            for (int i = 0; i < result.Length; i++)
                result[i] = reservation.Items[i] with { Bytes = checked(reservation.DescendantBytes[i]
                    + (descendantsOnly || reservation.State == 2 ? 0 : reservation.Items[i].Bytes)) };
            return Array.AsReadOnly(result);
        }
    }

    // Resize only unused credit. Live children keep their physical ownership
    // and return to this same envelope when they are actually freed. Restrict
    // this operation to roots and the original pool set so a child's later
    // refund cannot refer to a missing pool or manufacture parent credit.
    internal bool TryResizeRemaining(BudgetReservation envelope, IEnumerable<MemoryCharge> charges,
        IEnumerable<MemoryCharge>? capacityConstraints)
        => TryResize(envelope, charges, capacityConstraints, totalTarget: false);

    internal bool TryResizeTotal(BudgetReservation envelope, IEnumerable<MemoryCharge> charges,
        IEnumerable<MemoryCharge>? capacityConstraints)
        => TryResize(envelope, charges, capacityConstraints, totalTarget: true);

    /// <summary>Atomically target several open root envelopes' total ownership.
    /// Live descendants remain charged even below a target. Capacity checks use
    /// the combined post-transaction ledger, so a refusal leaves every envelope
    /// unchanged and released credit can fund another owner's growth.</summary>
    public bool TryResizeTotals(IEnumerable<(BudgetReservation Envelope, IReadOnlyList<MemoryCharge> TotalTarget)> targets,
        IEnumerable<MemoryCharge>? capacityConstraints = null)
        => TryResizeTotals(targets, capacityConstraints, requirePhysicalFit: false);

    /// <summary>Admission variant of the envelope transaction. When physical fit
    /// is required, every supplied constraint checks the complete post-transaction
    /// ledger even if ownership only shrinks. The ordinary overload deliberately
    /// permits cleanup under pressure. False leaves every owner unchanged.</summary>
    public bool TryResizeTotals(IEnumerable<(BudgetReservation Envelope, IReadOnlyList<MemoryCharge> TotalTarget)> targets,
        IEnumerable<MemoryCharge>? capacityConstraints, bool requirePhysicalFit)
    {
        if (requirePhysicalFit) ArgumentNullException.ThrowIfNull(capacityConstraints);
        ArgumentNullException.ThrowIfNull(targets);
        var plans = targets.Select(t => (t.Envelope, Requested: Normalize(t.TotalTarget))).ToArray();
        var limits = capacityConstraints == null ? null : NormalizeCapacityConstraints(capacityConstraints);
        if (requirePhysicalFit && limits!.Count == 0)
            throw new ArgumentException("Strict admission requires at least one physical capacity constraint.", nameof(capacityConstraints));
        var unique = new HashSet<BudgetReservation>(ReferenceEqualityComparer.Instance);
        var deltas = new Dictionary<string, long>(StringComparer.Ordinal);
        TaskCompletionSource? releasedSignal = null;
        lock (_gate)
        {
            foreach (var (envelope, requested) in plans)
            {
                ArgumentNullException.ThrowIfNull(envelope);
                if (!ReferenceEquals(envelope.Owner, this))
                    throw new ArgumentException("Reservation belongs to a different budget.", nameof(targets));
                if (!unique.Add(envelope))
                    throw new ArgumentException("Each envelope may have only one target.", nameof(targets));
                if (envelope.State != 0 || envelope.Parent != null)
                    throw new InvalidOperationException("Only an open root allocation envelope can resize unused credit.");
                if (requested.Length != envelope.Items.Length)
                    throw new ArgumentException("Resizing must preserve each envelope's pool set.", nameof(targets));
                for (int i = 0; i < requested.Length; i++)
                {
                    if (requested[i].Pool != envelope.Items[i].Pool)
                        throw new ArgumentException("Resizing must preserve each envelope's pool set.", nameof(targets));
                    requested[i] = requested[i] with { Bytes = Math.Max(0,
                        requested[i].Bytes - envelope.DescendantBytes[i]) };
                    long delta = requested[i].Bytes - envelope.Items[i].Bytes;
                    deltas[requested[i].Pool] = checked(deltas.GetValueOrDefault(requested[i].Pool) + delta);
                }
            }
            bool grows = false, releases = false;
            foreach (var (name, delta) in deltas)
            {
                grows |= delta > 0; releases |= delta < 0;
                var pool = _pools[name];
                if (delta > 0 && (delta > pool.Ceiling - pool.Reserved - pool.Committed
                    || delta > pool.Capacity - pool.Reserved - pool.Committed)) return false;
            }
            if ((grows || requirePhysicalFit) && limits != null)
                foreach (var (name, capacity) in limits)
                {
                    var pool = _pools[name];
                    if ((decimal)pool.Reserved + pool.Committed + deltas.GetValueOrDefault(name) > capacity) return false;
                }
            // Every plan, validation and aggregate delta exists before mutation.
            // The remaining transaction assigns value types and allocates nothing.
            foreach (var (name, delta) in deltas)
            {
                var pool = _pools[name];
                pool.Reserved += delta;
                pool.PeakOwned = Math.Max(pool.PeakOwned, pool.Reserved + pool.Committed);
                RecomputeCapacity(pool);
            }
            foreach (var (envelope, requested) in plans)
                for (int i = 0; i < requested.Length; i++) envelope.Items[i] = requested[i];
            if (releases)
            {
                releasedSignal = _changed;
                _changed = null;
            }
        }
        // Notification sees the completed owner transaction and cannot execute
        // client continuations or the diagnostic hook under the ledger lock.
        if (releasedSignal != null)
        {
            BeforeChangeNotification?.Invoke();
            releasedSignal.TrySetResult();
        }
        return true;
    }

    private bool TryResize(BudgetReservation envelope, IEnumerable<MemoryCharge> charges,
        IEnumerable<MemoryCharge>? capacityConstraints, bool totalTarget)
    {
        if (!ReferenceEquals(envelope.Owner, this)) throw new ArgumentException("Reservation belongs to a different budget.");
        var requested = Normalize(charges);
        var limits = capacityConstraints == null ? null : NormalizeCapacityConstraints(capacityConstraints);
        lock (_gate)
        {
            if (envelope.State != 0 || envelope.Parent != null)
                throw new InvalidOperationException("Only an open root allocation envelope can resize unused credit.");
            if (requested.Length != envelope.Items.Length)
                throw new ArgumentException("Resizing must preserve the envelope's pool set.", nameof(charges));
            // Resolve the target and live descendants in the same accounting
            // transaction. Children include reserved allocations and retained
            // allocations below closed intermediate envelopes, not just commits.
            for (int i = 0; i < requested.Length; i++)
            {
                if (requested[i].Pool != envelope.Items[i].Pool)
                    throw new ArgumentException("Resizing must preserve the envelope's pool set.", nameof(charges));
                if (totalTarget)
                    requested[i] = requested[i] with { Bytes = Math.Max(0,
                        requested[i].Bytes - envelope.DescendantBytes[i]) };
            }
            bool grows = false, releases = false;
            for (int i = 0; i < requested.Length; i++)
            {
                long delta = requested[i].Bytes - envelope.Items[i].Bytes;
                grows |= delta > 0; releases |= delta < 0;
                var pool = _pools[requested[i].Pool];
                if (delta > 0 && (delta > pool.Ceiling - pool.Reserved - pool.Committed
                    || delta > pool.Capacity - pool.Reserved - pool.Committed)) return false;
            }
            // Releasing credit must remain possible under pressure. Growth,
            // including an atomic multi-pool rebalance, respects every supplied
            // fresh absolute constraint using the post-transaction ledger.
            if (grows && limits != null)
                foreach (var (name, capacity) in limits)
                {
                    var pool = _pools[name];
                    int index = FindCharge(requested, name);
                    long delta = index >= 0 ? requested[index].Bytes - envelope.Items[index].Bytes : 0;
                    if ((decimal)pool.Reserved + pool.Committed + delta > capacity) return false;
                }
            for (int i = 0; i < requested.Length; i++)
            {
                var pool = _pools[requested[i].Pool];
                pool.Reserved += requested[i].Bytes - envelope.Items[i].Bytes;
                envelope.Items[i] = requested[i];
                pool.PeakOwned = Math.Max(pool.PeakOwned, pool.Reserved + pool.Committed);
                RecomputeCapacity(pool);
            }
            if (releases) Pulse();
            return true;
        }
    }

    internal BudgetReservation? Take(BudgetReservation envelope, IEnumerable<MemoryCharge> charges)
    {
        if (!ReferenceEquals(envelope.Owner, this)) throw new ArgumentException("Reservation belongs to a different budget.");
        var requested = Normalize(charges);
        lock (_gate)
        {
            if (envelope.State != 0) throw new InvalidOperationException("An allocation envelope must remain reserved and open.");
            int childCount = 0;
            foreach (var c in requested)
            {
                int index = FindCharge(envelope.Items, c.Pool);
                if (c.Bytes > (index >= 0 ? envelope.Items[index].Bytes : 0)) return null;
                if (c.Bytes > 0) childCount++;
            }
            MemoryCharge[] childCharges = [];
            if (childCount > 0)
            {
                BeforeReservationAllocation?.Invoke(ReservationAllocationPoint.ChildCharges);
                childCharges = new MemoryCharge[childCount];
                int next = 0;
                foreach (var c in requested)
                    if (c.Bytes > 0) childCharges[next++] = c;
            }
            var child = NewReservation(childCharges);
            child.Parent = envelope;
            // The child and its final charge array now exist. The remainder
            // mutates only value-type array elements and cannot allocate.
            foreach (var c in childCharges)
            {
                int index = FindCharge(envelope.Items, c.Pool);
                envelope.Items[index] = envelope.Items[index] with { Bytes = envelope.Items[index].Bytes - c.Bytes };
                envelope.DescendantBytes[index] += c.Bytes;
            }
            envelope.HasChildren = true;
            // Accounting is unchanged until the child allocation commits.
            return child;
        }
    }
}

/// <summary>Own until the physical allocation is actually freed, not merely enqueued
/// for freeing. Commit after successful allocation; Dispose rolls back or releases.</summary>
public sealed class BudgetReservation : IDisposable
{
    internal readonly MemoryBudget Owner;
    internal readonly MemoryCharge[] Items;
    // Aggregate charges in live descendants, pending or committed. Allocate
    // before any ledger mutation so child creation remains failure-atomic.
    internal readonly long[] DescendantBytes;
    internal int State;
    internal BudgetReservation? Parent;
    internal bool HasChildren;
    internal BudgetReservation(MemoryBudget owner, MemoryCharge[] items)
    {
        Owner = owner;
        Items = items;
        DescendantBytes = new long[items.Length];
    }
    public IReadOnlyList<MemoryCharge> Charges => Owner.Charges(this);
    /// <summary>Current pending and committed descendant ownership. Pools may
    /// overlap (UMA), so this is not an additive physical byte total.</summary>
    public IReadOnlyList<MemoryCharge> LiveChildCharges => Owner.OwnedCharges(this, descendantsOnly: true);
    /// <summary>Current remaining credit plus live descendants; a closed root
    /// still reports retained children until their physical release.</summary>
    public IReadOnlyList<MemoryCharge> OwnedCharges => Owner.OwnedCharges(this, descendantsOnly: false);
    public MemoryBudget Budget => Owner;
    /// <summary>Draw allocation credit without double counting a request's reserved
    /// peak. Disposing the child returns credit while this envelope stays open.</summary>
    public BudgetReservation? TryTake(IEnumerable<MemoryCharge> charges) => Owner.Take(this, charges);
    /// <summary>Atomically replace only this root envelope's unused reservation,
    /// preserving the pool set and all live child allocations. Bytes are the new
    /// remaining allowance, not a total including committed children. Shrinks
    /// always release credit; growth must fit current ledger and optional fresh
    /// absolute capacity constraints. False leaves the reservation unchanged.</summary>
    public bool TryResizeRemaining(IEnumerable<MemoryCharge> charges,
        IEnumerable<MemoryCharge>? capacityConstraints = null)
        => Owner.TryResizeRemaining(this, charges, capacityConstraints);
    /// <summary>Atomically target this root's total ownership including every
    /// live pending/committed descendant. Remaining credit becomes
    /// max(0, target - live descendants) per pool. A target below live children
    /// releases only unused credit; it cannot revoke physical owners. The same
    /// pool-set, growth and absolute constraint rules as TryResizeRemaining apply.</summary>
    public bool TryResizeTotal(IEnumerable<MemoryCharge> totalTarget,
        IEnumerable<MemoryCharge>? capacityConstraints = null)
        => Owner.TryResizeTotal(this, totalTarget, capacityConstraints);
    public void Commit() => Owner.Commit(this);
    public void Dispose() => Owner.Release(this);
}
