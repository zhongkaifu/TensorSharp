// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory;

/// <summary>Model-independent residency directory. Serializes transitions of the SAME
/// resource, not unrelated I/O. Reads share replicas, writes are exclusive, and live
/// execution leases cannot be evicted. Capacity pressure is returned to the request
/// scheduler rather than waiting while holding a partial working set.</summary>
public sealed class TieredMemoryScheduler : IAsyncDisposable
{
    private sealed class Replica(IResourceBuffer buffer, BudgetReservation charge)
    {
        public readonly IResourceBuffer Buffer = buffer;
        public readonly BudgetReservation Charge = charge;
        public int Readers;
        public long LastUse;
        public LinkedListNode<(Entry Entry, MemoryLocation Location, Replica Replica)>? LruNode;
    }
    private sealed class Entry(MemoryResource resource, IResourceSource? source)
    {
        public readonly MemoryResource Resource = resource;
        public readonly IResourceSource? Source = source;
        public readonly Dictionary<MemoryLocation, Replica> Replicas = new();
        // Failed rollback allocations are not valid replicas, but still own physical
        // memory and budget. Keep them reachable for a later Unregister retry.
        public readonly List<Replica> QuarantinedAllocations = new();
        public SsdSpillStore.Snapshot? Spill;
        public long Version;
        public bool Writer;
        public Exception? Fault;
        public TaskCompletionSource? Transition;
        public TaskCompletionSource Changed = Signal();
    }
    private readonly object _gate = new();
    private readonly Dictionary<ResourceKey, Entry> _entries = new();
    private readonly LinkedList<(Entry Entry, MemoryLocation Location, Replica Replica)> _lru = new();
    private readonly Dictionary<MemoryLocation, IMemoryBackend> _backends;
    private readonly MemoryBudget _budget;
    private readonly BoundedTransfers _transfers;
    private readonly SsdSpillStore _spillStore;
    private readonly IMemoryBackend? _host;
    private bool _stopping;
    private long _clock, _loads, _hits, _evictions, _spills;

    public TieredMemoryScheduler(MemoryBudget budget, IEnumerable<IMemoryBackend> backends,
        BoundedTransfers transfers, SsdSpillStore spillStore, MemoryLocation? demotionHost = null)
    {
        _budget = budget ?? throw new ArgumentNullException(nameof(budget));
        _backends = backends.ToDictionary(x => x.Location);
        if (_backends.Count == 0) throw new ArgumentException("At least one backend is required.", nameof(backends));
        _transfers = transfers ?? throw new ArgumentNullException(nameof(transfers));
        _spillStore = spillStore ?? throw new ArgumentNullException(nameof(spillStore));
        if (demotionHost is { } location)
        {
            _host = _backends[location];
            if (location.Tier != MemoryTier.Host) throw new ArgumentException("Demotion destination must be host RAM.");
        }
    }

    private static TaskCompletionSource Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);
    private static void Pulse(Entry entry)
    {
        var changed = entry.Changed;
        entry.Changed = Signal();
        changed.TrySetResult();
    }
    private static void EndTransition(Entry entry)
    {
        var transition = entry.Transition;
        entry.Transition = null;
        transition?.TrySetResult();
        Pulse(entry);
    }

    public void Register(MemoryResource resource, IResourceSource? source = null)
    {
        resource.Validate();
        if (source == null && !resource.Mutable) throw new ArgumentException("Immutable resources require a backing source.");
        if (source != null && source.ByteLength != resource.ByteLength) throw new ArgumentException("Backing source length mismatch.");
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_stopping, this);
            if (!_entries.TryAdd(resource.Key, new Entry(resource, source)))
                throw new ArgumentException($"Resource already registered: {resource.Key}");
        }
    }

    public ValueTask<ResourceLease> AcquireAsync(ResourceKey key, MemoryLocation location,
        ResourceAccess access = ResourceAccess.Read, CancellationToken cancellationToken = default,
        BudgetReservation? allocationEnvelope = null)
        => AcquireCoreAsync(key, location, access, true, cancellationToken, allocationEnvelope);

    private async ValueTask<ResourceLease> AcquireCoreAsync(ResourceKey key, MemoryLocation location,
        ResourceAccess access, bool allowEviction, CancellationToken cancellationToken, BudgetReservation? allocationEnvelope = null,
        bool waitForConflicts = true)
    {
        if (!_backends.TryGetValue(location, out var backend)) throw new ArgumentException($"Unknown location: {location}");
        if (access != ResourceAccess.Read && access != ResourceAccess.Write) throw new ArgumentOutOfRangeException(nameof(access));
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Entry entry;
            Task? wait = null;
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_stopping, this);
                entry = _entries[key];
                if (entry.Fault != null) throw new InvalidOperationException("Resource is quarantined after a backend release failure.", entry.Fault);
                if (access == ResourceAccess.Write && !entry.Resource.Mutable) throw new InvalidOperationException("Immutable resource cannot be written.");
                if (entry.Transition != null) wait = entry.Transition.Task;
                else if (entry.Writer || (access == ResourceAccess.Write && entry.Replicas.Values.Any(r => r.Readers > 0)))
                    wait = entry.Changed.Task;
                else if (access == ResourceAccess.Read && entry.Replicas.TryGetValue(location, out var hit))
                {
                    _hits++;
                    return Lease(entry, location, hit, access);
                }
                else
                {
                    if (access == ResourceAccess.Write && entry.Version == long.MaxValue) throw new OverflowException("Resource version exhausted.");
                    entry.Transition = Signal();
                }
            }
            if (wait != null)
            {
                if (!waitForConflicts) throw new MemoryPressureException("Working set has a conflicting lease or transition; retry the whole set.");
                await wait.WaitAsync(cancellationToken).ConfigureAwait(false);
                continue;
            }

            Replica? provisional = null;
            try
            {
                Replica? target;
                IResourceSource? source;
                lock (_gate)
                {
                    entry.Replicas.TryGetValue(location, out target);
                    source = entry.Replicas.Values.FirstOrDefault()?.Buffer
                        ?? (IResourceSource?)entry.Spill ?? (entry.Version == 0 ? entry.Source : null);
                }
                if (target == null)
                {
                    var reservation = await ReserveWithEvictionAsync(backend.GetAllocationCharges(entry.Resource.ByteLength),
                        allowEviction, cancellationToken, allocationEnvelope).ConfigureAwait(false);
                    provisional = target = await AllocateReplicaAsync(entry, backend, reservation, cancellationToken).ConfigureAwait(false);
                    if (source != null) await _transfers.CopyAsync(source, target.Buffer, cancellationToken).ConfigureAwait(false);
                    else if (entry.Version != 0) throw new InvalidOperationException("Mutable resource has lost its authoritative data.");
                    // Backends initialize new allocations to zero. This is the only
                    // source-less state: a newly registered mutable resource.
                }
                cancellationToken.ThrowIfCancellationRequested();
                if (access == ResourceAccess.Write)
                {
                    // All old readers are gone, and the transition excludes new ones.
                    // No stale replica or old SSD snapshot survives a version change.
                    foreach (var other in entry.Replicas.Where(x => x.Key != location).ToArray())
                    {
                        FreeReplica(entry, other.Value);
                        lock (_gate) entry.Replicas.Remove(other.Key);
                    }
                    entry.Spill?.Dispose();
                    entry.Spill = null;
                }
                lock (_gate)
                {
                    if (provisional != null)
                    {
                        entry.Replicas.Add(location, provisional);
                        provisional.LruNode = _lru.AddLast((entry, location, provisional));
                        provisional = null;
                        _loads++;
                    }
                    if (access == ResourceAccess.Write) entry.Version++;
                    var lease = Lease(entry, location, target, access);
                    EndTransition(entry);
                    return lease;
                }
            }
            catch (Exception error)
            {
                try { if (provisional != null) RollbackAllocation(entry, provisional, error); }
                finally { lock (_gate) EndTransition(entry); }
                throw;
            }
        }
    }

    private ResourceLease Lease(Entry entry, MemoryLocation location, Replica replica, ResourceAccess access)
    {
        // Caller holds _gate.
        replica.LastUse = ++_clock;
        Touch(replica);
        if (access == ResourceAccess.Write) entry.Writer = true;
        else replica.Readers++;
        return new ResourceLease(entry.Resource, location, entry.Version, access, replica.Buffer, () =>
        {
            lock (_gate)
            {
                if (access == ResourceAccess.Write) entry.Writer = false;
                else replica.Readers--;
                replica.LastUse = ++_clock;
                Touch(replica);
                Pulse(entry);
            }
        });
    }

    private void Touch(Replica replica)
    {
        // Caller holds _gate. Track only resident allocations, never the potentially
        // enormous catalog of resources that currently live only on SSD.
        if (replica.LruNode is { } node) { _lru.Remove(node); _lru.AddLast(node); }
    }

    private async ValueTask<BudgetReservation> ReserveWithEvictionAsync(IReadOnlyList<MemoryCharge> charges,
        bool allowEviction, CancellationToken cancellationToken, BudgetReservation? allocationEnvelope)
    {
        charges = _budget.Normalize(charges);
        if (!_budget.CanEverFit(charges)) throw new MemoryPressureException("One resource exceeds a physical pool budget. Partition it at an operator-supported boundary.");
        var attempted = new HashSet<(ResourceKey, MemoryLocation)>();
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var reservation = allocationEnvelope == null ? _budget.TryReserve(charges) : _budget.Take(allocationEnvelope, charges);
            if (reservation != null) return reservation;
            if (!allowEviction) throw new MemoryPressureException("No free budget for speculative prefetch.");
            var available = allocationEnvelope == null
                ? _budget.Snapshot().ToDictionary(x => x.Pool, x => x.Available)
                : allocationEnvelope.Charges.ToDictionary(x => x.Pool, x => x.Bytes);
            var constrained = charges.Where(c => c.Bytes > available.GetValueOrDefault(c.Pool)).Select(c => c.Pool).ToHashSet();
            Entry? victim = null;
            Replica? replica = null;
            MemoryLocation victimLocation = default;
            lock (_gate)
            {
                foreach (var candidateItem in _lru)
                {
                    var candidate = candidateItem.Entry;
                    if (candidate.Transition != null || candidate.Writer || candidate.Fault != null) continue;
                    var item = candidateItem.Replica;
                    if (item.Readers != 0 || attempted.Contains((candidate.Resource.Key, candidateItem.Location))) continue;
                    if (allocationEnvelope != null && !ReferenceEquals(item.Charge.Parent, allocationEnvelope)) continue;
                    if (!item.Charge.Charges.Any(c => c.Bytes > 0 && constrained.Contains(c.Pool))) continue;
                    victim = candidate; replica = item; victimLocation = candidateItem.Location;
                    break;
                }
                if (victim != null) victim.Transition = Signal();
            }
            if (victim == null)
                throw new MemoryPressureException("Budget is occupied by live leases, in-flight transfers, or non-evictable reservations. Defer the request or reduce its working set.");
            attempted.Add((victim.Resource.Key, victimLocation));
            try { await EvictAsync(victim, victimLocation, replica!, cancellationToken).ConfigureAwait(false); }
            catch (MemoryPressureException) { /* A full SSD need not prevent evicting another clean weight. */ }
            finally { lock (_gate) EndTransition(victim); }
        }
    }

    private async ValueTask EvictAsync(Entry entry, MemoryLocation location, Replica replica, CancellationToken cancellationToken)
    {
        // Prefer a RAM replica when it fits WITHOUT evicting anything else. Never
        // recursively acquire another entry while holding this transition.
        if (location.Tier == MemoryTier.Accelerator && _host != null && !entry.Replicas.ContainsKey(_host.Location))
        {
            var charge = _budget.TryReserve(_host.GetAllocationCharges(entry.Resource.ByteLength));
            if (charge != null)
            {
                Replica? provisional = null;
                try
                {
                    provisional = await AllocateReplicaAsync(entry, _host, charge, cancellationToken).ConfigureAwait(false);
                    await _transfers.CopyAsync(replica.Buffer, provisional.Buffer, cancellationToken).ConfigureAwait(false);
                    lock (_gate)
                    {
                        var demoted = provisional;
                        demoted.LastUse = ++_clock;
                        entry.Replicas.Add(_host.Location, demoted);
                        demoted.LruNode = _lru.AddLast((entry, _host.Location, demoted));
                        provisional = null;
                    }
                }
                catch (Exception error)
                {
                    if (provisional != null) RollbackAllocation(entry, provisional, error);
                    throw;
                }
            }
        }
        bool hasRecovery = entry.Replicas.Count > 1 || entry.Spill != null || (entry.Version == 0 && entry.Source != null);
        // Mutable zero-initialized resources also need a snapshot unless another
        // replica exists; this keeps one authoritative recovery path at all times.
        if (!hasRecovery)
        {
            var snapshot = await _spillStore.WriteAsync(replica.Buffer, cancellationToken).ConfigureAwait(false);
            lock (_gate) { entry.Spill = snapshot; _spills++; }
        }
        FreeReplica(entry, replica);
        lock (_gate) { entry.Replicas.Remove(location); _evictions++; }
    }

    private void FreeReplica(Entry entry, Replica replica)
    {
        try { replica.Buffer.Dispose(); }
        catch (Exception ex) { lock (_gate) entry.Fault = ex; throw; }
        replica.Charge.Dispose();
        lock (_gate)
        {
            if (replica.LruNode is { } node) _lru.Remove(node);
            replica.LruNode = null;
        }
    }

    private async ValueTask<Replica> AllocateReplicaAsync(Entry entry, IMemoryBackend backend,
        BudgetReservation reservation, CancellationToken cancellationToken)
    {
        IResourceBuffer? buffer = null;
        try
        {
            buffer = await backend.AllocateAsync(entry.Resource.ByteLength, cancellationToken).ConfigureAwait(false);
            reservation.Commit();
            if (buffer.ByteLength != entry.Resource.ByteLength) throw new InvalidOperationException("Backend allocation length mismatch.");
            return new Replica(buffer, reservation);
        }
        catch (ResourceAllocationException error) when (buffer == null)
        {
            reservation.Commit();
            QuarantineAllocation(entry, new Replica(error.UnreleasedBuffer, reservation), error);
            throw;
        }
        catch (Exception error)
        {
            if (buffer == null) reservation.Dispose();
            else RollbackAllocation(entry, new Replica(buffer, reservation), error);
            throw;
        }
    }

    private void QuarantineAllocation(Entry entry, Replica replica, Exception error)
    {
        lock (_gate)
        {
            entry.QuarantinedAllocations.Add(replica);
            entry.Fault = error;
        }
    }

    private void RollbackAllocation(Entry entry, Replica replica, Exception cause)
    {
        try { FreeReplica(entry, replica); }
        catch (Exception cleanupError)
        {
            var error = new AggregateException("Resource operation and allocation cleanup failed.", cause, cleanupError);
            QuarantineAllocation(entry, replica, error);
            throw error;
        }
    }

    /// <summary>Best-effort prefetch never evicts demand data. It is a performance
    /// hint; demand always retries the exact source on a miss.</summary>
    public async ValueTask<bool> TryPrefetchAsync(ResourceKey key, MemoryLocation location, CancellationToken cancellationToken = default)
    {
        try
        {
            using var lease = await AcquireCoreAsync(key, location, ResourceAccess.Read, false, cancellationToken).ConfigureAwait(false);
            return true;
        }
        catch (MemoryPressureException) { return false; }
    }

    /// <summary>All-or-nothing lease set for one operator/batch. A pressure failure
    /// releases partial leases, allowing the caller to queue, microbatch or stream
    /// smaller partitions without deadlocking on resources it already holds.</summary>
    public ValueTask<ResourceLeaseSet> AcquireReadSetAsync(IEnumerable<ResourceKey> keys, MemoryLocation location,
        CancellationToken cancellationToken = default)
        => AcquireReadSetAsync(keys.Select(key => new ResourcePlacement(key, location)), cancellationToken);

    /// <summary>One execution working set can span multiple devices. Failure releases
    /// every acquired pin; callers retry the whole set without holding half a collective.</summary>
    public async ValueTask<ResourceLeaseSet> AcquireReadSetAsync(IEnumerable<ResourcePlacement> placements,
        CancellationToken cancellationToken = default, BudgetReservation? allocationEnvelope = null)
    {
        ArgumentNullException.ThrowIfNull(placements);
        var leases = new List<ResourceLease>();
        try
        {
            foreach (var placement in placements.Distinct())
                leases.Add(await AcquireCoreAsync(placement.Resource, placement.Location, ResourceAccess.Read, true, cancellationToken,
                    allocationEnvelope,
                    waitForConflicts: false).ConfigureAwait(false));
            return new ResourceLeaseSet(leases);
        }
        catch { foreach (var lease in leases) lease.Dispose(); throw; }
    }

    public IReadOnlyList<ResidencySnapshot> Snapshot()
    {
        lock (_gate) return _entries.Values.SelectMany(e => e.Replicas.Select(r => new ResidencySnapshot(
            e.Resource.Key, r.Key, e.Version, e.Resource.ByteLength, r.Value.Readers, e.Writer, r.Value.LastUse))).ToArray();
    }

    public MemorySchedulerStats GetStats()
    {
        lock (_gate) return new(_loads, _hits, _evictions, _spills, _transfers.BytesCopied, _entries.Count,
            _entries.Values.Sum(e => e.Replicas.Values.Sum(r => r.Readers) + (e.Writer ? 1 : 0)));
    }

    /// <summary>Caller must quiesce execution and stop issuing acquires first.
    /// Failure to quiesce is explicit; no live pointer is silently freed.</summary>
    public void Unregister(ResourceKey key)
    {
        Entry entry;
        lock (_gate)
        {
            entry = _entries[key];
            if (entry.Transition != null || entry.Writer || entry.Replicas.Values.Any(r => r.Readers != 0))
                throw new InvalidOperationException("Resource still has an active lease or transfer.");
            entry.Transition = Signal();
        }
        try
        {
            foreach (var replica in entry.Replicas.ToArray())
            {
                FreeReplica(entry, replica.Value);
                lock (_gate) entry.Replicas.Remove(replica.Key);
            }
            foreach (var replica in entry.QuarantinedAllocations.ToArray())
            {
                FreeReplica(entry, replica);
                lock (_gate) entry.QuarantinedAllocations.Remove(replica);
            }
            entry.Spill?.Dispose();
            lock (_gate) _entries.Remove(key);
        }
        finally { lock (_gate) EndTransition(entry); }
    }

    /// <summary>Does not own the shared staging pool, spill store, or backing sources.</summary>
    public ValueTask DisposeAsync()
    {
        ResourceKey[] keys;
        lock (_gate)
        {
            if (_entries.Values.Any(e => e.Transition != null || e.Writer || e.Replicas.Values.Any(r => r.Readers != 0)))
                throw new InvalidOperationException("Drain executions and transfers before disposing the scheduler.");
            _stopping = true;
            keys = _entries.Keys.ToArray();
        }
        foreach (var key in keys) Unregister(key);
        return ValueTask.CompletedTask;
    }
}

/// <summary>Single-consumer handle. Await all I/O and all native execution before
/// disposing. ReleaseAfterAsync pins residency until an explicit successful fence;
/// a failed fence retains the lease for recovery/quarantine.</summary>
public sealed class ResourceLease : IDisposable, IResourceSource
{
    private readonly IResourceBuffer _buffer;
    private readonly Action _release;
    private int _state; // 0=owned, 1=retiring, 2=released
    internal ResourceLease(MemoryResource resource, MemoryLocation location, long version, ResourceAccess access,
        IResourceBuffer buffer, Action release)
    { Resource = resource; Location = location; Version = version; Access = access; _buffer = buffer; _release = release; }
    public MemoryResource Resource { get; }
    public MemoryLocation Location { get; }
    public long Version { get; }
    public ResourceAccess Access { get; }
    public long ByteLength => Resource.ByteLength;
    private void Check() { if (Volatile.Read(ref _state) != 0) throw new ObjectDisposedException(nameof(ResourceLease)); }
    public nint Pointer { get { Check(); return _buffer.Pointer; } }
    public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
    { Check(); return _buffer.ReadAsync(offset, destination, cancellationToken); }
    public ValueTask WriteAsync(long offset, ReadOnlyMemory<byte> source, CancellationToken cancellationToken = default)
    {
        Check();
        if (Access != ResourceAccess.Write) throw new InvalidOperationException("A write lease is required.");
        return _buffer.WriteAsync(offset, source, cancellationToken);
    }
    public async ValueTask ReleaseAfterAsync(Task executionFence)
    {
        ArgumentNullException.ThrowIfNull(executionFence);
        if (Interlocked.CompareExchange(ref _state, 1, 0) != 0) throw new InvalidOperationException("Lease is already retiring or released.");
        try { await executionFence.ConfigureAwait(false); }
        catch { Volatile.Write(ref _state, 0); throw; }
        Volatile.Write(ref _state, 2);
        _release();
    }
    public void Dispose()
    {
        int prior = Interlocked.CompareExchange(ref _state, 2, 0);
        if (prior == 1) throw new InvalidOperationException("Cannot release a lease before its execution fence completes.");
        if (prior == 0) _release();
    }
}

public sealed class ResourceLeaseSet : IDisposable
{
    internal ResourceLeaseSet(List<ResourceLease> leases) => Leases = leases.AsReadOnly();
    public IReadOnlyList<ResourceLease> Leases { get; }
    /// <summary>The fence must cover every participating rank/stream. A failed fence
    /// leaves all leases owned; callers must synchronize/recover before releasing.</summary>
    public async ValueTask ReleaseAfterAsync(Task executionFence)
    {
        ArgumentNullException.ThrowIfNull(executionFence);
        // Each lease enters its retiring state before waiting, so Dispose cannot
        // free a device's memory while a collective is still in flight.
        await Task.WhenAll(Leases.Select(lease => lease.ReleaseAfterAsync(executionFence).AsTask())).ConfigureAwait(false);
    }
    public void Dispose() { foreach (var lease in Leases) lease.Dispose(); }
}
