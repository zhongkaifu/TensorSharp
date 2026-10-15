// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp.Memory;

namespace TensorSharp.Runtime.Scheduling;

/// <summary>Cooperative admission and idle reclamation for registered engines sharing
/// a ledger. Conflicting demand ages in FIFO order; independent pools need not wait.
/// Model work always stays on its own worker/lock. External ledger users still compete
/// atomically, but do not participate in this engine admission order.</summary>
internal sealed class SharedBudgetReclamation
{
    private static readonly ConditionalWeakTable<MemoryBudget, SharedBudgetReclamation> Groups = new();
    private readonly object _gate = new();
    private readonly List<Registration> _members = new();
    private readonly LinkedList<Demand> _demands = new();
    private readonly Dictionary<Demand, ReclamationAttempts> _attempts = new(ReferenceEqualityComparer.Instance);
    private ReclamationLease? _reclaimer;

    internal sealed record Demand(Registration Owner, SequenceState Request, MemoryCharge[] Peak);

    private sealed class ReclamationAttempts(Task signal)
    {
        public Task Signal = signal;
        public readonly HashSet<Registration> Donors = new();
    }

    // Called under _gate. Independent blocked demands need independent attempt
    // histories: an unreclaimable older pool must not suppress reclamation for
    // another pool, and switching demands must not reset unsuccessful attempts.
    private ReclamationAttempts AttemptsFor(Demand demand, Task signal)
    {
        if (!_attempts.TryGetValue(demand, out var attempts))
            _attempts.Add(demand, attempts = new ReclamationAttempts(signal));
        else if (!ReferenceEquals(attempts.Signal, signal))
        {
            attempts.Signal = signal;
            attempts.Donors.Clear();
        }
        return attempts;
    }

    internal static Registration Register(MemoryBudget budget, Action wake)
    {
        var group = Groups.GetValue(budget, _ => new());
        var member = new Registration(group, budget, wake);
        lock (group._gate) group._members.Add(member);
        return member;
    }

    private static bool Conflicts(IReadOnlyList<MemoryCharge> a, IReadOnlyList<MemoryCharge> b) =>
        a.Any(x => x.Bytes > 0 && b.Any(y => y.Bytes > 0 && x.Pool == y.Pool));

    private static MemoryCharge[] NormalizeDemandPeak(IReadOnlyList<MemoryCharge> peak)
    {
        ArgumentNullException.ThrowIfNull(peak);
        var totals = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var charge in peak)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(charge.Pool);
            ArgumentOutOfRangeException.ThrowIfNegative(charge.Bytes);
            totals[charge.Pool] = checked(totals.GetValueOrDefault(charge.Pool) + charge.Bytes);
        }
        return totals.Where(pair => pair.Value > 0).OrderBy(pair => pair.Key, StringComparer.Ordinal)
            .Select(pair => new MemoryCharge(pair.Key, pair.Value)).ToArray();
    }

    private bool HasOlderConflict(Registration owner, IReadOnlyList<MemoryCharge> peak)
    {
        foreach (var demand in _demands)
        {
            if (ReferenceEquals(demand.Owner, owner)) break;
            if (Conflicts(demand.Peak, peak)) return true;
        }
        return false;
    }

    private Registration[] Peers(Registration? except = null) =>
        _members.Where(m => !ReferenceEquals(m, except)).ToArray();

    private static void Wake(IEnumerable<Registration> members)
    { foreach (var member in members) member.Wake(); }

    internal sealed class Registration(SharedBudgetReclamation group, MemoryBudget budget, Action wake) : IDisposable
    {
        private LinkedListNode<Demand>? _pending;
        private Task? _lastSignal;
        private bool _disposed;

        /// <summary>Only a blocked head takes a queue place. A successful atomic
        /// reservation transfers ownership to the request and relinquishes that place,
        /// so a busy engine cannot repeatedly leapfrog an older conflicting waiter.</summary>
        public BudgetReservation? TryReserve(SequenceState request, IReadOnlyList<MemoryCharge> peak,
            IReadOnlyList<MemoryCharge>? capacityConstraints = null, bool waitOnly = false)
            => TryReserve(request, peak, out _, capacityConstraints, waitOnly);

        // The generation belongs to the actual constrained ledger attempt,
        // never to a FIFO-only wait. Capture it before TryReserve so an external
        // release between failure and the caller recording it cannot be lost.
        internal BudgetReservation? TryReserve(SequenceState request, IReadOnlyList<MemoryCharge> peak,
            out Task? attemptedSignal, IReadOnlyList<MemoryCharge>? capacityConstraints = null, bool waitOnly = false)
        {
            attemptedSignal = null;
            Registration[] targets = [];
            BudgetReservation? result;
            lock (group._gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                // Validate every original charge, including zero-byte pools,
                // before changing a queue entry. FIFO/wait-only paths bypass
                // TryReserve; this return value is not allocation permission.
                _ = budget.CanEverFitConfiguredCeiling(peak);
                if (_pending != null && !ReferenceEquals(_pending.Value.Request, request)) RemoveDemand();
                if (waitOnly || group.HasOlderConflict(this, peak)) result = null;
                else
                {
                    if (capacityConstraints != null) attemptedSignal = budget.ChangeSignal;
                    result = capacityConstraints == null ? budget.TryReserve(peak)
                        : budget.TryReserve(peak, capacityConstraints);
                }
                if (result != null)
                {
                    if (RemoveDemand()) targets = group.Peers(this);
                }
                else
                {
                    var currentPeak = NormalizeDemandPeak(peak);
                    if (_pending == null)
                    {
                        _pending = group._demands.AddLast(new Demand(this, request, currentPeak));
                        targets = group.Peers(this);
                    }
                    else if (!_pending.Value.Peak.SequenceEqual(currentPeak))
                    {
                        // A physical boundary can re-estimate the same request's
                        // peak. Keep its FIFO position, but retire the old demand
                        // so old leases stop and donors get one fresh attempt.
                        // Equivalent charge ordering/splits/zeros do not wake peers
                        // or reset bounded attempts in the same budget generation.
                        group._attempts.Remove(_pending.Value);
                        _pending.Value = new Demand(this, request, currentPeak);
                        targets = group.Peers(this);
                    }
                }
            }
            SharedBudgetReclamation.Wake(targets);
            return result;
        }

        public bool CanAdmit(IReadOnlyList<MemoryCharge> peak)
        {
            lock (group._gate)
                return !_disposed && !group.HasOlderConflict(this, peak) && budget.CanReserve(peak);
        }

        public bool HasAdmissionTurn(IReadOnlyList<MemoryCharge> peak)
        {
            lock (group._gate) return !_disposed && !group.HasOlderConflict(this, peak);
        }

        public bool CanAttemptPhysicalAdmission(IReadOnlyList<MemoryCharge> peak)
        {
            lock (group._gate)
                return !_disposed && !group.HasOlderConflict(this, peak) && budget.CanReserveConfiguredCeiling(peak);
        }

        /// <summary>Refresh pressure after a budget generation changes. A stable
        /// generation is not a reason for blocked engines to keep waking each other.</summary>
        public void Publish(IReadOnlyList<MemoryCharge> demand, Task signal)
        {
            Registration[] targets;
            lock (group._gate)
            {
                if (_disposed || _pending == null || ReferenceEquals(signal, _lastSignal)) return;
                _lastSignal = signal;
                targets = group.Peers(this);
            }
            SharedBudgetReclamation.Wake(targets);
        }

        internal void Wake() => wake(); // Coalesced channel write, never model work.

        private bool RemoveDemand()
        {
            if (_pending == null) return false;
            group._attempts.Remove(_pending.Value);
            group._demands.Remove(_pending);
            _pending = null;
            _lastSignal = null;
            return true;
        }

        public void Cancel(SequenceState request)
        {
            Registration[] targets = [];
            lock (group._gate)
                if (_pending != null && ReferenceEquals(_pending.Value.Request, request) && RemoveDemand())
                    targets = group.Peers(this);
            SharedBudgetReclamation.Wake(targets);
        }

        /// <summary>Also used when an engine's compute gate closes: an engine that
        /// cannot execute relinquishes its queue position without touching live owners.</summary>
        public void Clear()
        {
            Registration[] targets = [];
            lock (group._gate) if (RemoveDemand()) targets = group.Peers(this);
            SharedBudgetReclamation.Wake(targets);
        }

        /// <summary>One donor at a time, for one live demand. Each donor gets at most
        /// one unsuccessful attempt per demand/budget generation. A lease is only a
        /// coordination right, not a memory reservation and not a model lock.</summary>
        public ReclamationLease? TryBeginReclamation()
        {
            lock (group._gate)
            {
                if (_disposed || group._reclaimer != null) return null;
                var signal = budget.ChangeSignal;
                foreach (var demand in group._demands)
                {
                    if (group.HasOlderConflict(demand.Owner, demand.Peak)
                        || !budget.CanEverFit(demand.Peak) || budget.CanReserve(demand.Peak)) continue;
                    if (!group.AttemptsFor(demand, signal).Donors.Add(this)) continue;
                    return group._reclaimer = new ReclamationLease(group, budget, demand, this);
                }
                return null;
            }
        }

        public void Dispose()
        {
            Registration[] targets;
            lock (group._gate)
            {
                if (_disposed) return;
                _disposed = true; RemoveDemand(); group._members.Remove(this);
                foreach (var attempts in group._attempts.Values) attempts.Donors.Remove(this);
                targets = group.Peers();
            }
            SharedBudgetReclamation.Wake(targets);
        }
    }

    internal sealed class ReclamationLease : IDisposable
    {
        private readonly SharedBudgetReclamation _group;
        private readonly MemoryBudget _budget;
        private readonly Demand _demand;
        private readonly Registration _donor;
        private bool _disposed;
        internal ReclamationLease(SharedBudgetReclamation group, MemoryBudget budget, Demand demand, Registration donor)
        { _group = group; _budget = budget; _demand = demand; _donor = donor; }
        public bool MoreWork { get; set; }

        public bool NeedsReclamation()
        {
            lock (_group._gate)
                return !_disposed && _group._demands.Contains(_demand)
                    && _budget.CanEverFit(_demand.Peak) && !_budget.CanReserve(_demand.Peak);
        }

        public void Dispose()
        {
            Registration[] targets;
            lock (_group._gate)
            {
                if (_disposed) return;
                _disposed = true; _group._reclaimer = null;
                if (_group._demands.Contains(_demand))
                {
                    // Own partial releases are already reflected by this completed
                    // pass. Do not treat their signal as fresh external progress and
                    // immediately trim the same donor again. A full bounded batch is
                    // the one case that deliberately yields and continues later.
                    var signal = _budget.ChangeSignal;
                    var attempts = _group.AttemptsFor(_demand, signal);
                    if (MoreWork) attempts.Donors.Remove(_donor);
                    else if (_group._members.Contains(_donor)) attempts.Donors.Add(_donor);
                }
                targets = _group.Peers();
            }
            SharedBudgetReclamation.Wake(targets);
        }
    }
}
