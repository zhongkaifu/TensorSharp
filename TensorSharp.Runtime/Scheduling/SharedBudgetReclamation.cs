// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Memory;

namespace TensorSharp.Runtime.Scheduling;

/// <summary>Cooperative pressure notifications between engines sharing a ledger.
/// Notifications only enqueue work; each engine reclaims its own idle state on its
/// worker under its model lock. No engine ever takes another model's lock.</summary>
internal sealed class SharedBudgetReclamation
{
    private static readonly ConditionalWeakTable<MemoryBudget, SharedBudgetReclamation> Groups = new();
    private readonly object _gate = new();
    private readonly List<Registration> _members = new();

    internal static Registration Register(MemoryBudget budget, Action wake)
    {
        var group = Groups.GetValue(budget, _ => new());
        var member = new Registration(group, budget, wake);
        lock (group._gate) group._members.Add(member);
        return member;
    }

    internal sealed class Registration(SharedBudgetReclamation group, MemoryBudget budget, Action wake) : IDisposable
    {
        private IReadOnlyList<MemoryCharge>? _demand;
        private Task? _lastSignal;
        private bool _disposed;

        /// <summary>One notification per budget generation prevents two blocked
        /// engines from repeatedly waking each other without freeing anything.</summary>
        public void Publish(IReadOnlyList<MemoryCharge> demand, Task signal)
        {
            Registration[] targets;
            lock (group._gate)
            {
                if (_disposed) return;
                _demand = demand;
                if (ReferenceEquals(signal, _lastSignal)) return;
                _lastSignal = signal;
                targets = group._members.Where(m => !ReferenceEquals(m, this)).ToArray();
            }
            foreach (var target in targets) target.Wake();
        }

        private void Wake() => wake(); // Only a coalesced channel write, never model work.

        public void Clear()
        {
            if (Volatile.Read(ref _demand) == null) return;
            lock (group._gate) { _demand = null; _lastSignal = null; }
        }

        public bool NeedsReclamation()
        {
            IReadOnlyList<MemoryCharge>[] demands;
            lock (group._gate)
                demands = group._members.Where(m => !ReferenceEquals(m, this) && m._demand != null)
                    .Select(m => m._demand!).ToArray();
            // Reservation still decides admission atomically. These observations
            // are only a stop condition for optional idle eviction.
            return demands.Any(d => budget.CanEverFit(d) && !budget.CanReserve(d));
        }

        public void Dispose()
        {
            lock (group._gate)
            {
                _disposed = true;
                _demand = null;
                group._members.Remove(this);
            }
        }
    }
}
