// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>Own a synchronous allocation scope and its root reservation. The
/// returned owner is allocated before entering the scope, so no wrapper
/// allocation can strand a successfully published allocation owner. The caller
/// must serialize entry and disposal; the factory must not publish an owner
/// which it fails to return.</summary>
internal static class OwnedAllocationScope
{
    /// <summary>Release only the selected pool's unused reservation after the
    /// caller confirms that its allocation is already bound. Live descendants
    /// remain owned and every other pool keeps its complete current promise.
    /// True means the total target was accepted, including idempotent success;
    /// it does not mean physical memory was freed. The caller must serialize
    /// changes to this open root envelope and separately admit any new owner.</summary>
    internal static bool ReleaseUnusedPoolReservation(BudgetReservation envelope, string pool)
    {
        ArgumentNullException.ThrowIfNull(envelope);
        ArgumentException.ThrowIfNullOrWhiteSpace(pool);
        var owned = envelope.OwnedCharges;
        var children = envelope.LiveChildCharges;
        int selected = -1;
        for (int i = 0; i < owned.Count; i++)
            if (owned[i].Pool == pool) { selected = i; break; }
        if (selected < 0) throw new ArgumentException("The allocation envelope does not own the selected pool.", nameof(pool));
        long childBytes = 0;
        for (int i = 0; i < children.Count; i++)
            if (children[i].Pool == pool) { childBytes = children[i].Bytes; break; }
        var target = new MemoryCharge[owned.Count];
        for (int i = 0; i < owned.Count; i++) target[i] = owned[i];
        target[selected] = new(pool, childBytes);
        // Metadata exists before the budget's atomic transaction. The budget
        // rechecks the actual descendant floor and rejects closed/non-root
        // owners before mutation. No hardware or future refund is assumed.
        return envelope.TryResizeTotal(target);
    }

    /// <summary>Optional state may miss under ledger or freshly observed
    /// physical pressure. Programming/accounting errors must remain visible.</summary>
    internal static BudgetReservation? TryReserveOptional(MemoryBudget budget,
        IReadOnlyList<MemoryCharge> charges, IReadOnlyList<MemoryCharge> limits)
    {
        ArgumentNullException.ThrowIfNull(budget);
        try { return budget.TryReserve(charges, limits); }
        catch (MemoryPressureException) { return null; }
    }

    internal static IDisposable Enter(IDisposable root, Func<IDisposable> enterScope)
    {
        ArgumentNullException.ThrowIfNull(root);
        Owner? owner = null;
        try
        {
            ArgumentNullException.ThrowIfNull(enterScope);
            owner = new Owner(root);
            owner.Scope = enterScope()
                ?? throw new InvalidOperationException("The allocation scope factory returned no owner.");
            return owner;
        }
        catch
        {
            if (owner == null) root.Dispose();
            else owner.Dispose();
            throw;
        }
    }

    private sealed class Owner(IDisposable root) : IDisposable
    {
        internal IDisposable? Scope;
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed) return;
            // A rejected restoration, such as out-of-order nested disposal,
            // must keep the reservation open and this owner retryable.
            Scope?.Dispose();
            Scope = null;
            root.Dispose();
            _disposed = true;
        }
    }
}
