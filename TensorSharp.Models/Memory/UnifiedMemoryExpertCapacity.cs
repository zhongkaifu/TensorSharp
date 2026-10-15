// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>Fenced capacity changes for a selected-expert cache and its persistent
/// shared-budget owner. Model adapters supply their immutable execution floor;
/// this policy neither estimates geometry nor releases model state. It applies
/// to any single-pool cache, independently of the pool's backend or device name.</summary>
internal static class UnifiedMemoryExpertCapacity
{
    internal static bool Resize(GgmlHostMoeExpertCacheScope cache, BudgetReservation envelope,
        string pool, long target, long immutableMinimum, IReadOnlyList<MemoryCharge> limits)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(envelope);
        ArgumentException.ThrowIfNullOrWhiteSpace(pool);
        ArgumentNullException.ThrowIfNull(limits);
        if (immutableMinimum < 0 || target < immutableMinimum || target > cache.MaximumBudgetBytes)
            throw new ArgumentOutOfRangeException(nameof(target));
        // Expert payload has one persistent owner. Reject a wrong owner before
        // either changing native capacity or reserving another pool's credit.
        var owned = envelope.OwnedCharges;
        if (owned.Count != 1 || owned[0].Pool != pool)
            throw new ArgumentException("Expert capacity needs its single-pool allocation envelope.", nameof(envelope));
        long previous = cache.BudgetBytes;
        if (target == previous) return true;
        MemoryCharge[] Charges(long bytes) => [new(pool, bytes)];
        if (target < previous)
        {
            // UpdateBudget fences and destroys native owners before their
            // children refund credit. Reserved children count just like commits.
            cache.UpdateBudget(target);
            long children = envelope.LiveChildCharges.Single(c => c.Pool == pool).Bytes;
            if (children > target || !envelope.TryResizeTotal(Charges(target)))
                throw new InvalidOperationException("Expert trim did not release its complete shared physical ownership; existing credit remains owned.");
            return true;
        }
        // The total target includes every live child, so it does not count
        // physical payload twice or depend on a separately sampled usage total.
        if (!envelope.TryResizeTotal(Charges(target), limits)) return false;
        try { cache.UpdateBudget(target); }
        catch
        {
            // A failed native update keeps its old ceiling but may have freed
            // entries. Restore only unused credit; retained children stay owned.
            if (!envelope.TryResizeTotal(Charges(previous)))
                throw new InvalidOperationException("Expert capacity could not restore its previous envelope after a failed native update.");
            throw;
        }
        return true;
    }
}
