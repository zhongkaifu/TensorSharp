// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;

namespace TensorSharp.Models;

internal readonly record struct UnifiedMemoryCapacitySample(
    IReadOnlyList<MemoryCharge> Needed, IReadOnlyList<MemoryCharge> Limits);

/// <summary>Admission advice only. Physical release and its lifetime charges stay
/// with the model adapter; this policy never refunds or resizes an owner.</summary>
internal static class UnifiedMemoryReclamationPolicy
{
    internal static bool HasPoolDeficit(MemoryBudget budget, string pool,
        UnifiedMemoryCapacitySample sample)
    {
        var owned = budget.Snapshot().Single(p => p.Pool == pool);
        return (decimal)sample.Needed.Single(c => c.Pool == pool).Bytes + owned.Committed + owned.Reserved
            > sample.Limits.Single(c => c.Pool == pool).Bytes;
    }

    internal static UnifiedMemoryCapacitySample DemoteRetainedDeviceState(
        MemoryBudget budget, string devicePool, UnifiedMemoryCapacitySample sample,
        Func<IReadOnlyList<MemoryCharge>> physicalConstraints,
        Func<IReadOnlyList<MemoryCharge>> estimate,
        Func<IReadOnlyList<MemoryCharge>, bool> demoteOne)
    {
        // Demotion releases Device ownership and can retain Host scratch. A
        // Host-only deficit cannot authorize it, even if that scratch fits.
        while (HasPoolDeficit(budget, devicePool, sample))
        {
            bool demoted = demoteOne(sample.Limits);
            // Actual ownership, physical availability and the request shape can
            // all change. A victim count is never substituted for freed bytes.
            var refreshedLimits = physicalConstraints();
            sample = new(estimate(), refreshedLimits);
            if (!demoted) break;
        }
        return sample;
    }
}
