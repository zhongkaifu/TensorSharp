// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using TensorSharp.Memory;

namespace TensorSharp.Runtime.Scheduling;

/// <summary>Executor-supplied upper bounds for a request's incremental peak on every
/// physical pool, including maximum state growth and scratch. Shared weights, pooled
/// device arenas and retained prefixes need separate lifetime charges. Estimates must
/// describe the selected execution path; the engine never guesses sizes from a model name.
/// Pass SequenceState.MemoryEnvelope to request-owned residency acquisitions to avoid
/// counting reserved peak and live allocations twice.</summary>
public sealed class RequestMemoryAdmission
{
    public RequestMemoryAdmission(MemoryBudget budget,
        Func<SequenceState, IReadOnlyList<MemoryCharge>> estimatePeak, int maxQueuedRequests = 1024)
    {
        Budget = budget ?? throw new ArgumentNullException(nameof(budget));
        EstimatePeak = estimatePeak ?? throw new ArgumentNullException(nameof(estimatePeak));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxQueuedRequests);
        MaxQueuedRequests = maxQueuedRequests;
    }
    public MemoryBudget Budget { get; }
    public Func<SequenceState, IReadOnlyList<MemoryCharge>> EstimatePeak { get; }
    public int MaxQueuedRequests { get; }
}
