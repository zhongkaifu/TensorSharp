// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;
using TensorSharp.Runtime.Paged;

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

    /// <summary>Optional allocation routing for a qualified serial execution
    /// adapter. Called under the model compute lock around one sequence's work.
    /// It must consume that sequence's envelope, dispose synchronously, and is
    /// only supported with the per-sequence, non-speculative execution path.</summary>
    public Func<SequenceState, IDisposable>? EnterSerialExecution { get; init; }

    /// <summary>Build the snapshot portion of request admission from model page
    /// geometry and currently available shared RAM. Divide optional residency
    /// across the intended concurrency instead of forcing every request to swap
    /// through a single page. Scratch/staging remain engine-owned. Additional
    /// model/request allocations must be supplied by additionalPeak or charged
    /// separately; this factory is not a whole-model memory estimator.</summary>
    /// <param name="executionHeadroomBytes">RAM held aside for execution owners
    /// not yet allocated, such as the adaptive model's workspace forecast.</param>
    public static RequestMemoryAdmission ForKvSnapshots(KvSnapshotOptions snapshots, long blockBytes,
        int blockTokens, int maxRunningRequests, long executionHeadroomBytes = 0,
        Func<SequenceState, IReadOnlyList<MemoryCharge>>? additionalPeak = null, int maxQueuedRequests = 1024)
    {
        ArgumentNullException.ThrowIfNull(snapshots);
        var budget = snapshots.SharedBudget ?? throw new ArgumentException("Snapshot admission requires a shared budget.", nameof(snapshots));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxRunningRequests);
        ArgumentOutOfRangeException.ThrowIfNegative(executionHeadroomBytes);
        snapshots.Validate(blockBytes);
        long page = checked((blockBytes + 63) / 64 * 64);
        long diskPage = checked((blockBytes + 4095) / 4096 * 4096);
        long fixedBytes = checked(page + ((long)snapshots.TransferBytes + 63) / 64 * 64);
        long free = budget.Snapshot().Single(p => p.Pool == snapshots.RamPool).Available;
        long available = (long)Math.Max(0, (decimal)free - fixedBytes - executionHeadroomBytes);
        return new(budget, seq =>
        {
            var extra = additionalPeak?.Invoke(seq) ?? Array.Empty<MemoryCharge>();
            long extraRam = 0;
            foreach (var charge in extra)
            {
                ArgumentOutOfRangeException.ThrowIfNegative(charge.Bytes);
                if (charge.Pool == snapshots.RamPool) extraRam = checked(extraRam + charge.Bytes);
            }
            long residentPages = Math.Max(1, Math.Max(0, available / maxRunningRequests - extraRam) / page);
            long pages = checked((seq.PromptTokens.Count + (long)seq.MaxNewTokens + blockTokens - 1) / blockTokens);
            var peak = new List<MemoryCharge>
            {
                new(snapshots.RamPool, checked(Math.Min(pages, residentPages) * page)),
                // Restored pages may retain their SSD recovery copy. Reserve
                // every logical page when any eviction can be necessary.
                new(snapshots.SsdPool, pages > residentPages ? checked(pages * diskPage) : 0)
            };
            peak.AddRange(extra);
            return peak;
        }, maxQueuedRequests);
    }
}
