// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling;

/// <summary>The execution shape used to derive an adaptive request peak. Bound
/// to one snapshot configuration so a smaller estimate cannot be reused with a
/// larger block, concurrency or workspace. Single-request, prefix-free execution
/// never swaps owners and therefore needs no request-owned snapshot pages.</summary>
public sealed record RequestMemoryExecutionShape(KvSnapshotOptions Snapshots, int BlockTokens,
    int MaximumRunningRequests, int MaximumPrefillTokens);

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

    /// <summary>Optional first phase for a separately owned hybrid holder:
    /// reserve its complete state/copy/export peak before prefix materialization,
    /// without assuming a hit or reserving wide prefill computation. EstimatePeak
    /// remains the no-hit execution ceiling check on submission. The required
    /// AfterPrefixMaterialization boundary must physically admit and resize to
    /// the actual remaining execution peak before any Forward is authorized.</summary>
    public Func<SequenceState, IReadOnlyList<MemoryCharge>>? EstimateMaterializationPeak { get; init; }

    /// <summary>Optional fresh physical-capacity check once for a new eligible
    /// queue head, and again immediately before its reservation if it had to wait.
    /// Returns absolute per-pool capacity ceilings, not additional charges.
    /// Reservation atomically checks these ceilings without changing the shared
    /// budget. Physical refusal fails this request rather than waiting for a signal
    /// that an external physical-memory release need not provide.
    /// The engine invokes this under its model compute lock with the compute gate
    /// open; never on submission or for an already admitted token step.</summary>
    public Func<SequenceState, IReadOnlyList<MemoryCharge>>? BeforeAdmission { get; init; }

    /// <summary>Alternative physical admission boundary with the current live
    /// requests. A shared multi-request arena must be reserved here, before
    /// admitting another request, rather than at its first joint execution.</summary>
    public Func<SequenceState, IReadOnlyList<SequenceState>, IReadOnlyList<MemoryCharge>>? BeforeAdmissionWithRunning { get; init; }
    internal bool HasPhysicalCheck => BeforeAdmission != null || BeforeAdmissionWithRunning != null;

    /// <summary>Optional allocation routing for a qualified serial execution
    /// adapter. Called under the model compute lock around one sequence's work.
    /// It must consume that sequence's envelope, dispose synchronously, and is
    /// only supported with the per-sequence, non-speculative execution path.</summary>
    public Func<SequenceState, IDisposable>? EnterSerialExecution { get; init; }

    /// <summary>Explicit shared-owner scope for one fused multi-request call.
    /// Every participating holder must already be prepared under its own
    /// request scope. This owner outlives any one participant's cancellation.</summary>
    public Func<IReadOnlyList<SequenceState>, IDisposable>? EnterBatchedExecution { get; init; }

    /// <summary>Quiescent transition after the last successful prefill chunk.
    /// Release completed prefill workspace before shrinking unused request
    /// credit to its decode peak. Live allocation credit must remain owned.</summary>
    public Action<SequenceState>? AfterPrefill { get; init; }

    /// <summary>Quiescent admission boundary after prefix lookup/adoption has
    /// completed and its allocation scope has closed, before any Forward.
    /// The adapter can shrink unused peak credit to the remaining prompt work
    /// and recover optional cache capacity. Committed clone/state credit stays
    /// owned. Failure is isolated to this newly admitted destination.</summary>
    public Action<SequenceState>? AfterPrefixMaterialization { get; init; }

    /// <summary>Optional admitted per-request prefill width. This getter must be
    /// side-effect free and return a positive bound without hardware probes or IO.
    /// Scheduling and every actual replay/prefill slice preserve this bound.</summary>
    public Func<SequenceState, int>? PrefillTokenLimit { get; init; }

    /// <summary>Revalidate a revoked prefix's actual execution peak before its
    /// replay or next prefill Forward. Runs inside the existing request scope;
    /// it must not enter another execution scope. A refusal fails only that request.</summary>
    public Action<SequenceState>? RevalidateExecution { get; init; }

    internal int GetPrefillTokenLimit(SequenceState sequence)
    {
        int limit = PrefillTokenLimit?.Invoke(sequence) ?? int.MaxValue;
        if (limit <= 0) throw new InvalidOperationException("An admitted prefill limit must be positive.");
        return limit;
    }

    /// <summary>Optional cache-capacity recovery after model release, native
    /// fences, prefix maintenance and disposal of a finished request's root
    /// reservation. Called under the model compute lock with the gate open,
    /// once per successful release, never before its physical owners settle.
    /// Retained children remain charged. This is optional maintenance: a
    /// failure does not reinterpret an already successful model release.</summary>
    public Action<IReadOnlyList<SequenceState>>? AfterSequenceMemoryRelease { get; init; }

    /// <summary>Keep each admitted request in an independent model-owned holder.
    /// The executor enters that request's allocation envelope before binding or
    /// forwarding it, including a solo request. Holder work is serialized under
    /// the compute lock; a multi-request native graph is not authorized by this
    /// contract. Prefix materialization uses the destination request's envelope.</summary>
    public bool UsePerSequenceHolders { get; init; }

    /// <summary>Re-estimate incremental execution growth after the physical
    /// boundary callback trims retained caches. Requires either one active
    /// serial request or explicitly isolated per-request holders: a queue head
    /// must not retain a stale estimate from before earlier work allocated state.</summary>
    public bool ReestimateAfterPhysicalCheck { get; init; }

    public RequestMemoryExecutionShape? ExecutionShape { get; init; }

    internal void ValidateConfiguration(SchedulerConfig config)
    {
        if (EstimateMaterializationPeak != null && (!UsePerSequenceHolders || !HasPhysicalCheck
            || !ReestimateAfterPhysicalCheck || AfterPrefixMaterialization == null))
            throw new ArgumentException("Materialization admission requires fenced independently owned holders and an execution transition.", nameof(config));
        if (BeforeAdmission != null && BeforeAdmissionWithRunning != null)
            throw new ArgumentException("Configure only one physical admission callback.", nameof(config));
        if (EnterBatchedExecution != null && (!UsePerSequenceHolders || !HasPhysicalCheck))
            throw new ArgumentException("Shared batched execution requires independently admitted holders and physical admission.", nameof(config));
        if (AfterPrefixMaterialization != null && !UsePerSequenceHolders)
            throw new ArgumentException("Prefix admission transitions require independently admitted holders.", nameof(config));
        if (UsePerSequenceHolders && (EnterSerialExecution == null || config.Speculation.Enabled
            || config.KvSnapshots != null || !config.ResolveExecutionOptions().PerSeqFusedEnabled
            || (config.ResolveExecutionOptions().BatchedFusedDecodeEnabled && EnterBatchedExecution == null)))
            throw new ArgumentException("Admitted holders require request scopes, an explicit shared scope for joint graphs, and no speculation or independent block snapshot adapter.", nameof(config));
        if (ReestimateAfterPhysicalCheck && (!HasPhysicalCheck || EnterSerialExecution == null
            || (config.MaxNumRunningSequences != 1 && !UsePerSequenceHolders)))
            throw new ArgumentException("Incremental peak refresh requires physical checks and independently owned serial execution state.", nameof(config));
        if (ExecutionShape is not { } shape) return;
        if (shape.BlockTokens <= 0 || shape.MaximumRunningRequests <= 0 || shape.MaximumPrefillTokens <= 0
            || shape.Snapshots == null || EnterSerialExecution == null || !ReferenceEquals(shape.Snapshots, config.KvSnapshots)
            || !ReferenceEquals(shape.Snapshots.SharedBudget, Budget)
            || config.BlockSize != shape.BlockTokens || config.MaxNumRunningSequences <= 0
            || config.MaxNumRunningSequences > shape.MaximumRunningRequests
            || config.EnablePrefixCaching || config.Speculation.Enabled)
            throw new ArgumentException("The scheduler must preserve the request estimator's snapshot, block, concurrency and non-speculative prefix-free execution contract.", nameof(config));
    }

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
