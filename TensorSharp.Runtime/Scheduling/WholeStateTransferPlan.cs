// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TensorSharp.Memory;

namespace TensorSharp.Runtime.Scheduling;

[Flags]
public enum WholeStateComponents
{
    None = 0,
    AttentionKv = 1,
    Recurrence = 2,
    Convolution = 4,
    Indexer = 8,
    PositionMetadata = 16,
}

/// <summary>Backend-declared coverage of a complete state transfer, independent
/// of the block snapshot interface. Coverage is not execution qualification.</summary>
public sealed record WholeStateTransferCapabilities(WholeStateComponents Required,
    WholeStateComponents Exported, WholeStateComponents Imported, bool FlushesJointAliases);

/// <summary>State observed at a fenced compute boundary. Active speculative
/// state and more than one physical device require separate transfer adapters.</summary>
public readonly record struct WholeStateExecutionState(int ComputedTokens, int PhysicalDevices = 1,
    bool HasJointAliases = false, bool HasActiveDraftState = false, bool HasUncommittedVerification = false);

/// <summary>Incremental allocation peaks. Capture payload is a new recovery
/// copy; an existing charged host mirror is excluded. RestoreState describes
/// new destination allocations, excluding recovery storage still owned by its
/// snapshot. Payload/state coexist with scratch during their respective phase.</summary>
public sealed record WholeStateTransferMemory(IReadOnlyList<MemoryCharge> CapturePayload,
    IReadOnlyList<MemoryCharge> CaptureScratch, IReadOnlyList<MemoryCharge> RestoreState,
    IReadOnlyList<MemoryCharge> RestoreScratch);

/// <summary>Prerequisite contract for complete-state request preemption. This
/// planner neither exports state nor authorizes a scheduler execution path.
/// A backend adapter must flush every joint alias, successfully export all
/// required state and revalidate this boundary before releasing device state.
/// A snapshot ends at ComputedTokens, never at a sampled uncomputed tail.
/// Runtime metadata remains managed memory; these peaks describe explicitly
/// allocated transfer payloads, not a GC or driver physical hard-limit claim.</summary>
public sealed class WholeStateTransferPlan
{
    private readonly SequenceState _sequence;
    private readonly object _computeLock;
    private readonly int[] _tokens;
    private readonly int _promptCount;
    private readonly float[]? _lastLogits;
    private readonly float[]? _lastLogitValues;
    private readonly int? _pendingDeviceToken;
    private readonly int _pendingDevicePosition;

    private WholeStateTransferPlan(SequenceState sequence, object computeLock,
        IReadOnlyList<MemoryCharge> capturePeak, IReadOnlyList<MemoryCharge> restorePeak)
    {
        _sequence = sequence;
        _computeLock = computeLock;
        ComputedTokens = sequence.NumComputedTokens;
        _promptCount = sequence.PromptTokens.Count;
        _tokens = new int[sequence.NumTotalTokens];
        for (int i = 0; i < _tokens.Length; ++i) _tokens[i] = sequence.TokenAt(i);
        _lastLogits = sequence.LastLogits;
        _lastLogitValues = sequence.LastLogits == null ? null : (float[])sequence.LastLogits.Clone();
        _pendingDeviceToken = sequence.PendingDeviceToken;
        _pendingDevicePosition = sequence.PendingDevicePosition;
        CapturePeak = capturePeak;
        RestorePeak = restorePeak;
    }

    public string RequestId => _sequence.RequestId;
    public int ComputedTokens { get; }
    public int TotalTokens => _tokens.Length;
    public int UncomputedTokens => TotalTokens - ComputedTokens;
    public IReadOnlyList<MemoryCharge> CapturePeak { get; }
    public IReadOnlyList<MemoryCharge> RestorePeak { get; }

    /// <summary>Logical bytes of managed token/logit copies used by the fence.
    /// GC headers and allocator overhead need additional headroom. An adapter
    /// must plan these copies separately from physical payload accounting and
    /// may obtain this estimate before creating a plan.</summary>
    public static long ManagedBoundaryCopyBytes(SequenceState sequence)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        return checked(4L * (sequence.NumTotalTokens + (long)(sequence.LastLogits?.Length ?? 0)));
    }

    public int UncomputedTokenAt(int offset)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        if (offset >= UncomputedTokens) throw new ArgumentOutOfRangeException(nameof(offset));
        return _tokens[ComputedTokens + offset];
    }

    public static WholeStateTransferPlan Prepare(SequenceState sequence, object computeLock,
        WholeStateExecutionState state, WholeStateTransferCapabilities capabilities, WholeStateTransferMemory memory)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        ArgumentNullException.ThrowIfNull(computeLock);
        ArgumentNullException.ThrowIfNull(capabilities);
        ArgumentNullException.ThrowIfNull(memory);
        RequireLock(computeLock);
        const WholeStateComponents known = WholeStateComponents.AttentionKv | WholeStateComponents.Recurrence
            | WholeStateComponents.Convolution | WholeStateComponents.Indexer | WholeStateComponents.PositionMetadata;
        if (capabilities.Required == WholeStateComponents.None || (capabilities.Required & ~known) != 0
            || (capabilities.Exported & capabilities.Required) != capabilities.Required
            || (capabilities.Imported & capabilities.Required) != capabilities.Required)
            throw new NotSupportedException("The complete required model state must be exported and restored; block KV coverage is insufficient.");
        if (state.PhysicalDevices != 1 || state.HasActiveDraftState || state.HasUncommittedVerification)
            throw new NotSupportedException("Complete-state transfer currently requires one device and a settled non-speculative request.");
        if (state.HasJointAliases && !capabilities.FlushesJointAliases)
            throw new NotSupportedException("Every joint graph alias must be flushed before complete-state export.");
        if (sequence.NumComputedTokens <= 0 || sequence.NumComputedTokens > sequence.NumTotalTokens
            || state.ComputedTokens != sequence.NumComputedTokens)
            throw new InvalidOperationException("The authoritative model position must equal the request's computed boundary.");
        if (sequence.PendingDeviceToken != null && sequence.PendingDevicePosition != sequence.NumComputedTokens)
            throw new InvalidOperationException("The pending device token does not belong to this computed boundary.");
        return new(sequence, computeLock, Combine(memory.CapturePayload, memory.CaptureScratch),
            Combine(memory.RestoreState, memory.RestoreScratch));
    }

    /// <summary>Revalidate immediately before committing capture/releasing any
    /// authoritative device binding. The caller must already have flushed all
    /// aliases and fenced every export; this method does not perform that I/O.</summary>
    public void ValidateBeforeDeviceRelease(SequenceState sequence, int authoritativeComputedTokens)
    {
        ValidateLogicalState(sequence);
        if (authoritativeComputedTokens != ComputedTokens)
            throw new InvalidOperationException("The authoritative model state changed during transfer.");
    }

    /// <summary>Preserved scheduler state may move between Running and Waiting,
    /// but must retain its computed head, exact tokens, logits and pending token.
    /// ResetForPreemption is a replay route, not whole-state recovery.</summary>
    public void ValidateLogicalState(SequenceState sequence)
    {
        RequireLock(_computeLock);
        if (!ReferenceEquals(sequence, _sequence) || sequence.NumComputedTokens != ComputedTokens
            || sequence.PromptTokens.Count != _promptCount || sequence.NumTotalTokens != TotalTokens
            || !ReferenceEquals(sequence.LastLogits, _lastLogits) || sequence.PendingDeviceToken != _pendingDeviceToken
            || (_pendingDeviceToken != null && sequence.PendingDevicePosition != _pendingDevicePosition))
            throw new InvalidOperationException("The request owner or generation boundary changed during transfer.");
        for (int i = 0; i < _tokens.Length; ++i)
            if (sequence.TokenAt(i) != _tokens[i])
                throw new InvalidOperationException("The token history changed during transfer.");
        if (_lastLogitValues != null)
            for (int i = 0; i < _lastLogitValues.Length; ++i)
                if (BitConverter.SingleToInt32Bits(sequence.LastLogits![i]) != BitConverter.SingleToInt32Bits(_lastLogitValues[i]))
                    throw new InvalidOperationException("The sampling logits changed during transfer.");
    }

    public BudgetReservation? TryReserveCapture(MemoryBudget budget, int authoritativeComputedTokens,
        IReadOnlyList<MemoryCharge>? physicalConstraints = null)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ValidateBeforeDeviceRelease(_sequence, authoritativeComputedTokens);
        return physicalConstraints == null ? budget.TryReserve(CapturePeak) : budget.TryReserve(CapturePeak, physicalConstraints);
    }

    public BudgetReservation? TryReserveRestore(MemoryBudget budget,
        IReadOnlyList<MemoryCharge>? physicalConstraints = null)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ValidateLogicalState(_sequence);
        return physicalConstraints == null ? budget.TryReserve(RestorePeak) : budget.TryReserve(RestorePeak, physicalConstraints);
    }

    private static IReadOnlyList<MemoryCharge> Combine(IReadOnlyList<MemoryCharge> payload,
        IReadOnlyList<MemoryCharge> scratch)
    {
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentNullException.ThrowIfNull(scratch);
        var totals = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var charge in payload.Concat(scratch))
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(charge.Pool);
            ArgumentOutOfRangeException.ThrowIfNegative(charge.Bytes);
            totals[charge.Pool] = checked(totals.GetValueOrDefault(charge.Pool) + charge.Bytes);
        }
        if (totals.Count == 0) throw new ArgumentException("A transfer phase must name its physical pools, including zero-byte phases.");
        return Array.AsReadOnly(totals.OrderBy(p => p.Key, StringComparer.Ordinal)
            .Select(p => new MemoryCharge(p.Key, p.Value)).ToArray());
    }

    private static void RequireLock(object computeLock)
    {
        if (!Monitor.IsEntered(computeLock))
            throw new InvalidOperationException("Complete-state transfer must run under the model compute lock.");
    }
}
