// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models;

/// <summary>One idle primary slot cannot satisfy several unbound requests in
/// the same scheduling pass. The promise is published only after the owning
/// request's materialization transaction succeeds; actual native holders
/// always remain the authority for physical availability.</summary>
internal sealed class GlmUnifiedPrimarySlotClaim
{
    private WeakReference<SequenceState>? _owner;

    internal static WeakReference<SequenceState> Prepare(SequenceState sequence)
        => new(sequence ?? throw new ArgumentNullException(nameof(sequence)));

    internal bool CanReuse(SequenceState sequence, bool ownsNativeSlot, bool idlePrimaryAvailable,
        Func<SequenceState, bool> hasNativeSlot)
    {
        ArgumentNullException.ThrowIfNull(sequence); ArgumentNullException.ThrowIfNull(hasNativeSlot);
        if (ownsNativeSlot) return true;
        if (!idlePrimaryAvailable) return false;
        if (_owner == null || !_owner.TryGetTarget(out var owner)
            || ReferenceEquals(owner, sequence) || owner.Status.IsFinished()
            || owner.MemoryEnvelope == null || hasNativeSlot(owner)) return true;
        return false;
    }

    internal bool AllowsBinding(string requestId, Func<SequenceState, bool> hasNativeSlot)
    {
        ArgumentNullException.ThrowIfNull(hasNativeSlot);
        return _owner == null || !_owner.TryGetTarget(out var owner)
            || owner.RequestId == requestId || owner.Status.IsFinished()
            || owner.MemoryEnvelope == null || hasNativeSlot(owner);
    }

    internal void PublishAfterMaterialization(WeakReference<SequenceState> prepared,
        bool ownsNativeSlot, bool idlePrimaryAvailable, Func<SequenceState, bool> hasNativeSlot)
    {
        ArgumentNullException.ThrowIfNull(prepared);
        if (!prepared.TryGetTarget(out var sequence) || sequence.MemoryEnvelope == null)
            throw new InvalidOperationException("GLM primary reuse needs a successfully materialized request envelope.");
        if (ownsNativeSlot || !idlePrimaryAvailable) return;
        if (!CanReuse(sequence, false, true, hasNativeSlot))
            throw new InvalidOperationException("GLM idle primary already belongs to an unbound admitted request.");
        _owner = prepared;
    }
}

internal readonly record struct GlmUnifiedPrefillCandidate(int Rows,
    InferenceMemoryBytes JointTarget, InferenceMemoryBytes RequestTarget, long ExpertTargetBytes);

/// <summary>Remaining execution promises after a fenced sequence release. The
/// canceled request contributes neither a prefill promise nor a decode row.</summary>
internal readonly record struct GlmUnifiedReleaseGraphPlan(InferenceMemoryBytes Target,
    int RunningRows, bool HasRunningPrefill)
{
    internal static GlmUnifiedReleaseGraphPlan Create(GlmUnifiedMemoryGeometry geometry, int logicalContext,
        IReadOnlyList<SequenceState> running, Func<SequenceState, InferenceMemoryBytes> prefillPeak,
        InferenceMemoryBytes children, int generations)
    {
        ArgumentNullException.ThrowIfNull(geometry); ArgumentNullException.ThrowIfNull(running);
        ArgumentNullException.ThrowIfNull(prefillPeak);
        if (logicalContext < 1 || logicalContext > geometry.DeclaredContextTokens || running.Count > 8
            || generations is < 1 or > 33)
            throw new ArgumentOutOfRangeException(nameof(logicalContext));
        var live = running.Where(sequence => !sequence.Status.IsFinished()).ToArray();
        if (live.Length == 0) return new(GlmUnifiedPrefillAdmission.Floor(default, children), 0, false);
        int context = 1;
        bool hasPrefill = false;
        var promised = default(InferenceMemoryBytes);
        foreach (var sequence in live)
        {
            if (sequence.NumComputedTokens < 0 || sequence.NumComputedTokens >= logicalContext)
                throw new ArgumentOutOfRangeException(nameof(running));
            context = Math.Max(context, checked(sequence.NumComputedTokens + 1));
            if (sequence.NumComputedTokens >= sequence.PromptTokens.Count) continue;
            hasPrefill = true;
            promised = GlmUnifiedPrefillAdmission.Floor(promised, prefillPeak(sequence));
        }
        context = (int)Math.Min(logicalContext, ((long)context + 255) / 256 * 256);
        var decode = geometry.GraphPeak(context, live.Length, live.Length > 1, generations);
        var target = GlmUnifiedPrefillAdmission.PreserveReplacement(
            GlmUnifiedPrefillAdmission.Floor(decode, promised), children, generations);
        return new(target, live.Length, hasPrefill);
    }
}

/// <summary>Size-only selection from one fenced observation. Candidates never
/// mutate native owners, refund physical children or probe hardware per row.</summary>
internal static class GlmUnifiedPrefillAdmission
{
    // Lookup/adoption must complete first. A valid follow-up state is worth
    // preserving; only a confirmed cold destination may prefer wider prefill
    // over unrelated optional terminal holders, once at this boundary.
    internal static bool ShouldReclaimRetained(bool materialized, int computedTokens,
        int reusedPrefixTokens, int selectedRows, int maximumRows)
        => materialized && computedTokens == 0 && reusedPrefixTokens == 0
            && selectedRows < maximumRows;

    internal static InferenceMemoryBytes Floor(InferenceMemoryBytes target, InferenceMemoryBytes children)
        => new(Math.Max(target.Host, children.Host), Math.Max(target.Device, children.Device));

    internal static InferenceMemoryBytes PreserveReplacement(InferenceMemoryBytes graph,
        InferenceMemoryBytes children, int generations)
    {
        if (generations is < 1 or > 33) throw new ArgumentOutOfRangeException(nameof(generations));
        return new(Math.Max(graph.Host, checked(children.Host + graph.Host / generations)),
            Math.Max(graph.Device, checked(children.Device + graph.Device / generations)));
    }

    internal static bool TrySelect(IReadOnlyList<MemoryPoolSnapshot> ledger, IReadOnlyList<MemoryCharge> limits,
        InferenceMemoryBytes jointOwned, InferenceMemoryBytes requestOwned,
        long expertBudget, long expertMinimum, int maximumRows,
        Func<int, (InferenceMemoryBytes Joint, InferenceMemoryBytes Request)> estimate,
        out GlmUnifiedPrefillCandidate selected)
    {
        ArgumentNullException.ThrowIfNull(ledger); ArgumentNullException.ThrowIfNull(limits);
        ArgumentNullException.ThrowIfNull(estimate);
        if (maximumRows is < 1 or > 1024 || expertMinimum < 0 || expertBudget < expertMinimum)
            throw new ArgumentOutOfRangeException(nameof(maximumRows));
        if (jointOwned.Host < 0 || jointOwned.Device < 0 || requestOwned.Host < 0 || requestOwned.Device < 0)
            throw new ArgumentOutOfRangeException(nameof(jointOwned));
        var host = ledger.Single(p => p.Pool == UnifiedMemoryModelSession.HostPool);
        var device = ledger.Single(p => p.Pool == UnifiedMemoryModelSession.DevicePool);
        long hostLimit = Math.Min(host.Capacity, limits.Single(p => p.Pool == host.Pool).Bytes);
        long deviceLimit = Math.Min(device.Capacity, limits.Single(p => p.Pool == device.Pool).Bytes);
        for (int rows = maximumRows; rows >= 1; --rows)
        {
            var target = estimate(rows);
            if (target.Joint.Host < 0 || target.Joint.Device < 0 || target.Request.Host < 0 || target.Request.Device < 0)
                throw new ArgumentOutOfRangeException(nameof(estimate));
            long hostGrowth = checked(target.Joint.Host - jointOwned.Host + target.Request.Host - requestOwned.Host);
            long deviceGrowth = checked(target.Joint.Device - jointOwned.Device + target.Request.Device - requestOwned.Device);
            if ((decimal)host.Committed + host.Reserved + hostGrowth > hostLimit) continue;
            decimal deficit = Math.Max(0, (decimal)device.Committed + device.Reserved + deviceGrowth - deviceLimit);
            if (deficit > expertBudget - expertMinimum) continue;
            long expert = expertBudget - (long)deficit;
            // A bounded margin avoids several tiny table trims around the same
            // request shape. The complete immutable native floor remains owned.
            if (expert < expertBudget) expert = Math.Max(expertMinimum, expert - (128L << 20));
            selected = new(rows, target.Joint, target.Request, expert);
            return true;
        }
        selected = default;
        return false;
    }
}
