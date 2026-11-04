// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Runtime.Speculative;

/// <summary>One-time default selection, not a mutable execution or timing policy.
/// An explicit environment choice and caller-owned governors remain authoritative.</summary>
internal static class SpeculationGovernorFactory
{
    internal readonly record struct Profile(object Target, object Head, Type Algorithm, int MaxDraft,
        float MinProbability, bool Enabled, string Source)
    {
        internal bool Matches(in Profile other) => ReferenceEquals(Target, other.Target)
            && ReferenceEquals(Head, other.Head) && Algorithm == other.Algorithm
            && MaxDraft == other.MaxDraft && MinProbability.Equals(other.MinProbability)
            && Enabled == other.Enabled && Source == other.Source;
    }

    internal static Profile Describe(ISpeculativeTarget target, ISpeculator speculator, bool qualifiedGreedy,
        string? explicitAdaptive)
    {
        ArgumentNullException.ThrowIfNull(target); ArgumentNullException.ThrowIfNull(speculator);
        var block = speculator as BlockDraftSpeculator;
        int k = Math.Min(speculator.MaxDraftTokens, Math.Max(0, target.SpecMaxDraftTokens));
        bool fixedProfile = explicitAdaptive == null && qualifiedGreedy
            && block != null && ReferenceEquals(block.HeadIdentity, target)
            && k == 7 && speculator.MinDraftProb == 0 && target.SpecPrefersFixedCostGovernor;
        // Preserve the existing semantics: only explicit "0" disables; every
        // other nonnull value enables. A null value permits this narrow default.
        bool enabled = explicitAdaptive != null ? explicitAdaptive != "0" : !fixedProfile;
        string source = explicitAdaptive != null ? "explicit-TS_SPEC_ADAPTIVE"
            : fixedProfile ? "qualified-native-ninfer-dflash-k7" : "legacy-default";
        return new(target, (object?)block?.HeadIdentity ?? target, speculator.GetType(), k,
            speculator.MinDraftProb, enabled, source);
    }

    internal static SpeculationCostGovernor Create(in Profile profile)
        => new() { Enabled = profile.Enabled, SelectionSource = profile.Source };

    internal static SpeculationCostGovernor CreateFor(ISpeculativeTarget target, ISpeculator speculator,
        bool qualifiedGreedy = false)
        => Create(Describe(target, speculator, qualifiedGreedy, SpeculationCostGovernor.AdaptiveEnvironmentSnapshot));

    internal static SpeculationCostGovernor Share(ref SpeculationCostGovernor? current, ref Profile? previous,
        in Profile profile)
    {
        // Live executions retain their instance. Never toggle a shared governor
        // when a later request resolves a different sampler/head/profile.
        if (current == null || !previous.HasValue || !profile.Matches(previous.Value))
        { current = Create(profile); previous = profile; }
        return current;
    }
}
