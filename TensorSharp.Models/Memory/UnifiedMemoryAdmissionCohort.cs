// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models;

/// <summary>Snapshot the current admitted owners under the engine's compute
/// lock. Admission-time sibling lists are not authoritative after new requests
/// arrive. This helper neither changes reservations nor grants physical credit.</summary>
internal static class UnifiedMemoryAdmissionCohort
{
    internal static SequenceState[] SnapshotOtherAdmitted(IEnumerable<SequenceState> candidates,
        SequenceState current)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(current);
        var result = new List<SequenceState>();
        var seen = new HashSet<SequenceState>(ReferenceEqualityComparer.Instance);
        foreach (var sequence in candidates)
        {
            if (sequence is null) throw new ArgumentException("A cohort candidate cannot be null.", nameof(candidates));
            if (ReferenceEquals(sequence, current) || sequence.Status.IsFinished()
                || sequence.MemoryEnvelope is null || !seen.Add(sequence)) continue;
            result.Add(sequence);
        }
        // Finish every fallible metadata allocation before the caller begins
        // any graph/reservation transaction. Decoders remain in this snapshot.
        return result.ToArray();
    }
}
