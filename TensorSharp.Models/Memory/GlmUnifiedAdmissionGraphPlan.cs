// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using TensorSharp.Memory.Planning;

namespace TensorSharp.Models;

/// <summary>Serial prefill and joint decode have different participating slot
/// contexts. A long-running sibling does not widen the new request's serial
/// prefill graph; it does widen the later shared decode graph.</summary>
internal readonly record struct GlmUnifiedAdmissionGraphPlan(int SerialContext, int JointContext,
    int PrefillRows, int JointRows, InferenceMemoryBytes PrefillPeak, InferenceMemoryBytes JointPeak)
{
    internal InferenceMemoryBytes MaximumPeak => new(Math.Max(PrefillPeak.Host, JointPeak.Host),
        Math.Max(PrefillPeak.Device, JointPeak.Device));

    internal static GlmUnifiedAdmissionGraphPlan Create(GlmUnifiedMemoryGeometry geometry,
        int logicalContext, int prefillRows, int promptTokens, int computedTokens,
        IReadOnlyList<int> runningComputedTokens, int generations)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(runningComputedTokens);
        if (logicalContext <= 0 || logicalContext > geometry.DeclaredContextTokens
            || prefillRows is < 1 or > 1024 || promptTokens < 0 || promptTokens > logicalContext
            || computedTokens < 0 || computedTokens >= logicalContext
            || runningComputedTokens.Count > 7)
            throw new ArgumentOutOfRangeException(nameof(logicalContext));
        int rows = Math.Min(prefillRows, Math.Max(1, promptTokens - computedTokens));
        int serial = Math.Max(Math.Max(1, promptTokens), checked(computedTokens + rows));
        int joint = serial;
        foreach (int computed in runningComputedTokens)
        {
            if (computed < 0 || computed >= logicalContext) throw new ArgumentOutOfRangeException(nameof(runningComputedTokens));
            joint = Math.Max(joint, checked(computed + 1));
        }
        int Pad(int tokens) => (int)Math.Min(logicalContext, ((long)tokens + 255) / 256 * 256);
        serial = Pad(serial); joint = Pad(joint);
        int jointRows = runningComputedTokens.Count + 1;
        return new(serial, joint, rows, jointRows, geometry.GraphPeak(serial, rows, false, generations),
            geometry.GraphPeak(joint, jointRows, jointRows > 1, generations));
    }
}
