// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Cli;

/// <summary>Resolve CLI overrides into the shared engine's speculation policy.</summary>
internal static class SpeculativeDecodingOptions
{
    internal static SpeculationOptions Resolve(int specDraftMax, float specDraftConfMin)
    {
        var options = SpeculationOptions.FromEnvironment();
        return new SpeculationOptions
        {
            Enabled = options.Enabled,
            ExplicitlyDisabled = options.ExplicitlyDisabled,
            SpeculatorName = options.SpeculatorName,
            MaxDraftTokens = specDraftMax > 0 ? specDraftMax : Math.Max(1, options.MaxDraftTokens),
            MaxDraftTokensExplicit = specDraftMax > 0 || options.MaxDraftTokensExplicit,
            MinDraftProb = specDraftConfMin >= 0f ? specDraftConfMin : options.MinDraftProb,
        };
    }
}
