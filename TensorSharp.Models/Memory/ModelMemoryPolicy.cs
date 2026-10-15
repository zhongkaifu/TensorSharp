// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;

namespace TensorSharp.Models;

/// <summary>Per-model execution geometry chosen before loading. Unlike process
/// environment variables this policy cannot change another model's context or
/// prefill shape. The adaptive loader supplies it after accounting for weights,
/// state, context and workspace. It is not itself an allocation quota.</summary>
public sealed record ModelMemoryPolicy
{
    public ModelMemoryPolicy(int contextTokens, int prefillChunkTokens)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(prefillChunkTokens);
        if (prefillChunkTokens > contextTokens)
            throw new ArgumentOutOfRangeException(nameof(prefillChunkTokens));
        ContextTokens = contextTokens;
        PrefillChunkTokens = prefillChunkTokens;
    }

    public int ContextTokens { get; }
    public int PrefillChunkTokens { get; }
    /// <summary>The admitted graph contains only the trunk. Resolve before
    /// loading so later environment changes cannot add unaccounted draft weights.</summary>
    public bool OmitEmbeddedDraftWeights { get; init; }

    /// <summary>Optional per-session selected-expert cache ceiling. Null retains
    /// ordinary environment-controlled placement; zero explicitly disables it.</summary>
    public long? ExpertCacheBytes { get; init; }
    public int ExpertCachePrefillRows { get; init; } = 8;
    /// <summary>Device holdout for this model's admitted execution and vision
    /// workspace, kept free when selecting whole expert layers.</summary>
    public long DeviceExecutionReserveBytes { get; init; }
    /// <summary>Per-model native GLM leading routed-expert CPU placement; null
    /// retains the ordinary capacity/environment policy.</summary>
    public int? NativeMoeCpuLayers { get; init; }

}
