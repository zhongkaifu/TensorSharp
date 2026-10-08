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
}
