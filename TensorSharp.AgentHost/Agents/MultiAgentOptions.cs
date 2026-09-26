// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System;

namespace TensorSharp.AgentHost.Agents;

/// <summary>Request-scoped limits. The model decides whether and what to delegate.</summary>
public sealed class MultiAgentOptions
{
    public bool Enabled { get; init; } = true;
    /// <summary>Executing descendants across the entire tree, excluding the root.
    /// Queued tasks and parents awaiting child results do not occupy a slot.</summary>
    public int MaxConcurrentAgents { get; init; } = 3;
    public int MaxAgents { get; init; } = 8;
    public int MaxDepth { get; init; } = 2;
    public int MaxRoundsPerAgent { get; init; } = 8;
    public int MaxTotalChildGenerations { get; init; } = 48;
    public int AgentTimeoutSeconds { get; init; } = 180;
    public int MaxResultCharacters { get; init; } = 8000;
    public int MaxTaskCharacters { get; init; } = 16000;
    /// <summary>Opt in to workers using mutable tools in private workspaces under the parent's sandbox policy.
    /// Explorer/reviewer roles remain read-only. Client-owned tools are never inherited.</summary>
    public bool AllowWorkerTools { get; init; }

    public void Validate()
    {
        Range(MaxConcurrentAgents, 1, 32, nameof(MaxConcurrentAgents));
        Range(MaxAgents, 1, 128, nameof(MaxAgents));
        Range(MaxDepth, 1, 8, nameof(MaxDepth));
        Range(MaxRoundsPerAgent, 1, 64, nameof(MaxRoundsPerAgent));
        Range(MaxTotalChildGenerations, 1, 1024, nameof(MaxTotalChildGenerations));
        Range(AgentTimeoutSeconds, 1, 3600, nameof(AgentTimeoutSeconds));
        Range(MaxResultCharacters, 256, 64000, nameof(MaxResultCharacters));
        Range(MaxTaskCharacters, 256, 64000, nameof(MaxTaskCharacters));
    }

    private static void Range(int value, int min, int max, string name)
    {
        if (value < min || value > max)
            throw new ArgumentOutOfRangeException(name, $"Must be between {min} and {max}.");
    }
}
