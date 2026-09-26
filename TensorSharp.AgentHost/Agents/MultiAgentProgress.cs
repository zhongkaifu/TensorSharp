// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

namespace TensorSharp.AgentHost.Agents;

/// <summary>A point-in-time view of a child task for host progress displays.
/// Contains no generation tokens or private reasoning, and reading it does not consume a result.</summary>
public sealed record MultiAgentProgress(
    string AgentId,
    string ParentId,
    string Task,
    string AgentType,
    string Status,
    string? Tool,
    string? ToolStatus,
    string? Detail,
    string? Result,
    string? Error)
{
    public string? WorkspaceId { get; init; }
    public string Permissions { get; init; } = "read-only";
    public System.Collections.Generic.IReadOnlyList<string> DependsOn { get; init; } = System.Array.Empty<string>();
}
