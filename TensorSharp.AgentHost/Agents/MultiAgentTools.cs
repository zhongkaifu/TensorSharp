// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Agents;

/// <summary>Flat schemas work with TensorSharp's local model tool renderers.</summary>
public static class MultiAgentTools
{
    public const string Spawn = "spawn_agent";
    public const string Wait = "wait_agent";
    public const string Send = "send_input";
    public const string Close = "close_agent";
    public const string List = "list_agents";

    public static bool IsTool(string? name) => name is Spawn or Wait or Send or Close or List;

    // Keep ordering and enforcement tied to the same allowlist. Coordination
    // tools are handled separately: they do not grant filesystem permissions.
    internal static bool IsReadOnlyTool(string? name) =>
        name is SkillTools.ReadToolName or SkillTools.ListToolName or SkillToolNames.ReadFile;

    public static List<ToolFunction> Merge(IReadOnlyList<ToolFunction>? tools)
    {
        var result = tools?.ToList() ?? new List<ToolFunction>();
        var names = new HashSet<string>(result.Select(t => t.Name), StringComparer.OrdinalIgnoreCase);
        foreach (ToolFunction tool in Create())
            if (names.Add(tool.Name)) result.Add(tool);
        // Templates can render tools before all system instructions. Put tools
        // shared with read-only children first so their declarations form one long
        // exact prefix. This stable partition preserves schemas, objects, and the
        // relative order within each group; it never expands a child's allowlist.
        return result.Where(tool => IsReadOnlyTool(tool.Name) || IsTool(tool.Name))
            .Concat(result.Where(tool => !IsReadOnlyTool(tool.Name) && !IsTool(tool.Name))).ToList();
    }

    public static List<ToolFunction> Create() => new()
    {
        Tool(Spawn, "Delegate one well-defined part of the user's task to a subagent in a private workspace. Returns immediately; independent tasks run in parallel up to host capacity and excess tasks queue. Give necessary context, expected outputs and acceptance checks. Declare prerequisites for dependent tasks. Do not duplicate assigned work.",
            new[] { "task_name", "task" },
            ("task_name", "string", "Unique short name using letters, digits, underscores or hyphens."),
            ("task", "string", "Self-contained task, relevant facts, ownership boundaries and expected result. Child has no parent conversation."),
            ("agent_type", "string", "explorer (default) for research, reviewer for verification, worker for implementation. Explorer/reviewer are read-only; worker writes require host opt-in."),
            ("permissions", "string", "read-only or workspace-write. Defaults to the role's host-permitted access. Explicit workspace-write requires worker and cannot exceed parent permissions. Writes are confined to the private child workspace."),
            ("input_files", "string", "Newline-separated parent-workspace relative file paths to copy. Include all needed files; no implicit parent or sibling filesystem access. No directories, traversal or symlinks."),
            ("depends_on", "string", "Comma- or newline-separated existing sibling agent IDs. Runs only after all complete successfully; failed prerequisites block this task. Reports and output files are supplied automatically.")),
        Tool(Wait, "Wait for child results without polling. Omit agent_id for all direct children. Timeout does not mean completion; inspect status. Verify reports before synthesis. Returned files[].path values are relative to YOUR workspace and ready for read_file. workspace_id and agent_id are logical identifiers, not filesystem paths; never prefix a returned file path with either ID.",
            Array.Empty<string>(), ("agent_id", "string", "Child ID returned by spawn_agent; omit for all direct children."),
            ("timeout_ms", "integer", "Wait up to 60000 ms; default 10000.")),
        Tool(Send, "Send a bounded follow-up to a child. A running child sees it at its next generation boundary; a completed child starts another turn with its own history.",
            new[] { "agent_id", "message" }, ("agent_id", "string", "Child ID."), ("message", "string", "Clarification or additional bounded task.")),
        Tool(Close, "Cancel a child and its descendants. Cancellation is not successful task completion.",
            new[] { "agent_id" }, ("agent_id", "string", "Child ID.")),
        Tool(List, "Inspect your children and their status. Use wait_agent when waiting for work rather than repeatedly listing.", Array.Empty<string>()),
    };

    private static ToolFunction Tool(string name, string description, string[] required,
        params (string Name, string Type, string Description)[] parameters) => new()
    {
        Name = name, Description = description, Required = required.ToList(),
        Parameters = parameters.ToDictionary(p => p.Name,
            p => new ToolParameter { Type = p.Type, Description = p.Description }),
    };
}
