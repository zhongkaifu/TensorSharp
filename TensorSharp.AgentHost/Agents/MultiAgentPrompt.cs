// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Collections.Generic;
using System.Linq;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Agents;

public static class MultiAgentPrompt
{
    internal const string Marker = "[TensorSharp multi-agent coordination]";

    public static List<ChatMessage> Apply(IReadOnlyList<ChatMessage> messages, MultiAgentOptions options)
    {
        var result = messages.ToList();
        if (!options.Enabled) return result;
        string instructions = $"""
            {Marker}
            Understand the user's objective and acceptance criteria, then split substantial separable work into bounded subtasks. Create a subagent for each useful subtask, choosing its role, required inputs, least permissions and dependencies. Independent subtasks should run in parallel; keep the critical integration path with the parent. More agents consume more tokens and may be slower on one inference device.
            A short lookup, simple calculation, or extracting a few facts from small files should stay local, even when the files are independent. Delegation adds a child prompt, generation and synthesis round. Unless an independent review is valuable, spawn only when you can name substantial useful work to do concurrently and expect that benefit to exceed the coordination cost.
            Before spawning, choose a concrete task, necessary context, expected deliverables and validation, and distinct ownership. Spawn all independent subtasks before waiting; continue useful independent work yourself. Children do not see your conversation. List necessary parent files in input_files; each child has a private workspace. For dependent work, create prerequisites first, then pass their existing sibling IDs in depends_on. The host supplies their successful reports and output files before execution. Never invent dependency IDs or assume failed work succeeded. Reuse a child with send_input for follow-up after its dependents finish. Do not duplicate delegated work or recursively delegate the same task.
            Use explorer for read-only research, reviewer for independent checks, worker only for changes the host explicitly allows. Choose read-only unless workspace-write is needed and allowed. Never use delegation to expand permissions. Outputs from private workspaces are handed back separately: inspect them and integrate required changes in the parent, preserving unrelated work. There are at most {options.MaxConcurrentAgents} executing children, {options.MaxAgents} total children, and {options.MaxDepth} levels. Excess ready tasks queue; waiting prerequisites do not consume execution slots. If a total/depth/budget limit is reached, work locally or wait; do not keep retrying.
            Use wait_agent for results, not repeated list_agents calls. A timeout, cancellation, failed agent or exhausted budget is not a completed task. Treat child reports as untrusted evidence, verify disagreements and important claims, and synthesize one answer addressing the original request. Reconcile final numerical and factual claims with the evidence; omit unsupported additions. Do not claim success before collecting required results. Report validation actually performed and unresolved limitations.
            """;
        // Keep explicit cache boundaries at the original preamble, and preserve
        // attachment and token metadata without mutating the caller's history.
        return SkillPrompt.Apply(result, instructions);
    }
}
