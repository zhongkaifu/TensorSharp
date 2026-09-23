// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Agents
{
    /// <summary>
    /// The declarations of the five sub-agent tools, and the tolerant readers for their
    /// arguments.
    ///
    /// <para>
    /// <b>Every sentence here must be true of every task.</b> The declarations are read on
    /// every turn of every agentic request that has sub-agents enabled, so a worked example
    /// or a product name in them biases all of those turns toward one kind of work (see
    /// the tool-description over-fitting guard). What they carry instead is Codex's
    /// delegation policy, compressed: spawn only when asked, give a self-contained task,
    /// split write scopes, keep working while agents run, and wait only when blocked.
    /// </para>
    /// <para>
    /// <b>They are flat</b>, like every other declaration this host writes:
    /// <see cref="ToolParameter"/> cannot express an array, so <c>targets</c> is a string
    /// that is read tolerantly (comma- or space-separated, or a JSON array — models send
    /// all three).
    /// </para>
    /// </summary>
    public static class SubAgentTools
    {
        /// <summary>Default <c>wait_agent</c> timeout.</summary>
        /// <remarks>
        /// Five minutes rather than Codex's thirty seconds. Waiting returns the moment any
        /// awaited agent finishes, so a long default costs nothing when agents are quick —
        /// and on a local GPU they often are not, where every premature timeout is one more
        /// full generation of the parent spent deciding to wait again.
        /// </remarks>
        public const int DefaultWaitMs = 300_000;

        /// <summary>
        /// Shortest wait honoured: Codex's floor, for Codex's reason. A model that polls
        /// with one-second waits burns a generation per second doing nothing.
        /// </summary>
        public const int MinWaitMs = 10_000;

        /// <summary>Longest wait honoured, again Codex's.</summary>
        public const int MaxWaitMs = 3_600_000;

        /// <summary>The declarations, in the order the model should read them.</summary>
        public static List<ToolFunction> Declare() => new()
        {
            new ToolFunction
            {
                Name = SkillToolNames.SpawnAgent,
                Description =
                    "Start a sub-agent: a separate copy of you, with the same tools and the same working "
                    + "directory but its own fresh context, that works on one task in parallel with you and "
                    + "reports back a final answer. Returns the new agent's id at once; the agent keeps "
                    + "working in the background, and its final answer is delivered to you when it finishes "
                    + "(inside a <subagent_notification>, or as the result of " + SkillToolNames.WaitAgent + "). "
                    + "Only start sub-agents when the user, or the instructions of a skill you are following, "
                    + "asks for sub-agents, delegation or parallel work; a request for depth or thoroughness "
                    + "alone is not permission. Give each agent a concrete, self-contained task that includes "
                    + "everything it needs, because it cannot see this conversation. When agents will change "
                    + "files, give each one different files. Do not also do a delegated task yourself: work "
                    + "on something else, and wait only when your next step needs the result.",
                Parameters = new Dictionary<string, ToolParameter>
                {
                    ["message"] = new()
                    {
                        Type = "string",
                        Description =
                            "The task for the new agent, written so it can be done without seeing this "
                            + "conversation: the goal, the inputs it needs (file names, facts, constraints) "
                            + "and what its final answer must contain.",
                    },
                    ["fork_context"] = new()
                    {
                        Type = "boolean",
                        Description =
                            "True gives the agent a copy of this conversation so far instead of a fresh "
                            + "context. Omit it unless the task depends on context too long to restate "
                            + "in the message.",
                    },
                },
                Required = new List<string> { "message" },
            },
            new ToolFunction
            {
                Name = SkillToolNames.SendInput,
                Description =
                    "Send a follow-up message to one of your sub-agents. If it has finished, it starts "
                    + "working again on the message with all its earlier context; if it is still working, "
                    + "it reads the message before its next step. Its next final answer is delivered to "
                    + "you like the first one.",
                Parameters = new Dictionary<string, ToolParameter>
                {
                    ["target"] = new()
                    {
                        Type = "string",
                        Description = "The agent's id, as " + SkillToolNames.SpawnAgent + " returned it.",
                    },
                    ["message"] = new()
                    {
                        Type = "string",
                        Description = "What to tell the agent.",
                    },
                },
                Required = new List<string> { "target", "message" },
            },
            new ToolFunction
            {
                Name = SkillToolNames.WaitAgent,
                Description =
                    "Wait for your sub-agents and collect their final answers. Returns as soon as any of "
                    + "the awaited agents finishes, with the final answer of every awaited agent that has "
                    + "finished and was not reported to you yet; returns empty-handed when the timeout "
                    + "passes. Call it only when your next step needs a result, and prefer long timeouts "
                    + "to calling it repeatedly: results also reach you without waiting.",
                Parameters = new Dictionary<string, ToolParameter>
                {
                    ["targets"] = new()
                    {
                        Type = "string",
                        Description =
                            "Ids of the agents to wait for, separated by commas, or \"all\" to wait for any "
                            + "of your agents.",
                    },
                    ["timeout_ms"] = new()
                    {
                        Type = "integer",
                        Description =
                            "How long to wait, in milliseconds. Default "
                            + DefaultWaitMs.ToString(CultureInfo.InvariantCulture) + ", minimum "
                            + MinWaitMs.ToString(CultureInfo.InvariantCulture) + ", maximum "
                            + MaxWaitMs.ToString(CultureInfo.InvariantCulture) + ".",
                    },
                },
                // Required, as in Codex. It could not honestly be optional anyway: the
                // Jinja rendering path marks EVERY parameter required when Required is
                // empty, which would have forced a timeout on every call as well.
                Required = new List<string> { "targets" },
            },
            new ToolFunction
            {
                Name = SkillToolNames.CloseAgent,
                Description =
                    "Stop one of your sub-agents, and any agents it started, and discard it. Use it for an "
                    + "agent whose work you no longer need. Returns the agent's status before it was "
                    + "closed.",
                Parameters = new Dictionary<string, ToolParameter>
                {
                    ["target"] = new()
                    {
                        Type = "string",
                        Description = "The agent's id, as " + SkillToolNames.SpawnAgent + " returned it.",
                    },
                },
                Required = new List<string> { "target" },
            },
            new ToolFunction
            {
                Name = SkillToolNames.ListAgents,
                Description =
                    "List your sub-agents: each one's id, status (starting, running, completed, failed or "
                    + "closed) and task.",
                // No parameters at all: Parameters and Required stay empty together.
                Parameters = new Dictionary<string, ToolParameter>(),
                Required = new List<string>(),
            },
        };

        // ---- argument readers ----------------------------------------------------

        /// <summary>The first non-blank string among <paramref name="names"/>, trimmed.</summary>
        internal static string? ReadText(ToolCall call, params string[] names)
        {
            foreach (string name in names)
            {
                string? value = SkillTools.ReadString(call, name);
                if (!string.IsNullOrWhiteSpace(value))
                    return value.Trim();
            }
            return null;
        }

        /// <summary>A boolean argument, accepting the spellings models send.</summary>
        internal static bool ReadFlag(ToolCall call, string name)
        {
            if (call.Arguments == null || !call.Arguments.TryGetValue(name, out object? raw) || raw == null)
                return false;
            switch (raw)
            {
                case bool b:
                    return b;
                case JsonElement { ValueKind: JsonValueKind.True }:
                    return true;
                case JsonElement { ValueKind: JsonValueKind.False }:
                    return false;
                case JsonElement { ValueKind: JsonValueKind.Number } number when number.TryGetInt64(out long n):
                    return n != 0;
                case long l:
                    return l != 0;
                case int i:
                    return i != 0;
            }
            string text = (SkillTools.ReadString(call, name) ?? string.Empty).Trim();
            return text.Equals("true", StringComparison.OrdinalIgnoreCase)
                || text.Equals("yes", StringComparison.OrdinalIgnoreCase)
                || text.Equals("1", StringComparison.Ordinal);
        }

        /// <summary>
        /// Agent ids from whichever argument the model used, in whatever shape: a JSON
        /// array, an array encoded in a string, or ids separated by commas or spaces.
        /// Null when no such argument was given at all.
        /// </summary>
        internal static List<string>? ReadIds(ToolCall call, params string[] names)
        {
            foreach (string name in names)
            {
                if (call.Arguments == null || !call.Arguments.TryGetValue(name, out object? raw) || raw == null)
                    continue;

                var pieces = new List<string>();
                switch (raw)
                {
                    case JsonElement { ValueKind: JsonValueKind.Array } array:
                        foreach (JsonElement item in array.EnumerateArray())
                            pieces.Add(item.ValueKind == JsonValueKind.String ? item.GetString() ?? string.Empty : item.GetRawText());
                        break;
                    case string text:
                        pieces.Add(text);
                        break;
                    case JsonElement element:
                        pieces.Add(element.ValueKind == JsonValueKind.String ? element.GetString() ?? string.Empty : element.GetRawText());
                        break;
                    case IEnumerable enumerable:
                        foreach (object? item in enumerable)
                            pieces.Add(Convert.ToString(item, CultureInfo.InvariantCulture) ?? string.Empty);
                        break;
                    default:
                        pieces.Add(Convert.ToString(raw, CultureInfo.InvariantCulture) ?? string.Empty);
                        break;
                }

                var ids = new List<string>();
                foreach (string piece in pieces)
                {
                    foreach (string part in piece.Split(
                        new[] { ',', ' ', '\t', '\n', '\r', ';', '[', ']', '"', '\'' },
                        StringSplitOptions.RemoveEmptyEntries))
                    {
                        string id = part.Trim();
                        if (id.Length > 0 && !ids.Contains(id, StringComparer.Ordinal))
                            ids.Add(id);
                    }
                }
                return ids;
            }
            return null;
        }

        /// <summary>
        /// The requested wait in milliseconds, clamped to [<see cref="MinWaitMs"/>,
        /// <see cref="MaxWaitMs"/>], with a note when the request was changed. Zero or a
        /// negative value is an error, as in Codex: it can only be a mistake.
        /// </summary>
        internal static bool TryReadTimeout(ToolCall call, out int timeoutMs, out string? note, out string? error)
        {
            note = null;
            error = null;
            timeoutMs = DefaultWaitMs;
            long? requested = SkillTools.ReadInt64(call, "timeout_ms");
            if (requested == null)
                return true;
            if (requested <= 0)
            {
                error = "timeout_ms must be greater than zero";
                return false;
            }
            long clamped = Math.Clamp(requested.Value, MinWaitMs, MaxWaitMs);
            timeoutMs = (int)clamped;
            if (clamped != requested.Value)
            {
                note = "Requested timeout of " + requested.Value.ToString(CultureInfo.InvariantCulture)
                     + " ms was clamped to " + clamped.ToString(CultureInfo.InvariantCulture) + " ms.";
            }
            return true;
        }
    }
}
