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
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorSharp.AgentHost.Agents
{
    /// <summary>
    /// One agent's handle on the turn's <see cref="SubAgentRuntime"/>: the top-level
    /// agent's, or a sub-agent's. It is what <see cref="SkillToolContext.Agents"/> holds,
    /// so the agent tools always know WHO is calling — which decides whose sub-agents a
    /// call can reach and how deep a spawn would nest.
    /// </summary>
    public sealed class SubAgentScope
    {
        private readonly SubAgentRuntime _runtime;

        internal SubAgentScope(SubAgentRuntime runtime, SubAgent? self)
        {
            _runtime = runtime;
            Self = self;
        }

        /// <summary>The agent this scope belongs to; null for the top-level agent.</summary>
        internal SubAgent? Self { get; }

        /// <summary>The agent's id, or <c>root</c> for the top-level agent.</summary>
        public string Name => Self?.Id ?? "root";

        /// <summary>0 for the top-level agent, 1 for its sub-agents, and so on.</summary>
        public int Depth => Self?.Depth ?? 0;

        /// <summary>The loop's live conversation, as last bound.</summary>
        internal List<ChatMessage>? Conversation { get; private set; }

        /// <summary>The tools that conversation is generated with.</summary>
        internal IReadOnlyList<ToolFunction>? Tools { get; private set; }

        /// <summary>
        /// Tell the runtime which conversation this agent's loop is running, and with which
        /// tools. A spawned agent copies its leading instructions (or all of it, when
        /// forked) and its tool list, so a loop must bind before it executes any tool call.
        /// The list is read only while one of this agent's own tool calls is executing, i.e.
        /// while the loop is not mutating it.
        /// </summary>
        public void Bind(List<ChatMessage> conversation, IReadOnlyList<ToolFunction>? tools)
        {
            ArgumentNullException.ThrowIfNull(conversation);
            Conversation = conversation;
            Tools = tools;
        }

        /// <summary>
        /// Answer one agent tool call.
        /// </summary>
        /// <param name="call">A call whose name is one of <see cref="SkillToolNames.AgentTools"/>.</param>
        /// <param name="onOutput">
        /// Live tap while the call blocks (<c>wait_agent</c>): one line per thing a sub-agent
        /// does, so a host can stream what its agents are up to. May be called from other threads.
        /// </param>
        /// <param name="cancellationToken">The caller's turn. Cancellation propagates rather than becoming a result.</param>
        /// <returns>The result to feed back. Mistakes are results, phrased so the model can act on them.</returns>
        public async Task<SkillToolResult> ExecuteAsync(
            ToolCall call, Action<string>? onOutput, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(call);
            try
            {
                return call.Name switch
                {
                    SkillToolNames.SpawnAgent => _runtime.Spawn(this, call),
                    SkillToolNames.SendInput => _runtime.SendInput(this, call),
                    SkillToolNames.WaitAgent => await _runtime.WaitAsync(this, call, onOutput, cancellationToken).ConfigureAwait(false),
                    SkillToolNames.CloseAgent => _runtime.Close(this, call),
                    SkillToolNames.ListAgents => _runtime.List(this),
                    _ => SkillToolResult.Failure($"'{call.Name}' is not a tool this host answers."),
                };
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex) when (ex is not OutOfMemoryException and not OperationCanceledException)
            {
                return SkillToolResult.Failure($"The {call.Name} call failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Finished sub-agents' answers this agent has not been given yet, and messages its
        /// own parent sent while it was working. Taking them marks them delivered.
        /// </summary>
        public SubAgentDeliveries TakePendingDeliveries() => _runtime.TakePendingDeliveries(this);

        /// <summary>
        /// True while one of this agent's sub-agents is still working, or has an answer
        /// this agent has not been given.
        /// </summary>
        public bool HasOutstandingWork => _runtime.HasOutstandingWork(this);

        /// <summary>
        /// Wait (up to <paramref name="maxWait"/>) for this agent's working sub-agents, stop
        /// any still working after that, and return the message that hands every
        /// undelivered answer over — or null when there was nothing outstanding.
        /// </summary>
        public Task<string?> CollectOutstandingAsync(
            TimeSpan maxWait, Action<string>? onActivity, CancellationToken cancellationToken) =>
            _runtime.CollectOutstandingAsync(this, maxWait, onActivity, cancellationToken);

        /// <summary>Stop every sub-agent of this agent that is still working. Returns their ids.</summary>
        public IReadOnlyList<string> StopOutstanding() => _runtime.StopOutstanding(this);

        /// <summary>
        /// Finished sub-agents' results this agent was never given, as plain text for the
        /// user, marking them delivered; null when there are none.
        /// </summary>
        public string? TakeUndeliveredResultsText(out IReadOnlyList<string> ids) =>
            _runtime.TakeUndeliveredResultsText(this, out ids);
    }

    /// <summary>
    /// The conversation-side half of sub-agents, shared by every tool loop so they cannot
    /// drift: where delivered answers go, and what the end-of-turn guard says.
    /// </summary>
    public static class SubAgentConversation
    {
        /// <summary>
        /// Longest a loop waits, at the end of a turn, for sub-agents its model forgot to
        /// wait for. The agents are bounded by their own round budgets; this only guards
        /// against one that is stuck.
        /// </summary>
        public static readonly TimeSpan EndOfTurnWait = TimeSpan.FromMilliseconds(SubAgentTools.MaxWaitMs);

        /// <summary>
        /// After a round's tool results were appended: fold finished sub-agents' answers
        /// into the last of those results, and add any message the agent's parent sent as a
        /// user turn. Returns true when anything was added.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Folded into a tool RESULT rather than sent as a user turn of their own, which is
        /// what Codex does. A user turn says "the user is speaking"; a tool result says
        /// "here is information about the work in progress", which is what a sub-agent's
        /// answer is — and the parent's round had tool calls, so a result is always there
        /// to carry it.
        /// </para>
        /// <para>
        /// Appending at the end of the conversation never costs KV reuse: everything before
        /// it is exactly what the previous round's cache holds.
        /// </para>
        /// </remarks>
        public static bool AppendDeliveries(List<ChatMessage> working, SubAgentScope? scope)
        {
            ArgumentNullException.ThrowIfNull(working);
            if (scope == null)
                return false;

            SubAgentDeliveries deliveries = scope.TakePendingDeliveries();
            if (deliveries.IsEmpty)
                return false;

            if (deliveries.Notification != null)
            {
                ChatMessage? result = LastRoundResult(working);
                if (result != null)
                    result.Content = (result.Content ?? string.Empty).TrimEnd() + "\n\n" + deliveries.Notification;
                else
                    working.Add(new ChatMessage { Role = "user", Content = deliveries.Notification });
            }

            foreach (string message in deliveries.Messages)
                working.Add(new ChatMessage { Role = "user", Content = message });
            return true;
        }

        /// <summary>
        /// A turn that must END with sub-agents outstanding and no round left to hand their
        /// results to the model: stop the ones still working, take the results of the ones
        /// that finished but were never given to it, and return what the USER must be told
        /// about both — or null when nothing was outstanding.
        /// </summary>
        /// <remarks>
        /// The two cases are different facts and the note says which is which: a stopped
        /// agent's work is missing, while a finished one's work exists and is shown here,
        /// just not used by the answer above it. Dropping the second silently was the
        /// original behaviour, and it lost a result that had already been paid for.
        /// </remarks>
        public static string? EndWithoutRound(SubAgentScope? scope)
        {
            if (scope is not { HasOutstandingWork: true })
                return null;

            IReadOnlyList<string> stopped = scope.StopOutstanding();
            string? late = scope.TakeUndeliveredResultsText(out IReadOnlyList<string> lateIds);

            var parts = new List<string>();
            if (stopped.Count > 0)
            {
                bool one = stopped.Count == 1;
                parts.Add("(Sub-agent" + (one ? " " : "s ") + string.Join(", ", stopped)
                    + (one ? " was" : " were") + " still working when this turn ran out of rounds and "
                    + (one ? "was" : "were") + " stopped; this answer does not include "
                    + (one ? "its" : "their") + " results.)");
            }
            if (late != null)
            {
                bool one = lateIds.Count == 1;
                parts.Add("(Sub-agent" + (one ? " " : "s ") + string.Join(", ", lateIds)
                    + " finished after this answer was written, so this answer does not use "
                    + (one ? "its" : "their") + " results. " + (one ? "It" : "They") + " reported:)\n" + late);
            }
            return parts.Count == 0 ? null : string.Join("\n\n", parts);
        }

        /// <summary>
        /// The last result of the latest round — a <c>tool</c> message, or the user turn a
        /// family without tool messages gets instead — or null when the conversation ends
        /// on the assistant.
        /// </summary>
        private static ChatMessage? LastRoundResult(List<ChatMessage> working)
        {
            for (int i = working.Count - 1; i >= 0; i--)
            {
                ChatMessage message = working[i];
                if (string.Equals(message.Role, "assistant", StringComparison.OrdinalIgnoreCase))
                    return null;
                if (string.Equals(message.Role, "tool", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(message.Role, "user", StringComparison.OrdinalIgnoreCase))
                {
                    return message;
                }
            }
            return null;
        }
    }
}
