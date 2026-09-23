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
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.Agents
{
    /// <summary>Where a sub-agent is in its life.</summary>
    public enum SubAgentStatus
    {
        /// <summary>Spawned, or given a follow-up, and not yet generating.</summary>
        Starting,

        /// <summary>Running its tool loop.</summary>
        Running,

        /// <summary>Finished its latest task with a final answer.</summary>
        Completed,

        /// <summary>Its latest task ended in an error instead of an answer.</summary>
        Errored,

        /// <summary>Closed by its parent, by the slot limit, or by the end of the turn. Terminal.</summary>
        Closed,
    }

    /// <summary>What the host is told about one agent it has to start generating for.</summary>
    public sealed class SubAgentLaunch
    {
        /// <summary>The agent's id, e.g. <c>agent_1</c>.</summary>
        public string AgentId { get; init; } = string.Empty;

        /// <summary>The id of the agent that spawned it, or null when the turn's own model did.</summary>
        public string? ParentId { get; init; }

        /// <summary>1 for a sub-agent of the parent, 2 for one of those agents' own, and so on.</summary>
        public int Depth { get; init; }

        /// <summary>
        /// True when the agent was started with a copy of its parent's conversation. A host
        /// that keeps per-conversation KV should run such an agent in the parent's cache
        /// scope, so the copied prefix is reused rather than prefilled again; a fresh agent
        /// belongs in a scope of its own, sharing only the public prefix.
        /// </summary>
        public bool ForkContext { get; init; }
    }

    /// <summary>
    /// What a host has to supply for sub-agents to run: a way to generate for each one.
    ///
    /// <para>
    /// The same split as <see cref="SkillTurnGenerator"/> itself: the runtime owns the
    /// agents, their conversations and their tool loops, and the host owns the one thing
    /// that differs between hosts — how a list of messages becomes a generation. The server
    /// submits each round to the continuous-batching engine, which is what lets several
    /// agents decode at once on one loaded model.
    /// </para>
    /// </summary>
    public sealed class SubAgentHostBinding
    {
        /// <summary>
        /// Called once per agent, when it is spawned. The generator it returns serves every
        /// round of every turn that agent runs, so a host can pin the agent to one cache
        /// scope and have its follow-up turns continue its own KV.
        /// </summary>
        public required Func<SubAgentLaunch, SkillTurnGenerator> CreateGenerator { get; init; }

        /// <summary>
        /// Bounds for each agent's tool loop: rounds, calls per round, how tool results are
        /// rendered, and which tools belong to the caller. Null uses the loop defaults.
        /// </summary>
        public SkillAgentLoopOptions? LoopOptions { get; init; }
    }

    /// <summary>
    /// What a loop should add to its conversation before its next generation: finished
    /// sub-agents' results, and messages its own parent sent it while it was working.
    /// </summary>
    public readonly record struct SubAgentDeliveries(string? Notification, IReadOnlyList<string> Messages)
    {
        /// <summary>Nothing to add.</summary>
        public static SubAgentDeliveries None { get; } = new(null, Array.Empty<string>());

        /// <summary>True when there is nothing to add.</summary>
        public bool IsEmpty => Notification == null && Messages.Count == 0;
    }

    /// <summary>
    /// The sub-agents of one turn: who they are, what they are doing, and the tools that
    /// start, steer, collect and stop them.
    ///
    /// <para>
    /// <b>One runtime per turn of the top-level agent.</b> Every agent it starts — and
    /// every agent those start, when nesting is allowed — lives here, shares the
    /// open-agent limit, and is stopped when the runtime is disposed at the end of the
    /// turn. A stateless HTTP request has nowhere to keep an agent between requests, and
    /// an agent still generating after its answer was sent would be spending the GPU on a
    /// result nobody can receive.
    /// </para>
    /// <para>
    /// <b>Delivery is exactly once.</b> A finished agent's answer reaches the agent that
    /// started it either as the result of <c>wait_agent</c> or — if that agent never waits
    /// — inside a <c>&lt;subagent_notification&gt;</c> appended to the next tool result it
    /// receives. Codex's first surface does both and its tool text admits the answer
    /// arrives twice; its second stops <c>wait_agent</c> returning content at all. Neither
    /// suits a model with an 8k context window, which can afford the answer once and
    /// cannot afford to be made to fetch it.
    /// </para>
    /// <para>
    /// <b>Agents cannot address each other</b>, only their own sub-agents. Codex's second
    /// surface lets any agent message any other; with small local models that becomes an
    /// uncontrolled chat between agents, and the tree shape is what makes "who is waiting
    /// for whom" answerable at all.
    /// </para>
    /// </summary>
    public sealed class SubAgentRuntime : IDisposable
    {
        /// <summary>
        /// Longest final answer handed back verbatim. Past it the answer is cut and the
        /// parent is told how to get the rest, because the point of a sub-agent is to keep
        /// its working out of the parent's context and a 40 KB answer defeats that.
        /// </summary>
        public const int MaxResultChars = 8000;

        /// <summary>How long disposal waits for stopped agents to wind down.</summary>
        private static readonly TimeSpan DisposeGrace = TimeSpan.FromSeconds(5);

        private readonly SubAgentOptions _options;
        private readonly SubAgentHostBinding _binding;
        private readonly SkillToolContext _baseContext;
        private readonly ILogger? _logger;
        private readonly CancellationTokenSource _turnCts;
        private readonly object _gate = new();
        private readonly Dictionary<string, SubAgent> _agents = new(StringComparer.Ordinal);
        private readonly List<Action<string>> _listeners = new();
        private TaskCompletionSource _changed = NewSignal();
        private int _nextId;
        private long _finishCounter;
        private bool _disposed;

        /// <summary>
        /// Makes this turn's workspace lanes distinct from every other turn's: agent ids
        /// restart at agent_1 each turn, while the workspace — and its lanes — lives as
        /// long as the conversation.
        /// </summary>
        private readonly string _laneTag = Guid.NewGuid().ToString("N").Substring(0, 8);

        /// <summary>Create the runtime for one turn.</summary>
        /// <param name="options">Limits. Copied.</param>
        /// <param name="binding">How to generate for an agent.</param>
        /// <param name="baseContext">
        /// The turn's tool context. Every agent gets a copy of it, so sub-agents reach the
        /// same skills, the same code runner and the same workspace as their parent.
        /// </param>
        /// <param name="logger">Optional.</param>
        /// <param name="turnToken">Cancelled when the turn is abandoned; every agent stops with it.</param>
        public SubAgentRuntime(
            SubAgentOptions options,
            SubAgentHostBinding binding,
            SkillToolContext baseContext,
            ILogger? logger = null,
            CancellationToken turnToken = default)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(binding);
            ArgumentNullException.ThrowIfNull(baseContext);
            _options = options.Clone();
            _binding = binding;
            _baseContext = baseContext;
            _logger = logger;
            _turnCts = CancellationTokenSource.CreateLinkedTokenSource(turnToken);
            Root = new SubAgentScope(this, null);
            RootContext = baseContext.WithAgents(Root);
        }

        /// <summary>The top-level agent's handle.</summary>
        public SubAgentScope Root { get; }

        /// <summary>The turn's tool context with <see cref="Root"/> attached: what the top-level loop should run tools with.</summary>
        public SkillToolContext RootContext { get; }

        /// <summary>The limits in force.</summary>
        public SubAgentOptions Options => _options;

        /// <summary>How many agents this turn has started, closed ones included.</summary>
        public int SpawnedCount
        {
            get { lock (_gate) return _agents.Count; }
        }

        /// <summary>A consistent snapshot of every agent, for logging and tests.</summary>
        public IReadOnlyList<SubAgentSnapshot> Snapshot()
        {
            lock (_gate)
            {
                return _agents.Values
                    .OrderBy(a => a.Number)
                    .Select(a => a.ToSnapshot())
                    .ToList();
            }
        }

        // ---- tools ---------------------------------------------------------------

        internal SkillToolResult Spawn(SubAgentScope caller, ToolCall call)
        {
            string? message = SubAgentTools.ReadText(call, "message", "task", "prompt", "input", "instructions");
            if (message == null)
                return SkillToolResult.Failure("Empty message can't be sent to an agent. Put the agent's task in 'message'.");

            int depth = caller.Depth + 1;
            if (depth > _options.MaxDepth)
                return SkillToolResult.Failure("Agent depth limit reached. Solve the task yourself.");

            List<ChatMessage>? conversation = caller.Conversation;
            IReadOnlyList<ToolFunction>? tools = caller.Tools;
            if (conversation == null)
            {
                // A host that attached the runtime without binding the loop's
                // conversation. Not the model's fault and not something it can fix.
                return SkillToolResult.Failure(
                    "sub-agents are not available in this conversation. Do the task yourself.");
            }

            bool fork = SubAgentTools.ReadFlag(call, "fork_context");
            SubAgent agent;
            string? evicted = null;
            var evictedSources = new List<CancellationTokenSource>();
            lock (_gate)
            {
                if (_disposed || _turnCts.IsCancellationRequested)
                    return SkillToolResult.Failure("this turn is ending, so no new agent can start.");

                int open = _agents.Values.Count(a => a.IsOpen);
                if (open >= _options.MaxThreads)
                {
                    // Codex's second surface, not its first: an idle agent whose answer
                    // was already delivered is closed to make room, so a model that never
                    // learned to call close_agent is not stopped cold by agents it has
                    // finished with. Only agents still working, or holding an answer the
                    // parent has not seen, keep their slot.
                    // Only from the caller's own subtree: an agent may close only agents it
                    // started (directly or through its own sub-agents). Evicting a sibling
                    // would close an agent its real parent still addresses, and tell the
                    // wrong agent about it.
                    SubAgent? victim = _agents.Values
                        .Where(a => a.IsOpen && !a.IsWorking && a.Delivered && !HasOpenChildrenLocked(a)
                                    && IsDescendantLocked(a, caller.Self))
                        .OrderBy(a => a.FinishOrder)
                        .FirstOrDefault();
                    if (victim == null)
                    {
                        _logger?.LogWarning(LogEventIds.SkillToolInvoked,
                            "agents.limit caller={Caller} open={Open} max={Max}",
                            caller.Name, open, _options.MaxThreads);
                        return SkillToolResult.Failure(
                            "agent thread limit reached: " + open.ToString(CultureInfo.InvariantCulture)
                            + " agents are already open, the most this host runs at once. Wait for one to finish "
                            + "with " + SkillToolNames.WaitAgent + ", close one you no longer need with "
                            + SkillToolNames.CloseAgent + ", or do the task yourself.");
                    }
                    CloseLocked(victim, evictedSources);
                    evicted = victim.Id;
                }

                int number = ++_nextId;
                agent = new SubAgent(
                    number, "agent_" + number.ToString(CultureInfo.InvariantCulture), caller.Self, depth,
                    message, fork, _turnCts.Token);
                agent.Scope = new SubAgentScope(this, agent);
                agent.Tools = tools;
                bool forked = false;
                agent.History = fork
                    ? BuildForkedHistory(conversation, call, agent, _binding.LoopOptions?.ToolResultsAreRendered ?? true, out forked)
                    : BuildFreshHistory(conversation, agent);
                // A fork whose spawning turn is not in the conversation started fresh.
                // Say so to the model and to the host: ForkContext would otherwise run a
                // fresh agent in the parent's cache scope, and the reply would claim
                // context the agent does not have.
                fork = forked;
                agent.Forked = forked;
                _agents.Add(agent.Id, agent);
            }
            foreach (CancellationTokenSource source in evictedSources)
                source.Cancel();

            try
            {
                agent.Generator = _binding.CreateGenerator(new SubAgentLaunch
                {
                    AgentId = agent.Id,
                    ParentId = caller.Self?.Id,
                    Depth = depth,
                    ForkContext = fork,
                });
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                lock (_gate)
                    agent.Status = SubAgentStatus.Closed;
                Signal();
                return SkillToolResult.Failure("the agent could not be started: " + ex.Message);
            }

            _logger?.LogInformation(LogEventIds.SkillToolInvoked,
                "agents.spawn id={Id} parent={Parent} depth={Depth} fork={Fork} evicted={Evicted} task={Task}",
                agent.Id, caller.Name, depth, fork, evicted ?? "-", Abbreviate(message, 120));
            Start(agent);

            var sb = new StringBuilder();
            sb.Append(agent.Id).Append(" started");
            if (fork)
                sb.Append(" with a copy of this conversation");
            sb.Append(". It is working in the background; its final answer will be delivered to you when it "
                    + "finishes. Continue with other work, or call ").Append(SkillToolNames.WaitAgent)
              .Append(" when your next step needs its result.");
            if (evicted != null)
                sb.Append(" (").Append(evicted).Append(", which had finished and already reported, was closed to make room.)");
            return SkillToolResult.Success(sb.ToString());
        }

        internal SkillToolResult SendInput(SubAgentScope caller, ToolCall call)
        {
            string? id = SubAgentTools.ReadText(call, "target", "agent_id", "id", "agent");
            if (id == null)
                return SkillToolResult.Failure(SkillToolNames.SendInput + " needs 'target': the id of the agent to message.");
            string? message = SubAgentTools.ReadText(call, "message", "input", "text", "task");
            if (message == null)
                return SkillToolResult.Failure("Empty message can't be sent to an agent");

            string? previousAnswer = null;
            SubAgent agent;
            bool restart = false;
            lock (_gate)
            {
                if (!TryResolveOwnedLocked(caller, id, out agent!, out string? error))
                    return SkillToolResult.Failure(error!);

                switch (agent.Status)
                {
                    case SubAgentStatus.Closed:
                        return SkillToolResult.Failure($"agent with id {agent.Id} is closed");

                    case SubAgentStatus.Starting:
                    case SubAgentStatus.Running:
                        agent.Inbox.Add(message);
                        return SkillToolResult.Success(
                            agent.Id + " is still working; it will read your message before its next step.");

                    default:
                        if (_disposed || _turnCts.IsCancellationRequested)
                            return SkillToolResult.Failure("this turn is ending, so the agent cannot start again.");
                        // An answer the parent never saw would be overwritten by the next
                        // one. Hand it over here instead of losing it.
                        if (!agent.Delivered && agent.Result != null)
                            previousAnswer = DescribeResultLocked(agent);
                        agent.History.Add(new ChatMessage { Role = "user", Content = FollowUpMessage(message) });
                        agent.Status = SubAgentStatus.Starting;
                        agent.Result = null;
                        agent.Delivered = true;
                        restart = true;
                        break;
                }
            }

            Signal();
            if (restart)
                Start(agent);

            string reply = agent.Id + " is working on your message, with its earlier context kept.";
            return SkillToolResult.Success(previousAnswer == null
                ? reply
                : previousAnswer + "\n\n" + reply);
        }

        internal async Task<SkillToolResult> WaitAsync(
            SubAgentScope caller, ToolCall call, Action<string>? onOutput, CancellationToken cancellationToken)
        {
            if (!SubAgentTools.TryReadTimeout(call, out int timeoutMs, out string? clampNote, out string? timeoutError))
                return SkillToolResult.Failure(timeoutError!);

            List<string>? requested = SubAgentTools.ReadIds(call, "targets", "target", "agent_ids", "ids", "agents", "agent_id");
            bool any = requested == null || requested.Count == 0
                || (requested.Count == 1 && requested[0].Trim().ToLowerInvariant() is "all" or "any" or "*");

            List<SubAgent> awaited;
            lock (_gate)
            {
                if (any)
                {
                    awaited = ChildrenLocked(caller).Where(a => a.IsOpen).ToList();
                    if (awaited.Count == 0)
                    {
                        return SkillToolResult.Success(ChildrenLocked(caller).Count == 0
                            ? "You have no sub-agents to wait for. Start one with " + SkillToolNames.SpawnAgent + "."
                            : "None of your sub-agents is open, so there is nothing to wait for.");
                    }
                }
                else
                {
                    awaited = new List<SubAgent>();
                    foreach (string id in requested!)
                    {
                        if (!TryResolveOwnedLocked(caller, id, out SubAgent? agent, out string? error))
                            return SkillToolResult.Failure(error!);
                        if (!awaited.Contains(agent!))
                            awaited.Add(agent!);
                    }
                }
            }

            long started = Stopwatch.GetTimestamp();
            using var waitCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _turnCts.Token);
            Task timeout = Task.Delay(timeoutMs, waitCts.Token);
            AddListener(onOutput);
            try
            {
                while (true)
                {
                    Task changed;
                    lock (_gate)
                    {
                        List<SubAgent> ready = awaited.Where(a => a.HasUndeliveredResult).ToList();
                        if (ready.Count > 0)
                            return SkillToolResult.Success(WithNote(DescribeReadyLocked(ready, awaited), clampNote));

                        if (!awaited.Any(a => a.IsWorking))
                            return SkillToolResult.Success(WithNote(DescribeNothingToWaitForLocked(awaited), clampNote));

                        changed = _changed.Task;
                    }

                    Task first = await Task.WhenAny(changed, timeout).ConfigureAwait(false);
                    cancellationToken.ThrowIfCancellationRequested();
                    if (_turnCts.IsCancellationRequested)
                        return SkillToolResult.Failure("this turn is ending; the agents were stopped.");
                    if (first == timeout)
                    {
                        lock (_gate)
                            return SkillToolResult.Success(WithNote(DescribeTimeoutLocked(awaited, started), clampNote));
                    }
                }
            }
            finally
            {
                RemoveListener(onOutput);
                waitCts.Cancel();
            }
        }

        internal SkillToolResult Close(SubAgentScope caller, ToolCall call)
        {
            string? id = SubAgentTools.ReadText(call, "target", "agent_id", "id", "agent");
            if (id == null)
                return SkillToolResult.Failure(SkillToolNames.CloseAgent + " needs 'target': the id of the agent to close.");

            SubAgentStatus previous;
            var stopping = new List<CancellationTokenSource>();
            string closedId;
            int cascaded;
            string? unseenAnswer = null;
            lock (_gate)
            {
                if (!TryResolveOwnedLocked(caller, id, out SubAgent? agent, out string? error))
                    return SkillToolResult.Failure(error!);
                closedId = agent!.Id;
                previous = agent.Status;
                if (previous == SubAgentStatus.Closed)
                    return SkillToolResult.Success(closedId + " was already closed.");
                // An answer the caller has not seen would vanish with the agent. Closing
                // means "stop working", not "discard what you already produced".
                if (agent.HasUndeliveredResult)
                    unseenAnswer = DescribeResultLocked(agent);
                CloseLocked(agent, stopping);
                cascaded = stopping.Count - 1;
            }

            foreach (CancellationTokenSource cts in stopping)
                cts.Cancel();
            Signal();
            _logger?.LogInformation(LogEventIds.SkillToolInvoked,
                "agents.close id={Id} previous={Previous} cascaded={Cascaded}", closedId, previous, cascaded);

            string reply = closedId + " closed; it was " + Describe(previous) + ".";
            if (cascaded > 0)
            {
                reply += " " + cascaded.ToString(CultureInfo.InvariantCulture)
                       + (cascaded == 1 ? " agent it had started was" : " agents it had started were") + " closed too.";
            }
            return SkillToolResult.Success(unseenAnswer == null ? reply : unseenAnswer + "\n\n" + reply);
        }

        internal SkillToolResult List(SubAgentScope caller)
        {
            lock (_gate)
            {
                List<SubAgent> children = ChildrenLocked(caller);
                if (children.Count == 0)
                    return SkillToolResult.Success("You have no sub-agents. Start one with " + SkillToolNames.SpawnAgent + ".");

                var sb = new StringBuilder();
                sb.Append(children.Count.ToString(CultureInfo.InvariantCulture))
                  .Append(children.Count == 1 ? " sub-agent:\n" : " sub-agents:\n");
                foreach (SubAgent agent in children)
                {
                    sb.Append(agent.Id).Append(": ").Append(Describe(agent.Status));
                    if (agent.IsWorking)
                        sb.Append(" for ").Append(FormatElapsed(agent.StartedAt));
                    else if (agent.HasUndeliveredResult)
                        sb.Append(" (answer not yet collected: call ").Append(SkillToolNames.WaitAgent).Append(')');
                    else if (agent.Status is SubAgentStatus.Completed or SubAgentStatus.Errored)
                        sb.Append(" (answer already delivered)");
                    sb.Append(" - task: ").Append(Abbreviate(agent.Task, 160)).Append('\n');
                }
                return SkillToolResult.Success(sb.ToString().TrimEnd());
            }
        }

        // ---- loop integration ----------------------------------------------------

        /// <param name="caller">Whose deliveries.</param>
        /// <param name="includeMessages">
        /// False takes only finished sub-agents' results and leaves the messages the
        /// caller's own parent sent it in its inbox, for whoever can put them in the
        /// conversation as user turns (the next round boundary, or the agent's run loop
        /// once its turn ends). The end-of-turn handover is one message about results; a
        /// parent's message drained into it would be dropped.
        /// </param>
        internal SubAgentDeliveries TakePendingDeliveries(SubAgentScope caller, bool includeMessages = true)
        {
            lock (_gate)
            {
                List<SubAgent> ready = ChildrenLocked(caller).Where(a => a.HasUndeliveredResult).ToList();
                string? notification = null;
                if (ready.Count > 0)
                {
                    var sb = new StringBuilder("<subagent_notification>\n");
                    foreach (SubAgent agent in ready)
                        sb.Append(DescribeResultLocked(agent)).Append('\n');
                    sb.Append("</subagent_notification>");
                    notification = sb.ToString();
                    foreach (SubAgent agent in ready)
                        _logger?.LogInformation(LogEventIds.SkillToolInvoked,
                            "agents.deliver id={Id} via=notification", agent.Id);
                }

                IReadOnlyList<string> messages = Array.Empty<string>();
                if (includeMessages && caller.Self is { Inbox.Count: > 0 } self)
                {
                    messages = self.Inbox.Select(FollowUpMessage).ToList();
                    self.Inbox.Clear();
                }

                return notification == null && messages.Count == 0
                    ? SubAgentDeliveries.None
                    : new SubAgentDeliveries(notification, messages);
            }
        }

        /// <summary>
        /// The results of the caller's finished sub-agents it has not been given, as plain
        /// text for a USER to read (no notification tags), marking them delivered; null when
        /// there are none. For a turn that has no round left to hand them to the model.
        /// </summary>
        internal string? TakeUndeliveredResultsText(SubAgentScope caller, out IReadOnlyList<string> ids)
        {
            lock (_gate)
            {
                List<SubAgent> ready = ChildrenLocked(caller).Where(a => a.HasUndeliveredResult).ToList();
                ids = ready.Select(a => a.Id).ToList();
                if (ready.Count == 0)
                    return null;
                foreach (SubAgent agent in ready)
                    _logger?.LogInformation(LogEventIds.SkillToolInvoked, "agents.deliver id={Id} via=answer-note", agent.Id);
                return string.Join("\n\n", ready.Select(DescribeResultLocked));
            }
        }

        internal bool HasOutstandingWork(SubAgentScope caller)
        {
            lock (_gate)
                return ChildrenLocked(caller).Any(a => a.IsOpen && (a.IsWorking || a.HasUndeliveredResult));
        }

        internal async Task<string?> CollectOutstandingAsync(
            SubAgentScope caller, TimeSpan maxWait, Action<string>? onActivity, CancellationToken cancellationToken)
        {
            using var waitCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _turnCts.Token);
            Task timeout = Task.Delay(maxWait, waitCts.Token);
            AddListener(onActivity);
            try
            {
                while (true)
                {
                    Task changed;
                    lock (_gate)
                    {
                        if (!ChildrenLocked(caller).Any(a => a.IsOpen && a.IsWorking))
                            break;
                        changed = _changed.Task;
                    }
                    Task first = await Task.WhenAny(changed, timeout).ConfigureAwait(false);
                    cancellationToken.ThrowIfCancellationRequested();
                    if (first == timeout || _turnCts.IsCancellationRequested)
                        break;
                }
            }
            finally
            {
                RemoveListener(onActivity);
                waitCts.Cancel();
            }

            IReadOnlyList<string> stopped = StopOutstanding(caller);
            // Results only: a message this agent's own parent sent meanwhile stays queued,
            // and RunAsync (or the next round boundary) turns it into a user turn.
            SubAgentDeliveries deliveries = TakePendingDeliveries(caller, includeMessages: false);
            if (deliveries.Notification == null && stopped.Count == 0)
                return null;

            var sb = new StringBuilder();
            sb.Append("Before you finish: sub-agents you started had not reported back to you.");
            if (deliveries.Notification != null)
                sb.Append(" They have finished, and their results are below.\n\n").Append(deliveries.Notification);
            if (stopped.Count > 0)
            {
                sb.Append("\n\n").Append(string.Join(", ", stopped))
                  .Append(stopped.Count == 1 ? " was" : " were")
                  .Append(" still working after ").Append(FormatDuration(maxWait))
                  .Append(" and " + (stopped.Count == 1 ? "has" : "have") + " been stopped; say in your answer what is missing because of that.");
            }
            sb.Append("\n\nNow write your final answer to the user, using these results. Do not redo work the sub-agents already did.");
            return sb.ToString();
        }

        internal IReadOnlyList<string> StopOutstanding(SubAgentScope caller)
        {
            var stopping = new List<CancellationTokenSource>();
            var ids = new List<string>();
            lock (_gate)
            {
                foreach (SubAgent agent in ChildrenLocked(caller))
                {
                    if (!agent.IsOpen || !agent.IsWorking)
                        continue;
                    ids.Add(agent.Id);
                    CloseLocked(agent, stopping);
                }
            }
            foreach (CancellationTokenSource cts in stopping)
                cts.Cancel();
            if (ids.Count > 0)
            {
                Signal();
                _logger?.LogWarning(LogEventIds.SkillToolInvoked,
                    "agents.stopped ids={Ids} caller={Caller} - still working when the turn had to end",
                    string.Join(",", ids), caller.Name);
            }
            return ids;
        }

        // ---- running an agent ----------------------------------------------------

        private void Start(SubAgent agent)
        {
            // Task.Run, not a bare call: the loop's first await would otherwise run the
            // agent's first generation setup on the parent's tool thread.
            agent.Worker = Task.Run(() => RunAsync(agent));
        }

        private async Task RunAsync(SubAgent agent)
        {
            CancellationToken token = agent.Cts.Token;
            SkillAgentLoopOptions loopOptions = ChildLoopOptions(agent);

            try
            {
                // Inside the try: a workspace released under the turn must fail this
                // agent visibly, not leave it "running" with a faulted worker forever.
                SkillToolContext context = _baseContext.ForSubAgent(agent.Scope, agent.Id + "-" + _laneTag);
                while (true)
                {
                    token.ThrowIfCancellationRequested();
                    List<ChatMessage> input;
                    lock (_gate)
                    {
                        if (agent.Status == SubAgentStatus.Closed)
                            return;
                        agent.Status = SubAgentStatus.Running;
                        agent.Turns++;
                        agent.StartedAt = Stopwatch.GetTimestamp();
                        agent.TurnRounds = 0;
                        agent.TurnInvocations.Clear();
                        input = agent.History;
                    }
                    Signal();
                    Report(agent, agent.Turns == 1 ? "started" : "working on a follow-up");

                    SkillLoopResult result = await SkillAgentLoop.RunAsync(
                        input, agent.Tools?.ToList(), context, agent.Generator!, loopOptions, token).ConfigureAwait(false);
                    result = await AnswerClientCallsAsync(agent, result, context, loopOptions, token).ConfigureAwait(false);
                    result = await CorrectEmptyTurnAsync(agent, result, context, loopOptions, token).ConfigureAwait(false);

                    List<ChatMessage> history = result.Messages;
                    ParsedOutput? parsed = result.Output.Parsed;
                    history.Add(new ChatMessage
                    {
                        Role = "assistant",
                        Content = parsed?.Content ?? string.Empty,
                        Thinking = string.IsNullOrEmpty(parsed?.Thinking) ? null : parsed!.Thinking,
                        RawOutputTokens = result.Output.RawTokens != null ? new List<int>(result.Output.RawTokens) : null,
                        RawPromptTrailingWhitespace = result.Output.RawPromptTrailingWhitespace,
                        RawGenerationSuffix = result.Output.RawGenerationSuffix,
                    });

                    bool again = false;
                    lock (_gate)
                    {
                        agent.History = history;
                        agent.Rounds += result.Rounds;
                        agent.ToolCalls += result.Invocations.Count;
                        agent.TurnRounds += result.Rounds;
                        agent.TurnInvocations.AddRange(result.Invocations);
                        if (agent.Status == SubAgentStatus.Closed)
                            return;
                        if (agent.Inbox.Count > 0)
                        {
                            // A message arrived while the last round was generating. It
                            // was sent expecting an answer that accounts for it, so the
                            // agent carries on rather than reporting an answer that doesn't.
                            foreach (string message in agent.Inbox)
                                history.Add(new ChatMessage { Role = "user", Content = FollowUpMessage(message) });
                            agent.Inbox.Clear();
                            again = true;
                        }
                        else
                        {
                            agent.Status = SubAgentStatus.Completed;
                            agent.Result = FinalAnswer(parsed);
                            agent.Delivered = false;
                            agent.FinishedAt = Stopwatch.GetTimestamp();
                            agent.FinishOrder = ++_finishCounter;
                        }
                    }

                    if (again)
                        continue;

                    // Report before Signal: the signal is what wakes a wait_agent, which
                    // then unregisters its tap, so the other order lets the one line a
                    // waiting host most wants to show — this one — miss it.
                    Report(agent, "finished (" + FormatElapsed(agent.StartedAt) + ")");
                    Signal();
                    _logger?.LogInformation(LogEventIds.SkillToolInvoked,
                        "agents.finish id={Id} status=completed turns={Turns} rounds={Rounds} toolCalls={ToolCalls} ms={Ms} hitRoundLimit={Capped} answerChars={Chars}",
                        agent.Id, agent.Turns, agent.Rounds, agent.ToolCalls,
                        (long)Stopwatch.GetElapsedTime(agent.StartedAt).TotalMilliseconds,
                        result.HitRoundLimit, agent.Result?.Length ?? 0);
                    return;
                }
            }
            catch (OperationCanceledException) when (token.IsCancellationRequested)
            {
                lock (_gate)
                {
                    if (agent.Status != SubAgentStatus.Closed)
                        MarkClosedLocked(agent);
                }
                Signal();
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                lock (_gate)
                {
                    if (agent.Status == SubAgentStatus.Closed)
                        return;
                    agent.Status = SubAgentStatus.Errored;
                    agent.Result = ex.Message;
                    agent.Delivered = false;
                    agent.FinishedAt = Stopwatch.GetTimestamp();
                    agent.FinishOrder = ++_finishCounter;
                }
                Report(agent, "failed: " + Abbreviate(ex.Message, 200));
                Signal();
                _logger?.LogWarning(LogEventIds.SkillToolInvoked, ex,
                    "agents.finish id={Id} status=errored turns={Turns} error={Error}", agent.Id, agent.Turns, ex.Message);
            }
        }

        /// <summary>
        /// Answer calls a sub-agent made to one of the CALLER's own tools. Only the
        /// top-level agent's client can service those, and the sub-agent's turn cannot be
        /// handed to it, so the sub-agent is told so and carries on. Bounded: a model that
        /// keeps calling it anyway ends with the answer it has.
        /// </summary>
        private static async Task<SkillLoopResult> AnswerClientCallsAsync(
            SubAgent agent, SkillLoopResult result, SkillToolContext context,
            SkillAgentLoopOptions loopOptions, CancellationToken token)
        {
            for (int attempt = 0; attempt < 2 && result.PendingClientToolCalls.Count > 0; attempt++)
            {
                List<ChatMessage> messages = result.Messages;
                ChatMessage? owner = messages.LastOrDefault(m =>
                    m.Role == "assistant" && m.ToolCalls != null
                    && result.PendingClientToolCalls.Any(c => m.ToolCalls.Contains(c)));
                if (owner == null)
                {
                    ParsedOutput? parsed = result.Output.Parsed;
                    messages.Add(new ChatMessage
                    {
                        Role = "assistant",
                        Content = parsed?.Content ?? string.Empty,
                        Thinking = string.IsNullOrEmpty(parsed?.Thinking) ? null : parsed!.Thinking,
                        ToolCalls = new List<ToolCall>(parsed?.ToolCalls ?? result.PendingClientToolCalls.ToList()),
                        RawOutputTokens = result.Output.RawTokens != null ? new List<int>(result.Output.RawTokens) : null,
                        RawPromptTrailingWhitespace = result.Output.RawPromptTrailingWhitespace,
                        RawGenerationSuffix = result.Output.RawGenerationSuffix,
                    });
                }

                foreach (ToolCall call in result.PendingClientToolCalls)
                {
                    string refusal = "Error: '" + call.Name + "' belongs to the application the user is working in, "
                        + "and only the agent that started you can call it. Finish your task without it, or say in "
                        + "your final answer what that agent should call and why.";
                    messages.Add(loopOptions.ToolResultsAreRendered
                        ? new ChatMessage { Role = "tool", Content = refusal, ToolCallId = call.Id }
                        : new ChatMessage { Role = "user", Content = "Result of your " + call.Name + " call:\n\n" + refusal });
                }

                int rounds = result.Rounds;
                result = await SkillAgentLoop.RunAsync(
                    messages, agent.Tools?.ToList(), context, agent.Generator!, loopOptions, token).ConfigureAwait(false);
                result = result with { Rounds = result.Rounds + rounds };
            }
            return result;
        }

        /// <summary>
        /// One correction for the two clearest non-results a sub-agent produces, both in a
        /// turn that ran no tool at all: a task "done" by writing a patch envelope as its
        /// ANSWER, and an answer with nothing in it.
        ///
        /// <para>
        /// Observed on gemma-4-E2B: asked to write a file, the sub-agent's whole reply was
        /// <c>*** Begin Patch / *** Add File: a.txt / +alpha / *** End Patch</c> as text, in
        /// one round — nothing was applied, and its parent reported the file as written.
        /// For a sub-agent that combination is unambiguous: its answer is read by another
        /// agent, never shown to a user who might have asked to SEE a patch, and a turn that
        /// ran no tools cannot have made the change it describes. So it is told so, once,
        /// and given the round to make the call. Anything subtler is left to the record
        /// <see cref="DescribeWorkLocked"/> hands the parent.
        /// </para>
        /// <para>
        /// The empty answer is the same argument: observed on gemma-4-E2B, a forked agent
        /// ended its first turn on its first token, and the parent — told only that its
        /// agent had written nothing — spent two rounds starting another one.
        /// </para>
        /// </summary>
        private static async Task<SkillLoopResult> CorrectEmptyTurnAsync(
            SubAgent agent, SkillLoopResult result, SkillToolContext context,
            SkillAgentLoopOptions loopOptions, CancellationToken token)
        {
            string content = result.Output.Parsed?.Content ?? string.Empty;
            if (result.Invocations.Count > 0 || result.PendingClientToolCalls.Count > 0)
                return result;

            string? correction = null;
            if (content.Contains("*** Begin Patch", StringComparison.Ordinal)
                && agent.Tools?.Any(t => string.Equals(t?.Name, SkillToolNames.ApplyPatch, StringComparison.Ordinal)) == true)
            {
                correction = "Your answer contains an apply_patch envelope as plain text, and text changes nothing: no "
                    + "file was created or changed, because you called no tool. To make that change, call the "
                    + SkillToolNames.ApplyPatch + " tool with the envelope, then reply with your final answer.";
            }
            else if (string.IsNullOrWhiteSpace(content) && string.IsNullOrWhiteSpace(result.Output.Parsed?.Thinking))
            {
                correction = "You ended your turn without doing anything or writing an answer. Do your task now "
                    + "with your tools, then reply with your final answer.";
            }
            if (correction == null)
                return result;

            List<ChatMessage> messages = result.Messages;
            ParsedOutput? parsed = result.Output.Parsed;
            messages.Add(new ChatMessage
            {
                Role = "assistant",
                Content = content,
                Thinking = string.IsNullOrEmpty(parsed?.Thinking) ? null : parsed!.Thinking,
                RawOutputTokens = result.Output.RawTokens != null ? new List<int>(result.Output.RawTokens) : null,
                RawPromptTrailingWhitespace = result.Output.RawPromptTrailingWhitespace,
                RawGenerationSuffix = result.Output.RawGenerationSuffix,
            });
            messages.Add(new ChatMessage { Role = "user", Content = correction });

            int rounds = result.Rounds;
            SkillLoopResult corrected = await SkillAgentLoop.RunAsync(
                messages, agent.Tools?.ToList(), context, agent.Generator!, loopOptions, token).ConfigureAwait(false);
            return corrected with { Rounds = corrected.Rounds + rounds };
        }

        private SkillAgentLoopOptions ChildLoopOptions(SubAgent agent)
        {
            SkillAgentLoopOptions baseOptions = _binding.LoopOptions ?? SkillAgentLoopOptions.Default;
            return new SkillAgentLoopOptions
            {
                MaxRounds = baseOptions.MaxRounds,
                MaxCallsPerRound = baseOptions.MaxCallsPerRound,
                ToolResultsAreRendered = baseOptions.ToolResultsAreRendered,
                ClientTools = baseOptions.ClientTools,
                OnInvocation = invocation =>
                {
                    Report(agent, "round " + invocation.Round.ToString(CultureInfo.InvariantCulture) + ": "
                        + invocation.Tool + (invocation.Ok ? string.Empty : " (failed)"));
                    baseOptions.OnInvocation?.Invoke(invocation);
                },
            };
        }

        // ---- conversation construction ------------------------------------------

        /// <summary>
        /// A fresh agent's first prompt: the parent's leading system messages VERBATIM,
        /// then the task.
        ///
        /// <para>
        /// Verbatim is the performance design, not a convenience. The engine reuses KV by
        /// longest common prefix, and the public prefix it checkpoints is exactly the
        /// leading system messages plus the tool block. A sub-agent whose system prompt
        /// carried its role, its id or one reworded sentence would diverge in the first
        /// few hundred tokens and prefill the whole instruction block again; one that
        /// matches byte for byte starts where the parent's checkpoint ends. So the role
        /// and the task go in the first USER message, and the tool block is the parent's
        /// own list, unchanged — including tools this agent is not allowed to use, which
        /// are refused when called rather than withheld.
        /// </para>
        /// </summary>
        private List<ChatMessage> BuildFreshHistory(List<ChatMessage> conversation, SubAgent agent)
        {
            var history = new List<ChatMessage>();
            foreach (ChatMessage message in conversation)
            {
                if (!IsInstructionRole(message.Role))
                    break;
                history.Add(CloneMessage(message));
            }
            history.Add(new ChatMessage { Role = "user", Content = Envelope(agent, forked: false) });
            return history;
        }

        /// <summary>
        /// A forked agent's first prompt: the parent's conversation up to and including the
        /// turn that spawned it, with that turn's calls answered, then its task as a user
        /// message.
        ///
        /// <para>
        /// Unlike Codex, the fork keeps the parent's tool calls. Codex drops them to give the
        /// child a clean transcript; here that would cut the shared prefix at the first
        /// dropped call, and reusing the parent's KV is the only reason to fork at all.
        /// </para>
        /// <para>
        /// The task is a USER turn, not the spawn call's tool result. Measured on
        /// gemma-4-E2B at temperature 0: with the task inside the tool result, the forked
        /// agent ended its turn on its very first token in all three runs — a model reads a
        /// tool result as the end of a step it took, not as a new instruction. The spawn call's result is
        /// what the parent itself was told, and the task follows as the next thing said to
        /// it. The shared prefix is the same either way: it ends where the parent's own
        /// generation ended.
        /// </para>
        /// </summary>
        private List<ChatMessage> BuildForkedHistory(
            List<ChatMessage> conversation, ToolCall call, SubAgent agent, bool resultsRendered, out bool forked)
        {
            forked = false;
            int spawner = -1;
            for (int i = conversation.Count - 1; i >= 0; i--)
            {
                ChatMessage message = conversation[i];
                if (message.Role == "assistant" && message.ToolCalls != null
                    && message.ToolCalls.Any(c => ReferenceEquals(c, call)
                        || (!string.IsNullOrEmpty(call.Id) && c.Id == call.Id)))
                {
                    spawner = i;
                    break;
                }
            }
            if (spawner < 0)
                return BuildFreshHistory(conversation, agent);

            forked = true;
            var history = new List<ChatMessage>(spawner + 8);
            for (int i = 0; i <= spawner; i++)
                history.Add(CloneMessage(conversation[i]));

            foreach (ToolCall sibling in conversation[spawner].ToolCalls!)
            {
                bool own = ReferenceEquals(sibling, call) || (!string.IsNullOrEmpty(call.Id) && sibling.Id == call.Id);
                ChatMessage? existing = own || string.IsNullOrEmpty(sibling.Id)
                    ? null
                    : conversation.Skip(spawner + 1).FirstOrDefault(m => m.Role == "tool" && m.ToolCallId == sibling.Id);
                string content = own
                    ? agent.Id + " started with a copy of this conversation."
                    : existing?.Content ?? "(The result of this call is not shown to the sub-agent.)";
                history.Add(resultsRendered
                    ? new ChatMessage { Role = "tool", Content = content, ToolCallId = sibling.Id }
                    : new ChatMessage { Role = "user", Content = "Result of your " + sibling.Name + " call:\n\n" + content });
            }
            history.Add(new ChatMessage { Role = "user", Content = Envelope(agent, forked: true) });
            return history;
        }

        /// <summary>
        /// A sub-agent's task, as the first thing said to it.
        ///
        /// <para>
        /// The file-naming rule is here because the shared directory is shared for real:
        /// measured on gemma-4-E4B, two of four concurrent agents each wrote its program to
        /// <c>solution.py</c>, the second write replaced the first, and both then ran the
        /// same program and reported the same (wrong for one of them) number — in all three
        /// runs. Codex's answer is to tell the model its workers must have disjoint write
        /// sets; a small model given an abstract rule still reaches for the same generic
        /// name, while a concrete prefix it can copy removes the collision.
        /// </para>
        /// </summary>
        private string Envelope(SubAgent agent, bool forked)
        {
            var sb = new StringBuilder();
            sb.Append("You are ").Append(agent.Id).Append(", a sub-agent. ");
            sb.Append(forked
                ? "You were forked from the conversation above to do the task below: you have its context, but you are now a separate agent - not the one that started you - and that agent is waiting for your result. "
                : "Another agent started you to do the task below, and it is waiting for your result. ");
            sb.Append("Work on it on your own, with your tools; you cannot ask the user anything. Do the work by calling "
                    + "your tools: text you write in your answer changes nothing. When you are done, "
                    + "reply with your final answer. It goes back to the agent that started you, not to the user, so "
                    + "make it complete and self-contained: the result itself, the paths of any files you created or "
                    + "changed, and anything you could not do. Other agents are working in the same working directory "
                    + "at the same time, so start the name of every file you create for your own use - programs, "
                    + "notes, intermediate output - with your id, like " + agent.Id + "_count.py, so it cannot "
                    + "collide with theirs; a file your task names keeps that name. Change only the files your task is "
                    + "about, and do not undo changes you did not make.");
            if (agent.Depth >= _options.MaxDepth)
                sb.Append(" You cannot start sub-agents of your own; do the work yourself.");
            sb.Append("\n\nTask:\n").Append(agent.Task);
            return sb.ToString();
        }

        private static string FollowUpMessage(string message) =>
            "Message from the agent that started you:\n\n" + message
            + "\n\nWhen you are done, reply with your final answer, as before.";

        private static bool IsInstructionRole(string? role) =>
            string.Equals(role, "system", StringComparison.OrdinalIgnoreCase)
            || string.Equals(role, "developer", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// A copy that shares no mutable list with the parent's conversation, so nothing
        /// either loop appends or rewrites can reach the other. Every field is carried —
        /// the raw tokens above all, since a forked prefix that re-tokenizes instead of
        /// splicing no longer matches the KV it was forked to reuse.
        /// </summary>
        internal static ChatMessage CloneMessage(ChatMessage message) => new()
        {
            Role = message.Role,
            Content = message.Content,
            ImagePaths = message.ImagePaths != null ? new List<string>(message.ImagePaths) : null,
            ImageTimestamps = message.ImageTimestamps != null ? new List<double?>(message.ImageTimestamps) : null,
            AudioPaths = message.AudioPaths != null ? new List<string>(message.AudioPaths) : null,
            TextFilePaths = message.TextFilePaths != null ? new List<string>(message.TextFilePaths) : null,
            TextFileNames = message.TextFileNames != null ? new List<string>(message.TextFileNames) : null,
            HasFileBackedTextAttachments = message.HasFileBackedTextAttachments,
            AttachmentPaths = message.AttachmentPaths != null ? new List<string>(message.AttachmentPaths) : null,
            AttachmentNames = message.AttachmentNames != null ? new List<string>(message.AttachmentNames) : null,
            IsVideo = message.IsVideo,
            ToolCalls = message.ToolCalls != null ? new List<ToolCall>(message.ToolCalls) : null,
            ToolCallId = message.ToolCallId,
            Thinking = message.Thinking,
            RawOutputTokens = message.RawOutputTokens != null ? new List<int>(message.RawOutputTokens) : null,
            RawPromptTrailingWhitespace = message.RawPromptTrailingWhitespace,
            RawGenerationSuffix = message.RawGenerationSuffix,
            CacheControl = message.CacheControl != null
                ? new CacheControlMarker { Type = message.CacheControl.Type }
                : null,
            ContentCacheBreakpoints = message.ContentCacheBreakpoints != null
                ? new List<int>(message.ContentCacheBreakpoints)
                : null,
        };

        // ---- bookkeeping ---------------------------------------------------------

        private List<SubAgent> ChildrenLocked(SubAgentScope caller) =>
            _agents.Values.Where(a => a.Parent == caller.Self).OrderBy(a => a.Number).ToList();

        private bool HasOpenChildrenLocked(SubAgent agent) =>
            _agents.Values.Any(a => a.Parent == agent && a.IsOpen);

        /// <summary>True when <paramref name="agent"/> was started by <paramref name="ancestor"/> or one of its sub-agents; every agent is the top-level agent's (null).</summary>
        private static bool IsDescendantLocked(SubAgent agent, SubAgent? ancestor)
        {
            if (ancestor == null)
                return true;
            for (SubAgent? parent = agent.Parent; parent != null; parent = parent.Parent)
            {
                if (parent == ancestor)
                    return true;
            }
            return false;
        }

        private bool TryResolveOwnedLocked(SubAgentScope caller, string id, out SubAgent? agent, out string? error)
        {
            string key = (id ?? string.Empty).Trim();
            if (_agents.TryGetValue(key, out agent) && agent.Parent == caller.Self)
            {
                error = null;
                return true;
            }

            // Numbers alone are how a model most often gets an id wrong ("1" for agent_1).
            if (int.TryParse(key, NumberStyles.Integer, CultureInfo.InvariantCulture, out int number)
                && _agents.TryGetValue("agent_" + number.ToString(CultureInfo.InvariantCulture), out agent)
                && agent.Parent == caller.Self)
            {
                error = null;
                return true;
            }

            agent = null;
            List<SubAgent> own = ChildrenLocked(caller);
            error = "agent with id " + key + " not found. "
                + (own.Count == 0
                    ? "You have no sub-agents."
                    : "Your agents are: " + string.Join(", ", own.Select(a => a.Id)) + ".");
            return false;
        }

        /// <summary>Close <paramref name="agent"/> and every open descendant; return the token sources to cancel OUTSIDE the lock.</summary>
        private List<CancellationTokenSource> CloseLocked(SubAgent agent, List<CancellationTokenSource> stopping)
        {
            foreach (SubAgent child in _agents.Values.Where(a => a.Parent == agent && a.IsOpen).ToList())
                CloseLocked(child, stopping);
            MarkClosedLocked(agent);
            stopping.Add(agent.Cts);
            return stopping;
        }

        private static void MarkClosedLocked(SubAgent agent)
        {
            agent.Status = SubAgentStatus.Closed;
            agent.Delivered = true;
            agent.Inbox.Clear();
            if (agent.FinishedAt == 0)
                agent.FinishedAt = Stopwatch.GetTimestamp();
        }

        private string DescribeResultLocked(SubAgent agent)
        {
            agent.Delivered = true;
            string elapsed = FormatSpan(agent.StartedAt, agent.FinishedAt);
            if (agent.Status == SubAgentStatus.Errored)
            {
                return agent.Id + " failed after " + elapsed + ": " + Abbreviate(agent.Result ?? "unknown error", 900)
                     + "\nIf you still need its result, give it the task again with " + SkillToolNames.SendInput
                     + " or do the task yourself.";
            }

            string answer = agent.Result ?? string.Empty;
            if (answer.Length > MaxResultChars)
            {
                answer = answer.Substring(0, MaxResultChars)
                    + "\n[... truncated: the full answer is " + agent.Result!.Length.ToString(CultureInfo.InvariantCulture)
                    + " characters. Ask " + agent.Id + " for the part you need with " + SkillToolNames.SendInput + ".]";
            }
            return agent.Id + " completed in " + elapsed + " " + DescribeWorkLocked(agent) + ". Its final answer:\n" + answer;
        }

        /// <summary>
        /// What the agent actually DID, from the host's own record rather than its words:
        /// how many rounds, which tools ran, which failed, which files they produced.
        ///
        /// <para>
        /// A sub-agent's answer is a claim, and a small model makes false ones — observed
        /// on this host: a sub-agent asked to write a file answered with a patch envelope
        /// as plain TEXT, which applies nothing, and its parent reported the file as
        /// written. The parent cannot inspect the sub-agent's transcript, so the record
        /// travels with the claim, and "it called no tools" is there to be read next to
        /// "I created the file".
        /// </para>
        /// </summary>
        private static string DescribeWorkLocked(SubAgent agent)
        {
            string rounds = agent.TurnRounds.ToString(CultureInfo.InvariantCulture)
                + (agent.TurnRounds == 1 ? " round" : " rounds");
            List<SkillToolInvocation> calls = agent.TurnInvocations;
            if (calls.Count == 0)
                return "(" + rounds + "; it called no tools)";

            var sb = new StringBuilder("(").Append(rounds).Append("; tools it ran: ");
            sb.Append(string.Join(", ", calls
                .GroupBy(c => c.Tool, StringComparer.Ordinal)
                .Select(g => g.Count() == 1 ? g.Key : g.Key + " x" + g.Count().ToString(CultureInfo.InvariantCulture))));
            int failed = calls.Count(c => !c.Ok);
            if (failed > 0)
                sb.Append("; ").Append(failed.ToString(CultureInfo.InvariantCulture)).Append(failed == 1 ? " call failed" : " calls failed");
            List<string> files = calls.SelectMany(c => c.Files).Select(f => f.Name).Distinct(StringComparer.Ordinal).ToList();
            if (files.Count > 0)
                sb.Append("; files produced: ").Append(string.Join(", ", files.Take(12))).Append(files.Count > 12 ? ", ..." : string.Empty);
            return sb.Append(')').ToString();
        }

        private string DescribeReadyLocked(List<SubAgent> ready, List<SubAgent> awaited)
        {
            var sb = new StringBuilder();
            foreach (SubAgent agent in ready)
            {
                if (sb.Length > 0)
                    sb.Append("\n\n");
                sb.Append(DescribeResultLocked(agent));
                _logger?.LogInformation(LogEventIds.SkillToolInvoked, "agents.deliver id={Id} via=wait_agent", agent.Id);
            }
            List<SubAgent> running = awaited.Where(a => a.IsWorking).ToList();
            if (running.Count > 0)
                sb.Append("\n\nStill running: ").Append(string.Join(", ", running.Select(a => a.Id + " (" + FormatElapsed(a.StartedAt) + ")"))).Append('.');
            return sb.ToString();
        }

        private static string DescribeNothingToWaitForLocked(List<SubAgent> awaited) =>
            "Nothing to wait for: none of those agents is working, and every answer was already delivered to you ("
            + string.Join(", ", awaited.Select(a => a.Id + ": " + Describe(a.Status))) + ").";

        private static string DescribeTimeoutLocked(List<SubAgent> awaited, long started)
        {
            List<SubAgent> running = awaited.Where(a => a.IsWorking).ToList();
            return "No agent finished within " + FormatElapsed(started) + ". Still running: "
                + string.Join(", ", running.Select(a => a.Id + " (" + FormatElapsed(a.StartedAt) + ")"))
                + ". Keep working on something else, or call " + SkillToolNames.WaitAgent + " again.";
        }

        private static string WithNote(string text, string? note) => note == null ? text : text + "\n" + note;

        private static string FinalAnswer(ParsedOutput? parsed)
        {
            string content = (parsed?.Content ?? string.Empty).Trim();
            if (content.Length > 0)
                return content;
            // A reasoning model that put its whole answer in the thinking channel. The
            // tail of that is still better evidence than nothing, and labelled as such.
            string thinking = (parsed?.Thinking ?? string.Empty).Trim();
            if (thinking.Length > 0)
            {
                string tail = thinking.Length > 2000 ? "..." + thinking.Substring(thinking.Length - 2000) : thinking;
                return "(The sub-agent wrote no final answer. The end of its reasoning was:)\n" + tail;
            }
            return "(The sub-agent ended without writing a final answer.)";
        }

        internal static string Describe(SubAgentStatus status) => status switch
        {
            SubAgentStatus.Starting => "starting",
            SubAgentStatus.Running => "running",
            SubAgentStatus.Completed => "completed",
            SubAgentStatus.Errored => "failed",
            _ => "closed",
        };

        private static string Abbreviate(string text, int max)
        {
            string flat = (text ?? string.Empty).Replace("\r", " ", StringComparison.Ordinal).Replace("\n", " ", StringComparison.Ordinal).Trim();
            return flat.Length <= max ? flat : flat.Substring(0, max - 3) + "...";
        }

        private static string FormatElapsed(long since) =>
            since == 0 ? "0 s" : FormatDuration(Stopwatch.GetElapsedTime(since));

        private static string FormatSpan(long from, long to) =>
            from == 0 || to == 0 ? "0 s" : FormatDuration(Stopwatch.GetElapsedTime(from, to));

        private static string FormatDuration(TimeSpan span)
        {
            if (span.TotalSeconds < 60)
                return span.TotalSeconds.ToString("0.0", CultureInfo.InvariantCulture) + " s";
            return ((int)span.TotalMinutes).ToString(CultureInfo.InvariantCulture) + " min "
                 + span.Seconds.ToString(CultureInfo.InvariantCulture) + " s";
        }

        private void Report(SubAgent agent, string text)
        {
            Action<string>[] listeners;
            lock (_gate)
            {
                agent.LastActivity = text;
                if (_listeners.Count == 0)
                    return;
                listeners = _listeners.ToArray();
            }
            string line = agent.Id + ": " + text;
            foreach (Action<string> listener in listeners)
            {
                try { listener(line); }
                catch (Exception ex) when (ex is not OutOfMemoryException) { /* a UI tap must never fail an agent */ }
            }
        }

        private void AddListener(Action<string>? listener)
        {
            if (listener == null)
                return;
            lock (_gate)
                _listeners.Add(listener);
        }

        private void RemoveListener(Action<string>? listener)
        {
            if (listener == null)
                return;
            lock (_gate)
                _listeners.Remove(listener);
        }

        private static TaskCompletionSource NewSignal() =>
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        /// <summary>Wake every waiter. A waiter captures the signal under the lock it evaluates in, so no change is missed.</summary>
        private void Signal()
        {
            TaskCompletionSource fired;
            lock (_gate)
            {
                fired = _changed;
                _changed = NewSignal();
            }
            fired.TrySetResult();
        }

        /// <summary>
        /// End the turn: stop every agent still working and give them a moment to let go
        /// of the engine and the workspace before the request that owns both is released.
        /// </summary>
        public void Dispose()
        {
            Task[] workers;
            List<SubAgentSnapshot> summary;
            lock (_gate)
            {
                if (_disposed)
                    return;
                _disposed = true;
                workers = _agents.Values.Select(a => a.Worker).Where(t => t != null).Cast<Task>().ToArray();
                summary = _agents.Values.Select(a => a.ToSnapshot()).ToList();
            }

            int stillWorking = summary.Count(a => a.Status is SubAgentStatus.Starting or SubAgentStatus.Running);
            _turnCts.Cancel();
            Signal();
            try
            {
                if (workers.Length > 0)
                    Task.WaitAll(workers, DisposeGrace);
            }
            catch (AggregateException) { /* each worker records its own outcome */ }

            if (summary.Count > 0)
            {
                _logger?.LogInformation(LogEventIds.SkillToolInvoked,
                    "agents.turn spawned={Spawned} completed={Completed} failed={Failed} closed={Closed} stoppedAtTurnEnd={Stopped}",
                    summary.Count,
                    summary.Count(a => a.Status == SubAgentStatus.Completed),
                    summary.Count(a => a.Status == SubAgentStatus.Errored),
                    summary.Count(a => a.Status == SubAgentStatus.Closed),
                    stillWorking);
                if (stillWorking > 0)
                {
                    _logger?.LogWarning(LogEventIds.SkillToolInvoked,
                        "agents.turn-ended-with-work stopped={Stopped} - the turn ended while sub-agents were still working; their results were discarded",
                        stillWorking);
                }
            }
            // _turnCts is deliberately not disposed: a worker that outlived the grace
            // period may still read its token on the way out, and a disposed source
            // throws where a cancelled one only answers "cancelled".
        }
    }

    /// <summary>One agent, as a snapshot.</summary>
    public sealed record SubAgentSnapshot(
        string Id, string? ParentId, int Depth, SubAgentStatus Status, int Turns, int Rounds,
        int ToolCalls, bool ResultDelivered, string Task, string? Result, string? LastActivity);

    /// <summary>One agent's live state. Every field is guarded by the runtime's lock.</summary>
    internal sealed class SubAgent
    {
        public SubAgent(int number, string id, SubAgent? parent, int depth, string task, bool forked, CancellationToken turn)
        {
            Number = number;
            Id = id;
            Parent = parent;
            Depth = depth;
            Task = task;
            Forked = forked;
            // Linked to the parent's own source when the parent is an agent, so closing
            // an agent reaches its whole subtree even between two lock-held passes.
            Cts = CancellationTokenSource.CreateLinkedTokenSource(parent?.Cts.Token ?? turn, turn);
        }

        public int Number { get; }
        public string Id { get; }
        public SubAgent? Parent { get; }
        public int Depth { get; }
        public string Task { get; }
        public bool Forked { get; set; }
        public CancellationTokenSource Cts { get; }
        public SubAgentScope Scope { get; set; } = null!;
        public IReadOnlyList<ToolFunction>? Tools { get; set; }
        public SkillTurnGenerator? Generator { get; set; }
        public Task? Worker { get; set; }
        public List<ChatMessage> History { get; set; } = new();
        public List<string> Inbox { get; } = new();
        public SubAgentStatus Status { get; set; } = SubAgentStatus.Starting;
        public string? Result { get; set; }
        public bool Delivered { get; set; } = true;
        public int Turns { get; set; }
        public int Rounds { get; set; }
        public int ToolCalls { get; set; }

        /// <summary>Rounds of the latest task, follow-ups steered into it included.</summary>
        public int TurnRounds { get; set; }

        /// <summary>Every tool call of the latest task, for the record the parent is given.</summary>
        public List<SkillToolInvocation> TurnInvocations { get; } = new();
        public long StartedAt { get; set; } = Stopwatch.GetTimestamp();
        public long FinishedAt { get; set; }
        public long FinishOrder { get; set; }
        public string? LastActivity { get; set; }

        public bool IsOpen => Status != SubAgentStatus.Closed;
        public bool IsWorking => Status is SubAgentStatus.Starting or SubAgentStatus.Running;
        public bool HasUndeliveredResult =>
            !Delivered && Status is SubAgentStatus.Completed or SubAgentStatus.Errored;

        public SubAgentSnapshot ToSnapshot() => new(
            Id, Parent?.Id, Depth, Status, Turns, Rounds, ToolCalls, Delivered, Task, Result, LastActivity);
    }
}
