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
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Cli
{
    /// <summary>
    /// The one way a CLI tool loop turns a conversation into prompt tokens: the parent's
    /// rounds, in both loops, and every sub-agent round.
    ///
    /// <para>
    /// One object rather than one call repeated in three places, because the sub-agent
    /// design depends on it. A fresh sub-agent's prompt starts with its parent's leading
    /// system message and tool list, verbatim, so that it continues the public prefix the
    /// parent's rounds already put in the engine's radix tree. That holds only if the two
    /// prompts are rendered by the same template adapter, with the same tokenizer, tools
    /// and thinking flag; a sub-agent rendered any other way diverges in the first few
    /// tokens and prefills the whole instruction block again.
    /// </para>
    /// </summary>
    internal sealed class CliRoundRenderer
    {
        private readonly KVCachePromptRenderer _renderer;
        private readonly string _chatTemplate;

        public CliRoundRenderer(IPromptRenderer renderer, ITokenizer tokenizer, string chatTemplate,
            string architecture, bool enableThinking)
        {
            ArgumentNullException.ThrowIfNull(renderer);
            _renderer = new KVCachePromptRenderer(renderer);
            Tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _chatTemplate = chatTemplate;
            Architecture = architecture;
            EnableThinking = enableThinking;
        }

        /// <summary>The renderer for <paramref name="model"/>'s own template and tokenizer.</summary>
        public static CliRoundRenderer For(IModelArchitecture model, IPromptRenderer renderer, bool enableThinking) =>
            new(renderer, model.Tokenizer, model.Config.ChatTemplate, model.Config.Architecture, enableThinking);

        public ITokenizer Tokenizer { get; }

        public string Architecture { get; }

        public bool EnableThinking { get; }

        /// <summary>
        /// Render a round's prompt, generation prompt included, splicing every recorded
        /// assistant turn's raw tokens back in so the prompt continues the KV the engine holds.
        /// </summary>
        /// <param name="trailingWhitespace">
        /// The exact whitespace the prompt ends with, recorded beside the tokens the round
        /// generates so the next render can replay the boundary.
        /// </param>
        public List<int> Render(List<ChatMessage> messages, List<ToolFunction> tools, out string trailingWhitespace) =>
            _renderer.RenderToTokens(Tokenizer, _chatTemplate, messages, Architecture,
                addGenerationPrompt: true, out _, out trailingWhitespace,
                tools: tools, enableThinking: EnableThinking);
    }

    /// <summary>What one sub-agent round did, for the log line and for tests.</summary>
    /// <param name="AgentId">The agent, e.g. <c>agent_1</c>.</param>
    /// <param name="Round">1 for the agent's first generation, counting across its follow-ups.</param>
    /// <param name="Fork">Whether the agent was started with a copy of its parent's conversation.</param>
    /// <param name="CacheScope">The engine cache scope the round ran in.</param>
    /// <param name="Prompt">The round's prompt tokens, media expanded.</param>
    /// <param name="SharedPrefixTokens">How many of them were declared public to the engine.</param>
    /// <param name="ReusedTokens">How many the engine did not have to prefill.</param>
    internal sealed record CliSubAgentRound(
        string AgentId, int Round, bool Fork, string CacheScope, IReadOnlyList<int> Prompt,
        int SharedPrefixTokens, int ReusedTokens, int OutputTokens, double TimeToFirstTokenMs,
        double DecodeMs, string FinishReason);

    /// <summary>
    /// How the CLI generates for its sub-agents: the <see cref="SubAgentHostBinding.CreateGenerator"/>
    /// of both CLI loops, and the CLI's counterpart of the server's <c>SubAgentGenerator</c>.
    ///
    /// <para>
    /// Every sub-agent round is rendered with the PARENT's renderer and settings — the same
    /// <see cref="CliRoundRenderer"/>, the same sampling, token budget and public-prefix
    /// computation — and submitted to the parent's own engine through
    /// <see cref="CliInferenceSession.GenerateInScopeAsync"/>, so the engine batches the
    /// agents' decoding with each other and with the parent's.
    /// </para>
    /// <para>
    /// <b>The cache scope is the KV design</b>, as on the server. A fresh sub-agent gets a
    /// scope of its own: its prompt starts with its parent's exact instructions and tool
    /// block, so it reuses the public prefix they share, and its later rounds continue its
    /// own state. A forked sub-agent runs in the PARENT's scope, because its prompt is the
    /// parent's conversation plus one tool result; in any other scope the parent's state is
    /// out of reach past the public boundary.
    /// </para>
    /// </summary>
    internal sealed class CliSubAgentGeneration
    {
        private readonly IModelArchitecture _model;
        private readonly CliRoundRenderer _renderer;
        private readonly CliInferenceSession _inference;
        private readonly int _maxTokens;
        private readonly SamplingConfig _sampling;
        private readonly bool _enablePrefixCache;
        private readonly Func<List<ChatMessage>, List<ToolFunction>, IReadOnlyList<int>> _sharedPrefix;
        private readonly ILogger _log;

        /// <param name="model">The loaded model: its media injector and its GPU lock.</param>
        /// <param name="renderer">The renderer the parent's rounds use.</param>
        /// <param name="inference">The parent's session, whose engine the agents share.</param>
        /// <param name="maxTokens">The parent's per-round token budget.</param>
        /// <param name="sampling">The parent's sampling.</param>
        /// <param name="enablePrefixCache">False under <c>--no-prefix-cache</c>, as for the parent.</param>
        /// <param name="sharedPrefix">
        /// The parent's own public-prefix computation, given a round's messages and tools.
        /// An agent's leading system message and tools are its parent's, so it yields the
        /// prefix the parent's rounds declared.
        /// </param>
        /// <param name="log">Optional.</param>
        public CliSubAgentGeneration(
            IModelArchitecture model,
            CliRoundRenderer renderer,
            CliInferenceSession inference,
            int maxTokens,
            SamplingConfig sampling,
            bool enablePrefixCache,
            Func<List<ChatMessage>, List<ToolFunction>, IReadOnlyList<int>> sharedPrefix,
            ILogger log)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _renderer = renderer ?? throw new ArgumentNullException(nameof(renderer));
            _inference = inference ?? throw new ArgumentNullException(nameof(inference));
            _maxTokens = maxTokens;
            _sampling = sampling ?? SamplingConfig.Greedy;
            _enablePrefixCache = enablePrefixCache;
            _sharedPrefix = sharedPrefix;
            _log = log;
        }

        /// <summary>Called after every sub-agent round, from the agent's thread.</summary>
        public Action<CliSubAgentRound> OnRound { get; init; }

        /// <summary>The generator for one agent, pinned to its cache scope for every round of every turn it runs.</summary>
        public SkillTurnGenerator CreateGenerator(SubAgentLaunch launch)
        {
            ArgumentNullException.ThrowIfNull(launch);
            // Resolved at spawn, on the parent's thread: the conversation the fork copied
            // is the one live in this scope now.
            string scope = launch.ForkContext ? _inference.CacheScope : CliInferenceSession.NewAgentScope();
            int rounds = 0;
            return async (messages, tools, cancellationToken) =>
            {
                int round = Interlocked.Increment(ref rounds);
                List<int> prompt = _renderer.Render(messages, tools, out string trailingWhitespace);
                string requestId = $"cli-{launch.AgentId}-{Guid.NewGuid():N}";
                IMultimodalInjector injector = _model.MultimodalInjector;
                try
                {
                    prompt = CliSubAgents.PreparePrompt(_model, messages, prompt, requestId, concurrent: true);
                    int shared = CliSharedPrefix.MatchingLength(_sharedPrefix?.Invoke(messages, tools), prompt);

                    // Stop sequences end a round exactly as they end the parent's.
                    var sampler = new TokenSampler(_sampling);
                    var streamed = new List<int>();
                    string trimmedAtStop = null;
                    bool Emit(int token)
                    {
                        if (_sampling.StopSequences == null || _sampling.StopSequences.Count == 0)
                            return true;
                        streamed.Add(token);
                        var (trimmed, shouldStop) = sampler.CheckStopSequences(_renderer.Tokenizer.Decode(streamed));
                        if (shouldStop)
                            trimmedAtStop = trimmed;
                        return !shouldStop;
                    }

                    CliInferenceSession.Result result = await _inference.GenerateInScopeAsync(
                        prompt, _maxTokens, _sampling, scope, Emit, cancellationToken, requestId,
                        injector?.GetPreparedMediaSpans(requestId), _enablePrefixCache, shared).ConfigureAwait(false);

                    string finishReason = trimmedAtStop != null ? "stop_sequence" : result.Completion.FinishReason;
                    ReportRound(new CliSubAgentRound(
                        launch.AgentId, round, launch.ForkContext, scope, prompt, _enablePrefixCache ? shared : 0,
                        result.Completion.PrefixCacheReusedTokens, result.Tokens.Count, result.PrefillMs,
                        result.DecodeMs, finishReason));

                    // Parsed exactly as the parent's rounds are: the same factory, primed
                    // with the same thinking flag, tools and prompt tail.
                    string decoded = trimmedAtStop ?? _renderer.Tokenizer.Decode(result.Tokens);
                    IOutputParser parser = CliOutputParser.Create(_renderer.Architecture, _renderer.EnableThinking,
                        tools, _renderer.Tokenizer, prompt);
                    bool useParser = _renderer.EnableThinking || (tools != null && tools.Count > 0) || parser.AlwaysRequired;
                    ParsedOutput parsed = useParser ? parser.Add(decoded, true) : new ParsedOutput { Content = decoded };

                    // RawGenerationSuffix stays unset, as it is for the parent's CLI rounds:
                    // the thinking flag cannot change inside a turn, so the renderer's
                    // current-request suffix is the one these tokens were generated after.
                    return new SkillTurnOutput(parsed, result.Tokens)
                    {
                        RawPromptTrailingWhitespace = trailingWhitespace,
                    };
                }
                finally
                {
                    injector?.ClearPreparedPromptState(requestId);
                }
            };
        }

        private void ReportRound(CliSubAgentRound round)
        {
            double tokensPerSec = round.OutputTokens > 0
                ? round.OutputTokens / Math.Max(round.DecodeMs / 1000.0, 1e-9)
                : 0;
            // The engine's own lines cannot say whose round it was, and concurrent agents
            // interleave; this one can, so per-agent KV reuse is readable from the log.
            _log?.LogInformation(LogEventIds.SkillToolInvoked,
                "agents.round id={Id} round={Round} fork={Fork} promptTokens={PromptTokens} sharedPrefix={SharedPrefix} kvReused={KvReused} evalTokens={EvalTokens} ttftMs={TtftMs:F0} tokensPerSec={TokensPerSec:F1} finishReason={FinishReason}",
                round.AgentId, round.Round, round.Fork, round.Prompt.Count, round.SharedPrefixTokens,
                round.ReusedTokens, round.OutputTokens, round.TimeToFirstTokenMs, tokensPerSec, round.FinishReason);
            try { OnRound?.Invoke(round); }
            catch (Exception ex) when (ex is not OutOfMemoryException) { /* an observer must never fail an agent */ }
        }
    }

    /// <summary>
    /// Sub-agents in the CLI: the flags, the tool declarations, one runtime per turn, and
    /// the pieces both hand-written tool loops share so they cannot drift from each other
    /// or from the server's loop.
    /// </summary>
    internal static class CliSubAgents
    {
        /// <summary>
        /// Read the sub-agent flags (then the environment) and take them out of the argument
        /// list, the way <c>CodeExecOptions.Parse</c> takes its own: the CLI's switch has no
        /// case for them and no unknown-flag trap either, so consuming them here is what keeps
        /// <c>--sub-agents-max-threads 3</c>'s value from ever reaching that switch as an
        /// argument of its own.
        /// </summary>
        /// <exception cref="ArgumentException">A value is missing or out of range.</exception>
        internal static SubAgentOptions Parse(string[] args, out string[] remaining)
        {
            SubAgentOptions options = SubAgentOptions.Parse(args).ApplyEnvironment();
            remaining = Strip(args ?? Array.Empty<string>());
            return options;
        }

        /// <summary>The arguments without any flag <see cref="SubAgentOptions"/> owns, or those flags' values.</summary>
        internal static string[] Strip(IReadOnlyList<string> args)
        {
            var kept = new List<string>(args.Count);
            for (int i = 0; i < args.Count; i++)
            {
                string arg = args[i] ?? string.Empty;
                if (SubAgentOptions.SwitchFlags.Any(f => string.Equals(arg, f, StringComparison.OrdinalIgnoreCase)))
                    continue;
                if (SubAgentOptions.ValueFlags.Any(f => string.Equals(arg, f, StringComparison.OrdinalIgnoreCase)))
                {
                    i++; // its value, which SubAgentOptions.Parse already required and validated
                    continue;
                }
                if (SubAgentOptions.ValueFlags.Any(f => arg.StartsWith(f + "=", StringComparison.OrdinalIgnoreCase)))
                    continue;
                kept.Add(args[i]);
            }
            return kept.ToArray();
        }

        /// <summary>
        /// Declare the agent tools AFTER every other tool, as the server's plan does, so the
        /// tool block the code tools pin stays byte-identical when sub-agents are off and
        /// only grows at its end when they are on. A name the operator's own <c>--tools</c>
        /// already uses keeps their definition. Returns <paramref name="tools"/> itself when
        /// sub-agents are off.
        /// </summary>
        internal static List<ToolFunction> AppendAgentTools(List<ToolFunction> tools, SubAgentOptions options, ILogger log)
        {
            if (options is not { Enabled: true })
                return tools;

            var merged = tools != null ? new List<ToolFunction>(tools) : new List<ToolFunction>();
            foreach (ToolFunction declaration in SubAgentTools.Declare())
            {
                if (merged.Any(t => string.Equals(t?.Name, declaration.Name, StringComparison.OrdinalIgnoreCase)))
                {
                    log?.LogWarning(LogEventIds.HostConfiguration,
                        "cli.agents.tool-shadowed name={ToolName} - your --tools definition wins", declaration.Name);
                    continue;
                }
                merged.Add(declaration);
            }
            return merged;
        }

        /// <summary>
        /// The runtime for one turn of the top-level model, or null when sub-agents are off
        /// or the turn offers no tools. Disposed at the end of the turn however it ends, so
        /// no sub-agent outlives the turn that could receive its answer.
        /// </summary>
        /// <param name="turn">The turn's cancellation; every agent stops with it.</param>
        internal static SubAgentRuntime StartTurn(
            SubAgentOptions options,
            SkillToolContext toolContext,
            CliSubAgentGeneration generation,
            SkillAgentLoopOptions loopOptions,
            ILogger log,
            CancellationToken turn)
        {
            if (options is not { Enabled: true } || toolContext == null)
                return null;
            ArgumentNullException.ThrowIfNull(generation);
            return new SubAgentRuntime(
                options,
                new SubAgentHostBinding
                {
                    CreateGenerator = generation.CreateGenerator,
                    // A sub-agent's loop is bounded like its parent's and knows which tools
                    // are the operator's, so it can refuse those rather than stall on them.
                    LoopOptions = loopOptions,
                },
                toolContext,
                log,
                turn);
        }

        /// <summary>
        /// Answer one of the parent's built-in tool calls. Agent tools go through the
        /// turn's runtime — bound first to the conversation the parent is rendering, which a
        /// spawn copies — and are awaited with a live tap, so what the sub-agents do streams
        /// while <c>wait_agent</c> blocks. Everything else goes where it always went.
        /// </summary>
        /// <param name="conversation">The conversation as the parent renders it now: system, user and every turn since.</param>
        internal static SkillToolResult Execute(
            ToolCall call,
            SkillToolContext context,
            SubAgentScope agents,
            Func<List<ChatMessage>> conversation,
            IReadOnlyList<ToolFunction> tools,
            CancellationToken cancellationToken)
        {
            if (agents == null || !SkillToolNames.IsAgentTool(call.Name))
                return SkillTools.Execute(call, context);

            agents.Bind(conversation(), tools);
            return agents.ExecuteAsync(call, Activity, cancellationToken).GetAwaiter().GetResult();
        }

        /// <summary>
        /// The end-of-turn guard's wait: the parent answered with sub-agents still out, so
        /// wait for them (bounded), streaming what they do, and return the message that hands
        /// their results over — or null when nothing was outstanding after all.
        /// </summary>
        internal static string CollectOutstanding(SubAgentScope agents, int round, ILogger log,
            CancellationToken cancellationToken)
        {
            Console.Error.WriteLine("[agent] collecting sub-agent results before answering");
            var clock = Stopwatch.StartNew();
            string handover = agents.CollectOutstandingAsync(
                SubAgentConversation.EndOfTurnWait, Activity, cancellationToken).GetAwaiter().GetResult();
            log?.LogInformation(LogEventIds.SkillToolInvoked,
                "agents.collect-before-answer round={Round} ms={Ms} delivered={Delivered}",
                round, (long)clock.Elapsed.TotalMilliseconds, handover != null);
            return handover;
        }

        /// <summary>One line of what a sub-agent is doing, on stderr beside the loop's <c>[skill]</c> lines.</summary>
        internal static void Activity(string line) => Console.Error.WriteLine("[agent] " + line);

        /// <summary>How an agent tool call is shown: its name and the first line of what it answered.</summary>
        internal static string DescribeCall(ToolCall call, SkillToolResult result)
        {
            string text = (result.Content ?? string.Empty).Trim();
            int newline = text.IndexOf('\n');
            if (newline >= 0)
                text = text.Substring(0, newline).TrimEnd();
            if (text.Length > 160)
                text = text.Substring(0, 157) + "...";
            return $"[agent] {call.Name}: {text}" + (result.Ok ? string.Empty : " (failed)");
        }

        /// <summary>
        /// Expand a round's media placeholders. <paramref name="concurrent"/> is set whenever
        /// sub-agents may be generating: the vision and audio encoders drive the same GPU
        /// backend as the engine's worker, and two threads on one Metal or CUDA queue abort
        /// the process, so preparation takes the model's GPU lock exactly as the server's
        /// pipeline does. With sub-agents off nothing else runs during preparation and the
        /// call is left exactly as it was.
        /// </summary>
        internal static List<int> PreparePrompt(IModelArchitecture model, List<ChatMessage> messages,
            List<int> tokens, string requestId, bool concurrent)
        {
            IMultimodalInjector injector = model.MultimodalInjector;
            if (injector == null)
                return tokens;
            if (!concurrent)
                return injector.ProcessPromptTokens(messages, tokens, requestId);
            lock (model.GpuComputeLock)
                return injector.ProcessPromptTokens(messages, tokens, requestId);
        }

        /// <summary>
        /// A public-prefix computation that runs once per distinct system prompt and tool
        /// list. Every agent of a turn renders the same leading system message and tools, and
        /// the one-shot path computes the prefix with six full renders, so without this each
        /// agent round would repeat that work for an identical answer.
        ///
        /// <para>
        /// Computed under the lock, not merely cached under it: a turn's agents start in the
        /// same instant, and measured on gemma-4-E2B the first three each rendered the same
        /// six probes at once. One computing while the others wait costs them nothing.
        /// </para>
        /// </summary>
        internal static Func<List<ChatMessage>, List<ToolFunction>, IReadOnlyList<int>> Memoize(
            Func<List<ChatMessage>, List<ToolFunction>, IReadOnlyList<int>> compute)
        {
            ArgumentNullException.ThrowIfNull(compute);
            var gate = new object();
            string cachedSystem = null;
            List<ToolFunction> cachedTools = null;
            IReadOnlyList<int> cached = null;
            return (messages, tools) =>
            {
                string system = messages is { Count: > 0 } && messages[0].Role == "system" ? messages[0].Content : null;
                lock (gate)
                {
                    if (cached != null && string.Equals(system, cachedSystem, StringComparison.Ordinal)
                        && SameTools(tools, cachedTools))
                    {
                        return cached;
                    }
                    cached = compute(messages, tools);
                    cachedSystem = system;
                    cachedTools = tools != null ? new List<ToolFunction>(tools) : null;
                    return cached;
                }
            };
        }

        private static bool SameTools(List<ToolFunction> a, List<ToolFunction> b)
        {
            if (a == null || b == null)
                return a == null && b == null;
            if (a.Count != b.Count)
                return false;
            for (int i = 0; i < a.Count; i++)
            {
                if (!ReferenceEquals(a[i], b[i]))
                    return false;
            }
            return true;
        }
    }
}
