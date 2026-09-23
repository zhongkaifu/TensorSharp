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
using System.Globalization;

namespace TensorSharp.AgentHost.Agents
{
    /// <summary>
    /// The sub-agent flags, shared by both hosts.
    ///
    /// <para>
    /// Off by default, and deliberately so. Declaring the five agent tools changes every
    /// agentic request's tool block — a few hundred tokens the model reads on every turn
    /// and a different public prefix for the KV cache — and a spawned agent is a whole
    /// extra generation loop on the same GPU. That is worth paying when an operator asks
    /// for it, and a regression for everyone who did not.
    /// </para>
    /// <para>
    /// Spelled the way <see cref="Skills.SkillHostOptions"/> spells its flags, for the
    /// same reason: a config-file key is a command-line flag, so both hosts must read
    /// identical names, and the tables below are what the drift tests compare the usage
    /// pages against.
    /// </para>
    /// </summary>
    public sealed class SubAgentOptions
    {
        /// <summary>Offers the sub-agent tools on agentic requests.</summary>
        public const string EnableFlag = "--sub-agents";

        /// <summary>How many sub-agents may be open at once in one turn.</summary>
        public const string MaxThreadsFlag = "--sub-agents-max-threads";

        /// <summary>How deep sub-agents may nest: 1 means sub-agents cannot start their own.</summary>
        public const string MaxDepthFlag = "--sub-agents-max-depth";

        /// <summary>Environment override for <see cref="EnableFlag"/>. Anything but <c>0</c> counts as on.</summary>
        public const string EnableEnvVar = "TS_SUB_AGENTS";

        /// <summary>Environment override for <see cref="MaxThreadsFlag"/>.</summary>
        public const string MaxThreadsEnvVar = "TS_SUB_AGENTS_MAX_THREADS";

        /// <summary>Environment override for <see cref="MaxDepthFlag"/>.</summary>
        public const string MaxDepthEnvVar = "TS_SUB_AGENTS_MAX_DEPTH";

        /// <summary>
        /// Default for <see cref="MaxThreads"/>.
        ///
        /// <para>
        /// Codex allows six open agents on its first surface and three on its second. Four
        /// is where a local GPU stops gaining: the engine batches decode across sequences,
        /// and on the families that batch well four concurrent decoders are already most of
        /// the aggregate throughput there is, while every extra sequence holds its own KV.
        /// </para>
        /// </summary>
        public const int DefaultMaxThreads = 4;

        /// <summary>Largest <see cref="MaxThreads"/> accepted: the engine's default running-sequence cap.</summary>
        public const int MaxThreadsLimit = 16;

        /// <summary>
        /// Default for <see cref="MaxDepth"/>: Codex's own. A sub-agent that may start
        /// sub-agents of its own is how a small model recurses until the budget is gone.
        /// </summary>
        public const int DefaultMaxDepth = 1;

        /// <summary>Largest <see cref="MaxDepth"/> accepted.</summary>
        public const int MaxDepthLimit = 4;

        /// <summary>Every switch this class owns, for the usage drift tests and the server's skip list.</summary>
        public static readonly IReadOnlyList<string> SwitchFlags = new[] { EnableFlag };

        /// <summary>Every flag this class owns that takes a value.</summary>
        public static readonly IReadOnlyList<string> ValueFlags = new[] { MaxThreadsFlag, MaxDepthFlag };

        private bool _maxThreadsSet;
        private bool _maxDepthSet;

        /// <summary>Whether the sub-agent tools are offered at all.</summary>
        public bool Enabled { get; set; }

        /// <summary>
        /// Sub-agents that may be open at once in one turn, the parent not counted.
        /// Spawning past it first closes the longest-finished agent whose answer was already
        /// delivered, and fails only when every open agent is still working.
        /// </summary>
        public int MaxThreads { get; set; } = DefaultMaxThreads;

        /// <summary>
        /// How deep agents may nest. The parent is depth 0 and its sub-agents depth 1, so
        /// the default of 1 lets the parent spawn and refuses a sub-agent that tries to.
        /// </summary>
        public int MaxDepth { get; set; } = DefaultMaxDepth;

        /// <summary>Read the sub-agent flags out of a host's raw argument list. Unknown arguments are ignored.</summary>
        /// <exception cref="ArgumentException">A value is missing or out of range. Thrown at startup, before a model loads.</exception>
        public static SubAgentOptions Parse(IReadOnlyList<string>? args)
        {
            var options = new SubAgentOptions();
            if (args == null)
                return options;

            for (int i = 0; i < args.Count; i++)
            {
                string arg = args[i] ?? string.Empty;
                if (string.Equals(arg, EnableFlag, StringComparison.OrdinalIgnoreCase))
                {
                    options.Enabled = true;
                    continue;
                }
                if (TryReadValue(args, ref i, MaxThreadsFlag, out string? threads))
                {
                    options.MaxThreads = ParseBounded(MaxThreadsFlag, threads!, MaxThreadsLimit);
                    options._maxThreadsSet = true;
                    continue;
                }
                if (TryReadValue(args, ref i, MaxDepthFlag, out string? depth))
                {
                    options.MaxDepth = ParseBounded(MaxDepthFlag, depth!, MaxDepthLimit);
                    options._maxDepthSet = true;
                }
            }
            return options;
        }

        /// <summary>
        /// Layer the environment under whatever the command line set. A malformed number in
        /// the environment is as fatal as one on the command line: a limit that silently
        /// fell back to its default is a limit nobody chose.
        /// </summary>
        public SubAgentOptions ApplyEnvironment()
        {
            if (!Enabled
                && Environment.GetEnvironmentVariable(EnableEnvVar) is { Length: > 0 } enable
                && !string.Equals(enable.Trim(), "0", StringComparison.Ordinal))
            {
                Enabled = true;
            }
            // The command line wins over the environment, as it does for every other flag family.
            if (!_maxThreadsSet && Environment.GetEnvironmentVariable(MaxThreadsEnvVar) is { Length: > 0 } threads)
                MaxThreads = ParseBounded(MaxThreadsEnvVar, threads, MaxThreadsLimit);
            if (!_maxDepthSet && Environment.GetEnvironmentVariable(MaxDepthEnvVar) is { Length: > 0 } depth)
                MaxDepth = ParseBounded(MaxDepthEnvVar, depth, MaxDepthLimit);
            return this;
        }

        /// <summary>
        /// The engine setting that bounds how many conversations' finished KV state is kept
        /// for reuse. Read once, when the engine is constructed.
        /// </summary>
        public const string RetainedStatesVariable = "TS_RETAINED_FUSED_CACHE_MAX";

        /// <summary>
        /// The retained-state count sub-agents need: two per concurrent conversation (each
        /// keeps a prompt-end checkpoint and its latest end state) for the parent and every
        /// open sub-agent.
        /// </summary>
        public int RecommendedRetainedStates => 2 * (MaxThreads + 1);

        /// <summary>
        /// Raise the engine's retained-state cap when sub-agents are on and the operator
        /// has not chosen one. Must run before the engine is constructed. Returns what was
        /// done, for the startup log, or null when sub-agents are off.
        ///
        /// <para>
        /// The default cap (4) is sized for conversations taking turns. The cap is one
        /// global LRU across scopes, and every sub-agent round publishes state into it, so
        /// with the default a parent waiting on its agents loses its own retained state
        /// within the agents' first round — measured on gemma-4-E2B: the parent's round
        /// after <c>wait_agent</c> reused only the public prefix, 3178 of 3690 prompt
        /// tokens, instead of everything it had already computed.
        /// </para>
        /// </summary>
        public string? ApplyEngineDefaults()
        {
            if (!Enabled)
                return null;

            string? configured = Environment.GetEnvironmentVariable(RetainedStatesVariable);
            if (!string.IsNullOrWhiteSpace(configured))
            {
                return int.TryParse(configured.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int set)
                    && set > 0 && set < RecommendedRetainedStates
                    ? $"{RetainedStatesVariable}={set} was set explicitly and is kept, though sub-agents need "
                      + $"{RecommendedRetainedStates} ({MaxThreads} agents + the parent, 2 states each): a parent "
                      + "may lose its cached state while its agents run and re-prefill its conversation"
                    : $"{RetainedStatesVariable}={configured.Trim()} was set explicitly and is kept";
            }

            string value = RecommendedRetainedStates.ToString(CultureInfo.InvariantCulture);
            Environment.SetEnvironmentVariable(RetainedStatesVariable, value);
            return $"{RetainedStatesVariable} raised from the default 4 to {value} ({MaxThreads} agents + the parent, "
                 + "2 retained states each), so a parent keeps its cached conversation while its sub-agents run";
        }

        /// <summary>A copy, so a host can hand one to each request without sharing mutable state.</summary>
        public SubAgentOptions Clone() => new()
        {
            Enabled = Enabled,
            MaxThreads = MaxThreads,
            MaxDepth = MaxDepth,
            _maxThreadsSet = _maxThreadsSet,
            _maxDepthSet = _maxDepthSet,
        };

        /// <summary>One line for a startup banner.</summary>
        public string Describe() => Enabled
            ? $"on (max {MaxThreads.ToString(CultureInfo.InvariantCulture)} open, depth {MaxDepth.ToString(CultureInfo.InvariantCulture)})"
            : "off";

        private static int ParseBounded(string name, string text, int max)
        {
            if (!int.TryParse(text?.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
                || value < 1 || value > max)
            {
                throw new ArgumentException(
                    $"Invalid value for {name}: '{text}'. Expected a whole number between 1 and "
                    + max.ToString(CultureInfo.InvariantCulture) + ".");
            }
            return value;
        }

        private static bool TryReadValue(IReadOnlyList<string> args, ref int index, string flag, out string? value)
        {
            value = null;
            string arg = args[index] ?? string.Empty;
            if (string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase))
            {
                if (index + 1 >= args.Count || string.IsNullOrWhiteSpace(args[index + 1]))
                    throw new ArgumentException($"Missing value for option {flag}.");
                value = args[++index];
                return true;
            }
            if (arg.StartsWith(flag + "=", StringComparison.OrdinalIgnoreCase))
            {
                value = arg.Substring(flag.Length + 1);
                if (string.IsNullOrWhiteSpace(value))
                    throw new ArgumentException($"Missing value for option {flag}.");
                return true;
            }
            return false;
        }
    }
}
