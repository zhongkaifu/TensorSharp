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
using System.IO;
using System.Linq;

using TensorSharp.Runtime;
namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// The Agent Skills flags, shared by both hosts.
    ///
    /// <para>
    /// The CLI parses its arguments with a <c>switch</c> and the server with its own
    /// options builder, but the two must spell every flag identically: a config JSON
    /// key <i>is</i> a CLI flag (<c>"skills-dir": [...]</c> becomes
    /// <c>--skills-dir ... --skills-dir ...</c>), so the same config file is expected
    /// to drive either host. Keeping the names and the resolution rules in one place,
    /// in the assembly both reference, is what stops them drifting - the same reason
    /// <c>SpeculativeCliFlags</c> exists for <c>--spec</c>.
    /// </para>
    /// </summary>
    public sealed class SkillHostOptions
    {
        /// <summary>Adds a directory to scan for skills. Repeatable.</summary>
        public const string RootsFlag = "--skills-dir";

        /// <summary>Selects a skill by name for this run. Repeatable.</summary>
        public const string SelectFlag = "--skill";

        /// <summary>Prints the registered skills and exits.</summary>
        public const string ListFlag = "--list-skills";

        /// <summary>Turns the whole feature off.</summary>
        public const string DisableFlag = "--no-skills";

        /// <summary>Stops non-selected skills being advertised to the model.</summary>
        public const string NoDiscoveryFlag = "--skills-no-discovery";

        /// <summary>Permits the model to run a selected skill's bundled scripts.</summary>
        public const string AllowScriptsFlag = "--skills-allow-exec";

        /// <summary>Caps how many times the model may fetch skill content per turn.</summary>
        public const string MaxRoundsFlag = "--skills-max-rounds";

        /// <summary>Chooses how hard to insist on OS isolation for a skill's scripts.</summary>
        public const string SandboxFlag = "--skills-sandbox";

        /// <summary>Lets a sandboxed skill script reach the network.</summary>
        public const string AllowNetworkFlag = "--skills-allow-network";

        /// <summary>Environment override for <see cref="RootsFlag"/>. Multiple paths are separated by the platform's path separator.</summary>
        public const string RootsEnvVar = "TS_SKILLS_DIR";

        /// <summary>Environment override for <see cref="DisableFlag"/>. Anything but <c>0</c> counts as on.</summary>
        public const string DisableEnvVar = "TS_NO_SKILLS";

        /// <summary>Environment override for <see cref="AllowScriptsFlag"/>. Anything but <c>0</c> counts as on.</summary>
        public const string AllowScriptsEnvVar = "TS_SKILLS_ALLOW_EXEC";

        /// <summary>Environment override for <see cref="MaxRoundsFlag"/>.</summary>
        public const string MaxRoundsEnvVar = "TS_SKILLS_MAX_ROUNDS";

        /// <summary>Environment override for <see cref="SandboxFlag"/> — <c>off</c>, <c>preferred</c> or <c>required</c>.</summary>
        public const string SandboxEnvVar = "TS_SKILLS_SANDBOX";

        /// <summary>Environment override for <see cref="AllowNetworkFlag"/>. Anything but <c>0</c> counts as on.</summary>
        public const string AllowNetworkEnvVar = "TS_SKILLS_ALLOW_NETWORK";

        /// <summary>The directory name used for skills next to a host binary when nothing is configured.</summary>
        public const string DefaultDirectoryName = "skills";

        /// <summary>Whether skills are available at all.</summary>
        public bool Enabled { get; set; } = true;

        /// <summary>Directories to scan, in precedence order.</summary>
        public List<string> Roots { get; } = new();

        /// <summary>Skills selected up front, by name.</summary>
        public List<string> Selected { get; } = new();

        /// <summary>True when the host should print the registry and exit.</summary>
        public bool ListOnly { get; set; }

        /// <summary>
        /// Advertise skills the caller did not select, so the model can notice one that
        /// fits. Off restricts each request to exactly what it named.
        /// </summary>
        public bool Discovery { get; set; } = true;

        /// <summary>
        /// Permit <see cref="SkillTools.RunToolName"/>.
        ///
        /// <para>
        /// Off by default and worth keeping off unless the operator means it: a skill is
        /// content someone uploaded, and running its scripts is arbitrary code execution
        /// on the host, under the host's account, decided by a model reading untrusted
        /// Markdown. It is genuinely useful on a developer's own machine, where the
        /// skills are the developer's own, and that is the case the flag is for.
        /// </para>
        /// </summary>
        public bool AllowScripts { get; set; }

        /// <summary>Round cap for <see cref="SkillAgentLoop"/>.</summary>
        public int MaxRounds { get; set; } = SkillAgentLoopOptions.Default.MaxRounds;

        /// <summary>
        /// True once <see cref="MaxRoundsFlag"/> or <see cref="MaxRoundsEnvVar"/> has set
        /// <see cref="MaxRounds"/>, as opposed to it still holding the default.
        ///
        /// <para>
        /// A host that can also RUN code raises its own default, because writing a
        /// program, running it, reading the traceback and fixing it is a longer activity
        /// than fetching skill files and one counter gates both. It must not raise a
        /// number the operator chose — that flag exists precisely to bound what one
        /// malfunctioning request costs.
        /// </para>
        /// </summary>
        public bool MaxRoundsSpecified { get; set; }

        /// <summary>
        /// The round cap to actually use, given whether this host can also run code.
        ///
        /// <para>
        /// Eight was sized for progressive disclosure — read a skill, read two of its
        /// references, page through a long one — and it is generous for that. Once the
        /// same counter also gates writing a program, running it, reading the traceback
        /// and fixing it, eight is not enough: the failure that prompted this spent
        /// three rounds reading skills, two producing a document and three on a slide
        /// deck it was still debugging when the budget ran out, with a correct program
        /// in the very next generation that was never run.
        /// </para>
        /// <para>
        /// An operator's chosen number is used as given, in both directions.
        /// </para>
        /// </summary>
        public int RoundsFor(bool codeExecutionOffered) =>
            codeExecutionOffered && !MaxRoundsSpecified
                ? Math.Max(MaxRounds, CodeExecutionRounds)
                : MaxRounds;

        /// <summary>Default round cap on a host that offers code execution.</summary>
        public const int CodeExecutionRounds = 24;

        /// <summary>
        /// How hard to insist on OS isolation when <see cref="AllowScripts"/> is on.
        ///
        /// <para>
        /// <see cref="SkillSandboxMode.Required"/> by default, which is what makes
        /// enabling script execution a bounded decision rather than an open one: on a
        /// host with no sandbox the tool refuses and says so, instead of quietly running
        /// uploaded code unconfined.
        /// </para>
        /// </summary>
        public SkillSandboxMode Sandbox { get; set; } = SkillSandboxMode.Required;

        /// <summary>Let a sandboxed script reach the network. Off by default.</summary>
        public bool AllowNetwork { get; set; }

        /// <summary>True when anything was configured, so a host can stay silent otherwise.</summary>
        public bool IsConfigured => Roots.Count > 0 || Selected.Count > 0 || !Enabled || AllowScripts;

        /// <summary>
        /// Read the skills flags out of a host's raw argument list.
        ///
        /// <para>
        /// Unknown arguments are ignored: both hosts parse the rest of their own command
        /// lines themselves, and this only claims the flags it owns.
        /// </para>
        /// </summary>
        /// <exception cref="ArgumentException">
        /// A flag is missing its value, or a value is not usable. Thrown at startup so a
        /// mistyped path fails before a model is loaded rather than on the first request.
        /// </exception>
        public static SkillHostOptions Parse(IReadOnlyList<string>? args)
        {
            var options = new SkillHostOptions();
            if (args == null)
                return options;

            for (int i = 0; i < args.Count; i++)
            {
                string arg = args[i] ?? string.Empty;

                if (Matches(arg, DisableFlag)) { options.Enabled = false; continue; }
                if (Matches(arg, ListFlag)) { options.ListOnly = true; continue; }
                if (Matches(arg, NoDiscoveryFlag)) { options.Discovery = false; continue; }
                if (Matches(arg, AllowScriptsFlag)) { options.AllowScripts = true; continue; }
                if (Matches(arg, AllowNetworkFlag)) { options.AllowNetwork = true; continue; }

                if (TryReadValue(args, ref i, SandboxFlag, out string? sandbox))
                {
                    options.Sandbox = ParseSandboxMode(sandbox!);
                    continue;
                }

                if (TryReadValue(args, ref i, RootsFlag, out string? root))
                {
                    options.Roots.Add(root!);
                    continue;
                }
                if (TryReadValue(args, ref i, SelectFlag, out string? name))
                {
                    options.Selected.Add(name!);
                    continue;
                }
                if (TryReadValue(args, ref i, MaxRoundsFlag, out string? rounds))
                {
                    if (!int.TryParse(rounds, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
                        || parsed < 1 || parsed > 64)
                    {
                        throw new ArgumentException(
                            $"Invalid value for {MaxRoundsFlag}: '{rounds}'. Expected a whole number between 1 and 64.");
                    }
                    options.MaxRounds = parsed;
                    options.MaxRoundsSpecified = true;
                }
            }

            return options;
        }

        /// <summary>
        /// Layer environment variables under whatever the command line supplied, then
        /// fall back to the conventional <c>skills</c> directory next to the binary.
        /// </summary>
        /// <param name="baseDirectory">
        /// Where the host binary lives. The default root is created if missing, so an
        /// operator can drop a skill directory in and restart without any flag at all.
        /// </param>
        public SkillHostOptions ApplyEnvironmentAndDefaults(string baseDirectory)
        {
            if (Environment.GetEnvironmentVariable(DisableEnvVar) is { } disable
                && !string.Equals(disable, "0", StringComparison.Ordinal))
            {
                Enabled = false;
            }

            if (!AllowScripts
                && Environment.GetEnvironmentVariable(AllowScriptsEnvVar) is { } allow
                && !string.Equals(allow, "0", StringComparison.Ordinal))
            {
                AllowScripts = true;
            }

            if (Environment.GetEnvironmentVariable(MaxRoundsEnvVar) is { } roundsText
                && int.TryParse(roundsText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int rounds)
                && rounds >= 1 && rounds <= 64)
            {
                MaxRounds = rounds;
                MaxRoundsSpecified = true;
            }

            if (Roots.Count == 0
                && Environment.GetEnvironmentVariable(RootsEnvVar) is { Length: > 0 } envRoots)
            {
                foreach (string part in envRoots.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
                {
                    string trimmed = part.Trim();
                    if (trimmed.Length > 0)
                        Roots.Add(trimmed);
                }
            }

            if (Environment.GetEnvironmentVariable(SandboxEnvVar) is { Length: > 0 } sandboxText)
                Sandbox = ParseSandboxMode(sandboxText);

            if (!AllowNetwork
                && Environment.GetEnvironmentVariable(AllowNetworkEnvVar) is { } allowNetwork
                && !string.Equals(allowNetwork, "0", StringComparison.Ordinal))
            {
                AllowNetwork = true;
            }

            if (Roots.Count == 0 && !string.IsNullOrWhiteSpace(baseDirectory))
                Roots.Add(Path.Combine(baseDirectory, DefaultDirectoryName));

            return this;
        }

        /// <summary>
        /// Read a sandbox mode name.
        /// </summary>
        /// <exception cref="ArgumentException">
        /// The value names no mode. Deliberately fatal rather than falling back to a
        /// default: a typo in the setting that decides whether uploaded code runs
        /// confined must not silently resolve to the weaker option.
        /// </exception>
        public static SkillSandboxMode ParseSandboxMode(string value) =>
            (value ?? string.Empty).Trim().ToLowerInvariant() switch
            {
                "off" or "none" or "0" => SkillSandboxMode.Off,
                "preferred" or "optional" or "best-effort" => SkillSandboxMode.Preferred,
                "required" or "strict" or "1" => SkillSandboxMode.Required,
                _ => throw new ArgumentException(
                    $"Invalid value for {SandboxFlag}: '{value}'. Expected off, preferred or required."),
            };

        /// <summary>Build the script-runner options these flags describe.</summary>
        public SkillScriptRunnerOptions ToScriptRunnerOptions() => new()
        {
            Sandbox = Sandbox,
            AllowNetwork = AllowNetwork,
        };

        /// <summary>
        /// Check every configured root exists, so a typo fails at startup with a message
        /// naming the flag rather than silently producing an empty registry.
        /// </summary>
        /// <param name="createDefault">
        /// Create a missing root instead of rejecting it. True for the conventional
        /// default directory, which is expected not to exist on a fresh machine; false
        /// for a path the operator typed, where a missing directory is a mistake.
        /// </param>
        /// <exception cref="ArgumentException">A configured root is missing.</exception>
        public void ValidateRoots(bool createDefault = false)
        {
            for (int i = 0; i < Roots.Count; i++)
            {
                string root = Roots[i];
                if (Directory.Exists(root))
                    continue;

                if (createDefault)
                {
                    try
                    {
                        Directory.CreateDirectory(root);
                        continue;
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        throw new ArgumentException(
                            $"Invalid value for {RootsFlag}: '{root}' could not be created ({ex.Message}).");
                    }
                }

                throw new ArgumentException(
                    $"Invalid value for {RootsFlag}: '{root}' is not an existing directory.");
            }
        }

        /// <summary>Build the registry options these flags describe.</summary>
        /// <param name="installDirectory">
        /// Where uploads land. Null leaves the registry read-only, which is what the CLI
        /// wants — it registers directories the user already has rather than copying them.
        /// </param>
        public SkillRegistryOptions ToRegistryOptions(string? installDirectory = null) => new()
        {
            Roots = Roots.ToArray(),
            InstallDirectory = installDirectory,
        };

        private static bool Matches(string arg, string flag) =>
            string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// Read <c>--flag value</c> or <c>--flag=value</c>. Both spellings are accepted
        /// because the server's own option reader accepts both and a config file may
        /// produce either.
        /// </summary>
        private static bool TryReadValue(IReadOnlyList<string> args, ref int index, string flag, out string? value)
        {
            value = null;
            string arg = args[index] ?? string.Empty;

            if (Matches(arg, flag))
            {
                if (index + 1 >= args.Count)
                    throw new ArgumentException($"Missing value for option {flag}.");
                value = args[++index];
                if (string.IsNullOrWhiteSpace(value))
                    throw new ArgumentException($"Missing value for option {flag}.");
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

    /// <summary>
    /// What a model family's chat format can actually carry, as far as skills care.
    /// </summary>
    /// <param name="ToolsRendered">
    /// False when tool declarations never reach the model, so progressive disclosure
    /// is impossible and skill bodies must be written into the prompt instead.
    /// </param>
    /// <param name="ToolResultsRendered">
    /// False when <c>role: "tool"</c> messages are dropped by the renderer, so results
    /// have to be fed back as a user turn.
    /// </param>
    public readonly record struct SkillModelCapabilities(bool ToolsRendered, bool ToolResultsRendered)
    {
        /// <summary>Both halves of the round trip work.</summary>
        public static SkillModelCapabilities Full { get; } = new(true, true);
    }

    /// <summary>
    /// Reads a model family's tool-carrying facts out of the chat-protocol table.
    /// </summary>
    /// <remarks>
    /// Deliberately a lookup rather than a list of architecture names here: those facts
    /// belong to the family's chat protocol, and the registry is the one place a family's
    /// text format is declared. Adding a family with an unusual renderer therefore fixes
    /// skills at the same time it fixes everything else.
    /// </remarks>
    public static class SkillCapabilities
    {
        /// <summary>What skills can rely on for <paramref name="architecture"/>.</summary>
        /// <remarks>
        /// <para>
        /// A tool declaration is only worth writing if the reply can be read back, and the
        /// two halves are decided in different places: the protocol table says whether the
        /// renderer writes declarations, while <see cref="OutputParserFactory"/> decides
        /// what parses the answer. They can disagree. An architecture with no
        /// <c>CreateOutputParser</c> - any family with no table entry at all - gets
        /// <see cref="PassthroughOutputParser"/>, which hands back every byte as content
        /// and never extracts a call.
        /// </para>
        /// <para>
        /// Declaring <c>skills_read</c> to such a model is strictly worse than staying
        /// quiet: the model emits the call, nothing answers it, and the raw tool markup
        /// reaches the user as though it were the answer. So the capability is the AND of
        /// the two halves, and a family that cannot complete the round trip gets its skill
        /// bodies written into the prompt instead - which works.
        /// </para>
        /// </remarks>
        public static SkillModelCapabilities For(string? architecture)
        {
            ChatProtocol? protocol = ChatProtocolRegistry.For(architecture);

            // Whether a `role: "tool"` message survives is a property of the RENDERER
            // alone, and stays independent of whether the model can receive tool
            // declarations. Collapsing the two flags together would make the loop feed
            // results back as user turns on families that do not need that adaptation.
            bool resultsRendered = protocol?.RendersToolResultMessages ?? true;

            // Declarations are the AND of the two halves: the renderer must write the
            // declaration, and the parser that will actually run must be able to read the
            // call back out. The parser is the half the table does not know about - an
            // architecture with no CreateOutputParser gets PassthroughOutputParser, which
            // returns every byte as content and can never produce a tool call.
            bool parserReadsToolCalls = OutputParserFactory.Create(architecture).HasToolSupport;
            bool declarationsRendered = (protocol?.RendersToolDeclarations ?? true) && parserReadsToolCalls;

            return new SkillModelCapabilities(declarationsRendered, resultsRendered);
        }
    }
}
