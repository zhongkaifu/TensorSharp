// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Translation of the speculative-decoding command-line flags into the
    /// environment variables <see cref="SpeculationOptions.FromEnvironment"/>
    /// and the model loaders read.
    ///
    /// The env var - not a parsed value object - is the contract here because
    /// the request has to reach places the flags cannot: the glm-dsa NATIVE
    /// loader sizes its graph cache from <c>TS_MTP_DRAFT</c> - from C++, while
    /// the model is loading, long before any decoder object exists - and the
    /// managed side of the same loader reads <c>TS_MTP_SPEC</c> to decide
    /// whether to page the NextN block into VRAM at all (a whole extra
    /// 256-expert decoder layer). Hosts must therefore apply these BEFORE they
    /// construct the model, and every value is still published under BOTH the
    /// current <c>TS_SPEC_*</c> name and the legacy <c>TS_MTP_*</c> one the
    /// native loader reads. Only the ENV spellings are dual: the flag surface
    /// is one name per concept, and a removed spelling errors with a pointer
    /// to its replacement instead of being accepted or silently ignored.
    ///
    /// Shared by <c>TensorSharp.Server</c> and <c>TensorSharp.Cli</c> so the two
    /// hosts cannot drift on flag names, validation or defaults.
    /// </summary>
    public static class SpeculativeCliFlags
    {
        /// <summary>Set to <c>1</c>/<c>0</c> by <c>--spec</c>/<c>--no-spec</c>.</summary>
        public const string SpecEnvVar = SpeculationEnvVars.LegacyEnabled;

        /// <summary>Maximum tokens drafted per speculative step (<c>--spec-draft</c>).</summary>
        public const string DraftEnvVar = SpeculationEnvVars.LegacyDraft;

        /// <summary>Minimum draft confidence to keep a drafted token (<c>--spec-pmin</c>).</summary>
        public const string PMinEnvVar = SpeculationEnvVars.LegacyPMin;

        /// <summary>Draft GGUF the operator named on <c>--draft-model</c>, for the
        /// attach-after-load path (a per-token head such as Gemma 4's assistant, or
        /// a DFlash drafter picked up on a runtime model switch).</summary>
        public const string DraftModelEnvVar = SpeculationEnvVars.LegacyDraftModel;

        /// <summary>Speculation algorithm (<c>--spec-type</c>).</summary>
        public const string TypeEnvVar = SpeculationEnvVars.Type;

        /// <summary>
        /// Every valueless switch <see cref="Apply"/> consumes.
        ///
        /// This exists because the flags are applied in a pass SEPARATE from the
        /// host's own argument parse, and that pass does not REMOVE what it
        /// consumes: TensorSharp.Server then walks the same argv and throws
        /// "Unknown option" for anything it does not recognise. Two hand-written
        /// lists of the same flag names is a drift bug waiting to happen, and it
        /// happened - the server knew only older spellings, so every documented
        /// <c>--spec*</c> flag made it refuse to start. Hosts MUST consume these
        /// tables rather than re-typing the names.
        /// </summary>
        public static readonly string[] SwitchFlags =
        {
            "--spec", "--no-spec",
        };

        /// <summary>Every <c>--flag VALUE</c> option <see cref="Apply"/> consumes.
        /// Longer names come first so a prefix match can never swallow a longer
        /// flag's value. See <see cref="SwitchFlags"/> for why this table exists.</summary>
        public static readonly string[] ValueFlags =
        {
            "--spec-draft", "--spec-type", "--spec-pmin", "--draft-model",
        };

        /// <summary>
        /// Spellings that used to be accepted and were removed because each
        /// duplicated a surviving flag, mapped to its replacement. There used to
        /// be up to three names per concept (<c>--spec-draft</c>,
        /// <c>--mtp-draft</c> and <c>--spec-draft-n-max</c> all set the same
        /// value), which left operators guessing which one was real.
        /// <see cref="Apply"/> rejects these with a pointer to the survivor -
        /// a hard error, never a silent ignore, because the CLI's argument
        /// switch drops unknown flags and "speculation quietly off" is exactly
        /// the failure this table exists to prevent.
        /// </summary>
        public static readonly (string Flag, string Survivor)[] RemovedFlags =
        {
            ("--mtp-spec", "--spec"),
            ("--no-mtp-spec", "--no-spec"),
            ("--mtp-draft", "--spec-draft"),
            ("--mtp-pmin", "--spec-pmin"),
            ("--mtp-type", "--spec-type"),
            ("--mtp-draft-model", "--draft-model"),
            ("--spec-draft-model", "--draft-model"),
            ("--spec-draft-n-max", "--spec-draft"),
            ("--spec-draft-conf-min", "--spec-pmin"),
        };

        /// <summary>Largest accepted draft window; see
        /// <see cref="SpeculationOptions.MaxAllowedDraftTokens"/>.</summary>
        public const int MaxDraftTokens = SpeculationOptions.MaxAllowedDraftTokens;

        /// <summary>
        /// Apply the speculative-decoding flags from <paramref name="args"/> to
        /// the process environment. Both <c>--opt V</c> and <c>--opt=V</c>
        /// spellings are accepted:
        ///
        /// <code>
        ///   --spec | --no-spec       explicit on/off for a drafter embedded in the checkpoint
        ///   --spec-type NAME         auto | draft-head | block | ngram (default: auto)
        ///   --spec-draft N           draft window
        ///   --spec-pmin X            draft-confidence gate in [0, 1]; 0 = never gate
        ///   --draft-model PATH       a drafter that ships as its own GGUF; naming it IS the request
        /// </code>
        ///
        /// Naming a file on <c>--draft-model</c> enables speculation by itself -
        /// the file only exists on the command line because the operator wants it
        /// used - unless an explicit <c>--no-spec</c> vetoes it. Which KIND of
        /// drafter the file is (a DFlash/DSpark block drafter fused before the
        /// layer split, or a per-token head attached after load) is read from the
        /// GGUF itself by the loaders, never asked of the operator.
        ///
        /// Returns true when at least one flag was applied, so the caller can
        /// emit a startup log line.
        /// </summary>
        /// <exception cref="ArgumentException">A flag carried a missing or unusable
        /// value, or a removed spelling was used (the error names the replacement).</exception>
        public static bool Apply(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            RejectRemoved(args);

            bool changed = false;
            bool explicitOnOff = false;
            bool draftModelNamed = false;
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                if (IsFlag(a, "--spec"))
                {
                    SetBoth(SpeculationEnvVars.Enabled, SpeculationEnvVars.LegacyEnabled, "1");
                    explicitOnOff = true;
                    changed = true;
                    continue;
                }
                if (IsFlag(a, "--no-spec"))
                {
                    SetBoth(SpeculationEnvVars.Enabled, SpeculationEnvVars.LegacyEnabled, "0");
                    explicitOnOff = true;
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--spec-type", out string? typeOpt))
                {
                    if (!SpeculatorRegistry.IsKnown(typeOpt))
                    {
                        throw new ArgumentException(
                            $"Invalid value for --spec-type: '{typeOpt}'. Expected one of: "
                            + $"{SpeculatorRegistry.Auto}, {string.Join(", ", SpeculatorRegistry.Names)}.");
                    }
                    Environment.SetEnvironmentVariable(SpeculationEnvVars.Type, typeOpt.Trim());
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--spec-draft", out string? draftOpt))
                {
                    if (!int.TryParse(draftOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int draft)
                        || draft < 1 || draft > MaxDraftTokens)
                    {
                        throw new ArgumentException(
                            $"Invalid value for --spec-draft: '{draftOpt}'. Expected an integer in [1, {MaxDraftTokens}].");
                    }
                    SetBoth(SpeculationEnvVars.Draft, SpeculationEnvVars.LegacyDraft,
                        draft.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--spec-pmin", out string? pminOpt))
                {
                    if (!float.TryParse(pminOpt, NumberStyles.Float, CultureInfo.InvariantCulture, out float pmin)
                        || !float.IsFinite(pmin)
                        || pmin < 0f || pmin > 1f)
                    {
                        throw new ArgumentException(
                            $"Invalid value for --spec-pmin: '{pminOpt}'. Expected a probability in [0, 1] "
                            + "(0 disables the confidence gate).");
                    }
                    SetBoth(SpeculationEnvVars.PMin, SpeculationEnvVars.LegacyPMin,
                        pmin.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                // Path to a drafter that ships as its own GGUF: DeepSeek V4's
                // DSpark, the DFlash / DFlash2 drafters for Muse-Glimmer and
                // Qwen 3.8, or Gemma 4's per-token assistant head. Qwen 3.6 and
                // GLM-5.2 embed their NextN block in the trunk GGUF and need no
                // such flag. Published for the attach-after-load path here; the
                // hosts additionally hand it to the model factory for the
                // drafters that must be resident before the layer split - and
                // TryAttachConfiguredDraftHead skips a drafter the factory
                // already attached, so publishing both ways cannot double-load.
                if (TryReadOption(args, ref i, "--draft-model", out string? draftModelOpt))
                {
                    if (string.IsNullOrWhiteSpace(draftModelOpt) || !File.Exists(draftModelOpt))
                        throw new ArgumentException($"--draft-model file not found: '{draftModelOpt}'.");
                    SetBoth(SpeculationEnvVars.DraftModel, SpeculationEnvVars.LegacyDraftModel, draftModelOpt);
                    draftModelNamed = true;
                    changed = true;
                    continue;
                }
            }

            // Naming a draft file IS the request: nobody passes --draft-model
            // hoping it stays idle, and requiring a separate --spec beside it
            // was a trap both hosts fell into differently (the CLI engaged, the
            // server silently did not). An explicit --spec/--no-spec anywhere on
            // the line still wins - the operator said so in words.
            if (draftModelNamed && !explicitOnOff)
                SetBoth(SpeculationEnvVars.Enabled, SpeculationEnvVars.LegacyEnabled, "1");

            return changed;
        }

        /// <summary>
        /// The startup warning for a line whose speculative flags only TUNE speculation
        /// (<c>--spec-type</c>, <c>--spec-draft</c>, <c>--spec-pmin</c>) while nothing turns
        /// it on - neither <c>--spec</c> nor <c>--draft-model</c>, and <c>TS_SPEC</c> unset.
        /// Null when there is nothing to say: no speculative flag was applied
        /// (<paramref name="flagsApplied"/> is <see cref="Apply"/>'s result), speculation is
        /// on, or it was turned off in words (<c>--no-spec</c>, <c>TS_SPEC=0</c>).
        /// </summary>
        /// <remarks>
        /// Both hosts log it: <c>--spec-type ngram</c> alone asked for something that will
        /// never run, and the engine's own plan line then says "off (not requested)", the
        /// opposite of what the operator typed.
        /// </remarks>
        public static string? DescribeInertTuning(bool flagsApplied, SpeculationOptions? options)
        {
            if (!flagsApplied || options == null || options.Enabled || options.ExplicitlyDisabled)
                return null;
            return "Speculative decoding stays OFF: --spec-type, --spec-draft and --spec-pmin only tune it, and "
                + "neither --spec nor --draft-model was given (TS_SPEC is unset). Add --spec to turn it on; "
                + "generation uses plain decoding meanwhile.";
        }

        /// <summary>
        /// Throw for any removed spelling in <paramref name="args"/>, naming its
        /// replacement. Called by <see cref="Apply"/>, so both hosts get it for
        /// free; public so a host that parses independently can reuse it.
        /// </summary>
        /// <exception cref="ArgumentException">A removed spelling was present.</exception>
        public static void RejectRemoved(string[] args)
        {
            if (args == null)
                return;
            foreach (string arg in args)
            {
                foreach ((string flag, string survivor) in RemovedFlags)
                {
                    if (string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase)
                        || arg.StartsWith(flag + "=", StringComparison.OrdinalIgnoreCase))
                    {
                        throw new ArgumentException(
                            $"{flag} was removed; use {survivor} instead. "
                            + "One name per option now - see the Speculative decoding section of --help.");
                    }
                }
            }
        }

        /// <summary>
        /// Reads <c>--opt VALUE</c> or <c>--opt=VALUE</c> at <paramref name="index"/>,
        /// advancing past a consumed value token.
        /// </summary>
        public static bool TryReadOption(string[] args, ref int index, string option, [NotNullWhen(true)] out string? value)
        {
            string arg = args[index];
            if (string.Equals(arg, option, StringComparison.OrdinalIgnoreCase))
            {
                if (index + 1 >= args.Length)
                    throw new ArgumentException($"Missing value for option '{option}'.");

                value = args[++index];
                return true;
            }

            string prefix = option + "=";
            if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                value = arg.Substring(prefix.Length);
                return true;
            }

            value = null;
            return false;
        }

        private static bool IsFlag(string arg, string option)
            => string.Equals(arg, option, StringComparison.OrdinalIgnoreCase);

        /// <summary>Publish under both spellings: managed readers prefer
        /// <c>TS_SPEC_*</c>, the glm-dsa native loader only knows
        /// <c>TS_MTP_*</c>.</summary>
        private static void SetBoth(string name, string legacyName, string value)
        {
            Environment.SetEnvironmentVariable(name, value);
            Environment.SetEnvironmentVariable(legacyName, value);
        }
    }
}
