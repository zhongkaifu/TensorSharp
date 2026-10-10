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
using System.Globalization;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// The operator's speculative-decoding request, resolved once and passed
    /// down: whether to speculate, with which algorithm, how wide a window and
    /// what confidence gate. Hosts (CLI, server) build one of these; nothing
    /// below reads the environment again.
    /// </summary>
    public sealed record SpeculationOptions
    {
        /// <summary>Speculative decoding requested. Default OFF - a per-token
        /// checkpoint may carry a per-token head, but engaging by its mere
        /// presence would silently change what a plain run does.
        /// CLI: <c>--spec</c>; env: <c>TS_SPEC</c>.</summary>
        public bool Enabled { get; init; }

        /// <summary>
        /// True when the operator supplied the enable variable and its resolved
        /// value is off. This is distinct from the default-off state: a separate
        /// block-drafter GGUF normally counts as an opt-in by its presence, but an
        /// explicit <c>--no-spec</c> / <c>TS_SPEC=0</c> must veto that convenience.
        /// </summary>
        public bool ExplicitlyDisabled { get; init; }

        /// <summary>Which algorithm to use; see <see cref="SpeculatorRegistry"/>.
        /// Default <see cref="SpeculatorRegistry.Auto"/> = "whatever drafter the
        /// checkpoint carries". CLI: <c>--spec-type</c>; env:
        /// <c>TS_SPEC_TYPE</c>.</summary>
        public string SpeculatorName { get; init; } = SpeculatorRegistry.Auto;

        /// <summary>Maximum tokens drafted per speculative step (llama.cpp
        /// n_max). CLI: <c>--spec-draft</c>; env: <c>TS_SPEC_DRAFT</c>.</summary>
        public int MaxDraftTokens { get; init; } = DefaultMaxDraftTokens;

        /// <summary>
        /// True when the operator actually asked for <see cref="MaxDraftTokens"/>,
        /// rather than inheriting the default. A model whose trunk makes a wide
        /// window expensive can narrow the DEFAULT
        /// (<see cref="ISpeculativeTarget.SpecPreferredDraftWindow"/>); it must not
        /// silently override a number the operator typed.
        /// </summary>
        public bool MaxDraftTokensExplicit { get; init; }

        /// <summary>
        /// Confidence gate, or null to let the algorithm pick its own
        /// (<see cref="ISpeculator.DefaultMinDraftProb"/>). The gates threshold
        /// DIFFERENT quantities per algorithm, so one shared default cannot
        /// serve them all - leave this unset unless the operator asked for a
        /// specific value. CLI: <c>--spec-pmin</c>; env: <c>TS_SPEC_PMIN</c>.
        /// </summary>
        public float? MinDraftProb { get; init; }

        /// <summary>Default draft window: 8. A confidence gate stops drafting at
        /// the first low-confidence token, so a longer window only extends
        /// confident streaks - measured 1.21x vs 1.08x (window 4) on
        /// Qwen3.6-35B-A3B ggml_cpu at unchanged 86% acceptance; neutral on 27B
        /// ggml_cuda.</summary>
        public const int DefaultMaxDraftTokens = 8;

        /// <summary>
        /// Largest accepted draft window. Bounded because the window is not
        /// free on either side of the boundary: the draft/verify buffers are
        /// <c>(N+1) x vocab</c> floats (on a 155k-token vocabulary, 40 MB at 64),
        /// and the glm-dsa native loader ignores anything above this when it
        /// sizes its graph cache from the same variable - so a larger window
        /// would decode through a cache too small for the graph shapes it
        /// produces, rebuilding one every step.
        /// </summary>
        public const int MaxAllowedDraftTokens = 64;

        /// <summary>Speculation off.</summary>
        public static SpeculationOptions Disabled => new();

        /// <summary>
        /// Read the <c>TS_SPEC_*</c> environment (the glm-dsa native loader reads
        /// <c>TS_SPEC_DRAFT</c> from C++ as well, to size its graph cache).
        /// </summary>
        /// <exception cref="ArgumentException">A removed variable is set
        /// (<see cref="SpeculationEnvVars.RejectRemoved"/>).</exception>
        public static SpeculationOptions FromEnvironment()
        {
            SpeculationEnvVars.RejectRemoved();
            string? enabledRaw = ReadString(SpeculationEnvVars.Enabled);
            int maxDraftTokens = ReadDraftTokens(SpeculationEnvVars.Draft, out bool maxDraftTokensExplicit);
            return new SpeculationOptions
            {
                Enabled = ReadBool(enabledRaw, false),
                ExplicitlyDisabled = enabledRaw != null && !ReadBool(enabledRaw, false),
                SpeculatorName = ReadString(SpeculationEnvVars.Type) ?? SpeculatorRegistry.Auto,
                MaxDraftTokens = maxDraftTokens,
                // An invalid env value is ignored rather than becoming an
                // explicit request for the fallback window.
                MaxDraftTokensExplicit = maxDraftTokensExplicit,
                MinDraftProb = ReadFloatOrNull(SpeculationEnvVars.PMin),
            };
        }

        private static string? ReadString(string name)
        {
            string? raw = Environment.GetEnvironmentVariable(name);
            return string.IsNullOrWhiteSpace(raw) ? null : raw.Trim();
        }

        private static bool ReadBool(string? raw, bool fallback)
        {
            if (raw == null)
                return fallback;
            return raw is "1" or "true" or "TRUE" or "True" or "yes" or "on";
        }

        private static int ReadDraftTokens(string name, out bool explicitlyConfigured)
        {
            string? raw = ReadString(name);
            if (raw != null
                && int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
                && v >= 1
                && v <= MaxAllowedDraftTokens)
            {
                explicitlyConfigured = true;
                return v;
            }

            explicitlyConfigured = false;
            return DefaultMaxDraftTokens;
        }

        private static float? ReadFloatOrNull(string name)
        {
            // Zero is a real value, not "unset": --spec-pmin 0 means "never gate a
            // draft on confidence", which the removed --spec-draft-conf-min spelling
            // could express and its survivor must keep expressing.
            string? raw = ReadString(name);
            return raw != null
                   && float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float v)
                   && float.IsFinite(v)
                   && v >= 0f && v <= 1f
                ? v
                : null;
        }
    }

    /// <summary>
    /// The environment-variable contract. Named constants rather than string
    /// literals because these are read from THREE languages/layers - managed
    /// hosts, the model loaders, and the glm-dsa native C++ loader (which reads
    /// <see cref="Draft"/>) - and a typo in one of them shows up only as
    /// speculation silently not engaging.
    /// </summary>
    public static class SpeculationEnvVars
    {
        /// <summary>Preferred: <c>1</c>/<c>0</c>, set by <c>--spec</c> / <c>--no-spec</c>.</summary>
        public const string Enabled = "TS_SPEC";

        /// <summary>Algorithm name (see <see cref="SpeculatorRegistry"/>).</summary>
        public const string Type = "TS_SPEC_TYPE";

        /// <summary>Maximum tokens drafted per speculative step.</summary>
        public const string Draft = "TS_SPEC_DRAFT";

        /// <summary>Draft-confidence gate.</summary>
        public const string PMin = "TS_SPEC_PMIN";

        /// <summary>Separate draft-head GGUF for architectures that ship one.</summary>
        public const string DraftModel = "TS_SPEC_DRAFT_MODEL";

        /// <summary>
        /// Names that used to be read beside the ones above, mapped to their
        /// replacement. One name per setting: a removed one is an error rather
        /// than a silent no-op, since "speculation quietly off" is exactly how a
        /// stale deployment would otherwise fail.
        /// </summary>
        public static readonly (string Name, string Survivor)[] RemovedNames =
        {
            ("TS_MTP_SPEC", Enabled),
            ("TS_MTP_DRAFT", Draft),
            ("TS_MTP_PMIN", PMin),
            ("TS_MTP_DRAFT_MODEL", DraftModel),
            // The per-family drafter paths: --draft-model reaches every model factory.
            ("TS_DSV4_DSPARK", DraftModel),
            ("TS_QWEN35_DFLASH", DraftModel),
            ("TS_MUSE_GLIMMER_DFLASH", DraftModel),
            ("TS_NEMOTRON_DFLASH", DraftModel),
        };

        /// <summary>Throw when a removed variable is set, naming its replacement.</summary>
        /// <exception cref="ArgumentException">A removed variable is set.</exception>
        public static void RejectRemoved()
        {
            foreach ((string name, string survivor) in RemovedNames)
            {
                if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable(name)))
                    throw new ArgumentException($"{name} was removed; set {survivor} instead.");
            }
        }
    }
}
