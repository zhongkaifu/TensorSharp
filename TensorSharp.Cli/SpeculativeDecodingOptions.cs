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
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Cli
{
    /// <summary>
    /// The CLI's speculative-decoding policy: one place where every generation
    /// path (single-shot, JSONL batch, multi-turn, interactive) decides whether a
    /// turn speculates, with which algorithm and window, and — when it cannot —
    /// why not. Which ALGORITHM serves a model is not decided here: that is
    /// <see cref="SpeculatorRegistry"/>, shared with the server's engine path.
    ///
    /// Two kinds of drafter reach this, and they are opted into differently:
    ///
    ///   * a BLOCK drafter that ships as its own GGUF (DeepSeek V4's DSpark,
    ///     Muse-Glimmer's DFlash) is requested by naming the file on
    ///     <c>--draft-model</c>, so its presence in the weights IS the request;
    ///   * a PER-TOKEN NextN/MTP head embedded in the trunk checkpoint (GLM-5.2,
    ///     Qwen 3.6) is always there once the checkpoint is loaded, so it needs an
    ///     explicit <c>--spec</c> — matching the server's opt-in default, and
    ///     for GLM also matching what the loader was told: the native loader only
    ///     pages the ~3 GiB draft layer into VRAM when the speculation env var
    ///     was already set, so a request that arrives after the model is loaded cannot
    ///     be honoured at all.
    /// </summary>
    internal static class SpeculativeDecodingOptions
    {
        /// <summary>Resolved knobs for a run, from the CLI flags and the TS_SPEC_*/TS_MTP_* environment.</summary>
        internal readonly struct Settings
        {
            /// <summary>True when <c>--spec</c> (or <c>TS_SPEC=1</c>) asked for
            /// speculation on a drafter that is not itself an explicit request.</summary>
            public bool Requested { get; init; }

            /// <summary>True when <c>--no-spec</c> (or a resolved speculation
            /// environment variable set to off) explicitly vetoed drafting. This
            /// stays distinct from the default-off state so naming a separate
            /// block drafter can still count as the request.</summary>
            public bool ExplicitlyDisabled { get; init; }

            /// <summary>Cap on tokens drafted per step. Always positive.</summary>
            public int MaxDraftTokens { get; init; }

            /// <summary>True when the operator actually named that cap, so a model
            /// that prefers a narrower DEFAULT window
            /// (ISpeculativeTarget.SpecPreferredDraftWindow) leaves it alone.</summary>
            public bool MaxDraftTokensExplicit { get; init; }

            /// <summary>Draft-confidence gate, or null to let the ALGORITHM apply its own
            /// default — 0.15 for a per-token head, 0.35 for a block drafter, 0 for
            /// n-gram. They threshold different quantities, so there is no shared
            /// default to fall back on and "unset" has to survive all the way
            /// down.</summary>
            public float? MinDraftProb { get; init; }

            /// <summary>The algorithm name the operator chose (<c>--spec-type</c>),
            /// or <c>auto</c> = "whatever drafter the checkpoint carries".</summary>
            public string SpeculatorName { get; init; }

            /// <summary>Whether anything was explicitly configured, for the startup log.</summary>
            public bool AnyExplicit { get; init; }

            /// <summary>The resolved knobs in the form the registry consumes.</summary>
            public SpeculationOptions ToSpeculationOptions() => new()
            {
                Enabled = Requested,
                ExplicitlyDisabled = ExplicitlyDisabled,
                SpeculatorName = string.IsNullOrWhiteSpace(SpeculatorName)
                    ? SpeculatorRegistry.Auto
                    : SpeculatorName,
                MaxDraftTokens = MaxDraftTokens,
                MaxDraftTokensExplicit = MaxDraftTokensExplicit,
                MinDraftProb = MinDraftProb,
            };
        }

        /// <summary>
        /// Resolve the run's speculation knobs from the environment (already
        /// carrying whatever <see cref="SpeculativeCliFlags.Apply"/> translated
        /// from the command line). The parameters are programmatic overrides for
        /// callers that decide per run; the command line has exactly one spelling
        /// per knob (--spec-draft / --spec-pmin), so for a normal CLI run both
        /// arrive unset and the environment decides.
        /// </summary>
        internal static Settings Resolve(int specDraftMax, float specDraftConfMin)
        {
            var cfg = SchedulerConfig.FromEnvironment();
            return new Settings
            {
                Requested = cfg.Speculation.Enabled,
                ExplicitlyDisabled = cfg.Speculation.ExplicitlyDisabled,
                SpeculatorName = cfg.Speculation.SpeculatorName,
                MaxDraftTokens = specDraftMax > 0 ? specDraftMax : Math.Max(1, cfg.Speculation.MaxDraftTokens),
                MaxDraftTokensExplicit = specDraftMax > 0 || cfg.Speculation.MaxDraftTokensExplicit,
                MinDraftProb = specDraftConfMin >= 0f ? specDraftConfMin : cfg.Speculation.MinDraftProb,
                AnyExplicit = cfg.Speculation.Enabled || cfg.Speculation.ExplicitlyDisabled
                              || specDraftMax > 0 || specDraftConfMin >= 0f
                              || cfg.Speculation.MinDraftProb.HasValue
                              || !string.Equals(cfg.Speculation.SpeculatorName, SpeculatorRegistry.Auto,
                                  StringComparison.OrdinalIgnoreCase),
            };
        }

        /// <summary>
        /// The decoder to serve a turn with, or null to decode one token per forward.
        /// <paramref name="declineReason"/> is set (for an operator-facing warning)
        /// only when speculation was ASKED for and could not be given; a model that
        /// simply has no drafter declines silently.
        ///
        /// The run's sampler is deliberately not a factor: verification draws every
        /// emitted token from a trunk row with whatever sampler the caller then
        /// passes to <see cref="SpeculativeDecoder.GenerateSampled"/>, so a
        /// temperature or a penalty changes how the tokens are drawn, not whether
        /// speculation is sound.
        /// </summary>
        /// <param name="existing">A decoder already built for this model, reused
        /// rather than rebuilt. It carries the hidden state pairing the trunk with
        /// the drafter across turns, and its buffers are sized by the vocabulary —
        /// on a 155k-token vocabulary a rebuilt one costs several MB per turn for
        /// nothing.</param>
        internal static SpeculativeDecoder TryCreate(
            ModelBase model, in Settings settings,
            bool hasMediaAttachments, out string declineReason,
            SpeculativeDecoder existing = null)
        {
            declineReason = null;

            if (model is not ISpeculativeTarget spec)
                return null;

            // A drafter that ships INSIDE the checkpoint is not a request: a
            // per-token head is resident in every Qwen 3.6 / GLM-5.2 file, so
            // engaging on its mere presence would silently change what `--input`
            // does. A BLOCK drafter, by contrast, only exists because the
            // operator named its GGUF on --draft-model, so its presence IS the
            // request.
            DraftHeadKind? draftHeadKind = (spec as IDraftHead)?.DraftHeadKind;
            if (!ShouldEngage(draftHeadKind, settings))
                return null;

            // A trunk whose verify cannot reproduce its own decode would emit a
            // different stream than plain greedy decoding: refused outright.
            if (spec.SpeculationRefusal is { } refusal)
            {
                declineReason = refusal;
                return null;
            }

            // Backends whose accelerated verify/draft kernels are missing run the
            // per-op fallback, which does not amortize the trunk over the window.
            if (!spec.SpeculationProfitable)
            {
                declineReason = "the draft head has no accelerated path on this backend, "
                              + "where speculation costs more than it saves.";
                return null;
            }

            // The speculative prefill has no place to queue per-chunk vision/audio
            // embeddings, which are injected by ModelBase.Forward's hook.
            if (hasMediaAttachments)
            {
                declineReason = "the turn carries image/audio/video attachments, "
                              + "whose embeddings only the plain prefill can inject.";
                return null;
            }

            if (existing != null)
                return existing;

            // One place decides which algorithm serves this model, shared with
            // the server's engine path. Window clamping (a block drafter can
            // never exceed its trained block) lives in the speculator itself.
            var speculator = SpeculatorRegistry.Create(
                spec, settings.ToSpeculationOptions(), out declineReason);
            if (speculator == null)
                return null;

            return new SpeculativeDecoder(spec, speculator)
            {
                PrefillChunkSize = spec.SpecPrefillChunkSize > 0 ? spec.SpecPrefillChunkSize : 512,
            };
        }

        /// <summary>
        /// Whether a discovered drafter represents an active request. Extracted
        /// from <see cref="TryCreate"/> so the explicit-off policy can be covered
        /// without loading a multi-gigabyte model in a command-line test.
        /// </summary>
        internal static bool ShouldEngage(DraftHeadKind? draftHeadKind, in Settings settings)
            => !settings.ExplicitlyDisabled
               && (settings.Requested || draftHeadKind == DraftHeadKind.Block);

        /// <summary>How the turn drafts, for logs.</summary>
        internal static string DescribeDrafter(SpeculativeDecoder decoder)
            => decoder.Speculator.Describe();

        /// <summary>
        /// How a turn verifies: argmax keeps the greedy stream exactly, anything
        /// else draws each emitted token with the run's own sampler.
        /// </summary>
        internal static string DescribeVerification(SamplingConfig sampling)
            => InteractiveSession.IsArgmaxSampling(sampling) ? "argmax" : "sampled";
    }
}
