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
using System.Linq;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Name -> algorithm. The single place a speculation algorithm becomes
    /// reachable from a command line, and the reason adding one does not touch
    /// any model, executor or scheduler code:
    ///
    /// <code>
    /// SpeculatorRegistry.Register("eagle", (target, opts) =>
    ///     target is IDraftHead h &amp;&amp; h.DraftHeadKind == DraftHeadKind.PerToken
    ///         ? new EagleSpeculator(h, target.Config.VocabSize, target.SpecFeatureSize, opts.MaxDraftTokens)
    ///         : null);
    /// </code>
    ///
    /// A factory returns null when the algorithm cannot serve THIS model, and
    /// <see cref="Create"/> turns that into an operator-facing decline reason.
    /// </summary>
    public static class SpeculatorRegistry
    {
        /// <summary>Use whatever drafter the loaded checkpoint carries - a
        /// per-token head, a block drafter, or nothing. The default.</summary>
        public const string Auto = "auto";

        /// <summary>Per-token learned head chaining its own hidden state
        /// (NextN/MTP, EAGLE-shaped). See <see cref="DraftHeadSpeculator"/>.</summary>
        public const string DraftHead = "draft-head";

        /// <summary>Semi-autoregressive block drafter (DSpark, DFlash).
        /// See <see cref="BlockDraftSpeculator"/>.</summary>
        public const string Block = "block";

        /// <summary>Weight-free suffix matching over the sequence's own tokens.
        /// Works on every model. See <see cref="NGramSpeculator"/>.</summary>
        public const string NGram = "ngram";

        /// <summary>Builds an algorithm for one (target model, options) pair, or
        /// returns null when it cannot serve that model.</summary>
        public delegate ISpeculator Factory(ISpeculativeTarget target, SpeculationOptions options);

        /// <summary>A registered algorithm: how to build it, and whether it
        /// needs the checkpoint to carry learned speculator weights. The flag is
        /// what lets the execution planner explain "no draft head" up front
        /// instead of routing a request onto a path that will decline.</summary>
        private readonly record struct Entry(Factory Factory, bool RequiresDraftHead);

        private static readonly Dictionary<string, Entry> Factories =
            new(StringComparer.OrdinalIgnoreCase)
            {
                [DraftHead] = new Entry(CreateDraftHead, RequiresDraftHead: true),
                [Block] = new Entry(CreateBlock, RequiresDraftHead: true),
                // N-gram may use a separate default: long exact suffix matches can
                // amortize a wide verify without a learned drafter's per-token cost.
                // Otherwise it shares the trunk's general default preference.
                [NGram] = new Entry(
                    (target, options) => new NGramSpeculator(ResolveNGramDraftWindow(target, options)),
                    RequiresDraftHead: false),
            };

        /// <summary>Every registered algorithm name, for the usage page and for
        /// validating a <c>--spec-type</c> value.</summary>
        public static IReadOnlyCollection<string> Names
        {
            get { lock (Factories) return Factories.Keys.OrderBy(n => n, StringComparer.Ordinal).ToArray(); }
        }

        /// <summary>Register (or replace) an algorithm. Call before any model is
        /// loaded; the CLI/server resolve the name at model-load time.</summary>
        /// <param name="requiresDraftHead">True when the algorithm can only run
        /// on a checkpoint that carries learned speculator weights, so the
        /// planner can decline early with a reason instead of routing a request
        /// onto a path that will bail.</param>
        public static void Register(string name, Factory factory, bool requiresDraftHead = true)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Speculator name must not be empty.", nameof(name));
            ArgumentNullException.ThrowIfNull(factory);
            if (string.Equals(name, Auto, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException($"'{Auto}' is reserved.", nameof(name));
            lock (Factories) Factories[name] = new Entry(factory, requiresDraftHead);
        }

        /// <summary>
        /// True when <paramref name="name"/> can only run on a checkpoint that
        /// ships learned speculator weights. <see cref="Auto"/> does (it means
        /// "use the checkpoint's own drafter"); <see cref="NGram"/> does not.
        /// An unknown name answers false, leaving the decision to
        /// <see cref="Create"/>, which reports it properly.
        /// </summary>
        public static bool RequiresDraftHead(string name)
        {
            if (string.IsNullOrWhiteSpace(name)
                || string.Equals(name.Trim(), Auto, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
            lock (Factories)
                return Factories.TryGetValue(name.Trim(), out var e) && e.RequiresDraftHead;
        }

        /// <summary>True when <paramref name="name"/> names a registered
        /// algorithm (or <see cref="Auto"/>).</summary>
        public static bool IsKnown(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return false;
            if (string.Equals(name, Auto, StringComparison.OrdinalIgnoreCase))
                return true;
            lock (Factories) return Factories.ContainsKey(name);
        }

        /// <summary>
        /// The speculator to serve <paramref name="target"/> with, or null with
        /// an operator-facing <paramref name="declineReason"/>. A model that
        /// simply has no drafter under <see cref="Auto"/> declines with a reason
        /// that names the weight-free alternative.
        /// </summary>
        public static ISpeculator Create(ISpeculativeTarget target, SpeculationOptions options,
            out string declineReason)
        {
            ArgumentNullException.ThrowIfNull(target);
            options ??= SpeculationOptions.Disabled;
            declineReason = null;

            // A correctness refusal outranks every algorithm choice: no speculator,
            // learned or weight-free, may run on a trunk whose verify diverges.
            if (target.SpeculationRefusal is { } refusal)
            {
                declineReason = refusal;
                return null;
            }

            string name = string.IsNullOrWhiteSpace(options.SpeculatorName)
                ? Auto
                : options.SpeculatorName.Trim();

            if (string.Equals(name, Auto, StringComparison.OrdinalIgnoreCase))
            {
                name = (target as IDraftHead)?.DraftHeadKind switch
                {
                    DraftHeadKind.PerToken => DraftHead,
                    DraftHeadKind.Block => Block,
                    _ => null,
                };
                if (name == null)
                {
                    declineReason =
                        "the loaded checkpoint carries no draft head (see the loader's message above; a "
                        + "trunk-only requantization and --tp with a borrowed LM head both land here). "
                        + $"Weight-free speculation is available with --spec-type {NGram}.";
                    return null;
                }
            }

            Entry entry;
            lock (Factories)
            {
                if (!Factories.TryGetValue(name, out entry))
                {
                    declineReason = $"unknown speculation algorithm '{name}' "
                                    + $"(known: {string.Join(", ", Names)}).";
                    return null;
                }
            }

            ISpeculator speculator = entry.Factory(target, options);
            if (speculator == null)
            {
                declineReason = $"the '{name}' speculator cannot serve this model "
                                + "(it needs a draft head the loaded weights do not provide).";
                return null;
            }

            if (options.MinDraftProb.HasValue)
                speculator.MinDraftProb = options.MinDraftProb.Value;
            return speculator;
        }

        private static ISpeculator CreateDraftHead(ISpeculativeTarget target, SpeculationOptions options)
        {
            if (target is not IDraftHead head || head.DraftHeadKind != DraftHeadKind.PerToken)
                return null;
            return new DraftHeadSpeculator(head, target.Config.VocabSize, target.SpecFeatureSize,
                ResolveDraftWindow(target, options));
        }

        private static ISpeculator CreateBlock(ISpeculativeTarget target, SpeculationOptions options)
        {
            if (target is not IDraftHead head || head.DraftHeadKind != DraftHeadKind.Block)
                return null;
            int block = head.DraftBlockSize;
            if (block < 1)
                return null;
            return new BlockDraftSpeculator(head, block, ResolveDraftWindow(target, options));
        }

        /// <summary>
        /// The window to draft with: what the operator asked for, or - when they
        /// asked for nothing - narrowed to what the trunk prefers. See
        /// <see cref="ISpeculativeTarget.SpecPreferredDraftWindow"/> for why the
        /// preference belongs to the target model and not to the algorithm.
        /// </summary>
        private static int ResolveDraftWindow(ISpeculativeTarget target, SpeculationOptions options)
        {
            int window = Math.Max(1, options.MaxDraftTokens);
            int preferred = target.SpecPreferredDraftWindow;
            if (!options.MaxDraftTokensExplicit && preferred > 0)
                window = Math.Min(window, preferred);
            return window;
        }

        private static int ResolveNGramDraftWindow(ISpeculativeTarget target, SpeculationOptions options)
        {
            if (!options.MaxDraftTokensExplicit && target.SpecPreferredNGramDraftWindow > 0)
            {
                int preferred = target.SpecPreferredNGramDraftWindow;
                // Older programmatic callers may set a non-default cap without
                // the CLI's Explicit flag. Preserve that cap; only widen the
                // unchanged shared default when the model recommends doing so.
                if (options.MaxDraftTokens != SpeculationOptions.DefaultMaxDraftTokens)
                    preferred = Math.Min(Math.Max(1, options.MaxDraftTokens), preferred);
                return Math.Clamp(preferred, 1, SpeculationOptions.MaxAllowedDraftTokens);
            }
            return ResolveDraftWindow(target, options);
        }
    }
}
