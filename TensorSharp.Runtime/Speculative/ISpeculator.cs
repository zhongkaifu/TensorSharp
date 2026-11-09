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

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Everything a speculator is given to propose the next few tokens. Passing
    /// one struct rather than a parameter list is what keeps the algorithm
    /// contract stable: an algorithm that needs a signal none of today's need
    /// (a tree of candidates, a retrieval index, the prompt) gets a new field
    /// here, and the existing speculators keep compiling untouched.
    /// </summary>
    public readonly struct DraftContext
    {
        /// <summary>The token at <see cref="Position"/>. It is NOT yet in the
        /// trunk's KV cache - the verify batch is [LastToken, d1..dK].</summary>
        public int LastToken { get; init; }

        /// <summary>Trunk tokens already committed to the KV cache.</summary>
        public int Position { get; init; }

        /// <summary>Hard cap on this step's window: the smaller of the
        /// speculator's own maximum, the caller's remaining token budget, the
        /// KV blocks the scheduler reserved and the context headroom. Always
        /// at least 1 when Propose is called at all.</summary>
        public int MaxTokens { get; init; }

        /// <summary>
        /// Trunk hidden state of the token BEFORE <see cref="LastToken"/>
        /// (llama.cpp's pending_h; zeros before the first prompt token), which
        /// a learned drafter chains from. Null for speculators that report
        /// <see cref="ISpeculator.NeedsHiddenState"/> = false, and never read
        /// by them.
        /// </summary>
        public float[]? CarryHidden { get; init; }

        /// <summary>
        /// Every token the sequence has emitted so far, oldest first. This is
        /// what a weight-free speculator (n-gram / suffix matching) mines, and
        /// what a learned one passes to <see cref="AdjustLogits"/> so its
        /// proposals are drawn from the same penalized distribution
        /// verification will draw from. May be null when the caller keeps no
        /// history (pure argmax runs with no penalties).
        /// </summary>
        public IReadOnlyList<int>? History { get; init; }

        /// <summary>
        /// Optional hook applied to a learned drafter's raw logits before its
        /// argmax, given the drafts already pending in THIS window. The caller
        /// uses it to replay its sampler's repetition/presence/frequency
        /// penalties, so drafting argmaxes the SAME distribution verification
        /// draws from - without it, penalized configs (the chat default
        /// repPen 1.1, including --temperature 0) diverge ever more often as
        /// the output history grows and acceptance decays to ~0.
        /// </summary>
        public Action<float[], IReadOnlyList<int>>? AdjustLogits { get; init; }
    }

    /// <summary>
    /// Layer 2 - a SPECULATION ALGORITHM. The one thing every speculative
    /// decoding method has in common: given where the sequence is, propose a
    /// few tokens the trunk is then asked to confirm.
    ///
    /// Implementations must be model-agnostic. Anything that varies per target
    /// model belongs behind <see cref="IDraftHead"/> (learned weights) or
    /// <see cref="ISpeculativeTarget"/> (the trunk), both of which are injected
    /// at construction. Adding EAGLE, Medusa, PARD or a tree-drafting variant
    /// is a new class here plus a line in <see cref="SpeculatorRegistry"/> -
    /// no model, executor or scheduler code changes.
    ///
    /// Correctness is NOT this interface's responsibility: whatever a
    /// speculator proposes, <see cref="SpeculativeExecution"/> emits only
    /// tokens drawn from a TRUNK row, so a bad proposal costs throughput and
    /// nothing else. A speculator is free to be wrong; it must not be slow.
    /// </summary>
    public interface ISpeculator : IDisposable
    {
        /// <summary>Stable identifier, matching the name it is registered under
        /// (<c>draft-head</c>, <c>block</c>, <c>ngram</c>). Appears in logs.</summary>
        string Name { get; }

        /// <summary>Human-readable one-liner for the startup log, e.g.
        /// <c>block(15)</c> or <c>ngram(n=3..5, window=1024)</c>.</summary>
        string Describe() => Name;

        /// <summary>Largest window this speculator will ever propose. A block
        /// drafter clamps this to its trained block size.</summary>
        int MaxDraftTokens { get; }

        /// <summary>
        /// Confidence gate below which drafting stops. What the number MEANS is
        /// the algorithm's business (a per-token head thresholds the top-1
        /// probability over its top-10 logits; a block drafter thresholds the
        /// CUMULATIVE prefix probability; n-gram thresholds a match score), so
        /// each algorithm also supplies its own <see cref="DefaultMinDraftProb"/>
        /// rather than sharing one that would badly mis-gate the others.
        /// </summary>
        float MinDraftProb { get; set; }

        /// <summary>The gate this algorithm uses when the operator did not
        /// choose one.</summary>
        float DefaultMinDraftProb { get; }

        /// <summary>
        /// True when <see cref="Propose"/> reads <see cref="DraftContext.CarryHidden"/>
        /// and <see cref="Commit"/> needs per-row hidden states. False for a
        /// weight-free speculator, which lets the loop skip the trunk's
        /// hidden-state capture entirely.
        /// </summary>
        bool NeedsHiddenState { get; }

        /// <summary>
        /// True when the TARGET keeps this speculator in sync during its own
        /// forward, so prompt prefill needs neither per-row hidden states nor a
        /// <see cref="Commit"/> call (DeepSeek V4's DSpark). Prefill then runs
        /// as one call and keeps the trunk's micro-batch pipelining.
        /// </summary>
        bool HandlesOwnPrefill { get; }

        /// <summary>
        /// Trailing committed positions this speculator's state can still read at
        /// the end of a prompt (a block drafter's KV ring). Prefill rows older than
        /// that are overwritten before any proposal, so they need no hidden-state
        /// capture or <see cref="Commit"/>. <see cref="int.MaxValue"/>: every row.
        /// </summary>
        int PrefillContextRows => int.MaxValue;

        /// <summary>
        /// True when proposals condition on the PROMPT's hidden rows - a block drafter's KV ring is filled from the
        /// prompt's feature rows - so a prompt prefilled on the plain path leaves the drafter without its context for
        /// the whole first answer. The scheduler then prefills through the speculative context even though the
        /// speculator could be seeded from the trunk's tokens afterwards; only the last
        /// <see cref="PrefillContextRows"/> rows capture hidden states. Default false.
        /// </summary>
        bool ConditionsOnPrompt => false;

        /// <summary>
        /// True when this algorithm can arm on a sequence that ADOPTED a KV prefix it
        /// never processed itself - the cache came from another request (prefix cache)
        /// or an earlier turn, so trunk positions exist that this speculator did not
        /// see.
        ///
        /// The executor refuses to arm on such a sequence by default, because a learned
        /// per-position draft head (NextN / MTP) chains its state token by token and a
        /// gap makes every subsequent proposal garbage. That default was costing every
        /// other algorithm the whole feature in normal chat use: from the SECOND turn
        /// on, a Web UI conversation always adopts a prefix, so speculation silently
        /// never armed and the operator saw a drafter that helped on turn one and did
        /// nothing afterwards.
        ///
        /// An algorithm may return true when a gap costs it ACCEPTANCE but not
        /// correctness and it recovers on its own:
        ///   * n-gram mines the emitted token history, which is complete regardless of
        ///     what the KV cache did;
        ///   * a BLOCK drafter (DFlash / DFlash2) reads its own sliding KV ring, which
        ///     refills from the freshly forwarded suffix and from every committed
        ///     token, so at worst its first few blocks draft from a short context.
        /// Every draft is verified by the trunk either way, so a stale speculator can
        /// only cost throughput - never a wrong token.
        /// </summary>
        bool CanArmAfterPrefixReuse => false;

        /// <summary>
        /// Propose up to <see cref="DraftContext.MaxTokens"/> continuation
        /// tokens, appended in order to <paramref name="draftOut"/> (which the
        /// caller has already cleared). Returning 0 is always legal and simply
        /// degrades the step to a plain single-token decode.
        /// </summary>
        int Propose(in DraftContext ctx, List<int> draftOut);

        /// <summary>
        /// Tokens <paramref name="tokens"/> have been COMMITTED to the trunk at
        /// <paramref name="startPos"/>; bring the speculator's own state up to
        /// date. Row k of <paramref name="hRows"/> is the trunk hidden state of
        /// the token PRECEDING <c>tokens[k]</c>, and is null when
        /// <see cref="NeedsHiddenState"/> is false.
        /// </summary>
        void Commit(int[] tokens, float[]? hRows, int startPos);

        /// <summary>Drop all per-sequence state (a new request, a KV reset).</summary>
        void Reset();
    }
}
