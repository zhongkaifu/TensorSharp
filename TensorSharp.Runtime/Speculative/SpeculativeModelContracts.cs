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
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Runtime.Speculative
{
    // ========================================================================
    // Speculative decoding is split into THREE independent layers. Keeping them
    // apart is what lets a new model reuse every algorithm, and a new algorithm
    // reuse every model:
    //
    //     Model architecture   !=   Speculation algorithm   !=   Speculator weights
    //     (ISpeculativeTarget)      (ISpeculator)                (IDraftHead impl)
    //
    //   1. MODEL ARCHITECTURE - what the trunk can do: forward several tokens
    //      at once with per-row logits, tap the hidden state of every row, and
    //      undo a rejected window. That is ISpeculativeTarget, and it is the
    //      only thing a model MUST implement to speculate at all (an n-gram
    //      drafter needs nothing else).
    //
    //   2. SPECULATION ALGORITHM - how the next few tokens are guessed:
    //      ISpeculator (see ISpeculator.cs). Model-agnostic by construction;
    //      the shared draft/verify/rollback loop in SpeculativeExecution never
    //      names an algorithm.
    //
    //   3. SPECULATOR WEIGHTS - the learned drafter a checkpoint happens to
    //      carry (a NextN/MTP block, a DSpark or DFlash block drafter, an
    //      EAGLE head). That is IDraftHead: a thin, model-specific adapter
    //      over weights that cannot transfer between target models.
    //
    // A model that ships its own drafter implements both role interfaces on the
    // same class (ISpeculativeModel is the convenience alias); a model with no
    // drafter implements only ISpeculativeTarget and still gets every
    // weight-free algorithm.
    // ========================================================================

    /// <summary>
    /// Layer 1 - the TARGET MODEL adapter. Everything the shared speculative
    /// loop needs from the trunk, and nothing about how tokens are drafted.
    ///
    /// Two capabilities together make verification possible:
    ///   * <see cref="SpecForward"/> verifies K+1 tokens and returns per-row
    ///     logits plus, optionally, the per-row hidden state a learned drafter
    ///     chains from. The target may divide the window into smaller batches
    ///     to preserve its decoding arithmetic; speed depends on that executor;
    ///   * the rollback trio (<see cref="SpecSnapshotRecurrentState"/> /
    ///     <see cref="SpecRestoreRecurrentState"/> / <see cref="SpecRewindCache"/>)
    ///     undoes the rejected tail of that batch.
    /// </summary>
    public interface ISpeculativeTarget : IModelArchitecture
    {
        /// <summary>
        /// True when speculation is expected to be PROFITABLE on the current
        /// backend - i.e. the model can drive its accelerated multi-token
        /// verify/draft kernels. When false the verify (seqLen=K+1) and the
        /// draft fall to the per-op fallback, which does not amortize the trunk
        /// over the speculative window and makes speculation net-negative; the
        /// executor then serves the sequence with the normal (fast) decode path
        /// instead of arming speculation. Default true (backends/models that
        /// always have their accelerated path, or where the per-op path still
        /// wins).
        /// </summary>
        bool SpeculationProfitable => true;

        /// <summary>Model/backend capability for the measured native fixed K7
        /// profile. False by default; the factory also requires the target's own
        /// block head, K7/pMin0 and a caller-qualified greedy request.</summary>
        bool SpecPrefersFixedCostGovernor => false;

        /// <summary>
        /// Non-null when this trunk must not speculate at all, with the reason in
        /// operator words. Unlike <see cref="SpeculationProfitable"/> (a speed
        /// judgment) this is a CORRECTNESS refusal: the trunk's multi-token verify
        /// cannot reproduce the logits its own single-token decode produces, so a
        /// speculative stream would diverge from plain greedy decoding. The engine
        /// then serves every request with plain decoding and says why, once.
        /// Default null (speculation allowed).
        /// </summary>
        string? SpeculationRefusal => null;

        /// <summary>Trunk tokens currently committed to the model's live KV cache.</summary>
        int CacheSeqLen { get; }

        /// <summary>Maximum trunk context length.</summary>
        new int MaxContextLength { get; }

        /// <summary>
        /// Width of one row of the hidden state the trunk taps for a learned
        /// drafter (<see cref="SpecForward"/>'s hAllOut). Defaults to the
        /// model's hidden size; block drafters that consume several
        /// concatenated layers (DeepSeek V4's DSpark, Muse-Glimmer's DFlash)
        /// report a wider row.
        /// </summary>
        int SpecFeatureSize => Config.HiddenSize;

        /// <summary>
        /// True when the drafter reads the taps of every committed row from the
        /// model's own state (prefill, plain steps and commits all write them), so
        /// the carried feature row is a placeholder: a plain prefill, a prefix
        /// reuse or a gap leaves no carry to recapture with a plain step first.
        /// </summary>
        bool SpecCarryOnDevice => false;

        /// <summary>Prompt-prefill chunk the model prefers for speculative
        /// prefill (0 = let the caller choose). Models whose trunk re-reads
        /// per-expert weights once per micro-batch want whole micro-batches.</summary>
        int SpecPrefillChunkSize => 0;

        /// <summary>
        /// Hard limit on drafted tokens in one verification window. Verification
        /// also forwards its pending anchor, so its width is this limit plus one.
        /// Unlike the preferred windows below, this cache/backend correctness
        /// bound also applies to explicit --spec-draft values and custom algorithms.
        /// Zero permits only plain steps; the default imposes no additional limit.
        /// Prompt prefill is governed separately by <see cref="SpecPrefillChunkSize"/>.
        /// </summary>
        int SpecMaxDraftTokens => int.MaxValue;

        /// <summary>
        /// Draft window this trunk would rather have by DEFAULT, or 0 for "no
        /// preference". It narrows the default only; an explicit
        /// <c>--spec-draft</c> overrides it within <see cref="SpecMaxDraftTokens"/>.
        ///
        /// This exists because the cost of a wide window is a property of the
        /// TRUNK, not of the drafter. On a model with recurrent state a verify
        /// over N rows runs the chunked recurrent scan instead of the single-token
        /// update, AND a partial rejection has to restore that state and re-advance
        /// over the accepted prefix - so the marginal token of window costs far
        /// more than it does on a dense-attention trunk, where the verify's own KV
        /// writes are reusable and the rollback is a position rewind. Measured on
        /// Qwen3.8-27B: the default window 8 gave 9.4 tok/s against 15.5 at 3,
        /// with the same drafter and the same acceptance per position.
        /// </summary>
        int SpecPreferredDraftWindow => 0;

        /// <summary>
        /// Cap on drafts for the SINGLE-sequence verify only, below the draft budget a
        /// batched verify (<see cref="IBatchedSpeculativeTarget"/>) may spend. On a
        /// recurrent trunk the single-sequence verify keeps one state snapshot per
        /// row, so its cost grows with the window, while the batched verify replays
        /// only the accepted prefix and costs about the same at 8 rows as at 4.
        /// Default: no extra cap.
        /// </summary>
        int SpecSoloMaxDraftTokens => int.MaxValue;

        /// <summary>
        /// Why <see cref="SpecSoloMaxDraftTokens"/> is below the requested draft count, and how to
        /// lift it, for the one warning the executor logs when it lowers a request (no silent caps).
        /// </summary>
        string? SpecSoloDraftCapReason => null;

        /// <summary>
        /// Optional default specifically for weight-free n-gram drafting. A positive
        /// value may exceed the shared default when a wider verification batch uses
        /// a faster backend kernel. Explicit --spec-draft still wins within
        /// <see cref="SpecMaxDraftTokens"/>; zero keeps
        /// the general trunk preference. Learned drafters keep their own policy.
        /// </summary>
        int SpecPreferredNGramDraftWindow => 0;

        /// <summary>
        /// Default cumulative confidence gate this trunk wants for a BLOCK drafter
        /// (<see cref="BlockDraftSpeculator.MinDraftProb"/>), or NaN for the
        /// algorithm's own default. Like the window preference it only replaces a
        /// default: an explicit <c>--spec-pmin</c> always wins. The block default is
        /// tuned where an extra verify row is expensive (a sparse-MoE trunk); where a
        /// verify row is nearly free, gating drafts away costs more than it saves.
        /// </summary>
        float SpecPreferredBlockGate => float.NaN;

        /// <summary>
        /// Default per-token confidence gate this trunk wants for a learned PER-TOKEN head
        /// (<see cref="DraftHeadSpeculator.MinDraftProb"/>), or NaN for the algorithm's own default
        /// (<see cref="DraftHeadSpeculator.DefaultGate"/>). Like the window preference it only replaces a
        /// default: an explicit <c>--spec-pmin</c> always wins. Where each verify row costs a sizeable share of a
        /// decode step (a sparse-MoE trunk whose rows each route their own experts), a draft the head is unsure of
        /// costs more than it saves.
        /// </summary>
        float SpecPreferredDraftGate => float.NaN;

        /// <summary>
        /// True when a verify batch has already written reusable attention KV for
        /// EVERY token it processed (so the accepted prefix's KV is correct in the
        /// live cache), and the model has no recurrent state that a re-forward would
        /// need to advance. When true the executor skips the redundant kept-prefix
        /// re-forward on partial acceptance and simply rewinds the cache position to
        /// keep the verify's writes - the dominant rollback cost on long contexts.
        /// Default false (the re-forward path) so recurrent models (e.g. Qwen 3.5
        /// GatedDeltaNet) are unaffected.
        /// </summary>
        bool SpecVerifyPersistsAcceptedKv => false;
        /// <summary>
        /// True when a plain single-token step inside a speculative session may run
        /// through the model's ordinary <see cref="IModelArchitecture.Forward"/>
        /// instead of a one-row <see cref="SpecForward"/>. Only a trunk whose two
        /// paths leave IDENTICAL state behind can say so - Gemma 4, where both run
        /// the same fused decode kernel over the same cache and differ only in
        /// whether the LM head is folded into the graph. A trunk with recurrent
        /// state that its verify graph and its decode graph keep in different
        /// device buffers (Qwen 3.5's GatedDeltaNet) must leave this false.
        /// </summary>
        bool SpecPlainStepUsesForward => false;

        /// <summary>
        /// The linear <see cref="SpecForward"/> path consumes queued image/audio
        /// embeddings and their position metadata exactly as ordinary Forward does.
        /// The scheduler may then capture the entire multimodal prefill for a learned
        /// head that cannot resume after a gap. Batched speculative trunks retain
        /// their separate multimodal restriction. Opt in only when both embedding
        /// injection and the draft head's positions are supported.
        /// </summary>
        bool SpecSupportsMultimodalPrefill => false;

        /// <summary>
        /// True when the model's own decode (<see cref="SpecPlainStepUsesForward"/>) and
        /// its speculative forward keep the recurrent state in different device
        /// families, so that alternating between them costs a state drain and re-seed
        /// each way (Qwen 3.5: the fused decode graph's resident slot against the
        /// verify graph's slices, ~50 MB per direction on the 9B, ~150 MB on the 27B). The trunk then takes
        /// the model's decode only for PARKED plain steps - a long run of them - and
        /// keeps an ordinary no-proposal plain step inside the speculative family.
        /// False (the default, Gemma 4) means both steps share one cache and the
        /// cheaper decode serves every plain step.
        /// </summary>
        bool SpecPlainStepCostsFamilySwitch => false;
        /// <summary>
        /// True when <see cref="SpecForward"/> operates on whatever cache the model has
        /// bound at the moment - a per-request fused holder included - rather than on
        /// the model's primary linear cache alone. Only such a trunk may speculate for
        /// a request served from a holder (a checkpoint clone, a retained
        /// conversation): Gemma 4's forward reads the active arrays, so it qualifies;
        /// Qwen 3.5's speculative session re-seats its recurrent state against the
        /// primary cache, and running it on a bound holder left the holder unbound and
        /// the request lost at position 0.
        /// </summary>
        bool SpecTrunkFollowsBoundCache => false;

        /// <summary>
        /// Concurrent decoders should take turns at solo speculative steps (one request keeps the turn for a stretch
        /// of steps) rather than decode one token each in the token-batched graph. True where a speculative step
        /// serves more tokens per second than a batched step for several sequences: Flash Next streaming its expert
        /// misses on a 48 GB M5 Pro decoded four requests at 30-35 tokens/s in all batched against 40-44 for one
        /// request speculating alone, the way Strata serves them (one at a time, 52-55). Needs per-request drafter
        /// state (<see cref="IPerRequestDraftState"/>). TS_SPEC_TURNS=0/1 overrides. False by default: on a discrete
        /// GPU a batched step costs little more than a solo one.
        /// </summary>
        bool SpecTurnsPreferred => false;

        /// <summary>
        /// Trunk forward identical to <see cref="IModelArchitecture.Forward"/>
        /// but additionally captures the post-final-norm hidden state of every
        /// row into <paramref name="hAllOut"/> (n*<see cref="SpecFeatureSize"/>
        /// floats; may be null when the drafter needs no hidden state) and, when
        /// <paramref name="allLogitsRows"/> is set, LM-head logits for every
        /// row into <paramref name="logitsOut"/> (n*vocab floats) instead of
        /// only the last row. Advances the KV caches like Forward().
        /// </summary>
        void SpecForward(int[] tokens, float[]? hAllOut, float[] logitsOut, bool allLogitsRows);

        /// <summary>Optional verify with the same state/hidden-row semantics as
        /// <see cref="SpecForward"/>, returning each raw row's first-max token instead
        /// of host logits. Only a caller whose sampler is plain greedy may use it;
        /// forced thinking tokens and stop checks remain the caller's responsibility.
        /// False MUST leave all model state and output buffers untouched. A failed
        /// forward throws and must never be retried as a full-logit forward.</summary>
        bool TrySpecForwardGreedyVerify(int[] tokens, float[]? hAllOut, int[] drawnOut) => false;

        /// <summary>Pre-grow the KV caches to cover a full speculative window
        /// (growing mid-draft would drop drafter rows written past the trunk
        /// position).</summary>
        void SpecEnsureCapacity(int requiredSeqLen);

        /// <summary>Snapshot the recurrent (GDN/SSM) state before a verify batch.</summary>
        void SpecSnapshotRecurrentState();

        /// <summary>
        /// How much of the verify that just ran was accepted, delivered on every
        /// speculative step (full acceptance included) before any rollback happens.
        /// A trunk that left post-verify state on the device until the accept count
        /// was known settles it here. Default: nothing to do.
        /// </summary>
        void SpecOnVerifyAccepted(int acceptedRows, int verifyRows) { }

        /// <summary>Restore the recurrent state captured by <see cref="SpecSnapshotRecurrentState"/>.</summary>
        void SpecRestoreRecurrentState();

        /// <summary>Rewind the attention KV position counter after rejected
        /// speculative tokens (no data movement needed).</summary>
        void SpecRewindCache(int length);
    }

    /// <summary>How a learned draft head proposes tokens - the one fact the
    /// factory needs to pick the matching <see cref="ISpeculator"/>.</summary>
    public enum DraftHeadKind
    {
        /// <summary>No usable drafter in the loaded weights. Weight-free
        /// algorithms (n-gram) still work.</summary>
        None,

        /// <summary>One token per pass, chaining its own hidden output into the
        /// next pass: a NextN/MTP block (Qwen 3.6, GLM-5.2, Gemma 4's separate
        /// assistant GGUF), or an EAGLE-style head.</summary>
        PerToken,

        /// <summary>A whole block per pass, semi-autoregressively, with a
        /// confidence head scoring each position: DeepSeek V4's DSpark,
        /// Muse-Glimmer's DFlash.</summary>
        Block,
    }

    /// <summary>
    /// Layer 3 - the SPECULATOR WEIGHTS a checkpoint carries, behind a thin
    /// model-specific adapter. These weights are trained against one target
    /// model and do not transfer (hidden-state distribution, tokenizer,
    /// embedding and LM head all differ), which is exactly why they sit behind
    /// their own interface instead of inside the algorithm.
    ///
    /// A model with no drafter simply reports <see cref="DraftHeadKind.None"/>;
    /// it can still be sped up by a weight-free speculator.
    /// </summary>
    public interface IDraftHead
    {
        /// <summary>How this head drafts, or <see cref="DraftHeadKind.None"/>
        /// when the loaded weights carry no usable drafter.</summary>
        DraftHeadKind DraftHeadKind { get; }

        /// <summary>True when the loaded weights contain a usable drafter.</summary>
        bool HasDraftHead => DraftHeadKind != DraftHeadKind.None;

        /// <summary>Tokens a <see cref="DraftHeadKind.Block"/> head emits per
        /// pass (its trained block size); 0 for a per-token head.</summary>
        int DraftBlockSize => 0;

        /// <summary>
        /// True when a <see cref="DraftHeadKind.Block"/> head's attention is causal inside
        /// the block (DSpine), so the first w rows of its block ARE a w-row block: the
        /// speculator then drafts only the rows a step can verify instead of the trained
        /// block. False for DFlash / DFlash2, whose block attention is non-causal - a
        /// narrower block drafts different tokens there, so they always draft the whole
        /// block and the speculator takes its prefix.
        /// </summary>
        bool DraftBlockIsCausal => false;

        /// <summary>
        /// True when <see cref="ISpeculativeTarget.SpecForward"/> replays this
        /// head's own state for every row it processes, so prompt prefill needs
        /// neither per-row hidden states nor a <see cref="DraftCatchUp"/> call.
        /// Prefill then runs as ONE call and keeps the trunk's micro-batch
        /// pipelining instead of stalling on a readback per chunk.
        ///
        /// Implementations must honour the buffer-size contract this implies:
        /// when <c>hAllOut</c> is too small to hold one row per token,
        /// <see cref="ISpeculativeTarget.SpecForward"/> fills only its first
        /// row, with the hidden state of the LAST token it processed.
        /// </summary>
        bool DraftSelfCatchUp => false;

        /// <summary>Trailing committed positions the head's private state keeps (a
        /// sliding KV ring); <see cref="int.MaxValue"/> when it keeps all of them.</summary>
        int DraftContextRows => int.MaxValue;
        /// <summary>
        /// True when this head can safely restart at any trunk position, including
        /// right after a request
        /// adopted a KV prefix it never saw (the previous turn of a chat, a shared
        /// system-prompt checkpoint, a prefix-cache hit). This includes stateless
        /// heads and heads that explicitly reset their private cache to a valid
        /// suffix. A stateful head must never read unwritten or stale prefix rows.
        /// A head that says true must accept a <see cref="DraftCatchUp"/> whose hidden
        /// rows are null: that reports a gap ending at startPos + tokens.Length.
        /// The executor captures a fresh trunk hidden row before proposing again.
        /// </summary>
        bool DraftHeadResumesAfterGap => false;

        /// <summary>
        /// True when the head's drafts condition on the prompt's rows, so a request whose
        /// prompt prefills from position 0 reads it through its speculative context (the
        /// head replays every prompt row) even when the head could restart after a gap
        /// (<see cref="DraftHeadResumesAfterGap"/>) at the first decode step instead.
        /// </summary>
        bool DraftHeadConditionsOnPrompt => false;

        /// <summary>One per-token draft step at <paramref name="pos"/>: consume
        /// (token, previous hidden), fill next-token logits and the chained
        /// hidden state. Only called for <see cref="DraftHeadKind.PerToken"/>.</summary>
        void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
            => throw new NotSupportedException("This draft head does not draft one token at a time.");

        /// <summary>
        /// Drafts one block at <paramref name="position"/> (the position of
        /// <paramref name="lastToken"/>, which is not yet in the trunk cache),
        /// given the trunk hidden state of the position before it. Writes the
        /// drafted tokens to <paramref name="draftOut"/> and their predicted
        /// acceptance probabilities to <paramref name="confOut"/>, and returns
        /// how many it produced. Only called for <see cref="DraftHeadKind.Block"/>.
        /// </summary>
        int DraftBlock(int lastToken, float[]? hPrev, int position, int[] draftOut, float[] confOut) => 0;

        /// <summary>Replay verified trunk tokens through the draft head so its
        /// KV cache tracks exact trunk hidden states (llama.cpp's draft-mtp
        /// process()). Row k of <paramref name="hRows"/> is the hidden state of
        /// the token PRECEDING tokens[k].</summary>
        void DraftCatchUp(int[] tokens, float[]? hRows, int startPos);

        /// <summary>
        /// True when this head can replay the verified tokens AND take the first
        /// draft step in ONE pass, via <see cref="DraftCatchUpAndStep"/>.
        ///
        /// This is what llama.cpp's draft-mtp does: it runs its block over
        /// <c>n_accepted + 1</c> rows rather than a catch-up pass followed by a
        /// separate first draft. On a head whose per-call cost is mostly fixed
        /// (a whole extra graph, its own launch and readback), that is one call
        /// saved per speculative step - measured at 6.4 ms of a 100 ms step on
        /// Qwen 3.8, i.e. the entire remaining gap to llama.cpp.
        /// </summary>
        bool SupportsFusedCatchUpStep => false;

        /// <summary>
        /// Replay verified trunk tokens and take the first draft step in one
        /// pass. <paramref name="tokens"/> is the verified run followed by the
        /// token the next draft starts from, so the last entry is NOT a replay;
        /// row k of <paramref name="hRows"/> is the hidden state of the token
        /// preceding tokens[k], as in <see cref="DraftCatchUp"/>. Fills
        /// <paramref name="logitsOut"/> and <paramref name="hOut"/> from the LAST
        /// row - byte-identical to what
        /// <c>DraftCatchUp(tokens[..^1], ...)</c> followed by
        /// <c>DraftStep(tokens[^1], ...)</c> would produce, because the block is
        /// causal over its own KV and the last row therefore sees exactly the
        /// replayed rows either way.
        /// Only called when <see cref="SupportsFusedCatchUpStep"/> is true.
        /// </summary>
        void DraftCatchUpAndStep(int[] tokens, float[] hRows, int startPos,
            float[] logitsOut, float[] hOut)
            => throw new NotSupportedException(
                "This draft head cannot fuse its catch-up with the first draft step.");

        /// <summary>
        /// True when this head can run a whole per-token draft chain on its device, via <see cref="DraftChain"/>: the
        /// speculator then makes one call per window instead of one per draft, and never reads draft logits. Used only
        /// when the draft distribution is not adjusted (no penalties), since the chain picks each step's first maximum.
        /// </summary>
        bool SupportsDraftChain => false;

        /// <summary>
        /// <see cref="DraftCatchUpAndStep"/> (or, for a single token, <see cref="DraftStep"/>) followed by up to
        /// <paramref name="maxTokens"/> - 1 further steps, each fed the previous step's first-max token and the head's own
        /// hidden output on the device. Writes each step's token to <paramref name="draftOut"/> and its confidence (the
        /// top-1 probability over the step's 10 largest logits, the per-token gate's measure) to
        /// <paramref name="probOut"/>, and returns the steps run (at most <paramref name="maxTokens"/>). The head's own
        /// state then covers every step's row; later calls start at or before positions it holds.
        /// </summary>
        int DraftChain(int[] tokens, float[] hRows, int startPos, int maxTokens, int[] draftOut, float[] probOut) => 0;
    }

    /// <summary>Convenience alias for the common case: a model that is its own
    /// speculative target AND ships its own draft head.</summary>
    public interface ISpeculativeModel : ISpeculativeTarget, IDraftHead { }

    /// <summary>
    /// A drafter that keeps one state per request (a DFlash ring is the context of ONE sequence). Concurrent
    /// requests that speculate - in a batched verify, or taking turns at solo speculative steps - each bind their own
    /// before drafting, so one request's drafts never read another's context.
    /// </summary>
    public interface IPerRequestDraftState
    {
        /// <summary>Point the drafter's per-sequence state at
        /// <paramref name="requestId"/> (created on first use); null selects the shared state.</summary>
        void BindDraftState(string? requestId);

        /// <summary>Return <paramref name="requestId"/>'s drafter state.</summary>
        void ReleaseDraftState(string requestId);
    }

    /// <summary>
    /// A trunk that verifies the draft windows of SEVERAL concurrent sequences, each
    /// in its own per-request cache, in one forward. Without it concurrent requests
    /// decode one token per step in a token-batched graph while a lone request
    /// verifies a whole window per step, so the second request in flight cut every
    /// request's rate to that of plain decoding.
    ///
    /// Greedy only: each verify row comes back as its argmax. The drafter keeps one
    /// state per request (<see cref="BindDraftState"/>), because several sequences
    /// now draft between the same pair of verifies.
    /// </summary>
    public interface IBatchedSpeculativeTarget : IPerRequestDraftState
    {
        /// <summary>Tokens per sequence (anchor plus drafts) one batched verify of
        /// <paramref name="sequences"/> sequences forwards, or 0 when the trunk cannot
        /// serve that many now.</summary>
        int BatchedSpecWindow(int sequences);

        /// <summary>Tokens committed to <paramref name="requestId"/>'s per-request
        /// cache, or -1 when it has none.</summary>
        int FusedCacheSeqLen(string requestId);

        /// <summary>Rows <paramref name="requestId"/>'s per-request cache can take past its
        /// committed tokens without growing, or 0 when it has none. A batched verify
        /// writes its whole window for every sequence, padding included, so the window
        /// must fit every sequence's free rows.</summary>
        int FusedCacheFreeRows(string requestId) => 0;

        /// <summary>Scope of one batched step: drafting inside it must not end or
        /// re-enter the trunk's single-sequence speculative session.</summary>
        IDisposable BeginBatchedSpeculation();

        /// <summary>
        /// Draft one block for each of several sequences in one pass, each over its
        /// own drafter state: <paramref name="anchors"/>[i] is sequence i's next token
        /// at <paramref name="positions"/>[i], <paramref name="carries"/>[i] its carried
        /// feature row. Fills each <paramref name="tokensOut"/>[i] / <paramref name="confOut"/>[i]
        /// with the drafter's full proposal block, before any gate. False when the
        /// drafter cannot batch; the caller then drafts per sequence.
        /// </summary>
        bool TryDraftBatch(IReadOnlyList<string> requestIds, int[] anchors, float[][] carries, int[] positions,
            int[][] tokensOut, float[][] confOut) => false;

        /// <summary>Width of the proposal block <see cref="TryDraftBatch"/> fills per sequence.</summary>
        int BatchDraftBlockWidth => 0;

        /// <summary>
        /// Forward <paramref name="tokens"/> ([sequences * window], sequence-major:
        /// each sequence's anchor then its drafts, padded) at
        /// <paramref name="positions"/> (each sequence's cache length). Fills
        /// <paramref name="drawn"/> with each row's argmax and
        /// <paramref name="features"/> ([sequences * window * SpecFeatureSize]) with
        /// each row's drafter taps. Returns false, with every cache untouched, when
        /// the trunk declines; throws when it failed after writing.
        /// </summary>
        bool TryBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int window,
            int[] tokens, int[] drawn, float[] features);

        /// <summary>Advance sequence i of the last verify by <c>kept[i]</c> tokens
        /// (1 .. window). Throws when the caches may be partially advanced.</summary>
        void CommitBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int[] kept);

        /// <summary>Whether <see cref="TryBatchedSpecVerifyLogits"/> serves the batched verify of
        /// sequences that sample: each row's logits come back, and each sequence draws its rows
        /// with its own sampler exactly as its solo step does.</summary>
        bool SupportsBatchedSpecLogits => false;

        /// <summary>
        /// <see cref="TryBatchedSpecVerify"/> that fills <paramref name="logits"/>
        /// ([sequences * window * vocab]) with every row's logits instead of their argmax.
        /// </summary>
        bool TryBatchedSpecVerifyLogits(IReadOnlyList<string> requestIds, int[] positions, int window,
            int[] tokens, float[] logits, float[] features) => false;
    }

    /// <summary>
    /// A batched trunk that can run one sequence's prompt chunk inside its next
    /// <see cref="IBatchedSpeculativeTarget.TryBatchedSpecVerify"/>: one forward over the
    /// chunk's rows and the verify windows, so the verify reads no weights of its own.
    /// </summary>
    public interface IMixedStepPrefillTarget
    {
        /// <summary>
        /// Stage <paramref name="tokens"/> of <paramref name="requestId"/> at
        /// <paramref name="position"/> (its cache length) for the next batched verify. The
        /// chunk takes the media rows queued since the last forward (queue them for its slice
        /// first). <paramref name="whole"/>: the chunk must run whole (it ends the prompt or a
        /// checkpoint); otherwise the trunk may run only a prefix of it. False when the chunk
        /// cannot ride a verify; its media is then dropped, as it is when no verify runs it.
        /// </summary>
        bool StageMixedPrefill(string requestId, int[] tokens, int position, bool whole);

        /// <summary>
        /// End the staged chunk: true with the tokens a verify ran (a prefix of the chunk) and
        /// its last row's logits; false when no verify ran it (nothing advanced). Throws when the
        /// forward that carried it failed after writing its cache.
        /// </summary>
        bool TakeMixedPrefill(string requestId, out int tokensRun, out float[]? logits);

        /// <summary>The prompt tokens a step that also decodes should carry
        /// (<see cref="Scheduling.SchedulerConfig.MixedStepPrefillBudget"/>) when the configuration
        /// does not set it; 0 leaves it unbounded.</summary>
        int PreferredMixedStepPrefillTokens => 0;
    }
}
