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
    ///   * <see cref="SpecForward"/> runs K+1 tokens as ONE batch and returns
    ///     per-row logits (so a whole draft window is checked for roughly the
    ///     cost of one decode step) plus, optionally, the per-row hidden state
    ///     a learned drafter chains from;
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

        /// <summary>
        /// Non-null when this trunk must not speculate at all, with the reason in
        /// operator words. Unlike <see cref="SpeculationProfitable"/> (a speed
        /// judgment) this is a CORRECTNESS refusal: the trunk's multi-token verify
        /// cannot reproduce the logits its own single-token decode produces, so a
        /// speculative stream would diverge from plain greedy decoding. The engine
        /// then serves every request with plain decoding and says why, once.
        /// Default null (speculation allowed).
        /// </summary>
        string SpeculationRefusal => null;

        /// <summary>Trunk tokens currently committed to the model's live KV cache.</summary>
        int CacheSeqLen { get; }

        /// <summary>Maximum trunk context length.</summary>
        int MaxContextLength { get; }

        /// <summary>
        /// Width of one row of the hidden state the trunk taps for a learned
        /// drafter (<see cref="SpecForward"/>'s hAllOut). Defaults to the
        /// model's hidden size; block drafters that consume several
        /// concatenated layers (DeepSeek V4's DSpark, Muse-Glimmer's DFlash)
        /// report a wider row.
        /// </summary>
        int SpecFeatureSize => Config.HiddenSize;

        /// <summary>Prompt-prefill chunk the model prefers for speculative
        /// prefill (0 = let the caller choose). Models whose trunk re-reads
        /// per-expert weights once per micro-batch want whole micro-batches.</summary>
        int SpecPrefillChunkSize => 0;

        /// <summary>
        /// Draft window this trunk would rather have by DEFAULT, or 0 for "no
        /// preference". It narrows the default only; an operator who passed
        /// <c>--spec-draft</c> gets exactly what they asked for.
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
        /// Optional default specifically for weight-free n-gram drafting. A positive
        /// value may exceed the shared default when a wider verification batch uses
        /// a faster backend kernel. Explicit --spec-draft still wins; zero keeps
        /// the general trunk preference. Learned drafters keep their own policy.
        /// </summary>
        int SpecPreferredNGramDraftWindow => 0;

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
        /// Trunk forward identical to <see cref="IModelArchitecture.Forward"/>
        /// but additionally captures the post-final-norm hidden state of every
        /// row into <paramref name="hAllOut"/> (n*<see cref="SpecFeatureSize"/>
        /// floats; may be null when the drafter needs no hidden state) and, when
        /// <paramref name="allLogitsRows"/> is set, LM-head logits for every
        /// row into <paramref name="logitsOut"/> (n*vocab floats) instead of
        /// only the last row. Advances the KV caches like Forward().
        /// </summary>
        void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows);

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

    /// <summary>
    /// Layer 1, batched variant: serving the speculative TRUNK passes through
    /// the batched paged path (paged KV via slot mapping, per-slot recurrent
    /// state) instead of the single live linear cache. This is the
    /// high-throughput option: the trunk runs on the same kernels as the
    /// non-speculative batched baseline, composes with prefix caching, and
    /// transitions gracefully to/from concurrent batches (the sequence's K/V
    /// always lives in paged storage). A learned draft head itself still runs
    /// on the linear cache at the draft layer - it is one decoder block whose
    /// state is private to the speculative context.
    /// </summary>
    public interface IBatchedSpeculativeTarget : ISpeculativeTarget
    {
        /// <summary>True when the loaded model/backend can serve speculative
        /// trunk passes through its batched paged path.</summary>
        bool SupportsBatchedSpecTrunk { get; }

        /// <summary>
        /// Trunk forward of <paramref name="tokens"/> for ONE sequence through
        /// the batched paged path. <paramref name="startPos"/> must equal
        /// <c>seq.NumComputedTokens</c> (the caller advances the sequence only
        /// after the step completes); the sequence's block table must already
        /// cover <c>startPos + tokens.Length</c> positions. Captures per-row
        /// post-final-norm hidden states into <paramref name="hAllOut"/>
        /// (n*featureSize floats; may be null) and logits into
        /// <paramref name="logitsOut"/> (n*vocab floats when
        /// <paramref name="allLogitsRows"/>, else vocab floats for the last row).
        /// </summary>
        void SpecForwardBatched(SequenceState seq, int[] tokens, int startPos,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows);

        /// <summary>Snapshot the per-slot recurrent (GDN/SSM) state of
        /// <paramref name="seq"/> before a verify batch.</summary>
        void SpecSnapshotRecurrentStateSlots(SequenceState seq);

        /// <summary>Restore the per-slot recurrent state captured by
        /// <see cref="SpecSnapshotRecurrentStateSlots"/>. Paged attention KV
        /// needs no rewind: reads are bounded by the per-pass sequence length
        /// and rejected slots are overwritten by the kept-prefix re-forward
        /// and subsequent steps.</summary>
        void SpecRestoreRecurrentStateSlots(SequenceState seq);
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
        /// <summary>
        /// True when this head keeps NO per-position state of its own - it drafts from
        /// the trunk's KV cache and the trunk hidden state it is handed - so it can
        /// start drafting at any trunk position, including right after a request
        /// adopted a KV prefix it never saw (the previous turn of a chat, a shared
        /// system-prompt checkpoint, a prefix-cache hit). A NextN/MTP block with its
        /// own KV cache must leave this false: a gap in what it replayed makes every
        /// later proposal garbage (harmless, but a wasted verify per step). A head that
        /// says true must accept a <see cref="DraftCatchUp"/> whose hidden rows are
        /// null: that is how a seeded start hands it the tokens already committed.
        /// </summary>
        bool DraftHeadResumesAfterGap => false;

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
        int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut) => 0;

        /// <summary>Replay verified trunk tokens through the draft head so its
        /// KV cache tracks exact trunk hidden states (llama.cpp's draft-mtp
        /// process()). Row k of <paramref name="hRows"/> is the hidden state of
        /// the token PRECEDING tokens[k].</summary>
        void DraftCatchUp(int[] tokens, float[] hRows, int startPos);

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
    }

    /// <summary>Convenience alias for the common case: a model that is its own
    /// speculative target AND ships its own draft head.</summary>
    public interface ISpeculativeModel : ISpeculativeTarget, IDraftHead { }

    /// <summary>Convenience alias for a model whose speculative trunk can also
    /// run through the batched paged path.</summary>
    public interface IBatchedSpeculativeModel : IBatchedSpeculativeTarget, ISpeculativeModel { }
}
