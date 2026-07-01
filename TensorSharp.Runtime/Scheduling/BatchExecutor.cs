// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Runs the work decided by <see cref="ContinuousBatchScheduler"/> against
    /// a single <see cref="IModelArchitecture"/>. Owns the KV-state ownership
    /// invariant: at any moment the model's KV tensors hold exactly one
    /// sequence's state; switching sequences extracts the outgoing state into
    /// paged blocks and injects the incoming state from paged blocks.
    ///
    /// In the future a batched-paged-attention path will allow N sequences to
    /// share the model's KV tensors via a slot mapping and block table tensor;
    /// see <see cref="IBatchedPagedModel"/> for the opt-in interface.
    /// </summary>
    public sealed class BatchExecutor
    {
        private readonly IModelArchitecture _model;
        private readonly BlockPool _pool;
        private readonly ContinuousBatchScheduler _scheduler;
        private readonly int _blockSize;
        private readonly ILogger _logger;

        // Currently-owning sequence (whose K/V state is in the model's tensors).
        private SequenceState _currentOwner;
        // Number of tokens the model currently holds for the current owner.
        // Equals model._cacheSeqLen for purely-attention models.
        private int _ownerTokensInModel;
        // Tokens forwarded for the current owner since the last ownership
        // change. Used by ExecuteStepPerSequence to rotate ownership at
        // DecodeQuantumTokens boundaries so a long-running owner doesn't
        // starve other scheduled sequences on the serial per-seq path.
        private int _ownerForwardedTokens;

        // ---- Live-cache continuation tracking (see ComputeLiveContinuationLcp) ----
        // The exact sequence whose tokens [0, _liveCacheLen) are currently resident
        // in the model's live KV cache, and whether that state is still trustworthy.
        // Set after every per-sequence forward; invalidated whenever the model's KV
        // cache is reset / rebuilt for a different sequence (or the batched path
        // takes over). Lets a same-session follow-up turn whose prompt extends this
        // sequence skip re-prefill entirely by continuing from the live cache —
        // critical for sliding-window models where the pooled snapshot can only
        // reuse one window.
        private SequenceState _liveCacheSeq;
        private int _liveCacheLen;
        private bool _liveCacheValid;

        // ---- Retained fused-cache continuation (cross-request prefix reuse) ----
        // The per-sequence fused path (concurrent N>=2 decode) keeps each request's
        // full K/V in its own holder and never writes the shared paged blocks, so a
        // finished concurrent request leaves nothing in the prefix-cache pool. For a
        // sliding-window model the pool can't restore a long prefix anyway, so a
        // multi-turn follow-up would re-prefill the whole conversation (KV reuse 0).
        // We retain a small LRU of finished fused holders (the model keeps the K/V
        // alive) keyed by their full token list, and re-adopt one for a later request
        // whose prompt exactly extends it — the cross-request analogue of the
        // single-stream live-cache continuation. See ComputeFusedContinuationLcp.
        private sealed class RetainedFusedCache
        {
            public string RequestId;   // model holder key (retained, not active)
            public int[] Tokens;       // full prompt+output tokens the holder's K/V covers
        }
        // Most-recently-retained at the tail; evict from the head.
        private readonly LinkedList<RetainedFusedCache> _retainedFused = new();
        // In-flight fused sequences by RequestId, so the release hook can snapshot
        // a finishing sequence's tokens (the release notification only carries an id).
        private readonly Dictionary<string, SequenceState> _fusedSeqById =
            new(StringComparer.Ordinal);

        // Re-used scratch buffer for inject/extract. Sized to one full block.
        private byte[] _scratch;

        // ---- NextN/MTP speculative decoding (see MtpSpeculativeExecution) ----
        // At most one sequence at a time runs speculatively: the draft head's
        // KV cache and pending hidden state live in the model's single live
        // (linear) cache, so continuity is (sequence identity, exact trunk
        // position). Any KV rebuild/swap — ownership change, batched/fused
        // step, preemption — invalidates the context; it re-arms only at a
        // fresh full prefill from position 0.
        private MtpSeqContext _mtpCtx;

        // One-time warning when --mtp-spec is requested but the model can't run its
        // accelerated MTP path on the current backend (speculation would be net-
        // negative), so the engine serves standard decode instead.
        private bool _mtpUnprofitableWarned;

        private sealed class MtpSeqContext
        {
            public SequenceState Seq;
            public MtpSpeculativeExecution Exec;
            // Non-null when the speculative trunk runs through the batched
            // paged path (IMtpBatchedSpeculativeModel) instead of the linear
            // cache. The trunk's own position must agree with NextPosition.
            public BatchedMtpTrunk BatchedTrunk;
            // Trunk position the next forward for Seq must start at; must equal
            // seq.NumComputedTokens (and, on the linear trunk, the model's
            // CacheSeqLen) to stay armed.
            public int NextPosition;
            // The token DRAWN from the last verify's mismatch/bonus row. It IS
            // the sequence's next output token (re-sampling from LastLogits
            // would bias toward the drafts); -1 when no draw is pending.
            public int PendingNextToken = -1;
        }

        /// <summary>Speculative trunk over the batched paged path: forwards go
        /// through <see cref="IMtpBatchedSpeculativeModel.SpecForwardBatched"/>
        /// (paged KV via the sequence's block table, per-slot recurrent
        /// state), so the spec trunk runs on the same kernels as the
        /// non-speculative batched baseline.</summary>
        internal sealed class BatchedMtpTrunk : IMtpSpecTrunk
        {
            private readonly IMtpBatchedSpeculativeModel _model;
            private readonly SequenceState _seq;

            /// <summary>Tokens committed to the trunk so far (advances with
            /// each Forward; rolled back on rejection).</summary>
            public int Position { get; private set; }

            public BatchedMtpTrunk(IMtpBatchedSpeculativeModel model, SequenceState seq, int position)
            {
                _model = model;
                _seq = seq;
                Position = position;
            }

            public void Forward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
            {
                _model.SpecForwardBatched(_seq, tokens, Position, hAllOut, logitsOut, allLogitsRows);
                Position += tokens.Length;
            }

            public void SnapshotRecurrentState() => _model.MtpSnapshotRecurrentStateSlots(_seq);

            public void Rollback(int position)
            {
                // Paged attention KV needs no rewind: every pass passes its
                // sequence length explicitly, and rejected slots are simply
                // overwritten by the kept-prefix re-forward / later steps.
                _model.MtpRestoreRecurrentStateSlots(_seq);
                Position = position;
            }
        }

        public BatchExecutor(
            IModelArchitecture model,
            BlockPool pool,
            ContinuousBatchScheduler scheduler,
            ILogger logger = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _pool = pool ?? throw new ArgumentNullException(nameof(pool));
            _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
            _blockSize = pool.BlockSize;
            _logger = logger ?? NullLogger.Instance;
        }

        public IModelArchitecture Model => _model;
        public SequenceState CurrentOwner => _currentOwner;

        /// <summary>Execute one scheduler step. When the underlying model
        /// implements <see cref="IBatchedPagedModel"/> all scheduled sequences
        /// are packed into a single <see cref="BatchedForwardContext"/> and
        /// dispatched through <see cref="IBatchedPagedModel.ForwardBatch"/> -
        /// this is the vLLM-style "one kernel many sequences" path. Otherwise
        /// the executor falls back to per-sequence forward with KV-state
        /// swap (the path that ships today).
        ///
        /// Set <c>TS_SCHED_DISABLE_BATCHED=1</c> to force the per-sequence
        /// fallback even when the model declares <see cref="IBatchedPagedModel"/>.
        /// Used to A/B the two paths on the same workload.</summary>
        public List<SequenceStepResult> ExecuteStep(SchedulerOutput output)
        {
            // Serialise GGML backend access with anything else that calls
            // into the same model (notably the chat pipeline's multimodal
            // vision/audio encoder, which runs on the request-handling
            // thread). Without this lock a parallel image-bearing request
            // and the engine's worker race the Metal command queue and
            // ggml_metal_synchronize aborts the process.
            lock (_model.GpuComputeLock)
            {
                // The scheduler freed the previous owner's blocks during
                // FinishSequence / PreemptSequence, but our _currentOwner
                // reference outlives that. Without this reset, the next
                // step's TryMigrateOwnerToPagedIfNeeded would call into
                // TryMigrateLinearKVToPaged with NumBlocks==0 (it returns
                // false and the executor logs a misleading "linear→paged
                // migration failed" warning), and EnsureOwnership on the
                // new sequence would try to extract state out of the dead
                // owner. Treat anything that's not Running as "no owner";
                // the model's linear cache will be reset cleanly when the
                // new sequence claims ownership.
                if (_currentOwner != null && _currentOwner.Status != SequenceStatus.Running)
                {
                    _currentOwner = null;
                    _ownerTokensInModel = 0;
                    _ownerForwardedTokens = 0;
                }

                bool batchedEnabled = !IsBatchedPathDisabled();

                // Split this step's work into:
                //   - multimodalWork: seqs whose injector bucket still has
                //     pending embeddings. These MUST go through per-sequence
                //     forward so the model can inject vision/audio
                //     embeddings at the right per-chunk positions (the
                //     batched paged kernel has no hook for this).
                //   - textWork: everything else; runs through the batched
                //     paged path for full continuous-batching throughput.
                // We route the two subsets separately rather than dragging
                // healthy text seqs onto the per-seq swap path just because
                // one image request happened to land in the same step -
                // legacy per-seq with concurrent multi-seq scheduling
                // produces garbled output on Gemma 4 (EnsureOwnership's
                // byte-level state swap doesn't isolate cleanly across
                // multi-seq iteration), so isolating multimodal here keeps
                // text correctness intact.
                // Models that handle multimodal in batched mode (e.g. Qwen3.5
                // Phase 4) declare SupportsBatchedMultimodal=true and run the
                // whole batch — multimodal + text — through ForwardBatch in
                // one shot. Other models still get the multimodal/text split
                // so per-seq Forward handles their vision-encoded sequences.
                bool batchedModalSafe = _model is IBatchedPagedModel mmCheck && mmCheck.SupportsBatchedMultimodal;
                var (multimodalWork, textWork) = batchedModalSafe
                    ? (new List<ScheduledSequenceWork>(), new List<ScheduledSequenceWork>(output.ScheduledWork))
                    : SplitMultimodalWork(output);

                // NextN/MTP speculative decoding. Models whose batched paged
                // path can serve the speculative trunk (paged KV + per-slot
                // recurrent state) run spec directly on that path — the same
                // kernels as the non-speculative batched baseline, no
                // linear-cache detour, composing with prefix caching and
                // concurrency transitions. When the handler declines (multi-
                // sequence step, disarmed context, prefix-reused admission)
                // the normal paths below serve the step. Models without
                // batched-trunk support fall back to the per-sequence
                // linear-cache route, which MUST run before the fused/batched
                // dispatch or speculation never engages.
                var mtpBatched = TryExecuteStepMtpBatchedTrunk(output);
                if (mtpBatched != null)
                    return mtpBatched;
                if (ShouldRouteMtpSpeculative(output))
                    return ExecuteStepPerSequence(output);

                // Per-sequence fused decode: for models that expose a fast fused
                // single-graph Forward backed by per-request KV caches, serve
                // concurrent (N>=2) sequences by running each through its own
                // fused Forward rather than the op-by-op batched paged path. The
                // batched path issues ~20 Metal-queue-draining dispatches per
                // layer, starving the GPU (~30% utilisation) so that aggregate
                // throughput at N=2 collapses below the single-stream rate; the
                // fused per-sequence path keeps the GPU saturated (one fused
                // decode graph per token per sequence).
                //
                // Multimodal sequences are handled by this path too: because
                // each request owns an isolated KV cache (no cross-sequence
                // byte-level swap), the per-seq Forward injects this request's
                // bucketed vision/audio embeddings into its own cache without
                // colliding with concurrent text sequences — the very isolation
                // the legacy multimodal/text split worked around on the swap
                // path. Routing everything through the fused path also avoids a
                // KV-storage split (fused sequences keep K/V in their linear
                // per-request cache, never the paged buffers the batched path
                // reads), which is what corrupted concurrent image+text output.
                if (batchedEnabled
                    && _model is IBatchedPagedModel fusedModel
                    && fusedModel.SupportsPerSequenceFusedForward
                    && IsPerSeqFusedEnabled()
                    && ShouldUsePerSeqFused(fusedModel, output))
                {
                    return ExecuteStepPerSequenceFused(fusedModel, output);
                }

                if (batchedEnabled && _model is IBatchedPagedModel batched && textWork.Count > 0 && multimodalWork.Count > 0)
                {
                    // Mixed batch (only reachable when the model declines batched
                    // multimodal): run each subset on its preferred path.
                    // The multimodal per-seq pass below may extract the current
                    // owner's linear K/V into pool blocks via EnsureOwnership
                    // before the text batched pass reads paged storage; if the
                    // owner's state was only ever in linear cache (came from the
                    // N=1 fast path), the batched pass would then attend to
                    // zeros for that owner. Migrate it up-front to keep the
                    // batched read consistent with where its K/V history lives.
                    TryMigrateOwnerToPagedIfNeeded(batched);
                    var results = new List<SequenceStepResult>(output.ScheduledWork.Count);
                    results.AddRange(ExecuteStepPerSequence(MakeSubOutput(multimodalWork)));
                    try
                    {
                        results.AddRange(ExecuteStepBatched(batched, MakeSubOutput(textWork)));
                        foreach (var work in textWork)
                            work.Sequence.KvStateInPagedStorage = true;
                    }
                    catch (NotSupportedException)
                    {
                        results.AddRange(ExecuteStepPerSequence(MakeSubOutput(textWork)));
                    }
                    return results;
                }

                if (batchedEnabled && _model is IBatchedPagedModel batched2 && multimodalWork.Count == 0 && output.ScheduledWork.Count > 0)
                {
                    // N=1 fast path: when there's only one scheduled sequence
                    // and the model has a fused single-pass decode kernel
                    // (e.g. Gemma 4's NativeGemma4ModelDecode), the per-seq
                    // Forward path is dramatically faster than the batched
                    // op-by-op path — ForwardBatch makes ~10 Ops.* dispatches
                    // per layer × 42 layers = ~420 kernel dispatches per token,
                    // while Forward submits the whole 42-layer decode as a
                    // single ggml graph compute. Profiling shows a ~4x speedup
                    // (≈21 tok/s vs ≈5 tok/s for Gemma 4 E4B Q8_0 on Metal).
                    //
                    // Safety: Forward writes K/V into the model's LINEAR cache,
                    // while ForwardBatch reads from PAGED buffers. When a
                    // second concurrent request arrives mid-decode, the
                    // scheduler emits both sequences in one step (N=2) which
                    // falls through to ExecuteStepBatched - and the first
                    // sequence's prior K/V is absent from paged storage,
                    // producing a token-repeat loop. We avoid that two ways:
                    //   1. Skip the fast path for sequences that have already
                    //      committed state to paged storage
                    //      (KvStateInPagedStorage), since Forward would attend
                    //      against an empty linear cache.
                    //   2. Restrict the fast path to models that EITHER implement
                    //      TryMigrateLinearKVToPaged (so the upcoming N=2
                    //      transition can copy the linear state into paged
                    //      storage before the batched kernel reads it) OR serve
                    //      N>=2 through the per-sequence FUSED path
                    //      (SupportsPerSequenceFusedForward) — those models never
                    //      read paged storage for concurrent decode, so the N=2
                    //      transition is handled by AdoptPrimaryCacheToFused (the
                    //      owner's linear cache is moved into its per-request
                    //      holder) and no linear→paged migration is needed.
                    //      This second case is what keeps BLOCK-QUANTIZED KV
                    //      caches (q4_0 / q8_0) on the fast path: their
                    //      ForwardBatch throws NotSupported, so they report
                    //      SupportsLinearKVMigration=false, but they still own a
                    //      single linear cache exactly like the f16 fast path.
                    //      Without this, block-quant N=1 steps fell into the
                    //      ExecuteStepBatched attempt below, which clears
                    //      _liveCacheValid *before* ForwardBatch throws — that
                    //      stale-false flag then aborted same-session live-cache
                    //      continuation ("请继续" follow-ups), so multi-turn KV
                    //      reuse dropped to 0 for q4_0/q8_0 while f16 reused fully.
                    //
                    // Gate: SupportsKVStateSnapshot is only relevant when
                    // ExecuteStepPerSequence has to *swap* ownership (i.e.
                    // the scheduled sequence isn't the current owner). For
                    // an in-progress single-seq decode the owner IS the
                    // scheduled sequence, so no swap happens and the
                    // snapshot capability is irrelevant.
                    //
                    // Disable entirely with TS_BATCHED_N1_FAST_PATH=0 if you
                    // need to A/B against the batched-only path.
                    if (IsBatchedN1FastPathEnabled()
                        && output.ScheduledWork.Count == 1
                        && (batched2.SupportsLinearKVMigration
                            || batched2.SupportsPerSequenceFusedForward))
                    {
                        var only = output.ScheduledWork[0].Sequence;
                        bool noSwapNeeded =
                            _currentOwner == null
                            || ReferenceEquals(_currentOwner, only);
                        bool sequenceStillInLinearCache = !only.KvStateInPagedStorage;
                        if ((noSwapNeeded || _model.SupportsKVStateSnapshot)
                            && sequenceStillInLinearCache)
                        {
                            return ExecuteStepPerSequence(output);
                        }
                    }

                    // Before dispatching through ForwardBatch, make sure any
                    // sequence whose K/V history only lives in the linear
                    // cache (because it was being served by the N=1 fast
                    // path, or by a previous step that fell through from a
                    // model whose ForwardBatch threw NotSupported) is
                    // migrated into paged storage. Without this the
                    // batched paged-attention kernel would read zeros for
                    // the owner's prior positions and the sequence would
                    // emit a token-repeat loop.
                    if (!TryMigrateOwnerToPagedIfNeeded(batched2))
                    {
                        // Two reasons we'd land here:
                        //   1. Model exposes SupportsLinearKVMigration but
                        //      TryMigrateLinearKVToPaged returned false for
                        //      this owner (e.g. a transient layout edge
                        //      case). Worth flagging so we can diagnose.
                        //   2. Model doesn't expose migration at all and an
                        //      owner accumulated linear-only state because
                        //      its ForwardBatch keeps throwing NotSupported
                        //      (e.g. a model whose batched path is gated off
                        //      via env var so every step is served by the
                        //      per-seq fallback anyway). This is expected,
                        //      not a failure — silently route to per-seq.
                        // In both cases the batched path would corrupt this
                        // owner; serve via per-seq where the linear cache
                        // is still authoritative.
                        if (batched2.SupportsLinearKVMigration)
                        {
                            _logger.LogWarning(
                                "BatchExecutor: linear→paged migration failed for {RequestId}; falling back to per-seq path.",
                                _currentOwner?.RequestId);
                        }
                        return ExecuteStepPerSequence(output);
                    }

                    try
                    {
                        var results = ExecuteStepBatched(batched2, output);
                        // Any sequence that just ran through the batched
                        // path now has its K/V in paged storage; sticky-
                        // mark it so future steps don't try to send it back
                        // through the linear-cache-only N=1 fast path.
                        foreach (var work in output.ScheduledWork)
                            work.Sequence.KvStateInPagedStorage = true;
                        return results;
                    }
                    catch (NotSupportedException)
                    {
                        // The model declared support but bailed for this specific
                        // batch. Fall through to the per-sequence swap path.
                    }
                }
                return ExecuteStepPerSequence(output);
            }
        }

        /// <summary>Migrate the current owner's K/V state from the legacy
        /// linear cache (populated by Forward, including the N=1 fast path
        /// and the per-seq fallback used when ForwardBatch throws
        /// NotSupported) into the model's paged storage so the upcoming
        /// ForwardBatch sees the owner's full history.
        ///
        /// Returns true when migration succeeded or no migration was needed
        /// (no owner, or owner already in paged storage). Returns false
        /// when migration was needed but cannot proceed — either because
        /// the model doesn't expose migration at all (common case for
        /// models whose batched path is gated off, so every step runs
        /// per-seq and accumulates linear-only state) or because the model
        /// supports migration but TryMigrateLinearKVToPaged bailed for this
        /// owner.
        /// The caller distinguishes the two via SupportsLinearKVMigration
        /// to decide whether the situation is expected (silent fallback)
        /// or worth logging (real migration failure).</summary>
        private bool TryMigrateOwnerToPagedIfNeeded(IBatchedPagedModel batched)
        {
            if (_currentOwner == null
                || _ownerTokensInModel <= 0
                || _currentOwner.KvStateInPagedStorage)
            {
                return true;
            }
            if (!batched.SupportsLinearKVMigration)
            {
                return false;
            }
            if (batched.TryMigrateLinearKVToPaged(_currentOwner, _blockSize))
            {
                _currentOwner.KvStateInPagedStorage = true;
                return true;
            }
            return false;
        }

        /// <summary>True when this step should be served by the per-sequence
        /// path so MTP speculative decoding can engage (or keep a sequence
        /// that already lives in the linear cache on a correct path).
        /// Speculation itself additionally requires an armed context — a
        /// routed sequence that can't arm (e.g. prefix-cache reuse skipped the
        /// full prefill) simply runs plain per-seq decode.</summary>
        private bool ShouldRouteMtpSpeculative(SchedulerOutput output)
        {
            if (!_scheduler.Config.MtpSpeculativeEnabled) return false;
            if (_model is not IMtpSpeculativeModel spec || !spec.HasMtp) return false;
            // Speculation is net-negative when the model can't drive its
            // accelerated MTP path on the current backend (the per-op verify/draft
            // fallback doesn't amortize the trunk over the draft window). Serve the
            // fast standard decode instead, with a single operator-facing notice.
            if (!spec.MtpSpeculationProfitable)
            {
                WarnMtpUnprofitableOnce();
                return false;
            }
            // Batched-trunk-capable models never take the per-sequence
            // detour: when TryExecuteStepMtpBatchedTrunk declines, the normal
            // batched path serves the step (faster and keeps the sequence's
            // K/V in paged storage).
            if (spec is IMtpBatchedSpeculativeModel bm && bm.SupportsBatchedSpecTrunk) return false;
            if (output.ScheduledWork.Count != 1) return false;

            var seq = output.ScheduledWork[0].Sequence;
            // A sequence whose K/V history lives in the model's paged storage
            // can't run on the per-seq path (Forward would attend against an
            // empty linear cache); keep it on the batched path.
            if (seq.KvStateInPagedStorage) return false;
            // Multimodal prefill needs Forward's embedding-inject hook, which
            // SpecForward doesn't have.
            if (_model.MultimodalInjector != null
                && _model.MultimodalInjector.HasPendingEmbeddings(seq.RequestId))
            {
                return false;
            }
            // A sequence living in a per-request fused cache must stay fused —
            // its tail K/V isn't reconstructable from the shared caches.
            if (_model is IBatchedPagedModel fused
                && fused.SupportsPerSequenceFusedForward
                && fused.HasFusedSequenceCache(seq.RequestId))
            {
                return false;
            }
            return true;
        }

        private void WarnMtpUnprofitableOnce()
        {
            if (_mtpUnprofitableWarned) return;
            _mtpUnprofitableWarned = true;
            _logger.LogWarning(
                "MTP speculative decoding was requested (--mtp-spec) but for the loaded model " +
                "on this backend the standard decode path is already faster than speculative " +
                "decode (its multi-token verify/draft runs op-by-op and cannot amortize a cheap, " +
                "fused/captured decode). Serving the fast standard decode instead — no action needed.");
        }

        private static bool IsBatchedN1FastPathEnabled()
        {
            // Default ON. The bug that previously made this dangerous (a
            // second concurrent request corrupting the first's output via
            // paged-vs-linear KV-cache divergence) is now handled by the
            // linear→paged migration at the N=1→batched transition, gated
            // on IBatchedPagedModel.SupportsLinearKVMigration. Disable via
            // TS_BATCHED_N1_FAST_PATH=0 to A/B the fully-batched path.
            string raw = Environment.GetEnvironmentVariable("TS_BATCHED_N1_FAST_PATH");
            if (string.IsNullOrEmpty(raw)) return true;
            return raw != "0" && !string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsBatchedPathDisabled()
        {
            string raw = Environment.GetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED");
            return !string.IsNullOrEmpty(raw) && raw != "0" && raw.ToLowerInvariant() != "false";
        }

        private (List<ScheduledSequenceWork> multimodal, List<ScheduledSequenceWork> text) SplitMultimodalWork(SchedulerOutput output)
        {
            var multimodal = new List<ScheduledSequenceWork>();
            var text = new List<ScheduledSequenceWork>();
            var injector = _model.MultimodalInjector;
            foreach (var work in output.ScheduledWork)
            {
                if (injector != null && injector.HasPendingEmbeddings(work.Sequence.RequestId))
                    multimodal.Add(work);
                else
                    text.Add(work);
            }
            return (multimodal, text);
        }

        private static SchedulerOutput MakeSubOutput(List<ScheduledSequenceWork> work)
        {
            var sub = new SchedulerOutput();
            foreach (var w in work) sub.ScheduledWork.Add(w);
            return sub;
        }

        private List<SequenceStepResult> ExecuteStepBatched(IBatchedPagedModel batched, SchedulerOutput output)
        {
            // NOTE: we must NOT clear _liveCacheValid here. Some models
            // (e.g. Gemma 4 with a block-quantized KV cache) decline the
            // batched path by throwing NotSupportedException from ForwardBatch,
            // and the caller falls back to ExecuteStepPerSequence. If we had
            // already cleared the live-cache claim, that fallback's
            // EnsureOwnership would see a stale-false flag and abort an
            // in-flight live-cache continuation, re-prefilling the whole
            // conversation. The claim is only genuinely invalidated once the
            // batched kernel actually ran and moved K/V into paged storage, so
            // we clear it AFTER ForwardBatch returns (below).
            int numSeqs = output.ScheduledWork.Count;
            var ctx = new BatchedForwardContext
            {
                Sequences = new List<SequenceState>(numSeqs),
                NumScheduledTokens = new List<int>(numSeqs),
                QueryStartLoc = new List<int>(numSeqs + 1),
                Positions = new List<int>(),
                SlotMapping = new List<int>(),
                BlockTables = new int[numSeqs][],
                MaxQueryLen = 0,
                MaxSeqLen = 0,
            };

            int cursor = 0;
            ctx.QueryStartLoc.Add(0);

            // Pre-fill tokens-to-forward array (decode samples a token first).
            var pendingTokens = new List<int[]>(numSeqs);
            for (int s = 0; s < numSeqs; s++)
            {
                var work = output.ScheduledWork[s];
                var seq = work.Sequence;

                int[] inputTokens;
                if (work.IsPrefill)
                {
                    inputTokens = BuildPrefillChunk(seq, work);
                }
                else
                {
                    int sampledFirst = SampleFromLogits(seq);
                    seq.AppendOutputToken(sampledFirst);
                    inputTokens = new[] { sampledFirst };
                }
                pendingTokens.Add(inputTokens);

                ctx.Sequences.Add(seq);
                ctx.NumScheduledTokens.Add(inputTokens.Length);

                // Per-token positions for this sequence + slot mappings.
                int startPos = seq.NumComputedTokens;
                for (int t = 0; t < inputTokens.Length; t++)
                {
                    int absPos = startPos + t;
                    ctx.Positions.Add(absPos);
                    int blockIdx = absPos / _blockSize;
                    int blockOffset = absPos % _blockSize;
                    int physBlockId = seq.BlockTable.Blocks[blockIdx].Id;
                    ctx.SlotMapping.Add(physBlockId * _blockSize + blockOffset);
                    cursor++;
                }
                ctx.QueryStartLoc.Add(cursor);

                int seqLen = startPos + inputTokens.Length;
                if (inputTokens.Length > ctx.MaxQueryLen) ctx.MaxQueryLen = inputTokens.Length;
                if (seqLen > ctx.MaxSeqLen) ctx.MaxSeqLen = seqLen;

                // Block table for this sequence.
                int numBlocks = seq.BlockTable.NumBlocks;
                var table = new int[numBlocks];
                for (int b = 0; b < numBlocks; b++)
                    table[b] = seq.BlockTable.Blocks[b].Id;
                ctx.BlockTables[s] = table;
            }

            // Dispatch the entire batch.
            var swForward = Stopwatch.StartNew();
            IReadOnlyList<float[]> perSeqLogits = batched.ForwardBatch(ctx);
            swForward.Stop();

            // The batched kernel actually ran: K/V now lives in paged storage
            // (slot mapping), not the single live linear cache the continuation
            // fast-path tracks, so drop any live-cache claim now that it is
            // genuinely stale. (If ForwardBatch had thrown, we never reach here
            // and the live-cache claim is correctly preserved for the per-seq
            // fallback.) Batched steps also clobber the linear-cache world the
            // speculative context depends on (per-slot GDN state swaps), so the
            // MTP context must re-arm from a fresh prefill.
            _liveCacheValid = false;
            _mtpCtx = null;

            if (perSeqLogits == null || perSeqLogits.Count != numSeqs)
                throw new InvalidOperationException(
                    $"ForwardBatch returned {perSeqLogits?.Count ?? -1} results for {numSeqs} sequences.");

            // Update per-sequence state and assemble results.
            var results = new List<SequenceStepResult>(numSeqs);
            for (int s = 0; s < numSeqs; s++)
            {
                var work = output.ScheduledWork[s];
                var seq = work.Sequence;
                var inputTokens = pendingTokens[s];

                seq.LastLogits = perSeqLogits[s];
                seq.AdvanceComputedTokens(inputTokens.Length);
                // In the batched path the model owns its own K/V layout
                // (paged storage referenced by slotMapping/block tables), so
                // the executor does NOT need to extract/inject between
                // sequences. _currentOwner stays null; subsequent batched
                // steps don't pay any swap cost.
                int sampled = work.IsPrefill ? -1 : inputTokens[0];
                if (!seq.FirstTokenAt.HasValue && sampled >= 0)
                    seq.FirstTokenAt = DateTime.UtcNow;

                results.Add(new SequenceStepResult
                {
                    Sequence = seq,
                    TokensForwarded = inputTokens.Length,
                    SampledToken = sampled,
                    IsPrefill = work.IsPrefill,
                    FullBlocksCaptured = 0, // batched path writes directly to blocks via slotMapping
                    ForwardElapsedTicks = swForward.ElapsedTicks / numSeqs,
                });

                // Notify the scheduler that any full blocks completed by
                // this step should enter the prefix-cache index. We pass the
                // pre-step computed-token count so it knows which blocks
                // are "newly full".
                int prevComputed = seq.NumComputedTokens - inputTokens.Length;
                int prevFullBlocks = prevComputed / _blockSize;
                _scheduler.OnBlocksCommitted(seq, prevFullBlocks * _blockSize);
            }
            return results;
        }

        private static bool IsPerSeqFusedEnabled()
        {
            // Default ON. Set TS_PER_SEQ_FUSED=0 to force concurrent requests
            // back onto the op-by-op batched paged path (A/B / debugging).
            string raw = Environment.GetEnvironmentVariable("TS_PER_SEQ_FUSED");
            if (string.IsNullOrEmpty(raw)) return true;
            return raw != "0" && !string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsBatchedFusedDecodeEnabled()
        {
            // Default OFF (v1, correctness-first). Set TS_BATCHED_FUSED_DECODE=1 to
            // route concurrent (N>=2) decode steps through the model's true
            // token-batched fused decode (one graph for all N) instead of the
            // round-robin per-seq forwards.
            string raw = Environment.GetEnvironmentVariable("TS_BATCHED_FUSED_DECODE");
            return raw == "1" || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase);
        }

        private static bool ShouldUsePerSeqFused(IBatchedPagedModel fused, SchedulerOutput output)
        {
            int count = output.ScheduledWork.Count;
            if (count == 0) return false;
            if (count >= 2) return true;
            // Single scheduled sequence: keep it on the fused path only if it
            // already owns a per-request fused cache. Such a sequence cannot
            // drop back to the single-stream/batched path because its tail K/V
            // lives only in its per-request linear cache (not reconstructable
            // from paged storage). A never-fused single sequence falls through
            // to the existing N==1 fast path (which keeps prefix-cache reuse
            // and live-cache continuation).
            return fused.HasFusedSequenceCache(output.ScheduledWork[0].Sequence.RequestId);
        }

        /// <summary>Run every scheduled sequence through the model's fused
        /// single-graph <see cref="IModelArchitecture.Forward"/> with its own
        /// per-request KV cache (bound via
        /// <see cref="IBatchedPagedModel.BindSequenceCache"/>). This is the
        /// high-throughput path for N&gt;=2 concurrency: each sequence's forward
        /// is one fused GPU graph (e.g. NativeGemma4ModelDecode), keeping the
        /// device saturated, instead of the op-by-op batched paged path whose
        /// per-op Metal-queue drains leave the GPU idle.
        ///
        /// No cross-sequence KV swap happens (each request owns its cache), so
        /// sliding-window models stay correct. Prefix-cache REUSE is honoured
        /// by injecting the reused prefix into a freshly-created cache once;
        /// the path does not itself CAPTURE blocks back into the shared pool
        /// (a concurrent request never writes the shared block storage), so it
        /// can't corrupt blocks shared via copy-on-write.</summary>
        private List<SequenceStepResult> ExecuteStepPerSequenceFused(IBatchedPagedModel fused, SchedulerOutput output)
        {
            int n = output.ScheduledWork.Count;
            var results = new List<SequenceStepResult>(n);
            if (n == 0) return results;

            // Transition from the single-stream (N==1) path: if a prior owner's
            // K/V is still live in the model's primary cache, hand it to that
            // request's per-request holder (zero-copy) so its history is
            // preserved and the primary is freed for future N==1 use.
            if (_currentOwner != null)
            {
                if (_currentOwner.Status == SequenceStatus.Running)
                    fused.AdoptPrimaryCacheToFused(_currentOwner.RequestId);
                _currentOwner = null;
                _ownerTokensInModel = 0;
                _ownerForwardedTokens = 0;
            }
            // The per-request caches make the single shared live-cache tracking
            // meaningless; drop any claim so a later same-session N==1 turn
            // re-establishes it cleanly from the primary cache.
            _liveCacheValid = false;
            // Fused per-request caches replace the shared linear cache the
            // speculative context tracks.
            _mtpCtx = null;

            // ---- TRUE token-batched decode fast path (opt-in) ----
            // When every scheduled item is a decode step (n>=2) and the model
            // supports it, decode all N tokens in ONE fused graph (weights loaded
            // once) instead of N serial per-seq forwards. Falls through to the
            // round-robin loop below when the model declines this batch.
            if (IsBatchedFusedDecodeEnabled() && n >= 2)
            {
                bool allDecode = true;
                foreach (var work in output.ScheduledWork)
                    if (work.IsPrefill) { allDecode = false; break; }
                if (allDecode)
                {
                    var reqIds = new string[n];
                    var btokens = new int[n];
                    var bpositions = new int[n];
                    for (int i = 0; i < n; i++)
                    {
                        var seq = output.ScheduledWork[i].Sequence;
                        // Peek-sample (deterministic for greedy); do NOT append yet
                        // so the round-robin fallback re-samples cleanly.
                        btokens[i] = SampleFromLogits(seq);
                        bpositions[i] = seq.NumComputedTokens;
                        reqIds[i] = seq.RequestId;
                    }
                    var outLogits = new float[n][];
                    if (fused.TryForwardBatchedFusedDecode(reqIds, btokens, bpositions, outLogits))
                    {
                        for (int i = 0; i < n; i++)
                        {
                            var seq = output.ScheduledWork[i].Sequence;
                            seq.AppendOutputToken(btokens[i]);
                            seq.LastLogits = outLogits[i];
                            seq.AdvanceComputedTokens(1);
                            if (!seq.FirstTokenAt.HasValue) seq.FirstTokenAt = DateTime.UtcNow;
                            results.Add(new SequenceStepResult
                            {
                                Sequence = seq,
                                TokensForwarded = 1,
                                SampledToken = btokens[i],
                                IsPrefill = false,
                                FullBlocksCaptured = 0,
                            });
                        }
                        return results;
                    }
                    // else: not appended — fall through to the round-robin loop.
                }
            }

            foreach (var work in output.ScheduledWork)
            {
                var seq = work.Sequence;
                int prevComputed = seq.NumComputedTokens;
                // Track this fused sequence so a clean finish can retain its holder
                // for cross-request prefix reuse (see TryRetainReleasedFusedCache).
                NoteFusedSequence(seq);
                try
                {
                    bool freshCache = fused.BindSequenceCache(seq.RequestId);

                    // Prefix-cache reuse: a freshly-created cache whose sequence
                    // was admitted with already-computed (reused) tokens needs
                    // that prefix injected from the shared paged blocks before
                    // its first forward. (Injection READS the shared blocks into
                    // this request's own cache; it never writes them.)
                    if (freshCache && seq.NumComputedTokens > 0)
                        InjectAllBlocks(seq, seq.NumComputedTokens);

                    int sampledToken = -1;
                    int[] inputTokens;
                    if (work.IsPrefill)
                    {
                        inputTokens = BuildPrefillChunk(seq, work);
                    }
                    else
                    {
                        sampledToken = SampleFromLogits(seq);
                        seq.AppendOutputToken(sampledToken);
                        inputTokens = new[] { sampledToken };
                    }

                    // Multimodal prefill chunks queue their overlapping
                    // embeddings so Forward injects them at the right positions
                    // (bucketed per RequestId).
                    if (_model.MultimodalInjector != null && work.IsPrefill)
                    {
                        _model.MultimodalInjector.QueuePromptEmbeddingsForSlice(
                            prevComputed, inputTokens.Length, seq.RequestId);
                    }

                    var swForward = Stopwatch.StartNew();
                    float[] logits = _model.Forward(inputTokens);
                    swForward.Stop();

                    // Every sequence's forward overwrites the model's shared
                    // logits buffer, so each must be cloned before the next
                    // sequence's forward in this same step.
                    seq.LastLogits = (float[])logits.Clone();

                    seq.AdvanceComputedTokens(inputTokens.Length);

                    if (!seq.FirstTokenAt.HasValue && sampledToken >= 0)
                        seq.FirstTokenAt = DateTime.UtcNow;

                    results.Add(new SequenceStepResult
                    {
                        Sequence = seq,
                        TokensForwarded = inputTokens.Length,
                        SampledToken = sampledToken,
                        IsPrefill = work.IsPrefill,
                        FullBlocksCaptured = 0,
                        ForwardElapsedTicks = swForward.ElapsedTicks,
                    });
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Fused per-seq step failed for sequence {RequestId}", seq.RequestId);
                    results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
                    seq.Error = ex;
                }
            }
            return results;
        }

        private List<SequenceStepResult> ExecuteStepPerSequence(SchedulerOutput output)
        {
            var results = new List<SequenceStepResult>(1);
            if (output.ScheduledWork.Count == 0)
                return results;

            // If a per-sequence-fused episode preceded this single-stream step,
            // the model's active KV cache may be a per-request holder. Reinstate
            // the primary cache before the in-place reset/inject logic below so
            // we never clobber a (possibly still-running) concurrent request's
            // cache. No-op when the primary cache is already active or the model
            // doesn't use per-request caches.
            if (_model is IBatchedPagedModel pf && pf.SupportsPerSequenceFusedForward)
                pf.RestorePrimaryCache();

            // The byte-level KV-state extract/inject in EnsureOwnership does
            // not correctly snapshot models with circular / sliding-window
            // caches (e.g. Gemma 4's 512-token SWA window): swapping out
            // mid-step and back in loses positions that have wrapped, so two
            // sequences interleaving on the same step produce garbled output.
            // Forward AT MOST ONE sequence per call here and let the
            // scheduler re-emit the others on subsequent steps. This makes
            // concurrent requests serially correct on this path (the price
            // of running without continuous batching) instead of corrupting
            // them. The batched paged path handles N-seq fan-out via slot
            // mapping and is not affected.
            //
            // Selection policy:
            //   1. Default to the first scheduled work.
            //   2. If a freshly-admitted (IsNewAdmission) non-owner is in the
            //      schedule AND the model can safely swap KV state, preempt
            //      the owner to give the new request its first token within
            //      one step (otherwise it would have to wait for the owner's
            //      full DecodeQuantumTokens streak to elapse).
            //   3. Else, when multiple sequences are scheduled and the
            //      current owner has accumulated DecodeQuantumTokens
            //      consecutive forwarded tokens, rotate to the first
            //      non-owner. Without this, an in-progress decode keeps
            //      pinning ownership and starves every other scheduled seq
            //      indefinitely (e.g. seq1 streaming while seq2 sits in
            //      prefill waiting for a turn it never gets).
            //   4. Else, prefer the current owner to amortize swap cost.
            //
            // Rotation only fires when SupportsKVStateSnapshot is true; that
            // gate preserves the original Gemma-4-with-wrapped-SWA-cache
            // safety property — when the swap is unsafe the model reports
            // false and we stay with the owner.
            var picked = output.ScheduledWork[0];
            if (_currentOwner != null && output.ScheduledWork.Count > 0)
            {
                // Rotating ownership to another sequence requires extracting the
                // current owner's state and injecting the newcomer's — a cross-
                // sequence snapshot round-trip. Models whose restore is not faithful
                // (Gemma 4 SWA) report SupportsCrossSequenceKvReuse=false, so we never
                // swap; concurrent sequences serialize on the owner instead of
                // producing corrupted output.
                bool canSwap = _model.SupportsKVStateSnapshot && _model.SupportsCrossSequenceKvReuse;
                int quantum = Math.Max(1, _scheduler.Config.DecodeQuantumTokens);
                bool quantumExceeded = canSwap && _ownerForwardedTokens >= quantum;

                ScheduledSequenceWork ownerWork = null;
                ScheduledSequenceWork firstNonOwner = null;
                ScheduledSequenceWork freshNonOwner = null;
                foreach (var candidate in output.ScheduledWork)
                {
                    if (ReferenceEquals(candidate.Sequence, _currentOwner))
                    {
                        ownerWork = candidate;
                    }
                    else
                    {
                        firstNonOwner ??= candidate;
                        if (canSwap && candidate.IsNewAdmission && freshNonOwner == null)
                            freshNonOwner = candidate;
                    }
                }

                if (freshNonOwner != null)
                    picked = freshNonOwner;
                else if (quantumExceeded && firstNonOwner != null)
                    picked = firstNonOwner;
                else if (ownerWork != null)
                    picked = ownerWork;
                else if (firstNonOwner != null)
                    picked = firstNonOwner;
            }

            {
                var work = picked;
                var seq = work.Sequence;
                int prevComputed = seq.NumComputedTokens;
                try
                {
                    EnsureOwnership(seq);

                    // NextN/MTP speculative decoding (handles its own advance/
                    // owner/live-cache/capture bookkeeping; null when the step
                    // must run on the plain path below).
                    var mtpResult = TryExecuteMtpStep(seq, work, prevComputed);
                    if (mtpResult != null)
                    {
                        results.Add(mtpResult);
                        return results;
                    }
                    // A context for this sequence that didn't serve the step is
                    // stale from here on: the plain Forward below advances the
                    // trunk without capturing the hidden states drafting needs.
                    if (_mtpCtx != null && ReferenceEquals(_mtpCtx.Seq, seq))
                        _mtpCtx = null;

                    // For decode steps we sample the next token from the
                    // sequence's last logits BEFORE forwarding it. The forward
                    // then runs on the freshly-sampled token; its returned
                    // logits drive the NEXT step's sample. For prefill we
                    // pass the next chunk of prompt tokens unchanged.
                    int sampledToken = -1;
                    int[] inputTokens;
                    if (work.IsPrefill)
                    {
                        inputTokens = BuildPrefillChunk(seq, work);
                    }
                    else
                    {
                        sampledToken = SampleFromLogits(seq);
                        seq.AppendOutputToken(sampledToken);
                        inputTokens = new[] { sampledToken };
                    }

                    // For multimodal sequences, queue the embeddings that
                    // overlap this prefill chunk into the model so Forward()
                    // can inject them at the right per-chunk positions.
                    // Engine-path callers (ChatGenerationPipeline + tests)
                    // bucket prepared embeddings by seq.RequestId so this
                    // looks up only THIS sequence's embeddings, even when
                    // other concurrent requests have their own pending media.
                    if (_model.MultimodalInjector != null && work.IsPrefill)
                    {
                        _model.MultimodalInjector.QueuePromptEmbeddingsForSlice(
                            prevComputed, inputTokens.Length, seq.RequestId);
                    }

                    var swForward = Stopwatch.StartNew();
                    float[] logits = _model.Forward(inputTokens);
                    swForward.Stop();

                    // Sampling happens at the *start* of the next step (see
                    // SampleFromLogits at the top of this branch), and
                    // BatchExecutor calls _model.Forward only once per step.
                    // The model's `_logitsBuffer` is overwritten on the next
                    // Forward, but we always sample before that next Forward
                    // fires — so a defensive 1 MB clone per token (Gemma 4
                    // vocab = 262144 × 4 bytes) is wasted memcpy and GC
                    // pressure (~20 µs / token). Borrow the model's buffer
                    // directly; the contract is: callers must consume
                    // LastLogits before this sequence's next forward.
                    //
                    // Exception: when a sequence is multi-step inactive
                    // (ownership-swapped or preempted), another sequence's
                    // forward could clobber our buffer. Clone in that case.
                    if (output.ScheduledWork.Count > 1)
                        seq.LastLogits = (float[])logits.Clone();
                    else
                        seq.LastLogits = logits;

                    seq.AdvanceComputedTokens(inputTokens.Length);
                    _ownerTokensInModel += inputTokens.Length;
                    _ownerForwardedTokens += inputTokens.Length;

                    // Record what the model's live KV cache now holds so a same-session
                    // follow-up turn can continue from it without re-prefilling. Valid
                    // until the cache is reset for a different sequence (EnsureOwnership)
                    // or the batched path takes over.
                    _liveCacheSeq = seq;
                    _liveCacheLen = seq.NumComputedTokens;
                    _liveCacheValid = true;

                    // Capture any newly-completed full blocks into the prefix cache.
                    int capturedFullBlocks = CaptureNewlyFullBlocks(seq);

                    if (!seq.FirstTokenAt.HasValue && sampledToken >= 0)
                        seq.FirstTokenAt = DateTime.UtcNow;

                    var result = new SequenceStepResult
                    {
                        Sequence = seq,
                        TokensForwarded = inputTokens.Length,
                        SampledToken = sampledToken,
                        IsPrefill = work.IsPrefill,
                        FullBlocksCaptured = capturedFullBlocks,
                        ForwardElapsedTicks = swForward.ElapsedTicks,
                    };
                    results.Add(result);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Step failed for sequence {RequestId}", seq.RequestId);
                    results.Add(new SequenceStepResult
                    {
                        Sequence = seq,
                        Error = ex,
                    });
                    // Reset ownership so the next step does a clean swap.
                    _currentOwner = null;
                    _ownerTokensInModel = 0;
                    _ownerForwardedTokens = 0;
                    _mtpCtx = null;
                    seq.Error = ex;
                }
            }

            return results;
        }

        /// <summary>Serve one scheduled work item via NextN/MTP speculative
        /// decoding on the LINEAR-cache trunk when the speculative context is
        /// (or can be) armed for it. Returns null when the step must run on
        /// the plain path instead. Assumes <see cref="EnsureOwnership"/> has
        /// already run for <paramref name="seq"/> (a fresh sequence therefore
        /// starts with a clean model cache). Models whose batched path can
        /// serve the speculative trunk never arm here — they are handled by
        /// <see cref="TryExecuteStepMtpBatchedTrunk"/> before the per-seq
        /// route is ever taken.</summary>
        private SequenceStepResult TryExecuteMtpStep(
            SequenceState seq, ScheduledSequenceWork work, int prevComputed)
        {
            if (!_scheduler.Config.MtpSpeculativeEnabled)
                return null;
            if (_model is not IMtpSpeculativeModel spec || !spec.HasMtp)
                return null;
            // Net-negative on backends without the accelerated MTP path; the normal
            // decode path serves the step (see ShouldRouteMtpSpeculative).
            if (!spec.MtpSpeculationProfitable)
                return null;
            if (spec is IMtpBatchedSpeculativeModel batchedSpec && batchedSpec.SupportsBatchedSpecTrunk)
                return null;
            // Multimodal prefill needs Forward's embedding-inject hook, which
            // SpecForward doesn't have.
            if (_model.MultimodalInjector != null
                && _model.MultimodalInjector.HasPendingEmbeddings(seq.RequestId))
            {
                return null;
            }

            // (Re-)arm at a fresh full prefill from position 0 with a clean
            // cache. A prefix-cache or live-cache adoption skips trunk
            // positions the MTP head never saw, so those admissions stay on
            // the plain path. Replaces any stale context (including this
            // sequence's own, e.g. after preemption + re-prefill).
            if (work.IsPrefill && prevComputed == 0 && spec.CacheSeqLen == 0)
            {
                var cfg = _scheduler.Config;
                _mtpCtx = new MtpSeqContext
                {
                    Seq = seq,
                    Exec = new MtpSpeculativeExecution(spec, cfg.MtpMaxDraftTokens)
                    {
                        MinDraftProb = cfg.MtpMinDraftProb,
                    },
                    NextPosition = 0,
                };
                seq.SpecStats = _mtpCtx.Exec.Stats;
                // One line per request so operators can SEE speculation engage
                // at generation start (the cumulative stats only log at finish).
                _logger.LogInformation(
                    "MTP speculative decoding armed for {RequestId} (maxDraft={MaxDraft}, pMin={PMin}, trunk=linear)",
                    seq.RequestId, cfg.MtpMaxDraftTokens, cfg.MtpMinDraftProb);
            }

            // Continuity gate: same sequence, exact trunk position, and the
            // model's live cache agrees. Anything else (swap, preemption,
            // interleaved batched step) ran through an invalidation above.
            if (_mtpCtx == null
                || _mtpCtx.BatchedTrunk != null
                || !ReferenceEquals(_mtpCtx.Seq, seq)
                || _mtpCtx.NextPosition != prevComputed
                || spec.CacheSeqLen != prevComputed)
            {
                return null;
            }

            return ExecuteMtpWorkCore(seq, work, prevComputed);
        }

        /// <summary>
        /// NextN/MTP speculative decoding with the trunk on the BATCHED paged
        /// path (see <see cref="IMtpBatchedSpeculativeModel"/>): solo text
        /// sequences arm at a fresh full prefill and run draft/verify with
        /// trunk passes through <c>SpecForwardBatched</c> — the same kernels
        /// the non-speculative batched baseline uses, with the sequence's K/V
        /// in paged storage throughout (prefix caching and concurrency
        /// transitions compose). Returns null when this step should be served
        /// by the normal paths (speculation disabled/unsupported, multi-
        /// sequence step, multimodal pending, disarmed context, prefix-reused
        /// admission); the normal batched path then drops any stale context.
        /// </summary>
        private List<SequenceStepResult> TryExecuteStepMtpBatchedTrunk(SchedulerOutput output)
        {
            if (!_scheduler.Config.MtpSpeculativeEnabled)
                return null;
            if (_model is not IMtpBatchedSpeculativeModel spec || !spec.SupportsBatchedSpecTrunk || !spec.HasMtp)
                return null;
            if (!spec.MtpSpeculationProfitable)
            {
                WarnMtpUnprofitableOnce();
                return null;
            }
            if (output.ScheduledWork.Count != 1)
                return null;
            var work = output.ScheduledWork[0];
            var seq = work.Sequence;
            if (_model.MultimodalInjector != null
                && _model.MultimodalInjector.HasPendingEmbeddings(seq.RequestId))
            {
                return null;
            }
            // A sequence living in a per-request fused cache must stay fused —
            // its tail K/V isn't reconstructable from paged storage.
            if (_model is IBatchedPagedModel fused
                && fused.SupportsPerSequenceFusedForward
                && fused.HasFusedSequenceCache(seq.RequestId))
            {
                return null;
            }

            int prevComputed = seq.NumComputedTokens;

            // (Re-)arm at a fresh full prefill from position 0. A prefix-cache
            // or live-cache adoption skips trunk positions the MTP head never
            // saw; those requests run on the normal batched path instead.
            if (work.IsPrefill && prevComputed == 0)
            {
                var cfg = _scheduler.Config;
                var trunk = new BatchedMtpTrunk(spec, seq, 0);
                _mtpCtx = new MtpSeqContext
                {
                    Seq = seq,
                    BatchedTrunk = trunk,
                    Exec = new MtpSpeculativeExecution(spec, cfg.MtpMaxDraftTokens, trunk)
                    {
                        MinDraftProb = cfg.MtpMinDraftProb,
                    },
                    NextPosition = 0,
                };
                seq.SpecStats = _mtpCtx.Exec.Stats;
                _logger.LogInformation(
                    "MTP speculative decoding armed for {RequestId} (maxDraft={MaxDraft}, pMin={PMin}, trunk=batched)",
                    seq.RequestId, cfg.MtpMaxDraftTokens, cfg.MtpMinDraftProb);
            }

            // Continuity gate: a batched context for this exact sequence at
            // this exact position. The drawn-token stash dies with a stale
            // context; the normal path re-samples from LastLogits (identical
            // under greedy, a one-token bias on rare disarm events otherwise).
            if (_mtpCtx == null
                || _mtpCtx.BatchedTrunk == null
                || !ReferenceEquals(_mtpCtx.Seq, seq)
                || _mtpCtx.NextPosition != prevComputed
                || _mtpCtx.BatchedTrunk.Position != prevComputed)
            {
                return null;
            }

            var results = new List<SequenceStepResult>(1);
            try
            {
                results.Add(ExecuteMtpWorkCore(seq, work, prevComputed));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Batched-trunk MTP step failed for sequence {RequestId}", seq.RequestId);
                _mtpCtx = null;
                seq.Error = ex;
                results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
            }
            return results;
        }

        /// <summary>Shared MTP step body for both trunks; the caller has
        /// already validated arming and continuity on <see cref="_mtpCtx"/>.</summary>
        private SequenceStepResult ExecuteMtpWorkCore(
            SequenceState seq, ScheduledSequenceWork work, int prevComputed)
        {
            bool batchedTrunk = _mtpCtx.BatchedTrunk != null;
            if (work.IsPrefill)
            {
                int[] chunk = BuildPrefillChunk(seq, work);
                var swPrefill = Stopwatch.StartNew();
                float[] logits = _mtpCtx.Exec.PrefillStep(chunk, prevComputed);
                swPrefill.Stop();
                seq.LastLogits = logits;
                CompleteMtpStepBookkeeping(seq, chunk.Length, batchedTrunk);
                _mtpCtx.NextPosition = seq.NumComputedTokens;

                return new SequenceStepResult
                {
                    Sequence = seq,
                    TokensForwarded = chunk.Length,
                    SampledToken = -1,
                    IsPrefill = true,
                    FullBlocksCaptured = batchedTrunk ? 0 : CaptureNewlyFullBlocks(seq),
                    ForwardElapsedTicks = swPrefill.ElapsedTicks,
                };
            }

            // ---- Speculative decode step ----
            // The next output token: the one DRAWN during the previous step's
            // verification when available (emitting anything else would bias
            // the stream toward the drafts), otherwise sampled as usual.
            int sampledToken;
            if (_mtpCtx.PendingNextToken >= 0)
            {
                sampledToken = _mtpCtx.PendingNextToken;
                _mtpCtx.PendingNextToken = -1;
            }
            else
            {
                sampledToken = SampleFromLogits(seq);
            }
            seq.AppendOutputToken(sampledToken);

            // Cap the draft window so this step's 1+K forwarded tokens fit in
            // (a) the request's remaining token budget and (b) the KV blocks
            // the scheduler allocated — it reserves capacity for ONE decode
            // token per step, so block-boundary steps degrade to plain decode
            // for one step instead of overrunning the block table.
            int kMax = Math.Min(
                seq.MaxNewTokens - seq.OutputTokens.Count,
                seq.BlockTable.FreeSlotsInCurrentBlocks - 1);

            var accepted = new List<int>();
            var penaltySampler = new TokenSampler(seq.SamplingConfig);
            var swDecode = Stopwatch.StartNew();
            MtpDecodeOutcome outcome = _mtpCtx.Exec.DecodeStep(
                sampledToken,
                prevComputed,
                kMax,
                // Each verify row is drawn with the request's own sampler over
                // the live output history (kept exact by onDraftAccepted).
                drawNext: rowLogits =>
                    new TokenSampler(seq.SamplingConfig).Sample(rowLogits, seq.OutputTokens),
                // Penalty-aligned drafting: the draft head must argmax the
                // same penalized distribution verification draws from, or
                // acceptance decays toward zero as the output history grows.
                adjustDraftLogits: (draftLogits, pendingDrafts) =>
                    penaltySampler.ApplyPenalties(draftLogits, seq.OutputTokens, pendingDrafts),
                onDraftAccepted: d =>
                {
                    seq.AppendOutputToken(d);
                    accepted.Add(d);
                });
            swDecode.Stop();

            seq.LastLogits = outcome.NextLogits;
            int advanced = 1 + outcome.AcceptedCount;
            CompleteMtpStepBookkeeping(seq, advanced, batchedTrunk);
            _mtpCtx.NextPosition = prevComputed + advanced;
            _mtpCtx.PendingNextToken = outcome.NextToken;

            int capturedBlocks = batchedTrunk ? 0 : CaptureNewlyFullBlocks(seq);
            if (!seq.FirstTokenAt.HasValue)
                seq.FirstTokenAt = DateTime.UtcNow;

            return new SequenceStepResult
            {
                Sequence = seq,
                TokensForwarded = advanced,
                SampledToken = sampledToken,
                ExtraTokens = accepted.Count > 0 ? accepted : null,
                IsPrefill = false,
                FullBlocksCaptured = capturedBlocks,
                ForwardElapsedTicks = swDecode.ElapsedTicks,
            };
        }

        /// <summary>Per-step bookkeeping for MTP steps (which may advance more
        /// than one token). Linear trunk mirrors the plain per-seq path
        /// (owner counters + live-cache tracking + the caller's block
        /// capture); batched trunk mirrors <see cref="ExecuteStepBatched"/>
        /// (K/V lives in the model's paged storage, blocks get hash-registered
        /// for prefix sharing).</summary>
        private void CompleteMtpStepBookkeeping(SequenceState seq, int tokensForwarded, bool batchedTrunk)
        {
            int prevComputed = seq.NumComputedTokens;
            seq.AdvanceComputedTokens(tokensForwarded);
            if (batchedTrunk)
            {
                _liveCacheValid = false;
                seq.KvStateInPagedStorage = true;
                int prevFullBlocks = prevComputed / _blockSize;
                _scheduler.OnBlocksCommitted(seq, prevFullBlocks * _blockSize);
            }
            else
            {
                _ownerTokensInModel += tokensForwarded;
                _ownerForwardedTokens += tokensForwarded;
                _liveCacheSeq = seq;
                _liveCacheLen = seq.NumComputedTokens;
                _liveCacheValid = true;
            }
        }

        /// <summary>
        /// Longest prompt prefix of <paramref name="seq"/> that can be served by
        /// continuing the model's LIVE KV cache (rather than the pooled snapshot),
        /// or 0 when live continuation doesn't apply. Returns a positive length only
        /// when ALL of:
        ///   - the model caps pooled prefix reuse (sliding-window / circular cache);
        ///   - a valid live cache from a prior sequence is resident;
        ///   - that prior sequence's entire token run is an exact prefix of this
        ///     prompt (the linear "continue the conversation" case);
        ///   - the reusable length exceeds what the pooled path could give (the cap);
        ///   - at least one new suffix token remains to forward.
        /// Invoked by the scheduler at admission. Thread-safety: the engine runs the
        /// scheduler and executor on the same worker thread, so the live-cache fields
        /// are not concurrently mutated here.
        /// </summary>
        public int ComputeLiveContinuationLcp(SequenceState seq)
        {
            if (seq == null || !_liveCacheValid || _liveCacheSeq == null || _liveCacheLen <= 0)
                return 0;
            if (!_model.SupportsKVStateSnapshot || !_model.SupportsCrossSequenceKvReuse)
                return 0;

            // Only worth it for capped models: an uncapped model already reuses the
            // full prefix through the pool (which also caches blocks for concurrent
            // sequences), so live continuation would add cost without benefit.
            int cap = _model.MaxReusablePrefixTokens;
            if (cap == int.MaxValue)
                return 0;

            int liveLen = Math.Min(_liveCacheLen, _liveCacheSeq.NumTotalTokens);
            if (liveLen <= cap)
                return 0; // pooled reuse already covers this prefix correctly
            if (seq.PromptTokens.Count <= liveLen)
                return 0; // no new suffix to forward (or prompt shorter than cache)

            // Require the entire live sequence to be an exact prefix of the new
            // prompt. This is the common multi-turn append ("请继续") case and avoids
            // truncating the circular cache (which would reintroduce the wrap issue).
            for (int i = 0; i < liveLen; i++)
            {
                if (seq.PromptTokens[i] != _liveCacheSeq.TokenAt(i))
                    return 0;
            }
            return liveLen;
        }

        /// <summary>Set <paramref name="seq"/> up to continue from the model's live
        /// KV cache for its first <paramref name="lcp"/> tokens: reserve blocks so
        /// block-table accounting matches, mark the reused prefix as computed, and
        /// flag the sequence so <see cref="EnsureOwnership"/> keeps the live cache
        /// instead of reset+inject. Returns false (caller falls back to the pooled
        /// path) if blocks can't be reserved. Invoked by the scheduler at admission.</summary>
        public bool TryAdoptLiveCache(SequenceState seq, int lcp)
        {
            if (seq == null || lcp <= 0) return false;
            if (seq.BlockTable.NumBlocks != 0) return false;

            int neededBlocks = (lcp + _blockSize - 1) / _blockSize;
            var blocks = _pool.AllocateNew(neededBlocks);
            if (blocks == null)
                return false; // pool pressure -> let the caller use the capped pool path

            for (int i = 0; i < blocks.Length; i++)
                seq.BlockTable.AppendBlock(blocks[i]);

            seq.SetComputedTokensForPrefixAdoption(lcp);
            seq.PrefixCacheReusedTokens = lcp;
            seq.UsesLiveCacheContinuation = true;
            return true;
        }

        private static bool IsRetainedFusedCacheEnabled()
        {
            // Default ON. Kill-switch for A/B or to cap VRAM use. The retained
            // holders pin one full KV cache each (same size as a concurrent
            // request's holder), so the budget below bounds the extra VRAM.
            string raw = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
            if (string.IsNullOrEmpty(raw)) return true;
            return raw != "0" && !string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase);
        }

        private static int RetainedFusedCacheBudget()
        {
            // How many finished fused holders to keep alive for cross-request
            // prefix reuse. Each pins a full per-request KV cache, so keep it
            // small (VRAM). Tune with TS_RETAINED_FUSED_CACHE_MAX.
            string raw = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX");
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int n) && n >= 0)
                return n;
            return 4;
        }

        /// <summary>True when the loaded model serves concurrent decode through
        /// per-request fused holders AND caps pooled prefix reuse (sliding-window /
        /// circular cache). Only such models need retained-fused continuation — an
        /// uncapped pure-attention model already reuses the full prefix through the
        /// shared pool. This targets Gemma 4 (the reported repro). Qwen 3.5/3.6 is
        /// deliberately excluded for now: it reports MaxReusablePrefixTokens=int.MaxValue
        /// (uncapped), and its recurrent GatedDeltaNet state can't be reconstructed
        /// from the pool either, so enabling it would need its own correctness pass on
        /// GDN-state reuse — a follow-up, not part of this fix.</summary>
        private bool ModelUsesRetainableFusedCache()
            => IsRetainedFusedCacheEnabled()
            && _model is IBatchedPagedModel f
            && f.SupportsPerSequenceFusedForward
            && _model.MaxReusablePrefixTokens != int.MaxValue;

        /// <summary>Longest retained fused-holder token run that is an EXACT prefix
        /// of <paramref name="seq"/>'s prompt, or 0 when no retained holder applies.
        /// The matched holder's K/V is the full circular cache from the finished
        /// request, so continuing from it reuses the entire conversation prefix (past
        /// the sliding-window cap) with no corruption — the cross-request analogue of
        /// <see cref="ComputeLiveContinuationLcp"/>. Invoked by the scheduler at
        /// admission (same worker thread as the executor).</summary>
        public int ComputeFusedContinuationLcp(SequenceState seq)
        {
            if (seq == null || _retainedFused.Count == 0) return 0;
            if (!ModelUsesRetainableFusedCache()) return 0;
            var match = FindRetainedFusedMatch(seq);
            return match?.Tokens.Length ?? 0;
        }

        /// <summary>Adopt a retained fused holder for <paramref name="seq"/>: re-key
        /// the model's retained K/V to this request (so its first fused
        /// <c>BindSequenceCache</c> continues from it), reserve placeholder blocks for
        /// accounting, and mark the reused prefix. Returns false (caller falls back to
        /// the pooled path) when the holder can't be reserved or re-keyed. Invoked by
        /// the scheduler at admission.</summary>
        public bool TryAdoptFusedContinuation(SequenceState seq, int lcp)
        {
            if (seq == null || lcp <= 0) return false;
            if (seq.BlockTable.NumBlocks != 0) return false;
            if (_model is not IBatchedPagedModel fused) return false;

            var match = FindRetainedFusedMatch(seq);
            if (match == null || match.Tokens.Length != lcp) return false;

            int neededBlocks = (lcp + _blockSize - 1) / _blockSize;
            var blocks = _pool.AllocateNew(neededBlocks);
            if (blocks == null)
                return false; // pool pressure -> let the caller use the capped pool path

            if (!fused.TryRebindRetainedCache(match.RequestId, seq.RequestId))
            {
                _pool.Free(blocks);
                return false;
            }

            for (int i = 0; i < blocks.Length; i++)
                seq.BlockTable.AppendBlock(blocks[i]);

            seq.SetComputedTokensForPrefixAdoption(lcp);
            seq.PrefixCacheReusedTokens = lcp;
            // The rebound holder is now this request's active fused cache; the
            // fused path's BindSequenceCache finds it (fresh==false) and continues
            // from it without injecting from the (empty) reserved blocks.
            _retainedFused.Remove(match);
            return true;
        }

        /// <summary>Find the retained fused holder whose token run is an exact prefix
        /// of <paramref name="seq"/>'s prompt and leaves at least one new suffix token
        /// to forward. Prefers the longest match.</summary>
        private RetainedFusedCache FindRetainedFusedMatch(SequenceState seq)
        {
            RetainedFusedCache best = null;
            foreach (var entry in _retainedFused)
            {
                int len = entry.Tokens.Length;
                // NB: no `len <= cap` skip. The fused path writes nothing to the shared
                // pool, so a retained holder is the ONLY reuse source for a concurrent
                // conversation — even one shorter than the sliding window.
                if (seq.PromptTokens.Count <= len) continue;    // no new suffix to forward
                if (best != null && len <= best.Tokens.Length) continue;
                bool prefix = true;
                for (int i = 0; i < len; i++)
                {
                    if (seq.PromptTokens[i] != entry.Tokens[i]) { prefix = false; break; }
                }
                if (prefix) best = entry;
            }
            return best;
        }

        /// <summary>Track an in-flight fused sequence so the release hook can snapshot
        /// its tokens. Called from the fused per-seq path for every scheduled seq.</summary>
        private void NoteFusedSequence(SequenceState seq)
        {
            if (ModelUsesRetainableFusedCache())
                _fusedSeqById[seq.RequestId] = seq;
        }

        /// <summary>Called by the engine when a sequence leaves the scheduler, BEFORE
        /// the model's <see cref="IBatchedPagedModel.OnSequenceReleased"/>. When the
        /// sequence finished cleanly on the fused path, retain its holder (the full
        /// circular K/V) for cross-request prefix reuse instead of letting the model
        /// dispose it. Returns true when the holder was retained (so the subsequent
        /// model release no-ops for it).</summary>
        public bool TryRetainReleasedFusedCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return false;
            if (!_fusedSeqById.TryGetValue(requestId, out var seq))
                return false;
            _fusedSeqById.Remove(requestId);

            if (!ModelUsesRetainableFusedCache()) return false;
            if (_model is not IBatchedPagedModel fused) return false;
            // Only retain clean finishes; aborted/errored sequences may hold
            // partial/inconsistent K/V, and preempted ones resume on their own.
            if (seq.Status != SequenceStatus.FinishedStopped
                && seq.Status != SequenceStatus.FinishedLengthCapped)
                return false;

            // Snapshot exactly the tokens whose K/V is resident in the holder
            // (NumComputedTokens == the model's _cacheSeqLen at finish), so a later
            // continuation's reused-prefix length matches the holder's cache extent
            // exactly. (At a clean finish this equals NumTotalTokens; clamp defends
            // against a speculative tail that advanced computed past the token list.)
            int len = Math.Min(seq.NumComputedTokens, seq.NumTotalTokens);
            // Retain any non-trivial fused conversation: the fused path contributes
            // nothing to the shared pool, so retention is the only cross-request reuse
            // source for it (not just the >window case). The LRU budget bounds VRAM.
            if (len < _blockSize) return false;
            if (!fused.RetainSequenceCache(requestId)) return false;

            var tokens = new int[len];
            for (int i = 0; i < len; i++) tokens[i] = seq.TokenAt(i);
            _retainedFused.AddLast(new RetainedFusedCache { RequestId = requestId, Tokens = tokens });

            // Evict oldest holders beyond the budget (frees their VRAM).
            int budget = RetainedFusedCacheBudget();
            while (_retainedFused.Count > budget)
            {
                var victim = _retainedFused.First.Value;
                _retainedFused.RemoveFirst();
                fused.DiscardRetainedCache(victim.RequestId);
            }
            return true;
        }

        /// <summary>Ensure the model's K/V state belongs to <paramref name="seq"/>.
        /// If a different sequence is the current owner, extract its state into
        /// its blocks, reset the model, and inject this sequence's state from
        /// its blocks.</summary>
        private void EnsureOwnership(SequenceState seq)
        {
            if (ReferenceEquals(_currentOwner, seq))
            {
                // Same owner: nothing to do. (Sanity check: model's cached count
                // should match the sequence's computed-token counter.)
                return;
            }

            // Any ownership change rebuilds the model's KV state (reset+inject
            // or live-cache adoption by a different request), which orphans the
            // MTP draft head's cache and pending hidden state.
            _mtpCtx = null;

            // Live-cache continuation: the new sequence's prompt extends exactly the
            // tokens still resident in the model's live KV cache (planned by the
            // scheduler via TryAdoptLiveCache). Keep the cache as-is and continue
            // from the reused prefix - no reset, no pooled inject. This is the only
            // way to reuse a prefix longer than a sliding-window model's window
            // without the lossy circular-cache snapshot reconstruction.
            if (seq.UsesLiveCacheContinuation)
            {
                if (_currentOwner == null
                    && _liveCacheValid
                    && seq.NumComputedTokens > 0
                    && _liveCacheLen == seq.NumComputedTokens)
                {
                    _currentOwner = seq;
                    _ownerTokensInModel = seq.NumComputedTokens;
                    _ownerForwardedTokens = 0;
                    return;
                }

                // The live cache was invalidated between scheduling and execution
                // (e.g. a concurrent sequence took ownership). Drop the reused-prefix
                // claim and re-prefill from scratch via the normal path below; the
                // sequence keeps its reserved blocks so accounting stays consistent.
                _logger.LogDebug(
                    "Live-cache continuation for {RequestId} no longer valid; re-prefilling.",
                    seq.RequestId);
                seq.ClearLiveCacheContinuation();
            }

            // Swap out the previous owner.
            if (_currentOwner != null)
            {
                if (_model.SupportsKVStateSnapshot && _ownerTokensInModel > 0)
                {
                    ExtractAllBlocks(_currentOwner, _ownerTokensInModel);
                }
                // Else: model can't snapshot; the previous owner's state is
                // lost. (This path forces re-prefill on the next admission.)
            }

            // Swap in the new owner. Resetting the model cache discards whatever
            // live state was resident, so any pending live-cache continuation that
            // referenced it is now stale.
            _model.ResetKVCache();
            _liveCacheValid = false;
            _ownerTokensInModel = 0;
            if (seq.NumComputedTokens > 0)
            {
                // Injecting a snapshot taken by another sequence is only valid when the
                // model can snapshot, can restore across sequences, AND the restored
                // prefix fits within what it can faithfully reconstruct. Gemma 4's
                // circular SWA cache only restores the last window's worth of positions,
                // so a snapshot longer than MaxReusablePrefixTokens (or any reuse for a
                // model that opts out entirely) is discarded and re-prefilled cleanly.
                if (!_model.SupportsKVStateSnapshot
                    || !_model.SupportsCrossSequenceKvReuse
                    || seq.NumComputedTokens > _model.MaxReusablePrefixTokens)
                {
                    // The model can't accept injected state. We have to discard
                    // the seq's "computed" claim and rerun. Mark it for re-prefill.
                    seq.ResetForPreemption();
                    var freed = seq.BlockTable.Clear();
                    if (freed.Count > 0) _pool.Free(freed);
                }
                else
                {
                    InjectAllBlocks(seq, seq.NumComputedTokens);
                    _ownerTokensInModel = seq.NumComputedTokens;
                }
            }
            _currentOwner = seq;
            _ownerForwardedTokens = 0;
        }

        private int[] BuildPrefillChunk(SequenceState seq, ScheduledSequenceWork work)
        {
            int want = work.NumScheduledTokens;
            int[] buf = new int[want];
            for (int i = 0; i < want; i++)
                buf[i] = seq.TokenAt(seq.NumComputedTokens + i);
            return buf;
        }

        private static int SampleFromLogits(SequenceState seq)
        {
            if (seq.LastLogits == null)
                throw new InvalidOperationException(
                    $"Sequence {seq.RequestId} has no LastLogits to sample from at position {seq.NumComputedTokens}.");
            var sampler = new TokenSampler(seq.SamplingConfig);
            return sampler.Sample(seq.LastLogits, seq.OutputTokens);
        }

        /// <summary>Extract all blocks for the current owner into PagedKvStorage.
        /// Called when swapping out.</summary>
        private void ExtractAllBlocks(SequenceState seq, int tokensInModel)
        {
            if (!_model.SupportsKVStateSnapshot) return;
            if (tokensInModel <= 0) return;

            int blocks = seq.BlockTable.NumBlocks;
            for (int b = 0; b < blocks; b++)
            {
                int startToken = b * _blockSize;
                if (startToken >= tokensInModel) break;
                int tokensInBlock = Math.Min(_blockSize, tokensInModel - startToken);
                var block = seq.BlockTable.Blocks[b];

                long expectedBytes = _model.ComputeKVBlockByteSize(tokensInBlock);
                if (expectedBytes <= 0) break;

                EnsureScratch((int)expectedBytes);
                var dst = _scratch.AsSpan(0, (int)expectedBytes);
                if (!_model.TryExtractKVBlock(startToken, tokensInBlock, dst))
                {
                    // For SWA-bounded models (e.g. Gemma 4) blocks whose positions
                    // have aged out of the sliding window can't be re-extracted —
                    // their K/V is gone from the model's circular cache. Those
                    // blocks were already captured into pool storage at the moment
                    // they first became full (via CaptureNewlyFullBlocks), so the
                    // pool slab still holds the correct bytes and skipping the
                    // re-extract here is harmless. We continue so the trailing
                    // in-window partial block still gets captured.
                    continue;
                }

                // Copy the bytes into the block's storage slab. For full blocks
                // we use the full-block byte size (so partial-block layout is
                // not confused with full-block layout). For the trailing
                // partial block we use the partial-byte size; the storage slab
                // is sized for one full block so partial fits.
                var slab = _pool.Storage.GetSpan(block.Id);
                dst.CopyTo(slab);
                block.Used = tokensInBlock;
            }
        }

        /// <summary>Inject all blocks for <paramref name="seq"/> into the model's
        /// fresh KV state. Called when swapping in.</summary>
        private void InjectAllBlocks(SequenceState seq, int tokensToInject)
        {
            if (!_model.SupportsKVStateSnapshot) return;
            if (tokensToInject <= 0) return;

            int blocks = seq.BlockTable.NumBlocks;
            for (int b = 0; b < blocks; b++)
            {
                int startToken = b * _blockSize;
                if (startToken >= tokensToInject) break;
                int tokensInBlock = Math.Min(_blockSize, tokensToInject - startToken);
                var block = seq.BlockTable.Blocks[b];

                long expectedBytes = _model.ComputeKVBlockByteSize(tokensInBlock);
                if (expectedBytes <= 0) break;

                var src = _pool.Storage.GetReadOnlySpan(block.Id);
                if (src.Length < expectedBytes)
                {
                    _logger.LogWarning(
                        "Inject would underflow for sequence {RequestId} block {Block}: have {Have} need {Need}",
                        seq.RequestId, b, src.Length, expectedBytes);
                    return;
                }
                var slice = src[..(int)expectedBytes];
                if (!_model.TryInjectKVBlock(startToken, tokensInBlock, slice))
                {
                    _logger.LogWarning(
                        "Inject failed for sequence {RequestId} block {Block} at {Start}",
                        seq.RequestId, b, startToken);
                    return;
                }
            }
        }

        /// <summary>For each newly-full block, extract its content into the
        /// pool's storage and ask the scheduler to register the content hash
        /// for prefix sharing.</summary>
        private int CaptureNewlyFullBlocks(SequenceState seq)
        {
            if (!_model.SupportsKVStateSnapshot) return 0;

            int fullBlocksNow = seq.NumComputedTokens / _blockSize;
            int captured = 0;
            int previouslyFull = fullBlocksNow;
            for (int b = 0; b < fullBlocksNow && b < seq.BlockTable.NumBlocks; b++)
            {
                var block = seq.BlockTable.Blocks[b];
                if (block.Used == _blockSize) continue; // already captured

                int startToken = b * _blockSize;
                long bytes = _model.ComputeKVBlockByteSize(_blockSize);
                EnsureScratch((int)bytes);
                var dst = _scratch.AsSpan(0, (int)bytes);
                if (!_model.TryExtractKVBlock(startToken, _blockSize, dst))
                    break;
                dst.CopyTo(_pool.Storage.GetSpan(block.Id));
                block.Used = _blockSize;
                captured++;
                if (b < previouslyFull) previouslyFull = b;
            }

            // Let the scheduler index the newly-full blocks (hash registration).
            if (captured > 0)
                _scheduler.OnBlocksCommitted(seq, previouslyFull * _blockSize);

            return captured;
        }

        private void EnsureScratch(int bytes)
        {
            if (_scratch == null || _scratch.Length < bytes)
                _scratch = new byte[bytes];
        }

        /// <summary>Reset internal state. Called by the engine on model reload.</summary>
        public void Reset()
        {
            _currentOwner = null;
            _ownerTokensInModel = 0;
            _ownerForwardedTokens = 0;
            _liveCacheSeq = null;
            _liveCacheLen = 0;
            _liveCacheValid = false;
            _mtpCtx = null;
            if (_model is IBatchedPagedModel fused)
            {
                foreach (var entry in _retainedFused)
                    fused.DiscardRetainedCache(entry.RequestId);
            }
            _retainedFused.Clear();
            _fusedSeqById.Clear();
            _model.ResetKVCache();
        }
    }

    /// <summary>Result of executing one ScheduledSequenceWork. Reported back
    /// to the engine for streaming + stop detection.</summary>
    public sealed class SequenceStepResult
    {
        public SequenceState Sequence { get; init; }
        public int TokensForwarded { get; init; }
        public int SampledToken { get; init; } = -1;

        /// <summary>Speculatively drafted tokens accepted by verification this
        /// step, in order, FOLLOWING <see cref="SampledToken"/>. Already
        /// appended to the sequence's OutputTokens by the executor; the engine
        /// streams them with per-token EOS / length checks. Null when the step
        /// produced no extra tokens.</summary>
        public IReadOnlyList<int> ExtraTokens { get; init; }

        public bool IsPrefill { get; init; }
        public int FullBlocksCaptured { get; init; }
        public long ForwardElapsedTicks { get; init; }
        public Exception Error { get; init; }

        public bool IsNoOp => TokensForwarded == 0 && Error == null;

        public static SequenceStepResult NoOp(SequenceState s) => new()
        {
            Sequence = s,
            TokensForwarded = 0,
        };
    }

    /// <summary>Future hook for true batched paged attention - opt-in marker
    /// for models that have a native paged forward kernel taking a batch
    /// metadata struct. Today nothing implements this; <see cref="BatchExecutor"/>
    /// falls back to per-sequence swap.</summary>
    public interface IBatchedPagedModel
    {
        /// <summary>Drive a single batched forward pass given the scheduler's
        /// per-step metadata. Returns per-sequence logits.</summary>
        IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx);

        /// <summary>True iff <c>ForwardBatch</c> handles multimodal
        /// (vision/audio embeddings + MRoPE positions) for batched
        /// sequences. When false (default), <see cref="BatchExecutor"/>
        /// peels multimodal sequences off into the per-sequence path so
        /// the model's batched kernels never see them. Set true once the
        /// per-batch position-table + embedding-inject plumbing is in place.</summary>
        bool SupportsBatchedMultimodal => false;

        /// <summary>True iff the model implements
        /// <see cref="TryMigrateLinearKVToPaged"/> for transitioning a
        /// sequence that has run through the N=1 fast path (which writes
        /// only to the legacy linear KV cache) over to the paged storage
        /// that <see cref="ForwardBatch"/> reads from. When false, the
        /// executor must not use the N=1 fast path for this model — a
        /// later second-sequence arrival would corrupt the first
        /// sequence's attention.</summary>
        bool SupportsLinearKVMigration => false;

        /// <summary>Copy the given sequence's K/V history out of the legacy
        /// linear KV cache (whatever per-model layout <c>Forward</c> writes)
        /// and into paged storage at slots derived from
        /// <c>owner.BlockTable</c> with the given block size. The model
        /// must be the one holding the linear state right now (i.e. this
        /// is called when the executor's <c>_currentOwner == owner</c>).
        ///
        /// Returns true on success. Returning false (or returning false
        /// from <see cref="SupportsLinearKVMigration"/>) tells the
        /// executor to keep the sequence on the per-seq path instead of
        /// dispatching it through <see cref="ForwardBatch"/>.</summary>
        bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize) => false;

        /// <summary>Notify the model that a sequence has been released by
        /// the scheduler (finished, aborted, errored, or preempted) so any
        /// per-sequence state the model holds keyed by <c>RequestId</c> can
        /// be reclaimed. Default no-op for models that don't keep such
        /// state. Hybrid models (Nemotron-H, Qwen 3.5) that allocate Mamba2 /
        /// GatedDeltaNet recurrent-state slots per active sequence MUST
        /// implement this; otherwise two concurrent sequences whose first
        /// attention block is shared via prefix-cache hit would collide on
        /// the same recurrent-state slot and trample each other's hidden
        /// state. Models implementing <see cref="SupportsPerSequenceFusedForward"/>
        /// also free the released request's per-request KV cache here.</summary>
        void OnSequenceReleased(string requestId) { }

        // ---- Per-sequence fused forward (high-throughput concurrent decode) ----
        //
        // When true, the executor serves concurrent (N>=2) sequences by running
        // each one through the model's fused single-graph <c>Forward</c> with its
        // own per-request KV cache, instead of the op-by-op batched paged path.
        // The op-by-op path issues ~20 Metal-queue-draining dispatches per layer,
        // which starves the GPU (~30% utilisation) and makes aggregate throughput
        // at N=2 fall below the single-stream rate; the fused per-sequence path
        // keeps the GPU saturated (one fused decode graph per token per sequence).
        //
        // A model opting in must:
        //   * give each RequestId its own KV cache (BindSequenceCache),
        //   * be able to hand the current single-stream owner's cache to a
        //     per-request holder cheaply (AdoptPrimaryCacheToFused), and
        //   * reinstate the single-stream cache for the N==1 path
        //     (RestorePrimaryCache).
        // It must also free per-request caches in OnSequenceReleased.

        /// <summary>True iff this model supports the per-sequence fused-decode
        /// path described above. Default false (model keeps the batched path).</summary>
        bool SupportsPerSequenceFusedForward => false;

        /// <summary>Make <paramref name="requestId"/>'s per-request KV cache the
        /// model's active cache (creating an empty one the first time). Returns
        /// true when the cache was freshly created, signalling the caller to
        /// inject any prefix-cache-reused prefix before the first forward.</summary>
        bool BindSequenceCache(string requestId) => false;

        /// <summary>Transition the current single-stream (N==1) owner — whose
        /// live K/V is in the model's primary cache — into a per-request holder
        /// without copying KV bytes, and give the primary cache a fresh empty
        /// allocation. Called once when the first concurrent step finds a prior
        /// owner so its history is preserved as an isolated per-request cache.</summary>
        void AdoptPrimaryCacheToFused(string requestId) { }

        /// <summary>Reinstate the primary (single-stream) cache as the model's
        /// active cache before an N==1 step that follows a multi-sequence
        /// episode. No-op when the primary cache is already active.</summary>
        void RestorePrimaryCache() { }

        /// <summary>True iff a per-request fused cache holder already exists for
        /// <paramref name="requestId"/> (i.e. the sequence has run on the fused
        /// path before and must stay on it — its tail K/V isn't reconstructable
        /// from paged storage).</summary>
        bool HasFusedSequenceCache(string requestId) => false;

        // ---- Retained fused-cache continuation (cross-request prefix reuse) ----
        //
        // The per-sequence fused path keeps each concurrent request's full K/V in
        // its own holder and never writes the shared paged block storage, so it
        // contributes nothing to the prefix-cache pool. For a sliding-window model
        // the pool can't restore a long prefix anyway (only the live circular cache
        // can), so a multi-turn follow-up ("请继续") that arrives while/after other
        // requests ran concurrently would re-prefill the whole conversation from
        // scratch (KV-reuse ratio 0). To fix that, the executor RETAINS a finished
        // fused request's holder and re-adopts it for a later request whose prompt
        // exactly extends the retained tokens — the cross-request analogue of the
        // single-stream live-cache continuation. The model side just keeps the
        // holder alive and lets it be re-keyed.

        /// <summary>Move <paramref name="requestId"/>'s per-request fused holder out
        /// of the active set into a retained set so a later request can re-adopt it
        /// (see <see cref="TryRebindRetainedCache"/>) instead of disposing it. The
        /// executor calls this when a fused sequence finishes cleanly, before
        /// <see cref="OnSequenceReleased"/> (which then no-ops for the holder).
        /// Returns true when a holder was retained. Default false (no retention).</summary>
        bool RetainSequenceCache(string requestId) => false;

        /// <summary>Re-key a retained holder from <paramref name="retainedRequestId"/>
        /// to <paramref name="newRequestId"/>, making it that request's active fused
        /// cache (it becomes the cache the next <see cref="BindSequenceCache"/> finds,
        /// so the new request continues from the retained K/V with no re-prefill).
        /// Returns false when no retained holder exists for the id. Default false.</summary>
        bool TryRebindRetainedCache(string retainedRequestId, string newRequestId) => false;

        /// <summary>Dispose a retained holder (LRU eviction / shutdown) and free its
        /// buffers. Default no-op.</summary>
        void DiscardRetainedCache(string requestId) { }

        /// <summary>TRUE token-batched decode: decode ONE token for each of N
        /// concurrent sequences in a single fused graph (one compute buffer,
        /// weights loaded once) instead of N serial per-sequence forwards. This
        /// raises the round-robin ~1x concurrency ceiling toward llama.cpp's ~Nx
        /// (decode is memory-bandwidth bound, so batching amortises the weight
        /// loads). <paramref name="requestIds"/>/<paramref name="tokens"/>/
        /// <paramref name="positions"/> are parallel arrays of length N: sequence i
        /// decodes <paramref name="tokens"/>[i] at <paramref name="positions"/>[i]
        /// against its own per-request KV holder. On success writes each sequence's
        /// logits into <paramref name="outLogits"/>[i] and returns true; returns
        /// false when the model can't batch this step (caller falls back to the
        /// per-sequence round-robin loop). Default false (opt-in).</summary>
        bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits) => false;
    }

    /// <summary>Per-step metadata for the batched paged attention path.
    /// Mirrors vLLM's <c>CommonAttentionMetadata</c>.</summary>
    public sealed class BatchedForwardContext
    {
        public List<SequenceState> Sequences { get; init; }
        public List<int> NumScheduledTokens { get; init; }
        public List<int> QueryStartLoc { get; init; }
        public List<int> Positions { get; init; }
        public List<int> SlotMapping { get; init; }
        public int[][] BlockTables { get; init; }
        public int MaxQueryLen { get; set; }
        public int MaxSeqLen { get; set; }

        // ---- NextN/MTP speculative-trunk extensions ----

        /// <summary>When set, these tokens are forwarded instead of reading
        /// <c>seq.TokenAt(...)</c>. Speculative verify batches forward drafted
        /// tokens that are not (yet) part of the sequence's token list.</summary>
        public int[] OverrideFlatTokens { get; init; }

        /// <summary>When non-null, receives the post-final-norm hidden state of
        /// every row (numTokens × hidden floats) — llama.cpp's h_nextn, consumed
        /// by the MTP draft head.</summary>
        public float[] CaptureHiddenAll { get; init; }

        /// <summary>When non-null, receives LM-head logits for every row
        /// (numTokens × vocab floats) — speculative verification needs per-row
        /// logits, not just the last position.</summary>
        public float[] CaptureLogitsAll { get; init; }
    }
}
