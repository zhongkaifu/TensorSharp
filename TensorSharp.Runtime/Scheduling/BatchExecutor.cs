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

        // Re-used scratch buffer for inject/extract. Sized to one full block.
        private byte[] _scratch;

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
                    //   2. Restrict the fast path to models that implement
                    //      TryMigrateLinearKVToPaged, so the upcoming N=2
                    //      transition can copy the linear state across before
                    //      the batched kernel reads it.
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
                        && batched2.SupportsLinearKVMigration)
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
            // The batched paged path manages K/V in paged storage (slot mapping),
            // not the single live linear cache the continuation fast-path tracks.
            // Drop any live-cache claim so a follow-up can't continue from stale state.
            _liveCacheValid = false;
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

        private List<SequenceStepResult> ExecuteStepPerSequence(SchedulerOutput output)
        {
            var results = new List<SequenceStepResult>(1);
            if (output.ScheduledWork.Count == 0)
                return results;

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
                    seq.Error = ex;
                }
            }

            return results;
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
        /// state.</summary>
        void OnSequenceReleased(string requestId) { }
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
    }
}
