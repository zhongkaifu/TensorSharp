// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;

using TensorSharp.Runtime.Speculative;

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
        // sequence skip re-prefill entirely by continuing from the live cache ÔÇö
        // critical for sliding-window models where the pooled snapshot can only
        // reuse one window.
        private SequenceState _liveCacheSeq;
        private int _liveCacheLen;
        private bool _liveCacheValid;

        // Why the LAST ComputeLiveContinuationLcp / ComputeFusedContinuationLcp call
        // returned 0, or null when it returned a usable prefix. Neither method can
        // tell whether the request ends up reusing anything - the scheduler tries the
        // live cache, then retained model state, then the pooled blocks - so the
        // reasons are recorded here and the SCHEDULER reports the one outcome that is
        // true. Before this, ComputeLiveContinuationLcp announced "This turn
        // re-prefills its full prompt (KV reuse 0)" at Information level on a path
        // that a retained holder then served at 99% reuse, which is exactly the wrong
        // thing to tell an operator hunting latency.
        private string? _lastLiveDeclineReason;
        private string? _lastFusedDeclineReason;

        // ---- Retained fused-cache continuation (cross-request prefix reuse) ----
        // The per-sequence fused path (concurrent N>=2 decode) keeps each request's
        // complete continuation state in its own holder and never writes the shared
        // paged blocks, so a finished concurrent request leaves nothing in the
        // prefix-cache pool. That matters both for circular K/V (Gemma 4), whose
        // byte snapshots are window-capped, and for hybrid attention+recurrent models
        // (Qwen 3.5/3.6), whose attention K/V is incomplete without matching GDN
        // state. We retain a small LRU of finished fused holders keyed by their full
        // token list, and re-adopt one for a later request whose prompt extends it
        // (allowing a short generated control-token tail that history rendering omits)
        // ÔÇö the cross-request analogue of the single-stream live-cache continuation.
        // See ComputeFusedContinuationLcp.
        private sealed class RetainedFusedCache
        {
            public string RequestId;   // model holder key (retained, not active)
            public int[] Tokens;       // full prompt+output tokens the holder's K/V covers
            public string MediaFingerprint; // prevents placeholder-identical media cross-reuse
            // A shared-prefix checkpoint: cloned on adoption rather than re-keyed, and
            // never removed by that adoption. Its tokens are text-only by construction
            // (the chat layer marks the prefix before any media), so no fingerprint
            // gate applies to it.
            public bool IsPrefixCheckpoint;
        }
        // Most-recently-retained at the tail; evict from the head.
        private readonly LinkedList<RetainedFusedCache> _retainedFused = new();
        // Allocation is separable from ownership transfer, including under memory
        // pressure. The internal allocator also permits deterministic OOM coverage.
        internal Func<int, int[]> AllocateRetainedCacheTokens { get; set; }
            = static length => new int[length];
        internal Action<SequenceState, int> ReserveRetainedAdoptionMetadata { get; set; }

        // ---- Shared-prefix checkpoints (see IBatchedPagedModel.SupportsPrefixCheckpoints) ----
        // The model's complete state at the end of the prompt prefix every conversation
        // shares, keyed "prefix:<n>" so no request id can collide with it. Unlike a
        // retained holder it is never consumed: a request that starts from it gets a
        // CLONE. Most recent at the tail; evicted from the head past the budget.
        private readonly LinkedList<RetainedFusedCache> _prefixCheckpoints = new();
        private int _prefixCheckpointSerial;

        // Checkpoint keys name holders the MODEL owns, and a model outlives the executors
        // built on it (every reload builds a new engine on the same weights in tests; a
        // host may rebuild an engine for other reasons). "prefix:1" from a previous
        // executor would collide with this one's first key and make the model decline
        // the checkpoint -- observed as a saved-nothing store. So the key carries the
        // executor's own number.
        private static int s_executorSerial;
        private readonly string _checkpointKeyPrefix = $"prefix:{Interlocked.Increment(ref s_executorSerial)}.";
        // A retained holder can include one or two generated control tokens (most
        // commonly EOS) that the chat history intentionally does not render. After
        // re-keying such a holder, truncate its active model cache to this target
        // before forwarding the new prompt suffix.
        private readonly Dictionary<string, int> _pendingRetainedFusedTruncations =
            new(StringComparer.Ordinal);
        // In-flight fused sequences by RequestId, so the release hook can snapshot
        // a finishing sequence's tokens (the release notification only carries an id).
        private readonly Dictionary<string, SequenceState> _fusedSeqById =
            new(StringComparer.Ordinal);

        // Re-used scratch buffer for inject/extract. Sized to one full block.
        private byte[] _scratch;

        // ---- NextN/MTP speculative decoding (see SpeculativeExecution) ----
        // At most one sequence at a time runs speculatively: the draft head's
        // KV cache and pending hidden state live in the model's single live
        // (linear) cache, so continuity is (sequence identity, exact trunk
        // position). Any KV rebuild/swap ÔÇö ownership change, batched/fused
        // step, preemption ÔÇö invalidates the context; it re-arms only at a
        // fresh full prefill from position 0.
        private SpecSeqContext _specCtx;

        // One-time warning when --spec is requested but the model can't run its
        // accelerated MTP path on the current backend (speculation would be net-
        // negative), so the engine serves standard decode instead.
        private bool _speculationUnprofitableWarned;

        // One-time warning when speculation is requested but no registered
        // algorithm can serve the loaded model (no draft head, unknown
        // --spec-type).
        private bool _speculationDeclineWarned;

        // The user's rule for every one of these: a fallback must never be
        // silent. Each fires ONCE per executor (i.e. per loaded model) so the
        // console names the degradation without turning into per-token spam.
        private bool _forwardBatchDeclineWarned;
        private bool _fusedBatchedDeclineWarned;
        private bool _specPrefixReuseDeclineWarned;
        private bool _crossSeqSerializationWarned;
        private readonly HashSet<string> _fallbackTransitionsWarned = new(StringComparer.Ordinal);

        private sealed class SpecSeqContext
        {
            public SequenceState Seq;
            public SpeculativeExecution Exec;
            // Non-null when the speculative trunk runs through the batched
            // paged path (IBatchedSpeculativeModel) instead of the linear
            // cache. The trunk's own position must agree with NextPosition.
            public BatchedSpecTrunk BatchedTrunk;
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
        /// through <see cref="IBatchedSpeculativeModel.SpecForwardBatched"/>
        /// (paged KV via the sequence's block table, per-slot recurrent
        /// state), so the spec trunk runs on the same kernels as the
        /// non-speculative batched baseline.</summary>
        internal sealed class BatchedSpecTrunk : ISpecTrunk
        {
            private readonly IBatchedSpeculativeTarget _model;
            private readonly SequenceState _seq;

            /// <summary>Tokens committed to the trunk so far (advances with
            /// each Forward; rolled back on rejection).</summary>
            public int Position { get; private set; }

            public BatchedSpecTrunk(IBatchedSpeculativeTarget model, SequenceState seq, int position)
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

            public void SnapshotRecurrentState() => _model.SpecSnapshotRecurrentStateSlots(_seq);

            public void Rollback(int position)
            {
                // Paged attention KV needs no rewind: every pass passes its
                // sequence length explicitly, and rejected slots are simply
                // overwritten by the kept-prefix re-forward / later steps.
                _model.SpecRestoreRecurrentStateSlots(_seq);
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
            ReserveRetainedAdoptionMetadata = (seq, count) =>
            {
                seq.BlockTable.EnsureBlockCapacity(count);
                _pendingRetainedFusedTruncations.EnsureCapacity(_pendingRetainedFusedTruncations.Count + 1);
            };
        }

        public IModelArchitecture Model => _model;
        public SequenceState CurrentOwner => _currentOwner;

        /// <summary>Execute one scheduler step. Path selection is centralised
        /// in <see cref="ExecutionPlanner"/>: the executor snapshots the
        /// model's <see cref="ExecutionCapabilities"/>, the operator's
        /// <see cref="ExecutionOptions"/> (TS_* overrides) and this step's
        /// <see cref="ExecutionStepFeatures"/>, and the planner returns an
        /// <see cref="ExecutionPlan"/> — an ordered candidate chain plus the
        /// reasons rejected paths were rejected. The executor then runs the
        /// first candidate that accepts the step; declinable candidates
        /// (MTP arming/continuity, linear→paged migration, a model refusing a
        /// specific batch) fall through to the next entry in the chain, whose
        /// last entry (<see cref="ExecutionPathKind.PerSequence"/>) never
        /// declines.
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
                // step's TryPrepareSequencesForPagedIfNeeded would call into
                // TryMigrateLinearKVToPaged with NumBlocks==0 (it returns
                // false and the executor logs a misleading "linearÔåÆpaged
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

                // Centralised path selection: snapshot what the loaded
                // model+backend can do (ExecutionCapabilities), what the
                // operator overrode (ExecutionOptions, the TS_* switches) and
                // what this step's requests need (ExecutionStepFeatures), then
                // ask the planner for the candidate chain. All the "which
                // path?" logic lives in ExecutionPlanner.PlanStep; this method
                // only executes the plan. Options/capabilities are snapshotted
                // per step because several are env-var backed and tests toggle
                // them at runtime.
                var options = ExecutionOptions.FromEnvironment();
                var caps = ExecutionCapabilities.FromModel(_model);
                var features = ComputeStepFeatures(output, caps);
                var plan = ExecutionPlanner.PlanStep(caps, options, PlanConfig, features);

                if (plan.SpeculationUnprofitable)
                    WarnSpeculationUnprofitableOnce();
                if (plan.SpeculationRefusal != null)
                    WarnSpeculationRefusedOnce(plan.SpeculationRefusal);
                LogPlanTransition(plan);

                for (int i = 0; i < plan.Candidates.Count; i++)
                {
                    if (i > 0)
                    {
                        // The Information plan line above reports the SELECTED
                        // candidate; when that candidate declines at runtime the
                        // user must hear that what actually ran is the fallback,
                        // or the log claims the opposite of what is happening.
                        // Once per transition; Debug afterwards.
                        string transition = plan.Candidates[i - 1] + "->" + plan.Candidates[i];
                        if (_fallbackTransitionsWarned.Add(transition))
                        {
                            _logger.LogWarning(
                                "BatchExecutor: the {Declined} path declined at runtime; executing on " +
                                "{Next} instead. Reported once per transition; later occurrences log at Debug.",
                                plan.Candidates[i - 1], plan.Candidates[i]);
                            _lastPlanDescription = null;
                        }
                        else
                        {
                            _logger.LogDebug(
                                "BatchExecutor: {Declined} declined this step; falling back to {Next}.",
                                plan.Candidates[i - 1], plan.Candidates[i]);
                        }
                    }
                    var results = TryExecutePath(plan.Candidates[i], output, options);
                    if (results != null)
                        return results;
                }

                // Unreachable: the planner always terminates the chain with a
                // path that cannot decline (PerSequence). Kept as a hard
                // fallback so a planner bug degrades to correctness, not loss.
                return ExecuteStepPerSequence(output);
            }
        }

        /// <summary>Request-side features of this step (the planner input that
        /// changes per step, as opposed to the model capabilities and operator
        /// overrides).</summary>
        private ExecutionStepFeatures ComputeStepFeatures(SchedulerOutput output, ExecutionCapabilities caps)
        {
            int count = output.ScheduledWork.Count;
            var injector = _model.MultimodalInjector;
            int multimodalPending = 0;
            if (injector != null)
            {
                foreach (var work in output.ScheduledWork)
                {
                    if (injector.HasPendingEmbeddings(work.Sequence.RequestId))
                        multimodalPending++;
                }
            }

            bool soloMm = false, soloPaged = false, soloFused = false, soloSwap = false;
            if (count == 1)
            {
                var solo = output.ScheduledWork[0].Sequence;
                soloMm = injector != null && injector.HasPendingEmbeddings(solo.RequestId);
                soloPaged = solo.KvStateInPagedStorage;
                soloFused = caps.SupportsPerSequenceFusedForward
                    && _model is IBatchedPagedModel fused
                    && fused.HasFusedSequenceCache(solo.RequestId);
                soloSwap = _currentOwner != null && !ReferenceEquals(_currentOwner, solo);
            }

            return new ExecutionStepFeatures
            {
                SequenceCount = count,
                MultimodalPendingCount = multimodalPending,
                SoloHasPendingMultimodal = soloMm,
                SoloKvInPagedStorage = soloPaged,
                SoloHasFusedCache = soloFused,
                SoloRequiresOwnershipSwap = soloSwap,
            };
        }

        /// <summary>Dispatch one plan candidate. Returns null when a declinable
        /// candidate passes on the step (the caller then tries the next
        /// candidate in the plan's chain).</summary>
        private List<SequenceStepResult> TryExecutePath(
            ExecutionPathKind path, SchedulerOutput output, ExecutionOptions options)
        {
            switch (path)
            {
                case ExecutionPathKind.SpeculativeBatchedTrunk:
                    // Declinable: the arming/continuity gate lives in the handler.
                    return TryExecuteStepSpecBatchedTrunk(output);

                case ExecutionPathKind.SpeculativePerSequence:
                case ExecutionPathKind.SingleSequenceFused:
                    // Both run the per-sequence executor; the plan kinds
                    // differ only in WHY the route was chosen (linear-trunk
                    // speculation, N=1 fused fast path, universal fallback),
                    // which the plan log already records.
                    return ExecuteStepPerSequence(output);

                case ExecutionPathKind.PerSequence:
                    return ExecuteStepPerSequenceStateSafe(output);

                case ExecutionPathKind.PerSequenceFused:
                    return ExecuteStepPerSequenceFused((IBatchedPagedModel)_model, output, options);

                case ExecutionPathKind.MixedMultimodalSplit:
                    return ExecuteStepMixedMultimodalSplit((IBatchedPagedModel)_model, output);

                case ExecutionPathKind.BatchedPaged:
                    return TryExecuteStepBatchedPaged((IBatchedPagedModel)_model, output);

                default:
                    throw new InvalidOperationException($"Unknown execution path: {path}");
            }
        }

        /// <summary>Mixed step (model without batched multimodal support):
        /// multimodal sequences run per-sequence so the model can inject
        /// vision/audio embeddings at the right per-chunk positions; text
        /// sequences run through the batched paged path for full
        /// continuous-batching throughput. Routing the two subsets separately
        /// keeps healthy text sequences off the per-seq swap path (which
        /// produces garbled output under concurrent multi-seq iteration on
        /// e.g. Gemma 4).</summary>
        private List<SequenceStepResult> ExecuteStepMixedMultimodalSplit(
            IBatchedPagedModel batched, SchedulerOutput output)
        {
            var (multimodalWork, textWork) = SplitMultimodalWork(output);
            var textOutput = MakeSubOutput(textWork);

            // Run the declinable text batch before committing any multimodal
            // per-sequence work. If the current linear owner participates in
            // that text batch, copy it to paged storage but keep the linear
            // owner authoritative until ForwardBatch accepts. A decline can
            // then fall back on the complete linear tail; a failed migration
            // likewise falls back before either subset advances.
            if (!TryPrepareSequencesForPagedIfNeeded(
                    batched,
                    textOutput,
                    out var linearFallbackSequences))
            {
                var safeTextResults = ExecuteLinearAndPagedSplit(
                    batched, textOutput, linearFallbackSequences);
                if (safeTextResults == null)
                    return ExecuteStepPerSequence(output);

                safeTextResults.AddRange(ExecuteStepPerSequence(MakeSubOutput(multimodalWork)));
                return safeTextResults;
            }

            var results = new List<SequenceStepResult>(output.ScheduledWork.Count);
            try
            {
                results.AddRange(ExecuteStepBatched(batched, textOutput));
                foreach (var work in textWork)
                    work.Sequence.KvStateInPagedStorage = true;
                CommitPreparedSequencesToPaged(linearFallbackSequences);
            }
            catch (NotSupportedException ex)
            {
                _logger.LogDebug(ex,
                    "BatchExecutor: model declined the text subset of a mixed batch; selecting a state-safe fallback.");
                var safeTextResults = ExecuteLinearAndPagedSplit(
                    batched, textOutput, linearFallbackSequences);
                if (safeTextResults == null)
                    return ExecuteStepPerSequence(output);

                results.AddRange(safeTextResults);
            }

            results.AddRange(ExecuteStepPerSequence(MakeSubOutput(multimodalWork)));
            return results;
        }

        /// <summary>Batched paged dispatch (vLLM-style ForwardBatch). Returns
        /// null (declining the step to the plan's next candidate) when the
        /// owner's linear-to-paged migration fails or the model refuses this
        /// specific batch with NotSupportedException.</summary>
        private List<SequenceStepResult> TryExecuteStepBatchedPaged(
            IBatchedPagedModel batched, SchedulerOutput output)
        {
            // Before dispatching through ForwardBatch, make sure any sequence
            // whose K/V history only lives in the linear cache (because it was
            // being served by the N=1 fast path, or by a previous step that
            // fell back to per-seq) is migrated into paged storage. Without
            // this the batched paged-attention kernel would read zeros for the
            // owner's prior positions and the sequence would emit a
            // token-repeat loop.
            if (!TryPrepareSequencesForPagedIfNeeded(
                    batched,
                    output,
                    out var linearFallbackSequences))
            {
                // Migration was needed but couldn't proceed: either the model
                // supports migration and it failed for this owner (worth
                // flagging so we can diagnose), or the model never exposed
                // migration and the owner accumulated linear-only state
                // (expected, not a failure; serve via per-seq where the linear
                // cache is still authoritative).
                if (batched.SupportsLinearKVMigration)
                {
                    _logger.LogWarning(
                        "BatchExecutor: linear-to-paged migration failed for {RequestId}; falling back to per-seq path.",
                        _currentOwner?.RequestId);
                }
                return ExecuteLinearAndPagedSplit(
                    batched, output, linearFallbackSequences);
            }

            try
            {
                var results = ExecuteStepBatched(batched, output);
                // Any sequence that just ran through the batched path now has
                // its K/V in paged storage; sticky-mark it so future steps
                // don't try to send it back through the linear-cache-only N=1
                // fast path.
                foreach (var work in output.ScheduledWork)
                    work.Sequence.KvStateInPagedStorage = true;
                CommitPreparedSequencesToPaged(linearFallbackSequences);
                return results;
            }
            catch (NotSupportedException ex)
            {
                // The model declared support but bailed for this specific
                // batch. Decline to the plan's next candidate (per-seq swap).
                if (!_forwardBatchDeclineWarned)
                {
                    _forwardBatchDeclineWarned = true;
                    _logger.LogWarning(
                        "The model declined the batched forward for this step ({Reason}); serving " +
                        "sequences one at a time on the per-sequence path. Sustained declines mean " +
                        "concurrent throughput is degraded toward single-stream. Reported once.",
                        ex.Message);
                }
                _logger.LogDebug(ex, "BatchExecutor: ForwardBatch declined the batch; falling back.");
                return ExecuteLinearAndPagedSplit(
                    batched, output, linearFallbackSequences);
            }
        }

        // Last logged plan description; plans are re-logged only when the
        // decision (selected path, fallback chain, or rejection reasons)
        // actually changes, so steady-state decode stays quiet while
        // concurrency/feature transitions leave an audit trail.
        private string _lastPlanDescription;

        private void LogPlanTransition(ExecutionPlan plan)
        {
            if (!_logger.IsEnabled(LogLevel.Information)) return;
            string desc = plan.Describe();
            if (string.Equals(desc, _lastPlanDescription, StringComparison.Ordinal)) return;
            _lastPlanDescription = desc;
            _logger.LogInformation("BatchExecutor execution plan: {Plan}", desc);
        }

        /// <summary>Prepare every scheduled sequence whose authoritative K/V is
        /// linear-only for an upcoming paged batch. This includes the current
        /// live owner even when its sticky paged flag is still true: a prior
        /// per-sequence step may have appended a partial tail only to linear K/V.
        ///
        /// A non-current linear sequence is first restored from its pooled
        /// snapshot and then migrated. Migrations are copy-only; no sticky flag
        /// or ownership state is committed until ForwardBatch accepts, so a
        /// decline can still restore/swap the complete linear histories.
        /// <paramref name="linearFallbackSequences"/> records exactly those
        /// sequences whose state is safe to serve through Forward after a
        /// decline. Already-paged sequences are deliberately absent because
        /// model-owned paged arrays are not the same storage as BlockPool.</summary>
        private bool TryPrepareSequencesForPagedIfNeeded(
            IBatchedPagedModel batched,
            SchedulerOutput output,
            out HashSet<SequenceState> linearFallbackSequences)
        {
            linearFallbackSequences = new HashSet<SequenceState>();
            if (output == null || output.ScheduledWork.Count == 0)
                return true;

            SequenceState initialOwner = _currentOwner;
            var candidates = new List<SequenceState>();
            foreach (var work in output.ScheduledWork)
            {
                var seq = work.Sequence;
                if (seq.NumComputedTokens <= 0)
                    continue;
                if (seq.KvStateInPagedStorage && !ReferenceEquals(seq, initialOwner))
                    continue;
                if (!candidates.Contains(seq))
                    candidates.Add(seq);
            }

            // Restore/migrate the original owner last. That leaves its live
            // linear tail resident if the declinable batch refuses the step.
            candidates.Sort((left, right) =>
            {
                bool leftIsOwner = ReferenceEquals(left, initialOwner);
                bool rightIsOwner = ReferenceEquals(right, initialOwner);
                return leftIsOwner == rightIsOwner ? 0 : leftIsOwner ? 1 : -1;
            });

            foreach (var seq in candidates)
            {
                linearFallbackSequences.Add(seq);
                if (!batched.SupportsLinearKVMigration)
                    return false;

                if (!ReferenceEquals(_currentOwner, seq))
                {
                    // Without a portable snapshot there is no way to make a
                    // non-current linear history resident for migration.
                    if (!_model.SupportsKVStateSnapshot
                        || !_model.SupportsCrossSequenceKvReuse)
                    {
                        return false;
                    }
                    EnsureOwnership(seq);
                }

                if (!ReferenceEquals(_currentOwner, seq)
                    || _ownerTokensInModel < seq.NumComputedTokens
                    || !batched.TryMigrateLinearKVToPaged(seq, _blockSize))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>After a multi-sequence ForwardBatch decline, serve only
        /// histories known to be linear-authoritative through Forward. A sticky
        /// paged resident stays on model-owned paged storage via a singleton
        /// ForwardBatch; pooled snapshots are not assumed to mirror those
        /// model-owned arrays. Returns null when every sequence can use the
        /// planner's ordinary per-sequence fallback.</summary>
        private List<SequenceStepResult> ExecuteLinearAndPagedSplit(
            IBatchedPagedModel batched,
            SchedulerOutput output,
            HashSet<SequenceState> linearFallbackSequences)
        {
            var linearWork = new List<ScheduledSequenceWork>();
            var pagedWork = new List<ScheduledSequenceWork>();
            foreach (var work in output.ScheduledWork)
            {
                var seq = work.Sequence;
                bool canUseLinear = seq.NumComputedTokens == 0
                    || !seq.KvStateInPagedStorage
                    || linearFallbackSequences.Contains(seq);
                (canUseLinear ? linearWork : pagedWork).Add(work);
            }

            if (pagedWork.Count == 0)
                return null;

            var resultBySequence = new Dictionary<SequenceState, SequenceStepResult>();
            if (linearWork.Count > 0)
            {
                foreach (var result in ExecuteStepPerSequence(MakeSubOutput(linearWork)))
                    resultBySequence[result.Sequence] = result;
            }

            foreach (var work in pagedWork)
            {
                try
                {
                    var singleton = MakeSubOutput(new List<ScheduledSequenceWork> { work });
                    var singletonResults = ExecuteStepBatched(batched, singleton);
                    work.Sequence.KvStateInPagedStorage = true;
                    resultBySequence[work.Sequence] = singletonResults[0];
                }
                catch (NotSupportedException ex)
                {
                    var error = new NotSupportedException(
                        $"The model declined singleton paged execution for {work.Sequence.RequestId}; " +
                        "linear fallback is unsafe because its K/V history exists only in model-owned paged storage.",
                        ex);
                    _logger.LogError(error,
                        "BatchExecutor cannot safely fall back paged sequence {RequestId} to linear execution.",
                        work.Sequence.RequestId);
                    resultBySequence[work.Sequence] = new SequenceStepResult
                    {
                        Sequence = work.Sequence,
                        Error = error,
                    };
                }
            }

            var ordered = new List<SequenceStepResult>(output.ScheduledWork.Count);
            foreach (var work in output.ScheduledWork)
            {
                // The universal per-sequence executor intentionally advances
                // at most one linear sequence per scheduler step. Any other
                // linear work remains uncommitted and is re-emitted next step.
                if (resultBySequence.TryGetValue(work.Sequence, out var result))
                    ordered.Add(result);
            }
            return ordered;
        }

        /// <summary>Universal per-sequence fallback that does not move a
        /// sequence whose authoritative state is already paged into the empty
        /// linear cache. This matters when the operator toggles the batched path
        /// off between steps (options are intentionally re-read per step), or a
        /// model capability latches off after paged work was accepted. Existing
        /// paged residents drain through singleton ForwardBatch calls; new and
        /// linear-resident work continues through the ordinary fallback.</summary>
        private List<SequenceStepResult> ExecuteStepPerSequenceStateSafe(
            SchedulerOutput output)
        {
            if (_model is IBatchedPagedModel batched)
            {
                var linearFallbackSequences = new HashSet<SequenceState>();
                if (_currentOwner != null && _ownerTokensInModel > 0)
                    linearFallbackSequences.Add(_currentOwner);

                var splitResults = ExecuteLinearAndPagedSplit(
                    batched, output, linearFallbackSequences);
                if (splitResults != null)
                    return splitResults;
            }

            return ExecuteStepPerSequence(output);
        }

        /// <summary>Commit a successful linear-to-paged handoff after the model
        /// accepts the batched step. Before this point the live linear owner
        /// remains authoritative for a runtime decline.</summary>
        private void CommitPreparedSequencesToPaged(
            HashSet<SequenceState> linearFallbackSequences)
        {
            if (_currentOwner != null && linearFallbackSequences.Contains(_currentOwner))
                ClearLinearOwner();
        }

        /// <summary>Give <paramref name="seq"/> its own copy of logits it may be
        /// borrowing from the model's reusable output buffer.</summary>
        private static void DetachBorrowedLogits(SequenceState seq)
        {
            if (seq?.LastLogits != null)
                seq.LastLogits = (float[])seq.LastLogits.Clone();
        }

        private void ClearLinearOwner()
        {
            _currentOwner = null;
            _ownerTokensInModel = 0;
            _ownerForwardedTokens = 0;
            _liveCacheValid = false;
        }

        private bool _speculationRefusedWarned;

        private void WarnSpeculationRefusedOnce(string reason)
        {
            if (_speculationRefusedWarned) return;
            _speculationRefusedWarned = true;
            _logger.LogWarning(
                "Speculative decoding was requested but the loaded model refuses it: {Reason} "
                + "Every request is served with plain decoding.", reason);
        }

        private void WarnSpeculationUnprofitableOnce()
        {
            if (_speculationUnprofitableWarned) return;
            _speculationUnprofitableWarned = true;
            _logger.LogWarning(
                "MTP speculative decoding was requested (--spec) but for the loaded model " +
                "on this backend the standard decode path is already faster than speculative " +
                "decode (its multi-token verify/draft runs op-by-op and cannot amortize a cheap, " +
                "fused/captured decode). Serving the fast standard decode instead ÔÇö no action needed.");
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

            // Pre-fill tokens-to-forward array. Decode only peeks the sampled
            // token here: ForwardBatch is allowed to decline this particular
            // step, so committing the token before that call would make the
            // per-sequence fallback append it a second time.
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
                    int sampledFirst = PeekPendingOrSample(seq);
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

            // Models read the batch's input tokens from OverrideFlatTokens.
            // For prefill chunks this is exactly what seq.TokenAt(startTok + i)
            // returns (BuildPrefillChunk reads the same positions), and for
            // decode steps it carries the sampled token that has NOT been
            // committed to the sequence's token list yet (the executor appends
            // it only after ForwardBatch accepts the batch, so a decline can
            // fall back to per-sequence without double-appending). Without the
            // override, TokenAt throws ArgumentOutOfRangeException on decode.
            var flatTokens = new int[ctx.QueryStartLoc[ctx.QueryStartLoc.Count - 1]];
            int ft = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var inputTokens = pendingTokens[s];
                for (int i = 0; i < inputTokens.Length; i++)
                    flatTokens[ft++] = inputTokens[i];
            }
            ctx.OverrideFlatTokens = flatTokens;

            // Dispatch the entire batch.
            var swForward = Stopwatch.StartNew();
            IReadOnlyList<float[]> perSeqLogits;
            try
            {
                perSeqLogits = batched.ForwardBatch(ctx);
            }
            catch (NotSupportedException)
            {
                // Preserve each already-sampled decode token for the fallback.
                // This is required even when it came from host sampling: sampling
                // can advance RNG state, and re-sampling would change the request.
                for (int s = 0; s < numSeqs; s++)
                {
                    if (output.ScheduledWork[s].IsPrefill) continue;
                    var seq = output.ScheduledWork[s].Sequence;
                    seq.PendingDeviceToken = pendingTokens[s][0];
                    seq.PendingDevicePosition = seq.NumComputedTokens;
                }
                throw;
            }
            swForward.Stop();
            if (perSeqLogits == null || perSeqLogits.Count != numSeqs)
                throw new InvalidOperationException(
                    $"ForwardBatch returned {perSeqLogits?.Count ?? -1} results for {numSeqs} sequences.");

            // Only invalidate linear state after ForwardBatch ACCEPTS the
            // batch. A model may decline with NotSupportedException and fall
            // back to the per-sequence path; invalidating before that call
            // destroys a planned live-cache continuation on the fallback.
            _liveCacheValid = false;
            // A batched step advanced every trunk past the linear speculative context;
            // drop it (and its drafter's buffers) rather than leaving it to the GC.
            _specCtx?.Exec?.Dispose();
            _specCtx = null;

            // Update per-sequence state and assemble results.
            var results = new List<SequenceStepResult>(numSeqs);
            for (int s = 0; s < numSeqs; s++)
            {
                var work = output.ScheduledWork[s];
                var seq = work.Sequence;
                var inputTokens = pendingTokens[s];

                if (!work.IsPrefill)
                {
                    seq.PendingDeviceToken = null;
                    seq.AppendOutputToken(inputTokens[0]);
                }
                seq.LastLogits = perSeqLogits[s];
                seq.AdvanceComputedTokens(inputTokens.Length);
                // These positions now live in the model's paged arrays, not in pool
                // storage: a later sequence adopting these blocks must read them there.
                seq.BlockTable.SetHoldsModelPagedKv(
                    seq.NumComputedTokens - inputTokens.Length, seq.NumComputedTokens, true);
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

        // Continuous-batching routing trace, off unless TS_CB_DEBUG=1. Prints
        // which path each step took, the scheduled work, and the current owner -
        // the context a per-sequence cache bug is impossible to read without.
        // The request a late arm (see TryExecuteStepSpecPerSequence) was declined for,
        // so the decision - and the speculator it would allocate - is made once per
        // request rather than at every decode step.
        private string _lateArmDeclinedFor;

        // The speculation policy in force. Starts as the scheduler config's and is
        // replaced by SetSpeculation when the host toggles speculation at run time (a
        // settings switch), so a change applies to the next turn instead of the next
        // model load. _planConfig is the scheduler config with this policy, for the
        // planner, which is a pure function of its config.
        private SpeculationOptions _speculation;
        private SchedulerConfig _planConfig;

        /// <summary>The speculation policy this executor currently applies.</summary>
        public SpeculationOptions Speculation => _speculation ?? _scheduler.Config.Speculation;

        private SchedulerConfig PlanConfig => _planConfig ?? _scheduler.Config;

        /// <summary>
        /// Replace the speculation policy for every step from now on. Engine thread
        /// only, between steps: every armed context is dropped (the next step re-arms
        /// under the new policy, over the tokens the caches already hold), and the
        /// one-time decline notices are re-armed so the new policy reports its own.
        /// </summary>
        public void SetSpeculation(SpeculationOptions options)
        {
            options ??= SpeculationOptions.Disabled;
            _speculation = options;
            _planConfig = _scheduler.Config.WithSpeculation(options);
            _specCtx?.Exec?.Dispose();
            _specCtx = null;
            foreach (var ctx in _fusedSpecCtx.Values)
                ctx.Exec?.Dispose();
            _fusedSpecCtx.Clear();
            _fusedSpecDeclined.Clear();
            _lateArmDeclinedFor = null;
            _speculationDeclineWarned = false;
            _specPrefixReuseDeclineWarned = false;
            _sharedGovernor = null;   // a new policy is measured afresh
        }

        private static readonly bool _cbDebug =
            string.Equals(Environment.GetEnvironmentVariable("TS_CB_DEBUG"), "1", StringComparison.Ordinal);

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
        private List<SequenceStepResult> ExecuteStepPerSequenceFused(
            IBatchedPagedModel fused, SchedulerOutput output, ExecutionOptions options)
        {
            int n = output.ScheduledWork.Count;
            var results = new List<SequenceStepResult>(n);
            if (n == 0) return results;
            if (_cbDebug)
            {
                var ids = new List<string>(n);
                foreach (var w in output.ScheduledWork)
                    ids.Add($"{w.Sequence.RequestId}:{(w.IsPrefill ? "P" : "D")}@{w.Sequence.NumComputedTokens}");
                Console.Error.WriteLine($"[cb] FUSED step n={n} owner={_currentOwner?.RequestId ?? "<none>"}" +
                    $" ownerStatus={(_currentOwner != null ? _currentOwner.Status.ToString() : "-")} work=[{string.Join(",", ids)}]");
            }

            // Transition from the single-stream (N==1) path: if a prior owner's
            // K/V is still live in the model's primary cache, hand it to that
            // request's per-request holder (zero-copy) so its history is
            // preserved and the primary is freed for future N==1 use.
            if (_currentOwner != null)
            {
                if (_currentOwner.Status == SequenceStatus.Running)
                {
                    // N=1 forwards borrow the model-owned logits buffer. The
                    // next request's Forward may overwrite it before this owner
                    // resumes, so detach it once at the ownership transition.
                    if (_currentOwner.LastLogits != null)
                        _currentOwner.LastLogits = (float[])_currentOwner.LastLogits.Clone();
                    fused.AdoptPrimaryCacheToFused(_currentOwner.RequestId);
                }
                _currentOwner = null;
                _ownerTokensInModel = 0;
                _ownerForwardedTokens = 0;
            }
            // The per-request caches make the single shared live-cache tracking
            // meaningless; drop any claim so a later same-session N==1 turn
            // re-establishes it cleanly from the primary cache.
            _liveCacheValid = false;
            // Fused per-request caches replace the shared linear cache the
            // speculative context tracks. A context that belongs to the owner just
            // adopted into its holder moves with it - its linear trunk forwards on
            // whatever cache is bound, which is now that holder - so the request
            // resumes speculating (TrySpeculativeFusedDecode catches it up over the
            // tokens the interlude took plainly) instead of re-arming from scratch
            // once it is solo again. A solo request always starts in the primary
            // cache, so this is every staggered arrival's first transition.
            if (_specCtx != null)
            {
                bool carry = _specCtx.BatchedTrunk == null
                    && _specCtx.Exec != null
                    && _specCtx.Seq != null
                    && fused.HasFusedSequenceCache(_specCtx.Seq.RequestId)
                    && _model is ISpeculativeTarget carriedSpec
                    && carriedSpec.SpecTrunkFollowsBoundCache
                    && !_fusedSpecCtx.ContainsKey(_specCtx.Seq.RequestId);
                if (carry)
                    _fusedSpecCtx[_specCtx.Seq.RequestId] = _specCtx;
                else
                    _specCtx.Exec?.Dispose();
                _specCtx = null;
            }

            // Track every request entering this path before the token-batched
            // fast path can return. Previously NoteFusedSequence lived only in
            // the serial fallback loop below, so an N=1 owner adopted directly
            // into an already-ready all-decode batch could finish with a valid
            // holder that the release hook did not know it should retain.
            foreach (var work in output.ScheduledWork)
                NoteFusedSequence(work.Sequence);

            // ---- TRUE token-batched decode fast path ----
            // When every scheduled item is a decode step (n>=2) and the model
            // supports it, decode all N tokens in ONE fused graph (weights loaded
            // once) instead of N serial per-seq forwards. Falls through to the
            // round-robin loop below when the model declines this batch.
            // A mixed step no longer forfeits batching: the decode SUBSET
            // batches here and the prefill chunks run through the per-sequence
            // loop below in the same step (the chunked-prefill mixing the
            // vLLM/SGLang schedulers do) - without this, a single admission
            // degraded every in-flight decode to a serial weight sweep.
            HashSet<string> handledBatched = null;
            if (options.BatchedFusedDecodeEnabled && n >= 2)
            {
                var decodeWork = new List<ScheduledSequenceWork>(n);
                foreach (var work in output.ScheduledWork)
                    if (!work.IsPrefill &&
                        fused.CanBatchDecode(work.Sequence.RequestId, work.Sequence.NumComputedTokens))
                        decodeWork.Add(work);
                int dn = decodeWork.Count;
                if (dn >= 2)
                {
                    var reqIds = new string[dn];
                    var btokens = new int[dn];
                    var bpositions = new int[dn];
                    bool allGreedy = true;
                    for (int i = 0; i < dn; i++)
                    {
                        var seq = decodeWork[i].Sequence;
                        // Peek-sample (deterministic for greedy); do NOT append or
                        // consume the device-sampled stash yet, so a decline below
                        // leaves the round-robin fallback a valid token source
                        // (device-sampled sequences have no LastLogits to re-sample).
                        btokens[i] = PeekPendingOrSample(seq);
                        bpositions[i] = seq.NumComputedTokens;
                        reqIds[i] = seq.RequestId;
                        if (allGreedy && !seq.GetOrCreateSampler().IsPlainGreedyArgmax)
                            allGreedy = false;
                    }
                    if (allGreedy)
                    {
                        // Device-sampled fast path: no host logits at all. The
                        // model returns each sequence's NEXT token (argmax of
                        // this step's logits); it is stashed on the sequence and
                        // consumed by TakePendingOrSample at the next step.
                        var nextTokens = new int[dn];
                        if (fused.TryForwardBatchedFusedDecodeSampled(reqIds, btokens, bpositions, nextTokens))
                        {
                            for (int i = 0; i < dn; i++)
                            {
                                var seq = decodeWork[i].Sequence;
                                seq.AppendOutputToken(btokens[i]);
                                seq.LastLogits = null;
                                seq.AdvanceComputedTokens(1);
                                seq.PendingDeviceToken = nextTokens[i];   // replaces the consumed stash
                                seq.PendingDevicePosition = seq.NumComputedTokens;
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
                            if (dn == n)
                                return results;
                            handledBatched = new HashSet<string>(reqIds);
                        }
                        // else: fall through to the logits variant below.
                    }
                    if (handledBatched == null)
                    {
                        var outLogits = new float[dn][];
                        if (fused.TryForwardBatchedFusedDecode(reqIds, btokens, bpositions, outLogits))
                        {
                            for (int i = 0; i < dn; i++)
                            {
                                var seq = decodeWork[i].Sequence;
                                seq.AppendOutputToken(btokens[i]);
                                seq.PendingDeviceToken = null;   // consumed by this step
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
                            if (dn == n)
                                return results;
                            handledBatched = new HashSet<string>(reqIds);
                        }
                        else
                        {
                            // The peek above may have advanced a stochastic
                            // sampler. Preserve that exact draw for the serial
                            // fallback; sampling again would advance its RNG a
                            // second time and change the seeded output stream.
                            // This also leaves an existing device-sampled token
                            // intact at the same position.
                            for (int i = 0; i < dn; i++)
                            {
                                var seq = decodeWork[i].Sequence;
                                seq.PendingDeviceToken = btokens[i];
                                seq.PendingDevicePosition = bpositions[i];
                            }
                        }
                        // Otherwise nothing was appended; fall through to the
                        // round-robin loop. A successful mixed-step batch sets
                        // handledBatched, so only warn when logits batching declined.
                        if (handledBatched == null && !_fusedBatchedDeclineWarned)
                        {
                            _fusedBatchedDeclineWarned = true;
                            _logger.LogWarning(
                                "The model declined the default batched fused-decode path for " +
                                "a {Count}-sequence decode step; serving sequences round-robin on the " +
                                "serial fused path instead (concurrency stays near 1x). Reported once.",
                                dn);
                        }
                    }
                }
            }

            foreach (var work in output.ScheduledWork)
            {
                // Decode subset already served by the batched fast path
                // above; only the prefill chunks of a mixed step remain.
                if (handledBatched != null && handledBatched.Contains(work.Sequence.RequestId))
                    continue;
                var seq = work.Sequence;
                int prevComputed = seq.NumComputedTokens;
                try
                {
                    bool freshCache = fused.BindSequenceCache(seq.RequestId);

                    if (_pendingRetainedFusedTruncations.Remove(
                            seq.RequestId,
                            out int retainedTruncationTarget))
                    {
                        if (freshCache)
                        {
                            throw new InvalidOperationException(
                                $"Retained fused cache for {seq.RequestId} was rebound but its holder was not found.");
                        }
                        // A retained holder was matched with a trailing rewind
                        // (FindRetainedFusedMatch). If the model declines it the holder's
                        // head is still where it was, so decoding would continue from
                        // positions this turn's prompt does not contain.
                        if (!_model.TryTruncateKVCache(retainedTruncationTarget))
                        {
                            _logger.LogInformation(
                                "Retained fused cache for {RequestId} could not rewind to {Tokens}; " +
                                "re-prefilling its prompt from the beginning.",
                                seq.RequestId, retainedTruncationTarget);
                            _model.ResetKVCache();
                            seq.ClearLiveCacheContinuation();
                            seq.PrefixCheckpointTaken = false;
                            // The scheduled chunk describes the old suffix. Let
                            // the next step plan from zero, including any shared
                            // checkpoint boundary, using the reserved blocks.
                            results.Add(new SequenceStepResult
                            {
                                Sequence = seq,
                                TokensForwarded = 0,
                                IsPrefill = true,
                            });
                            continue;
                        }
                    }

                    // A planned live-cache continuation is only materialized on
                    // this path when the prior owner's primary cache was adopted
                    // into this request's own holder above (owner still running
                    // under the same RequestId). A FRESH holder that still
                    // carries a continuation claim means the claimed live state
                    // is gone — its reserved blocks are accounting placeholders
                    // that were never written, and for snapshot-less models the
                    // inject below is skipped entirely — so decoding would
                    // silently run against an empty cache at a non-zero
                    // position. Drop the claim and re-prefill from position 0.
                    if (freshCache && seq.UsesLiveCacheContinuation)
                    {
                        // The reuse was PROMISED at admission and revoked here;
                        // without this line the user sees a full re-prefill on a
                        // turn the scheduler already counted as a cache hit.
                        _logger.LogInformation(
                            "Planned live-cache continuation for {RequestId} was revoked before " +
                            "execution (the claimed live state is gone); re-prefilling the full prompt.",
                            seq.RequestId);
                        seq.ClearLiveCacheContinuation();
                    }

                    // Prefix-cache reuse: a freshly-created cache whose sequence
                    // was admitted with already-computed (reused) tokens needs
                    // that prefix injected from the shared paged blocks before
                    // its first forward. (Injection READS the shared blocks into
                    // this request's own cache; it never writes them.)
                    if (freshCache && seq.NumComputedTokens > 0)
                        InjectAllBlocks(seq, seq.NumComputedTokens);

                    // A solo decode step on the bound holder may speculate (see
                    // TrySpeculativeFusedDecode); a mixed step keeps every request
                    // plain so its neighbours' latency stays bounded.
                    // Strictly solo: letting the lone decoder keep speculating beside a
                    // newcomer's prefill chunk was measured and rejected - the newcomer's
                    // first token slipped from 776 to 1429 ms on E4B (each chunk step
                    // also carried a verify), and on Qwen every such step switched
                    // holders and drained the verify family's state.
                    // ...and solo in the ENGINE, not just in this step: while a newcomer
                    // prefills, the scheduler hands the lone decoder steps of its own
                    // between the newcomer's chunks, and speculating on those (a verify
                    // plus a draft per step) delayed the newcomer's first token by 600 ms
                    // on E4B. A request that is the only one running speculates; one
                    // with a neighbour in flight decodes plainly until it is alone again.
                    if (!work.IsPrefill && n == 1 && _scheduler.RunningCount == 1
                        && TrySpeculativeFusedDecode(seq, prevComputed, out SequenceStepResult specResult))
                    {
                        results.Add(specResult);
                        continue;
                    }

                    int sampledToken = -1;
                    int[] inputTokens;
                    if (work.IsPrefill)
                    {
                        inputTokens = BuildPrefillChunk(seq, work);
                    }
                    else
                    {
                        sampledToken = TakePendingOrSample(seq);
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
                    if (work.IsPrefill)
                        MaybeCheckpointSharedPrefix(seq);

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
            if (_cbDebug)
            {
                var ids = new List<string>();
                foreach (var w in output.ScheduledWork)
                    ids.Add($"{w.Sequence.RequestId}:{(w.IsPrefill ? "P" : "D")}@{w.Sequence.NumComputedTokens}");
                Console.Error.WriteLine($"[cb] SOLO step n={output.ScheduledWork.Count} owner={_currentOwner?.RequestId ?? "<none>"}" +
                    $" work=[{string.Join(",", ids)}]");
            }

            // If a per-sequence-fused episode preceded this single-stream step,
            // the model's active KV cache may be a per-request holder. Reinstate
            // the primary cache before the in-place reset/inject logic below so
            // we never clobber a (possibly still-running) concurrent request's
            // cache. No-op when the primary cache is already active or the model
            // doesn't use per-request caches.
            // Reinstated regardless of the CURRENT capability value: the
            // capability can latch off after holders already exist (a fused-path
            // failure), and skipping the restore would leave a per-request
            // holder checked out for the universal path to trample.
            if (_model is IBatchedPagedModel pf)
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
            // safety property ÔÇö when the swap is unsafe the model reports
            // false and we stay with the owner.
            var picked = output.ScheduledWork[0];
            if (_currentOwner != null && output.ScheduledWork.Count > 0)
            {
                // Rotating ownership to another sequence requires extracting the
                // current owner's state and injecting the newcomer's ÔÇö a cross-
                // sequence snapshot round-trip. Models whose restore is not faithful
                // (Gemma 4 SWA) report SupportsCrossSequenceKvReuse=false, so we never
                // swap; concurrent sequences serialize on the owner instead of
                // producing corrupted output.
                bool canSwap = _model.SupportsKVStateSnapshot && _model.SupportsCrossSequenceKvReuse;
                if (!canSwap && output.ScheduledWork.Count > 1 && !_crossSeqSerializationWarned)
                {
                    // Without a swap the second request LOOKS hung, not slow: it
                    // streams nothing until the current owner finishes.
                    _crossSeqSerializationWarned = true;
                    _logger.LogWarning(
                        "{Count} concurrent requests are scheduled but this model cannot swap KV " +
                        "state between sequences, so they are served to completion one at a time - " +
                        "a later request does not stream until the current one finishes. Reported once.",
                        output.ScheduledWork.Count);
                }
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
                    var mtpResult = TryExecuteSpeculativeStep(seq, work, prevComputed);
                    if (mtpResult != null)
                    {
                        results.Add(mtpResult);
                        return results;
                    }
                    // A context for this sequence that didn't serve the step is
                    // stale from here on: the plain Forward below advances the
                    // trunk without capturing the hidden states drafting needs.
                    if (_specCtx != null && ReferenceEquals(_specCtx.Seq, seq))
                    {
                        _specCtx.Exec?.Dispose();
                        _specCtx = null;
                    }

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
                        sampledToken = TakePendingOrSample(seq);
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

                    // Forward writes the shared linear cache, not the model's
                    // paged arrays. Even if this sequence previously completed
                    // a paged step, its newly-advanced tail is now linear-only
                    // and must be restored/migrated before another batch.
                    seq.KvStateInPagedStorage = false;

                    // Sampling happens at the *start* of the next step (see
                    // SampleFromLogits at the top of this branch), and
                    // BatchExecutor calls _model.Forward only once per step.
                    // The model's `_logitsBuffer` is overwritten on the next
                    // Forward, but we always sample before that next Forward
                    // fires ÔÇö so a defensive 1 MB clone per token (Gemma 4
                    // vocab = 262144 ├ù 4 bytes) is wasted memcpy and GC
                    // pressure (~20 ┬Ás / token). Borrow the model's buffer
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

                    if (work.IsPrefill)
                        MaybeCheckpointSharedPrefix(seq);

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
                    _specCtx = null;
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
        /// serve the speculative trunk never arm here ÔÇö they are handled by
        /// <see cref="TryExecuteStepSpecBatchedTrunk"/> before the per-seq
        /// route is ever taken.</summary>
        private SequenceStepResult TryExecuteSpeculativeStep(
            SequenceState seq, ScheduledSequenceWork work, int prevComputed)
        {
            if (!Speculation.Enabled)
                return null;
            if (_model is not ISpeculativeTarget spec)
                return null;
            // Net-negative on backends without the accelerated multi-token
            // verify/draft path; the normal decode path serves the step.
            if (!spec.SpeculationProfitable)
                return null;
            if (spec is IBatchedSpeculativeTarget batchedSpec && batchedSpec.SupportsBatchedSpecTrunk)
                return null;
            // Multimodal prefill needs Forward's embedding-inject hook, which
            // SpecForward doesn't have.
            if (_model.MultimodalInjector != null
                && _model.MultimodalInjector.HasPendingEmbeddings(seq.RequestId))
            {
                return null;
            }

            // (Re-)arm on a prefill whose trunk cache agrees with what the
            // scheduler says is already computed. That covers the fresh
            // position-0 case AND an admission that ADOPTED a KV prefix (prefix
            // cache, or the previous turn of a chat).
            //
            // Prefix adoption used to be refused outright, because it skips trunk
            // positions a learned per-position draft head never saw. That is true
            // of NextN/MTP heads and stays true - they decline below - but it was
            // applied to every algorithm, and in a Web UI conversation EVERY turn
            // after the first adopts a prefix. So speculation armed on turn one,
            // never again, and DFlash measured as "no faster than plain" for the
            // rest of the session. Which algorithms survive a gap is now the
            // algorithm's own call (ISpeculator.CanArmAfterPrefixReuse).
            //
            // Replaces any stale context (including this sequence's own, e.g.
            // after preemption + re-prefill).
            //
            // NOT on a chunk boundary of a prompt this context is already serving.
            // A prompt longer than one prefill chunk arrives as several prefill
            // steps, and every one of them satisfies the arming test above (the
            // trunk position always agrees with the scheduler between chunks). Arming
            // again on chunk two threw away the context chunk one built - an n-gram
            // drafter lost the tokens it had mined - and, worse, looked to the gate
            // below like a KV-prefix reuse at position 1024: a per-token draft head
            // was then declined for the rest of the request. With the phone's
            // 1024-token chunk and a 5-7k-token agent prompt, that was EVERY request.
            bool continuesThisContext = _specCtx != null
                && _specCtx.BatchedTrunk == null
                && ReferenceEquals(_specCtx.Seq, seq)
                && _specCtx.NextPosition == prevComputed;
            if (work.IsPrefill && spec.CacheSeqLen == prevComputed && !continuesThisContext)
            {
                var exec = TryArmSpeculation(spec, seq, trunk: null, trunkLabel: "linear");
                if (exec == null)
                    return null;
                if (exec.CanSeedCommitted)
                {
                    // A drafter that can be seeded from the trunk's tokens (n-gram, a
                    // head that resumes after a gap) does not need the prompt to run
                    // through this context: that prefill takes the per-op speculative
                    // forward, capturing hidden rows for every prompt token and
                    // catching the head up over all of them, and cost a 1.2k-token
                    // first turn 33% (E2B n-gram: 478 vs 358 ms) to 36% (E4B draft
                    // head: 856 vs 627 ms) of its time to first token - and every
                    // request arriving behind it waited on that. Prefill plainly on
                    // the fused path; the late-arming branch below seeds the drafter
                    // at the first decode step.
                    exec.Dispose();
                    return null;
                }
                if (prevComputed > 0 && !exec.Speculator.CanArmAfterPrefixReuse)
                {
                    // This algorithm chains per-position state; a gap makes every
                    // proposal garbage. Serve the request on the plain path - and
                    // SAY so, because in a chat session every turn after the first
                    // adopts a prefix, so "--spec engaged on turn one and never
                    // again" looks exactly like speculation silently broken.
                    if (!_specPrefixReuseDeclineWarned)
                    {
                        _specPrefixReuseDeclineWarned = true;
                        _logger.LogWarning(
                            "Speculative decoding cannot arm for this request: the {Algorithm} " +
                            "algorithm does not resume after a reused KV prefix ({Tokens} tokens " +
                            "adopted), so turns that reuse cache decode plainly. Speculation engages " +
                            "only on turns that prefill from position 0. Reported once.",
                            exec.Speculator.Describe(), prevComputed);
                    }
                    exec.Dispose();
                    return null;
                }
                if (prevComputed > 0)
                {
                    // The trunk already holds prevComputed tokens this context never
                    // saw (a reused prefix). Hand them to the speculator now: an
                    // n-gram drafter armed here used to get its first commit at
                    // position prevComputed against an EMPTY corpus, treat it as a
                    // gap it could not account for, and stay silent for the rest of
                    // the request - every follow-up turn on the linear path drafted
                    // nothing, while the same turn on a holder drafted at 95%.
                    if (!exec.CanSeedCommitted)
                    {
                        exec.Dispose();
                        return null;
                    }
                    var committed = new int[prevComputed];
                    for (int i = 0; i < prevComputed; i++)
                        committed[i] = seq.TokenAt(i);
                    exec.SeedCommitted(committed);
                }
                _specCtx = new SpecSeqContext
                {
                    Seq = seq,
                    Exec = exec,
                    NextPosition = prevComputed,
                };
                seq.SpecStats = exec.Stats;
            }

            // Late arming, at a DECODE step with no context for this sequence: its
            // prefill ran on the plain path - a media turn, whose image/audio
            // embeddings only Forward's inject hook can place, is the everyday case -
            // and until now such a request decoded plainly to its last token. A
            // speculator that needs no hidden state (n-gram) can start here from the
            // tokens the trunk already holds; a learned head cannot (no trunk hidden
            // state for the last committed token to chain from) and is left alone.
            else if (!work.IsPrefill && !continuesThisContext && spec.CacheSeqLen == prevComputed
                     && prevComputed > 0
                     && !string.Equals(_lateArmDeclinedFor, seq.RequestId, StringComparison.Ordinal))
            {
                var exec = TryArmSpeculation(spec, seq, trunk: null, trunkLabel: "linear, after a plain prefill");
                if (exec == null)
                {
                    _lateArmDeclinedFor = seq.RequestId;   // decided once per request, not per token
                    return null;
                }
                if (!exec.CanSeedCommitted)
                {
                    exec.Dispose();
                    _lateArmDeclinedFor = seq.RequestId;
                    return null;
                }
                var committed = new int[prevComputed];
                for (int i = 0; i < prevComputed; i++)
                    committed[i] = seq.TokenAt(i);
                exec.SeedCommitted(committed);
                _specCtx = new SpecSeqContext
                {
                    Seq = seq,
                    Exec = exec,
                    NextPosition = prevComputed,
                };
                seq.SpecStats = exec.Stats;
            }

            // Continuity gate: same sequence, exact trunk position, and the
            // model's live cache agrees. Anything else (swap, preemption,
            // interleaved batched step) ran through an invalidation above.
            if (_specCtx == null
                || _specCtx.BatchedTrunk != null
                || !ReferenceEquals(_specCtx.Seq, seq)
                || _specCtx.NextPosition != prevComputed
                || spec.CacheSeqLen != prevComputed)
            {
                return null;
            }

            return ExecuteSpeculativeWorkCore(seq, work, prevComputed);
        }

        /// <summary>
        /// NextN/MTP speculative decoding with the trunk on the BATCHED paged
        /// path (see <see cref="IBatchedSpeculativeModel"/>): solo text
        /// sequences arm at a fresh full prefill and run draft/verify with
        /// trunk passes through <c>SpecForwardBatched</c> ÔÇö the same kernels
        /// the non-speculative batched baseline uses, with the sequence's K/V
        /// in paged storage throughout (prefix caching and concurrency
        /// transitions compose). Static routing gates live in
        /// <see cref="ExecutionPlanner"/>; this handler returns null only when
        /// the speculative context can't arm or lost continuity (disarmed
        /// context, prefix-reused admission), and the plan's next candidate
        /// then serves the step and drops any stale context.
        /// </summary>
        private List<SequenceStepResult> TryExecuteStepSpecBatchedTrunk(SchedulerOutput output)
        {
            // Static routing gates (speculation requested, batched-trunk
            // capability, profitability, solo step, no pending multimodal, not
            // fused-resident) are enforced by ExecutionPlanner before this
            // path becomes a plan candidate; only the DYNAMIC arming and
            // continuity checks below stay here.
            if (_model is not IBatchedSpeculativeTarget spec)
                return null;
            var work = output.ScheduledWork[0];
            var seq = work.Sequence;

            int prevComputed = seq.NumComputedTokens;

            // (Re-)arm at a fresh full prefill from position 0. A prefix-cache
            // or live-cache adoption skips trunk positions the MTP head never
            // saw; those requests run on the normal batched path instead.
            if (work.IsPrefill && prevComputed == 0)
            {
                var trunk = new BatchedSpecTrunk(spec, seq, 0);
                var exec = TryArmSpeculation(spec, seq, trunk, trunkLabel: "batched");
                if (exec == null)
                    return null;
                _specCtx = new SpecSeqContext
                {
                    Seq = seq,
                    BatchedTrunk = trunk,
                    Exec = exec,
                    NextPosition = 0,
                };
                seq.SpecStats = exec.Stats;
            }

            // Continuity gate: a batched context for this exact sequence at
            // this exact position. The drawn-token stash dies with a stale
            // context; the normal path re-samples from LastLogits (identical
            // under greedy, a one-token bias on rare disarm events otherwise).
            if (_specCtx == null
                || _specCtx.BatchedTrunk == null
                || !ReferenceEquals(_specCtx.Seq, seq)
                || _specCtx.NextPosition != prevComputed
                || _specCtx.BatchedTrunk.Position != prevComputed)
            {
                return null;
            }

            var results = new List<SequenceStepResult>(1);
            try
            {
                results.Add(ExecuteSpeculativeWorkCore(seq, work, prevComputed));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Batched-trunk MTP step failed for sequence {RequestId}", seq.RequestId);
                _specCtx = null;
                seq.Error = ex;
                results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
            }
            return results;
        }

        /// <summary>
        /// Build the speculative execution for one sequence: ask the registry
        /// for the algorithm the operator configured, wrap it around this
        /// trunk, and log one line per request so operators can SEE speculation
        /// engage at generation start (the cumulative stats only log at finish).
        /// Returns null - with a one-time operator notice - when no algorithm
        /// can serve this model, and the plan's next candidate then serves the
        /// step with plain decoding.
        /// </summary>
        // One governor for every speculative execution this executor arms: its
        // verdict is about the (model, drafter, backend) triple, which is the same
        // for every request, and a chat's turns are short - re-measuring from
        // scratch on each of them cost more than it decided. Replaced with the policy.
        private SpeculationCostGovernor _sharedGovernor;
        private SpeculationCostGovernor SharedGovernor => _sharedGovernor ??= new SpeculationCostGovernor();

        /// <summary>Whether a request can profit from drafting at all: it needs at
        /// least two tokens to decode, or every proposal would be wasted.</summary>
        private static bool CanSpeculate(SequenceState seq) => seq.MaxNewTokens >= 2;

        private SpeculativeExecution TryArmSpeculation(
            ISpeculativeTarget spec, SequenceState seq, ISpecTrunk trunk, string trunkLabel)
        {
            if (!CanSpeculate(seq))
                return null;
            var speculator = SpeculatorRegistry.Create(
                spec, Speculation, out string declineReason);
            if (speculator == null)
            {
                WarnSpeculationDeclinedOnce(declineReason);
                return null;
            }

            // Arming replaces any previous context (a new request, a re-prefill
            // after preemption). Today's speculators hold only managed buffers,
            // but the contract is IDisposable so a future one can own native
            // state - release it here rather than leaving it to the GC.
            _specCtx?.Exec?.Dispose();

            var exec = new SpeculativeExecution(spec, speculator, trunk, SharedGovernor);
            _logger.LogInformation(
                "Speculative decoding armed for {RequestId} (algorithm={Algorithm}, maxDraft={MaxDraft}, "
                + "pMin={PMin}, trunk={Trunk})",
                seq.RequestId, speculator.Describe(), exec.MaxDraftTokens, exec.MinDraftProb, trunkLabel);
            return exec;
        }

        private void WarnSpeculationDeclinedOnce(string reason)
        {
            if (_speculationDeclineWarned || string.IsNullOrEmpty(reason))
                return;
            _speculationDeclineWarned = true;
            _logger.LogWarning(
                "Speculative decoding was requested but is not available: {Reason}. "
                + "Serving standard decoding instead.", reason);
        }

        /// <summary>Shared MTP step body for both trunks; the caller has
        /// already validated arming and continuity on <see cref="_specCtx"/>.</summary>
        private SequenceStepResult ExecuteSpeculativeWorkCore(
            SequenceState seq, ScheduledSequenceWork work, int prevComputed)
        {
            bool batchedTrunk = _specCtx.BatchedTrunk != null;
            if (work.IsPrefill)
            {
                int[] chunk = BuildPrefillChunk(seq, work);
                var swPrefill = Stopwatch.StartNew();
                float[] logits = _specCtx.Exec.PrefillStep(chunk, prevComputed);
                swPrefill.Stop();
                seq.LastLogits = logits;
                CompleteSpeculativeStepBookkeeping(seq, chunk.Length, batchedTrunk);
                _specCtx.NextPosition = seq.NumComputedTokens;

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
            SpeculativeStepOutcome outcome = RunSpeculativeDecodeStep(
                _specCtx, seq, prevComputed, out int sampledToken, out List<int> accepted, out long decodeTicks);

            seq.LastLogits = outcome.NextLogits;
            int advanced = 1 + outcome.AcceptedCount;
            CompleteSpeculativeStepBookkeeping(seq, advanced, batchedTrunk);
            _specCtx.NextPosition = prevComputed + advanced;
            _specCtx.PendingNextToken = outcome.NextToken;
            PublishDrawnToken(seq, outcome.NextToken);

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
                ForwardElapsedTicks = decodeTicks,
            };
        }

        /// <summary>
        /// The token a verify DREW for the sequence's next position, stashed on the
        /// sequence as well as on the speculative context: if the next step leaves the
        /// speculative path (a neighbour admitted, a context dropped), the plain paths'
        /// Take/PeekPendingOrSample then emit exactly that draw instead of drawing the
        /// same row a second time - which under temperature would bias the stream
        /// toward the rejected draft. Greedy never noticed; sampling did, on every
        /// admission.
        /// </summary>
        private static void PublishDrawnToken(SequenceState seq, int drawn)
        {
            if (drawn < 0) return;
            seq.PendingDeviceToken = drawn;
            seq.PendingDevicePosition = seq.NumComputedTokens;
        }

        /// <summary>
        /// One speculative decode step for <paramref name="seq"/> over
        /// <paramref name="ctx"/>: takes the next output token (the one drawn by the
        /// previous verify when there is one), caps the window to the request's
        /// budget and block capacity, and runs draft / verify / rollback. The caller
        /// does the path's own bookkeeping with the outcome.
        /// </summary>
        private SpeculativeStepOutcome RunSpeculativeDecodeStep(
            SpecSeqContext ctx, SequenceState seq, int prevComputed,
            out int sampledToken, out List<int> accepted, out long elapsedTicks)
        {
            // The next output token: the one DRAWN during the previous step's
            // verification when available (emitting anything else would bias
            // the stream toward the drafts), otherwise sampled as usual.
            if (ctx.PendingNextToken >= 0)
            {
                sampledToken = ctx.PendingNextToken;
                ctx.PendingNextToken = -1;
                seq.PendingDeviceToken = null;   // the same draw, published for the plain paths; consumed here
            }
            else
            {
                sampledToken = TakePendingOrSample(seq);
            }
            seq.AppendOutputToken(sampledToken);

            // Cap the draft window so this step's 1+K forwarded tokens fit in
            // (a) the request's remaining token budget and (b) the KV blocks
            // the scheduler allocated - it reserves capacity for ONE decode
            // token per step, so block-boundary steps degrade to plain decode
            // for one step instead of overrunning the block table.
            int kMax = Math.Min(
                seq.MaxNewTokens - seq.OutputTokens.Count,
                seq.BlockTable.FreeSlotsInCurrentBlocks - 1);

            // Nothing is forwarded PAST a stop. The engine ends the sequence at an
            // end-of-turn token (or a repetition stop) and trims the token list
            // there; rows the verify had accepted beyond it stayed in the cache, so a
            // retained holder was longer than the tokens it was recorded as holding
            // and - on a sliding-window ring - those rows had evicted positions the
            // next turn still attends to. A chat prompt contains the turn boundary
            // several times, so an n-gram drafter proposes exactly that continuation
            // and the trunk agrees with it. The window therefore does not start at a
            // stop token, and the accept loop ends at the first stop it draws.
            bool stopRepetition = _scheduler.Config.StopRepetition && (seq.SamplingConfig?.StopRepetition ?? true);
            bool IsStop(int token) => _model.Tokenizer != null && _model.Tokenizer.IsEos(token);
            if (IsStop(sampledToken))
                kMax = 0;
            // The sampled token can itself complete a repetition loop. The engine then
            // stops at it and truncates every accepted draft from the sequence - but a
            // bound holder that cannot be truncated (Qwen 3.5) would keep those rows,
            // and the next turn would continue from a cache longer than its tokens.
            // Do not verify past a token the guard is about to stop at.
            if (kMax > 0 && stopRepetition
                && RepetitionGuard.IsLooping(seq.OutputTokens, seq.OutputTokens.Count, out _, out _))
                kMax = 0;
            bool stopped = false;

            var acceptedTokens = new List<int>();
            var penaltySampler = seq.GetOrCreateSampler();
            var swDecode = Stopwatch.StartNew();
            SpeculativeStepOutcome outcome = ctx.Exec.DecodeStep(
                sampledToken,
                prevComputed,
                kMax,
                // Each verify row is drawn with the request's own sampler over
                // the live output history (kept exact by onDraftAccepted). Once a
                // stop has been accepted no further row is drawn: -1 matches no
                // draft, so the loop ends there and the rows past it are rejected.
                drawNext: rowLogits =>
                    stopped ? -1 : seq.GetOrCreateSampler().Sample(rowLogits, seq.OutputTokens),
                // Penalty-aligned drafting: the draft head must argmax the
                // same penalized distribution verification draws from, or
                // acceptance decays toward zero as the output history grows.
                adjustDraftLogits: (draftLogits, pendingDrafts) =>
                    penaltySampler.ApplyPenalties(draftLogits, seq.OutputTokens, pendingDrafts),
                onDraftAccepted: d =>
                {
                    seq.AppendOutputToken(d);
                    acceptedTokens.Add(d);
                    if (IsStop(d)
                        || (stopRepetition
                            && RepetitionGuard.IsLooping(seq.OutputTokens, seq.OutputTokens.Count, out _, out _)))
                        stopped = true;
                });
            swDecode.Stop();
            accepted = acceptedTokens;
            elapsedTicks = swDecode.ElapsedTicks;
            return outcome;
        }

        // ---- Speculation on a per-request fused holder ----
        //
        // Every turn a real chat makes after its first lives in a per-request fused
        // holder: the clone of the shared-prefix checkpoint that starts a new chat,
        // the retained holder the previous turn left. The planner routes those to
        // the per-sequence fused path and rejects the linear speculative route for
        // them ("sequence lives in a per-request fused cache"), so a host that
        // continues conversations from holders - TensorAgent - never speculated at
        // all. The bound holder IS the model's active cache (BindSequenceCache
        // repoints the live arrays), so the linear speculative trunk runs on it
        // unchanged; what the fused path needs is its own context per request,
        // armed over the tokens the holder already holds.
        private readonly Dictionary<string, SpecSeqContext> _fusedSpecCtx = new(StringComparer.Ordinal);
        private readonly HashSet<string> _fusedSpecDeclined = new(StringComparer.Ordinal);

        private void DisposeFusedSpecContext(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            if (_fusedSpecCtx.Remove(requestId, out var ctx))
                ctx.Exec?.Dispose();
            _fusedSpecDeclined.Remove(requestId);
        }

        /// <summary>
        /// A speculative decode step for a solo sequence served from its bound
        /// per-request fused holder. Returns false - nothing forwarded - when
        /// speculation is off, cannot serve this model, or cannot arm over this
        /// request; the caller then runs the plain fused step.
        /// </summary>
        private bool TrySpeculativeFusedDecode(
            SequenceState seq, int prevComputed, out SequenceStepResult result)
        {
            result = null;
            if (!Speculation.Enabled || prevComputed <= 0 || !CanSpeculate(seq))
                return false;
            if (_model is not ISpeculativeTarget spec || !spec.SpeculationProfitable)
                return false;
            // Only a trunk that forwards on the BOUND cache can serve a holder; one
            // written against the primary linear cache (Qwen 3.5) would unbind it.
            if (!spec.SpecTrunkFollowsBoundCache)
            {
                WarnSpeculationDeclinedOnce(
                    $"{_model.GetType().Name}'s speculative trunk forwards on its primary cache, not on a bound holder; "
                    + "turns served from a cached holder (every turn after a chat's first) decode plainly");
                return false;
            }
            // A batched-trunk model keeps its own paged route.
            if (spec is IBatchedSpeculativeTarget batchedSpec && batchedSpec.SupportsBatchedSpecTrunk)
                return false;
            if (_model.MultimodalInjector != null && _model.MultimodalInjector.HasPendingEmbeddings(seq.RequestId))
                return false;
            // The bound holder must agree with the scheduler about the position.
            if (spec.CacheSeqLen != prevComputed)
                return false;

            if (_fusedSpecCtx.TryGetValue(seq.RequestId, out var ctx) && ctx.NextPosition < prevComputed
                && ctx.Exec != null && ctx.Exec.CanSeedCommitted)
            {
                // The holder advanced plainly while this request shared its steps
                // with neighbours (or took a plain step after a decline). Resume the
                // context over those tokens rather than re-arming: a re-arm
                // re-indexed the whole context and re-ran the probe on every
                // 1 -> N -> 1 transition, which staggered short requests hit constantly.
                var gap = new int[prevComputed - ctx.NextPosition];
                for (int i = 0; i < gap.Length; i++)
                    gap[i] = seq.TokenAt(ctx.NextPosition + i);
                ctx.Exec.CatchUp(gap, ctx.NextPosition);
                ctx.NextPosition = prevComputed;
                ctx.PendingNextToken = -1;   // the plain path drew and consumed it
            }
            if (!_fusedSpecCtx.TryGetValue(seq.RequestId, out ctx) || ctx.NextPosition != prevComputed)
            {
                if (ctx != null)
                {
                    // Something forwarded past this context (a truncation, a head that
                    // cannot resume); start over from the holder's tokens.
                    _fusedSpecCtx.Remove(seq.RequestId);
                    ctx.Exec?.Dispose();
                }
                if (_fusedSpecDeclined.Contains(seq.RequestId))
                    return false;
                var speculator = SpeculatorRegistry.Create(spec, Speculation, out string declineReason);
                if (speculator == null)
                {
                    WarnSpeculationDeclinedOnce(declineReason);
                    _fusedSpecDeclined.Add(seq.RequestId);
                    return false;
                }
                var exec = new SpeculativeExecution(spec, speculator, trunk: null, SharedGovernor);
                if (!exec.CanSeedCommitted)
                {
                    exec.Dispose();
                    _fusedSpecDeclined.Add(seq.RequestId);
                    return false;
                }
                var committed = new int[prevComputed];
                for (int i = 0; i < prevComputed; i++)
                    committed[i] = seq.TokenAt(i);
                exec.SeedCommitted(committed);
                ctx = new SpecSeqContext { Seq = seq, Exec = exec, NextPosition = prevComputed };
                _fusedSpecCtx[seq.RequestId] = ctx;
                seq.SpecStats = exec.Stats;
                _logger.LogInformation(
                    "Speculative decoding armed for {RequestId} (algorithm={Algorithm}, maxDraft={MaxDraft}, "
                    + "pMin={PMin}, trunk=fused holder over {Committed} committed tokens)",
                    seq.RequestId, speculator.Describe(), exec.MaxDraftTokens, exec.MinDraftProb, prevComputed);
            }

            SpeculativeStepOutcome outcome = RunSpeculativeDecodeStep(
                ctx, seq, prevComputed, out int sampledToken, out List<int> accepted, out long decodeTicks);

            // The fused path's own bookkeeping: the holder is the request's, so
            // there is no live-cache claim and no block capture to make.
            seq.LastLogits = outcome.NextLogits;
            int advanced = 1 + outcome.AcceptedCount;
            seq.AdvanceComputedTokens(advanced);
            ctx.NextPosition = prevComputed + advanced;
            ctx.PendingNextToken = outcome.NextToken;
            PublishDrawnToken(seq, outcome.NextToken);
            if (!seq.FirstTokenAt.HasValue)
                seq.FirstTokenAt = DateTime.UtcNow;

            result = new SequenceStepResult
            {
                Sequence = seq,
                TokensForwarded = advanced,
                SampledToken = sampledToken,
                ExtraTokens = accepted.Count > 0 ? accepted : null,
                IsPrefill = false,
                FullBlocksCaptured = 0,
                ForwardElapsedTicks = decodeTicks,
            };
            return true;
        }

        /// <summary>Per-step bookkeeping for MTP steps (which may advance more
        /// than one token). Linear trunk mirrors the plain per-seq path
        /// (owner counters + live-cache tracking + the caller's block
        /// capture); batched trunk mirrors <see cref="ExecuteStepBatched"/>
        /// (K/V lives in the model's paged storage, blocks get hash-registered
        /// for prefix sharing).</summary>
        private void CompleteSpeculativeStepBookkeeping(SequenceState seq, int tokensForwarded, bool batchedTrunk)
        {
            int prevComputed = seq.NumComputedTokens;
            seq.AdvanceComputedTokens(tokensForwarded);
            if (batchedTrunk)
            {
                _liveCacheValid = false;
                seq.KvStateInPagedStorage = true;
                seq.BlockTable.SetHoldsModelPagedKv(prevComputed, seq.NumComputedTokens, true);
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
                // The trunk's live cache is the sequence's own here, exactly as on the
                // plain linear path, so the shared-prefix boundary is checkpointed the
                // same way (a prefill chunk is the only step that can land on it).
                if (tokensForwarded > 1 || seq.NumComputedTokens <= seq.PromptTokens.Count)
                    MaybeCheckpointSharedPrefix(seq);
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
            _lastLiveDeclineReason = null;
            if (seq == null)
                return 0;
            if (!_liveCacheValid || _liveCacheSeq == null || _liveCacheLen <= 0)
                return LiveContinuationDeclined(seq, "no live cache resident (reset, batched step, or first request)");
            // An explicit cache policy must be enforced by the block-granular
            // pooled path. Reusing the complete live holder here would let a
            // cache-none request (or a finite breakpoint) silently reuse past
            // the boundary selected by the client.
            if (_liveCacheSeq.CacheBreakpoints != null || seq.CacheBreakpoints != null)
                return LiveContinuationDeclined(seq, "the source or target request has an explicit cache boundary");
            // Multimodal placeholders can render to identical token IDs while
            // carrying different image/audio embeddings. Their K/V is reusable
            // only when the prepared media fingerprint is identical as well.
            if (!string.Equals(
                    _liveCacheSeq.MediaFingerprint,
                    seq.MediaFingerprint,
                    StringComparison.Ordinal))
                return LiveContinuationDeclined(seq, "the request media fingerprint differs from the live cache");
            // Only worth it when the pooled path cannot already reuse the full
            // prefix. Models that opt out of cross-sequence snapshots have an
            // effective pooled cap of zero, but continuing their still-live
            // primary cache is safe and avoids a complete re-prefill.
            int cap = _model.SupportsCrossSequenceKvReuse
                ? _model.MaxReusablePrefixTokens
                : 0;
            if (cap == int.MaxValue)
            {
                // Pooled reuse is uncapped for this model, so the block path already
                // covers the whole prefix. Not a decline worth reporting as a loss.
                _lastLiveDeclineReason = "not needed (pooled prefix reuse is uncapped for this model)";
                return 0;
            }

            int liveLen = Math.Min(_liveCacheLen, _liveCacheSeq.NumTotalTokens);

            if (liveLen <= cap)
                return LiveContinuationDeclined(seq,
                    $"live prefix {liveLen} within the pooled reuse cap {cap}");

            // Longest common prefix between the new prompt and what the cache holds.
            int lcp = 0;
            int limit = Math.Min(liveLen, seq.PromptTokens.Count);
            while (lcp < limit && seq.PromptTokens[lcp] == _liveCacheSeq.TokenAt(lcp))
                lcp++;

            if (seq.PromptTokens.Count <= lcp)
            {
                // The cache already holds this ENTIRE prompt (and usually the answer it
                // produced): the same question asked again, a regenerated turn, a new
                // chat that opens with the first chat's exact words. Continuing needs at
                // least one prompt token to forward for fresh logits, so keep everything
                // but the last one and rewind the rest — the same rewind a trailing
                // control token gets below, and subject to the same limit. Declining
                // outright here re-prefilled the whole conversation for a prompt the
                // cache had already seen to the last token.
                if (!_model.SupportsKVCacheTruncation)
                {
                    return LiveContinuationDeclined(seq,
                        $"prompt ({seq.PromptTokens.Count} tokens) has no new suffix past the matched prefix ({lcp}) " +
                        "and this model cannot rewind its KV state to re-forward the last token");
                }
                lcp = seq.PromptTokens.Count - 1;
                if (lcp <= 0)
                    return LiveContinuationDeclined(seq, "prompt is a single token the cache already holds");
            }

            var exactReuse = _model as IExactFusedCacheReuse;
            if (exactReuse?.SupportsExactFusedCacheReuse != true) exactReuse = null;
            if (lcp == liveLen)
                return exactReuse == null || exactReuse.CanReuseLivePrefix(liveLen, lcp)
                    ? liveLen : LiveContinuationDeclined(seq, "native live holder is failed or its head differs");

            // A rewind has to land where the model can put its head - DeepSeek V4.1 only
            // stops on a compression-block boundary. Align DOWN before measuring the
            // rewind, so the depth limit below and the truncate at execution time are
            // talking about the same target rather than the plan promising a position the
            // model then declines.
            int align = _model.KVCacheTruncationGranularity;
            if (align > 1 && lcp % align != 0)
            {
                lcp -= lcp % align;
                if (lcp <= 0)
                    return LiveContinuationDeclined(seq,
                        $"the matched prefix is shorter than this model's {align}-token rewind granularity");
            }

            // The cache holds tokens the prompt does not reproduce. Continuing means
            // rewinding past them, which is only sound when the model can rewind.
            int rewind = liveLen - lcp;
            if (exactReuse == null && rewind > MaxLiveContinuationRewindTokens)
            {
                return LiveContinuationDeclined(seq,
                    $"prompt diverges from the live cache at token {lcp} of {liveLen}, which would need a " +
                    $"{rewind}-token rewind (limit {MaxLiveContinuationRewindTokens}); " +
                    $"context prompt=[{DescribeTokenWindow(k => seq.PromptTokens[k], seq.PromptTokens.Count, lcp)}] " +
                    $"cached=[{DescribeTokenWindow(k => _liveCacheSeq.TokenAt(k), liveLen, lcp)}]");
            }
            if (!_model.SupportsKVCacheTruncation)
            {
                return LiveContinuationDeclined(seq,
                    $"prompt diverges from the live cache at token {lcp} of {liveLen} and this model cannot " +
                    "rewind its KV state (recurrent layers have no reverse)");
            }
            if (exactReuse != null ? !exactReuse.CanReuseLivePrefix(liveLen, lcp)
                                   : !_model.CanTruncateKVCache(liveLen, lcp))
                return LiveContinuationDeclined(seq,
                    $"rewinding the live cache from {liveLen} to {lcp} would lose required history");
            if (lcp <= cap)
                return LiveContinuationDeclined(seq,
                    $"matched prefix {lcp} (after a {rewind}-token rewind) is within the pooled reuse cap {cap}");
            // The eligibility check above rejects loss of required history.
            // Prefer a nearby exact checkpoint when available, including one a
            // few control tokens shorter than the live match. Keep the shared
            // prefix itself independent of any conversation-tail rewind.
            if (exactReuse == null && cap != int.MaxValue && lcp < seq.SharedPrefixTokens)
            {
                return LiveContinuationDeclined(seq,
                    $"prompt diverges from the live cache at token {lcp} of {liveLen}, inside the shared prefix " +
                    $"({seq.SharedPrefixTokens} tokens); a rewind on a circular cache would not be exact there");
            }
            if (cap != int.MaxValue && HasExactRetainedContinuation(seq, lcp - MaxLiveContinuationRewindTokens))
            {
                return LiveContinuationDeclined(seq,
                    $"prompt diverges from the live cache at token {lcp} of {liveLen}; a retained state serves " +
                    "that prefix exactly, and a rewind on a circular cache would not be exact");
            }

            _logger.LogDebug(
                "Live-cache continuation for {RequestId} rewinding {Rewind} trailing token(s) the prompt does " +
                "not reproduce: keeping {Lcp} of {LiveLen}.",
                seq.RequestId, rewind, lcp, liveLen);
            return lcp;
        }

        /// <summary>
        /// How many trailing cached tokens a live continuation may rewind past.
        ///
        /// <para>
        /// The case this exists for is one or two tokens long: a turn ends on a
        /// generation-only control token — an EOS, or Gemma 4's
        /// <c>&lt;|tool_response&gt;</c> — that the model samples and the engine
        /// forwards, but that the chat template never reproduces when it re-renders
        /// that turn as history. Without a rewind the whole conversation re-prefills
        /// from token 0 on the next turn, which is why an EOS-terminated turn used to
        /// report 0% reuse while a max_tokens-terminated turn of the same conversation
        /// reported ~95%, and why an Agent Skills lookup reused nothing at all.
        /// </para>
        /// <para>
        /// This is a matching limit, not a guarantee that the model can restore
        /// those rows. CanTruncateKVCache independently rejects unsafe depths;
        /// wrapped Gemma caches need an exact checkpoint or a re-prefill.
        /// </para>
        /// </summary>
        private const int MaxLiveContinuationRewindTokens = 16;

        /// <summary>Record why live-cache continuation was refused and return 0. The
        /// path had five distinct bare `return 0`s, which is why a report of "KV
        /// reuse is 0" carried no way to tell which one fired.
        ///
        /// <para>
        /// The reason is REMEMBERED, not announced. Refusing here says nothing about
        /// what the request ends up reusing: the scheduler tries retained model state
        /// and the pooled blocks next, and on a fused model (Qwen 3.5/3.6, Gemma 4)
        /// the retained holder is the NORMAL source of reuse, so this path is taken
        /// on essentially every request while reuse runs at 95-100%. Announcing "this
        /// turn re-prefills its full prompt (KV reuse 0)" from here was therefore
        /// wrong on almost every line it printed. The scheduler reports the single
        /// truthful outcome once the three mechanisms have all had their turn.
        /// </para></summary>
        private int LiveContinuationDeclined(SequenceState seq, string reason)
        {
            _lastLiveDeclineReason = reason;
            _logger.LogDebug(
                "Live-cache continuation declined for {RequestId}: {Reason}.",
                seq.RequestId, reason);
            return 0;
        }

        /// <summary>Why the last <see cref="ComputeLiveContinuationLcp"/> found no
        /// usable live prefix, or null when it did. Read by the scheduler at
        /// admission to explain a turn that reused nothing.</summary>
        public string? LastLiveContinuationDeclineReason => _lastLiveDeclineReason;

        /// <summary>Why the last <see cref="ComputeFusedContinuationLcp"/> found no
        /// usable retained holder or shared-prefix checkpoint, or null when it did.</summary>
        public string? LastFusedContinuationDeclineReason => _lastFusedDeclineReason;

        /// <summary>Render the few tokens either side of <paramref name="center"/> as
        /// "id:piece" so a prefix divergence names the actual text that differs.
        /// Without this a mismatch report is a pair of bare integers.</summary>
        private string DescribeTokenWindow(Func<int, int> tokenAt, int count, int center, int radius = 3)
        {
            var sb = new StringBuilder();
            int from = Math.Max(0, center - radius);
            int to = Math.Min(count - 1, center + radius);
            for (int k = from; k <= to; k++)
            {
                if (sb.Length > 0) sb.Append(' ');
                int id = tokenAt(k);
                if (k == center) sb.Append('*');
                sb.Append(id);
                string piece = null;
                try { piece = _model.Tokenizer?.Decode(new List<int> { id }); }
                catch (Exception) { /* a lone special/partial token may not decode */ }
                if (!string.IsNullOrEmpty(piece))
                    sb.Append(':').Append(piece.Replace("\n", "\\n").Replace("\r", "\\r"));
            }
            return sb.ToString();
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
            if (seq.BlockTable.NumBlocks != 0)
            {
                LiveContinuationDeclined(seq,
                    $"blocks already reserved ({seq.BlockTable.NumBlocks}) before adoption");
                return false;
            }

            int neededBlocks = (lcp + _blockSize - 1) / _blockSize;
            var blocks = _pool.AllocateNew(neededBlocks);
            if (blocks == null)
            {
                // Pool pressure -> the caller falls back to the capped pool path.
                // Silent before: a planned continuation that failed HERE looked
                // exactly like one that was never planned.
                LiveContinuationDeclined(seq,
                    $"block pool could not reserve {neededBlocks} block(s) for the {lcp}-token prefix");
                return false;
            }

            for (int i = 0; i < blocks.Length; i++)
                seq.BlockTable.AppendBlock(blocks[i]);

            seq.SetComputedTokensForPrefixAdoption(lcp);
            seq.PrefixCacheReusedTokens = lcp;
            seq.UsesLiveCacheContinuation = true;
            return true;
        }

        /// <summary>True when the loaded model serves concurrent decode through
        /// per-request fused holders and explicitly supports retaining/re-keying a
        /// completed holder. Do not infer this from the pooled-prefix cap: Qwen's
        /// byte snapshots are deliberately not cross-request reusable, but its full
        /// request-owned attention + recurrent-state holder is.</summary>
        private bool ModelUsesRetainableFusedCache()
            => ExecutionOptions.FromEnvironment().RetainedFusedCacheEnabled
            && _model is IBatchedPagedModel f
            && f.SupportsPerSequenceFusedForward
            && f.SupportsRetainedFusedCache;

        /// <summary>True when the loaded model can copy its complete state at a
        /// shared-prefix boundary and start later requests from a clone of it.</summary>
        private bool ModelSupportsPrefixCheckpoints()
        {
            var options = ExecutionOptions.FromEnvironment();
            // A clone lives in a per-request fused holder and runs on the per-sequence
            // fused path; if the planner cannot route a fused-resident solo request
            // there (TS_PER_SEQ_FUSED=0, TS_SCHED_DISABLE_BATCHED, a model without
            // batched paged attention), the request would fall to the linear path with
            // placeholder blocks nothing ever wrote and silently re-prefill with its
            // reuse misreported. And adoption itself lives behind the scheduler's
            // prefix-caching switch, so a checkpoint nothing can adopt is a copy for
            // nothing.
            return options.PrefixCheckpointsEnabled
                && options.PerSeqFusedEnabled
                && !options.BatchedPathDisabled
                && _scheduler.PrefixCacheConfigured
                && ModelUsesRetainableFusedCache()
                && _model is IBatchedPagedModel f
                && f.SupportsPrefixCheckpoints
                && ExecutionCapabilities.FromModel(_model).SupportsBatchedPagedAttention;
        }

        /// <summary>Whether shared-prefix checkpoints are in use for this model, so the
        /// scheduler ends prefill chunks at the boundary the chat layer marked.</summary>
        public bool PrefixCheckpointsSupported => ModelSupportsPrefixCheckpoints();

        /// <summary>
        /// After a prefill step: if this sequence has just reached the end of its shared
        /// prefix, checkpoint the model's state there — unless an identical checkpoint
        /// already exists. Called on every per-sequence path with the sequence's cache
        /// active (the primary cache on the linear path, its own holder on the fused
        /// one); the model copies whichever is active.
        /// </summary>
        private void MaybeCheckpointSharedPrefix(SequenceState seq)
        {
            if (seq == null || seq.PrefixCheckpointTaken || seq.SharedPrefixTokens <= 0)
                return;
            if (seq.NumComputedTokens != seq.SharedPrefixTokens)
                return;
            // Whatever happens below, this sequence is done with the boundary: the
            // scheduler stops aligning to it and this is not asked again.
            seq.PrefixCheckpointTaken = true;
            if (!ModelSupportsPrefixCheckpoints() || _model is not IBatchedPagedModel fused)
                return;
            if (seq.CacheBreakpoints != null)
                return; // an explicit cache policy is served by the pooled path only

            int n = seq.SharedPrefixTokens;
            for (var node = _prefixCheckpoints.First; node != null; node = node.Next)
            {
                var existing = node.Value;
                if (existing.Tokens.Length != n) continue;
                bool same = true;
                for (int i = 0; i < n && same; i++)
                    same = existing.Tokens[i] == seq.PromptTokens[i];
                if (same)
                {
                    // Keep the most recently confirmed prefix at the tail (LRU).
                    _prefixCheckpoints.Remove(node);
                    _prefixCheckpoints.AddLast(node);
                    return;
                }
            }

            LinkedListNode<RetainedFusedCache> prepared;
            try
            {
                // Prepare both the record and its list node before the model owns
                // a new checkpoint. Publication after success must not allocate.
                var tokens = AllocateRetainedCacheTokens(n);
                for (int i = 0; i < n; i++) tokens[i] = seq.PromptTokens[i];
                prepared = new LinkedListNode<RetainedFusedCache>(new RetainedFusedCache
                {
                    RequestId = $"{_checkpointKeyPrefix}{++_prefixCheckpointSerial}",
                    Tokens = tokens,
                    IsPrefixCheckpoint = true,
                });
            }
            catch (OutOfMemoryException)
            {
                return; // The optional copy has not changed model ownership.
            }
            string key = prepared.Value.RequestId;
            var sw = Stopwatch.StartNew();
            bool taken;
            try
            {
                taken = fused.TryCheckpointActiveCache(key);
            }
            catch (OutOfMemoryException)
            {
                return; // The model rolls back an unpublished copy.
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Shared-prefix checkpoint for {RequestId} at {Tokens} tokens failed; new chats will re-prefill this prefix.",
                    seq.RequestId, n);
                return;
            }
            if (!taken)
            {
                _logger.LogInformation(
                    "Shared-prefix checkpoint for {RequestId} at {Tokens} tokens was declined by the model; " +
                    "new chats will re-prefill this prefix.",
                    seq.RequestId, n);
                return;
            }

            _prefixCheckpoints.AddLast(prepared);

            EvictPrefixCheckpointsBeyondBudget(fused);
            _logger.LogInformation(
                "Shared-prefix checkpoint {Key} taken at {Tokens} tokens for {RequestId} in {Ms:F0} ms " +
                "({Count} kept); a new chat that starts with this prefix will continue from a copy of it.",
                key, n, seq.RequestId, sw.Elapsed.TotalMilliseconds, _prefixCheckpoints.Count);
            SavePrefixCheckpointToStore(fused, key, prepared.Value.Tokens);
        }

        /// <summary>Longest reusable prefix from a retained fused holder, or 0 when
        /// no retained holder applies. A short trailing control-token tail may be
        /// rewound when the rendered history intentionally omitted it.
        /// The matched holder contains the finished request's complete model-owned
        /// continuation state: circular K/V for Gemma 4, or attention K/V plus GDN
        /// recurrent state for Qwen 3.5/3.6. Continuing from it can therefore reuse
        /// the entire recorded prefix without reconstructing an incomplete paged
        /// snapshot ÔÇö the cross-request analogue of
        /// <see cref="ComputeLiveContinuationLcp"/>. Invoked by the scheduler at
        /// admission (same worker thread as the executor).</summary>
        public int ComputeFusedContinuationLcp(SequenceState seq)
        {
            _lastFusedDeclineReason = null;
            if (seq == null) return 0;
            if (!ModelUsesRetainableFusedCache())
            {
                _lastFusedDeclineReason =
                    "unavailable (this model+backend does not serve requests from retainable per-request holders)";
                return 0;
            }
            // A checkpoint saved by an earlier process is worth reading only for a
            // prompt that would clone it, and only once: after this it is in memory.
            TryRestorePrefixCheckpointFromStore(seq);
            if (_retainedFused.Count == 0 && _prefixCheckpoints.Count == 0)
            {
                _lastFusedDeclineReason =
                    "nothing retained yet (no finished request's holder and no shared-prefix checkpoint)";
                return 0;
            }
            FindRetainedFusedMatch(seq, out int lcp);
            if (lcp <= 0)
            {
                _lastFusedDeclineReason =
                    $"none of the {_retainedFused.Count} retained holder(s) and " +
                    $"{_prefixCheckpoints.Count} checkpoint(s) is a prefix of this prompt";
            }
            return lcp;
        }

        /// <summary>
        /// Where shared-prefix checkpoints outlive the process, or null for nowhere.
        /// Set by the host; read on the engine thread at admission (restore) and at
        /// the prefix boundary (save). See <see cref="IPrefixCheckpointStore"/>.
        /// </summary>
        public IPrefixCheckpointStore PrefixCheckpointStore
        {
            get => Volatile.Read(ref _checkpointStore);
            set => Volatile.Write(ref _checkpointStore, value);
        }

        private IPrefixCheckpointStore _checkpointStore;

        // Prefix hashes the store answered with bytes this model rejected, or whose
        // restored copy could not be cloned: not asked for again in this process, or
        // every new chat would stream the same hundreds of megabytes back in only to
        // throw them away. The next successful checkpoint of that prefix overwrites the
        // file anyway.
        private readonly HashSet<long> _storeLookupsToSkip = new();

        private static long PrefixHash(SequenceState seq, int n)
        {
            unchecked
            {
                long h = 1469598103934665603L ^ n;
                for (int i = 0; i < n; i++)
                    h = (h ^ seq.PromptTokens[i]) * 1099511628211L;
                return h;
            }
        }

        /// <summary>What a saved checkpoint is filed under: the model's own K/V
        /// identity plus the file format the executor speaks.</summary>
        private string CheckpointModelFingerprint =>
            (_model.KVStateFingerprint ?? string.Empty) + "|prefix-checkpoint-v1";

        private bool HasPrefixCheckpointFor(SequenceState seq, int n)
        {
            foreach (var existing in _prefixCheckpoints)
            {
                if (existing.Tokens.Length != n) continue;
                bool same = true;
                for (int i = 0; i < n && same; i++)
                    same = existing.Tokens[i] == seq.PromptTokens[i];
                if (same) return true;
            }
            return false;
        }

        private void EvictPrefixCheckpointsBeyondBudget(IBatchedPagedModel fused)
        {
            int budget = Math.Max(1, ExecutionOptions.FromEnvironment().PrefixCheckpointBudget);
            while (_prefixCheckpoints.Count > budget)
            {
                var victim = _prefixCheckpoints.First.Value;
                fused.DiscardRetainedCache(victim.RequestId);
                _prefixCheckpoints.RemoveFirst();
            }
        }

        /// <summary>
        /// A prompt about to be admitted starts with a shared prefix no checkpoint in
        /// memory covers: if a store has one from an earlier process, read it in now,
        /// so admission finds it exactly as it would find one taken minutes ago. On the
        /// engine thread, once per prefix per process; the read is the checkpoint's
        /// size (a hundred to a few hundred megabytes) from local storage.
        /// </summary>
        private void TryRestorePrefixCheckpointFromStore(SequenceState seq)
        {
            var store = PrefixCheckpointStore;
            if (store == null || seq.SharedPrefixTokens <= 0 || seq.PrefixCheckpointTaken || seq.CacheBreakpoints != null)
                return;
            if (!ModelSupportsPrefixCheckpoints() || _model is not IBatchedPagedModel fused
                || !fused.SupportsRetainedCacheSerialization)
                return;
            int n = seq.SharedPrefixTokens;
            if (n > seq.PromptTokens.Count || HasPrefixCheckpointFor(seq, n))
                return;
            long prefixHash = PrefixHash(seq, n);
            if (_storeLookupsToSkip.Contains(prefixHash))
                return;

            System.IO.Stream payload = null;
            var sw = Stopwatch.StartNew();
            try
            {
                var tokens = AllocateRetainedCacheTokens(n);
                for (int i = 0; i < n; i++) tokens[i] = seq.PromptTokens[i];
                var prepared = new LinkedListNode<RetainedFusedCache>(new RetainedFusedCache
                {
                    RequestId = $"{_checkpointKeyPrefix}{++_prefixCheckpointSerial}",
                    Tokens = tokens,
                    IsPrefixCheckpoint = true,
                });
                _storeLookupsToSkip.EnsureCapacity(checked(_storeLookupsToSkip.Count + 1));
                if (!store.TryOpen(CheckpointModelFingerprint, tokens, out payload) || payload == null)
                    return;
                long bytes = payload.CanSeek ? payload.Length - payload.Position : -1;
                string key = prepared.Value.RequestId;
                if (!fused.TryImportRetainedCache(key, payload))
                {
                    _storeLookupsToSkip.Add(prefixHash);
                    _logger.LogWarning(
                        "A saved shared-prefix checkpoint for {Tokens} tokens does not fit this model and was ignored; the prefix is prefilled and saved again.",
                        n);
                    return;
                }
                _prefixCheckpoints.AddLast(prepared);
                EvictPrefixCheckpointsBeyondBudget(fused);
                _logger.LogInformation(
                    "Shared-prefix checkpoint {Key} restored from disk for {RequestId}: {Tokens} tokens, {MB:F0} MB in {Ms:F0} ms; " +
                    "this and every later chat that starts with this prefix continue from a copy of it.",
                    key, seq.RequestId, n, bytes / 1048576.0, sw.Elapsed.TotalMilliseconds);
            }
            catch (OutOfMemoryException)
            {
                // No allocating log/skip-set update while handling memory pressure.
                // Any completed import was already published through its ready node.
            }
            catch (Exception ex)
            {
                _storeLookupsToSkip.Add(prefixHash);
                _logger.LogWarning(ex,
                    "Restoring a shared-prefix checkpoint from disk failed; the prefix is prefilled and saved again.");
            }
            finally
            {
                payload?.Dispose();
            }
        }

        /// <summary>
        /// A checkpoint was just taken in memory: keep it for the next process too.
        /// The model writes its bytes straight into the store's stream on this thread
        /// (a sequential local write of the checkpoint's size), so no second copy of
        /// the state is ever held in memory.
        /// </summary>
        private void SavePrefixCheckpointToStore(IBatchedPagedModel fused, string key, int[] tokens)
        {
            var store = PrefixCheckpointStore;
            if (store == null || !fused.SupportsRetainedCacheSerialization)
                return;
            var sw = Stopwatch.StartNew();
            long written = 0;
            try
            {
                bool saved = store.Save(CheckpointModelFingerprint, tokens, stream =>
                {
                    long start = stream.CanSeek ? stream.Position : 0;
                    if (!fused.TryExportRetainedCache(key, stream))
                        throw new InvalidOperationException("the model declined to export the checkpoint");
                    written = stream.CanSeek ? stream.Position - start : 0;
                });
                if (saved)
                    _logger.LogInformation(
                        "Shared-prefix checkpoint {Key} saved to disk: {MB:F0} MB in {Ms:F0} ms; the next launch starts from it.",
                        key, written / 1048576.0, sw.Elapsed.TotalMilliseconds);
                else
                    _logger.LogWarning(
                        "Shared-prefix checkpoint {Key} could not be saved; the next launch prefills the prefix again.", key);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Saving shared-prefix checkpoint {Key} failed; the next launch prefills the prefix again.", key);
            }
        }

        /// <summary>Adopt a retained fused holder for <paramref name="seq"/>: re-key
        /// the model's retained continuation state to this request (so its first fused
        /// <c>BindSequenceCache</c> continues from it), reserve placeholder blocks for
        /// accounting, and mark the reused prefix. Returns false (caller falls back to
        /// the pooled path) when the holder can't be reserved or re-keyed. Invoked by
        /// the scheduler at admission.</summary>
        public bool TryAdoptFusedContinuation(SequenceState seq, int lcp)
        {
            if (seq == null || lcp <= 0) return false;
            if (seq.BlockTable.NumBlocks != 0) return false;
            if (_model is not IBatchedPagedModel fused) return false;

            var match = FindRetainedFusedMatch(seq, out int matchedLcp);
            if (match == null || matchedLcp != lcp) return false;

            // Resolve the existing LRU node and reserve every mutable metadata
            // container before the model rekeys/clones a native holder. Neither
            // list publication nor a pending rewind may allocate after transfer.
            var matchedNode = match.IsPrefixCheckpoint
                ? _prefixCheckpoints.Find(match) : _retainedFused.Find(match);
            if (matchedNode == null) return false;
            int neededBlocks = (int)(((long)lcp + _blockSize - 1) / _blockSize);
            try { ReserveRetainedAdoptionMetadata(seq, neededBlocks); }
            catch (OutOfMemoryException) { return false; }
            var blocks = _pool.AllocateNew(neededBlocks);
            if (blocks == null)
            {
                // Pool pressure -> let the caller use the capped pool path. Said out
                // loud: from the request's side this is a full re-prefill with no cause.
                var stats = _pool.GetStats();
                _logger.LogInformation(
                    "Retained state {Key} matches {RequestId} for {Lcp} tokens but the block pool cannot back it "
                    + "({Needed} blocks needed, {Free} of {Total} free); re-prefilling instead.",
                    match.RequestId, seq.RequestId, lcp, neededBlocks, stats.freeBlocks, stats.totalBlocks);
                return false;
            }

            bool bound;
            try
            {
                bound = match.IsPrefixCheckpoint
                    // A checkpoint is copied, never moved: the next new chat needs it too.
                    ? fused.TryCloneRetainedCache(match.RequestId, seq.RequestId)
                    : fused.TryRebindRetainedCache(match.RequestId, seq.RequestId);
            }
            catch (Exception ex)
            {
                // A clone is a whole-cache allocation; on a phone near its memory limit
                // it can fail outright. Treated exactly like a refusal: blocks back,
                // checkpoint gone, this request re-prefills — never a leak of reserved
                // blocks and never the same throw on every scheduler pass.
                _logger.LogWarning(ex,
                    "Adopting retained state {Key} for {RequestId} failed; discarding it and re-prefilling.",
                    match.RequestId, seq.RequestId);
                bound = false;
            }
            if (!bound)
            {
                _pool.Free(blocks);
                if (match.IsPrefixCheckpoint)
                {
                    // Declined once, declined always: drop it so the next request does
                    // not pay the search for a copy the model will not make -- and not
                    // read it back from the store either, which would be the same copy.
                    _logger.LogWarning(
                        "Shared-prefix checkpoint {Key} could not be cloned for {RequestId}; discarding it.",
                        match.RequestId, seq.RequestId);
                    fused.DiscardRetainedCache(match.RequestId);
                    _prefixCheckpoints.Remove(matchedNode);
                    _storeLookupsToSkip.Add(PrefixHash(seq, Math.Min(match.Tokens.Length, seq.PromptTokens.Count)));
                }
                return false;
            }

            for (int i = 0; i < blocks.Length; i++)
                seq.BlockTable.AppendBlock(blocks[i]);

            seq.SetComputedTokensForPrefixAdoption(lcp);
            seq.PrefixCacheReusedTokens = lcp;
            if (match.IsPrefixCheckpoint)
            {
                // Its own prefix is now covered; nothing to checkpoint again, and the
                // copy the request holds is exactly the state after those tokens.
                seq.PrefixCheckpointTaken = true;
                _prefixCheckpoints.Remove(matchedNode);
                _prefixCheckpoints.AddLast(matchedNode);   // most recently used; no new node allocation
                _logger.LogInformation(
                    "Shared-prefix checkpoint {Key} cloned for {RequestId}: {Tokens} prompt tokens continue from the copy.",
                    match.RequestId, seq.RequestId, lcp);
                return true;
            }
            if (lcp < match.Tokens.Length)
                _pendingRetainedFusedTruncations[seq.RequestId] = lcp;
            // The rebound holder is now this request's active fused cache; the
            // fused path's BindSequenceCache finds it (fresh==false) and continues
            // from it without injecting from the (empty) reserved blocks.
            _retainedFused.Remove(matchedNode);
            return true;
        }

        /// <summary>True when a retained holder or shared-prefix checkpoint covers at
        /// least <paramref name="minimum"/> tokens of this prompt as an exact prefix,
        /// with nothing to rewind.</summary>
        private bool HasExactRetainedContinuation(SequenceState seq, int minimum)
        {
            if (!ModelUsesRetainableFusedCache() || seq.CacheBreakpoints != null)
                return false;
            foreach (var entry in RetainedCandidates())
            {
                if (!entry.IsPrefixCheckpoint
                    && !string.Equals(entry.MediaFingerprint, seq.MediaFingerprint, StringComparison.Ordinal))
                    continue;
                int len = entry.Tokens.Length;
                if (len < minimum || len >= seq.PromptTokens.Count)
                    continue;
                bool exact = true;
                for (int i = 0; i < len && exact; i++)
                    exact = seq.PromptTokens[i] == entry.Tokens[i];
                if (exact)
                    return true;
            }
            return false;
        }

        private IEnumerable<RetainedFusedCache> RetainedCandidates()
        {
            foreach (var entry in _retainedFused) yield return entry;
            foreach (var entry in _prefixCheckpoints) yield return entry;
        }

        /// <summary>Find the retained fused holder whose token run is a prefix of
        /// <paramref name="seq"/>'s prompt, allowing the same one-or-two-token trailing
        /// control-token rewind as live-cache continuation, and leaving at least one
        /// new suffix token to forward. Prefers the longest reusable prefix.</summary>
        private RetainedFusedCache FindRetainedFusedMatch(SequenceState seq, out int reusableLength)
        {
            reusableLength = 0;
            // Explicit policies deliberately use the block-granular pooled path:
            // rebinding a whole request-owned holder must not bypass either the
            // source or target request's client-selected cache boundary.
            if (seq.CacheBreakpoints != null)
                return null;

            RetainedFusedCache bestExact = null, bestRewound = null;
            int exactLcp = 0, rewoundLcp = 0;
            bool circular = _model.MaxReusablePrefixTokens != int.MaxValue;
            foreach (var entry in RetainedCandidates())
            {
                // A checkpoint's tokens are text-only by construction; a retained
                // conversation holder may hold media and must match on it.
                if (!entry.IsPrefixCheckpoint
                    && !string.Equals(entry.MediaFingerprint, seq.MediaFingerprint, StringComparison.Ordinal))
                    continue;
                int len = entry.Tokens.Length;
                // NB: no `len <= cap` skip. The fused path writes nothing to the shared
                // pool, so a retained holder is the only reuse source for a concurrent
                // conversation even when the paged path could otherwise represent the
                // prefix length.
                int lcp = 0;
                int limit = Math.Min(len, seq.PromptTokens.Count);
                while (lcp < limit && seq.PromptTokens[lcp] == entry.Tokens[lcp])
                    lcp++;

                if (seq.PromptTokens.Count <= lcp) continue;   // no new suffix to forward
                var exactReuse = _model as IExactFusedCacheReuse;
                if (exactReuse?.SupportsExactFusedCacheReuse != true) exactReuse = null;
                if (exactReuse != null && !entry.IsPrefixCheckpoint && lcp < len)
                {
                    int align = Math.Max(1, _model.KVCacheTruncationGranularity);
                    lcp -= lcp % align;
                    if (lcp <= 0) continue;
                }
                int rewind = len - lcp;
                // A checkpoint is adopted whole or not at all: the prompt must extend
                // exactly the tokens it was taken after.
                if (entry.IsPrefixCheckpoint && rewind > 0)
                    continue;
                if (exactReuse != null && !entry.IsPrefixCheckpoint)
                {
                    if (!exactReuse.CanReuseRetainedPrefix(entry.RequestId, len, lcp)) continue;
                }
                else if (rewind > 0
                    && (rewind > MaxLiveContinuationRewindTokens
                        || !_model.CanTruncateKVCache(len, lcp)))
                    continue;
                if (rewind == 0)
                {
                    if (lcp > exactLcp) { bestExact = entry; exactLcp = lcp; }
                }
                else if (lcp > rewoundLcp)
                {
                    bestRewound = entry; rewoundLcp = lcp;
                }
            }

            // Among eligible candidates, prefer an exact nearby checkpoint on
            // circular caches, matching the live path. A conversation holder
            // with a permitted rewind can still beat a much shorter system
            // checkpoint. Elsewhere the longest match wins.
            bool preferExact = bestExact != null
                && (bestRewound == null
                    || (circular ? exactLcp >= rewoundLcp - MaxLiveContinuationRewindTokens
                                 : exactLcp >= rewoundLcp));
            reusableLength = preferExact ? exactLcp : rewoundLcp;
            return preferExact ? bestExact : bestRewound;
        }

        /// <summary>Track an in-flight fused sequence so the release hook can snapshot
        /// its tokens. Called from the fused per-seq path for every scheduled seq.</summary>
        private void NoteFusedSequence(SequenceState seq)
        {
            if (ModelUsesRetainableFusedCache())
                _fusedSeqById[seq.RequestId] = seq;
        }

        /// <summary>Forget executor-owned state for a fused request that is being
        /// released without retention (abort/cancellation). The model's release
        /// hook still owns the holder itself; this only prevents an abandoned
        /// sequence or a not-yet-applied retained-holder truncation from remaining
        /// reachable inside the executor.</summary>
        internal void DiscardReleasedFusedCacheBookkeeping(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            _pendingRetainedFusedTruncations.Remove(requestId);
            _fusedSeqById.Remove(requestId);
            DisposeFusedSpecContext(requestId);
        }

        /// <summary>Remove stale retained metadata (and its model holder) before a
        /// new finished request reuses the same public RequestId. The scheduler only
        /// requires ids to be unique while in flight, so sequential id reuse must not
        /// leave two token snapshots pointing at the model's single id-keyed holder.</summary>
        private void DiscardRetainedFusedCacheWithRequestId(
            IBatchedPagedModel fused,
            string requestId)
        {
            bool found = false;
            var node = _retainedFused.First;
            while (node != null)
            {
                var next = node.Next;
                if (string.Equals(node.Value.RequestId, requestId, StringComparison.Ordinal))
                {
                    found = true;
                    break;
                }
                node = next;
            }

            if (!found) return;
            // A failed native disposal still owns storage. Keep its metadata so
            // cleanup can be retried instead of orphaning an untracked holder.
            fused.DiscardRetainedCache(requestId);
            node = _retainedFused.First;
            while (node != null)
            {
                var next = node.Next;
                if (string.Equals(node.Value.RequestId, requestId, StringComparison.Ordinal))
                    _retainedFused.Remove(node);
                node = next;
            }
        }

        /// <summary>Called by the engine when a sequence leaves the scheduler, BEFORE
        /// the model's <see cref="IBatchedPagedModel.OnSequenceReleased"/>. When the
        /// sequence finished cleanly on the fused path, retain its complete
        /// request-owned continuation holder for cross-request prefix reuse instead
        /// of letting the model dispose it. This is circular K/V for Gemma 4 and
        /// attention K/V plus GDN recurrent state for Qwen 3.5/3.6. Returns true when
        /// the holder was retained (so the subsequent model release no-ops for it).</summary>
        public bool TryRetainReleasedFusedCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return false;
            DisposeFusedSpecContext(requestId);
            _pendingRetainedFusedTruncations.Remove(requestId);
            if (!_fusedSeqById.TryGetValue(requestId, out var seq))
                return false;
            _fusedSeqById.Remove(requestId);

            if (!ModelUsesRetainableFusedCache()) return false;
            if (_model is not IBatchedPagedModel fused) return false;
            // Clean finishes, and clean STOPS. An abort is processed between steps,
            // so a stopped sequence's holder is consistent at its last forwarded
            // token — and the chat layer now records exactly those tokens, so the
            // next turn of that conversation extends the holder precisely. Errored
            // sequences may hold partial state and preempted ones resume on their own.
            bool cleanStop = seq.Status == SequenceStatus.FinishedAborted
                && seq.Error == null
                && seq.NumComputedTokens >= seq.NumTotalTokens;
            if (seq.Status != SequenceStatus.FinishedStopped
                && seq.Status != SequenceStatus.FinishedLengthCapped
                && !cleanStop)
                return false;

            // A retained holder represents the model's complete fused state. Even
            // models that can rewind a short generated control-token tail must not let
            // holder adoption bypass an explicit request cache boundary.
            if (seq.CacheBreakpoints != null)
                return false;

            // Snapshot exactly the tokens whose model state is resident in the holder
            // (NumComputedTokens == the model's _cacheSeqLen at finish), so a later
            // continuation's reused-prefix length matches the holder's cache extent
            // exactly. (At a clean finish this equals NumTotalTokens; clamp defends
            // against a speculative tail that advanced computed past the token list.)
            int len = Math.Min(seq.NumComputedTokens, seq.NumTotalTokens);
            // Retain any non-trivial fused conversation: the fused path contributes
            // nothing to the shared pool, so retention is the only cross-request reuse
            // source for it (not just the >window case). The LRU budget bounds VRAM.
            if (len < _blockSize) return false;

            // Request ids are unique only while in flight. If a caller reuses one
            // sequentially, discard the older retained holder before the model's
            // id-keyed retained dictionary is populated with this new cache. Keeping
            // both metadata entries would let an old token match rebind the new,
            // unrelated holder.
            if (!fused.HasFusedSequenceCache(requestId)) return false;
            LinkedListNode<RetainedFusedCache> prepared;
            int budget;
            try
            {
                // Nothing may allocate between successful native transfer and
                // publication in the LRU. Prepare the record AND list node now.
                budget = ExecutionOptions.FromEnvironment().RetainedFusedCacheBudget;
                var tokens = AllocateRetainedCacheTokens(len);
                for (int i = 0; i < len; i++) tokens[i] = seq.TokenAt(i);
                prepared = new LinkedListNode<RetainedFusedCache>(new RetainedFusedCache
                {
                    RequestId = requestId,
                    Tokens = tokens,
                    MediaFingerprint = seq.MediaFingerprint,
                });
            }
            catch (OutOfMemoryException)
            {
                // The request still owns its native holder; ordinary release is
                // responsible for freeing it. Avoid allocating a diagnostic here.
                return false;
            }
            DiscardRetainedFusedCacheWithRequestId(fused, requestId);
            if (!fused.RetainSequenceCache(requestId)) return false;
            _retainedFused.AddLast(prepared);

            // Evict oldest holders beyond the budget (frees their VRAM).
            while (_retainedFused.Count > budget)
            {
                var victim = _retainedFused.First.Value;
                fused.DiscardRetainedCache(victim.RequestId);
                _retainedFused.RemoveFirst();
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
            _specCtx = null;

            // A step that forwarded the outgoing owner alone let it BORROW the
            // model's logits buffer (see ExecuteStepPerSequence), on the promise
            // that it samples before the next Forward. The incoming sequence's
            // Forward breaks that promise: the owner would later sample its next
            // token from another request's logits. Seen as the first request of
            // a concurrent burst answering with nothing, or with another prompt's
            // content. Detach it here, once per ownership change.
            DetachBorrowedLogits(_currentOwner);

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
                    && _liveCacheSeq?.CacheBreakpoints == null
                    && seq.CacheBreakpoints == null
                    && string.Equals(
                        _liveCacheSeq?.MediaFingerprint,
                        seq.MediaFingerprint,
                        StringComparison.Ordinal)
                    && seq.NumComputedTokens > 0
                    && _liveCacheLen >= seq.NumComputedTokens
                    && (_liveCacheLen == seq.NumComputedTokens || _model.SupportsKVCacheTruncation))
                {
                    // The scheduler may have matched a prefix shorter than what the cache
                    // holds, because the previous turn ended on a control token the
                    // template does not re-render (see MaxLiveContinuationRewindTokens).
                    // Drop those trailing positions so the model's cache and this sequence
                    // agree on where the next token goes - the same rewind speculative
                    // decoding performs when a draft is rejected.
                    //
                    // TryTruncate, not Truncate: a model whose rewind depth depends on
                    // where the sequence is (DeepSeek V4.1's sliding-window ring) can
                    // decline this one, and claiming the live prefix anyway would decode
                    // the rest of the turn against positions the cache still holds.
                    bool headAgrees = _liveCacheLen == seq.NumComputedTokens;
                    if (!headAgrees)
                    {
                        headAgrees = _model.TryTruncateKVCache(seq.NumComputedTokens);
                        if (headAgrees)
                        {
                            _liveCacheLen = seq.NumComputedTokens;
                        }
                        else
                        {
                            _logger.LogInformation(
                                "Live-cache continuation for {RequestId} needed a {Rewind}-token rewind the " +
                                "model declined at execution time; its prompt re-prefills after all.",
                                seq.RequestId, _liveCacheLen - seq.NumComputedTokens);
                        }
                    }

                    if (headAgrees)
                    {
                        _liveCacheSeq = seq;
                        _currentOwner = seq;
                        _ownerTokensInModel = seq.NumComputedTokens;
                        _ownerForwardedTokens = 0;
                        return;
                    }

                    seq.ClearLiveCacheContinuation();
                }
                else
                {
                    // The live cache was invalidated between scheduling and execution
                    // (e.g. a concurrent sequence took ownership). Drop the reused-prefix
                    // claim and re-prefill from scratch via the normal path below; the
                    // sequence keeps its reserved blocks so accounting stays consistent.
                    // Information, not Debug: admission has already reported this request
                    // as reusing the live prefix, and this retracts that. A turn whose
                    // announced reuse silently became a full re-prefill is exactly the
                    // latency mystery this whole logging path exists to prevent.
                    _logger.LogInformation(
                        "Live-cache continuation for {RequestId} was no longer valid at execution time " +
                        "(another sequence took the cache); its prompt re-prefills after all.",
                        seq.RequestId);
                    seq.ClearLiveCacheContinuation();
                }
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
            if (!seq.PrefillReservationTaken)
            {
                // Reserve the prompt plus its declared generation budget in a
                // single model-specific allocation. Hybrid models with large
                // attention caches can otherwise cross a power-of-two boundary
                // during decode and perform a multi-gigabyte grow/copy mid-stream.
                //
                // Once per request, on its FIRST chunk -- which is not always at token
                // zero. A request that continues a retained holder or the live cache
                // starts at the reused prefix, and reserving only at zero left every
                // such continuation to grow geometrically inside the forward instead
                // (2048 -> 4096 -> ... -> the whole window), each doubling a copy of
                // the cache and a rebuild of the graphs bound to it.
                seq.PrefillReservationTaken = true;
                _model.PrepareForPrefill(ResolvePrefillReservation(
                    seq.PromptTokens.Count,
                    seq.MaxNewTokens,
                    _model.MaxContextLength,
                    ExecutionOptions.FromEnvironment().KvGenerationReserveMax));
            }

            int want = work.NumScheduledTokens;
            int[] buf = new int[want];
            for (int i = 0; i < want; i++)
                buf[i] = seq.TokenAt(seq.NumComputedTokens + i);
            return buf;
        }

        /// <summary>How many K/V rows a request reserves before its first prefill
        /// chunk: the prompt plus its generation budget, the latter capped by
        /// <see cref="ExecutionOptions.KvGenerationReserveMax"/> when one is set, and
        /// the whole never past the window. The cap exists because a reply limit at or
        /// above the window (a phone user asking for the longest reply the app offers)
        /// otherwise reserves the ENTIRE window for every request, however short the
        /// conversation -- and on a device where the K/V is paid twice, in host memory
        /// and in its device mirror, that reservation is what got the app killed. Past
        /// the cap the cache still grows on demand while the reply runs.</summary>
        internal static int ResolvePrefillReservation(
            int promptTokens, int maxNewTokens, int maxContext, int generationReserveMax)
        {
            long generation = Math.Max(0, maxNewTokens);
            long requested;
            if (generationReserveMax > 0)
            {
                generation = Math.Min(generation, generationReserveMax);
                // Capped mode is memory-bound mode, where a conversation's every tool
                // round arrives as a continuation a thousand tokens longer than the last.
                // Reserved to the token, each round would grow the cache -- a copy of
                // every resident row, a rebuild of the graphs bound to it, a re-upload of
                // the device mirror -- so the reservation is snapped up to a coarse step
                // and a growth is paid once per step instead. The slack is bounded (one
                // step of K/V) where geometric doubling was unbounded (half the window).
                requested = Math.Max(0, promptTokens) + generation;
                requested = (requested + CappedReservationStep - 1) / CappedReservationStep * CappedReservationStep;
            }
            else
            {
                requested = Math.Max(0, promptTokens) + generation;
            }
            if (maxContext > 0)
                requested = Math.Min(requested, maxContext);
            return (int)Math.Min(requested, int.MaxValue);
        }

        /// <summary>Granularity of a capped reservation; see <see cref="ResolvePrefillReservation"/>.</summary>
        internal const int CappedReservationStep = 2048;

        /// <summary>
        /// Give memory back while it is still ours to give. Keeps the newest retained
        /// conversation holder (the turn most likely to continue) and every shared-prefix
        /// checkpoint (small, and what makes a new chat fast); evicts every other retained
        /// holder, then lets the model drop whatever it parked for reuse. Engine thread
        /// only, between steps. Returns a one-line account for the log.
        /// </summary>
        public string TrimIdleMemory()
        {
            int evicted = 0;
            if (_model is IBatchedPagedModel fused)
            {
                while (_retainedFused.Count > 1)
                {
                    var victim = _retainedFused.First.Value;
                    fused.DiscardRetainedCache(victim.RequestId);
                    _retainedFused.RemoveFirst();
                    evicted++;
                }
            }
            _model.TrimIdleMemory();
            return $"evicted {evicted} retained holder(s); kept {_retainedFused.Count} retained and "
                + $"{_prefixCheckpoints.Count} shared-prefix checkpoint(s)";
        }

        /// <summary>Consume a token the batched greedy path sampled on-device
        /// last step (bit-equivalent to re-sampling the logits it summarizes),
        /// falling back to host sampling from LastLogits. Any position drift —
        /// preemption, recompute, rollback — invalidates the stash.</summary>
        private static int TakePendingOrSample(SequenceState seq)
        {
            if (seq.GetOrCreateSampler().TryGetForcedThinkingToken(seq.OutputTokens, out int thinkingEnd))
            {
                seq.PendingDeviceToken = null;
                return thinkingEnd;
            }
            if (seq.PendingDeviceToken.HasValue)
            {
                int t = seq.PendingDeviceToken.Value;
                seq.PendingDeviceToken = null;
                if (seq.PendingDevicePosition == seq.NumComputedTokens)
                    return t;
            }
            return SampleFromLogits(seq);
        }

        /// <summary>Non-destructive form of <see cref="TakePendingOrSample"/> for
        /// the batched fast path's peek: a stash that is still valid stays on the
        /// sequence until a successful step replaces (or clears) it, so a decline
        /// leaves the fallback loop a token source.</summary>
        private static int PeekPendingOrSample(SequenceState seq)
        {
            if (seq.GetOrCreateSampler().TryGetForcedThinkingToken(seq.OutputTokens, out int thinkingEnd))
                return thinkingEnd;
            if (seq.PendingDeviceToken.HasValue)
            {
                if (seq.PendingDevicePosition == seq.NumComputedTokens)
                    return seq.PendingDeviceToken.Value;
                seq.PendingDeviceToken = null;   // stale: position moved under it
            }
            return SampleFromLogits(seq);
        }

        private static int SampleFromLogits(SequenceState seq)
        {
            if (seq.LastLogits == null)
                throw new InvalidOperationException(
                    $"Sequence {seq.RequestId} has no LastLogits to sample from at position {seq.NumComputedTokens}.");
            var sampler = seq.GetOrCreateSampler();
            return sampler.Sample(seq.LastLogits, seq.OutputTokens);
        }

        /// <summary>Extract all blocks for the current owner into PagedKvStorage.
        /// Called when swapping out.</summary>
        private void ExtractAllBlocks(SequenceState seq, int tokensInModel)
        {
            if (!_model.SupportsKVStateSnapshot || !_model.SupportsCrossSequenceKvReuse) return;
            if (tokensInModel <= 0) return;

            int blocks = Math.Min(
                seq.BlockTable.NumBlocks,
                CapturableBlocks((tokensInModel + _blockSize - 1) / _blockSize));
            for (int b = 0; b < blocks; b++)
            {
                int startToken = b * _blockSize;
                if (startToken >= tokensInModel) break;
                int tokensInBlock = Math.Min(_blockSize, tokensInModel - startToken);
                var block = seq.BlockTable.Blocks[b];

                // A recurrent full block was captured at the exact Forward
                // boundary where it first became available. Re-extracting it on
                // a later owner swap would overwrite that checkpoint with the
                // owner's current (later) recurrent state. The trailing partial
                // block is still refreshed because its endpoint is current.
                if (_model.RequiresPerBlockCapture && block.Used == tokensInBlock)
                    continue;

                long expectedBytes = _model.ComputeKVBlockByteSize(tokensInBlock);
                if (expectedBytes <= 0) break;

                EnsureScratch((int)expectedBytes);
                var dst = _scratch.AsSpan(0, (int)expectedBytes);
                if (!_model.TryExtractKVBlock(startToken, tokensInBlock, dst))
                {
                    // For SWA-bounded models (e.g. Gemma 4) blocks whose positions
                    // have aged out of the sliding window can't be re-extracted ÔÇö
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
                block.HoldsSnapshotBytes = tokensInBlock == _blockSize;
            }
        }

        /// <summary>Inject all blocks for <paramref name="seq"/> into the model's
        /// fresh KV state. Called when swapping in.</summary>
        private void InjectAllBlocks(SequenceState seq, int tokensToInject)
        {
            if (!_model.SupportsKVStateSnapshot || !_model.SupportsCrossSequenceKvReuse) return;
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
            if (!_model.SupportsKVStateSnapshot
                || !_model.SupportsCrossSequenceKvReuse
                || !_scheduler.PrefixCachingEnabled)
                return 0;

            int fullBlocksNow = seq.NumComputedTokens / _blockSize;
            int captured = 0;
            int previouslyFull = fullBlocksNow;
            for (int b = 0; b < CapturableBlocks(fullBlocksNow) && b < seq.BlockTable.NumBlocks; b++)
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
                block.HoldsSnapshotBytes = true;
                block.IsRestorablePrefixEnd = !_model.RequiresPerBlockCapture
                    || (b == fullBlocksNow - 1 && seq.NumComputedTokens % _blockSize == 0);
                captured++;
                if (b < previouslyFull) previouslyFull = b;
            }

            // Let the scheduler index the newly-full blocks (hash registration).
            if (captured > 0)
                _scheduler.OnBlocksCommitted(seq, previouslyFull * _blockSize);

            return captured;
        }

        /// <summary>
        /// Blocks worth snapshotting out of <paramref name="fullBlocks"/>. A model
        /// with a circular / ring KV cache (Gemma 4, Muse-Glimmer) caps how long a
        /// pooled prefix it can faithfully restore, and <see cref="EnsureOwnership"/>
        /// discards any snapshot longer than that cap rather than injecting it. So
        /// blocks past the cap are captured, stored and never used - and on a
        /// device-resident cache each capture round drags the whole K/V back over
        /// PCIe first. Reuse beyond the cap comes from live-cache / retained-fused
        /// continuation, neither of which reads the pool.
        /// </summary>
        private int CapturableBlocks(int fullBlocks)
        {
            int cap = _model.MaxReusablePrefixTokens;
            if (cap == int.MaxValue)
                return fullBlocks;
            return Math.Min(fullBlocks, cap / _blockSize);
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
            _pendingRetainedFusedTruncations.Clear();
            _specCtx = null;
            if (_model is IBatchedPagedModel fused)
            {
                foreach (var entry in _retainedFused)
                    fused.DiscardRetainedCache(entry.RequestId);
                foreach (var entry in _prefixCheckpoints)
                    fused.DiscardRetainedCache(entry.RequestId);
            }
            _retainedFused.Clear();
            _prefixCheckpoints.Clear();
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

    /// <summary>Opt-in contract for true batched paged attention: models with
    /// a native paged forward kernel taking a batch metadata struct (Gemma 4,
    /// Qwen 3.5, Nemotron-H, GptOss, Mistral 3). Path selection against this
    /// contract is centralised in <see cref="ExecutionPlanner"/>, which reads
    /// the declared capability getters below (via
    /// <see cref="ExecutionCapabilities.FromModel"/>) instead of probing
    /// behaviour through exceptions.</summary>
    public interface IBatchedPagedModel
    {
        /// <summary>Drive a single batched forward pass given the scheduler's
        /// per-step metadata. Returns per-sequence logits.</summary>
        IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx);

        /// <summary>Master availability switch for this model's batched path.
        /// False when a per-model opt-out (e.g. <c>TS_QWEN35_BATCHED=0</c>,
        /// <c>TS_GPTOSS_BATCHED=0</c>) or a static limitation (e.g. Gemma 4
        /// with MoE layers or a block-quantized KV cache) makes
        /// <see cref="ForwardBatch"/> unusable, so <see cref="ExecutionPlanner"/>
        /// routes around the batched path up front instead of relying on a
        /// NotSupportedException fallback. <c>ForwardBatch</c> may still throw
        /// NotSupportedException for a specific batch it cannot serve (the
        /// executor treats that as a decline); this getter covers the
        /// model-static part of that decision. Default true.</summary>
        bool BatchedForwardAvailable => true;

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
        /// executor must not use the N=1 fast path for this model ÔÇö a
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

        /// <summary>Transition the current single-stream (N==1) owner ÔÇö whose
        /// live K/V is in the model's primary cache ÔÇö into a per-request holder
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
        /// path before and must stay on it ÔÇö its tail K/V isn't reconstructable
        /// from paged storage).</summary>
        bool HasFusedSequenceCache(string requestId) => false;

        // ---- Retained fused-cache continuation (cross-request prefix reuse) ----
        //
        // The per-sequence fused path keeps each concurrent request's complete
        // continuation state in its own holder and never writes the shared paged
        // block storage, so it contributes nothing to the prefix-cache pool. A
        // circular-cache model cannot reconstruct a long prefix from that pool, and
        // a hybrid attention+recurrent model cannot reconstruct its GDN state from
        // attention K/V alone. Without retention, a multi-turn follow-up that arrives
        // while/after other requests ran concurrently can therefore re-prefill the
        // whole conversation (KV-reuse ratio 0). The executor instead RETAINS a
        // finished fused holder and re-adopts it for a later request whose prompt
        // exactly extends the retained tokens ÔÇö the cross-request analogue of the
        // single-stream live-cache continuation. The model keeps the complete holder
        // alive and lets it be re-keyed.

        /// <summary>Whether this model implements the retained-holder lifecycle
        /// below. This is distinct from <see cref="IModelArchitecture.MaxReusablePrefixTokens"/>,
        /// which describes byte snapshots in the shared paged pool rather than a
        /// complete request-owned fused cache. Default false.</summary>
        bool SupportsRetainedFusedCache => false;

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
        /// so the new request continues from the retained model state with no
        /// re-prefill).
        /// Returns false when no retained holder exists for the id. Default false.</summary>
        bool TryRebindRetainedCache(string retainedRequestId, string newRequestId) => false;

        /// <summary>Dispose a retained holder (LRU eviction / shutdown) and free its
        /// buffers and any recurrent state. Default no-op.</summary>
        void DiscardRetainedCache(string requestId) { }

        // ---- Shared-prefix checkpoints (the prompt every conversation begins with) ----
        //
        // A retained holder is consumed by the one request that continues it, and it
        // holds one whole conversation. Neither fits the prefix every chat on a host
        // shares - the system prompt, tool schemas and skill descriptions, thousands
        // of tokens that never change between chats. A NEW chat extends that prefix
        // and nothing else, so on a sliding-window model (Gemma 4) or a hybrid
        // recurrent one (Qwen 3.5) it re-prefilled all of it: the live cache diverged
        // too far back to rewind, no retained holder matched, and the pool cannot
        // restore a circular or recurrent state. A checkpoint is the model's complete
        // state copied at exactly that boundary, kept apart from the LRU, and CLONED
        // into each request that starts from it - the checkpoint itself is never
        // consumed. The executor decides where the boundary is (the chat layer marks
        // it on the request) and when to take one; the model only copies.
        bool SupportsPrefixCheckpoints => false;
        /// <summary>Deep-copy the ACTIVE cache (primary or checked-out holder) into a
        /// retained holder under <paramref name="key"/>. The copy must be
        /// independent: continuing from either side must not disturb the other.</summary>
        bool TryCheckpointActiveCache(string key) => false;
        /// <summary>Deep-copy the retained holder <paramref name="retainedKey"/> into a
        /// fresh active holder for <paramref name="newRequestId"/>, leaving the retained
        /// one intact so the next request can clone it too.</summary>
        bool TryCloneRetainedCache(string retainedKey, string newRequestId) => false;

        // ---- Checkpoints that outlive the process --------------------------------
        //
        // A checkpoint is host bytes: never bound, never device-dirty, complete. That
        // is what makes it the one kind of holder that can be written to a file and
        // read back by a later process running the same weights (see
        // IPrefixCheckpointStore). The model owns the format and validates it on the
        // way in; the executor decides when to write and when to read.

        /// <summary>Whether <see cref="TryExportRetainedCache"/> and
        /// <see cref="TryImportRetainedCache"/> are implemented. Default false.</summary>
        bool SupportsRetainedCacheSerialization => false;

        /// <summary>Write the complete state of the retained holder <paramref name="key"/>
        /// to <paramref name="destination"/>, in a format only this model family reads.
        /// False (and nothing written) when the holder is not one whose host bytes are
        /// the truth, or does not exist.</summary>
        bool TryExportRetainedCache(string key, System.IO.Stream destination) => false;

        /// <summary>Create the retained holder <paramref name="key"/> from bytes an
        /// earlier <see cref="TryExportRetainedCache"/> wrote on the same model. False
        /// (and nothing created) when the bytes do not describe this model: another
        /// architecture, geometry, K/V precision or format version.</summary>
        bool TryImportRetainedCache(string key, System.IO.Stream source) => false;

        /// <summary>TRUE token-batched decode: decode ONE token for each of N
        /// concurrent sequences in a single fused graph (one compute buffer,
        /// weights loaded once) instead of N serial per-sequence forwards. This
        /// raises the round-robin ~1x concurrency ceiling toward llama.cpp's ~Nx
        /// (decode is memory-bandwidth bound, so batching amortises the weight
        /// loads). <paramref name="requestIds"/>/<paramref name="tokens"/>/
        /// <paramref name="positions"/> are parallel arrays of length N: sequence i
        /// decodes <paramref name="tokens"/>[i] at <paramref name="positions"/>[i]
        /// against its own per-request holder (including recurrent state for hybrid
        /// models). On success writes each sequence's logits into
        /// <paramref name="outLogits"/>[i] and returns true; returns
        /// false when the model can't batch this step (caller falls back to the
        /// per-sequence round-robin loop). Default false (opt-in).</summary>
        bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits) => false;

        /// <summary>Greedy fast path of <see cref="TryForwardBatchedFusedDecode"/>:
        /// instead of materializing [vocab] host logits per sequence, the model
        /// samples each sequence's next token ON-DEVICE (argmax) and returns just
        /// the token ids — for a 200k vocab at N=32 that replaces a ~25 MB
        /// PCIe download plus N host argmax scans per step (this is how
        /// vLLM/SGLang sample). Only called when every scheduled sequence's
        /// sampler is a plain argmax (see TokenSampler.IsPlainGreedyArgmax).
        /// Default false (opt-in per model).</summary>
        bool TryForwardBatchedFusedDecodeSampled(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, int[] outNextTokens) => false;

        /// <summary>Whether this sequence can join the token-batched decode at
        /// <paramref name="position"/> right now (holder exists, no cache growth
        /// needed). One sequence crossing a growth boundary then falls back to
        /// the per-sequence loop ALONE instead of declining the whole batch —
        /// previously that decline degraded every in-flight decode to a serial
        /// weight sweep for the step. Default true.</summary>
        bool CanBatchDecode(string requestId, int position) => true;
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
        /// <c>seq.TokenAt(...)</c>. The batched executor always sets this for
        /// plain steps (its content is identical to what <c>TokenAt</c> would
        /// return for prefill chunks, and it carries the sampled-but-not-yet-
        /// committed decode tokens that are absent from the sequence's token
        /// list), and speculative verify batches use it to forward drafted
        /// tokens that are not (yet) part of the sequence's token list.</summary>
        public int[] OverrideFlatTokens { get; set; }

        /// <summary>When non-null, receives the post-final-norm hidden state of
        /// every row (numTokens ├ù hidden floats) ÔÇö llama.cpp's h_nextn, consumed
        /// by the MTP draft head.</summary>
        public float[] CaptureHiddenAll { get; init; }

        /// <summary>When non-null, receives LM-head logits for every row
        /// (numTokens ├ù vocab floats) ÔÇö speculative verification needs per-row
        /// logits, not just the last position.</summary>
        public float[] CaptureLogitsAll { get; init; }
    }
}
