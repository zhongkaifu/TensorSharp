// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling.PrefixCache;

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
        private SequenceState? _currentOwner;
        // Number of tokens the model currently holds for the current owner.
        // Equals model._cacheSeqLen for purely-attention models.
        private int _ownerTokensInModel;
        // Tokens forwarded for the current owner since the last ownership
        // change. Used by ExecuteStepPerSequence to rotate ownership at
        // DecodeQuantumTokens boundaries so a long-running owner doesn't
        // starve other scheduled sequences on the serial per-seq path.
        private int _ownerForwardedTokens;

        // ---- Live-cache continuation tracking ----
        // The exact sequence whose tokens [0, _liveCacheLen) are currently resident
        // in the model's live KV cache, and whether that state is still trustworthy.
        // Set after every per-sequence forward; invalidated whenever the model's KV
        // cache is reset / rebuilt for a different sequence (or the batched path
        // takes over). Lets a same-session follow-up turn whose prompt extends this
        // sequence skip re-prefill entirely by continuing from the live cache (the
        // radix prefix cache's KeepPrimary plan) - critical for sliding-window models
        // where the pooled snapshot can only reuse one window.
        private SequenceState? _liveCacheSeq;
        private int _liveCacheLen;
        private bool _liveCacheValid;

        // Why the LAST ComputeFusedContinuationLcp call returned 0, or null when it
        // returned a usable prefix; the scheduler reports it on the admission line.
        private string? _lastFusedDeclineReason;

        /// <summary>A holder or live cache of scope <paramref name="owner"/> may serve a
        /// request of scope <paramref name="requester"/> past the public prefix only when
        /// they are the same conversation. An unscoped side (an engine-level caller that
        /// does not scope its requests) matches every scope.</summary>
        internal static bool ScopeAllows(string? owner, string? requester)
            => owner == null || requester == null || string.Equals(owner, requester, StringComparison.Ordinal);

        /// <summary>A scope for logs: the first eight characters of the already-hashed
        /// scope, never a session id.</summary>
        internal static string DescribeScope(string? scope)
            => scope == null ? "unscoped" : scope.Length <= 8 ? scope : scope.Substring(0, 8);

        /// <summary>The media check of every reuse path: <paramref name="lcp"/> matching
        /// tokens of <paramref name="seq"/> against a cache whose media is
        /// <paramref name="cachedSpans"/>, clamped to the first media span that differs,
        /// is cut, or (for a model that cannot continue past media) exists.</summary>
        private int ClampReuseToMedia(int lcp, SequenceState seq, IReadOnlyList<PromptMediaSpan> cachedSpans)
        {
            int clamped = PromptMediaSpans.ClampReusablePrefix(
                lcp, seq.MediaSpans, cachedSpans, _model.SupportsReuseAcrossMediaSpan);
            if (!_model.SupportsReuseAcrossMediaSpan && !_mediaSpanReuseClampWarned
                && clamped < PromptMediaSpans.ClampReusablePrefix(lcp, seq.MediaSpans, cachedSpans))
            {
                // A model limit, not a content difference: say so once, because every
                // turn after an image now re-prefills from that image.
                _mediaSpanReuseClampWarned = true;
                _logger.LogWarning(
                    "Prompt reuse stops at the first image/audio span on this model: it cannot continue a cached " +
                    "prefix past media exactly (M-RoPE decode positions), so every turn after an attachment " +
                    "re-prefills from that attachment (this request: {Clamped} of {Matched} matching tokens kept). " +
                    "Text before the first attachment is still reused. Reported once.",
                    clamped, lcp);
            }
            // Media still to prefill after the reused prefix: only where the model can
            // prefill it at a non-zero position exactly.
            if (clamped > 0 && !_model.CanPrefillMediaAfterReusedPrefix(seq.PromptTokens.Count))
            {
                foreach (var span in seq.MediaSpans)
                {
                    if (span.End <= clamped)
                        continue;
                    // Except up to the public prefix: while checkpoints are in use the
                    // scheduler ends a prefill chunk there (AlignSharedPrefixBoundary), so
                    // the media is prefilled after position P whether or not anything is
                    // reused, and cloning the checkpoint is the same computation. Refusing
                    // it re-prefilled the system prompt for an identical reply (Gemma 4
                    // E4B, Metal: 1163 reused -> 0, time to first token 1.51 s -> 1.84 s).
                    int kept = seq.SharedPrefixTokens > 0 && RadixCache?.PublicCheckpointsSupported == true
                        ? Math.Min(clamped, seq.SharedPrefixTokens)
                        : 0;
                    if (!_mediaAfterReusedPrefixWarned)
                    {
                        _mediaAfterReusedPrefixWarned = true;
                        if (kept == 0)
                            _logger.LogWarning(
                                "A {PromptTokens}-token prompt whose attachment follows a reusable {Clamped}-token prefix " +
                                "prefills from zero instead: this model cannot prefill media after a reused prefix " +
                                "exactly at this length. Reported once.",
                                seq.PromptTokens.Count, clamped);
                        else
                            _logger.LogWarning(
                                "A {PromptTokens}-token prompt whose attachment follows a reusable {Clamped}-token prefix " +
                                "reuses only its {Kept}-token public prefix instead: this model cannot prefill media " +
                                "after a longer reused prefix exactly at this length. Reported once.",
                                seq.PromptTokens.Count, clamped, kept);
                    }
                    return kept;
                }
            }
            return clamped;
        }

        private bool _mediaSpanReuseClampWarned;
        private bool _mediaAfterReusedPrefixWarned;

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
        private byte[] _scratch = Array.Empty<byte>();

        // ---- NextN/MTP speculative decoding (see SpeculativeExecution) ----
        // At most one sequence at a time runs speculatively: the draft head's
        // KV cache and pending hidden state live in the model's single live
        // (linear) cache, so continuity is (sequence identity, exact trunk
        // position). Any KV rebuild/swap — ownership change, batched/fused
        // step, preemption — invalidates the context; it re-arms only at a
        // fresh full prefill from position 0.
        private SpecSeqContext? _specCtx;

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
        private bool _fusedBatchedSuccessReported;
        private bool _specPrefixReuseDeclineWarned;
        private bool _crossSeqSerializationWarned;
        private bool _pastCapSerializationWarned;
        private readonly HashSet<string> _fallbackTransitionsWarned = new(StringComparer.Ordinal);

        private void ReportBatchedFusedDecodeSuccess(int count)
        {
            if (_fusedBatchedSuccessReported) return;
            _fusedBatchedSuccessReported = true;
            _logger.LogInformation("Batched fused decode accepted {Count} sequences in one graph. Reported once.", count);
        }

        private sealed class SpecSeqContext
        {
            public required SequenceState Seq;
            public required SpeculativeExecution Exec;
            // Trunk position the next forward for Seq must start at; must equal
            // seq.NumComputedTokens (and, on the linear trunk, the model's
            // CacheSeqLen) to stay armed.
            public int NextPosition;
            // The token DRAWN from the last verify's mismatch/bonus row. It IS
            // the sequence's next output token (re-sampling from LastLogits
            // would bias toward the drafts); -1 when no draw is pending.
            public int PendingNextToken = -1;
        }

        public BatchExecutor(
            IModelArchitecture model,
            BlockPool pool,
            ContinuousBatchScheduler scheduler,
            ILogger? logger = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _pool = pool ?? throw new ArgumentNullException(nameof(pool));
            _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
            _blockSize = pool.BlockSize;
            _logger = logger ?? NullLogger.Instance;
        }

        public IModelArchitecture Model => _model;
        public SequenceState? CurrentOwner => _currentOwner;
        internal PrefixCacheCoordinator? RadixCache { get; private set; }
        public bool RadixPrefixCacheEnabled => RadixCache != null;

        internal void InitializeRadixCache(SchedulerConfig configuration)
        {
            if (!configuration.EnablePrefixCaching || _model is not IPrefixCacheModel prefixModel) return;
            PrefixCacheCapabilities capabilities = prefixModel.GetPrefixCacheCapabilities();
            RadixCache = new PrefixCacheCoordinator(_model, _pool, _scheduler, capabilities, _logger,
                configuration.PromptTailCheckpointMinTokens);
            _scheduler.AttachRadixCacheContinuation(RadixCache, ComputeFusedContinuationLcp, TryAdoptFusedContinuation);
        }

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
            // ggml_metal_synchronize aborts the process. A request already
            // waiting for the lock goes first (see GpuLockWaiters).
            GpuLockWaiters.YieldToWaiters(_model.GpuComputeLock, MaxYieldToPreparation);
            lock (_model.GpuComputeLock)
            {
                // A request adapter (text LoRA) is selected on the model for exactly this
                // step and cleared before the lock is released, in the same lock region,
                // so no other GPU user (a media encoder between steps, a warm-up) and no
                // other step ever sees it. Base steps only verify that none is mixed in.
                IAdapterAwareModel? adapterModel = SelectStepAdapter(output);
                try
                {
                    if (RadixCache != null)
                    {
                        RadixCache.Drain();
                        if (RadixCache.DisplacePrimary())
                        {
                            _liveCacheValid = false;
                            _liveCacheSeq = null;
                            _liveCacheLen = 0;
                            _currentOwner = null;
                            _ownerTokensInModel = 0;
                            _ownerForwardedTokens = 0;
                        }
                        foreach (var work in output.ScheduledWork)
                            _fusedSeqById[work.Sequence.RequestId] = work.Sequence;
                    }
                    // The scheduler freed the previous owner's blocks during
                    // FinishSequence / PreemptSequence, but our _currentOwner
                    // reference outlives that. Without this reset, the next
                    // step's TryPrepareSequencesForPagedIfNeeded would call into
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
                        long pathStarted = _stepLog ? Stopwatch.GetTimestamp() : 0;
                        var results = TryExecutePath(plan.Candidates[i], output, options);
                        if (results != null)
                        {
                            if (FirstTokenLogitsDump.IsArmed)
                                NoteDumpPrefillSteps(plan.Candidates[i], results);
                            if (_cbDebug)
                                TraceStepResults(plan.Candidates[i], output, results);
                            if (_stepLog)
                                LogStepTiming(plan.Candidates[i], pathStarted, results);
                            return results;
                        }
                    }

                    // Unreachable: the planner always terminates the chain with a
                    // path that cannot decline (PerSequence). Kept as a hard
                    // fallback so a planner bug degrades to correctness, not loss.
                    return ExecuteStepPerSequence(output);
                }
                finally
                {
                    _stepAdapterActive = false;
                    adapterModel?.SelectStepAdapter(null);
                }
            }
        }

        // True while the current step runs with a request adapter (SelectStepAdapter).
        // Speculation stays off for such a step: no drafter or verify trunk applies the
        // adapter. Never set on a base step, so base traffic speculates exactly as before.
        private bool _stepAdapterActive;

        /// <summary>Whether this step may arm or continue speculative decoding.</summary>
        private bool SpeculationEnabledThisStep => Speculation.Enabled && !_stepAdapterActive;

        /// <summary>Select <see cref="SchedulerOutput.Adapter"/> on the model for this step,
        /// after checking that every scheduled sequence runs with exactly that adapter.
        /// Returns the model to clear the selection on, or null for a base step. A step that
        /// mixes adapters, or names one the model cannot apply, throws before anything is
        /// selected or forwarded, and the engine fails that step's sequences only.</summary>
        private IAdapterAwareModel? SelectStepAdapter(SchedulerOutput output)
        {
            IRequestAdapter? adapter = output.Adapter;
            foreach (var work in output.ScheduledWork)
            {
                if (!ReferenceEquals(work.Sequence.Adapter, adapter))
                    throw new InvalidOperationException(
                        $"A step may run one request adapter: {work.Sequence.RequestId} runs with " +
                        $"{work.Sequence.Adapter?.Id ?? "the base model"}, but the step was scheduled for " +
                        $"{adapter?.Id ?? "the base model"}.");
            }
            if (adapter == null)
                return null;
            if (_model is not IAdapterAwareModel aware)
                throw new InvalidOperationException(
                    $"The loaded model cannot apply request adapters, so LoRA adapter {adapter.Id} cannot run.");
            // Bound when the adapter was loaded; a model loaded after that, or an adapter a
            // late request still holds after its release, is bound here, once.
            if (!aware.TryBindAdapter(adapter, out string? error))
                throw new InvalidOperationException(
                    error ?? $"LoRA adapter {adapter.Id} cannot be applied to the loaded model.");
            aware.SelectStepAdapter(adapter);
            _stepAdapterActive = true;
            return aware;
        }

        private long _cbStepIndex;

        // Bound on how long a step waits for a request thread to take the GPU lock
        // (TS_GPU_LOCK_YIELD_MS; 0 lets the worker take it first, as before).
        private static readonly TimeSpan MaxYieldToPreparation = TimeSpan.FromMilliseconds(
            int.TryParse(Environment.GetEnvironmentVariable("TS_GPU_LOCK_YIELD_MS"), out int yieldMs) && yieldMs >= 0 ? yieldMs : 100);

        /// <summary>TS_SCHED_PARALLEL_SAMPLE=0: a batched decode step samples its rows one after another
        /// (settable for tests).</summary>
        internal static bool ParallelSampling { get; set; } =
            !string.Equals(Environment.GetEnvironmentVariable("TS_SCHED_PARALLEL_SAMPLE"), "0", StringComparison.Ordinal);

        /// <summary>Batched decode steps whose rows were sampled in parallel, process-wide (tests).</summary>
        internal static long ParallelSampledSteps;

        /// <summary>Rows from which a batched step samples them in parallel.</summary>
        private const int ParallelSamplingMinRows = 4;

        private static readonly bool _stepLog =
            string.Equals(Environment.GetEnvironmentVariable("TS_SCHED_STEP_LOG"), "1", StringComparison.Ordinal);
        private long _stepLogLastEnd;

        /// <summary>TS_SCHED_STEP_LOG=1: one line per step with the idle gap since the
        /// previous step returned, the time the serving path took, and each sequence's
        /// prefill/decode share, so a request that waits between its own forwards shows
        /// where the wall clock went.</summary>
        private void LogStepTiming(ExecutionPathKind path, long pathStarted, List<SequenceStepResult> results)
        {
            long now = Stopwatch.GetTimestamp();
            double gapMs = _stepLogLastEnd == 0 ? 0 : (pathStarted - _stepLogLastEnd) * 1000.0 / Stopwatch.Frequency;
            double stepMs = (now - pathStarted) * 1000.0 / Stopwatch.Frequency;
            _stepLogLastEnd = now;
            var sb = new StringBuilder();
            sb.Append("[step] t=").Append(DateTime.UtcNow.ToString("HH:mm:ss.fff"))
              .Append(" gap=").Append(gapMs.ToString("F1")).Append(" ms=").Append(stepMs.ToString("F1"))
              .Append(' ').Append(path);
            foreach (var r in results)
            {
                string id = r.Sequence?.RequestId ?? "?";
                sb.Append(' ').Append(id.Length > 9 ? id.Substring(id.Length - 4) : id)
                  .Append(r.IsPrefill ? ":P" : ":D").Append(r.TokensForwarded);
            }
            try { Console.Error.WriteLine(sb.ToString()); }
            catch (Exception) { }
        }

        /// <summary>TS_CB_DEBUG=1 step trace: the path that served the step, every
        /// scheduled sequence (request, prefill/decode, tokens forwarded, computed
        /// tokens after the step) and a fingerprint of the logits it left behind: the
        /// top two tokens with their margin and a hash of the whole row. Two runs of
        /// the same workload can then be diffed step by step, telling a different
        /// batch composition from the same composition producing different logits.</summary>
        private void TraceStepResults(ExecutionPathKind path, SchedulerOutput output, List<SequenceStepResult> results)
        {
            var sb = new StringBuilder();
            sb.Append("[cb] step#").Append(_cbStepIndex++).Append(' ').Append(path)
              .Append(" scheduled=").Append(output.ScheduledWork.Count).Append(" served=").Append(results.Count);
            foreach (var r in results)
            {
                var seq = r.Sequence;
                sb.Append(" | ").Append(seq?.RequestId).Append(r.IsPrefill ? ":P" : ":D")
                  .Append(" fwd=").Append(r.TokensForwarded)
                  .Append(" computed=").Append(seq?.NumComputedTokens)
                  .Append(" in=").Append(r.SampledToken);
                if (r.Error != null) sb.Append(" error=").Append(r.Error.GetType().Name);
                sb.Append(' ').Append(DescribeLogits(seq?.LastLogits));
            }
            // The step has already committed; a failing diagnostic writer must not
            // turn it into a failed step.
            try { Console.Error.WriteLine(sb.ToString()); }
            catch (Exception) { }
        }

        /// <summary>TS_DUMP_LOGITS: remember which path ran each prompt chunk, for the sidecar of the
        /// first-token dump (<see cref="FirstTokenLogitsDump"/>). Only while a dump is still owed.</summary>
        private static void NoteDumpPrefillSteps(ExecutionPathKind path, List<SequenceStepResult> results)
        {
            foreach (var r in results)
            {
                if (r.IsPrefill && r.Error == null && r.TokensForwarded > 0 && r.Sequence != null
                    && r.Sequence.OutputTokens.Count == 0 && !r.Sequence.ExcludedFromFirstTokenDump)
                    r.Sequence.NoteDumpPrefillStep(path.ToString(), r.TokensForwarded);
            }
        }

        /// <summary>
        /// TS_DUMP_LOGITS: the logits <paramref name="seq"/>'s FIRST generated token is about to be
        /// chosen from - its last prompt row, however the prompt was run - written before the sampler
        /// can mask or penalize them in place. Null when the dump is off, already written, or there are
        /// no host logits to write. Callers check <c>OutputTokens.Count == 0</c> and
        /// <see cref="FirstTokenLogitsDump.IsArmed"/> first, so a run without the variable pays nothing.
        /// </summary>
        private static FirstTokenLogitsDump.Capture? CaptureFirstTokenLogits(SequenceState seq)
        {
            // A warm-up request (submitted inside FirstTokenLogitsDump.Suppress) is excluded by its
            // own flag: this runs on the engine worker, whose async flow says nothing about it.
            if (seq.LastLogits == null || seq.ExcludedFromFirstTokenDump)
                return null;
            return FirstTokenLogitsDump.TryCaptureCore(seq.LastLogits, new FirstTokenLogitsInfo
            {
                Source = "engine",
                RequestId = seq.RequestId,
                PromptTokens = seq.PromptTokens.Count,
                PromptFingerprint = FirstTokenLogitsDump.Fingerprint(seq.PromptTokens),
                MediaSpans = seq.MediaSpans.Count,
                Adapter = seq.Adapter?.Id,
                PrefixReusedTokens = seq.PrefixCacheReusedTokens,
                PrefillSteps = seq.DumpPrefillSteps?.ToArray(),
            });
        }

        internal static string DescribeLogits(float[]? logits)
        {
            if (logits == null || logits.Length < 2) return "logits=<none>";
            int top1 = logits[0] >= logits[1] ? 0 : 1, top2 = 1 - top1;
            ulong hash = 14695981039346656037UL;
            for (int i = 0; i < logits.Length; i++)
            {
                float v = logits[i];
                if (i >= 2)
                {
                    if (v > logits[top1]) { top2 = top1; top1 = i; }
                    else if (v > logits[top2]) top2 = i;
                }
                hash = (hash ^ (uint)BitConverter.SingleToInt32Bits(v)) * 1099511628211UL;
            }
            return string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"top1={top1}:{logits[top1]:R} top2={top2}:{logits[top2]:R} margin={logits[top1] - logits[top2]:R} hash={hash:x16}");
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
                SoloIsDecode = count == 1 && !output.ScheduledWork[0].IsPrefill,
                SoloKvInPagedStorage = soloPaged,
                SoloHasFusedCache = soloFused,
                SoloRequiresOwnershipSwap = soloSwap,
                AdapterActive = output.Adapter != null,
            };
        }

        /// <summary>Dispatch one plan candidate. Returns null when a declinable
        /// candidate passes on the step (the caller then tries the next
        /// candidate in the plan's chain).</summary>
        private List<SequenceStepResult>? TryExecutePath(
            ExecutionPathKind path, SchedulerOutput output, ExecutionOptions options)
        {
            switch (path)
            {
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
        private List<SequenceStepResult>? TryExecuteStepBatchedPaged(
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
                        "BatchExecutor: linear-to-paged migration failed for {EngineRequestId}; falling back to per-seq path.",
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
        private string? _lastPlanDescription;

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

            SequenceState? initialOwner = _currentOwner;
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
        private List<SequenceStepResult>? ExecuteLinearAndPagedSplit(
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
                        "BatchExecutor cannot safely fall back paged sequence {EngineRequestId} to linear execution.",
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
        private static void DetachBorrowedLogits(SequenceState? seq)
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
                "fused/captured decode). Serving the fast standard decode instead — no action needed.");
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
        private string? _lateArmDeclinedFor;

        // The speculation policy in force. Starts as the scheduler config's and is
        // replaced by SetSpeculation when the host toggles speculation at run time (a
        // settings switch), so a change applies to the next turn instead of the next
        // model load. _planConfig is the scheduler config with this policy, for the
        // planner, which is a pure function of its config.
        private SpeculationOptions? _speculation;
        private SchedulerConfig? _planConfig;

        /// <summary>The speculation policy this executor currently applies.</summary>
        public SpeculationOptions Speculation => _speculation ?? _scheduler.Config.Speculation;

        private SchedulerConfig PlanConfig => _planConfig ?? _scheduler.Config;

        /// <summary>
        /// Replace the speculation policy for every step from now on. Engine thread
        /// only, between steps: every armed context is dropped (the next step re-arms
        /// under the new policy, over the tokens the caches already hold), and the
        /// one-time decline notices are re-armed so the new policy reports its own.
        /// </summary>
        public void SetSpeculation(SpeculationOptions? options)
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
            _sharedGovernorProfile = null;
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
                bool carry = fused.HasFusedSequenceCache(_specCtx.Seq.RequestId)
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
            HashSet<string>? handledBatched = null;
            if (options.BatchedFusedDecodeEnabled && n >= 2)
            {
                var decodeWork = new List<ScheduledSequenceWork>(n);
                foreach (var work in output.ScheduledWork)
                    if (!work.IsPrefill &&
                        fused.CanBatchDecode(work.Sequence.RequestId, work.Sequence.NumComputedTokens))
                        decodeWork.Add(work);
                int dn = decodeWork.Count;
                // A lone decode beside prefill chunks speculates too. Decoding it one
                // token per step behind another request's 2048-token chunk (~300 ms a
                // step) stretched a 41-token reply over seconds; the batched verify of a
                // single sequence costs about what that plain decode does.
                bool speculate = dn >= 2 || (dn == 1
                    && !_pendingRetainedFusedTruncations.ContainsKey(decodeWork[0].Sequence.RequestId));
                // The step's prompt chunk rides the batched verify when the model can carry it:
                // one forward for both, so the verify reads no weights of its own.
                HashSet<string>? mixedHandled = null;
                ScheduledSequenceWork? mixedWork = speculate ? TryStageMixedPrefill(fused, output, results, ref mixedHandled) : null;
                long specStarted = Stopwatch.GetTimestamp();
                bool speculated = speculate && TryRunBatchedSpeculativeSubset(decodeWork, results, ref handledBatched);
                if (mixedWork != null)
                    CompleteMixedPrefill(mixedWork, Stopwatch.GetTimestamp() - specStarted, results, ref mixedHandled);
                // No batched verify: on a trunk that prefers it, one sequence takes a solo speculative turn instead.
                if (!speculated && dn >= 2 && dn == n && TrySpeculativeTurn(fused, decodeWork, results))
                    return results;
                if (!speculated && dn >= 2)
                {
                    var reqIds = new string[dn];
                    var btokens = new int[dn];
                    var bpositions = new int[dn];
                    try
                    {
                        bool allGreedy = true;
                        // Each sequence samples with its own sampler (its own RNG and buffers) from its own
                        // logits, so a batch's draws run side by side: a full-vocabulary top-k is ~0.3 ms a
                        // row at Qwen's 248K tokens.
                        int[]? peeked = null;
                        if (ParallelSampling && dn >= ParallelSamplingMinRows && !FirstTokenLogitsDump.IsArmed)
                        {
                            var drawn = new int[dn];
                            try
                            {
                                System.Threading.Tasks.Parallel.For(0, dn, i => drawn[i] = PeekPendingOrSample(decodeWork[i].Sequence));
                            }
                            catch (AggregateException ae) when (ae.InnerExceptions.Count > 0)
                            {
                                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ae.InnerExceptions[0]).Throw();
                            }
                            peeked = drawn;
                            Interlocked.Increment(ref ParallelSampledSteps);
                        }
                        for (int i = 0; i < dn; i++)
                        {
                            var seq = decodeWork[i].Sequence;
                            // Peek-sample (deterministic for greedy); do NOT append or
                            // consume the device-sampled stash yet, so a decline below
                            // leaves the round-robin fallback a valid token source
                            // (device-sampled sequences have no LastLogits to re-sample).
                            btokens[i] = peeked != null ? peeked[i] : PeekPendingOrSample(seq);
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
                            long batchStarted = Stopwatch.GetTimestamp();
                            bool sampledBatchSucceeded = fused.TryForwardBatchedFusedDecodeSampled(reqIds, btokens, bpositions, nextTokens);
                            long batchTicksPerSequence = (Stopwatch.GetTimestamp() - batchStarted) / dn;
                            if (sampledBatchSucceeded)
                            {
                                ReportBatchedFusedDecodeSuccess(dn);
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
                                        ForwardElapsedTicks = batchTicksPerSequence,
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
                            long batchStarted = Stopwatch.GetTimestamp();
                            bool logitsBatchSucceeded = fused.TryForwardBatchedFusedDecode(reqIds, btokens, bpositions, outLogits);
                            long batchTicksPerSequence = (Stopwatch.GetTimestamp() - batchStarted) / dn;
                            if (logitsBatchSucceeded)
                            {
                                ReportBatchedFusedDecodeSuccess(dn);
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
                                        ForwardElapsedTicks = batchTicksPerSequence,
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
                                    "serial fused path for this step. Reason: {Reason}. Reported once.",
                                    dn, fused.BatchedFusedDecodeDeclineReason ?? "model did not provide a reason");
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        // A throwing kernel may already have written some caches.
                        // Fail its entire decode subset; retrying them serially
                        // would consume a token against partially advanced state.
                        // Prefills and other requests remain independently runnable.
                        // Sampling can fail before every reqIds entry was filled.
                        // Use the scheduled subset rather than that partial array.
                        handledBatched = new HashSet<string>(decodeWork.Select(work => work.Sequence.RequestId));
                        results.RemoveAll(result => handledBatched.Contains(result.Sequence.RequestId));
                        _logger.LogError(ex,
                            "Batched fused decode failed for {Count} sequences; failing the affected requests without retrying the batch.",
                            dn);
                        foreach (var work in decodeWork)
                        {
                            var seq = work.Sequence;
                            seq.Error = ex;
                            results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
                        }
                    }
                }
                if (mixedHandled != null)
                {
                    handledBatched ??= new HashSet<string>(StringComparer.Ordinal);
                    handledBatched.UnionWith(mixedHandled);
                }
                if (handledBatched != null && handledBatched.Count == n)
                    return results;
            }

            // ---- Batched fused prefill ----
            // Concurrent arrivals' prompt chunks evaluate in ONE fused graph when the
            // model supports it, so they share every weight read instead of prefilling
            // one after another. Only plain chunks qualify: a request whose holder must
            // first be created around an injected prefix, rewound from a retained
            // state (unless the model rewinds it in the batched call:
            // BatchedPrefillRewindsHolders), or fed media embeddings keeps the
            // per-sequence loop below.
            if (options.BatchedFusedDecodeEnabled && n >= 2)
            {
                PrebindCachedPrefixPrefills(fused, output, results, ref handledBatched);
                if (handledBatched != null && handledBatched.Count == n)
                    return results;
                var prefillWork = new List<ScheduledSequenceWork>(n);
                foreach (var work in output.ScheduledWork)
                {
                    var seq = work.Sequence;
                    if (!work.IsPrefill || (handledBatched != null && handledBatched.Contains(seq.RequestId)))
                        continue;
                    // A holder cloned from a longer cached state is rewound on its first step;
                    // a model that rewinds it inside the batched call (exactly) takes it here.
                    if (_pendingRetainedFusedTruncations.TryGetValue(seq.RequestId, out int rewindTo)
                        && !(fused.BatchedPrefillRewindsHolders && rewindTo == seq.NumComputedTokens))
                        continue;
                    if (!fused.HasFusedSequenceCache(seq.RequestId) && (seq.NumComputedTokens > 0 || seq.UsesLiveCacheContinuation))
                        continue;
                    if (_model.MultimodalInjector != null && _model.MultimodalInjector.HasPendingEmbeddings(seq.RequestId))
                        continue;
                    prefillWork.Add(work);
                }
                if (prefillWork.Count >= 2)
                {
                    int pn = prefillWork.Count;
                    var reqIds = new string[pn];
                    var chunks = new int[pn][];
                    var starts = new int[pn];
                    for (int i = 0; i < pn; i++)
                    {
                        var seq = prefillWork[i].Sequence;
                        reqIds[i] = seq.RequestId;
                        starts[i] = seq.NumComputedTokens;
                        // No reservation yet: it targets the ACTIVE cache, and the
                        // per-sequence loop below binds each holder before reserving
                        // when the model declines the batched prefill.
                        chunks[i] = PrefillTokens(seq, prefillWork[i]);
                    }
                    var outLogits = new float[pn][];
                    try
                    {
                        long started = Stopwatch.GetTimestamp();
                        if (fused.TryForwardBatchedFusedPrefill(reqIds, chunks, starts, outLogits))
                        {
                            long ticksPerSequence = (Stopwatch.GetTimestamp() - started) / pn;
                            handledBatched ??= new HashSet<string>(StringComparer.Ordinal);
                            for (int i = 0; i < pn; i++)
                            {
                                var seq = prefillWork[i].Sequence;
                                _pendingRetainedFusedTruncations.Remove(seq.RequestId);   // rewound by the model
                                seq.PrefillReservationTaken = true;   // the batched prefill sized its own holders
                                seq.LastLogits = outLogits[i];
                                seq.AdvanceComputedTokens(chunks[i].Length);
                                MaybeCheckpointSharedPrefix(seq);
                                results.Add(new SequenceStepResult
                                {
                                    Sequence = seq,
                                    TokensForwarded = chunks[i].Length,
                                    SampledToken = -1,
                                    IsPrefill = true,
                                    FullBlocksCaptured = 0,
                                    ForwardElapsedTicks = ticksPerSequence,
                                });
                                handledBatched.Add(seq.RequestId);
                            }
                            if (handledBatched.Count == n)
                                return results;
                        }
                    }
                    catch (Exception ex)
                    {
                        // As for the batched decode: caches may be partially written,
                        // so fail the affected prefills rather than retrying them.
                        handledBatched ??= new HashSet<string>(StringComparer.Ordinal);
                        _logger.LogError(ex, "Batched fused prefill failed for {Count} sequences; failing the affected requests.", pn);
                        foreach (var work in prefillWork)
                        {
                            var seq = work.Sequence;
                            seq.Error = ex;
                            results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
                            handledBatched.Add(seq.RequestId);
                        }
                        if (handledBatched.Count == n)
                            return results;
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
                // A lone holder's decode takes the batched verify when the trunk has one:
                // it replays only the accepted prefix, so the drafter's whole block costs
                // about what three drafts do on the solo verify. Tried before binding: the
                // batched verify reads holders where they are parked.
                if (!work.IsPrefill && n == 1 && _scheduler.RunningCount == 1
                    && fused.HasFusedSequenceCache(seq.RequestId)
                    && !_pendingRetainedFusedTruncations.ContainsKey(seq.RequestId))
                {
                    HashSet<string>? soloHandled = null;
                    if (TryRunBatchedSpeculativeSubset(new List<ScheduledSequenceWork> { work }, results, ref soloHandled))
                        continue;
                }
                try
                {
                    if (!BindFusedHolder(fused, seq, results))
                        continue;

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
                        && TrySpeculativeFusedDecode(seq, prevComputed, out SequenceStepResult? specResult))
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
                    float[] logits = work.IsPrefill ? _model.ForwardPrompt(inputTokens) : _model.Forward(inputTokens);
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
                catch (SequenceSlotUnavailableException ex) when (seq.NumComputedTokens == prevComputed
                                                                   && !fused.HasFusedSequenceCache(seq.RequestId))
                {
                    // Device memory holds no slot for a NEW request. Nothing of it ran and nothing of it
                    // is held, so it waits for a running request to finish rather than failing: this is
                    // what four concurrent chats on GLM-5.3-Flash, whose context the loader sizes to fill
                    // the devices, used to return as "sequence-slot allocation failed".
                    _logger.LogDebug("No sequence slot for {EngineRequestId} yet: {Reason}", seq.RequestId, ex.Message);
                    results.Add(new SequenceStepResult { Sequence = seq, SlotUnavailable = true });
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Fused per-seq step failed for sequence {EngineRequestId}", seq.RequestId);
                    results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
                    seq.Error = ex;
                }
            }
            return results;
        }

        // TS_SCHED_MIXED_STEP=0 keeps a step's prompt chunk out of its batched verify.
        private static readonly bool s_mixedStep = Environment.GetEnvironmentVariable("TS_SCHED_MIXED_STEP") != "0";

        /// <summary>
        /// Stage the step's one prompt chunk to ride its batched verify (see
        /// <see cref="IMixedStepPrefillTarget"/>), its holder bound and its media queued as the
        /// per-sequence loop does. Null when the step has no such chunk or the model declines it;
        /// a sequence whose step ended while binding goes to <paramref name="handled"/>.
        /// </summary>
        private ScheduledSequenceWork? TryStageMixedPrefill(IBatchedPagedModel fused, SchedulerOutput output,
            List<SequenceStepResult> results, ref HashSet<string>? handled)
        {
            if (!s_mixedStep || _model is not IMixedStepPrefillTarget target)
                return null;
            ScheduledSequenceWork? pick = null;
            foreach (var work in output.ScheduledWork)
            {
                if (!work.IsPrefill)
                    continue;
                // Several chunks share their weight reads in the batched prefill instead.
                if (pick != null)
                    return null;
                pick = work;
            }
            if (pick == null)
                return null;
            var seq = pick.Sequence;
            int prevComputed = seq.NumComputedTokens;
            try
            {
                if (!BindFusedHolder(fused, seq, results))
                {
                    (handled ??= new HashSet<string>(StringComparer.Ordinal)).Add(seq.RequestId);
                    return null;
                }
                int[] tokens = BuildPrefillChunk(seq, pick);
                int position = seq.NumComputedTokens, end = position + tokens.Length;
                _model.MultimodalInjector?.QueuePromptEmbeddingsForSlice(position, tokens.Length, seq.RequestId);
                bool whole = end >= seq.PromptTokens.Count || seq.IsPublicCheckpointBoundary(end);
                return target.StageMixedPrefill(seq.RequestId, tokens, position, whole) ? pick : null;
            }
            catch (SequenceSlotUnavailableException ex) when (seq.NumComputedTokens == prevComputed
                                                               && !fused.HasFusedSequenceCache(seq.RequestId))
            {
                _logger.LogDebug("No sequence slot for {EngineRequestId} yet: {Reason}", seq.RequestId, ex.Message);
                results.Add(new SequenceStepResult { Sequence = seq, SlotUnavailable = true });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Fused per-seq step failed for sequence {EngineRequestId}", seq.RequestId);
                results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
                seq.Error = ex;
            }
            (handled ??= new HashSet<string>(StringComparer.Ordinal)).Add(seq.RequestId);
            return null;
        }

        /// <summary>Book the staged chunk the batched verify ran (a prefix of it when the model
        /// trimmed it to its GEMM tiles); one no verify ran stays for the per-sequence loop.</summary>
        private void CompleteMixedPrefill(ScheduledSequenceWork work, long elapsedTicks, List<SequenceStepResult> results,
            ref HashSet<string>? handled)
        {
            var seq = work.Sequence;
            try
            {
                bool carried = ((IMixedStepPrefillTarget)_model).TakeMixedPrefill(seq.RequestId, out int ran, out float[]? logits);
                if (s_batchedSpecLog)
                    Console.Error.WriteLine($"[mixed-step] {seq.RequestId} @{seq.NumComputedTokens} " +
                        $"{(carried ? $"ran {ran}/{work.NumScheduledTokens}" : $"not carried ({work.NumScheduledTokens})")}");
                if (!carried)
                    return;
                seq.LastLogits = logits;
                seq.AdvanceComputedTokens(ran);
                MaybeCheckpointSharedPrefix(seq);
                results.Add(new SequenceStepResult
                {
                    Sequence = seq,
                    TokensForwarded = ran,
                    SampledToken = -1,
                    IsPrefill = true,
                    FullBlocksCaptured = 0,
                    ForwardElapsedTicks = elapsedTicks,
                });
            }
            catch (Exception ex)
            {
                // The forward that carried it may have written its cache: fail it rather than rerun it.
                _logger.LogError(ex, "Mixed-step prefill failed for sequence {EngineRequestId}", seq.RequestId);
                results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
                seq.Error = ex;
            }
            (handled ??= new HashSet<string>(StringComparer.Ordinal)).Add(seq.RequestId);
        }

        /// <summary>Bind <paramref name="seq"/>'s per-request holder for this step's forward: rewind a
        /// retained holder matched past the prompt, drop a revoked live-cache claim and inject a
        /// reused prefix into a fresh holder. False when the step is over for the sequence (a result
        /// was added).</summary>
        private bool BindFusedHolder(IBatchedPagedModel fused, SequenceState seq, List<SequenceStepResult> results)
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
                        "Retained fused cache for {EngineRequestId} could not rewind to {Tokens}; " +
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
                    return false;
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
                    "Planned live-cache continuation for {EngineRequestId} was revoked before " +
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
                InjectPrefixOrRecompute(seq, seq.NumComputedTokens);
            return true;
        }

        /// <summary>
        /// Binds the holders of the step's prompt chunks that continue a cached prefix (the radix,
        /// a shared-prefix checkpoint, the live cache) but have no holder yet, injecting the prefix
        /// and taking the reservation as the per-sequence loop does, so they can join the batched
        /// prefill, which takes only sequences whose holder sits at the chunk's position. These ran
        /// one after another before: concurrent chats on one cached document (RTX 5090, Qwen3.8
        /// 27B, two chats on a 127K-token document: 47-token suffixes, first tokens at 147 and
        /// 281 ms). Only when at least two chunks would then batch; a sequence whose step ends
        /// while binding goes to <paramref name="handled"/>.
        /// </summary>
        private void PrebindCachedPrefixPrefills(IBatchedPagedModel fused, SchedulerOutput output,
            List<SequenceStepResult> results, ref HashSet<string>? handled)
        {
            List<ScheduledSequenceWork>? pending = null;
            int ready = 0;
            foreach (var work in output.ScheduledWork)
            {
                var seq = work.Sequence;
                if (!work.IsPrefill || (handled != null && handled.Contains(seq.RequestId))
                    || (_model.MultimodalInjector != null && _model.MultimodalInjector.HasPendingEmbeddings(seq.RequestId)))
                    continue;
                if (_pendingRetainedFusedTruncations.TryGetValue(seq.RequestId, out int rewindTo))
                {
                    if (fused.BatchedPrefillRewindsHolders && rewindTo == seq.NumComputedTokens)
                        ready++;
                    continue;
                }
                if (fused.HasFusedSequenceCache(seq.RequestId) || (seq.NumComputedTokens == 0 && !seq.UsesLiveCacheContinuation))
                    ready++;
                else
                    (pending ??= new List<ScheduledSequenceWork>()).Add(work);
            }
            if (pending == null || ready + pending.Count < 2 || !fused.BatchedPrefillTakesPrebound)
                return;
            foreach (var work in pending)
            {
                var seq = work.Sequence;
                int prevComputed = seq.NumComputedTokens;
                try
                {
                    if (BindFusedHolder(fused, seq, results))
                    {
                        BuildPrefillChunk(seq, work);   // only its reservation: the chunk is taken again below
                        continue;
                    }
                }
                catch (SequenceSlotUnavailableException ex) when (seq.NumComputedTokens == prevComputed
                                                                   && !fused.HasFusedSequenceCache(seq.RequestId))
                {
                    _logger.LogDebug("No sequence slot for {EngineRequestId} yet: {Reason}", seq.RequestId, ex.Message);
                    results.Add(new SequenceStepResult { Sequence = seq, SlotUnavailable = true });
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Fused per-seq step failed for sequence {EngineRequestId}", seq.RequestId);
                    results.Add(new SequenceStepResult { Sequence = seq, Error = ex });
                    seq.Error = ex;
                }
                (handled ??= new HashSet<string>(StringComparer.Ordinal)).Add(seq.RequestId);
            }
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
                // A swap-out is lossless only while everything the owner holds fits the
                // pooled snapshot: ExtractAllBlocks stops at CapturableBlocks, and a ring
                // cache past its rows (Muse-Glimmer 4352, Gemma 4's window) cannot be
                // snapshotted at all. Swapping such an owner out would make it re-forward
                // everything past the cap each time it comes back, so stay with it, as a
                // model that cannot swap does; it finishes and the others then resume.
                if (canSwap && _ownerTokensInModel > _model.MaxReusablePrefixTokens)
                {
                    canSwap = false;
                    if (output.ScheduledWork.Count > 1 && !_pastCapSerializationWarned)
                    {
                        _pastCapSerializationWarned = true;
                        _logger.LogInformation(
                            "{EngineRequestId} holds {Tokens} tokens, past the {Cap} this model can swap out and " +
                            "restore; the other {Others} scheduled request(s) wait until it finishes instead " +
                            "of each swap re-prefilling it. Reported once.",
                            _currentOwner.RequestId, _ownerTokensInModel, _model.MaxReusablePrefixTokens,
                            output.ScheduledWork.Count - 1);
                    }
                }
                else if (!canSwap && output.ScheduledWork.Count > 1 && !_crossSeqSerializationWarned)
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

                ScheduledSequenceWork? ownerWork = null;
                ScheduledSequenceWork? firstNonOwner = null;
                ScheduledSequenceWork? freshNonOwner = null;
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
                    float[] logits = work.IsPrefill ? _model.ForwardPrompt(inputTokens) : _model.Forward(inputTokens);
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
                    // fires — so a defensive 1 MB clone per token (Gemma 4
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
                    _logger.LogError(ex, "Step failed for sequence {EngineRequestId}", seq.RequestId);
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
        /// starts with a clean model cache).</summary>
        private SequenceStepResult? TryExecuteSpeculativeStep(
            SequenceState seq, ScheduledSequenceWork work, int prevComputed)
        {
            if (!SpeculationEnabledThisStep)
                return null;
            if (_model is not ISpeculativeTarget spec)
                return null;
            // Net-negative on backends without the accelerated multi-token
            // verify/draft path; the normal decode path serves the step.
            if (!spec.SpeculationProfitable)
                return null;
            // Prepared spans remain available for retry throughout the request.
            // Only prefill requires injection, and a trunk must explicitly opt in
            // before its speculative forward may consume those embeddings.
            if (work.IsPrefill && !spec.SpecSupportsMultimodalPrefill
                && _model.MultimodalInjector != null
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
            // of heads without a safe suffix restart - they decline below - but it was
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
                && ReferenceEquals(_specCtx.Seq, seq)
                && _specCtx.NextPosition == prevComputed;
            if (work.IsPrefill && spec.CacheSeqLen == prevComputed && !continuesThisContext)
            {
                var exec = TryArmSpeculation(spec, seq, trunk: null, trunkLabel: "linear");
                if (exec == null)
                    return null;
                if (exec.CanSeedCommitted && !exec.PrefillsThroughContext)
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
            // speculator that needs no hidden state (n-gram), or a learned head
            // with a safe gap restart, can start from the tokens the trunk holds.
            // A restarted learned head first captures a fresh hidden carry.
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
                || !ReferenceEquals(_specCtx.Seq, seq)
                || _specCtx.NextPosition != prevComputed
                || spec.CacheSeqLen != prevComputed)
            {
                return null;
            }

            return ExecuteSpeculativeWorkCore(_specCtx, seq, work, prevComputed);
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
        // Share measurements while the resolved model/head/configuration policy
        // agrees. A different request profile receives a new governor, so a live
        // execution never has its selected policy changed by the next request.
        private SpeculationCostGovernor? _sharedGovernor;
        private SpeculationGovernorFactory.Profile? _sharedGovernorProfile;
        private SpeculationCostGovernor SharedGovernorFor(ISpeculativeTarget target, ISpeculator speculator, SequenceState sequence)
        {
            var profile = SpeculationGovernorFactory.Describe(target, speculator,
                CanUseFixedGovernorGreedyProfile(sequence, FirstTokenLogitsDump.IsArmed || _cbDebug),
                SpeculationCostGovernor.AdaptiveEnvironmentSnapshot);
            return SpeculationGovernorFactory.Share(ref _sharedGovernor, ref _sharedGovernorProfile, profile);
        }

        /// <summary>Whether a request can profit from drafting at all: it needs at
        /// least two tokens to decode, or every proposal would be wasted.</summary>
        private static bool CanSpeculate(SequenceState seq) => seq.MaxNewTokens >= 2;

        private SpeculativeExecution? TryArmSpeculation(
            ISpeculativeTarget spec, SequenceState seq, ISpecTrunk? trunk, string trunkLabel)
        {
            if (!CanSpeculate(seq))
                return null;
            var speculator = SpeculatorRegistry.Create(
                spec, Speculation, out string? declineReason);
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

            var exec = new SpeculativeExecution(spec, speculator, trunk, SharedGovernorFor(spec, speculator, seq));
            _logger.LogInformation(
                "Speculative decoding armed for {EngineRequestId} (algorithm={Algorithm}, maxDraft={MaxDraft}, "
                + "pMin={PMin}, trunk={Trunk}, adaptive={Adaptive}, adaptiveSource={AdaptiveSource})",
                seq.RequestId, speculator.Describe(), exec.MaxDraftTokens, exec.MinDraftProb, trunkLabel,
                exec.AdaptiveSpeculation, exec.Governor.SelectionSource);
            return exec;
        }

        private void WarnSpeculationDeclinedOnce(string? reason)
        {
            if (_speculationDeclineWarned || string.IsNullOrEmpty(reason))
                return;
            _speculationDeclineWarned = true;
            _logger.LogWarning(
                "Speculative decoding was requested but is not available: {Reason}. "
                + "Serving standard decoding instead.", reason);
        }

        /// <summary>The MTP step body; the caller has already validated arming and
        /// continuity on <see cref="_specCtx"/>.</summary>
        private SequenceStepResult ExecuteSpeculativeWorkCore(
            SpecSeqContext context, SequenceState seq, ScheduledSequenceWork work, int prevComputed)
        {
            (_model as IPerRequestDraftState)?.BindDraftState(seq.RequestId);
            if (work.IsPrefill)
            {
                int[] chunk = BuildPrefillChunk(seq, work);
                if (_model is ISpeculativeTarget { SpecSupportsMultimodalPrefill: true })
                    _model.MultimodalInjector?.QueuePromptEmbeddingsForSlice(
                        prevComputed, chunk.Length, seq.RequestId);
                var swPrefill = Stopwatch.StartNew();
                float[] logits = context.Exec.PrefillStep(chunk, prevComputed, seq.PromptTokens.Count);
                swPrefill.Stop();
                seq.LastLogits = logits;
                CompleteSpeculativeStepBookkeeping(seq, chunk.Length);
                context.NextPosition = seq.NumComputedTokens;

                return new SequenceStepResult
                {
                    Sequence = seq,
                    TokensForwarded = chunk.Length,
                    SampledToken = -1,
                    IsPrefill = true,
                    FullBlocksCaptured = CaptureNewlyFullBlocks(seq),
                    ForwardElapsedTicks = swPrefill.ElapsedTicks,
                };
            }

            // ---- Speculative decode step ----
            SpeculativeStepOutcome outcome = RunSpeculativeDecodeStep(
                context, seq, prevComputed, out int sampledToken, out List<int> accepted, out long decodeTicks);

            seq.LastLogits = outcome.NextLogits;
            int advanced = 1 + outcome.AcceptedCount;
            CompleteSpeculativeStepBookkeeping(seq, advanced);
            context.NextPosition = prevComputed + advanced;
            context.PendingNextToken = outcome.NextToken;
            PublishDrawnToken(seq, outcome.NextToken);

            int capturedBlocks = CaptureNewlyFullBlocks(seq);
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
                sampledToken = PendingVerifyDraw(seq, ctx.PendingNextToken);
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
                // Without penalties the adjustment is a no-op; leaving it out lets a head draft its whole window on
                // its device (IDraftHead.DraftChain), where no draft logits reach the host.
                adjustDraftLogits: penaltySampler.AdjustsDraftLogits
                    ? (draftLogits, pendingDrafts) => penaltySampler.ApplyPenalties(draftLogits, seq.OutputTokens, pendingDrafts)
                    : null,
                onDraftAccepted: d =>
                {
                    seq.AppendOutputToken(d);
                    acceptedTokens.Add(d);
                    if (IsStop(d)
                        || (stopRepetition
                            && RepetitionGuard.IsLooping(seq.OutputTokens, seq.OutputTokens.Count, out _, out _)))
                        stopped = true;
                },
                drawGreedy: CanUseGreedySpecVerify(seq, FirstTokenLogitsDump.IsArmed || _cbDebug) ? drawn =>
                {
                    if (stopped) return -1;
                    // Keep this check per row, AFTER onDraftAccepted updated the
                    // committed history, exactly as TokenSampler.Sample does.
                    if (!penaltySampler.TryGetForcedThinkingToken(seq.OutputTokens, out int forced))
                        forced = penaltySampler.CloseInsteadOfStop(seq.OutputTokens, drawn);
                    if (forced == drawn)
                        return drawn;
                    if ((uint)forced >= (uint)_model.Config.VocabSize)
                        throw new InvalidOperationException("Thinking end token is outside the model vocabulary.");
                    return forced;
                } : null);
            swDecode.Stop();
            accepted = acceptedTokens;
            elapsedTicks = swDecode.ElapsedTicks;
            return outcome;
        }

        // Full-logit consumers and prompts containing media (including images,
        // video or audio in earlier turns) retain the original solo verify path.
        // MediaSpans survives prefill/prefix reuse; pending embedding queues do not.
        // A future logprob/logit observer must join this gate rather than consume
        // a token-only outcome. The existing batched verify path is unchanged.
        internal static bool CanUseGreedySpecVerify(SequenceState seq, bool requiresLogits) =>
            seq.OutputTokens.Count > 0 && CanUseFixedGovernorGreedyProfile(seq, requiresLogits);
        // Arming precedes the first draw. History is a decode-time requirement,
        // while sampler/media/logit-consumer eligibility is already knowable.
        internal static bool CanUseFixedGovernorGreedyProfile(SequenceState seq, bool requiresLogits) =>
            seq.MediaSpans.Count == 0 && seq.GetOrCreateSampler().IsPlainGreedyArgmax && !requiresLogits;

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

        // Speculative turns (TrySpeculativeTurn): TS_SPEC_TURNS=0 never, =1 always (where the trunk can), unset where
        // the trunk prefers them (ISpeculativeTarget.SpecTurnsPreferred).
        private static readonly string? s_specTurns = Environment.GetEnvironmentVariable("TS_SPEC_TURNS");
        /// <summary>Speculative steps one sequence keeps the turn for before the next takes it (if still decoding).
        /// A turn's first step pays for switching holders (100-250 ms on Flash Next: the arena hands the holder back,
        /// the drafter catches up), so turning every step served 6-13 tokens/s against 25-35 batched; 64 steps
        /// (~180 tokens) amortize it.</summary>
        internal const int SpeculativeTurnSteps = 64;
        private readonly Dictionary<string, long> _turnLastServed = new(StringComparer.Ordinal);
        private string? _turnHolder;
        private int _turnSteps;
        private long _turnClock;
        private long _speculativeTurns;

        /// <summary>Solo speculative steps served as turns to concurrent decoders.</summary>
        internal long SpeculativeTurns => _speculativeTurns;

        /// <summary>
        /// Concurrent decoders on a trunk with no batched verify that prefers turns (<see cref="ISpeculativeTarget.SpecTurnsPreferred"/>):
        /// the turn's sequence takes one solo speculative step on its own holder while the others wait, and keeps the
        /// turn for <see cref="SpeculativeTurnSteps"/> steps; then the sequence served longest ago takes it. Each drafts
        /// over its own drafter state (<see cref="IPerRequestDraftState"/>), and its context catches up over the tokens
        /// it took in batched steps. What a sequence emits does not change, only when (in bursts). Returns false -
        /// nothing forwarded, the holder checked in by the batched step - when turns are off or this sequence cannot
        /// speculate now.
        /// </summary>
        private bool TrySpeculativeTurn(IBatchedPagedModel fused, List<ScheduledSequenceWork> decodeWork,
            List<SequenceStepResult> results)
        {
            if (s_specTurns == "0" || !SpeculationEnabledThisStep)
                return false;
            if (_model is not ISpeculativeTarget spec || !spec.SpeculationProfitable || !spec.SpecTrunkFollowsBoundCache
                || _model is not IPerRequestDraftState)
                return false;
            if (s_specTurns != "1" && !spec.SpecTurnsPreferred)
                return false;
            bool Eligible(SequenceState candidate) => CanSpeculate(candidate)
                && !_pendingRetainedFusedTruncations.ContainsKey(candidate.RequestId)
                && !_fusedSpecDeclined.Contains(candidate.RequestId)
                && fused.HasFusedSequenceCache(candidate.RequestId);
            ScheduledSequenceWork? chosen = null;
            if (_turnHolder != null && _turnSteps < SpeculativeTurnSteps)
                foreach (var work in decodeWork)
                    if (string.Equals(work.Sequence.RequestId, _turnHolder, StringComparison.Ordinal) && Eligible(work.Sequence))
                    { chosen = work; break; }
            if (chosen == null)
            {
                long oldest = long.MaxValue;
                foreach (var work in decodeWork)
                {
                    if (!Eligible(work.Sequence)) continue;
                    long last = _turnLastServed.TryGetValue(work.Sequence.RequestId, out long t) ? t : 0;
                    if (last < oldest) { oldest = last; chosen = work; }
                }
                if (chosen == null) { _turnHolder = null; return false; }
                _turnHolder = chosen.Sequence.RequestId;
                _turnSteps = 0;
            }
            var seq = chosen.Sequence;
            int prevComputed = seq.NumComputedTokens;
            if (!BindFusedHolder(fused, seq, results))
                return true;   // its result (a rewind that re-prefills) is the step's; the others wait one step
            if (!TrySpeculativeFusedDecode(seq, prevComputed, out SequenceStepResult? specResult))
            {
                _turnHolder = null;
                return false;
            }
            results.Add(specResult);
            _turnLastServed[seq.RequestId] = ++_turnClock;
            ++_turnSteps;
            ++_speculativeTurns;
            if (_cbDebug)
                Console.Error.WriteLine($"[cb] spec turn {seq.RequestId} step {_turnSteps}: {specResult.TokensForwarded} tokens");
            return true;
        }

        private void DisposeFusedSpecContext(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            _turnLastServed.Remove(requestId);
            if (string.Equals(_turnHolder, requestId, StringComparison.Ordinal)) _turnHolder = null;
            if (_fusedSpecCtx.Remove(requestId, out var ctx))
                ctx.Exec?.Dispose();
            _fusedSpecDeclined.Remove(requestId);
            (_model as IPerRequestDraftState)?.ReleaseDraftState(requestId);
        }

        /// <summary>
        /// A speculative decode step for a solo sequence served from its bound
        /// per-request fused holder. Returns false - nothing forwarded - when
        /// speculation is off, cannot serve this model, or cannot arm over this
        /// request; the caller then runs the plain fused step.
        /// </summary>
        private bool TrySpeculativeFusedDecode(
            SequenceState seq, int prevComputed, [NotNullWhen(true)] out SequenceStepResult? result)
        {
            result = null;
            if (!SpeculationEnabledThisStep || prevComputed <= 0 || !CanSpeculate(seq))
                return false;
            if (_model is not ISpeculativeTarget spec || !spec.SpeculationProfitable)
                return false;
            // Only a trunk that forwards on the BOUND cache can serve a holder; one
            // written against only the primary linear cache would unbind it.
            if (!spec.SpecTrunkFollowsBoundCache)
            {
                WarnSpeculationDeclinedOnce(
                    $"{_model.GetType().Name}'s speculative trunk forwards on its primary cache, not on a bound holder; "
                    + "turns served from a cached holder (every turn after a chat's first) decode plainly");
                return false;
            }
            // This is decode: any prepared media spans belong to the already
            // computed prompt and are retained only for replay/retry.
            // The bound holder must agree with the scheduler about the position.
            if (spec.CacheSeqLen != prevComputed)
                return false;
            (_model as IPerRequestDraftState)?.BindDraftState(seq.RequestId);
            if (!TryGetFusedSpecContext(spec, seq, prevComputed, out SpecSeqContext? ctx))
                return false;

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

        private long _batchedSpecSteps;

        /// <summary>Batched speculative steps served (each verifies every decoding
        /// sequence's window in one trunk forward).</summary>
        internal long BatchedSpeculativeSteps => _batchedSpecSteps;

        /// <summary>
        /// One speculative step for SEVERAL decoding sequences on their per-request
        /// holders: each drafts a window with its own drafter state, the trunk
        /// verifies every window in one forward (<see cref="IBatchedSpeculativeTarget"/>),
        /// and each sequence keeps its accepted prefix. Without this, a second request
        /// in flight dropped every request to one token per step of the batched
        /// decode while a lone request verified a whole window per step.
        /// Greedy only. Returns false with nothing consumed when any sequence or the
        /// trunk declines; the caller then runs the batched decode.
        /// </summary>
        private bool TryBatchedSpeculativeDecode(List<ScheduledSequenceWork> decodeWork, List<SequenceStepResult> results)
        {
            int n = decodeWork.Count;
            if (!SpeculationEnabledThisStep || n < 1)
                return false;
            if (_model is not ISpeculativeTarget spec || _model is not IBatchedSpeculativeTarget batched
                || !spec.SpeculationProfitable || !spec.SpecTrunkFollowsBoundCache)
                return false;
            int window = batched.BatchedSpecWindow(n);
            if (window < 2)
                return false;
            // Sequences that sample join when the trunk returns every row's logits: each draws its
            // rows with its own sampler, as its solo step does (a draft is kept while the draw equals it).
            bool sampledVerify = false;
            for (int i = 0; i < n; i++)
            {
                var seq = decodeWork[i].Sequence;
                int prev = seq.NumComputedTokens;
                if (prev <= 0 || !CanSpeculate(seq) || batched.FusedCacheSeqLen(seq.RequestId) != prev)
                    return false;
                if (!seq.GetOrCreateSampler().IsPlainGreedyArgmax)
                {
                    if (!batched.SupportsBatchedSpecLogits)
                        return false;
                    sampledVerify = true;
                }
            }

            using var scope = batched.BeginBatchedSpeculation();
            var ctxs = new SpecSeqContext[n];
            for (int i = 0; i < n; i++)
            {
                var seq = decodeWork[i].Sequence;
                batched.BindDraftState(seq.RequestId);
                if (!TryGetFusedSpecContext(spec, seq, seq.NumComputedTokens, out ctxs[i]))
                    return false;
            }
            // Drafts are proposed for the widest window; the verify then takes four rows
            // when every sequence's confidence-gated draft fits them, the whole window
            // otherwise. A row still costs its column of every projection and FFN, but
            // at three or four sequences eight rows cost ~10% more than four, while a
            // confident block keeps drafts the narrow window would have cut. (An
            // acceptance average chose the window before drafting and kept it narrow
            // through runs of long accepted blocks.)
            int laneBudget = 0;
            foreach (var ctx in ctxs)
                laneBudget = Math.Max(laneBudget, ctx.Exec.MaxDraftTokens);
            window = Math.Min(window, 1 + laneBudget);
            // Every sequence's window, padding included, must fit its cache without growth.
            for (int i = 0; i < n; i++)
                window = Math.Min(window, batched.FusedCacheFreeRows(decodeWork[i].Sequence.RequestId));
            if (window < 2)
                return false;
            var requestIds = new string[n];
            var positions = new int[n];
            var anchors = new int[n];
            var drafts = new List<int>[n];
            var kMaxes = new int[n];
            bool stopRepetition = _scheduler.Config.StopRepetition;
            bool IsStop(int token) => _model.Tokenizer != null && _model.Tokenizer.IsEos(token);
            // The verify-drawn token when one is pending, else the stash or a sample;
            // nothing is consumed until the verify has succeeded. A lane whose anchor
            // ends it (a stop token, or the last token its budget allows) emits it
            // without a forward: verifying it would compute a row nothing reads.
            var peeked = new int[n];
            List<(SequenceState Seq, SpecSeqContext Ctx, int Anchor)>? ending = null;
            for (int i = 0; i < n; i++)
            {
                var seq = decodeWork[i].Sequence;
                peeked[i] = ctxs[i].PendingNextToken >= 0 ? PendingVerifyDraw(seq, ctxs[i].PendingNextToken) : PeekPendingOrSample(seq);
                if (IsStop(peeked[i]) || seq.OutputTokens.Count + 1 >= seq.MaxNewTokens)
                    (ending ??= new()).Add((seq, ctxs[i], peeked[i]));
            }
            if (ending != null)
            {
                var liveWork = new List<ScheduledSequenceWork>(n);
                var liveCtxs = new List<SpecSeqContext>(n);
                var livePeeked = new List<int>(n);
                for (int i = 0; i < n; i++)
                    if (!ending.Exists(e => ReferenceEquals(e.Seq, decodeWork[i].Sequence)))
                    {
                        liveWork.Add(decodeWork[i]);
                        liveCtxs.Add(ctxs[i]);
                        livePeeked.Add(peeked[i]);
                    }
                if (liveWork.Count == 0)
                {
                    foreach (var (seq, ctx, anchor) in ending)
                        EmitEndingAnchor(seq, ctx, anchor, results);
                    return true;
                }
                decodeWork = liveWork;
                ctxs = liveCtxs.ToArray();
                peeked = livePeeked.ToArray();
                n = liveWork.Count;
                requestIds = new string[n];
                positions = new int[n];
                anchors = new int[n];
                drafts = new List<int>[n];
                kMaxes = new int[n];
            }
            long started = Stopwatch.GetTimestamp();
            for (int i = 0; i < n; i++)
            {
                var seq = decodeWork[i].Sequence;
                int prev = seq.NumComputedTokens;
                var ctx = ctxs[i];
                requestIds[i] = seq.RequestId;
                positions[i] = prev;
                anchors[i] = peeked[i];
                // Same window caps as the solo step: the token budget after the anchor,
                // the blocks the scheduler reserved, no window from a stop token, and
                // none from a token the repetition guard is about to stop at.
                int kMax = Math.Min(window - 1, Math.Min(
                    seq.MaxNewTokens - seq.OutputTokens.Count - 1,
                    seq.BlockTable.FreeSlotsInCurrentBlocks - 1));
                if (IsStop(anchors[i]))
                    kMax = 0;
                if (kMax > 0 && stopRepetition && (seq.SamplingConfig?.StopRepetition ?? true))
                {
                    seq.OutputTokens.Add(anchors[i]);
                    bool looping = RepetitionGuard.IsLooping(seq.OutputTokens, seq.OutputTokens.Count, out _, out _);
                    seq.OutputTokens.RemoveAt(seq.OutputTokens.Count - 1);
                    if (looping)
                        kMax = 0;
                }
                drafts[i] = new List<int>(window);
                kMaxes[i] = kMax;
            }

            // Draft every sequence's block in one pass when the drafter can; otherwise
            // (or if the batched pass declines) one draft per sequence.
            bool draftedTogether = false;
            int blockWidth = batched.BatchDraftBlockWidth;
            if (blockWidth > 0)
            {
                var lanes = new List<int>(n);
                var carries = new List<float[]>(n);
                var caps = new List<int>(n);
                for (int i = 0; i < n; i++)
                    if (ctxs[i].Exec.TryPrepareBatchDraft(positions[i], kMaxes[i], out float[]? carry, out int cap))
                    {
                        lanes.Add(i);
                        carries.Add(carry);
                        caps.Add(cap);
                    }
                if (lanes.Count >= 2)
                {
                    int m = lanes.Count;
                    var laneIds = new string[m];
                    var laneAnchors = new int[m];
                    var lanePositions = new int[m];
                    var blockTokens = new int[m][];
                    var blockConf = new float[m][];
                    for (int j = 0; j < m; j++)
                    {
                        laneIds[j] = requestIds[lanes[j]];
                        laneAnchors[j] = anchors[lanes[j]];
                        lanePositions[j] = positions[lanes[j]];
                        blockTokens[j] = new int[blockWidth];
                        blockConf[j] = new float[blockWidth];
                    }
                    long tDraft0 = Stopwatch.GetTimestamp();
                    draftedTogether = batched.TryDraftBatch(laneIds, laneAnchors, carries.ToArray(), lanePositions,
                        blockTokens, blockConf);
                    long share = (Stopwatch.GetTimestamp() - tDraft0) / m;
                    if (draftedTogether)
                        for (int j = 0; j < m; j++)
                            ctxs[lanes[j]].Exec.AcceptBatchDraft(blockTokens[j], blockConf[j], caps[j], drafts[lanes[j]], share);
                }
            }
            if (!draftedTogether)
            {
                for (int i = 0; i < n; i++)
                {
                    batched.BindDraftState(requestIds[i]);
                    ctxs[i].Exec.ProposeForBatch(anchors[i], positions[i], kMaxes[i], drafts[i]);
                }
            }
            int longest = 0;
            for (int i = 0; i < n; i++)
                longest = Math.Max(longest, drafts[i].Count);
            if (window > NarrowBatchedWindow && 1 + longest <= NarrowBatchedWindow)
                window = NarrowBatchedWindow;
            var tokens = new int[n * window];
            for (int i = 0; i < n; i++)
            {
                // Pad short windows with the last real token; padded rows are never accepted.
                int last = anchors[i];
                for (int t = 0; t < window; t++)
                {
                    if (t == 0) tokens[i * window] = anchors[i];
                    else if (t - 1 < drafts[i].Count) last = tokens[i * window + t] = drafts[i][t - 1];
                    else tokens[i * window + t] = last;
                }
            }

            int feat = spec.SpecFeatureSize;
            long featNeeded = (long)n * window * feat;
            if (_batchedSpecFeatures == null || _batchedSpecFeatures.LongLength < featNeeded)
                _batchedSpecFeatures = new float[featNeeded];
            var drawn = new int[n * window];
            long drafted = Stopwatch.GetTimestamp();
            int vocab = _model.Config.VocabSize;
            if (sampledVerify)
            {
                long logitsNeeded = (long)n * window * vocab;
                if (_batchedSpecLogits == null || _batchedSpecLogits.LongLength < logitsNeeded)
                    _batchedSpecLogits = new float[logitsNeeded];
                if (!batched.TryBatchedSpecVerifyLogits(requestIds, positions, window, tokens, _batchedSpecLogits, _batchedSpecFeatures))
                    return false;
            }
            else if (!batched.TryBatchedSpecVerify(requestIds, positions, window, tokens, drawn, _batchedSpecFeatures))
                return false;
            long verified = Stopwatch.GetTimestamp();
            long verifyTicks = verified - started;

            // Accept per sequence exactly as the solo step does: a draft is kept while
            // it equals its row's draw, and nothing is accepted past a stop. A row's draw
            // is what the solo step's drawGreedy returns: a forced reasoning-close token
            // over the device argmax, and a stop inside open reasoning turned into one.
            int RowDraw(SequenceState seq, int raw)
            {
                var sampler = seq.GetOrCreateSampler();
                if (!sampler.TryGetForcedThinkingToken(seq.OutputTokens, out int row))
                    row = sampler.CloseInsteadOfStop(seq.OutputTokens, raw);
                if (row != raw && (uint)row >= (uint)_model.Config.VocabSize)
                    throw new InvalidOperationException("Thinking end token is outside the model vocabulary.");
                return row;
            }
            var kept = new int[n];
            var accepted = new List<int>[n];
            var nexts = new int[n];
            // The anchors, in order: each sequence's pending draw, or a draw from its logits.
            for (int i = 0; i < n; i++)
            {
                var seq = decodeWork[i].Sequence;
                var ctx = ctxs[i];
                if (ctx.PendingNextToken >= 0)
                {
                    ctx.PendingNextToken = -1;
                    seq.PendingDeviceToken = null;
                }
                else
                {
                    TakePendingOrSample(seq);
                }
                seq.AppendOutputToken(anchors[i]);
            }
            if (sampledVerify && (_batchedSpecRows == null || _batchedSpecRows.Length < n))
                Array.Resize(ref _batchedSpecRows, n);
            // Each sequence's drafts against its own rows. A sequence's acceptance touches only its own state (output,
            // sampler, grammar), so sequences that sample draw in parallel: a sampled row is a copy, penalties and a
            // top-k pass over the whole vocabulary, and four sequences' rows one after another held the GPU idle.
            void Accept(int i)
            {
                var seq = decodeWork[i].Sequence;
                bool seqStopRepetition = stopRepetition && (seq.SamplingConfig?.StopRepetition ?? true);
                bool stopped = IsStop(anchors[i]);
                var acc = new List<int>();
                int m = 0;
                // A row is drawn once: a sampled row advances the sequence's sampler.
                int Draw(int row)
                {
                    if (!sampledVerify)
                        return RowDraw(seq, drawn[i * window + row]);
                    float[] buf = _batchedSpecRows![i] is { } b && b.Length == vocab ? b : (_batchedSpecRows[i] = new float[vocab]);
                    Array.Copy(_batchedSpecLogits!, ((long)i * window + row) * vocab, buf, 0, vocab);
                    var sampler = seq.GetOrCreateSampler();
                    if (!sampler.IsPlainGreedyArgmax)
                        return sampler.Sample(buf, seq.OutputTokens);
                    int best = 0;
                    for (int v = 1; v < vocab; v++)
                        if (buf[v] > buf[best]) best = v;
                    return RowDraw(seq, best);
                }
                int draw = stopped ? -1 : Draw(0);
                while (!stopped && m < drafts[i].Count && draw == drafts[i][m])
                {
                    int d = drafts[i][m];
                    seq.AppendOutputToken(d);
                    acc.Add(d);
                    m++;
                    if (IsStop(d) || (seqStopRepetition
                            && RepetitionGuard.IsLooping(seq.OutputTokens, seq.OutputTokens.Count, out _, out _)))
                        stopped = true;
                    else
                        draw = Draw(m);
                }
                kept[i] = m + 1;
                accepted[i] = acc;
                // The row after the kept ones: its draw is the next token (drawn above unless a stop ended the run).
                nexts[i] = stopped ? Draw(m) : draw;
            }
            if (sampledVerify && n > 1)
            {
                try
                {
                    Parallel.For(0, n, Accept);
                }
                catch (AggregateException ae) when (ae.InnerExceptions.Count == 1)
                {
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ae.InnerExceptions[0]).Throw();
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                    Accept(i);
            }

            batched.CommitBatchedSpecVerify(requestIds, positions, kept);
            long committed = Stopwatch.GetTimestamp();

            long ticksPerSequence = (Stopwatch.GetTimestamp() - started) / n;
            for (int i = 0; i < n; i++)
            {
                var seq = decodeWork[i].Sequence;
                var ctx = ctxs[i];
                int m = kept[i] - 1;
                batched.BindDraftState(seq.RequestId);
                ctx.Exec.CommitBatch(anchors[i], positions[i], drafts[i], m,
                    _batchedSpecFeatures, (long)i * window * feat, verifyTicks / n);
                int next = nexts[i];
                seq.LastLogits = null;
                seq.AdvanceComputedTokens(kept[i]);
                ctx.NextPosition = positions[i] + kept[i];
                ctx.PendingNextToken = next;
                PublishDrawnToken(seq, next);
                if (!seq.FirstTokenAt.HasValue)
                    seq.FirstTokenAt = DateTime.UtcNow;
                results.Add(new SequenceStepResult
                {
                    Sequence = seq,
                    TokensForwarded = kept[i],
                    SampledToken = anchors[i],
                    ExtraTokens = accepted[i].Count > 0 ? accepted[i] : null,
                    IsPrefill = false,
                    FullBlocksCaptured = 0,
                    ForwardElapsedTicks = ticksPerSequence,
                });
            }
            if (ending != null)
                foreach (var (seq, ctx, anchor) in ending)
                    EmitEndingAnchor(seq, ctx, anchor, results);
            _batchedSpecSteps++;
            if (s_batchedSpecLog)
            {
                static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
                long done = Stopwatch.GetTimestamp();
                Console.Error.WriteLine(
                    $"[batched-spec] n={n} window={window} kept=[{string.Join(",", kept)}] draft={Ms(drafted - started):F1} " +
                    $"verify={Ms(verified - drafted):F1} commit={Ms(committed - verified):F1} drafter={Ms(done - committed):F1} ms");
            }
            return true;
        }

        /// <summary>Emit a batched-speculation lane's anchor that ends its sequence (a stop
        /// token or the last of its budget) without forwarding it; the engine finishes the
        /// sequence on it, with the cache one row short of the output.</summary>
        private void EmitEndingAnchor(SequenceState seq, SpecSeqContext ctx, int anchor, List<SequenceStepResult> results)
        {
            if (ctx.PendingNextToken >= 0)
            {
                ctx.PendingNextToken = -1;
                seq.PendingDeviceToken = null;
            }
            else
            {
                TakePendingOrSample(seq);
            }
            seq.AppendOutputToken(anchor);
            seq.LastLogits = null;
            if (!seq.FirstTokenAt.HasValue)
                seq.FirstTokenAt = DateTime.UtcNow;
            results.Add(new SequenceStepResult
            {
                Sequence = seq,
                TokensForwarded = 0,
                SampledToken = anchor,
                IsPrefill = false,
                FullBlocksCaptured = 0,
            });
        }

        private static readonly bool s_batchedSpecLog =
            Environment.GetEnvironmentVariable("TS_BATCHED_SPEC_LOG") == "1";

        private float[]? _batchedSpecFeatures;
        // A sampled batched verify's rows' logits, and per sequence the row handed to its sampler.
        private float[]? _batchedSpecLogits;
        private float[]?[]? _batchedSpecRows;

        // The verify window a step falls back to when every gated draft fits it.
        private const int NarrowBatchedWindow = 4;

        /// <summary>Run <see cref="TryBatchedSpeculativeDecode"/> for a step's decode
        /// subset. True when the subset was served (or failed and was marked failed:
        /// a verify or commit that throws may have written their caches, so the
        /// requests fail instead of being re-served from that state).</summary>
        private bool TryRunBatchedSpeculativeSubset(
            List<ScheduledSequenceWork> decodeWork, List<SequenceStepResult> results, ref HashSet<string>? handled)
        {
            try
            {
                if (!TryBatchedSpeculativeDecode(decodeWork, results))
                    return false;
            }
            catch (Exception ex)
            {
                var failed = new HashSet<string>(decodeWork.Select(work => work.Sequence.RequestId), StringComparer.Ordinal);
                results.RemoveAll(result => failed.Contains(result.Sequence.RequestId));
                _logger.LogError(ex,
                    "Batched speculative decode failed for {Count} sequences; failing the affected requests.",
                    decodeWork.Count);
                foreach (var work in decodeWork)
                {
                    work.Sequence.Error = ex;
                    results.Add(new SequenceStepResult { Sequence = work.Sequence, Error = ex });
                }
                handled = failed;
                return true;
            }
            handled = new HashSet<string>(decodeWork.Select(work => work.Sequence.RequestId), StringComparer.Ordinal);
            return true;
        }

        /// <summary>
        /// The speculative context of a sequence served from its per-request fused
        /// holder: resumed over tokens the holder advanced plainly since its last
        /// speculative step, or armed over the holder's committed tokens. False when
        /// speculation cannot arm for this request.
        /// </summary>
        private bool TryGetFusedSpecContext(ISpeculativeTarget spec, SequenceState seq, int prevComputed,
            [NotNullWhen(true)] out SpecSeqContext? ctx)
        {
            if (_fusedSpecCtx.TryGetValue(seq.RequestId, out ctx) && ctx.NextPosition < prevComputed
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
            if (_fusedSpecCtx.TryGetValue(seq.RequestId, out ctx) && ctx.NextPosition == prevComputed)
                return true;
            if (ctx != null)
            {
                // Something forwarded past this context (a truncation, a head that
                // cannot resume); start over from the holder's tokens.
                _fusedSpecCtx.Remove(seq.RequestId);
                ctx.Exec?.Dispose();
                ctx = null;
            }
            if (_fusedSpecDeclined.Contains(seq.RequestId))
                return false;
            var speculator = SpeculatorRegistry.Create(spec, Speculation, out string? declineReason);
            if (speculator == null)
            {
                WarnSpeculationDeclinedOnce(declineReason);
                _fusedSpecDeclined.Add(seq.RequestId);
                return false;
            }
            var exec = new SpeculativeExecution(spec, speculator, trunk: null, SharedGovernorFor(spec, speculator, seq));
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
                "Speculative decoding armed for {EngineRequestId} (algorithm={Algorithm}, maxDraft={MaxDraft}, "
                + "pMin={PMin}, trunk=fused holder over {Committed} committed tokens, adaptive={Adaptive}, adaptiveSource={AdaptiveSource})",
                seq.RequestId, speculator.Describe(), exec.MaxDraftTokens, exec.MinDraftProb, prevComputed,
                exec.AdaptiveSpeculation, exec.Governor.SelectionSource);
            return true;
        }

        /// <summary>Per-step bookkeeping for MTP steps (which may advance more
        /// than one token). Linear trunk mirrors the plain per-seq path
        /// (owner counters + live-cache tracking + the caller's block
        /// capture); batched trunk mirrors <see cref="ExecuteStepBatched"/>
        /// (K/V lives in the model's paged storage, blocks get hash-registered
        /// for prefix sharing).</summary>
        private void CompleteSpeculativeStepBookkeeping(SequenceState seq, int tokensForwarded)
        {
            seq.AdvanceComputedTokens(tokensForwarded);
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

        /// <summary>The first condition of <see cref="EnsureOwnership"/>'s live-continuation gate that
        /// failed, for the line that retracts reuse admission announced. It used to blame "another sequence"
        /// for every failure, including a request that simply carried explicit cache breakpoints.</summary>
        private string LiveContinuationGateFailure(SequenceState seq)
        {
            if (_currentOwner != null) return "another sequence owns the model";
            if (!_liveCacheValid || _liveCacheSeq == null) return "another sequence took the cache";
            if (_liveCacheSeq.CacheBreakpoints != null || seq.CacheBreakpoints != null)
                return "a request with explicit cache breakpoints does not continue the live cache";
            if (!ScopeAllows(_liveCacheSeq.CacheScope, seq.CacheScope) && seq.NumComputedTokens > seq.SharedPrefixTokens)
                return "the live cache belongs to another conversation";
            if (seq.NumComputedTokens <= 0) return "nothing was claimed";
            if (_liveCacheLen < seq.NumComputedTokens)
                return $"the live cache holds {_liveCacheLen} of the {seq.NumComputedTokens} claimed tokens";
            if (!_model.SupportsKVCacheTruncation) return "the model cannot rewind its live cache";
            return "a media span cuts the claimed prefix";
        }

        /// <summary>Why the last <see cref="ComputeFusedContinuationLcp"/> found no
        /// usable retained holder or shared-prefix checkpoint, or null when it did.</summary>
        public string? LastFusedContinuationDeclineReason => _lastFusedDeclineReason;

        /// <summary>True when the loaded model serves concurrent decode through
        /// per-request fused holders and explicitly supports retaining/re-keying a
        /// completed holder. Do not infer this from the pooled-prefix cap: Qwen's
        /// byte snapshots are deliberately not cross-request reusable, but its full
        /// request-owned attention + recurrent-state holder is.</summary>
        private bool ModelUsesRetainableFusedCache()
            => ExecutionOptions.FromEnvironment().RetainedFusedCacheBudget != 0
            && _model is IBatchedPagedModel f
            && f.SupportsPerSequenceFusedForward
            && f.SupportsRetainedFusedCache;

        /// <summary>Whether shared-prefix checkpoints are in use for this model, so the
        /// scheduler ends prefill chunks at the boundary the chat layer marked.</summary>
        public bool PrefixCheckpointsSupported => RadixCache?.CheckpointsSupported == true;

        /// <summary>
        /// After a prefill step: if this sequence has just reached a checkpoint boundary
        /// of its prompt, let the prefix cache copy the model's state there. Called on
        /// every per-sequence path with the sequence's cache active (the primary cache on
        /// the linear path, its own holder on the fused one).
        /// </summary>
        private void MaybeCheckpointSharedPrefix(SequenceState seq) => RadixCache?.CaptureCheckpoint(seq);

        /// <summary>How many leading prompt tokens of <paramref name="seq"/> the prefix
        /// cache can resume, or 0 (with <see cref="LastFusedContinuationDeclineReason"/>
        /// saying why). Invoked by the scheduler at admission (same worker thread as the
        /// executor).</summary>
        public int ComputeFusedContinuationLcp(SequenceState seq)
        {
            _lastFusedDeclineReason = null;
            if (seq == null || RadixCache == null) return 0;
            int reused = RadixCache.ComputeReusablePrefix(seq);
            _lastFusedDeclineReason = reused > 0 ? null : RadixCache.LastDeclineReason ?? "no resumable radix prefix";
            if (_cbDebug)
                Console.Error.WriteLine($"[cb] admit {seq.RequestId} scope={seq.CacheScope ?? "<none>"} prompt={seq.PromptTokens.Count} " +
                    $"reuse={reused} {RadixCache.LastPlanDescription}{(reused > 0 ? "" : $" ({_lastFusedDeclineReason})")}");
            return reused;
        }

        /// <summary>
        /// Where shared-prefix checkpoints outlive the process, or null for nowhere.
        /// Set by the host; read on the engine thread at admission (restore) and at
        /// the prefix boundary (save). See <see cref="IPrefixCheckpointStore"/>.
        /// </summary>
        public IPrefixCheckpointStore? PrefixCheckpointStore
        {
            get => RadixCache?.CheckpointStore;
            set
            {
                if (RadixCache != null) RadixCache.CheckpointStore = value;
            }
        }

        /// <summary>Adopt the prefix <see cref="ComputeFusedContinuationLcp"/> found for
        /// <paramref name="seq"/>: the prefix cache materializes it for this request (a
        /// retained holder re-keyed to it, a checkpoint cloned, pages bound) and marks the
        /// reused prefix. A holder that covers more than the prompt reuses is rewound on
        /// the request's first fused step. Returns false when nothing was adopted.
        /// Invoked by the scheduler at admission.</summary>
        public bool TryAdoptFusedContinuation(SequenceState seq, int lcp)
        {
            if (RadixCache == null) return false;
            // Reserve rewind metadata before materialization transfers ownership.
            _pendingRetainedFusedTruncations.EnsureCapacity(_pendingRetainedFusedTruncations.Count + 1);
            bool adopted = RadixCache.TryAdopt(seq, lcp,
                (sequence, tokens) => _pendingRetainedFusedTruncations[sequence.RequestId] = tokens);
            // The plan found a prefix, so the admission-time reason was cleared; say why adoption failed.
            if (!adopted)
                _lastFusedDeclineReason = RadixCache.LastDeclineReason ?? $"adopting the planned {lcp}-token prefix failed";
            return adopted;
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
            RadixCache?.ReleaseRequest(requestId);
        }

        /// <summary>Called by the engine when a sequence leaves the scheduler, BEFORE
        /// the model's <see cref="IBatchedPagedModel.OnSequenceReleased"/>. When the
        /// sequence finished cleanly, the prefix cache retains its complete
        /// request-owned continuation state for the conversation's next turn instead
        /// of letting the model dispose it. Returns true when the state was retained
        /// (so the subsequent model release no-ops for it).</summary>
        public bool TryRetainReleasedFusedCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return false;
            DisposeFusedSpecContext(requestId);
            _pendingRetainedFusedTruncations.Remove(requestId);
            if (!_fusedSeqById.TryGetValue(requestId, out var seq))
            {
                RadixCache?.ReleaseRequest(requestId);
                return false;
            }
            _fusedSeqById.Remove(requestId);

            if (RadixCache != null)
            {
                try
                {
                    bool cleanFinish = seq.Status is SequenceStatus.FinishedStopped or SequenceStatus.FinishedLengthCapped
                        || (seq.Status == SequenceStatus.FinishedAborted && seq.Error == null
                            && seq.NumComputedTokens >= seq.NumTotalTokens);
                    bool primary = _liveCacheValid && ReferenceEquals(_liveCacheSeq, seq)
                        && !(_model is IBatchedPagedModel holder && holder.HasFusedSequenceCache(requestId));
                    bool retained = cleanFinish && RadixCache.RetainFinished(seq, primary);
                    if (_cbDebug)
                        Console.Error.WriteLine($"[cb] release {requestId} status={seq.Status} computed={seq.NumComputedTokens}/{seq.NumTotalTokens} " +
                            $"primary={primary} retained={retained} prefixCache=[{RadixCache.Tree.Cached}] poolSlabBytes={_pool.Storage.AllocatedBytes}");
                    if (retained && primary)
                    {
                        _liveCacheValid = false;
                        _currentOwner = null;
                        _ownerTokensInModel = 0;
                        _ownerForwardedTokens = 0;
                    }
                    return retained;
                }
                finally { RadixCache.ReleaseRequest(seq); }
            }
            return false;
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
            // tokens still resident in the model's live KV cache (the radix prefix
            // cache's KeepPrimary plan). Keep the cache as-is and continue
            // from the reused prefix - no reset, no pooled inject. This is the only
            // way to reuse a prefix longer than a sliding-window model's window
            // without the lossy circular-cache snapshot reconstruction.
            if (seq.UsesLiveCacheContinuation)
            {
                if (_currentOwner == null
                    && _liveCacheValid
                    && _liveCacheSeq != null
                    && _liveCacheSeq.CacheBreakpoints == null
                    && seq.CacheBreakpoints == null
                    && (ScopeAllows(_liveCacheSeq.CacheScope, seq.CacheScope)
                        || seq.NumComputedTokens <= seq.SharedPrefixTokens)
                    && seq.NumComputedTokens > 0
                    && ClampReuseToMedia(seq.NumComputedTokens, seq, _liveCacheSeq.MediaSpans) == seq.NumComputedTokens
                    && _liveCacheLen >= seq.NumComputedTokens
                    && (_liveCacheLen == seq.NumComputedTokens || _model.SupportsKVCacheTruncation))
                {
                    // The scheduler may have matched a prefix shorter than what the cache
                    // holds, because the previous turn ended on a control token the
                    // template does not re-render (the prefix cache's rewind cap bounds it).
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
                                "Live-cache continuation for {EngineRequestId} needed a {Rewind}-token rewind the " +
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
                        "Live-cache continuation for {EngineRequestId} was no longer valid at execution time " +
                        "({Why}); its prompt re-prefills after all.",
                        seq.RequestId, LiveContinuationGateFailure(seq));
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
                // Rebuild the incoming sequence in place: inject what its blocks can
                // faithfully restore (InjectAllBlocks stops at the model's pooled cap -
                // Gemma 4's window, Muse-Glimmer's ring - at any block it never captured,
                // and at nothing for a model that cannot take injected state) and forward
                // the rest again. The step about to run was planned, and its blocks
                // reserved, at THIS position, so the position, the blocks and a decoder's
                // pending logits must all survive. The old "cannot inject" branch
                // (ResetForPreemption + BlockTable.Clear) dropped them while the step
                // still forwarded its planned chunk: "AdvanceTokens(256) wants 1 blocks but
                // only 0 are allocated" for a Muse-Glimmer multi-agent parent that resumed
                // past its 4352-token cap after its children had held the model.
                InjectPrefixOrRecompute(seq, seq.NumComputedTokens);
                _ownerTokensInModel = seq.NumComputedTokens;
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

            int[] chunk = PrefillTokens(seq, work);
            // The prompt's next chunk, of the same width, as a hint: a model can prepare its host work while
            // this chunk computes (IModelArchitecture.PrepareNextPrefillChunk; it checks the chunk it is given).
            int nextStart = seq.NumComputedTokens + chunk.Length;
            int nextLen = Math.Min(chunk.Length, seq.PromptTokens.Count - nextStart);
            if (nextLen > 0)
            {
                var next = new int[nextLen];
                for (int i = 0; i < nextLen; i++)
                    next[i] = seq.TokenAt(nextStart + i);
                _model.PrepareNextPrefillChunk(next, nextStart);
            }
            return chunk;
        }

        private static int[] PrefillTokens(SequenceState seq, ScheduledSequenceWork work)
        {
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
        /// Give memory back while it is still ours to give: the prefix cache evicts what
        /// it can spare, then the model drops whatever it parked for reuse. Engine thread
        /// only, between steps. Returns a one-line account for the log.
        /// </summary>
        public string TrimIdleMemory()
        {
            if (RadixCache != null) return RadixCache.TrimIdleMemory();
            _model.TrimIdleMemory();
            return "prefix caching is off; the model released what it parked for reuse";
        }

        /// <summary>Consume a token the batched greedy path sampled on-device
        /// last step (bit-equivalent to re-sampling the logits it summarizes),
        /// falling back to host sampling from LastLogits. Any position drift —
        /// preemption, recompute, rollback — invalidates the stash.</summary>
        private static int TakePendingOrSample(SequenceState seq)
        {
            // Every path's first token passes through here or PeekPendingOrSample, which is
            // what makes this the one place the first-token dump can be taken for all of them.
            if (seq.EarlyFirstToken >= 0 && seq.OutputTokens.Count == 0)
            {
                seq.PendingDeviceToken = null;
                return seq.EarlyFirstToken;
            }
            var dump = seq.OutputTokens.Count == 0 && FirstTokenLogitsDump.IsArmed
                ? CaptureFirstTokenLogits(seq) : null;
            if (seq.GetOrCreateSampler().TryGetForcedThinkingToken(seq.OutputTokens, out int thinkingEnd))
            {
                seq.PendingDeviceToken = null;
                dump?.Complete(thinkingEnd, "forced-thinking-end");
                return thinkingEnd;
            }
            if (seq.PendingDeviceToken.HasValue)
            {
                int t = seq.PendingDeviceToken.Value;
                seq.PendingDeviceToken = null;
                if (seq.PendingDevicePosition == seq.NumComputedTokens)
                {
                    // A device draw meets the same reasoning-close rule as a host sample.
                    t = seq.GetOrCreateSampler().CloseInsteadOfStop(seq.OutputTokens, t);
                    dump?.Complete(t, "device-pending");
                    return t;
                }
            }
            int sampled = SampleFromLogits(seq);
            dump?.Complete(sampled, "sampled");
            return sampled;
        }

        /// <summary>Non-destructive form of <see cref="TakePendingOrSample"/> for
        /// the batched fast path's peek: a stash that is still valid stays on the
        /// sequence until a successful step replaces (or clears) it, so a decline
        /// leaves the fallback loop a token source.</summary>
        private static int PeekPendingOrSample(SequenceState seq)
        {
            if (seq.EarlyFirstToken >= 0 && seq.OutputTokens.Count == 0)
                return seq.EarlyFirstToken;
            // A peeked first token is the one the step emits (a decline stashes it for the
            // fallback), so the dump is taken here as well; it is written once either way.
            var dump = seq.OutputTokens.Count == 0 && FirstTokenLogitsDump.IsArmed
                ? CaptureFirstTokenLogits(seq) : null;
            if (seq.GetOrCreateSampler().TryGetForcedThinkingToken(seq.OutputTokens, out int thinkingEnd))
            {
                dump?.Complete(thinkingEnd, "forced-thinking-end");
                return thinkingEnd;
            }
            if (seq.PendingDeviceToken.HasValue)
            {
                if (seq.PendingDevicePosition == seq.NumComputedTokens)
                {
                    // The close it may start is history-derived, so a later Take repeats it.
                    int t = seq.GetOrCreateSampler().CloseInsteadOfStop(seq.OutputTokens, seq.PendingDeviceToken.Value);
                    dump?.Complete(t, "device-pending");
                    return t;
                }
                seq.PendingDeviceToken = null;   // stale: position moved under it
            }
            int sampled = SampleFromLogits(seq);
            dump?.Complete(sampled, "sampled");
            return sampled;
        }

        /// <summary>The token the previous verify drew for this position, unless a reasoning close
        /// requested since (the repetition guard, the templated-loop detector: both ask after the
        /// step) forces another. The draw was never forwarded, so replacing it is free, and the close
        /// then writes its whole hand-over text rather than starting after a token that left it.</summary>
        private static int PendingVerifyDraw(SequenceState seq, int pending)
            => seq.GetOrCreateSampler().TryGetForcedThinkingToken(seq.OutputTokens, out int forced) ? forced : pending;

        /// <summary>Draws the first output token of a sequence whose prompt has just been
        /// prefilled, so the engine can stream it before the step that forwards it. The draw
        /// is the one that step makes (<see cref="SequenceState.EarlyFirstToken"/>); false
        /// when the sequence already has output or no logits or device draw to take it from.</summary>
        internal static bool TryDrawEarlyFirstToken(SequenceState seq, out int token)
        {
            token = -1;
            if (seq.OutputTokens.Count != 0 || seq.EarlyFirstToken >= 0 || seq.NumUncomputedTokens != 0)
                return false;
            bool device = seq.PendingDeviceToken.HasValue && seq.PendingDevicePosition == seq.NumComputedTokens;
            if (seq.LastLogits == null && !device)
                return false;
            token = PeekPendingOrSample(seq);
            seq.EarlyFirstToken = token;
            return true;
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

                // Cached radix pages are immutable. An ownership swap only needs
                // to refresh the request's private tail, never shared full slabs.
                if (tokensInBlock == _blockSize && block.HoldsSnapshotBytes
                    && RadixCache != null && RadixCache.Tree.TryGetBlockOwner(block, out _))
                    continue;

                // A recurrent full block was captured at the exact Forward
                // boundary where it first became available. Re-extracting it on
                // a later owner swap would overwrite that checkpoint with the
                // owner's current (later) recurrent state. The trailing partial
                // block is still refreshed because its endpoint is current.
                if (_model.RequiresPerBlockCapture && block.Used == tokensInBlock)
                    continue;

                // The model's running state is the state at tokensInModel, so on a
                // recurrent model only the block ending there can carry it; any other
                // full block captured here (one CaptureNewlyFullBlocks never took) holds
                // its K/V rows only and is not a restore point.
                bool restorable = !_model.RequiresPerBlockCapture
                    || startToken + tokensInBlock == tokensInModel;
                long expectedBytes = SnapshotByteSize(tokensInBlock, restorable);
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
                // is sized for one full block so partial fits. A recurrent model's
                // full block is allocated at exactly the length written (SlabFor).
                var slab = SlabFor(block, tokensInBlock, (int)expectedBytes);
                dst.CopyTo(slab);
                block.Used = tokensInBlock;
                block.HoldsSnapshotBytes = tokensInBlock == _blockSize;
                if (_model.RequiresPerBlockCapture && tokensInBlock == _blockSize)
                    block.IsRestorablePrefixEnd = restorable;
            }
        }

        /// <summary>
        /// Bytes of one block snapshot of <paramref name="tokensInBlock"/> tokens. A
        /// per-block-capture (recurrent) model carries its running state only in a block
        /// that is a restore point (<paramref name="withRecurrentState"/>): the state copied
        /// into any other block is never read (a resume only ever lands on a restorable
        /// block, <see cref="LastRestorableInjectedLength"/>), and on Nemotron-H 8B it is
        /// ~99 MiB a block against ~4 MiB of K/V. Every other family writes the full block,
        /// exactly as before.
        /// </summary>
        private long SnapshotByteSize(int tokensInBlock, bool withRecurrentState)
        {
            long full = _model.ComputeKVBlockByteSize(tokensInBlock);
            if (withRecurrentState || !_model.RequiresPerBlockCapture || full <= 0)
                return full;
            long kvOnly = _model.ComputeKVBlockByteSizeWithoutRecurrentState(tokensInBlock);
            return kvOnly > 0 && kvOnly < full ? kvOnly : full;
        }

        /// <summary>The pool slab a snapshot of <paramref name="bytes"/> is written into: the
        /// full-size slab as always, or for a per-block-capture model's FULL block one of
        /// exactly <paramref name="bytes"/>, so a K/V-only block holds (and is charged) only
        /// that. A trailing partial block keeps the full-size slab: it always carries the
        /// state, is re-extracted at a new length on every swap-out, and is never a page,
        /// so an exact-length slab would only reallocate the ~99 MiB state section each time.</summary>
        private Span<byte> SlabFor(KvBlock block, int tokensInBlock, int bytes)
            => _model.RequiresPerBlockCapture && tokensInBlock == _blockSize
                ? _pool.Storage.GetSpan(block.Id, bytes)
                : _pool.Storage.GetSpan(block.Id);

        /// <summary>The bytes a per-block-capture model injects from a slab of
        /// <paramref name="slabLength"/> bytes holding <paramref name="tokensInBlock"/> tokens:
        /// the <paramref name="full"/> snapshot (for a partial block, the leading bytes of its
        /// full-size slab; see <see cref="SlabFor"/>), or for a full block its K/V-only form
        /// (which restores the K/V rows and leaves the running state as the blocks before it
        /// left it). 0 when the slab holds neither.</summary>
        private long RecurrentInjectByteSize(int tokensInBlock, int slabLength, long full)
        {
            if (slabLength == full) return full;
            if (tokensInBlock != _blockSize) return slabLength > full ? full : 0;
            long kvOnly = _model.ComputeKVBlockByteSizeWithoutRecurrentState(tokensInBlock);
            return kvOnly > 0 && slabLength == kvOnly ? kvOnly : 0;
        }

        /// <summary>Inject all blocks for <paramref name="seq"/> into the model's
        /// fresh KV state. Called when swapping in. Returns how many leading tokens
        /// were actually injected: <paramref name="tokensToInject"/> on success, less
        /// when a block could not be read or the model refused it (the model then
        /// holds exactly the blocks before it, per the TryInjectKVBlock contract).</summary>
        private int InjectAllBlocks(SequenceState seq, int tokensToInject)
        {
            if (tokensToInject <= 0) return 0;
            if (!_model.SupportsKVStateSnapshot || !_model.SupportsCrossSequenceKvReuse) return 0;

            // A pooled restore is faithful only up to the model's cap: a sliding-window
            // ring cannot be rebuilt from blocks past its rows, and those are not captured
            // (CapturableBlocks). Whole blocks only (a full block's slab is laid out for a
            // full block); the caller forwards the rest again (InjectPrefixOrRecompute).
            int cap = _model.MaxReusablePrefixTokens;
            if (tokensToInject > cap)
                tokensToInject = cap / _blockSize * _blockSize;
            if (tokensToInject <= 0) return 0;

            int injected = 0;
            int blocks = seq.BlockTable.NumBlocks;
            for (int b = 0; b < blocks; b++)
            {
                int startToken = b * _blockSize;
                if (startToken >= tokensToInject) break;
                int tokensInBlock = Math.Min(_blockSize, tokensToInject - startToken);
                var block = seq.BlockTable.Blocks[b];

                // Only bytes this sequence actually captured: a full block needs its
                // snapshot (CaptureNewlyFullBlocks / ExtractAllBlocks / an adopted page),
                // the trailing partial block its swap-out extract. A placeholder block
                // (live / end-state adoption) or one the ring outran before it could be
                // captured still holds a zero slab, and injecting it would be silent
                // corruption.
                if (tokensInBlock == _blockSize ? !block.HoldsSnapshotBytes : block.Used != tokensInBlock)
                    break;

                long expectedBytes = _model.ComputeKVBlockByteSize(tokensInBlock);
                if (expectedBytes <= 0) break;

                var src = _pool.Storage.GetReadOnlySpan(block.Id);
                if (_model.RequiresPerBlockCapture)
                {
                    // A recurrent model's full block is either form; the slab length says which.
                    expectedBytes = RecurrentInjectByteSize(tokensInBlock, src.Length, expectedBytes);
                    if (expectedBytes <= 0)
                    {
                        _logger.LogWarning(
                            "Inject found a block snapshot of neither known size for sequence {EngineRequestId} block {Block}: have {Have} bytes",
                            seq.RequestId, b, src.Length);
                        break;
                    }
                }

                if (src.Length < expectedBytes)
                {
                    _logger.LogWarning(
                        "Inject would underflow for sequence {EngineRequestId} block {Block}: have {Have} need {Need}",
                        seq.RequestId, b, src.Length, expectedBytes);
                    break;
                }
                var slice = src[..(int)expectedBytes];
                if (!_model.TryInjectKVBlock(startToken, tokensInBlock, slice))
                {
                    _logger.LogWarning(
                        "Inject failed for sequence {EngineRequestId} block {Block} at {Start}",
                        seq.RequestId, b, startToken);
                    break;
                }
                injected = startToken + tokensInBlock;
            }
            return injected;
        }

        /// <summary>
        /// Make the model hold <paramref name="seq"/>'s first <paramref name="tokens"/>
        /// computed tokens from its blocks (radix design M0e, D10/P16).
        ///
        /// <para>An inject that stops early used to be silent: the sequence kept its
        /// claimed <see cref="SequenceState.NumComputedTokens"/>, so the next forward
        /// appended at the model's real head while every position after it was
        /// numbered as if the whole prefix were there, and
        /// <see cref="SequenceState.PrefixCacheReusedTokens"/> still reported the
        /// promised reuse. A materialization failure is now a MISS, never a partial
        /// state: the reuse accounting drops to what was materialized, the request is
        /// reset to that length, and the lost tail is forwarded again right here, so
        /// the model and the step's already-planned work agree again. The re-forward
        /// is done in place rather than left to the next schedule because the
        /// scheduler still treats a sequence whose prompt is computed as a decoder
        /// and would never re-forward generated tokens (D7, M0d).</para>
        /// </summary>
        private void InjectPrefixOrRecompute(SequenceState seq, int tokens)
        {
            int injected = InjectAllBlocks(seq, tokens);
            if (injected >= tokens && EndsAtRestorablePoint(seq, injected))
                return;

            int usable = LastRestorableInjectedLength(seq, injected);
            if (usable < injected)
            {
                // A per-block-capture (recurrent) model restores its running state from
                // the LAST injected block, which is only a real checkpoint at a
                // restorable endpoint. Rebuild from the deepest one, or from nothing.
                _model.ResetKVCache();
                if (usable > 0 && InjectAllBlocks(seq, usable) < usable)
                {
                    _model.ResetKVCache();
                    usable = 0;
                }
            }
            else if (usable == 0)
            {
                _model.ResetKVCache();
            }

            _logger.LogWarning(
                "Materializing the cached prefix of {EngineRequestId} stopped at {Materialized} of {Claimed} tokens; " +
                "reuse is corrected to {Reused} and the remaining {Recompute} tokens are forwarded again.",
                seq.RequestId, usable, tokens, Math.Min(seq.PrefixCacheReusedTokens, usable), tokens - usable);

            seq.PrefixCacheReusedTokens = Math.Min(seq.PrefixCacheReusedTokens, usable);
            seq.RevokeComputedTokensTo(usable);
            RecomputeLostTail(seq, tokens);
        }

        /// <summary>The injected length a model can actually resume from. Attention-only
        /// models resume at any injected block end; a per-block-capture model only at a
        /// block whose payload was captured at its own endpoint.</summary>
        private int LastRestorableInjectedLength(SequenceState seq, int injected)
        {
            if (injected <= 0 || !_model.RequiresPerBlockCapture)
                return Math.Max(0, injected);
            // A shortfall always stops at a block start, so every injected block is full.
            // (A complete inject that ended on a block with no state - see
            // EndsAtRestorablePoint - is block-aligned too.)
            for (int b = injected / _blockSize - 1; b >= 0; b--)
            {
                if (seq.BlockTable.Blocks[b].IsRestorablePrefixEnd)
                    return (b + 1) * _blockSize;
            }
            return 0;
        }

        /// <summary>
        /// Whether the model holds a resumable state after injecting <paramref name="injected"/>
        /// tokens of <paramref name="seq"/>. Always for an attention-only model. A
        /// per-block-capture model restores its running state from the last injected block,
        /// which is right only when that block was captured at its own end: a trailing
        /// partial block (extracted at the owner's current end, state included) or a full
        /// block marked <see cref="KvBlock.IsRestorablePrefixEnd"/>. Any other full block
        /// holds K/V rows only (or, before K/V-only blocks existed, the state of a later
        /// position), so a complete inject ending on one is no resume point either.
        /// </summary>
        private bool EndsAtRestorablePoint(SequenceState seq, int injected)
        {
            if (!_model.RequiresPerBlockCapture || injected <= 0 || injected % _blockSize != 0)
                return true;
            int last = injected / _blockSize - 1;
            return last < seq.BlockTable.NumBlocks && seq.BlockTable.Blocks[last].IsRestorablePrefixEnd;
        }

        /// <summary>Forward <paramref name="seq"/>'s tokens from its (just revoked)
        /// computed count up to <paramref name="target"/>, in scheduler-sized chunks,
        /// without sampling. The logits the sequence will sample from next are kept.</summary>
        private void RecomputeLostTail(SequenceState seq, int target)
        {
            int start = seq.NumComputedTokens;
            if (start >= target)
                return;
            // The logits a decoder samples from next may be the model's own buffer
            // (borrowed on a single-work step), which the forwards below overwrite.
            float[]? pendingLogits = seq.LastLogits != null ? (float[])seq.LastLogits.Clone() : null;
            float[]? lastLogits = null;
            int chunk = Math.Max(1, _scheduler.Config.MaxNumBatchedTokens);
            int promptTokens = seq.PromptTokens.Count;
            while (start < target)
            {
                int n = Math.Min(chunk, target - start);
                // Never mix prompt and generated tokens in one forward, as the planned
                // steps never do: a prompt slice queues its media embeddings and M-RoPE
                // position table for exactly its prompt tokens, so a forward that ran on
                // into generated tokens would carry a position table shorter than itself.
                if (start < promptTokens)
                    n = Math.Min(n, promptTokens - start);
                var tokens = new int[n];
                for (int i = 0; i < n; i++)
                    tokens[i] = seq.TokenAt(start + i);
                if (_model.MultimodalInjector != null && start < promptTokens)
                {
                    _model.MultimodalInjector.QueuePromptEmbeddingsForSlice(
                        start, Math.Min(n, promptTokens - start), seq.RequestId);
                }
                long forwardStarted = Stopwatch.GetTimestamp();
                lastLogits = _model.Forward(tokens);
                long forwardTicks = Stopwatch.GetTimestamp() - forwardStarted;
                if (start < promptTokens)
                {
                    seq.ReplayPrefillElapsedTicks += forwardTicks;
                    if (seq.OutputTokens.Count == 0 && FirstTokenLogitsDump.IsArmed)
                        seq.NoteDumpPrefillStep("Replay", n);
                }
                else
                    seq.ReplayDecodeElapsedTicks += forwardTicks;
                seq.AdvanceComputedTokens(n);
                start += n;
            }
            seq.LastLogits = pendingLogits ?? (lastLogits != null ? (float[])lastLogits.Clone() : null);
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
                // Decided BEFORE extracting: a recurrent model's running state is the
                // state at NumComputedTokens, so it belongs to (and is only copied into)
                // the block ending exactly there. Every other block takes its K/V rows
                // only (SnapshotByteSize).
                bool restorable = !_model.RequiresPerBlockCapture
                    || (b == fullBlocksNow - 1 && seq.NumComputedTokens % _blockSize == 0);
                long bytes = SnapshotByteSize(_blockSize, restorable);
                EnsureScratch((int)bytes);
                var dst = _scratch.AsSpan(0, (int)bytes);
                if (!_model.TryExtractKVBlock(startToken, _blockSize, dst))
                    break;
                dst.CopyTo(SlabFor(block, _blockSize, (int)bytes));
                block.Used = _blockSize;
                block.HoldsSnapshotBytes = true;
                block.IsRestorablePrefixEnd = restorable;
                captured++;
                if (b < previouslyFull) previouslyFull = b;
                // Inside the reply only the newest restore point keeps its state.
                if (restorable && _model.RequiresPerBlockCapture && (b + 1) * _blockSize > seq.PromptTokens.Count)
                    HoldDecodeRestorePoint(seq, b);
            }

            // Let the scheduler index the newly-full blocks (hash registration).
            if (captured > 0)
                _scheduler.OnBlocksCommitted(seq, previouslyFull * _blockSize);

            return captured;
        }

        /// <summary>
        /// Make block <paramref name="blockIndex"/> - a restore point inside the generated text,
        /// just captured with the running state - the sequence's held decode restore point
        /// (<see cref="SequenceState.HeldDecodeRestoreBlock"/>), and rewrite the one it replaces
        /// as K/V rows only. Decode fills a block every block-size tokens and every one ends at
        /// the forward that filled it, so without this each became a restore point:
        /// a 2,000-token reply on Nemotron-H 8B left ~790 MiB of Mamba2 state in the prefix
        /// cache, of which a later turn can use one block at most (the last restore point
        /// before it diverges from the reply; a continuation of the whole reply takes the end
        /// state). The block replaced is still this sequence's alone - the prefix cache takes
        /// held blocks only once the sequence stops - and the model still holds its rows, as
        /// the sequence that just forwarded is the owner. Otherwise it keeps its state.
        /// </summary>
        private void HoldDecodeRestorePoint(SequenceState seq, int blockIndex)
        {
            int previous = seq.HeldDecodeRestoreBlock;
            seq.HeldDecodeRestoreBlock = blockIndex;
            if (previous < 0 || previous >= blockIndex || previous >= seq.BlockTable.NumBlocks)
                return;
            KvBlock block = seq.BlockTable.Blocks[previous];
            if (!block.IsRestorablePrefixEnd || block.RefCount != 1 || !block.HoldsSnapshotBytes
                || (RadixCache != null && RadixCache.Tree.TryGetBlockOwner(block, out _)))
                return;
            long bytes = SnapshotByteSize(_blockSize, withRecurrentState: false);
            if (bytes <= 0 || bytes >= _pool.Storage.SlabLength(block.Id))
                return;
            EnsureScratch((int)bytes);
            var dst = _scratch.AsSpan(0, (int)bytes);
            if (!_model.TryExtractKVBlock(previous * _blockSize, _blockSize, dst))
                return;
            dst.CopyTo(SlabFor(block, _blockSize, (int)bytes));
            block.IsRestorablePrefixEnd = false;
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
            SetSpeculation(Speculation);
            RadixCache?.Reset();
            _currentOwner = null;
            _ownerTokensInModel = 0;
            _ownerForwardedTokens = 0;
            _liveCacheSeq = null;
            _liveCacheLen = 0;
            _liveCacheValid = false;
            _pendingRetainedFusedTruncations.Clear();
            _specCtx = null;
            _fusedSeqById.Clear();
            _model.ResetKVCache();
        }
    }

    /// <summary>Result of executing one ScheduledSequenceWork. Reported back
    /// to the engine for streaming + stop detection.</summary>
    public sealed class SequenceStepResult
    {
        public required SequenceState Sequence { get; init; }
        public int TokensForwarded { get; init; }
        public int SampledToken { get; init; } = -1;

        /// <summary>Speculatively drafted tokens accepted by verification this
        /// step, in order, FOLLOWING <see cref="SampledToken"/>. Already
        /// appended to the sequence's OutputTokens by the executor; the engine
        /// streams them with per-token EOS / length checks. Null when the step
        /// produced no extra tokens.</summary>
        public IReadOnlyList<int>? ExtraTokens { get; init; }

        public bool IsPrefill { get; init; }
        public int FullBlocksCaptured { get; init; }
        public long ForwardElapsedTicks { get; init; }
        public Exception? Error { get; init; }

        /// <summary>The model had no device slot for this new request (<see cref="SequenceSlotUnavailableException"/>).
        /// Nothing was forwarded; the scheduler re-queues the request instead of failing it.</summary>
        public bool SlotUnavailable { get; init; }

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
        /// False when a static limitation (e.g. Gemma 4 with MoE layers or a
        /// block-quantized KV cache, or tensor parallelism) makes
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
        /// only to the linear KV cache) over to the paged storage
        /// that <see cref="ForwardBatch"/> reads from. When false, the
        /// executor must not use the N=1 fast path for this model — a
        /// later second-sequence arrival would corrupt the first
        /// sequence's attention.</summary>
        bool SupportsLinearKVMigration => false;

        /// <summary>Copy the given sequence's K/V history out of the
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

        /// <summary>True when every sequence's complete state lives in a native
        /// slot the model allocates itself for the whole context (a direct-CUDA
        /// engine's sequence slots), so the scheduler's block pool holds no KV
        /// bytes and only tracks reservations. The engine then sizes that pool for
        /// one context per running sequence rather than one context in all, which
        /// would otherwise cap every concurrent request's tokens together at one
        /// context and preempt healthy slots, and it refuses a request whose
        /// prompt plus output allowance cannot fit one slot. The model reports a
        /// full device with <see cref="SequenceSlotUnavailableException"/>.
        /// Requires <see cref="SupportsPerSequenceFusedForward"/>. Default false.
        /// DeepSeek V4.1 is recognised by its architecture as well.</summary>
        bool UsesNativeSequenceSlots => false;

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
        // The per-sequence fused path keeps each concurrent request's complete
        // continuation state in its own holder and never writes the shared paged
        // block storage, so it contributes nothing to the prefix-cache pool. A
        // circular-cache model cannot reconstruct a long prefix from that pool, and
        // a hybrid attention+recurrent model cannot reconstruct its GDN state from
        // attention K/V alone. Without retention, a multi-turn follow-up that arrives
        // while/after other requests ran concurrently can therefore re-prefill the
        // whole conversation (KV-reuse ratio 0). The executor instead RETAINS a
        // finished fused holder and re-adopts it for a later request whose prompt
        // exactly extends the retained tokens — the cross-request analogue of the
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

        /// <summary>Reason the most recent logits batched-decode attempt declined,
        /// when provided by the model. A decline describes one step, not every
        /// subsequent decode step.</summary>
        string? BatchedFusedDecodeDeclineReason => null;

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

        /// <summary>Prefill several sequences' prompt chunks in ONE fused graph:
        /// sequence i evaluates <paramref name="tokens"/>[i] starting at
        /// <paramref name="positions"/>[i] (its holder's head) against its own
        /// per-request holder, allocating a fresh holder for a request that has
        /// none yet. Concurrent arrivals then share every weight read instead of
        /// prefilling one after another. On success writes each sequence's
        /// last-token logits into <paramref name="outLogits"/>[i] and returns
        /// true; false (nothing advanced) leaves the step to the per-sequence
        /// loop. Default false (opt-in).</summary>
        bool TryForwardBatchedFusedPrefill(
            IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] positions, float[][] outLogits) => false;

        /// <summary>Whether <see cref="TryForwardBatchedFusedPrefill"/> also takes a request whose holder
        /// still holds rows past the prefix it reused (a cached state cloned for it, otherwise rewound on its
        /// first step): the model rewinds such a holder to <c>positions[i]</c> itself, which must then be
        /// exact. Default false: those requests prefill in the per-sequence loop, one after another.</summary>
        bool BatchedPrefillRewindsHolders => false;

        /// <summary>Whether <see cref="TryForwardBatchedFusedPrefill"/> takes a request whose holder the
        /// executor created and filled from a cached prefix earlier in the same step (bound, injected,
        /// reserved, not yet forwarded). The executor then binds such requests before the batched
        /// prefill instead of leaving each to the per-sequence loop. Default false.</summary>
        bool BatchedPrefillTakesPrebound => false;
    }

    /// <summary>Per-step metadata for the batched paged attention path.
    /// Mirrors vLLM's <c>CommonAttentionMetadata</c>.</summary>
    public sealed class BatchedForwardContext
    {
        public required List<SequenceState> Sequences { get; init; }
        public required List<int> NumScheduledTokens { get; init; }
        public required List<int> QueryStartLoc { get; init; }
        public required List<int> Positions { get; init; }
        public required List<int> SlotMapping { get; init; }
        public required int[][] BlockTables { get; init; }
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
        public int[]? OverrideFlatTokens { get; set; }

        /// <summary>When non-null, receives the post-final-norm hidden state of
        /// every row (numTokens ├ù hidden floats) — llama.cpp's h_nextn, consumed
        /// by the MTP draft head.</summary>
        public float[]? CaptureHiddenAll { get; init; }

        /// <summary>When non-null, receives LM-head logits for every row
        /// (numTokens ├ù vocab floats) — speculative verification needs per-row
        /// logits, not just the last position.</summary>
        public float[]? CaptureLogitsAll { get; init; }
    }
}
