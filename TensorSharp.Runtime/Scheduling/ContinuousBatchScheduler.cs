// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// vLLM-style iteration-level (a.k.a. continuous) batching scheduler.
    /// Maintains a waiting queue (FCFS by submission order, priority break-tie)
    /// and a running set. Each call to <see cref="Schedule"/> picks the work
    /// for the next forward pass, allocating KV blocks from the pool, exploiting
    /// prefix-cache hits, and preempting low-priority sequences when blocks are
    /// exhausted.
    ///
    /// Not thread-safe; <see cref="InferenceEngine"/> calls it from the engine
    /// worker thread under its own lock.
    /// </summary>
    public sealed class ContinuousBatchScheduler
    {
        private readonly SchedulerConfig _cfg;
        private readonly BlockPool _pool;
        private readonly string _fingerprint;

        // When false, cross-sequence K/V reuse is unsafe for this model (e.g. Gemma 4's
        // sliding-window cache restores incorrectly into a fresh sequence). We then
        // neither adopt cached prefix blocks nor register blocks for others to adopt,
        // so every sequence re-prefills and gets correct logits.
        private readonly bool _crossSeqKvReuse;

        // Upper bound on how many leading prompt tokens may be adopted from the prefix
        // cache. Sliding-window models cap this at their window size because the
        // circular-cache snapshot can only be faithfully restored within one window.
        private readonly int _maxReusablePrefixTokens;

        // Hybrid recurrent models can only snapshot their running state at an
        // actual Forward boundary. Large fused prefill chunks still capture the
        // attention bytes for every covered block, but only the boundary block
        // is a valid prefix endpoint.
        private readonly bool _requiresPerBlockCapture;

        // Whether a pooled prefix may extend past a media span (see
        // IModelArchitecture.SupportsReuseAcrossMediaSpan).
        private readonly bool _reuseAcrossMediaSpan;
        // See IModelArchitecture.CanPrefillMediaAfterReusedPrefix; null means yes.
        private readonly Func<int, bool> _canPrefillMediaAfterReusedPrefix;

        // Live-cache continuation hooks (wired by the engine to the executor). The
        // first computes how many leading prompt tokens can be served by continuing
        // the model's live KV cache (beyond the pooled-snapshot cap); the second
        // sets the sequence up to do so. Lets same-session follow-up turns reuse the
        // whole conversation prefix on sliding-window models. Null when unwired.
        private Func<SequenceState, int> _liveContinuationLcp;
        private Func<SequenceState, int, bool> _liveContinuationAdopt;

        // Retained fused-cache continuation hooks (wired by the engine to the
        // executor). Cross-request analogue of the live-cache hooks above: they
        // re-adopt a FINISHED concurrent request's retained per-request state holder
        // for a new request whose prompt exactly extends it. Unlike live-cache
        // continuation (one shared cache, sole-sequence only), each retained holder
        // is independent, so multiple concurrent admissions can each continue from
        // their own holder. Null when unwired.
        private Func<SequenceState, int> _fusedContinuationLcp;
        private Func<SequenceState, int, bool> _fusedContinuationAdopt;

        private readonly LinkedList<SequenceState> _waiting = new();
        private readonly Dictionary<string, LinkedListNode<SequenceState>> _waitingIndex = new();

        // Running set: keyed by request id, ordered by sn for fairness.
        private readonly Dictionary<string, SequenceState> _running = new();
        private readonly List<SequenceState> _runningOrder = new();
        // When a mixed decode/prefill step runs out of token budget part-way
        // through the prefill set, resume at that sequence next iteration.
        // Request ids survive additions/removals better than a numeric cursor.
        private string _nextPrefillRequestId;
        private readonly ILogger _logger;

        public ContinuousBatchScheduler(
            SchedulerConfig cfg,
            BlockPool pool,
            string modelFingerprint,
            ILogger logger = null,
            bool supportsCrossSequenceKvReuse = true,
            int maxReusablePrefixTokens = int.MaxValue,
            bool requiresPerBlockCapture = false,
            bool supportsReuseAcrossMediaSpan = true,
            Func<int, bool> canPrefillMediaAfterReusedPrefix = null)
        {
            _reuseAcrossMediaSpan = supportsReuseAcrossMediaSpan;
            _canPrefillMediaAfterReusedPrefix = canPrefillMediaAfterReusedPrefix;
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            _pool = pool ?? throw new ArgumentNullException(nameof(pool));
            _logger = logger ?? NullLogger.Instance;
            _fingerprint = modelFingerprint ?? string.Empty;
            _crossSeqKvReuse = supportsCrossSequenceKvReuse;
            _maxReusablePrefixTokens = maxReusablePrefixTokens <= 0 ? int.MaxValue : maxReusablePrefixTokens;
            // Per-boundary alignment only serves pooled prefix checkpoints.
            // If cross-sequence reuse is disabled, those checkpoints are never
            // consumed and splitting the prefill merely costs throughput.
            _requiresPerBlockCapture = requiresPerBlockCapture && supportsCrossSequenceKvReuse;
        }

        /// <summary>Whether cross-sequence prefix-cache reuse is enabled for this
        /// model. False for models (e.g. Gemma 4 SWA) whose K/V snapshot cannot be
        /// faithfully restored into a different sequence.</summary>
        private bool PrefixCachingActive => _cfg.EnablePrefixCaching && _crossSeqKvReuse;

        /// <summary>Wire the live-cache continuation hooks (see the fields). Called
        /// once by the engine after the executor is constructed.</summary>
        public void AttachLiveCacheContinuation(
            Func<SequenceState, int> computeLcp,
            Func<SequenceState, int, bool> adopt)
        {
            _liveContinuationLcp = computeLcp;
            _liveContinuationAdopt = adopt;
        }

        /// <summary>Wire the retained fused-cache continuation hooks (see the fields).
        /// Called once by the engine after the executor is constructed.</summary>
        // Whether prefill chunks should end exactly at a sequence's shared-prefix
        // boundary so the executor can checkpoint the model's state there. Set by
        // the engine when the model can take such checkpoints; otherwise the
        // boundary is ignored and chunks are sized for throughput alone.
        private bool _alignToSharedPrefix;

        /// <summary>Enable shared-prefix alignment (see <see cref="AlignSharedPrefixBoundary"/>).</summary>
        public void EnablePrefixCheckpoints() => _alignToSharedPrefix = true;

        /// <summary>
        /// Cut a prefill chunk at the sequence's shared-prefix boundary when the chunk
        /// would otherwise run past it, so the state the executor checkpoints is the
        /// state after exactly those tokens. Costs one extra chunk boundary per new
        /// conversation, and only until a checkpoint exists.
        /// </summary>
        private int AlignSharedPrefixBoundary(SequenceState seq, int want)
        {
            if (!_alignToSharedPrefix || want <= 0 || seq.SharedPrefixTokens <= 0 || seq.PrefixCheckpointTaken)
                return want;
            int start = seq.NumComputedTokens;
            int boundary = seq.SharedPrefixTokens;
            if (boundary > start && start + want > boundary)
                return boundary - start;
            return want;
        }

        public void AttachFusedCacheContinuation(
            Func<SequenceState, int> computeLcp,
            Func<SequenceState, int, bool> adopt)
        {
            _fusedContinuationLcp = computeLcp;
            _fusedContinuationAdopt = adopt;
        }

        // Why the executor's last live / retained lookup found nothing. Admission
        // owns the reuse decision end to end, so it is the only place that can say
        // truthfully what a request reused and, when it reused nothing, which of the
        // three mechanisms could have helped and why none did.
        private Func<string?>? _liveDeclineReason;
        private Func<string?>? _fusedDeclineReason;
        private Func<string?>? _fusedAdoptionSource;
        private Func<int>? _blockedByScopeTokens;

        /// <summary>Wire the executor's reuse diagnostics so admission can explain a
        /// turn that reused nothing, name what served one that reused something, and
        /// count what conversation scoping withheld. Optional: without it the summary
        /// still reports what was reused, just without the per-mechanism detail.</summary>
        public void AttachReuseDiagnostics(
            Func<string?> liveReason,
            Func<string?> fusedReason,
            Func<string?>? fusedSource = null,
            Func<int>? blockedByScopeTokens = null)
        {
            _liveDeclineReason = liveReason;
            _fusedDeclineReason = fusedReason;
            _fusedAdoptionSource = fusedSource;
            _blockedByScopeTokens = blockedByScopeTokens;
        }

        /// <summary>
        /// One truthful line per admitted prompt: how many tokens the request reuses,
        /// which mechanism served them, and how many still have to be prefilled.
        ///
        /// <para>
        /// This exists because the three reuse mechanisms are tried in sequence and
        /// each used to narrate its own attempt. The live-cache attempt in particular
        /// announced that the turn would "re-prefill its full prompt (KV reuse 0)"
        /// before the retained-holder path — the mechanism that actually serves every
        /// fused model — had even been asked. Operators reading the log concluded the
        /// cache was broken on turns that were reusing 99% of their prompt.
        /// </para>
        /// </summary>
        private void LogPromptReuseOutcome(
            SequenceState seq, bool servedByLiveCache, bool servedByRetainedState, string? liveReason)
        {
            int prompt = seq.PromptTokens.Count;
            if (prompt <= 0) return;
            int reused = seq.PrefixCacheReusedTokens;
            int blocked = _blockedByScopeTokens?.Invoke() ?? 0;
            string scope = seq.CacheScope == null
                ? "unscoped"
                : seq.CacheScope.Length <= 8 ? seq.CacheScope : seq.CacheScope.Substring(0, 8);

            // Isolation at work, not a fault: another conversation's state matched past
            // the public prefix and was not used. Debug, because on a multi-client host
            // it is routine; it is the line that proves a leak is NOT happening.
            if (blocked > 0)
            {
                _logger.LogDebug(
                    "Prompt reuse for {RequestId}: blocked by scope - another conversation's state matched " +
                    "{Blocked} more token(s) past the public prefix ({Public} tokens); scope {Scope}.",
                    seq.RequestId, blocked, seq.SharedPrefixTokens, scope);
            }

            if (reused > 0)
            {
                string source = servedByLiveCache
                    ? $"the model's live KV cache of this conversation ({reused} tokens)"
                    : servedByRetainedState
                        ? _fusedAdoptionSource?.Invoke() ?? $"retained model state ({reused} tokens)"
                        : $"pooled prefix-cache blocks ({reused} tokens)";
                _logger.LogInformation(
                    "Prompt reuse for {RequestId}: {Reused}/{Prompt} tokens ({Percent:F1}%) continue from " +
                    "{Source}; {Prefill} token(s) to prefill; scope {Scope}.",
                    seq.RequestId, reused, prompt, 100.0 * reused / prompt,
                    source,
                    Math.Max(0, prompt - reused),
                    scope);
                return;
            }

            // Nothing reused. This is the case that costs a full prefill, so name
            // every mechanism and why it could not help - a first request in a fresh
            // conversation lands here legitimately, and so does a genuine regression.
            _logger.LogInformation(
                "No prompt reuse for {RequestId}: all {Prompt} prompt token(s) re-prefill. " +
                "Live KV cache: {LiveReason}. Retained state: {RetainedReason}. Pooled blocks: {PooledReason}.",
                seq.RequestId, prompt,
                liveReason ?? "not attempted",
                (_fusedContinuationLcp == null ? "not wired" : _fusedDeclineReason?.Invoke()) ?? "no match",
                PrefixCachingActive
                    ? "no matching blocks in the index"
                    : _cfg.EnablePrefixCaching
                        ? "unavailable for this model (its K/V cannot be restored into another sequence)"
                        : "disabled (TS_SCHED_PREFIX_CACHE)");
        }

        public int WaitingCount => _waiting.Count;
        public int RunningCount => _running.Count;
        public BlockPool Pool => _pool;
        public SchedulerConfig Config => _cfg;

        /// <summary>Whether this scheduler can actually publish reusable prefix
        /// blocks. The executor uses it to avoid extracting KV snapshots when
        /// prefix caching is disabled or unsafe for the loaded model.</summary>
        public bool PrefixCachingEnabled => PrefixCachingActive;

        /// <summary>The operator's prefix-cache switch alone (TS_SCHED_PREFIX_CACHE),
        /// which is what gates EVERY reuse at admission — live, retained holder,
        /// checkpoint and pooled — where <see cref="PrefixCachingEnabled"/> also asks
        /// whether the POOLED path can serve this model (it never can for Qwen 3.5,
        /// whose retained holders and checkpoints are exactly the alternative).</summary>
        public bool PrefixCacheConfigured => _cfg.EnablePrefixCaching;

        /// <summary>Snapshot all requests currently owned by the scheduler.
        /// Used by the engine's failure path when scheduling itself throws and
        /// no per-step <see cref="SchedulerOutput"/> is available.</summary>
        public List<SequenceState> GetInFlightSequencesSnapshot()
        {
            var result = new List<SequenceState>(_runningOrder.Count + _waiting.Count);
            var seen = new HashSet<string>(StringComparer.Ordinal);

            foreach (var seq in _runningOrder)
            {
                if (seq == null || seq.Status.IsFinished()) continue;
                if (seen.Add(seq.RequestId)) result.Add(seq);
            }

            foreach (var seq in _waiting)
            {
                if (seq == null || seq.Status.IsFinished()) continue;
                if (seen.Add(seq.RequestId)) result.Add(seq);
            }

            return result;
        }

        /// <summary>Snapshot the currently-running sequences in fairness order.
        /// Used by the engine only after an empty schedule proves that none of
        /// them can make forward progress.</summary>
        public List<SequenceState> GetRunningSequencesSnapshot()
        {
            var result = new List<SequenceState>(_runningOrder.Count);
            foreach (var seq in _runningOrder)
            {
                if (seq == null || seq.Status.IsFinished()) continue;
                result.Add(seq);
            }
            return result;
        }

        /// <summary>Submit a sequence. It enters the waiting queue; the next
        /// <see cref="Schedule"/> call will try to admit it.</summary>
        public void Submit(SequenceState seq)
        {
            if (seq == null) throw new ArgumentNullException(nameof(seq));
            if (_waitingIndex.ContainsKey(seq.RequestId) || _running.ContainsKey(seq.RequestId))
                throw new InvalidOperationException($"Sequence {seq.RequestId} is already submitted.");

            // A single request can eventually reclaim blocks from other requests
            // through preemption, but it can never exceed the pool's physical
            // capacity. Reject an impossible reservation before spending time on
            // a partial prefill that is guaranteed to stall at the boundary.
            long capacityTokens = (long)_pool.NumBlocks * _cfg.BlockSize;
            long requestedTokens = (long)seq.PromptTokens.Count + seq.MaxNewTokens;
            if (requestedTokens > capacityTokens)
            {
                var ex = new InvalidOperationException(
                    $"KV cache capacity exceeded for request {seq.RequestId}: prompt " +
                    $"({seq.PromptTokens.Count} tokens) plus max output ({seq.MaxNewTokens} " +
                    $"tokens) requires {requestedTokens} token slots, but the configured " +
                    $"KV block pool holds {capacityTokens}. Shorten the request or enlarge " +
                    $"TS_SCHED_NUM_BLOCKS.");
                seq.Status = SequenceStatus.FinishedError;
                seq.FinishReason = "error";
                seq.Error = ex;
                throw ex;
            }

            var node = _waiting.AddLast(seq);
            _waitingIndex[seq.RequestId] = node;
            seq.Status = SequenceStatus.Waiting;
        }

        /// <summary>Abort a sequence by id. If running, frees its blocks. If
        /// waiting, just removes it. Idempotent.</summary>
        public bool Abort(string requestId)
        {
            if (_waitingIndex.TryGetValue(requestId, out var node))
            {
                return FinishSequence(node.Value, SequenceStatus.FinishedAborted, "aborted", cacheBlocks: false);
            }
            if (_running.TryGetValue(requestId, out var seq))
            {
                return FinishSequence(seq, SequenceStatus.FinishedAborted, "aborted", cacheBlocks: true);
            }
            return false;
        }

        /// <summary>
        /// Decide the work for the next forward pass.
        /// </summary>
        public SchedulerOutput Schedule()
        {
            var output = new SchedulerOutput();
            int tokenBudget = _cfg.MaxNumBatchedTokens;

            // When there's at most one sequence in the whole system there is no
            // concurrent decode to interleave with, so the small prefill chunk
            // (which exists only for fairness) just forces a long prompt onto the
            // slow per-op, GPU-syncs-every-op path for every chunk that crosses
            // the sliding-window boundary. Feed a lone prompt in one big chunk
            // (bounded by the batched-token budget) like the CLI does, keeping it
            // on the fused single-graph prefill path. The moment a 2nd request
            // appears this reverts to small chunks automatically.
            bool noContention = (_running.Count + _waiting.Count) <= 1;
            int soloPrefillCap = Math.Min(
                _cfg.SoloPrefillChunkSize, _cfg.MaxNumBatchedTokens);

            // Snapshot the order so block-pressure preemption may safely mutate
            // _runningOrder below. Split the snapshot by phase: like llama.cpp,
            // vLLM and SGLang, every runnable decoder gets its one-token slot
            // before any long prompt is allowed to consume the remaining budget.
            var runningSnapshot = new List<SequenceState>(_runningOrder);
            // Allocate in rank order (priority, then submission): a re-admitted
            // preemption victim is appended to _runningOrder, and visiting it first
            // would let it claim blocks an older sequence then cannot preempt back
            // from (work already planned this step is never a victim).
            runningSnapshot.Sort((a, b) => VictimRank(a).CompareTo(VictimRank(b)));
            var decodeSnapshot = new List<SequenceState>(runningSnapshot.Count);
            var prefillSnapshot = new List<SequenceState>(runningSnapshot.Count);
            foreach (var seq in runningSnapshot)
            {
                if (!_running.ContainsKey(seq.RequestId)) continue;
                int promptUncomputed = Math.Max(
                    0, seq.PromptTokens.Count - seq.NumComputedTokens);
                if (promptUncomputed == 0)
                    decodeSnapshot.Add(seq);
                else
                    prefillSnapshot.Add(seq);
            }
            bool hasActiveDecode = decodeSnapshot.Count > 0;

            // -------------------------------------------------------------- 1. Decode first.
            // A fixed running-order traversal can otherwise let enough prefills
            // exhaust MaxNumBatchedTokens before a later decoder is visited.
            // Decode work is tiny and unlocks true token-batched model paths.
            // --------------------------------------------------------------
            foreach (var seq in decodeSnapshot)
            {
                if (tokenBudget <= 0) break;
                // A prior allocation attempt may have preempted a later entry
                // from this snapshot to recover KV blocks.
                if (!_running.ContainsKey(seq.RequestId)) continue;
                TryScheduleRunningSequence(
                    seq, want: 1, isPrefill: false, output, ref tokenBudget);
            }

            // -------------------------------------------------------------- 2. Existing prefills.
            // With no decoder active, split the whole remaining batch budget
            // across all runnable/admittable prompts. The old fixed per-request
            // cap used only half of a 4096-token step for two prompts. In a mixed
            // step retain the configured latency cap and rotate the first prefill
            // whenever the budget is exhausted, avoiding permanent tail-request
            // starvation (4/4/1, 4/4/1, ...).
            // --------------------------------------------------------------
            RotatePrefillsToResumePoint(prefillSnapshot);
            int admissibleWaiting = Math.Min(
                _waiting.Count,
                Math.Max(0, _cfg.MaxNumRunningSequences - _running.Count));
            int prefillCandidatesRemaining = prefillSnapshot.Count + admissibleWaiting;

            for (int i = 0; i < prefillSnapshot.Count; i++)
            {
                var seq = prefillSnapshot[i];
                if (tokenBudget <= 0)
                {
                    _nextPrefillRequestId = seq.RequestId;
                    break;
                }
                if (!_running.ContainsKey(seq.RequestId))
                {
                    prefillCandidatesRemaining = Math.Max(0, prefillCandidatesRemaining - 1);
                    continue;
                }

                int promptUncomputed = Math.Max(
                    0, seq.PromptTokens.Count - seq.NumComputedTokens);
                int cap = GetPrefillCap(
                    noContention, hasActiveDecode, tokenBudget,
                    prefillCandidatesRemaining, soloPrefillCap);
                int desired = Math.Min(promptUncomputed, cap);
                int want = Math.Min(desired, tokenBudget);
                want = AlignRecurrentPrefillBoundary(seq, want, promptUncomputed);
                want = AlignSharedPrefixBoundary(seq, want);
                if (want <= 0)
                {
                    _nextPrefillRequestId = seq.RequestId;
                    break;
                }

                bool scheduled = TryScheduleRunningSequence(
                    seq, want, isPrefill: true, output, ref tokenBudget);
                prefillCandidatesRemaining = Math.Max(0, prefillCandidatesRemaining - 1);
                if (!scheduled || want < desired)
                    _nextPrefillRequestId = seq.RequestId;
                else if (tokenBudget == 0 && i + 1 < prefillSnapshot.Count)
                    _nextPrefillRequestId = prefillSnapshot[i + 1].RequestId;
            }

            // -------------------------------------------------------------- 3. Admit waiting sequences.
            // --------------------------------------------------------------
            while (_waiting.Count > 0 && tokenBudget > 0 && _running.Count < _cfg.MaxNumRunningSequences)
            {
                var node = _waiting.First;
                var seq = node.Value;

                // A sequence preempted earlier in this same output must stay
                // parked until the next iteration. Re-admitting it here would
                // put its id in both ScheduledWork and PreemptedRequestIds: the
                // engine would execute the new prefill and then release that
                // now-running request's model-owned cache after the step.
                if (output.PreemptedRequestIds.Contains(seq.RequestId))
                    break;

                // Admit by capacity. A newcomer is admitted only when the free pool
                // can hold its whole prompt on top of what the prompts already
                // running still have to allocate. Admitting on the strength of its
                // FIRST chunk alone let four 20k-token prompts start against a pool
                // that holds three: they prefilled in parallel until the pool ran dry,
                // then preempted each other's nearly finished prefills over and over
                // (a request re-prefilled 15 times; waves took 700 s against 39 s
                // solo). Waiting in the queue costs nothing; a preempted prefill costs
                // all of its work. A lone request always fits (Submit checks it).
                if (_running.Count > 0 && !HasPromptCapacityFor(seq))
                {
                    LogCapacityWait(seq);
                    break;
                }

                // Try prefix cache lookup before allocating blocks (only for
                // brand-new sequences; preempted ones already had their blocks
                // freed and need a fresh re-prefill, no shortcut).
                bool plannedLiveContinuation = false;
                bool plannedFusedContinuation = false;
                string? liveDeclineReason = null;
                if (seq.BlockTable.NumBlocks == 0 && _cfg.EnablePrefixCaching)
                {
                    // Live-cache continuation: when this is the SOLE sequence about to
                    // run (nothing else running or already scheduled this step), the
                    // model's live KV cache from the previous turn is still intact and
                    // its prompt may extend it. Continuing from that live cache reuses
                    // the whole conversation prefix - past the pooled-snapshot window
                    // cap - with no corruption. Gated to the sole-sequence case so no
                    // concurrent sequence can clobber the live cache before we run.
                    if (_running.Count == 0
                        && output.ScheduledWork.Count == 0
                        && _liveContinuationLcp != null
                        && _liveContinuationAdopt != null)
                    {
                        int lcp = _liveContinuationLcp(seq);
                        // A short live prefix on a pooled-capable model: take the live
                        // cache unless the pooled blocks actually cover at least as much
                        // (whole blocks only, one token left, and only if captured).
                        int pooledCovers = lcp > 0 && PrefixCachingActive && lcp <= _maxReusablePrefixTokens
                            ? PlanPrefixBlockAdoption(seq, logBacktrack: false, out _).Count * _cfg.BlockSize
                            : 0;
                        if (lcp > 0 && pooledCovers >= lcp)
                            liveDeclineReason = $"pooled prefix-cache blocks cover {pooledCovers} tokens, at least the live match of {lcp}";
                        else if (lcp > 0 && _liveContinuationAdopt(seq, lcp))
                            plannedLiveContinuation = true;
                        else
                            liveDeclineReason = _liveDeclineReason?.Invoke() ?? "no usable live prefix";
                    }
                    else if (_liveContinuationLcp != null)
                    {
                        // Not even attempted. The sole-sequence gate is the usual
                        // reason and it is invisible from the request's telemetry,
                        // which just reports 0% reuse.
                        liveDeclineReason =
                            $"not attempted (another sequence holds the live cache: running={_running.Count}, "
                            + $"scheduledThisStep={output.ScheduledWork.Count})";
                        _logger.LogDebug(
                            "Live-cache continuation not attempted for {RequestId}: running={Running} scheduledThisStep={Scheduled}.",
                            seq.RequestId, _running.Count, output.ScheduledWork.Count);
                    }

                    // Retained fused-cache continuation: a finished concurrent
                    // request's complete model-owned state remains alive; if this
                    // prompt extends it exactly, continue from that holder without
                    // reconstructing circular K/V (Gemma 4) or separating attention
                    // K/V from recurrent GDN state (Qwen 3.5/3.6). Each retained holder
                    // is independent ÔÇö no shared live cache to clobber ÔÇö so this is
                    // NOT gated to the sole-sequence case and doesn't block co-admitting
                    // other sequences this step. It restores multi-turn prefix reuse
                    // after a concurrent fused round left nothing in the paged pool.
                    if (!plannedLiveContinuation
                        && _fusedContinuationLcp != null
                        && _fusedContinuationAdopt != null)
                    {
                        int flcp = _fusedContinuationLcp(seq);
                        if (flcp > 0 && _fusedContinuationAdopt(seq, flcp))
                            plannedFusedContinuation = true;
                    }

                    if (!plannedLiveContinuation
                        && !plannedFusedContinuation
                        && PrefixCachingActive)
                        AdoptPrefixBlocksCapped(seq);

                    // Every mechanism has now had its turn, so the outcome is finally
                    // knowable. Exactly one line, whatever happened.
                    LogPromptReuseOutcome(
                        seq, plannedLiveContinuation, plannedFusedContinuation, liveDeclineReason);
                }

                int promptUncomputed = Math.Max(0, seq.PromptTokens.Count - seq.NumComputedTokens);
                if (promptUncomputed <= 0)
                {
                    // The whole prompt was a prefix-cache hit. Force a 1-token
                    // forward to produce fresh logits.
                    promptUncomputed = 1;
                }

                int cap = GetPrefillCap(
                    noContention, hasActiveDecode, tokenBudget,
                    Math.Max(1, prefillCandidatesRemaining), soloPrefillCap);
                int desired = Math.Min(promptUncomputed, cap);
                int want = Math.Min(desired, tokenBudget);
                want = Math.Min(want, tokenBudget);
                want = AlignRecurrentPrefillBoundary(seq, want, promptUncomputed);
                want = AlignSharedPrefixBoundary(seq, want);
                if (want <= 0) break;

                if (!TryEnsureBlocksForStep(seq, want))
                {
                    // Can't fit. Stop admitting; we'll retry next step.
                    break;
                }

                // Promote to running.
                _waiting.Remove(node);
                _waitingIndex.Remove(seq.RequestId);
                _running[seq.RequestId] = seq;
                _runningOrder.Add(seq);
                seq.Status = SequenceStatus.Running;
                seq.FirstScheduledAt = DateTime.UtcNow;

                bool isPrefill = want > 1 || promptUncomputed > 0;
                output.ScheduledWork.Add(new ScheduledSequenceWork(seq, want, true, isPrefill));
                tokenBudget -= want;
                prefillCandidatesRemaining = Math.Max(0, prefillCandidatesRemaining - 1);
                if (want < desired)
                    _nextPrefillRequestId = seq.RequestId;

                // A live-cache continuation depends on the model's live cache staying
                // intact until this sequence runs. Don't admit any other sequence this
                // step (it could take ownership first and reset the cache); the others
                // wait one step.
                if (plannedLiveContinuation)
                    break;
            }

            return output;
        }

        /// <summary>
        /// Add one already-running sequence to this iteration after ensuring its
        /// KV allocation. Returns false when block pressure prevents progress.
        /// </summary>
        private bool TryScheduleRunningSequence(
            SequenceState seq,
            int want,
            bool isPrefill,
            SchedulerOutput output,
            ref int tokenBudget)
        {
            want = Math.Min(want, tokenBudget);
            if (want <= 0) return false;

            if (!TryEnsureBlocksForStep(seq, want)
                && !TryPreemptForBlocks(seq, want, output))
            {
                return false;
            }

            bool isFresh = seq.FirstScheduledAt == null;
            if (isFresh) seq.FirstScheduledAt = DateTime.UtcNow;
            output.ScheduledWork.Add(
                new ScheduledSequenceWork(seq, want, isFresh, isPrefill));
            tokenBudget -= want;
            return true;
        }

        /// <summary>Choose a latency cap for mixed work, or an even share of the
        /// full device token budget when all contenders are still prefilling.</summary>
        private int GetPrefillCap(
            bool noContention,
            bool hasActiveDecode,
            int tokenBudget,
            int candidatesRemaining,
            int soloPrefillCap)
        {
            if (noContention)
                return soloPrefillCap;
            if (hasActiveDecode)
                return _cfg.MaxPrefillChunkSize;

            candidatesRemaining = Math.Max(1, candidatesRemaining);
            int evenShare = tokenBudget / candidatesRemaining;
            if (tokenBudget % candidatesRemaining != 0)
                evenShare++;
            return Math.Max(1, evenShare);
        }

        /// <summary>Move the prefill that received only a partial quantum last
        /// iteration to the front, implementing a stable round-robin cursor.</summary>
        private void RotatePrefillsToResumePoint(List<SequenceState> prefills)
        {
            if (prefills.Count <= 1 || string.IsNullOrEmpty(_nextPrefillRequestId))
                return;
            int start = prefills.FindIndex(
                seq => string.Equals(
                    seq.RequestId, _nextPrefillRequestId,
                    StringComparison.Ordinal));
            if (start <= 0) return;

            var prefix = prefills.GetRange(0, start);
            prefills.RemoveRange(0, start);
            prefills.AddRange(prefix);
        }

        /// <summary>Called by the executor after the forward pass completes.
        /// Updates accounting and finishes sequences as needed.</summary>
        public void NotifyStepCompleted(SchedulerOutput output)
        {
            // The executor has already populated each scheduled seq's
            // LastLogits + advanced NumComputedTokens. We only finish sequences
            // that decided to stop (EOS / length cap) - those are reported back
            // to us via NotifyStop().
        }

        /// <summary>Mark a sequence as finished and release its blocks. The
        /// sequence id is added to <paramref name="output"/> so the executor
        /// can drop any per-step buffers.</summary>
        public void NotifyStop(SequenceState seq, SequenceStatus finalStatus, string reason, SchedulerOutput output)
        {
            if (seq == null) return;
            if (FinishSequence(seq, finalStatus, reason, cacheBlocks: true))
                output.FinishedRequestIds.Add(seq.RequestId);
        }

        /// <summary>Mark a sequence as errored and release any scheduler-owned
        /// blocks. Error paths deliberately skip prefix-cache registration
        /// because the model state for the failed step may be partial.</summary>
        public bool NotifyError(SequenceState seq, Exception error, SchedulerOutput output = null)
        {
            if (seq == null) return false;
            bool finished = FinishSequence(seq, SequenceStatus.FinishedError, "error", cacheBlocks: false, error: error);
            if (finished && output != null)
                output.FinishedRequestIds.Add(seq.RequestId);
            return finished;
        }

        /// <summary>Free a finished sequence's blocks and remove from running.</summary>
        private bool FinishSequence(
            SequenceState seq,
            SequenceStatus finalStatus,
            string reason,
            bool cacheBlocks,
            Exception error = null)
        {
            if (seq == null || seq.Status.IsFinished()) return false;

            // Cache the final partial trailing block into the prefix cache
            // (if there are any full blocks) before freeing.
            if (cacheBlocks)
                CacheFullBlocksForSequence(seq);

            var freed = seq.BlockTable.Clear();
            if (freed.Count > 0) _pool.Free(freed);

            seq.Status = finalStatus;
            seq.FinishReason = reason;
            seq.Error = error;
            seq.LastLogits = null;

            if (_waitingIndex.TryGetValue(seq.RequestId, out var waitingNode))
            {
                _waiting.Remove(waitingNode);
                _waitingIndex.Remove(seq.RequestId);
            }

            if (_running.Remove(seq.RequestId))
                _runningOrder.Remove(seq);

            return true;
        }

        /// <summary>Make sure the sequence has block table capacity for
        /// <paramref name="extraTokens"/> more tokens. Allocates new blocks
        /// from the pool as needed. Returns false when the pool can't
        /// satisfy the request.</summary>
        private bool TryEnsureBlocksForStep(SequenceState seq, int extraTokens)
        {
            int needed = seq.NumComputedTokens + extraTokens;
            int neededBlocks = (needed + _cfg.BlockSize - 1) / _cfg.BlockSize;
            int currentBlocks = seq.BlockTable.NumBlocks;
            int delta = neededBlocks - currentBlocks;
            if (delta <= 0) return true;

            var newBlocks = _pool.AllocateNew(delta);
            if (newBlocks == null) return false;
            for (int i = 0; i < newBlocks.Length; i++)
                seq.BlockTable.AppendBlock(newBlocks[i]);
            return true;
        }

        /// <summary>
        /// Keep large fused chunks for throughput, but split the final prompt
        /// chunk at the preceding block boundary.  That produces a recent exact
        /// recurrent checkpoint (and, for a block-aligned prompt, leaves its last
        /// block for a separate forward because prefix adoption must retain one
        /// token for fresh logits).
        /// </summary>
        private int AlignRecurrentPrefillBoundary(
            SequenceState seq,
            int want,
            int promptUncomputed)
        {
            if (!_requiresPerBlockCapture || want <= 0)
                return want;

            int start = seq.NumComputedTokens;
            int end = start + want;

            // An explicit client breakpoint (cache_control / prompt_cache_breakpoint)
            // caps both registration and adoption at its block boundary. For a
            // recurrent model the blocks inside a fused round are non-restorable
            // (they carry the round-end state), so without a real checkpoint at the
            // marker the marked prefix stays in the index but is never adoptable -
            // every follow-up request matches it and re-prefills it. Snap the round
            // so it ends exactly at the last block boundary the breakpoint admits;
            // the capture then records a genuine checkpoint there and the marked
            // prefix becomes fully restorable.
            int markerBoundary = (seq.CacheBreakpointLimit / _cfg.BlockSize) * _cfg.BlockSize;
            if (markerBoundary > start && markerBoundary < end)
                return markerBoundary - start;

            int tail = end % _cfg.BlockSize;
            bool reachesPromptEnd = want >= promptUncomputed;
            if (reachesPromptEnd && tail == 0)
                tail = _cfg.BlockSize;

            if (tail == 0)
                return want;

            int aligned = want - tail;
            return aligned > 0 ? aligned : want;
        }

        private static int BlocksFor(int tokens, int blockSize)
            => tokens <= 0 ? 0 : (tokens + blockSize - 1) / blockSize;

        /// <summary>
        /// Whether the free pool can hold <paramref name="candidate"/>'s whole prompt
        /// after every running prompt has allocated the rest of its own. Decode growth
        /// is not reserved: it arrives one token at a time and is served by preempting
        /// the newest sequence, which re-prefills a prompt that was already admitted.
        /// Prompt blocks the candidate would adopt from a running sequence are not
        /// counted (see the body).
        /// </summary>
        private bool HasPromptCapacityFor(SequenceState candidate)
        {
            long outstanding = 0;
            foreach (var running in _runningOrder)
            {
                int missing = BlocksFor(running.PromptTokens.Count, _cfg.BlockSize) - running.BlockTable.NumBlocks;
                if (missing > 0) outstanding += missing;
            }
            int need = BlocksFor(candidate.PromptTokens.Count, _cfg.BlockSize) - candidate.BlockTable.NumBlocks;
            long available = (long)_pool.NumFreeBlocks - outstanding;
            if (available >= need)
                return true;

            // Prefix blocks another sequence is still using are shared on adoption
            // and cost the pool nothing, so they do not count against the newcomer
            // (without this, requests sharing a long system prompt or document ran one
            // at a time once their prompts summed past the pool). An idle cached block
            // sits in the free queue, so adopting it costs a free block like a new one.
            // Only consulted when the cheap check fails, i.e. while a request waits.
            if (!PrefixCachingActive || candidate.BlockTable.NumBlocks != 0)
                return false;
            _capacityPlanScratch.Clear();
            FillPrefixBlockAdoptionPlan(candidate, logBacktrack: false, _capacityPlanScratch, out _);
            foreach (var block in _capacityPlanScratch)
                if (block.RefCount > 0) need--;
            _capacityPlanScratch.Clear();
            if (available < need)
                return false;

            // The discount is real only if admission ends up on the pooled path. A
            // retained fused holder is tried first and the executor backs it with new
            // blocks for its whole prefix (TryAdoptFusedContinuation), so a model with
            // both kinds of reuse (Gemma 4) would be admitted on a discount it never
            // takes. Asked last, and only when the discount is what admits the request.
            if (_fusedContinuationLcp != null && _fusedContinuationAdopt != null
                && _fusedContinuationLcp(candidate) > 0)
                return false;
            return true;
        }

        // HasPromptCapacityFor runs on every Schedule() while a request waits; its plan
        // is only counted, so it reuses one list instead of allocating one per call.
        private readonly List<KvBlock> _capacityPlanScratch = new();

        private string _capacityWaitLoggedFor;

        private void LogCapacityWait(SequenceState seq)
        {
            if (string.Equals(_capacityWaitLoggedFor, seq.RequestId, StringComparison.Ordinal))
                return;
            _capacityWaitLoggedFor = seq.RequestId;
            _logger.LogInformation(
                "KV block pool: {RequestId} ({PromptTokens} prompt tokens) waits for capacity; " +
                "{Running} running request(s) still need their prompts' blocks ({Free}/{Total} blocks free). " +
                "It is admitted when one of them finishes. Raise TS_SCHED_NUM_BLOCKS to run more at once.",
                seq.RequestId, seq.PromptTokens.Count, _running.Count, _pool.NumFreeBlocks, _pool.NumBlocks);
        }

        /// <summary>Scheduling rank: higher priority first, then earlier submission.
        /// A larger value is a better preemption victim.</summary>
        private static long VictimRank(SequenceState s) => -(long)s.Priority * (1L << 40) + s.Sn;

        /// <summary>Preempt the lowest-ranked running sequence that ranks BELOW
        /// <paramref name="needyForBlocks"/> (lower priority, or the same priority and
        /// submitted later) and free its blocks, then retry allocation. Returns true
        /// if successful. A sequence never preempts one that outranks it: when the
        /// needy sequence is itself the newest it simply waits a step with its blocks
        /// intact, and the older sequences finish and free theirs. Letting the newest
        /// preempt the oldest turned a full pool into a livelock of long prefills
        /// preempting each other at 14-20k computed tokens.</summary>
        private bool TryPreemptForBlocks(SequenceState needyForBlocks, int extraTokens, SchedulerOutput output)
        {
            SequenceState victim = null;
            long victimRank = VictimRank(needyForBlocks);
            foreach (var s in _runningOrder)
            {
                if (ReferenceEquals(s, needyForBlocks)) continue;
                // Work already emitted for this iteration must retain its block
                // table until the executor consumes the plan. Preempting it here
                // would leave a stale ScheduledWork entry pointing at a reset
                // sequence and could forward the wrong token positions.
                if (output.ScheduledWork.Exists(
                        work => ReferenceEquals(work.Sequence, s)))
                    continue;
                long rank = VictimRank(s);
                if (rank > victimRank)
                {
                    victimRank = rank;
                    victim = s;
                }
            }
            if (victim == null) return false;

            // A preemption is a running request visibly stalling and later
            // re-prefilling; leaving it unlogged makes that look like a hang.
            _logger.LogWarning(
                "KV block pool pressure: preempting {VictimRequestId} ({VictimTokens} computed tokens) " +
                "to make room; it re-queues and re-prefills when capacity frees up. Raise " +
                "TS_SCHED_NUM_BLOCKS to avoid this.",
                victim.RequestId, victim.NumComputedTokens);
            PreemptSequence(victim);
            output.PreemptedRequestIds.Add(victim.RequestId);
            return TryEnsureBlocksForStep(needyForBlocks, extraTokens);
        }

        /// <summary>Take a running sequence's blocks back and re-park it in
        /// the waiting queue. Full blocks are added to the prefix cache before
        /// freeing so the next admission's hash lookup recovers most of the
        /// work.</summary>
        private void PreemptSequence(SequenceState victim)
        {
            CacheFullBlocksForSequence(victim);
            var freed = victim.BlockTable.Clear();
            if (freed.Count > 0) _pool.Free(freed);

            _running.Remove(victim.RequestId);
            _runningOrder.Remove(victim);
            victim.Status = SequenceStatus.Preempted;
            victim.ResetForPreemption();

            // Re-park at the front of the waiting queue so the victim resumes
            // soon - we don't want preemption to permanently demote a request.
            _waiting.AddFirst(victim);
            _waitingIndex[victim.RequestId] = _waiting.First;
        }

        /// <summary>Look up a sequence's prompt prefix in the block hash index
        /// and adopt the longest chain of matching full blocks - capped so the
        /// sequence still has at least one token to forward (we need fresh
        /// logits, and the partial trailing block of an exact-prefix hit has
        /// no cached logits). Updates the sequence's block table, computed-
        /// token counter, and <see cref="SequenceState.PrefixCacheReusedTokens"/>.
        /// </summary>
        private void AdoptPrefixBlocksCapped(SequenceState seq)
        {
            if (seq.BlockTable.NumBlocks > 0) return;
            var adoptable = PlanPrefixBlockAdoption(seq, logBacktrack: true, out bool adoptInPagedStorage);
            for (int i = 0; i < adoptable.Count; i++)
            {
                _pool.Touch(adoptable[i]);
                seq.BlockTable.AppendBlock(adoptable[i]);
            }

            int adoptedTokens = adoptable.Count * _cfg.BlockSize;
            if (adoptedTokens > 0)
            {
                seq.PrefixCacheReusedTokens = adoptedTokens;
                seq.SetComputedTokensForPrefixAdoption(adoptedTokens);
                // Set both ways: a preempted sequence re-admitted onto pool snapshots
                // must not keep a stale paged-resident flag from its previous run.
                seq.KvStateInPagedStorage = adoptInPagedStorage;
            }
        }

        /// <summary>The pooled blocks <see cref="AdoptPrefixBlocksCapped"/> would adopt
        /// for <paramref name="seq"/>, without touching refcounts or the block table.
        /// <paramref name="adoptInPagedStorage"/> says where they are read from: the
        /// model's paged arrays (blocks a batched paged step wrote) or pool snapshots.</summary>
        private List<KvBlock> PlanPrefixBlockAdoption(SequenceState seq, bool logBacktrack, out bool adoptInPagedStorage)
        {
            var matching = new List<KvBlock>();
            FillPrefixBlockAdoptionPlan(seq, logBacktrack, matching, out adoptInPagedStorage);
            return matching;
        }

        /// <summary><see cref="PlanPrefixBlockAdoption"/> into a caller-owned, empty list.</summary>
        private void FillPrefixBlockAdoptionPlan(
            SequenceState seq, bool logBacktrack, List<KvBlock> matching, out bool adoptInPagedStorage)
        {
            adoptInPagedStorage = false;
            if (seq.PromptTokens.Count < _cfg.BlockSize) return;

            var hashes = GetPromptBlockHashes(seq);
            int maxAdoptableTokens = Math.Max(0, seq.PromptTokens.Count - 1);
            // An explicit boundary limits reuse as well as registration. Otherwise a
            // request that says "cache none" (empty/[0]) could still adopt blocks that
            // an earlier implicit-cache request placed in the shared index, and a request
            // with a finite boundary could reuse volatile prompt content past it.
            if (seq.CacheBreakpoints != null)
                maxAdoptableTokens = Math.Min(maxAdoptableTokens, seq.CacheBreakpointLimit);
            // Cap reuse at the model's reliably-restorable prefix length (sliding
            // window for circular caches). Adopting beyond this would inject a wrapped
            // snapshot that the model can't faithfully reconstruct -> corrupt output.
            if (_maxReusablePrefixTokens != int.MaxValue)
                maxAdoptableTokens = Math.Min(maxAdoptableTokens, _maxReusablePrefixTokens);
            // A reused prefix never ends inside a media span (the block hashes already
            // carry each span's content identity, so the spans before it match).
            maxAdoptableTokens = PromptMediaSpans.ClampReusablePrefix(
                maxAdoptableTokens, seq.MediaSpans, seq.MediaSpans, _reuseAcrossMediaSpan);
            // Any adoption leaves the prompt's media to prefill at a non-zero position.
            if (seq.MediaSpans.Count > 0 && _canPrefillMediaAfterReusedPrefix != null
                && !_canPrefillMediaAfterReusedPrefix(seq.PromptTokens.Count))
                maxAdoptableTokens = 0;
            int maxAdoptableBlocks = maxAdoptableTokens / _cfg.BlockSize;

            for (int i = 0; i < hashes.Count && i < maxAdoptableBlocks; i++)
            {
                if (!_pool.TryFindByHash(hashes[i], out var block))
                    break;
                matching.Add(block);
            }

            // Where the matched K/V actually lives decides how it can be adopted. A
            // block written by a batched paged step (IBatchedPagedModel.ForwardBatch)
            // exists only in the model's paged arrays - nothing was extracted into the
            // pool - so it is served by adopting it IN PAGED STORAGE: the sequence starts
            // as a paged resident and its first forward reads those slots directly.
            // Restoring such a block into the linear cache instead injects bytes that
            // were never captured, which is how a repeated Mistral 3 / Hy-MT2 prompt
            // longer than one block turned into fluent garbage. A block with a pool
            // snapshot keeps the linear restore it always had. Recurrent models keep
            // state outside the paged arrays and media prompts are peeled onto the
            // per-sequence path, so neither adopts paged-only blocks.
            int pagedChain = 0;
            while (pagedChain < matching.Count && matching[pagedChain].HoldsModelPagedKv)
                pagedChain++;
            int snapshotChain = 0;
            while (snapshotChain < matching.Count
                   && !(matching[snapshotChain].HoldsModelPagedKv && !matching[snapshotChain].HoldsSnapshotBytes))
                snapshotChain++;
            adoptInPagedStorage = !_requiresPerBlockCapture
                && seq.MediaSpans.Count == 0
                && pagedChain > snapshotChain;
            int usableBlocks = adoptInPagedStorage ? pagedChain : snapshotChain;
            if (usableBlocks < matching.Count)
            {
                if (logBacktrack)
                    _logger.LogInformation(
                    "Prefix cache matched {Matched} block(s) for {RequestId} but only {Usable} hold K/V this " +
                    "request can read ({Where}); the rest of the prompt re-prefills.",
                    matching.Count, seq.RequestId, usableBlocks,
                    adoptInPagedStorage ? "model paged storage" : "pool snapshots");
                matching.RemoveRange(usableBlocks, matching.Count - usableBlocks);
            }

            int lastRestorable = -1;
            for (int i = 0; i < matching.Count; i++)
            {
                if (matching[i].IsRestorablePrefixEnd)
                    lastRestorable = i;
            }

            // Interior blocks from one large recurrent prefill carry correct
            // attention slices but the chunk-end recurrent state. They are safe
            // only when followed by a real checkpoint, whose injection overwrites
            // that transient state. Backtrack to the newest such endpoint before
            // changing refcounts or the sequence block table.
            int adopted = lastRestorable + 1;
            if (adopted < matching.Count)
            {
                // The cache MATCHED more than it can deliver; without this line
                // the user sees kvCacheReusedTokens far below a warm cache's
                // promise with no explanation.
                if (logBacktrack)
                    _logger.LogInformation(
                    "Prefix cache matched {Matched} block(s) for {RequestId} but only {Adopted} are " +
                    "restorable (a recurrent checkpoint boundary caps adoption); the rest of the " +
                    "prompt re-prefills.",
                    matching.Count, seq.RequestId, adopted);
                matching.RemoveRange(adopted, matching.Count - adopted);
            }
        }

        /// <summary>After advancing tokens or finishing, check whether the
        /// sequence has any newly-full blocks and (if not already cached)
        /// add them to the block hash index.</summary>
        public void OnBlocksCommitted(SequenceState seq, int previousTokens)
        {
            if (!PrefixCachingActive) return;
            int prevFull = previousTokens / _cfg.BlockSize;
            int curFull = seq.NumComputedTokens / _cfg.BlockSize;
            if (curFull <= prevFull) return;

            int allTokensCovered = curFull * _cfg.BlockSize;
            // Build hashes from prompt+output prefix that's now block-aligned.
            var hashes = ComputeHashesForPrefix(seq, allTokensCovered);
            for (int b = prevFull; b < curFull; b++)
            {
                var block = seq.BlockTable.Blocks[b];
                if (block.ContentHash != null) continue;
                if (!IsBlockAllowedByExplicitMarkers(seq, b)) continue;
                // The recurrent batched-paged path keeps state in model-owned
                // slot pools and does not populate PagedKvStorage. Only blocks
                // explicitly extracted by CaptureNewlyFullBlocks are portable.
                if (_requiresPerBlockCapture && block.Used != _cfg.BlockSize)
                    continue;
                _pool.RegisterFullBlock(block, hashes[b], _cfg.BlockSize);
            }
        }

        /// <summary>
        /// Whether a full block may be registered in the prefix-cache index.
        /// <para>
        /// A request that carried explicit <c>cache_control</c> markers has told
        /// us which prefix is worth keeping; blocks past the last breakpoint are
        /// request-specific and are left out of the index so they cannot evict
        /// the prefixes the client asked to keep. A block qualifies only when it
        /// ends at or before that breakpoint — a block straddling it holds
        /// tokens from both sides, and the index is block-granular, so it is
        /// dropped (the floor-to-block-size loss the design accepts).
        /// </para>
        /// <para>
        /// Sequences without markers — the overwhelming majority — return true
        /// here and keep the default behaviour of caching every full block.
        /// </para>
        /// </summary>
        private bool IsBlockAllowedByExplicitMarkers(SequenceState seq, int blockIndex)
        {
            if (seq.CacheBreakpoints == null)
                return true;
            int limit = seq.CacheBreakpointLimit;
            return (blockIndex + 1) * _cfg.BlockSize <= limit;
        }

        private void CacheFullBlocksForSequence(SequenceState seq)
        {
            if (!PrefixCachingActive) return;
            // Hash only over positions that actually exist in the token list.
            // A speculative step that hit a mid-batch stop can leave
            // NumComputedTokens ahead of NumTotalTokens (the dropped tail's
            // K/V was committed but its tokens were truncated); clamping keeps
            // ComputeHashesForPrefix from indexing past the end of the list.
            int committed = Math.Min(seq.NumComputedTokens, seq.NumTotalTokens);
            int curFull = committed / _cfg.BlockSize;
            if (curFull == 0) return;
            int allTokensCovered = curFull * _cfg.BlockSize;
            var hashes = ComputeHashesForPrefix(seq, allTokensCovered);
            for (int b = 0; b < curFull && b < seq.BlockTable.Blocks.Count; b++)
            {
                var block = seq.BlockTable.Blocks[b];
                if (block.ContentHash != null) continue;
                if (!IsBlockAllowedByExplicitMarkers(seq, b)) continue;
                // Only register blocks whose K/V was actually extracted into
                // pool storage. CaptureNewlyFullBlocks sets Used==BlockSize
                // after a successful TryExtractKVBlock; blocks where extract
                // was declined (e.g. Gemma 4 SWA-local blocks past the
                // sliding window, where the circular cache has wrapped and
                // the byte-level snapshot is ill-defined) keep Used at 0.
                // Registering those would seed the prefix-cache index with
                // junk data ÔÇö a future sequence with a matching prompt
                // prefix would adopt them, claim NumComputedTokens past the
                // SWA window, and then EnsureOwnership.InjectAllBlocks would
                // fail on the same wrap-aliased positions (the "Inject
                // failed for sequence X block 2 at 512" warning).
                if (block.Used != _cfg.BlockSize) continue;
                _pool.RegisterFullBlock(block, hashes[b], _cfg.BlockSize);
            }
        }

        private List<KvBlockHash> ComputeHashesForPrefix(SequenceState seq, int tokens)
        {
            // Concatenate prompt + output into a token list for the first
            // <paramref name="tokens"/> positions.
            var list = new List<int>(tokens);
            for (int i = 0; i < tokens; i++)
                list.Add(seq.TokenAt(i));
            return ComputeHashesForTokens(seq, list, tokens);
        }

        private List<KvBlockHash> ComputeHashesForTokens(SequenceState seq, IReadOnlyList<int> tokens, int count)
            => KvBlockHasher.ComputeBlockHashes(tokens, _cfg.BlockSize, _fingerprint, b => BlockSalt(seq, b));

        /// <summary>Test hook: false recomputes the prompt's block hashes on every plan,
        /// which is what every plan did before they were cached.</summary>
        internal bool CachePromptBlockHashes { get; set; } = true;

        /// <summary>Test hook: how many times the prompt's block hashes were computed.</summary>
        internal int PromptBlockHashComputations { get; private set; }

        internal IReadOnlyList<KvBlockHash> GetPromptBlockHashesForTest(SequenceState seq) => GetPromptBlockHashes(seq);

        /// <summary>
        /// The full-block hashes of <paramref name="seq"/>'s prompt, computed once per
        /// sequence. A request waiting for capacity is planned on every
        /// <see cref="Schedule"/> call, and hashing its prompt (SHA-256 per block, with
        /// media and scope salts) is the whole cost of that plan for a long prompt; the
        /// pool lookups that follow are cheap and are always redone, because the pool
        /// changes between calls.
        ///
        /// <para>What the hashes depend on, and why each is covered: the prompt tokens,
        /// media spans, cache scope and shared-prefix length are fixed when the sequence
        /// is constructed (PromptTokens is copied from the caller's list there and nothing
        /// edits it afterwards; its count is still checked as a guard); the fingerprint and block size belong
        /// to the scheduler, and a sequence handed to another engine (a rebuilt one after
        /// a model change) finds a different pair and hashes again. Preemption, pool
        /// eviction and registration change which hashes are found, never the hashes.</para>
        /// </summary>
        private IReadOnlyList<KvBlockHash> GetPromptBlockHashes(SequenceState seq)
        {
            var cached = seq.CachedPromptBlockHashes;
            if (CachePromptBlockHashes && cached != null
                && cached.Matches(_fingerprint, _cfg.BlockSize, seq.PromptTokens.Count))
                return cached.Hashes;

            PromptBlockHashComputations++;
            var hashes = ComputeHashesForTokens(seq, seq.PromptTokens, seq.PromptTokens.Count);
            if (CachePromptBlockHashes)
                seq.CachedPromptBlockHashes = new PromptBlockHashes(
                    _fingerprint, _cfg.BlockSize, seq.PromptTokens.Count, hashes);
            return hashes;
        }

        /// <summary>
        /// What block <paramref name="blockIndex"/> of <paramref name="seq"/> is salted
        /// with beyond its tokens, or null. Two things, each only where it applies:
        /// <list type="bullet">
        /// <item>its media: the content identity of every span overlapping the block.
        /// Placeholder token ids are identical for any image, so without it two prompts
        /// with different pictures would share K/V; blocks before the first span stay
        /// unsalted, so a media prompt still shares its text-only leading blocks, and the
        /// parent chain carries the salt into every later block.</item>
        /// <item>its conversation: once a block extends past the request's public prefix
        /// (<see cref="SequenceState.SharedPrefixTokens"/>, the system prompt and tool
        /// declarations) it carries the request's <see cref="SequenceState.CacheScope"/>,
        /// so another conversation shares the public blocks and nothing after them.</item>
        /// </list>
        /// </summary>
        private string BlockSalt(SequenceState seq, int blockIndex)
        {
            int start = blockIndex * _cfg.BlockSize;
            int end = start + _cfg.BlockSize;
            string media = PromptMediaSpans.BlockSalt(seq.MediaSpans, start, end);
            string scope = seq.CacheScope != null && end > seq.SharedPrefixTokens
                ? "scope:" + seq.CacheScope
                : null;
            if (media == null) return scope;
            return scope == null ? media : media + scope;
        }
    }
}
