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
using TensorSharp.Runtime.Scheduling.PrefixCache;

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

        // Hybrid recurrent models can only snapshot their running state at an
        // actual Forward boundary. Large fused prefill chunks still capture the
        // attention bytes for every covered block, but only the boundary block
        // is a valid prefix endpoint.
        private readonly bool _requiresPerBlockCapture;

        // The radix prefix cache and the executor's hooks into it (wired by the engine
        // when prefix caching is on): how many leading prompt tokens a new request can
        // resume from what the cache holds, and the adoption of that prefix. Null
        // when prefix caching is off; every request then prefills from scratch.
        private Func<SequenceState, int>? _prefixCacheLcp;
        private Func<SequenceState, int, bool>? _prefixCacheAdopt;
        private PrefixCacheCoordinator? _radixCache;

        private readonly LinkedList<SequenceState> _waiting = new();
        private readonly Dictionary<string, LinkedListNode<SequenceState>> _waitingIndex = new();

        // Running set: keyed by request id, ordered by sn for fairness.
        private readonly Dictionary<string, SequenceState> _running = new();
        private readonly List<SequenceState> _runningOrder = new();
        // When a mixed decode/prefill step runs out of token budget part-way
        // through the prefill set, resume at that sequence next iteration.
        // Request ids survive additions/removals better than a numeric cursor.
        private string? _nextPrefillRequestId;
        private readonly ILogger _logger;

        // The request adapter of the running set (SequenceState.Adapter). Admission keeps
        // the running set adapter-homogeneous ("adapter epochs"), so this is the adapter of
        // every running sequence, and it changes only once the running set has drained.
        private IRequestAdapter? _epochAdapter;
        private string? _adapterWaitLoggedFor;

        public ContinuousBatchScheduler(
            SchedulerConfig cfg,
            BlockPool pool,
            ILogger? logger = null,
            bool supportsCrossSequenceKvReuse = true,
            bool requiresPerBlockCapture = false)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            _pool = pool ?? throw new ArgumentNullException(nameof(pool));
            _logger = logger ?? NullLogger.Instance;
            // Per-boundary alignment only serves prefix pages that other sequences
            // restore. If cross-sequence reuse is disabled, those pages are never
            // consumed and splitting the prefill merely costs throughput.
            _requiresPerBlockCapture = requiresPerBlockCapture && supportsCrossSequenceKvReuse;
        }

        /// <summary>Whether this scheduler reuses prompt prefixes: prefix caching is on
        /// and the radix cache is attached.</summary>
        private bool PrefixCachingActive => _cfg.EnablePrefixCaching && _radixCache != null;

        // Whether prefill chunks should end exactly at a sequence's shared-prefix
        // boundary so the executor can checkpoint the model's state there. Set by
        // the engine when the model can take such checkpoints; otherwise the
        // boundary is ignored and chunks are sized for throughput alone.
        private bool _alignToSharedPrefix;

        /// <summary>Enable shared-prefix alignment (see <see cref="AlignSharedPrefixBoundary"/>).</summary>
        public void EnablePrefixCheckpoints() => _alignToSharedPrefix = true;

        // Public boundaries at most this many tokens past a request's reused prefix do not
        // end a prefill step (ISoloPrefillBudgetModel.CheckpointGapTokens); 0 cuts at every one.
        private int _checkpointGapTokens;

        /// <summary>Set the model's checkpoint gap (see <see cref="ISoloPrefillBudgetModel.CheckpointGapTokens"/>).</summary>
        public void SetCheckpointGapTokens(int tokens) => _checkpointGapTokens = Math.Max(0, tokens);

        // Whether a scheduled prefill step ends on a public boundary, the only place its
        // producer checkpoints; with a gap, a step may run past a boundary instead.
        private static bool EndsOnPublicBoundary(ScheduledSequenceWork work)
            => work.Sequence.IsPublicCheckpointBoundary(work.Sequence.NumComputedTokens + work.NumScheduledTokens);

        // The model's own step size for a lone prompt (ISoloPrefillBudgetModel), set by
        // the engine unless the configuration chose SoloPrefillChunkSize.
        private ISoloPrefillBudgetModel? _soloPrefillPolicy;

        /// <summary>Let <paramref name="policy"/> size a lone prompt's prefill steps.</summary>
        public void SetSoloPrefillPolicy(ISoloPrefillBudgetModel? policy) => _soloPrefillPolicy = policy;


        /// <summary>
        /// Cut a prefill chunk at the sequence's public checkpoint boundaries when the chunk
        /// would otherwise run past it, so the state the executor checkpoints is the
        /// state after exactly those tokens. Reused boundaries are already behind
        /// the sequence's computed position and need no additional forward pass.
        /// </summary>
        // A budget cut that leaves at most this many tokens (and an eighth of the step) before the
        // point the prompt stops at anyway - a checkpoint boundary, the prompt end, the chunk cap -
        // runs them in this step: the step of their own they would take costs a whole forward pass.
        private const int TinyPrefillRemainder = 64;

        private int RunTinyRemainder(SequenceState seq, int want, int desired, int tokenBudget)
        {
            if (want <= 0 || want >= desired)
                return want;
            int start = seq.NumComputedTokens, cut = start + want, stop = start + desired;
            bool cutIsBoundary = false;
            void Boundary(int b)
            {
                if (b == cut)
                    cutIsBoundary = true;
                else if (b > cut && b < stop)
                    stop = b;
            }
            if (_alignToSharedPrefix && _radixCache != null)
            {
                if (seq.CacheBreakpoints != null)
                    foreach (int breakpoint in seq.CacheBreakpoints)
                        Boundary(breakpoint);
                foreach (int boundary in seq.PublicCheckpointBoundaries)
                    if (seq.IsPublicCheckpointBoundary(boundary))
                        Boundary(boundary);
            }
            if (_radixCache != null)
            {
                Boundary(_radixCache.AutomaticCheckpointBoundary(seq));
                Boundary(_radixCache.PromptTailCheckpointBoundary(seq));
            }
            int extended = stop - start;
            return !cutIsBoundary && extended - want <= Math.Min(TinyPrefillRemainder, want / 8) && extended <= tokenBudget
                ? extended : want;
        }

        private int AlignSharedPrefixBoundary(SequenceState seq, int want)
        {
            if (!_alignToSharedPrefix || want <= 0)
                return want;
            int start = seq.NumComputedTokens;
            if (_radixCache != null && seq.CacheBreakpoints != null)
            {
                foreach (int breakpoint in seq.CacheBreakpoints)
                    if (breakpoint > start && breakpoint <= seq.PromptTokens.Count && breakpoint < start + want)
                        want = breakpoint - start;
            }
            if (_radixCache != null && _radixCache.PublicCheckpointsSupported)
            {
                // A boundary just past the prefix this request reused gets no step of its own when
                // the model says so: the reused prefix already serves later requests to within the gap.
                int reused = seq.PrefixCacheReusedTokens;
                foreach (int boundary in seq.PublicCheckpointBoundaries)
                    if (boundary > start && boundary < start + want && seq.IsPublicCheckpointBoundary(boundary)
                        && !(reused > 0 && boundary - reused <= _checkpointGapTokens))
                        want = boundary - start;
            }
            if (_radixCache != null)
            {
                foreach (int boundary in new[] { _radixCache.AutomaticCheckpointBoundary(seq), _radixCache.PromptTailCheckpointBoundary(seq) })
                    if (boundary > start && boundary < start + want)
                        want = boundary - start;
            }
            return want;
        }

        internal void AttachRadixCacheContinuation(
            PrefixCacheCoordinator coordinator,
            Func<SequenceState, int> computeLcp,
            Func<SequenceState, int, bool> adopt)
        {
            _radixCache = coordinator;
            _prefixCacheLcp = computeLcp;
            _prefixCacheAdopt = adopt;
        }

        // Why the executor's last prefix-cache lookup found nothing, for the
        // admission log line.
        private Func<string?>? _prefixCacheDeclineReason;

        /// <summary>Wire the executor's reuse diagnostics so admission can explain a
        /// turn that reused nothing. Optional: without it the admission line still
        /// reports what was reused.</summary>
        public void AttachReuseDiagnostics(Func<string?> declineReason)
        {
            _prefixCacheDeclineReason = declineReason;
        }

        public int WaitingCount => _waiting.Count;
        public int RunningCount => _running.Count;
        public BlockPool Pool => _pool;
        public SchedulerConfig Config => _cfg;

        /// <summary>Whether this scheduler reuses prompt prefixes (prefix caching is on
        /// and the radix cache is attached). The executor uses it to avoid capturing
        /// state nothing will read.</summary>
        public bool PrefixCachingEnabled => PrefixCachingActive;

        /// <summary>The operator's prefix-cache switch alone (TS_SCHED_PREFIX_CACHE).</summary>
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
            // appears this reverts to small chunks automatically. A model with its
            // own solo step (SoloPrefillStep) may go past the batched-token budget;
            // a request that arrives meanwhile waits for that one step. A waiter the
            // adapter-epoch gate (phase 3) holds back is no contender this step.
            bool noContention = (_running.Count + CountEpochWaiting(limit: 2)) <= 1;
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
            // Prompt tokens this step may still carry beside its decodes (see
            // SchedulerConfig.MixedStepPrefillBudget); unbounded without a decode.
            int mixedPrefillLeft = hasActiveDecode && _cfg.MixedStepPrefillBudget > 0
                ? _cfg.MixedStepPrefillBudget : int.MaxValue;
            SequenceState? firstPrefillServed = null;
            int firstPrefillTokens = 0;

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
            // With no decoder active, hand out the whole remaining batch budget
            // to the runnable/admittable prompts, evenly (or oldest first, see
            // SchedulerConfig.PrefillOnlyFirstComeFirstServed). The old fixed
            // per-request cap used only half of a 4096-token step for two prompts. In a mixed
            // step retain the configured latency cap and rotate the first prefill
            // whenever the budget is exhausted, avoiding permanent tail-request
            // starvation (4/4/1, 4/4/1, ...).
            // --------------------------------------------------------------
            RotatePrefillsToResumePoint(prefillSnapshot);
            int admissibleWaiting = CountEpochWaiting(
                Math.Max(0, _cfg.MaxNumRunningSequences - _running.Count));
            int prefillCandidatesRemaining = prefillSnapshot.Count + admissibleWaiting;
            // The lone prompt step whose page the waiting requests sharing its prefix wait for
            // (LonePagePublishBoundary), and where that step ends.
            SequenceState? pageProducer = null;
            int pageProducerEnd = 0;

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
                // The only prompt running, with nothing decoding: the waiters it would hold
                // back (phase 3) are no contenders, so it takes a lone prompt's step.
                int pageBoundary = !hasActiveDecode && _running.Count == 1 && prefillSnapshot.Count == 1
                    ? LonePagePublishBoundary(seq) : 0;
                bool lone = noContention || pageBoundary > 0;
                int cap = GetPrefillCap(
                    lone, hasActiveDecode, tokenBudget,
                    prefillCandidatesRemaining, soloPrefillCap);
                if (lone) SoloPrefillStep(seq, promptUncomputed, ref cap, ref tokenBudget);
                int desired = Math.Min(promptUncomputed, cap);
                int want = Math.Min(desired, Math.Min(tokenBudget, mixedPrefillLeft));
                want = RunTinyRemainder(seq, want, desired, tokenBudget);
                want = AlignRecurrentPrefillBoundary(seq, want, promptUncomputed);
                want = AlignSharedPrefixBoundary(seq, want);
                want = CutAtBoundary(seq, want, pageBoundary);
                if (want <= 0)
                {
                    _nextPrefillRequestId = MixedPrefillResumePoint(seq, firstPrefillServed, firstPrefillTokens, mixedPrefillLeft);
                    break;
                }

                bool scheduled = TryScheduleRunningSequence(
                    seq, want, isPrefill: true, output, ref tokenBudget);
                prefillCandidatesRemaining = Math.Max(0, prefillCandidatesRemaining - 1);
                if (scheduled)
                {
                    if (pageBoundary > 0)
                        NotePageProducer(seq, want, ref pageProducer, ref pageProducerEnd);
                    mixedPrefillLeft -= want;
                    if (firstPrefillServed == null)
                    {
                        firstPrefillServed = seq;
                        firstPrefillTokens = want;
                    }
                }
                if (!scheduled || want < desired)
                    _nextPrefillRequestId = seq.RequestId;
                else if (mixedPrefillLeft <= 0 && i + 1 < prefillSnapshot.Count)
                    _nextPrefillRequestId = MixedPrefillResumePoint(prefillSnapshot[i + 1], firstPrefillServed, firstPrefillTokens, mixedPrefillLeft);
                else if (tokenBudget == 0 && i + 1 < prefillSnapshot.Count)
                    _nextPrefillRequestId = prefillSnapshot[i + 1].RequestId;
                if (mixedPrefillLeft <= 0)
                    break;
            }

            // -------------------------------------------------------------- 3. Admit waiting sequences.
            // --------------------------------------------------------------
            // Walk the queue in arrival order, each waiter once. A waiter held back for a
            // producer below keeps its place, so arrival order - and the adapter-epoch turns
            // that follow it - outlast the hold.
            LinkedListNode<SequenceState>? cursor = _waiting.First;
            while (cursor is { } node && tokenBudget > 0 && mixedPrefillLeft > 0
                && _running.Count < Math.Min(_cfg.MaxNumRunningSequences, _sequenceSlotCap))
            {
                cursor = node.Next;
                var seq = node.Value;

                // A sequence preempted earlier in this same output must stay
                // parked until the next iteration. Re-admitting it here would
                // put its id in both ScheduledWork and PreemptedRequestIds: the
                // engine would execute the new prefill and then release that
                // now-running request's model-owned cache after the step.
                if (output.PreemptedRequestIds.Contains(seq.RequestId))
                    break;

                // Adapter epochs: one step runs one adapter (or none), so a request whose
                // adapter differs from the running set's waits until that set drains, and
                // so does everything queued behind it. Strict first-come-first-served, so
                // neither side can starve the other. Base-only traffic never gets here.
                if (!ReferenceEquals(seq.Adapter, _epochAdapter))
                {
                    if (_running.Count > 0)
                    {
                        LogAdapterWait(seq);
                        break;
                    }
                    _epochAdapter = seq.Adapter;
                }

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
                bool soleAdmission = false;
                if (seq.BlockTable.NumBlocks == 0 && PrefixCachingActive)
                {
                    _radixCache!.PrimaryAvailable = _running.Count == 0 && output.ScheduledWork.Count == 0;
                    int length = _prefixCacheLcp?.Invoke(seq) ?? 0;
                    if (WaitForScheduledPublicCheckpoint(seq, output, length)
                        || WaitForLonePageProducer(seq, length, pageProducer, pageProducerEnd))
                    {
                        // The producer is already scheduled to advance this step.
                        // Let its prefill publish the longer common prefix before
                        // admitting a sibling that would compute the same tokens.
                        // It stays where it is and the walk moves on, so unrelated
                        // requests behind it can still run and a stalled producer
                        // never blocks work. (Moving it to the tail put it behind
                        // later arrivals: behind a later adapter request's whole
                        // epoch, too.)
                        prefillCandidatesRemaining = Math.Max(0, prefillCandidatesRemaining - 1);
                        continue;
                    }
                    if (length > 0 && _prefixCacheAdopt?.Invoke(seq, length) == true)
                        soleAdmission = _radixCache.RequiresSoleAdmission;
                    else if (length > 0 && _running.Count > 0 && _radixCache.LastAdoptDeclinedByModel)
                    {
                        // The model has no memory for another clone of a cached prefix now; a full
                        // re-prefill beside the running requests would need more. Wait for one to finish.
                        LogCloneWait(seq, length);
                        break;
                    }
                    string? why = seq.PrefixCacheReusedTokens == 0 ? _prefixCacheDeclineReason?.Invoke() : null;
                    string? source = seq.PrefixCacheReusedTokens > 0 ? _radixCache.LastSource : null;
                    _logger.LogInformation(
                        "Radix prompt reuse for {RequestId}: {Reused}/{Prompt} tokens{Source}; {Prefill} token(s) to prefill{Why}.",
                        seq.RequestId, seq.PrefixCacheReusedTokens, seq.PromptTokens.Count,
                        source == null ? string.Empty : $" from {source}",
                        seq.PromptTokens.Count - seq.PrefixCacheReusedTokens,
                        why == null ? string.Empty : $" ({why})");
                }

                int promptUncomputed = Math.Max(0, seq.PromptTokens.Count - seq.NumComputedTokens);
                if (promptUncomputed <= 0)
                {
                    // The whole prompt was a prefix-cache hit. Force a 1-token
                    // forward to produce fresh logits.
                    promptUncomputed = 1;
                }

                // The first prompt of an idle engine, when the requests behind it share its
                // prefix and only its lone step can publish it: they wait (above), so it takes
                // a lone prompt's step, cut where the prefix they share ends.
                int pageBoundary = pageProducer == null && _running.Count == 0 && output.ScheduledWork.Count == 0
                    ? LonePagePublishBoundary(seq) : 0;
                bool lone = noContention || pageBoundary > 0;
                int cap = GetPrefillCap(
                    lone, hasActiveDecode, tokenBudget,
                    Math.Max(1, prefillCandidatesRemaining), soloPrefillCap);
                if (lone) SoloPrefillStep(seq, promptUncomputed, ref cap, ref tokenBudget);
                int desired = Math.Min(promptUncomputed, cap);
                int want = Math.Min(desired, Math.Min(tokenBudget, mixedPrefillLeft));
                want = RunTinyRemainder(seq, want, desired, tokenBudget);
                want = AlignRecurrentPrefillBoundary(seq, want, promptUncomputed);
                want = AlignSharedPrefixBoundary(seq, want);
                want = CutAtBoundary(seq, want, pageBoundary);
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
                if (pageBoundary > 0)
                    NotePageProducer(seq, want, ref pageProducer, ref pageProducerEnd);
                tokenBudget -= want;
                mixedPrefillLeft -= want;
                prefillCandidatesRemaining = Math.Max(0, prefillCandidatesRemaining - 1);
                if (want < desired)
                    _nextPrefillRequestId = seq.RequestId;

                // A prefix the radix cache serves from the model's primary (live) cache
                // depends on that cache staying intact until this sequence runs. Don't
                // admit any other sequence this step (it could take ownership first and
                // reset the cache); the others wait one step.
                if (soleAdmission)
                    break;
            }

            // Every running sequence carries the epoch's adapter, so this step's do too.
            // A drained running set ends the epoch (and its reference to the adapter).
            if (_running.Count == 0)
                _epochAdapter = null;
            output.Adapter = output.ScheduledWork.Count > 0 ? _epochAdapter : null;
            return output;
        }

        /// <summary>Whether a running or waiting sequence still runs with
        /// <paramref name="adapter"/>, so the model must keep its copy of the factors.</summary>
        internal bool ReferencesAdapter(IRequestAdapter adapter)
        {
            foreach (var seq in _runningOrder)
                if (ReferenceEquals(seq.Adapter, adapter)) return true;
            foreach (var seq in _waiting)
                if (ReferenceEquals(seq.Adapter, adapter)) return true;
            return false;
        }

        /// <summary>How many waiters, at most <paramref name="limit"/>, the adapter-epoch gate lets
        /// phase 3 admit this step: the run at the head of the queue on the running set's adapter
        /// (or, with nothing running, on the head's). The rest wait for the epoch to drain, so
        /// they must not shrink the prefill share of what runs. For base-only traffic this is
        /// min(waiting, limit), the count used before epochs.</summary>
        private int CountEpochWaiting(int limit)
        {
            if (limit <= 0 || _waiting.First is not { } head)
                return 0;
            IRequestAdapter? epoch = _running.Count > 0 ? _epochAdapter : head.Value.Adapter;
            int count = 0;
            for (var node = head; node != null && count < limit && ReferenceEquals(node.Value.Adapter, epoch); node = node.Next)
                count++;
            return count;
        }

        private void LogAdapterWait(SequenceState seq)
        {
            if (string.Equals(_adapterWaitLoggedFor, seq.RequestId, StringComparison.Ordinal))
                return;
            _adapterWaitLoggedFor = seq.RequestId;
            _logger.LogInformation(
                "{RequestId} (adapter {Adapter}) waits for the {Running} running request(s) of adapter {Epoch} " +
                "to finish: one step runs one LoRA adapter (or the base model), so requests of different " +
                "adapters take turns, in arrival order.",
                seq.RequestId, seq.Adapter?.Id ?? "none", _running.Count, _epochAdapter?.Id ?? "none");
        }

        /// <summary>The prefill a step whose mixed prompt budget ran out resumes with: the
        /// one this step served, while its prompt lasts, when the oldest prompt keeps the
        /// budget (MixedStepPrefillFirstComeFirstServed); otherwise the next in rotation.</summary>
        private string MixedPrefillResumePoint(SequenceState next, SequenceState? served, int servedTokens, int mixedPrefillLeft)
        {
            if (mixedPrefillLeft <= 0 && _cfg.MixedStepPrefillFirstComeFirstServed && served != null
                && served.NumComputedTokens + servedTokens < served.PromptTokens.Count)
                return served.RequestId;
            return next.RequestId;
        }

        private bool WaitForScheduledPublicCheckpoint(SequenceState sequence, SchedulerOutput output, int reusableLength)
        {
            if (_radixCache == null || !_alignToSharedPrefix) return false;
            foreach (ScheduledSequenceWork work in output.ScheduledWork)
                if (work.NumScheduledTokens > 0
                    && (_checkpointGapTokens == 0 || EndsOnPublicBoundary(work))
                    && _radixCache.CanSharePendingPublicCheckpoint(sequence, work.Sequence, reusableLength))
                    return true;
            return false;
        }

        // -------------------------------------------------------------- lone page producers
        //
        // A family whose only reuse across requests is its pages, written by a lone forward
        // alone (PrefixCacheCoordinator.PagesNeedLoneForward: Nemotron-H's recurrent pages hold
        // the Mamba2 state only where a per-sequence Forward ended, and a step of two prompts
        // runs batched, publishing nothing). Before this, requests that shared a system prompt
        // shared it only when the first happened to reach an idle engine alone: when two or more
        // arrived before the engine's next step - as a warm server's concurrent burst always does
        // - they were admitted together, each forwarded the whole prompt batched, no page of it
        // was kept, and the next burst found nothing either (kvReused=0 on every request, a full
        // re-prefill each time). Now the first prompt forwards the shared prefix alone, cut at its
        // block boundary, and the others wait that one step and resume from its page.

        /// <summary>
        /// Where <paramref name="producer"/>'s next prefill step should end so the requests waiting
        /// behind it resume from the page it publishes there; 0 for no such step. Only for a family
        /// whose pages only a lone forward writes, only for a producer whose step stays on the
        /// linear path, and only when EVERY request this step could admit shares that prefix with
        /// it: one that does not would join the step, which then runs batched and publishes
        /// nothing. The boundary is the shortest block-aligned prefix they share with it, so its
        /// page serves them all.
        /// </summary>
        private int LonePagePublishBoundary(SequenceState producer)
        {
            if (_waiting.First is not { } head || (head.Next == null && ReferenceEquals(head.Value, producer))
                || !PrefixCachingActive || producer.KvStateInPagedStorage || producer.MediaSpans.Count > 0
                || !_radixCache!.PagesNeedLoneForward)
                return 0;
            // Every waiter phase 3 may visit: a held one takes no running slot, so the running cap
            // does not end the visits; only a request of another adapter epoch does.
            int window = CountEpochWaiting(int.MaxValue);
            int boundary = int.MaxValue;
            int sharers = 0, visited = 0;
            for (var node = _waiting.First; node != null && visited < window; node = node.Next, visited++)
            {
                if (ReferenceEquals(node.Value, producer))
                    continue;
                boundary = Math.Min(boundary, SharedPageBoundary(node.Value, producer));
                if (boundary <= producer.NumComputedTokens)
                    return 0;
                sharers++;
            }
            return sharers > 0 ? boundary : 0;
        }

        /// <summary>
        /// The longest block-aligned prefix of <paramref name="sequence"/>'s prompt that a page
        /// <paramref name="producer"/> publishes can serve: the tokens the two prompts share, short
        /// of <paramref name="sequence"/>'s last prompt token (always forwarded for its logits),
        /// inside both requests' cache breakpoints, before any media, and inside both requests'
        /// public shared prefix (a page past the producer's is its own conversation's alone).
        /// 0 for a request that is not new or may not reuse at all - an adapter request is
        /// cache-isolated (its breakpoint limit is 0) - and for a producer that publishes nothing.
        /// </summary>
        private int SharedPageBoundary(SequenceState sequence, SequenceState producer)
        {
            if (ReferenceEquals(sequence, producer) || sequence.FirstScheduledAt != null
                || sequence.NumComputedTokens != 0 || sequence.BlockTable.NumBlocks != 0)
                return 0;
            int limit = Math.Min(sequence.PromptTokens.Count - 1, producer.PromptTokens.Count);
            limit = Math.Min(limit, Math.Min(sequence.SharedPrefixTokens, producer.SharedPrefixTokens));
            if (sequence.CacheBreakpoints != null) limit = Math.Min(limit, sequence.CacheBreakpointLimit);
            if (producer.CacheBreakpoints != null) limit = Math.Min(limit, producer.CacheBreakpointLimit);
            foreach (var span in sequence.MediaSpans) limit = Math.Min(limit, span.Start);
            int shared = 0;
            while (shared < limit && sequence.PromptTokens[shared] == producer.PromptTokens[shared])
                shared++;
            return shared - shared % _cfg.BlockSize;
        }

        /// <summary>End a prefill chunk at <paramref name="boundary"/> when it would run past it.</summary>
        private static int CutAtBoundary(SequenceState seq, int want, int boundary)
        {
            int start = seq.NumComputedTokens;
            return boundary > start && boundary < start + want ? boundary - start : want;
        }

        /// <summary>Record <paramref name="seq"/>'s step of <paramref name="want"/> tokens as this
        /// step's lone page producer when it ends on a block boundary: the page it publishes is a
        /// restore point only there.</summary>
        private void NotePageProducer(SequenceState seq, int want, ref SequenceState? producer, ref int producerEnd)
        {
            int end = seq.NumComputedTokens + want;
            if (want > 0 && end % _cfg.BlockSize == 0)
            {
                producer = seq;
                producerEnd = end;
            }
        }

        /// <summary>Whether <paramref name="sequence"/> waits for this step's lone page producer:
        /// the page it publishes at <paramref name="producerEnd"/> serves more of its prompt than
        /// the <paramref name="reusableLength"/> tokens the cache serves now.</summary>
        private bool WaitForLonePageProducer(
            SequenceState sequence, int reusableLength, SequenceState? producer, int producerEnd)
        {
            if (producer == null || producerEnd <= reusableLength
                || SharedPageBoundary(sequence, producer) < producerEnd)
                return false;
            if (!string.Equals(_lonePageWaitLoggedFor, producer.RequestId, StringComparison.Ordinal))
            {
                _lonePageWaitLoggedFor = producer.RequestId;
                _logger.LogInformation(
                    "{RequestId} and the requests sharing its first {Tokens} prompt tokens wait one step for {Producer} " +
                    "to forward them alone: this model keeps a shared prefix only from a lone prefill step, so they " +
                    "then resume from it instead of each forwarding it again.",
                    sequence.RequestId, producerEnd, producer.RequestId);
            }
            return true;
        }

        private string? _lonePageWaitLoggedFor;

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

        /// <summary>A lone prompt's step from the model's policy, when it has one for this
        /// position: it replaces the cap and, for this one sequence, the token budget.</summary>
        private void SoloPrefillStep(SequenceState seq, int promptUncomputed, ref int cap, ref int tokenBudget)
        {
            if (_soloPrefillPolicy == null || promptUncomputed <= 0) return;
            int step = _soloPrefillPolicy.PreferredSoloPrefillTokens(seq.NumComputedTokens, promptUncomputed);
            if (step <= 0) return;
            // Image rows replace token embeddings inside the call, which a model
            // that pipelines its prefill runs one micro-batch at a time: such a
            // step keeps the scheduler's own, budget-bounded size.
            int s0 = seq.NumComputedTokens, s1 = s0 + step;
            foreach (var span in seq.MediaSpans)
                if (span.Start < s1 && span.End > s0) return;
            cap = step;
            tokenBudget = Math.Max(tokenBudget, step);
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
            if (_cfg.PrefillOnlyFirstComeFirstServed)
                return Math.Max(1, tokenBudget);

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
        public bool NotifyError(SequenceState seq, Exception error, SchedulerOutput? output = null)
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
            Exception? error = null)
        {
            if (seq == null || seq.Status.IsFinished()) return false;
            // A finishing request frees a slot; let admission try again.
            _sequenceSlotCap = int.MaxValue;

            // Cache the final partial trailing block into the prefix cache
            // (if there are any full blocks) before freeing.
            if (cacheBlocks)
                CacheFullBlocksForSequence(seq);

            // A family that serves concurrency on the batched route keeps a cleanly finished sequence's blocks
            // and its own state for the conversation's next turn; the prefix cache's references outlive the free.
            if (cacheBlocks && error == null && _radixCache != null
                && finalStatus is SequenceStatus.FinishedStopped or SequenceStatus.FinishedLengthCapped
                    or SequenceStatus.FinishedAborted)
                _radixCache.RetainPagedFinished(seq);

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

            _radixCache?.EnsureFreePages(delta);
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

            // The prefix every conversation shares (SharedPrefixTokens), on a family that keeps it
            // across requests only as pages with the state at their end (no public checkpoint holds
            // it; PrefixCacheCoordinator.SharedPrefixPageBoundary). A lone prompt that runs past it
            // was split only at its own tail boundary - 2048 for a 2067-token prompt behind a
            // 1890-token system prompt - so every page of the shared prefix was cached without
            // state and no later request could resume from one: kvReused=0 on every request of an
            // A40 run whose first, lone request happened to be the long one. Snap the round to the
            // shared prefix's block boundary as well. It never lies past the marker's (the gate
            // checks the breakpoint limit), and a request that reused it starts at or past it.
            int sharedBoundary = _radixCache?.SharedPrefixPageBoundary(seq) ?? 0;
            if (sharedBoundary > start && sharedBoundary < end)
                return sharedBoundary - start;

            // An explicit client breakpoint (cache_control / prompt_cache_breakpoint)
            // caps both registration and adoption at its block boundary. For a
            // recurrent model the blocks inside a fused round are non-restorable
            // (they hold their K/V rows only, no recurrent state), so without a real
            // checkpoint at the marker the marked prefix stays in the index but is
            // never adoptable - every follow-up request matches it and re-prefills
            // it. Snap the round
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
            // Blocks of a cached prefix a running request already holds through its own clone of it
            // are shared, not allocated (PrefixCacheCoordinator.SharedClones). Admission only runs
            // this check beside running requests, so the primary cache is not on offer.
            if (_radixCache != null && need > 0)
            {
                _radixCache.PrimaryAvailable = false;
                need -= _radixCache.ShareablePrefixBlocks(candidate);
            }
            _radixCache?.EnsureFreePages((int)Math.Min(_pool.NumBlocks, Math.Max(0, need + outstanding)));
            return (long)_pool.NumFreeBlocks - outstanding >= need;
        }

        private string? _cloneWaitLoggedFor;

        private void LogCloneWait(SequenceState seq, int length)
        {
            if (string.Equals(_cloneWaitLoggedFor, seq.RequestId, StringComparison.Ordinal))
                return;
            _cloneWaitLoggedFor = seq.RequestId;
            _logger.LogInformation(
                "Prefix cache: {RequestId} waits to clone its {Length}-token cached prefix; the model has no memory for " +
                "another sequence beside the {Running} running request(s). It is admitted when one of them finishes.",
                seq.RequestId, length, _running.Count);
        }

        private string? _capacityWaitLoggedFor;

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
            SequenceState? victim = null;
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

        // Requests the model can hold slots for, learned when a new request found none
        // (DeferForSequenceSlot); lifted when any request finishes and frees one.
        private int _sequenceSlotCap = int.MaxValue;
        private bool _sequenceSlotCapReported;

        /// <summary>
        /// The model had no device slot for <paramref name="seq"/> (<see cref="SequenceSlotUnavailableException"/>):
        /// park it back at the front of the waiting queue, and admit nothing past the requests that hold slots
        /// until one finishes. Without the cap it would be re-admitted, and fail to get a slot, every step.
        /// At least one request is always admissible: a lone request runs on the model's primary cache.
        /// </summary>
        public void DeferForSequenceSlot(SequenceState seq, SchedulerOutput output, int contextTokens)
        {
            if (seq == null || seq.Status.IsFinished() || !_running.ContainsKey(seq.RequestId)) return;
            PreemptSequence(seq);
            output.PreemptedRequestIds.Add(seq.RequestId);
            _sequenceSlotCap = Math.Max(1, _running.Count);
            if (!_sequenceSlotCapReported)
            {
                _sequenceSlotCapReported = true;
                _logger.LogInformation(
                    "The model has no device slot for another concurrent request: {Running} running request(s) hold " +
                    "every slot its memory fits, so {RequestId} and later arrivals wait for one to finish. Each slot " +
                    "holds a whole context ({Context} tokens); a smaller MAX_CONTEXT fits more at once. Reported once.",
                    _running.Count, seq.RequestId, contextTokens);
            }
        }

        /// <summary>Take a running sequence's blocks back and re-park it in
        /// the waiting queue. Full blocks go to the prefix cache before freeing
        /// so its re-admission recovers most of the work.</summary>
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
            _waitingIndex[victim.RequestId] = _waiting.AddFirst(victim);
        }

        /// <summary>After advancing tokens or finishing, hand the sequence's newly
        /// full blocks to the prefix cache.</summary>
        public void OnBlocksCommitted(SequenceState seq, int previousTokens)
        {
            if (!PrefixCachingActive) return;
            int prevFull = previousTokens / _cfg.BlockSize;
            int curFull = seq.NumComputedTokens / _cfg.BlockSize;
            if (curFull > prevFull)
                _radixCache!.CapturePages(seq);
        }
        private void CacheFullBlocksForSequence(SequenceState seq)
        {
            // The sequence stops decoding here, so its newest decode restore point is
            // final and goes to the prefix cache with the rest (OnBlocksCommitted keeps it
            // back while a later one may still replace it).
            seq.HeldDecodeRestoreBlock = -1;
            if (PrefixCachingActive)
                _radixCache!.CapturePages(seq);
        }
    }
}
