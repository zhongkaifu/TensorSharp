// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;
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
        private readonly Dictionary<string, MemoryCharge[]> _memoryPeaks = new(StringComparer.Ordinal);
        private readonly Dictionary<string, SequenceState> _memoryOwners = new(StringComparer.Ordinal);

        public ContinuousBatchScheduler(
            SchedulerConfig cfg,
            BlockPool pool,
            ILogger? logger = null,
            bool supportsCrossSequenceKvReuse = true,
            bool requiresPerBlockCapture = false)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cfg.PrefillChunkTokenLimit);
            cfg.MemoryAdmission?.ValidateConfiguration(cfg);
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

        /// <summary>
        /// Cut a prefill chunk at the sequence's public checkpoint boundaries when the chunk
        /// would otherwise run past it, so the state the executor checkpoints is the
        /// state after exactly those tokens. Reused boundaries are already behind
        /// the sequence's computed position and need no additional forward pass.
        /// </summary>
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
                foreach (int boundary in seq.PublicCheckpointBoundaries)
                    if (boundary > start && boundary < start + want && seq.IsPublicCheckpointBoundary(boundary))
                        want = boundary - start;
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
        public bool MemoryAdmissionBlocked { get; private set; }
        internal IReadOnlyList<MemoryCharge>? BlockedMemoryPeak { get; private set; }
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
            if (_cfg.MemoryAdmission?.ExecutionShape is { } shape && seq.BlockTable.BlockSize != shape.BlockTokens)
                throw new ArgumentException("The request block size differs from its admitted snapshot geometry.", nameof(seq));
            if (_waitingIndex.ContainsKey(seq.RequestId) || _running.ContainsKey(seq.RequestId))
                throw new InvalidOperationException($"Sequence {seq.RequestId} is already submitted.");
            if (_memoryOwners.ContainsKey(seq.RequestId))
                throw new InvalidOperationException("The previous request with this id has not released its physical memory.");

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

            if (_cfg.MemoryAdmission is { } admission)
            {
                if (_waiting.Count >= admission.MaxQueuedRequests)
                    throw new MemoryPressureException("The memory admission queue is full.");
                var peak = admission.EstimatePeak(seq).ToArray();
                if (!admission.Budget.CanEverFit(peak))
                    throw new MemoryPressureException("Request peak exceeds a configured physical memory pool even in isolation.");
                _memoryPeaks.Add(seq.RequestId, peak);
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
            MemoryAdmissionBlocked = false;
            BlockedMemoryPeak = null;
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
            int waitingVisitsRemaining = _waiting.Count;
            while (_waiting.First is { } node && tokenBudget > 0 && _running.Count < Math.Min(_cfg.MaxNumRunningSequences, _sequenceSlotCap)
                && waitingVisitsRemaining-- > 0)
            {
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
                if (_cfg.MemoryAdmission is { } admission && seq.MemoryEnvelope == null)
                {
                    var peak = _memoryPeaks[seq.RequestId];
                    // A budget can shrink while a request is queued. Waiting on its
                    // change signal cannot help an envelope that no longer fits even
                    // in isolation; reject just this request and visit the next one.
                    if (!admission.Budget.CanEverFit(peak))
                    {
                        NotifyError(seq, new MemoryPressureException(
                            "Request peak exceeds a physical memory pool after its capacity changed."), output);
                        continue;
                    }
                    var envelope = admission.Budget.TryReserve(peak);
                    if (envelope == null)
                    {
                        MemoryAdmissionBlocked = true;
                        BlockedMemoryPeak = peak;
                        break;
                    }
                    seq.MemoryEnvelope = envelope;
                    _memoryOwners.Add(seq.RequestId, seq);
                }
                bool soleAdmission = false;
                if (seq.BlockTable.NumBlocks == 0 && PrefixCachingActive)
                {
                    _radixCache!.PrimaryAvailable = _running.Count == 0 && output.ScheduledWork.Count == 0;
                    int length = _prefixCacheLcp?.Invoke(seq) ?? 0;
                    if (WaitForScheduledPublicCheckpoint(seq, output, length))
                    {
                        // The producer is already scheduled to advance this step.
                        // Let its prefill publish the longer common prefix before
                        // admitting a sibling that would compute the same tokens.
                        // Visit each waiter once so unrelated requests behind it
                        // can still run and a stalled producer never blocks work.
                        _waiting.Remove(node);
                        _waiting.AddLast(node);
                        prefillCandidatesRemaining = Math.Max(0, prefillCandidatesRemaining - 1);
                        continue;
                    }
                    if (length > 0 && _prefixCacheAdopt?.Invoke(seq, length) == true)
                        soleAdmission = _radixCache.RequiresSoleAdmission;
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

                // A prefix the radix cache serves from the model's primary (live) cache
                // depends on that cache staying intact until this sequence runs. Don't
                // admit any other sequence this step (it could take ownership first and
                // reset the cache); the others wait one step.
                if (soleAdmission)
                    break;
            }

            return output;
        }

        private bool WaitForScheduledPublicCheckpoint(SequenceState sequence, SchedulerOutput output, int reusableLength)
        {
            if (_radixCache == null || !_alignToSharedPrefix) return false;
            foreach (ScheduledSequenceWork work in output.ScheduledWork)
                if (work.NumScheduledTokens > 0
                    && _radixCache.CanSharePendingPublicCheckpoint(sequence, work.Sequence, reusableLength))
                    return true;
            return false;
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
            candidatesRemaining = Math.Max(1, candidatesRemaining);
            int evenShare = tokenBudget / candidatesRemaining;
            if (tokenBudget % candidatesRemaining != 0)
                evenShare++;
            int cap = noContention ? soloPrefillCap : hasActiveDecode
                ? _cfg.MaxPrefillChunkSize : Math.Max(1, evenShare);
            cap = Math.Min(cap, _cfg.PrefillChunkTokenLimit);
            // Fairness and solo-throughput policies may increase a chunk, but
            // cannot exceed the shape whose workspace was admitted. Apply this
            // to both existing and newly admitted prefills.
            return _cfg.MemoryAdmission?.ExecutionShape is { } shape
                ? Math.Min(cap, shape.MaximumPrefillTokens) : cap;
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

            seq.BlockTable.ReleaseAll(_pool);

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
            _memoryPeaks.Remove(seq.RequestId);

            return true;
        }

        /// <summary>Call only AFTER model release/fences, including preemption and
        /// cancellation. Retained allocations drawn from the envelope remain charged.
        /// A failing release hook must leave its envelope reserved for recovery.</summary>
        public void NotifyMemoryReleased(string requestId)
        {
            if (_memoryOwners.TryGetValue(requestId, out var seq))
            {
                if (seq.Status == SequenceStatus.Running)
                    throw new InvalidOperationException("Cannot release a running request's memory envelope.");
                seq.MemoryEnvelope!.Dispose();
                seq.MemoryEnvelope = null;
                _memoryOwners.Remove(requestId);
            }
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
            _radixCache?.EnsureFreePages((int)Math.Min(_pool.NumBlocks, Math.Max(0, need + outstanding)));
            return (long)_pool.NumFreeBlocks - outstanding >= need;
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
            victim.BlockTable.ReleaseAll(_pool);

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
