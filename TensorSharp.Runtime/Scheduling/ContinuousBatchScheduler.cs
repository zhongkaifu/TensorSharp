// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
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

        // Live-cache continuation hooks (wired by the engine to the executor). The
        // first computes how many leading prompt tokens can be served by continuing
        // the model's live KV cache (beyond the pooled-snapshot cap); the second
        // sets the sequence up to do so. Lets same-session follow-up turns reuse the
        // whole conversation prefix on sliding-window models. Null when unwired.
        private Func<SequenceState, int> _liveContinuationLcp;
        private Func<SequenceState, int, bool> _liveContinuationAdopt;

        // Retained fused-cache continuation hooks (wired by the engine to the
        // executor). Cross-request analogue of the live-cache hooks above: they
        // re-adopt a FINISHED concurrent request's retained per-request KV holder
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

        public ContinuousBatchScheduler(
            SchedulerConfig cfg,
            BlockPool pool,
            string modelFingerprint,
            ILogger logger = null,
            bool supportsCrossSequenceKvReuse = true,
            int maxReusablePrefixTokens = int.MaxValue)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            _pool = pool ?? throw new ArgumentNullException(nameof(pool));
            _fingerprint = modelFingerprint ?? string.Empty;
            _crossSeqKvReuse = supportsCrossSequenceKvReuse;
            _maxReusablePrefixTokens = maxReusablePrefixTokens <= 0 ? int.MaxValue : maxReusablePrefixTokens;
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
        public void AttachFusedCacheContinuation(
            Func<SequenceState, int> computeLcp,
            Func<SequenceState, int, bool> adopt)
        {
            _fusedContinuationLcp = computeLcp;
            _fusedContinuationAdopt = adopt;
        }

        public int WaitingCount => _waiting.Count;
        public int RunningCount => _running.Count;
        public BlockPool Pool => _pool;
        public SchedulerConfig Config => _cfg;

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

        /// <summary>Submit a sequence. It enters the waiting queue; the next
        /// <see cref="Schedule"/> call will try to admit it.</summary>
        public void Submit(SequenceState seq)
        {
            if (seq == null) throw new ArgumentNullException(nameof(seq));
            if (_waitingIndex.ContainsKey(seq.RequestId) || _running.ContainsKey(seq.RequestId))
                throw new InvalidOperationException($"Sequence {seq.RequestId} is already submitted.");

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
            int prefillCap = noContention
                ? Math.Min(_cfg.SoloPrefillChunkSize, _cfg.MaxNumBatchedTokens)
                : _cfg.MaxPrefillChunkSize;

            // -------------------------------------------------------------- 1. Run existing running set first.
            //    This guarantees decoding sequences make forward progress even
            //    when long-prompt waiters are vying for blocks.
            // --------------------------------------------------------------
            // Snapshot the order so we can mutate _runningOrder safely on preemption.
            var runningSnapshot = new List<SequenceState>(_runningOrder);
            foreach (var seq in runningSnapshot)
            {
                if (tokenBudget <= 0) break;
                if (!_running.ContainsKey(seq.RequestId)) continue;

                int promptUncomputed = Math.Max(0, seq.PromptTokens.Count - seq.NumComputedTokens);
                bool isPrefill = promptUncomputed > 0;
                int want = isPrefill
                    ? Math.Min(promptUncomputed, prefillCap)
                    : 1; // decode step
                want = Math.Min(want, tokenBudget);
                if (want <= 0) break;

                // Allocate any blocks needed to host these tokens.
                if (!TryEnsureBlocksForStep(seq, want))
                {
                    // Out of blocks. Try preempting a lower-priority running seq.
                    if (!TryPreemptForBlocks(seq, want, output))
                    {
                        // Still can't fit: skip this seq this step.
                        continue;
                    }
                }

                bool isFresh = seq.FirstScheduledAt == null;
                if (isFresh) seq.FirstScheduledAt = DateTime.UtcNow;

                output.ScheduledWork.Add(new ScheduledSequenceWork(seq, want, isFresh, isPrefill));
                tokenBudget -= want;
            }

            // -------------------------------------------------------------- 2. Admit waiting sequences.
            // --------------------------------------------------------------
            while (_waiting.Count > 0 && tokenBudget > 0 && _running.Count < _cfg.MaxNumRunningSequences)
            {
                var node = _waiting.First;
                var seq = node.Value;

                // Try prefix cache lookup before allocating blocks (only for
                // brand-new sequences; preempted ones already had their blocks
                // freed and need a fresh re-prefill, no shortcut).
                bool plannedLiveContinuation = false;
                bool plannedFusedContinuation = false;
                if (seq.BlockTable.NumBlocks == 0 && PrefixCachingActive)
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
                        if (lcp > 0 && _liveContinuationAdopt(seq, lcp))
                            plannedLiveContinuation = true;
                    }

                    // Retained fused-cache continuation: a finished concurrent
                    // request's full circular KV is kept alive; if this prompt extends
                    // it exactly, continue from that retained holder (reusing the whole
                    // conversation prefix past the pooled window cap). Each retained
                    // holder is independent ÔÇö no shared live cache to clobber ÔÇö so this
                    // is NOT gated to the sole-sequence case and doesn't block
                    // co-admitting other sequences this step. This is the path that
                    // gives multi-turn "Þ»Àþ╗ºþ╗¡" follow-ups their prefix reuse back after
                    // a concurrent (per-seq fused) round left nothing in the pool.
                    if (!plannedLiveContinuation
                        && _fusedContinuationLcp != null
                        && _fusedContinuationAdopt != null)
                    {
                        int flcp = _fusedContinuationLcp(seq);
                        if (flcp > 0 && _fusedContinuationAdopt(seq, flcp))
                            plannedFusedContinuation = true;
                    }

                    if (!plannedLiveContinuation && !plannedFusedContinuation)
                        AdoptPrefixBlocksCapped(seq);
                }

                int promptUncomputed = Math.Max(0, seq.PromptTokens.Count - seq.NumComputedTokens);
                if (promptUncomputed <= 0)
                {
                    // The whole prompt was a prefix-cache hit. Force a 1-token
                    // forward to produce fresh logits.
                    promptUncomputed = 1;
                }

                int want = Math.Min(promptUncomputed, prefillCap);
                want = Math.Min(want, tokenBudget);
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

                // A live-cache continuation depends on the model's live cache staying
                // intact until this sequence runs. Don't admit any other sequence this
                // step (it could take ownership first and reset the cache); the others
                // wait one step.
                if (plannedLiveContinuation)
                    break;
            }

            return output;
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

        /// <summary>Preempt the lowest-priority running sequence (other than
        /// <paramref name="needyForBlocks"/>) and free its blocks, then retry
        /// allocation. Returns true if successful.</summary>
        private bool TryPreemptForBlocks(SequenceState needyForBlocks, int extraTokens, SchedulerOutput output)
        {
            // Find the candidate to preempt: highest sn (latest submission) and
            // lowest priority, not the requester.
            SequenceState victim = null;
            int victimScore = int.MinValue;
            foreach (var s in _runningOrder)
            {
                if (ReferenceEquals(s, needyForBlocks)) continue;
                int score = -s.Priority * 1_000_000 + (int)(s.Sn & 0xfffff);
                if (score > victimScore)
                {
                    victimScore = score;
                    victim = s;
                }
            }
            if (victim == null) return false;

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
            if (seq.PromptTokens.Count < _cfg.BlockSize) return;

            var hashes = KvBlockHasher.ComputeBlockHashes(seq.PromptTokens, _cfg.BlockSize, EffectiveFingerprint(seq));
            int maxAdoptableTokens = Math.Max(0, seq.PromptTokens.Count - 1);
            // Cap reuse at the model's reliably-restorable prefix length (sliding
            // window for circular caches). Adopting beyond this would inject a wrapped
            // snapshot that the model can't faithfully reconstruct -> corrupt output.
            if (_maxReusablePrefixTokens != int.MaxValue)
                maxAdoptableTokens = Math.Min(maxAdoptableTokens, _maxReusablePrefixTokens);
            int maxAdoptableBlocks = maxAdoptableTokens / _cfg.BlockSize;

            int adopted = 0;
            for (int i = 0; i < hashes.Count && i < maxAdoptableBlocks; i++)
            {
                if (!_pool.TryFindByHash(hashes[i], out var block))
                    break;
                _pool.Touch(block);
                seq.BlockTable.AppendBlock(block);
                adopted++;
            }

            int adoptedTokens = adopted * _cfg.BlockSize;
            if (adoptedTokens > 0)
            {
                seq.PrefixCacheReusedTokens = adoptedTokens;
                seq.SetComputedTokensForPrefixAdoption(adoptedTokens);
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
                _pool.RegisterFullBlock(block, hashes[b], _cfg.BlockSize);
            }
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
            return KvBlockHasher.ComputeBlockHashes(list, _cfg.BlockSize, EffectiveFingerprint(seq));
        }

        /// <summary>
        /// The model fingerprint, additionally salted with the sequence's media
        /// fingerprint when the prompt carries images/audio/video. This keeps the
        /// prefix-cache block hashes content-aware: identical media reuses cached
        /// K/V, but different media (sharing the same placeholder token IDs) can
        /// never adopt a stale neighbour's blocks. Text-only sequences fall back to
        /// the bare model fingerprint, so their hashes are unchanged.
        /// </summary>
        private string EffectiveFingerprint(SequenceState seq)
        {
            string media = seq?.MediaFingerprint;
            if (string.IsNullOrEmpty(media))
                return _fingerprint;
            return string.Concat(_fingerprint, "mm:", media);
        }
    }
}
