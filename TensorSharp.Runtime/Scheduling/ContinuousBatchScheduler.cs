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
        private readonly ILogger _logger;
        private readonly string _fingerprint;

        private readonly LinkedList<SequenceState> _waiting = new();
        private readonly Dictionary<string, LinkedListNode<SequenceState>> _waitingIndex = new();

        // Running set: keyed by request id, ordered by sn for fairness.
        private readonly Dictionary<string, SequenceState> _running = new();
        private readonly List<SequenceState> _runningOrder = new();

        public ContinuousBatchScheduler(
            SchedulerConfig cfg,
            BlockPool pool,
            string modelFingerprint,
            ILogger logger = null)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            _pool = pool ?? throw new ArgumentNullException(nameof(pool));
            _logger = logger ?? NullLogger.Instance;
            _fingerprint = modelFingerprint ?? string.Empty;
        }

        public int WaitingCount => _waiting.Count;
        public int RunningCount => _running.Count;
        public BlockPool Pool => _pool;
        public SchedulerConfig Config => _cfg;

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
                _waiting.Remove(node);
                _waitingIndex.Remove(requestId);
                node.Value.Status = SequenceStatus.FinishedAborted;
                return true;
            }
            if (_running.TryGetValue(requestId, out var seq))
            {
                FinishSequence(seq, SequenceStatus.FinishedAborted, "aborted");
                return true;
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
                    ? Math.Min(promptUncomputed, _cfg.MaxPrefillChunkSize)
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
                if (seq.BlockTable.NumBlocks == 0 && _cfg.EnablePrefixCaching)
                {
                    AdoptPrefixBlocksCapped(seq);
                }

                int promptUncomputed = Math.Max(0, seq.PromptTokens.Count - seq.NumComputedTokens);
                if (promptUncomputed <= 0)
                {
                    // The whole prompt was a prefix-cache hit. Force a 1-token
                    // forward to produce fresh logits.
                    promptUncomputed = 1;
                }

                int want = Math.Min(promptUncomputed, _cfg.MaxPrefillChunkSize);
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
            FinishSequence(seq, finalStatus, reason);
            output.FinishedRequestIds.Add(seq.RequestId);
        }

        /// <summary>Free a finished sequence's blocks and remove from running.</summary>
        private void FinishSequence(SequenceState seq, SequenceStatus finalStatus, string reason)
        {
            if (seq.Status.IsFinished()) return;

            // Cache the final partial trailing block into the prefix cache
            // (if there are any full blocks) before freeing.
            CacheFullBlocksForSequence(seq);

            var freed = seq.BlockTable.Clear();
            if (freed.Count > 0) _pool.Free(freed);

            seq.Status = finalStatus;
            seq.FinishReason = reason;
            if (_running.Remove(seq.RequestId))
                _runningOrder.Remove(seq);
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

            var hashes = KvBlockHasher.ComputeBlockHashes(seq.PromptTokens, _cfg.BlockSize, _fingerprint);
            int maxAdoptableTokens = Math.Max(0, seq.PromptTokens.Count - 1);
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
            if (!_cfg.EnablePrefixCaching) return;
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
            if (!_cfg.EnablePrefixCaching) return;
            int curFull = seq.NumComputedTokens / _cfg.BlockSize;
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
                // junk data — a future sequence with a matching prompt
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
            return KvBlockHasher.ComputeBlockHashes(list, _cfg.BlockSize, _fingerprint);
        }
    }
}
