// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Per-request mutable state owned by the scheduler. Combines vLLM's
    /// <c>Request</c> + <c>SequenceData</c> + <c>BlockTable</c> entry into one
    /// object since TensorSharp does not split request and sequence the way
    /// vLLM does (we don't yet support n&gt;1 sampling per request).
    /// </summary>
    public sealed class SequenceState
    {
        private static long _counter;

        public SequenceState(
            string requestId,
            IReadOnlyList<int> promptTokens,
            int maxNewTokens,
            int blockSize,
            SamplingConfig samplingConfig,
            object userTag = null,
            string mediaFingerprint = null)
        {
            if (promptTokens == null) throw new ArgumentNullException(nameof(promptTokens));
            if (promptTokens.Count == 0) throw new ArgumentException("Prompt must be non-empty.", nameof(promptTokens));
            if (maxNewTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens));

            Sn = Interlocked.Increment(ref _counter);
            RequestId = requestId ?? $"seq-{Sn:X}";
            BlockTable = new BlockTable(blockSize);
            PromptTokens = new List<int>(promptTokens);
            OutputTokens = new List<int>(capacity: Math.Min(maxNewTokens, 256));
            MaxNewTokens = maxNewTokens;
            SamplingConfig = samplingConfig ?? SamplingConfig.Default;
            Status = SequenceStatus.Waiting;
            SubmittedAt = DateTime.UtcNow;
            UserTag = userTag;
            MediaFingerprint = string.IsNullOrEmpty(mediaFingerprint) ? null : mediaFingerprint;
        }

        /// <summary>Monotonic submission sequence number. Used as FCFS tiebreaker
        /// when multiple sequences share the same priority.</summary>
        public long Sn { get; }

        public string RequestId { get; }
        public List<int> PromptTokens { get; }
        public List<int> OutputTokens { get; }
        public int MaxNewTokens { get; }
        public SamplingConfig SamplingConfig { get; }

        /// <summary>Sticky reference the caller can use to associate a session,
        /// HTTP request, or telemetry context with this sequence.</summary>
        public object UserTag { get; }

        /// <summary>
        /// Stable identifier of the multimodal content (images/audio/video) baked
        /// into this prompt's embeddings, or <c>null</c> for text-only prompts.
        /// Mixed into the prefix-cache block hashes so two prompts that share
        /// identical placeholder token IDs but carry <em>different</em> media never
        /// adopt each other's K/V blocks (which would otherwise surface a stale
        /// image/audio). Identical media still shares the cache.
        /// </summary>
        public string MediaFingerprint { get; }

        public SequenceStatus Status { get; internal set; }
        public DateTime SubmittedAt { get; }
        public int Priority { get; set; } = 0;

        /// <summary>The sequence's logical→physical block mapping.</summary>
        public BlockTable BlockTable { get; }

        /// <summary>Total tokens whose K/V is committed to the block table. After
        /// each step the executor calls <see cref="AdvanceComputedTokens"/>.</summary>
        public int NumComputedTokens { get; private set; }

        /// <summary>Total tokens in this sequence (prompt + generated so far).</summary>
        public int NumTotalTokens => PromptTokens.Count + OutputTokens.Count;

        /// <summary>Tokens that still need a forward pass to be committed
        /// (prefill + any not-yet-decoded generation).</summary>
        public int NumUncomputedTokens => NumTotalTokens - NumComputedTokens;

        /// <summary>The logits produced by the most recent forward at the
        /// sequence's "current" position. Used by the next step's sampler.</summary>
        public float[] LastLogits { get; internal set; }

        /// <summary>Reason the sequence finished, set when <see cref="Status"/>
        /// becomes one of the Finished* values.</summary>
        public string FinishReason { get; internal set; }

        /// <summary>Optional exception when <see cref="Status"/> is FinishedError.</summary>
        public Exception Error { get; internal set; }

        /// <summary>Per-step telemetry: when scheduling started for this seq.</summary>
        public DateTime? FirstScheduledAt { get; internal set; }

        /// <summary>When the very first token of generation was produced.</summary>
        public DateTime? FirstTokenAt { get; internal set; }

        /// <summary>How many tokens were restored from prefix cache (block hash
        /// hits) at admission time. Diagnostic only.</summary>
        public int PrefixCacheReusedTokens { get; internal set; }

        /// <summary>Cumulative NextN/MTP speculative-decoding counters for this
        /// request, attached by the executor when speculation arms. Null when
        /// the sequence never ran speculatively. Diagnostic only; the engine
        /// logs it when the request finishes.</summary>
        public MtpSpecStats SpecStats { get; internal set; }

        /// <summary>True when this sequence reuses the model's LIVE KV cache
        /// directly (its prompt extends exactly the tokens still resident in the
        /// model's cache from the previous request on the same session), instead of
        /// adopting K/V blocks from the prefix-cache pool. The executor keeps the
        /// live cache as-is rather than reset+inject, which is the only correct way
        /// to reuse a prefix longer than a sliding-window model's window (the pooled
        /// snapshot can't faithfully reconstruct the wrapped circular cache).</summary>
        public bool UsesLiveCacheContinuation { get; internal set; }

        /// <summary>True once this sequence's K/V state has been written into
        /// the model's paged storage (via <c>ForwardBatch</c> or an explicit
        /// linear→paged migration). Prevents the executor from later routing
        /// the same sequence back through the N=1 fast path's
        /// <c>Forward()</c>, which would write into the disjoint linear KV
        /// cache and corrupt the sequence's attention.</summary>
        public bool KvStateInPagedStorage { get; internal set; }

        /// <summary>Returns the token at logical position <paramref name="pos"/>
        /// (prompt or generated).</summary>
        public int TokenAt(int pos)
        {
            if (pos < PromptTokens.Count) return PromptTokens[pos];
            int outIdx = pos - PromptTokens.Count;
            if (outIdx < OutputTokens.Count) return OutputTokens[outIdx];
            throw new ArgumentOutOfRangeException(nameof(pos));
        }

        public void AdvanceComputedTokens(int n)
        {
            NumComputedTokens += n;
            BlockTable.AdvanceTokens(n);
        }

        /// <summary>Set computed-token counter directly without touching the
        /// block table. Used by the scheduler when admitting a sequence that
        /// reused prefix-cache blocks - the blocks are already populated, so
        /// we mark them as committed without going through AdvanceTokens (which
        /// would double-count against the block table).</summary>
        internal void SetComputedTokensForPrefixAdoption(int n)
        {
            NumComputedTokens = n;
            BlockTable.AdvanceTokens(n);
        }

        /// <summary>Reset computed-token counter to 0. Called when the
        /// sequence is preempted - its blocks were freed, the next admission
        /// re-prefills from scratch (with the help of the prefix cache).</summary>
        internal void ResetForPreemption()
        {
            NumComputedTokens = 0;
            LastLogits = null;
            // A preempted sequence re-prefills on re-admission; any prior live-cache
            // continuation claim is stale (its blocks were freed and the model's live
            // cache has since moved on), so drop it and let admission re-decide.
            UsesLiveCacheContinuation = false;
        }

        /// <summary>Abandon a planned live-cache continuation (see
        /// <see cref="UsesLiveCacheContinuation"/>) when the model's live cache is
        /// no longer usable at execution time. Drops the reused-prefix claim so the
        /// sequence re-prefills from position 0, but KEEPS the already-allocated
        /// blocks (their token counter is reset) so block-table accounting stays
        /// consistent through the re-prefill.</summary>
        internal void ClearLiveCacheContinuation()
        {
            UsesLiveCacheContinuation = false;
            NumComputedTokens = 0;
            PrefixCacheReusedTokens = 0;
            LastLogits = null;
            BlockTable.ResetTokensKeepingBlocks();
        }

        public void AppendOutputToken(int token)
        {
            OutputTokens.Add(token);
        }

        /// <summary>
        /// Shrink the committed-token accounting back to the current total
        /// token count. Called after a speculative step's output tail was
        /// truncated at a mid-batch stop (EOS / max-tokens landing before the
        /// step's last accepted token): the step advanced
        /// <see cref="NumComputedTokens"/> over the WHOLE forwarded batch, but
        /// the discarded tail tokens no longer exist in
        /// <see cref="OutputTokens"/>. Leaving the committed count ahead of
        /// <see cref="NumTotalTokens"/> makes prefix-cache block registration
        /// hash positions past the end of the token list (<see cref="TokenAt"/>
        /// throws <see cref="ArgumentOutOfRangeException"/>). The sequence is
        /// finishing, so the dropped tail's K/V (the model's live cache, or the
        /// about-to-be-freed paged blocks) is irrelevant; only the accounting
        /// must agree. The block table's token counter is left as-is because
        /// the finishing path frees its blocks wholesale via
        /// <see cref="Paged.BlockTable.Clear"/> and never reads the counter in
        /// between.
        /// </summary>
        internal void TrimComputedToTotalTokens()
        {
            int total = NumTotalTokens;
            if (NumComputedTokens > total)
                NumComputedTokens = total;
        }

        public bool ShouldStopForLength() => OutputTokens.Count >= MaxNewTokens;

        public override string ToString()
            => $"Seq({RequestId}, sn={Sn}, status={Status}, prompt={PromptTokens.Count}, out={OutputTokens.Count}, computed={NumComputedTokens})";
    }
}
