// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;
using System.Text;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>The execution paths the engine can serve one scheduler step
    /// with. One enum member per genuinely distinct dispatch, so "which path
    /// combinations exist" is an enumerable question rather than emergent
    /// control flow.</summary>
    public enum ExecutionPathKind
    {
        /// <summary>NextN/MTP speculative decoding with the trunk on the
        /// batched paged path (same kernels as the non-speculative batched
        /// baseline). Declinable: the executor's arming/continuity gate may
        /// pass the step to the next candidate.</summary>
        SpeculativeBatchedTrunk,

        /// <summary>Per-sequence route chosen so LINEAR-trunk MTP speculation
        /// can engage (or keep serving a sequence whose state lives in the
        /// linear cache). Runs plain per-sequence decode when the speculative
        /// context can't arm.</summary>
        SpeculativePerSequence,

        /// <summary>Concurrent sequences served by per-request fused Forward
        /// (own KV holder per request); may internally use true token-batched
        /// fused decode when enabled.</summary>
        PerSequenceFused,

        /// <summary>Mixed step: multimodal sequences on the per-sequence path,
        /// text sequences on the batched paged path.</summary>
        MixedMultimodalSplit,

        /// <summary>N=1 fast path: a solo sequence served through the model's
        /// fused single-graph Forward against the linear KV cache.</summary>
        SingleSequenceFused,

        /// <summary>vLLM-style batched paged attention (ForwardBatch).
        /// Declinable: linear→paged migration failure or a model
        /// NotSupportedException falls through to the next candidate.</summary>
        BatchedPaged,

        /// <summary>Per-sequence forward with KV-state ownership swap — the
        /// universal fallback; never declines.</summary>
        PerSequence,
    }

    /// <summary>
    /// Per-step request features — what THIS step's scheduled work needs from
    /// the execution path (the request-side counterpart of
    /// <see cref="ExecutionCapabilities"/>). Computed by the executor from the
    /// scheduler output plus its own dynamic state, then handed to
    /// <see cref="ExecutionPlanner"/>.
    /// </summary>
    public sealed record ExecutionStepFeatures
    {
        /// <summary>Number of scheduled work items this step.</summary>
        public int SequenceCount { get; init; }

        /// <summary>How many scheduled sequences still have pending multimodal
        /// embeddings (require the per-sequence embedding-inject hook unless
        /// the model handles multimodal in its batched path).</summary>
        public int MultimodalPendingCount { get; init; }

        /// <summary>Solo step only: the scheduled sequence has pending
        /// multimodal embeddings.</summary>
        public bool SoloHasPendingMultimodal { get; init; }

        /// <summary>Solo step only: the sequence's K/V history lives in paged
        /// storage (it must not be served from the linear cache).</summary>
        public bool SoloKvInPagedStorage { get; init; }

        /// <summary>Solo step only: the sequence already owns a per-request
        /// fused cache and must stay on the fused path (its tail K/V is not
        /// reconstructable from paged storage).</summary>
        public bool SoloHasFusedCache { get; init; }

        /// <summary>Solo step only: a different sequence currently owns the
        /// model's linear KV cache, so serving this one per-sequence requires
        /// an ownership swap (needs KV snapshot support).</summary>
        public bool SoloRequiresOwnershipSwap { get; init; }
    }

    /// <summary>A path that was considered but not selected, with the reason —
    /// makes "why did the engine not take the fast path?" a logged fact
    /// instead of archaeology.</summary>
    public sealed record ExecutionPathRejection(ExecutionPathKind Path, string Reason)
    {
        public override string ToString() => $"{Path}: {Reason}";
    }

    /// <summary>
    /// The planner's decision for one scheduler step: an ordered candidate
    /// chain (first entry is the selected path; later entries serve the step
    /// if an earlier declinable candidate passes) plus the reasons every
    /// rejected path was rejected. The final candidate is always
    /// <see cref="ExecutionPathKind.PerSequence"/>-safe, i.e. never declines.
    /// </summary>
    public sealed record ExecutionPlan
    {
        /// <summary>Paths to try, in priority order. Never empty.</summary>
        public IReadOnlyList<ExecutionPathKind> Candidates { get; init; }

        /// <summary>Why paths that could plausibly have served this step were
        /// not selected.</summary>
        public IReadOnlyList<ExecutionPathRejection> Rejections { get; init; }

        /// <summary>Speculative decoding was requested but is unprofitable on
        /// this backend; the executor surfaces a one-time operator notice.</summary>
        public bool SpeculationUnprofitable { get; init; }

        /// <summary>Speculative decoding was requested but the model refuses it for
        /// correctness (see <see cref="ISpeculativeTarget.SpeculationRefusal"/>);
        /// the executor surfaces this reason once.</summary>
        public string SpeculationRefusal { get; init; }

        /// <summary>The path this plan selects (first candidate).</summary>
        public ExecutionPathKind Selected => Candidates[0];

        /// <summary>Compact single-line description for logging:
        /// selected path, fallback chain, and rejection reasons.</summary>
        public string Describe()
        {
            var sb = new StringBuilder();
            sb.Append(Selected);
            if (Candidates.Count > 1)
            {
                sb.Append(" (fallbacks: ");
                for (int i = 1; i < Candidates.Count; i++)
                {
                    if (i > 1) sb.Append(" -> ");
                    sb.Append(Candidates[i]);
                }
                sb.Append(')');
            }
            if (Rejections != null && Rejections.Count > 0)
            {
                sb.Append("; rejected: ");
                for (int i = 0; i < Rejections.Count; i++)
                {
                    if (i > 0) sb.Append("; ");
                    sb.Append(Rejections[i]);
                }
            }
            return sb.ToString();
        }
    }
}
