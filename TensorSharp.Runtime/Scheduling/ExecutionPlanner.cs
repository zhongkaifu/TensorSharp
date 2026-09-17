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
    /// <summary>
    /// The single decision point for which execution path serves a scheduler
    /// step:
    ///
    ///     model+backend capabilities (<see cref="ExecutionCapabilities"/>)
    ///   + operator overrides        (<see cref="ExecutionOptions"/>)
    ///   + engine config             (<see cref="SchedulerConfig"/>)
    ///   + step request features     (<see cref="ExecutionStepFeatures"/>)
    ///   → <see cref="ExecutionPlan"/> (selected path, fallback chain,
    ///     rejection reasons)
    ///
    /// Pure function of its inputs — no model calls, no environment reads, no
    /// executor state — so every path combination is unit-testable without a
    /// model, and the plan (including why fast paths were rejected) can be
    /// logged verbatim. The executor keeps only the DYNAMIC gates that depend
    /// on its private state (MTP arming/continuity, migration success, a model
    /// declining a specific batch): those paths are "declinable" candidates
    /// and fall through to the next entry in the plan's chain, so
    /// exception-driven fallback is confined to the two model-declined cases
    /// instead of being the routing mechanism.
    /// </summary>
    public static class ExecutionPlanner
    {
        public static ExecutionPlan PlanStep(
            ExecutionCapabilities caps,
            ExecutionOptions options,
            SchedulerConfig config,
            ExecutionStepFeatures features)
        {
            var candidates = new List<ExecutionPathKind>(3);
            var rejections = new List<ExecutionPathRejection>();
            bool mtpUnprofitable = false;
            string speculationRefusal = null;

            bool batchedEnabled = !options.BatchedPathDisabled;
            bool batchedImpl = caps.SupportsBatchedPagedAttention;
            bool solo = features.SequenceCount == 1;
            // A solo sequence resident in a per-request fused cache must stay
            // on the fused path (tail K/V not reconstructable elsewhere).
            bool fusedResident = caps.SupportsPerSequenceFusedForward && features.SoloHasFusedCache;

            // Multimodal split: models that handle multimodal in their batched
            // path keep everything together; others get multimodal sequences
            // peeled onto the per-sequence path.
            bool batchedModalSafe = batchedImpl && caps.SupportsBatchedMultimodal;
            int multimodalCount = batchedModalSafe ? 0 : features.MultimodalPendingCount;
            int textCount = features.SequenceCount - multimodalCount;

            // ---- Speculative decoding ----
            // Deliberately NOT gated on options.BatchedPathDisabled: the
            // speculative routes predate that switch and must keep engaging
            // under it.
            if (config.Speculation.Enabled)
            {
                if (!caps.SupportsSpeculativeTrunk)
                {
                    rejections.Add(new ExecutionPathRejection(
                        ExecutionPathKind.SpeculativePerSequence,
                        "requested (--spec) but this architecture has no speculative trunk "
                        + "(it cannot verify a draft window in one pass)"));
                }
                else if (!caps.HasDraftHead
                         && SpeculatorRegistry.RequiresDraftHead(config.Speculation.SpeculatorName))
                {
                    rejections.Add(new ExecutionPathRejection(
                        ExecutionPathKind.SpeculativePerSequence,
                        $"requested (--spec) but the loaded weights have no draft head; "
                        + $"--spec-type {SpeculatorRegistry.NGram} needs none"));
                }
                else if (caps.SpeculationRefusal != null)
                {
                    speculationRefusal = caps.SpeculationRefusal;
                    rejections.Add(new ExecutionPathRejection(
                        ExecutionPathKind.SpeculativePerSequence,
                        "refused by the model: " + caps.SpeculationRefusal));
                }
                else if (!caps.SpeculationProfitable)
                {
                    // Serving standard decode is faster than per-op speculative
                    // verify/draft on this backend; executor logs a one-time notice.
                    mtpUnprofitable = true;
                    rejections.Add(new ExecutionPathRejection(
                        ExecutionPathKind.SpeculativePerSequence,
                        "speculation unprofitable on this backend; serving standard decode"));
                }
                else if (caps.SupportsBatchedSpecTrunk)
                {
                    // Batched-trunk models never take the per-sequence detour:
                    // if the trunk declines, the normal batched path serves the
                    // step (keeps K/V in paged storage).
                    if (!solo)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SpeculativeBatchedTrunk, "multi-sequence step"));
                    else if (features.SoloHasPendingMultimodal)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SpeculativeBatchedTrunk, "pending multimodal embeddings need the per-seq inject hook"));
                    else if (fusedResident)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SpeculativeBatchedTrunk, "sequence lives in a per-request fused cache"));
                    else
                        candidates.Add(ExecutionPathKind.SpeculativeBatchedTrunk); // declinable (arming/continuity)
                }
                else
                {
                    // Linear-trunk speculation requires the per-sequence route.
                    if (!solo)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SpeculativePerSequence, "multi-sequence step"));
                    else if (features.SoloKvInPagedStorage)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SpeculativePerSequence, "sequence K/V lives in paged storage; linear cache would be empty"));
                    else if (features.SoloHasPendingMultimodal)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SpeculativePerSequence, "pending multimodal embeddings need Forward's inject hook"));
                    else if (fusedResident)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SpeculativePerSequence, "sequence lives in a per-request fused cache (the fused path speculates on it itself when the trunk follows the bound holder)"));
                    else
                    {
                        candidates.Add(ExecutionPathKind.SpeculativePerSequence); // terminal
                        return Build(candidates, rejections, mtpUnprofitable, speculationRefusal);
                    }
                }
            }

            // ---- Per-sequence fused decode (N>=2, a fused-resident solo, or
            // a solo request that must take ownership from another live
            // sequence). The last case moves the prior primary cache into a
            // per-request holder instead of serializing a potentially enormous
            // hybrid recurrent state through managed paged slabs.
            if (batchedEnabled && batchedImpl && caps.SupportsPerSequenceFusedForward)
            {
                bool wanted = features.SequenceCount >= 2
                    || fusedResident
                    || features.SoloRequiresOwnershipSwap;
                if (!wanted)
                {
                    // Solo never-fused sequences intentionally fall through to
                    // the N=1 fast path / batched path (keeps prefix-cache
                    // reuse and live-cache continuation) — not a rejection
                    // worth logging.
                }
                else if (!options.PerSeqFusedEnabled)
                {
                    rejections.Add(new ExecutionPathRejection(
                        ExecutionPathKind.PerSequenceFused, "disabled via TS_PER_SEQ_FUSED=0"));
                }
                else
                {
                    candidates.Add(ExecutionPathKind.PerSequenceFused); // terminal
                    return Build(candidates, rejections, mtpUnprofitable, speculationRefusal);
                }
            }

            // ---- Mixed multimodal/text step: split across both paths ----
            if (batchedEnabled && batchedImpl && multimodalCount > 0 && textCount > 0)
            {
                candidates.Add(ExecutionPathKind.MixedMultimodalSplit); // terminal
                return Build(candidates, rejections, mtpUnprofitable, speculationRefusal);
            }

            // ---- Batched paged path (with the N=1 fused fast path in front) ----
            if (batchedEnabled && batchedImpl && multimodalCount == 0 && features.SequenceCount > 0)
            {
                if (solo)
                {
                    if (!options.BatchedN1FastPathEnabled)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SingleSequenceFused, "disabled via TS_BATCHED_N1_FAST_PATH=0"));
                    else if (!caps.SupportsLinearKvMigration)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SingleSequenceFused,
                            "model cannot migrate linear KV to paged storage (a second request would corrupt attention)"));
                    else if (features.SoloKvInPagedStorage)
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SingleSequenceFused, "sequence K/V already committed to paged storage"));
                    else if (features.SoloRequiresOwnershipSwap
                        && (!caps.SupportsKvStateSnapshot || !caps.SupportsCrossSequenceKvReuse))
                        rejections.Add(new ExecutionPathRejection(
                            ExecutionPathKind.SingleSequenceFused,
                            "ownership swap required but model lacks a reusable KV snapshot"));
                    else
                    {
                        candidates.Add(ExecutionPathKind.SingleSequenceFused); // terminal
                        return Build(candidates, rejections, mtpUnprofitable, speculationRefusal);
                    }
                }

                if (caps.BatchedForwardAvailable)
                {
                    // Declinable: linear→paged migration may fail, or the model
                    // may refuse this specific batch (NotSupportedException).
                    candidates.Add(ExecutionPathKind.BatchedPaged);
                }
                else
                {
                    rejections.Add(new ExecutionPathRejection(
                        ExecutionPathKind.BatchedPaged,
                        "model declares its batched path unavailable (per-model opt-out or unsupported layer/KV format)"));
                }
            }
            else if (batchedImpl && !batchedEnabled)
            {
                rejections.Add(new ExecutionPathRejection(
                    ExecutionPathKind.BatchedPaged, "disabled via TS_SCHED_DISABLE_BATCHED"));
            }
            else if (batchedImpl && multimodalCount > 0)
            {
                rejections.Add(new ExecutionPathRejection(
                    ExecutionPathKind.BatchedPaged,
                    "all scheduled sequences have pending multimodal embeddings and the model lacks batched multimodal support"));
            }

            // ---- Universal fallback ----
            candidates.Add(ExecutionPathKind.PerSequence);
            return Build(candidates, rejections, mtpUnprofitable, speculationRefusal);
        }

        /// <summary>Startup capability report: which paths are statically
        /// available for the loaded model under the current configuration, and
        /// why the unavailable ones are unavailable. Logged once by
        /// <see cref="InferenceEngine"/> so operators see the path landscape
        /// without decoding per-step logs or env-var archaeology.</summary>
        public static string BuildCapabilityReport(
            ExecutionCapabilities caps,
            ExecutionOptions options,
            SchedulerConfig config)
        {
            var sb = new StringBuilder();

            sb.Append("batched paged attention: ");
            if (!caps.SupportsBatchedPagedAttention)
                sb.Append("unavailable (model does not implement IBatchedPagedModel)");
            else if (options.BatchedPathDisabled)
                sb.Append("disabled (TS_SCHED_DISABLE_BATCHED)");
            else if (!caps.BatchedForwardAvailable)
                sb.Append("unavailable (model opt-out or unsupported layer/KV format)");
            else
                sb.Append("available").Append(caps.SupportsBatchedMultimodal
                    ? " (including multimodal)"
                    : " (multimodal peels off to per-sequence)");

            sb.Append("\nper-sequence fused concurrent decode: ");
            if (!caps.SupportsPerSequenceFusedForward)
                sb.Append("unavailable (model/backend does not opt in)");
            else if (!options.PerSeqFusedEnabled)
                sb.Append("disabled (TS_PER_SEQ_FUSED=0)");
            else
                sb.Append(options.BatchedFusedDecodeEnabled
                    ? "available (+ token-batched fused decode)"
                    : "available");

            sb.Append("\nN=1 single-sequence fused fast path: ");
            if (!caps.SupportsBatchedPagedAttention)
                sb.Append("n/a (no batched contract)");
            else if (!options.BatchedN1FastPathEnabled)
                sb.Append("disabled (TS_BATCHED_N1_FAST_PATH=0)");
            else if (!caps.SupportsLinearKvMigration)
                sb.Append("unavailable (no linear->paged KV migration)");
            else
                sb.Append("available");

            sb.Append("\nSpeculative decoding: ");
            if (!config.Speculation.Enabled)
                sb.Append("off (not requested)");
            else if (!caps.SupportsSpeculativeTrunk)
                sb.Append("requested but unavailable (architecture has no speculative trunk)");
            else if (!caps.HasDraftHead
                     && SpeculatorRegistry.RequiresDraftHead(config.Speculation.SpeculatorName))
            {
                sb.Append("requested but unavailable (no draft head in weights; --spec-type ")
                  .Append(SpeculatorRegistry.NGram).Append(" needs none)");
            }
            else if (caps.SpeculationRefusal != null)
                sb.Append("requested but refused by the model (serving standard decode): ").Append(caps.SpeculationRefusal);
            else if (!caps.SpeculationProfitable)
                sb.Append("requested but unprofitable on this backend (serving standard decode)");
            else
            {
                sb.Append("available (algorithm=").Append(config.Speculation.SpeculatorName)
                  .Append(caps.SupportsBatchedSpecTrunk ? ", trunk=batched)" : ", trunk=linear)");
            }

            sb.Append("\nKV snapshot/swap fallback: ");
            if (!caps.SupportsKvStateSnapshot)
                sb.Append("unavailable (no block snapshot; per-seq swaps force re-prefill)");
            else if (!caps.SupportsCrossSequenceKvReuse)
                sb.Append("snapshot only (no cross-sequence reuse)");
            else if (caps.MaxReusablePrefixTokens != int.MaxValue)
                sb.Append("available (prefix reuse capped at ").Append(caps.MaxReusablePrefixTokens).Append(" tokens)");
            else
                sb.Append("available (unbounded prefix reuse)");

            sb.Append("\nretained fused-cache continuation: ");
            if (!caps.SupportsPerSequenceFusedForward)
                sb.Append("n/a for this model");
            else if (!caps.SupportsRetainedFusedCache)
                sb.Append("unavailable (model does not support retained holders)");
            else if (!options.RetainedFusedCacheEnabled)
                sb.Append("disabled (TS_RETAINED_FUSED_CACHE=0)");
            else
                sb.Append("on (budget=").Append(options.RetainedFusedCacheBudget).Append(')');

            string overrides = options.DescribeOverrides();
            if (!string.IsNullOrEmpty(overrides))
                sb.Append("\nactive TS_* overrides: ").Append(overrides);

            return sb.ToString();
        }

        private static ExecutionPlan Build(
            List<ExecutionPathKind> candidates,
            List<ExecutionPathRejection> rejections,
            bool mtpUnprofitable,
            string speculationRefusal = null)
        {
            // A plan whose last candidate can decline would leave the step
            // unserved; PerSequence never declines, SpeculativePerSequence /
            // PerSequenceFused / SingleSequenceFused / MixedMultimodalSplit
            // are terminal by construction.
            if (candidates.Count == 0 || candidates[^1] == ExecutionPathKind.SpeculativeBatchedTrunk
                || candidates[^1] == ExecutionPathKind.BatchedPaged)
            {
                candidates.Add(ExecutionPathKind.PerSequence);
            }
            return new ExecutionPlan
            {
                Candidates = candidates,
                Rejections = rejections,
                SpeculationUnprofitable = mtpUnprofitable,
                SpeculationRefusal = speculationRefusal,
            };
        }
    }
}
