// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Text;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Structured capability snapshot of the loaded (model × backend)
    /// combination — the single surface <see cref="ExecutionPlanner"/> consults
    /// to choose an execution path, instead of executor code probing interface
    /// casts, <c>Supports*</c> getters and backend names at every decision
    /// point.
    ///
    /// In TensorSharp a model instance embodies its backend choice (e.g.
    /// Qwen 3.5 reports <c>SupportsPerSequenceFusedForward</c> only on
    /// GGML-CUDA/Metal, Gemma 4 reports <c>SpeculationProfitable</c> only
    /// where its fused verify kernels exist), so this record is the merged
    /// "backend capabilities + model requirements" view: each field is the
    /// answer to "can THIS model on THIS backend run that path?". Model-level
    /// environment opt-outs (e.g. <c>TS_QWEN35_BATCHED=0</c>) surface through
    /// <see cref="BatchedForwardAvailable"/> rather than through
    /// exception-driven fallback, so routing decisions are declared up front.
    ///
    /// Built per step via <see cref="FromModel"/> — the getters are cheap flag
    /// reads, and several are env-var backed so they must not be cached for
    /// the engine's lifetime.
    /// </summary>
    public sealed record ExecutionCapabilities
    {
        /// <summary>Model implements <see cref="IBatchedPagedModel"/>: the
        /// vLLM-style batched paged-attention contract (ForwardBatch over
        /// slot-mapped paged KV storage).</summary>
        public bool SupportsBatchedPagedAttention { get; init; }

        /// <summary>Model-declared master switch for its batched path
        /// (<see cref="IBatchedPagedModel.BatchedForwardAvailable"/>): false when
        /// a per-model opt-out (e.g. <c>TS_QWEN35_BATCHED=0</c>) or a static
        /// limitation (e.g. Gemma 4 with MoE layers or block-quantized KV)
        /// makes ForwardBatch unusable, so the planner routes around it
        /// instead of relying on a NotSupportedException fallback.</summary>
        public bool BatchedForwardAvailable { get; init; }

        /// <summary>ForwardBatch handles multimodal sequences (vision/audio
        /// embeddings + MRoPE) itself; when false the executor peels
        /// multimodal sequences onto a per-sequence path.</summary>
        public bool SupportsBatchedMultimodal { get; init; }

        /// <summary>Model serves concurrent sequences through per-request
        /// fused Forward (own KV holder per request) — the high-throughput
        /// N&gt;=2 path on models whose fused single-graph decode beats the
        /// op-by-op batched kernels.</summary>
        public bool SupportsPerSequenceFusedForward { get; init; }

        /// <summary>The model can retain and later re-key a completed per-request
        /// fused holder for an exact-prefix continuation.</summary>
        public bool SupportsRetainedFusedCache { get; init; }

        /// <summary>Model can migrate a sequence's K/V history from the linear
        /// cache (written by Forward / the N=1 fast path) into paged storage —
        /// prerequisite for the N=1 fast path, so a second concurrent request
        /// can transition the owner onto the batched path safely.</summary>
        public bool SupportsLinearKvMigration { get; init; }

        /// <summary>Block-level KV snapshot/restore (paged KV pool, ownership
        /// swap on the per-sequence path).</summary>
        public bool SupportsKvStateSnapshot { get; init; }

        /// <summary>K/V captured by one sequence can be re-injected into a
        /// different sequence's fresh cache (cross-request prefix reuse and
        /// per-seq ownership rotation).</summary>
        public bool SupportsCrossSequenceKvReuse { get; init; }

        /// <summary>Longest leading prefix whose snapshot restores faithfully
        /// (sliding-window models cap this at the window size).</summary>
        public int MaxReusablePrefixTokens { get; init; } = int.MaxValue;

        /// <summary>Model has a multimodal injector (vision/audio requests are
        /// possible at all).</summary>
        public bool HasMultimodalInjector { get; init; }

        /// <summary>Model implements <see cref="ISpeculativeTarget"/>: it can
        /// verify a whole draft window in one trunk pass and undo a rejected
        /// tail. This is the ONLY capability a weight-free speculator
        /// (<see cref="NGramSpeculator"/>) needs.</summary>
        public bool SupportsSpeculativeTrunk { get; init; }

        /// <summary>Loaded weights contain a usable learned draft head (a
        /// NextN/MTP block, a DSpark/DFlash block drafter). Algorithms that
        /// need trained speculator weights require this; n-gram does not.</summary>
        public bool HasDraftHead { get; init; }

        /// <summary>Speculative decoding is expected to be profitable on this
        /// backend (the accelerated multi-token verify/draft kernels exist);
        /// when false the engine serves standard decode even if speculation
        /// was requested.</summary>
        public bool SpeculationProfitable { get; init; }

        /// <summary>Non-null when the model refuses speculation for correctness
        /// (<see cref="ISpeculativeTarget.SpeculationRefusal"/>); the engine serves
        /// plain decoding and reports this reason.</summary>
        public string SpeculationRefusal { get; init; }

        /// <summary>The speculative trunk can run through the batched paged
        /// path (<see cref="IBatchedSpeculativeTarget"/>), composing with
        /// prefix caching and concurrency transitions.</summary>
        public bool SupportsBatchedSpecTrunk { get; init; }

        /// <summary>Snapshot the capability surface of <paramref name="model"/>.
        /// Cheap (flag/property reads); called once per engine step.</summary>
        public static ExecutionCapabilities FromModel(IModelArchitecture model)
        {
            var batched = model as IBatchedPagedModel;
            var spec = model as ISpeculativeTarget;
            bool specTrunk = spec != null;
            bool draftHead = spec is IDraftHead head && head.HasDraftHead;
            return new ExecutionCapabilities
            {
                SupportsBatchedPagedAttention = batched != null,
                BatchedForwardAvailable = batched != null && batched.BatchedForwardAvailable,
                SupportsBatchedMultimodal = batched != null && batched.SupportsBatchedMultimodal,
                SupportsPerSequenceFusedForward = batched != null && batched.SupportsPerSequenceFusedForward,
                SupportsRetainedFusedCache = batched != null && batched.SupportsRetainedFusedCache,
                SupportsLinearKvMigration = batched != null && batched.SupportsLinearKVMigration,
                SupportsKvStateSnapshot = model.SupportsKVStateSnapshot,
                SupportsCrossSequenceKvReuse = model.SupportsCrossSequenceKvReuse,
                MaxReusablePrefixTokens = model.MaxReusablePrefixTokens,
                HasMultimodalInjector = model.MultimodalInjector != null,
                SupportsSpeculativeTrunk = specTrunk,
                HasDraftHead = draftHead,
                SpeculationProfitable = specTrunk && spec.SpeculationRefusal == null && spec.SpeculationProfitable,
                SpeculationRefusal = specTrunk ? spec.SpeculationRefusal : null,
                SupportsBatchedSpecTrunk = specTrunk
                    && spec is IBatchedSpeculativeTarget batchedSpec
                    && batchedSpec.SupportsBatchedSpecTrunk,
            };
        }

        /// <summary>Human-readable multi-line capability listing (startup log).</summary>
        public string Describe()
        {
            var sb = new StringBuilder();
            sb.Append("batchedPagedAttention=").Append(Flag(SupportsBatchedPagedAttention));
            if (SupportsBatchedPagedAttention && !BatchedForwardAvailable)
                sb.Append(" (declared unavailable by model)");
            sb.Append(", batchedMultimodal=").Append(Flag(SupportsBatchedMultimodal));
            sb.Append(", perSeqFused=").Append(Flag(SupportsPerSequenceFusedForward));
            sb.Append(", retainedFused=").Append(Flag(SupportsRetainedFusedCache));
            sb.Append(", linearKvMigration=").Append(Flag(SupportsLinearKvMigration));
            sb.Append(", kvSnapshot=").Append(Flag(SupportsKvStateSnapshot));
            sb.Append(", crossSeqKvReuse=").Append(Flag(SupportsCrossSequenceKvReuse));
            sb.Append(", maxReusablePrefix=");
            sb.Append(MaxReusablePrefixTokens == int.MaxValue ? "unbounded" : MaxReusablePrefixTokens.ToString());
            sb.Append(", multimodal=").Append(Flag(HasMultimodalInjector));
            sb.Append(", specTrunk=").Append(Flag(SupportsSpeculativeTrunk));
            if (SupportsSpeculativeTrunk)
            {
                sb.Append(", draftHead=").Append(Flag(HasDraftHead));
                sb.Append(", specProfitable=").Append(Flag(SpeculationProfitable));
                sb.Append(", specBatchedTrunk=").Append(Flag(SupportsBatchedSpecTrunk));
            }
            return sb.ToString();
        }

        private static string Flag(bool value) => value ? "yes" : "no";
    }
}
