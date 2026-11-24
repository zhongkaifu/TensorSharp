// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Why a request decoded without speculation although speculation was on: short, stable codes
    /// for <see cref="Scheduling.SequenceState.SpeculationDeclined"/>, the declined
    /// <c>inference.speculation</c> span and the <c>tensorsharp.speculative.requests</c> counter.
    /// The operator-facing sentence behind each is logged where it is decided.
    /// </summary>
    public static class SpeculationDeclineReason
    {
        /// <summary>A LoRA adapter was active for the step; drafters and verify trunks do not apply it.</summary>
        public const string Adapter = "adapter";

        /// <summary>The architecture has no speculative trunk: it cannot verify a draft window in one pass.</summary>
        public const string NoTrunk = "no_trunk";

        /// <summary>The algorithm needs learned draft weights and the loaded model has none.</summary>
        public const string NoDraftHead = "no_draft_head";

        /// <summary>The model refuses speculation for correctness (<see cref="ISpeculativeTarget.SpeculationRefusal"/>).</summary>
        public const string RefusedByModel = "refused_by_model";

        /// <summary>Speculation is slower than plain decoding on this backend.</summary>
        public const string Unprofitable = "unprofitable";

        /// <summary>The step decoded several sequences together and the linear trunk speculates one at a time.</summary>
        public const string MultiSequence = "multi_sequence";

        /// <summary>The sequence's K/V lives in paged storage, where the linear trunk has none of it.</summary>
        public const string PagedStorage = "paged_storage";

        /// <summary>The prompt's media embeddings need the plain forward's inject hook.</summary>
        public const string PendingMultimodal = "pending_multimodal";

        /// <summary>The sequence lives in a per-request fused cache.</summary>
        public const string FusedCache = "fused_cache";

        /// <summary>The request asked for fewer than two tokens, so a draft could never pay off.</summary>
        public const string TooFewTokens = "too_few_tokens";

        /// <summary><see cref="SpeculatorRegistry.Create"/> declined: no draft capacity, an unknown
        /// algorithm, or one that cannot serve this model.</summary>
        public const string Unavailable = "unavailable";

        /// <summary>The request reused a KV prefix and its algorithm cannot resume after a gap.</summary>
        public const string PrefixReuse = "prefix_reuse";

        /// <summary>The drafter could not be seeded from the tokens the trunk already holds.</summary>
        public const string CannotSeed = "cannot_seed";

        /// <summary>No step recorded a reason.</summary>
        public const string Unknown = "unknown";
    }
}
