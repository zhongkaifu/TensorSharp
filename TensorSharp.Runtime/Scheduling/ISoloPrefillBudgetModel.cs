// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// A model that knows better than the scheduler's budget how large a lone request's prefill
    /// steps should be: Muse Spark's pipelined layer split fills and drains a pipeline of
    /// micro-batches across its GPUs on every forward call, so a step of a few thousand tokens
    /// leaves most of the GPUs idle. When no other request is in the system and the user did not
    /// choose <see cref="SchedulerConfig.SoloPrefillChunkSize"/>, the scheduler asks the model
    /// before each step; requests that share the engine still get the ordinary budget.
    /// </summary>
    public interface ISoloPrefillBudgetModel
    {
        /// <summary>Tokens the next step of a lone prompt should prefill, starting at prompt
        /// position <paramref name="start"/> with <paramref name="remaining"/> prompt tokens left
        /// (it may exceed <see cref="SchedulerConfig.MaxNumBatchedTokens"/>); 0 for the
        /// scheduler's own step.</summary>
        int PreferredSoloPrefillTokens(int start, int remaining);

        /// <summary>Prompt tokens too few to be worth a prefill step of their own: a public
        /// shared-prefix checkpoint boundary at most this far past the prefix a request reused from
        /// the prefix cache does not end a prefill step. A later request that reuses the same prefix
        /// prefills those tokens inside its own first step anyway, while the cut would cost this
        /// request one more forward pass, which for a model whose pass has a large fixed cost is most
        /// of a short turn's latency. 0 (the default) cuts at every boundary.</summary>
        int CheckpointGapTokens => 0;

        /// <summary>The scheduler's own lone-prompt step (<see cref="SchedulerConfig.SoloPrefillChunkSize"/> bounded by
        /// <see cref="SchedulerConfig.MaxNumBatchedTokens"/>), told once when the policy attaches: a step the model chose
        /// that does not run as it planned (the shared decoder runtime's pipelined prompts, where a request turns out to have
        /// its prompt followed by a drafter) runs in the steps it would have had.</summary>
        void UseSchedulerSoloStep(int tokens) { }
    }
}
