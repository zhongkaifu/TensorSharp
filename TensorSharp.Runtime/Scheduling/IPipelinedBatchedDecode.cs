// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// A model that keeps two groups' batched decode steps in flight across the GPUs of a layer split. A decode step's
    /// logits are first read when its sequences sample their next token, one step later, so the executor issues a group's
    /// step and moves on: while the last GPU runs one group's layers the first runs the other group's, and both stay busy
    /// instead of one waiting for the other's half of every step. A group's logits are filled when the group samples
    /// again (<see cref="FinishPipelinedDecode"/>) or when anything else needs the model (<see cref="DrainPipelinedDecode"/>).
    /// </summary>
    public interface IPipelinedBatchedDecode
    {
        /// <summary>Groups the model keeps in flight at once (2), or 0 when it cannot pipeline (one GPU, a draft head).</summary>
        int PipelinedDecodeGroups { get; }

        /// <summary>Whether a group's step is in flight, its rows' logits not filled yet.</summary>
        bool HasPendingPipelinedDecode { get; }

        /// <summary>
        /// Issue group <paramref name="group"/>'s batched decode step (after finishing the group's previous one). True with
        /// each row's logits array in <paramref name="outLogits"/>, filled when the step finishes; false when the model
        /// declines the step, which then changed nothing.
        /// </summary>
        bool TryIssuePipelinedDecode(int group, IReadOnlyList<string> requestIds, int[] tokens, int[] positions,
            float[][] outLogits);

        /// <summary>Wait for group <paramref name="group"/>'s step in flight, if any, and fill its rows' logits.</summary>
        void FinishPipelinedDecode(int group);

        /// <summary>Finish every group's step in flight.</summary>
        void DrainPipelinedDecode();
    }
}
