// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Lifecycle of a <see cref="SequenceState"/>. Mirrors vLLM's RequestStatus.
    /// </summary>
    public enum SequenceStatus
    {
        /// <summary>Submitted but no KV blocks allocated yet. Lives in
        /// <c>ContinuousBatchScheduler._waiting</c>.</summary>
        Waiting,

        /// <summary>Currently in the scheduler's <c>_running</c> set: has KV
        /// blocks, will be picked up next step.</summary>
        Running,

        /// <summary>Was running, lost its KV blocks to a higher-priority sequence.
        /// Lives in <c>_waiting</c> with a non-zero <c>NumComputedTokens</c> so
        /// that the next scheduling pass restores from prefix cache.</summary>
        Preempted,

        /// <summary>EOS sampled or stop sequence matched. KV blocks freed.</summary>
        FinishedStopped,

        /// <summary>Generation reached <c>MaxTokens</c>.</summary>
        FinishedLengthCapped,

        /// <summary>Client cancelled before completion.</summary>
        FinishedAborted,

        /// <summary>An internal error terminated this sequence.</summary>
        FinishedError,
    }

    public static class SequenceStatusExtensions
    {
        public static bool IsFinished(this SequenceStatus s)
            => s == SequenceStatus.FinishedStopped
            || s == SequenceStatus.FinishedLengthCapped
            || s == SequenceStatus.FinishedAborted
            || s == SequenceStatus.FinishedError;
    }
}
