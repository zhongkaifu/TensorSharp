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
    /// One per-sequence decision from the scheduler for a single step.
    /// Mirrors vLLM's per-row entries in <c>SchedulerOutput.num_scheduled_tokens</c>
    /// + <c>scheduled_new_reqs</c>.
    /// </summary>
    public sealed class ScheduledSequenceWork
    {
        public ScheduledSequenceWork(
            SequenceState seq,
            int numScheduledTokens,
            bool isNewAdmission,
            bool isPrefill)
        {
            Sequence = seq;
            NumScheduledTokens = numScheduledTokens;
            IsNewAdmission = isNewAdmission;
            IsPrefill = isPrefill;
        }

        public SequenceState Sequence { get; }

        /// <summary>How many tokens of this sequence to forward this step. For
        /// a decode step this is 1. For prefill it can be up to the unprocessed
        /// prompt length (or smaller if chunked).</summary>
        public int NumScheduledTokens { get; }

        /// <summary>True iff this is the first step the sequence appears in
        /// (we just transitioned from Waiting/Preempted to Running).</summary>
        public bool IsNewAdmission { get; }

        /// <summary>True when <see cref="NumScheduledTokens"/> &gt; 1; informational
        /// flag passed to the executor.</summary>
        public bool IsPrefill { get; }

        /// <summary>Position offset where this step's tokens land in the
        /// sequence's logical timeline. Equal to
        /// <see cref="SequenceState.NumComputedTokens"/> at scheduling time.</summary>
        public int StartPosition => Sequence.NumComputedTokens;

        public override string ToString()
            => $"Work({Sequence.RequestId} +{NumScheduledTokens} {(IsPrefill ? "prefill" : "decode")}{(IsNewAdmission ? " new" : "")})";
    }

    /// <summary>
    /// What the scheduler picked for one step. Consumed by the executor.
    /// Mirrors vLLM's <c>SchedulerOutput</c>.
    /// </summary>
    public sealed class SchedulerOutput
    {
        public SchedulerOutput()
        {
            ScheduledWork = new List<ScheduledSequenceWork>();
            PreemptedRequestIds = new List<string>();
            FinishedRequestIds = new List<string>();
        }

        public List<ScheduledSequenceWork> ScheduledWork { get; }
        public List<string> PreemptedRequestIds { get; }
        public List<string> FinishedRequestIds { get; }

        /// <summary>Sum of <see cref="ScheduledSequenceWork.NumScheduledTokens"/>
        /// across all scheduled work. Used for token-budget telemetry.</summary>
        public int TotalScheduledTokens
        {
            get
            {
                int s = 0;
                for (int i = 0; i < ScheduledWork.Count; i++)
                    s += ScheduledWork[i].NumScheduledTokens;
                return s;
            }
        }

        public bool IsEmpty => ScheduledWork.Count == 0 && FinishedRequestIds.Count == 0;
    }
}
