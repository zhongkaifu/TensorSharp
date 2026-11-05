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
    /// <summary>Cumulative speculative-decoding counters for one execution
    /// (one request on the engine path; one Generate call on the standalone
    /// path). The engine logs these when a request finishes.</summary>
    public sealed class SpeculationStats
    {
        public long TokensDrafted { get; internal set; }
        public long TokensAccepted { get; internal set; }
        public long VerifySteps { get; internal set; }
        public long PlainSteps { get; internal set; }
        public long RollbackSteps { get; internal set; }

        /// <summary>Decode steps forced to plain because the cost governor measured
        /// speculation as a net loss on this model/drafter pair.</summary>
        public long ParkedSteps { get; internal set; }

        /// <summary>Measured ms per emitted token, plain vs speculative. Both
        /// 0 until the governor has enough samples.</summary>
        public double PlainMsPerToken { get; internal set; }
        public double SpecMsPerToken { get; internal set; }

        /// <summary>The cost governor's verdicts during THIS request (the governor
        /// itself is shared by every request an executor runs): rounds and held
        /// wins that ended in a win / a loss, and steps spent parked.</summary>
        public int GovernorWins { get; internal set; }
        public int GovernorLosses { get; internal set; }
        public int GovernorParkedSteps { get; internal set; }

        public double AcceptanceRate => TokensDrafted > 0 ? (double)TokensAccepted / TokensDrafted : 0;

        // Wall-clock phase breakdown (Stopwatch ticks) so a slow speculative
        // request can be attributed: drafting, trunk verify forwards,
        // recurrent-state snapshots, rollback re-forwards, drafter catch-up
        // replays, and plain (non-drafting) steps.
        internal long DraftTicks;
        internal long VerifyTicks;
        internal long SnapshotTicks;
        internal long RollbackTicks;
        internal long CatchUpTicks;
        internal long PlainTicks;

        public double DraftMs => TicksToMs(DraftTicks);
        public double VerifyMs => TicksToMs(VerifyTicks);
        public double SnapshotMs => TicksToMs(SnapshotTicks);
        public double RollbackMs => TicksToMs(RollbackTicks);
        public double CatchUpMs => TicksToMs(CatchUpTicks);
        public double PlainMs => TicksToMs(PlainTicks);

        private static double TicksToMs(long ticks)
            => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

        public void Reset()
        {
            TokensDrafted = TokensAccepted = VerifySteps = PlainSteps = RollbackSteps = ParkedSteps = 0;
            PlainMsPerToken = SpecMsPerToken = 0;
            DraftTicks = VerifyTicks = SnapshotTicks = RollbackTicks = CatchUpTicks = PlainTicks = 0;
        }
    }

    /// <summary>Result of one <see cref="SpeculativeExecution.DecodeStep"/>.</summary>
    public readonly struct SpeculativeStepOutcome
    {
        /// <summary>Drafted tokens accepted by verification (each already
        /// reported through the <c>onDraftAccepted</c> callback, in order).</summary>
        public int AcceptedCount { get; init; }

        /// <summary>The token DRAWN from the first mismatching verify row (or
        /// the bonus row after full acceptance). It is the sequence's next
        /// output token and must be emitted as-is on the caller's next step -
        /// re-drawing from <see cref="NextLogits"/> later would bias the stream
        /// toward the drafts (the classic speculative-sampling pitfall).
        /// -1 when the step degraded to a plain decode (the caller samples
        /// NextLogits), or its row-selection callback stopped the accept loop.</summary>
        public int NextToken { get; init; }

        /// <summary>Caller-owned copy of the logits at the sequence's new last
        /// position (verify row m, or the plain step's logits). Null only after an
        /// explicitly requested greedy verify: consume NextToken as drawn, or stop
        /// when the caller's row-selection callback returned -1.</summary>
        public float[]? NextLogits { get; init; }

        /// <summary>False when the step ran as a plain single-token decode.</summary>
        public bool UsedSpeculation { get; init; }
    }
}
