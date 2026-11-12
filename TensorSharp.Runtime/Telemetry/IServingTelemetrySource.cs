// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

#nullable enable

namespace TensorSharp.Runtime.Telemetry
{
    /// <summary>
    /// A model whose engine keeps sequence slots of its own (a direct-CUDA engine) reports them
    /// through this: <see cref="TensorSharpMetrics"/> reads it on the metrics collection thread
    /// for every engine whose model implements it. The read must be cheap, thread-safe and must
    /// not wait on GPU work: return counters the engine already keeps.
    /// </summary>
    public interface IServingTelemetrySource
    {
        ServingTelemetrySnapshot ReadServingTelemetry();
    }

    /// <summary>
    /// What a slot-keeping engine counts. Totals are monotonic over the engine's life; the others
    /// are the state at the read. Values an engine does not track stay 0.
    /// </summary>
    public readonly record struct ServingTelemetrySnapshot
    {
        /// <summary>Slots bound to a request, and slots kept for a conversation's next turn.</summary>
        public int RunningSlots { get; init; }
        public int RetainedSlots { get; init; }
        /// <summary>Device bytes every slot's caches and states hold together.</summary>
        public long SlotBytes { get; init; }
        /// <summary>Positions every slot's caches hold together (their capacity, not their use).</summary>
        public long SlotCapacityTokens { get; init; }

        /// <summary>Times a slot's caches grew, and times a slot could not grow for lack of memory.</summary>
        public long SlotGrows { get; init; }
        public long SlotGrowFailures { get; init; }
        /// <summary>Retained slots given back so a new request could have a slot, or a running one grow.</summary>
        public long ReclaimsForNewSlot { get; init; }
        public long ReclaimsForGrowth { get; init; }

        /// <summary>Rewinds by route, and the tokens the replay route forwarded again.</summary>
        public long RewindsToCheckpoint { get; init; }
        public long RewindsWithReplay { get; init; }
        public long RewindsToZero { get; init; }
        public long RewindsRefused { get; init; }
        public long ReplayedTokens { get; init; }

        /// <summary>CUDA graphs captured and launched, by kind, and captures that failed.</summary>
        public long SoloGraphCaptures { get; init; }
        public long SoloGraphLaunches { get; init; }
        public long BatchedGraphCaptures { get; init; }
        public long BatchedGraphLaunches { get; init; }
        public long GraphCaptureFailures { get; init; }

        /// <summary>
        /// Where a forward's wall time went, summed: the host issuing its work, and the host then
        /// waiting for the GPU to finish it (the logits sync). Issue time close to the whole means
        /// the forward is bound by the host (launch overhead); wait time close to it means the GPU.
        /// </summary>
        public double ForwardIssueSeconds { get; init; }
        public double ForwardWaitSeconds { get; init; }

        /// <summary>
        /// Decode steps an engine keeps in flight on lanes of their own (two groups of sequences on a layer split, one
        /// GPU running each group's half): steps and rows, how many steps were done before the host came back for their
        /// logits (the GPUs waited for the host) against how many the host waited for, and the host's time issuing a
        /// lane's step, waiting for one, and with a lane empty between a step's finish and its next issue (sampling
        /// and scheduling: time that lane's GPU work could have been running).
        /// </summary>
        public long DecodeLaneSteps { get; init; }
        public long DecodeLaneRows { get; init; }
        public long DecodeLaneFinishesReady { get; init; }
        public long DecodeLaneFinishesWaited { get; init; }
        public double DecodeLaneIssueSeconds { get; init; }
        public double DecodeLaneWaitSeconds { get; init; }
        public double DecodeLaneIdleSeconds { get; init; }
    }
}
