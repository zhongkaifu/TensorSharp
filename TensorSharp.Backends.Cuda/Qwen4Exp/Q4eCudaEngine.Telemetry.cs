// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// What the Qwen3.8-Flash-Next direct-CUDA engine counts for serving telemetry: its sequence slots,
// their growth and rewinds, its captured graphs, and how a forward's time splits between the host
// issuing work and waiting for the GPU. The engine thread writes them; the metrics thread reads a
// snapshot (Qwen4ExpCudaModel turns it into TensorSharp's instruments), so every field is read
// and written through Interlocked.
using System.Threading;

namespace TensorSharp.Cuda
{
    public sealed partial class Q4eCudaEngine
    {
        /// <summary>A snapshot of the engine's serving counters (see <see cref="ReadTelemetry"/>).</summary>
        public readonly record struct TelemetryCounters(
            int Slots, long SlotBytes, long SlotCapacityTokens,
            long Grows, long GrowFailures,
            long RewindsToCheckpoint, long RewindsWithReplay, long RewindsToZero, long RewindsRefused, long ReplayedTokens,
            long SoloGraphCaptures, long SoloGraphLaunches, long BatchedGraphCaptures, long BatchedGraphLaunches,
            long GraphCaptureFailures, double ForwardIssueSeconds, double ForwardWaitSeconds,
            long LaneSteps = 0, long LaneRows = 0, long LaneFinishesReady = 0, long LaneFinishesWaited = 0,
            double LaneIssueSeconds = 0, double LaneWaitSeconds = 0, double LaneIdleSeconds = 0);

        private int _tSlots;
        private long _tSlotBytes, _tSlotCapacity;
        private long _tGrows, _tGrowFailures;
        private long _tRewindCheckpoint, _tRewindReplay, _tRewindZero, _tRewindRefused, _tReplayTokens;
        private long _tSoloCaptures, _tSoloLaunches, _tBatchedCaptures, _tBatchedLaunches, _tCaptureFailures;
        private long _tIssueTicks, _tWaitTicks;
        // Decode lanes (DecodeLanes.cs): rows their steps carried, finishes that found the step done already (the GPUs
        // were waiting for the host) against finishes that waited for it, and the host's time issuing a lane's step,
        // waiting for one, and between a step's end as the host saw it and its lane's next issue.
        private long _tLaneRows, _tLaneReady, _tLaneWaited, _tLaneIssueTicks, _tLaneWaitTicks, _tLaneIdleTicks;

        /// <summary>The serving counters, from any thread.</summary>
        public TelemetryCounters ReadTelemetry()
        {
            double f = System.Diagnostics.Stopwatch.Frequency;
            return new TelemetryCounters(
                Volatile.Read(ref _tSlots), Interlocked.Read(ref _tSlotBytes), Interlocked.Read(ref _tSlotCapacity),
                Interlocked.Read(ref _tGrows), Interlocked.Read(ref _tGrowFailures),
                Interlocked.Read(ref _tRewindCheckpoint), Interlocked.Read(ref _tRewindReplay),
                Interlocked.Read(ref _tRewindZero), Interlocked.Read(ref _tRewindRefused), Interlocked.Read(ref _tReplayTokens),
                Interlocked.Read(ref _tSoloCaptures), Interlocked.Read(ref _tSoloLaunches),
                Interlocked.Read(ref _tBatchedCaptures), Interlocked.Read(ref _tBatchedLaunches),
                Interlocked.Read(ref _tCaptureFailures),
                Interlocked.Read(ref _tIssueTicks) / f, Interlocked.Read(ref _tWaitTicks) / f,
                Interlocked.Read(ref _tLaneSteps), Interlocked.Read(ref _tLaneRows),
                Interlocked.Read(ref _tLaneReady), Interlocked.Read(ref _tLaneWaited),
                Interlocked.Read(ref _tLaneIssueTicks) / f, Interlocked.Read(ref _tLaneWaitTicks) / f,
                Interlocked.Read(ref _tLaneIdleTicks) / f);
        }

        /// <summary>A slot came (<paramref name="sign"/> 1) or went (-1): its bytes and capacity.</summary>
        private void CountSlot(Slot s, int sign)
        {
            long bytes = 0;
            foreach (long b in s.DeviceBytes)
                bytes += b;
            Interlocked.Add(ref _tSlots, sign);
            Interlocked.Add(ref _tSlotBytes, sign * bytes);
            Interlocked.Add(ref _tSlotCapacity, sign * (long)s.Capacity);
        }

        /// <summary>The forward that began at <paramref name="start"/>: the host's issue time (to its
        /// last launch before the logits sync, <see cref="_issueEnd"/>), and the wait after it. A
        /// forward that read no logits never waited.</summary>
        private void CountForwardTime(long start)
        {
            long end = System.Diagnostics.Stopwatch.GetTimestamp();
            long issueEnd = _issueEnd > start && _issueEnd <= end ? _issueEnd : end;
            Interlocked.Add(ref _tIssueTicks, issueEnd - start);
            Interlocked.Add(ref _tWaitTicks, end - issueEnd);
        }
    }
}
