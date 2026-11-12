// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Two batched decode steps in flight under a layer split ("decode lanes"). A step is the first GPU's layers, then the
// next GPU's, so with one step at a time each GPU waits while the other runs its half: on two H100 NVL a 16-row step took
// 40 ms, half of it on each GPU. Two groups of sequences, each with its own step in flight, keep both busy: while the last
// GPU runs one group's half the first runs the other's (each device's stream runs one group's half at a time, so the
// device scratch is shared as before). A group's logits are first read when it samples again, so a lane's step is issued
// without waiting (IssueBatchedDecodeLane) and finished when the group samples next (FinishBatchedDecodeLane) or when any
// other call needs the engine (DrainDecodeLanes, which every other public entry runs first).
//
// What the host writes before a step reads it, and what a step writes for the host to read later, is per lane: the
// tokens, positions, per-device boundary, PLE rows and state tables in pinned memory, and the pinned logits. Lane 0 is the
// engine's own buffers (every other path keeps using them, after a drain), lane 1 a second set; a lane's buffers are
// swapped into the engine's fields while it issues or captures, so its graphs (keyed by lane) bake in its own addresses.
// A lane's step waits for nothing but its own previous step, which finished before the group sampled, where a step on its
// own waits for every device.
using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        /// <summary>A decode lane's pinned buffers, the event its head records, and its step in flight.</summary>
        private sealed class DecodeLane
        {
            public IntPtr PinnedTokens, PosPinned, PinnedLogits;
            public IntPtr[] Boundary, Ple, StateTable;   // per device (zero where the device has none)
            public IntPtr HeadDone;                        // recorded on the last device after the logits copy
            public bool Owned;                             // lane 1: its buffers are its own (lane 0's are the engine's)
            public Slot[] Rows;                            // the step in flight, or null
            public float[][] Out;                          // its rows' logits, filled when it finishes
            public float[] Flat;                           // an uncaptured step's logits
            public long Started;
        }

        private DecodeLane[] _decodeLanes;
        // True while a lane issues: its inputs are free (its previous step finished), so the step waits on no device.
        private bool _laneIssue;
        // The lane whose buffers the engine's fields hold (graph keys carry it).
        private int _laneInFields;

        // TS_Q4E_DECODE_LANES=0: one batched decode step at a time on a layer split too.
        private readonly bool _decodeLanesEnv =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_DECODE_LANES"), "0", StringComparison.Ordinal);

        /// <summary>Whether two batched decode steps can be in flight: a layer split over two or more GPUs, with captured
        /// batched steps (an uncaptured step waits on every device).</summary>
        public bool SupportsDecodeLanes => _devs.Length >= 2 && _decodeLanesEnv && _batchGraphsEnabled && CanCaptureDecode;

        /// <summary>Whether a lane's step is in flight.</summary>
        public bool HasPendingDecodeLanes
            => _decodeLanes != null && (_decodeLanes[0].Rows != null || _decodeLanes[1].Rows != null);

        /// <summary>Lane 0 over the engine's own buffers, lane 1 a second set of the same sizes.</summary>
        private void EnsureDecodeLanes()
        {
            if (_decodeLanes != null)
                return;
            int nd = _devs.Length;
            var own = new DecodeLane
            {
                PinnedTokens = _pinnedTokens, PosPinned = _posPinned, PinnedLogits = _pinnedLogits,
                Boundary = new IntPtr[nd], Ple = new IntPtr[nd], StateTable = new IntPtr[nd],
            };
            var second = new DecodeLane { Boundary = new IntPtr[nd], Ple = new IntPtr[nd], StateTable = new IntPtr[nd], Owned = true };
            int rowsMax = Math.Max(VerifyMaxRows, Math.Max(MaxBatchedDecodeRows, PerSlotMaxRows));
            try
            {
                CudaDriverApi.cuMemHostAlloc(out second.PinnedTokens, new UIntPtr((ulong)MaxBatchedDecodeRows * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out second.PosPinned, new UIntPtr((ulong)rowsMax * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out second.PinnedLogits,
                    new UIntPtr((ulong)MaxBatchedDecodeRows * (ulong)_m.NVocab * 4), 0x1).ThrowOnError();
                for (int d = 0; d < nd; d++)
                {
                    var dev = _devs[d];
                    own.Boundary[d] = dev.BoundaryPinned;
                    own.Ple[d] = dev.PlePinned;
                    own.StateTable[d] = dev.BatchStatePinned;
                    CudaDriverApi.cuMemHostAlloc(out second.Boundary[d],
                        new UIntPtr((ulong)MaxBatchedDecodeRows * (ulong)HcDim * 4), 0x1).ThrowOnError();
                    if (dev.PlePinned != IntPtr.Zero)
                        CudaDriverApi.cuMemHostAlloc(out second.Ple[d],
                            new UIntPtr((ulong)MaxBatchedDecodeRows * (ulong)_m.NEmbd * 4), 0x1).ThrowOnError();
                    if (dev.BatchStatePinned != IntPtr.Zero)
                        CudaDriverApi.cuMemHostAlloc(out second.StateTable[d], new UIntPtr((ulong)BatchStateTableBytes), 0x1).ThrowOnError();
                }
                var last = _devs[_lastDev];
                last.MakeCurrent();
                CudaDriverApi.cuEventCreate(out own.HeadDone, 0x02 /*DISABLE_TIMING*/).ThrowOnError();
                CudaDriverApi.cuEventCreate(out second.HeadDone, 0x02).ThrowOnError();
            }
            catch
            {
                FreeLaneBuffers(second);
                if (own.HeadDone != IntPtr.Zero) CudaDriverApi.cuEventDestroy(own.HeadDone);
                throw;
            }
            _decodeLanes = new[] { own, second };
        }

        private static void FreeLaneBuffers(DecodeLane lane)
        {
            if (lane == null)
                return;
            if (lane.HeadDone != IntPtr.Zero) CudaDriverApi.cuEventDestroy(lane.HeadDone);
            lane.HeadDone = IntPtr.Zero;
            if (!lane.Owned)
                return;
            if (lane.PinnedTokens != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(lane.PinnedTokens);
            if (lane.PosPinned != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(lane.PosPinned);
            if (lane.PinnedLogits != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(lane.PinnedLogits);
            foreach (var arr in new[] { lane.Boundary, lane.Ple, lane.StateTable })
                if (arr != null)
                    foreach (var p in arr)
                        if (p != IntPtr.Zero) CudaDriverApi.cuMemFreeHost(p);
            lane.PinnedTokens = lane.PosPinned = lane.PinnedLogits = IntPtr.Zero;
        }

        private void DisposeDecodeLanes()
        {
            if (_decodeLanes == null)
                return;
            UseLaneFields(0);
            try { _devs[_lastDev].MakeCurrent(); } catch { }
            foreach (var lane in _decodeLanes)
                FreeLaneBuffers(lane);
            _decodeLanes = null;
        }

        /// <summary>Put lane <paramref name="lane"/>'s buffers in the engine's fields (lane 0's are the engine's own).</summary>
        private void UseLaneFields(int lane)
        {
            if (_decodeLanes == null || _laneInFields == lane)
                return;
            var l = _decodeLanes[lane];
            _pinnedTokens = l.PinnedTokens;
            _posPinned = l.PosPinned;
            _pinnedLogits = l.PinnedLogits;
            for (int d = 0; d < _devs.Length; d++)
            {
                _devs[d].BoundaryPinned = l.Boundary[d];
                _devs[d].PlePinned = l.Ple[d];
                _devs[d].BatchStatePinned = l.StateTable[d];
            }
            _laneInFields = lane;
        }

        /// <summary>
        /// Issue a batched decode step on lane <paramref name="lane"/> (0 or 1; its previous step finished first) without
        /// waiting for it: 1 when it is in flight (its rows' logits land in <paramref name="outRows"/> at
        /// <see cref="FinishBatchedDecodeLane"/>), 2 when it ran uncaptured and finished already (a key's first runs),
        /// 0 when it is declined (the rows as <see cref="ForwardBatchedDecode"/> declines them; nothing changed).
        /// </summary>
        public int IssueBatchedDecodeLane(int lane, int[] slotIds, int[] tokens, int[] positions, float[][] outRows)
        {
            if (!SupportsDecodeLanes || lane < 0 || lane > 1)
                return 0;
            // A lane's step waits only for its own previous step; a prompt chunk in flight finishes first.
            DrainPrompt();
            EnsureDecodeLanes();
            var L = _decodeLanes[lane];
            if (L.Rows != null)
                FinishBatchedDecodeLane(lane);
            int n = slotIds?.Length ?? 0;
            // One row too: a group of one on a lane still runs the batched step's kernels (each row computes alike).
            if (n < 1 || n > MaxBatchedDecodeRows || tokens == null || tokens.Length != n || positions == null
                || positions.Length != n || outRows == null || outRows.Length < n)
                return 0;
            var rows = new Slot[n];
            var seen = new System.Collections.Generic.HashSet<int>();
            for (int i = 0; i < n; i++)
            {
                // A row of the other lane's step in flight is Failed until that step finishes.
                if (!seen.Add(slotIds[i]) || !_slots.TryGetValue(slotIds[i], out rows[i])
                    || rows[i].Failed || rows[i].NPast != positions[i] || positions[i] + 1 > _m.NCtx)
                    return 0;
            }
            try
            {
                foreach (Slot r in rows)
                    EnsureCapacity(r, r.NPast + 1);
            }
            catch (InvalidOperationException)
            {
                return 0;
            }
            int vocab = _m.NVocab;
            for (int i = 0; i < n; i++)
                if (outRows[i] == null || outRows[i].Length < vocab)
                    outRows[i] = new float[vocab];

            UseLaneFields(lane);
            bool done;
            try
            {
                StageBatchStates(rows);
                foreach (var r in rows)
                    r.Failed = true;
                _rowsInFlight = n;
                ArmHostExperts();
                L.Started = System.Diagnostics.Stopwatch.GetTimestamp();
                _issueEnd = 0;
                _laneIssue = true;
                try
                {
                    if (ForwardBatchGraphed(rows, tokens, positions, 1, greedy: false, hidden: false))
                    {
                        var last = _devs[_lastDev];
                        last.MakeCurrent();
                        CudaDriverApi.cuEventRecord(L.HeadDone, last.Stream).ThrowOnError();
                        done = false;
                    }
                    else
                    {
                        // Not captured yet: the step runs uncaptured and waits for itself, into this lane's buffers.
                        _laneIssue = false;
                        L.Flat ??= new float[(long)MaxBatchedDecodeRows * vocab];
                        ForwardUbatch(null, tokens, 0, n, 0, L.Flat, rows, positions);
                        done = true;
                    }
                }
                finally
                {
                    _laneIssue = false;
                    _rowsInFlight = 0;
                }
            }
            finally
            {
                UseLaneFields(0);
            }
            L.Rows = rows;
            L.Out = outRows;
            if (!done)
                return 1;
            for (int i = 0; i < n; i++)
                Array.Copy(L.Flat, (long)i * vocab, outRows[i], 0, vocab);
            CompleteLane(L);
            return 2;
        }

        /// <summary>Wait for lane <paramref name="lane"/>'s step in flight, if any, and copy its rows' logits out.</summary>
        public void FinishBatchedDecodeLane(int lane)
        {
            if (_decodeLanes == null || lane < 0 || lane > 1)
                return;
            var L = _decodeLanes[lane];
            if (L.Rows == null)
                return;
            var last = _devs[_lastDev];
            last.MakeCurrent();
            CudaDriverApi.cuEventSynchronize(L.HeadDone).ThrowOnError();
            int n = L.Rows.Length, vocab = _m.NVocab;
            long src = (long)L.PinnedLogits, rowBytes = (long)vocab * 4;
            var outRows = L.Out;
            // 1 MB a row: the copies run side by side.
            if (n >= 4)
                System.Threading.Tasks.Parallel.For(0, n, i => Marshal.Copy((IntPtr)(src + i * rowBytes), outRows[i], 0, vocab));
            else
                for (int i = 0; i < n; i++)
                    Marshal.Copy((IntPtr)(src + i * rowBytes), outRows[i], 0, vocab);
            CompleteLane(L);
        }

        /// <summary>A lane's rows past their step: heads moved on, nothing in flight.</summary>
        private void CompleteLane(DecodeLane L)
        {
            int n = L.Rows.Length;
            foreach (var r in L.Rows)
            {
                r.NPast++;
                r.Failed = false;
            }
            L.Rows = null;
            L.Out = null;
            CheckHostExperts(n);
            AdaptResidency();
            CountForwardTime(L.Started);
            System.Threading.Interlocked.Increment(ref _tLaneSteps);
        }

        /// <summary>Finish every lane's step in flight and a prompt chunk left in flight (any call but a lane's own, or the
        /// prompt's next chunk, runs this first).</summary>
        public void DrainDecodeLanes()
        {
            DrainPrompt();
            DrainLanes();
        }

        /// <summary>Finish every lane's step in flight; a prompt chunk in flight stays (PromptPipeline.cs).</summary>
        private void DrainLanes()
        {
            if (_decodeLanes == null)
                return;
            // The older step first: lane 0's group is issued first in a pipelined decode step.
            FinishBatchedDecodeLane(0);
            FinishBatchedDecodeLane(1);
        }

        private long _tLaneSteps;

        /// <summary>Batched decode steps run on a decode lane (tests, diagnostics).</summary>
        internal long DecodeLaneSteps => System.Threading.Interlocked.Read(ref _tLaneSteps);
    }
}
