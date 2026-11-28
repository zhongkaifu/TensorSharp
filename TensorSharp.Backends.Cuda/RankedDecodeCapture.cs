// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;
using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// A direct-CUDA tensor-parallel decode step recorded as one CUDA graph over every rank's device
    /// (<see cref="IRankedDecodeCapture"/>). Unrecorded, each rank's kernels were launched one by one from the host in turn:
    /// Qwen3.8 Flash Next decoded 27.6 tok/s under <c>--tp 2</c> on two RTX PRO 4000s against 96.2 over two whole-layer
    /// stages, whose steps replay their recordings.
    /// <para>
    /// The recording begins on rank 0's stream and every other rank's stream joins it through an event, so each rank's kernels
    /// land in the one graph on their own device; the device-ordered all-reduce's event waits between the ranks' streams
    /// (CudaP2PCommunicator) become the graph's edges. Each rank keeps its own stable inputs, dynamic lengths and the pool
    /// blocks its kernels used (<see cref="RuntimeDecodeCapture"/>, <see cref="CudaPrefillGraphCache.BeginJoinedCapture"/>);
    /// rank 0 launches the graph. A launch on rank 0's stream is not ordered with the other ranks' streams, so each launch
    /// first waits for what they queued (their inputs' uploads, the rows copied in) and each rank's stream then waits for
    /// the launch.
    /// </para>
    /// </summary>
    internal sealed class RankedDecodeCapture : IRankedDecodeCapture
    {
        private const string Key = RuntimeDecodeCapture.RecordingKey;

        /// <summary>CUDA_ERROR_OUT_OF_MEMORY.</summary>
        private const int OutOfMemory = 2;

        private readonly RuntimeDecodeCapture[] _ranks;
        private readonly IntPtr[] _events;
        private bool _disposed;

        public RankedDecodeCapture(RuntimeDecodeCapture[] ranks)
        {
            _ranks = ranks ?? throw new ArgumentNullException(nameof(ranks));
            _events = new IntPtr[ranks.Length];
            for (int r = 0; r < ranks.Length; r++)
            {
                ranks[r].Allocator.Context.MakeCurrent();
                CudaDriverApi.cuEventCreate(out _events[r], 2).ThrowOnError();   // CU_EVENT_DISABLE_TIMING
            }
        }

        public bool Captured { get; private set; }

        public IDecodeCapture Rank(int rank) => _ranks[rank];

        private IntPtr Stream(int rank) => _ranks[rank].Allocator.Stream.Handle;

        /// <summary>Rank 0's stream waits for what every other rank's stream queued.</summary>
        private void OriginWaitsForRanks()
        {
            for (int r = 1; r < _ranks.Length; r++)
            {
                _ranks[r].Allocator.Context.MakeCurrent();
                CudaDriverApi.cuEventRecord(_events[r], Stream(r)).ThrowOnError();
                _ranks[0].Allocator.Context.MakeCurrent();
                CudaDriverApi.cuStreamWaitEvent(Stream(0), _events[r], 0).ThrowOnError();
            }
        }

        /// <summary>Every other rank's stream waits for what rank 0's stream queued (the launch).</summary>
        private void RanksWaitForOrigin()
        {
            _ranks[0].Allocator.Context.MakeCurrent();
            CudaDriverApi.cuEventRecord(_events[0], Stream(0)).ThrowOnError();
            for (int r = 1; r < _ranks.Length; r++)
            {
                _ranks[r].Allocator.Context.MakeCurrent();
                CudaDriverApi.cuStreamWaitEvent(Stream(r), _events[0], 0).ThrowOnError();
            }
        }

        public bool CaptureAndLaunch(Func<bool> run, ReadOnlySpan<int> attendLengths, ReadOnlySpan<int> writeRows,
            IReadOnlyList<IReadOnlyList<Tensor>>[] sequences, IReadOnlyList<Tensor>[] written)
        {
            int n = _ranks.Length;
            if (Captured || _disposed || sequences == null || written == null || sequences.Length != n || written.Length != n)
                return false;
            for (int r = 0; r < n; r++)
                if (!_ranks[r].CanRecord(attendLengths, writeRows, sequences[r])) return false;
            for (int r = 0; r < n; r++)
                _ranks[r].Prepare(attendLengths, writeRows, sequences[r]);
            // What the ranks queued before the recording (their inputs' uploads) precedes its launch on rank 0's stream.
            OriginWaitsForRanks();
            RuntimeDecodeCapture origin = _ranks[0];
            if (!origin.Graphs.BeginCapture(Key)) return false;
            int joined = 0;
            string refusal = null;
            try
            {
                // Every other rank's stream joins the recording.
                origin.Allocator.Context.MakeCurrent();
                CudaDriverApi.cuEventRecord(_events[0], Stream(0)).ThrowOnError();
                for (int r = 1; r < n && refusal == null; r++)
                {
                    if (_ranks[r].Graphs.BeginJoinedCapture(Key, _events[0])) joined = r;
                    else refusal = $"rank {r} could not join the recording";
                }
                if (refusal == null)
                {
                    foreach (RuntimeDecodeCapture rank in _ranks) rank.EnqueueLeadingCopies();
                    var dyn = new CudaDecodeDynParams[n];
                    for (int r = 0; r < n; r++) dyn[r] = _ranks[r].Dyn;
                    CudaDecodeDynParams.ActivateAll(dyn);
                    if (!run()) refusal = "the step declined";
                }
            }
            catch (CudaGraphCaptureAbortedException aborted)
            {
                refusal = aborted.Message;
            }
            catch (CudaException e) when (e.ErrorCode == OutOfMemory)
            {
                refusal = "out of device memory";
            }
            catch
            {
                CudaDecodeDynParams.Deactivate();
                Abort(joined, "the step threw");
                throw;
            }
            CudaDecodeDynParams.Deactivate();
            if (refusal != null)
            {
                Abort(joined, refusal);
                return false;
            }
            // Rank 0's stream waits for every rank's recorded work: the recording ends there.
            for (int r = 1; r < n && refusal == null; r++)
            {
                _ranks[r].Allocator.Context.MakeCurrent();
                if (CudaDriverApi.cuEventRecord(_events[r], Stream(r)) != 0) { refusal = $"rank {r} could not end its part"; break; }
                origin.Allocator.Context.MakeCurrent();
                if (CudaDriverApi.cuStreamWaitEvent(Stream(0), _events[r], 0) != 0) refusal = $"rank {r} could not rejoin rank 0";
            }
            if (refusal != null)
            {
                Abort(joined, refusal);
                return false;
            }
            bool recorded = false;
            try
            {
                recorded = origin.Graphs.EndCaptureAndLaunch(Key, origin.FirstStable, written[0]);
            }
            catch (CudaException e) when (e.ErrorCode == OutOfMemory)
            {
                recorded = false;
            }
            finally
            {
                for (int r = 1; r < n; r++)
                    _ranks[r].Graphs.EndJoinedCapture(Key, _ranks[r].FirstStable, written[r], recorded);
            }
            if (!recorded) return false;
            RanksWaitForOrigin();
            foreach (RuntimeDecodeCapture rank in _ranks) rank.MarkRecordedInGroup();
            Captured = true;
            return true;
        }

        /// <summary>The recording failed: the joined streams rejoin rank 0's (an invalidated recording may refuse), the recording
        /// ends there, and every joined rank's tracking ends.</summary>
        private void Abort(int joined, string reason)
        {
            for (int r = 1; r <= joined; r++)
            {
                _ranks[r].Allocator.Context.MakeCurrent();
                if (CudaDriverApi.cuEventRecord(_events[r], Stream(r)) != 0) continue;
                _ranks[0].Allocator.Context.MakeCurrent();
                CudaDriverApi.cuStreamWaitEvent(Stream(0), _events[r], 0);
            }
            _ranks[0].Graphs.AbortCapture(Key, reason);
            for (int r = 1; r <= joined; r++)
                _ranks[r].Graphs.AbortJoinedCapture(Key);
        }

        public bool TryLaunch(ReadOnlySpan<int> attendLengths, ReadOnlySpan<int> writeRows)
        {
            if (!Captured) throw new InvalidOperationException("no tensor-parallel decode step is recorded");
            foreach (RuntimeDecodeCapture rank in _ranks) rank.PrepareLaunch(attendLengths, writeRows);
            OriginWaitsForRanks();
            try
            {
                _ranks[0].Graphs.Replay(Key);
            }
            catch (CudaException e) when (e.ErrorCode == OutOfMemory)
            {
                return false;
            }
            for (int r = 1; r < _ranks.Length; r++)
                _ranks[r].Graphs.MarkReplayed(Key);
            RanksWaitForOrigin();
            foreach (RuntimeDecodeCapture rank in _ranks) rank.Allocator.OnRecordedStepLaunched();
            return true;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // A launch may still run on any device: every rank's blocks go back only once it is done.
            foreach (RuntimeDecodeCapture rank in _ranks)
            {
                rank.Allocator.Context.MakeCurrent();
                rank.Allocator.Stream.Synchronize();
            }
            foreach (RuntimeDecodeCapture rank in _ranks) rank.Dispose();
            for (int r = 0; r < _ranks.Length; r++)
            {
                if (_events[r] == IntPtr.Zero) continue;
                _ranks[r].Allocator.Context.MakeCurrent();
                CudaDriverApi.cuEventDestroy(_events[r]);
                _events[r] = IntPtr.Zero;
            }
        }
    }
}
