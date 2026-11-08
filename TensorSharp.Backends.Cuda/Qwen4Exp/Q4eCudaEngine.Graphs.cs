// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Captured single-sequence decode steps. One decode step issues about 30 launches per layer; on
// the A40 the host needs longer to queue them than the GPU needs to run them, so the step ran at
// the launch rate. A captured step is one graph launch per device.
//
// Each device captures its run of layers once per (sequence, regime) and replays it. What changes
// from step to step is read at launch time, never baked in:
//   - the token and the position come from pinned host memory through the graph's first copies,
//     and every kernel that depends on the position reads the device copy (the KV and indexer
//     cache writes, RoPE, the block pooling, the scores and the selection, the attention span);
//   - the PLE rows are gathered on the host into the pinned staging the graph copies from;
//   - the streams cross a layer-split boundary through the pinned boundary buffer, ordered by an
//     event wait queued between the graph launches.
// The regime is the set of QSA layers whose selection runs at this position (the sparse path is
// only issued past the indexer's width): the sets are nested as the position grows, so their size
// names them.
//
// Every buffer a captured step touches belongs to the engine or the sequence, so graphs stay valid
// until their sequence is freed (CudaDecodeCapture refuses a step that allocates or reads shared
// scratch). A capture that fails for any reason turns graphs off for the engine with a note and
// the step runs uncaptured.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private readonly bool _graphsEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_GRAPHS"), "0", StringComparison.Ordinal);
        private bool _graphsFailed;
        /// <summary>Per (sequence, regime): one executable graph per device holding layers.</summary>
        private readonly Dictionary<(int Slot, int Regime), IntPtr[]> _graphs = new Dictionary<(int, int), IntPtr[]>();
        /// <summary>The step's position, read by every device's graph (PORTABLE pinned memory).</summary>
        private IntPtr _posPinned;

        /// <summary>Decode steps may be captured: not while stages are timed or the sparse selection or
        /// the router is probed (they read the device mid-step).</summary>
        private bool CanCaptureDecode => _graphsEnabled && !_graphsFailed && _perf < 2 && !_sampling && SparseProbe == null
            && RouterProbe == null;

        /// <summary>Captured decode steps this engine holds (tests, diagnostics).</summary>
        internal int CapturedDecodeGraphs => _graphs.Count;

        /// <summary>QSA layers whose selection runs for a row at <paramref name="p0"/>.</summary>
        private int DecodeRegime(int p0)
        {
            int n = 0;
            foreach (var L in _layers)
                if (L.Ratio > 0 && p0 + 1 > _m.IdxTopK + L.Ratio - 1)
                    n++;
            return n;
        }

        /// <summary>One decode step of <paramref name="slot"/> through its captured graphs, capturing
        /// them on first use. False when graphs are off, having just failed: the caller runs the step
        /// uncaptured.</summary>
        private bool ForwardDecodeGraphed(Slot slot, int token, int p0, float[] logitsOut)
        {
            // The pinned inputs are rewritten below: every device's previous work must have run.
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            *(int*)_pinnedTokens = token;
            *(int*)_posPinned = p0;
            if (_pleLayer >= 0)
                GatherPleRows(slot, new[] { token }, 0, 1, p0, null, null);

            var key = (slot.Id, DecodeRegime(p0));
            if (!_graphs.TryGetValue(key, out IntPtr[] execs))
            {
                try
                {
                    execs = CaptureDecode(slot, p0);
                    System.Threading.Interlocked.Increment(ref _tSoloCaptures);
                }
                catch (Exception ex)
                {
                    System.Threading.Interlocked.Increment(ref _tCaptureFailures);
                    _graphsFailed = true;
                    Console.Error.WriteLine($"[q4e-cuda] decode steps run uncaptured from here on: the capture failed ({ex.Message})");
                    return false;
                }
                _graphs[key] = execs;
            }

            for (int d = 0; d <= _lastDev; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                if (d > 0)
                    CudaDriverApi.cuStreamWaitEvent(dev.Stream, _devs[d - 1].XsReadyEv, 0).ThrowOnError();
                CudaDriverApi.cuGraphLaunch(execs[d], dev.Stream).ThrowOnError();
                if (d < _lastDev)
                    CudaDriverApi.cuEventRecord(dev.XsReadyEv, dev.Stream).ThrowOnError();
            }
            // When the host had queued the step: telemetry's issue/wait split, and TS_Q4E_PERF.
            _issueEnd = System.Diagnostics.Stopwatch.GetTimestamp();
            FinishHead(1, logitsOut, false, null);
            System.Threading.Interlocked.Increment(ref _tSoloLaunches);
            return true;
        }

        private IntPtr[] CaptureDecode(Slot slot, int p0)
        {
            var execs = new IntPtr[_lastDev + 1];
            try
            {
                for (int d = 0; d <= _lastDev; d++)
                {
                    var dev = _devs[d];
                    dev.MakeCurrent();
                    execs[d] = CudaDecodeCapture.Capture(dev.Alloc, () => IssueDecodeStep(dev, slot, p0));
                }
                return execs;
            }
            catch
            {
                DestroyGraphs(execs);
                throw;
            }
        }

        /// <summary>One device's part of a decode step, as captured: the position (and on device 0 the
        /// token) from pinned memory, the streams from the previous device's boundary buffer, its
        /// layers, then the boundary copy or, on the last device, the head up to the pinned logits.</summary>
        private void IssueDecodeStep(Dev dev, Slot slot, int p0)
        {
            var m = _m;
            var boundary = new UIntPtr((ulong)HcDim * 4);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.PosDev), _posPinned, new UIntPtr(4), dev.Stream).ThrowOnError();
            if (dev.Ordinal == 0)
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Tokens), _pinnedTokens, new UIntPtr(4), dev.Stream).ThrowOnError();
                dev.DK.Embed(_tokEmbd.Ptr, dev.Tokens, dev.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, 1, m.NEmbd, dev.Stream);
            }
            else
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Xs), _devs[dev.Ordinal - 1].BoundaryPinned, boundary, dev.Stream).ThrowOnError();
            }
            for (int il = 0; il < m.NLayer; il++)
                if (_layers[il].Device == dev.Ordinal)
                    RunLayer(dev, il, slot, 1, p0, null, null, Ptr(dev.PosDev));
            if (dev.Ordinal < _lastDev)
                CudaDriverApi.cuMemcpyDtoHAsync(dev.BoundaryPinned, Ptr(dev.Xs), boundary, dev.Stream).ThrowOnError();
            else
                IssueHead(1, logits: true, allRows: false, hidden: false);
        }

        // ---------------------------------------------------------------- batched decode steps
        //
        // An uncaptured batched step issues every kernel from the host, about 1,700 launches plus
        // ~100 per row for this model, so it was bound by launch overhead: measured on two H100 NVL
        // (Q8_0, layer split 2), a step cost 10.5 ms at one row (captured) but 17 ms at two rows and
        // 26 ms at four. A captured step replays the same kernels in the same order, so every row
        // still computes exactly what its uncaptured step computes.
        //
        // The kernels bake in each row's cache pointers and run the QSA selection only in the regime
        // the row's position is in, so a graph belongs to one ordered set of slots and their regimes.
        // Positions are read from the device (PosDev[row]), tokens and PLE rows from pinned memory,
        // and the recurrent states through the per-row pointer table StageBatchStates writes, so the
        // same graph serves every step while the set and the regimes stay the same.

        private readonly bool _batchGraphsEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_BATCH_GRAPHS"), "0", StringComparison.Ordinal);
        /// <summary>Executable graphs per batched step key, at most <see cref="MaxBatchGraphs"/>, the
        /// least recently used dropped first.</summary>
        private readonly Dictionary<string, (IntPtr[] Execs, int[] Slots, LinkedListNode<string> Node)> _batchGraphs =
            new Dictionary<string, (IntPtr[], int[], LinkedListNode<string>)>(StringComparer.Ordinal);
        private readonly LinkedList<string> _batchGraphLru = new LinkedList<string>();
        /// <summary>Row counts whose capture failed: their steps run uncaptured from then on.</summary>
        private readonly HashSet<int> _batchCaptureFailed = new HashSet<int>();
        private readonly int _maxBatchGraphs = Math.Max(1, EnvInt("TS_Q4E_BATCH_GRAPHS_MAX", 64));

        /// <summary>Captured batched decode steps this engine holds (tests, diagnostics).</summary>
        internal int CapturedBatchedDecodeGraphs => _batchGraphs.Count;

        private string BatchGraphKey(int[] slotIds, int[] positions)
        {
            var sb = new System.Text.StringBuilder(slotIds.Length * 8);
            for (int i = 0; i < slotIds.Length; i++)
                sb.Append(slotIds[i]).Append(':').Append(DecodeRegime(positions[i])).Append(',');
            return sb.ToString();
        }

        /// <summary>One batched decode step (the rows' states already staged) through its captured
        /// graphs, capturing them on first use. False when graphs are off or this row count's capture
        /// failed: the caller runs the step uncaptured.</summary>
        private bool ForwardBatchedDecodeGraphed(Slot[] rows, int[] slotIds, int[] tokens, int[] positions, float[] logitsOut)
        {
            int n = rows.Length;
            if (!_batchGraphsEnabled || !CanCaptureDecode || _batchCaptureFailed.Contains(n))
                return false;
            // The pinned inputs are rewritten below: every device's previous work must have run.
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            for (int r = 0; r < n; r++)
            {
                ((int*)_pinnedTokens)[r] = tokens[r];
                ((int*)_posPinned)[r] = positions[r];
            }
            // The rows' tokens join their histories at their positions, so the uncaptured step a
            // failed capture falls back to writes the same entries again.
            if (_pleLayer >= 0)
                GatherPleRows(null, tokens, 0, n, 0, rows, positions);

            string key = BatchGraphKey(slotIds, positions);
            IntPtr[] execs;
            if (_batchGraphs.TryGetValue(key, out var entry))
            {
                execs = entry.Execs;
                _batchGraphLru.Remove(entry.Node);
                _batchGraphLru.AddLast(entry.Node);
            }
            else
            {
                try
                {
                    execs = CaptureBatchedDecode(rows, positions);
                    System.Threading.Interlocked.Increment(ref _tBatchedCaptures);
                }
                catch (Exception ex)
                {
                    System.Threading.Interlocked.Increment(ref _tCaptureFailures);
                    _batchCaptureFailed.Add(n);
                    Console.Error.WriteLine($"[q4e-cuda] batched decode steps of {n} rows run uncaptured from here on: the capture failed ({ex.Message})");
                    return false;
                }
                while (_batchGraphs.Count >= _maxBatchGraphs && _batchGraphLru.First != null)
                    RemoveBatchGraph(_batchGraphLru.First.Value);
                _batchGraphs[key] = (execs, (int[])slotIds.Clone(), _batchGraphLru.AddLast(key));
            }

            for (int d = 0; d <= _lastDev; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                if (d > 0)
                    CudaDriverApi.cuStreamWaitEvent(dev.Stream, _devs[d - 1].XsReadyEv, 0).ThrowOnError();
                CudaDriverApi.cuGraphLaunch(execs[d], dev.Stream).ThrowOnError();
                if (d < _lastDev)
                    CudaDriverApi.cuEventRecord(dev.XsReadyEv, dev.Stream).ThrowOnError();
            }
            // When the host had queued the step: telemetry's issue/wait split, and TS_Q4E_PERF.
            _issueEnd = System.Diagnostics.Stopwatch.GetTimestamp();
            FinishHead(n, logitsOut, true, null);
            System.Threading.Interlocked.Increment(ref _tBatchedLaunches);
            return true;
        }

        private IntPtr[] CaptureBatchedDecode(Slot[] rows, int[] positions)
        {
            var execs = new IntPtr[_lastDev + 1];
            try
            {
                for (int d = 0; d <= _lastDev; d++)
                {
                    var dev = _devs[d];
                    dev.MakeCurrent();
                    execs[d] = CudaDecodeCapture.Capture(dev.Alloc, () => IssueBatchedDecodeStep(dev, rows, positions));
                }
                return execs;
            }
            catch
            {
                DestroyGraphs(execs);
                throw;
            }
        }

        /// <summary>One device's part of a batched decode step, as captured: the rows' positions (and
        /// on device 0 their tokens) from pinned memory, the streams from the previous device's
        /// boundary buffer, its layers, then the boundary copy or, on the last device, the head up to
        /// every row's pinned logits.</summary>
        private void IssueBatchedDecodeStep(Dev dev, Slot[] rows, int[] positions)
        {
            var m = _m;
            int n = rows.Length;
            var boundary = new UIntPtr((ulong)((long)n * HcDim * 4));
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.PosDev), _posPinned, new UIntPtr((ulong)n * 4), dev.Stream).ThrowOnError();
            if (dev.Ordinal == 0)
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Tokens), _pinnedTokens, new UIntPtr((ulong)n * 4), dev.Stream).ThrowOnError();
                dev.DK.Embed(_tokEmbd.Ptr, dev.Tokens, dev.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, n, m.NEmbd, dev.Stream);
            }
            else
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Xs), _devs[dev.Ordinal - 1].BoundaryPinned, boundary, dev.Stream).ThrowOnError();
            }
            for (int il = 0; il < m.NLayer; il++)
                if (_layers[il].Device == dev.Ordinal)
                    RunLayer(dev, il, null, n, 0, rows, positions, Ptr(dev.PosDev));
            if (dev.Ordinal < _lastDev)
                CudaDriverApi.cuMemcpyDtoHAsync(dev.BoundaryPinned, Ptr(dev.Xs), boundary, dev.Stream).ThrowOnError();
            else
                IssueHead(n, logits: true, allRows: true, hidden: false);
        }

        private void RemoveBatchGraph(string key)
        {
            if (!_batchGraphs.TryGetValue(key, out var entry))
                return;
            DestroyGraphs(entry.Execs);
            _batchGraphLru.Remove(entry.Node);
            _batchGraphs.Remove(key);
        }

        private void DestroySlotGraphs(int slotId)
        {
            List<(int, int)> keys = null;
            foreach (var key in _graphs.Keys)
                if (key.Slot == slotId)
                    (keys ??= new List<(int, int)>()).Add(key);
            if (keys != null)
                foreach (var key in keys)
                {
                    DestroyGraphs(_graphs[key]);
                    _graphs.Remove(key);
                }
            // A batched graph holding the slot holds its cache pointers too.
            List<string> batchKeys = null;
            foreach (var kv in _batchGraphs)
                if (Array.IndexOf(kv.Value.Slots, slotId) >= 0)
                    (batchKeys ??= new List<string>()).Add(kv.Key);
            if (batchKeys != null)
                foreach (var key in batchKeys)
                    RemoveBatchGraph(key);
        }

        private void DestroyGraphs(IntPtr[] execs)
        {
            for (int d = 0; d < execs.Length; d++)
            {
                if (execs[d] == IntPtr.Zero)
                    continue;
                _devs[d].MakeCurrent();
                CudaDriverApi.cuGraphExecDestroy(execs[d]);
                execs[d] = IntPtr.Zero;
            }
        }
    }
}
