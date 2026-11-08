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
// A speculative verify window of n rows (Verify.cs) is captured the same way per (sequence, regime, n): the
// rows' tokens and positions come from pinned memory, the recurrent states are saved inside the graph, and the
// head writes every row's logits and hidden streams.
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
        /// <summary>Per (sequence, regime, window): one executable graph per device holding layers. Window 0 is a
        /// decode step, n a verify window of n rows.</summary>
        private readonly Dictionary<(int Slot, int Regime, int Window), IntPtr[]> _graphs = new Dictionary<(int, int, int), IntPtr[]>();
        /// <summary>The step's position, read by every device's graph (PORTABLE pinned memory).</summary>
        private IntPtr _posPinned;

        /// <summary>Decode steps may be captured: not while stages are timed or the sparse selection or
        /// the router is probed (they read the device mid-step).</summary>
        private bool CanCaptureDecode => _graphsEnabled && !_graphsFailed && _perf < 2 && SparseProbe == null
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

            var key = (slot.Id, DecodeRegime(p0), 0);
            if (!_graphs.TryGetValue(key, out IntPtr[] execs))
            {
                try
                {
                    execs = CaptureSteps(dev => IssueDecodeStep(dev, slot, p0));
                }
                catch (Exception ex)
                {
                    _graphsFailed = true;
                    Console.Error.WriteLine($"[q4e-cuda] decode steps run uncaptured from here on: the capture failed ({ex.Message})");
                    return false;
                }
                _graphs[key] = execs;
            }

            LaunchSteps(execs);
            FinishHead(1, logitsOut, false, null);
            return true;
        }

        /// <summary>
        /// A verify window of <paramref name="tokens"/> at <paramref name="p0"/> through its captured graphs (Verify.cs
        /// sets the capture flags around it). False when graphs are off: the caller runs the window uncaptured.
        /// </summary>
        /// <param name="idsOut">A greedy verify: each row's first-max token instead of its logits (a graph of its own).</param>
        private bool ForwardVerifyGraphed(Slot slot, int[] tokens, int p0, float[] logitsAll, float[] hAll, int[] idsOut = null)
        {
            int nt = tokens.Length;
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            for (int t = 0; t < nt; t++)
            {
                ((int*)_pinnedTokens)[t] = tokens[t];
                ((int*)_posPinned)[t] = p0 + t;
            }
            if (_pleLayer >= 0)
                GatherPleRows(slot, tokens, 0, nt, p0, null, null);

            bool greedy = idsOut != null;
            var key = (slot.Id, DecodeRegime(p0 + nt - 1), greedy ? GreedyWindowKey + nt : nt);
            if (!_graphs.TryGetValue(key, out IntPtr[] execs))
            {
                try
                {
                    execs = CaptureSteps(dev => IssueVerifyStep(dev, slot, p0, nt, greedy));
                }
                catch (Exception ex)
                {
                    _graphsFailed = true;
                    Console.Error.WriteLine($"[q4e-cuda] decode steps run uncaptured from here on: a verify capture failed ({ex.Message})");
                    return false;
                }
                _graphs[key] = execs;
            }
            LaunchSteps(execs);
            FinishHead(nt, logitsAll, true, hAll, 0, idsOut);
            return true;
        }

        /// <summary>Graph keys of greedy verify windows: this plus the window's rows.</summary>
        private const int GreedyWindowKey = 64;

        /// <summary>One graph launch per device, the next device waiting on the previous one's boundary copy.</summary>
        private void LaunchSteps(IntPtr[] execs)
        {
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
            if (_perf > 0)
                _issueEnd = System.Diagnostics.Stopwatch.GetTimestamp();
        }

        private IntPtr[] CaptureSteps(Action<Dev> issue)
        {
            var execs = new IntPtr[_lastDev + 1];
            try
            {
                for (int d = 0; d <= _lastDev; d++)
                {
                    var dev = _devs[d];
                    dev.MakeCurrent();
                    execs[d] = CudaDecodeCapture.Capture(dev.Alloc, () => issue(dev));
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
            bool ready = false;
            for (int il = 0; il < m.NLayer; il++)
                if (_layers[il].Device == dev.Ordinal)
                    ready = RunLayer(dev, il, slot, 1, p0, null, null, Ptr(dev.PosDev), ready);
            if (dev.Ordinal < _lastDev)
                CudaDriverApi.cuMemcpyDtoHAsync(dev.BoundaryPinned, Ptr(dev.Xs), boundary, dev.Stream).ThrowOnError();
            else
                IssueHead(1, logits: true, allRows: false, hidden: false);
        }

        /// <summary>One device's part of a verify window, as captured: <see cref="IssueDecodeStep"/> over
        /// <paramref name="nt"/> rows, the recurrent states saved as the window runs (Verify.cs), and on the last
        /// device every row's logits and hidden streams.</summary>
        private void IssueVerifyStep(Dev dev, Slot slot, int p0, int nt, bool greedy = false)
        {
            var m = _m;
            var boundary = new UIntPtr((ulong)nt * (ulong)HcDim * 4);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.PosDev), _posPinned, new UIntPtr((ulong)nt * 4), dev.Stream).ThrowOnError();
            if (dev.Ordinal == 0)
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Tokens), _pinnedTokens, new UIntPtr((ulong)nt * 4), dev.Stream).ThrowOnError();
                dev.DK.Embed(_tokEmbd.Ptr, dev.Tokens, dev.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, nt, m.NEmbd, dev.Stream);
            }
            else
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Xs), _devs[dev.Ordinal - 1].BoundaryPinned, boundary, dev.Stream).ThrowOnError();
            }
            bool ready = false;
            for (int il = 0; il < m.NLayer; il++)
                if (_layers[il].Device == dev.Ordinal)
                    ready = RunLayer(dev, il, slot, nt, p0, null, null, Ptr(dev.PosDev), ready);
            if (dev.Ordinal < _lastDev)
                CudaDriverApi.cuMemcpyDtoHAsync(dev.BoundaryPinned, Ptr(dev.Xs), boundary, dev.Stream).ThrowOnError();
            else
                IssueHead(nt, logits: true, allRows: true, hidden: true, greedy: greedy);
        }

        private void DestroySlotGraphs(int slotId)
        {
            List<(int, int, int)> keys = null;
            foreach (var key in _graphs.Keys)
                if (key.Slot == slotId)
                    (keys ??= new List<(int, int, int)>()).Add(key);
            if (keys == null)
                return;
            foreach (var key in keys)
            {
                DestroyGraphs(_graphs[key]);
                _graphs.Remove(key);
            }
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
