// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Captured batched steps: several slots' rows in one forward, a batched decode step (one row a sequence) or a batched
// verify (a window a sequence). An uncaptured batched step issues every kernel from the host, about 1,700 launches plus
// ~100 a row for this model, so it was bound by launch overhead: on two H100 NVL (Q8_0, layer split 2) a step cost
// 10.5 ms at one row (captured) but 17 ms at two rows and 26 ms at four; on the RTX 5090 an uncaptured 4-row verify
// window spent 12.6 ms issuing. A captured step replays the same kernels in the same order, so every row computes exactly
// what its uncaptured step computes (a test holds them bit for bit).
//
// What changes from step to step is read at launch time: the tokens and every row's position come from pinned memory
// (each sequence's attention reads its rows' device positions), the recurrent layers read each row's states through the
// batch state table StageBatchStates uploads before the launch, the PLE rows are gathered on the host before it, and a
// layer split hands the streams over through the pinned boundary buffer. What the host decides is in the key: the rows a
// sequence has, the head's form (first-max tokens or logits, with or without hidden streams), and each sequence's slot and
// regime (the QSA layers whose selection runs at its rows' positions; the kernels bake in each slot's cache pointers).
//
// A key is captured on its third run (concurrent requests come and go, and a capture costs more than the steps of a set
// of sequences that lasts a few), and at most TS_Q4E_BATCH_GRAPHS_MAX are kept, the least recently used dropped first.
// Each holds device memory: a slot that cannot be allocated or grown drops them all first. Steps that stage their missed
// experts (planned on the host) are not captured. A row count whose capture failed runs uncaptured from then on; solo
// graphs are unaffected. TS_Q4E_BATCH_GRAPHS=0 runs every batched step uncaptured.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private readonly bool _batchGraphsEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_BATCH_GRAPHS"), "0", StringComparison.Ordinal);
        /// <summary>Executable graphs per batched step key, at most <see cref="_maxBatchGraphs"/>, the least recently used
        /// dropped first.</summary>
        private readonly Dictionary<string, (IntPtr[] Execs, int[] Slots, LinkedListNode<string> Node)> _batchGraphs =
            new Dictionary<string, (IntPtr[], int[], LinkedListNode<string>)>(StringComparer.Ordinal);
        private readonly LinkedList<string> _batchGraphLru = new LinkedList<string>();
        /// <summary>Row counts whose capture failed: their steps run uncaptured from then on.</summary>
        private readonly HashSet<int> _batchCaptureFailed = new HashSet<int>();
        private readonly int _maxBatchGraphs = Math.Max(1, EnvInt("TS_Q4E_BATCH_GRAPHS_MAX", 64));
        // Uncaptured runs of each key not captured yet (BatchCaptureAfter).
        private readonly Dictionary<string, int> _batchKeyRuns = new Dictionary<string, int>(StringComparer.Ordinal);

        /// <summary>A batched step's key is captured on its third run.</summary>
        private const int BatchCaptureAfter = 2;

        /// <summary>Captured batched steps this engine holds (tests, diagnostics).</summary>
        internal int CapturedBatchGraphs => _batchGraphs.Count;

        /// <summary>Captured batched steps this engine holds (the batched decode steps' tests and telemetry name).</summary>
        internal int CapturedBatchedDecodeGraphs => _batchGraphs.Count;

        /// <summary>Whether a batched step of <paramref name="nt"/> rows may run through a captured graph: decode-size rows,
        /// and missed experts (if any) left to the host rather than staged.</summary>
        private bool CanCaptureBatch(int nt)
            => _batchGraphsEnabled && CanCaptureDecode && nt <= PerSlotMaxRows && !_batchCaptureFailed.Contains(nt)
               && (_expertStage == null || (_hostServer != null && nt <= _hostExpertRows));

        /// <summary>One batched decode step (the rows' states already staged) through its captured graphs; the logits of every
        /// row in <paramref name="logitsOut"/>. False when it runs uncaptured (the caller runs it).</summary>
        private bool ForwardBatchedDecodeGraphed(Slot[] rows, int[] slotIds, int[] tokens, int[] positions, float[] logitsOut)
        {
            if (!ForwardBatchGraphed(rows, tokens, positions, 1, greedy: false, hidden: false))
                return false;
            FinishHead(rows.Length, logitsOut, true, null);
            return true;
        }

        /// <summary>
        /// A batched step's rows (<paramref name="perSeq"/> consecutive rows a sequence) through its captured graphs; the
        /// head's results wait in pinned memory for FinishHead. False when the step runs uncaptured (the caller runs it):
        /// graphs off, a key not yet on its third run, or a failed capture (the sequences' PLE histories have then taken the
        /// step's tokens already, which the uncaptured step writes again at the same positions).
        /// </summary>
        private bool ForwardBatchGraphed(Slot[] rows, int[] tokens, int[] positions, int perSeq, bool greedy, bool hidden)
        {
            int nt = tokens.Length;
            if (!CanCaptureBatch(nt) || perSeq < 1 || nt % perSeq != 0)
                return false;
            var key = new StringBuilder(16 + 8 * (nt / perSeq));
            key.Append(perSeq).Append(greedy ? 'g' : 'l').Append(hidden ? 'h' : '-');
            var slots = new int[nt / perSeq];
            for (int r = 0, i = 0; r < nt; r += perSeq, i++)
            {
                slots[i] = rows[r].Id;
                key.Append(':').Append(rows[r].Id).Append('@').Append(DecodeRegime(positions[r] + perSeq - 1));
            }
            string k = key.ToString();
            bool captured = _batchGraphs.TryGetValue(k, out var entry);
            if (!captured)
            {
                _batchKeyRuns.TryGetValue(k, out int runs);
                if (runs < BatchCaptureAfter)
                {
                    _batchKeyRuns[k] = runs + 1;
                    return false;
                }
                _batchKeyRuns.Remove(k);
            }
            // The pinned inputs are rewritten below: every device's previous work must have run.
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            Marshal.Copy(tokens, 0, _pinnedTokens, nt);
            for (int t = 0; t < nt; t++)
                ((int*)_posPinned)[t] = positions[t];
            if (_pleLayer >= 0)
                GatherPleRows(null, tokens, 0, nt, 0, rows, positions);

            IntPtr[] execs;
            if (captured)
            {
                execs = entry.Execs;
                _batchGraphLru.Remove(entry.Node);
                _batchGraphLru.AddLast(entry.Node);
            }
            else
            {
                try
                {
                    execs = CaptureSteps(d => IssueBatchStep(d, rows, positions, nt, greedy, hidden));
                    System.Threading.Interlocked.Increment(ref _tBatchedCaptures);
                }
                catch (Exception ex)
                {
                    System.Threading.Interlocked.Increment(ref _tCaptureFailures);
                    _batchCaptureFailed.Add(nt);
                    Console.Error.WriteLine($"[q4e-cuda] batched steps of {nt} rows run uncaptured from here on: the capture failed ({ex.Message})");
                    return false;
                }
                while (_batchGraphs.Count >= _maxBatchGraphs && _batchGraphLru.First != null)
                    RemoveBatchGraph(_batchGraphLru.First.Value);
                _batchGraphs[k] = (execs, slots, _batchGraphLru.AddLast(k));
            }
            LaunchSteps(execs);
            System.Threading.Interlocked.Increment(ref _tBatchedLaunches);
            return true;
        }

        /// <summary>One device's part of a batched step, as captured: the rows' positions (and on device 0 their tokens) from
        /// pinned memory, the streams from the previous device's boundary buffer, its layers (each sequence's attention with
        /// its rows' device positions), then the boundary copy or, on the last device, the head for every row.</summary>
        private void IssueBatchStep(Dev dev, Slot[] rows, int[] positions, int nt, bool greedy, bool hidden)
        {
            var m = _m;
            var boundary = new UIntPtr((ulong)((long)nt * HcDim * 4));
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
                    ready = RunLayer(dev, il, null, nt, 0, rows, positions, Ptr(dev.PosDev), ready);
            if (dev.Ordinal < _lastDev)
                CudaDriverApi.cuMemcpyDtoHAsync(dev.BoundaryPinned, Ptr(dev.Xs), boundary, dev.Stream).ThrowOnError();
            else
                IssueHead(nt, logits: true, allRows: true, hidden: hidden, greedy: greedy);
        }

        private void RemoveBatchGraph(string key)
        {
            if (!_batchGraphs.TryGetValue(key, out var entry))
                return;
            DestroyGraphs(entry.Execs);
            _batchGraphLru.Remove(entry.Node);
            _batchGraphs.Remove(key);
        }

        /// <summary>Drop the captured batched steps that hold <paramref name="slotId"/> (and forget its keys' runs); every one
        /// when it is -1 (with the run counts).</summary>
        private void DestroyBatchGraphs(int slotId)
        {
            if (slotId < 0)
                _batchKeyRuns.Clear();
            else
            {
                string id = ":" + slotId + "@";
                List<string> stale = null;
                foreach (var k in _batchKeyRuns.Keys)
                    if (k.Contains(id, StringComparison.Ordinal))
                        (stale ??= new List<string>()).Add(k);
                if (stale != null)
                    foreach (var k in stale)
                        _batchKeyRuns.Remove(k);
            }
            List<string> keys = null;
            foreach (var (key, graph) in _batchGraphs)
                if (slotId < 0 || Array.IndexOf(graph.Slots, slotId) >= 0)
                    (keys ??= new List<string>()).Add(key);
            if (keys == null)
                return;
            foreach (var key in keys)
                RemoveBatchGraph(key);
        }

        /// <summary>
        /// Give captured graphs' device memory to a slot that needs it (they are a cache, captured again on use): the batched
        /// steps first, then the solo decode and verify graphs of the slots other than the active one (whose next step needs
        /// its own). Once the streams have run (a graph still running is freed when it completes). False when there was
        /// nothing to give. With 4 lanes on the RTX 5090 the plan left ~0.4 GiB beside the lanes, and the graphs captured over
        /// a few rounds took it: the fourth of four requests waited 2.4-2.6 s for a running one to finish.
        /// </summary>
        private bool ReleaseBatchGraphsForMemory()
        {
            if (_batchGraphs.Count > 0)
                DestroyBatchGraphs(-1);
            else
            {
                List<(int, int, int)> others = null;
                foreach (var key in _graphs.Keys)
                    if (key.Slot != _active.Id)
                        (others ??= new List<(int, int, int)>()).Add(key);
                if (others == null)
                    return false;
                foreach (var key in others)
                {
                    DestroyGraphs(_graphs[key]);
                    _graphs.Remove(key);
                }
            }
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            return true;
        }
    }
}
