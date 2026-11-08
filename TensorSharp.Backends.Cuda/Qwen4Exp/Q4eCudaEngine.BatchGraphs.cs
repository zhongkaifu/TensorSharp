// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Captured batched steps: several slots' rows in one forward (a batched decode step, one row a sequence; a batched
// verify, a window a sequence). Uncaptured, such a step issued every layer's launches from the host, about as long as
// a solo verify window's (12.6 ms on the RTX 5090 before its graph existed), and a 4-row batched decode step ran at
// that rate rather than the GPU's.
//
// The captured step is the solo verify's (Graphs.cs) with the rows of each sequence where they sit: the tokens and every
// row's position come from pinned memory, attention reads a sequence's caches with its rows' device positions, the
// recurrent layers read each row's states through the batch state table (uploaded before each launch), and the PLE rows
// are gathered on the host before it. What the host decides per call is in the key: the rows a sequence has, the head's
// form (first-max tokens or logits, with or without hidden streams), and each sequence's slot and regime (the QSA layers
// whose selection runs at its rows' positions). A key's graph stays valid until one of its slots is freed.
//
// Only steps that keep their missed experts on the host are captured (a staging step plans its copies on the host), and
// a key only once it has run uncaptured twice: concurrent requests come and go, and a capture costs more than the steps
// of a set of sequences that lasts a few steps. A capture that fails turns batched captures off for the engine with a
// note; solo graphs are unaffected.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private readonly Dictionary<string, (IntPtr[] Execs, int[] Slots)> _batchGraphs =
            new Dictionary<string, (IntPtr[] Execs, int[] Slots)>();
        private bool _batchGraphsFailed;
        // Uncaptured runs of each key not captured yet (BatchCaptureAfter).
        private readonly Dictionary<string, int> _batchKeyRuns = new Dictionary<string, int>();

        /// <summary>A batched step's key is captured on its third run.</summary>
        private const int BatchCaptureAfter = 2;

        /// <summary>Captured batched steps kept at most: a new capture past it drops the others first (each holds device
        /// memory, and a slot that cannot be allocated drops them all too, SlotAlloc).</summary>
        private const int MaxBatchGraphs = 8;

        // TS_Q4E_BATCH_GRAPHS=0: batched steps run uncaptured (for measuring the graphs).
        private readonly bool _batchGraphsEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_BATCH_GRAPHS"), "0", StringComparison.Ordinal);

        /// <summary>Captured batched steps this engine holds (tests, diagnostics).</summary>
        internal int CapturedBatchGraphs => _batchGraphs.Count;

        /// <summary>Whether a batched step of <paramref name="nt"/> rows may run through a captured graph: one GPU, decode-size
        /// rows, and missed experts (if any) left to the host rather than staged.</summary>
        private bool CanCaptureBatch(int nt)
            => _batchGraphsEnabled && !_batchGraphsFailed && CanCaptureDecode && _devs.Length == 1 && nt <= PerSlotMaxRows
               && (_expertStage == null || (_hostServer != null && nt <= _hostExpertRows));

        /// <summary>
        /// A batched step's rows (<paramref name="perSeq"/> consecutive rows a sequence) through its captured graph, capturing
        /// it on the key's third run; the head's results wait in pinned memory for FinishHead. False when the step runs
        /// uncaptured (the caller runs it); after a failed capture the sequences' PLE histories have taken the step's tokens
        /// already, which the uncaptured step repeats at the same positions.
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
            bool captured = _batchGraphs.TryGetValue(k, out var graph);
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
            var dev = _devs[0];
            dev.MakeCurrent();
            // The pinned inputs are rewritten below: the previous step's work must have run.
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            Marshal.Copy(tokens, 0, _pinnedTokens, nt);
            for (int t = 0; t < nt; t++)
                ((int*)_posPinned)[t] = positions[t];
            if (_pleLayer >= 0)
                GatherPleRows(null, tokens, 0, nt, 0, rows, positions);
            if (!captured)
            {
                if (_batchGraphs.Count >= MaxBatchGraphs)
                    DestroyBatchGraphs(-2);
                try
                {
                    graph = (CaptureSteps(d => IssueBatchStep(d, rows, positions, nt, greedy, hidden)), slots);
                }
                catch (Exception ex)
                {
                    _batchGraphsFailed = true;
                    Console.Error.WriteLine($"[q4e-cuda] batched steps run uncaptured from here on: a capture failed ({ex.Message})");
                    return false;
                }
                _batchGraphs[k] = graph;
            }
            LaunchSteps(graph.Execs);
            return true;
        }

        /// <summary>A batched step as captured: the tokens and positions from pinned memory, every layer over the rows (each
        /// sequence's attention with its rows' device positions), then the head for every row.</summary>
        private void IssueBatchStep(Dev dev, Slot[] rows, int[] positions, int nt, bool greedy, bool hidden)
        {
            var m = _m;
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.PosDev), _posPinned, new UIntPtr((ulong)nt * 4), dev.Stream).ThrowOnError();
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Tokens), _pinnedTokens, new UIntPtr((ulong)nt * 4), dev.Stream).ThrowOnError();
            dev.DK.Embed(_tokEmbd.Ptr, dev.Tokens, dev.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, nt, m.NEmbd, dev.Stream);
            bool ready = false;
            for (int il = 0; il < m.NLayer; il++)
                ready = RunLayer(dev, il, null, nt, 0, rows, positions, Ptr(dev.PosDev), ready);
            IssueHead(nt, logits: true, allRows: true, hidden: hidden, greedy: greedy);
        }

        /// <summary>Drop the captured batched steps that hold <paramref name="slotId"/>: every one when it is -1 (with the run
        /// counts) or -2 (the counts kept).</summary>
        private void DestroyBatchGraphs(int slotId)
        {
            if (slotId == -1)
                _batchKeyRuns.Clear();
            else if (slotId >= 0)
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
            {
                DestroyGraphs(_batchGraphs[key].Execs);
                _batchGraphs.Remove(key);
            }
        }
    }
}
