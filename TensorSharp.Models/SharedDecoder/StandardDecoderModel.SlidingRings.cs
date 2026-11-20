// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading;

namespace TensorSharp.Models
{
    /// <summary>
    /// Ring caches for sliding-window attention layers, once for every family (<see cref="SlidingWindow"/>): such a layer's K/V
    /// keep R cells, not one per position. A position p lives in cell p mod R; R = W + <see cref="RingStepRows"/> (aligned)
    /// cells hold the W - 1 positions before a step's first row and every row a step writes for its sequence, which is what
    /// llama.cpp sizes its SWA cache to (n_swa + n_ubatch, padded to 256). LFM3-1.7B's five 256-key layers then keep 768
    /// cells a slot (7.5 MB in F16) and its K/V cost the six full layers' 12,288 bytes a token, where linear caches cost
    /// 22,528.
    /// <list type="bullet">
    /// <item>Until a slot's positions reach R the ring is a linear cache (cell = position), so every short request runs the
    /// linear code unchanged; a ring is never larger than its slot.</item>
    /// <item>A step whose rows reach R reads the whole ring under a mask built from the positions each cell holds (graph
    /// path), or a linear copy of its window (per-op path); a prompt chunk's rows past what the ring holds are split into
    /// steps of at most <see cref="RingStepRows"/> rows a sequence (<see cref="MaxSequenceStepRows"/>).</item>
    /// <item>A verify window rolls back exactly: R &gt;= W + its rows - 1 keeps every kept row's window intact. Copies
    /// (slot growth, checkpoints, prefix clones, checkpoint files) move a ring whole, cells in place.</item>
    /// <item>A truncation keeps the positions the next row's window reads: refused (CanTruncateKVCache, TryTruncateKVCacheCore) when
    /// the ring has overwritten them.</item>
    /// </list>
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>The most rows of one sequence a step writes into a ring (beyond the window): llama.cpp's default ubatch.</summary>
        internal const int RingStepRows = 512;

        /// <summary>
        /// The ring step where the placement's per-rank sums cross the host (<see cref="RankSumsCrossHost"/>): each prompt step
        /// pays a host round trip for each of its reductions, so a 2048-row step (legacy's prompt chunk, and the runtime's
        /// <c>TS_PREFILL_CHUNK</c> default where no ring caps the step) crosses a quarter as often as a 512-row one. Measured
        /// on 2x RTX PRO 4500, Muse-Glimmer UD-IQ2_XXS, ggml-vulkan <c>--tp 2</c>: prefill 0.82x / 0.88x legacy's at 4K / 16K
        /// with 512-row steps, 1.00x / 1.03x with 2048, decode unchanged.
        /// </summary>
        internal const int HostStagedRingStepRows = 2048;

        /// <summary>
        /// The rows of one sequence a step writes into a ring for this placement; the rings hold the window plus that many
        /// cells (aligned). <see cref="RingStepRows"/>, but <see cref="HostStagedRingStepRows"/> when per-rank graphs sum
        /// their partials through the host. The placement-chosen seam for the step width (design D1 / G4): a family or
        /// backend that measures another width declares it here. <see cref="RingGeometryForTests"/> overrides it.
        /// </summary>
        protected virtual int RingStepRowsForPlacement => _rankSumsCrossHost ? HostStagedRingStepRows : RingStepRows;

        /// <summary>The rows a step writes into a ring, as the rings were built (tests, probes).</summary>
        internal int RingStepRowsInUse { get; private set; } = RingStepRows;

        /// <summary>Ring capacities are multiples of this (ggml-cuda groups a KV head's queries only over 256-multiple K; Metal's
        /// 64-cell decode windows divide it).</summary>
        internal const int RingAlignment = 256;

        /// <summary>Tests run small rings (rows a step, alignment) on synthetic models: set before the model is constructed.</summary>
        internal static readonly AsyncLocal<(int StepRows, int Alignment)?> RingGeometryForTests = new();

        // Per decoder layer: 0 for a linear cache, else its ring's cells. Draft layers keep linear caches.
        private int[] _ringCells = Array.Empty<int>();

        // The most rows of one sequence a step may carry where rings exist (int.MaxValue without rings).
        private int _maxSequenceStepRows = int.MaxValue;

        /// <summary>Rows of one sequence a step may carry: a ring layer's R cells hold the W - 1 positions before the step and
        /// its rows. Forwards with more are split into steps of this many rows (int.MaxValue without ring caches).</summary>
        internal int MaxSequenceStepRows => _maxSequenceStepRows;

        /// <summary>The ring cells of a cache entry (0: linear).</summary>
        private int RingOf(int entry) => entry >= 0 && entry < _ringCells.Length ? _ringCells[entry] : 0;

        /// <summary>The cells entry <paramref name="entry"/>'s cache tensor has in a slot of <paramref name="capacity"/>
        /// cells: its ring's, or the slot's while that is smaller.</summary>
        private int CacheCells(int entry, int capacity) => RingOf(entry) is > 0 and var ring && ring < capacity ? ring : capacity;

        /// <summary>Whether a step of <paramref name="count"/> rows from <paramref name="start"/> reaches a ring's cells past
        /// its first lap (cell != position).</summary>
        private bool RingWrapped(int entry, int start, int count) => RingOf(entry) is > 0 and var ring && start + count > ring;

        /// <summary>Ring cells of the sliding-window layers (tests, diagnostics); 0 for linear layers.</summary>
        internal int RingCellsForTests(int layer) => RingOf(layer);

        /// <summary>The active cache's K/V bytes (tests, probes): those that grow with the context, per token, and the rings'
        /// fixed bytes at their full size.</summary>
        internal (long BytesPerToken, long RingBytes) KvCacheFootprint()
        {
            long perToken = 0, rings = 0;
            foreach (Tensor[][] caches in new[] { _cacheK, _cacheV })
                for (int e = 0; caches != null && e < caches.Length; e++)
                    if (caches[e] != null)
                        foreach (Tensor t in caches[e])
                        {
                            if (t == null) continue;
                            long perCell = t.Storage.ByteLength / Math.Max(1, t.Sizes[1]);
                            if (RingOf(e) is > 0 and var ring) rings += perCell * ring;
                            else perToken += perCell;
                        }
            return (perToken, rings);
        }

        /// <summary>A streamed step's kind term: which rings each sequence's rows reach past their first lap (a wrapped ring
        /// emits its own write cells, the whole ring and its mask), so a step of the same rows on the other side of a wrap does
        /// not run on its template. Empty without rings.</summary>
        private string RingWrapVariant(int[] counts, int[] starts)
        {
            if (!HasRingCaches) return "";
            var text = new System.Text.StringBuilder("|rings");
            for (int s = 0; s < counts.Length; s++)
            {
                int wrapped = 0;
                for (int l = 0; l < _ringCells.Length; l++)
                    if (_ringCells[l] > 0 && starts[s] + counts[s] > _ringCells[l]) wrapped++;
                text.Append(':').Append(wrapped);
            }
            return text.ToString();
        }

        private void InitRingCaches()
        {
            (int stepRows, int alignment) = RingGeometryForTests.Value ?? (RingStepRowsForPlacement, RingAlignment);
            RingStepRowsInUse = stepRows;
            _ringCells = new int[TotalLayers];
            _maxSequenceStepRows = int.MaxValue;
            int window = 0, rings = 0, cells = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                int w = SlidingWindow(l);
                // Every layer of the model, owned here or not: a multi-node pipeline's steps are bounded by every node's rings.
                if (w <= 0 || _recurrentLayer[l]) continue;
                int ring = (int)((w + (long)stepRows + alignment - 1) / alignment * alignment);
                // A context the ring holds whole never wraps: the layer stays linear.
                if (ring >= _maxContextLength) continue;
                _ringCells[l] = ring;
                _maxSequenceStepRows = Math.Min(_maxSequenceStepRows, ring - w + 1);
                window = Math.Max(window, w);
                cells = Math.Max(cells, ring);
                rings++;
            }
            if (rings > 0)
                Console.WriteLine($"  Sliding-window caches: {rings} layers keep rings of {cells} cells (window {window} + {stepRows} rows a step" +
                    (RingGeometryForTests.Value == null && _rankSumsCrossHost ? ", wide: the ranks' sums cross the host" : "") + "); " +
                    $"prompt steps carry at most {Math.Min(_maxSequenceStepRows, stepRows)} rows a sequence.");
            if (_maxSequenceStepRows != int.MaxValue) _maxSequenceStepRows = Math.Min(_maxSequenceStepRows, stepRows);
        }

        /// <summary>The cells holding positions [<paramref name="from"/>, <paramref name="to"/>) of a ring of
        /// <paramref name="ring"/> cells, as at most two runs (cell, rows) in position order.</summary>
        private static (int Cell, int Rows, int Cell2, int Rows2) RingRuns(int ring, int from, int to)
        {
            int rows = to - from;
            if (rows <= 0) return (0, 0, 0, 0);
            if (rows > ring) throw new InvalidOperationException($"{rows} positions do not fit a ring of {ring} cells.");
            int cell = from % ring;
            int first = Math.Min(rows, ring - cell);
            return (cell, first, 0, rows - first);
        }

        // ---------------------------------------------------------------- steps of at most MaxSequenceStepRows rows

        /// <summary>
        /// A forward of more rows than a ring holds beside its window, as consecutive steps of
        /// <see cref="MaxSequenceStepRows"/> rows (each <paramref name="run"/>(piece, wants its logits)): queued media rows go
        /// with the piece that holds them, a staged position table is consumed piece by piece (BeginRotaryPositions), and a
        /// speculating request's follow of its prompt holds for every piece. Returns the last piece's result.
        /// </summary>
        private float[] InRingSteps(int[] tokens, bool wantLogits, Func<int[], bool, float[]> run, bool overlapSteps = false)
        {
            int step = _maxSequenceStepRows;
            var media = _pendingMedia.ToArray();
            _pendingMedia.Clear();
            bool follow = _followPlainPrompt;
            float[] result = null;
            // A plain prompt's pieces overlap on the device (IGraphExecutor.OverlapSteps): each piece's run returns with its work
            // queued while the next is traced and built. The caches hold the whole prompt first, so no piece grows them.
            IDisposable overlap = null;
            if (overlapSteps && OverlapPromptSteps && !follow && media.Length == 0 && GraphExecutor != null && _mesh.Stages == 1 && !_rankGraphs && _graphTpRanks == null)
            {
                EnsureCacheCapacity(checked(_cacheSeqLen + tokens.Length));
                overlap = GraphExecutor.OverlapSteps();
            }
            try
            {
                for (int at = 0; at < tokens.Length; at += step)
                {
                    int n = Math.Min(step, tokens.Length - at);
                    foreach (var (rows, offset) in media)
                    {
                        int from = Math.Max(offset, at), to = Math.Min(offset + (int)rows.Sizes[0], at + n);
                        if (from < to) _pendingMedia.Add((rows.Narrow(0, from - offset, to - from), from - at));
                    }
                    _followPlainPrompt = follow;
                    result = run(tokens.AsSpan(at, n).ToArray(), wantLogits && at + n == tokens.Length);
                }
            }
            finally
            {
                overlap?.Dispose();
                foreach (var (rows, _) in media) rows.Dispose();
            }
            return result;
        }

        // ---------------------------------------------------------------- truncation

        /// <summary>Whether a ring sequence of <paramref name="cached"/> rows truncated to <paramref name="target"/> keeps every
        /// position the next row's window reads (positions (target - W, target)).</summary>
        private bool RingsKeepTruncation(int cached, int target)
        {
            if (target == 0 || target == cached) return true;
            for (int l = 0; l < _ringCells.Length; l++)
            {
                if (_ringCells[l] == 0) continue;
                int needed = Math.Max(0, target - SlidingWindow(l) + 1);
                if (needed < cached - _ringCells[l]) return false;
            }
            return true;
        }

        // ---------------------------------------------------------------- per-op attention over a wrapped ring

        /// <summary>
        /// The per-op attention of <paramref name="count"/> rows from <paramref name="start"/> on a sliding-window layer whose
        /// ring the rows reach past its first lap: the rows' K/V are written to their cells, then the window
        /// [start - W + 1, start + count) is copied out of the ring in position order (F32) and the linear kernels attend it as
        /// a cache of its own (positions shifted by its first one; the band is relative, so the masks are the same).
        /// </summary>
        private Tensor RingAttention(RankScope scope, Tensor cacheK, Tensor cacheV, int ring, Tensor q, Tensor k, Tensor v, int start, int count,
            int band)
        {
            int kvHeads = scope.LocalKvHeads;
            WriteRingRows(cacheK, k, kvHeads, _keyDim, ring, start, count);
            WriteRingRows(cacheV, v, kvHeads, _valueDim, ring, start, count);
            k.Dispose();
            v.Dispose();
            int end = start + count, lo = Math.Max(0, start - band + 1);
            int floor = _attentionFloor;
            using Tensor windowK = RingWindow(cacheK, ring, lo, end);
            using Tensor windowV = RingWindow(cacheV, ring, lo, end);
            // The rows' own K/V as the cache rounded them: the linear kernels write them into the window again.
            Tensor freshK = WindowRows(windowK, start - lo, count, kvHeads, _keyDim);
            Tensor freshV = WindowRows(windowV, start - lo, count, kvHeads, _valueDim);
            _attentionFloor = Math.Max(0, floor - lo);
            try
            {
                return count == 1
                    ? DecodeAttention(scope, windowK, windowV, end - lo, q, freshK, freshV, start - lo, band)
                    : PrefillAttention(scope, windowK, windowV, q, freshK, freshV, start - lo, count, band);
            }
            finally
            {
                _attentionFloor = floor;
            }
        }

        /// <summary>Rows [<paramref name="first"/>, + <paramref name="count"/>) of a window [heads, rows, dim] as the flat
        /// [count, heads * dim] a projection gives.</summary>
        private Tensor WindowRows(Tensor window, int first, int count, int heads, int dim)
        {
            using Tensor rows = window.Narrow(1, first, count);
            using Tensor dense = Ops.NewContiguous(rows);
            return ReshapeFromHeads(dense, heads, count, dim);
        }

        /// <summary>Write the flat rows <paramref name="flat"/> [count, heads * dim] of positions [start, start + count) into
        /// their ring cells (at most two runs).</summary>
        private void WriteRingRows(Tensor cache, Tensor flat, int heads, int dim, int ring, int start, int count)
        {
            using Tensor byHead = ReshapeToHeads(flat, heads, count, dim);
            (int cell, int rows, int cell2, int rows2) = RingRuns(ring, start, start + count);
            WriteRun(cell, 0, rows);
            if (rows2 > 0) WriteRun(cell2, rows, rows2);

            void WriteRun(int at, int from, int n)
            {
                if (from == 0 && n == count)
                {
                    CopyToCache(cache, byHead, at, n);
                    return;
                }
                using Tensor part = byHead.Narrow(1, from, n);
                using Tensor dense = Ops.NewContiguous(part);
                CopyToCache(cache, dense, at, n);
            }
        }

        /// <summary>Positions [<paramref name="from"/>, <paramref name="to"/>) of a ring cache, in order, as an F32
        /// [heads, to - from, dim] tensor beside it.</summary>
        private Tensor RingWindow(Tensor cache, int ring, int from, int to)
        {
            int heads = (int)cache.Sizes[0], dim = (int)cache.Sizes[2];
            using Tensor all = ExpandKVHeads(cache, 1, (int)cache.Sizes[1]);
            var window = new Tensor(all.Storage.Allocator, DType.Float32, heads, to - from, dim);
            (int cell, int rows, int cell2, int rows2) = RingRuns(ring, from, to);
            Copy(cell, 0, rows);
            if (rows2 > 0) Copy(cell2, rows, rows2);
            return window;

            void Copy(int at, int into, int n)
            {
                using Tensor src = all.Narrow(1, at, n);
                using Tensor dst = window.Narrow(1, into, n);
                Ops.Copy(dst, src);
            }
        }
    }
}
