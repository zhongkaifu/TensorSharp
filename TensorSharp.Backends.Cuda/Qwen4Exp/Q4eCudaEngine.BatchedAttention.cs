// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A batched step's attention layers in one launch per kernel. The rows of a batched decode step (or batched verify) belong
// to different sequences, each with its own K/V and QSA caches, so a layer's attention was issued a sequence at a time:
// the prep, the QSA raw-key store and pooling, and the split attention for every sequence, about four launches a row and
// layer, ~50 a row a step on Qwen3.8-Flash-Next's 12 attention layers. Here each of those kernels runs once for the whole
// step, every row reading its own sequence's cache pointers from a table passed by value (Q4eKernels.RowPtrs). A block
// computes what the one-sequence kernel's block computed for the same row (same position, cells and split count), so
// every row's answer is unchanged, bit for bit. The QSA selection, past the indexer's width, still runs a sequence at a
// time (it is a handful of launches only for sequences that far in). Streamed layers (TS_Q4E_KV_RESIDENT: paged K/V,
// raw keys in a ring) keep the one-sequence path. TS_Q4E_BATCHED_ATTN=0 turns it off.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private readonly bool _batchedAttn = Environment.GetEnvironmentVariable("TS_Q4E_BATCHED_ATTN") != "0";

        /// <summary>Batched steps' attention layers that ran as one launch per kernel (tests prove the path engaged).</summary>
        internal long BatchedAttentionLayers { get; private set; }

        /// <summary>Whether layer <paramref name="il"/> of a batched step of <paramref name="nt"/> rows runs its attention in
        /// one launch per kernel: the kernels loaded, device positions, the split attention, at most a table's rows, and
        /// every row's layer resident (no paged K/V, no raw-key ring).</summary>
        private bool BatchedAttentionFor(Dev dev, DevLayer L, int il, Slot[] rows, int nt, IntPtr pos)
        {
            if (!_batchedAttn || !dev.QK.HasBatchedAttention || pos == IntPtr.Zero || nt > Q4eKernels.MaxTableRows
                || nt <= _batchRowsPerSeq || RowsInFlight(nt) != nt || AttnSplits(dev, il, _batchRowsPerSeq, 0) <= 0)
                return false;
            for (int r = 0; r < nt; r += _batchRowsPerSeq)
            {
                var c = rows[r].Layers[il];
                if (c.Kv != null || (L.Ratio > 0 && c.IdxRing > 0))
                    return false;
            }
            return true;
        }

        /// <summary>The attention of every row of a batched step (rows [0, nt), <see cref="_batchRowsPerSeq"/> a sequence):
        /// the projections, then the prep, the QSA store and pooling and the split attention each in one launch over all
        /// rows, then the output projection.</summary>
        private void AttentionRows(Dev dev, DevLayer L, int il, Slot[] rows, int nt, int[] positions, IntPtr pos, IntPtr curQ)
        {
            var m = _m;
            int qd = m.NHead * m.HeadDim, perSeq = _batchRowsPerSeq;
            IntPtr cur = curQ != IntPtr.Zero ? curQ : QuantizeShared(dev, dev.Cur, m.NEmbd, nt);
            MatMul(dev, L.Wq, dev.Cur, cur, dev.Qg, nt);
            MatMul(dev, L.Wk, dev.Cur, cur, dev.K, nt);
            MatMul(dev, L.Wv, dev.Cur, cur, dev.V, nt);
            if (L.Ratio > 0)
                MatMulF32(dev, L.IdxK, dev.Cur, dev.IdxRaw, m.NEmbd, m.IdxDim, nt);
            StageEnd(dev, StAttnProj);

            var first = rows[0].Layers[il];
            bool q8 = first.KScale != null;
            Q4eKernels.RowPtrs k = default, v = default, ks = default, vs = default, raw = default, pooled = default;
            for (int t = 0; t < nt; t++)
            {
                var c = rows[t - t % perSeq].Layers[il];
                k.P[t] = (ulong)Ptr(c.KCache);
                v.P[t] = (ulong)Ptr(c.VCache);
                if (q8)
                {
                    ks.P[t] = (ulong)Ptr(c.KScale);
                    vs.P[t] = (ulong)Ptr(c.VScale);
                }
                if (L.Ratio > 0)
                {
                    raw.P[t] = (ulong)Ptr(c.IdxRaw);
                    pooled.P[t] = (ulong)Ptr(c.IdxPooled);
                }
            }
            dev.QK.AttnPrepRows(Ptr(dev.Qg), Ptr(dev.K), Ptr(dev.V), Ptr(L.QNorm), Ptr(L.KNorm), Ptr(dev.Q), Ptr(dev.Gate),
                k, v, pos, m.NHead, m.NKvHead, m.NRot, m.RopeBase, m.RopeFreqScale, m.RmsEps, nt, dev.Stream, q8, ks, vs);

            IntPtr cells = IntPtr.Zero, cellCnt = IntPtr.Zero;
            if (L.Ratio > 0)
            {
                int R = L.Ratio, d = m.IdxDim;
                dev.QK.QsaStoreRows(Ptr(dev.IdxRaw), raw, pos, d, nt, dev.Stream);
                dev.QK.QsaPoolRows(raw, Ptr(L.IdxKNorm), pooled, pos, R, d, m.NRot, m.RopeBase, m.RopeFreqScale, m.RmsEps, nt,
                    dev.Stream);
                // Past the width the selection bites, a sequence at a time as the one-sequence path runs it; the rows of
                // the sequences short of it read every cell (a negative count).
                bool any = false;
                for (int r = 0; r < nt; r += perSeq)
                    if (positions[r] + perSeq > m.IdxTopK + R - 1)
                    {
                        if (!any)
                        {
                            CudaDriverApi.cuMemsetD32Async(Ptr(dev.CellCnt), unchecked((uint)-1), new UIntPtr((ulong)nt), dev.Stream)
                                .ThrowOnError();
                            any = true;
                        }
                        var c = rows[r].Layers[il];
                        IntPtr rowPos = (IntPtr)((long)pos + 4L * r);
                        IntPtr qIdx = At(dev.IdxQ, (long)r * m.IdxHeads * d);
                        using (Tensor rowsIn = Block(dev.Cur, r, perSeq, m.NEmbd))
                        using (Tensor rowsOut = Block(dev.IdxQ, r, perSeq, (long)m.IdxHeads * d))
                            MatMulF32(dev, L.IdxQ, rowsIn, rowsOut, m.NEmbd, m.IdxHeads * d, perSeq);
                        dev.QK.QsaQuery(qIdx, Ptr(L.IdxQNorm), rowPos, positions[r], m.IdxHeads, d, m.NRot, m.RopeBase,
                            m.RopeFreqScale, m.RmsEps, perSeq, dev.Stream);
                        IntPtr scores = At(dev.Scores, (long)r * ScoreStride);
                        IntPtr rowCells = (IntPtr)((long)Ptr(dev.Cells) + (long)r * CellStride * 4);
                        IntPtr rowCnt = (IntPtr)((long)Ptr(dev.CellCnt) + (long)r * 4);
                        dev.QK.QsaScores(qIdx, Ptr(c.IdxPooled), scores, rowPos, positions[r], m.IdxHeads, d, R, ScoreStride,
                            ScoreStride, perSeq, dev.Stream);
                        dev.QK.QsaSelect(scores, rowCells, rowCnt, rowPos, positions[r], R, m.IdxTopK, ScoreStride, CellStride,
                            perSeq, dev.Stream);
                    }
                if (any)
                {
                    cells = Ptr(dev.Cells);
                    cellCnt = Ptr(dev.CellCnt);
                }
            }
            StageEnd(dev, StAttnPrep);

            dev.QK.AttentionSplitRows(Ptr(dev.Q), k, v, Ptr(dev.Gate), cells, cellCnt, Ptr(dev.AttnO), pos, m.NHead, m.NKvHead,
                m.AttnScale, CellStride, nt, AttnSplits(dev, il, perSeq, 0), Ptr(dev.AttnPart), Ptr(dev.AttnArrive), dev.Stream,
                q8, ks, vs);
            StageEnd(dev, StAttnCore);
            BatchedAttentionLayers++;

            MatMulQ(dev, L.Wo, dev.AttnO, dev.BlockOut, nt);
            StageEnd(dev, StAttnOut);
        }
    }
}
