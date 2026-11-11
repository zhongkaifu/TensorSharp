// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Globalization;
using System.Linq;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// QSA, Qwen 3.8's sparse attention layers (llama.cpp src/models/qwen4exp.cpp, build_qsa_sel). Every token's raw
    /// indexer key (<c>indexer.k_proj · x</c>) is kept in the slot's extra key cache. A block is <c>ratio</c> consecutive
    /// cells; its pooled key is the mean of its members' raw keys, RMS-normed and rotated to its first cell. A query
    /// row scores every block it can see with the lightning indexer (the heads' rectified dot products, summed) and
    /// attends the cells of its top <c>top_k / ratio</c> blocks and of its own incomplete tail block, under the causal
    /// mask. While a sequence holds no more cells than that selection keeps, it attends densely, which is the same.
    /// Blocks span the attention window (padded past the cells, invisible), so a decode graph replays within a window
    /// bucket. Rows at text positions only: a position table (media) is refused once the selection is sparse.
    /// </summary>
    public sealed partial class Qwen4ExpRuntimeModel
    {
        private sealed class QsaSpec
        {
            public readonly int Heads, Dim, TopK;

            public QsaSpec(GgufFile gguf, ModelConfig config)
            {
                string a = config.Architecture;
                Heads = (int)gguf.GetUint32($"{a}.attention.indexer.head_count", 0);
                Dim = (int)gguf.GetUint32($"{a}.attention.indexer.key_length", 0);
                TopK = (int)gguf.GetUint32($"{a}.attention.indexer.top_k", 0);
            }
        }

        private QsaSpec _qsa;
        private QsaSpec Qsa => _qsa ??= new QsaSpec(_gguf, Config);

        private int QsaRatio(int layer) => layer < Ple.CompressRatios.Length ? Ple.CompressRatios[layer] : 0;

        private bool? _anyQsa;

        /// <summary>Each sequence's attention, dense (within the budget) or selecting: the two emit different graphs.</summary>
        protected override string EmissionVariant(int[] counts, int[] starts)
        {
            _anyQsa ??= Enumerable.Range(0, Config.NumLayers).Any(l => QsaRatio(l) > 0);
            if (_anyQsa != true) return null;
            var modes = new char[counts.Length];
            for (int s = 0; s < counts.Length; s++) modes[s] = starts[s] + counts[s] > Ple.QsaBudget ? 's' : 'd';
            return new string(modes);
        }

        /// <summary>A QSA layer's slots keep each cell's raw indexer key.</summary>
        protected override int ExtraKeyCacheWidth(int layer) => QsaRatio(layer) > 0 ? Qsa.Dim : 0;

        /// <summary>Every row's raw indexer key into its sequence's cache, dense or not: the selection pools them once a
        /// sequence outgrows the dense budget.</summary>
        protected override void EmitAttentionInput(GraphStep step, int layer, int x)
        {
            if (QsaRatio(layer) <= 0) return;
            int d = Qsa.Dim, kvType = KvCacheGraphType;
            int raw = step.MatMul(LayerName(layer, "indexer.k_proj.weight"), x);   // [d, rows]
            for (int s = 0; s < step.Sequences; s++)
            {
                int count = step.Counts[s];
                Slot slot = step.Slots[s];
                int cache = step.State(ExtraKeyCache(slot.K, layer), kvType, d, slot.Capacity, 1);
                int rows = step.Sequences == 1 ? raw : step.View2(raw, d, count, d * 4L, step.First[s] * d * 4L);
                step.Effect(step.Op(GraphOp.SetRows, cache, step.Reshape(step.Cont(rows), d, count, 1), step.WriteRows(step.Starts[s], count)));
            }
        }

        protected override int EmitAttentionMask(GraphStep step, int layer, int x, int s, int mask, int keys)
        {
            int ratio = QsaRatio(layer);
            int start = step.Starts[s], count = step.Counts[s];
            if (ratio <= 0) return mask;
            if (start + count <= Ple.QsaBudget)
            {
                step.LimitAdvance(start, (int)Math.Min(int.MaxValue, Ple.QsaBudget - count));   // past it, a step selects
                return mask;
            }
            if (step.PositionTable >= 0 || step.Slots[s].RopeDelta != 0)
                throw new NotSupportedException(string.Create(CultureInfo.InvariantCulture,
                    $"{Config.Architecture}: QSA's block selection over media positions is not ported ({start + count} cells, dense up to {Ple.QsaBudget})."));
            QsaSpec q = Qsa;
            int d = q.Dim, heads = q.Heads, kvType = KvCacheGraphType;
            Slot slot = step.Slots[s];

            // The plan: blocks over the window, complete ones holding cells, the rest padded and invisible.
            int pools = (keys + ratio - 1) / ratio, complete = (start + count) / ratio;
            var gather = new int[ratio * pools];          // the members a block's key pools (padded blocks read cell 0)
            var cells = new int[ratio * pools];           // the members a picked block keeps (padded: out of range)
            var poolPositions = new int[pools];
            for (int b = 0; b < pools; b++)
            {
                poolPositions[b] = b * ratio;
                for (int i = 0; i < ratio; i++)
                {
                    int c = b * ratio + i;
                    gather[b * ratio + i] = b < complete ? c : 0;
                    cells[b * ratio + i] = b < complete ? c : keys;
                }
            }
            var visible = new ushort[pools * count];      // 0 where row q sees all of block b, -inf elsewhere (F16)
            var tail = new int[(ratio - 1) * count];      // row q's incomplete block's cells (out of range past them)
            for (int r = 0; r < count; r++)
            {
                int p = start + r, tailStart = (p + 1) / ratio * ratio;
                for (int b = 0; b < pools; b++)
                    visible[r * pools + b] = b < complete && (b + 1) * ratio - 1 <= p ? (ushort)0 : (ushort)0xFC00;
                for (int t = 0; t < ratio - 1; t++)
                    tail[r * (ratio - 1) + t] = tailStart + t <= p ? tailStart + t : keys;
            }

            // Each block's pooled key: the mean of its members, normed, rotated to its first cell.
            int cache = step.State(ExtraKeyCache(slot.K, layer), kvType, d, slot.Capacity, 1);
            int rowsOf = step.View2(cache, d, slot.Capacity, RowBytes(kvType, d), 0);
            int members = step.Op(GraphOp.GetRows, rowsOf, step.InputI32(gather, gather.Length));   // [d, ratio * pools]
            int sum = -1;
            for (int i = 0; i < ratio; i++)
            {
                int member = step.View2(members, d, pools, ratio * d * 4L, i * d * 4L);
                sum = sum < 0 ? step.Cont(member) : step.Op(GraphOp.Add, sum, member);
            }
            int pooled = step.Op(GraphOp.Scale, stackalloc[] { sum }, default, stackalloc[] { 1f / ratio });
            pooled = step.RmsNorm(pooled, LayerName(layer, "indexer.k_norm.weight"));
            pooled = step.RopeAt(step.Reshape(pooled, d, 1, pools), GraphRotary, step.InputI32(poolPositions, pools));

            // The rows' indexer queries, scored against every block.
            int xs = step.Sequences == 1 ? x : step.View2(x, Config.HiddenSize, count, Config.HiddenSize * 4L, step.First[s] * Config.HiddenSize * 4L);
            int query = step.MatMul(LayerName(layer, "indexer.q_proj.weight"), step.Cont(xs));                // [d * heads, count]
            query = step.RmsNorm(step.Reshape(query, d, heads, count), LayerName(layer, "indexer.q_norm.weight"));
            query = step.RopeAt(query, GraphRotary, step.SequencePositions(s));
            var weights = new float[heads * count];
            Array.Fill(weights, 1f / MathF.Sqrt(d));   // the reference's unweighted head sum, scaled
            int score = step.Op(GraphOp.LightningIndexer, stackalloc[]
                { query, pooled, step.InputF32(weights, heads, count), step.InputF16(visible, pools, count) });   // [pools, count]
            int picks = step.Op(GraphOp.TopK, stackalloc[] { score }, stackalloc long[] { Math.Min(pools, q.TopK / ratio) });
            return step.Op(GraphOp.TopKMask, stackalloc[]
            {
                mask, picks, score, step.Reshape(step.InputI32(cells, cells.Length), ratio, pools),
                step.Reshape(step.InputI32(tail, tail.Length), ratio - 1, count),
            }, stackalloc long[] { 1, 1, 1 });
        }
    }
}
