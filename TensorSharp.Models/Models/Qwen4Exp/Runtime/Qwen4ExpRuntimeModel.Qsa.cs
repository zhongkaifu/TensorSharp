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
    /// QSA, Qwen 3.8's sparse attention layers (llama.cpp src/models/qwen4exp.cpp, build_qsa_sel). A block is
    /// <c>ratio</c> consecutive cells; its pooled key is the mean of its members' raw indexer keys
    /// (<c>indexer.k_proj · x</c>), RMS-normed and rotated to its first cell. A row of the slot's extra key cache is
    /// [raw key | pooled key]: every step writes its rows' raw keys and pools the blocks it completes into their last
    /// members' rows (llama.cpp's kpool), dense or not, so the selection reads each block's pooled key instead of pooling
    /// the window again. A query row scores every block it can see with the lightning indexer (the heads' rectified dot
    /// products, summed) and attends the cells of its top <c>top_k / ratio</c> blocks and of its own incomplete tail
    /// block, under the causal mask. While a sequence holds no more cells than that selection keeps, it attends densely,
    /// which is the same. Blocks span the attention window (padded past the cells, invisible), so a decode graph
    /// replays within a window bucket. Rows at text positions only: a position table (media) is refused once the
    /// selection is sparse.
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

        /// <summary>Rows past the budget attend at most the budget's cells (the top blocks and the tail), so the backend
        /// may attend only those: ggml-metal's sparse flash attention for decode and verify rows, TensorSharp's kernel
        /// (tsm_qsa_attn) for a prompt chunk over a wide window.</summary>
        protected override int AttentionKvMax(GraphStep step, int layer, int s)
            => QsaRatio(layer) > 0 && step.Starts[s] + step.Counts[s] > Ple.QsaBudget ? (int)Ple.QsaBudget : 0;

        /// <summary>A QSA layer's slots keep each cell's [raw indexer key | its block's pooled key, in the block's last
        /// cell].</summary>
        protected override int ExtraKeyCacheWidth(int layer) => QsaRatio(layer) > 0 ? 2 * Qsa.Dim : 0;

        /// <summary>The cache of <paramref name="layer"/> for sequence <paramref name="s"/>, [2d, capacity]: the step's
        /// rank's copy (every rank keeps the whole indexer cache and selects the same blocks).</summary>
        private int QsaCache(GraphStep step, int layer, int s)
        {
            Slot slot = step.Slots[s];
            return step.State(ExtraKeyCache(slot.K, layer, Math.Max(step.Rank, 0)), KvCacheGraphType, 2 * Qsa.Dim, slot.Capacity, 1);
        }

        /// <summary>Every row's raw indexer key into its sequence's cache, and the blocks the step completes pooled into
        /// their last members' rows, dense or not: a selecting step reads every block's pooled key.</summary>
        protected override void EmitAttentionInput(GraphStep step, int layer, int x)
        {
            int ratio = QsaRatio(layer);
            if (ratio <= 0) return;
            int d = Qsa.Dim, kvType = KvCacheGraphType;
            const long f = sizeof(float);
            int raw = step.MatMul(LayerName(layer, "indexer.k_proj.weight"), x);   // [d, rows]
            for (int s = 0; s < step.Sequences; s++)
            {
                int count = step.Counts[s], start = step.Starts[s];
                Slot slot = step.Slots[s];
                int cache = QsaCache(step, layer, s);
                int rows = step.Sequences == 1 ? raw : step.View2(raw, d, count, d * f, step.First[s] * d * f);
                // [raw | 0]: a row's pooled half is written once its block is complete (llama.cpp's packed rows).
                int packed = step.Op(GraphOp.Concat, stackalloc[] { step.Cont(rows), step.Op(GraphOp.Scale, stackalloc[] { step.Cont(rows) }, default, stackalloc[] { 0f }) },
                    stackalloc long[] { 0 });
                step.Effect(step.Op(GraphOp.SetRows, cache, step.Reshape(packed, 2 * d, count, 1), step.WriteRows(start, count)));

                // The blocks to pool: those the step completes; a decode or verify step pools its last ceil(count /
                // ratio) complete blocks (again where they were complete before: the same values), so its graph is the
                // same at every position.
                int end = (start + count) / ratio;
                int first = count <= SmallChunkRows ? end - (count + ratio - 1) / ratio : (start + ratio) / ratio - 1;
                first = Math.Max(first, 0);
                if (end <= first) continue;
                int n = end - first;
                var members = new int[ratio * n];
                var lastRows = new int[n];
                var positions = new int[n];
                for (int b = 0; b < n; b++)
                {
                    int block = first + b;
                    positions[b] = block * ratio;
                    lastRows[b] = block * ratio + ratio - 1;
                    for (int i = 0; i < ratio; i++) members[b * ratio + i] = block * ratio + i;
                }
                // The members' raw keys (the cache's raw half, rows of 2d), read after this step's write.
                int rawHalf = step.View2(cache, d, slot.Capacity, RowBytes(kvType, 2 * d), 0);
                int gathered = step.Op(GraphOp.GetRows, rawHalf, step.InputI32(members, members.Length));   // [d, ratio * n]
                int sum = -1;
                for (int i = 0; i < ratio; i++)
                {
                    int member = step.View2(gathered, d, n, ratio * d * f, i * d * f);
                    sum = sum < 0 ? step.Cont(member) : step.Op(GraphOp.Add, sum, member);
                }
                int pooled = step.Op(GraphOp.Scale, stackalloc[] { sum }, default, stackalloc[] { 1f / ratio });
                pooled = step.RmsNorm(pooled, LayerName(layer, "indexer.k_norm.weight"));
                pooled = step.RopeAt(step.Reshape(pooled, d, 1, n), GraphRotary, step.InputI32(positions, n));
                // [the last member's raw key | the pooled key] into the last member's row.
                int lastRaw = step.Cont(step.View2(gathered, d, n, ratio * d * f, (ratio - 1) * d * f));
                int row = step.Op(GraphOp.Concat, stackalloc[] { lastRaw, step.Reshape(pooled, d, n) }, stackalloc long[] { 0 });
                step.Effect(step.Op(GraphOp.SetRows, cache, step.Reshape(row, 2 * d, n, 1), step.InputRows(lastRows)));
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
            var pooledRows = new int[pools];              // the row holding a block's pooled key (padded blocks read row 0)
            var cells = new int[ratio * pools];           // the members a picked block keeps (padded: out of range)
            for (int b = 0; b < pools; b++)
            {
                pooledRows[b] = b < complete ? b * ratio + ratio - 1 : 0;
                for (int i = 0; i < ratio; i++)
                    cells[b * ratio + i] = b < complete ? b * ratio + i : keys;
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

            // Each block's pooled key, from its last member's row (the cache's pooled half).
            int pooledHalf = step.View2(QsaCache(step, layer, s), d, slot.Capacity, RowBytes(kvType, 2 * d), RowBytes(kvType, d));
            int pooled = step.Reshape(step.Op(GraphOp.GetRows, pooledHalf, step.InputI32(pooledRows, pools)), d, 1, pools);

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
