// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Graph;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// Products whose weight matrix is picked per row at run time (GraphOp.MulMatId): a stacked table's matrices chosen by
    /// indices that ride the run, so one program serves every choice (one codebook's logits of a per-codebook head, the
    /// codebooks picked per call).
    /// </summary>
    public sealed partial class NetworkBuilder
    {
        /// <summary>
        /// Product weight <paramref name="name"/> [width, experts * rows] as <paramref name="experts"/> stacked matrices
        /// [width, rows, experts] for <see cref="MulMatId"/> (<see cref="NetworkWeights.ExpertStack"/>); an invalid value
        /// where the weight is held dense (no block format on this backend: read it with
        /// <see cref="Weight(string, long[])"/> as [width, rows, experts] and multiply every matrix).
        /// </summary>
        public NetValue Experts(string name, long experts)
        {
            string key = $"{name}|experts{experts}";
            if (_weightLeaves.TryGetValue(key, out NetValue v)) return v;
            IGraphExpertWeights set = Weights.ExpertStack(name, experts);
            if (set == null) return default;
            // GGML binds the whole table's host bytes as the 3-D operand; the tensor executors run its members.
            int id = _b.Leaf(GraphLeafRole.Weight, set.Format, stackalloc long[] { set.Ne0, set.Ne1, set.Ne2 }, set.Data, set.Bytes, null, set);
            v = new NetValue(id, set.Ne0, set.Ne1, set.Ne2, 1, false, set.Format);
            _weightLeaves[key] = v;
            return v;
        }

        /// <summary>
        /// ggml_mul_mat_id: row t of <paramref name="x"/> [width, k or 1, rows] times matrix ids[j, t] of
        /// <paramref name="experts"/> [width, out, n] (<see cref="Experts"/>) for each of the k slots j of the row;
        /// <paramref name="ids"/> is I32 [k, rows]. Returns [out, k, rows].
        /// </summary>
        public NetValue MulMatId(NetValue experts, NetValue x, NetValue ids)
        {
            if (!experts.Valid || experts.Ne2 < 1) throw new ArgumentException("mul_mat_id over a value that is not stacked experts");
            if (ids.Format != GraphFormat.I32 || ids.Ne2 * ids.Ne3 != 1) throw new ArgumentException($"mul_mat_id ids {ids}: I32 [k, rows]");
            long k = ids.Ne0, rows = ids.Ne1;
            if (x.Ne0 != experts.Ne0 || x.Ne2 != rows || (x.Ne1 != 1 && x.Ne1 != k) || x.Ne3 != 1)
                throw new ArgumentException($"mul_mat_id of {experts} with {x} by ids {ids}");
            return Emit(GraphOp.MulMatId, experts, Cont(x), ids, experts.Ne1, k, rows, 1);
        }

        private NetValue Emit(GraphOp op, NetValue a, NetValue b, NetValue c, long ne0, long ne1, long ne2, long ne3)
            => new(_b.Op(op, stackalloc[] { a.Id, b.Id, c.Id }), ne0, ne1, ne2, ne3, false, GraphFormat.F32);
    }
}
