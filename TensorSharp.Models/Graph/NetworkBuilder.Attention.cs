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
    public sealed partial class NetworkBuilder
    {
        /// <summary>
        /// Attention as products (llama.cpp mtmd build_attn without flash attention): scores = k^T q per head (query head h reads
        /// KV head h / (heads / kvHeads)), softmax(scores * <paramref name="scale"/> + mask) over the keys, then the values'
        /// product, in F32. <paramref name="q"/> [d, heads, rows], <paramref name="k"/> [d, kvHeads, keys], <paramref name="v"/>
        /// [dv, kvHeads, keys], <paramref name="mask"/> F16 [keys, rows] or none. Returns [dv * heads, rows].
        /// </summary>
        public NetValue AttentionByProducts(NetValue q, NetValue k, NetValue v, float scale, NetValue mask = default)
        {
            long d = q.Ne0, heads = q.Ne1, rows = q.Ne2, kvHeads = k.Ne1, keys = k.Ne2, dv = v.Ne0;
            if (k.Ne0 != d || v.Ne1 != kvHeads || v.Ne2 != keys || kvHeads < 1 || heads % kvHeads != 0 || q.Ne3 * k.Ne3 * v.Ne3 != 1)
                throw new ArgumentException($"attention of q {q} over k {k}, v {v}");
            if (mask.Valid && (mask.Ne0 != keys || mask.Ne1 != rows || mask.Format != GraphFormat.F16))
                throw new ArgumentException($"attention mask {mask}: F16 [keys {keys}, rows {rows}]");
            // Executors whose products read strided views (ggml-metal, ggml-cuda) take q and k as permuted views, as llama.cpp does.
            NetValue qp = Packed(Permute(q, 0, 2, 1, 3));                           // [d, rows, heads]
            NetValue kp = Packed(Permute(k, 0, 2, 1, 3));                           // [d, keys, kvHeads]
            NetValue scores = Emit(GraphOp.MulMat, kp, qp, keys, rows, heads, 1);   // [keys, rows, heads]
            Span<int> srcs = stackalloc[] { scores.Id, mask.Valid ? mask.Id : 0 };
            NetValue p = new(_b.Op(GraphOp.SoftMaxExt, mask.Valid ? srcs : srcs[..1], default, stackalloc[] { scale, 0f }), keys, rows, heads, 1,
                false, GraphFormat.F32);
            NetValue vt = Cont(Permute(v, 1, 2, 0, 3));                             // [keys, dv, kvHeads]
            NetValue o = Emit(GraphOp.MulMat, vt, p, dv, rows, heads, 1);            // [dv, rows, heads]
            return Reshape(Cont(Permute(o, 0, 2, 1, 3)), dv * heads, rows);
        }
    }
}
