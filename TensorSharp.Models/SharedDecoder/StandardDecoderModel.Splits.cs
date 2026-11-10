// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models
{
    /// <summary>
    /// How a leaf of a tensor-parallel graph spreads over the ranks (ggml_backend_meta_split_state):
    /// along <see cref="Axis"/>, <see cref="Ne"/> holding each (segment, rank)'s extent, segment s
    /// repeating <see cref="Nr"/>[s] times. A fused projection is one segment per part, so every rank
    /// gets a slice of each part.
    /// </summary>
    internal sealed record LeafSplit(int Axis, long[] Ne, uint[] Nr);

    /// <summary>
    /// Tensor parallelism on whole-forward graphs: the graph keeps global shapes and ggml's meta
    /// device runs it over the ranks, so the runtime only says how each weight and state spreads.
    /// The rules follow Megatron's: attention and FFN inputs are split by output rows (heads,
    /// width), their outputs by input columns (partial sums the meta device reduces), routed
    /// experts along their FFN width, the head by vocabulary rows, K/V caches by KV heads; the rest
    /// (norms, routers, the embedding) is mirrored.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>The split of weight <paramref name="name"/> [ne] under tensor-parallel graphs, or
        /// null (mirrored, or no tensor parallelism). A family overrides it for its own blocks.</summary>
        private protected LeafSplit SplitOf(string name, long[] ne, int ggmlType)
            => _graphTpRanks == null ? null : GraphWeightSplit(name, ne, ggmlType);

        private protected virtual LeafSplit GraphWeightSplit(string name, long[] ne, int ggmlType)
        {
            int dot = name.LastIndexOf(".weight", StringComparison.Ordinal);
            string stem = dot > 0 ? name[..dot] : name;
            string part = stem.Contains('.') ? stem[(stem.LastIndexOf('.') + 1)..] : stem;
            int heads = Config.NumHeads, kvHeads = Config.NumKVHeads;
            long qStride = GatedAttentionOutput ? 2L * _keyDim : _keyDim;
            switch (part)
            {
                case "attn_q": return Segments(1, (heads * qStride, heads));
                case "attn_k": return Segments(1, ((long)kvHeads * _keyDim, kvHeads));
                case "attn_v": return Segments(1, ((long)kvHeads * _valueDim, kvHeads));
                case "attn_qkv":
                    return Segments(1, (heads * qStride, heads), ((long)kvHeads * _keyDim, kvHeads), ((long)kvHeads * _valueDim, kvHeads));
                case "attn_output": return Segments(0, ((long)heads * _valueDim, heads), block: BlockOf(ggmlType));
                case "ffn_gate":
                case "ffn_up":
                case "ffn_gate_shexp":
                case "ffn_up_shexp":
                    return Even(1, ne[1]);
                case "ffn_gate_up":
                case "ffn_gate_up_shexp":
                    return Segments(1, (ne[1] / 2, 0), (ne[1] / 2, 0));
                case "ffn_down":
                case "ffn_down_shexp":
                    return Even(0, ne[0], BlockOf(ggmlType));
                case "ffn_gate_exps":
                case "ffn_up_exps":
                    return Even(1, ne[1]);
                case "ffn_gate_up_exps":
                    return Segments(1, (ne[1] / 2, 0), (ne[1] / 2, 0));
                case "ffn_down_exps":
                    return Even(0, ne[0], BlockOf(ggmlType));
                case "output":
                    return Even(1, ne[1]);
                case "token_embd":
                    // The tied head: the embedding rows themselves are looked up on the host.
                    return Even(1, ne[1]);
                default:
                    return null;
            }
        }

        /// <summary>K/V caches [d, capacity, KV heads] split by KV heads.</summary>
        private protected LeafSplit KvCacheSplit()
            => _graphTpRanks == null ? null : Segments(2, (Config.NumKVHeads, Config.NumKVHeads));

        /// <summary>Elements of a quantized row block (a split along axis 0 must keep whole blocks).</summary>
        private static long BlockOf(int ggmlType) => ggmlType switch
        {
            0 or 1 or 30 => 1,                    // F32, F16, BF16
            2 or 3 or 6 or 7 or 8 or 20 => 32,    // Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL
            _ => 256,                             // K-quants and i-quants
        };

        /// <summary>An even split of <paramref name="extent"/> along <paramref name="axis"/> into whole
        /// <paramref name="block"/>s per rank (the last ranks take the remainder blocks).</summary>
        private LeafSplit Even(int axis, long extent, long block = 1)
        {
            int n = _graphTpRanks.Length;
            if (extent % block != 0) throw new NotSupportedException($"{Config.Architecture}: extent {extent} is not whole blocks of {block}.");
            long blocks = extent / block;
            var ne = new long[n];
            for (int r = 0; r < n; r++) ne[r] = (blocks * (r + 1) / n - blocks * r / n) * block;
            return new LeafSplit(axis, ne, new uint[] { 1 });
        }

        /// <summary>One segment per (extent, units) part, each split by whole units (heads; 0: evenly).</summary>
        private LeafSplit Segments(int axis, params (long Extent, long Units)[] parts) => Segments(axis, 1, parts);

        private LeafSplit Segments(int axis, (long Extent, long Units) part, long block) => Segments(axis, block, part);

        private LeafSplit Segments(int axis, long block, params (long Extent, long Units)[] parts)
        {
            int n = _graphTpRanks.Length;
            var ne = new long[parts.Length * n];
            for (int s = 0; s < parts.Length; s++)
            {
                (long extent, long units) = parts[s];
                if (units <= 0) units = extent / block;
                if (units % n != 0 || extent % units != 0)
                    throw new NotSupportedException($"{Config.Architecture}: {extent} elements in {units} units do not split over {n} ranks.");
                long unit = extent / units;
                if (unit * (units / n) % block != 0)
                    throw new NotSupportedException($"{Config.Architecture}: a rank's slice of {extent} is not whole blocks of {block}.");
                for (int r = 0; r < n; r++) ne[s * n + r] = unit * (units / n);
            }
            var nr = new uint[parts.Length];
            Array.Fill(nr, 1u);
            return new LeafSplit(axis, ne, nr);
        }
    }
}
