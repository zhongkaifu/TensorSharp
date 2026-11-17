// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// LFM3 in whole-forward graphs: per-head Q/K RMSNorm then the layer's rotary, the head-wise output gate, the short
    /// convolutions over each sequence's slot window (<see cref="StandardDecoderModel.EmitShortConvMixer"/>), with the runtime's
    /// sliding-window attention and sandwich norms. Every per-position input is a standard one (positions, write rows, decode
    /// and sliding-window masks), so decode steps advance in place instead of tracing again.
    /// </summary>
    public partial class Lfm3RuntimeModel
    {
        /// <summary>A non-null rotary (the graph path's precondition); each layer picks its own in <see cref="EmitQueryKey"/>.</summary>
        protected override RotarySpec GraphRotary => _ropeSliding;

        protected override bool RecurrentMixerLowers => true;

        /// <summary>Decode steps run under ggml-metal's concurrency reorder, as llama.cpp's scheduler always runs them: the
        /// attention block's Q/K/V products and the gate product beside the flash attention encode concurrently. LFM3-1.7B
        /// Q4_K_M on an M5 Pro, interleaved: tg128 188.5 -> 192.0 tok/s at depth 0, 180.9 -> 184.6 at 4K (llama.cpp 188.3 /
        /// 180.8).</summary>
        protected override bool GraphConcurrencyReorder => true;

        /// <summary>The short convolutions trace one rank's channels (its in/out projection shards, filter slice and states).</summary>
        protected override bool RanksLower => true;

        /// <summary>The short convolutions bind their windows as state leaves and add no inputs; attention follows positions
        /// through the step's standard inputs, its sliding-window masks included.</summary>
        protected override bool DecodeStepsAdvance => true;

        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            RotarySpec spec = RopeOf(layer);
            q = step.Rope(step.RmsNorm(q, LayerName(layer, "attn_q_norm.weight")), spec);
            k = step.Rope(step.RmsNorm(k, LayerName(layer, "attn_k_norm.weight")), spec);
        }

        /// <summary>attn [dv * heads, rows] times sigmoid(attn_gate · x) [heads, rows], one gate per head broadcast over its
        /// dims (the rank's heads: attn_gate is sharded by head blocks).</summary>
        protected override int EmitAttentionOutput(GraphStep step, int layer, int x, int attn)
        {
            int heads = Config.NumHeads / step.Degree, dv = ValueDim, rows = step.Rows;
            int gate = step.Op(GraphOp.Sigmoid, step.MatMul(LayerName(layer, "attn_gate.weight"), x));   // [heads, rows]
            int gated = step.Op(GraphOp.Mul, step.Reshape(attn, dv, heads, rows), step.Reshape(gate, 1, heads, rows));
            return step.Reshape(gated, (long)dv * heads, rows);
        }

        protected override int EmitRecurrentMixer(GraphStep step, int layer, int normed)
        {
            int channels = Config.HiddenSize / step.Degree;
            int kernel = step.Rank >= 0 && Mesh.TensorParallel > 1
                ? step.TensorWeight(_convKernelRank[layer][step.RankIndex], _convTaps, channels)
                : step.Weight(ConvWeight(layer));
            return EmitShortConvMixer(step, layer, normed, InProj(layer), kernel, OutProj(layer), channels, _convTaps - 1);
        }
    }
}
