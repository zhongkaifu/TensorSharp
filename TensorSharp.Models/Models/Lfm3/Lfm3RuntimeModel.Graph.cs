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

        /// <summary>A prompt step's last layer runs its feed-forward on the rows the head reads only, as llama.cpp's LFM3 graph
        /// does (the stream cut to the output rows after the last mixer). LFM3-1.7B on ggml_metal, M5 Pro, interleaved: pp512
        /// 5,710 -> 5,872 tok/s on Q4_K_M and 5,919 -> 6,086 on Q8_0, pp128 +2%; Q8_0 teacher-forced against llama.cpp Metal
        /// stays 0/320 top-1 mismatches (median relL2 3.10e-4; the prompt rows closer: 1.01e-3 max on the long prompt, was
        /// 1.06e-3).</summary>
        protected override bool PrunesLastLayerRows => true;

        /// <summary>On ggml-cuda, Q4_K and Q8_0 feed-forwards keep ffn_gate and ffn_up apart: ggml-cuda then runs the two
        /// products and the SwiGLU as one MMQ (prompt) or MMVQ (decode) kernel with the GLU in its write-back, as llama.cpp's
        /// LFM3 graph has it, where the fused weight's product of twice the rows was followed by a SwiGLU pass. LFM3-1.7B on an
        /// A40, interleaved (fused -> apart): Q4_K_M pp512 15,455 -> 15,849, pp4096 14,039 -> 14,764 tok/s, tg128 at 4K 328.8
        /// -> 334.3; Q8_0 pp512 16,515 -> 18,139, pp4096 16,048 -> 16,883, tg128 238.0 -> 243.3. The fused weight stays where it
        /// measured faster: QAD-Q4_0 (pp512 -2.4%, pp4096 -5.0% apart), BF16 (-16.6%, -15.0%; its prompt products take cuBLAS,
        /// which has no GLU write-back) and NVFP4 (decode -2.7% apart).</summary>
        protected override bool KeepsGateUpApart(int gateType, int upType)
            => _backend == BackendType.GgmlCuda && gateType == upType && gateType is 8 /* Q8_0 */ or 12 /* Q4_K */;

        /// <summary>A plain prompt's 512-row steps overlap on the device: the short convolutions' windows and the caches stay on
        /// the device between steps, so nothing between them reads the device back.</summary>
        protected override bool OverlapsPromptSteps => true;

        /// <summary>On ggml-vulkan a prompt chunk attends the padded F16 cache window, not F32 copies of the cached prefix.
        /// LFM3-1.7B Q4_K_M on an A40, interleaved: pp4096 10,144 -> 10,693 tok/s, pp16384 8,265 -> 9,419, pp32768 6,576 ->
        /// 7,861, decode unchanged (llama.cpp's Vulkan build: 11,014 / 10,767 / 9,778).</summary>
        protected override bool PadsPromptWindowsOnVulkan => true;

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
