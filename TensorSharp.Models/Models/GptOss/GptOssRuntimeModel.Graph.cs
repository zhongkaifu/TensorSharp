// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// GPT-OSS in whole-forward graphs. Attention is the runtime's (its biased projections, sinks and sliding-window rings);
    /// the family emits the rotary and the routed experts, in llama.cpp's op order (openai-moe-iswa.cpp, build_moe_ffn with
    /// SOFTMAX_WEIGHT gating and LLM_FFN_SWIGLU_OAI_MOE), the chain ggml-cuda's fusions (top-k MoE, mul_mat_id + add_id + GLU,
    /// the experts' weighted sum) match.
    /// </summary>
    public sealed partial class GptOssRuntimeModel
    {
        protected override RotarySpec GraphRotary => _rotary;

        /// <summary>Attention follows positions through the step's standard inputs, its sliding-window masks included, and the
        /// emission has no side effects of its own: decode steps advance in place.</summary>
        protected override bool DecodeStepsAdvance => true;

        /// <summary>A plain prompt's steps overlap on the device: no recurrent state, and the caches stay on the device between
        /// steps.</summary>
        protected override bool OverlapsPromptSteps => true;

        /// <summary>A prompt step's last layer runs its experts on the rows the head reads only (llama.cpp's graph, and the legacy
        /// model, narrowed the last layer's rows before its MoE): the experts are sized by <see cref="GraphStep.FeedForwardRows"/>.
        /// Measured on an A40 (ggml_cuda, gpt-oss-20b MXFP4, pp512): the last layer's three expert products over 511 rows cost
        /// 2.5 ms of a 85 ms prompt.</summary>
        protected override bool PrunesLastLayerRows => true;

        protected override bool PrunesExpertLastLayer => true;

        /// <summary>ggml-cuda's fusions get their allocation dependencies: the experts' weighted sum (moe_weighted_reduction) ran
        /// fused in every decode step but in 1 of 24 layers of a prompt step without them, the rest a broadcast multiply and three
        /// adds (A40, pp512: 5.8 ms of separate multiplies and 7 ms more adds over three prompts).</summary>
        protected override bool GgmlCudaBackendFusions => true;

        /// <summary>The routed experts over the normed rows <paramref name="x"/> [hidden, rows].</summary>
        protected override int EmitFeedForward(GraphStep step, int layer, int x)
        {
            // The last layer of a pruned prompt step reads its sequences' last rows only.
            int rows = step.FeedForwardRows, hidden = Config.HiddenSize, k = _expertsUsed, experts = _experts;
            bool allRows = rows == step.Rows;
            string gateStack = LayerName(layer, "ffn_gate_exps.weight"), upStack = LayerName(layer, "ffn_up_exps.weight");
            string downStack = LayerName(layer, "ffn_down_exps.weight");

            // The router: logits plus their bias, the top k of the logits, the softmax over those k (SOFTMAX_WEIGHT).
            int logits = step.Op(GraphOp.Add, step.MatMul(LayerName(layer, "ffn_gate_inp.weight"), x),
                step.Weight(LayerName(layer, "ffn_gate_inp.bias")));                                                 // [E, rows]
            bool host = IsExpertLayerOffloaded(layer) && GraphExecutor.Capabilities.HostExperts;
            // The router and the device's experts are one region an executor may run on its MoE engine (direct CUDA's CudaMoe:
            // the routes stay on the device, so decode steps record); the others run its ops.
            bool region = !host && ForcedRoutesForTests == null && DeviceRoutesForTests == null;
            bool downBias = step.Rank <= 0;
            if (region)
                step.BeginFused(GraphFusion.OaiRoutedExperts, stackalloc[]
                {
                    x, logits, step.Experts(gateStack), step.Experts(upStack), step.Experts(downStack),
                    step.Weight(LayerName(layer, "ffn_gate_exps.bias")), step.Weight(LayerName(layer, "ffn_up_exps.bias")),
                    step.Weight(LayerName(layer, "ffn_down_exps.bias")),
                }, stackalloc long[] { k, downBias ? 1 : 0 }, stackalloc[] { SwigluAlpha, SwigluLimit });
            int forced = allRows ? EmitForcedRoutes(step, layer, k, experts) : -1;                                    // tests
            int selected = forced >= 0 ? forced : step.Op(GraphOp.TopK, stackalloc[] { logits }, stackalloc long[] { k }); // [K, rows]
            if (allRows) EmitDeviceRoutes(step, layer, selected, logits, k, experts);                                 // tests
            int weights = step.Op(GraphOp.GetRows, step.Reshape(logits, 1, experts, rows), selected);                // [1, K, rows]
            weights = step.Reshape(step.Op(GraphOp.SoftMaxExt, step.Reshape(weights, k, rows)), 1, k, rows);

            if (host)
            {
                // --n-cpu-moe / --cpu-moe: the experts stay in host memory and the program pauses for them (GGML's host MoE
                // segment; the tensor executors' expert host): the biased gate and up products, the clamped SwiGLU, the biased
                // down product, the weighted sum.
                return step.Op(GraphOp.MoeHostExperts, stackalloc[]
                {
                    x, selected, weights, step.Experts(gateStack), step.Experts(upStack), step.Experts(downStack),
                    step.Weight(LayerName(layer, "ffn_gate_exps.bias")), step.Weight(LayerName(layer, "ffn_up_exps.bias")),
                    step.Weight(LayerName(layer, "ffn_down_exps.bias")),
                }, stackalloc long[] { layer, 1 /* the clamped SwiGLU */, 0, 0, 1 /* srcs[6..8]: the experts' biases */ },
                stackalloc[] { SwigluAlpha, SwigluLimit });
            }

            int input = step.Reshape(x, hidden, 1, rows);
            int up = step.Op(GraphOp.MulMatId, step.Experts(upStack), input, selected);                              // [ff, K, rows]
            up = step.Op(GraphOp.AddId, up, step.Weight(LayerName(layer, "ffn_up_exps.bias")), selected);
            int gate = step.Op(GraphOp.MulMatId, step.Experts(gateStack), input, selected);
            gate = step.Op(GraphOp.AddId, gate, step.Weight(LayerName(layer, "ffn_gate_exps.bias")), selected);
            int act = step.Op(GraphOp.SwigluOai, stackalloc[] { gate, up }, default, stackalloc[] { SwigluAlpha, SwigluLimit });
            int down = step.Op(GraphOp.MulMatId, step.Experts(downStack), act, selected);                            // [hidden, K, rows]
            // The down bias once: rank 0 of per-rank graphs, whose partials the all-reduce sums.
            if (downBias)
                down = step.Op(GraphOp.AddId, down, step.Weight(LayerName(layer, "ffn_down_exps.bias")), selected);
            down = step.Op(GraphOp.Mul, down, weights);
            long rowBytes = hidden * 4L;
            int output = -1;
            for (int j = 0; j < k; j++)
            {
                int slice = step.View2(down, hidden, rows, rowBytes * k, j * rowBytes);
                output = output < 0 ? slice : step.Op(GraphOp.Add, output, slice);
            }
            output = step.Cont(output);
            if (region) step.EndFused(stackalloc[] { output });
            return output;
        }
    }
}
