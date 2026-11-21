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
    /// Muse-Glimmer in whole-forward graphs, node for node as the legacy model's fused native graph (deleted in stage S9) emitted each
    /// layer: the weightless input RMSNorm, per-head Q/K RMSNorm then the NORM rotary on the sliding-window layers, the
    /// elementwise output gate from the attention input, and the runtime's sliding-window attention, sandwich norms (at
    /// <see cref="MuseGlimmerRuntimeModel.PostNormEpsilon"/>), SwiGLU feed-forward and logit transform. Every per-position
    /// input is a standard one (positions, write rows, decode and sliding-window masks), so decode steps advance in place.
    /// The measured choices below start at LFM3's (the closest family: sliding-window rings, sandwich norms, QK norms);
    /// stage S7 measures each on Muse-Glimmer.
    /// </summary>
    internal sealed partial class MuseGlimmerRuntimeModel
    {
        /// <summary>The NORM rotary of the sliding-window layers; the full layers emit none (<see cref="EmitQueryKey"/>).</summary>
        protected override RotarySpec GraphRotary => _rope;

        /// <summary>Decode steps (the replayable graphs) run under ggml-metal's concurrency reorder, as legacy's persistent
        /// decode graph did (graph_optimize on its persist builds only). Measured on an M5 Pro (stage S7, ABBA, one binary):
        /// without it UD-IQ2_XXS decodes 1.4-1.5% slower at depth 512 and 4K, UD-Q4_K_XL 0.2-0.3%; prompts are unchanged.</summary>
        protected override bool GraphConcurrencyReorder => true;

        /// <summary>A prompt step's last layer runs its feed-forward on the rows the head reads only, as llama.cpp's graph does.
        /// Without it pp512 is 1.4-1.5% slower on ggml_metal (both UD quants) and MLX (stage S7, M5 Pro); decode is unchanged.</summary>
        protected override bool PrunesLastLayerRows => true;

        /// <summary>
        /// LFM3's rule, with one addition measured on Muse-Glimmer (stage S7, M5 Pro, ABBA):
        /// <list type="bullet">
        /// <item>ggml-cuda keeps same-type Q8_0 and Q4_K gate/up apart (one MMQ/MMVQ with the GLU in its write-back);</item>
        /// <item>ggml-cpu keeps the types it repacks apart, and the UD files' IQ2_S pairs too: ggml-cpu reads both products from
        /// the GGUF's mapped pages, where the fused weight was an anonymous copy of both. UD-IQ2_XXS on ggml_cpu: peak resident
        /// 17.8 -> 13.4 GB, the system's reclaimable memory 4.0 GB higher, pp512 and decode within 0.4% (the same products row
        /// for row);</item>
        /// <item>everything else stays fused, as legacy runs it: apart measured 0.997-1.000 on ggml_metal (IQ2_S and Q4_K pairs,
        /// no memory saved: Metal wires either) and 0.975-0.991 on MLX.</item>
        /// </list>
        /// </summary>
        protected override bool KeepsGateUpApart(int gateType, int upType)
            => _gateUpApart is bool apart
                ? apart && gateType == upType
                : (_backend == BackendType.GgmlCuda && gateType == upType && gateType is 8 /* Q8_0 */ or 12 /* Q4_K */)
                  || (_backend == BackendType.GgmlCpu && gateType == upType && (CpuRepacks(gateType) || gateType == 22 /* IQ2_S */));

        /// <summary>Benchmarks (eng/validation/MuseGlimmerKvGrowProbe <c>--gate-up</c>): every same-type gate/up pair kept apart
        /// (true) or fused (false) in place of <see cref="KeepsGateUpApart"/>'s rule, so both arms of that measurement run in one
        /// binary. Set before the model is constructed; no option, switch or environment variable reaches it.</summary>
        internal static readonly System.Threading.AsyncLocal<bool?> GateUpApartForTests = new();

        // Read before the base constructor runs (field initializers do), as the weights are fused inside it.
        private readonly bool? _gateUpApart = GateUpApartForTests.Value;

        /// <summary>The types ggml-cpu repacks (ggml_ops_cpu_graph.cpp repackable_type), when this build and machine repack.</summary>
        private static bool CpuRepacks(int ggmlType)
            => ggmlType is 2 /* Q4_0 */ or 8 /* Q8_0 */ or 12 /* Q4_K */ or 13 /* Q5_K */ or 14 /* Q6_K */ or 20 /* IQ4_NL */ or 39 /* MXFP4 */
               && TensorSharp.GGML.GgmlBasicOps.CpuRepacksWeights();

        /// <summary>A plain prompt's steps overlap on the device: no recurrent state, and the caches stay on the device between
        /// steps, so nothing between them reads the device back.</summary>
        protected override bool OverlapsPromptSteps => true;

        /// <summary>On ggml-vulkan a prompt chunk attends the padded F16 cache window, as decode rows do (LFM3's measurement;
        /// validated for Muse-Glimmer on the VM in S2's CUDA part).</summary>
        protected override bool PadsPromptWindowsOnVulkan => true;

        /// <summary>Attention follows positions through the step's standard inputs, its sliding-window masks included, and the
        /// emission has no side effects of its own.</summary>
        protected override bool DecodeStepsAdvance => true;

        /// <summary>The weightless input RMSNorm over the embedded rows [hidden, rows] (media rows already injected).</summary>
        protected override int EmitStreamInput(GraphStep step, int embedded)
            => step.Op(GraphOp.RmsNorm, stackalloc[] { embedded }, default, stackalloc[] { Config.Eps });

        /// <summary>Per-head Q/K RMSNorm, then the NORM rotary on the sliding-window layers only.</summary>
        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            q = step.RmsNorm(q, LayerName(layer, "attn_q_norm.weight"));
            k = step.RmsNorm(k, LayerName(layer, "attn_k_norm.weight"));
            if (!_isSwa[layer]) return;
            q = step.Rope(q, _rope);
            k = step.Rope(k, _rope);
        }

        /// <summary>attn [dv * heads, rows] times sigmoid(attn_gate · x) [q_dim, rows], elementwise (the rank's heads' rows under
        /// tensor parallelism: attn_gate shards by head blocks).</summary>
        protected override int EmitAttentionOutput(GraphStep step, int layer, int x, int attn)
            => step.Op(GraphOp.Mul, attn, step.Op(GraphOp.Sigmoid, step.MatMul(LayerName(layer, "attn_gate.weight"), x)));
    }
}
