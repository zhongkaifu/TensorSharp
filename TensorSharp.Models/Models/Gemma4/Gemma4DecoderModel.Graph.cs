// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>Gemma 4's pieces of a step's graph (see <see cref="Gemma4DecoderModel"/>).</summary>
    public abstract partial class Gemma4DecoderModel
    {
        private protected const long F = sizeof(float);

        /// <summary>The per-layer embeddings [ple, layers, rows] of a step (one node a step; ranks trace steps of their own).</summary>
        private readonly ConditionalWeakTable<GraphStep, StrongBox<int>> _stepPle = new();

        /// <summary>Formats ggml's get_rows reads on every backend: a table of another format is gathered on the host.</summary>
        private static bool GathersOnDevice(int format) => format is 0 or 1 or 2 or 3 or 6 or 7 or 8 or 30;

        protected override bool DecodeStepsAdvance => true;

        // ---------------------------------------------------------------- stream input

        /// <summary>The embedded rows times sqrt(n_embd) (media rows as the encoder made them: legacy scaled the embedding before
        /// it wrote them), then the step's per-layer embeddings from the tokens and the scaled rows.</summary>
        protected override int EmitStreamInput(GraphStep step, int embedded)
        {
            int rows = step.Rows;
            float scale = MathF.Sqrt(Config.HiddenSize);
            int h;
            if (step.MediaRows is { } media)
            {
                var column = new float[rows];
                for (int i = 0; i < rows; i++) column[i] = media[i] ? 1f : scale;
                h = step.Op(GraphOp.Mul, embedded, step.ConstantInputF32(column, 1, rows));
            }
            else
            {
                h = step.Op(GraphOp.Scale, stackalloc[] { embedded }, default, stackalloc[] { scale });
            }
            if (_pleDim > 0)
                _stepPle.AddOrUpdate(step, new StrongBox<int>(EmitPerLayerInputs(step, h)));
            return h;
        }

        /// <summary>
        /// legacy's ComputePLE as graph ops: the token rows of <c>per_layer_token_embd</c> times sqrt(ple), plus the RMSNorm (by
        /// <c>per_layer_proj_norm</c>) of <c>per_layer_model_proj</c> over the scaled embedded rows over sqrt(n_embd), the sum
        /// times 1/sqrt(2): [ple, layers, rows].
        /// </summary>
        private int EmitPerLayerInputs(GraphStep step, int h)
        {
            int rows = step.Rows, layers = Config.NumLayers;
            long width = (long)_pleDim * layers;
            int tokens = EmitPleTokenRows(step, width);
            tokens = step.Op(GraphOp.Scale, stackalloc[] { tokens }, default, stackalloc[] { MathF.Sqrt(_pleDim) });
            int proj = step.MatMul("per_layer_model_proj.weight", h);                                       // [L ple, rows]
            proj = step.Op(GraphOp.Scale, stackalloc[] { proj }, default, stackalloc[] { 1f / MathF.Sqrt(Config.HiddenSize) });
            proj = step.RmsNorm(step.Reshape(proj, _pleDim, layers, rows), "per_layer_proj_norm.weight");  // [ple, L, rows]
            int sum = step.Op(GraphOp.Add, proj, step.Reshape(tokens, _pleDim, layers, rows));
            return step.Op(GraphOp.Scale, stackalloc[] { sum }, default, stackalloc[] { 1f / MathF.Sqrt(2f) });
        }

        /// <summary>The step's tokens' rows of the per-layer token table [L ple, rows]: gathered in the graph where the backend's
        /// get_rows reads the table's format, else on the host (an input the decode advance refills).</summary>
        private int EmitPleTokenRows(GraphStep step, long width)
        {
            int rows = step.Rows;
            int[] stepTokens = StepTokens ?? throw new InvalidOperationException($"{Config.Architecture}: a step without its tokens.");
            if (stepTokens.Length != rows)
                throw new InvalidOperationException($"{Config.Architecture}: {stepTokens.Length} tokens for a step of {rows} rows.");
            const string table = "per_layer_token_embd.weight";
            if (_quantWeights.TryGetValue(table, out QuantizedWeight q) && !GathersOnDevice(q.GgmlType))
            {
                float[] gathered = GatherPleRows(q, stepTokens, width);
                int input = step.InputF32(gathered, width, rows);
                step.RefillOnAdvance(input, () =>
                {
                    int[] now = StepTokens;
                    if (now == null || now.Length != rows) return false;
                    float[] fresh = GatherPleRows(q, now, width);
                    Array.Copy(fresh, gathered, gathered.Length);
                    return true;
                });
                return input;
            }
            var ids = (int[])stepTokens.Clone();
            int idsInput = step.InputI32(ids, rows);
            step.RefillOnAdvance(idsInput, () =>
            {
                int[] now = StepTokens;
                if (now == null || now.Length != rows) return false;
                now.CopyTo(ids, 0);
                return true;
            });
            return step.Op(GraphOp.GetRows, step.Weight(table), idsInput);
        }

        private static unsafe float[] GatherPleRows(QuantizedWeight table, int[] tokens, long width)
        {
            if (!table.HasHostData)
                throw new InvalidOperationException("The per-layer token table's host bytes were released; its rows cannot be gathered.");
            var rows = new float[width * tokens.Length];
            long rowBytes = NativeDequant.RowSize(table.GgmlType, table.Ne0);
            fixed (float* dst = rows)
            {
                byte* basePtr = (byte*)table.Data.ToPointer();
                for (int i = 0; i < tokens.Length; i++)
                    ManagedQuantizedOps.DequantizeRowToFloat32(table.GgmlType, (IntPtr)(basePtr + tokens[i] * rowBytes), dst + i * width, (int)width);
            }
            return rows;
        }

        /// <summary>Whole-layer stages hand the per-layer embeddings [ple, L, rows] on beside the stream.</summary>
        protected override int StageCarryWidth => _pleDim > 0 ? _pleDim * Config.NumLayers : 0;

        protected override int EmitStageCarry(GraphStep step) => step.Reshape(PerLayerInputs(step), (long)_pleDim * Config.NumLayers, step.Rows);

        protected override void TakeStageCarry(GraphStep step, int carry)
            => _stepPle.AddOrUpdate(step, new StrongBox<int>(step.Reshape(carry, _pleDim, Config.NumLayers, step.Rows)));

        private int PerLayerInputs(GraphStep step)
            => _stepPle.TryGetValue(step, out StrongBox<int> node) ? node.Value
                : throw new InvalidOperationException($"{Config.Architecture}: a step's per-layer embeddings are emitted with its stream input.");

        // ---------------------------------------------------------------- attention

        /// <summary>The layer's queries [d, heads, rows], keys and values [d, kv heads, rows] (none for a KV-shared layer), the
        /// values unweighted-RMS-normed; a layer without <c>attn_v</c> takes its raw keys as values.</summary>
        protected override bool EmitAttentionProjections(GraphStep step, int layer, int x, out int q, out int k, out int v)
        {
            int rows = step.Rows, degree = step.Degree;
            int d = LayerKeyDim(layer), heads = Config.NumHeads / degree, kvHeads = _kvHeads[layer] / degree;
            long qDim = (long)heads * d, kDim = (long)kvHeads * d;
            string qkvName = LayerName(layer, "attn_qkv.weight");
            if (_kvSource[layer] == layer && HasWeight(qkvName))
            {
                int qkv = step.MatMul(qkvName, x);
                long width = qDim + 2 * kDim;
                q = step.View3(qkv, d, heads, rows, d * F, width * F, 0);
                k = step.View3(qkv, d, kvHeads, rows, d * F, width * F, qDim * F);
                v = step.View3(qkv, d, kvHeads, rows, d * F, width * F, (qDim + kDim) * F);
            }
            else
            {
                q = step.View3(step.MatMul(LayerName(layer, "attn_q.weight"), x), d, heads, rows, d * F, qDim * F, 0);
                if (_kvSource[layer] != layer)
                {
                    k = v = -1;
                    return true;
                }
                k = step.Reshape(step.MatMul(LayerName(layer, "attn_k.weight"), x), d, kvHeads, rows);
                v = _hasValue[layer] ? step.Reshape(step.MatMul(LayerName(layer, "attn_v.weight"), x), d, kvHeads, rows) : k;
            }
            // The unweighted V norm (legacy's ApplyUnweightedRMSNorm).
            v = step.Op(GraphOp.RmsNorm, stackalloc[] { step.Cont(v) }, default, stackalloc[] { Config.Eps });
            return true;
        }

        /// <summary>Per-head RMSNorm of the queries and keys, then the layer's NeoX rotary (the global one with its factors).</summary>
        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            RotarySpec rope = _isLocal[layer] ? _localRope : _globalRope;
            q = step.Rope(step.RmsNorm(q, LayerName(layer, "attn_q_norm.weight")), rope);
            if (k >= 0)
                k = step.Rope(step.RmsNorm(k, LayerName(layer, "attn_k_norm.weight")), rope);
        }

        // ---------------------------------------------------------------- feed-forward

        /// <summary>A MoE layer's routed branch reads the raw residual: it is the block's context.</summary>
        protected override int EmitBlockInput(GraphStep step, int layer, int stream, BlockSite site, out int context)
        {
            int normed = base.EmitBlockInput(step, layer, stream, site, out context);
            if (site == BlockSite.Ffn && IsMoeLayer(layer)) context = stream;
            return normed;
        }

        /// <summary>The dense GeGLU feed-forward; a MoE layer's output is [dense | routed] along ne0 (two partials the runtime
        /// reduces as one), which <see cref="EmitBlockOutput"/> norms apart.</summary>
        protected override int EmitFeedForward(GraphStep step, int layer, int x, int context)
        {
            int dense = EmitGegluFfn(step, layer, x);
            if (!IsMoeLayer(layer)) return dense;
            int routed = EmitRoutedExperts(step, layer, context);
            return step.Op(GraphOp.Concat, stackalloc[] { dense, routed }, stackalloc long[] { 0 });
        }

        protected override int EmitFeedForward(GraphStep step, int layer, int x) => EmitFeedForward(step, layer, x, -1);

        private int EmitGegluFfn(GraphStep step, int layer, int x)
        {
            string gateUp = LayerName(layer, "ffn_gate_up.weight");
            int act = HasWeight(gateUp)
                ? step.Op(GraphOp.Geglu, step.MatMul(gateUp, x))
                : step.Op(GraphOp.GegluSplit, step.MatMul(LayerName(layer, "ffn_gate.weight"), x), step.MatMul(LayerName(layer, "ffn_up.weight"), x));
            return step.MatMul(LayerName(layer, "ffn_down.weight"), act);
        }

        /// <summary>
        /// The routed branch over the raw residual <paramref name="h"/> [n_embd, rows]: router input RMSNorm(h) (no weight) /
        /// sqrt(n_embd) * <c>ffn_gate_inp.scale</c>; softmax over the experts, the top k renormalized and times each expert's
        /// <c>ffn_down_exps.scale</c>; GeGLU experts (fused <c>ffn_gate_up_exps</c>, gate the first half) over
        /// <c>pre_ffw_norm_2</c>(h); the weighted sum [n_embd, rows].
        /// </summary>
        private int EmitRoutedExperts(GraphStep step, int layer, int h)
        {
            if (h < 0) throw new InvalidOperationException($"{Config.Architecture} layer {layer}: the routed experts need the residual.");
            int rows = step.Rows, hidden = Config.HiddenSize, k = _expertsUsed, experts = _numExperts;
            int router = step.Op(GraphOp.RmsNorm, stackalloc[] { h }, default, stackalloc[] { Config.Eps });
            router = step.Op(GraphOp.Scale, stackalloc[] { router }, default, stackalloc[] { 1f / MathF.Sqrt(hidden) });
            string routerScale = LayerName(layer, "ffn_gate_inp.scale");
            if (HasWeight(routerScale)) router = step.Op(GraphOp.Mul, router, step.Weight(routerScale));
            int logits = step.MatMul(LayerName(layer, "ffn_gate_inp.weight"), router);                         // [E, rows]
            int moeIn = step.RmsNorm(h, FirstWeight(layer, "pre_ffw_norm_2.weight", "ffn_pre_norm_2.weight"));
            string fused = LayerName(layer, "ffn_gate_up_exps.weight"), downName = LayerName(layer, "ffn_down_exps.weight");
            string expertScale = LayerName(layer, "ffn_down_exps.scale");
            bool scaled = _weights.TryGetValue(expertScale, out Tensor scales);
            // Experts kept in host memory (--n-cpu-moe / --cpu-moe, or the runtime's placement plan) run at a pause of the
            // program on an executor that takes host experts: GGML's host MoE segment, the tensor executors' expert host (the
            // host's routes through the host kernel, the device's tier through the executor's ops), GeGLU over the fused stack.
            bool host = IsExpertLayerOffloaded(layer) && GraphExecutor.Capabilities.HostExperts && HasExperts(fused);
            // The selection and the experts as one region an executor may run as its own kernels (direct CUDA's Gemma MoE
            // decode kernels); tests that force or read the routes take them from the ops.
            bool region = !host && HasExperts(fused) && ForcedRoutesForTests == null && DeviceRoutesForTests == null;
            if (region)
            {
                if (scaled)
                    step.BeginFused(GraphFusion.GegluRoutedExperts,
                        stackalloc[] { moeIn, logits, step.Experts(fused), step.Experts(downName), step.Weight(expertScale) }, stackalloc long[] { k, 1 });
                else
                    step.BeginFused(GraphFusion.GegluRoutedExperts,
                        stackalloc[] { moeIn, logits, step.Experts(fused), step.Experts(downName) }, stackalloc long[] { k, 0 });
            }
            int probs = step.Op(GraphOp.SoftMaxExt, logits);
            int forced = EmitForcedRoutes(step, layer, k, experts);                                                 // tests
            int selected = forced >= 0 ? forced : step.Op(GraphOp.TopK, stackalloc[] { probs }, stackalloc long[] { k });   // [K, rows]
            EmitDeviceRoutes(step, layer, selected, probs, k, experts);                                             // tests
            int picked = step.Reshape(step.Op(GraphOp.GetRows, step.Reshape(probs, 1, experts, rows), selected), k, rows);
            int weights = step.Op(GraphOp.Div, picked, step.Op(GraphOp.SumRows, picked));
            weights = step.Reshape(weights, 1, k, rows);
            if (scaled)
            {
                int table = step.Reshape(step.Weight(expertScale), 1, scales.ElementCount());                    // [1, E]
                // The top-k is a view of its sort: copied before it is flattened.
                int ids = step.Reshape(step.Op(GraphOp.Cont, selected), (long)k * rows);
                int each = step.Op(GraphOp.GetRows, table, ids);                                                  // [1, K rows]
                weights = step.Op(GraphOp.Mul, weights, step.Reshape(each, 1, k, rows));
            }

            if (host)
            {
                int stack = step.Experts(fused);
                return step.Op(GraphOp.MoeHostExperts, stackalloc[] { moeIn, selected, weights, stack, stack, step.Experts(downName) },
                    stackalloc long[] { layer, 2 /* gelu(gate) * up */, 1 /* fused gate|up */ });
            }
            int input = step.Reshape(moeIn, hidden, 1, rows);
            int act;
            if (HasExperts(fused))
            {
                int gateUp = step.Op(GraphOp.MulMatId, step.Experts(fused), input, selected);                     // [2 ff, K, rows]
                act = step.Op(GraphOp.Geglu, gateUp);                                                             // [ff, K, rows]
            }
            else
            {
                int gate = step.Op(GraphOp.MulMatId, step.Experts(LayerName(layer, "ffn_gate_exps.weight")), input, selected);
                int up = step.Op(GraphOp.MulMatId, step.Experts(LayerName(layer, "ffn_up_exps.weight")), input, selected);
                act = step.Op(GraphOp.GegluSplit, gate, up);
            }
            int down = step.Op(GraphOp.MulMatId, step.Experts(downName), act, selected);                          // [n, K, rows]
            down = step.Op(GraphOp.Mul, down, weights);
            long rowBytes = hidden * F;
            int output = -1;
            for (int j = 0; j < k; j++)
            {
                int slice = step.View2(down, hidden, rows, rowBytes * k, j * rowBytes);
                output = output < 0 ? slice : step.Op(GraphOp.Add, output, slice);
            }
            output = step.Op(GraphOp.Cont, output);
            if (region) step.EndFused(stackalloc[] { output });
            return output;
        }

        private bool HasExperts(string name) => _stackedExpertWeights.ContainsKey(name);

        // ---------------------------------------------------------------- block output

        /// <summary>The feed-forward's output through its post-norm(s) into the stream, then the per-layer embedding's gated
        /// projection and the layer's output scale. The attention's output takes the runtime's sandwich norm.</summary>
        protected override int EmitBlockOutput(GraphStep step, int layer, int stream, BlockSite site, int context, int output)
        {
            if (site == BlockSite.Mixer) return base.EmitBlockOutput(step, layer, stream, site, context, output);
            int rows = step.Rows, hidden = Config.HiddenSize;
            int ffn = output;
            if (IsMoeLayer(layer))
            {
                int dense = step.Cont(step.View2(output, hidden, rows, 2L * hidden * F, 0));
                int routed = step.Cont(step.View2(output, hidden, rows, 2L * hidden * F, hidden * F));
                dense = step.RmsNorm(dense, FirstWeight(layer, "post_ffw_norm_1.weight", "ffn_post_norm_1.weight"));
                routed = step.RmsNorm(routed, FirstWeight(layer, "post_ffw_norm_2.weight", "ffn_post_norm_2.weight"));
                ffn = step.Op(GraphOp.Add, dense, routed);
            }
            int h = step.Op(GraphOp.Add, step.RmsNorm(ffn, FirstWeight(layer, "post_ffw_norm.weight", "ffn_post_norm.weight")), stream);
            if (HasPle(layer)) h = EmitPerLayerEmbedding(step, layer, h);
            float scale = LayerOutputScale(step, layer);
            return scale != 1f ? step.Op(GraphOp.Scale, stackalloc[] { h }, default, stackalloc[] { scale }) : h;
        }

        /// <summary>The scale on <paramref name="layer"/>'s output in <paramref name="step"/>: <c>layer_output_scale</c>
        /// (DiffusionGemma scales a prompt's rows by its encoder scalar instead).</summary>
        private protected virtual float LayerOutputScale(GraphStep step, int layer) => _layerScale[layer];

        /// <summary>h + RMSNorm(proj(gelu(inp_gate h) * ple_l)) by <c>post_norm</c>.</summary>
        private int EmitPerLayerEmbedding(GraphStep step, int layer, int h)
        {
            int rows = step.Rows;
            int ple = PerLayerInputs(step);
            int slice = step.Cont(step.View2(ple, _pleDim, rows, (long)_pleDim * Config.NumLayers * F, (long)layer * _pleDim * F));
            int gate = step.MatMul(LayerName(layer, "inp_gate.weight"), h);                                      // [ple, rows]
            int gated = step.Op(GraphOp.Mul, step.Op(GraphOp.Gelu, gate), slice);
            int proj = step.MatMul(LayerName(layer, "proj.weight"), gated);                                       // [n_embd, rows]
            return step.Op(GraphOp.Add, h, step.RmsNorm(proj, LayerName(layer, "post_norm.weight")));
        }
    }
}
