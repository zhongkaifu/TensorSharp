// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen 3.5 in whole-forward graphs: per-head QK-norm before the partial NeoX rotary, and the
    /// Gated DeltaNet as graph nodes over each sequence's slot state - the packed in-projection
    /// [qkv | z | beta | alpha] for every row at once, then per sequence the causal conv window,
    /// SiLU, Q/K L2-norm, ggml's gated delta rule (value head h reads key head h mod nK) and the
    /// gated RMSNorm, with the conv window and the delta state written back in place.
    /// </summary>
    public sealed partial class Qwen35RuntimeModel
    {
        protected override RotarySpec GraphRotary => _rotary;

        protected override bool RecurrentMixerLowers => DeviceRecurrence && ShardDegree == 1;

        /// <summary>Qwen 3.5's fused decode measured +8% decode with ggml-metal's reordering (A3B).</summary>
        protected override bool GraphReorderOnMetal => true;

        /// <summary>ggml-metal's gated delta rule finishes reading a head's state before writing it, so a
        /// decode row may update the state in place (Qwen35Model's fused decode relies on the same);
        /// unified memory makes the backing and the state view the same bytes for every binding.
        /// Elsewhere the new state is copied.</summary>
        private bool InPlaceRecurrentState => _backend == BackendType.GgmlMetal;

        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            q = step.Rope(step.RmsNorm(q, LayerName(layer, "attn_q_norm.weight")), _rotary);
            k = step.Rope(step.RmsNorm(k, LayerName(layer, "attn_k_norm.weight")), _rotary);
        }

        protected override int EmitRecurrentMixer(GraphStep step, int layer, int normed)
        {
            const long f = sizeof(float);
            int kHeads = _numKHeads, vHeads = _numVHeads, dk = _headKDim, dv = _headVDim;
            long qk = (long)kHeads * dk, v = (long)vHeads * dv;
            long channels = 2 * qk + v, packed = channels + v + 2L * vHeads;
            int convRows = _convKernel - 1;
            float eps = Config.Eps;

            // Decode rows take the packed in-projection in one product; a prompt chunk takes its four
            // row blocks as separate products, so each part is contiguous (no slice copies).
            bool split = step.Rows > step.Sequences;
            string inProj = InProjName(layer);
            int proj = -1, qkvAll = -1, zAll = -1, betaAll = -1, alphaAll = -1;
            if (split)
            {
                qkvAll = step.MatMulRows(inProj, 0, channels, normed);                       // [channels, rows]
                zAll = step.MatMulRows(inProj, channels, v, normed);                         // [v, rows]
                betaAll = step.MatMulRows(inProj, channels + v, vHeads, normed);             // [vHeads, rows]
                alphaAll = step.MatMulRows(inProj, channels + v + vHeads, vHeads, normed);   // [vHeads, rows]
            }
            else
            {
                proj = step.MatMul(inProj, normed);                                          // [packed, rows]
            }
            int dt = step.TensorWeight(_deviceDtBias[layer][0]);
            int a = step.TensorWeight(_deviceA[layer][0]);
            int convW = step.TensorWeight(_deviceConvW[layer][0], _convKernel, channels);
            int ssmNorm = step.TensorWeight(_deviceSsmNorm[layer]);
            long nb1 = packed * f;

            int output = -1;
            for (int s = 0; s < step.Sequences; s++)
            {
                int count = step.Counts[s];
                long row0 = step.First[s] * nb1;
                var state = (DeviceGdnState)step.SequenceState(s, layer);

                int qkv, z, betaRaw, alphaRaw;
                if (split)
                {
                    long col = step.First[s];
                    qkv = step.Sequences == 1 ? qkvAll : step.View2(qkvAll, channels, count, channels * f, col * channels * f);
                    z = step.Sequences == 1 ? zAll : step.View2(zAll, v, count, v * f, col * v * f);
                    betaRaw = step.Sequences == 1 ? betaAll : step.View2(betaAll, vHeads, count, vHeads * f, col * vHeads * f);
                    alphaRaw = step.Sequences == 1 ? alphaAll : step.View2(alphaAll, vHeads, count, vHeads * f, col * vHeads * f);
                }
                else
                {
                    qkv = step.Cont(step.View2(proj, channels, count, nb1, row0));
                    z = step.Cont(step.View2(proj, v, count, nb1, row0 + channels * f));
                    betaRaw = step.Cont(step.View2(proj, vHeads, count, nb1, row0 + (channels + v) * f));
                    alphaRaw = step.Cont(step.View2(proj, vHeads, count, nb1, row0 + (channels + v + vHeads) * f));
                }

                int beta = step.Reshape(step.Op(GgmlGraphOp.Sigmoid, betaRaw), 1, vHeads, count, 1);
                int g = step.Op(GgmlGraphOp.Mul, step.Op(GgmlGraphOp.Softplus, step.Op(GgmlGraphOp.Add, alphaRaw, dt)), a);
                g = step.Reshape(g, 1, vHeads, count, 1);

                // The causal conv over [previous window | this chunk], and the new window.
                int window = step.State(state.Conv, GgmlType.F32, convRows, channels);
                // ggml_concat writes a contiguous result, so it reads the transposed view directly.
                int convIn = step.Op(GgmlGraphOp.Concat, stackalloc[] { window, step.Transpose(qkv) }, stackalloc long[] { 0 });
                int conv = step.Op(GgmlGraphOp.Silu, step.Op(GgmlGraphOp.SsmConv, convIn, convW));   // [channels, count]
                // convIn is a fresh contiguous [convRows + count, channels].
                step.Effect(step.Op(GgmlGraphOp.Cpy,
                    step.View2(convIn, convRows, channels, (convRows + count) * f, count * f), window));

                // Q, K, V as [head, heads, tokens, 1] views of the conv output; where the backend reads
                // strided rows they feed the norms and the recurrence directly, else contiguous copies.
                long token = channels * f, all = channels * count * f;
                int q4 = step.View4(conv, dk, kHeads, count, 1, dk * f, token, all, 0);
                int k4 = step.View4(conv, dk, kHeads, count, 1, dk * f, token, all, qk * f);
                int v4 = step.View4(conv, dv, vHeads, count, 1, dv * f, token, all, 2 * qk * f);
                if (!step.StridedInputs)
                {
                    q4 = step.Cont(q4);
                    k4 = step.Cont(k4);
                    v4 = step.Cont(v4);
                }
                q4 = step.L2Norm(q4, dk, eps);
                k4 = step.L2Norm(k4, dk, eps);

                // Every graph binds the state through one key per backend: on Metal the slot's backing
                // buffer [one row's outputs | state], so a decode row's recurrence can write its result
                // onto it and update the state in place; elsewhere the state itself, copied back.
                long stateFloats = (long)dk * dv * vHeads;
                bool viaBacking = InPlaceRecurrentState;
                int backing = -1, s4;
                if (viaBacking)
                {
                    backing = step.State(state.Backing, GgmlType.F32, state.RowFloats + stateFloats);
                    s4 = step.View4(backing, dk, dv, vHeads, 1, dk * f, dk * dv * f, stateFloats * f, state.RowFloats * f);
                }
                else
                {
                    s4 = step.Reshape(step.State(state.Ssm, GgmlType.F32, dk, dv, vHeads), dk, dv, vHeads, 1);
                }
                int gdn = step.Op(GgmlGraphOp.GatedDeltaNet, stackalloc[] { q4, k4, v4, g, beta, s4 }, stackalloc long[] { 1 });
                int scores = step.View2(gdn, v, count, v * f, 0);
                int newState = step.Op(GgmlGraphOp.View, stackalloc[] { gdn },
                    stackalloc long[] { 4, dk, dv, vHeads, 1, dk * f, dk * dv * f, dk * dv * vHeads * f, v * count * f });
                int copy = step.Op(GgmlGraphOp.Cpy, newState, s4);
                if (viaBacking && count == 1)
                    // [outputs | state] of one row is exactly the backing: the recurrence updates the
                    // state in place, as Qwen35Model's fused decode does on Metal.
                    step.Alias(gdn, backing, 0, copy);
                else
                    step.Effect(copy);

                int normedOut = step.RmsNorm(step.Reshape(step.Cont(scores), dv, vHeads, count), ssmNorm, eps);
                int gated = step.Op(GgmlGraphOp.Mul, normedOut, step.Op(GgmlGraphOp.Silu, step.Reshape(z, dv, vHeads, count)));
                int part = step.Reshape(gated, v, count);
                output = output < 0 ? part : step.Op(GgmlGraphOp.Concat, stackalloc[] { output, part }, stackalloc long[] { 1 });
            }
            return step.MatMul(LayerName(layer, "ssm_out.weight"), output);
        }
    }
}
