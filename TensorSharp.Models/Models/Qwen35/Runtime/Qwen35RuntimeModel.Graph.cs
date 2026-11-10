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

        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            string p = $"blk.{layer}.";
            q = step.Rope(step.RmsNorm(q, p + "attn_q_norm.weight"), _rotary);
            k = step.Rope(step.RmsNorm(k, p + "attn_k_norm.weight"), _rotary);
        }

        protected override int EmitRecurrentMixer(GraphStep step, int layer, int normed)
        {
            const long f = sizeof(float);
            int kHeads = _numKHeads, vHeads = _numVHeads, dk = _headKDim, dv = _headVDim;
            long qk = (long)kHeads * dk, v = (long)vHeads * dv;
            long channels = 2 * qk + v, packed = channels + v + 2L * vHeads;
            int convRows = _convKernel - 1;
            float eps = Config.Eps;

            int proj = step.MatMul(InProjName(layer), normed);                     // [packed, rows]
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

                int qkv = step.Op(GgmlGraphOp.Cont, step.View2(proj, channels, count, nb1, row0));
                int z = step.Op(GgmlGraphOp.Cont, step.View2(proj, v, count, nb1, row0 + channels * f));
                int betaRaw = step.Op(GgmlGraphOp.Cont, step.View2(proj, vHeads, count, nb1, row0 + (channels + v) * f));
                int alphaRaw = step.Op(GgmlGraphOp.Cont, step.View2(proj, vHeads, count, nb1, row0 + (channels + v + vHeads) * f));

                int beta = step.Reshape(step.Op(GgmlGraphOp.Sigmoid, betaRaw), 1, vHeads, count, 1);
                int g = step.Op(GgmlGraphOp.Mul, step.Op(GgmlGraphOp.Softplus, step.Op(GgmlGraphOp.Add, alphaRaw, dt)), a);
                g = step.Reshape(g, 1, vHeads, count, 1);

                // The causal conv over [previous window | this chunk], and the new window.
                int window = step.State(state.Conv, GgmlType.F32, convRows, channels);
                int convIn = step.Op(GgmlGraphOp.Concat, stackalloc[] { window, step.Op(GgmlGraphOp.Transpose, qkv) }, stackalloc long[] { 0 });
                int conv = step.Op(GgmlGraphOp.Silu, step.Op(GgmlGraphOp.SsmConv, convIn, convW));   // [channels, count]
                var (_, convNb, _) = step.G.Shape(convIn);
                step.G.Effect(step.Op(GgmlGraphOp.Cpy,
                    step.View2(convIn, convRows, channels, convNb[1], count * convNb[0]), window));

                int qc = step.L2Norm(step.Op(GgmlGraphOp.Cont, step.View3(conv, dk, kHeads, count, dk * f, channels * f, 0)), dk, eps);
                int kc = step.L2Norm(step.Op(GgmlGraphOp.Cont, step.View3(conv, dk, kHeads, count, dk * f, channels * f, qk * f)), dk, eps);
                int vc = step.Op(GgmlGraphOp.Cont, step.View3(conv, dv, vHeads, count, dv * f, channels * f, 2 * qk * f));

                int ssm = step.State(state.Ssm, GgmlType.F32, dk, dv, vHeads);
                int s4 = step.Reshape(ssm, dk, dv, vHeads, 1);
                int gdn = step.Op(GgmlGraphOp.GatedDeltaNet,
                    stackalloc[] { step.Reshape(qc, dk, kHeads, count, 1), step.Reshape(kc, dk, kHeads, count, 1),
                                   step.Reshape(vc, dv, vHeads, count, 1), g, beta, s4 },
                    stackalloc long[] { 1 });
                int scores = step.View2(gdn, v, count, v * f, 0);
                int newState = step.Op(GgmlGraphOp.View, stackalloc[] { gdn },
                    stackalloc long[] { 4, dk, dv, vHeads, 1, dk * f, dk * dv * f, dk * dv * vHeads * f, v * count * f });
                step.G.Effect(step.Op(GgmlGraphOp.Cpy, newState, s4));

                int normedOut = step.RmsNorm(step.Reshape(step.Op(GgmlGraphOp.Cont, scores), dv, vHeads, count), ssmNorm, eps);
                int gated = step.Op(GgmlGraphOp.Mul, normedOut, step.Op(GgmlGraphOp.Silu, step.Reshape(z, dv, vHeads, count)));
                int part = step.Reshape(gated, v, count);
                output = output < 0 ? part : step.Op(GgmlGraphOp.Concat, stackalloc[] { output, part }, stackalloc long[] { 1 });
            }
            return step.MatMul($"blk.{layer}.ssm_out.weight", output);
        }
    }
}
