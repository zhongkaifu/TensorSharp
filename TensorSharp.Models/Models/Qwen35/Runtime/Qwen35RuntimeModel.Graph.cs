// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.GGML;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen 3.5 in whole-forward graphs: per-head QK-norm before the partial NeoX rotary, and the
    /// Gated DeltaNet as graph nodes over each sequence's slot state - the packed in-projection
    /// [qkv | z | beta | alpha] for every row at once, then per sequence the causal conv window,
    /// SiLU, Q/K L2-norm, ggml's gated delta rule (value head h reads key head h mod nK) and the
    /// gated RMSNorm, with the conv window and the delta state written back in place.
    /// </summary>
    public partial class Qwen35RuntimeModel
    {
        protected override RotarySpec GraphRotary => _rotary;

        protected override bool RecurrentMixerLowers => TensorRecurrence && (ShardDegree == 1 || RankGraphs);

        /// <summary>The mixer traces one rank's heads (its sharded projections, per-rank constants and states).</summary>
        protected override bool RanksLower => true;

        /// <summary>Qwen 3.5's fused decode measured +8% decode with ggml-metal's reordering (A3B).</summary>
        protected override bool GraphConcurrencyReorder => true;

        /// <summary>ggml-metal's gated delta rule finishes reading a head's state before writing it, so a
        /// decode row may update the state in place (Qwen35Model's fused decode relies on the same);
        /// unified memory makes the backing and the state view the same bytes for every binding.
        /// Elsewhere the new state is copied.</summary>
        private bool InPlaceRecurrentState => GraphExecutor?.Capabilities.InPlaceStateAlias == true;

        /// <summary>The recurrence binds its states as leaves and adds no inputs; attention and the experts (a host
        /// seam's parameters are its layer) follow positions through the step's standard inputs only.</summary>
        protected override bool DecodeStepsAdvance => true;

        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            q = step.Rope(step.RmsNorm(q, LayerName(layer, "attn_q_norm.weight")), _rotary);
            k = step.Rope(step.RmsNorm(k, LayerName(layer, "attn_k_norm.weight")), _rotary);
        }

        // Commit programs replay per (verify buffer, kept, rows): only copies, no inputs.
        private readonly System.Collections.Generic.Dictionary<(IntPtr, int, int), IGraphPlan> _commitGraphs = new();

        protected override void OnGraphCacheCleared()
        {
            foreach (IGraphPlan g in _commitGraphs.Values) g.Dispose();
            _commitGraphs.Clear();
        }

        /// <summary>Set the active request's recurrent states to those after <paramref name="kept"/> rows of
        /// the last <paramref name="rows"/>-row verify, from its verify buffers (0 rows: the live state is
        /// still the one before the window).</summary>
        protected override void CommitRecurrentPrefix(int kept, int rows)
        {
            // The per-op path's states keep their own window (a multi-node pipeline verifies per op).
            if (!TensorRecurrence || PerOpVerifyOpen())
            {
                base.CommitRecurrentPrefix(kept, rows);
                return;
            }
            CommitHostVerify(kept, rows);
            if (kept == 0) return;
            // One commit graph per pipeline stage and tensor-parallel rank: each binds its own device's states (a rank's
            // hold its share of the heads).
            for (int stage = 0; stage < Mesh.Stages; stage++)
            for (int rank = 0; rank < Mesh.TensorParallel; rank++)
            {
                IntPtr key = IntPtr.Zero;
                for (int layer = 0; layer < Config.NumLayers && key == IntPtr.Zero; layer++)
                    if (_isRecurrent[layer] && OwnsLayer(layer) && Mesh.StageOf(layer) == stage
                        && ActiveRecurrentState(layer, rank) is DeviceGdnState first && first.Verify != null)
                        key = StateIdentity(first.Verify);
                if (key == IntPtr.Zero) continue;
                EnterGraphStage(stage);
                try
                {
                    if (!_commitGraphs.TryGetValue((key, kept, rows), out IGraphPlan g))
                    {
                        g = CompileForDevice(BuildCommitProgram(kept, rows, stage, rank), stage, rank);
                        _commitGraphs[(key, kept, rows)] = g;
                    }
                    if (!g.Run(ReadOnlySpan<int>.Empty, ReadOnlySpan<IntPtr>.Empty, ReadOnlySpan<long>.Empty))
                        throw new InvalidOperationException($"{Config.Architecture} recurrent-state commit failed: {g.LastError}");
                }
                finally
                {
                    EnterGraphStage(0);
                }
            }
        }

        /// <summary>A per-op verify window is open on the active request's states (they settle it themselves).</summary>
        private bool PerOpVerifyOpen()
        {
            foreach (RecurrentState state in ActiveRecurrentStates())
                if (state is DeviceGdnState { Verifying: true }) return true;
            return false;
        }

        private GraphProgram BuildCommitProgram(int kept, int rows, int stage, int rank)
        {
            const long f = sizeof(float);
            var g = new GraphProgramBuilder(record: true);
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!_isRecurrent[layer] || !OwnsLayer(layer) || Mesh.StageOf(layer) != stage
                    || ActiveRecurrentState(layer, rank) is not DeviceGdnState state || state.Verify == null)
                    continue;
                long stateFloats = state.StateFloats, stateBytes = stateFloats * f;
                int cap = state.SnapshotCapacity, convRows = state.ConvRows, channels = state.Channels;
                // Result layout [rows' outputs | state after the last row | state 1 row before | ...].
                long offset = state.RowFloats * rows * f + (rows - kept) * stateBytes;
                int verify = GraphLeaf(g, state.Verify, cap * (state.RowFloats + stateFloats));
                int source = g.Op(GraphOp.View, stackalloc[] { verify }, stackalloc long[] { 1, stateFloats, 1, 1, 1, stateBytes, 0, 0, offset });
                int target = InPlaceRecurrentState
                    ? g.Op(GraphOp.View, stackalloc[] { GraphLeaf(g, state.Backing, state.RowFloats + stateFloats) },
                        stackalloc long[] { 1, stateFloats, 1, 1, 1, stateBytes, 0, 0, state.RowFloats * f })
                    : g.Op(GraphOp.Reshape, stackalloc[] { GraphLeaf(g, state.Ssm, _headKDim, _headVDim, state.Ssm.Sizes[0]) }, stackalloc long[] { stateFloats });
                g.Effect(g.Op(GraphOp.Cpy, source, target));
                int hist = GraphLeaf(g, state.ConvHistory, convRows + cap, channels);
                int window = g.Op(GraphOp.View, stackalloc[] { hist },
                    stackalloc long[] { 2, convRows, channels, 1, 1, (convRows + cap) * f, 0, 0, kept * f });
                g.Effect(g.Op(GraphOp.Cpy, window, GraphLeaf(g, state.Conv, convRows, channels)));
                EmitCommitExtras(g, layer, state, kept, rows);
            }
            return g.ToProgram(replayable: true, reorder: false);
        }

        /// <summary>A family's host-side state settled by a verify that kept <paramref name="kept"/> of its
        /// <paramref name="rows"/> (0: none, the state before the window stays).</summary>
        protected virtual void CommitHostVerify(int kept, int rows) { }

        /// <summary>What else the commit program copies for a layer's state (a family's extra histories).</summary>
        protected virtual void EmitCommitExtras(GraphProgramBuilder g, int layer, DeviceGdnState state, int kept, int rows) { }

        protected int GraphLeaf(GraphProgramBuilder g, Tensor t, params long[] ne)
            => g.Leaf(GraphLeafRole.State, GraphFormat.F32, ne, StateIdentity(t), t.Storage.ByteLength, handle: t);

        /// <summary>A state tensor's bytes where the executor binds host memory, else a stable identity.</summary>
        private IntPtr StateIdentity(Tensor t) => GraphExecutor.Capabilities.BindsHostPointers
            ? TensorComputePrimitives.GetStoragePointer(t)
            : GraphIdentity.Of(t.Storage, t.StorageOffset);

        protected override int EmitRecurrentMixer(GraphStep step, int layer, int normed)
        {
            const long f = sizeof(float);
            // A tensor-parallel rank traces its own heads (its shards of the projections, constants and state).
            int kHeads = _numKHeads / step.Degree, vHeads = _numVHeads / step.Degree, dk = _headKDim, dv = _headVDim;
            long qk = (long)kHeads * dk, v = (long)vHeads * dv;
            long channels = 2 * qk + v, packed = channels + v + 2L * vHeads;
            int convRows = _convKernel - 1;
            float eps = Config.Eps;

            // Decode rows take the packed in-projection in one product; a prompt chunk takes its four
            // row blocks as separate products, so each part is contiguous (no slice copies).
            string inProj = InProjName(layer);
            bool isPacked = _quantWeights.ContainsKey(inProj) || _weights.ContainsKey(inProj)
                || (step.Rank >= 0 && (_tpQuantWeights.ContainsKey(inProj) || _tpWeights.ContainsKey(inProj)));
            bool split = !isPacked || step.Rows > step.Sequences;
            int proj = -1, qkvAll = -1, zAll = -1, betaAll = -1, alphaAll = -1;
            if (!isPacked)
            {
                // The projections as they were quantized (mixed types are not packed).
                qkvAll = step.MatMul(LayerName(layer, "attn_qkv.weight"), normed);
                zAll = step.MatMul(LayerName(layer, "attn_gate.weight"), normed);
                betaAll = step.MatMul(LayerName(layer, "ssm_beta.weight"), normed);
                alphaAll = step.MatMul(LayerName(layer, "ssm_alpha.weight"), normed);
            }
            else if (split)
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
            int dt = step.TensorWeight(_deviceDtBias[layer][step.RankIndex]);
            int a = step.TensorWeight(_deviceA[layer][step.RankIndex]);
            int convW = step.TensorWeight(_deviceConvW[layer][step.RankIndex], _convKernel, channels);
            int ssmNorm = step.TensorWeight(step.Replicated(_deviceSsmNorm[layer]));
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

                int window = step.State(state.Conv, GraphFormat.F32, convRows, channels);
                // A verify window leaves the live window alone: the commit sets it from the conv history.
                bool keep = step.RecurrentSnapshots && count > 1;

                // Every graph binds the state through one key per backend: on Metal the slot's backing
                // buffer [one row's outputs | state], so a decode row's recurrence can write its result
                // onto it and update the state in place; elsewhere the state itself, copied back.
                long stateFloats = (long)dk * dv * vHeads;
                bool viaBacking = InPlaceRecurrentState;
                int backing = -1, s4;
                if (viaBacking)
                {
                    backing = step.State(state.Backing, GraphFormat.F32, state.RowFloats + stateFloats);
                    s4 = step.View4(backing, dk, dv, vHeads, 1, dk * f, dk * dv * f, stateFloats * f, state.RowFloats * f);
                }
                else
                {
                    s4 = step.Reshape(step.State(state.Ssm, GraphFormat.F32, dk, dv, vHeads), dk, dv, vHeads, 1);
                }

                // A prompt chunk's whole mixer, from its projections to the gated output, is one region a backend
                // may run as one kernel (GraphFusion.DeltaNetPrompt: Metal's prompt-width delta-net kernels).
                bool mixer = !keep && count > 1;
                if (mixer)
                    step.BeginFused(GraphFusion.DeltaNetPrompt, stackalloc[] { qkv, z, alphaRaw, betaRaw, convW, dt, a, ssmNorm, window, s4 },
                        stackalloc long[] { dk, kHeads, dv, vHeads, MixerSigmoidGate ? 1 : 0, _convKernel }, stackalloc[] { eps });

                // The delta rule's inputs, one region a backend may run as one kernel (GraphFusion.DeltaNetPrep):
                // the gates, the causal conv over [previous window | this chunk] and the L2-normed Q and K.
                step.BeginFused(GraphFusion.DeltaNetPrep, stackalloc[] { window, qkv, convW, alphaRaw, betaRaw, dt, a },
                    stackalloc long[] { dk, kHeads, dv, vHeads }, stackalloc[] { eps });
                int beta = step.Reshape(step.Op(GraphOp.Sigmoid, betaRaw), 1, vHeads, count, 1);
                int g = step.Op(GraphOp.Mul, step.Op(GraphOp.Softplus, step.Op(GraphOp.Add, alphaRaw, dt)), a);
                g = step.Reshape(g, 1, vHeads, count, 1);
                // ggml_concat writes a contiguous result, so it reads the transposed view directly.
                int convIn = step.Op(GraphOp.Concat, stackalloc[] { window, step.Transpose(qkv) }, stackalloc long[] { 0 });
                int conv = step.Op(GraphOp.Silu, step.Op(GraphOp.SsmConv, convIn, convW));   // [channels, count]
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
                step.EndFused(stackalloc[] { convIn, q4, k4, v4, g, beta });

                // convIn is a fresh contiguous [convRows + count, channels]: its last rows are the new window.
                if (!keep)
                    step.Effect(step.Op(GraphOp.Cpy,
                        step.View2(convIn, convRows, channels, (convRows + count) * f, count * f), window));

                // A verify window keeps the state after each of its rows: the recurrence writes its whole
                // result into the request's verify buffer in place, the live state stays the one before the
                // window, and the commit copies the accepted prefix's state into place.
                long stateBytes = stateFloats * f;
                // The verify buffer binds before the recurrence, so an executor can write the window's result there in place.
                int verify = -1, cap = 0;
                if (keep)
                {
                    if (state.EnsureSpecBuffers(count)) NoteGraphBindingsChanged();
                    cap = state.SnapshotCapacity;
                    verify = step.State(state.Verify, GraphFormat.F32, cap * (state.RowFloats + stateFloats));
                }
                int gdn = step.Op(GraphOp.GatedDeltaNet, stackalloc[] { q4, k4, v4, g, beta, s4 },
                    stackalloc long[] { keep ? count : 1 });
                int scores = step.View2(gdn, v, count, v * f, 0);
                int newState = step.Op(GraphOp.View, stackalloc[] { gdn },
                    stackalloc long[] { 4, dk, dv, vHeads, 1, dk * f, dk * dv * f, dk * dv * vHeads * f, v * count * f });
                if (keep)
                {
                    long total = v * count + stateFloats * count;
                    int fallback = step.Op(GraphOp.Cpy, step.View2(gdn, total, 1, total * f, 0), step.View2(verify, total, 1, total * f, 0));
                    step.Alias(gdn, verify, 0, fallback);
                    int hist = step.State(state.ConvHistory, GraphFormat.F32, convRows + cap, channels);
                    step.Effect(step.Op(GraphOp.Cpy, convIn,
                        step.View2(hist, convRows + count, channels, (convRows + cap) * f, 0)));
                }
                else
                {
                    int copy = step.Op(GraphOp.Cpy, newState, s4);
                    if (viaBacking && count == 1)
                        // [outputs | state] of one row is exactly the backing: the recurrence updates the
                        // state in place, as Qwen35Model's fused decode does on Metal.
                        step.Alias(gdn, backing, 0, copy);
                    else
                        step.Effect(copy);
                }

                int y = step.Reshape(step.Cont(scores), dv, vHeads, count), zh = step.Reshape(z, dv, vHeads, count);
                bool sigmoid = MixerSigmoidGate;
                step.BeginFused(GraphFusion.GatedRmsNorm, stackalloc[] { y, zh, ssmNorm }, stackalloc long[] { sigmoid ? 1 : 0 }, stackalloc[] { eps });
                int gated = step.Op(GraphOp.Mul, step.RmsNorm(y, ssmNorm, eps), step.Op(sigmoid ? GraphOp.Sigmoid : GraphOp.Silu, zh));
                step.EndFused(stackalloc[] { gated });
                if (mixer) step.EndFused(stackalloc[] { gated });
                int part = step.Reshape(gated, v, count);
                output = output < 0 ? part : step.Op(GraphOp.Concat, stackalloc[] { output, part }, stackalloc long[] { 1 });
            }
            return step.MatMul(LayerName(layer, "ssm_out.weight"), output);
        }
    }
}
