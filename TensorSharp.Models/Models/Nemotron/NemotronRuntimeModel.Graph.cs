// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>Nemotron-H's pieces of a step's graph (see <see cref="NemotronRuntimeModel"/>).</summary>
    public sealed partial class NemotronRuntimeModel
    {
        private const long F = sizeof(float);

        /// <summary>The Mamba-2 mixer binds its states as leaves and adds constant inputs only; attention and the experts follow
        /// positions through the step's standard inputs.</summary>
        protected override bool DecodeStepsAdvance => true;

        // ---------------------------------------------------------------- Mamba-2

        /// <summary>
        /// Mamba-2 over the normed rows [hidden, rows] of every sequence of the step, each on its own state: the in-projection for
        /// all rows at once, then per sequence the causal conv over [its window | its rows] (bias, SiLU), ggml's selective scan
        /// (dt + bias, A), the D skip, the silu(z) gate and the grouped RMSNorm, with the conv window and the scan state written
        /// back (a verify window keeps them aside, <see cref="CommitRecurrentPrefix"/>). Returns this rank's partial
        /// [hidden, rows] of <c>ssm_out</c>.
        /// </summary>
        protected override int EmitRecurrentMixer(GraphStep step, int layer, int normed)
        {
            int degree = step.Degree;
            int heads = _nHead / degree, groups = _nGroup / degree, hd = _ssmHeadDim, ds = _dState;
            long inner = (long)heads * hd, bc = (long)groups * ds, channels = inner + 2 * bc, packed = inner + channels + heads;
            int convRows = _convKernel - 1;
            float eps = Config.Eps;
            Mamba2Constants k = _mamba[layer][step.RankIndex];
            int convW = step.TensorWeight(k.ConvWeight, _convKernel, channels);
            int convB = k.ConvBias != null ? step.TensorWeight(k.ConvBias, channels) : -1;
            int dtBias = step.TensorWeight(k.DtBias, heads);
            int a = step.TensorWeight(k.A, 1, heads);
            int d = k.D != null ? step.TensorWeight(k.D, 1, heads) : -1;
            long groupWidth = inner / groups;
            int norm = k.Norm != null ? step.TensorWeight(k.Norm, groupWidth, groups) : -1;
            int ids = step.ConstantInputI32(new[] { 0 }, 1);

            int proj = step.MatMul(LayerName(layer, "ssm_in.weight"), normed);                     // [z | xBC | dt, rows]
            long pitch = packed * F;
            var parts = new int[step.Sequences];
            for (int s = 0; s < step.Sequences; s++)
            {
                int count = step.Counts[s];
                long row0 = step.First[s] * pitch;
                var state = (Mamba2State)step.SequenceState(s, layer);
                bool keep = step.RecurrentSnapshots && count > 1;
                if (keep && state.EnsureSpecBuffers(count)) NoteGraphBindingsChanged();

                int z = step.Cont(step.View2(proj, inner, count, pitch, row0));                     // [inner, count]
                int xbc = step.View2(proj, channels, count, pitch, row0 + inner * F);              // [channels, count]
                int dt = step.View2(proj, heads, count, pitch, row0 + (inner + channels) * F);     // [heads, count]

                // The causal conv over [the window (oldest first) | the rows], channels' time innermost.
                int window = step.State(state.Conv, GraphFormat.F32, convRows, channels);
                int convIn = step.Op(GraphOp.Concat, stackalloc[] { window, step.Transpose(xbc) }, stackalloc long[] { 0 });   // [convRows + count, channels]
                int conv = step.Op(GraphOp.SsmConv, convIn, convW);                                 // [channels, count]
                if (convB >= 0) conv = step.Op(GraphOp.Add, conv, convB);
                conv = step.Op(GraphOp.Silu, conv);
                if (keep)
                {
                    // A verify window leaves the live window alone; the commit takes it from the window's conv inputs.
                    int hist = step.State(state.ConvHistory, GraphFormat.F32, convRows + state.SnapshotCapacity, channels);
                    step.Effect(step.Op(GraphOp.Cpy, convIn, step.View2(hist, convRows + count, channels, (convRows + count) * F, 0)));
                }
                else
                {
                    step.Effect(step.Op(GraphOp.Cpy, step.View2(convIn, convRows, channels, (convRows + count) * F, count * F), window));
                }

                long token = channels * F, all = channels * count * F;
                int x = step.View4(conv, hd, heads, count, 1, hd * F, token, all, 0);
                int b = step.View4(conv, ds, groups, count, 1, ds * F, token, all, inner * F);
                int c = step.View4(conv, ds, groups, count, 1, ds * F, token, all, (inner + bc) * F);
                int dtb = step.Reshape(step.Op(GraphOp.Add, dt, dtBias), heads, count, 1);         // [heads, count, 1]
                long stateFloats = (long)ds * hd * heads;
                int stLeaf = step.State(state.Ssm, GraphFormat.F32, ds, hd, heads);
                int st = step.Reshape(stLeaf, ds, hd, heads, 1);
                int scan = step.Op(GraphOp.SsmScan, stackalloc[] { st, x, dtb, a, b, c, ids }, stackalloc long[] { keep ? count : 1 });
                long yFloats = inner * count;
                if (keep)
                {
                    // Every row's state, newest first: the commit takes the accepted prefix's.
                    int verify = step.State(state.Verify, GraphFormat.F32, state.SnapshotCapacity * stateFloats);
                    long floats = count * stateFloats;
                    step.Effect(step.Op(GraphOp.Cpy, step.View2(scan, floats, 1, floats * F, yFloats * F), step.View2(verify, floats, 1, floats * F, 0)));
                }
                else
                {
                    // The new state into the live one, as views of [D] a sequence right after the scan: the copy ggml-cuda folds
                    // into its scan kernel (ggml_cuda_try_ssm_scan_cache_fusion).
                    int next = step.View2(scan, stateFloats, 1, stateFloats * F, yFloats * F);
                    step.Effect(step.Op(GraphOp.Cpy, next, step.View2(stLeaf, stateFloats, 1, stateFloats * F, 0)));
                }

                int y = step.View4(scan, hd, heads, count, 1, hd * F, inner * F, yFloats * F, 0);  // [hd, heads, count]
                if (d >= 0) y = step.Op(GraphOp.Add, y, step.Op(GraphOp.Mul, x, d));
                y = step.Reshape(step.Cont(y), inner, count);
                int gated = step.Op(GraphOp.Mul, y, step.Op(GraphOp.Silu, z));
                if (norm >= 0)
                {
                    int grouped = step.Reshape(gated, groupWidth, groups, count);
                    gated = step.Reshape(step.RmsNorm(grouped, norm, eps), inner, count);
                }
                parts[s] = gated;
            }
            int output = parts[0];
            for (int s = 1; s < parts.Length; s++)
                output = step.Op(GraphOp.Concat, stackalloc[] { output, parts[s] }, stackalloc long[] { 1 });
            return step.MatMul(LayerName(layer, "ssm_out.weight"), output);
        }

        // ---------------------------------------------------------------- speculation: the accepted prefix's states

        // Commit programs replay per (verify buffer, kept, rows, stage, rank): only copies, no inputs.
        private readonly Dictionary<(IntPtr, int, int, int, int), IGraphPlan> _commitGraphs = new();

        protected override void OnGraphCacheCleared()
        {
            foreach (IGraphPlan g in _commitGraphs.Values) g.Dispose();
            _commitGraphs.Clear();
        }

        /// <summary>Set the active request's Mamba-2 states and conv windows to those after <paramref name="kept"/> rows of the
        /// last <paramref name="rows"/>-row verify window, from its verify buffers (0: the live states are still the ones
        /// before the window).</summary>
        protected override void CommitRecurrentPrefix(int kept, int rows)
        {
            if (kept <= 0) return;
            for (int stage = 0; stage < Mesh.Stages; stage++)
            for (int rank = 0; rank < Mesh.TensorParallel; rank++)
            {
                var layers = new List<(int Layer, Mamba2State State)>();
                for (int layer = 0; layer < Config.NumLayers; layer++)
                    if (_block[layer] == Block.Mamba2 && OwnsLayer(layer) && Mesh.StageOf(layer) == stage
                        && ActiveRecurrentState(layer, rank) is Mamba2State state && state.Verify != null && state.SnapshotCapacity >= rows)
                        layers.Add((layer, state));
                if (layers.Count == 0) continue;
                IntPtr key = StateIdentity(layers[0].State.Verify);
                EnterGraphStage(stage);
                try
                {
                    if (!_commitGraphs.TryGetValue((key, kept, rows, stage, rank), out IGraphPlan g))
                    {
                        g = CompileForDevice(BuildCommitProgram(layers, kept, rows), stage, rank);
                        _commitGraphs[(key, kept, rows, stage, rank)] = g;
                    }
                    if (!g.Run(ReadOnlySpan<int>.Empty, ReadOnlySpan<IntPtr>.Empty, ReadOnlySpan<long>.Empty))
                        throw new InvalidOperationException($"{Config.Architecture} Mamba-2 state commit failed: {g.LastError}");
                }
                finally
                {
                    EnterGraphStage(0);
                }
            }
        }

        /// <summary>Per layer: the state after <paramref name="kept"/> rows (slot rows - kept of the window's) into the live state,
        /// and the conv window after them (the window's conv inputs from row kept on) into the live window.</summary>
        private GraphProgram BuildCommitProgram(List<(int Layer, Mamba2State State)> layers, int kept, int rows)
        {
            var g = new GraphProgramBuilder(record: true);
            foreach ((int _, Mamba2State state) in layers)
            {
                long stateFloats = state.StateFloats;
                int convRows = state.ConvRows, channels = state.Channels;
                int verify = GraphLeaf(g, state.Verify, state.SnapshotCapacity * stateFloats);
                // The leaves as the steps bind them (the same extents), flattened where a copy reads them so.
                int live = g.Op(GraphOp.Reshape, stackalloc[] { GraphLeaf(g, state.Ssm, state.Ssm.Sizes[2], state.Ssm.Sizes[1], state.Ssm.Sizes[0]) },
                    stackalloc long[] { stateFloats });
                int history = GraphLeaf(g, state.ConvHistory, convRows + state.SnapshotCapacity, channels);
                int conv = GraphLeaf(g, state.Conv, convRows, channels);
                int source = g.Op(GraphOp.View, stackalloc[] { verify },
                    stackalloc long[] { 1, stateFloats, 1, 1, 1, stateFloats * F, 0, 0, (rows - kept) * stateFloats * F });
                g.Effect(g.Op(GraphOp.Cpy, source, live));
                long pitch = convRows + rows;
                int window = g.Op(GraphOp.View, stackalloc[] { history },
                    stackalloc long[] { 2, convRows, channels, 1, 1, pitch * F, 0, 0, kept * F });
                g.Effect(g.Op(GraphOp.Cpy, window, conv));
            }
            return g.ToProgram(replayable: true, reorder: false);
        }

        /// <summary>A state tensor's bytes where the executor binds host memory, else a stable identity.</summary>
        private IntPtr StateIdentity(Tensor t) => GraphExecutor.Capabilities.BindsHostPointers
            ? TensorComputePrimitives.GetStoragePointer(t)
            : GraphIdentity.Of(t.Storage, t.StorageOffset);

        private int GraphLeaf(GraphProgramBuilder g, Tensor t, params long[] ne)
            => g.Leaf(GraphLeafRole.State, GraphFormat.F32, ne, StateIdentity(t), t.Storage.ByteLength, handle: t);

        // ---------------------------------------------------------------- feed-forward

        /// <summary>A feed-forward layer's block over its normed rows [hidden, rows]: the routed experts, or the dense ReLU^2 FFN.</summary>
        protected override int EmitFeedForward(GraphStep step, int layer, int x)
            => _layerExperts[layer] > 0 ? EmitExperts(step, layer, x) : EmitReluSquaredFfn(step, layer, x, "ffn_up.weight", "ffn_down.weight");

        /// <summary>down(relu(up x)^2).</summary>
        private int EmitReluSquaredFfn(GraphStep step, int layer, int x, string up, string down)
            => step.MatMul(LayerName(layer, down), ReluSquared(step, step.MatMul(LayerName(layer, up), x)));

        private static int ReluSquared(GraphStep step, int x) => step.Op(GraphOp.Sqr, step.Op(GraphOp.Relu, x));

        /// <summary>
        /// The routed branch over <paramref name="x"/> [hidden, rows]: the router's sigmoid probabilities over the hidden rows,
        /// the top k by probability + <c>exp_probs_b</c>, their unbiased probabilities as weights (over their sum, at least
        /// 2^-14, when renormalized; times <c>expert_weights_scale</c>; times each expert's NVFP4 down scale); the experts over
        /// the latent rows where the layer has a bottleneck: relu(up)^2 (each expert's up scale first), down, the weighted sum,
        /// back through the latent projection; plus the shared expert over the hidden rows.
        /// </summary>
        private int EmitExperts(GraphStep step, int layer, int x)
        {
            int rows = step.Rows, experts = _layerExperts[layer], k = _layerExpertsUsed[layer];
            string latentIn = LatentInName(layer), latentOut = LatentOutName(layer);
            bool latent = HasWeight(latentIn);

            // The shared expert reads only x and is read only at the end: a region an executor may run beside the routed ones.
            int shared = -1;
            if (HasWeight(LayerName(layer, "ffn_up_shexp.weight")))
            {
                step.BeginFused(GraphFusion.ConcurrentBranch, stackalloc[] { x });
                shared = EmitReluSquaredFfn(step, layer, x, "ffn_up_shexp.weight", "ffn_down_shexp.weight");
                step.EndFused(stackalloc[] { shared });
            }

            int logits = step.MatMul(LayerName(layer, "ffn_gate_inp.weight"), x);                                     // [E, rows]
            int probs = step.Op(GraphOp.Sigmoid, logits);
            string biasName = HasWeight(LayerName(layer, "exp_probs_b.bias")) ? LayerName(layer, "exp_probs_b.bias") : LayerName(layer, "exp_probs_b");
            int selection = HasWeight(biasName) ? step.Op(GraphOp.Add, probs, step.Weight(biasName)) : probs;
            int forced = EmitForcedRoutes(step, layer, k, experts);                                                 // tests
            int selected = forced >= 0 ? forced : step.Op(GraphOp.TopK, stackalloc[] { selection }, stackalloc long[] { k });   // [K, rows]
            EmitDeviceRoutes(step, layer, selected, selection, k, experts);                                         // tests
            int weights = step.Reshape(step.Op(GraphOp.GetRows, step.Reshape(probs, 1, experts, rows), selected), k, rows);
            if (_expertWeightsNorm)
            {
                int sum = step.Op(GraphOp.Clamp, stackalloc[] { step.Op(GraphOp.SumRows, weights) }, default,
                    stackalloc[] { 6.103515625e-5f, float.PositiveInfinity });                                       // [1, rows]
                weights = step.Op(GraphOp.Div, weights, sum);
            }
            if (_expertWeightsScale != 1f)
                weights = step.Op(GraphOp.Scale, stackalloc[] { weights }, default, stackalloc[] { _expertWeightsScale });
            weights = step.Reshape(weights, 1, k, rows);
            int downScale = ExpertScales(step, layer, "ffn_down_exps.scale", selected, k, rows);
            if (downScale >= 0) weights = step.Op(GraphOp.Mul, weights, downScale);

            int input = latent ? step.MatMul(latentIn, x) : x;
            int width = (int)ExpertInputWidth(layer);
            int upStack = step.Experts(LayerName(layer, "ffn_up_exps.weight")), downStack = step.Experts(LayerName(layer, "ffn_down_exps.weight"));
            int upScale = ExpertScales(step, layer, "ffn_up_exps.scale", selected, k, rows);
            int output;
            if (IsExpertLayerOffloaded(layer) && GraphExecutor.Capabilities.HostExperts)
            {
                // The experts stay in host memory (--n-cpu-moe / --cpu-moe): the program pauses for them. One projection,
                // relu(s up)^2 (the up stack in the gate slot), each route's up scale s applied before the square.
                output = upScale >= 0
                    ? step.Op(GraphOp.MoeHostExperts, stackalloc[] { input, selected, weights, upStack, upStack, downStack, upScale },
                        stackalloc long[] { layer, 3 /* relu(up)^2 */, 0, 1 /* srcs[6]: up scales */ })
                    : step.Op(GraphOp.MoeHostExperts, stackalloc[] { input, selected, weights, upStack, upStack, downStack },
                        stackalloc long[] { layer, 3 /* relu(up)^2 */ });
            }
            else
            {
                int up = step.Op(GraphOp.MulMatId, upStack, step.Reshape(input, width, 1, rows), selected);              // [ff, K, rows]
                if (upScale >= 0) up = step.Op(GraphOp.Mul, up, upScale);
                int act = ReluSquared(step, up);
                int down = step.Op(GraphOp.MulMatId, downStack, act, selected);                                        // [width, K, rows]
                down = step.Op(GraphOp.Mul, down, weights);
                long rowBytes = width * F;
                output = -1;
                for (int j = 0; j < k; j++)
                {
                    int slice = step.View2(down, width, rows, rowBytes * k, j * rowBytes);
                    output = output < 0 ? slice : step.Op(GraphOp.Add, output, slice);
                }
                output = step.Cont(output);
            }
            if (latent) output = step.MatMul(latentOut, output);
            return shared >= 0 ? step.Op(GraphOp.Add, output, shared) : output;
        }

        /// <summary>The routed experts' input width: the latent bottleneck's, else the hidden size.</summary>
        private long ExpertInputWidth(int layer)
            => _stackedExpertWeights.TryGetValue(LayerName(layer, "ffn_up_exps.weight"), out StackedExpertWeights up) ? up.PerExpertNe0 : Config.HiddenSize;

        /// <summary>The selected experts' NVFP4 outer scales [1, K, rows] from a per-expert F32 vector, or -1 where the layer has none.</summary>
        private int ExpertScales(GraphStep step, int layer, string suffix, int selected, int k, int rows)
        {
            string name = LayerName(layer, suffix);
            if (!_weights.TryGetValue(name, out Tensor vector)) return -1;
            int table = step.Reshape(step.Weight(name), 1, vector.ElementCount());                                     // [1, E]
            int ids = step.Reshape(step.Op(GraphOp.Cont, selected), (long)k * rows);                                 // the top-k views its sort
            return step.Reshape(step.Op(GraphOp.GetRows, table, ids), 1, k, rows);
        }
    }
}
