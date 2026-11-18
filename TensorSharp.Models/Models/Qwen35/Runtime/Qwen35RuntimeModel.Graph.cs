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

        /// <summary>Tests: a step's query/key prep and cache writes run as their ops, not as a
        /// <see cref="GraphFusion.AttentionPrep"/> region.</summary>
        internal bool AttentionPrepRegionDisabledForTests { get; set; }

        /// <summary>
        /// A step's query/key prep (<see cref="EmitQueryKey"/>'s norms and rotary on the copies of q and k) and its K/V cache writes
        /// as one region (<see cref="GraphFusion.AttentionPrep"/>), which an executor may run as one kernel (direct CUDA, on the
        /// direct engines' ts_q35_attn_prep / ts_q4e_attn_prep layout), decode rows, verify windows and prompt chunks alike. The ops
        /// are about seven launches a layer and step - the queries' copy out of the gated projection, the two weighted norms, the
        /// two rotaries and the two cache writes, a prompt chunk's after copies that regroup K and V head-first: for a decode token
        /// of Qwen3.8 Flash Next's 12 attention layers ~170 us on an A40 (UD-Q2_K_XL, short context) where the engine's one kernel
        /// took 36; over a 4K prompt of Qwen3.8-27B on an RTX PRO 4000 the norms, rotaries, strided copies and cache writes took
        /// ~28 ms (with the layers' other norms and copies) where the engine's kernel took 10.6. Text positions only (a media step's
        /// rotary is multi-axis), and F16 or F32 caches, the ones the kernel writes: a declined region runs its ops with each norm's
        /// weight a product of its own, a launch more than the step without the region. A q8_0 cache's too where the executor
        /// takes q8_0 in its own kernels alone (direct CUDA, whose prep kernel quantizes the rows as its row writes do); an
        /// executor that takes block caches in every view (GGML) keeps the graphs it ran.
        /// </summary>
        protected override bool TakesAttentionPrep(GraphStep step, int layer)
            => !AttentionPrepRegionDisabledForTests && step.PositionTable < 0 && step.Positions >= 0 && _rotary != null
               && (KvCacheGraphType is GraphFormat.F16 or GraphFormat.F32
                   || (KvCacheGraphType == GraphFormat.Q8_0 && GraphExecutor is { Capabilities.QuantizedKvCache: false }));

        protected override void BeginAttentionPrep(GraphStep step, int layer, in AttentionPrepOperands prep)
        {
            RotarySpec rope = _rotary;
            step.BeginFused(GraphFusion.AttentionPrep,
                stackalloc[]
                {
                    prep.Queries, prep.Keys, prep.Values, step.Weight(LayerName(layer, "attn_q_norm.weight")),
                    step.Weight(LayerName(layer, "attn_k_norm.weight")), prep.KCache, prep.VCache, step.Positions, prep.Rows,
                },
                stackalloc long[]
                {
                    prep.Heads, prep.KvHeads, prep.HeadDim, prep.ValueDim, rope.Dims, rope.NeoX ? 2 : 0, rope.OriginalContext,
                    AttentionReadsQueriesAlone(step, layer) ? 1 : 0,
                },
                stackalloc[] { Config.Eps, rope.FreqBase, rope.FreqScale, rope.ExtFactor, rope.AttnFactor, rope.BetaFast, rope.BetaSlow });
        }

        /// <summary>Whether only the attention reads <paramref name="layer"/>'s rotated queries in this step, through its head-major
        /// view (an executor may then lay them out so; <see cref="GraphFusion.AttentionPrep"/>); a family whose region reads them as
        /// rows says no there.</summary>
        protected virtual bool AttentionReadsQueriesAlone(GraphStep step, int layer) => true;

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
            // A window left the live states (and conv windows) as they were before it: none kept, nothing to set.
            if (kept == 0) return;
            // One commit graph per pipeline stage and tensor-parallel rank: each binds its own device's states (a rank's
            // hold its share of the heads).
            for (int stage = 0; stage < Mesh.Stages; stage++)
            for (int rank = 0; rank < Mesh.TensorParallel; rank++)
            {
                IntPtr key = IntPtr.Zero;
                for (int layer = 0; layer < Config.NumLayers && key == IntPtr.Zero; layer++)
                    if (_isRecurrent[layer] && OwnsLayer(layer) && Mesh.StageOf(layer) == stage
                        && ActiveRecurrentState(layer, rank) is DeviceGdnState first && first.CommitSource(rows) is { } source)
                        key = StateIdentity(source);
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

        /// <summary>
        /// The commit of a <paramref name="rows"/>-row window that kept <paramref name="kept"/> (at least one) rows, over the stage's
        /// (the rank's) recurrent states: per layer the state after the kept rows (stepped again on the live state, or copied
        /// from the window's) and the conv window after them. Where the layers' shapes agree the ops are one region
        /// (<see cref="GraphFusion.DeltaNetCommit"/>), which an executor may run as a few launches for every layer (direct CUDA):
        /// the ops are about fifteen nodes a layer, 1.1 ms of host time a verify step of Qwen3.8-27B's 48 recurrent layers.
        /// </summary>
        private GraphProgram BuildCommitProgram(int kept, int rows, int stage, int rank)
        {
            var g = new GraphProgramBuilder(record: true);
            var layers = new System.Collections.Generic.List<(int Layer, DeviceGdnState State)>();
            for (int layer = 0; layer < Config.NumLayers; layer++)
                if (_isRecurrent[layer] && OwnsLayer(layer) && Mesh.StageOf(layer) == stage
                    && ActiveRecurrentState(layer, rank) is DeviceGdnState state && state.CommitSource(rows) != null)
                    layers.Add((layer, state));
            int n = layers.Count;
            if (n == 0) return g.ToProgram(replayable: true, reorder: false);

            // Every layer's leaves come first (the region's inputs, five a layer), then its ops.
            var plan = new CommitLayer[n];
            var inputs = new int[5 * n];
            var ip = new long[8 + 3 * n];
            DeviceGdnState first = layers[0].State;
            int dk = _headKDim, dv = _headVDim;
            int KeyHeads(DeviceGdnState s) => (int)((s.Channels - (long)s.VHeads * dv) / (2L * dk));
            bool region = dk == dv && !CommitRegionDisabledForTests;
            for (int i = 0; i < n; i++)
            {
                DeviceGdnState state = layers[i].State;
                plan[i] = CommitLayerOf(g, state, kept, rows);
                region &= state.VHeads == first.VHeads && KeyHeads(state) == KeyHeads(first) && state.BackingRows == first.BackingRows
                    && state.ConvRows == first.ConvRows && state.Channels == first.Channels;
                CommitLayer c = plan[i];
                inputs[5 * i] = c.Source;
                inputs[5 * i + 1] = c.Target;
                inputs[5 * i + 2] = c.Inputs;
                inputs[5 * i + 3] = c.History;
                inputs[5 * i + 4] = c.Conv;
                ip[8 + 3 * i] = c.Steps;
                ip[9 + 3 * i] = c.From;
                ip[10 + 3 * i] = c.To;
            }
            ip[0] = dk;
            ip[1] = first.VHeads;
            ip[2] = KeyHeads(first);
            ip[3] = first.BackingRows;
            ip[4] = first.ConvRows;
            ip[5] = first.Channels;
            ip[6] = first.ConvRows + rows;   // the window wrote its conv inputs at its own pitch (convRows + rows a channel)
            ip[7] = kept;

            if (region) g.BeginFused(GraphFusion.DeltaNetCommit, inputs, ip);
            int last = -1;
            for (int i = 0; i < n; i++)
            {
                last = EmitCommitLayer(g, layers[i].State, plan[i], kept, rows);
                if (region) DeltaNetCommitRegionLayers++;
            }
            if (region) g.EndFused(stackalloc[] { last });
            foreach ((int layer, DeviceGdnState state) in layers) EmitCommitExtras(g, layer, state, kept, rows);
            return g.ToProgram(replayable: true, reorder: false);
        }

        /// <summary>A layer's part of a commit: its leaves (<see cref="Source"/> .. <see cref="Conv"/>, the history's leaf standing
        /// in for those the layer does not read) and the state's work - <see cref="Steps"/> rows stepped again from the source,
        /// 0 the source's state taken, -1 none - from <see cref="From"/> floats into the source to <see cref="To"/> floats into
        /// the target (see <see cref="GraphFusion.DeltaNetCommit"/>).</summary>
        private readonly record struct CommitLayer(int Source, int Target, int Inputs, int History, int Conv, long Steps, long From, long To);

        private CommitLayer CommitLayerOf(GraphProgramBuilder g, DeviceGdnState state, int kept, int rows)
        {
            long stateFloats = state.StateFloats;
            int cap = state.SnapshotCapacity, convRows = state.ConvRows, channels = state.Channels;
            int history = GraphLeaf(g, state.ConvHistory, convRows + cap, channels);
            int conv = GraphLeaf(g, state.Conv, convRows, channels);
            if (state.WindowInPlace(rows))
            {
                // The window left the live state as it was and saved its inputs: the kept rows step again on that state, in
                // place in the backing ([kept rows' outputs | state] ends on the state).
                int backing = GraphLeaf(g, state.Backing, state.StateOffset + stateFloats);
                int inputs = GraphLeaf(g, state.SavedInputs, state.SavedInputFloats);
                return new CommitLayer(backing, backing, inputs, history, conv, kept, state.StateOffset,
                    (state.BackingRows - kept) * state.RowFloats);
            }
            // Result layout [rows' outputs | state after the last row | state 1 row before | ...].
            int verify = GraphLeaf(g, state.Verify, cap * (state.RowFloats + stateFloats));
            long from = state.RowFloats * rows + (rows - kept) * stateFloats;
            return InPlaceRecurrentState
                ? new CommitLayer(verify, GraphLeaf(g, state.Backing, state.StateOffset + stateFloats), verify, history, conv, 0, from, state.StateOffset)
                : new CommitLayer(verify, GraphLeaf(g, state.Ssm, _headKDim, _headVDim, state.Ssm.Sizes[0]), verify, history, conv, 0, from, 0);
        }

        /// <summary>A layer's commit ops over its leaves; returns the conv window's write.</summary>
        private int EmitCommitLayer(GraphProgramBuilder g, DeviceGdnState state, in CommitLayer c, int kept, int rows)
        {
            const long f = sizeof(float);
            long stateFloats = state.StateFloats, stateBytes = stateFloats * f;
            int convRows = state.ConvRows, channels = state.Channels;
            if (state.WindowInPlace(rows))
                EmitWindowRestep(g, state, kept, c.Target, c.Inputs);
            else
            {
                int source = g.Op(GraphOp.View, stackalloc[] { c.Source }, stackalloc long[] { 1, stateFloats, 1, 1, 1, stateBytes, 0, 0, c.From * f });
                int target = InPlaceRecurrentState
                    ? g.Op(GraphOp.View, stackalloc[] { c.Target }, stackalloc long[] { 1, stateFloats, 1, 1, 1, stateBytes, 0, 0, c.To * f })
                    : g.Op(GraphOp.Reshape, stackalloc[] { c.Target }, stackalloc long[] { stateFloats });
                g.Effect(g.Op(GraphOp.Cpy, source, target));
            }
            // The window wrote its conv inputs at its own pitch (convRows + rows a channel).
            long histPitch = convRows + rows;
            int window = g.Op(GraphOp.View, stackalloc[] { c.History },
                stackalloc long[] { 2, convRows, channels, 1, 1, histPitch * f, 0, 0, kept * f });
            int write = g.Op(GraphOp.Cpy, window, c.Conv);
            g.Effect(write);
            return write;
        }

        /// <summary>
        /// The state after <paramref name="kept"/> (at least one) rows of a window that left it as it was: the recurrence over
        /// the kept rows' saved inputs on the live state, on the decode rows' kernels (the window's rounding), its result in
        /// place in the backing - [kept rows' outputs | state] ends on the state it read.
        /// </summary>
        private void EmitWindowRestep(GraphProgramBuilder g, DeviceGdnState state, int kept, int backing, int inputs)
        {
            const long f = sizeof(float);
            int dk = _headKDim, dv = _headVDim, vHeads = state.VHeads, rows = state.BackingRows;
            int kHeads = (int)((state.Channels - (long)vHeads * dv) / (2L * dk));
            long stateFloats = state.StateFloats;
            int before = g.Op(GraphOp.View, stackalloc[] { backing },
                stackalloc long[] { 4, dk, dv, vHeads, 1, dk * f, dk * dv * f, stateFloats * f, state.StateOffset * f });
            // Planar regions of `rows` tokens: q | k [rows, kHeads, dk], v [rows, vHeads, dv], gates | betas [rows, vHeads].
            long qkRegion = (long)rows * kHeads * dk * f, vRegion = (long)rows * vHeads * dv * f, gRegion = (long)rows * vHeads * f;
            int Region(long width, long heads, long at) => g.Op(GraphOp.View, stackalloc[] { inputs },
                stackalloc long[] { 4, width, heads, kept, 1, width * f, width * heads * f, width * heads * kept * f, at });
            int q = Region(dk, kHeads, 0), k = Region(dk, kHeads, qkRegion), v = Region(dv, vHeads, 2 * qkRegion);
            int gate = Region(1, vHeads, 2 * qkRegion + vRegion), beta = Region(1, vHeads, 2 * qkRegion + vRegion + gRegion);
            int gdn = g.Op(GraphOp.GatedDeltaNet, stackalloc[] { q, k, v, gate, beta, before }, stackalloc long[] { 1, 0, 1 });
            long total = state.RowFloats * kept + stateFloats, at = (rows - kept) * state.RowFloats * f;
            int fallback = g.Op(GraphOp.Cpy,
                g.Op(GraphOp.View, stackalloc[] { gdn }, stackalloc long[] { 1, total, 1, 1, 1, total * f, 0, 0, 0 }),
                g.Op(GraphOp.View, stackalloc[] { backing }, stackalloc long[] { 1, total, 1, 1, 1, total * f, 0, 0, at }));
            g.Alias(gdn, backing, at, fallback);
        }

        /// <summary>Tests: commits run as their ops, not as a <see cref="GraphFusion.DeltaNetCommit"/> region.</summary>
        internal bool CommitRegionDisabledForTests { get; set; }

        /// <summary>Tests: layers whose commit ops a compiled commit program put in a <see cref="GraphFusion.DeltaNetCommit"/>
        /// region.</summary>
        internal int DeltaNetCommitRegionLayers { get; private set; }

        /// <summary>Tests: the rows the active request's first recurrent state keeps in front of its state (above one, its
        /// verify windows leave the state as it was and the commit steps their kept rows on it); 0 without one.</summary>
        internal int WindowStateRowsForTests
        {
            get
            {
                foreach (RecurrentState state in ActiveRecurrentStates())
                    if (state is DeviceGdnState gdn) return gdn.BackingRows;
                return 0;
            }
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

            // Decode rows take the packed in-projection in one product; so do a verify window's rows where the executor's
            // delta-net regions read the four blocks where the product put them (GraphCapabilities.WindowRegionsReadPackedRows:
            // the blocks' copies were 192 a Qwen3.8-27B 8-row window). A prompt chunk takes its four row blocks as separate products, so
            // each part is contiguous (no slice copies).
            string inProj = InProjName(layer);
            bool isPacked = _quantWeights.ContainsKey(inProj) || _weights.ContainsKey(inProj)
                || (step.Rank >= 0 && (_tpQuantWeights.ContainsKey(inProj) || _tpWeights.ContainsKey(inProj)));
            bool windowViews = step.RecurrentSnapshots && GraphExecutor?.Capabilities.WindowRegionsReadPackedRows == true;
            bool split = !isPacked || (step.Rows > step.Sequences && !windowViews);
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

                int qkv, z, betaRaw, alphaRaw, z4 = -1;
                if (split)
                {
                    long col = step.First[s];
                    qkv = step.Sequences == 1 ? qkvAll : step.View2(qkvAll, channels, count, channels * f, col * channels * f);
                    z = step.Sequences == 1 ? zAll : step.View2(zAll, v, count, v * f, col * v * f);
                    betaRaw = step.Sequences == 1 ? betaAll : step.View2(betaAll, vHeads, count, vHeads * f, col * vHeads * f);
                    alphaRaw = step.Sequences == 1 ? alphaAll : step.View2(alphaAll, vHeads, count, vHeads * f, col * vHeads * f);
                }
                else if (windowViews && count > 1)
                {
                    // The window's rows a packed row apart, read where they are by the regions that take them.
                    qkv = step.View2(proj, channels, count, nb1, row0);
                    z = step.View2(proj, v, count, nb1, row0 + channels * f);
                    z4 = step.View4(proj, dv, vHeads, count, 1, dv * f, nb1, nb1 * count, row0 + channels * f);
                    betaRaw = step.View2(proj, vHeads, count, nb1, row0 + (channels + v) * f);
                    alphaRaw = step.View2(proj, vHeads, count, nb1, row0 + (channels + v + vHeads) * f);
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
                    backing = step.State(state.Backing, GraphFormat.F32, state.StateOffset + stateFloats);
                    s4 = step.View4(backing, dk, dv, vHeads, 1, dk * f, dk * dv * f, stateFloats * f, state.StateOffset * f);
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
                // window, and the commit copies the accepted prefix's state into place. Where the executor can, a
                // window writes only its rows' outputs and saves its inputs, the live state read and left as it
                // was (the direct engines' verify scan): the commit steps the kept rows on it again, in place.
                long stateBytes = stateFloats * f;
                // The verify buffer binds before the recurrence, so an executor can write the window's result there in place.
                int verify = -1, cap = 0;
                bool inPlaceWindow = keep && viaBacking && state.WindowInPlace(count);
                if (keep)
                {
                    if (state.EnsureSpecBuffers(count)) NoteGraphBindingsChanged();
                    cap = state.SnapshotCapacity;
                    if (!inPlaceWindow) verify = step.State(state.Verify, GraphFormat.F32, cap * (state.RowFloats + stateFloats));
                }
                int gdn = inPlaceWindow
                    ? step.Op(GraphOp.GatedDeltaNet,
                        stackalloc[] { q4, k4, v4, g, beta, s4, step.State(state.SavedInputs, GraphFormat.F32, state.SavedInputFloats) },
                        stackalloc long[] { 0, state.BackingRows, 1 })
                    : step.Op(GraphOp.GatedDeltaNet, stackalloc[] { q4, k4, v4, g, beta, s4 }, stackalloc long[] { keep ? count : 1 });
                int scores = step.View2(gdn, v, count, v * f, 0);
                if (keep)
                {
                    if (!inPlaceWindow)
                    {
                        long total = v * count + stateFloats * count;
                        int fallback = step.Op(GraphOp.Cpy, step.View2(gdn, total, 1, total * f, 0), step.View2(verify, total, 1, total * f, 0));
                        step.Alias(gdn, verify, 0, fallback);
                    }
                    // At the window's own pitch, one contiguous copy whatever its rows (at the capacity's pitch a window
                    // narrower than it wrote [channels] rows of a few floats each: 27 us a layer for 2- and 3-row windows
                    // of Qwen3.8 Flash Next, 1 ms a window).
                    int hist = step.State(state.ConvHistory, GraphFormat.F32, convRows + cap, channels);
                    long histPitch = convRows + count;
                    step.Effect(step.Op(GraphOp.Cpy, convIn,
                        step.View2(hist, convRows + count, channels, histPitch * f, 0)));
                }
                else
                {
                    int newState = step.Op(GraphOp.View, stackalloc[] { gdn },
                        stackalloc long[] { 4, dk, dv, vHeads, 1, dk * f, dk * dv * f, dk * dv * vHeads * f, v * count * f });
                    int copy = step.Op(GraphOp.Cpy, newState, s4);
                    if (viaBacking && count == 1)
                        // [outputs | state] of one row ends the backing: the recurrence updates the
                        // state in place, as Qwen35Model's fused decode does on Metal.
                        step.Alias(gdn, backing, (state.BackingRows - 1) * state.RowFloats * f, copy);
                    else
                        step.Effect(copy);
                }

                int y = step.Reshape(step.Cont(scores), dv, vHeads, count), zh = z4 >= 0 ? z4 : step.Reshape(z, dv, vHeads, count);
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
