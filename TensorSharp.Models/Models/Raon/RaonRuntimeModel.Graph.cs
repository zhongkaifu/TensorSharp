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
    public sealed partial class RaonRuntimeModel
    {
        private const long F = sizeof(float);

        // ---------------------------------------------------------------- the residual streams

        /// <summary>The embedding's hc copies (stream c at elements [c·n, (c+1)·n) of a row); the step's per-layer embeddings
        /// <c>(RMSNorm(per_layer_model_proj · e) + sqrt(d) · per_layer_token_embd[t]) / sqrt(2)</c> are made here too.</summary>
        protected override int EmitStreamInput(GraphStep step, int embedded)
        {
            RaonSpec p = Spec;
            int rows = step.Rows;
            int proj = step.MatMul("per_layer_model_proj.weight", embedded);                                   // [L·d, rows]
            int normed = step.RmsNorm(step.Reshape(proj, p.PleDim, (long)p.Layers * rows), "per_layer_proj_norm.weight");
            float[] gathered = GatherPleRows(rows);
            int table = step.InputF32(gathered, p.PleWidth, rows);
            // A decode step run again one position on reads the new token's rows into the same input.
            step.RefillOnAdvance(table, () =>
            {
                GatherPleRows(rows, gathered);
                return true;
            });
            int scaled = step.Op(GraphOp.Scale, stackalloc[] { table }, default, stackalloc[] { MathF.Sqrt(p.PleDim) });
            int sum = step.Op(GraphOp.Add, step.Reshape(normed, p.PleWidth, rows), scaled);
            _pleNode = step.Op(GraphOp.Scale, stackalloc[] { sum }, default, stackalloc[] { 1f / MathF.Sqrt(2f) });
            _pleStep = step;

            int stream = embedded;
            for (int c = 1; c < p.Hc; c++)
                stream = step.Op(GraphOp.Concat, stackalloc[] { stream, embedded }, stackalloc long[] { 0 });
            return stream;
        }

        /// <summary>
        /// A Raon-mHC site's read: the 17 mixes <c>hc_fn · rms(streams) + hc_base</c> (an unweighted RMS over all hc·n values),
        /// the pre weights sigmoid(mixes[0..hc)) summing the streams into the block's input (then the layer's norm), and as the
        /// context the post weights sigmoid(mixes[hc..2hc)) and the stream mix <c>comb = e / (1 + relu(max(e ⊙ r))) + c0</c>,
        /// <c>e = K · mixes</c>, from the site's chart.
        /// </summary>
        protected override int EmitBlockInput(GraphStep step, int layer, int stream, BlockSite site, out int context)
        {
            RaonSpec p = Spec;
            int rows = step.Rows, hc = p.Hc, mix = p.MixWidth, cells = hc * hc;
            long n = p.Hidden;
            string part = site == BlockSite.Mixer ? "attn" : "ffn";
            int flat = step.Cont(stream);
            int xn = step.Op(GraphOp.RmsNorm, stackalloc[] { flat }, default, stackalloc[] { p.HcEps });
            int mixes = step.Op(GraphOp.Add, step.MatMul(LayerName(layer, $"hc_{part}_fn.weight"), xn),
                step.Weight(LayerName(layer, $"hc_{part}_base.weight")));                                       // [mix, rows]
            int pre = step.Op(GraphOp.Sigmoid, step.Cont(step.View2(mixes, hc, rows, mix * F, 0)));
            int post = step.Op(GraphOp.Sigmoid, step.Cont(step.View2(mixes, hc, rows, mix * F, hc * F)));
            if (p.Magnitude != 1f) post = step.Op(GraphOp.Scale, stackalloc[] { post }, default, stackalloc[] { p.Magnitude });

            (Tensor kc, Tensor rc, Tensor c0c) = _charts[(layer, site == BlockSite.Ffn)];
            int e = step.Op(GraphOp.MulMat, step.TensorWeight(kc, mix, cells), mixes);                        // [cells, rows]
            int scaled = step.Op(GraphOp.Mul, e, step.TensorWeight(rc, cells));
            int most = RowMax(step, scaled, cells, rows);                                                     // [1, rows]
            int rho = step.Op(GraphOp.ScaleBias, stackalloc[] { step.Op(GraphOp.Relu, most) }, default, stackalloc[] { 1f, 1f });
            int comb = step.Op(GraphOp.Add, step.Op(GraphOp.Div, e, rho), step.TensorWeight(c0c, cells));     // [dst + hc·src, rows]
            context = step.Op(GraphOp.Concat, stackalloc[] { post, comb }, stackalloc long[] { 0 });           // [hc + cells, rows]

            int weighted = step.Op(GraphOp.Mul, step.Reshape(flat, n, hc, rows), step.Reshape(pre, 1, hc, rows));
            int read = SumStreams(step, weighted, rows);
            return step.RmsNorm(read, LayerName(layer, site == BlockSite.Mixer ? "attn_norm.weight" : "ffn_norm.weight"));
        }

        /// <summary>
        /// A site's write-back: <c>stream[dst] = post[dst] · out + Σ_src comb[dst, src] · stream[src]</c>; after the feed-forward
        /// the layer's per-layer embedding joins every stream.
        /// </summary>
        protected override int EmitBlockOutput(GraphStep step, int layer, int stream, BlockSite site, int context, int output)
        {
            RaonSpec p = Spec;
            int rows = step.Rows, hc = p.Hc, cells = hc * hc;
            long n = p.Hidden, width = p.StreamFloats, ctx = hc + cells;
            int post = step.Cont(step.View2(context, hc, rows, ctx * F, 0));
            int comb = step.Cont(step.View2(context, cells, rows, ctx * F, hc * F));
            // [src, n, rows] x [src, dst, rows] -> [n, dst, rows]
            int streams = step.Cont(step.Permute(step.Reshape(step.Cont(stream), n, hc, rows), 1, 0, 2, 3));
            int mixT = step.Cont(step.Permute(step.Reshape(comb, hc, hc, rows), 1, 0, 2, 3));
            int mixed = step.Op(GraphOp.MulMat, streams, mixT);
            // post ⊗ out as one-term products: [1, n, rows] x [1, hc, rows] -> [n, hc, rows]
            int written = step.Op(GraphOp.MulMat, step.Reshape(step.Cont(output), 1, n, rows), step.Reshape(post, 1, hc, rows));
            int result = step.Reshape(step.Op(GraphOp.Add, mixed, written), width, rows);
            return site == BlockSite.Ffn ? EmitPerLayerInput(step, layer, result) : result;
        }

        /// <summary>The per-layer embedding of <paramref name="layer"/>: <c>g = gelu_tanh(inp_gate · Σ streams) ⊙ ple_l</c>
        /// (the converter folded the streams' mean into inp_gate), then <c>RMSNorm(proj · g)</c> added to every stream.</summary>
        private int EmitPerLayerInput(GraphStep step, int layer, int stream)
        {
            RaonSpec p = Spec;
            int rows = step.Rows;
            long n = p.Hidden;
            int sum = SumStreams(step, step.Reshape(stream, n, p.Hc, rows), rows);
            int gate = GeluTanh(step, step.MatMul(LayerName(layer, "inp_gate.weight"), sum));                 // [d, rows]
            int ple = step.Cont(step.View2(PleRows(step), p.PleDim, rows, p.PleWidth * F, (long)layer * p.PleDim * F));
            int g = step.Op(GraphOp.Mul, gate, ple);
            int delta = step.RmsNorm(step.MatMul(LayerName(layer, "proj.weight"), g), LayerName(layer, "post_norm.weight"));
            int added = step.Op(GraphOp.Add, step.Reshape(stream, n, p.Hc, rows), step.Reshape(delta, n, 1, rows));
            return step.Reshape(added, p.StreamFloats, rows);
        }

        /// <summary><c>0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))</c> from exact ops: ggml-cpu's GELU reads an fp16 table.</summary>
        private static int GeluTanh(GraphStep step, int x)
        {
            int cube = step.Op(GraphOp.Mul, step.Op(GraphOp.Mul, x, x), x);
            int inner = step.Op(GraphOp.Add, x, step.Op(GraphOp.Scale, stackalloc[] { cube }, default, stackalloc[] { 0.044715f }));
            int t = step.Op(GraphOp.Tanh, step.Op(GraphOp.Scale, stackalloc[] { inner }, default, stackalloc[] { MathF.Sqrt(2f / MathF.PI) }));
            return step.Op(GraphOp.Mul, x, step.Op(GraphOp.ScaleBias, stackalloc[] { t }, default, stackalloc[] { 0.5f, 0.5f }));
        }

        /// <summary>Whole-layer stages hand the step's per-layer embeddings [layers x ple, rows] on beside the hc streams: the
        /// first stage makes them from the embedded rows (<see cref="EmitStreamInput"/>), every layer reads its own.</summary>
        protected override int StageCarryWidth => Spec.PleWidth;

        protected override int EmitStageCarry(GraphStep step) => PleRows(step);

        protected override void TakeStageCarry(GraphStep step, int carry)
        {
            _pleNode = carry;
            _pleStep = step;
        }

        /// <summary>The mean of the streams, then the output norm.</summary>
        protected override int EmitStreamOutput(GraphStep step, int stream)
            => step.RmsNorm(EmitCaptureRow(step, stream), "output_norm.weight");

        /// <summary>The mean of the hc streams [n, rows].</summary>
        protected override int EmitCaptureRow(GraphStep step, int stream)
        {
            RaonSpec p = Spec;
            int sum = SumStreams(step, step.Reshape(step.Cont(stream), p.Hidden, p.Hc, step.Rows), step.Rows);
            return step.Op(GraphOp.Scale, stackalloc[] { sum }, default, stackalloc[] { 1f / p.Hc });
        }

        /// <summary>x [n, hc, rows] summed over its streams: [n, rows].</summary>
        private int SumStreams(GraphStep step, int x, int rows)
        {
            RaonSpec p = Spec;
            long n = p.Hidden, width = p.StreamFloats;
            int sum = step.View2(x, n, rows, width * F, 0);
            for (int c = 1; c < p.Hc; c++)
                sum = step.Op(GraphOp.Add, sum, step.View2(x, n, rows, width * F, c * n * F));
            return sum;
        }

        /// <summary>The largest of each column's <paramref name="width"/> values (a power of two): [1, rows], halving with
        /// max(a, b) = b + relu(a - b).</summary>
        private static int RowMax(GraphStep step, int x, int width, int rows)
        {
            int cur = x;
            for (int w = width; w > 1; w /= 2)
            {
                int h = w / 2;
                int a = step.Cont(step.View2(cur, h, rows, w * F, 0));
                int b = step.Cont(step.View2(cur, h, rows, w * F, h * F));
                cur = step.Op(GraphOp.Add, b, step.Op(GraphOp.Relu, step.Op(GraphOp.Sub, a, b)));
            }
            return cur;
        }

        // ---------------------------------------------------------------- the feed-forward

        /// <summary>SwiGLU over the fused <c>ffn_up</c> = [gate | up].</summary>
        protected override int EmitFeedForward(GraphStep step, int layer, int x)
        {
            string up = LayerName(layer, "ffn_up.weight"), down = LayerName(layer, "ffn_down.weight");
            bool region = Unscaled(up) && Unscaled(down);
            if (region) step.BeginFused(GraphFusion.SwigluFfn, stackalloc[] { x, step.Weight(up), step.Weight(down) });
            int y = step.MatMul(down, step.Op(GraphOp.Swiglu, step.MatMul(up, x)));
            if (region) step.EndFused(stackalloc[] { y });
            return y;
        }

        private bool Unscaled(string name) => _quantWeights.TryGetValue(name, out QuantizedWeight q) && q.Scale == 1.0f;

        // ---------------------------------------------------------------- gated MLA (NoPE, absorbed)

        /// <summary>
        /// Queries [latent + extra, heads, rows]: <c>q_b · RMSNorm(q_a · x)</c>, its nope part taken into the latent through
        /// <c>attn_k_b</c>; the one key [latent + extra, 1, rows]: <c>[RMSNorm(latent) | extra]</c> of <c>attn_kv_a_mqa · x</c>;
        /// the value its latent part. No rotary anywhere.
        /// </summary>
        protected override bool EmitAttentionProjections(GraphStep step, int layer, int x, out int q, out int k, out int v)
        {
            RaonSpec p = Spec;
            int rows = step.Rows, heads = p.MlaHeads, nope = p.QkNope, extra = p.Extra, lat = p.KvLora;
            long qHead = p.QkHead, qWidth = (long)heads * qHead, kvWidth = p.MlaKey;
            int qa = step.RmsNorm(step.MatMul(LayerName(layer, "attn_q_a.weight"), x), LayerName(layer, "attn_q_a_norm.weight"));
            int qb = step.MatMul(LayerName(layer, "attn_q_b.weight"), qa);                                    // [heads·192, rows]
            int qNope = step.Cont(step.Permute(step.View3(qb, nope, heads, rows, qHead * F, qWidth * F, 0), 0, 2, 1, 3));   // [nope, rows, heads]
            int qLat = step.Op(GraphOp.MulMat, step.Weight(LayerName(layer, "attn_k_b.weight")), qNope);       // [lat, rows, heads]
            qLat = step.Cont(step.Permute(qLat, 0, 2, 1, 3));                                                   // [lat, heads, rows]
            int qExtra = step.Cont(step.View3(qb, extra, heads, rows, qHead * F, qWidth * F, nope * F));      // [extra, heads, rows]
            q = step.Op(GraphOp.Concat, stackalloc[] { qLat, qExtra }, stackalloc long[] { 0 });              // [lat + extra, heads, rows]

            int kv = step.MatMul(LayerName(layer, "attn_kv_a_mqa.weight"), x);                                 // [lat + extra, rows]
            int latent = step.RmsNorm(step.Cont(step.View2(kv, lat, rows, kvWidth * F, 0)), LayerName(layer, "attn_kv_a_norm.weight"));
            int kExtra = step.Cont(step.View2(kv, extra, rows, kvWidth * F, lat * F));
            k = step.Reshape(step.Op(GraphOp.Concat, stackalloc[] { latent, kExtra }, stackalloc long[] { 0 }), kvWidth, 1, rows);
            v = step.Reshape(latent, lat, 1, rows);
            return true;
        }

        /// <summary>The runtime scales the scores by 1/sqrt(key width); Raon's are 1/sqrt(nope + extra), the width of the
        /// unabsorbed query.</summary>
        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            RaonSpec p = Spec;
            q = step.Op(GraphOp.Scale, stackalloc[] { q }, default, stackalloc[] { MathF.Sqrt((float)p.MlaKey / p.QkHead) });
        }

        /// <summary>Each head's latent context through <c>attn_v_b</c> [heads·vMla, rows], times <c>sigmoid(attn_gate · x)</c>.</summary>
        protected override int EmitAttentionOutput(GraphStep step, int layer, int x, int attn)
        {
            RaonSpec p = Spec;
            int rows = step.Rows, heads = p.MlaHeads;
            int context = step.Cont(step.Permute(step.Reshape(attn, p.KvLora, heads, rows), 0, 2, 1, 3));     // [lat, rows, heads]
            int values = step.Op(GraphOp.MulMat, step.Weight(LayerName(layer, "attn_v_b.weight")), context);  // [vMla, rows, heads]
            values = step.Reshape(step.Cont(step.Permute(values, 0, 2, 1, 3)), (long)heads * p.VMla, rows);
            return step.Op(GraphOp.Mul, values, step.Op(GraphOp.Sigmoid, step.MatMul(LayerName(layer, "attn_gate.weight"), x)));
        }

        // ---------------------------------------------------------------- RDA (Raon Delta Attention)

        private bool InPlaceRecurrentState => GraphExecutor?.Capabilities.InPlaceStateAlias == true;

        /// <summary>
        /// One RDA layer over every sequence of the step: <c>attn_qkv · x</c> = q|k|v|g; the causal conv and SiLU over q|k|v
        /// (the window the state's raw rows), L2-normed q and k; the per-channel log decay
        /// <c>bound · sigmoid(ssm_f_b · f_a + dt)</c>; the erase gate beta = sigmoid(b) and the write gate folded into the
        /// values (<c>v · sigmoid(w) / sigmoid(b)</c>, in log space); the gated delta rule; then
        /// <c>RMSNorm_head(out) · sigmoid(g)</c> through <c>attn_output</c>.
        /// </summary>
        protected override int EmitRecurrentMixer(GraphStep step, int layer, int normed)
        {
            RaonSpec p = Spec;
            int heads = p.RdaHeads, d = p.RdaDim, rows = step.Rows, rank = p.DecayRank, convRows = p.ConvKernel - 1;
            long inner = p.RdaInner, channels = p.RdaChannels;
            string qkvName = LayerName(layer, "attn_qkv.weight"), fName = LayerName(layer, "ssm_f_a.weight");
            int qkvAll = step.MatMulRows(qkvName, 0, channels, normed);                                       // [channels, rows]
            int gateAll = step.MatMulRows(qkvName, channels, inner, normed);                                  // [inner, rows]
            int fa = step.MatMulRows(fName, 0, rank, normed);                                                 // [rank, rows]
            int erase = step.MatMulRows(fName, rank, heads, normed);                                          // [heads, rows]
            int write = step.MatMulRows(fName, rank + heads, heads, normed);                                  // [heads, rows]
            int decay = step.Op(GraphOp.Add, step.MatMul(LayerName(layer, "ssm_f_b.weight"), fa), step.Weight(LayerName(layer, "ssm_dt.bias")));
            int logDecay = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Sigmoid, decay) }, default, stackalloc[] { p.GateLowerBound });
            int betaAll = step.Op(GraphOp.Sigmoid, erase);
            // sigmoid(w) / sigmoid(b) = exp(softplus(-b) - softplus(-w)): two saturated sigmoids cannot divide 0 by 0.
            int ratioAll = step.Op(GraphOp.Exp, step.Op(GraphOp.Sub,
                step.Op(GraphOp.Softplus, step.Op(GraphOp.Neg, erase)), step.Op(GraphOp.Softplus, step.Op(GraphOp.Neg, write))));
            int convW = step.Weight(LayerName(layer, "ssm_conv1d.weight"));                                  // [kernel, channels]
            int ssmNorm = step.Weight(LayerName(layer, "ssm_norm.weight"));
            long stateFloats = (long)d * d * heads;
            bool viaBacking = InPlaceRecurrentState;

            int output = -1;
            for (int s = 0; s < step.Sequences; s++)
            {
                int count = step.Counts[s];
                long first = step.First[s];
                var state = (Qwen35RuntimeModel.DeviceGdnState)step.SequenceState(s, layer);
                int Rows(int all, long width) => step.Sequences == 1 ? all : step.Cont(step.View2(all, width, count, width * F, first * width * F));
                int qkv = Rows(qkvAll, channels), gate = Rows(gateAll, inner);
                int window = step.State(state.Conv, GraphFormat.F32, convRows, channels);
                // A verify window leaves the live state and conv window alone: the commit sets them from its buffers.
                bool keep = step.RecurrentSnapshots && count > 1;

                int convIn = step.Op(GraphOp.Concat, stackalloc[] { window, step.Transpose(qkv) }, stackalloc long[] { 0 });   // [convRows + count, channels]
                int conv = step.Op(GraphOp.Silu, step.Op(GraphOp.SsmConv, convIn, convW));                               // [channels, count]
                long token = channels * F, all = channels * count * F;
                int q4 = step.Cont(step.View4(conv, d, heads, count, 1, d * F, token, all, 0));
                int k4 = step.Cont(step.View4(conv, d, heads, count, 1, d * F, token, all, inner * F));
                int v4 = step.Cont(step.View4(conv, d, heads, count, 1, d * F, token, all, 2 * inner * F));
                q4 = step.L2Norm(q4, d, L2Eps);
                k4 = step.L2Norm(k4, d, L2Eps);
                v4 = step.Op(GraphOp.Mul, v4, step.Reshape(Rows(ratioAll, heads), 1, heads, count, 1));
                int g4 = step.Reshape(Rows(logDecay, inner), d, heads, count, 1);
                int b4 = step.Reshape(Rows(betaAll, heads), 1, heads, count, 1);

                int backing = -1, s4;
                if (viaBacking)
                {
                    backing = step.State(state.Backing, GraphFormat.F32, state.StateOffset + stateFloats);
                    s4 = step.View4(backing, d, d, heads, 1, d * F, (long)d * d * F, stateFloats * F, state.StateOffset * F);
                }
                else
                {
                    s4 = step.Reshape(step.State(state.Ssm, GraphFormat.F32, d, d, heads), d, d, heads, 1);
                }

                int verify = -1;
                if (keep)
                {
                    if (state.EnsureSpecBuffers(count)) NoteGraphBindingsChanged();
                    verify = step.State(state.Verify, GraphFormat.F32, state.SnapshotCapacity * (state.RowFloats + stateFloats));
                }
                else
                {
                    step.Effect(step.Op(GraphOp.Cpy, step.View2(convIn, convRows, channels, (convRows + count) * F, count * F), window));
                }
                int gdn = step.Op(GraphOp.GatedDeltaNet, stackalloc[] { q4, k4, v4, g4, b4, s4 }, stackalloc long[] { keep ? count : 1 });
                if (keep)
                {
                    // [rows' outputs | state after the last row | state a row before | ...] into the verify buffer, and the
                    // window's conv inputs into the history the commit reads.
                    long total = inner * count + stateFloats * count;
                    int fallback = step.Op(GraphOp.Cpy, step.View2(gdn, total, 1, total * F, 0), step.View2(verify, total, 1, total * F, 0));
                    step.Alias(gdn, verify, 0, fallback);
                    int history = step.State(state.ConvHistory, GraphFormat.F32, convRows + state.SnapshotCapacity, channels);
                    step.Effect(step.Op(GraphOp.Cpy, convIn, step.View2(history, convRows + count, channels, (convRows + count) * F, 0)));
                }
                else
                {
                    int newState = step.Op(GraphOp.View, stackalloc[] { gdn },
                        stackalloc long[] { 4, d, d, heads, 1, d * F, (long)d * d * F, stateFloats * F, inner * count * F });
                    int copy = step.Op(GraphOp.Cpy, newState, s4);
                    if (viaBacking && count == 1)
                        step.Alias(gdn, backing, (state.BackingRows - 1) * state.RowFloats * F, copy);
                    else
                        step.Effect(copy);
                }

                int y = step.Reshape(step.Cont(step.View2(gdn, inner, count, inner * F, 0)), d, heads, count);
                int gated = step.Op(GraphOp.Mul, step.RmsNorm(y, ssmNorm, Config.Eps),
                    step.Op(GraphOp.Sigmoid, step.Reshape(gate, d, heads, count)));
                int part = step.Reshape(gated, inner, count);
                output = output < 0 ? part : step.Op(GraphOp.Concat, stackalloc[] { output, part }, stackalloc long[] { 1 });
            }
            return step.MatMul(LayerName(layer, "attn_output.weight"), output);
        }

        // ---------------------------------------------------------------- verify windows

        protected override void OnGraphCacheCleared()
        {
            foreach (IGraphPlan g in _commitGraphs.Values) g.Dispose();
            _commitGraphs.Clear();
        }

        /// <summary>The active request's RDA states after <paramref name="kept"/> rows of the last <paramref name="rows"/>-row
        /// verify, copied from its verify buffers (0 rows: the live states are still the ones before the window).</summary>
        protected override void CommitRecurrentPrefix(int kept, int rows)
        {
            if (kept == 0) return;
            IntPtr key = IntPtr.Zero;
            for (int layer = 0; layer < Config.NumLayers && key == IntPtr.Zero; layer++)
                if (Spec.IsRda[layer] && ActiveRecurrentState(layer) is Qwen35RuntimeModel.DeviceGdnState first && first.Verify != null)
                    key = StateIdentity(first.Verify);
            if (key == IntPtr.Zero) return;
            if (!_commitGraphs.TryGetValue((key, kept, rows), out IGraphPlan plan))
            {
                plan = CompileForDevice(BuildCommitProgram(kept, rows), 0, 0);
                _commitGraphs[(key, kept, rows)] = plan;
            }
            if (!plan.Run(ReadOnlySpan<int>.Empty, ReadOnlySpan<IntPtr>.Empty, ReadOnlySpan<long>.Empty))
                throw new InvalidOperationException($"{Config.Architecture} recurrent-state commit failed: {plan.LastError}");
        }

        /// <summary>Per RDA layer: the state after the kept rows from the verify buffer, and the conv window after them from
        /// the window's history.</summary>
        private GraphProgram BuildCommitProgram(int kept, int rows)
        {
            var g = new GraphProgramBuilder(record: true);
            RaonSpec p = Spec;
            int d = p.RdaDim;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!Spec.IsRda[layer] || ActiveRecurrentState(layer) is not Qwen35RuntimeModel.DeviceGdnState state || state.Verify == null) continue;
                long stateFloats = state.StateFloats, stateBytes = stateFloats * F;
                int cap = state.SnapshotCapacity, convRows = state.ConvRows, channels = state.Channels;
                int verify = Leaf(g, state.Verify, cap * (state.RowFloats + stateFloats));
                long from = state.RowFloats * rows + (rows - kept) * stateFloats;
                int source = g.Op(GraphOp.View, stackalloc[] { verify }, stackalloc long[] { 1, stateFloats, 1, 1, 1, stateBytes, 0, 0, from * F });
                int target = InPlaceRecurrentState
                    ? g.Op(GraphOp.View, stackalloc[] { Leaf(g, state.Backing, state.StateOffset + stateFloats) },
                        stackalloc long[] { 1, stateFloats, 1, 1, 1, stateBytes, 0, 0, state.StateOffset * F })
                    : g.Op(GraphOp.Reshape, stackalloc[] { Leaf(g, state.Ssm, d, d, state.Ssm.Sizes[0]) }, stackalloc long[] { stateFloats });
                g.Effect(g.Op(GraphOp.Cpy, source, target));
                int history = Leaf(g, state.ConvHistory, convRows + cap, channels);
                long pitch = convRows + rows;   // the window wrote its conv inputs at its own pitch
                int window = g.Op(GraphOp.View, stackalloc[] { history }, stackalloc long[] { 2, convRows, channels, 1, 1, pitch * F, 0, 0, kept * F });
                g.Effect(g.Op(GraphOp.Cpy, window, Leaf(g, state.Conv, convRows, channels)));
            }
            return g.ToProgram(replayable: true, reorder: false);
        }

        private int Leaf(GraphProgramBuilder g, Tensor t, params long[] ne)
            => g.Leaf(GraphLeafRole.State, GraphFormat.F32, ne, StateIdentity(t), t.Storage.ByteLength, handle: t);

        private IntPtr StateIdentity(Tensor t) => GraphExecutor.Capabilities.BindsHostPointers
            ? TensorComputePrimitives.GetStoragePointer(t)
            : GraphIdentity.Of(t.Storage, t.StorageOffset);
    }
}
