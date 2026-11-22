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
    /// Inkling 2 in whole-forward graphs. The residual stream between layers is <see cref="_streamSlots"/> rows of the hidden
    /// size each: slot 0 the current stream, slots 1 .. the blocks B_k (the stream entering layer k * block size; B_0 the
    /// normed embedding), and for the loop the previous pass's bank (its blocks after its first core block) and its output.
    /// A block input mixes slot 0 and the blocks pushed so far, which are contiguous slots.
    /// </summary>
    public partial class Inkling2RuntimeModel
    {
        private RotarySpec _placeholderRotary;

        /// <summary>A non-null rotary (the graph path's precondition); Inkling 2 rotates nothing (<see cref="EmitQueryKey"/>).</summary>
        protected override RotarySpec GraphRotary => _placeholderRotary ??= new RotarySpec(_headDim, neoX: true, 10000f, 1f);

        protected override bool RecurrentMixerLowers => true;

        protected override int StreamWidth => _streamSlots * Config.HiddenSize;

        private const float MixEps = 1e-6f;
        private const int Cur = 0;
        private static int BlockSlot(int k) => 1 + k;
        private int BankSlot(int j) => 1 + _blocks + j;
        private int BankOutSlot => 1 + _blocks + _bankBlocks;

        /// <summary>Blocks a run's attention input mixes (pushed before its stored layer).</summary>
        private int BlocksBefore(int layer) => (PhysicalLayer(layer) + _blockSize - 1) / _blockSize;

        /// <summary>Blocks its MoE input mixes (its own push included).</summary>
        private int BlocksAfter(int layer) => PhysicalLayer(layer) / _blockSize + 1;

        private bool StartsBlock(int layer) => PhysicalLayer(layer) % _blockSize == 0;

        // ---------------------------------------------------------------- the stream

        /// <summary>Slots [first, first + count) of the stream as [hidden, count, rows] (strided).</summary>
        private int Slots(GraphStep step, int stream, int first, int count)
        {
            long h = Config.HiddenSize, f = sizeof(float);
            return step.View3(stream, h, count, step.Rows, h * f, _streamSlots * h * f, first * h * f);
        }

        /// <summary>One slot of the stream, contiguous [hidden, rows].</summary>
        private int StreamSlot(GraphStep step, int stream, int slot)
            => step.Cont(step.Reshape(step.Cont(Slots(step, stream, slot, 1)), Config.HiddenSize, step.Rows));

        /// <summary>The stream [slots * hidden, rows] made of one [hidden, rows] value a slot.</summary>
        private int Assemble(GraphStep step, int[] values)
        {
            int h = Config.HiddenSize, rows = step.Rows;
            int stream = -1;
            foreach (int v in values)
            {
                int v3 = step.Reshape(step.Cont(v), h, 1, rows);
                stream = stream < 0 ? v3 : step.Op(GraphOp.Concat, stackalloc[] { stream, v3 }, stackalloc long[] { 1 });
            }
            return step.Reshape(stream, (long)_streamSlots * h, rows);
        }

        /// <summary>The stream's slot <paramref name="slot"/> as a strided view [hidden, rows].</summary>
        private int SlotView(GraphStep step, int stream, int slot)
        {
            long h = Config.HiddenSize, f = sizeof(float);
            return step.View2(stream, h, step.Rows, _streamSlots * h * f, slot * h * f);
        }

        /// <summary>
        /// The stream rebuilt slot by slot: <paramref name="from"/>[i] = ~k keeps the old stream's slot k at slot i, and a node
        /// id (&gt;= 0) places that [hidden, rows] value there. Runs of consecutive old slots are copied as one view, so a layer
        /// that changes one or two slots costs a few copies, not a copy of each slot and a join of all of them.
        /// </summary>
        private int Rebuild(GraphStep step, int stream, int[] from)
        {
            int h = Config.HiddenSize, rows = step.Rows;
            long f = sizeof(float);
            int result = -1;
            for (int i = 0; i < from.Length;)
            {
                int piece;
                if (from[i] >= 0)
                {
                    piece = step.Reshape(step.Cont(from[i]), h, rows);
                    i++;
                }
                else
                {
                    int k = ~from[i], run = 1;
                    while (i + run < from.Length && from[i + run] < 0 && ~from[i + run] == k + run) run++;
                    piece = step.Cont(step.View2(stream, (long)run * h, rows, _streamSlots * h * f, k * h * f));
                    i += run;
                }
                result = result < 0 ? piece : step.Op(GraphOp.Concat, stackalloc[] { result, piece }, stackalloc long[] { 0 });
            }
            return result;
        }

        /// <summary>Every slot kept where it is (<see cref="Rebuild"/>'s identity), to change some.</summary>
        private int[] Kept()
        {
            var from = new int[_streamSlots];
            for (int i = 0; i < from.Length; i++) from[i] = ~i;
            return from;
        }

        protected override int EmitStreamInput(GraphStep step, int embedded)
        {
            // The normed embedding is the current stream and B_0 (layer 0's block start pushes it); the other slots are written
            // before any block reads them.
            int x = step.RmsNorm(embedded, "token_embd_norm.weight");
            var values = new int[_streamSlots];
            Array.Fill(values, x);
            return Assemble(step, values);
        }

        /// <summary>A later loop pass starts from the prelude's stream: the bank takes the previous pass's blocks after its first
        /// core block and its output, and the current stream goes back to the block the core's first layer pushed.</summary>
        protected override int EmitLayerPrologue(GraphStep step, int layer, int stream)
        {
            if (!StartsPass(layer)) return stream;
            int[] from = Kept();
            int coreBlock = _loopStart / _blockSize;
            for (int j = 0; j < _bankBlocks; j++) from[BankSlot(j)] = ~BlockSlot(coreBlock + 1 + j);
            from[BankOutSlot] = ~Cur;
            from[Cur] = ~BlockSlot(coreBlock);
            return Rebuild(step, stream, from);
        }

        // ---------------------------------------------------------------- mixes and controllers

        private int RmsNoWeight(GraphStep step, int x, float eps)
            => step.Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps });

        /// <summary>Attention over depth: the states [hidden, n, rows] RMS-normed, scored against the query weight's columns
        /// ([hidden, queries]), softmax over the n states, and their mix: [hidden, queries, rows].</summary>
        private int Mix(GraphStep step, int states, int query, int queries)
        {
            int h = Config.HiddenSize;
            float scale = 1.0f / MathF.Sqrt(h);
            int s = RmsNoWeight(step, step.Cont(states), MixEps);                                   // [h, n, rows]
            int scores = step.Op(GraphOp.MulMat, query, s);                                      // [queries, n, rows]
            scores = step.Cont(step.Permute(scores, 1, 0, 2, 3));                                    // [n, queries, rows]
            int probs = step.Op(GraphOp.SoftMaxExt, stackalloc[] { scores }, default, stackalloc[] { scale, 0f });
            int sT = step.Cont(step.Transpose(s));                                                   // [n, h, rows]
            return step.Op(GraphOp.MulMat, sT, probs);                                               // [h, queries, rows]
        }

        private int QueryWeight(GraphStep step, string name, int queries)
            => queries == 1 ? step.WeightAs(name, Config.HiddenSize, 1) : step.Weight(name);

        /// <summary>Column <paramref name="j"/> of a mix [hidden, queries, rows] as [hidden, rows].</summary>
        private int MixColumn(GraphStep step, int mix, int queries, int j)
        {
            long h = Config.HiddenSize, f = sizeof(float);
            return step.Cont(step.View2(mix, h, step.Rows, queries * h * f, j * h * f));
        }

        /// <summary>The previous pass's messages for a later pass's run of a core layer: [hidden, 4, rows] (q, k, v, ffn).</summary>
        private int Messages(GraphStep step, int layer, int stream)
            => Mix(step, Slots(step, stream, BankSlot(0), _bankBlocks + 1), QueryWeight(step, LayerName(layer, "loop_msg_query.weight"), 4), 4);

        /// <summary><c>x - tanh(W_forget g) x + tanh(W_gain g) m</c>, <c>g = W_down [rms(x); rms(m)]</c>.</summary>
        private int Control(GraphStep step, int layer, int x, int message, string controller)
        {
            int g = step.Op(GraphOp.Concat, stackalloc[] { RmsNoWeight(step, x, MixEps), RmsNoWeight(step, message, MixEps) }, stackalloc long[] { 0 });
            g = step.MatMul(LayerName(layer, $"loop_{controller}_ctrl_down.weight"), g);
            int gain = step.Op(GraphOp.Tanh, step.MatMul(LayerName(layer, $"loop_{controller}_ctrl_gain.weight"), g));
            int forget = step.Op(GraphOp.Tanh, step.MatMul(LayerName(layer, $"loop_{controller}_ctrl_forget.weight"), g));
            return step.Op(GraphOp.Add, step.Op(GraphOp.Sub, x, step.Op(GraphOp.Mul, forget, x)), step.Op(GraphOp.Mul, gain, message));
        }

        // ---------------------------------------------------------------- block inputs and outputs

        protected override int EmitBlockInput(GraphStep step, int layer, int stream, BlockSite site, out int context)
        {
            context = -1;
            int h = Config.HiddenSize;
            if (site == BlockSite.Ffn)
            {
                int n = BlocksAfter(layer) + 1;
                int x = MixColumn(step, Mix(step, Slots(step, stream, Cur, n), QueryWeight(step, LayerName(layer, "ffn_res_query.weight"), 1), 1), 1, 0);
                if (ReadsMessages(layer))
                    x = Control(step, layer, x, MixColumn(step, Messages(step, layer, stream), 4, 3), "ffn");
                return step.RmsNorm(x, LayerName(layer, "ffn_norm.weight"));
            }

            bool sliding = IsSliding(layer);
            int xq, xk, xv;
            if (PhysicalLayer(layer) > 0)
            {
                int queries = sliding ? 1 : 3;
                int mixed = Mix(step, Slots(step, stream, Cur, BlocksBefore(layer) + 1),
                    QueryWeight(step, LayerName(layer, "attn_res_query.weight"), queries), queries);
                xq = MixColumn(step, mixed, queries, 0);
                xk = sliding ? xq : MixColumn(step, mixed, queries, 1);
                xv = sliding ? xq : MixColumn(step, mixed, queries, 2);
            }
            else
            {
                xq = xk = xv = StreamSlot(step, stream, Cur);
            }
            bool messages = ReadsMessages(layer);
            if (messages)
            {
                int m = Messages(step, layer, stream);
                xq = Control(step, layer, xq, MixColumn(step, m, 4, 0), "q");
                xk = Control(step, layer, xk, MixColumn(step, m, 4, 1), "k");
                xv = Control(step, layer, xv, MixColumn(step, m, 4, 2), "v");
            }
            if (sliding)
            {
                string norm = LayerName(layer, "attn_norm.weight");
                xq = step.RmsNorm(xq, norm);
                if (messages)
                {
                    xk = step.RmsNorm(xk, norm);
                    xv = step.RmsNorm(xv, norm);
                }
                else
                {
                    xk = xv = xq;
                }
            }
            else
            {
                xq = step.RmsNorm(xq, LayerName(layer, "attn_q_inp_norm.weight"));
                xk = step.RmsNorm(xk, LayerName(layer, "attn_k_inp_norm.weight"));
                xv = step.RmsNorm(xv, LayerName(layer, "attn_v_inp_norm.weight"));
            }
            // The attention reads the three inputs packed [3 * hidden, rows] (EmitAttentionProjections).
            int qk = step.Op(GraphOp.Concat, stackalloc[] { xq, xk }, stackalloc long[] { 0 });
            return step.Op(GraphOp.Concat, stackalloc[] { qk, xv }, stackalloc long[] { 0 });
        }

        protected override int EmitBlockOutput(GraphStep step, int layer, int stream, BlockSite site, int context, int output)
        {
            int h = Config.HiddenSize;
            int[] from = Kept();
            if (site == BlockSite.Ffn)
            {
                int moe = Convolve(step, layer, output, 2 * _kvDim + h, h, "ffn_post_conv.weight", -1, 0);
                from[Cur] = step.Op(GraphOp.Add, step.Cont(SlotView(step, stream, Cur)), moe);
                return Rebuild(step, stream, from);
            }
            int o = Convolve(step, layer, output, 2 * _kvDim, h, "attn_post_conv.weight", -1, 0);
            if (StartsBlock(layer))
            {
                // The block's first layer starts a new residual stream: the stream it read joins the blocks.
                from[BlockSlot(PhysicalLayer(layer) / _blockSize)] = ~Cur;
                from[Cur] = o;
            }
            else
            {
                from[Cur] = step.Op(GraphOp.Add, step.Cont(SlotView(step, stream, Cur)), o);
            }
            return Rebuild(step, stream, from);
        }

        protected override int EmitStreamOutput(GraphStep step, int stream)
        {
            int x = MixColumn(step, Mix(step, Slots(step, stream, Cur, _blocks + 1), QueryWeight(step, "output_res_query.weight", 1), 1), 1, 0);
            return step.RmsNorm(x, "output_norm.weight");
        }

        // ---------------------------------------------------------------- short convolutions

        /// <summary>
        /// The causal short convolution of <paramref name="x"/> [channels, rows] with its residual, per sequence over the
        /// channels [offset, offset + channels) of the run's conv window: ggml_ssm_conv with the static taps, plus, where
        /// <paramref name="dynamic"/> [taps * heads, rows] is given, each head's input-dependent taps (lag L reads x[t - L]);
        /// then <c>(x + conv) / sqrt(taps * (dynamic ? 2 : 1) + 1)</c>. The window advances by the rows.
        /// </summary>
        private int Convolve(GraphStep step, int layer, int x, int offset, int channels, string weight, int dynamic, int heads)
        {
            const long f = sizeof(float);
            int taps = _convTaps, history = taps - 1;
            int kernel = step.Weight(LayerName(layer, weight));                                      // [taps, channels]
            float scale = 1.0f / MathF.Sqrt(taps * (dynamic >= 0 ? 2 : 1) + 1);
            int n = step.Sequences, output = -1;
            for (int s = 0; s < n; s++)
            {
                int count = step.Counts[s];
                long first = step.First[s];
                int xs = step.Cont(n == 1 ? x : step.View2(x, channels, count, channels * f, first * channels * f));
                var state = (ShortConvState)step.SequenceState(s, layer);
                int leaf = step.State(state.Conv, GraphFormat.F32, history, _convChannels);
                int window = step.View2(leaf, history, channels, history * f, offset * history * f);           // [taps - 1, channels]
                long width = (long)taps * heads;
                int ds = dynamic < 0 ? -1 : step.Cont(n == 1 ? dynamic : step.View2(dynamic, width, count, width * f, first * width * f));
                // A verify window leaves the live window as it was and writes its inputs into the state's history at the window's
                // own pitch, the layer's four convolutions each at its channels: the commit takes the accepted rows' window
                // (CommitShortConvWindows).
                bool verifying = step.RecurrentSnapshots && count > 1;
                int historyLeaf = -1;
                if (verifying)
                {
                    if (state.EnsureSpecBuffers(count)) NoteGraphBindingsChanged();
                    historyLeaf = step.State(state.ConvHistory, GraphFormat.F32, history + state.SnapshotCapacity, _convChannels);
                }
                int convIn = step.Op(GraphOp.Concat, stackalloc[] { window, step.Transpose(xs) }, stackalloc long[] { 0 });   // [taps - 1 + count, channels]
                int conv = step.Op(GraphOp.SsmConv, convIn, kernel);                                               // [channels, count]
                long pitch = (history + count) * f;
                if (ds >= 0)
                {
                    int d3 = step.Reshape(ds, taps, heads, count);
                    for (int lag = 0; lag < taps; lag++)
                    {
                        int tap = step.Cont(step.View3(d3, 1, heads, count, taps * f, width * f, lag * f));            // [1, heads, count]
                        int lagged = step.View2(convIn, count, channels, pitch, (history - lag) * f);                  // [count, channels]
                        int w = step.Reshape(step.Cont(step.Transpose(lagged)), channels / heads, heads, count);
                        conv = step.Op(GraphOp.Add, conv, step.Reshape(step.Op(GraphOp.Mul, w, tap), channels, count));
                    }
                }
                if (verifying)
                    step.Effect(step.Op(GraphOp.Cpy, convIn, step.View2(historyLeaf, history + count, channels, pitch, offset * pitch)));
                else
                    // convIn is a fresh contiguous [taps - 1 + count, channels]: its last taps - 1 columns are the new window.
                    step.Effect(step.Op(GraphOp.Cpy, step.View2(convIn, history, channels, pitch, count * f), window));
                int y = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Add, xs, conv) }, default, stackalloc[] { scale });
                output = output < 0 ? y : step.Op(GraphOp.Concat, stackalloc[] { output, y }, stackalloc long[] { 1 });
            }
            return output;
        }

        // ---------------------------------------------------------------- attention

        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k) { }

        /// <summary>Rows of <paramref name="x"/> [d, heads, rows] scaled down to an L2 norm of at most
        /// <paramref name="threshold"/>, after the optional per-channel scale.</summary>
        private int Clip(GraphStep step, int x, string scale, float threshold)
        {
            if (scale != null) x = step.Op(GraphOp.Mul, x, step.Weight(scale));
            int norm = step.Op(GraphOp.Sqrt, step.Op(GraphOp.SumRows, step.Op(GraphOp.Sqr, x)));                 // [1, heads, rows]
            int biased = step.Op(GraphOp.ScaleBias, stackalloc[] { norm }, default, stackalloc[] { 1f, 1e-6f });
            int ones = step.Op(GraphOp.ScaleBias, stackalloc[] { norm }, default, stackalloc[] { 0f, 1f });
            int factor = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Div, ones, biased) }, default, stackalloc[] { threshold });
            factor = step.Op(GraphOp.Clamp, stackalloc[] { factor }, default, stackalloc[] { 0f, 1f });
            return step.Op(GraphOp.Mul, x, factor);
        }

        /// <summary>The full layers' log-scaled positions [1, 1, rows]: 1 + alpha log(max(1, (pos + 1) / floor)).</summary>
        private int LogScale(GraphStep step)
        {
            var tau = new float[step.Rows];
            for (int s = 0; s < step.Sequences; s++)
                for (int i = 0; i < step.Counts[s]; i++)
                {
                    int pos = step.Starts[s] + i;
                    tau[step.First[s] + i] = 1f + _logScaleAlpha * MathF.Log(MathF.Max(1f, (pos + 1f) / _logScaleFloor));
                }
            return step.ConstantInputF32(tau, 1, 1, step.Rows);
        }

        // The current attention's relative table [heads, extent + 1, rows]: (B r - t), traced by EmitAttentionProjections and
        // gathered by EmitAttentionCore for each sequence of the same layer.
        private GraphStep _tableStep;
        private int _tableLayer = -1, _table = -1;

        private static int Input(GraphStep step, int packed, int j, int hidden)
        {
            long f = sizeof(float);
            return step.Cont(step.View2(packed, hidden, step.Rows, 3L * hidden * f, j * hidden * f));
        }

        protected override bool EmitAttentionProjections(GraphStep step, int layer, int x, out int q, out int k, out int v)
        {
            int h = Config.HiddenSize, heads = Config.NumHeads, kvHeads = Config.NumKVHeads, d = _headDim, rows = step.Rows;
            bool sliding = IsSliding(layer);
            int xq = Input(step, x, 0, h), xk = Input(step, x, 1, h), xv = Input(step, x, 2, h);

            q = Clip(step, step.Reshape(step.MatMul(LayerName(layer, "attn_q.weight"), xq), d, heads, rows), LayerName(layer, "attn_q_clip.weight"), _qkClip);
            int kDyn = sliding ? -1 : step.MatMul(LayerName(layer, "attn_k_conv_proj.weight"), xk);
            int kc = Convolve(step, layer, step.MatMul(LayerName(layer, "attn_k.weight"), xk), 0, _kvDim, "attn_k_conv.weight", kDyn, kvHeads);
            k = Clip(step, step.Reshape(kc, d, kvHeads, rows), null, _qkClip);
            int vDyn = sliding ? -1 : step.MatMul(LayerName(layer, "attn_v_conv_proj.weight"), xv);
            int vc = Convolve(step, layer, step.MatMul(LayerName(layer, "attn_v.weight"), xv), _kvDim, _kvDim, "attn_v_conv.weight", vDyn, kvHeads);
            v = RmsNoWeight(step, step.Reshape(vc, d, kvHeads, rows), Config.Eps);
            int r = Clip(step, step.Reshape(step.MatMul(LayerName(layer, "attn_rel.weight"), xq), _relDim, heads, rows),
                LayerName(layer, "attn_rel_clip.weight"), _relClip);
            if (!sliding)
            {
                int tau = LogScale(step);
                q = step.Op(GraphOp.Mul, q, tau);
                r = step.Op(GraphOp.Mul, r, tau);
            }
            // The sink logit t is per query: exp(s - t) / (1 + sum exp(s - t)) = exp(s) / (exp(t) + sum exp(s)), so every
            // key's bias takes - t and a zero sink closes the softmax (llama.cpp's inkling2.cpp).
            int t = step.MatMul(LayerName(layer, "attn_sink_proj.weight"), xq);                                     // [heads, rows]
            int extent = ExtentOf(layer);
            int table = step.Op(GraphOp.MulMat, step.TensorWeight(_relTables[PhysicalLayer(layer)], _relDim, extent + 1), r);   // [extent + 1, heads, rows]
            table = step.Op(GraphOp.Sub, table, step.Reshape(t, 1, heads, rows));
            _table = step.Cont(step.Permute(table, 1, 0, 2, 3));                                                    // [heads, extent + 1, rows]
            _tableStep = step;
            _tableLayer = layer;
            q = step.Cont(q);
            k = step.Cont(k);
            v = step.Cont(v);
            return true;
        }

        /// <summary>Bytes a chunk of query rows may give its scores [keys, rows, heads] (F32), as the native explicit attention
        /// bounds them: a long prompt's chunk attends in row slices.</summary>
        private const long ScoreBudgetBytes = 256L << 20;

        /// <summary>
        /// One sequence's attention in F32 (ggml's flash attention takes an F16 mask only and has no kernel for a mask per head):
        /// scores q·k * scale, plus each key's relative bias (B r - t)[distance] (distances outside [0, extent) take the zero
        /// row) and the plain mask, a zero sink logit closing every row's softmax, then the values. q [d, rows, heads], k
        /// [d, keys, kvHeads], v [dv, keys, kvHeads], mask [keys, rows] (F16); returns [dv, heads, rows].
        /// </summary>
        protected override int EmitAttentionCore(GraphStep step, int layer, int x, int s, int q, int k, int v, int mask, int keys,
            in AttentionKeys layout, float scale)
        {
            if (!ReferenceEquals(step, _tableStep) || layer != _tableLayer)
                throw new InvalidOperationException($"{Config.Architecture}: attention for layer {layer} without its relative table.");
            int heads = Config.NumHeads, count = step.Counts[s], extent = ExtentOf(layer), d = _headDim;
            const long f = sizeof(float);
            int qc = step.Cont(q);                                                                                    // [d, count, heads]
            // A cache window may hold F16 or block-quantized rows, which do not transpose: the values widen first.
            int v32 = step.Op(GraphOp.Cast, stackalloc[] { v }, stackalloc long[] { GraphFormat.F32 });
            int vT = step.Cont(step.Transpose(v32));                                                                  // [keys, dv, kvHeads]
            int plain = step.Op(GraphOp.Cast, stackalloc[] { mask }, stackalloc long[] { GraphFormat.F32 });       // [keys, count]
            int slice = (int)Math.Clamp(ScoreBudgetBytes / Math.Max(1L, (long)keys * heads * f), 1, count);
            int output = -1;
            for (int c0 = 0; c0 < count; c0 += slice)
            {
                int len = Math.Min(slice, count - c0);
                int qs = len == count ? qc : step.View3(qc, d, len, heads, d * f, d * count * f, c0 * d * f);
                int scores = step.Op(GraphOp.MulMat, k, step.Cont(qs));                                               // [keys, len, heads]
                scores = step.Op(GraphOp.Scale, stackalloc[] { scores }, default, stackalloc[] { scale });
                int rowsMask = len == count ? plain : step.Cont(step.View2(plain, keys, len, keys * f, c0 * keys * f));
                scores = step.Op(GraphOp.Add, scores, step.Op(GraphOp.Add, Bias(step, s, c0, len, keys, extent, layout), step.Reshape(rowsMask, keys, len, 1)));
                // The zero sink: one more logit a row, with no value (zeros made from the queries, which are finite: a masked
                // score is -inf, and -inf * 0 is NaN).
                int zero = step.Op(GraphOp.Scale, stackalloc[] { step.Cont(step.View3(qc, 1, len, heads, d * f, d * count * f, c0 * d * f)) },
                    default, stackalloc[] { 0f });
                int withSink = step.Op(GraphOp.Concat, stackalloc[] { scores, zero }, stackalloc long[] { 0 });       // [keys + 1, len, heads]
                int probs = step.Op(GraphOp.SoftMaxExt, withSink);
                probs = step.Cont(step.View3(probs, keys, len, heads, (keys + 1) * f, (keys + 1) * len * f, 0));
                int part = step.Op(GraphOp.MulMat, vT, probs);                                                         // [dv, len, heads]
                output = output < 0 ? part : step.Op(GraphOp.Concat, stackalloc[] { output, part }, stackalloc long[] { 1 });
            }
            return step.Cont(step.Permute(output, 0, 2, 1, 3));                                                        // [dv, heads, count]
        }

        /// <summary>Rows [first, first + len) of sequence <paramref name="s"/>: each key's relative bias [keys, len, heads].</summary>
        private int Bias(GraphStep step, int s, int first, int len, int keys, int extent, in AttentionKeys layout)
        {
            int heads = Config.NumHeads;
            const long f = sizeof(float);
            long rowBytes = heads * (extent + 1) * f;
            int table = step.Sequences == 1 && first == 0 && len == step.Rows ? _table
                : step.Cont(step.View3(_table, heads, extent + 1, len, heads * f, rowBytes, (step.First[s] + first) * rowBytes));
            var distance = new int[checked(keys * len)];
            int last = layout.RowStart + layout.Rows - 1;
            for (int i = 0; i < len; i++)
            {
                int p = layout.RowStart + first + i;
                for (int j = 0; j < keys; j++)
                {
                    int keyPosition = layout.Ring > 0
                        ? (last >= j ? j + (last - j) / layout.Ring * layout.Ring : -1)
                        : layout.FirstPosition + j;
                    int dd = keyPosition < 0 ? -1 : p - keyPosition;
                    distance[i * keys + j] = dd >= 0 && dd < extent ? dd : extent;
                }
            }
            int index = step.Reshape(step.ConstantInputI32(distance, (long)keys * len), keys, len);
            int bias = step.Op(GraphOp.GetRows, table, index);                                                       // [heads, keys, len]
            return step.Cont(step.Permute(bias, 2, 0, 1, 3));                                                        // [keys, len, heads]
        }

        /// <summary>attn [heads * d, rows] times sigmoid(attn_gate · xq), element-wise.</summary>
        protected override int EmitAttentionOutput(GraphStep step, int layer, int x, int attn)
        {
            int gate = step.Op(GraphOp.Sigmoid, step.MatMul(LayerName(layer, "attn_gate.weight"), Input(step, x, 0, Config.HiddenSize)));
            return step.Op(GraphOp.Mul, step.Cont(attn), gate);
        }

        // ---------------------------------------------------------------- experts and head

        /// <summary>
        /// Sigmoid routing (the bias picks the experts, the weights are the unbiased probabilities normalized and times
        /// expert_weights_scale), the fused gate/up stack (gate the first half), SITU =
        /// beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb), the down stack and the weighted sum.
        /// </summary>
        protected override int EmitFeedForward(GraphStep step, int layer, int x)
        {
            int rows = step.Rows, h = Config.HiddenSize;
            int experts = (int)_gguf.GetUint32($"{Config.Architecture}.expert_count");
            int k = (int)_gguf.GetUint32($"{Config.Architecture}.expert_used_count");
            int ff = (int)_gguf.GetUint32($"{Config.Architecture}.expert_feed_forward_length");
            const long f = sizeof(float);

            int logits = step.MatMul(LayerName(layer, "ffn_gate_inp.weight"), x);                                     // [E, rows]
            int probs = step.Op(GraphOp.Sigmoid, logits);
            int selection = step.Op(GraphOp.Add, probs, step.Weight(LayerName(layer, "exp_probs_b.bias")));
            int forced = EmitForcedRoutes(step, layer, k, experts);                                                  // tests
            int selected = forced >= 0 ? forced : step.Op(GraphOp.TopK, stackalloc[] { selection }, stackalloc long[] { k });   // [K, rows]
            EmitDeviceRoutes(step, layer, selected, selection, k, experts);                                          // tests
            int picked = step.Reshape(step.Op(GraphOp.GetRows, step.Reshape(probs, 1, experts, rows), selected), k, rows);
            int sum = step.Op(GraphOp.ScaleBias, stackalloc[] { step.Op(GraphOp.SumRows, picked) }, default, stackalloc[] { 1f, 1e-20f });
            int weights = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Div, picked, sum) }, default, stackalloc[] { _expertScale });
            weights = step.Reshape(weights, 1, k, rows);

            int input = step.Reshape(x, h, 1, rows);
            int gateUp = step.Op(GraphOp.MulMatId, step.Experts(LayerName(layer, "ffn_gate_up_exps.weight")), input, selected);   // [2 ff, K, rows]
            int g = step.Cont(step.View3(gateUp, ff, k, rows, 2L * ff * f, 2L * ff * k * f, 0));
            int u = step.Cont(step.View3(gateUp, ff, k, rows, 2L * ff * f, 2L * ff * k * f, ff * f));
            int act = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, step.Op(GraphOp.Scale, stackalloc[] { g }, default, stackalloc[] { 1f / _situBeta })) },
                default, stackalloc[] { _situBeta });
            act = step.Op(GraphOp.Mul, act, step.Op(GraphOp.Sigmoid, g));
            if (_situLinearBeta > 0)
                u = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, step.Op(GraphOp.Scale, stackalloc[] { u }, default, stackalloc[] { 1f / _situLinearBeta })) },
                    default, stackalloc[] { _situLinearBeta });
            act = step.Op(GraphOp.Mul, act, u);                                                                       // [ff, K, rows]
            int down = step.Op(GraphOp.MulMatId, step.Experts(LayerName(layer, "ffn_down_exps.weight")), act, selected);       // [h, K, rows]
            down = step.Op(GraphOp.Mul, down, weights);
            long rowBytes = h * f;
            int output = -1;
            for (int j = 0; j < k; j++)
            {
                int slice = step.View2(down, h, rows, rowBytes * k, j * rowBytes);
                output = output < 0 ? slice : step.Op(GraphOp.Add, output, slice);
            }
            return step.Cont(output);
        }

        /// <summary>The padded vocabulary's first <c>vocab_size_unpadded</c> logits, softcapped: cap tanh(logits / cap).</summary>
        protected override int EmitLogits(GraphStep step, int logits, int rows)
        {
            long f = sizeof(float);
            int x = step.Cont(step.View2(logits, Config.VocabSize, rows, _vocabPadded * f, 0));
            if (_softcap > 0)
                x = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, step.Op(GraphOp.Scale, stackalloc[] { x }, default, stackalloc[] { 1f / _softcap })) },
                    default, stackalloc[] { _softcap });
            return x;
        }
    }
}
