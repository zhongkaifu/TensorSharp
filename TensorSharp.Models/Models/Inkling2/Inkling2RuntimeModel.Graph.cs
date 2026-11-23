// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
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
            // In place on the ggml executors; the tensor executor's plan frees a node after its last reader, which an in-place
            // write into a live node defeats, so it rebuilds the stream.
            if (!StreamRebuilt && _backend is BackendType.GgmlCuda or BackendType.GgmlCpu or BackendType.GgmlMetal or BackendType.GgmlVulkan)
                return WriteSlots(step, stream, from);
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

        private static readonly bool StreamRebuilt = Environment.GetEnvironmentVariable("TS_INKLING_STREAM_REBUILD") == "1";

        // The current slot's value the last FFN wrote (the next run's entry value), as its own tensor: a capture must not read
        // it through the stream, which later runs overwrite in place.
        private GraphStep _curStep;
        private int _curLayer = -1, _curValue = -1;

        /// <summary>
        /// <see cref="Rebuild"/> in place: the slots that change are written into the stream by one SetRows whose source joins
        /// their values (a moved slot's old value copied out), so every read of the old stream feeds the one write and the
        /// slots that stay are not copied. The rows written are the stream's [hidden] rows r * slots + slot.
        /// </summary>
        private int WriteSlots(GraphStep step, int stream, int[] from)
        {
            int h = Config.HiddenSize, rows = step.Rows, slots = _streamSlots;
            int source = -1, changed = 0;
            Span<int> targets = stackalloc int[from.Length];
            for (int i = 0; i < from.Length; i++)
            {
                if (from[i] < 0 && ~from[i] == i) continue;
                int value = from[i] >= 0 ? step.Reshape(step.Cont(from[i]), h, rows) : StreamSlot(step, stream, ~from[i]);
                source = source < 0 ? value : step.Op(GraphOp.Concat, stackalloc[] { source, value }, stackalloc long[] { 1 });
                targets[changed++] = i;
            }
            if (changed == 0) return stream;
            var index = new int[changed * rows];
            for (int t = 0; t < changed; t++)
                for (int r = 0; r < rows; r++)
                    index[t * rows + r] = r * slots + targets[t];
            int ids = step.ConstantInputI32(index, index.Length);
            int dst = step.Reshape(stream, h, (long)slots * rows, 1);
            int written = step.Op(GraphOp.SetRows, dst, step.Reshape(source, h, (long)changed * rows, 1), ids);
            return step.Reshape(written, (long)slots * h, rows);
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
            step.BeginFused(GraphFusion.InklingMix, stackalloc[] { states, query }, default, stackalloc[] { MixEps, scale });
            int s = RmsNoWeight(step, step.Cont(states), MixEps);                                   // [h, n, rows]
            int scores = step.Op(GraphOp.MulMat, query, s);                                      // [queries, n, rows]
            scores = step.Cont(step.Permute(scores, 1, 0, 2, 3));                                    // [n, queries, rows]
            int probs = step.Op(GraphOp.SoftMaxExt, stackalloc[] { scores }, default, stackalloc[] { scale, 0f });
            int sT = step.Cont(step.Transpose(s));                                                   // [n, h, rows]
            int mix = step.Op(GraphOp.MulMat, sT, probs);                                            // [h, queries, rows]
            step.EndFused(stackalloc[] { mix });
            return mix;
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
            string down = LayerName(layer, $"loop_{controller}_ctrl_down.weight"), gainW = LayerName(layer, $"loop_{controller}_ctrl_gain.weight"),
                forgetW = LayerName(layer, $"loop_{controller}_ctrl_forget.weight");
            x = step.Cont(x);
            message = step.Cont(message);
            step.BeginFused(GraphFusion.InklingControl, stackalloc[] { x, message, step.Weight(down), step.Weight(gainW), step.Weight(forgetW) },
                default, stackalloc[] { MixEps });
            int g = step.Op(GraphOp.Concat, stackalloc[] { RmsNoWeight(step, x, MixEps), RmsNoWeight(step, message, MixEps) }, stackalloc long[] { 0 });
            g = step.MatMul(down, g);
            int gain = step.Op(GraphOp.Tanh, step.MatMul(gainW, g));
            int forget = step.Op(GraphOp.Tanh, step.MatMul(forgetW, g));
            int y = step.Op(GraphOp.Add, step.Op(GraphOp.Sub, x, step.Op(GraphOp.Mul, forget, x)), step.Op(GraphOp.Mul, gain, message));
            step.EndFused(stackalloc[] { y });
            return y;
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
            if (step.Rows > PackedProductRows)
            {
                // A prompt step's projections read the three inputs as they are.
                _inputStep = step;
                _inputLayer = layer;
                _inputQ = xq;
                _inputK = xk;
                _inputV = xv;
                return xq;
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
                from[Cur] = step.Op(GraphOp.Add, SlotView(step, stream, Cur), moe);
                // The next run's entry value of the current slot, as its own tensor (EmitCaptureRow).
                _curStep = step;
                _curLayer = layer;
                _curValue = from[Cur];
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
                from[Cur] = step.Op(GraphOp.Add, SlotView(step, stream, Cur), o);
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
            // Several sequences, all verifying or none: one fused node (InklingConvBatch) over their leaves, its ops the
            // per-sequence ones below without their own regions.
            int[] leaves = null, histories = null;
            bool batched = false;
            if (n > 1 && n <= 16 && !ConvBatchDisabled)
            {
                bool all = true, none = true;
                for (int s = 0; s < n; s++) { bool v = step.RecurrentSnapshots && step.Counts[s] > 1; all &= v; none &= !v; }
                if (all || none)
                {
                    batched = true;
                    leaves = new int[n];
                    histories = all ? new int[n] : null;
                    var inputs = new List<int> { x, kernel };
                    if (dynamic >= 0) inputs.Add(dynamic);
                    var ints = new List<long> { dynamic >= 0 ? heads : 0, all ? 1 : 0, n, offset, history };
                    for (int s = 0; s < n; s++) ints.Add(step.Counts[s]);
                    for (int s = 0; s < n; s++)
                    {
                        var st = (ShortConvState)step.SequenceState(s, layer);
                        leaves[s] = step.State(st.Conv, GraphFormat.F32, history, _convChannels);
                        inputs.Add(leaves[s]);
                        if (all)
                        {
                            if (st.EnsureSpecBuffers(step.Counts[s])) NoteGraphBindingsChanged();
                            histories[s] = step.State(st.ConvHistory, GraphFormat.F32, history + st.SnapshotCapacity, _convChannels);
                            inputs.Add(histories[s]);
                        }
                    }
                    step.BeginFused(GraphFusion.InklingConvBatch, CollectionsMarshal.AsSpan(inputs), CollectionsMarshal.AsSpan(ints), stackalloc[] { scale });
                }
            }
            for (int s = 0; s < n; s++)
            {
                int count = step.Counts[s];
                long first = step.First[s];
                // One sequence's rows as they lie (the kernel reads them at their stride); the region's ops copy them.
                long xRow = RowBytesOf(step, x, channels * f), dRow = dynamic < 0 ? 0 : RowBytesOf(step, dynamic, (long)taps * heads * f);
                int xsIn = n == 1 ? x : step.View2(x, channels, count, xRow, first * xRow);
                var state = (ShortConvState)step.SequenceState(s, layer);
                int leaf = batched ? leaves[s] : step.State(state.Conv, GraphFormat.F32, history, _convChannels);
                int window = step.View2(leaf, history, channels, history * f, offset * history * f);           // [taps - 1, channels]
                long width = (long)taps * heads;
                int dsIn = dynamic < 0 ? -1 : n == 1 ? dynamic : step.View2(dynamic, width, count, dRow, first * dRow);
                int xs = xsIn, ds = dsIn;
                // A verify window leaves the live window as it was and writes its inputs into the state's history at the window's
                // own pitch, the layer's four convolutions each at its channels: the commit takes the accepted rows' window
                // (CommitShortConvWindows). Its rows run as ops; a fused kernel advances the window.
                bool verifying = step.RecurrentSnapshots && count > 1;
                int historyLeaf = -1;
                long pitch = (history + count) * f;
                if (verifying)
                {
                    if (!batched)
                    {
                        if (state.EnsureSpecBuffers(count)) NoteGraphBindingsChanged();
                        historyLeaf = step.State(state.ConvHistory, GraphFormat.F32, history + state.SnapshotCapacity, _convChannels);
                    }
                    else historyLeaf = histories[s];
                    // The region writes the history itself (its kernel leaves the live window to the commit).
                    int historyView = step.View2(historyLeaf, history + count, channels, pitch, offset * pitch);
                    if (batched) { }
                    else if (ds >= 0)
                        step.BeginFused(GraphFusion.InklingConv, stackalloc[] { xsIn, window, kernel, dsIn, historyView }, stackalloc long[] { heads, 1 }, stackalloc[] { scale });
                    else
                        step.BeginFused(GraphFusion.InklingConv, stackalloc[] { xsIn, window, kernel, historyView }, stackalloc long[] { 0, 1 }, stackalloc[] { scale });
                }
                else if (batched) { }
                else if (ds >= 0)
                    step.BeginFused(GraphFusion.InklingConv, stackalloc[] { xsIn, window, kernel, dsIn }, stackalloc long[] { heads }, stackalloc[] { scale });
                else
                    step.BeginFused(GraphFusion.InklingConv, stackalloc[] { xsIn, window, kernel }, stackalloc long[] { 0 }, stackalloc[] { scale });
                // The region's ops read the rows contiguous.
                xs = step.Cont(xsIn);
                if (dsIn >= 0) ds = step.Cont(dsIn);
                int convIn = step.Op(GraphOp.Concat, stackalloc[] { window, step.Transpose(xs) }, stackalloc long[] { 0 });   // [taps - 1 + count, channels]
                int conv = step.Op(GraphOp.SsmConv, convIn, kernel);                                               // [channels, count]
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
                if (!batched) step.EndFused(stackalloc[] { y });
                output = output < 0 ? y : step.Op(GraphOp.Concat, stackalloc[] { output, y }, stackalloc long[] { 1 });
            }
            if (batched) step.EndFused(stackalloc[] { output });
            return output;
        }

        /// <summary>TS_INKLING_CONV_BATCH=0: a batched step's sequences convolve one fused node each.</summary>
        private static readonly bool ConvBatchDisabled = Environment.GetEnvironmentVariable("TS_INKLING_CONV_BATCH") == "0";

        // ---------------------------------------------------------------- attention

        protected override void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k) { }

        /// <summary>Rows of <paramref name="x"/> [d, heads, rows] scaled down to an L2 norm of at most
        /// <paramref name="threshold"/>, after the optional per-channel scale.</summary>
        private int Clip(GraphStep step, int x, string scale, float threshold)
        {
            // x may be a packed product's slice [d, heads, rows] at its row stride: the kernel reads it there, the region's
            // ops (other executors) copy it.
            if (scale != null)
                step.BeginFused(GraphFusion.InklingClip, stackalloc[] { x, step.Weight(scale) }, stackalloc long[] { 1 }, stackalloc[] { threshold });
            else
                step.BeginFused(GraphFusion.InklingClip, stackalloc[] { x }, stackalloc long[] { 0 }, stackalloc[] { threshold });
            int y = ClipOps(step, step.Cont(x), scale, threshold);
            step.EndFused(stackalloc[] { y });
            return y;
        }

        private int ClipOps(GraphStep step, int x, string scale, float threshold)
        {
            if (scale != null) x = step.Op(GraphOp.Mul, x, step.Weight(scale));
            int norm = step.Op(GraphOp.Sqrt, step.Op(GraphOp.SumRows, step.Op(GraphOp.Sqr, x)));                 // [1, heads, rows]
            int biased = step.Op(GraphOp.ScaleBias, stackalloc[] { norm }, default, stackalloc[] { 1f, 1e-6f });
            int ones = step.Op(GraphOp.ScaleBias, stackalloc[] { norm }, default, stackalloc[] { 0f, 1f });
            int factor = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Div, ones, biased) }, default, stackalloc[] { threshold });
            factor = step.Op(GraphOp.Clamp, stackalloc[] { factor }, default, stackalloc[] { 0f, 1f });
            return step.Op(GraphOp.Mul, x, factor);
        }

        /// <summary>The full layers' log-scaled positions [1, 1, rows]: 1 + alpha log(max(1, (pos + 1) / floor)), read from the
        /// table by the step's positions (an input the runtime advances), so a decode step advances without a trace.</summary>
        private int LogScale(GraphStep step)
        {
            if (step.Positions >= 0 && _tauTable != null)
            {
                int table = step.TensorWeight(_tauTable, 1, _tauTable.Sizes[0]);
                return step.Reshape(step.Op(GraphOp.GetRows, table, step.Positions), 1, 1, step.Rows);
            }
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
        private int _tableLayer = -1, _table = -1, _fusedSink = -1;
        // The fused attention's bias table (FusedAttention): B r - t [extent + 1, heads, rows], each (head, row) contiguous; its
        // element bytes (4, or 2 for a prompt step's F16 table).
        private int _fusedTable = -1, _fusedTableBytes = 4;

        /// <summary>TS_INKLING_REL_F16=0: prompt steps keep the F32 bias table.</summary>
        private static readonly bool RelTableF16Disabled = Environment.GetEnvironmentVariable("TS_INKLING_REL_F16") == "0";

        /// <summary>The attention runs as one kernel (GraphFusion.InklingAttention): an executor that serves it
        /// (<see cref="GraphCapabilities.RelativeBiasAttention"/>) over F16 or F32 caches, unless TS_INKLING_FUSED_ATTENTION=0.
        /// The kernel finds each row's keys from the positions, so no mask or distance index is built.</summary>
        private bool FusedAttention => GraphExecutor?.Capabilities.RelativeBiasAttention == true && !FusedAttentionDisabled
                                       && KvCacheGraphType is GraphFormat.F16 or GraphFormat.F32
                                       && _headDim == 128 && Config.NumHeads % Config.NumKVHeads == 0 && Config.NumHeads / Config.NumKVHeads <= 8;

        private static readonly bool FusedAttentionDisabled = Environment.GetEnvironmentVariable("TS_INKLING_FUSED_ATTENTION") == "0";

        /// <summary>Decode steps depend on the positions only through the step's own inputs once the attention reads them on
        /// the device (its keys found from the positions, tau read by them): such a step advances without a trace.</summary>
        protected override bool DecodeStepsAdvance => FusedAttention;

        // The fused attention reads the cache windows at their strides: a decode step copies none.
        protected override bool AttendsStridedWindows => FusedAttention;

        private static int Input(GraphStep step, int packed, int j, int hidden)
        {
            long f = sizeof(float);
            return step.Cont(step.View2(packed, hidden, step.Rows, 3L * hidden * f, j * hidden * f));
        }

        /// <summary>Rows a step may read the packed projections with in one product a distinct input (decode steps, batched
        /// sequences, verify windows); longer steps take one product a projection.</summary>
        private const int PackedProductRows = 8;

        /// <summary>Columns ggml's vector product kernel takes in one pass (MMVQ_MAX_BATCH_SIZE).</summary>
        private const int VectorProductColumns = 8;

        // The gate projection of the current attention, when the packed product made it (EmitAttentionOutput reads it).
        private GraphStep _gateStep;
        private int _gateLayer = -1, _gate = -1;

        // A prompt step's attention inputs, unpacked (EmitBlockInput keeps them for the projections and the gate).
        private GraphStep _inputStep;
        private int _inputLayer = -1, _inputQ = -1, _inputK = -1, _inputV = -1;

        private bool InputsKept(GraphStep step, int layer) => ReferenceEquals(step, _inputStep) && layer == _inputLayer;

        /// <summary>Rows [first, first + count) of the step's columns of y [total, ...]: column <paramref name="column"/> of
        /// the first step row, then every <paramref name="stride"/>-th. Contiguous for one row; copied for more.</summary>
        private static int Slice(GraphStep step, int y, long total, long first, long count, int column, int stride)
        {
            const long f = sizeof(float);
            if (step.Rows == 1) return step.View2(y, count, 1, count * f, (column * total + first) * f);
            return step.Cont(step.View2(y, count, step.Rows, stride * total * f, (column * total + first) * f));
        }

        /// <summary>A slice as <see cref="Slice"/>'s, but a view at its row stride (never copied): for consumers that read rows at
        /// a stride (the clip and convolution regions).</summary>
        private int SliceView(GraphStep step, int y, long total, long first, long count, int column, int stride)
        {
            const long f = sizeof(float);
            int view = step.View2(y, count, step.Rows, stride * total * f, (column * total + first) * f);
            if (!ReferenceEquals(step, _sliceStep)) { _sliceStep = step; _sliceRows.Clear(); }
            _sliceRows[view] = stride * total * f;
            return view;
        }

        // The current step's strided slices and their row bytes (SliceView): Heads and the convolutions read them at their stride.
        private GraphStep _sliceStep;
        private readonly Dictionary<int, long> _sliceRows = new();

        /// <summary>The row bytes of <paramref name="x"/>: a strided slice's, else <paramref name="dense"/>.</summary>
        private long RowBytesOf(GraphStep step, int x, long dense)
            => ReferenceEquals(step, _sliceStep) && _sliceRows.TryGetValue(x, out long bytes) ? bytes : dense;

        /// <summary>x [d heads, rows] as [d, heads, rows]: a strided slice as a view at its row stride, else reshaped.</summary>
        private int Heads(GraphStep step, int x, int d, int heads, int rows)
        {
            const long f = sizeof(float);
            return ReferenceEquals(step, _sliceStep) && _sliceRows.TryGetValue(x, out long bytes)
                ? step.View3(x, d, heads, rows, d * f, bytes, 0)
                : step.Reshape(x, d, heads, rows);
        }

        /// <summary>The projections from the packed weight: q, gate, rel, sink, k, k_conv, v, v_conv (-1 where absent).</summary>
        private void PackedProjections(GraphStep step, int layer, int x, int xq, int xk, int xv, ProjectionRows p, out int qy,
            out int gy, out int ry, out int ty, out int ky, out int kDyn, out int vy, out int vDyn)
        {
            int h = Config.HiddenSize, rows = step.Rows;
            string w = LayerName(layer, "attn_proj.weight");
            int qn = p.Gate - p.Q, gn = p.Rel - p.Gate, rn = p.Sink - p.Rel, tn = p.K - p.Sink;
            int kn = (p.KConv >= 0 ? p.KConv : p.V) - p.K, vn = (p.VConv >= 0 ? p.VConv : p.Total) - p.V;
            // A sliding layer of the first pass reads one input for all three.
            bool shared = IsSliding(layer) && !ReadsMessages(layer);
            if (rows <= PackedProductRows && (shared || 3 * rows <= VectorProductColumns))
            {
                // One product: one column a row where the inputs are shared, else the three inputs as columns 3r, 3r + 1, 3r + 2
                // (the packed x [3h, rows] is exactly that as [h, 3 rows]).
                int y = shared ? step.MatMul(w, xq) : step.MatMul(w, step.Reshape(x, h, 3 * rows));
                int stride = shared ? 1 : 3;
                int kc = shared ? 0 : 1, vc = shared ? 0 : 2;
                // q and rel go to the clip and k, v and their taps to the convolutions as views (read at their strides).
                qy = SliceView(step, y, p.Total, p.Q, qn, 0, stride);
                gy = Slice(step, y, p.Total, p.Gate, gn, 0, stride);
                ry = SliceView(step, y, p.Total, p.Rel, rn, 0, stride);
                ty = Slice(step, y, p.Total, p.Sink, tn, 0, stride);
                ky = SliceView(step, y, p.Total, p.K, kn, kc, stride);
                kDyn = p.KConv >= 0 ? SliceView(step, y, p.Total, p.KConv, p.V - p.KConv, kc, stride) : -1;
                vy = SliceView(step, y, p.Total, p.V, vn, vc, stride);
                vDyn = p.VConv >= 0 ? SliceView(step, y, p.Total, p.VConv, p.Total - p.VConv, vc, stride) : -1;
                return;
            }
            if (rows <= PackedProductRows)
            {
                // One product a distinct input.
                int a = step.MatMulRows(w, p.Q, p.K - p.Q, xq);
                int b = step.MatMulRows(w, p.K, p.KRows, xk);
                int c = step.MatMulRows(w, p.V, p.VRows, xv);
                qy = SliceView(step, a, p.K, 0, qn, 0, 1);
                gy = Slice(step, a, p.K, p.Gate, gn, 0, 1);
                ry = SliceView(step, a, p.K, p.Rel, rn, 0, 1);
                ty = Slice(step, a, p.K, p.Sink, tn, 0, 1);
                ky = SliceView(step, b, p.KRows, 0, kn, 0, 1);
                kDyn = p.KConv >= 0 ? SliceView(step, b, p.KRows, kn, p.V - p.KConv, 0, 1) : -1;
                vy = SliceView(step, c, p.VRows, 0, vn, 0, 1);
                vDyn = p.VConv >= 0 ? SliceView(step, c, p.VRows, vn, p.Total - p.VConv, 0, 1) : -1;
                return;
            }
            // A prompt chunk: one product a projection, as with separate weights.
            qy = step.MatMulRows(w, p.Q, qn, xq);
            gy = step.MatMulRows(w, p.Gate, gn, xq);
            ry = step.MatMulRows(w, p.Rel, rn, xq);
            ty = step.MatMulRows(w, p.Sink, tn, xq);
            ky = step.MatMulRows(w, p.K, kn, xk);
            kDyn = p.KConv >= 0 ? step.MatMulRows(w, p.KConv, p.V - p.KConv, xk) : -1;
            vy = step.MatMulRows(w, p.V, vn, xv);
            vDyn = p.VConv >= 0 ? step.MatMulRows(w, p.VConv, p.Total - p.VConv, xv) : -1;
        }

        protected override bool EmitAttentionProjections(GraphStep step, int layer, int x, out int q, out int k, out int v)
        {
            int h = Config.HiddenSize, heads = Config.NumHeads, kvHeads = Config.NumKVHeads, d = _headDim, rows = step.Rows;
            bool sliding = IsSliding(layer);
            bool kept = InputsKept(step, layer);
            int xq = kept ? _inputQ : Input(step, x, 0, h), xk = kept ? _inputK : Input(step, x, 1, h), xv = kept ? _inputV : Input(step, x, 2, h);

            int qy, ry, ty, ky, kDyn, vy, vDyn;
            if (_projections?[PhysicalLayer(layer)] is { } packed)
            {
                PackedProjections(step, layer, x, xq, xk, xv, packed, out qy, out int gy, out ry, out ty, out ky, out kDyn, out vy, out vDyn);
                _gateStep = step;
                _gateLayer = layer;
                _gate = gy;
            }
            else
            {
                qy = step.MatMul(LayerName(layer, "attn_q.weight"), xq);
                kDyn = sliding ? -1 : step.MatMul(LayerName(layer, "attn_k_conv_proj.weight"), xk);
                ky = step.MatMul(LayerName(layer, "attn_k.weight"), xk);
                vDyn = sliding ? -1 : step.MatMul(LayerName(layer, "attn_v_conv_proj.weight"), xv);
                vy = step.MatMul(LayerName(layer, "attn_v.weight"), xv);
                ry = step.MatMul(LayerName(layer, "attn_rel.weight"), xq);
                ty = step.MatMul(LayerName(layer, "attn_sink_proj.weight"), xq);
                _gateLayer = -1;
            }
            q = Clip(step, Heads(step, qy, d, heads, rows), LayerName(layer, "attn_q_clip.weight"), _qkClip);
            int kc = Convolve(step, layer, ky, 0, _kvDim, "attn_k_conv.weight", kDyn, kvHeads);
            k = Clip(step, step.Reshape(kc, d, kvHeads, rows), null, _qkClip);
            int vc = Convolve(step, layer, vy, _kvDim, _kvDim, "attn_v_conv.weight", vDyn, kvHeads);
            v = RmsNoWeight(step, step.Reshape(vc, d, kvHeads, rows), Config.Eps);
            int r = Clip(step, Heads(step, ry, _relDim, heads, rows),
                LayerName(layer, "attn_rel_clip.weight"), _relClip);
            if (!sliding)
            {
                int tau = LogScale(step);
                q = step.Op(GraphOp.Mul, q, tau);
                r = step.Op(GraphOp.Mul, r, tau);
            }
            // The sink logit t is per query: exp(s - t) / (1 + sum exp(s - t)) = exp(s) / (exp(t) + sum exp(s)), so every
            // key's bias takes - t and a zero sink closes the softmax (llama.cpp's inkling2.cpp).
            int t = ty;                                                                                               // [heads, rows]
            int extent = ExtentOf(layer);
            if (FusedAttention)
            {
                // The kernel gathers each score's bias from the row's table: no permute, no distance index.
                // The kernel subtracts each row's sink logit t itself (_fusedSink), so the table is not written twice.
                int relW = step.TensorWeight(_relTables[PhysicalLayer(layer)], _relDim, extent + 1);
                // Prompt rows (every sequence past 16 rows: the kernels that read either) take the table in F16, half the bytes
                // written and read (a 4,096-row step's table is 403 MB in F32 at each layer).
                _fusedTableBytes = !RelTableF16Disabled && Array.TrueForAll(step.Counts, c => c > 16) ? 2 : 4;
                if (_fusedTableBytes == 2)
                {
                    step.BeginFused(GraphFusion.InklingRelTable, stackalloc[] { relW, r }, default, default);
                    int t32 = step.Op(GraphOp.MulMat, relW, r);
                    int t16 = step.Op(GraphOp.Cast, stackalloc[] { t32 }, stackalloc long[] { GraphFormat.F16 });
                    step.EndFused(stackalloc[] { t16 });
                    _fusedTable = t16;                                                                                // [extent + 1, heads, rows] F16
                }
                else
                    _fusedTable = step.Op(GraphOp.MulMat, relW, r);                                                   // [extent + 1, heads, rows]
                _fusedSink = step.Cont(t);                                                                            // [heads, rows]
                _table = -1;
                _tableStep = step;
                _tableLayer = layer;
                q = step.Cont(q);
                k = step.Cont(k);
                v = step.Cont(v);
                return true;
            }
            int table = step.Op(GraphOp.MulMat, step.TensorWeight(_relTables[PhysicalLayer(layer)], _relDim, extent + 1), r);   // [extent + 1, heads, rows]
            table = step.Op(GraphOp.Sub, table, step.Reshape(t, 1, heads, rows));
            _table = step.Cont(step.Permute(table, 1, 0, 2, 3));                                                    // [heads, extent + 1, rows]
            _fusedTable = -1;
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
            // The fused kernel reads the queries at their strides (a sequence's view of the step's rows): no copy.
            if (_fusedTable >= 0)
                return FusedAttentionCore(step, layer, s, q, k, v, layout, scale, count, extent);
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

        /// <summary>
        /// One sequence's attention as one kernel (GraphFusion.InklingAttention): the keys a row sees come from the layout
        /// (a ring's cells or a linear window from its first position), the band and the attention floor, and each score's
        /// bias is gathered from the row's table (B r - t by distance). The region's ops are a placeholder of the result's shape: the family emits it
        /// only where it is served (<see cref="FusedAttention"/>), and the executor fails the build where it cannot serve it.
        /// </summary>
        private int FusedAttentionCore(GraphStep step, int layer, int s, int qc, int k, int v, in AttentionKeys layout, float scale,
            int count, int extent)
        {
            int heads = Config.NumHeads, d = _headDim;
            const long f = sizeof(float);
            long first = step.First[s], width = extent + 1;
            long tb = _fusedTableBytes;
            int table = step.Sequences == 1 ? _fusedTable
                : step.View3(_fusedTable, width, heads, count, width * tb, width * heads * tb, first * width * heads * tb);
            int sink = step.Sequences == 1 ? _fusedSink : step.View2(_fusedSink, heads, count, heads * f, first * heads * f);
            long ring = layout.Ring, firstKey = layout.Ring > 0 ? 0 : layout.FirstPosition;
            step.BeginFused(GraphFusion.InklingAttention, stackalloc[] { qc, k, v, table, step.Positions, sink },
                stackalloc long[] { ring, extent, SlidingWindow(layer), firstKey, step.FloorOf(s), first }, stackalloc[] { scale });
            int placeholder = step.Op(GraphOp.Scale, stackalloc[] { step.Cont(step.Permute(qc, 0, 2, 1, 3)) }, default, stackalloc[] { 0f });
            step.EndFused(stackalloc[] { placeholder });
            return placeholder;                                                                                       // [dv, heads, count]
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
            int projected = ReferenceEquals(step, _gateStep) && layer == _gateLayer ? _gate
                : step.MatMul(LayerName(layer, "attn_gate.weight"), InputsKept(step, layer) ? _inputQ : Input(step, x, 0, Config.HiddenSize));
            int gate = step.Op(GraphOp.Sigmoid, projected);
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
            int bias = step.Weight(LayerName(layer, "exp_probs_b.bias"));
            // Tests that force the routes or read them back take them from the router's ops: no fused router then.
            bool fusedRoute = ForcedRoutesForTests == null && DeviceRoutesForTests == null;
            if (fusedRoute)
                step.BeginFused(GraphFusion.InklingRoute, stackalloc[] { logits, bias }, stackalloc long[] { k }, stackalloc[] { _expertScale });
            int probs = step.Op(GraphOp.Sigmoid, logits);
            int selection = step.Op(GraphOp.Add, probs, bias);
            int forced = EmitForcedRoutes(step, layer, k, experts);                                                  // tests
            int selected = forced >= 0 ? forced : step.Op(GraphOp.TopK, stackalloc[] { selection }, stackalloc long[] { k });   // [K, rows]
            EmitDeviceRoutes(step, layer, selected, selection, k, experts);                                          // tests
            int picked = step.Reshape(step.Op(GraphOp.GetRows, step.Reshape(probs, 1, experts, rows), selected), k, rows);
            int sum = step.Op(GraphOp.ScaleBias, stackalloc[] { step.Op(GraphOp.SumRows, picked) }, default, stackalloc[] { 1f, 1e-20f });
            int weights = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Div, picked, sum) }, default, stackalloc[] { _expertScale });
            weights = step.Reshape(weights, 1, k, rows);
            if (fusedRoute) step.EndFused(stackalloc[] { selected, weights });

            int input = step.Reshape(x, h, 1, rows);
            int gateUp = ExpertProduct(step, LayerName(layer, "ffn_gate_up_exps.weight"), input, selected);          // [2 ff, K, rows]
            // NVFP4 experts: each expert's outer scale (amax / denominator) multiplies its products; gate/up's before SITU,
            // down's through the route weights.
            int upScale = ExpertScales(step, layer, "ffn_gate_up_exps.scale", selected, k, rows);                          // [1, K, rows]
            int downScale = ExpertScales(step, layer, "ffn_down_exps.scale", selected, k, rows);
            if (downScale >= 0) weights = step.Op(GraphOp.Mul, weights, downScale);
            if (upScale >= 0)
                step.BeginFused(GraphFusion.InklingSitu, stackalloc[] { gateUp, upScale }, default, stackalloc[] { _situBeta, _situLinearBeta });
            else
                step.BeginFused(GraphFusion.InklingSitu, stackalloc[] { gateUp }, default, stackalloc[] { _situBeta, _situLinearBeta });
            if (upScale >= 0) gateUp = step.Op(GraphOp.Mul, gateUp, upScale);
            int g = step.Cont(step.View3(gateUp, ff, k, rows, 2L * ff * f, 2L * ff * k * f, 0));
            int u = step.Cont(step.View3(gateUp, ff, k, rows, 2L * ff * f, 2L * ff * k * f, ff * f));
            int act = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, step.Op(GraphOp.Scale, stackalloc[] { g }, default, stackalloc[] { 1f / _situBeta })) },
                default, stackalloc[] { _situBeta });
            act = step.Op(GraphOp.Mul, act, step.Op(GraphOp.Sigmoid, g));
            if (_situLinearBeta > 0)
                u = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, step.Op(GraphOp.Scale, stackalloc[] { u }, default, stackalloc[] { 1f / _situLinearBeta })) },
                    default, stackalloc[] { _situLinearBeta });
            act = step.Op(GraphOp.Mul, act, u);                                                                       // [ff, K, rows]
            step.EndFused(stackalloc[] { act });
            int down = ExpertProduct(step, LayerName(layer, "ffn_down_exps.weight"), act, selected);                  // [h, K, rows]
            step.BeginFused(GraphFusion.InklingExpertSum, stackalloc[] { down, weights });
            down = step.Op(GraphOp.Mul, down, weights);
            long rowBytes = h * f;
            int output = -1;
            for (int j = 0; j < k; j++)
            {
                int slice = step.View2(down, h, rows, rowBytes * k, j * rowBytes);
                output = output < 0 ? slice : step.Op(GraphOp.Add, output, slice);
            }
            output = step.Cont(output);
            step.EndFused(stackalloc[] { output });
            return output;
        }

        /// <summary>The routed experts' product (MulMatId) as the InklingExperts region: served for a few rows, each expert
        /// the rows picked read once for all of its pairs.</summary>
        private static int ExpertProduct(GraphStep step, string name, int input, int selected)
        {
            int experts = step.Experts(name);
            step.BeginFused(GraphFusion.InklingExperts, stackalloc[] { experts, input, selected });
            int product = step.Op(GraphOp.MulMatId, experts, input, selected);
            step.EndFused(stackalloc[] { product });
            return product;
        }

        /// <summary>The selected experts' outer scales [1, K, rows] from a per-expert F32 vector, or -1 where the layer has none.</summary>
        private int ExpertScales(GraphStep step, int layer, string suffix, int selected, int k, int rows)
        {
            string name = LayerName(layer, suffix);
            if (!_weights.TryGetValue(name, out Tensor vector)) return -1;
            int table = step.Reshape(step.Weight(name), 1, vector.ElementCount());                                     // [1, E]
            int picked = step.Op(GraphOp.GetRows, table, step.Reshape(selected, (long)k * rows));                    // [1, K rows]
            return step.Reshape(picked, 1, k, rows);
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
