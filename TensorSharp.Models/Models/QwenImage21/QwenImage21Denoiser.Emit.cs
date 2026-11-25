// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.QwenImage21
{
    /// <summary>A graph emitter with what Qwen-Image-2.1's steps need beyond the shared one: state leaves over host blocks or
    /// tensors (the prefix cache) and an input read from host memory a LoRA set owns (a per-step output head).</summary>
    internal sealed class QwenImage21Emitter : GraphEmitter
    {
        public QwenImage21Emitter(IGraphExecutor executor, bool record) : base(executor, record) { }

        /// <summary>A state leaf over <paramref name="bytes"/> of host memory at <paramref name="data"/> (an executor binding host
        /// bytes) or over <paramref name="tensor"/> (one binding objects).</summary>
        public int State(int format, ReadOnlySpan<long> ne, IntPtr data, long bytes, Tensor tensor)
            => tensor == null
                ? Leaf(GraphLeafRole.State, format, ne, data, bytes)
                : Leaf(GraphLeafRole.State, format, ne, GraphIdentity.Of(tensor.Storage, tensor.StorageOffset), tensor.Storage.ByteLength, null, tensor);

        /// <summary>An input read every run from <paramref name="bytes"/> of host memory at <paramref name="data"/>.</summary>
        public int InputHost(int format, ReadOnlySpan<long> ne, IntPtr data, long bytes)
        {
            int node = Leaf(GraphLeafRole.Input, format, ne, IntPtr.Zero, 0);
            RecordInput(node, data, bytes, null, null);
            return node;
        }
    }

    public sealed partial class QwenImage21Denoiser
    {
        private const long PrecisionF32 = 10;   // ggml's GGML_PREC_F32

        /// <summary>The retired graph's ggml_cuda offload (its TS_QWEN21_CUBLAS_MATMUL / _MIN_ROWS defaults): Q4_K and Q6_K
        /// products of 256 rows or more on cuBLAS BF16 (int8 or FP8 where --diffusion-products asks), every unmasked attention
        /// on cuDNN whatever its query's strides, BF16 weights on ggml's own products.</summary>
        internal static readonly GraphWideTokenRoutes CudaRoutes = new(MinRows: 256, Q6K: true, StridedAttention: true, FloatWeightsOnGgml: true);
        private const int GgmlF16 = 1;

        /// <summary>What one step's program reads.</summary>
        private sealed class StepValues
        {
            public Mode Mode;
            public QwenImage21Layout Layout;
            public PrefixCache Cache;
            public QwenImage21LoraSet Lora;
            public int Step;
            public float[] Images;   // [channels, ImageRows]: references then the target (the target alone when cached)
            public int ImageRows;
            public float[] Text;     // [text dim, TextRows] (null when cached)
            public int TextRows;
            public float[] Time;     // [256, 2]
            public float[] Cos, Sin; // [head/2, rows] from the first computed row
        }

        // The F16 masks of a layout's segments, made once a layout and style (a mask per segment that needs one, else null).
        private readonly ConditionalWeakTable<QwenImage21Layout, ushort[][]> _masks = new(), _cachedMasks = new();

        /// <summary>The program of one prediction: the retired graph's nodes, in its order (ggml_ops_qwen_image21.cpp build_graph).</summary>
        /// <param name="rank">The tensor-parallel rank whose program this is (its heads, MLP columns and stored prefix), or -1
        /// on one device.</param>
        private void Emit(QwenImage21Emitter e, StepValues s, int rank)
        {
            GraphProgramBuilder b = e.Builder;
            QwenImage21Weights w = _w;
            int r0 = Math.Max(rank, 0);
            int dim = w.Dim, heads = w.Heads / _ranks, hd = w.HeadDim, half = hd / 2;
            QwenImage21Block[] blocks = rank >= 0 && _rankBlocks != null ? _rankBlocks[rank] : w.Blocks;
            float eps = QwenImage21Weights.Eps;
            QwenImage21Layout layout = s.Layout;
            bool cached = s.Mode == Mode.Cached;
            QwenImage21Segment targetSegment = layout.Target;
            int first = cached ? layout.Prefix : 0;
            int seq = layout.Rows - first;
            int prefix = cached ? 0 : layout.Prefix;
            IReadOnlyList<QwenImage21Segment> segments = cached ? new[] { targetSegment } : layout.Segments;
            QwenImage21LoraSet lora = s.Lora;
            // Every product of a step has an image's rows (the time MLP's two aside): ggml_cuda's wide routes as the retired
            // graph took them, the managed executor's full worker width.
            e.MarkWideTokenProducts(CudaRoutes);
            // A plan binds one LoRA set's factors: another set compiles its own even where it reuses the addresses.
            if (lora != null) e.Mix(lora.Generation);
            e.Mix((long)s.Mode);

            var leaves = new Dictionary<IntPtr, int>();
            int Leaf(QwenImage21Weight x)
            {
                if (leaves.TryGetValue(x.Data, out int node)) return node;
                ReadOnlySpan<long> ne = x.Vector ? stackalloc long[] { x.Ne0 } : stackalloc long[] { x.Ne0, x.Ne1 };
                node = _tensorAllocator == null
                    ? e.HostWeight(x.Format, ne, x.Data, x.Bytes)
                    : e.HostWeight(x.Format, ne, GraphIdentity.Of(Handle(x)), x.Bytes, Handle(x));
                leaves[x.Data] = node;
                return node;
            }
            QwenImage21Weight VectorAt(IntPtr data, long count) => new(GraphFormat.F32, count, 1, data, count * sizeof(float), Vector: true);

            int View2(int x, long ne0, long ne1, long nb1, long offset) =>
                b.Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 2, ne0, ne1, 1, 1, nb1, ne1 * nb1, ne1 * nb1, offset });
            int View3(int x, long ne0, long ne1, long ne2, long nb1, long nb2, long offset) =>
                b.Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 3, ne0, ne1, ne2, 1, nb1, nb2, ne2 * nb2, offset });
            int View4(int x, long ne0, long ne1, long ne2, long ne3, long nb1, long nb2, long nb3, long offset) =>
                b.Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 4, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset });
            int Reshape(int x, params long[] ne) => b.Op(GraphOp.Reshape, stackalloc[] { x }, ne);
            int Unary(GraphOp op, int x) => b.Op(op, stackalloc[] { x });
            int ScaleBias(int x, float scale, float bias) => b.Op(GraphOp.ScaleBias, stackalloc[] { x }, default, stackalloc[] { scale, bias });
            int Norm(int x) => b.Op(GraphOp.Norm, stackalloc[] { x }, default, stackalloc[] { eps });
            int RmsNorm(int x) => b.Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps });
            int Concat(int a, int c, int axis) => b.Op(GraphOp.Concat, stackalloc[] { a, c }, stackalloc long[] { axis });
            int Permute(int x, int a0, int a1, int a2, int a3) => b.Op(GraphOp.Permute, stackalloc[] { x }, stackalloc long[] { a0, a1, a2, a3 });
            int Cast(int x, int format) => b.Op(GraphOp.Cast, stackalloc[] { x }, stackalloc long[] { format });
            // A [width, rows] F32 contiguous tensor's rows [start, start + length).
            int Slice(int x, long width, long start, long length) => View2(x, width, length, width * 4, start * width * 4);

            // ---- LoRA (the retired graph's Builder.shrink / adapt / linear)
            int Product(int factor, int x) => b.Op(GraphOp.MulMat, stackalloc[] { factor, x }, stackalloc long[] { PrecisionF32 });
            int Factor(IntPtr data, int format, long ne0, long ne1)
            {
                long bytes = ne0 * ne1 * (format == GgmlF16 ? 2 : 4);
                return Leaf(new QwenImage21Weight(format, ne0, ne1, data, bytes));
            }
            // The down projections of `updates` applied to one input: [rank, rows] each, -1 where an update has no low-rank
            // term. Updates whose down factors lie back to back run as one stacked shrink; an update that reuses the previous
            // one's factor (a fused gate_up LoRA seen as its two halves) reuses its product.
            int[] Shrink(QwenImage21Lora[] updates, int x, long xRows)
            {
                var result = new int[updates.Length];
                Array.Fill(result, -1);
                for (int i = 0; i < updates.Length;)
                {
                    QwenImage21Lora f = updates[i];
                    if (f.Rank <= 0) { ++i; continue; }
                    long elem = f.Type == GgmlF16 ? 2 : 4;
                    int end = i + 1;
                    long rank = f.Rank;
                    IntPtr next = f.Down + checked((nint)(elem * f.In * f.Rank));
                    while (end < updates.Length && updates[end].Rank > 0 && updates[end].Type == f.Type && updates[end].In == f.In &&
                           updates[end].Down == next)
                    {
                        rank += updates[end].Rank;
                        next += checked((nint)(elem * updates[end].In * updates[end].Rank));
                        ++end;
                    }
                    int t = Product(Factor(f.Down, f.Type, f.In, rank), x);
                    long offset = 0;
                    for (int j = i; j < end; ++j)
                    {
                        result[j] = end - i == 1 ? t : View2(t, updates[j].Rank, xRows, rank * 4, offset * 4);
                        offset += updates[j].Rank;
                    }
                    if (end - i == 1)
                        while (end < updates.Length && updates[end].Rank > 0 && updates[end].Down == f.Down && updates[end].Rank == f.Rank &&
                               updates[end].Type == f.Type && updates[end].In == f.In)
                            result[end++] = t;
                    i = end;
                }
                return result;
            }
            int Linear(QwenImage21Weight weight, int x, long xRows, QwenImage21Lora update = default, int shrunk = -1)
            {
                int y = b.Op(GraphOp.MulMat, Leaf(weight), x);
                if (update.Rank <= 0 && update.RowScale == IntPtr.Zero) return y;
                if (update.Out != weight.Ne1) throw new InvalidOperationException("Qwen-Image-2.1: LoRA output shape mismatch.");
                if (update.RowScale != IntPtr.Zero) y = b.Op(GraphOp.Mul, y, Leaf(VectorAt(update.RowScale, update.Out)));
                if (update.Rank <= 0) return y;
                if (shrunk < 0) shrunk = Shrink(new[] { update }, x, xRows)[0];
                return b.Op(GraphOp.Add, y, Product(Factor(update.Up, update.Type, update.Rank, update.Out), shrunk));
            }
            QwenImage21Lora Global(string module) => lora?.Global(module, r0) ?? default;

            // ---- modulation: rows below the prefix take row 1 (t = 0), the rest row 0; a scale is 1 + m, a gate tanh(m).
            int Modulate(int x, int m, long rows, bool gate)
            {
                int target = View2(m, dim, 1, dim * 4, 0);
                int preceding = View2(m, dim, 1, dim * 4, dim * 4);
                target = gate ? Unary(GraphOp.Tanh, target) : ScaleBias(target, 1f, 1f);
                preceding = gate ? Unary(GraphOp.Tanh, preceding) : ScaleBias(preceding, 1f, 1f);
                int result = b.Op(GraphOp.Mul, Slice(x, dim, prefix, rows - prefix), target);
                if (prefix > 0) result = Concat(b.Op(GraphOp.Mul, Slice(x, dim, 0, prefix), preceding), result, 1);
                return result;
            }

            // ---- rotary: pairs (2j, 2j + 1) rotated by the row's table; the even results first, then the odd (the same
            // half-split permutation of q and k leaves their products unchanged).
            int cos = -1, sin = -1;
            bool managed = _style == Style.Managed;
            // A PairRotary region: its ops are the retired graph's (a GGML backend builds them as they are); the managed
            // executor computes the region in one pass.
            int Rope(int x)
            {
                b.BeginFused(GraphFusion.PairRotary, stackalloc[] { x, cos, sin });
                int x4 = Reshape(x, 2, half, heads, seq);
                long nb1 = 8, nb2 = hd * 4L, nb3 = (long)hd * heads * 4;
                int even = View4(x4, 1, half, heads, seq, nb1, nb2, nb3, 0);
                int odd = View4(x4, 1, half, heads, seq, nb1, nb2, nb3, 4);
                int c = Reshape(cos, 1, half, 1, seq);
                int sn = Reshape(sin, 1, half, 1, seq);
                int ep = Reshape(b.Op(GraphOp.Sub, b.Op(GraphOp.Mul, even, c), b.Op(GraphOp.Mul, odd, sn)), half, heads, seq);
                int op = Reshape(b.Op(GraphOp.Add, b.Op(GraphOp.Mul, odd, c), b.Op(GraphOp.Mul, even, sn)), half, heads, seq);
                int rotated = Concat(ep, op, 0);
                b.EndFused(stackalloc[] { rotated });
                return rotated;
            }

            // ---- inputs
            int images = e.InputF32(s.Images, w.Channels, s.ImageRows);
            int text = cached ? -1 : e.InputF32(s.Text, w.TextDim, s.TextRows);
            int time = e.InputF32(s.Time, QwenImage21Weights.TimestepChannels, 2);
            cos = e.InputF32(s.Cos, half, seq);
            sin = e.InputF32(s.Sin, half, seq);
            ushort[][] maskValues = MasksFor(layout, cached, segments);
            var masks = new int[segments.Count];
            for (int i = 0; i < segments.Count; i++)
                masks[i] = maskValues[i] == null ? -1
                    : e.InputF16(maskValues[i], MaskKeys(segments[i]), MaskQueries(segments[i]));

            // ---- time MLP, shared AdaLN
            time = Unary(GraphOp.Silu, Linear(w.TimeOut,
                Unary(GraphOp.Silu, Linear(w.TimeIn, time, 2, Global("time_text_embed.timestep_embedder.linear_1"))), 2,
                Global("time_text_embed.timestep_embedder.linear_2")));
            int mod = Linear(w.Modulation, time, 2, Global("modulation.1"));
            var modulation = new int[4];
            for (int i = 0; i < 4; i++)
                modulation[i] = Unary(GraphOp.Cont, View2(mod, dim, 2, 4L * dim * 4, (long)i * dim * 4));
            images = Linear(w.ImageIn, images, s.ImageRows, Global("img_in"));
            int joint;
            if (cached)
            {
                joint = images;
            }
            else
            {
                IntPtr textNorm = lora != null && lora.TextNorm != IntPtr.Zero ? lora.TextNorm : w.TextNorm.Data;
                int norm = ScaleBias(Leaf(textNorm == w.TextNorm.Data ? w.TextNorm : VectorAt(textNorm, w.TextDim)), 1f, 1f);
                text = b.Op(GraphOp.Mul, RmsNorm(text), norm);
                text = Linear(w.TextOut, Unary(GraphOp.Gelu, Linear(w.TextIn, text, s.TextRows, Global("txt_in.in_layer"))), s.TextRows,
                    Global("txt_in.out_layer"));
                joint = -1;
                foreach (QwenImage21Segment g in segments)
                {
                    int h = Slice(g.IsImage ? images : text, dim, g.SourceStart, g.Length);
                    joint = joint < 0 ? h : Concat(joint, h, 1);
                }
            }

            // ---- the blocks
            int kvFormat = cached ? s.Cache.Format : GraphFormat.F32;
            for (int i = 0; i < w.Layers; i++)
            {
                QwenImage21Block blk = blocks[i];
                QwenImage21Weight gateW = blk.Gate, upW = blk.Up;
                if (lora != null && lora.SplitGateUp[i] && blk.FusedGateUp)
                {
                    gateW = blk.Gate.Rows(0, w.Ff);
                    upW = blk.Gate.Rows(w.Ff, w.Ff);   // (a sharded block is split already)
                }
                QwenImage21BlockLora l = lora != null ? lora.Block(i, r0) : default;
                bool hasLora = lora != null;
                int h = Modulate(Norm(joint), modulation[0], seq, gate: false);
                int[] qkv = hasLora ? Shrink(new[] { l.Q, l.K, l.V }, h, seq) : new[] { -1, -1, -1 };
                int q = Reshape(Linear(blk.Q, h, seq, l.Q, qkv[0]), hd, heads, seq);
                int k = Reshape(Linear(blk.K, h, seq, l.K, qkv[1]), hd, heads, seq);
                int v = Reshape(Linear(blk.V, h, seq, l.V, qkv[2]), hd, heads, seq);
                QwenImage21Weight normQ = lora != null && lora.NormQ[i] != IntPtr.Zero ? VectorAt(lora.NormQ[i], hd) : blk.NormQ;
                QwenImage21Weight normK = lora != null && lora.NormK[i] != IntPtr.Zero ? VectorAt(lora.NormK[i], hd) : blk.NormK;
                q = b.Op(GraphOp.Mul, RmsNorm(q), Leaf(normQ));
                k = b.Op(GraphOp.Mul, RmsNorm(k), Leaf(normK));
                q = Rope(q);
                k = Rope(k);
                if (s.Mode == Mode.Extract)
                {
                    // Store this layer's prefix rows as the attention reads them.
                    foreach (var (source, stored) in new[] { (k, CacheLeaf(e, s.Cache, i, key: true, r0)), (v, CacheLeaf(e, s.Cache, i, key: false, r0)) })
                    {
                        int rows = Permute(source, 0, 2, 1, 3);
                        rows = View3(rows, hd, layout.Prefix, heads, (long)hd * heads * 4, hd * 4L, 0);
                        e.Effect(b.Op(GraphOp.Cpy, rows, stored));
                    }
                }
                int qp = Permute(q, 0, 2, 1, 3);
                // kp/vp [D, keys, heads] and their strides.
                int kp, vp;
                long kNb1, kNb2;
                int kType;
                if (cached)
                {
                    kp = Joined(e, b, s.Cache, i, key: true, k, kvFormat, r0);
                    vp = Joined(e, b, s.Cache, i, key: false, v, kvFormat, r0);
                    long elem = kvFormat == GraphFormat.F16 ? 2 : 4;
                    kNb1 = hd * elem;
                    kNb2 = (long)hd * layout.Rows * elem;
                    kType = kvFormat;
                }
                else
                {
                    kp = Permute(k, 0, 2, 1, 3);
                    vp = Permute(v, 0, 2, 1, 3);
                    kNb1 = (long)hd * heads * 4;
                    kNb2 = hd * 4L;
                    kType = GraphFormat.F32;
                }
                int attention = Attention(b, qp, kp, vp, kNb1, kNb2, kType, first, segments, masks, hd, heads, seq, layout.Rows);
                h = Linear(blk.Out, attention, seq, l.Out);
                // A row-parallel rank's partial (its LoRA term included: the ranks' up * (down_r x_r) sum to the update).
                if (_ranks > 1) h = b.Op(GraphOp.AllReduce, h);
                joint = b.Op(GraphOp.Add, joint, Modulate(h, modulation[1], seq, gate: true));
                h = Modulate(Norm(joint), modulation[2], seq, gate: false);
                int Mlp(int x, long rows)
                {
                    int activated;
                    if (upW == null)
                    {
                        // A fused projection carries at most one update, sized for both halves; SwiGLU reads [gate | up] as it lies.
                        int gu = Linear(gateW, x, rows, l.Gate);
                        activated = Unary(GraphOp.Swiglu, gu);
                    }
                    else
                    {
                        int[] gu = hasLora ? Shrink(new[] { l.Gate, l.Up }, x, rows) : new[] { -1, -1 };
                        int gate = Linear(gateW, x, rows, l.Gate, gu[0]);
                        int up = Linear(upW, x, rows, l.Up, gu[1]);
                        activated = b.Op(GraphOp.SwigluSplit, gate, up);
                    }
                    return Linear(blk.Down, activated, rows, l.Down);
                }
                if (managed && seq > ManagedMlpRows)
                {
                    // Row chunks: each row's MLP is its own, and the chunk's [rows, 2 ff] product is freed before the next.
                    int mlp = -1;
                    for (int c0 = 0; c0 < seq; c0 += ManagedMlpRows)
                    {
                        int cn = Math.Min(ManagedMlpRows, seq - c0);
                        int part = Mlp(Slice(h, dim, c0, cn), cn);
                        mlp = mlp < 0 ? part : Concat(mlp, part, 1);
                    }
                    h = mlp;
                }
                else h = Mlp(h, seq);
                if (_ranks > 1) h = b.Op(GraphOp.AllReduce, h);
                joint = b.Op(GraphOp.Add, joint, Modulate(h, modulation[3], seq, gate: true));
            }

            // ---- the output head
            int targetRows = Slice(joint, dim, prefix, seq - prefix);
            int scale = ScaleBias(Linear(w.NormOut, View2(time, dim, 1, dim * 4, 0), 1, Global("norm_out.linear")), 1f, 1f);
            int normalized = b.Op(GraphOp.Mul, Norm(targetRows), scale);
            int output;
            if (lora != null && lora.OutputHeads.Length > 0)
            {
                // A per-step head replaces proj_out; its values are uploaded with the inputs.
                int head = e.InputHost(GraphFormat.F32, stackalloc long[] { dim, w.Channels }, lora.OutputHeads[s.Step],
                    (long)dim * w.Channels * sizeof(float));
                output = Product(head, normalized);
            }
            else
            {
                output = Linear(w.ProjOut, normalized, seq - prefix, Global("proj_out"));
            }
            e.Output(output, (long)(seq - prefix) * w.Channels);
        }

        /// <summary>The state leaf of layer <paramref name="layer"/>'s stored keys or values [D, prefix, heads].</summary>
        private int CacheLeaf(QwenImage21Emitter e, PrefixCache cache, int layer, bool key, int rank)
        {
            int heads = _w.Heads / _ranks;
            ReadOnlySpan<long> ne = stackalloc long[] { _w.HeadDim, cache.Prefix, heads };
            long bytes = (long)_w.HeadDim * cache.Prefix * heads * (cache.Format == GraphFormat.F16 ? 2 : 4);
            return cache.Keys != null
                ? e.State(cache.Format, ne, key ? cache.Keys[rank][layer] : cache.Values[rank][layer], bytes, null)
                : e.State(cache.Format, ne, IntPtr.Zero, bytes, key ? cache.KeyTensors[layer] : cache.ValueTensors[layer]);
        }

        /// <summary>A cached step's keys or values of the whole sequence: the stored prefix, then this step's target rows, in the
        /// layout the attention reads.</summary>
        private int Joined(QwenImage21Emitter e, GraphProgramBuilder b, PrefixCache cache, int layer, bool key, int current, int format,
            int rank)
        {
            int stored = CacheLeaf(e, cache, layer, key, rank);
            int tail = b.Op(GraphOp.Permute, stackalloc[] { current }, stackalloc long[] { 0, 2, 1, 3 });
            tail = format == GraphFormat.F32 ? b.Op(GraphOp.Cont, tail) : b.Op(GraphOp.Cast, stackalloc[] { tail }, stackalloc long[] { format });
            return b.Op(GraphOp.Concat, stackalloc[] { stored, tail }, stackalloc long[] { 1 });
        }

        // A segment's mask extents: on GGML keys padded to 256 and queries to 64 (the retired graph's flash-attention masks);
        // on the managed executor exactly the segment's keys and queries, which its attention reads as a causal pattern.
        private long MaskKeys(QwenImage21Segment g) => _style == Style.Managed ? g.End : (g.End + 255L) / 256 * 256;
        private long MaskQueries(QwenImage21Segment g) => _style == Style.Managed ? g.Length : (g.Length + 63L) / 64 * 64;

        /// <summary>
        /// Each attended segment's F16 mask, or null where it needs none: a text run's causal mask; an image run's (keys up to
        /// its end) only on Vulkan, whose flash attention takes its keys padded. CPU, Metal, CUDA and the managed executor
        /// attend an image run's exact keys unmasked.
        /// </summary>
        private ushort[][] MasksFor(QwenImage21Layout layout, bool cached, IReadOnlyList<QwenImage21Segment> segments)
        {
            ushort[][] Make()
            {
                var result = new ushort[segments.Count][];
                for (int i = 0; i < segments.Count; i++)
                {
                    QwenImage21Segment g = segments[i];
                    if (g.IsImage && _style != Style.Vulkan) continue;
                    long nk = MaskKeys(g), nq = MaskQueries(g);
                    var data = new ushort[checked(nk * nq)];
                    Array.Fill(data, (ushort)0xFC00);   // -inf
                    for (int q = 0; q < g.Length; ++q)
                    {
                        int keys = g.IsImage ? g.End : g.Start + q + 1;
                        Array.Clear(data, (int)(q * nk), keys);
                    }
                    result[i] = data;
                }
                return result;
            }
            // A cached step attends its target segment alone.
            return (cached ? _cachedMasks : _masks).GetValue(layout, _ => Make());
        }

        /// <summary>
        /// The attention of every segment's queries over the keys before its end (the retired graph's Builder.attention): flash
        /// attention at F32 accumulation, each backend's K/V form - Metal F16 copies of each segment's keys, CUDA one F16
        /// copy of the whole sequence's keys that every segment views, Vulkan keys padded to their mask, CPU contiguous F32
        /// copies - and the segments' results concatenated along the rows.
        /// </summary>
        private int Attention(GraphProgramBuilder b, int qp, int kp, int vp, long kNb1, long kNb2, int kType, int qBase,
            IReadOnlyList<QwenImage21Segment> segments, int[] masks, int hd, int heads, int seq, int totalRows)
        {
            bool sharedHalf = _style == Style.Cuda;
            bool converted = kType != GraphFormat.F32;
            long qNb1 = (long)hd * heads * 4, qNb2 = hd * 4L;   // the permuted q [D, rows, heads]
            int kFlash = kp, vFlash = vp;
            long fNb1 = kNb1, fNb2 = kNb2;
            if (sharedHalf && !converted)
            {
                kFlash = b.Op(GraphOp.Cast, stackalloc[] { kp }, stackalloc long[] { GraphFormat.F16 });
                vFlash = b.Op(GraphOp.Cast, stackalloc[] { vp }, stackalloc long[] { GraphFormat.F16 });
                // The cast is contiguous [D, keys, heads] F16.
                long keys = qBase == 0 ? seq : totalRows;
                fNb1 = hd * 2L;
                fNb2 = hd * keys * 2L;
            }
            int width = hd * heads;
            float scale = 1f / MathF.Sqrt(hd);
            int joined = -1;
            for (int i = 0; i < segments.Count; i++)
            {
                QwenImage21Segment g = segments[i];
                long nq = g.Length, nk = g.End;
                int qs = b.Op(GraphOp.View, stackalloc[] { qp },
                    stackalloc long[] { 3, hd, nq, heads, 1, qNb1, qNb2, heads * qNb2, (g.Start - qBase) * qNb1 });
                int ks = b.Op(GraphOp.View, stackalloc[] { kp }, stackalloc long[] { 3, hd, nk, heads, 1, kNb1, kNb2, heads * kNb2, 0 });
                int vs = b.Op(GraphOp.View, stackalloc[] { vp }, stackalloc long[] { 3, hd, nk, heads, 1, kNb1, kNb2, heads * kNb2, 0 });
                int mask = masks[i];
                long pad = !sharedHalf && mask >= 0 ? MaskKeys(g) - nk : 0;
                int kPart = sharedHalf ? b.Op(GraphOp.View, stackalloc[] { kFlash }, stackalloc long[] { 3, hd, nk, heads, 1, fNb1, fNb2, heads * fNb2, 0 }) : ks;
                int vPart = sharedHalf ? b.Op(GraphOp.View, stackalloc[] { vFlash }, stackalloc long[] { 3, hd, nk, heads, 1, fNb1, fNb2, heads * fNb2, 0 }) : vs;
                int kPad = pad > 0 ? b.Op(GraphOp.Pad, stackalloc[] { kPart }, stackalloc long[] { 0, pad, 0, 0 }) : kPart;
                int vPad = pad > 0 ? b.Op(GraphOp.Pad, stackalloc[] { vPart }, stackalloc long[] { 0, pad, 0, 0 }) : vPart;
                if (_style == Style.Metal)
                {
                    // CAST accepts the strided segment and produces contiguous F16.
                    if (kType == GraphFormat.F32)
                    {
                        kPad = b.Op(GraphOp.Cast, stackalloc[] { kPad }, stackalloc long[] { GraphFormat.F16 });
                        vPad = b.Op(GraphOp.Cast, stackalloc[] { vPad }, stackalloc long[] { GraphFormat.F16 });
                    }
                }
                else if (!sharedHalf && pad == 0 && !(qBase != 0 && nk == totalRows))
                {
                    // A segment's keys before the sequence's end are a strided view of the permuted projection.
                    kPad = b.Op(GraphOp.Cont, kPad);
                    vPad = b.Op(GraphOp.Cont, vPad);
                }
                // The retired graph built this flash attention and kept it only where the backend runs it (ggml-cuda takes
                // head sizes 40 to 256 and 576, for one): elsewhere its explicit attention over the F32 views.
                int kvFormat = sharedHalf || (_style == Style.Metal && kType == GraphFormat.F32) ? GraphFormat.F16 : kType;
                int @out;
                if (FlashSupported(hd, nq, nk + pad, heads, kvFormat, mask >= 0 ? MaskKeys(g) : 0, mask >= 0 ? MaskQueries(g) : 0))
                {
                    ReadOnlySpan<int> srcs = mask >= 0 ? stackalloc[] { qs, kPad, vPad, mask } : stackalloc[] { qs, kPad, vPad };
                    int fa = b.Op(GraphOp.FlashAttnExt, srcs, stackalloc long[] { PrecisionF32 }, stackalloc[] { scale, 0f, 0f });
                    @out = b.Op(GraphOp.Reshape, stackalloc[] { fa }, stackalloc long[] { width, nq });
                }
                else @out = ExplicitAttention(b, qs, ks, vs, g.IsImage ? -1 : mask, MaskKeys(g), nq, nk, scale, width);
                joined = joined < 0 ? @out : b.Op(GraphOp.Concat, stackalloc[] { joined, @out }, stackalloc long[] { 1 });
            }
            return joined;
        }

        /// <summary>The retired graph's attention where the backend has no flash attention for the segment (its Builder.attention
        /// fallback): scores at F32 accumulation over the segment's F32 keys, a text run's causal mask cut from its F16 mask,
        /// softmax, and the values' product.</summary>
        private static int ExplicitAttention(GraphProgramBuilder b, int qs, int ks, int vs, int mask, long maskKeys, long nq, long nk,
            float scale, long width)
        {
            int scores = b.Op(GraphOp.MulMat, stackalloc[] { ks, qs }, stackalloc long[] { PrecisionF32 });
            int probs;
            if (mask >= 0)
            {
                long nb1 = maskKeys * 2;
                int causal = b.Op(GraphOp.View, stackalloc[] { mask }, stackalloc long[] { 2, nk, nq, 1, 1, nb1, nb1 * nq, nb1 * nq, 0 });
                causal = b.Op(GraphOp.Cast, stackalloc[] { b.Op(GraphOp.Cont, causal) }, stackalloc long[] { GraphFormat.F32 });
                probs = b.Op(GraphOp.SoftMaxExt, stackalloc[] { scores, causal }, ReadOnlySpan<long>.Empty, stackalloc[] { scale, 0f });
            }
            else probs = b.Op(GraphOp.SoftMaxExt, stackalloc[] { scores }, ReadOnlySpan<long>.Empty, stackalloc[] { scale, 0f });
            int vt = b.Op(GraphOp.Cont, b.Op(GraphOp.Permute, stackalloc[] { vs }, stackalloc long[] { 1, 0, 2, 3 }));
            int av = b.Op(GraphOp.MulMat, vt, probs);
            int permuted = b.Op(GraphOp.Cont, b.Op(GraphOp.Permute, stackalloc[] { av }, stackalloc long[] { 0, 2, 1, 3 }));
            return b.Op(GraphOp.Reshape, stackalloc[] { permuted }, stackalloc long[] { width, nq });
        }

        // Whether the GGML backend runs a segment's flash attention (ggml's supports_op on the node), by its operands' shapes.
        private readonly Dictionary<(long, long, long, long, int, long, long), bool> _flashSupport = new();

        private bool FlashSupported(long hd, long nq, long nk, long heads, int format, long maskCols, long maskRows)
        {
            if (_executor is not GgmlGraphExecutor ggml) return true;
            var key = (hd, nq, nk, heads, format, maskCols, maskRows);
            lock (_flashSupport)
            {
                if (!_flashSupport.TryGetValue(key, out bool supported))
                    _flashSupport[key] = supported = ggml.SupportsFlashAttention(hd, nq, nk, heads, format, maskCols, maskRows, (int)PrecisionF32);
                return supported;
            }
        }

        /// <summary>A tensor executor's object for a weight leaf: an F32 vector as a tensor, a projection or factor as itself.</summary>
        private object Handle(QwenImage21Weight x)
        {
            if (_handles.TryGetValue(x.Data, out object known)) return known;
            object h = x;
            if (x.Vector)
            {
                var values = new float[x.Ne0];
                Marshal.Copy(x.Data, values, 0, values.Length);
                var t = new Tensor(_tensorAllocator, DType.Float32, x.Ne0);
                t.SetElementsAsFloat(values);
                h = t;
            }
            _handles[x.Data] = h;
            return h;
        }
    }
}
