// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Graph;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// The layer helpers of <see cref="NetworkBuilder"/>: shape ops, elementwise ops, norms, products, convolutions, rotary
    /// embeddings and attention, each composed from ops every executor runs (GraphOp.DepthwiseConv2d and GraphOp.GeluErf
    /// included). Extents are ggml's (innermost first); activations are rows of features [features, rows].
    /// </summary>
    public sealed partial class NetworkBuilder
    {
        /// <summary>GGML_ROPE_TYPE_NEOX.</summary>
        public const int RopeNeoxMode = 2;

        /// <summary>GGML_ROPE_TYPE_VISION: ggml_rope_multi's 2-D rotary over a whole head (pairs (i, i + d / 2)).</summary>
        public const int RopeVisionMode = 24;

        /// <summary>GGML_PREC_F32: attention accumulates in F32.</summary>
        private const long PrecisionF32 = 10;

        // ================================================================ shapes

        /// <summary><paramref name="x"/> packed (a copy where it is a strided view, else itself).</summary>
        public NetValue Cont(NetValue x) => x.Strided ? Emit(GraphOp.Cont, x, x.Ne0, x.Ne1, x.Ne2, x.Ne3) : x;

        /// <summary>The same values with extents <paramref name="ne"/> (a strided view is packed first).</summary>
        public NetValue Reshape(NetValue x, params long[] ne)
        {
            long n = 1;
            foreach (long d in ne) n *= d;
            if (n != x.Elements) throw new ArgumentException($"reshape of {x} to [{string.Join(", ", ne)}]");
            NetValue packed = Cont(x);
            if (Same(packed, ne)) return packed;
            return Value(_b.Op(GraphOp.Reshape, stackalloc[] { packed.Id }, Extents(ne)), ne, false, x.Format);
        }

        private static bool Same(NetValue x, ReadOnlySpan<long> ne)
        {
            for (int i = 0; i < 4; i++)
                if (x[i] != (i < ne.Length ? ne[i] : 1)) return false;
            return true;
        }

        /// <summary>ggml_permute: axis i of <paramref name="x"/> becomes axis a_i of the result (a strided view).</summary>
        public NetValue Permute(NetValue x, int a0, int a1, int a2, int a3)
        {
            int[] axes = { a0, a1, a2, a3 };
            var seen = new bool[4];
            var ne = new long[4];
            for (int i = 0; i < 4; i++)
            {
                if (axes[i] < 0 || axes[i] > 3 || seen[axes[i]]) throw new ArgumentException($"permute ({a0}, {a1}, {a2}, {a3})");
                seen[axes[i]] = true;
                ne[axes[i]] = x[i];
            }
            if (a0 == 0 && a1 == 1 && a2 == 2 && a3 == 3) return x;
            return Value(_b.Op(GraphOp.Permute, stackalloc[] { x.Id }, stackalloc long[] { a0, a1, a2, a3 }), ne, true, x.Format);
        }

        /// <summary>ggml_transpose: ne0 and ne1 swapped (a strided view).</summary>
        public NetValue Transpose(NetValue x) => Value(_b.Op(GraphOp.Transpose, stackalloc[] { x.Id }), stackalloc long[] { x.Ne1, x.Ne0, x.Ne2, x.Ne3 }, true, x.Format);

        /// <summary>ggml_view over <paramref name="x"/>'s bytes: extents <paramref name="ne"/>, byte strides of axes 1..3 in
        /// <paramref name="nb"/>, the first element <paramref name="offsetBytes"/> in (x is packed first where it is strided).</summary>
        public NetValue View(NetValue x, long[] ne, long[] nb, long offsetBytes)
        {
            if (ne.Length < 1 || ne.Length > 4 || nb.Length != ne.Length - 1) throw new ArgumentException("a view takes 1-4 extents and a stride for each past the first");
            NetValue src = Cont(x);
            int es = src.Format == GraphFormat.F16 ? 2 : 4;
            long packed = ne[0] * es;
            bool strided = false;
            for (int i = 1; i < ne.Length; i++)
            {
                if (ne[i] > 1 && nb[i - 1] != packed) strided = true;
                packed *= ne[i];
            }
            Span<long> ip = stackalloc long[9];
            ip[0] = ne.Length;
            for (int i = 0; i < 4; i++) ip[1 + i] = i < ne.Length ? ne[i] : 1;
            for (int i = 0; i < 3; i++) ip[5 + i] = i < nb.Length ? nb[i] : 0;
            ip[8] = offsetBytes;
            return Value(_b.Op(GraphOp.View, stackalloc[] { src.Id }, ip), ne, strided, src.Format);
        }

        /// <summary>Columns [<paramref name="first"/>, + <paramref name="count"/>) of every row of <paramref name="x"/> (a block
        /// of features: one of the B, C, x thirds of a short conv's in-projection); strided unless it is whole rows.</summary>
        public NetValue Columns(NetValue x, long first, long count)
        {
            if (first < 0 || count < 1 || first + count > x.Ne0) throw new ArgumentOutOfRangeException(nameof(count), $"columns [{first}, {first + count}) of {x}");
            if (first == 0 && count == x.Ne0) return x;
            NetValue src = Cont(x);
            long row = src.Ne0 * 4;
            return src.Ne2 * src.Ne3 == 1
                ? View(src, new[] { count, src.Ne1 }, new[] { row }, first * 4)
                : View(src, new[] { count, src.Ne1, src.Ne2, src.Ne3 }, new[] { row, row * src.Ne1, row * src.Ne1 * src.Ne2 }, first * 4);
        }

        /// <summary>Rows [<paramref name="first"/>, + <paramref name="count"/>) of a 2-D <paramref name="x"/> [features, rows]
        /// (packed: rows lie together).</summary>
        public NetValue Rows(NetValue x, long first, long count)
        {
            if (x.Ne2 * x.Ne3 != 1) throw new ArgumentException($"rows of a 2-D value, not {x}");
            if (first < 0 || count < 1 || first + count > x.Ne1) throw new ArgumentOutOfRangeException(nameof(count), $"rows [{first}, {first + count}) of {x}");
            if (first == 0 && count == x.Ne1) return x;
            NetValue src = Cont(x);
            return View(src, new[] { src.Ne0, count }, new[] { src.Ne0 * 4 }, first * src.Ne0 * 4);
        }

        /// <summary>Elements [<paramref name="first"/>, + <paramref name="count"/>) of <paramref name="x"/> along
        /// <paramref name="axis"/>, the other axes whole (a view: packed where <paramref name="axis"/> is the outermost one with an
        /// extent above 1, e.g. the last keys of [d, kvHeads, keys] a stream carries on).</summary>
        public NetValue Slice(NetValue x, int axis, long first, long count)
        {
            if (axis < 0 || axis > 3) throw new ArgumentOutOfRangeException(nameof(axis));
            if (first < 0 || count < 1 || first + count > x[axis]) throw new ArgumentOutOfRangeException(nameof(count), $"[{first}, {first + count}) of axis {axis} of {x}");
            if (first == 0 && count == x[axis]) return x;
            NetValue src = Cont(x);
            int es = src.Format == GraphFormat.F16 ? 2 : 4;
            var ne = src.Ne;
            var nb = new long[3];
            long stride = ne[0] * es, offset = first * es;
            for (int i = 1; i < 4; i++)
            {
                nb[i - 1] = stride;
                if (i == axis) offset = first * stride;
                stride *= ne[i];
            }
            ne[axis] = count;
            return View(src, ne, nb, offset);
        }

        /// <summary>The last <paramref name="count"/> elements of <paramref name="x"/> along <paramref name="axis"/>.</summary>
        public NetValue Tail(NetValue x, int axis, long count) => Slice(x, axis, x[axis] - count, count);

        /// <summary>ggml_concat of <paramref name="a"/> and <paramref name="b"/> along <paramref name="axis"/> (the other extents
        /// equal).</summary>
        public NetValue Concat(NetValue a, NetValue b, int axis)
        {
            if (axis < 0 || axis > 3) throw new ArgumentOutOfRangeException(nameof(axis));
            var ne = a.Ne;
            for (int i = 0; i < 4; i++)
                if (i != axis && a[i] != b[i]) throw new ArgumentException($"concat of {a} and {b} along {axis}");
            ne[axis] = a[axis] + b[axis];
            return Emit(GraphOp.Concat, a, b, ne[0], ne[1], ne[2], ne[3], stackalloc long[] { axis });
        }

        // ================================================================ elementwise

        public NetValue Add(NetValue a, NetValue b) => Binary(GraphOp.Add, a, b);
        public NetValue Sub(NetValue a, NetValue b) => Binary(GraphOp.Sub, a, b);
        public NetValue Mul(NetValue a, NetValue b) => Binary(GraphOp.Mul, a, b);
        public NetValue Div(NetValue a, NetValue b) => Binary(GraphOp.Div, a, b);

        /// <summary>ggml's binary ops: <paramref name="b"/> repeats over <paramref name="a"/> (each of b's extents 1 or a's).</summary>
        private NetValue Binary(GraphOp op, NetValue a, NetValue b)
        {
            for (int i = 0; i < 4; i++)
                if (b[i] != 1 && b[i] != a[i]) throw new ArgumentException($"{op}: {b} does not repeat over {a}");
            return Emit(op, a, b, a.Ne0, a.Ne1, a.Ne2, a.Ne3);
        }

        /// <summary>The sum of each row (ne0): [1, ne1, ne2, ne3].</summary>
        public NetValue SumRows(NetValue x) => Emit(GraphOp.SumRows, Cont(x), 1, x.Ne1, x.Ne2, x.Ne3);

        /// <summary>x * s.</summary>
        public NetValue Scale(NetValue x, float s) => s == 1f ? x : Emit(GraphOp.Scale, Packed(x), x.Ne0, x.Ne1, x.Ne2, x.Ne3, default, stackalloc[] { s });

        public NetValue Silu(NetValue x) => Unary(GraphOp.Silu, x);
        public NetValue Sigmoid(NetValue x) => Unary(GraphOp.Sigmoid, x);
        public NetValue Tanh(NetValue x) => Unary(GraphOp.Tanh, x);
        public NetValue Relu(NetValue x) => Unary(GraphOp.Relu, x);
        public NetValue Exp(NetValue x) => Unary(GraphOp.Exp, x);

        /// <summary>The exact GELU, 0.5 x (1 + erf(x / sqrt 2)) (torch's nn.GELU(), ggml_gelu_erf).</summary>
        public NetValue GeluErf(NetValue x) => Unary(GraphOp.GeluErf, x);

        /// <summary>silu(gate) * up.</summary>
        public NetValue SwiGlu(NetValue gate, NetValue up)
        {
            if (gate.Elements != up.Elements) throw new ArgumentException($"swiglu of {gate} and {up}");
            return Emit(GraphOp.SwigluSplit, Packed(gate), Packed(up), gate.Ne0, gate.Ne1, gate.Ne2, gate.Ne3);
        }

        private NetValue Unary(GraphOp op, NetValue x) => Emit(op, Packed(x), x.Ne0, x.Ne1, x.Ne2, x.Ne3);

        /// <summary><paramref name="x"/> as the executor's norms and unary kernels read it: a strided view stays one where they
        /// read row-contiguous views (ggml-metal, ggml-cuda), else it is packed first.</summary>
        private NetValue Packed(NetValue x) => x.Strided && !Capabilities.StridedViews ? Cont(x) : x;

        // ================================================================ norms

        /// <summary>rms_norm(x) over each row (ne0), times <paramref name="weight"/> (a [ne0] vector) where given.</summary>
        public NetValue RmsNorm(NetValue x, float eps, NetValue weight = default)
        {
            NetValue n = Emit(GraphOp.RmsNorm, Packed(x), x.Ne0, x.Ne1, x.Ne2, x.Ne3, default, stackalloc[] { eps });
            return weight.Valid ? Mul(n, weight) : n;
        }

        /// <summary>rms_norm(x) * weight <paramref name="weight"/> (by name).</summary>
        public NetValue RmsNorm(NetValue x, string weight, float eps) => RmsNorm(x, eps, Vector(weight, x.Ne0));

        /// <summary>Layer norm over each row: (x - mean) / sqrt(var + eps) * weight + bias (weight and bias optional).</summary>
        public NetValue LayerNorm(NetValue x, float eps, NetValue weight = default, NetValue bias = default)
        {
            NetValue n = Emit(GraphOp.Norm, Packed(x), x.Ne0, x.Ne1, x.Ne2, x.Ne3, default, stackalloc[] { eps });
            if (weight.Valid) n = Mul(n, weight);
            if (bias.Valid) n = Add(n, bias);
            return n;
        }

        /// <summary>Layer norm with weight and bias by name (null: none).</summary>
        public NetValue LayerNorm(NetValue x, string weight, string bias, float eps)
            => LayerNorm(x, eps, weight == null ? default : Vector(weight, x.Ne0), bias == null ? default : Vector(bias, x.Ne0));

        /// <summary>A weight read as a [n] vector (a [1, 1, n] bias of a conv, say).</summary>
        public NetValue Vector(string name, long n) => Weight(name, n);

        // ================================================================ products

        /// <summary>weight · x (ggml_mul_mat: [weight.ne1, x.ne1, x.ne2, x.ne3]); <paramref name="weight"/> is a product
        /// weight [in, out] (any block format) or an F32 value.</summary>
        public NetValue MatMul(NetValue weight, NetValue x)
        {
            if (weight.Ne0 != x.Ne0) throw new ArgumentException($"product of weight {weight} with {x}: widths differ");
            if (weight.Ne2 * weight.Ne3 != 1 && (x.Ne2 % weight.Ne2 != 0 || x.Ne3 % weight.Ne3 != 0))
                throw new ArgumentException($"batched product of {weight} with {x}");
            return Emit(GraphOp.MulMat, weight, Packed(x), weight.Ne1, x.Ne1, x.Ne2, x.Ne3);
        }

        /// <summary>A linear layer: weight · x (+ bias), with a product weight's per-tensor scale applied.</summary>
        public NetValue Linear(NetValue x, string weight, string bias = null)
        {
            NetValue w = Weight(weight);
            NetValue y = Scale(MatMul(w, x), WeightScale(weight));
            return bias == null ? y : Add(y, Vector(bias, w.Ne1));
        }

        /// <summary>A linear layer over an explicit weight value (a re-read weight, a row block) and optional bias value.</summary>
        public NetValue Linear(NetValue x, NetValue weight, NetValue bias = default)
        {
            NetValue y = MatMul(weight, x);
            return bias.Valid ? Add(y, bias) : y;
        }

        /// <summary>A SwiGLU feed-forward: down · (silu(gate · x) * (up · x)).</summary>
        public NetValue FeedForward(NetValue x, string gate, string up, string down)
            => Linear(SwiGlu(Linear(x, gate), Linear(x, up)), down);

        /// <summary>A GELU MLP: fc2 · gelu_erf(fc1 · x + b1) + b2 (biases optional).</summary>
        public NetValue GeluMlp(NetValue x, string fc1, string bias1, string fc2, string bias2)
            => Linear(GeluErf(Linear(x, fc1, bias1)), fc2, bias2);

        /// <summary>Rows of <paramref name="table"/> ([width, rows]: a product weight or an F32 value) at
        /// <paramref name="indices"/> (I32 [n]): [width, n] (an embedding lookup, an FSQ codebook, a nearest upsample).</summary>
        public NetValue GetRows(NetValue table, NetValue indices)
        {
            if (indices.Format != GraphFormat.I32) throw new ArgumentException("row indices are I32");
            return Emit(GraphOp.GetRows, table.Strided ? Cont(table) : table, indices, table.Ne0, indices.Ne0, indices.Ne1, indices.Ne2);
        }

        // ================================================================ convolutions

        /// <summary>
        /// A causal depthwise 1-D conv over time (ggml_ssm_conv, no activation): <paramref name="x"/> [C, T] rows of channels,
        /// <paramref name="kernel"/> [K, C] (ggml_ssm_conv's layout: a GGUF conv_dw weight), the K - 1 columns before the rows
        /// in <paramref name="state"/> [K - 1, C] (time innermost; default: zeros, a stream's start). Returns y [C, T]: y[c, t] =
        /// sum over k of window[t + k, c] * kernel[k, c]. <paramref name="window"/> is the conv's input [K - 1 + T, C]; its last K
        /// - 1 columns (<see cref="ConvTail"/>) are the next call's state.
        /// </summary>
        public NetValue CausalConv1d(NetValue x, NetValue kernel, NetValue state, out NetValue window)
        {
            long k = kernel.Ne0, c = kernel.Ne1, t = x.Ne1;
            if (x.Ne0 != c || x.Ne2 * x.Ne3 != 1) throw new ArgumentException($"causal conv of {x} with kernel {kernel}");
            if (!state.Valid) state = Zeros(k - 1, c);
            if (state.Ne0 != k - 1 || state.Ne1 != c) throw new ArgumentException($"causal conv state {state} for kernel {kernel}");
            window = Concat(state, Transpose(x), 0);
            return Emit(GraphOp.SsmConv, window, kernel, c, t, 1, 1);
        }

        /// <summary>The last <paramref name="columns"/> time columns of a causal conv's <paramref name="window"/> [L, C] (a strided
        /// view [columns, C]): the next call's state, written with <see cref="Write"/>.</summary>
        public NetValue ConvTail(NetValue window, long columns)
            => View(window, new[] { columns, window.Ne1 }, new[] { window.Ne0 * 4 }, (window.Ne0 - columns) * 4);

        /// <summary>
        /// A depthwise 2-D conv of a feature map <paramref name="x"/> [C, W, H] (or [C, W, H, N]) with
        /// <paramref name="kernel"/> [KW, KH, 1, C] (a GGUF depthwise conv2d weight), zero padding on both sides: GraphOp.DepthwiseConv2d.
        /// Returns [C, OW, OH, N].
        /// </summary>
        public NetValue DepthwiseConv2d(NetValue x, NetValue kernel, int strideW = 1, int strideH = 1, int padW = 0, int padH = 0,
            int dilationW = 1, int dilationH = 1)
        {
            long kw = kernel.Ne0, kh = kernel.Ne1, c = x.Ne0;
            if (kernel.Elements != kw * kh * c) throw new ArgumentException($"depthwise kernel {kernel} over {c} channels");
            long ow = (x.Ne1 + 2L * padW - dilationW * (kw - 1) - 1) / strideW + 1;
            long oh = (x.Ne2 + 2L * padH - dilationH * (kh - 1) - 1) / strideH + 1;
            if (ow < 1 || oh < 1) throw new ArgumentException($"a {kw}x{kh} kernel over {x}");
            return Emit(GraphOp.DepthwiseConv2d, Cont(x), kernel, c, ow, oh, x.Ne3,
                stackalloc long[] { strideW, strideH, padW, padH, dilationW, dilationH });
        }

        /// <summary>Depthwise conv with the kernel by name (a [KW, KH, 1, C] GGUF tensor).</summary>
        public NetValue DepthwiseConv2d(NetValue x, string kernel, int stride = 1, int pad = 0)
            => DepthwiseConv2d(x, Weight(kernel), stride, stride, pad, pad);

        /// <summary>
        /// A dense 2-D conv of a feature map <paramref name="x"/> [Cin, W, H] with GGUF weight <paramref name="weight"/>
        /// [KW, KH, Cin, Cout] (torch's [Cout, Cin, KH, KW]) and optional bias (Cout values): im2col by views (each tap a strided
        /// view of the zero-padded input, the taps concatenated) and one product with the weight re-laid once on the host as
        /// [KW * KH * Cin (tap-major), Cout] F32 (weight "name#im2col"). Returns [Cout, OW, OH]. For few input channels
        /// (an audio front end's first conv) or small maps; a depthwise conv takes <see cref="DepthwiseConv2d(NetValue, NetValue,
        /// int, int, int, int, int, int)"/>, a pointwise one <see cref="Linear(NetValue, NetValue, NetValue)"/>.
        /// </summary>
        public NetValue Conv2d(NetValue x, string weight, string bias, int strideW, int strideH, int padW, int padH)
        {
            NetworkWeight w = Weights.Get(weight);
            if (w.Ne.Length != 4 && w.Ne.Length != 3) throw new ArgumentException($"conv2d weight '{weight}' [{string.Join(", ", w.Ne)}]: [KW, KH, Cin, Cout]");
            long kw = w.Ne[0], kh = w.Ne[1], cin = w.Ne[2], cout = w.Ne.Length > 3 ? w.Ne[3] : 1;
            if (x.Ne0 != cin || x.Ne3 != 1) throw new ArgumentException($"conv2d of {x} with weight [{kw}, {kh}, {cin}, {cout}]");
            NetValue padded = PadMap(Cont(x), padW, padH);
            long wp = padded.Ne1, hp = padded.Ne2;
            long ow = (wp - kw) / strideW + 1, oh = (hp - kh) / strideH + 1;
            if (ow < 1 || oh < 1) throw new ArgumentException($"a {kw}x{kh} conv over {x} padded ({padW}, {padH})");
            NetValue columns = default;
            for (long ky = 0; ky < kh; ky++)
            {
                for (long kx = 0; kx < kw; kx++)
                {
                    NetValue tap = View(padded, new[] { cin, ow, oh }, new[] { strideW * cin * 4, strideH * wp * cin * 4 }, (kx * cin + ky * wp * cin) * 4);
                    columns = columns.Valid ? Concat(columns, tap, 0) : Cont(tap);
                }
            }
            NetworkWeight laid = Weights.GetOrAdd(weight + "#im2col", set =>
            {
                float[] src = set.ReadF32(weight);
                long taps = kw * kh;
                var dst = new float[taps * cin * cout];
                for (long co = 0; co < cout; co++)
                    for (long ci = 0; ci < cin; ci++)
                        for (long ky = 0; ky < kh; ky++)
                            for (long kx = 0; kx < kw; kx++)
                                dst[co * taps * cin + (ky * kw + kx) * cin + ci] = src[kx + kw * (ky + kh * (ci + cin * co))];
                return set.AddF32(weight + "#im2col", dst, taps * cin, cout);
            });
            NetValue y = MatMul(Weight(laid.Name), Reshape(columns, kw * kh * cin, ow * oh));
            if (bias != null) y = Add(y, Vector(bias, cout));
            return Reshape(y, cout, ow, oh);
        }

        /// <summary>A feature map [C, W, H] with <paramref name="padW"/> zero columns on each side and <paramref name="padH"/>
        /// zero rows above and below.</summary>
        public NetValue PadMap(NetValue x, int padW, int padH)
        {
            if (padW > 0)
            {
                NetValue side = Zeros(x.Ne0, padW, x.Ne2);
                x = Concat(Concat(side, x, 1), side, 1);
            }
            if (padH > 0)
            {
                NetValue band = Zeros(x.Ne0, x.Ne1, padH);
                x = Concat(Concat(band, x, 2), band, 2);
            }
            return x;
        }

        /// <summary>Nearest upsampling along rows: each row of <paramref name="x"/> [C, T] repeated <paramref name="factor"/>
        /// times in place ([C, T * factor]), as a row gather.</summary>
        public NetValue UpsampleNearest(NetValue x, int factor)
        {
            if (factor < 1 || x.Ne2 * x.Ne3 != 1) throw new ArgumentException($"upsample of {x} by {factor}");
            if (factor == 1) return x;
            var rows = new int[x.Ne1 * factor];
            for (int i = 0; i < rows.Length; i++) rows[i] = i / factor;
            return GetRows(x, InputI32(rows));
        }

        // ================================================================ rotary embeddings

        /// <summary>Positions start .. start + count - 1 (I32 [count]).</summary>
        public NetValue Positions(int start, int count)
        {
            var p = new int[count];
            for (int i = 0; i < count; i++) p[i] = start + i;
            return InputI32(p);
        }

        /// <summary>NeoX rotary over the first <paramref name="dims"/> values of each head of <paramref name="x"/> [d, heads,
        /// rows] at <paramref name="positions"/> (I32 [rows]): pairs (i, i + dims / 2) turned by position * base^(-2i / dims).</summary>
        public NetValue RopeNeox(NetValue x, NetValue positions, int dims, float freqBase, float freqScale = 1f)
        {
            if (positions.Format != GraphFormat.I32 || positions.Ne0 != x.Ne2) throw new ArgumentException($"rotary positions {positions} for {x}");
            if (dims < 2 || dims > x.Ne0 || dims % 2 != 0) throw new ArgumentException($"rotary over {dims} of {x.Ne0} values");
            return Emit(GraphOp.RopeExt, x, positions, x.Ne0, x.Ne1, x.Ne2, x.Ne3, stackalloc long[] { dims, RopeNeoxMode, 0 },
                stackalloc[] { freqBase, freqScale, 0f, 1f, 32f, 1f });
        }

        /// <summary>
        /// A vision tower's 2-D rotary (ggml_rope_multi, VISION mode) over <paramref name="x"/> [d, heads, rows]: pair (i, i + d / 2)
        /// for i &lt; d / 4 turned by the first axis' position * base^(-4i / d), for d / 4 &lt;= i &lt; d / 2 by the second axis'
        /// position * base^(-4 (i - d / 4) / d). <paramref name="table"/> is I32 [4 * rows], axis-major
        /// (<see cref="PositionTable2D"/>).
        /// </summary>
        public NetValue Rope2D(NetValue x, NetValue table, float freqBase)
        {
            long d = x.Ne0;
            if (d % 4 != 0) throw new ArgumentException($"2-D rotary over {d}-wide heads (a multiple of 4)");
            if (table.Format != GraphFormat.I32 || table.Ne0 != 4 * x.Ne2) throw new ArgumentException($"2-D rotary positions {table} for {x}");
            return Emit(GraphOp.RopeMulti, x, table, x.Ne0, x.Ne1, x.Ne2, x.Ne3,
                stackalloc long[] { d / 2, RopeVisionMode, 0, d / 4, d / 4, 0, 0 }, stackalloc[] { freqBase, 1f, 0f, 1f, 32f, 1f });
        }

        /// <summary>The axis-major position table of <see cref="Rope2D"/>: [first..., second..., first..., second...] (the last two
        /// copies are what ggml's position layout carries; VISION mode reads the first two).</summary>
        public NetValue PositionTable2D(int[] first, int[] second)
        {
            if (first.Length != second.Length) throw new ArgumentException("2-D positions: one of each axis a row");
            int n = first.Length;
            var table = new int[4 * n];
            first.CopyTo(table, 0);
            second.CopyTo(table, n);
            first.CopyTo(table, 2 * n);
            second.CopyTo(table, 3 * n);
            return InputI32(table);
        }

        /// <summary>The (row, column) positions of a <paramref name="width"/> x <paramref name="height"/> grid in raster order
        /// (column fastest): rows first, as <see cref="PositionTable2D"/> takes them (a vision tower's (y, x)).</summary>
        public NetValue RasterPositions(int width, int height)
        {
            var y = new int[width * height];
            var xs = new int[width * height];
            for (int i = 0; i < y.Length; i++)
            {
                y[i] = i / width;
                xs[i] = i % width;
            }
            return PositionTable2D(y, xs);
        }

        // ================================================================ attention

        /// <summary>
        /// Attention of <paramref name="q"/> [d, heads, rows] over <paramref name="k"/> [d, kvHeads, keys] and
        /// <paramref name="v"/> [dv, kvHeads, keys] (grouped: query head h reads KV head h / (heads / kvHeads)), softmax(q k^T *
        /// <paramref name="scale"/> + mask) v, accumulated in F32. <paramref name="mask"/> is an F16 [keys, rows] input
        /// (<see cref="Mask"/>; default: none, every key). Returns [dv * heads, rows]. Where
        /// <see cref="NetworkGraph.AttentionHalfKv"/> is set (GGML's executors by default) the keys and values are rounded to
        /// F16 first, as llama.cpp's encoders run them.
        /// </summary>
        public NetValue Attention(NetValue q, NetValue k, NetValue v, float scale, NetValue mask = default)
        {
            long d = q.Ne0, heads = q.Ne1, rows = q.Ne2, kvHeads = k.Ne1, keys = k.Ne2, dv = v.Ne0;
            if (k.Ne0 != d || v.Ne1 != kvHeads || v.Ne2 != keys || kvHeads < 1 || heads % kvHeads != 0 || q.Ne3 * k.Ne3 * v.Ne3 != 1)
                throw new ArgumentException($"attention of q {q} over k {k}, v {v}");
            if (mask.Valid && (mask.Ne0 != keys || mask.Ne1 != rows || mask.Format != GraphFormat.F16))
                throw new ArgumentException($"attention mask {mask}: F16 [keys {keys}, rows {rows}]");
            if (_graph.AttentionByProducts) return AttentionByProducts(q, k, v, scale, mask);
            NetValue qp = Permute(q, 0, 2, 1, 3), kp = Permute(k, 0, 2, 1, 3), vp = Permute(v, 0, 2, 1, 3);
            if (_graph.AttentionHalfKv)
            {
                kp = Cast(kp, GraphFormat.F16);
                vp = Cast(vp, GraphFormat.F16);
            }
            Span<int> srcs = stackalloc[] { qp.Id, kp.Id, vp.Id, mask.Valid ? mask.Id : 0 };
            int fa = _b.Op(GraphOp.FlashAttnExt, mask.Valid ? srcs : srcs[..3], stackalloc long[] { PrecisionF32 }, stackalloc[] { scale, 0f, 0f });
            return Reshape(new NetValue(fa, dv, heads, rows, 1, false, GraphFormat.F32), dv * heads, rows);
        }

        /// <summary>A packed copy of <paramref name="x"/> in <paramref name="format"/> (F32 or F16; executors other than GGML's
        /// take F32 only).</summary>
        public NetValue Cast(NetValue x, int format)
        {
            if (format != GraphFormat.F32 && format != GraphFormat.F16) throw new ArgumentException($"a cast to format {format}");
            return Value(_b.Op(GraphOp.Cast, stackalloc[] { x.Id }, stackalloc long[] { format }), x.Ne, false, format);
        }

        /// <summary>An attention mask from <see cref="NetworkMasks"/> (F16 bits [keys, rows]) as an input.</summary>
        public NetValue Mask(ushort[] bits, long keys, long rows) => InputF16(bits, keys, rows);

        /// <summary>
        /// One grouped-query self-attention block's core over rows <paramref name="x"/> [n, rows] with projections by name:
        /// q, k, v products, each head RMS-normed by its weight where named (<paramref name="qNorm"/>, <paramref name="kNorm"/>),
        /// the rotary <paramref name="rotate"/> applied to q and k (null: none), attention under <paramref name="mask"/>, then the
        /// output product. <paramref name="keyPrefix"/> / <paramref name="valuePrefix"/> ([d, kvHeads, P], post-norm and
        /// post-rotary: a stream's carried keys) go before this call's keys. The keys and values of this call, as attended (with
        /// any prefix), come back in <paramref name="keys"/> and <paramref name="values"/> for a caller that carries them.
        /// </summary>
        public NetValue SelfAttention(NetValue x, int heads, int kvHeads, int headDim, string q, string k, string v, string output,
            string qNorm, string kNorm, float normEps, Func<NetValue, NetValue> rotate, float scale, NetValue mask,
            NetValue keyPrefix, NetValue valuePrefix, out NetValue keys, out NetValue values)
        {
            long rows = x.Ne1;
            NetValue qh = Reshape(Linear(x, q), headDim, heads, rows);
            NetValue kh = Reshape(Linear(x, k), headDim, kvHeads, rows);
            NetValue vh = Reshape(Linear(x, v), headDim, kvHeads, rows);
            if (qNorm != null) qh = RmsNorm(qh, qNorm, normEps);
            if (kNorm != null) kh = RmsNorm(kh, kNorm, normEps);
            if (rotate != null)
            {
                qh = rotate(qh);
                kh = rotate(kh);
            }
            keys = keyPrefix.Valid ? Concat(keyPrefix, kh, 2) : kh;
            values = valuePrefix.Valid ? Concat(valuePrefix, vh, 2) : vh;
            NetValue attended = Attention(qh, keys, values, scale, mask);
            return output == null ? attended : Linear(attended, output);
        }

        /// <summary>A per-head sigmoid gate on attention rows: <paramref name="attention"/> [d * heads, rows] times sigmoid of
        /// <paramref name="gate"/> [heads, rows] (one scalar a head and row: LFM3's attn_gate).</summary>
        public NetValue HeadGate(NetValue attention, NetValue gate, int headDim)
        {
            long heads = gate.Ne0, rows = gate.Ne1;
            if (attention.Ne0 != headDim * heads || attention.Ne1 != rows) throw new ArgumentException($"head gate {gate} over {attention}");
            NetValue g = Reshape(Sigmoid(gate), 1, heads, rows);
            return Reshape(Mul(Reshape(attention, headDim, heads, rows), g), headDim * heads, rows);
        }
    }
}
