// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Graph
{
    /// <summary>A run of rows [<see cref="Start"/>, <see cref="End"/>) of a packed sequence that shares one modulation column
    /// base (<see cref="GraphFusion.AdaLnModulate"/>, <see cref="GraphFusion.GatedResidual"/>): a modality at one timestep.</summary>
    public readonly record struct GraphSegment(int Start, int End, int Column)
    {
        public int Count => End - Start;
    }

    /// <summary>
    /// The diffusion-transformer regions (<see cref="GraphFusion.AdaLnModulate"/>, <see cref="GraphFusion.GatedResidual"/>,
    /// <see cref="GraphFusion.TableRopeQkNorm"/>, <see cref="GraphFusion.LoraLinear"/>, and <see cref="GraphFusion.SwigluFfn"/>
    /// with biases) recorded with their reference ops. The op sequences are the unfused graph MiniMax-H3's native DiT builds
    /// (ggml_ops_minimax_h3.cpp: modulate, gatedResidual, h3_vit_rope, h3_attend's flash path, the unmerged LoRA terms), so a
    /// backend that does not serve a region computes what that graph computes there, node for node.
    /// <para>Activations are F32 and contiguous unless a parameter gives strides (q/k/v views of a projection); a byte offset
    /// is <c>4 * elements</c>. Every method opens and closes its region; the returned nodes are the region's results.</para>
    /// </summary>
    public static class GraphDitRegions
    {
        private const long F = sizeof(float);

        /// <summary>ggml's accumulation precision GGML_PREC_F32 (a product's or an attention's ip[0]).</summary>
        public const long PrecisionF32 = 10;

        /// <summary>The values of a segment table input ([3, segments] I32: start, end, column base a segment).</summary>
        public static int[] SegmentTable(IReadOnlyList<GraphSegment> segments)
        {
            ArgumentNullException.ThrowIfNull(segments);
            var table = new int[3 * segments.Count];
            for (int i = 0; i < segments.Count; i++)
            {
                table[3 * i] = segments[i].Start;
                table[3 * i + 1] = segments[i].End;
                table[3 * i + 2] = segments[i].Column;
            }
            return table;
        }

        /// <summary>Throws unless <paramref name="segments"/> are sorted, contiguous and cover [0, <paramref name="rows"/>), and
        /// each column base plus <paramref name="maxOffset"/> lies inside <paramref name="columns"/> - the table the regions
        /// require (their ops concatenate the segments in order).</summary>
        public static void CheckSegments(IReadOnlyList<GraphSegment> segments, long rows, long columns, int maxOffset)
        {
            ArgumentNullException.ThrowIfNull(segments);
            if (segments.Count == 0) throw new ArgumentException("a packed sequence has at least one segment", nameof(segments));
            long next = 0;
            for (int i = 0; i < segments.Count; i++)
            {
                GraphSegment s = segments[i];
                if (s.Start != next || s.End < s.Start)
                    throw new ArgumentException($"segment {i} [{s.Start}, {s.End}) does not continue at row {next}", nameof(segments));
                if (s.Column < 0 || s.Column + (long)maxOffset >= columns)
                    throw new ArgumentException($"segment {i}'s column base {s.Column} + {maxOffset} is outside {columns} columns", nameof(segments));
                next = s.End;
            }
            if (next != rows) throw new ArgumentException($"the segments cover {next} rows of {rows}", nameof(segments));
        }

        private static int View2(GraphProgramBuilder g, int x, long ne0, long ne1, long nb1, long offset)
            => g.Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 2, ne0, ne1, 1, 1, nb1, 0, 0, offset });

        private static int View3(GraphProgramBuilder g, int x, long ne0, long ne1, long ne2, long nb1, long nb2, long offset)
            => g.Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 3, ne0, ne1, ne2, 1, nb1, nb2, 0, offset });

        private static int Concat(GraphProgramBuilder g, int a, int b, int dim)
            => g.Op(GraphOp.Concat, stackalloc[] { a, b }, stackalloc long[] { dim });

        private static int RmsNorm(GraphProgramBuilder g, int x, float eps)
            => g.Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps });

        private static int Scale(GraphProgramBuilder g, int x, float s)
            => g.Op(GraphOp.Scale, stackalloc[] { x }, default, stackalloc[] { s });

        /// <summary>A weight product as the region's ops take it: plain w · x unless the caller supplies its own (a family's
        /// overflow guard around a quantized product, say).</summary>
        public delegate int Product(GraphProgramBuilder g, int weight, int x);

        private static int MulMat(GraphProgramBuilder g, int weight, int x) => g.Op(GraphOp.MulMat, weight, x);

        /// <summary>
        /// <see cref="GraphFusion.AdaLnModulate"/> over <paramref name="x"/> [hidden, rows]: RMSNorm(x) * weight once, then
        /// for each segment n + n * mods[col + scale] + mods[col + shift] and the segments concatenated in order.
        /// <paramref name="mods"/> is [hidden, columns] contiguous; <paramref name="segmentTable"/> the I32 [3, segments]
        /// node holding <paramref name="segments"/> (<see cref="SegmentTable"/>), which the ops read through views.
        /// </summary>
        public static int AdaLnModulate(GraphProgramBuilder g, int x, int weight, int mods, int segmentTable,
            IReadOnlyList<GraphSegment> segments, long hidden, int shift, int scale, float eps)
        {
            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(segments);
            if (segments.Count == 0) throw new ArgumentException("a packed sequence has at least one segment", nameof(segments));
            g.BeginFused(GraphFusion.AdaLnModulate, stackalloc[] { x, weight, mods, segmentTable },
                stackalloc long[] { shift, scale }, stackalloc[] { eps });
            int normed = g.Op(GraphOp.Mul, RmsNorm(g, x, eps), weight);
            int y = -1;
            long row = hidden * F;
            foreach (GraphSegment s in segments)
            {
                if (s.Count <= 0) continue;
                int part = View2(g, normed, hidden, s.Count, row, s.Start * row);
                int sh = View2(g, mods, hidden, 1, row, (s.Column + shift) * row);
                int sc = View2(g, mods, hidden, 1, row, (s.Column + scale) * row);
                part = g.Op(GraphOp.Add, g.Op(GraphOp.Add, part, g.Op(GraphOp.Mul, part, sc)), sh);
                y = y < 0 ? part : Concat(g, y, part, 1);
            }
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>
        /// <see cref="GraphFusion.GatedResidual"/>: base + (delta + (mm + bias)) * mods[col + gate] per segment.
        /// <paramref name="bias"/> [hidden] and <paramref name="delta"/> [hidden, rows] are optional (-1: none).
        /// </summary>
        public static int GatedResidual(GraphProgramBuilder g, int baseNode, int mm, int bias, int delta, int mods, int segmentTable,
            IReadOnlyList<GraphSegment> segments, long hidden, int gate)
        {
            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(segments);
            if (segments.Count == 0) throw new ArgumentException("a packed sequence has at least one segment", nameof(segments));
            Span<int> inputs = stackalloc int[6];
            int n = 0;
            inputs[n++] = baseNode;
            inputs[n++] = mm;
            if (bias >= 0) inputs[n++] = bias;
            if (delta >= 0) inputs[n++] = delta;
            inputs[n++] = mods;
            inputs[n++] = segmentTable;
            g.BeginFused(GraphFusion.GatedResidual, inputs[..n], stackalloc long[] { gate, bias >= 0 ? 1 : 0, delta >= 0 ? 1 : 0 });
            int update = mm;
            if (bias >= 0) update = g.Op(GraphOp.Add, update, bias);
            if (delta >= 0) update = g.Op(GraphOp.Add, delta, update);
            int y = -1;
            long row = hidden * F;
            foreach (GraphSegment s in segments)
            {
                if (s.Count <= 0) continue;
                int b = View2(g, baseNode, hidden, s.Count, row, s.Start * row);
                int u = View2(g, update, hidden, s.Count, row, s.Start * row);
                int gv = View2(g, mods, hidden, 1, row, (s.Column + gate) * row);
                int part = g.Op(GraphOp.Add, b, g.Op(GraphOp.Mul, u, gv));
                y = y < 0 ? part : Concat(g, y, part, 1);
            }
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>The signed sine table <see cref="GraphFusion.TableRopeQkNorm"/> reads, from a sine table [rot, rows]:
        /// [-sin(first half), sin(second half)] (negation is exact, so a host-made one holds the same values).</summary>
        public static int SignedSine(GraphProgramBuilder g, int sin, long rot, long rows)
        {
            ArgumentNullException.ThrowIfNull(g);
            if (rot <= 0 || rot % 2 != 0) throw new ArgumentOutOfRangeException(nameof(rot), "the rotated width is even and positive");
            long half = rot / 2;
            int first = View2(g, sin, half, rows, rot * F, 0);
            int second = View2(g, sin, half, rows, rot * F, half * F);
            return Concat(g, g.Op(GraphOp.Neg, g.Op(GraphOp.Cont, first)), second, 0);
        }

        /// <summary>The geometry of one of attention's operands [d, heads, rows]: byte strides between heads and between rows
        /// (a view of a projection: q, k and v of a stacked [q | k | v] product have a head stride of d floats and a row
        /// stride of the product's row).</summary>
        public readonly record struct HeadLayout(long HeadBytes, long RowBytes)
        {
            /// <summary>A contiguous [d, heads, rows] tensor.</summary>
            public static HeadLayout Contiguous(long d, long heads) => new(d * F, d * heads * F);
        }

        /// <summary>
        /// <see cref="GraphFusion.TableRopeQkNorm"/>: q and k [d, heads, rows] each RMS-normed per head (times its weight,
        /// -1: none) and rotated over their first <paramref name="rot"/> dims by the tables [rot, rows] (cos and the signed
        /// sine, <see cref="SignedSine"/>). Returns the rotated q and k, contiguous [d, heads, rows].
        /// </summary>
        public static (int Q, int K) TableRopeQkNorm(GraphProgramBuilder g, int q, int k, int qNorm, int kNorm, int cos, int sinSigned,
            long d, long heads, long rows, int rot, float eps)
        {
            BeginRope(g, q, k, -1, qNorm, kNorm, cos, sinSigned, d, rot, eps, attend: false, 0, 0);
            var (qr, kr) = RopeOps(g, q, k, qNorm, kNorm, cos, sinSigned, d, heads, rows, rot, eps);
            g.EndFused(stackalloc[] { qr, kr });
            return (qr, kr);
        }

        /// <summary>
        /// <see cref="GraphFusion.TableRopeQkNorm"/> with the attention that reads the rotated q and k: softmax(scale q k^T) v
        /// per head, no mask, as the H3 graph's flash path computes it: q, k and v permuted to [d, rows, heads] and copied, K
        /// and V cast to F16, V divided by <paramref name="vScale"/> first (a power of two: the F16 accumulator's headroom) and
        /// the result [d * heads, rows] multiplied by it after. <paramref name="v"/> is [d, heads, rows] in any head and row
        /// strides.
        /// </summary>
        public static int TableRopeQkNormAttention(GraphProgramBuilder g, int q, int k, int v, int qNorm, int kNorm, int cos,
            int sinSigned, long d, long heads, long rows, int rot, float eps, float scale, float vScale)
        {
            CheckVScale(vScale);
            BeginRope(g, q, k, v, qNorm, kNorm, cos, sinSigned, d, rot, eps, attend: true, scale, vScale);
            var (qr, kr) = RopeOps(g, q, k, qNorm, kNorm, cos, sinSigned, d, heads, rows, rot, eps);
            int y = Attention(g, qr, kr, v, d, heads, rows, scale, vScale);
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>
        /// <see cref="GraphFusion.DitAttention"/>: softmax(scale q k^T) v per head over q, k and v [d, heads, rows] (F32, any
        /// head and row strides), no mask, as the H3 graph's flash path computes it: each permuted to [d, rows, heads] and
        /// copied, K and V cast to F16, V divided by <paramref name="vScale"/> first (a power of two: the F16 accumulator's
        /// headroom) and the result [d * heads, rows] multiplied by it after.
        /// </summary>
        public static int Attention(GraphProgramBuilder g, int q, int k, int v, long d, long heads, long rows, float scale, float vScale)
        {
            ArgumentNullException.ThrowIfNull(g);
            CheckVScale(vScale);
            g.BeginFused(GraphFusion.DitAttention, stackalloc[] { q, k, v }, default, stackalloc[] { scale, vScale });
            Span<long> permute = stackalloc long[] { 0, 2, 1, 3 };
            int qp = g.Op(GraphOp.Cont, g.Op(GraphOp.Permute, stackalloc[] { q }, permute));
            int kp = g.Op(GraphOp.Cont, g.Op(GraphOp.Permute, stackalloc[] { k }, permute));
            int vp = g.Op(GraphOp.Cont, g.Op(GraphOp.Permute, stackalloc[] { v }, permute));
            int kf = g.Op(GraphOp.Cast, stackalloc[] { kp }, stackalloc long[] { GraphFormat.F16 });
            int vf = g.Op(GraphOp.Cast, stackalloc[] { vScale == 1f ? vp : Scale(g, vp, 1f / vScale) }, stackalloc long[] { GraphFormat.F16 });
            int attention = g.Op(GraphOp.FlashAttnExt, stackalloc[] { qp, kf, vf }, stackalloc long[] { PrecisionF32 },
                stackalloc[] { scale, 0f, 0f });
            int merged = g.Op(GraphOp.Reshape, stackalloc[] { g.Op(GraphOp.Cont, attention) }, stackalloc long[] { d * heads, rows });
            int y = vScale == 1f ? merged : Scale(g, merged, vScale);
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>The power-of-two V scale an F16 attention accumulator keeps its headroom with, for <paramref name="keys"/>
        /// keys (MiniMax-H3's h3_flash_v_scale): the smallest power of two that brings the keys down to 256, 1 for 256 or
        /// fewer.</summary>
        public static float FlashVScale(long keys)
        {
            float s = 1f;
            while (keys > 256 * s) s *= 2f;
            return s;
        }

        private static void CheckVScale(float vScale)
        {
            if (!(vScale > 0f) || MathF.Log2(vScale) != MathF.Round(MathF.Log2(vScale)))
                throw new ArgumentOutOfRangeException(nameof(vScale), "V's scale is a power of two (exact through the F16 cast)");
        }

        private static void BeginRope(GraphProgramBuilder g, int q, int k, int v, int qNorm, int kNorm, int cos, int sinSigned, long d,
            int rot, float eps, bool attend, float scale, float vScale)
        {
            ArgumentNullException.ThrowIfNull(g);
            if (rot <= 0 || rot % 2 != 0 || rot > d) throw new ArgumentOutOfRangeException(nameof(rot), "the rotated width is even, positive and at most d");
            Span<int> inputs = stackalloc int[7];
            int n = 0;
            inputs[n++] = q;
            inputs[n++] = k;
            if (qNorm >= 0) inputs[n++] = qNorm;
            if (kNorm >= 0) inputs[n++] = kNorm;
            inputs[n++] = cos;
            inputs[n++] = sinSigned;
            if (attend) inputs[n++] = v;
            long norms = (qNorm >= 0 ? 1 : 0) | (kNorm >= 0 ? 2 : 0);
            if (attend) g.BeginFused(GraphFusion.TableRopeQkNorm, inputs[..n], stackalloc long[] { rot, norms, 1 }, stackalloc[] { eps, scale, vScale });
            else g.BeginFused(GraphFusion.TableRopeQkNorm, inputs[..n], stackalloc long[] { rot, norms, 0 }, stackalloc[] { eps });
        }

        private static (int Q, int K) RopeOps(GraphProgramBuilder g, int q, int k, int qNorm, int kNorm, int cos, int sinSigned,
            long d, long heads, long rows, int rot, float eps)
        {
            int qn = RmsNorm(g, q, eps);
            if (qNorm >= 0) qn = g.Op(GraphOp.Mul, qn, qNorm);
            int kn = RmsNorm(g, k, eps);
            if (kNorm >= 0) kn = g.Op(GraphOp.Mul, kn, kNorm);
            return (Rope(g, qn, cos, sinSigned, d, heads, rows, rot), Rope(g, kn, cos, sinSigned, d, heads, rows, rot));
        }

        /// <summary>h3_vit_rope over a contiguous x [d, heads, rows]: x * cos + [x2, x1] * sinSigned on the first rot dims (the
        /// half-swapped head read straight out of x), the rest concatenated after.</summary>
        private static int Rope(GraphProgramBuilder g, int x, int cos, int sinSigned, long d, long heads, long rows, int rot)
        {
            long nb1 = d * F, nb2 = d * heads * F, half = rot / 2;
            int xr = View3(g, x, rot, heads, rows, nb1, nb2, 0);
            int swapped = Concat(g, View3(g, x, half, heads, rows, nb1, nb2, half * F), View3(g, x, half, heads, rows, nb1, nb2, 0), 0);
            int c3 = g.Op(GraphOp.Reshape, stackalloc[] { cos }, stackalloc long[] { rot, 1, rows });
            int s3 = g.Op(GraphOp.Reshape, stackalloc[] { sinSigned }, stackalloc long[] { rot, 1, rows });
            int rotated = g.Op(GraphOp.Add, g.Op(GraphOp.Mul, xr, c3), g.Op(GraphOp.Mul, swapped, s3));
            if (rot == d) return rotated;
            return Concat(g, rotated, View3(g, x, d - rot, heads, rows, nb1, nb2, rot * F), 0);
        }

        /// <summary>
        /// <see cref="GraphFusion.VitAttention"/> over <paramref name="x"/> [dim, sequences * seq] (dim = heads * d): the
        /// interleaved qkv product of <paramref name="weight"/> (plus <paramref name="bias"/>, -1: none), then for each
        /// sequence its q, k and v views, q and k RMS-normed without a gain and rotated by the shared tables [rot, seq], the
        /// attention (<see cref="TableRopeQkNormAttention"/>), and the sequences concatenated, as MiniMax-H3's ViT decoder
        /// builds them (h3_vit_decode without its fused nodes). V's scale is <see cref="FlashVScale"/> of <paramref name="seq"/>.
        /// </summary>
        public static int VitAttention(GraphProgramBuilder g, int x, int weight, int bias, int cos, int sinSigned, long d, long heads,
            long seq, int sequences, int rot, float eps, float scale)
        {
            ArgumentNullException.ThrowIfNull(g);
            if (sequences < 1) throw new ArgumentOutOfRangeException(nameof(sequences), "at least one sequence");
            if (seq < 1) throw new ArgumentOutOfRangeException(nameof(seq), "a sequence has at least one row");
            float vScale = FlashVScale(seq);
            Span<int> inputs = stackalloc int[5];
            int n = 0;
            inputs[n++] = x;
            inputs[n++] = weight;
            if (bias >= 0) inputs[n++] = bias;
            inputs[n++] = cos;
            inputs[n++] = sinSigned;
            g.BeginFused(GraphFusion.VitAttention, inputs[..n], stackalloc long[] { heads, rot, sequences, bias >= 0 ? 1 : 0 },
                stackalloc[] { eps, scale, vScale });
            int qkv = MulMat(g, weight, x);
            if (bias >= 0) qkv = g.Op(GraphOp.Add, qkv, bias);
            // Per head interleaved: head h holds [q | k | v] at [h * 3d, (h + 1) * 3d) of a row; a sequence starts seq rows on.
            long headStride = 3 * d * F, row = 3 * d * heads * F;
            int merged = -1;
            for (int s = 0; s < sequences; s++)
            {
                long at = s * seq * row;
                int q = View3(g, qkv, d, heads, seq, headStride, row, at);
                int k = View3(g, qkv, d, heads, seq, headStride, row, at + d * F);
                int v = View3(g, qkv, d, heads, seq, headStride, row, at + 2 * d * F);
                int one = TableRopeQkNormAttention(g, q, k, v, -1, -1, cos, sinSigned, d, heads, seq, rot, eps, scale, vScale);
                merged = merged < 0 ? one : Concat(g, merged, one, 1);
            }
            g.EndFused(stackalloc[] { merged });
            return merged;
        }

        /// <summary>
        /// <see cref="GraphFusion.ScaledResidual"/>'s projection form: base + (w · x + bias) * scale (bias and scale -1:
        /// none), the add, scale and residual in the order MiniMax-H3's ViT decoder builds them.
        /// </summary>
        public static int ScaledResidual(GraphProgramBuilder g, int baseNode, int x, int weight, int bias, int scale)
        {
            ArgumentNullException.ThrowIfNull(g);
            Span<int> inputs = stackalloc int[5];
            int n = 0;
            inputs[n++] = baseNode;
            inputs[n++] = x;
            inputs[n++] = weight;
            if (bias >= 0) inputs[n++] = bias;
            if (scale >= 0) inputs[n++] = scale;
            g.BeginFused(GraphFusion.ScaledResidual, inputs[..n], stackalloc long[] { 0, (bias >= 0 ? 1 : 0) | (scale >= 0 ? 2 : 0) });
            int y = MulMat(g, weight, x);
            if (bias >= 0) y = g.Op(GraphOp.Add, y, bias);
            int result = g.Op(GraphOp.Add, baseNode, scale >= 0 ? g.Op(GraphOp.Mul, y, scale) : y);
            g.EndFused(stackalloc[] { result });
            return result;
        }

        /// <summary>
        /// <see cref="GraphFusion.ScaledResidual"/>'s SwiGLU form: base + (w2 · swiglu(w1 · x + b1) + b2) * scale, w1 the
        /// gate|up weight (gate rows first). The biases come together or not at all (-1); scale -1: none.
        /// </summary>
        public static int ScaledSwigluResidual(GraphProgramBuilder g, int baseNode, int x, int gateUp, int gateUpBias, int down,
            int downBias, int scale)
        {
            ArgumentNullException.ThrowIfNull(g);
            if ((gateUpBias >= 0) != (downBias >= 0))
                throw new ArgumentException("the SwiGLU form takes both biases or neither");
            bool biased = gateUpBias >= 0;
            Span<int> inputs = stackalloc int[7];
            int n = 0;
            inputs[n++] = baseNode;
            inputs[n++] = x;
            inputs[n++] = gateUp;
            if (biased) inputs[n++] = gateUpBias;
            inputs[n++] = down;
            if (biased) inputs[n++] = downBias;
            if (scale >= 0) inputs[n++] = scale;
            g.BeginFused(GraphFusion.ScaledResidual, inputs[..n], stackalloc long[] { 1, (biased ? 1 : 0) | (scale >= 0 ? 2 : 0) });
            int gv = MulMat(g, gateUp, x);
            if (biased) gv = g.Op(GraphOp.Add, gv, gateUpBias);
            int y = MulMat(g, down, g.Op(GraphOp.Swiglu, gv));
            if (biased) y = g.Op(GraphOp.Add, y, downBias);
            int result = g.Op(GraphOp.Add, baseNode, scale >= 0 ? g.Op(GraphOp.Mul, y, scale) : y);
            g.EndFused(stackalloc[] { result });
            return result;
        }

        /// <summary>One low-rank update of a <see cref="GraphFusion.LoraLinear"/>: its slot (rows [slot * slotRows, + slotRows)
        /// of the output) and its factors down [in, rank] and up [rank, slotRows].</summary>
        public readonly record struct LoraPart(int Slot, int Down, int Up);

        /// <summary>
        /// <see cref="GraphFusion.LoraLinear"/>: w · x [out, rows] with up · (down · x) added to each part's slot (the
        /// unmerged products in F32). <paramref name="slotRows"/> 0 is one slot of all <paramref name="outRows"/>; parts are
        /// in slot order, one a slot. <paramref name="product"/> makes w · x (null: a plain product).
        /// </summary>
        public static int LoraLinear(GraphProgramBuilder g, int x, int weight, IReadOnlyList<LoraPart> parts, long outRows, long slotRows,
            long rows, Product? product = null)
        {
            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(parts);
            long slot = slotRows > 0 ? slotRows : outRows;
            if (outRows % slot != 0 || outRows / slot > 3) throw new ArgumentOutOfRangeException(nameof(slotRows), "one to three equal slots of the output rows");
            long mask = 0;
            int last = -1;
            Span<int> inputs = stackalloc int[2 + 2 * 3];
            int n = 0;
            inputs[n++] = x;
            inputs[n++] = weight;
            foreach (LoraPart p in parts)
            {
                if (p.Slot <= last || p.Slot * slot >= outRows) throw new ArgumentException("one part a slot, in slot order, inside the output", nameof(parts));
                last = p.Slot;
                mask |= 1L << p.Slot;
                inputs[n++] = p.Down;
                inputs[n++] = p.Up;
            }
            g.BeginFused(GraphFusion.LoraLinear, inputs[..n], stackalloc long[] { mask, slotRows });
            int y = (product ?? MulMat)(g, weight, x);
            int Delta(LoraPart p)
            {
                int shrunk = g.Op(GraphOp.MulMat, stackalloc[] { p.Down, x }, stackalloc long[] { PrecisionF32 });
                return g.Op(GraphOp.MulMat, stackalloc[] { p.Up, shrunk }, stackalloc long[] { PrecisionF32 });
            }
            if (slot == outRows)
            {
                if (parts.Count > 0) y = g.Op(GraphOp.Add, Delta(parts[0]), y);
            }
            else if (parts.Count > 0)
            {
                // Each slot of the product, its update added where it has one, concatenated back in order.
                int result = -1, at = 0;
                for (int s = 0; s * slot < outRows; s++)
                {
                    int band = View2(g, y, slot, rows, outRows * F, s * slot * F);
                    if (at < parts.Count && parts[at].Slot == s) band = g.Op(GraphOp.Add, Delta(parts[at++]), band);
                    result = result < 0 ? band : Concat(g, result, band, 0);
                }
                y = result;
            }
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>
        /// <see cref="GraphFusion.SwigluFfn"/> over a gate|up weight (gate rows then up rows) with optional biases (-1: none;
        /// any bias makes it the biased form, Ip[1] = 1): down · swiglu(gate_up · x + b1) + b2, as the H3 graph's mlp
        /// computes it. <paramref name="product"/> makes each weight product (null: a plain product).
        /// </summary>
        public static int SwigluFfn(GraphProgramBuilder g, int x, int gateUp, int down, int gateUpBias, int downBias, Product? product = null)
        {
            ArgumentNullException.ThrowIfNull(g);
            if ((gateUpBias >= 0) != (downBias >= 0))
                throw new ArgumentException("the biased form takes both biases (a zero bias where a projection has none)");
            bool biased = gateUpBias >= 0;
            if (biased) g.BeginFused(GraphFusion.SwigluFfn, stackalloc[] { x, gateUp, down, gateUpBias, downBias }, stackalloc long[] { 0, 1 });
            else g.BeginFused(GraphFusion.SwigluFfn, stackalloc[] { x, gateUp, down });
            Product mm = product ?? MulMat;
            int gv = mm(g, gateUp, x);
            if (biased) gv = g.Op(GraphOp.Add, gv, gateUpBias);
            int y = mm(g, down, g.Op(GraphOp.Swiglu, gv));
            if (biased) y = g.Op(GraphOp.Add, y, downBias);
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>
        /// <see cref="GraphFusion.HeadRotary"/>: x [d * heads, rows] (contiguous), RMS-normed over each whole row times
        /// <paramref name="normWeight"/> [d * heads] first where one is given (-1: none), then rotated half against half by
        /// tables cos and sin [d / 2, heads, rows] that differ per head: [x1 cos - x2 sin, x2 cos + x1 sin], each product its
        /// own op (the two halves copied out first), the halves concatenated back. Returns [d, heads, rows].
        /// </summary>
        public static int HeadRotary(GraphProgramBuilder g, int x, int cos, int sin, long d, long heads, long rows, int normWeight = -1,
            float eps = 0f)
        {
            ArgumentNullException.ThrowIfNull(g);
            if (d <= 0 || d % 2 != 0) throw new ArgumentOutOfRangeException(nameof(d), "a rotated head has an even, positive width");
            if (normWeight >= 0) g.BeginFused(GraphFusion.HeadRotary, stackalloc[] { x, cos, sin, normWeight }, stackalloc long[] { 1 }, stackalloc[] { eps });
            else g.BeginFused(GraphFusion.HeadRotary, stackalloc[] { x, cos, sin }, stackalloc long[] { 0 });
            int normed = normWeight >= 0 ? RowNorm(g, x, normWeight, layer: false, eps) : x;
            long half = d / 2, nb1 = d * F, nb2 = d * heads * F;
            int first = g.Op(GraphOp.Cont, View3(g, normed, half, heads, rows, nb1, nb2, 0));
            int second = g.Op(GraphOp.Cont, View3(g, normed, half, heads, rows, nb1, nb2, half * F));
            int r1 = g.Op(GraphOp.Sub, g.Op(GraphOp.Mul, first, cos), g.Op(GraphOp.Mul, second, sin));
            int r2 = g.Op(GraphOp.Add, g.Op(GraphOp.Mul, second, cos), g.Op(GraphOp.Mul, first, sin));
            int y = Concat(g, r1, r2, 0);
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>
        /// <see cref="GraphFusion.HeadGate"/>: attn [d * heads, rows] times <paramref name="scale"/> * sigmoid(logits) per head,
        /// logits [heads, rows]: the sigmoid, the scale, then a multiply of attn viewed [d, heads, rows] by the gates viewed
        /// [1, heads, rows]. Returns [d * heads, rows].
        /// </summary>
        public static int HeadGate(GraphProgramBuilder g, int attn, int logits, long d, long heads, long rows, float scale)
        {
            ArgumentNullException.ThrowIfNull(g);
            g.BeginFused(GraphFusion.HeadGate, stackalloc[] { attn, logits }, default, stackalloc[] { scale });
            int gates = Scale(g, g.Op(GraphOp.Sigmoid, logits), scale);
            int g3 = g.Op(GraphOp.Reshape, stackalloc[] { gates }, stackalloc long[] { 1, heads, rows });
            int a3 = g.Op(GraphOp.Reshape, stackalloc[] { attn }, stackalloc long[] { d, heads, rows });
            int y = g.Op(GraphOp.Reshape, stackalloc[] { g.Op(GraphOp.Mul, a3, g3) }, stackalloc long[] { d * heads, rows });
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>
        /// <see cref="GraphFusion.RowNorm"/>: the RMS norm (<paramref name="layer"/> false) or the layer norm of x's rows [n, rows],
        /// times <paramref name="weight"/> [n] (-1: none). Returns [n, rows].
        /// </summary>
        public static int RowNorm(GraphProgramBuilder g, int x, int weight, bool layer, float eps)
        {
            ArgumentNullException.ThrowIfNull(g);
            if (weight >= 0) g.BeginFused(GraphFusion.RowNorm, stackalloc[] { x, weight }, stackalloc long[] { layer ? 1 : 0, 1 }, stackalloc[] { eps });
            else g.BeginFused(GraphFusion.RowNorm, stackalloc[] { x }, stackalloc long[] { layer ? 1 : 0, 0 }, stackalloc[] { eps });
            int y = g.Op(layer ? GraphOp.Norm : GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps });
            if (weight >= 0) y = g.Op(GraphOp.Mul, y, weight);
            g.EndFused(stackalloc[] { y });
            return y;
        }

        /// <summary>
        /// <see cref="GraphFusion.FullAttention"/>: softmax(scale q k^T) v per head in F32 over q [d, heads, rows] and k, v [d,
        /// heads, keys] (contiguous): each permuted to [d, rows, heads] and copied, an F32 flash attention, the result [d * heads,
        /// rows]; then, given <paramref name="gateLogits"/> [heads, rows] (-1: none), each head's rows times
        /// <paramref name="gateScale"/> * sigmoid(logit) (a nested <see cref="GraphFusion.HeadGate"/>).
        /// </summary>
        public static int FullAttention(GraphProgramBuilder g, int q, int k, int v, long d, long heads, long rows, float scale,
            int gateLogits = -1, float gateScale = 0f)
        {
            ArgumentNullException.ThrowIfNull(g);
            if (gateLogits >= 0) g.BeginFused(GraphFusion.FullAttention, stackalloc[] { q, k, v, gateLogits }, default, stackalloc[] { scale, gateScale });
            else g.BeginFused(GraphFusion.FullAttention, stackalloc[] { q, k, v }, default, stackalloc[] { scale });
            Span<long> permute = stackalloc long[] { 0, 2, 1, 3 };
            int qp = g.Op(GraphOp.Cont, g.Op(GraphOp.Permute, stackalloc[] { q }, permute));
            int kp = g.Op(GraphOp.Cont, g.Op(GraphOp.Permute, stackalloc[] { k }, permute));
            int vp = g.Op(GraphOp.Cont, g.Op(GraphOp.Permute, stackalloc[] { v }, permute));
            int attention = g.Op(GraphOp.FlashAttnExt, stackalloc[] { qp, kp, vp }, stackalloc long[] { PrecisionF32 },
                stackalloc[] { scale, 0f, 0f });
            int y = g.Op(GraphOp.Reshape, stackalloc[] { g.Op(GraphOp.Cont, attention) }, stackalloc long[] { d * heads, rows });
            if (gateLogits >= 0) y = HeadGate(g, y, gateLogits, d, heads, rows, gateScale);
            g.EndFused(stackalloc[] { y });
            return y;
        }
    }
}
