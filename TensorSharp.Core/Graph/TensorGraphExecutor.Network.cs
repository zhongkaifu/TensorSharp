// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
// The interpreter passes null result tensors to Ops (the op registry allocates the result), which
// Ops' signatures do not annotate; nullable analysis is off for this file.
#nullable disable
using System;
using System.Collections.Generic;

namespace TensorSharp.Graph
{
    /// <summary>
    /// The ops non-decoder networks (vision towers, audio encoders, small speech decoders) add to the interpreter:
    /// ggml_norm, the exact GELU, a depthwise 2-D convolution and ggml's VISION 2-D rotary. Each has a reference here
    /// built from the backend's generic ops (the direct CUDA executor runs these, PTX unchanged); a backend with a
    /// kernel of its own overrides it.
    /// </summary>
    public abstract partial class TensorGraphExecutor
    {
        /// <summary>GGML_ROPE_TYPE_VISION.</summary>
        private const int RopeTypeVision = 24;

        // Abramowitz and Stegun 7.1.26: erf(z) = 1 - (a1 t + ... + a5 t^5) exp(-z^2), t = 1 / (1 + p z), z >= 0, |error|
        // below 1.5e-7 (ggml-metal's erf_approx).
        private const float ErfP = 0.3275911f, ErfA1 = 0.254829592f, ErfA2 = -0.284496736f, ErfA3 = 1.421413741f,
            ErfA4 = -1.453152027f, ErfA5 = 1.061405429f;

        /// <summary>ggml_norm over x's innermost rows.</summary>
        private Tensor LayerNormRows(Tensor x0, float eps, RunState run)
        {
            Tensor x = Contig(x0, run);
            long cols = x.Sizes[^1], rows = x.ElementCount() / cols;
            using Tensor rows2 = x.View(rows, cols);
            using Tensor normed = NormRows(rows2, eps);
            return normed.View(x.Sizes);
        }

        /// <summary>ggml_norm of the rows of <paramref name="x"/> ([rows, cols], contiguous F32): (x - mean) / sqrt(var + eps),
        /// var the rows' mean squared deviation. Default: the centred rows' RMS norm (the same quotient).</summary>
        protected virtual Tensor NormRows(Tensor x, float eps)
        {
            long rows = x.Sizes[0], cols = x.Sizes[1];
            var scratch = new RunState();
            try
            {
                Tensor sum = scratch.Keep(SumRows(x));
                Tensor mean = scratch.Keep(Ops.Mul(null, sum, 1f / cols));
                Tensor column = scratch.Keep(mean.View(rows, 1));
                Tensor centred = scratch.Keep(Binary(x, column, scratch, (r, a, b) => Ops.Sub(r, a, b), GraphOp.Sub));
                return Ops.RMSNorm(null, centred, null, null, eps);
            }
            finally
            {
                foreach (Tensor t in scratch.Owned) t.Dispose();
                foreach (Tensor t in scratch.Lasting) t.Dispose();
            }
        }

        /// <summary>
        /// 0.5 x (1 + erf(x / sqrt 2)) of a contiguous F32 tensor. Default: erf by Abramowitz and Stegun 7.1.26 in the
        /// backend's elementwise ops, folded as 0.5 (x + |x| (1 - P(t) exp(-x^2 / 2))) with t = 1 / (1 + p |x| / sqrt 2).
        /// </summary>
        protected virtual Tensor GeluErf(Tensor x)
        {
            using Tensor ax = Ops.Abs(null, x);
            using Tensor den = Ops.Mul(null, ax, ErfP / MathF.Sqrt(2f));
            Ops.Add(den, den, 1f);
            using Tensor t = Ops.Div(null, 1f, den);
            using Tensor poly = Ops.Mul(null, t, ErfA5);
            foreach (float a in new[] { ErfA4, ErfA3, ErfA2, ErfA1 })
            {
                Ops.Add(poly, poly, a);
                Ops.Mul(poly, poly, t);
            }
            using Tensor e = Ops.Mul(null, x, x);
            Ops.Mul(e, e, -0.5f);
            Ops.Exp(e, e);
            Ops.Mul(poly, poly, e);
            Ops.Mul(poly, poly, ax);
            Tensor r = Ops.Add(null, x, ax);
            Ops.Sub(r, r, poly);
            Ops.Mul(r, r, 0.5f);
            return r;
        }

        /// <summary>ggml_pad: <paramref name="x"/> with Ip[i] zeros after its extent along ne[i] (keys padded to their mask).</summary>
        private Tensor PadNode(Tensor x, GraphNode n, RunState run)
        {
            int dims = x.DimensionCount;
            long Ip(int i) => n.Ip != null && n.Ip.Length > i ? n.Ip[i] : 0;
            for (int i = dims; i < 4; i++)
                if (Ip(i) != 0) throw new NotSupportedException($"padding ne[{i}] of a {dims}-dimensional tensor");
            var sizes = new long[dims];
            bool any = false;
            for (int d = 0; d < dims; d++)
            {
                long pad = Ip(dims - 1 - d);
                if (pad < 0) throw new NotSupportedException("negative padding");
                any |= pad > 0;
                sizes[d] = x.Sizes[d] + pad;
            }
            var result = new Tensor(Allocator, x.ElementType, sizes);
            Ops.Fill(result, 0f);
            if (!any)
            {
                Ops.Copy(result, x);
                return result;
            }
            Tensor inner = result;
            var narrowed = new List<Tensor>();
            for (int d = 0; d < dims; d++)
                if (sizes[d] != x.Sizes[d])
                {
                    inner = inner.Narrow(d, 0, x.Sizes[d]);
                    narrowed.Add(inner);
                }
            Ops.Copy(inner, x);
            foreach (Tensor t in narrowed) t.Dispose();
            return result;
        }

        private Tensor DepthwiseConv2dNode(Tensor x, Tensor kernel, GraphNode n, RunState run)
        {
            long[] nx = Ne(x), nk = Ne(kernel);
            long c = nx[0], w = nx[1], h = nx[2], b = nx[3], kw = nk[0], kh = nk[1];
            if (nk[0] * nk[1] * nk[2] * nk[3] != kw * kh * c || kw < 1 || kh < 1)
                throw new NotSupportedException($"a depthwise kernel [{string.Join(",", nk)}] over {c} channels (it holds KW x KH values a channel)");
            int Ip(int i, int fallback) => n.Ip != null && n.Ip.Length > i ? checked((int)n.Ip[i]) : fallback;
            int s0 = Ip(0, 1), s1 = Ip(1, 1), p0 = Ip(2, 0), p1 = Ip(3, 0), d0 = Ip(4, 1), d1 = Ip(5, 1);
            if (s0 < 1 || s1 < 1 || d0 < 1 || d1 < 1 || p0 < 0 || p1 < 0)
                throw new NotSupportedException($"depthwise conv stride ({s0}, {s1}), padding ({p0}, {p1}), dilation ({d0}, {d1})");
            if (w + 2L * p0 - (long)d0 * (kw - 1) - 1 < 0 || h + 2L * p1 - (long)d1 * (kh - 1) - 1 < 0)
                throw new NotSupportedException($"a {kw}x{kh} depthwise kernel over a {w}x{h} input padded ({p0}, {p1})");
            Tensor x4 = run.Keep(Contig(x, run).View(b, h, w, c));
            Tensor k3 = run.Keep(Contig(F32(kernel, run.Owned), run).View(c, kh, kw));
            return DepthwiseConv2d(x4, k3, s0, s1, p0, p1, d0, d1);
        }

        /// <summary>
        /// <see cref="GraphOp.DepthwiseConv2d"/>: x [N, H, W, C] and the kernel [C, KH, KW] (both contiguous F32), stride,
        /// padding and dilation along W (0) and H (1); returns [N, OH, OW, C]. Default: the zero-padded input's strided tap
        /// views, each times its taps' channel vector, summed in (ky, kx) order.
        /// </summary>
        protected virtual Tensor DepthwiseConv2d(Tensor x, Tensor kernel, int s0, int s1, int p0, int p1, int d0, int d1)
        {
            long b = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2], c = x.Sizes[3], kh = kernel.Sizes[1], kw = kernel.Sizes[2];
            long oh = (h + 2L * p1 - d1 * (kh - 1) - 1) / s1 + 1, ow = (w + 2L * p0 - d0 * (kw - 1) - 1) / s0 + 1;
            var scratch = new RunState();
            Tensor acc = null;
            try
            {
                Tensor padded = x;
                if (p0 > 0 || p1 > 0)
                {
                    padded = scratch.Keep(new Tensor(Allocator, DType.Float32, b, h + 2L * p1, w + 2L * p0, c));
                    Ops.Fill(padded, 0f);
                    using Tensor band = padded.Narrow(1, p1, h);
                    using Tensor inner = band.Narrow(2, p0, w);
                    CopyInto(inner, x);
                }
                // The taps channels innermost: tap (ky, kx) is a contiguous [C] vector.
                using Tensor permuted = kernel.Permute(1, 2, 0);
                Tensor taps = scratch.Keep(Contiguous(permuted));
                long[] strides = { padded.Strides[0], s1 * padded.Strides[1], s0 * padded.Strides[2], padded.Strides[3] };
                for (long ky = 0; ky < kh; ky++)
                {
                    for (long kx = 0; kx < kw; kx++)
                    {
                        using Tensor tap = new Tensor(new[] { b, oh, ow, c }, strides, padded.Storage,
                            padded.StorageOffset + ky * d1 * padded.Strides[1] + kx * d0 * padded.Strides[2]);
                        using Tensor vector = new Tensor(new[] { 1L, 1L, 1L, c }, new[] { c, c, c, 1L }, taps.Storage,
                            taps.StorageOffset + (ky * kw + kx) * c);
                        Tensor product = Binary(tap, vector, scratch, (r, l, m) => Ops.Mul(r, l, m), GraphOp.Mul);
                        if (acc == null)
                        {
                            acc = product;
                            continue;
                        }
                        Ops.Add(acc, acc, product);
                        product.Dispose();
                    }
                }
                Tensor result = acc;
                acc = null;
                return result;
            }
            finally
            {
                acc?.Dispose();
                foreach (Tensor t in scratch.Owned) t.Dispose();
                foreach (Tensor t in scratch.Lasting) t.Dispose();
            }
        }

        /// <summary>
        /// ggml_rope_multi's VISION mode over x [d, heads, tokens] (d = 2 n_dims): pair (i, i + n_dims), i &lt; n_dims, turns
        /// by the angle of its section's axis (sections restart their frequencies: indep_sects), the positions axis-major
        /// [4 * tokens]. The angles are computed on the host as ggml-cpu computes them (each section's theta multiplied on
        /// by base^(-2 / n_dims) a pair); the rotation runs on the device as x * cos + (-x_hi | x_lo) * sin.
        /// </summary>
        private Tensor RopeVision(Tensor x, int[] pos, GraphNode n, RunState run)
        {
            long[] ne = Ne(x);
            int d = (int)ne[0], heads = (int)ne[1], tokens = (int)ne[2];
            int dims = (int)n.Ip[0];
            if (ne[3] != 1) throw new NotSupportedException("batched vision rotary");
            if (2 * dims != d)
                throw new NotSupportedException($"vision rotary of {dims} dims over a {d}-wide head (it turns pairs (i, i + n_dims) of a 2 n_dims head)");
            int Ip(int i) => n.Ip.Length > i ? (int)n.Ip[i] : 0;
            int s0 = Ip(3), s1 = Ip(4), s2 = Ip(5), s3 = Ip(6), sect = s0 + s1 + s2 + s3;
            if (sect <= 0) throw new NotSupportedException("vision rotary without sections");
            float freqBase = n.Fp[0], freqScale = n.Fp[1], extFactor = n.Fp[2], attnFactor = n.Fp[3];
            if (extFactor != 0f) throw new NotSupportedException("YaRN vision rotary");
            if (pos.Length < 4L * tokens) throw new NotSupportedException($"vision rotary positions: {pos.Length} for {tokens} tokens (needs 4 a token)");
            float thetaScale = MathF.Pow(freqBase, -2.0f / dims);
            var cos = new float[(long)tokens * d];
            var sin = new float[(long)tokens * d];
            for (int t = 0; t < tokens; t++)
            {
                float bt = pos[t], bh = pos[tokens + t], bw = pos[2 * tokens + t], be = pos[3 * tokens + t];
                float tt = bt, th = bh, tw = bw, te = be;
                for (int i = 0; i < dims; i++)
                {
                    int sector = i % sect;
                    if (sector == 0) tt = bt;
                    else if (sector == s0) th = bh;
                    else if (sector == s0 + s1) tw = bw;
                    else if (sector == s0 + s1 + s2) te = be;
                    float theta = sector < s0 ? tt : sector < s0 + s1 ? th : sector < s0 + s1 + s2 ? tw : te;
                    float cv = MathF.Cos(theta * freqScale) * attnFactor, sv = MathF.Sin(theta * freqScale) * attnFactor;
                    long at = (long)t * d + i;
                    cos[at] = cos[at + dims] = cv;
                    sin[at] = sin[at + dims] = sv;
                    tt *= thetaScale;
                    th *= thetaScale;
                    tw *= thetaScale;
                    te *= thetaScale;
                }
            }
            Tensor x3 = run.Keep(Contig(x, run).View(tokens, heads, d));
            Tensor cosT = run.Keep(new Tensor(Allocator, DType.Float32, tokens, 1, d));
            cosT.SetElementsAsFloat(cos);
            Tensor sinT = run.Keep(new Tensor(Allocator, DType.Float32, tokens, 1, d));
            sinT.SetElementsAsFloat(sin);
            Tensor lo = run.Keep(x3.Narrow(2, 0, dims)), hi = run.Keep(x3.Narrow(2, dims, dims));
            Tensor negHi = run.Keep(Ops.Neg(null, In(hi, run)));
            Tensor rotated = run.Keep(Concat(negHi, In(lo, run), 0, run));
            Tensor straight = run.Keep(Binary(x3, cosT, run, (r, a, b) => Ops.Mul(r, a, b), GraphOp.Mul));
            Tensor turned = run.Keep(Binary(rotated, sinT, run, (r, a, b) => Ops.Mul(r, a, b), GraphOp.Mul));
            return Ops.Add(null, straight, turned);
        }
    }
}
