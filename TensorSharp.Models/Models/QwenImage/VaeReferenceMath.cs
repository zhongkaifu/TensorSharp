// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Managed primitives for the single-image (T=1) Qwen-Image-2.1 VAE (QwenImage21Vae
// holds the encoder/decoder drivers). Every operation follows the diffusers/sglang
// reference; large CPU elementwise passes use spatial tiling/parallel ranges, and
// supported backends accelerate convolution and spatial attention. The scalar
// equations remain the oracle.
//
// Tensor convention here: a feature map is a flat float[] in planar CHW order
// (channel c, row y, col x at index (c*H + y)*W + x). Time is degenerate (T=1).
// Causal Conv3d with temporal kernel KD on a length-1 time axis equals a 2D conv
// using only the *last* temporal kernel slice (front-padding makes the earlier
// slices multiply zeros).
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    internal sealed class Feature
    {
        public int C, H, W;
        public float[] D;
        public Feature(int c, int h, int w) { C = c; H = h; W = w; D = new float[(long)c * h * w]; }
        public Feature(int c, int h, int w, float[] d) { C = c; H = h; W = w; D = d; }
        public int Idx(int c, int y, int x) => (c * H + y) * W + x;
        /// <summary>A feature whose every element the caller overwrites: skips the zeroing pass
        /// (hundreds of MB per decoder map at 1024 px and above), and inside a managed encode or
        /// decode recycles a released map of the same size (<see cref="VaeFeaturePool"/>).</summary>
        public static Feature Uninitialized(int c, int h, int w)
        {
            int length = checked(c * h * w);
            return new(c, h, w, VaeFeaturePool.Current is { } pool ? pool.Rent(length) : GC.AllocateUninitializedArray<float>(length));
        }
    }

    internal static unsafe partial class VaeReferenceMath
    {
        // Set by QwenImage21Vae on a GGML backend: the conv stack runs on the device via
        // TSGgml_Conv2dF32 instead of the pure-C# loops. Every device conv is F32 end to
        // end: the Qwen-Image-2.1 VAE reaches finite activations above 65504 before its
        // final norm. Where QwenImage21Vae enables it, the fused whole-VAE graph
        // (TSGgml_QwenVaeRun) runs the entire encode/decode as one device-resident ggml
        // graph instead, falling back to the per-conv path when the backend can't run it.
        internal static bool UseGpuConv = true;

        // ---- primitive ops ----------------------------------------------------

        internal static void SiluInPlace(float[] d)
        {
            void Apply(int start, int count)
            {
                int end = start + count;
                for (int i = start; i < end; i++)
                {
                    float v = d[i];
                    d[i] = v / (1f + MathF.Exp(-v));
                }
            }
            // Large decoded feature maps contain hundreds of millions of values.
            // Partition contiguous ranges while keeping small previews serial.
            const int chunkSize = 64 * 1024;
            if (d.Length < 2 * chunkSize) Apply(0, d.Length);
            else Parallel.For(0, (d.Length - 1) / chunkSize + 1, chunk =>
            {
                int start = chunk * chunkSize;
                Apply(start, Math.Min(chunkSize, d.Length - start));
            });
        }

        // RMS norm over the channel dimension (F.normalize(dim=1) * sqrt(C) * gamma). Double
        // accumulation in channel order and the original float multiplication order, vectorized
        // across pixels: bit-identical to the scalar definition (see RmsNormChannelFast).
        internal static Feature RmsNormChannel(Feature x, float[] gamma) => RmsNormChannelFast(x, gamma, silu: false);

        // General 2D convolution. weight is OC*IC*KH*KW row-major (oc,ic,kh,kw).
        // Internal (with TryGpuConv2dMaybeTiled) for QwenVaeConvTilingTests. Off the device it
        // runs the packed-GEMM convolution (VaeCpuOps), or the scalar loop when a test sets
        // UseScalarCpu.
        internal static Feature Conv2d(Feature x, float[] weight, int OC, int IC, int KH, int KW,
            float[] bias, int strideH, int strideW, int padT, int padB, int padL, int padR)
        {
            if (x.C != IC) throw new ArgumentException($"conv IC {IC} != input C {x.C}");
            int H = x.H, W = x.W;
            int Hp = H + padT + padB, Wp = W + padL + padR;
            int Ho = (Hp - KH) / strideH + 1, Wo = (Wp - KW) / strideW + 1;

            if (UseGpuConv && TryGpuConv2dMaybeTiled(x, weight, OC, IC, KH, KW, bias,
                    strideH, strideW, padT, padB, padL, padR, Ho, Wo, out Feature gpu))
                return gpu;

            if (!UseScalarCpu)
                return Conv2dCpu(x, PackConvWeight(weight, OC, IC * KH * KW), bias, OC, KH, KW,
                    strideH, strideW, padT, padB, padL, padR);
            return Conv2dScalar(x, weight, OC, IC, KH, KW, bias, strideH, strideW, padT, padB, padL, padR);
        }

        // The original direct convolution: the oracle for the GEMM path.
        internal static Feature Conv2dScalar(Feature x, float[] weight, int OC, int IC, int KH, int KW,
            float[] bias, int strideH, int strideW, int padT, int padB, int padL, int padR)
        {
            if (x.C != IC) throw new ArgumentException($"conv IC {IC} != input C {x.C}");
            int H = x.H, W = x.W;
            int Hp = H + padT + padB, Wp = W + padL + padR;
            int Ho = (Hp - KH) / strideH + 1, Wo = (Wp - KW) / strideW + 1;
            var outp = new Feature(OC, Ho, Wo);
            int hw = H * W;
            Parallel.For(0, OC, oc =>
            {
                long wBase = (long)oc * IC * KH * KW;
                float b = bias != null ? bias[oc] : 0f;
                for (int oy = 0; oy < Ho; oy++)
                {
                    int iy0 = oy * strideH - padT;
                    for (int ox = 0; ox < Wo; ox++)
                    {
                        int ix0 = ox * strideW - padL;
                        float acc = b;
                        for (int ic = 0; ic < IC; ic++)
                        {
                            long wc = wBase + (long)ic * KH * KW;
                            int cbase = ic * hw;
                            for (int ky = 0; ky < KH; ky++)
                            {
                                int iy = iy0 + ky;
                                if ((uint)iy >= (uint)H) continue;
                                int rowBase = cbase + iy * W;
                                long wrow = wc + (long)ky * KW;
                                for (int kx = 0; kx < KW; kx++)
                                {
                                    int ix = ix0 + kx;
                                    if ((uint)ix >= (uint)W) continue;
                                    acc += x.D[rowBase + ix] * weight[wrow + kx];
                                }
                            }
                        }
                        outp.D[(oc * Ho + oy) * Wo + ox] = acc;
                    }
                }
            });
            return outp;
        }

        // im2col scratch budget (bytes). The F32 device conv materializes an im2col tensor of
        // ~IC*KH*KW * OH * OW * 4 bytes; at high resolution that is several GB and either
        // spills into WDDM shared VRAM (≈3x slower VAE) or OOMs. When the estimate exceeds
        // this budget the conv is split into horizontal output bands (below).
        // Im2colBudgetForTest > 0 replaces the budget, so tests can force bands at small sizes.
        internal static long Im2colBudgetForTest;

        private static long Im2colBudgetBytes() => Im2colBudgetForTest > 0 ? Im2colBudgetForTest : 1024L * 1024 * 1024;   // 1 GiB

        // Run the device conv whole when its im2col fits the budget; otherwise split the
        // output into horizontal bands. Each band re-runs the SAME conv on a vertical slice
        // of the input (manually zero-padded so the device op runs with pad 0), so the result
        // is bit-identical to the un-tiled conv — only the transient im2col is bounded. The
        // surrounding group-norm / feature maps are never split, so there are NO tile seams
        // (unlike whole-VAE tiling).
        internal static bool TryGpuConv2dMaybeTiled(Feature x, float[] weight, int OC, int IC, int KH, int KW,
            float[] bias, int strideH, int strideW, int padT, int padB, int padL, int padR,
            int Ho, int Wo, out Feature result)
        {
            const int elementBytes = sizeof(float);
            long im2col = (long)IC * KH * KW * Ho * Wo * elementBytes;
            long budget = Im2colBudgetBytes();
            if (im2col <= budget)
                return TryGpuConv2d(x, weight, OC, IC, KH, KW, bias, strideH, strideW,
                    padT, padB, padL, padR, Ho, Wo, out result);

            long perRow = Math.Max(1, (long)IC * KH * KW * Wo * elementBytes);
            int bandHo = (int)Math.Max(1, budget / perRow);
            int H = x.H;
            var outp = new Feature(OC, Ho, Wo);
            for (int oy0 = 0; oy0 < Ho; oy0 += bandHo)
            {
                int oy1 = Math.Min(Ho, oy0 + bandHo);
                int rows = oy1 - oy0;
                int ir0 = oy0 * strideH - padT;             // first input row (in unpadded coords) the band reads
                int ir1 = (oy1 - 1) * strideH - padT + KH;  // one past the last (exclusive)
                int realStart = Math.Max(0, ir0), realEnd = Math.Min(H, ir1);
                int bandPadT = realStart - ir0;             // missing top rows (global zero padding), >= 0
                int bandPadB = ir1 - realEnd;               // missing bottom rows, >= 0
                // Manually pad the band on all sides (vertical band pad + horizontal conv pad)
                // so the device conv runs with pad 0 and emits exactly [OC, rows, Wo].
                Feature band = PadBand(x, realStart, realEnd, bandPadT, bandPadB, padL, padR);
                if (!TryGpuConv2d(band, weight, OC, IC, KH, KW, bias, strideH, strideW,
                        0, 0, 0, 0, rows, Wo, out Feature ob))
                { result = null; return false; }
                for (int oc = 0; oc < OC; oc++)
                    Array.Copy(ob.D, (long)oc * rows * Wo, outp.D, ((long)oc * Ho + oy0) * Wo, (long)rows * Wo);
            }
            result = outp;
            return true;
        }

        // Vertical slice rows [r0,r1) of x, zero-padded by (padTop,padBot) rows and (padL,padR)
        // columns, into a fresh [C, padTop+(r1-r0)+padBot, padL+W+padR] feature (zeros elsewhere).
        private static Feature PadBand(Feature x, int r0, int r1, int padTop, int padBot, int padL, int padR)
        {
            int C = x.C, H = x.H, W = x.W, srcRows = r1 - r0;
            int oh = padTop + srcRows + padBot, ow = padL + W + padR;
            var d = new float[(long)C * oh * ow];
            for (int c = 0; c < C; c++)
                for (int ry = 0; ry < srcRows; ry++)
                    Array.Copy(x.D, ((long)c * H + (r0 + ry)) * W,
                               d, ((long)c * oh + (padTop + ry)) * ow + padL, W);
            return new Feature(C, oh, ow, d);
        }

        // Device convolution via TSGgml_Conv2dF32. C# Feature [C,H,W] == ggml [W,H,C];
        // weight [OC,IC,KH,KW] == ggml [KW,KH,IC,OC]; output [OC,OH,OW] == ggml
        // [OW,OH,OC] — same byte order, no transposes. Returns false (falls back to
        // the C# path) if the device op can't run it.
        private static unsafe bool TryGpuConv2d(Feature x, float[] weight, int OC, int IC, int KH, int KW,
            float[] bias, int strideH, int strideW, int padT, int padB, int padL, int padR,
            int Ho, int Wo, out Feature result)
        {
            var outp = new Feature(OC, Ho, Wo);
            bool ok;
            fixed (float* xp = x.D, wp = weight, op = outp.D, bp = bias)
            {
                var d = new Conv2dArgs
                {
                    Input = (IntPtr)xp, W = x.W, H = x.H, C = x.C,
                    Weight = (IntPtr)wp, WType = 0 /* F32 */, KW = KW, KH = KH, IC = IC, OC = OC,
                    WeightBytes = (long)OC * IC * KH * KW * sizeof(float),
                    Bias = (IntPtr)bp, Output = (IntPtr)op,
                    StrideW = strideW, StrideH = strideH,
                    PadL = padL, PadR = padR, PadT = padT, PadB = padB,
                    StructBytes = Marshal.SizeOf<Conv2dArgs>(),
                };
                ok = GgmlBasicOps.TryConv2dF32(in d);
            }
            result = ok ? outp : null;
            return ok;
        }

        // Causal Conv3d on T=1 uses the last temporal slice (kd = KD-1) of the 5D weight
        // (oc,ic,kd,kh,kw): a KH x KW 2D conv with spatial padding. VaeWeights extracts that
        // slice once per layer (KernelSlice / PackedKernel), not on every call.
        internal static float[] LastTemporalSlice(float[] w5d, int OC, int IC, int KD, int KH, int KW)
        {
            if (KD == 1) return w5d;   // the (oc,ic,1,kh,kw) weight already is the 2D kernel
            var slice = new float[(long)OC * IC * KH * KW];
            int khw = KH * KW, kdhw = KD * KH * KW;
            for (int oc = 0; oc < OC; oc++)
                for (int ic = 0; ic < IC; ic++)
                    Array.Copy(w5d, ((long)oc * IC + ic) * kdhw + (long)(KD - 1) * khw, slice, ((long)oc * IC + ic) * khw, khw);
            return slice;
        }

        // ---- composite blocks -------------------------------------------------

        private static Feature AttentionBlock(VaeWeights w, string prefix, Feature x)
        {
            int C = x.C, H = x.H, W = x.W, hw = H * W;
            var identity = x;
            var xn = RmsNormChannel(x, w.Get(prefix + ".norm.gamma"));
            if (FastCpu)
            {
                // Each map is released to the decode's pool once read (the caller hands x over).
                var qkvFast = ConvLayer(w, prefix + ".to_qkv", xn, 3 * C, C, 1, 1, 1, 1, 0, 0, 0, 0);
                VaeFeaturePool.Release(xn);
                var attended = AttentionCpu(qkvFast.D, C, H, W);
                VaeFeaturePool.Release(qkvFast);
                var projected = ConvLayer(w, prefix + ".proj", attended, C, C, 1, 1, 1, 1, 0, 0, 0, 0);
                VaeFeaturePool.Release(attended);
                AddInPlace(projected, identity);
                VaeFeaturePool.Release(identity);
                return projected;
            }
            // to_qkv: 1x1 conv C -> 3C
            var qkv = Conv2d(xn, w.Get(prefix + ".to_qkv.weight"), 3 * C, C, 1, 1,
                w.Get(prefix + ".to_qkv.bias"), 1, 1, 0, 0, 0, 0);
            // q,k,v: [C, hw] each (channel-planar). attention over hw positions, single head, dim=C.
            float scale = 1f / MathF.Sqrt(C);
            var outp = new Feature(C, H, W);
            if (UseGpuConv && GgmlBasicOps.TryQwenVaeAttention(qkv.D, outp.D, C, hw))
            {
                var projected = Conv2d(outp, w.Get(prefix + ".proj.weight"), C, C, 1, 1,
                    w.Get(prefix + ".proj.bias"), 1, 1, 0, 0, 0, 0);
                return AddInPlace(projected, identity);
            }
            // scores[i,j] = sum_c q[c,i]*k[c,j] * scale ; softmax over j ; out[c,i] = sum_j p[i,j]*v[c,j]
            Parallel.For(0, hw, i =>
            {
                var scores = new float[hw];
                float mx = float.NegativeInfinity;
                for (int j = 0; j < hw; j++)
                {
                    float s = 0;
                    for (int c = 0; c < C; c++) s += qkv.D[c * hw + i] * qkv.D[(C + c) * hw + j];
                    s *= scale;
                    scores[j] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0;
                for (int j = 0; j < hw; j++) { float e = MathF.Exp(scores[j] - mx); scores[j] = e; sum += e; }
                float invSum = 1f / sum;
                for (int c = 0; c < C; c++)
                {
                    float acc = 0;
                    int vbase = (2 * C + c) * hw;
                    for (int j = 0; j < hw; j++) acc += scores[j] * qkv.D[vbase + j];
                    outp.D[c * hw + i] = acc * invSum;
                }
            });
            var proj = Conv2d(outp, w.Get(prefix + ".proj.weight"), C, C, 1, 1,
                w.Get(prefix + ".proj.bias"), 1, 1, 0, 0, 0, 0);
            return AddInPlace(proj, identity);
        }

        // Downsample (encoder): ZeroPad2d((0,1,0,1)) + Conv2d(dim,dim,3,stride2,pad0).
        private static Feature Downsample(VaeWeights w, string prefix, Feature x, int dim)
        {
            return ConvLayer(w, prefix + ".resample.1", x, dim, dim, 3, 3,
                2, 2, /*padT*/0, /*padB*/1, /*padL*/0, /*padR*/1);
        }
    }
}
