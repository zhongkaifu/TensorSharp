// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Managed CPU reference for AutoencoderKLQwenImage (single image / T=1). See
// QwenImageVae.Reference.cs for the single-frame simplification rationale. Every
// operation is a faithful, un-optimized port of the diffusers/sglang reference,
// validated numerically against diffusers (tools/qwen_image_vae_reference.py).
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
    }

    internal static class VaeReferenceMath
    {
        // When set (by QwenImageVae on a GGML backend), the conv stack runs on the
        // device via TSGgml_Conv2d instead of the pure-C# scalar loops. This is THE
        // fix for the ~459 s/1 MP CPU-bound, GPU-idle VAE encode. Disable with
        // TS_QWEN_VAE_GPU=0.
        internal static bool UseGpuConv =
            Environment.GetEnvironmentVariable("TS_QWEN_VAE_GPU") != "0";

        private const int BaseDim = 96;
        private const int ZDim = 16;
        private static readonly int[] DimMult = { 1, 2, 4, 4 };
        private const int NumRes = 2;
        // temperal_downsample (encoder) = (False,True,True); temperal_upsample (decoder) = (True,True,False)

        // ---- primitive ops ----------------------------------------------------

        private static void SiluInPlace(float[] d)
        {
            for (long i = 0; i < d.Length; i++)
            {
                float v = d[i];
                d[i] = v / (1f + MathF.Exp(-v));
            }
        }

        // RMS norm over the channel dimension (F.normalize(dim=1) * sqrt(C) * gamma).
        private static Feature RmsNormChannel(Feature x, float[] gamma)
        {
            int C = x.C, H = x.H, W = x.W, hw = H * W;
            var outp = new Feature(C, H, W);
            float scale = MathF.Sqrt(C);
            Parallel.For(0, hw, p =>
            {
                double ss = 0;
                for (int c = 0; c < C; c++) { float v = x.D[c * hw + p]; ss += (double)v * v; }
                float inv = (float)(1.0 / Math.Sqrt(ss + 1e-12));
                for (int c = 0; c < C; c++)
                    outp.D[c * hw + p] = x.D[c * hw + p] * inv * scale * gamma[c];
            });
            return outp;
        }

        // General 2D convolution. weight is OC*IC*KH*KW row-major (oc,ic,kh,kw).
        private static Feature Conv2d(Feature x, float[] weight, int OC, int IC, int KH, int KW,
            float[] bias, int strideH, int strideW, int padT, int padB, int padL, int padR)
        {
            if (x.C != IC) throw new ArgumentException($"conv IC {IC} != input C {x.C}");
            int H = x.H, W = x.W;
            int Hp = H + padT + padB, Wp = W + padL + padR;
            int Ho = (Hp - KH) / strideH + 1, Wo = (Wp - KW) / strideW + 1;

            if (UseGpuConv && TryGpuConv2d(x, weight, OC, IC, KH, KW, bias,
                    strideH, strideW, padT, padB, padL, padR, Ho, Wo, out Feature gpu))
                return gpu;

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

        // Device convolution via TSGgml_Conv2d. C# Feature [C,H,W] == ggml [W,H,C];
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
                ok = GgmlBasicOps.TryConv2d(in d);
            }
            result = ok ? outp : null;
            return ok;
        }

        // Causal Conv3d on T=1: use the last temporal slice (kd = KD-1) of the 5D
        // weight (oc,ic,kd,kh,kw), reducing to a KH x KW 2D conv with spatial padding.
        private static Feature CausalConv3dT1(Feature x, float[] w5d, int OC, int IC, int KD, int KH, int KW,
            float[] bias, int pad)
        {
            // Extract the kd = KD-1 slice into a (OC,IC,KH,KW) kernel.
            var slice = new float[(long)OC * IC * KH * KW];
            int khw = KH * KW, kdhw = KD * KH * KW;
            Parallel.For(0, OC, oc =>
            {
                for (int ic = 0; ic < IC; ic++)
                {
                    long src = ((long)oc * IC + ic) * kdhw + (long)(KD - 1) * khw;
                    long dst = ((long)oc * IC + ic) * khw;
                    Array.Copy(w5d, src, slice, dst, khw);
                }
            });
            return Conv2d(x, slice, OC, IC, KH, KW, bias, 1, 1, pad, pad, pad, pad);
        }

        private static Feature AddInPlace(Feature a, Feature b)
        {
            for (long i = 0; i < a.D.Length; i++) a.D[i] += b.D[i];
            return a;
        }

        private static Feature NearestUpsample2x(Feature x)
        {
            int C = x.C, H = x.H, W = x.W, Ho = H * 2, Wo = W * 2;
            var outp = new Feature(C, Ho, Wo);
            Parallel.For(0, C, c =>
            {
                for (int oy = 0; oy < Ho; oy++)
                    for (int ox = 0; ox < Wo; ox++)
                        outp.D[(c * Ho + oy) * Wo + ox] = x.D[(c * H + oy / 2) * W + ox / 2];
            });
            return outp;
        }

        // ---- composite blocks -------------------------------------------------

        private static Feature ResidualBlock(VaeWeights w, string prefix, Feature x, int inDim, int outDim)
        {
            Feature h;
            if (inDim != outDim)
                h = CausalConv3dT1(x, w.Get(prefix + ".shortcut.weight"), outDim, inDim, 1, 1, 1,
                        w.Get(prefix + ".shortcut.bias"), 0);
            else
                h = new Feature(x.C, x.H, x.W, (float[])x.D.Clone());

            var t = RmsNormChannel(x, w.Get(prefix + ".residual.0.gamma"));
            SiluInPlace(t.D);
            t = CausalConv3dT1(t, w.Get(prefix + ".residual.2.weight"), outDim, inDim, 3, 3, 3,
                    w.Get(prefix + ".residual.2.bias"), 1);
            t = RmsNormChannel(t, w.Get(prefix + ".residual.3.gamma"));
            SiluInPlace(t.D);
            t = CausalConv3dT1(t, w.Get(prefix + ".residual.6.weight"), outDim, outDim, 3, 3, 3,
                    w.Get(prefix + ".residual.6.bias"), 1);
            return AddInPlace(t, h);
        }

        private static Feature AttentionBlock(VaeWeights w, string prefix, Feature x)
        {
            int C = x.C, H = x.H, W = x.W, hw = H * W;
            var identity = x;
            var xn = RmsNormChannel(x, w.Get(prefix + ".norm.gamma"));
            // to_qkv: 1x1 conv C -> 3C
            var qkv = Conv2d(xn, w.Get(prefix + ".to_qkv.weight"), 3 * C, C, 1, 1,
                w.Get(prefix + ".to_qkv.bias"), 1, 1, 0, 0, 0, 0);
            // q,k,v: [C, hw] each (channel-planar). attention over hw positions, single head, dim=C.
            float scale = 1f / MathF.Sqrt(C);
            var outp = new Feature(C, H, W);
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
            return Conv2d(x, w.Get(prefix + ".resample.1.weight"), dim, dim, 3, 3,
                w.Get(prefix + ".resample.1.bias"), 2, 2, /*padT*/0, /*padB*/1, /*padL*/0, /*padR*/1);
        }

        // Upsample (decoder): nearest x2 + Conv2d(dim, dim/2, 3, pad1).
        private static Feature Upsample(VaeWeights w, string prefix, Feature x, int dim)
        {
            var up = NearestUpsample2x(x);
            return Conv2d(up, w.Get(prefix + ".resample.1.weight"), dim / 2, dim, 3, 3,
                w.Get(prefix + ".resample.1.bias"), 1, 1, 1, 1, 1, 1);
        }

        private static Feature MidBlock(VaeWeights w, string prefix, Feature x, int dim)
        {
            x = ResidualBlock(w, prefix + ".0", x, dim, dim);
            x = AttentionBlock(w, prefix + ".1", x);
            x = ResidualBlock(w, prefix + ".2", x, dim, dim);
            return x;
        }

        // ---- encoder / decoder drivers ---------------------------------------

        public static VaeLatent Encode(VaeWeights w, RgbImage image)
        {
            int H = image.Height, Wd = image.Width;
            var x = new Feature(3, H, Wd, image.ToPlanarChw());
            // normalize pixels [0,1] -> [-1,1]
            for (long i = 0; i < x.D.Length; i++) x.D[i] = x.D[i] * 2f - 1f;

            // conv_in (encoder.conv1): 3 -> 96
            x = CausalConv3dT1(x, w.Get("encoder.conv1.weight"), BaseDim, 3, 3, 3, 3,
                    w.Get("encoder.conv1.bias"), 1);

            int[] dims = { 96, 96, 192, 384, 384 };
            int idx = 0;
            for (int i = 0; i < 4; i++)
            {
                int inDim = dims[i], outDim = dims[i + 1];
                x = ResidualBlock(w, $"encoder.downsamples.{idx++}", x, inDim, outDim);
                x = ResidualBlock(w, $"encoder.downsamples.{idx++}", x, outDim, outDim);
                if (i != 3) x = Downsample(w, $"encoder.downsamples.{idx++}", x, outDim);
            }

            x = MidBlock(w, "encoder.middle", x, 384);

            x = RmsNormChannel(x, w.Get("encoder.head.0.gamma"));
            SiluInPlace(x.D);
            x = CausalConv3dT1(x, w.Get("encoder.head.2.weight"), ZDim * 2, 384, 3, 3, 3,
                    w.Get("encoder.head.2.bias"), 1);

            // quant_conv (conv1): 32 -> 32, 1x1x1
            x = CausalConv3dT1(x, w.Get("conv1.weight"), ZDim * 2, ZDim * 2, 1, 1, 1,
                    w.Get("conv1.bias"), 0);

            // DiagonalGaussian.mode(): take the mean (first z_dim channels)
            int lh = x.H, lw = x.W, lhw = lh * lw;
            var latent = new float[(long)ZDim * lhw];
            Array.Copy(x.D, 0, latent, 0, (long)ZDim * lhw);
            return new VaeLatent(ZDim, lh, lw, latent);
        }

        public static RgbImage Decode(VaeWeights w, VaeLatent latent)
        {
            var x = new Feature(latent.Channels, latent.Height, latent.Width, (float[])latent.Data.Clone());

            // post_quant_conv (conv2): 16 -> 16, 1x1x1
            x = CausalConv3dT1(x, w.Get("conv2.weight"), ZDim, ZDim, 1, 1, 1, w.Get("conv2.bias"), 0);

            // conv_in (decoder.conv1): 16 -> 384
            x = CausalConv3dT1(x, w.Get("decoder.conv1.weight"), 384, ZDim, 3, 3, 3,
                    w.Get("decoder.conv1.bias"), 1);

            x = MidBlock(w, "decoder.middle", x, 384);

            // up_blocks (flat decoder.upsamples), temperal_upsample (True,True,False):
            //   stage in/out + upsample(dim->dim/2) at i=0,1,2; none at i=3.
            int[] inDims = { 384, 192, 192, 96 };
            int[] outDims = { 384, 384, 192, 96 };
            int idx = 0;
            for (int i = 0; i < 4; i++)
            {
                int cur = inDims[i];
                for (int r = 0; r < NumRes + 1; r++)   // num_res_blocks + 1 = 3
                {
                    x = ResidualBlock(w, $"decoder.upsamples.{idx++}", x, cur, outDims[i]);
                    cur = outDims[i];
                }
                if (i != 3) x = Upsample(w, $"decoder.upsamples.{idx++}", x, outDims[i]);
            }

            x = RmsNormChannel(x, w.Get("decoder.head.0.gamma"));
            SiluInPlace(x.D);
            x = CausalConv3dT1(x, w.Get("decoder.head.2.weight"), 3, 96, 3, 3, 3,
                    w.Get("decoder.head.2.bias"), 1);

            // [-1,1] -> [0,1] (clamp happens at PNG encode)
            for (long i = 0; i < x.D.Length; i++) x.D[i] = (x.D[i] + 1f) * 0.5f;
            return RgbImage.FromPlanarChw(x.W, x.H, x.D);
        }
    }
}
