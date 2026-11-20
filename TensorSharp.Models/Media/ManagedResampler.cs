// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Threading.Tasks;

namespace TensorSharp.Models.Media
{
    /// <summary>
    /// Managed resizers over packed RGB u8 buffers.
    ///
    /// <para>The Lanczos path is the Pillow-compatible Lanczos-3 that Muse-Glimmer's
    /// preprocessor was ported with (llama.cpp tools/mtmd/mtmd-image.cpp
    /// <c>img_tool::resize_pillow(use_lanczos = true)</c>, itself adapted from Pillow's
    /// src/libImaging/Resample.c). It lives here so the managed image codec and the model
    /// share one copy; the properties that must be preserved for parity are:</para>
    /// <list type="bullet">
    ///   <item><description>separable: horizontal pass then vertical pass, each over u8 pixels;</description></item>
    ///   <item><description>per-output-pixel normalized filter coefficients;</description></item>
    ///   <item><description>22-bit fixed-point integer accumulation with a rounding bias.</description></item>
    /// </list>
    /// </summary>
    internal static class ManagedResampler
    {
        private const int PrecisionBits = 32 - 8 - 2;   // 22
        private const double FilterSupport = 3.0;        // Lanczos-3
        private const double BicubicSupport = 2.0;       // Pillow's BICUBIC (a = -0.5)

        private static double Sinc(double v)
        {
            if (v == 0.0)
                return 1.0;
            double piV = v * 3.141592653589793238462643383279502884;
            return Math.Sin(piV) / piV;
        }

        private static double Lanczos3(double x)
        {
            if (-3.0 <= x && x < 3.0)
                return Sinc(x) * Sinc(x / 3.0);
            return 0.0;
        }

        // Pillow's bicubic_filter (src/libImaging/Resample.c), a = -0.5.
        private static double Bicubic(double x)
        {
            const double a = -0.5;
            if (x < 0.0) x = -x;
            if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
            return 0.0;
        }

        /// <summary>
        /// Precompute the filter coefficients for one dimension. Returns the kernel
        /// size; bounds[xx*2+0] is the first contributing input index and
        /// bounds[xx*2+1] the contributing count for output pixel xx.
        /// </summary>
        private static int PrecomputeWeights(int inSize, int outSize, out int[] bounds, out int[] weights, bool bicubic = false)
        {
            if (inSize <= 0 || outSize <= 0)
                throw new ArgumentException("PrecomputeWeights requires positive sizes");

            double scale = (double)inSize / outSize;
            double filterScale = scale;
            if (filterScale < 1.0)
                filterScale = 1.0;

            double support = (bicubic ? BicubicSupport : FilterSupport) * filterScale;
            int ksize = (int)Math.Ceiling(support) * 2 + 1;

            double[] pre = new double[(long)outSize * ksize];
            bounds = new int[outSize * 2];

            for (int xx = 0; xx < outSize; xx++)
            {
                double center = (xx + 0.5) * scale;
                double ww = 0.0;
                double ss = 1.0 / filterScale;

                int xmin = (int)(center - support + 0.5);
                if (xmin < 0)
                    xmin = 0;

                int xmax = (int)(center + support + 0.5);
                if (xmax > inSize)
                    xmax = inSize;

                xmax -= xmin;

                int x;
                for (x = 0; x < xmax; x++)
                {
                    double t = (x + xmin - center + 0.5) * ss;
                    double w = bicubic ? Bicubic(t) : Lanczos3(t);
                    pre[(long)xx * ksize + x] = w;
                    ww += w;
                }

                for (x = 0; x < xmax; x++)
                {
                    if (ww != 0.0)
                        pre[(long)xx * ksize + x] /= ww;
                }

                for (; x < ksize; x++)
                    pre[(long)xx * ksize + x] = 0;

                bounds[xx * 2 + 0] = xmin;
                bounds[xx * 2 + 1] = xmax;
            }

            weights = new int[(long)outSize * ksize];
            double fxpScale = Math.ScaleB(1.0, PrecisionBits);
            for (long i = 0; i < (long)outSize * ksize; i++)
            {
                // Pillow adds +/- 0.5 then truncates toward zero (a plain round
                // would round twice).
                double rounded = pre[i] * fxpScale + (pre[i] < 0 ? -0.5 : 0.5);
                weights[i] = (int)rounded;
            }

            return ksize;
        }

        private static byte Clip8(int value)
        {
            if (value < 0) return 0;
            if (value > 255) return 255;
            return (byte)value;
        }

        private static byte[] ResampleHorizontal(byte[] src, int srcW, int srcH, int outW,
            int ksize, int[] bounds, int[] weights)
        {
            byte[] dst = new byte[(long)outW * srcH * 3];
            Parallel.For(0, srcH, yy =>
            {
                long srcRow = (long)yy * srcW * 3;
                long dstRow = (long)yy * outW * 3;
                for (int xx = 0; xx < outW; xx++)
                {
                    int xmin = bounds[xx * 2 + 0];
                    int xcnt = bounds[xx * 2 + 1];

                    int ss0 = 1 << (PrecisionBits - 1);
                    int ss1 = 1 << (PrecisionBits - 1);
                    int ss2 = 1 << (PrecisionBits - 1);

                    long wBase = (long)xx * ksize;
                    for (int x = 0; x < xcnt; x++)
                    {
                        long p = srcRow + (long)(x + xmin) * 3;
                        int w = weights[wBase + x];
                        ss0 += src[p] * w;
                        ss1 += src[p + 1] * w;
                        ss2 += src[p + 2] * w;
                    }

                    long q = dstRow + (long)xx * 3;
                    dst[q] = Clip8(ss0 >> PrecisionBits);
                    dst[q + 1] = Clip8(ss1 >> PrecisionBits);
                    dst[q + 2] = Clip8(ss2 >> PrecisionBits);
                }
            });
            return dst;
        }

        private static byte[] ResampleVertical(byte[] src, int srcW, int srcH, int outH,
            int ksize, int[] bounds, int[] weights)
        {
            byte[] dst = new byte[(long)srcW * outH * 3];
            Parallel.For(0, outH, yy =>
            {
                int ymin = bounds[yy * 2 + 0];
                int ycnt = bounds[yy * 2 + 1];
                long dstRow = (long)yy * srcW * 3;
                long wBase = (long)yy * ksize;

                for (int xx = 0; xx < srcW; xx++)
                {
                    int ss0 = 1 << (PrecisionBits - 1);
                    int ss1 = 1 << (PrecisionBits - 1);
                    int ss2 = 1 << (PrecisionBits - 1);

                    for (int y = 0; y < ycnt; y++)
                    {
                        long p = ((long)(y + ymin) * srcW + xx) * 3;
                        int w = weights[wBase + y];
                        ss0 += src[p] * w;
                        ss1 += src[p + 1] * w;
                        ss2 += src[p + 2] * w;
                    }

                    long q = dstRow + (long)xx * 3;
                    dst[q] = Clip8(ss0 >> PrecisionBits);
                    dst[q + 1] = Clip8(ss1 >> PrecisionBits);
                    dst[q + 2] = Clip8(ss2 >> PrecisionBits);
                }
            });
            return dst;
        }

        /// <summary>
        /// Two-pass separable Lanczos-3 resize over a packed RGB u8 buffer.
        /// Mirrors img_tool::resize_pillow's need_horizontal / need_vertical logic.
        /// </summary>
        internal static byte[] ResizeLanczosPillow(byte[] src, int srcW, int srcH, int dstW, int dstH) =>
            ResizePillow(src, srcW, srcH, dstW, dstH, bicubic: false, pillowPassOrder: false);

        /// <summary>
        /// Pillow's <c>Image.resize</c> of an 8-bit RGB image with LANCZOS (<paramref name="bicubic"/> false) or BICUBIC,
        /// byte for byte: the same two separable passes, coefficients and fixed-point rounding as
        /// <see cref="ResizeLanczosPillow"/>, with Pillow's bicubic kernel (a = -0.5, support 2) for the second. With
        /// <paramref name="pillowPassOrder"/> the passes run in the order Pillow 13 runs them (ImagingResampleInner: the
        /// vertical pass first when the height shrinks by more than twice what the width does, else the horizontal pass
        /// first); without it the horizontal pass always runs first, as llama.cpp's resize_pillow (which Muse-Glimmer
        /// matches) does.
        /// </summary>
        internal static byte[] ResizePillow(byte[] src, int srcW, int srcH, int dstW, int dstH, bool bicubic, bool pillowPassOrder = true)
        {
            if (dstW <= 0 || dstH <= 0)
                throw new ArgumentException($"Invalid target size {dstW}x{dstH}");

            bool needHorizontal = dstW != srcW;
            bool needVertical = dstH != srcH;

            if (needHorizontal && needVertical)
            {
                int kH = PrecomputeWeights(srcW, dstW, out int[] bH, out int[] wH, bicubic);
                int kV = PrecomputeWeights(srcH, dstH, out int[] bV, out int[] wV, bicubic);
                bool horizontalFirst = !pillowPassOrder || !(srcH - dstH > 0 && srcH - dstH > (long)(srcW - dstW) * 2);
                if (!horizontalFirst)
                {
                    byte[] rows = ResampleVertical(src, srcW, srcH, dstH, kV, bV, wV);
                    return ResampleHorizontal(rows, srcW, dstH, dstW, kH, bH, wH);
                }
                byte[] temp = ResampleHorizontal(src, srcW, srcH, dstW, kH, bH, wH);
                return ResampleVertical(temp, dstW, srcH, dstH, kV, bV, wV);
            }

            if (needHorizontal)
            {
                int kH = PrecomputeWeights(srcW, dstW, out int[] bH, out int[] wH, bicubic);
                return ResampleHorizontal(src, srcW, srcH, dstW, kH, bH, wH);
            }

            if (needVertical)
            {
                int kV = PrecomputeWeights(srcH, dstH, out int[] bV, out int[] wV, bicubic);
                return ResampleVertical(src, srcW, srcH, dstH, kV, bV, wV);
            }

            return src;
        }

        /// <summary>
        /// Bilinear resize over packed RGB u8: the same pixel-centre sampling as
        /// <see cref="ImageProcessorUtils.BilinearResize"/> (which works on RGBA), so the two
        /// agree wherever a model's reference resizes bilinearly.
        /// </summary>
        internal static byte[] ResizeBilinearRgb8(byte[] rgb, int srcW, int srcH, int dstW, int dstH)
        {
            if (dstW <= 0 || dstH <= 0)
                throw new ArgumentException($"Invalid target size {dstW}x{dstH}");
            if (dstW == srcW && dstH == srcH)
                return rgb;

            byte[] result = new byte[(long)dstW * dstH * 3];
            double xRatio = (double)srcW / dstW;
            double yRatio = (double)srcH / dstH;

            Parallel.For(0, dstH, dy =>
            {
                double srcY = (dy + 0.5) * yRatio - 0.5;
                int y0 = Math.Max(0, (int)srcY);
                int y1 = Math.Min(srcH - 1, y0 + 1);
                double fy = srcY - y0;
                if (fy < 0) fy = 0;

                for (int dx = 0; dx < dstW; dx++)
                {
                    double srcX = (dx + 0.5) * xRatio - 0.5;
                    int x0 = Math.Max(0, (int)srcX);
                    int x1 = Math.Min(srcW - 1, x0 + 1);
                    double fx = srcX - x0;
                    if (fx < 0) fx = 0;

                    for (int c = 0; c < 3; c++)
                    {
                        double v00 = rgb[(y0 * srcW + x0) * 3 + c];
                        double v01 = rgb[(y0 * srcW + x1) * 3 + c];
                        double v10 = rgb[(y1 * srcW + x0) * 3 + c];
                        double v11 = rgb[(y1 * srcW + x1) * 3 + c];

                        double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                                   v10 * (1 - fx) * fy + v11 * fx * fy;
                        result[(dy * dstW + dx) * 3 + c] = (byte)Math.Clamp(v + 0.5, 0, 255);
                    }
                }
            });

            return result;
        }
    }
}
