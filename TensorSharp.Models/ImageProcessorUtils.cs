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
using System.IO;
using System.Threading.Tasks;
using StbImageSharp;
using TensorSharp.Models.Media;

namespace TensorSharp.Models
{
    internal static class ImageProcessorUtils
    {
        /// <summary>
        /// Decode an image file to RGBA8 for the chat vision processors.
        ///
        /// <para>PNG and JPEG are decoded here, in managed code, exactly as they always were —
        /// in particular WITHOUT applying EXIF orientation, since every vision-model reference
        /// (llama.cpp's stb_image path) ignores it too. Everything else (HEIC/HEIF from an
        /// iPhone, GIF, WebP, BMP, ...) is the platform image provider's business:
        /// Magick.NET on desktop, ImageIO on iOS, via <see cref="MediaCodecs.Image"/>.</para>
        /// </summary>
        internal static byte[] DecodeImageToRGBA(byte[] fileBytes, out int width, out int height)
        {
            if (ImageFormatSniffer.IsPng(fileBytes))
                return PngCodec.Decode(fileBytes, out width, out height);

            if (IsJpeg(fileBytes))
                return DecodeJPEG(fileBytes, out width, out height);

            return MediaCodecs.Image.DecodeRgba(fileBytes, out width, out height);
        }

        internal static (int width, int height) ReadImageDimensions(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);

            if (ImageFormatSniffer.IsPng(fileBytes))
                return PngCodec.ReadDimensions(fileBytes);

            if (IsJpeg(fileBytes))
            {
                DecodeJPEG(fileBytes, out int width, out int height);
                return (width, height);
            }

            return MediaCodecs.Image.ReadDimensions(fileBytes);
        }

        private static bool IsJpeg(byte[] fileBytes) =>
            fileBytes.Length >= 2 &&
            fileBytes[0] == 0xFF &&
            fileBytes[1] == 0xD8;

        private static byte[] DecodeJPEG(byte[] data, out int width, out int height)
        {
            try
            {
                ImageResult decoded = ImageResult.FromMemory(data, ColorComponents.RedGreenBlueAlpha);
                width = decoded.Width;
                height = decoded.Height;
                return decoded.Data;
            }
            catch (Exception ex)
            {
                throw new InvalidDataException("Failed to decode JPEG image.", ex);
            }
        }

        internal static byte[] CompositeOverWhite(byte[] rgba, int width, int height)
        {
            byte[] result = new byte[width * height * 4];
            Parallel.For(0, height, y =>
            {
                int srcRow = y * width * 4;
                for (int x = 0; x < width; x++)
                {
                    int pixBase = srcRow + x * 4;
                    int a = rgba[pixBase + 3];
                    if (a == 255)
                    {
                        result[pixBase] = rgba[pixBase];
                        result[pixBase + 1] = rgba[pixBase + 1];
                        result[pixBase + 2] = rgba[pixBase + 2];
                    }
                    else
                    {
                        float alpha = a / 255f;
                        result[pixBase] = (byte)(rgba[pixBase] * alpha + 255 * (1 - alpha));
                        result[pixBase + 1] = (byte)(rgba[pixBase + 1] * alpha + 255 * (1 - alpha));
                        result[pixBase + 2] = (byte)(rgba[pixBase + 2] * alpha + 255 * (1 - alpha));
                    }

                    result[pixBase + 3] = 255;
                }
            });
            return result;
        }

        internal static byte[] BilinearResize(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
        {
            byte[] result = new byte[dstW * dstH * 4];
            double xRatio = (double)srcW / dstW;
            double yRatio = (double)srcH / dstH;

            Parallel.For(0, dstH, dy =>
            {
                double srcY = (dy + 0.5) * yRatio - 0.5;
                int y0 = Math.Max(0, (int)srcY);
                int y1 = Math.Min(srcH - 1, y0 + 1);
                double fy = srcY - y0;

                for (int dx = 0; dx < dstW; dx++)
                {
                    double srcX = (dx + 0.5) * xRatio - 0.5;
                    int x0 = Math.Max(0, (int)srcX);
                    int x1 = Math.Min(srcW - 1, x0 + 1);
                    double fx = srcX - x0;

                    for (int c = 0; c < 3; c++)
                    {
                        double v00 = rgba[(y0 * srcW + x0) * 4 + c];
                        double v01 = rgba[(y0 * srcW + x1) * 4 + c];
                        double v10 = rgba[(y1 * srcW + x0) * 4 + c];
                        double v11 = rgba[(y1 * srcW + x1) * 4 + c];

                        double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                                   v10 * (1 - fx) * fy + v11 * fx * fy;
                        result[(dy * dstW + dx) * 4 + c] = (byte)Math.Clamp(v + 0.5, 0, 255);
                    }
                    result[(dy * dstW + dx) * 4 + 3] = 255;
                }
            });

            return result;
        }

        internal static float[] ResizeRgbaToChannelFirstNormalized(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
        {
            int pixels = dstW * dstH;
            float[] result = new float[3 * pixels];
            double xRatio = (double)srcW / dstW;
            double yRatio = (double)srcH / dstH;

            Parallel.For(0, dstH, dy =>
            {
                double srcY = (dy + 0.5) * yRatio - 0.5;
                int y0 = Math.Max(0, (int)srcY);
                int y1 = Math.Min(srcH - 1, y0 + 1);
                double fy = srcY - y0;

                for (int dx = 0; dx < dstW; dx++)
                {
                    double srcX = (dx + 0.5) * xRatio - 0.5;
                    int x0 = Math.Max(0, (int)srcX);
                    int x1 = Math.Min(srcW - 1, x0 + 1);
                    double fx = srcX - x0;

                    int dstIdx = dy * dstW + dx;
                    result[dstIdx] = BilinearSampleNormalized(rgba, srcW, x0, y0, x1, y1, fx, fy, 0);
                    result[pixels + dstIdx] = BilinearSampleNormalized(rgba, srcW, x0, y0, x1, y1, fx, fy, 1);
                    result[2 * pixels + dstIdx] = BilinearSampleNormalized(rgba, srcW, x0, y0, x1, y1, fx, fy, 2);
                }
            });

            return result;
        }

        private static float BilinearSampleNormalized(byte[] rgba, int srcW, int x0, int y0, int x1, int y1,
            double fx, double fy, int channel)
        {
            float v00 = CompositeChannelToNormalized(rgba, (y0 * srcW + x0) * 4, channel);
            float v01 = CompositeChannelToNormalized(rgba, (y0 * srcW + x1) * 4, channel);
            float v10 = CompositeChannelToNormalized(rgba, (y1 * srcW + x0) * 4, channel);
            float v11 = CompositeChannelToNormalized(rgba, (y1 * srcW + x1) * 4, channel);

            double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                       v10 * (1 - fx) * fy + v11 * fx * fy;
            return Math.Clamp((float)v, -1f, 1f);
        }

        private static float CompositeChannelToNormalized(byte[] rgba, int pixelBase, int channel)
        {
            float alpha = rgba[pixelBase + 3] / 255f;
            float composited = rgba[pixelBase + channel] * alpha + 255f * (1f - alpha);
            return composited / 255f * 2f - 1f;
        }

        /// <summary>
        /// Resize (bilinear, composite over white) and normalize to channel-first
        /// [C, H, W] using explicit per-channel mean/std: (pixel/255 - mean) / std.
        /// Used by models whose mmproj declares its own image_mean / image_std
        /// (e.g. the Gemma 4 unified vision embedder with mean=0, std=1 -> [0,1]).
        /// </summary>
        internal static float[] ResizeRgbaToChannelFirstNormalized(
            byte[] rgba, int srcW, int srcH, int dstW, int dstH, float[] mean, float[] std)
        {
            int pixels = dstW * dstH;
            float[] result = new float[3 * pixels];
            double xRatio = (double)srcW / dstW;
            double yRatio = (double)srcH / dstH;

            Parallel.For(0, dstH, dy =>
            {
                double srcY = (dy + 0.5) * yRatio - 0.5;
                int y0 = Math.Max(0, (int)srcY);
                int y1 = Math.Min(srcH - 1, y0 + 1);
                double fy = srcY - y0;

                for (int dx = 0; dx < dstW; dx++)
                {
                    double srcX = (dx + 0.5) * xRatio - 0.5;
                    int x0 = Math.Max(0, (int)srcX);
                    int x1 = Math.Min(srcW - 1, x0 + 1);
                    double fx = srcX - x0;

                    int dstIdx = dy * dstW + dx;
                    for (int c = 0; c < 3; c++)
                    {
                        float v00 = CompositeChannel01(rgba, (y0 * srcW + x0) * 4, c);
                        float v01 = CompositeChannel01(rgba, (y0 * srcW + x1) * 4, c);
                        float v10 = CompositeChannel01(rgba, (y1 * srcW + x0) * 4, c);
                        float v11 = CompositeChannel01(rgba, (y1 * srcW + x1) * 4, c);

                        double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                                   v10 * (1 - fx) * fy + v11 * fx * fy;
                        result[c * pixels + dstIdx] = (float)((v - mean[c]) / std[c]);
                    }
                }
            });

            return result;
        }

        // Composite an RGBA channel over a white background and rescale to [0, 1].
        private static float CompositeChannel01(byte[] rgba, int pixelBase, int channel)
        {
            float alpha = rgba[pixelBase + 3] / 255f;
            float composited = rgba[pixelBase + channel] * alpha + 255f * (1f - alpha);
            return composited / 255f;
        }

        // ---- bicubic resampling ------------------------------------------------------------
        //
        // The Gemma-4 family's processor declares resample=3 (BICUBIC), and its reference
        // implementation resizes with an ANTIALIASED bicubic: when downscaling, the filter's
        // support widens by 1/scale so every source pixel contributes. Point-sampled bicubic
        // (the naive "read 4x4 neighbours" form) is NOT the same operation and aliases badly on
        // a large downscale, which is the usual case for a photo going to a ~960x624 canvas.
        //
        // This is the PIL/torchvision resampling formulation: a separable two-pass filter with a
        // per-output-pixel footprint and normalized weights, Catmull-Rom-style cubic with a=-0.5.

        private const double BicubicSupport = 2.0;

        private static double CubicFilter(double x)
        {
            // PIL's bicubic kernel (a = -0.5).
            const double a = -0.5;
            x = Math.Abs(x);
            if (x < 1.0)
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
            if (x < 2.0)
                return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
            return 0.0;
        }

        /// <summary>
        /// Precompute, for one axis, each output pixel's source window and normalized weights.
        /// </summary>
        private static void BuildCubicWeights(int srcSize, int dstSize,
            out int[] bounds, out double[] weights, out int windowSize, bool byteDomain = false)
        {
            double scale = (double)srcSize / dstSize;
            double filterScale = Math.Max(1.0, scale);
            double support = BicubicSupport * filterScale;
            windowSize = (int)Math.Ceiling(support) * 2 + 1;

            bounds = new int[dstSize * 2];
            weights = new double[(long)dstSize * windowSize <= int.MaxValue ? dstSize * windowSize : 0];
            if (weights.Length == 0)
                throw new InvalidOperationException($"resize window too large for {srcSize}->{dstSize}.");

            for (int i = 0; i < dstSize; i++)
            {
                double center = (i + 0.5) * scale;
                int xmin = (int)Math.Max(0, Math.Floor(center - support + 0.5));
                int xmax = (int)Math.Min(srcSize, byteDomain
                    ? Math.Truncate(center + support + 0.5) : Math.Ceiling(center + support + 0.5));
                int count = xmax - xmin;

                double sum = 0;
                int wBase = i * windowSize;
                for (int k = 0; k < count; k++)
                {
                    // The byte path follows the Pillow/mtmd coefficient order
                    // before 22-bit quantization. Preserve the existing float
                    // Gemma path's arithmetic and footprint unchanged.
                    double w = CubicFilter(byteDomain
                        ? (k + xmin - center + 0.5) * (1.0 / filterScale)
                        : (xmin + k + 0.5 - center) / filterScale);
                    weights[wBase + k] = w;
                    sum += w;
                }
                if (sum != 0)
                    for (int k = 0; k < count; k++)
                        weights[wBase + k] /= sum;

                bounds[i * 2] = xmin;
                bounds[i * 2 + 1] = count;
            }
        }

        /// <summary>Known temporary array payload for the Qwen byte bicubic
        /// path, excluding the caller's RGBA and returned CHW arrays. This is
        /// checked before allocating coefficient tables or horizontal pixels.
        /// Both temporary doubles and quantized coefficients remain covered;
        /// no collection between the two passes is assumed.</summary>
        internal static long GetByteBicubicResizeScratchBytes(int srcW, int srcH, int dstW, int dstH)
        {
            if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0)
                throw new ArgumentOutOfRangeException(nameof(srcW), "Resize dimensions must be positive.");
            checked
            {
                long sourcePixels = (long)srcW * srcH, outputPixels = (long)dstW * dstH;
                if (sourcePixels > Array.MaxLength / 4 || outputPixels > Array.MaxLength / 3)
                    throw new NotSupportedException("Qwen byte resize exceeds the supported managed array range.");
                if (srcW == dstW && srcH == dstH) return 0;
                long horizontal = 3L * srcH * dstW;
                long Window(int source, int target) => checked((long)Math.Ceiling(BicubicSupport * Math.Max(1.0, (double)source / target)) * 2 + 1);
                long xWindow = Window(srcW, dstW), yWindow = Window(srcH, dstH);
                long xCoefficients = dstW * xWindow, yCoefficients = dstH * yWindow;
                if (horizontal > Array.MaxLength || xWindow > int.MaxValue || yWindow > int.MaxValue
                    || xCoefficients > Array.MaxLength || yCoefficients > Array.MaxLength
                    || 2L * dstW > Array.MaxLength || 2L * dstH > Array.MaxLength)
                    throw new NotSupportedException("Qwen bicubic scratch exceeds the supported managed array range.");
                return horizontal + (sizeof(double) + sizeof(int)) * (xCoefficients + yCoefficients)
                    + 2L * sizeof(int) * (dstW + (long)dstH);
            }
        }

        private const int ByteBicubicPrecisionBits = 22;
        private static int[] QuantizeByteCubicWeights(double[] weights)
        {
            var result = new int[weights.Length];
            const double scale = 1 << ByteBicubicPrecisionBits;
            for (int i = 0; i < weights.Length; i++)
                result[i] = checked((int)Math.Round(weights[i] * scale, MidpointRounding.AwayFromZero));
            return result;
        }

        private static byte CompositeChannelToByte(byte[] rgba, int pixelBase, int channel)
        {
            int alpha = rgba[pixelBase + 3];
            return (byte)((rgba[pixelBase + channel] * alpha + 255 * (255 - alpha) + 127) / 255);
        }

        private static byte RoundedByteCubic(long accumulator) => (byte)Math.Clamp(
            accumulator >> ByteBicubicPrecisionBits, 0L, 255L);

        /// <summary>Qwen's antialiased byte-domain bicubic preprocessing. Each
        /// separable pass rounds/clamps to uint8 before the next pass, using
        /// 22-bit coefficients as Pillow/torchvision uint8/unchanged mtmd do.
        /// The existing Gemma float-intermediate bicubic path is independent.</summary>
        internal static float[] ResizeRgbaToChannelFirstBicubicByte(
            byte[] rgba, int srcW, int srcH, int dstW, int dstH)
        {
            ArgumentNullException.ThrowIfNull(rgba);
            GetByteBicubicResizeScratchBytes(srcW, srcH, dstW, dstH);
            if (rgba.LongLength != checked(4L * srcW * srcH))
                throw new ArgumentException("RGBA payload differs from the source dimensions.", nameof(rgba));
            int outPixels = checked(dstW * dstH);
            var result = new float[checked(3 * outPixels)];
            if (srcW == dstW && srcH == dstH)
            {
                Parallel.For(0, srcH, y =>
                {
                    for (int x = 0; x < srcW; x++)
                    {
                        int pixel = y * srcW + x;
                        for (int c = 0; c < 3; c++)
                            result[c * outPixels + pixel] = CompositeChannelToByte(rgba, pixel * 4, c) / 255f * 2f - 1f;
                    }
                });
                return result;
            }

            BuildCubicWeights(srcW, dstW, out int[] xBounds, out double[] xFloating, out int xWindow, byteDomain: true);
            BuildCubicWeights(srcH, dstH, out int[] yBounds, out double[] yFloating, out int yWindow, byteDomain: true);
            int[] xWeights = QuantizeByteCubicWeights(xFloating), yWeights = QuantizeByteCubicWeights(yFloating);
            int horizontalPixels = checked(srcH * dstW);
            var horizontal = new byte[checked(3 * horizontalPixels)];
            const long rounding = 1L << (ByteBicubicPrecisionBits - 1);
            Parallel.For(0, srcH, sy =>
            {
                for (int dx = 0; dx < dstW; dx++)
                {
                    int xmin = xBounds[dx * 2], count = xBounds[dx * 2 + 1], weightBase = dx * xWindow;
                    long red = rounding, green = rounding, blue = rounding;
                    for (int k = 0; k < count; k++)
                    {
                        int pixelBase = (sy * srcW + xmin + k) * 4, weight = xWeights[weightBase + k];
                        red += (long)weight * CompositeChannelToByte(rgba, pixelBase, 0);
                        green += (long)weight * CompositeChannelToByte(rgba, pixelBase, 1);
                        blue += (long)weight * CompositeChannelToByte(rgba, pixelBase, 2);
                    }
                    int pixel = sy * dstW + dx;
                    horizontal[pixel] = RoundedByteCubic(red);
                    horizontal[horizontalPixels + pixel] = RoundedByteCubic(green);
                    horizontal[2 * horizontalPixels + pixel] = RoundedByteCubic(blue);
                }
            });
            Parallel.For(0, dstH, dy =>
            {
                int ymin = yBounds[dy * 2], count = yBounds[dy * 2 + 1], weightBase = dy * yWindow;
                for (int dx = 0; dx < dstW; dx++)
                    for (int c = 0; c < 3; c++)
                    {
                        long accumulator = rounding;
                        int plane = c * horizontalPixels;
                        for (int k = 0; k < count; k++)
                            accumulator += (long)yWeights[weightBase + k] * horizontal[plane + (ymin + k) * dstW + dx];
                        byte pixel = RoundedByteCubic(accumulator);
                        result[c * outPixels + dy * dstW + dx] = pixel / 255f * 2f - 1f;
                    }
            });
            return result;
        }

        /// <summary>
        /// Antialiased bicubic resize of an RGBA buffer, compositing over white, into channel-first
        /// [C, H, W] normalized as <c>(pixel/255 - mean) / std</c>.
        ///
        /// The image is stretched to fill exactly <paramref name="dstW"/> x <paramref name="dstH"/>.
        /// It is deliberately NOT letterboxed: the Gemma-4 reference processor derives the target
        /// canvas by flooring each side independently to a multiple of patch*pooling, which already
        /// approximately preserves the aspect ratio, and then resizes into it. Padding to preserve
        /// the ratio exactly would inject black bars that become real, unmasked, pooled patches and
        /// shift every patch's learned position embedding.
        /// </summary>
        internal static float[] ResizeRgbaToChannelFirstBicubic(
            byte[] rgba, int srcW, int srcH, int dstW, int dstH, float[] mean, float[] std)
        {
            BuildCubicWeights(srcW, dstW, out int[] xBounds, out double[] xWeights, out int xWindow);
            BuildCubicWeights(srcH, dstH, out int[] yBounds, out double[] yWeights, out int yWindow);

            // Pass 1: horizontal, into a [3][srcH][dstW] intermediate held at full precision.
            var horizontal = new float[3 * srcH * dstW];
            Parallel.For(0, srcH, sy =>
            {
                long rowBase = (long)sy * srcW;
                for (int dx = 0; dx < dstW; dx++)
                {
                    int xmin = xBounds[dx * 2], count = xBounds[dx * 2 + 1];
                    int wBase = dx * xWindow;
                    double r = 0, g = 0, b = 0;
                    for (int k = 0; k < count; k++)
                    {
                        double w = xWeights[wBase + k];
                        int pixelBase = (int)((rowBase + xmin + k) * 4);
                        r += w * CompositeChannel01(rgba, pixelBase, 0);
                        g += w * CompositeChannel01(rgba, pixelBase, 1);
                        b += w * CompositeChannel01(rgba, pixelBase, 2);
                    }
                    int idx = sy * dstW + dx;
                    horizontal[idx] = (float)r;
                    horizontal[srcH * dstW + idx] = (float)g;
                    horizontal[2 * srcH * dstW + idx] = (float)b;
                }
            });

            // Pass 2: vertical, straight into the channel-first output.
            int outPixels = dstW * dstH;
            var result = new float[3 * outPixels];
            Parallel.For(0, dstH, dy =>
            {
                int ymin = yBounds[dy * 2], count = yBounds[dy * 2 + 1];
                int wBase = dy * yWindow;
                for (int dx = 0; dx < dstW; dx++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        int plane = c * srcH * dstW;
                        double acc = 0;
                        for (int k = 0; k < count; k++)
                            acc += yWeights[wBase + k] * horizontal[plane + (ymin + k) * dstW + dx];
                        result[c * outPixels + dy * dstW + dx] = (float)((acc - mean[c]) / std[c]);
                    }
                }
            });

            return result;
        }
    }
}
