// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using TensorSharp.Models.Media;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Image load/save/resize helpers for the Qwen-Image-2.1 pipeline (and the Wan and
    /// MiniMax-H3 pipelines, which share them), backed by the platform image codec on
    /// <see cref="MediaCodecs.Image"/> — Magick.NET on desktop, so the results the
    /// pipelines were validated against are unchanged.
    ///
    /// The canonical in-memory representation here is interleaved <b>HWC RGB</b>
    /// (row-major: pixel (y,x) channel c at index <c>(y*W + x)*3 + c</c>) with
    /// float values in <c>[0,1]</c>. The VAE/diffusion math (which works in
    /// channel-planar [-1,1]) is applied by the pipeline, not here.
    /// </summary>
    public sealed class RgbImage
    {
        public int Width { get; }
        public int Height { get; }
        /// <summary>Interleaved HWC RGB, length <c>Width*Height*3</c>, values in [0,1].</summary>
        public float[] Pixels { get; }

        /// <summary>Optional straight alpha, length Width*Height, values in [0,1]. Null means opaque.</summary>
        public float[] Alpha { get; }

        public RgbImage(int width, int height, float[] pixels) : this(width, height, pixels, null) { }

        public RgbImage(int width, int height, float[] pixels, float[] alpha)
        {
            ArgumentNullException.ThrowIfNull(pixels);
            if (pixels.Length != (long)width * height * 3)
                throw new ArgumentException($"pixel buffer {pixels.Length} != {width}x{height}x3");
            if (alpha != null && alpha.LongLength != (long)width * height)
                throw new ArgumentException($"alpha buffer {alpha.Length} != {width}x{height}", nameof(alpha));
            Width = width; Height = height; Pixels = pixels; Alpha = alpha;
        }

        /// <summary>Channel-planar CHW copy (channel c at <c>c*W*H + y*W + x</c>), values in [0,1].</summary>
        public float[] ToPlanarChw()
        {
            int w = Width, h = Height, hw = w * h;
            var dst = new float[3 * hw];
            var src = Pixels;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int p = (y * w + x) * 3;
                    int q = y * w + x;
                    dst[q] = src[p];
                    dst[hw + q] = src[p + 1];
                    dst[2 * hw + q] = src[p + 2];
                }
            return dst;
        }

        public static RgbImage FromPlanarChw(int width, int height, float[] chw)
        {
            int hw = width * height;
            var px = new float[3 * hw];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int q = y * width + x;
                    int p = q * 3;
                    px[p] = chw[q];
                    px[p + 1] = chw[hw + q];
                    px[p + 2] = chw[2 * hw + q];
                }
            return new RgbImage(width, height, px);
        }
    }

    public static class ImageIO
    {
        public static RgbImage Load(string path) => Load(path, preserveAlpha: false);
        public static RgbImage Load(string path, bool preserveAlpha) => Decode(File.ReadAllBytes(path), preserveAlpha);

        /// <summary>Decode any supported image, upright (EXIF orientation applied: phone
        /// photos are stored rotated with an orientation tag, and the reference pipelines'
        /// loaders all honor it). Alpha, if any, is dropped rather than composited — the
        /// generative pipelines take the colour values as stored.</summary>
        public static RgbImage Decode(byte[] data) => Decode(data, preserveAlpha: false);

        /// <summary>Decode RGB with optional straight alpha for models such as Qwen-Image-2.1.</summary>
        public static RgbImage Decode(byte[] data, bool preserveAlpha)
        {
            byte[] rgba = MediaCodecs.Image.DecodeRgba(data, out int w, out int h);
            var px = new float[(long)w * h * 3];
            for (long i = 0, p = 0; i < px.Length; i += 3, p += 4)
            {
                px[i] = rgba[p] / 255f;
                px[i + 1] = rgba[p + 1] / 255f;
                px[i + 2] = rgba[p + 2] / 255f;
            }
            float[] alpha = null;
            if (preserveAlpha)
            {
                alpha = new float[(long)w * h];
                for (long i = 0; i < alpha.Length; i++) alpha[i] = rgba[i * 4 + 3] / 255f;
            }
            return new RgbImage(w, h, px, alpha);
        }

        public static byte[] EncodePng(RgbImage img) =>
            MediaCodecs.Image.EncodePng(img.Alpha == null ? ToRgb8(img) : ToRgba8(img), img.Width, img.Height, img.Alpha == null ? 3 : 4);

        public static void SavePng(string path, RgbImage img) => File.WriteAllBytes(path, EncodePng(img));

        /// <summary>
        /// Resize so the area is approximately <paramref name="targetArea"/> pixels while
        /// preserving aspect ratio, snapping both dimensions to a multiple of
        /// <paramref name="multiple"/>.
        /// </summary>
        public static RgbImage ResizeToArea(RgbImage img, long targetArea, int multiple = 16)
        {
            double ar = (double)img.Width / img.Height;
            int h = (int)Math.Round(Math.Sqrt(targetArea / ar));
            int w = (int)Math.Round(h * ar);
            w = Math.Max(multiple, (w / multiple) * multiple);
            h = Math.Max(multiple, (h / multiple) * multiple);
            return Resize(img, w, h);
        }

        /// <summary>Resize to exactly w x h WITHOUT distorting: the source is centre-cropped
        /// to the target aspect first, then scaled.
        ///
        /// <para>Stretching is the wrong trade for anything a generative model conditions
        /// on. A 4:3 photo squeezed into 5:3 makes every face 25% narrower, and the model
        /// faithfully reproduces the distortion for the whole clip — which reads as
        /// "blurry" and "the person looks wrong" rather than as a geometry error. Losing
        /// a strip off the edges is the cheaper mistake.</para></summary>
        public static RgbImage ResizeCover(RgbImage img, int w, int h)
        {
            if (w == img.Width && h == img.Height) return img;
            double target = w / (double)h, source = img.Width / (double)img.Height;
            int cropW = img.Width, cropH = img.Height;
            if (source > target) cropW = Math.Max(1, (int)Math.Round(img.Height * target));
            else if (source < target) cropH = Math.Max(1, (int)Math.Round(img.Width / target));

            RgbImage cropped = img;
            if (cropW != img.Width || cropH != img.Height)
            {
                int x0 = (img.Width - cropW) / 2, y0 = (img.Height - cropH) / 2;
                var px = new float[(long)cropW * cropH * 3];
                for (int y = 0; y < cropH; y++)
                    Array.Copy(img.Pixels, ((long)(y0 + y) * img.Width + x0) * 3,
                               px, (long)y * cropW * 3, (long)cropW * 3);
                float[] alpha = null;
                if (img.Alpha != null)
                {
                    alpha = new float[(long)cropW * cropH];
                    for (int y = 0; y < cropH; y++)
                        Array.Copy(img.Alpha, (long)(y0 + y) * img.Width + x0, alpha, (long)y * cropW, cropW);
                }
                cropped = new RgbImage(cropW, cropH, px, alpha);
            }
            return Resize(cropped, w, h);
        }

        /// <summary>Lanczos resize to exactly w x h, aspect ratio ignored (the caller has
        /// already chosen the geometry). The filter is the provider's: ImageMagick's Lanczos
        /// on desktop, which the pipelines' parity fixtures were produced with.</summary>
        public static RgbImage Resize(RgbImage img, int w, int h)
        {
            if (w == img.Width && h == img.Height) return img;
            byte[] rgb = MediaCodecs.Image.ResizeRgb8(ToRgb8(img), img.Width, img.Height, w, h, ResizeFilter.Lanczos);
            if (rgb.Length != (long)w * h * 3)
                throw new InvalidDataException($"image codec returned {rgb.Length} bytes for a {w}x{h} resize");
            var px = new float[(long)w * h * 3];
            for (long i = 0; i < px.Length; i++)
                px[i] = rgb[i] / 255f;
            float[] alpha = null;
            if (img.Alpha != null)
            {
                // Use the same codec, filter and geometry for the independent straight-alpha
                // plane. Providers expose RGB resampling, so replicate alpha into three channels.
                var alphaRgb = new byte[(long)img.Width * img.Height * 3];
                for (long i = 0; i < img.Alpha.Length; i++)
                    alphaRgb[i * 3] = alphaRgb[i * 3 + 1] = alphaRgb[i * 3 + 2] = Quantize(img.Alpha[i]);
                var resized = MediaCodecs.Image.ResizeRgb8(alphaRgb, img.Width, img.Height, w, h, ResizeFilter.Lanczos);
                if (resized.Length != (long)w * h * 3)
                    throw new InvalidDataException($"image codec returned {resized.Length} bytes for a {w}x{h} alpha resize");
                alpha = new float[(long)w * h];
                for (long i = 0; i < alpha.Length; i++) alpha[i] = resized[i * 3] / 255f;
            }
            return new RgbImage(w, h, px, alpha);
        }

        private static byte[] ToRgba8(RgbImage img)
        {
            var bytes = new byte[(long)img.Width * img.Height * 4];
            for (long i = 0; i < img.Alpha.Length; i++)
            {
                bytes[i * 4] = Quantize(img.Pixels[i * 3]);
                bytes[i * 4 + 1] = Quantize(img.Pixels[i * 3 + 1]);
                bytes[i * 4 + 2] = Quantize(img.Pixels[i * 3 + 2]);
                bytes[i * 4 + 3] = Quantize(img.Alpha[i]);
            }
            return bytes;
        }

        private static byte Quantize(float value) => (byte)Math.Clamp(value * 255f + 0.5f, 0f, 255f);

        // float [0,1] -> u8 with round-half-up, the quantisation every codec call shares.
        internal static byte[] ToRgb8(RgbImage img)
        {
            var bytes = new byte[(long)img.Width * img.Height * 3];
            var src = img.Pixels;
            for (long i = 0; i < bytes.Length; i++)
            {
                float v = src[i] * 255f + 0.5f;
                bytes[i] = (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
            return bytes;
        }
    }
}
