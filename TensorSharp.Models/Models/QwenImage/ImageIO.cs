// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using ImageMagick;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Image load/save/resize helpers for the Qwen-Image edit pipeline, backed by
    /// the Magick.NET dependency already used by <c>Gemma3.ImageProcessor</c>.
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

        public RgbImage(int width, int height, float[] pixels)
        {
            if (pixels.Length != (long)width * height * 3)
                throw new ArgumentException($"pixel buffer {pixels.Length} != {width}x{height}x3");
            Width = width; Height = height; Pixels = pixels;
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
        public static RgbImage Load(string path) => Decode(File.ReadAllBytes(path));

        public static RgbImage Decode(byte[] data)
        {
            using var image = new MagickImage(data);
            image.Alpha(AlphaOption.Off);
            if (image.ColorSpace != ColorSpace.sRGB)
                image.ColorSpace = ColorSpace.sRGB;
            int w = (int)image.Width, h = (int)image.Height;
            using var pixels = image.GetPixels();
            byte[] rgb = pixels.ToByteArray(0, 0, (uint)w, (uint)h, "RGB");
            var px = new float[(long)w * h * 3];
            for (long i = 0; i < px.Length; i++)
                px[i] = rgb[i] / 255f;
            return new RgbImage(w, h, px);
        }

        public static byte[] EncodePng(RgbImage img)
        {
            var bytes = new byte[(long)img.Width * img.Height * 3];
            var src = img.Pixels;
            for (long i = 0; i < bytes.Length; i++)
            {
                float v = src[i] * 255f + 0.5f;
                bytes[i] = (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
            var settings = new PixelReadSettings((uint)img.Width, (uint)img.Height,
                StorageType.Char, PixelMapping.RGB);
            using var image = new MagickImage(bytes, settings);
            image.Format = MagickFormat.Png;
            return image.ToByteArray();
        }

        public static void SavePng(string path, RgbImage img) => File.WriteAllBytes(path, EncodePng(img));

        /// <summary>
        /// Resize so the area is approximately <paramref name="targetArea"/> pixels while
        /// preserving aspect ratio, snapping both dimensions to a multiple of
        /// <paramref name="multiple"/> (Qwen-Image needs dims divisible by VAE-scale*patch = 16).
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

        public static RgbImage Resize(RgbImage img, int w, int h)
        {
            if (w == img.Width && h == img.Height) return img;
            var bytes = new byte[(long)img.Width * img.Height * 3];
            for (long i = 0; i < bytes.Length; i++)
            {
                float v = img.Pixels[i] * 255f + 0.5f;
                bytes[i] = (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
            var readSettings = new PixelReadSettings((uint)img.Width, (uint)img.Height,
                StorageType.Char, PixelMapping.RGB);
            using var image = new MagickImage(bytes, readSettings);
            image.FilterType = FilterType.Lanczos;
            image.Resize(new MagickGeometry((uint)w, (uint)h) { IgnoreAspectRatio = true });
            image.Alpha(AlphaOption.Off);
            using var pixels = image.GetPixels();
            byte[] rgb = pixels.ToByteArray(0, 0, (uint)w, (uint)h, "RGB");
            var px = new float[(long)w * h * 3];
            for (long i = 0; i < px.Length; i++)
                px[i] = rgb[i] / 255f;
            return new RgbImage(w, h, px);
        }
    }
}
