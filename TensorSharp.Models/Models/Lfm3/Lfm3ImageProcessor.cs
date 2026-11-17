// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>An 8-bit RGB picture (RGB24, rows top to bottom, no padding): a decoded image or video frame.</summary>
    public sealed class Lfm3RgbImage
    {
        public Lfm3RgbImage(int width, int height, byte[] rgb)
        {
            if (width < 1 || height < 1) throw new ArgumentOutOfRangeException(nameof(width), $"a {width} x {height} picture");
            ArgumentNullException.ThrowIfNull(rgb);
            if (rgb.LongLength != (long)width * height * 3)
                throw new ArgumentException($"{rgb.LongLength} bytes for a {width} x {height} RGB24 picture", nameof(rgb));
            Width = width;
            Height = height;
            Rgb = rgb;
        }

        public int Width { get; }
        public int Height { get; }

        /// <summary>R, G, B per pixel, row-major.</summary>
        public byte[] Rgb { get; }
    }

    /// <summary>
    /// LFM3-VL / LFM3-Omni vision preprocessing, as llama.cpp's mtmd (<c>mtmd_image_preprocessor_lfm3vl</c>) and the HF
    /// processor (<c>Lfm3VlImageProcessor</c> / <c>Lfm3VlVideoProcessor</c>) define it:
    /// <list type="bullet">
    /// <item>RGB: alpha is dropped, not composited (mtmd's stb_image load asks for 3 channels; PIL's RGBA to RGB conversion
    /// drops it too);</item>
    /// <item>size: <see cref="SmartResize"/> (images: 32-aligned, 65,536..2,097,152 pixels, i.e. 64..2048 tokens) or
    /// <see cref="SmartResizeVideo"/> (one size for all frames of a clip under a 25,165,824-pixel budget);</item>
    /// <item>resampling: <see cref="Resize"/>, torchvision's antialiased bicubic on uint8 (a = -0.5, int16 weights, each
    /// pass rounded to uint8), a straight stretch to the target (no letterbox);</item>
    /// <item>values: (u8 / 255 - 0.5) / 0.5 in single precision (<see cref="Normalize"/>);</item>
    /// <item>patches: 16 x 16, rows in raster order (x fastest), each patch's 768 values in [channel][row][column] order,
    /// the order of the GGUF patch kernels <c>v.patch_embd.weight</c> / <c>.weight.1</c> (<see cref="Patchify"/>).</item>
    /// </list>
    /// Every step is integer-exact or the same single-precision arithmetic as llama.cpp's, so the pixels a frame gives are
    /// bit-identical to the ones mtmd feeds its graph (the test project checks this against dumps of llama.cpp).
    /// </summary>
    public static class Lfm3ImageProcessor
    {
        /// <summary>Patch edge (<c>clip.vision.patch_size</c>).</summary>
        public const int PatchSize = 16;

        /// <summary>Patches merged into one embedding row along each axis (<c>clip.vision.spatial_merge_size</c>).</summary>
        public const int MergeSize = 2;

        /// <summary>Image sides are multiples of this: <see cref="PatchSize"/> x <see cref="MergeSize"/>.</summary>
        public const int Factor = PatchSize * MergeSize;

        /// <summary>Frames one temporal patch merges (the two patch kernels).</summary>
        public const int TemporalPatchSize = 2;

        /// <summary>Values one patch carries: 3 channels x 16 x 16.</summary>
        public const int PatchValues = 3 * PatchSize * PatchSize;

        /// <summary>Default image pixel bounds (<c>clip.vision.image_min_pixels</c> / <c>image_max_pixels</c>): 64..2048 tokens.</summary>
        public const long ImageMinPixels = 65_536, ImageMaxPixels = 2_097_152;

        /// <summary>Video pixel bounds over frames x height x width (<c>Lfm3VlVideoProcessor.size</c>; mtmd
        /// <c>get_video_frame_size</c>): at most 12,288 tokens a clip.</summary>
        public const long VideoMinPixels = 4_096, VideoMaxPixels = 25_165_824;

        // ================================================================ size

        /// <summary>The size a still image is resized to (width, height): sides rounded half to even to multiples of
        /// <paramref name="factor"/>, scaled down (floor) when that exceeds <paramref name="maxPixels"/> and up (ceiling) when
        /// below <paramref name="minPixels"/>, at least <paramref name="factor"/> each; mtmd's <c>smart_resize</c> with one
        /// frame (no aspect-ratio limit: HF raises above 200:1, llama.cpp proceeds, and so does this).</summary>
        public static (int Width, int Height) SmartResize(int width, int height, long minPixels = ImageMinPixels,
            long maxPixels = ImageMaxPixels, int factor = Factor)
            => SmartResizeCore(width, height, factor, minPixels, maxPixels, frames: 1, temporalFactor: 1);

        /// <summary>The one size every frame of a <paramref name="frames"/>-frame clip is resized to (width, height): as
        /// <see cref="SmartResize"/> over frames x height x width with the frame count rounded half to even to pairs, the
        /// scale computed from the unrounded count, and sides below <paramref name="factor"/> first scaled up to it (mtmd's
        /// <c>get_video_frame_size</c> = HF <c>Lfm3VlVideoProcessor</c>'s <c>smart_resize</c>).</summary>
        public static (int Width, int Height) SmartResizeVideo(int frames, int width, int height, long minPixels = VideoMinPixels,
            long maxPixels = VideoMaxPixels, int factor = Factor)
        {
            if (frames < 1) throw new ArgumentOutOfRangeException(nameof(frames), $"a clip of {frames} frames");
            return SmartResizeCore(width, height, factor, minPixels, maxPixels, frames, TemporalPatchSize);
        }

        // mtmd_image_preprocessor_qwen3vl::smart_resize (mtmd-image.cpp), in double as transformers computes it.
        private static (int Width, int Height) SmartResizeCore(int width, int height, int factor, long minPixels, long maxPixels,
            int frames, int temporalFactor)
        {
            if (width < 1 || height < 1) throw new ArgumentOutOfRangeException(nameof(width), $"a {width} x {height} image");
            double h = height, w = width;
            if (temporalFactor > 1 && (h < factor || w < factor))
            {
                double scale = Math.Max(factor / h, factor / w);
                h = (int)(h * scale);
                w = (int)(w * scale);
            }
            long tBar = (long)Math.Round((double)frames / temporalFactor, MidpointRounding.ToEven) * temporalFactor;
            int hBar = (int)Math.Round(h / factor, MidpointRounding.ToEven) * factor;
            int wBar = (int)Math.Round(w / factor, MidpointRounding.ToEven) * factor;
            if (tBar * hBar * wBar > maxPixels)
            {
                double beta = Math.Sqrt(frames * h * w / maxPixels);
                hBar = (int)Math.Floor(h / beta / factor) * factor;
                wBar = (int)Math.Floor(w / beta / factor) * factor;
            }
            else if (tBar * hBar * wBar < minPixels)
            {
                double beta = Math.Sqrt(minPixels / (frames * h * w));
                hBar = (int)Math.Ceiling(h * beta / factor) * factor;
                wBar = (int)Math.Ceiling(w * beta / factor) * factor;
            }
            return (Math.Max(factor, wBar), Math.Max(factor, hBar));
        }

        /// <summary>Embedding rows a <paramref name="width"/> x <paramref name="height"/> preprocessed frame (or frame pair)
        /// gives: one per 32 x 32 block.</summary>
        public static int TokenCount(int width, int height) => width / Factor * (height / Factor);

        // ================================================================ decode

        /// <summary>Decode an image file to RGB24 (PNG and JPEG in managed code, without EXIF rotation, as llama.cpp's
        /// stb_image loads them; other formats through the platform codec). Alpha is dropped.</summary>
        public static Lfm3RgbImage LoadImage(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            byte[] rgba = ImageProcessorUtils.DecodeImageToRGBA(File.ReadAllBytes(path), out int width, out int height);
            return new Lfm3RgbImage(width, height, DropAlpha(rgba, width, height));
        }

        /// <summary>RGB24 from RGBA8: the color channels as they are, alpha dropped (not composited over a background).</summary>
        public static byte[] DropAlpha(byte[] rgba, int width, int height)
        {
            ArgumentNullException.ThrowIfNull(rgba);
            long pixels = (long)width * height;
            if (rgba.LongLength != pixels * 4) throw new ArgumentException($"{rgba.LongLength} bytes for a {width} x {height} RGBA picture");
            var rgb = new byte[pixels * 3];
            for (long i = 0; i < pixels; i++)
            {
                rgb[i * 3] = rgba[i * 4];
                rgb[i * 3 + 1] = rgba[i * 4 + 1];
                rgb[i * 3 + 2] = rgba[i * 4 + 2];
            }
            return rgb;
        }

        // ================================================================ resize

        private const int PrecisionBits = 32 - 8 - 2;
        private const double BicubicSupport = 2.0;

        /// <summary>
        /// torchvision's antialiased bicubic resize of a uint8 image (a = -0.5; the filter widens by the scale when shrinking),
        /// with its int16 weights and a uint8 rounding after each pass (horizontal, then vertical): llama.cpp's
        /// <c>resize_pillow</c> with <c>RESIZE_ALGO_FLAG_TORCH</c> (mtmd-image.cpp), integer for integer. A picture already
        /// at the target size is returned as it is.
        /// </summary>
        public static Lfm3RgbImage Resize(Lfm3RgbImage image, int width, int height)
        {
            ArgumentNullException.ThrowIfNull(image);
            if (width < 1 || height < 1 || width > 65536 || height > 65536)
                throw new ArgumentOutOfRangeException(nameof(width), $"resize target {width} x {height}");
            if (width == image.Width && height == image.Height) return image;
            byte[] pixels = image.Rgb;
            int w = image.Width;
            if (width != image.Width)
            {
                (int[] bounds, int[] weights, int k) = Weights(image.Width, width);
                pixels = Horizontal(pixels, image.Width, image.Height, width, bounds, weights, k);
                w = width;
            }
            if (height != image.Height)
            {
                (int[] bounds, int[] weights, int k) = Weights(image.Height, height);
                pixels = Vertical(pixels, w, height, bounds, weights, k);
            }
            return new Lfm3RgbImage(width, height, pixels);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double Cubic(double x)
        {
            const double a = -0.5;
            if (x < 0) x = -x;
            if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
            return 0.0;
        }

        /// <summary>Per output pixel: its first source pixel and count (<c>bounds</c>), and <c>k</c> fixed-point weights
        /// (22 fractional bits, rounded at torchvision's int16 precision).</summary>
        private static (int[] Bounds, int[] Weights, int K) Weights(int inSize, int outSize)
        {
            double scale = (double)inSize / outSize, filterScale = Math.Max(1.0, scale);
            double support = BicubicSupport * filterScale;
            int k = (int)Math.Ceiling(support) * 2 + 1;
            var pre = new double[(long)outSize * k];
            var bounds = new int[outSize * 2];
            double ss = 1.0 / filterScale;
            for (int xx = 0; xx < outSize; xx++)
            {
                double center = (xx + 0.5) * scale, ww = 0.0;
                int xmin = (int)(center - support + 0.5);
                if (xmin < 0) xmin = 0;
                int xmax = (int)(center + support + 0.5);
                if (xmax > inSize) xmax = inSize;
                xmax -= xmin;
                long row = (long)xx * k;
                for (int x = 0; x < xmax; x++)
                {
                    double v = Cubic((x + xmin - center + 0.5) * ss);
                    pre[row + x] = v;
                    ww += v;
                }
                if (ww != 0.0)
                    for (int x = 0; x < xmax; x++) pre[row + x] /= ww;
                bounds[xx * 2] = xmin;
                bounds[xx * 2 + 1] = xmax;
            }
            // torchvision rounds the weights to int16 with as many fractional bits as the largest weight allows.
            double max = 0;
            foreach (double v in pre) max = Math.Max(max, v);
            int precision;
            for (precision = 0; precision < PrecisionBits; precision++)
                if ((int)(0.5 + max * (1 << (precision + 1))) >= 1 << 15) break;
            double fxp = Math.ScaleB(1.0, precision);
            var weights = new int[pre.Length];
            for (long i = 0; i < pre.Length; i++)
                weights[i] = (int)(pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5)) * (1 << (PrecisionBits - precision));
            return (bounds, weights, k);
        }

        /// <summary>Whether a pass of <paramref name="taps"/> pixel taps is worth the thread pool: a dispatch costs
        /// milliseconds on a loaded machine, more than a small image's whole pass.</summary>
        private static bool Parallel(long taps) => taps >= 1L << 22 && Environment.ProcessorCount > 1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static byte Clip8(int v) => v < 0 ? (byte)0 : v > 255 ? (byte)255 : (byte)v;

        private static byte[] Horizontal(byte[] src, int inW, int inH, int outW, int[] bounds, int[] weights, int k)
        {
            var dst = new byte[(long)outW * inH * 3];
            if (Parallel(outW * (long)inH * k))
                System.Threading.Tasks.Parallel.For(0, inH, y => HorizontalRow(src, dst, inW, outW, bounds, weights, k, y));
            else
                for (int y = 0; y < inH; y++) HorizontalRow(src, dst, inW, outW, bounds, weights, k, y);
            return dst;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void HorizontalRow(byte[] src, byte[] dst, int inW, int outW, int[] bounds, int[] weights, int k, int y)
        {
            long srcRow = (long)y * inW * 3, dstRow = (long)y * outW * 3;
            for (int xx = 0; xx < outW; xx++)
            {
                int xmin = bounds[xx * 2], count = bounds[xx * 2 + 1], wBase = xx * k;
                int s0 = 1 << (PrecisionBits - 1), s1 = s0, s2 = s0;
                long p = srcRow + (long)xmin * 3;
                for (int x = 0; x < count; x++, p += 3)
                {
                    int w = weights[wBase + x];
                    s0 += src[p] * w;
                    s1 += src[p + 1] * w;
                    s2 += src[p + 2] * w;
                }
                dst[dstRow + xx * 3] = Clip8(s0 >> PrecisionBits);
                dst[dstRow + xx * 3 + 1] = Clip8(s1 >> PrecisionBits);
                dst[dstRow + xx * 3 + 2] = Clip8(s2 >> PrecisionBits);
            }
        }

        private static byte[] Vertical(byte[] src, int w, int outH, int[] bounds, int[] weights, int k)
        {
            int rowElems = w * 3;
            var dst = new byte[(long)rowElems * outH];
            if (Parallel((long)w * outH * k))
            {
                System.Threading.Tasks.Parallel.For(0, outH, () => new int[rowElems], (yy, _, acc) =>
                {
                    VerticalRow(src, dst, rowElems, bounds, weights, k, yy, acc);
                    return acc;
                }, _ => { });
            }
            else
            {
                var acc = new int[rowElems];
                for (int yy = 0; yy < outH; yy++) VerticalRow(src, dst, rowElems, bounds, weights, k, yy, acc);
            }
            return dst;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void VerticalRow(byte[] src, byte[] dst, int rowElems, int[] bounds, int[] weights, int k, int yy, int[] acc)
        {
            int ymin = bounds[yy * 2], count = bounds[yy * 2 + 1], wBase = yy * k;
            Array.Fill(acc, 1 << (PrecisionBits - 1));
            for (int y = 0; y < count; y++)
            {
                long row = (long)(ymin + y) * rowElems;
                int w = weights[wBase + y];
                for (int i = 0; i < rowElems; i++) acc[i] += src[row + i] * w;
            }
            long dstRow = (long)yy * rowElems;
            for (int i = 0; i < rowElems; i++) dst[dstRow + i] = Clip8(acc[i] >> PrecisionBits);
        }

        // ================================================================ values and patches

        /// <summary>(u8 / 255 - 0.5) / 0.5 per channel, planar [3][height][width] (the graph input mtmd builds, in the same
        /// single-precision steps).</summary>
        public static float[] Normalize(Lfm3RgbImage image)
        {
            ArgumentNullException.ThrowIfNull(image);
            int plane = image.Width * image.Height;
            var values = new float[3 * (long)plane];
            byte[] rgb = image.Rgb;
            for (int i = 0; i < plane; i++)
                for (int c = 0; c < 3; c++)
                {
                    float f = rgb[i * 3 + c] / 255.0f;
                    values[(long)c * plane + i] = (f - 0.5f) / 0.5f;
                }
            return values;
        }

        /// <summary>(u8 / 255 - 0.5) / 0.5 of each byte value, computed as <see cref="Normalize"/> computes it.</summary>
        private static readonly float[] s_values = BuildValues();

        private static float[] BuildValues()
        {
            var values = new float[256];
            for (int u = 0; u < 256; u++)
            {
                float f = u / 255.0f;
                values[u] = (f - 0.5f) / 0.5f;
            }
            return values;
        }

        /// <summary>
        /// <see cref="Patchify(float[], int, int, float[], int, int)"/> of <see cref="Normalize"/> in one pass over the bytes
        /// (the same values: each byte value's float is computed once with the same steps): the encoder's per-call input.
        /// </summary>
        public static float[] Patchify(Lfm3RgbImage image, float[] destination = null, int stride = PatchValues, int offset = 0)
        {
            ArgumentNullException.ThrowIfNull(image);
            int width = image.Width, height = image.Height;
            if (width % PatchSize != 0 || height % PatchSize != 0) throw new ArgumentException($"a {width} x {height} image is not whole {PatchSize}-pixel patches");
            int gw = width / PatchSize, gh = height / PatchSize, n = gw * gh;
            if (offset < 0 || offset + PatchValues > stride) throw new ArgumentOutOfRangeException(nameof(offset));
            destination ??= new float[(long)n * stride];
            if (destination.LongLength < (long)n * stride) throw new ArgumentException($"{destination.LongLength} values for {n} rows of {stride}");
            float[] dst = destination;
            byte[] rgb = image.Rgb;
            // A thread pool dispatch costs more than a small image's pass (milliseconds on a loaded machine).
            if (Parallel((long)n * PatchValues))
                System.Threading.Tasks.Parallel.For(0, gh, py => PatchRow(rgb, width, gw, py, dst, stride, offset));
            else
                for (int py = 0; py < gh; py++) PatchRow(rgb, width, gw, py, dst, stride, offset);
            return destination;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PatchRow(byte[] rgb, int width, int gw, int py, float[] dst, int stride, int offset)
        {
            float[] values = s_values;
            for (int px = 0; px < gw; px++)
            {
                long row = (long)(py * gw + px) * stride + offset;
                for (int ky = 0; ky < PatchSize; ky++)
                {
                    long src = ((long)(py * PatchSize + ky) * width + px * PatchSize) * 3;
                    long r = row + ky * PatchSize, g = r + PatchSize * PatchSize, b = g + PatchSize * PatchSize;
                    for (int kx = 0; kx < PatchSize; kx++, src += 3)
                    {
                        dst[r + kx] = values[rgb[src]];
                        dst[g + kx] = values[rgb[src + 1]];
                        dst[b + kx] = values[rgb[src + 2]];
                    }
                }
            }
        }

        /// <summary>
        /// The 16 x 16 patches of planar values [3][height][width] as rows of <see cref="PatchValues"/> (channel, row,
        /// column; column fastest), written at <paramref name="offset"/> of each <paramref name="stride"/>-wide row of
        /// <paramref name="destination"/> (null: a new [patches][768] array). Patches go in raster order, x fastest.
        /// </summary>
        public static float[] Patchify(float[] planar, int width, int height, float[] destination = null, int stride = PatchValues, int offset = 0)
        {
            ArgumentNullException.ThrowIfNull(planar);
            if (width % PatchSize != 0 || height % PatchSize != 0) throw new ArgumentException($"a {width} x {height} image is not whole {PatchSize}-pixel patches");
            if (planar.LongLength != 3L * width * height) throw new ArgumentException($"{planar.LongLength} values for a 3 x {height} x {width} image");
            int gw = width / PatchSize, gh = height / PatchSize, n = gw * gh;
            if (offset < 0 || offset + PatchValues > stride) throw new ArgumentOutOfRangeException(nameof(offset));
            destination ??= new float[(long)n * stride];
            if (destination.LongLength < (long)n * stride) throw new ArgumentException($"{destination.LongLength} values for {n} rows of {stride}");
            int plane = width * height;
            float[] dst = destination;
            System.Threading.Tasks.Parallel.For(0, gh, py =>
            {
                for (int px = 0; px < gw; px++)
                {
                    long row = (long)(py * gw + px) * stride + offset;
                    for (int c = 0; c < 3; c++)
                        for (int ky = 0; ky < PatchSize; ky++)
                        {
                            int src = c * plane + (py * PatchSize + ky) * width + px * PatchSize;
                            Array.Copy(planar, src, dst, row + (c * PatchSize + ky) * PatchSize, PatchSize);
                        }
                }
            });
            return destination;
        }
    }
}
