// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image's reference pictures (docs/design/diffusion-runtime.md section 4.2), as diffusers' MiniMax-H3-Image
// pipeline (main d5df05ed) conditions on them:
//
//   before_encoder.MiniMaxH3ImageEditResizeStep   bound_short_edge: the VAE copy to a short edge of 1024-2048, the vision
//                                                 copy to exactly 1024, sides rounded to 32, PIL Lanczos on 8 bits
//   encoders (Qwen3-VL processor)                 the vision copy's smart resize (multiples of 32, 65,536..16,777,216 px)
//   encoders.MiniMaxH3ImageEditVaeEncoderStep     the VAE copy cropped to whole 16-pixel cells; a posterior sample drawn off
//                                                 the request's generator, rounded to float16, cropped to whole 2x2 patches
//                                                 and normalized with the VAE's latents_mean / latents_std
//   before_denoise.MiniMaxH3ImageEditPrepareLatentsStep   scale_noise at t = 0.999 with noise drawn after the target's
//
// The pictures arrive upright (ImageIO.Decode applies the EXIF orientation, as ImageOps.exif_transpose does) and as RGB
// (alpha dropped, as Image.convert("RGB") drops it).
using System;
using TensorSharp.Models.Media;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>MiniMax-H3-Image's reference policy: the resized copies each consumer sees, a reference's latent rows from
    /// its posterior and their augmentation, each draw in diffusers' order.</summary>
    public sealed class MiniMaxH3ImageReferences : IReferencePolicy
    {
        /// <summary>Copies are rounded to multiples of this (the pipeline's canvas_multiple), and no side may be shorter.</summary>
        public const int Multiple = 32;

        /// <summary>The VAE copy's short-edge bounds (reference_min/max_short_edge).</summary>
        public const int VaeMinShortEdge = 1024, VaeMaxShortEdge = 2048;

        /// <summary>The vision copy's short edge (vision_min_short_edge = vision_max_short_edge).</summary>
        public const int VisionShortEdge = 1024;

        /// <summary>The Qwen3-VL image processor's pixel bounds (processor/preprocessor_config.json size).</summary>
        public const long VisionMinPixels = 65_536, VisionMaxPixels = 16_777_216;

        /// <summary>The level references are noised to and held at (reference_noise_aug).</summary>
        public const float NoiseAugmentation = 0.999f;

        private const int Downsample = MiniMaxH3ImageVaeCodec.Downsample, Patch = 2, Channels = MiniMaxH3ImageVaeCodec.LatentChannels;

        private readonly Func<(float[] Mean, float[] Std)> _latentStatistics;
        private readonly Func<LatentTensor, float[]> _toRows;

        /// <param name="latentStatistics">The VAE's latents_mean and latents_std (read when a posterior is normalized).</param>
        /// <param name="toRows">The family's latent-to-token-rows conversion.</param>
        internal MiniMaxH3ImageReferences(Func<(float[] Mean, float[] Std)> latentStatistics, Func<LatentTensor, float[]> toRows)
        {
            _latentStatistics = latentStatistics ?? throw new ArgumentNullException(nameof(latentStatistics));
            _toRows = toRows ?? throw new ArgumentNullException(nameof(toRows));
        }

        // ---- sizes (diffusers' functions, Python's round: half to even)

        /// <summary>
        /// diffusers' bound_short_edge: a short edge within [<paramref name="min"/>, <paramref name="max"/>] keeps the
        /// original size; otherwise the picture is scaled (keeping its ratio) so the short edge lands on the nearer bound, and
        /// both sides round to <paramref name="multiple"/>.
        /// </summary>
        public static (int Width, int Height) BoundShortEdge(int width, int height, int min, int max, int multiple = Multiple)
        {
            int shortEdge = Math.Min(width, height);
            if (min <= shortEdge && shortEdge <= max) return (width, height);
            double scale = (double)(shortEdge < min ? min : max) / shortEdge;
            return (Math.Max(multiple, RoundToMultiple(width * scale, multiple)),
                    Math.Max(multiple, RoundToMultiple(height * scale, multiple)));
        }

        /// <summary>
        /// The Qwen2-VL image processor's smart_resize at factor 32 (patch 16 x merge 2): sides rounded to multiples of 32,
        /// then scaled down (floor) or up (ceil) into the processor's pixel bounds, keeping the ratio.
        /// </summary>
        public static (int Width, int Height) SmartResize(int width, int height,
            long minPixels = VisionMinPixels, long maxPixels = VisionMaxPixels, int factor = Multiple)
        {
            if (Math.Max(width, height) / (double)Math.Min(width, height) > 200)
                throw new ArgumentException($"A reference's aspect ratio must be below 200, got {width}x{height}.");
            int hBar = RoundToMultiple(height, factor), wBar = RoundToMultiple(width, factor);
            if ((long)hBar * wBar > maxPixels)
            {
                double beta = Math.Sqrt((double)height * width / maxPixels);
                hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
                wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
            }
            else if ((long)hBar * wBar < minPixels)
            {
                double beta = Math.Sqrt(minPixels / ((double)height * width));
                hBar = (int)Math.Ceiling(height * beta / factor) * factor;
                wBar = (int)Math.Ceiling(width * beta / factor) * factor;
            }
            return (wBar, hBar);
        }

        /// <summary>Python's <c>round(value / multiple) * multiple</c> (round half to even).</summary>
        internal static int RoundToMultiple(double value, int multiple) =>
            (int)Math.Round(value / multiple, MidpointRounding.ToEven) * multiple;

        /// <summary>The size of reference <paramref name="index"/>'s VAE copy, before its crop to whole cells.</summary>
        public static (int Width, int Height) VaeCopySize(int width, int height) =>
            BoundShortEdge(width, height, VaeMinShortEdge, VaeMaxShortEdge);

        /// <summary>The vision tower's input size for a picture of <paramref name="width"/> x <paramref name="height"/>: the
        /// 1024 short edge, then the processor's smart resize.</summary>
        public static (int Width, int Height) VisionCopySize(int width, int height)
        {
            var (w, h) = BoundShortEdge(width, height, VisionShortEdge, VisionShortEdge);
            return SmartResize(w, h);
        }

        /// <summary>
        /// Pillow's <c>Image.resize</c> of the picture's 8-bit RGB (LANCZOS, or BICUBIC for the vision processor), as diffusers
        /// and ComfyUI resize references. The picture is quantized as every codec call is (round half up, which recovers a
        /// decoded file's bytes exactly); alpha is dropped. The copy's pixels are its bytes / 255.
        /// </summary>
        public static RgbImage PillowResize(RgbImage image, int width, int height, bool bicubic = false)
        {
            ArgumentNullException.ThrowIfNull(image);
            byte[] rgb = QwenImage.ImageIO.ToRgb8(image);
            byte[] resized = width == image.Width && height == image.Height
                ? rgb
                : ManagedResampler.ResizePillow(rgb, image.Width, image.Height, width, height, bicubic);
            var px = new float[resized.LongLength];
            for (long i = 0; i < px.LongLength; i++) px[i] = resized[i] / 255f;
            return new RgbImage(width, height, px);
        }

        private static void CheckSides(RgbImage image, int index)
        {
            ArgumentNullException.ThrowIfNull(image);
            if (Math.Min(image.Width, image.Height) < Multiple)
                throw new ArgumentException(
                    $"A reference image needs at least {Multiple} pixels on each side; <Picture {index + 1}> is {image.Width}x{image.Height}.");
        }

        // ---- IReferencePolicy

        public PresentedImage ForVision(RgbImage image, int index)
        {
            CheckSides(image, index);
            var (bw, bh) = BoundShortEdge(image.Width, image.Height, VisionShortEdge, VisionShortEdge);
            RgbImage bounded = bw == image.Width && bh == image.Height && image.Alpha == null ? image : PillowResize(image, bw, bh);
            var (vw, vh) = SmartResize(bw, bh);
            if ((long)vw * vh > MiniMaxH3.MiniMaxH3VisionEncoder.MaxPixels)
                throw new ArgumentException(
                    $"<Picture {index + 1}> ({image.Width}x{image.Height}) is too elongated for the vision tower: its {vw}x{vh} copy " +
                    $"is above {MiniMaxH3.MiniMaxH3VisionEncoder.MaxPixels} pixels. Crop it nearer to 4:1.");
            RgbImage vision = vw == bw && vh == bh ? bounded : PillowResize(bounded, vw, vh, bicubic: true);
            return new PresentedImage(vision, index);
        }

        /// <summary>The VAE copy: bounded to a 1024-2048 short edge, then cropped at the top left to whole 16-pixel cells
        /// (a reference keeps its own size; it is never resized to the output).</summary>
        public RgbImage ForVae(RgbImage image, int index)
        {
            CheckSides(image, index);
            var (w, h) = VaeCopySize(image.Width, image.Height);
            RgbImage copy = w == image.Width && h == image.Height && image.Alpha == null ? image : PillowResize(image, w, h);
            int cw = copy.Width / Downsample * Downsample, ch = copy.Height / Downsample * Downsample;
            return cw == copy.Width && ch == copy.Height ? copy : Crop(copy, cw, ch);
        }

        /// <summary>
        /// Reference <paramref name="index"/>'s rows: a posterior sample over the whole latent (its noise the next of the
        /// stream), rounded to float16 and back, cropped at the top left to whole 2x2 patches, normalized with the latent
        /// statistics (<c>(z - mean) / std</c>), as token rows.
        /// </summary>
        public LatentRef FromPosterior(LatentPosterior posterior, int index, INoiseSource noise)
        {
            ArgumentNullException.ThrowIfNull(posterior);
            ArgumentNullException.ThrowIfNull(noise);
            LatentShape s = posterior.Shape;
            if (s.Channels != Channels || s.Frames != 1)
                throw new ArgumentException($"MiniMax-H3-Image posteriors are [{Channels}, 1, H, W], not {s}.");
            LatentTensor sample = posterior.Sample(noise.Next(s.Count));
            int ch = s.Height / Patch * Patch, cw = s.Width / Patch * Patch;
            if (ch == 0 || cw == 0)
                throw new ArgumentException($"<Picture {index + 1}>'s latent ({s.Height}x{s.Width}) holds no whole 2x2 patch.");
            var (mean, std) = _latentStatistics();
            var cropped = new LatentShape(Channels, 1, ch, cw);
            var z = new float[cropped.Count];
            for (int c = 0; c < Channels; c++)
                for (int y = 0; y < ch; y++)
                    for (int x = 0; x < cw; x++)
                    {
                        float v = (float)(System.Half)sample.Data[((long)c * s.Height + y) * s.Width + x];
                        z[((long)c * ch + y) * cw + x] = (v - mean[c]) / std[c];
                    }
            return new LatentRef(index, cropped, _toRows(new LatentTensor(cropped, z)));
        }

        /// <summary>A masked edit's source latent: the posterior's mean through the references' own steps (F16, then the
        /// latent statistics), without a sample, so the blend does not depend on the noise stream.</summary>
        public LatentTensor SourceLatent(LatentPosterior posterior)
        {
            ArgumentNullException.ThrowIfNull(posterior);
            LatentShape s = posterior.Shape;
            if (s.Channels != Channels || s.Frames != 1)
                throw new ArgumentException($"MiniMax-H3-Image posteriors are [{Channels}, 1, H, W], not {s}.");
            var (mean, std) = _latentStatistics();
            long plane = (long)s.Height * s.Width;
            var z = new float[s.Count];
            for (int c = 0; c < Channels; c++)
                for (long i = 0; i < plane; i++)
                {
                    float v = (float)(System.Half)posterior.Mean[c * plane + i];
                    z[c * plane + i] = (v - mean[c]) / std[c];
                }
            return new LatentTensor(s, z);
        }

        /// <summary>scale_noise at t = 0.999 in float32 (<c>t * z + (1 - t) * eps</c>, the timestep a float32 tensor), with eps
        /// the next of the stream over the reference's own latent.</summary>
        public void Augment(LatentRef reference, INoiseSource noise)
        {
            ArgumentNullException.ThrowIfNull(reference);
            ArgumentNullException.ThrowIfNull(noise);
            float[] eps = _toRows(new LatentTensor(reference.Shape, noise.Next(reference.Shape.Count)));
            if (eps.LongLength != reference.Rows.LongLength)
                throw new InvalidOperationException("The reference's rows and its augmentation noise differ in length.");
            const float t = NoiseAugmentation, rest = 1f - NoiseAugmentation;
            float[] rows = reference.Rows;
            for (long i = 0; i < rows.LongLength; i++)
            {
                float kept = t * rows[i];
                float added = rest * eps[i];
                rows[i] = kept + added;
            }
        }

        private static RgbImage Crop(RgbImage image, int width, int height)
        {
            var px = new float[(long)width * height * 3];
            for (int y = 0; y < height; y++)
                Array.Copy(image.Pixels, (long)y * image.Width * 3, px, (long)y * width * 3, (long)width * 3);
            return new RgbImage(width, height, px);
        }
    }
}
