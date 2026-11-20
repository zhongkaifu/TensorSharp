// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Numerics;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>How a mask image's samples select the area an edit regenerates.</summary>
    public enum EditMaskMode
    {
        /// <summary>RGB luminance: white edits, black preserves. Alpha is ignored.</summary>
        Grayscale,
        /// <summary>Inverse alpha: transparent edits, opaque preserves. RGB is ignored.</summary>
        Alpha
    }

    /// <summary>A request's mask and how to interpret it (the hosts' <c>maskMode</c>, <c>maskInvert</c>,
    /// <c>maskFeather</c>, <c>maskCrop</c> and <c>maskCropPadding</c>).</summary>
    public sealed record EditMaskOptions
    {
        /// <summary>The mask image, exactly the first reference's size; null for an unmasked request.</summary>
        public RgbImage Mask { get; init; }

        public EditMaskMode MaskMode { get; init; } = EditMaskMode.Grayscale;

        /// <summary>Invert the selection after interpreting its grayscale or alpha channel.</summary>
        public bool MaskInvert { get; init; }

        /// <summary>Inward box feather radius in source pixels (0..1024). Never expands the selection.</summary>
        public int MaskFeather { get; init; }

        /// <summary>Sample only the selection bounds plus <see cref="MaskCropPadding"/> of context.</summary>
        public bool MaskCrop { get; init; }

        /// <summary>Context padding around a cropped selection, in source pixels (0..16384).</summary>
        public int MaskCropPadding { get; init; } = 64;
    }

    /// <summary>
    /// Request-local mask geometry for a masked edit (0 keeps, 1 regenerates). The original canvas and blend weights
    /// never pass through the VAE or an image codec, so zero-weight pixels retain their exact RGB and alpha. Shared by
    /// every diffusion family: the latent reduction (<see cref="LatentWeights"/>), the fixed-noise flow reinjection
    /// (<see cref="Reinject"/>) and the final composite (<see cref="Composite"/>) do not depend on the model.
    /// </summary>
    public sealed class EditMask
    {
        public RgbImage Source { get; }
        public RgbImage Reference { get; }
        public float[] Weights { get; }
        public int X { get; }
        public int Y { get; }
        public int Width { get; }
        public int Height { get; }
        public bool IsEmpty { get; }
        public bool IsFull { get; }

        private EditMask(RgbImage source, float[] weights, int x, int y, int width, int height, bool empty, bool full)
        {
            Source = source; Weights = weights; X = x; Y = y; Width = width; Height = height;
            IsEmpty = empty; IsFull = full;
            Reference = x == 0 && y == 0 && width == source.Width && height == source.Height
                ? source : Crop(source, x, y, width, height);
        }

        /// <summary>The selection of <paramref name="o"/> over <paramref name="source"/> (the first reference), or null
        /// for an unmasked request, which must leave every mask option at its default.</summary>
        public static EditMask Create(EditMaskOptions o, RgbImage source)
        {
            ArgumentNullException.ThrowIfNull(o);
            if (!Enum.IsDefined(o.MaskMode)) throw new ArgumentException("Unknown mask mode.", nameof(o.MaskMode));
            if (o.MaskFeather < 0 || o.MaskFeather > 1024)
                throw new ArgumentOutOfRangeException(nameof(o.MaskFeather), "Mask feather must be between 0 and 1024 source pixels.");
            if (o.MaskCropPadding < 0 || o.MaskCropPadding > 16384)
                throw new ArgumentOutOfRangeException(nameof(o.MaskCropPadding), "Mask crop padding must be between 0 and 16384 source pixels.");
            if (o.Mask == null)
            {
                if (o.MaskMode != EditMaskMode.Grayscale || o.MaskInvert || o.MaskFeather != 0 || o.MaskCrop || o.MaskCropPadding != 64)
                    throw new ArgumentException("Mask processing options require a mask.");
                return null;
            }
            if (source == null) throw new ArgumentException("A mask requires a first reference image to edit.");
            if (source.Width <= 0 || source.Height <= 0 || o.Mask.Width != source.Width || o.Mask.Height != source.Height)
                throw new ArgumentException("The mask must have exactly the same width and height as the first reference image.");
            if (o.MaskMode == EditMaskMode.Alpha && o.Mask.Alpha == null)
                throw new ArgumentException("Alpha mask mode requires an alpha plane; transparent pixels select the area to edit.");

            int width = source.Width, height = source.Height;
            var weights = new float[checked(width * height)];
            int left = width, top = height, right = -1, bottom = -1;
            bool full = true;
            for (int i = 0; i < weights.Length; i++)
            {
                float value;
                if (o.MaskMode == EditMaskMode.Alpha)
                {
                    value = 1f - ValidateWeight(o.Mask.Alpha[i]);
                }
                else
                {
                    float r = ValidateWeight(o.Mask.Pixels[i * 3]);
                    float g = ValidateWeight(o.Mask.Pixels[i * 3 + 1]);
                    float b = ValidateWeight(o.Mask.Pixels[i * 3 + 2]);
                    // Integer luminance coefficients sum exactly to 256: equal RGB values
                    // (including solid white) keep their exact mask weight.
                    value = (r * 77f + g * 150f + b * 29f) / 256f;
                }
                if (o.MaskInvert) value = 1f - value;
                weights[i] = value;
                full &= value == 1f;
                if (value <= 0f) continue;
                int x = i % width, y = i / width;
                left = Math.Min(left, x); right = Math.Max(right, x);
                top = Math.Min(top, y); bottom = Math.Max(bottom, y);
            }
            bool empty = right < 0;
            if (o.MaskFeather > 0 && !empty && !full) FeatherInward(weights, width, height, o.MaskFeather);
            if (!o.MaskCrop || empty)
                return new EditMask(source, weights, 0, 0, width, height, empty, full);
            int padding = o.MaskCropPadding;
            left = Math.Max(0, left - padding); top = Math.Max(0, top - padding);
            right = (int)Math.Min(width - 1L, (long)right + padding);
            bottom = (int)Math.Min(height - 1L, (long)bottom + padding);
            return new EditMask(source, weights, left, top, right - left + 1, bottom - top + 1, empty, full);
        }

        private static float ValidateWeight(float value)
        {
            if (!float.IsFinite(value) || value < 0 || value > 1)
                throw new ArgumentException("Mask samples must be finite values between 0 and 1.");
            return value;
        }

        /// <summary>Scale the crop at the full-canvas sampling density, snapping upward to the
        /// <paramref name="multiple"/>-pixel grid (at least one cell of it).</summary>
        public (int Width, int Height) SamplingDimensions(int fullWidth, int fullHeight, int multiple = 32) =>
            (GridSize(fullWidth, Width, Source.Width, multiple), GridSize(fullHeight, Height, Source.Height, multiple));

        private static int GridSize(int sampleSize, int cropSize, int sourceSize, int multiple) =>
            Math.Max(multiple, checked((int)Math.Ceiling((double)sampleSize * cropSize / sourceSize / multiple) * multiple));

        /// <summary>
        /// Conservative coverage reduction to the latent grid (one weight a latent cell). Maximum pooling keeps a
        /// one-pixel brush stroke editable after the VAE's reduction; averaging would almost erase a thin selection.
        /// Row-major [latentHeight, latentWidth]; a family broadcasts it over its channels in its own token layout.
        /// </summary>
        public float[] LatentWeights(int latentWidth, int latentHeight)
        {
            if (latentWidth <= 0 || latentHeight <= 0) throw new ArgumentOutOfRangeException();
            var result = new float[checked(latentWidth * latentHeight)];
            for (int y = 0; y < latentHeight; y++)
            {
                int y0 = Y + (int)((long)y * Height / latentHeight);
                int y1 = Y + (int)(((long)(y + 1) * Height + latentHeight - 1) / latentHeight);
                for (int x = 0; x < latentWidth; x++)
                {
                    int x0 = X + (int)((long)x * Width / latentWidth);
                    int x1 = X + (int)(((long)(x + 1) * Width + latentWidth - 1) / latentWidth);
                    float max = 0;
                    for (int yy = y0; yy < y1 && max < 1f; yy++)
                        for (int xx = x0; xx < x1; xx++) max = Math.Max(max, Weights[yy * Source.Width + xx]);
                    result[y * latentWidth + x] = max;
                }
            }
            return result;
        }

        /// <summary>
        /// Fixed-seed flow reinjection at the UPDATED state's sigma, without per-step allocation: where a weight m is
        /// below 1, <c>x = m*x + (1-m)*((1-sigma)*source + sigma*noise)</c>, so the last step (sigma 0) leaves the source
        /// exactly in the kept values. Each weight of <paramref name="mask"/> covers <paramref name="channels"/>
        /// contiguous values of the latent rows (64 for Qwen-Image's tokens; 1 for a mask already in the rows' layout).
        /// </summary>
        public static void Reinject(Span<float> latents, ReadOnlySpan<float> source, ReadOnlySpan<float> noise,
            ReadOnlySpan<float> mask, int channels, float sigma)
        {
            if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
            if (latents.Length != source.Length || latents.Length != noise.Length || latents.Length != checked(mask.Length * channels))
                throw new ArgumentException($"Masked flow requires {channels} channels per mask token and matching latent lengths.");
            if (!float.IsFinite(sigma) || sigma < 0 || sigma > 1) throw new ArgumentOutOfRangeException(nameof(sigma));
            int lanes = Vector<float>.Count;
            var sigmaV = new Vector<float>(sigma);
            var cleanV = new Vector<float>(1f - sigma);
            for (int token = 0; token < mask.Length; token++)
            {
                float amount = mask[token];
                if (amount == 1f) continue;
                int start = token * channels;
                var edit = new Vector<float>(amount);
                var keep = new Vector<float>(1f - amount);
                int c = 0;
                if (Vector.IsHardwareAccelerated)
                    for (; c <= channels - lanes; c += lanes)
                    {
                        int at = start + c;
                        var original = new Vector<float>(source.Slice(at, lanes)) * cleanV + new Vector<float>(noise.Slice(at, lanes)) * sigmaV;
                        var result = amount == 0f ? original : new Vector<float>(latents.Slice(at, lanes)) * edit + original * keep;
                        result.CopyTo(latents.Slice(at, lanes));
                    }
                for (; c < channels; c++)
                {
                    int at = start + c;
                    float original = (1f - sigma) * source[at] + sigma * noise[at];
                    latents[at] = amount == 0f ? original : amount * latents[at] + (1f - amount) * original;
                }
            }
        }

        /// <summary>The decoded picture resized to the selection and blended into the source canvas: weight-0 pixels
        /// are the source's samples exactly (alpha included), weight-1 pixels the generated ones.</summary>
        public RgbImage Composite(RgbImage generated) => Composite(generated, null);

        /// <summary><see cref="Composite(RgbImage)"/> with the decode resized to the selection by <paramref name="resize"/>
        /// (a family's own filter; null is <see cref="ImageIO.Resize"/>). The weight-0 pixels never pass through it.</summary>
        public RgbImage Composite(RgbImage generated, Func<RgbImage, int, int, RgbImage> resize)
        {
            ArgumentNullException.ThrowIfNull(generated);
            if (generated.Width != Width || generated.Height != Height)
                generated = resize != null ? resize(generated, Width, Height) : QwenImage.ImageIO.Resize(generated, Width, Height);
            var result = (float[])Source.Pixels.Clone();
            var alpha = Source.Alpha == null ? null : (float[])Source.Alpha.Clone();
            if (alpha == null && generated.Alpha != null)
            {
                alpha = new float[Weights.Length];
                Array.Fill(alpha, 1f);
            }
            for (int y = 0; y < Height; y++)
                for (int x = 0; x < Width; x++)
                {
                    int pixel = (y + Y) * Source.Width + x + X;
                    float weight = Weights[pixel];
                    if (weight == 0f) continue; // Exact original samples, including hidden RGB under alpha=0.
                    int from = (y * Width + x) * 3, to = pixel * 3;
                    for (int c = 0; c < 3; c++)
                        result[to + c] = weight == 1f ? generated.Pixels[from + c]
                            : result[to + c] * (1f - weight) + generated.Pixels[from + c] * weight;
                    if (alpha != null)
                    {
                        float generatedAlpha = generated.Alpha?[y * Width + x] ?? 1f;
                        alpha[pixel] = weight == 1f ? generatedAlpha : alpha[pixel] * (1f - weight) + generatedAlpha * weight;
                    }
                }
            return new RgbImage(Source.Width, Source.Height, result, alpha);
        }

        /// <summary>An owned copy of the source (an empty mask's answer, without inference).</summary>
        public RgbImage UnchangedCopy() => new RgbImage(Source.Width, Source.Height,
            (float[])Source.Pixels.Clone(), Source.Alpha == null ? null : (float[])Source.Alpha.Clone());

        private static RgbImage Crop(RgbImage source, int x, int y, int width, int height)
        {
            var pixels = new float[checked(width * height * 3)];
            var alpha = source.Alpha == null ? null : new float[checked(width * height)];
            for (int row = 0; row < height; row++)
            {
                Array.Copy(source.Pixels, ((y + row) * source.Width + x) * 3, pixels, row * width * 3, width * 3);
                if (alpha != null) Array.Copy(source.Alpha, (y + row) * source.Width + x, alpha, row * width, width);
            }
            return new RgbImage(width, height, pixels, alpha);
        }

        private static void FeatherInward(float[] weights, int width, int height, int radius)
        {
            // Separable sliding sums: O(pixels), independent of brush radius. Edge clamping
            // keeps a selection reaching the image border fully selected at that border.
            var horizontal = new float[weights.Length];
            int diameter = radius * 2 + 1;
            for (int y = 0; y < height; y++)
            {
                int row = y * width;
                double sum = (radius + 1.0) * weights[row];
                for (int x = 1; x <= Math.Min(radius, width - 1); x++) sum += weights[row + x];
                if (radius >= width) sum += (radius - width + 1.0) * weights[row + width - 1];
                for (int x = 0; x < width; x++)
                {
                    horizontal[row + x] = (float)(sum / diameter);
                    sum -= weights[row + Math.Clamp(x - radius, 0, width - 1)];
                    sum += weights[row + Math.Clamp(x + radius + 1, 0, width - 1)];
                }
            }
            for (int x = 0; x < width; x++)
            {
                double sum = (radius + 1.0) * horizontal[x];
                for (int y = 1; y <= Math.Min(radius, height - 1); y++) sum += horizontal[y * width + x];
                if (radius >= height) sum += (radius - height + 1.0) * horizontal[(height - 1) * width + x];
                for (int y = 0; y < height; y++)
                {
                    int i = y * width + x;
                    weights[i] = Math.Min(weights[i], Math.Max(0, (float)(sum / diameter)));
                    sum -= horizontal[Math.Clamp(y - radius, 0, height - 1) * width + x];
                    sum += horizontal[Math.Clamp(y + radius + 1, 0, height - 1) * width + x];
                }
            }
        }
    }
}
