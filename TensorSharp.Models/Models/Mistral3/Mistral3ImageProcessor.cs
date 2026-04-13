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

namespace TensorSharp.Models
{
    /// <summary>
    /// Image processor for Mistral 3 / Pixtral models.
    /// Processing pipeline:
    /// 1. Composite transparent images over white background
    /// 2. Resize to fit longest_edge while preserving aspect ratio
    /// 3. Pad to be divisible by patch_size
    /// 4. Normalize with CLIP default mean/std
    /// </summary>
    public class Mistral3ImageProcessor
    {
        public int ImageSize { get; }
        public int PatchSize { get; }
        public int NumChannels { get; }
        public int LongestEdge { get; }

        // CLIP default normalization parameters
        private static readonly float[] ClipMean = { 0.48145466f, 0.4578275f, 0.40821073f };
        private static readonly float[] ClipStd = { 0.26862954f, 0.26130258f, 0.27577711f };

        // Special token IDs for Mistral 3 vision
        public const int ImgTokenId = 10;
        public const int ImgBreakTokenId = 12;
        public const int ImgEndTokenId = 13;

        public Mistral3ImageProcessor(int imageSize = 1540, int patchSize = 14,
            int numChannels = 3, int longestEdge = 1540)
        {
            ImageSize = imageSize;
            PatchSize = patchSize;
            NumChannels = numChannels;
            LongestEdge = longestEdge;
        }

        /// <summary>
        /// Process an image file for Mistral 3 vision encoder.
        /// Returns (pixelValues, finalWidth, finalHeight).
        /// pixelValues is in channel-first format [C, H, W], normalized with CLIP mean/std.
        /// </summary>
        public (float[] pixels, int width, int height) ProcessImage(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            byte[] rgba = Gemma3ImageProcessor.DecodeImageToRGBA(fileBytes, out int origWidth, out int origHeight);

            // Composite over white background
            rgba = Gemma3ImageProcessor.CompositeOverWhite(rgba, origWidth, origHeight);

            // Resize to fit longest_edge
            double ratio = Math.Max((double)origHeight / LongestEdge, (double)origWidth / LongestEdge);
            int newWidth = origWidth, newHeight = origHeight;
            if (ratio > 1.0)
            {
                newWidth = (int)Math.Floor(origWidth / ratio);
                newHeight = (int)Math.Floor(origHeight / ratio);
            }

            // Pad to be divisible by patch_size
            int patchesX = (newWidth - 1) / PatchSize + 1;
            int patchesY = (newHeight - 1) / PatchSize + 1;
            int finalWidth = patchesX * PatchSize;
            int finalHeight = patchesY * PatchSize;

            // Resize and normalize
            float[] pixels = ResizeAndNormalize(rgba, origWidth, origHeight, finalWidth, finalHeight);

            Console.WriteLine($"Mistral3 image: {origWidth}x{origHeight} → {finalWidth}x{finalHeight} " +
                $"({patchesX}x{patchesY} patches)");

            return (pixels, finalWidth, finalHeight);
        }

        /// <summary>
        /// Bilinear resize + CLIP normalization in a single pass.
        /// Output is channel-first: [R..., G..., B...].
        /// </summary>
        private float[] ResizeAndNormalize(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
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
                        double v00 = rgba[(y0 * srcW + x0) * 4 + c] / 255.0;
                        double v01 = rgba[(y0 * srcW + x1) * 4 + c] / 255.0;
                        double v10 = rgba[(y1 * srcW + x0) * 4 + c] / 255.0;
                        double v11 = rgba[(y1 * srcW + x1) * 4 + c] / 255.0;

                        double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                                   v10 * (1 - fx) * fy + v11 * fx * fy;

                        result[c * pixels + dstIdx] = (float)((v - ClipMean[c]) / ClipStd[c]);
                    }
                }
            });

            return result;
        }

        /// <summary>
        /// Compute the number of vision tokens for a processed image.
        /// After patch merging, tokens = (patchesW / mergeSize) * (patchesH / mergeSize).
        /// Each row becomes [IMG]...[IMG] tokens, rows separated by [IMG_BREAK], ending with [IMG_END].
        /// </summary>
        public int ComputeVisionTokenCount(int imageWidth, int imageHeight, int spatialMergeSize)
        {
            int patchesW = imageWidth / PatchSize;
            int patchesH = imageHeight / PatchSize;
            int mergedW = patchesW / spatialMergeSize;
            int mergedH = patchesH / spatialMergeSize;

            // mergedH rows of mergedW [IMG] tokens each
            // Plus (mergedH - 1) [IMG_BREAK] tokens and 1 [IMG_END] token
            return mergedW * mergedH + mergedH;
        }
    }
}
