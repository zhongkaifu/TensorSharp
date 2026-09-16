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

namespace TensorSharp.Models
{
    public class Qwen35ImageProcessor
    {
        public int PatchSize { get; }
        public int MergeSize { get; }
        public int Factor { get; }
        public int ShortestEdge { get; }
        public int LongestEdge { get; }

        public Qwen35ImageProcessor(int patchSize = 14, int mergeSize = 2,
            int shortestEdge = 64 * 1024, int longestEdge = 2 * 1024 * 1024)
        {
            PatchSize = patchSize;
            MergeSize = mergeSize;
            Factor = patchSize * mergeSize;
            ShortestEdge = shortestEdge;
            LongestEdge = longestEdge;
        }

        public static (int width, int height) ReadImageDimensions(string path)
        {
            return ImageProcessorUtils.ReadImageDimensions(path);
        }

        public (int height, int width) SmartResize(int height, int width)
        {
            int factor = Factor;
            if (height < factor || width < factor)
                throw new ArgumentException($"Image too small: {height}x{width}, minimum {factor}x{factor}");

            int hBar = (int)Math.Round((double)height / factor, MidpointRounding.ToEven) * factor;
            int wBar = (int)Math.Round((double)width / factor, MidpointRounding.ToEven) * factor;

            if ((long)hBar * wBar > LongestEdge)
            {
                double beta = Math.Sqrt((double)height * width / LongestEdge);
                hBar = (int)Math.Floor(height / beta / factor) * factor;
                wBar = (int)Math.Floor(width / beta / factor) * factor;
            }
            else if ((long)hBar * wBar < ShortestEdge)
            {
                double beta = Math.Sqrt((double)ShortestEdge / (height * width));
                hBar = (int)Math.Ceiling(height * beta / factor) * factor;
                wBar = (int)Math.Ceiling(width * beta / factor) * factor;
            }

            return (hBar, wBar);
        }

        public int ComputeImageTokenCount(int origHeight, int origWidth)
        {
            var (resizedH, resizedW) = SmartResize(origHeight, origWidth);
            int gridH = resizedH / PatchSize;
            int gridW = resizedW / PatchSize;
            return (gridH / MergeSize) * (gridW / MergeSize);
        }

        public int ComputeImageTokenCount(string imagePath)
        {
            var (width, height) = ReadImageDimensions(imagePath);
            return ComputeImageTokenCount(height, width);
        }

        public (int gridHeight, int gridWidth) GetPatchGrid(int origHeight, int origWidth)
        {
            var (resizedH, resizedW) = SmartResize(origHeight, origWidth);
            return (resizedH / PatchSize, resizedW / PatchSize);
        }

        /// <summary>
        /// Full image processing pipeline: load, composite, resize, normalize to channel-first float array.
        /// Returns (normalizedPixels, resizedHeight, resizedWidth).
        /// </summary>
        public (float[] pixels, int resizedH, int resizedW) ProcessImage(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            byte[] rgba = ImageProcessorUtils.DecodeImageToRGBA(fileBytes, out int origWidth, out int origHeight);

            var (resizedH, resizedW) = SmartResize(origHeight, origWidth);
            float[] pixels = ImageProcessorUtils.ResizeRgbaToChannelFirstNormalized(
                rgba, origWidth, origHeight, resizedW, resizedH);
            return (pixels, resizedH, resizedW);
        }

        /// <summary>
        /// Load, composite, resize to a caller-chosen size and normalize one video frame.
        /// Every frame of a clip is resized to the clip's <see cref="SmartResizeVideo"/>
        /// size so the frames of a temporal pair, and every pair of the clip, share one
        /// patch grid.
        /// </summary>
        public float[] ProcessImage(string imagePath, int resizedH, int resizedW)
        {
            if (resizedH < Factor || resizedW < Factor || resizedH % Factor != 0 || resizedW % Factor != 0)
                throw new ArgumentException($"Video frame size {resizedH}x{resizedW} is not a multiple of {Factor}.");
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            byte[] rgba = ImageProcessorUtils.DecodeImageToRGBA(fileBytes, out int origWidth, out int origHeight);
            return ImageProcessorUtils.ResizeRgbaToChannelFirstNormalized(
                rgba, origWidth, origHeight, resizedW, resizedH);
        }

        /// <summary>Qwen3-VL video processor pixel floor over the whole padded clip (4 merged tokens).</summary>
        public const long VideoMinPixels = 4L * 32 * 32;
        /// <summary>Qwen3-VL video processor pixel budget over the whole padded clip (24 576 merged tokens).</summary>
        public const long VideoMaxPixels = 24576L * 32 * 32;

        /// <summary>
        /// The Qwen3-VL video processor's <c>smart_resize</c>: the frame size is rounded to
        /// the patch factor like a still image, but the pixel floor and budget are
        /// applied to <c>t_bar * h * w</c> where <c>t_bar</c> is the frame count rounded
        /// up to the temporal patch size, so a long clip is scaled down as a whole rather
        /// than frame by frame. Returns (height, width) shared by every frame of the clip.
        /// </summary>
        public (int height, int width) SmartResizeVideo(int frameCount, int height, int width,
            int temporalPatchSize = 2, long minPixels = VideoMinPixels, long maxPixels = VideoMaxPixels)
        {
            if (frameCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(frameCount));
            int factor = Factor;
            if (height < factor || width < factor)
                throw new ArgumentException($"Video frame too small: {height}x{width}, minimum {factor}x{factor}");
            if ((double)Math.Max(height, width) / Math.Min(height, width) > 200)
                throw new ArgumentException($"Video frame aspect ratio {height}x{width} exceeds 200:1.");

            long tBar = (long)Math.Ceiling((double)frameCount / temporalPatchSize) * temporalPatchSize;
            int hBar = (int)Math.Round((double)height / factor, MidpointRounding.ToEven) * factor;
            int wBar = (int)Math.Round((double)width / factor, MidpointRounding.ToEven) * factor;

            if (tBar * hBar * wBar > maxPixels)
            {
                double beta = Math.Sqrt((double)tBar * height * width / maxPixels);
                hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
                wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
            }
            else if (tBar * hBar * wBar < minPixels)
            {
                double beta = Math.Sqrt((double)minPixels / ((double)tBar * height * width));
                hBar = (int)Math.Ceiling(height * beta / factor) * factor;
                wBar = (int)Math.Ceiling(width * beta / factor) * factor;
            }

            return (hBar, wBar);
        }
    }
}

