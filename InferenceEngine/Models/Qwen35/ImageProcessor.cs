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

namespace InferenceEngine
{
    public class Qwen35ImageProcessor
    {
        public int PatchSize { get; }
        public int MergeSize { get; }
        public int Factor { get; }
        public int ShortestEdge { get; }
        public int LongestEdge { get; }

        private const float ImageMean = 0.5f;
        private const float ImageStd = 0.5f;

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
            return Gemma3ImageProcessor.ReadImageDimensions(path);
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
            byte[] rgba = Gemma3ImageProcessor.DecodeImageToRGBA(fileBytes, out int origWidth, out int origHeight);

            var (resizedH, resizedW) = SmartResize(origHeight, origWidth);
            float[] pixels = Gemma3ImageProcessor.ResizeRgbaToChannelFirstNormalized(
                rgba, origWidth, origHeight, resizedW, resizedH);
            return (pixels, resizedH, resizedW);
        }

        private static float[] PackChannelFirst(byte[] rgba, int width, int height)
        {
            int numPixels = width * height;
            float[] result = new float[3 * numPixels];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pixIdx = y * width + x;
                    float r = (rgba[pixIdx * 4] / 255f - ImageMean) / ImageStd;
                    float g = (rgba[pixIdx * 4 + 1] / 255f - ImageMean) / ImageStd;
                    float b = (rgba[pixIdx * 4 + 2] / 255f - ImageMean) / ImageStd;

                    result[pixIdx] = r;
                    result[numPixels + pixIdx] = g;
                    result[2 * numPixels + pixIdx] = b;
                }
            }

            return result;
        }
    }
}
