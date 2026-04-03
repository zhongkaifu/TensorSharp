using System;
using System.IO;

namespace InferenceEngine
{
    public class Gemma4ImageProcessor
    {
        public int PatchSize { get; }
        public int NMerge { get; }

        private readonly int _minPixels;
        private readonly int _maxPixels;

        public Gemma4ImageProcessor(int patchSize = 16, int nMerge = 3, int minTokens = 40, int maxTokens = 280)
        {
            PatchSize = patchSize;
            NMerge = nMerge;
            int patchArea = patchSize * patchSize * nMerge * nMerge;
            _minPixels = minTokens * patchArea;
            _maxPixels = maxTokens * patchArea;
        }

        /// <summary>
        /// Process an image file into normalized pixel values in channel-first format [C, H, W].
        /// Normalization: 2 * pixel/255 - 1 (maps [0,255] → [-1,1]).
        /// Returns pixel data and the actual target dimensions.
        /// </summary>
        public (float[] pixels, int width, int height) ProcessImage(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            int origWidth, origHeight;
            byte[] rgba = Gemma3ImageProcessor.DecodeImageToRGBA(fileBytes, out origWidth, out origHeight);
            byte[] composited = Gemma3ImageProcessor.CompositeOverWhite(rgba, origWidth, origHeight);

            int alignSize = PatchSize * NMerge;
            SmartResize(origWidth, origHeight, alignSize, out int targetW, out int targetH);

            byte[] resized = Gemma3ImageProcessor.BilinearResize(composited, origWidth, origHeight, targetW, targetH);

            float[] pixels = PackChannelFirst(resized, targetW, targetH);
            return (pixels, targetW, targetH);
        }

        private void SmartResize(int origW, int origH, int alignSize, out int targetW, out int targetH)
        {
            int totalPx = origW * origH;
            if (_maxPixels > 0 && totalPx > 0)
            {
                double factor = Math.Sqrt((double)_maxPixels / totalPx);
                targetH = Math.Max(alignSize, (int)Math.Floor(factor * origH / alignSize) * alignSize);
                targetW = Math.Max(alignSize, (int)Math.Floor(factor * origW / alignSize) * alignSize);
            }
            else
            {
                targetH = Math.Max(alignSize, (origH / alignSize) * alignSize);
                targetW = Math.Max(alignSize, (origW / alignSize) * alignSize);
            }
        }

        /// <summary>
        /// Pack RGBA pixels into channel-first float [R..., G..., B...] normalized to [-1, 1].
        /// </summary>
        private static float[] PackChannelFirst(byte[] rgba, int width, int height)
        {
            int pixels = width * height;
            float[] result = new float[3 * pixels];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pixIdx = y * width + x;
                    result[pixIdx] = rgba[pixIdx * 4] / 255f * 2f - 1f;
                    result[pixels + pixIdx] = rgba[pixIdx * 4 + 1] / 255f * 2f - 1f;
                    result[2 * pixels + pixIdx] = rgba[pixIdx * 4 + 2] / 255f * 2f - 1f;
                }
            }

            return result;
        }

        public int ComputeOutputTokens(int imageWidth, int imageHeight)
        {
            int patchesX = imageWidth / PatchSize;
            int patchesY = imageHeight / PatchSize;
            int mergedX = patchesX / NMerge;
            int mergedY = patchesY / NMerge;
            return mergedX * mergedY;
        }
    }
}
