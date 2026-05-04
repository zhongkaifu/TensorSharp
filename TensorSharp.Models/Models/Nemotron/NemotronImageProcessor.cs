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
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// Image processor for the NVIDIA Nemotron 3 Nano Omni model (RADIO/v2_vl projector).
    /// Mirrors the ollama <c>nemotronh.ImageProcessor</c> behaviour:
    ///   1. Composite RGBA over white background
    ///   2. Choose a tile grid that best matches the source aspect ratio (max <c>maxTiles</c>),
    ///      OR run dynamic-resolution mode if min/max patches metadata is present
    ///   3. Bicubic resize to <c>gridW*imageSize x gridH*imageSize</c> (or dynamic patch grid)
    ///   4. Crop into <c>imageSize x imageSize</c> tiles, channel-first [C,H,W]
    ///   5. Optional thumbnail tile when more than one tile was produced
    ///   6. Normalize each tile with the CLIP mean/std loaded from the mmproj
    /// </summary>
    public sealed class NemotronImageProcessor
    {
        public int ImageSize { get; }
        public int PatchSize { get; }
        public int NumChannels { get; }
        public int MaxTiles { get; }
        public int MinNumPatches { get; }
        public int MaxNumPatches { get; }
        public bool UseThumbnail { get; }
        public int ProjectorScaleFactor { get; }
        public float[] ImageMean { get; }
        public float[] ImageStd { get; }

        public NemotronImageProcessor(
            int imageSize,
            int patchSize,
            int numChannels,
            int maxTiles,
            int minNumPatches,
            int maxNumPatches,
            bool useThumbnail,
            int projectorScaleFactor,
            float[] imageMean,
            float[] imageStd)
        {
            ImageSize = imageSize > 0 ? imageSize : 512;
            PatchSize = patchSize > 0 ? patchSize : 16;
            NumChannels = numChannels > 0 ? numChannels : 3;
            MaxTiles = maxTiles > 0 ? maxTiles : 12;
            MinNumPatches = Math.Max(0, minNumPatches);
            MaxNumPatches = Math.Max(0, maxNumPatches);
            UseThumbnail = useThumbnail;
            ProjectorScaleFactor = projectorScaleFactor > 0 ? projectorScaleFactor : 2;
            ImageMean = imageMean ?? new float[] { 0.48145466f, 0.4578275f, 0.40821073f };
            ImageStd = imageStd ?? new float[] { 0.26862954f, 0.26130258f, 0.27577711f };
        }

        public sealed class ProcessedTile
        {
            public required float[] Pixels { get; init; }
            public required int Width { get; init; }
            public required int Height { get; init; }
            public int PatchesX => Width / 16;
            public int PatchesY => Height / 16;
        }

        public List<ProcessedTile> ProcessImage(string imagePath)
        {
            byte[] file = File.ReadAllBytes(imagePath);
            byte[] rgba = Gemma3ImageProcessor.DecodeImageToRGBA(file, out int origW, out int origH);
            rgba = Gemma3ImageProcessor.CompositeOverWhite(rgba, origW, origH);

            return UseDynamicResolution()
                ? ProcessDynamicResolution(rgba, origW, origH)
                : ProcessTiled(rgba, origW, origH);
        }

        private bool UseDynamicResolution() => MinNumPatches > 0 || MaxNumPatches > 0;

        private List<ProcessedTile> ProcessTiled(byte[] rgba, int origW, int origH)
        {
            var ratios = NemotronTargetRatios(MaxTiles);
            var (gridW, gridH) = FindClosestAspectRatio(
                (double)origW / origH, ratios, origW, origH, ImageSize);

            int targetW = ImageSize * gridW;
            int targetH = ImageSize * gridH;

            float[] resized = ResizeImageBicubicCHW(rgba, origW, origH, targetW, targetH);

            var tiles = new List<ProcessedTile>(gridW * gridH + 1);
            for (int row = 0; row < gridH; row++)
            {
                for (int col = 0; col < gridW; col++)
                {
                    float[] tile = CropCHWRegion(resized, targetW, targetH, NumChannels,
                        col * ImageSize, row * ImageSize, ImageSize, ImageSize);
                    NormalizeVisionCHWInPlace(tile, ImageMean, ImageStd);
                    tiles.Add(new ProcessedTile
                    {
                        Pixels = tile,
                        Width = ImageSize,
                        Height = ImageSize,
                    });
                }
            }

            if (UseThumbnail && tiles.Count > 1)
            {
                float[] thumb = ResizeImageBicubicCHW(rgba, origW, origH, ImageSize, ImageSize);
                NormalizeVisionCHWInPlace(thumb, ImageMean, ImageStd);
                tiles.Add(new ProcessedTile
                {
                    Pixels = thumb,
                    Width = ImageSize,
                    Height = ImageSize,
                });
            }

            return tiles;
        }

        private List<ProcessedTile> ProcessDynamicResolution(byte[] rgba, int origW, int origH)
        {
            var (patchesW, patchesH) = DynamicPatchGrid(origW, origH);
            int targetW = patchesW * PatchSize;
            int targetH = patchesH * PatchSize;
            float[] resized = ResizeImageBicubicCHW(rgba, origW, origH, targetW, targetH);
            NormalizeVisionCHWInPlace(resized, ImageMean, ImageStd);
            return new List<ProcessedTile>
            {
                new ProcessedTile
                {
                    Pixels = resized,
                    Width = targetW,
                    Height = targetH,
                },
            };
        }

        private (int width, int height) DynamicPatchGrid(int origW, int origH)
        {
            int patchesH = Math.Max(1, (int)Math.Round((double)origH / PatchSize + 0.5));
            int patchesW = Math.Max(1, (int)Math.Round((double)origW / PatchSize + 0.5));

            int patches = patchesH * patchesW;
            int currentNumPatchesAvailable = MaxNumPatches;
            if (currentNumPatchesAvailable <= 0)
                currentNumPatchesAvailable = Math.Max(patches, MinNumPatches);

            double factor = Math.Min(Math.Sqrt((double)currentNumPatchesAvailable / patches), 1.0);
            int targetH = Math.Max(1, (int)Math.Floor(factor * patchesH));
            int targetW = Math.Max(1, (int)Math.Floor(factor * patchesW));

            if (currentNumPatchesAvailable > MinNumPatches && targetH * targetW < MinNumPatches)
            {
                double upFactor = Math.Sqrt((double)MinNumPatches / (targetH * targetW));
                targetH = (int)Math.Ceiling(upFactor * targetH);
                targetW = (int)Math.Ceiling(upFactor * targetW);
            }

            targetH = RoundPatchGridForPixelShuffle(targetH, targetW, currentNumPatchesAvailable, ProjectorScaleFactor);
            targetW = RoundPatchGridForPixelShuffle(targetW, targetH, currentNumPatchesAvailable, ProjectorScaleFactor);

            return (targetW, targetH);
        }

        private static int RoundPatchGridForPixelShuffle(int v, int other, int maxPatches, int divisor)
        {
            if (divisor <= 1) return v;
            int rem = v % divisor;
            if (rem == 0) return v;

            int inc = divisor - rem;
            if ((v + inc) * other <= maxPatches)
                return v + inc;
            return Math.Max(divisor, v - rem);
        }

        private readonly struct NemotronImageRatio : IEquatable<NemotronImageRatio>
        {
            public readonly int Width;
            public readonly int Height;
            public NemotronImageRatio(int w, int h) { Width = w; Height = h; }
            public bool Equals(NemotronImageRatio other) => Width == other.Width && Height == other.Height;
            public override bool Equals(object obj) => obj is NemotronImageRatio r && Equals(r);
            public override int GetHashCode() => (Width * 397) ^ Height;
        }

        private static List<NemotronImageRatio> NemotronTargetRatios(int maxTiles)
        {
            var raw = new List<NemotronImageRatio>(maxTiles * maxTiles);
            for (int n = 1; n <= maxTiles; n++)
            {
                for (int w = 1; w <= n; w++)
                {
                    for (int h = 1; h <= n; h++)
                    {
                        if (w * h > maxTiles) continue;
                        raw.Add(new NemotronImageRatio(w, h));
                    }
                }
            }

            var unique = new List<NemotronImageRatio>(raw.Count);
            foreach (var r in raw)
            {
                if (!unique.Contains(r))
                    unique.Add(r);
            }

            unique.Sort((a, b) => (a.Width * a.Height).CompareTo(b.Width * b.Height));
            return unique;
        }

        private static (int width, int height) FindClosestAspectRatio(
            double aspectRatio, List<NemotronImageRatio> targetRatios, int width, int height, int imageSize)
        {
            var best = new NemotronImageRatio(1, 1);
            double bestDiff = double.MaxValue;
            int area = width * height;

            foreach (var r in targetRatios)
            {
                double targetAr = (double)r.Width / r.Height;
                double diff = Math.Abs(aspectRatio - targetAr);
                if (diff < bestDiff)
                {
                    bestDiff = diff;
                    best = r;
                    continue;
                }
                if (diff == bestDiff && area > (int)(0.5 * imageSize * imageSize * r.Width * r.Height))
                    best = r;
            }
            return (best.Width, best.Height);
        }

        /// <summary>
        /// PyTorch-equivalent bicubic resize from RGBA bytes to channel-first F32 [C,H,W],
        /// values in [0,1]. Matches torch.nn.functional.interpolate(mode="bicubic",
        /// align_corners=False, antialias=False) up to floating point precision.
        /// </summary>
        private static float[] ResizeImageBicubicCHW(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
        {
            int srcPlane = srcW * srcH;
            float[] src = new float[3 * srcPlane];
            Parallel.For(0, srcH, y =>
            {
                for (int x = 0; x < srcW; x++)
                {
                    int idx = (y * srcW + x) * 4;
                    int dst = y * srcW + x;
                    src[dst] = rgba[idx] / 255.0f;
                    src[srcPlane + dst] = rgba[idx + 1] / 255.0f;
                    src[2 * srcPlane + dst] = rgba[idx + 2] / 255.0f;
                }
            });

            int dstPlane = dstW * dstH;
            float[] dstArr = new float[3 * dstPlane];
            double scaleX = (double)srcW / dstW;
            double scaleY = (double)srcH / dstH;

            Parallel.For(0, dstH, oy =>
            {
                double srcY = scaleY * (oy + 0.5) - 0.5;
                int yBase = (int)Math.Floor(srcY);
                double yFrac = ClampUnit(srcY - yBase);
                Span<double> wy = stackalloc double[4];
                TorchBicubicWeights(yFrac, wy);

                Span<double> wx = stackalloc double[4];
                for (int ox = 0; ox < dstW; ox++)
                {
                    double srcX = scaleX * (ox + 0.5) - 0.5;
                    int xBase = (int)Math.Floor(srcX);
                    double xFrac = ClampUnit(srcX - xBase);
                    TorchBicubicWeights(xFrac, wx);

                    for (int c = 0; c < 3; c++)
                    {
                        double sum = 0;
                        int channelBase = c * srcPlane;
                        for (int ky = 0; ky < 4; ky++)
                        {
                            int iy = ClampIndex(yBase - 1 + ky, 0, srcH - 1);
                            int rowBase = channelBase + iy * srcW;
                            for (int kx = 0; kx < 4; kx++)
                            {
                                int ix = ClampIndex(xBase - 1 + kx, 0, srcW - 1);
                                sum += src[rowBase + ix] * wy[ky] * wx[kx];
                            }
                        }
                        dstArr[c * dstPlane + oy * dstW + ox] = (float)sum;
                    }
                }
            });

            return dstArr;
        }

        private static float[] CropCHWRegion(float[] values, int width, int height, int channels,
            int left, int top, int cropW, int cropH)
        {
            float[] outArr = new float[channels * cropW * cropH];
            int channelSize = width * height;
            int cropSize = cropW * cropH;
            for (int c = 0; c < channels; c++)
            {
                int srcBase = c * channelSize;
                int dstBase = c * cropSize;
                for (int y = 0; y < cropH; y++)
                {
                    Buffer.BlockCopy(values, (srcBase + (top + y) * width + left) * sizeof(float),
                        outArr, (dstBase + y * cropW) * sizeof(float),
                        cropW * sizeof(float));
                }
            }
            return outArr;
        }

        private static void NormalizeVisionCHWInPlace(float[] values, float[] mean, float[] std)
        {
            int channelSize = values.Length / 3;
            for (int c = 0; c < 3; c++)
            {
                float m = mean[c];
                float s = std[c];
                int baseOff = c * channelSize;
                for (int i = 0; i < channelSize; i++)
                    values[baseOff + i] = (values[baseOff + i] - m) / s;
            }
        }

        private static void TorchBicubicWeights(double t, Span<double> weights)
        {
            const double a = -0.75;
            weights[0] = BicubicConvolution2(t + 1.0, a);
            weights[1] = BicubicConvolution1(t, a);
            weights[2] = BicubicConvolution1(1.0 - t, a);
            weights[3] = BicubicConvolution2(2.0 - t, a);
        }

        private static double BicubicConvolution1(double x, double a) =>
            ((a + 2) * x - (a + 3)) * x * x + 1;

        private static double BicubicConvolution2(double x, double a) =>
            ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;

        private static double ClampUnit(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);

        private static int ClampIndex(int v, int lo, int hi) =>
            v < lo ? lo : (v > hi ? hi : v);

        /// <summary>
        /// Pack tile pixels into the patch-major layout expected by the vision encoder.
        /// Output is flattened as [numPatches][channels*patchSize*patchSize] — token-major
        /// with each token packed as channel, patch-row, patch-col. Mirrors
        /// <c>packVisionPatchesCHW</c> in the reference Go implementation.
        /// </summary>
        public static float[] PackPatchesCHW(float[] values, int width, int height, int channels, int patchSize)
        {
            int patchesX = width / patchSize;
            int patchesY = height / patchSize;
            int patchDim = channels * patchSize * patchSize;
            int plane = width * height;

            float[] patches = new float[patchDim * patchesX * patchesY];
            int offset = 0;
            for (int py = 0; py < patchesY; py++)
            {
                for (int px = 0; px < patchesX; px++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        int channelBase = c * plane;
                        for (int yy = 0; yy < patchSize; yy++)
                        {
                            int rowBase = (py * patchSize + yy) * width;
                            int colBase = px * patchSize;
                            Buffer.BlockCopy(values, (channelBase + rowBase + colBase) * sizeof(float),
                                patches, offset * sizeof(float), patchSize * sizeof(float));
                            offset += patchSize;
                        }
                    }
                }
            }
            return patches;
        }
    }
}
