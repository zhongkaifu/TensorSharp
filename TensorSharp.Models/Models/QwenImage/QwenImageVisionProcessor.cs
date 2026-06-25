// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Qwen2.5-VL image preprocessing for the text-encoder vision tower: resize the
    /// condition image to ~384² area, smart-resize to a multiple of 28 (patch*merge),
    /// CLIP-normalize, and patchify into <c>[gridH*gridW, 1176]</c> in the merge-grouped,
    /// <c>[c,t,ph,pw]</c> layout the encoder expects (matches the transformers processor).
    /// </summary>
    internal static class QwenImageVisionProcessor
    {
        private const int Patch = 14, Merge = 2, Factor = 28;     // patch*merge
        private const long MinPixels = 256L * 28 * 28, MaxPixels = 512L * 28 * 28;
        private const long ConditionArea = 384L * 384;
        private static readonly float[] Mean = { 0.48145467f, 0.45782751f, 0.40821072f };
        private static readonly float[] Std = { 0.26862955f, 0.26130259f, 0.27577710f };

        public static float[] Preprocess(RgbImage input, out int gridH, out int gridW)
        {
            RgbImage cond = ImageIO.ResizeToArea(input, ConditionArea, multiple: 1);
            (int hb, int wb) = SmartResize(cond.Height, cond.Width);
            RgbImage img = ImageIO.Resize(cond, wb, hb);
            gridH = hb / Patch; gridW = wb / Patch;
            return Patchify(img, gridH, gridW);
        }

        private static (int, int) SmartResize(int h, int w)
        {
            int hb = Math.Max(Factor, (int)Math.Round((double)h / Factor) * Factor);
            int wb = Math.Max(Factor, (int)Math.Round((double)w / Factor) * Factor);
            long pixels = (long)hb * wb;
            if (pixels > MaxPixels)
            {
                double beta = Math.Sqrt((double)h * w / MaxPixels);
                hb = (int)(Math.Floor(h / beta / Factor) * Factor);
                wb = (int)(Math.Floor(w / beta / Factor) * Factor);
            }
            else if (pixels < MinPixels)
            {
                double beta = Math.Sqrt((double)MinPixels / ((double)h * w));
                hb = (int)(Math.Ceiling(h * beta / Factor) * Factor);
                wb = (int)(Math.Ceiling(w * beta / Factor) * Factor);
            }
            return (Math.Max(Factor, hb), Math.Max(Factor, wb));
        }

        // [gridH*gridW, 1176], merge-grouped sequence, per-patch layout [c, t(2), ph, pw].
        private static float[] Patchify(RgbImage img, int gridH, int gridW)
        {
            int W = img.Width, seq = gridH * gridW;
            var px = img.Pixels;     // HWC [0,1]
            var outp = new float[(long)seq * 1176];
            int bh = gridH / 2, bw = gridW / 2;
            for (int byi = 0; byi < bh; byi++)
                for (int bxi = 0; bxi < bw; bxi++)
                    for (int mh = 0; mh < Merge; mh++)
                        for (int mw = 0; mw < Merge; mw++)
                        {
                            int gy = byi * Merge + mh, gx = bxi * Merge + mw;
                            int tok = ((byi * bw + bxi) * Merge + mh) * Merge + mw;
                            long tb = (long)tok * 1176;
                            int py0 = gy * Patch, px0 = gx * Patch;
                            for (int c = 0; c < 3; c++)
                                for (int ph = 0; ph < Patch; ph++)
                                    for (int pw = 0; pw < Patch; pw++)
                                    {
                                        int y = py0 + ph, x = px0 + pw;
                                        float v = (px[((long)y * W + x) * 3 + c] - Mean[c]) / Std[c];
                                        // temporal duplicate (t=0,1)
                                        long b = tb + ((long)c * 2 * Patch * Patch) + (long)ph * Patch + pw;
                                        outp[b] = v;                          // t=0
                                        outp[b + (long)Patch * Patch] = v;    // t=1
                                    }
                        }
            return outp;
        }
    }
}
