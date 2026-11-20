// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Numerics;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// Host-side geometry of the vision tower of the <c>muse-glimmer</c> projector type (Muse-Glimmer and Muse Spark): the
    /// learned position grid resampled to an image's patch grid.
    /// </summary>
    internal static class MuseGlimmerVisionGrid
    {
        /// <summary>
        /// The learned [side * side, hidden] position table bilinearly resampled to a <paramref name="gridW"/> x
        /// <paramref name="gridH"/> patch grid: one row of <paramref name="destination"/> per entry of
        /// <paramref name="rasterOrder"/> (raster patch indices y * gridW + x).
        /// <para>
        /// The model was trained with the table resampled to the TRANSPOSED grid: the reference passes
        /// <c>meshgrid(pos_h, pos_w, indexing="xy")</c> to <c>grid_sample</c>, whose first coordinate is the width, so the
        /// table's rows are stretched over gridW and its columns over gridH, and the patches (raster order over the
        /// gridW x gridH grid) read that result in its own raster order: patch t reads table row t / gridH, column t % gridH.
        /// On a square grid that is the patch's own (row, column); on any other grid it is a different walk. HF transformers
        /// 7394e56 (#50847: <c>get_vision_interpolation_indices_and_weights</c> on <c>grid_thw[:, [0, 2, 1]]</c>) and
        /// llama.cpp d3c3e242c (#32433: <c>ggml_interpolate(..., n_patches_y, n_patches_x, ...)</c>) read it so; the order
        /// before them (the table's rows over the grid's rows) differs from the trained one on every non-square image
        /// (MuseGlimmerVisionPositionGridTests).
        /// </para>
        /// <para>
        /// The arithmetic is the reference's (transformers' <c>_interpolation_axis_taps_weights</c>, bilinear,
        /// align_corners=False, padding "zeros" as <c>grid_sample</c>'s default; vLLM's and SGLang's ports do the same):
        /// half-pixel centres, src = (i + 0.5) * side / size - 0.5, taps floor(src) and floor(src) + 1 weighted 1 - |src -
        /// tap|, and a tap off the table weighs nothing. That last rule matters only where a side of the grid has more patches
        /// than the table (more than 32, above 448 px): the edge rows there take less of the table than the clamped bilinear
        /// of ggml_interpolate gives them (llama.cpp d3c3e242c clamps, so it differs from the reference on those grids).
        /// </para>
        /// </summary>
        internal static unsafe void ResamplePositionTable(float* table, int side, int hiddenSize, int gridW, int gridH,
            int[] rasterOrder, float* destination)
        {
            ArgumentNullException.ThrowIfNull(rasterOrder);
            int numTokens = rasterOrder.Length;
            int vLen = Vector<float>.Count;

            fixed (int* orderSrc = rasterOrder)
            {
                long tableL = (long)table;
                long dstL = (long)destination;
                long orderL = (long)orderSrc;

                Parallel.For(0, numTokens, i =>
                {
                    float* src = (float*)tableL;
                    float* dstRow = (float*)dstL + (long)i * hiddenSize;
                    int raster = ((int*)orderL)[i];

                    // The table's row axis runs over the grid's width, its column axis over the height.
                    int row = raster / gridH;
                    int col = raster - row * gridH;
                    Taps(row, gridW, side, out int r0, out float wr0, out float wr1);
                    Taps(col, gridH, side, out int c0, out float wc0, out float wc1);
                    int r1 = Math.Min(r0 + 1, side - 1), c1 = Math.Min(c0 + 1, side - 1);
                    r0 = Math.Max(r0, 0);
                    c0 = Math.Max(c0, 0);

                    // The reference's tap order: (r0, c0), (r0, c1), (r1, c0), (r1, c1).
                    float w00 = wr0 * wc0, w01 = wr0 * wc1, w10 = wr1 * wc0, w11 = wr1 * wc1;
                    float* p00 = src + (long)(r0 * side + c0) * hiddenSize;
                    float* p01 = src + (long)(r0 * side + c1) * hiddenSize;
                    float* p10 = src + (long)(r1 * side + c0) * hiddenSize;
                    float* p11 = src + (long)(r1 * side + c1) * hiddenSize;

                    var v00 = new Vector<float>(w00);
                    var v01 = new Vector<float>(w01);
                    var v10 = new Vector<float>(w10);
                    var v11 = new Vector<float>(w11);

                    int d = 0;
                    for (; d <= hiddenSize - vLen; d += vLen)
                    {
                        var a = TensorComputePrimitives.LoadVector(p00 + d);
                        var b = TensorComputePrimitives.LoadVector(p01 + d);
                        var c = TensorComputePrimitives.LoadVector(p10 + d);
                        var e = TensorComputePrimitives.LoadVector(p11 + d);
                        TensorComputePrimitives.StoreVector(dstRow + d, a * v00 + b * v01 + c * v10 + e * v11);
                    }
                    for (; d < hiddenSize; d++)
                        dstRow[d] = p00[d] * w00 + p01[d] * w01 + p10[d] * w10 + p11[d] * w11;
                });
            }
        }

        /// <summary>One axis of the reference's bilinear taps: grid index <paramref name="index"/> of <paramref name="size"/>
        /// over a table side of <paramref name="side"/>; the first tap and the two taps' weights (0 for a tap off the table).</summary>
        private static void Taps(int index, int size, int side, out int first, out float w0, out float w1)
        {
            float src = (index + 0.5f) * side / size - 0.5f;
            float floor = MathF.Floor(src);
            first = (int)floor;
            w0 = MathF.Max(1f - MathF.Abs(src - floor), 0f);
            w1 = MathF.Max(1f - MathF.Abs(src - floor - 1f), 0f);
            if (first < 0 || first > side - 1) w0 = 0f;
            if (first + 1 < 0 || first + 1 > side - 1) w1 = 0f;
        }
    }
}
