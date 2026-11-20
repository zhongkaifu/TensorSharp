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
        /// The arithmetic is ggml_compute_forward_upscale_f32's GGML_SCALE_MODE_BILINEAR branch (no antialias, no
        /// align-corners flag), which uses half-pixel centres: sf = out / in, src = (dst + 0.5) / sf - 0.5, floor, clamp BOTH
        /// neighbours to [0, in - 1] and only THEN take the fraction against the clamped low index (clamped again to [0, 1]).
        /// </para>
        /// </summary>
        internal static unsafe void ResamplePositionTable(float* table, int side, int hiddenSize, int gridW, int gridH,
            int[] rasterOrder, float* destination)
        {
            ArgumentNullException.ThrowIfNull(rasterOrder);
            int numTokens = rasterOrder.Length;
            // The table's column axis runs over the grid's height, its row axis over the width.
            float sfx = (float)gridH / side;
            float sfy = (float)gridW / side;
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
                    int* order = (int*)orderL;

                    // (gx, gy): where the patch reads the table, column then row.
                    int raster = order[i];
                    int gy = raster / gridH;
                    int gx = raster - gy * gridH;

                    float xf = ((float)gx + 0.5f) / sfx - 0.5f;
                    int x0 = (int)MathF.Floor(xf);
                    int x1 = x0 + 1;
                    x0 = Math.Max(0, Math.Min(x0, side - 1));
                    x1 = Math.Max(0, Math.Min(x1, side - 1));
                    float dx = Math.Clamp(xf - x0, 0f, 1f);

                    float yf = ((float)gy + 0.5f) / sfy - 0.5f;
                    int y0 = (int)MathF.Floor(yf);
                    int y1 = y0 + 1;
                    y0 = Math.Max(0, Math.Min(y0, side - 1));
                    y1 = Math.Max(0, Math.Min(y1, side - 1));
                    float dy = Math.Clamp(yf - y0, 0f, 1f);

                    float w00 = (1f - dx) * (1f - dy);
                    float w10 = dx * (1f - dy);
                    float w01 = (1f - dx) * dy;
                    float w11 = dx * dy;

                    float* p00 = src + (long)(y0 * side + x0) * hiddenSize;
                    float* p10 = src + (long)(y0 * side + x1) * hiddenSize;
                    float* p01 = src + (long)(y1 * side + x0) * hiddenSize;
                    float* p11 = src + (long)(y1 * side + x1) * hiddenSize;

                    var v00 = new Vector<float>(w00);
                    var v10 = new Vector<float>(w10);
                    var v01 = new Vector<float>(w01);
                    var v11 = new Vector<float>(w11);

                    int d = 0;
                    for (; d <= hiddenSize - vLen; d += vLen)
                    {
                        var a = TensorComputePrimitives.LoadVector(p00 + d);
                        var b = TensorComputePrimitives.LoadVector(p10 + d);
                        var c = TensorComputePrimitives.LoadVector(p01 + d);
                        var e = TensorComputePrimitives.LoadVector(p11 + d);
                        TensorComputePrimitives.StoreVector(dstRow + d, a * v00 + b * v10 + c * v01 + e * v11);
                    }
                    for (; d < hiddenSize; d++)
                        dstRow[d] = w00 * p00[d] + w10 * p10[d] + w01 * p01[d] + w11 * p11[d];
                });
            }
        }
    }
}
