// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The MiniMax-H3 video VAE's clip decode around the per-tile transformer: the
// spatial tiling with cross-faded seams, the temporal windows with cross-faded
// overlaps, and the conversion to RGB frames. The decode writes each tile's patch
// tokens straight into the window's planes and each frame straight into its
// RgbImage; these tests hold it, bit for bit, to the plain formulation it replaced
// (unpack every tile, blend into a window, slice, cross-fade and copy the windows
// into a clip, then clamp and split the clip into frames).
using System;
using System.Collections.Generic;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using Xunit;

namespace InferenceWeb.Tests
{
    public class MiniMaxH3VaeClipDecodeTests
    {
        private const int Lc = MiniMaxH3VideoVae.LatentChannels;
        private const int Ps = MiniMaxH3VideoVae.PatchSpatial;
        private const int Pt = MiniMaxH3VideoVae.PatchTemporal;
        private const int PatchDim = MiniMaxH3VideoVae.OutChannels * Pt * Ps * Ps;

        // A stand-in for the transformer: every output value depends on the tile's
        // own latents, the token and the position inside the patch, so a tile put in
        // the wrong place, a swapped axis or a wrong ramp changes the result.
        private static float[] FakeTile(float[] tile, int latentT, int latentH, int latentW)
        {
            int tokens = latentT * latentH * latentW;
            Assert.Equal((long)tokens * Lc, tile.LongLength);
            var patches = new float[(long)tokens * PatchDim];
            for (int token = 0; token < tokens; token++)
            {
                float a = tile[(long)token * Lc], b = tile[(long)token * Lc + Lc - 1];
                for (int k = 0; k < PatchDim; k++)
                    patches[(long)token * PatchDim + k] =
                        a * 0.75f + b * 0.25f + MathF.Sin(0.37f * k + 0.011f * token) * 0.5f;
            }
            return patches;
        }

        private static float[] Latent(int latentT, int latentH, int latentW, int seed)
        {
            var rng = new Random(seed);
            var latent = new float[(long)latentT * latentH * latentW * Lc];
            for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 4 - 2);
            return latent;
        }

        public static IEnumerable<object[]> Shapes() => new[]
        {
            new object[] { 1, 8, 8 },      // one latent frame, one tile
            new object[] { 7, 24, 40 },    // one window, 2 x 3 tiles
            new object[] { 10, 20, 36 },   // two windows, the last latent frame repeated
            new object[] { 12, 24, 40 },   // two windows exactly
            new object[] { 17, 16, 48 },   // three windows, abutting tiles forced to overlap
        };

        [Theory]
        [MemberData(nameof(Shapes))]
        public void ClipDecodeMatchesThePlainFormulationBitForBit(int latentT, int latentH, int latentW)
        {
            float[] latent = Latent(latentT, latentH, latentW, seed: latentT * 31 + latentW);
            var reported = new List<(int Done, int Total)>();
            var expectedReported = new List<(int Done, int Total)>();

            float[] expected = Reference.Decode(FakeTile, latent, latentT, latentH, latentW,
                                                (d, n) => expectedReported.Add((d, n)));
            float[] actual = MiniMaxH3VideoVae.DecodeClip(FakeTile, latent, latentT, latentH, latentW,
                                                          (d, n) => reported.Add((d, n)));

            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                    Assert.Fail($"value {i} of {expected.Length}: {actual[i]:R} != {expected[i]:R}");
            Assert.Equal(expectedReported, reported);
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void FrameDecodeMatchesClampingAndSplittingTheClip(int latentT, int latentH, int latentW)
        {
            float[] latent = Latent(latentT, latentH, latentW, seed: latentT * 17 + latentH);
            int height = latentH * Ps, width = latentW * Ps, hw = height * width;
            int decoded = MiniMaxH3VideoVae.DecodedFrameCount(latentT);

            float[] clip = Reference.Decode(FakeTile, latent, latentT, latentH, latentW, null);
            MiniMaxH3VideoVae.ToRgb(clip, decoded, height, width);

            foreach (int maxFrames in new[] { decoded, Math.Max(1, decoded - 3), decoded + 5 })
            {
                RgbImage[] frames = MiniMaxH3VideoVae.DecodeClipFrames(
                    FakeTile, latent, latentT, latentH, latentW, maxFrames);
                Assert.Equal(Math.Min(maxFrames, decoded), frames.Length);
                for (int f = 0; f < frames.Length; f++)
                {
                    Assert.Equal(width, frames[f].Width);
                    Assert.Equal(height, frames[f].Height);
                    float[] px = frames[f].Pixels;
                    for (int c = 0; c < 3; c++)
                        for (int q = 0; q < hw; q++)
                        {
                            float want = clip[((long)c * decoded + f) * hw + q];
                            if (BitConverter.SingleToInt32Bits(want) != BitConverter.SingleToInt32Bits(px[q * 3 + c]))
                                Assert.Fail($"frame {f} channel {c} pixel {q}: {px[q * 3 + c]:R} != {want:R}");
                        }
                }
            }
        }

        [Fact]
        public void FrameDecodeRefusesANonFiniteValue()
        {
            const int latentT = 7, latentH = 8, latentW = 8;
            float[] latent = Latent(latentT, latentH, latentW, seed: 5);
            MiniMaxH3VideoVae.TileDecoder poisoned = (tile, t, h, w) =>
            {
                float[] patches = FakeTile(tile, t, h, w);
                patches[patches.Length / 2] = float.NaN;
                return patches;
            };
            var ex = Assert.Throws<InvalidOperationException>(() =>
                MiniMaxH3VideoVae.DecodeClipFrames(poisoned, latent, latentT, latentH, latentW, maxFrames: 100));
            Assert.Contains("non-finite", ex.Message);
        }

        [Fact]
        public void SegmentsOfAFiveSecondClipAreSeventeenFramesAWindowPlusTheTail()
        {
            // 1344x768 for 5 s at 24 fps is 37 latent frames: 7 windows of 7 latent
            // frames, each giving frames 3..19 of its 28, and the last one's 23..27.
            var segments = MiniMaxH3VideoVae.PlanSegments(37, out int pad, out int chunks, out int total);
            Assert.Equal(0, pad);
            Assert.Equal(7, chunks);
            Assert.Equal(8, segments.Count);
            Assert.Equal(7 * 17 + 5, total);
            Assert.Equal(MiniMaxH3VideoVae.DecodedFrameCount(37), total);
            Assert.Equal(new MiniMaxH3VideoVae.Segment(0, 3, 17, 0, 0), segments[0]);
            Assert.Equal(new MiniMaxH3VideoVae.Segment(1, 3, 17, 5, 23), segments[1]);
            Assert.Equal(new MiniMaxH3VideoVae.Segment(6, 23, 5, 0, 0), segments[7]);
        }

        // The decode as it was written before the windows were accumulated in place:
        // kept here, unchanged apart from taking the tile decoder as an argument.
        private static class Reference
        {
            private const int DecodeTokensPerChunk = 5, DecodeTokenOverlap = 2, DecodeTokenDrop = 3;
            private const int DecodeFramesPerChunk = 20, DecodeFramePrePadding = 3, DecodeFrameOverlap = 5;

            public static float[] Decode(MiniMaxH3VideoVae.TileDecoder decodeTile, float[] latent,
                                         int latentT, int latentH, int latentW, Action<int, int> onTile)
            {
                int expected = MiniMaxH3VideoVae.DecodedFrameCount(latentT);
                long voxel = (long)latentH * latentW * Lc;

                int pseudo = latentT + DecodeTokenDrop;
                int pad = (DecodeTokensPerChunk - pseudo % DecodeTokensPerChunk) % DecodeTokensPerChunk;
                pseudo += pad;
                int chunks = pseudo / DecodeTokensPerChunk - 1;
                if (chunks < 1) { pad += DecodeTokensPerChunk; chunks += 1; }

                int paddedT = latentT + pad;
                float[] padded = latent;
                if (pad > 0)
                {
                    padded = new float[(long)paddedT * voxel];
                    Array.Copy(latent, padded, (long)latentT * voxel);
                    for (int i = 0; i < pad; i++)
                        Array.Copy(latent, (long)(latentT - 1) * voxel,
                                   padded, (long)(latentT + i) * voxel, voxel);
                }

                int height = latentH * Ps, width = latentW * Ps;
                long hw = (long)height * width;
                var result = new List<float[]>();
                var counts = new List<int>();
                float[] overlap = null; int overlapCount = 0;

                for (int i = 0; i < chunks; i++)
                {
                    int start = i * DecodeTokensPerChunk;
                    int end = Math.Min(start + DecodeTokensPerChunk + DecodeTokenOverlap, paddedT);
                    int take = end - start;
                    var chunk = new float[(long)take * voxel];
                    Array.Copy(padded, (long)start * voxel, chunk, 0, (long)take * voxel);

                    int window = i;
                    float[] pixels = DecodeWindow(decodeTile, chunk, take, latentH, latentW,
                        onTile == null ? null : (done, tiles) => onTile(window * tiles + done, chunks * tiles));
                    int decoded = take * Pt;

                    int firstEnd = Math.Min(DecodeFramesPerChunk, decoded);
                    int firstStart = Math.Min(DecodeFramePrePadding, firstEnd);
                    int firstCount = firstEnd - firstStart;
                    float[] first = SliceFrames(pixels, decoded, hw, firstStart, firstCount);

                    if (overlap != null)
                    {
                        BlendTemporal(overlap, overlapCount, first, firstCount, hw, DecodeFrameOverlap);
                        overlap = null; overlapCount = 0;
                    }
                    result.Add(first); counts.Add(firstCount);

                    if (decoded > DecodeFramesPerChunk + DecodeFramePrePadding)
                    {
                        overlapCount = decoded - (DecodeFramesPerChunk + DecodeFramePrePadding);
                        overlap = SliceFrames(pixels, decoded, hw,
                                              DecodeFramesPerChunk + DecodeFramePrePadding, overlapCount);
                    }
                    if (i == chunks - 1 && overlap != null)
                    {
                        result.Add(overlap); counts.Add(overlapCount);
                        overlap = null; overlapCount = 0;
                    }
                }

                int total = 0;
                foreach (int c in counts) total += c;
                int keep = Math.Min(total, expected);
                var outp = new float[3 * (long)keep * hw];
                int written = 0;
                for (int p = 0; p < result.Count && written < keep; p++)
                {
                    int n = Math.Min(counts[p], keep - written);
                    for (int ch = 0; ch < 3; ch++)
                        Array.Copy(result[p], (long)ch * counts[p] * hw,
                                   outp, ((long)ch * keep + written) * hw, (long)n * hw);
                    written += n;
                }
                return outp;
            }

            private static float[] SliceFrames(float[] src, int frames, long hw, int start, int count)
            {
                var dst = new float[3 * (long)count * hw];
                for (int ch = 0; ch < 3; ch++)
                    Array.Copy(src, ((long)ch * frames + start) * hw,
                               dst, (long)ch * count * hw, (long)count * hw);
                return dst;
            }

            private static void BlendTemporal(float[] previous, int previousCount,
                                              float[] current, int currentCount, long hw, int extent)
            {
                extent = Math.Min(extent, Math.Min(previousCount, currentCount));
                int from = previousCount - extent;
                for (int ch = 0; ch < 3; ch++)
                    for (int t = 0; t < extent; t++)
                    {
                        float wb = t / (float)extent, wa = 1f - wb;
                        long dst = ((long)ch * currentCount + t) * hw;
                        long src = ((long)ch * previousCount + (from + t)) * hw;
                        for (long i = 0; i < hw; i++)
                            current[dst + i] = previous[src + i] * wa + current[dst + i] * wb;
                    }
            }

            private static float[] DecodeWindow(MiniMaxH3VideoVae.TileDecoder decodeTile, float[] latent,
                                                int latentT, int latentH, int latentW, Action<int, int> onTile)
            {
                var rows = MiniMaxH3VideoVae.PlanTiles(latentH);
                var cols = MiniMaxH3VideoVae.PlanTiles(latentW);
                int frames = latentT * Pt;
                int height = latentH * Ps, width = latentW * Ps;
                long hw = (long)height * width;
                long plane = (long)frames * hw;

                var acc = new float[3 * plane];
                var weight = new float[plane];
                int done = 0, totalTiles = rows.Count * cols.Count;

                foreach (var r in rows)
                    foreach (var c in cols)
                    {
                        var tile = new float[(long)latentT * r.Length * c.Length * Lc];
                        for (int t = 0; t < latentT; t++)
                            for (int y = 0; y < r.Length; y++)
                                for (int x = 0; x < c.Length; x++)
                                {
                                    long src = (((long)t * latentH + (r.Start + y)) * latentW
                                                + (c.Start + x)) * Lc;
                                    long dst = (((long)t * r.Length + y) * c.Length + x) * Lc;
                                    Array.Copy(latent, src, tile, dst, Lc);
                                }

                        float[] patches = decodeTile(tile, latentT, r.Length, c.Length);
                        float[] px = MiniMaxH3VideoVae.UnpackPatches(patches, latentT, r.Length, c.Length);

                        int th = r.Length * Ps, tw = c.Length * Ps;
                        long thw = (long)th * tw;
                        long tplane = (long)frames * thw;
                        int y0 = r.Start * Ps, x0 = c.Start * Ps;

                        var wy = Ramp(th, r.LeadOverlap * Ps, r.TrailOverlap * Ps);
                        var wx = Ramp(tw, c.LeadOverlap * Ps, c.TrailOverlap * Ps);

                        for (int f = 0; f < frames; f++)
                            for (int y = 0; y < th; y++)
                            {
                                long dstRow = ((long)f * height + (y0 + y)) * width + x0;
                                long srcRow = ((long)f * th + y) * tw;
                                for (int x = 0; x < tw; x++)
                                {
                                    float w = wy[y] * wx[x];
                                    weight[dstRow + x] += w;
                                    for (int ch = 0; ch < 3; ch++)
                                        acc[ch * plane + dstRow + x] += px[ch * tplane + srcRow + x] * w;
                                }
                            }
                        onTile?.Invoke(++done, totalTiles);
                    }

                for (long i = 0; i < plane; i++)
                {
                    float w = weight[i];
                    if (w <= 0f) continue;
                    float inv = 1f / w;
                    for (int ch = 0; ch < 3; ch++) acc[ch * plane + i] *= inv;
                }
                return acc;
            }

            private static float[] Ramp(int length, int lead, int trail)
            {
                var w = new float[length];
                for (int i = 0; i < length; i++)
                {
                    float v = 1f;
                    if (lead > 0 && i < lead) v = (i + 0.5f) / lead;
                    if (trail > 0 && i >= length - trail) v = Math.Min(v, (length - i - 0.5f) / trail);
                    w[i] = Math.Max(v, 1e-6f);
                }
                return w;
            }
        }
    }
}
