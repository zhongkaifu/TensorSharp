// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>
    /// Standard normals from Philox4x32-10 and Box-Muller, as stable-diffusion.cpp's <c>--rng cuda</c> draws them (Qwen-Image-2.1's
    /// noise, and the seeded draws of LTX-2.5 and MiniMax-H3).
    /// </summary>
    public static class PhiloxNoise
    {
        /// <summary>
        /// <paramref name="count"/> standard normals of <paramref name="stream"/> at Philox word 0 = <paramref name="draw"/>, from
        /// counter index <paramref name="first"/>. sd.cpp's counter is (offset, 0, i, 0); the stream fills the two words it leaves
        /// at zero, so stream 0 is its sequence bit for bit and the key keeps meaning the seed (mixing the stream into the key
        /// would make an edit's noise some other seed's text-to-image noise). For one key Philox maps distinct counters to distinct
        /// 128-bit blocks; Box-Muller reads two of the four words and rounds to float, so two streams' values coincide only by
        /// chance. Word 0 is a sampler's independent repeated draws for one seed and stream; draw 0 is the initial noise.
        /// </summary>
        public static float[] Draw(int count, long seed, ulong stream = 0, uint draw = 0, uint first = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(count);
            var result = new float[count];
            const float inv32 = 2.3283064e-10f;
            const float inv32Tau = inv32 * 6.2831855f;
            uint streamLow = (uint)stream, streamHigh = (uint)(stream >> 32);
            for (int i = 0; i < count; i++)
            {
                uint a = draw, b = streamLow, c = first + (uint)i, d = streamHigh;
                uint k0 = (uint)seed, k1 = (uint)((ulong)seed >> 32);
                for (int round = 0; round < 10; round++)
                {
                    ulong p0 = (ulong)a * 0xD2511F53u, p1 = (ulong)c * 0xCD9E8D57u;
                    (a, b, c, d) = ((uint)(p1 >> 32) ^ b ^ k0, (uint)p1, (uint)(p0 >> 32) ^ d ^ k1, (uint)p0);
                    k0 = unchecked(k0 + 0x9E3779B9u);
                    k1 = unchecked(k1 + 0xBB67AE85u);
                }
                float u = (float)a * inv32 + inv32 / 2;
                float v = (float)b * inv32Tau + inv32Tau / 2;
                result[i] = (float)(Math.Sqrt(-2f * Math.Log(u)) * Math.Sin(v));
            }
            return result;
        }
    }
}
