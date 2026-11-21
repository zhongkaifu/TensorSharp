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
    /// A request's Gaussian noise as one stream: each <see cref="Next"/> continues where the last one ended, the way
    /// successive <c>torch.randn(..., generator=g)</c> calls on one generator do. A family's order of draws (an edit's
    /// reference posteriors, then the target noise, then the reference augmentation, for diffusers' MiniMax-H3-Image)
    /// is therefore part of what a seed reproduces.
    /// </summary>
    public interface INoiseSource
    {
        /// <summary>The next <paramref name="count"/> standard-normal float32 values of the stream.</summary>
        float[] Next(long count);
    }

    /// <summary>
    /// <c>torch.randn</c> on the CPU generator for float32 tensors, as a stream: a 32-bit Mersenne Twister seeded
    /// with <c>torch.manual_seed</c> (which keeps the seed's low 32 bits), 24-bit uniforms, and the Box-Muller
    /// transform over blocks of 16 (pairs j and j+8), the last block drawn again when a call's count is not a
    /// multiple of 16. The generator's state carries across calls, so <c>Next(a)</c> then <c>Next(b)</c> are the two
    /// draws of one generator. One draw from a fresh stream (<see cref="Normal"/>) is the noise ComfyUI's KSampler
    /// starts from (comfy.sample.prepare_noise), so a TensorSharp image at the same seed starts from the same latent.
    /// PyTorch's AVX2 build evaluates log/sin/cos with SLEEF's 1-ULP functions and its other builds with libm, so a
    /// value can differ from either in the last bit.
    /// </summary>
    public sealed class TorchCpuRandn : INoiseSource
    {
        private readonly Mt19937 _mt;

        public TorchCpuRandn(ulong seed)
        {
            Seed = seed;
            _mt = new Mt19937((uint)seed);
        }

        /// <summary>The seed as <c>torch.manual_seed</c> was given it (only its low 32 bits seed the generator).</summary>
        public ulong Seed { get; }

        /// <summary>32-bit words the stream has drawn: a call of n values draws n, plus 16 when n is not a multiple of 16.</summary>
        public long WordsDrawn => _mt.Drawn;

        /// <summary>One draw from a fresh generator: <c>torch.randn(count, generator=torch.manual_seed(seed))</c>.</summary>
        public static float[] Normal(long count, ulong seed)
        {
            Check(count);
            return new TorchCpuRandn(seed).Next(count);
        }

        /// <summary>The next <paramref name="count"/> values (16 or more: torch draws fewer through its scalar
        /// <c>normal_distribution&lt;double&gt;</c>, which this stream does not reproduce).</summary>
        public float[] Next(long count)
        {
            Check(count);
            var data = new float[count];
            for (long i = 0; i < count; i++) data[i] = Uniform(_mt);
            for (long i = 0; i + 16 <= count; i += 16) Fill16(data, i);
            if (count % 16 != 0)
            {
                long last = count - 16;
                for (int j = 0; j < 16; j++) data[last + j] = Uniform(_mt);
                Fill16(data, last);
            }
            return data;
        }

        private static void Check(long count)
        {
            if (count < 16)
                throw new ArgumentOutOfRangeException(nameof(count), "only the vectorized (16 or more values) path is reproduced.");
        }

        private static float Uniform(Mt19937 mt) => (mt.Next() & 0xFFFFFFu) * (1.0f / 16777216f);

        private static void Fill16(float[] d, long o)
        {
            for (int j = 0; j < 8; j++)
            {
                float u1 = 1f - d[o + j];
                float u2 = d[o + j + 8];
                float radius = MathF.Sqrt(-2f * MathF.Log(u1));
                float theta = (float)(2.0 * Math.PI * u2);
                d[o + j] = radius * MathF.Cos(theta);
                d[o + j + 8] = radius * MathF.Sin(theta);
            }
        }

        private sealed class Mt19937
        {
            private readonly uint[] _state = new uint[624];
            private int _index = 624;

            public Mt19937(uint seed)
            {
                _state[0] = seed;
                for (int i = 1; i < 624; i++)
                    _state[i] = unchecked(1812433253u * (_state[i - 1] ^ (_state[i - 1] >> 30)) + (uint)i);
            }

            public long Drawn { get; private set; }

            public uint Next()
            {
                if (_index >= 624) Twist();
                Drawn++;
                uint y = _state[_index++];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9D2C5680u;
                y ^= (y << 15) & 0xEFC60000u;
                y ^= y >> 18;
                return y;
            }

            private void Twist()
            {
                for (int i = 0; i < 624; i++)
                {
                    uint y = (_state[i] & 0x80000000u) | (_state[(i + 1) % 624] & 0x7FFFFFFFu);
                    uint v = _state[(i + 397) % 624] ^ (y >> 1);
                    if ((y & 1u) != 0) v ^= 0x9908B0DFu;
                    _state[i] = v;
                }
                _index = 0;
            }
        }
    }
}
