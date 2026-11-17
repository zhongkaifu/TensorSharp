// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Numerics;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// The speech sampler's random stream: std::mt19937 with libc++'s <c>std::shuffle</c>,
    /// <c>uniform_int_distribution&lt;ptrdiff_t&gt;</c> and <c>uniform_real_distribution&lt;double&gt;</c>, draw for draw. A
    /// seed therefore gives the codebook orders and sampling draws llama.cpp's LFM3 pipeline (mtmd-helper-gen.cpp) takes when
    /// it is built against libc++ (clang on macOS); libstdc++ builds draw their distributions differently.
    /// </summary>
    public sealed class Lfm3SpeechRandom
    {
        private const int N = 624, M = 397;
        private readonly uint[] _mt = new uint[N];
        private int _index;

        public Lfm3SpeechRandom(uint seed)
        {
            _mt[0] = seed;
            for (int i = 1; i < N; i++) _mt[i] = 1812433253u * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + (uint)i;
            _index = N;
        }

        /// <summary>The next 32-bit output of mt19937.</summary>
        public uint NextUInt32()
        {
            if (_index >= N)
            {
                for (int i = 0; i < N; i++)
                {
                    uint y = (_mt[i] & 0x80000000u) | (_mt[(i + 1) % N] & 0x7fffffffu);
                    _mt[i] = _mt[(i + M) % N] ^ (y >> 1) ^ ((y & 1u) != 0 ? 0x9908b0dfu : 0u);
                }
                _index = 0;
            }
            uint z = _mt[_index++];
            z ^= z >> 11;
            z ^= (z << 7) & 0x9d2c5680u;
            z ^= (z << 15) & 0xefc60000u;
            z ^= z >> 18;
            return z;
        }

        /// <summary>
        /// libc++ <c>uniform_int_distribution&lt;ptrdiff_t&gt;(a, b)</c>: the fewest whole bits covering the range from one
        /// engine output (its low bits), rejected until in range (ranges past 32 bits are not used here).
        /// </summary>
        public long UniformInt(long a, long b)
        {
            if (b < a) throw new ArgumentOutOfRangeException(nameof(b));
            ulong range = (ulong)(b - a) + 1;
            if (range == 1) return a;
            int bits = 64 - BitOperations.LeadingZeroCount(range) - 1;
            if ((range & (ulong.MaxValue >> (64 - bits))) != 0) bits++;
            if (bits > 32) throw new NotSupportedException("ranges past 32 bits");
            uint mask = bits == 32 ? uint.MaxValue : (1u << bits) - 1;
            ulong u;
            do u = NextUInt32() & mask; while (u >= range);
            return a + (long)u;
        }

        /// <summary>libc++ <c>std::shuffle</c>: for i from the front, swap with i + uniform(0, n - 1 - i).</summary>
        public void Shuffle(Span<int> items)
        {
            int d = items.Length;
            if (d <= 1) return;
            d--;
            for (int first = 0; first < items.Length - 1; first++, d--)
            {
                int i = (int)UniformInt(0, d);
                if (i != 0) (items[first], items[first + i]) = (items[first + i], items[first]);
            }
        }

        /// <summary>libc++ <c>uniform_real_distribution&lt;double&gt;(0, 1)</c>: generate_canonical over two outputs,
        /// (a + b 2^32) / 2^64 in double.</summary>
        public double NextDouble()
        {
            const double R = 4294967296.0;
            double sp = NextUInt32();
            sp += NextUInt32() * R;
            return sp / (R * R);
        }
    }
}
