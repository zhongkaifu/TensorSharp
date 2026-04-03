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
using System.Runtime.InteropServices;

namespace InferenceEngine
{
    [StructLayout(LayoutKind.Sequential)]
    public struct half
    {
        public ushort x;

        public half(float value)
        {
            x = FloatToHalf(value);
        }

        public static implicit operator float(half h)
        {
            return HalfToFloat(h.x);
        }

        public static implicit operator half(float f)
        {
            return new half(f);
        }

        private static ushort FloatToHalf(float value)
        {
            int bits = System.BitConverter.SingleToInt32Bits(value);
            int sign = (bits >> 16) & 0x8000;
            int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
            int mantissa = bits & 0x7FFFFF;

            if (exponent <= 0)
                return (ushort)sign;
            if (exponent > 30)
                return (ushort)(sign | 0x7C00);

            return (ushort)(sign | (exponent << 10) | (mantissa >> 13));
        }

        private static float HalfToFloat(ushort value)
        {
            int sign = (value >> 15) & 1;
            int exponent = (value >> 10) & 0x1F;
            int mantissa = value & 0x3FF;

            if (exponent == 0)
            {
                if (mantissa == 0) return sign == 0 ? 0f : -0f;
                float result = mantissa / 1024f * (1f / 16384f);
                return sign == 0 ? result : -result;
            }
            if (exponent == 31)
            {
                return mantissa == 0
                    ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity)
                    : float.NaN;
            }

            float val = (1f + mantissa / 1024f) * MathF.Pow(2, exponent - 15);
            return sign == 0 ? val : -val;
        }
    }
}
