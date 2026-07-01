// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using System;
using System.Runtime.InteropServices;

namespace TensorSharp
{
    public enum DType
    {
        Float32 = 0,
        Float16 = 1,
        Float64 = 2,
        Int32 = 3,
        UInt8 = 4,
        // Block-quantized 8-bit float (GGML Q8_0). 32 elements per block, 34 bytes
        // per block (16-bit scale + 32 int8 values), giving ~1.0625 bytes/elem.
        // Direct element-level access is unsupported - this type is intended for
        // KV-cache storage where reads/writes go through the native GGML kernels
        // (which understand the block layout natively).
        Q8_0 = 5,
        // Block-quantized 4-bit float (GGML Q4_0). 32 elements per block, 18 bytes
        // per block (16-bit scale + 32 int4 values packed two-per-byte), giving
        // ~0.5625 bytes/elem - half the footprint of Q8_0. Same restriction as
        // Q8_0: no direct element access; reads/writes go through the native GGML
        // kernels (ggml_cpy F32<->Q4_0 and flash_attn_ext both handle Q4_0 K/V).
        Q4_0 = 6,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Half
    {
        public ushort value;
    }


    public static class DTypeExtensions
    {
        public static int Size(this DType value)
        {
            switch (value)
            {
                case DType.Float16: return 2;
                case DType.Float32: return 4;
                case DType.Float64: return 8;
                case DType.Int32: return 4;
                case DType.UInt8: return 1;
                // Q8_0 has a fractional 1.0625 bytes/elem due to its block layout.
                // We return 1 here for legacy callers that scale offsets - actual
                // buffer allocation must use Q8_0Bytes() for block-aligned sizing.
                case DType.Q8_0: return 1;
                // Q4_0 is ~0.5625 bytes/elem (block layout). Like Q8_0 we return a
                // nominal 1 for legacy offset-scaling callers; real buffer sizing
                // must go through Q4_0Bytes() for block-aligned allocation.
                case DType.Q4_0: return 1;
                default:
                    throw new NotSupportedException("Element type " + value + " not supported.");
            }
        }

        /// <summary>
        /// Total bytes for storing <paramref name="elementCount"/> Q8_0 elements,
        /// rounded up to the 32-element block boundary (1 block = 34 bytes:
        /// 2-byte F16 scale + 32 int8 quants).
        /// </summary>
        public static long Q8_0Bytes(long elementCount)
        {
            const int blockElems = 32;
            const int blockBytes = 34;
            long blocks = (elementCount + blockElems - 1) / blockElems;
            return blocks * blockBytes;
        }

        /// <summary>
        /// Total bytes for storing <paramref name="elementCount"/> Q4_0 elements,
        /// rounded up to the 32-element block boundary (1 block = 18 bytes:
        /// 2-byte F16 scale + 32 int4 quants packed two-per-byte).
        /// </summary>
        public static long Q4_0Bytes(long elementCount)
        {
            const int blockElems = 32;
            const int blockBytes = 18;
            long blocks = (elementCount + blockElems - 1) / blockElems;
            return blocks * blockBytes;
        }

        public static long ByteLengthFor(this DType value, long elementCount)
        {
            if (value == DType.Q8_0)
                return Q8_0Bytes(elementCount);
            if (value == DType.Q4_0)
                return Q4_0Bytes(elementCount);
            return elementCount * value.Size();
        }

        public static Type ToCLRType(this DType value)
        {
            switch (value)
            {
                case DType.Float16: return typeof(Half);
                case DType.Float32: return typeof(float);
                case DType.Float64: return typeof(double);
                case DType.Int32: return typeof(int);
                case DType.UInt8: return typeof(byte);
                default:
                    throw new NotSupportedException("Element type " + value + " not supported.");
            }
        }
    }

    public static class DTypeBuilder
    {
        public static DType FromCLRType(Type type)
        {
            if (type == typeof(Half))
            {
                return DType.Float16;
            }
            else if (type == typeof(float))
            {
                return DType.Float32;
            }
            else if (type == typeof(double))
            {
                return DType.Float64;
            }
            else if (type == typeof(int))
            {
                return DType.Int32;
            }
            else if (type == typeof(byte))
            {
                return DType.UInt8;
            }
            else
            {
                throw new NotSupportedException("No corresponding DType value for CLR type " + type);
            }
        }
    }
}
