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
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace InferenceEngine
{
    internal static class ManagedQuantizedOps
    {
        private const int QK4_0 = 32;
        private const int QK4_1 = 32;
        private const int QK5_0 = 32;
        private const int QK5_1 = 32;
        private const int QK8_0 = 32;
        private const int QK8_1 = 32;
        private const int QK4_NL = 32;
        private const int QK_MXFP4 = 32;
        private const int QK_K = 256;
        private const int K_SCALE_SIZE = 12;

        private static readonly sbyte[] Iq4NlValues =
        {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        };

        private static readonly sbyte[] Mxfp4Values =
        {
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
        };

        public static bool SupportsCpuQuantizedStorage(GgmlTensorType type)
        {
            return type switch
            {
                GgmlTensorType.F16 => true,
                GgmlTensorType.BF16 => true,
                GgmlTensorType.Q4_0 => true,
                GgmlTensorType.Q4_1 => true,
                GgmlTensorType.Q5_0 => true,
                GgmlTensorType.Q5_1 => true,
                GgmlTensorType.Q8_0 => true,
                GgmlTensorType.Q8_1 => true,
                GgmlTensorType.Q4_K => true,
                GgmlTensorType.Q5_K => true,
                GgmlTensorType.Q6_K => true,
                GgmlTensorType.IQ4_NL => true,
                GgmlTensorType.MXFP4 => true,
                _ => false,
            };
        }

        public static bool SupportsDequantization(GgmlTensorType type)
        {
            return type switch
            {
                GgmlTensorType.F32 => true,
                GgmlTensorType.F16 => true,
                GgmlTensorType.BF16 => true,
                GgmlTensorType.I8 => true,
                GgmlTensorType.I16 => true,
                GgmlTensorType.I32 => true,
                GgmlTensorType.I64 => true,
                GgmlTensorType.F64 => true,
                _ => SupportsCpuQuantizedStorage(type),
            };
        }

        public static long RowSize(int ggmlType, long ne)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            long blockSize = GgufFile.GetBlockSize(type);
            if (ne % blockSize != 0)
                throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {ne}.");

            return (ne / blockSize) * GgufFile.GetTypeSize(type);
        }

        public static unsafe void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (byte* srcBase = src)
            fixed (float* dstBase = dst)
            {
                DequantizeToFloat32(type, srcBase + srcOffset, dstBase + dstOffset, numElements);
            }
        }

        public static unsafe void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (float* dstBase = dst)
            {
                DequantizeToFloat32(type, (byte*)src.ToPointer(), dstBase + dstOffset, numElements);
            }
        }

        public static unsafe void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            DequantizeToFloat32(type, (byte*)src.ToPointer(), (float*)dst.ToPointer(), numElements);
        }

        public static unsafe void DequantizeRowToFloat32(int ggmlType, IntPtr src, float* dst, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            DequantizeToFloat32(type, (byte*)src.ToPointer(), dst, numElements);
        }

        public static unsafe void DotRowBatchToFloat32(int ggmlType, byte[] src, int srcOffset,
            float[] inputs, int inputOffset, int inputRowStride, int rowCount, long numElements,
            float[] outputs, int outputOffset)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (byte* srcBase = src)
            fixed (float* inputBase = inputs)
            fixed (float* outputBase = outputs)
            {
                DotRowBatchToFloat32(
                    ggmlType,
                    (IntPtr)(srcBase + srcOffset),
                    inputBase + inputOffset,
                    inputRowStride,
                    rowCount,
                    numElements,
                    outputBase + outputOffset);
            }
        }

        public static unsafe void DotRowBatchToFloat32(int ggmlType, IntPtr src, float* inputs,
            int inputRowStride, int rowCount, long numElements, float* outputs)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");
            if (rowCount < 1)
                throw new ArgumentOutOfRangeException(nameof(rowCount));
            if (inputRowStride < numElements)
                throw new ArgumentOutOfRangeException(nameof(inputRowStride));

            long blockSize = GgufFile.GetBlockSize(type);
            if (numElements % blockSize != 0)
                throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {numElements}.");

            for (int row = 0; row < rowCount; row++)
                outputs[row] = 0.0f;

            if (type == GgmlTensorType.F32)
            {
                float* weight = (float*)src.ToPointer();
                for (int row = 0; row < rowCount; row++)
                    outputs[row] = DotFloat(inputs + (long)row * inputRowStride, weight, (int)numElements);
                return;
            }

            float* scratch = stackalloc float[QK_K];
            byte* chunkPtr = (byte*)src.ToPointer();
            long elementOffset = 0;

            while (elementOffset < numElements)
            {
                int chunkElements = GetDotChunkSize(type, numElements - elementOffset);
                DequantizeToFloat32(type, chunkPtr, scratch, chunkElements);

                float* inputChunk = inputs + elementOffset;
                for (int row = 0; row < rowCount; row++)
                {
                    outputs[row] += DotFloat(inputChunk + (long)row * inputRowStride, scratch, chunkElements);
                }

                chunkPtr += GetDotChunkBytes(type, chunkElements);
                elementOffset += chunkElements;
            }
        }

        private static unsafe void DequantizeToFloat32(GgmlTensorType type, byte* src, float* dst, long numElements)
        {
            switch (type)
            {
                case GgmlTensorType.F32:
                    Buffer.MemoryCopy(src, dst, numElements * sizeof(float), numElements * sizeof(float));
                    return;
                case GgmlTensorType.F16:
                    DequantizeF16(src, dst, numElements);
                    return;
                case GgmlTensorType.BF16:
                    DequantizeBf16(src, dst, numElements);
                    return;
                case GgmlTensorType.I8:
                    DequantizeI8(src, dst, numElements);
                    return;
                case GgmlTensorType.I16:
                    DequantizeI16(src, dst, numElements);
                    return;
                case GgmlTensorType.I32:
                    DequantizeI32(src, dst, numElements);
                    return;
                case GgmlTensorType.I64:
                    DequantizeI64(src, dst, numElements);
                    return;
                case GgmlTensorType.F64:
                    DequantizeF64(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_0:
                    DequantizeQ40(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_1:
                    DequantizeQ41(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_0:
                    DequantizeQ50(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_1:
                    DequantizeQ51(src, dst, numElements);
                    return;
                case GgmlTensorType.Q8_0:
                    DequantizeQ80(src, dst, numElements);
                    return;
                case GgmlTensorType.Q8_1:
                    DequantizeQ81(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_K:
                    DequantizeQ4K(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_K:
                    DequantizeQ5K(src, dst, numElements);
                    return;
                case GgmlTensorType.Q6_K:
                    DequantizeQ6K(src, dst, numElements);
                    return;
                case GgmlTensorType.IQ4_NL:
                    DequantizeIq4Nl(src, dst, numElements);
                    return;
                case GgmlTensorType.MXFP4:
                    DequantizeMxfp4(src, dst, numElements);
                    return;
                default:
                    throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");
            }
        }

        private static unsafe void DequantizeF16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = HalfToSingle(ReadUInt16(src + i * 2));
        }

        private static unsafe void DequantizeBf16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
            {
                uint bits = (uint)ReadUInt16(src + i * 2) << 16;
                dst[i] = BitConverter.Int32BitsToSingle((int)bits);
            }
        }

        private static unsafe void DequantizeI8(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ((sbyte*)src)[i];
        }

        private static unsafe void DequantizeI16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = (short)ReadUInt16(src + i * 2);
        }

        private static unsafe void DequantizeI32(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ReadInt32(src + i * 4);
        }

        private static unsafe void DequantizeI64(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ReadInt64(src + i * 8);
        }

        private static unsafe void DequantizeF64(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = (float)ReadDouble(src + i * 8);
        }

        private static unsafe void DequantizeQ40(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_0 != 0)
                throw new NotSupportedException($"Q4_0 requires {QK4_0}-element alignment, got {numElements}.");

            int nb = (int)(numElements / QK4_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * (2 + QK4_0 / 2);
                float d = HalfToSingle(ReadUInt16(block));
                byte* qs = block + 2;
                float* y = dst + i * QK4_0;
                for (int j = 0; j < QK4_0 / 2; j++)
                {
                    int x0 = (qs[j] & 0x0F) - 8;
                    int x1 = (qs[j] >> 4) - 8;
                    y[j] = x0 * d;
                    y[j + QK4_0 / 2] = x1 * d;
                }
            }
        }

        private static unsafe void DequantizeQ41(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_1 != 0)
                throw new NotSupportedException($"Q4_1 requires {QK4_1}-element alignment, got {numElements}.");

            int nb = (int)(numElements / QK4_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * (4 + QK4_1 / 2);
                float d = HalfToSingle(ReadUInt16(block));
                float m = HalfToSingle(ReadUInt16(block + 2));
                byte* qs = block + 4;
                float* y = dst + i * QK4_1;
                for (int j = 0; j < QK4_1 / 2; j++)
                {
                    int x0 = qs[j] & 0x0F;
                    int x1 = qs[j] >> 4;
                    y[j] = x0 * d + m;
                    y[j + QK4_1 / 2] = x1 * d + m;
                }
            }
        }

        private static unsafe void DequantizeQ50(byte* src, float* dst, long numElements)
        {
            if (numElements % QK5_0 != 0)
                throw new NotSupportedException($"Q5_0 requires {QK5_0}-element alignment, got {numElements}.");

            int blockBytes = 2 + 4 + QK5_0 / 2;
            int nb = (int)(numElements / QK5_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                uint qh = ReadUInt32(block + 2);
                byte* qs = block + 6;
                float* y = dst + i * QK5_0;
                for (int j = 0; j < QK5_0 / 2; j++)
                {
                    int xh0 = (int)(((qh >> j) << 4) & 0x10);
                    int xh1 = (int)((qh >> (j + 12)) & 0x10);
                    int x0 = ((qs[j] & 0x0F) | xh0) - 16;
                    int x1 = ((qs[j] >> 4) | xh1) - 16;
                    y[j] = x0 * d;
                    y[j + QK5_0 / 2] = x1 * d;
                }
            }
        }

        private static unsafe void DequantizeQ51(byte* src, float* dst, long numElements)
        {
            if (numElements % QK5_1 != 0)
                throw new NotSupportedException($"Q5_1 requires {QK5_1}-element alignment, got {numElements}.");

            int blockBytes = 4 + 4 + QK5_1 / 2;
            int nb = (int)(numElements / QK5_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float m = HalfToSingle(ReadUInt16(block + 2));
                uint qh = ReadUInt32(block + 4);
                byte* qs = block + 8;
                float* y = dst + i * QK5_1;
                for (int j = 0; j < QK5_1 / 2; j++)
                {
                    int xh0 = (int)(((qh >> j) << 4) & 0x10);
                    int xh1 = (int)((qh >> (j + 12)) & 0x10);
                    int x0 = (qs[j] & 0x0F) | xh0;
                    int x1 = (qs[j] >> 4) | xh1;
                    y[j] = x0 * d + m;
                    y[j + QK5_1 / 2] = x1 * d + m;
                }
            }
        }

        private static unsafe void DequantizeQ80(byte* src, float* dst, long numElements)
        {
            if (numElements % QK8_0 != 0)
                throw new NotSupportedException($"Q8_0 requires {QK8_0}-element alignment, got {numElements}.");

            int blockBytes = 2 + QK8_0;
            int nb = (int)(numElements / QK8_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                sbyte* qs = (sbyte*)(block + 2);
                float* y = dst + i * QK8_0;
                for (int j = 0; j < QK8_0; j++)
                    y[j] = qs[j] * d;
            }
        }

        private static unsafe void DequantizeQ81(byte* src, float* dst, long numElements)
        {
            if (numElements % QK8_1 != 0)
                throw new NotSupportedException($"Q8_1 requires {QK8_1}-element alignment, got {numElements}.");

            int blockBytes = 4 + QK8_1;
            int nb = (int)(numElements / QK8_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                sbyte* qs = (sbyte*)(block + 4);
                float* y = dst + i * QK8_1;
                for (int j = 0; j < QK8_1; j++)
                    y[j] = qs[j] * d;
            }
        }

        private static unsafe void DequantizeQ4K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q4_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 4 + K_SCALE_SIZE + QK_K / 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float min = HalfToSingle(ReadUInt16(block + 2));
                byte* scales = block + 4;
                byte* q = block + 4 + K_SCALE_SIZE;
                float* y = dst + i * QK_K;
                int isIdx = 0;
                for (int j = 0; j < QK_K; j += 64)
                {
                    GetScaleMinK4(isIdx, scales, out byte sc1, out byte m1q);
                    GetScaleMinK4(isIdx + 1, scales, out byte sc2, out byte m2q);
                    float d1 = d * sc1;
                    float d2 = d * sc2;
                    float m1 = min * m1q;
                    float m2 = min * m2q;
                    for (int l = 0; l < 32; l++)
                        y[j + l] = d1 * (q[l] & 0x0F) - m1;
                    for (int l = 0; l < 32; l++)
                        y[j + l + 32] = d2 * (q[l] >> 4) - m2;
                    q += 32;
                    isIdx += 2;
                }
            }
        }

        private static unsafe void DequantizeQ5K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q5_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float min = HalfToSingle(ReadUInt16(block + 2));
                byte* scales = block + 4;
                byte* qh = block + 4 + K_SCALE_SIZE;
                byte* ql = qh + QK_K / 8;
                float* y = dst + i * QK_K;
                int isIdx = 0;
                byte u1 = 1;
                byte u2 = 2;
                for (int j = 0; j < QK_K; j += 64)
                {
                    GetScaleMinK4(isIdx, scales, out byte sc1, out byte m1q);
                    GetScaleMinK4(isIdx + 1, scales, out byte sc2, out byte m2q);
                    float d1 = d * sc1;
                    float d2 = d * sc2;
                    float m1 = min * m1q;
                    float m2 = min * m2q;
                    for (int l = 0; l < 32; l++)
                        y[j + l] = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; l++)
                        y[j + l + 32] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
                    ql += 32;
                    isIdx += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }

        private static unsafe void DequantizeQ6K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q6_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                byte* ql = block;
                byte* qh = ql + QK_K / 2;
                sbyte* scales = (sbyte*)(qh + QK_K / 4);
                float d = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float* y = dst + i * QK_K;

                for (int n = 0; n < QK_K; n += 128)
                {
                    for (int l = 0; l < 32; l++)
                    {
                        int isIdx = l / 16;
                        sbyte q1 = (sbyte)(((ql[l] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)) - 32);
                        sbyte q2 = (sbyte)(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32);
                        sbyte q3 = (sbyte)(((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32);
                        sbyte q4 = (sbyte)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32);
                        y[n + l] = d * scales[isIdx] * q1;
                        y[n + l + 32] = d * scales[isIdx + 2] * q2;
                        y[n + l + 64] = d * scales[isIdx + 4] * q3;
                        y[n + l + 96] = d * scales[isIdx + 6] * q4;
                    }

                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
            }
        }

        private static unsafe void DequantizeIq4Nl(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_NL != 0)
                throw new NotSupportedException($"IQ4_NL requires {QK4_NL}-element alignment, got {numElements}.");

            int blockBytes = 2 + QK4_NL / 2;
            int nb = (int)(numElements / QK4_NL);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                byte* qs = block + 2;
                float* y = dst + i * QK4_NL;
                for (int j = 0; j < QK4_NL / 2; j++)
                {
                    y[j] = d * Iq4NlValues[qs[j] & 0x0F];
                    y[j + QK4_NL / 2] = d * Iq4NlValues[qs[j] >> 4];
                }
            }
        }

        private static unsafe void DequantizeMxfp4(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_MXFP4 != 0)
                throw new NotSupportedException($"MXFP4 requires {QK_MXFP4}-element alignment, got {numElements}.");

            int blockBytes = 1 + QK_MXFP4 / 2;
            int nb = (int)(numElements / QK_MXFP4);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = E8M0ToFp32Half(block[0]);
                byte* qs = block + 1;
                float* y = dst + i * QK_MXFP4;
                for (int j = 0; j < QK_MXFP4 / 2; j++)
                {
                    y[j] = d * Mxfp4Values[qs[j] & 0x0F];
                    y[j + QK_MXFP4 / 2] = d * Mxfp4Values[qs[j] >> 4];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetDotChunkSize(GgmlTensorType type, long remaining)
        {
            return type switch
            {
                GgmlTensorType.F16 or GgmlTensorType.BF16 or
                GgmlTensorType.I8 or GgmlTensorType.I16 or GgmlTensorType.I32 or
                GgmlTensorType.I64 or GgmlTensorType.F64 => (int)Math.Min(remaining, QK_K),
                _ => (int)Math.Min(remaining, GgufFile.GetBlockSize(type)),
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetDotChunkBytes(GgmlTensorType type, int chunkElements)
        {
            return type switch
            {
                GgmlTensorType.F32 => chunkElements * sizeof(float),
                GgmlTensorType.F16 or GgmlTensorType.BF16 => chunkElements * sizeof(ushort),
                GgmlTensorType.I8 => chunkElements,
                GgmlTensorType.I16 => chunkElements * sizeof(short),
                GgmlTensorType.I32 => chunkElements * sizeof(int),
                GgmlTensorType.I64 or GgmlTensorType.F64 => chunkElements * sizeof(long),
                _ => (int)GgufFile.GetTypeSize(type),
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LoadVec(float* ptr) => Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)ptr);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotFloat(float* lhs, float* rhs, int length)
        {
            int vectorSize = Vector<float>.Count;
            Vector<float> acc0 = Vector<float>.Zero;
            Vector<float> acc1 = Vector<float>.Zero;
            int i = 0;

            for (; i <= length - 2 * vectorSize; i += 2 * vectorSize)
            {
                acc0 += LoadVec(lhs + i) * LoadVec(rhs + i);
                acc1 += LoadVec(lhs + i + vectorSize) * LoadVec(rhs + i + vectorSize);
            }

            Vector<float> acc = acc0 + acc1;
            for (; i <= length - vectorSize; i += vectorSize)
            {
                acc += LoadVec(lhs + i) * LoadVec(rhs + i);
            }

            float sum = Vector.Sum(acc);
            for (; i < length; i++)
            {
                sum += lhs[i] * rhs[i];
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe ushort ReadUInt16(byte* p) => Unsafe.ReadUnaligned<ushort>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe uint ReadUInt32(byte* p) => Unsafe.ReadUnaligned<uint>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe int ReadInt32(byte* p) => Unsafe.ReadUnaligned<int>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ReadInt64(byte* p) => Unsafe.ReadUnaligned<long>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe double ReadDouble(byte* p) => Unsafe.ReadUnaligned<double>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HalfToSingle(ushort value) => (float)BitConverter.UInt16BitsToHalf(value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float E8M0ToFp32Half(byte value)
        {
            uint bits = value < 2 ? 0x00200000u << value : ((uint)value - 1u) << 23;
            return BitConverter.Int32BitsToSingle((int)bits);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
        {
            if (j < 4)
            {
                d = (byte)(q[j] & 63);
                m = (byte)(q[j + 4] & 63);
                return;
            }

            d = (byte)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }
}
