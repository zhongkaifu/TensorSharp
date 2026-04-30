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
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace TensorSharp.Models
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
        private const int Q4_0BlockBytes = 2 + QK4_0 / 2;
        private const int Q4_1BlockBytes = 4 + QK4_1 / 2;
        private const int Q5_0BlockBytes = 2 + 4 + QK5_0 / 2;
        private const int Q5_1BlockBytes = 4 + 4 + QK5_1 / 2;
        private const int Q8_0BlockBytes = 2 + QK8_0;
        private const int Q8_1BlockBytes = 4 + QK8_1;
        private const int Q4_KBlockBytes = 4 + K_SCALE_SIZE + QK_K / 2;
        private const int Q5_KBlockBytes = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
        private const int Q6_KBlockBytes = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
        private const int Q8_KBlockBytes = 4 + QK_K + 2 * (QK_K / 16);

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

        public static unsafe bool TryAddmmQuantizedToFloat32(
            int ggmlType,
            IntPtr weights,
            long ne0,
            long ne1,
            float* input,
            int inputRowStride,
            int rowCount,
            float* output,
            int outputRowStride)
        {
            var type = (GgmlTensorType)ggmlType;
            if (ne0 > int.MaxValue || ne1 > int.MaxValue)
                return false;

            if (!TryGetDirectMatMulPlan(type, (int)ne0, out ActivationQuantKind activationKind, out int activationRowBytes))
                return false;

            if (weights == IntPtr.Zero)
                throw new ArgumentException("Quantized weights pointer cannot be null.", nameof(weights));
            if (inputRowStride < ne0)
                throw new ArgumentOutOfRangeException(nameof(inputRowStride));
            if (outputRowStride < ne1)
                throw new ArgumentOutOfRangeException(nameof(outputRowStride));

            long totalActivationBytes = (long)rowCount * activationRowBytes;
            if (totalActivationBytes > int.MaxValue)
                return false;

            byte[] rented = ArrayPool<byte>.Shared.Rent((int)totalActivationBytes);
            try
            {
                fixed (byte* activationBase = rented)
                {
                    for (int row = 0; row < rowCount; row++)
                    {
                        byte* dst = activationBase + (long)row * activationRowBytes;
                        float* src = input + (long)row * inputRowStride;
                        QuantizeActivation(src, dst, (int)ne0, activationKind);
                    }

                    byte* weightBase = (byte*)weights.ToPointer();
                    int weightRowBytes = (int)RowSize(ggmlType, ne0);
                    int outDim = (int)ne1;
                    int inDim = (int)ne0;
                    nint activationAddress = (nint)activationBase;
                    nint weightAddress = (nint)weightBase;
                    nint outputAddress = (nint)output;

                    void ComputeColumnRange(int startCol, int endCol)
                    {
                        byte* activationPtr = (byte*)activationAddress;
                        byte* weightPtr = (byte*)weightAddress;
                        float* outputPtr = (float*)outputAddress;

                        for (int col = startCol; col < endCol; col++)
                        {
                            byte* weightRow = weightPtr + (long)col * weightRowBytes;
                            for (int row = 0; row < rowCount; row++)
                            {
                                byte* activationRow = activationPtr + (long)row * activationRowBytes;
                                outputPtr[(long)row * outputRowStride + col] =
                                    DotQuantized(type, weightRow, activationRow, inDim);
                            }
                        }
                    }

                    bool useParallel = outDim >= 128 && (long)rowCount * outDim >= 512 && Environment.ProcessorCount > 1;
                    if (useParallel)
                    {
                        Parallel.For(0, outDim, col => ComputeColumnRange(col, col + 1));
                    }
                    else
                    {
                        ComputeColumnRange(0, outDim);
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }

            return true;
        }

        public static unsafe bool TryAddmmQuantizedToFloat32(
            int ggmlType,
            byte[] weights,
            int weightsOffset,
            long ne0,
            long ne1,
            float[] input,
            int inputOffset,
            int inputRowStride,
            int rowCount,
            float[] output,
            int outputOffset,
            int outputRowStride)
        {
            if (weights == null)
                throw new ArgumentNullException(nameof(weights));
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (output == null)
                throw new ArgumentNullException(nameof(output));

            fixed (byte* weightPtr = weights)
            fixed (float* inputPtr = input)
            fixed (float* outputPtr = output)
            {
                return TryAddmmQuantizedToFloat32(
                    ggmlType,
                    (IntPtr)(weightPtr + weightsOffset),
                    ne0,
                    ne1,
                    inputPtr + inputOffset,
                    inputRowStride,
                    rowCount,
                    outputPtr + outputOffset,
                    outputRowStride);
            }
        }

        private enum ActivationQuantKind
        {
            Q8_0,
            Q8_1,
            Q8_K,
        }

        private static bool TryGetDirectMatMulPlan(
            GgmlTensorType type,
            int elementCount,
            out ActivationQuantKind activationKind,
            out int activationRowBytes)
        {
            activationKind = default;
            activationRowBytes = 0;

            switch (type)
            {
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q5_0:
                case GgmlTensorType.Q8_0:
                case GgmlTensorType.Q8_1:
                    if (elementCount % QK8_0 != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_0;
                    activationRowBytes = elementCount / QK8_0 * Q8_0BlockBytes;
                    return true;

                case GgmlTensorType.Q4_1:
                case GgmlTensorType.Q5_1:
                    if (elementCount % QK8_1 != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_1;
                    activationRowBytes = elementCount / QK8_1 * Q8_1BlockBytes;
                    return true;

                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                case GgmlTensorType.Q6_K:
                    if (elementCount % QK_K != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_K;
                    activationRowBytes = elementCount / QK_K * Q8_KBlockBytes;
                    return true;

                default:
                    return false;
            }
        }

        private static unsafe void QuantizeActivation(float* src, byte* dst, int elementCount, ActivationQuantKind kind)
        {
            switch (kind)
            {
                case ActivationQuantKind.Q8_0:
                    QuantizeF32ToQ8_0(src, dst, elementCount);
                    return;
                case ActivationQuantKind.Q8_1:
                    QuantizeF32ToQ8_1(src, dst, elementCount);
                    return;
                case ActivationQuantKind.Q8_K:
                    QuantizeF32ToQ8_K(src, dst, elementCount);
                    return;
                default:
                    throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
            }
        }

        private static unsafe float DotQuantized(GgmlTensorType type, byte* weightRow, byte* activationRow, int elementCount)
        {
            return type switch
            {
                GgmlTensorType.Q4_0 => VecDotQ4_0Q8_0(weightRow, activationRow, elementCount / QK4_0),
                GgmlTensorType.Q4_1 => VecDotQ4_1Q8_1(weightRow, activationRow, elementCount / QK4_1),
                GgmlTensorType.Q5_0 => VecDotQ5_0Q8_0(weightRow, activationRow, elementCount / QK5_0),
                GgmlTensorType.Q5_1 => VecDotQ5_1Q8_1(weightRow, activationRow, elementCount / QK5_1),
                GgmlTensorType.Q8_0 => VecDotQ8_0Q8_0(weightRow, activationRow, elementCount / QK8_0),
                GgmlTensorType.Q8_1 => VecDotQ8_1Q8_0(weightRow, activationRow, elementCount / QK8_1),
                GgmlTensorType.Q4_K => VecDotQ4_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.Q5_K => VecDotQ5_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.Q6_K => VecDotQ6_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                _ => throw new NotSupportedException($"Direct managed quantized matmul does not support {type}."),
            };
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

        private static unsafe void QuantizeF32ToQ8_0(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK8_0;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK8_0;
                byte* blockDst = dst + block * Q8_0BlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK8_0);
                float scale = maxAbs / 127.0f;
                WriteHalf(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 2);
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK8_0);
                    continue;
                }

                float invScale = 1.0f / scale;
                for (int i = 0; i < QK8_0; i++)
                    qs[i] = ClampToInt8(MathF.Round(blockSrc[i] * invScale));
            }
        }

        private static unsafe void QuantizeF32ToQ8_1(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK8_1;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK8_1;
                byte* blockDst = dst + block * Q8_1BlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK8_1);
                float scale = maxAbs / 127.0f;
                WriteHalf(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 4);
                int sum = 0;
                if (scale != 0.0f)
                {
                    float invScale = 1.0f / scale;
                    for (int i = 0; i < QK8_1; i++)
                    {
                        sbyte q = ClampToInt8(MathF.Round(blockSrc[i] * invScale));
                        qs[i] = q;
                        sum += q;
                    }
                }
                else
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK8_1);
                }

                WriteHalf(blockDst + 2, scale * sum);
            }
        }

        private static unsafe void QuantizeF32ToQ8_K(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK_K;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK_K;
                byte* blockDst = dst + block * Q8_KBlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK_K);
                float scale = maxAbs / 127.0f;
                Unsafe.WriteUnaligned(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 4);
                short* bsums = (short*)(blockDst + 4 + QK_K);
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK_K);
                    Unsafe.InitBlockUnaligned(bsums, 0, QK_K / 16 * sizeof(short));
                    continue;
                }

                float invScale = 1.0f / scale;
                for (int group = 0; group < QK_K / 16; group++)
                {
                    int sum = 0;
                    int offset = group * 16;
                    for (int i = 0; i < 16; i++)
                    {
                        sbyte q = ClampToInt8(MathF.Round(blockSrc[offset + i] * invScale));
                        qs[offset + i] = q;
                        sum += q;
                    }

                    bsums[group] = (short)sum;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void WriteHalf(byte* dst, float value)
        {
            Unsafe.WriteUnaligned(dst, BitConverter.HalfToUInt16Bits((System.Half)value));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static sbyte ClampToInt8(float value)
        {
            int rounded = (int)value;
            if (rounded > 127) return 127;
            if (rounded < -127) return -127;
            return (sbyte)rounded;
        }

        private static unsafe float MaxAbs(float* src, int length)
        {
            if (Avx512F.IsSupported && length >= 16)
            {
                Vector512<float> max = Vector512<float>.Zero;
                int i = 0;
                for (; i <= length - 16; i += 16)
                    max = Avx512F.Max(max, Vector512.Abs(Avx512F.LoadVector512(src + i)));

                float result = HorizontalMax(max);
                for (; i < length; i++)
                {
                    float abs = MathF.Abs(src[i]);
                    if (abs > result) result = abs;
                }

                return result;
            }

            int vectorSize = Vector<float>.Count;
            Vector<float> maxVec = Vector<float>.Zero;
            int j = 0;
            for (; j <= length - vectorSize; j += vectorSize)
                maxVec = Vector.Max(maxVec, Vector.Abs(LoadVec(src + j)));

            float maxAbs = 0.0f;
            for (int lane = 0; lane < Vector<float>.Count; lane++)
                if (maxVec[lane] > maxAbs) maxAbs = maxVec[lane];

            for (; j < length; j++)
            {
                float abs = MathF.Abs(src[j]);
                if (abs > maxAbs) maxAbs = abs;
            }

            return maxAbs;
        }

        private static unsafe float VecDotQ4_0Q8_0(byte* q4, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q4Block = q4 + block * Q4_0BlockBytes;
                byte* q8Block = q8 + block * Q8_0BlockBytes;
                float d4 = HalfToSingle(ReadUInt16(q4Block));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                byte* qs = q4Block + 2;
                sbyte* qx = (sbyte*)(q8Block + 2);

                int isum = 0;
                for (int i = 0; i < QK4_0 / 2; i++)
                {
                    int low = (qs[i] & 0x0F) - 8;
                    int high = (qs[i] >> 4) - 8;
                    isum += low * qx[i] + high * qx[i + QK4_0 / 2];
                }

                sum += d4 * d8 * isum;
            }

            return sum;
        }

        private static unsafe float VecDotQ4_1Q8_1(byte* q4, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q4Block = q4 + block * Q4_1BlockBytes;
                byte* q8Block = q8 + block * Q8_1BlockBytes;
                float d4 = HalfToSingle(ReadUInt16(q4Block));
                float m4 = HalfToSingle(ReadUInt16(q4Block + 2));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                float s8 = HalfToSingle(ReadUInt16(q8Block + 2));
                byte* qs = q4Block + 4;
                sbyte* qx = (sbyte*)(q8Block + 4);

                int isum = 0;
                for (int i = 0; i < QK4_1 / 2; i++)
                    isum += (qs[i] & 0x0F) * qx[i] + (qs[i] >> 4) * qx[i + QK4_1 / 2];

                sum += d4 * d8 * isum + m4 * s8;
            }

            return sum;
        }

        private static unsafe float VecDotQ5_0Q8_0(byte* q5, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q5Block = q5 + block * Q5_0BlockBytes;
                byte* q8Block = q8 + block * Q8_0BlockBytes;
                float d5 = HalfToSingle(ReadUInt16(q5Block));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                uint qh = ReadUInt32(q5Block + 2);
                byte* qs = q5Block + 6;
                sbyte* qx = (sbyte*)(q8Block + 2);

                int isum = 0;
                for (int i = 0; i < QK5_0 / 2; i++)
                {
                    int xh0 = (int)(((qh >> i) << 4) & 0x10);
                    int xh1 = (int)((qh >> (i + 12)) & 0x10);
                    int x0 = ((qs[i] & 0x0F) | xh0) - 16;
                    int x1 = ((qs[i] >> 4) | xh1) - 16;
                    isum += x0 * qx[i] + x1 * qx[i + QK5_0 / 2];
                }

                sum += d5 * d8 * isum;
            }

            return sum;
        }

        private static unsafe float VecDotQ5_1Q8_1(byte* q5, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q5Block = q5 + block * Q5_1BlockBytes;
                byte* q8Block = q8 + block * Q8_1BlockBytes;
                float d5 = HalfToSingle(ReadUInt16(q5Block));
                float m5 = HalfToSingle(ReadUInt16(q5Block + 2));
                uint qh = ReadUInt32(q5Block + 4);
                byte* qs = q5Block + 8;
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                float s8 = HalfToSingle(ReadUInt16(q8Block + 2));
                sbyte* qx = (sbyte*)(q8Block + 4);

                int isum = 0;
                for (int i = 0; i < QK5_1 / 2; i++)
                {
                    int xh0 = (int)(((qh >> i) << 4) & 0x10);
                    int xh1 = (int)((qh >> (i + 12)) & 0x10);
                    int x0 = (qs[i] & 0x0F) | xh0;
                    int x1 = (qs[i] >> 4) | xh1;
                    isum += x0 * qx[i] + x1 * qx[i + QK5_1 / 2];
                }

                sum += d5 * d8 * isum + m5 * s8;
            }

            return sum;
        }

        private static unsafe float VecDotQ8_0Q8_0(byte* q8w, byte* q8x, int blockCount)
        {
            if (Avx512F.IsSupported && Avx512BW.IsSupported)
                return VecDotQ8_0Q8_0Avx512(q8w, q8x, blockCount);
            if (Avx2.IsSupported)
                return VecDotQ8_0Q8_0Avx2(q8w, q8x, blockCount);

            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float dw = HalfToSingle(ReadUInt16(wb));
                float dx = HalfToSingle(ReadUInt16(xb));
                sbyte* qw = (sbyte*)(wb + 2);
                sbyte* qx = (sbyte*)(xb + 2);

                int isum = 0;
                for (int i = 0; i < QK8_0; i++)
                    isum += qw[i] * qx[i];
                sum += dw * dx * isum;
            }

            return sum;
        }

        private static unsafe float VecDotQ8_1Q8_0(byte* q8w, byte* q8x, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_1BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float dw = HalfToSingle(ReadUInt16(wb));
                float dx = HalfToSingle(ReadUInt16(xb));
                sbyte* qw = (sbyte*)(wb + 4);
                sbyte* qx = (sbyte*)(xb + 2);

                int isum = 0;
                for (int i = 0; i < QK8_1; i++)
                    isum += qw[i] * qx[i];
                sum += dw * dx * isum;
            }

            return sum;
        }

        private static unsafe float VecDotQ8_0Q8_0Avx512(byte* q8w, byte* q8x, int blockCount)
        {
            Vector512<float> acc = Vector512<float>.Zero;
            Vector512<short> ones = Vector512.Create((short)1);

            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float scale = HalfToSingle(ReadUInt16(wb)) * HalfToSingle(ReadUInt16(xb));

                Vector256<sbyte> qwBytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb + 2);
                Vector256<sbyte> qxBytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
                Vector512<short> qw = Avx512BW.ConvertToVector512Int16(qwBytes);
                Vector512<short> qx = Avx512BW.ConvertToVector512Int16(qxBytes);
                Vector512<short> products = Avx512BW.MultiplyLow(qw, qx);
                Vector512<int> pairSums = Avx512BW.MultiplyAddAdjacent(products, ones);
                Vector512<float> dotParts = Avx512F.ConvertToVector512Single(pairSums);

                acc = Avx512F.FusedMultiplyAdd(Vector512.Create(scale), dotParts, acc);
            }

            return HorizontalSum(acc);
        }

        private static unsafe float VecDotQ8_0Q8_0Avx2(byte* q8w, byte* q8x, int blockCount)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            Vector256<short> ones = Vector256.Create((short)1);

            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float scale = HalfToSingle(ReadUInt16(wb)) * HalfToSingle(ReadUInt16(xb));

                Vector256<sbyte> qw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb + 2);
                Vector256<sbyte> qx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
                Vector256<sbyte> absW = Avx2.Sign(qw, qw);
                Vector256<sbyte> signedX = Avx2.Sign(qx, qw);
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absW.AsByte(), signedX);
                Vector256<int> pairSums = Avx2.MultiplyAddAdjacent(prod, ones);
                Vector256<float> dotParts = Avx.ConvertToVector256Single(pairSums);
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(Vector256.Create(scale), dotParts, acc)
                    : Avx.Add(acc, Avx.Multiply(Vector256.Create(scale), dotParts));
            }

            return HorizontalSum(acc);
        }

        private static unsafe float VecDotQ4_KQ8_K(byte* q4k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            byte* scBuf = stackalloc byte[8];
            byte* mnBuf = stackalloc byte[8];

            for (int block = 0; block < superBlockCount; block++)
            {
                float d4 = HalfToSingle(ReadUInt16(q4k));
                float dmin = HalfToSingle(ReadUInt16(q4k + 2));
                UnpackQ4Q5Scales(q4k + 4, scBuf, mnBuf);
                byte* qs = q4k + 16;
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                for (int j = 0; j < 8; j++)
                {
                    int pairIndex = j / 2;
                    bool highNibble = (j & 1) != 0;
                    sbyte* q8Vals = q8Values + j * 32;
                    int prodSum = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int raw = qs[pairIndex * 32 + i];
                        int q = highNibble ? raw >> 4 : raw & 0x0F;
                        prodSum += q * q8Vals[i];
                    }

                    int q8Sum = bsums[j * 2] + bsums[j * 2 + 1];
                    sum += d8 * (d4 * scBuf[j] * prodSum - dmin * mnBuf[j] * q8Sum);
                }

                q4k += Q4_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        private static unsafe float VecDotQ5_KQ8_K(byte* q5k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            byte* scBuf = stackalloc byte[8];
            byte* mnBuf = stackalloc byte[8];

            for (int block = 0; block < superBlockCount; block++)
            {
                float d5 = HalfToSingle(ReadUInt16(q5k));
                float dmin = HalfToSingle(ReadUInt16(q5k + 2));
                UnpackQ4Q5Scales(q5k + 4, scBuf, mnBuf);
                byte* qh = q5k + 16;
                byte* qs = q5k + 48;
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                for (int j = 0; j < 8; j++)
                {
                    int pairIndex = j / 2;
                    bool highNibble = (j & 1) != 0;
                    sbyte* q8Vals = q8Values + j * 32;
                    int prodSum = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int raw = qs[pairIndex * 32 + i];
                        int lo4 = highNibble ? raw >> 4 : raw & 0x0F;
                        int bit5 = (qh[i] >> j) & 1;
                        prodSum += (lo4 | (bit5 << 4)) * q8Vals[i];
                    }

                    int q8Sum = bsums[j * 2] + bsums[j * 2 + 1];
                    sum += d8 * (d5 * scBuf[j] * prodSum - dmin * mnBuf[j] * q8Sum);
                }

                q5k += Q5_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        private static unsafe float VecDotQ6_KQ8_K(byte* q6k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;

            for (int block = 0; block < superBlockCount; block++)
            {
                byte* ql = q6k;
                byte* qh = q6k + QK_K / 2;
                sbyte* scales = (sbyte*)(q6k + QK_K / 2 + QK_K / 4);
                float d6 = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                float scaleBase = d6 * d8;

                for (int sub = 0; sub < 16; sub++)
                {
                    float scale = scaleBase * scales[sub];
                    sbyte* q8Vals = q8Values + sub * 16;
                    int half = sub / 8;
                    int sh = sub % 8;
                    int qlOffset = half * 64 + (sh % 4) * 16;
                    bool isUpper = sh >= 4;
                    int qhOffset = half * 32 + (sh % 2) * 16;
                    int qhShift = (sh / 2) * 2;

                    int isum = 0;
                    for (int i = 0; i < 16; i++)
                    {
                        int lo4 = isUpper ? (ql[qlOffset + i] >> 4) & 0x0F : ql[qlOffset + i] & 0x0F;
                        int hi2 = (qh[qhOffset + i] >> qhShift) & 0x03;
                        int q6 = (lo4 | (hi2 << 4)) - 32;
                        isum += q6 * q8Vals[i];
                    }

                    sum += scale * isum;
                }

                q6k += Q6_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        private static unsafe void UnpackQ4Q5Scales(byte* packed, byte* scales, byte* mins)
        {
            for (int i = 0; i < 8; i++)
                GetScaleMinK4(i, packed, out scales[i], out mins[i]);
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
            return TensorPrimitives.Dot(
                new ReadOnlySpan<float>(lhs, length),
                new ReadOnlySpan<float>(rhs, length));
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
        private static unsafe float ReadSingle(byte* p) => Unsafe.ReadUnaligned<float>(ref *p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HalfToSingle(ushort value) => (float)BitConverter.UInt16BitsToHalf(value);

        private static unsafe float HorizontalSum(Vector256<float> v)
        {
            float* tmp = stackalloc float[8];
            Avx.Store(tmp, v);
            float sum = 0.0f;
            for (int i = 0; i < 8; i++)
                sum += tmp[i];
            return sum;
        }

        private static unsafe float HorizontalSum(Vector512<float> v)
        {
            float* tmp = stackalloc float[16];
            Avx512F.Store(tmp, v);
            float sum = 0.0f;
            for (int i = 0; i < 16; i++)
                sum += tmp[i];
            return sum;
        }

        private static unsafe float HorizontalMax(Vector512<float> v)
        {
            float* tmp = stackalloc float[16];
            Avx512F.Store(tmp, v);
            float max = tmp[0];
            for (int i = 1; i < 16; i++)
                if (tmp[i] > max) max = tmp[i];
            return max;
        }

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

