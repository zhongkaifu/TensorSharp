// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.Core;
using TensorSharp.Cpu.LinearAlgebra;

namespace TensorSharp.Cpu
{
    public enum BlasOp : byte
    {
        NonTranspose = (byte)'n',
        Transpose = (byte)'T',
        ConjugateTranspose = (byte)'C',
    }


    /// <summary>
    /// The matrix data storage format.
    /// </summary>
    public enum Order
    {
        /// <summary>
        /// The matrix array uses a row-major layout.
        /// </summary>
        Row = 101,

        /// <summary>
        /// The matrix array uses a column-major layout.
        /// </summary>
        Column = 102
    }

    /// <summary>
    /// Matrix transpose type.
    /// </summary>
    public enum Transpose
    {
        /// <summary>
        /// Don't transpose the matrix.  Equivalent to trans='N'
        /// </summary>
        NoTrans = 111,

        /// <summary>
        /// Transpose the matrix.  Equivalent to trans='T'
        /// </summary>
        Trans = 112,

        /// <summary>
        /// Conjugate transpose the matrix. The only refers to complex matrices. Real matrices will just be transposed.  Equivalent to trans='C'
        /// </summary>
        ConjTrans = 113
    }

    unsafe public static class MatrixMultiplication
    {
        const string mklDllName = "mkl_rt.2.dll";
        internal const string mklDllNameLinux = "mkl_rt";
        private const int ParallelWorkThreshold = 1 << 15;
        private static readonly SGEMM ManagedSgemm = new SGEMM();

        static MatrixMultiplication()
        {
            NativeLibrary.SetDllImportResolver(typeof(MatrixMultiplication).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            IntPtr libHandle = IntPtr.Zero;

            if (libraryName == mklDllName)
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    NativeLibrary.TryLoad(mklDllNameLinux, assembly, DllImportSearchPath.SafeDirectories, out libHandle);
                }
            }
            //On Windows, use the default library name
            return libHandle;
        }



        public static Transpose ConvertBlasOp(BlasOp op)
        {
            if (op == BlasOp.NonTranspose)
            {
                return Transpose.NoTrans;
            }
            else if (op == BlasOp.Transpose)
            {
                return Transpose.Trans;
            }

            return Transpose.ConjTrans;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldParallelize(int outerWork, int innerWork)
        {
            return outerWork > 1 && (long)outerWork * innerWork >= ParallelWorkThreshold;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool UsesManagedBlas(Tensor a, Tensor b, Tensor c)
        {
            return a.Allocator.BlasEnum != BlasEnum.MKL &&
                   b.Allocator.BlasEnum != BlasEnum.MKL &&
                   c.Allocator.BlasEnum != BlasEnum.MKL;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsRowMajorContiguous2D(Tensor tensor)
        {
            return tensor.DimensionCount == 2 &&
                   tensor.ElementType == DType.Float32 &&
                   tensor.Strides[1] == 1 &&
                   tensor.Strides[0] == tensor.Sizes[1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsColumnMajorContiguous2D(Tensor tensor)
        {
            return tensor.DimensionCount == 2 &&
                   tensor.ElementType == DType.Float32 &&
                   tensor.Strides[0] == 1 &&
                   tensor.Strides[1] == tensor.Sizes[0];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsRowMajorContiguous3D(Tensor tensor)
        {
            return tensor.DimensionCount == 3 &&
                   tensor.ElementType == DType.Float32 &&
                   tensor.Strides[2] == 1 &&
                   tensor.Strides[1] == tensor.Sizes[2] &&
                   tensor.Strides[0] == tensor.Sizes[1] * tensor.Sizes[2];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsColumnMajorContiguous3D(Tensor tensor)
        {
            return tensor.DimensionCount == 3 &&
                   tensor.ElementType == DType.Float32 &&
                   tensor.Strides[1] == 1 &&
                   tensor.Strides[2] == tensor.Sizes[1] &&
                   tensor.Strides[0] == tensor.Sizes[1] * tensor.Sizes[2];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LoadVec(float* ptr)
        {
            return Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)ptr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StoreVec(float* ptr, Vector<float> value)
        {
            Unsafe.WriteUnaligned(ref *(byte*)ptr, value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotContiguous(float* lhs, float* rhs, int length)
        {
            return TensorPrimitives.Dot(
                new ReadOnlySpan<float>(lhs, length),
                new ReadOnlySpan<float>(rhs, length));
        }

        private static unsafe void ApplyBeta(float* row, int length, float beta)
        {
            if (beta == 1.0f)
            {
                return;
            }

            if (beta == 0.0f)
            {
                new Span<float>(row, length).Fill(0.0f);
                return;
            }

            int vectorSize = Vector<float>.Count;
            Vector<float> vecBeta = new Vector<float>(beta);
            int i = 0;
            for (; i <= length - vectorSize; i += vectorSize)
            {
                StoreVec(row + i, LoadVec(row + i) * vecBeta);
            }

            for (; i < length; i++)
            {
                row[i] *= beta;
            }
        }

        private static unsafe void ManagedGemmRowRowOne(float alpha, float* aRow, float* bBase, int bRowStride, float beta, float* cRow, int n, int k)
        {
            ApplyBeta(cRow, n, beta);
            if (alpha == 0.0f)
            {
                return;
            }

            int vectorSize = Vector<float>.Count;
            for (int kk = 0; kk < k; kk++)
            {
                float scaledA = alpha * aRow[kk];
                if (scaledA == 0.0f)
                {
                    continue;
                }

                float* bRow = bBase + kk * bRowStride;
                Vector<float> vecA = new Vector<float>(scaledA);
                int j = 0;
                for (; j <= n - vectorSize; j += vectorSize)
                {
                    StoreVec(cRow + j, LoadVec(cRow + j) + LoadVec(bRow + j) * vecA);
                }

                for (; j < n; j++)
                {
                    cRow[j] += scaledA * bRow[j];
                }
            }
        }

        private static unsafe void ManagedGemmRowRowFour(
            float alpha,
            float* aRow0, float* aRow1, float* aRow2, float* aRow3,
            float* bBase, int bRowStride,
            float beta,
            float* cRow0, float* cRow1, float* cRow2, float* cRow3,
            int n, int k)
        {
            ApplyBeta(cRow0, n, beta);
            ApplyBeta(cRow1, n, beta);
            ApplyBeta(cRow2, n, beta);
            ApplyBeta(cRow3, n, beta);

            if (alpha == 0.0f)
            {
                return;
            }

            int vectorSize = Vector<float>.Count;
            for (int kk = 0; kk < k; kk++)
            {
                float scaledA0 = alpha * aRow0[kk];
                float scaledA1 = alpha * aRow1[kk];
                float scaledA2 = alpha * aRow2[kk];
                float scaledA3 = alpha * aRow3[kk];

                bool update0 = scaledA0 != 0.0f;
                bool update1 = scaledA1 != 0.0f;
                bool update2 = scaledA2 != 0.0f;
                bool update3 = scaledA3 != 0.0f;
                if (!update0 && !update1 && !update2 && !update3)
                {
                    continue;
                }

                float* bRow = bBase + kk * bRowStride;
                int j = 0;
                if (update0 && update1 && update2 && update3)
                {
                    Vector<float> vecA0 = new Vector<float>(scaledA0);
                    Vector<float> vecA1 = new Vector<float>(scaledA1);
                    Vector<float> vecA2 = new Vector<float>(scaledA2);
                    Vector<float> vecA3 = new Vector<float>(scaledA3);
                    for (; j <= n - vectorSize; j += vectorSize)
                    {
                        Vector<float> vecB = LoadVec(bRow + j);
                        StoreVec(cRow0 + j, LoadVec(cRow0 + j) + vecB * vecA0);
                        StoreVec(cRow1 + j, LoadVec(cRow1 + j) + vecB * vecA1);
                        StoreVec(cRow2 + j, LoadVec(cRow2 + j) + vecB * vecA2);
                        StoreVec(cRow3 + j, LoadVec(cRow3 + j) + vecB * vecA3);
                    }

                    for (; j < n; j++)
                    {
                        float b = bRow[j];
                        cRow0[j] += scaledA0 * b;
                        cRow1[j] += scaledA1 * b;
                        cRow2[j] += scaledA2 * b;
                        cRow3[j] += scaledA3 * b;
                    }

                    continue;
                }

                Vector<float> vecA0Optional = new Vector<float>(scaledA0);
                Vector<float> vecA1Optional = new Vector<float>(scaledA1);
                Vector<float> vecA2Optional = new Vector<float>(scaledA2);
                Vector<float> vecA3Optional = new Vector<float>(scaledA3);
                for (; j <= n - vectorSize; j += vectorSize)
                {
                    Vector<float> vecB = LoadVec(bRow + j);
                    if (update0) StoreVec(cRow0 + j, LoadVec(cRow0 + j) + vecB * vecA0Optional);
                    if (update1) StoreVec(cRow1 + j, LoadVec(cRow1 + j) + vecB * vecA1Optional);
                    if (update2) StoreVec(cRow2 + j, LoadVec(cRow2 + j) + vecB * vecA2Optional);
                    if (update3) StoreVec(cRow3 + j, LoadVec(cRow3 + j) + vecB * vecA3Optional);
                }

                for (; j < n; j++)
                {
                    float b = bRow[j];
                    if (update0) cRow0[j] += scaledA0 * b;
                    if (update1) cRow1[j] += scaledA1 * b;
                    if (update2) cRow2[j] += scaledA2 * b;
                    if (update3) cRow3[j] += scaledA3 * b;
                }
            }
        }

        private static unsafe void DotContiguousFour(
            float* lhs0, float* lhs1, float* lhs2, float* lhs3,
            float* rhs,
            int length,
            out float sum0, out float sum1, out float sum2, out float sum3)
        {
            int vectorSize = Vector<float>.Count;
            Vector<float> acc0 = Vector<float>.Zero;
            Vector<float> acc1 = Vector<float>.Zero;
            Vector<float> acc2 = Vector<float>.Zero;
            Vector<float> acc3 = Vector<float>.Zero;
            int i = 0;

            for (; i <= length - vectorSize; i += vectorSize)
            {
                Vector<float> r = LoadVec(rhs + i);
                acc0 += LoadVec(lhs0 + i) * r;
                acc1 += LoadVec(lhs1 + i) * r;
                acc2 += LoadVec(lhs2 + i) * r;
                acc3 += LoadVec(lhs3 + i) * r;
            }

            sum0 = Vector.Sum(acc0);
            sum1 = Vector.Sum(acc1);
            sum2 = Vector.Sum(acc2);
            sum3 = Vector.Sum(acc3);
            for (; i < length; i++)
            {
                float r = rhs[i];
                sum0 += lhs0[i] * r;
                sum1 += lhs1[i] * r;
                sum2 += lhs2[i] * r;
                sum3 += lhs3[i] * r;
            }
        }

        private static unsafe void DotContiguousFourByFour(
            float* lhs0, float* lhs1, float* lhs2, float* lhs3,
            float* rhs0, float* rhs1, float* rhs2, float* rhs3,
            int length,
            out float sum00, out float sum01, out float sum02, out float sum03,
            out float sum10, out float sum11, out float sum12, out float sum13,
            out float sum20, out float sum21, out float sum22, out float sum23,
            out float sum30, out float sum31, out float sum32, out float sum33)
        {
            int vectorSize = Vector<float>.Count;
            Vector<float> acc00 = Vector<float>.Zero;
            Vector<float> acc01 = Vector<float>.Zero;
            Vector<float> acc02 = Vector<float>.Zero;
            Vector<float> acc03 = Vector<float>.Zero;
            Vector<float> acc10 = Vector<float>.Zero;
            Vector<float> acc11 = Vector<float>.Zero;
            Vector<float> acc12 = Vector<float>.Zero;
            Vector<float> acc13 = Vector<float>.Zero;
            Vector<float> acc20 = Vector<float>.Zero;
            Vector<float> acc21 = Vector<float>.Zero;
            Vector<float> acc22 = Vector<float>.Zero;
            Vector<float> acc23 = Vector<float>.Zero;
            Vector<float> acc30 = Vector<float>.Zero;
            Vector<float> acc31 = Vector<float>.Zero;
            Vector<float> acc32 = Vector<float>.Zero;
            Vector<float> acc33 = Vector<float>.Zero;
            int i = 0;

            for (; i <= length - vectorSize; i += vectorSize)
            {
                Vector<float> a0 = LoadVec(lhs0 + i);
                Vector<float> a1 = LoadVec(lhs1 + i);
                Vector<float> a2 = LoadVec(lhs2 + i);
                Vector<float> a3 = LoadVec(lhs3 + i);
                Vector<float> b0 = LoadVec(rhs0 + i);
                Vector<float> b1 = LoadVec(rhs1 + i);
                Vector<float> b2 = LoadVec(rhs2 + i);
                Vector<float> b3 = LoadVec(rhs3 + i);

                acc00 += a0 * b0;
                acc01 += a0 * b1;
                acc02 += a0 * b2;
                acc03 += a0 * b3;
                acc10 += a1 * b0;
                acc11 += a1 * b1;
                acc12 += a1 * b2;
                acc13 += a1 * b3;
                acc20 += a2 * b0;
                acc21 += a2 * b1;
                acc22 += a2 * b2;
                acc23 += a2 * b3;
                acc30 += a3 * b0;
                acc31 += a3 * b1;
                acc32 += a3 * b2;
                acc33 += a3 * b3;
            }

            sum00 = Vector.Sum(acc00);
            sum01 = Vector.Sum(acc01);
            sum02 = Vector.Sum(acc02);
            sum03 = Vector.Sum(acc03);
            sum10 = Vector.Sum(acc10);
            sum11 = Vector.Sum(acc11);
            sum12 = Vector.Sum(acc12);
            sum13 = Vector.Sum(acc13);
            sum20 = Vector.Sum(acc20);
            sum21 = Vector.Sum(acc21);
            sum22 = Vector.Sum(acc22);
            sum23 = Vector.Sum(acc23);
            sum30 = Vector.Sum(acc30);
            sum31 = Vector.Sum(acc31);
            sum32 = Vector.Sum(acc32);
            sum33 = Vector.Sum(acc33);

            for (; i < length; i++)
            {
                float a0 = lhs0[i];
                float a1 = lhs1[i];
                float a2 = lhs2[i];
                float a3 = lhs3[i];
                float b0 = rhs0[i];
                float b1 = rhs1[i];
                float b2 = rhs2[i];
                float b3 = rhs3[i];

                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum02 += a0 * b2;
                sum03 += a0 * b3;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
                sum12 += a1 * b2;
                sum13 += a1 * b3;
                sum20 += a2 * b0;
                sum21 += a2 * b1;
                sum22 += a2 * b2;
                sum23 += a2 * b3;
                sum30 += a3 * b0;
                sum31 += a3 * b1;
                sum32 += a3 * b2;
                sum33 += a3 * b3;
            }
        }

        private static unsafe void ManagedGemmRowColOne(float alpha, float* aRow, float* bBase, int bColStride, float beta, float* cRow, int n, int k)
        {
            if (alpha == 0.0f)
            {
                ApplyBeta(cRow, n, beta);
                return;
            }

            for (int j = 0; j < n; j++)
            {
                float value = alpha * DotContiguous(aRow, bBase + j * bColStride, k);
                cRow[j] = beta == 0.0f ? value : value + beta * cRow[j];
            }
        }

        private static unsafe void ManagedGemmRowColFour(
            float alpha,
            float* aRow0, float* aRow1, float* aRow2, float* aRow3,
            float* bBase, int bColStride,
            float beta,
            float* cRow0, float* cRow1, float* cRow2, float* cRow3,
            int n, int k)
        {
            if (alpha == 0.0f)
            {
                ApplyBeta(cRow0, n, beta);
                ApplyBeta(cRow1, n, beta);
                ApplyBeta(cRow2, n, beta);
                ApplyBeta(cRow3, n, beta);
                return;
            }

            int j = 0;
            for (; j + 3 < n; j += 4)
            {
                DotContiguousFourByFour(
                    aRow0, aRow1, aRow2, aRow3,
                    bBase + j * bColStride,
                    bBase + (j + 1) * bColStride,
                    bBase + (j + 2) * bColStride,
                    bBase + (j + 3) * bColStride,
                    k,
                    out float dot00, out float dot01, out float dot02, out float dot03,
                    out float dot10, out float dot11, out float dot12, out float dot13,
                    out float dot20, out float dot21, out float dot22, out float dot23,
                    out float dot30, out float dot31, out float dot32, out float dot33);

                if (beta == 0.0f)
                {
                    cRow0[j] = alpha * dot00;
                    cRow0[j + 1] = alpha * dot01;
                    cRow0[j + 2] = alpha * dot02;
                    cRow0[j + 3] = alpha * dot03;
                    cRow1[j] = alpha * dot10;
                    cRow1[j + 1] = alpha * dot11;
                    cRow1[j + 2] = alpha * dot12;
                    cRow1[j + 3] = alpha * dot13;
                    cRow2[j] = alpha * dot20;
                    cRow2[j + 1] = alpha * dot21;
                    cRow2[j + 2] = alpha * dot22;
                    cRow2[j + 3] = alpha * dot23;
                    cRow3[j] = alpha * dot30;
                    cRow3[j + 1] = alpha * dot31;
                    cRow3[j + 2] = alpha * dot32;
                    cRow3[j + 3] = alpha * dot33;
                }
                else
                {
                    cRow0[j] = alpha * dot00 + beta * cRow0[j];
                    cRow0[j + 1] = alpha * dot01 + beta * cRow0[j + 1];
                    cRow0[j + 2] = alpha * dot02 + beta * cRow0[j + 2];
                    cRow0[j + 3] = alpha * dot03 + beta * cRow0[j + 3];
                    cRow1[j] = alpha * dot10 + beta * cRow1[j];
                    cRow1[j + 1] = alpha * dot11 + beta * cRow1[j + 1];
                    cRow1[j + 2] = alpha * dot12 + beta * cRow1[j + 2];
                    cRow1[j + 3] = alpha * dot13 + beta * cRow1[j + 3];
                    cRow2[j] = alpha * dot20 + beta * cRow2[j];
                    cRow2[j + 1] = alpha * dot21 + beta * cRow2[j + 1];
                    cRow2[j + 2] = alpha * dot22 + beta * cRow2[j + 2];
                    cRow2[j + 3] = alpha * dot23 + beta * cRow2[j + 3];
                    cRow3[j] = alpha * dot30 + beta * cRow3[j];
                    cRow3[j + 1] = alpha * dot31 + beta * cRow3[j + 1];
                    cRow3[j + 2] = alpha * dot32 + beta * cRow3[j + 2];
                    cRow3[j + 3] = alpha * dot33 + beta * cRow3[j + 3];
                }
            }

            for (; j < n; j++)
            {
                DotContiguousFour(
                    aRow0, aRow1, aRow2, aRow3,
                    bBase + j * bColStride,
                    k,
                    out float dot0, out float dot1, out float dot2, out float dot3);

                float value0 = alpha * dot0;
                float value1 = alpha * dot1;
                float value2 = alpha * dot2;
                float value3 = alpha * dot3;
                if (beta == 0.0f)
                {
                    cRow0[j] = value0;
                    cRow1[j] = value1;
                    cRow2[j] = value2;
                    cRow3[j] = value3;
                }
                else
                {
                    cRow0[j] = value0 + beta * cRow0[j];
                    cRow1[j] = value1 + beta * cRow1[j];
                    cRow2[j] = value2 + beta * cRow2[j];
                    cRow3[j] = value3 + beta * cRow3[j];
                }
            }
        }

        private static unsafe bool TryManagedGemm(float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (!UsesManagedBlas(a, b, c) || !IsRowMajorContiguous2D(a) || !IsRowMajorContiguous2D(c))
            {
                return false;
            }

            float* aPtr = (float*)CpuNativeHelpers.GetBufferStart(a);
            float* bPtr = (float*)CpuNativeHelpers.GetBufferStart(b);
            float* cPtr = (float*)CpuNativeHelpers.GetBufferStart(c);
            int m = (int)c.Sizes[0];
            int n = (int)c.Sizes[1];
            int k = (int)a.Sizes[1];
            int aRowStride = (int)a.Strides[0];
            int cRowStride = (int)c.Strides[0];

            if (IsRowMajorContiguous2D(b))
            {
                int bRowStride = (int)b.Strides[0];

                void ComputeRowRow(int row)
                {
                    ManagedGemmRowRowOne(alpha, aPtr + row * aRowStride, bPtr, bRowStride, beta, cPtr + row * cRowStride, n, k);
                }

                void ComputeRowRowBlock(int block)
                {
                    int row = block * 4;
                    ManagedGemmRowRowFour(
                        alpha,
                        aPtr + row * aRowStride,
                        aPtr + (row + 1) * aRowStride,
                        aPtr + (row + 2) * aRowStride,
                        aPtr + (row + 3) * aRowStride,
                        bPtr, bRowStride, beta,
                        cPtr + row * cRowStride,
                        cPtr + (row + 1) * cRowStride,
                        cPtr + (row + 2) * cRowStride,
                        cPtr + (row + 3) * cRowStride,
                        n, k);
                }

                int fullRows = m & ~3;
                if (fullRows >= 4)
                {
                    int blockCount = fullRows / 4;
                    if (ShouldParallelize(fullRows, n * k))
                    {
                        Parallel.For(0, blockCount, ComputeRowRowBlock);
                    }
                    else
                    {
                        for (int block = 0; block < blockCount; block++)
                        {
                            ComputeRowRowBlock(block);
                        }
                    }

                    for (int row = fullRows; row < m; row++)
                    {
                        ComputeRowRow(row);
                    }
                }
                else if (ShouldParallelize(m, n * k))
                {
                    Parallel.For(0, m, ComputeRowRow);
                }
                else
                {
                    for (int row = 0; row < m; row++)
                    {
                        ComputeRowRow(row);
                    }
                }

                return true;
            }

            if (!IsColumnMajorContiguous2D(b))
            {
                return false;
            }

            int bColStride = (int)b.Strides[1];
            if (m == 1 && ShouldParallelize(n, k))
            {
                float* aRow = aPtr;
                float* cRow = cPtr;
                void ComputeColumn(int col)
                {
                    float value = alpha * DotContiguous(aRow, bPtr + col * bColStride, k);
                    cRow[col] = beta == 0.0f ? value : value + beta * cRow[col];
                }

                Parallel.For(0, n, ComputeColumn);
                return true;
            }

            void ComputeRowCol(int row)
            {
                ManagedGemmRowColOne(alpha, aPtr + row * aRowStride, bPtr, bColStride, beta, cPtr + row * cRowStride, n, k);
            }

            if (m >= 4)
            {
                int fullRows = m & ~3;

                void ComputeRowColBlock(int block)
                {
                    int row = block * 4;
                    ManagedGemmRowColFour(
                        alpha,
                        aPtr + row * aRowStride,
                        aPtr + (row + 1) * aRowStride,
                        aPtr + (row + 2) * aRowStride,
                        aPtr + (row + 3) * aRowStride,
                        bPtr, bColStride, beta,
                        cPtr + row * cRowStride,
                        cPtr + (row + 1) * cRowStride,
                        cPtr + (row + 2) * cRowStride,
                        cPtr + (row + 3) * cRowStride,
                        n, k);
                }

                int blockCount = fullRows / 4;
                if (ShouldParallelize(fullRows, n * k))
                {
                    Parallel.For(0, blockCount, ComputeRowColBlock);
                }
                else
                {
                    for (int block = 0; block < blockCount; block++)
                    {
                        ComputeRowColBlock(block);
                    }
                }

                for (int row = fullRows; row < m; row++)
                {
                    ComputeRowCol(row);
                }
            }
            else if (ShouldParallelize(m, n * k))
            {
                Parallel.For(0, m, ComputeRowCol);
            }
            else
            {
                for (int row = 0; row < m; row++)
                {
                    ComputeRowCol(row);
                }
            }

            return true;
        }

        private static unsafe bool TryManagedGemmBatch(float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (!UsesManagedBlas(a, b, c) || !IsRowMajorContiguous3D(a) || !IsRowMajorContiguous3D(c))
            {
                return false;
            }

            float* aPtr = (float*)CpuNativeHelpers.GetBufferStart(a);
            float* bPtr = (float*)CpuNativeHelpers.GetBufferStart(b);
            float* cPtr = (float*)CpuNativeHelpers.GetBufferStart(c);
            int batch = (int)c.Sizes[0];
            int m = (int)c.Sizes[1];
            int n = (int)c.Sizes[2];
            int k = (int)a.Sizes[2];
            int aBatchStride = (int)a.Strides[0];
            int aRowStride = (int)a.Strides[1];
            int cBatchStride = (int)c.Strides[0];
            int cRowStride = (int)c.Strides[1];

            if (IsRowMajorContiguous3D(b))
            {
                int bBatchStrideRow = (int)b.Strides[0];
                int bRowStride = (int)b.Strides[1];

                void ComputeBatchRowRow(int batchIndex)
                {
                    float* aBatch = aPtr + batchIndex * aBatchStride;
                    float* bBatch = bPtr + batchIndex * bBatchStrideRow;
                    float* cBatch = cPtr + batchIndex * cBatchStride;
                    int row = 0;
                    for (; row + 3 < m; row += 4)
                    {
                        ManagedGemmRowRowFour(
                            alpha,
                            aBatch + row * aRowStride,
                            aBatch + (row + 1) * aRowStride,
                            aBatch + (row + 2) * aRowStride,
                            aBatch + (row + 3) * aRowStride,
                            bBatch, bRowStride, beta,
                            cBatch + row * cRowStride,
                            cBatch + (row + 1) * cRowStride,
                            cBatch + (row + 2) * cRowStride,
                            cBatch + (row + 3) * cRowStride,
                            n, k);
                    }

                    for (; row < m; row++)
                    {
                        ManagedGemmRowRowOne(alpha, aBatch + row * aRowStride, bBatch, bRowStride, beta, cBatch + row * cRowStride, n, k);
                    }
                }

                if (ShouldParallelize(batch, m * n * k))
                {
                    Parallel.For(0, batch, ComputeBatchRowRow);
                }
                else
                {
                    for (int batchIndex = 0; batchIndex < batch; batchIndex++)
                    {
                        ComputeBatchRowRow(batchIndex);
                    }
                }

                return true;
            }

            if (!IsColumnMajorContiguous3D(b))
            {
                return false;
            }

            int bBatchStrideCol = (int)b.Strides[0];
            int bColStride = (int)b.Strides[2];

            void ComputeBatchRowCol(int batchIndex)
            {
                float* aBatch = aPtr + batchIndex * aBatchStride;
                float* bBatch = bPtr + batchIndex * bBatchStrideCol;
                float* cBatch = cPtr + batchIndex * cBatchStride;
                int row = 0;
                for (; row + 3 < m; row += 4)
                {
                    ManagedGemmRowColFour(
                        alpha,
                        aBatch + row * aRowStride,
                        aBatch + (row + 1) * aRowStride,
                        aBatch + (row + 2) * aRowStride,
                        aBatch + (row + 3) * aRowStride,
                        bBatch, bColStride, beta,
                        cBatch + row * cRowStride,
                        cBatch + (row + 1) * cRowStride,
                        cBatch + (row + 2) * cRowStride,
                        cBatch + (row + 3) * cRowStride,
                        n, k);
                }

                for (; row < m; row++)
                {
                    ManagedGemmRowColOne(alpha, aBatch + row * aRowStride, bBatch, bColStride, beta, cBatch + row * cRowStride, n, k);
                }
            }

            if (ShouldParallelize(batch, m * n * k))
            {
                Parallel.For(0, batch, ComputeBatchRowCol);
            }
            else
            {
                for (int batchIndex = 0; batchIndex < batch; batchIndex++)
                {
                    ComputeBatchRowCol(batchIndex);
                }
            }

            return true;
        }


        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type lhs = '{lhs.ElementType}', rhs = '{rhs.ElementType}' result = '{result.ElementType}'");
            }

            if (result != null && !(result.Storage is CpuStorage))
            {
                throw new ArgumentException("result must be a CPU tensor", nameof(result));
            }

            if (!(lhs.Storage is CpuStorage))
            {
                throw new ArgumentException("lhs must be a CPU tensor", nameof(lhs));
            }

            if (!(rhs.Storage is CpuStorage))
            {
                throw new ArgumentException("rhs must be a CPU tensor", nameof(rhs));
            }

            if (lhs.DimensionCount != 1)
            {
                throw new ArgumentException("lhs must have 1 dimension (ie. be a vector)", nameof(lhs));
            }

            if (rhs.DimensionCount != 1)
            {
                throw new ArgumentException("rhs must have 1 dimension (ie. be a vector)", nameof(rhs));
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, 1);

            if (writeTarget.ElementType == DType.Float32)
            {
                Run_Dot_float(writeTarget, lhs, rhs);
            }
            else if (writeTarget.ElementType == DType.Float64)
            {
                Run_Dot_double(writeTarget, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException("CPU vector dot product with element type " + result.ElementType + " not supported");
            }

            return writeTarget;
        }

        private static void Run_Dot_float(Tensor result, Tensor lhs, Tensor rhs)
        {
            unsafe
            {
                float* resultPtr = (float*)CpuNativeHelpers.GetBufferStart(result);
                float* lhsPtr = (float*)CpuNativeHelpers.GetBufferStart(lhs);
                float* rhsPtr = (float*)CpuNativeHelpers.GetBufferStart(rhs);

                int n = (int)lhs.Sizes[0];
                int incx = (int)lhs.Strides[0];
                int incy = (int)rhs.Strides[0];
                *resultPtr = OpenBlasNative.sdot_(&n, lhsPtr, &incx, rhsPtr, &incy);
            }
        }

        private static void Run_Dot_double(Tensor result, Tensor lhs, Tensor rhs)
        {
            unsafe
            {
                double* resultPtr = (double*)CpuNativeHelpers.GetBufferStart(result);
                double* lhsPtr = (double*)CpuNativeHelpers.GetBufferStart(lhs);
                double* rhsPtr = (double*)CpuNativeHelpers.GetBufferStart(rhs);

                int n = (int)lhs.Sizes[0];
                int incx = (int)lhs.Strides[0];
                int incy = (int)rhs.Strides[0];
                *resultPtr = OpenBlasNative.ddot_(&n, lhsPtr, &incx, rhsPtr, &incy);
            }
        }

        public static Tensor Mul_M_V(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type  lhs = '{lhs.ElementType}', rhs = '{rhs.ElementType}' result = '{result.ElementType}'");
            }

            if (result != null && (result.Storage is CpuStorage))
            {
                throw new ArgumentException("result must be a CPU tensor", nameof(result));
            }

            if (!(lhs.Storage is CpuStorage))
            {
                throw new ArgumentException("lhs must be a CPU tensor", nameof(lhs));
            }

            if (!(rhs.Storage is CpuStorage))
            {
                throw new ArgumentException("rhs must be a CPU tensor", nameof(rhs));
            }

            if (lhs.DimensionCount != 2)
            {
                throw new ArgumentException("lhs must have 2 dimensions", nameof(lhs));
            }

            if (rhs.DimensionCount != 1)
            {
                throw new ArgumentException("rhs must have 1 dimension (ie. be a vector)", nameof(rhs));
            }

            Tensor lhsClone;
            if (lhs.Strides[1] == 1) // If lhs is already row-major, do nothing
            {
                lhsClone = lhs.CopyRef();
            }
            else if (lhs.Strides[0] == 1) // If lhs is column-major, transpose it
            {
                lhsClone = lhs.Transpose();
            }
            else // If lhs is not contiguous in either dimension, make a temporary contiguous copy
            {
                lhsClone = Ops.NewContiguous(lhs);
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, rhs, false, lhs.Sizes[0]);

            try
            {
                if (writeTarget.ElementType == DType.Float32)
                {
                    Run_M_V_float(writeTarget, lhsClone, rhs);
                }
                else if (writeTarget.ElementType == DType.Float64)
                {
                    Run_M_V_double(writeTarget, lhsClone, rhs);
                }
                else
                {
                    throw new NotSupportedException("CPU Matrix-Vector multiplication with element type " + result.ElementType + " not supported");
                }
            }
            finally
            {
                lhsClone.Dispose();
            }

            return writeTarget;
        }

        private static void Run_M_V_float(Tensor result, Tensor mat, Tensor vec)
        {
            // Require lhs to be row-major. This means we must tell BLAS to transpose it (BLAS expects column-major matrices)
            if (mat.Strides[1] != 1)
            {
                throw new ArgumentException("lhs must be contiguous in the last dimension");
            }

            unsafe
            {
                float* yPtr = (float*)CpuNativeHelpers.GetBufferStart(result);
                float* aPtr = (float*)CpuNativeHelpers.GetBufferStart(mat);
                float* xPtr = (float*)CpuNativeHelpers.GetBufferStart(vec);

                byte trans = (byte)'t';
                int m = (int)mat.Sizes[1];
                int n = (int)mat.Sizes[0];
                int incx = (int)vec.Strides[0];
                int lda = (int)mat.Strides[0];
                int incy = (int)result.Strides[0];
                float alpha = 1;
                float beta = 0;
                OpenBlasNative.sgemv_(&trans, &m, &n, &alpha, aPtr, &lda, xPtr, &incx, &beta, yPtr, &incy);
            }
        }

        private static void Run_M_V_double(Tensor result, Tensor lhs, Tensor rhs)
        {
            // Require lhs to be row-major. This means we must tell BLAS to transpose it (BLAS expects column-major matrices)
            if (lhs.Strides[1] != 1)
            {
                throw new ArgumentException("lhs must be contiguous in the last dimension");
            }

            unsafe
            {
                double* resultPtr = (double*)CpuNativeHelpers.GetBufferStart(result);
                double* lhsPtr = (double*)CpuNativeHelpers.GetBufferStart(lhs);
                double* rhsPtr = (double*)CpuNativeHelpers.GetBufferStart(rhs);

                byte trans = (byte)'t';
                int m = (int)rhs.Sizes[1];
                int n = (int)lhs.Sizes[0];
                int lda = (int)rhs.Strides[0];
                int ldb = (int)lhs.Strides[0];
                int ldc = (int)result.Strides[0];
                double alpha = 1;
                double beta = 0;
                OpenBlasNative.dgemv_(&trans, &m, &n, &alpha, rhsPtr, &lda, lhsPtr, &ldb, &beta, resultPtr, &ldc);
            }
        }



        public static Tensor Mul_M_M(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type lhs = '{lhs.ElementType}', rhs = '{rhs.ElementType}' result = '{result.ElementType}'");
            }

            if (result != null && !(result.Storage is CpuStorage))
            {
                throw new ArgumentException("result must be a CPU tensor", nameof(result));
            }

            if (!(lhs.Storage is CpuStorage))
            {
                throw new ArgumentException("lhs must be a CPU tensor", nameof(lhs));
            }

            if (!(rhs.Storage is CpuStorage))
            {
                throw new ArgumentException("rhs must be a CPU tensor", nameof(rhs));
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes[0], rhs.Sizes[1]);

            Gemm(1, lhs, rhs, 0, writeTarget);


            return writeTarget;
        }

        
        public static void Gemm(float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Sizes[0] != c.Sizes[0] || b.Sizes[1] != c.Sizes[1] || a.Sizes[1] != b.Sizes[0])
            {
                throw new InvalidOperationException("Size mismatch");
            }

            if (TryManagedGemm(alpha, a, b, beta, c))
            {
                return;
            }

            bool copyC = false;
            Tensor aClone;
            Tensor bClone;
            Tensor cClone;
            if (c.Strides[0] == 1 &&
                c.Strides[1] != 0 && c.Strides[1] != 1)
            {
                // If c is contiguous in dimension 0 (column-major)
                aClone = a.CopyRef();
                bClone = b.CopyRef();
                cClone = c.CopyRef();
            }
            else if (c.Strides[1] == 1 &&
                c.Strides[0] != 0 && c.Strides[0] != 1)
            {
                // If c is contiguous in dimension 1 (row-major)
                // using (a * b)' == b' * a'
                // we can pass row-major matrices to BLAS functions that expect column-major by swapping A and B,
                // and transposing all 3 matrices

                cClone = c.Transpose();
                aClone = b.Transpose(); // Note swap of a and b
                bClone = a.Transpose();
            }
            else
            {
                Tensor cNew = new Tensor(c.Allocator, c.ElementType, c.Sizes[1], c.Sizes[0]);
                cClone = cNew.Transpose();
                Ops.Copy(cClone, c);
                cNew.Dispose();
                copyC = true;

                aClone = a.CopyRef();
                bClone = b.CopyRef();
            }

            try
            {

                BlasOp aOp;
                if (aClone.Strides[0] == 1 &&
                    aClone.Strides[1] != 0 && aClone.Strides[1] != 1)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    aOp = BlasOp.NonTranspose;
                }
                else if (aClone.Strides[1] == 1 &&
                    aClone.Strides[0] != 0 && aClone.Strides[0] != 1)
                {
                    aOp = BlasOp.Transpose;
                    Tensor aNew = aClone.Transpose();
                    aClone.Dispose();
                    aClone = aNew;
                }
                else
                {
                    Tensor aNew = new Tensor(aClone.Allocator, aClone.ElementType, aClone.Sizes[1], aClone.Sizes[0]);
                    Tensor aClone2 = aNew.Transpose();
                    Ops.Copy(aClone2, aClone);
                    aClone.Dispose();
                    aClone = aClone2;
                    aNew.Dispose();

                    aOp = BlasOp.NonTranspose;
                }


                BlasOp bOp;
                if (bClone.Strides[0] == 1 &&
                    bClone.Strides[1] != 0 && bClone.Strides[1] != 1)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    bOp = BlasOp.NonTranspose;
                }
                else if (bClone.Strides[1] == 1 &&
                    bClone.Strides[0] != 0 && bClone.Strides[0] != 1)
                {
                    bOp = BlasOp.Transpose;
                    Tensor bNew = bClone.Transpose();
                    bClone.Dispose();
                    bClone = bNew;
                }
                else
                {
                    Tensor bNew = new Tensor(bClone.Allocator, bClone.ElementType, bClone.Sizes[1], bClone.Sizes[0]);
                    Tensor bClone2 = bNew.Transpose();
                    Ops.Copy(bClone2, bClone);
                    bClone.Dispose();
                    bClone = bClone2;
                    bNew.Dispose();

                    bOp = BlasOp.NonTranspose;
                }

                GemmOp(aOp, bOp, alpha, aClone, bClone, beta, cClone);

                if (copyC)
                {
                    Ops.Copy(c, cClone);
                }
            }
            finally
            {
                aClone.Dispose();
                bClone.Dispose();
                cClone.Dispose();
            }
        }

        public static void GemmBatch(float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Sizes[1] != c.Sizes[1] || b.Sizes[2] != c.Sizes[2] || a.Sizes[2] != b.Sizes[1])
            {
                throw new InvalidOperationException("Size mismatch");
            }

            if (TryManagedGemmBatch(alpha, a, b, beta, c))
            {
                return;
            }

            if (UsesManagedBlas(a, b, c))
            {
                int batchSize = (int)c.Sizes[0];
                void ComputeBatch(int batchIndex)
                {
                    using Tensor aBatch = a.Select(0, batchIndex);
                    using Tensor bBatch = b.Select(0, batchIndex);
                    using Tensor cBatch = c.Select(0, batchIndex);
                    Gemm(alpha, aBatch, bBatch, beta, cBatch);
                }

                if (ShouldParallelize(batchSize, (int)(c.Sizes[1] * c.Sizes[2])))
                {
                    Parallel.For(0, batchSize, ComputeBatch);
                }
                else
                {
                    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
                    {
                        ComputeBatch(batchIndex);
                    }
                }

                return;
            }

            BlasOp aOp = default(BlasOp);
            BlasOp bOp = default(BlasOp);
            bool copyC = false;

            Tensor aClone = null;
            Tensor bClone = null;
            Tensor cClone = null;


            if (c.Strides[1] == 1 &&
                c.Strides[2] != 0 && c.Strides[2] != 1)
            {
                // If c is contiguous in dimension 0 (column-major)
                aClone = a.CopyRef();
                bClone = b.CopyRef();
                cClone = c.CopyRef();
            }
            else if (c.Strides[2] == 1 &&
                c.Strides[1] != 0 && c.Strides[1] != 1)
            {
                // If c is contiguous in dimension 1 (row-major)
                // using (a * b)' == b' * a'
                // we can pass row-major matrices to BLAS functions that expect column-major by swapping A and B,
                // and transposing all 3 matrices

                cClone = c.Transpose(1, 2);
                aClone = b.Transpose(1, 2); // Note swap of a and b
                bClone = a.Transpose(1, 2);
            }
            else
            {
                Tensor cNew = new Tensor(c.Allocator, c.ElementType, c.Sizes[0], c.Sizes[2], c.Sizes[1]);
                cClone = cNew.Transpose(1, 2);
                Ops.Copy(cClone, c);
                cNew.Dispose();
                copyC = true;

                aClone = a.CopyRef();
                bClone = b.CopyRef();
            }

            try
            {
                if (aClone.Strides[1] == 1 &&
                    aClone.Strides[2] != 0 && aClone.Strides[2] != 1)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    aOp = BlasOp.NonTranspose;
                }
                else if (aClone.Strides[2] == 1 &&
                    aClone.Strides[1] != 0 && aClone.Strides[1] != 1)
                {
                    aOp = BlasOp.Transpose;
                    Tensor aNew = aClone.Transpose(1, 2);
                    aClone.Dispose();
                    aClone = aNew;
                }
                else
                {
                    Tensor aNew = new Tensor(aClone.Allocator, aClone.ElementType, aClone.Sizes[0], aClone.Sizes[2], aClone.Sizes[1]);
                    Tensor aClone2 = aNew.Transpose(1, 2);
                    Ops.Copy(aClone2, aClone);
                    aClone.Dispose();
                    aClone = aClone2;
                    aNew.Dispose();

                    aOp = BlasOp.NonTranspose;
                }

                if (bClone.Strides[1] == 1 &&
                    bClone.Strides[2] != 0 && bClone.Strides[2] != 1)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    bOp = BlasOp.NonTranspose;
                }
                else if (bClone.Strides[2] == 1 &&
                    bClone.Strides[1] != 0 && bClone.Strides[1] != 1)
                {
                    bOp = BlasOp.Transpose;
                    Tensor bNew = bClone.Transpose(1, 2);
                    bClone.Dispose();
                    bClone = bNew;
                }
                else
                {
                    Tensor bNew = new Tensor(bClone.Allocator, bClone.ElementType, bClone.Sizes[0], bClone.Sizes[2], bClone.Sizes[1]);
                    Tensor bClone2 = bNew.Transpose(1, 2);
                    Ops.Copy(bClone2, bClone);
                    bClone.Dispose();
                    bClone = bClone2;
                    bNew.Dispose();

                    bOp = BlasOp.NonTranspose;
                }

                GemmOpBatch(aOp, bOp, alpha, aClone, bClone, beta, cClone);

                if (copyC)
                {
                    Ops.Copy(c, cClone);
                }
            }
            finally
            {
                aClone.Dispose();
                bClone.Dispose();
                cClone.Dispose();
            }
        }

        [DllImport(mklDllName)]
        public static extern unsafe void cblas_sgemm(Order order, byte transa, byte transb, int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float beta, float* c, int ldc);

        private static void GemmOp(BlasOp transA, BlasOp transB, float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Strides[0] != 1)
            {
                throw new ArgumentException("a must be contiguous in the first dimension (column major / fortran order)");
            }

            if (b.Strides[0] != 1)
            {
                throw new ArgumentException("b must be contiguous in the first dimension (column major / fortran order)");
            }

            if (c.Strides[0] != 1)
            {
                throw new ArgumentException("c must be contiguous in the first dimension (column major / fortran order)");
            }

            unsafe
            {
                // dimensons: (m x k) * (k * n) = (m x n)
                bool nta = transA == BlasOp.NonTranspose;
                bool ntb = transB == BlasOp.NonTranspose;
                byte transa = (byte)transA;
                byte transb = (byte)transB;
                int m = (int)a.Sizes[nta ? 0 : 1];
                int k = (int)b.Sizes[ntb ? 0 : 1];
                int n = (int)b.Sizes[ntb ? 1 : 0];
                int lda = (int)a.Strides[1];
                int ldb = (int)b.Strides[1];
                int ldc = (int)c.Strides[1];

                if (c.ElementType == DType.Float32)
                {
                    try
                    {
                        float* aPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(a);
                        float* bPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(b);
                        float* cPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(c);

                        if (a.Allocator.BlasEnum == BlasEnum.MKL || b.Allocator.BlasEnum == BlasEnum.MKL || c.Allocator.BlasEnum == BlasEnum.MKL)
                        {
                            transa = (byte)ConvertBlasOp(transA);
                            transb = (byte)ConvertBlasOp(transB);
                            cblas_sgemm(Order.Column, transa, transb, m, n, k, alpha, aPtrSingle, lda, bPtrSingle, ldb, beta, cPtrSingle, ldc);
                        }
                        else
                        {
                            ManagedSgemm.Run(System.Text.ASCIIEncoding.ASCII.GetString(&transa, 1), System.Text.ASCIIEncoding.ASCII.GetString(&transb, 1), m, n, k, alpha, aPtrSingle, lda, bPtrSingle, ldb, beta, cPtrSingle, ldc);
                        }
                    }
                    catch (Exception err)
                    {
                        Logger.WriteLine(Logger.Level.err, $"Error message: {err.Message}");
                        Logger.WriteLine(Logger.Level.debug, $"Call stack: {err.StackTrace}");
                        throw;
                    }
                }
                else if (c.ElementType == DType.Float64)
                {
                    double* aPtrDouble = (double*)CpuNativeHelpers.GetBufferStart(a);
                    double* bPtrDouble = (double*)CpuNativeHelpers.GetBufferStart(b);
                    double* cPtrDouble = (double*)CpuNativeHelpers.GetBufferStart(c);
                    double alphaDouble = alpha;
                    double betaDouble = beta;
                    OpenBlasNative.dgemm_(&transa, &transb, &m, &n, &k, &alphaDouble, aPtrDouble, &lda, bPtrDouble, &ldb, &betaDouble, cPtrDouble, &ldc);
                }
                else
                {
                    throw new NotSupportedException("CPU GEMM with element type " + c.ElementType + " not supported");
                }
            }
        }

        [DllImport(mklDllName)]
        public static extern unsafe void cblas_sgemm_batch_strided(Order order, byte transa, byte transb, int m, int n, int k, float alpha, float* a, int lda, int stra, float* b, int ldb, int strb, float beta, float* c, int ldc, int stridec, int batch_size);

        private static void GemmOpBatch(BlasOp transA, BlasOp transB, float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Strides[1] != 1)
            {
                throw new ArgumentException($"a must be contiguous in the first dimension (column major / fortran order). ({a.Strides[0]},{a.Strides[1]}) ({b.Strides[0]},{b.Strides[1]}) ({c.Strides[0]},{c.Strides[1]})");
            }

            if (b.Strides[1] != 1)
            {
                throw new ArgumentException("b must be contiguous in the first dimension (column major / fortran order)");
            }

            if (c.Strides[1] != 1)
            {
                throw new ArgumentException($"c must be contiguous in the first dimension (column major / fortran order) ({a.Strides[0]}, {a.Strides[1]}, {a.Strides[2]}) ({b.Strides[0]}, {b.Strides[1]}, {b.Strides[2]}) ({c.Strides[0]}, {c.Strides[1]}, {c.Strides[2]})");
            }

            unsafe
            {
                bool nta = transA == BlasOp.NonTranspose;
                bool ntb = transB == BlasOp.NonTranspose;
                byte transa = (byte)transA;
                byte transb = (byte)transB;
                int m = (int)a.Sizes[nta ? 1 : 2];
                int k = (int)b.Sizes[ntb ? 1 : 2];
                int n = (int)b.Sizes[ntb ? 2 : 1];
                int lda = (int)a.Strides[2];
                int ldb = (int)b.Strides[2];
                int ldc = (int)c.Strides[2];

                int stra = (int)a.Strides[0];
                int strb = (int)b.Strides[0];
                int strc = (int)c.Strides[0];
                int batchSize = (int)c.Sizes[0];

                if (c.ElementType == DType.Float32)
                {
                    try
                    {
                        float* aPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(a);
                        float* bPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(b);
                        float* cPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(c);

                        transa = (byte)ConvertBlasOp(transA);
                        transb = (byte)ConvertBlasOp(transB);
                        cblas_sgemm_batch_strided(Order.Column, transa, transb, m, n, k, alpha, aPtrSingle, lda, stra, bPtrSingle, ldb, strb, beta, cPtrSingle, ldc, strc, batchSize);
                    }
                    catch (Exception err)
                    {
                        Logger.WriteLine(Logger.Level.err, $"Error message: {err.Message}");
                        Logger.WriteLine(Logger.Level.debug, $"Call stack: {err.StackTrace}");
                        throw;
                    }
                }
                //else if (c.ElementType == DType.Float64)
                //{
                //    CUdeviceptr aPtrDouble = CpuNativeHelpers.GetBufferStart(a);
                //    CUdeviceptr bPtrDouble = CpuNativeHelpers.GetBufferStart(b);
                //    CUdeviceptr cPtrDouble = CpuNativeHelpers.GetBufferStart(c);
                //    double alphaDouble = alpha;
                //    double betaDouble = beta;
                //    CublasStatus _statusF64 = CudaBlasNativeMethods.cublasDgemmStridedBatched(blas.Value.CublasHandle,
                //        transa, transb, m, n, k, ref alphaDouble, aPtrDouble, lda, stra, bPtrDouble, ldb, strb, ref betaDouble, cPtrDouble, ldc, strc, batchSize);
                //    if (_statusF64 != CublasStatus.Success)
                //    {
                //        throw new CudaBlasException(_statusF64);
                //    }
                //}
                else
                {
                    throw new NotSupportedException("GEMM Batch with element type " + c.ElementType + " not supported");
                }
            }
        }
    }
}
