// Copyright (c) Zhongkai Fu. All rights reserved.
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
using System.Reflection.Metadata;
using System.Runtime.InteropServices;
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
                            SGEMM sgemm = new SGEMM();
                            sgemm.Run(System.Text.ASCIIEncoding.ASCII.GetString(&transa, 1), System.Text.ASCIIEncoding.ASCII.GetString(&transb, 1), m, n, k, alpha, aPtrSingle, lda, bPtrSingle, ldb, beta, cPtrSingle, ldc);
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
