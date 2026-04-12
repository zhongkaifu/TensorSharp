// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using System.Runtime.InteropServices;

namespace TensorSharp.Cpu
{
    // When used with 64bit openblas, this interface requires that it is compiled with 32-bit ints
    public static class OpenBlasNative
    {
        private const string dll = "libopenblas.dll";
        private const CallingConvention cc = CallingConvention.Cdecl;

        [DllImport(dll, CallingConvention = cc)]
        public static extern unsafe void sgemm_(byte* transa, byte* transb, int* m, int* n, int* k,
            float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);

        [DllImport(dll, CallingConvention = cc)]
        public static extern unsafe void dgemm_(byte* transa, byte* transb, int* m, int* n, int* k,
            double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);


        [DllImport(dll, CallingConvention = cc)]
        public static extern unsafe void sgemv_(byte* trans, int* m, int* n,
            float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy);

        [DllImport(dll, CallingConvention = cc)]
        public static extern unsafe void dgemv_(byte* trans, int* m, int* n,
            double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);


        [DllImport(dll, CallingConvention = cc)]
        public static extern unsafe float sdot_(int* n, float* x, int* incx, float* y, int* incy);

        [DllImport(dll, CallingConvention = cc)]
        public static extern unsafe double ddot_(int* n, double* x, int* incx, double* y, int* incy);
    }
}
