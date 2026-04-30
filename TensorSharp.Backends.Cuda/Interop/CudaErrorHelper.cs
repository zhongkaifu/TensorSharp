using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static class CudaErrorHelper
    {
        public static void ThrowOnError(this int result)
        {
            if (result == 0)
                return;

            string message = "Unknown CUDA error";
            if (CudaDriverApi.cuGetErrorString(result, out IntPtr strPtr) == 0 && strPtr != IntPtr.Zero)
                message = Marshal.PtrToStringAnsi(strPtr) ?? message;

            throw new CudaException(result, message);
        }

        public static void ThrowOnCublasError(this int result)
        {
            if (result == 0)
                return;

            string message = result switch
            {
                1 => "CUBLAS_STATUS_NOT_INITIALIZED",
                3 => "CUBLAS_STATUS_ALLOC_FAILED",
                7 => "CUBLAS_STATUS_INVALID_VALUE",
                8 => "CUBLAS_STATUS_ARCH_MISMATCH",
                11 => "CUBLAS_STATUS_MAPPING_ERROR",
                13 => "CUBLAS_STATUS_EXECUTION_FAILED",
                14 => "CUBLAS_STATUS_INTERNAL_ERROR",
                15 => "CUBLAS_STATUS_NOT_SUPPORTED",
                _ => "Unknown cuBLAS error",
            };

            throw new CudaException(result, $"cuBLAS: {message}");
        }
    }
}
