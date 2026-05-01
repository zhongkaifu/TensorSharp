using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaCublasHandle : IDisposable
    {
        private IntPtr handle;

        private CudaCublasHandle(IntPtr handle)
        {
            this.handle = handle;
        }

        public IntPtr Handle => handle;

        public static CudaCublasHandle Create()
        {
            CublasApi.cublasCreate(out IntPtr handle).ThrowOnCublasError();
            try
            {
                CublasApi.cublasSetMathMode(handle, CublasApi.CUBLAS_TENSOR_OP_MATH).ThrowOnCublasError();
                return new CudaCublasHandle(handle);
            }
            catch
            {
                CublasApi.cublasDestroy(handle);
                throw;
            }
        }

        public void SetStream(IntPtr stream)
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaCublasHandle));

            CublasApi.cublasSetStream(handle, stream).ThrowOnCublasError();
        }

        public void Dispose()
        {
            IntPtr nativeHandle = Interlocked.Exchange(ref handle, IntPtr.Zero);
            if (nativeHandle != IntPtr.Zero)
                CublasApi.cublasDestroy(nativeHandle);
        }
    }
}
