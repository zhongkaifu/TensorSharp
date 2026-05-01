using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed class CudaAllocator : IAllocator, IDisposable
    {
        private int disposed;

        public CudaAllocator(int deviceId = 0)
        {
            CudaBackend.Register();

            CudaContext context = null;
            CudaStream stream = null;
            CudaCublasHandle blas = null;
            CudaKernels kernels = null;

            try
            {
                context = CudaContext.Create(deviceId);
                context.MakeCurrent();
                stream = CudaStream.Create();
                blas = CudaCublasHandle.Create();
                blas.SetStream(stream.Handle);
                kernels = CudaKernels.TryCreate();
            }
            catch
            {
                kernels?.Dispose();
                blas?.Dispose();
                stream?.Dispose();
                context?.Dispose();
                throw;
            }

            Context = context;
            Stream = stream;
            Blas = blas;
            Kernels = kernels;
            DeviceId = deviceId;
        }

        public BlasEnum BlasEnum => BlasEnum.CUDA;

        public int DeviceId { get; }

        internal CudaContext Context { get; }

        internal CudaStream Stream { get; }

        internal CudaCublasHandle Blas { get; }

        internal CudaKernels Kernels { get; }

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CudaStorage(this, elementType, elementCount);
        }

        public float GetAllocatedMemoryRatio()
        {
            Context.MakeCurrent();
            CudaDriverApi.cuMemGetInfo(out UIntPtr free, out UIntPtr total).ThrowOnError();
            ulong totalBytes = total.ToUInt64();
            if (totalBytes == 0)
                return 0.0f;

            ulong freeBytes = free.ToUInt64();
            return (float)(1.0 - (double)freeBytes / totalBytes);
        }

        public void Synchronize()
        {
            Context.MakeCurrent();
            Stream.Synchronize();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposed, 1) != 0)
                return;

            Context.MakeCurrent();
            Kernels?.Dispose();
            Blas.Dispose();
            Stream.Dispose();
            Context.Dispose();
        }
    }
}
