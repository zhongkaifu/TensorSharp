using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed class CudaAllocator : IAllocator, IDisposable
    {
        public CudaAllocator(int deviceId = 0)
        {
            CudaBackend.Register();
            Context = CudaContext.Create(deviceId);
            Context.MakeCurrent();
            Stream = CudaStream.Create();
            Blas = CudaCublasHandle.Create();
            Blas.SetStream(Stream.Handle);
            Kernels = CudaKernels.TryCreate();
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
            Kernels?.Dispose();
            Blas.Dispose();
            Stream.Dispose();
            Context.Dispose();
        }
    }
}
