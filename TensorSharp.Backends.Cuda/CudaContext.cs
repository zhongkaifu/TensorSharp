using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaContext : IDisposable
    {
        private IntPtr context;

        private CudaContext(IntPtr context, int deviceId)
        {
            this.context = context;
            DeviceId = deviceId;
        }

        public int DeviceId { get; }

        public IntPtr Handle => context;

        public static CudaContext Create(int deviceId)
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGet(out int device, deviceId).ThrowOnError();
            CudaDriverApi.cuCtxCreate(out IntPtr context, 0, device).ThrowOnError();
            return new CudaContext(context, deviceId);
        }

        public void MakeCurrent()
        {
            if (context == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaContext));

            CudaDriverApi.cuCtxSetCurrent(context).ThrowOnError();
        }

        public void Dispose()
        {
            IntPtr handle = Interlocked.Exchange(ref context, IntPtr.Zero);
            if (handle != IntPtr.Zero)
                CudaDriverApi.cuCtxDestroy(handle);
        }
    }
}
