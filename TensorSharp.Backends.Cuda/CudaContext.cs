using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaContext : IDisposable
    {
        private IntPtr context;
        private readonly int device;

        private CudaContext(IntPtr context, int deviceId, int device)
        {
            this.context = context;
            DeviceId = deviceId;
            this.device = device;
        }

        public int DeviceId { get; }

        public IntPtr Handle => context;

        public static CudaContext Create(int deviceId)
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGet(out int device, deviceId).ThrowOnError();

            // cuBLAS and the CUDA runtime use the device primary context. Retaining
            // it keeps the direct backend compatible with GGML CUDA probing and WSL.
            CudaDriverApi.cuDevicePrimaryCtxRetain(out IntPtr context, device).ThrowOnError();

            try
            {
                CudaDriverApi.cuCtxSetCurrent(context).ThrowOnError();
            }
            catch
            {
                CudaDriverApi.cuDevicePrimaryCtxRelease(device);
                throw;
            }

            return new CudaContext(context, deviceId, device);
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
            if (handle == IntPtr.Zero)
                return;

            if (CudaDriverApi.cuCtxGetCurrent(out IntPtr current) == 0 && current == handle)
                CudaDriverApi.cuCtxSetCurrent(IntPtr.Zero);

            CudaDriverApi.cuDevicePrimaryCtxRelease(device);
        }
    }
}
