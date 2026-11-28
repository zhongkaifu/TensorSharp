using System;

namespace TensorSharp.MLX
{
    [Serializable]
    public sealed class MlxAllocator : IAllocator, IDisposable
    {
        private bool disposed;

        public MlxAllocator(int deviceId = 0)
        {
            MlxBackend.Register();
            MlxBackend.EnsureAvailable(deviceId);
            DeviceId = deviceId;
            // MLX latches its GEMM precision on its first GEMM, once per process (D04: process scope): the precision this
            // process runs with is claimed when the first allocator comes up, so a family that needs another one later
            // (LTX 2.5's Float32 denoiser) is told instead of silently running with this one.
            string tf32 = Environment.GetEnvironmentVariable("MLX_ENABLE_TF32");
            ProcessSettings.TryClaim("MLX_ENABLE_TF32", string.IsNullOrWhiteSpace(tf32) ? "default" : tf32.Trim(), "the MLX backend", out _);
        }

        public BlasEnum BlasEnum => BlasEnum.MLX;

        public int DeviceId { get; }

        public Storage Allocate(DType elementType, long elementCount)
        {
            ThrowIfDisposed();
            return new MlxStorage(this, elementType, elementCount);
        }

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }

        public MlxMemorySnapshot GetMemorySnapshot()
        {
            return MlxBackend.GetMemorySnapshot();
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(MlxAllocator));
        }

        public void Dispose()
        {
            if (disposed)
                return;

            disposed = true;
            MlxQuantizedOps.ClearDeviceCache(DeviceId);
            MlxBackend.ClearCache();
        }
    }
}
