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
