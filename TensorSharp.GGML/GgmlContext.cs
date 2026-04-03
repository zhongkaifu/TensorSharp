using System;
using System.Reflection;

namespace TensorSharp.GGML
{
    public sealed class GgmlContext
    {
        internal GgmlMemoryPool MemoryPool { get; }

        public GgmlContext(int[] deviceIds, GgmlBackendType backendType)
        {
            if (deviceIds == null || deviceIds.Length == 0)
            {
                throw new ArgumentException("At least one device id is required for the GGML backend.", nameof(deviceIds));
            }

            if (deviceIds.Length != 1)
            {
                throw new NotSupportedException("GGML backends currently support a single device only.");
            }

            DeviceId = deviceIds[0];
            MemoryPool = new GgmlMemoryPool();
            MemoryPool.EnsureInitialBlocks();
            BackendType = backendType;
            GgmlNative.EnsureAvailable(backendType);
            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        public int DeviceId { get; }

        public GgmlBackendType BackendType { get; }
    }
}
