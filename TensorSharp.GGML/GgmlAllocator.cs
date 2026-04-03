using System;

namespace TensorSharp.GGML
{
    [Serializable]
    public class GgmlAllocator : IAllocator
    {
        private readonly GgmlContext context;
        private readonly int deviceId;

        public GgmlAllocator(GgmlContext context, int deviceId)
        {
            this.context = context ?? throw new ArgumentNullException(nameof(context));
            this.deviceId = deviceId;
        }

        public BlasEnum BlasEnum => context.BackendType == GgmlBackendType.Metal ? BlasEnum.GGML_METAL : BlasEnum.GGML_CPU;

        public int DeviceId => deviceId;

        public GgmlContext Context => context;

        public Storage Allocate(DType elementType, long elementCount)
        {
            if (elementType == DType.Float16)
            {
                throw new NotSupportedException("The GGML Metal backend currently supports Float32 tensors only. Disable AMP to use this backend.");
            }

            return new GgmlStorage(this, context, elementType, elementCount);
        }

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }
    }
}
