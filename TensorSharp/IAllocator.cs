namespace TensorSharp
{
    public enum BlasEnum
    {
        DotNet,
        MKL,
        CUDA,
        GGML_METAL,
        GGML_CPU
    }


    public interface IAllocator
    {
        BlasEnum BlasEnum { get; }
        int DeviceId { get; }
        Storage Allocate(DType elementType, long elementCount);

        float GetAllocatedMemoryRatio();
    }
}
