namespace TensorSharp.MLX
{
    public readonly record struct MlxMemorySnapshot(ulong ActiveBytes, ulong CacheBytes, ulong PeakBytes);
}
