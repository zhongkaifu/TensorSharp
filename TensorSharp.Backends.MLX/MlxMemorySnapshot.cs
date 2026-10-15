namespace TensorSharp.MLX
{
    /// <summary>Process-wide MLX allocator gauges sampled without forcing lazy
    /// evaluation or Metal synchronization. Active/cache can overlap process RSS;
    /// do not sum them with RSS or infer scope ownership from their totals.</summary>
    public readonly record struct MlxMemorySnapshot(ulong ActiveBytes, ulong CacheBytes, ulong PeakBytes);
}
