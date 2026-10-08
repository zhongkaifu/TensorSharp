// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory;

public enum MemoryTier { Host, Accelerator }
public enum ResourceKind { Weight, Expert, KvPage, RecurrentState, Prefix, Adapter, Activation, Workspace, Multimodal, Other }
public enum ResourceAccess { Read, Write }

/// <summary>Owner must distinguish model revision, request/tenant and adapters as
/// applicable. Epoch distinguishes reloads and recycled native allocations.</summary>
public readonly record struct ResourceKey(string Owner, long Epoch, string Name);

/// <summary>Placement is identified by node and device/NUMA location, not a model name.</summary>
public readonly record struct MemoryLocation(string Node, string Device, MemoryTier Tier);

public sealed record MemoryResource(ResourceKey Key, long ByteLength, ResourceKind Kind,
    bool Mutable = false, string Layout = "opaque", double ReloadCost = 1)
{
    internal void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(Key.Owner);
        ArgumentException.ThrowIfNullOrWhiteSpace(Key.Name);
        ArgumentOutOfRangeException.ThrowIfNegative(Key.Epoch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ByteLength);
        ArgumentException.ThrowIfNullOrWhiteSpace(Layout);
        if (!double.IsFinite(ReloadCost) || ReloadCost <= 0) throw new ArgumentOutOfRangeException(nameof(ReloadCost));
    }
}

/// <summary>Read exactly destination.Length bytes or throw. Implementations must
/// quiesce any outstanding I/O before returning or throwing, including cancellation.</summary>
public interface IResourceSource
{
    long ByteLength { get; }
    ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default);
}

/// <summary>Opaque device allocation. Pointer is only valid while its residency lease
/// is held. Methods complete transfers before returning; Dispose must free physically
/// and must not return the allocation to an unaccounted private pool.</summary>
public interface IResourceBuffer : IResourceSource, IDisposable
{
    nint Pointer { get; }
    ValueTask WriteAsync(long offset, ReadOnlyMemory<byte> source, CancellationToken cancellationToken = default);
}

public interface IMemoryBackend
{
    MemoryLocation Location { get; }
    /// <summary>Upper bound on physical charges, including alignment, mirrors and UMA
    /// constraints. No allocation may exceed this bound; driver overhead is headroom.</summary>
    IReadOnlyList<MemoryCharge> GetAllocationCharges(long byteLength);
    ValueTask<IResourceBuffer> AllocateAsync(long byteLength, CancellationToken cancellationToken = default);
}

public readonly record struct ResidencySnapshot(ResourceKey Resource, MemoryLocation Location,
    long Version, long Bytes, int Readers, bool Writer, long LastUse);
public readonly record struct MemorySchedulerStats(long Loads, long Hits, long Evictions, long Spills,
    long TransferBytes, int Resources, int ActiveLeases);

internal static class MemoryRange
{
    internal static void Check(long total, long offset, int count)
    {
        if (offset < 0 || count < 0 || offset > total || count > total - offset)
            throw new ArgumentOutOfRangeException(nameof(offset));
    }
    internal static long Align(long value, long alignment) => checked((value + alignment - 1) / alignment * alignment);
}
