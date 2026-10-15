// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory;

/// <summary>A byte-exact view, useful for expert rows, tensor tiles and checkpoints.
/// The operator adapter owns layout/alignment correctness. Source is borrowed.</summary>
public sealed class ResourceSlice : IResourceSource
{
    private readonly IResourceSource _source;
    private readonly long _offset;
    public ResourceSlice(IResourceSource source, long offset, long byteLength)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (offset < 0 || byteLength <= 0 || offset > source.ByteLength || byteLength > source.ByteLength - offset)
            throw new ArgumentOutOfRangeException(nameof(offset));
        _source = source;
        _offset = offset;
        ByteLength = byteLength;
    }
    public long ByteLength { get; }
    public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
    {
        MemoryRange.Check(ByteLength, offset, destination.Length);
        return _source.ReadAsync(checked(_offset + offset), destination, cancellationToken);
    }
}
