// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Memory;

namespace TensorSharp.Runtime;

public readonly record struct GgufFileRegion(string Path, long Offset, long ByteLength);

public partial class GgufFile
{
    /// <summary>Resolve a tensor to its actual shard without mmap or reading its data.
    /// Only tensor metadata owned by this reader is accepted.</summary>
    public GgufFileRegion GetTensorFileRegion(string tensorName)
    {
        var tensor = Tensors[tensorName];
        var owner = _tensorOwner.TryGetValue(tensorName, out var shard) ? shard : this;
        long rows = 1;
        if (tensor.Shape.Length == 0) throw new InvalidDataException("Tensor shape cannot be empty.");
        long width = checked((long)tensor.Shape[0]);
        long block = GetBlockSize(tensor.Type);
        if (width <= 0 || width % block != 0) throw new InvalidDataException("Tensor width is not quantization-block aligned.");
        foreach (ulong dim in tensor.Shape.AsSpan(1))
        {
            if (dim == 0) throw new InvalidDataException("Empty tensor dimension.");
            rows = checked(rows * checked((long)dim));
        }
        long bytes = checked(checked(width / block * GetTypeSize(tensor.Type)) * rows);
        long offset = checked(owner.DataOffset + checked((long)tensor.Offset));
        long fileLength = owner._stream.Length;
        if (offset < 0 || offset > fileLength || bytes > fileLength - offset)
            throw new InvalidDataException($"Tensor {tensorName} extends outside its GGUF shard.");
        return new(Path.GetFullPath(owner._path), offset, bytes);
    }

    /// <summary>Model-neutral on-disk tensor catalog. No model-name switches, no
    /// tensor payload allocation. Identity must identify the exact immutable model
    /// revision (and any transformations); epoch distinguishes reloads.</summary>
    public GgufMemoryCatalog CreateMemoryCatalog(string modelIdentity, long epoch = 0)
        => new(this, modelIdentity, epoch);
}

/// <summary>Owns one read handle per shard. Dispose after unregistering resources.
/// This catalog is an opt-in loader seam; existing model constructors do not become
/// out-of-core merely by constructing it.</summary>
public sealed class GgufMemoryCatalog : IDisposable
{
    private readonly Dictionary<string, FileDataSource> _files = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (MemoryResource Resource, FileRegionSource Source)> _resources = new(StringComparer.Ordinal);
    private bool _disposed;

    internal GgufMemoryCatalog(GgufFile gguf, string identity, long epoch)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identity);
        ArgumentOutOfRangeException.ThrowIfNegative(epoch);
        try
        {
            foreach (var tensor in gguf.Tensors.Values)
            {
                var region = gguf.GetTensorFileRegion(tensor.Name);
                if (!_files.TryGetValue(region.Path, out var file))
                    _files.Add(region.Path, file = new FileDataSource(region.Path));
                var resource = new MemoryResource(new(identity, epoch, tensor.Name), region.ByteLength,
                    ResourceKind.Weight, Layout: $"gguf:{tensor.Type}:{string.Join('x', tensor.Shape)}");
                _resources.Add(tensor.Name, (resource, new FileRegionSource(file, region.Offset, region.ByteLength)));
            }
        }
        catch { Dispose(); throw; }
    }

    public IEnumerable<MemoryResource> Resources
    {
        get { ObjectDisposedException.ThrowIf(_disposed, this); foreach (var item in _resources.Values) yield return item.Resource; }
    }

    /// <summary>Return a tensor. Source shares its shard handle.</summary>
    public (MemoryResource Resource, IResourceSource Source) Get(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var item = _resources[name];
        return (item.Resource, item.Source);
    }

    /// <summary>The caller defines expert/row/tile boundaries from its layout; the
    /// scheduler never guesses them from tensor names. No payload is copied.</summary>
    public (MemoryResource Resource, IResourceSource Source) GetSlice(string tensorName, string resourceName,
        long offset, long byteLength, ResourceKind kind = ResourceKind.Weight)
    {
        var item = Get(tensorName);
        var source = new ResourceSlice(item.Source, offset, byteLength);
        ArgumentException.ThrowIfNullOrWhiteSpace(resourceName);
        return (item.Resource with { Key = item.Resource.Key with { Name = resourceName }, ByteLength = byteLength, Kind = kind }, source);
    }

    public void Dispose()
    {
        if (_disposed) return;
        foreach (var file in _files.Values) file.Dispose();
        _disposed = true;
    }
}
