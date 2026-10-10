// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>A logical row concatenation over borrowed immutable sources. The
/// model's catalog owns their file handles; this view neither materializes nor
/// takes ownership of any source. Reads may cross tensor boundaries.</summary>
internal sealed class ConcatenatedWeightSource : IResourceSource
{
    private readonly IResourceSource[] _sources;
    private readonly long[] _ends;

    internal ConcatenatedWeightSource(params IResourceSource[] sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        if (sources.Length < 2) throw new ArgumentException("Concatenation requires at least two sources.", nameof(sources));
        _sources = (IResourceSource[])sources.Clone();
        _ends = new long[sources.Length];
        long length = 0;
        for (int i = 0; i < _sources.Length; i++)
        {
            ArgumentNullException.ThrowIfNull(_sources[i]);
            if (_sources[i].ByteLength <= 0) throw new ArgumentException("Sources must be nonempty.", nameof(sources));
            length = checked(length + _sources[i].ByteLength);
            _ends[i] = length;
        }
        ByteLength = length;
    }

    public long ByteLength { get; }

    public async ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
    {
        if (offset < 0 || offset > ByteLength || destination.Length > ByteLength - offset)
            throw new ArgumentOutOfRangeException(nameof(offset));
        cancellationToken.ThrowIfCancellationRequested();
        for (int i = 0; !destination.IsEmpty; i++)
        {
            if (offset >= _ends[i]) continue;
            long start = i == 0 ? 0 : _ends[i - 1];
            int count = checked((int)Math.Min(destination.Length, _ends[i] - offset));
            // Await every segment before returning, including on failure; no
            // outstanding I/O can retain the reusable staging buffer.
            await _sources[i].ReadAsync(offset - start, destination[..count], cancellationToken).ConfigureAwait(false);
            offset += count;
            destination = destination[count..];
        }
    }
}
