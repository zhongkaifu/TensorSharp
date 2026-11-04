// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>Bounded reads of one original tensor; never exposes another object's bytes.</summary>
internal sealed unsafe class NinferTensorWeightSource : ICudaWeightSource
{
    private readonly NinferArtifactReader _reader;
    private readonly NinferTensorDescriptor _tensor;
    private readonly bool _attentionPermutation;
    private readonly int _rowBytes;

    internal NinferTensorWeightSource(NinferArtifactReader reader, NinferTensorDescriptor tensor, bool attentionPermutation = false)
    {
        reader.ReadObjectRange(tensor, 0, Span<byte>.Empty);
        if (attentionPermutation && (tensor.Format != NinferTensorFormat.BFloat16
            || tensor.Shape.Count != 2 || tensor.Shape[0] != 14336 || tensor.Shape[1] > int.MaxValue / 2))
            throw new ArgumentException("Expected original BF16 attention Q|K|gate|V matrix.");
        _reader = reader; _tensor = tensor; _attentionPermutation = attentionPermutation;
        _rowBytes = attentionPermutation ? checked((int)tensor.Shape[1] * 2) : 0;
    }

    internal static int AttentionSourceRow(int destinationRow)
    {
        if ((uint)destinationRow >= 14336) throw new ArgumentOutOfRangeException(nameof(destinationRow));
        if (destinationRow < 12288)
            return destinationRow / 512 * 256 + destinationRow % 256 + (destinationRow % 512 >= 256 ? 7168 : 0);
        return destinationRow < 13312 ? 6144 + destinationRow - 12288 : destinationRow;
    }

    public void Read(long offset, IntPtr dst, long bytes)
    {
        CheckRange(offset, bytes, checked((long)_tensor.ByteLength), dst);
        if (bytes == 0) { _reader.ReadObjectRange(_tensor, checked((ulong)offset), Span<byte>.Empty); return; }
        long done = 0;
        while (done < bytes)
        {
            long position = checked(offset + done), original = position;
            int count = checked((int)Math.Min(8 << 20, bytes - done));
            if (_attentionPermutation)
            {
                int row = checked((int)(position / _rowBytes)), inRow = checked((int)(position % _rowBytes));
                // Each 256-row Q/gate head is contiguous in the original file.
                // K and V each form one further contiguous region. Stop exactly
                // where that row mapping jumps, retaining the 8 MiB read cap.
                int runEnd = row < 12288 ? (row / 256 + 1) * 256 : row < 13312 ? 13312 : 14336;
                long contiguousBytes = checked((long)(runEnd - row) * _rowBytes - inRow);
                count = checked((int)Math.Min(count, contiguousBytes));
                original = checked((long)AttentionSourceRow(row) * _rowBytes + inRow);
            }
            _reader.ReadObjectRange(_tensor, checked((ulong)original), new Span<byte>((byte*)dst + done, count));
            done += count;
        }
    }

    internal static void CheckRange(long offset, long bytes, long total, IntPtr dst)
    {
        if (offset < 0 || bytes < 0 || offset > total || bytes > total - offset)
            throw new ArgumentOutOfRangeException(nameof(offset), "Weight read must stay inside its declared tensor.");
        if (bytes > 0 && (dst.ToInt64() <= 0 || unchecked((ulong)dst.ToInt64()) > (ulong)long.MaxValue - (ulong)bytes))
            throw new ArgumentException("Nonempty weight read requires a nonzero, nonoverflowing host pointer.", nameof(dst));
    }
}

/// <summary>Virtual code|scale planes. One cached converted tile, no whole matrix materialization.</summary>
internal sealed unsafe class NinferNvfp4WeightSource : ICudaNinferOriginalProjectionSource, IDisposable
{
    internal const int MaximumOriginalReadBytes = 128 << 20;
    private readonly object _gate = new();
    private readonly NinferNvfp4PlaneReader _planes;
    private NinferNvfp4PlaneTile? _tile;
    private int _tileIndex = -1;
    private bool _disposed;
    internal long CodeBytes { get; }
    internal long Bytes { get; }
    public long OriginalBytes => checked((long)_planes.OriginalWeights.ByteLength);
    public int Rows => _planes.Rows;
    public int Columns => _planes.Columns;
    public CudaNinferOriginalRowMapping RowMapping => _planes.RowMapping switch
    {
        NinferNvfp4RowMapping.Identity => CudaNinferOriginalRowMapping.Identity,
        NinferNvfp4RowMapping.GateUpRows8 => CudaNinferOriginalRowMapping.GateUpRows8,
        NinferNvfp4RowMapping.AttentionQueryGateKeyValue => CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue,
        _ => throw new NotSupportedException("Unsupported original NInfer row mapping.")
    };

    internal NinferNvfp4WeightSource(NinferNvfp4PlaneReader planes)
    {
        _planes = planes;
        CodeBytes = checked((long)planes.Rows * planes.Columns / 2);
        Bytes = checked(CodeBytes + (long)planes.Rows * planes.Columns / 16);
    }

    public void Read(long offset, IntPtr dst, long bytes)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            NinferTensorWeightSource.CheckRange(offset, bytes, Bytes, dst);
            long done = 0;
            while (done < bytes)
            {
                long at = checked(offset + done);
                bool codes = at < CodeBytes;
                long relative = codes ? at : at - CodeBytes;
                int tileBytes = codes ? _planes.CodeBytesPerTile : _planes.ScaleBytesPerTile;
                int index = checked((int)(relative / tileBytes)), inTile = checked((int)(relative % tileBytes));
                if (index != _tileIndex) { _tile = _planes.ReadTile(index); _tileIndex = index; }
                ReadOnlySpan<byte> plane = codes ? _tile!.Codes.Span : _tile!.Scales.Span;
                int take = checked((int)Math.Min(bytes - done, tileBytes - inTile));
                plane.Slice(inTile, take).CopyTo(new Span<byte>((byte*)dst + done, take));
                done += take;
            }
        }
    }

    public void ReadOriginal(long offset, IntPtr dst, long bytes)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            NinferTensorWeightSource.CheckRange(offset, bytes, OriginalBytes, dst);
            if (bytes == 0)
            {
                _planes.ReadOriginalRange(checked((ulong)offset), Span<byte>.Empty);
                return;
            }
            for (long done = 0; done < bytes;)
            {
                int count = checked((int)Math.Min(MaximumOriginalReadBytes, bytes - done));
                _planes.ReadOriginalRange(checked((ulong)(offset + done)), new Span<byte>((byte*)dst + done, count));
                done += count;
            }
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true; _tile = null; _tileIndex = -1; _planes.Dispose();
        }
    }
}
