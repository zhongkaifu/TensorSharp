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

    internal NinferTensorWeightSource(NinferArtifactReader reader, NinferTensorDescriptor tensor)
    {
        reader.ReadObjectRange(tensor, 0, Span<byte>.Empty);
        _reader = reader; _tensor = tensor;
    }

    public void Read(long offset, IntPtr dst, long bytes)
    {
        CheckRange(offset, bytes, checked((long)_tensor.ByteLength), dst);
        if (bytes == 0) { _reader.ReadObjectRange(_tensor, checked((ulong)offset), Span<byte>.Empty); return; }
        for (long done = 0; done < bytes;)
        {
            int count = checked((int)Math.Min(8 << 20, bytes - done));
            _reader.ReadObjectRange(_tensor, checked((ulong)(offset + done)), new Span<byte>((byte*)dst + done, count));
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

/// <summary>Row-planar NInfer matrix (FP8 row-scaled or q8_g32: codes [rows][k], then per-row scale
/// words) read in a destination row order: every destination row's codes and its scale words come
/// from one source row. Encoded bytes are copied, never converted. Contiguous source runs are read
/// in one range.</summary>
internal sealed unsafe class NinferRowPlanarWeightSource : ICudaWeightSource
{
    private readonly NinferArtifactReader _reader;
    private readonly NinferTensorDescriptor _tensor;
    private readonly Func<int, int> _sourceRow;
    private readonly int _rows, _codeRowBytes, _scaleRowBytes;
    private readonly long _codeBytes, _scaleOffset;
    internal long Bytes { get; }

    internal NinferRowPlanarWeightSource(NinferArtifactReader reader, NinferTensorDescriptor tensor, Func<int, int>? sourceRow = null)
    {
        reader.ReadObjectRange(tensor, 0, Span<byte>.Empty);
        if (tensor.Shape.Count != 2) throw new ArgumentException("Expected a rank-2 matrix.");
        _rows = checked((int)tensor.Shape[0]);
        int columns = checked((int)tensor.Shape[1]);
        (_codeRowBytes, _scaleRowBytes) = tensor.Format switch
        {
            NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale => (columns, 2),
            NinferTensorFormat.W8G32Float16Scale when tensor.Geometry.PaddedK == (ulong)columns => (columns, columns / 32 * 2),
            _ => throw new NotSupportedException("Row-planar sources take FP8 row-scaled or unpadded q8_g32 matrices.")
        };
        _reader = reader; _tensor = tensor; _sourceRow = sourceRow ?? (r => r);
        _codeBytes = (long)_rows * _codeRowBytes;
        _scaleOffset = checked((long)tensor.Geometry.ScalePlaneOffset!.Value);
        Bytes = _codeBytes + (long)_rows * _scaleRowBytes;
    }

    public void Read(long offset, IntPtr dst, long bytes)
    {
        NinferTensorWeightSource.CheckRange(offset, bytes, Bytes, dst);
        long done = 0;
        while (done < bytes)
        {
            long at = offset + done;
            bool codes = at < _codeBytes;
            int rowBytes = codes ? _codeRowBytes : _scaleRowBytes;
            long plane = codes ? at : at - _codeBytes;
            int row = checked((int)(plane / rowBytes)), inRow = checked((int)(plane % rowBytes));
            int source = _sourceRow(row), run = 1;
            long want = Math.Min(bytes - done, (codes ? _codeBytes : Bytes - _codeBytes) - plane);
            // Extend over destination rows whose source rows continue contiguously.
            while ((long)run * rowBytes - inRow < want && row + run < _rows && _sourceRow(row + run) == source + run && run < 4096) run++;
            int take = checked((int)Math.Min(want, Math.Min((long)run * rowBytes - inRow, 8L << 20)));
            ulong origin = checked((ulong)((codes ? 0 : _scaleOffset) + (long)source * rowBytes + inRow));
            _reader.ReadObjectRange(_tensor, origin, new Span<byte>((byte*)dst + done, take));
            done += take;
        }
    }

    /// <summary>Gate|up halves interleaved by 8 rows: destination 16-row group g = gate rows 8g..8g+7, then up rows 8g..8g+7.</summary>
    internal static Func<int, int> GateUpRows8(int rows) => r => r / 16 * 8 + r % 8 + (r % 16 >= 8 ? rows / 2 : 0);

    /// <summary>Attention Q6144|K1024|gate6144|V1024 to [Q head, gate head] x 24, K, V.</summary>
    internal static int AttentionQueryGateKeyValue(int r)
        => r < 12288 ? r / 512 * 256 + r % 256 + (r % 512 >= 256 ? 7168 : 0) : r < 13312 ? 6144 + r - 12288 : r;
}
