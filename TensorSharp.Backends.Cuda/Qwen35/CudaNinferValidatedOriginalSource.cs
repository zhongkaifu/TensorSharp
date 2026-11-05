// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;

namespace TensorSharp.Cuda;

/// <summary>Single-loader adapter for an original NVFP4 upload. It is not thread-safe:
/// each synchronous matrix load owns one instance. It never modifies payload bytes.
/// RequireCompleteValidation must succeed before the matrix is repacked or published.</summary>
internal sealed unsafe class CudaNinferValidatedOriginalSource : ICudaWeightSource
{
    internal const long MaximumOriginalBytes = 128L << 20;
    internal const int MaximumCoverageIntervals = 4096;
    private readonly record struct ReadRange(long Begin, long End);
    private readonly List<ReadRange> _coverage = new(4);
    private readonly int _rows, _columns;
    private readonly CudaNinferOriginalRowMapping _mapping;
    private readonly long _codeBytes, _scaleEnd;
    private readonly uint _expectedDivisorBits;
    private uint _divisorBits;
    private byte _divisorByteMask;
    private bool _faulted, _validated;
    internal ICudaNinferOriginalProjectionSource Original { get; }
    internal long Bytes { get; }
    internal bool IsValidated => _validated && !_faulted;

    internal CudaNinferValidatedOriginalSource(CudaNinferProjectionDesc descriptor)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        if (descriptor.Format != CudaNinferProjectionFormat.NvFp4Planes || descriptor.SourceOffset != 0
            || descriptor.Source is not ICudaNinferOriginalProjectionSource original)
            throw new NotSupportedException("Original NVFP4 upload requires a tensor-relative source capability and zero virtual-plane offset.");
        if (original.Rows != descriptor.Rows || original.Columns != descriptor.Columns
            || original.Rows <= 0 || original.Rows % 128 != 0 || original.Columns <= 0 || original.Columns % 64 != 0)
            throw new ArgumentException("Original source geometry differs from the projection descriptor.", nameof(descriptor));
        _codeBytes = checked((long)descriptor.Rows * descriptor.Columns / 2);
        _scaleEnd = checked(_codeBytes + (long)descriptor.Rows * descriptor.Columns / 16);
        Bytes = checked(descriptor.Bytes + sizeof(float));
        if (Bytes != checked(_scaleEnd + sizeof(float)) || original.OriginalBytes != Bytes || Bytes > MaximumOriginalBytes)
            throw new NotSupportedException("Original NVFP4 upload exceeds the 128 MiB staging bound or differs from its complete encoded object. Use the existing CPU-plane load for larger matrices.");
        _mapping = original.RowMapping;
        bool mappingValid = _mapping switch
        {
            CudaNinferOriginalRowMapping.Identity => true,
            CudaNinferOriginalRowMapping.GateUpRows8 => descriptor.Rows % 256 == 0,
            CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue => descriptor.Rows == 14336,
            _ => false
        };
        if (!mappingValid) throw new ArgumentException("Unsupported original projection row mapping.", nameof(descriptor));
        if (!float.IsFinite(descriptor.WeightScaleDivisor) || descriptor.WeightScaleDivisor <= 0)
            throw new ArgumentException("The projection weight divisor must be finite and positive.", nameof(descriptor));
        _expectedDivisorBits = BitConverter.SingleToUInt32Bits(descriptor.WeightScaleDivisor);
        _rows = descriptor.Rows; _columns = descriptor.Columns;
        Original = original;
    }

    public void Read(long offset, IntPtr destination, long bytes)
    {
        RefuseFaulted();
        try
        {
            if (offset < 0 || bytes < 0 || offset > Bytes || bytes > Bytes - offset)
                throw new ArgumentOutOfRangeException(nameof(offset), "Original upload read exceeds its object.");
            if (bytes > 0 && (destination.ToInt64() <= 0
                || unchecked((ulong)destination.ToInt64()) > (ulong)long.MaxValue - (ulong)bytes))
                throw new ArgumentException("Original upload requires a nonzero, nonoverflowing host destination.", nameof(destination));
            // The complete object is capped at128MiB, so every forwarded read is bounded.
            Original.ReadOriginal(offset, destination, bytes);
            if (bytes == 0) return; // Liveness was checked, but no coverage was earned.
            var data = new ReadOnlySpan<byte>((void*)destination, checked((int)bytes));
            long end = offset + bytes;
            long scalesBegin = Math.Max(offset, _codeBytes), scalesEnd = Math.Min(end, _scaleEnd);
            if (scalesBegin < scalesEnd)
            {
                var scales = data.Slice(checked((int)(scalesBegin - offset)), checked((int)(scalesEnd - scalesBegin)));
                int invalid = scales.IndexOfAnyInRange((byte)0x7f, byte.MaxValue);
                if (invalid >= 0)
                    throw new InvalidDataException($"Original NVFP4 E4M3 scale0x{scales[invalid]:x2} at byte{scalesBegin + invalid} is incompatible; no masking or clipping is permitted.");
            }
            for (long at = Math.Max(offset, _scaleEnd); at < end; at++)
            {
                int index = checked((int)(at - _scaleEnd)), shift = index * 8;
                uint value = data[checked((int)(at - offset))];
                byte present = (byte)(1 << index);
                if ((_divisorByteMask & present) != 0 && ((_divisorBits >> shift) & 255u) != value)
                    throw new InvalidDataException("Original weight-divisor bytes changed between reads.");
                _divisorBits = (_divisorBits & ~(255u << shift)) | (value << shift);
                _divisorByteMask |= present;
            }
            if (_divisorByteMask == 15) ValidateDivisor();
            AddCoverage(offset, end);
            _validated = false;
        }
        catch { _faulted = true; _validated = false; throw; }
    }

    internal void RequireCompleteValidation()
    {
        RefuseFaulted();
        try
        {
            if (Original.OriginalBytes != Bytes || Original.Rows != _rows || Original.Columns != _columns || Original.RowMapping != _mapping)
                throw new InvalidDataException("Original source metadata changed during upload.");
            Original.ReadOriginal(Bytes, IntPtr.Zero, 0);
            if (_coverage.Count != 1 || _coverage[0].Begin != 0 || _coverage[0].End != Bytes || _divisorByteMask != 15)
                throw new InvalidOperationException("Original projection validation is incomplete; no matrix may be published.");
            ValidateDivisor();
            _validated = true;
        }
        catch { _faulted = true; _validated = false; throw; }
    }

    private void ValidateDivisor()
    {
        float divisor = BitConverter.UInt32BitsToSingle(_divisorBits);
        if (!float.IsFinite(divisor) || divisor <= 0 || _divisorBits != _expectedDivisorBits)
            throw new InvalidDataException("Original weight-divisor bits differ from the finite positive projection calibration.");
    }

    private void AddCoverage(long begin, long end)
    {
        int first = 0;
        while (first < _coverage.Count && _coverage[first].End < begin) first++;
        int last = first;
        while (last < _coverage.Count && _coverage[last].Begin <= end)
        {
            begin = Math.Min(begin, _coverage[last].Begin);
            end = Math.Max(end, _coverage[last].End);
            last++;
        }
        int remove = last - first;
        if (_coverage.Count - remove + 1 > MaximumCoverageIntervals)
            throw new NotSupportedException("Original upload exceeded its bounded4096 fragmented-read intervals.");
        if (remove != 0) _coverage.RemoveRange(first, remove);
        _coverage.Insert(first, new(begin, end));
    }

    private void RefuseFaulted()
    {
        if (_faulted) throw new InvalidOperationException("The original projection upload validator is permanently faulted.");
    }
}
