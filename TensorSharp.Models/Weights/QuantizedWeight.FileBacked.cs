// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.Memory;

namespace TensorSharp.Models;

public partial class QuantizedWeight
{
    /// <summary>True for an immutable file region consumed only by the explicit
    /// streaming operator. Data and CacheKey are zero: it must never reach a raw
    /// pointer binder, weight fusion, preload or captured graph.</summary>
    public bool IsStreamed => FileSource != null;
    internal IResourceSource FileSource { get; private set; }

    internal long StreamingRowBytes => GetStreamingRowBytes(GgmlType, Ne0);

    internal static long GetStreamingRowBytes(int ggmlType, long width)
    {
        if (width <= 0 || width > int.MaxValue)
            throw new NotSupportedException("Streamed weight width must be a positive 32-bit dimension.");
        return ggmlType switch
        {
            8 when width % 32 == 0 => checked(width / 32 * 34),
            1 => checked(width * sizeof(ushort)),
            _ => throw new NotSupportedException("Streamed weights require F16 or block-aligned Q8_0 rows."),
        };
    }

    internal static QuantizedWeight CreateFileBacked(IResourceSource source, int ggmlType, long ne0, long ne1)
    {
        ArgumentNullException.ThrowIfNull(source);
        long rowBytes = GetStreamingRowBytes(ggmlType, ne0);
        if (ne1 <= 0 || ne1 > int.MaxValue)
            throw new NotSupportedException("Streamed weights require a nonempty two-dimensional matrix.");
        if (source.ByteLength != checked(rowBytes * ne1))
            throw new ArgumentException("File region length does not match the streamed matrix.", nameof(source));
        return new QuantizedWeight(IntPtr.Zero, source.ByteLength, ggmlType, ne0, ne1, false, source)
        { FileSource = source };
    }

    /// <summary>Borrow immutable row sources without copying weights, creating
    /// raw pointer identities, or adding another physical file-byte charge.</summary>
    internal static QuantizedWeight CreateFileBackedConcatenation(params QuantizedWeight[] weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.Length < 2 || weights[0] == null)
            throw new ArgumentException("Concatenation requires at least two streamed weights.", nameof(weights));
        QuantizedWeight first = weights[0];
        var sources = new IResourceSource[weights.Length];
        long rows = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            QuantizedWeight weight = weights[i];
            if (weight == null || !weight.IsStreamed || weight.Ne0 != first.Ne0 ||
                weight.GgmlType != first.GgmlType || weight.Scale != first.Scale || !float.IsFinite(weight.Scale))
                throw new ArgumentException("Concatenated streamed weights must share type, width and finite output scale.", nameof(weights));
            rows = checked(rows + weight.Ne1);
            sources[i] = weight.FileSource;
        }
        var result = CreateFileBacked(new ConcatenatedWeightSource(sources), first.GgmlType, first.Ne0, rows);
        result.Scale = first.Scale;
        return result;
    }
}
