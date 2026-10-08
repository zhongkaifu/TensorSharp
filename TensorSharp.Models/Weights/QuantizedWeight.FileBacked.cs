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

    internal static QuantizedWeight CreateFileBacked(IResourceSource source, int ggmlType, long ne0, long ne1)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (ggmlType != 8 || ne0 <= 0 || ne0 > int.MaxValue || ne0 % 32 != 0 || ne1 <= 0 || ne1 > int.MaxValue)
            throw new NotSupportedException("Streamed weights require a nonempty Q8_0 matrix with block-aligned input width.");
        if (source.ByteLength != checked(checked(ne0 / 32 * 34) * ne1))
            throw new ArgumentException("File region length does not match the Q8_0 matrix.", nameof(source));
        return new QuantizedWeight(IntPtr.Zero, source.ByteLength, ggmlType, ne0, ne1, false, source)
        { FileSource = source };
    }
}
