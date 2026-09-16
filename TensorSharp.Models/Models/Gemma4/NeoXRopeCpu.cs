// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace TensorSharp.Models;

/// <summary>In-place NeoX rotations for contiguous [sequence, heads, dimension] CPU data.</summary>
internal static unsafe class NeoXRopeCpu
{
    internal static void Apply(float* data, int sequenceLength, int heads, int headDimension,
        float[] cosines, float[] sines, int rotaryHalf)
    {
        if (sequenceLength < 0 || heads <= 0 || rotaryHalf < 0 || headDimension < 2L * rotaryHalf)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Invalid NeoX rotation shape.");
        if (cosines == null || sines == null || cosines.LongLength < (long)sequenceLength * rotaryHalf
            || sines.LongLength < (long)sequenceLength * rotaryHalf)
            throw new ArgumentException("NeoX sine and cosine tables must cover every position.");
        if (sequenceLength == 0 || rotaryHalf == 0) return;
        if (data == null) throw new ArgumentNullException(nameof(data));

        fixed (float* cos = cosines, sin = sines)
        {
            // Each work item owns contiguous rows, keeping scheduling overhead
            // below the rotation work and avoiding per-position delegate calls.
            const int rowsPerWorkItem = 32;
            if (sequenceLength >= 64 && (long)sequenceLength * heads * rotaryHalf >= 65536)
            {
                nint dataAddress = (nint)data, cosAddress = (nint)cos, sinAddress = (nint)sin;
                int workItems = (sequenceLength - 1) / rowsPerWorkItem + 1;
                Parallel.For(0, workItems, block =>
                {
                    int first = block * rowsPerWorkItem;
                    Rows((float*)dataAddress, first, first + Math.Min(rowsPerWorkItem, sequenceLength - first),
                        heads, headDimension, (float*)cosAddress, (float*)sinAddress, rotaryHalf);
                });
            }
            else
                Rows(data, 0, sequenceLength, heads, headDimension, cos, sin, rotaryHalf);
        }
    }

    private static void Rows(float* data, int first, int end, int heads, int headDimension,
        float* cosines, float* sines, int rotaryHalf)
    {
        int width = Vector<float>.Count;
        for (int position = first; position < end; position++)
        {
            float* cos = cosines + (long)position * rotaryHalf;
            float* sin = sines + (long)position * rotaryHalf;
            for (int h = 0; h < heads; h++)
            {
                float* head = data + ((long)position * heads + h) * headDimension;
                int j = 0;
                if (Vector.IsHardwareAccelerated)
                    for (; j <= rotaryHalf - width; j += width)
                    {
                        var c = Unsafe.ReadUnaligned<Vector<float>>(cos + j);
                        var s = Unsafe.ReadUnaligned<Vector<float>>(sin + j);
                        var x = Unsafe.ReadUnaligned<Vector<float>>(head + j);
                        var y = Unsafe.ReadUnaligned<Vector<float>>(head + j + rotaryHalf);
                        Unsafe.WriteUnaligned(head + j, x * c - y * s);
                        Unsafe.WriteUnaligned(head + j + rotaryHalf, x * s + y * c);
                    }
                for (; j < rotaryHalf; j++)
                {
                    float x = head[j], y = head[j + rotaryHalf];
                    head[j] = x * cos[j] - y * sin[j];
                    head[j + rotaryHalf] = x * sin[j] + y * cos[j];
                }
            }
        }
    }
}
