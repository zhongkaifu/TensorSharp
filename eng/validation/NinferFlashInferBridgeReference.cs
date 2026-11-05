// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda;

namespace TensorSharp.Validation;

/// <summary>Independent placement oracle. No native bridge index formula or
/// integer rounding-bias implementation is reused.</summary>
internal static class NinferFlashInferBridgeReference
{
    internal static ushort NearestBf16(uint bits)
    {
        if ((bits & 0x7f800000U) == 0x7f800000U) throw new ArgumentException("Finite F32 input required.");
        ushort sign = (ushort)((bits >> 16) & 0x8000);
        uint magnitude = bits & 0x7fffffff;
        int lower = (int)(magnitude >> 16);
        double x = BitConverter.UInt32BitsToSingle(magnitude);
        double a = BitConverter.UInt32BitsToSingle((uint)lower << 16);
        // The finite overflow midpoint is measured against mathematical2^128.
        // Rounding to that endpoint encodes BF16 infinity, including an even tie.
        double b = lower == 0x7f7f ? Math.ScaleB(1.0, 128)
            : BitConverter.UInt32BitsToSingle((uint)(lower + 1) << 16);
        double down = x - a, up = b - x;
        int best = down < up || down == up && (lower & 1) == 0 ? lower : lower + 1;
        return (ushort)(sign | best);
    }

    internal static uint[] FiniteBf16Inputs()
    {
        var values = new uint[65280]; int at = 0;
        for (uint sign = 0; sign <= 1; sign++) for (uint word = 0; word < 0x7f80; word++)
            values[at++] = (sign << 31) | (word << 16);
        return values;
    }
    internal static uint[] Midpoints(bool negative)
    {
        var values = new uint[32640 * 3]; int at = 0;
        for (uint word = 0; word < 0x7f80; word++)
            foreach (uint tail in new uint[] { 0x7fff, 0x8000, 0x8001 })
                values[at++] = (negative ? 0x80000000U : 0) | (word << 16) | tail;
        return values;
    }
    internal static ushort[] Cast(uint[] input)
    {
        var output = new ushort[input.Length];
        for (int i = 0; i < input.Length; i++) output[i] = NearestBf16(input[i]);
        return output;
    }
    internal static uint[] Scatter(uint[] input, int n, int rows, CudaNinferOriginalRowMapping mapping)
    {
        if (input.Length != checked(n * rows)) throw new ArgumentException("Complete compact rows required.");
        var output = new uint[input.Length];
        for (int token = 0; token < rows; token++)
        {
            var src = input.AsSpan(token * n, n); var dst = output.AsSpan(token * n, n);
            switch (mapping)
            {
                case CudaNinferOriginalRowMapping.Identity: src.CopyTo(dst); break;
                case CudaNinferOriginalRowMapping.GateUpRows8:
                    if (n != 34816) throw new ArgumentException("Original gate/up shape required.");
                    int outAt = 0;
                    for (int first = 0; first < n / 2; first += 8)
                    {
                        src.Slice(first, 8).CopyTo(dst[outAt..]); outAt += 8;
                        src.Slice(n / 2 + first, 8).CopyTo(dst[outAt..]); outAt += 8;
                    }
                    break;
                case CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue:
                    if (n != 14336) throw new ArgumentException("Original GQA shape required.");
                    for (int head = 0; head < 24; head++)
                    {
                        src.Slice(head * 256, 256).CopyTo(dst[(head * 512)..]);
                        src.Slice(7168 + head * 256, 256).CopyTo(dst[(head * 512 + 256)..]);
                    }
                    src.Slice(6144, 1024).CopyTo(dst[12288..]);
                    src.Slice(13312, 1024).CopyTo(dst[13312..]);
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(mapping));
            }
        }
        return output;
    }
}
