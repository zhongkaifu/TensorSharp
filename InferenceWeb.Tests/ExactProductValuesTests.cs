// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;
using TensorSharp.Graph;
using TensorSharp.Models;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The managed executor's exact products (<see cref="TensorGraphExecutor.FloatProducts"/>) dequantize their weight's rows in
/// chunks on the worker pool (<see cref="ManagedGraphExecutor.DequantizeValues"/>): the values are the bits of one
/// dequantization call over the whole weight, which the F32 reference tools' fixtures were written from.
/// </summary>
public sealed class ExactProductValuesTests
{
    private sealed record Weight(int Format, IntPtr Data, long Ne0, long Ne1, long Bytes) : IGraphWeight
    {
        public long Ne2 => 1;
        public IntPtr CacheKey => Data;
    }

    /// <summary>Blocks of random bytes with finite F16 scales at the given offsets of each block.</summary>
    [Theory]
    [InlineData(GgmlTensorType.Q8_0, new[] { 0 })]
    [InlineData(GgmlTensorType.Q4_K, new[] { 0, 2 })]
    [InlineData(GgmlTensorType.IQ2_XXS, new[] { 0 })]
    public void ChunkedValuesAreTheOneCallDequantizationsBits(GgmlTensorType type, int[] scaleOffsets)
    {
        const long ne0 = 512, rows = 1000;
        long blockElements = GgufFile.GetBlockSize(type), blockBytes = GgufFile.GetTypeSize(type);
        long bytes = rows * (ne0 / blockElements) * blockBytes;
        var rng = new Random((int)type);
        var raw = new byte[bytes];
        rng.NextBytes(raw);
        for (long b = 0; b < bytes / blockBytes; b++)
            foreach (int at in scaleOffsets)
                BitConverter.TryWriteBytes(raw.AsSpan((int)(b * blockBytes + at), 2), (Half)(0.001 + 0.05 * rng.NextDouble()));

        IntPtr data = Marshal.AllocHGlobal((nint)bytes);
        var once = new float[rows * ne0];
        var chunked = new float[rows * ne0];
        GCHandle ho = GCHandle.Alloc(once, GCHandleType.Pinned), hc = GCHandle.Alloc(chunked, GCHandleType.Pinned);
        try
        {
            Marshal.Copy(raw, 0, data, (int)bytes);
            NativeDequant.DequantizeToFloat32Native((int)type, data, ho.AddrOfPinnedObject(), rows * ne0);
            ManagedGraphExecutor.DequantizeValues(new Weight((int)type, data, ne0, rows, bytes), hc.AddrOfPinnedObject());
        }
        finally
        {
            ho.Free();
            hc.Free();
            Marshal.FreeHGlobal(data);
        }
        int first = -1;
        for (int i = 0; i < once.Length && first < 0; i++)
            if (BitConverter.SingleToInt32Bits(once[i]) != BitConverter.SingleToInt32Bits(chunked[i])) first = i;
        if (first >= 0)
            Assert.Fail($"{type}: value {first} (row {first / ne0}) is {chunked[first]:R}, the one-call dequantization's {once[first]:R}");
        Assert.Contains(once, v => v != 0);
    }
}
