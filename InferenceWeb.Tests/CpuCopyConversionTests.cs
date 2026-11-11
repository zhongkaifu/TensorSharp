// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp;
using TensorSharp.Cpu;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>The CPU copy op between strided views of different element types: F32 rows written into
/// an F16 cache view and read back. Its strided paths used to walk every buffer as F32, which wrote
/// past the end of an F16 buffer (heap corruption, found by the graph executor's cache writes).</summary>
public class CpuCopyConversionTests
{
    private static readonly IAllocator Cpu = new CpuAllocator(BlasEnum.DotNet);

    [Fact]
    public void StridedCopiesConvertBetweenF32AndF16AndStayInsideTheirBuffers()
    {
        const int heads = 3, cap = 8, d = 5, rows = 4, start = 2;
        // A cache [heads, cap, d] with a guard element after it in the same storage.
        using var storage = new Tensor(Cpu, DType.Float16, heads * cap * d + 1);
        Ops.Fill(storage, 7.5f);
        using var guard = storage.Narrow(0, heads * cap * d, 1);
        using var cache = storage.Narrow(0, 0, heads * cap * d).View(heads, cap, d);
        using (var zeros = new Tensor(Cpu, DType.Float32, heads, cap, d))
        {
            Ops.Fill(zeros, 0);
            Ops.Copy(cache, zeros);
        }

        // Fresh rows [rows, heads, d] (F32), written as [heads, rows, d] at rows start.. of the cache.
        var values = new float[rows * heads * d];
        for (int i = 0; i < values.Length; i++) values[i] = i * 0.25f - 3f;
        using var fresh = new Tensor(Cpu, DType.Float32, rows, heads, d);
        fresh.SetElementsAsFloat(values);
        using var permuted = fresh.Permute(1, 0, 2);
        using var target = cache.Narrow(1, start, rows);
        Ops.Copy(target, permuted);

        // Read the window back as F32 through another strided copy.
        using var back = new Tensor(Cpu, DType.Float32, heads, rows, d);
        Ops.Copy(back, target);
        float[] read = back.GetElementsAsFloat(heads * rows * d);
        for (int h = 0; h < heads; h++)
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < d; k++)
                    Assert.Equal(values[(r * heads + h) * d + k], read[(h * rows + r) * d + k]);   // exact in F16
        Assert.Equal(7.5f, Wide(guard)[0]);
        float[] all = Wide(cache);
        for (int h = 0; h < heads; h++)
            for (int r = 0; r < cap; r++)
                if (r < start || r >= start + rows)
                    for (int k = 0; k < d; k++) Assert.Equal(0f, all[(h * cap + r) * d + k]);
    }

    private static float[] Wide(Tensor half)
    {
        using var wide = new Tensor(Cpu, DType.Float32, half.Sizes);
        Ops.Copy(wide, half);
        return wide.GetElementsAsFloat((int)wide.ElementCount());
    }

    [Fact]
    public void AStridedCopyOfAnUnsupportedTypeIsRefusedNotReinterpreted()
    {
        using var bytes = new Tensor(Cpu, DType.UInt8, 4, 6);
        using var src = new Tensor(Cpu, DType.UInt8, 6, 4);
        using var transposed = src.Transpose();
        Assert.Throws<NotSupportedException>(() => Ops.Copy(bytes, transposed));
    }
}
