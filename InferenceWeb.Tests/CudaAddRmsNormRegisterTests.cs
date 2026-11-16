// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The residual add and the next block's norm (CudaGraphKernels.LaunchAddRmsNorm): rows of up to 8,192 values run
/// ts_graph_add_rms_norm_reg_f32, which holds a row in registers with its loads issued up front. Its thread sums,
/// reductions and expressions are the loop kernel's (ts_graph_add_rms_norm_f32), so the sums and the normed rows must be
/// the same bits, at widths that fill the 16 registers, leave lanes idle and are not multiples of 512, and for one row
/// (decode) as for a window's.
/// </summary>
public sealed class CudaAddRmsNormRegisterTests
{
    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    [CudaTheory]
    [InlineData(1, 5120)]   // Qwen3.8-27B's hidden width, a decode row
    [InlineData(8, 5120)]   // its verify window
    [InlineData(13, 4096)]
    [InlineData(3, 8192)]   // all 16 registers
    [InlineData(5, 1000)]   // not a multiple of 512: some threads hold one value fewer
    [InlineData(2, 300)]    // fewer values than threads
    public void RegisterKernelWritesTheLoopKernelsBits(int rows, int cols)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels created, out string reason), reason);
        using CudaGraphKernels kernels = created;
        var rng = new Random(rows * 131 + cols);
        float[] F(int n, double scale) => Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
        using var a = Tensor.FromArray(allocator, F(rows * cols, 3));
        using var b = Tensor.FromArray(allocator, F(rows * cols, 1));
        using var w = Tensor.FromArray(allocator, F(cols, 2));
        (float[] Sum, float[] Normed) Run(bool loop)
        {
            using var sum = new Tensor(allocator, DType.Float32, rows * cols);
            using var normed = new Tensor(allocator, DType.Float32, rows * cols);
            foreach (Tensor t in new[] { a, b, w, sum, normed }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.AddRmsNormLoopForTests = loop;
            kernels.LaunchAddRmsNorm(P(a), P(b), P(w), P(sum), P(normed), rows, cols, 1e-6f, allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            ((CudaStorage)sum.Storage).MarkDeviceModified();
            ((CudaStorage)normed.Storage).MarkDeviceModified();
            return (sum.GetElementsAsFloat(rows * cols), normed.GetElementsAsFloat(rows * cols));
        }
        var expected = Run(loop: true);
        var actual = Run(loop: false);
        for (int i = 0; i < rows * cols; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(expected.Sum[i]) == BitConverter.SingleToInt32Bits(actual.Sum[i]),
                $"sum {i}: {actual.Sum[i]:R} from the register kernel, {expected.Sum[i]:R} from the loop");
            Assert.True(BitConverter.SingleToInt32Bits(expected.Normed[i]) == BitConverter.SingleToInt32Bits(actual.Normed[i]),
                $"normed {i}: {actual.Normed[i]:R} from the register kernel, {expected.Normed[i]:R} from the loop");
        }
    }
}
