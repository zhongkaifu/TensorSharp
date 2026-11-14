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
/// The graph executor's broadcasting binary kernel (ggml's repeat rule: each of b's extents 1 or a's) against the
/// elementwise definition, for the shapes the runtime emits: a per-row scalar (a shared expert's gate), a vector over
/// every row, a per-stream weight, and b with every extent of a.
/// </summary>
public sealed class CudaBinaryBroadcastTests
{
    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    [CudaTheory]
    [InlineData(2560, 7, 1, 1, 1, 7, 1, 1, 2)]       // [n, rows] * [1, rows]
    [InlineData(2560, 1, 1, 1, 1, 1, 1, 1, 2)]       // one decode row times its scalar
    [InlineData(64, 5, 3, 1, 64, 1, 1, 1, 0)]        // a vector added to every row of every plane
    [InlineData(32, 4, 6, 2, 1, 4, 1, 2, 3)]         // per (row, sequence) divisors
    [InlineData(17, 3, 2, 1, 17, 3, 2, 1, 1)]        // no repeat
    public void MatchesTheDefinition(long a0, long a1, long a2, long a3, long b0, long b1, long b2, long b3, int op)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            long[] na = { a0, a1, a2, a3 }, nb = { b0, b1, b2, b3 };
            long n = a0 * a1 * a2 * a3, m = b0 * b1 * b2 * b3;
            var rng = new Random((int)(n + 7 * m + op));
            float[] a = Enumerable.Range(0, (int)n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            float[] b = Enumerable.Range(0, (int)m).Select(_ => (float)(0.5 + rng.NextDouble())).ToArray();
            using var at = Tensor.FromArray(allocator, a);
            using var bt = Tensor.FromArray(allocator, b);
            using var output = new Tensor(allocator, DType.Float32, n);
            foreach (Tensor t in new[] { at, bt, output }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchBinaryBroadcast(P(at), P(bt), P(output), na, nb, op, allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            ((CudaStorage)output.Storage).MarkDeviceModified();
            float[] got = output.GetElementsAsFloat((int)n);
            for (long i = 0; i < n; i++)
            {
                long r = i, i0 = r % a0; r /= a0;
                long i1 = r % a1; r /= a1;
                long i2 = r % a2, i3 = r / a2;
                long j = (((b3 == 1 ? 0 : i3) * b2 + (b2 == 1 ? 0 : i2)) * b1 + (b1 == 1 ? 0 : i1)) * b0 + (b0 == 1 ? 0 : i0);
                float x = a[i], y = b[j];
                float want = op switch { 0 => x + y, 1 => x - y, 2 => x * y, _ => x / y };
                Assert.Equal(want, got[i]);
            }
        }
    }
}
