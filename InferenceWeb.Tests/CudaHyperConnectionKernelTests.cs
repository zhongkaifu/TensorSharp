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
/// The hyper-connection read's kernels write the q8_1 copies their outputs' products read (the normed streams, the
/// bottleneck, the mix): byte for byte what the row quantizer writes from the same values, and the values as without.
/// </summary>
public sealed class CudaHyperConnectionKernelTests
{
    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static void Upload(params Tensor[] tensors)
    {
        foreach (Tensor t in tensors) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
    }

    private static void Written(params Tensor[] tensors)
    {
        foreach (Tensor t in tensors) ((CudaStorage)t.Storage).MarkDeviceModified();
    }

    [CudaTheory]
    [InlineData(4, 4, 2560, 64)]
    [InlineData(1, 4, 256, 32)]
    [InlineData(32, 2, 512, 128)]
    public void FusedQuantizedCopiesMatchTheRowQuantizer(int rows, int hc, int n, int lr)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(29);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            int width = hc * n;
            using var x = Tensor.FromArray(allocator, R(rows * width, 3));
            using var gamma = Tensor.FromArray(allocator, R(width, 1.5));
            using var lo = Tensor.FromArray(allocator, R(rows * lr, 4));
            using var gates = Tensor.FromArray(allocator, R(rows * width, 5));
            using var xn = new Tensor(allocator, DType.Float32, rows * width);
            using var xn2 = new Tensor(allocator, DType.Float32, rows * width);
            using var silu = new Tensor(allocator, DType.Float32, rows * lr);
            using var silu2 = new Tensor(allocator, DType.Float32, rows * lr);
            using var mix = new Tensor(allocator, DType.Float32, rows * n);
            using var mix2 = new Tensor(allocator, DType.Float32, rows * n);
            Tensor Q(long values) => new Tensor(allocator, DType.UInt8, values / 32 * CudaKernels.Q81BlockBytes);
            using Tensor xnq = Q(rows * width), xnRef = Q(rows * width), siluq = Q(rows * lr), siluRef = Q(rows * lr), mixq = Q(rows * n), mixRef = Q(rows * n);
            Upload(x, gamma, lo, gates, xn, xn2, silu, silu2, mix, mix2, xnq, xnRef, siluq, siluRef, mixq, mixRef);
            IntPtr stream = allocator.Stream.Handle;
            allocator.Context.MakeCurrent();
            kernels.LaunchHcNorm(P(x), P(gamma), P(xn), rows, n, hc, 1e-6f, stream, P(xnq));
            kernels.LaunchHcNorm(P(x), P(gamma), P(xn2), rows, n, hc, 1e-6f, stream);
            kernels.LaunchScaleSilu(P(lo), P(silu), rows * lr, 1f / hc, stream, P(siluq));
            kernels.LaunchScaleSilu(P(lo), P(silu2), rows * lr, 1f / hc, stream);
            kernels.LaunchHcGateMix(P(xn), P(gates), P(mix), rows, n, hc, stream, P(mixq));
            kernels.LaunchHcGateMix(P(xn), P(gates), P(mix2), rows, n, hc, stream);
            CudaQuantizedOps.QuantizeRowsQ81(allocator, P(xn2), P(xnRef), width, rows);
            CudaQuantizedOps.QuantizeRowsQ81(allocator, P(silu2), P(siluRef), lr, rows);
            CudaQuantizedOps.QuantizeRowsQ81(allocator, P(mix2), P(mixRef), n, rows);
            allocator.Stream.Synchronize();
            Written(xn, xn2, silu, silu2, mix, mix2, xnq, xnRef, siluq, siluRef, mixq, mixRef);
            void Same(Tensor a, Tensor b, string what)
            {
                byte[] x1 = new byte[a.ElementCount()], x2 = new byte[b.ElementCount()];
                a.CopyToArray(x1);
                b.CopyToArray(x2);
                Assert.True(x2.Count(v => v != 0) > x2.Length / 4, $"{what}: the reference read back as zeros");
                Assert.True(x1.AsSpan().SequenceEqual(x2), $"{what} differs ({rows} rows, {hc}x{n}, rank {lr})");
            }
            void SameF(Tensor a, Tensor b, string what)
            {
                float[] f1 = a.GetElementsAsFloat((int)a.ElementCount()), f2 = b.GetElementsAsFloat((int)b.ElementCount());
                Assert.True(f1.Select(BitConverter.SingleToInt32Bits).SequenceEqual(f2.Select(BitConverter.SingleToInt32Bits)), $"{what} values differ");
            }
            SameF(xn, xn2, "normed streams");
            SameF(silu, silu2, "bottleneck");
            SameF(mix, mix2, "mix");
            Same(xnq, xnRef, "normed streams' q8_1");
            Same(siluq, siluRef, "bottleneck's q8_1");
            Same(mixq, mixRef, "mix's q8_1");
        }
    }
}
