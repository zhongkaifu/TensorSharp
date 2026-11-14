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
    /// <summary>The injection logits folded into the norm and the mix: a double reference's values, and each row of a
    /// window the same bits as that row read alone (a verify window's rows are its decode steps').</summary>
    [CudaTheory]
    [InlineData(4, 4, 2560)]
    [InlineData(3, 2, 512)]
    [InlineData(20, 8, 256)]   // a prompt chunk's narrow blocks
    public void FoldedInjectionMatchesTheReferenceAndEachRowAlone(int rows, int hc, int n)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(31);
            float[] R(int count, double scale) => Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
            int width = hc * n;
            float[] xs = R(rows * width, 3), gs = R(width, 1.5), ws = R(hc * width, 0.05);
            using var x = Tensor.FromArray(allocator, xs);
            using var gamma = Tensor.FromArray(allocator, gs);
            using var injW = Tensor.FromArray(allocator, ws);
            using var gates = Tensor.FromArray(allocator, R(rows * width, 5));
            using var xn = new Tensor(allocator, DType.Float32, rows * width);
            using var part = new Tensor(allocator, DType.Float32, rows * hc * hc);
            using var mix = new Tensor(allocator, DType.Float32, rows * n);
            using var inject = new Tensor(allocator, DType.Float32, rows * hc);
            Upload(x, gamma, injW, gates, xn, part, mix, inject);
            IntPtr stream = allocator.Stream.Handle;
            kernels.LaunchHcNorm(P(x), P(gamma), P(xn), rows, n, hc, 1e-6f, stream, IntPtr.Zero, P(injW), P(part));
            kernels.LaunchHcGateMix(P(xn), P(gates), P(mix), rows, n, hc, stream, IntPtr.Zero, P(part), P(inject));
            allocator.Stream.Synchronize();
            Written(xn, part, mix, inject);
            float[] got = inject.GetElementsAsFloat(rows * hc);
            for (int r = 0; r < rows; r++)
            {
                for (int o = 0; o < hc; o++)
                {
                    double want = 0, mag = 0;
                    for (int c = 0; c < hc; c++)
                    {
                        double ss = 0;
                        for (int i = 0; i < n; i++) ss += (double)xs[(r * hc + c) * n + i] * xs[(r * hc + c) * n + i];
                        double scale = 1 / Math.Sqrt(ss / n + 1e-6);
                        for (int i = 0; i < n; i++)
                        {
                            double v = xs[(r * hc + c) * n + i] * scale * gs[c * n + i] * ws[(long)o * width + c * n + i];
                            want += v;
                            mag += Math.Abs(v);
                        }
                    }
                    Assert.True(Math.Abs(got[r * hc + o] - want) <= 1e-5 * mag + 1e-6, $"row {r} output {o}: {got[r * hc + o]} against {want}");
                }
                // The row alone.
                using var x1 = Tensor.FromArray(allocator, xs.Skip(r * width).Take(width).ToArray());
                using var g1 = Tensor.FromArray(allocator, gates.GetElementsAsFloat(rows * width).Skip(r * width).Take(width).ToArray());
                using var xn1 = new Tensor(allocator, DType.Float32, width);
                using var part1 = new Tensor(allocator, DType.Float32, hc * hc);
                using var mix1 = new Tensor(allocator, DType.Float32, n);
                using var inject1 = new Tensor(allocator, DType.Float32, hc);
                Upload(x1, g1, xn1, part1, mix1, inject1);
                kernels.LaunchHcNorm(P(x1), P(gamma), P(xn1), 1, n, hc, 1e-6f, stream, IntPtr.Zero, P(injW), P(part1));
                kernels.LaunchHcGateMix(P(xn1), P(g1), P(mix1), 1, n, hc, stream, IntPtr.Zero, P(part1), P(inject1));
                allocator.Stream.Synchronize();
                Written(xn1, part1, mix1, inject1);
                float[] alone = inject1.GetElementsAsFloat(hc);
                if (rows * hc < 64)   // the norm's block width is the same for the row alone
                    Assert.Equal(alone.Select(BitConverter.SingleToInt32Bits), got.Skip(r * hc).Take(hc).Select(BitConverter.SingleToInt32Bits));
            }
        }
    }
}
