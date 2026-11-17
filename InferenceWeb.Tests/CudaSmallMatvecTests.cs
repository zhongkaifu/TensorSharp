// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Diagnostics;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The graph executor's small-matvec kernel (unquantized products with few outputs and rows: a hyper-connection's
/// injection, a shared expert's gate, a delta net's alpha and beta) against a double reference, timed beside cuBLAS on the
/// same operands.
/// </summary>
public sealed class CudaSmallMatvecTests
{
    private readonly ITestOutputHelper _output;

    public CudaSmallMatvecTests(ITestOutputHelper output) => _output = output;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    [CudaTheory]
    [InlineData(1, 10240, 4)]     // Qwen3.8 Flash Next's injection, a decode row
    [InlineData(5, 10240, 4)]     // verify rows
    [InlineData(16, 10240, 4)]    // two row groups
    [InlineData(1, 2560, 1)]      // a shared expert's gate
    [InlineData(1, 2560, 48)]     // a delta net's alpha / beta
    [InlineData(1, 2560, 64)]
    [InlineData(1, 2560, 512)]    // a router (timed only: past the executor's limit)
    [InlineData(3, 1001, 7)]      // k not a multiple of 4: the scalar loop
    public void MatchesTheReference(int rows, int k, int outs)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(rows * 7919 + k + outs);
            float[] x = Enumerable.Range(0, rows * k).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            float[] w = Enumerable.Range(0, outs * k).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            using var xt = Tensor.FromArray(allocator, x);
            using var wt = Tensor.FromArray(allocator, w);
            using Tensor x2 = xt.View(rows, k), w2 = wt.View(outs, k);
            using var y = new Tensor(allocator, DType.Float32, rows, outs);
            using var yb = new Tensor(allocator, DType.Float32, rows, outs);
            foreach (Tensor t in new[] { xt, wt, y }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            void Kernel()
            {
                allocator.Context.MakeCurrent();
                kernels.LaunchSmallMatvec(P(x2), P(w2), P(y), rows, k, outs, k, allocator.Stream.Handle);
            }
            using Tensor w2t = w2.Transpose();
            void Blas() => Ops.Addmm(yb, 0, yb, 1f, x2, w2t);

            Kernel();
            allocator.Stream.Synchronize();
            ((CudaStorage)y.Storage).MarkDeviceModified();
            float[] got = y.GetElementsAsFloat(rows * outs);
            double worst = 0;
            for (int r = 0; r < rows; r++)
                for (int o = 0; o < outs; o++)
                {
                    double want = 0, mag = 0;
                    for (int i = 0; i < k; i++)
                    {
                        want += x[r * k + i] * (double)w[o * k + i];
                        mag += Math.Abs(x[r * k + i] * (double)w[o * k + i]);
                    }
                    worst = Math.Max(worst, Math.Abs(got[r * outs + o] - want) / Math.Max(mag, 1e-30));
                }

            double Time(Action launch)
            {
                for (int i = 0; i < 20; i++) launch();
                allocator.Synchronize();
                var watch = Stopwatch.StartNew();
                const int n = 200;
                for (int i = 0; i < n; i++) launch();
                allocator.Synchronize();
                return watch.Elapsed.TotalMilliseconds * 1000 / n;
            }
            double kernelUs = Time(Kernel), blasUs = Time(Blas);
            _output.WriteLine($"{rows} x {k} -> {outs}: worst error {worst:E2} of the magnitude; kernel {kernelUs:F1} us, cuBLAS {blasUs:F1} us a product");
            Assert.True(worst < 1e-5, $"worst error {worst:E2}");
        }
    }
}
