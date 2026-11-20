// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// GraphOp.GeluErf on the CUDA graph executor runs as one kernel (ts_graph_gelu_erf_f32): 0.5 x (1 + erf(x / sqrt 2)) within
/// float rounding of a double-precision reference, over a contiguous row block and a strided view of one, with no host
/// fallback (the generic decomposition's abs and exp have no CUDA kernels and ran element by element on the host).
/// </summary>
public sealed class CudaGeluErfTests
{
    /// <summary>erf in double: its Maclaurin series up to |x| = 4 (terms past 1e-17 of the sum), 1 - erfc beyond, where erfc is
    /// below 1.6e-8 and its continued fraction converges in a few terms.</summary>
    private static double Erf(double x)
    {
        double a = Math.Abs(x);
        if (a > 4)
        {
            // erfc(a) = exp(-a^2) / sqrt(pi) * 1 / (a + 1/2 / (a + 1 / (a + 3/2 / (a + ...)))).
            double f = a;
            for (int k = 60; k >= 1; k--) f = a + k / 2.0 / f;
            return Math.Sign(x) * (1 - Math.Exp(-a * a) / Math.Sqrt(Math.PI) / f);
        }
        double term = x, sum = x;
        for (int n = 1; n < 200; n++)
        {
            term *= -x * x / n;
            double add = term / (2 * n + 1);
            sum += add;
            if (Math.Abs(add) < 1e-17 * Math.Abs(sum)) break;
        }
        return 2 / Math.Sqrt(Math.PI) * sum;
    }

    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void GeluErfIsOneKernelWithinFloatRounding(bool strided)
    {
        const int rows = 37, cols = 96;
        const long f = sizeof(float);
        int width = strided ? cols + 32 : cols;
        var x = new float[rows * width];
        var rng = new Random(7);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 16 - 8);
        float[] edges = { 0f, -0f, 1e-30f, -1e-30f, 1f, -1f, 3.5f, -3.5f, 6f, -6f, 20f, -20f, 45f, -45f, 100f, -100f };
        Array.Copy(edges, x, edges.Length);

        var g = new GraphProgramBuilder(record: true);
        int input = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { width, rows }, IntPtr.Zero, x.Length * f);
        int src = strided
            ? g.Op(GraphOp.View, stackalloc[] { input }, stackalloc long[] { 2, cols, rows, 1, 1, width * f, 0, 0, 0 })
            : input;
        int y = g.Op(GraphOp.GeluErf, src);
        g.Output(y, rows * cols);
        using var allocator = new CudaAllocator();
        using var executor = new CudaGraphExecutor(allocator);
        using IGraphPlan plan = executor.Compile(g.ToProgram(replayable: true, reorder: false));
        long fallbacks = CudaCpuFallback.ThreadCalls;
        GCHandle h = GCHandle.Alloc(x, GCHandleType.Pinned);
        try
        {
            Assert.True(plan.Run(new[] { input }, new[] { h.AddrOfPinnedObject() }, new[] { x.Length * f }), plan.LastError);
        }
        finally
        {
            h.Free();
        }
        Assert.Equal(fallbacks, CudaCpuFallback.ThreadCalls);
        float[] got = plan.Outputs[0];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                double v = x[r * width + c], want = 0.5 * v * (1 + Erf(v / Math.Sqrt(2)));
                double err = Math.Abs(got[r * cols + c] - want) / Math.Max(1, Math.Abs(want));
                Assert.True(err <= 4e-7, $"gelu_erf({v:R}) = {got[r * cols + c]:R}, reference {want:R}");
            }
    }
}
