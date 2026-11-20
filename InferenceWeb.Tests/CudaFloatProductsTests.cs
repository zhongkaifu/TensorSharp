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
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A program whose products keep their activations in F32 (<see cref="GraphProgram.FloatProducts"/>) on the CUDA graph
/// executor: a q8_0 weight's product is its values (code * d) times the F32 rows, within F32 rounding of a double-precision
/// reference at decode, int8-GEMM and F16-GEMM row counts, where the executor's own products (the same program without the
/// flag) round the activations and sit orders of magnitude further off.
/// </summary>
public sealed class CudaFloatProductsTests(ITestOutputHelper output)
{
    private sealed record Weight(IntPtr Data, long Ne0, long Ne1, long Bytes) : IGraphWeight
    {
        public int Format => GraphFormat.Q8_0;
        public long Ne2 => 1;
        public IntPtr CacheKey => Data;
    }

    [CudaTheory]
    [InlineData(20)]
    [InlineData(100)]
    [InlineData(700)]
    public void AFloatProductProgramMultipliesTheWeightsValuesInF32(int rows)
    {
        const int k = 256, outDim = 96;
        const long f = sizeof(float);
        var rng = new Random(rows);
        long blocks = (long)outDim * (k / 32), bytes = blocks * 34;
        var raw = new byte[bytes];
        var values = new float[outDim * k];
        for (long b = 0; b < blocks; b++)
        {
            System.Half d = (System.Half)(0.005 + 0.04 * rng.NextDouble());
            BitConverter.TryWriteBytes(raw.AsSpan((int)(b * 34), 2), d);
            for (int j = 0; j < 32; j++)
            {
                sbyte q = (sbyte)rng.Next(-127, 128);
                raw[b * 34 + 2 + j] = (byte)q;
                values[b * 32 + j] = q * (float)d;
            }
        }
        var x = new float[rows * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int r = 0; r < rows; r++) x[r * k + rng.Next(k)] *= 60;   // one outlier a row, as a vision tower's rows carry
        var reference = new double[rows * outDim];
        double scale = 0;
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < outDim; o++)
            {
                double s = 0;
                for (int i = 0; i < k; i++) s += (double)values[o * k + i] * x[r * k + i];
                reference[r * outDim + o] = s;
                scale = Math.Max(scale, Math.Abs(s));
            }

        using var allocator = new CudaAllocator();
        IntPtr host = Marshal.AllocHGlobal((nint)bytes);
        GCHandle hx = GCHandle.Alloc(x, GCHandleType.Pinned);
        var error = new double[2];
        try
        {
            Marshal.Copy(raw, 0, host, (int)bytes);
            var weight = new Weight(host, k, outDim, bytes);
            for (int floatProducts = 0; floatProducts < 2; floatProducts++)
            {
                var g = new GraphProgramBuilder(record: true) { FloatProducts = floatProducts == 1 };
                int input = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { k, rows }, IntPtr.Zero, rows * k * f);
                int w = g.Leaf(GraphLeafRole.Weight, GraphFormat.Q8_0, new long[] { k, outDim }, host, bytes, handle: weight);
                g.Output(g.Op(GraphOp.MulMat, w, input), (long)rows * outDim);
                using var executor = new CudaGraphExecutor(allocator);
                using IGraphPlan plan = executor.Compile(g.ToProgram(replayable: true, reorder: false));
                Assert.True(plan.Run(new[] { input }, new[] { hx.AddrOfPinnedObject() }, new[] { rows * k * f }), plan.LastError);
                float[] got = plan.Outputs[0];
                for (int i = 0; i < got.Length; i++)
                    error[floatProducts] = Math.Max(error[floatProducts], Math.Abs(got[i] - reference[i]) / scale);
            }
        }
        finally
        {
            hx.Free();
            CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
            Marshal.FreeHGlobal(host);
        }
        output.WriteLine($"{rows} rows: largest error over the largest value {error[1]:E2} with F32 products, {error[0]:E2} with the executor's");
        Assert.True(error[1] <= 2e-6, $"{rows} rows: the F32 products are {error[1]:E3} from the reference");
        Assert.True(error[0] > 10 * error[1], $"{rows} rows: the executor's own products ({error[0]:E3}) are as close as the F32 ones ({error[1]:E3})");
    }
}
