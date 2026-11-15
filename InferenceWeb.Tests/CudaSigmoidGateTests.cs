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
/// A gated attention output (GraphFusion.SigmoidGate) on the CUDA graph executor: attn * sigmoid(gate) with the gate a view
/// of the query projection (heads of queries and gates interleaved) read in place equals the region's ops (the gate copied
/// out, the sigmoid, the product) to the bit, for a projection of queries and gates alone and one packed with more columns
/// (rows a wider pitch apart), over prompt rows and a decode row.
/// </summary>
public sealed class CudaSigmoidGateTests
{
    [CudaTheory]
    [InlineData(70, 0)]
    [InlineData(1, 0)]
    [InlineData(33, 96)]
    public void SigmoidGateRegionMatchesItsOps(int rows, int extraColumns)
    {
        const int d = 64, heads = 3;
        const long f = sizeof(float);
        long qDim = (long)d * heads, width = 2 * qDim + extraColumns;
        var rng = new Random(rows + extraColumns);
        float[] attn = new float[rows * qDim], proj = new float[rows * width];
        for (int i = 0; i < attn.Length; i++) attn[i] = (float)(rng.NextDouble() * 4 - 2);
        for (int i = 0; i < proj.Length; i++) proj[i] = (float)(rng.NextDouble() * 16 - 8);
        using var allocator = new CudaAllocator();
        var got = new float[2][];
        for (int fused = 0; fused < 2; fused++)
        {
            var g = new GraphProgramBuilder(record: true);
            int a = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { qDim, rows }, IntPtr.Zero, rows * qDim * f);
            int p = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { width, rows }, IntPtr.Zero, rows * width * f);
            int gate = g.Op(GraphOp.View, stackalloc[] { p }, stackalloc long[] { 3, d, heads, rows, 1, 2 * d * f, width * f, 0, d * f });
            if (fused == 1) g.BeginFused(GraphFusion.SigmoidGate, stackalloc[] { a, gate });
            int cont = g.Op(GraphOp.Cont, gate);
            int sig = g.Op(GraphOp.Sigmoid, g.Op(GraphOp.Reshape, stackalloc[] { cont }, stackalloc long[] { qDim, rows }));
            int y = g.Op(GraphOp.Mul, a, sig);
            if (fused == 1) g.EndFused(stackalloc[] { y });
            g.Output(y, rows * qDim);
            using var executor = new CudaGraphExecutor(allocator);
            using IGraphPlan plan = executor.Compile(g.ToProgram(replayable: true, reorder: false));
            GCHandle ha = GCHandle.Alloc(attn, GCHandleType.Pinned), hp = GCHandle.Alloc(proj, GCHandleType.Pinned);
            try
            {
                Assert.True(plan.Run(new[] { a, p }, new[] { ha.AddrOfPinnedObject(), hp.AddrOfPinnedObject() },
                    new[] { rows * qDim * f, rows * width * f }), plan.LastError);
            }
            finally
            {
                ha.Free();
                hp.Free();
            }
            got[fused] = (float[])plan.Outputs[0].Clone();
            Assert.Equal(fused, executor.SigmoidGateRegions);
        }
        for (int i = 0; i < got[0].Length; i++)
            if (BitConverter.SingleToInt32Bits(got[0][i]) != BitConverter.SingleToInt32Bits(got[1][i]))
                Assert.Fail($"{rows} rows, {extraColumns} extra columns: element {i} is {got[1][i]:R} in the region, {got[0][i]:R} from its ops");
    }
}
