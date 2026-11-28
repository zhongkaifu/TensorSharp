// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Graph;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The tensor executor keeps a large F32 input (past 65,536 floats: a prompt chunk's rows) on the host until its first reader
/// uploads it. A view is such a reader: a prompt-sized projection fed as an input and viewed head by head (as
/// CudaAttentionPrepRegionTests feeds its 2,048-row chunks) read a null tensor, where a small one had been uploaded already.
/// </summary>
public sealed class GraphLargeInputViewTests
{
    private static readonly IAllocator Cpu = new CpuAllocator(BlasEnum.DotNet);

    private sealed class Executor : TensorGraphExecutor
    {
        public Executor() : base(Cpu) { }

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x) => throw new NotSupportedException();
    }

    [Theory]
    [InlineData(8)]      // 8 x 2 x 1,024 floats: uploaded with the input
    [InlineData(64)]     // 131,072 floats: kept on the host until read
    public void AViewOfAnInputReadsItsRowsAtAnySize(int rows)
    {
        const int d = 256, heads = 4;
        int width = 2 * heads * d;
        float[] projection = Enumerable.Range(0, width * rows).Select(i => (float)(i % 977) - 400f).ToArray();
        var g = new GraphProgramBuilder(record: true);
        int input = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { width, rows }, IntPtr.Zero, projection.Length * 4L);
        // Each head's first d values of the q|gate pairs, copied out as GraphStep.Cont copies a strided view.
        int q = g.Op(GraphOp.View, stackalloc[] { input }, stackalloc long[] { 3, d, heads, rows, 1, 2L * d * 4, width * 4L, 0, 0 });
        g.Output(g.Op(GraphOp.Cont, stackalloc[] { q }), (long)d * heads * rows);
        using IGraphPlan plan = new Executor().Compile(g.ToProgram(replayable: true, reorder: false));
        GCHandle pin = GCHandle.Alloc(projection, GCHandleType.Pinned);
        try
        {
            Assert.True(plan.Run(new[] { input }, new[] { pin.AddrOfPinnedObject() }, new[] { projection.Length * 4L }), plan.LastError);
        }
        finally
        {
            pin.Free();
        }
        float[] got = plan.Outputs[0];
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < d; i++)
                    Assert.Equal(projection[r * width + h * 2 * d + i], got[(r * heads + h) * d + i]);
    }

    /// <summary>A product's first operand bound as an input (MiniMax-H3-Image's PDD head, [hidden, patch] F32 every step): past
    /// 65,536 floats it stays on the host until read, and the product read its absent tensor (the real 5,376-wide head on
    /// direct CUDA; the test checkpoints' heads are small enough to be uploaded with the input).</summary>
    [Theory]
    [InlineData(16)]     // 256 x 16 floats: uploaded with the input
    [InlineData(320)]    // 81,920 floats: kept on the host until read
    public void AProductOfAnInputFirstOperandReadsItAtAnySize(int outputs)
    {
        const int k = 256, tokens = 3;
        float[] head = Enumerable.Range(0, k * outputs).Select(i => (float)((i * 37) % 101 - 50) / 64f).ToArray();
        float[] x = Enumerable.Range(0, k * tokens).Select(i => (float)((i * 11) % 23 - 11) / 8f).ToArray();
        var g = new GraphProgramBuilder(record: true);
        int w = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { k, outputs }, IntPtr.Zero, head.Length * 4L);
        int a = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { k, tokens }, IntPtr.Zero, x.Length * 4L);
        g.Output(g.Op(GraphOp.MulMat, w, a), (long)outputs * tokens);
        using IGraphPlan plan = new Executor().Compile(g.ToProgram(replayable: true, reorder: false));
        GCHandle hw = GCHandle.Alloc(head, GCHandleType.Pinned), hx = GCHandle.Alloc(x, GCHandleType.Pinned);
        try
        {
            Assert.True(plan.Run(new[] { w, a }, new[] { hw.AddrOfPinnedObject(), hx.AddrOfPinnedObject() },
                new[] { head.Length * 4L, x.Length * 4L }), plan.LastError);
        }
        finally
        {
            hw.Free();
            hx.Free();
        }
        float[] got = plan.Outputs[0];
        for (int t = 0; t < tokens; t++)
            for (int o = 0; o < outputs; o++)
            {
                double want = 0;
                for (int i = 0; i < k; i++) want += (double)head[o * k + i] * x[t * k + i];
                Assert.True(Math.Abs(got[t * outputs + o] - want) <= 1e-4 * (1 + Math.Abs(want)),
                    $"token {t} output {o}: {got[t * outputs + o]} against {want}");
            }
    }
}
