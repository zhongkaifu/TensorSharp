// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Attention over K/V that are not cache windows under an explicit additive mask, on the CUDA graph executor: a decode row
/// takes the generic attention (TensorGraphExecutor.Attention), whose mask went onto the scores as a view expanded over the
/// heads - an operand direct CUDA has no kernel for, so every such row ran on the element-by-element host fallback (64
/// concurrent LFM3 requests on an A40 hit it). The mask is now added on the device (CudaGraphExecutor.AddAttentionMask): no
/// host fallback runs, and the rows equal a host reference.
/// </summary>
public sealed class CudaMaskedAttentionTests
{
    [CudaTheory]
    [InlineData(1, 37, 4, 2)]
    [InlineData(1, 300, 16, 4)]
    [InlineData(3, 129, 8, 8)]
    public void MaskedRowsRunOnTheDeviceAndMatchAHostReference(int rows, int keys, int heads, int kvHeads)
    {
        const int d = 64;
        const long f = sizeof(float);
        float scale = 1f / MathF.Sqrt(d);
        var rng = new Random(rows * 1000 + keys);
        float[] q = Random(rng, d * rows * heads), k = Random(rng, d * keys * kvHeads), v = Random(rng, d * keys * kvHeads);
        // ggml's mask [keys, rows] in F16: a fifth of the keys closed, the rest small biases, key 0 open in every row.
        var mask = new float[rows * keys];
        var half = new ushort[mask.Length];
        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = i % keys != 0 && rng.Next(5) == 0 ? float.NegativeInfinity : (float)Math.Round(rng.NextDouble() - 0.5, 2);
            half[i] = BitConverter.HalfToUInt16Bits((Half)mask[i]);
            mask[i] = (float)BitConverter.UInt16BitsToHalf(half[i]);
        }

        var g = new GraphProgramBuilder(record: true);
        int ql = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { d, rows, heads }, IntPtr.Zero, q.Length * f);
        int kl = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { d, keys, kvHeads }, IntPtr.Zero, k.Length * f);
        int vl = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { d, keys, kvHeads }, IntPtr.Zero, v.Length * f);
        int ml = g.Leaf(GraphLeafRole.Input, GraphFormat.F16, new long[] { keys, rows }, IntPtr.Zero, half.Length * sizeof(ushort));
        int fa = g.Op(GraphOp.FlashAttnExt, stackalloc[] { ql, kl, vl, ml }, stackalloc long[] { 10 }, stackalloc[] { scale, 0f, 0f });
        g.Output(fa, (long)rows * heads * d);

        using var allocator = new CudaAllocator();
        using var executor = new CudaGraphExecutor(allocator);
        using IGraphPlan plan = executor.Compile(g.ToProgram(replayable: false, reorder: false));
        var pins = new[] { GCHandle.Alloc(q, GCHandleType.Pinned), GCHandle.Alloc(k, GCHandleType.Pinned),
            GCHandle.Alloc(v, GCHandleType.Pinned), GCHandle.Alloc(half, GCHandleType.Pinned) };
        long fallbacks = CudaCpuFallback.ThreadCalls;
        try
        {
            Assert.True(plan.Run(new[] { ql, kl, vl, ml }, Array.ConvertAll(pins, p => p.AddrOfPinnedObject()),
                new[] { q.Length * f, k.Length * f, v.Length * f, half.Length * (long)sizeof(ushort) }), plan.LastError);
        }
        finally
        {
            foreach (GCHandle p in pins) p.Free();
        }
        Assert.Equal(fallbacks, CudaCpuFallback.ThreadCalls);

        // The reference: ggml's result [d, heads, rows]; head h reads KV head h / (heads / kvHeads).
        float[] got = plan.Outputs[0];
        int group = heads / kvHeads;
        var scores = new double[keys];
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < heads; h++)
            {
                int kv = h / group;
                double max = double.NegativeInfinity;
                for (int j = 0; j < keys; j++)
                {
                    double s = 0;
                    for (int i = 0; i < d; i++) s += (double)q[(h * rows + r) * d + i] * k[(kv * keys + j) * d + i];
                    scores[j] = s * scale + mask[r * keys + j];
                    max = Math.Max(max, scores[j]);
                }
                double sum = 0;
                for (int j = 0; j < keys; j++) sum += scores[j] = Math.Exp(scores[j] - max);
                for (int i = 0; i < d; i++)
                {
                    double o = 0;
                    for (int j = 0; j < keys; j++) o += scores[j] / sum * v[(kv * keys + j) * d + i];
                    float actual = got[(r * heads + h) * d + i];
                    Assert.True(Math.Abs(actual - o) <= 1e-4 + 1e-4 * Math.Abs(o),
                        $"row {r} head {h} element {i}: {actual:R}, the reference {o:R}");
                }
            }
    }

    private static float[] Random(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
