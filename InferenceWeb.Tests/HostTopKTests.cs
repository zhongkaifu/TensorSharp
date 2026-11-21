// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Graph;

namespace InferenceWeb.Tests;

/// <summary>
/// The managed executors' ggml_top_k (TensorGraphExecutor.TopK, the managed CPU's and MLX's): the k picks of each row are a
/// full sort's first k, largest first and a tie to the lower index, NaN below every number, whether the row is selected in
/// one pass (up to 64 picks) or sorted (more).
/// </summary>
public sealed class HostTopKTests
{
    private static readonly IAllocator Cpu = new CpuAllocator(BlasEnum.DotNet);

    private sealed class ManagedExecutor : TensorGraphExecutor
    {
        public ManagedExecutor() : base(Cpu) { }
        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x) => throw new NotSupportedException();
    }

    [Theory]
    [InlineData(1, 7)]
    [InlineData(3, 40)]
    [InlineData(16, 5000)]
    [InlineData(64, 5000)]
    [InlineData(65, 300)]
    public void PicksAreAFullSortsFirstK(int k, int cols)
    {
        var rng = new Random(391 + k);
        var rows = new List<float[]>();
        // Distinct values, many ties, NaNs and infinities, a constant row, and a rising row (every value displaces the run).
        rows.Add(Enumerable.Range(0, cols).Select(_ => (float)rng.NextDouble()).ToArray());
        rows.Add(Enumerable.Range(0, cols).Select(_ => (float)rng.Next(4)).ToArray());
        rows.Add(Enumerable.Range(0, cols).Select(i => i % 5 == 0 ? float.NaN : i % 7 == 0 ? float.NegativeInfinity
            : i % 11 == 0 ? float.PositiveInfinity : (float)rng.NextDouble()).ToArray());
        rows.Add(Enumerable.Repeat(0.5f, cols).ToArray());
        rows.Add(Enumerable.Range(0, cols).Select(i => (float)i).ToArray());
        int n = rows.Count;
        var values = new float[(long)cols * n];
        for (int r = 0; r < n; r++) Array.Copy(rows[r], 0, values, (long)r * cols, cols);

        var g = new GraphProgramBuilder(record: true);
        int input = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { cols, n }, IntPtr.Zero, values.LongLength * 4);
        int picks = g.Op(GraphOp.Cont, stackalloc[] { g.Op(GraphOp.TopK, stackalloc[] { input }, stackalloc long[] { k }) });
        g.Output(picks, (long)k * n);   // read back as F32 values (exact below 2^24)
        using IGraphPlan plan = new ManagedExecutor().Compile(g.ToProgram(replayable: true, reorder: false));
        GCHandle pin = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            Assert.True(plan.Run(new[] { input }, new[] { pin.AddrOfPinnedObject() }, new[] { values.LongLength * 4 }), plan.LastError);
        }
        finally
        {
            pin.Free();
        }
        float[] got = plan.Outputs[0];
        for (int r = 0; r < n; r++)
        {
            float[] row = rows[r];
            int[] order = Enumerable.Range(0, cols).ToArray();
            Array.Sort(order, (i, j) => row[j].CompareTo(row[i]) is var cmp && cmp != 0 ? cmp : i.CompareTo(j));
            int[] picked = Enumerable.Range(0, k).Select(i => (int)got[r * k + i]).ToArray();
            Assert.True(order.Take(k).SequenceEqual(picked),
                $"k {k} row {r}: picked [{string.Join(",", picked)}], a full sort [{string.Join(",", order.Take(k))}]");
        }
    }
}
