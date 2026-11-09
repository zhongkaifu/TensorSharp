// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The Flash Next engine's QSA block scores for a prompt on tensor cores (<see cref="Q4eKernels.QsaScoresMma"/>, after Strata's
/// prompt scoring) against the warp-per-block kernel it replaces for prompts and against the scores in double: four indexer
/// heads of 128, blocks of 4 positions, rows that start at the context's beginning and rows deep into it. Both operands go to
/// the tensor cores as F16 high and low halves, so the scores stay within F32 rounding of the double ones and the block
/// selection that reads them keeps the same cells. A row writes only the blocks it sees.
/// </summary>
public sealed class Q4eQsaScoresMmaTests
{
    private const int Heads = 4, D = 128, R = 4, TopK = 512, Sentinel = -7;
    private readonly ITestOutputHelper _output;

    public Q4eQsaScoresMmaTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(0, 333)]
    [InlineData(3000, 200)]
    [InlineData(9000, 512)]
    public void ThePromptScores_MatchTheWarpKernelAndTheDoubleScores(int p0, int nt)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var qk = Q4eKernels.Create();
        Assert.True(qk.HasQsaScoresMma, "the loaded tensorsharp_q4e_kernels module has no tensor-core QSA scores for this GPU");
        IntPtr stream = allocator.Stream.Handle;
        var rnd = new Random(p0 + nt);
        int blocks = (p0 + nt) / R, stride = blocks + 64, cellStride = TopK + R - 1;

        // RMS-normed heads and keys with a spread of magnitudes, as the indexer's query and pooled keys are.
        float[] Unit(int rows)
        {
            var x = new float[(long)rows * D];
            for (int r = 0; r < rows; r++)
            {
                double ss = 0;
                for (int d = 0; d < D; d++) { x[(long)r * D + d] = (float)((rnd.NextDouble() * 2 - 1) * (d % 17 == 3 ? 4 : 1)); ss += x[(long)r * D + d] * x[(long)r * D + d]; }
                float scale = (float)(Math.Sqrt(D / ss) * (0.5 + rnd.NextDouble()));
                for (int d = 0; d < D; d++) x[(long)r * D + d] *= scale;
            }
            return x;
        }
        float[] q = Unit(nt * Heads);
        float[] pooled = Unit(Math.Max(1, blocks));
        var init = new float[(long)nt * stride];
        Array.Fill(init, Sentinel);

        using Tensor qT = Up(allocator, q);
        using Tensor pooledT = Up(allocator, pooled);
        using Tensor warpScores = Up(allocator, init);
        using Tensor mmaScores = Up(allocator, init);
        using Tensor warpCells = new Tensor(allocator, DType.Int32, nt, cellStride);
        using Tensor mmaCells = new Tensor(allocator, DType.Int32, nt, cellStride);
        using Tensor warpCnt = new Tensor(allocator, DType.Int32, nt);
        using Tensor mmaCnt = new Tensor(allocator, DType.Int32, nt);
        IntPtr P(Tensor t) => Dsv4CudaEngine.Ptr(t);
        qk.QsaScores(P(qT), P(pooledT), P(warpScores), IntPtr.Zero, p0, Heads, D, R, stride, blocks, nt, stream);
        qk.QsaScoresMma(P(qT), P(pooledT), P(mmaScores), p0, D, R, stride, blocks, nt, stream);
        qk.QsaSelect(P(warpScores), P(warpCells), P(warpCnt), IntPtr.Zero, p0, R, TopK, stride, cellStride, nt, stream);
        qk.QsaSelect(P(mmaScores), P(mmaCells), P(mmaCnt), IntPtr.Zero, p0, R, TopK, stride, cellStride, nt, stream);
        allocator.Stream.Synchronize();
        float[] w = Down(warpScores, nt * stride), m = Down(mmaScores, nt * stride);

        double worstMma = 0, worstWarp = 0;
        object gate = new();
        Parallel.For(0, nt, t =>
        {
            int nb = (p0 + t + 1) / R;
            double localMma = 0, localWarp = 0;
            for (int j = 0; j < stride; j++)
            {
                long at = (long)t * stride + j;
                if (j >= nb)
                {
                    Assert.Equal(Sentinel, m[at]);
                    continue;
                }
                double s = 0, mag = 0;
                for (int h = 0; h < Heads; h++)
                {
                    double dot = 0, absDot = 0;
                    for (int d = 0; d < D; d++)
                    {
                        double p = (double)q[((long)t * Heads + h) * D + d] * pooled[(long)j * D + d];
                        dot += p;
                        absDot += Math.Abs(p);
                    }
                    s += Math.Max(dot, 0);
                    mag += absDot;
                }
                localMma = Math.Max(localMma, Math.Abs(m[at] - s) / mag);
                localWarp = Math.Max(localWarp, Math.Abs(w[at] - s) / mag);
            }
            lock (gate)
            {
                worstMma = Math.Max(worstMma, localMma);
                worstWarp = Math.Max(worstWarp, localWarp);
            }
        });

        ((CudaStorage)warpCnt.Storage).MarkDeviceModified();
        ((CudaStorage)mmaCnt.Storage).MarkDeviceModified();
        ((CudaStorage)warpCells.Storage).MarkDeviceModified();
        ((CudaStorage)mmaCells.Storage).MarkDeviceModified();
        int[] wc = warpCnt.GetElementsAsInt(nt), mc = mmaCnt.GetElementsAsInt(nt);
        int[] wcells = warpCells.GetElementsAsInt(nt * cellStride), mcells = mmaCells.GetElementsAsInt(nt * cellStride);
        int sameRows = 0, selecting = 0;
        for (int t = 0; t < nt; t++)
        {
            if (wc[t] >= 0)
                selecting++;
            bool same = wc[t] == mc[t];
            for (int i = 0; same && i < Math.Max(0, wc[t]); i++)
                same = wcells[t * cellStride + i] == mcells[t * cellStride + i];
            if (same)
                sameRows++;
        }
        _output.WriteLine($"p0 {p0}, {nt} rows: worst |score - double| / sum|q.k|: tensor cores {worstMma:E2}, warp kernel {worstWarp:E2}; " +
            $"selection identical in {sameRows}/{nt} rows ({selecting} past the width)");
        Assert.True(worstMma <= 4e-6, $"tensor-core scores strayed {worstMma:E2} of sum|q.k| from the double scores");
        // A row may keep another block only where two blocks' scores tie within that rounding at the cut.
        Assert.True(nt - sameRows <= nt / 100, $"{nt - sameRows} of {nt} rows selected other blocks");
    }
}
