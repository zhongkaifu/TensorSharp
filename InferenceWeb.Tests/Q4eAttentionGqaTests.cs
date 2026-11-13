// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;
using static InferenceWeb.Tests.Q4eAttentionMmaTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The Flash Next engine's decode attention a KV head's query heads at a time (Q4eKernels: TS_Q4E_ATTN_GQA, a block per KV
/// head and four of its query heads, each cell's K and V loaded once for them) against a block per query head. Every head
/// keeps the per-head kernel's arithmetic, so the outputs must be equal bit for bit: rows short of the split threshold
/// (one block) and past it (16 splits, merged by the last block to arrive), dense rows and rows over a listed QSA
/// selection, the F16 and the Q8_0 cache, one sequence's rows and a batched step's rows each over its own caches.
/// </summary>
public sealed class Q4eAttentionGqaTests
{
    private const int HeadDim = 256, Rows = 8, Splits = 16, Listed = 300, Ctx = 1200;

    private readonly ITestOutputHelper _output;

    public Q4eAttentionGqaTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(24, 2, false, false)]
    [InlineData(24, 2, true, false)]
    [InlineData(24, 2, false, true)]
    [InlineData(24, 2, true, true)]
    [InlineData(8, 2, true, true)]
    public void TheGroupedKernel_GivesThePerHeadBits(int nHead, int nKv, bool q8, bool batchedRows)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var qk = Q4eKernels.Create();
        Assert.True(qk.HasAttentionGqa, "the loaded tensorsharp_q4e_kernels module has no grouped attention");
        IntPtr stream = allocator.Stream.Handle;
        var rnd = new Random(nHead * 13 + nKv + (q8 ? 5 : 0) + (batchedRows ? 3 : 0));
        float U(double a) => (float)((rnd.NextDouble() * 2 - 1) * a);

        var q = new float[Rows * nHead * HeadDim];
        var gate = new float[q.Length];
        for (int i = 0; i < q.Length; i++)
        {
            q[i] = U(2.0);
            gate[i] = U(3.0);
        }
        // Positions: some rows below the split threshold (256 cells), some past it; two caches for a batched step.
        int[] positions = { 40, 700, 255, 1100, 900, 3, 600, 1199 };
        var cells = new int[Rows * Listed];
        var cellCnt = new int[Rows];
        for (int t = 0; t < Rows; t++)
        {
            int p = positions[t];
            if (t % 3 != 1 || p + 1 < Listed)
            {
                cellCnt[t] = -1;
                continue;
            }
            cellCnt[t] = Listed;
            var pick = new int[p + 1];
            for (int c = 0; c <= p; c++)
                pick[c] = c;
            for (int c = 0; c < Listed; c++)
            {
                int j = c + rnd.Next(p + 1 - c);
                (pick[c], pick[j]) = (pick[j], pick[c]);
                cells[t * Listed + c] = pick[c];
            }
        }
        long kvValues = (long)Ctx * nKv * HeadDim;
        Tensor[] k = new Tensor[2], v = new Tensor[2], ks = new Tensor[2], vs = new Tensor[2];
        for (int c = 0; c < 2; c++)
        {
            var kF = new float[kvValues];
            var vF = new float[kvValues];
            for (long i = 0; i < kvValues; i++)
            {
                kF[i] = U(2.0);
                vF[i] = U(1.5);
            }
            if (q8)
            {
                (byte[] kc, byte[] kscale) = Q8(kF);
                (byte[] vc, byte[] vscale) = Q8(vF);
                k[c] = Up(allocator, kc);
                v[c] = Up(allocator, vc);
                ks[c] = Up(allocator, kscale);
                vs[c] = Up(allocator, vscale);
            }
            else
            {
                k[c] = Up(allocator, F16(kF));
                v[c] = Up(allocator, F16(vF));
            }
        }
        using var dQ = Up(allocator, q);
        using var dGate = Up(allocator, gate);
        using var dCells = Up(allocator, cells);
        using var dCnt = Up(allocator, cellCnt);
        using var dPos = Up(allocator, positions);
        using var part = new Tensor(allocator, DType.Float32, (long)Rows * nHead * Splits * (2 + HeadDim));
        using var arrive = new Tensor(allocator, DType.Int32, (long)Rows * nHead);
        allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(arrive), Rows * nHead, 0.0f, stream);
        using var outHead = new Tensor(allocator, DType.Float32, q.Length);
        using var outGroup = new Tensor(allocator, DType.Float32, q.Length);
        IntPtr P(Tensor t) => t == null ? IntPtr.Zero : Dsv4CudaEngine.Ptr(t);
        float scale = 1.0f / MathF.Sqrt(HeadDim);
        try
        {
            foreach (bool grouped in new[] { false, true })
            {
                Q4eKernels.AttnGqaOverride = grouped;
                Tensor o = grouped ? outGroup : outHead;
                if (batchedRows)
                {
                    IntPtr[] Table(Tensor[] c)
                    {
                        var a = new IntPtr[Rows];
                        for (int t = 0; t < Rows; t++)
                            a[t] = P(c[t % 2]);
                        return a;
                    }
                    var kp = Q4eKernels.RowPtrs.Of(Table(k));
                    var vp = Q4eKernels.RowPtrs.Of(Table(v));
                    var ksp = q8 ? Q4eKernels.RowPtrs.Of(Table(ks)) : default;
                    var vsp = q8 ? Q4eKernels.RowPtrs.Of(Table(vs)) : default;
                    qk.AttentionSplitRows(P(dQ), kp, vp, P(dGate), P(dCells), P(dCnt), P(o), P(dPos), nHead, nKv, scale, Listed,
                        Rows, Splits, P(part), P(arrive), stream, q8, ksp, vsp);
                }
                else
                    qk.AttentionSplit(P(dQ), P(k[0]), P(v[0]), P(dGate), P(dCells), P(dCnt), P(o), P(dPos), 0, nHead, nKv, scale,
                        Listed, Rows, Splits, P(part), P(arrive), stream, P(ks[0]), P(vs[0]));
            }
            allocator.Stream.Synchronize();
        }
        finally
        {
            Q4eKernels.AttnGqaOverride = null;
            for (int c = 0; c < 2; c++)
            {
                k[c]?.Dispose();
                v[c]?.Dispose();
                ks[c]?.Dispose();
                vs[c]?.Dispose();
            }
        }
        float[] want = Down(outHead, q.Length), got = Down(outGroup, q.Length);
        for (int i = 0; i < q.Length; i++)
            if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(want[i]))
                Assert.Fail($"row {i / (nHead * HeadDim)}, head {i / HeadDim % nHead}, dim {i % HeadDim}: grouped {got[i]:R}, per head {want[i]:R}");
        _output.WriteLine($"{nHead}/{nKv} heads, {(q8 ? "Q8_0" : "F16")} cache, {(batchedRows ? "batched rows" : "one sequence")}: " +
            $"the grouped kernel bit-equal to a block per head over {Rows} rows");
    }
}
