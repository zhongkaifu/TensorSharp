// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// Two prefill kernels of the direct-CUDA Qwen 3.8 Flash Next engine, reorganized for speed, against the kernels
/// they replace, bit for bit: the gated-delta (KDA) recurrence with one state row per warp
/// (<c>ts_glm_kda_scan_r1_f32</c>, which GLM's engine shares) against four rows a warp, and attention with four
/// query heads of a KV head per block (<c>ts_q4e_attention_g4_f32</c>) against a block per head.
/// </summary>
public sealed class Qwen4ExpPrefillKernelTests
{
    /// <summary>Every output row and every final state element of a ubatch's scan, for one sequence and for
    /// several (a batched GLM prefill), equals the four-rows-a-warp scan's.</summary>
    [CudaTheory]
    [InlineData(1, 37)]
    [InlineData(2, 64)]
    [InlineData(3, 2)]
    public void TheScanWithAStateRowAWarp_IsTheFourRowScanBitForBit(int nSeq, int rowsPerSeq)
    {
        const int H = 6, HD = GlmKernels.KdaHeadDim, Scr = GlmKernels.KdaScratch;
        var rng = new Random(nSeq * 1000 + rowsPerSeq);
        var scr = new float[(long)nSeq * H * rowsPerSeq * Scr];
        for (int i = 0; i < scr.Length; i++)
        {
            int f = i % Scr;
            scr[i] = f < 2 * HD ? (float)(rng.NextDouble() - 0.5) * 0.25f // q, k
                : f < 3 * HD ? (float)(rng.NextDouble() - 0.5)          // v
                : f < 4 * HD ? -(float)rng.NextDouble() * 0.3f          // log decay per key channel
                : (float)rng.NextDouble();                              // beta
        }
        var initial = Enumerable.Range(0, nSeq)
            .Select(s => Enumerable.Range(0, H * HD * HD).Select(i => MathF.Sin(0.013f * i + s) * 0.05f).ToArray())
            .ToArray();

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var gk = GlmKernels.Create();

        (float[] core, float[][] states) Run(bool fourRows)
        {
            bool saved = GlmKernels.KdaScanFourRows;
            GlmKernels.KdaScanFourRows = fourRows;
            var stateTensors = initial.Select(st => Up(allocator, st)).ToArray();
            try
            {
                // The scan takes a device array of the sequences' state pointers.
                using var ptrs = Up(allocator, stateTensors.SelectMany(t => BitConverter.GetBytes((long)Dsv4CudaEngine.Ptr(t))).ToArray());
                using var scrT = Up(allocator, scr);
                using var core = new Tensor(allocator, DType.Float32, (long)nSeq * rowsPerSeq * H * HD);
                gk.KdaScan(Dsv4CudaEngine.Ptr(scrT), Dsv4CudaEngine.Ptr(ptrs), Dsv4CudaEngine.Ptr(core), rowsPerSeq, nSeq, H,
                    allocator.Stream.Handle);
                allocator.Stream.Synchronize();
                return (Down(core, nSeq * rowsPerSeq * H * HD), stateTensors.Select(t => Down(t, H * HD * HD)).ToArray());
            }
            finally
            {
                GlmKernels.KdaScanFourRows = saved;
                foreach (var t in stateTensors) t.Dispose();
            }
        }

        var (wantCore, wantStates) = Run(fourRows: true);
        var (gotCore, gotStates) = Run(fourRows: false);
        Assert.Contains(wantCore, v => v != 0f);
        Assert.All(wantCore, v => Assert.True(float.IsFinite(v)));
        AssertBits(wantCore, gotCore, "outputs");
        for (int s = 0; s < nSeq; s++)
        {
            Assert.NotEqual(Bits(initial[s]), Bits(wantStates[s]));
            AssertBits(wantStates[s], gotStates[s], $"sequence {s}'s state");
        }
    }

    /// <summary>A prefill ubatch's attention in groups of four query heads equals a block per head, to the bit:
    /// rows over the cells 0..p (the start of a prompt and deep into one) and rows over their own lists of
    /// selected cells, a few of them dense.</summary>
    [CudaTheory]
    [InlineData(40, 0, false)]
    [InlineData(33, 3000, false)]
    [InlineData(48, 5000, true)]
    public void PrefillAttentionInGroupsOfFourHeads_IsTheHeadABlockAttentionBitForBit(int rows, int p0, bool sparse)
    {
        const int NH = 8, NKV = 2, HD = Q4eKernels.AttnHeadDim, CellStride = 160;
        var rng = new Random(rows * 7 + p0);
        int nCtx = p0 + rows;
        var kv = new byte[2][];
        for (int c = 0; c < 2; c++)
        {
            kv[c] = new byte[(long)nCtx * NKV * HD * 2];
            for (long i = 0; i < kv[c].Length; i += 2)
                BitConverter.TryWriteBytes(kv[c].AsSpan((int)i, 2), BitConverter.HalfToUInt16Bits((System.Half)(rng.NextDouble() * 2 - 1)));
        }
        float[] q = Enumerable.Range(0, rows * NH * HD).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] gate = Enumerable.Range(0, rows * NH * HD).Select(_ => (float)(rng.NextDouble() * 4 - 2)).ToArray();
        var cells = new int[rows * CellStride];
        var counts = new int[rows];
        for (int t = 0; t < rows; t++)
        {
            int p = p0 + t;
            counts[t] = t % 7 == 3 ? -1 : 1 + rng.Next(CellStride);
            for (int i = 0; i < Math.Max(0, counts[t]); i++)
                cells[t * CellStride + i] = rng.Next(p + 1);
        }

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var qk = Q4eKernels.Create();
        using var kCache = Up(allocator, kv[0]);
        using var vCache = Up(allocator, kv[1]);
        using var qT = Up(allocator, q);
        using var gateT = Up(allocator, gate);
        using var cellsT = Up(allocator, cells);
        using var countsT = Up(allocator, counts);
        using var output = new Tensor(allocator, DType.Float32, (long)rows * NH * HD);

        float[] Run(bool grouped)
        {
            bool saved = Q4eKernels.GroupedPrefillAttention;
            Q4eKernels.GroupedPrefillAttention = grouped;
            try
            {
                allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(output), rows * NH * HD, float.NaN, allocator.Stream.Handle);
                qk.Attention(Dsv4CudaEngine.Ptr(qT), Dsv4CudaEngine.Ptr(kCache), Dsv4CudaEngine.Ptr(vCache), Dsv4CudaEngine.Ptr(gateT),
                    sparse ? Dsv4CudaEngine.Ptr(cellsT) : IntPtr.Zero, sparse ? Dsv4CudaEngine.Ptr(countsT) : IntPtr.Zero,
                    Dsv4CudaEngine.Ptr(output), IntPtr.Zero, p0, NH, NKV, 1f / MathF.Sqrt(HD), CellStride, rows,
                    allocator.Stream.Handle);
                allocator.Stream.Synchronize();
                return Down(output, rows * NH * HD);
            }
            finally { Q4eKernels.GroupedPrefillAttention = saved; }
        }

        float[] want = Run(grouped: false);
        float[] got = Run(grouped: true);
        Assert.All(want, v => Assert.True(float.IsFinite(v)));
        AssertBits(want, got, "attention outputs");
    }

    private static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    private static void AssertBits(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
                Assert.Fail($"{what}: element {i} is {got[i]:R}, the original kernel's {want[i]:R}");
    }
}
