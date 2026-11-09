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
/// The direct-CUDA Qwen 3.8 Flash Next engine's prefill gated-delta (KDA) recurrence with one state row per warp
/// (<c>ts_glm_kda_scan_r1_f32</c>, which GLM's engine shares), reorganized for speed, against the four-rows-a-warp
/// scan it replaces there, bit for bit.
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

    private static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    private static void AssertBits(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
                Assert.Fail($"{what}: element {i} is {got[i]:R}, the original kernel's {want[i]:R}");
    }
}
