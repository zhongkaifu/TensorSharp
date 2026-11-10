// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;
using Layout = TensorSharp.Cuda.Dsv4Kernels.Q80ExpertLayout;

namespace InferenceWeb.Tests;

/// <summary>
/// The layouts the direct engine's Q8_0 decode experts run in (<see cref="Dsv4Kernels.Q80ExpertLayout"/>): a warp a
/// row, a warp on a slot's gate and up rows, and eight lanes a row (for gate/up, on a slot's two rows) with a lane on
/// whole blocks read as aligned 64-bit words. They are one arithmetic, so the launch rules may pick by speed alone: held here to the bit, at
/// widths and row counts that leave part blocks of rows, part warps and part lane sets, with the weights at every
/// phase of a word and through a residency table.
/// </summary>
public sealed class Q80ExpertLayoutTests
{
    private const int Q8_0 = (int)GgmlTensorType.Q8_0;
    private const int Experts = 6;

    private readonly ITestOutputHelper _output;

    public Q80ExpertLayoutTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Gate and up rows are the warp-a-row kernel's to the bit in every other layout, and no layout writes past the
    /// slots it was given. 2560 -> 640 is Qwen 3.8 Flash Next's shape; the others have row counts off a multiple of
    /// four (a warp's rows), of 8 and 16 (a block's) and inputs of 1 to 33 blocks (lanes with none, one and several).
    /// </summary>
    [CudaTheory]
    [InlineData(640, 2560, 0, false, 10)]
    [InlineData(640, 2560, 0, true, 10)]
    [InlineData(633, 2560, 34, false, 10)]
    [InlineData(100, 96, 102, false, 10)]
    [InlineData(17, 32, 68, false, 7)]
    [InlineData(250, 608, 170, true, 10)]
    [InlineData(333, 224, 238, false, 3)]
    [InlineData(16, 256, 4, false, 10)]
    [InlineData(1, 576, 6, false, 1)]
    [InlineData(640, 1056, 374, false, 10)]
    public void GateAndUpRows_AreTheSameBitsInEveryLayout(int ff, int inDim, int weightOffset, bool table, int used)
    {
        const int nt = 3;
        int slots = nt * used;
        long rowBytes = inDim / 32 * 34;
        long expertBytes = ff * rowBytes;
        byte[] gate = QuantRows(Q8_0, Experts * ff, rowBytes, seed: 31);
        byte[] up = QuantRows(Q8_0, Experts * ff, rowBytes, seed: 32);
        var act = new float[nt, inDim];
        for (int t = 0; t < nt; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0191f * (k + 3) * (t + 1)) - 0.4f * MathF.Cos(0.057f * k + 2 * t);
        int[] sel = Enumerable.Range(0, slots).Select(s => (s * 5 + 1) % Experts).ToArray();

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        IntPtr stream = allocator.Stream.Handle;
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.HasQ80VecDecode && dk.HasQ80LaneDecode);
        using var gateW = Up(allocator, Shifted(gate, weightOffset));
        using var upW = Up(allocator, Shifted(up, weightOffset));
        IntPtr gatePtr = Dsv4CudaEngine.Ptr(gateW) + weightOffset, upPtr = Dsv4CudaEngine.Ptr(upW) + weightOffset;
        // A table that places expert e where the stacked tensor has another: every layout must follow it.
        using var gateTab = table ? Up(allocator, PointerTable(gatePtr, expertBytes)) : null;
        using var upTab = table ? Up(allocator, PointerTable(upPtr, expertBytes)) : null;
        using var actT = Up(allocator, act);
        using var actQ = new Tensor(allocator, DType.UInt8, nt * (inDim / 32) * 36L);
        using var selT = Up(allocator, sel);
        // A slot past the launch's: it must stay as filled.
        using var gateOut = new Tensor(allocator, DType.Float32, slots + 1, ff);
        using var upOut = new Tensor(allocator, DType.Float32, slots + 1, ff);
        allocator.Kernels.LaunchQuantizeQ81Rows(Dsv4CudaEngine.Ptr(actT), Dsv4CudaEngine.Ptr(actQ), inDim, nt, stream,
            warpCooperative: true);

        (float[] Gate, float[] Up) Run(Layout layout)
        {
            allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(gateOut), (slots + 1) * ff, float.NaN, stream);
            allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(upOut), (slots + 1) * ff, float.NaN, stream);
            dk.MoeGateUpDecodeQ80VecAs(layout, gatePtr, upPtr, Dsv4CudaEngine.Ptr(actQ), selT, gateOut, upOut, ff, inDim,
                rowBytes, nt, used, stream,
                table ? Dsv4CudaEngine.Ptr(gateTab) : IntPtr.Zero, table ? Dsv4CudaEngine.Ptr(upTab) : IntPtr.Zero);
            allocator.Stream.Synchronize();
            return (Down(gateOut, (slots + 1) * ff), Down(upOut, (slots + 1) * ff));
        }

        (float[] gate0, float[] up0) = Run(Layout.Warp);
        AssertWritten(gate0, slots * ff, "gate, a warp a row");
        AssertWritten(up0, slots * ff, "up, a warp a row");
        Assert.Contains(gate0.Take(slots * ff), v => v != 0f);
        foreach (Layout layout in new[] { Layout.WarpPair, Layout.Lanes })
        {
            (float[] g, float[] u) = Run(layout);
            AssertSameBits(gate0, g, slots * ff, ff, $"gate, {layout}");
            AssertSameBits(up0, u, slots * ff, ff, $"up, {layout}");
        }
        _output.WriteLine($"gate/up {inDim} -> {ff}, {slots} slots, weights at +{weightOffset}{(table ? ", through a table" : "")}: " +
            "three layouts, the same bits");
    }

    /// <summary>
    /// Down rows are the warp-a-row kernel's to the bit with eight lanes a row, at row counts that leave part
    /// thread blocks, part warps and part lane sets.
    /// </summary>
    [CudaTheory]
    [InlineData(2560, 640, 0, false, 10)]
    [InlineData(2560, 640, 0, true, 70)]
    [InlineData(2557, 640, 34, false, 61)]
    [InlineData(100, 96, 102, false, 10)]
    [InlineData(17, 32, 68, true, 64)]
    [InlineData(333, 224, 238, false, 59)]
    [InlineData(30, 1056, 6, false, 5)]
    [InlineData(1, 576, 4, false, 3)]
    [InlineData(250, 608, 170, false, 60)]
    public void DownRows_AreTheSameBitsWithEightLanesARow(int e, int ff, int weightOffset, bool table, int slots)
    {
        long rowBytes = ff / 32 * 34;
        long expertBytes = e * rowBytes;
        byte[] down = QuantRows(Q8_0, Experts * e, rowBytes, seed: 33);
        var act = new float[slots, ff];
        for (int s = 0; s < slots; s++)
            for (int k = 0; k < ff; k++)
                act[s, k] = MathF.Cos(0.0233f * (k + 1) * (s % 7 + 1)) + 0.25f * MathF.Sin(0.071f * k - s);
        int[] sel = Enumerable.Range(0, slots).Select(s => (s * 5 + 2) % Experts).ToArray();

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        IntPtr stream = allocator.Stream.Handle;
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.HasQ80VecDecode && dk.HasQ80LaneDecode);
        using var downW = Up(allocator, Shifted(down, weightOffset));
        IntPtr downPtr = Dsv4CudaEngine.Ptr(downW) + weightOffset;
        using var downTab = table ? Up(allocator, PointerTable(downPtr, expertBytes)) : null;
        using var actT = Up(allocator, act);
        using var actQ = new Tensor(allocator, DType.UInt8, slots * (ff / 32) * 36L);
        using var selT = Up(allocator, sel);
        using var downOut = new Tensor(allocator, DType.Float32, slots + 1, e);
        allocator.Kernels.LaunchQuantizeQ81Rows(Dsv4CudaEngine.Ptr(actT), Dsv4CudaEngine.Ptr(actQ), ff, slots, stream,
            warpCooperative: true);

        float[] Run(Layout layout)
        {
            allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(downOut), (slots + 1) * e, float.NaN, stream);
            dk.MoeDownDecodeQ80VecAs(layout, downPtr, actQ, selT, downOut, e, ff, rowBytes, slots, stream,
                table ? Dsv4CudaEngine.Ptr(downTab) : IntPtr.Zero);
            allocator.Stream.Synchronize();
            return Down(downOut, (slots + 1) * e);
        }

        float[] warp = Run(Layout.Warp);
        AssertWritten(warp, slots * e, "down, a warp a row");
        Assert.Contains(warp.Take(slots * e), v => v != 0f);
        AssertSameBits(warp, Run(Layout.Lanes), slots * e, e, "down, Lanes");
        _output.WriteLine($"down {ff} -> {e}, {slots} slots, weights at +{weightOffset}{(table ? ", through a table" : "")}: " +
            "eight lanes a row, the same bits");
    }

    /// <summary>
    /// What the gate/up launch picks: a warp a row below forty slots and a slot's two rows in eight lanes a row from
    /// there; without the lane kernels the warp layouts as before, the pair from 32; a warp a row whenever gate and
    /// up rows are to stay apart.
    /// </summary>
    [Theory]
    [InlineData(10, true, true, true, (int)Layout.Warp)]
    [InlineData(30, true, true, true, (int)Layout.Warp)]
    [InlineData(32, true, true, true, (int)Layout.WarpPair)]
    [InlineData(39, true, true, true, (int)Layout.WarpPair)]
    [InlineData(40, true, true, true, (int)Layout.Lanes)]
    [InlineData(160, true, true, true, (int)Layout.Lanes)]
    [InlineData(160, true, true, false, (int)Layout.Lanes)]
    [InlineData(39, true, true, false, (int)Layout.Warp)]
    [InlineData(160, true, false, true, (int)Layout.Warp)]
    [InlineData(10, false, true, true, (int)Layout.Warp)]
    [InlineData(31, false, true, true, (int)Layout.Warp)]
    [InlineData(32, false, true, true, (int)Layout.WarpPair)]
    [InlineData(160, false, true, true, (int)Layout.WarpPair)]
    [InlineData(160, false, false, true, (int)Layout.Warp)]
    [InlineData(160, false, true, false, (int)Layout.Warp)]
    public void TheGateUpLayout_FollowsTheSlotCount(int slots, bool lanes, bool pair, bool warpPair, int expected)
        => Assert.Equal((Layout)expected, Dsv4Kernels.MoeGateUpQ80LayoutFor(slots, lanes, pair, warpPair));

    /// <summary>Device pointers to each expert's rows, as a residency table holds them, with expert e's entry at
    /// another expert's rows of the stacked tensor.</summary>
    private static byte[] PointerTable(IntPtr stacked, long expertBytes)
    {
        var bytes = new byte[Experts * 8];
        for (int e = 0; e < Experts; e++)
            BitConverter.TryWriteBytes(bytes.AsSpan(e * 8, 8), (ulong)((long)stacked + (e * 5 + 3) % Experts * expertBytes));
        return bytes;
    }

    private static void AssertWritten(float[] got, int count, string what)
    {
        for (int i = 0; i < count; i++)
            if (!float.IsFinite(got[i]))
                Assert.Fail($"{what}: element {i} is {got[i]}");
        for (int i = count; i < got.Length; i++)
            if (!float.IsNaN(got[i]))
                Assert.Fail($"{what}: wrote {got[i]:R} past its slots, at {i - count}");
    }

    private static void AssertSameBits(float[] expected, float[] got, int count, int rows, string what)
    {
        for (int i = 0; i < count; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(got[i]))
                Assert.Fail($"{what}: slot {i / rows} row {i % rows} is {got[i]:R}, a warp a row gives {expected[i]:R}");
        for (int i = count; i < got.Length; i++)
            if (!float.IsNaN(got[i]))
                Assert.Fail($"{what}: wrote {got[i]:R} past its slots, at {i - count}");
    }
}
