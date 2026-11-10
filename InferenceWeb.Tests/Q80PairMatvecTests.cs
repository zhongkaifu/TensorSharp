// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The Q8_0 pair matvec (<see cref="CudaKernels.LaunchQuantMatmulQ80Pair"/>): int8 tensor cores and two dp4a
/// kernels (four lanes on a block, a lane on a block), one arithmetic between them. Held at Qwen 3.8 Flash Next's decode
/// projections - GDN qkv and gate, attention q and k/v, their output projections, the hyper-connection mixers'
/// down and up (the up one 320 wide), the shared expert - and the Q8_0 decode experts the direct engine reads with
/// the dense matvec's layout (<see cref="Dsv4Kernels.MoeGateUpDecodeQ80Vec"/>).
/// </summary>
public sealed class Q80PairMatvecTests
{
    private const int Q8_0 = (int)GgmlTensorType.Q8_0;
    private const int MaxRows = CudaKernels.Q80PairMaxRows;

    private readonly ITestOutputHelper _output;

    public Q80PairMatvecTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Every row of a 2..16-row call equals the one-row call on that row, to the bit, so a batched decode step
    /// answers each sequence as its own step does whichever of the kernels runs it; and the rows are within
    /// 1e-4 relative L2 of the dequantized weights times the q8_1-rounded activations (what remains is
    /// accumulation order), so the pair cannot agree with itself by being wrong.
    /// </summary>
    [CudaTheory(TensorCoreModule = "tensorsharp_kernels")]
    [InlineData(2560, 10240)]
    [InlineData(2560, 6144)]
    [InlineData(6144, 2560)]
    [InlineData(2560, 12288)]
    [InlineData(2560, 512)]
    [InlineData(10240, 320)]
    [InlineData(320, 10240)]
    [InlineData(2560, 640)]
    [InlineData(640, 2560)]
    public void EveryRowOfAPairCall_IsTheOneRowCallBitForBit(int inDim, int outDim)
    {
        long rowBytes = inDim / 32 * 34;
        byte[] weights = QuantRows(Q8_0, outDim, rowBytes, seed: 23);
        var act = new float[MaxRows, inDim];
        for (int t = 0; t < MaxRows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0173f * (k + 1) * (t + 2)) + 0.3f * MathF.Cos(0.061f * k - 3 * t);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        IntPtr stream = allocator.Stream.Handle;
        using var w = Up(allocator, weights);
        using var input = Up(allocator, act);
        long q81Row = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
        using var xq = new Tensor(allocator, DType.UInt8, MaxRows * q81Row);
        CudaQuantizedOps.QuantizeRowsQ81(allocator, Dsv4CudaEngine.Ptr(input), Dsv4CudaEngine.Ptr(xq), inDim, MaxRows);
        using var single = new Tensor(allocator, DType.Float32, MaxRows, outDim);
        using var batched = new Tensor(allocator, DType.Float32, MaxRows, outDim);

        for (int r = 0; r < MaxRows; r++)
            allocator.Kernels.LaunchQuantMatmulQ80Pair(Dsv4CudaEngine.Ptr(w), Dsv4CudaEngine.Ptr(xq) + (nint)(r * q81Row),
                Dsv4CudaEngine.Ptr(single) + (nint)((long)r * outDim * 4), inDim, outDim, 1, stream);
        allocator.Stream.Synchronize();
        float[] alone = Down(single, MaxRows * outDim);

        float[] got = null;
        for (int rows = 2; rows <= MaxRows; rows++)
        {
            allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(batched), MaxRows * outDim, float.NaN, stream);
            allocator.Kernels.LaunchQuantMatmulQ80Pair(Dsv4CudaEngine.Ptr(w), Dsv4CudaEngine.Ptr(xq), Dsv4CudaEngine.Ptr(batched),
                inDim, outDim, rows, stream);
            allocator.Stream.Synchronize();
            got = Down(batched, MaxRows * outDim);
            for (int i = 0; i < rows * outDim; i++)
                if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(alone[i]))
                    Assert.Fail($"{inDim} -> {outDim}: row {i / outDim} of a {rows}-row call gives {got[i]:R} at column " +
                        $"{i % outDim}, the one-row call {alone[i]:R}");
        }

        float[,] reference = QuantizeDequantizeQ8_1(act);
        var wRow = new float[inDim];
        double num = 0, den = 0;
        for (int o = 0; o < outDim; o++)
        {
            NativeDequant.DequantizeToFloat32(Q8_0, weights, checked((int)(o * rowBytes)), wRow, 0, inDim);
            foreach (int t in new[] { 0, 1, MaxRows - 1 })
            {
                double want = 0;
                for (int k = 0; k < inDim; k++)
                    want += (double)wRow[k] * reference[t, k];
                double d = got[t * outDim + o] - want;
                num += d * d;
                den += want * want;
            }
        }
        double rel = Math.Sqrt(num / den);
        _output.WriteLine($"{inDim} -> {outDim} ({CudaKernels.Q80PairSlices(inDim, outDim)} K slices): rows of 2..{MaxRows}-row " +
            $"calls bit-equal to one-row calls; relative L2 {rel:E2} against the dequantized weights");
        Assert.True(rel <= 1e-4, $"{inDim} -> {outDim}: {rel:E2} from the dequantized weights");
    }

    /// <summary>
    /// Each of the pair's kernels gives the tensor cores' bits at every row count it takes (the twin 1-4 rows, the
    /// lane kernel 1-8), so <see cref="CudaKernels.Q80PairKernelFor"/> chooses for speed alone: a shape and row count
    /// it never sends to a kernel still holds that kernel to the same sums.
    /// </summary>
    [CudaTheory(TensorCoreModule = "tensorsharp_kernels")]
    [InlineData(2560, 10240)]
    [InlineData(2560, 6144)]
    [InlineData(6144, 2560)]
    [InlineData(2560, 12288)]
    [InlineData(2560, 512)]
    [InlineData(10240, 320)]
    [InlineData(320, 10240)]
    [InlineData(2560, 640)]
    [InlineData(640, 2560)]
    public void EachPairKernel_GivesTheTensorCoresBits(int inDim, int outDim)
    {
        const int Rows = 8;
        long rowBytes = inDim / 32 * 34;
        byte[] weights = QuantRows(Q8_0, outDim, rowBytes, seed: 29);
        var act = new float[Rows, inDim];
        for (int t = 0; t < Rows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0191f * (k + 3) * (t + 1)) + 0.25f * MathF.Cos(0.047f * k + 5 * t);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        IntPtr stream = allocator.Stream.Handle;
        using var w = Up(allocator, weights);
        using var input = Up(allocator, act);
        long q81Row = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
        using var xq = new Tensor(allocator, DType.UInt8, Rows * q81Row);
        CudaQuantizedOps.QuantizeRowsQ81(allocator, Dsv4CudaEngine.Ptr(input), Dsv4CudaEngine.Ptr(xq), inDim, Rows);
        using var want = new Tensor(allocator, DType.Float32, Rows, outDim);
        using var got = new Tensor(allocator, DType.Float32, Rows, outDim);

        int checkedCalls = 0;
        for (int rows = 1; rows <= Rows; rows++)
        {
            allocator.Kernels.LaunchQuantMatmulQ80PairKernel(CudaKernels.Q80PairKernel.Mma, Dsv4CudaEngine.Ptr(w),
                Dsv4CudaEngine.Ptr(xq), Dsv4CudaEngine.Ptr(want), inDim, outDim, rows, stream);
            allocator.Stream.Synchronize();
            float[] mma = Down(want, rows * outDim);
            foreach (var kernel in new[] { CudaKernels.Q80PairKernel.Twin, CudaKernels.Q80PairKernel.Lane })
            {
                if (kernel == CudaKernels.Q80PairKernel.Twin && rows > 4)
                    continue;
                allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(got), Rows * outDim, float.NaN, stream);
                allocator.Kernels.LaunchQuantMatmulQ80PairKernel(kernel, Dsv4CudaEngine.Ptr(w), Dsv4CudaEngine.Ptr(xq),
                    Dsv4CudaEngine.Ptr(got), inDim, outDim, rows, stream);
                allocator.Stream.Synchronize();
                float[] k = Down(got, Rows * outDim);
                for (int i = 0; i < rows * outDim; i++)
                    if (BitConverter.SingleToInt32Bits(k[i]) != BitConverter.SingleToInt32Bits(mma[i]))
                        Assert.Fail($"{inDim} -> {outDim}, {rows} rows: the {kernel} kernel gives {k[i]:R} at row " +
                            $"{i / outDim}, column {i % outDim}, the tensor cores {mma[i]:R}");
                for (int i = rows * outDim; i < Rows * outDim; i++)
                    if (!float.IsNaN(k[i]))
                        Assert.Fail($"{inDim} -> {outDim}, {rows} rows: the {kernel} kernel wrote row {i / outDim}");
                checkedCalls++;
            }
        }
        _output.WriteLine($"{inDim} -> {outDim}: twin (1-4 rows) and lane kernel (1-8) bit-equal to the tensor cores " +
            $"in {checkedCalls} calls");
    }

    /// <summary>The kernel each of the model's decode shapes gets, by rows (speed only; the bits are the same).</summary>
    [Theory]
    [InlineData(2560, 10240, 1, "Twin")]
    [InlineData(2560, 10240, 2, "Twin")]
    [InlineData(2560, 10240, 4, "Lane")]
    [InlineData(2560, 10240, 8, "Lane")]
    [InlineData(2560, 10240, 9, "Mma")]
    [InlineData(6144, 2560, 4, "Twin")]
    [InlineData(6144, 2560, 6, "Mma")]
    [InlineData(2560, 640, 2, "Lane")]
    [InlineData(2560, 640, 6, "Mma")]
    [InlineData(10240, 320, 2, "Twin")]
    [InlineData(10240, 320, 3, "Lane")]
    [InlineData(10240, 320, 8, "Lane")]
    [InlineData(2560, 248320, 3, "Lane")]
    [InlineData(2560, 248320, 6, "Mma")]
    [InlineData(2560, 6144, 16, "Mma")]
    public void ThePairsKernel_FollowsShapeAndRows(int inDim, int outDim, int rows, string kernel)
        => Assert.Equal(kernel, CudaKernels.Q80PairKernelFor(inDim, outDim, rows).ToString());

    /// <summary>A launch takes the kernel timed for its shape and rows; a shape that was never timed takes the
    /// heuristic and is named once, not silently.</summary>
    [Fact]
    public void ALaunch_TakesTheTimedKernel_AndAnUntimedShapeIsNamedOnce()
    {
        var timed = Enumerable.Repeat(CudaKernels.Q80PairKernel.Mma, MaxRows).ToArray();
        timed[0] = CudaKernels.Q80PairKernel.Lane;
        var table = new Dictionary<(int InDim, int OutDim), CudaKernels.Q80PairKernel[]> { [(320, 10240)] = timed };
        var warned = new System.Collections.Concurrent.ConcurrentDictionary<(int InDim, int OutDim), bool>();
        var log = new System.IO.StringWriter();

        for (int rows = 1; rows <= MaxRows; rows++)
            Assert.Equal(timed[rows - 1], CudaKernels.PickQ80PairKernel(table, 320, 10240, rows, warned, log));
        Assert.Equal("", log.ToString());

        for (int rows = 1; rows <= MaxRows; rows++)
            Assert.Equal(CudaKernels.Q80PairKernelFor(640, 2560, rows),
                CudaKernels.PickQ80PairKernel(table, 640, 2560, rows, warned, log));
        string said = log.ToString();
        Assert.Contains("640 -> 2560", said);
        Assert.Single(said.Split('\n', StringSplitOptions.RemoveEmptyEntries));
        Assert.Equal("lane 1, mma 2-16", CudaKernels.DescribeQ80PairPicks(timed));
    }

    /// <summary>A timed kernel replaces the heuristic's pick only when it is faster by more than the noise margin
    /// (3% of the pick's time or 0.10 us, whichever is larger): near-ties keep the heuristic's pick, so two runs that
    /// time the same kernels a hair apart still choose alike; a clear winner wins.</summary>
    [Theory]
    [InlineData(3.67, 3.64, 9.0, "Twin")]    // 640 -> 2560 at 4 rows on the RTX 5090: twin 3.67, lane 3.64 (a tie)
    [InlineData(4.47, 4.09, 4.76, "Lane")]   // 320 -> 10240 at 2 rows: the lane kernel by 0.38 us (8.5%)
    [InlineData(2.40, 2.33, 4.00, "Twin")]   // 0.07 us is under the 0.10 us floor though it is 2.9%
    [InlineData(20.0, 19.5, 30.0, "Twin")]   // 0.5 us is under 3% of 20 us
    [InlineData(20.0, 19.3, 30.0, "Lane")]   // 0.7 us is over both
    public void ANearTie_KeepsTheHeuristicsPick(double twinUs, double laneUs, double mmaUs, string want)
    {
        var micros = new Dictionary<CudaKernels.Q80PairKernel, double>
        {
            [CudaKernels.Q80PairKernel.Twin] = twinUs,
            [CudaKernels.Q80PairKernel.Lane] = laneUs,
            [CudaKernels.Q80PairKernel.Mma] = mmaUs,
        };
        Assert.Equal(want, CudaKernels.ChooseQ80PairKernel(CudaKernels.Q80PairKernel.Twin, micros).ToString());
        // A heuristic pick that is the fastest stays whatever the margin.
        Assert.Equal("Mma", CudaKernels.ChooseQ80PairKernel(CudaKernels.Q80PairKernel.Mma,
            new Dictionary<CudaKernels.Q80PairKernel, double> { [CudaKernels.Q80PairKernel.Mma] = 1.0, [CudaKernels.Q80PairKernel.Lane] = 1.5 }).ToString());
    }

    /// <summary>
    /// The kernels a tuning run picks, at Flash Next's pair shapes (the TS_Q4E_HC_Q8 up mixer, the ISTA checkpoint's
    /// Q8_0 shared-expert down) and the down mixer's: each row count's pick takes that many rows, and a launch through
    /// the timed table gives the tensor cores' bits, so a timing-based choice can never change an output.
    /// </summary>
    [CudaTheory(TensorCoreModule = "tensorsharp_kernels")]
    [InlineData(320, 10240)]
    [InlineData(640, 2560)]
    [InlineData(10240, 320)]
    public void TheTimedKernels_GiveTheTensorCoresBits(int inDim, int outDim)
    {
        long rowBytes = inDim / 32 * 34;
        var act = new float[MaxRows, inDim];
        for (int t = 0; t < MaxRows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.0211f * (k + 5) * (t + 1)) + 0.2f * MathF.Cos(0.033f * k + 7 * t);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        IntPtr stream = allocator.Stream.Handle;
        using var w = Up(allocator, QuantRows(Q8_0, outDim, rowBytes, seed: 31));
        using var input = Up(allocator, act);
        long q81Row = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
        using var xq = new Tensor(allocator, DType.UInt8, MaxRows * q81Row);
        CudaQuantizedOps.QuantizeRowsQ81(allocator, Dsv4CudaEngine.Ptr(input), Dsv4CudaEngine.Ptr(xq), inDim, MaxRows);
        using var want = new Tensor(allocator, DType.Float32, MaxRows, outDim);
        using var got = new Tensor(allocator, DType.Float32, MaxRows, outDim);

        string summary = allocator.Kernels.TuneQ80Pair(new[] { (Dsv4CudaEngine.Ptr(w), inDim, outDim) }, stream);
        _output.WriteLine(summary);
        Assert.Contains($"{inDim}x{outDim}", summary);
        for (int rows = 1; rows <= MaxRows; rows++)
        {
            var kernel = allocator.Kernels.Q80PairKernelOnDevice(inDim, outDim, rows);
            Assert.True(rows <= CudaKernels.Q80PairKernelMaxRows(kernel), $"{rows} rows timed onto the {kernel} kernel");
            allocator.Kernels.LaunchQuantMatmulQ80PairKernel(CudaKernels.Q80PairKernel.Mma, Dsv4CudaEngine.Ptr(w),
                Dsv4CudaEngine.Ptr(xq), Dsv4CudaEngine.Ptr(want), inDim, outDim, rows, stream);
            allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(got), MaxRows * outDim, float.NaN, stream);
            allocator.Kernels.LaunchQuantMatmulQ80Pair(Dsv4CudaEngine.Ptr(w), Dsv4CudaEngine.Ptr(xq), Dsv4CudaEngine.Ptr(got),
                inDim, outDim, rows, stream);
            allocator.Stream.Synchronize();
            float[] mma = Down(want, rows * outDim), timedOut = Down(got, rows * outDim);
            for (int i = 0; i < rows * outDim; i++)
                if (BitConverter.SingleToInt32Bits(timedOut[i]) != BitConverter.SingleToInt32Bits(mma[i]))
                    Assert.Fail($"{inDim} -> {outDim}, {rows} rows on the timed {kernel} kernel: {timedOut[i]:R} at row " +
                        $"{i / outDim}, column {i % outDim}; the tensor cores {mma[i]:R}");
        }
    }

    /// <summary>The K slices fix the arithmetic, so they depend on the weight's shape alone: never on the rows.</summary>
    [Theory]
    [InlineData(2560, 10240, 1)]
    [InlineData(2560, 12288, 1)]
    [InlineData(2560, 6144, 2)]
    [InlineData(640, 2560, 2)]
    [InlineData(6144, 2560, 4)]
    [InlineData(2560, 640, 8)]
    [InlineData(2560, 512, 8)]
    [InlineData(10240, 320, 16)]
    [InlineData(320, 10240, 1)]
    [InlineData(2560, 248320, 1)]
    public void ThePairsSlices_FollowTheShape(int inDim, int outDim, int slices)
        => Assert.Equal(slices, CudaKernels.Q80PairSlices(inDim, outDim));

    /// <summary>
    /// The direct engine's Q8_0 decode experts with the dense matvec's row layout: a token's expert output is the
    /// same to the bit however many tokens share the launch (the 16-token launch's 64 slots take the gate+up pair
    /// kernel, the 1-7-token ones the separate rows), and within 1e-4 relative L2 of the shared warp-per-block kernel
    /// (the two sum a row in different orders).
    /// </summary>
    [CudaFact]
    public void Q80VecDecodeExperts_AnswerEachTokenAlike_AtEveryBatchSize()
    {
        const int cap = 16, experts = 16, used = 4, e = 2560, ff = 640;
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        using Tensor gate = Up(allocator, Q80Weights(experts * ff, e, 1));
        using Tensor up = Up(allocator, Q80Weights(experts * ff, e, 2));
        using Tensor down = Up(allocator, Q80Weights(experts * e, ff, 3));
        var g = new DeviceWeight { Ptr = Dsv4CudaEngine.Ptr(gate), Type = Q8_0, Ne0 = e, Ne1 = ff, RowBytes = e / 32 * 34 };
        var u = new DeviceWeight { Ptr = Dsv4CudaEngine.Ptr(up), Type = Q8_0, Ne0 = e, Ne1 = ff, RowBytes = e / 32 * 34 };
        var d = new DeviceWeight { Ptr = Dsv4CudaEngine.Ptr(down), Type = Q8_0, Ne0 = ff, Ne1 = e, RowBytes = ff / 32 * 34 };
        using Tensor cur = Up(allocator, Enumerable.Range(0, cap * e).Select(i => MathF.Sin(i * .237f)).ToArray());
        using Tensor sharedDown = Up(allocator, Enumerable.Repeat(.125f, cap * e).ToArray());
        using var output = new Tensor(allocator, DType.Float32, cap, e);
        var owned = new List<Tensor>();
        Tensor Alloc(DType type, long[] sizes) { var t = new Tensor(allocator, type, sizes); owned.Add(t); return t; }
        try
        {
            var scratch = new CudaMoeScratch(Alloc, cap, experts, used, e, ff, false);
            scratch.Sel.SetElementsAsInt(Enumerable.Range(0, cap * used).Select(i => (i / used * 3 + i % used * 5) % experts).ToArray());
            scratch.SelW.SetElementsAsFloat(Enumerable.Repeat(1f / used, cap * used).ToArray());
            scratch.Sel.Storage.EnsureDeviceCurrent();
            scratch.SelW.Storage.EnsureDeviceCurrent();

            float[] Run(int nt, bool vec)
            {
                allocator.Kernels.LaunchFillF32(Dsv4CudaEngine.Ptr(output), cap * e, float.NaN, allocator.Stream.Handle);
                CudaMoe.Experts(dk, allocator.Kernels, scratch, g, u, d, cur, sharedDown, output,
                    nt, used, experts, e, ff, 0f, cap, allocator.Stream.Handle, q80VecDecode: vec);
                allocator.Stream.Synchronize();
                return Down(output, nt * e);
            }

            float[] all = Run(cap, vec: true);
            Assert.All(all, v => Assert.True(float.IsFinite(v)));
            foreach (int nt in new[] { 1, 2, 3, 7, 15 })
            {
                float[] some = Run(nt, vec: true);
                for (int i = 0; i < nt * e; i++)
                    if (BitConverter.SingleToInt32Bits(some[i]) != BitConverter.SingleToInt32Bits(all[i]))
                        Assert.Fail($"token {i / e} of a {nt}-token launch gives {some[i]:R} at {i % e}, of a {cap}-token one {all[i]:R}");
            }
            float[] shared = Run(cap, vec: false);
            double num = 0, den = 0;
            for (int i = 0; i < cap * e; i++)
            {
                double diff = all[i] - shared[i];
                num += diff * diff;
                den += (double)shared[i] * shared[i];
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"Q8_0 decode experts, row layout of the dense matvec: tokens bit-equal at every batch size; " +
                $"relative L2 {rel:E2} from the shared kernel");
            Assert.True(rel <= 1e-4, $"{rel:E2} from the shared decode experts");
        }
        finally { allocator.Synchronize(); foreach (Tensor t in owned) t.Dispose(); }
    }

    private static byte[] Q80Weights(int rows, int width, int seed)
    {
        var rng = new Random(seed);
        var result = new byte[rows * width / 32 * 34];
        for (int i = 0; i < result.Length; i += 34)
        {
            BitConverter.TryWriteBytes(result.AsSpan(i, 2), BitConverter.HalfToUInt16Bits((System.Half).002f));
            for (int j = 2; j < 34; j++) result[i + j] = unchecked((byte)rng.Next(-127, 128));
        }
        return result;
    }
}
