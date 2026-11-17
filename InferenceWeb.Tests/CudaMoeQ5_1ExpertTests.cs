// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// Q5_1 routed experts on the direct-CUDA expert kernels (<see cref="CudaMoe"/>: Qwen3.8-Flash-Next
/// and GLM-5.3-Flash on <c>--backend cuda</c>). Unsloth's UD-Q4_K_XL Qwen3.8-Flash-Next keeps 43 of its
/// 48 down projections in Q5_1, and the engine refused the file at load. Every route an expert matrix
/// can take is held to ggml's own dequantization of the same bytes: the per-token decode kernels, the
/// grouped warp kernels, the register-staged kernels and (in <see cref="Dsv4ExpertKernelTests"/>) the
/// tensor-core kernel, each next to formats the kernels already took, at the same tolerance. Then the
/// whole expert block on each route, and the refusal of a layout no route decodes.
/// </summary>
public sealed class CudaMoeQ5_1ExpertTests
{
    private const int Q5_1 = (int)GgmlTensorType.Q5_1;
    private const int Experts = 5;
    private const int Tokens = 40;
    private const int Used = 2;

    private readonly ITestOutputHelper _output;

    public CudaMoeQ5_1ExpertTests(ITestOutputHelper output) => _output = output;

    // ---------------------------------------------------------------- the reference

    /// <summary>ggml's <c>dequantize_row_q5_1</c> (ggml-quants.c) written out: per 32 values an F16
    /// scale d and offset m, 32 fifth bits, then 16 bytes whose low nibbles are values 0-15 and high
    /// nibbles values 16-31; value = code * d + m. <paramref name="fused"/> rounds code * d + m once,
    /// as a compiler that contracts the expression to a fused multiply-add (clang on arm64) builds it.</summary>
    internal static void DequantizeQ5_1(byte[] src, long offset, float[] dst, int n, bool fused = false)
    {
        for (int b = 0; b < n / 32; b++)
        {
            int at = checked((int)(offset + b * 24L));
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(src[at] | src[at + 1] << 8));
            float m = (float)BitConverter.UInt16BitsToHalf((ushort)(src[at + 2] | src[at + 3] << 8));
            uint qh = BitConverter.ToUInt32(src, at + 4);
            for (int j = 0; j < 16; j++)
            {
                int x0 = (src[at + 8 + j] & 0x0F) | (int)(((qh >> j) << 4) & 0x10);
                int x1 = (src[at + 8 + j] >> 4) | (int)((qh >> (j + 12)) & 0x10);
                dst[b * 32 + j] = fused ? MathF.FusedMultiplyAdd(x0, d, m) : x0 * d + m;
                dst[b * 32 + j + 16] = fused ? MathF.FusedMultiplyAdd(x1, d, m) : x1 * d + m;
            }
        }
    }

    /// <summary>Asserts that ggml's dequantization of one Q5_1 row (<paramref name="ggml"/>) is
    /// <see cref="DequantizeQ5_1"/>'s bit for bit, value by value (either rounding of code * d + m).</summary>
    private static void AssertIsGgmlsQ5_1(byte[] bytes, long offset, float[] ggml, int n)
    {
        var plain = new float[n];
        var fused = new float[n];
        DequantizeQ5_1(bytes, offset, plain, n);
        DequantizeQ5_1(bytes, offset, fused, n, fused: true);
        for (int k = 0; k < n; k++)
        {
            int bits = BitConverter.SingleToInt32Bits(ggml[k]);
            if (bits != BitConverter.SingleToInt32Bits(plain[k]) && bits != BitConverter.SingleToInt32Bits(fused[k]))
                Assert.Fail($"Q5_1 value {k}: ggml {ggml[k]} vs reference {plain[k]}");
        }
    }

    /// <summary>The reference the kernel tests use (ggml through GgmlOps) is ggml's Q5_1 bit for bit:
    /// every code, both fifth-bit halves, offsets on both sides of zero.</summary>
    [Fact]
    public void TheReferenceDequantization_IsGgmlsQ5_1()
    {
        const int n = 4096;
        long rowBytes = NativeDequant.RowSize(Q5_1, n);
        Assert.Equal(n / 32 * 24, rowBytes);
        byte[] bytes = QuantRows(Q5_1, 3, rowBytes, seed: 11);
        var mine = new float[n];
        var ggml = new float[n];
        var codes = new HashSet<int>();
        for (int r = 0; r < 3; r++)
        {
            GgmlGgufTensorDequant.DequantizeToFloat32(Q5_1, bytes, (int)(r * rowBytes), ggml, 0, n);
            AssertIsGgmlsQ5_1(bytes, r * rowBytes, ggml, n);
            DequantizeQ5_1(bytes, r * rowBytes, mine, n);
            for (int b = 0; b < n / 32; b++)
            {
                int at = (int)(r * rowBytes) + b * 24;
                float d = (float)BitConverter.UInt16BitsToHalf((ushort)(bytes[at] | bytes[at + 1] << 8));
                float m = (float)BitConverter.UInt16BitsToHalf((ushort)(bytes[at + 2] | bytes[at + 3] << 8));
                for (int j = 0; j < 32; j++)
                    codes.Add((int)MathF.Round((mine[b * 32 + j] - m) / d));
            }
        }
        Assert.Equal(Enumerable.Range(0, 32), codes.OrderBy(c => c));
        Assert.Contains(mine, v => v < 0);
        Assert.Contains(mine, v => v > 0);
    }

    // ---------------------------------------------------------------- kernel routes

    /// <summary>
    /// The grouped expert kernels a long ubatch runs, gate/up (each member slot reads its token's row
    /// through slotToken) and down (activations already in slot order): the warp-per-row kernels over
    /// interleaved q8_1 activations, and the register-staged kernels over the split q8_1 layout where
    /// they take the width. One expert has no members and one more than a 32-token pass; the row
    /// counts end mid-tile. Held, as the decode kernels are, to the dequantized weights against each
    /// row's q8_1 rounding.
    /// </summary>
    [CudaTheory]
    [InlineData(Q5_1, 512, 160)]
    [InlineData(Q5_1, 1280, 100)]    // staged: some lanes stage a second sub-block
    [InlineData(Q5_1, 96, 45)]       // warp kernels only: not whole 256-value blocks
    [InlineData((int)GgmlTensorType.IQ4_NL, 512, 160)]
    [InlineData((int)GgmlTensorType.Q5_K, 1280, 100)]
    [InlineData((int)GgmlTensorType.Q8_0, 96, 45)]
    public void TheGroupedExpertProjections_MatchTheDequantizedWeights(int ggmlType, int inDim, int rows)
    {
        long rowBytes = NativeDequant.RowSize(ggmlType, inDim);
        byte[] gate = QuantRows(ggmlType, Experts * rows, rowBytes, seed: 5);
        byte[] up = QuantRows(ggmlType, Experts * rows, rowBytes, seed: 6);
        var act = new float[Tokens, inDim];
        var slotAct = new float[Tokens * Used, inDim];
        for (int t = 0; t < Tokens; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.013f * (k + 1) * (t + 3)) + 0.25f * MathF.Cos(0.071f * k - t);
        for (int s = 0; s < Tokens * Used; s++)
            for (int k = 0; k < inDim; k++)
                slotAct[s, k] = 0.5f * MathF.Cos(0.021f * (k + 7) * (s + 1)) - 0.1f * MathF.Sin(0.3f * s);
        (int[] counts, int[] offsets, int[] slotToken) = Group();

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        IntPtr stream = allocator.Stream.Handle;
        using var gateW = Up(allocator, gate);
        using var upW = Up(allocator, up);
        using var actT = Up(allocator, act);
        using var slotActT = Up(allocator, slotAct);
        using var countsT = Up(allocator, counts);
        using var offsetsT = Up(allocator, offsets);
        using var slotTokenT = Up(allocator, slotToken);
        int slots = Tokens * Used;
        using var actQ = new Tensor(allocator, DType.UInt8, (long)Tokens * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var slotQ = new Tensor(allocator, DType.UInt8, (long)slots * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var actQs = new Tensor(allocator, DType.UInt8, (long)Tokens * inDim);
        using var actD = new Tensor(allocator, DType.Float32, (long)Tokens * (inDim / 32));
        using var slotQs = new Tensor(allocator, DType.UInt8, (long)slots * inDim);
        using var slotD = new Tensor(allocator, DType.Float32, (long)slots * (inDim / 32));
        using var gateOut = new Tensor(allocator, DType.Float32, slots, rows);
        using var upOut = new Tensor(allocator, DType.Float32, slots, rows);
        using var downOut = new Tensor(allocator, DType.Float32, slots, rows);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(actT), Ptr(actQ), inDim, Tokens, stream, warpCooperative: true);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(slotActT), Ptr(slotQ), inDim, slots, stream, warpCooperative: true);
        allocator.Kernels.LaunchQuantizeQ81SplitRows(Ptr(actT), Ptr(actQs), Ptr(actD), inDim, Tokens, stream);
        allocator.Kernels.LaunchQuantizeQ81SplitRows(Ptr(slotActT), Ptr(slotQs), Ptr(slotD), inDim, slots, stream);

        float[,] rounded = QuantizeDequantizeQ8_1(act);
        float[,] slotRounded = QuantizeDequantizeQ8_1(slotAct);
        float[] tokenRow(int s) => Row(rounded, slotToken[s]);
        float[] slotRow(int s) => Row(slotRounded, s);

        var routes = new List<string> { "warp" };
        if (Dsv4Kernels.StagedSupports(inDim))
            routes.Add("staged");
        foreach (string route in routes)
        {
            Clear(allocator, gateOut, upOut, downOut);
            if (route == "warp")
            {
                dk.MoeGateUp(Ptr(gateW), Ptr(upW), actQ, countsT, offsetsT, slotTokenT, gateOut, upOut,
                    ggmlType, rows, inDim, rowBytes, Experts, stream);
                dk.MoeDown(Ptr(gateW), slotQ, countsT, offsetsT, downOut, ggmlType, rows, inDim, rowBytes, Experts, stream);
            }
            else
            {
                dk.MoeGateUpStaged(Ptr(gateW), Ptr(upW), actQs, actD, countsT, offsetsT, slotTokenT, gateOut, upOut,
                    ggmlType, rows, inDim, rowBytes, Experts, stream);
                dk.MoeDownStaged(Ptr(gateW), slotQs, slotD, countsT, offsetsT, downOut, ggmlType, rows, inDim, rowBytes,
                    Experts, stream);
            }
            allocator.Stream.Synchronize();
            double worst = 0;
            worst = Math.Max(worst, AssertGrouped($"{route} gate", gate, ggmlType, rowBytes, rows, inDim, counts, offsets,
                tokenRow, Down(gateOut, slots * rows)));
            worst = Math.Max(worst, AssertGrouped($"{route} up", up, ggmlType, rowBytes, rows, inDim, counts, offsets,
                tokenRow, Down(upOut, slots * rows)));
            worst = Math.Max(worst, AssertGrouped($"{route} down", gate, ggmlType, rowBytes, rows, inDim, counts, offsets,
                slotRow, Down(downOut, slots * rows)));
            _output.WriteLine($"{(GgmlTensorType)ggmlType} {route} {inDim}x{rows}: worst error {worst:E2} of |w.a|");
        }
    }

    /// <summary>The per-token down projection a decode step runs: every (token, selected expert) slot
    /// reads its own q8_1 row (gate/up's twin is <see cref="Dsv4ExpertKernelTests"/>'s decode test).
    /// A width off the 256 grid is Q5_1's alone among the down layouts here.</summary>
    [CudaTheory]
    [InlineData(Q5_1, 512, 160)]
    [InlineData(Q5_1, 160, 77)]
    [InlineData((int)GgmlTensorType.IQ4_NL, 512, 160)]
    [InlineData((int)GgmlTensorType.Q8_0, 160, 77)]
    public void TheDecodeDownProjection_ReadsEachSlotsOwnRow(int ggmlType, int ff, int rows)
    {
        int[] sel = { 4, 0, 2, 4, 1, 3 };   // three tokens' two experts; expert 4 serves two
        long rowBytes = NativeDequant.RowSize(ggmlType, ff);
        byte[] down = QuantRows(ggmlType, Experts * rows, rowBytes, seed: 7);
        var act = new float[sel.Length, ff];
        for (int s = 0; s < sel.Length; s++)
            for (int k = 0; k < ff; k++)
                act[s, k] = MathF.Sin(0.019f * (k + 3) * (s + 2)) * (1 + 0.5f * MathF.Cos(0.11f * k));

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        using var downW = Up(allocator, down);
        using var actT = Up(allocator, act);
        using var selT = Up(allocator, sel);
        using var actQ = new Tensor(allocator, DType.UInt8, (long)sel.Length * (ff / 32) * CudaKernels.Q81BlockBytes);
        using var output = new Tensor(allocator, DType.Float32, sel.Length, rows);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(actT), Ptr(actQ), ff, sel.Length, allocator.Stream.Handle, warpCooperative: true);
        dk.MoeDownDecode(Ptr(downW), actQ, selT, output, ggmlType, rows, ff, rowBytes, sel.Length, allocator.Stream.Handle);
        allocator.Stream.Synchronize();

        float[,] rounded = QuantizeDequantizeQ8_1(act);
        float[] got = Down(output, sel.Length * rows);
        var w = new float[ff];
        double worst = 0;
        for (int s = 0; s < sel.Length; s++)
        {
            float[] a = Row(rounded, s);
            for (int r = 0; r < rows; r++)
            {
                Dequantize(ggmlType, down, ((long)sel[s] * rows + r) * rowBytes, w, ff);
                worst = Math.Max(worst, AssertDot($"slot {s} (expert {sel[s]}) row {r}", w, a, got[(long)s * rows + r]));
            }
        }
        _output.WriteLine($"{(GgmlTensorType)ggmlType} decode down {ff}x{rows}: worst error {worst:E2} of |w.a|");
    }

    // ---------------------------------------------------------------- the expert block, route by route

    /// <summary>
    /// <see cref="CudaMoe.Experts"/> as Qwen3.8-Flash-Next's UD-Q4_K_XL runs it - Q4_K gate/up over a
    /// Q5_1 down projection - against the block computed in double from ggml's dequantized weights and
    /// unquantized activations: a decode-size step on the per-slot kernels, then a ubatch past the
    /// per-slot bound on the tensor cores (both widths on the 128 grid) or on the grouped warp kernels
    /// (an expert width off it). IQ4_NL down, the format the engine already ran, is the control: both
    /// are held to the same bound, which the q8_1 and F16 activation rounding sets.
    /// </summary>
    [CudaTheory]
    [InlineData(Q5_1, 256)]
    [InlineData(Q5_1, 96)]
    [InlineData((int)GgmlTensorType.IQ4_NL, 256)]
    [InlineData((int)GgmlTensorType.IQ4_NL, 96)]
    public void TheExpertBlock_OnEveryRoute_MatchesTheFloatBlock(int downType, int ff)
    {
        const int e = 256, experts = 6, used = 2, cap = 40, perSlotMaxRows = 16;
        const int guType = (int)GgmlTensorType.Q4_K;
        long guRowBytes = NativeDequant.RowSize(guType, e), downRowBytes = NativeDequant.RowSize(downType, ff);
        byte[] gate = QuantRows(guType, experts * ff, guRowBytes, seed: 21);
        byte[] up = QuantRows(guType, experts * ff, guRowBytes, seed: 22);
        byte[] down = QuantRows(downType, experts * e, downRowBytes, seed: 23);
        float[] cur = Enumerable.Range(0, cap * e).Select(i => 0.05f * MathF.Sin(i * 0.237f) + 0.02f * MathF.Cos(i * 0.0113f)).ToArray();
        float[] shared = Enumerable.Range(0, cap * e).Select(i => 0.01f * MathF.Cos(i * 0.71f)).ToArray();
        int[] sel = Enumerable.Range(0, cap * used).Select(i => (i * 5 + i / used) % experts).ToArray();
        float[] selW = Enumerable.Range(0, cap * used).Select(i => 0.3f + 0.4f * ((i * 7) % 5) / 4f).ToArray();
        for (int t = 0; t < cap; t++)
            Assert.NotEqual(sel[t * used], sel[t * used + 1]);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        using var gateT = Up(allocator, gate);
        using var upT = Up(allocator, up);
        using var downT = Up(allocator, down);
        var g = new DeviceWeight { Ptr = Ptr(gateT), Type = guType, Ne0 = e, Ne1 = ff, RowBytes = guRowBytes };
        var u = new DeviceWeight { Ptr = Ptr(upT), Type = guType, Ne0 = e, Ne1 = ff, RowBytes = guRowBytes };
        var d = new DeviceWeight { Ptr = Ptr(downT), Type = downType, Ne0 = ff, Ne1 = e, RowBytes = downRowBytes };
        using var curT = Up(allocator, cur);
        using var sharedT = Up(allocator, shared);
        using var output = new Tensor(allocator, DType.Float32, cap, e);
        var owned = new List<Tensor>();
        Tensor Alloc(DType type, long[] sizes) { var t = new Tensor(allocator, type, sizes); owned.Add(t); return t; }
        try
        {
            var scratch = new CudaMoeScratch(Alloc, cap, experts, used, e, ff);
            scratch.Sel.SetElementsAsInt(sel);
            scratch.SelW.SetElementsAsFloat(selW);
            scratch.Sel.Storage.EnsureDeviceCurrent();
            scratch.SelW.Storage.EnsureDeviceCurrent();
            double[,] want = FloatBlock(gate, up, down, guType, downType, guRowBytes, downRowBytes, cur, shared, sel, selW,
                cap, used, e, ff);
            foreach (int nt in new[] { 3, cap })
            {
                CudaMoe.Experts(dk, allocator.Kernels, scratch, g, u, d, curT, sharedT, output,
                    nt, used, experts, e, ff, 0f, perSlotMaxRows, allocator.Stream.Handle);
                allocator.Stream.Synchronize();
                float[] got = Down(output, nt * e);
                double num = 0, den = 0;
                for (int t = 0; t < nt; t++)
                {
                    for (int c = 0; c < e; c++)
                    {
                        double diff = got[t * e + c] - want[t, c];
                        num += diff * diff;
                        den += want[t, c] * want[t, c];
                    }
                }
                double rel = Math.Sqrt(num / den);
                // The route the module took (on one without the tensor-core body, the staged or warp kernels).
                var route = CudaMoe.SelectRoute(nt, perSlotMaxRows, dk.HasMoeMma, dk.DecodesLowBitIqExperts, guType, downType, e, ff);
                _output.WriteLine($"{(GgmlTensorType)downType} down, ff {ff}, {nt} tokens ({route}): relative error {rel:E2}");
                Assert.All(got, v => Assert.True(float.IsFinite(v)));
                Assert.True(rel <= 1e-2, $"{route}: the expert block strayed {rel:E2} from the float block");
            }
        }
        finally
        {
            allocator.Synchronize();
            foreach (Tensor t in owned)
                t.Dispose();
        }
    }

    /// <summary>out[t] = shared[t] + sum_j w[t, j] * down_j(silu(gate_j x) * up_j x), in double.</summary>
    private static double[,] FloatBlock(byte[] gate, byte[] up, byte[] down, int guType, int downType,
        long guRowBytes, long downRowBytes, float[] cur, float[] shared, int[] sel, float[] selW,
        int nt, int used, int e, int ff)
    {
        var result = new double[nt, e];
        var w = new float[Math.Max(e, ff)];
        var h = new double[ff];
        for (int t = 0; t < nt; t++)
        {
            for (int c = 0; c < e; c++)
                result[t, c] = shared[t * e + c];
            for (int j = 0; j < used; j++)
            {
                int x = sel[t * used + j];
                for (int r = 0; r < ff; r++)
                {
                    double gv = 0, uv = 0;
                    Dequantize(guType, gate, ((long)x * ff + r) * guRowBytes, w, e);
                    for (int k = 0; k < e; k++) gv += (double)w[k] * cur[t * e + k];
                    Dequantize(guType, up, ((long)x * ff + r) * guRowBytes, w, e);
                    for (int k = 0; k < e; k++) uv += (double)w[k] * cur[t * e + k];
                    h[r] = gv / (1 + Math.Exp(-gv)) * uv;
                }
                for (int c = 0; c < e; c++)
                {
                    Dequantize(downType, down, ((long)x * e + c) * downRowBytes, w, ff);
                    double y = 0;
                    for (int k = 0; k < ff; k++) y += w[k] * h[k];
                    result[t, c] += selW[t * used + j] * y;
                }
            }
        }
        return result;
    }

    // ---------------------------------------------------------------- refusals

    /// <summary>The refusal names the tensor, its layout and every layout the kernels take; that list
    /// is exactly the set some route decodes; and Q5_1 is in it.</summary>
    [Fact]
    public void AnExpertLayoutNoRouteDecodes_IsRefusedByTensorName()
    {
        CudaWeightDesc Desc(int type, string name) => new CudaWeightDesc { GgmlType = type, Name = name };
        var gate = Desc((int)GgmlTensorType.Q4_K, "blk.3.ffn_gate_exps.weight");
        var up = Desc((int)GgmlTensorType.Q4_K, "blk.3.ffn_up_exps.weight");
        CudaMoe.RequireExpertWeights("q4e-cuda", 3, gate, up, Desc(Q5_1, "blk.3.ffn_down_exps.weight"));

        var ex = Assert.Throws<NotSupportedException>(() =>
            CudaMoe.RequireExpertWeights("q4e-cuda", 3, gate, up, Desc((int)GgmlTensorType.Q4_0, "blk.3.ffn_down_exps.weight")));
        Assert.StartsWith("[q4e-cuda] blk.3.ffn_down_exps.weight is Q4_0 (ggml type 2)", ex.Message);
        Assert.Contains(CudaMoe.SupportedExpertTypes, ex.Message);
        Assert.Contains("--backend ggml_cuda", ex.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out _));

        // A descriptor without a name still says which tensor.
        ex = Assert.Throws<NotSupportedException>(() =>
            CudaMoe.RequireExpertWeights("glm-cuda", 7, Desc((int)GgmlTensorType.Q5_0, null), up, up));
        Assert.StartsWith("[glm-cuda] blk.7.ffn_gate_exps.weight is Q5_0", ex.Message);

        ex = Assert.Throws<NotSupportedException>(() =>
            CudaMoe.RequireExpertWeights("q4e-cuda", 3, gate, Desc((int)GgmlTensorType.Q5_K, "blk.3.ffn_up_exps.weight"), gate));
        Assert.Contains("blk.3.ffn_gate_exps.weight is Q4_K but blk.3.ffn_up_exps.weight is Q5_K", ex.Message);

        var listed = CudaMoe.SupportedExpertTypes.Split(", ").ToHashSet();
        var decoded = Enumerable.Range(0, 64).Where(CudaMoe.SupportsExpertType).Select(CudaMoe.GgmlTypeName).ToHashSet();
        Assert.Equal(listed.OrderBy(s => s), decoded.OrderBy(s => s));
        Assert.Contains("Q5_1", listed);
        Assert.True(CudaMoe.GroupedSupportsType(Q5_1));
    }

    /// <summary>A kernel module built from source older than the Q5_1 decoder (the committed PTX
    /// baseline before its refresh, which a machine without nvcc loads) would read Q5_1 rows as Q8_0:
    /// such a module refuses Q5_1 experts, by tensor name, and takes every other layout as before.</summary>
    [Fact]
    public void Q5_1Experts_AreRefusedOnAModuleThatPredatesTheirDecoder()
    {
        CudaWeightDesc Desc(int type, string name) => new CudaWeightDesc { GgmlType = type, Name = name };
        var gate = Desc((int)GgmlTensorType.Q4_K, "blk.5.ffn_gate_exps.weight");
        var up = Desc((int)GgmlTensorType.Q4_K, "blk.5.ffn_up_exps.weight");
        var q51Down = Desc(Q5_1, "blk.5.ffn_down_exps.weight");
        var q80Down = Desc((int)GgmlTensorType.Q8_0, "blk.5.ffn_down_exps.weight");

        CudaMoe.RequireExpertDecoders("q4e-cuda", decodesQ5_1: true, 5, gate, up, q51Down);
        CudaMoe.RequireExpertDecoders("q4e-cuda", decodesQ5_1: false, 5, gate, up, q80Down);
        var ex = Assert.Throws<NotSupportedException>(() =>
            CudaMoe.RequireExpertDecoders("q4e-cuda", decodesQ5_1: false, 5, gate, up, q51Down));
        Assert.StartsWith("[q4e-cuda] blk.5.ffn_down_exps.weight is Q5_1, and the loaded tensorsharp_dsv4_kernels module", ex.Message);
        Assert.Contains("nvcc", ex.Message);
        Assert.Contains("--backend ggml_cuda", ex.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out _));
    }

    /// <summary>The module this build loads carries the Q5_1 decoder's marker, so the engines take
    /// Q5_1 experts on it (a stale module would fail every kernel test above as well).</summary>
    [CudaFact]
    public void TheLoadedKernelModule_DecodesQ5_1Experts()
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.DecodesQ5_1Experts);
    }

    /// <summary>
    /// The GLM direct-CUDA engine refuses such a layer from the loader's descriptors, before it creates a
    /// device context: on a machine with no GPU at all the constructor still answers with the refusal.
    /// (In the per-device upload, two devices refusing at once was an AggregateException that aborted
    /// <c>--layer-split 2</c>.)
    /// </summary>
    [Fact]
    public void TheEngine_RefusesTheLayerBeforeTouchingADevice()
    {
        CudaWeightDesc Desc(int type, string name) => new CudaWeightDesc
        {
            GgmlType = type, Name = name, Ne0 = 256, Ne1 = 128, Ne2 = 4, RowBytes = 256, Source = new NoBytes(),
        };
        var glm = new GlmCudaEngine.ModelDesc
        {
            NLayer = 2, NEmbd = 256, KdaHeadDim = 128, KvLoraRank = 512, NExpert = 4, NExpertUsed = 2,
            Layers = new[] { 0, 1 }.Select(il => new GlmCudaEngine.LayerDesc
            {
                Moe = il == 1,
                GateExps = Desc((int)GgmlTensorType.IQ4_XS, $"blk.{il}.ffn_gate_exps.weight"),
                UpExps = Desc((int)GgmlTensorType.IQ4_XS, $"blk.{il}.ffn_up_exps.weight"),
                DownExps = Desc((int)GgmlTensorType.Q4_0, $"blk.{il}.ffn_down_exps.weight"),
            }).ToArray(),
        };
        var ex = Assert.Throws<NotSupportedException>(() => new GlmCudaEngine(glm, 2));
        Assert.StartsWith("[glm-cuda] blk.1.ffn_down_exps.weight is Q4_0", ex.Message);
    }

    /// <summary>DeepSeek V4 takes the expert list GLM and Qwen3.8 share, Q5_1 included: its resident layers
    /// are refused by name before the upload when no kernel decodes them (IQ1_M here) or when the loaded
    /// module predates the Q5_1 decoder, and offloaded layers are left to the host matmul.</summary>
    [Fact]
    public void Dsv4_RefusesResidentExpertsByName_AndLeavesOffloadedOnesToTheHost()
    {
        CudaWeightDesc Desc(int type, string name) => new CudaWeightDesc { GgmlType = type, Name = name, Source = new NoBytes() };
        var m = new Dsv4CudaEngine.ModelDesc
        {
            NLayer = 3,
            Layers = new[] { 0, 1, 2 }.Select(il => new Dsv4CudaEngine.LayerDesc
            {
                GateExps = Desc((int)GgmlTensorType.IQ3_S, $"blk.{il}.ffn_gate_exps.weight"),
                UpExps = Desc((int)GgmlTensorType.IQ3_S, $"blk.{il}.ffn_up_exps.weight"),
                DownExps = Desc(il == 0 ? (int)GgmlTensorType.IQ1_M : (int)GgmlTensorType.Q5_1, $"blk.{il}.ffn_down_exps.weight"),
            }).ToArray(),
        };
        var ex = Assert.Throws<NotSupportedException>(() =>
            Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 0, decodesQ5_1: true, decodesLowBitIq: true));
        Assert.Contains("blk.0.ffn_down_exps.weight is IQ1_M (GPU-resident)", ex.Message);
        Assert.Contains($"The GPU expert kernels implement {CudaMoe.SupportedExpertTypes};", ex.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out _));
        // Layer 0's experts stay on the host (with no host matmul here, the engine refuses that offload itself).
        Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 1, decodesQ5_1: true, decodesLowBitIq: true);

        // On a module built before the Q5_1 decoder, the first resident Q5_1 layer is refused by name.
        ex = Assert.Throws<NotSupportedException>(() =>
            Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 1, decodesQ5_1: false, decodesLowBitIq: true));
        Assert.StartsWith("[dsv4-cuda] blk.1.ffn_down_exps.weight is Q5_1, and the loaded tensorsharp_dsv4_kernels module",
            ex.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out _));

        m.Layers[2].UpExps = Desc((int)GgmlTensorType.Q6_K, "blk.2.ffn_up_exps.weight");
        ex = Assert.Throws<NotSupportedException>(() =>
            Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 1, decodesQ5_1: true, decodesLowBitIq: true));
        Assert.Contains("blk.2.ffn_gate_exps.weight (IQ3_S) and blk.2.ffn_up_exps.weight (Q6_K) differ", ex.Message);
    }

    private sealed class NoBytes : ICudaWeightSource
    {
        public void Read(long offset, IntPtr dst, long bytes) => throw new InvalidOperationException("nothing may be read");
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>Every token takes expert 1 (40 members: two passes of 32) and one of 0, 2, 4; expert 3 is
    /// never chosen. Returns the per-expert counts, their exclusive scan and each slot row's token.</summary>
    private static (int[] Counts, int[] Offsets, int[] SlotToken) Group()
    {
        var sel = new int[Tokens * Used];
        for (int t = 0; t < Tokens; t++)
        {
            sel[t * Used] = 1;
            sel[t * Used + 1] = new[] { 0, 2, 4 }[t % 3];
        }
        var counts = new int[Experts];
        foreach (int x in sel)
            counts[x]++;
        var offsets = new int[Experts];
        for (int x = 1; x < Experts; x++)
            offsets[x] = offsets[x - 1] + counts[x - 1];
        var cursor = (int[])offsets.Clone();
        var slotToken = new int[sel.Length];
        for (int s = 0; s < sel.Length; s++)
            slotToken[cursor[sel[s]]++] = s / Used;
        Assert.Equal(0, counts[3]);
        Assert.True(counts[1] > 32);
        return (counts, offsets, slotToken);
    }

    /// <summary>Each member slot's output of a grouped launch against its dequantized expert row;
    /// returns the worst error as a fraction of |w.a|.</summary>
    private static double AssertGrouped(string what, byte[] weights, int ggmlType, long rowBytes, int rows, int inDim,
        int[] counts, int[] offsets, Func<int, float[]> activation, float[] got)
    {
        var w = new float[inDim];
        double worst = 0;
        for (int x = 0; x < Experts; x++)
        {
            for (int r = 0; r < rows; r++)
            {
                Dequantize(ggmlType, weights, ((long)x * rows + r) * rowBytes, w, inDim);
                for (int i = 0; i < counts[x]; i++)
                {
                    int s = offsets[x] + i;
                    worst = Math.Max(worst, AssertDot($"{what}: expert {x} row {r} member {i}", w, activation(s),
                        got[(long)s * rows + r]));
                }
            }
        }
        return worst;
    }

    /// <summary>The decode kernels' bound (<see cref="Dsv4ExpertKernelTests"/>): within 1e-3 of the
    /// products' absolute sum.</summary>
    private static double AssertDot(string what, float[] w, float[] a, float got)
    {
        double sum = 0, mag = 0;
        for (int k = 0; k < a.Length; k++)
        {
            sum += (double)w[k] * a[k];
            mag += Math.Abs((double)w[k] * a[k]);
        }
        double err = Math.Abs(got - sum);
        Assert.True(err <= 1e-3 * mag + 1e-6, $"{what}: {got} vs {sum} (|w.a| {mag})");
        return err / Math.Max(mag, 1e-30);
    }

    /// <summary>One row through ggml (GgmlOps), and for Q5_1 through the written-out reference too:
    /// they must agree bit for bit.</summary>
    private static void Dequantize(int ggmlType, byte[] bytes, long offset, float[] dst, int n)
    {
        NativeDequant.DequantizeToFloat32(ggmlType, bytes, checked((int)offset), dst, 0, n);
        if (ggmlType == Q5_1)
            AssertIsGgmlsQ5_1(bytes, offset, dst, n);
    }

    private static void Clear(CudaAllocator allocator, params Tensor[] tensors)
    {
        foreach (Tensor t in tensors)
            allocator.Kernels.LaunchFillF32(Ptr(t), checked((int)t.ElementCount()), float.NaN, allocator.Stream.Handle);
    }

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);
}
