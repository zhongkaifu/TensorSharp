// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// Q2_0 routed experts (ISTA's Qwen3.8-Flash-Next pack) on the int8 tensor-core kernels
/// (<c>ts_dsv4_moe_q2i8_*</c>, TS_Q2_0_EXPERTS_INT8) against the path they replace, the F16 tensor-core kernel,
/// on the model's expert shapes (2560 -> 640 gate/up, 640 -> 2560 down). Both, and the per-slot decode kernels
/// (q8_1 activations against the integer codes: the arithmetic of ggml-cuda's MMQ/MMVQ), are held to the whole
/// expert block computed in double from the dequantized weights and unrounded activations.
/// </summary>
public sealed class CudaMoeQ2_0Int8Tests
{
    private const int Q2_0 = 42;
    private const int E = 2560, Ff = 640;

    private readonly ITestOutputHelper _output;

    public CudaMoeQ2_0Int8Tests(ITestOutputHelper output) => _output = output;

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

    /// <summary>Q2_0 rows: per 64 values an F16 scale d and 16 code bytes, weight j in byte j / 4 at bit
    /// 2 * (j % 4), value d * (code - 1). Scales of a trained matrix's size, codes uniform.</summary>
    private static byte[] Q2_0Rows(int rows, int width, int seed)
    {
        var rng = new Random(seed);
        int blocks = width / 64;
        var bytes = new byte[(long)rows * blocks * 18];
        rng.NextBytes(bytes);
        for (long b = 0; b < (long)rows * blocks; b++)
            BitConverter.TryWriteBytes(bytes.AsSpan((int)(b * 18), 2), (System.Half)(0.008f + 0.02f * (float)rng.NextDouble()));
        return bytes;
    }

    private static void DequantizeQ2_0(byte[] src, long offset, float[] dst, int n)
    {
        for (int b = 0; b < n / 64; b++)
        {
            int at = checked((int)(offset + b * 18L));
            float d = (float)BitConverter.ToHalf(src, at);
            for (int j = 0; j < 64; j++)
                dst[b * 64 + j] = d * (((src[at + 2 + j / 4] >> (2 * (j % 4))) & 3) - 1);
        }
    }

    /// <summary>Activations with a hidden state's spread: unit-scale values and a few channels ten times larger,
    /// which set their 32-value blocks' int8 scale.</summary>
    private static float[] HiddenStates(int rows, int width, int seed)
    {
        var rng = new Random(seed);
        var x = new float[(long)rows * width];
        for (int i = 0; i < x.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            x[i] = (float)(z * ((i % width) % 97 == 3 ? 10.0 : 1.0));
        }
        return x;
    }

    /// <summary>out[t] = shared[t] + sum_j w[t, j] * down_j(silu(gate_j x) * up_j x) in double, one expert's
    /// weights dequantized at a time.</summary>
    private static double[,] FloatBlock(byte[] gate, byte[] up, byte[] down, float[] cur, float[] shared,
        int[] sel, float[] selW, int nt, int used, int experts)
    {
        long guRow = E / 64 * 18L, downRow = Ff / 64 * 18L;
        var result = new double[nt, E];
        for (int t = 0; t < nt; t++)
            for (int c = 0; c < E; c++)
                result[t, c] = shared[(long)t * E + c];
        var g = new float[(long)Ff * E];
        var u = new float[(long)Ff * E];
        var d = new float[(long)E * Ff];
        var row = new float[Math.Max(E, Ff)];
        for (int x = 0; x < experts; x++)
        {
            var members = new List<(int t, int j)>();
            for (int t = 0; t < nt; t++)
                for (int j = 0; j < used; j++)
                    if (sel[t * used + j] == x)
                        members.Add((t, j));
            if (members.Count == 0)
                continue;
            for (int r = 0; r < Ff; r++)
            {
                DequantizeQ2_0(gate, ((long)x * Ff + r) * guRow, row, E);
                Array.Copy(row, 0, g, (long)r * E, E);
                DequantizeQ2_0(up, ((long)x * Ff + r) * guRow, row, E);
                Array.Copy(row, 0, u, (long)r * E, E);
            }
            for (int c = 0; c < E; c++)
            {
                DequantizeQ2_0(down, ((long)x * E + c) * downRow, row, Ff);
                Array.Copy(row, 0, d, (long)c * Ff, Ff);
            }
            var contrib = new double[members.Count][];
            Parallel.For(0, members.Count, mi =>
            {
                (int t, int j) = members[mi];
                var h = new double[Ff];
                for (int r = 0; r < Ff; r++)
                {
                    double gv = 0, uv = 0;
                    long wo = (long)r * E, xo = (long)t * E;
                    for (int k = 0; k < E; k++)
                    {
                        gv += (double)g[wo + k] * cur[xo + k];
                        uv += (double)u[wo + k] * cur[xo + k];
                    }
                    h[r] = gv / (1 + Math.Exp(-gv)) * uv;
                }
                var y = new double[E];
                for (int c = 0; c < E; c++)
                {
                    double s = 0;
                    long wo = (long)c * Ff;
                    for (int k = 0; k < Ff; k++)
                        s += d[wo + k] * h[k];
                    y[c] = selW[t * used + j] * s;
                }
                contrib[mi] = y;
            });
            for (int mi = 0; mi < members.Count; mi++)
                for (int c = 0; c < E; c++)
                    result[members[mi].t, c] += contrib[mi][c];
        }
        return result;
    }

    private static double RelativeError(float[] got, double[,] want, int nt)
    {
        double num = 0, den = 0;
        for (int t = 0; t < nt; t++)
            for (int c = 0; c < E; c++)
            {
                double diff = got[(long)t * E + c] - want[t, c];
                num += diff * diff;
                den += want[t, c] * want[t, c];
            }
        return Math.Sqrt(num / den);
    }

    private static double RelativeError(float[] got, float[] want)
    {
        double num = 0, den = 0;
        for (int i = 0; i < got.Length; i++)
        {
            double diff = got[i] - (double)want[i];
            num += diff * diff;
            den += (double)want[i] * want[i];
        }
        return Math.Sqrt(num / den);
    }

    /// <summary>
    /// A prompt ubatch of the expert block on the int8 kernels next to the F16 tensor-core kernel and the per-slot
    /// q8_1 kernels. The int8 route rounds the activations (and H) to int8 per 32 values, as the per-slot route does,
    /// with F32 instead of F16 block scales, so its error against the float block must stay within the per-slot
    /// route's; the F16 route rounds the activations to F16 and is the closer of the three. One expert has no members
    /// and one more than a 64-token pass. With <paramref name="viaTable"/> the experts are read through a residency
    /// table (stored in reverse order), as the engine reads VRAM slots.
    /// </summary>
    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheInt8ExpertBlock_StaysWithinTheQ8_1Error_OnFlashNextShapes(bool viaTable)
    {
        const int experts = 24, used = 8, nt = 160, perSlotMaxRows = 16;
        long guRowBytes = E / 64 * 18L, downRowBytes = Ff / 64 * 18L;
        byte[] gate = Q2_0Rows(experts * Ff, E, seed: 31);
        byte[] up = Q2_0Rows(experts * Ff, E, seed: 32);
        byte[] down = Q2_0Rows(experts * E, Ff, seed: 33);
        float[] cur = HiddenStates(nt, E, seed: 34);
        float[] shared = Enumerable.Range(0, nt * E).Select(i => 0.01f * MathF.Cos(i * 0.71f)).ToArray();
        // Experts 0, 3, ..., 21 for every seventh token on top of the rotation: expert 0 takes more than one pass;
        // nothing selects expert 23.
        int[] sel = Enumerable.Range(0, nt * used).Select(i => (i % used * 3 + (i / used % 7 == 0 ? 0 : i / used)) % 23).ToArray();
        float[] selW = Enumerable.Range(0, nt * used).Select(i => 0.05f + 0.2f * ((i * 7) % 5) / 4f).ToArray();
        int[] counts = new int[experts];
        foreach (int x in sel)
            counts[x]++;
        Assert.Equal(0, counts[experts - 1]);
        Assert.True(counts.Max() > 64, $"no expert takes a second pass (most members {counts.Max()})");
        for (int t = 0; t < nt; t++)
            Assert.Equal(used, sel.Skip(t * used).Take(used).Distinct().Count());

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        if (!dk.HasMoeQ2I8)
        {
            // A module without the kernels' bodies (compute_75 baseline, or a pre-sm_80 GPU): nothing to compare.
            _output.WriteLine("NOT RUN: the loaded module has no int8 Q2_0 expert kernels for this GPU");
            return;
        }
        var owned = new List<Tensor>();
        Tensor Own(Tensor t) { owned.Add(t); return t; }
        Tensor Alloc(DType type, long[] sizes) => Own(new Tensor(allocator, type, sizes));
        try
        {
            DeviceWeight Weight(byte[] bytes, int ne0, int ne1, long rowBytes)
            {
                var w = new DeviceWeight { Type = Q2_0, Ne0 = ne0, Ne1 = ne1, RowBytes = rowBytes };
                if (!viaTable)
                {
                    w.Ptr = Ptr(Own(Up(allocator, bytes)));
                    return w;
                }
                // The experts stored last to first, each behind a table entry.
                long expertBytes = ne1 * rowBytes;
                var reversed = new byte[bytes.Length];
                for (int x = 0; x < experts; x++)
                    Buffer.BlockCopy(bytes, (int)(x * expertBytes), reversed, (int)((experts - 1 - x) * expertBytes), (int)expertBytes);
                IntPtr basePtr = Ptr(Own(Up(allocator, reversed)));
                var table = new byte[experts * 8];
                for (int x = 0; x < experts; x++)
                    BitConverter.TryWriteBytes(table.AsSpan(x * 8, 8), (ulong)((long)basePtr + (experts - 1 - x) * expertBytes));
                w.Ptr = IntPtr.Zero;
                w.Table = Ptr(Own(Up(allocator, table)));
                return w;
            }
            DeviceWeight g = Weight(gate, E, Ff, guRowBytes), u = Weight(up, E, Ff, guRowBytes), d = Weight(down, Ff, E, downRowBytes);
            Tensor curT = Own(Up(allocator, cur));
            Tensor sharedT = Own(Up(allocator, shared));
            Tensor output = Alloc(DType.Float32, new long[] { nt, E });
            var scratch = new CudaMoeScratch(Alloc, nt, experts, used, E, Ff);
            scratch.Sel.SetElementsAsInt(sel);
            scratch.SelW.SetElementsAsFloat(selW);
            scratch.Sel.Storage.EnsureDeviceCurrent();
            scratch.SelW.Storage.EnsureDeviceCurrent();
            Assert.True(CudaMoe.UsesQ2I8(true, dk.HasMoeQ2I8, g, u, d, E, Ff, used));
            Assert.Equal(CudaMoe.ExpertRoute.TensorCore,
                CudaMoe.SelectRoute(nt, perSlotMaxRows, dk.HasMoeMma, dk.DecodesLowBitIqExperts, Q2_0, Q2_0, E, Ff, dk.DecodesLegacyQuants));

            float[] Run(bool int8, int maxPerSlot)
            {
                CudaMoe.Experts(dk, allocator.Kernels, scratch, g, u, d, curT, sharedT, output,
                    nt, used, experts, E, Ff, 0f, maxPerSlot, allocator.Stream.Handle, q2i8: int8);
                allocator.Stream.Synchronize();
                return Down(output, nt * E);
            }
            float[] i8 = Run(true, perSlotMaxRows);
            float[] f16 = Run(false, perSlotMaxRows);
            float[] q81 = Run(false, nt);
            float[] i8Again = Run(true, perSlotMaxRows);
            Assert.All(i8, v => Assert.True(float.IsFinite(v)));
            Assert.Equal(i8, i8Again);

            double[,] want = FloatBlock(gate, up, down, cur, shared, sel, selW, nt, used, experts);
            double eI8 = RelativeError(i8, want, nt), eF16 = RelativeError(f16, want, nt), eQ81 = RelativeError(q81, want, nt);
            double i8VsF16 = RelativeError(i8, f16);
            _output.WriteLine($"{(viaTable ? "table" : "stacked")}: relative L2 error against the float block: int8 {eI8:E2}, " +
                $"F16 tensor cores {eF16:E2}, per-slot q8_1 {eQ81:E2}; int8 against F16 {i8VsF16:E2} " +
                $"(members per expert {counts.Min()}-{counts.Max()})");
            Assert.True(eF16 < eQ81, $"the F16 route ({eF16:E2}) is not closer than the q8_1 route ({eQ81:E2})");
            Assert.True(eI8 <= 1.25 * eQ81, $"the int8 route strayed {eI8:E2} from the float block, past the q8_1 route's {eQ81:E2}");
        }
        finally
        {
            allocator.Synchronize();
            foreach (Tensor t in owned)
                t.Dispose();
        }
    }

    /// <summary>
    /// Timing, not a gate: one Flash Next expert layer (512 experts, top 10) over a 1024-row prompt ubatch, the
    /// engine's default, on the F16 and the int8 tensor-core kernels. Uniform random routes, so nearly every
    /// expert's weights are read once per ubatch, as in a long prompt.
    /// </summary>
    [CudaTheory]
    [Trait("Category", "Bench")]
    [InlineData(1024)]
    [InlineData(4096)]
    public void ALayerOfFlashNextExperts_F16AgainstInt8_Timing(int nt)
    {
        const int experts = 512, used = 10, perSlotMaxRows = 16, reps = 20;
        long guRowBytes = E / 64 * 18L, downRowBytes = Ff / 64 * 18L;
        byte[] gate = Q2_0Rows(experts * Ff, E, seed: 41);
        byte[] up = Q2_0Rows(experts * Ff, E, seed: 42);
        byte[] down = Q2_0Rows(experts * E, Ff, seed: 43);
        float[] cur = HiddenStates(nt, E, seed: 44);
        var rng = new Random(45);
        int[] sel = new int[nt * used];
        for (int t = 0; t < nt; t++)
        {
            var picked = new HashSet<int>();
            while (picked.Count < used)
                picked.Add(rng.Next(experts));
            picked.ToArray().CopyTo(sel, t * used);
        }
        float[] selW = Enumerable.Repeat(0.1f, nt * used).ToArray();

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        if (!dk.HasMoeQ2I8)
        {
            _output.WriteLine("NOT RUN: the loaded module has no int8 Q2_0 expert kernels for this GPU");
            return;
        }
        var owned = new List<Tensor>();
        Tensor Own(Tensor t) { owned.Add(t); return t; }
        Tensor Alloc(DType type, long[] sizes) => Own(new Tensor(allocator, type, sizes));
        try
        {
            var g = new DeviceWeight { Ptr = Ptr(Own(Up(allocator, gate))), Type = Q2_0, Ne0 = E, Ne1 = Ff, RowBytes = guRowBytes };
            var u = new DeviceWeight { Ptr = Ptr(Own(Up(allocator, up))), Type = Q2_0, Ne0 = E, Ne1 = Ff, RowBytes = guRowBytes };
            var d = new DeviceWeight { Ptr = Ptr(Own(Up(allocator, down))), Type = Q2_0, Ne0 = Ff, Ne1 = E, RowBytes = downRowBytes };
            Tensor curT = Own(Up(allocator, cur));
            Tensor sharedT = Alloc(DType.Float32, new long[] { nt, E });
            Tensor output = Alloc(DType.Float32, new long[] { nt, E });
            var scratch = new CudaMoeScratch(Alloc, nt, experts, used, E, Ff);
            scratch.Sel.SetElementsAsInt(sel);
            scratch.SelW.SetElementsAsFloat(selW);
            scratch.Sel.Storage.EnsureDeviceCurrent();
            scratch.SelW.Storage.EnsureDeviceCurrent();
            double Time(bool int8)
            {
                void Once() => CudaMoe.Experts(dk, allocator.Kernels, scratch, g, u, d, curT, sharedT, output,
                    nt, used, experts, E, Ff, 0f, perSlotMaxRows, allocator.Stream.Handle, q2i8: int8);
                for (int i = 0; i < 3; i++)
                    Once();
                allocator.Stream.Synchronize();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < reps; i++)
                    Once();
                allocator.Stream.Synchronize();
                return sw.Elapsed.TotalMilliseconds / reps;
            }
            double f16 = Time(false), i8 = Time(true), f16Again = Time(false), i8Again = Time(true);
            // The int8 kernels alone, on the grouping and activations the last int8 call left in the scratch.
            IntPtr xa = Ptr(scratch.ExpUp), hq = Ptr(scratch.ExpGate), stream = allocator.Stream.Handle;
            double Kernel(Action launch)
            {
                for (int i = 0; i < 3; i++)
                    launch();
                allocator.Stream.Synchronize();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < reps; i++)
                    launch();
                allocator.Stream.Synchronize();
                return sw.Elapsed.TotalMilliseconds / reps;
            }
            double gu = Kernel(() => dk.MoeQ2I8GateUp(g.Ptr, u.Ptr, xa, scratch.Counts, scratch.Offsets, scratch.SlotToken, hq,
                Ff, E, guRowBytes, 0f, experts, stream));
            double dn = Kernel(() => dk.MoeQ2I8Down(d.Ptr, hq, scratch.Counts, scratch.Offsets, scratch.ExpDown, E, Ff, downRowBytes,
                experts, stream));
            _output.WriteLine($"512 experts, top 10, {nt} rows, 2560 x 640 Q2_0: F16 tensor cores {f16:F3} / {f16Again:F3} ms, " +
                $"int8 tensor cores {i8:F3} / {i8Again:F3} ms a layer (int8 gate/up {gu:F3} ms, down {dn:F3} ms)");
        }
        finally
        {
            allocator.Synchronize();
            foreach (Tensor t in owned)
                t.Dispose();
        }
    }

    /// <summary>The int8 kernels run only when switched on, on Q2_0 gate/up/down, with widths on their grid, room
    /// for the tokens' activations in the scratch, and stacked weights on a 4-byte boundary.</summary>
    [Fact]
    public void TheInt8Route_TakesOnlyWhatItCanRead()
    {
        var w = new DeviceWeight { Type = Q2_0, Ptr = (IntPtr)0x1000, RowBytes = E / 64 * 18 };
        var odd = w with { Ptr = (IntPtr)0x1002 };
        var tabled = odd with { Table = (IntPtr)0x2000 };
        var q8 = w with { Type = 8 };
        Assert.True(CudaMoe.UsesQ2I8(true, true, w, w, w, E, Ff, 10));
        Assert.False(CudaMoe.UsesQ2I8(false, true, w, w, w, E, Ff, 10));
        Assert.False(CudaMoe.UsesQ2I8(true, false, w, w, w, E, Ff, 10));
        Assert.False(CudaMoe.UsesQ2I8(true, true, q8, w, w, E, Ff, 10));
        Assert.False(CudaMoe.UsesQ2I8(true, true, w, w, q8, E, Ff, 10));
        Assert.False(CudaMoe.UsesQ2I8(true, true, w, w, w, E, 576, 10));
        Assert.False(CudaMoe.UsesQ2I8(true, true, w, w, w, 2496, Ff, 10));
        Assert.False(CudaMoe.UsesQ2I8(true, true, odd, w, w, E, Ff, 10));
        Assert.True(CudaMoe.UsesQ2I8(true, true, tabled, w, w, E, Ff, 10));
        // One selected expert of 128 features cannot hold a 2560-wide token's 3,200 activation bytes.
        Assert.False(CudaMoe.UsesQ2I8(true, true, w, w, w, E, 128, 1));
    }
}
