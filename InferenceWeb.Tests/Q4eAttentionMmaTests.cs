// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The Flash Next engine's prompt attention on tensor cores (<see cref="Q4eKernels.AttentionMma"/>: a block per row and KV
/// head, the group's query heads one 16-row mma tile, F16 operands, F32 accumulation) against the one-block-per-(head, row)
/// kernel it replaces for prompt-size calls, which keeps F32. Rows attend densely (0..p) or over listed cells (the QSA
/// selection: a row's own list in any order), at Flash Next's 24 query heads over 2 KV heads and at a two-head group, over
/// the F16 and the Q8_0 cache. What remains is the operands' F16 rounding.
/// </summary>
public sealed class Q4eAttentionMmaTests
{
    private const int HeadDim = 256, Rows = 48, P0 = 2000, Ctx = P0 + Rows, Listed = 300;

    private readonly ITestOutputHelper _output;

    public Q4eAttentionMmaTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(24, 2, false)]
    [InlineData(24, 2, true)]
    [InlineData(4, 2, false)]
    [InlineData(4, 2, true)]
    public void TheTensorCoreKernel_AnswersAsTheScalarKernel(int nHead, int nKv, bool q8)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var qk = Q4eKernels.Create();
        Assert.True(qk.HasAttentionMma, "the loaded tensorsharp_q4e_kernels module has no tensor-core attention for this GPU");
        IntPtr stream = allocator.Stream.Handle;
        var rnd = new Random(nHead * 31 + nKv + (q8 ? 7 : 0));
        float U(double a) => (float)((rnd.NextDouble() * 2 - 1) * a);

        var q = new float[Rows * nHead * HeadDim];
        var gate = new float[q.Length];
        for (int i = 0; i < q.Length; i++)
        {
            q[i] = U(2.0);
            gate[i] = U(3.0);
        }
        // Keys and values of every cell; a Q8_0 cache stores codes and an F16 scale per 32 values.
        long kvValues = (long)Ctx * nKv * HeadDim;
        var kF = new float[kvValues];
        var vF = new float[kvValues];
        for (long i = 0; i < kvValues; i++)
        {
            kF[i] = U(2.0);
            vF[i] = U(1.5);
        }
        // Even rows dense (0..p), odd rows over Listed distinct cells of 0..p in a shuffled order.
        var cells = new int[Rows * Listed];
        var cellCnt = new int[Rows];
        for (int t = 0; t < Rows; t++)
        {
            if (t % 2 == 0)
            {
                cellCnt[t] = -1;
                continue;
            }
            cellCnt[t] = Listed;
            int p = P0 + t;
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

        using var dQ = Up(allocator, q);
        using var dGate = Up(allocator, gate);
        using var dCells = Up(allocator, cells);
        using var dCnt = Up(allocator, cellCnt);
        using var outScalar = new Tensor(allocator, DType.Float32, q.Length);
        using var outMma = new Tensor(allocator, DType.Float32, q.Length);
        Tensor dK, dV, dKs = null, dVs = null;
        if (q8)
        {
            (byte[] kc, byte[] ks) = Q8(kF);
            (byte[] vc, byte[] vs) = Q8(vF);
            dK = Up(allocator, kc);
            dV = Up(allocator, vc);
            dKs = Up(allocator, ks);
            dVs = Up(allocator, vs);
        }
        else
        {
            dK = Up(allocator, F16(kF));
            dV = Up(allocator, F16(vF));
        }
        try
        {
            IntPtr P(Tensor t) => t == null ? IntPtr.Zero : Dsv4CudaEngine.Ptr(t);
            float scale = 1.0f / MathF.Sqrt(HeadDim);
            qk.Attention(P(dQ), P(dK), P(dV), P(dGate), P(dCells), P(dCnt), P(outScalar), IntPtr.Zero, P0, nHead, nKv, scale,
                Listed, Rows, stream, P(dKs), P(dVs));
            qk.AttentionMma(P(dQ), P(dK), P(dV), P(dGate), P(dCells), P(dCnt), P(outMma), IntPtr.Zero, P0, nHead, nKv, scale,
                Listed, Rows, stream, P(dKs), P(dVs));
            allocator.Stream.Synchronize();
            float[] want = Down(outScalar, q.Length), got = Down(outMma, q.Length);
            double num = 0, den = 0, worstRow = 0;
            for (int t = 0; t < Rows; t++)
            {
                double rn = 0, rd = 0;
                for (int i = t * nHead * HeadDim; i < (t + 1) * nHead * HeadDim; i++)
                {
                    Assert.False(float.IsNaN(got[i]), $"row {t}: NaN at {i}");
                    double d = got[i] - want[i];
                    rn += d * d;
                    rd += (double)want[i] * want[i];
                }
                num += rn;
                den += rd;
                worstRow = Math.Max(worstRow, Math.Sqrt(rn / rd));
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"{nHead} heads over {nKv}, {(q8 ? "Q8_0" : "F16")} cache: relative L2 {rel:E2}, worst row {worstRow:E2}");
            // F16 queries, keys, values and probabilities, about 2^-11 relative each: measured 3.2e-4 .. 5.3e-4 a row on an
            // RTX 5090 (relative L2 2.5e-4 .. 4.2e-4).
            Assert.True(worstRow <= 1e-3, $"a row's attention differs from the scalar kernel's by {worstRow:E2}");
        }
        finally
        {
            dK.Dispose();
            dV.Dispose();
            dKs?.Dispose();
            dVs?.Dispose();
        }
    }

    internal static byte[] F16(float[] v)
    {
        var b = new byte[v.Length * 2];
        for (int i = 0; i < v.Length; i++)
            BitConverter.TryWriteBytes(b.AsSpan(2 * i), BitConverter.HalfToUInt16Bits((System.Half)v[i]));
        return b;
    }

    /// <summary>ggml's Q8_0 of each 32 values: int8 codes round(x / d), d = max|x| / 127 as F16.</summary>
    internal static (byte[] Codes, byte[] Scales) Q8(float[] v)
    {
        var codes = new byte[v.Length];
        var scales = new byte[v.Length / 32 * 2];
        for (int b = 0; b < v.Length / 32; b++)
        {
            float amax = 0;
            for (int i = 0; i < 32; i++)
                amax = MathF.Max(amax, MathF.Abs(v[b * 32 + i]));
            float d = amax / 127f;
            System.Half dh = (System.Half)d;
            BitConverter.TryWriteBytes(scales.AsSpan(2 * b), BitConverter.HalfToUInt16Bits(dh));
            float id = d != 0 ? 1f / d : 0f;
            for (int i = 0; i < 32; i++)
                codes[b * 32 + i] = (byte)(sbyte)Math.Clamp((int)MathF.Round(v[b * 32 + i] * id), -127, 127);
        }
        return (codes, scales);
    }
}
