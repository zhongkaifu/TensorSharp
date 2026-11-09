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
/// The Flash Next engine's prompt recurrence in chunks of 32 tokens (<see cref="Q4eKernels.GdnChunked"/>, the WY form of the
/// gated delta rule after Strata's chunked prompt recurrence) against the token-by-token scan it replaces for prompts
/// (GlmKernels.KdaScan): the same GDN scratch rows (unit q and k shared by a key head's three value heads, v, a negative log
/// decay and beta per row and value head) and the same starting state, at Flash Next's 16 key / 48 value heads. Rows: one
/// chunk, a partial last chunk, and two super-blocks (2048 rows each) with a partial chunk. What may differ is FP32 rounding:
/// the chunked form sums in another order.
/// </summary>
public sealed class Q4eGdnChunkedTests
{
    private const int HK = 16, HV = 48, HD = 128, Scr = 4 * HD + 1;
    private readonly ITestOutputHelper _output;

    public Q4eGdnChunkedTests(ITestOutputHelper output) => _output = output;

    [CudaTheory]
    [InlineData(128)]
    [InlineData(200)]
    [InlineData(4171)]
    public void TheChunkedRecurrence_AnswersAsTheTokenByTokenScan(int rows)
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var qk = Q4eKernels.Create();
        using var gk = GlmKernels.Create();
        Assert.True(qk.HasGdnChunked, "the loaded tensorsharp_q4e_kernels module has no chunked recurrence for this GPU");
        IntPtr stream = allocator.Stream.Handle;
        var rnd = new Random(rows);
        float U(double a) => (float)((rnd.NextDouble() * 2 - 1) * a);

        var scr = new float[(long)HV * rows * Scr];
        var q = new float[HD];
        var k = new float[HD];
        for (int t = 0; t < rows; t++)
            for (int kh = 0; kh < HK; kh++)
            {
                double qs = 0, ks = 0;
                for (int d = 0; d < HD; d++) { q[d] = U(1); k[d] = U(1); qs += q[d] * q[d]; ks += k[d] * k[d]; }
                for (int vi = 0; vi < 3; vi++)
                {
                    int h = kh + vi * HK;
                    long row = ((long)h * rows + t) * Scr;
                    float g = -(float)(0.001 + 0.4 * rnd.NextDouble());
                    for (int d = 0; d < HD; d++)
                    {
                        scr[row + d] = (float)(q[d] / Math.Sqrt(qs));
                        scr[row + HD + d] = (float)(k[d] / Math.Sqrt(ks));
                        scr[row + 2 * HD + d] = U(1);
                        scr[row + 3 * HD + d] = g;
                    }
                    scr[row + 4 * HD] = (float)(0.05 + 0.9 * rnd.NextDouble());
                }
            }
        var state0 = new float[(long)HV * HD * HD];
        for (long i = 0; i < state0.Length; i++) state0[i] = U(0.2);

        using var dScr = Up(allocator, scr);
        using var stateA = Up(allocator, state0);
        using var stateB = Up(allocator, state0);
        using var coreA = new Tensor(allocator, DType.Float32, (long)rows * HV * HD);
        using var coreB = new Tensor(allocator, DType.Float32, (long)rows * HV * HD);
        using var scratch = new Tensor(allocator, DType.Float32, Q4eKernels.GdnChunkScratchFloats(HV));
        IntPtr P(Tensor t) => Dsv4CudaEngine.Ptr(t);
        static int[] PtrArray(IntPtr p) => new[] { (int)((long)p & 0xFFFFFFFF), (int)((long)p >> 32) };
        using var ptrA = Up(allocator, PtrArray(P(stateA)));
        using var ptrB = Up(allocator, PtrArray(P(stateB)));

        gk.KdaScan(P(dScr), P(ptrA), P(coreA), rows, 1, HV, stream);
        long tp = Q4eKernels.GdnChunkTPFloats(HV) * 4;
        IntPtr tm = P(scratch);
        qk.GdnChunked(P(dScr), P(ptrB), P(coreB), tm, (IntPtr)((long)tm + tp), (IntPtr)((long)tm + 2 * tp), rows, HK, HV, stream);
        allocator.Stream.Synchronize();

        (double rel, double worst) Compare(float[] want, float[] got, int rowLength)
        {
            double num = 0, den = 0, worstRow = 0;
            for (int r = 0; r < want.Length / rowLength; r++)
            {
                double rn = 0, rd = 0;
                for (int i = r * rowLength; i < (r + 1) * rowLength; i++)
                {
                    Assert.False(float.IsNaN(got[i]), $"NaN at {i}");
                    double d = got[i] - want[i];
                    rn += d * d;
                    rd += (double)want[i] * want[i];
                }
                num += rn;
                den += rd;
                worstRow = Math.Max(worstRow, Math.Sqrt(rn / Math.Max(rd, 1e-30)));
            }
            return (Math.Sqrt(num / den), worstRow);
        }
        var (outRel, outWorst) = Compare(Down(coreA, rows * HV * HD), Down(coreB, rows * HV * HD), HV * HD);
        var (stRel, stWorst) = Compare(Down(stateA, HV * HD * HD), Down(stateB, HV * HD * HD), HD * HD);
        _output.WriteLine($"{rows} rows: outputs relative L2 {outRel:E2} (worst row {outWorst:E2}), state relative L2 {stRel:E2} (worst head {stWorst:E2})");
        // FP32 sums in another order over up to 4171 decayed updates: ~1e-6 relative expected.
        Assert.True(outWorst <= 1e-4, $"a row's outputs differ from the token-by-token scan's by {outWorst:E2}");
        Assert.True(stWorst <= 1e-4, $"a head's final state differs from the token-by-token scan's by {stWorst:E2}");
    }
}
