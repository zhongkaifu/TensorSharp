// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A block drafter's ring attention on the CUDA graph executor's tensor-core kernel (ts_graph_ring_attn_f16_d64 / _d128 and
/// ts_graph_ring_attn_combine_f32, the direct engine's ts_q35_df_attn_f16 over the runtime's F32 ring and additive mask)
/// against attention computed in double on the host: the queries read in place from their [rows, heads, d] layout as a
/// [heads, rows, d] view (the drafter's permuted queries), slots outside the window closed by -inf in the mask (their V huge,
/// so a leak shows), F16 and F32 masks, a ring of the 27B drafter's shape (2,080 slots: nine key ranges) and groups wider
/// than a block's 64 rows. The outputs must agree within F16 operand rounding, and two runs must give the same bits.
/// </summary>
public sealed class CudaRingAttentionKernelTests
{
    private readonly ITestOutputHelper _output;

    public CudaRingAttentionKernelTests(ITestOutputHelper output) => _output = output;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    [CudaTheory]
    [InlineData(64, 4, 2, 5, 32, true)]       // the synthetic DFlash2 drafter: block 5, a 32-slot ring
    [InlineData(64, 4, 2, 4, 32, false)]      // an F32 mask
    [InlineData(128, 32, 8, 16, 2080, true)]  // Qwen3.8-27B's DFlash2 drafter: block 16 over a 2,080-slot ring
    [InlineData(128, 16, 1, 9, 300, true)]    // 144 rows a KV head: three row tiles
    public void RingAttentionMatchesTheReference(int d, int heads, int kvHeads, int rows, int keys, bool halfMask)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels created, out string reason), reason);
        using CudaGraphKernels kernels = created;
        if (!kernels.TakesRingAttention)
        {
            _output.WriteLine("no ring attention kernels for this GPU");
            return;
        }
        int group = heads / kvHeads;
        var rng = new Random(d * 7 + rows + keys);
        float[] q = Enumerable.Range(0, rows * heads * d).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] k = Enumerable.Range(0, kvHeads * keys * d).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] v = Enumerable.Range(0, kvHeads * keys * d).Select(_ => (float)(rng.NextDouble() * 4 - 2)).ToArray();
        // The drafter's window: a band of slots no row sees (older positions), others closed row by row, the block's own slots
        // open to every row. A slot no row sees holds a huge V, so any weight it got would show.
        var open = new bool[rows, keys];
        int band = keys / 4;
        for (int r = 0; r < rows; r++)
            for (int s = 0; s < keys; s++)
                open[r, s] = s >= keys - rows || (!(s >= band && s < band + 4) && (s + 3 * r) % 7 != 0);
        for (int h = 0; h < kvHeads; h++)
            for (int s = 0; s < keys; s++)
                if (Enumerable.Range(0, rows).All(r => !open[r, s]))
                    for (int c = 0; c < d; c++) v[((long)h * keys + s) * d + c] = 1000f;
        float[] maskValues = new float[rows * keys];
        for (int r = 0; r < rows; r++)
            for (int s = 0; s < keys; s++)
                maskValues[r * keys + s] = open[r, s] ? 0f : float.NegativeInfinity;
        float scale = 1f / MathF.Sqrt(d);

        using var qRows = Tensor.FromArray(allocator, q);
        using var kRing = Tensor.FromArray(allocator, k);
        using var vRing = Tensor.FromArray(allocator, v);
        using var mask32 = Tensor.FromArray(allocator, maskValues);
        using var mask16 = new Tensor(allocator, DType.Float16, rows, keys);
        foreach (Tensor t in new[] { mask32, mask16 }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        kernels.LaunchConvert(P(mask32), P(mask16), maskValues.Length, true, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        ((CudaStorage)mask16.Storage).MarkDeviceModified();
        Tensor mask = halfMask ? mask16 : mask32;
        using var partial = new Tensor(allocator, DType.Float32, CudaGraphKernels.RingAttentionPartialFloats(rows, heads, keys, d));

        float[] Run()
        {
            using var output = new Tensor(allocator, DType.Float32, rows, heads, d);
            foreach (Tensor t in new[] { qRows, kRing, vRing, partial, output }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            // The queries as the drafter's [heads, rows, d] view of its [rows, heads, d] rows: a head d floats on, a row heads * d.
            kernels.LaunchRingAttention(P(qRows), (long)heads * d, d, P(kRing), P(vRing), keys, P(mask), halfMask, keys, rows, heads, kvHeads, d,
                scale, P(partial), P(output), allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            ((CudaStorage)output.Storage).MarkDeviceModified();
            return output.GetElementsAsFloat(rows * heads * d);
        }

        float[] got = Run(), again = Run();
        double worst = 0, num = 0, den = 0;
        var scores = new double[keys];
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < heads; h++)
            {
                int kvh = h / group;
                double max = double.NegativeInfinity;
                for (int s = 0; s < keys; s++)
                {
                    double dot = 0;
                    for (int c = 0; c < d; c++) dot += (double)q[((long)r * heads + h) * d + c] * k[((long)kvh * keys + s) * d + c];
                    scores[s] = open[r, s] ? dot * scale : double.NegativeInfinity;
                    max = Math.Max(max, scores[s]);
                }
                double sum = 0;
                for (int s = 0; s < keys; s++) sum += scores[s] = Math.Exp(scores[s] - max);
                for (int c = 0; c < d; c++)
                {
                    double want = 0;
                    for (int s = 0; s < keys; s++) want += scores[s] / sum * v[((long)kvh * keys + s) * d + c];
                    double have = got[((long)r * heads + h) * d + c];
                    worst = Math.Max(worst, Math.Abs(have - want));
                    num += (have - want) * (have - want);
                    den += want * want;
                }
            }
        double rel = Math.Sqrt(num / den);
        _output.WriteLine($"d {d}, {heads}/{kvHeads} heads, {rows} rows over {keys} slots: relative L2 {rel:E2}, worst |diff| {worst:E2}");
        Assert.True(rel < 5e-3, $"relative L2 {rel:E2} from the reference");
        Assert.True(worst < 2e-2, $"worst difference {worst:E2} from the reference");
        for (int i = 0; i < got.Length; i++)
            if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(again[i]))
                Assert.Fail($"value {i} is {again[i]:R} on the second run, {got[i]:R} on the first");
    }
}
