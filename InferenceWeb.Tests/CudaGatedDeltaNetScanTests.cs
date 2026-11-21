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

namespace InferenceWeb.Tests;

/// <summary>
/// A verify window's gated delta-net scan (outputs alone, its inputs saved for the commit's re-step) runs on
/// ts_graph_gated_delta_net_scan4_f32, which reads each token's inputs a token ahead. Its outputs and saved inputs must be the
/// one-row kernel's bits (the decode row's arithmetic, which the speculative contract rests on), and the rows4 kernel's with
/// the state after the rows kept (the shipped body), for scalar and per-key gates, shared key heads and several widths; and
/// the row sums are the eight-lane grouping (ts_graph_gdn8, every gated delta rule of the image).
/// </summary>
public sealed class CudaGatedDeltaNetScanTests
{
    [CudaTheory]
    [InlineData(128, 48, 16, 8, false)]
    [InlineData(128, 48, 16, 2, false)]
    [InlineData(128, 12, 4, 16, false)]
    [InlineData(64, 6, 3, 5, false)]
    [InlineData(128, 8, 8, 8, true)]
    [InlineData(64, 4, 2, 3, true)]
    public void AheadScanIsTheOneRowKernelsBits(int s, int heads, int keyHeads, int tokens, bool perKeyDecay)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(5 + tokens);
            float[] Uniform(long count, double lo, double hi)
            {
                var a = new float[count];
                for (long i = 0; i < count; i++) a[i] = (float)(lo + rng.NextDouble() * (hi - lo));
                return a;
            }
            long rowFloats = (long)tokens * heads * s, stateFloats = (long)heads * s * s;
            long saveFloats = CudaGraphKernels.GatedDeltaNetInputFloats(tokens, s, heads, keyHeads, perKeyDecay);
            using var q = Tensor.FromArray(allocator, Uniform((long)tokens * keyHeads * s, -0.3, 0.3));
            using var k = Tensor.FromArray(allocator, Uniform((long)tokens * keyHeads * s, -0.3, 0.3));
            using var v = Tensor.FromArray(allocator, Uniform(rowFloats, -1, 1));
            using var g = Tensor.FromArray(allocator, Uniform((long)tokens * heads * (perKeyDecay ? s : 1), -0.6, -0.01));
            using var beta = Tensor.FromArray(allocator, Uniform((long)tokens * heads, 0.05, 0.95));
            using var state = Tensor.FromArray(allocator, Uniform(stateFloats, -0.2, 0.2));
            using var scanOut = new Tensor(allocator, DType.Float32, rowFloats);
            using var oneRowOut = new Tensor(allocator, DType.Float32, rowFloats);
            using var keptOut = new Tensor(allocator, DType.Float32, rowFloats + stateFloats);
            using var scanSave = new Tensor(allocator, DType.Float32, saveFloats);
            using var oneRowSave = new Tensor(allocator, DType.Float32, saveFloats);
            foreach (Tensor t in new[] { q, k, v, g, beta, state, scanOut, oneRowOut, keptOut, scanSave, oneRowSave })
                ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
            IntPtr stream = allocator.Stream.Handle;
            // The verify window's scan (no slot, no saved state: the ahead-reading kernel), the one-row kernel's, and the
            // rows4 kernel's with the state after the rows written (the body the commit's re-step runs).
            kernels.LaunchGatedDeltaNetRows4(P(q), P(k), P(v), P(g), P(beta), P(state), P(scanOut), s, heads, keyHeads, tokens, 1, 0,
                perKeyDecay, stream, IntPtr.Zero, P(scanSave), tokens);
            kernels.LaunchGatedDeltaNet(P(q), P(k), P(v), P(g), P(beta), P(state), P(oneRowOut), s, heads, keyHeads, tokens, 1, 0,
                perKeyDecay, stream, IntPtr.Zero, P(oneRowSave), tokens);
            kernels.LaunchGatedDeltaNetRows4(P(q), P(k), P(v), P(g), P(beta), P(state), P(keptOut), s, heads, keyHeads, tokens, 1, 1,
                perKeyDecay, stream);
            allocator.Stream.Synchronize();
            foreach (Tensor t in new[] { scanOut, oneRowOut, keptOut, scanSave, oneRowSave })
                ((CudaStorage)t.Storage).MarkDeviceModified();
            float[] scan = scanOut.GetElementsAsFloat((int)rowFloats);
            Assert.Contains(scan, x => x != 0);
            Assert.Equal(Bits(oneRowOut.GetElementsAsFloat((int)rowFloats)), Bits(scan));
            Assert.Equal(Bits(keptOut.GetElementsAsFloat((int)rowFloats)), Bits(scan));
            Assert.Equal(Bits(oneRowSave.GetElementsAsFloat((int)saveFloats)), Bits(scanSave.GetElementsAsFloat((int)saveFloats)));
        }
    }

    /// <summary>
    /// The recurrence sums each value row's S keys over eight lanes (lane i keeping keys i, i + 8, ...: its fused multiply-add
    /// sum, then a three-step xor butterfly), as the direct engine laid it out: with no decay (g = 0, so each decay is an exact
    /// 1) the state after the rows is that grouping's bits, stepped here on the host, and the outputs agree to F32 rounding.
    /// </summary>
    [CudaTheory]
    [InlineData(128, 48, 16, 8)]
    [InlineData(64, 6, 3, 5)]
    public void TheRowSumsAreEightLanesEach(int s, int heads, int keyHeads, int tokens)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            Assert.True(kernels.TakesGdn8, "the image has no eight-lane recurrence");
            var rng = new Random(91 + tokens);
            float[] Uniform(long count, double lo, double hi)
            {
                var a = new float[count];
                for (long i = 0; i < count; i++) a[i] = (float)(lo + rng.NextDouble() * (hi - lo));
                return a;
            }
            long rowFloats = (long)tokens * heads * s, stateFloats = (long)heads * s * s;
            float[] qh = Uniform((long)tokens * keyHeads * s, -0.3, 0.3), kh = Uniform((long)tokens * keyHeads * s, -0.3, 0.3);
            float[] vh = Uniform(rowFloats, -1, 1), bh = Uniform((long)tokens * heads, 0.05, 0.95), sh = Uniform(stateFloats, -0.2, 0.2);
            using var q = Tensor.FromArray(allocator, qh);
            using var k = Tensor.FromArray(allocator, kh);
            using var v = Tensor.FromArray(allocator, vh);
            using var g = Tensor.FromArray(allocator, new float[(long)tokens * heads]);
            using var beta = Tensor.FromArray(allocator, bh);
            using var state = Tensor.FromArray(allocator, sh);
            using var kept = new Tensor(allocator, DType.Float32, rowFloats + stateFloats);
            foreach (Tensor t in new[] { q, k, v, g, beta, state, kept })
                ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
            kernels.LaunchGatedDeltaNetRows4(P(q), P(k), P(v), P(g), P(beta), P(state), P(kept), s, heads, keyHeads, tokens, 1, 1,
                false, allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            ((CudaStorage)kept.Storage).MarkDeviceModified();
            float[] device = kept.GetElementsAsFloat((int)(rowFloats + stateFloats));

            float[] st = (float[])sh.Clone();
            var outputs = new float[rowFloats];
            int per = s / 8;
            float scale = 1f / MathF.Sqrt(s);
            var partial = new float[8];
            for (int t = 0; t < tokens; t++)
                for (int h = 0; h < heads; h++)
                {
                    long qk = ((long)t * keyHeads + h % keyHeads) * s, vo = ((long)t * heads + h) * s;
                    float b = bh[(long)t * heads + h];
                    for (int r = 0; r < s; r++)
                    {
                        long row = ((long)h * s + r) * s;
                        float Sum(Func<int, int, float> lane)
                        {
                            for (int li = 0; li < 8; li++) partial[li] = lane(li, per);
                            for (int off = 4; off > 0; off >>= 1)
                            {
                                var next = new float[8];
                                for (int li = 0; li < 8; li++) next[li] = partial[li] + partial[li ^ off];
                                next.CopyTo(partial, 0);
                            }
                            return partial[0];
                        }
                        float kv = Sum((li, n) =>
                        {
                            float acc = 0f;
                            for (int j = 0; j < n; j++) acc = MathF.FusedMultiplyAdd(st[row + li + 8 * j], kh[qk + li + 8 * j], acc);
                            return acc;
                        });
                        float delta = (vh[vo + r] - kv) * b;
                        for (int c = 0; c < s; c++) st[row + c] = MathF.FusedMultiplyAdd(kh[qk + c], delta, st[row + c]);
                        float y = Sum((li, n) =>
                        {
                            float acc = 0f;
                            for (int j = 0; j < n; j++) acc = MathF.FusedMultiplyAdd(st[row + li + 8 * j], qh[qk + li + 8 * j], acc);
                            return acc;
                        });
                        outputs[vo + r] = y * scale;
                    }
                }
            Assert.Equal(Bits(st), Bits(device.AsSpan((int)rowFloats).ToArray()));
            for (long i = 0; i < rowFloats; i++)
                Assert.True(Math.Abs(device[i] - outputs[i]) <= 1e-6f * (1f + Math.Abs(outputs[i])), $"output {i}: {device[i]} on the device, {outputs[i]} stepped");
        }
    }

    private static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();
}
