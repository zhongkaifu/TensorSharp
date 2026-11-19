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
/// the state after the rows kept (the shipped body), for scalar and per-key gates, shared key heads and several widths.
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

    private static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();
}
