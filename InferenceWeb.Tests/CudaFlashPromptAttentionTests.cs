// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A prompt chunk's attention on the prompt kernels (ts_flash_prompt_*): scheduled for the causal triangle (the heaviest row
/// tiles first, a warp skipping the key tiles past its rows) they give the plain kernel's bits, over Qwen3.8-27B's 2,048-row
/// first chunk, its 1,570-row second over the first's keys, a floor and a split launch.
/// </summary>
public sealed class CudaFlashPromptAttentionTests
{
    private readonly ITestOutputHelper _output;

    public CudaFlashPromptAttentionTests(ITestOutputHelper output) => _output = output;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    /// <summary>The operands of one chunk: q [heads][nq][d] F32, the head-first F16 caches (values they hold exactly).</summary>
    private sealed class Chunk : IDisposable
    {
        public readonly Tensor Qt, Kt, Vt, Out, Partial;
        public readonly int D, Heads, KvHeads, Nq, Len, Floor, Cap, Splits;

        public Chunk(CudaAllocator allocator, CudaFlashPrefillKernels flash, int d, int heads, int kvHeads, int nq, int len, int floor)
        {
            (D, Heads, KvHeads, Nq, Len, Floor, Cap) = (d, heads, kvHeads, nq, len, floor, len + 29);
            var rng = new Random(d * 31 + nq + len);
            var Q = new float[heads * nq * d];
            for (int i = 0; i < Q.Length; i++) Q[i] = (float)(rng.NextDouble() * 2 - 1);
            var K = new float[kvHeads * Cap * d];
            var V = new float[kvHeads * Cap * d];
            for (int i = 0; i < K.Length; i++)
            {
                K[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
                V[i] = (float)(System.Half)(rng.NextDouble() * 2 - 1);
            }
            Splits = flash.Splits(nq * (heads / kvHeads), kvHeads, len);
            Qt = Tensor.FromArray(allocator, Q);
            using var k32 = Tensor.FromArray(allocator, K);
            using var v32 = Tensor.FromArray(allocator, V);
            Kt = new Tensor(allocator, DType.Float16, K.Length);
            Vt = new Tensor(allocator, DType.Float16, V.Length);
            Out = new Tensor(allocator, DType.Float32, (long)nq * heads * d);
            Partial = Splits > 1 ? new Tensor(allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, d, Splits)) : null;
            foreach (Tensor t in new[] { Qt, k32, v32, Kt, Vt, Out }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            allocator.Kernels.LaunchConvertF32F16(P(k32), P(Kt), K.Length, allocator.Stream.Handle);
            allocator.Kernels.LaunchConvertF32F16(P(v32), P(Vt), V.Length, allocator.Stream.Handle);
            allocator.Stream.Synchronize();
        }

        public float[] Run(CudaAllocator allocator, CudaFlashPrefillKernels flash, bool prompt)
        {
            allocator.Context.MakeCurrent();
            flash.LaunchAttention(P(Qt), P(Kt), P(Vt), Cap, Nq, Len, Floor, Heads, KvHeads, D, 1f / MathF.Sqrt(D), Splits,
                Partial == null ? IntPtr.Zero : P(Partial), P(Out), allocator.Stream.Handle, prompt: prompt);
            allocator.Stream.Synchronize();
            ((CudaStorage)Out.Storage).MarkDeviceModified();
            return Out.GetElementsAsFloat(Nq * Heads * D);
        }

        public void Dispose()
        {
            foreach (Tensor t in new[] { Qt, Kt, Vt, Out, Partial }) t?.Dispose();
        }
    }

    [CudaTheory]
    [InlineData(256, 24, 4, 2048, 2048, 0)]   // Qwen3.8-27B's first 2,048-row chunk
    [InlineData(256, 24, 4, 1570, 3618, 0)]   // its second chunk over the first's keys
    [InlineData(128, 16, 2, 300, 700, 33)]    // 128-wide heads, keys below a floor masked
    [InlineData(256, 8, 1, 24, 4000, 0)]      // a few rows over a long window: split over the keys
    public void PromptSchedulingKeepsThePlainKernelsBits(int d, int heads, int kvHeads, int nq, int len, int floor)
    {
        using var allocator = new CudaAllocator();
        CudaFlashPrefillKernels flash = allocator.FlashPrefillKernels;
        if (flash == null || !flash.RunsAttention(d))
        {
            _output.WriteLine("no flash prefill attention for this GPU");
            return;
        }
        using var chunk = new Chunk(allocator, flash, d, heads, kvHeads, nq, len, floor);
        float[] plain = chunk.Run(allocator, flash, prompt: false);
        float[] scheduled = chunk.Run(allocator, flash, prompt: true);
        int differ = 0;
        for (int i = 0; i < plain.Length; i++)
            if (BitConverter.SingleToInt32Bits(plain[i]) != BitConverter.SingleToInt32Bits(scheduled[i])) differ++;
        _output.WriteLine($"d {d}, {heads}/{kvHeads} heads, {nq} rows over {len} keys, {chunk.Splits} splits: {differ} of {plain.Length} differ");
        Assert.Equal(0, differ);
    }
}
