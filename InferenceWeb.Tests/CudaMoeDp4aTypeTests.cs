// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The MoE dp4a kernels for the expert types of the Qwen3.8 Flash Next checkpoints (UD-Q2_K_XL: IQ2_XS, IQ3_XXS, IQ4_NL;
/// ISTA-DASLab's GSQ-RCO: Q2_0) against the exact per-element kernels (qvalue_at) on the same experts: the dp4a path
/// quantizes the activations to q8_1, so the two agree to a few 1e-2, for a decode row and a few verify rows.
/// </summary>
public sealed class CudaMoeDp4aTypeTests
{
    private readonly ITestOutputHelper _output;

    public CudaMoeDp4aTypeTests(ITestOutputHelper output) => _output = output;

    /// <summary>Bytes per block and values per block of each type.</summary>
    private static (int Bytes, int Values) Block(int type) => type switch
    {
        17 => (74, 256),   // IQ2_XS
        18 => (98, 256),   // IQ3_XXS
        20 => (18, 32),    // IQ4_NL
        42 => (18, 64),    // Q2_0
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    /// <summary>Random codes (every code is a valid encoding of these types) under small F16 block scales.</summary>
    private static byte[] RandomRows(int type, int rows, int cols, Random rng)
    {
        (int bytes, int values) = Block(type);
        int blocks = rows * (cols / values);
        var data = new byte[blocks * bytes];
        rng.NextBytes(data);
        for (int b = 0; b < blocks; b++)
            BitConverter.TryWriteBytes(data.AsSpan(b * bytes, 2), (System.Half)(0.004 + rng.NextDouble() * 0.01));
        return data;
    }

    [CudaTheory]
    [InlineData(17, 1)]
    [InlineData(17, 5)]
    [InlineData(18, 1)]
    [InlineData(18, 5)]
    [InlineData(20, 1)]
    [InlineData(20, 5)]
    [InlineData(42, 1)]
    [InlineData(42, 5)]
    public void Dp4aMatchesTheExactKernels(int type, int rows)
    {
        const int hidden = 512, ff = 256, experts = 4, k = 2;
        using var allocator = new CudaAllocator();
        if (allocator.Kernels?.MoeDp4aIqQ2_0 != true)
        {
            _output.WriteLine("the loaded kernels have no dp4a path for these types");
            return;
        }
        Assert.True(CudaFusedOps.IsMoeDp4aDimensionSupported(type, hidden) && CudaFusedOps.IsMoeDp4aDimensionSupported(type, ff));
        var rng = new Random(31 + type);
        var hosts = new IntPtr[3 * experts];
        var keys = new IntPtr[3 * experts];
        var tables = new IntPtr[3];
        try
        {
            var resident = new IntPtr[3 * experts];
            for (int m = 0; m < 3; m++)
                for (int e = 0; e < experts; e++)
                {
                    int i = m * experts + e, inDim = m < 2 ? hidden : ff, outDim = m < 2 ? ff : hidden;
                    byte[] w = RandomRows(type, outDim, inDim, rng);
                    hosts[i] = Marshal.AllocHGlobal(w.Length);
                    Marshal.Copy(w, 0, hosts[i], w.Length);
                    keys[i] = new IntPtr(0x7a0000 + 64 * type + i);
                    CudaQuantizedOps.PreloadQuantizedWeight(allocator, keys[i], hosts[i], type, inDim, outDim, w.Length);
                    Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, keys[i], out resident[i], out _));
                }
            for (int m = 0; m < 3; m++)
                tables[m] = CudaQuantizedOps.CreateDevicePointerTable(allocator, resident.AsSpan(m * experts, experts).ToArray());

            var input = new float[rows, hidden];
            var logits = new float[rows, experts];
            for (int r = 0; r < rows; r++)
            {
                for (int i = 0; i < hidden; i++) input[r, i] = (float)(rng.NextDouble() * 2 - 1);
                for (int e = 0; e < experts; e++) logits[r, e] = (float)(rng.NextDouble() * 2 - 1);
            }
            float[] Run(bool dp4a)
            {
                using var lt = Tensor.FromArray(allocator, logits);
                using var xt = Tensor.FromArray(allocator, input);
                using var output = new Tensor(allocator, DType.Float32, rows, hidden);
                using var selected = new Tensor(allocator, DType.Int32, rows * k);
                using var weights = new Tensor(allocator, DType.Float32, rows * k);
                using var gate = new Tensor(allocator, DType.Float32, rows * k, ff);
                using var up = new Tensor(allocator, DType.Float32, rows * k, ff);
                using var xq = new Tensor(allocator, DType.UInt8, (long)rows * (hidden / 32) * CudaFusedOps.Q81BlockBytes);
                using var hq = new Tensor(allocator, DType.UInt8, (long)rows * k * (ff / 32) * CudaFusedOps.Q81BlockBytes);
                bool ran = rows == 1
                    ? CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU(lt, xt, output, selected, weights, gate, up, IntPtr.Zero,
                        tables[0], tables[1], tables[2], type, type, experts, k, hidden, ff, null, IntPtr.Zero, xq, hq, dp4a, dp4a)
                    : CudaFusedOps.TryMoEExpertFFNPrefillSwiGLU(lt, xt, output, selected, weights, gate, up, IntPtr.Zero,
                        tables[0], tables[1], tables[2], type, type, experts, k, hidden, ff, rows, null, IntPtr.Zero, xq, hq, dp4a, dp4a);
                Assert.True(ran);
                return output.GetElementsAsFloat(rows * hidden);
            }
            float[] exact = Run(false), fast = Run(true);
            double num = 0, den = 0;
            for (int i = 0; i < exact.Length; i++)
            {
                num += (fast[i] - exact[i]) * (double)(fast[i] - exact[i]);
                den += exact[i] * (double)exact[i];
            }
            double rel = Math.Sqrt(num / den);
            _output.WriteLine($"type {type}, {rows} row(s): relative L2 {rel:E2}");
            // The dp4a path rounds the activations and the SwiGLU hidden to int8 blocks (as ggml and the engines' int8
            // expert kernels do): IQ2_XS / IQ3_XXS / IQ4_NL measured 1e-2 or less on an A40, Q2_0's decode row 2.5e-2.
            Assert.True(den > 0 && rel < 3e-2, $"type {type}: relative L2 {rel:E2}");
        }
        finally
        {
            allocator.Synchronize();
            for (int m = 0; m < 3; m++)
                if (tables[m] != IntPtr.Zero) CudaQuantizedOps.FreeDeviceBuffer(allocator, tables[m]);
            for (int i = 0; i < hosts.Length; i++)
                if (hosts[i] != IntPtr.Zero) Marshal.FreeHGlobal(hosts[i]);
        }
    }
}
