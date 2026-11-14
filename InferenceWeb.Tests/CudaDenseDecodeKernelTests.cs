// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Decode-size products (1 and 4 rows) of resident quantized weights at Qwen3.8 Flash Next's projection shapes: the
/// decode-once kernels (Dsv4Kernels.DenseQ81Rows, the direct engines' decode path, which the CUDA executor uses for
/// these rows) against the per-type q8_1 kernels the executor ran before, on the same quantized activations: the
/// same products to the sums' order, and the time of each (the executor's choice is the measured faster one).
/// </summary>
public sealed class CudaDenseDecodeKernelTests
{
    private readonly ITestOutputHelper _output;

    public CudaDenseDecodeKernelTests(ITestOutputHelper output) => _output = output;

    /// <summary>Bytes per block, values per block and the offset of the block's F16 scale (the others random codes).</summary>
    private static (int Bytes, int Values, int ScaleAt, int MinAt) Block(int type) => type switch
    {
        8 => (34, 32, 0, -1),      // Q8_0
        12 => (144, 256, 0, 2),    // Q4_K
        13 => (176, 256, 0, 2),    // Q5_K
        14 => (210, 256, 208, -1), // Q6_K
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    private static byte[] RandomRows(int type, int rows, int cols, Random rng)
    {
        (int bytes, int values, int scaleAt, int minAt) = Block(type);
        int blocks = rows * (cols / values);
        var data = new byte[(long)blocks * bytes];
        rng.NextBytes(data);
        for (long b = 0; b < blocks; b++)
        {
            BitConverter.TryWriteBytes(data.AsSpan((int)(b * bytes + scaleAt), 2), (System.Half)(0.001 + rng.NextDouble() * 0.002));
            if (minAt >= 0)
                BitConverter.TryWriteBytes(data.AsSpan((int)(b * bytes + minAt), 2), (System.Half)(rng.NextDouble() * 0.002));
        }
        return data;
    }

    [CudaTheory]
    [InlineData(13, 2560, 10240)]   // attn_qkv
    [InlineData(13, 2560, 6144)]    // attn_gate
    [InlineData(13, 2560, 12288)]   // attn_q
    [InlineData(13, 2560, 640)]     // shared expert gate / up
    [InlineData(13, 6144, 2560)]    // attn_output
    [InlineData(14, 6144, 2560)]    // ssm_out
    [InlineData(14, 2560, 512)]     // attn_k / attn_v
    [InlineData(8, 10240, 320)]     // hyper-connection down
    [InlineData(8, 320, 10240)]     // hyper-connection up
    [InlineData(8, 640, 2560)]      // shared expert down
    [InlineData(12, 2560, 4096)]
    public void DecodeOnceKernelsMatchTheQ81Kernels(int type, int inDim, int outDim)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Dsv4Kernels dk;
        try { dk = Dsv4Kernels.Create(); }
        catch (InvalidOperationException ex)
        {
            _output.WriteLine($"no dsv4 module: {ex.Message}");
            return;
        }
        using (dk)
        {
            Assert.True(dk.DenseFastReads(type, inDim), $"type {type} with {inDim} inputs is not a decode-once format");
            var rng = new Random(type * 7 + inDim + outDim);
            byte[] w = RandomRows(type, outDim, inDim, rng);
            IntPtr host = Marshal.AllocHGlobal(w.Length);
            try
            {
                Marshal.Copy(w, 0, host, w.Length);
                var key = new IntPtr(0x7c0000 + type * 4096 + outDim);
                CudaQuantizedOps.PreloadQuantizedWeight(allocator, key, host, type, inDim, outDim, w.Length);
                Assert.True(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, key, out IntPtr wPtr, out _));
                foreach (int rows in new[] { 1, 4 })
                {
                    var x = new float[rows * inDim];
                    for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
                    using var xt = Tensor.FromArray(allocator, x);
                    using var x2 = xt.View(rows, inDim);
                    using var xq = new Tensor(allocator, DType.UInt8, (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes);
                    using var fast = new Tensor(allocator, DType.Float32, rows, outDim);
                    using var before = new Tensor(allocator, DType.Float32, rows, outDim);
                    foreach (Tensor t in new[] { xt, xq, fast, before }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
                    IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);
                    allocator.Context.MakeCurrent();
                    CudaQuantizedOps.QuantizeRowsQ81(allocator, P(xt), P(xq), inDim, rows);
                    bool q81 = CudaQuantizedOps.ResidentMatmulTakesQ81(type, inDim, outDim, rows);
                    void Fast()
                    {
                        allocator.Context.MakeCurrent();
                        dk.DenseQ81Rows(wPtr, P(xq), P(fast), type, outDim, inDim, w.Length / outDim, rows, allocator.Stream.Handle);
                    }
                    void Before()
                    {
                        allocator.Context.MakeCurrent();
                        if (q81) Assert.True(CudaQuantizedOps.TryResidentMatmulQ81(allocator, wPtr, type, P(xq), P(before), inDim, outDim, rows));
                        else Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(before, x2, key, host, type, inDim, outDim, w.Length));
                    }
                    double Time(Action launch)
                    {
                        for (int i = 0; i < 20; i++) launch();
                        allocator.Synchronize();
                        var watch = Stopwatch.StartNew();
                        const int n = 300;
                        for (int i = 0; i < n; i++) launch();
                        allocator.Synchronize();
                        return watch.Elapsed.TotalMilliseconds * 1000 / n;
                    }
                    double fastUs = Time(Fast), beforeUs = Time(Before);
                    foreach (Tensor t in new[] { fast, before }) ((CudaStorage)t.Storage).MarkDeviceModified();
                    float[] a = fast.GetElementsAsFloat(rows * outDim), b = before.GetElementsAsFloat(rows * outDim);
                    double num = 0, den = 0;
                    for (int i = 0; i < a.Length; i++)
                    {
                        num += (a[i] - b[i]) * (double)(a[i] - b[i]);
                        den += b[i] * (double)b[i];
                    }
                    double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
                    _output.WriteLine($"type {type} {inDim} -> {outDim}, {rows} row(s): decode-once {fastUs:F1} us, before " +
                        $"{beforeUs:F1} us ({(q81 ? "q8_1 kernel" : "addmm")}); relative L2 {rel:E2}");
                    // The same quantized activations: the products differ by the sums' order (an addmm path that quantizes
                    // the activations itself, or dequantizes the weights, by the activations' rounding).
                    Assert.True(rel < (q81 ? 1e-4 : 2e-2), $"type {type} {inDim}->{outDim} rows {rows}: relative L2 {rel:E2}");
                }
            }
            finally
            {
                allocator.Synchronize();
                Marshal.FreeHGlobal(host);
            }
        }
    }
}
